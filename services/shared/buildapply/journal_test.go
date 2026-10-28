// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

func journalFixture(t *testing.T) (*Journal, *OperationLock) {
	t.Helper()
	j, err := openJournalAt(filepath.Join(packageFixtureTempDir(t), "operation-state"), true)
	if err != nil {
		t.Fatalf("create private journal: %v", err)
	}
	t.Cleanup(func() {
		if err := j.Close(); err != nil {
			t.Errorf("close journal: %v", err)
		}
	})
	lock, err := j.TryLock()
	if err != nil {
		t.Fatalf("acquire journal lock: %v", err)
	}
	t.Cleanup(func() {
		if lock.file != nil {
			if err := lock.Close(); err != nil {
				t.Errorf("release journal lock: %v", err)
			}
		}
	})
	return j, lock
}

func journalInputs(t *testing.T, j *Journal) (Receipt, Inventory) {
	t.Helper()
	id, err := NewOperationID()
	if err != nil {
		t.Fatalf("mint operation identity: %v", err)
	}
	digest := strings.Repeat("a", 64)
	// This is Package.inventory's length-delimited record grammar.
	record, err := json.Marshal(struct {
		Path       string
		Size       int64
		Mode       uint32
		Hash, Link string
	}{"nvpair", 5, 0o755, digest, ""})
	if err != nil {
		t.Fatalf("encode synthetic package record: %v", err)
	}
	hash := sha256.Sum256(append([]byte(fmt.Sprintf("%d:", len(record))), record...))
	inventory := Inventory{
		OperationID: id, Platform: "linux", Architecture: "amd64", PackageSHA256: hex.EncodeToString(hash[:]),
		Entries: []InventoryEntry{{Path: "nvpair", Size: 5, Mode: 0o755, SHA256: digest}},
	}
	receipt := Receipt{
		OperationID: id, RunnerToken: id, PreviewID: "preview_" + id, Target: filepath.Join(filepath.Dir(j.Path()), "target"),
		CandidateSHA256: inventory.PackageSHA256, PreviousSHA256: digest,
		Protected: ProtectedDigests{Policy: digest, EngineSettings: digest, Intent: digest, Locks: digest, IdentityMembership: digest},
		Expected:  JournalExpectations{StopEngines: true, UnloadModels: true, RemainPaused: true, PreserveIntentLocks: true, OnActive: nodepolicy.FinishActive, DrainTimeoutSeconds: 120},
		Consent:   Consent{AllowEngineStop: true, AllowInterruption: true},
		Fences: Fences{
			RuntimeGeneration: Known("runtime"), ServiceInstance: Known("service"), BrokerGeneration: Known("broker"), ManagerIncarnation: Known("manager"),
			PolicyRevision: Known("policy"), SettingsRevision: Known("settings"), IntentRevision: Known("intent"), LocksRevision: Known("locks"), AvailabilityRevision: Known("availability"),
		},
		Phase: PhasePrepare, Rename: RenameNone, Handoff: HandoffNone,
	}
	layout, _ := LayoutFor(runtime.GOOS)
	gui := RunnerReservation{PID: 101, Identity: ProcessIdentity{Executable: filepath.Join(receipt.Target, filepath.FromSlash(layout.Executable)), Created: "123"}}
	service := RunnerReservation{PID: 102, Identity: ProcessIdentity{Executable: filepath.Join(receipt.Target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable), Created: "456"}}
	identity := Identity{BuildID: Known("git:" + strings.Repeat("b", 40)), SourceRevision: Known(strings.Repeat("b", 40)), Dirty: Known(false), SHA256: Known(digest)}
	now := time.Now().UTC()
	endpoint := filepath.Join(filepath.Dir(receipt.Target), "service.sock")
	if runtime.GOOS == "windows" {
		endpoint = `\\.\pipe\pair-build-apply-injected`
	}
	receipt.Runtime = RuntimeAuthority{AcceptedAt: now, Deadline: now.Add(30 * time.Minute), CandidatePath: filepath.Join(filepath.Dir(receipt.Target), "candidate"), Endpoint: endpoint, OldGUI: gui, OldService: service, OldProcesses: []RunnerReservation{gui, service}, ServiceLaunch: LaunchContext{EnvironmentSHA256: digest}, GUILaunch: LaunchContext{EnvironmentSHA256: digest}, ExpectedService: identity, ExpectedGUI: identity, RunnerSHA256: digest, TargetIdentity: "123:456", CandidateMetadataSHA256: digest}
	return receipt, inventory
}

func beginJournal(t *testing.T, j *Journal, lock *OperationLock) Receipt {
	t.Helper()
	receipt, inventory := journalInputs(t, j)
	state, err := j.Begin(lock, receipt, inventory)
	if err != nil || state.Current == nil {
		t.Fatalf("begin operation: %+v, %v", state, err)
	}
	return *state.Current
}

func saveJournal(t *testing.T, j *Journal, lock *OperationLock, receipt Receipt) JournalState {
	t.Helper()
	state, err := j.Save(lock, receipt)
	if err != nil {
		t.Fatalf("save %s/%s/%s: %v", receipt.Phase, receipt.Rename, receipt.Handoff, err)
	}
	return state
}

func journalVerifiedSuccess(r Receipt) Receipt {
	// Synthetic trusted-owner observations for persistence tests, not native
	// process or policy evidence. Native verification belongs to the service.
	r.Verified = &BuildApplyOperation{OperationID: r.OperationID, PreviewID: r.PreviewID, Phase: r.Phase, Outcome: r.Outcome, OwnershipAcknowledged: true,
		ReadbackConfidence: ConfidenceVerified, Fences: r.Fences, Saved: Known(true), Enforced: Known(true), PhysicalStop: Known(true), RemainsPaused: Known(true),
		RunningService: r.Runtime.ExpectedService, RunningGUI: r.Runtime.ExpectedGUI}
	return r
}

func TestOperationLockExcludesContendersAndReleases(t *testing.T) {
	j, owner := journalFixture(t)
	contender, err := j.TryLock()
	if err == nil {
		if err := contender.Close(); err != nil {
			t.Errorf("close unexpected contender: %v", err)
		}
		t.Fatal("contender acquired held operation lock")
	}
	if err := owner.Close(); err != nil {
		t.Fatalf("release operation lock: %v", err)
	}
	next, err := j.TryLock()
	if err != nil {
		t.Fatalf("acquire released operation lock: %v", err)
	}
	if err := next.Close(); err != nil {
		t.Fatalf("release next operation lock: %v", err)
	}
}

func TestJournalPhaseAndRenameIntentSurviveReopen(t *testing.T) {
	j, lock := journalFixture(t)
	receipt := beginJournal(t, j, lock)
	receipt.Phase = PhaseQuiesce
	saveJournal(t, j, lock, receipt)
	receipt.Pause = &JournalPauseFence{Revision: "paused", Origin: nodepolicy.PausedLocally, Owned: true, PolicySHA256: strings.Repeat("b", 64)}
	saveJournal(t, j, lock, receipt)
	receipt.Handoff = HandoffIntent
	saveJournal(t, j, lock, receipt)
	receipt.Handoff = HandoffReserved
	// Synthetic evidence only. No process is inspected, started or contacted.
	receipt.Runner = &RunnerReservation{PID: 23, Identity: ProcessIdentity{Executable: j.runnerPath(), Created: "123:456"}}
	saveJournal(t, j, lock, receipt)
	receipt.Handoff = HandoffAcknowledged
	receipt.RunnerReady = true
	saveJournal(t, j, lock, receipt)
	for _, phase := range []Phase{PhaseShutdown, PhaseSwap} {
		receipt.Phase = phase
		saveJournal(t, j, lock, receipt)
	}
	for _, boundary := range []RenameState{RenameTargetIntent, RenamePreviousRetained, RenamePreparedIntent, RenameInstalled} {
		t.Run(string(boundary), func(t *testing.T) {
			receipt.Rename = boundary
			saveJournal(t, j, lock, receipt)
			reader, err := openJournalAt(j.Path(), false)
			if err != nil {
				t.Fatalf("reopen journal: %v", err)
			}
			state, readErr := reader.Read()
			closeErr := reader.Close()
			if readErr != nil || closeErr != nil || state.Current == nil || !reflect.DeepEqual(*state.Current, receipt) {
				t.Fatalf("durable boundary = %+v, read/close errors = %v / %v", state, readErr, closeErr)
			}
		})
	}
	for _, phase := range []Phase{PhaseRestart, PhaseVerifyBuild, PhaseVerifySettings, PhaseFinish} {
		receipt.Phase = phase
		saveJournal(t, j, lock, receipt)
	}
	outcome := OutcomeSucceeded
	receipt.Outcome = &outcome
	receipt = journalVerifiedSuccess(receipt)
	state := saveJournal(t, j, lock, receipt)
	if state.Current != nil || state.LastTerminal == nil || state.LastTerminal.PreviewID != receipt.PreviewID {
		t.Fatalf("terminal publication = %+v", state)
	}
}

func TestJournalBoundsCurrentAndLastTerminal(t *testing.T) {
	j, lock := journalFixture(t)
	first := beginJournal(t, j, lock)
	next, inventory := journalInputs(t, j)
	busy, err := j.Begin(lock, next, inventory)
	if !errors.Is(err, ErrOperationBusy) || busy.Current == nil || busy.Current.OperationID != first.OperationID {
		t.Fatalf("second admission = %+v, %v", busy, err)
	}
	refused := OutcomeRefused
	first.Outcome = &refused
	saveJournal(t, j, lock, first)
	state, err := j.Begin(lock, next, inventory)
	if err != nil || state.Current == nil || state.LastTerminal == nil {
		t.Fatalf("admit after terminal = %+v, %v", state, err)
	}
	if state.LastTerminal.OperationID != first.OperationID || state.Current.OperationID != next.OperationID {
		t.Fatalf("bounded receipt slots = %+v", state)
	}
	final := *state.Current
	final.Outcome = &refused
	state = saveJournal(t, j, lock, final)
	if state.Current != nil || state.LastTerminal == nil || state.LastTerminal.OperationID != next.OperationID {
		t.Fatalf("terminal replacement = %+v", state)
	}
}

func TestJournalRejectsBoundaryAndAuthorityChanges(t *testing.T) {
	test := func(name string, change func(*Receipt)) {
		t.Run(name, func(t *testing.T) {
			j, lock := journalFixture(t)
			receipt := beginJournal(t, j, lock)
			before := receipt
			change(&receipt)
			if _, err := j.Save(lock, receipt); err == nil {
				t.Fatal("invalid receipt update accepted")
			}
			state, err := j.Read()
			if err != nil || state.Current == nil || !reflect.DeepEqual(*state.Current, before) {
				t.Fatalf("refused update changed authority: %+v, %v", state, err)
			}
		})
	}
	test("skips phase", func(r *Receipt) { r.Phase = PhaseSwap })
	test("changes target", func(r *Receipt) { r.Target += "-other" })
	test("changes preview", func(r *Receipt) { r.PreviewID = "another_preview" })
	test("changes protected digest", func(r *Receipt) { r.Protected.Intent = strings.Repeat("b", 64) })
	test("extends accepted deadline", func(r *Receipt) {
		r.Runtime.AcceptedAt = r.Runtime.AcceptedAt.Add(time.Second)
		r.Runtime.Deadline = r.Runtime.Deadline.Add(time.Second)
	})
	test("changes launch arguments", func(r *Receipt) { r.Runtime.ServiceLaunch.Args = []string{"--log-level=debug"} })
	test("changes launch environment", func(r *Receipt) { r.Runtime.ServiceLaunch.EnvironmentSHA256 = strings.Repeat("b", 64) })
	test("changes expected build", func(r *Receipt) { r.Runtime.ExpectedService.BuildID = Known("git:" + strings.Repeat("c", 40)) })
	test("changes copied runner", func(r *Receipt) { r.Runtime.RunnerSHA256 = strings.Repeat("b", 64) })
	test("changes launch token", func(r *Receipt) { r.RunnerToken = strings.Repeat("b", 32) })
	test("disables physical stop", func(r *Receipt) { r.Expected.StopEngines = false })
	test("invents phase", func(r *Receipt) { r.Phase = "custom" })
	test("renames before shutdown", func(r *Receipt) { r.Rename = RenameTargetIntent })
	test("handoff before quiesce", func(r *Receipt) { r.Handoff = HandoffIntent })
	test("premature success", func(r *Receipt) { value := OutcomeSucceeded; r.Outcome = &value })
}

func TestJournalRequiresExplicitRecoveryAfterFailedPhase(t *testing.T) {
	j, lock := journalFixture(t)
	receipt := beginJournal(t, j, lock)
	failed := OutcomeFailed
	receipt.Outcome = &failed
	state := saveJournal(t, j, lock, receipt)
	if state.Current == nil || state.LastTerminal != nil {
		t.Fatalf("failure released pending gate: %+v", state)
	}
	receipt.Outcome = nil
	if _, err := j.Save(lock, receipt); !errors.Is(err, ErrRecoveryNeeded) {
		t.Fatalf("implicit retry error = %v", err)
	}
	if _, err := j.Recover(lock, strings.Repeat("b", 32)); err == nil {
		t.Fatal("recovery accepted another operation identity")
	}
	if _, err := j.Recover(lock, receipt.OperationID); err != nil {
		t.Fatalf("explicit recovery: %v", err)
	}
	saveJournal(t, j, lock, receipt)
}

func TestJournalAdoptsLandedReceiptButRetainsUncertainty(t *testing.T) {
	j, lock := journalFixture(t)
	receipt := beginJournal(t, j, lock)
	original := j.replacer.Replace
	injected := errors.New("synthetic post-publication failure")
	j.replacer.Replace = func(tmp, target string) error {
		if err := original(tmp, target); err != nil {
			return err
		}
		return errors.Join(durablefile.ErrReplaced, injected)
	}
	receipt.Phase = PhaseQuiesce
	state, err := j.Save(lock, receipt)
	if !errors.Is(err, durablefile.ErrReplaced) || !errors.Is(err, injected) || state.Current == nil || state.Current.Phase != PhaseQuiesce {
		t.Fatalf("landed uncertain receipt = %+v, %v", state, err)
	}
	if _, err := j.Save(lock, receipt); !errors.Is(err, ErrRecoveryNeeded) {
		t.Fatalf("uncertain continuation error = %v", err)
	}
	j.replacer.Replace = original
	if _, err := j.Recover(lock, receipt.OperationID); err != nil {
		t.Fatalf("recover landed receipt: %v", err)
	}
	saveJournal(t, j, lock, receipt)
}

func TestJournalPrepublicationFailurePreservesPreviousReceipt(t *testing.T) {
	j, lock := journalFixture(t)
	receipt := beginJournal(t, j, lock)
	injected := errors.New("synthetic rename refusal")
	j.replacer.Replace = func(string, string) error { return injected }
	receipt.Phase = PhaseQuiesce
	state, err := j.Save(lock, receipt)
	if !errors.Is(err, injected) || errors.Is(err, durablefile.ErrReplaced) || state.Current == nil || state.Current.Phase != PhasePrepare {
		t.Fatalf("failed publication = %+v, %v", state, err)
	}
}

func TestJournalDistinguishesAbsentFromInvalidState(t *testing.T) {
	parent := packageFixtureTempDir(t)
	if _, err := openJournalAt(filepath.Join(parent, "absent"), false); !os.IsNotExist(err) {
		t.Fatalf("absent read-only open error = %v", err)
	}
	j, _ := journalFixture(t)
	state, err := j.Read()
	if err != nil || state.Current != nil || state.LastTerminal != nil {
		t.Fatalf("empty private root = %+v, %v", state, err)
	}
	if err := j.writeFile("operation.json", []byte(`{"current":null,"lastTerminal":null}`)); err != nil {
		t.Fatalf("write empty receipt fixture: %v", err)
	}
	if _, err := j.Read(); !errors.Is(err, ErrInvalidJournal) {
		t.Fatalf("recorded empty receipt error = %v", err)
	}
}

func TestJournalRejectsMalformedOrUnknownJSON(t *testing.T) {
	for _, fixture := range []struct{ name, json string }{
		{"malformed", `{"current":`},
		{"unknown field", `{"current":null,"lastTerminal":null,"privateBackup":{}}`},
		{"duplicate field", `{"current":null,"current":null,"lastTerminal":null}`},
		{"case alias duplicate", `{"current":null,"Current":null,"lastTerminal":null}`},
		{"trailing object", `{"current":null,"lastTerminal":null} {}`},
		{"null root", `null`},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			j, _ := journalFixture(t)
			if err := j.writeFile("operation.json", []byte(fixture.json)); err != nil {
				t.Fatalf("write malformed fixture: %v", err)
			}
			if _, err := j.Read(); !errors.Is(err, ErrInvalidJournal) {
				t.Fatalf("invalid journal error = %v", err)
			}
		})
	}
}

func TestJournalRejectsOversizedFilesBeforeDecode(t *testing.T) {
	for _, fixture := range []struct {
		name  string
		limit int64
	}{{"operation.json", MaxReceiptBytes}, {"inventory.json", MaxInventoryBytes}} {
		t.Run(fixture.name, func(t *testing.T) {
			j, lock := journalFixture(t)
			beginJournal(t, j, lock)
			f, err := j.root.root.OpenFile(fixture.name, os.O_WRONLY, 0)
			if err != nil {
				t.Fatalf("open bounded fixture: %v", err)
			}
			truncateErr := f.Truncate(fixture.limit + 1)
			closeErr := f.Close()
			if truncateErr != nil || closeErr != nil {
				t.Fatalf("size/close fixture: %v / %v", truncateErr, closeErr)
			}
			if _, err := j.Read(); !errors.Is(err, ErrInvalidJournal) {
				t.Fatalf("oversized status error = %v", err)
			}
		})
	}
}

func TestJournalBindsInventoryToExactBytesAndOperation(t *testing.T) {
	j, lock := journalFixture(t)
	receipt := beginJournal(t, j, lock)
	inventory, err := j.ReadInventory(receipt)
	if err != nil || inventory.OperationID != receipt.OperationID || inventory.PackageSHA256 != receipt.CandidateSHA256 {
		t.Fatalf("read inventory = %+v, %v", inventory, err)
	}
	if err := j.writeFile("inventory.json", []byte(`{"entries":[]}`)); err != nil {
		t.Fatalf("replace synthetic inventory: %v", err)
	}
	if _, err := j.Read(); !errors.Is(err, ErrInvalidJournal) {
		t.Fatalf("altered inventory status error = %v", err)
	}
}

func TestJournalRejectsInventoryTraversalAndMalformedEntries(t *testing.T) {
	test := func(name string, change func(*Inventory)) {
		t.Run(name, func(t *testing.T) {
			j, lock := journalFixture(t)
			receipt, inventory := journalInputs(t, j)
			change(&inventory)
			if _, err := j.Begin(lock, receipt, inventory); err == nil {
				t.Fatal("invalid inventory admitted")
			}
			entries, err := os.ReadDir(j.Path())
			if err != nil || len(entries) != 1 || entries[0].Name() != "operation.lock" {
				t.Fatalf("invalid admission wrote state: %+v, %v", entries, err)
			}
		})
	}
	test("parent traversal", func(v *Inventory) { v.Entries[0].Path = "../other" })
	test("absolute path", func(v *Inventory) { v.Entries[0].Path = "/other" })
	test("missing directory", func(v *Inventory) { v.Entries[0].Path = "absent/file" })
	test("stream path", func(v *Inventory) { v.Entries[0].Path = "file:stream" })
	test("duplicate path", func(v *Inventory) { v.Entries = append(v.Entries, v.Entries[0]) })
	test("oversized entry", func(v *Inventory) { v.Entries[0].Size = MaxFileBytes + 1 })
	test("negative size", func(v *Inventory) { v.Entries[0].Size = -1 })
	test("privilege mode", func(v *Inventory) { v.Entries[0].Mode |= os.ModeSetuid })
	test("unapproved link", func(v *Inventory) { v.Entries[0].Mode = os.ModeSymlink; v.Entries[0].Link = "../other" })
	test("wrong package digest", func(v *Inventory) { v.PackageSHA256 = strings.Repeat("b", 64) })
	test("unknown identity", func(v *Inventory) { v.OperationID = "../other" })
}

func TestJournalDerivesExactSiblingsFromOpaqueIdentity(t *testing.T) {
	target := filepath.Join(packageFixtureTempDir(t), "NVIDIA PAIR.app")
	id, err := NewOperationID()
	if err != nil || !ValidOperationID(id) {
		t.Fatalf("random operation identity = %q, %v", id, err)
	}
	p, err := DeriveOperationPaths(target, id)
	if err != nil || p.Target != target || p.Prepared != target+"."+id+".prepared" || p.Previous != target+"."+id+".previous" {
		t.Fatalf("derived operation siblings = %+v, %v", p, err)
	}
	for _, bad := range []string{"", "../other", strings.Repeat("a", 31), strings.Repeat("a", 33), strings.Repeat("A", 32), strings.Repeat("g", 32), id + ".prepared"} {
		if _, err := DeriveOperationPaths(target, bad); err == nil {
			t.Errorf("accepted operation identity %q", bad)
		}
	}
	unclean := filepath.Join(filepath.Dir(target), "child") + string(filepath.Separator) + ".." + string(filepath.Separator) + "target"
	if _, err := DeriveOperationPaths(unclean, id); err == nil {
		t.Fatal("accepted unclean target spelling")
	}
}

func TestJournalRejectsUnheldOrDifferentRootLock(t *testing.T) {
	j, lock := journalFixture(t)
	_, otherLock := journalFixture(t)
	receipt, inventory := journalInputs(t, j)
	if _, err := j.Begin(nil, receipt, inventory); err == nil {
		t.Fatal("mutation accepted without a lock")
	}
	if _, err := j.Begin(otherLock, receipt, inventory); err == nil {
		t.Fatal("mutation accepted another root's lock")
	}
	if err := lock.Close(); err != nil {
		t.Fatalf("close lock: %v", err)
	}
	if _, err := j.Begin(lock, receipt, inventory); err == nil {
		t.Fatal("mutation accepted closed lock")
	}
}

func TestJournalCreatedStateIsPrivate(t *testing.T) {
	j, lock := journalFixture(t)
	beginJournal(t, j, lock)
	if err := validatePrivateRoot(j.root); err != nil {
		t.Fatalf("created root privacy: %v", err)
	}
	for _, name := range []string{"operation.lock", "operation.json", "inventory.json"} {
		f, err := j.root.root.Open(name)
		if err != nil {
			t.Fatalf("open private %s: %v", name, err)
		}
		info, statErr := f.Stat()
		if statErr == nil {
			statErr = nativePrivateCheck(f, info)
		}
		closeErr := f.Close()
		if statErr != nil || closeErr != nil {
			t.Fatalf("validate/close private %s: %v / %v", name, statErr, closeErr)
		}
	}
	runner, err := j.CreateRunner(lock)
	if err != nil {
		t.Fatalf("create private empty runner: %v", err)
	}
	if err := runner.Close(); err != nil {
		t.Fatalf("close private runner: %v", err)
	}
	if second, err := j.CreateRunner(lock); err == nil {
		if err := second.Close(); err != nil {
			t.Errorf("close unexpectedly overwritten runner: %v", err)
		}
		t.Fatal("existing runner overwritten")
	}
}

func TestJournalRejectsForeignReadPermissionsOnUnix(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix mode boundary; Windows ACL cases are in journal_windows_test.go")
	}
	j, lock := journalFixture(t)
	beginJournal(t, j, lock)
	name := filepath.Join(j.Path(), "operation.json")
	if err := os.Chmod(name, 0o644); err != nil {
		t.Fatalf("make synthetic receipt world-readable: %v", err)
	}
	if _, err := j.Read(); !errors.Is(err, ErrInvalidJournal) {
		t.Fatalf("world-readable receipt error = %v", err)
	}
	info, err := os.Stat(name)
	if err != nil || info.Mode().Perm() != 0o644 {
		t.Fatalf("reader mutated existing permissions: %v / %v", info, err)
	}
}

func TestJournalRejectsLinkedReceipt(t *testing.T) {
	j, lock := journalFixture(t)
	beginJournal(t, j, lock)
	if err := j.root.root.Rename("operation.json", "original.json"); err != nil {
		t.Fatalf("retain original synthetic receipt: %v", err)
	}
	if err := j.root.root.Link("original.json", "operation.json"); err != nil {
		t.Fatalf("hardlink synthetic receipt: %v", err)
	}
	if _, err := j.Read(); !errors.Is(err, ErrInvalidJournal) {
		t.Fatalf("hardlinked receipt error = %v", err)
	}
}

func TestJournalCollectionAndPublicationBounds(t *testing.T) {
	data := []byte(`{"entries":[` + strings.Repeat(`null,`, MaxEntries) + `null]}`)
	var inventory Inventory
	if err := strictJournalJSON(data, &inventory); err == nil || inventory.Entries != nil {
		t.Fatalf("oversized collection decoded before rejection: entries=%d, error=%v", len(inventory.Entries), err)
	}
	if _, err := boundedJSON(strings.Repeat("a", MaxReceiptBytes), MaxReceiptBytes); err == nil {
		t.Fatal("oversized publication accepted")
	}
}

func TestJournalRejectsSubstitutedRoot(t *testing.T) {
	j, lock := journalFixture(t)
	receipt := beginJournal(t, j, lock)
	if err := os.Rename(j.Path(), j.Path()+"-retained"); err != nil {
		if runtime.GOOS == "windows" && errors.Is(err, syscall.Errno(32)) {
			// A held Windows directory can prevent the substitution itself.
			// Require its exact identity and journal to remain unchanged rather
			// than treating that native protection as a successful substitution.
			info, statErr := os.Stat(j.Path())
			if statErr != nil || !os.SameFile(j.root.info, info) {
				t.Fatalf("refused rename changed root: %v", statErr)
			}
			if _, statErr := os.Stat(j.Path() + "-retained"); !os.IsNotExist(statErr) {
				t.Fatalf("refused rename published a sibling: %v", statErr)
			}
			state, readErr := j.Read()
			if readErr != nil || !reflect.DeepEqual(state.Current, &receipt) {
				t.Fatalf("refused rename changed receipt: state=%+v err=%v", state, readErr)
			}
			t.Log("held Windows directory lifetime prevented root substitution")
			return
		}
		t.Fatalf("move synthetic root: %v", err)
	}
	replacement, err := openJournalAt(j.Path(), true)
	if err != nil {
		t.Fatalf("create substitute root: %v", err)
	}
	defer func() {
		if err := replacement.Close(); err != nil {
			t.Errorf("close substitute root: %v", err)
		}
	}()
	if _, err := j.Read(); !errors.Is(err, ErrInvalidJournal) {
		t.Fatalf("substituted root read error = %v", err)
	}
	receipt.Phase = PhaseQuiesce
	if _, err := j.Save(lock, receipt); err == nil {
		t.Fatal("receipt published through substituted root")
	}
	if _, err := replacement.root.root.Lstat("operation.json"); !os.IsNotExist(err) {
		t.Fatalf("substitute root received a receipt: %v", err)
	}
}

func TestJournalRecoveryPermissionEndsAtNextFailure(t *testing.T) {
	j, lock := journalFixture(t)
	receipt := beginJournal(t, j, lock)
	failed := OutcomeFailed
	receipt.Outcome = &failed
	saveJournal(t, j, lock, receipt)
	if _, err := j.Recover(lock, receipt.OperationID); err != nil {
		t.Fatalf("recover synthetic failed phase: %v", err)
	}
	receipt.Outcome = nil
	saveJournal(t, j, lock, receipt)
	receipt.Outcome = &failed
	saveJournal(t, j, lock, receipt)
	receipt.Outcome = nil
	if _, err := j.Save(lock, receipt); !errors.Is(err, ErrRecoveryNeeded) {
		t.Fatalf("later failure reused prior recovery permission: %v", err)
	}
}

func TestJournalRecoversUncertainTerminalPublication(t *testing.T) {
	j, lock := journalFixture(t)
	receipt := beginJournal(t, j, lock)
	original := j.replacer.Replace
	injected := errors.New("synthetic terminal sync failure")
	j.replacer.Replace = func(tmp, target string) error {
		if err := original(tmp, target); err != nil {
			return err
		}
		if filepath.Base(target) == "operation.json" {
			return errors.Join(durablefile.ErrReplaced, injected)
		}
		return nil
	}
	refused := OutcomeRefused
	receipt.Outcome = &refused
	state, err := j.Save(lock, receipt)
	if !errors.Is(err, durablefile.ErrReplaced) || !state.TerminalPending || state.Current == nil || state.Current.Outcome == nil || *state.Current.Outcome != OutcomeUncertain || state.LastTerminal == nil || state.LastTerminal.OperationID != receipt.OperationID {
		t.Fatalf("uncertain terminal publication = %+v, %v", state, err)
	}
	reader, readErr := openJournalAt(j.Path(), false)
	if readErr != nil {
		t.Fatal(readErr)
	}
	fresh, readErr := reader.Read()
	closeErr := reader.Close()
	if readErr != nil || closeErr != nil || !fresh.TerminalPending || fresh.Current == nil {
		t.Fatalf("reopened uncertainty lost gate: %+v, %v / %v", fresh, readErr, closeErr)
	}
	j.replacer.Replace = original
	state, err = j.Recover(lock, receipt.OperationID)
	if err != nil || state.Current != nil || state.LastTerminal == nil || state.LastTerminal.OperationID != receipt.OperationID {
		t.Fatalf("terminal durability recovery = %+v, %v", state, err)
	}
	next, inventory := journalInputs(t, j)
	if _, err := j.Begin(lock, next, inventory); err != nil {
		t.Fatalf("new admission after terminal durability recovery: %v", err)
	}
}

func TestJournalPublicProjectionKeepsDurablePreviewCorrelation(t *testing.T) {
	j, lock := journalFixture(t)
	receipt := beginJournal(t, j, lock)
	operation := BuildApplyOperation{
		OperationID: receipt.OperationID, PreviewID: receipt.PreviewID, Phase: receipt.Phase,
		ReadbackConfidence: ConfidenceUnknown, Fences: receipt.Fences,
	}
	if err := receipt.ValidateOperation(operation); err != nil {
		t.Fatalf("validate receipt projection: %v", err)
	}
	status := BuildApplyStatus{State: StatusPending, Operation: &operation}
	encoded, err := EncodeStatusResult(Result[BuildApplyStatus]{Available: true, Value: &status})
	if err != nil {
		t.Fatalf("encode receipt projection: %v", err)
	}
	decoded, err := DecodeStatusResult(encoded, receipt.PreviewID)
	if err != nil || decoded.Value == nil || decoded.Value.Operation == nil || decoded.Value.Operation.PreviewID != receipt.PreviewID {
		t.Fatalf("round-trip durable preview correlation = %+v, %v", decoded, err)
	}
	operation.PreviewID = "different_preview"
	if err := receipt.ValidateOperation(operation); err == nil {
		t.Fatal("public operation accepted a different preview correlation")
	}
	if _, err := DecodeStatusResult(encoded, "different_preview"); err == nil {
		t.Fatal("lost-start-reply projection matched a different preview")
	}
}

func TestJournalAdmissionRequiresConsentAndCompleteOwnerFacts(t *testing.T) {
	test := func(name string, change func(*Receipt)) {
		t.Run(name, func(t *testing.T) {
			j, lock := journalFixture(t)
			receipt, inventory := journalInputs(t, j)
			change(&receipt)
			if _, err := j.Begin(lock, receipt, inventory); err == nil {
				t.Fatal("incomplete/unapproved admission accepted")
			}
			for _, name := range []string{"operation.json", "inventory.json", "runner"} {
				if _, err := j.root.root.Lstat(name); !os.IsNotExist(err) {
					t.Errorf("refused admission wrote %s: %v", name, err)
				}
			}
		})
	}
	test("no engine-stop consent", func(r *Receipt) { r.Consent.AllowEngineStop = false })
	test("unknown manager lifetime", func(r *Receipt) { r.Fences.ManagerIncarnation = Observation[string]{} })
	test("unknown intent revision", func(r *Receipt) { r.Fences.IntentRevision = Observation[string]{} })
	test("unknown locks digest", func(r *Receipt) { r.Protected.Locks = "" })
	test("invalid owner-fence UTF-8", func(r *Receipt) { r.Fences.IntentRevision = Known(string([]byte{0xff})) })
	test("invalid total budget", func(r *Receipt) { r.Runtime.Deadline = r.Runtime.Deadline.Add(time.Second) })
	test("partial recovery budget", func(r *Receipt) { r.RecoveryDeadline = time.Now().Add(time.Minute) })
	test("private environment backup", func(r *Receipt) { r.Runtime.ServiceLaunch.Environment = []string{"API_TOKEN=private"} })
	test("unrelated old image", func(r *Receipt) { r.Runtime.OldGUI.Identity.Executable += "-unrelated" })
	test("nonlocal endpoint", func(r *Receipt) { r.Runtime.Endpoint = `\\other-host\pipe\service` })
}

func journalRestartReceipt(t *testing.T, j *Journal, lock *OperationLock) Receipt {
	t.Helper()
	r := beginJournal(t, j, lock)
	r.PreparedIntent, r.PreparedIdentity = true, "123:789"
	saveJournal(t, j, lock, r)
	r.Phase = PhaseQuiesce
	r.Pause = &JournalPauseFence{Revision: "paused", Origin: nodepolicy.PausedLocally, Owned: true, PolicySHA256: strings.Repeat("b", 64)}
	saveJournal(t, j, lock, r)
	r.Handoff = HandoffIntent
	saveJournal(t, j, lock, r)
	r.Handoff = HandoffReserved
	r.Runner = &RunnerReservation{PID: 23, Identity: ProcessIdentity{Executable: j.runnerPath(), Created: "123:456"}}
	saveJournal(t, j, lock, r)
	r.Handoff, r.RunnerReady = HandoffAcknowledged, true
	saveJournal(t, j, lock, r)
	for _, phase := range []Phase{PhaseShutdown, PhaseSwap} {
		r.Phase = phase
		saveJournal(t, j, lock, r)
	}
	for _, rename := range []RenameState{RenameTargetIntent, RenamePreviousRetained, RenamePreparedIntent, RenameInstalled} {
		r.Rename = rename
		saveJournal(t, j, lock, r)
	}
	r.Phase = PhaseRestart
	saveJournal(t, j, lock, r)
	return r
}

func TestJournalLaunchReservationCannotChangeWithoutRecovery(t *testing.T) {
	j, lock := journalFixture(t)
	r := journalRestartReceipt(t, j, lock)
	r.ServiceLaunch = LaunchReservation{Intent: true, Token: strings.Repeat("a", 32)}
	saveJournal(t, j, lock, r)
	r.ServiceLaunch.Token = strings.Repeat("b", 32)
	if _, err := j.Save(lock, r); err == nil {
		t.Fatal("a retry replaced the pending launch correlation")
	}
	r.ServiceLaunch.Token = strings.Repeat("a", 32)
	r.ServiceLaunch.Process = &RunnerReservation{PID: 103, Identity: ProcessIdentity{Executable: r.Runtime.OldService.Identity.Executable, Created: "789"}}
	saveJournal(t, j, lock, r)
	r.ServiceLaunch.Acknowledged = true
	saveJournal(t, j, lock, r)
	r.ServiceLaunch.Process = &RunnerReservation{PID: 104, Identity: ProcessIdentity{Executable: r.Runtime.OldService.Identity.Executable, Created: "abc"}}
	if _, err := j.Save(lock, r); err == nil {
		t.Fatal("a retry replaced an acknowledged process lifetime")
	}
}

func TestJournalSucceededPublicationUncertaintyRequiresLiveReverification(t *testing.T) {
	j, lock := journalFixture(t)
	r := journalRestartReceipt(t, j, lock)
	for _, phase := range []Phase{PhaseVerifyBuild, PhaseVerifySettings, PhaseFinish} {
		r.Phase = phase
		saveJournal(t, j, lock, r)
	}
	outcome := OutcomeSucceeded
	r.Outcome, r.CleanupPending, r.ReleasePending = &outcome, true, true
	r = journalVerifiedSuccess(r)
	original := j.replacer.Replace
	j.replacer.Replace = func(tmp, target string) error {
		if err := original(tmp, target); err != nil {
			return err
		}
		if filepath.Base(target) == "operation.json" {
			return durablefile.ErrReplaced
		}
		return nil
	}
	if _, err := j.Save(lock, r); !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("terminal uncertainty: %v", err)
	}
	reader, err := openJournalAt(j.Path(), false)
	if err != nil {
		t.Fatal(err)
	}
	state, readErr := reader.Read()
	closeErr := reader.Close()
	if readErr != nil || closeErr != nil || !state.TerminalPending || state.Current == nil || state.Current.Outcome == nil || *state.Current.Outcome != OutcomeUncertain || state.LastTerminal == nil {
		t.Fatalf("fresh reader lost unresolved success: %+v, %v / %v", state, readErr, closeErr)
	}
	j.replacer.Replace = original
	state, err = j.Recover(lock, r.OperationID)
	if err != nil || state.TerminalPending || state.Current == nil || state.LastTerminal != nil || state.Current.Verified != nil || state.Current.Outcome == nil || *state.Current.Outcome != OutcomeUncertain {
		t.Fatalf("recovery trusted stale terminal evidence: %+v, %v", state, err)
	}
	_, inventory := journalInputs(t, j)
	if _, err := j.Begin(lock, r, inventory); !errors.Is(err, ErrOperationBusy) {
		t.Fatalf("unverified recovery released startup admission: %v", err)
	}
}

func TestJournalOrphanRunnerIsUnavailable(t *testing.T) {
	j, lock := journalFixture(t)
	f, err := j.CreateRunner(lock)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := j.Read(); !errors.Is(err, ErrInvalidJournal) {
		t.Fatalf("orphan copied executable became idle: %v", err)
	}
}

func TestJournalRemovesExecutableRunnerForNextOperation(t *testing.T) {
	j, lock := journalFixture(t)
	f, err := j.CreateRunner(lock)
	if err != nil {
		t.Fatal(err)
	}
	data := []byte("synthetic executable bytes")
	if _, err := f.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := f.Chmod(0o700); err != nil {
		t.Fatal(err)
	}
	if err := errors.Join(f.Sync(), f.Close()); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	if err := j.RemoveRunner(lock, hex.EncodeToString(digest[:])); err != nil {
		t.Fatalf("remove verified executable runner: %v", err)
	}
	if _, err := os.Stat(j.runnerPath()); !os.IsNotExist(err) {
		t.Fatalf("runner still present: %v", err)
	}
	next, err := j.CreateRunner(lock)
	if err != nil {
		t.Fatalf("next operation cannot reuse runner slot: %v", err)
	}
	if err := next.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestJournalRemovesUnfinishedRunnerCopies(t *testing.T) {
	for _, data := range []string{"partial", "complete synthetic executable bytes"} {
		t.Run(data, func(t *testing.T) {
			j, lock := journalFixture(t)
			f, err := j.CreateRunner(lock)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := f.Write([]byte(data)); err != nil {
				t.Fatal(err)
			}
			if err := errors.Join(f.Sync(), f.Close()); err != nil {
				t.Fatal(err)
			}
			digest := sha256.Sum256([]byte(data))
			if err := j.RemoveRunner(lock, hex.EncodeToString(digest[:])); err != nil {
				t.Fatalf("remove private unfinished copy: %v", err)
			}
			if _, err := os.Stat(j.runnerPath()); !os.IsNotExist(err) {
				t.Fatalf("unfinished copy still present: %v", err)
			}
		})
	}
}

func TestJournalInterruptedAdmissionRefusesWithoutLifecycleRecovery(t *testing.T) {
	test := func(name, boundary string, land bool) {
		t.Run(name, func(t *testing.T) {
			j, lock := journalFixture(t)
			r, inventory := journalInputs(t, j)
			original := j.replacer.Replace
			j.replacer.Replace = func(tmp, target string) error {
				if filepath.Base(target) != boundary {
					return original(tmp, target)
				}
				if land {
					if err := original(tmp, target); err != nil {
						return err
					}
					return durablefile.ErrReplaced
				}
				return errors.New("injected publication interruption")
			}
			if _, err := j.Begin(lock, r, inventory); err == nil {
				t.Fatal("interrupted admission returned authority")
			}
			reader, err := openJournalAt(j.Path(), false)
			if err != nil {
				t.Fatal(err)
			}
			state, readErr := reader.Read()
			closeErr := reader.Close()
			if readErr != nil || closeErr != nil || !state.TerminalPending || state.Current == nil || state.Current.OperationID != r.OperationID || state.Current.PreparedIntent || state.Current.Failure == nil || state.Current.Failure.Reason != "admission_publication_uncertain" {
				t.Fatalf("admission crash lost its zero-effect authority: %+v, %v / %v", state, readErr, closeErr)
			}
			j.replacer.Replace = original
			if _, err := j.Recover(lock, strings.Repeat("f", 32)); err == nil {
				t.Fatal("another operation recovered admission")
			}
			state, err = j.Recover(lock, r.OperationID)
			if err != nil || state.Current != nil || state.TerminalPending || state.LastTerminal == nil || state.LastTerminal.Outcome == nil || *state.LastTerminal.Outcome != OutcomeRefused || state.LastTerminal.PreviewID != r.PreviewID {
				t.Fatalf("admission did lifecycle work instead of refusing: %+v, %v", state, err)
			}
			state, err = j.Read()
			if err != nil || state.Current != nil || state.TerminalPending {
				t.Fatalf("recovered admission kept an unresolved gate: %+v, %v", state, err)
			}
		})
	}
	test("admission intent landed before inventory", "admission.json", true)
	test("inventory publication did not land", "inventory.json", false)
	test("inventory publication landed", "inventory.json", true)
	test("initial receipt did not land", "operation.json", false)
	test("initial receipt landed", "operation.json", true)
}

func TestJournalCleanupRetainsTerminalAuthority(t *testing.T) {
	j, lock := journalFixture(t)
	r := journalRestartReceipt(t, j, lock)
	for _, phase := range []Phase{PhaseVerifyBuild, PhaseVerifySettings, PhaseFinish} {
		r.Phase = phase
		saveJournal(t, j, lock, r)
	}
	outcome := OutcomeSucceeded
	r.Outcome, r.CleanupPending, r.ReleasePending = &outcome, true, true
	r = journalVerifiedSuccess(r)
	saveJournal(t, j, lock, r)
	r.ReleasePending, r.CleanupStarted = false, true
	if _, err := j.SaveCleanup(lock, r); err != nil {
		t.Fatal(err)
	}
	r.CleanupPending = false
	if _, err := j.SaveCleanup(lock, r); err != nil {
		t.Fatal(err)
	}
	changed := r
	changed.PreviewID = "another_preview"
	if _, err := j.SaveCleanup(lock, changed); err == nil {
		t.Fatal("cleanup replaced the terminal operation authority")
	}
	state, err := j.Read()
	if err != nil || state.Current != nil || state.LastTerminal == nil || state.LastTerminal.Outcome == nil || *state.LastTerminal.Outcome != OutcomeSucceeded || state.LastTerminal.CleanupPending || state.LastTerminal.ReleasePending || state.LastTerminal.PreviewID != r.PreviewID {
		t.Fatalf("cleanup changed the applied outcome: %+v, %v", state, err)
	}
}

func TestJournalSuccessRequiresVerifiedSelectedBuildAndPause(t *testing.T) {
	test := func(name string, change func(*Receipt)) {
		t.Run(name, func(t *testing.T) {
			j, lock := journalFixture(t)
			r := journalRestartReceipt(t, j, lock)
			for _, phase := range []Phase{PhaseVerifyBuild, PhaseVerifySettings, PhaseFinish} {
				r.Phase = phase
				saveJournal(t, j, lock, r)
			}
			outcome := OutcomeSucceeded
			r.Outcome = &outcome
			r = journalVerifiedSuccess(r)
			change(&r)
			if _, err := j.Save(lock, r); err == nil {
				t.Fatal("unverified success released the startup gate")
			}
			state, err := j.Read()
			if err != nil || state.Current == nil || state.LastTerminal != nil {
				t.Fatalf("rejected success lost pending authority: %+v, %v", state, err)
			}
		})
	}
	test("no owner readback", func(r *Receipt) { r.Verified = nil })
	test("GUI build differs", func(r *Receipt) { r.Verified.RunningGUI.BuildID = Known("git:" + strings.Repeat("f", 40)) })
	test("physical stop unknown", func(r *Receipt) { r.Verified.PhysicalStop = Observation[bool]{} })
	test("pause not enforced", func(r *Receipt) { r.Verified.RemainsPaused = Known(false) })
	test("ownership not acknowledged", func(r *Receipt) { r.RunnerReady = false })
}
