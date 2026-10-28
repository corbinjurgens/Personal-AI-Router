// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/buildapply"
	"nvpair-shared/durablefile"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

// The fixture contains no filesystem/process/IPC adapter. Native execution and
// private fixture ACL validation belong to the parent's serialized interval.
type buildApplyRunFixture struct {
	plan          buildApplyPlan
	state         buildapply.JournalState
	events        []string
	failAt        string
	failSave      buildapply.Phase
	landUncertain bool
	trees         map[string]string
	wrongService  bool
	unknownGUI    bool
	peerPause     bool
	stopCount     int
	launchCount   int
	failTerminal  bool
}

// buildApplyTempDir resolves the test root: macOS TMPDIR sits under the /var
// symlink, which the linked-ancestor checks correctly refuse.
func buildApplyTempDir(t *testing.T) string {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func newBuildApplyRunFixture(t *testing.T) *buildApplyRunFixture {
	t.Helper()
	// These regular-file identity witnesses are never executed. Process
	// execution uses only the separately registered fixed-path test image.
	base := filepath.Join(buildApplyTempDir(t), "private")
	if err := createBuildApplyDirectory(base); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(base, "target")
	layout, ok := buildapply.LayoutFor(runtime.GOOS)
	if !ok {
		t.Fatal("fixture platform unsupported")
	}
	serviceImage := filepath.Join(target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable)
	guiImage := filepath.Join(target, filepath.FromSlash(layout.Executable))
	brokerImage := filepath.Join(filepath.Dir(serviceImage), brokerName)
	if runtime.GOOS == "windows" {
		brokerImage += ".exe"
	}
	for _, image := range []string{serviceImage, guiImage, brokerImage} {
		if err := os.MkdirAll(filepath.Dir(image), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(image, []byte("synthetic identity witness, never executed"), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Now()
	digest := strings.Repeat("a", 64)
	identity := buildapply.Identity{BuildID: buildapply.Known("git:" + strings.Repeat("b", 40)), SourceRevision: buildapply.Known(strings.Repeat("b", 40)), Dirty: buildapply.Known(false), SHA256: buildapply.Known(digest)}
	f := &buildApplyRunFixture{trees: map[string]string{target: strings.Repeat("c", 64)}}
	f.plan = buildApplyPlan{accepted: now, deadline: now.Add(buildApplyTotalBudget), expires: now.Add(time.Minute), ownerFactsComplete: true,
		oldProcesses:  []buildapply.RunnerReservation{{PID: 101, Identity: buildapply.ProcessIdentity{Executable: guiImage, Created: "123"}}, {PID: 102, Identity: buildapply.ProcessIdentity{Executable: serviceImage, Created: "456"}}},
		serviceLaunch: buildApplyLaunch{executable: serviceImage, env: []string{}},
		guiLaunch:     buildApplyLaunch{executable: guiImage, env: []string{}}, expectedService: identity, expectedGUI: identity}
	f.plan.receipt = buildapply.Receipt{OperationID: strings.Repeat("1", 32), PreviewID: "preview_one", Target: target, CandidateSHA256: digest, PreviousSHA256: strings.Repeat("c", 64),
		Protected: buildapply.ProtectedDigests{Policy: digest, EngineSettings: digest, Intent: digest, Locks: digest, IdentityMembership: digest},
		Expected:  buildapply.JournalExpectations{StopEngines: true, UnloadModels: true, RemainPaused: true, PreserveIntentLocks: true, OnActive: nodepolicy.FinishActive, DrainTimeoutSeconds: 120},
		Consent:   buildapply.Consent{AllowInterruption: true, AllowEngineStop: true}, Phase: buildapply.PhasePrepare, Rename: buildapply.RenameNone, Handoff: buildapply.HandoffNone,
		Fences: buildapply.Fences{RuntimeGeneration: buildapply.Known("runtime"), ServiceInstance: buildapply.Known("service"), BrokerGeneration: buildapply.Known("broker"), ManagerIncarnation: buildapply.Known("manager"), PolicyRevision: buildapply.Known("policy"), SettingsRevision: buildapply.Known("settings"), IntentRevision: buildapply.Known("intent"), LocksRevision: buildapply.Known("locks"), AvailabilityRevision: buildapply.Known("availability")}}
	f.plan.inventory = buildapply.Inventory{OperationID: f.plan.receipt.OperationID, PackageSHA256: digest, Platform: runtime.GOOS, Architecture: runtime.GOARCH}
	endpoint := filepath.Join(filepath.Dir(target), "service.sock")
	if runtime.GOOS == "windows" {
		endpoint = `\\.\pipe\pair-build-apply-injected`
	}
	f.plan.receipt.Runtime = buildapply.RuntimeAuthority{AcceptedAt: now, Deadline: now.Add(buildApplyTotalBudget), CandidatePath: filepath.Join(filepath.Dir(target), "candidate"), Endpoint: endpoint, OldProcesses: f.plan.oldProcesses, OldGUI: f.plan.oldProcesses[0], OldService: f.plan.oldProcesses[1], ServiceLaunch: buildapply.LaunchContext{EnvironmentSHA256: digest}, GUILaunch: buildapply.LaunchContext{EnvironmentSHA256: digest}, ExpectedService: identity, ExpectedGUI: identity, RunnerSHA256: digest, TargetIdentity: "123:456", CandidateMetadataSHA256: digest}
	f.plan.receipt.RunnerToken = f.plan.receipt.OperationID
	f.plan.receipt.Runtime.ServiceLaunch.EnvironmentSHA256 = buildApplyEnvironmentDigest(f.plan.serviceLaunch.env)
	f.plan.receipt.Runtime.GUILaunch.EnvironmentSHA256 = buildApplyEnvironmentDigest(f.plan.guiLaunch.env)
	return f
}

func (f *buildApplyRunFixture) PreparedIdentity() string { return "777:888" }

func (f *buildApplyRunFixture) event(name string) error {
	f.events = append(f.events, name)
	if f.failAt == name {
		return errors.New("injected boundary failure")
	}
	return nil
}
func (f *buildApplyRunFixture) Begin(r buildapply.Receipt, i buildapply.Inventory) (buildapply.JournalState, error) {
	if err := f.event("begin"); err != nil {
		return f.state, err
	}
	r.InventorySHA256 = strings.Repeat("d", 64)
	f.state.Current = &r
	return f.state, nil
}
func (f *buildApplyRunFixture) Save(r buildapply.Receipt) (buildapply.JournalState, error) {
	name := "save:" + string(r.Phase) + ":" + string(r.Rename) + ":" + string(r.Handoff)
	if err := f.event(name); err != nil {
		return f.state, err
	}
	if f.failTerminal && r.Outcome != nil && *r.Outcome == buildapply.OutcomeSucceeded {
		return f.state, durablefile.ErrReplaced
	}
	if f.failSave == r.Phase && (r.Outcome == nil || *r.Outcome != buildapply.OutcomeUncertain) {
		if f.landUncertain {
			f.state.Current = &r
		}
		return f.state, durablefile.ErrReplaced
	}
	if r.Outcome != nil && *r.Outcome == buildapply.OutcomeSucceeded {
		f.state.Current = nil
		f.state.LastTerminal = &r
	} else {
		f.state.Current = &r
	}
	return f.state, nil
}
func (f *buildApplyRunFixture) SaveCleanup(r buildapply.Receipt) (buildapply.JournalState, error) {
	f.state.LastTerminal = &r
	return f.state, nil
}
func (f *buildApplyRunFixture) Freeze(ctx context.Context, p buildApplyPlan) error {
	if err := f.event("freeze"); err != nil {
		return err
	}
	paths, err := buildapply.DeriveOperationPaths(p.receipt.Target, p.receipt.OperationID)
	if err != nil {
		return err
	}
	f.trees[paths.Prepared] = p.receipt.CandidateSHA256
	return ctx.Err()
}
func (f *buildApplyRunFixture) Digest(ctx context.Context, path string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return f.trees[path], nil
}
func (f *buildApplyRunFixture) Rename(ctx context.Context, from, to string) error {
	name := "rename-prepared"
	if from == f.plan.receipt.Target {
		name = "rename-target"
	}
	if err := f.event(name); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if f.trees[from] == "" || f.trees[to] != "" {
		return errors.New("unsafe rename")
	}
	f.trees[to] = f.trees[from]
	delete(f.trees, from)
	return nil
}
func (f *buildApplyRunFixture) Prepare(ctx context.Context, r buildapply.PrepareRequest) (buildApplyPrepared, error) {
	if err := f.event("prepare-pause"); err != nil {
		return buildApplyPrepared{}, err
	}
	if r.PreviewID != f.plan.receipt.PreviewID || !reflect.DeepEqual(r.Fences, f.plan.receipt.Fences) || !r.Consent.AllowEngineStop {
		return buildApplyPrepared{}, errors.New("wrong prepare authority")
	}
	pause := buildapply.JournalPauseFence{Revision: "paused", Origin: nodepolicy.PausedLocally, Owned: true, PolicySHA256: strings.Repeat("e", 64)}
	if f.peerPause {
		pause.Origin = nodepolicy.PausedByPeer
		pause.Owned = false
	}
	op := projectBuildApplyReceipt(f.plan.receipt)
	op.Phase = buildapply.PhaseQuiesce
	op.Saved = buildapply.Known(true)
	op.Enforced = buildapply.Known(true)
	op.PhysicalStop = buildapply.Known(true)
	op.RemainsPaused = buildapply.Known(true)
	op.Fences.PolicyRevision = buildapply.Known("post-pause")
	return buildApplyPrepared{pause: pause, operation: op}, ctx.Err()
}
func (f *buildApplyRunFixture) Reserve(ctx context.Context, r buildapply.Receipt) (buildapply.RunnerReservation, error) {
	if err := f.event("reserve"); err != nil {
		return buildapply.RunnerReservation{}, err
	}
	if r.Handoff != buildapply.HandoffIntent || r.Pause == nil {
		return buildapply.RunnerReservation{}, errors.New("missing handoff intent")
	}
	return buildapply.RunnerReservation{PID: 202, Identity: buildapply.ProcessIdentity{Executable: filepath.Join(filepath.Dir(r.Target), "runner"), Created: "456"}}, ctx.Err()
}
func (f *buildApplyRunFixture) Acknowledge(ctx context.Context, r buildapply.Receipt) error {
	if r.Handoff != buildapply.HandoffReserved || r.Runner == nil {
		return errors.New("missing exact reservation")
	}
	return f.event("acknowledge")
}
func (f *buildApplyRunFixture) Stop(ctx context.Context, old []buildapply.RunnerReservation) error {
	if err := f.event("stop"); err != nil {
		return err
	}
	if f.state.Current == nil || f.state.Current.Handoff != buildapply.HandoffAcknowledged || f.state.Current.Phase != buildapply.PhaseShutdown {
		return errors.New("shutdown preceded durable acknowledgement")
	}
	if !reflect.DeepEqual(old, f.plan.oldProcesses) {
		return errors.New("changed process inventory")
	}
	f.stopCount++
	return ctx.Err()
}
func (f *buildApplyRunFixture) Restart(ctx context.Context, p buildApplyPlan) (buildApplySuccessors, error) {
	if err := f.event("restart"); err != nil {
		return buildApplySuccessors{}, err
	}
	f.launchCount++
	return buildApplySuccessors{service: buildapply.RunnerReservation{PID: 303, Identity: buildapply.ProcessIdentity{Executable: p.serviceLaunch.executable, Created: "789"}}, gui: buildapply.RunnerReservation{PID: 404, Identity: buildapply.ProcessIdentity{Executable: p.guiLaunch.executable, Created: "abc"}}, serviceReady: true}, ctx.Err()
}
func (f *buildApplyRunFixture) ReadBuild(ctx context.Context, p buildApplyPlan) (buildApplyBuildReadback, error) {
	if err := f.event("verify-build"); err != nil {
		return buildApplyBuildReadback{}, err
	}
	service, gui := p.expectedService, p.expectedGUI
	if f.wrongService {
		service.BuildID = buildapply.Known("git:" + strings.Repeat("f", 40))
	}
	if f.unknownGUI {
		gui = buildapply.Identity{}
	}
	return buildApplyBuildReadback{service: service, gui: gui, serviceProcess: buildapply.RunnerReservation{PID: 303, Identity: buildapply.ProcessIdentity{Executable: p.serviceLaunch.executable, Created: "789"}}, guiProcess: buildapply.RunnerReservation{PID: 404, Identity: buildapply.ProcessIdentity{Executable: p.guiLaunch.executable, Created: "abc"}}, brokerReady: true, guiAcknowledged: true}, ctx.Err()
}
func (f *buildApplyRunFixture) ReadSettings(ctx context.Context, r buildapply.Receipt) (buildApplySettingsReadback, error) {
	if err := f.event("verify-settings"); err != nil {
		return buildApplySettingsReadback{}, err
	}
	if r.Pause == nil {
		return buildApplySettingsReadback{}, errors.New("missing Pause")
	}
	digests := r.Protected
	digests.Policy = r.Pause.PolicySHA256
	fences := r.Fences
	fences.PolicyRevision = buildapply.Known("post-pause")
	fences.ServiceInstance = buildapply.Known("new-service")
	return buildApplySettingsReadback{fences: fences, protected: digests, pause: *r.Pause, stopEngines: true, unloadModels: true, saved: true, enforced: true, physicalStop: true, activationVerified: true}, ctx.Err()
}
func (f *buildApplyRunFixture) Finish(ctx context.Context, r buildapply.FinishRequest) error {
	if f.state.LastTerminal == nil || f.state.LastTerminal.Outcome == nil || *f.state.LastTerminal.Outcome != buildapply.OutcomeSucceeded || f.state.Current != nil {
		return errors.New("maintenance released before durable terminal commit")
	}
	if r.Fences.PolicyRevision.Value == nil || *r.Fences.PolicyRevision.Value != "post-pause" || r.Fences.ServiceInstance.Value == nil || *r.Fences.ServiceInstance.Value != "new-service" {
		return errors.New("finish used stale admission fences")
	}
	return f.event("finish")
}

func TestBuildApplyTransactionCleanupFailureRetainsSucceededReceipt(t *testing.T) {
	f := newBuildApplyRunFixture(t)
	f.failAt = "finish"
	o, err := f.transaction().Run(context.Background())
	var cleanup *buildApplyCleanupError
	if !errors.As(err, &cleanup) || o.Outcome == nil || *o.Outcome != buildapply.OutcomeSucceeded {
		t.Fatalf("cleanup changed applied outcome: %+v, %v", o, err)
	}
	if f.state.Current != nil || f.state.LastTerminal == nil || !f.state.LastTerminal.ReleasePending || !f.state.LastTerminal.CleanupPending || o.RemainsPaused.Value == nil || !*o.RemainsPaused.Value {
		t.Fatalf("cleanup receipt/Pause: %+v", f.state)
	}
}
func (f *buildApplyRunFixture) transaction() *buildApplyTransaction {
	return &buildApplyTransaction{plan: f.plan, store: f, files: f, owners: f}
}

func TestBuildApplyTransactionFinishesWhilePaused(t *testing.T) {
	f := newBuildApplyRunFixture(t)
	o, err := f.transaction().Run(context.Background())
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if o.Outcome == nil || *o.Outcome != buildapply.OutcomeSucceeded || o.PreviewID != f.plan.receipt.PreviewID || !o.RemainsPaused.Known || o.RemainsPaused.Value == nil || !*o.RemainsPaused.Value {
		t.Fatalf("unexpected operation: %+v", o)
	}
	if f.stopCount != 1 || f.launchCount != 1 {
		t.Fatalf("stop/start counts: %d/%d", f.stopCount, f.launchCount)
	}
	paths, err := buildapply.DeriveOperationPaths(f.plan.receipt.Target, f.plan.receipt.OperationID)
	if err != nil {
		t.Fatal(err)
	}
	if f.trees[paths.Target] != f.plan.receipt.CandidateSHA256 || f.trees[paths.Previous] != f.plan.receipt.PreviousSHA256 {
		t.Fatalf("retained trees: %v", f.trees)
	}
	if f.state.Current != nil || f.state.LastTerminal == nil {
		t.Fatalf("terminal state: %+v", f.state)
	}
	if err := f.state.LastTerminal.ValidateOperation(o); err != nil {
		t.Fatalf("receipt projection: %v", err)
	}
}

func TestBuildApplyTransactionTerminalPublicationUncertaintyRemainsPaused(t *testing.T) {
	f := newBuildApplyRunFixture(t)
	f.failTerminal = true
	o, err := f.transaction().Run(context.Background())
	if err == nil || o.Outcome == nil || *o.Outcome != buildapply.OutcomeUncertain || o.Failure == nil || o.Failure.Reason != "terminal_publication_uncertain" {
		t.Fatalf("terminal uncertainty: %+v, %v", o, err)
	}
	if f.state.Current == nil || f.state.LastTerminal != nil || o.RemainsPaused.Value == nil || !*o.RemainsPaused.Value {
		t.Fatalf("terminal uncertainty lost gate/Pause: %+v", o)
	}
}

func TestBuildApplyTransactionInvalidAdmissionHasZeroEffects(t *testing.T) {
	test := func(name string, change func(*buildApplyPlan)) {
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			change(&f.plan)
			if _, err := f.transaction().Run(context.Background()); err == nil {
				t.Fatal("invalid plan was admitted")
			}
			if len(f.events) != 0 {
				t.Fatalf("invalid plan produced effects: %v", f.events)
			}
		})
	}
	test("expired preview", func(p *buildApplyPlan) { p.expires = time.Now().Add(-time.Second) })
	test("missing protected digest", func(p *buildApplyPlan) { p.receipt.Protected.Intent = "" })
	test("drain budget does not fit", func(p *buildApplyPlan) { p.receipt.Expected.DrainTimeoutSeconds = 1800 })
	test("missing process creation identity", func(p *buildApplyPlan) { p.oldProcesses[0].Identity.Created = "" })
	test("untrusted successor path", func(p *buildApplyPlan) {
		p.serviceLaunch.executable = filepath.Join(filepath.Dir(p.receipt.Target), "arbitrary")
	})
	test("changed launch environment", func(p *buildApplyPlan) { p.guiLaunch.env = []string{"LOCALAPPDATA=changed"} })
	test("changed launch arguments", func(p *buildApplyPlan) { p.serviceLaunch.args = []string{"--log-level=debug"} })
	test("changed execution deadline", func(p *buildApplyPlan) { p.deadline = p.deadline.Add(time.Second) })
}

func TestBuildApplySettingsReadbackRejectsUnexpectedSemantics(t *testing.T) {
	test := func(name string, change func(*buildApplySettingsReadback)) {
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			r := f.plan.receipt
			r.Pause = &buildapply.JournalPauseFence{Revision: "pause", Origin: nodepolicy.PausedLocally, Owned: true, PolicySHA256: strings.Repeat("e", 64)}
			read, err := f.ReadSettings(context.Background(), r)
			if err != nil {
				t.Fatal(err)
			}
			if err := verifyBuildApplySettings(r, read); err != nil {
				t.Fatalf("valid readback: %v", err)
			}
			change(&read)
			if err := verifyBuildApplySettings(r, read); err == nil {
				t.Fatal("unexpected semantics accepted")
			}
		})
	}
	test("changed engine settings", func(r *buildApplySettingsReadback) { r.protected.EngineSettings = strings.Repeat("f", 64) })
	test("changed saved intent", func(r *buildApplySettingsReadback) { r.protected.Intent = strings.Repeat("f", 64) })
	test("changed locks", func(r *buildApplySettingsReadback) { r.protected.Locks = strings.Repeat("f", 64) })
	test("changed identity membership", func(r *buildApplySettingsReadback) { r.protected.IdentityMembership = strings.Repeat("f", 64) })
	test("newer local Pause", func(r *buildApplySettingsReadback) { r.pause.Revision = "newer" })
	test("unknown activation", func(r *buildApplySettingsReadback) { r.activationVerified = false })
	test("recovery still required", func(r *buildApplySettingsReadback) { r.recoveryRequired = true })
	test("false canonical unload", func(r *buildApplySettingsReadback) { r.unloadModels = false })
}

func TestBuildApplyTransactionPhaseFailuresRetainRecovery(t *testing.T) {
	test := func(name, boundary string, wantPhase buildapply.Phase) {
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			f.failAt = boundary
			o, err := f.transaction().Run(context.Background())
			if err == nil || o.Outcome == nil || *o.Outcome != buildapply.OutcomeUncertain || o.Phase != wantPhase || o.Failure == nil {
				t.Fatalf("operation=%+v, error=%v", o, err)
			}
			if f.state.Current == nil || f.state.LastTerminal != nil {
				t.Fatalf("unresolved state: %+v", f.state)
			}
			valid := false
			for _, hash := range f.trees {
				if hash == f.plan.receipt.PreviousSHA256 {
					valid = true
				}
			}
			if !valid {
				t.Fatal("only valid predecessor was lost")
			}
		})
	}
	test("frozen copy failure", "freeze", buildapply.PhasePrepare)
	test("physical stop unknown", "prepare-pause", buildapply.PhaseQuiesce)
	test("runner reservation failure", "reserve", buildapply.PhaseQuiesce)
	test("handoff acknowledgement lost", "acknowledge", buildapply.PhaseQuiesce)
	test("old process exit unknown", "stop", buildapply.PhaseShutdown)
	test("first rename failure", "rename-target", buildapply.PhaseSwap)
	test("second rename failure", "rename-prepared", buildapply.PhaseSwap)
	test("new service startup unknown", "restart", buildapply.PhaseRestart)
	test("GUI build read unavailable", "verify-build", buildapply.PhaseVerifyBuild)
	test("newer local safety veto", "verify-settings", buildapply.PhaseVerifySettings)
}

func TestBuildApplyTransactionDurabilityFailuresPreventEffects(t *testing.T) {
	for _, phase := range []buildapply.Phase{buildapply.PhaseQuiesce, buildapply.PhaseShutdown, buildapply.PhaseSwap, buildapply.PhaseRestart, buildapply.PhaseVerifyBuild, buildapply.PhaseVerifySettings, buildapply.PhaseFinish} {
		t.Run(string(phase), func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			f.failSave = phase
			f.landUncertain = true
			o, err := f.transaction().Run(context.Background())
			if !errors.Is(err, durablefile.ErrReplaced) && (o.Failure == nil || o.Outcome == nil || *o.Outcome != buildapply.OutcomeUncertain) {
				t.Fatalf("uncertain publication: %+v, %v", o, err)
			}
			if o.Outcome == nil || *o.Outcome == buildapply.OutcomeSucceeded || f.state.LastTerminal != nil {
				t.Fatalf("uncertain publication became success: %+v", o)
			}
		})
	}
}

func TestBuildApplyTransactionRejectsWrongAndUnknownReadbacks(t *testing.T) {
	test := func(name string, wrong, unknown bool) {
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			f.wrongService = wrong
			f.unknownGUI = unknown
			o, err := f.transaction().Run(context.Background())
			if err == nil || o.Failure == nil || o.Failure.Reason != "running_build_unverified" || f.state.Current == nil {
				t.Fatalf("readback: %+v, %v", o, err)
			}
			if o.RemainsPaused.Value == nil || !*o.RemainsPaused.Value {
				t.Fatalf("Pause observation: %+v", o.RemainsPaused)
			}
		})
	}
	test("wrong actual service bytes", true, false)
	test("GUI acknowledgement unknown", false, true)
}

func TestBuildApplyTransactionPreservesPeerPause(t *testing.T) {
	f := newBuildApplyRunFixture(t)
	f.peerPause = true
	if _, err := f.transaction().Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	p := f.state.LastTerminal.Pause
	if p == nil || p.Origin != nodepolicy.PausedByPeer || p.Owned {
		t.Fatalf("peer Pause was adopted: %+v", p)
	}
}

func TestBuildApplyBrokerResultRequiresExactPreviewAndPhysicalProof(t *testing.T) {
	test := func(name string, change func(*buildapply.BuildApplyOperation)) {
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			prepared, err := f.Prepare(context.Background(), buildapply.PrepareRequest{OperationID: f.plan.receipt.OperationID, PreviewID: f.plan.receipt.PreviewID, Fences: f.plan.receipt.Fences, Consent: f.plan.receipt.Consent})
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &prepared.operation})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := decodeBuildApplyOwnerOperation(data, f.plan.receipt.OperationID, f.plan.receipt.PreviewID, buildapply.PhaseQuiesce); err != nil {
				t.Fatalf("valid owner result: %v", err)
			}
			change(&prepared.operation)
			data, err = json.Marshal(buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &prepared.operation})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := decodeBuildApplyOwnerOperation(data, f.plan.receipt.OperationID, f.plan.receipt.PreviewID, buildapply.PhaseQuiesce); err == nil {
				t.Fatal("uncorrelated or incomplete result accepted")
			}
		})
	}
	test("another preview", func(o *buildapply.BuildApplyOperation) { o.PreviewID = "another_preview" })
	test("unknown physical stop", func(o *buildapply.BuildApplyOperation) { o.PhysicalStop = buildapply.Observation[bool]{} })
	test("uncertain owner result", func(o *buildapply.BuildApplyOperation) { v := buildapply.OutcomeUncertain; o.Outcome = &v })
}

func TestBuildApplyRecoveryReconcilesRenameCrashes(t *testing.T) {
	test := func(name string, boundary buildapply.RenameState, firstLanded, secondLanded bool) {
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			tx := f.transaction()
			tx.receipt = f.plan.receipt
			tx.receipt.Phase = buildapply.PhaseSwap
			tx.receipt.Rename = boundary
			tx.operation = projectBuildApplyReceipt(tx.receipt)
			paths, err := buildapply.DeriveOperationPaths(tx.receipt.Target, tx.receipt.OperationID)
			if err != nil {
				t.Fatal(err)
			}
			f.trees[paths.Prepared] = tx.receipt.CandidateSHA256
			if firstLanded {
				f.trees[paths.Previous] = tx.receipt.PreviousSHA256
				delete(f.trees, paths.Target)
			}
			if secondLanded {
				f.trees[paths.Target] = tx.receipt.CandidateSHA256
				delete(f.trees, paths.Prepared)
			}
			if err := tx.swap(context.Background()); err != nil {
				t.Fatalf("reconcile: %v", err)
			}
			if tx.receipt.Rename != buildapply.RenameInstalled || f.trees[paths.Target] != tx.receipt.CandidateSHA256 || f.trees[paths.Previous] != tx.receipt.PreviousSHA256 {
				t.Fatalf("reconciled trees: %v", f.trees)
			}
		})
	}
	test("before first rename", buildapply.RenameTargetIntent, false, false)
	test("after first rename before receipt", buildapply.RenameTargetIntent, true, false)
	test("after retained predecessor receipt", buildapply.RenamePreviousRetained, true, false)
	test("before second rename", buildapply.RenamePreparedIntent, true, false)
	test("after second rename before receipt", buildapply.RenamePreparedIntent, true, true)
	test("after installed receipt", buildapply.RenameInstalled, true, true)
}

func TestBuildApplyRecoveryRejectsAmbiguousTreeWithoutDeleting(t *testing.T) {
	f := newBuildApplyRunFixture(t)
	tx := f.transaction()
	tx.receipt = f.plan.receipt
	tx.receipt.Phase = buildapply.PhaseSwap
	tx.receipt.Rename = buildapply.RenamePreparedIntent
	before := map[string]string{}
	for k, v := range f.trees {
		before[k] = v
	}
	if err := tx.swap(context.Background()); err == nil {
		t.Fatal("accepted ambiguous directory state")
	}
	if !reflect.DeepEqual(before, f.trees) {
		t.Fatalf("changed ambiguous state: %v", f.trees)
	}
}

type buildApplyAdmissionFixture struct {
	plan         buildApplyPlan
	state        buildapply.JournalState
	readErr      error
	readCount    int
	captureCount int
	executeCount int
	mu           sync.Mutex
	entered      chan struct{}
	release      chan struct{}
}

func (f *buildApplyAdmissionFixture) ReadJournal() (buildapply.JournalState, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.readCount++
	return f.state, f.readErr
}
func (f *buildApplyAdmissionFixture) Capture(ctx context.Context, p buildApplyAdmissionSnapshot, r buildapply.StartRequest) (buildApplyPlan, error) {
	f.mu.Lock()
	f.captureCount++
	f.mu.Unlock()
	return f.plan, ctx.Err()
}
func (f *buildApplyAdmissionFixture) Execute(ctx context.Context, p buildApplyPlan) buildapply.Result[buildapply.BuildApplyOperation] {
	f.mu.Lock()
	f.executeCount++
	f.mu.Unlock()
	if f.entered != nil {
		close(f.entered)
		select {
		case <-f.release:
		case <-ctx.Done():
			return buildapply.Unavailable[buildapply.BuildApplyOperation]()
		}
	}
	o := projectBuildApplyReceipt(p.receipt)
	o.OwnershipAcknowledged = true
	return buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &o}
}
func buildApplyAdmissionService(t *testing.T) (*Service, *client, buildapply.StartRequest, *buildApplyAdmissionFixture) {
	t.Helper()
	f := newBuildApplyRunFixture(t)
	s := newService(config{})
	c := &client{}
	run := &brokerRun{exited: make(chan struct{})}
	s.broker = run
	s.clients[c] = struct{}{}
	s.buildPreview = &buildApplyPreviewState{client: c, broker: run, expires: f.plan.expires, gui: buildapply.ProcessIdentity{Executable: f.plan.guiLaunch.executable, Created: "123"}, preview: buildapply.BuildApplyPreview{PreviewID: f.plan.receipt.PreviewID, ObservationsComplete: true, CurrentGUI: f.plan.expectedGUI, RequestedGUI: f.plan.expectedGUI, RequestedService: f.plan.expectedService, PackageSHA256: buildapply.Known(f.plan.receipt.CandidateSHA256), Impact: buildapply.Impact{Complete: true, Fences: f.plan.receipt.Fences}}}
	r := buildapply.StartRequest{PreviewID: f.plan.receipt.PreviewID, Consent: f.plan.receipt.Consent, GUI: buildapply.GUIFacts{PID: 101, Executable: f.plan.guiLaunch.executable, BuildID: *f.plan.expectedGUI.BuildID.Value}}
	return s, c, r, &buildApplyAdmissionFixture{plan: f.plan}
}

func TestBuildApplyAdmissionFalseConsentHasZeroEffects(t *testing.T) {
	s, c, r, f := buildApplyAdmissionService(t)
	r.Consent.AllowEngineStop = false
	got := s.startBuildApply(c, r, f)
	if got.Available || f.readCount != 0 || f.captureCount != 0 || f.executeCount != 0 {
		t.Fatalf("refusal=%+v, reads/capture/execute=%d/%d/%d", got, f.readCount, f.captureCount, f.executeCount)
	}
}
func TestBuildApplyAdmissionStalePreviewHasZeroMutation(t *testing.T) {
	s, c, r, f := buildApplyAdmissionService(t)
	s.buildPreview.expires = time.Now().Add(-time.Second)
	got := s.startBuildApply(c, r, f)
	if got.Available || f.captureCount != 0 || f.executeCount != 0 {
		t.Fatalf("stale result=%+v, capture/execute=%d/%d", got, f.captureCount, f.executeCount)
	}
}
func TestBuildApplyAdmissionMissingOwnerFactsHasZeroMutation(t *testing.T) {
	s, c, r, f := buildApplyAdmissionService(t)
	f.plan.ownerFactsComplete = false
	got := s.startBuildApply(c, r, f)
	if got.Available || f.executeCount != 0 {
		t.Fatalf("missing evidence admitted: %+v", got)
	}
}

func TestBuildApplyAdmissionChangedOwnerFenceHasZeroMutation(t *testing.T) {
	s, c, r, f := buildApplyAdmissionService(t)
	f.plan.receipt.Fences.SettingsRevision = buildapply.Known("changed")
	got := s.startBuildApply(c, r, f)
	if got.Available || f.executeCount != 0 {
		t.Fatalf("changed owner fence admitted: %+v", got)
	}
}

func TestBuildApplyAdmissionChangedPreviewCorrelationHasZeroMutation(t *testing.T) {
	s, c, r, f := buildApplyAdmissionService(t)
	f.plan.receipt.PreviewID = "another_preview"
	got := s.startBuildApply(c, r, f)
	if got.Available || f.executeCount != 0 {
		t.Fatalf("changed preview correlation admitted: %+v", got)
	}
}
func TestBuildApplyAdmissionUntrustedJournalHasZeroMutation(t *testing.T) {
	s, c, r, f := buildApplyAdmissionService(t)
	f.readErr = errors.New("untrusted existing state")
	got := s.startBuildApply(c, r, f)
	if got.Available || f.captureCount != 0 || f.executeCount != 0 {
		t.Fatalf("untrusted state admitted: %+v", got)
	}
}
func TestBuildApplyAdmissionDuplicateSurvivesDisconnectAndLostReply(t *testing.T) {
	s, c, r, f := buildApplyAdmissionService(t)
	f.entered = make(chan struct{})
	f.release = make(chan struct{})
	first := make(chan buildapply.Result[buildapply.BuildApplyOperation], 1)
	go func() { first <- s.startBuildApply(c, r, f) }()
	select {
	case <-f.entered:
	case <-time.After(time.Second):
		t.Fatal("execution not reached")
	}
	s.mu.Lock()
	delete(s.clients, c)
	s.invalidateBuildApplyPreviewLocked(c)
	s.mu.Unlock()
	close(f.release)
	var accepted buildapply.Result[buildapply.BuildApplyOperation]
	select {
	case accepted = <-first:
	case <-time.After(time.Second):
		t.Fatal("accepted transaction depended on client")
	}
	duplicate := s.startBuildApply(&client{}, r, f)
	if !accepted.Available || !reflect.DeepEqual(accepted, duplicate) || f.executeCount != 1 {
		t.Fatalf("accepted=%+v duplicate=%+v executions=%d", accepted, duplicate, f.executeCount)
	}
}

func TestBuildApplyAdmissionRecordedOperationRequiresExactPreview(t *testing.T) {
	s, c, r, f := buildApplyAdmissionService(t)
	f.state.Current = &f.plan.receipt
	matching := s.startBuildApply(c, r, f)
	r.PreviewID = "another_preview"
	other := s.startBuildApply(c, r, f)
	if !matching.Available || matching.Value.PreviewID != f.plan.receipt.PreviewID || other.Available || f.executeCount != 0 {
		t.Fatalf("matching=%+v other=%+v", matching, other)
	}
}

func TestBuildApplyStatusIdlePendingTerminalAndUnreadable(t *testing.T) {
	f := newBuildApplyRunFixture(t)
	test := func(name string, state buildapply.JournalState, readErr error, want buildapply.StatusState, available bool) {
		t.Run(name, func(t *testing.T) {
			got := readBuildApplyStatus(func() (buildapply.JournalState, error) { return state, readErr }, "")
			if got.Available != available {
				t.Fatalf("availability: %+v", got)
			}
			if available && (got.Value == nil || got.Value.State != want) {
				t.Fatalf("status: %+v", got)
			}
			data, err := buildapply.EncodeStatusResult(got)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := buildapply.DecodeStatusResult(data, "")
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, decoded) {
				t.Fatalf("status round trip: %+v", decoded)
			}
		})
	}
	test("idle", buildapply.JournalState{}, nil, buildapply.StatusIdle, true)
	test("pending", buildapply.JournalState{Current: &f.plan.receipt}, nil, buildapply.StatusPending, true)
	r := f.plan.receipt
	outcome := buildapply.OutcomeUncertain
	r.Outcome = &outcome
	test("uncertain remains retained", buildapply.JournalState{Current: &r}, nil, buildapply.StatusTerminal, true)
	test("terminal publication unresolved", buildapply.JournalState{Current: &r, TerminalPending: true}, nil, buildapply.StatusUnknown, true)
	test("unreadable is unavailable", buildapply.JournalState{}, errors.New("read failed"), "", false)
}

func TestBuildApplyTransactionPreservedImageOverridesStayInsideBundle(t *testing.T) {
	f := newBuildApplyRunFixture(t)
	image := filepath.Join(filepath.Dir(f.plan.serviceLaunch.executable), brokerName)
	if runtime.GOOS == "windows" {
		image += ".exe"
	}
	if err := validateBuildApplyImageArguments(f.plan.receipt.Target, []string{"--broker-path", image}); err != nil {
		t.Fatalf("canonical preserved override: %v", err)
	}
	if err := validateBuildApplyImageArguments(f.plan.receipt.Target, []string{"--broker-path=" + filepath.Join(filepath.Dir(f.plan.receipt.Target), filepath.Base(image))}); err == nil {
		t.Fatal("preserved override escaped the replacement bundle")
	}
	if err := validateBuildApplyArguments([]string{"--", "--scanner-mdns=false", "--log-level", "debug"}, false); err != nil {
		t.Fatalf("ordinary broker arguments: %v", err)
	}
}

func TestBuildApplyRecoveryNativeRenameReconcilesExactDirectories(t *testing.T) {
	base := filepath.Join(buildApplyTempDir(t), "private")
	if err := createBuildApplyDirectory(base); err != nil {
		t.Fatal(err)
	}
	f := newBuildApplyRunFixture(t)
	f.plan.receipt.Target = filepath.Join(base, "target")
	paths, err := buildapply.DeriveOperationPaths(f.plan.receipt.Target, f.plan.receipt.OperationID)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct{ path, value string }{{paths.Target, "old build data"}, {paths.Prepared, "selected build data"}} {
		if err := createBuildApplyDirectory(item.path); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(item.path, "payload.dat"), []byte(item.value), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	files := &nativeBuildApplyFiles{identities: map[string]os.FileInfo{}}
	for _, path := range []string{paths.Target, paths.Prepared} {
		info, err := os.Lstat(path)
		if err != nil {
			t.Fatal(err)
		}
		files.identities[path] = info
	}
	ctx := context.Background()
	f.plan.receipt.PreviousSHA256, err = files.Digest(ctx, paths.Target)
	if err != nil {
		t.Fatal(err)
	}
	f.plan.receipt.CandidateSHA256, err = files.Digest(ctx, paths.Prepared)
	if err != nil {
		t.Fatal(err)
	}
	f.plan.receipt.Phase = buildapply.PhaseSwap
	tx := &buildApplyTransaction{plan: f.plan, store: f, files: files, receipt: f.plan.receipt, operation: projectBuildApplyReceipt(f.plan.receipt)}
	if err := tx.swap(ctx); err != nil {
		t.Fatal(err)
	}
	installed, err := files.Digest(ctx, paths.Target)
	if err != nil {
		t.Fatal(err)
	}
	previous, err := files.Digest(ctx, paths.Previous)
	if err != nil {
		t.Fatal(err)
	}
	if installed != f.plan.receipt.CandidateSHA256 || previous != f.plan.receipt.PreviousSHA256 || tx.receipt.Rename != buildapply.RenameInstalled {
		t.Fatalf("native rename lost selected or retained bytes: %s / %s / %s", installed, previous, tx.receipt.Rename)
	}
}

func TestBuildApplyTransactionNativeMetadataPreservesDataFile(t *testing.T) {
	base := filepath.Join(buildApplyTempDir(t), "private")
	if err := createBuildApplyDirectory(base); err != nil {
		t.Fatal(err)
	}
	source, destination := filepath.Join(base, "source.dat"), filepath.Join(base, "destination.dat")
	for _, path := range []string{source, destination} {
		if err := os.WriteFile(path, []byte("bounded fixture data"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	want, err := readBuildApplyNativeMetadata(source)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeBuildApplyNativeMetadata(destination, want); err != nil {
		t.Fatal(err)
	}
	got, err := readBuildApplyNativeMetadata(destination)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("native file metadata: %+v / %+v", got, want)
	}
}

func TestBuildApplyRecoveryRefusesSubstitutedIncompletePreparation(t *testing.T) {
	base := filepath.Join(buildApplyTempDir(t), "private")
	if err := createBuildApplyDirectory(base); err != nil {
		t.Fatal(err)
	}
	f := newBuildApplyRunFixture(t)
	f.plan.receipt.Target = filepath.Join(base, "target")
	f.plan.receipt.Phase, f.plan.receipt.PreparedIntent = buildapply.PhasePrepare, true
	paths, err := buildapply.DeriveOperationPaths(f.plan.receipt.Target, f.plan.receipt.OperationID)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{paths.Target, paths.Prepared} {
		if err := createBuildApplyDirectory(path); err != nil {
			t.Fatal(err)
		}
	}
	f.plan.receipt.Runtime.TargetIdentity, err = buildApplyDirectoryIdentity(paths.Target)
	if err != nil {
		t.Fatal(err)
	}
	f.plan.receipt.PreparedIdentity, err = buildApplyDirectoryIdentity(paths.Prepared)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(paths.Prepared, paths.Prepared+"-retained"); err != nil {
		t.Fatal(err)
	}
	if err := createBuildApplyDirectory(paths.Prepared); err != nil {
		t.Fatal(err)
	}
	sentinel := filepath.Join(paths.Prepared, "unrelated.dat")
	if err := os.WriteFile(sentinel, []byte("unrelated private data"), 0o600); err != nil {
		t.Fatal(err)
	}
	files := &nativeBuildApplyFiles{}
	f.plan.receipt.PreviousSHA256, err = files.Digest(context.Background(), paths.Target)
	if err != nil {
		t.Fatal(err)
	}
	if err := files.restoreIdentities(context.Background(), f.plan.receipt); err == nil {
		t.Fatal("substituted preparation restored as owned")
	}
	tx := &buildApplyTransaction{plan: f.plan, receipt: f.plan.receipt, store: f}
	if err := resumeBuildApplyPreparation(context.Background(), tx, nil, files); err == nil {
		t.Fatal("substituted incomplete preparation accepted")
	}
	data, err := os.ReadFile(sentinel)
	if err != nil || string(data) != "unrelated private data" {
		t.Fatalf("recovery deleted or changed substituted tree: %q, %v", data, err)
	}
}

func TestBuildApplyTransactionPreparationSyncsParentBeforeQuiesce(t *testing.T) {
	prepared := filepath.Join(buildApplyTempDir(t), "prepared")
	entries := []buildapply.InventoryEntry{{Path: "resources", Mode: os.ModeDir | 0o700}, {Path: "resources/payload", Mode: 0o600}}
	want := []string{filepath.Join(prepared, "resources"), prepared, filepath.Dir(prepared)}
	var calls []string
	if err := syncBuildApplyPreparation(prepared, entries, func(path string) error { calls = append(calls, path); return nil }); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("preparation sync order = %v, want %v", calls, want)
	}
	failed := errors.New("parent sync failed")
	if err := syncBuildApplyPreparation(prepared, entries, func(path string) error {
		if path == filepath.Dir(prepared) {
			return failed
		}
		return nil
	}); !errors.Is(err, failed) {
		t.Fatalf("preparation ignored parent publication failure: %v", err)
	}
}

func TestBuildApplyRecoveryPinsDirectoryIdentityBeforeSubstitution(t *testing.T) {
	base := filepath.Join(buildApplyTempDir(t), "private")
	if err := createBuildApplyDirectory(base); err != nil {
		t.Fatal(err)
	}
	f := newBuildApplyRunFixture(t)
	f.plan.receipt.Target = filepath.Join(base, "target")
	paths, err := buildapply.DeriveOperationPaths(f.plan.receipt.Target, f.plan.receipt.OperationID)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{paths.Target, paths.Prepared} {
		if err := createBuildApplyDirectory(path); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(path, "payload.dat"), []byte("identical bounded data"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	f.plan.receipt.Runtime.TargetIdentity, err = buildApplyDirectoryIdentity(paths.Target)
	if err != nil {
		t.Fatal(err)
	}
	f.plan.receipt.PreparedIdentity, err = buildApplyDirectoryIdentity(paths.Prepared)
	if err != nil {
		t.Fatal(err)
	}
	files := &nativeBuildApplyFiles{}
	if err := files.restoreIdentities(context.Background(), f.plan.receipt); err != nil {
		t.Fatal(err)
	}
	// Do not compare FileInfo before substitution: a Windows pathname-backed
	// Lstat witness can otherwise acquire the old file ID accidentally.
	if err := os.Rename(paths.Prepared, paths.Prepared+"-retained"); err != nil {
		t.Fatal(err)
	}
	if err := createBuildApplyDirectory(paths.Prepared); err != nil {
		t.Fatal(err)
	}
	sentinel := filepath.Join(paths.Prepared, "payload.dat")
	if err := os.WriteFile(sentinel, []byte("identical bounded data"), 0o600); err != nil {
		t.Fatal(err)
	}
	id, info, err := buildApplyDirectorySnapshot(paths.Prepared)
	if err != nil {
		t.Fatal(err)
	}
	if id == f.plan.receipt.PreparedIdentity || os.SameFile(files.identities[paths.Prepared], info) {
		t.Fatal("restored identity followed the replacement pathname")
	}
	if _, err := files.Digest(context.Background(), paths.Prepared); err == nil {
		t.Fatal("identical bytes hid a substituted directory")
	}
	if err := files.Rename(context.Background(), paths.Prepared, paths.Previous); err == nil {
		t.Fatal("rename accepted a substituted directory")
	}
	if err := removeBuildApplyTree(context.Background(), paths.Prepared, f.plan.receipt.PreparedIdentity); err == nil {
		t.Fatal("cleanup accepted a substituted directory")
	}
	data, err := os.ReadFile(sentinel)
	if err != nil || string(data) != "identical bounded data" {
		t.Fatalf("substituted directory changed: %q, %v", data, err)
	}
}

func TestBuildApplyRecoveryRunnerCopyModes(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix executable permissions are not enforced on Windows")
	}
	for _, complete := range []bool{false, true} {
		name := "partial bytes at private copy mode"
		if complete {
			name = "complete bytes at private copy mode"
		}
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			profile := buildApplyTempDir(t)
			t.Setenv("HOME", profile)
			t.Setenv("XDG_CONFIG_HOME", profile)
			j, err := buildapply.OpenJournal(true)
			if err != nil {
				t.Fatal(err)
			}
			defer j.Close()
			lock, err := j.TryLock()
			if err != nil {
				t.Fatal(err)
			}
			defer lock.Close()
			sourceBytes, err := os.ReadFile(f.plan.serviceLaunch.executable)
			if err != nil {
				t.Fatal(err)
			}
			digest := sha256.Sum256(sourceBytes)
			r := f.plan.receipt
			r.Runtime.RunnerSHA256 = hex.EncodeToString(digest[:])
			r.Runtime.TargetIdentity, err = buildApplyDirectoryIdentity(r.Target)
			if err != nil {
				t.Fatal(err)
			}
			files := &nativeBuildApplyFiles{}
			r.PreviousSHA256, err = files.Digest(context.Background(), r.Target)
			if err != nil {
				t.Fatal(err)
			}
			runner, err := j.CreateRunner(lock)
			if err != nil {
				t.Fatal(err)
			}
			copyBytes := sourceBytes
			if !complete {
				copyBytes = sourceBytes[:len(sourceBytes)/2]
			}
			if _, err := runner.Write(copyBytes); err != nil {
				t.Fatal(err)
			}
			if err := errors.Join(runner.Sync(), runner.Close()); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			path, err := ensureBuildApplyRunner(ctx, j, lock, r)
			if err != nil {
				t.Fatalf("recover unfinished runner: %v", err)
			}
			got, err := os.ReadFile(path)
			if err != nil || !bytes.Equal(got, sourceBytes) {
				t.Fatalf("recovered runner bytes differ: %v", err)
			}
			info, err := os.Stat(path)
			if err != nil || info.Mode().Perm() != 0o700 {
				t.Fatalf("recovered runner is not executable: %v, %v", info, err)
			}
		})
	}
}

func TestBuildApplyAdmissionSuccessorRequiresExactReservedAuthority(t *testing.T) {
	test := func(name string, change func(*buildapply.Receipt, *buildapply.RunnerReservation), wantError bool) {
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			r := f.plan.receipt
			r.Phase, r.Rename, r.Handoff, r.RunnerReady = buildapply.PhaseRestart, buildapply.RenameInstalled, buildapply.HandoffAcknowledged, true
			r.Runner = &buildapply.RunnerReservation{PID: 203, Identity: buildapply.ProcessIdentity{Executable: filepath.Join(filepath.Dir(r.Target), "runner"), Created: "789"}}
			self := buildapply.RunnerReservation{PID: 204, Identity: buildapply.ProcessIdentity{Executable: f.plan.serviceLaunch.executable, Created: "abc"}}
			reservation := self
			r.ServiceLaunch = buildapply.LaunchReservation{Intent: true, Token: strings.Repeat("2", 32), Process: &reservation}
			change(&r, &self)
			err := validateBuildApplySuccessor(r, self, strings.Repeat("2", 32), f.plan.serviceLaunch.args, f.plan.serviceLaunch.env, time.Now())
			if (err != nil) != wantError {
				t.Fatalf("successor authorization: %v", err)
			}
		})
	}
	test("exact reserved successor", func(*buildapply.Receipt, *buildapply.RunnerReservation) {}, false)
	test("same PID different lifetime", func(_ *buildapply.Receipt, self *buildapply.RunnerReservation) { self.Identity.Created = "def" }, true)
	test("wrong launch token", func(r *buildapply.Receipt, _ *buildapply.RunnerReservation) {
		r.ServiceLaunch.Token = strings.Repeat("3", 32)
	}, true)
	test("handoff not acknowledged", func(r *buildapply.Receipt, _ *buildapply.RunnerReservation) { r.RunnerReady = false }, true)
	test("expired operation", func(r *buildapply.Receipt, _ *buildapply.RunnerReservation) {
		r.Runtime.Deadline = time.Now().Add(-time.Second)
	}, true)
	test("failed operation", func(r *buildapply.Receipt, _ *buildapply.RunnerReservation) {
		outcome := buildapply.OutcomeUncertain
		r.Outcome = &outcome
	}, true)
	test("launch arguments changed", func(r *buildapply.Receipt, _ *buildapply.RunnerReservation) {
		r.Runtime.ServiceLaunch.Args = []string{"--log-level=debug"}
	}, true)
	test("environment changed", func(r *buildapply.Receipt, _ *buildapply.RunnerReservation) {
		r.Runtime.ServiceLaunch.EnvironmentSHA256 = strings.Repeat("f", 64)
	}, true)
}

type buildApplyWriteBuffer struct{ bytes.Buffer }

func (*buildApplyWriteBuffer) Close() error { return nil }

func TestBuildApplyTransactionExpiredQueuedOwnerCommandHasZeroTransmission(t *testing.T) {
	var written buildApplyWriteBuffer
	run := &brokerRun{proc: &brokerProc{stdin: &written}, exited: make(chan struct{})}
	run.wmu.Lock()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- run.writeMessageContext(ctx, &jsonrpc.Message{JSONRPC: "2.0", Method: "build:apply-prepare"})
	}()
	cancel()
	run.wmu.Unlock()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) || written.Len() != 0 {
			t.Fatalf("expired owner frame: %v, transmitted=%d", err, written.Len())
		}
	case <-time.After(time.Second):
		t.Fatal("expired owner writer did not return")
	}
}

func TestBuildApplyAdmissionProductionEvidenceRemainsUnavailable(t *testing.T) {
	s := newService(config{})
	if _, err := s.cfg.buildApplyOwners.Read(context.Background()); err == nil {
		t.Fatal("production owner read accepted an absent broker")
	}
}

func TestBuildApplyAdmissionRequiresEveryOwnerProjection(t *testing.T) {
	test := func(name string, change func(*buildApplyEvidence), wantComplete bool) {
		t.Run(name, func(t *testing.T) {
			f := newBuildApplyRunFixture(t)
			d := f.plan.receipt.Protected
			e := buildApplyEvidence{
				Facts:     buildApplyOwnerCompleteness{IntentProvenance: true, LockMap: true, EngineUniverse: true, ModelDataRoots: true, ScopedStopAuthority: true, DownloadCopyReservations: true, ManagerIdleRunID: true},
				Protected: d, ProtectedRoots: []string{}, OwnedWorkers: []buildapply.RunnerReservation{},
				Impact: buildapply.Impact{Complete: true, Fences: f.plan.receipt.Fences,
					Availability: buildapply.Known(nodepolicy.Available), PauseOrigin: buildapply.Known("none"), Engines: buildapply.Known([]buildapply.EngineFact{}),
					ActiveWork: buildapply.Known(0), ModelOperations: buildapply.Known([]buildapply.ModelOperation{}), SettingsDifferences: buildapply.Known([]buildapply.SettingsDifference{}),
					Maintenance:        buildapply.MaintenanceState{BuildApplyActive: buildapply.Known(false), ModelSettingsActive: buildapply.Known(false)},
					ExpectedSettings:   buildapply.ExpectedSettings{PolicyDigest: buildapply.Known(d.Policy), EngineSettingsDigest: buildapply.Known(d.EngineSettings), IntentDigest: buildapply.Known(d.Intent), LocksDigest: buildapply.Known(d.Locks), StopEngines: buildapply.Known(true), UnloadModels: buildapply.Known(true)},
					InterruptionPolicy: buildapply.InterruptionPolicy{OnActive: buildapply.Known(nodepolicy.FinishActive), DrainTimeoutSeconds: buildapply.Known(120)},
				},
			}
			change(&e)
			if got := completeBuildApplyEvidence(e); got != wantComplete {
				t.Fatalf("owner completeness=%v, want=%v", got, wantComplete)
			}
		})
	}
	test("all authoritative projections", func(*buildApplyEvidence) {}, true)
	test("intent provenance missing", func(e *buildApplyEvidence) { e.Facts.IntentProvenance = false }, false)
	test("lock map missing", func(e *buildApplyEvidence) { e.Facts.LockMap = false }, false)
	test("engine universe incomplete", func(e *buildApplyEvidence) { e.Facts.EngineUniverse = false }, false)
	test("model roots unknown", func(e *buildApplyEvidence) { e.Facts.ModelDataRoots = false }, false)
	test("stop ownership missing", func(e *buildApplyEvidence) { e.Facts.ScopedStopAuthority = false }, false)
	test("download reservations incomplete", func(e *buildApplyEvidence) { e.Facts.DownloadCopyReservations = false }, false)
	test("manager idle acknowledgement missing", func(e *buildApplyEvidence) { e.Facts.ManagerIdleRunID = false }, false)
	test("owned workers unknown", func(e *buildApplyEvidence) { e.OwnedWorkers = nil }, false)
	test("model settings owns mutation", func(e *buildApplyEvidence) { e.Impact.Maintenance.ModelSettingsActive = buildapply.Known(true) }, false)
	test("unsafe model copy active", func(e *buildApplyEvidence) {
		e.Impact.ModelOperations = buildapply.Known([]buildapply.ModelOperation{{Engine: "fixture", Kind: "copy", BlocksApply: true}})
	}, false)
	test("intent digest disagrees", func(e *buildApplyEvidence) {
		e.Impact.ExpectedSettings.IntentDigest = buildapply.Known(strings.Repeat("f", 64))
	}, false)
}
