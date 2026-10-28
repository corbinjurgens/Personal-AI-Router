// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"time"
	"unicode/utf8"

	"nvpair-shared/appdir"
	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

const (
	MaxReceiptBytes   = 2 << 20
	MaxInventoryBytes = 64 << 20
)

var (
	ErrInvalidJournal = errors.New("invalid or unavailable build-apply journal")
	ErrOperationBusy  = errors.New("build-apply operation is pending")
	ErrRecoveryNeeded = errors.New("explicit build-apply recovery is required")
)

// ProtectedDigests are computed by the trusted owners over canonical semantics.
// No policy, credentials, identity keys, intent map or settings backup is saved.
type ProtectedDigests struct {
	Policy             string `json:"policy"`
	EngineSettings     string `json:"engineSettings"`
	Intent             string `json:"intent"`
	Locks              string `json:"locks"`
	IdentityMembership string `json:"identityMembership"`
}

// JournalExpectations contains only the explicitly approved semantic differences.
type JournalExpectations struct {
	StopEngines         bool                `json:"stopEngines"`
	UnloadModels        bool                `json:"unloadModels"`
	RemainPaused        bool                `json:"remainPaused"`
	PreserveIntentLocks bool                `json:"preserveIntentLocks"`
	OnActive            nodepolicy.OnActive `json:"onActive"`
	DrainTimeoutSeconds int                 `json:"drainTimeoutSeconds"`
}

type RenameState string

const (
	RenameNone             RenameState = "none"
	RenameTargetIntent     RenameState = "target-to-previous"
	RenamePreviousRetained RenameState = "previous-retained"
	RenamePreparedIntent   RenameState = "prepared-to-target"
	RenameInstalled        RenameState = "installed"
)

type HandoffState string

const (
	HandoffNone         HandoffState = "none"
	HandoffIntent       HandoffState = "intent"
	HandoffReserved     HandoffState = "reserved"
	HandoffAcknowledged HandoffState = "acknowledged"
)

// RunnerReservation is evidence supplied by the process owner, never PID-only
// authority. The service must independently inspect this lifetime before saving
// it; the successor must prove it is that same lifetime before acknowledging.
// The journal and its OS lock do not perform process supervision or liveness tests.
type RunnerReservation struct {
	PID      int             `json:"pid"`
	Identity ProcessIdentity `json:"identity"`
}

// JournalPauseFence is the exact durable Pause accepted by the lifecycle owner.
// It is recorded after quiesce readback, before handoff. It never authorizes
// Resume, and a peer-origin or pre-existing Pause cannot become transaction-owned.
type JournalPauseFence struct {
	Revision     string                 `json:"revision"`
	Origin       nodepolicy.PauseOrigin `json:"origin"`
	Owned        bool                   `json:"owned"`
	PolicySHA256 string                 `json:"policySha256"`
}

// LaunchContext contains arguments and profile-routing environment from the
// trusted current processes. The digest fences that profile context. Credentials
// and incidental shell variables are inherited in memory and never backed up.
type LaunchContext struct {
	Args              []string `json:"args"`
	Environment       []string `json:"environment"`
	EnvironmentSHA256 string   `json:"environmentSha256"`
}

type LaunchReservation struct {
	Intent       bool               `json:"intent"`
	Token        string             `json:"token"`
	Process      *RunnerReservation `json:"process"`
	Acknowledged bool               `json:"acknowledged"`
}

type RuntimeAuthority struct {
	AcceptedAt              time.Time           `json:"acceptedAt"`
	Deadline                time.Time           `json:"deadline"`
	CandidatePath           string              `json:"candidatePath"`
	Endpoint                string              `json:"endpoint"`
	OldProcesses            []RunnerReservation `json:"oldProcesses"`
	OldGUI                  RunnerReservation   `json:"oldGui"`
	OldService              RunnerReservation   `json:"oldService"`
	ServiceLaunch           LaunchContext       `json:"serviceLaunch"`
	GUILaunch               LaunchContext       `json:"guiLaunch"`
	ExpectedService         Identity            `json:"expectedService"`
	ExpectedGUI             Identity            `json:"expectedGui"`
	RunnerSHA256            string              `json:"runnerSha256"`
	TargetIdentity          string              `json:"targetIdentity"`
	CandidateMetadataSHA256 string              `json:"candidateMetadataSha256"`
	WasPaused               bool                `json:"wasPaused"`
}

// Receipt is local transaction authority, not a renderer request or a backup.
// Failed/interrupted/uncertain outcomes stay Current, retaining the startup gate.
// Only succeeded/refused outcomes move atomically to LastTerminal. PreviewID is
// authoritative and must match the public operation's PreviewID.
type Receipt struct {
	OperationID       string               `json:"operationId"`
	PreviewID         string               `json:"previewId"`
	Target            string               `json:"target"`
	CandidateSHA256   string               `json:"candidateSha256"`
	PreviousSHA256    string               `json:"previousSha256"`
	InventorySHA256   string               `json:"inventorySha256"`
	Protected         ProtectedDigests     `json:"protected"`
	Expected          JournalExpectations  `json:"expected"`
	Consent           Consent              `json:"consent"`
	Fences            Fences               `json:"fences"`
	Pause             *JournalPauseFence   `json:"pause"`
	Phase             Phase                `json:"phase"`
	Rename            RenameState          `json:"rename"`
	Handoff           HandoffState         `json:"handoff"`
	Runner            *RunnerReservation   `json:"runner"`
	RunnerToken       string               `json:"runnerToken"`
	Outcome           *Outcome             `json:"outcome"`
	Runtime           RuntimeAuthority     `json:"runtime"`
	PreparedIdentity  string               `json:"preparedIdentity"`
	PreparedIntent    bool                 `json:"preparedIntent"`
	ServiceLaunch     LaunchReservation    `json:"serviceLaunch"`
	GUILaunch         LaunchReservation    `json:"guiLaunch"`
	Verified          *BuildApplyOperation `json:"verified"`
	Progress          *BuildApplyOperation `json:"progress"`
	Failure           *BuildApplyFailure   `json:"failure"`
	CleanupPending    bool                 `json:"cleanupPending"`
	CleanupStarted    bool                 `json:"cleanupStarted"`
	ReleasePending    bool                 `json:"releasePending"`
	CleanupRunner     *RunnerReservation   `json:"cleanupRunner"`
	CleanupDeadline   time.Time            `json:"cleanupDeadline"`
	RunnerReady       bool                 `json:"runnerReady"`
	RecoveryStartedAt time.Time            `json:"recoveryStartedAt"`
	RecoveryDeadline  time.Time            `json:"recoveryDeadline"`
	RecoveryCaller    *RunnerReservation   `json:"recoveryCaller"`
	ReadbackPause     *JournalPauseFence   `json:"readbackPause"`
}

// ValidateOperation fences a service-owned status/start projection to this
// durable receipt. Runtime observations remain the service owner's responsibility;
// a receipt's declared expectations alone cannot establish enforced readback.
func (r Receipt) ValidateOperation(operation BuildApplyOperation) error {
	if operation.OperationID != r.OperationID || operation.PreviewID != r.PreviewID || operation.Phase != r.Phase || !reflect.DeepEqual(operation.Outcome, r.Outcome) {
		return errors.New("public operation does not match its durable receipt")
	}
	return operation.validate()
}

// JournalState is one atomic file, so terminal publication cannot leave a
// missing current receipt between separate archive/remove writes. A zero state
// means idle; any read/validation error means unknown and must refuse startup.
type JournalState struct {
	Current      *Receipt `json:"current"`
	LastTerminal *Receipt `json:"lastTerminal"`
	// A durable admission/terminal intent survives uncertain publication. Its
	// presence gates startup independently of which operation.json bytes landed.
	TerminalPending bool `json:"-"`
}

type InventoryEntry struct {
	Path   string      `json:"path"`
	Size   int64       `json:"size"`
	Mode   os.FileMode `json:"mode"`
	SHA256 string      `json:"sha256"`
	Link   string      `json:"link"`
}

type Inventory struct {
	OperationID   string           `json:"operationId"`
	Platform      string           `json:"platform"`
	Architecture  string           `json:"architecture"`
	PackageSHA256 string           `json:"packageSha256"`
	Entries       []InventoryEntry `json:"entries"`
}

// InventoryFromPackage snapshots measured metadata, not candidate declarations.
// The caller must still freeze and reverify the copied bytes before quiescing.
func InventoryFromPackage(id string, p *Package) (Inventory, error) {
	if p == nil {
		return Inventory{}, errors.New("missing measured package")
	}
	v := Inventory{OperationID: id, Platform: p.Platform, Architecture: p.Architecture, PackageSHA256: p.SHA256}
	for _, e := range p.Entries {
		v.Entries = append(v.Entries, InventoryEntry{Path: e.Path, Size: e.Size, Mode: e.Mode, SHA256: e.SHA256, Link: e.Link})
	}
	return v, v.validate()
}

func NewOperationID() (string, error) {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(id[:]), nil
}

func ValidOperationID(id string) bool { return lowerHex(id, 32) }

func lowerHex(value string, size int) bool {
	if len(value) != size {
		return false
	}
	for _, c := range value {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

type OperationPaths struct {
	Target   string
	Prepared string
	Previous string
}

// DeriveOperationPaths is also the cleanup boundary: callers must derive again
// and validate actual held directory identities before renaming/removing anything.
// This function does not require the target to exist during a half-completed swap.
func DeriveOperationPaths(target, id string) (OperationPaths, error) {
	if !ValidOperationID(id) || localRootSyntax(target) != nil || filepath.Clean(target) != target || filepath.Dir(target) == target {
		return OperationPaths{}, errors.New("invalid operation target or identity")
	}
	base := filepath.Base(target)
	if !validRelative(base) {
		return OperationPaths{}, errors.New("invalid target basename")
	}
	p := OperationPaths{Target: target, Prepared: target + "." + id + ".prepared", Previous: target + "." + id + ".previous"}
	if localRootSyntax(p.Prepared) != nil || localRootSyntax(p.Previous) != nil {
		return OperationPaths{}, errors.New("operation sibling path exceeds bounds")
	}
	return p, nil
}

// Journal holds the canonical private root, never a candidate-selected path.
// OpenJournal(false) and Read do not create state. ErrNotExist at open is idle;
// all other errors are unknown. Mutation requires this journal's live OS lock.
type Journal struct {
	root      *RealRoot
	replacer  durablefile.Replacer
	uncertain bool
	recovery  string
}

func OpenJournal(create bool) (*Journal, error) {
	name, err := appdir.Path("build-apply")
	if err != nil {
		return nil, err
	}
	return openJournalAt(name, create)
}

// ReadJournal is the no-start status primitive. Only genuinely absent canonical
// state is idle. A corrupt, oversized, linked, substituted or unreadable journal
// returns an error, never a fabricated empty state.
func ReadJournal() (JournalState, error) {
	j, err := OpenJournal(false)
	if os.IsNotExist(err) {
		return JournalState{}, nil
	}
	if err != nil {
		return JournalState{}, err
	}
	defer j.Close()
	return j.Read()
}

func openJournalAt(name string, create bool) (*Journal, error) {
	if err := localRootSyntax(name); err != nil {
		return nil, err
	}
	name = filepath.Clean(name)
	root, err := OpenRealRoot(name)
	if os.IsNotExist(err) && create {
		parentName := filepath.Dir(name)
		var missing []string
		for {
			parent, openErr := OpenRealRoot(parentName)
			if openErr == nil {
				for i := len(missing) - 1; i >= 0; i-- {
					next, makeErr := privateChild(parent, missing[i], true)
					parent.Close()
					if makeErr != nil {
						return nil, makeErr
					}
					parent = next
				}
				root, err = privateChild(parent, filepath.Base(name), true)
				parent.Close()
				break
			}
			if !os.IsNotExist(openErr) || filepath.Dir(parentName) == parentName {
				return nil, openErr
			}
			missing = append(missing, filepath.Base(parentName))
			parentName = filepath.Dir(parentName)
		}
	}
	if err != nil {
		return nil, err
	}
	if err := validatePrivateRoot(root); err != nil {
		root.Close()
		return nil, err
	}
	j := &Journal{root: root}
	j.replacer = durablefile.Replacer{
		CreateTemp: j.createTemp,
		Replace: func(tmp, target string) error {
			return nativeJournalReplace(root, filepath.Base(tmp), filepath.Base(target))
		},
		SyncDir:     func(string) error { return nativeJournalSync(root) },
		IsTransient: func(error) bool { return false },
	}
	return j, nil
}

func (j *Journal) Close() error { return j.root.Close() }
func (j *Journal) Path() string { return j.root.Path }

func (j *Journal) TryLock() (*OperationLock, error) { return TryOperationLock(j.root) }

// RunnerPath derives the sole allowed copied runner. Its directory is created
// privately on demand, only after admission; no executable is copied or launched.
func (j *Journal) RunnerPath(lock *OperationLock) (string, error) {
	if err := j.checkLock(lock); err != nil {
		return "", err
	}
	runner, err := privateChild(j.root, "runner", true)
	if err != nil {
		return "", err
	}
	if err := runner.Close(); err != nil {
		return "", err
	}
	return j.runnerPath(), nil
}

// CreateRunner exclusively creates an empty private runner for the service's
// trusted copy owner. The caller owns copy/hash, Sync, Close, executable mode and
// process identity checks. An existing runner is evidence, never overwritten.
func (j *Journal) CreateRunner(lock *OperationLock) (*os.File, error) {
	if _, err := j.RunnerPath(lock); err != nil {
		return nil, err
	}
	runner, err := privateChild(j.root, "runner", false)
	if err != nil {
		return nil, err
	}
	defer runner.Close()
	f, err := runner.root.OpenFile(filepath.Base(j.runnerPath()), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	if err := nativePrivateCreatedFile(f); err != nil {
		f.Close()
		return nil, err
	}
	if err := validatePrivateRoot(runner); err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}

func (j *Journal) runnerPath() string {
	name := "nvpair-service"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(j.root.Path, "runner", name)
}

func privateChild(parent *RealRoot, name string, create bool) (*RealRoot, error) {
	if !validRelative(name) || strings.Contains(name, "/") {
		return nil, errors.New("invalid private directory name")
	}
	if err := parent.Revalidate(); err != nil {
		return nil, err
	}
	if _, err := parent.root.Lstat(name); os.IsNotExist(err) && create {
		if err := nativePrivateMkdir(parent, name); err != nil && !os.IsExist(err) {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}
	child, err := OpenRealRoot(filepath.Join(parent.Path, name))
	if err != nil {
		return nil, err
	}
	if err := validatePrivateRoot(child); err != nil {
		child.Close()
		return nil, err
	}
	if err := parent.Revalidate(); err != nil {
		child.Close()
		return nil, err
	}
	named, err := parent.root.Lstat(name)
	if err != nil || !os.SameFile(named, child.info) {
		child.Close()
		return nil, errors.New("private child directory was substituted")
	}
	return child, nil
}

func validatePrivateRoot(root *RealRoot) error {
	if err := root.Revalidate(); err != nil {
		return err
	}
	f, err := root.root.Open(".")
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.IsDir() || !os.SameFile(root.info, info) {
		return errors.New("private root was substituted")
	}
	return nativePrivateCheck(f, info)
}

func (j *Journal) checkLock(lock *OperationLock) error {
	if lock == nil || lock.file == nil || lock.root != j.root {
		return errors.New("journal mutation requires its held operation lock")
	}
	return validateLockFile(j.root, lock.file)
}

func (j *Journal) Read() (JournalState, error) {
	if err := validatePrivateRoot(j.root); err != nil {
		return JournalState{}, fmt.Errorf("%w: %w", ErrInvalidJournal, err)
	}
	admission, admissionErr := j.readAdmissionIntent()
	if admissionErr != nil && !os.IsNotExist(admissionErr) {
		return JournalState{}, fmt.Errorf("%w: %w", ErrInvalidJournal, admissionErr)
	}
	data, err := j.readFile("operation.json", MaxReceiptBytes)
	if os.IsNotExist(err) {
		if admissionErr == nil {
			return j.admissionState(JournalState{}, admission)
		}
		if _, runnerErr := j.root.root.Lstat("runner"); !os.IsNotExist(runnerErr) {
			return JournalState{}, fmt.Errorf("%w: runner without receipt", ErrInvalidJournal)
		}
		if _, terminalErr := j.root.root.Lstat("terminal.json"); !os.IsNotExist(terminalErr) {
			return JournalState{}, fmt.Errorf("%w: terminal intent without receipt", ErrInvalidJournal)
		}
		if _, invErr := j.root.root.Lstat("inventory.json"); !os.IsNotExist(invErr) {
			return JournalState{}, fmt.Errorf("%w: inventory without receipt", ErrInvalidJournal)
		}
		return JournalState{}, nil
	}
	if err != nil {
		return JournalState{}, fmt.Errorf("%w: %w", ErrInvalidJournal, err)
	}
	var state JournalState
	if err := strictJournalJSON(data, &state); err != nil {
		return JournalState{}, fmt.Errorf("%w: %w", ErrInvalidJournal, err)
	}
	if state.Current == nil && state.LastTerminal == nil {
		return JournalState{}, fmt.Errorf("%w: empty receipt", ErrInvalidJournal)
	}
	for _, receipt := range []*Receipt{state.Current, state.LastTerminal} {
		if receipt != nil {
			if err := j.validateReceipt(*receipt); err != nil {
				return JournalState{}, fmt.Errorf("%w: %w", ErrInvalidJournal, err)
			}
		}
	}
	if state.Current != nil && closedReceipt(*state.Current) || state.LastTerminal != nil && !closedReceipt(*state.LastTerminal) ||
		state.Current != nil && state.LastTerminal != nil && state.Current.OperationID == state.LastTerminal.OperationID {
		return JournalState{}, fmt.Errorf("%w: inconsistent current/terminal receipt", ErrInvalidJournal)
	}
	if admissionErr == nil {
		return j.admissionState(state, admission)
	}
	if state.Current != nil {
		if _, err := j.ReadInventory(*state.Current); err != nil {
			return JournalState{}, err
		}
	}
	if intent, intentErr := j.readTerminalIntent(); intentErr == nil {
		current := state.Current
		if current == nil {
			current = state.LastTerminal
		}
		if current == nil || current.OperationID != intent.OperationID || current.PreviewID != intent.PreviewID {
			return JournalState{}, fmt.Errorf("%w: terminal intent mismatch", ErrInvalidJournal)
		}
		state.TerminalPending = true
		// The durable intent is an unresolved current operation even if the
		// final bytes landed. Keep terminal evidence separately, without exposing
		// an uncertain publication as a successful current outcome.
		uncertain := OutcomeUncertain
		intent.Outcome = &uncertain
		intent.Verified = nil
		intent.Progress = nil
		intent.Failure = &BuildApplyFailure{Phase: intent.Phase, Reason: "terminal_publication_uncertain", Detail: "The terminal publication did not release its durable gate.", ReadbackConfidence: ConfidenceUnknown, SafeNextAction: "Explicitly recover this operation while the node remains paused."}
		state.Current = &intent
	} else if !os.IsNotExist(intentErr) {
		return JournalState{}, fmt.Errorf("%w: %w", ErrInvalidJournal, intentErr)
	}
	return state, nil
}

func (j *Journal) readAdmissionIntent() (Receipt, error) {
	data, err := j.readFile("admission.json", MaxReceiptBytes)
	if err != nil {
		return Receipt{}, err
	}
	var r Receipt
	if err := strictJournalJSON(data, &r); err != nil {
		return r, err
	}
	if err := j.validateReceipt(r); err != nil {
		return r, err
	}
	if !initialReceipt(r) {
		return r, errors.New("admission intent already contains lifecycle work")
	}
	return r, nil
}

func (j *Journal) admissionState(state JournalState, intent Receipt) (JournalState, error) {
	if _, err := j.root.root.Lstat("terminal.json"); !os.IsNotExist(err) {
		return JournalState{}, fmt.Errorf("%w: overlapping publication intents", ErrInvalidJournal)
	}
	if state.Current != nil && !reflect.DeepEqual(*state.Current, intent) {
		return JournalState{}, fmt.Errorf("%w: admission differs from current authority", ErrInvalidJournal)
	}
	if last := state.LastTerminal; last != nil && last.OperationID == intent.OperationID {
		refused := intent
		outcome := OutcomeRefused
		refused.Outcome = &outcome
		if !reflect.DeepEqual(*last, refused) {
			return JournalState{}, fmt.Errorf("%w: admission differs from terminal authority", ErrInvalidJournal)
		}
	}
	// Begin cannot return or authorize preparation until this intent is gone.
	// Inventory may still be absent or belong to the last terminal operation.
	outcome := OutcomeUncertain
	intent.Outcome = &outcome
	intent.Failure = &BuildApplyFailure{Phase: PhasePrepare, Reason: "admission_publication_uncertain", Detail: "Admission publication stopped before any lifecycle work was authorized.", ReadbackConfidence: ConfidenceUnknown, SafeNextAction: "Explicitly recover this operation to record refusal, then preview again."}
	state.Current, state.TerminalPending = &intent, true
	return state, nil
}

func (j *Journal) clearAdmissionIntent() error {
	if _, err := j.readAdmissionIntent(); err != nil {
		return err
	}
	if err := j.root.root.Remove("admission.json"); err != nil {
		return err
	}
	return nativeJournalSync(j.root)
}

func (j *Journal) readTerminalIntent() (Receipt, error) {
	data, err := j.readFile("terminal.json", MaxReceiptBytes)
	if err != nil {
		return Receipt{}, err
	}
	var r Receipt
	if err := strictJournalJSON(data, &r); err != nil {
		return r, err
	}
	if err := j.validateReceipt(r); err != nil {
		return r, err
	}
	if !closedReceipt(r) {
		return r, errors.New("nonterminal publication intent")
	}
	return r, nil
}

func (j *Journal) clearTerminalIntent() error {
	if _, err := j.readTerminalIntent(); err != nil {
		return err
	}
	if err := j.root.root.Remove("terminal.json"); err != nil {
		return err
	}
	return nativeJournalSync(j.root)
}

func (j *Journal) ReadInventory(receipt Receipt) (Inventory, error) {
	if err := j.validateReceipt(receipt); err != nil {
		return Inventory{}, err
	}
	data, err := j.readFile("inventory.json", MaxInventoryBytes)
	if err != nil {
		return Inventory{}, fmt.Errorf("%w: %w", ErrInvalidJournal, err)
	}
	hash := sha256.Sum256(data)
	if hex.EncodeToString(hash[:]) != receipt.InventorySHA256 {
		return Inventory{}, fmt.Errorf("%w: inventory digest changed", ErrInvalidJournal)
	}
	var inventory Inventory
	if err := strictJournalJSON(data, &inventory); err != nil {
		return Inventory{}, fmt.Errorf("%w: %w", ErrInvalidJournal, err)
	}
	if err := inventory.validate(); err != nil {
		return Inventory{}, fmt.Errorf("%w: %w", ErrInvalidJournal, err)
	}
	if inventory.OperationID != receipt.OperationID || inventory.PackageSHA256 != receipt.CandidateSHA256 {
		return Inventory{}, fmt.Errorf("%w: inventory belongs to different bytes/operation", ErrInvalidJournal)
	}
	return inventory, nil
}

// Begin records admission authority before inventory/receipt publication. Until
// that intent is cleared, no lifecycle is authorized and exact-ID recovery may
// only record refusal. A crash cannot strand an authority-free inventory.
func (j *Journal) Begin(lock *OperationLock, receipt Receipt, inventory Inventory) (JournalState, error) {
	if err := j.checkLock(lock); err != nil {
		return JournalState{}, err
	}
	if j.uncertain {
		return JournalState{}, ErrRecoveryNeeded
	}
	state, err := j.Read()
	if err != nil {
		return JournalState{}, err
	}
	if state.Current != nil || state.TerminalPending {
		return state, ErrOperationBusy
	}
	if state.LastTerminal != nil && (state.LastTerminal.OperationID == receipt.OperationID || state.LastTerminal.PreviewID == receipt.PreviewID) {
		return state, errors.New("operation/preview identity already used")
	}
	if err := inventory.validate(); err != nil {
		return state, err
	}
	data, err := boundedJSON(inventory, MaxInventoryBytes)
	if err != nil {
		return state, err
	}
	hash := sha256.Sum256(data)
	receipt.InventorySHA256 = hex.EncodeToString(hash[:])
	if inventory.OperationID != receipt.OperationID || inventory.PackageSHA256 != receipt.CandidateSHA256 || !initialReceipt(receipt) {
		return state, errors.New("invalid initial operation")
	}
	if err := j.validateReceipt(receipt); err != nil {
		return state, err
	}
	if _, err := DeriveOperationPaths(receipt.Target, receipt.OperationID); err != nil {
		return state, err
	}
	if containsPath(receipt.Target, j.root.Path) || containsPath(j.root.Path, receipt.Target) {
		return state, errors.New("operation root overlaps target")
	}
	// Reject an oversized combined current/terminal state before replacing the
	// inventory belonging to the previous terminal operation.
	proposed := state
	proposed.Current = &receipt
	if _, err := boundedJSON(proposed, MaxReceiptBytes); err != nil {
		return state, err
	}
	admission, err := boundedJSON(receipt, MaxReceiptBytes)
	if err != nil {
		return state, err
	}
	if err := j.writeFile("admission.json", admission); err != nil {
		j.uncertain = true
		return state, err
	}
	if err := j.writeFile("inventory.json", data); err != nil {
		j.uncertain = true
		return state, err
	}
	state.Current = &receipt
	state, err = j.publish(state)
	if err != nil {
		return state, err
	}
	if err := j.clearAdmissionIntent(); err != nil {
		j.uncertain = true
		landed, readErr := j.Read()
		return landed, errors.Join(ErrRecoveryNeeded, err, readErr)
	}
	return state, nil
}

// Save records a phase/rename/handoff boundary under exclusive ownership.
// Irreversible work is authorized only after a nil publication error. On
// ErrReplaced the returned state adopts the landed bytes but is uncertain;
// callers retain the gate and use explicit recovery, never assume no change.
func (j *Journal) Save(lock *OperationLock, receipt Receipt) (JournalState, error) {
	if err := j.checkLock(lock); err != nil {
		return JournalState{}, err
	}
	if j.uncertain {
		return JournalState{}, ErrRecoveryNeeded
	}
	state, err := j.Read()
	if err != nil {
		return JournalState{}, err
	}
	if state.Current == nil || state.Current.OperationID != receipt.OperationID {
		return state, errors.New("operation is not current")
	}
	if state.TerminalPending && j.recovery != receipt.OperationID {
		return state, ErrRecoveryNeeded
	}
	if err := j.validateReceipt(receipt); err != nil {
		return state, err
	}
	if err := receiptTransition(*state.Current, receipt, j.recovery == receipt.OperationID); err != nil {
		return state, err
	}
	if _, err := j.ReadInventory(receipt); err != nil {
		return state, err
	}
	if closedReceipt(receipt) {
		intent, encodeErr := boundedJSON(receipt, MaxReceiptBytes)
		if encodeErr != nil {
			return state, encodeErr
		}
		if err := j.writeFile("terminal.json", intent); err != nil {
			j.uncertain = true
			landed, readErr := j.Read()
			return landed, errors.Join(err, readErr)
		}
		state.Current = nil
		state.LastTerminal = &receipt
	} else {
		state.Current = &receipt
	}
	state, err = j.publish(state)
	if err == nil && closedReceipt(receipt) {
		// Successful terminal publication is not reversed by cleanup failure.
		// A retained intent keeps startup gated and status unknown until explicit
		// recovery rechecks the terminal facts; the success evidence is retained.
		if cleanupErr := j.clearTerminalIntent(); cleanupErr != nil {
			j.uncertain = true
			landed, readErr := j.Read()
			return landed, errors.Join(ErrRecoveryNeeded, cleanupErr, readErr)
		}
	}
	if err == nil && receipt.Outcome != nil {
		j.recovery = ""
	}
	return state, err
}

// Recover revalidates and republishes exactly the recorded operation. It does
// not infer process death, perform renames, clear outcomes or choose binaries.
// The recovery owner must reconcile actual directory/process identities before
// Save advances any boundary. Recovery may repeat a failed phase, never rewind.
func (j *Journal) Recover(lock *OperationLock, id string) (JournalState, error) {
	if err := j.checkLock(lock); err != nil {
		return JournalState{}, err
	}
	state, err := j.Read()
	if err != nil {
		return JournalState{}, err
	}
	receipt := state.Current
	if receipt == nil {
		receipt = state.LastTerminal
	}
	if !ValidOperationID(id) || receipt == nil || receipt.OperationID != id {
		return state, errors.New("recovery requires the exact recorded operation")
	}
	if admission, admissionErr := j.readAdmissionIntent(); admissionErr == nil {
		if admission.OperationID != id {
			return state, ErrRecoveryNeeded
		}
		outcome := OutcomeRefused
		admission.Outcome = &outcome
		state.Current, state.LastTerminal, state.TerminalPending = nil, &admission, false
		state, err = j.publish(state)
		if err != nil {
			return state, err
		}
		if err := j.clearAdmissionIntent(); err != nil {
			return state, err
		}
		j.uncertain, j.recovery = false, ""
		return state, nil
	} else if !os.IsNotExist(admissionErr) {
		return state, admissionErr
	}
	if _, err := j.ReadInventory(*receipt); err != nil {
		return state, err
	}
	if state.TerminalPending {
		intent, intentErr := j.readTerminalIntent()
		if intentErr != nil {
			return state, intentErr
		}
		if intent.Outcome != nil && *intent.Outcome == OutcomeRefused {
			state.Current = nil
			state.LastTerminal = &intent
			state.TerminalPending = false
			state, err = j.publish(state)
			if err != nil {
				return state, err
			}
			if err := j.clearTerminalIntent(); err != nil {
				return state, err
			}
			j.uncertain = false
			j.recovery = id
			return state, nil
		}
		uncertain := OutcomeUncertain
		intent.Outcome = &uncertain
		intent.Verified = nil
		intent.Progress = nil
		intent.Failure = nil
		state.Current = &intent
		if state.LastTerminal != nil && state.LastTerminal.OperationID == id {
			state.LastTerminal = nil
		}
		state.TerminalPending = false
		state, err = j.publish(state)
		if err != nil {
			return state, err
		}
		if err := j.clearTerminalIntent(); err != nil {
			return state, err
		}
	}
	state, err = j.publish(state)
	if err == nil {
		j.uncertain = false
		j.recovery = id
	}
	return state, err
}

func (j *Journal) publish(state JournalState) (JournalState, error) {
	data, err := boundedJSON(state, MaxReceiptBytes)
	if err != nil {
		return JournalState{}, err
	}
	err = j.writeFile("operation.json", data)
	if err == nil || errors.Is(err, durablefile.ErrReplaced) {
		if err != nil {
			j.uncertain = true
			landed, readErr := j.Read()
			return landed, errors.Join(err, readErr)
		}
		return state, err
	}
	// Do not return the proposed state as if it had landed.
	old, readErr := j.Read()
	if readErr != nil {
		return JournalState{}, errors.Join(err, readErr)
	}
	return old, err
}

func closedReceipt(r Receipt) bool {
	return r.Outcome != nil && (*r.Outcome == OutcomeSucceeded || *r.Outcome == OutcomeRefused)
}

func initialReceipt(r Receipt) bool {
	return r.Phase == PhasePrepare && r.Rename == RenameNone && r.Handoff == HandoffNone && r.Outcome == nil && r.RunnerToken == r.OperationID &&
		!r.PreparedIntent && r.PreparedIdentity == "" && r.Runner == nil && !r.RunnerReady && r.Pause == nil && r.ReadbackPause == nil && r.Progress == nil && r.Verified == nil && r.Failure == nil &&
		!r.ServiceLaunch.Intent && !r.GUILaunch.Intent && !r.CleanupPending && !r.CleanupStarted && !r.ReleasePending && r.CleanupRunner == nil && r.CleanupDeadline.IsZero() &&
		r.RecoveryStartedAt.IsZero() && r.RecoveryDeadline.IsZero() && r.RecoveryCaller == nil
}

func phaseIndex(phase Phase) int {
	for i, value := range []Phase{PhasePrepare, PhaseQuiesce, PhaseShutdown, PhaseSwap, PhaseRestart, PhaseVerifyBuild, PhaseVerifySettings, PhaseFinish} {
		if value == phase {
			return i
		}
	}
	return -1
}

func renameIndex(value RenameState) int {
	for i, state := range []RenameState{RenameNone, RenameTargetIntent, RenamePreviousRetained, RenamePreparedIntent, RenameInstalled} {
		if value == state {
			return i
		}
	}
	return -1
}

func handoffIndex(value HandoffState) int {
	for i, state := range []HandoffState{HandoffNone, HandoffIntent, HandoffReserved, HandoffAcknowledged} {
		if value == state {
			return i
		}
	}
	return -1
}

func (j *Journal) validateReceipt(r Receipt) error {
	if !ValidOperationID(r.OperationID) || !ValidOperationID(r.RunnerToken) || !token(r.PreviewID, false) || phaseIndex(r.Phase) < 0 || renameIndex(r.Rename) < 0 || handoffIndex(r.Handoff) < 0 {
		return errors.New("invalid receipt identity or boundary")
	}
	if _, err := DeriveOperationPaths(r.Target, r.OperationID); err != nil {
		return err
	}
	if err := ValidateRuntimeAuthority(r.Target, r.Runtime); err != nil {
		return err
	}
	if r.PreparedIdentity != "" && !validCreation(r.PreparedIdentity) {
		return errors.New("invalid prepared directory identity")
	}
	layout, _ := LayoutFor(runtime.GOOS)
	for _, slot := range []struct {
		state LaunchReservation
		path  string
	}{{r.ServiceLaunch, filepath.Join(r.Target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable)}, {r.GUILaunch, filepath.Join(r.Target, filepath.FromSlash(layout.Executable))}} {
		if slot.state.Intent != ValidOperationID(slot.state.Token) {
			return errors.New("launch intent lacks its exact correlation token")
		}
		if slot.state.Process != nil && !slot.state.Intent || slot.state.Acknowledged && slot.state.Process == nil {
			return errors.New("launch lacks durable intent/reservation")
		}
		if slot.state.Process != nil && (!validReservation(*slot.state.Process) || slot.state.Process.Identity.Executable != slot.path) {
			return errors.New("invalid successor process reservation")
		}
		if slot.state.Intent && phaseIndex(r.Phase) < phaseIndex(PhaseRestart) {
			return errors.New("successor launch precedes restart intent")
		}
	}
	if r.Verified != nil {
		if err := r.ValidateOperation(*r.Verified); err != nil {
			return err
		}
	}
	if r.Progress != nil {
		if err := r.ValidateOperation(*r.Progress); err != nil {
			return err
		}
	}
	if r.Failure != nil && (!validFailure(*r.Failure) || r.Failure.Phase != r.Phase || r.Outcome == nil || *r.Outcome == OutcomeSucceeded) {
		return errors.New("failure does not match the durable operation")
	}
	if r.RecoveryStartedAt.IsZero() != r.RecoveryDeadline.IsZero() || !r.RecoveryStartedAt.IsZero() && r.RecoveryDeadline.Sub(r.RecoveryStartedAt) != 30*time.Minute {
		return errors.New("invalid explicit recovery budget")
	}
	if r.RecoveryCaller != nil && !validReservation(*r.RecoveryCaller) {
		return errors.New("invalid recovery caller lifetime")
	}
	if r.CleanupRunner != nil && (!validReservation(*r.CleanupRunner) || r.CleanupDeadline.IsZero() || r.Outcome == nil || *r.Outcome != OutcomeSucceeded) {
		return errors.New("invalid terminal cleanup lease")
	}
	if r.RunnerReady && (r.Runner == nil || r.Handoff != HandoffAcknowledged && r.RecoveryStartedAt.IsZero()) {
		return errors.New("runner acknowledgement lacks reservation")
	}
	if r.ReadbackPause != nil && (r.Pause == nil || r.ReadbackPause.Origin != r.Pause.Origin || r.ReadbackPause.Owned != r.Pause.Owned || r.ReadbackPause.PolicySHA256 != r.Pause.PolicySHA256 || !token(r.ReadbackPause.Revision, false)) {
		return errors.New("readback cannot replace the accepted Pause semantics/provenance")
	}
	for _, digest := range []string{r.CandidateSHA256, r.PreviousSHA256, r.InventorySHA256, r.Protected.Policy, r.Protected.EngineSettings, r.Protected.Intent, r.Protected.Locks, r.Protected.IdentityMembership} {
		if !lowerHex(digest, 64) {
			return errors.New("missing/invalid protected digest")
		}
	}
	if containsPath(r.Target, j.root.Path) || containsPath(j.root.Path, r.Target) {
		return errors.New("operation root overlaps target")
	}
	if containsPath(r.Runtime.CandidatePath, j.root.Path) || containsPath(j.root.Path, r.Runtime.CandidatePath) {
		return errors.New("operation root overlaps candidate")
	}
	if !knownFences(r.Fences) || !r.Consent.AllowEngineStop || !r.Consent.AllowInterruption || !r.Expected.StopEngines || !r.Expected.UnloadModels || !r.Expected.RemainPaused || !r.Expected.PreserveIntentLocks ||
		(r.Expected.OnActive != nodepolicy.FinishActive && r.Expected.OnActive != nodepolicy.CancelActive) || r.Expected.DrainTimeoutSeconds < 0 || r.Expected.DrainTimeoutSeconds > 1800 {
		return errors.New("incomplete or unapproved receipt expectations/fences")
	}
	for _, fence := range []Observation[string]{
		r.Fences.RuntimeGeneration, r.Fences.ServiceInstance, r.Fences.BrokerGeneration, r.Fences.ManagerIncarnation,
		r.Fences.PolicyRevision, r.Fences.SettingsRevision, r.Fences.IntentRevision, r.Fences.LocksRevision, r.Fences.AvailabilityRevision,
	} {
		if !utf8.ValidString(*fence.Value) {
			return errors.New("invalid UTF-8 in an owner fence")
		}
	}
	if (r.Runner != nil) != (handoffIndex(r.Handoff) >= handoffIndex(HandoffReserved)) && !(r.Runner != nil && !r.RecoveryStartedAt.IsZero()) {
		return errors.New("inconsistent runner reservation")
	}
	if r.Handoff != HandoffNone && phaseIndex(r.Phase) < phaseIndex(PhaseQuiesce) {
		return errors.New("handoff precedes quiesce")
	}
	if r.Pause != nil && (!token(r.Pause.Revision, false) || !lowerHex(r.Pause.PolicySHA256, 64) ||
		(r.Pause.Origin != nodepolicy.PausedLocally && r.Pause.Origin != nodepolicy.PausedByPeer) || r.Pause.Owned && r.Pause.Origin != nodepolicy.PausedLocally) {
		return errors.New("invalid durable Pause fence")
	}
	if r.Pause != nil && phaseIndex(r.Phase) < phaseIndex(PhaseQuiesce) {
		return errors.New("Pause fence precedes quiesce")
	}
	if r.Handoff != HandoffNone && r.Pause == nil {
		return errors.New("handoff lacks the accepted durable Pause fence")
	}
	if r.Runner != nil && (r.Runner.PID <= 0 || int64(r.Runner.PID) > 1<<31-1 || r.Runner.Identity.Executable != j.runnerPath() || !validCreation(r.Runner.Identity.Created)) {
		return errors.New("invalid exact runner lifetime")
	}
	if phaseIndex(r.Phase) >= phaseIndex(PhaseShutdown) && r.Handoff != HandoffAcknowledged ||
		phaseIndex(r.Phase) < phaseIndex(PhaseSwap) && r.Rename != RenameNone || phaseIndex(r.Phase) > phaseIndex(PhaseSwap) && r.Rename != RenameInstalled {
		return errors.New("phase lacks durable handoff/rename evidence")
	}
	if r.Outcome != nil {
		switch *r.Outcome {
		case OutcomeSucceeded:
			if r.Phase != PhaseFinish || r.Rename != RenameInstalled {
				return errors.New("success requires final readback phase")
			}
			v := r.Verified
			if v == nil || !r.RunnerReady || !v.OwnershipAcknowledged || v.ReadbackConfidence != ConfidenceVerified || !knownFences(v.Fences) || !reflect.DeepEqual(v.RunningService, r.Runtime.ExpectedService) || !reflect.DeepEqual(v.RunningGUI, r.Runtime.ExpectedGUI) {
				return errors.New("success lacks verified selected-build readback")
			}
			for _, proof := range []Observation[bool]{v.Saved, v.Enforced, v.PhysicalStop, v.RemainsPaused} {
				if !proof.Known || proof.Value == nil || !*proof.Value {
					return errors.New("success lacks enforced paused readback")
				}
			}
		case OutcomeRefused:
			if r.Phase != PhasePrepare || r.Handoff != HandoffNone || r.Rename != RenameNone {
				return errors.New("refusal cannot clear an interrupted lifecycle")
			}
		case OutcomeFailed, OutcomeInterrupted, OutcomeUncertain:
		default:
			return errors.New("invalid receipt outcome")
		}
	}
	return nil
}

func receiptTransition(old, next Receipt, recovering bool) error {
	a, b := old, next
	a.Phase, b.Phase = "", ""
	a.Rename, b.Rename = "", ""
	a.Handoff, b.Handoff = "", ""
	a.Runner, b.Runner = nil, nil
	a.RunnerToken, b.RunnerToken = "", ""
	a.Pause, b.Pause = nil, nil
	a.Outcome, b.Outcome = nil, nil
	a.PreparedIdentity, b.PreparedIdentity = "", ""
	a.PreparedIntent, b.PreparedIntent = false, false
	a.ServiceLaunch, b.ServiceLaunch = LaunchReservation{}, LaunchReservation{}
	a.GUILaunch, b.GUILaunch = LaunchReservation{}, LaunchReservation{}
	a.Verified, b.Verified = nil, nil
	a.Progress, b.Progress = nil, nil
	a.Failure, b.Failure = nil, nil
	a.CleanupPending, b.CleanupPending = false, false
	a.CleanupStarted, b.CleanupStarted = false, false
	a.ReleasePending, b.ReleasePending = false, false
	a.CleanupRunner, b.CleanupRunner = nil, nil
	a.CleanupDeadline, b.CleanupDeadline = time.Time{}, time.Time{}
	a.RunnerReady, b.RunnerReady = false, false
	a.RecoveryStartedAt, b.RecoveryStartedAt = time.Time{}, time.Time{}
	a.RecoveryDeadline, b.RecoveryDeadline = time.Time{}, time.Time{}
	a.RecoveryCaller, b.RecoveryCaller = nil, nil
	a.ReadbackPause, b.ReadbackPause = nil, nil
	if !reflect.DeepEqual(a, b) {
		return errors.New("receipt authority is immutable")
	}
	if old.Outcome != nil && !recovering && !reflect.DeepEqual(old, next) {
		return ErrRecoveryNeeded
	}
	if old.Pause != nil && !reflect.DeepEqual(old.Pause, next.Pause) {
		return errors.New("accepted Pause fence cannot be replaced")
	}
	if old.RunnerToken != next.RunnerToken && !recovering {
		return ErrRecoveryNeeded
	}
	if old.PreparedIdentity != "" && old.PreparedIdentity != next.PreparedIdentity && !(recovering && old.Phase == PhasePrepare && old.Handoff == HandoffNone) {
		return errors.New("prepared directory identity is immutable")
	}
	if old.PreparedIntent && !next.PreparedIntent {
		return errors.New("prepared creation intent cannot be revoked")
	}
	if !reflect.DeepEqual(old.RecoveryCaller, next.RecoveryCaller) && !recovering {
		return ErrRecoveryNeeded
	}
	if (!old.RecoveryStartedAt.Equal(next.RecoveryStartedAt) || !old.RecoveryDeadline.Equal(next.RecoveryDeadline) || old.RunnerReady && !next.RunnerReady) && !recovering {
		return ErrRecoveryNeeded
	}
	for _, pair := range [][2]LaunchReservation{{old.ServiceLaunch, next.ServiceLaunch}, {old.GUILaunch, next.GUILaunch}} {
		if pair[0].Intent && pair[0].Token != pair[1].Token && !recovering {
			return errors.New("launch correlation requires explicit recovery")
		}
		if pair[0].Intent && !pair[1].Intent || pair[0].Acknowledged && !pair[1].Acknowledged && !recovering || pair[0].Process != nil && !reflect.DeepEqual(pair[0].Process, pair[1].Process) && !recovering {
			return errors.New("successor reservation cannot be replaced without explicit recovery")
		}
	}
	if old.Runner != nil && !reflect.DeepEqual(old.Runner, next.Runner) {
		// Explicit recovery already owns the OS lock. Its process owner must
		// first prove the previous lifetime is gone, then durably acknowledge
		// the new exact lifetime without advancing filesystem/lifecycle work.
		if !recovering || next.Handoff != old.Handoff && next.Handoff != HandoffAcknowledged || next.Phase != old.Phase || next.Rename != old.Rename {
			return errors.New("reserved runner lifetime requires explicit recovery")
		}
	}
	phaseDelta := phaseIndex(next.Phase) - phaseIndex(old.Phase)
	renameDelta := renameIndex(next.Rename) - renameIndex(old.Rename)
	handoffDelta := handoffIndex(next.Handoff) - handoffIndex(old.Handoff)
	if phaseDelta < 0 || phaseDelta > 1 || renameDelta < 0 || renameDelta > 1 || handoffDelta < 0 || handoffDelta > 1 ||
		phaseDelta != 0 && (renameDelta != 0 || handoffDelta != 0) {
		return errors.New("receipt boundary must progress one durable step at a time")
	}
	return nil
}

func (r Receipt) Deadline() time.Time {
	if !r.RecoveryDeadline.IsZero() {
		return r.RecoveryDeadline
	}
	return r.Runtime.Deadline
}

func OpenPrivateOperationRoot(path string) (*RealRoot, error) {
	r, err := OpenRealRoot(path)
	if err != nil {
		return nil, err
	}
	if err := validatePrivateRoot(r); err != nil {
		r.Close()
		return nil, err
	}
	return r, nil
}

func validReservation(p RunnerReservation) bool {
	return p.PID > 0 && int64(p.PID) <= 1<<31-1 && localPath(p.Identity.Executable) && validCreation(p.Identity.Created)
}

func ValidateRuntimeAuthority(target string, a RuntimeAuthority) error {
	if a.AcceptedAt.IsZero() || a.Deadline.Sub(a.AcceptedAt) != 30*time.Minute || localRootSyntax(a.CandidatePath) != nil || a.CandidatePath == target || containsPath(target, a.CandidatePath) || containsPath(a.CandidatePath, target) ||
		!responseText(a.Endpoint, MaxPathBytes) || a.Endpoint == "" || !lowerHex(a.RunnerSHA256, 64) || !lowerHex(a.CandidateMetadataSHA256, 64) || !validCreation(a.TargetIdentity) ||
		!validReservation(a.OldGUI) || !validReservation(a.OldService) || len(a.OldProcesses) < 2 || len(a.OldProcesses) > 256 {
		return errors.New("incomplete durable runtime authority")
	}
	layout, ok := LayoutFor(runtime.GOOS)
	if !ok || a.OldGUI.Identity.Executable != filepath.Join(target, filepath.FromSlash(layout.Executable)) || a.OldService.Identity.Executable != filepath.Join(target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable) {
		return errors.New("old GUI/service images do not belong to the selected target")
	}
	if runtime.GOOS == "windows" {
		const prefix = `\\.\pipe\`
		if len(a.Endpoint) <= len(prefix) || !strings.EqualFold(a.Endpoint[:len(prefix)], prefix) || strings.ContainsAny(a.Endpoint[len(prefix):], `\/:*?`) {
			return errors.New("maintenance endpoint must be a local named pipe")
		}
	} else if localRootSyntax(a.Endpoint) != nil || filepath.Clean(a.Endpoint) != a.Endpoint {
		return errors.New("maintenance endpoint must be a local absolute socket")
	}
	gui, service := false, false
	seen := map[int]bool{}
	for _, p := range a.OldProcesses {
		if !validReservation(p) || seen[p.PID] {
			return errors.New("invalid old process inventory")
		}
		seen[p.PID] = true
		gui = gui || p == a.OldGUI
		service = service || p == a.OldService
	}
	if !gui || !service || a.OldGUI.PID == a.OldService.PID {
		return errors.New("old GUI/service lifetimes are not independently inventoried")
	}
	for _, i := range []Identity{a.ExpectedService, a.ExpectedGUI} {
		if !validIdentity(i) || !i.BuildID.Known || !i.SourceRevision.Known || !i.Dirty.Known || !i.SHA256.Known {
			return errors.New("missing expected diagnostic build identity")
		}
	}
	for _, c := range []LaunchContext{a.ServiceLaunch, a.GUILaunch} {
		if !lowerHex(c.EnvironmentSHA256, 64) || len(c.Args) > 256 || len(c.Environment) > 32 {
			return errors.New("unbounded launch context")
		}
		for _, arg := range c.Args {
			if !responseText(arg, MaxPathBytes) {
				return errors.New("invalid launch argument")
			}
		}
		for _, env := range c.Environment {
			key, value, ok := strings.Cut(env, "=")
			if !ok || !responseText(value, MaxPathBytes) {
				return errors.New("invalid profile environment")
			}
			switch key {
			case "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "NVPAIR_SERVICE_ENDPOINT":
			default:
				return errors.New("non-profile environment cannot be backed up")
			}
		}
	}
	return nil
}

// RemoveRunner removes only a dead, hash-verified canonical helper. The process
// owner supplies independent death proof before calling this lock-owned step.
func (j *Journal) RemoveRunner(lock *OperationLock, expectedSHA256 string) error {
	if err := j.checkLock(lock); err != nil {
		return err
	}
	if !lowerHex(expectedSHA256, 64) {
		return errors.New("missing runner digest")
	}
	runner, err := privateChild(j.root, "runner", false)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	defer runner.Close()
	name := filepath.Base(j.runnerPath())
	info, err := runner.root.Lstat(name)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return errors.New("runner is not a regular owned file")
	}
	f, err := runner.root.Open(name)
	if err != nil {
		return err
	}
	if err := nativePrivateRunnerCheck(f, info); err != nil {
		f.Close()
		return err
	}
	h := sha256.New()
	n, readErr := io.Copy(h, io.LimitReader(f, MaxFileBytes+1))
	held, statErr := f.Stat()
	closeErr := f.Close()
	if err := errors.Join(readErr, statErr, closeErr); err != nil {
		return err
	}
	named, err := runner.root.Lstat(name)
	if err != nil || n > MaxFileBytes || !sameIdentity(info, held) || !sameIdentity(info, named) || hex.EncodeToString(h.Sum(nil)) != expectedSHA256 {
		return errors.New("runner bytes/identity changed")
	}
	if err := validatePrivateRoot(runner); err != nil {
		return err
	}
	return runner.root.Remove(name)
}

func (j *Journal) SaveCleanup(lock *OperationLock, r Receipt) (JournalState, error) {
	if err := j.checkLock(lock); err != nil {
		return JournalState{}, err
	}
	state, err := j.Read()
	if err != nil {
		return state, err
	}
	if state.TerminalPending || state.Current != nil || state.LastTerminal == nil {
		return state, ErrRecoveryNeeded
	}
	old := *state.LastTerminal
	old.CleanupPending = r.CleanupPending
	old.CleanupStarted = r.CleanupStarted
	old.ReleasePending = r.ReleasePending
	old.CleanupRunner = r.CleanupRunner
	old.CleanupDeadline = r.CleanupDeadline
	if err := j.validateReceipt(r); err != nil {
		return state, err
	}
	if !reflect.DeepEqual(old, r) {
		return state, errors.New("cleanup cannot change terminal authority")
	}
	state.LastTerminal = &r
	return j.publish(state)
}

func validCreation(value string) bool {
	if value == "" || len(value) > 128 {
		return false
	}
	for _, c := range value {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c == ':') {
			return false
		}
	}
	return true
}

func (v Inventory) validate() error {
	if !ValidOperationID(v.OperationID) || !lowerHex(v.PackageSHA256, 64) || len(v.Entries) == 0 || len(v.Entries) > MaxEntries ||
		(v.Platform != "windows" && v.Platform != "darwin" && v.Platform != "linux") || (v.Architecture != "amd64" && v.Architecture != "arm64") {
		return errors.New("invalid inventory header")
	}
	hash := sha256.New()
	seen := map[string]os.FileMode{}
	var total int64
	// Reserve room for the bounded header and JSON delimiters before any
	// whole-inventory marshal. Per-entry marshals stay path-size bounded.
	encodedBytes := 1024
	previous := ""
	for _, e := range v.Entries {
		folded := foldPackagePath(e.Path)
		if !validRelative(e.Path) || e.Path <= previous {
			return errors.New("invalid, duplicate or unsorted inventory path")
		}
		if _, exists := seen[folded]; exists {
			return errors.New("colliding inventory paths")
		}
		if parent := path.Dir(e.Path); parent != "." && !seen[foldPackagePath(parent)].IsDir() {
			return errors.New("inventory parent is missing or linked")
		}
		if e.Mode&^(os.ModeType|os.ModePerm) != 0 || e.Size < 0 || e.Size > MaxFileBytes {
			return errors.New("invalid inventory mode or size")
		}
		switch {
		case e.Mode.IsRegular():
			if !lowerHex(e.SHA256, 64) || e.Link != "" || e.Size > MaxPackageBytes-total {
				return errors.New("invalid inventory file")
			}
			total += e.Size
		case e.Mode&os.ModeType == os.ModeDir:
			if e.Size != 0 || e.SHA256 != "" || e.Link != "" {
				return errors.New("invalid inventory directory")
			}
		case e.Mode&os.ModeType == os.ModeSymlink:
			resolved := path.Clean(path.Join(path.Dir(e.Path), e.Link))
			if v.Platform != "darwin" || !strings.HasPrefix(e.Path, "Contents/Frameworks/") || !strings.Contains(e.Path, ".framework/") ||
				e.Link == "" || len(e.Link) > MaxPathBytes || strings.ContainsAny(e.Link, "\\:\x00\r\n") || strings.HasPrefix(e.Link, "/") || !validRelative(resolved) || e.SHA256 != "" {
				return errors.New("invalid inventory framework link")
			}
		default:
			return errors.New("special inventory entry")
		}
		encoded, err := json.Marshal(struct {
			Path       string
			Size       int64
			Mode       uint32
			Hash, Link string
		}{e.Path, e.Size, uint32(e.Mode), e.SHA256, e.Link})
		if err != nil {
			return err
		}
		fmt.Fprintf(hash, "%d:", len(encoded))
		hash.Write(encoded)
		stored, err := json.Marshal(e)
		if err != nil {
			return err
		}
		encodedBytes += len(stored) + 1
		if encodedBytes > MaxInventoryBytes {
			return errors.New("inventory exceeds publication bound")
		}
		seen[folded] = e.Mode
		previous = e.Path
	}
	if hex.EncodeToString(hash.Sum(nil)) != v.PackageSHA256 {
		return errors.New("inventory package digest mismatch")
	}
	return nil
}

func strictJournalJSON(data []byte, value any) error {
	if !utf8.Valid(data) || !json.Valid(data) {
		return errors.New("malformed journal JSON")
	}
	if err := journalJSONStructure(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return err
	}
	return decodeStrict(data, value)
}

// Check nesting, duplicate fields and collection bounds before decoding slices.
// A tiny JSON element must not allocate an arbitrarily large InventoryEntry list.
func journalJSONStructure(decoder *json.Decoder, depth int) error {
	if depth > 16 {
		return errors.New("journal JSON exceeds nesting bound")
	}
	value, err := decoder.Token()
	if err != nil {
		return err
	}
	delim, compound := value.(json.Delim)
	if !compound {
		return nil
	}
	switch delim {
	case '{':
		keys := map[string]bool{}
		for decoder.More() {
			value, err := decoder.Token()
			if err != nil {
				return err
			}
			key, ok := value.(string)
			key = strings.ToLower(key)
			if !ok || keys[key] || len(keys) >= 64 {
				return errors.New("duplicate or excessive journal fields")
			}
			keys[key] = true
			if err := journalJSONStructure(decoder, depth+1); err != nil {
				return err
			}
		}
	case '[':
		entries := 0
		for decoder.More() {
			if entries >= MaxEntries {
				return errors.New("journal JSON exceeds collection bound")
			}
			entries++
			if err := journalJSONStructure(decoder, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("unexpected journal JSON delimiter")
	}
	_, err = decoder.Token()
	return err
}

func boundedJSON(value any, limit int) ([]byte, error) {
	var buffer bytes.Buffer
	writer := &boundedJournalWriter{buffer: &buffer, remaining: limit}
	if err := json.NewEncoder(writer).Encode(value); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

type boundedJournalWriter struct {
	buffer    *bytes.Buffer
	remaining int
}

func (w *boundedJournalWriter) Write(data []byte) (int, error) {
	if len(data) > w.remaining {
		return 0, errors.New("journal exceeds publication bound")
	}
	w.remaining -= len(data)
	return w.buffer.Write(data)
}

func (j *Journal) readFile(name string, limit int64) ([]byte, error) {
	if err := validatePrivateRoot(j.root); err != nil {
		return nil, err
	}
	before, err := j.root.root.Lstat(name)
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Size() <= 0 || before.Size() > limit {
		return nil, errors.New("journal file is not a bounded regular file")
	}
	f, err := j.root.root.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	held, err := f.Stat()
	if err != nil || !sameIdentity(before, held) {
		return nil, errors.New("journal file was substituted")
	}
	if err := nativePrivateCheck(f, held); err != nil {
		return nil, err
	}
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	after, statErr := f.Stat()
	named, nameErr := j.root.root.Lstat(name)
	if int64(len(data)) > limit || statErr != nil || nameErr != nil || !sameIdentity(before, after) || !sameIdentity(before, named) {
		return nil, errors.New("journal file changed during read")
	}
	if err := nativePrivateCheck(f, after); err != nil {
		return nil, err
	}
	if err := validatePrivateRoot(j.root); err != nil {
		return nil, err
	}
	return data, nil
}

func (j *Journal) writeFile(name string, data []byte) error {
	if err := validatePrivateRoot(j.root); err != nil {
		return err
	}
	if info, err := j.root.root.Lstat(name); err == nil {
		if !info.Mode().IsRegular() {
			return errors.New("journal destination is linked or nonregular")
		}
		// Existing state is validated, never repaired by chmod/ACL mutation.
		if _, err := j.readFile(name, MaxInventoryBytes); err != nil {
			return err
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	err := j.replacer.WriteFile(filepath.Join(j.root.Path, name), data, 0o600)
	if err != nil && !errors.Is(err, durablefile.ErrReplaced) {
		return err
	}
	if afterErr := validatePrivateRoot(j.root); afterErr != nil {
		return errors.Join(err, durablefile.ErrReplaced, afterErr)
	}
	return err
}

type journalTemp struct {
	*os.File
	name string
}

func (f *journalTemp) Name() string { return f.name }

func (j *Journal) createTemp(dir, base string, perm os.FileMode) (durablefile.File, error) {
	if dir != j.root.Path || (base != "operation.json" && base != "inventory.json" && base != "terminal.json" && base != "admission.json") {
		return nil, errors.New("noncanonical journal publication")
	}
	if err := validatePrivateRoot(j.root); err != nil {
		return nil, err
	}
	id, err := NewOperationID()
	if err != nil {
		return nil, err
	}
	name := base + "." + id + ".tmp"
	f, err := j.root.root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	if err := nativePrivateCreatedFile(f); err != nil {
		f.Close()
		j.root.root.Remove(name)
		return nil, err
	}
	if err := validatePrivateRoot(j.root); err != nil {
		f.Close()
		j.root.root.Remove(name)
		return nil, err
	}
	return &journalTemp{File: f, name: filepath.Join(dir, name)}, nil
}
