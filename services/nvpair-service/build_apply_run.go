// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/shirou/gopsutil/v4/process"
	"nvpair-shared/buildapply"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/servicectl"
)

const (
	buildApplyLiveBudget     = 10 * time.Second
	buildApplyReadBudget     = 60 * time.Second
	buildApplyTotalBudget    = 30 * time.Minute
	buildApplyShutdownBudget = defaultShutdownRequestTimeout + defaultStopGrace + outputDrainTimeout + 10*time.Second
)

// The plan is constructed exclusively by trusted owners, never decoded from
// renderer input or a candidate manifest. All owners must be present before
// opening a writable journal or preparing any filesystem state.
type buildApplyPlan struct {
	receipt            buildapply.Receipt
	inventory          buildapply.Inventory
	candidate          *buildapply.Package
	current            *buildapply.Package
	accepted           time.Time
	deadline           time.Time
	expires            time.Time
	ownerFactsComplete bool
	oldProcesses       []buildapply.RunnerReservation
	serviceLaunch      buildApplyLaunch
	guiLaunch          buildApplyLaunch
	expectedService    buildapply.Identity
	expectedGUI        buildapply.Identity
}

type buildApplyLaunch struct {
	executable string
	args       []string
	env        []string
}

func (p buildApplyPlan) validate(now time.Time) error {
	r := p.receipt
	if !p.ownerFactsComplete || !r.Consent.AllowEngineStop || !r.Consent.AllowInterruption ||
		!buildapply.ValidOperationID(r.OperationID) || r.PreviewID == "" || !now.Before(p.expires) ||
		p.accepted.IsZero() || p.accepted.After(now) || p.deadline.Sub(p.accepted) != buildApplyTotalBudget || !now.Before(p.deadline) ||
		!r.Expected.RemainPaused || !r.Expected.PreserveIntentLocks || !r.Expected.StopEngines || !r.Expected.UnloadModels ||
		(r.Expected.OnActive != nodepolicy.FinishActive && r.Expected.OnActive != nodepolicy.CancelActive) ||
		r.Expected.DrainTimeoutSeconds < 0 || time.Duration(r.Expected.DrainTimeoutSeconds)*time.Second+buildApplyShutdownBudget+4*buildApplyReadBudget >= p.deadline.Sub(now) {
		return errors.New("incomplete, stale or unapproved apply authority/budget")
	}
	if _, err := buildapply.DeriveOperationPaths(r.Target, r.OperationID); err != nil {
		return err
	}
	if err := buildapply.ValidateRuntimeAuthority(r.Target, r.Runtime); err != nil {
		return err
	}
	if r.RunnerToken != r.OperationID || !r.Runtime.AcceptedAt.Equal(p.accepted) || !r.Runtime.Deadline.Equal(p.deadline) ||
		!reflect.DeepEqual(r.Runtime.OldProcesses, p.oldProcesses) || !reflect.DeepEqual(r.Runtime.ExpectedService, p.expectedService) || !reflect.DeepEqual(r.Runtime.ExpectedGUI, p.expectedGUI) ||
		!slices.Equal(r.Runtime.ServiceLaunch.Args, p.serviceLaunch.args) || !slices.Equal(r.Runtime.GUILaunch.Args, p.guiLaunch.args) ||
		buildApplyEnvironmentDigest(p.serviceLaunch.env) != r.Runtime.ServiceLaunch.EnvironmentSHA256 || buildApplyEnvironmentDigest(p.guiLaunch.env) != r.Runtime.GUILaunch.EnvironmentSHA256 {
		return errors.New("in-memory execution differs from its immutable durable authority")
	}
	if err := validateBuildApplyArguments(p.serviceLaunch.args, false); err != nil {
		return err
	}
	if err := validateBuildApplyImageArguments(r.Target, p.serviceLaunch.args); err != nil {
		return err
	}
	if err := validateBuildApplyArguments(p.guiLaunch.args, true); err != nil {
		return err
	}
	// Leave room for both the last terminal receipt and bounded public readbacks.
	// This check precedes even creation of the private operation root.
	r.InventorySHA256 = strings.Repeat("0", 64)
	encoded, err := json.Marshal(r)
	if err != nil || len(encoded) > buildapply.MaxReceiptBytes/2-2*buildapply.MaxStatusBytes {
		return errors.New("durable admission authority exceeds its publication budget")
	}
	if r.Phase != buildapply.PhasePrepare || r.Handoff != buildapply.HandoffNone || r.Rename != buildapply.RenameNone || r.Outcome != nil ||
		p.inventory.OperationID != r.OperationID || p.inventory.PackageSHA256 != r.CandidateSHA256 || len(p.oldProcesses) == 0 {
		return errors.New("invalid initial transaction")
	}
	for _, f := range []buildapply.Observation[string]{r.Fences.RuntimeGeneration, r.Fences.ServiceInstance, r.Fences.BrokerGeneration, r.Fences.ManagerIncarnation, r.Fences.PolicyRevision, r.Fences.SettingsRevision, r.Fences.IntentRevision, r.Fences.LocksRevision, r.Fences.AvailabilityRevision} {
		if !f.Known || f.Value == nil || *f.Value == "" {
			return errors.New("missing owner fence")
		}
	}
	for _, v := range p.oldProcesses {
		if v.PID <= 0 || v.Identity.Created == "" || !filepath.IsAbs(v.Identity.Executable) {
			return errors.New("incomplete old process lifetime")
		}
	}
	for _, digest := range []string{r.CandidateSHA256, r.PreviousSHA256, r.Protected.Policy, r.Protected.EngineSettings, r.Protected.Intent, r.Protected.Locks, r.Protected.IdentityMembership} {
		if !validBuildApplyDigest(digest) {
			return errors.New("missing trusted protected digest")
		}
	}
	for _, launch := range []buildApplyLaunch{p.serviceLaunch, p.guiLaunch} {
		if !filepath.IsAbs(launch.executable) || launch.env == nil {
			return errors.New("missing trusted launch context")
		}
	}
	if !verifiedBuildApplyIdentity(p.expectedService) || !verifiedBuildApplyIdentity(p.expectedGUI) {
		return errors.New("missing expected build identities")
	}
	layout, ok := buildapply.LayoutFor(p.inventory.Platform)
	if !ok || p.inventory.Platform != runtime.GOOS || p.inventory.Architecture != runtime.GOARCH ||
		!sameBuildApplyImage(p.serviceLaunch.executable, filepath.Join(r.Target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable)) ||
		!sameBuildApplyImage(p.guiLaunch.executable, filepath.Join(r.Target, filepath.FromSlash(layout.Executable))) {
		return errors.New("launch context does not name the canonical target images")
	}
	return nil
}

// Narrow owner interfaces allow every irreversible step and durability failure
// to be exercised without starting an executable or contacting an installation.
// There are no optional callbacks that silently skip a required phase.
type buildApplyStore interface {
	Begin(buildapply.Receipt, buildapply.Inventory) (buildapply.JournalState, error)
	Save(buildapply.Receipt) (buildapply.JournalState, error)
	SaveCleanup(buildapply.Receipt) (buildapply.JournalState, error)
}

type buildApplyJournalStore struct {
	journal *buildapply.Journal
	lock    *buildapply.OperationLock
}

func (s buildApplyJournalStore) Begin(r buildapply.Receipt, i buildapply.Inventory) (buildapply.JournalState, error) {
	return s.journal.Begin(s.lock, r, i)
}
func (s buildApplyJournalStore) Save(r buildapply.Receipt) (buildapply.JournalState, error) {
	return s.journal.Save(s.lock, r)
}
func (s buildApplyJournalStore) SaveCleanup(r buildapply.Receipt) (buildapply.JournalState, error) {
	return s.journal.SaveCleanup(s.lock, r)
}

type buildApplyCleanupError struct{ cause error }

func (e *buildApplyCleanupError) Error() string {
	return "Build applied while paused; ownership cleanup requires explicit recovery."
}
func (e *buildApplyCleanupError) Unwrap() error { return e.cause }

type buildApplyFiles interface {
	Freeze(context.Context, buildApplyPlan) error
	Digest(context.Context, string) (string, error)
	Rename(context.Context, string, string) error
	PreparedIdentity() string
}

type buildApplyLifecycle interface {
	Prepare(context.Context, buildapply.PrepareRequest) (buildApplyPrepared, error)
	// Reserve launches once and independently inspects that lifetime while the
	// original owner still holds the lock. Acknowledge transfers the lock to
	// that reservation and returns only after its durable acknowledgement.
	Reserve(context.Context, buildapply.Receipt) (buildapply.RunnerReservation, error)
	Acknowledge(context.Context, buildapply.Receipt) error
	Stop(context.Context, []buildapply.RunnerReservation) error
	Restart(context.Context, buildApplyPlan) (buildApplySuccessors, error)
	ReadBuild(context.Context, buildApplyPlan) (buildApplyBuildReadback, error)
	ReadSettings(context.Context, buildapply.Receipt) (buildApplySettingsReadback, error)
	Finish(context.Context, buildapply.FinishRequest) error
}

type buildApplyPrepared struct {
	pause     buildapply.JournalPauseFence
	operation buildapply.BuildApplyOperation
}

func buildApplyObservedTrue(v buildapply.Observation[bool]) bool {
	return v.Known && v.Value != nil && *v.Value
}

func buildApplyFencesComplete(f buildapply.Fences) bool {
	for _, v := range []buildapply.Observation[string]{f.RuntimeGeneration, f.ServiceInstance, f.BrokerGeneration, f.ManagerIncarnation, f.PolicyRevision, f.SettingsRevision, f.IntentRevision, f.LocksRevision, f.AvailabilityRevision} {
		if !v.Known || v.Value == nil || *v.Value == "" {
			return false
		}
	}
	return true
}

type buildApplyBuildReadback struct {
	service         buildapply.Identity
	gui             buildapply.Identity
	serviceProcess  buildapply.RunnerReservation
	guiProcess      buildapply.RunnerReservation
	brokerReady     bool
	guiAcknowledged bool
}

type buildApplySuccessors struct {
	service, gui buildapply.RunnerReservation
	serviceReady bool
}

func verifyBuildApplyBuild(p buildApplyPlan, read buildApplyBuildReadback) error {
	if !verifiedBuildApplyIdentity(read.service) || !verifiedBuildApplyIdentity(read.gui) ||
		!reflect.DeepEqual(read.service, p.expectedService) || !reflect.DeepEqual(read.gui, p.expectedGUI) ||
		!read.brokerReady || !read.guiAcknowledged {
		return errors.New("selected running build/readiness/GUI acknowledgement is unavailable")
	}
	for _, pair := range []struct {
		process buildapply.RunnerReservation
		launch  buildApplyLaunch
	}{{read.serviceProcess, p.serviceLaunch}, {read.guiProcess, p.guiLaunch}} {
		if pair.process.PID <= 0 || pair.process.Identity.Created == "" || !sameBuildApplyImage(pair.process.Identity.Executable, pair.launch.executable) {
			return errors.New("selected successor process lifetime is unavailable")
		}
		for _, old := range p.oldProcesses {
			if old == pair.process {
				return errors.New("successor is an old process lifetime")
			}
		}
	}
	return nil
}

type buildApplySettingsReadback struct {
	fences             buildapply.Fences
	protected          buildapply.ProtectedDigests
	pause              buildapply.JournalPauseFence
	stopEngines        bool
	unloadModels       bool
	saved              bool
	enforced           bool
	physicalStop       bool
	recoveryRequired   bool
	activationVerified bool
}

func verifyBuildApplySettings(r buildapply.Receipt, read buildApplySettingsReadback) error {
	pause := r.Pause
	if r.ReadbackPause != nil {
		pause = r.ReadbackPause
	}
	if pause == nil || !reflect.DeepEqual(*pause, read.pause) || !read.stopEngines || !read.unloadModels ||
		!read.saved || !read.enforced || !read.physicalStop || read.recoveryRequired || !read.activationVerified || !buildApplyFencesComplete(read.fences) {
		return errors.New("paused settings/activation evidence is incomplete or changed")
	}
	expected := r.Protected
	// The accepted Pause is the only permitted policy semantic difference.
	// Secret-bearing snapshots stay inside their owners; only digests cross.
	expected.Policy = r.Pause.PolicySHA256
	if !reflect.DeepEqual(expected, read.protected) {
		return errors.New("protected policy/settings/intent/locks/identity semantics changed")
	}
	return nil
}

type buildApplyTransaction struct {
	plan        buildApplyPlan
	store       buildApplyStore
	files       buildApplyFiles
	owners      buildApplyLifecycle
	receipt     buildapply.Receipt
	operation   buildapply.BuildApplyOperation
	handoffOnly bool
}

func (t *buildApplyTransaction) save() error {
	t.operation.Phase = t.receipt.Phase
	t.operation.Outcome = t.receipt.Outcome
	t.operation.OwnershipAcknowledged = t.receipt.RunnerReady && t.receipt.Handoff == buildapply.HandoffAcknowledged
	progress := t.operation
	t.receipt.Progress = &progress
	state, err := t.store.Save(t.receipt)
	if err != nil {
		return err
	}
	r := state.Current
	if r == nil {
		r = state.LastTerminal
	}
	if r == nil || !reflect.DeepEqual(*r, t.receipt) {
		return errors.New("receipt publication was not confirmed")
	}
	t.operation.Phase, t.operation.Outcome = r.Phase, r.Outcome
	return r.ValidateOperation(t.operation)
}

func (t *buildApplyTransaction) phase(p buildapply.Phase) error {
	t.receipt.Phase = p
	t.operation.Phase = p
	return t.save()
}

// fail attempts to retain an uncertain current receipt. A failed terminal
// publication never creates success, clears Pause or starts cleanup/rollback.
func (t *buildApplyTransaction) fail(reason string) (buildapply.BuildApplyOperation, error) {
	outcome := buildapply.OutcomeUncertain
	t.receipt.Outcome = &outcome
	t.receipt.Verified = nil
	t.operation.Outcome = &outcome
	t.operation.Failure = &buildapply.BuildApplyFailure{Phase: t.receipt.Phase, Reason: reason,
		Detail:             "The operation could not verify this boundary; retained bundles and Pause require explicit recovery.",
		ReadbackConfidence: buildapply.ConfidenceUnknown, SafeNextAction: "Read status and explicitly recover this operation."}
	t.receipt.Failure = t.operation.Failure
	err := t.save()
	return t.operation, errors.Join(errors.New(reason), err)
}

func (t *buildApplyTransaction) Run(ctx context.Context) (buildapply.BuildApplyOperation, error) {
	if err := ctx.Err(); err != nil {
		return buildapply.BuildApplyOperation{}, err
	}
	if err := t.plan.validate(time.Now()); err != nil {
		return buildapply.BuildApplyOperation{}, err
	}
	ctx, cancel := context.WithDeadline(ctx, t.plan.deadline)
	defer cancel()
	t.receipt = t.plan.receipt
	t.operation = projectBuildApplyReceipt(t.receipt)
	state, err := t.store.Begin(t.receipt, t.plan.inventory)
	if err != nil {
		return t.operation, err
	}
	if state.Current == nil {
		return t.operation, errors.New("begin did not publish ownership")
	}
	t.receipt = *state.Current
	t.receipt.PreparedIntent = true
	if err := t.save(); err != nil {
		return t.fail("prepared_creation_intent_uncertain")
	}
	if err := t.files.Freeze(ctx, t.plan); err != nil {
		return t.fail("prepare_unverified")
	}
	t.receipt.PreparedIdentity = t.files.PreparedIdentity()
	if err := t.save(); err != nil {
		return t.fail("prepared_identity_uncertain")
	}
	deadline, _ := ctx.Deadline()
	remainingRequired := time.Duration(t.receipt.Expected.DrainTimeoutSeconds)*time.Second + buildApplyShutdownBudget + 4*buildApplyReadBudget
	if time.Until(deadline) <= remainingRequired {
		return t.fail("saved_drain_budget_no_longer_fits")
	}
	if err := t.phase(buildapply.PhaseQuiesce); err != nil {
		return t.fail("quiesce_intent_uncertain")
	}
	prepareCtx, stopPrepare := context.WithTimeout(ctx, time.Duration(t.receipt.Expected.DrainTimeoutSeconds)*time.Second+buildApplyShutdownBudget)
	prepared, err := t.owners.Prepare(prepareCtx, buildapply.PrepareRequest{OperationID: t.receipt.OperationID, PreviewID: t.receipt.PreviewID, Fences: t.receipt.Fences, Consent: t.receipt.Consent})
	stopPrepare()
	if err != nil {
		return t.fail("pause_or_physical_stop_unverified")
	}
	op := prepared.operation
	if op.OperationID != t.receipt.OperationID || op.PreviewID != t.receipt.PreviewID || op.Phase != buildapply.PhaseQuiesce || op.Outcome != nil ||
		!buildApplyObservedTrue(op.Saved) || !buildApplyObservedTrue(op.Enforced) || !buildApplyObservedTrue(op.PhysicalStop) || !buildApplyObservedTrue(op.RemainsPaused) || !buildApplyFencesComplete(op.Fences) {
		return t.fail("pause_or_physical_stop_unverified")
	}
	pause := prepared.pause
	if pause.Revision == "" || pause.PolicySHA256 == "" || (pause.Origin != nodepolicy.PausedLocally && pause.Origin != nodepolicy.PausedByPeer) || pause.Owned && pause.Origin != nodepolicy.PausedLocally {
		return t.fail("pause_fence_invalid")
	}
	t.receipt.Pause = &pause
	t.receipt.ReadbackPause = &pause
	t.operation.Fences = op.Fences
	t.operation.Saved = op.Saved
	t.operation.Enforced = op.Enforced
	t.operation.PhysicalStop = op.PhysicalStop
	t.operation.RemainsPaused = op.RemainsPaused
	if err := t.save(); err != nil {
		return t.fail("pause_publication_uncertain")
	}
	t.receipt.Handoff = buildapply.HandoffIntent
	if err := t.save(); err != nil {
		return t.fail("handoff_intent_uncertain")
	}
	// The native and injected owners honor the same exact process reservation
	// and durable acknowledgement contract before shutdown is permitted.
	runner, err := t.owners.Reserve(ctx, t.receipt)
	if err != nil {
		return t.fail("handoff_unverified")
	}
	t.receipt.Runner = &runner
	t.receipt.Handoff = buildapply.HandoffReserved
	if err := t.save(); err != nil {
		return t.fail("runner_reservation_uncertain")
	}
	if t.handoffOnly {
		return t.operation, nil
	}
	if err := t.owners.Acknowledge(ctx, t.receipt); err != nil {
		return t.fail("handoff_ack_unverified")
	}
	t.receipt.Handoff = buildapply.HandoffAcknowledged
	t.receipt.RunnerReady = true
	if err := t.save(); err != nil {
		return t.fail("handoff_ack_uncertain")
	}
	t.operation.OwnershipAcknowledged = true
	return t.runRemaining(ctx)
}

func buildApplyPhaseIndex(p buildapply.Phase) int {
	for i, v := range []buildapply.Phase{buildapply.PhasePrepare, buildapply.PhaseQuiesce, buildapply.PhaseShutdown, buildapply.PhaseSwap, buildapply.PhaseRestart, buildapply.PhaseVerifyBuild, buildapply.PhaseVerifySettings, buildapply.PhaseFinish} {
		if p == v {
			return i
		}
	}
	return -1
}

func (t *buildApplyTransaction) advance(p buildapply.Phase) error {
	if buildApplyPhaseIndex(t.receipt.Phase) >= buildApplyPhaseIndex(p) {
		return nil
	}
	return t.phase(p)
}

func (t *buildApplyTransaction) runRemaining(ctx context.Context) (buildapply.BuildApplyOperation, error) {
	if buildApplyPhaseIndex(t.receipt.Phase) <= buildApplyPhaseIndex(buildapply.PhaseShutdown) {
		if err := t.advance(buildapply.PhaseShutdown); err != nil {
			return t.fail("shutdown_intent_uncertain")
		}
		stopCtx, cancel := context.WithTimeout(ctx, buildApplyShutdownBudget)
		err := t.owners.Stop(stopCtx, t.plan.oldProcesses)
		cancel()
		if err != nil {
			return t.fail("process_exit_unverified")
		}
	}
	t.operation.PhysicalStop = buildapply.Known(true)
	t.operation.RemainsPaused = buildapply.Known(true)
	if err := t.advance(buildapply.PhaseSwap); err != nil {
		return t.fail("swap_intent_uncertain")
	}
	if err := t.swap(ctx); err != nil {
		return t.fail("swap_unverified")
	}
	if err := t.advance(buildapply.PhaseRestart); err != nil {
		return t.fail("restart_intent_uncertain")
	}
	readCtx, readCancel := context.WithTimeout(ctx, buildApplyReadBudget)
	successors, err := t.owners.Restart(readCtx, t.plan)
	readCancel()
	if err != nil || !successors.serviceReady {
		return t.fail("successor_start_unverified")
	}
	if err := t.advance(buildapply.PhaseVerifyBuild); err != nil {
		return t.fail("build_read_intent_uncertain")
	}
	readCtx, readCancel = context.WithTimeout(ctx, buildApplyReadBudget)
	build, err := t.owners.ReadBuild(readCtx, t.plan)
	readCancel()
	if err != nil || verifyBuildApplyBuild(t.plan, build) != nil || build.serviceProcess != successors.service || build.guiProcess != successors.gui {
		return t.fail("running_build_unverified")
	}
	if digest, err := t.files.Digest(ctx, t.receipt.Target); err != nil || digest != t.receipt.CandidateSHA256 {
		return t.fail("installed_bytes_changed")
	}
	t.operation.RunningService, t.operation.RunningGUI = build.service, build.gui
	if err := t.advance(buildapply.PhaseVerifySettings); err != nil {
		return t.fail("settings_read_intent_uncertain")
	}
	readCtx, readCancel = context.WithTimeout(ctx, buildApplyReadBudget)
	settings, err := t.owners.ReadSettings(readCtx, t.receipt)
	if err == nil {
		err = verifyBuildApplySettings(t.receipt, settings)
	}
	readCancel()
	if err != nil {
		return t.fail("settings_or_pause_unverified")
	}
	t.operation.Fences = settings.fences
	if err := t.advance(buildapply.PhaseFinish); err != nil {
		return t.fail("finish_intent_uncertain")
	}
	if err := ctx.Err(); err != nil {
		return t.fail("transaction_deadline")
	}
	t.operation.Saved, t.operation.Enforced = buildapply.Known(true), buildapply.Known(true)
	t.operation.ReadbackConfidence = buildapply.ConfidenceVerified
	outcome := buildapply.OutcomeSucceeded
	t.receipt.Outcome, t.operation.Outcome = &outcome, &outcome
	t.receipt.CleanupPending = true
	t.receipt.ReleasePending = true
	verified := t.operation
	t.receipt.Verified = &verified
	if err := t.save(); err != nil {
		return t.fail("terminal_publication_uncertain")
	}
	// The broker's activation/settings veto remains held through the durable
	// terminal commit. Release failure is cleanup, not a failed build readback.
	readCtx, readCancel = context.WithTimeout(ctx, buildApplyReadBudget)
	err = t.owners.Finish(readCtx, buildapply.FinishRequest{OperationID: t.receipt.OperationID, Fences: t.operation.Fences})
	readCancel()
	if err != nil {
		return t.operation, &buildApplyCleanupError{cause: err}
	}
	t.receipt.ReleasePending = false
	if _, err := t.store.SaveCleanup(t.receipt); err != nil {
		return t.operation, &buildApplyCleanupError{cause: err}
	}
	// Retained-bundle cleanup runs independently after this durable success.
	return t.operation, nil
}

func verifiedBuildApplyIdentity(i buildapply.Identity) bool {
	return i.BuildID.Known && i.BuildID.Value != nil && *i.BuildID.Value != "" && i.SHA256.Known && i.SHA256.Value != nil && len(*i.SHA256.Value) == 64 && i.SourceRevision.Known && i.SourceRevision.Value != nil && i.Dirty.Known && i.Dirty.Value != nil
}

func validBuildApplyDigest(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// Reconcile actual hashes at both rename boundaries. Nothing removes either
// sibling or launches the predecessor, even after a half-completed rename.
func (t *buildApplyTransaction) swap(ctx context.Context) error {
	p, err := buildapply.DeriveOperationPaths(t.receipt.Target, t.receipt.OperationID)
	if err != nil {
		return err
	}
	for {
		target, err := t.files.Digest(ctx, p.Target)
		if err != nil {
			return err
		}
		prepared, err := t.files.Digest(ctx, p.Prepared)
		if err != nil {
			return err
		}
		previous, err := t.files.Digest(ctx, p.Previous)
		if err != nil {
			return err
		}
		switch t.receipt.Rename {
		case buildapply.RenameNone:
			if target != t.receipt.PreviousSHA256 || prepared != t.receipt.CandidateSHA256 || previous != "" {
				return errors.New("unexpected initial directory identities")
			}
			t.receipt.Rename = buildapply.RenameTargetIntent
		case buildapply.RenameTargetIntent:
			if target == t.receipt.PreviousSHA256 && prepared == t.receipt.CandidateSHA256 && previous == "" {
				if err := t.files.Rename(ctx, p.Target, p.Previous); err != nil {
					return err
				}
			} else if target != "" || previous != t.receipt.PreviousSHA256 || prepared != t.receipt.CandidateSHA256 {
				return errors.New("first rename cannot be reconciled")
			}
			// Re-read actual bytes before recording completion.
			if actual, err := t.files.Digest(ctx, p.Previous); err != nil || actual != t.receipt.PreviousSHA256 {
				return errors.New("predecessor readback failed")
			}
			t.receipt.Rename = buildapply.RenamePreviousRetained
		case buildapply.RenamePreviousRetained:
			if target != "" || previous != t.receipt.PreviousSHA256 || prepared != t.receipt.CandidateSHA256 {
				return errors.New("predecessor or frozen candidate changed")
			}
			t.receipt.Rename = buildapply.RenamePreparedIntent
		case buildapply.RenamePreparedIntent:
			if target == "" && previous == t.receipt.PreviousSHA256 && prepared == t.receipt.CandidateSHA256 {
				if err := t.files.Rename(ctx, p.Prepared, p.Target); err != nil {
					return err
				}
			} else if target != t.receipt.CandidateSHA256 || previous != t.receipt.PreviousSHA256 || prepared != "" {
				return errors.New("second rename cannot be reconciled")
			}
			if actual, err := t.files.Digest(ctx, p.Target); err != nil || actual != t.receipt.CandidateSHA256 {
				return errors.New("installed readback failed")
			}
			t.receipt.Rename = buildapply.RenameInstalled
		case buildapply.RenameInstalled:
			if target != t.receipt.CandidateSHA256 || previous != t.receipt.PreviousSHA256 || prepared != "" {
				return errors.New("installed directories changed")
			}
			return nil
		default:
			return errors.New("unknown rename boundary")
		}
		if err := t.save(); err != nil {
			return err
		}
	}
}

type nativeBuildApplyFiles struct {
	journal     *buildapply.Journal
	lock        *buildapply.OperationLock
	identities  map[string]os.FileInfo
	preparedID  string
	transaction *buildApplyTransaction
}

func (f *nativeBuildApplyFiles) PreparedIdentity() string { return f.preparedID }

type buildApplyContextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r buildApplyContextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}

func (files *nativeBuildApplyFiles) Freeze(ctx context.Context, p buildApplyPlan) error {
	if p.candidate == nil || p.current == nil || files.journal == nil || files.lock == nil || p.candidate.SHA256 != p.receipt.CandidateSHA256 {
		return errors.New("missing held candidate")
	}
	measured, err := buildapply.InventoryFromPackage(p.receipt.OperationID, p.candidate)
	if err != nil || !reflect.DeepEqual(measured, p.inventory) {
		return errors.New("frozen inventory differs from measured candidate")
	}
	if err := p.current.Revalidate(ctx); err != nil {
		return err
	}
	if p.current.SHA256 != p.receipt.PreviousSHA256 || p.current.Root.Path != p.receipt.Target {
		return errors.New("current installation changed")
	}
	if err := buildapply.CheckTargetParent(p.current.Root, p.candidate.Bytes); err != nil {
		return err
	}
	if id, err := buildApplyDirectoryIdentity(p.receipt.Target); err != nil || id != p.receipt.Runtime.TargetIdentity {
		return errors.New("target directory identity changed")
	}
	if err := p.candidate.Revalidate(ctx); err != nil {
		return err
	}
	metadata, err := buildApplyMetadataDigest(ctx, p.candidate.Root.Path)
	if err != nil || metadata != p.receipt.Runtime.CandidateMetadataSHA256 {
		return errors.New("candidate metadata changed")
	}
	paths, err := buildapply.DeriveOperationPaths(p.receipt.Target, p.receipt.OperationID)
	if err != nil {
		return err
	}
	parent, err := buildapply.OpenRealRoot(filepath.Dir(paths.Target))
	if err != nil {
		return err
	}
	defer parent.Close()
	if err := createBuildApplyDirectory(paths.Prepared); err != nil {
		return err
	}
	// Retain incomplete preparation on error. It has a journal and an exact
	// operation-derived name, and never substitutes for the live bundle.
	root, err := os.OpenRoot(paths.Prepared)
	if err != nil {
		return err
	}
	defer root.Close()
	preparedIdentity, err := root.Stat(".")
	if err != nil {
		return err
	}
	targetIdentity, err := os.Lstat(paths.Target)
	if err != nil {
		return err
	}
	files.identities = map[string]os.FileInfo{paths.Target: targetIdentity, paths.Prepared: preparedIdentity}
	files.preparedID, err = buildApplyDirectoryIdentity(paths.Prepared)
	if err != nil {
		return err
	}
	if files.transaction != nil {
		files.transaction.receipt.PreparedIdentity = files.preparedID
		if err := files.transaction.save(); err != nil {
			return err
		}
	}
	complete := false
	defer func() {
		if !complete {
			_ = files.discardPrepared(context.Background(), p)
		}
	}()
	for _, e := range p.inventory.Entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		name := filepath.FromSlash(e.Path)
		switch {
		case e.Mode.IsDir():
			if err := root.Mkdir(name, e.Mode.Perm()); err != nil {
				return err
			}
			if err := root.Chmod(name, e.Mode.Perm()); err != nil {
				return err
			}
		case e.Mode.IsRegular():
			source, err := p.candidate.OpenVerifiedFile(ctx, e.Path)
			if err != nil {
				return err
			}
			dest, err := root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, e.Mode.Perm())
			if err != nil {
				source.Close()
				return err
			}
			hash := sha256.New()
			n, copyErr := io.Copy(io.MultiWriter(dest, hash), io.LimitReader(buildApplyContextReader{ctx, source}, e.Size+1))
			if copyErr == nil && (n != e.Size || hex.EncodeToString(hash.Sum(nil)) != e.SHA256) {
				copyErr = errors.New("copied bytes differ from selected inventory")
			}
			if copyErr == nil {
				copyErr = dest.Chmod(e.Mode.Perm())
			}
			if copyErr == nil {
				copyErr = dest.Sync()
			}
			err = errors.Join(copyErr, dest.Close(), source.Close())
			if err != nil {
				return err
			}
		case e.Mode&os.ModeSymlink != 0:
			if err := root.Symlink(e.Link, name); err != nil {
				return err
			}
		default:
			return errors.New("unsupported inventory entry")
		}
	}
	if err := p.candidate.Revalidate(ctx); err != nil {
		return err
	}
	for i := len(p.inventory.Entries) - 1; i >= 0; i-- {
		e := p.inventory.Entries[i]
		m, err := readBuildApplyNativeMetadata(filepath.Join(p.candidate.Root.Path, filepath.FromSlash(e.Path)))
		if err != nil {
			return err
		}
		if err := writeBuildApplyNativeMetadata(filepath.Join(paths.Prepared, filepath.FromSlash(e.Path)), m); err != nil {
			return err
		}
	}
	rootMetadata, err := readBuildApplyNativeMetadata(p.candidate.Root.Path)
	if err != nil {
		return err
	}
	// The operation owns root access permissions. Bundle-entry executable
	// modes and native metadata are preserved inside this private boundary.
	rootMetadata.Mode = uint32(os.ModeDir | 0o700)
	if err := writeBuildApplyNativeMetadata(paths.Prepared, rootMetadata); err != nil {
		return err
	}
	metadataAfter, err := buildApplyMetadataDigest(ctx, p.candidate.Root.Path)
	if err != nil || metadataAfter != metadata {
		return errors.New("source metadata changed during freeze")
	}
	frozenMetadata, err := buildApplyMetadataDigest(ctx, paths.Prepared)
	if err != nil || frozenMetadata != metadata {
		return errors.New("frozen bundle metadata differs")
	}
	if err := parent.Revalidate(); err != nil {
		return err
	}
	if err := p.current.Revalidate(ctx); err != nil {
		return err
	}
	digest, err := files.Digest(ctx, paths.Prepared)
	if err != nil || digest != p.receipt.CandidateSHA256 {
		return errors.New("frozen candidate verification failed")
	}
	if _, err := copyBuildApplyRunner(ctx, files.journal, files.lock, p.current); err != nil {
		return err
	}
	if err := syncBuildApplyPreparation(paths.Prepared, p.inventory.Entries, syncBuildApplyDirectory); err != nil {
		return err
	}
	if err := parent.Revalidate(); err != nil {
		return err
	}
	complete = true
	return nil
}

// Publish child contents before the prepared directory, then publish its name
// in the installation parent before quiesce can become durable in the journal.
func syncBuildApplyPreparation(prepared string, entries []buildapply.InventoryEntry, syncDirectory func(string) error) error {
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].Mode.IsDir() {
			if err := syncDirectory(filepath.Join(prepared, filepath.FromSlash(entries[i].Path))); err != nil {
				return err
			}
		}
	}
	if err := syncDirectory(prepared); err != nil {
		return err
	}
	return syncDirectory(filepath.Dir(prepared))
}

type buildApplyMetadata struct {
	Mode              uint32
	Modified, Created int64
	Attributes        uint32
	Xattrs            map[string][]byte
	Streams           map[string][]byte
}

func buildApplyMetadataDigest(ctx context.Context, path string) (string, error) {
	real, err := buildapply.OpenRealRoot(path)
	if err != nil {
		return "", err
	}
	defer real.Close()
	root, err := os.OpenRoot(path)
	if err != nil {
		return "", err
	}
	defer root.Close()
	h := sha256.New()
	entries, total := 0, 0
	err = fs.WalkDir(root.FS(), ".", func(name string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		entries++
		if entries > buildapply.MaxEntries+1 {
			return buildapply.ErrPackageTooLarge
		}
		m, err := readBuildApplyNativeMetadata(filepath.Join(path, filepath.FromSlash(name)))
		if err != nil {
			return err
		}
		if name == "." {
			m.Mode = 0
		}
		data, err := json.Marshal(struct {
			Name     string
			Metadata buildApplyMetadata
		}{name, m})
		if err != nil {
			return err
		}
		total += len(data)
		if total > buildapply.MaxInventoryBytes {
			return buildapply.ErrPackageTooLarge
		}
		fmt.Fprintf(h, "%d:", len(data))
		h.Write(data)
		return nil
	})
	if err != nil {
		return "", err
	}
	if err := real.Revalidate(); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func removeBuildApplyTree(ctx context.Context, path, id string) error {
	actual, err := buildApplyDirectoryIdentity(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil || actual != id {
		return errors.New("cleanup directory identity changed")
	}
	parent, err := buildapply.OpenRealRoot(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer parent.Close()
	root, err := os.OpenRoot(path)
	if err != nil {
		return err
	}
	held, err := root.Open(".")
	if err != nil {
		root.Close()
		return err
	}
	rootID, _, identityErr := buildApplyDirectoryFileSnapshot(held)
	closeErr := held.Close()
	if err := errors.Join(identityErr, closeErr); err != nil || rootID != id {
		root.Close()
		return errors.New("opened cleanup directory identity changed")
	}
	var names []string
	err = fs.WalkDir(root.FS(), ".", func(name string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(names) > buildapply.MaxEntries {
			return buildapply.ErrPackageTooLarge
		}
		if name != "." {
			names = append(names, name)
		}
		return nil
	})
	if err != nil {
		root.Close()
		return err
	}
	for i := len(names) - 1; i >= 0; i-- {
		if err := ctx.Err(); err != nil {
			root.Close()
			return err
		}
		name := filepath.FromSlash(names[i])
		info, err := root.Lstat(name)
		if err != nil {
			root.Close()
			return err
		}
		if info.Mode()&os.ModeSymlink == 0 {
			if err := root.Chmod(name, 0o700); err != nil {
				root.Close()
				return err
			}
		}
		if err := root.Remove(name); err != nil {
			root.Close()
			return err
		}
	}
	if err := root.Chmod(".", 0o700); err != nil {
		root.Close()
		return err
	}
	if err := root.Close(); err != nil {
		return err
	}
	if err := parent.Revalidate(); err != nil {
		return err
	}
	actual, err = buildApplyDirectoryIdentity(path)
	if err != nil || actual != id {
		return errors.New("cleanup root changed")
	}
	pr, err := os.OpenRoot(parent.Path)
	if err != nil {
		return err
	}
	defer pr.Close()
	return pr.Remove(filepath.Base(path))
}

func (files *nativeBuildApplyFiles) discardPrepared(ctx context.Context, p buildApplyPlan) error {
	ctx, cancel := context.WithTimeout(ctx, buildApplyLiveBudget)
	defer cancel()
	if err := p.current.Revalidate(ctx); err != nil {
		return err
	}
	paths, err := buildapply.DeriveOperationPaths(p.receipt.Target, p.receipt.OperationID)
	if err != nil {
		return err
	}
	if files.preparedID == "" {
		return nil
	}
	return removeBuildApplyTree(ctx, paths.Prepared, files.preparedID)
}

const (
	buildApplyOperationEnv        = "NVPAIR_BUILD_APPLY_OPERATION"
	buildApplyRoleEnv             = "NVPAIR_BUILD_APPLY_ROLE"
	buildApplyLaunchEnv           = "NVPAIR_BUILD_APPLY_LAUNCH"
	buildApplyNativeReadMethod    = "service/build-apply-owner-read"
	buildApplyNativePrepareMethod = "service/build-apply-owner-prepare"
	buildApplyNativeFinishMethod  = "service/build-apply-owner-finish"
)

type buildApplyEvidenceOwner interface {
	Read(context.Context) (buildApplyEvidence, error)
}

type serviceBuildApplyEvidence struct{ service *Service }

func (owner serviceBuildApplyEvidence) Read(ctx context.Context) (buildApplyEvidence, error) {
	s := owner.service
	s.mu.Lock()
	run := s.broker
	s.mu.Unlock()
	envelope, err := s.readBuildApplyImpact(ctx, run)
	if err != nil {
		return buildApplyEvidence{}, err
	}
	roots, err := buildApplyManagerRoots(envelope)
	if err != nil {
		return buildApplyEvidence{}, err
	}
	profile, err := buildApplyProtectedRoots()
	if err != nil {
		return buildApplyEvidence{}, err
	}
	m := *envelope.ManagerEvidence.Value
	i := envelope.Impact
	e := buildApplyEvidence{Impact: i, ProtectedRoots: append(profile, roots...), ManagerCaptureToken: m.CaptureToken, ManagerIncarnation: m.ManagerIncarnation}
	authorityKnown := m.StopAuthority.Known
	if authorityKnown {
		for _, entry := range *m.StopAuthority.Value {
			if !entry.Scoped.Known {
				authorityKnown = false
				break
			}
		}
	}
	e.Facts = buildApplyOwnerCompleteness{
		IntentProvenance:         m.Intent.Known,
		LockMap:                  m.Locks.Known,
		EngineUniverse:           m.Universe.Known,
		ModelDataRoots:           true,
		ScopedStopAuthority:      authorityKnown,
		DownloadCopyReservations: m.Reservations.Known && m.Reservations.Value.BlocksApply.Known && m.Reservations.Value.UnsafeCopiesComplete,
		ManagerIdleRunID:         m.ManagerIncarnation != "",
	}
	if i.ExpectedSettings.PolicyDigest.Known {
		e.Protected.Policy = *i.ExpectedSettings.PolicyDigest.Value
	}
	if i.ExpectedSettings.EngineSettingsDigest.Known {
		e.Protected.EngineSettings = *i.ExpectedSettings.EngineSettingsDigest.Value
	}
	if i.ExpectedSettings.IntentDigest.Known && m.Intent.Known {
		e.Protected.Intent = *i.ExpectedSettings.IntentDigest.Value
	}
	if i.ExpectedSettings.LocksDigest.Known && m.Locks.Known {
		e.Protected.Locks = *i.ExpectedSettings.LocksDigest.Value
	}
	s.mu.Lock()
	current := s.broker == run && !s.stopping
	if current {
		bindBuildApplyImpact(&e.Impact, s.startedAt, s.runs)
	}
	s.mu.Unlock()
	if !current {
		return buildApplyEvidence{}, errors.New("broker changed during owner projection")
	}
	return e, nil
}

type buildApplyEvidence struct {
	Facts               buildApplyOwnerCompleteness    `json:"facts"`
	ProtectedRoots      []string                       `json:"protectedRoots"`
	ManagerCaptureToken string                         `json:"managerCaptureToken"`
	ManagerIncarnation  string                         `json:"managerIncarnation"`
	Impact              buildapply.Impact              `json:"impact"`
	Protected           buildapply.ProtectedDigests    `json:"protected"`
	Pause               buildapply.JournalPauseFence   `json:"pause"`
	Saved               bool                           `json:"saved"`
	Enforced            bool                           `json:"enforced"`
	PhysicalStop        bool                           `json:"physicalStop"`
	ActivationVerified  bool                           `json:"activationVerified"`
	RecoveryRequired    bool                           `json:"recoveryRequired"`
	OwnedWorkers        []buildapply.RunnerReservation `json:"ownedWorkers"`
}

type buildApplyOwnerCompleteness struct {
	IntentProvenance         bool `json:"intentProvenance"`
	LockMap                  bool `json:"lockMap"`
	EngineUniverse           bool `json:"engineUniverse"`
	ModelDataRoots           bool `json:"modelDataRoots"`
	ScopedStopAuthority      bool `json:"scopedStopAuthority"`
	DownloadCopyReservations bool `json:"downloadCopyReservations"`
	ManagerIdleRunID         bool `json:"managerIdleRunId"`
}

func completeBuildApplyEvidence(e buildApplyEvidence) bool {
	if e.OwnedWorkers == nil || len(e.OwnedWorkers) > 254 {
		return false
	}
	f := e.Facts
	i := e.Impact
	if !f.IntentProvenance || !f.LockMap || !f.EngineUniverse || !f.ModelDataRoots || !f.ScopedStopAuthority || !f.DownloadCopyReservations || !f.ManagerIdleRunID || e.ProtectedRoots == nil || e.RecoveryRequired || !i.Complete || len(i.Refusals) != 0 || !buildApplyFencesComplete(i.Fences) ||
		!i.Availability.Known || i.Availability.Value == nil || !i.PauseOrigin.Known || i.PauseOrigin.Value == nil || !i.Engines.Known || i.Engines.Value == nil || !i.ModelOperations.Known || i.ModelOperations.Value == nil || !i.ActiveWork.Known || i.ActiveWork.Value == nil || *i.ActiveWork.Value < 0 || !i.SettingsDifferences.Known || i.SettingsDifferences.Value == nil || !i.InterruptionPolicy.OnActive.Known || i.InterruptionPolicy.OnActive.Value == nil || !i.InterruptionPolicy.DrainTimeoutSeconds.Known || i.InterruptionPolicy.DrainTimeoutSeconds.Value == nil || !buildApplyObservedTrue(i.ExpectedSettings.StopEngines) || !buildApplyObservedTrue(i.ExpectedSettings.UnloadModels) || !i.Maintenance.BuildApplyActive.Known || i.Maintenance.BuildApplyActive.Value == nil || !i.Maintenance.ModelSettingsActive.Known || i.Maintenance.ModelSettingsActive.Value == nil || *i.Maintenance.ModelSettingsActive.Value {
		return false
	}
	for _, digest := range []string{e.Protected.Policy, e.Protected.EngineSettings, e.Protected.Intent, e.Protected.Locks, e.Protected.IdentityMembership} {
		if !validBuildApplyDigest(digest) {
			return false
		}
	}
	if !i.ExpectedSettings.PolicyDigest.Known || i.ExpectedSettings.PolicyDigest.Value == nil || !validBuildApplyDigest(*i.ExpectedSettings.PolicyDigest.Value) {
		return false
	}
	for _, pair := range []struct {
		observation buildapply.Observation[string]
		digest      string
	}{{i.ExpectedSettings.EngineSettingsDigest, e.Protected.EngineSettings}, {i.ExpectedSettings.IntentDigest, e.Protected.Intent}, {i.ExpectedSettings.LocksDigest, e.Protected.Locks}} {
		if !pair.observation.Known || pair.observation.Value == nil || *pair.observation.Value != pair.digest {
			return false
		}
	}
	for _, engine := range *i.Engines.Value {
		if !buildApplyObservedTrue(engine.ScopedStopAuthority) || !engine.Running.Known || engine.Running.Value == nil || !engine.Active.Known || engine.Active.Value == nil || !engine.LoadedModels.Known || engine.LoadedModels.Value == nil || !engine.DesiredState.Known || engine.DesiredState.Value == nil || !engine.Locked.Known || engine.Locked.Value == nil {
			return false
		}
	}
	for _, op := range *i.ModelOperations.Value {
		if op.BlocksApply {
			return false
		}
	}
	return true
}

type buildApplyNativeRequest struct {
	OperationID string                       `json:"operationId"`
	Runner      buildapply.RunnerReservation `json:"runner"`
}
type buildApplyOwnerCommand struct {
	OperationID string                       `json:"operationId"`
	Runner      buildapply.RunnerReservation `json:"runner"`
	Prepare     *buildapply.PrepareRequest   `json:"prepare"`
	Finish      *buildapply.FinishRequest    `json:"finish"`
}
type buildApplyNativeSnapshot struct {
	Process     buildapply.RunnerReservation  `json:"process"`
	Identity    buildapply.Identity           `json:"identity"`
	BrokerReady bool                          `json:"brokerReady"`
	Evidence    *buildApplyEvidence           `json:"evidence"`
	GUI         *buildapply.RunnerReservation `json:"gui"`
}

func cleanBuildApplyEnvironment(env []string) []string {
	result := make([]string, 0, len(env))
	for _, entry := range env {
		key, _, _ := strings.Cut(entry, "=")
		if !strings.HasPrefix(key, "NVPAIR_BUILD_APPLY_") {
			result = append(result, entry)
		}
	}
	return result
}
func buildApplyEnvironmentDigest(env []string) string {
	env = profileBuildApplyEnvironment(cleanBuildApplyEnvironment(env))
	data, _ := json.Marshal(env)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}
func profileBuildApplyEnvironment(env []string) []string {
	var profile []string
	for _, entry := range env {
		key, _, _ := strings.Cut(entry, "=")
		switch key {
		case "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "NVPAIR_SERVICE_ENDPOINT":
			profile = append(profile, entry)
		}
	}
	sort.Strings(profile)
	return profile
}
func mergeBuildApplyEnvironment(base, profile []string) []string {
	values := map[string]string{}
	for _, entry := range append(cleanBuildApplyEnvironment(base), profile...) {
		key, _, ok := strings.Cut(entry, "=")
		if ok {
			values[key] = entry
		}
	}
	env := make([]string, 0, len(values))
	for _, entry := range values {
		env = append(env, entry)
	}
	sort.Strings(env)
	return env
}

func buildApplyProcessGone(ctx context.Context, want buildapply.RunnerReservation) (bool, error) {
	if want.PID <= 0 || want.Identity.Created == "" {
		return false, errors.New("missing process lifetime")
	}
	exists, err := process.PidExistsWithContext(ctx, int32(want.PID))
	if err != nil {
		return false, err
	}
	if !exists {
		return true, nil
	}
	actual, err := inspectBuildApplyProcess(ctx, want.PID)
	if err != nil {
		return false, err
	}
	return actual != want, nil
}

func readBuildApplyReceipt(id string) (buildapply.Receipt, error) {
	state, err := buildapply.ReadJournal()
	if err != nil {
		return buildapply.Receipt{}, err
	}
	r := state.Current
	if r == nil {
		r = state.LastTerminal
	}
	if r == nil || r.OperationID != id {
		return buildapply.Receipt{}, buildapply.ErrRecoveryNeeded
	}
	return *r, nil
}

func validateBuildApplySuccessor(r buildapply.Receipt, self buildapply.RunnerReservation, token string, args, env []string, now time.Time) error {
	if r.Outcome != nil || !now.Before(r.Deadline()) || !r.RunnerReady || r.Runner == nil || r.Handoff != buildapply.HandoffAcknowledged ||
		r.Rename != buildapply.RenameInstalled || buildApplyPhaseIndex(r.Phase) < buildApplyPhaseIndex(buildapply.PhaseRestart) ||
		!r.ServiceLaunch.Intent || r.ServiceLaunch.Token != token || r.ServiceLaunch.Process == nil || *r.ServiceLaunch.Process != self ||
		buildApplyEnvironmentDigest(env) != r.Runtime.ServiceLaunch.EnvironmentSHA256 || !slices.Equal(args, r.Runtime.ServiceLaunch.Args) {
		return buildapply.ErrRecoveryNeeded
	}
	for _, old := range r.Runtime.OldProcesses {
		if old == self {
			return buildapply.ErrRecoveryNeeded
		}
	}
	return nil
}

func checkBuildApplyProcessEntry(ctx context.Context) error {
	id, role := os.Getenv(buildApplyOperationEnv), os.Getenv(buildApplyRoleEnv)
	if id == "" {
		return buildApplyStartupGate()
	}
	state, err := buildapply.ReadJournal()
	if err != nil {
		return err
	}
	if state.Current == nil && !state.TerminalPending {
		return nil
	}
	if !buildapply.ValidOperationID(id) || role != "service" {
		return buildapply.ErrRecoveryNeeded
	}
	ctx, cancel := context.WithTimeout(ctx, buildApplyLiveBudget)
	defer cancel()
	self, err := inspectBuildApplyProcess(ctx, os.Getpid())
	if err != nil {
		return err
	}
	for {
		r, err := readBuildApplyReceipt(id)
		if err != nil {
			return err
		}
		if !time.Now().Before(r.Deadline()) || !r.RunnerReady || r.Handoff != buildapply.HandoffAcknowledged || !r.ServiceLaunch.Intent || r.ServiceLaunch.Token != os.Getenv(buildApplyLaunchEnv) {
			return buildapply.ErrRecoveryNeeded
		}
		if r.ServiceLaunch.Process != nil {
			if err := validateBuildApplySuccessor(r, self, os.Getenv(buildApplyLaunchEnv), os.Args[1:], os.Environ(), time.Now()); err != nil {
				return err
			}
			id, err := buildApplyDirectoryIdentity(r.Target)
			if err != nil || id != r.PreparedIdentity {
				return buildapply.ErrRecoveryNeeded
			}
			if r.Runner == nil {
				return buildapply.ErrRecoveryNeeded
			}
			gone, err := buildApplyProcessGone(ctx, *r.Runner)
			if err != nil || gone {
				return buildapply.ErrRecoveryNeeded
			}
			sha, err := buildApplyFileSHA(ctx, self.Identity.Executable)
			if err != nil {
				return err
			}
			identity := currentBuildApplyIdentity()
			identity.SHA256 = buildapply.Known(sha)
			if !reflect.DeepEqual(identity, r.Runtime.ExpectedService) {
				return buildapply.ErrRecoveryNeeded
			}
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(25 * time.Millisecond):
		}
	}
}

func buildApplyExecutorAllowed(r buildapply.Receipt, executor buildapply.RunnerReservation) bool {
	if r.RunnerReady && r.Runner != nil && *r.Runner == executor {
		return true
	}
	return r.Outcome != nil && *r.Outcome == buildapply.OutcomeSucceeded && r.CleanupRunner != nil && *r.CleanupRunner == executor && time.Now().Before(r.CleanupDeadline)
}

func buildApplyCallerReservation(ctx context.Context, r buildapply.Receipt) (buildapply.RunnerReservation, error) {
	self, err := inspectBuildApplyProcess(ctx, os.Getpid())
	if err != nil {
		return self, err
	}
	if !buildApplyExecutorAllowed(r, self) {
		return self, buildapply.ErrRecoveryNeeded
	}
	return self, nil
}
func (s *Service) checkBuildApplyEntry(ctx context.Context) error {
	return checkBuildApplyProcessEntry(ctx)
}

func (s *Service) checkBuildApplyStop(params json.RawMessage) error {
	state, err := buildapply.ReadJournal()
	if err != nil {
		return err
	}
	if state.Current == nil && !state.TerminalPending {
		return nil
	}
	var request buildApplyNativeRequest
	if err := decodeBuildApplyNativeRequest(params, &request); err != nil {
		return err
	}
	r, err := readBuildApplyReceipt(request.OperationID)
	if err != nil {
		return err
	}
	if !r.RunnerReady || r.Runner == nil || *r.Runner != request.Runner || r.Phase != buildapply.PhaseShutdown {
		return buildapply.ErrRecoveryNeeded
	}
	self, err := inspectBuildApplyProcess(context.Background(), os.Getpid())
	if err != nil || self != r.Runtime.OldService {
		return buildapply.ErrRecoveryNeeded
	}
	actual, err := inspectBuildApplyProcess(context.Background(), request.Runner.PID)
	if err != nil || actual != request.Runner {
		return buildapply.ErrRecoveryNeeded
	}
	return nil
}

func decodeBuildApplyNativeRequest(data []byte, value *buildApplyNativeRequest) error {
	if len(data) == 0 || len(data) > buildapply.MaxRequestBytes {
		return errors.New("invalid helper request")
	}
	d := json.NewDecoder(strings.NewReader(string(data)))
	d.DisallowUnknownFields()
	if err := d.Decode(value); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return errors.New("trailing helper request")
	}
	if !buildapply.ValidOperationID(value.OperationID) {
		return errors.New("invalid operation identity")
	}
	return nil
}

func (s *Service) respondBuildApplyNativeRead(c *client, msg *jsonrpc.Message) {
	var request buildApplyNativeRequest
	if err := decodeBuildApplyNativeRequest(msg.Params, &request); err != nil {
		c.respondError(msg.ID, -32602, "Invalid helper read.")
		return
	}
	r, err := readBuildApplyReceipt(request.OperationID)
	if err != nil || !buildApplyExecutorAllowed(r, request.Runner) {
		c.respond(msg.ID, buildapply.Unavailable[buildApplyNativeSnapshot]())
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), buildApplyLiveBudget)
	defer cancel()
	actual, err := inspectBuildApplyProcess(ctx, request.Runner.PID)
	if err != nil || actual != request.Runner {
		c.respond(msg.ID, buildapply.Unavailable[buildApplyNativeSnapshot]())
		return
	}
	self, err := inspectBuildApplyProcess(ctx, os.Getpid())
	if err != nil {
		c.respond(msg.ID, buildapply.Unavailable[buildApplyNativeSnapshot]())
		return
	}
	cleanupRead := r.Outcome != nil && *r.Outcome == buildapply.OutcomeSucceeded && r.CleanupRunner != nil && *r.CleanupRunner == request.Runner
	if self != r.Runtime.OldService && (r.ServiceLaunch.Process == nil || self != *r.ServiceLaunch.Process) && !cleanupRead {
		c.respond(msg.ID, buildapply.Unavailable[buildApplyNativeSnapshot]())
		return
	}
	snapshot := buildApplyNativeSnapshot{Process: self, Identity: currentBuildApplyIdentity()}
	layout, _ := buildapply.LayoutFor(runtime.GOOS)
	sha, err := buildApplyFileSHA(ctx, filepath.Join(r.Target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable))
	if err != nil {
		c.respond(msg.ID, buildapply.Unavailable[buildApplyNativeSnapshot]())
		return
	}
	snapshot.Identity.SHA256 = buildapply.Known(sha)
	if cleanupRead && (!reflect.DeepEqual(snapshot.Identity, r.Runtime.ExpectedService) || !sameBuildApplyImage(self.Identity.Executable, filepath.Join(r.Target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable)) || s.cfg.endpoint != r.Runtime.Endpoint) {
		c.respond(msg.ID, buildapply.Unavailable[buildApplyNativeSnapshot]())
		return
	}
	s.mu.Lock()
	snapshot.BrokerReady = len(s.readyFrame) != 0
	snapshot.GUI = s.buildGUIReady
	s.mu.Unlock()
	if evidence, err := s.cfg.buildApplyOwners.Read(ctx); err == nil {
		snapshot.Evidence = &evidence
		result := buildapply.Result[buildApplyNativeSnapshot]{Available: true, Value: &snapshot}
		data, encodeErr := json.Marshal(result)
		if encodeErr != nil || len(data) > buildapply.MaxStatusBytes {
			snapshot.Evidence = nil
		}
	}
	c.respond(msg.ID, buildapply.Result[buildApplyNativeSnapshot]{Available: true, Value: &snapshot})
}

func (s *Service) callBuildApplyBrokerCommand(ctx context.Context, method string, request any) (json.RawMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if method != "build:apply-prepare" && method != "build:apply-finish" && method != "build:apply-impact" {
		return nil, errors.New("invalid maintenance owner method")
	}
	params, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	ch := make(chan *jsonrpc.Message, 1)
	s.mu.Lock()
	run := s.broker
	if run == nil || s.stopping {
		s.mu.Unlock()
		return nil, errors.New("broker owner unavailable")
	}
	s.nextID++
	id := s.nextID
	s.pending[id] = &pendingRequest{method: method, run: run, internal: func(m *jsonrpc.Message) {
		select {
		case ch <- m:
		default:
		}
	}}
	s.mu.Unlock()
	defer func() { s.mu.Lock(); delete(s.pending, id); s.mu.Unlock() }()
	go func() {
		if err := run.writeMessageContext(ctx, &jsonrpc.Message{JSONRPC: "2.0", ID: rawID(id), Method: method, Params: params}); err != nil {
			select {
			case ch <- nil:
			default:
			}
		}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case reply := <-ch:
		if reply == nil || reply.Error != nil {
			return nil, errors.New("broker maintenance request unavailable")
		}
		s.mu.Lock()
		current := s.broker == run && !s.stopping
		s.mu.Unlock()
		if !current {
			return nil, errors.New("broker changed during maintenance request")
		}
		return reply.Result, nil
	}
}

func (s *Service) respondBuildApplyOwnerCommand(c *client, msg *jsonrpc.Message) {
	if len(msg.Params) > buildapply.MaxRequestBytes {
		c.respondError(msg.ID, -32602, "Invalid helper owner request.")
		return
	}
	var request buildApplyOwnerCommand
	d := json.NewDecoder(strings.NewReader(string(msg.Params)))
	d.DisallowUnknownFields()
	if err := d.Decode(&request); err != nil || d.Decode(new(any)) != io.EOF {
		c.respondError(msg.ID, -32602, "Invalid helper owner request.")
		return
	}
	r, err := readBuildApplyReceipt(request.OperationID)
	if err != nil || !buildApplyExecutorAllowed(r, request.Runner) {
		c.respond(msg.ID, buildapply.Unavailable[buildapply.BuildApplyOperation]())
		return
	}
	deadline := r.Deadline()
	if r.CleanupRunner != nil && *r.CleanupRunner == request.Runner {
		deadline = r.CleanupDeadline
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	actual, err := inspectBuildApplyProcess(ctx, request.Runner.PID)
	if err != nil || actual != request.Runner {
		c.respond(msg.ID, buildapply.Unavailable[buildapply.BuildApplyOperation]())
		return
	}
	method := "build:apply-prepare"
	var ownerRequest any = request.Prepare
	if msg.Method == buildApplyNativePrepareMethod {
		if request.Prepare == nil || request.Finish != nil || request.Prepare.OperationID != r.OperationID || request.Prepare.PreviewID != r.PreviewID || r.Outcome != nil {
			c.respondError(msg.ID, -32602, "Uncorrelated prepare request.")
			return
		}
	} else {
		if request.Finish == nil || request.Prepare != nil || request.Finish.OperationID != r.OperationID || r.Phase != buildapply.PhaseFinish {
			c.respondError(msg.ID, -32602, "Uncorrelated finish request.")
			return
		}
		method = "build:apply-finish"
		ownerRequest = request.Finish
		if r.Outcome != nil && *r.Outcome == buildapply.OutcomeSucceeded {
			// The terminal receipt precedes reservation release. Publish the
			// applied result even if the subsequent cleanup reply is lost.
			s.publishBuildApplyChanged(projectBuildApplyReceipt(r))
		}
	}
	data, err := s.callBuildApplyBrokerCommand(ctx, method, ownerRequest)
	if err != nil {
		c.respond(msg.ID, buildapply.Unavailable[buildapply.BuildApplyOperation]())
		return
	}
	c.respond(msg.ID, data)
}

func (s *Service) respondBuildApplyGUIReady(c *client, id *json.RawMessage, request buildapply.GUIReadyRequest) {
	ctx, cancel := context.WithTimeout(context.Background(), buildApplyLiveBudget)
	defer cancel()
	r, err := readBuildApplyReceipt(request.OperationID)
	if err != nil || r.GUILaunch.Process == nil || r.Runtime.ExpectedGUI.BuildID.Value == nil || *r.Runtime.ExpectedGUI.BuildID.Value != request.GUI.BuildID || request.GUI.PID != r.GUILaunch.Process.PID || !sameBuildApplyImage(request.GUI.Executable, r.GUILaunch.Process.Identity.Executable) {
		c.respond(id, buildapply.Unavailable[buildapply.BuildApplyOperation]())
		return
	}
	actual, err := inspectBuildApplyProcess(ctx, request.GUI.PID)
	if err != nil || actual != *r.GUILaunch.Process {
		c.respond(id, buildapply.Unavailable[buildapply.BuildApplyOperation]())
		return
	}
	s.mu.Lock()
	s.buildGUIReady = &actual
	s.mu.Unlock()
	for {
		r, err = readBuildApplyReceipt(request.OperationID)
		if err != nil {
			break
		}
		if r.GUILaunch.Acknowledged {
			op := projectBuildApplyReceipt(r)
			c.respond(id, buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &op})
			return
		}
		select {
		case <-ctx.Done():
			c.respond(id, buildapply.Unavailable[buildapply.BuildApplyOperation]())
			return
		case <-time.After(25 * time.Millisecond):
		}
	}
	c.respond(id, buildapply.Unavailable[buildapply.BuildApplyOperation]())
}

func buildApplyFileSHA(ctx context.Context, path string) (string, error) {
	root, err := buildapply.OpenRealRoot(filepath.Dir(path))
	if err != nil {
		return "", err
	}
	defer root.Close()
	r, err := os.OpenRoot(root.Path)
	if err != nil {
		return "", err
	}
	defer r.Close()
	info, err := r.Lstat(filepath.Base(path))
	if err != nil || !info.Mode().IsRegular() || info.Size() > buildapply.MaxFileBytes {
		return "", errors.New("image is not a bounded real file")
	}
	f, err := r.Open(filepath.Base(path))
	if err != nil {
		return "", err
	}
	h := sha256.New()
	n, readErr := io.Copy(h, io.LimitReader(buildApplyContextReader{ctx, f}, info.Size()+1))
	held, statErr := f.Stat()
	closeErr := f.Close()
	named, nameErr := r.Lstat(filepath.Base(path))
	if err := errors.Join(readErr, statErr, closeErr, nameErr); err != nil {
		return "", err
	}
	if n != info.Size() || !os.SameFile(info, held) || !os.SameFile(info, named) || !held.ModTime().Equal(info.ModTime()) || !named.ModTime().Equal(info.ModTime()) {
		return "", errors.New("image changed while hashing")
	}
	if err := root.Revalidate(); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func validateBuildApplyArguments(args []string, gui bool) error {
	values := map[string]bool{"endpoint": true, "log-dir": true, "log-level": true, "broker-path": true, "ipc": true, "scanner-path": true, "scanner-mdns": true, "scanner-mdns-service": true, "manual-ollama-port": true, "manual-lmstudio-port": true, "manual-node-info-port": true, "node-info-path": true, "proxy-path": true, "proxy-engines": true, "workload-manager-path": true, "errors-path": true, "engine-manager-path": true, "engine-manager-http-port": true, "engine-manager-control-port": true, "engine-manager-control-host": true, "node-info-port": true, "errors-port": true, "workload-manager-port": true, "cluster-manager-port": true, "manual-nodes-path": true, "settings-path": true, "cluster-manager-path": true, "scheduler-path": true, "cluster-dir": true}
	if gui {
		values = map[string]bool{"user-data-dir": true, "profile-directory": true}
	}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" && !gui {
			continue
		}
		if gui && arg == "--hidden" {
			continue
		}
		if !strings.HasPrefix(arg, "-") {
			return errors.New("unrecognized existing launch argument")
		}
		key, _, equal := strings.Cut(strings.TrimLeft(arg, "-"), "=")
		if !values[key] {
			return errors.New("existing launch argument is outside the profile/lifecycle allowlist")
		}
		if !gui && key == "scanner-mdns" && !equal {
			continue // Go boolean flags use their bare spelling or =false.
		}
		if !equal {
			i++
			if i >= len(args) || strings.HasPrefix(args[i], "--") {
				return errors.New("missing existing launch argument value")
			}
		}
	}
	return nil
}

func validateBuildApplyImageArguments(target string, args []string) error {
	layout, ok := buildapply.LayoutFor(runtime.GOOS)
	if !ok {
		return errors.New("unsupported target layout")
	}
	images := map[string]string{"broker-path": brokerName, "scanner-path": "nvpair-node-scanner", "node-info-path": "nvpair-node-info", "proxy-path": "nvpair-proxy", "workload-manager-path": "nvpair-workload-manager", "errors-path": "nvpair-errors", "engine-manager-path": "nvpair-engine-manager", "manual-nodes-path": "nvpair-manual-nodes", "settings-path": "nvpair-node-settings", "cluster-manager-path": "nvpair-cluster-manager", "scheduler-path": "nvpair-job-scheduler"}
	for i := 0; i < len(args); i++ {
		key, value, equal := strings.Cut(strings.TrimLeft(args[i], "-"), "=")
		image, exists := images[key]
		if !exists {
			continue
		}
		if !equal {
			i++
			if i >= len(args) {
				return errors.New("missing trusted image argument")
			}
			value = args[i]
		}
		if value == "" {
			continue // The ordinary resolver selects the canonical sibling.
		}
		if runtime.GOOS == "windows" {
			image += ".exe"
		}
		if !filepath.IsAbs(value) || !sameBuildApplyImage(value, filepath.Join(target, filepath.FromSlash(layout.ServiceDir), image)) {
			return errors.New("a preserved worker override escapes the selected bundle")
		}
	}
	return nil
}

func captureBuildApplyChildren(ctx context.Context, roots []buildapply.RunnerReservation) ([]buildapply.RunnerReservation, error) {
	all := append([]buildapply.RunnerReservation(nil), roots...)
	seen := map[int]bool{}
	for _, r := range roots {
		seen[r.PID] = true
	}
	for i := 0; i < len(all); i++ {
		if len(all) > 256 {
			return nil, errors.New("owned process inventory exceeds bound")
		}
		parent, err := process.NewProcessWithContext(ctx, int32(all[i].PID))
		if err != nil {
			return nil, err
		}
		children, err := parent.ChildrenWithContext(ctx)
		if err != nil {
			return nil, err
		}
		for _, child := range children {
			if !seen[int(child.Pid)] {
				identity, err := inspectBuildApplyProcess(ctx, int(child.Pid))
				if err != nil {
					return nil, err
				}
				all = append(all, identity)
				seen[identity.PID] = true
			}
		}
		after, err := inspectBuildApplyProcess(ctx, all[i].PID)
		if err != nil || after != all[i] {
			return nil, errors.New("owned process changed during inventory")
		}
	}
	return all, nil
}

func (s *Service) captureBuildApplyPlan(ctx context.Context, snapshot buildApplyAdmissionSnapshot, request buildapply.StartRequest) (p buildApplyPlan, err error) {
	// Read this missing production owner first. No package, runner, policy or
	// journal mutation is reachable while any manager fact is unavailable.
	evidence, err := s.cfg.buildApplyOwners.Read(ctx)
	if err != nil {
		return p, err
	}
	if !completeBuildApplyEvidence(evidence) || evidence.Impact.ExpectedSettings.PolicyDigest.Value == nil || *evidence.Impact.ExpectedSettings.PolicyDigest.Value != evidence.Protected.Policy || buildApplyObservedTrue(evidence.Impact.Maintenance.BuildApplyActive) || !reflect.DeepEqual(evidence.Impact.Fences, snapshot.preview.Impact.Fences) {
		return p, errors.New("required live owner facts are incomplete or changed")
	}
	brokerData, err := s.callBuildApplyBrokerCommand(ctx, "build:apply-impact", buildapply.ImpactRequest{})
	if err != nil {
		return p, err
	}
	var brokerImpact buildapply.Result[buildapply.ImpactEnvelope]
	if len(brokerData) > buildapply.MaxStatusBytes || json.Unmarshal(brokerData, &brokerImpact) != nil || !brokerImpact.Available || brokerImpact.Value == nil || !brokerImpact.Value.Impact.Complete {
		return p, errors.New("broker owner facts are unavailable or changed")
	}
	s.mu.Lock()
	bindBuildApplyImpact(&brokerImpact.Value.Impact, s.startedAt, s.runs)
	s.mu.Unlock()
	if !reflect.DeepEqual(brokerImpact.Value.Impact.Fences, evidence.Impact.Fences) {
		return p, errors.New("broker owner fences changed")
	}
	managerRoots, err := buildApplyManagerRoots(*brokerImpact.Value)
	if err != nil {
		return p, err
	}
	if len(evidence.ProtectedRoots) < len(managerRoots) || brokerImpact.Value.ManagerEvidence.Value.CaptureToken != evidence.ManagerCaptureToken ||
		brokerImpact.Value.ManagerEvidence.Value.ManagerIncarnation != evidence.ManagerIncarnation ||
		!reflect.DeepEqual(managerRoots, evidence.ProtectedRoots[len(evidence.ProtectedRoots)-len(managerRoots):]) {
		return p, errors.New("manager root capture changed")
	}
	excluded := evidence.ProtectedRoots
	if err := buildapply.ValidateLocations(snapshot.candidatePath, snapshot.targetPath, excluded); err != nil {
		return p, err
	}
	defer func() {
		if err != nil {
			if p.candidate != nil {
				p.candidate.Close()
			}
			if p.current != nil {
				p.current.Close()
			}
			p.candidate = nil
			p.current = nil
		}
	}()
	p.candidate, err = buildapply.InspectPackage(ctx, snapshot.candidatePath, runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return p, err
	}
	p.current, err = buildapply.InspectPackage(ctx, snapshot.targetPath, runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return p, err
	}
	if snapshot.targetSHA256 == "" || p.current.SHA256 != snapshot.targetSHA256 {
		return p, errors.New("target changed since preview")
	}
	if snapshot.preview.PackageSHA256.Value == nil || p.candidate.SHA256 != *snapshot.preview.PackageSHA256.Value {
		return p, errors.New("candidate changed")
	}
	if err = buildapply.CheckTargetParent(p.current.Root, p.candidate.Bytes); err != nil {
		return p, err
	}
	gui, err := inspectBuildApplyProcess(ctx, request.GUI.PID)
	if err != nil || gui.Identity != snapshot.gui {
		return p, errors.New("native GUI lifetime changed")
	}
	service, err := inspectBuildApplyProcess(ctx, os.Getpid())
	if err != nil {
		return p, err
	}
	layout, _ := buildapply.LayoutFor(runtime.GOOS)
	if !sameBuildApplyImage(service.Identity.Executable, filepath.Join(snapshot.targetPath, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable)) {
		return p, errors.New("service no longer belongs to the selected target")
	}
	roots := []buildapply.RunnerReservation{gui, service}
	for _, worker := range evidence.OwnedWorkers {
		actual, err := inspectBuildApplyProcess(ctx, worker.PID)
		if err != nil || actual != worker {
			return p, errors.New("an owned worker lifetime changed")
		}
		duplicate := false
		for _, root := range roots {
			if root.PID == worker.PID {
				if root != worker {
					return p, errors.New("conflicting owned worker lifetime")
				}
				duplicate = true
			}
		}
		if !duplicate {
			roots = append(roots, worker)
		}
	}
	old, err := captureBuildApplyChildren(ctx, roots)
	if err != nil {
		return p, err
	}
	guiProcess, err := process.NewProcessWithContext(ctx, int32(gui.PID))
	if err != nil {
		return p, err
	}
	guiArgs, err := guiProcess.CmdlineSliceWithContext(ctx)
	if err != nil || len(guiArgs) == 0 {
		return p, errors.New("GUI launch arguments unavailable")
	}
	var savedGUIArgs []string
	for _, arg := range guiArgs[1:] {
		if !strings.HasPrefix(arg, "--pair-build-apply=") {
			savedGUIArgs = append(savedGUIArgs, arg)
		}
	}
	serviceArgs := append([]string(nil), os.Args[1:]...)
	if err = validateBuildApplyArguments(serviceArgs, false); err != nil {
		return p, err
	}
	if err = validateBuildApplyImageArguments(snapshot.targetPath, serviceArgs); err != nil {
		return p, err
	}
	if err = validateBuildApplyArguments(savedGUIArgs, true); err != nil {
		return p, err
	}
	guiEnv, err := guiProcess.EnvironWithContext(ctx)
	if err != nil {
		return p, err
	}
	serviceEnv := mergeBuildApplyEnvironment(os.Environ(), profileBuildApplyEnvironment(os.Environ()))
	newGUIEnv := mergeBuildApplyEnvironment(serviceEnv, profileBuildApplyEnvironment(guiEnv))
	opID, err := buildapply.NewOperationID()
	if err != nil {
		return p, err
	}
	inventory, err := buildapply.InventoryFromPackage(opID, p.candidate)
	if err != nil {
		return p, err
	}
	targetID, err := buildApplyDirectoryIdentity(snapshot.targetPath)
	if err != nil {
		return p, err
	}
	metadata, err := buildApplyMetadataDigest(ctx, snapshot.candidatePath)
	if err != nil {
		return p, err
	}
	if p.current.Service.SHA256.Value == nil {
		return p, errors.New("trusted runner digest unavailable")
	}
	now := time.Now().UTC()
	p.receipt = buildapply.Receipt{OperationID: opID, PreviewID: request.PreviewID, Target: snapshot.targetPath, CandidateSHA256: p.candidate.SHA256, PreviousSHA256: p.current.SHA256, Protected: evidence.Protected, Consent: request.Consent, Fences: evidence.Impact.Fences, Phase: buildapply.PhasePrepare, Rename: buildapply.RenameNone, Handoff: buildapply.HandoffNone}
	impact := evidence.Impact
	if impact.InterruptionPolicy.OnActive.Value == nil || impact.InterruptionPolicy.DrainTimeoutSeconds.Value == nil {
		return p, errors.New("saved drain policy unavailable")
	}
	p.receipt.Expected = buildapply.JournalExpectations{StopEngines: true, UnloadModels: true, RemainPaused: true, PreserveIntentLocks: true, OnActive: *impact.InterruptionPolicy.OnActive.Value, DrainTimeoutSeconds: *impact.InterruptionPolicy.DrainTimeoutSeconds.Value}
	if impact.Availability.Value == nil {
		return p, errors.New("availability owner unavailable")
	}
	p.receipt.Runtime = buildapply.RuntimeAuthority{AcceptedAt: now, Deadline: now.Add(buildApplyTotalBudget), CandidatePath: snapshot.candidatePath, Endpoint: s.cfg.endpoint, OldProcesses: old, OldGUI: gui, OldService: service, ServiceLaunch: buildapply.LaunchContext{Args: serviceArgs, Environment: profileBuildApplyEnvironment(serviceEnv), EnvironmentSHA256: buildApplyEnvironmentDigest(serviceEnv)}, GUILaunch: buildapply.LaunchContext{Args: savedGUIArgs, Environment: profileBuildApplyEnvironment(guiEnv), EnvironmentSHA256: buildApplyEnvironmentDigest(newGUIEnv)}, ExpectedService: p.candidate.Service, ExpectedGUI: p.candidate.GUI, RunnerSHA256: *p.current.Service.SHA256.Value, TargetIdentity: targetID, CandidateMetadataSHA256: metadata, WasPaused: *impact.Availability.Value == nodepolicy.Paused}
	p.inventory = inventory
	p.receipt.RunnerToken = opID
	p.accepted = now
	p.deadline = p.receipt.Runtime.Deadline
	p.expires = snapshot.expires
	p.ownerFactsComplete = true
	p.oldProcesses = old
	p.serviceLaunch = buildApplyLaunch{executable: service.Identity.Executable, args: serviceArgs, env: serviceEnv}
	p.guiLaunch = buildApplyLaunch{executable: gui.Identity.Executable, args: savedGUIArgs, env: newGUIEnv}
	p.expectedService = p.candidate.Service
	p.expectedGUI = p.candidate.GUI
	last, err := s.cfg.buildApplyOwners.Read(ctx)
	if err != nil || !reflect.DeepEqual(last, evidence) {
		return p, errors.New("owner facts changed during live admission")
	}
	if err = buildapply.ValidateLocations(snapshot.candidatePath, snapshot.targetPath, last.ProtectedRoots); err != nil {
		return p, err
	}
	if err = buildapply.ValidateRuntimeAuthority(p.receipt.Target, p.receipt.Runtime); err != nil {
		return p, err
	}
	if err = p.current.Revalidate(ctx); err != nil {
		return p, err
	}
	err = p.candidate.Revalidate(ctx)
	return p, err
}

func (s *Service) executeBuildApplyPlan(ctx context.Context, p buildApplyPlan) buildapply.Result[buildapply.BuildApplyOperation] {
	if err := p.validate(time.Now()); err != nil {
		return buildapply.Unavailable[buildapply.BuildApplyOperation]()
	}
	j, err := buildapply.OpenJournal(true)
	if err != nil {
		return buildapply.Unavailable[buildapply.BuildApplyOperation]()
	}
	defer j.Close()
	lock, err := j.TryLock()
	if err != nil {
		return buildapply.MaintenanceBusy[buildapply.BuildApplyOperation]()
	}
	defer func() {
		if lock != nil {
			_ = lock.Close()
		}
	}()
	state, err := j.Read()
	if err != nil || state.Current != nil || state.TerminalPending {
		return buildapply.Unavailable[buildapply.BuildApplyOperation]()
	}
	if state.LastTerminal != nil {
		last := state.LastTerminal
		if last.CleanupPending {
			return refusedBuildApply("cleanup_pending", "Explicitly recover the previous operation's retained cleanup before another apply.")
		}
		if last.Runner != nil {
			gone, err := buildApplyProcessGone(ctx, *last.Runner)
			if err != nil || !gone {
				return buildapply.MaintenanceBusy[buildapply.BuildApplyOperation]()
			}
		}
		if err := j.RemoveRunner(lock, last.Runtime.RunnerSHA256); err != nil {
			return buildapply.Unavailable[buildapply.BuildApplyOperation]()
		}
	}
	store := &buildApplyJournalStore{journal: j, lock: lock}
	tx := &buildApplyTransaction{plan: p, store: store, handoffOnly: true}
	tx.files = &nativeBuildApplyFiles{journal: j, lock: lock, transaction: tx}
	native := &nativeBuildApplyLifecycle{tx: tx, store: store, service: s}
	tx.owners = native
	_, err = tx.Run(ctx)
	if err != nil {
		return operationResultFromJournal(j, p.receipt.OperationID)
	}
	if err := lock.Close(); err != nil {
		return buildapply.Unavailable[buildapply.BuildApplyOperation]()
	}
	lock = nil
	store.lock = nil
	wait, cancel := context.WithTimeout(ctx, buildApplyLiveBudget)
	defer cancel()
	for {
		state, err := j.Read()
		if err != nil {
			return buildapply.Unavailable[buildapply.BuildApplyOperation]()
		}
		r := state.Current
		if r == nil {
			r = state.LastTerminal
		}
		if r != nil && r.OperationID == p.receipt.OperationID && r.RunnerReady {
			op := projectBuildApplyReceipt(*r)
			s.publishBuildApplyChanged(op)
			return buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &op}
		}
		select {
		case <-wait.Done():
			return operationResultFromJournal(j, p.receipt.OperationID)
		case <-time.After(25 * time.Millisecond):
		}
	}
}

func operationResultFromJournal(j *buildapply.Journal, id string) buildapply.Result[buildapply.BuildApplyOperation] {
	state, err := j.Read()
	if err != nil {
		return buildapply.Unavailable[buildapply.BuildApplyOperation]()
	}
	r := state.Current
	if r == nil {
		r = state.LastTerminal
	}
	if r == nil || r.OperationID != id {
		return buildapply.Unavailable[buildapply.BuildApplyOperation]()
	}
	op := projectBuildApplyReceipt(*r)
	return buildapply.Result[buildapply.BuildApplyOperation]{Available: true, Value: &op}
}

type nativeBuildApplyLifecycle struct {
	tx         *buildApplyTransaction
	store      *buildApplyJournalStore
	service    *Service
	recovering bool
}

func (n *nativeBuildApplyLifecycle) snapshot(ctx context.Context) (buildApplyNativeSnapshot, error) {
	r := n.tx.receipt
	if r.Runner == nil {
		return buildApplyNativeSnapshot{}, errors.New("runner reservation missing")
	}
	data, err := callBuildApplyService(ctx, r.Runtime.Endpoint, buildApplyNativeReadMethod, buildApplyNativeRequest{OperationID: r.OperationID, Runner: *r.Runner})
	if err != nil {
		return buildApplyNativeSnapshot{}, err
	}
	var result buildapply.Result[buildApplyNativeSnapshot]
	if len(data) > buildapply.MaxStatusBytes || json.Unmarshal(data, &result) != nil || !result.Available || result.Value == nil {
		return buildApplyNativeSnapshot{}, errors.New("native service readback unavailable")
	}
	return *result.Value, nil
}

func (n *nativeBuildApplyLifecycle) evidence(ctx context.Context) (buildApplyEvidence, error) {
	if n.service != nil {
		e, err := n.service.cfg.buildApplyOwners.Read(ctx)
		if err != nil {
			return e, err
		}
		if !completeBuildApplyEvidence(e) {
			return e, errors.New("required owner facts are incomplete")
		}
		return e, nil
	}
	snapshot, err := n.snapshot(ctx)
	if err != nil {
		return buildApplyEvidence{}, err
	}
	if snapshot.Evidence == nil {
		return buildApplyEvidence{}, errors.New("required owner facts unavailable")
	}
	if !completeBuildApplyEvidence(*snapshot.Evidence) {
		return buildApplyEvidence{}, errors.New("required owner facts are incomplete")
	}
	return *snapshot.Evidence, nil
}

func (n *nativeBuildApplyLifecycle) Prepare(ctx context.Context, request buildapply.PrepareRequest) (buildApplyPrepared, error) {
	var op buildapply.BuildApplyOperation
	var err error
	if n.service != nil {
		var data json.RawMessage
		data, err = n.service.callBuildApplyBrokerCommand(ctx, "build:apply-prepare", request)
		if err == nil {
			op, err = decodeBuildApplyOwnerOperation(data, request.OperationID, request.PreviewID, buildapply.PhaseQuiesce)
		}
	} else {
		op, err = prepareBuildApplyBroker(ctx, n.tx.receipt.Runtime.Endpoint, request)
	}
	if err != nil {
		return buildApplyPrepared{}, err
	}
	evidence, err := n.evidence(ctx)
	if err != nil {
		return buildApplyPrepared{}, err
	}
	pause := evidence.Pause
	if n.tx.receipt.Pause != nil {
		pause.Owned = n.tx.receipt.Pause.Owned
	} else {
		pause.Owned = buildApplyNewPauseOwned(n.tx.plan)
	}
	if !evidence.Saved || !evidence.Enforced || !evidence.PhysicalStop || !evidence.ActivationVerified || evidence.RecoveryRequired {
		return buildApplyPrepared{}, errors.New("pause owner proof unavailable")
	}
	return buildApplyPrepared{operation: op, pause: pause}, nil
}

func buildApplyNewPauseOwned(p buildApplyPlan) bool { return p.receipt.Runtime.WasPaused == false }

func (n *nativeBuildApplyLifecycle) Reserve(ctx context.Context, r buildapply.Receipt) (buildapply.RunnerReservation, error) {
	path, err := n.store.journal.RunnerPath(n.store.lock)
	if err != nil {
		return buildapply.RunnerReservation{}, err
	}
	env := append(cleanBuildApplyEnvironment(n.tx.plan.serviceLaunch.env), buildApplyOperationEnv+"="+r.OperationID, buildApplyRoleEnv+"=runner", buildApplyLaunchEnv+"="+r.RunnerToken)
	return launchBuildApplyProcess(ctx, buildApplyLaunch{executable: path, args: []string{"apply-build", "run", r.OperationID}, env: env})
}
func (n *nativeBuildApplyLifecycle) Acknowledge(ctx context.Context, r buildapply.Receipt) error {
	_, err := acknowledgeBuildApplyRunner(ctx, n.store.journal, n.store.lock, r)
	return err
}

type buildApplyExitedProof struct{}

func (buildApplyExitedProof) Wait(context.Context) error { return nil }
func (buildApplyExitedProof) Close() error               { return nil }

func (n *nativeBuildApplyLifecycle) Stop(ctx context.Context, old []buildapply.RunnerReservation) error {
	r := n.tx.receipt
	err := stopBuildApplyOwned(ctx, old, func(ctx context.Context, want buildapply.RunnerReservation) (buildApplyExitProof, error) {
		gone, err := buildApplyProcessGone(ctx, want)
		if err != nil {
			return nil, err
		}
		if gone {
			return buildApplyExitedProof{}, nil
		}
		return openBuildApplyExit(ctx, want)
	}, func(ctx context.Context) error {
		gone, err := buildApplyProcessGone(ctx, r.Runtime.OldService)
		if err != nil {
			return err
		}
		if gone {
			return nil
		}
		_, err = callBuildApplyService(ctx, r.Runtime.Endpoint, "service/stop", buildApplyNativeRequest{OperationID: r.OperationID, Runner: *r.Runner})
		return err
	})
	if err != nil {
		return err
	}
	return verifyBuildApplyBundleReleased(ctx, r.Target)
}

func (n *nativeBuildApplyLifecycle) save() error { return n.tx.save() }

func (n *nativeBuildApplyLifecycle) launchOne(ctx context.Context, role string, launch buildApplyLaunch) (buildapply.RunnerReservation, error) {
	r := &n.tx.receipt
	slot := &r.ServiceLaunch
	if role == "gui" {
		slot = &r.GUILaunch
	}
	if slot.Process != nil {
		gone, err := buildApplyProcessGone(ctx, *slot.Process)
		if err != nil {
			return buildapply.RunnerReservation{}, err
		}
		if !gone {
			return *slot.Process, nil
		}
		if !n.recovering {
			return buildapply.RunnerReservation{}, errors.New("selected successor exited; explicit recovery required")
		}
		*slot = buildapply.LaunchReservation{}
	}
	fresh := !slot.Intent
	if fresh {
		token, err := buildapply.NewOperationID()
		if err != nil {
			return buildapply.RunnerReservation{}, err
		}
		slot.Intent = true
		slot.Token = token
		if err := n.save(); err != nil {
			return buildapply.RunnerReservation{}, err
		}
	}
	args := append([]string(nil), launch.args...)
	if role == "gui" {
		args = append(args, "--pair-build-apply="+r.OperationID)
	}
	env := append(cleanBuildApplyEnvironment(launch.env), buildApplyOperationEnv+"="+r.OperationID, buildApplyRoleEnv+"="+role, buildApplyLaunchEnv+"="+slot.Token)
	ownedLaunch := buildApplyLaunch{executable: launch.executable, args: args, env: env}
	if !fresh {
		found, err := findBuildApplyLaunch(ctx, ownedLaunch, r.OperationID, role, slot.Token)
		if err != nil {
			return buildapply.RunnerReservation{}, err
		}
		if found != nil {
			slot.Process = found
			if err := n.save(); err != nil {
				return buildapply.RunnerReservation{}, err
			}
			return *found, nil
		}
		if !n.recovering {
			return buildapply.RunnerReservation{}, errors.New("launch reply is uncertain; explicit recovery required")
		}
	}
	actual, err := launchBuildApplyProcess(ctx, ownedLaunch)
	if err != nil {
		return actual, err
	}
	slot.Process = &actual
	if err := n.save(); err != nil {
		return actual, err
	}
	return actual, nil
}

func findBuildApplyLaunch(ctx context.Context, launch buildApplyLaunch, id, role, token string) (*buildapply.RunnerReservation, error) {
	processes, err := process.ProcessesWithContext(ctx)
	if err != nil {
		return nil, err
	}
	if len(processes) > 65536 {
		return nil, errors.New("process inventory bound exceeded")
	}
	var found *buildapply.RunnerReservation
	for _, p := range processes {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		exe, err := p.ExeWithContext(ctx)
		if err != nil || !sameBuildApplyImage(exe, launch.executable) {
			continue
		}
		args, err := p.CmdlineSliceWithContext(ctx)
		if err != nil {
			return nil, err
		}
		if len(args) == 0 || !slices.Equal(args[1:], launch.args) {
			continue
		}
		env, err := p.EnvironWithContext(ctx)
		if err != nil {
			return nil, err
		}
		values := map[string]string{}
		for _, v := range env {
			k, value, _ := strings.Cut(v, "=")
			values[k] = value
		}
		if values[buildApplyOperationEnv] != id || values[buildApplyRoleEnv] != role || values[buildApplyLaunchEnv] != token {
			continue
		}
		actual, err := inspectBuildApplyProcess(ctx, int(p.Pid))
		if err != nil {
			return nil, err
		}
		if found != nil {
			return nil, errors.New("multiple processes claim one launch reservation")
		}
		copy := actual
		found = &copy
	}
	return found, nil
}

func (n *nativeBuildApplyLifecycle) Restart(ctx context.Context, p buildApplyPlan) (buildApplySuccessors, error) {
	r := &n.tx.receipt
	if digest, err := n.tx.files.Digest(ctx, r.Target); err != nil || digest != r.CandidateSHA256 {
		return buildApplySuccessors{}, errors.New("installed candidate changed")
	}
	if metadata, err := buildApplyMetadataDigest(ctx, r.Target); err != nil || metadata != r.Runtime.CandidateMetadataSHA256 {
		return buildApplySuccessors{}, errors.New("installed candidate metadata changed")
	}
	service, err := n.launchOne(ctx, "service", p.serviceLaunch)
	if err != nil {
		return buildApplySuccessors{}, err
	}
	for {
		snapshot, err := n.snapshot(ctx)
		if err == nil && snapshot.Process == service && snapshot.BrokerReady {
			break
		}
		select {
		case <-ctx.Done():
			return buildApplySuccessors{}, ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
	r.ServiceLaunch.Acknowledged = true
	if err := n.save(); err != nil {
		return buildApplySuccessors{}, err
	}
	if _, err := n.ReadSettings(ctx, *r); err != nil {
		return buildApplySuccessors{}, err
	}
	gui, err := n.launchOne(ctx, "gui", p.guiLaunch)
	if err != nil {
		return buildApplySuccessors{}, err
	}
	for {
		snapshot, err := n.snapshot(ctx)
		if err == nil && snapshot.GUI != nil && *snapshot.GUI == gui {
			break
		}
		select {
		case <-ctx.Done():
			return buildApplySuccessors{}, ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
	r.GUILaunch.Acknowledged = true
	if err := n.save(); err != nil {
		return buildApplySuccessors{}, err
	}
	return buildApplySuccessors{service: service, gui: gui, serviceReady: true}, nil
}

func (n *nativeBuildApplyLifecycle) ReadBuild(ctx context.Context, p buildApplyPlan) (buildApplyBuildReadback, error) {
	snapshot, err := n.snapshot(ctx)
	if err != nil {
		return buildApplyBuildReadback{}, err
	}
	r := n.tx.receipt
	if r.GUILaunch.Process == nil || snapshot.GUI == nil || *snapshot.GUI != *r.GUILaunch.Process || !r.GUILaunch.Acknowledged {
		return buildApplyBuildReadback{}, errors.New("GUI acknowledgement unavailable")
	}
	actual, err := inspectBuildApplyProcess(ctx, snapshot.GUI.PID)
	if err != nil || actual != *snapshot.GUI {
		return buildApplyBuildReadback{}, errors.New("GUI lifetime changed")
	}
	sha, err := buildApplyFileSHA(ctx, p.guiLaunch.executable)
	if err != nil || p.expectedGUI.SHA256.Value == nil || sha != *p.expectedGUI.SHA256.Value {
		return buildApplyBuildReadback{}, errors.New("running GUI bytes differ")
	}
	return buildApplyBuildReadback{service: snapshot.Identity, gui: p.expectedGUI, serviceProcess: snapshot.Process, guiProcess: actual, brokerReady: snapshot.BrokerReady, guiAcknowledged: true}, nil
}

func (n *nativeBuildApplyLifecycle) ReadSettings(ctx context.Context, r buildapply.Receipt) (buildApplySettingsReadback, error) {
	e, err := n.evidence(ctx)
	if err != nil {
		return buildApplySettingsReadback{}, err
	}
	read := buildApplySettingsReadback{fences: e.Impact.Fences, protected: e.Protected, pause: e.Pause, stopEngines: e.Impact.ExpectedSettings.StopEngines.Value != nil && *e.Impact.ExpectedSettings.StopEngines.Value, unloadModels: e.Impact.ExpectedSettings.UnloadModels.Value != nil && *e.Impact.ExpectedSettings.UnloadModels.Value, saved: e.Saved, enforced: e.Enforced, physicalStop: e.PhysicalStop, recoveryRequired: e.RecoveryRequired, activationVerified: e.ActivationVerified}
	if r.Pause != nil {
		read.pause.Owned = r.Pause.Owned
	}
	if buildApplyObservedTrue(e.Impact.Maintenance.BuildApplyActive) {
		return read, verifyBuildApplySettings(r, read)
	}
	if !e.Impact.Maintenance.BuildApplyActive.Known || e.Impact.Maintenance.BuildApplyActive.Value == nil {
		return read, errors.New("maintenance owner state unavailable")
	}
	// A new broker can report a different persisted revision for the same
	// protected Pause semantics. Check those semantics before reconstruction;
	// the guarded prepare below accepts only the exact current owner fences.
	semanticReceipt := r
	if r.Pause != nil {
		pause := *r.Pause
		pause.Revision = read.pause.Revision
		semanticReceipt.ReadbackPause = &pause
	}
	if err := verifyBuildApplySettings(semanticReceipt, read); err != nil {
		return read, err
	}
	// An already finished reservation is a correlated duplicate, not a new
	// Pause. Otherwise reconstitute only after exact protected/Pause readback.
	if r.Phase == buildapply.PhaseFinish {
		if _, err := finishBuildApplyBroker(ctx, r.Runtime.Endpoint, buildapply.FinishRequest{OperationID: r.OperationID, Fences: read.fences}, r.PreviewID); err == nil {
			return read, nil
		}
	}
	_, err = prepareBuildApplyBroker(ctx, r.Runtime.Endpoint, buildapply.PrepareRequest{OperationID: r.OperationID, PreviewID: r.PreviewID, Fences: read.fences, Consent: r.Consent})
	if err != nil {
		return read, err
	}
	after, err := n.evidence(ctx)
	if err != nil {
		return read, err
	}
	if !reflect.DeepEqual(after.Protected, e.Protected) || after.Pause.Origin != e.Pause.Origin || after.Pause.PolicySHA256 != e.Pause.PolicySHA256 {
		return read, errors.New("protected state changed during reservation reconstruction")
	}
	pause := after.Pause
	pause.Owned = r.Pause.Owned
	n.tx.receipt.ReadbackPause = &pause
	if err := n.save(); err != nil {
		return read, err
	}
	read.fences = after.Impact.Fences
	read.pause = pause
	read.saved = after.Saved
	read.enforced = after.Enforced
	read.physicalStop = after.PhysicalStop
	read.activationVerified = after.ActivationVerified
	read.recoveryRequired = after.RecoveryRequired
	return read, verifyBuildApplySettings(n.tx.receipt, read)
}
func (n *nativeBuildApplyLifecycle) Finish(ctx context.Context, r buildapply.FinishRequest) error {
	_, err := finishBuildApplyBroker(ctx, n.tx.receipt.Runtime.Endpoint, r, n.tx.receipt.PreviewID)
	return err
}

func planBuildApplyReceipt(ctx context.Context, r buildapply.Receipt, i buildapply.Inventory) (buildApplyPlan, error) {
	if err := buildapply.ValidateRuntimeAuthority(r.Target, r.Runtime); err != nil {
		return buildApplyPlan{}, err
	}
	if err := validateBuildApplyArguments(r.Runtime.ServiceLaunch.Args, false); err != nil {
		return buildApplyPlan{}, err
	}
	if err := validateBuildApplyImageArguments(r.Target, r.Runtime.ServiceLaunch.Args); err != nil {
		return buildApplyPlan{}, err
	}
	if err := validateBuildApplyArguments(r.Runtime.GUILaunch.Args, true); err != nil {
		return buildApplyPlan{}, err
	}
	layout, _ := buildapply.LayoutFor(runtime.GOOS)
	serviceEnv := mergeBuildApplyEnvironment(os.Environ(), r.Runtime.ServiceLaunch.Environment)
	guiEnv := mergeBuildApplyEnvironment(os.Environ(), r.Runtime.GUILaunch.Environment)
	if buildApplyEnvironmentDigest(serviceEnv) != r.Runtime.ServiceLaunch.EnvironmentSHA256 || buildApplyEnvironmentDigest(guiEnv) != r.Runtime.GUILaunch.EnvironmentSHA256 {
		return buildApplyPlan{}, errors.New("trusted launch profile changed; restore it before explicit recovery")
	}
	p := buildApplyPlan{receipt: r, inventory: i, accepted: r.Runtime.AcceptedAt, deadline: r.Deadline(), ownerFactsComplete: true, oldProcesses: r.Runtime.OldProcesses, expectedService: r.Runtime.ExpectedService, expectedGUI: r.Runtime.ExpectedGUI,
		serviceLaunch: buildApplyLaunch{executable: filepath.Join(r.Target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable), args: r.Runtime.ServiceLaunch.Args, env: serviceEnv}, guiLaunch: buildApplyLaunch{executable: filepath.Join(r.Target, filepath.FromSlash(layout.Executable)), args: r.Runtime.GUILaunch.Args, env: guiEnv}}
	if r.Phase == buildapply.PhasePrepare {
		var err error
		p.current, err = buildapply.InspectPackage(ctx, r.Target, runtime.GOOS, runtime.GOARCH)
		if err != nil {
			return p, err
		}
		if p.current.SHA256 != r.PreviousSHA256 {
			p.current.Close()
			p.current = nil
			return p, errors.New("predecessor changed before preparation recovery")
		}
		paths, err := buildapply.DeriveOperationPaths(r.Target, r.OperationID)
		if err != nil {
			p.current.Close()
			p.current = nil
			return p, err
		}
		files := &nativeBuildApplyFiles{}
		if digest, err := files.Digest(ctx, paths.Prepared); err == nil && digest == r.CandidateSHA256 {
			if metadata, err := buildApplyMetadataDigest(ctx, paths.Prepared); err == nil && metadata == r.Runtime.CandidateMetadataSHA256 {
				return p, nil
			}
		}
		p.candidate, err = buildapply.InspectPackage(ctx, r.Runtime.CandidatePath, runtime.GOOS, runtime.GOARCH)
		if err != nil {
			p.current.Close()
			p.current = nil
			return p, err
		}
	}
	return p, nil
}

func (files *nativeBuildApplyFiles) restoreIdentities(ctx context.Context, r buildapply.Receipt) error {
	paths, err := buildapply.DeriveOperationPaths(r.Target, r.OperationID)
	if err != nil {
		return err
	}
	files.identities = map[string]os.FileInfo{}
	files.preparedID = r.PreparedIdentity
	for _, path := range []string{paths.Target, paths.Prepared, paths.Previous} {
		id, info, err := buildApplyDirectorySnapshot(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return err
		}
		if id != r.Runtime.TargetIdentity && (r.PreparedIdentity == "" || id != r.PreparedIdentity) {
			return errors.New("operation directory identity was substituted")
		}
		files.identities[path] = info
	}
	return nil
}

func buildApplyDirectoryIdentity(path string) (string, error) {
	id, _, err := buildApplyDirectorySnapshot(path)
	return id, err
}

func buildApplyDirectorySnapshot(path string) (string, os.FileInfo, error) {
	r, err := buildapply.OpenRealRoot(path)
	if err != nil {
		return "", nil, err
	}
	defer r.Close()
	f, err := os.Open(path)
	if err != nil {
		return "", nil, err
	}
	id, info, snapshotErr := buildApplyDirectoryFileSnapshot(f)
	if err := errors.Join(snapshotErr, f.Close(), r.Revalidate()); err != nil {
		return "", nil, err
	}
	return id, info, nil
}

// Digest uses the same length-delimited inventory grammar as the package
// inspector and includes the entire tree. Absence is distinct from unreadable.
func (files *nativeBuildApplyFiles) Digest(ctx context.Context, name string) (string, error) {
	real, err := buildapply.OpenRealRoot(name)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	defer real.Close()
	root, err := os.OpenRoot(name)
	if err != nil {
		return "", err
	}
	defer root.Close()
	rootBefore, err := root.Stat(".")
	if err != nil {
		return "", err
	}
	if selected := files.identities[name]; selected != nil && !os.SameFile(selected, rootBefore) {
		return "", errors.New("operation directory identity changed")
	}
	var entries []buildapply.InventoryEntry
	identities := map[string]os.FileInfo{}
	inventoryBytes := 1024
	var total int64
	err = fs.WalkDir(root.FS(), ".", func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if path == "." {
			return nil
		}
		if len(entries) >= buildapply.MaxEntries || len(path) > buildapply.MaxPathBytes {
			return buildapply.ErrPackageTooLarge
		}
		info, err := root.Lstat(filepath.FromSlash(path))
		if err != nil {
			return err
		}
		identities[path] = info
		e := buildapply.InventoryEntry{Path: path, Mode: info.Mode() & (os.ModeType | os.ModePerm)}
		switch {
		case info.IsDir():
		case info.Mode().IsRegular():
			if info.Size() > buildapply.MaxFileBytes || info.Size() > buildapply.MaxPackageBytes-total {
				return buildapply.ErrPackageTooLarge
			}
			f, err := root.Open(filepath.FromSlash(path))
			if err != nil {
				return err
			}
			held, err := f.Stat()
			if err != nil || !os.SameFile(info, held) {
				f.Close()
				return errors.New("file substituted while hashing")
			}
			hash := sha256.New()
			n, readErr := io.Copy(hash, io.LimitReader(buildApplyContextReader{ctx, f}, info.Size()+1))
			after, statErr := f.Stat()
			named, nameErr := root.Lstat(filepath.FromSlash(path))
			closeErr := f.Close()
			if err := errors.Join(readErr, statErr, nameErr, closeErr); err != nil {
				return err
			}
			if n != info.Size() || !os.SameFile(info, after) || !os.SameFile(info, named) || !info.ModTime().Equal(after.ModTime()) || !info.ModTime().Equal(named.ModTime()) {
				return errors.New("file changed while hashing")
			}
			e.Size, e.SHA256 = n, hex.EncodeToString(hash.Sum(nil))
			total += n
		case info.Mode()&os.ModeSymlink != 0:
			link, err := root.Readlink(filepath.FromSlash(path))
			if err != nil {
				return err
			}
			e.Link = link
		default:
			return errors.New("special file in operation bundle")
		}
		encoded, err := json.Marshal(e)
		if err != nil {
			return err
		}
		inventoryBytes += len(encoded) + 1
		if inventoryBytes > buildapply.MaxInventoryBytes {
			return buildapply.ErrPackageTooLarge
		}
		entries = append(entries, e)
		return nil
	})
	if err != nil {
		return "", err
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Path < entries[j].Path })
	hash := sha256.New()
	for _, e := range entries {
		record, err := json.Marshal(struct {
			Path       string
			Size       int64
			Mode       uint32
			Hash, Link string
		}{e.Path, e.Size, uint32(e.Mode), e.SHA256, e.Link})
		if err != nil {
			return "", err
		}
		fmt.Fprintf(hash, "%d:", len(record))
		hash.Write(record)
	}
	if err := real.Revalidate(); err != nil {
		return "", err
	}
	for path, before := range identities {
		after, err := root.Lstat(filepath.FromSlash(path))
		if err != nil || !os.SameFile(before, after) || before.Mode() != after.Mode() || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
			return "", errors.New("tree changed during inventory read")
		}
	}
	rootAfter, err := root.Stat(".")
	if err != nil || !os.SameFile(rootBefore, rootAfter) || !rootBefore.ModTime().Equal(rootAfter.ModTime()) {
		return "", errors.New("tree root changed during inventory read")
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func (files *nativeBuildApplyFiles) Rename(ctx context.Context, from, to string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if filepath.Dir(from) != filepath.Dir(to) {
		return errors.New("rename must stay in one held parent")
	}
	parent, err := buildapply.OpenRealRoot(filepath.Dir(from))
	if err != nil {
		return err
	}
	defer parent.Close()
	root, err := os.OpenRoot(parent.Path)
	if err != nil {
		return err
	}
	defer root.Close()
	if _, err := root.Lstat(filepath.Base(to)); !os.IsNotExist(err) {
		return errors.New("rename destination is not absent")
	}
	source, err := buildapply.OpenRealRoot(from)
	if err != nil {
		return err
	}
	if err := parent.Revalidate(); err != nil {
		source.Close()
		return err
	}
	if err := source.Revalidate(); err != nil {
		source.Close()
		return err
	}
	want := files.identities[from]
	actual, err := root.Lstat(filepath.Base(from))
	if err != nil || want == nil || !os.SameFile(want, actual) {
		source.Close()
		return errors.New("rename source lacks its held operation identity")
	}
	// OpenRealRoot's Windows handle denies deletion. Close that validation
	// handle before the native adapter acquires and verifies its rename handle.
	if err := source.Close(); err != nil {
		return err
	}
	if err := renameBuildApplyDirectory(root, filepath.Base(from), filepath.Base(to), want); err != nil {
		return err
	}
	landed, err := root.Lstat(filepath.Base(to))
	if err != nil || !os.SameFile(want, landed) {
		return errors.New("renamed directory identity is uncertain")
	}
	files.identities[to] = want
	delete(files.identities, from)
	return syncBuildApplyDirectory(parent.Path)
}

func inspectBuildApplyProcess(ctx context.Context, pid int) (buildapply.RunnerReservation, error) {
	if pid <= 0 || int64(pid) > 1<<31-1 {
		return buildapply.RunnerReservation{}, errors.New("invalid process")
	}
	p, err := process.NewProcessWithContext(ctx, int32(pid))
	if err != nil {
		return buildapply.RunnerReservation{}, err
	}
	created, err := p.CreateTimeWithContext(ctx)
	if err != nil || created <= 0 {
		return buildapply.RunnerReservation{}, errors.New("missing creation identity")
	}
	exe, err := p.ExeWithContext(ctx)
	if err != nil {
		return buildapply.RunnerReservation{}, err
	}
	native, err := buildapply.InspectProcess(pid)
	if err != nil || !sameBuildApplyImage(native.Executable, exe) || native.Created == "" {
		return buildapply.RunnerReservation{}, errors.New("native and process image evidence disagree")
	}
	// Keep the native token for independent Windows handle comparison. The
	// gopsutil creation read brackets image inspection and must remain stable.
	createdAfter, err := p.CreateTimeWithContext(ctx)
	if err != nil || createdAfter != created {
		return buildapply.RunnerReservation{}, errors.New("process lifetime changed")
	}
	nativeAfter, err := buildapply.InspectProcess(pid)
	if err != nil || nativeAfter != native {
		return buildapply.RunnerReservation{}, errors.New("native process lifetime changed during inspection")
	}
	return buildapply.RunnerReservation{PID: pid, Identity: native}, nil
}

func copyBuildApplyRunner(ctx context.Context, j *buildapply.Journal, lock *buildapply.OperationLock, current *buildapply.Package) (string, error) {
	if current == nil {
		return "", errors.New("missing trusted current package")
	}
	name := filepath.ToSlash(filepath.Join(current.Layout.ServiceDir, current.Layout.ServiceExecutable))
	source, err := current.OpenVerifiedFile(ctx, name)
	if err != nil {
		return "", err
	}
	defer source.Close()
	path, err := j.RunnerPath(lock)
	if err != nil {
		return "", err
	}
	if _, err := os.Lstat(path); err == nil {
		sha, err := buildApplyFileSHA(ctx, path)
		if err != nil || current.Service.SHA256.Value == nil || sha != *current.Service.SHA256.Value {
			return "", errors.New("existing runner is not the trusted current image")
		}
		return path, nil
	} else if !os.IsNotExist(err) {
		return "", err
	}
	dest, err := j.CreateRunner(lock)
	if err != nil {
		return "", err
	}
	hash := sha256.New()
	_, err = io.Copy(io.MultiWriter(dest, hash), buildApplyContextReader{ctx, source})
	if err == nil && (!current.Service.SHA256.Known || current.Service.SHA256.Value == nil || hex.EncodeToString(hash.Sum(nil)) != *current.Service.SHA256.Value) {
		err = errors.New("runner copy differs from trusted service")
	}
	if err == nil {
		err = dest.Chmod(0o700)
	}
	if err == nil {
		err = dest.Sync()
	}
	return dest.Name(), errors.Join(err, dest.Close())
}

func launchBuildApplyProcess(ctx context.Context, launch buildApplyLaunch) (buildapply.RunnerReservation, error) {
	if err := ctx.Err(); err != nil {
		return buildapply.RunnerReservation{}, err
	}
	if !filepath.IsAbs(launch.executable) || launch.env == nil {
		return buildapply.RunnerReservation{}, errors.New("missing trusted launch")
	}
	cmd := exec.Command(launch.executable, launch.args...)
	cmd.Dir, cmd.Env = filepath.Dir(launch.executable), append([]string(nil), launch.env...)
	configureBuildApplyDetached(cmd)
	if err := cmd.Start(); err != nil {
		return buildapply.RunnerReservation{}, err
	}
	pid := cmd.Process.Pid
	go func() { _ = cmd.Wait() }()
	identity, err := inspectBuildApplyProcess(ctx, pid)
	if err != nil || !sameBuildApplyImage(identity.Identity.Executable, launch.executable) {
		return buildapply.RunnerReservation{}, errors.New("launched lifetime could not be reserved")
	}
	return identity, nil
}

func acknowledgeBuildApplyRunner(ctx context.Context, j *buildapply.Journal, lock *buildapply.OperationLock, r buildapply.Receipt) (buildapply.Receipt, error) {
	self, err := inspectBuildApplyProcess(ctx, os.Getpid())
	if err != nil || r.Runner == nil || *r.Runner != self || r.Handoff != buildapply.HandoffReserved || r.Pause == nil {
		return r, buildapply.ErrRecoveryNeeded
	}
	r.Handoff = buildapply.HandoffAcknowledged
	r.RunnerReady = true
	state, err := j.Save(lock, r)
	if err != nil {
		return r, err
	}
	if state.Current == nil || !reflect.DeepEqual(*state.Current, r) {
		return r, buildapply.ErrRecoveryNeeded
	}
	return r, nil
}

type buildApplyExitProof interface {
	Wait(context.Context) error
	Close() error
}

// Acquire independent exit proofs before ordering shutdown. Opening a handle
// after a successful stop reply would lose the old Windows process lifetime.
func stopBuildApplyOwned(ctx context.Context, old []buildapply.RunnerReservation, open func(context.Context, buildapply.RunnerReservation) (buildApplyExitProof, error), stop func(context.Context) error) error {
	if len(old) == 0 {
		return errors.New("complete owned process inventory required")
	}
	proofs := make([]buildApplyExitProof, 0, len(old))
	defer func() {
		for _, p := range proofs {
			_ = p.Close()
		}
	}()
	for _, want := range old {
		p, err := open(ctx, want)
		if err != nil {
			return err
		}
		proofs = append(proofs, p)
	}
	stopErr := stop(ctx)
	// Even on lost replies, observe all reserved lifetimes, preserving errors.
	var waitErr error
	for _, p := range proofs {
		waitErr = errors.Join(waitErr, p.Wait(ctx))
	}
	return errors.Join(stopErr, waitErr)
}

func decodeBuildApplyOwnerOperation(data []byte, id, preview string, phase buildapply.Phase) (buildapply.BuildApplyOperation, error) {
	var result buildapply.Result[buildapply.BuildApplyOperation]
	if len(data) == 0 || len(data) > buildapply.MaxStatusBytes || json.Unmarshal(data, &result) != nil || !result.Available || result.Value == nil {
		return buildapply.BuildApplyOperation{}, errors.New("required broker operation evidence is unavailable")
	}
	op := *result.Value
	if op.OperationID != id || op.PreviewID != preview || op.Phase != phase || op.Outcome != nil || !buildApplyFencesComplete(op.Fences) ||
		!buildApplyObservedTrue(op.Saved) || !buildApplyObservedTrue(op.Enforced) || !buildApplyObservedTrue(op.PhysicalStop) || !buildApplyObservedTrue(op.RemainsPaused) {
		return buildapply.BuildApplyOperation{}, errors.New("broker operation is uncorrelated or not physically enforced")
	}
	return op, nil
}

func prepareBuildApplyBroker(ctx context.Context, endpoint string, r buildapply.PrepareRequest) (buildapply.BuildApplyOperation, error) {
	if !r.Consent.AllowEngineStop || !r.Consent.AllowInterruption || r.PreviewID == "" || !buildapply.ValidOperationID(r.OperationID) {
		return buildapply.BuildApplyOperation{}, errors.New("explicit correlated interruption consent required")
	}
	receipt, err := readBuildApplyReceipt(r.OperationID)
	if err != nil {
		return buildapply.BuildApplyOperation{}, buildapply.ErrRecoveryNeeded
	}
	executor, err := buildApplyCallerReservation(ctx, receipt)
	if err != nil {
		return buildapply.BuildApplyOperation{}, err
	}
	data, err := callBuildApplyService(ctx, endpoint, buildApplyNativePrepareMethod, buildApplyOwnerCommand{OperationID: r.OperationID, Runner: executor, Prepare: &r})
	if err != nil {
		return buildapply.BuildApplyOperation{}, err
	}
	return decodeBuildApplyOwnerOperation(data, r.OperationID, r.PreviewID, buildapply.PhaseQuiesce)
}

func finishBuildApplyBroker(ctx context.Context, endpoint string, r buildapply.FinishRequest, preview string) (buildapply.BuildApplyOperation, error) {
	if !buildapply.ValidOperationID(r.OperationID) || preview == "" || !buildApplyFencesComplete(r.Fences) {
		return buildapply.BuildApplyOperation{}, errors.New("current correlated finish authority required")
	}
	receipt, err := readBuildApplyReceipt(r.OperationID)
	if err != nil {
		return buildapply.BuildApplyOperation{}, buildapply.ErrRecoveryNeeded
	}
	executor, err := buildApplyCallerReservation(ctx, receipt)
	if err != nil {
		return buildapply.BuildApplyOperation{}, err
	}
	data, err := callBuildApplyService(ctx, endpoint, buildApplyNativeFinishMethod, buildApplyOwnerCommand{OperationID: r.OperationID, Runner: executor, Finish: &r})
	if err != nil {
		return buildapply.BuildApplyOperation{}, err
	}
	return decodeBuildApplyOwnerOperation(data, r.OperationID, preview, buildapply.PhaseFinish)
}

func ensureBuildApplyRunner(ctx context.Context, j *buildapply.Journal, lock *buildapply.OperationLock, r buildapply.Receipt) (string, error) {
	path, err := j.RunnerPath(lock)
	if err != nil {
		return "", err
	}
	if info, err := os.Lstat(path); err == nil {
		sha, err := buildApplyFileSHA(ctx, path)
		if err != nil {
			return "", err
		}
		if sha == r.Runtime.RunnerSHA256 && (runtime.GOOS == "windows" || info.Mode().Perm() == 0o700) {
			return path, nil
		}
		// A crash during the initial trusted copy can leave partial bytes or
		// complete bytes still at the initial non-executable Unix mode.
		// Explicit recovery may discard only that private canonical file, after
		// independent image-release proof and a second hash/identity check.
		if err := verifyBuildApplyBundleReleased(ctx, filepath.Dir(path)); err != nil {
			return "", errors.Join(errors.New("incomplete runner ownership is not released"), err)
		}
		if err := j.RemoveRunner(lock, sha); err != nil {
			return "", err
		}
	} else if !os.IsNotExist(err) {
		return "", err
	}
	paths, err := buildapply.DeriveOperationPaths(r.Target, r.OperationID)
	if err != nil {
		return "", err
	}
	files := &nativeBuildApplyFiles{}
	oldRoot := ""
	for _, candidate := range []string{paths.Target, paths.Previous} {
		digest, err := files.Digest(ctx, candidate)
		if err != nil {
			return "", err
		}
		if digest == r.PreviousSHA256 {
			id, err := buildApplyDirectoryIdentity(candidate)
			if err != nil || id != r.Runtime.TargetIdentity {
				return "", errors.New("predecessor directory identity changed")
			}
			oldRoot = candidate
			break
		}
	}
	if oldRoot == "" {
		return "", errors.New("verified predecessor cannot supply the trusted runner")
	}
	layout, _ := buildapply.LayoutFor(runtime.GOOS)
	sourcePath := filepath.Join(oldRoot, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable)
	sha, err := buildApplyFileSHA(ctx, sourcePath)
	if err != nil || sha != r.Runtime.RunnerSHA256 {
		return "", errors.New("predecessor runner bytes changed")
	}
	root, err := buildapply.OpenRealRoot(filepath.Dir(sourcePath))
	if err != nil {
		return "", err
	}
	defer root.Close()
	rr, err := os.OpenRoot(root.Path)
	if err != nil {
		return "", err
	}
	defer rr.Close()
	source, err := rr.Open(filepath.Base(sourcePath))
	if err != nil {
		return "", err
	}
	defer source.Close()
	dest, err := j.CreateRunner(lock)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	_, copyErr := io.Copy(io.MultiWriter(dest, h), io.LimitReader(buildApplyContextReader{ctx, source}, buildapply.MaxFileBytes+1))
	if copyErr == nil && hex.EncodeToString(h.Sum(nil)) != r.Runtime.RunnerSHA256 {
		copyErr = errors.New("recovery runner copy changed")
	}
	if copyErr == nil {
		copyErr = dest.Chmod(0o700)
	}
	if copyErr == nil {
		copyErr = dest.Sync()
	}
	return path, errors.Join(copyErr, dest.Close())
}

func runNativeBuildApply(ctx context.Context, id string, reserved bool) (runErr error) {
	j, err := buildapply.OpenJournal(false)
	if err != nil {
		return err
	}
	defer j.Close()
	state, err := j.Read()
	if err != nil {
		return err
	}
	r := state.Current
	if r == nil {
		r = state.LastTerminal
	}
	if r == nil || r.OperationID != id {
		return buildapply.ErrRecoveryNeeded
	}
	if !reserved && state.Current == nil && !state.TerminalPending {
		if r.Outcome != nil && *r.Outcome == buildapply.OutcomeRefused {
			return nil
		}
		lock, err := j.TryLock()
		if err != nil {
			return err
		}
		defer lock.Close()
		return cleanupBuildApplyTerminal(ctx, j, lock, *r)
	}
	acquire, cancelAcquire := context.WithTimeout(ctx, buildApplyLiveBudget)
	defer cancelAcquire()
	lock, err := j.TryLock()
	for err != nil && reserved && isBuildApplyLockBusy(err) {
		select {
		case <-acquire.Done():
			return acquire.Err()
		case <-time.After(25 * time.Millisecond):
		}
		lock, err = j.TryLock()
	}
	if err != nil {
		return err
	}
	defer func() {
		if lock != nil {
			_ = lock.Close()
		}
	}()
	state, err = j.Read()
	if err != nil || state.Current == nil && !state.TerminalPending {
		return buildapply.ErrRecoveryNeeded
	}
	r = state.Current
	if r == nil {
		r = state.LastTerminal
	}
	if r == nil || r.OperationID != id {
		return buildapply.ErrRecoveryNeeded
	}
	if !reserved {
		if r.Runner != nil {
			gone, err := buildApplyProcessGone(acquire, *r.Runner)
			if err != nil || !gone {
				return errors.Join(buildapply.ErrOperationBusy, err)
			}
		}
		state, err = j.Recover(lock, id)
		if err != nil {
			return err
		}
		if state.Current == nil {
			if state.LastTerminal != nil && state.LastTerminal.OperationID == id && state.LastTerminal.Outcome != nil && *state.LastTerminal.Outcome == buildapply.OutcomeRefused {
				return nil
			}
			return buildapply.ErrRecoveryNeeded
		}
		receipt := *state.Current
		now := time.Now().UTC()
		receipt.RecoveryStartedAt = now
		receipt.RecoveryDeadline = now.Add(buildApplyTotalBudget)
		receipt.Outcome = nil
		receipt.Verified = nil
		receipt.Progress = nil
		receipt.Failure = nil
		receipt.RunnerReady = false
		caller, err := inspectBuildApplyProcess(acquire, os.Getpid())
		if err != nil {
			return err
		}
		receipt.RecoveryCaller = &caller
		path, err := ensureBuildApplyRunner(acquire, j, lock, receipt)
		if err != nil {
			return err
		}
		previousLaunch := buildApplyLaunch{executable: path, args: []string{"apply-build", "run", id}}
		found, err := findBuildApplyLaunch(acquire, previousLaunch, id, "runner", receipt.RunnerToken)
		if err != nil {
			return err
		}
		if found != nil {
			return buildapply.ErrOperationBusy
		}
		token, err := buildapply.NewOperationID()
		if err != nil {
			return err
		}
		receipt.RunnerToken = token
		if _, err := j.Save(lock, receipt); err != nil {
			return err
		}
		env := mergeBuildApplyEnvironment(os.Environ(), receipt.Runtime.ServiceLaunch.Environment)
		if buildApplyEnvironmentDigest(env) != receipt.Runtime.ServiceLaunch.EnvironmentSHA256 {
			return errors.New("recovery launch profile differs from accepted context")
		}
		env = append(env, buildApplyOperationEnv+"="+id, buildApplyRoleEnv+"=runner", buildApplyLaunchEnv+"="+receipt.RunnerToken)
		self, err := launchBuildApplyProcess(acquire, buildApplyLaunch{executable: path, args: []string{"apply-build", "run", id}, env: env})
		if err != nil {
			return err
		}
		receipt.Runner = &self
		if _, err := j.Save(lock, receipt); err != nil {
			return err
		}
		if err := lock.Close(); err != nil {
			return err
		}
		lock = nil
		for {
			state, err = j.Read()
			if err != nil {
				return err
			}
			if state.Current != nil && state.Current.OperationID == id && state.Current.RunnerReady {
				return nil
			}
			select {
			case <-acquire.Done():
				return acquire.Err()
			case <-time.After(25 * time.Millisecond):
			}
		}
	}
	receipt := *r
	self, err := inspectBuildApplyProcess(acquire, os.Getpid())
	if err != nil || receipt.Runner == nil || self != *receipt.Runner || os.Getenv(buildApplyOperationEnv) != id || os.Getenv(buildApplyRoleEnv) != "runner" || os.Getenv(buildApplyLaunchEnv) != receipt.RunnerToken {
		return buildapply.ErrRecoveryNeeded
	}
	sha, err := buildApplyFileSHA(acquire, self.Identity.Executable)
	if err != nil || sha != receipt.Runtime.RunnerSHA256 {
		return buildapply.ErrRecoveryNeeded
	}
	// Once the exact copied runner owns the lock, even setup/readback errors
	// receive a durable failed-boundary projection before this one-shot exits.
	defer func() {
		if runErr == nil || lock == nil {
			return
		}
		state, readErr := j.Read()
		if readErr != nil || state.TerminalPending || state.Current == nil || state.Current.OperationID != id || state.Current.Runner == nil || *state.Current.Runner != self || state.Current.Outcome != nil {
			return
		}
		failed := *state.Current
		outcome := buildapply.OutcomeUncertain
		failed.Outcome = &outcome
		failed.Verified, failed.Progress = nil, nil
		failed.Failure = &buildapply.BuildApplyFailure{Phase: failed.Phase, Reason: "runner_boundary_unverified", Detail: "The reserved runner exited without completing this durable boundary.", ReadbackConfidence: buildapply.ConfidenceUnknown, SafeNextAction: "Read status and explicitly recover this operation while paused."}
		_, saveErr := j.Save(lock, failed)
		runErr = errors.Join(runErr, saveErr)
	}()
	if !time.Now().Before(receipt.Deadline()) {
		return context.DeadlineExceeded
	}
	if !receipt.RecoveryStartedAt.IsZero() {
		state, err = j.Recover(lock, id)
		if err != nil {
			return err
		}
		receipt = *state.Current
		receipt.Outcome = nil
		receipt.Verified = nil
		receipt.Progress = nil
		receipt.Failure = nil
	}
	receipt.RunnerReady = true
	if receipt.Handoff == buildapply.HandoffReserved {
		receipt.Handoff = buildapply.HandoffAcknowledged
	}
	if _, err := j.Save(lock, receipt); err != nil {
		return err
	}
	if receipt.RecoveryCaller != nil {
		gone, err := buildApplyProcessGone(acquire, *receipt.RecoveryCaller)
		if err != nil {
			return err
		}
		if !gone {
			if err := proveBuildApplyExit(acquire, *receipt.RecoveryCaller); err != nil {
				return err
			}
		}
	}
	i, err := j.ReadInventory(receipt)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithDeadline(ctx, receipt.Deadline())
	defer cancel()
	p, err := planBuildApplyReceipt(ctx, receipt, i)
	if err != nil {
		return err
	}
	if p.current != nil {
		defer func() {
			if p.current != nil {
				p.current.Close()
			}
		}()
	}
	if p.candidate != nil {
		defer p.candidate.Close()
	}
	store := &buildApplyJournalStore{journal: j, lock: lock}
	tx := &buildApplyTransaction{plan: p, store: store, receipt: receipt, operation: projectBuildApplyReceipt(receipt)}
	files := &nativeBuildApplyFiles{journal: j, lock: lock, transaction: tx}
	tx.files = files
	native := &nativeBuildApplyLifecycle{tx: tx, store: store, recovering: !receipt.RecoveryStartedAt.IsZero()}
	tx.owners = native
	if err := files.restoreIdentities(ctx, receipt); err != nil {
		return err
	}
	if err := resumeBuildApplyPreparation(ctx, tx, native, files); err != nil {
		_, failure := tx.fail("recovery_prepare_unverified")
		return errors.Join(err, failure)
	}
	if p.current != nil {
		if err := p.current.Close(); err != nil {
			return err
		}
		p.current, tx.plan.current = nil, nil
	}
	_, err = tx.runRemaining(ctx)
	if err != nil {
		return err
	}
	state, err = j.Read()
	if err != nil {
		return err
	}
	if state.LastTerminal != nil && !state.TerminalPending {
		_ = cleanupBuildApplyTerminal(ctx, j, lock, *state.LastTerminal)
	}
	return nil
}

func resumeBuildApplyPreparation(ctx context.Context, tx *buildApplyTransaction, native *nativeBuildApplyLifecycle, files *nativeBuildApplyFiles) error {
	r := &tx.receipt
	if r.Phase == buildapply.PhasePrepare {
		if !r.PreparedIntent {
			r.PreparedIntent = true
			if err := tx.save(); err != nil {
				return err
			}
		}
		paths, err := buildapply.DeriveOperationPaths(r.Target, r.OperationID)
		if err != nil {
			return err
		}
		if id, err := buildApplyDirectoryIdentity(paths.Prepared); err == nil {
			if r.PreparedIdentity == "" || id != r.PreparedIdentity {
				return errors.New("prepared directory identity changed; refusing recovery cleanup")
			}
		} else if !os.IsNotExist(err) {
			return err
		}
		digest, err := files.Digest(ctx, paths.Prepared)
		if err != nil {
			return err
		}
		metadata := ""
		if digest == r.CandidateSHA256 {
			metadata, err = buildApplyMetadataDigest(ctx, paths.Prepared)
			if err != nil {
				return err
			}
		}
		if digest == r.CandidateSHA256 && metadata == r.Runtime.CandidateMetadataSHA256 {
			id, err := buildApplyDirectoryIdentity(paths.Prepared)
			if err != nil {
				return err
			}
			if r.PreparedIdentity != "" && r.PreparedIdentity != id {
				return errors.New("frozen directory identity changed")
			}
			r.PreparedIdentity = id
			files.preparedID = id
		} else {
			if digest != "" {
				private, err := buildapply.OpenPrivateOperationRoot(paths.Prepared)
				if err != nil {
					return err
				}
				private.Close()
				id, err := buildApplyDirectoryIdentity(paths.Prepared)
				if err != nil {
					return err
				}
				if r.PreparedIdentity == "" || id != r.PreparedIdentity {
					return errors.New("prepared directory identity changed; refusing recovery cleanup")
				}
				if current, err := files.Digest(ctx, r.Target); err != nil || current != r.PreviousSHA256 {
					return errors.New("verified predecessor is unavailable")
				}
				if err := removeBuildApplyTree(ctx, paths.Prepared, id); err != nil {
					return err
				}
			}
			if err := files.Freeze(ctx, tx.plan); err != nil {
				return err
			}
			r.PreparedIdentity = files.PreparedIdentity()
		}
		if err := tx.save(); err != nil {
			return err
		}
		if err := files.restoreIdentities(ctx, *r); err != nil {
			return err
		}
		if err := syncBuildApplyPreparation(paths.Prepared, tx.plan.inventory.Entries, syncBuildApplyDirectory); err != nil {
			return err
		}
		if err := tx.phase(buildapply.PhaseQuiesce); err != nil {
			return err
		}
	}
	if r.Phase == buildapply.PhaseQuiesce {
		if r.Pause == nil {
			budget := time.Duration(r.Expected.DrainTimeoutSeconds)*time.Second + buildApplyShutdownBudget
			if time.Until(r.Deadline()) <= budget+4*buildApplyReadBudget {
				return errors.New("saved drain budget no longer fits")
			}
			prepareCtx, cancel := context.WithTimeout(ctx, budget)
			prepared, err := native.Prepare(prepareCtx, buildapply.PrepareRequest{OperationID: r.OperationID, PreviewID: r.PreviewID, Fences: r.Fences, Consent: r.Consent})
			cancel()
			if err != nil {
				return err
			}
			r.Pause = &prepared.pause
			r.ReadbackPause = &prepared.pause
			tx.operation.Fences = prepared.operation.Fences
			if err := tx.save(); err != nil {
				return err
			}
		} else {
			e, err := native.evidence(ctx)
			if err != nil {
				return err
			}
			read := buildApplySettingsReadback{fences: e.Impact.Fences, protected: e.Protected, pause: e.Pause, stopEngines: true, unloadModels: true, saved: e.Saved, enforced: e.Enforced, physicalStop: e.PhysicalStop, activationVerified: e.ActivationVerified, recoveryRequired: e.RecoveryRequired}
			read.pause.Owned = r.Pause.Owned
			if err := verifyBuildApplySettings(*r, read); err != nil {
				return err
			}
		}
		if r.Handoff == buildapply.HandoffNone {
			r.Handoff = buildapply.HandoffIntent
			if err := tx.save(); err != nil {
				return err
			}
		}
		if r.Handoff == buildapply.HandoffIntent {
			r.Handoff = buildapply.HandoffReserved
			if err := tx.save(); err != nil {
				return err
			}
		}
		r.Handoff = buildapply.HandoffAcknowledged
		r.RunnerReady = true
		if err := tx.save(); err != nil {
			return err
		}
	}
	return nil
}

func cleanupBuildApplyTerminal(ctx context.Context, j *buildapply.Journal, lock *buildapply.OperationLock, r buildapply.Receipt) error {
	if r.Outcome == nil || *r.Outcome != buildapply.OutcomeSucceeded {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, buildApplyReadBudget)
	defer cancel()
	if r.ReleasePending {
		self, err := inspectBuildApplyProcess(ctx, os.Getpid())
		if err != nil {
			return err
		}
		layout, _ := buildapply.LayoutFor(runtime.GOOS)
		selectedService := filepath.Join(r.Target, filepath.FromSlash(layout.ServiceDir), layout.ServiceExecutable)
		expectedSHA := ""
		if sameBuildApplyImage(self.Identity.Executable, selectedService) && r.Runtime.ExpectedService.SHA256.Value != nil {
			expectedSHA = *r.Runtime.ExpectedService.SHA256.Value
		}
		runnerPath, err := j.RunnerPath(lock)
		if err != nil {
			return err
		}
		if sameBuildApplyImage(self.Identity.Executable, runnerPath) {
			expectedSHA = r.Runtime.RunnerSHA256
		}
		sha, err := buildApplyFileSHA(ctx, self.Identity.Executable)
		if err != nil || expectedSHA == "" || sha != expectedSHA {
			return errors.New("cleanup executor does not belong to the selected operation")
		}
		r.CleanupRunner = &self
		r.CleanupDeadline = time.Now().UTC().Add(buildApplyReadBudget)
		if _, err := j.SaveCleanup(lock, r); err != nil {
			return err
		}
		data, err := callBuildApplyService(ctx, r.Runtime.Endpoint, buildApplyNativeReadMethod, buildApplyNativeRequest{OperationID: r.OperationID, Runner: self})
		if err != nil {
			return err
		}
		var result buildapply.Result[buildApplyNativeSnapshot]
		if json.Unmarshal(data, &result) != nil || !result.Available || result.Value == nil || result.Value.Evidence == nil {
			return errors.New("cleanup owner readback unavailable")
		}
		e := result.Value.Evidence
		if !e.Impact.Maintenance.BuildApplyActive.Known || e.Impact.Maintenance.BuildApplyActive.Value == nil {
			return errors.New("maintenance release state unavailable")
		}
		if *e.Impact.Maintenance.BuildApplyActive.Value {
			if !completeBuildApplyEvidence(*e) || !e.Saved || !e.Enforced || !e.PhysicalStop || !e.ActivationVerified || e.Impact.Availability.Value == nil || *e.Impact.Availability.Value != nodepolicy.Paused {
				return errors.New("cleanup requires an enforced durable Pause")
			}
			if _, err := finishBuildApplyBroker(ctx, r.Runtime.Endpoint, buildapply.FinishRequest{OperationID: r.OperationID, Fences: e.Impact.Fences}, r.PreviewID); err != nil {
				return err
			}
		}
		r.ReleasePending = false
		if _, err := j.SaveCleanup(lock, r); err != nil {
			return err
		}
	}
	paths, err := buildapply.DeriveOperationPaths(r.Target, r.OperationID)
	if err != nil {
		return err
	}
	files := &nativeBuildApplyFiles{}
	current, err := files.Digest(ctx, paths.Target)
	if err != nil || current != r.CandidateSHA256 {
		return errors.New("cleanup requires verified installed candidate")
	}
	id, err := buildApplyDirectoryIdentity(paths.Target)
	if err != nil || id != r.PreparedIdentity {
		return errors.New("installed directory identity changed")
	}
	previous, err := files.Digest(ctx, paths.Previous)
	if err != nil {
		return err
	}
	if previous != "" {
		if !r.CleanupStarted && previous != r.PreviousSHA256 {
			return errors.New("retained predecessor changed")
		}
		for _, p := range r.Runtime.OldProcesses {
			gone, err := buildApplyProcessGone(ctx, p)
			if err != nil || !gone {
				return errors.New("predecessor process ownership is not released")
			}
		}
		if !r.CleanupStarted {
			r.CleanupStarted = true
			if _, err := j.SaveCleanup(lock, r); err != nil {
				return err
			}
		}
		if err := removeBuildApplyTree(ctx, paths.Previous, r.Runtime.TargetIdentity); err != nil {
			return err
		}
	}
	r.CleanupPending = false
	if _, err := j.SaveCleanup(lock, r); err != nil {
		return err
	}
	if r.Runner != nil {
		gone, err := buildApplyProcessGone(ctx, *r.Runner)
		if err != nil {
			return err
		}
		if gone {
			return j.RemoveRunner(lock, r.Runtime.RunnerSHA256)
		}
	}
	return nil
}

// The sole IPC path is the existing private service endpoint. Context expiry
// closes the transport, so neither a blocked writer nor a missing response can
// turn a finite runner into a resident helper. Notifications are bounded too.
func callBuildApplyService(ctx context.Context, endpoint, method string, request any) (json.RawMessage, error) {
	conn, err := servicectl.DialEndpoint(ctx, endpoint)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			_ = conn.Close()
		case <-done:
		}
	}()
	params, err := json.Marshal(request)
	if err != nil || len(params) > buildapply.MaxRequestBytes {
		return nil, errors.New("invalid local transaction request")
	}
	id := json.RawMessage(`1`)
	msg := jsonrpc.Message{JSONRPC: "2.0", ID: &id, Method: method, Params: params}
	if err := json.NewEncoder(conn).Encode(msg); err != nil {
		return nil, err
	}
	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 4096), buildapply.MaxStatusBytes)
	for scanner.Scan() {
		var response jsonrpc.Message
		if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
			return nil, err
		}
		if response.ID == nil {
			continue
		}
		if string(*response.ID) != "1" || response.JSONRPC != "2.0" {
			return nil, errors.New("uncorrelated service response")
		}
		if response.Error != nil {
			return nil, errors.New("service refused transaction request")
		}
		return response.Result, nil
	}
	return nil, errors.Join(errors.New("service response unavailable"), scanner.Err(), ctx.Err())
}
