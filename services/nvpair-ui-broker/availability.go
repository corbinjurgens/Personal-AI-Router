// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// availability.go orchestrates pausing and resuming this node
// (FORK_DESIGN.md §3.4). Pausing only affects work executed on this machine:
// requests entering this node are still routed to other machines.
//
// Transitions run one at a time. A newer node:set-availability cancels the one
// in progress, which stops at its next step and reports that it was
// superseded, so the latest request always wins and nothing deadlocks waiting
// for a drain that is no longer wanted.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

// errAvailabilitySuperseded is returned to a caller whose transition was
// replaced by a newer one before it finished.
var errAvailabilitySuperseded = errors.New("superseded by a later node:set-availability request")

// Task 5 enables immediate dispatch after physical-stop and interruption proof.
var errImmediatePauseUnavailable = errors.New("Immediate pause is unavailable until safe execution stopping is enabled.")

// pauseCancelSettle bounds how long a pause waits for cancelled work to
// unwind after the drain timeout, before it unloads and stops anyway.
var pauseCancelSettle = 10 * time.Second

// setAvailability moves the node to state and returns once it is reached. The
// transition itself does not depend on ctx: a caller that stops waiting (a
// disconnected client or peer) does not abandon a half-finished pause.
//
// origin records who asked. Either origin's accepted pause refuses all later
// peer writes, including Resume and repeated Pause, before superseding work.
func (b *Broker) setAvailability(ctx context.Context, state nodepolicy.Availability, origin nodepolicy.PauseOrigin) (nodepolicy.Availability, error) {
	return b.setAvailabilityAs(ctx, state, origin, "")
}

// setAvailabilityAs is setAvailability for a verified caller. An empty caller
// is this node's own client path. r.mu owns transition acceptance: permission
// is checked before registration can supersede an in-flight transition.
// transitionMu serializes execution of work already accepted here.
func (b *Broker) setAvailabilityAs(ctx context.Context, state nodepolicy.Availability, origin nodepolicy.PauseOrigin, caller string) (nodepolicy.Availability, error) {
	return b.setAvailabilityRequestAs(ctx, state, origin, caller, nil)
}

func (b *Broker) setAvailabilityRequestAs(ctx context.Context, state nodepolicy.Availability, origin nodepolicy.PauseOrigin, caller string, rawMode json.RawMessage) (nodepolicy.Availability, error) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if err := b.authorizeRemoteManagementLocked(r, caller, "availability", ""); err != nil {
		r.mu.Unlock()
		return "", err
	}
	if origin == nodepolicy.PausedByPeer && r.managementBlockedLocked() {
		r.mu.Unlock()
		return "", &localVetoError{}
	}
	if state != nodepolicy.Available && state != nodepolicy.Paused {
		r.mu.Unlock()
		return "", &invalidParamsError{fmt.Errorf("state must be %q or %q", nodepolicy.Available, nodepolicy.Paused)}
	}
	mode, err := nodepolicy.ParsePauseMode(state, rawMode)
	if err != nil {
		r.mu.Unlock()
		return "", &invalidParamsError{err}
	}
	if mode == nodepolicy.PauseImmediate {
		r.mu.Unlock()
		if caller != "" || origin != nodepolicy.PausedLocally {
			return "", &invalidParamsError{errors.New("Immediate pause must be requested on this node.")}
		}
		return "", errImmediatePauseUnavailable
	}
	return b.setAvailabilityModeLocked(ctx, r, state, origin, mode)
}

// setAvailabilityModeLocked is also the internal immediate entry point. The
// public dispatcher above continues to refuse immediate until Task 5.
// Caller holds r.mu and has authorized the request.
func (b *Broker) setAvailabilityModeLocked(ctx context.Context, r *policyRuntime, state nodepolicy.Availability, origin nodepolicy.PauseOrigin, mode nodepolicy.PauseMode) (nodepolicy.Availability, error) {
	var saveErr error
	var saved bool
	if state == nodepolicy.Paused && mode == nodepolicy.PauseImmediate {
		policy, persisted, err := r.persistAvailabilityLocked(nodepolicy.Paused, origin)
		if err != nil && !persisted {
			r.mu.Unlock()
			return "", fmt.Errorf("the paused state could not be saved: %w", err)
		}
		r.current = policy
		r.live = nodepolicy.Draining
		r.cancelActive = true
		saved = persisted
		if err != nil {
			saveErr = &policySaveError{err: err, revision: policy.Revision}
		}
	}
	if state == nodepolicy.Paused {
		r.pendingPauseRequested = true
		r.localPauseRequested = origin == nodepolicy.PausedLocally || r.localPauseRequested || pausedLocally(r.current)
	}
	r.transitionGen++
	gen := r.transitionGen
	if r.transitionCancel != nil {
		r.transitionCancel()
	}
	tctx, cancel := context.WithCancel(context.Background())
	r.transitionCancel = cancel
	r.mu.Unlock()

	if state == nodepolicy.Paused {
		// Publish before waiting for engineConfigMu or transitionMu. Admission
		// takes engineConfigMu -> r.mu, never the reverse; pending state closes
		// the interval while already accepted work finishes its journal.
		b.publishNodeProjection()
		if mode == nodepolicy.PauseImmediate {
			b.pushAvailabilityAndActivate(saved)
			b.notifyAvailability()
		}
		if err := b.vetoRemoteEngineSwitch("", false); err != nil {
			cancel()
			r.mu.Lock()
			if r.transitionGen == gen {
				r.localPauseRequested = false
				r.pendingPauseRequested = false
			}
			r.mu.Unlock()
			b.publishNodeProjection()
			return "", err
		}
		// An install admitted before this pause must not start its engine
		// afterwards; one submitted later is not affected.
		b.engineConfigMu.Lock()
		b.supersedeInstallStartsLocked("")
		b.engineConfigMu.Unlock()
		b.cancelRemoteSwitch()
	}

	type outcome struct {
		availability nodepolicy.Availability
		err          error
	}
	done := make(chan outcome, 1)
	go func() {
		defer cancel()
		availability, err := b.runAvailabilityTransition(tctx, gen, state, origin, mode)
		// The transition saved its provenance or failed before saving any;
		// either way the stored policy now says whether a local pause holds.
		r.mu.Lock()
		if r.transitionGen == gen {
			// A failed Resume must retain the fence until a later local
			// transition actually makes availability authoritative.
			if state == nodepolicy.Paused {
				r.localPauseRequested = false
				r.pendingPauseRequested = false
			}
		}
		r.mu.Unlock()
		b.publishNodeProjection()
		done <- outcome{availability, err}
	}()
	select {
	case o := <-done:
		return o.availability, errors.Join(o.err, saveErr)
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (b *Broker) runAvailabilityTransition(ctx context.Context, gen uint64, state nodepolicy.Availability, origin nodepolicy.PauseOrigin, mode nodepolicy.PauseMode) (nodepolicy.Availability, error) {
	r := b.nodePolicyRuntime()
	r.transitionMu.Lock()
	defer r.transitionMu.Unlock()
	r.mu.Lock()
	latest := r.transitionGen == gen
	r.mu.Unlock()
	if !latest || ctx.Err() != nil {
		return "", errAvailabilitySuperseded
	}
	if state == nodepolicy.Paused {
		availability, err := b.pauseNode(ctx, origin, mode)
		if availability == nodepolicy.Paused {
			// transitionMu keeps a later Resume behind the durable completion
			// proof, including a landed save that returned a durability warning.
			err = errors.Join(err, b.settleSwitchPauseObligations())
		}
		return availability, err
	}
	return b.resumeNode(ctx, gen, origin)
}

// persistAvailabilityLocked returns the current policy with state as its
// availability, saved under the next revision. A pause records origin as its
// pausedBy provenance and a resume clears it. That revision is active once the
// proxy has acknowledged the availability push and the policy push that follows
// it. While policy recovery is
// required nothing is saved, so the corrupt file and recoveryRequired stay as
// they are; the change then lives in memory only and persisted is false.
// A landed replacement is persisted even when durability failed; the caller
// adopts it and completes the transition before returning the save warning.
// Caller holds r.mu.
func (r *policyRuntime) persistAvailabilityLocked(state nodepolicy.Availability, origin nodepolicy.PauseOrigin) (policy nodepolicy.Policy, persisted bool, err error) {
	policy = r.current
	policy.Availability = state
	policy.PausedBy = ""
	if state == nodepolicy.Paused {
		policy.PausedBy = origin
	}
	if r.recoveryRequired {
		return policy, false, nil
	}
	policy, err = r.publishLocked(policy)
	return policy, err == nil || errors.Is(err, durablefile.ErrReplaced), err
}

// pushAvailabilityAndActivate sends the availability, then, for a change that
// was saved, the policy carrying its revision, and announces the outcome. An
// unsaved change (during recovery) has no revision of its own: it announces only
// when the availability answer changed what activation reports.
func (b *Broker) pushAvailabilityAndActivate(persisted bool) {
	changed := b.pushPolicyNow(context.Background(), pushAvailability)
	if persisted {
		b.activatePolicyNow()
		b.notifyPolicyChanged()
		return
	}
	b.announceActivationChange(changed)
}

// pauseNode drains when configured, then asks each stop owner to prove shutdown.
// Local retries re-read inventory even after a stored or previously completed
// pause: a replacement manager may have discovered another running engine.
func (b *Broker) pauseNode(ctx context.Context, origin nodepolicy.PauseOrigin, mode nodepolicy.PauseMode) (nodepolicy.Availability, error) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	policy, persisted := r.current, false
	var err error
	// Recheck physical shutdown on a repeated pause without assigning another
	// revision to an already completed transition. A failed proxy activation
	// still takes the ordinary publication path so a retry can activate it.
	if r.current.Availability != nodepolicy.Paused || r.current.PausedBy != origin ||
		(mode != nodepolicy.PauseImmediate && (r.live != nodepolicy.Paused || r.availabilityErr != "" || r.recoveryRequired)) {
		policy, persisted, err = r.persistAvailabilityLocked(nodepolicy.Paused, origin)
	}
	if err != nil && !persisted {
		r.mu.Unlock()
		return "", fmt.Errorf("the paused state could not be saved: %w", err)
	}
	r.current = policy
	if persisted {
		r.recoveryRequired = false
		r.activationErr = ""
	}
	r.live = nodepolicy.Draining
	r.cancelActive = mode == nodepolicy.PauseImmediate || policy.Pause.OnActive == nodepolicy.CancelActive
	pause := policy.Pause
	r.mu.Unlock()
	b.publishNodeProjection()
	slog.Info("pausing node", "onActive", pause.OnActive, "drainTimeoutSeconds", pause.DrainTimeoutSeconds)

	// Pushes use their own context: a superseded pause still leaves the proxy
	// with the state the next transition then replaces. Availability goes
	// first, so admission stops before anything else; the policy follows, its
	// revision being the one this change took, and the announcement after it
	// says whether the proxy activated that revision, which needs both pushes
	// acknowledged.
	b.pushAvailabilityAndActivate(persisted)
	b.notifyAvailability()

	idle := func(st nodepolicy.AdmissionState) bool { return st.Active == 0 }
	drainTimeout := time.Duration(pause.DrainTimeoutSeconds) * time.Second
	if mode != nodepolicy.PauseImmediate && !b.waitAdmission(ctx, drainTimeout, idle) {
		if ctx.Err() != nil {
			return "", errAvailabilitySuperseded
		}
		slog.Info("drain timed out; cancelling running work")
		r.mu.Lock()
		r.cancelActive = true
		r.mu.Unlock()
		b.announceActivationChange(b.pushPolicyNow(context.Background(), pushAvailability))
		b.waitAdmission(ctx, pauseCancelSettle, idle)
	}
	if ctx.Err() != nil {
		return "", errAvailabilitySuperseded
	}

	// Recovery retries reservation release and can verify shutdown on a hidden
	// replacement. Ordinary pauses wait for accepted settings/switch cleanup
	// before intent-neutral sleep. The durable pause fence remains in force.
	recovered, shutdownErr := b.prepareSwitchPause()
	if !recovered && shutdownErr == nil {
		b.settingsApplyMu.Lock()
		worker, source := b.pauseManager()
		shutdownErr = b.pauseEnginesOn(ctx, worker)
		if shutdownErr == nil && ctx.Err() == nil {
			shutdownErr = b.publishPhysicalPause(ctx, worker, source)
		}
		b.settingsApplyMu.Unlock()
	}
	if ctx.Err() != nil {
		return "", errAvailabilitySuperseded
	}
	if shutdownErr != nil {
		b.announceActivationChange(b.pushPolicyNow(context.Background(), pushAvailability))
		b.notifyAvailability()
		if !strings.HasPrefix(shutdownErr.Error(), "Pause could not stop every engine: ") {
			detail := "switch recovery is incomplete: " + shutdownErr.Error()
			if len(detail) > 512 {
				detail = detail[:512]
			}
			shutdownErr = fmt.Errorf("Pause could not stop every engine: %s", detail)
		}
		return nodepolicy.Draining, shutdownErr
	}
	r.mu.Lock()
	proved := r.live == nodepolicy.Paused
	r.mu.Unlock()
	if !proved {
		return nodepolicy.Draining, errors.New("Pause could not stop every engine: engine manager changed during shutdown")
	}
	b.announceActivationChange(b.pushPolicyNow(context.Background(), pushAvailability))
	b.notifyAvailability()
	slog.Info("node paused")
	if err != nil {
		return nodepolicy.Paused, &policySaveError{err: err, revision: policy.Revision}
	}
	return nodepolicy.Paused, nil
}

// Pin the actual worker/source pair, including a hidden recovering manager.
func (b *Broker) pauseManager() (*rpcWorker, *engineManagerSource) {
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	return b.engineMgr, b.engineMgrSource
}

// Retirement and publication share r.mu. Worker replacement shares workersMu;
// neither may intervene between validating the proof's source and publishing it.
func (b *Broker) publishPhysicalPause(ctx context.Context, worker *rpcWorker, source *engineManagerSource) error {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if ctx.Err() != nil {
		return errAvailabilitySuperseded
	}
	if worker == nil || source == nil || r.engineManagerSource != source || b.engineMgr != worker || b.engineMgrSource != source {
		return errors.New("Pause could not stop every engine: engine manager changed during shutdown")
	}
	select {
	case <-worker.Done():
		return errors.New("Pause could not stop every engine: engine manager exited during shutdown")
	default:
	}
	if r.current.Availability != nodepolicy.Paused {
		return errAvailabilitySuperseded
	}
	r.live = nodepolicy.Paused
	r.cancelActive = false
	return nil
}

// pauseEnginesOn also serves recovery while the replacement remains hidden.
// Every RPC uses the same worker, so replacement cannot combine two inventories
// into a shutdown proof. The caller serializes lifecycle and transitions.
func (b *Broker) pauseEnginesOn(ctx context.Context, worker *rpcWorker) error {
	r := b.nodePolicyRuntime()
	known := make(map[string]bool)
	for _, engine := range b.runningSnapshot() {
		known[engine] = true
	}
	for engine := range b.loadedSnapshot() {
		known[engine] = true
	}
	r.mu.Lock()
	for engine := range r.intent {
		known[engine] = true
	}
	r.mu.Unlock()
	var failures []string
	// running:false can also mean an unprobed or failed-reclaim orphan.
	// Only explicit physical-stop evidence may satisfy a failed Sleep.
	proved := make(map[string]bool)
	var stops sync.WaitGroup
	var proofMu sync.Mutex
	requestSleep := func(engine string) {
		// A lost response can still have stopped the engine. A local Resume
		// must retry saved-On wake through the manager's normal guards.
		r.mu.Lock()
		r.pauseSlept[engine] = true
		r.mu.Unlock()
		stops.Add(1)
		go func() {
			defer stops.Done()
			// Only the manager owns adoption and model residency. Never unload
			// here, or let one slow engine delay another engine's stop request.
			raw, err := callEngineOn(ctx, worker, engineSleepMethod, map[string]string{"engine": engine})
			var status struct {
				Engine              string `json:"engine"`
				Running             *bool  `json:"running"`
				TerminationVerified bool   `json:"terminationVerified"`
			}
			if err == nil && json.Unmarshal(raw, &status) == nil && status.Engine == engine && status.Running != nil && !*status.Running && status.TerminationVerified {
				proofMu.Lock()
				proved[engine] = true
				proofMu.Unlock()
			} else {
				slog.Warn("could not verify engine stop for pause", "engine", engine, "err", err)
			}
		}()
	}
	// Begin known engines before a status sweep can wait on a different
	// engine's lifecycle/file operation. Notifications can lag, so also stop
	// every engine newly found in the same manager's fresh inventory.
	for engine := range known {
		requestSleep(engine)
	}
	engines, inventoryErr := pauseEngineInventory(ctx, worker)
	if inventoryErr != nil {
		failures = append(failures, "engine inventory unavailable")
	}
	for engine := range engines {
		if !known[engine] {
			known[engine] = true
			requestSleep(engine)
		}
	}
	stops.Wait()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	stopped, verifyErr := pauseEngineInventory(ctx, worker)
	b.workersMu.Lock()
	current := worker != nil && b.engineMgr == worker
	b.workersMu.Unlock()
	if current {
		select {
		case <-worker.Done():
			current = false
		default:
		}
	}
	if verifyErr != nil || !current {
		failures = append(failures, "engine shutdown readback unavailable")
	} else {
		for engine, status := range stopped {
			known[engine] = true
			if status.Running {
				failures = append(failures, engine+": still running")
			} else if !proved[engine] && !status.TerminationVerified {
				failures = append(failures, engine+": stop unverified")
			}
		}
		for engine := range known {
			if _, reported := stopped[engine]; !reported {
				failures = append(failures, engine+": stop unverified")
			}
		}
	}
	if len(failures) > 0 {
		sort.Strings(failures)
		detail := strings.Join(failures, "; ")
		if len(detail) > 512 {
			detail = detail[:512]
		}
		return fmt.Errorf("Pause could not stop every engine: %s", detail)
	}
	return ctx.Err()
}

// resumeNode makes the node available again and starts the engines a pause
// stopped. Their saved intent was never changed, so engine:wake (and, after a
// launch that skipped restoration, engine:restore-enabled) starts exactly the
// ones saved On. A paired node's resume is checked against a local pause again
// here, where it would take effect.
func (b *Broker) resumeNode(ctx context.Context, gen uint64, origin nodepolicy.PauseOrigin) (nodepolicy.Availability, error) {
	// Local Resume may supersede a pause before physical completion. Retire its
	// switch obligation only after conservative cleanup, without inventing a
	// successful pause or preventing recovery from an incomplete shutdown.
	if origin == nodepolicy.PausedLocally {
		if err := b.supersedeSwitchPauseObligations(gen); err != nil {
			return "", err
		}
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	// A later Pause may have registered after the transition's initial check.
	// Only the still-current Resume may release its fence or publish available.
	if r.transitionGen != gen || ctx.Err() != nil {
		r.mu.Unlock()
		return "", errAvailabilitySuperseded
	}
	if origin == nodepolicy.PausedByPeer && r.managementBlockedLocked() {
		r.mu.Unlock()
		return "", &localVetoError{}
	}
	if r.live == nodepolicy.Available && r.current.Availability == nodepolicy.Available && r.availabilityErr == "" {
		// A Resume can supersede a pause before it saves anything. This is
		// still the authoritative transition that releases its pending fence.
		r.pendingPauseRequested = false
		r.localPauseRequested = false
		r.mu.Unlock()
		return nodepolicy.Available, nil
	}
	policy, persisted, err := r.persistAvailabilityLocked(nodepolicy.Available, origin)
	if err != nil && !persisted {
		r.mu.Unlock()
		return "", fmt.Errorf("the available state could not be saved: %w", err)
	}
	r.current = policy
	if persisted {
		r.recoveryRequired = false
		r.activationErr = ""
	}
	r.live = nodepolicy.Available
	r.pendingPauseRequested = false
	r.localPauseRequested = false
	r.cancelActive = false
	slept := make([]string, 0, len(r.pauseSlept))
	for engine := range r.pauseSlept {
		slept = append(slept, engine)
	}
	sort.Strings(slept)
	r.pauseSlept = map[string]bool{}
	restore := r.restoreSkipped
	r.restoreSkipped = false
	r.mu.Unlock()
	b.publishNodeProjection()

	b.pushAvailabilityAndActivate(persisted)
	b.notifyAvailability()
	slog.Info("node resumed", "waking", slept, "restore", restore)
	go b.wakeAfterResume(slept, restore)
	if err != nil {
		return nodepolicy.Available, &policySaveError{err: err, revision: policy.Revision}
	}
	return nodepolicy.Available, nil
}

func (b *Broker) wakeAfterResume(slept []string, restore bool) {
	if restore {
		if w := b.getEngineMgr(); w != nil {
			if err := w.Notify(restoreEnabledEnginesMethod, nil); err != nil {
				slog.Warn("failed to request enabled-engine restoration after resume", "err", err)
			}
		}
	}
	for _, engine := range slept {
		if _, err := b.callEngine(context.Background(), engineWakeMethod, map[string]string{"engine": engine}); err != nil {
			slog.Warn("could not start engine after resume", "engine", engine, "err", err)
		}
	}
}

// skipRestoreWhilePaused reports whether engine restoration must be skipped
// because the node is paused, and records that resuming restores them instead.
func (b *Broker) skipRestoreWhilePaused() bool {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	skip := r.managementBlockedLocked()
	if skip {
		r.restoreSkipped = true
	}
	return skip
}

// pauseEngineInventory requires explicit running readback for every reported
// engine. Missing data is unknown, never evidence of a stopped engine.
type pauseEngineStatus struct {
	Running             bool
	TerminationVerified bool
}

func pauseEngineInventory(ctx context.Context, worker *rpcWorker) (map[string]pauseEngineStatus, error) {
	raw, err := callEngineOn(ctx, worker, "engine:get-installed", nil)
	if err != nil {
		return nil, err
	}
	var inventory struct {
		Engines *[]struct {
			Engine              string `json:"engine"`
			Running             *bool  `json:"running"`
			TerminationVerified bool   `json:"terminationVerified"`
		} `json:"engines"`
	}
	if err := json.Unmarshal(raw, &inventory); err != nil || inventory.Engines == nil {
		return nil, errors.New("engine inventory is incomplete")
	}
	engines := make(map[string]pauseEngineStatus)
	for _, engine := range *inventory.Engines {
		if engine.Engine == "" || engine.Running == nil {
			return nil, errors.New("engine running state is incomplete")
		}
		if _, duplicate := engines[engine.Engine]; duplicate {
			return nil, errors.New("engine inventory contains duplicate engines")
		}
		engines[engine.Engine] = pauseEngineStatus{Running: *engine.Running, TerminationVerified: engine.TerminationVerified}
	}
	return engines, nil
}

func (b *Broker) loadedSnapshot() map[string][]string {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string][]string, len(r.loaded))
	for engine, models := range r.loaded {
		if len(models) > 0 {
			out[engine] = append([]string{}, models...)
		}
	}
	return out
}

func (b *Broker) runningSnapshot() []string {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, 0, len(r.running))
	for engine := range r.running {
		out = append(out, engine)
	}
	sort.Strings(out)
	return out
}
