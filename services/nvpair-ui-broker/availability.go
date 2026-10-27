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
		return o.availability, o.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (b *Broker) runAvailabilityTransition(ctx context.Context, gen uint64, state nodepolicy.Availability, origin nodepolicy.PauseOrigin, mode nodepolicy.PauseMode) (nodepolicy.Availability, error) {
	if mode != nodepolicy.PauseConfigured {
		return "", errImmediatePauseUnavailable
	}
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
		availability, err := b.pauseNode(ctx, origin)
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

// pauseNode drains, then unconditionally unloads models and stops engines.
// Local retries re-read inventory even after a stored or previously completed
// pause: a replacement manager may have discovered another running engine.
func (b *Broker) pauseNode(ctx context.Context, origin nodepolicy.PauseOrigin) (nodepolicy.Availability, error) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	policy, persisted := r.current, false
	var err error
	// Recheck physical shutdown on a repeated pause without assigning another
	// revision to an already completed transition. A failed proxy activation
	// still takes the ordinary publication path so a retry can activate it.
	if r.current.Availability != nodepolicy.Paused || r.current.PausedBy != origin ||
		r.live != nodepolicy.Paused || r.availabilityErr != "" || r.recoveryRequired {
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
	r.cancelActive = policy.Pause.OnActive == nodepolicy.CancelActive
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
	if !b.waitAdmission(ctx, drainTimeout, idle) {
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
		worker := b.getEngineMgr()
		shutdownErr = b.pauseEnginesOn(ctx, worker)
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
	r.live = nodepolicy.Paused
	r.cancelActive = false
	r.mu.Unlock()
	b.announceActivationChange(b.pushPolicyNow(context.Background(), pushAvailability))
	b.notifyAvailability()
	slog.Info("node paused")
	if err != nil {
		return nodepolicy.Paused, &policySaveError{err: err, revision: policy.Revision}
	}
	return nodepolicy.Paused, nil
}

// pauseEnginesOn also serves recovery while the replacement remains hidden.
// Every RPC uses the same worker, so replacement cannot combine two inventories
// into a shutdown proof. The caller serializes lifecycle and transitions.
func (b *Broker) pauseEnginesOn(ctx context.Context, worker *rpcWorker) error {
	r := b.nodePolicyRuntime()
	// Notifications may lag an activation accepted before the pause projection.
	engines, inventoryErr := pauseEngineInventory(ctx, worker)
	known := make(map[string]bool)
	for _, engine := range b.runningSnapshot() {
		known[engine] = true
	}
	for engine := range engines {
		known[engine] = true
	}
	loaded := b.loadedSnapshot()
	models, modelErr := pauseModelInventory(ctx, worker)
	for engine, fresh := range models {
		loaded[engine] = append(loaded[engine], fresh...)
	}
	for engine, models := range loaded {
		known[engine] = true
		if ctx.Err() != nil {
			return ctx.Err()
		}
		seen := make(map[string]bool)
		var unique []string
		for _, model := range models {
			if !seen[model] {
				seen[model] = true
				unique = append(unique, model)
			}
		}
		for _, model := range unique {
			if model == "" {
				continue
			}
			if _, err := callEngineOn(ctx, worker, engineUnloadModelMethod, map[string]string{"engine": engine, "model": model}); err != nil {
				slog.Warn("model unload failed", "engine", engine, "reason", "pause", "err", err)
			}
		}
	}
	var failures []string
	// running:false can also mean an unprobed or failed-reclaim orphan.
	// Only explicit physical-stop evidence may satisfy a failed Sleep.
	stopFailed := make(map[string]bool)
	if inventoryErr != nil {
		failures = append(failures, "engine inventory unavailable")
	}
	// Inventory failure cannot hide known engines; stop the others as well.
	ordered := make([]string, 0, len(known))
	for engine := range known {
		ordered = append(ordered, engine)
	}
	sort.Strings(ordered)
	for _, engine := range ordered {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// A lost response can still have stopped the engine. A local Resume
		// must retry saved-On wake through the manager's normal guards.
		r.mu.Lock()
		r.pauseSlept[engine] = true
		r.mu.Unlock()
		if _, err := callEngineOn(ctx, worker, engineSleepMethod, map[string]string{"engine": engine}); err != nil {
			if ctx.Err() != nil {
				// Superseded mid-stop: engine-manager may still finish the
				// stop, so let the resume that superseded us wake it.
				// Transitions are serialized, so it reads this afterwards.
				return ctx.Err()
			}
			slog.Warn("could not stop engine for pause", "engine", engine, "err", err)
			stopFailed[engine] = true
			continue
		}
	}
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
		for engine := range stopFailed {
			failures = append(failures, engine+": stop failed")
		}
	} else {
		for engine, status := range stopped {
			known[engine] = true
			if status.Running {
				failures = append(failures, engine+": still running")
			} else if stopFailed[engine] && !status.TerminationVerified {
				failures = append(failures, engine+": stop unverified")
			}
		}
		for engine := range known {
			if _, reported := stopped[engine]; !reported {
				failures = append(failures, engine+": stop unverified")
			}
		}
	}
	// Unload failures are satisfied by verified engine shutdown. Without a
	// model readback and shutdown proof we cannot claim physical pause.
	if modelErr != nil && (verifyErr != nil || len(failures) > 0) {
		failures = append(failures, "loaded model inventory unavailable")
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

func pauseModelInventory(ctx context.Context, worker *rpcWorker) (map[string][]string, error) {
	raw, err := callEngineOn(ctx, worker, "engine:models", nil)
	if err != nil {
		return nil, err
	}
	var inventory struct {
		LoadedByEngine map[string][]string `json:"loadedByEngine"`
	}
	if err := json.Unmarshal(raw, &inventory); err != nil || inventory.LoadedByEngine == nil {
		return nil, errors.New("loaded model inventory is incomplete")
	}
	return inventory.LoadedByEngine, nil
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
