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
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

// errAvailabilitySuperseded is returned to a caller whose transition was
// replaced by a newer one before it finished.
var errAvailabilitySuperseded = errors.New("superseded by a later node:set-availability request")

// pauseCancelSettle bounds how long a pause waits for cancelled work to
// unwind after the drain timeout, before it unloads and stops anyway.
var pauseCancelSettle = 10 * time.Second

// setAvailability moves the node to state and returns once it is reached. The
// transition itself does not depend on ctx: a caller that stops waiting (a
// disconnected client or peer) does not abandon a half-finished pause.
//
// origin is who asked: this node's own clients or a paired node. A node paused
// locally, or with a local pause accepted and not yet finished, refuses a paired
// node's resume with localVetoError, before the request can supersede anything.
// A paired node's pause over a local one stays local, so it never downgrades
// the veto.
func (b *Broker) setAvailability(ctx context.Context, state nodepolicy.Availability, origin nodepolicy.PauseOrigin) (nodepolicy.Availability, error) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if origin == nodepolicy.PausedByPeer && (pausedLocally(r.current) || r.localPauseRequested) {
		if state == nodepolicy.Available {
			r.mu.Unlock()
			return "", &localVetoError{}
		}
		origin = nodepolicy.PausedLocally
	}
	r.localPauseRequested = state == nodepolicy.Paused && origin == nodepolicy.PausedLocally
	r.transitionGen++
	gen := r.transitionGen
	if r.transitionCancel != nil {
		r.transitionCancel()
	}
	tctx, cancel := context.WithCancel(context.Background())
	r.transitionCancel = cancel
	r.mu.Unlock()

	type outcome struct {
		availability nodepolicy.Availability
		err          error
	}
	done := make(chan outcome, 1)
	go func() {
		defer cancel()
		availability, err := b.runAvailabilityTransition(tctx, gen, state, origin)
		// The transition saved its provenance or failed before saving any;
		// either way the stored policy now says whether a local pause holds.
		r.mu.Lock()
		if r.transitionGen == gen {
			r.localPauseRequested = false
		}
		r.mu.Unlock()
		done <- outcome{availability, err}
	}()
	select {
	case o := <-done:
		return o.availability, o.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (b *Broker) runAvailabilityTransition(ctx context.Context, gen uint64, state nodepolicy.Availability, origin nodepolicy.PauseOrigin) (nodepolicy.Availability, error) {
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
		return b.pauseNode(ctx, origin)
	}
	return b.resumeNode(origin)
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

// pauseNode drains, then unloads and stops engines as the policy says. A node
// already paused is paused again only when origin changes whether the pause is
// local, which records the new provenance.
func (b *Broker) pauseNode(ctx context.Context, origin nodepolicy.PauseOrigin) (nodepolicy.Availability, error) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if r.live == nodepolicy.Paused && r.current.Availability == nodepolicy.Paused && r.availabilityErr == "" &&
		pausedLocally(r.current) == (origin == nodepolicy.PausedLocally) {
		r.mu.Unlock()
		return nodepolicy.Paused, nil
	}
	policy, persisted, err := r.persistAvailabilityLocked(nodepolicy.Paused, origin)
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
	slog.Info("pausing node", "onActive", pause.OnActive, "drainTimeoutSeconds", pause.DrainTimeoutSeconds,
		"unloadModels", pause.UnloadModels, "stopEngines", pause.StopEngines)

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

	if pause.UnloadModels {
		for engine, models := range b.loadedSnapshot() {
			if ctx.Err() != nil {
				return "", errAvailabilitySuperseded
			}
			b.unloadModels(engine, models, "pause")
		}
	}
	if pause.StopEngines {
		for _, engine := range b.runningSnapshot() {
			if ctx.Err() != nil {
				return "", errAvailabilitySuperseded
			}
			if _, err := b.callEngine(ctx, engineSleepMethod, map[string]string{"engine": engine}); err != nil {
				if ctx.Err() != nil {
					// Superseded mid-stop: engine-manager may still finish the
					// stop, so let the resume that superseded us wake it.
					// Transitions are serialized, so it reads this afterwards.
					r.mu.Lock()
					r.pauseSlept[engine] = true
					r.mu.Unlock()
					return "", errAvailabilitySuperseded
				}
				slog.Warn("could not stop engine for pause", "engine", engine, "err", err)
				continue
			}
			r.mu.Lock()
			r.pauseSlept[engine] = true
			r.mu.Unlock()
		}
	}
	if ctx.Err() != nil {
		return "", errAvailabilitySuperseded
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

// resumeNode makes the node available again and starts the engines a pause
// stopped. Their saved intent was never changed, so engine:wake (and, after a
// launch that skipped restoration, engine:restore-enabled) starts exactly the
// ones saved On. A paired node's resume is checked against a local pause again
// here, where it would take effect.
func (b *Broker) resumeNode(origin nodepolicy.PauseOrigin) (nodepolicy.Availability, error) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if origin == nodepolicy.PausedByPeer && pausedLocally(r.current) {
		r.mu.Unlock()
		return "", &localVetoError{}
	}
	if r.live == nodepolicy.Available && r.current.Availability == nodepolicy.Available && r.availabilityErr == "" {
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
// because the node is paused with pause.stopEngines, and records that resuming
// has to restore them instead.
func (b *Broker) skipRestoreWhilePaused() bool {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	skip := r.current.Availability == nodepolicy.Paused && r.current.Pause.StopEngines
	if skip {
		r.restoreSkipped = true
	}
	return skip
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
