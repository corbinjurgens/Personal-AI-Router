// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"nvpair-shared/nodepolicy"
)

type strictAdmissionError struct{ Reason string }

func (e *strictAdmissionError) Error() string { return "strict admission: " + e.Reason }
func (e *strictAdmissionError) Is(target error) bool {
	t, ok := target.(*strictAdmissionError)
	return ok && t.Reason == e.Reason
}

var (
	errStrictUnavailable = &strictAdmissionError{Reason: "unavailable"}
	errStrictOccupied    = &strictAdmissionError{Reason: "occupied"}
	errStrictCold        = &strictAdmissionError{Reason: "cold-residency"}
)

type strictTimer interface {
	Chan() <-chan time.Time
	Stop()
}
type realStrictTimer struct{ t *time.Timer }

func (t realStrictTimer) Chan() <-chan time.Time { return t.t.C }
func (t realStrictTimer) Stop()                  { t.t.Stop() }

var timeNow = time.Now

func newStrictTimer(d time.Duration) strictTimer { return realStrictTimer{time.NewTimer(d)} }

type nodeStrictModel = nodepolicy.StrictAdmissionModel
type strictAdmissionState struct {
	models     map[admKey]nodeStrictModel
	generation map[admKey]uint64
	markers    map[uint64]capacityMarker
	path       string
	storageErr error
	now        func() time.Time
	timer      func(time.Duration) strictTimer
	write      func(string, []byte, os.FileMode) error
	read       func(string) ([]byte, error)
}

// installStrictCapability is an internal authoritative input seam, deliberately
// not exposed as an RPC or derived from the caller's header. The feed and
// qualification owner are subsequent slices. Until installed, strict fails closed.
func (a *admissionController) installStrictCapability(c nodepolicy.StrictAdmissionCapability) error {
	if err := c.Validate(); err != nil {
		return err
	}
	models := make(map[admKey]nodeStrictModel)
	for _, m := range c.Models {
		k := admKey{engine: m.Engine, model: a.normalize(m.Engine, m.Model)}
		if !a.validStrictModelKey(k) {
			return errors.New("invalid strict model key")
		}
		if _, exists := models[k]; exists {
			return errors.New("duplicate normalized strict model")
		}
		m.Model = k.model
		models[k] = m
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for k, m := range models {
		if limit := a.concurrencyLimitLocked(k); limit <= 0 || limit != m.MaxConcurrent {
			return errors.New("strict capacity must match finite effective policy/profile concurrency")
		}
	}
	for k := range a.strict.models {
		a.strict.generation[k]++
	}
	for k := range models {
		if _, replacing := a.strict.models[k]; !replacing {
			a.strict.generation[k]++
		}
	}
	a.strict.models = models
	a.signalLocked()
	return nil
}

func (a *admissionController) revalidateStrictCapabilitiesLocked() {
	for k, m := range a.strict.models {
		if limit := a.concurrencyLimitLocked(k); limit <= 0 || limit != m.MaxConcurrent {
			delete(a.strict.models, k)
			a.strict.generation[k]++
		}
	}
}

func (a *admissionController) StrictReadiness() nodepolicy.StrictAdmissionReadiness {
	return nodepolicy.StrictAdmissionReadiness{}
}

// AdmitStrict shares the concrete path's lock, in-flight counts, residency,
// notifications and tickets. Engine IDs are canonical built-in IDs; models use
// the existing engine adapter normalization (Ollama adds :latest, LM Studio exact).
func (a *admissionController) AdmitStrict(ctx context.Context, engine, model string, req nodepolicy.StrictAdmissionRequest) (strictAdmissionTicket, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	k := admKey{engine: engine, model: a.normalize(engine, model)}
	start := a.strict.now()
	pre := start.Add(time.Duration(req.PreOutputBudgetMs) * time.Millisecond)
	queue := start.Add(time.Duration(req.QueueBudgetMs) * time.Millisecond)
	cold := start.Add(time.Duration(req.ColdStartBudgetMs) * time.Millisecond)
	queued := false
	var waitedUntil time.Time
	var waitedReason error
	coldNeeded := false
	defer func() {
		if queued {
			a.queued--
			a.signalLocked()
		}
	}()
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		m, ok := a.strict.models[k]
		if !ok || m.MaxConcurrent < 1 || m.Identity != req.Identity || a.strict.storageErr != nil || a.availability != nodepolicy.Available || a.drain[engine] {
			return nil, errStrictUnavailable
		}
		if on, known := a.intent[engine]; known && !on {
			return nil, errStrictUnavailable
		}
		now := a.strict.now()
		if coldNeeded && !now.Before(cold) {
			return nil, errStrictCold
		}
		if !waitedUntil.IsZero() && !now.Before(waitedUntil) {
			return nil, waitedReason
		}
		if !now.Before(pre) {
			return nil, errStrictOccupied
		}
		var sends []notification
		deadline := pre
		reason := errStrictOccupied
		if a.inflight[k] >= m.MaxConcurrent {
			if queue.Before(deadline) {
				deadline = queue
			}
		} else {
			plan, evict := a.fitLocked(k, a.policy)
			// In-flight and quarantined capacity are not proof that the model
			// finished loading. Only the destination residency report is warm.
			_, unloading := a.unloadSentAt[k]
			resident := a.loaded[engine][k.model] && !unloading
			healthy := a.healthy(engine)
			if plan == fitNever {
				return nil, errStrictUnavailable
			}
			if healthy && resident && plan == fitOK {
				dispatchDeadline := pre
				if coldNeeded && cold.Before(dispatchDeadline) {
					dispatchDeadline = cold
				}
				now = a.strict.now()
				if !now.Before(dispatchDeadline) {
					if coldNeeded {
						return nil, errStrictCold
					}
					return nil, errStrictOccupied
				}
				if a.nextTicket == ^uint64(0) {
					return nil, errStrictUnavailable
				}
				a.nextTicket++
				base := &admissionTicket{a: a, id: a.nextTicket, key: k}
				a.tickets[base.id] = base
				a.inflight[k]++
				a.active++
				a.activeByEngine[engine]++
				a.lastActivity[engine] = now.UnixMilli()
				return &strictTicket{base: base, generation: a.strict.generation[k], identity: req.Identity, deadline: dispatchDeadline, ctx: ctx, warmOnly: true}, nil
			}
			if plan == fitWait {
				if queue.Before(deadline) {
					deadline = queue
				}
			} else {
				coldNeeded = true
				reason = errStrictCold
				if cold.Before(deadline) {
					deadline = cold
				}
				if !now.Before(deadline) {
					return nil, reason
				}
				if !healthy {
					if on, known := a.intent[engine]; !known || !on || !a.policy.Idle.StartOnDemand {
						return nil, errStrictUnavailable
					}
					if last, ok := a.wakeSentAt[engine]; !ok || now.Sub(last) >= wakeResendInterval {
						a.wakeSentAt[engine] = now
						sends = append(sends, notification{nodepolicy.NotifyAdmissionWake, nodepolicy.WakeRequest{Engine: engine}})
					}
				} else if plan == fitEvict && a.policy.Admission.SwitchModels {
					sends = a.unloadLocked(evict, a.policy)
				} else if plan == fitEvict {
					return nil, errStrictCold
				}
				// Cold unloaded models need no extra wake RPC: the inference
				// call loads them, but only with a nonzero cold allowance.
				if healthy && plan == fitOK && !unloading {
					now = a.strict.now()
					if !now.Before(deadline) {
						return nil, errStrictCold
					}
					if a.nextTicket == ^uint64(0) {
						return nil, errStrictUnavailable
					}
					a.nextTicket++
					base := &admissionTicket{a: a, id: a.nextTicket, key: k}
					a.tickets[base.id] = base
					a.inflight[k]++
					a.active++
					a.activeByEngine[engine]++
					a.lastActivity[engine] = now.UnixMilli()
					return &strictTicket{base: base, generation: a.strict.generation[k], identity: req.Identity, deadline: deadline, ctx: ctx}, nil
				}
			}
		}
		if coldNeeded && !cold.After(deadline) {
			deadline, reason = cold, errStrictCold
		}
		if !now.Before(deadline) {
			return nil, reason
		}
		if !queued {
			a.queued++
			queued = true
		}
		ch := a.changed
		waitedUntil, waitedReason = deadline, reason
		timer := a.strict.timer(deadline.Sub(now))
		st := a.noteChangeLocked()
		a.mu.Unlock()
		a.send(sends)
		a.sendState(st)
		select {
		case <-ch:
		case <-timer.Chan():
		case <-ctx.Done():
		}
		timer.Stop()
		a.mu.Lock()
	}
}

type strictTicket struct {
	base       *admissionTicket
	identity   nodepolicy.StrictModelIdentity
	generation uint64
	deadline   time.Time
	ctx        context.Context
	dispatched bool
	finished   bool
	warmOnly   bool
}

func (t *strictTicket) BeforeStrictDispatch() error {
	a := t.base.a
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	if t.finished {
		return errors.New("strict ticket already finished")
	}
	if t.dispatched {
		return nil
	}
	mc, ok := a.strict.models[t.base.key]
	if !ok || a.strict.generation[t.base.key] != t.generation {
		t.finishLocked(strictNotDispatched)
		return errStrictUnavailable
	}
	if !ok || mc.Identity != t.identity || a.availability != nodepolicy.Available || a.drain[t.base.key.engine] || !a.healthy(t.base.key.engine) {
		return errStrictUnavailable
	}
	if on, known := a.intent[t.base.key.engine]; known && !on {
		return errStrictUnavailable
	}
	if err := t.ctx.Err(); err != nil {
		return err
	}
	if !a.strict.now().Before(t.deadline) {
		return errStrictCold
	}
	if _, unloading := a.unloadSentAt[t.base.key]; t.warmOnly && (!a.loaded[t.base.key.engine][t.base.key.model] || unloading) {
		// Release as not-dispatched under the same lock, durably clearing
		// any marker from a previously failed dispatch attempt.
		t.finishLocked(strictNotDispatched)
		return errStrictCold
	}
	m := capacityMarker{ID: t.base.id, Engine: t.base.key.engine, Model: t.base.key.model}
	if _, exists := a.strict.markers[m.ID]; !exists && len(a.strict.markers) >= maxStrictCapacitySlots {
		t.finishLocked(strictNotDispatched)
		return errStrictUnavailable
	}
	a.strict.markers[m.ID] = m
	if err := a.persistStrictCapacityLocked(); err != nil {
		return fmt.Errorf("persist strict dispatch: %w", err)
	}
	if err := t.ctx.Err(); err != nil {
		return err
	}
	if !a.strict.now().Before(t.deadline) {
		return errStrictCold
	}
	t.dispatched = true
	return nil
}

// Bound connection, headers and first body with the same remaining monotonic
// deadline used by admission. A successful first byte ends this timer; it does
// not impose a duration limit on an active healthy generation.
func (t *strictTicket) watchPreOutput(cancel context.CancelFunc) func() {
	a := t.base.a
	a.mu.Lock()
	timer := a.strict.timer(t.deadline.Sub(a.strict.now()))
	a.mu.Unlock()
	done := make(chan struct{})
	var once sync.Once
	stop := func() { once.Do(func() { close(done); timer.Stop() }) }
	go func() {
		select {
		case <-timer.Chan():
			once.Do(func() { cancel(); close(done) })
		case <-t.ctx.Done():
			stop()
		case <-done:
		}
	}()
	return stop
}

func (a *admissionController) releaseStrictSlotLocked(k admKey) {
	a.inflight[k]--
	if a.inflight[k] == 0 {
		delete(a.inflight, k)
	}
	a.active--
	a.activeByEngine[k.engine]--
	if a.activeByEngine[k.engine] == 0 {
		delete(a.activeByEngine, k.engine)
	}
	if a.residencyKnown {
		a.finishedAt[k] = a.strict.now()
	}
	a.lastActivity[k.engine] = a.strict.now().UnixMilli()
}

func (t *strictTicket) Finish(e strictTerminalEvidence) {
	a := t.base.a
	a.mu.Lock()
	defer func() { st := a.noteChangeLocked(); a.mu.Unlock(); a.sendState(st) }()
	t.finishLocked(e)
}

func (t *strictTicket) finishLocked(e strictTerminalEvidence) {
	a := t.base.a
	if t.finished {
		return
	}
	t.finished = true
	delete(a.tickets, t.base.id)
	_, marked := a.strict.markers[t.base.id]
	if e == strictUncertain || (t.dispatched && e == strictNotDispatched) {
		a.strict.markers[t.base.id] = capacityMarker{ID: t.base.id, Engine: t.base.key.engine, Model: t.base.key.model}
		if err := a.persistStrictCapacityLocked(); err != nil {
			slog.Error("strict capacity remains quarantined after persistence failure", "err", err)
		}
		a.signalLocked()
		return
	}
	if e != strictNotDispatched && e != strictQualifiedComplete && e != strictAuthoritativeIdle && e != strictVerifiedReplacement {
		// Unknown evidence cannot release a possibly dispatched slot.
		a.strict.markers[t.base.id] = capacityMarker{ID: t.base.id, Engine: t.base.key.engine, Model: t.base.key.model}
		if err := a.persistStrictCapacityLocked(); err != nil {
			slog.Error("strict capacity quarantine persistence failed", "err", err)
		}
		return
	}
	if marked {
		m := a.strict.markers[t.base.id]
		delete(a.strict.markers, t.base.id)
		if err := a.persistStrictCapacityLocked(); err != nil {
			a.strict.markers[t.base.id] = m
			slog.Error("strict capacity clear failed; retaining quarantine", "err", err)
			return
		}
	}
	a.releaseStrictSlotLocked(t.base.key)
	a.signalLocked()
}

func parseStrictAdmission(h http.Header) (*nodepolicy.StrictAdmissionRequest, error) {
	var values []string
	present := false
	for k, v := range h {
		if strings.EqualFold(k, nodepolicy.StrictAdmissionHeader) {
			present = true
			values = append(values, v...)
		}
	}
	if !present {
		return nil, nil
	}
	if len(values) != 1 {
		return nil, errors.New("strict admission requires exactly one header value")
	}
	var req nodepolicy.StrictAdmissionRequest
	if err := json.Unmarshal([]byte(values[0]), &req); err != nil {
		return nil, fmt.Errorf("invalid strict admission header: %w", err)
	}
	return &req, nil
}

func stripInternalRoutingHeaders(h http.Header) {
	for k := range h {
		if strings.HasPrefix(strings.ToLower(k), "x-pair-") || strings.EqualFold(k, engineIdentityProbeHeader) {
			delete(h, k)
		}
	}
}

func writeStrictAdmissionError(w http.ResponseWriter, err error) {
	status := http.StatusServiceUnavailable
	if errors.Is(err, errStrictOccupied) {
		status = http.StatusTooManyRequests
	}
	if errors.Is(err, context.DeadlineExceeded) {
		status = http.StatusGatewayTimeout
	}
	writeIngressError(w, status, "strict-admission", err.Error())
}

// strictDestinationAdmission is A1's process-wide admission seam, shared by all
// facades and paired ingress. ingress.go:serveIngress and http_handler.go:handleHTTP's
// self dispatch will call it after trusted-contract parsing and model
// normalization, before wake/unload or inference. A1a implements the controller;
// the authoritative capability/qualification feed remains withdrawn until A1b/A2.
// Identity, availability, finite capacity and current qualification must all be
// revalidated atomically. Deadlines are monotonic, derived from remaining
// durations, and may only shrink. Concrete requests keep their existing path.
type strictDestinationAdmission interface {
	AdmitStrict(context.Context, string, string, nodepolicy.StrictAdmissionRequest) (strictAdmissionTicket, error)
	StrictReadiness() nodepolicy.StrictAdmissionReadiness
}

// strictAdmissionTicket distinguishes cancellation from proved termination.
// BeforeStrictDispatch must durably mark the engine dirty before any inference
// bytes are sent; failure prevents dispatch. Finish is idempotent. Uncertain
// work retains capacity across cancellation/restart until authoritative engine
// reconciliation; elapsed time alone never releases it.
type strictAdmissionTicket interface {
	BeforeStrictDispatch() error
	Finish(strictTerminalEvidence)
}

type strictTerminalEvidence string

const (
	strictNotDispatched       strictTerminalEvidence = "not-dispatched"
	strictQualifiedComplete   strictTerminalEvidence = "qualified-complete"
	strictAuthoritativeIdle   strictTerminalEvidence = "authoritative-idle"
	strictVerifiedReplacement strictTerminalEvidence = "verified-replacement"
	strictUncertain           strictTerminalEvidence = "uncertain"
)
