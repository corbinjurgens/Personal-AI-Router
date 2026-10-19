// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Unit coverage for the node-wide admission controller, driven directly so
// each rule in FORK_DESIGN.md §3.3 is pinned without HTTP in the way. The
// ingress and self-dispatch tests in admission_http_test.go prove the same
// controller is what both entry points consult.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func strictRequest() nodepolicy.StrictAdmissionRequest {
	return nodepolicy.StrictAdmissionRequest{Version: 1, QueueBudgetMs: 1000, ColdStartBudgetMs: 1000, PreOutputBudgetMs: 1000,
		Request:  nodepolicy.RoutingRequestKey{OriginNodeID: "origin", ProxyRunID: "run", RequestID: "1"},
		Identity: nodepolicy.StrictModelIdentity{EngineRunID: "engine", SettingsFingerprint: strings.Repeat("a", 64), ModelDigest: strings.Repeat("b", 64)}}
}

func configureStrict(t *testing.T, a *admissionController, path string) {
	t.Helper()
	a.strict.path = path
	a.strict.storageErr = nil
	p := a.currentPolicy()
	p.Admission.MaxConcurrentPerModel = 1
	a.setPolicy(p)
	if err := a.installStrictCapability(nodepolicy.StrictAdmissionCapability{Version: 1, Models: []nodepolicy.StrictAdmissionModel{{Engine: "ollama", Model: "m", MaxConcurrent: 1, Identity: strictRequest().Identity}}}); err != nil {
		t.Fatal(err)
	}
	a.setResidency(map[string][]string{"ollama": {"m"}})
}

func newStrictHarness(t *testing.T) *admHarness {
	t.Helper()
	h := newAdmHarness(t)
	configureStrict(t, h.a, filepath.Join(t.TempDir(), "capacity.json"))
	return h
}

type manualStrictClock struct {
	mu      sync.Mutex
	now     time.Time
	timers  []*manualStrictTimer
	created chan time.Duration
}
type manualStrictTimer struct {
	ch      chan time.Time
	at      time.Time
	stopped bool
	clock   *manualStrictClock
}

func (t *manualStrictTimer) Chan() <-chan time.Time { return t.ch }
func (t *manualStrictTimer) Stop()                  { t.clock.mu.Lock(); defer t.clock.mu.Unlock(); t.stopped = true }
func (c *manualStrictClock) Now() time.Time         { c.mu.Lock(); defer c.mu.Unlock(); return c.now }
func (c *manualStrictClock) Timer(d time.Duration) strictTimer {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := &manualStrictTimer{ch: make(chan time.Time, 1), at: c.now.Add(d), clock: c}
	c.timers = append(c.timers, t)
	c.created <- d
	return t
}
func (c *manualStrictClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
	for _, t := range c.timers {
		if !t.stopped && !c.now.Before(t.at) {
			t.ch <- c.now
			t.stopped = true
		}
	}
}
func useStrictClock(a *admissionController) *manualStrictClock {
	c := &manualStrictClock{now: time.Now(), created: make(chan time.Duration, 32)}
	a.strict.now = c.Now
	a.strict.timer = c.Timer
	return c
}
func awaitStrictTimer(t *testing.T, c *manualStrictClock) time.Duration {
	t.Helper()
	select {
	case d := <-c.created:
		return d
	case <-time.After(3 * time.Second):
		t.Fatal("strict timer not registered")
		return 0
	}
}

type strictResult struct {
	ticket strictAdmissionTicket
	err    error
}

func asyncStrict(a *admissionController, ctx context.Context, req nodepolicy.StrictAdmissionRequest) <-chan strictResult {
	ch := make(chan strictResult, 1)
	go func() { ticket, err := a.AdmitStrict(ctx, "ollama", "m", req); ch <- strictResult{ticket, err} }()
	return ch
}
func awaitStrict(t *testing.T, ch <-chan strictResult) strictResult {
	t.Helper()
	select {
	case r := <-ch:
		return r
	case <-time.After(3 * time.Second):
		t.Fatal("strict admission stuck")
		return strictResult{}
	}
}
func mustStrictTicket(t *testing.T, a *admissionController) strictAdmissionTicket {
	t.Helper()
	ticket, err := a.AdmitStrict(context.Background(), "ollama", "m", strictRequest())
	if err != nil {
		t.Fatal(err)
	}
	return ticket
}

func TestStrictSharesConcreteCapacity(t *testing.T) {
	t.Run("strict first", func(t *testing.T) {
		h := newStrictHarness(t)
		first := mustStrictTicket(t, h.a)
		defer first.Finish(strictNotDispatched)
		mustReject(t, h.admit("ollama", "m", 0), nodepolicy.RejectBusy)
		if h.a.currentActive() != 1 {
			t.Fatal("concrete admission exceeded strict-held capacity")
		}
	})
	t.Run("concrete first", func(t *testing.T) {
		h := newStrictHarness(t)
		first := h.admit("ollama", "m", -1)
		if first.ticket == nil {
			t.Fatalf("concrete admission=%+v", first)
		}
		defer first.ticket.release()
		req := strictRequest()
		req.QueueBudgetMs = 0
		if ticket, err := h.a.AdmitStrict(context.Background(), "ollama", "m", req); ticket != nil || !errors.Is(err, errStrictOccupied) {
			t.Fatalf("ticket=%v error=%v", ticket, err)
		}
		if h.a.currentActive() != 1 {
			t.Fatal("strict admission exceeded concrete-held capacity")
		}
	})
}

func TestStrictCapabilityMatchesEffectivePolicy(t *testing.T) {
	for _, scenario := range []string{"unlimited", "global mismatch", "profile mismatch", "matching profile"} {
		t.Run(scenario, func(t *testing.T) {
			h := newStrictHarness(t)
			p := h.a.currentPolicy()
			p.Admission.MaxConcurrentPerModel = 2
			switch scenario {
			case "unlimited":
				p.Admission.MaxConcurrentPerModel = 0
			case "profile mismatch":
				p.Admission.MaxConcurrentPerModel = 1
				p.Profiles = []nodepolicy.Profile{{Engine: "ollama", Model: "m", MaxConcurrent: 2}}
			case "matching profile":
				p.Profiles = []nodepolicy.Profile{{Engine: "ollama", Model: "m", MaxConcurrent: 1}}
			}
			if err := h.a.activatePolicy(p); err != nil {
				t.Fatal(err)
			}
			err := h.a.installStrictCapability(nodepolicy.StrictAdmissionCapability{Version: 1, Models: []nodepolicy.StrictAdmissionModel{{Engine: "ollama", Model: "m:latest", MaxConcurrent: 1, Identity: strictRequest().Identity}}})
			if scenario == "matching profile" {
				if err != nil {
					t.Fatal(err)
				}
				ticket := mustStrictTicket(t, h.a)
				ticket.Finish(strictNotDispatched)
			} else if err == nil {
				t.Fatal("mismatched capability installed")
			}
		})
	}
}

func TestStrictPolicyChangeWithdrawsCapability(t *testing.T) {
	h := newStrictHarness(t)
	ticket := mustStrictTicket(t, h.a)
	p := h.a.currentPolicy()
	p.Admission.MaxConcurrentPerModel = 2
	if err := h.a.activatePolicy(p); err != nil {
		t.Fatal(err)
	}
	if err := ticket.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("dispatch error=%v", err)
	}
	ticket.Finish(strictNotDispatched)
	if next, err := h.a.AdmitStrict(context.Background(), "ollama", "m", strictRequest()); next != nil || !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("ticket=%v error=%v", next, err)
	}
}

func TestStrictCapabilityGenerationFencesHeldTickets(t *testing.T) {
	for _, scenario := range []string{"policy withdrawal and reinstall", "same capability replacement", "explicit withdrawal and reinstall"} {
		t.Run(scenario, func(t *testing.T) {
			h := newStrictHarness(t)
			p := h.a.currentPolicy()
			p.Admission.MaxConcurrentPerModel = 2
			if err := h.a.activatePolicy(p); err != nil {
				t.Fatal(err)
			}
			capability := nodepolicy.StrictAdmissionCapability{Version: 1, Models: []nodepolicy.StrictAdmissionModel{{Engine: "ollama", Model: "m", MaxConcurrent: 2, Identity: strictRequest().Identity}}}
			if err := h.a.installStrictCapability(capability); err != nil {
				t.Fatal(err)
			}
			first, second := mustStrictTicket(t, h.a), mustStrictTicket(t, h.a)
			if scenario == "policy withdrawal and reinstall" {
				p.Admission.MaxConcurrentPerModel = 1
				if err := h.a.activatePolicy(p); err != nil {
					t.Fatal(err)
				}
				capability.Models[0].MaxConcurrent = 1
			} else if scenario == "explicit withdrawal and reinstall" {
				if err := h.a.installStrictCapability(nodepolicy.StrictAdmissionCapability{Version: 1, Models: []nodepolicy.StrictAdmissionModel{{Engine: "lmstudio", Model: "other", MaxConcurrent: 2, Identity: strictRequest().Identity}}}); err != nil {
					t.Fatal(err)
				}
			}
			if err := h.a.installStrictCapability(capability); err != nil {
				t.Fatal(err)
			}
			for _, ticket := range []strictAdmissionTicket{first, second} {
				if err := ticket.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
					t.Fatalf("dispatch error=%v", err)
				}
			}
			if h.a.currentActive() != 0 || len(h.a.tickets) != 0 || len(h.a.strict.markers) != 0 {
				t.Fatal("stale tickets retained capacity or dispatch markers")
			}
			first.Finish(strictNotDispatched)
			second.Finish(strictNotDispatched)
			next := mustStrictTicket(t, h.a)
			if err := next.BeforeStrictDispatch(); err != nil {
				t.Fatal(err)
			}
			// Replacement must leave already-dispatched ownership intact.
			if err := h.a.installStrictCapability(capability); err != nil {
				t.Fatal(err)
			}
			if err := next.BeforeStrictDispatch(); err != nil || h.a.currentActive() != 1 {
				t.Fatalf("dispatched ownership changed: %v", err)
			}
			next.Finish(strictQualifiedComplete)
		})
	}
}

func TestStrictPendingUnloadBlocksZeroCold(t *testing.T) {
	h := newStrictHarness(t)
	k := admKey{engine: "ollama", model: "m:latest"}
	h.a.mu.Lock()
	h.a.unloadLocked([]admKey{k}, h.a.policy)
	h.a.mu.Unlock()
	req := strictRequest()
	req.ColdStartBudgetMs = 0
	if ticket, err := h.a.AdmitStrict(context.Background(), "ollama", "m", req); ticket != nil || !errors.Is(err, errStrictCold) {
		t.Fatalf("ticket=%v error=%v", ticket, err)
	}
	// A still-loaded report does not acknowledge the outstanding unload.
	h.a.setResidency(map[string][]string{"ollama": {"m"}})
	if ticket, err := h.a.AdmitStrict(context.Background(), "ollama", "m", req); ticket != nil || !errors.Is(err, errStrictCold) {
		t.Fatalf("ticket=%v error=%v", ticket, err)
	}
}

func TestStrictWarmTicketRechecksResidency(t *testing.T) {
	for _, scenario := range []string{"residency lost", "unload pending", "warm with cold allowance"} {
		t.Run(scenario, func(t *testing.T) {
			h := newStrictHarness(t)
			req := strictRequest()
			if scenario != "warm with cold allowance" {
				req.ColdStartBudgetMs = 0
			}
			ticket, err := h.a.AdmitStrict(context.Background(), "ollama", "m", req)
			if err != nil {
				t.Fatal(err)
			}
			if scenario != "unload pending" {
				h.a.setResidency(map[string][]string{})
			} else {
				h.a.mu.Lock()
				h.a.unloadLocked([]admKey{{engine: "ollama", model: "m:latest"}}, h.a.policy)
				h.a.mu.Unlock()
			}
			if err := ticket.BeforeStrictDispatch(); !errors.Is(err, errStrictCold) {
				t.Fatalf("dispatch error=%v", err)
			}
			if h.a.currentActive() != 0 || len(h.a.strict.markers) != 0 {
				t.Fatal("rejected warm ticket retained capacity or dispatch marker")
			}
			ticket.Finish(strictNotDispatched)
			if h.a.currentActive() != 0 {
				t.Fatal("caller Finish released the slot twice")
			}
		})
	}
}

func TestStrictHeldModelExcludedFromEviction(t *testing.T) {
	for _, scenario := range []string{"admitted", "dispatched"} {
		t.Run(scenario, func(t *testing.T) {
			h := newStrictHarness(t)
			ticket := mustStrictTicket(t, h.a)
			if scenario == "dispatched" {
				if err := ticket.BeforeStrictDispatch(); err != nil {
					t.Fatal(err)
				}
			}
			defer ticket.Finish(strictQualifiedComplete)
			h.a.mu.Lock()
			p := h.a.policy
			p.Admission.MaxResidentModels = 1
			plan, evict := h.a.fitLocked(admKey{engine: "ollama", model: "other:latest"}, p)
			h.a.mu.Unlock()
			if plan != fitWait || len(evict) != 0 {
				t.Fatalf("plan=%v evictions=%v", plan, evict)
			}
		})
	}
}

func TestStrictColdDeadlineSurvivesCapacityWait(t *testing.T) {
	for _, scenario := range []string{"held through deadline", "released after deadline"} {
		t.Run(scenario, func(t *testing.T) {
			h := newStrictHarness(t)
			c := useStrictClock(h.a)
			h.a.setIntent(map[string]bool{"ollama": true})
			h.setHealthy("ollama", false)
			req := strictRequest()
			req.ColdStartBudgetMs = 100
			ch := asyncStrict(h.a, context.Background(), req)
			awaitStrictTimer(t, c)
			// Make a concrete owner win capacity before the cold waiter wakes.
			h.a.mu.Lock()
			c.Advance(50 * time.Millisecond)
			h.mu.Lock()
			h.healthy["ollama"] = true
			h.mu.Unlock()
			k := admKey{engine: "ollama", model: "m:latest"}
			h.a.inflight[k]++
			h.a.active++
			h.a.activeByEngine[k.engine]++
			h.a.signalLocked()
			h.a.mu.Unlock()
			if d := awaitStrictTimer(t, c); d != 50*time.Millisecond {
				t.Fatalf("remaining cold timer=%v", d)
			}
			h.a.mu.Lock()
			c.Advance(50 * time.Millisecond)
			if scenario == "released after deadline" {
				c.Advance(time.Millisecond)
				h.a.releaseStrictSlotLocked(k)
				h.a.signalLocked()
			}
			h.a.mu.Unlock()
			if r := awaitStrict(t, ch); r.ticket != nil || !errors.Is(r.err, errStrictCold) {
				t.Fatalf("result=%+v", r)
			}
			if scenario == "held through deadline" {
				h.a.mu.Lock()
				h.a.releaseStrictSlotLocked(k)
				h.a.mu.Unlock()
			}
		})
	}
}

func TestStrictTwoOriginsShareLastSlot(t *testing.T) {
	h := newStrictHarness(t)
	c := useStrictClock(h.a)
	start := make(chan struct{})
	results := make(chan strictResult, 2)
	for _, origin := range []string{"origin-a", "origin-b"} {
		go func(origin string) {
			req := strictRequest()
			req.Request.OriginNodeID = origin
			<-start
			ticket, err := h.a.AdmitStrict(context.Background(), "ollama", "m:latest", req)
			results <- strictResult{ticket, err}
		}(origin)
	}
	close(start)
	first := awaitStrict(t, results)
	if first.err != nil {
		t.Fatal(first.err)
	}
	awaitStrictTimer(t, c)
	h.a.mu.Lock()
	active, queued := h.a.active, h.a.queued
	h.a.mu.Unlock()
	if active != 1 || queued != 1 {
		t.Fatalf("active=%d queued=%d", active, queued)
	}
	first.ticket.Finish(strictNotDispatched)
	second := awaitStrict(t, results)
	if second.err != nil {
		t.Fatal(second.err)
	}
	second.ticket.Finish(strictNotDispatched)
	if h.a.currentActive() != 0 {
		t.Fatal("slot leaked")
	}
}

func TestStrictTerminalReleaseIsIdempotent(t *testing.T) {
	for _, e := range []strictTerminalEvidence{strictNotDispatched, strictQualifiedComplete, strictAuthoritativeIdle, strictVerifiedReplacement} {
		t.Run(string(e), func(t *testing.T) {
			h := newStrictHarness(t)
			ticket := mustStrictTicket(t, h.a)
			if e != strictNotDispatched {
				if err := ticket.BeforeStrictDispatch(); err != nil {
					t.Fatal(err)
				}
			}
			ticket.Finish(e)
			ticket.Finish(e)
			if h.a.currentActive() != 0 {
				t.Fatal("slot leaked")
			}
			next := mustStrictTicket(t, h.a)
			next.Finish(strictNotDispatched)
		})
	}
}

func TestStrictQueuedCancellationRemovesWaiter(t *testing.T) {
	h := newStrictHarness(t)
	c := useStrictClock(h.a)
	first := mustStrictTicket(t, h.a)
	defer first.Finish(strictNotDispatched)
	ctx, cancel := context.WithCancel(context.Background())
	ch := asyncStrict(h.a, ctx, strictRequest())
	awaitStrictTimer(t, c)
	cancel()
	if r := awaitStrict(t, ch); !errors.Is(r.err, context.Canceled) {
		t.Fatalf("error=%v", r.err)
	}
	h.a.mu.Lock()
	defer h.a.mu.Unlock()
	if h.a.queued != 0 || h.a.active != 1 {
		t.Fatalf("queued=%d active=%d", h.a.queued, h.a.active)
	}
}

func TestStrictOccupiedBudgetRejects(t *testing.T) {
	h := newStrictHarness(t)
	c := useStrictClock(h.a)
	first := mustStrictTicket(t, h.a)
	defer first.Finish(strictNotDispatched)
	req := strictRequest()
	req.QueueBudgetMs = 125
	ch := asyncStrict(h.a, context.Background(), req)
	if d := awaitStrictTimer(t, c); d != 125*time.Millisecond {
		t.Fatalf("timer=%v", d)
	}
	c.Advance(125 * time.Millisecond)
	if r := awaitStrict(t, ch); !errors.Is(r.err, errStrictOccupied) {
		t.Fatalf("error=%v", r.err)
	}
}

func TestStrictZeroColdHasNoWakeOrUnload(t *testing.T) {
	for _, scenario := range []string{"wake", "switch", "unloaded"} {
		t.Run(scenario, func(t *testing.T) {
			h := newStrictHarness(t)
			if scenario == "wake" {
				h.a.setIntent(map[string]bool{"ollama": true})
				h.setHealthy("ollama", false)
			}
			if scenario == "switch" {
				h.policy(func(p *nodepolicy.Policy) {
					p.Admission.MaxConcurrentPerModel = 1
					p.Admission.MaxResidentModels = 1
					p.Admission.SwitchModels = true
				})
				h.a.setResidency(map[string][]string{"ollama": {"other"}})
			}
			if scenario == "unloaded" {
				h.a.setResidency(map[string][]string{})
			}
			req := strictRequest()
			req.ColdStartBudgetMs = 0
			_, err := h.a.AdmitStrict(context.Background(), "ollama", "m", req)
			if !errors.Is(err, errStrictCold) {
				t.Fatalf("error=%v", err)
			}
			if len(h.notes(nodepolicy.NotifyAdmissionWake)) != 0 || len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 {
				t.Fatal("zero cold caused engine effects")
			}
		})
	}
}

func TestStrictSwitchTimeoutRejects(t *testing.T) {
	h := newStrictHarness(t)
	c := useStrictClock(h.a)
	h.policy(func(p *nodepolicy.Policy) {
		p.Admission.MaxConcurrentPerModel = 1
		p.Admission.MaxResidentModels = 1
		p.Admission.SwitchModels = true
	})
	h.a.setResidency(map[string][]string{"ollama": {"other"}})
	req := strictRequest()
	req.ColdStartBudgetMs = 150
	ch := asyncStrict(h.a, context.Background(), req)
	if d := awaitStrictTimer(t, c); d != 150*time.Millisecond {
		t.Fatalf("timer=%v", d)
	}
	c.Advance(150 * time.Millisecond)
	if r := awaitStrict(t, ch); !errors.Is(r.err, errStrictCold) {
		t.Fatalf("result=%+v", r)
	}
	if h.a.currentActive() != 0 {
		t.Fatal("timeout admitted")
	}
}

func TestStrictCapabilityRequiresFiniteCapacityAndIdentity(t *testing.T) {
	for _, limit := range []int{0, -1, 1048577} {
		t.Run(fmt.Sprint(limit), func(t *testing.T) {
			h := newStrictHarness(t)
			if err := h.a.installStrictCapability(nodepolicy.StrictAdmissionCapability{Version: 1, Models: []nodepolicy.StrictAdmissionModel{{Engine: "ollama", Model: "m", MaxConcurrent: limit, Identity: strictRequest().Identity}}}); err == nil {
				t.Fatal("invalid limit accepted")
			}
		})
	}
	t.Run("absent", func(t *testing.T) {
		h := newAdmHarness(t)
		_, err := h.a.AdmitStrict(context.Background(), "ollama", "m", strictRequest())
		if !errors.Is(err, errStrictUnavailable) {
			t.Fatal(err)
		}
	})
	t.Run("stale identity", func(t *testing.T) {
		h := newStrictHarness(t)
		req := strictRequest()
		req.Identity.EngineRunID = "stale"
		_, err := h.a.AdmitStrict(context.Background(), "ollama", "m", req)
		if !errors.Is(err, errStrictUnavailable) {
			t.Fatal(err)
		}
	})
}

func TestStrictQueueAndColdSharePreOutputCap(t *testing.T) {
	h := newStrictHarness(t)
	c := useStrictClock(h.a)
	first := mustStrictTicket(t, h.a)
	ch := asyncStrict(h.a, context.Background(), strictRequest())
	awaitStrictTimer(t, c)
	c.Advance(700 * time.Millisecond)
	h.a.setIntent(map[string]bool{"ollama": true})
	h.setHealthy("ollama", false)
	first.Finish(strictNotDispatched)
	// Wake has only the remainder of the original deadline, never a fresh
	// cold budget after the queue wait.
	for {
		if d := awaitStrictTimer(t, c); d <= 300*time.Millisecond {
			break
		}
	}
	c.Advance(300 * time.Millisecond)
	r := awaitStrict(t, ch)
	if r.ticket != nil || r.err == nil {
		t.Fatalf("result=%+v", r)
	}
}

func TestStrictStateChangeAtBudgetDoesNotAdmit(t *testing.T) {
	for _, scenario := range []string{"queue", "cold"} {
		t.Run(scenario, func(t *testing.T) {
			h := newStrictHarness(t)
			c := useStrictClock(h.a)
			var first strictAdmissionTicket
			if scenario == "queue" {
				first = mustStrictTicket(t, h.a)
			} else {
				h.a.setIntent(map[string]bool{"ollama": true})
				h.setHealthy("ollama", false)
			}
			req := strictRequest()
			req.QueueBudgetMs = 100
			req.ColdStartBudgetMs = 100
			ch := asyncStrict(h.a, context.Background(), req)
			awaitStrictTimer(t, c)
			// Hold the admission mutex while moving both time and state, so the
			// waiter observes eligibility and exhaustion in the same reevaluation.
			h.a.mu.Lock()
			c.Advance(100 * time.Millisecond)
			if scenario == "cold" {
				h.mu.Lock()
				h.healthy["ollama"] = true
				h.mu.Unlock()
			}
			h.a.signalLocked()
			h.a.mu.Unlock()
			if first != nil {
				first.Finish(strictNotDispatched)
			}
			if r := awaitStrict(t, ch); r.ticket != nil || r.err == nil {
				t.Fatalf("result=%+v", r)
			}
		})
	}
}

func TestStrictWarmAfterWakeKeepsColdDeadline(t *testing.T) {
	h := newStrictHarness(t)
	c := useStrictClock(h.a)
	h.a.setIntent(map[string]bool{"ollama": true})
	h.setHealthy("ollama", false)
	req := strictRequest()
	req.ColdStartBudgetMs = 200
	ch := asyncStrict(h.a, context.Background(), req)
	awaitStrictTimer(t, c)
	c.Advance(150 * time.Millisecond)
	h.setHealthy("ollama", true)
	r := awaitStrict(t, ch)
	if r.err != nil {
		t.Fatal(r.err)
	}
	c.Advance(50 * time.Millisecond)
	if err := r.ticket.BeforeStrictDispatch(); !errors.Is(err, errStrictCold) {
		t.Fatalf("error=%v", err)
	}
	r.ticket.Finish(strictNotDispatched)
}

func TestStrictAvailabilityAndOffAreTyped(t *testing.T) {
	for _, state := range []string{"paused", "draining", "off"} {
		t.Run(state, func(t *testing.T) {
			h := newStrictHarness(t)
			switch state {
			case "paused":
				h.a.setAvailability(nodepolicy.Paused, false)
			case "draining":
				h.a.setDrain("ollama", true)
			case "off":
				h.a.setIntent(map[string]bool{"ollama": false})
			}
			_, err := h.a.AdmitStrict(context.Background(), "ollama", "m", strictRequest())
			var typed *strictAdmissionError
			if !errors.Is(err, errStrictUnavailable) || !errors.As(err, &typed) {
				t.Fatalf("error=%v", err)
			}
		})
	}
}

func TestStrictInFlightDoesNotProveWarmResidency(t *testing.T) {
	h := newStrictHarness(t)
	p := h.a.currentPolicy()
	p.Admission.MaxConcurrentPerModel = 2
	if err := h.a.activatePolicy(p); err != nil {
		t.Fatal(err)
	}
	if err := h.a.installStrictCapability(nodepolicy.StrictAdmissionCapability{Version: 1, Models: []nodepolicy.StrictAdmissionModel{{Engine: "ollama", Model: "m", MaxConcurrent: 2, Identity: strictRequest().Identity}}}); err != nil {
		t.Fatal(err)
	}
	h.a.setResidency(map[string][]string{})
	first := mustStrictTicket(t, h.a)
	defer first.Finish(strictNotDispatched)
	req := strictRequest()
	req.ColdStartBudgetMs = 0
	if _, err := h.a.AdmitStrict(context.Background(), "ollama", "m", req); !errors.Is(err, errStrictCold) {
		t.Fatalf("error=%v", err)
	}
}

func TestStrictPublishesActivityAndRelease(t *testing.T) {
	h := newStrictHarness(t)
	ticket := mustStrictTicket(t, h.a)
	notes := h.notes(nodepolicy.NotifyAdmissionState)
	if len(notes) == 0 {
		t.Fatal("no active state published")
	}
	active, ok := notes[len(notes)-1].params.(nodepolicy.AdmissionState)
	if !ok || active.ActiveByEngine["ollama"] != 1 {
		t.Fatalf("state=%+v", notes[len(notes)-1])
	}
	ticket.Finish(strictNotDispatched)
	notes = h.notes(nodepolicy.NotifyAdmissionState)
	idle, ok := notes[len(notes)-1].params.(nodepolicy.AdmissionState)
	if !ok || idle.Active != 0 || idle.ActiveByEngine["ollama"] != 0 || idle.LastActivityMs["ollama"] == 0 {
		t.Fatalf("state=%+v", notes[len(notes)-1])
	}
}

// setPolicy installs p without activatePolicy's revision order check, so a
// test can give the controller any policy, including one with no revision.
func (a *admissionController) setPolicy(p nodepolicy.Policy) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.policy = p
	a.revalidateStrictCapabilitiesLocked()
	a.signalLocked()
}

// admHarness is a controller with recorded notifications and settable engine
// health.
type admHarness struct {
	a *admissionController

	mu      sync.Mutex
	sent    []notification
	healthy map[string]bool
}

func newAdmHarness(t *testing.T) *admHarness {
	t.Helper()
	h := &admHarness{healthy: map[string]bool{"ollama": true, "lmstudio": true, "llamacpp": true}}
	h.a = newAdmissionController(
		func(method string, params any) {
			h.mu.Lock()
			defer h.mu.Unlock()
			h.sent = append(h.sent, notification{method, params})
		},
		func(engine string) bool {
			h.mu.Lock()
			defer h.mu.Unlock()
			return h.healthy[engine]
		},
		normalizeEngineModel,
	)
	return h
}

func (h *admHarness) setHealthy(engine string, ok bool) {
	h.mu.Lock()
	h.healthy[engine] = ok
	h.mu.Unlock()
	h.a.backendChanged()
}

func (h *admHarness) notes(method string) []notification {
	h.mu.Lock()
	defer h.mu.Unlock()
	var out []notification
	for _, n := range h.sent {
		if n.method == method {
			out = append(out, n)
		}
	}
	return out
}

func (h *admHarness) policy(mutate func(*nodepolicy.Policy)) {
	p := nodepolicy.Default()
	mutate(&p)
	if err := p.Validate(); err != nil {
		panic(err)
	}
	h.a.setPolicy(p)
}

func (h *admHarness) admit(engine, model string, waitCap time.Duration) admissionDecision {
	return h.a.admit(context.Background(), admissionRequest{engine: engine, model: model, waitCap: waitCap})
}

// admitAsync starts an admission and returns its eventual decision.
func (h *admHarness) admitAsync(engine, model string, waitCap time.Duration) <-chan admissionDecision {
	ch := make(chan admissionDecision, 1)
	go func() { ch <- h.admit(engine, model, waitCap) }()
	return ch
}

func mustAdmit(t *testing.T, d admissionDecision) *admissionTicket {
	t.Helper()
	if d.ticket == nil {
		t.Fatalf("not admitted: reject=%q err=%v", d.reject, d.err)
	}
	return d.ticket
}

func mustReject(t *testing.T, d admissionDecision, want nodepolicy.RejectReason) {
	t.Helper()
	if d.ticket != nil {
		d.ticket.release()
		t.Fatalf("admitted, want rejection %q", want)
	}
	if d.reject != want {
		t.Fatalf("reject = %q (err %v), want %q", d.reject, d.err, want)
	}
}

func awaitDecision(t *testing.T, ch <-chan admissionDecision, within time.Duration) admissionDecision {
	t.Helper()
	select {
	case d := <-ch:
		return d
	case <-time.After(within):
		t.Fatalf("admission did not decide within %v", within)
		return admissionDecision{}
	}
}

func assertPending(t *testing.T, ch <-chan admissionDecision) {
	t.Helper()
	select {
	case d := <-ch:
		t.Fatalf("admission decided while it should still wait: ticket=%v reject=%q", d.ticket != nil, d.reject)
	case <-time.After(50 * time.Millisecond):
	}
}

func TestAdmission_DefaultPolicyAdmits(t *testing.T) {
	h := newAdmHarness(t)
	tk := mustAdmit(t, h.admit("ollama", "qwen3", -1))
	tk.release()
	tk.release() // idempotent
	if h.a.active != 0 {
		t.Fatalf("active = %d after release, want 0", h.a.active)
	}
}

func TestAdmission_PausedAndDrainingReject(t *testing.T) {
	h := newAdmHarness(t)
	h.a.setAvailability(nodepolicy.Paused, false)
	mustReject(t, h.admit("ollama", "qwen3", -1), nodepolicy.RejectPaused)

	h.a.setAvailability(nodepolicy.Draining, false)
	mustReject(t, h.admit("ollama", "qwen3", -1), nodepolicy.RejectDraining)

	h.a.setAvailability(nodepolicy.Available, false)
	h.a.setDrain("ollama", true)
	mustReject(t, h.admit("ollama", "qwen3", -1), nodepolicy.RejectDraining)
	// A settings-restart drain is per engine.
	mustAdmit(t, h.admit("lmstudio", "qwen3-8b", -1)).release()
	h.a.setDrain("ollama", false)
	mustAdmit(t, h.admit("ollama", "qwen3", -1)).release()
}

// Draining refuses new work but lets admitted work finish, and reports the
// moment the node has drained.
func TestAdmission_DrainingFinishesActive(t *testing.T) {
	h := newAdmHarness(t)
	active := mustAdmit(t, h.admit("ollama", "qwen3", -1))
	h.a.setAvailability(nodepolicy.Draining, false)
	mustReject(t, h.admit("ollama", "qwen3", -1), nodepolicy.RejectDraining)
	if h.a.active != 1 {
		t.Fatalf("draining disturbed the active request: active = %d", h.a.active)
	}
	active.release()
	states := h.notes(nodepolicy.NotifyAdmissionState)
	if len(states) == 0 {
		t.Fatal("no admission/state sent")
	}
	last, _ := states[len(states)-1].params.(nodepolicy.AdmissionState)
	if last.Active != 0 {
		t.Fatalf("last admission/state active = %d, want an immediate 0 once drained", last.Active)
	}
	if last.LastActivityMs["ollama"] == 0 {
		t.Fatal("admission/state carries no lastActivityMs for the engine")
	}
}

func TestAdmission_CancelActiveCancelsAdmitted(t *testing.T) {
	h := newAdmHarness(t)
	var cancelled atomic.Int32
	d := h.a.admit(context.Background(), admissionRequest{
		engine: "ollama", model: "qwen3", waitCap: -1,
		cancel: func() { cancelled.Add(1) },
	})
	tk := mustAdmit(t, d)
	defer tk.release()

	h.a.setAvailability(nodepolicy.Draining, false)
	if cancelled.Load() != 0 {
		t.Fatal("draining without cancelActive cancelled running work")
	}
	h.a.setAvailability(nodepolicy.Draining, true)
	if cancelled.Load() != 1 {
		t.Fatalf("cancelActive cancelled %d executions, want 1", cancelled.Load())
	}
}

func TestAdmission_EngineOffRejectsWithoutWake(t *testing.T) {
	h := newAdmHarness(t)
	h.setHealthy("ollama", false)

	h.a.setIntent(map[string]bool{"ollama": false})
	mustReject(t, h.admit("ollama", "qwen3", -1), nodepolicy.RejectOff)

	// No saved intent at all: nothing would start it either.
	h.a.setIntent(map[string]bool{})
	mustReject(t, h.admit("ollama", "qwen3", -1), nodepolicy.RejectOff)

	// Saved On but start-on-demand off.
	h.a.setIntent(map[string]bool{"ollama": true})
	h.policy(func(p *nodepolicy.Policy) { p.Idle.StartOnDemand = false })
	mustReject(t, h.admit("ollama", "qwen3", -1), nodepolicy.RejectOff)

	if got := h.notes(nodepolicy.NotifyAdmissionWake); len(got) != 0 {
		t.Fatalf("sent %d admission/wake for an engine that must not be started", len(got))
	}
}

func TestAdmission_WakeWaitsForHealthy(t *testing.T) {
	h := newAdmHarness(t)
	h.setHealthy("ollama", false)
	h.a.setIntent(map[string]bool{"ollama": true})

	ch := h.admitAsync("ollama", "qwen3", -1)
	waitForCond(t, 2*time.Second, "admission/wake", func() bool {
		return len(h.notes(nodepolicy.NotifyAdmissionWake)) == 1
	})
	wake, _ := h.notes(nodepolicy.NotifyAdmissionWake)[0].params.(nodepolicy.WakeRequest)
	if wake.Engine != "ollama" {
		t.Fatalf("wake names engine %q, want ollama", wake.Engine)
	}
	assertPending(t, ch)

	h.setHealthy("ollama", true)
	mustAdmit(t, awaitDecision(t, ch, 2*time.Second)).release()
}

// The wake wait is bounded too, and a router that will not wait at all still
// starts the engine for next time.
func TestAdmission_WakeTimeout(t *testing.T) {
	h := newAdmHarness(t)
	h.setHealthy("ollama", false)
	h.a.setIntent(map[string]bool{"ollama": true})
	started := time.Now()
	mustReject(t, h.admit("ollama", "qwen3", 0), nodepolicy.RejectWake)
	if time.Since(started) > time.Second {
		t.Fatalf("a zero wait took %v", time.Since(started))
	}
	if len(h.notes(nodepolicy.NotifyAdmissionWake)) != 1 {
		t.Fatal("a zero-wait request still has to ask for the engine to start")
	}
	// Pausing mid-wake ends the wait as paused.
	ch := h.admitAsync("ollama", "qwen3", -1)
	assertPending(t, ch)
	h.a.setAvailability(nodepolicy.Paused, false)
	mustReject(t, awaitDecision(t, ch, 2*time.Second), nodepolicy.RejectPaused)
}

func TestAdmission_ConcurrencyQueuesThenBusy(t *testing.T) {
	h := newAdmHarness(t)
	h.policy(func(p *nodepolicy.Policy) {
		p.Admission.MaxConcurrentPerModel = 1
		p.Admission.MaxResidentModels = 0
	})
	first := mustAdmit(t, h.admit("ollama", "qwen3", -1))

	queued := h.admitAsync("ollama", "qwen3", -1)
	assertPending(t, queued)
	waitForCond(t, time.Second, "queued count", func() bool {
		h.a.mu.Lock()
		defer h.a.mu.Unlock()
		return h.a.queued == 1
	})
	first.release()
	second := mustAdmit(t, awaitDecision(t, queued, 2*time.Second))

	// Another model is unaffected by this model's limit.
	mustAdmit(t, h.admit("ollama", "llama3", -1)).release()

	// The queue wait is bounded; past it the request is handed back as busy.
	started := time.Now()
	mustReject(t, h.admit("ollama", "qwen3", 100*time.Millisecond), nodepolicy.RejectBusy)
	if waited := time.Since(started); waited < 80*time.Millisecond {
		t.Fatalf("busy after %v, want it to have queued first", waited)
	}
	second.release()
}

// A profile's maxConcurrent overrides the node default, and finds the model
// under the engine's own naming.
func TestAdmission_ProfileConcurrencyUsesNormalizedModel(t *testing.T) {
	h := newAdmHarness(t)
	h.policy(func(p *nodepolicy.Policy) {
		p.Admission.MaxResidentModels = 0
		p.Profiles = []nodepolicy.Profile{{Name: "q", Engine: "ollama", Model: "qwen3", MaxConcurrent: 1}}
	})
	d := h.admit("ollama", "qwen3:latest", -1)
	tk := mustAdmit(t, d)
	if !d.hasProfile || d.profile.Name != "q" {
		t.Fatalf("profile = %+v (has=%v), want q", d.profile, d.hasProfile)
	}
	mustReject(t, h.admit("ollama", "qwen3", 0), nodepolicy.RejectBusy)
	tk.release()
}

func TestAdmission_WaitHeaderZeroRejectsImmediately(t *testing.T) {
	h := newAdmHarness(t)
	h.policy(func(p *nodepolicy.Policy) { p.Admission.MaxConcurrentPerModel = 1 })
	tk := mustAdmit(t, h.admit("ollama", "qwen3", -1))
	defer tk.release()
	started := time.Now()
	mustReject(t, h.admit("ollama", "qwen3", 0), nodepolicy.RejectBusy)
	if waited := time.Since(started); waited > 50*time.Millisecond {
		t.Fatalf("a zero wait queued for %v", waited)
	}
}

func TestAdmission_ResidentSwitchUnloadsThenAdmits(t *testing.T) {
	h := newAdmHarness(t)
	h.policy(func(p *nodepolicy.Policy) {
		p.Admission.MaxResidentModels = 1
		p.Admission.SwitchModels = true
		p.Admission.SwitchTimeoutSeconds = 30
	})
	h.a.setResidency(map[string][]string{"lmstudio": {"old-model"}})

	ch := h.admitAsync("ollama", "qwen3", -1)
	waitForCond(t, 2*time.Second, "admission/unload", func() bool {
		return len(h.notes(nodepolicy.NotifyAdmissionUnload)) == 1
	})
	unload, _ := h.notes(nodepolicy.NotifyAdmissionUnload)[0].params.(nodepolicy.UnloadRequest)
	if unload.Engine != "lmstudio" || len(unload.Models) != 1 || unload.Models[0] != "old-model" {
		t.Fatalf("unload = %+v, want lmstudio [old-model] (residency is across engines)", unload)
	}
	assertPending(t, ch)

	// The broker unloads and reports the new residency.
	h.a.setResidency(map[string][]string{"lmstudio": {}})
	tk := mustAdmit(t, awaitDecision(t, ch, 2*time.Second))
	tk.release()
}

func TestAdmission_SwitchTimeoutAdmits(t *testing.T) {
	h := newAdmHarness(t)
	h.policy(func(p *nodepolicy.Policy) {
		p.Admission.MaxResidentModels = 1
		p.Admission.SwitchTimeoutSeconds = 1
	})
	h.a.setResidency(map[string][]string{"ollama": {"llama3:latest"}})
	started := time.Now()
	// Even a zero queue wait does not cut an eviction short: the switch is
	// bounded by its own timeout and ends in admission.
	tk := mustAdmit(t, h.admit("ollama", "qwen3", 0))
	defer tk.release()
	if waited := time.Since(started); waited < 900*time.Millisecond {
		t.Fatalf("admitted after %v, want the switch timeout to elapse first", waited)
	}
	if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 1 {
		t.Fatal("no admission/unload sent before admitting")
	}
}

func TestAdmission_NoSwitchWaitsThenNoFit(t *testing.T) {
	h := newAdmHarness(t)
	h.policy(func(p *nodepolicy.Policy) {
		p.Admission.MaxResidentModels = 1
		p.Admission.SwitchModels = false
	})
	h.a.setResidency(map[string][]string{"ollama": {"llama3"}})
	mustReject(t, h.admit("ollama", "qwen3", 100*time.Millisecond), nodepolicy.RejectNoFit)
	if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 {
		t.Fatal("unloaded a model with switchModels off")
	}
	// The resident model itself is always admitted.
	mustAdmit(t, h.admit("ollama", "llama3:latest", 0)).release()
}

// A model with work in flight is resident and never evicted; one that just
// finished is presumed loaded until the next residency report.
func TestAdmission_InFlightAndRecentModelsCountAsResident(t *testing.T) {
	h := newAdmHarness(t)
	h.policy(func(p *nodepolicy.Policy) {
		p.Admission.MaxResidentModels = 1
		p.Admission.SwitchTimeoutSeconds = 30
	})
	h.a.setResidency(map[string][]string{"ollama": {}})
	a := mustAdmit(t, h.admit("ollama", "llama3", -1))
	mustReject(t, h.admit("ollama", "qwen3", 100*time.Millisecond), nodepolicy.RejectNoFit)
	if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 {
		t.Fatal("asked to unload a model with work in flight")
	}
	a.release()

	ch := h.admitAsync("ollama", "qwen3", -1)
	waitForCond(t, 2*time.Second, "unload of the just-finished model", func() bool {
		return len(h.notes(nodepolicy.NotifyAdmissionUnload)) == 1
	})
	h.a.setResidency(map[string][]string{"ollama": {}})
	mustAdmit(t, awaitDecision(t, ch, 2*time.Second)).release()
}

// Before the broker has ever reported residency only in-flight work counts, so
// nothing waits on an eviction nobody can confirm.
func TestAdmission_UnknownResidencyCountsOnlyInFlight(t *testing.T) {
	h := newAdmHarness(t)
	a := mustAdmit(t, h.admit("ollama", "llama3", -1))
	a.release()
	mustAdmit(t, h.admit("ollama", "qwen3", 0)).release()
	if len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 {
		t.Fatal("evicted on residency the broker never reported")
	}
}

func TestAdmission_MemoryBudget(t *testing.T) {
	const gb = int64(1) << 30
	h := newAdmHarness(t)
	h.policy(func(p *nodepolicy.Policy) {
		p.Admission.MaxResidentModels = 0
		p.Admission.MemoryBudgetBytes = 10 * gb
		p.Admission.SwitchTimeoutSeconds = 30
		p.Profiles = []nodepolicy.Profile{
			{Name: "a", Engine: "ollama", Model: "a:1", MemoryBytes: 6 * gb},
			{Name: "b", Engine: "ollama", Model: "b:1", MemoryBytes: 6 * gb},
			{Name: "c", Engine: "ollama", Model: "c:1", MemoryBytes: 3 * gb},
			{Name: "huge", Engine: "ollama", Model: "huge:1", MemoryBytes: 11 * gb},
		}
	})
	h.a.setResidency(map[string][]string{"ollama": {"a:1"}})

	// 6 + 3 fits.
	c := mustAdmit(t, h.admit("ollama", "c:1", 0))
	// 6 + 3 + 6 does not; unloading idle a makes room, c is busy and stays.
	ch := h.admitAsync("ollama", "b:1", -1)
	waitForCond(t, 2*time.Second, "unload of a", func() bool {
		return len(h.notes(nodepolicy.NotifyAdmissionUnload)) == 1
	})
	unload, _ := h.notes(nodepolicy.NotifyAdmissionUnload)[0].params.(nodepolicy.UnloadRequest)
	if len(unload.Models) != 1 || unload.Models[0] != "a:1" {
		t.Fatalf("unload = %+v, want only the idle a:1", unload)
	}
	h.a.setResidency(map[string][]string{"ollama": {"c:1"}})
	b := mustAdmit(t, awaitDecision(t, ch, 2*time.Second))

	// A model larger than the whole budget never fits.
	mustReject(t, h.admit("ollama", "huge:1", -1), nodepolicy.RejectNoFit)
	// Unknown memory counts as zero.
	mustAdmit(t, h.admit("ollama", "unprofiled", 0)).release()
	b.release()
	c.release()
}

func TestAdmission_ContextCancelEndsWait(t *testing.T) {
	h := newAdmHarness(t)
	h.policy(func(p *nodepolicy.Policy) { p.Admission.MaxConcurrentPerModel = 1 })
	tk := mustAdmit(t, h.admit("ollama", "qwen3", -1))
	defer tk.release()
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan admissionDecision, 1)
	go func() { ch <- h.a.admit(ctx, admissionRequest{engine: "ollama", model: "qwen3", waitCap: -1}) }()
	assertPending(t, ch)
	cancel()
	d := awaitDecision(t, ch, time.Second)
	if d.err == nil || d.ticket != nil {
		t.Fatalf("cancelled wait = %+v, want a context error", d)
	}
	h.a.mu.Lock()
	queued := h.a.queued
	h.a.mu.Unlock()
	if queued != 0 {
		t.Fatalf("queued = %d after an abandoned wait, want 0", queued)
	}
}

func TestParseAdmissionWait(t *testing.T) {
	for _, c := range []struct {
		raw  string
		want time.Duration
	}{
		{"", -1}, {"0", 0}, {"7", 7 * time.Second}, {"-1", -1}, {"soon", -1},
	} {
		h := http.Header{}
		if c.raw != "" {
			h.Set(nodepolicy.AdmissionWaitHeader, c.raw)
		}
		if got := parseAdmissionWait(h); got != c.want {
			t.Errorf("parseAdmissionWait(%q) = %v, want %v", c.raw, got, c.want)
		}
	}
}
