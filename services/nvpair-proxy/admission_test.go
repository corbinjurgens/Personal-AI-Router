// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Unit coverage for the node-wide admission controller, driven directly so
// each rule in FORK_DESIGN.md §3.3 is pinned without HTTP in the way. The
// ingress and self-dispatch tests in admission_http_test.go prove the same
// controller is what both entry points consult.

import (
	"context"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

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
