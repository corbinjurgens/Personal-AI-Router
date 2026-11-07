// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

type diagnosticStatusReply struct {
	Run         modelconfig.DiagnosticRun       `json:"run"`
	State       string                          `json:"state"`
	Token       string                          `json:"token,omitempty"`
	WorkStopped bool                            `json:"workStopped"`
	Markers     map[string]time.Time            `json:"markers,omitempty"`
	Identity    *modelconfig.DiagnosticIdentity `json:"identity,omitempty"`
	Low         *modelconfig.MeasuredPhase      `json:"low,omitempty"`
	High        *modelconfig.MeasuredPhase      `json:"high,omitempty"`
}

func diagnosticIdentityFixture() modelconfig.DiagnosticIdentity {
	return modelconfig.DiagnosticIdentity{ConfigurationKey: strings.Repeat("a", 64), NativeBuildID: "sha256:" + strings.Repeat("b", 64), BuildInfo: "b11146", CPUModel: "fixture CPU", TotalRAMBytes: "68719476736",
		GPUs:      []modelconfig.DiagnosticGPU{{UUID: "gpu-a", Name: "GPU A", TotalMemoryBytes: "17179869184"}, {UUID: "gpu-b", Name: "GPU B", TotalMemoryBytes: "17179869184"}},
		Artifacts: []modelconfig.ArtifactSummary{{Role: "model", Identity: "fixture artifact", Size: "100", Assurance: "sampled"}}, PerSlotContext: 1024, Slots: 1,
		Placement: []modelconfig.DiagnosticPlacement{{DomainID: "ram", Kind: "host"}, {DomainID: "gpu-a", Kind: "device", DeviceID: "gpu-a"}, {DomainID: "gpu-b", Kind: "device", DeviceID: "gpu-b"}}}
}

func diagnosticBrokerFixture(t *testing.T) (*Broker, *fakeWorker, *brokerDiagnostic) {
	t.Helper()
	b, _ := newPolicyBroker(t)
	peer, fake := newFakeWorkerPipe(t)
	worker := &rpcWorker{peer: peer, done: make(chan struct{})}
	b.setEngineMgr(worker)
	_, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	op := &brokerDiagnostic{runID: "diagnostic-run", worker: worker, cancel: cancel, done: make(chan struct{}), wake: make(chan struct{}, 1), token: "reservation", reserveAttempted: true, runAttempted: true, accepted: true}
	op.run = modelconfig.DiagnosticRun{RunID: op.runID, Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture", ConfigurationKey: strings.Repeat("a", 64), Phase: "high", Outcome: "running", StartedAt: "2026-11-07T00:00:00Z"}
	op.review.preview = modelconfig.DiagnosticPreview{DiagnosticIdentity: diagnosticIdentityFixture(), DiagnosticPreviewRequest: modelconfig.DiagnosticPreviewRequest{Engine: op.run.Engine, ModelID: op.run.ModelID, ProfileKey: op.run.ProfileKey}}
	close(op.done)
	lease, err := b.reserveBuildApplySettings()
	if err != nil {
		t.Fatalf("reserve settings lease: %v", err)
	}
	op.lease = lease
	t.Cleanup(lease.Release)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.drains["llamacpp"] = true
	r.mu.Unlock()
	op.drain = &nodeDrainLease{broker: b, added: []string{"llamacpp"}}
	b.diagnostics.active = op
	return b, fake, op
}

func requireDiagnosticHeld(t *testing.T, b *Broker, op *brokerDiagnostic) {
	t.Helper()
	b.diagnostics.mu.Lock()
	active := b.diagnostics.active == op
	b.diagnostics.mu.Unlock()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	drained := r.drains["llamacpp"]
	r.mu.Unlock()
	b.engineConfigMu.Lock()
	leases := len(b.buildApplyMaintenance.settings)
	b.engineConfigMu.Unlock()
	if !active || !drained || leases != 1 {
		t.Fatalf("diagnostic hold: active=%v drained=%v leases=%d", active, drained, leases)
	}
}

func requireDiagnosticReleased(t *testing.T, b *Broker) {
	t.Helper()
	b.diagnostics.mu.Lock()
	active := b.diagnostics.active
	b.diagnostics.mu.Unlock()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	drained := r.drains["llamacpp"]
	r.mu.Unlock()
	b.engineConfigMu.Lock()
	leases := len(b.buildApplyMaintenance.settings)
	b.engineConfigMu.Unlock()
	if active != nil || drained || leases != 0 {
		t.Fatalf("diagnostic leaked resources: active=%v drained=%v leases=%d", active != nil, drained, leases)
	}
}

func TestModelDiagnosticSavedPauseGate(t *testing.T) {
	test := func(name string, mode nodepolicy.PauseMode, wantError bool) {
		t.Run(name, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			r := b.nodePolicyRuntime()
			r.mu.Lock()
			r.current.PauseMode = mode
			r.current.Availability = nodepolicy.Paused
			r.mu.Unlock()
			_, _, err := b.diagnosticGate()
			if (err != nil) != wantError {
				t.Fatalf("gate error=%v, wantError=%v", err, wantError)
			}
			r.mu.Lock()
			got := r.current.PauseMode
			r.mu.Unlock()
			if got != mode {
				t.Fatalf("saved pause changed to %s", got)
			}
		})
	}
	test("configured pause permits consent", nodepolicy.PauseConfigured, false)
	test("Halt refuses", nodepolicy.PauseImmediate, true)
}

func TestModelDiagnosticStartWithoutConsentHasNoEffects(t *testing.T) {
	b, _ := newPolicyBroker(t)
	peer, fake := newFakeWorkerPipe(t)
	b.setEngineMgr(&rpcWorker{peer: peer})
	_, err := b.startDiagnostic(context.Background(), modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: modelconfig.DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture"}, PreviewID: "preview"})
	if err == nil {
		t.Fatal("start without consent accepted")
	}
	if calls := fake.snapshot(); len(calls) != 0 {
		t.Fatalf("unconsented start made worker calls: %+v", calls)
	}
	if len(b.buildApplyMaintenance.settings) != 0 {
		t.Fatal("unconsented start acquired lease")
	}
}

func TestModelDiagnosticStalePreviewHasNoEffects(t *testing.T) {
	b, _ := newPolicyBroker(t)
	peer, fake := newFakeWorkerPipe(t)
	b.setEngineMgr(&rpcWorker{peer: peer})
	_, err := b.startDiagnostic(context.Background(), modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: modelconfig.DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture"}, PreviewID: "missing", Consent: true})
	if err == nil {
		t.Fatal("unknown preview accepted")
	}
	if calls := fake.snapshot(); len(calls) != 0 {
		t.Fatalf("stale start made worker calls: %+v", calls)
	}
	if len(b.buildApplyMaintenance.settings) != 0 {
		t.Fatal("stale start acquired lease")
	}
}

func TestModelDiagnosticCancelWaitsForJoin(t *testing.T) {
	b, fake, op := diagnosticBrokerFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	run := op.run.Clone()
	setDiagnosticContractAnswer(t, fake, op, func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-run-status":
			return diagnosticStatusReply{Run: run, State: "running"}, ""
		case "engine:diagnostic-abort":
			var p diagnosticAuthority
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.RunID != op.runID || p.Token != "reservation" {
				return nil, "wrong authority"
			}
			close(entered)
			<-release
			terminal := run.Clone()
			terminal.Outcome = "cancelled"
			terminal.Phase = "done"
			return diagnosticStatusReply{Run: terminal, State: "terminal", WorkStopped: true}, ""
		}
		return nil, "unexpected method"
	})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := b.cancelDiagnostic(ctx, op.runID); done <- err }()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal("abort not called")
	}
	requireDiagnosticHeld(t, b, op)
	unblock()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal("cancel did not finish after join")
	}
	requireDiagnosticReleased(t, b)
}

func TestModelDiagnosticCleanupFailureRetainsHoldForRetry(t *testing.T) {
	b, fake, op := diagnosticBrokerFixture(t)
	run := op.run.Clone()
	aborts := 0
	setDiagnosticContractAnswer(t, fake, op, func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:diagnostic-run-status" {
			return diagnosticStatusReply{Run: run, State: "running"}, ""
		}
		if method == "engine:diagnostic-abort" {
			aborts++
			terminal := run.Clone()
			terminal.Phase = "done"
			if aborts == 1 {
				terminal.Outcome = "cleanup-failed"
				return diagnosticStatusReply{Run: terminal, State: "cleanup-failed"}, ""
			}
			terminal.Outcome = "cancelled"
			return diagnosticStatusReply{Run: terminal, State: "terminal", WorkStopped: true}, ""
		}
		return nil, "unexpected method"
	})
	if _, err := b.cancelDiagnostic(context.Background(), op.runID); err == nil {
		t.Fatal("unconfirmed cleanup accepted")
	}
	requireDiagnosticHeld(t, b, op)
	if _, err := b.cancelDiagnostic(context.Background(), op.runID); err != nil {
		t.Fatalf("cleanup retry: %v", err)
	}
	requireDiagnosticReleased(t, b)
}

func TestModelDiagnosticLostReserveReplyReconcilesBeforeRelease(t *testing.T) {
	b, fake, op := diagnosticBrokerFixture(t)
	op.token = ""
	op.accepted, op.runAttempted = false, false
	op.reserveUncertain = true
	op.reserveRequest = diagnosticReserveRequest{RunID: op.runID, Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture", ConfigurationKey: strings.Repeat("a", 64), EffectsFence: "effects"}
	run := op.run.Clone()
	fake.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-run-status":
			return diagnosticStatusReply{Run: run, State: "reserved", WorkStopped: true}, ""
		case "engine:diagnostic-reserve":
			var p diagnosticReserveRequest
			if err := json.Unmarshal(raw, &p); err != nil {
				return nil, err.Error()
			}
			if p.RunID != op.runID || p.EffectsFence != "effects" {
				return nil, "different reservation retry"
			}
			return map[string]string{"runId": op.runID, "token": "reservation", "state": "reserved"}, ""
		case "engine:diagnostic-release":
			return nil, ""
		}
		return nil, "unexpected method"
	})
	if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
	want := []string{"engine:diagnostic-run-status", "engine:diagnostic-reserve", "engine:diagnostic-run-status", "engine:diagnostic-release"}
	if got := fake.methods("engine:"); !reflect.DeepEqual(got, want) {
		t.Fatalf("reconciliation calls=%v, want %v", got, want)
	}
}

func TestModelDiagnosticStatusReturnsActiveRun(t *testing.T) {
	b, _, op := diagnosticBrokerFixture(t)
	run, err := b.diagnosticStatus("")
	if err != nil {
		t.Fatal(err)
	}
	if run == nil || run.RunID != op.runID {
		t.Fatalf("active status=%+v", run)
	}
	run.Reason = "client mutation"
	if op.run.Reason != "" {
		t.Fatal("status exposed mutable operation")
	}
}

func TestModelDiagnosticAdmissionRequiresExplicitIdleEvidence(t *testing.T) {
	test := func(name, raw string, wantError, wantIdle bool) {
		t.Run(name, func(t *testing.T) {
			idle, err := nodeDrainSnapshotIdle(json.RawMessage(raw), "proxy-run")
			if (err != nil) != wantError || idle != wantIdle {
				t.Fatalf("idle=%v error=%v", idle, err)
			}
		})
	}
	test("complete zero snapshot", `{"runId":"proxy-run","active":0,"queued":0,"activeByEngine":{"llamacpp":0}}`, false, true)
	test("missing queued is unknown", `{"runId":"proxy-run","active":0,"activeByEngine":{}}`, true, false)
	test("queued request is busy", `{"runId":"proxy-run","active":0,"queued":1,"activeByEngine":{}}`, false, false)
	test("null engine count is unknown", `{"runId":"proxy-run","active":0,"queued":0,"activeByEngine":{"llamacpp":null}}`, true, false)
	test("replacement proxy is unknown", `{"runId":"replacement","active":0,"queued":0,"activeByEngine":{}}`, true, false)
}

func diagnosticMemoryFixture(at time.Time, available uint64) *noderec.MemoryObservation {
	fact := func(bytes uint64) noderec.MemoryFact {
		return noderec.MemoryFact{Bytes: strconv.FormatUint(bytes, 10), Source: "fixture", SampledAt: at.UTC(), Valid: true}
	}
	return &noderec.MemoryObservation{RunID: "memory-run", Domains: []noderec.MemoryDomain{{ID: "ram", Kind: "system", Total: fact(10000), Free: fact(available)}}}
}

func TestModelDiagnosticSamplesRequireNewFreshIdentities(t *testing.T) {
	at := time.Date(2026, 11, 7, 0, 0, 0, 0, time.UTC)
	var samples diagnosticSamples
	observation := diagnosticMemoryFixture(at, 9000)
	if err := samples.observe(observation, at, nil); err != nil {
		t.Fatal(err)
	}
	if err := samples.observe(observation, at.Add(time.Second), nil); err != nil {
		t.Fatal(err)
	}
	if got := len(samples.pools["ram"]); got != 1 {
		t.Fatalf("duplicate fact counted %d times", got)
	}
	if err := samples.observe(diagnosticMemoryFixture(at.Add(time.Second), 8000), at.Add(10*time.Second), nil); err != nil {
		t.Fatal(err)
	}
	if got := len(samples.pools["ram"]); got != 1 {
		t.Fatalf("stale fact counted: %d", got)
	}
	changed := diagnosticMemoryFixture(at.Add(2*time.Second), 8000)
	changed.RunID = "replacement"
	if err := samples.observe(changed, at.Add(2*time.Second), nil); err == nil {
		t.Fatal("producer replacement accepted into run")
	}
}

func TestModelDiagnosticMedianAndPhaseMinimums(t *testing.T) {
	at := time.Date(2026, 11, 7, 0, 0, 0, 0, time.UTC)
	identity := diagnosticIdentityFixture()
	identity.Placement = identity.Placement[:1]
	var samples diagnosticSamples
	for i, value := range []uint64{9000, 7000, 8000, 6000, 5500, 5000, 4500, 4000, 3500} {
		time := at.Add(time.Duration(i) * 2 * time.Second)
		if err := samples.observe(diagnosticMemoryFixture(time, value), time, nil); err != nil {
			t.Fatal(err)
		}
	}
	if !samples.baselineReady(at) {
		t.Fatal("three fresh baseline samples were not accepted")
	}
	status := diagnosticManagerStatus{Run: &modelconfig.DiagnosticRun{RunID: "run", Outcome: "failed"}, Markers: map[string]time.Time{"baseline-ready": at, "loaded": at.Add(6 * time.Second), "low-start": at.Add(10 * time.Second), "low-end": at.Add(12 * time.Second), "high-start": at.Add(14 * time.Second), "high-end": at.Add(16 * time.Second)}}
	profile, err := samples.measured(identity, status, at.Add(18*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	pool := profile.Pools[0]
	if pool.BaselineAvailableBytes != "8000" || pool.AfterLoadAvailableBytes != "5000" || pool.LowMinAvailableBytes != "4500" || pool.HighMinAvailableBytes != "3500" || pool.PeakUsedBytes != "4500" {
		t.Fatalf("sampled pool=%+v", pool)
	}
}

func TestModelDiagnosticAfterLoadNeedsTwoSamplesFromEveryPool(t *testing.T) {
	at := time.Now().UTC()
	samples := diagnosticSamples{baselinePools: []string{"ram", "gpu"}, pools: map[string][]diagnosticPoolSample{
		"ram": {{at: at.Add(-time.Second)}, {at: at}, {at: at.Add(time.Second)}},
		"gpu": {{at: at.Add(-time.Second)}, {at: at}},
	}}
	if samples.phaseReady(at, 2) {
		t.Fatal("pre-load sample filled the missing GPU sample")
	}
	samples.pools["gpu"] = append(samples.pools["gpu"], diagnosticPoolSample{at: at.Add(time.Second)})
	if !samples.phaseReady(at, 2) {
		t.Fatal("two after-load samples from every pool were not ready")
	}
}

func TestModelDiagnosticOverlapRequiresEveryPoolSample(t *testing.T) {
	at := time.Date(2026, 11, 7, 0, 0, 0, 0, time.UTC)
	first, end := float64(1000), float64(3000)
	overlapped := true
	high := modelconfig.MeasuredPhase{Overlapped: &overlapped, Requests: []modelconfig.MeasuredRequest{{FirstTokenMS: &first, EndMS: &end}}}
	placements := diagnosticIdentityFixture().Placement
	pools := map[string][]diagnosticPoolSample{}
	for _, placement := range placements {
		pools[placement.DomainID] = []diagnosticPoolSample{{at: at.Add(2 * time.Second)}}
	}
	if !diagnosticOverlapSampled(high, at, placements, pools) {
		t.Fatal("overlap samples rejected")
	}
	pools["gpu-b"] = []diagnosticPoolSample{{at: at.Add(4 * time.Second)}}
	if diagnosticOverlapSampled(high, at, placements, pools) {
		t.Fatal("unsampled second GPU was covered")
	}
}

func TestModelDiagnosticUMAUsesOnePool(t *testing.T) {
	identity := diagnosticIdentityFixture()
	identity.Placement = []modelconfig.DiagnosticPlacement{{DomainID: "unified", Kind: "shared"}}
	status := diagnosticManagerStatus{Run: &modelconfig.DiagnosticRun{RunID: "run", Outcome: "failed"}}
	profile, err := (&diagnosticSamples{}).measured(identity, status, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if len(profile.Pools) != 1 || profile.Pools[0].Kind != "shared" || profile.Pools[0].PeakUsedBytes != "" {
		t.Fatalf("UMA pools=%+v", profile.Pools)
	}
	if slices.Contains(profile.Notes, "") {
		t.Fatal("empty coverage note")
	}
}

func TestModelDiagnosticCheckpointDrainsBeforePolicyReplay(t *testing.T) {
	b, _ := newPolicyBroker(t)
	peer, manager := newFakeWorkerPipe(t)
	worker := &rpcWorker{peer: peer, done: make(chan struct{})}
	b.setEngineMgr(worker)
	run := modelconfig.DiagnosticRun{RunID: "interrupted", Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture", ConfigurationKey: strings.Repeat("a", 64), Phase: "restoring", Outcome: "cleanup-failed", StartedAt: "2026-11-07T00:00:00Z"}
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == "engine:diagnostic-run-status" {
			return diagnosticStatusReply{Run: run, State: "cleanup-failed", Token: "recovered"}, ""
		}
		return nil, "unexpected method"
	})
	pp, proxy, _ := attachAckingProxy(t, b)
	if !b.replayPolicyToProxy(context.Background(), pp) {
		t.Fatal("pending checkpoint replay failed")
	}
	t.Cleanup(func() {
		var released atomic.Bool
		manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
			if method == "engine:diagnostic-release" {
				released.Store(true)
				return nil, ""
			}
			if released.Load() {
				return diagnosticStatusReply{State: "none"}, ""
			}
			terminal := run.Clone()
			terminal.Outcome = "failed"
			terminal.Phase = "done"
			return diagnosticStatusReply{Run: terminal, State: "cleanup-complete", Token: "recovered", WorkStopped: true}, ""
		})
		if _, err := b.cancelDiagnostic(context.Background(), run.RunID); err != nil {
			t.Errorf("release recovered fixture: %v", err)
		}
	})
	status, err := b.diagnosticStatus("")
	if err != nil {
		t.Fatal(err)
	}
	if status == nil || status.RunID != run.RunID {
		t.Fatalf("recovered active status=%+v", status)
	}
	drained := map[string]bool{}
	policySeen := false
	for _, call := range proxy.snapshot() {
		if call.method == nodepolicy.MethodSetEngineDrain {
			var p nodepolicy.SetEngineDrainParams
			if err := json.Unmarshal(call.params, &p); err != nil {
				t.Fatal(err)
			}
			drained[p.Engine] = p.Drain
		}
		if call.method == nodepolicy.MethodSetPolicy {
			policySeen = true
			for _, profile := range b.frontedEngineProfiles() {
				if !drained[profile.Name] {
					t.Fatalf("policy preceded drain for %s", profile.Name)
				}
			}
		}
	}
	if !policySeen {
		t.Fatal("proxy never received policy")
	}
}

func TestModelDiagnosticUnknownCheckpointBlocksIngress(t *testing.T) {
	b, _ := newPolicyBroker(t)
	peer, manager := newFakeWorkerPipe(t)
	b.setEngineMgr(&rpcWorker{peer: peer})
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) { return nil, "checkpoint unreadable" })
	pp, proxy, _ := attachAckingProxy(t, b)
	if b.replayPolicyToProxy(context.Background(), pp) {
		t.Fatal("ingress activated without checkpoint evidence")
	}
	for _, call := range proxy.snapshot() {
		if call.method == nodepolicy.MethodSetPolicy {
			t.Fatal("policy sent despite unknown checkpoint")
		}
	}
}

func TestModelDiagnosticSamplingJoinsOnCancel(t *testing.T) {
	b, manager, op := diagnosticBrokerFixture(t)
	run := op.run.Clone()
	entered := make(chan struct{})
	var once sync.Once
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		once.Do(func() { close(entered) })
		return diagnosticStatusReply{Run: run, State: "running"}, ""
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- b.sampleDiagnostic(ctx, op) }()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("sampler did not start")
	}
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("cancelled sampler returned success")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("run-owned sampling did not join")
	}
}

func TestModelDiagnosticTerminalWithoutStopProofKeepsHold(t *testing.T) {
	b, manager, op := diagnosticBrokerFixture(t)
	run := op.run.Clone()
	run.Outcome = "failed"
	run.Phase = "done"
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		return diagnosticStatusReply{Run: run, State: "cleanup-complete", WorkStopped: false}, ""
	})
	if _, err := b.cancelDiagnostic(context.Background(), op.runID); err == nil {
		t.Fatal("terminal label substituted for stop proof")
	}
	requireDiagnosticHeld(t, b, op)
}

func TestModelDiagnosticPrivateObservationCannotRelayFromClient(t *testing.T) {
	b, out := newPolicyBroker(t)
	peer, manager := newFakeWorkerPipe(t)
	b.setEngineMgr(&rpcWorker{peer: peer})
	b.handleMessage(clientRequest(1, diagnosticObserveMethod, nil))
	frames := out.frames()
	if len(frames) != 1 || frames[0].Error == nil || frames[0].Error.Code != -32601 {
		t.Fatalf("private observation reply=%+v", frames)
	}
	if calls := manager.snapshot(); len(calls) != 0 {
		t.Fatalf("client reached private observation: %+v", calls)
	}
}

func TestModelDiagnosticConfiguredPausePreviewsUnknownNativeAllocation(t *testing.T) {
	b, _ := newPolicyBroker(t)
	b.scannerSource = &scannerProcess{}
	b.telemetry = newTelemetryCache()
	now := time.Now().UTC()
	boot, err := noderec.MemoryBootClock()
	if err != nil {
		t.Fatal(err)
	}
	input := noderec.NodeTelemetry{HostUUID: b.nodeID, Memory: diagnosticMemoryFixture(now, 9000), MemoryHop: noderec.NewMemoryHop("scanner", noderec.MemoryBootClock), MemoryRequest: &noderec.MemoryRequestContext{Generation: "1", RequestBootNS: strconv.FormatUint(boot, 10), VerifiedSelfLoopback: true}}
	if _, ok := b.telemetry.Upsert(sourceScanner, input, now); !ok {
		t.Fatal("fresh local sample refused")
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.current.PauseMode = nodepolicy.PauseConfigured
	r.current.Availability = nodepolicy.Paused
	r.mu.Unlock()
	identity := diagnosticIdentityFixture()
	identity.ConfigurationKey = ""
	identity.ExecutionKey = strings.Repeat("c", 64)
	identity.ConfigurationKeyReason = "native-fit"
	identity.BuildInfo = ""
	identity.PerSlotContext = 0
	identity.Slots = 0
	identity.Placement = nil
	seedDiagnosticHardware(t, b, identity, now)
	peer, manager := newFakeWorkerPipe(t)
	b.setEngineMgr(&rpcWorker{peer: peer})
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method != diagnosticObserveMethod {
			return nil, "unexpected effect"
		}
		return diagnosticObservation{Identity: identity, EffectsFence: "effects", DrainEngines: []string{"llamacpp"}, StartsEngine: true, Supported: true}, ""
	})
	preview, err := b.previewDiagnostic(context.Background(), modelconfig.DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	if preview.HighPromptTokens != 0 || preview.LowPromptTokens != 0 || preview.BuildInfo != "" || len(preview.Placement) != 0 || !preview.StartsEngine || preview.ExecutionKey != identity.ExecutionKey {
		t.Fatalf("invented stopped-engine runtime facts: %+v", preview)
	}
	if len(preview.Warnings) == 0 {
		t.Fatal("unknown allocation was not disclosed")
	}
	if len(b.buildApplyMaintenance.settings) != 0 {
		t.Fatal("preview acquired execution maintenance")
	}
}

func TestModelDiagnosticRequestedIdentityIncludesSecondGPU(t *testing.T) {
	expected := diagnosticIdentityFixture()
	actual := expected.Clone()
	actual.GPUs[1].Name = "changed second GPU"
	if diagnosticRequestedIdentityMatches(actual, expected) {
		t.Fatal("second GPU identity ignored")
	}
	actual = expected.Clone()
	actual.GPUs[0], actual.GPUs[1] = actual.GPUs[1], actual.GPUs[0]
	if !diagnosticRequestedIdentityMatches(actual, expected) {
		t.Fatal("GPU enumeration order changed identity")
	}
}

func diagnosticAcquisitionFixture(t *testing.T) (*Broker, *fakeWorker, *fakeWorker, *brokerDiagnostic) {
	t.Helper()
	b, _ := newPolicyBroker(t)
	peer, manager := newFakeWorkerPipe(t)
	worker := &rpcWorker{peer: peer, done: make(chan struct{})}
	b.setEngineMgr(worker)
	pp, proxy, _ := attachAckingProxy(t, b)
	proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			return map[string]any{"runId": "proxy-run", "active": 0, "queued": 0, "activeByEngine": map[string]int{}}, ""
		}
		return map[string]bool{"ok": true}, ""
	})
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.incarnation = pp
	r.runID = "proxy-run"
	r.mu.Unlock()
	generation, policy, err := b.diagnosticGate()
	if err != nil {
		t.Fatal(err)
	}
	_, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	op := &brokerDiagnostic{runID: "acquiring", worker: worker, cancel: cancel, done: make(chan struct{}), wake: make(chan struct{}, 1)}
	op.run = modelconfig.DiagnosticRun{RunID: op.runID, Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture", ConfigurationKey: strings.Repeat("a", 64), Phase: "reserving", Outcome: "running", StartedAt: time.Now().Format(time.RFC3339Nano)}
	op.review = diagnosticReview{generation: generation, policy: policy, worker: worker, effects: "effects", preview: modelconfig.DiagnosticPreview{DiagnosticPreviewRequest: modelconfig.DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "fixture.gguf", ProfileKey: "fixture"}, DiagnosticIdentity: diagnosticIdentityFixture()}}
	b.diagnostics.active = op
	t.Cleanup(func() {
		if op.drain != nil {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			if err := op.drain.release(ctx); err != nil {
				t.Errorf("fixture drain cleanup: %v", err)
			}
		}
		if op.lease != nil {
			op.lease.Release()
		}
	})
	return b, manager, proxy, op
}

func TestModelDiagnosticRunRefusalReleasesLedger(t *testing.T) {
	b, manager, _, op := diagnosticAcquisitionFixture(t)
	run := op.run.Clone()
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-reserve":
			return map[string]string{"runId": op.runID, "token": "token", "state": "reserved"}, ""
		case "engine:diagnostic-run":
			return nil, "run refused"
		case "engine:diagnostic-run-status":
			return diagnosticStatusReply{Run: run, State: "reserved", Token: "token", WorkStopped: true}, ""
		case "engine:diagnostic-release":
			return nil, ""
		}
		return nil, "unexpected method"
	})
	if err := b.acquireDiagnostic(context.Background(), op); err == nil {
		t.Fatal("run refusal accepted")
	}
	if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
	if slices.Contains(manager.methods("engine:"), "engine:diagnostic-abort") {
		t.Fatal("refused run was treated as accepted work")
	}
}

func TestModelDiagnosticDrainFailureDoesNotStartManagerRun(t *testing.T) {
	b, manager, proxy, op := diagnosticAcquisitionFixture(t)
	run := op.run.Clone()
	manager.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-reserve":
			return map[string]string{"runId": op.runID, "token": "token", "state": "reserved"}, ""
		case "engine:diagnostic-run-status":
			return diagnosticStatusReply{Run: run, State: "reserved", Token: "token", WorkStopped: true}, ""
		case "engine:diagnostic-release":
			return nil, ""
		}
		return nil, "unexpected effect"
	})
	proxy.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		if method == nodepolicy.MethodAdmissionSnapshot {
			return nil, "snapshot unavailable"
		}
		return map[string]bool{"ok": true}, ""
	})
	if err := b.acquireDiagnostic(context.Background(), op); err == nil {
		t.Fatal("unproved drain accepted")
	}
	if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
	if slices.Contains(manager.methods("engine:"), "engine:diagnostic-run") {
		t.Fatal("run started without idle proof")
	}
}

func TestModelDiagnosticLostRunAcceptanceIsJoined(t *testing.T) {
	b, manager, _, op := diagnosticAcquisitionFixture(t)
	run := op.run.Clone()
	accepted := false
	setDiagnosticContractAnswer(t, manager, op, func(method string, raw json.RawMessage) (any, string) {
		switch method {
		case "engine:diagnostic-reserve":
			return map[string]string{"runId": op.runID, "token": "token", "state": "reserved"}, ""
		case "engine:diagnostic-run":
			accepted = true
			return diagnosticStatusReply{Run: run, State: "accepted"}, holdSuccessReply
		case "engine:diagnostic-run-status":
			if accepted {
				return diagnosticStatusReply{Run: run, State: "running", Token: "token"}, ""
			}
			return diagnosticStatusReply{Run: run, State: "reserved", Token: "token"}, ""
		case "engine:diagnostic-abort":
			terminal := run.Clone()
			terminal.Outcome = "cancelled"
			terminal.Phase = "done"
			return diagnosticStatusReply{Run: terminal, State: "cleanup-complete", WorkStopped: true}, ""
		}
		return nil, "unexpected method"
	})
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := b.acquireDiagnostic(ctx, op); err != nil {
		t.Fatalf("accepted run was not reconciled: %v", err)
	}
	if !op.accepted {
		t.Fatal("lost accepted reply treated as refusal")
	}
	// The real record/release wire calls are part of the acceptance regression.
	if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
		t.Fatal(err)
	}
	requireDiagnosticReleased(t, b)
	if !slices.Contains(manager.methods("engine:"), "engine:diagnostic-abort") {
		t.Fatal("accepted work was released without a join")
	}
}
