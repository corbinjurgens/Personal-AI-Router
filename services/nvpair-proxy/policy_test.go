// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"nvpair-shared/nodepolicy"
)

type bufferAckWriter func([]byte) (int, error)

func (w bufferAckWriter) Write(data []byte) (int, error) { return w(data) }
func (w bufferAckWriter) Read([]byte) (int, error)       { return 0, errors.New("no input") }

func policyBufferState(b *bufferBudget) (int64, time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.capacity, b.wait
}

func TestPolicyBufferSettingsBeforeAck(t *testing.T) {
	var p *Proxy
	seen := false
	w := bufferAckWriter(func(data []byte) (int, error) {
		var out frame
		if err := json.Unmarshal(data, &out); err != nil {
			t.Fatal(err)
		}
		if out.ID != nil && out.Error == nil {
			capacity, wait := policyBufferState(p.buffers)
			if capacity != 384<<20 || wait != 0 || p.admission.currentPolicy().Revision != "7" {
				t.Fatalf("ack preceded activation: capacity=%d wait=%s revision=%s", capacity, wait, p.admission.currentPolicy().Revision)
			}
			seen = true
		}
		return len(data), nil
	})
	p = NewProxy(NewCodec(w))
	root := p.buffers
	if capacity, wait := policyBufferState(root); capacity != 256<<20 || wait != 5*time.Second || len(p.enabledFacades()) != 0 {
		t.Fatalf("facade-less defaults = capacity %d, wait %s", capacity, wait)
	}
	pol := policyAt("7")
	pol.Routing.Buffers = nodepolicy.BufferLimits{CapacityMiB: 384, WaitSeconds: 0}
	id := json.RawMessage(`1`)
	p.handlePolicyMethod(&Message{ID: &id, Params: egressJSON(t, nodepolicy.SetPolicyParams{OriginNodeID: "buffer-origin", Policy: pol})}, nodepolicy.MethodSetPolicy)
	if !seen || p.buffers != root {
		t.Fatal("activation lacked an acknowledgement or replaced the process root")
	}
}

func TestPolicyBufferHeadroomAndAtomicRejection(t *testing.T) {
	for _, tc := range []struct {
		name       string
		individual int64
		capacity   int
		valid      bool
	}{
		{"below schema minimum", 64 << 20, 191, false},
		{"minimum at default body cap", 64 << 20, 192, true},
		{"overlap without probe", 96 << 20, 192, false},
		{"calculated next MiB", 96 << 20, 193, true},
		{"largest fitting individual", (1024<<20 - 1) / 2, 1024, true},
		{"impossible at maximum", 512 << 20, 1024, false},
		{"integer overflow", math.MaxInt64, 1024, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, w := policyProxy(t)
			p.maxRequestBytes = tc.individual
			root, previous := p.buffers, p.admission.currentPolicy()
			p.egress.ready, p.egress.sequence, p.egress.notAfter, p.egress.revision = true, 12, 123, previous.Revision
			pol := policyAt("7")
			pol.Routing.Buffers = nodepolicy.BufferLimits{CapacityMiB: tc.capacity, WaitSeconds: 0}
			out := rpc(t, p, w, nodepolicy.MethodSetPolicy, pol)
			if tc.valid {
				ack := ackOf(t, out)
				capacity, wait := policyBufferState(root)
				if ack.Revision != "7" || capacity != int64(tc.capacity)<<20 || wait != 0 {
					t.Fatalf("accepted settings not installed: ack %+v capacity %d wait %s", ack, capacity, wait)
				}
			} else {
				capacity, wait := policyBufferState(root)
				if out.Error == nil || !reflect.DeepEqual(p.admission.currentPolicy(), previous) || capacity != 256<<20 || wait != 5*time.Second || !p.egress.ready || p.egress.sequence != 12 || p.egress.notAfter != 123 || p.egress.transition != nil || p.originNodeID != "test-origin" {
					t.Fatalf("rejection changed policy/root/authority: error %+v capacity %d wait %s origin %q", out.Error, capacity, wait, p.originNodeID)
				}
			}
			if p.buffers != root {
				t.Fatal("policy activation replaced the root")
			}
		})
	}
}

func TestPolicyBufferConversionsRejectOverflow(t *testing.T) {
	p := NewProxy(NewCodec(&recordingWriter{}))
	pol := nodepolicy.Default()
	pol.Routing.Buffers.CapacityMiB = math.MaxInt
	if _, _, err := p.bufferSettings(pol); err == nil {
		t.Fatal("capacity multiplication overflow accepted")
	}
	pol.Routing.Buffers.CapacityMiB = (1 << 44) + 256
	if _, _, err := p.bufferSettings(pol); err == nil {
		t.Fatal("capacity overflow wrapping to a valid positive capacity accepted")
	}
	pol = nodepolicy.Default()
	pol.Routing.Buffers.WaitSeconds = math.MaxInt
	if _, _, err := p.bufferSettings(pol); err == nil {
		t.Fatal("duration multiplication overflow accepted")
	}
}

func TestPolicyBufferLeaseDrainAndIncreaseWake(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		p, w := policyProxy(t)
		activate := func(revision string, capacity, wait int) {
			t.Helper()
			pol := policyAt(revision)
			pol.Routing.Buffers = nodepolicy.BufferLimits{CapacityMiB: capacity, WaitSeconds: wait}
			rpcOK(t, p, w, nodepolicy.MethodSetPolicy, pol)
		}
		activate("1", 384, 5)
		root := p.buffers
		owner := bufferMustAcquire(t, root, 300<<20)
		borrower, err := owner.borrow()
		if err != nil {
			t.Fatal(err)
		}
		defer borrower.close()
		owner.close()
		queued := bufferStartAcquire(t, root, context.Background(), 100<<20)
		synctest.Wait()
		activate("2", 192, 0)
		bufferAssertTotals(t, root, 300<<20, 1, 1)
		queued.pending(t)
		if lease, err := root.acquire(context.Background(), 1); lease != nil || !errors.Is(err, errBufferBusy) {
			t.Fatalf("cap decrease admitted bytes above ceiling: %v %v", lease, err)
		}
		activate("3", 512, 0)
		synctest.Wait()
		queued.success(t).close()
		if p.buffers != root {
			t.Fatal("activation replaced a borrowed root")
		}
		bufferAssertTotals(t, root, 300<<20, 1, 0)
		borrower.close()
		bufferAssertTotals(t, root, 0, 0, 0)
	})
}

func TestPolicyBufferWaitOnlyChangesNewAcquisitions(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		p, w := policyProxy(t)
		root := p.buffers
		bufferMustAcquire(t, root, 256<<20)
		queued := bufferStartAcquire(t, root, context.Background(), 1)
		synctest.Wait()
		pol := policyAt("1")
		pol.Routing.Buffers.WaitSeconds = 0
		rpcOK(t, p, w, nodepolicy.MethodSetPolicy, pol)
		if lease, err := root.acquire(context.Background(), 1); lease != nil || !errors.Is(err, errBufferBusy) {
			t.Fatalf("new wait=0 acquisition = %v %v", lease, err)
		}
		bufferAdvance(4 * time.Second)
		queued.pending(t)
		bufferAdvance(time.Second)
		_, err := queued.result(t)
		bufferAssertBusy(t, err)
	})
}

// frame is one JSON-RPC line the proxy wrote upward.
type frame struct {
	ID     *json.RawMessage `json:"id"`
	Method string           `json:"method"`
	Params json.RawMessage  `json:"params"`
	Result json.RawMessage  `json:"result"`
	Error  *RPCError        `json:"error"`
}

func frames(t *testing.T, w *recordingWriter) []frame {
	t.Helper()
	var out []frame
	for _, line := range w.lines() {
		var f frame
		if err := json.Unmarshal(line, &f); err != nil {
			t.Fatalf("bad frame %q: %v", line, err)
		}
		out = append(out, f)
	}
	return out
}

// notificationsOf returns the params of every notification with method.
func notificationsOf(t *testing.T, w *recordingWriter, method string) []json.RawMessage {
	t.Helper()
	var out []json.RawMessage
	for _, f := range frames(t, w) {
		if f.ID == nil && f.Method == method {
			out = append(out, f.Params)
		}
	}
	return out
}

var rpcSeq int

// rpc delivers one request to the proxy and returns its response frame.
func rpc(t *testing.T, p *Proxy, w *recordingWriter, method string, params any) frame {
	t.Helper()
	rpcSeq++
	if method == nodepolicy.MethodSetPolicy {
		if _, envelope := params.(nodepolicy.SetPolicyParams); !envelope {
			params = map[string]any{"originNodeId": "test-origin", "policy": params}
		}
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	id := json.RawMessage(strconv.Itoa(rpcSeq))
	p.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: method, Params: raw})
	for _, f := range frames(t, w) {
		if f.ID != nil && f.Method == "" && string(*f.ID) == string(id) {
			return f
		}
	}
	t.Fatalf("no response to %s", method)
	return frame{}
}

func rpcOK(t *testing.T, p *Proxy, w *recordingWriter, method string, params any) {
	t.Helper()
	if f := rpc(t, p, w, method, params); f.Error != nil {
		t.Fatalf("%s rejected: %s", method, f.Error.Message)
	}
}

func rpcErr(t *testing.T, p *Proxy, w *recordingWriter, method string, params any) {
	t.Helper()
	if f := rpc(t, p, w, method, params); f.Error == nil {
		t.Fatalf("%s accepted invalid params %v", method, params)
	}
}

func policyProxy(t *testing.T) (*Proxy, *recordingWriter) {
	t.Helper()
	w := &recordingWriter{}
	return newTestProxy(anyProfile(t), NewCodec(w), NewDiscovery(), 11435), w
}

func TestPolicyMethods_DefaultsUntilSet(t *testing.T) {
	p, _ := policyProxy(t)
	got := p.admission.currentPolicy()
	if got.Admission.MaxResidentModels != nodepolicy.Default().Admission.MaxResidentModels {
		t.Fatalf("initial policy is not nodepolicy.Default(): %+v", got.Admission)
	}
	if p.admission.currentAvailability() != nodepolicy.Available {
		t.Fatal("initial availability is not available")
	}
}

func TestPolicyMethods_SetPolicyValidatesAndKeepsPrevious(t *testing.T) {
	p, w := policyProxy(t)
	pol := nodepolicy.Default()
	pol.Admission.MaxResidentModels = 3
	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, pol)
	if got := p.admission.currentPolicy().Admission.MaxResidentModels; got != 3 {
		t.Fatalf("maxResidentModels = %d, want 3", got)
	}

	bad := pol
	bad.Version = 99
	rpcErr(t, p, w, nodepolicy.MethodSetPolicy, bad)
	bad = pol
	bad.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "no/pe", Model: "m"}}}
	rpcErr(t, p, w, nodepolicy.MethodSetPolicy, bad)
	if got := p.admission.currentPolicy().Admission.MaxResidentModels; got != 3 {
		t.Fatalf("an invalid policy replaced the previous one: maxResidentModels = %d", got)
	}
}

func TestPolicyMethods_Availability(t *testing.T) {
	p, w := policyProxy(t)
	rpcOK(t, p, w, nodepolicy.MethodSetAvailability, nodepolicy.SetAvailabilityParams{State: nodepolicy.Paused})
	if p.admission.currentAvailability() != nodepolicy.Paused {
		t.Fatal("availability not applied")
	}
	rpcErr(t, p, w, nodepolicy.MethodSetAvailability, map[string]string{"state": "asleep"})
	rpcErr(t, p, w, nodepolicy.MethodSetAvailability, map[string]any{"state": "available", "cancel": true})
	if p.admission.currentAvailability() != nodepolicy.Paused {
		t.Fatal("an invalid availability replaced the previous one")
	}
}

func TestPolicyMethods_DrainResidencyIntent(t *testing.T) {
	p, w := policyProxy(t)
	rpcOK(t, p, w, nodepolicy.MethodSetEngineDrain, nodepolicy.SetEngineDrainParams{Engine: "ollama", Drain: true})
	rpcErr(t, p, w, nodepolicy.MethodSetEngineDrain, nodepolicy.SetEngineDrainParams{Engine: "v/llm", Drain: true})
	if !p.admission.drain["ollama"] {
		t.Fatal("drain not applied")
	}

	rpcOK(t, p, w, nodepolicy.MethodSetResidency, nodepolicy.SetResidencyParams{
		LoadedByEngine: map[string][]string{"ollama": {"qwen3"}},
	})
	if !p.admission.isLoaded("ollama", "qwen3:latest") {
		t.Fatal("residency not applied under the engine's naming")
	}
	rpcErr(t, p, w, nodepolicy.MethodSetResidency, nodepolicy.SetResidencyParams{
		LoadedByEngine: map[string][]string{"v/llm": {"x"}},
	})
	rpcErr(t, p, w, nodepolicy.MethodSetResidency, map[string]any{})
	if !p.admission.isLoaded("ollama", "qwen3") {
		t.Fatal("invalid residency replaced the previous one")
	}

	rpcOK(t, p, w, nodepolicy.MethodSetEngineIntent, nodepolicy.SetEngineIntentParams{
		EnabledByEngine: map[string]bool{"ollama": true, "lmstudio": false},
	})
	if on, known := p.admission.engineIntent("lmstudio"); !known || on {
		t.Fatal("intent not applied")
	}
	rpcErr(t, p, w, nodepolicy.MethodSetEngineIntent, nodepolicy.SetEngineIntentParams{
		EnabledByEngine: map[string]bool{"v/llm": true},
	})
	if on, _ := p.admission.engineIntent("ollama"); !on {
		t.Fatal("invalid intent replaced the previous one")
	}
}

// engine-manager reports every registered engine, so a descriptor engine in
// a drain, residency or intent message is ordinary state, not a reason to
// reject the whole message.
func TestPolicyMethods_AcceptDescriptorEngineIDs(t *testing.T) {
	p, w := policyProxy(t)
	rpcOK(t, p, w, nodepolicy.MethodSetEngineDrain, nodepolicy.SetEngineDrainParams{Engine: "llamacpp-custom", Drain: true})
	if !p.admission.drain["llamacpp-custom"] {
		t.Fatal("descriptor engine drain not applied")
	}
	rpcOK(t, p, w, nodepolicy.MethodSetResidency, nodepolicy.SetResidencyParams{
		LoadedByEngine: map[string][]string{"ollama": {"qwen3"}, "llamacpp-custom": {"org/model:Q4_K_M"}},
	})
	if !p.admission.isLoaded("llamacpp-custom", "org/model:Q4_K_M") || !p.admission.isLoaded("ollama", "qwen3:latest") {
		t.Fatal("residency with a descriptor engine not applied")
	}
	rpcOK(t, p, w, nodepolicy.MethodSetEngineIntent, nodepolicy.SetEngineIntentParams{
		EnabledByEngine: map[string]bool{"ollama": true, "strata": false},
	})
	if on, known := p.admission.engineIntent("strata"); !known || on {
		t.Fatal("descriptor engine intent not applied")
	}
	policy := nodepolicy.Default()
	policy.Tiers = map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "llamacpp-custom", Model: "m"}}}
	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, policy)
}

// The node policy methods are process-scoped: an engine-addressed copy is not
// one of them.
func TestPolicyMethods_AreProcessScoped(t *testing.T) {
	p, w := policyProxy(t)
	f := rpc(t, p, w, "ollama:"+nodepolicy.MethodSetAvailability, nodepolicy.SetAvailabilityParams{State: nodepolicy.Paused})
	if f.Error == nil {
		t.Fatal("an engine-addressed node/set-availability was accepted")
	}
	if p.admission.currentAvailability() != nodepolicy.Available {
		t.Fatal("an engine-addressed request changed availability")
	}
}

// policyAt is the default policy stamped with revision.
func policyAt(revision string) nodepolicy.Policy {
	pol := nodepolicy.Default()
	pol.Revision = revision
	return pol
}

// ackOf decodes a successful response as the acknowledgement it must be.
func ackOf(t *testing.T, f frame) nodepolicy.PolicyAck {
	t.Helper()
	if f.Error != nil {
		t.Fatalf("request rejected: %s", f.Error.Message)
	}
	var ack nodepolicy.PolicyAck
	if err := json.Unmarshal(f.Result, &ack); err != nil {
		t.Fatalf("result %s is not an acknowledgement: %v", f.Result, err)
	}
	return ack
}

func TestPolicyActivation_SetPolicyAcknowledgesItsRevisionAndTheRunID(t *testing.T) {
	p, w := policyProxy(t)
	if p.runID == "" {
		t.Fatal("the proxy has no run id to acknowledge with")
	}
	pol := policyAt("1700")
	pol.Admission.MaxResidentModels = 3
	ack := ackOf(t, rpc(t, p, w, nodepolicy.MethodSetPolicy, pol))
	if ack.Revision != "1700" || ack.RunID != p.runID {
		t.Fatalf("acknowledgement = %+v, want revision 1700 from run %q", ack, p.runID)
	}
	got := p.admission.currentPolicy()
	if got.Revision != "1700" || got.Admission.MaxResidentModels != 3 {
		t.Fatalf("active pair = revision %q, maxResidentModels %d; want 1700 and 3", got.Revision, got.Admission.MaxResidentModels)
	}
}

func TestPolicyActivation_StartsWithoutARevision(t *testing.T) {
	p, _ := policyProxy(t)
	if got := p.admission.currentPolicy().Revision; got != "" {
		t.Fatalf("revision before any activation = %q, want none", got)
	}
}

func TestPolicyActivation_RefusesAnOutOfOrderRevisionAndKeepsThePreviousPolicy(t *testing.T) {
	p, w := policyProxy(t)
	newer := policyAt("200")
	newer.Admission.MaxResidentModels = 4
	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, newer)

	older := policyAt("100")
	older.Admission.MaxResidentModels = 2
	if f := rpc(t, p, w, nodepolicy.MethodSetPolicy, older); f.Error == nil {
		t.Fatal("a revision older than the active one was acknowledged")
	}
	got := p.admission.currentPolicy()
	if got.Revision != "200" || got.Admission.MaxResidentModels != 4 {
		t.Fatalf("after the refusal the pair = revision %q, maxResidentModels %d; want 200 and 4", got.Revision, got.Admission.MaxResidentModels)
	}

	// The same revision again is an idempotent replay, not an out-of-order one.
	if ack := ackOf(t, rpc(t, p, w, nodepolicy.MethodSetPolicy, newer)); ack.Revision != "200" {
		t.Fatalf("replay of the active revision acknowledged %q", ack.Revision)
	}
}

func TestPolicyActivation_AvailabilityAnswersOkAndNeverMovesTheRevision(t *testing.T) {
	p, w := policyProxy(t)
	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, policyAt("100"))
	f := rpc(t, p, w, nodepolicy.MethodSetAvailability, nodepolicy.SetAvailabilityParams{State: nodepolicy.Paused})
	if f.Error != nil {
		t.Fatalf("node/set-availability rejected: %s", f.Error.Message)
	}
	var result map[string]bool
	if err := json.Unmarshal(f.Result, &result); err != nil || !result["ok"] || len(result) != 1 {
		t.Fatalf("node/set-availability result = %s (%v), want {\"ok\":true}", f.Result, err)
	}
	if got := p.admission.currentPolicy().Revision; got != "100" {
		t.Fatalf("held revision after an availability change = %q, want 100 (only node/set-policy activates a revision)", got)
	}
	if p.admission.currentAvailability() != nodepolicy.Paused {
		t.Fatal("the availability was not applied")
	}
}

// strictReports decodes every admission/strict-readiness notification sent so
// far, failing on one that does not validate.
func strictReports(t *testing.T, w *recordingWriter) []nodepolicy.StrictAdmissionReadiness {
	t.Helper()
	var out []nodepolicy.StrictAdmissionReadiness
	for _, raw := range notificationsOf(t, w, nodepolicy.NotifyStrictReadiness) {
		var r nodepolicy.StrictAdmissionReadiness
		if err := json.Unmarshal(raw, &r); err != nil {
			t.Fatalf("decode strict readiness %s: %v", raw, err)
		}
		if err := r.Validate(); err != nil {
			t.Fatalf("strict readiness %s does not validate: %v", raw, err)
		}
		out = append(out, r)
	}
	return out
}

func TestStrictReadiness_ReportedAfterEachAcknowledgement(t *testing.T) {
	p, w := policyProxy(t)
	if got := strictReports(t, w); len(got) != 0 {
		t.Fatalf("reports before any policy = %+v, want none", got)
	}
	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, policyAt("1700"))
	got := strictReports(t, w)
	if len(got) != 1 {
		t.Fatalf("reports after one acknowledgement = %d, want 1", len(got))
	}
	if got[0].PolicyRevision != "1700" || got[0].ProxyRunID != p.runID || got[0].Capability != nil {
		t.Fatalf("report = %+v, want revision 1700 from run %q with no capability", got[0], p.runID)
	}

	rpcErr(t, p, w, nodepolicy.MethodSetPolicy, policyAt("1600"))
	rpcOK(t, p, w, nodepolicy.MethodSetAvailability, nodepolicy.SetAvailabilityParams{State: nodepolicy.Paused})
	if got := strictReports(t, w); len(got) != 1 {
		t.Fatalf("reports after a refused policy and an availability change = %d, want still 1", len(got))
	}
}

func TestStrictReadiness_ReportsTheInstalledCapability(t *testing.T) {
	p, w := policyProxy(t)
	a := p.admission
	a.mu.Lock()
	a.strict.path = filepath.Join(t.TempDir(), "capacity.json")
	a.strict.storageErr = nil
	refOllama := nodepolicy.QualificationRef{ID: "q-ollama", Revision: "1"}
	refLMStudio := nodepolicy.QualificationRef{ID: "q-lmstudio", Revision: "1"}
	a.strict.registry = qualificationRegistry{
		refOllama:   testQualification(refOllama.ID, "ollama", "m"),
		refLMStudio: testQualification(refLMStudio.ID, "lmstudio", "Exact Name"),
	}
	a.mu.Unlock()
	qualified := func(revision string, limit int) nodepolicy.Policy {
		pol := policyAt(revision)
		pol.Admission.MaxConcurrentPerModel = limit
		pol.Profiles = []nodepolicy.Profile{
			{Name: "o", Engine: "ollama", Model: "m", ContextTokens: 4096, MaxOutputTokens: 1024, Qualification: &refOllama},
			{Name: "l", Engine: "lmstudio", Model: "Exact Name", ContextTokens: 4096, MaxOutputTokens: 1024, Qualification: &refLMStudio},
		}
		return pol
	}
	identity := func(engines map[string]nodepolicy.EngineIdentity) nodepolicy.SetModelIdentityParams {
		return nodepolicy.SetModelIdentityParams{Engines: engines}
	}
	both := map[string]nodepolicy.EngineIdentity{"ollama": testIdentity("m"), "lmstudio": testIdentity("Exact Name")}
	last := func(t *testing.T) nodepolicy.StrictAdmissionReadiness {
		t.Helper()
		got := strictReports(t, w)
		if len(got) == 0 {
			t.Fatal("no strict readiness report")
		}
		return got[len(got)-1]
	}
	models := func(r nodepolicy.StrictAdmissionReadiness) []string {
		var out []string
		if r.Capability != nil {
			for _, m := range r.Capability.Models {
				out = append(out, fmt.Sprintf("%s/%s/%d", m.Engine, m.Model, m.MaxConcurrent))
			}
		}
		return out
	}

	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, qualified("10", 1))
	if report := last(t); report.Capability != nil {
		t.Fatalf("report before any engine identity = %+v, want no capability", report)
	}
	rpcOK(t, p, w, nodepolicy.MethodSetModelIdentity, identity(both))
	report := last(t)
	want := []nodepolicy.StrictAdmissionModel{
		{Engine: "lmstudio", Model: "Exact Name", MaxConcurrent: 1, Identity: strictRequest().Identity},
		{Engine: "ollama", Model: "m:latest", MaxConcurrent: 1, Identity: strictRequest().Identity},
	}
	if report.PolicyRevision != "10" || report.ProxyRunID != p.runID || report.Capability == nil ||
		report.Capability.Version != nodepolicy.StrictAdmissionVersion || !reflect.DeepEqual(report.Capability.Models, want) {
		t.Fatalf("report after the identity = %+v (capability %+v), want revision 10 with %+v", report, report.Capability, want)
	}

	t.Run("a changed limit is reinstalled under the new revision", func(t *testing.T) {
		rpcOK(t, p, w, nodepolicy.MethodSetPolicy, qualified("11", 2))
		if report := last(t); report.PolicyRevision != "11" || !slices.Equal(models(report), []string{"lmstudio/Exact Name/2", "ollama/m:latest/2"}) {
			t.Fatalf("report after the limit changed = %+v, want revision 11 with both at 2", report)
		}
	})
	t.Run("an identity without an engine withdraws its model", func(t *testing.T) {
		rpcOK(t, p, w, nodepolicy.MethodSetModelIdentity, identity(map[string]nodepolicy.EngineIdentity{"lmstudio": testIdentity("Exact Name")}))
		if got := models(last(t)); !slices.Equal(got, []string{"lmstudio/Exact Name/2"}) {
			t.Fatalf("models after ollama lost its identity = %v", got)
		}
	})
	t.Run("a same-name replacement withdraws the model", func(t *testing.T) {
		replaced := testIdentity()
		replaced.ModelDigests["Exact Name"] = strings.Repeat("f", 64)
		rpcOK(t, p, w, nodepolicy.MethodSetModelIdentity, identity(map[string]nodepolicy.EngineIdentity{"ollama": testIdentity("m"), "lmstudio": replaced}))
		if got := models(last(t)); !slices.Equal(got, []string{"ollama/m:latest/2"}) {
			t.Fatalf("models after the lmstudio model was replaced = %v", got)
		}
	})
	t.Run("an invalid identity is refused and the previous one kept", func(t *testing.T) {
		before := len(strictReports(t, w))
		bad := testIdentity("m")
		bad.LaunchDigest = "not-a-digest"
		rpcErr(t, p, w, nodepolicy.MethodSetModelIdentity, identity(map[string]nodepolicy.EngineIdentity{"ollama": bad}))
		if got := strictReports(t, w); len(got) != before {
			t.Fatalf("a refused identity sent a report: %+v", got[len(got)-1])
		}
		if got := models(p.admission.StrictReadiness()); !slices.Equal(got, []string{"ollama/m:latest/2"}) {
			t.Fatalf("models after a refused identity = %v", got)
		}
	})
	t.Run("untrusted strict storage withdraws it", func(t *testing.T) {
		a.mu.Lock()
		a.strict.storageErr = errors.New("unreadable capacity document")
		a.mu.Unlock()
		rpcOK(t, p, w, nodepolicy.MethodSetModelIdentity, identity(both))
		if report := last(t); report.Capability != nil {
			t.Fatalf("report with untrusted storage = %+v, want no capability", report)
		}
	})
}
