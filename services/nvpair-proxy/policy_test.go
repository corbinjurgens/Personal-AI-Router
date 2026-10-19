// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"strconv"
	"testing"

	"nvpair-shared/nodepolicy"
)

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
