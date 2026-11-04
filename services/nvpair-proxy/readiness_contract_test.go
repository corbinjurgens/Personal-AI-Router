// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func TestStrictDestinationContractRejectsLegacyWaitAsStrict(t *testing.T) {
	for _, tc := range []struct{ name, input string }{
		{"legacy wait header", `0`},
		{"incomplete envelope", `{"queueBudgetMs":0}`},
		{"unsupported version", `{"version":2,"queueBudgetMs":0,"coldStartBudgetMs":0,"preOutputBudgetMs":1,"request":{},"identity":{}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var request nodepolicy.StrictAdmissionRequest
			if err := json.Unmarshal([]byte(tc.input), &request); err == nil {
				t.Fatal("legacy/incomplete/unsupported request accepted as strict")
			}
		})
	}
}

// readinessEnvelopeFixture uses only production admission inputs and synthetic
// evidence; it neither listens nor starts an engine or contacts a service.
type readinessEnvelopeFixture struct {
	a      *admissionController
	q      *localQualification
	policy nodepolicy.Policy
	id     nodepolicy.EngineIdentity
	engine string
}

func newReadinessEnvelopeFixture(t *testing.T, engine string) readinessEnvelopeFixture {
	t.Helper()
	digest, launch := strings.Repeat("b", 64), strings.Repeat("c", 64)
	ref := nodepolicy.QualificationRef{ID: "ne-readiness", Revision: "1"}
	q := &localQualification{
		ID: ref.ID, Revision: ref.Revision, Engine: engine, Model: "m",
		ModelDigests: []string{digest}, LaunchDigests: []string{launch},
		Tokenizer: "synthetic-byte-tokenizer", ChatTemplate: "synthetic-template",
		ContextTokens: 4096, MaxOutputTokens: 1024, Routes: []string{routeChatCompletions},
		Accounting: qualificationAccounting{Method: accountingUTF8Bytes, RequestOverheadTokens: 8, MessageOverheadTokens: 4},
		Evidence:   "synthetic readiness fixture; qualifies no real model",
	}
	if err := q.validate(); err != nil {
		t.Fatalf("qualification fixture: %v", err)
	}
	ram, vram := int64(8192), int64(4096)
	id := nodepolicy.EngineIdentity{
		EngineRunID: "ne-engine", SettingsFingerprint: strings.Repeat("a", 64),
		LaunchDigest: launch, ModelDigests: map[string]string{"m": digest},
		Serving: map[string]engines.ServingEnvelope{"m": {
			ModelID: "m", ContentDigest: digest, PerSlotContextTokens: 4096, TotalContextTokens: 8192,
			Slots: 2, MaxOutputTokens: 1024, Memory: engines.ServingMemory{HostRAMBytes: &ram, VRAMBytes: &vram},
			NativeSourceDigest: strings.Repeat("d", 64), Retention: "-1", RouterCapacity: 1,
		}},
	}
	if err := id.Validate(); err != nil {
		t.Fatalf("identity fixture: %v", err)
	}
	p := nodepolicy.Default()
	p.Revision = "1"
	p.Admission.MaxConcurrentPerModel = 2
	p.Profiles = []nodepolicy.Profile{{Name: "ne-model", Engine: engine, Model: "m", ContextTokens: 4096,
		MaxOutputTokens: 1024, MemoryBytes: 4096, Qualification: &ref}}
	if err := p.Validate(); err != nil {
		t.Fatalf("policy fixture: %v", err)
	}
	// Isolate the constructor's evidence/capacity reads as well as later writes.
	dataDir := t.TempDir()
	switch runtime.GOOS {
	case "windows":
		t.Setenv("LOCALAPPDATA", dataDir)
	case "darwin":
		t.Setenv("HOME", dataDir)
	default:
		t.Setenv("XDG_CONFIG_HOME", dataDir)
	}
	a := newAdmissionController(func(string, any) {}, func(string) bool { return true }, normalizeEngineModel)
	a.strict.path = filepath.Join(dataDir, "capacity.json")
	a.strict.storageErr = nil
	a.strict.registry = qualificationRegistry{ref: q}
	return readinessEnvelopeFixture{a: a, q: q, policy: p, id: id, engine: engine}
}

func (f readinessEnvelopeFixture) install(t *testing.T) {
	t.Helper()
	if err := f.a.activatePolicy(f.policy); err != nil {
		t.Fatalf("activate fixture policy: %v", err)
	}
	f.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.engine: f.id})
	f.a.setResidency(map[string][]string{f.engine: {"m"}})
}

func (f readinessEnvelopeFixture) request() nodepolicy.StrictAdmissionRequest {
	return nodepolicy.StrictAdmissionRequest{
		Version: 1, QueueBudgetMs: 1000, ColdStartBudgetMs: 1000, PreOutputBudgetMs: 1000,
		Request: nodepolicy.RoutingRequestKey{OriginNodeID: "origin", ProxyRunID: "run", RequestID: "1"},
		Identity: nodepolicy.StrictModelIdentity{EngineRunID: f.id.EngineRunID,
			SettingsFingerprint: f.id.SettingsFingerprint, ModelDigest: f.id.ModelDigests["m"]},
	}
}

func (f readinessEnvelopeFixture) qualify(t *testing.T) strictValidation {
	t.Helper()
	req := f.request()
	if err := req.Validate(); err != nil {
		t.Fatalf("strict request fixture: %v", err)
	}
	v, err := f.a.qualifyStrict(f.engine, "m", routeChatCompletions,
		[]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":16}`), req.Identity)
	if err != nil {
		t.Fatalf("qualify fixture request: %v", err)
	}
	return v
}

func TestNEReadinessAcceptsExactEnvelopeBounds(t *testing.T) {
	f := newReadinessEnvelopeFixture(t, "llamacpp")
	f.install(t)
	r := f.a.StrictReadiness()
	if r.Capability == nil || len(r.Capability.Models) != 1 {
		t.Fatalf("exact context/output/slots/VRAM bounds unavailable: %+v", r)
	}
	m := r.Capability.Models[0]
	if m.Engine != "llamacpp" || m.Model != "m" || m.MaxConcurrent != 2 || m.Identity != f.request().Identity {
		t.Fatalf("strict entry = %+v, want llama.cpp m, concurrency 2 and fixture identity", m)
	}
	f.qualify(t)
}

func TestNEReadinessRejectsEnvelopeBoundsAndMissingMemory(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*readinessEnvelopeFixture, *engines.ServingEnvelope)
	}{
		{"qualification context above envelope", func(f *readinessEnvelopeFixture, _ *engines.ServingEnvelope) { f.q.ContextTokens = 4097 }},
		{"qualification output above envelope", func(f *readinessEnvelopeFixture, _ *engines.ServingEnvelope) { f.q.MaxOutputTokens = 1025 }},
		{"effective policy concurrency above slots", func(f *readinessEnvelopeFixture, _ *engines.ServingEnvelope) {
			f.policy.Admission.MaxConcurrentPerModel = 3
		}},
		{"effective profile concurrency above slots", func(f *readinessEnvelopeFixture, _ *engines.ServingEnvelope) { f.policy.Profiles[0].MaxConcurrent = 3 }},
		{"profile memory below VRAM", func(f *readinessEnvelopeFixture, _ *engines.ServingEnvelope) { f.policy.Profiles[0].MemoryBytes = 4095 }},
		{"unknown profile memory", func(f *readinessEnvelopeFixture, _ *engines.ServingEnvelope) { f.policy.Profiles[0].MemoryBytes = 0 }},
		{"absent memory", func(_ *readinessEnvelopeFixture, e *engines.ServingEnvelope) { e.Memory = engines.ServingMemory{} }},
		{"null VRAM", func(_ *readinessEnvelopeFixture, e *engines.ServingEnvelope) { e.Memory.VRAMBytes = nil }},
		{"null host RAM", func(_ *readinessEnvelopeFixture, e *engines.ServingEnvelope) { e.Memory.HostRAMBytes = nil }},
		{"negative VRAM", func(_ *readinessEnvelopeFixture, e *engines.ServingEnvelope) { *e.Memory.VRAMBytes = -1 }},
		{"unsafe integer slots", func(_ *readinessEnvelopeFixture, e *engines.ServingEnvelope) { e.Slots = 1 << 53 }},
		{"profile context above qualification", func(f *readinessEnvelopeFixture, _ *engines.ServingEnvelope) {
			f.policy.Profiles[0].ContextTokens = 4097
		}},
		{"profile output above qualification", func(f *readinessEnvelopeFixture, _ *engines.ServingEnvelope) {
			f.policy.Profiles[0].MaxOutputTokens = 1025
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newReadinessEnvelopeFixture(t, "llamacpp")
			e := f.id.Serving["m"].Clone()
			tc.change(&f, &e)
			f.id.Serving["m"] = e
			f.install(t)
			if got := f.a.StrictReadiness(); got.Capability != nil {
				t.Fatalf("out-of-envelope model advertised: %+v", got)
			}
			if _, err := f.a.qualifyStrict(f.engine, "m", routeChatCompletions, nil, f.request().Identity); !errors.Is(err, errStrictUnavailable) {
				t.Fatalf("qualification error = %v, want strict unavailable", err)
			}
		})
	}
}

func TestNEReadinessAcceptsExplicitCPUOnlyMemoryAndProfileConcurrency(t *testing.T) {
	f := newReadinessEnvelopeFixture(t, "llamacpp")
	e := f.id.Serving["m"].Clone()
	*e.Memory.VRAMBytes = 0
	f.id.Serving["m"] = e
	f.policy.Profiles[0].MemoryBytes = 0
	f.policy.Profiles[0].MaxConcurrent = 2
	f.policy.Admission.MaxConcurrentPerModel = 3
	f.install(t)
	r := f.a.StrictReadiness()
	if r.Capability == nil || len(r.Capability.Models) != 1 || r.Capability.Models[0].MaxConcurrent != 2 {
		t.Fatalf("CPU-only envelope with finite profile override refused: %+v", r)
	}
	f.qualify(t)
}

func TestNEReadinessResolvesNormalizedEnvelopeAndContent(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*readinessEnvelopeFixture)
		want   bool
	}{
		{"equal normalized aliases", func(f *readinessEnvelopeFixture) {
			e := f.id.Serving["m"].Clone()
			e.ModelID = "m:latest"
			f.id.Serving["m:latest"] = e
			f.id.ModelDigests["m:latest"] = f.id.ModelDigests["m"]
		}, true},
		{"envelope and content under normalized alias", func(f *readinessEnvelopeFixture) {
			e := f.id.Serving["m"].Clone()
			delete(f.id.Serving, "m")
			e.ModelID = "m:latest"
			f.id.Serving["m:latest"] = e
			f.id.ModelDigests["m:latest"] = f.id.ModelDigests["m"]
		}, true},
		{"empty present map", func(f *readinessEnvelopeFixture) { f.id.Serving = map[string]engines.ServingEnvelope{} }, false},
		{"missing model envelope", func(f *readinessEnvelopeFixture) {
			e := f.id.Serving["m"]
			delete(f.id.Serving, "m")
			e.ModelID = "other"
			f.id.Serving["other"] = e
		}, false},
		{"normalized envelope disagreement", func(f *readinessEnvelopeFixture) {
			e := f.id.Serving["m"].Clone()
			e.ModelID, e.Retention = "m:latest", "60"
			f.id.Serving["m:latest"] = e
		}, false},
		{"normalized content disagreement", func(f *readinessEnvelopeFixture) { f.id.ModelDigests["m:latest"] = strings.Repeat("e", 64) }, false},
		{"envelope content mismatch", func(f *readinessEnvelopeFixture) {
			e := f.id.Serving["m"]
			e.ContentDigest = strings.Repeat("e", 64)
			f.id.Serving["m"] = e
		}, false},
		{"envelope model key mismatch", func(f *readinessEnvelopeFixture) {
			e := f.id.Serving["m"]
			e.ModelID = "other"
			f.id.Serving["m"] = e
		}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newReadinessEnvelopeFixture(t, "ollama")
			tc.change(&f)
			f.install(t)
			r := f.a.StrictReadiness()
			if got := r.Capability != nil; got != tc.want {
				t.Fatalf("strict capability present = %v, want %v: %+v", got, tc.want, r)
			}
			if tc.want {
				if len(r.Capability.Models) != 1 || r.Capability.Models[0].Model != "m:latest" {
					t.Fatalf("normalized capability = %+v, want exactly m:latest", r.Capability)
				}
				f.qualify(t)
			}
		})
	}
}

func TestNEReadinessEnvelopeReplacementFencesValidationAndTicket(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*engines.ServingEnvelope)
	}{
		{"source identity", func(e *engines.ServingEnvelope) { e.NativeSourceDigest = strings.Repeat("e", 64) }},
		{"context", func(e *engines.ServingEnvelope) { e.PerSlotContextTokens, e.TotalContextTokens = 8192, 16384 }},
		{"output", func(e *engines.ServingEnvelope) { e.MaxOutputTokens = 2048 }},
		{"slots", func(e *engines.ServingEnvelope) { e.Slots, e.TotalContextTokens = 3, 12288 }},
		{"host RAM", func(e *engines.ServingEnvelope) { *e.Memory.HostRAMBytes = 16384 }},
		{"VRAM", func(e *engines.ServingEnvelope) { *e.Memory.VRAMBytes = 2048 }},
		{"placement", func(e *engines.ServingEnvelope) { e.Placement = []string{"--n-gpu-layers", "2"} }},
		{"retention", func(e *engines.ServingEnvelope) { e.Retention = "60" }},
		{"router capacity", func(e *engines.ServingEnvelope) { e.RouterCapacity = 2 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newReadinessEnvelopeFixture(t, "llamacpp")
			f.install(t)
			before := f.a.StrictReadiness()
			v := f.qualify(t)
			held, err := f.a.AdmitStrict(context.Background(), f.engine, "m", f.request(), v.generation)
			if err != nil {
				t.Fatalf("admit held ticket: %v", err)
			}
			defer held.Finish(strictNotDispatched)
			e := f.id.Serving["m"].Clone()
			tc.change(&e)
			next := f.id
			next.Serving = map[string]engines.ServingEnvelope{"m": e}
			if !f.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.engine: next}) {
				t.Fatal("envelope replacement did not report a capability change")
			}
			if after := f.a.StrictReadiness(); !reflect.DeepEqual(before, after) {
				t.Fatalf("outward entry changed: before=%+v after=%+v", before, after)
			}
			if _, err := f.a.AdmitStrict(context.Background(), f.engine, "m", f.request(), v.generation); !errors.Is(err, errStrictUnavailable) {
				t.Fatalf("old validation admission error = %v, want unavailable", err)
			}
			if err := held.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
				t.Fatalf("old held dispatch error = %v, want unavailable", err)
			}
			if got := f.a.currentActive(); got != 0 {
				t.Fatalf("undispatched stale ticket retained %d active slots", got)
			}
			fresh := f.qualify(t)
			if fresh.generation <= v.generation {
				t.Fatalf("generation = %d, want above %d", fresh.generation, v.generation)
			}
			ticket, err := f.a.AdmitStrict(context.Background(), f.engine, "m", f.request(), fresh.generation)
			if err != nil {
				t.Fatalf("fresh validation admission: %v", err)
			}
			ticket.Finish(strictNotDispatched)
		})
	}
}

func TestNEReadinessIdenticalEnvelopePreservesGeneration(t *testing.T) {
	f := newReadinessEnvelopeFixture(t, "llamacpp")
	f.install(t)
	v := f.qualify(t)
	next := f.id
	next.Serving = map[string]engines.ServingEnvelope{"m": f.id.Serving["m"].Clone()}
	if f.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.engine: next}) {
		t.Fatal("identical detached envelope reported a capability change")
	}
	if got := f.qualify(t).generation; got != v.generation {
		t.Fatalf("identical envelope changed generation from %d to %d", v.generation, got)
	}
}

func TestNEReadinessWithdrawalReconnectFencesOldValidationAndTicket(t *testing.T) {
	f := newReadinessEnvelopeFixture(t, "llamacpp")
	f.install(t)
	v := f.qualify(t)
	held, err := f.a.AdmitStrict(context.Background(), f.engine, "m", f.request(), v.generation)
	if err != nil {
		t.Fatalf("admit held ticket: %v", err)
	}
	defer held.Finish(strictNotDispatched)
	f.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{})
	if got := f.a.StrictReadiness(); got.Capability != nil {
		t.Fatalf("withdrawn identity advertised: %+v", got)
	}
	if _, err := f.a.qualifyStrict(f.engine, "m", routeChatCompletions, nil, f.request().Identity); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("withdrawn qualification error = %v, want unavailable", err)
	}
	f.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.engine: f.id})
	if _, err := f.a.AdmitStrict(context.Background(), f.engine, "m", f.request(), v.generation); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("pre-withdrawal validation error = %v, want unavailable", err)
	}
	if err := held.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("pre-withdrawal ticket error = %v, want unavailable", err)
	}
	if fresh := f.qualify(t); fresh.generation <= v.generation {
		t.Fatalf("reconnect generation = %d, want above %d", fresh.generation, v.generation)
	}
}

func TestNEReadinessLegacyOllamaRetainsDigestSemantics(t *testing.T) {
	f := newReadinessEnvelopeFixture(t, "ollama")
	f.id.Serving = nil
	f.policy.Profiles[0].MemoryBytes = 0
	f.install(t)
	f.qualify(t)
	next := f.id
	next.ModelDigests = map[string]string{"m": strings.Repeat("e", 64)}
	f.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.engine: next})
	if got := f.a.StrictReadiness(); got.Capability != nil {
		t.Fatalf("unqualified replacement Ollama digest advertised: %+v", got)
	}
	f.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.engine: f.id})
	f.qualify(t)
}
