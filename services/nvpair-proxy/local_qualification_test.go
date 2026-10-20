// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

// testLaunchDigest is the launch every test qualification covers.
var testLaunchDigest = strings.Repeat("c", 64)

// testQualification qualifies engine and model under the identity
// strictRequest names.
func testQualification(id, engine, model string) *localQualification {
	return &localQualification{
		ID: id, Revision: "1", Engine: engine, Model: model,
		ModelDigests: []string{strictRequest().Identity.ModelDigest}, LaunchDigests: []string{testLaunchDigest},
		Tokenizer: "test-tokenizer", ChatTemplate: "test-template", ContextTokens: 4096, MaxOutputTokens: 1024,
		Routes:     []string{routeChatCompletions},
		Accounting: qualificationAccounting{Method: accountingUTF8Bytes, RequestOverheadTokens: 8, MessageOverheadTokens: 4},
		Evidence:   "test fixture",
	}
}

// testIdentity is the engine identity whose strict entries carry exactly
// strictRequest().Identity for every model named.
func testIdentity(models ...string) nodepolicy.EngineIdentity {
	id := strictRequest().Identity
	digests := map[string]string{}
	for _, m := range models {
		digests[m] = id.ModelDigest
	}
	return nodepolicy.EngineIdentity{EngineRunID: id.EngineRunID, SettingsFingerprint: id.SettingsFingerprint, LaunchDigest: testLaunchDigest, ModelDigests: digests}
}

// installQualified installs strict capacity for keys through the production
// inputs: a qualification for each, the active policy's profiles naming them
// (concurrency from the policy), and the engine identity.
func installQualified(t *testing.T, a *admissionController, keys ...admKey) {
	t.Helper()
	registry := qualificationRegistry{}
	identities := map[string]nodepolicy.EngineIdentity{}
	p := a.currentPolicy()
	p.Profiles = nil
	for i, k := range keys {
		q := testQualification(fmt.Sprintf("test-%d", i), k.engine, k.model)
		ref := nodepolicy.QualificationRef{ID: q.ID, Revision: q.Revision}
		registry[ref] = q
		id, ok := identities[k.engine]
		if !ok {
			id = testIdentity()
		}
		id.ModelDigests[k.model] = strictRequest().Identity.ModelDigest
		identities[k.engine] = id
		p.Profiles = append(p.Profiles, nodepolicy.Profile{
			Name: fmt.Sprintf("p%d", i), Engine: k.engine, Model: k.model, ContextTokens: 4096, MaxOutputTokens: 1024, Qualification: &ref,
		})
	}
	a.mu.Lock()
	a.strict.registry = registry
	a.mu.Unlock()
	a.setPolicy(p)
	a.setModelIdentity(identities)
}

// admitStrictNow is AdmitStrict for the model's current installation, as
// serveStrictLocal calls it after qualifyStrict.
func (a *admissionController) admitStrictNow(ctx context.Context, engine, model string, req nodepolicy.StrictAdmissionRequest) (strictAdmissionTicket, error) {
	a.mu.Lock()
	generation := a.strict.generation[admKey{engine: engine, model: a.normalize(engine, model)}]
	a.mu.Unlock()
	return a.AdmitStrict(ctx, engine, model, req, generation)
}

// strictBody is a qualified chat request for model with a small output bound.
func strictBody(model string) string {
	return `{"model":"` + model + `","messages":[{"role":"user","content":"hi"}],"max_tokens":16}`
}

type qualificationCase struct {
	Name         string          `json:"name"`
	Route        string          `json:"route"`
	Body         json.RawMessage `json:"body"`
	RawBody      string          `json:"rawBody"`
	InputTokens  int             `json:"inputTokens"`
	OutputTokens int             `json:"outputTokens"`
	Reject       string          `json:"reject"`
}

// qualificationFixture is testdata/local-qualification.json: a synthetic
// artifact and the request cases it is tested with.
func qualificationFixture(t *testing.T) (*localQualification, nodepolicy.Profile, []qualificationCase) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "local-qualification.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Artifact json.RawMessage     `json:"artifact"`
		Cases    []qualificationCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	registry, err := parseQualificationDocument(doc.Artifact)
	if err != nil {
		t.Fatalf("fixture artifact: %v", err)
	}
	q, ok := registry.lookup(nodepolicy.QualificationRef{ID: "synthetic-fixture", Revision: "1"})
	if !ok {
		t.Fatal("fixture artifact lacks synthetic-fixture revision 1")
	}
	profile := nodepolicy.Profile{Name: "fixture", Engine: q.Engine, Model: q.Model, ContextTokens: q.ContextTokens,
		MaxOutputTokens: q.MaxOutputTokens, RequestOptions: q.RequestOptions, Qualification: &nodepolicy.QualificationRef{ID: q.ID, Revision: q.Revision}}
	return q, profile, doc.Cases
}

func TestQualificationFixtureCases(t *testing.T) {
	q, profile, cases := qualificationFixture(t)
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			route := tc.Route
			if route == "" {
				route = routeChatCompletions
			}
			body := []byte(tc.RawBody)
			if tc.RawBody == "" {
				body = tc.Body
			}
			_, bound, err := q.validateRequest(profile, route, body)
			if tc.Reject != "" {
				var invalid *strictQualificationError
				if !errors.As(err, &invalid) || !strings.Contains(err.Error(), tc.Reject) {
					t.Fatalf("error = %v, want a qualification rejection containing %q", err, tc.Reject)
				}
				return
			}
			if err != nil {
				t.Fatalf("rejected: %v", err)
			}
			if bound.input != tc.InputTokens || bound.output != tc.OutputTokens {
				t.Fatalf("bound = %d input, %d output; want %d, %d", bound.input, bound.output, tc.InputTokens, tc.OutputTokens)
			}
		})
	}
}

func TestQualificationBorrowedViewsPreserveScalarAndStructDecoding(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	profile.RequestOptions = nil
	for _, tc := range []struct {
		name, message, extras, reject string
	}{
		{name: "nested field case folding", message: `{"RoLe":"user","CoNtEnT":"hi"}`},
		{name: "optional null zero values", message: `{"role":"user","content":"hi","name":null,"tool_call_id":null,"tool_calls":null}`, extras: `,"stream":null,"stream_options":null,"stop":[null,"x"],"seed":null,"temperature":null`},
		{name: "int64 minimum", extras: `,"seed":-9223372036854775808`},
		{name: "int64 maximum", extras: `,"seed":9223372036854775807`},
		{name: "int64 overflow", extras: `,"seed":9223372036854775808`, reject: "seed must be an integer"},
		{name: "int64 underflow", extras: `,"seed":-9223372036854775809`, reject: "seed must be an integer"},
		{name: "finite float", extras: `,"temperature":1e308`},
		{name: "overflowing float", extras: `,"temperature":1e309`, reject: "temperature must be a number"},
		{name: "null function parameters rejected", extras: `,"tools":[{"type":"function","function":{"name":"f","parameters":null}}]`, reject: "parameters must be an object"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			message := tc.message
			if message == "" {
				message = `{"role":"user","content":"hi"}`
			}
			body := []byte(`{"model":"m","messages":[` + message + `],"max_tokens":16` + tc.extras + `}`)
			_, bound, err := q.validateRequest(profile, routeChatCompletions, body)
			if tc.reject != "" {
				var rejected *strictQualificationError
				if !errors.As(err, &rejected) || !strings.Contains(rejected.reason, tc.reject) {
					t.Fatalf("error = %v, want qualification rejection containing %q", err, tc.reject)
				}
				return
			}
			if err != nil || bound.input != 18 || bound.output != 16 {
				t.Fatalf("bound = %+v, error = %v; want 18 input and 16 output", bound, err)
			}
		})
	}
}

func TestQualificationUniqueKeyWorkspaceIsChargedAndReleased(t *testing.T) {
	body := []byte(`{"a":1}`)
	for _, tc := range []struct {
		name     string
		capacity int64
		limit    bool
	}{
		{name: "refuses member storage before allocation", capacity: 288, limit: true},
		{name: "releases completed member storage", capacity: 289},
	} {
		t.Run(tc.name, func(t *testing.T) {
			budget := bufferTestBudget(t, tc.capacity, 0)
			scope, err := budget.newRequest(tc.capacity)
			if err != nil {
				t.Fatal(err)
			}
			owner := &ownedInferenceBuffers{scope: scope}
			defer owner.close()
			ctx := context.WithValue(context.Background(), ownedInferenceContextKey{}, owner)
			ok, err := uniqueKeysContext(body, ctx)
			if tc.limit {
				bufferAssertLimit(t, err)
				if ok {
					t.Fatal("a refused scan reported unique keys")
				}
			} else if err != nil || !ok {
				t.Fatalf("unique = %v, error = %v; want a completed scan", ok, err)
			}
			bufferAssertTotals(t, budget, 0, 0, 0)
		})
	}
}

func TestQualificationFloatRepresentationRefusesBeforeParsing(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	profile.RequestOptions = nil
	number := "0." + strings.Repeat("0", 1024)
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"temperature":` + number + `}`)
	// The number conversion, possible error clone, and workspace bookkeeping
	// need 2*len(number)+512 bytes even when its numeric value is simply zero.
	capacity := 2*int64(len(number)) + 511
	budget := bufferTestBudget(t, capacity, 0)
	scope, err := budget.newRequest(capacity)
	if err != nil {
		t.Fatal(err)
	}
	owner := &ownedInferenceBuffers{scope: scope}
	defer owner.close()
	ctx := context.WithValue(context.Background(), ownedInferenceContextKey{}, owner)
	_, _, err = q.validateRequest(profile, routeChatCompletions, body, ctx)
	bufferAssertLimit(t, err)
	bufferAssertTotals(t, budget, 0, 0, 0)
}

func TestQualificationDynamicFieldErrorRetainsMeasuredStorage(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	key := strings.Repeat("\n\u0085", 4096)
	quoted, err := json.Marshal(key)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":16,` + string(quoted) + `:0}`)
	budget := bufferTestBudget(t, 1<<20, 0)
	scope, err := budget.newRequest(1 << 20)
	if err != nil {
		t.Fatal(err)
	}
	owner := &ownedInferenceBuffers{scope: scope}
	defer owner.close()
	ctx := context.WithValue(context.Background(), ownedInferenceContextKey{}, owner)
	_, _, err = q.validateRequest(profile, routeChatCompletions, body, ctx)
	var rejected *strictQualificationError
	if !errors.As(err, &rejected) {
		t.Fatalf("error = %v, want an unsupported-field qualification rejection", err)
	}
	want := fmt.Sprintf("field %q is not supported", key)
	if rejected.reason != want || rejected.Error() != "strict qualification: "+want {
		t.Fatal("the retained error changed Go quoted field spelling")
	}
	bufferAssertTotals(t, budget, 3*int64(len(want))+int64(len("strict qualification: "))+256, 1, 0)
	owner.close()
	bufferAssertTotals(t, budget, 0, 0, 0)
}

func TestQualificationUniqueKeysPreservesUnicodeAndDepthRules(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		unique    bool
	}{
		{name: "Kelvin sign", raw: `{"K":0,"\u212a":1}`},
		{name: "long s", raw: `{"S":0,"\u017f":1}`},
		{name: "dotless i follows upper then lower rule", raw: `{"i":0,"\u0131":1}`},
		{name: "distinct Unicode keys", raw: `{"é":0,"e":1}`, unique: true},
		{name: "exact strict nesting", raw: strings.Repeat("[", maxStrictBodyDepth) + "0" + strings.Repeat("]", maxStrictBodyDepth), unique: true},
		{name: "excess strict nesting", raw: strings.Repeat("[", maxStrictBodyDepth+1) + "0" + strings.Repeat("]", maxStrictBodyDepth+1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := uniqueKeys([]byte(tc.raw)); got != tc.unique {
				t.Fatalf("unique = %v, want %v", got, tc.unique)
			}
		})
	}
}

// The input bound plus the output headroom must fit the served context: a
// request exactly at the context is accepted and one byte more is not.
func TestQualificationContextBoundary(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	body := func(n, output int) []byte {
		return []byte(fmt.Sprintf(`{"model":"m","messages":[{"role":"user","content":%q}],"max_tokens":%d}`, strings.Repeat("a", n), output))
	}
	// request 8 + message 4 + "user" 4 + content.
	fits := q.ContextTokens - q.MaxOutputTokens - 16
	if _, bound, err := q.validateRequest(profile, routeChatCompletions, body(fits, q.MaxOutputTokens)); err != nil || bound.input+bound.output != q.ContextTokens {
		t.Fatalf("a request filling the context: bound %+v, error %v", bound, err)
	}
	if _, _, err := q.validateRequest(profile, routeChatCompletions, body(fits+1, q.MaxOutputTokens)); err == nil || !strings.Contains(err.Error(), "exceed the qualified context") {
		t.Fatalf("a request one token over the context: error %v", err)
	}
	// Less output headroom leaves room for more input.
	if _, _, err := q.validateRequest(profile, routeChatCompletions, body(fits+1, q.MaxOutputTokens-1)); err != nil {
		t.Fatalf("one token less output: %v", err)
	}

	t.Run("a profile serving less context than qualified bounds it", func(t *testing.T) {
		smaller := profile
		smaller.ContextTokens = 2048
		if _, _, err := q.validateRequest(smaller, routeChatCompletions, body(fits, 16)); err == nil {
			t.Fatal("a request over the profile's context was accepted")
		}
	})
	t.Run("a profile with a lower output ceiling bounds it", func(t *testing.T) {
		lower := profile
		lower.MaxOutputTokens = 100
		if _, _, err := q.validateRequest(lower, routeChatCompletions, body(10, 101)); err == nil || !strings.Contains(err.Error(), "ceiling 100") {
			t.Fatalf("error = %v, want the profile's ceiling", err)
		}
	})
}

// The profile's defaults are merged before the request is accounted and
// forwarded; a request cannot override them with its own context length.
func TestQualificationAppliesProfileDefaults(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	effective, _, err := q.validateRequest(profile, routeChatCompletions, []byte(strictBody("fixture-model")))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(effective, &got); err != nil {
		t.Fatal(err)
	}
	if compactLen(got["options"]) != len(`{"num_ctx":4096}`) || !strings.Contains(string(got["options"]), "4096") || string(got["keep_alive"]) != `"5m"` {
		t.Fatalf("effective body = %s, want the profile's defaults merged in", effective)
	}
	t.Run("defaults may supply the output bound", func(t *testing.T) {
		withOutput := profile
		withOutput.RequestOptions = json.RawMessage(`{"max_tokens":64}`)
		_, bound, err := q.validateRequest(withOutput, routeChatCompletions, []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
		if err != nil || bound.output != 64 {
			t.Fatalf("bound %+v, error %v; want the default output bound", bound, err)
		}
	})
	t.Run("defaults are checked like the request", func(t *testing.T) {
		tooLarge := profile
		tooLarge.RequestOptions = json.RawMessage(`{"max_tokens":5000}`)
		if _, _, err := q.validateRequest(tooLarge, routeChatCompletions, []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)); err == nil {
			t.Fatal("a default output bound over the ceiling was accepted")
		}
	})
}

func TestQualificationDocumentRejects(t *testing.T) {
	valid := func() map[string]any {
		return map[string]any{
			"id": "q", "revision": "1", "engine": "ollama", "model": "m:latest",
			"modelDigests": []string{strings.Repeat("1", 64)}, "launchDigests": []string{strings.Repeat("2", 64)},
			"tokenizer": "t", "chatTemplate": "c", "contextTokens": 4096, "maxOutputTokens": 1024,
			"routes":     []string{routeChatCompletions},
			"accounting": map[string]any{"method": accountingUTF8Bytes, "requestOverheadTokens": 1, "messageOverheadTokens": 1, "toolOverheadTokens": 1, "schemaOverheadTokens": 1},
			"evidence":   "measured",
		}
	}
	doc := func(entries ...map[string]any) []byte {
		data, _ := json.Marshal(map[string]any{"version": 1, "qualifications": entries})
		return data
	}
	if reg, err := parseQualificationDocument(doc(valid())); err != nil || len(reg) != 1 {
		t.Fatalf("valid document: %d entries, %v", len(reg), err)
	}
	for _, tc := range []struct {
		name string
		data func() []byte
		want string
	}{
		{"another version", func() []byte { return []byte(`{"version":2,"qualifications":[]}`) }, "version"},
		{"no version", func() []byte { return []byte(`{"qualifications":[]}`) }, "version"},
		{"unknown field", func() []byte { e := valid(); e["guess"] = true; return doc(e) }, "unknown field"},
		{"estimated accounting", func() []byte {
			e := valid()
			e["accounting"] = map[string]any{"method": "bytes-div-4"}
			return doc(e)
		}, "accounting method"},
		{"unsupported route", func() []byte { e := valid(); e["routes"] = []string{"/api/chat"}; return doc(e) }, "route"},
		{"no model digest", func() []byte { e := valid(); e["modelDigests"] = []string{}; return doc(e) }, "modelDigests"},
		{"prefixed launch digest", func() []byte {
			e := valid()
			e["launchDigests"] = []string{"sha256:" + strings.Repeat("2", 64)}
			return doc(e)
		}, "launchDigests"},
		{"output filling the context", func() []byte { e := valid(); e["maxOutputTokens"] = 4096; return doc(e) }, "maxOutputTokens below it"},
		{"context beyond the bound", func() []byte {
			e := valid()
			e["contextTokens"] = maxQualifiedContextTokens + 1
			return doc(e)
		}, "contextTokens must be"},
		{"overhead filling an extreme context", func() []byte {
			e := valid()
			e["contextTokens"] = int64(1) << 62
			e["accounting"] = map[string]any{"method": accountingUTF8Bytes, "requestOverheadTokens": (int64(1) << 62) - 1}
			return doc(e)
		}, "contextTokens must be"},
		{"defaults naming a model", func() []byte { e := valid(); e["requestOptions"] = map[string]any{"model": "x"}; return doc(e) }, "may not set model"},
		{"no evidence", func() []byte { e := valid(); e["evidence"] = " "; return doc(e) }, "evidence"},
		{"repeated id and revision", func() []byte { return doc(valid(), valid()) }, "repeated"},
		{"trailing data", func() []byte { return append(doc(valid()), []byte(` {}`)...) }, "trailing"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseQualificationDocument(tc.data()); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want one containing %q", err, tc.want)
			}
		})
	}
}

func TestLoadQualificationRegistry(t *testing.T) {
	dir := t.TempDir()
	if reg, err := loadQualificationRegistry(filepath.Join(dir, "missing.json"), os.ReadFile); err != nil || len(reg) != 0 {
		t.Fatalf("missing file: %d entries, %v; want an empty registry", len(reg), err)
	}
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"version":1,"qualifications":[{"id":"q"}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if reg, err := loadQualificationRegistry(bad, os.ReadFile); err == nil || len(reg) != 0 {
		t.Fatalf("invalid file: %d entries, %v; want an empty registry and an error", len(reg), err)
	}
	if reg, err := loadQualificationRegistry(bad, func(string) ([]byte, error) { return nil, errors.New("denied") }); err == nil || len(reg) != 0 {
		t.Fatalf("unreadable file: %d entries, %v", len(reg), err)
	}
}

// The capability is derived from qualification, identity and policy, and
// installed only when every part matches.
func TestStrictCapabilityDerivation(t *testing.T) {
	installed := func(a *admissionController) (nodeStrictModel, bool) {
		a.mu.Lock()
		defer a.mu.Unlock()
		m, ok := a.strict.models[admKey{engine: "ollama", model: "m:latest"}]
		return m, ok
	}
	t.Run("matching inputs install the identity", func(t *testing.T) {
		h := newStrictHarness(t)
		m, ok := installed(h.a)
		want := nodeStrictModel{Engine: "ollama", Model: "m:latest", MaxConcurrent: 1, Identity: strictRequest().Identity}
		if !ok || m != want {
			t.Fatalf("installed = %+v, %v; want %+v", m, ok, want)
		}
	})
	for _, tc := range []struct {
		name string
		edit func(*admissionController)
	}{
		{"launch not qualified (changed context or template flag)", func(a *admissionController) {
			id := testIdentity("m")
			id.LaunchDigest = strings.Repeat("d", 64)
			a.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": id})
		}},
		{"same-name model replaced", func(a *admissionController) {
			id := testIdentity()
			id.ModelDigests["m:latest"] = strings.Repeat("e", 64)
			a.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": id})
		}},
		{"model names disagree on content", func(a *admissionController) {
			id := testIdentity("m")
			id.ModelDigests["m:latest"] = strings.Repeat("e", 64)
			a.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": id})
		}},
		{"settings withdrawn", func(a *admissionController) {
			a.setModelIdentity(map[string]nodepolicy.EngineIdentity{})
		}},
		{"profile serves more context than qualified", func(a *admissionController) {
			p := a.currentPolicy()
			p.Profiles[0].ContextTokens = 8192
			a.setPolicy(p)
		}},
		{"profile allows more output than qualified", func(a *admissionController) {
			p := a.currentPolicy()
			p.Profiles[0].MaxOutputTokens = 2048
			a.setPolicy(p)
		}},
		{"profile names a stale qualification revision", func(a *admissionController) {
			p := a.currentPolicy()
			p.Profiles[0].Qualification = &nodepolicy.QualificationRef{ID: p.Profiles[0].Qualification.ID, Revision: "2"}
			a.setPolicy(p)
		}},
		{"profile defaults differ from the qualified ones", func(a *admissionController) {
			p := a.currentPolicy()
			p.Profiles[0].RequestOptions = json.RawMessage(`{"options":{"num_ctx":8192}}`)
			a.setPolicy(p)
		}},
		{"profile no longer qualified", func(a *admissionController) {
			p := a.currentPolicy()
			p.Profiles[0].Qualification = nil
			a.setPolicy(p)
		}},
		{"two qualified profiles name the model", func(a *admissionController) {
			p := a.currentPolicy()
			dup := p.Profiles[0]
			dup.Name, dup.Model = "dup", "m:latest"
			p.Profiles = append(p.Profiles, dup)
			a.setPolicy(p)
		}},
		{"an unqualified profile names the model after it", func(a *admissionController) {
			p := a.currentPolicy()
			p.Profiles = append(p.Profiles, nodepolicy.Profile{Name: "dup", Engine: "ollama", Model: "m:latest", MaxConcurrent: 3})
			a.setPolicy(p)
		}},
		{"an unqualified profile names the model before it", func(a *admissionController) {
			p := a.currentPolicy()
			p.Profiles = append([]nodepolicy.Profile{{Name: "dup", Engine: "ollama", Model: "m:latest", MaxConcurrent: 3}}, p.Profiles...)
			a.setPolicy(p)
		}},
		{"unbounded concurrency", func(a *admissionController) {
			p := a.currentPolicy()
			p.Admission.MaxConcurrentPerModel = 0
			a.setPolicy(p)
		}},
		{"concurrency beyond the contract", func(a *admissionController) {
			p := a.currentPolicy()
			p.Admission.MaxConcurrentPerModel = 1048577
			a.setPolicy(p)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newStrictHarness(t)
			tc.edit(h.a)
			if m, ok := installed(h.a); ok {
				t.Fatalf("installed %+v", m)
			}
			if _, err := h.a.qualifyStrict("ollama", "m", routeChatCompletions, []byte(strictBody("m")), strictRequest().Identity); !errors.Is(err, errStrictUnavailable) {
				t.Fatalf("qualifyStrict = %v, want unavailable", err)
			}
		})
	}
	t.Run("an unrelated identity change keeps the installation", func(t *testing.T) {
		h := newStrictHarness(t)
		before := h.a.strict.generation[admKey{engine: "ollama", model: "m:latest"}]
		ticket := mustStrictTicket(t, h.a)
		h.a.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": testIdentity("m"), "lmstudio": testIdentity("other")})
		if after := h.a.strict.generation[admKey{engine: "ollama", model: "m:latest"}]; after != before {
			t.Fatalf("generation %d -> %d after another engine's identity changed", before, after)
		}
		if err := ticket.BeforeStrictDispatch(); err != nil {
			t.Fatalf("held ticket fenced by an unrelated change: %v", err)
		}
		ticket.Finish(strictQualifiedComplete)
	})
	t.Run("a changed effective concurrency reinstalls under it", func(t *testing.T) {
		h := newStrictHarness(t)
		ticket := mustStrictTicket(t, h.a)
		p := h.a.currentPolicy()
		p.Admission.MaxConcurrentPerModel = 2
		h.a.setPolicy(p)
		if m, ok := installed(h.a); !ok || m.MaxConcurrent != 2 {
			t.Fatalf("installed %+v, %v; want concurrency 2", m, ok)
		}
		if err := ticket.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
			t.Fatalf("ticket admitted under the old installation: %v", err)
		}
		ticket.Finish(strictNotDispatched)
	})
}

func TestStrictQualifyThenAdmitIsFencedByReinstallation(t *testing.T) {
	h := newStrictHarness(t)
	validated, err := h.a.qualifyStrict("ollama", "m", routeChatCompletions, []byte(strictBody("m")), strictRequest().Identity)
	if err != nil {
		t.Fatal(err)
	}
	// The qualification changes between validation and admission.
	p := h.a.currentPolicy()
	p.Profiles[0].MaxOutputTokens = 8
	h.a.setPolicy(p)
	if ticket, err := h.a.AdmitStrict(context.Background(), "ollama", "m", strictRequest(), validated.generation); ticket != nil || !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("ticket %v, error %v; want unavailable", ticket, err)
	}
	if h.a.currentActive() != 0 || len(h.notes(nodepolicy.NotifyAdmissionWake)) != 0 || len(h.notes(nodepolicy.NotifyAdmissionUnload)) != 0 {
		t.Fatal("a fenced admission had side effects")
	}
}

// Every strict request the qualification does not cover is answered before
// wake, unload, slot or inference, on both entry points.
func TestStrictHTTPRejectsUnqualifiedRequestsBeforeSideEffects(t *testing.T) {
	for _, path := range []string{"ingress", "self"} {
		for _, tc := range []struct {
			name   string
			body   string
			setup  func(*Proxy)
			status int
			code   string
		}{
			{name: "no output bound", body: `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, status: http.StatusBadRequest, code: "strict-qualification"},
			{name: "over the context", body: fmt.Sprintf(`{"model":"m","messages":[{"role":"user","content":%q}],"max_tokens":1024}`, strings.Repeat("あ", 1100)), status: http.StatusBadRequest, code: "strict-qualification"},
			{name: "context override", body: `{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"options":{"num_ctx":65536}}`, status: http.StatusBadRequest, code: "strict-qualification"},
			{name: "settings changed", body: strictBody("m"), setup: func(p *Proxy) {
				p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{})
			}, status: http.StatusServiceUnavailable, code: "strict-admission"},
			{name: "stale identity", body: strictBody("m"), setup: func(p *Proxy) {
				id := testIdentity("m")
				id.EngineRunID = "engine-restarted"
				p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{"ollama": id})
			}, status: http.StatusServiceUnavailable, code: "strict-admission"},
		} {
			t.Run(path+"/"+tc.name, func(t *testing.T) {
				profile, _ := profileFor("ollama")
				out := &recordingWriter{}
				p := newTestProxy(profile, NewCodec(out), NewDiscovery(), profile.FacadePort)
				configureStrict(t, p.admission, filepath.Join(t.TempDir(), "capacity.json"))
				e := newRecordingEngine(t, http.StatusOK)
				// A cold model on a stopped engine with start on demand: any
				// admission would ask for a wake.
				p.admission.setResidency(map[string][]string{})
				p.admission.setIntent(map[string]bool{"ollama": true})
				setBackend(t, p.soleFacade(), e.URL, false)
				if tc.setup != nil {
					tc.setup(p)
				}
				w := httptest.NewRecorder()
				serveStrictHTTP(p, path, w, strictHTTPRequestWith(t, tc.body))
				var answer struct {
					Code string `json:"code"`
				}
				_ = json.Unmarshal(w.Body.Bytes(), &answer)
				if w.Code != tc.status || answer.Code != tc.code {
					t.Fatalf("status %d code %q body %s, want %d %q", w.Code, answer.Code, w.Body, tc.status, tc.code)
				}
				if e.hits() != 0 || p.admission.currentActive() != 0 || len(p.admission.tickets) != 0 || len(p.admission.strict.markers) != 0 {
					t.Fatalf("side effects: hits %d active %d tickets %d markers %d", e.hits(), p.admission.currentActive(), len(p.admission.tickets), len(p.admission.strict.markers))
				}
				for _, line := range out.lines() {
					var msg struct {
						Method string `json:"method"`
					}
					_ = json.Unmarshal(line, &msg)
					if msg.Method == nodepolicy.NotifyAdmissionWake || msg.Method == nodepolicy.NotifyAdmissionUnload {
						t.Fatalf("sent %s for a rejected request", msg.Method)
					}
				}
			})
		}
	}
}

func TestStrictHTTPForwardsTheEffectiveBody(t *testing.T) {
	for _, path := range []string{"ingress", "self"} {
		t.Run(path, func(t *testing.T) {
			p, e := strictHTTPFixture(t)
			pol := p.admission.currentPolicy()
			pol.Profiles[0].RequestOptions = json.RawMessage(`{"keep_alive":"5m"}`)
			q := p.admission.strict.registry[*pol.Profiles[0].Qualification]
			q.RequestOptions = pol.Profiles[0].RequestOptions
			p.admission.setPolicy(pol)
			w := httptest.NewRecorder()
			serveStrictHTTP(p, path, w, strictHTTPRequestWith(t, strictBody("m")))
			if w.Code != http.StatusOK || e.hits() != 1 {
				t.Fatalf("status %d hits %d body %s", w.Code, e.hits(), w.Body)
			}
			if got := e.lastBody(); !strings.Contains(got, `"keep_alive":"5m"`) || !strings.Contains(got, `"max_tokens":16`) {
				t.Fatalf("engine received %s, want the request with the profile's defaults", got)
			}
		})
	}
}
