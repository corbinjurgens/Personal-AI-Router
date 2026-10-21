// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

// testLaunchDigest is the launch every test qualification covers.
var testLaunchDigest = strings.Repeat("c", 64)

// testQualification qualifies engine and model under the identity
// strictRequest names.
func testQualification(id, engine, model string) *localQualification {
	tools, replay := true, false
	return &localQualification{
		ID: id, Revision: "1", Engine: engine, Model: model,
		ModelDigests: []string{strictRequest().Identity.ModelDigest}, LaunchDigests: []string{testLaunchDigest},
		Tokenizer: "test-tokenizer", ChatTemplate: "test-template", ContextTokens: 4096, MaxOutputTokens: 1024,
		Routes:     []string{routeChatCompletions},
		Shapes:     &qualificationShapes{Tools: &tools, StructuredOutput: []string{}, ReasoningReplay: &replay},
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

// Exercise the same shapes supplied by the caller and by profile defaults.
// Every expected bound uses decoded UTF-8 lengths or independently compact
// synthetic JSON literals, never the production accounting helpers.
func TestQualificationShapeMatrix(t *testing.T) {
	testQualificationShapeCases(t, false)
}

func testQualificationShapeCases(t *testing.T, defaults bool) {
	t.Helper()
	const user = `"messages":[{"role":"user","content":"hi"}]`
	const tool = `{"type":"function","function":{"name":"f"}}`
	const call = `{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}`
	const named = `{"type":"function","function":{"name":"f"}}`
	const schema = `{"type":"object"}`
	const format = `{"type":"json_schema","json_schema":{"name":"s","schema":` + schema + `}}`
	definitions := user + `,"tools":[` + tool + `]`
	history := `"messages":[{"role":"assistant","content":null,"tool_calls":[` + call + `]}]`
	for _, tc := range []struct {
		name, fields, gate string
		input              int
	}{
		{"plain text", user, "", 8 + 4 + 4 + 2},
		{"empty tools", user + `,"tools":[]`, "", 18},
		{"none without tools", user + `,"tool_choice":"none"`, "", 18 + 4},
		{"auto without tools", user + `,"tool_choice":"auto"`, "", 18 + 4},
		{"none with empty tools", user + `,"tools":[],"tool_choice":"none"`, "", 18 + 4},
		{"auto with empty tools", user + `,"tools":[],"tool_choice":"auto"`, "", 18 + 4},
		{"definitions with optional parameters omitted", definitions, "tools", 18 + 16 + len(tool)},
		{"definitions with object parameters", user + `,"tools":[{"type":"function","function":{"name":"f","parameters":{}}}]`, "tools", 18 + 16 + len(`{"type":"function","function":{"name":"f","parameters":{}}}`)},
		{"required choice", definitions + `,"tool_choice":"required"`, "tools", 18 + 16 + len(tool) + 8},
		{"named choice", definitions + `,"tool_choice":` + named, "tools", 18 + 16 + len(tool) + len(named)},
		{"assistant calls without definitions", history, "tools", 8 + 4 + 9 + len(call)},
		{"assistant calls with empty definitions", history + `,"tools":[]`, "tools", 8 + 4 + 9 + len(call)},
		{"empty assistant calls", `"messages":[{"role":"assistant","content":"ok","tool_calls":[]}]`, "tools", 8 + 4 + 9 + 2},
		{"tool result without definitions", `"messages":[{"role":"tool","tool_call_id":"c","content":"{}"}]`, "tools", 8 + 4 + 4 + 1 + 2},
		{"native history with empty definitions", `"messages":[{"role":"assistant","content":null,"tool_calls":[` + call + `]},{"role":"tool","tool_call_id":"c","content":"{}"}],"tools":[]`, "tools", 8 + 4 + 9 + len(call) + 4 + 4 + 1 + 2},
		{"text format", user + `,"response_format":{"type":"text"}`, "", 18},
		{"object format", user + `,"response_format":{"type":"json_object"}`, "json_object", 18 + 16},
		{"schema format", user + `,"response_format":` + format, "json_schema", 18 + 16 + 1 + len(schema)},
		{"UTF8 and escaped text", `"messages":[{"role":"user","content":"日\u672c\n"}]`, "", 8 + 4 + 4 + len("日本\n")},
		{"streaming usage", user + `,"stream":true,"stream_options":{"include_usage":true}`, "", 18},
	} {
		t.Run(tc.name, func(t *testing.T) {
			run := func(t *testing.T, qualified bool, modes []string) {
				t.Helper()
				q, profile, _ := qualificationFixture(t)
				tools := qualified && tc.gate == "tools"
				q.Shapes.Tools = &tools
				q.Shapes.StructuredOutput = modes
				fields := `"max_completion_tokens":16,` + tc.fields
				want := `{"model":"m",` + fields + `}`
				body := []byte(want)
				profile.RequestOptions = nil
				if defaults {
					body = []byte(`{"model":"m"}`)
					profile.RequestOptions = json.RawMessage(`{` + fields + `}`)
				}
				q.RequestOptions = profile.RequestOptions
				effective, bound, err := q.validateRequest(profile, routeChatCompletions, body)
				if tc.gate != "" && !qualified {
					var invalid *strictQualificationError
					if !errors.As(err, &invalid) || !strings.Contains(invalid.reason, "not qualified") {
						t.Fatalf("error = %v, want shape qualification denial", err)
					}
					return
				}
				if err != nil || bound != (strictBound{input: tc.input, output: 16}) {
					t.Fatalf("bound = %+v, error = %v; want %d input and 16 output", bound, err, tc.input)
				}
				var gotFields, wantFields map[string]json.RawMessage
				if err := json.Unmarshal(effective, &gotFields); err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal([]byte(want), &wantFields); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(gotFields, wantFields) || (!defaults && !bytes.Equal(effective, body)) {
					t.Fatalf("effective body = %s, want unchanged fields from %s", effective, want)
				}
			}
			modes := []string{}
			if tc.gate == "json_object" || tc.gate == "json_schema" {
				modes = []string{tc.gate}
			}
			t.Run("qualified", func(t *testing.T) { run(t, true, modes) })
			if tc.gate != "" {
				t.Run("undeclared", func(t *testing.T) { run(t, false, []string{}) })
			}
			if tc.gate == "json_object" {
				t.Run("schema evidence does not cover object", func(t *testing.T) { run(t, false, []string{"json_schema"}) })
			}
			if tc.gate == "json_schema" {
				t.Run("object evidence does not cover schema", func(t *testing.T) { run(t, false, []string{"json_object"}) })
			}
		})
	}

	// Malformed shapes are refused even with tools and both JSON modes covered.
	reject := func(name, fields, reason string) {
		t.Run(name, func(t *testing.T) {
			q, profile, _ := qualificationFixture(t)
			fields = `"max_completion_tokens":16,` + fields
			body := []byte(`{"model":"m",` + fields + `}`)
			profile.RequestOptions = nil
			if defaults {
				body = []byte(`{"model":"m"}`)
				profile.RequestOptions = json.RawMessage(`{` + fields + `}`)
			}
			q.RequestOptions = profile.RequestOptions
			_, _, err := q.validateRequest(profile, routeChatCompletions, body)
			var invalid *strictQualificationError
			if !errors.As(err, &invalid) || !strings.Contains(invalid.reason, reason) {
				t.Fatalf("error = %v, want qualification denial containing %q", err, reason)
			}
		})
	}
	reject("null tools", user+`,"tools":null`, "tools must be a non-null array")
	reject("object tools", user+`,"tools":{}`, "tools must be an array")
	reject("null definition", user+`,"tools":[null]`, "named function")
	reject("unnamed definition", user+`,"tools":[{"type":"function","function":{}}]`, "named function")
	reject("null parameters", user+`,"tools":[{"type":"function","function":{"name":"f","parameters":null}}]`, "parameters must be an object")
	reject("array parameters", user+`,"tools":[{"type":"function","function":{"name":"f","parameters":[]}}]`, "parameters must be an object")
	reject("null choice", user+`,"tool_choice":null`, "tool_choice is not supported")
	reject("unknown choice", user+`,"tool_choice":"any"`, "tool_choice is not supported")
	reject("array choice", user+`,"tool_choice":[]`, "tool_choice is not supported")
	reject("unnamed choice", definitions+`,"tool_choice":{"type":"function","function":{}}`, "tool_choice is not supported")
	reject("required without definitions", user+`,"tool_choice":"required"`, "tool_choice requires nonempty tools")
	reject("required with empty definitions", user+`,"tools":[],"tool_choice":"required"`, "tool_choice requires nonempty tools")
	reject("named without definitions", user+`,"tool_choice":`+named, "tool_choice requires nonempty tools")
	reject("named with empty definitions", user+`,"tools":[],"tool_choice":`+named, "tool_choice requires nonempty tools")
	reject("null calls", `"messages":[{"role":"assistant","content":"ok","tool_calls":null}]`, "tool_calls must be a non-null array")
	reject("object calls", `"messages":[{"role":"assistant","content":"ok","tool_calls":{}}]`, "unsupported shape")
	reject("null tool id", `"messages":[{"role":"tool","content":"ok","tool_call_id":null}]`, "tool_call_id must be a string")
	reject("wrong type tool id", `"messages":[{"role":"tool","content":"ok","tool_call_id":3}]`, "unsupported shape")
	reject("calls on user", `"messages":[{"role":"user","content":"ok","tool_calls":[`+call+`]}]`, "belong to an assistant")
	reject("empty calls on user", `"messages":[{"role":"user","content":"ok","tool_calls":[]}]`, "belong to an assistant")
	reject("tool id on assistant", `"messages":[{"role":"assistant","content":"ok","tool_call_id":"c"}]`, "belongs to a tool")
	reject("unnamed call", `"messages":[{"role":"assistant","content":null,"tool_calls":[{"type":"function","function":{}}]}]`, "named function")
	reject("assistant null without calls", `"messages":[{"role":"assistant","content":null}]`, "content is required")
	reject("assistant null with empty calls", `"messages":[{"role":"assistant","content":null,"tool_calls":[]}]`, "content is required")
	reject("user null with calls", `"messages":[{"role":"user","content":null,"tool_calls":[`+call+`]}]`, "belong to an assistant")
	reject("null format", user+`,"response_format":null`, "response_format")
	reject("array format", user+`,"response_format":[]`, "unsupported shape")
	reject("unknown format", user+`,"response_format":{"type":"xml"}`, "response_format is not supported")
	reject("null schema", user+`,"response_format":{"type":"json_schema","json_schema":null}`, "response_format is not supported")
	reject("array schema", user+`,"response_format":{"type":"json_schema","json_schema":{"name":"s","schema":[]}}`, "response_format is not supported")
	reject("wrong type format name", user+`,"response_format":{"type":3}`, "unsupported shape")
	for _, choice := range []string{`"required"`, named} {
		t.Run("requiring choice without tools evidence "+choice, func(t *testing.T) {
			q, profile, _ := qualificationFixture(t)
			tools := false
			q.Shapes.Tools = &tools
			profile.RequestOptions = nil
			fields := `"max_completion_tokens":16,` + user + `,"tool_choice":` + choice
			body := []byte(`{"model":"m",` + fields + `}`)
			if defaults {
				body = []byte(`{"model":"m"}`)
				profile.RequestOptions = json.RawMessage(`{` + fields + `}`)
			}
			q.RequestOptions = profile.RequestOptions
			_, _, err := q.validateRequest(profile, routeChatCompletions, body)
			var invalid *strictQualificationError
			if !errors.As(err, &invalid) || invalid.reason != "tool_choice is not qualified by shapes.tools" {
				t.Fatalf("error = %v, want tool_choice evidence denial", err)
			}
		})
	}
	for _, key := range []string{"reasoning_effort", "parallel_tool_calls", "unrelated_unknown", "options", "keep_alive"} {
		for _, value := range []string{"null", "false", `"high"`, "{}"} {
			reject(key+" "+value, user+`,"`+key+`":`+value, `field "`+key+`" is not supported`)
		}
	}
	// Replay stays refused for every role/value while the owner decision is pending.
	for _, role := range []string{"assistant", "user", "tool"} {
		for _, value := range []string{`""`, `"日本"`, "null", "3"} {
			reject("replay refused "+role+" "+value, `"messages":[{"role":"`+role+`","content":"ok","reasoning_content":`+value+`}]`, "unsupported shape")
		}
	}
}

func TestQualificationShapeDefaultsAndBounds(t *testing.T) {
	t.Run("default shapes", func(t *testing.T) { testQualificationShapeCases(t, true) })
	// Compact tool JSON, schema JSON, and choice JSON are charged byte for byte;
	// content and schema names/descriptions use their decoded UTF-8 byte lengths.
	const tool = `{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}`
	const call = `{"id":"c","type":"function","function":{"name":"f","arguments":"{\"日\":1}"}}`
	const choice = `{"type":"function","function":{"name":"f"}}`
	const schema = `{"type":"object","properties":{"日":{"type":"string"}}}`
	const defaults = `{"tools":[ ` + tool + ` ],"tool_choice":` + choice + `,"response_format":{"type":"json_schema","json_schema":{"name":"答","description":"説","schema":` + schema + `}},"max_completion_tokens":16}`
	const body = `{"model":"m","messages":[{"role":"user","content":"日\n"},{"role":"assistant","content":null,"tool_calls":[ ` + call + ` ]},{"role":"tool","tool_call_id":"c","content":"{}"}]}`
	// Request + three roles/messages + text/id + call + definition/choice/schema.
	const input = 8 + (4 + 4 + len("日\n")) + (4 + 9 + len(call)) + (4 + 4 + 1 + 2) + 16 + len(tool) + len(choice) + 16 + len("答") + len("説") + len(schema)
	for _, tc := range []struct {
		name           string
		context, limit int
		extra          string
		wantError      string
		extraByte      bool
	}{
		{"exact context", input + 16, 16, "", "", false},
		{"one byte over context", input + 16, 16, "", "exceed the qualified context", true},
		{"output ceiling", 4096, 15, "", "exceeds the qualified ceiling", false},
		{"equal maximum aliases", input + 16, 16, `,"max_tokens":16`, "", false},
		{"unequal maximum aliases", 4096, 32, `,"max_tokens":17`, "disagree", false},
		{"raw store false", input + 16, 16, `,"store":false`, "", false},
		{"raw store true", input + 16, 16, `,"store":true`, "PAIR does not store requests", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q, profile, _ := qualificationFixture(t)
			profile.RequestOptions = json.RawMessage(defaults)
			q.RequestOptions = profile.RequestOptions
			q.ContextTokens, profile.ContextTokens = tc.context, tc.context
			q.MaxOutputTokens, profile.MaxOutputTokens = tc.limit, tc.limit
			request := strings.TrimSuffix(body, "}") + tc.extra + "}"
			if tc.extraByte {
				request = strings.Replace(request, `日\n`, `日\nx`, 1)
			}
			effective, bound, err := q.validateRequest(profile, routeChatCompletions, []byte(request))
			if tc.wantError != "" {
				var invalid *strictQualificationError
				if !errors.As(err, &invalid) || !strings.Contains(invalid.reason, tc.wantError) {
					t.Fatalf("error = %v, want %q", err, tc.wantError)
				}
				return
			}
			if err != nil || bound != (strictBound{input: input, output: 16}) {
				t.Fatalf("bound = %+v, error = %v; want %d input and 16 output", bound, err, input)
			}
			var got map[string]json.RawMessage
			if err := json.Unmarshal(effective, &got); err != nil {
				t.Fatal(err)
			}
			if _, exists := got["store"]; exists {
				t.Fatal("store forwarded")
			}
		})
	}
	t.Run("store request and default controls", TestQualificationStoreNoOp)
}

func TestQualificationBorrowedViewsPreserveScalarAndStructDecoding(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	profile.RequestOptions = nil
	for _, tc := range []struct {
		name, message, extras, reject string
	}{
		{name: "nested field case folding", message: `{"RoLe":"user","CoNtEnT":"hi"}`},
		{name: "optional scalar null zero values", message: `{"role":"user","content":"hi","name":null}`, extras: `,"stream":null,"stream_options":null,"stop":[null,"x"],"seed":null,"temperature":null`},
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
	if string(got["seed"]) != "7" || string(got["frequency_penalty"]) != "0" {
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

func TestQualificationStoreNoOp(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	const omitted = `{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true,"stream_options":{"include_usage":true},"temperature":0.25,"stop":["done"]}`
	const defaults = `{"seed":7,"frequency_penalty":0}`
	const merged = `{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true,"stream_options":{"include_usage":true},"temperature":0.25,"stop":["done"],"seed":7,"frequency_penalty":0}`
	decode := func(t *testing.T, raw []byte) map[string]json.RawMessage {
		t.Helper()
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			t.Fatalf("decode forwarding body: %v", err)
		}
		return fields
	}
	accept := func(name, raw, requestDefaults, controlDefaults, want string, exact bool) {
		t.Run(name, func(t *testing.T) {
			p := profile
			p.RequestOptions = json.RawMessage(requestDefaults)
			got, bound, err := q.validateRequest(p, routeChatCompletions, []byte(raw))
			if err != nil {
				t.Fatalf("request rejected: %v", err)
			}
			control := profile
			control.RequestOptions = json.RawMessage(controlDefaults)
			controlBody, controlBound, err := q.validateRequest(control, routeChatCompletions, []byte(omitted))
			if err != nil {
				t.Fatalf("independent omitted-store control rejected: %v", err)
			}
			if bound != controlBound || bound.input != 18 || bound.output != 16 {
				t.Fatalf("bound = %+v, omitted-store control = %+v; want 18 input and 16 output", bound, controlBound)
			}
			fields := decode(t, got)
			if _, present := fields["store"]; present {
				t.Fatal("forwarding body retains store")
			}
			wantFields := decode(t, []byte(want))
			if !reflect.DeepEqual(fields, wantFields) || !reflect.DeepEqual(fields, decode(t, controlBody)) {
				t.Fatalf("forwarded fields = %s; want every field in %s and the omitted-store control", got, want)
			}
			if exact && !bytes.Equal(got, []byte(want)) {
				t.Fatalf("absent-store bytes = %q, want %q", got, want)
			}
		})
	}
	withFalse := strings.TrimSuffix(omitted, "}") + `,"store":false}`
	accept("raw false stripped after merging defaults", withFalse, defaults, defaults, merged, false)
	accept("first member false stripped", ` { "store" : false ,`+omitted[1:]+" ", "", "", omitted, false)
	accept("middle member false stripped", strings.Replace(omitted, `,"max_tokens"`, `,"store":false,"max_tokens"`, 1), "", "", omitted, false)
	accept("escaped store key stripped", strings.Replace(withFalse, `"store"`, `"\u0073tore"`, 1), "", "", omitted, false)
	accept("absent preserves exact bytes", " \n"+omitted+"\t ", "", "", " \n"+omitted+"\t ", true)
	accept("absent preserves merged bytes", omitted, defaults, defaults, merged, true)
	accept("default false stripped", omitted, `{"store":false,"seed":7,"frequency_penalty":0}`, defaults, merged, false)
	accept("raw false overrides default true before stripping", withFalse, `{"store":true,"seed":7,"frequency_penalty":0}`, defaults, merged, false)
	accept("raw false overrides default null before stripping", withFalse, `{"store":null,"seed":7,"frequency_penalty":0}`, defaults, merged, false)
	reject := func(name, raw, requestDefaults, want string) {
		t.Run(name, func(t *testing.T) {
			p := profile
			p.RequestOptions = json.RawMessage(requestDefaults)
			_, _, err := q.validateRequest(p, routeChatCompletions, []byte(raw))
			var invalid *strictQualificationError
			if !errors.As(err, &invalid) || !strings.Contains(invalid.reason, want) {
				t.Fatalf("error = %v; want qualification refusal containing %q", err, want)
			}
		})
	}
	for _, tc := range []struct{ name, value string }{
		{"true", "true"},
		{"null", "null"},
		{"string", `"false"`},
		{"number", "0"},
		{"array", "[]"},
		{"object", "{}"},
	} {
		want := "store must be the JSON boolean false"
		if tc.value == "true" {
			want = "PAIR does not store requests"
		}
		reject("raw "+tc.name+" refused", strings.TrimSuffix(omitted, "}")+`,"store":`+tc.value+`}`, "", want)
		reject("default "+tc.name+" refused", omitted, `{"store":`+tc.value+`}`, want)
	}
	reject("raw true refused before default false", strings.TrimSuffix(omitted, "}")+`,"store":true}`, `{"store":false}`, "PAIR does not store requests")
	reject("duplicate raw store refused before stripping", strings.TrimSuffix(withFalse, "}")+`,"store":false}`, "", "repeats a key")
	reject("escaped duplicate store refused before stripping", strings.TrimSuffix(withFalse, "}")+`,"\u0073tore":false}`, "", "repeats a key")
	reject("duplicate default store refused before stripping", omitted, `{"store":false,"store":false}`, "repeats a key")
	reject("other incoming unsupported field still refused", strings.TrimSuffix(withFalse, "}")+`,"unqualified":false}`, "", `field "unqualified" is not supported`)
	t.Run("forwarding copy owns its measured storage", func(t *testing.T) {
		p := profile
		p.RequestOptions = nil
		control := strings.Replace(strictBody("fixture-model"), `"hi"`, `"`+strings.Repeat("a", 2048)+`"`, 1)
		withStore := strings.TrimSuffix(control, "}") + `,"store":false}`
		charge := int64(len(control)) + 256
		for _, tc := range []struct {
			name, body string
			capacity   int64
			retained   int64
			limit      bool
		}{
			{name: "absent borrows without another grant", body: control, capacity: 1 << 20},
			{name: "false retains exact forwarding capacity", body: withStore, capacity: 1 << 20, retained: charge},
			{name: "copy footprint refused before allocation", body: withStore, capacity: charge - 1, limit: true},
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
				forwarded, bound, err := q.validateRequest(p, routeChatCompletions, []byte(tc.body), ctx)
				if tc.limit {
					bufferAssertLimit(t, err)
					if forwarded != nil || bound != (strictBound{}) {
						t.Fatalf("refused copy returned body or bounds: %d bytes, %+v", len(forwarded), bound)
					}
				} else {
					if err != nil || bound.input != 2064 || bound.output != 16 || string(forwarded) != control {
						t.Fatalf("forwarding bytes preserved = %v, bound = %+v, error = %v", string(forwarded) == control, bound, err)
					}
					if tc.retained > 0 && int64(cap(forwarded))+256 != tc.retained {
						t.Fatalf("forwarding capacity = %d; want measured charge %d", cap(forwarded), tc.retained)
					}
				}
				live := 0
				if tc.retained > 0 {
					live = 1
				}
				bufferAssertTotals(t, budget, tc.retained, live, 0)
				owner.close()
				bufferAssertTotals(t, budget, 0, 0, 0)
			})
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
			"shapes":     map[string]any{"tools": true, "structuredOutput": []string{}, "reasoningReplay": false},
			"accounting": map[string]any{"method": accountingUTF8Bytes, "requestOverheadTokens": 1, "messageOverheadTokens": 1, "toolOverheadTokens": 1, "schemaOverheadTokens": 1},
			"evidence":   "measured",
		}
	}
	doc := func(entries ...map[string]any) []byte {
		data, err := json.Marshal(map[string]any{"version": qualificationDocumentVersion, "qualifications": entries})
		if err != nil {
			t.Fatal(err)
		}
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
		{"another version", func() []byte { return []byte(`{"version":3,"qualifications":[]}`) }, "version"},
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

func TestQualificationShapesDocument(t *testing.T) {
	if qualificationDocumentVersion != 2 {
		t.Fatalf("artifact version = %d, want coordinated revision 2", qualificationDocumentVersion)
	}
	document := func(t *testing.T, version int, shapes string) []byte {
		t.Helper()
		data, err := json.Marshal(testQualification("q", "ollama", "m"))
		if err != nil {
			t.Fatal(err)
		}
		var entry map[string]json.RawMessage
		if err := json.Unmarshal(data, &entry); err != nil {
			t.Fatal(err)
		}
		delete(entry, "shapes")
		if shapes != "" {
			entry["shapes"] = json.RawMessage(shapes)
		}
		data, err = json.Marshal(map[string]any{"version": version, "qualifications": []map[string]json.RawMessage{entry}})
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	accept := func(name, shapes string, tools bool, modes []string) {
		t.Run(name, func(t *testing.T) {
			reg, err := parseQualificationDocument(document(t, 2, shapes))
			if err != nil {
				t.Fatal(err)
			}
			q, ok := reg.lookup(nodepolicy.QualificationRef{ID: "q", Revision: "1"})
			if !ok || q.Shapes == nil || q.Shapes.Tools == nil || q.Shapes.ReasoningReplay == nil {
				t.Fatal("parsed qualification lacks explicit shape evidence")
			}
			if *q.Shapes.Tools != tools || *q.Shapes.ReasoningReplay || !reflect.DeepEqual(q.Shapes.StructuredOutput, modes) {
				t.Fatalf("shapes = %+v, want tools %v, modes %v and replay false", q.Shapes, tools, modes)
			}
			// NE produces these opaque canonical digests; parsing must preserve
			// them without adding another provenance acceptance field.
			want := testQualification("q", "ollama", "m")
			if !reflect.DeepEqual(q.ModelDigests, want.ModelDigests) || !reflect.DeepEqual(q.LaunchDigests, want.LaunchDigests) ||
				q.Tokenizer != want.Tokenizer || q.ChatTemplate != want.ChatTemplate || q.ContextTokens != want.ContextTokens || q.MaxOutputTokens != want.MaxOutputTokens {
				t.Fatalf("identity or bounds changed: %+v", q)
			}
		})
	}
	accept("text only", `{"tools":false,"structuredOutput":[],"reasoningReplay":false}`, false, []string{})
	accept("tools only", `{"tools":true,"structuredOutput":[],"reasoningReplay":false}`, true, []string{})
	accept("json object grants only that mode", `{"tools":false,"structuredOutput":["json_object"],"reasoningReplay":false}`, false, []string{"json_object"})
	accept("json schema grants only that mode", `{"tools":false,"structuredOutput":["json_schema"],"reasoningReplay":false}`, false, []string{"json_schema"})
	accept("both modes and NE identity", `{"tools":true,"structuredOutput":["json_schema","json_object"],"reasoningReplay":false}`, true, []string{"json_schema", "json_object"})

	reject := func(name string, version int, shapes, want string) {
		t.Run(name, func(t *testing.T) {
			reg, err := parseQualificationDocument(document(t, version, shapes))
			if err == nil || !strings.Contains(err.Error(), want) || len(reg) != 0 {
				t.Fatalf("registry = %v, error = %v; want empty registry and %q", reg, err, want)
			}
		})
	}
	reject("old evidence", 1, "", "version must be 2")
	reject("future version", 3, `{"tools":false,"structuredOutput":[],"reasoningReplay":false}`, "version must be 2")
	reject("missing shapes", 2, "", "shapes must be a non-null object")
	reject("null shapes", 2, `null`, "shapes must be a non-null object")
	reject("array shapes", 2, `[]`, "cannot unmarshal")
	reject("boolean shapes", 2, `false`, "cannot unmarshal")
	reject("string shapes", 2, `"tools"`, "cannot unmarshal")
	reject("required members checked in order", 2, `{}`, "shapes.tools must be")
	reject("missing tools", 2, `{"structuredOutput":[],"reasoningReplay":false}`, "shapes.tools must be")
	reject("null tools", 2, `{"tools":null,"structuredOutput":[],"reasoningReplay":false}`, "shapes.tools must be")
	reject("string tools", 2, `{"tools":"false","structuredOutput":[],"reasoningReplay":false}`, "cannot unmarshal")
	reject("number tools", 2, `{"tools":0,"structuredOutput":[],"reasoningReplay":false}`, "cannot unmarshal")
	reject("missing modes and replay", 2, `{"tools":false}`, "shapes.structuredOutput must be")
	reject("null modes", 2, `{"tools":false,"structuredOutput":null,"reasoningReplay":false}`, "shapes.structuredOutput must be")
	reject("string modes", 2, `{"tools":false,"structuredOutput":"json_object","reasoningReplay":false}`, "cannot unmarshal")
	reject("object modes", 2, `{"tools":false,"structuredOutput":{},"reasoningReplay":false}`, "cannot unmarshal")
	reject("unknown mode", 2, `{"tools":false,"structuredOutput":["text"],"reasoningReplay":false}`, "unsupported shapes.structuredOutput mode")
	reject("null mode", 2, `{"tools":false,"structuredOutput":[null],"reasoningReplay":false}`, "unsupported shapes.structuredOutput mode")
	reject("number mode", 2, `{"tools":false,"structuredOutput":[1],"reasoningReplay":false}`, "cannot unmarshal")
	reject("duplicate mode", 2, `{"tools":false,"structuredOutput":["json_object","json_object"],"reasoningReplay":false}`, "repeated shapes.structuredOutput mode")
	reject("missing replay", 2, `{"tools":false,"structuredOutput":[]}`, "shapes.reasoningReplay must be")
	reject("null replay", 2, `{"tools":false,"structuredOutput":[],"reasoningReplay":null}`, "shapes.reasoningReplay must be")
	reject("string replay", 2, `{"tools":false,"structuredOutput":[],"reasoningReplay":"false"}`, "cannot unmarshal")
	reject("replay acceptance remains forbidden", 2, `{"tools":false,"structuredOutput":[],"reasoningReplay":true}`, "shapes.reasoningReplay must be false")
	reject("unknown shape", 2, `{"tools":false,"structuredOutput":[],"reasoningReplay":false,"vision":false}`, "unknown field")
	reject("effort not evidence", 2, `{"tools":false,"structuredOutput":[],"reasoningReplay":false,"reasoning_effort":false}`, "unknown field")
	reject("parallel not evidence", 2, `{"tools":false,"structuredOutput":[],"reasoningReplay":false,"parallel_tool_calls":false}`, "unknown field")
	reject("duplicate shape member", 2, `{"tools":false,"tools":true,"structuredOutput":[],"reasoningReplay":false}`, "repeated keys")
	reject("escaped duplicate member", 2, `{"tools":false,"to\u006fls":true,"structuredOutput":[],"reasoningReplay":false}`, "repeated keys")
	t.Run("duplicate shapes object", func(t *testing.T) {
		data := document(t, 2, `{"tools":false,"structuredOutput":[],"reasoningReplay":false}`)
		data = bytes.Replace(data, []byte(`"shapes":`), []byte(`"shapes":{},"shapes":`), 1)
		if _, err := parseQualificationDocument(data); err == nil || !strings.Contains(err.Error(), "repeated keys") {
			t.Fatalf("error = %v, want duplicate object rejection", err)
		}
	})
}

func TestLoadQualificationRegistry(t *testing.T) {
	dir := t.TempDir()
	if reg, err := loadQualificationRegistry(filepath.Join(dir, "missing.json"), os.ReadFile); err != nil || len(reg) != 0 {
		t.Fatalf("missing file: %d entries, %v; want an empty registry", len(reg), err)
	}
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"version":2,"qualifications":[{"id":"q"}]}`), 0o600); err != nil {
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
			pol.Profiles[0].RequestOptions = json.RawMessage(`{"temperature":0.5}`)
			q := p.admission.strict.registry[*pol.Profiles[0].Qualification]
			q.RequestOptions = pol.Profiles[0].RequestOptions
			p.admission.setPolicy(pol)
			w := httptest.NewRecorder()
			serveStrictHTTP(p, path, w, strictHTTPRequestWith(t, strictBody("m")))
			if w.Code != http.StatusOK || e.hits() != 1 {
				t.Fatalf("status %d hits %d body %s", w.Code, e.hits(), w.Body)
			}
			var got struct {
				Temperature float64 `json:"temperature"`
				MaxTokens   int     `json:"max_tokens"`
			}
			if err := json.Unmarshal([]byte(e.lastBody()), &got); err != nil {
				t.Fatal(err)
			}
			if got.Temperature != 0.5 || got.MaxTokens != 16 {
				t.Fatalf("engine received %+v, want the request with the profile's defaults", got)
			}
		})
	}
}
