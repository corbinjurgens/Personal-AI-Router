// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

type openCodeCase struct {
	Name, Qualification, Arithmetic, Reject string
	Body                                    json.RawMessage
	InputTokens, OutputTokens               int
}

// The JSON declares only differing shapes. Expand these into complete revision-2
// artifacts with the shared synthetic identity, then use the production parser.
func openCodeFixtures(t *testing.T) (qualificationRegistry, []openCodeCase) {
	t.Helper()
	data, err := os.ReadFile("testdata/opencode-shapes.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Evidence       string
		Sources        []struct{ Host, CLI, Native, Inventory, Conditions string }
		Qualifications []struct {
			ID     string
			Shapes *qualificationShapes
		}
		Cases []openCodeCase
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(fixture.Evidence, "qualifies no real model or template") {
		t.Fatal("fixture must identify its synthetic-only evidence boundary")
	}
	pins := map[string]string{"MSI": "2.1.1/ba546ec1598a70db2570e9c6258028d955ba8f58/dd0b69e3", "PC3": "2.0.32/843a6c514e606b436a7fe713422ace0a93ad3d61/a88a4f7"}
	for _, source := range fixture.Sources {
		if pins[source.Host] != source.CLI+"/"+source.Native+"/"+source.Inventory || source.Conditions == "" {
			t.Fatalf("missing or changed immutable inventory pin for %s", source.Host)
		}
		delete(pins, source.Host)
	}
	if len(pins) != 0 {
		t.Fatal("missing host inventories")
	}
	var doc struct {
		Version        int                   `json:"version"`
		Qualifications []*localQualification `json:"qualifications"`
	}
	doc.Version = qualificationDocumentVersion
	for _, entry := range fixture.Qualifications {
		q := testQualification(entry.ID, "llamacpp", "fixture-model")
		q.Shapes = entry.Shapes
		q.RequestOptions = json.RawMessage(`{"temperature":0.5,"max_completion_tokens":16}`)
		q.Accounting.ToolOverheadTokens, q.Accounting.SchemaOverheadTokens = 16, 16
		q.Evidence = fixture.Evidence
		doc.Qualifications = append(doc.Qualifications, q)
	}
	data, err = json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := parseQualificationDocument(data)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, tc := range fixture.Cases {
		if tc.Name == "" || names[tc.Name] {
			t.Fatalf("empty or duplicate fixture name %q", tc.Name)
		}
		names[tc.Name] = true
		if _, ok := registry.lookup(nodepolicy.QualificationRef{ID: tc.Qualification, Revision: "1"}); !ok {
			t.Fatalf("%s names missing qualification %q", tc.Name, tc.Qualification)
		}
	}
	for _, name := range []string{"plain-text", "pc3-store-false", "pc3-store-true", "maximum-omitted-defaults", "maximum-alias", "native-streaming-usage", "empty-tools", "first-turn-required", "first-turn-auto", "first-turn-none", "first-turn-named", "no-active-tools-omitted", "no-active-tools-none", "tool-roundtrip-no-active-tools", "tool-history-definitions-omitted", "unqualified-definitions", "unqualified-assistant-history", "unqualified-tool-result", "unqualified-required-choice", "unqualified-named-choice", "required-without-definitions", "overlay-json-object", "overlay-json-schema", "unqualified-json-object", "unqualified-json-schema", "replay-with-tool-roundtrip-refused", "empty-replay-refused", "wrong-role-replay-refused", "null-replay-refused", "effort-refused", "overlay-parallel-refused", "image-refused", "provider-option-refused"} {
		if !names[name] {
			t.Fatalf("missing required fixture %s", name)
		}
	}
	return registry, fixture.Cases
}

func openCodeProfile(q *localQualification) nodepolicy.Profile {
	return nodepolicy.Profile{Name: "synthetic-opencode", Engine: q.Engine, Model: q.Model,
		ContextTokens: 4096, MaxOutputTokens: 1024, MemoryBytes: 4096,
		RequestOptions: q.RequestOptions, Qualification: &nodepolicy.QualificationRef{ID: q.ID, Revision: q.Revision}}
}

// This oracle uses encoding/json on decoded fixture fields and json.Compact,
// independently of the validator's borrowed-view parser and byte counters.
func openCodeInputBound(t *testing.T, body []byte) int {
	t.Helper()
	compact := func(raw json.RawMessage) int {
		var buf bytes.Buffer
		if err := json.Compact(&buf, raw); err != nil {
			t.Fatal(err)
		}
		return buf.Len()
	}
	var request struct {
		Messages []struct {
			Role, Name string
			ToolCallID string `json:"tool_call_id"`
			Content    string
			ToolCalls  []json.RawMessage `json:"tool_calls"`
		}
		Tools  []json.RawMessage
		Choice json.RawMessage `json:"tool_choice"`
		Format *struct {
			Type   string
			Schema *struct {
				Name, Description string
				Schema            json.RawMessage
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatal(err)
	}
	n := 8
	for _, message := range request.Messages {
		n += 4 + len(message.Role) + len(message.Name) + len(message.Content) + len(message.ToolCallID)
		for _, call := range message.ToolCalls {
			n += compact(call)
		}
	}
	for _, tool := range request.Tools {
		n += 16 + compact(tool)
	}
	if len(request.Choice) > 0 {
		var choice string
		if request.Choice[0] == '"' {
			if err := json.Unmarshal(request.Choice, &choice); err != nil {
				t.Fatal(err)
			}
			n += len(choice)
		} else {
			n += compact(request.Choice)
		}
	}
	if f := request.Format; f != nil && f.Type != "text" {
		n += 16
		if f.Schema != nil {
			n += len(f.Schema.Name) + len(f.Schema.Description) + compact(f.Schema.Schema)
		}
	}
	return n
}

func openCodeEffective(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatal(err)
	}
	delete(fields, "store")
	if _, ok := fields["temperature"]; !ok {
		fields["temperature"] = json.RawMessage(`0.5`)
	}
	if _, ok := fields["max_completion_tokens"]; !ok {
		fields["max_completion_tokens"] = json.RawMessage(`16`)
	}
	return fields
}

func assertOpenCodeEffective(t *testing.T, got []byte, raw []byte) {
	t.Helper()
	var actual map[string]json.RawMessage
	if err := json.Unmarshal(got, &actual); err != nil {
		t.Fatal(err)
	}
	want := openCodeEffective(t, raw)
	// Decode values to ignore whitespace while checking every field, including
	// tool IDs/arguments/results, empty arrays, maxima and merged defaults.
	decode := func(fields map[string]json.RawMessage) map[string]any {
		result := map[string]any{}
		for key, value := range fields {
			var decoded any
			if err := json.Unmarshal(value, &decoded); err != nil {
				t.Fatal(err)
			}
			result[key] = decoded
		}
		return result
	}
	if !reflect.DeepEqual(decode(actual), decode(want)) {
		t.Fatal("forwarded effective fields changed")
	}
}

func TestOpenCodeShapeFixtures(t *testing.T) {
	registry, cases := openCodeFixtures(t)
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			q := registry[nodepolicy.QualificationRef{ID: tc.Qualification, Revision: "1"}]
			effective, bound, err := q.validateRequest(openCodeProfile(q), routeChatCompletions, tc.Body)
			if tc.Reject != "" {
				var invalid *strictQualificationError
				if !errors.As(err, &invalid) || !strings.Contains(err.Error(), tc.Reject) {
					t.Fatalf("error = %v; want strict qualification containing %q", err, tc.Reject)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if tc.Arithmetic == "" || openCodeInputBound(t, tc.Body) != tc.InputTokens {
				t.Fatal("fixture hand arithmetic disagrees with independent oracle")
			}
			if bound.input != tc.InputTokens || bound.output != tc.OutputTokens {
				t.Fatalf("bound = %+v; want %d/%d", bound, tc.InputTokens, tc.OutputTokens)
			}
			assertOpenCodeEffective(t, effective, tc.Body)
		})
	}
}

// llama.cpp variant of strictHTTPFixture, reusing recordingEngine and the NE
// envelope constructor. The production policy/identity join installs readiness.
func openCodeHTTPFixture(t *testing.T, q *localQualification, warm bool) (*Proxy, *recordingEngine, *recordingWriter, nodepolicy.StrictAdmissionRequest) {
	t.Helper()
	ne := newReadinessEnvelopeFixture(t, "llamacpp")
	req := ne.request()
	envelope := ne.id.Serving["m"]
	envelope.ModelID = q.Model
	ne.id.Serving = map[string]engines.ServingEnvelope{q.Model: envelope}
	ne.id.ModelDigests = map[string]string{q.Model: req.Identity.ModelDigest}
	profile, ok := profileFor("llamacpp")
	if !ok {
		t.Fatal("missing llama.cpp profile")
	}
	out := &recordingWriter{}
	p := newTestProxy(profile, NewCodec(out), NewDiscovery(), profile.FacadePort)
	p.admission.strict.path = ne.a.strict.path
	p.admission.strict.storageErr = nil
	p.admission.strict.registry = qualificationRegistry{*openCodeProfile(q).Qualification: q}
	policy := ne.policy
	policy.Profiles = []nodepolicy.Profile{openCodeProfile(q)}
	policy.Idle.StartOnDemand = true
	if err := p.admission.activatePolicy(policy); err != nil {
		t.Fatal(err)
	}
	p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{"llamacpp": ne.id})
	p.admission.setIntent(map[string]bool{"llamacpp": true})
	residency := map[string][]string{}
	if warm {
		residency["llamacpp"] = []string{q.Model}
	}
	p.admission.setResidency(residency)
	e := openCodeEngine(t, "application/json", openCodeJSONResponse)
	setBackend(t, p.soleFacade(), e.URL, warm)
	if readiness := p.admission.StrictReadiness(); readiness.Capability == nil || len(readiness.Capability.Models) == 0 {
		t.Fatal("synthetic NE-bound model did not install")
	}
	return p, e, out, req
}

func openCodeRequest(t *testing.T, body []byte, strict nodepolicy.StrictAdmissionRequest) *http.Request {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, routeChatCompletions, bytes.NewReader(body))
	header, err := json.Marshal(strict)
	if err != nil {
		t.Fatal(err)
	}
	r.Header.Set(nodepolicy.StrictAdmissionHeader, string(header))
	return r
}

func TestStrictHTTPRefusesUnqualifiedShapesBeforeSideEffects(t *testing.T) {
	registry, cases := openCodeFixtures(t)
	for _, path := range []string{"ingress", "self"} {
		for _, tc := range cases {
			if tc.Reject == "" {
				continue
			}
			t.Run(path+"/"+tc.Name, func(t *testing.T) {
				q := registry[nodepolicy.QualificationRef{ID: tc.Qualification, Revision: "1"}]
				p, e, out, req := openCodeHTTPFixture(t, q, false)
				w := httptest.NewRecorder()
				serveStrictHTTP(p, path, w, openCodeRequest(t, tc.Body, req))
				var answer struct{ Code, Error string }
				if err := json.Unmarshal(w.Body.Bytes(), &answer); err != nil {
					t.Fatal(err)
				}
				if w.Code != http.StatusBadRequest || answer.Code != "strict-qualification" || !strings.Contains(answer.Error, tc.Reject) {
					t.Fatalf("status=%d answer=%+v; want shape refusal %q", w.Code, answer, tc.Reject)
				}
				if e.hits() != 0 || p.admission.currentActive() != 0 || len(p.admission.tickets) != 0 || len(p.admission.strict.markers) != 0 {
					t.Fatal("shape refusal caused inference/admission side effects")
				}
				if _, err := os.Stat(p.admission.strict.path); !os.IsNotExist(err) {
					t.Fatalf("capacity marker file exists or cannot be checked: %v", err)
				}
				for _, line := range out.lines() {
					var message struct{ Method string }
					if err := json.Unmarshal(line, &message); err != nil {
						t.Fatal(err)
					}
					if message.Method == nodepolicy.NotifyAdmissionWake || message.Method == nodepolicy.NotifyAdmissionUnload {
						t.Fatalf("shape refusal sent %s", message.Method)
					}
				}
			})
		}
	}
}

func TestStrictHTTPOpenCodeQualifiedForwarding(t *testing.T) {
	registry, cases := openCodeFixtures(t)
	for _, path := range []string{"ingress", "self"} {
		for _, tc := range cases {
			if tc.Reject != "" || tc.Name == "native-streaming-usage" {
				continue
			}
			t.Run(path+"/"+tc.Name, func(t *testing.T) {
				q := registry[nodepolicy.QualificationRef{ID: tc.Qualification, Revision: "1"}]
				p, e, _, req := openCodeHTTPFixture(t, q, true)
				// Ordinary JSON proves forwarding mechanics, not native streaming.
				var body map[string]json.RawMessage
				if err := json.Unmarshal(tc.Body, &body); err != nil {
					t.Fatal(err)
				}
				body["stream"] = json.RawMessage(`false`)
				raw, err := json.Marshal(body)
				if err != nil {
					t.Fatal(err)
				}
				w := httptest.NewRecorder()
				serveStrictHTTP(p, path, w, openCodeRequest(t, raw, req))
				if w.Code != http.StatusOK || e.hits() != 1 {
					t.Fatalf("status=%d hits=%d", w.Code, e.hits())
				}
				assertOpenCodeEffective(t, []byte(e.lastBody()), raw)
				if w.Body.String() != openCodeJSONResponse || w.Header().Get("Content-Type") != "application/json" || w.Header().Get("X-Synthetic-Fixture") != "opencode" {
					t.Fatal("JSON response/header changed")
				}
			})
		}
	}
	t.Run("native streaming usage then replay refusal", func(t *testing.T) {
		q := registry[nodepolicy.QualificationRef{ID: "tools", Revision: "1"}]
		for _, path := range []string{"ingress", "self"} {
			t.Run(path, func(t *testing.T) { testOpenCodeStreaming(t, q, path) })
		}
	})
}

const openCodeJSONResponse = `{"id":"synthetic","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"completion_tokens":2,"total_tokens":20}}`

// Reuse recordingEngine's synchronized observations with a real Chat response
// or SSE script. Allocate only loopback, check the port before serving/dialing.
func openCodeEngine(t *testing.T, contentType, payload string) *recordingEngine {
	t.Helper()
	e := &recordingEngine{}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		e.mu.Lock()
		e.bodies = append(e.bodies, string(body))
		e.headers = append(e.headers, r.Header.Clone())
		e.mu.Unlock()
		w.Header().Set("Content-Type", contentType)
		w.Header().Set("X-Synthetic-Fixture", "opencode")
		if _, err := io.WriteString(w, payload); err != nil {
			t.Error(err)
		}
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if port == 8080 || port == 11434 || port == 1234 || (port >= 14318 && port <= 14323) {
		if err := ln.Close(); err != nil {
			t.Fatal(err)
		}
		t.Fatal("protected ephemeral port")
	}
	e.Server = &httptest.Server{Listener: ln, Config: &http.Server{Handler: handler}}
	e.Start()
	t.Cleanup(e.Close)
	return e
}

func testOpenCodeStreaming(t *testing.T, q *localQualification, path string) {
	t.Helper()
	p, _, _, req := openCodeHTTPFixture(t, q, true)
	payload := "data: {\"id\":\"synthetic\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"雪\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"synthetic\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"synthetic\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"synthetic\",\"choices\":[],\"usage\":{\"prompt_tokens\":18,\"completion_tokens\":2,\"total_tokens\":20}}\n\n" + "data: [DONE]\n\n"
	e := openCodeEngine(t, "text/event-stream", payload)
	setBackend(t, p.soleFacade(), e.URL, true)
	body := []byte(`{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":16,"stream":true,"stream_options":{"include_usage":true}}`)
	_, bound, err := q.validateRequest(openCodeProfile(q), routeChatCompletions, body)
	if err != nil || bound.input != 18 || bound.output != 16 {
		t.Fatalf("stream bound=%+v error=%v", bound, err)
	}
	w := httptest.NewRecorder()
	serveStrictHTTP(p, path, w, openCodeRequest(t, body, req))
	if w.Code != http.StatusOK || e.hits() != 1 || w.Body.String() != payload || w.Header().Get("Content-Type") != "text/event-stream" || w.Header().Get("X-Synthetic-Fixture") != "opencode" {
		t.Fatalf("SSE status=%d hits=%d response fidelity=%v", w.Code, e.hits(), w.Body.String() == payload)
	}
	assertOpenCodeEffective(t, []byte(e.lastBody()), body)
	// Automatic native replay is deliberately refused under the binding amendment.
	// A successful second inference and replay byte accounting await owner approval.
	replay := []byte(`{"model":"fixture-model","messages":[{"role":"assistant","content":"ok","reasoning_content":"雪"}],"max_completion_tokens":16}`)
	_, replayBound, err := q.validateRequest(openCodeProfile(q), routeChatCompletions, replay)
	if err == nil || replayBound != (strictBound{}) {
		t.Fatal("replay returned accepted bounds")
	}
	w = httptest.NewRecorder()
	req.Request.RequestID = "2"
	serveStrictHTTP(p, path, w, openCodeRequest(t, replay, req))
	var answer struct{ Code string }
	if err := json.Unmarshal(w.Body.Bytes(), &answer); err != nil {
		t.Fatal(err)
	}
	if w.Code != http.StatusBadRequest || answer.Code != "strict-qualification" || e.hits() != 1 {
		t.Fatal("second-turn replay was forwarded")
	}
}
