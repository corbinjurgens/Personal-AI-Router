// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestModelControlsIngress(t *testing.T) {
	test := func(name, target, query, body string, reject bool) {
		t.Run(name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, target, nil)
			r.URL.RawQuery = query
			err := validateRequestIdentifiers(r, []byte(body))
			if reject {
				if err == nil || err.Error() != "invalid model identifier" || bufferRequestStatus(err) != http.StatusBadRequest {
					t.Fatalf("identifier refusal = %v", err)
				}
			} else if err != nil {
				t.Fatalf("valid or opaque content refused: %v", err)
			}
		})
	}
	test("escaped newline", "/api/chat", "", `{"model":"a\nb"}`, true)
	test("escaped folded name", "/api/show", "", `{"N\u0041ME":"a\u0000b"}`, true)
	test("DEL", "/api/chat", "", `{"model":"a\u007fb"}`, true)
	test("C1 lower boundary", "/api/chat", "", `{"model":"a\u0080b"}`, true)
	test("C1 upper boundary", "/api/chat", "", `{"model":"a\u009fb"}`, true)
	test("raw invalid UTF8", "/api/chat", "", "{\"model\":\"a"+string([]byte{0xff})+"b\"}", true)
	test("empty identifier", "/api/show", "", `{"name":""}`, true)
	test("invalid earlier duplicate", "/api/chat", "", `{"model":"a\nb","model":"good"}`, true)
	test("completed invalid prefix", "/api/show", "", `{"name":"a\nb","later":`, true)
	test("escaped query", "/props", "mo%64el=a%0Ab", "", true)
	test("invalid query UTF8", "/props", "model=a%FFb", "", true)
	test("query DEL", "/props", "model=a%7Fb", "", true)
	test("query C1", "/props", "model=a%C2%85b", "", true)
	test("model detail control", "/v1/models/a%0Ab", "", "", true)
	test("native detail C1", "/api/v0/models/a%C2%85b", "", "", true)
	test("model detail UTF8", "/v1/models/a%FFb", "", "", true)
	test("256 byte ingress", "/api/chat", "", `{"model":"`+strings.Repeat("a", 256)+`"}`, false)
	test("257 byte ingress", "/api/chat", "", `{"model":"`+strings.Repeat("a", 257)+`"}`, true)
	test("ordinary Unicode and spaces", "/api/chat", "", `{"model":" 組/model:tag "}`, false)
	test("opaque fields", "/api/chat", "", "{\"model\":\"good\",\"prompt\":\"a\\nb\",\"nested\":{\"model\":\"bad\\n\"},\"opaque\":\""+string([]byte{0xff})+"\"}", false)
	test("invalid percent remains ignored", "/props", "model=bad%zz", "", false)
}

func TestModelControlsIngressRawBuffer(t *testing.T) {
	test := func(name, body string, reject bool) {
		t.Run(name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
			_, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, defaultMaxRequestBytes)
			if reject {
				if err == nil || err.Error() != "invalid model identifier" || model != "" {
					t.Fatalf("raw buffer model/error = %q/%v", model, err)
				}
			} else if err != nil || model != "good" {
				t.Fatalf("valid model/error = %q/%v", model, err)
			}
		})
	}
	test("model before decoder replacement", "{\"model\":\""+string([]byte{0xff})+"\"}", true)
	test("name before decoder replacement", "{\"model\":\"good\",\"name\":\""+string([]byte{0xff})+"\"}", true)
	test("valid model with opaque invalid bytes", "{\"model\":\"good\",\"prompt\":\""+string([]byte{0xff})+"\"}", false)
}

func TestModelControlsIngressFacade(t *testing.T) {
	test := func(name, method, target, body string, status, hits int) {
		t.Run(name, func(t *testing.T) {
			fx := newTierFixture(t, ollamaCase(t), nodepolicy.Default(), map[string][]string{"peer": {"good:tag"}}, nil)
			rec := httptest.NewRecorder()
			fx.p.soleFacade().handleHTTP(rec, httptest.NewRequest(method, target, strings.NewReader(body)))
			if rec.Code != status || fx.eng["peer"].hits() != hits {
				t.Fatalf("facade status/hits = %d/%d, want %d/%d", rec.Code, fx.eng["peer"].hits(), status, hits)
			}
			if hits == 0 && len(notificationsOf(t, fx.w, workloadSubmittedMethod)) != 0 {
				t.Fatal("invalid identifier submitted a workload")
			}
			if strings.Contains(rec.Body.String(), "sentinel") || bytes.Contains(bytes.Join(fx.w.lines(), nil), []byte("sentinel")) {
				t.Fatal("invalid identifier echoed")
			}
			if status == http.StatusBadRequest {
				var refusal struct {
					Error string `json:"error"`
					Code  string `json:"code"`
				}
				if err := json.Unmarshal(rec.Body.Bytes(), &refusal); err != nil {
					t.Fatal(err)
				}
				if refusal.Code != "request-identifier" && refusal.Code != "request-body" {
					t.Fatalf("refusal = %+v", refusal)
				}
			}
		})
	}
	test("valid model and opaque content", http.MethodPost, "/api/chat", `{"model":"good:tag","messages":[{"content":"a\nb"}],"opaque":{"model":"bad\u0085"}}`, http.StatusOK, 1)
	test("body control", http.MethodPost, "/api/chat", `{"model":"sentinel\nmodel"}`, http.StatusBadRequest, 0)
	test("control operation name", http.MethodPost, "/api/show", `{"name":"sentinel\u0085model"}`, http.StatusBadRequest, 0)
	test("query control", http.MethodGet, "/props?model=sentinel%0Amodel", "", http.StatusBadRequest, 0)
	test("detail control", http.MethodGet, "/v1/models/sentinel%7Fmodel", "", http.StatusBadRequest, 0)
}

func TestModelControlsIngressEarlyBodyErrorDoesNotPublishPath(t *testing.T) {
	test := func(name, method, target, body string, sharedEntry bool) {
		t.Run(name, func(t *testing.T) {
			var logs bytes.Buffer
			previous := slog.Default()
			slog.SetDefault(slog.New(slog.NewTextHandler(&logs, nil)))
			t.Cleanup(func() { slog.SetDefault(previous) })
			slog.Info("model controls log capture active")
			fx := newTierFixture(t, ollamaCase(t), nodepolicy.Default(), map[string][]string{"peer": {"good:tag"}}, nil)
			f := fx.p.soleFacade()
			rec := httptest.NewRecorder()
			r := httptest.NewRequest(method, target, strings.NewReader(body))
			wantSequence := fx.p.requestSequence
			if sharedEntry {
				// The listener gates normally admit only fixed inference routes.
				// Exercise its shared owner's early-error telemetry contract too.
				start := time.Now()
				f.serveInference(rec, r, start, "fixture", defaultStreamFrameIdle, concreteDispatchPolicy(start))
				wantSequence++
			} else {
				r.RemoteAddr = "127.0.0.1:4567"
				f.handlePlain(rec, r)
			}
			var refusal struct {
				Code  string `json:"code"`
				Error string `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &refusal); err != nil {
				t.Fatal(err)
			}
			if rec.Code != http.StatusBadRequest || refusal.Code != "request-body" || refusal.Error != "invalid model identifier" {
				t.Fatalf("body-first refusal = %d/%+v", rec.Code, refusal)
			}
			requests := notificationsOf(t, fx.w, engines.AddressMethod(f.profile.Name, "proxy/request"))
			if len(requests) != 1 {
				t.Fatalf("rejection notification count = %d, want 1", len(requests))
			}
			var event RequestEvent
			if err := json.Unmarshal(requests[0], &event); err != nil {
				t.Fatal(err)
			}
			if event.Path != "" || event.Status != http.StatusBadRequest || event.Error != "invalid model identifier" || event.NodeID != "" || event.Target != "" {
				t.Fatalf("early rejection metadata = %+v", event)
			}
			if !strings.Contains(logs.String(), "model controls log capture active") || (sharedEntry && !strings.Contains(logs.String(), "proxy request rejected")) {
				t.Fatal("log capture missed its positive control or inference rejection")
			}
			for _, capture := range [][]byte{rec.Body.Bytes(), bytes.Join(fx.w.lines(), nil), logs.Bytes()} {
				if bytes.Contains(capture, []byte("path-sentinel")) || bytes.Contains(capture, []byte("body-sentinel")) {
					t.Fatal("unvalidated path or identifier reached a response, notification or log")
				}
			}
			if fx.eng["peer"].hits() != 0 || len(notificationsOf(t, fx.w, workloadSubmittedMethod)) != 0 || len(notificationsOf(t, fx.w, engines.AddressMethod(f.profile.Name, "proxy/request-started"))) != 0 {
				t.Fatal("body-first refusal caused upstream, workload or started effects")
			}
			if fx.p.requestSequence != wantSequence || len(fx.p.activeRoutingRequests()) != 0 {
				t.Fatal("early-error repair changed canonical registration or cleanup")
			}
			ownedAssertHandlerDrained(t, fx.p, nil)
		})
	}
	test("passthrough escaped body and DEL path", http.MethodGet, "/v1/models/path-sentinel%7Fmodel", `{"name":"body-sentinel\nmodel"}`, false)
	test("passthrough raw UTF8 body and C1 path", http.MethodGet, "/api/v0/models/path-sentinel%C2%85model", "{\"name\":\"body-sentinel"+string([]byte{0xff})+"\"}", false)
	test("passthrough raw UTF8 body and raw UTF8 path", http.MethodGet, "/v1/models/path-sentinel%FFmodel", "{\"model\":\"body-sentinel"+string([]byte{0xff})+"\"}", false)
	test("shared owner escaped body and DEL path", http.MethodPost, "/v1/models/path-sentinel%7Fmodel", `{"model":"body-sentinel\nmodel"}`, true)
	test("shared owner raw UTF8 body and C1 path", http.MethodPost, "/api/v0/models/path-sentinel%C2%85model", "{\"model\":\"body-sentinel"+string([]byte{0xff})+"\"}", true)
}

func TestModelControlsSelectionInventory(t *testing.T) {
	profile := ollamaCase(t).profile
	bad := []string{"a\n", "a\x7f", "a\u0085", string([]byte{0xff}), strings.Repeat("a", 513), ""}
	for _, model := range bad {
		if nodeAdvertisesModel(profile, Node{Models: []string{model}}, "a") {
			t.Fatal("invalid inventory became a selectable normalized model")
		}
	}
	if nodeAdvertisesModel(profile, Node{Models: []string{strings.Repeat("a", 512)}}, strings.Repeat("a", 512)) {
		t.Fatal("normalization selected an identifier over 512 bytes")
	}
	d := NewDiscovery()
	d.SetSubscribed([]Node{{ID: "subscribed", Models: append(slices.Clone(bad), "good"), Loaded: append(slices.Clone(bad), "good")}})
	d.AddManual(Node{ID: "manual", Models: append(slices.Clone(bad), "good"), Loaded: append(slices.Clone(bad), "good")})
	for _, node := range d.Nodes() {
		if !slices.Equal(node.Models, []string{"good"}) || !slices.Equal(node.Loaded, []string{"good"}) {
			t.Fatalf("inventory was not filtered individually: %+v", node)
		}
	}
	t.Run("nil attribution fallback", func(t *testing.T) {
		n, ok := subscribedToNode(profile, noderec.DirectoryNode{HostUUID: "host", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{profile.DiscoveryService: {Port: 30001}}, Models: []string{"bad\n", "good"}})
		if !ok || !slices.Equal(n.Models, []string{"good"}) {
			t.Fatalf("legacy inventory = %+v, %v", n, ok)
		}
	})
	t.Run("known empty attribution", func(t *testing.T) {
		n, ok := subscribedToNode(profile, noderec.DirectoryNode{HostUUID: "host", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{profile.DiscoveryService: {Port: 30001}}, Models: []string{"good"}, ModelsByEngine: map[string][]string{}})
		if !ok || len(n.Models) != 0 {
			t.Fatalf("known empty inventory used union fallback: %+v, %v", n, ok)
		}
	})
}

func TestModelControlsSelectionRawDiscoveryAndManual(t *testing.T) {
	p, writer := policyProxy(t)
	f := p.soleFacade()
	// Raw bytes must reach the accepting owner; marshaling a Go string would
	// repair the synthetic invalid UTF-8 before the regression reaches it.
	raw := `{"nodes":[{"hostUuid":"host","name":"fixture","ip":"127.0.0.1","services":{"` + string(f.profile.DiscoveryService) + `":{"port":30001}},"models":["legacy"],"modelsByEngine":{"` + f.profile.Name + `":["bad` + string([]byte{0xff}) + `","bad\n","good"]},"loadedByEngine":{"` + f.profile.Name + `":["bad\n","good"]},"policy":{"revision":"1","roles":{"weak":{"engine":"` + f.profile.Name + `","model":"bad\n"},"medium":{"engine":"` + f.profile.Name + `","model":"good"}},"servedModels":[{"engine":"` + f.profile.Name + `","model":"bad\n"},{"engine":"` + f.profile.Name + `","model":"good"}],"strictAdmission":{"version":99,"models":{"opaque":"bad\n"}}}}]}`
	p.handleMessage(&Message{JSONRPC: "2.0", Method: engines.AddressMethod(f.profile.Name, noderec.NotifyNodes), Params: json.RawMessage(raw)})
	nodes := f.discovery.Nodes()
	if len(nodes) != 1 || !slices.Equal(nodes[0].Models, []string{"good"}) || !slices.Equal(nodes[0].Loaded, []string{"good"}) || len(nodes[0].Roles) != 1 || len(nodes[0].ServedModels) != 1 {
		t.Fatalf("raw discovery did not retain only valid siblings: %+v", nodes)
	}
	if nodes[0].StrictAdmission.State() != nodepolicy.StrictCapabilityUnsupported {
		t.Fatal("future capability was reinterpreted")
	}
	rpcOK(t, p, writer, engines.AddressMethod(f.profile.Name, "node/add-manual"), json.RawMessage(`{"id":"manual","host":"fixture","port":30002,"addresses":["127.0.0.1"],"models":["bad`+string([]byte{0xff})+`","good"]}`))
	for _, node := range f.discovery.Nodes() {
		if !slices.Equal(node.Models, []string{"good"}) {
			t.Fatalf("raw manual inventory = %+v", node)
		}
	}
}

func TestModelControlsSelectionResidency(t *testing.T) {
	p, _ := policyProxy(t)
	raw := json.RawMessage(`{"loadedByEngine":{"ollama":["bad\n","bad` + string([]byte{0xff}) + `","good"],"lmstudio":["bad\u0085"]}}`)
	if err := p.applySetResidency(raw); err != nil {
		t.Fatalf("mixed successful inventory refused: %v", err)
	}
	if !p.admission.isLoaded("ollama", "good") || p.admission.isLoaded("ollama", "bad") || p.admission.isLoaded("ollama", "bad�") {
		t.Fatal("residency lost a valid sibling or retained a repaired model")
	}
	p.admission.mu.Lock()
	known, empty := p.admission.residencyKnown, p.admission.loaded["lmstudio"]
	p.admission.mu.Unlock()
	if !known || empty == nil || len(empty) != 0 {
		t.Fatal("all-bad residency was not known empty")
	}
}

func TestModelControlsSelectionSelfCache(t *testing.T) {
	w := &recordingWriter{}
	p := newTestProxy(ollamaCase(t).profile, NewCodec(w), NewDiscovery(), 30001)
	f := p.soleFacade()
	test := func(name, models string, want []string) {
		t.Run(name, func(t *testing.T) {
			raw := `{"nodes":[{"hostUuid":"self","ip":"127.0.0.1","services":{"ol":{"port":30001}},"models":["legacy"],"modelsByEngine":{"ollama":` + models + `}}]}`
			p.handleMessage(&Message{Method: engines.AddressMethod("ollama", noderec.NotifyNodes), Params: json.RawMessage(raw)})
			id, cached := p.selfIdentity("ollama")
			if id != "self" || !slices.Equal(cached, want) {
				t.Fatalf("self cache id/models = %q/%v, want self/%v", id, cached, want)
			}
			if len(f.discovery.Nodes()) != 1 {
				t.Fatal("empty inventory removed the node")
			}
		})
	}
	test("valid siblings", `["bad\n","good"]`, []string{"good"})
	test("all bad replaces cache without union fallback", `["bad\u0085"]`, nil)
}

func TestModelControlsSelectionShortAliasConcreteBound(t *testing.T) {
	test := func(name, model string, role, selectable bool) {
		t.Run(name, func(t *testing.T) {
			pol := ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "seed:tag"}}})
			fx := newTierFixture(t, ollamaCase(t), pol, map[string][]string{"peer": {model}}, nil)
			// Selection still guards metadata supplied through an in-memory
			// snapshot; strict policy writes are tested separately below.
			pol.Tiers["weak"][0].Model = model
			fx.p.admission.setPolicy(pol)
			if role {
				fx.declare(t, "peer", map[string]nodepolicy.AdvertisedRole{"weak": {Engine: "ollama", Model: model}})
			}
			got := fx.p.soleFacade().resolveTierCandidates("weak", requestNeeds{}, true)
			if selectable {
				if len(got) != 1 || got[0].model != model {
					t.Fatalf("valid selected model = %+v", got)
				}
				rec := fx.do(t, "/api/chat", `{"model":"weak"}`)
				if rec.Code != http.StatusOK || fx.eng["peer"].hits() != 1 {
					t.Fatalf("short alias status/hits = %d/%d", rec.Code, fx.eng["peer"].hits())
				}
				var forwarded struct {
					Model string `json:"model"`
				}
				if err := json.Unmarshal([]byte(fx.eng["peer"].lastBody()), &forwarded); err != nil {
					t.Fatal(err)
				}
				if forwarded.Model != model {
					t.Fatal("selected concrete model changed")
				}
			} else if len(got) != 0 || fx.eng["peer"].hits() != 0 {
				t.Fatalf("invalid model selected: %+v", got)
			}
		})
	}
	test("local 512 bytes", strings.Repeat("a", 508)+":tag", false, true)
	test("role 512 bytes", strings.Repeat("a", 508)+":tag", true, true)
	test("local multibyte 512 bytes", strings.Repeat("界", 169)+"a:tag", false, true)
	test("local 513 bytes", strings.Repeat("a", 509)+":tag", false, false)
	test("role 513 bytes", strings.Repeat("a", 509)+":tag", true, false)
	test("role multibyte 513 bytes", strings.Repeat("界", 169)+"aa:tag", true, false)
	test("normalization exceeds 512", strings.Repeat("a", 512), false, false)
	test("role control before trim", "seed:tag\n", true, false)
}

func TestModelControlsSelectionPolicyBounds(t *testing.T) {
	test := func(name, model string, valid bool) {
		t.Run(name, func(t *testing.T) {
			pol, err := nodepolicy.Parse([]byte(apiTierPolicy))
			if err != nil {
				t.Fatal(err)
			}
			pol.Routing.Deployments[0].Model = model
			if err := pol.Validate(); (err == nil) != valid {
				t.Fatalf("deployment validity = %v", err)
			}
			pol = ollamaTierPolicy(map[string][]nodepolicy.TierEntry{"weak": {{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: model}}})
			if err := pol.Validate(); (err == nil) != valid {
				t.Fatalf("local tier validity = %v", err)
			}
			pol = nodepolicy.Default()
			pol.Roles = map[string]nodepolicy.Role{"weak": {Engine: "ollama", Model: model}}
			if err := pol.Validate(); (err == nil) != valid {
				t.Fatalf("role validity = %v", err)
			}
			if nodepolicy.PublicDispatchAvailable() {
				t.Fatal("public dispatch enabled")
			}
		})
	}
	test("512 bytes", strings.Repeat("a", 508)+":tag", true)
	test("513 bytes", strings.Repeat("a", 509)+":tag", false)
	test("multibyte 512 bytes", strings.Repeat("界", 169)+"a:tag", true)
	test("multibyte 513 bytes", strings.Repeat("界", 169)+"aa:tag", false)
	test("control", "bad\n", false)
}

type modelControlsTransport struct{ body string }

func (rt modelControlsTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(rt.body))}, nil
}

func TestModelControlsSelectionModelList(t *testing.T) {
	test := func(name, body string, role routeRole, want int) {
		t.Run(name, func(t *testing.T) {
			previous := modelListClient
			modelListClient = &http.Client{Transport: modelControlsTransport{body: body}}
			t.Cleanup(func() { modelListClient = previous })
			p, _ := policyProxy(t)
			f := p.soleFacade()
			rec := httptest.NewRecorder()
			status, err := f.serveModelList(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil), route{Path: "/v1/models", Role: role}, []candidate{{id: "fixture", url: &url.URL{Scheme: "http", Host: "fixture.invalid"}}})
			if err != nil || status != http.StatusOK {
				t.Fatalf("list status/error = %d/%v", status, err)
			}
			var result struct {
				Data   []json.RawMessage `json:"data"`
				Models []json.RawMessage `json:"models"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			rows := result.Data
			if role == roleModelListNativeGET {
				rows = result.Models
			}
			if len(rows) != want {
				t.Fatalf("list rows = %s, want %d entries", rec.Body.Bytes(), want)
			}
			if want == 1 {
				var identity struct {
					ID    string `json:"id"`
					Model string `json:"model"`
					Name  string `json:"name"`
				}
				if err := json.Unmarshal(rows[0], &identity); err != nil {
					t.Fatal(err)
				}
				if identity.ID != "good" && identity.Model != "good" && identity.Name != "good" {
					t.Fatalf("valid sibling lost: %+v", identity)
				}
			}
		})
	}
	test("OpenAI mixed raw entries", `{"data":[{"id":"bad`+string([]byte{0xff})+`"},{"id":"bad\n"},{"id":"good"}]}`, roleModelListOpenAIGET, 1)
	test("native invalid name cannot hide behind model", `{"models":[{"model":"good","name":"bad\n"},{"model":"bad\n","name":"good"},{"model":"good"}]}`, roleModelListNativeGET, 1)
	test("native absent model uses valid name", `{"models":[{"name":"good"}]}`, roleModelListNativeGET, 1)
	test("native null model cannot fall back", `{"models":[{"model":null,"name":"good"},{"model":"good"}]}`, roleModelListNativeGET, 1)
	test("all bad is successful empty", `{"data":[{"id":"bad\u0085"}]}`, roleModelListOpenAIGET, 0)
	test("normalized overlong entry", `{"models":[{"model":"`+strings.Repeat("a", 512)+`"},{"model":"good"}]}`, roleModelListNativeGET, 1)
}

func TestModelControlsSelectionModelListFailureSemantics(t *testing.T) {
	dialect := func(name, path, envelope, valid, missing string) {
		t.Run(name, func(t *testing.T) {
			test := func(name, rows string, origin, healthyPeer bool, wantStatus int, wantModels []string) {
				t.Run(name, func(t *testing.T) {
					const browserOrigin = "http://app.test"
					wantOrigin := ""
					if origin {
						wantOrigin = browserOrigin
					}
					var upstreamHits atomic.Int32
					upstream := func(rows string) *httptest.Server {
						t.Helper()
						server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							upstreamHits.Add(1)
							if r.URL.Path != path || r.Header.Get("Origin") != wantOrigin {
								t.Error("model-list forwarding changed path or Origin")
							}
							w.Header().Set("Access-Control-Allow-Origin", browserOrigin)
							w.Header().Set("Content-Type", "application/json")
							if _, err := io.WriteString(w, `{"`+envelope+`":[`+rows+`]}`); err != nil {
								t.Errorf("write model inventory: %v", err)
							}
						}))
						t.Cleanup(server.Close)
						return server
					}
					p, writer := policyProxy(t)
					f := p.soleFacade()
					f.discovery.AddManual(nodeFor(t, "first", upstream(rows).URL))
					wantHits := int32(1)
					if healthyPeer {
						f.discovery.AddManual(nodeFor(t, "second", upstream(valid).URL))
						wantHits++
					}
					r := httptest.NewRequest(http.MethodGet, path, nil)
					r.RemoteAddr = "127.0.0.1:4567"
					if origin {
						r.Header.Set("Origin", browserOrigin)
					}
					rec := httptest.NewRecorder()
					f.handlePlain(rec, r)
					if rec.Code != wantStatus || upstreamHits.Load() != wantHits {
						t.Fatalf("model list status/hits = %d/%d, want %d/%d", rec.Code, upstreamHits.Load(), wantStatus, wantHits)
					}
					if rec.Header().Get("Access-Control-Allow-Origin") != wantOrigin {
						t.Fatalf("model list Origin permission = %q, want %q", rec.Header().Get("Access-Control-Allow-Origin"), wantOrigin)
					}
					if rec.Code == http.StatusOK {
						var result struct {
							Data   []json.RawMessage `json:"data"`
							Models []json.RawMessage `json:"models"`
						}
						if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
							t.Fatal(err)
						}
						entries := result.Data
						if envelope == "models" {
							entries = result.Models
						}
						if entries == nil {
							t.Fatal("successful inventory lost known-empty array")
						}
						got := make([]string, 0, len(entries))
						for _, raw := range entries {
							var model struct {
								ID    string `json:"id"`
								Model string `json:"model"`
								Name  string `json:"name"`
							}
							if err := json.Unmarshal(raw, &model); err != nil {
								t.Fatal(err)
							}
							id := model.ID
							if envelope == "models" {
								id = model.Model
								if id == "" {
									id = model.Name
								}
							}
							got = append(got, id)
						}
						if !slices.Equal(got, wantModels) {
							t.Fatalf("models = %v, want %v", got, wantModels)
						}
					} else {
						var result struct {
							Error string `json:"error"`
						}
						if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
							t.Fatal(err)
						}
						if result.Error != "model inventory unavailable" {
							t.Fatalf("failure = %+v", result)
						}
					}
					requests := notificationsOf(t, writer, engines.AddressMethod(f.profile.Name, "proxy/request"))
					if len(requests) != 1 {
						t.Fatalf("completion count = %d, want 1", len(requests))
					}
					var event RequestEvent
					if err := json.Unmarshal(requests[0], &event); err != nil {
						t.Fatal(err)
					}
					if event.Path != path || event.Status != wantStatus || event.Target != "cluster" {
						t.Fatalf("list completion = %+v", event)
					}
				})
			}
			test("nonobject record", `42`, false, false, http.StatusServiceUnavailable, nil)
			test("null record", `null`, false, false, http.StatusServiceUnavailable, nil)
			test("empty object", `{}`, false, false, http.StatusServiceUnavailable, nil)
			test("missing dialect identity", missing, false, false, http.StatusServiceUnavailable, nil)
			test("nonobject after valid sibling", valid+`,42`, false, false, http.StatusServiceUnavailable, nil)
			test("missing identity after valid sibling", valid+`,`+missing, false, false, http.StatusServiceUnavailable, nil)
			test("malformed candidate and valid peer", `42`, false, true, http.StatusOK, []string{"good"})
			test("Origin nonobject after valid sibling", valid+`,42`, true, false, http.StatusBadGateway, nil)
			test("Origin missing identity after valid sibling", valid+`,`+missing, true, false, http.StatusBadGateway, nil)
			test("Origin malformed candidate and valid peer", `42`, true, true, http.StatusBadGateway, nil)
			invalid := strings.ReplaceAll(valid, "good", `bad\n`)
			test("supplied invalid sibling", invalid+`,`+valid, false, false, http.StatusOK, []string{"good"})
			test("Origin supplied invalid sibling", invalid+`,`+valid, true, false, http.StatusOK, []string{"good"})
			test("all supplied invalid is known empty", invalid, false, false, http.StatusOK, nil)
			test("Origin all supplied invalid is known empty", invalid, true, false, http.StatusOK, nil)
			test("Origin successful empty array", "", true, false, http.StatusOK, nil)
		})
	}
	dialect("OpenAI", "/v1/models", "data", `{"id":"good"}`, `{"name":"bad\n"}`)
	dialect("native", "/api/tags", "models", `{"model":"good"}`, `{"id":"bad\n"}`)
}

func TestModelControlsStrictCapabilityIdentityThroughProjectionAndOverlay(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		models := []nodepolicy.StrictAdmissionModel{
			{Engine: tc.profile.Name, Model: "Org/Model:Tag", MaxConcurrent: 2, Identity: strictRequest().Identity},
			{Engine: otherEngine(t, tc).Name, Model: "Other/Model:Tag", MaxConcurrent: 3, Identity: strictRequest().Identity},
		}
		before := slices.Clone(models)
		capability := &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion, Models: models}
		if capability.State() != nodepolicy.StrictCapabilityValid {
			t.Fatal("invalid capability fixture")
		}
		directory := noderec.DirectoryNode{HostUUID: "host", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{tc.profile.DiscoveryService: {Port: 30001}}, Models: []string{"Org/Model:Tag"}, Policy: &nodepolicy.Advertisement{Revision: "1", StrictAdmission: capability}}
		assertIdentity := func(node Node, want *nodepolicy.StrictAdmissionCapability) {
			t.Helper()
			if node.StrictAdmission != want || unsafe.SliceData(node.StrictAdmission.Models) != unsafe.SliceData(want.Models) || len(node.StrictAdmission.Models) != len(want.Models) || cap(node.StrictAdmission.Models) != cap(want.Models) {
				t.Fatal("valid capability or model-slice identity changed")
			}
		}
		projected, ok := subscribedToNode(tc.profile, directory)
		if !ok {
			t.Fatal("directory projection failed")
		}
		assertIdentity(projected, capability)
		d := NewDiscovery()
		d.SetSubscribed([]Node{projected})
		stored := d.Nodes()
		if len(stored) != 1 {
			t.Fatal("overlay lost the node")
		}
		assertIdentity(stored[0], capability)
		if discovered, updated, removed := d.SetSubscribed(stored); len(discovered)+len(updated)+len(removed) != 0 {
			t.Fatal("repeated filtering announced node churn")
		}
		assertIdentity(d.Nodes()[0], capability)

		replacement := &nodepolicy.StrictAdmissionCapability{Version: capability.Version, Models: slices.Clone(models)}
		replacement.Models[0].MaxConcurrent++
		directory.Policy = &nodepolicy.Advertisement{Revision: "2", StrictAdmission: replacement}
		projected, ok = subscribedToNode(tc.profile, directory)
		if !ok {
			t.Fatal("readiness projection failed")
		}
		assertIdentity(projected, replacement)
		if discovered, updated, removed := d.SetSubscribed([]Node{projected}); len(discovered)+len(updated)+len(removed) != 0 {
			t.Fatal("valid readiness update announced node churn")
		}
		assertIdentity(d.Nodes()[0], replacement)
		if !slices.Equal(models, before) || !slices.Equal(capability.Models, before) || unsafe.SliceData(capability.Models) != unsafe.SliceData(models) {
			t.Fatal("readiness replacement mutated the original advertisement")
		}
	})
}

func TestModelControlsStrictCapabilityRemovalClonesWithoutMutation(t *testing.T) {
	profile := ollamaCase(t).profile
	first := nodepolicy.StrictAdmissionModel{Engine: "ollama", Model: "Org/Model:Tag", MaxConcurrent: 2, Identity: strictRequest().Identity}
	second := nodepolicy.StrictAdmissionModel{Engine: "lmstudio", Model: "Other/Model:Tag", MaxConcurrent: 3, Identity: strictRequest().Identity}
	bad := first
	bad.Model = "bad\n"
	badUTF8 := second
	badUTF8.Model = string([]byte{0xff, 0xfe})
	test := func(name string, models, want []nodepolicy.StrictAdmissionModel) {
		t.Run(name, func(t *testing.T) {
			before := slices.Clone(models)
			source := &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion, Models: models}
			directory := noderec.DirectoryNode{HostUUID: "host", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{profile.DiscoveryService: {Port: 30001}}, Models: []string{first.Model}, Policy: &nodepolicy.Advertisement{Revision: "1", StrictAdmission: source}}
			projected, ok := subscribedToNode(profile, directory)
			if !ok || projected.StrictAdmission == nil {
				t.Fatal("projection lost the capability")
			}
			clean := projected.StrictAdmission
			if clean == source || unsafe.SliceData(clean.Models) == unsafe.SliceData(models) || !slices.Equal(clean.Models, want) || clean.Models == nil {
				t.Fatal("invalid entries were not removed with independent ordered backing")
			}
			if len(want) == 0 && clean.State() != nodepolicy.StrictCapabilityInvalid {
				t.Fatal("all-invalid capability became eligible")
			}
			d := NewDiscovery()
			d.SetSubscribed([]Node{projected})
			stored := d.Nodes()
			if len(stored) != 1 || stored[0].StrictAdmission != clean || unsafe.SliceData(stored[0].StrictAdmission.Models) != unsafe.SliceData(clean.Models) {
				t.Fatal("overlay recloned the already filtered capability")
			}

			// Also feed the original advertisement directly to the overlay's
			// typed acceptance owner rather than only filtering its projection.
			direct := projected
			direct.StrictAdmission = source
			if discovered, updated, removed := d.SetSubscribed([]Node{direct}); len(discovered)+len(updated)+len(removed) != 0 {
				t.Fatal("capability filtering announced node churn")
			}
			stored = d.Nodes()
			clean = stored[0].StrictAdmission
			if clean == source || unsafe.SliceData(clean.Models) == unsafe.SliceData(models) || !slices.Equal(clean.Models, want) || clean.Models == nil {
				t.Fatal("direct overlay intake did not clone only the invalid entries")
			}
			d.SetSubscribed(stored)
			if got := d.Nodes()[0].StrictAdmission; got != clean || unsafe.SliceData(got.Models) != unsafe.SliceData(clean.Models) {
				t.Fatal("repeated overlay filtering replaced clean backing")
			}
			if source.Version != nodepolicy.StrictAdmissionVersion || !slices.Equal(source.Models, before) || unsafe.SliceData(source.Models) != unsafe.SliceData(models) {
				t.Fatal("filtering mutated the source capability or slice")
			}
		})
	}
	test("mixed invalid siblings", []nodepolicy.StrictAdmissionModel{first, bad, second, badUTF8}, []nodepolicy.StrictAdmissionModel{first, second})
	test("all invalid remains known empty", []nodepolicy.StrictAdmissionModel{bad, badUTF8}, []nodepolicy.StrictAdmissionModel{})
}

func TestModelControlsStrictCapabilityNilEmptyAndUnsupportedIdentity(t *testing.T) {
	const opaque = `{"version":99,"models":{"opaque":"bad\u0000"},"future":"kept"}`
	var future nodepolicy.StrictAdmissionCapability
	if err := json.Unmarshal([]byte(opaque), &future); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name       string
		capability *nodepolicy.StrictAdmissionCapability
		state      nodepolicy.StrictCapabilityState
	}{
		{"absent capability", nil, nodepolicy.StrictCapabilityAbsent},
		{"nil supported models", &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion}, nodepolicy.StrictCapabilityInvalid},
		{"known-empty supported backing", &nodepolicy.StrictAdmissionCapability{Version: nodepolicy.StrictAdmissionVersion, Models: make([]nodepolicy.StrictAdmissionModel, 0, 4)}, nodepolicy.StrictCapabilityInvalid},
		{"opaque unsupported JSON", &future, nodepolicy.StrictCapabilityUnsupported},
		{"unsupported model slice is uninterpreted", &nodepolicy.StrictAdmissionCapability{Version: 99, Models: []nodepolicy.StrictAdmissionModel{{Model: "bad\n"}}}, nodepolicy.StrictCapabilityUnsupported},
	} {
		t.Run(tc.name, func(t *testing.T) {
			profile := ollamaCase(t).profile
			var models []nodepolicy.StrictAdmissionModel
			if tc.capability != nil {
				models = tc.capability.Models
			}
			before := slices.Clone(models)
			directory := noderec.DirectoryNode{HostUUID: "host", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{profile.DiscoveryService: {Port: 30001}}, Policy: &nodepolicy.Advertisement{Revision: "1", StrictAdmission: tc.capability}}
			projected, ok := subscribedToNode(profile, directory)
			if !ok || projected.StrictAdmission != tc.capability {
				t.Fatal("projection replaced the original capability")
			}
			d := NewDiscovery()
			d.SetSubscribed([]Node{projected})
			d.SetSubscribed(d.Nodes())
			got := d.Nodes()[0].StrictAdmission
			if got != tc.capability || got.State() != tc.state {
				t.Fatal("overlay changed capability identity or eligibility")
			}
			if got != nil && ((got.Models == nil) != (models == nil) || len(got.Models) != len(models) || cap(got.Models) != cap(models) || unsafe.SliceData(got.Models) != unsafe.SliceData(models) || !slices.Equal(got.Models, before)) {
				t.Fatal("overlay changed nil/empty/unsupported model backing")
			}
			if got == &future {
				raw, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(raw, []byte(opaque)) {
					t.Fatal("unsupported capability content changed")
				}
			}
		})
	}
}
