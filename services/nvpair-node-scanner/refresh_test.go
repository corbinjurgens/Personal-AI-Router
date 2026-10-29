// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestModelControlsScannerInventory(t *testing.T) {
	ascii512 := strings.Repeat("a", 512)
	multibyte512 := strings.Repeat("界", 170) + "ab"
	cases := []struct {
		name, token string
		valid       bool
	}{
		{"empty", `""`, false},
		{"raw invalid UTF8", "\"raw-" + string([]byte{0xff}) + "\"", false},
		{"escaped NUL", `"bad\u0000model"`, false},
		{"escaped C0 end", `"bad\u001fmodel"`, false},
		{"escaped DEL", `"bad\u007fmodel"`, false},
		{"escaped C1 start", `"bad\u0080model"`, false},
		{"escaped C1 end", `"bad\u009fmodel"`, false},
		{"ASCII 512 bytes", `"` + ascii512 + `"`, true},
		{"ASCII 513 bytes", `"` + ascii512 + `a"`, false},
		{"multibyte 512 bytes", `"` + multibyte512 + `"`, true},
		{"multibyte 513 bytes", `"` + strings.Repeat("界", 171) + `"`, false},
		{"spaces tags slash and Unicode", `" model/日本語:tag "`, true},
		{"valid replacement character", `"model\ufffd"`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			list := `["first",` + tc.token + `,"last"]`
			strictEntry := strings.Replace(strictModelJSON("j304-token"), `"j304-token"`, tc.token, 1)
			strict := strictModelJSON("first") + "," + strictEntry + "," + strictModelJSON("last")
			// Served references have a separate encoded-object bound. Keep
			// identifier positives (including512 bytes) in inventory/roles/proof,
			// with representable permissions; invalid tokens still test filtering.
			servedToken := tc.token
			wantServed := nodepolicy.ServedModels{{Engine: "ollama", Model: "first"}}
			if tc.valid {
				servedToken = `"last"`
				wantServed = append(wantServed, nodepolicy.Role{Engine: "ollama", Model: "last"})
			}
			body := fmt.Sprintf(`{"models":%s,"modelsByEngine":{"ollama":%s,"lmstudio":[]},"loadedByEngine":{"ollama":%s},"policy":{"revision":"7","servedModels":[{"engine":"ollama","model":"first"},{"engine":"ollama","model":%s}],"roles":{"weak":{"engine":"ollama","model":%s},"strong":{"engine":"ollama","model":"last"}},"strictAdmission":{"version":1,"models":[%s]}}}`, list, list, list, servedToken, tc.token, strict)
			d := newRefreshTestDaemon()
			d.modelsHTTP.Transport = serviceBuildTransport(func(r *http.Request) (*http.Response, error) {
				if r.URL.Host != "127.0.0.1:24001" || r.URL.Path != "/v1/models" {
					t.Fatalf("unexpected inventory request: %s", r.URL)
				}
				return serviceBuildResponse(body, http.StatusOK), nil
			})
			seedEMNode(d, "model-controls", "127.0.0.1", 24001)
			if !d.refreshNodeModels("model-controls", "127.0.0.1", 24001, "") {
				t.Fatal("successful mixed inventory did not update")
			}
			want := []string{"first", "last"}
			if tc.valid {
				var model string
				if err := json.Unmarshal([]byte(tc.token), &model); err != nil {
					t.Fatalf("decode positive fixture: %v", err)
				}
				want = []string{"first", model, "last"}
			}
			node, present := d.dir.get("model-controls")
			if !present || !reflect.DeepEqual(node.Models, want) || !reflect.DeepEqual(node.ModelsByEngine, map[string][]string{"ollama": want, "lmstudio": {}}) || !reflect.DeepEqual(node.LoadedByEngine, map[string][]string{"ollama": want}) {
				t.Fatal("inventory lost valid siblings, order, attribution or known-empty engine")
			}
			if !reflect.DeepEqual(d.lastModels[node.HostUUID], want) || !reflect.DeepEqual(d.lastModelsByEngine[node.HostUUID], node.ModelsByEngine) || !reflect.DeepEqual(d.lastLoadedByEngine[node.HostUUID], node.LoadedByEngine) || !reflect.DeepEqual(d.lastPolicy[node.HostUUID], node.Policy) {
				t.Fatal("cache differs from published sanitized inventory")
			}
			if node.Policy == nil || node.Policy.Revision != "7" || node.Policy.Roles["strong"].Model != "last" {
				t.Fatal("valid advertisement metadata was lost")
			}
			_, hasWeak := node.Policy.Roles["weak"]
			if hasWeak != tc.valid || !reflect.DeepEqual(node.Policy.ServedModels, wantServed) {
				t.Fatal("advertisement did not filter model entries individually")
			}
			if tc.valid && node.Policy.Roles["weak"].Model != want[1] {
				t.Fatal("valid role model changed while using bounded served references")
			}
			capability := node.Policy.StrictAdmission
			if capability.State() != nodepolicy.StrictCapabilityValid {
				t.Fatal("invalid model sibling discarded the valid supported capability")
			}
			var strictModels []string
			for _, entry := range capability.Models {
				if entry.Engine != "ollama" {
					t.Fatal("supported capability lost engine attribution")
				}
				strictModels = append(strictModels, entry.Model)
			}
			if !reflect.DeepEqual(strictModels, want) {
				t.Fatal("supported capability lost valid model siblings or order")
			}
		})
	}
}

func TestModelControlsScannerServedReferenceBounds(t *testing.T) {
	for _, tc := range []struct{ name, model string }{
		{"ASCII identifier512 with oversized served reference", strings.Repeat("a", 512)},
		{"multibyte identifier512 with oversized served reference", strings.Repeat("界", 170) + "ab"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := nodepolicy.ValidateModelIdentifier(tc.model); err != nil {
				t.Fatalf("identifier positive violates its separate contract: %v", err)
			}
			// Encode raw input without ServedModels.MarshalJSON's strict write
			// guard, then exercise the actual receive-side container contract.
			ref, err := json.Marshal([]nodepolicy.Role{{Engine: "ollama", Model: tc.model}})
			if err != nil {
				t.Fatal(err)
			}
			var refused nodepolicy.ServedModels
			if err := json.Unmarshal(ref, &refused); err == nil {
				t.Fatal("oversized encoded served reference was accepted")
			}
			d := newRefreshTestDaemon()
			body := `{"models":["good"],"modelsByEngine":{"ollama":["good"]},"loadedByEngine":{"ollama":["good"]},"policy":{"revision":"7","servedModels":[{"engine":"ollama","model":"good"}]}}`
			d.modelsHTTP.Transport = serviceBuildTransport(func(*http.Request) (*http.Response, error) {
				return serviceBuildResponse(body, http.StatusOK), nil
			})
			seedEMNode(d, "served-reference-bound", "127.0.0.1", 24001)
			if !d.refreshNodeModels("served-reference-bound", "127.0.0.1", 24001, "") {
				t.Fatal("bounded served-reference positive did not publish")
			}
			before, _ := d.dir.get("served-reference-bound")
			body = `{"models":["replacement"],"policy":{"revision":"8","servedModels":` + string(ref) + `}}`
			if d.refreshNodeModels("served-reference-bound", "127.0.0.1", 24001, "") {
				t.Fatal("oversized served-reference document published")
			}
			after, _ := d.dir.get("served-reference-bound")
			if !reflect.DeepEqual(after, before) || !reflect.DeepEqual(d.lastModels[before.HostUUID], before.Models) ||
				!reflect.DeepEqual(d.lastModelsByEngine[before.HostUUID], before.ModelsByEngine) ||
				!reflect.DeepEqual(d.lastLoadedByEngine[before.HostUUID], before.LoadedByEngine) || !reflect.DeepEqual(d.lastPolicy[before.HostUUID], before.Policy) {
				t.Fatal("oversized served reference replaced last-good directory or cache")
			}
		})
	}
}

func TestModelControlsScannerFutureCapability(t *testing.T) {
	d := newRefreshTestDaemon()
	body := `{"models":["good"],"policy":{"revision":"9","strictAdmission":{"version":2,"models":[{"model":"raw-` + string([]byte{0xff}) + `"},7,null]}}}`
	d.modelsHTTP.Transport = serviceBuildTransport(func(*http.Request) (*http.Response, error) {
		return serviceBuildResponse(body, http.StatusOK), nil
	})
	got, ok := d.fetchModels("127.0.0.1", 24001, "")
	if !ok || !reflect.DeepEqual(got.models, []string{"good"}) || got.policy == nil || got.policy.StrictAdmission.State() != nodepolicy.StrictCapabilityUnsupported || len(got.policy.StrictAdmission.Models) != 0 {
		t.Fatal("unsupported capability was interpreted, enabled or cost the valid flat inventory")
	}
}

func TestModelControlsScannerSuccessfulEmptyAndLastGood(t *testing.T) {
	d := newRefreshTestDaemon()
	body := `{"models":["good"],"modelsByEngine":{"ollama":["good"]},"loadedByEngine":{"ollama":["good"]},"policy":{"revision":"8","servedModels":[{"engine":"ollama","model":"good"}]}}`
	d.modelsHTTP.Transport = serviceBuildTransport(func(*http.Request) (*http.Response, error) {
		return serviceBuildResponse(body, http.StatusOK), nil
	})
	seedEMNode(d, "empty-controls", "127.0.0.1", 24001)
	if !d.refreshNodeModels("empty-controls", "127.0.0.1", 24001, "") {
		t.Fatal("valid initial inventory did not update")
	}
	lastGood, _ := d.dir.get("empty-controls")
	for _, tc := range []struct{ name, raw string }{
		{"nonstring flat model", `{"models":[7]}`},
		{"nonarray engine models", `{"modelsByEngine":{"ollama":7}}`},
		{"trailing document", `{"models":["new"]} {}`},
		{"invalid engine syntax in served reference", `{"models":["new"],"policy":{"servedModels":[{"engine":"unrecognized/engine","model":"new"}]}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body = tc.raw
			if d.refreshNodeModels("empty-controls", "127.0.0.1", 24001, "") {
				t.Fatal("malformed document updated inventory")
			}
			node, _ := d.dir.get("empty-controls")
			if !reflect.DeepEqual(node.Models, []string{"good"}) || !reflect.DeepEqual(d.lastModels[node.HostUUID], node.Models) ||
				!reflect.DeepEqual(node, lastGood) || !reflect.DeepEqual(d.lastModelsByEngine[node.HostUUID], lastGood.ModelsByEngine) ||
				!reflect.DeepEqual(d.lastLoadedByEngine[node.HostUUID], lastGood.LoadedByEngine) || !reflect.DeepEqual(d.lastPolicy[node.HostUUID], lastGood.Policy) {
				t.Fatal("malformed document replaced last-good directory or cache")
			}
		})
	}
	t.Run("valid dynamic engine advertisement", func(t *testing.T) {
		body = `{"models":["new"],"modelsByEngine":{"unrecognized-engine":["new"]},"loadedByEngine":{"unrecognized-engine":["new"]},"policy":{"revision":"8","servedModels":[{"engine":"unrecognized-engine","model":"new"}]}}`
		if !d.refreshNodeModels("empty-controls", "127.0.0.1", 24001, "") {
			t.Fatal("valid dynamic engine advertisement did not update inventory")
		}
		node, present := d.dir.get("empty-controls")
		wantModels := []string{"new"}
		wantByEngine := map[string][]string{"unrecognized-engine": {"new"}}
		wantServed := nodepolicy.ServedModels{{Engine: "unrecognized-engine", Model: "new"}}
		if !present || !reflect.DeepEqual(node.Models, wantModels) || !reflect.DeepEqual(node.ModelsByEngine, wantByEngine) ||
			!reflect.DeepEqual(node.LoadedByEngine, wantByEngine) || node.Policy == nil || node.Policy.Revision != "8" || !reflect.DeepEqual(node.Policy.ServedModels, wantServed) {
			t.Fatal("accepted dynamic engine or model was lost from the directory advertisement")
		}
		if !reflect.DeepEqual(d.lastModels[node.HostUUID], wantModels) || !reflect.DeepEqual(d.lastModelsByEngine[node.HostUUID], wantByEngine) ||
			!reflect.DeepEqual(d.lastLoadedByEngine[node.HostUUID], wantByEngine) || !reflect.DeepEqual(d.lastPolicy[node.HostUUID], node.Policy) {
			t.Fatal("valid dynamic engine advertisement was not cached with its attribution")
		}
	})
	body = `{"models":["","bad\u007f"],"modelsByEngine":{"ollama":["bad\u0080"]},"loadedByEngine":{"ollama":["bad\u0000"]},"policy":{"revision":"9","servedModels":[{"engine":"ollama","model":""}]}}`
	if !d.refreshNodeModels("empty-controls", "127.0.0.1", 24001, "") {
		t.Fatal("successful all-invalid inventory did not replace last-good")
	}
	node, _ := d.dir.get("empty-controls")
	if node.Models == nil || len(node.Models) != 0 || !reflect.DeepEqual(node.ModelsByEngine, map[string][]string{"ollama": {}}) || !reflect.DeepEqual(node.LoadedByEngine, map[string][]string{"ollama": {}}) || node.Policy == nil || node.Policy.ServedModels == nil || len(node.Policy.ServedModels) != 0 {
		t.Fatal("all-invalid success did not retain known-empty inventory and deny-all permission")
	}
	if d.lastModels[node.HostUUID] == nil || len(d.lastModels[node.HostUUID]) != 0 || !reflect.DeepEqual(d.lastModelsByEngine[node.HostUUID], node.ModelsByEngine) || !reflect.DeepEqual(d.lastLoadedByEngine[node.HostUUID], node.LoadedByEngine) {
		t.Fatal("successful known-empty result was not cached")
	}
	for _, tc := range []struct {
		name, raw string
		present   bool
	}{
		{"absent attribution", `{"models":["good"]}`, false},
		{"null attribution", `{"models":["good"],"modelsByEngine":null,"loadedByEngine":null}`, false},
		{"present empty attribution", `{"models":["good"],"modelsByEngine":{},"loadedByEngine":{}}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body = tc.raw
			got, ok := d.fetchModels("127.0.0.1", 24001, "")
			if !ok || (got.byEngine != nil) != tc.present || (got.loadedByEngine != nil) != tc.present || !reflect.DeepEqual(got.models, []string{"good"}) {
				t.Fatal("fetch collapsed absent and present-empty attribution or changed valid flat models")
			}
		})
	}
}

// Inventory presence must participate in the directory's change decision,
// independently of the flat models and the shared notification serializer.
func TestModelControlsScannerAttributionPresenceTransition(t *testing.T) {
	d := newRefreshTestDaemon()
	var wire bytes.Buffer
	d.codec = NewCodec(&wire)
	assertEvent := func(present bool) {
		t.Helper()
		var frame Message
		if err := json.Unmarshal(wire.Bytes(), &frame); err != nil {
			t.Fatalf("decode actual refresh notification: %v", err)
		}
		var event noderec.NodeEvent
		if err := json.Unmarshal(frame.Params, &event); err != nil {
			t.Fatalf("decode refresh node event: %v", err)
		}
		if frame.Method != noderec.NotifyNodeUpdated || !frame.IsNotification() || event.Node.HostUUID != "presence-transition" ||
			(event.Node.ModelsByEngine != nil) != present || !reflect.DeepEqual(event.Node.Models, []string{"good"}) {
			t.Fatalf("refresh notification lost identity, models or attribution presence: %+v", event.Node)
		}
		if event.Node.LoadedByEngine != nil {
			t.Fatal("models attribution transition manufactured loaded attribution")
		}
		wire.Reset()
	}
	body := `{"models":["good"]}`
	d.modelsHTTP.Transport = serviceBuildTransport(func(*http.Request) (*http.Response, error) {
		return serviceBuildResponse(body, http.StatusOK), nil
	})
	seedEMNode(d, "presence-transition", "127.0.0.1", 24001)
	if !d.refreshNodeModels("presence-transition", "127.0.0.1", 24001, "") {
		t.Fatal("initial flat inventory did not update")
	}
	assertEvent(false)
	body = `{"models":["good"],"modelsByEngine":{}}`
	if !d.refreshNodeModels("presence-transition", "127.0.0.1", 24001, "") {
		t.Fatal("nil-to-present-empty attribution must publish a directory change")
	}
	assertEvent(true)
	node, _ := d.dir.get("presence-transition")
	if node.ModelsByEngine == nil || d.lastModelsByEngine[node.HostUUID] == nil || len(node.EngineModels("ollama")) != 0 || !reflect.DeepEqual(node.Models, []string{"good"}) {
		t.Fatal("authoritative empty refresh did not replace flat fallback in directory and cache")
	}
	if d.refreshNodeModels("presence-transition", "127.0.0.1", 24001, "") {
		t.Fatal("identical present-empty attribution emitted another update")
	}
	if wire.Len() != 0 {
		t.Fatal("unchanged present-empty refresh emitted a notification")
	}
	body = `{"models":["good"]}`
	if !d.refreshNodeModels("presence-transition", "127.0.0.1", 24001, "") {
		t.Fatal("present-empty-to-nil attribution must publish a directory change")
	}
	assertEvent(false)
	node, _ = d.dir.get("presence-transition")
	if node.ModelsByEngine != nil || d.lastModelsByEngine[node.HostUUID] != nil || !reflect.DeepEqual(node.EngineModels("ollama"), []string{"good"}) {
		t.Fatal("legacy nil attribution did not restore its existing flat fallback")
	}
	if d.refreshNodeModels("presence-transition", "127.0.0.1", 24001, "") || wire.Len() != 0 {
		t.Fatal("unchanged nil attribution emitted another update")
	}
}

// Exercise the real notification serializer rather than only its typed input.
func TestModelControlsScannerEmptyAttributionNotification(t *testing.T) {
	var wire bytes.Buffer
	d := &daemon{codec: NewCodec(&wire)}
	d.emit(noderec.NotifyNodeUpdated, noderec.DirectoryNode{
		HostUUID: "empty-attribution", Models: []string{"good"},
		ModelsByEngine: map[string][]string{},
	})
	var frame struct {
		Params noderec.NodeEvent `json:"params"`
	}
	if err := json.Unmarshal(wire.Bytes(), &frame); err != nil {
		t.Fatalf("decode actual emitted node event: %v", err)
	}
	if frame.Params.Node.ModelsByEngine == nil || len(frame.Params.Node.EngineModels("ollama")) != 0 || !reflect.DeepEqual(frame.Params.Node.Models, []string{"good"}) {
		t.Fatal("notification serialization lost authoritative empty attribution and enabled flat fallback")
	}
}

// The model-inventory half of the periodic refresh loop (daemon.refreshPeersLoop) converges
// a peer's DirectoryNode.Models / ModelsByEngine to its live engine-manager
// /v1/models even when no mDNS change fires. These unit tests drive the loop's
// per-node/sweep methods directly (nil codec, since emit is nil-safe) against a
// stub /v1/models server, asserting directory + cache state and the emit
// decision (refreshNodeModels' bool return).

// modelsStub is a swappable /v1/models handler: a test can change the status and
// body between refresh calls to simulate a pull, a delete, or a transient
// failure without touching any mDNS record.
type modelsStub struct {
	mu     sync.Mutex
	status int
	body   map[string]any
}

func (s *modelsStub) set(status int, body map[string]any) {
	s.mu.Lock()
	s.status = status
	s.body = body
	s.mu.Unlock()
}

func (s *modelsStub) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		s.mu.Lock()
		status, body := s.status, s.body
		s.mu.Unlock()
		if status == 0 {
			status = http.StatusOK
		}
		w.WriteHeader(status)
		if body != nil {
			_ = json.NewEncoder(w).Encode(body)
		}
	}
}

// startModelsStub stands up a stub engine-manager em surface and returns it plus
// the loopback port the daemon should dial.
func startModelsStub(t *testing.T, status int, body map[string]any) (*modelsStub, int) {
	t.Helper()
	stub := &modelsStub{}
	stub.set(status, body)
	srv := httptest.NewServer(stub.handler())
	t.Cleanup(srv.Close)
	return stub, portFromURL(t, srv.URL)
}

func portFromURL(t *testing.T, raw string) int {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse stub url %q: %v", raw, err)
	}
	p, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatalf("stub port from %q: %v", raw, err)
	}
	return p
}

// newRefreshTestDaemon builds a daemon wired for the refresh path only: a real
// directory + modelsHTTP client and initialized caches, no codec/responder (emit
// is nil-safe and the refresh path never advertises). The registry is stamped
// with a self uuid that no peer test seeds, so refreshNodeModels dials each peer
// at its advertised address (the self-loopback branch is exercised separately in
// TestRefreshSelfDialsLoopback).
func newRefreshTestDaemon() *daemon {
	return &daemon{
		reg:                newRegistry("self-node-uuid", "", selfAddrs("127.0.0.1")),
		dir:                newDirectory(),
		modelsHTTP:         &http.Client{Timeout: modelsFetchTimeout},
		lastInfo:           make(map[string]NodeInfoResponse),
		lastInfoAt:         make(map[string]time.Time),
		lastModels:         make(map[string][]string),
		lastModelsByEngine: make(map[string]map[string][]string),
		lastLoadedByEngine: make(map[string]map[string][]string),
	}
}

// seedEMNode adds a directory entry advertising engine-manager (em) at ip:port,
// with no model inventory yet (the state before any enrichment).
func seedEMNode(d *daemon, hostUUID, ip string, emPort int) {
	d.dir.upsert(noderec.DirectoryNode{
		HostUUID: hostUUID,
		Name:     hostUUID,
		IP:       ip,
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{
			noderec.ServiceEngineManager: {Port: emPort},
		},
	})
}

// TestRefreshPopulatesInitialEmptyInventory covers the primary bug: a known em
// peer whose first enrichment returned nothing converges once its engine serves
// models, with no mDNS record change — a single refresh fetch populates the
// directory and reports a change.
func TestRefreshPopulatesInitialEmptyInventory(t *testing.T) {
	d := newRefreshTestDaemon()
	_, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models":         []string{"llama3:8b", "qwen:0.5b"},
		"modelsByEngine": map[string][]string{"ollama": {"llama3:8b"}, "lmstudio": {"qwen:0.5b"}},
	})
	seedEMNode(d, "peer-A", "127.0.0.1", port)

	if changed := d.refreshNodeModels("peer-A", "127.0.0.1", port, ""); !changed {
		t.Fatal("initial empty inventory should report a change once populated")
	}
	got, _ := d.dir.get("peer-A")
	if want := []string{"llama3:8b", "qwen:0.5b"}; !reflect.DeepEqual(got.Models, want) {
		t.Errorf("Models = %v, want %v", got.Models, want)
	}
	if want := map[string][]string{"ollama": {"llama3:8b"}, "lmstudio": {"qwen:0.5b"}}; !reflect.DeepEqual(got.ModelsByEngine, want) {
		t.Errorf("ModelsByEngine = %v, want %v", got.ModelsByEngine, want)
	}
}

func TestRefreshModelsFallsBackToASecondPublishedAddress(t *testing.T) {
	d := newRefreshTestDaemon()
	_, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models": []string{"reachable-model"},
	})
	const peerUUID = "peer-multihomed-models"
	d.dir.upsert(noderec.DirectoryNode{
		HostUUID: peerUUID,
		Name:     peerUUID,
		IP:       "127.0.0.2",
		IPs:      []string{"127.0.0.2", "127.0.0.1"},
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{
			noderec.ServiceEngineManager: {Port: port},
		},
	})

	if !d.refreshNodeModelsCandidates(peerUUID, "127.0.0.2", []string{"127.0.0.2", "127.0.0.1"}, port, "") {
		t.Fatal("model inventory did not refresh through the second published address")
	}
	got, _ := d.dir.get(peerUUID)
	if want := []string{"reachable-model"}; !reflect.DeepEqual(got.Models, want) {
		t.Fatalf("Models = %v, want %v", got.Models, want)
	}
}

// TestRefreshSelfDialsLoopback is the regression for the periodic model
// refresh: our OWN node must be dialed over loopback, not its advertised LAN
// ip=, so a pull/delete/late-engine-start keeps converging after startup even
// when inbound to the LAN address is firewalled (the Windows failure this
// addresses). The stub listens ONLY on loopback while the self entry advertises
// an unreachable LAN address, so the fetch can only succeed via the loopback
// dial. The stored (advertised) address is still used as applyModels' stale-
// address guard, so it must be preserved.
func TestRefreshSelfDialsLoopback(t *testing.T) {
	d := newRefreshTestDaemon() // reg self uuid = "self-node-uuid"
	_, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models":         []string{"llama3:8b"},
		"modelsByEngine": map[string][]string{"ollama": {"llama3:8b"}},
	})
	// TEST-NET-1 (RFC 5737): guaranteed non-routable — only a loopback dial can
	// reach the stub, which httptest binds on 127.0.0.1:port.
	const unreachableLAN = "192.0.2.1"
	seedEMNode(d, "self-node-uuid", unreachableLAN, port)

	if changed := d.refreshNodeModels("self-node-uuid", unreachableLAN, port, ""); !changed {
		t.Fatal("self model refresh should converge via loopback despite an unreachable advertised LAN address")
	}
	got, _ := d.dir.get("self-node-uuid")
	if want := []string{"llama3:8b"}; !reflect.DeepEqual(got.Models, want) {
		t.Errorf("self Models = %v, want %v (must be dialed over loopback)", got.Models, want)
	}
	if got.IP != unreachableLAN {
		t.Errorf("self IP = %q, want the advertised address %q preserved for the stale-address guard", got.IP, unreachableLAN)
	}
}

// TestRefreshPullDeleteAndUnchanged covers three required behaviors together: a
// pull and a delete each change the inventory and emit exactly one update, while
// an unchanged inventory emits nothing (no repeated updates).
func TestRefreshPullDeleteAndUnchanged(t *testing.T) {
	d := newRefreshTestDaemon()
	stub, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models":         []string{"a"},
		"modelsByEngine": map[string][]string{"ollama": {"a"}},
	})
	seedEMNode(d, "peer-B", "127.0.0.1", port)

	// Initial populate is a change.
	if !d.refreshNodeModels("peer-B", "127.0.0.1", port, "") {
		t.Fatal("initial populate should report a change")
	}
	// Unchanged inventory: no further change events.
	if d.refreshNodeModels("peer-B", "127.0.0.1", port, "") {
		t.Error("unchanged inventory should not report a change")
	}

	// Pull: add a model -> exactly one change.
	stub.set(http.StatusOK, map[string]any{
		"models":         []string{"a", "b"},
		"modelsByEngine": map[string][]string{"ollama": {"a", "b"}},
	})
	if !d.refreshNodeModels("peer-B", "127.0.0.1", port, "") {
		t.Fatal("a pull should report a change")
	}
	if d.refreshNodeModels("peer-B", "127.0.0.1", port, "") {
		t.Error("re-reading the same post-pull inventory should not report a change")
	}

	// Delete: remove a model -> exactly one change.
	stub.set(http.StatusOK, map[string]any{
		"models":         []string{"a"},
		"modelsByEngine": map[string][]string{"ollama": {"a"}},
	})
	if !d.refreshNodeModels("peer-B", "127.0.0.1", port, "") {
		t.Fatal("a delete should report a change")
	}
	if d.refreshNodeModels("peer-B", "127.0.0.1", port, "") {
		t.Error("re-reading the same post-delete inventory should not report a change")
	}
	got, _ := d.dir.get("peer-B")
	if want := []string{"a"}; !reflect.DeepEqual(got.Models, want) {
		t.Errorf("final Models = %v, want %v", got.Models, want)
	}

	// Deleting the final model carries an explicit empty per-engine inventory.
	// It must clear the stale model once and then remain stable.
	stub.set(http.StatusOK, map[string]any{
		"models":         []string{},
		"modelsByEngine": map[string][]string{"ollama": {}},
	})
	if !d.refreshNodeModels("peer-B", "127.0.0.1", port, "") {
		t.Fatal("deleting the last model should report a change")
	}
	if d.refreshNodeModels("peer-B", "127.0.0.1", port, "") {
		t.Error("re-reading the same explicit empty inventory should not report a change")
	}
	got, _ = d.dir.get("peer-B")
	if len(got.Models) != 0 {
		t.Errorf("Models after last delete = %v, want empty", got.Models)
	}
	if want := map[string][]string{"ollama": {}}; !reflect.DeepEqual(got.ModelsByEngine, want) {
		t.Errorf("ModelsByEngine after last delete = %v, want %v", got.ModelsByEngine, want)
	}
}

// TestRefreshReorderOnlyIsNotAChange asserts the compare is order-insensitive: a
// response that returns the same models in a different order is not treated as a
// change (no spurious update).
func TestRefreshReorderOnlyIsNotAChange(t *testing.T) {
	d := newRefreshTestDaemon()
	stub, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models":         []string{"a", "b"},
		"modelsByEngine": map[string][]string{"ollama": {"a", "b"}},
	})
	seedEMNode(d, "peer-R", "127.0.0.1", port)
	if !d.refreshNodeModels("peer-R", "127.0.0.1", port, "") {
		t.Fatal("initial populate should report a change")
	}
	stub.set(http.StatusOK, map[string]any{
		"models":         []string{"b", "a"},
		"modelsByEngine": map[string][]string{"ollama": {"b", "a"}},
	})
	if d.refreshNodeModels("peer-R", "127.0.0.1", port, "") {
		t.Error("a mere reordering of the same models should not report a change")
	}
}

// TestRefreshFetchFailureRetainsInventory covers requirement 5: a transient
// fetch failure preserves the last successful flat and per-engine inventories
// (no blanking) and reports no change.
func TestRefreshFetchFailureRetainsInventory(t *testing.T) {
	d := newRefreshTestDaemon()
	stub, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models":         []string{"keep-me"},
		"modelsByEngine": map[string][]string{"ollama": {"keep-me"}},
	})
	seedEMNode(d, "peer-C", "127.0.0.1", port)
	if !d.refreshNodeModels("peer-C", "127.0.0.1", port, "") {
		t.Fatal("initial populate should report a change")
	}

	// Now the endpoint fails.
	stub.set(http.StatusInternalServerError, nil)
	if d.refreshNodeModels("peer-C", "127.0.0.1", port, "") {
		t.Error("a failed fetch should not report a change")
	}
	got, _ := d.dir.get("peer-C")
	if want := []string{"keep-me"}; !reflect.DeepEqual(got.Models, want) {
		t.Errorf("Models after failure = %v, want retained %v", got.Models, want)
	}
	if want := map[string][]string{"ollama": {"keep-me"}}; !reflect.DeepEqual(got.ModelsByEngine, want) {
		t.Errorf("ModelsByEngine after failure = %v, want retained %v", got.ModelsByEngine, want)
	}
	d.infoMu.Lock()
	cachedFlat := d.lastModels["peer-C"]
	cachedByEngine := d.lastModelsByEngine["peer-C"]
	d.infoMu.Unlock()
	if want := []string{"keep-me"}; !reflect.DeepEqual(cachedFlat, want) {
		t.Errorf("cached flat inventory = %v, want retained %v", cachedFlat, want)
	}
	if want := map[string][]string{"ollama": {"keep-me"}}; !reflect.DeepEqual(cachedByEngine, want) {
		t.Errorf("cached per-engine inventory = %v, want retained %v", cachedByEngine, want)
	}
}

// TestRefreshPartialModelsByEngine covers requirement 4: a partial
// modelsByEngine response (attribution covering fewer models than the flat
// union, as a mixed-version peer might send) is propagated exactly as returned.
func TestRefreshPartialModelsByEngine(t *testing.T) {
	d := newRefreshTestDaemon()
	_, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models":         []string{"x", "y"},
		"modelsByEngine": map[string][]string{"ollama": {"x"}},
	})
	seedEMNode(d, "peer-D", "127.0.0.1", port)
	if !d.refreshNodeModels("peer-D", "127.0.0.1", port, "") {
		t.Fatal("initial populate should report a change")
	}
	got, _ := d.dir.get("peer-D")
	if want := []string{"x", "y"}; !reflect.DeepEqual(got.Models, want) {
		t.Errorf("Models = %v, want %v", got.Models, want)
	}
	if want := map[string][]string{"ollama": {"x"}}; !reflect.DeepEqual(got.ModelsByEngine, want) {
		t.Errorf("ModelsByEngine = %v, want the partial map propagated verbatim %v", got.ModelsByEngine, want)
	}
}

// TestRefreshPopulatesLoadedByEngine covers Phase 2: a peer's loadedByEngine is
// carried onto the directory node from the same /v1/models fetch.
func TestRefreshPopulatesLoadedByEngine(t *testing.T) {
	d := newRefreshTestDaemon()
	_, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models":         []string{"llama3:8b", "qwen:0.5b"},
		"modelsByEngine": map[string][]string{"ollama": {"llama3:8b"}, "lmstudio": {"qwen:0.5b"}},
		"loadedByEngine": map[string][]string{"ollama": {"llama3:8b"}, "lmstudio": {}},
	})
	seedEMNode(d, "peer-L", "127.0.0.1", port)

	if !d.refreshNodeModels("peer-L", "127.0.0.1", port, "") {
		t.Fatal("initial populate should report a change")
	}
	got, _ := d.dir.get("peer-L")
	if want := map[string][]string{"ollama": {"llama3:8b"}, "lmstudio": {}}; !reflect.DeepEqual(got.LoadedByEngine, want) {
		t.Errorf("LoadedByEngine = %v, want %v", got.LoadedByEngine, want)
	}
}

// TestRefreshLoadedOnlyChangeReportsChange covers the loaded-set reconcile: when
// only loadedByEngine changes (a JIT load / TTL eviction with the same installed
// list), the refresh reports a change and updates the field so remote cards stay
// live.
func TestRefreshLoadedOnlyChangeReportsChange(t *testing.T) {
	d := newRefreshTestDaemon()
	stub, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models":         []string{"a"},
		"modelsByEngine": map[string][]string{"ollama": {"a"}},
		"loadedByEngine": map[string][]string{"ollama": {}},
	})
	seedEMNode(d, "peer-M", "127.0.0.1", port)
	if !d.refreshNodeModels("peer-M", "127.0.0.1", port, "") {
		t.Fatal("initial populate should report a change")
	}

	// Same installed list, but the model is now resident -> a change.
	stub.set(http.StatusOK, map[string]any{
		"models":         []string{"a"},
		"modelsByEngine": map[string][]string{"ollama": {"a"}},
		"loadedByEngine": map[string][]string{"ollama": {"a"}},
	})
	if !d.refreshNodeModels("peer-M", "127.0.0.1", port, "") {
		t.Fatal("a loaded-set change should report a change even when the model list is unchanged")
	}
	got, _ := d.dir.get("peer-M")
	if want := map[string][]string{"ollama": {"a"}}; !reflect.DeepEqual(got.LoadedByEngine, want) {
		t.Errorf("LoadedByEngine after load = %v, want %v", got.LoadedByEngine, want)
	}

	// A mere reordering of the same loaded set is not a change.
	stub.set(http.StatusOK, map[string]any{
		"models":         []string{"a"},
		"modelsByEngine": map[string][]string{"ollama": {"a"}},
		"loadedByEngine": map[string][]string{"ollama": {"a"}},
	})
	if d.refreshNodeModels("peer-M", "127.0.0.1", port, "") {
		t.Error("an identical loaded set should not report a change")
	}
}

// TestRefreshLoadedRetainedOnFailure confirms a transient fetch failure preserves
// the last successful loadedByEngine rather than blanking it.
func TestRefreshLoadedRetainedOnFailure(t *testing.T) {
	d := newRefreshTestDaemon()
	stub, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models":         []string{"keep-me"},
		"modelsByEngine": map[string][]string{"ollama": {"keep-me"}},
		"loadedByEngine": map[string][]string{"ollama": {"keep-me"}},
	})
	seedEMNode(d, "peer-N", "127.0.0.1", port)
	if !d.refreshNodeModels("peer-N", "127.0.0.1", port, "") {
		t.Fatal("initial populate should report a change")
	}

	stub.set(http.StatusInternalServerError, nil)
	if d.refreshNodeModels("peer-N", "127.0.0.1", port, "") {
		t.Error("a failed fetch should not report a change")
	}
	got, _ := d.dir.get("peer-N")
	if want := map[string][]string{"ollama": {"keep-me"}}; !reflect.DeepEqual(got.LoadedByEngine, want) {
		t.Errorf("LoadedByEngine after failure = %v, want retained %v", got.LoadedByEngine, want)
	}
	d.infoMu.Lock()
	cachedLoaded := d.lastLoadedByEngine["peer-N"]
	d.infoMu.Unlock()
	if want := map[string][]string{"ollama": {"keep-me"}}; !reflect.DeepEqual(cachedLoaded, want) {
		t.Errorf("cached loaded inventory = %v, want retained %v", cachedLoaded, want)
	}
}

// TestRefreshPolicyAdvertisement covers the policy advertisement on the
// periodic refresh: a revision change alone is a change, an identical one is
// not, a failed fetch keeps the last good value, and a successful answer
// without one makes the node undeclared.
func TestRefreshPolicyAdvertisement(t *testing.T) {
	d := newRefreshTestDaemon()
	inventory := func(revision string) map[string]any {
		return map[string]any{
			"models": []string{"a"},
			"policy": map[string]any{"revision": revision, "roles": map[string]any{"weak": map[string]any{"engine": "ollama", "model": "a"}}},
		}
	}
	stub, port := startModelsStub(t, http.StatusOK, inventory("1"))
	seedEMNode(d, "peer-P", "127.0.0.1", port)
	if !d.refreshNodeModels("peer-P", "127.0.0.1", port, "") {
		t.Fatal("initial populate should report a change")
	}
	want := &nodepolicy.Advertisement{Revision: "1", Roles: map[string]nodepolicy.AdvertisedRole{nodepolicy.TierWeak: {Engine: "ollama", Model: "a"}}}
	if got, _ := d.dir.get("peer-P"); !reflect.DeepEqual(got.Policy, want) {
		t.Fatalf("Policy = %+v, want %+v", got.Policy, want)
	}
	if d.refreshNodeModels("peer-P", "127.0.0.1", port, "") {
		t.Error("an identical advertisement should not report a change")
	}

	stub.set(http.StatusOK, inventory("2"))
	if !d.refreshNodeModels("peer-P", "127.0.0.1", port, "") {
		t.Fatal("a policy revision change should report a change even when the models are unchanged")
	}
	want.Revision = "2"

	stub.set(http.StatusInternalServerError, nil)
	if d.refreshNodeModels("peer-P", "127.0.0.1", port, "") {
		t.Error("a failed fetch should not report a change")
	}
	if got, _ := d.dir.get("peer-P"); !reflect.DeepEqual(got.Policy, want) {
		t.Errorf("Policy after failure = %+v, want retained %+v", got.Policy, want)
	}
	d.infoMu.Lock()
	cached := d.lastPolicy["peer-P"]
	d.infoMu.Unlock()
	if !reflect.DeepEqual(cached, want) {
		t.Errorf("cached policy = %+v, want retained %+v", cached, want)
	}

	stub.set(http.StatusOK, map[string]any{"models": []string{"a"}})
	if !d.refreshNodeModels("peer-P", "127.0.0.1", port, "") {
		t.Fatal("an answer without a policy should report a change")
	}
	if got, _ := d.dir.get("peer-P"); got.Policy != nil {
		t.Errorf("Policy after an answer without one = %+v, want nil", got.Policy)
	}
}

// TestApplyModelsGuard covers requirement 6 at the directory layer: a completed
// fetch is discarded (ok == false) when the node was removed or re-addressed
// while it was in flight, so it can neither resurrect a removed node nor
// overwrite a re-addressed one.
func TestApplyModelsGuard(t *testing.T) {
	d := newDirectory()

	// Removed mid-sweep: node absent -> not resurrected.
	if _, changed, ok := d.applyModels("ghost", "127.0.0.1", 14322, modelInventory{models: []string{"a"}}); ok || changed {
		t.Errorf("applyModels on an absent node = (changed %v, ok %v), want (false, false)", changed, ok)
	}
	if _, present := d.get("ghost"); present {
		t.Error("applyModels must not resurrect a removed node")
	}

	// Re-addressed mid-sweep: node present but IP changed -> not overwritten.
	d.upsert(noderec.DirectoryNode{
		HostUUID: "peer-E",
		IP:       "10.0.0.9",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineManager: {Port: 14322}},
		Models:   []string{"old"},
	})
	if _, changed, ok := d.applyModels("peer-E", "127.0.0.1", 14322, modelInventory{models: []string{"new"}}); ok || changed {
		t.Errorf("applyModels with a stale IP = (changed %v, ok %v), want (false, false)", changed, ok)
	}
	// em port changed -> also discarded.
	if _, changed, ok := d.applyModels("peer-E", "10.0.0.9", 99999, modelInventory{models: []string{"new"}}); ok || changed {
		t.Errorf("applyModels with a stale em port = (changed %v, ok %v), want (false, false)", changed, ok)
	}
	got, _ := d.get("peer-E")
	if want := []string{"old"}; !reflect.DeepEqual(got.Models, want) {
		t.Errorf("Models after guarded rejects = %v, want unchanged %v", got.Models, want)
	}
}

// TestRefreshNodeModelsRemovedMidSweep exercises the guard through the daemon's
// per-node path: a node evicted between the sweep snapshot and the fetch's
// completion is not resurrected in the directory (the requirement-6 guarantee).
// The cache may hold the fetched value (it is written before the directory
// guard, as last-known-good), so this asserts only the directory invariant.
func TestRefreshNodeModelsRemovedMidSweep(t *testing.T) {
	d := newRefreshTestDaemon()
	_, port := startModelsStub(t, http.StatusOK, map[string]any{
		"models": []string{"a"},
	})
	// The node is not in the directory (simulating removal before the in-flight
	// fetch completes); the fetch succeeds but must not be applied to the
	// directory.
	if d.refreshNodeModels("gone", "127.0.0.1", port, "") {
		t.Error("a fetch for a removed node should report no change")
	}
	if _, present := d.dir.get("gone"); present {
		t.Error("an in-flight result must not resurrect a removed node")
	}
}

// TestRefreshOnceConcurrentSlowPeers covers requirement 7: multiple slow peers
// are refreshed concurrently, so one slow peer does not consume another's
// timeout budget. It proves concurrency deterministically with an entry barrier
// rather than a wall-clock threshold: both stub handlers must be ENTERED before
// either is released. A serial sweep would block inside the first handler and
// never enter the second, so the barrier wait fails fast. The time.After bounds
// are failure ceilings, not success-timing assertions, so the test isn't flaky.
func TestRefreshOnceConcurrentSlowPeers(t *testing.T) {
	const peers = 2 // must stay <= modelsRefreshConcurrency
	entered := make(chan struct{}, peers)
	release := make(chan struct{})
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		entered <- struct{}{}
		<-release // hold until both peers are concurrently in-flight
		_ = json.NewEncoder(w).Encode(map[string]any{"models": []string{"m"}})
	}
	s1 := httptest.NewServer(http.HandlerFunc(handler))
	s2 := httptest.NewServer(http.HandlerFunc(handler))
	t.Cleanup(s1.Close)
	t.Cleanup(s2.Close)

	d := newRefreshTestDaemon()
	seedEMNode(d, "peer-1", "127.0.0.1", portFromURL(t, s1.URL))
	seedEMNode(d, "peer-2", "127.0.0.1", portFromURL(t, s2.URL))

	done := make(chan struct{})
	go func() { d.refreshModelsOnce(); close(done) }()

	// Both handlers must be entered before either is released; only a concurrent
	// sweep can reach this point.
	for i := 0; i < peers; i++ {
		select {
		case <-entered:
		case <-time.After(5 * time.Second):
			t.Fatalf("only %d/%d peers entered concurrently; sweep is not concurrent", i, peers)
		}
	}
	close(release)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("refreshModelsOnce did not complete after releasing peers")
	}
	if got, _ := d.dir.get("peer-1"); !reflect.DeepEqual(got.Models, []string{"m"}) {
		t.Errorf("peer-1 Models = %v, want [m]", got.Models)
	}
	if got, _ := d.dir.get("peer-2"); !reflect.DeepEqual(got.Models, []string{"m"}) {
		t.Errorf("peer-2 Models = %v, want [m]", got.Models)
	}
}
