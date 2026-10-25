// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestModelControlsRoutingAcrossProcesses(t *testing.T) {
	concrete := strings.Repeat("a", 512)
	overlong := strings.Repeat("b", 513)
	multibyte512 := strings.Repeat("界", 170) + "ab"
	multibyte513 := strings.Repeat("界", 171)
	var hits atomic.Int32
	var catalogReads atomic.Int32
	var selected atomic.Value
	var catalog atomic.Value
	catalog.Store(`{"data":[{"id":"first"},{"id":"raw-` + string([]byte{0xff}) + `"},{"id":"bad\u0000"},{"id":"bad\u001f"},{"id":"bad\u007f"},{"id":"bad\u0080"},{"id":"bad\u009f"},{"id":""},{"id":"` + concrete + `"},{"id":"` + overlong + `"},{"id":"` + multibyte512 + `"},{"id":"` + multibyte513 + `"},{"id":"last"}]}`)
	upstream := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet && r.URL.Path == "/health" {
			// The broker's private configured local engine stays unavailable;
			// inference below uses only the explicit synthetic manual overlay.
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		if r.Method == http.MethodGet && r.URL.Path == "/models" {
			catalogReads.Add(1)
			if _, err := io.WriteString(w, catalog.Load().(string)); err != nil {
				t.Errorf("write raw model catalog: %v", err)
			}
			return
		}
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "unexpected fixture route", http.StatusBadRequest)
			return
		}
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode selected concrete model: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		selected.Store(body.Model)
		hits.Add(1)
		if _, err := io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`); err != nil {
			t.Errorf("write positive inference response: %v", err)
		}
	}))
	t.Cleanup(upstream.Close)
	f := newProxyServiceFixture(t)
	f.withInertEngineManager()
	scannerPort := freePort(t)
	f.env = append(f.env, "NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(scannerPort))
	// Private manager manifests do not override the broker's default llama
	// backend port. Reuse the existing explicit-journal fixture contract before
	// startup so its health loop cannot contact the installed port 8080.
	jointWriteJSON(t, filepath.Join(filepath.Dir(f.clusterDir), "engine-settings-operations.json"), map[string]any{
		"llamacpp": map[string]any{
			"explicit": true,
			"snapshot": map[string]any{
				"engine": "llamacpp", "revision": 1, "appliedRevision": 1,
				"phase": "idle", "format": "pair-arguments-v1",
				"settings": map[string]any{"serverPort": portOfURL(t, upstream.URL), "proxyPort": f.facadePort, "launchText": ""},
			},
		},
	})
	f.start()
	c := f.readyClient()
	policy := servicePolicy(t, c)
	policy.Tiers["weak"] = []nodepolicy.TierEntry{{Kind: nodepolicy.EntryLocal, Engine: "llamacpp", Model: concrete}}
	policy.Tiers["medium"] = []nodepolicy.TierEntry{{Kind: nodepolicy.EntryLocal, Engine: "llamacpp", Model: multibyte512}}
	policy = setServicePolicy(t, c, policy)
	nextID := 30400
	call := func(t *testing.T, method string, params any) jsonrpc.Message {
		t.Helper()
		nextID++
		m := c.mustCall(t, strconv.Itoa(nextID), method, mustMarshal(t, params))
		return *m
	}
	egressProcessActivePolicy(t, call, policy.Revision)
	models := []string{"first", "bad\x00", "bad\x1f", "bad\x7f", "bad\u0080", "bad\u009f", "", concrete, overlong, multibyte512, multibyte513, "last"}
	event := noderec.NodeEvent{Node: noderec.DirectoryNode{
		HostUUID: "j304-private-owner", Name: "j304-private-owner", IP: "127.0.0.1",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceLlamaCPP: {Port: portOfURL(t, upstream.URL)}},
		Models:   models, ModelsByEngine: map[string][]string{"llamacpp": models, "ollama": {}},
		LoadedByEngine: map[string][]string{"llamacpp": models},
		Policy: &nodepolicy.Advertisement{Revision: "304", Roles: map[string]nodepolicy.AdvertisedRole{
			"weak": {Engine: "llamacpp", Model: concrete}, "medium": {Engine: "llamacpp", Model: multibyte512}, "strong": {Engine: "llamacpp", Model: overlong},
		}},
	}}
	client := &http.Client{Timeout: serviceCallTimeout}
	t.Cleanup(client.CloseIdleConnections)
	publish := func(method string) {
		t.Helper()
		frame := jsonrpc.Message{JSONRPC: "2.0", Method: method, Params: mustMarshal(t, event)}
		egressProcessHTTP(t, client, http.MethodPost, scannerPort, "/", mustMarshal(t, frame), http.StatusOK)
	}
	publish(noderec.NotifyNodeDiscovered)
	want := []string{"first", concrete, multibyte512, "last"}
	awaitModels := func(expected []string) {
		t.Helper()
		neEventually(t, "relayed sanitized inventory reaches proxy", serviceCallTimeout, func() bool {
			m := call(t, "llamacpp-proxy:nodes/list", map[string]any{})
			var reply struct {
				Nodes []struct {
					ID     string   `json:"id"`
					Models []string `json:"models"`
				} `json:"nodes"`
			}
			egressProcessDecode(t, m.Result, &reply)
			for _, node := range reply.Nodes {
				if node.ID == event.Node.HostUUID {
					return len(node.Models) == len(expected) && (len(expected) == 0 || reflect.DeepEqual(node.Models, expected))
				}
			}
			return false
		})
	}
	awaitModels(want)
	// The existing fixture's explicit manual overlay permits plaintext only to
	// this checked private listener. The subscribed role metadata stays real.
	manual := func(inventory []string) {
		call(t, "llamacpp-proxy:node/add-manual", map[string]any{"id": event.Node.HostUUID, "host": "127.0.0.1", "port": portOfURL(t, upstream.URL), "addresses": []string{"127.0.0.1"}, "models": inventory})
	}
	manual(models)
	awaitModels(want)
	t.Run("client catalog retains valid siblings and aliases", func(t *testing.T) {
		before := catalogReads.Load()
		raw := egressProcessHTTP(t, client, http.MethodGet, f.facadePort, "/v1/models", nil, http.StatusOK)
		if catalogReads.Load() <= before {
			t.Fatal("client catalog did not read the owned raw inventory")
		}
		var reply struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if err := json.Unmarshal(raw, &reply); err != nil {
			t.Fatalf("decode client catalog: %v", err)
		}
		var ids []string
		for _, entry := range reply.Data {
			ids = append(ids, entry.ID)
		}
		expected := append(append([]string{}, want...), "weak", "weak-api", "weak-local", "medium", "medium-api", "medium-local")
		if !reflect.DeepEqual(ids, expected) {
			t.Fatal("client catalog lost valid ordering/aliases or retained invalid identifiers")
		}
	})
	t.Run("short alias selects concrete 512 bytes", func(t *testing.T) {
		egressProcessHTTP(t, client, http.MethodPost, f.facadePort, "/v1/chat/completions", []byte(`{"model":"weak","messages":[],"stream":false}`), http.StatusOK)
		if hits.Load() != 1 || selected.Load() != concrete {
			t.Fatal("short alias did not deliver the exact 512-byte concrete model")
		}
	})
	t.Run("short alias selects multibyte concrete 512 bytes", func(t *testing.T) {
		egressProcessHTTP(t, client, http.MethodPost, f.facadePort, "/v1/chat/completions", []byte(`{"model":"medium","messages":[],"stream":false}`), http.StatusOK)
		if hits.Load() != 2 || selected.Load() != multibyte512 {
			t.Fatal("short alias did not deliver the exact multibyte 512-byte model")
		}
	})
	t.Run("direct ingress keeps 256 byte cap", func(t *testing.T) {
		egressProcessHTTP(t, client, http.MethodPost, f.facadePort, "/v1/chat/completions", mustMarshal(t, map[string]any{"model": concrete, "messages": []any{}, "stream": false}), http.StatusBadRequest)
		if hits.Load() != 2 {
			t.Fatal("over-ingress-cap concrete request reached upstream")
		}
	})
	t.Run("successful all invalid inventory removes selection", func(t *testing.T) {
		event.Node.Models = []string{overlong, multibyte513, "bad\x7f"}
		event.Node.ModelsByEngine = map[string][]string{"llamacpp": event.Node.Models}
		event.Node.LoadedByEngine = map[string][]string{"llamacpp": event.Node.Models}
		event.Node.Policy = &nodepolicy.Advertisement{Revision: "305", Roles: map[string]nodepolicy.AdvertisedRole{"weak": {Engine: "llamacpp", Model: overlong}}}
		publish(noderec.NotifyNodeUpdated)
		manual(event.Node.Models)
		awaitModels(nil)
		before := hits.Load()
		egressProcessHTTP(t, client, http.MethodPost, f.facadePort, "/v1/chat/completions", []byte(`{"model":"weak","messages":[],"stream":false}`), http.StatusNotFound)
		if hits.Load() != before {
			t.Fatal("all-invalid inventory allowed tier inference upstream")
		}
		egressProcessHTTP(t, client, http.MethodPost, f.facadePort, "/v1/chat/completions", []byte(`{"model":"first","messages":[],"stream":false}`), http.StatusBadGateway)
		if hits.Load() != before || hits.Load() != 2 {
			t.Fatal("all-invalid inventory allowed concrete inference upstream")
		}
	})
	t.Run("all invalid client catalog retains useful aliases", func(t *testing.T) {
		catalog.Store(`{"data":[{"id":"raw-` + string([]byte{0xff}) + `"},{"id":""},{"id":"bad\u007f"},{"id":"` + overlong + `"},{"id":"` + multibyte513 + `"}]}`)
		before := catalogReads.Load()
		raw := egressProcessHTTP(t, client, http.MethodGet, f.facadePort, "/v1/models", nil, http.StatusOK)
		if catalogReads.Load() <= before {
			t.Fatal("all-invalid client catalog did not read the owned raw inventory")
		}
		var reply struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if err := json.Unmarshal(raw, &reply); err != nil {
			t.Fatalf("decode all-invalid catalog: %v", err)
		}
		var ids []string
		for _, entry := range reply.Data {
			ids = append(ids, entry.ID)
		}
		if !reflect.DeepEqual(ids, []string{"weak", "weak-api", "weak-local", "medium", "medium-api", "medium-local"}) {
			t.Fatal("all-invalid catalog leaked an invalid ID or removed configured aliases")
		}
	})
	t.Run("outer empty attribution suppresses flat fallback on both wires", func(t *testing.T) {
		event.Node.Models = []string{"good"}
		event.Node.ModelsByEngine = map[string][]string{}
		event.Node.LoadedByEngine = nil
		event.Node.Policy = nil
		publish(noderec.NotifyNodeUpdated)
		manual(event.Node.Models)
		awaitModels(nil)
		neEventually(t, "client JSON retains authoritative empty attribution", serviceCallTimeout, func() bool {
			m := call(t, "discovery:get-nodes", map[string]any{})
			var reply struct {
				Nodes []struct {
					HostUUID       string              `json:"hostUuid"`
					Models         []string            `json:"models"`
					ModelsByEngine map[string][]string `json:"modelsByEngine"`
				} `json:"nodes"`
			}
			egressProcessDecode(t, m.Result, &reply)
			for _, node := range reply.Nodes {
				if node.HostUUID == event.Node.HostUUID {
					return reflect.DeepEqual(node.Models, []string{"good"}) && node.ModelsByEngine != nil && len(node.ModelsByEngine) == 0
				}
			}
			return false
		})
		egressProcessHTTP(t, client, http.MethodPost, f.facadePort, "/v1/chat/completions", []byte(`{"model":"good","messages":[],"stream":false}`), http.StatusBadGateway)
		if hits.Load() != 2 {
			t.Fatal("present-empty attribution allowed flat-only model selection")
		}
	})
}

type routingUpstream struct {
	server *httptest.Server
	hits   atomic.Int32
}

func newRoutingUpstream(t *testing.T, status int) *routingUpstream {
	t.Helper()
	upstream := &routingUpstream{}
	upstream.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstream.hits.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		if status == http.StatusOK {
			_, _ = io.WriteString(w, `{"done":true,"choices":[]}`)
			return
		}
		_, _ = io.WriteString(w, `{"error":"model not found"}`)
	}))
	t.Cleanup(upstream.server.Close)
	return upstream
}

func TestStrictModelRoutingAcrossProcesses(t *testing.T) {
	if portBusy(11435) || portBusy(1234) {
		t.Skip("ollama-proxy (11435) or lmstudio-proxy (1234) default port already in use; skipping")
	}

	owner404 := newRoutingUpstream(t, http.StatusNotFound)
	ownerOK := newRoutingUpstream(t, http.StatusOK)
	ineligible := newRoutingUpstream(t, http.StatusOK)

	stdin, msgs, stderr, cleanup := startBrokerWith(t,
		"--proxy-path", proxyBin, "--proxy-engines", "ollama,lmstudio",
	)
	t.Cleanup(cleanup)
	go func() {
		for range stderr {
		}
	}()

	waitForMethod(t, msgs, "app:ready", 10*time.Second)
	ollamaPort := waitProxyReady(t, stdin, msgs, 15*time.Second)
	lmstudioPort := waitLMStudioProxyReady(t, stdin, msgs, 15*time.Second)

	type proxyCase struct {
		name      string
		rpcPrefix string
		path      string
		port      int
	}
	cases := []proxyCase{
		{name: "ollama", rpcPrefix: "ollama-proxy", path: "/api/chat", port: ollamaPort},
		{name: "ollama-anthropic", rpcPrefix: "ollama-proxy", path: "/v1/messages", port: ollamaPort},
		{name: "lmstudio", rpcPrefix: "lmstudio-proxy", path: "/v1/chat/completions", port: lmstudioPort},
		{name: "lmstudio-anthropic", rpcPrefix: "lmstudio-proxy", path: "/v1/messages", port: lmstudioPort},
	}
	client := &http.Client{Timeout: 5 * time.Second}
	t.Cleanup(client.CloseIdleConnections)
	snapshot := func() [3]int32 {
		return [3]int32{owner404.hits.Load(), ownerOK.hits.Load(), ineligible.hits.Load()}
	}

	for caseIndex, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			targetModel := "strict-routing-" + tc.name
			owner404ID := tc.name + "-owner-404"
			ownerOKID := tc.name + "-owner-ok"
			missingID := tc.name + "-known-missing"
			unknownID := tc.name + "-unknown"
			nodes := []struct {
				id     string
				port   int
				models []string
			}{
				{id: owner404ID, port: portOfURL(t, owner404.server.URL), models: []string{targetModel}},
				{id: ownerOKID, port: portOfURL(t, ownerOK.server.URL), models: []string{targetModel}},
				{id: missingID, port: portOfURL(t, ineligible.server.URL), models: []string{"different-model"}},
				{id: unknownID, port: portOfURL(t, ineligible.server.URL)},
			}

			requestID := 1000 + caseIndex*10
			for _, node := range nodes {
				params := map[string]any{
					"id":        node.id,
					"host":      "127.0.0.1",
					"port":      node.port,
					"addresses": []string{"127.0.0.1"},
				}
				if node.models != nil {
					params["models"] = node.models
				}
				callBrokerRPC(t, stdin, msgs, requestID, tc.rpcPrefix+":node/add-manual", params)
				requestID++
			}
			callBrokerRPC(t, stdin, msgs, requestID, tc.rpcPrefix+":node/set-priority", map[string]any{
				// Both facades share one proxy process, so every snapshot must
				// advance the process-wide generation.
				"generation": caseIndex + 1,
				"nodes":      []string{missingID, unknownID, owner404ID, ownerOKID},
			})

			before := snapshot()
			endpoint := fmt.Sprintf("http://127.0.0.1:%d%s", tc.port, tc.path)
			resp, err := client.Post(endpoint, "application/json",
				bytes.NewBufferString(fmt.Sprintf(`{"model":%q,"messages":[]}`, targetModel)))
			if err != nil {
				t.Fatalf("target-model request failed: %v", err)
			}
			body, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("target-model response = %d %s, want owner failover success", resp.StatusCode, body)
			}
			after := snapshot()
			if after[0] != before[0]+1 || after[1] != before[1]+1 || after[2] != before[2] {
				t.Fatalf("target-model hit deltas = %v -> %v, want owner 404/200 once and ineligible zero",
					before, after)
			}

			before = snapshot()
			resp, err = client.Post(endpoint, "application/json",
				bytes.NewBufferString(`{"model":"no-advertised-owner","messages":[]}`))
			if err != nil {
				t.Fatalf("ownerless request failed: %v", err)
			}
			body, _ = io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusBadGateway ||
				!strings.Contains(string(body), "no available node advertises the requested model") {
				t.Fatalf("ownerless response = %d %s, want actionable local 502", resp.StatusCode, body)
			}
			if after = snapshot(); after != before {
				t.Fatalf("ownerless request reached an upstream: hits %v -> %v", before, after)
			}
		})
	}
}
