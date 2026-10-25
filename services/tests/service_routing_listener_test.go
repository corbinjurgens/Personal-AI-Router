// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

// Real production entry points own the history construction at every hop. The
// upstream, scanner and service endpoint are the existing isolated fixtures;
// no history records are injected into the product for this test.
func TestRoutingHistoryProductionChain(t *testing.T) {
	f, service, upstream, listener := i7HybridFixture(t)
	a := f.rpc
	b := service.dial()
	b.notification(t, "app:ready")
	clients := []*serviceRPCClient{a, b}
	const rawID = `"j318-\u0061"`
	history := func(t *testing.T, c *serviceRPCClient, method string) nodepolicy.RoutingHistoryInitial {
		t.Helper()
		var initial nodepolicy.RoutingHistoryInitial
		neEventually(t, "history control returns an admitted baseline", 30*time.Second, func() bool {
			m, err := c.call(rawID, method, json.RawMessage(`{}`), serviceCallTimeout)
			if err != nil {
				t.Fatalf("%s: %v", method, err)
			}
			if m.ID == nil || string(*m.ID) != rawID {
				t.Fatalf("history reply changed raw string ID: %+v", m.ID)
			}
			if m.Error != nil {
				if m.Error.Code == jsonrpc.HistoryBusyCode {
					return false
				}
				t.Fatalf("%s: %+v", method, m.Error)
			}
			var diagnostic nodepolicy.RoutingHistoryDiagnostic
			initial, diagnostic, err = nodepolicy.DecodeRoutingHistoryInitial(m.Result)
			if err != nil || diagnostic.Count != 0 {
				t.Fatalf("history baseline: diagnostic=%+v error=%v", diagnostic, err)
			}
			return true
		})
		return initial
	}
	for _, c := range clients {
		if initial := history(t, c, nodepolicy.MethodRoutingSubscribe); len(initial.Requests) != 0 {
			t.Fatalf("fresh private service already has history: %+v", initial)
		}
	}
	row := func(t *testing.T, c *serviceRPCClient, key nodepolicy.RoutingRequestKey, accept func(nodepolicy.RoutingHistoryRecord) bool) nodepolicy.RoutingHistoryRecord {
		t.Helper()
		var result nodepolicy.RoutingHistoryRecord
		neEventually(t, "accepted production history reaches the local client", 30*time.Second, func() bool {
			for _, candidate := range history(t, c, nodepolicy.MethodRoutingGetInitial).Requests {
				if candidate.Request == key {
					result = candidate
					return accept(candidate)
				}
			}
			return false
		})
		return result
	}
	terminal := func(record nodepolicy.RoutingHistoryRecord) bool { return record.EndedAt != nil }
	push := func(t *testing.T, c *serviceRPCClient, key nodepolicy.RoutingRequestKey, minSequence int64) nodepolicy.RoutingHistoryRecord {
		t.Helper()
		var result nodepolicy.RoutingHistoryRecord
		var decodeErr error
		_, err := c.next(30*time.Second, func(m *jsonrpc.Message) bool {
			if !m.IsNotification() || m.Method != nodepolicy.NotifyRoutingChanged {
				return false
			}
			candidate, diagnostic, err := nodepolicy.DecodeRoutingHistoryRecord(m.Params)
			if err != nil || diagnostic.Count != 0 {
				decodeErr = fmt.Errorf("history push: diagnostic=%+v error=%v", diagnostic, err)
				return true // Leave next's locked predicate before failing the test.
			}
			if candidate.Request != key || !terminal(candidate) || candidate.Seq < minSequence {
				return false
			}
			result = candidate
			return true
		})
		if decodeErr != nil {
			t.Fatal(decodeErr)
		}
		if err != nil {
			t.Fatalf("terminal history push: %v", err)
		}
		return result
	}
	var run nodepolicy.RoutingRequestKey
	for _, route := range []struct {
		name, body string
		port       int
		hybrid     bool
		model      string
	}{
		{"hybrid listener", string(i7HybridBody()), listener, true, egressProcessModel},
		{"concrete facade", string(i7BuffersBody(128)), f.port, false, egressProcessModel},
		{"model-less facade", `{"messages":[],"stream":true}`, f.port, false, ""},
	} {
		t.Run(route.name, func(t *testing.T) {
			slot := upstream.slot(t, route.name)
			t.Cleanup(slot.release)
			upstream.mu.Lock()
			slot.hybrid = route.hybrid
			upstream.mu.Unlock()
			request := i7BuffersStart(t, f.client, route.port, "/v1/chat/completions", route.name, []byte(route.body))
			i7HybridAccepted(t, slot, request)
			egressProcessBarrier(t, request.firstByte, "client observed real upstream output")
			key := i7HybridActiveCount(t, f, 1)[0]
			if err := key.Validate(); err != nil {
				t.Fatal(err)
			}
			nonce, err := hex.DecodeString(key.ProxyRunID)
			if err != nil || len(nonce) != 16 || strings.ToLower(key.ProxyRunID) != key.ProxyRunID {
				t.Fatalf("production proxy run is not a 128-bit nonce: %+v", key)
			}
			if run.ProxyRunID == "" {
				run = key
			} else if key.OriginNodeID != run.OriginNodeID || key.ProxyRunID != run.ProxyRunID {
				t.Fatalf("facade and listener use different process identities: %+v %+v", run, key)
			}
			if route.hybrid && slot.strict.Request != key {
				t.Fatalf("destination strict request=%+v, history owner=%+v", slot.strict.Request, key)
			}
			open := row(t, a, key, func(r nodepolicy.RoutingHistoryRecord) bool { return r.EndedAt == nil && len(r.Attempts) == 1 })
			if open.Requested.Model != route.model && !route.hybrid {
				t.Fatalf("requested model=%q, want %q", open.Requested.Model, route.model)
			}
			slot.release()
			i7BuffersCompletedStream(t, request)
			i7HybridActiveCount(t, f, 0)
			for _, c := range clients {
				completed := row(t, c, key, terminal)
				if completed.State != "completed" || completed.Status == nil || *completed.Status != http.StatusOK || completed.Seq <= open.Seq || completed.Served == nil || len(completed.Attempts) != 1 {
					t.Fatalf("delivered history did not complete: %+v", completed)
				}
				destination := completed.Attempts[0].Destination
				if destination.Kind != "local" || destination.NodeID == nil || *destination.NodeID != "i7-synthetic-node" || destination.Engine == nil || *destination.Engine != "llamacpp" || !reflect.DeepEqual(completed.Served, &destination) {
					t.Fatalf("history lost its actual delivered destination: %+v", completed)
				}
				if route.model == "" {
					if destination.Model != nil || completed.Workload != nil {
						t.Fatalf("model-less delivery invented a model or workload: %+v", completed)
					}
				} else if destination.Model == nil || *destination.Model != route.model {
					t.Fatalf("concrete destination model=%v, want %q", destination.Model, route.model)
				}
				changed := push(t, c, key, completed.Seq)
				if changed.CreatedAt != completed.CreatedAt || changed.Seq < completed.Seq || changed.State != completed.State {
					t.Fatalf("initial/push differ across the real relay: initial=%+v changed=%+v", completed, changed)
				}
				if route.model == "" {
					// This is the same nullable local destination consumed by the
					// production desktop IPC and TUI history-factory regressions.
					data, err := json.Marshal(changed)
					if err != nil {
						t.Fatal(err)
					}
					t.Logf("production_model_less_record=%s", data)
				}
			}
		})
	}
	if t.Failed() {
		return
	}

	t.Run("registered errors and pre-registration refusal", func(t *testing.T) {
		before := history(t, a, nodepolicy.MethodRoutingGetInitial)
		response, err := f.client.Get(fmt.Sprintf("http://127.0.0.1:%d/not-an-inference-route", listener))
		if err != nil {
			t.Fatal(err)
		}
		_, readErr := io.Copy(io.Discard, response.Body)
		closeErr := response.Body.Close()
		if readErr != nil || closeErr != nil || response.StatusCode != http.StatusNotFound {
			t.Fatalf("pre-registration refusal: status=%d read=%v close=%v", response.StatusCode, readErr, closeErr)
		}
		if after := history(t, a, nodepolicy.MethodRoutingGetInitial); len(after.Requests) != len(before.Requests) {
			t.Fatalf("pre-registration refusal created history: before=%d after=%d", len(before.Requests), len(after.Requests))
		}
		for _, invalid := range []struct{ name, body, strict string }{
			{"invalid control model", `{"model":"bad\nmodel","stream":false}`, ""},
			{"invalid stream", `{"model":"g-t4-concrete","stream":"yes"}`, ""},
			{"strict grammar", `{"model":"g-t4-concrete","stream":false}`, "not-json"},
		} {
			request, err := http.NewRequest(http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", f.port), strings.NewReader(invalid.body))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Content-Type", "application/json")
			if invalid.strict != "" {
				request.Header.Set(nodepolicy.StrictAdmissionHeader, invalid.strict)
			}
			response, err := f.client.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			body, readErr := io.ReadAll(response.Body)
			closeErr := response.Body.Close()
			if readErr != nil || closeErr != nil || response.StatusCode != http.StatusBadRequest || bytes.Contains(body, []byte("bad\\nmodel")) {
				t.Fatalf("%s: status=%d read=%v close=%v", invalid.name, response.StatusCode, readErr, closeErr)
			}
		}
		neEventually(t, "registered early errors traverse production history", 30*time.Second, func() bool {
			initial := history(t, b, nodepolicy.MethodRoutingGetInitial)
			errors := 0
			for _, record := range initial.Requests {
				if record.Reason == "request-invalid" {
					if record.State != "failed" || record.EndedAt == nil || len(record.Attempts) != 0 || record.Served != nil || record.Workload != nil || strings.ContainsRune(record.Requested.Model, '\n') {
						t.Fatalf("invalid registered request retained effects: %+v", record)
					}
					errors++
				}
			}
			return errors == 3 && len(initial.Requests) == len(before.Requests)+3
		})
	})

	t.Run("newest window survives process churn", func(t *testing.T) {
		lastID := uint64(0)
		for _, record := range history(t, a, nodepolicy.MethodRoutingGetInitial).Requests {
			id, err := strconv.ParseUint(record.Request.RequestID, 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			lastID = max(lastID, id)
		}
		lastID += nodepolicy.RoutingHistoryRecords + 4
		for range nodepolicy.RoutingHistoryRecords + 4 {
			egressProcessHTTP(t, f.client, http.MethodPost, f.port, "/v1/chat/completions", []byte(`{"model":"bad\nmodel"}`), http.StatusBadRequest)
		}
		var newest nodepolicy.RoutingHistoryInitial
		neEventually(t, "more than 256 actual requests converge to the retained window", 30*time.Second, func() bool {
			newest = history(t, a, nodepolicy.MethodRoutingGetInitial)
			if len(newest.Requests) != nodepolicy.RoutingHistoryRecords {
				return false
			}
			seen := make(map[uint64]bool, nodepolicy.RoutingHistoryRecords)
			for i, record := range newest.Requests {
				id, err := strconv.ParseUint(record.Request.RequestID, 10, 64)
				if err != nil {
					t.Fatal(err)
				}
				if id <= lastID-nodepolicy.RoutingHistoryRecords || id > lastID || record.Request.OriginNodeID != run.OriginNodeID || record.Request.ProxyRunID != run.ProxyRunID || record.State != "failed" || record.Reason != "request-invalid" {
					return false
				}
				if seen[id] {
					t.Fatalf("retained window repeats request %d", id)
				}
				seen[id] = true
				if i > 0 {
					previous := newest.Requests[i-1]
					previousID, err := strconv.ParseUint(previous.Request.RequestID, 10, 64)
					if err != nil {
						t.Fatal(err)
					}
					if previous.CreatedAt < record.CreatedAt || previous.CreatedAt == record.CreatedAt && previousID <= id {
						t.Fatalf("retained ranks are not newest first: previous=%+v current=%+v", previous.Request, record.Request)
					}
				}
			}
			for id := lastID - nodepolicy.RoutingHistoryRecords + 1; id <= lastID; id++ {
				if !seen[id] {
					t.Fatalf("retained window omitted request %d", id)
				}
			}
			return true
		})
		other := history(t, b, nodepolicy.MethodRoutingGetInitial)
		if !reflect.DeepEqual(newest, other) {
			t.Fatal("the two local subscriptions disagree on the newest retained window")
		}
		// Both clients also retain ordinary RPC progress after the history churn.
		for _, c := range clients {
			serviceStatus(t, c)
		}
	})
	if t.Failed() {
		return
	}

	t.Run("subscription reconnect and departed process", func(t *testing.T) {
		// A replacement local client obtains the accepted bank, independently
		// of the old client's queued notifications and raw control IDs.
		before := history(t, a, nodepolicy.MethodRoutingGetInitial)
		b.mustClose(t)
		reconnected := service.dial()
		reconnected.notification(t, "app:ready")
		if got := history(t, reconnected, nodepolicy.MethodRoutingSubscribe); !reflect.DeepEqual(got, before) {
			t.Fatal("replacement local subscription lost the accepted window")
		}
		clients = []*serviceRPCClient{a, reconnected}
		oldProcess := i7HybridProxyIdentity(t, service)
		slot := upstream.slot(t, "departed-source")
		t.Cleanup(slot.release)
		upstream.mu.Lock()
		slot.preOutput = true
		upstream.mu.Unlock()
		request := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "departed-source", i7BuffersBody(128))
		i7HybridAccepted(t, slot, request)
		oldKey := i7HybridActiveCount(t, f, 1)[0]
		row(t, a, oldKey, func(record nodepolicy.RoutingHistoryRecord) bool {
			return record.EndedAt == nil && len(record.Attempts) == 1
		})
		if err := service.inventory.kill(oldProcess); err != nil {
			t.Fatal(err)
		}
		i7BuffersAwait(t, request)
		egressProcessBarrier(t, slot.cancelled, "departed source cancelled its owned upstream request")
		for _, c := range clients {
			departed := row(t, c, oldKey, terminal)
			if departed.State != "failed" || departed.Reason != "proxy-exited" || departed.Served != nil {
				t.Fatalf("departed accepted row was not retired: %+v", departed)
			}
		}
		neEventually(t, "the broker owns one replacement proxy", serviceCallTimeout, func() bool {
			owned, err := service.inventory.scan()
			if err != nil {
				t.Fatal(err)
			}
			count := 0
			for _, identity := range owned {
				if identity.Path == service.proxyPath && identity != oldProcess {
					count++
				}
			}
			return count == 1
		})
		i7HybridListenerReady(t, f.client, listener)
		egressProcessOK(t, f.call(t, "llamacpp-proxy:node/add-manual", map[string]any{
			"id": "i7-synthetic-node", "host": "127.0.0.1", "port": upstream.port,
			"addresses": []string{"127.0.0.1"}, "models": []string{egressProcessModel},
		}))
		nextSlot := upstream.slot(t, "replacement-source")
		t.Cleanup(nextSlot.release)
		nextRequest := i7BuffersStart(t, f.client, f.port, "/v1/chat/completions", "replacement-source", i7BuffersBody(128))
		i7HybridAccepted(t, nextSlot, nextRequest)
		key := i7HybridActiveCount(t, f, 1)[0]
		if key.OriginNodeID != oldKey.OriginNodeID || key.ProxyRunID == oldKey.ProxyRunID || key.RequestID != "1" {
			t.Fatalf("replacement process reused the wrong identity: old=%+v current=%+v", oldKey, key)
		}
		i7HybridCancel(t, f, oldKey, false, nodepolicy.RoutingCancelNotFound)
		egressProcessBarrier(t, nextRequest.firstByte, "stale cancellation left replacement output usable")
		nextSlot.release()
		i7BuffersCompletedStream(t, nextRequest)
		i7HybridActiveCount(t, f, 0)
		for _, c := range clients {
			completed := row(t, c, key, terminal)
			if completed.State != "completed" || completed.Served == nil {
				t.Fatalf("replacement inference did not traverse history: %+v", completed)
			}
			if old := row(t, c, oldKey, terminal); old.Reason != "proxy-exited" {
				t.Fatalf("replacement changed old-source history: %+v", old)
			}
		}
	})
	if t.Failed() {
		return
	}
	t.Run("broker restart clears the previous history generation", func(t *testing.T) {
		before := serviceStatus(t, a)
		if initial := history(t, a, nodepolicy.MethodRoutingGetInitial); len(initial.Requests) == 0 {
			t.Fatal("broker restart requires a nonempty accepted history bank")
		}
		owned, err := service.inventory.scan()
		if err != nil {
			t.Fatal(err)
		}
		killed := false
		for _, identity := range owned {
			if identity.Path == service.brokerPath && int(identity.PID) == before.BrokerPID {
				if err := service.inventory.kill(identity); err != nil {
					t.Fatal(err)
				}
				killed = true
			}
		}
		if !killed {
			t.Fatal("running broker has no matching owned process identity")
		}
		neEventually(t, "service owns a replacement broker", serviceCallTimeout, func() bool {
			m, err := a.call(`"restart-status"`, "service/status", nil, serviceCallTimeout)
			if err != nil || m.Error != nil {
				return false
			}
			var after serviceStatusSnapshot
			if err := json.Unmarshal(m.Result, &after); err != nil {
				t.Fatal(err)
			}
			return after.PID == before.PID && after.BrokerPID > 0 && after.BrokerPID != before.BrokerPID && after.BrokerRestarts > before.BrokerRestarts
		})
		for _, c := range clients {
			neEventually(t, "old broker history is absent from the replacement generation", serviceCallTimeout, func() bool {
				return len(history(t, c, nodepolicy.MethodRoutingGetInitial).Requests) == 0
			})
			serviceStatus(t, c)
		}
	})
}

// The routing listener is served by the proxy process, which a node with no
// engine facade configured runs only while the policy enables the listener.
func TestServiceRoutingListenerWithoutFacades(t *testing.T) {
	f := newProxyServiceFixture(t)
	f.facadePort = 0
	// The proxy is not expected to run until the listener is enabled, so it is
	// not part of what a ready fixture must show; it is inventoried separately.
	proxyOnly := &serviceProcessInventory{paths: []string{f.proxyPath}, known: make(map[int32]serviceProcessIdentity)}
	f.inventory.paths = slices.DeleteFunc(f.inventory.paths, func(p string) bool { return p == f.proxyPath })
	f.service = f.startChild(
		"--broker-path", f.brokerPath, "--endpoint", f.endpoint, "--log-dir", filepath.Join(f.dir, "logs"),
		"--", "--scanner-path", f.scannerPath, "--cluster-dir", f.clusterDir,
		"--proxy-path", f.proxyPath, "--proxy-engines=",
	)
	c := f.readyClient()
	proxies := func() int {
		t.Helper()
		owned, err := proxyOnly.scan()
		if err != nil {
			t.Fatalf("proxy inventory: %v", err)
		}
		return len(owned)
	}
	waitProxies := func(want int, why string) {
		t.Helper()
		deadline := time.Now().Add(serviceCallTimeout)
		for proxies() != want {
			if time.Now().After(deadline) {
				t.Fatalf("%d proxy processes, want %d: %s", proxies(), want, why)
			}
			time.Sleep(50 * time.Millisecond)
		}
	}
	type setResult struct {
		Policy         nodepolicy.Policy `json:"policy"`
		Activation     string            `json:"activation"`
		ActiveRevision string            `json:"activeRevision"`
	}
	setListener := func(id string, listener nodepolicy.Listener) setResult {
		t.Helper()
		want := servicePolicy(t, c)
		want.Tiers["strong"] = []nodepolicy.TierEntry{{Kind: nodepolicy.EntryLocal, Engine: "ollama", Model: "big:70b"}}
		want.Routing.Listener = listener
		if err := want.Validate(); err != nil {
			t.Fatalf("invalid listener fixture: %v", err)
		}
		m := c.mustPolicyCall(t, id, want)
		if m.Error != nil {
			t.Fatalf("policy:set RPC error: %+v", m.Error)
		}
		var got setResult
		if err := json.Unmarshal(m.Result, &got); err != nil {
			t.Fatalf("decode policy:set: %v", err)
		}
		return got
	}
	listed := func(port int) ([]string, error) {
		client := http.Client{Timeout: time.Second}
		resp, err := client.Get("http://127.0.0.1:" + strconv.Itoa(port) + "/v1/models")
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}
		var list struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if err := json.Unmarshal(body, &list); err != nil {
			t.Fatalf("decode model list %q: %v", body, err)
		}
		ids := []string{}
		for _, d := range list.Data {
			ids = append(ids, d.ID)
		}
		return ids, nil
	}
	waitListed := func(port int) {
		t.Helper()
		want := []string{"strong", "strong-api", "strong-local"}
		deadline := time.Now().Add(serviceCallTimeout)
		for {
			ids, err := listed(port)
			if err == nil && slices.Equal(ids, want) {
				return
			}
			if time.Now().After(deadline) {
				t.Fatalf("listener on %d lists %v (err %v), want %v", port, ids, err, want)
			}
			time.Sleep(50 * time.Millisecond)
		}
	}

	if got := proxies(); got != 0 {
		t.Fatalf("%d proxy processes with no facade and the listener disabled, want none", got)
	}

	port := freePort(t)
	t.Logf("fixture_routing_listener port=%d", port)
	enabled := setListener("301", nodepolicy.Listener{Enabled: true, Port: port})
	// The proxy starts in the background, so the revision is pending until the
	// new process has acknowledged the policy it is replayed.
	waitActive := func(revision string) {
		t.Helper()
		deadline := time.Now().Add(serviceCallTimeout)
		for {
			got := fetchServicePolicy(t, c)
			if got.Policy.Revision == revision && got.Activation == "active" && got.ActiveRevision == revision {
				return
			}
			if time.Now().After(deadline) {
				t.Fatalf("revision %q never activated: at %q activation=%q activeRevision=%q", revision, got.Policy.Revision, got.Activation, got.ActiveRevision)
			}
			time.Sleep(50 * time.Millisecond)
		}
	}
	waitProxies(1, "the policy enabled the listener")
	waitActive(enabled.Policy.Revision)
	waitListed(port)

	disabled := setListener("302", nodepolicy.Listener{Enabled: false, Port: port})
	if disabled.Policy.Routing.Listener.Enabled {
		t.Fatal("the listener is still enabled in the saved policy")
	}
	waitProxies(0, "the listener was disabled and no facade is configured")
	if ids, err := listed(port); err == nil {
		t.Fatalf("the disabled listener still answers on %d with %v", port, ids)
	}

	again := freePort(t)
	t.Logf("fixture_routing_listener reenabled_port=%d", again)
	setListener("303", nodepolicy.Listener{Enabled: true, Port: again})
	waitProxies(1, "the listener was enabled again")
	waitListed(again)

	// Fixture cleanup now also requires that stopping the service ended the proxy.
	f.inventory.paths = append(f.inventory.paths, f.proxyPath)
}
