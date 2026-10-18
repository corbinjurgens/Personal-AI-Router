// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

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
