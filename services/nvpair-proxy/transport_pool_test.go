// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"testing"

	"nvpair-shared/clustertrusttest"
)

func TestCandidateTransportReusesPlainTransport(t *testing.T) {
	p := testProxy(anyProfile(t), NewDiscovery(), 11435)
	a := p.candidateTransport(candidate{})
	b := p.candidateTransport(candidate{id: "manual"})
	if a != b {
		t.Fatalf("plain candidates returned distinct Transports")
	}
	if a == nil {
		t.Fatal("plain Transport is nil")
	}
}

func TestCandidateTransportPeerUsesFreshPinnedConnections(t *testing.T) {
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	f.directHealthy.Store(true)
	c := f.candidate(t)
	tr := f.p.candidateTransport(c)
	for range 3 {
		proxyReverseRoundTrip(t, tr, c.url.String())
	}
	if got := f.directCalls.Load(); got != 3 {
		t.Fatalf("direct connections = %d, want 3 fresh connections", got)
	}
	if got := f.reverseCalls.Load(); got != 0 {
		t.Fatalf("reverse connections = %d, want 0 with healthy direct TLS", got)
	}
	if got := f.closedRequests.Load(); got != 3 {
		t.Fatalf("pinned ingress Connection: close requests = %d, want 3", got)
	}
}

func TestDropUnpinnedPeerTransportsRejectsExistingCandidate(t *testing.T) {
	f := newProxyReverseFixture(t, "ollama", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	c := f.candidate(t)
	tr := f.p.candidateTransport(c)
	proxyReverseRoundTrip(t, tr, c.url.String())
	before := f.reverseCalls.Load()
	clustertrusttest.RemovePeerPin(t, f.selfDir, "uuid-peer")
	f.p.mesh.Refresh()
	f.p.dropUnpinnedPeerTransports()
	request := proxyReverseRequest(t, http.MethodPost, c.url.String(), nil)
	if response, err := tr.RoundTrip(request); err == nil {
		response.Body.Close()
		t.Fatal("transport retained authority after peer pin removal")
	}
	if got := f.reverseCalls.Load(); got != before {
		t.Fatalf("removed peer triggered %d additional reverse dials", got-before)
	}
}
