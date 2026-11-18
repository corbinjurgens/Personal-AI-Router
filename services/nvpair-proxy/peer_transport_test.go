// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/tls"
	"errors"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
)

type proxyTestChanges struct {
	mu sync.Mutex
	ch chan struct{}
}

func (s *proxyTestChanges) Changed() <-chan struct{} { s.mu.Lock(); defer s.mu.Unlock(); return s.ch }
func (s *proxyTestChanges) notify() {
	s.mu.Lock()
	defer s.mu.Unlock()
	close(s.ch)
	s.ch = make(chan struct{})
}

func TestProxyDirectRouteRetriedAfterChange(t *testing.T) {
	f := newProxyReverseFixture(t, "llamacpp", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) }))
	// Exercise the proxy's production adapter constructor and dialers. Its
	// temporary private IPC path has no listener, so reverse setup fails.
	f.p.peerServices = nil
	changes := &proxyTestChanges{ch: make(chan struct{})}
	f.p.mon = changes
	var healthy atomic.Bool
	var handshakes atomic.Int32
	getConfig := f.server.TLS.GetConfigForClient
	f.server.TLS.GetConfigForClient = func(hello *tls.ClientHelloInfo) (*tls.Config, error) {
		handshakes.Add(1)
		if !healthy.Load() {
			return nil, errors.New("direct path down")
		}
		return getConfig(hello)
	}
	c := f.candidate(t)
	tr := f.p.candidateTransport(c)
	for range 2 {
		req := proxyReverseRequest(t, http.MethodPost, c.url.String(), nil)
		response, err := tr.RoundTrip(req)
		if response != nil {
			response.Body.Close()
		}
		if err == nil {
			t.Fatal("unreachable direct route unexpectedly succeeded")
		}
	}
	if got := handshakes.Load(); got != 1 {
		t.Fatalf("handshakes before wake = %d, want one then backoff", got)
	}
	healthy.Store(true)
	changes.notify()
	proxyReverseRoundTrip(t, tr, c.url.String())
	if got := handshakes.Load(); got != 2 {
		t.Fatalf("handshakes after wake = %d, want direct retry", got)
	}
}
