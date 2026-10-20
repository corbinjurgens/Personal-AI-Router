// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func TestRequestRegistrySaturationRefusesBeforeEffects(t *testing.T) {
	w := &recordingWriter{}
	p := newBoundProxy(NewCodec(w))
	requests := make([]*routingRequest, 0, 1024)
	cancelled := 0
	for range 1024 {
		request, err := p.registerRoutingRequest(newWorkloadCtl(func(string) { cancelled++ }))
		if err != nil {
			t.Fatalf("registry filled before 1024 entries: %v", err)
		}
		requests = append(requests, request)
		defer p.unregisterRoutingRequest(request)
	}
	if err := p.registerWorkload(requests[0], workloadKey{engine: "ollama", id: "7"}); err != nil {
		t.Fatal(err)
	}
	f := newFacade(p, anyProfile(t), NewDiscovery(), 0)
	body := &identityBody{}
	rec := httptest.NewRecorder()
	f.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", body))
	if rec.Code != http.StatusServiceUnavailable || body.reads != 0 || len(w.lines()) != 0 || len(p.activeRoutingRequests()) != 1024 || p.requestSequence != 1024 {
		t.Fatalf("saturated request status=%d reads=%d notifications=%d entries=%d sequence=%d", rec.Code, body.reads, len(w.lines()), len(p.activeRoutingRequests()), p.requestSequence)
	}
	control := p.routingRequestControl(requests[0].key)
	if control == nil || !control.cancel(false) || cancelled != 1 {
		t.Fatal("registry pressure hid a live cancellation target")
	}
	p.unregisterRoutingRequest(requests[0])
	if len(p.workloadControls("ollama", "7")) != 0 {
		t.Fatal("terminal request retained a workload index")
	}
	rec = httptest.NewRecorder()
	f.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"a","model":"b"}`)))
	if rec.Code != http.StatusBadRequest || len(p.activeRoutingRequests()) != 1023 || p.requestSequence != 1025 {
		t.Fatalf("released slot was not reused and cleaned: status=%d entries=%d sequence=%d", rec.Code, len(p.activeRoutingRequests()), p.requestSequence)
	}
}

func TestRequestRegistryTracksActualRequestAndFinalCleanup(t *testing.T) {
	tc := anyCase(t)
	engine := newStallingEngine(t)
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "a", engine.URL, tc.advertisedModel))
	w := &recordingWriter{}
	p := newTestProxy(tc.profile, NewCodec(w), disc, tc.profile.FacadePort)
	p.soleFacade().nextRequestID.Store(70)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	done := make(chan struct{})
	go func() {
		defer close(done)
		p.soleFacade().handleHTTP(httptest.NewRecorder(), tc.inferenceRequest().WithContext(ctx))
	}()
	waitChan(t, engine.received, "registered request at the engine")
	keys := p.activeRoutingRequests()
	if len(keys) != 1 || keys[0].OriginNodeID != "test-origin" || keys[0].ProxyRunID != p.runID || keys[0].RequestID != "1" {
		t.Fatalf("live keys=%+v", keys)
	}
	controls := p.workloadControls(tc.profile.Name, "71")
	if len(controls) != 1 || p.routingRequestControl(keys[0]) != controls[0] {
		t.Fatal("workload index did not resolve the canonical request")
	}
	stale := keys[0]
	stale.ProxyRunID = "old-run"
	if p.routingRequestControl(stale) != nil {
		t.Fatal("old process key resolved a request")
	}
	stale = keys[0]
	stale.OriginNodeID = "other-origin"
	if p.routingRequestControl(stale) != nil {
		t.Fatal("other origin key resolved a request")
	}
	if !p.routingRequestControl(keys[0]).cancel(false) {
		t.Fatal("canonical key could not cancel live request")
	}
	waitChan(t, done, "cancelled request cleanup")
	if len(p.activeRoutingRequests()) != 0 || len(p.workloadControls(tc.profile.Name, "71")) != 0 || p.routingRequestControl(keys[0]) != nil {
		t.Fatal("completed request remained discoverable or cancellable")
	}
}

func TestRequestRegistryWorkloadDetachesWithoutEndingRequest(t *testing.T) {
	p, w := policyProxy(t)
	first, err := p.registerRoutingRequest(newWorkloadCtl(func(string) {}))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(first)
	key := workloadKey{engine: "ollama", id: "7"}
	if err := p.registerWorkload(first, key); err != nil {
		t.Fatal(err)
	}
	p.unregisterWorkload(first)
	response := rpc(t, p, w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{OriginatedFrom: "test-origin", WorkloadID: "7", Engine: "ollama", RunID: p.runID})
	var result nodepolicy.WorkloadCancelResult
	if response.Error != nil || json.Unmarshal(response.Result, &result) != nil || result.Found {
		t.Fatal("detached workload still cancelled its request")
	}
	if p.routingRequestControl(first.key) != first.control || len(p.activeRoutingRequests()) != 1 {
		t.Fatal("workload detachment ended the canonical request")
	}
	second, err := p.registerRoutingRequest(newWorkloadCtl(func(string) {}))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(second)
	if err := p.registerWorkload(second, key); err != nil {
		t.Fatal(err)
	}
	p.unregisterRoutingRequest(first)
	controls := p.workloadControls("ollama", "7")
	if len(controls) != 1 || controls[0] != second.control {
		t.Fatal("old owner cleanup removed a reused workload association")
	}
}

func TestRequestRegistryTerminalWorkloadDetachesBeforeRequestCleanup(t *testing.T) {
	tc := anyCase(t)
	engine := newRecordingEngine(t, http.StatusOK)
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "a", engine.URL, tc.advertisedModel))
	w := &recordingWriter{}
	p := newTestProxy(tc.profile, NewCodec(w), disc, tc.profile.FacadePort)
	writer := &registryResponseWriter{header: make(http.Header), entered: make(chan struct{}), release: make(chan struct{})}
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(writer.release) }) }
	t.Cleanup(release)
	done := make(chan struct{})
	go func() { defer close(done); p.soleFacade().handleHTTP(writer, tc.inferenceRequest()) }()
	waitChan(t, writer.entered, "blocked downstream write")
	keys := p.activeRoutingRequests()
	if len(keys) != 1 {
		t.Fatalf("active keys=%+v", keys)
	}
	control := p.routingRequestControl(keys[0])
	if control == nil || !control.cancel(false) {
		t.Fatal("live request cancellation failed")
	}
	waitForCond(t, time.Second, "terminal workload event", func() bool { return len(notificationsOf(t, w, workloadErroredMethod)) == 1 })
	if len(p.workloadControls(tc.profile.Name, "1")) != 0 {
		t.Fatal("terminal workload retained cancellation association")
	}
	if len(p.activeRoutingRequests()) != 1 {
		t.Fatal("request disappeared while downstream cleanup was still blocked")
	}
	release()
	waitChan(t, done, "downstream release and final cleanup")
	if len(p.activeRoutingRequests()) != 0 {
		t.Fatal("final cleanup retained the request")
	}
}

func TestRequestRegistryRegistersBeforeReadingBody(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	f := newFacade(p, anyProfile(t), NewDiscovery(), 0)
	entered, release := make(chan struct{}), make(chan struct{})
	body := &registryBlockingBody{entered: entered, release: release}
	done := make(chan struct{})
	go func() {
		defer close(done)
		f.handleHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/chat", body))
	}()
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	waitChan(t, entered, "body read")
	keys := p.activeRoutingRequests()
	if len(keys) != 1 || p.routingRequestControl(keys[0]) == nil {
		t.Fatal("body read began before the request was registered")
	}
	close(release)
	waitChan(t, done, "invalid-body cleanup")
	if len(p.activeRoutingRequests()) != 0 {
		t.Fatal("early rejection leaked registry metadata")
	}
}

type registryBlockingBody struct {
	entered, release chan struct{}
	read             bool
}

func (b *registryBlockingBody) Read(out []byte) (int, error) {
	if b.read {
		return 0, io.EOF
	}
	b.read = true
	close(b.entered)
	select {
	case <-b.release:
	case <-time.After(2 * time.Second):
		return 0, io.ErrUnexpectedEOF
	}
	return copy(out, []byte(`{"model":"a","model":"b"}`)), nil
}
func (*registryBlockingBody) Close() error { return nil }

type registryResponseWriter struct {
	header           http.Header
	entered, release chan struct{}
	once             sync.Once
}

func (w *registryResponseWriter) Header() http.Header { return w.header }
func (*registryResponseWriter) WriteHeader(int)       {}
func (w *registryResponseWriter) Write(b []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	<-w.release
	return len(b), nil
}
func (*registryResponseWriter) Flush() {}
