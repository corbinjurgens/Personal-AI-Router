// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"nvpair-shared/nodepolicy"
)

// Test-only registry observation; production snapshots must obtain a root
// lease through snapshotRoutingRequests before copying or encoding any keys.
func (p *Proxy) activeRoutingRequests() []nodepolicy.RoutingRequestKey {
	p.identityMu.Lock()
	keys := make([]nodepolicy.RoutingRequestKey, 0, len(p.requests))
	for _, request := range p.requests {
		keys = append(keys, request.key)
	}
	p.identityMu.Unlock()
	sortRoutingRequests(keys)
	return keys
}

func registryTestBytes(p *Proxy) int64 {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	var total int64
	for _, request := range p.requests {
		key := request.key
		total += 512 + int64(len(key.OriginNodeID)+len(key.ProxyRunID)+len(key.RequestID))
	}
	for _, slots := range []int{p.requestStorage.slots, p.workloadStorage.slots} {
		if slots != 0 {
			total += 256 + 256*int64(slots)
		}
	}
	return total
}

// Preserve a model-bearing HTTP fixture's original inference-byte headroom
// while funding its canonical row and initially absent map backing owners.
func registryNextTestBytes(p *Proxy) int64 {
	p.identityMu.Lock()
	defer p.identityMu.Unlock()
	n := int64(512 + len(p.originNodeID) + len(p.runID) + len(strconv.FormatInt(p.requestSequence+1, 10)))
	if p.requestStorage.lease == nil {
		n += 256 + 8*256
	}
	if p.workloadStorage.lease == nil {
		n += 256 + 8*256
	}
	return n
}

func TestRequestRegistryZeroFacadeMetadataLifetime(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
	request, err := p.registerRoutingRequest(newRoutingRequest(nil))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(request)
	want := int64(512 + len("test-origin") + len(p.runID) + len("1") + 256 + 8*256)
	if len(p.enabledFacades()) != 0 {
		t.Fatal("metadata fixture enabled a facade")
	}
	bufferAssertTotals(t, p.buffers, want, 2, 0)
	if request.owner.cancelRouting(false) != "cancelled" {
		t.Fatal("cancel failed")
	}
	bufferAssertTotals(t, p.buffers, want, 2, 0)
	request.owner.finish()
	bufferAssertTotals(t, p.buffers, want, 2, 0)
	p.unregisterRoutingRequest(request)
	p.unregisterRoutingRequest(request)
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRequestRegistryRetainedBackingPartialDrainAndReuse(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	var rows []*routingRequest
	for range 40 {
		row, err := p.registerRoutingRequest(newRoutingRequest(nil))
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, row)
		defer p.unregisterRoutingRequest(row)
	}
	backing, slots, inserts := p.requestStorage.lease, p.requestStorage.slots, p.requestStorage.inserts
	if backing == nil || slots < len(rows) {
		t.Fatal("registry backing was not funded")
	}
	for _, row := range rows[1:] {
		p.unregisterRoutingRequest(row)
	}
	if len(p.requests) != 1 || p.requestStorage.lease != backing || p.requestStorage.slots != slots || p.requestStorage.inserts != inserts {
		t.Fatal("partial drain rebuilt backing or retained terminal rows")
	}
	want := int64(512+len(rows[0].key.OriginNodeID)+len(rows[0].key.ProxyRunID)+len(rows[0].key.RequestID)) + 256 + 256*int64(slots)
	bufferAssertTotals(t, p.buffers, want, 2, 0)
	// Leave room for exactly one new row, not another backing allocation.
	rowBytes := int64(512 + len(p.originNodeID) + len(p.runID) + len("41"))
	hold := bufferMustAcquire(t, p.buffers, p.buffers.capacity-want-rowBytes)
	defer hold.close()
	reused, err := p.registerRoutingRequest(newRoutingRequest(nil))
	if err != nil {
		t.Fatalf("funded backing was not reusable: %v", err)
	}
	defer p.unregisterRoutingRequest(reused)
	if p.requestStorage.lease != backing || p.requestStorage.slots != slots {
		t.Fatal("row reuse replaced still-funded backing")
	}
	bufferAssertTotals(t, p.buffers, p.buffers.capacity, 4, 0)
	p.unregisterRoutingRequest(reused)
	hold.close()
	// Churn must eventually compact on insertion with overlapping funding,
	// while never accumulating one retained lease per departed row.
	for range 256 {
		row, err := p.registerRoutingRequest(newRoutingRequest(nil))
		if err != nil {
			t.Fatal(err)
		}
		p.unregisterRoutingRequest(row)
		if len(p.requests) != 1 || p.requestStorage.slots > slots {
			t.Fatal("churn retained rows or grew backing without a live need")
		}
		bufferAssertTotals(t, p.buffers, registryTestBytes(p), 2, 0)
	}
	if p.requestStorage.lease == backing {
		t.Fatal("insertion history did not retire churned backing")
	}
	p.buffers.mu.Lock()
	retired := backing.closed
	p.buffers.mu.Unlock()
	if !retired {
		t.Fatal("replaced backing retained its old root lease")
	}
	p.unregisterRoutingRequest(rows[0])
	if p.requests != nil || p.requestStorage.lease != nil || p.requestStorage.slots != 0 {
		t.Fatal("empty registry retained backing ownership")
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRequestRegistryRebuildRequiresOverlapGrant(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	var rows []*routingRequest
	for range 8 {
		row, err := p.registerRoutingRequest(newRoutingRequest(nil))
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, row)
		defer p.unregisterRoutingRequest(row)
	}
	for _, row := range rows[1:] {
		p.unregisterRoutingRequest(row)
	}
	backing := p.requestStorage.lease
	metadata := registryTestBytes(p)
	rowBytes := int64(512 + len(p.originNodeID) + len(p.runID) + len("9"))
	hold := bufferMustAcquire(t, p.buffers, p.buffers.capacity-metadata-rowBytes)
	defer hold.close()
	owner := newRoutingRequest(nil)
	if row, err := p.registerRoutingRequest(owner); err == nil || row != nil {
		t.Fatal("rebuild released old backing before obtaining overlap capacity")
	}
	if p.requestSequence != 8 || len(p.requests) != 1 || p.requestStorage.lease != backing || owner.key.RequestID != "" {
		t.Fatal("failed rebuild changed registry or identity")
	}
	bufferAssertTotals(t, p.buffers, p.buffers.capacity-rowBytes, 3, 0)
	if rows[0].owner.cancelRouting(false) != "cancelled" {
		t.Fatal("failed rebuild hid cancellation")
	}
	hold.close()
	row, err := p.registerRoutingRequest(owner)
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(row)
	if row.key.RequestID != "9" || p.requestStorage.lease == backing {
		t.Fatal("released capacity did not admit a charged rebuild")
	}
	bufferAssertTotals(t, p.buffers, registryTestBytes(p), 3, 0)
}

func TestRequestRegistryWorkloadBackingPartialDrainAndEmptyRelease(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	var rows []*routingRequest
	for i := range 40 {
		row, err := p.registerRoutingRequest(newRoutingRequest(nil))
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, row)
		defer p.unregisterRoutingRequest(row)
		if err := p.registerWorkload(row, workloadKey{engine: "ollama", id: strconv.Itoa(i + 1)}); err != nil {
			t.Fatal(err)
		}
	}
	backing, slots, inserts := p.workloadStorage.lease, p.workloadStorage.slots, p.workloadStorage.inserts
	if backing == nil {
		t.Fatal("workload-index backing was not funded")
	}
	for _, row := range rows[1:] {
		p.unregisterWorkload(row)
	}
	if len(p.workloadRequests) != 1 || p.workloadStorage.lease != backing || p.workloadStorage.slots != slots || p.workloadStorage.inserts != inserts {
		t.Fatal("workload partial drain discarded funding or retained terminal associations")
	}
	metadata := registryTestBytes(p)
	bufferAssertTotals(t, p.buffers, metadata, 42, 0)
	hold := bufferMustAcquire(t, p.buffers, p.buffers.capacity-metadata)
	defer hold.close()
	if err := p.registerWorkload(rows[1], workloadKey{engine: "ollama", id: "reused"}); err != nil {
		t.Fatalf("funded index backing could not be reused at full root: %v", err)
	}
	if p.workloadStorage.lease != backing {
		t.Fatal("index reuse allocated replacement backing")
	}
	p.unregisterWorkload(rows[1])
	bufferAssertTotals(t, p.buffers, p.buffers.capacity, 43, 0)
	hold.close()
	for i := range 256 {
		if err := p.registerWorkload(rows[1], workloadKey{engine: "ollama", id: strconv.Itoa(i + 100)}); err != nil {
			t.Fatal(err)
		}
		p.unregisterWorkload(rows[1])
		if len(p.workloadRequests) != 1 || p.workloadStorage.slots > slots {
			t.Fatal("workload churn retained associations or unbounded backing")
		}
		bufferAssertTotals(t, p.buffers, registryTestBytes(p), 42, 0)
	}
	indexBytes := int64(256 + 256*p.workloadStorage.slots)
	emptyHold := bufferMustAcquire(t, p.buffers, p.buffers.capacity-registryTestBytes(p))
	defer emptyHold.close()
	p.unregisterWorkload(rows[0])
	if p.workloadRequests != nil || p.workloadStorage.lease != nil || p.workloadStorage.slots != 0 {
		t.Fatal("empty workload index retained its backing")
	}
	bufferAssertTotals(t, p.buffers, p.buffers.capacity-indexBytes, 42, 0)
	emptyHold.close()
	bufferAssertTotals(t, p.buffers, registryTestBytes(p), 41, 0)
	// Canonical requests remain live independently of the now-empty index.
	if len(p.requests) != len(rows) || p.routingRequestControl(rows[0].key) != rows[0].owner {
		t.Fatal("index cleanup retired canonical owners")
	}
	for _, row := range rows {
		p.unregisterRoutingRequest(row)
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRequestRegistryBackingAdmissionRollback(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	rowBytes := int64(512 + len(p.originNodeID) + len(p.runID) + 1)
	if err := p.buffers.reconfigure(rowBytes, 0); err != nil {
		t.Fatal(err)
	}
	owner := newRoutingRequest(nil)
	if row, err := p.registerRoutingRequest(owner); err == nil || row != nil {
		t.Fatal("unfunded backing admitted a registry entry")
	}
	if p.requests != nil || p.requestStorage.lease != nil || p.requestSequence != 0 || owner.key.RequestID != "" {
		t.Fatal("failed backing grant published partial state")
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRequestRegistryWorkloadBackingAdmissionRollback(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	row, err := p.registerRoutingRequest(newRoutingRequest(nil))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(row)
	hold := bufferMustAcquire(t, p.buffers, p.buffers.capacity-registryTestBytes(p))
	defer hold.close()
	key := workloadKey{engine: "ollama", id: "7"}
	if err := p.registerWorkload(row, key); err == nil {
		t.Fatal("unfunded workload-index backing was allocated")
	}
	if p.workloadRequests != nil || p.workloadStorage.lease != nil || row.owner.workload != nil {
		t.Fatal("index grant failure published a workload association")
	}
	bufferAssertTotals(t, p.buffers, p.buffers.capacity, 3, 0)
	if row.owner.cancelRouting(false) != "cancelled" {
		t.Fatal("index refusal hid cancellation")
	}
	hold.close()
	p.unregisterRoutingRequest(row)
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRequestRegistrySnapshotOwnsKeysAfterRemoval(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	request, err := p.registerRoutingRequest(newRoutingRequest(nil))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(request)
	metadata := registryTestBytes(p)
	id := json.RawMessage(`1`)
	keys, lease, err := p.snapshotRoutingRequests(&id)
	if err != nil {
		t.Fatal(err)
	}
	defer lease.close()
	if len(keys) != 1 || keys[0] != request.key {
		t.Fatalf("snapshot=%+v", keys)
	}
	// The chosen ownership model is an independently charged clone, not a
	// shallow string whose registry lease could end during response encoding.
	if unsafe.StringData(keys[0].OriginNodeID) == unsafe.StringData(request.key.OriginNodeID) ||
		unsafe.StringData(keys[0].ProxyRunID) == unsafe.StringData(request.key.ProxyRunID) ||
		unsafe.StringData(keys[0].RequestID) == unsafe.StringData(request.key.RequestID) {
		t.Fatal("snapshot key still borrows registry string backing")
	}
	p.buffers.mu.Lock()
	snapshotBytes := p.buffers.used - metadata
	p.buffers.mu.Unlock()
	if snapshotBytes < int64(128+len(keys[0].OriginNodeID)+len(keys[0].ProxyRunID)+len(keys[0].RequestID)) {
		t.Fatal("snapshot copy storage was not charged")
	}
	p.unregisterRoutingRequest(request)
	bufferAssertTotals(t, p.buffers, snapshotBytes, 1, 0)
	if keys[0] != request.owner.key {
		t.Fatal("entry removal changed snapshot identity")
	}
	lease.close()
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRequestRegistryRootFundedEntriesAndTransportLimit(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	p.originNodeID = strings.Repeat("o", 128)
	p.runID = strings.Repeat("r", 128)
	for range 4096 {
		request, err := p.registerRoutingRequest(newRoutingRequest(nil))
		if err != nil {
			t.Fatalf("funded registry was independently capped: %v", err)
		}
		defer p.unregisterRoutingRequest(request)
	}
	bufferAssertTotals(t, p.buffers, registryTestBytes(p), 4097, 0)
	id := json.RawMessage(`1`)
	keys, lease, err := p.snapshotRoutingRequests(&id)
	if err == nil || keys != nil || lease != nil {
		t.Fatal("oversized IPC reply admitted a snapshot copy")
	}
	bufferAssertTotals(t, p.buffers, registryTestBytes(p), 4097, 0)
	for _, request := range p.requests {
		p.unregisterRoutingRequest(request)
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRequestRegistrySnapshotCallerIDCapacity(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	request, err := p.registerRoutingRequest(newRoutingRequest(nil))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(request)
	metadata := registryTestBytes(p)
	// This input fits the inbound codec, but HTML escaping alone makes its
	// outgoing envelope exceed the same codec's maximum frame size.
	id := json.RawMessage(`"` + strings.Repeat("<", (1<<20)/6) + `"`)
	keys, lease, err := p.snapshotRoutingRequests(&id)
	if err == nil || keys != nil || lease != nil {
		t.Fatal("oversized caller ID admitted a snapshot copy")
	}
	bufferAssertTotals(t, p.buffers, metadata, 2, 0)
}

func TestRequestRegistrySaturationRefusesBeforeEffects(t *testing.T) {
	w := &recordingWriter{}
	p := newBoundProxy(NewCodec(w))
	cancelled := 0
	request, err := p.registerRoutingRequest(newRoutingRequest(func(string) { cancelled++ }))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(request)
	if err := p.registerWorkload(request, workloadKey{engine: "ollama", id: "7"}); err != nil {
		t.Fatal(err)
	}
	metadata := registryTestBytes(p)
	bufferAssertTotals(t, p.buffers, metadata, 3, 0)
	lease := bufferMustAcquire(t, p.buffers, p.buffers.capacity-metadata)
	defer lease.close()
	f := newFacade(p, anyProfile(t), NewDiscovery(), 0)
	body := &identityBody{}
	rec := httptest.NewRecorder()
	f.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", body))
	if rec.Code != http.StatusServiceUnavailable || body.reads != 0 || len(w.lines()) != 0 || len(p.activeRoutingRequests()) != 1 || p.requestSequence != 1 {
		t.Fatalf("saturated request status=%d reads=%d notifications=%d entries=%d sequence=%d", rec.Code, body.reads, len(w.lines()), len(p.activeRoutingRequests()), p.requestSequence)
	}
	control := p.routingRequestControl(request.key)
	if control == nil || !control.cancel(false) || cancelled != 1 {
		t.Fatal("registry pressure hid a live cancellation target")
	}
	bufferAssertTotals(t, p.buffers, p.buffers.capacity, 4, 0)
	p.unregisterRoutingRequest(request)
	if len(p.workloadControls("ollama", "7")) != 0 {
		t.Fatal("terminal request retained a workload index")
	}
	// Free byte capacity after cleanup; metadata refusal must not consume an
	// identity, so the next admitted request remains counter 2.
	lease.close()
	rec = httptest.NewRecorder()
	f.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"a","model":"b"}`)))
	if rec.Code != http.StatusBadRequest || len(p.activeRoutingRequests()) != 0 || p.requestSequence != 2 {
		t.Fatalf("released slot was not reused and cleaned: status=%d entries=%d sequence=%d", rec.Code, len(p.activeRoutingRequests()), p.requestSequence)
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
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
	first, err := p.registerRoutingRequest(newRoutingRequest(func(string) {}))
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
	if p.routingRequestControl(first.key) != first.owner || len(p.activeRoutingRequests()) != 1 {
		t.Fatal("workload detachment ended the canonical request")
	}
	second, err := p.registerRoutingRequest(newRoutingRequest(func(string) {}))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(second)
	if err := p.registerWorkload(second, key); err != nil {
		t.Fatal(err)
	}
	p.unregisterRoutingRequest(first)
	controls := p.workloadControls("ollama", "7")
	if len(controls) != 1 || controls[0] != second.owner {
		t.Fatal("old owner cleanup removed a reused workload association")
	}
}

func TestRequestRegistryPublicReplacementDetachesConcreteWorkload(t *testing.T) {
	p, writer := policyProxy(t)
	owner := newRoutingRequest(nil)
	request, err := p.registerRoutingRequest(owner)
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(request)
	key := workloadKey{engine: "ollama", id: "7"}
	if err := p.registerWorkload(request, key); err != nil {
		t.Fatal(err)
	}
	target := nodepolicy.WorkloadCancelParams{OriginatedFrom: "test-origin", RunID: p.runID, Engine: key.engine, WorkloadID: key.id}
	resolvedBeforeDetach := p.workloadCancelControl(target)
	local, err := owner.newDispatchAttempt(context.Background(), func() {}, "local")
	if err != nil {
		t.Fatal(err)
	}
	local.finished(publicAttemptTerminal{})
	public, err := owner.newAttempt()
	if err != nil {
		t.Fatal(err)
	}
	publicCancelled := false
	if err := public.bindCancel(func() { publicCancelled = true }); err != nil {
		t.Fatal(err)
	}
	if resolvedBeforeDetach != owner || resolvedBeforeDetach.cancelQualified(false, &key) || publicCancelled || owner.requestCancelled() {
		t.Fatal("previously resolved concrete workload cancelled its public replacement")
	}
	response := rpc(t, p, writer, nodepolicy.MethodWorkloadCancel, target)
	var result nodepolicy.WorkloadCancelResult
	if response.Error != nil {
		t.Fatalf("stale workload cancel: %+v", response.Error)
	}
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatal(err)
	}
	if result.Found || len(p.workloadControls(key.engine, key.id)) != 0 || p.routingRequestControl(request.key) != owner || owner.active != public {
		t.Fatal("public replacement lost the canonical request or retained the old index")
	}
	if !p.routingRequestControl(request.key).cancel(false) || !publicCancelled || !owner.requestCancelled() {
		t.Fatal("routing key failed to cancel active public replacement")
	}
	public.finished(publicAttemptTerminal{})
	p.unregisterRoutingRequest(request)
	if len(p.activeRoutingRequests()) != 0 || len(p.workloadControls(key.engine, key.id)) != 0 {
		t.Fatal("replacement cleanup retained registry/index entries")
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

type registryPanicWriter struct{ *httptest.ResponseRecorder }

func (*registryPanicWriter) Write([]byte) (int, error) { panic("registry cleanup fixture") }

func TestRequestRegistryRetainedThroughFinalBorrower(t *testing.T) {
	for _, scenario := range []string{"success", "upstream body error", "cancellation", "panic"} {
		t.Run(scenario, func(t *testing.T) {
			closing, release := make(chan struct{}), make(chan struct{})
			var releaseOnce sync.Once
			unblock := func() { releaseOnce.Do(func() { close(release) }) }
			defer unblock()
			stopEngine := make(chan struct{})
			engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if _, err := io.WriteString(w, continuitySSEEvent); err != nil {
					return
				}
				w.(http.Flusher).Flush()
				switch scenario {
				case "upstream body error":
					panic(http.ErrAbortHandler)
				case "cancellation":
					select {
					case <-r.Context().Done():
					case <-stopEngine:
					}
				default:
					_, _ = io.WriteString(w, "data: [DONE]\n\n")
				}
			}))
			defer engine.Close()
			defer close(stopEngine)
			defer unblock()
			discovery := NewDiscovery()
			discovery.AddManual(nodeForModel(t, "a", engine.URL, "m"))
			p := newTestProxy(ollamaOnlyProfile(t), NewCodec(&recordingWriter{}), discovery, 0)
			p.buffers = bufferTestBudget(t, 2<<20, 0)
			transport := p.plainHTTPTransport()
			dial := transport.DialContext
			var retainOnce sync.Once
			transport.DialContext = func(ctx context.Context, network, address string) (net.Conn, error) {
				if buffers := inferenceBuffers(ctx); buffers != nil {
					retainOnce.Do(func() {
						borrower := buffers.borrowParserAllowance()
						if borrower == nil {
							t.Error("no actual parser borrower to retain")
							return
						}
						buffers.retainRelease(func() { close(closing); <-release; borrower.close() })
					})
				}
				return dial(ctx, network, address)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)).WithContext(ctx)
			var writer http.ResponseWriter = httptest.NewRecorder()
			if scenario == "panic" {
				writer = &registryPanicWriter{httptest.NewRecorder()}
			}
			done := make(chan struct{})
			panicked := make(chan interface{}, 1)
			go func() { defer func() { panicked <- recover(); close(done) }(); p.soleFacade().handleHTTP(writer, r) }()
			if scenario == "cancellation" {
				waitForCond(t, time.Second, "live committed request before cancellation", func() bool {
					keys := p.activeRoutingRequests()
					if len(keys) != 1 {
						return false
					}
					owner := p.routingRequestControl(keys[0])
					owner.mu.Lock()
					defer owner.mu.Unlock()
					return owner.committed
				})
				keys := p.activeRoutingRequests()
				if !p.routingRequestControl(keys[0]).cancel(false) {
					t.Fatal("request cancellation did not win")
				}
			}
			waitChan(t, closing, "actual final borrower release barrier")
			keys := p.activeRoutingRequests()
			p.buffers.mu.Lock()
			used := p.buffers.used
			p.buffers.mu.Unlock()
			if len(keys) != 1 || p.routingRequestControl(keys[0]) == nil || used <= 0 {
				unblock()
				waitChan(t, done, "failed barrier cleanup")
				t.Fatalf("registry disappeared while final borrower remained: entries=%d bytes=%d", len(keys), used)
			}
			select {
			case <-done:
				t.Error("handler returned before final borrower joined")
			default:
			}
			unblock()
			waitChan(t, done, "final borrower and registry cleanup")
			gotPanic := <-panicked
			if scenario == "panic" {
				if gotPanic != "registry cleanup fixture" {
					t.Fatalf("panic unwind=%v", gotPanic)
				}
			} else if gotPanic != nil {
				t.Fatalf("unexpected handler panic: %v", gotPanic)
			}
			ownedAssertHandlerDrained(t, p, nil)
		})
	}
}

func TestRequestRegistryQueuedRegenerateCompatibility(t *testing.T) {
	p, writer := policyProxy(t)
	owner := newRoutingRequest(nil)
	registered, err := p.registerRoutingRequest(owner)
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(registered)
	key := workloadKey{engine: "ollama", id: "queued"}
	if err := p.registerWorkload(registered, key); err != nil {
		t.Fatal(err)
	}
	response := rpc(t, p, writer, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{OriginatedFrom: "test-origin", RunID: p.runID, Engine: key.engine, WorkloadID: key.id, Regenerate: true})
	if response.Error != nil {
		t.Fatalf("queued regenerate RPC: %+v", response.Error)
	}
	var result nodepolicy.WorkloadCancelResult
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatal(err)
	}
	owner.mu.Lock()
	found := result.Found && owner.state == routingPrepared && owner.reason == "regenerating between concrete attempts" && owner.active == nil && owner.lastNode == "" && len(owner.excluded) == 0 && !owner.cancelled
	owner.mu.Unlock()
	if !found || p.routingRequestControl(registered.key) != owner || len(p.workloadControls(key.engine, key.id)) != 1 {
		t.Fatal("queued regenerate fabricated a destination or lost the existing concrete acknowledgement")
	}
}
