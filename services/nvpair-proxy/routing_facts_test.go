// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

const nativeStrictProducerBytes = unsafe.Sizeof(routingFactAttempt{}) + unsafe.Sizeof(routingStrictFactBody{}) + 128
const nativeGraphBytes = unsafe.Sizeof(routingNativeFacts{}) + 32*800 + 528 + 288 + 1616 + 800 + 512 + nativeStrictProducerBytes

// These bounds also fail during cross-compilation, so supported 64-bit target
// layouts can be checked without executing a foreign test binary.
var (
	_ [3328 - unsafe.Sizeof(routingNativeFacts{})]byte
	_ [96 - unsafe.Sizeof(routingNativeAttempt{})]byte
	_ [776 - unsafe.Sizeof(routingDestinationDescriptor{})]byte
	_ [516 - unsafe.Sizeof(routingRequestedDescriptor{})]byte
	_ [288 - unsafe.Sizeof(routingWorkloadDescriptor{})]byte
	_ [512 - unsafe.Sizeof(routingProcessDescriptor{})]byte
	_ [6144 - unsafe.Sizeof(routingSourceDirectory{})]byte
	_ [65536 - unsafe.Sizeof(routingNativeCopy{})]byte
	_ [272 - unsafe.Sizeof(routingSpend{})]byte
	_ [176 - unsafe.Sizeof(publicAttemptLifecycle{})]byte
	_ [32768 - nativeGraphBytes]byte
)

func nativeRegistered(t *testing.T, p *Proxy, at time.Time) *routingRequest {
	t.Helper()
	r, err := p.registerRoutingRequest(newRoutingRequest(nil))
	if err != nil {
		t.Fatal(err)
	}
	p.attachRoutingFacts(r, at)
	t.Cleanup(func() { p.unregisterRoutingRequest(r) })
	return r
}

func nativeCopy(t *testing.T, r *routingSpend) routingNativeCopy {
	t.Helper()
	r.mu.Lock()
	f := r.facts
	r.mu.Unlock()
	if f == nil {
		t.Fatal("native facts detached")
	}
	var out routingNativeCopy
	if !f.copyInto(&out) {
		t.Fatal("stable native copy rejected")
	}
	return out
}

func nativeAttempt(t *testing.T, r *routingSpend, engine string) *publicAttemptLifecycle {
	t.Helper()
	a, err := r.newDispatchAttempt(context.Background(), func() {}, "node", destinationDescriptor("m", "node", engine, false))
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func nativeOnlySource(t *testing.T, p *Proxy) routingNativeCopy {
	t.Helper()
	d := p.routingFactSource()
	if d == nil || !d.scan() || d.reader.count != 1 {
		t.Fatal("expected one retained native source")
	}
	defer d.reader.clear()
	var out routingNativeCopy
	if !d.accept(0, 0, &out) {
		t.Fatal("source copy refused")
	}
	return out
}

func TestRoutingNativeFactsLayouts(t *testing.T) {
	if strconv.IntSize != 64 {
		t.Skip("C4 specifies supported 64-bit targets")
	}
	for _, tc := range []struct {
		name       string
		got, limit uintptr
	}{
		{"request payload", unsafe.Sizeof(routingNativeFacts{}), 3328},
		{"attempt cell", unsafe.Sizeof(routingNativeAttempt{}), 96},
		{"destination", unsafe.Sizeof(routingDestinationDescriptor{}), 776},
		{"requested", unsafe.Sizeof(routingRequestedDescriptor{}), 516},
		{"workload", unsafe.Sizeof(routingWorkloadDescriptor{}), 288},
		{"process", unsafe.Sizeof(routingProcessDescriptor{}), 512},
		{"directory including sole scan bank", unsafe.Sizeof(routingSourceDirectory{}), 6144},
		{"pointer bank", unsafe.Sizeof(routingSourceScan{}.sources), 2048},
		{"read scratch", unsafe.Sizeof(routingNativeCopy{}), 65536},
		{"control owner", unsafe.Sizeof(routingSpend{}), 264 + 8},
		{"actual attempt including D baseline and arbitrary callback tails", unsafe.Sizeof(publicAttemptLifecycle{}), 176},
		{"strict observation handle", unsafe.Sizeof(routingFactAttempt{}), 16},
		{"strict raw wrapper", unsafe.Sizeof(routingStrictFactBody{}), 24},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.got > tc.limit {
				t.Fatalf("layout %d exceeds %d", tc.got, tc.limit)
			}
		})
	}
	// Concurrent first registrations can each prepare one process descriptor
	// before the sole directory wins publication. Its startup draft also fits.
	// Ordinary attempts add zero bytes over their original 176B for ANY
	// simultaneous live/tail count. Their monotonic word is inside the ring;
	// the 40B D baseline replaces duplicate reason/state headers and padding.
	// Strict bypasses ordinary dispatch: exactly one 16B observation handle and
	// one 24B raw wrapper, plus 128B for added producer closure/capture backing.
	// They are handler-owned, or borrowed by the joined strict peek. None is
	// captured by AfterFunc/admission callbacks; the result owner also clears
	// statusCapture's completed body link before an old flush can retain it.
	// Thus none survives actual unregister as an added callback-tail graph.
	if nativeGraphBytes > 32768 {
		t.Fatalf("native graph plus all drafts/winning overlap=%d", nativeGraphBytes)
	}
	// Fixed proxy control C separately funds the ledger observation counter,
	// both nested registry sidecar pointer fields (one remains nil), and the
	// single monotonic reference. No inference grant funds these 48 bytes.
	fixedControl := unsafe.Sizeof(uint64(0)) + 2*unsafe.Sizeof((*routingSourceDirectory)(nil)) + unsafe.Sizeof(routingFactsClock)
	if fixedControl != 48 || 6144+2048+512+26416+fixedControl > 131072 {
		t.Fatalf("fixed C control accounting=%d", fixedControl)
	}
	// The native historical graph may point only to immutable inline leaves.
	for _, kind := range []reflect.Type{reflect.TypeOf(routingRequestedDescriptor{}), reflect.TypeOf(routingDestinationDescriptor{}), reflect.TypeOf(routingWorkloadDescriptor{}), reflect.TypeOf(routingProcessDescriptor{})} {
		for i := 0; i < kind.NumField(); i++ {
			switch kind.Field(i).Type.Kind() {
			case reflect.String, reflect.Pointer, reflect.Slice, reflect.Map, reflect.Interface, reflect.Func:
				t.Fatalf("descriptor %s retains %s", kind.Name(), kind.Field(i).Name)
			}
		}
	}
}

func TestRoutingNativeFactsEntry(t *testing.T) {
	for _, tc := range []struct {
		name, body, path string
		model            string
		mode             uint8
		reason           routingFactReason
	}{
		{"invalid identifier remains empty", `{"model":"bad\nmodel"}`, routeChatCompletions, "", 0, factRequestInvalid},
		{"valid alias with no owner", `{"model":"m"}`, routeChatCompletions, "m", 1, factNoOwner},
		{"embedding operation", `{"model":"m"}`, "/api/embed", "m", 3, factNoOwner},
		{"invalid stream retains validated identifier", `{"model":"m","stream":"yes"}`, routeChatCompletions, "m", 1, factRequestInvalid},
		{"strict early grammar", `{"model":"m"}`, routeChatCompletions, "m", 1, factRequestInvalid},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newTestProxy(ollamaOnlyProfile(t), NewCodec(&recordingWriter{}), NewDiscovery(), 0)
			r := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			if tc.name == "strict early grammar" {
				r.Header.Set(nodepolicy.StrictAdmissionHeader, "not-json")
			}
			w := httptest.NewRecorder()
			start := time.Now().Add(-time.Second)
			p.soleFacade().serveInference(w, r, start, "9", defaultStreamFrameIdle, concreteDispatchPolicy(start))
			out := nativeOnlySource(t, p)
			if out.id != 1 || out.created != uint64(start.UnixMilli()) || out.ended < out.created || out.flags&(factEndKnown|factFrozen) != factEndKnown|factFrozen || out.total != 0 || out.served != 0 || out.binding != 0 {
				t.Fatalf("early record identity/lifetime: %+v", out)
			}
			if got := string(out.requested.model[:out.requested.length]); got != tc.model || out.requested.mode != tc.mode || routingFactReason(out.disposition>>8&255) != tc.reason {
				t.Fatalf("requested=%q mode=%d reason=%d", got, out.requested.mode, out.disposition>>8&255)
			}
			if len(p.activeRoutingRequests()) != 0 || p.requestSequence != 1 {
				t.Fatal("native capture changed registry cleanup")
			}
			bufferAssertTotals(t, p.buffers, 0, 0, 0)
		})
	}
}

func TestRequestOptionsNativeFactsInvalidHygiene(t *testing.T) {
	p, _, engine := reviewR2TierFixture(t, false)
	pol := p.admission.currentPolicy()
	pol.Profiles = []nodepolicy.Profile{{Name: "m", Engine: "ollama", Model: "m", RequestOptions: json.RawMessage(`{"private_configured_key":1}`), RequestOptionsMode: nodepolicy.RequestOptionsEnforced}}
	p.admission.setPolicy(pol)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","private_configured_key":"private-value","PRIVATE_CONFIGURED_KEY":"other-private-value"}`))
	start := time.Now()
	p.soleFacade().serveInference(w, r, start, "9", defaultStreamFrameIdle, concreteDispatchPolicy(start))
	out := nativeOnlySource(t, p)
	if w.Code != http.StatusBadRequest || engine.hits() != 0 || routingFactReason(out.disposition>>8&255) != factRequestInvalid || out.total != 0 {
		t.Fatalf("status=%d hits=%d facts=%+v", w.Code, engine.hits(), out)
	}
	var refusal struct {
		Code  string `json:"code"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &refusal); err != nil {
		t.Fatalf("decode refusal: %v", err)
	}
	if refusal.Code != "ambiguous-options" || refusal.Error != ambiguousOptionsMessage {
		t.Fatalf("unsafe or untyped refusal=%+v", refusal)
	}
	// Native copies have only fixed inline fields. Inspect their actual bytes;
	// JSON encoding would omit the unexported observation fields.
	encoded := unsafe.Slice((*byte)(unsafe.Pointer(&out)), int(unsafe.Sizeof(out)))
	if strings.Contains(string(encoded), "private") || strings.Contains(string(encoded), "PRIVATE") {
		t.Fatal("native facts retained submitted option data")
	}
	if failureFactReason(fmt.Errorf("preparation: %w", &ambiguousOptionsError{})) != factRequestInvalid {
		t.Fatal("wrapped ambiguity lost request-invalid classification")
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRoutingNativeFactsConcreteDelivery(t *testing.T) {
	for _, model := range []string{"m", ""} {
		name := "known model"
		if model == "" {
			name = "model-less ROOT amendment"
		}
		t.Run(name, func(t *testing.T) {
			e := newRecordingEngine(t, http.StatusOK)
			discovery := NewDiscovery()
			discovery.AddManual(nodeForModel(t, "node", e.URL, "m"))
			p := newTestProxy(ollamaOnlyProfile(t), NewCodec(&recordingWriter{}), discovery, 0)
			body := `{}`
			if model != "" {
				body = `{"model":"m"}`
			}
			w := httptest.NewRecorder()
			p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(body)))
			out := nativeOnlySource(t, p)
			c := out.attempts[0]
			if w.Code != http.StatusOK || e.hits() != 1 || out.total != 1 || routingFactState(out.disposition&255) != factCompleted || out.served != 1 || c.flags&factFirstByteKnown == 0 || c.destination.kind != 1 || c.destination.modelKnown != (model != "") {
				t.Fatalf("delivery status=%d hits=%d state=%d served=%d attempt=%+v", w.Code, e.hits(), out.disposition, out.served, c)
			}
			if string(c.destination.first[:c.destination.firstLen]) != "node" || string(c.destination.second[:c.destination.secondLen]) != "ollama" || c.destination.modelLen != uint16(len(model)) || c.admission != 0 || c.continuation != 0 || c.flags&factMoneyApplicable != 0 {
				t.Fatalf("destination/waits/account=%+v", c)
			}
			if model == "" && out.binding != 0 {
				t.Fatal("model-less request invented a workload")
			}
		})
	}
}

func TestRoutingNativeRevisionNoopsAndSourceVersions(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	row := nativeRegistered(t, p, time.Now())
	r := row.owner
	r.observeRequested(requestedDescriptor("m", "weak", routeChatCompletions))
	before := nativeCopy(t, r)
	r.observeRequested(requestedDescriptor("m", "weak", routeChatCompletions))
	if nativeCopy(t, r).revision != before.revision {
		t.Fatal("equivalent requested metadata advanced revision")
	}
	a := nativeAttempt(t, r, "ollama")
	if !a.acceptResponse() || !a.commit() {
		t.Fatal("raw commitment setup failed")
	}
	a.observeStatus(http.StatusOK)
	a.observeAdmission(12 * time.Millisecond)
	before = nativeCopy(t, r)
	a.commit()
	a.observeStatus(http.StatusOK)
	a.observeAdmission(12 * time.Millisecond)
	if nativeCopy(t, r).revision != before.revision {
		t.Fatal("duplicate raw/status/admission advanced revision")
	}
	o := accountAttemptObservation{version: 2, held: 10, flags: factHeldKnown | factChargeKnown, state: factAccountOK, accounting: factAccountingRetained}
	a.observeAccount(o)
	before = nativeCopy(t, r)
	o.version = 1
	o.held = 99
	a.observeAccount(o)
	o.version = 2
	a.observeAccount(o)
	if nativeCopy(t, r).revision != before.revision || nativeCopy(t, r).held != 10 {
		t.Fatal("stale/duplicate D value accepted")
	}
	r.observeSink(a, routingSinkFact{version: 5, status: http.StatusOK, protocol: true, delivered: true, outcome: streamOutcomeCompleted})
	ended := nativeCopy(t, r).ended
	before = nativeCopy(t, r)
	r.observeSink(a, routingSinkFact{version: 4, reason: factStreamInterrupted})
	if nativeCopy(t, r).revision != before.revision {
		t.Fatal("old sink version advanced revision")
	}
	o.version, o.held, o.charged, o.accounting = 3, 0, 7, factAccountingFailed
	a.observeAccount(o)
	a.finished(publicAttemptTerminal{Err: errChatAccounting})
	out := nativeCopy(t, r)
	if out.ended != ended || routingFactState(out.disposition&255) != factCompleted || out.charged != 7 || out.held != 0 || uint8(out.flags>>16) != factAccountingFailed || out.served != 1 {
		t.Fatalf("late accounting reopened inference: %+v", out)
	}
}

func TestRoutingNativeRevisionRingWrapAndTrimmedAccounting(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	r := nativeRegistered(t, p, time.Now()).owner
	old := nativeAttempt(t, r, "ollama")
	old.observeAccount(accountAttemptObservation{version: 1, held: 20, flags: factHeldKnown | factChargeKnown, accounting: factAccountingRetained})
	old.finished(publicAttemptTerminal{})
	for range 40 {
		a := nativeAttempt(t, r, "ollama")
		a.finished(publicAttemptTerminal{})
	}
	before := nativeCopy(t, r)
	old.observeAccount(accountAttemptObservation{version: 2, charged: 12, flags: factHeldKnown | factChargeKnown, accounting: factAccountingRecorded})
	out := nativeCopy(t, r)
	if out.total != 41 || out.held != 0 || out.charged != 12 || out.attempts[0].ordinal != 33 || out.attempts[0].accountVersion != before.attempts[0].accountVersion || out.attempts[0].held != 0 {
		t.Fatal("trimmed D observation overwrote reused cell or lost aggregate")
	}
	for i := range out.attempts {
		if out.attempts[i].ordinal < 10 || out.attempts[i].ordinal > 41 {
			t.Fatalf("unbounded/nonlatest ring ordinal=%d", out.attempts[i].ordinal)
		}
	}
}

func TestRoutingNativeAssociationGenerations(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	row := nativeRegistered(t, p, time.Now())
	r := row.owner
	key := workloadKey{engine: "ollama", id: "job"}
	if err := p.registerWorkload(row, key); err != nil {
		t.Fatal(err)
	}
	if got := nativeCopy(t, r).binding; got != factAttached {
		t.Fatalf("pending binding=%d", got)
	}
	a := nativeAttempt(t, r, "lmstudio")
	if nativeCopy(t, r).binding&^factAttached != 0 {
		t.Fatal("cross-engine attempt bound the card")
	}
	a.finished(publicAttemptTerminal{})
	a = nativeAttempt(t, r, "ollama")
	if got := nativeCopy(t, r).binding; got != factAttached|2 {
		t.Fatalf("first matching binding=%d", got)
	}
	a.finished(publicAttemptTerminal{})
	for range 33 {
		a = nativeAttempt(t, r, "ollama")
		a.finished(publicAttemptTerminal{})
	}
	if got := nativeCopy(t, r).binding; got != 2 {
		t.Fatalf("trim did not detach presentation binding=%d", got)
	}
	if p.workloadCancelControl(nodepolicy.WorkloadCancelParams{OriginatedFrom: row.key.OriginNodeID, RunID: row.key.ProxyRunID, Engine: "ollama", WorkloadID: "job"}) != r {
		t.Fatal("trim changed cancellation index")
	}
	p.unregisterWorkload(row)
	if err := p.registerWorkload(row, workloadKey{engine: "lmstudio", id: "new-job"}); err != nil {
		t.Fatal(err)
	}
	a = nativeAttempt(t, r, "lmstudio")
	out := nativeCopy(t, r)
	if out.workload.generation != 2 || out.binding != factAttached|a.ordinal {
		t.Fatalf("new generation inherited old binding: %+v", out.workload)
	}
	r.mu.Lock()
	r.nativeWorkloadDetachedLocked(1)
	r.mu.Unlock()
	if after := nativeCopy(t, r); after.revision != out.revision || after.binding != out.binding {
		t.Fatal("stale-generation detach changed current association")
	}
	a.finished(publicAttemptTerminal{})
	api, err := r.newAttempt(destinationDescriptor("api-model", "deployment", "account", true))
	if err != nil {
		t.Fatal(err)
	}
	if got := nativeCopy(t, r).binding; got&factAttached != 0 || got&^factAttached != a.ordinal {
		t.Fatal("API creation lost historical workload or kept live index")
	}
	api.finished(publicAttemptTerminal{})
}

func TestRoutingNativeLifetimeWindowAndRollback(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	start := time.Now().Add(-time.Hour)
	first := nativeRegistered(t, p, start)
	d := p.routingFactSource()
	if !d.scan() {
		t.Fatal("first scan refused")
	}
	oldSource := d.reader.sources[0]
	for i := 1; i < 300; i++ {
		row := nativeRegistered(t, p, start.Add(time.Duration(i)*time.Millisecond))
		p.unregisterRoutingRequest(row)
	}
	if d.count.Load() != 256 || d.reader.sources[0] != oldSource {
		t.Fatal("retained source bank changed behind its reader")
	}
	var out routingNativeCopy
	if d.accept(0, 0, &out) {
		t.Fatal("changed membership accepted old scan")
	}
	first.owner.observeRequested(requestedDescriptor("late-old-source", "", routeChatCompletions))
	rollback := nativeRegistered(t, p, start.Add(-time.Hour))
	p.unregisterRoutingRequest(rollback)
	if !d.scan() {
		t.Fatal("replacement scan refused")
	}
	for i := 0; i < d.reader.count; i++ {
		if !d.accept(i, 0, &out) || out.id < 45 || out.id == 301 {
			t.Fatal("old live update/rollback request reinserted into newest window")
		}
	}
	d.reader.clear()
	for _, f := range d.reader.sources {
		if f != nil {
			t.Fatal("scan clear retained a detached graph")
		}
	}
	p.unregisterRoutingRequest(first)
	if p.requestStorage.source != d || p.requestStorage.lease != nil || len(p.requests) != 0 {
		t.Fatal("last active registry discard lost native directory")
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRoutingNativeRevisionEpochRetryAndLimit(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	r := nativeRegistered(t, p, time.Now()).owner
	d := p.routingFactSource()
	if !d.scan() {
		t.Fatal("scan failed")
	}
	var old, current routingNativeCopy
	if !d.accept(0, 0, &old) {
		t.Fatal("initial copy refused")
	}
	r.observeRequested(requestedDescriptor("new", "", routeChatCompletions))
	if !d.accept(0, old.revision, &current) || current.revision <= old.revision || old.requested.length != 0 {
		t.Fatal("old reader changed or new revision lost")
	}
	if d.accept(0, current.revision, &old) {
		t.Fatal("duplicate revision accepted")
	}
	r.mu.Lock()
	f := r.facts
	f.epoch.Add(1)
	r.mu.Unlock()
	if f.copyInto(&old) {
		t.Fatal("odd epoch copied")
	}
	r.mu.Lock()
	f.epoch.Add(1)
	r.mu.Unlock()
	d.epoch.Add(1)
	if d.scan() || d.reader.count != 0 || d.reader.sources[0] != nil {
		t.Fatal("odd source epoch kept prior bank")
	}
	d.epoch.Add(1)
	r.mu.Lock()
	f.epoch.Store(2 * (factMaxRevision - 1))
	r.mu.Unlock()
	r.observeRequested(requestedDescriptor("overflow", "", routeChatCompletions))
	if !f.copyInto(&current) || current.revision != factMaxRevision || current.flags&(factFrozen|factLimitExceeded) != factFrozen|factLimitExceeded {
		t.Fatal("revision overflow did not freeze observation")
	}
	if _, err := r.newDispatchAttempt(context.Background(), func() {}, "node"); err != nil {
		t.Fatalf("diagnostic overflow changed routing: %v", err)
	}
}

func TestRoutingNativeLifetimeLateCallbacksAndBeforeFreeze(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	row := nativeRegistered(t, p, time.Now())
	r := row.owner
	a := nativeAttempt(t, r, "ollama")
	f := r.facts
	r.finish() // The entry may finish before its final borrower joins.
	a.observeAccount(accountAttemptObservation{version: 1, held: 15, flags: factHeldKnown | factChargeKnown, accounting: factAccountingRetained})
	if r.facts != f || nativeCopy(t, r).held != 15 {
		t.Fatal("generic finish detached or froze observations")
	}
	p.unregisterRoutingRequest(row)
	var frozen, later routingNativeCopy
	if !f.copyInto(&frozen) || r.facts != nil || frozen.flags&factFrozen == 0 {
		t.Fatal("actual unregister did not detach/freeze")
	}
	ctx, cancel := context.WithCancel(context.Background())
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	context.AfterFunc(ctx, func() {
		close(entered)
		<-release
		r.expireBeforeOutput()
		a.firstSuccessfulByte()
		a.observeAccount(accountAttemptObservation{version: 2, charged: 9, flags: factChargeKnown})
		close(done)
	})
	cancel()
	<-entered
	close(release)
	<-done
	if !f.copyInto(&later) || later != frozen {
		t.Fatal("late callback retained/reacquired mutable native payload")
	}
}

func TestRoutingNativeFactsTypedCancellation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		cause routingFactReason
		state routingFactState
	}{
		{"client", factClientGone, factCancelled}, {"cancel", factCancel, factCancelled}, {"shutdown", factProxyShutdown, factCancelled}, {"deadline", factDeadline, factFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newBoundProxy(NewCodec(&recordingWriter{}))
			row, err := p.registerRoutingRequest(newRoutingRequest(nil))
			if err != nil {
				t.Fatal(err)
			}
			if !row.owner.cancelWithCause("synthetic unrelated prose", tc.cause) {
				t.Fatal("cancel did not win")
			}
			p.attachRoutingFacts(row, time.Now().Add(time.Hour))
			out := nativeCopy(t, row.owner)
			if routingFactState(out.disposition&255) != tc.state || routingFactReason(out.disposition>>8&255) != tc.cause || out.ended < out.created {
				t.Fatal("pre-attach cancellation lost typed cause/backward-clock end")
			}
			before := out.revision
			row.owner.cancelWithCause("different prose", factUnknown)
			if nativeCopy(t, row.owner).revision != before {
				t.Fatal("losing cancellation advanced revision")
			}
			p.unregisterRoutingRequest(row)
		})
	}
}

func TestRoutingNativeFactsCredentialRefusalBeforeReservation(t *testing.T) {
	f := newChatAccountFixture(t)
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	r := nativeRegistered(t, p, time.Now()).owner
	snapshot, err := f.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	a, err := r.newAttempt(destinationDescriptor(snapshot.deployment.Model, snapshot.deployment.ID, snapshot.account.ID, true))
	if err != nil {
		t.Fatal(err)
	}
	lookups := 0
	resolve := func(nodepolicy.CredentialRef) (string, bool) {
		lookups++
		return "", false
	}
	_, err = runChatAccountAttempt(f.ctx, snapshot, []byte(f.body), f.profile, f.ledger, f.transport, resolve, f.budget.acquire, chatTestGuard(f.guard), a, f.emit)
	if !errors.Is(err, errChatCredential) || lookups != 1 || f.calls.Load() != 0 {
		t.Fatalf("credential refusal: err=%v lookups=%d calls=%d", err, lookups, f.calls.Load())
	}
	if len(f.ledger.state.Holds) != 0 {
		t.Fatal("credential refusal created a ledger hold")
	}
	out := nativeCopy(t, r)
	c := out.attempts[0]
	if uint8(c.flags>>24) != factAccountBadCredential || routingFactReason(c.disposition>>8&255) != factAccountCredential || c.flags&(factMoneyApplicable|factHeldKnown|factChargeKnown|factEndKnown) != factMoneyApplicable|factHeldKnown|factChargeKnown|factEndKnown {
		t.Fatal("typed pre-reservation credential refusal lost account state or known-zero amounts")
	}
	if c.held != 0 || c.charged != 0 || c.disposition>>16 != 0 || uint8(c.flags>>16) != factAccountingNone {
		t.Fatal("unsent credential refusal invented money, status or accounting")
	}
}

func TestRoutingNativeFactsAccountOwners(t *testing.T) {
	f := newChatAccountFixture(t)
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	r := nativeRegistered(t, p, time.Now()).owner
	snapshot, err := f.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	a, err := r.newAttempt(destinationDescriptor(snapshot.deployment.Model, snapshot.deployment.ID, snapshot.account.ID, true))
	if err != nil {
		t.Fatal(err)
	}
	remaining, err := a.tryAccount(snapshot.account)
	if err != nil {
		t.Fatal(err)
	}
	reservation, err := f.ledger.reserveChatAccountAllowance(snapshot, 1, 1, remaining, a)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.associated(reservation); err != nil {
		t.Fatal(err)
	}
	reserved, err := reservation.Hold.Reserved.Int64()
	if err != nil {
		t.Fatal(err)
	}
	before := nativeCopy(t, r)
	if before.held != uint64(reserved) || before.granted != uint64(reserved) || before.flags&factCeilingKnown == 0 || before.attempts[0].destination.kind != 2 || before.attempts[0].flags&factMoneyApplicable == 0 {
		t.Fatal("D reserve did not transfer native held/granted/account applicability")
	}
	if err := f.ledger.recordOutcome(reservation.Hold.Attempt, outcomeOK, 0, time.Time{}, a); err != nil {
		t.Fatal(err)
	}
	if err := f.ledger.reconcile(reservation.Hold.Attempt, nil, a); err != nil {
		t.Fatal(err)
	}
	if err := f.ledger.recordQualification(reservation.Hold.Attempt, a); err != nil {
		t.Fatal(err)
	}
	out := nativeCopy(t, r)
	if out.held != 0 || out.charged != uint64(reserved) || out.granted != uint64(reserved) || uint8(out.flags>>16) != factAccountingRecorded {
		t.Fatal("settlement refilled grant or lost known charge")
	}
	before = out
	if err := f.ledger.recordQualification(reservation.Hold.Attempt, a); err != nil {
		t.Fatal(err)
	}
	if nativeCopy(t, r).revision != before.revision {
		t.Fatal("duplicate D qualification advanced revision")
	}
	a.finished(publicAttemptTerminal{})
}

func TestRoutingNativeFactsDeliveredAccountingFailure(t *testing.T) {
	f := newChatAccountFixture(t)
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	r := nativeRegistered(t, p, time.Now()).owner
	w := &responseSinkWriter{header: make(http.Header)}
	r.sink = newResponseSink(w, func() { r.cancelWithCause("synthetic client loss", factClientGone) })
	f.ctx = withResponseSink(f.ctx, r.sink)
	snapshot, err := f.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	a, err := r.newAttempt(destinationDescriptor(snapshot.deployment.Model, snapshot.deployment.ID, snapshot.account.ID, true))
	if err != nil {
		t.Fatal(err)
	}
	failed := false
	f.ledger.write = func(path string, data []byte, mode os.FileMode) error {
		s, err := decodeLedger(data)
		if err != nil {
			return err
		}
		if len(s.Holds) > 0 && s.Holds[0].State == holdReconciled && !failed {
			failed = true
			return errors.New("synthetic accounting write failure")
		}
		return durablefile.WriteFile(path, data, mode)
	}
	result, err := runChatAccountAttempt(f.ctx, snapshot, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, chatTestGuard(f.guard), a, f.emit)
	if !errors.Is(err, errChatAccounting) || !failed || !result.Complete {
		t.Fatalf("accounting failure setup: complete=%v err=%v", result.Complete, err)
	}
	out := nativeCopy(t, r)
	if routingFactState(out.disposition&255) != factCompleted || out.served != 1 || uint8(out.flags>>16) != factAccountingFailed || out.held == 0 || out.charged != 0 {
		t.Fatalf("delivered accounting failure facts: state=%d served=%d accounting=%d held=%d charge=%d", out.disposition, out.served, out.flags>>16, out.held, out.charged)
	}
}

func TestRoutingNativeFactsStrictEntry(t *testing.T) {
	p, engine := strictHTTPFixture(t)
	w := httptest.NewRecorder()
	serveStrictHTTP(p, "self", w, strictHTTPRequestWith(t, strictBody("m")))
	out := nativeOnlySource(t, p)
	if w.Code != http.StatusOK || engine.hits() != 1 || out.total != 1 || out.served != 1 || routingFactState(out.disposition&255) != factCompleted || out.attempts[0].flags&factFirstByteKnown == 0 {
		t.Fatalf("strict entry missing actual source facts: status=%d hits=%d state=%d served=%d", w.Code, engine.hits(), out.disposition, out.served)
	}
	if out.binding != 0 || len(p.activeRoutingRequests()) != 0 {
		t.Fatal("strict observation created workload/active lifetime")
	}
}

func TestRoutingNativeFactsStrictProtocolAndFailedFlush(t *testing.T) {
	t.Run("managed protocol completion without sink", func(t *testing.T) {
		p, _ := strictHTTPFixture(t)
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if _, err := io.Copy(io.Discard, r.Body); err != nil {
				t.Errorf("read synthetic request: %v", err)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			if _, err := io.WriteString(w, continuitySSEEvent+"data: [DONE]\n\n"); err != nil {
				t.Errorf("write synthetic stream: %v", err)
			}
		}))
		defer upstream.Close()
		setBackend(t, p.soleFacade(), upstream.URL, true)
		body := strings.TrimSuffix(strictBody("m"), "}") + `,"stream":true}`
		w := httptest.NewRecorder()
		serveStrictHTTP(p, "self", w, strictHTTPRequestWith(t, body))
		out := nativeOnlySource(t, p)
		if w.Code != http.StatusOK || routingFactState(out.disposition&255) != factCompleted || out.served != 1 || routingFactOutcome(out.attempts[0].disposition&255) != factCommitted || out.ended != out.attempts[0].ended {
			t.Fatal("strict protocol owner did not supply coherent completion/delivery")
		}
		if err := nativeEndError(out); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("failed flush is observed without ordinary cancellation effects", func(t *testing.T) {
		p, engine := strictHTTPFixture(t)
		w := &responseSinkWriter{header: make(http.Header), flushErr: io.ErrClosedPipe}
		serveStrictHTTP(p, "self", w, strictHTTPRequestWith(t, strictBody("m")))
		out := nativeOnlySource(t, p)
		if engine.hits() != 1 || out.served != 0 || routingFactState(out.disposition&255) != factCancelled || routingFactReason(out.disposition>>8&255) != factClientGone {
			t.Fatal("strict statusCapture flush failure was not observed")
		}
		if err := nativeEndError(out); err != nil {
			t.Fatal(err)
		}
	})
}

func TestRoutingNativeRevisionConcurrentCopy(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	r := nativeRegistered(t, p, time.Now()).owner
	f := r.facts
	type production struct {
		ordinal uint64
		err     error
	}
	progress, resume := make(chan uint64), make(chan struct{})
	result, exited, stop := make(chan production, 1), make(chan struct{}), make(chan struct{})
	defer func() { close(stop); <-exited }()
	go func() {
		defer close(exited)
		var completed uint64
		for range 128 {
			a, err := r.newDispatchAttempt(context.Background(), func() {}, "node", destinationDescriptor("m", "node", "ollama", false))
			if err != nil {
				result <- production{completed, err}
				return
			}
			// An accepted copy must observe this actual in-flight attempt
			// before its producer may publish status/end or create the next.
			select {
			case progress <- a.ordinal:
			case <-stop:
				return
			}
			select {
			case <-resume:
			case <-stop:
				return
			}
			a.observeStatus(http.StatusOK)
			a.finished(publicAttemptTerminal{})
			completed = a.ordinal
		}
		result <- production{completed, nil}
	}()
	acceptedDuringProduction := 0
	check := func(out routingNativeCopy) {
		t.Helper()
		for _, c := range out.attempts {
			if c.ordinal != 0 && (c.destination.kind != 1 || c.destination.modelLen != 1 || c.ordinal > out.total) {
				t.Fatal("coherent epoch exposed partial descriptor/ordinal")
			}
		}
	}
	for expected := uint64(1); expected <= 128; expected++ {
		select {
		case ordinal := <-progress:
			if ordinal != expected {
				t.Fatalf("producer ordinal=%d want=%d", ordinal, expected)
			}
		case done := <-result:
			t.Fatalf("producer stopped before reader overlap: ordinal=%d err=%v", done.ordinal, done.err)
		case <-time.After(time.Second):
			t.Fatal("producer did not reach overlap barrier")
		}
		var out routingNativeCopy
		if !f.copyInto(&out) || out.total != expected || out.attempts[(expected-1)%32].flags&factEndKnown != 0 {
			t.Fatal("no accepted copy of the in-flight producer phase")
		}
		check(out)
		acceptedDuringProduction++
		resume <- struct{}{}
		// These reads also race actual status/end and next-cell publication;
		// an odd or changed epoch may reject, but every accepted copy is checked.
		for range 8 {
			if f.copyInto(&out) {
				check(out)
			}
		}
	}
	done := <-result
	if done.err != nil || done.ordinal != 128 || acceptedDuringProduction != 128 {
		t.Fatalf("production/copies: %+v accepted=%d", done, acceptedDuringProduction)
	}
	var final routingNativeCopy
	if !f.copyInto(&final) || final.total != 128 || final.attempts[31].ordinal != 128 || final.attempts[31].flags&factEndKnown == 0 {
		t.Fatal("final producer ordinal/end was not published")
	}
}

func nativeEndError(out routingNativeCopy) error {
	for _, c := range out.attempts {
		if c.ordinal == 0 {
			continue
		}
		if c.started < out.created {
			return fmt.Errorf("attempt %d starts before request", c.ordinal)
		}
		if c.flags&factEndKnown != 0 && c.ended < c.started {
			return fmt.Errorf("attempt %d ends before start", c.ordinal)
		}
		if out.flags&factEndKnown != 0 && (c.flags&factEndKnown == 0 || c.ended > out.ended || c.started > out.ended) {
			return fmt.Errorf("terminal request has unfinished/out-of-bounds attempt %d", c.ordinal)
		}
	}
	return nil
}

func TestRoutingNativeRevisionCancellationIsOnePublication(t *testing.T) {
	for _, strict := range []bool{false, true} {
		name := "ordinary active attempt"
		if strict {
			name = "strict observation without active slot"
		}
		t.Run(name, func(t *testing.T) {
			p := newBoundProxy(NewCodec(&recordingWriter{}))
			r := nativeRegistered(t, p, time.Now().Add(time.Hour)).owner
			if strict {
				if r.strictFactAttempt(destinationDescriptor("m", "node", "ollama", false)) == nil {
					t.Fatal("strict observation absent")
				}
			} else {
				nativeAttempt(t, r, "ollama")
			}
			before := nativeCopy(t, r)
			f := r.facts
			type readResult struct {
				accepted int
				err      error
			}
			ready, stop, exited := make(chan struct{}), make(chan struct{}), make(chan struct{})
			read := make(chan readResult, 1)
			var once sync.Once
			stopReader := func() { once.Do(func() { close(stop) }) }
			defer func() { stopReader(); <-exited }()
			// The reader must accept the old revision while the owner lock is
			// held, then overlap the real cancel transition after its release.
			r.mu.Lock()
			go func() {
				defer close(exited)
				n := 0
				for {
					var out routingNativeCopy
					if f.copyInto(&out) {
						n++
						if n == 1 {
							close(ready)
						}
						if err := nativeEndError(out); err != nil {
							read <- readResult{n, err}
							return
						}
					}
					select {
					case <-stop:
						read <- readResult{accepted: n}
						return
					default:
					}
				}
			}()
			<-ready
			cancelled := make(chan bool, 1)
			go func() { cancelled <- r.cancel(false) }()
			r.mu.Unlock()
			if !<-cancelled {
				t.Fatal("cancel did not win")
			}
			after := nativeCopy(t, r)
			stopReader()
			reads := <-read
			if reads.err != nil || reads.accepted == 0 {
				t.Fatalf("reader overlap: %+v", reads)
			}
			if after.revision != before.revision+1 || after.ended != after.attempts[0].ended || routingFactReason(after.disposition>>8&255) != factCancel || routingFactReason(after.attempts[0].disposition>>8&255) != factCancel {
				t.Fatalf("cancel was not one coherent event: revisions=%d/%d ends=%d/%d", before.revision, after.revision, after.ended, after.attempts[0].ended)
			}
			if err := nativeEndError(after); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRoutingNativeFactsTerminalClockBounds(t *testing.T) {
	for _, registrationRollback := range []bool{true, false} {
		name := "rollback after registration"
		if !registrationRollback {
			name = "attempt start ahead of terminal wall clock"
		}
		t.Run(name, func(t *testing.T) {
			p := newBoundProxy(NewCodec(&recordingWriter{}))
			created, started := time.Now(), time.Now().Add(2*time.Hour)
			if registrationRollback {
				created, started = time.Now().Add(time.Hour), time.Now()
			}
			r := nativeRegistered(t, p, created).owner
			d := destinationDescriptor("m", "node", "ollama", false)
			r.mu.Lock()
			a, err := r.newAttemptLocked(started, d)
			r.mu.Unlock()
			if err != nil || !a.acceptResponse() || !a.commit() {
				t.Fatalf("attempt setup: %v", err)
			}
			r.observeSink(a, routingSinkFact{version: 1, status: http.StatusOK, protocol: true, delivered: true, outcome: streamOutcomeCompleted})
			out := nativeCopy(t, r)
			if out.created != uint64(created.UnixMilli()) || out.ended != out.attempts[0].ended || out.attempts[0].started != uint64(max(created.UnixMilli(), started.UnixMilli())) {
				t.Fatalf("incoherent event bounds: created=%d start=%d request end=%d attempt end=%d", out.created, out.attempts[0].started, out.ended, out.attempts[0].ended)
			}
			if err := nativeEndError(out); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRoutingNativeLifetimeDelayedCancellationBeforeFreeze(t *testing.T) {
	for _, tc := range []struct {
		name                           string
		sinkResult, process, completed bool
	}{
		{name: "cancelled stream result before delayed AfterFunc", sinkResult: true},
		{name: "finish folds client cancellation without result callback"},
		{name: "finish folds process cancellation", process: true},
		{name: "completed inference stays completed", completed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newBoundProxy(NewCodec(&recordingWriter{}))
			row := nativeRegistered(t, p, time.Now())
			r := row.owner
			ctx, cancel := context.WithCancel(context.Background())
			process, cancelProcess := context.WithCancel(context.Background())
			defer cancel()
			defer cancelProcess()
			r.mu.Lock()
			r.observationContext, r.processContext = ctx, process
			r.mu.Unlock()
			a := nativeAttempt(t, r, "ollama")
			if !a.acceptResponse() || !a.commit() {
				t.Fatal("raw commitment setup failed")
			}
			f := r.facts
			if tc.completed {
				r.observeSink(a, routingSinkFact{version: 1, status: http.StatusOK, protocol: true, delivered: true, outcome: streamOutcomeCompleted})
			}
			before := nativeCopy(t, r)
			entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer func() { unblock(); <-done }()
			context.AfterFunc(ctx, func() {
				close(entered)
				<-release
				r.cancelWithCause("client disconnected before completion", routingContextCause(ctx, process))
				close(done)
			})
			if tc.process {
				cancelProcess()
			}
			cancel()
			<-entered // Cancellation is visible, its existing owner callback has not run.
			if tc.sinkResult {
				sink := newResponseSink(httptest.NewRecorder(), nil)
				sink.streamResult(streamFrameResult{Outcome: streamOutcomeCancelled})
				r.observeSink(a, sink.nativeFact())
			}
			p.unregisterRoutingRequest(row)
			var frozen routingNativeCopy
			if !f.copyInto(&frozen) || frozen.flags&(factEndKnown|factFrozen) != factEndKnown|factFrozen {
				t.Fatal("freeze retained an open cancelled request")
			}
			if err := nativeEndError(frozen); err != nil {
				t.Fatal(err)
			}
			if tc.completed {
				if frozen.disposition != before.disposition || frozen.ended != before.ended || frozen.served != before.served {
					t.Fatal("late cancellation reopened completed inference")
				}
			} else {
				cause := factClientGone
				if tc.process {
					cause = factProxyShutdown
				}
				if routingFactState(frozen.disposition&255) != factCancelled || routingFactReason(frozen.disposition>>8&255) != cause || routingFactReason(frozen.attempts[0].disposition>>8&255) != cause {
					t.Fatal("typed cancellation was lost before done/freeze")
				}
			}
			r.mu.Lock()
			detached := r.facts == nil && r.observationContext == nil && r.processContext == nil
			r.mu.Unlock()
			if !detached {
				t.Fatal("unregistered tail retained observation sources")
			}
			unblock()
			<-done
			var after routingNativeCopy
			if !f.copyInto(&after) || after != frozen {
				t.Fatal("delayed AfterFunc mutated frozen source")
			}
		})
	}
}

// Keep structured decoding in entry tests explicit; no raw serialized-content
// match is evidence for privacy or protocol shape.
func TestRoutingNativeFactsNoPublicEmitter(t *testing.T) {
	out := &recordingWriter{}
	p := newTestProxy(ollamaOnlyProfile(t), NewCodec(out), NewDiscovery(), 0)
	p.soleFacade().handleHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m"}`)))
	for _, line := range out.lines() {
		var envelope struct {
			Method string `json:"method"`
		}
		if err := json.Unmarshal(line, &envelope); err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(envelope.Method, "routing/history") || envelope.Method == "routing/request" {
			t.Fatal("Task312 activated an emitter")
		}
	}
}

func TestRoutingNativeLifetimeLastBorrower(t *testing.T) {
	setForTest(t, &firstBodyTimeout, 50*time.Millisecond)
	headers, send := make(chan struct{}), make(chan struct{})
	var sendOnce sync.Once
	unblockEngine := func() { sendOnce.Do(func() { close(send) }) }
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		close(headers)
		<-send
		_, _ = io.WriteString(w, continuitySSEEvent+"data: [DONE]\n\n")
	}))
	defer engine.Close()
	defer unblockEngine()
	fallback := newRecordingEngine(t, http.StatusOK)
	discovery := NewDiscovery()
	discovery.AddManual(nodeForModel(t, "a", engine.URL, "m"))
	discovery.AddManual(nodeForModel(t, "b", fallback.URL, "m"))
	p := newTestProxy(ollamaOnlyProfile(t), NewCodec(&recordingWriter{}), discovery, 0)
	p.SetPriority([]string{"a", "b"})
	w, done := httptest.NewRecorder(), make(chan struct{})
	go func() {
		defer close(done)
		p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
	}()
	waitChan(t, headers, "upstream headers")
	keys := p.activeRoutingRequests()
	if len(keys) != 1 {
		t.Fatalf("active=%d", len(keys))
	}
	r := p.routingRequestControl(keys[0])
	committed, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	unblockCallback := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblockCallback()
	r.mu.Lock()
	stop := r.stopPreOutput
	var blocked atomic.Bool
	r.stopPreOutput = func() bool {
		if blocked.CompareAndSwap(false, true) {
			close(committed)
			<-release
		}
		return stop()
	}
	r.mu.Unlock()
	unblockEngine()
	waitChan(t, committed, "raw success before peek publication")
	waitChan(t, done, "handler exit while raw borrower remains")
	r.mu.Lock()
	pending, f, finished := r.firstBodyDone, r.facts, r.done
	r.mu.Unlock()
	if pending == nil || f == nil || !finished || p.routingRequestControl(keys[0]) != r {
		t.Fatal("pending raw borrower lost native/registry owner")
	}
	var before routingNativeCopy
	if !f.copyInto(&before) || before.flags&factFrozen != 0 || before.flags&factEndKnown == 0 || before.binding&factAttached != 0 {
		t.Fatal("generic finish froze too early or missed final detach")
	}
	select {
	case <-pending:
		t.Fatal("raw borrower completed before callback release")
	default:
	}
	unblockCallback()
	waitChan(t, pending, "last raw borrower join")
	waitForCond(t, time.Second, "actual unregister", func() bool { return len(p.activeRoutingRequests()) == 0 })
	r.mu.Lock()
	detached := r.facts == nil
	r.mu.Unlock()
	var after routingNativeCopy
	if !detached || !f.copyInto(&after) || after.flags&factFrozen == 0 || after.ended != before.ended || fallback.hits() != 0 {
		t.Fatal("last-borrower freeze changed inference or end time")
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRoutingNativeFactsAccountRefusals(t *testing.T) {
	for _, tc := range []struct {
		name    string
		outcome accountOutcome
		state   uint8
		reason  routingFactReason
	}{
		{"cooldown", outcomeRateLimited, factAccountCooling, factAccountCooldown},
		{"exhausted", outcomeExhausted, factAccountIsExhausted, factAccountExhausted},
		{"credential", outcomeCredentialFailure, factAccountBadCredential, factAccountCredential},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			snapshot, err := f.ledger.prepareChatAccount("chat-a")
			if err != nil {
				t.Fatal(err)
			}
			hold, err := f.ledger.reserveChatAccountAllowance(snapshot, 1, 1, 1000000000)
			if err != nil {
				t.Fatal(err)
			}
			if err := f.ledger.recordOutcome(hold.Hold.Attempt, tc.outcome, time.Minute, f.ledger.now().Add(time.Hour)); err != nil {
				t.Fatal(err)
			}
			p := newBoundProxy(NewCodec(&recordingWriter{}))
			r := nativeRegistered(t, p, time.Now()).owner
			a, err := r.newAttempt(destinationDescriptor(snapshot.deployment.Model, snapshot.deployment.ID, snapshot.account.ID, true))
			if err != nil {
				t.Fatal(err)
			}
			reservation, reserveErr := f.ledger.reserveChatAccountAllowance(snapshot, 1, 1, 1000000000, a)
			if reserveErr == nil || reservation.Durability != accountReserveNone {
				t.Fatal("expected actual D no-hold refusal")
			}
			out := nativeCopy(t, r)
			c := out.attempts[0]
			if uint8(c.flags>>24) != tc.state || routingFactReason(c.disposition>>8&255) != tc.reason || c.held != 0 || c.charged != 0 || c.flags&(factHeldKnown|factChargeKnown) != factHeldKnown|factChargeKnown {
				t.Fatalf("refusal state=%d reason=%d flags=%d", c.flags>>24, c.disposition>>8&255, c.flags)
			}
			a.finished(publicAttemptTerminal{})
		})
	}
}

func TestRoutingNativeFactsPaidCeilingAndUncertainReserve(t *testing.T) {
	f := newChatAccountFixture(t)
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	r := nativeRegistered(t, p, time.Now()).owner
	snapshot, err := f.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	a, err := r.newAttempt(destinationDescriptor(snapshot.deployment.Model, snapshot.deployment.ID, snapshot.account.ID, true))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "100"}); err != nil {
		t.Fatal(err)
	}
	before := nativeCopy(t, r)
	if _, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "100"}); err != nil {
		t.Fatal(err)
	}
	if nativeCopy(t, r).revision != before.revision {
		t.Fatal("duplicate paid ceiling advanced revision")
	}
	f.ledger.write = func(path string, data []byte, mode os.FileMode) error {
		if err := durablefile.WriteFile(path, data, mode); err != nil {
			return err
		}
		// Cancellation wins during the durable reserve, while D still owns
		// its lock. The later transfer must not deadlock or lose liability.
		r.cancel(false)
		return durablefile.ErrReplaced
	}
	reservation, reserveErr := f.ledger.reserveChatAccountAllowance(snapshot, 1, 1, 100, a)
	if reserveErr == nil || reservation.Durability != accountReserveUncertain {
		t.Fatalf("expected uncertain reserve: %v %+v", reserveErr, reservation)
	}
	if err := a.associated(reservation); err != nil {
		t.Fatal(err)
	}
	out := nativeCopy(t, r)
	if out.held == 0 || out.granted != out.held || !r.uncertain || routingFactState(out.disposition&255) != factCancelled {
		t.Fatal("cancelled uncertain reserve lost landed liability")
	}
	if routingFactReason(out.attempts[0].disposition>>8&255) != factCancel {
		t.Fatal("late D failure replaced the winning cancellation cause")
	}
	if _, err := r.newAttempt(); !errors.Is(err, errRoutingSpend) {
		t.Fatal("observation granted retry after uncertain reserve")
	}
	if _, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "50"}); !errors.Is(err, errRoutingSpend) {
		t.Fatal("cancelled attempt accepted allowance")
	}
	if nativeCopy(t, r).ceiling != 50 {
		t.Fatal("refused paid candidate did not tighten observed ceiling")
	}
	a.finished(publicAttemptTerminal{})
}

func TestRoutingNativeFactsAccountingSurvivesOwnCommitPrune(t *testing.T) {
	for _, last := range []string{"reconcile", "outcome", "qualification"} {
		t.Run(last, func(t *testing.T) {
			f := newChatAccountFixture(t)
			p := newBoundProxy(NewCodec(&recordingWriter{}))
			r := nativeRegistered(t, p, time.Now()).owner
			snapshot, err := f.ledger.prepareChatAccount("chat-a")
			if err != nil {
				t.Fatal(err)
			}
			a, err := r.newAttempt(destinationDescriptor(snapshot.deployment.Model, snapshot.deployment.ID, snapshot.account.ID, true))
			if err != nil {
				t.Fatal(err)
			}
			remaining, err := a.tryAccount(snapshot.account)
			if err != nil {
				t.Fatal(err)
			}
			reservation, err := f.ledger.reserveChatAccountAllowance(snapshot, 1, 1, remaining, a)
			if err != nil {
				t.Fatal(err)
			}
			if err := a.associated(reservation); err != nil {
				t.Fatal(err)
			}
			complete := func(which string) {
				t.Helper()
				var err error
				switch which {
				case "reconcile":
					err = f.ledger.reconcile(reservation.Hold.Attempt, nil, a)
				case "outcome":
					err = f.ledger.recordOutcome(reservation.Hold.Attempt, outcomeOK, 0, time.Time{}, a)
				case "qualification":
					err = f.ledger.recordQualification(reservation.Hold.Attempt, a)
				}
				if err != nil {
					t.Fatalf("%s: %v", which, err)
				}
			}
			for _, which := range []string{"reconcile", "outcome", "qualification"} {
				if which != last {
					complete(which)
				}
			}
			completeOtherLedgerAttempts(t, f.ledger)
			complete(last)
			if f.ledger.state.hold(reservation.Hold.Attempt) != nil {
				t.Fatal("test did not prune the completed hold in its own commit")
			}
			amount, err := reservation.Hold.Reserved.Int64()
			if err != nil {
				t.Fatal(err)
			}
			out := nativeCopy(t, r)
			if out.held != 0 || out.charged != uint64(amount) || out.granted != uint64(amount) || uint8(out.flags>>16) != factAccountingRecorded {
				t.Fatalf("pruned native totals: held=%d charged=%d granted=%d accounting=%d", out.held, out.charged, out.granted, out.flags>>16)
			}
			if err := f.ledger.recordQualification(reservation.Hold.Attempt, a); !errors.Is(err, errAttemptUnknown) {
				t.Fatalf("pruned duplicate: %v", err)
			}
			if nativeCopy(t, r).revision != out.revision {
				t.Fatal("rejected pruned duplicate changed native facts")
			}
			a.finished(publicAttemptTerminal{})
		})
	}
}
