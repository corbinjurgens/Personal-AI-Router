// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	"nvpair-shared/nodepolicy"
)

func historyTestCycle(t *testing.T, h *proxyRoutingHistory) {
	t.Helper()
	select {
	case <-h.writeWake:
		h.writeCycle()
	default:
		t.Fatal("no history cycle scheduled")
	}
}

func historyTestRecord(t *testing.T, h *proxyRoutingHistory, id uint64) nodepolicy.RoutingHistoryRecord {
	t.Helper()
	at := h.current.find(id)
	if at < 0 {
		t.Fatalf("missing projected key %d", id)
	}
	w := routingHistoryJSON{data: h.frames[1][:0]}
	h.encodeRecord(&w, &h.current.cells[at])
	if w.err != nil {
		t.Fatal(w.err)
	}
	var record nodepolicy.RoutingHistoryRecord
	if err := json.Unmarshal(w.data, &record); err != nil {
		t.Fatalf("emitted record failed shared validation: %v", err)
	}
	return record
}

func historyTestScan(t *testing.T, h *proxyRoutingHistory) {
	t.Helper()
	if !h.reconcile(false) || !h.ready {
		t.Fatal("stable source did not produce a complete window")
	}
	d := h.proxy.routingFactSource()
	if d.reader.count != 0 || d.reader.epoch != 0 {
		t.Fatal("scan metadata survived copier turn")
	}
	for _, f := range d.reader.sources {
		if f != nil {
			t.Fatal("copier retained a native pointer")
		}
	}
	if h.scratch != (routingProjectionCell{}) {
		t.Fatal("scratch was not cleared")
	}
}

func TestRoutingHistoryProjection(t *testing.T) {
	t.Run("delivered success retains failed accounting and exact money", func(t *testing.T) {
		p := newBoundProxy(NewCodec(&recordingWriter{}))
		r := nativeRegistered(t, p, time.UnixMilli(1000))
		r.owner.observeRequested(requestedDescriptor("alias", "strong", "/v1/chat/completions"))
		a, err := r.owner.newAttempt(destinationDescriptor("chosen", "deployment", "account", true))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "9223372036854775807"}); err != nil {
			t.Fatal(err)
		}
		if err := a.associated(accountReservation{Durability: accountReserveDurable, Hold: accountHold{Reserved: "9007199254740993"}}); err != nil {
			t.Fatal(err)
		}
		a.observeAccount(accountAttemptObservation{version: 1, held: 7, charged: 9007199254740992, flags: factHeldKnown | factChargeKnown, state: factAccountOK, accounting: factAccountingFailed})
		a.observeStatus(200)
		a.observeAdmission(17 * time.Millisecond)
		r.owner.observeSink(a, routingSinkFact{version: 1, status: 200, protocol: true, delivered: true})
		h := p.newRoutingHistory()
		historyTestScan(t, h)
		got := historyTestRecord(t, h, 1)
		money := func(p *nodepolicy.NanoUSD) string {
			if p == nil {
				return "null"
			}
			return string(*p)
		}
		if got.State != "completed" || got.Served == nil || got.Served.Kind != "api" || got.Served.Model == nil || *got.Served.Model != "chosen" || got.Requested.Model != "alias" || got.Requested.Tier != "strong" || got.Requested.Mode != "chat" {
			t.Fatalf("destination/request projection: %+v", got)
		}
		if got.Spend.Accounting != "failed" || money(got.Spend.CeilingNanoUSD) != "9223372036854775807" || money(got.Spend.GrantedNanoUSD) != "9007199254740993" || money(got.Spend.HeldNanoUSD) != "7" || money(got.Spend.ChargedNanoUSD) != "9007199254740992" {
			t.Fatalf("money projection: %+v", got.Spend)
		}
		if len(got.Attempts) != 1 || got.Attempts[0].Account == nil || got.Attempts[0].Account.Accounting != "failed" || got.Attempts[0].Waits.AdmissionMs != 17 || got.Attempts[0].Waits.FirstByteMs != nil || got.Attempts[0].Status == nil || *got.Attempts[0].Status != 200 || got.Attempts[0].Outcome != "committed" {
			t.Fatalf("attempt projection: %+v", got.Attempts)
		}
		// Late terminal accounting updates preserve the original inference end.
		ended := *got.EndedAt
		a.observeAccount(accountAttemptObservation{version: 2, held: 0, charged: 9007199254740993, flags: factHeldKnown | factChargeKnown, state: factAccountOK, accounting: factAccountingRecorded})
		historyTestScan(t, h)
		late := historyTestRecord(t, h, 1)
		if late.Seq != got.Seq+1 || late.State != "completed" || *late.EndedAt != ended || money(late.Spend.HeldNanoUSD) != "0" || money(late.Spend.ChargedNanoUSD) != "9007199254740993" || late.Spend.Accounting != "failed" {
			t.Fatalf("late accounting reopened or lost terminal facts: %+v", late)
		}
		p.unregisterRoutingRequest(r)
		historyTestScan(t, h)
		if after := historyTestRecord(t, h, 1); after.State != "completed" || after.Seq != late.Seq+1 || len(p.activeRoutingRequests()) != 0 {
			t.Fatal("final-before-unlink source did not survive independently")
		}
	})
	t.Run("model unknown preserves real local attempt", func(t *testing.T) {
		p := newBoundProxy(NewCodec(&recordingWriter{}))
		r := nativeRegistered(t, p, time.UnixMilli(2000))
		a, err := r.owner.newDispatchAttempt(context.Background(), func() {}, "real-node", destinationDescriptor("", "real-node", "ollama", false))
		if err != nil {
			t.Fatal(err)
		}
		a.observeStatus(200)
		r.owner.observeSink(a, routingSinkFact{version: 1, status: 200, protocol: true, delivered: true})
		h := p.newRoutingHistory()
		historyTestScan(t, h)
		got := historyTestRecord(t, h, 1)
		if len(got.Attempts) != 1 || got.Attempts[0].Destination.Model != nil || got.Requested.Model != "" || got.Served == nil || got.Served.Model != nil || *got.Served.NodeID != "real-node" || *got.Served.Engine != "ollama" || got.State != "completed" {
			t.Fatalf("model-less inference was fabricated/dropped: %+v", got)
		}
		if got.Spend.GrantedNanoUSD != nil || got.Spend.HeldNanoUSD != nil || got.Spend.CeilingNanoUSD != nil || got.Spend.ChargedNanoUSD != nil || got.Spend.Accounting != "none" {
			t.Fatal("unknown money became known zero")
		}
	})
	t.Run("prefix trim preserves historical workload binding", func(t *testing.T) {
		p := newBoundProxy(NewCodec(&recordingWriter{}))
		r := nativeRegistered(t, p, time.UnixMilli(3000))
		r.owner.observeRequested(requestedDescriptor("m", "", "/api/generate"))
		if err := p.registerWorkload(r, workloadKey{engine: "ollama", id: "card"}); err != nil {
			t.Fatal(err)
		}
		h := p.newRoutingHistory()
		for i := 0; i < 35; i++ {
			a := nativeAttempt(t, r.owner, "ollama")
			if i == 0 {
				historyTestScan(t, h)
				linked := historyTestRecord(t, h, 1)
				if linked.Workload == nil || !linked.Workload.Attached || linked.Workload.AttemptIndex != 0 || linked.Workload.OriginNodeID != r.key.OriginNodeID || linked.Workload.ProxyRunID != r.key.ProxyRunID {
					t.Fatal("actual first matching workload association was not projected")
				}
			}
			a.observeAttempt(routingAttemptResult{factRedispatched, factTransport, 503, true})
			a.finished(publicAttemptTerminal{})
		}
		r.owner.observeIngress(routingIngressResult{factFailed, factAttemptsExhausted, 502})
		historyTestScan(t, h)
		got := historyTestRecord(t, h, 1)
		if got.DroppedAttempts != 3 || len(got.Attempts) != 32 || got.Workload == nil || got.Workload.Attached || got.Workload.AttemptIndex != 0 || got.Workload.Engine != "ollama" || got.Workload.ID != "card" {
			t.Fatalf("trim/binding: %+v", got)
		}
		for i, a := range got.Attempts {
			if a.Index != int64(i+3) || a.ContinuationOf != nil || a.Outcome != "redispatched" || a.Reason != "transport" || a.EndedAt == nil {
				t.Fatalf("attempt %d lost original ordinal/facts: %+v", i, a)
			}
		}
	})
	t.Run("epoch and native revision precede sequence", func(t *testing.T) {
		p := newBoundProxy(NewCodec(&recordingWriter{}))
		r := nativeRegistered(t, p, time.UnixMilli(4000))
		h := p.newRoutingHistory()
		d := p.routingFactSource()
		d.epoch.Add(1)
		if h.reconcile(false) || h.current.count != 0 || h.generationLength != 0 {
			t.Fatal("odd source epoch assigned sequence/generation")
		}
		d.epoch.Add(1)
		r.owner.mu.Lock()
		f := r.owner.facts
		f.epoch.Add(1)
		r.owner.mu.Unlock()
		if h.reconcile(false) || h.current.count != 0 {
			t.Fatal("odd native revision assigned sequence")
		}
		f.epoch.Add(1)
		historyTestScan(t, h)
		seq, revision := h.current.cells[0].sequence, h.current.cells[0].facts.revision
		historyTestScan(t, h)
		if seq != 1 || h.current.cells[0].sequence != seq || h.current.cells[0].facts.revision != revision {
			t.Fatal("equal native revision advanced sequence")
		}
		r.owner.observeRequested(requestedDescriptor("accepted", "medium", "/api/embed"))
		historyTestScan(t, h)
		if got := historyTestRecord(t, h, 1); got.Seq != 2 || got.Requested.Mode != "embedding" || got.Requested.Model != "accepted" {
			t.Fatalf("new native revision not projected once: %+v", got)
		}
	})
}

func TestRoutingHistoryCapacity(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	if p.routingHistory != nil {
		t.Fatal("production construction allocated history")
	}
	h := p.newRoutingHistory()
	if p.newRoutingHistory() != h || h.current.cells == h.frozen.cells || h.active || len(h.writeWake) != 0 {
		t.Fatal("demand construction duplicated banks or activated output")
	}
	if routingHistoryBytes != 34152448 || unsafe.Sizeof(*h.current.cells) != 16777216 || unsafe.Sizeof(*h.frozen.cells) != 16777216 || unsafe.Sizeof(h.scratch) != 65536 || len(h.frames[0]) != 135168 || len(h.frames[1]) != 135168 || unsafe.Sizeof(h.guard) != 131072 || routingHistoryControlUsed > 131072 {
		t.Fatal("C4 exact bank/control dimensions changed")
	}
	if len(h.current.order) != 256 || len(h.frozen.order) != 256 || cap(h.wake) != 1 || cap(h.writeWake) != 1 || routingHistoryReconcile != 5*time.Second {
		t.Fatal("unfunded indexes, queued cycles or timer dimension")
	}
	var checkPacked func(reflect.Type)
	checkPacked = func(typ reflect.Type) {
		switch typ.Kind() {
		case reflect.Array:
			checkPacked(typ.Elem())
		case reflect.Struct:
			for i := 0; i < typ.NumField(); i++ {
				checkPacked(typ.Field(i).Type)
			}
		case reflect.Uint64, reflect.Uint16, reflect.Uint8, reflect.Bool:
		default:
			t.Fatalf("packed bank gained a referent or dynamic representation: %v", typ)
		}
	}
	checkPacked(reflect.TypeOf(routingProjectionCell{}))
	// Include fixed enum tables and their static text, not just their headers.
	static := uintptr(0)
	for _, values := range [][]string{routingHistoryStates[:], routingHistoryOutcomes[:], routingHistoryTiers[:], routingHistoryModes[:], routingHistoryAccounting[:], routingHistoryAccountStates[:], routingFactReasons[:]} {
		static += uintptr(len(values)) * unsafe.Sizeof("")
		for _, s := range values {
			static += uintptr(len(s))
		}
	}
	if routingHistoryControlUsed+static > 131072 {
		t.Fatal("fixed control graph and enum referents exceed C")
	}
	t.Logf("2W+S+2P+R+C=%d; simultaneous named controls=%d, enum backing=%d, total C=%d", routingHistoryBytes, routingHistoryControlUsed, static, routingHistoryControlUsed+static)
}

func TestRoutingHistoryCapacityControlCensus(t *testing.T) {
	h := newBoundProxy(NewCodec(&recordingWriter{})).newRoutingHistory()
	owner := unsafe.Sizeof(*h) - unsafe.Sizeof(h.scratch) - unsafe.Sizeof(h.frames) - unsafe.Sizeof(h.guard) +
		unsafe.Sizeof(h.proxy.routingHistoryMu) + unsafe.Sizeof(h.proxy.routingHistory)
	if owner != routingHistoryOwnerControlBytes {
		t.Fatal("owner/index/header census does not match actual fields")
	}
	// Each dynamic owner referent must name its funded backing or its borrowed
	// baseline owner. sizeof alone would miss an added string/map/slice/closure.
	allowed := map[string]bool{"proxy": true, "current.cells": true, "frozen.cells": true, "wake": true, "writeWake": true}
	var check func(reflect.Type, string)
	check = func(typ reflect.Type, path string) {
		switch typ.Kind() {
		case reflect.Struct:
			for i := 0; i < typ.NumField(); i++ {
				f := typ.Field(i)
				name := f.Name
				if path != "" {
					name = path + "." + name
				}
				check(f.Type, name)
			}
		case reflect.Array:
			check(typ.Elem(), path)
		case reflect.Pointer, reflect.Chan:
			if !allowed[path] {
				t.Fatalf("uncensused owner referent: %s (%s)", path, typ)
			}
			delete(allowed, path)
		case reflect.String, reflect.Slice, reflect.Map, reflect.Func, reflect.Interface, reflect.UnsafePointer:
			t.Fatalf("uncensused dynamic owner field: %s (%s)", path, typ)
		}
	}
	check(reflect.TypeOf(h).Elem(), "")
	if len(allowed) != 0 {
		t.Fatal("census includes a referent the actual owner does not have")
	}
	actualOverlap := 32*unsafe.Sizeof(routingDestinationDescriptor{}) + unsafe.Sizeof(routingRequestedDescriptor{}) + unsafe.Sizeof(routingWorkloadDescriptor{})
	if actualOverlap > routingHistoryDescriptorOverlap || unsafe.Sizeof(routingSourceScan{}.sources) != 256*8 {
		t.Fatal("native descriptor overlap or sole scan-pointer bank exceeded its row")
	}
	// Actual standard context layout, not sizeof(interface). Registration and
	// removal are synchronous here; the empty callback cannot run before cancel.
	root, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	stop := context.AfterFunc(root, func() {})
	t.Cleanup(func() { stop() })
	parent := reflect.ValueOf(root).Elem()
	children := parent.FieldByName("children")
	if !children.IsValid() || children.Len() != 1 {
		t.Fatal("standard parent association shape changed")
	}
	child := children.MapKeys()[0]
	for child.Kind() == reflect.Interface || child.Kind() == reflect.Pointer {
		child = child.Elem()
	}
	if child.Type().Size() > routingHistoryAfterFuncBytes {
		t.Fatalf("actual AfterFunc object exceeds row: %s/%d", child.Type(), child.Type().Size())
	}
	if !stop() || children.Len() != 0 || children.IsNil() {
		t.Fatal("callback must detach while retained parent-map backing stays charged")
	}
	if unsafe.Sizeof(time.Timer{}) > routingHistoryTimerObjectBytes {
		t.Fatal("actual public timer prefix exceeds the conservative timer estimate")
	}
	record := slog.NewRecord(time.Time{}, slog.LevelWarn, "routing history projection limited", 0)
	record.AddAttrs(slog.String("category", "limit-exceeded"), slog.Int("count", 1))
	if record.NumAttrs() != 2 || reflect.ValueOf(record).FieldByName("back").Len() != 0 {
		t.Fatal("fixed diagnostic exceeded Record's inline attributes")
	}
	rows := []struct {
		name  string
		bytes uintptr
	}{
		{"owner, indexes, process copy, flags and constructor", routingHistoryOwnerControlBytes},
		{"native directory including scan, process and clock", routingHistoryNativeControlBytes},
		{"one replaced-descriptor overlap", routingHistoryDescriptorOverlap},
		{"wake, writer wake, joins and conservative Done channels", routingHistoryRoleChannelBytes},
		{"one timer, physical channel and old/new heap entries", routingHistoryTimerControlBytes},
		{"one AfterFunc, stop capture and retained association", routingHistoryCallbackBytes},
		{"copier select, idle writer select and callback join waiters", routingHistoryWaitControlBytes},
		{"worker launch and copier defers", routingHistoryWorkerCaptureBytes},
		{"active write captures, state and return box", routingHistoryWriteCaptureBytes},
		{"canonical writer stack cursor and frame-step locals", routingHistoryCycleStepBytes},
		{"uint128 digits, transient text and header", routingHistoryGenerationBytes},
		{"one encoder, number conversion and frame aliases", routingHistoryEncoderBytes},
		{"shared fixed grammar stack and captures", routingHistoryGrammarBytes},
		{"ack, generation referent and fixed error", routingHistoryAckControlBytes},
		{"fixed diagnostic input and handoff", routingHistoryDiagnosticBytes},
	}
	var total uintptr
	for _, row := range rows {
		if row.bytes == 0 {
			t.Fatalf("empty census row: %s", row.name)
		}
		total += row.bytes
		t.Logf("%s: %d", row.name, row.bytes)
	}
	if total != routingHistoryControlUsed || total > routingHistoryControlBytes {
		t.Fatalf("simultaneous named census=%d declared=%d C=%d", total, routingHistoryControlUsed, routingHistoryControlBytes)
	}
}

func TestRoutingHistoryProjectionNativeExhaustion(t *testing.T) {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	p.originNodeID, p.runID = "private-origin-sentinel", "private-run-sentinel"
	limited := nativeRegistered(t, p, time.UnixMilli(1000))
	other := nativeRegistered(t, p, time.UnixMilli(1001))
	limited.owner.observeRequested(requestedDescriptor("stable", "", "/api/chat"))
	h := p.newRoutingHistory()
	historyTestScan(t, h)
	historyTestCycle(t, h)
	if logs.Len() != 0 || !h.accepted(nodepolicy.RoutingHistoryAccepted{Schema: 1, Generation: "1"}) {
		t.Fatal("initial valid facts emitted an exhaustion diagnostic")
	}
	// Same checked boundary as TestRoutingNativeRevisionEpochRetryAndLimit:
	// the next attempted native update publishes only frozen+limit-exceeded.
	limited.owner.mu.Lock()
	facts := limited.owner.facts
	facts.epoch.Store(2 * (factMaxRevision - 1))
	limited.owner.reason = "private-error-prompt-sentinel"
	limited.owner.mu.Unlock()
	limited.owner.observeRequested(requestedDescriptor("private-model-prompt-sentinel", "", "/api/chat"))
	historyTestScan(t, h)
	got := historyTestRecord(t, h, 1)
	if got.Seq != 2 || got.Requested.Model != "stable" || h.current.cells[h.current.find(1)].facts.revision != factMaxRevision || h.failed {
		t.Fatal("native limit did not freeze only its affected record")
	}
	var diagnostic struct {
		Time     string `json:"time"`
		Level    string `json:"level"`
		Message  string `json:"msg"`
		Category string `json:"category"`
		Count    int    `json:"count"`
	}
	if err := json.Unmarshal(logs.Bytes(), &diagnostic); err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(logs.Bytes(), &fields); err != nil {
		t.Fatal(err)
	}
	if len(fields) != 5 || diagnostic.Time == "" || diagnostic.Level != "WARN" || diagnostic.Message != "routing history projection limited" || diagnostic.Category != "limit-exceeded" || diagnostic.Count != 1 {
		t.Fatalf("diagnostic is not the fixed category/count shape: %+v", diagnostic)
	}
	for _, field := range []string{"time", "level", "msg", "category", "count"} {
		if _, ok := fields[field]; !ok {
			t.Fatalf("missing fixed diagnostic field %s", field)
		}
	}
	if bytes.Contains(logs.Bytes(), []byte("private-")) {
		t.Fatal("native exhaustion diagnostic exposed identity, model or raw error/prompt text")
	}
	oneDiagnostic := bytes.Clone(logs.Bytes())
	historyTestCycle(t, h)
	if !h.reconcile(true) {
		t.Fatal("unacknowledged exhausted record could not replay")
	}
	historyTestCycle(t, h)
	if !h.accepted(nodepolicy.RoutingHistoryAccepted{Schema: 1, Generation: "2"}) {
		t.Fatal("exhausted record's coherent generation could not be accepted")
	}
	other.owner.observeRequested(requestedDescriptor("other-progress", "medium", "/api/embed"))
	historyTestScan(t, h)
	historyTestCycle(t, h)
	if !bytes.Equal(logs.Bytes(), oneDiagnostic) || h.failed || historyTestRecord(t, h, 1).Seq != 2 || historyTestRecord(t, h, 2).Seq != 2 || h.generation.Low != 3 {
		t.Fatal("exhaustion re-reported, stopped other records, or advanced the frozen record")
	}
	if _, err := limited.owner.newDispatchAttempt(context.Background(), func() {}, "node"); err != nil {
		t.Fatalf("diagnostic exhaustion changed ordinary attempt admission: %v", err)
	}
}

func TestRoutingHistoryCapacityMaximumStrings(t *testing.T) {
	test := func(name, model string, escapedBytes int) {
		t.Run(name, func(t *testing.T) {
			p := newBoundProxy(NewCodec(&recordingWriter{}))
			h := p.newRoutingHistory()
			origin, run, first, second := strings.Repeat("o", 128), strings.Repeat("r", 128), strings.Repeat("d", 128), strings.Repeat("a", 128)
			h.process.originLen, h.process.runLen = uint16(copy(h.process.origin[:], origin)), uint16(copy(h.process.run[:], run))
			f := &h.current.cells[0].facts
			f.id, f.created, f.ended, f.total, f.served = math.MaxInt64, 9007199254740000, 9007199254740002, 9007199254740000, 9007199254740000
			f.disposition = factDisposition(uint8(factCompleted), factDeploymentQuarantined, 599)
			f.flags = factEndKnown | factMoneyApplicable | factCeilingKnown | factHeldKnown | factChargeKnown | uint64(factAccountingRetained)<<16
			f.ceiling, f.granted, f.held, f.charged = math.MaxInt64, math.MaxInt64, math.MaxInt64, math.MaxInt64
			f.requested.length, f.requested.tier, f.requested.mode = uint16(copy(f.requested.model[:], model)), 3, 2
			h.current.count, h.current.cells[0].sequence = 1, factMaxRevision
			d := destinationDescriptor(model, first, second, true)
			if d == nil || len(model) != 512 {
				t.Fatal("maximum legal descriptor rejected")
			}
			str := func(s string) *string { return &s }
			number := func(n int64) *int64 { return &n }
			money := nodepolicy.NanoUSD("9223372036854775807")
			destination := nodepolicy.RoutingHistoryDestination{Kind: "api", Model: str(model), DeploymentID: str(first), AccountID: str(second)}
			want := nodepolicy.RoutingHistoryRecord{
				Schema: 1, Request: nodepolicy.RoutingRequestKey{OriginNodeID: origin, ProxyRunID: run, RequestID: "9223372036854775807"}, Seq: 9007199254740991,
				Requested: nodepolicy.RoutingHistoryRequested{Model: model, Tier: "strong", Mode: "completion"}, CreatedAt: 9007199254740000, EndedAt: number(9007199254740002),
				State: "completed", Reason: "deployment-quarantined", Status: number(599), Served: &destination,
				Spend: nodepolicy.RoutingHistorySpend{Currency: "USD", CeilingNanoUSD: &money, GrantedNanoUSD: &money, HeldNanoUSD: &money, ChargedNanoUSD: &money, Accounting: "retained"}, DroppedAttempts: 9007199254739968,
			}
			for i := 0; i < 32; i++ {
				ordinal := f.total - 31 + uint64(i)
				a := &f.attempts[(ordinal-1)%32]
				a.ordinal, a.started, a.ended, a.destination = ordinal, 9007199254740001, 9007199254740002, *d
				a.disposition = factDisposition(uint8(factCommitted), factDeploymentQuarantined, 599)
				a.admission, a.firstByte = factMaxRevision, factMaxRevision
				a.held, a.charged = math.MaxInt64, math.MaxInt64
				a.flags = factEndKnown | factHeldKnown | factChargeKnown | factFirstByteKnown | uint64(factAccountingRetained)<<16 | uint64(factAccountBadCredential)<<24
				want.Attempts = append(want.Attempts, nodepolicy.RoutingHistoryAttempt{Index: int64(ordinal - 1), StartedAt: 9007199254740001, EndedAt: number(9007199254740002), Destination: destination, Outcome: "committed", Reason: "deployment-quarantined", Status: number(599), Waits: nodepolicy.RoutingHistoryWaits{AdmissionMs: 9007199254740991, FirstByteMs: number(9007199254740991)}, Account: &nodepolicy.RoutingHistoryAccount{State: "credential-failure", HeldNanoUSD: &money, ChargedNanoUSD: &money, Accounting: "retained"}})
			}
			got := historyTestRecord(t, h, math.MaxInt64)
			if !reflect.DeepEqual(got, want) {
				t.Fatal("independent maximum money/timing/destination oracle differs")
			}
			wire, err := json.Marshal(want)
			if err != nil {
				t.Fatal(err)
			}
			if len(wire) >= 131072 {
				t.Fatal("maximum native record exceeds R")
			}
			// A separate standard encoder supplies punctuation/field overhead.
			// Remove ONLY model bytes, then add independent UTF-8/escaping widths
			// for requested + served + every retained attempt occurrence.
			want.Requested.Model = ""
			want.Served.Model = str("")
			for i := range want.Attempts {
				want.Attempts[i].Destination.Model = str("")
			}
			base, err := json.Marshal(want)
			if err != nil {
				t.Fatal(err)
			}
			wantBytes := len(base) + (len(want.Attempts)+2)*escapedBytes
			if len(wire) != wantBytes {
				t.Fatalf("UTF-8/escape arithmetic = %d, encoded oracle=%d", wantBytes, len(wire))
			}
			w := routingHistoryJSON{data: h.frames[1][:0]}
			h.encodeRecord(&w, &h.current.cells[0])
			if w.err != nil || !bytes.Equal(w.data, wire) {
				t.Fatalf("bounded encoding differs in bytes/size: err=%v bytes=%d want=%d", w.err, len(w.data), wantBytes)
			}
			if allocations := testing.AllocsPerRun(5, func() {
				out := routingHistoryJSON{data: h.frames[1][:0]}
				h.encodeRecord(&out, &h.current.cells[0])
			}); allocations != 0 {
				t.Fatalf("encoder allocated %v objects instead of using admitted frame", allocations)
			}
			t.Logf("maximum packed record: %d encoded bytes, model bytes=512, escaped content=%d", wantBytes, escapedBytes)
		})
	}
	test("quote and backslash expansion", strings.Repeat(`"\`, 256), 1024)
	test("multibyte UTF-8 plus escapes", strings.Repeat("界", 170)+`"\`, 514)
}

func TestRoutingHistoryCapacityMaximumWindow(t *testing.T) {
	h := newBoundProxy(NewCodec(&recordingWriter{})).newRoutingHistory()
	origin, run := strings.Repeat("o", 128), strings.Repeat("r", 128)
	h.process.originLen, h.process.runLen = uint16(copy(h.process.origin[:], origin)), uint16(copy(h.process.run[:], run))
	const generation = "340282366920938463463374607431768211455"
	h.generationLength = copy(h.generationText[:], generation)
	h.frozen.count = 256
	want := nodepolicy.RoutingHistoryWindow{Schema: 1, Generation: generation, Keys: make([]nodepolicy.RoutingHistoryWindowKey, 0, 256)}
	for i := 0; i < 256; i++ {
		id := uint64(math.MaxInt64 - 255 + i)
		h.frozen.cells[i].facts.id, h.frozen.cells[i].facts.created, h.frozen.cells[i].sequence = id, factMaxRevision, factMaxRevision
		want.Keys = append(want.Keys, nodepolicy.RoutingHistoryWindowKey{Request: nodepolicy.RoutingRequestKey{OriginNodeID: origin, ProxyRunID: run, RequestID: strconv.FormatUint(id, 10)}, CreatedAt: 9007199254740991, Seq: 9007199254740991})
	}
	h.frozen.sort()
	want.Cutoff = &nodepolicy.RoutingHistoryCutoff{Request: want.Keys[0].Request, CreatedAt: 9007199254740991}
	w := routingHistoryJSON{data: h.frames[1][:0]}
	h.encodeWindow(&w)
	encoded, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	// Independent literal-width arithmetic: all identities are unescaped ASCII,
	// every counter is 19 digits, safe integer fields are 16 digits.
	keyBytes := len(`{"request":{"originNodeId":"","proxyRunId":"","requestId":""},"createdAt":,"seq":}`) + 128 + 128 + 19 + 16 + 16
	cutoffBytes := len(`{"request":{"originNodeId":"","proxyRunId":"","requestId":""},"createdAt":}`) + 128 + 128 + 19 + 16
	wantBytes := len(`{"schema":1,"generation":"","cutoff":,"keys":[]}`) + 39 + cutoffBytes + 256*keyBytes + 255
	if w.err != nil || len(w.data) != wantBytes || !bytes.Equal(w.data, encoded) || wantBytes > 124928 || wantBytes >= 131072 {
		t.Fatalf("maximum window: bytes=%d oracle=%d arithmetic=%d err=%v", len(w.data), len(encoded), wantBytes, w.err)
	}
	var decoded nodepolicy.RoutingHistoryWindow
	if err := json.Unmarshal(w.data, &decoded); err != nil {
		t.Fatal(err)
	}
	t.Logf("maximum manifest=%d bytes; frame headroom=4096; P=135168", wantBytes)
}

func TestRoutingHistoryReconciliation(t *testing.T) {
	w := &recordingWriter{}
	p := newBoundProxy(NewCodec(w))
	r := nativeRegistered(t, p, time.UnixMilli(5000))
	h := p.newRoutingHistory()
	historyTestScan(t, h)
	historyTestCycle(t, h)
	first := bytes.Join(w.lines(), []byte{'\n'})
	if h.generation.Low != 1 || h.active || h.acknowledged {
		t.Fatal("initial cycle did not await acknowledgement")
	}
	// Loss of the ack replays byte-identical begin, record and end.
	if !h.reconcile(true) {
		t.Fatal("retry scan failed")
	}
	historyTestCycle(t, h)
	lines := w.lines()
	if len(lines) != 6 || !bytes.Equal(first, bytes.Join(lines[3:], []byte{'\n'})) || h.generation.Low != 1 {
		t.Fatal("ack loss changed generation or replay contents")
	}
	if h.accepted(nodepolicy.RoutingHistoryAccepted{Schema: 1, Generation: "2"}) || h.accepted(nodepolicy.RoutingHistoryAccepted{Schema: 1, Generation: "01"}) || !h.accepted(nodepolicy.RoutingHistoryAccepted{Schema: 1, Generation: "1"}) {
		t.Fatal("ack generation validation mismatch")
	}
	h.reconcile(true)
	if h.active || len(h.writeWake) != 0 || len(w.lines()) != 6 {
		t.Fatal("clean ack repeated full-window IPC")
	}
	r.owner.observeRequested(requestedDescriptor("new", "weak", "/api/chat"))
	historyTestScan(t, h)
	if h.generation.Low != 2 || h.acknowledged {
		t.Fatal("changed content did not mint a fresh generation")
	}
	historyTestCycle(t, h)
	if got := historyTestRecord(t, h, 1); got.Seq != 2 {
		t.Fatal("changed native revision did not advance per-key seq once")
	}
	// Even when a source is currently odd, the completed generation remains
	// replayable. The timer itself never mints another generation.
	r.owner.mu.Lock()
	f := r.owner.facts
	f.epoch.Add(1)
	r.owner.mu.Unlock()
	if h.reconcile(true) || !h.active || h.generation.Low != 2 {
		t.Fatal("source pressure suppressed coherent replay or minted a generation")
	}
	historyTestCycle(t, h)
	f.epoch.Add(1)
	// Verify the real shared uint128 counter is used beyond uint64.
	h.generation = nodepolicy.RoutingHistoryGeneration{Low: math.MaxUint64}
	r.owner.observeRequested(requestedDescriptor("beyond", "", "/api/chat"))
	historyTestScan(t, h)
	if string(h.generationText[:h.generationLength]) != "18446744073709551616" {
		t.Fatal("generation is not canonical uint128 decimal")
	}
}
