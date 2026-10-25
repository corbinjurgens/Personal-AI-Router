// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-tui/rpc"
)

func routingWireCorpus(t *testing.T) []nodepolicy.RoutingHistoryRecord {
	t.Helper()
	raw, err := os.ReadFile("../../shared/nodepolicy/testdata/routing-request.json")
	if err != nil {
		t.Fatal(err)
	}
	var corpus struct {
		Cases []struct {
			Name   string
			Record nodepolicy.RoutingHistoryRecord
		}
	}
	if err := json.Unmarshal(raw, &corpus); err != nil {
		t.Fatal(err)
	}
	if len(corpus.Cases) == 0 {
		t.Fatal("empty compatibility corpus")
	}
	rows := make([]nodepolicy.RoutingHistoryRecord, len(corpus.Cases))
	for i, item := range corpus.Cases {
		rows[i] = item.Record
	}
	return rows
}

func routingWireApply(t *testing.T, w *routingHistoryWire, initial bool, records ...nodepolicy.RoutingHistoryRecord) {
	t.Helper()
	var value any = records[0]
	if initial {
		value = nodepolicy.RoutingHistoryInitial{Requests: records}
	}
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.consume(1, initial, raw); err != nil {
		t.Fatal(err)
	}
}

func TestRoutingHistoryWireCorpusPackedRoundTrip(t *testing.T) {
	w := newRoutingHistoryWire()
	w.reset(1)
	for _, row := range routingWireCorpus(t) {
		var cell routingHistoryCell
		if err := cell.store(row); err != nil {
			t.Fatal(err)
		}
		got, err := cell.record()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, row) {
			t.Fatalf("packed record changed schema/null/money semantics: %+v", got.Request)
		}
	}
	rows := routingWireCorpus(t)
	routingWireApply(t, w, true, rows...)
	if w.current.count != len(rows) {
		t.Fatalf("rows = %d", w.current.count)
	}
}

func TestRoutingHistoryWireMergeAndTerminal(t *testing.T) {
	w := newRoutingHistoryWire()
	w.reset(1)
	terminal := routingWireCorpus(t)[0]
	routingWireApply(t, w, false, terminal)
	older := terminal
	older.Seq--
	routingWireApply(t, w, true, older)
	active := terminal
	active.Seq++
	active.State = "routing"
	active.EndedAt = nil
	active.Served = nil
	// Merge tests use typed, validated inputs independently of the raw guard.
	if err := w.merge(w.current, active); err != nil {
		t.Fatal(err)
	}
	got, err := w.current.cells[0].record()
	if err != nil {
		t.Fatal(err)
	}
	if got.Seq != terminal.Seq || got.State != "completed" {
		t.Fatalf("stale baseline or active update reopened terminal: %+v", got)
	}
	changed := terminal
	changed.CreatedAt++
	if err := w.merge(w.current, changed); err == nil {
		t.Fatal("mutable creation time accepted")
	}
}

func TestRoutingHistoryWireNumericRankAndEvictionFloor(t *testing.T) {
	w := newRoutingHistoryWire()
	w.reset(1)
	row := routingWireCorpus(t)[0]
	row.Seq = 1
	for i := 1; i <= 300; i++ {
		row.Request.RequestID = strconv.Itoa(i)
		if err := w.merge(w.current, row); err != nil {
			t.Fatal(err)
		}
	}
	if w.current.count != nodepolicy.RoutingHistoryRecords {
		t.Fatalf("window = %d", w.current.count)
	}
	first, err := w.current.cells[0].record()
	if err != nil {
		t.Fatal(err)
	}
	last, err := w.current.cells[w.current.count-1].record()
	if err != nil {
		t.Fatal(err)
	}
	if first.Request.RequestID != "300" || last.Request.RequestID != "45" {
		t.Fatalf("numeric rank = %s..%s", first.Request.RequestID, last.Request.RequestID)
	}
	row.Request.RequestID = "9"
	row.Seq = 99
	if err := w.merge(w.current, row); err != nil {
		t.Fatal(err)
	}
	last, err = w.current.cells[w.current.count-1].record()
	if err != nil {
		t.Fatal(err)
	}
	if last.Request.RequestID != "45" {
		t.Fatal("evicted row resurrected")
	}
}

func TestRoutingHistoryWireGenerationAndCapacity(t *testing.T) {
	w := newRoutingHistoryWire()
	w.reset(1)
	row := routingWireCorpus(t)[0]
	routingWireApply(t, w, false, row)
	w.reset(2)
	raw, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.consume(1, false, raw); err == nil {
		t.Fatal("old callback accepted")
	}
	w.reset(1)
	if w.generation != 2 || w.current.count != 0 {
		t.Fatal("old reset changed replacement")
	}
	if len(w.current.cells) != 256 || len(w.current.cells[0].data) != 65536 {
		t.Fatal("packed bank layout changed")
	}
	if routingHistoryDataBytes != 117710848 {
		t.Fatal("TUI data grant arithmetic changed")
	}
}

func TestRoutingHistoryWireExactMoneyAndWait(t *testing.T) {
	if routingHistoryMoney(nil) != "—" {
		t.Fatal("null became zero")
	}
	for _, tc := range []struct{ raw, want string }{{"0", "$0"}, {"1", "$0.000000001"}, {"123456789", "$0.123456789"}, {"1000000000", "$1"}, {"9223372036854775807", "$9223372036.854775807"}} {
		value, err := nodepolicy.ParseNanoUSD(tc.raw)
		if err != nil {
			t.Fatal(err)
		}
		if got := routingHistoryMoney(&value); got != tc.want {
			t.Fatalf("%s = %s", tc.raw, got)
		}
	}
	ms := int64(1234)
	if routingHistoryWait(nil) != "—" || routingHistoryWait(&ms) != "1.2 s" {
		t.Fatal("wait/null formatting changed")
	}
}

func TestRoutingHistoryWireCorpusGuardParity(t *testing.T) {
	w := newRoutingHistoryWire()
	w.reset(1)
	rawBank := make([]byte, 0, jsonrpc.HistoryFrameBytes)
	offsets := make([]uint32, 0, jsonrpc.HistoryInitialBytes/4)
	var correlation jsonrpc.HistoryCorrelation
	if err := correlation.EnableGeneratedIDs(); err != nil {
		t.Fatal(err)
	}
	for _, row := range routingWireCorpus(t) {
		id, err := correlation.BeginGeneratedMethod(nodepolicy.MethodRoutingGetInitial)
		if err != nil {
			t.Fatal(err)
		}
		value, err := json.Marshal(nodepolicy.RoutingHistoryInitial{Requests: []nodepolicy.RoutingHistoryRecord{row}})
		if err != nil {
			t.Fatal(err)
		}
		frame := append([]byte(`{"jsonrpc":"2.0","id":`), id...)
		frame = append(frame, `,"result":`...)
		frame = append(frame, value...)
		frame = append(frame, '}', '\n')
		codec, err := jsonrpc.NewHistoryCodec(bytes.NewBuffer(frame), 8*1024*1024, rawBank, nil, offsets, &correlation, nodepolicy.RoutingHistoryTransportSchema, nil)
		if err != nil {
			t.Fatal(err)
		}
		msg, err := codec.Read()
		if err != nil {
			t.Fatal(err)
		}
		if err := w.consume(1, true, msg.Result); err != nil {
			t.Fatal(err)
		}
		correlation.Complete()
	}
	if w.current.count != len(routingWireCorpus(t)) {
		t.Fatal("guarded corpus lost records")
	}
}

func TestRoutingHistoryWireLongIntegralSpelling(t *testing.T) {
	// Integral JSON spellings remain exact through the shared transport guard.
	row := routingWireCorpus(t)[0]
	row.Seq = 1
	value, err := json.Marshal(nodepolicy.RoutingHistoryInitial{Requests: []nodepolicy.RoutingHistoryRecord{row}})
	if err != nil {
		t.Fatal(err)
	}
	token := "1." + strings.Repeat("0", 66003) + "E+00"
	value = bytes.Replace(value, []byte(`"seq":1`), []byte(`"seq":`+token), 1)
	var correlation jsonrpc.HistoryCorrelation
	if err := correlation.EnableGeneratedIDs(); err != nil {
		t.Fatal(err)
	}
	id, err := correlation.BeginGeneratedMethod(nodepolicy.MethodRoutingGetInitial)
	if err != nil {
		t.Fatal(err)
	}
	frame := append([]byte(`{"jsonrpc":"2.0","id":`), id...)
	frame = append(frame, `,"result":`...)
	frame = append(frame, value...)
	frame = append(frame, '}', '\n')
	codec, err := jsonrpc.NewHistoryCodec(bytes.NewBuffer(frame), 8*1024*1024, make([]byte, 0, jsonrpc.HistoryFrameBytes), nil, make([]uint32, 0, jsonrpc.HistoryInitialBytes/4), &correlation, nodepolicy.RoutingHistoryTransportSchema, nil)
	if err != nil {
		t.Fatal(err)
	}
	msg, err := codec.Read()
	if err != nil {
		t.Fatal(err)
	}
	w := newRoutingHistoryWire()
	w.reset(1)
	if err := w.consume(1, true, msg.Result); err != nil {
		t.Fatal(err)
	}
	if err := w.withRecord(1, 0, func(got nodepolicy.RoutingHistoryRecord) {
		if got.Seq != 1 {
			t.Fatal("integral spelling changed numeric value")
		}
	}); err != nil {
		t.Fatal(err)
	}
}

func TestRoutingHistoryWireMaximumPackedMetadata(t *testing.T) {
	row := routingWireCorpus(t)[1]
	row.Request.OriginNodeID = strings.Repeat("o", 128)
	row.Request.ProxyRunID = strings.Repeat("r", 128)
	row.Requested.Model = strings.Repeat("\\", 512)
	row.Attempts = make([]nodepolicy.RoutingHistoryAttempt, nodepolicy.RoutingHistoryAttempts)
	model, account, deployment := strings.Repeat(`"`, 512), strings.Repeat("a", 128), strings.Repeat("d", 128)
	maximum, err := nodepolicy.ParseNanoUSD("9223372036854775807")
	if err != nil {
		t.Fatal(err)
	}
	for i := range row.Attempts {
		a := routingWireCorpus(t)[1].Attempts[0]
		a.Index = int64(i)
		a.Destination.Model = &model
		a.Destination.AccountID = &account
		a.Destination.DeploymentID = &deployment
		a.Account.HeldNanoUSD = &maximum
		a.Account.ChargedNanoUSD = &maximum
		row.Attempts[i] = a
	}
	if err := row.Validate(); err != nil {
		t.Fatal(err)
	}
	var cell routingHistoryCell
	if err := cell.store(row); err != nil {
		t.Fatal(err)
	}
	got, err := cell.record()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, row) {
		t.Fatal("maximum known metadata changed")
	}
	if reflect.TypeOf(cell).Size() != 65536 {
		t.Fatal("cell has unaccounted backing")
	}
}

func TestRoutingHistoryWireOutputLoanFencesReset(t *testing.T) {
	w := newRoutingHistoryWire()
	w.reset(1)
	routingWireApply(t, w, false, routingWireCorpus(t)[0])
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	go func() {
		done <- w.withRecord(1, 0, func(nodepolicy.RoutingHistoryRecord) { close(entered); <-release })
	}()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("output borrower not entered")
	}
	if w.mu.TryLock() {
		w.mu.Unlock()
		t.Fatal("output loan has no bank fence")
	}
	reset := make(chan struct{})
	go func() { w.reset(2); close(reset) }()
	once.Do(func() { close(release) })
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("output borrower did not complete")
	}
	select {
	case <-reset:
	case <-time.After(3 * time.Second):
		t.Fatal("reset did not follow actual release")
	}
	if err := w.withRecord(1, 0, func(nodepolicy.RoutingHistoryRecord) {}); err == nil {
		t.Fatal("old output callback reused replacement bank")
	}
}

func TestRoutingHistoryWireRecoveryLabels(t *testing.T) {
	rows := routingWireCorpus(t)
	if got := routingHistoryAttemptLabel(rows[3], 0); got != "Streaming paused, waiting to resume" {
		t.Fatal(got)
	}
	if got := routingHistoryAttemptLabel(rows[4], 0); got != "Interrupted, resumed" {
		t.Fatal(got)
	}
	if got := routingHistoryReason(rows[1].Reason); got != "Stream interrupted" {
		t.Fatal(got)
	}
	if routingHistoryReason("synthetic-secret") != "Unknown" {
		t.Fatal("raw reason echoed")
	}
	if got := routingHistoryRequestReason(rows[3]); got != "Streaming paused, waiting to resume" {
		t.Fatal(got)
	}
}

func TestRoutingHistoryWireModelNotSpecified(t *testing.T) {
	row := routingWireCorpus(t)[0]
	row.Requested.Model = ""
	row.Served.Model = nil
	for i := range row.Attempts {
		row.Attempts[i].Destination.Model = nil
	}
	if err := row.Validate(); err != nil {
		t.Fatal(err)
	}
	w := newRoutingHistoryWire()
	w.reset(1)
	routingWireApply(t, w, true, row)
	if err := w.withRecord(1, 0, func(got nodepolicy.RoutingHistoryRecord) {
		if !reflect.DeepEqual(got, row) {
			t.Fatal("packed model-unknown record changed")
		}
		if routingHistoryModel(got.Served.Model) != "model not specified" || routingHistoryModel(got.Attempts[0].Destination.Model) != "model not specified" {
			t.Fatal("model-unknown destination label changed")
		}
	}); err != nil {
		t.Fatal(err)
	}
	model := "synthetic-model"
	if routingHistoryModel(&model) != model {
		t.Fatal("known model label changed")
	}
}

func TestRoutingHistoryWireMaximumInitialThroughClient(t *testing.T) {
	rows := make([]nodepolicy.RoutingHistoryRecord, nodepolicy.RoutingHistoryRecords)
	base := routingWireCorpus(t)[0]
	for i := range rows {
		rows[i] = base
		rows[i].Request.RequestID = strconv.Itoa(i + 1)
	}
	value, err := json.Marshal(nodepolicy.RoutingHistoryInitial{Requests: rows})
	if err != nil {
		t.Fatal(err)
	}
	// Unknown fields fill the legal M and E grants, but must never enter the
	// packed banks or the ordinary event channel. Fixture storage is test-only.
	value = append(value[:len(value)-1], `,"unknown":"`...)
	value = append(value, strings.Repeat("x", jsonrpc.HistoryInitialBytes-len(value)-2)...)
	value = append(value, '"', '}')
	w := newRoutingHistoryWire()
	a, b := net.Pipe()
	t.Cleanup(func() { _ = a.Close(); _ = b.Close() })
	for _, conn := range []net.Conn{a, b} {
		if err := conn.SetDeadline(time.Now().Add(time.Minute)); err != nil {
			t.Fatal(err)
		}
	}
	client, err := rpc.NewHistoryClient(a, a, w.consume, w.reset)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	done := make(chan error, 1)
	go func() { done <- client.Run(ctx) }()
	client.SetHistorySubscription(true)
	server := rpc.NewCodec(b, b)
	request, err := server.Read()
	if err != nil {
		t.Fatal(err)
	}
	if request.ID == nil || request.Method != nodepolicy.MethodRoutingSubscribe {
		t.Fatalf("initial request = %+v", request)
	}
	frame := append([]byte(`{"jsonrpc":"2.0","id":`), *request.ID...)
	frame = append(frame, `,"result":`...)
	frame = append(frame, value...)
	frame = append(frame, `,"unknown":"`...)
	frame = append(frame, strings.Repeat("y", jsonrpc.HistoryFrameBytes-len(frame)-3)...)
	frame = append(frame, '"', '}', '\n')
	if len(frame) != jsonrpc.HistoryFrameBytes || len(value) != jsonrpc.HistoryInitialBytes {
		t.Fatal("fixture does not reach the exact physical and initial limits")
	}
	if _, err := b.Write(frame); err != nil {
		t.Fatal(err)
	}
	if err := server.Write(&rpc.Message{JSONRPC: "2.0", Method: "ordinary-barrier"}); err != nil {
		t.Fatal(err)
	}
	select {
	case event := <-client.Events():
		if event.Message == nil || event.Message.Method != "ordinary-barrier" {
			t.Fatalf("history leaked a payload event: %+v", event)
		}
	case <-time.After(time.Minute):
		t.Fatal("maximum initial blocked ordinary notifications")
	}
	if state := client.HistoryState(); state.Loading || state.Pending || state.Quarantined {
		t.Fatalf("maximum initial was not committed: %+v", state)
	}
	if w.current.count != len(rows) {
		t.Fatalf("maximum initial retained %d rows", w.current.count)
	}
	for index, want := range []string{strconv.Itoa(len(rows)), "1"} {
		position := index * (len(rows) - 1)
		if err := w.withRecord(1, position, func(got nodepolicy.RoutingHistoryRecord) {
			if got.Request.RequestID != want {
				t.Fatalf("rank at %d = %s, want %s", position, got.Request.RequestID, want)
			}
		}); err != nil {
			t.Fatal(err)
		}
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("client did not detach")
	}
}
