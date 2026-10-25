// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

func TestHistoryProductionEntryServesHistoryAndOrdinaryRPC(t *testing.T) {
	// Reuse the fixed test image's command/broker roles. This exercises run ->
	// serve -> Run, including the real private endpoint and exec-based spawner.
	box := newNativeSandbox(t, unusedEndpoint(t))
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), nativeRunCap)
	defer cancel()
	cmd := box.command(t, ctx, "--endpoint", box.endpoint, "--log-dir", filepath.Join(box.dir, "logs"), "--broker-path", exe, "--", processBrokerArg)
	for i, entry := range cmd.Env {
		key, _, _ := strings.Cut(entry, "=")
		if strings.EqualFold(key, fakeBrokerEnv) {
			cmd.Env[i] = fakeBrokerEnv + "=1"
		}
	}
	var output bytes.Buffer
	cmd.Stdout, cmd.Stderr = &output, &output
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	joined := false
	t.Cleanup(func() {
		cancel()
		if !joined {
			<-done
		}
	})
	waitUntil(t, "production service endpoint", func() bool { return alreadyAnswering(ctx, box.endpoint) })
	h := &harness{t: t, endpoint: box.endpoint}
	client := h.dial()
	client.notification("app:ready")
	// The ordinary fake broker deliberately has no history responder. The
	// first initial stays pending. An ordinary response is its dispatch barrier;
	// another initial must receive the service's local history-only busy reply.
	client.request(`"pending"`, nodepolicy.MethodRoutingGetInitial, `{}`)
	reply := client.call(`"echo\u0031"`, "echo", `{"ordinary":true}`)
	var echo struct {
		Ordinary bool `json:"ordinary"`
	}
	if reply.Error != nil {
		t.Fatalf("ordinary echo: %+v", reply.Error)
	}
	if err := json.Unmarshal(reply.Result, &echo); err != nil || !echo.Ordinary {
		t.Fatalf("ordinary echo changed: %+v / %v", echo, err)
	}
	reply = client.call(`"history\u0031"`, nodepolicy.MethodRoutingGetInitial, `{}`)
	if reply.Error == nil || reply.Error.Code != jsonrpc.HistoryBusyCode {
		t.Fatalf("history loading reply = %+v", reply)
	}
	if reply := client.call("3", "service/stop", `{}`); reply.Error != nil {
		t.Fatalf("stop: %+v", reply.Error)
	}
	err = <-done
	joined = true
	if err != nil || ctx.Err() != nil {
		t.Fatalf("production service did not join cleanly: %v / %v; %s", err, ctx.Err(), output.String())
	}
}

type historyTestInput struct{ requests chan []byte }

func (w *historyTestInput) Write(raw []byte) (int, error) {
	w.requests <- append([]byte(nil), raw...)
	return len(raw), nil
}

func (w *historyTestInput) Close() error { return nil }

type historyTestHarness struct {
	s      *Service
	run    *brokerRun
	output *io.PipeWriter
	input  *historyTestInput
	kills  atomic.Int32
	peers  []*historyTestPeer
}

type historyTestPeer struct {
	conn   net.Conn
	reader *bufio.Reader
	raw    []byte
}

func newHistoryTestHarness(t *testing.T) *historyTestHarness {
	t.Helper()
	h := &historyTestHarness{s: newServiceWithHistory(config{})}
	reader, writer := io.Pipe()
	h.output = writer
	h.input = &historyTestInput{requests: make(chan []byte, 32)}
	h.run = &brokerRun{proc: &brokerProc{
		stdin: h.input, stdout: reader,
		kill: func() error { h.kills.Add(1); return reader.Close() },
	}, exited: make(chan struct{})}
	h.s.broker = h.run
	h.s.startHistory()
	done := make(chan struct{})
	go func() { defer close(done); h.s.pumpStdout(h.run) }()
	t.Cleanup(func() {
		for _, peer := range h.peers {
			_ = peer.conn.Close()
		}
		_ = writer.Close()
		_ = reader.Close()
		close(h.s.stopCh)
		joined := make(chan struct{})
		go func() { h.s.history.wg.Wait(); <-done; close(joined) }()
		select {
		case <-joined:
		case <-time.After(5 * time.Second):
			t.Error("history borrowers did not join")
		}
	})
	return h
}

func (h *historyTestHarness) peer() *historyTestPeer {
	return h.peerWithChunk(0)
}

type historyTestShortConn struct {
	net.Conn
	chunk int
}

func (c *historyTestShortConn) Write(raw []byte) (int, error) {
	return c.Conn.Write(raw[:min(len(raw), c.chunk)])
}

func (h *historyTestHarness) peerWithChunk(chunk int) *historyTestPeer {
	server, peer := net.Pipe()
	if chunk > 0 {
		h.s.attach(&historyTestShortConn{Conn: server, chunk: chunk})
	} else {
		h.s.attach(server)
	}
	p := &historyTestPeer{conn: peer, reader: bufio.NewReaderSize(peer, 4096), raw: make([]byte, 0, 4096)}
	h.peers = append(h.peers, p)
	return p
}

func (p *historyTestPeer) send(t *testing.T, raw string) {
	t.Helper()
	if err := p.conn.SetWriteDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(p.conn, raw+"\n"); err != nil {
		t.Fatal(err)
	}
}

func (p *historyTestPeer) next(t *testing.T) jsonrpc.Message {
	t.Helper()
	return p.nextWithTimeout(t, 10*time.Second)
}

func (p *historyTestPeer) nextWithTimeout(t *testing.T, timeout time.Duration) jsonrpc.Message {
	t.Helper()
	if err := p.conn.SetReadDeadline(time.Now().Add(timeout)); err != nil {
		t.Fatal(err)
	}
	var err error
	p.raw, err = readHistoryPhysical(p.reader, p.raw, jsonrpc.HistoryFrameBytes)
	if err != nil {
		t.Fatal(err)
	}
	var msg jsonrpc.Message
	if err := json.Unmarshal(p.raw, &msg); err != nil {
		t.Fatal(err)
	}
	return msg
}

func (h *historyTestHarness) emit(t *testing.T, raw []byte) {
	t.Helper()
	if _, err := h.output.Write(append(raw, '\n')); err != nil {
		t.Fatal(err)
	}
}

func (h *historyTestHarness) request(t *testing.T) jsonrpc.Message {
	t.Helper()
	select {
	case raw := <-h.input.requests:
		var msg jsonrpc.Message
		if err := json.Unmarshal(raw, &msg); err != nil {
			t.Fatal(err)
		}
		return msg
	case <-time.After(5 * time.Second):
		t.Fatal("broker request missing")
		return jsonrpc.Message{}
	}
}

func historyAwait(t *testing.T, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatal("history condition not reached")
		}
		time.Sleep(time.Millisecond)
	}
}

func historyTestRecord(id string, seq int64) nodepolicy.RoutingHistoryRecord {
	return nodepolicy.RoutingHistoryRecord{
		Schema: 1, Request: nodepolicy.RoutingRequestKey{OriginNodeID: "origin", ProxyRunID: "run", RequestID: id},
		Seq: seq, Requested: nodepolicy.RoutingHistoryRequested{Model: "model", Mode: "chat"}, CreatedAt: 1,
		State: "routing", Spend: nodepolicy.RoutingHistorySpend{Currency: "USD", Accounting: "none"},
		Attempts: []nodepolicy.RoutingHistoryAttempt{},
	}
}

func historyTestJSON(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func (h *historyTestHarness) changed(t *testing.T, record nodepolicy.RoutingHistoryRecord) {
	t.Helper()
	h.emit(t, historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", Method: "routing:changed", Params: historyTestJSON(t, record)}))
}

func (h *historyTestHarness) seed(t *testing.T, record nodepolicy.RoutingHistoryRecord) {
	t.Helper()
	h.s.mu.Lock()
	defer h.s.mu.Unlock()
	if err := record.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := h.s.history.work.set(&record); err != nil {
		t.Fatal(err)
	}
	if !h.s.history.merge(&h.s.history.work) {
		t.Fatal("seed rejected")
	}
	h.s.history.loaded = true
}

func TestHistoryCapacityOwnership(t *testing.T) {
	if got := unsafe.Sizeof(historyCursor{}); got != 32 {
		t.Fatalf("client state = %d, want 32", got)
	}
	if got := unsafe.Sizeof(historyCell{}); got != 65536 {
		t.Fatalf("cell = %d, want 65536", got)
	}
	s := newServiceWithHistory(config{})
	h := s.history
	if newService(config{}).history != nil {
		t.Fatal("ordinary construction enabled history")
	}
	if h.raw != nil || h.initial != nil || h.offsets != nil {
		t.Fatal("construction eagerly allocated data grants")
	}
	h.rawGrant()
	h.initialGrant()
	if cap(h.raw) != 33558528 || cap(h.initial) != 33558528 || cap(h.offsets)*4 != 33554432 {
		t.Fatal("initial grants do not match H/H/M")
	}
	data := cap(h.raw) + cap(h.initial) + 4*cap(h.offsets) + int(unsafe.Sizeof(h.work)) + 131072
	for i := range h.banks {
		bank := h.bank(i)
		data += int(unsafe.Sizeof(bank.cells))
	}
	for i := range h.writes {
		h.outputGrant(i)
		if cap(h.writes[i].storage) != 135168 {
			t.Fatalf("writer %d capacity = %d", i, cap(h.writes[i].storage))
		}
		data += cap(h.writes[i].storage)
	}
	if data != 152281088 {
		t.Fatalf("admitted data capacity = %d, want 152281088", data)
	}
	control := unsafe.Sizeof(*h) - unsafe.Sizeof(h.work)
	if control != 7976 {
		t.Fatalf("inline control owner = %d, want 7976", control)
	}
	// Includes the one control E bank. The timer replaces the old scheduler
	// ticker; these representation checks do not prove inherited semantic C.
	if control+unsafe.Sizeof(time.Timer{}) > 131072 {
		t.Fatal("control owner and timer representation exceed C")
	}
}

// These fixtures drive the real control writer and read/guard/apply owner
// synchronously. No scheduler or wall-clock delay stands in for a barrier.
func historyGeneratedFixture(t *testing.T) *historyTestHarness {
	t.Helper()
	h := &historyTestHarness{s: newServiceWithHistory(config{}), input: &historyTestInput{requests: make(chan []byte, 32)}}
	h.run = historyReadRun(h.input, nil)
	h.run.proc.kill = func() error { h.kills.Add(1); return nil }
	historyInstallRun(h.s, h.run)
	return h
}

func historyGeneratedRead(t *testing.T, h *historyTestHarness, frame string) *jsonrpc.Message {
	t.Helper()
	msg, _, reply, err := h.s.readHistoryMessage(h.run, bufio.NewReader(strings.NewReader(frame+"\n")))
	if err != nil || reply {
		t.Fatalf("generated broker read: reply=%v err=%v", reply, err)
	}
	return msg
}

func historyGeneratedBegin(t *testing.T, h *historyTestHarness) jsonrpc.Message {
	t.Helper()
	h.s.mu.Lock()
	h.s.history.retryCancelled = false
	h.s.history.refetch = true
	h.s.mu.Unlock()
	h.s.historyRefetch()
	return h.request(t)
}

func TestHistoryGeneratedQuarantineNoLateAOrdinaryAliveBSuccess(t *testing.T) {
	test := func(name string, reverse bool) {
		t.Run(name, func(t *testing.T) {
			h := historyGeneratedFixture(t)
			a := historyGeneratedBegin(t, h)
			ordinary := h.s.callBroker(h.run, "echo")
			o := h.request(t)
			if string(*a.ID) != `"jh:1"` || string(*o.ID) != "1" {
				t.Fatal("history changed the ordinary numeric allocator")
			}
			first, second := `"\u006ah\u003a1"`, string(*o.ID)
			if reverse {
				first, second = second, first
			}
			if msg := historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":`+first+`,"\u0069d":`+second+`,"result":{"requests":[]}}`); msg != nil {
				t.Fatal("ambiguous frame escaped guard")
			}
			msg := historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":1,"result":{"alive":true}}`)
			if msg == nil {
				t.Fatal("ordinary response was discarded")
			}
			h.s.routeResponse(msg)
			select {
			case got := <-ordinary:
				if got == nil || string(got.Result) != `{"alive":true}` {
					t.Fatal("ordinary O was settled by the ambiguity")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("ordinary O did not settle after its own response")
			}
			b := historyGeneratedBegin(t, h)
			if string(*b.ID) != `"jh:2"` {
				t.Fatal("B waited for or reused absent A")
			}
			historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":"jh:2","result":{"requests":[]}}`)
			if !h.s.history.loaded || h.s.history.correlation.Quarantined() || h.s.history.refetch || h.s.history.initialSlot >= 0 || h.kills.Load() != 0 {
				t.Fatal("applied B did not restore same-connection history")
			}
		})
	}
	test("history ID first", false)
	test("history ID last", true)
}

func TestHistoryGeneratedRepeatedQuarantineLateResultsPreserveNewerPending(t *testing.T) {
	h := historyGeneratedFixture(t)
	for _, candidates := range []string{`"id":"jh:1","ID":7`, `"id":7,"\u0069d":"\u006ah:2"`} {
		historyGeneratedBegin(t, h)
		historyGeneratedRead(t, h, `{"jsonrpc":"2.0",`+candidates+`,"result":{"requests":[]}}`)
	}
	b := historyGeneratedBegin(t, h)
	historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":`+string(*b.ID)+`,"result":{"requests":[]}}`)
	historyGeneratedRead(t, h, `{"broken":`)
	current := historyGeneratedBegin(t, h)
	for _, id := range []string{`"jh:1"`, `"\u006ah:2"`, `"jh:3"`, `"jh:99"`, `"jh:01"`, `"jh:18446744073709551616"`} {
		for _, body := range []string{`"result":{"requests":null}`, `"error":{"code":-32096,"message":"routing history busy; retry"}`} {
			if msg := historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":`+id+`,`+body+`}`); msg != nil {
				t.Fatal("stale or reserved response reached ordinary dispatch")
			}
			if !h.s.history.correlation.Quarantined() || !h.s.history.correlation.MatchesPending(*current.ID) || h.s.history.initialSlot < 0 {
				t.Fatal("stale result/error released newer quarantine or pending")
			}
		}
	}
	historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":`+string(*current.ID)+`,"result":{"requests":[]}}`)
	if h.s.history.correlation.Quarantined() || !h.s.history.loaded {
		t.Fatal("current valid apply did not recover")
	}
}

func TestHistoryGeneratedBusyAndErrorPreserveQuarantineUntilApplied(t *testing.T) {
	h := historyGeneratedFixture(t)
	historyGeneratedRead(t, h, `{"broken":`)
	for _, body := range []string{`{"code":-32096,"message":"routing history busy; retry"}`, `{"code":-32062,"message":"history-invalid"}`} {
		request := historyGeneratedBegin(t, h)
		historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":`+string(*request.ID)+`,"error":`+body+`}`)
		if !h.s.history.correlation.Quarantined() || h.s.history.loaded || !h.s.history.refetch || h.s.history.initialSlot >= 0 {
			t.Fatal("busy/error restored quarantine or lost loading retry")
		}
	}
	request := historyGeneratedBegin(t, h)
	historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":`+string(*request.ID)+`,"result":{"requests":[]}}`)
	if h.s.history.correlation.Quarantined() || !h.s.history.loaded || h.s.history.refetch {
		t.Fatal("applied current initial did not finish loading")
	}
}

type historyTestControlBarrier struct {
	delegate    io.WriteCloser
	entered     chan struct{}
	release     chan struct{}
	failed      bool
	written     int
	err         error
	releaseOnce sync.Once
}

func (w *historyTestControlBarrier) unblock() { w.releaseOnce.Do(func() { close(w.release) }) }

func (w *historyTestControlBarrier) Write(raw []byte) (int, error) {
	if !w.failed {
		w.failed = true
		close(w.entered)
		<-w.release
		if w.err != nil {
			return w.written, w.err
		}
	}
	return w.delegate.Write(raw)
}

func (w *historyTestControlBarrier) Close() error { return w.delegate.Close() }

func TestHistoryGeneratedQuarantineKeepsControlLoanUntilActualWrite(t *testing.T) {
	test := func(name string, written int, writeErr error) {
		t.Run(name, func(t *testing.T) {
			h := historyGeneratedFixture(t)
			barrier := &historyTestControlBarrier{delegate: h.input, entered: make(chan struct{}), release: make(chan struct{}), written: written, err: writeErr}
			h.run.proc.stdin = barrier
			h.s.history.refetch = true
			done := make(chan struct{})
			go func() { defer close(done); h.s.historyRefetch() }()
			t.Cleanup(func() { barrier.unblock(); historyWaitClosed(t, done) })
			historyWaitClosed(t, barrier.entered)
			original := append([]byte(nil), h.s.history.control[:h.s.history.controlN]...)
			historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":"jh:1","id":7,"result":{"requests":[]}}`)
			if h.s.history.pending || !h.s.history.controlActive || h.s.history.initialSlot >= 0 {
				t.Fatal("logical failure released the blocked control loan")
			}
			h.s.historyRefetch()
			if !bytes.Equal(original, h.s.history.control[:h.s.history.controlN]) || h.s.history.pending {
				t.Fatal("B borrowed or overwrote A control E before actual completion")
			}
			barrier.unblock()
			historyWaitClosed(t, done)
			if h.s.history.controlActive {
				t.Fatal("returned Write retained control admission")
			}
			if written != 0 && writeErr != nil {
				if h.kills.Load() != 1 || !h.s.history.controlBroken {
					t.Fatal("partial failed transport was reused")
				}
				return
			}
			if writeErr == nil {
				h.request(t) // A completed successfully after logical failure.
			}
			b := historyGeneratedBegin(t, h)
			if string(*b.ID) != `"jh:2"` || h.kills.Load() != 0 {
				t.Fatal("actual completion did not permit a fresh consumed ordinal")
			}
		})
	}
	test("blocked successful Write", 0, nil)
	test("blocked zero-progress failure", 0, errors.New("synthetic control failure"))
	test("blocked partial failure", 3, errors.New("synthetic control failure"))
	test("blocked unknown progress failure", -1, errors.New("synthetic control failure"))
}

func TestHistoryGeneratedOldControlCallbackCannotMutateReplacement(t *testing.T) {
	h := historyGeneratedFixture(t)
	barrier := &historyTestControlBarrier{delegate: h.input, entered: make(chan struct{}), release: make(chan struct{}), written: 3, err: errors.New("old transport failure")}
	h.run.proc.stdin = barrier
	h.s.history.refetch = true
	done := make(chan struct{})
	go func() { defer close(done); h.s.historyRefetch() }()
	t.Cleanup(func() { barrier.unblock(); historyWaitClosed(t, done) })
	historyWaitClosed(t, barrier.entered)
	h.s.brokerExited(h.run) // Actual source withdrawal fences old callbacks.
	freshInput := &historyTestInput{requests: make(chan []byte, 4)}
	fresh := historyReadRun(freshInput, nil)
	historyInstallRun(h.s, fresh)
	h.s.history.refetch = true
	h.s.historyRefetch()
	if h.s.history.pending || !h.s.history.controlActive {
		t.Fatal("replacement borrowed E before old Write returned")
	}
	barrier.unblock()
	historyWaitClosed(t, done)
	if h.s.history.controlBroken || h.s.history.correlation.Quarantined() || h.s.history.pending {
		t.Fatal("old callback mutated replacement correlation")
	}
	freshHarness := &historyTestHarness{s: h.s, run: fresh, input: freshInput}
	request := historyGeneratedBegin(t, freshHarness)
	if string(*request.ID) != `"jh:1"` {
		t.Fatal("fenced generation did not reset its own frontier")
	}
	_, _, _, err := h.s.readHistoryMessage(h.run, bufio.NewReader(strings.NewReader(`{"jsonrpc":"2.0","id":"jh:1","result":{"requests":[]}}`+"\n")))
	if !errors.Is(err, io.EOF) || !h.s.history.correlation.MatchesPending(*request.ID) || h.s.history.loaded {
		t.Fatal("old response with a reused ordinal escaped its actual source fence")
	}
	historyGeneratedRead(t, freshHarness, `{"jsonrpc":"2.0","id":"jh:1","result":{"requests":[]}}`)
	if !h.s.history.loaded {
		t.Fatal("replacement failed to apply its own response")
	}
}

func TestHistoryGeneratedRetryBackoffCancellationAndExhaustion(t *testing.T) {
	for _, test := range []struct {
		step uint8
		want time.Duration
	}{{0, 250 * time.Millisecond}, {1, 500 * time.Millisecond}, {2, time.Second}, {3, time.Second}, {255, time.Second}} {
		if got := historyRetryDelay(test.step); got != test.want {
			t.Fatalf("retry step %d = %v, want %v", test.step, got, test.want)
		}
	}
	h := historyGeneratedFixture(t)
	t.Cleanup(func() {
		if h.s.history.retryTimer != nil {
			h.s.history.retryTimer.Stop()
		}
	})
	h.s.history.refetch = true
	h.s.history.retryTimer = time.NewTimer(time.Hour)
	h.s.history.retryArmed = true
	h.s.history.exhausted = true // Shared owner tests exercise the real MaxUint64 boundary.
	h.s.historyRefetch()
	if h.s.history.refetch || h.s.history.retryArmed || h.s.history.pending {
		t.Fatal("exhausted history retained automatic retry")
	}
	ordinary := h.s.callBroker(h.run, "echo")
	request := h.request(t)
	msg := historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":`+string(*request.ID)+`,"result":{"alive":true}}`)
	if msg == nil {
		t.Fatal("history exhaustion discarded an ordinary response")
	}
	h.s.routeResponse(msg)
	select {
	case got := <-ordinary:
		if got == nil || string(got.Result) != `{"alive":true}` {
			t.Fatal("history exhaustion stopped ordinary RPC")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("ordinary response did not settle during history exhaustion")
	}
	h.s.history.exhausted = false
	h.s.history.refetch = true
	h.s.history.retryTimer.Reset(time.Hour)
	h.s.history.retryArmed = true
	c := &client{out: make(chan []byte, 1)}
	id := json.RawMessage(`"unsubscribe"`)
	h.s.handleHistoryRequest(c, &jsonrpc.Message{JSONRPC: "2.0", ID: &id, Method: nodepolicy.MethodRoutingUnsubscribe})
	if h.s.history.refetch || h.s.history.retryArmed {
		t.Fatal("unsubscribe retained its retry timer")
	}
	h.s.history.refetch = true
	h.s.history.retryTimer.Reset(time.Hour)
	h.s.history.retryArmed = true
	h.s.brokerExited(h.run)
	if h.s.history.refetch || h.s.history.retryArmed {
		t.Fatal("generation loss retained its retry timer")
	}
}

func TestHistoryDisconnectCancelsUnusedRetry(t *testing.T) {
	s := newServiceWithHistory(config{})
	h := s.history
	h.refetch = true
	h.retryTimer = time.NewTimer(time.Hour)
	h.retryArmed = true
	t.Cleanup(func() { h.retryTimer.Stop() })
	clients := make([]*client, 2)
	for i := range clients {
		server, peer := net.Pipe()
		t.Cleanup(func() { _ = peer.Close() })
		clients[i] = newClient(i, server)
		clients[i].history.flags = historySubscribed
		s.clients[clients[i]] = struct{}{}
		t.Cleanup(func() { clients[i].closeNow(); historyWaitClosed(t, clients[i].written) })
	}
	s.detach(clients[0], false)
	if !h.refetch || !h.retryArmed || h.retryCancelled {
		t.Fatal("disconnect cancelled another subscriber's retry")
	}
	s.detach(clients[1], false)
	if h.refetch || h.retryArmed || !h.retryCancelled {
		t.Fatal("last subscriber disconnect retained automatic retry")
	}
	s.historyRefetch()
	if h.pending || h.initialSlot >= 0 {
		t.Fatal("cancelled retry acquired another transfer")
	}
}

func TestHistoryGeneratedBusyRetryUsesSingleSchedulerTimer(t *testing.T) {
	h := historyGeneratedFixture(t)
	h.s.startHistory()
	t.Cleanup(func() {
		close(h.s.stopCh)
		done := make(chan struct{})
		go func() { h.s.history.wg.Wait(); close(done) }()
		historyWaitClosed(t, done)
	})
	h.s.mu.Lock()
	h.s.history.refetch = true
	h.s.history.signal()
	h.s.mu.Unlock()
	var timer *time.Timer
	for _, test := range []struct {
		id   string
		step uint8
	}{{`"jh:1"`, 1}, {`"jh:2"`, 2}, {`"jh:3"`, 2}} {
		request := h.request(t) // Actual timer/control Write completion barrier.
		if request.ID == nil || string(*request.ID) != test.id {
			t.Fatalf("retry token = %v, want %s", request.ID, test.id)
		}
		h.s.mu.Lock()
		if timer == nil {
			timer = h.s.history.retryTimer
		}
		valid := timer != nil && h.s.history.retryTimer == timer && h.s.history.retryStep == test.step && !h.s.history.retryArmed && h.s.history.pending
		h.s.mu.Unlock()
		if !valid {
			t.Fatal("busy retry replaced its timer or admitted concurrent pending work")
		}
		if test.id == `"jh:3"` {
			c := &client{out: make(chan []byte, 1)}
			id := json.RawMessage(`"unsubscribe"`)
			h.s.handleHistoryRequest(c, &jsonrpc.Message{JSONRPC: "2.0", ID: &id, Method: nodepolicy.MethodRoutingUnsubscribe})
		}
		historyGeneratedRead(t, h, `{"jsonrpc":"2.0","id":`+test.id+`,"error":{"code":-32096,"message":"routing history busy; retry"}}`)
	}
	h.s.mu.Lock()
	cancelled := !h.s.history.refetch && !h.s.history.retryArmed && !h.s.history.pending && h.s.history.initialSlot < 0
	h.s.mu.Unlock()
	if !cancelled {
		t.Fatal("late busy response restarted an unsubscribed retry")
	}
}

func TestHistoryPackedRoundTrip(t *testing.T) {
	record := historyTestRecord("9223372036854775807", 9007199254740991)
	zero := nodepolicy.NanoUSD("0")
	maximum := nodepolicy.NanoUSD("9223372036854775807")
	record.Spend.HeldNanoUSD, record.Spend.CeilingNanoUSD = &zero, &maximum
	cell := new(historyCell)
	if err := cell.set(&record); err != nil {
		t.Fatal(err)
	}
	e := historyEncoder{out: make([]byte, 0, 131072)}
	e.record(cell)
	if e.err != nil {
		t.Fatal(e.err)
	}
	actual, _, err := nodepolicy.DecodeRoutingHistoryRecord(e.out)
	if err != nil {
		t.Fatal(err)
	}
	if actual.Request.RequestID != "9223372036854775807" || actual.Seq != 9007199254740991 || actual.Workload != nil || actual.EndedAt != nil || actual.Attempts == nil {
		t.Fatalf("identity/nullability changed: %+v", actual)
	}
	if actual.Spend.HeldNanoUSD == nil || string(*actual.Spend.HeldNanoUSD) != "0" || actual.Spend.CeilingNanoUSD == nil || string(*actual.Spend.CeilingNanoUSD) != "9223372036854775807" {
		t.Fatalf("money changed: %+v", actual.Spend)
	}
}

func TestHistoryIntegerSpellingsUseSharedSchema(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":{}}`)
	if msg := p.next(t); msg.Error != nil {
		t.Fatal(msg.Error)
	}
	historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot < 0 })
	raw := string(historyTestJSON(t, historyTestRecord("1", 3)))
	raw = strings.Replace(raw, `"seq":3`, `"seq":3e0`, 1)
	raw = strings.Replace(raw, `"createdAt":1`, `"createdAt":1.0`, 1)
	h.emit(t, []byte(`{"jsonrpc":"2.0","method":"routing:changed","params":`+raw+`}`))
	msg := p.next(t)
	var record nodepolicy.RoutingHistoryRecord
	if err := json.Unmarshal(msg.Params, &record); err != nil {
		t.Fatal(err)
	}
	if record.Seq != 3 || record.CreatedAt != 1 {
		t.Fatalf("numeric spellings changed: %+v", record)
	}
}

func TestHistoryMaximumEncodedCallerID(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	p := h.peer()
	id := `"` + strings.Repeat("x", 510) + `"`
	p.send(t, `{"jsonrpc":"2.0","id":`+id+`,"method":"routing:get-initial","params":{}}`)
	msg := p.next(t)
	if msg.Error != nil || msg.ID == nil || string(*msg.ID) != id {
		t.Fatal("legal 512-byte encoded ID was not restored exactly")
	}
}

func TestHistoryInitialPinsUntilActualWrite(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	blocked := h.peer()
	blocked.send(t, `{"jsonrpc":"2.0","id":"\u0061<>&","method":"routing:subscribe","params":{}}`)
	historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot >= 0 })
	other := h.peer()
	other.send(t, `{"jsonrpc":"2.0","id":"\u0062<>&","method":"routing:get-initial","params":{}}`)
	busy := other.next(t)
	if busy.Error == nil || busy.Error.Code != -32096 || busy.Error.Message != "routing history busy; retry" || string(*busy.ID) != `"\u0062<>&"` {
		t.Fatalf("busy response = %+v", busy)
	}
	other.send(t, `{"jsonrpc":"2.0","id":3,"method":"service/status"}`)
	status := other.next(t)
	if status.Error != nil || string(*status.ID) != "3" {
		t.Fatalf("ordinary status blocked: %+v", status)
	}
	other.send(t, `{"jsonrpc":"2.0","id":4,"method":"echo","params":{"alive":true}}`)
	request := h.request(t)
	h.emit(t, historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", ID: request.ID, Result: []byte(`{"alive":true}`)}))
	ordinary := other.next(t)
	if ordinary.Error != nil || string(*ordinary.ID) != "4" {
		t.Fatalf("ordinary relay blocked: %+v", ordinary)
	}
	initial := blocked.next(t)
	if string(*initial.ID) != `"\u0061<>&"` || initial.Error != nil {
		t.Fatalf("raw initial ID changed: %+v", initial)
	}
	historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot < 0 })
	other.send(t, `{"jsonrpc":"2.0","id":5,"method":"routing:get-initial","params":{}}`)
	if reply := other.next(t); reply.Error != nil {
		t.Fatalf("released initial remained busy: %+v", reply.Error)
	}
}

func TestHistoryInitialFollowsCachedAppReady(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	h.s.mu.Lock()
	h.s.readyFrame = []byte("{\"jsonrpc\":\"2.0\",\"method\":\"app:ready\",\"params\":{}}\n")
	h.s.mu.Unlock()
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:get-initial","params":{}}`)
	if msg := p.next(t); msg.Method != "app:ready" {
		t.Fatalf("history overtook cached app:ready: %+v", msg)
	}
	if msg := p.next(t); msg.ID == nil || string(*msg.ID) != "1" || msg.Error != nil {
		t.Fatalf("initial after app:ready = %+v", msg)
	}
}

func TestHistoryUnsubscribeUsesInitialShapeAndStopsPushes(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":{}}`)
	if msg := p.next(t); msg.Error != nil {
		t.Fatal(msg.Error)
	}
	historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot < 0 })
	p.send(t, `{"jsonrpc":"2.0","id":2,"method":"routing:unsubscribe","params":{}}`)
	msg := p.next(t)
	var initial nodepolicy.RoutingHistoryInitial
	if err := json.Unmarshal(msg.Result, &initial); err != nil {
		t.Fatal(err)
	}
	if msg.Error != nil || initial.Requests == nil || len(initial.Requests) != 0 {
		t.Fatal("unsubscribe did not return an empty initial")
	}
	h.changed(t, historyTestRecord("1", 2))
	h.emit(t, []byte(`{"jsonrpc":"2.0","method":"test:barrier","params":{}}`))
	if next := p.next(t); next.Method != "test:barrier" {
		t.Fatalf("unsubscribed client received history: %+v", next)
	}
	select {
	case <-h.input.requests:
		t.Fatal("local unsubscribe reached shared broker stream")
	default:
	}
}

func TestHistoryFairRegistryBeyondEightSubscribers(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	peers := make([]*historyTestPeer, 12)
	for i := range peers {
		peers[i] = h.peer()
		peers[i].send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":{}}`)
		if reply := peers[i].next(t); reply.Error != nil {
			t.Fatalf("subscriber %d refused: %+v", i, reply.Error)
		}
		historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot < 0 })
	}
	h.changed(t, historyTestRecord("1", 2))
	historyAwait(t, func() bool {
		h.s.mu.Lock()
		defer h.s.mu.Unlock()
		n := 0
		for _, w := range h.s.history.writes {
			if w.active {
				n++
			}
		}
		return n == 8
	})
	for seq := int64(3); seq <= 300; seq++ {
		h.changed(t, historyTestRecord("1", seq))
	}
	historyAwait(t, func() bool {
		h.s.mu.Lock()
		defer h.s.mu.Unlock()
		return h.s.history.banks[h.s.history.current].cells[0].seq == 300
	})
	h.s.mu.Lock()
	if len(h.s.clients) != 12 {
		t.Fatalf("registered subscribers = %d, want 12", len(h.s.clients))
	}
	for c := range h.s.clients {
		if len(c.out) != 0 {
			t.Errorf("history used ordinary queue for client %d", c.id)
		}
	}
	if h.s.history.banks[h.s.history.current].n != 1 {
		t.Error("updates accumulated instead of coalescing")
	}
	h.s.mu.Unlock()
	peers[8].send(t, `{"jsonrpc":"2.0","id":2,"method":"service/status"}`)
	if reply := peers[8].next(t); reply.ID == nil || string(*reply.ID) != "2" {
		t.Fatalf("ordinary status did not progress: %+v", reply)
	}
	// One released writer must visit waiting subscribers before wrapping back
	// to the first slow subscriber. No subscriber owns a waiting goroutine.
	if first := peers[0].next(t); first.Method != "routing:changed" {
		t.Fatalf("first lease = %+v", first)
	}
	for i := 8; i < 12; i++ {
		msg := peers[i].next(t)
		var record nodepolicy.RoutingHistoryRecord
		if err := json.Unmarshal(msg.Params, &record); err != nil {
			t.Fatal(err)
		}
		if msg.Method != "routing:changed" || record.Seq != 300 {
			t.Fatalf("subscriber %d = %s seq %d, want latest 300", i, msg.Method, record.Seq)
		}
	}
}

func TestHistoryDelayedBaselineMergesPushAndRankFloor(t *testing.T) {
	h := newHistoryTestHarness(t)
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:get-initial","params":{}}`)
	request := h.request(t)
	h.changed(t, historyTestRecord("1", 3))
	baseline := nodepolicy.RoutingHistoryInitial{Requests: []nodepolicy.RoutingHistoryRecord{historyTestRecord("1", 1)}}
	h.emit(t, historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", ID: request.ID, Result: historyTestJSON(t, baseline)}))
	msg := p.next(t)
	var result nodepolicy.RoutingHistoryInitial
	if err := json.Unmarshal(msg.Result, &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Requests) != 1 || result.Requests[0].Seq != 3 {
		t.Fatalf("stale baseline replaced push: %+v", result)
	}
	for id := 2; id <= 300; id++ {
		h.changed(t, historyTestRecord(strconv.Itoa(id), 1))
	}
	historyAwait(t, func() bool {
		h.s.mu.Lock()
		defer h.s.mu.Unlock()
		bank := h.s.history.banks[h.s.history.current]
		for i := 0; i < bank.n; i++ {
			if string(bank.cells[i].id[:bank.cells[i].lengths[2]]) == "300" {
				return true
			}
		}
		return false
	})
	h.changed(t, historyTestRecord("1", 900))
	h.emit(t, []byte(`{"jsonrpc":"2.0","method":"test:barrier","params":{}}`))
	if msg := p.next(t); msg.Method != "test:barrier" {
		t.Fatalf("barrier = %+v", msg)
	}
	h.s.mu.Lock()
	bank := h.s.history.banks[h.s.history.current]
	if bank.n != 256 {
		t.Errorf("cache rows = %d, want 256", bank.n)
	}
	for i := 0; i < bank.n; i++ {
		id, err := strconv.Atoi(string(bank.cells[i].id[:bank.cells[i].lengths[2]]))
		if err != nil {
			t.Error(err)
		}
		if id < 45 || id > 300 {
			t.Errorf("retained ID = %d, want 45..300", id)
		}
	}
	h.s.mu.Unlock()
}

func TestHistoryTerminalCannotReopen(t *testing.T) {
	h := newHistoryTestHarness(t)
	terminal := historyTestRecord("1", 1)
	end := int64(1)
	terminal.State, terminal.EndedAt = "failed", &end
	h.seed(t, terminal)
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":{}}`)
	if msg := p.next(t); msg.Error != nil {
		t.Fatal(msg.Error)
	}
	historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot < 0 })
	h.changed(t, historyTestRecord("1", 2))
	changedRank := terminal
	changedRank.CreatedAt, changedRank.Seq = 0, 3
	h.changed(t, changedRank)
	terminal.Seq = 4
	known := nodepolicy.NanoUSD("1234567890123456789")
	terminal.Spend.ChargedNanoUSD = &known
	h.changed(t, terminal)
	msg := p.next(t)
	var actual nodepolicy.RoutingHistoryRecord
	if err := json.Unmarshal(msg.Params, &actual); err != nil {
		t.Fatal(err)
	}
	if actual.Seq != 4 || actual.State != "failed" || actual.CreatedAt != 1 || actual.Spend.ChargedNanoUSD == nil || *actual.Spend.ChargedNanoUSD != known {
		t.Fatalf("terminal merge = %+v", actual)
	}
}

func TestHistoryPartialFailedWriteReleasesActualBorrower(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	p := h.peerWithChunk(3)
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:get-initial","params":{}}`)
	var prefix [3]byte
	if err := p.conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadFull(p.conn, prefix[:]); err != nil {
		t.Fatal(err)
	}
	if string(prefix[:]) != `{"j` {
		t.Fatalf("partial prefix = %q", prefix)
	}
	h.changed(t, historyTestRecord("1", 2))
	historyAwait(t, func() bool {
		h.s.mu.Lock()
		defer h.s.mu.Unlock()
		return h.s.history.banks[h.s.history.current].cells[0].seq == 2
	})
	h.s.mu.Lock()
	slot := h.s.history.initialSlot
	if slot < 0 {
		h.s.mu.Unlock()
		t.Fatal("partial writer released initial early")
	}
	frame := h.s.history.writes[slot].frame
	var msg jsonrpc.Message
	if err := json.Unmarshal(frame, &msg); err != nil {
		h.s.mu.Unlock()
		t.Fatal(err)
	}
	var initial nodepolicy.RoutingHistoryInitial
	if err := json.Unmarshal(msg.Result, &initial); err != nil {
		h.s.mu.Unlock()
		t.Fatal(err)
	}
	if len(initial.Requests) != 1 || initial.Requests[0].Seq != 1 {
		h.s.mu.Unlock()
		t.Fatal("blocked initial buffer was reused")
	}
	h.s.mu.Unlock()
	if err := p.conn.Close(); err != nil {
		t.Fatal(err)
	}
	historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot < 0 })
	other := h.peer()
	other.send(t, `{"jsonrpc":"2.0","id":2,"method":"routing:get-initial","params":{}}`)
	reply := other.next(t)
	if err := json.Unmarshal(reply.Result, &initial); err != nil {
		t.Fatal(err)
	}
	if len(initial.Requests) != 1 || initial.Requests[0].Seq != 2 {
		t.Fatal("cache failed to converge after actual failed-write completion")
	}
}

type historyTestLog struct {
	mu   sync.Mutex
	data bytes.Buffer
}

func (w *historyTestLog) Write(raw []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.data.Write(raw)
}

func (w *historyTestLog) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.data.String()
}

func TestHistoryMalformedBoundariesNeverTraceRaw(t *testing.T) {
	var logs historyTestLog
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	h := newHistoryTestHarness(t)
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","method":"routing:changed","params":{"private-secret":"synthetic-client-secret"}}`)
	h.emit(t, []byte(`{"jsonrpc":"2.0","method":"routing:changed","params":{"private-secret":"synthetic-broker-secret","private-secret":0}}`))
	h.emit(t, []byte(`{"synthetic-raw-secret":`))
	h.emit(t, []byte(`{"jsonrpc":"2.0","method":"test:barrier","params":{}}`))
	if msg := p.next(t); msg.Method != "test:barrier" {
		t.Fatalf("malformed input escaped to client: %+v", msg)
	}
	if h.kills.Load() != 0 {
		t.Fatal("in-budget malformed history killed broker")
	}
	h.s.mu.Lock()
	if bank := h.s.history.banks[h.s.history.current]; bank != nil && bank.n != 0 {
		t.Error("client injection applied history")
	}
	h.s.mu.Unlock()
	output := logs.String()
	if !strings.Contains(output, "client attached") {
		t.Fatal("log capture did not observe production seam")
	}
	for _, marker := range []string{"synthetic-client-secret", "synthetic-broker-secret", "synthetic-raw-secret", "private-secret"} {
		if strings.Contains(output, marker) {
			t.Fatalf("raw marker reached log: %s", marker)
		}
	}
}

func TestHistoryRawAmbiguityPreservesOrdinaryPending(t *testing.T) {
	test := func(name string, reverse bool) {
		t.Run(name, func(t *testing.T) {
			h := newHistoryTestHarness(t)
			history := h.peer()
			ordinary := h.peer()
			history.send(t, `{"jsonrpc":"2.0","id":"history","method":"routing:get-initial","params":{}}`)
			pending := h.request(t)
			ordinary.send(t, `{"jsonrpc":"2.0","id":7,"method":"echo","params":{}}`)
			echo := h.request(t)
			first, second := string(*pending.ID), string(*echo.ID)
			if reverse {
				first, second = second, first
			}
			frame := `{"jsonrpc":"2.0","id":` + first + `,"\u0069d":` + second + `,"result":{"requests":[]},"private-secret":"synthetic-raw-secret"}`
			h.emit(t, []byte(frame))
			failure := history.next(t)
			if failure.Error == nil || failure.Error.Code != -32062 || failure.Error.Message != "history-invalid" {
				t.Fatalf("pending history did not fail once: %+v", failure)
			}
			h.emit(t, historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", ID: echo.ID, Result: []byte(`{"alive":true}`)}))
			if reply := ordinary.next(t); reply.Error != nil || string(*reply.ID) != "7" {
				t.Fatalf("ambiguity resolved ordinary caller: %+v", reply)
			}
			if h.kills.Load() != 0 {
				t.Fatal("in-budget ambiguity restarted broker")
			}
			ordinary.send(t, `{"jsonrpc":"2.0","id":8,"method":"service/status"}`)
			if reply := ordinary.next(t); reply.Error != nil {
				t.Fatal(reply.Error)
			}
		})
	}
	test("history ID first", false)
	test("history ID last", true)
}

func TestHistoryQuarantineNeedsGuardedInitial(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:subscribe","params":{}}`)
	if msg := p.next(t); msg.Error != nil {
		t.Fatal(msg.Error)
	}
	historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot < 0 })
	// Drive caller-requested recovery explicitly in this case. Generated
	// automatic refetch is covered separately through its control/read owners.
	h.s.mu.Lock()
	h.s.history.retryCancelled = true
	h.s.mu.Unlock()
	h.emit(t, []byte(`{"malformed-synthetic-secret":`))
	h.changed(t, historyTestRecord("1", 2))
	h.emit(t, []byte(`{"jsonrpc":"2.0","method":"test:barrier","params":{}}`))
	if msg := p.next(t); msg.Method != "test:barrier" {
		t.Fatalf("quarantine applied history before refetch: %+v", msg)
	}
	p.send(t, `{"jsonrpc":"2.0","id":2,"method":"routing:get-initial","params":{}}`)
	request := h.request(t)
	initial := nodepolicy.RoutingHistoryInitial{Requests: []nodepolicy.RoutingHistoryRecord{historyTestRecord("1", 3)}}
	h.emit(t, historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", ID: request.ID, Result: historyTestJSON(t, initial)}))
	msg := p.next(t)
	var actual nodepolicy.RoutingHistoryInitial
	if msg.ID == nil || string(*msg.ID) != "2" || msg.Error != nil {
		t.Fatalf("recovery reply = %+v", msg)
	}
	if err := json.Unmarshal(msg.Result, &actual); err != nil {
		t.Fatal(err)
	}
	if len(actual.Requests) != 1 || actual.Requests[0].Seq != 3 {
		t.Fatal("guarded initial did not re-establish history")
	}
	if h.kills.Load() != 0 {
		t.Fatal("quarantine restarted broker")
	}
}

func TestHistoryClientMethodAmbiguityBeforeDispatch(t *testing.T) {
	test := func(name, first, second string) {
		t.Run(name, func(t *testing.T) {
			h := newHistoryTestHarness(t)
			h.seed(t, historyTestRecord("1", 1))
			p := h.peer()
			p.send(t, `{"jsonrpc":"2.0","id":1,"method":"`+first+`","\u006dethod":"`+second+`","params":{}}`)
			p.send(t, `{"jsonrpc":"2.0","id":2,"method":"service/status"}`)
			msg := p.next(t)
			if msg.ID == nil || string(*msg.ID) != "2" || msg.Error != nil {
				t.Fatalf("ambiguous method reached dispatch: %+v", msg)
			}
			h.s.mu.Lock()
			if h.s.history.initialSlot >= 0 {
				t.Error("ambiguous frame acquired initial lease")
			}
			for c := range h.s.clients {
				if c.history.flags&historySubscribed != 0 {
					t.Error("ambiguous frame subscribed client")
				}
			}
			h.s.mu.Unlock()
			select {
			case <-h.input.requests:
				t.Fatal("ambiguous client frame reached broker")
			default:
			}
		})
	}
	test("history first", "routing:subscribe", "service/status")
	test("history last", "service/status", "routing:subscribe")
}

func TestHistoryPhysicalEOFKeepsFramingClassification(t *testing.T) {
	test := func(name, raw string, want error) {
		t.Run(name, func(t *testing.T) {
			_, err := readHistoryPhysical(bufio.NewReader(strings.NewReader(raw)), nil, jsonrpc.WorkerFrameBytes)
			if err != want {
				t.Fatalf("error = %v, want %v", err, want)
			}
		})
	}
	test("empty stream", "", io.EOF)
	test("truncated frame", "{", jsonrpc.ErrHistoryFraming)
}

func TestHistoryPhysicalReadTimeoutPreservesCause(t *testing.T) {
	server, peer := net.Pipe()
	defer server.Close()
	defer peer.Close()
	if err := peer.SetReadDeadline(time.Now()); err != nil {
		t.Fatal(err)
	}
	_, err := readHistoryPhysical(bufio.NewReader(peer), nil, jsonrpc.WorkerFrameBytes)
	if !errors.Is(err, jsonrpc.ErrHistoryFraming) {
		t.Fatalf("error = %v, want history framing failure", err)
	}
	var timeout net.Error
	if !errors.As(err, &timeout) || !timeout.Timeout() {
		t.Fatalf("error = %v, want wrapped read timeout", err)
	}
	if !strings.Contains(err.Error(), "i/o timeout") {
		t.Fatalf("error = %v, want visible read timeout", err)
	}
}

func TestHistoryPhysicalOverflowStopsOnlyAtH(t *testing.T) {
	test := func(name string, size int, want error) {
		t.Run(name, func(t *testing.T) {
			raw := strings.Repeat("x", size-1) + "\n"
			_, err := readHistoryPhysical(bufio.NewReader(strings.NewReader(raw)), make([]byte, 0, 4096), jsonrpc.WorkerFrameBytes)
			if err != want {
				t.Fatalf("error = %v, want %v", err, want)
			}
		})
	}
	test("physical H drains logical overflow", 33558528, jsonrpc.ErrHistoryInvalid)
	test("physical H plus one terminates", 33558529, jsonrpc.ErrHistoryFraming)
}

func TestHistoryOrdinaryLogicalFrameLimit(t *testing.T) {
	test := func(name string, size int, accepted bool) {
		t.Run(name, func(t *testing.T) {
			prefix := `{"jsonrpc":"2.0","method":"ordinary:changed","params":{"padding":"`
			suffix := "\"}}\n"
			frame := prefix + strings.Repeat("x", size-len(prefix)-len(suffix)) + suffix
			s := newServiceWithHistory(config{})
			run := historyReadRun(&historyTestInput{requests: make(chan []byte, 1)}, strings.NewReader(frame))
			historyInstallRun(s, run)
			msg, raw, reply, err := s.readHistoryMessage(run, bufio.NewReader(run.proc.stdout))
			if err != nil || reply {
				t.Fatalf("ordinary frame read: reply=%v err=%v", reply, err)
			}
			if !accepted {
				if msg != nil || raw != nil {
					t.Fatal("physical H raised the ordinary logical frame limit")
				}
				return
			}
			if msg == nil || msg.Method != "ordinary:changed" || !bytes.Equal(raw, []byte(frame)) {
				t.Fatal("ordinary frame at its logical limit was changed or rejected")
			}
		})
	}
	test("ordinary limit accepted", jsonrpc.WorkerFrameBytes, true)
	test("ordinary limit plus one discarded", jsonrpc.WorkerFrameBytes+1, false)
}

func TestHistoryBrokerOverflowUsesTerminalFramingPath(t *testing.T) {
	var logs historyTestLog
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	h := newHistoryTestHarness(t)
	p := h.peer()
	writeDone := make(chan error, 1)
	go func() {
		// Exactly H without a newline must terminate immediately; no H+1st
		// byte or EOF is needed to wake the framing owner.
		n, err := io.WriteString(h.output, `{"synthetic-overflow-secret":"`+strings.Repeat("s", 33558528-len(`{"synthetic-overflow-secret":"`)))
		if err == nil && n != 33558528 {
			err = io.ErrShortWrite
		}
		writeDone <- err
	}()
	historyAwait(t, func() bool { return h.kills.Load() == 1 })
	select {
	case err := <-writeDone:
		if err != nil && !errors.Is(err, io.ErrClosedPipe) {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("overflow writer was not interrupted")
	}
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"service/status"}`)
	if msg := p.next(t); msg.Error != nil {
		t.Fatal(msg.Error)
	}
	if strings.Contains(logs.String(), "synthetic-overflow-secret") {
		t.Fatal("overflow prefix reached raw trace")
	}
}

func TestHistoryUnsentOutputCannotFollowBrokerReset(t *testing.T) {
	s := newServiceWithHistory(config{})
	server, peer := net.Pipe()
	defer peer.Close()
	c := newClient(1, server)
	t.Cleanup(func() { c.closeNow(); historyWaitClosed(t, c.written) })
	s.clients[c] = struct{}{}
	run := historyReadRun(&historyTestInput{requests: make(chan []byte, 1)}, nil)
	historyInstallRun(s, run)
	w := historyWrite{client: c, generation: s.history.generation, frame: historyChangedFrame(t, "1")}
	c.history.writeMu.Lock()
	done := make(chan struct{})
	writeErrors := make(chan error, 1)
	go func() { defer close(done); writeErrors <- s.writeHistoryFrame(w) }()
	s.brokerExited(run)
	s.broadcast(notifyBrokerRestarted, map[string]int{"brokerPid": 2, "restarts": 1})
	c.history.writeMu.Unlock()
	p := &historyTestPeer{conn: peer, reader: bufio.NewReader(peer)}
	if msg := p.next(t); msg.Method != notifyBrokerRestarted {
		t.Fatalf("retired unsent frame escaped its generation fence: %s", msg.Method)
	}
	historyWaitClosed(t, done)
	if err := <-writeErrors; err != nil {
		t.Fatal(err)
	}
	c.notify("test:barrier", nil)
	if msg := p.next(t); msg.Method != "test:barrier" {
		t.Fatalf("old history followed the broker reset: %s", msg.Method)
	}
}

func TestHistoryStartedOutputPrecedesBrokerReset(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	p := h.peerWithChunk(3)
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:get-initial","params":{}}`)
	var prefix [3]byte
	if err := p.conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadFull(p.conn, prefix[:]); err != nil {
		t.Fatal(err)
	}
	h.s.brokerExited(h.run)
	h.s.broadcast(notifyBrokerRestarted, map[string]int{"brokerPid": 2, "restarts": 1})
	rest, err := readHistoryPhysical(p.reader, nil, jsonrpc.HistoryFrameBytes)
	if err != nil {
		t.Fatal(err)
	}
	var initial jsonrpc.Message
	if err := json.Unmarshal(append(prefix[:], rest...), &initial); err != nil {
		t.Fatal(err)
	}
	if initial.ID == nil || string(*initial.ID) != "1" || initial.Error != nil {
		t.Fatal("started initial was interleaved with the broker reset")
	}
	if msg := p.next(t); msg.Method != notifyBrokerRestarted {
		t.Fatalf("reset did not follow the completed history write: %s", msg.Method)
	}
}

func TestHistoryOversizedClientRequestUsesSharedInitialArena(t *testing.T) {
	test := func(name, id, method string, code int, wantID string) {
		t.Run(name, func(t *testing.T) {
			h := newHistoryTestHarness(t)
			p := h.peer()
			p.send(t, `{"jsonrpc":"2.0","id":`+id+`,"method":"`+method+`","params":{"padding":"`+strings.Repeat("x", 9<<20)+`"}}`)
			msg := p.next(t)
			var envelope struct{ ID json.RawMessage }
			if err := json.Unmarshal(p.raw, &envelope); err != nil {
				t.Fatal(err)
			}
			if msg.Error == nil || msg.Error.Code != code || string(envelope.ID) != wantID || len(p.raw) > jsonrpc.HistoryEnvelopeBytes {
				t.Fatalf("oversized request refusal: %s", p.raw)
			}
			h.s.mu.Lock()
			if h.s.history.initialSlot >= 0 || cap(h.s.history.initial) != jsonrpc.HistoryFrameBytes {
				t.Error("oversized input did not use the sole initial arena")
			}
			for _, bank := range h.s.history.banks {
				if bank != nil {
					t.Error("refused input created history cache data")
				}
			}
			h.s.mu.Unlock()
			p.send(t, `{"jsonrpc":"2.0","id":2,"method":"service/status"}`)
			if next := p.next(t); next.ID == nil || string(*next.ID) != "2" || next.Error != nil {
				t.Fatal("ordinary RPC did not progress after oversized refusal")
			}
			historyAwait(t, func() bool {
				if !h.s.history.initialMu.TryLock() {
					return false
				}
				h.s.history.initialMu.Unlock()
				return true
			})
			if h.kills.Load() != 0 {
				t.Fatal("in-budget client input killed the broker")
			}
		})
	}
	test("public params", "1", "routing:subscribe", jsonrpc.HistoryInvalidCode, "1")
	test("private method", `"\u0061"`, "routing/request", codeMethodNotFound, `"\u0061"`)
	test("oversized ID", `"`+strings.Repeat("x", 511)+`"`, "routing:get-initial", jsonrpc.HistoryInvalidCode, "null")
}

func TestHistoryBrokerExitPinsAdmittedInitial(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	p := h.peerWithChunk(3)
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:get-initial","params":{}}`)
	var prefix [3]byte
	if err := p.conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadFull(p.conn, prefix[:]); err != nil {
		t.Fatal(err)
	}
	h.s.brokerExited(h.run)
	h.s.mu.Lock()
	if h.s.history.initialSlot < 0 {
		h.s.mu.Unlock()
		t.Fatal("broker exit reused an admitted output frame")
	}
	if bank := h.s.history.banks[h.s.history.current]; bank != nil && bank.n != 0 {
		h.s.mu.Unlock()
		t.Fatal("broker exit retained history cache")
	}
	h.s.mu.Unlock()
	if err := p.conn.Close(); err != nil {
		t.Fatal(err)
	}
	historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot < 0 })
}

func TestHistoryGeneratedInitialOutputPinnedAcrossStaleResponses(t *testing.T) {
	h := newHistoryTestHarness(t)
	p := h.peerWithChunk(3)
	p.send(t, `{"jsonrpc":"2.0","id":"caller","method":"routing:get-initial","params":{}}`)
	request := h.request(t)
	response := historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", ID: request.ID, Result: []byte(`{"requests":[]}`)})
	h.emit(t, response)
	var prefix [3]byte
	if err := p.conn.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadFull(p.conn, prefix[:]); err != nil {
		t.Fatal(err)
	}
	h.s.mu.Lock()
	slot := h.s.history.initialSlot
	original := append([]byte(nil), h.s.history.writes[slot].frame...)
	h.s.mu.Unlock()
	// A late result is discarded without completing the original borrow or
	// producing an extra reply. A following ordinary frame is the pump barrier.
	observer := h.peer()
	observer.send(t, `{"jsonrpc":"2.0","id":999,"method":"routing:get-initial","params":{}}`)
	if msg := observer.next(t); msg.Error == nil || msg.Error.Code != jsonrpc.HistoryBusyCode {
		t.Fatal("blocked output admitted another initial borrower")
	}
	h.emit(t, response)
	h.emit(t, []byte(`{"jsonrpc":"2.0","method":"test:barrier","params":{}}`))
	if msg := observer.next(t); msg.Method != "test:barrier" {
		t.Fatalf("pump barrier = %+v", msg)
	}
	h.s.mu.Lock()
	pinned := h.s.history.initialSlot == slot && bytes.Equal(original, h.s.history.writes[slot].frame)
	h.s.mu.Unlock()
	if !pinned {
		t.Fatal("late result released or overwrote the actual output borrower")
	}
	if err := p.conn.Close(); err != nil {
		t.Fatal(err)
	}
	historyAwait(t, func() bool { h.s.mu.Lock(); defer h.s.mu.Unlock(); return h.s.history.initialSlot < 0 })
	observer.send(t, `{"jsonrpc":"2.0","id":1000,"method":"routing:get-initial","params":{}}`)
	if msg := observer.next(t); msg.Error != nil || msg.ID == nil || string(*msg.ID) != "1000" {
		t.Fatal("actual output completion did not release initial admission")
	}
}

func TestHistoryMaximumInitialAndExactRawID(t *testing.T) {
	h := newHistoryTestHarness(t)
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","id":"\u0061<>&","method":"routing:get-initial","params":{}}`)
	request := h.request(t)
	initial := make([]byte, 0, 33554432)
	initial = append(initial, `{"requests":[`...)
	for i := 0; i < 256; i++ {
		if i > 0 {
			initial = append(initial, ',')
		}
		record := historyTestRecord(strconv.Itoa(i+1), 1)
		record.Requested.Model = strings.Repeat(`"`, 512)
		node, engine, model := strings.Repeat("n", 128), strings.Repeat("e", 128), record.Requested.Model
		for index := 0; index < 32; index++ {
			record.Attempts = append(record.Attempts, nodepolicy.RoutingHistoryAttempt{
				Index: int64(index), StartedAt: 1, Outcome: "pending",
				Destination: nodepolicy.RoutingHistoryDestination{Kind: "local", Model: &model, NodeID: &node, Engine: &engine},
			})
		}
		if err := record.Validate(); err != nil {
			t.Fatal(err)
		}
		raw := historyTestJSON(t, record)
		target := 131072
		if i == 255 {
			target = 33554432 - len(initial) - len("]}")
		}
		prefix := append(raw[:len(raw)-1], `,"unknown-synthetic-secret":"`...)
		padding := target - len(prefix) - len(`"}`)
		if padding < 0 {
			t.Fatalf("fixture exceeds record grant: %d", padding)
		}
		initial = append(initial, prefix...)
		initial = append(initial, bytes.Repeat([]byte{'s'}, padding)...)
		initial = append(initial, '"', '}')
	}
	initial = append(initial, ']', '}')
	if len(initial) != 33554432 {
		t.Fatalf("initial fixture = %d, want M", len(initial))
	}
	frame := append([]byte(`{"jsonrpc":"2.0","id":`), *request.ID...)
	frame = append(frame, `,"result":`...)
	frame = append(frame, initial...)
	frame = append(frame, '}')
	h.emit(t, frame)
	// Projecting and transferring the maximum payload can exceed the ordinary
	// peer deadline under the Windows race detector.
	msg := p.nextWithTimeout(t, 60*time.Second)
	if msg.Error != nil || msg.ID == nil || string(*msg.ID) != `"\u0061<>&"` {
		t.Fatalf("maximum initial refused or ID changed: %+v", msg.Error)
	}
	if len(msg.Result) <= 8<<20 {
		t.Fatalf("positive initial is only %d bytes; ordinary cap not exercised", len(msg.Result))
	}
	var result nodepolicy.RoutingHistoryInitial
	if err := json.Unmarshal(msg.Result, &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Requests) != 256 || result.Requests[0].Request.RequestID != "256" || result.Requests[255].Request.RequestID != "1" || len(result.Requests[0].Attempts) != 32 {
		t.Fatal("maximum initial data/rank was not preserved")
	}
	if bytes.Contains(p.raw, []byte("unknown-synthetic-secret")) {
		t.Fatal("unknown source field escaped projection")
	}
	if h.kills.Load() != 0 {
		t.Fatal("legal maximum restarted broker")
	}
}

func TestHistoryIncomingRequestRefusals(t *testing.T) {
	var logs historyTestLog
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	bounded := `"` + strings.Repeat(`\u0061`, 85) + `"`
	over := `"` + strings.Repeat(`\u0061`, 85) + `a"`
	test := func(name, id, method, params, wantID string, code int, message string) {
		t.Run(name, func(t *testing.T) {
			h := newHistoryTestHarness(t)
			p := h.peer()
			p.send(t, `{"jsonrpc":"2.0","id":`+id+`,"method":"`+method+`","params":`+params+`}`)
			msg := p.next(t)
			var envelope struct {
				ID json.RawMessage `json:"id"`
			}
			if err := json.Unmarshal(p.raw, &envelope); err != nil {
				t.Fatal(err)
			}
			if string(envelope.ID) != wantID || msg.Error == nil || msg.Error.Code != code || msg.Error.Message != message {
				t.Fatalf("refusal = %s, want id %s and fixed code %d", p.raw, wantID, code)
			}
			if len(p.raw) > 4096 || bytes.Contains(p.raw, []byte("synthetic-secret")) {
				t.Fatal("refusal exceeded E or echoed input")
			}
			h.s.mu.Lock()
			if h.s.history.initialSlot >= 0 || h.s.history.initial != nil || h.s.history.originalN != 0 {
				t.Error("refusal acquired initial storage/work")
			}
			for _, bank := range h.s.history.banks {
				if bank != nil {
					t.Error("refusal allocated a cache bank")
				}
			}
			for c := range h.s.clients {
				if c.history.flags != 0 {
					t.Error("refusal changed subscription/lease state")
				}
			}
			h.s.mu.Unlock()
			select {
			case <-h.input.requests:
				t.Fatal("refusal forwarded to broker")
			default:
			}
			p.send(t, `{"jsonrpc":"2.0","id":"barrier","method":"service/status"}`)
			if next := p.next(t); next.ID == nil || string(*next.ID) != `"barrier"` || next.Error != nil {
				t.Fatal("ordinary RPC failed after refusal")
			}
		})
	}
	test("513 raw bytes", `"`+strings.Repeat("x", 511)+`"`, "routing:subscribe", `{}`, "null", -32062, "history-invalid")
	test("escaped 513 bytes", over, "routing:get-initial", `{}`, "null", -32062, "history-invalid")
	test("bounded private call", bounded, "routing/request", `{"synthetic-secret":1}`, bounded, -32601, "method not found")
	test("oversized private call", over, "routing/history-window", `{}`, "null", -32062, "history-invalid")
	test("invalid public params", `"\u0061<>&"`, "routing:subscribe", `null`, `"\u0061<>&"`, -32062, "history-invalid")
	test("array public params", `"caller"`, "routing:get-initial", `[]`, `"caller"`, -32062, "history-invalid")
	if !strings.Contains(logs.String(), "client attached") || strings.Contains(logs.String(), "synthetic-secret") {
		t.Fatal("request-refusal log capture missing positive control or leaked raw data")
	}
}

func TestHistoryRefusalDoesNotAttributeDuplicatedRequest(t *testing.T) {
	h := newHistoryTestHarness(t)
	p := h.peer()
	for _, frame := range []string{
		`{"jsonrpc":"2.0","id":1,"id":2,"method":"routing:subscribe","params":null}`,
		`{"jsonrpc":"2.0","id":1,"method":"routing/request","params":{"unknown":{"a":1,"a":2}}}`,
		`{"jsonrpc":"2.0","method":"routing:changed","params":null}`,
	} {
		p.send(t, frame)
	}
	p.send(t, `{"jsonrpc":"2.0","id":9,"method":"service/status"}`)
	if msg := p.next(t); msg.ID == nil || string(*msg.ID) != "9" || msg.Error != nil {
		t.Fatal("ambiguous/notification input acquired refusal attribution")
	}
}

func TestHistoryBrokerRequestIDCollisionPreservesPending(t *testing.T) {
	test := func(name, method, params string, code int) {
		t.Run(name, func(t *testing.T) {
			h := newHistoryTestHarness(t)
			p := h.peer()
			p.send(t, `{"jsonrpc":"2.0","id":"caller","method":"routing:get-initial","params":{}}`)
			pending := h.request(t)
			h.emit(t, []byte(`{"jsonrpc":"2.0","id":`+string(*pending.ID)+`,"method":"`+method+`","params":`+params+`}`))
			refusal := h.request(t)
			if refusal.ID == nil || string(*refusal.ID) != string(*pending.ID) || refusal.Error == nil || refusal.Error.Code != code {
				t.Fatalf("incoming request refusal = %+v", refusal)
			}
			h.emit(t, historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", ID: pending.ID, Result: []byte(`{"requests":[]}`)}))
			if reply := p.next(t); reply.ID == nil || string(*reply.ID) != `"caller"` || reply.Error != nil {
				t.Fatal("incoming request settled unrelated outbound pending call")
			}
		})
	}
	test("private request", "routing/request", `{}`, -32601)
	test("invalid public request", "routing:subscribe", `null`, -32062)
	test("valid incoming request", "routing:subscribe", `{}`, -32601)
	test("ordinary incoming collision", "echo", `{}`, -32601)
}

func TestHistoryLegalEscapedCallerID(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.seed(t, historyTestRecord("1", 1))
	p := h.peer()
	id := `"` + strings.Repeat(`\u0061`, 85) + `"`
	p.send(t, `{"jsonrpc":"2.0","id":`+id+`,"method":"routing:get-initial","params":{}}`)
	if reply := p.next(t); reply.ID == nil || string(*reply.ID) != id || reply.Error != nil {
		t.Fatal("legal escaped 512-byte ID was refused or rewritten")
	}
}

type historyTestFailControl struct {
	delegate    io.WriteCloser
	once        sync.Once
	transmitted int
}

func (w *historyTestFailControl) Write(raw []byte) (int, error) {
	failed := false
	w.once.Do(func() { failed = true })
	if failed {
		return min(w.transmitted, len(raw)), errors.New("synthetic control write failure")
	}
	return w.delegate.Write(raw)
}

func (w *historyTestFailControl) Close() error { return w.delegate.Close() }

func TestHistoryZeroProgressControlFailureRecovers(t *testing.T) {
	h := newHistoryTestHarness(t)
	h.run.proc.stdin = &historyTestFailControl{delegate: h.input}
	p := h.peer()
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:get-initial","params":{}}`)
	if reply := p.next(t); reply.Error == nil || reply.Error.Code != -32062 {
		t.Fatal("failed control write was not answered")
	}
	p.send(t, `{"jsonrpc":"2.0","id":2,"method":"echo","params":{}}`)
	echo := h.request(t)
	h.emit(t, historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", ID: echo.ID, Result: []byte(`{}`)}))
	if reply := p.next(t); reply.Error != nil || string(*reply.ID) != "2" {
		t.Fatal("ordinary relay did not survive zero-progress failure")
	}
	p.send(t, `{"jsonrpc":"2.0","id":3,"method":"routing:get-initial","params":{}}`)
	request := h.request(t)
	h.emit(t, historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", ID: request.ID, Result: []byte(`{"requests":[]}`)}))
	if reply := p.next(t); reply.Error != nil || string(*reply.ID) != "3" {
		t.Fatal("history remained busy after zero-progress failure")
	}
	if h.kills.Load() != 0 {
		t.Fatal("zero-progress failure killed a surviving broker")
	}
}

func TestHistoryZeroProgressRefetchFailureReleasesCorrelation(t *testing.T) {
	h := historyGeneratedFixture(t)
	h.run.proc.stdin = &historyTestFailControl{delegate: h.input}
	h.s.mu.Lock()
	h.s.history.refetch = true
	h.s.mu.Unlock()
	h.s.historyRefetch()
	h.s.mu.Lock()
	defer h.s.mu.Unlock()
	if !h.s.history.refetch || h.s.history.initialSlot >= 0 {
		t.Fatal("refetch failure did not release/requeue its owner")
	}
	if _, err := h.s.history.correlation.BeginGeneratedMethod(nodepolicy.MethodRoutingSubscribe); err != nil {
		t.Fatalf("zero-progress refetch left retired correlation: %v", err)
	}
	h.s.history.correlation.Complete()
}

func TestHistoryTransmittedControlFailureRetiresTransport(t *testing.T) {
	test := func(name string, transmitted int) {
		t.Run(name, func(t *testing.T) {
			h := newHistoryTestHarness(t)
			h.run.proc.stdin = &historyTestFailControl{delegate: h.input, transmitted: transmitted}
			p := h.peer()
			p.send(t, `{"jsonrpc":"2.0","id":1,"method":"routing:get-initial","params":{}}`)
			if reply := p.next(t); reply.Error == nil {
				t.Fatal("failed control was not answered")
			}
			historyAwait(t, func() bool { return h.kills.Load() >= 1 })
			t.Logf("actual failed transport kill requests: %d", h.kills.Load())
			h.s.mu.Lock()
			broken := h.s.history.controlBroken
			h.s.history.refetch = true
			h.s.mu.Unlock()
			h.s.historyRefetch()
			if !broken {
				t.Fatal("possibly transmitted failed frame did not fence control admission")
			}
			select {
			case <-h.input.requests:
				t.Fatal("possibly transmitted failed transport admitted another control")
			default:
			}
			h.s.brokerExited(h.run)
			h.s.mu.Lock()
			_, err := h.s.history.correlation.BeginGeneratedMethod(nodepolicy.MethodRoutingSubscribe)
			if err == nil {
				h.s.history.correlation.Complete()
			}
			h.s.mu.Unlock()
			if err != nil {
				t.Fatal("actual transport retirement did not release classification")
			}
		})
	}
	test("partial write", 3)
	test("whole frame with error", 4096)
}

type historyTestReadSignal struct {
	reader  io.Reader
	entered chan struct{}
	once    sync.Once
	reads   atomic.Int32
}

func (r *historyTestReadSignal) Read(dst []byte) (int, error) {
	r.reads.Add(1)
	r.once.Do(func() { close(r.entered) })
	return r.reader.Read(dst)
}

type historyTestStalledOutput struct {
	entered chan struct{}
	release chan struct{}
	frame   []byte
	once    sync.Once
}

func (w *historyTestStalledOutput) unblock() { w.once.Do(func() { close(w.release) }) }

func (w *historyTestStalledOutput) Write(raw []byte) (int, error) {
	close(w.entered)
	<-w.release
	w.frame = append([]byte(nil), raw...)
	return len(raw), nil
}

func (w *historyTestStalledOutput) Close() error { return nil }

func historyWaitClosed(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatal("history owner barrier did not complete")
	}
}

func historyReadRun(input io.WriteCloser, output io.Reader) *brokerRun {
	return &brokerRun{proc: &brokerProc{stdin: input, stdout: output, kill: func() error { return nil }}, exited: make(chan struct{})}
}

func historyInstallRun(s *Service, run *brokerRun) {
	s.mu.Lock()
	s.broker = run
	s.mu.Unlock()
}

func historyStartPump(s *Service, run *brokerRun) <-chan struct{} {
	done := make(chan struct{})
	go func() { defer close(done); s.pumpStdout(run) }()
	return done
}

func historyChangedFrame(t *testing.T, id string) []byte {
	t.Helper()
	return append(historyTestJSON(t, &jsonrpc.Message{JSONRPC: "2.0", Method: "routing:changed", Params: historyTestJSON(t, historyTestRecord(id, 1))}), '\n')
}

func TestHistoryReadLeaseDoesNotJoinOldOrdinaryOutputs(t *testing.T) {
	s := newServiceWithHistory(config{})
	var stopped []*historyTestStalledOutput
	var pumps []<-chan struct{}
	t.Cleanup(func() {
		for _, w := range stopped {
			w.unblock()
		}
		for _, done := range pumps {
			historyWaitClosed(t, done)
		}
	})
	var backing *byte
	for i := 0; i < 4; i++ {
		if s.broker != nil {
			s.brokerExited(s.broker)
		}
		writer := &historyTestStalledOutput{entered: make(chan struct{}), release: make(chan struct{})}
		stopped = append(stopped, writer)
		id := strconv.Itoa(i + 1)
		// Both frames fit one bufio read. The second must never be consumed
		// after the ordinary callback resumes on this retired generation.
		method := "ordinary:callback"
		if i%2 != 0 {
			method = "routing/request"
		}
		frames := []byte(`{"jsonrpc":"2.0","id":"old-` + id + `","method":"` + method + `","params":{"owned":"old"}}` + "\n")
		frames = append(frames, historyChangedFrame(t, "999")...)
		old := historyReadRun(writer, bytes.NewReader(frames))
		historyInstallRun(s, old)
		pumps = append(pumps, historyStartPump(s, old))
		historyWaitClosed(t, writer.entered)
		s.brokerExited(old)
		fresh := historyReadRun(&historyTestInput{requests: make(chan []byte, 1)}, bytes.NewReader(historyChangedFrame(t, strconv.Itoa(100+i))))
		historyInstallRun(s, fresh)
		historyWaitClosed(t, historyStartPump(s, fresh))
		s.history.rawMu.Lock()
		if cap(s.history.raw) != 33558528 {
			t.Error("raw arena capacity changed")
		}
		if backing == nil {
			backing = &s.history.raw[:cap(s.history.raw)][0]
		} else if backing != &s.history.raw[:cap(s.history.raw)][0] {
			t.Error("replacement allocated another H")
		}
		s.history.rawMu.Unlock()
	}
	// All four ordinary writes remain stalled while replacement generations
	// have already reused the one arena and committed their own history.
	for i, w := range stopped {
		w.unblock()
		historyWaitClosed(t, pumps[i])
		var reply jsonrpc.Message
		if err := json.Unmarshal(w.frame, &reply); err != nil {
			t.Fatal(err)
		}
		if reply.ID == nil || string(*reply.ID) != `"old-`+strconv.Itoa(i+1)+`"` || reply.Error == nil {
			t.Fatal("old ordinary message borrowed reused raw data")
		}
	}
	bank := s.history.banks[s.history.current]
	if bank.n != 1 || string(bank.cells[0].id[:bank.cells[0].lengths[2]]) != "103" {
		t.Fatal("retired buffered next frame changed current history")
	}
}

func TestHistoryReadLeaseRetiresAfterActualPipeInterruption(t *testing.T) {
	s := newServiceWithHistory(config{})
	reader, writer := io.Pipe()
	defer writer.Close()
	defer reader.Close()
	signal := &historyTestReadSignal{reader: reader, entered: make(chan struct{})}
	old := historyReadRun(&historyTestInput{requests: make(chan []byte, 1)}, signal)
	historyInstallRun(s, old)
	oldDone := historyStartPump(s, old)
	var freshDone <-chan struct{}
	var waiterDone <-chan struct{}
	t.Cleanup(func() {
		_ = reader.Close()
		historyWaitClosed(t, oldDone)
		if waiterDone != nil {
			historyWaitClosed(t, waiterDone)
		}
		if freshDone != nil {
			historyWaitClosed(t, freshDone)
		}
	})
	historyWaitClosed(t, signal.entered)
	if s.history.rawMu.TryLock() {
		s.history.rawMu.Unlock()
		t.Fatal("active Read did not retain H")
	}
	waiter := make(chan struct{})
	waiterDone = waiter
	waiterErr := make(chan error, 1)
	go func() {
		defer close(waiter)
		_, _, _, err := s.readHistoryMessage(old, bufio.NewReader(old.proc.stdout))
		waiterErr <- err
	}()
	s.brokerExited(old)
	freshSignal := &historyTestReadSignal{reader: bytes.NewReader(historyChangedFrame(t, "2")), entered: make(chan struct{})}
	fresh := historyReadRun(&historyTestInput{requests: make(chan []byte, 1)}, freshSignal)
	historyInstallRun(s, fresh)
	freshDone = historyStartPump(s, fresh)
	if s.history.rawMu.TryLock() {
		s.history.rawMu.Unlock()
		t.Fatal("withdrawal reused active read backing")
	}
	// Only interruption by the actual old transport owner can finish its Read.
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	_ = reader.Close() // duplicate retirement does not authorize a second reuse
	historyWaitClosed(t, oldDone)
	historyWaitClosed(t, waiterDone)
	if err := <-waiterErr; !errors.Is(err, io.EOF) {
		t.Fatalf("retired waiting read: %v", err)
	}
	historyWaitClosed(t, freshSignal.entered)
	historyWaitClosed(t, freshDone)
	bank := s.history.banks[s.history.current]
	if bank.n != 1 || string(bank.cells[0].id[:bank.cells[0].lengths[2]]) != "2" {
		t.Fatal("replacement did not progress after actual read completion")
	}
	if signal.reads.Load() != 1 {
		t.Fatal("retired waiter read old transport after obtaining the arena")
	}
}

func TestHistoryReadLeasePinsProjectedApply(t *testing.T) {
	s := newServiceWithHistory(config{})
	entered, release := make(chan struct{}), make(chan struct{})
	integer := serviceHistoryRecordSchema.Integer
	var once sync.Once
	var releaseOnce sync.Once
	var done, withdrawn chan struct{}
	t.Cleanup(func() {
		releaseOnce.Do(func() { close(release) })
		if done != nil {
			historyWaitClosed(t, done)
		}
		if withdrawn != nil {
			historyWaitClosed(t, withdrawn)
		}
		serviceHistoryRecordSchema.Integer = integer
	})
	serviceHistoryRecordSchema.Integer = func(raw []byte) (int64, bool) { once.Do(func() { close(entered); <-release }); return integer(raw) }
	old := historyReadRun(&historyTestInput{requests: make(chan []byte, 1)}, bytes.NewReader(historyChangedFrame(t, "1")))
	historyInstallRun(s, old)
	done = make(chan struct{})
	readErr := make(chan error, 1)
	go func() {
		defer close(done)
		_, _, _, err := s.readHistoryMessage(old, bufio.NewReader(old.proc.stdout))
		readErr <- err
	}()
	historyWaitClosed(t, entered)
	if s.history.rawMu.TryLock() {
		s.history.rawMu.Unlock()
		t.Fatal("projected apply released raw lease early")
	}
	withdrawn = make(chan struct{})
	go func() { s.brokerExited(old); close(withdrawn) }()
	releaseOnce.Do(func() { close(release) })
	historyWaitClosed(t, done)
	if err := <-readErr; err != nil {
		t.Fatalf("projected read: %v", err)
	}
	historyWaitClosed(t, withdrawn)
	serviceHistoryRecordSchema.Integer = integer
	fresh := historyReadRun(&historyTestInput{requests: make(chan []byte, 1)}, bytes.NewReader(historyChangedFrame(t, "2")))
	historyInstallRun(s, fresh)
	historyWaitClosed(t, historyStartPump(s, fresh))
	bank := s.history.banks[s.history.current]
	if bank.n != 1 || string(bank.cells[0].id[:bank.cells[0].lengths[2]]) != "2" {
		t.Fatal("projected old borrower escaped its generation fence")
	}
}

func TestHistoryOrdinaryMessageOwnsDataAfterReadLease(t *testing.T) {
	test := func(name, frame, want string, request bool) {
		t.Run(name, func(t *testing.T) {
			s := newServiceWithHistory(config{})
			old := historyReadRun(&historyTestInput{requests: make(chan []byte, 1)}, strings.NewReader(frame+"\n"))
			historyInstallRun(s, old)
			msg, _, reply, err := s.readHistoryMessage(old, bufio.NewReader(old.proc.stdout))
			if err != nil || msg == nil || reply {
				t.Fatalf("ordinary read: %v", err)
			}
			s.brokerExited(old)
			fresh := historyReadRun(&historyTestInput{requests: make(chan []byte, 1)}, bytes.NewReader(historyChangedFrame(t, "2")))
			historyInstallRun(s, fresh)
			historyWaitClosed(t, historyStartPump(s, fresh))
			data := msg.Result
			if request {
				data = msg.Params
			}
			if msg.ID == nil || string(*msg.ID) != `"owned"` || string(data) != want {
				t.Fatal("ordinary decoded data borrowed overwritten H")
			}
		})
	}
	test("response", `{"jsonrpc":"2.0","id":"owned","result":{"marker":"owned-response"}}`, `{"marker":"owned-response"}`, false)
	test("request", `{"jsonrpc":"2.0","id":"owned","method":"ordinary","params":{"marker":"owned-request"}}`, `{"marker":"owned-request"}`, true)
}
