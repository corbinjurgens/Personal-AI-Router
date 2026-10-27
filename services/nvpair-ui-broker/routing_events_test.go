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
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"
	"unsafe"

	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/pipeio"
)

func TestRoutingHistoryMainAssemblyStartsBeforeSupervisionAndJoinsFailure(t *testing.T) {
	client, server := net.Pipe()
	t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
	lifetime := newBrokerMainLifetime()
	t.Cleanup(lifetime.cancel)
	dir := t.TempDir()
	b, err := newMainBroker(lifetime, workerPaths{scanner: filepath.Join(dir, "missing-scanner"), clusterDir: filepath.Join(dir, "cluster")}, server, true)
	if err != nil {
		t.Fatal(err)
	}
	if b.routingHistory == nil || b.routingHistory.lifetime != lifetime || b.codec == nil || !b.codec.ReadRetirementEnabled() {
		t.Fatal("main assembly omitted history or its guarded local codec")
	}
	drained := make(chan struct{})
	go func() { defer close(drained); _, _ = io.Copy(io.Discard, client) }()
	if err := b.serveMainLifetime(lifetime); err == nil {
		t.Fatal("missing scanner unexpectedly started")
	}
	historyAwaitReadSignal(t, drained)
	if !b.routingHistory.started || b.routingHistory.parent.Err() != context.Canceled {
		t.Fatal("history did not start before supervision or join its failed startup")
	}
}

func TestRoutingHistoryMainAssemblyClosesFailedTransport(t *testing.T) {
	client, server := net.Pipe()
	t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
	lifetime := newBrokerMainLifetime()
	t.Cleanup(lifetime.cancel)
	// A stdio role requires the captured input/output owners, not a connection
	// silently treated as ordinary JSON-RPC when the guard cannot attach.
	b, err := newMainBroker(lifetime, workerPaths{clusterDir: t.TempDir()}, server, false)
	if !errors.Is(err, jsonrpc.ErrHistoryInvalid) || b != nil || lifetime.root.Err() == nil {
		t.Fatalf("failed assembly = %v / %v, parent = %v", b, err, lifetime.root.Err())
	}
	if _, err := client.Read(make([]byte, 1)); err != io.EOF {
		t.Fatalf("failed assembly retained transport: %v", err)
	}
}

type historyMainTransport struct {
	io.Reader
	*historyHeldOutput
	closed chan struct{}
	once   sync.Once
}

func (w *historyMainTransport) Close() error {
	w.once.Do(func() { close(w.closed) })
	return nil
}

func TestRoutingHistoryMainAssemblyJoinsBlockedInitialOnFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := &historyHeldOutput{entered: make(chan struct{}), finish: make(chan struct{})}
		var release sync.Once
		finish := func() { release.Do(func() { close(w.finish) }) }
		defer finish()
		transport := &historyMainTransport{Reader: strings.NewReader(""), historyHeldOutput: w, closed: make(chan struct{})}
		lifetime := newBrokerMainLifetime()
		defer lifetime.cancel()
		dir := t.TempDir()
		b, err := newMainBroker(lifetime, workerPaths{scanner: filepath.Join(dir, "missing-scanner"), clusterDir: filepath.Join(dir, "cluster")}, transport, true)
		if err != nil {
			t.Fatal(err)
		}
		b.handleMessage(routingBrokerRequest(`"initial"`, nodepolicy.MethodRoutingGetInitial, `{}`))
		<-w.entered
		done := make(chan error, 1)
		go func() { done <- b.serveMainLifetime(lifetime) }()
		<-transport.closed
		synctest.Wait()
		select {
		case err := <-done:
			t.Fatalf("Serve returned before the initial Write actually completed: %v", err)
		default:
		}
		if !b.routingHistory.initialBusy.Load() {
			t.Fatal("cancellation released the frozen initial before write completion")
		}
		finish()
		if err := <-done; err == nil {
			t.Fatal("missing scanner unexpectedly started")
		}
		if b.routingHistory.initialBusy.Load() {
			t.Fatal("joined initial retained admission")
		}
		historyRequireReadDetached(t, b.codec)
	})
}

// Hold the required reply's actual Write borrower even after closing the
// underlying connection has interrupted its transport operation.
type historyRequiredReplyTransport struct {
	net.Conn
	entered, closed, interrupted, completed, finish        chan struct{}
	ordinaryEntered, allowOrdinary, replyReady, allowReply chan struct{}
	borrowed                                               []byte
	closeOnce                                              sync.Once
}

func (w *historyRequiredReplyTransport) Write(raw []byte) (int, error) {
	var msg Message
	if err := json.Unmarshal(raw, &msg); err != nil {
		return 0, err
	}
	if msg.Error == nil {
		if msg.Method == "test:ordinary-before-refusal" {
			close(w.ordinaryEntered)
			<-w.allowOrdinary
		}
		return w.Conn.Write(raw)
	}
	// Keep the required frame out of the pipe until the ordinary-frame reader
	// has actually stopped. This barrier holds the codec gate, so no ordinary
	// writer can slip between that handoff and the required Write below.
	close(w.replyReady)
	<-w.allowReply
	w.borrowed = raw
	close(w.entered)
	n, err := w.Conn.Write(raw)
	close(w.interrupted)
	<-w.finish
	close(w.completed) // final application access of the actual Write borrower
	return n, err
}

func (w *historyRequiredReplyTransport) Close() error {
	err := w.Conn.Close()
	w.closeOnce.Do(func() { close(w.closed) })
	return err
}

func TestRoutingHistoryMainAssemblyInterruptsRequiredReplyOnCancellation(t *testing.T) {
	test := func(name string, cancelRoot bool) {
		t.Run(name, func(t *testing.T) {
			isolateOllamaHostTestConfig(t)
			scanner := buildServicePortFixture(t)
			client, server := net.Pipe()
			transport := &historyRequiredReplyTransport{
				Conn: server, entered: make(chan struct{}), closed: make(chan struct{}),
				interrupted: make(chan struct{}), completed: make(chan struct{}), finish: make(chan struct{}),
				ordinaryEntered: make(chan struct{}), allowOrdinary: make(chan struct{}), replyReady: make(chan struct{}), allowReply: make(chan struct{}),
			}
			var finishOnce sync.Once
			finish := func() { finishOnce.Do(func() { close(transport.finish) }) }
			var allowReplyOnce sync.Once
			allowReply := func() { allowReplyOnce.Do(func() { close(transport.allowReply) }) }
			var allowOrdinaryOnce sync.Once
			allowOrdinary := func() { allowOrdinaryOnce.Do(func() { close(transport.allowOrdinary) }) }
			lifetime := newBrokerMainLifetime()
			dir := t.TempDir()
			b, err := newMainBroker(lifetime, workerPaths{scanner: scanner, clusterDir: filepath.Join(dir, "cluster")}, transport, true)
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() { done <- b.serveMainLifetime(lifetime) }()
			joined := false
			var ordinaryDrained chan struct{}
			t.Cleanup(func() {
				allowOrdinary()
				allowReply()
				finish()
				lifetime.cancel()
				_ = client.Close()
				_ = server.Close()
				if ordinaryDrained != nil {
					select {
					case <-ordinaryDrained:
					case <-time.After(5 * time.Second):
						t.Error("ordinary-frame reader did not join during cleanup")
					}
				}
				if !joined {
					select {
					case <-done:
					case <-time.After(5 * time.Second):
						t.Error("production Serve did not join during cleanup")
					}
				}
			})
			if err := client.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
				t.Fatal(err)
			}
			peer := NewCodec(client)
			for {
				msg, err := peer.Read()
				if err != nil {
					t.Fatalf("production app:ready: %v", err)
				}
				if msg.Method == "app:ready" {
					break
				}
			}
			ordinaryDrained = make(chan struct{})
			var drainErr error
			go func() {
				defer close(ordinaryDrained)
				for {
					if _, err := peer.Read(); err != nil {
						drainErr = err
						return
					}
				}
			}()
			// app:ready does not fence the startup policy announcement. Force an
			// ordinary frame to own the gate before the refusal, independent of
			// whether that announcement happens to win this run's scheduling.
			ordinary := make(chan error, 1)
			go func() { ordinary <- b.codec.Notify("test:ordinary-before-refusal", nil) }()
			historyAwaitReadSignal(t, transport.ordinaryEntered)
			// The raw guard supplies the required refusal and its borrowed ID.
			// No initial writer is involved.
			if _, err := io.WriteString(client, `{"jsonrpc":"2.0","id":"required\u0031","method":"routing:subscribe","params":null}`+"\n"); err != nil {
				t.Fatal(err)
			}
			allowOrdinary()
			historyAwaitReadSignal(t, transport.replyReady)
			// Only interrupt the client's read side, then join the actual reader
			// before admitting the required frame. The real connection remains
			// open and the client never reads any byte of the required reply.
			if err := client.SetReadDeadline(time.Now()); err != nil {
				t.Fatal(err)
			}
			historyAwaitReadSignal(t, ordinaryDrained)
			if !errors.Is(drainErr, os.ErrDeadlineExceeded) {
				t.Fatalf("ordinary-frame reader stopped before handoff: %v", drainErr)
			}
			select {
			case err := <-ordinary:
				if err != nil {
					t.Fatalf("competing ordinary frame: %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("competing ordinary frame did not complete")
			}
			allowReply()
			historyAwaitReadSignal(t, transport.entered)
			pinned := bytes.Clone(transport.borrowed)
			var refusal Message
			if err := json.Unmarshal(pinned, &refusal); err != nil {
				t.Fatal(err)
			}
			if refusal.ID == nil || string(*refusal.ID) != `"required\u0031"` || refusal.Error == nil || refusal.Error.Code != jsonrpc.HistoryInvalidCode {
				t.Fatalf("required local refusal = %+v", refusal)
			}
			if cancelRoot {
				lifetime.cancel()
			} else {
				b.cancel() // The actual Serve cancellation owner also cancels history.
			}
			historyAwaitReadSignal(t, transport.closed)
			historyAwaitReadSignal(t, transport.interrupted)
			if b.routingHistory.parent.Err() != context.Canceled {
				t.Fatal("production shutdown did not cancel the history branch")
			}
			if !cancelRoot && lifetime.root.Err() != nil {
				t.Fatal("Serve cancellation changed the separate main lifetime")
			}
			select {
			case err := <-done:
				joined = true
				t.Fatalf("Serve returned before the required Write actually completed: %v", err)
			default:
			}
			select {
			case <-transport.completed:
				t.Fatal("required Write completed before its actual borrower was released")
			default:
			}
			if !bytes.Equal(pinned, transport.borrowed) {
				t.Fatal("cancellation reused the required reply frame")
			}
			codec := b.codec
			historyRequireReadPending(t, codec.RetireRead())
			readers := &b.routingHistory.readers
			backing := reflect.ValueOf(readers.slots[historyLocalControl].raw).Pointer()
			if _, err := readers.acquire(historyLocalControl, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
				t.Fatalf("required reply borrower released its read bank early: %v", err)
			}
			finish()
			select {
			case err := <-done:
				joined = true
				if err != nil {
					t.Fatalf("production Serve cancellation: %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("production Serve did not join after actual Write completion")
			}
			historyAwaitReadSignal(t, transport.completed)
			historyRequireReadDetached(t, codec)
			replacement, err := readers.acquire(historyLocalControl, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil)
			if err != nil {
				t.Fatalf("completed required reply retained its read bank: %v", err)
			}
			t.Cleanup(func() { replacement.RetireRead() })
			if reflect.ValueOf(readers.slots[historyLocalControl].raw).Pointer() != backing {
				t.Fatal("replacement allocated another read bank")
			}
			historyRequireReadDetached(t, codec)
		})
	}
	test("root lifetime cancellation", true)
	test("Serve cancellation", false)
}

func TestRoutingHistoryPolicyAcknowledgementBindsAtDispatch(t *testing.T) {
	b, _ := newPolicyBroker(t)
	lifetime := newBrokerMainLifetime()
	t.Cleanup(lifetime.cancel)
	var err error
	b.routingHistory, err = newBrokerRoutingHistory(b, lifetime)
	if err != nil {
		t.Fatal(err)
	}
	pp, _, acks := attachAckingProxy(t, b)
	if !b.replayPolicyToProxy(t.Context(), pp) {
		t.Fatal("policy replay failed")
	}
	b.workersMu.Lock()
	bound := b.historySourceLocked(pp) && b.historyRun == acks.runID
	b.workersMu.Unlock()
	if !bound {
		t.Fatal("actual policy acknowledgement did not bind the published source")
	}
	row := historyRecord("9", 10, 1)
	row.Request.ProxyRunID = acks.runID
	// A cycle dropped before binding must be accepted on the exact same
	// generation replay, even when the repeated policy ACK changes no state.
	b.workersMu.Lock()
	b.historyOrigin, b.historyRun = "", ""
	b.workersMu.Unlock()
	historySend(t, b, pp, "1", row)
	historyRequireRows(t, b)
	if changed := b.sendPolicyStateLocked(t.Context(), pp, pushPolicy); changed {
		t.Fatal("repeated valid policy acknowledgement unexpectedly changed activation")
	}
	historySend(t, b, pp, "1", row)
	historyRequireRows(t, b, row)
	// Lost accepted reply: replay is idempotent and schedules an exact-source
	// acknowledgement again, without another accepted record/version.
	version := b.routingHistory.version
	b.routingHistory.ackSource, b.routingHistory.ackGeneration = nil, ""
	historySend(t, b, pp, "1", row)
	if b.routingHistory.version != version || b.routingHistory.ackSource != pp || b.routingHistory.ackGeneration != "1" {
		t.Fatal("same-generation replay changed accepted rows or lost its acknowledgement")
	}
	oldAck := historyMarshal(t, nodepolicy.PolicyAck{OriginNodeID: b.nodeID, RunID: acks.runID, Revision: b.nodePolicy.current.Revision})
	fresh := historyActivate(t, b, "replacement-without-requests")
	b.recordActivation(pp, "1", oldAck, nil, nil)
	if b.bindRoutingHistoryActivation(pp) {
		t.Fatal("old policy callback rebound a departed process")
	}
	historySend(t, b, pp, "2", row)
	b.routingHistory.mu.Lock()
	b.routingHistory.retireLocked()
	b.routingHistory.mu.Unlock()
	rows := historyRows(t, b)
	if len(rows) != 1 || rows[0].State != "failed" || rows[0].Reason != "proxy-exited" || b.proxyProcessHandle() != fresh {
		t.Fatalf("empty replacement failed departure retirement: %+v", rows)
	}
}

func historyFixture(t *testing.T) (*Broker, *proxyProcess, *syncBuffer) {
	t.Helper()
	out := &syncBuffer{}
	b := &Broker{codec: NewCodec(out), nodeID: "self-node"}
	lifetime := newBrokerMainLifetime()
	t.Cleanup(lifetime.cancel)
	var err error
	b.routingHistory, err = newBrokerRoutingHistory(b, lifetime)
	if err != nil {
		t.Fatal(err)
	}
	p := historyActivate(t, b, "run-a")
	return b, p, out
}

func TestRoutingHistoryPrivateLifetimeFanout(t *testing.T) {
	test := func(name string, external bool) {
		t.Run(name, func(t *testing.T) {
			life := newBrokerMainLifetime()
			defer life.cancel()
			b := &Broker{}
			h, err := newBrokerRoutingHistory(b, life)
			if err != nil {
				t.Fatal(err)
			}
			b.routingHistory = h
			serve, stop, err := b.newServeContext(nil, life)
			if err != nil {
				t.Fatal(err)
			}
			defer stop()
			if reflect.ValueOf(life.root).Elem().FieldByName("children").Len() != 2 {
				t.Fatal("main lifetime does not own exactly Serve and history")
			}
			ordinary, stopOrdinary := context.WithCancel(serve)
			defer stopOrdinary()
			if reflect.ValueOf(h.parent).Elem().FieldByName("children").Len() != 0 {
				t.Fatal("ordinary registration entered history parent")
			}
			if _, err := newBrokerRoutingHistory(b, life); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
				t.Fatalf("duplicate history lifetime = %v", err)
			}
			if _, _, err := b.newServeContext(nil, life); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
				t.Fatalf("duplicate Serve lifetime = %v", err)
			}
			if external {
				life.cancel()
			} else {
				stop()
			}
			if serve.Err() != context.Canceled || h.parent.Err() != context.Canceled || ordinary.Err() != context.Canceled {
				t.Fatal("cancellation did not fan out to both lifetimes and ordinary work")
			}
		})
	}
	test("external main cancellation", true)
	test("internal Serve cancellation", false)
}

func TestRoutingHistoryGenericServeContextPreservesParent(t *testing.T) {
	type contextKey struct{}
	parent, cancel := context.WithDeadline(context.WithValue(t.Context(), contextKey{}, "preserved"), time.Now().Add(time.Hour))
	defer cancel()
	b := &Broker{}
	serve, stop, err := b.newServeContext(parent, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	wantDeadline, wantOK := parent.Deadline()
	gotDeadline, gotOK := serve.Deadline()
	if serve.Value(contextKey{}) != "preserved" || gotOK != wantOK || gotDeadline != wantDeadline {
		t.Fatal("generic Serve lost parent values or deadline")
	}
	cancel()
	if serve.Err() != context.Canceled {
		t.Fatal("generic Serve lost parent cancellation")
	}
}

func TestRoutingHistoryLocalRetirementCallbackFinalAccess(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, _, _ := historyFixture(t)
		h := b.routingHistory
		ctx, cancel, err := b.newServeContext(nil, h.lifetime)
		if err != nil {
			t.Fatal(err)
		}
		defer cancel()
		input, feed := io.Pipe()
		t.Cleanup(func() { _ = input.Close(); _ = feed.Close() })
		codec, err := h.readers.acquire(historyLocalControl, readWriter{input, io.Discard}, func() { _ = input.Close() }, nil)
		if err != nil {
			t.Fatal(err)
		}
		b.codec = codec
		entered, release := make(chan struct{}), make(chan struct{})
		var once sync.Once
		finish := func() { once.Do(func() { close(release) }) }
		t.Cleanup(finish)
		h.beforeLocalRetirementDone = func() { close(entered); <-release }
		done := make(chan error, 1)
		go func() { done <- b.readLoop(ctx) }()
		synctest.Wait()
		cancel()
		<-entered
		synctest.Wait()
		select {
		case err := <-done:
			t.Fatalf("read loop returned before retirement callback's final access: %v", err)
		default:
		}
		if err := b.readLoop(ctx); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
			t.Fatalf("qualified lifecycle admitted a replacement read loop: %v", err)
		}
		finish()
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	})
}

func TestRoutingHistoryLocalRetirementStoppedBeforeStart(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, _, _ := historyFixture(t)
		h := b.routingHistory
		ctx, cancel, err := b.newServeContext(nil, h.lifetime)
		if err != nil {
			t.Fatal(err)
		}
		defer cancel()
		codec, err := h.readers.acquire(historyLocalControl, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil)
		if err != nil {
			t.Fatal(err)
		}
		b.codec = codec
		started := make(chan struct{}, 1)
		h.beforeLocalRetirementDone = func() { started <- struct{}{} }
		if err := b.readLoop(ctx); err != nil {
			t.Fatal(err)
		}
		cancel()
		synctest.Wait()
		select {
		case <-started:
			t.Fatal("prevented local retirement callback started after loop completion")
		default:
		}
	})
}

func historyMarshal(t *testing.T, value any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode history fixture: %v", err)
	}
	return raw
}

func historyActivate(t *testing.T, b *Broker, run string) *proxyProcess {
	t.Helper()
	p := &proxyProcess{done: make(chan struct{})}
	b.setProxyProcessHandle(p)
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.incarnation, r.runID, r.activeRevision, r.activationErr = p, "", "", ""
	r.current.Revision = "1"
	r.mu.Unlock()
	ack := nodepolicy.PolicyAck{OriginNodeID: b.nodeID, RunID: run, Revision: "1"}
	if !b.recordActivation(p, "1", historyMarshal(t, ack), nil, nil) {
		t.Fatal("verified activation not recorded")
	}
	if !b.bindRoutingHistoryActivation(p) {
		t.Fatal("verified activation not bound")
	}
	return p
}

func historyRecord(id string, created, seq int64) nodepolicy.RoutingHistoryRecord {
	return nodepolicy.RoutingHistoryRecord{Schema: 1,
		Request: nodepolicy.RoutingRequestKey{OriginNodeID: "self-node", ProxyRunID: "run-a", RequestID: id},
		Seq:     seq, CreatedAt: created, State: "routing",
		Requested: nodepolicy.RoutingHistoryRequested{Model: "synthetic-model", Mode: "chat"},
		Spend:     nodepolicy.RoutingHistorySpend{Currency: "USD", Accounting: "none"},
		Attempts:  []nodepolicy.RoutingHistoryAttempt{}}
}

func historyTerminal(record nodepolicy.RoutingHistoryRecord) nodepolicy.RoutingHistoryRecord {
	end := record.CreatedAt + 1
	record.State, record.Reason, record.EndedAt = "failed", "no-owner", &end
	return record
}

func historyManifest(t *testing.T, generation string, records ...nodepolicy.RoutingHistoryRecord) nodepolicy.RoutingHistoryWindow {
	t.Helper()
	w := nodepolicy.RoutingHistoryWindow{Schema: 1, Generation: generation, Keys: []nodepolicy.RoutingHistoryWindowKey{}}
	for _, record := range records {
		if err := record.Validate(); err != nil {
			t.Fatalf("invalid record fixture: %v", err)
		}
		w.Keys = append(w.Keys, nodepolicy.RoutingHistoryWindowKey{Request: record.Request, CreatedAt: record.CreatedAt, Seq: record.Seq})
	}
	slices.SortFunc(w.Keys, func(a, b nodepolicy.RoutingHistoryWindowKey) int {
		if c := strings.Compare(a.Request.OriginNodeID, b.Request.OriginNodeID); c != 0 {
			return c
		}
		if c := strings.Compare(a.Request.ProxyRunID, b.Request.ProxyRunID); c != 0 {
			return c
		}
		if len(a.Request.RequestID) != len(b.Request.RequestID) {
			return len(a.Request.RequestID) - len(b.Request.RequestID)
		}
		return strings.Compare(a.Request.RequestID, b.Request.RequestID)
	})
	if len(w.Keys) == nodepolicy.RoutingHistoryRecords {
		least := w.Keys[0]
		for _, key := range w.Keys[1:] {
			if key.CreatedAt < least.CreatedAt {
				least = key
			}
		}
		w.Cutoff = &nodepolicy.RoutingHistoryCutoff{Request: least.Request, CreatedAt: least.CreatedAt}
	}
	if err := w.Validate(); err != nil {
		t.Fatalf("invalid window fixture: %v", err)
	}
	return w
}

func historySend(t *testing.T, b *Broker, p *proxyProcess, generation string, records ...nodepolicy.RoutingHistoryRecord) {
	t.Helper()
	b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingHistoryWindow, historyMarshal(t, historyManifest(t, generation, records...)))
	for _, record := range records {
		b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingRequest, historyMarshal(t, record))
	}
	b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingHistoryWindowEnd, historyMarshal(t, nodepolicy.RoutingHistoryWindowEnd{Schema: 1, Generation: generation}))
}

func historyRows(t *testing.T, b *Broker) []nodepolicy.RoutingHistoryRecord {
	t.Helper()
	h := b.routingHistory
	h.mu.Lock()
	defer h.mu.Unlock()
	rows := make([]nodepolicy.RoutingHistoryRecord, 0, h.accepted.count)
	for i := 0; i < h.accepted.count; i++ {
		var record nodepolicy.RoutingHistoryRecord
		if err := historyUnpackRecord(h.accepted.cell(i)[:h.accepted.length[i]], &record); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, record)
	}
	return rows
}

func historyRequireRows(t *testing.T, b *Broker, want ...nodepolicy.RoutingHistoryRecord) {
	t.Helper()
	got := historyRows(t, b)
	if len(got) != len(want) {
		t.Fatalf("accepted rows = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if !reflect.DeepEqual(got[i], want[i]) {
			t.Fatalf("row %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestRoutingHistoryStageAcceptance(t *testing.T) {
	test := func(name string, change func(*testing.T, *Broker, *proxyProcess, nodepolicy.RoutingHistoryRecord)) {
		t.Run(name, func(t *testing.T) {
			b, p, _ := historyFixture(t)
			row := historyRecord("9", 10, 1)
			historySend(t, b, p, "1", row)
			change(t, b, p, row)
			historyRequireRows(t, b, row)
		})
	}
	test("incomplete end preserves accepted", func(t *testing.T, b *Broker, p *proxyProcess, row nodepolicy.RoutingHistoryRecord) {
		row.Seq = 2
		b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingHistoryWindow, historyMarshal(t, historyManifest(t, "2", row)))
		b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingHistoryWindowEnd, historyMarshal(t, nodepolicy.RoutingHistoryWindowEnd{Schema: 1, Generation: "2"}))
	})
	test("record without begin has no effects", func(t *testing.T, b *Broker, p *proxyProcess, row nodepolicy.RoutingHistoryRecord) {
		row.Seq = 2
		b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingRequest, historyMarshal(t, row))
	})
	test("equal generation conflict discards stage", func(t *testing.T, b *Broker, p *proxyProcess, row nodepolicy.RoutingHistoryRecord) {
		row.Seq = 2
		historySend(t, b, p, "1", row)
	})
	test("unchanged generation replay is idempotent", func(t *testing.T, b *Broker, p *proxyProcess, row nodepolicy.RoutingHistoryRecord) {
		version := b.routingHistory.version
		historySend(t, b, p, "1", row)
		if b.routingHistory.version != version {
			t.Fatal("replay produced an accepted upsert")
		}
	})
	test("same sequence conflicting facts rejected", func(t *testing.T, b *Broker, p *proxyProcess, row nodepolicy.RoutingHistoryRecord) {
		row.Requested.Model = "other-model"
		historySend(t, b, p, "2", row)
	})
	test("immutable creation rejected", func(t *testing.T, b *Broker, p *proxyProcess, row nodepolicy.RoutingHistoryRecord) {
		row.Seq = 2
		row.CreatedAt++
		historySend(t, b, p, "2", row)
	})
	test("foreign bound run rejected", func(t *testing.T, b *Broker, p *proxyProcess, row nodepolicy.RoutingHistoryRecord) {
		row.Request.ProxyRunID = "other-run"
		historySend(t, b, p, "2", row)
	})
	test("later begin supersedes incomplete window", func(t *testing.T, b *Broker, p *proxyProcess, row nodepolicy.RoutingHistoryRecord) {
		other := historyRecord("10", 11, 1)
		b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingHistoryWindow, historyMarshal(t, historyManifest(t, "2", other)))
		historySend(t, b, p, "3", row)
	})
}

func TestRoutingHistoryOrdering(t *testing.T) {
	t.Run("terminal cannot reopen and old sequence cannot replace", func(t *testing.T) {
		b, p, _ := historyFixture(t)
		row := historyTerminal(historyRecord("9", 10, 2))
		historySend(t, b, p, "1", row)
		active := historyRecord("9", 10, 3)
		historySend(t, b, p, "2", active)
		older := historyTerminal(historyRecord("9", 10, 1))
		historySend(t, b, p, "3", older)
		historyRequireRows(t, b, row)
	})
	t.Run("full window floor prevents evicted old live resurrection", func(t *testing.T) {
		b, p, _ := historyFixture(t)
		rows := make([]nodepolicy.RoutingHistoryRecord, 0, 256)
		for i := 1; i <= 256; i++ {
			rows = append(rows, historyRecord(strconv.Itoa(i), 100, 1))
		}
		historySend(t, b, p, "1", rows...)
		rows[0] = historyRecord("257", 100, 1)
		historySend(t, b, p, "2", rows...)
		historySend(t, b, p, "3", historyRecord("1", 100, 99))
		got := historyRows(t, b)
		if len(got) != 256 || got[0].Request.RequestID != "257" || got[255].Request.RequestID != "2" {
			t.Fatalf("retained ranks = %d / %s / %s", len(got), got[0].Request.RequestID, got[len(got)-1].Request.RequestID)
		}
	})
	t.Run("backward timestamp cannot outrank retained window", func(t *testing.T) {
		b, p, _ := historyFixture(t)
		rows := make([]nodepolicy.RoutingHistoryRecord, 256)
		for i := range rows {
			rows[i] = historyRecord(strconv.Itoa(i+1), 100, 1)
		}
		historySend(t, b, p, "1", rows...)
		historySend(t, b, p, "2", historyRecord("1000", 99, 1))
		got := historyRows(t, b)
		if len(got) != 256 || got[0].Request.RequestID != "256" {
			t.Fatal("clock rollback displaced retained request")
		}
	})
}

func TestRoutingHistoryBinding(t *testing.T) {
	b, p, _ := historyFixture(t)
	row := historyRecord("9", 10, 1)
	b.workersMu.Lock()
	b.historyOrigin, b.historyRun = "", ""
	b.workersMu.Unlock()
	historySend(t, b, p, "1", row)
	historyRequireRows(t, b)
	if !b.bindRoutingHistoryActivation(p) {
		t.Fatal("verified binding refused")
	}
	historySend(t, b, p, "1", row)
	historyRequireRows(t, b, row)
	other := &proxyProcess{done: make(chan struct{})}
	if b.bindRoutingHistoryActivation(other) {
		t.Fatal("other handle bound from current activation")
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.runID = "different-run"
	r.mu.Unlock()
	if b.bindRoutingHistoryActivation(p) {
		t.Fatal("binding changed the already established run")
	}
}

func TestRoutingHistoryPrivateRefusal(t *testing.T) {
	b, p, out := historyFixture(t)
	row := historyRecord("9", 10, 1)
	raw := historyMarshal(t, row)
	for _, method := range []string{nodepolicy.NotifyRoutingHistoryWindow, nodepolicy.NotifyRoutingRequest, nodepolicy.NotifyRoutingHistoryWindowEnd, nodepolicy.NotifyRoutingHistoryAccepted, nodepolicy.NotifyRoutingChanged, engines.AddressMethod("ollama", nodepolicy.NotifyRoutingRequest)} {
		b.handleMessage(routingBrokerRequest("1", method, string(raw)))
		b.handleMessage(&Message{JSONRPC: "2.0", Method: method, Params: raw})
		b.forwardClusterManagerNotification(method, raw)
		b.forwardEngineNotificationFrom(nil, method, raw)
		b.forwardProxyProcessNotification(0, 0, method, raw)
		b.forwardProxyNotificationFrom(p, 0, 0, method, raw)
	}
	historyRequireRows(t, b)
	frames := out.frames()
	if len(frames) != 6 {
		t.Fatalf("client refusals/peer effects = %d frames, want 6", len(frames))
	}
	for _, frame := range frames {
		if frame.Error == nil || frame.Error.Code != -32601 || len(frame.Params) != 0 {
			t.Fatalf("private method escaped: %+v", frame)
		}
	}
	for _, method := range []string{nodepolicy.MethodRoutingSubscribe, nodepolicy.MethodRoutingUnsubscribe, nodepolicy.MethodRoutingGetInitial} {
		if _, denied := internalClientMethods[method]; denied {
			t.Fatalf("public history method denied: %s", method)
		}
	}
}

func TestRoutingHistoryRPC(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, p, out := historyFixture(t)
		row := historyRecord("9", 10, 1)
		historySend(t, b, p, "1", row)
		for _, call := range []struct{ id, method string }{{`"subscribe-id"`, nodepolicy.MethodRoutingSubscribe}, {`"initial-id"`, nodepolicy.MethodRoutingGetInitial}, {`"unsubscribe-id"`, nodepolicy.MethodRoutingUnsubscribe}} {
			b.handleMessage(routingBrokerRequest(call.id, call.method, `{}`))
			synctest.Wait()
			frames := out.frames()
			reply := frames[len(frames)-1]
			var initial nodepolicy.RoutingHistoryInitial
			if reply.ID == nil || string(*reply.ID) != call.id || reply.Error != nil {
				t.Fatalf("reply = %+v", reply)
			}
			if err := json.Unmarshal(reply.Result, &initial); err != nil {
				t.Fatal(err)
			}
			if len(initial.Requests) != 1 || !reflect.DeepEqual(initial.Requests[0], row) {
				t.Fatalf("RPC initial = %+v", initial)
			}
		}
		if b.subscribed || b.workloadsSubscribed || b.engineSubscribed || b.routingHistory.subscribed {
			t.Fatal("history modified another stream subscription or failed unsubscribe")
		}
	})
}

type historyBlockingOutput struct {
	once    sync.Once
	entered chan struct{}
	release chan struct{}
	out     syncBuffer
}

func (w *historyBlockingOutput) Read([]byte) (int, error) { return 0, io.EOF }
func (w *historyBlockingOutput) Write(data []byte) (int, error) {
	var frame Message
	if err := json.Unmarshal(data, &frame); err != nil {
		return 0, err
	}
	if frame.Method == "" && frame.Error == nil {
		w.once.Do(func() { close(w.entered) })
		<-w.release
	}
	return w.out.Write(data)
}

func TestRoutingHistoryInitialBusyLifetime(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, p, _ := historyFixture(t)
		blocked := &historyBlockingOutput{entered: make(chan struct{}), release: make(chan struct{})}
		b.codec = NewCodec(blocked)
		row := historyRecord("9", 10, 1)
		historySend(t, b, p, "1", row)
		b.handleMessage(routingBrokerRequest(`"old"`, nodepolicy.MethodRoutingGetInitial, `{}`))
		<-blocked.entered
		if !b.routingHistory.initialBusy.Load() {
			t.Fatal("initial grant released during active write")
		}
		row.Seq = 2
		historySend(t, b, p, "2", row)
		// The error uses the ordinary path; it can wait for the current physical
		// frame, but cannot allocate another initial or block the bank owner.
		done := make(chan struct{})
		go func() {
			b.handleMessage(routingBrokerRequest(`"busy"`, nodepolicy.MethodRoutingGetInitial, `{}`))
			close(done)
		}()
		synctest.Wait()
		if b.routingHistory.initial.count != 1 || historyWord(b.routingHistory.initial.cell(0), historySlotSequence) != 1 {
			t.Fatal("frozen initial reused while writing")
		}
		close(blocked.release)
		<-done
		synctest.Wait()
		if b.routingHistory.initialBusy.Load() {
			t.Fatal("actual write completion retained initial")
		}
		frames := blocked.out.frames()
		if len(frames) != 2 || frames[1].Error == nil || frames[1].Error.Code != jsonrpc.HistoryBusyCode {
			t.Fatalf("busy replies = %+v", frames)
		}
		var initial nodepolicy.RoutingHistoryInitial
		if err := json.Unmarshal(frames[0].Result, &initial); err != nil {
			t.Fatal(err)
		}
		if initial.Requests[0].Seq != 1 {
			t.Fatal("active write saw mutable bank")
		}
		historyRequireRows(t, b, row)
	})
}

func TestRoutingHistoryInitialDeadlineCompletionBeforeReuse(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, _, _ := historyFixture(t)
		h := b.routingHistory
		started, release := make(chan struct{}), make(chan struct{})
		var releaseOnce sync.Once
		releaseDeadline := func() { releaseOnce.Do(func() { close(release) }) }
		t.Cleanup(releaseDeadline)
		h.beforeInitialDeadlineDone = func() { close(started); <-release }
		stop := make(chan struct{})
		t.Cleanup(func() { close(stop) })
		w := &historySteppedOutput{steps: make(chan historyWriteStep), stop: stop}
		b.codec = NewCodec(readWriter{strings.NewReader(""), w})
		ownerDone := make(chan error, 1)
		go func() { ownerDone <- b.codec.Notify("ordinary-owner", nil) }()
		owner := historyNextWrite(t, w)
		b.handleMessage(routingBrokerRequest(`"expired"`, nodepolicy.MethodRoutingGetInitial, `{}`))
		<-started // the actual 250ms callback published acquisition cancellation
		close(owner.finish)
		if err := <-ownerDone; err != nil {
			t.Fatal(err)
		}
		fallback := historyNextWrite(t, w)
		if fallback.frame.ID == nil || string(*fallback.frame.ID) != `"expired"` || fallback.frame.Error == nil || fallback.frame.Error.Code != jsonrpc.HistoryBusyCode || len(fallback.frame.Result) != 0 {
			t.Fatalf("expired acquisition wrote something other than the mandatory busy reply: %+v", fallback.frame)
		}
		close(fallback.finish)
		synctest.Wait()
		if !h.initialBusy.Load() || len(w.out.frames()) != 2 {
			t.Fatal("initial admission was reused before the started deadline callback completed")
		}
		busyDone := make(chan struct{})
		go func() {
			b.handleMessage(routingBrokerRequest(`"still-busy"`, nodepolicy.MethodRoutingGetInitial, `{}`))
			close(busyDone)
		}()
		busy := historyNextWrite(t, w)
		if busy.frame.ID == nil || string(*busy.frame.ID) != `"still-busy"` || busy.frame.Error == nil || busy.frame.Error.Code != jsonrpc.HistoryBusyCode {
			t.Fatalf("started callback allowed another initial: %+v", busy.frame)
		}
		close(busy.finish)
		<-busyDone
		releaseDeadline()
		synctest.Wait()
		if h.initialBusy.Load() {
			t.Fatal("joined deadline callback retained initial admission")
		}
		b.handleMessage(routingBrokerRequest(`"reused"`, nodepolicy.MethodRoutingGetInitial, `{}`))
		reused := historyNextWrite(t, w)
		if reused.frame.ID == nil || string(*reused.frame.ID) != `"reused"` || reused.frame.Error != nil {
			t.Fatalf("joined initial admission was not reusable: %+v", reused.frame)
		}
		var initial nodepolicy.RoutingHistoryInitial
		if err := json.Unmarshal(reused.frame.Result, &initial); err != nil || initial.Requests == nil || len(initial.Requests) != 0 {
			t.Fatalf("reused initial result = %+v, error = %v", initial, err)
		}
		close(reused.finish)
		synctest.Wait()
		if h.initialBusy.Load() {
			t.Fatal("reused initial did not complete")
		}
	})
}

func TestRoutingHistoryInitialDeadlineAdmittedWriteLifetime(t *testing.T) {
	test := func(name string, cancelParent bool) {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				b, _, _ := historyFixture(t)
				h := b.routingHistory
				parent, cancel := context.WithCancel(t.Context())
				defer cancel()
				h.parent = parent
				started, release := make(chan struct{}), make(chan struct{})
				var releaseOnce sync.Once
				releaseDeadline := func() { releaseOnce.Do(func() { close(release) }) }
				t.Cleanup(releaseDeadline)
				h.beforeInitialDeadlineDone = func() { close(started); <-release }
				stop := make(chan struct{})
				t.Cleanup(func() { close(stop) })
				w := &historySteppedOutput{steps: make(chan historyWriteStep), stop: stop}
				// The actual local constructor supplies no write interrupt.
				b.codec = NewCodec(readWriter{strings.NewReader(""), w})
				b.handleMessage(routingBrokerRequest(`"admitted"`, nodepolicy.MethodRoutingGetInitial, `{}`))
				active := historyNextWrite(t, w)
				<-started
				if cancelParent {
					cancel()
				}
				synctest.Wait()
				if !h.initialBusy.Load() || len(w.out.frames()) != 0 {
					t.Fatal("acquire expiry or parent cancellation released a local active Write")
				}
				close(active.finish)
				synctest.Wait()
				frames := w.out.frames()
				if !h.initialBusy.Load() || len(frames) != 1 || frames[0].ID == nil || string(*frames[0].ID) != `"admitted"` || frames[0].Error != nil {
					t.Fatal("actual output completion bypassed the started deadline join or changed the admitted result")
				}
				releaseDeadline()
				synctest.Wait()
				if h.initialBusy.Load() {
					t.Fatal("parent cleanup did not join the started acquisition callback")
				}
				ordinaryDone := make(chan error, 1)
				go func() { ordinaryDone <- b.codec.Notify("ordinary-after-acquire-expiry", nil) }()
				ordinary := historyNextWrite(t, w)
				close(ordinary.finish)
				if err := <-ordinaryDone; err != nil {
					t.Fatalf("acquisition deadline retired the ordinary codec: %v", err)
				}
			})
		})
	}
	test("acquire expiry leaves admitted local output alone", false)
	test("parent cancellation still joins started acquisition callback", true)
}

func TestRoutingHistoryInitialDeadlineStoppedBeforeStart(t *testing.T) {
	test := func(name string, canceledParent bool) {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				b, _, out := historyFixture(t)
				h := b.routingHistory
				if canceledParent {
					parent, cancel := context.WithCancelCause(t.Context())
					// An inherited cause is not this owner's acquisition expiry.
					cancel(context.DeadlineExceeded)
					h.parent = parent
				}
				started := make(chan struct{}, 1)
				h.beforeInitialDeadlineDone = func() { started <- struct{}{} }
				b.handleMessage(routingBrokerRequest(`"stopped"`, nodepolicy.MethodRoutingGetInitial, `{}`))
				synctest.Wait()
				wantFrames := 1
				if canceledParent {
					wantFrames = 0
				}
				if h.initialBusy.Load() || len(out.frames()) != wantFrames {
					t.Fatal("unstarted deadline did not stop or changed parent cancellation classification")
				}
				<-time.After(250 * time.Millisecond)
				synctest.Wait()
				select {
				case <-started:
					t.Fatal("stopped-unstarted deadline callback ran after admission reuse")
				default:
				}
			})
		})
	}
	test("successful initial stops unstarted callback", false)
	test("canceled parent does not become deadline busy", true)
}

func TestRoutingHistoryInitialPinnedAcrossDeparture(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, p, _ := historyFixture(t)
		blocked := &historyBlockingOutput{entered: make(chan struct{}), release: make(chan struct{})}
		b.codec = NewCodec(blocked)
		row := historyRecord("9", 10, 1)
		historySend(t, b, p, "1", row)
		b.handleMessage(routingBrokerRequest(`"before"`, nodepolicy.MethodRoutingGetInitial, `{}`))
		<-blocked.entered
		b.clearProxyHandles()
		replacement := historyActivate(t, b, "run-b")
		newer := historyRecord("1", 20, 1)
		newer.Request.ProxyRunID = "run-b"
		historySend(t, b, replacement, "1", newer)
		h := b.routingHistory
		h.mu.Lock()
		h.retireLocked()
		h.mu.Unlock()
		if historyWord(h.initial.cell(0), historySlotSequence) != 1 {
			t.Fatal("departure mutated frozen initial")
		}
		close(blocked.release)
		synctest.Wait()
		b.handleMessage(routingBrokerRequest(`"after"`, nodepolicy.MethodRoutingGetInitial, `{}`))
		synctest.Wait()
		frames := blocked.out.frames()
		if len(frames) != 2 {
			t.Fatalf("initial replies = %d", len(frames))
		}
		var before, after nodepolicy.RoutingHistoryInitial
		if err := json.Unmarshal(frames[0].Result, &before); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(frames[1].Result, &after); err != nil {
			t.Fatal(err)
		}
		if len(before.Requests) != 1 || before.Requests[0].State != "routing" || len(after.Requests) != 2 || after.Requests[0].Request.ProxyRunID != "run-b" || after.Requests[1].State != "failed" || after.Requests[1].Reason != "proxy-exited" || after.Requests[1].Seq != 2 {
			t.Fatalf("initial departure ordering = %+v / %+v", before, after)
		}
	})
}

func TestRoutingHistoryAcceptedPushes(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, p, out := historyFixture(t)
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		b.routingHistory.parent = ctx
		ackOut := &syncBuffer{}
		p.peer = NewPeer(NewCodec(ackOut))
		go b.routingHistory.run()
		b.handleMessage(routingBrokerRequest("1", nodepolicy.MethodRoutingSubscribe, `{}`))
		synctest.Wait()
		row := historyRecord("9", 10, 1)
		historySend(t, b, p, "1", row)
		synctest.Wait()
		historySend(t, b, p, "1", row)
		synctest.Wait()
		conflict := row
		conflict.Seq = 2
		conflict.CreatedAt++
		historySend(t, b, p, "2", conflict)
		synctest.Wait()
		frames := out.frames()
		if len(frames) != 2 || frames[1].Method != nodepolicy.NotifyRoutingChanged {
			t.Fatalf("accepted-only frames = %+v", frames)
		}
		var pushed nodepolicy.RoutingHistoryRecord
		if err := json.Unmarshal(frames[1].Params, &pushed); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(pushed, row) {
			t.Fatal("push was not the accepted record")
		}
		acks := ackOut.frames()
		if len(acks) != 2 {
			t.Fatalf("ack/replay count = %d", len(acks))
		}
		for _, ack := range acks {
			var accepted nodepolicy.RoutingHistoryAccepted
			if err := json.Unmarshal(ack.Params, &accepted); err != nil {
				t.Fatal(err)
			}
			if ack.Method != nodepolicy.NotifyRoutingHistoryAccepted || accepted.Generation != "1" {
				t.Fatal("conflicting stage acknowledged")
			}
		}
		b.handleMessage(routingBrokerRequest("2", nodepolicy.MethodRoutingUnsubscribe, `{}`))
		synctest.Wait()
		row.Seq = 3
		historySend(t, b, p, "3", row)
		synctest.Wait()
		if len(out.frames()) != 3 {
			t.Fatal("unsubscribed history emitted a push")
		}
	})
}

func TestRoutingHistoryRawFailureOrdinaryCallSurvives(t *testing.T) {
	client, server := net.Pipe()
	t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
	b, p, _ := historyFixture(t)
	if err := p.attachRoutingHistoryPeer(b, client, func() { _ = client.Close() }); err != nil {
		t.Fatal(err)
	}
	p.onNotify = func(method string, params json.RawMessage) { b.forwardProxyNotificationFrom(p, 0, 0, method, params) }
	failures := make(chan jsonrpc.HistoryFailure, 4)
	if err := p.peer.SetHistoryReadHandlers(nil, func(f jsonrpc.HistoryFailure) { b.routingHistoryFailure(p, f); failures <- f }); err != nil {
		t.Fatal(err)
	}
	go p.servePeer()
	worker := NewCodec(server)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	result := make(chan error, 1)
	go func() {
		raw, rpcErr, err := p.peer.Call(ctx, "ordinary-status", json.RawMessage(`{}`))
		if err == nil {
			var response struct {
				OK bool `json:"ok"`
			}
			if rpcErr != nil {
				err = errors.New("ordinary call failed")
			} else if decodeErr := json.Unmarshal(raw, &response); decodeErr != nil {
				err = decodeErr
			} else if !response.OK {
				err = errors.New("ordinary result changed")
			}
		}
		result <- err
	}()
	request, err := worker.Read()
	if err != nil {
		t.Fatal(err)
	}
	row := historyRecord("9", 10, 1)
	if err := worker.Notify(nodepolicy.NotifyRoutingHistoryWindow, historyManifest(t, "1", row)); err != nil {
		t.Fatal(err)
	}
	bad := `{"jsonrpc":"2.0","method":"routing/request","params":{"synthetic-secret-key":1,"synthetic-secret-key":2}}` + "\n"
	if _, err := io.WriteString(server, bad); err != nil {
		t.Fatal(err)
	}
	failure := <-failures
	if failure.Disposition != "discard" {
		t.Fatalf("disposition = %+v", failure)
	}
	ambiguous := `{"jsonrpc":"2.0","method":"routing/request","method":"ordinary-notification","params":{}}` + "\n"
	if _, err := io.WriteString(server, ambiguous); err != nil {
		t.Fatal(err)
	}
	quarantined := <-failures
	if quarantined.Disposition != "quarantine" || !quarantined.ScheduleRefetch || quarantined.FailPending {
		t.Fatalf("notification-only quarantine = %+v", quarantined)
	}
	if err := worker.Respond(request.ID, map[string]bool{"ok": true}); err != nil {
		t.Fatal(err)
	}
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	historyRequireRows(t, b)
	if err := worker.Notify(nodepolicy.NotifyRoutingHistoryWindow, historyManifest(t, "1", row)); err != nil {
		t.Fatal(err)
	}
	if err := worker.Notify(nodepolicy.NotifyRoutingRequest, row); err != nil {
		t.Fatal(err)
	}
	if err := worker.Notify(nodepolicy.NotifyRoutingHistoryWindowEnd, nodepolicy.RoutingHistoryWindowEnd{Schema: 1, Generation: "1"}); err != nil {
		t.Fatal(err)
	}
	// A subsequent ordinary response is the pump barrier after end handling.
	go func() { _, _, err := p.peer.Call(ctx, "barrier", json.RawMessage(`{}`)); result <- err }()
	barrier, err := worker.Read()
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.Respond(barrier.ID, struct{}{}); err != nil {
		t.Fatal(err)
	}
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	historyRequireRows(t, b, row)
}

type historyBrokenOutput struct{}

func (historyBrokenOutput) Read([]byte) (int, error)  { return 0, io.EOF }
func (historyBrokenOutput) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestRoutingHistoryBrokenAckPreservesClientWriter(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, p, out := historyFixture(t)
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		b.routingHistory.parent = ctx
		p.peer = NewPeer(NewCodec(historyBrokenOutput{}))
		go b.routingHistory.run()
		b.handleMessage(routingBrokerRequest("1", nodepolicy.MethodRoutingSubscribe, `{}`))
		synctest.Wait()
		row := historyRecord("9", 10, 1)
		historySend(t, b, p, "1", row)
		synctest.Wait()
		row.Seq = 2
		historySend(t, b, p, "2", row)
		synctest.Wait()
		frames := out.frames()
		if len(frames) != 3 {
			t.Fatalf("client replies/pushes after broken ack = %d", len(frames))
		}
		for i, wantSeq := range []int64{1, 2} {
			var got nodepolicy.RoutingHistoryRecord
			if err := json.Unmarshal(frames[i+1].Params, &got); err != nil {
				t.Fatal(err)
			}
			if frames[i+1].Method != nodepolicy.NotifyRoutingChanged || got.Seq != wantSeq {
				t.Fatalf("accepted push after broken source write = %+v", got)
			}
		}
	})
}

func TestRoutingHistoryRawClientDispatcher(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, p, out := historyFixture(t)
		row := historyRecord("9", 10, 1)
		historySend(t, b, p, "1", row)
		// Two successive raw IDs deliberately occupy the same reader backing.
		input := `{"jsonrpc":"2.0","id":"first\u0031","method":"routing:get-initial","params":{}}` + "\n" +
			`{"jsonrpc":"2.0","id":"second","method":"routing:subscribe","params":{}}` + "\n" +
			`{"jsonrpc":"2.0","id":3,"method":"version","params":{}}` + "\n"
		codec, err := b.routingHistory.readers.acquire(historyLocalControl, readWriter{Reader: strings.NewReader(input), Writer: out}, func() {}, nil)
		if err != nil {
			t.Fatal(err)
		}
		b.codec = codec
		if err := b.readLoop(t.Context()); err != nil {
			t.Fatal(err)
		}
		synctest.Wait()
		frames := out.frames()
		if len(frames) != 3 {
			t.Fatalf("dispatched replies = %d", len(frames))
		}
		ids := map[string]bool{}
		for _, frame := range frames {
			if frame.ID == nil {
				t.Fatal("missing caller ID")
			}
			ids[string(*frame.ID)] = true
		}
		if !ids[`"first\u0031"`] || !ids[`"second"`] || !ids[`3`] {
			t.Fatalf("borrowed caller IDs changed: %+v", ids)
		}
		if !b.routingHistory.subscribed {
			t.Fatal("raw dispatcher skipped subscribe")
		}
	})
}

func TestRoutingHistoryLocalErrorsKeepLiteralRawIDs(t *testing.T) {
	for _, size := range []int{86, 510, 511} {
		t.Run(strconv.Itoa(size)+" literal less-than bytes", func(t *testing.T) {
			b, _, out := historyFixture(t)
			b.routingHistory.initialBusy.Store(true)
			token := `"` + strings.Repeat("<", size) + `"`
			input := `{"jsonrpc":"2.0","id":` + token + `,"method":"routing:get-initial","params":{}}` + "\n" +
				`{"jsonrpc":"2.0","id":` + token + `,"method":"routing:subscribe","params":null}` + "\n" +
				`{"jsonrpc":"2.0","id":` + token + `,"method":"routing/request","params":{}}` + "\n"
			codec, err := b.routingHistory.readers.acquire(historyLocalControl, readWriter{strings.NewReader(input), out}, func() {}, nil)
			if err != nil {
				t.Fatal(err)
			}
			b.codec = codec
			if err := b.readLoop(t.Context()); err != nil {
				t.Fatal(err)
			}
			frames := out.frames()
			if len(frames) != 3 {
				t.Fatalf("required local replies = %d", len(frames))
			}
			for i, code := range []int{jsonrpc.HistoryBusyCode, jsonrpc.HistoryInvalidCode, -32601} {
				wantToken := token
				if size == 511 {
					wantToken, code = "null", jsonrpc.HistoryInvalidCode
				}
				gotToken := "null"
				if frames[i].ID != nil {
					gotToken = string(*frames[i].ID)
				}
				if gotToken != wantToken || frames[i].Error == nil || frames[i].Error.Code != code || frames[i].Error.Data != nil {
					t.Fatalf("local error %d changed raw identity or fixed code: %+v", i, frames[i])
				}
			}
			// A constructed Message has no raw-guard provenance. Direct private
			// dispatch retains ordinary serialization and has no history ID cap;
			// canonical transport refusals are asserted independently above.
			b.handleMessage(routingBrokerRequest(token, nodepolicy.NotifyRoutingRequest, `{}`))
			last := out.frames()[3]
			actual := "null"
			if last.ID != nil {
				actual = string(*last.ID)
			}
			ordinaryToken := `"` + strings.Repeat(`\u003c`, size) + `"`
			if actual != ordinaryToken || last.Error == nil || last.Error.Code != -32601 || last.Error.Message != "method not found" || last.Error.Data != nil {
				t.Fatal("direct private fixture changed baseline ordinary refusal")
			}
			b.handleMessage(routingBrokerRequest(token, nodepolicy.MethodRoutingSubscribe, `null`))
			invalid := out.frames()[4]
			wantToken := token
			if size == 511 {
				wantToken = "null"
			}
			actual = "null"
			if invalid.ID != nil {
				actual = string(*invalid.ID)
			}
			if actual != wantToken || invalid.Error == nil || invalid.Error.Code != jsonrpc.HistoryInvalidCode {
				t.Fatal("direct public invalid-params reply changed raw ID")
			}
		})
	}
}

func TestRoutingHistoryAbsentOwnerUsesOrdinaryReplies(t *testing.T) {
	methods := []string{nodepolicy.MethodRoutingSubscribe, nodepolicy.MethodRoutingUnsubscribe, nodepolicy.MethodRoutingGetInitial}
	ids := []struct{ name, fields, want string }{
		{"object ID", `"id":{}`, `{}`},
		{"aliased duplicate ID", `"id":"first","ID":"second"`, `"second"`},
		{"literal HTML-sensitive ID", `"id":"` + strings.Repeat("<", 86) + `"`, `"` + strings.Repeat(`\u003c`, 86) + `"`},
		{"ordinary ID above history limits", `"id":"` + strings.Repeat("<", 1024) + `"`, `"` + strings.Repeat(`\u003c`, 1024) + `"`},
	}
	for _, method := range methods {
		for _, id := range ids {
			t.Run(method+"/"+id.name, func(t *testing.T) {
				out, logs := &syncBuffer{}, &syncBuffer{}
				previous := slog.Default()
				slog.SetDefault(slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
				t.Cleanup(func() { slog.SetDefault(previous) })
				input := `{"jsonrpc":"2.0",` + id.fields + `,"method":"` + method + `","params":{"synthetic-private-key":"absent-owner-payload-sentinel"}}` + "\n" +
					`{"jsonrpc":"2.0","id":9,"method":"version","params":{}}` + "\n"
				// Match main's current ordinary codec construction, without a J owner
				// or the shared history guard that could validate a raw reply token.
				b := &Broker{codec: NewCodec(readWriter{strings.NewReader(input), out}), nodeID: "self-node"}
				if err := b.readLoop(t.Context()); err != nil {
					t.Fatal(err)
				}
				frames := out.frames()
				if len(frames) != 2 {
					t.Fatalf("ordinary refusal/progress replies = %d", len(frames))
				}
				refusal := frames[0]
				if refusal.ID == nil || string(*refusal.ID) != id.want || refusal.Error == nil || refusal.Error.Code != -32601 || refusal.Error.Message != "method not found" || refusal.Error.Data != nil || refusal.Method != "" || len(refusal.Params) != 0 || len(refusal.Result) != 0 {
					t.Fatalf("absent owner changed baseline ordinary refusal: %+v", refusal)
				}
				if frames[1].ID == nil || string(*frames[1].ID) != "9" || frames[1].Error != nil || len(frames[1].Result) == 0 {
					t.Fatal("ordinary RPC did not progress after absent-owner refusal")
				}
				if b.routingHistory != nil || b.historyIncarnation != 0 || b.historyOrigin != "" || b.historyRun != "" || b.historyRetiredThrough != 0 || b.engineProxies != nil || b.routingForward.Load() {
					t.Fatal("absent-owner refusal constructed history or changed history/facade state")
				}
				logs.mu.Lock()
				captured := logs.buf.String()
				logs.mu.Unlock()
				if captured != "" {
					t.Fatal("successful absent-owner refusal produced a diagnostic")
				}
			})
		}
	}
}

func TestRoutingHistoryComponentMethodsStayLocal(t *testing.T) {
	methods := []string{
		nodepolicy.NotifyRoutingHistoryWindow, nodepolicy.NotifyRoutingRequest, nodepolicy.NotifyRoutingHistoryWindowEnd,
		nodepolicy.NotifyRoutingHistoryAccepted, nodepolicy.NotifyRoutingChanged,
		nodepolicy.MethodRoutingSubscribe, nodepolicy.MethodRoutingUnsubscribe, nodepolicy.MethodRoutingGetInitial,
	}
	ids := []struct{ fields, want string }{
		{`"id":{}`, `{}`},
		{`"id":"first","ID":"second"`, `"second"`},
		{`"id":"` + strings.Repeat("<", 86) + `"`, `"` + strings.Repeat(`\u003c`, 86) + `"`},
		{`"id":"` + strings.Repeat("<", 1024) + `"`, `"` + strings.Repeat(`\u003c`, 1024) + `"`},
	}
	for _, profile := range engineProxyProfiles {
		for _, availability := range []string{"live proxy", "unavailable proxy"} {
			t.Run(profile.ComponentName()+"/"+availability, func(t *testing.T) {
				synctest.Test(t, func(t *testing.T) {
					b, _, out := historyFixture(t)
					var worker *fakeWorker
					if availability == "live proxy" {
						_, worker = attachFakeProxy(t, b)
					}
					manager := attachFakeEngine(t, b)
					// Source binding publishes startup identity/projection state.
					// Join that fixture work before measuring the denial corpus.
					waitPushesSettled(t, b)
					synctest.Wait()
					if worker != nil {
						worker.mu.Lock()
						worker.calls = nil
						worker.mu.Unlock()
					}
					manager.mu.Lock()
					manager.calls, manager.projections = nil, nil
					manager.mu.Unlock()
					out.mu.Lock()
					out.buf.Reset()
					out.mu.Unlock()
					workloads := &syncBuffer{}
					b.setWorkloadMgr(&workloadManagerProcess{stdin: privacyWriteCloser{workloads}})
					incarnation, retired := b.historyIncarnation, b.historyRetiredThrough
					logs := &syncBuffer{}
					previous := slog.Default()
					slog.SetDefault(slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
					t.Cleanup(func() { slog.SetDefault(previous) })
					var input strings.Builder
					var wantIDs []string
					for _, method := range methods {
						// Exercise both the component's bare spelling and an already
						// facade-addressed alias using the actual descriptor resolver.
						for _, suffix := range []string{method, profile.addressed(method)} {
							addressed := profile.ComponentName() + ":" + suffix
							for _, id := range ids {
								input.WriteString(`{"jsonrpc":"2.0",` + id.fields + `,"method":"` + addressed + `","params":{"synthetic-private-key":"component-history-payload-sentinel"}}` + "\n")
								wantIDs = append(wantIDs, id.want)
							}
							input.WriteString(`{"jsonrpc":"2.0","method":"` + addressed + `","params":{"synthetic-private-key":"component-history-notification-sentinel"}}` + "\n")
						}
					}
					input.WriteString(`{"jsonrpc":"2.0","id":9,"method":"version","params":{}}` + "\n")
					b.codec = NewCodec(readWriter{strings.NewReader(input.String()), out})
					if err := b.readLoop(t.Context()); err != nil {
						t.Fatal(err)
					}
					synctest.Wait()
					frames := out.frames()
					if len(frames) != len(wantIDs)+1 {
						t.Fatalf("component refusal/progress replies = %d, want %d", len(frames), len(wantIDs)+1)
					}
					for i, want := range wantIDs {
						refusal := frames[i]
						if refusal.ID == nil || string(*refusal.ID) != want || refusal.Error == nil || refusal.Error.Code != -32601 || refusal.Error.Message != "method not found" || refusal.Error.Data != nil || refusal.Method != "" || len(refusal.Params) != 0 || len(refusal.Result) != 0 {
							t.Fatalf("component refusal %d changed ordinary fixed contract: %+v", i, refusal)
						}
					}
					progress := frames[len(wantIDs)]
					if progress.ID == nil || string(*progress.ID) != "9" || progress.Error != nil || len(progress.Result) == 0 {
						t.Fatal("ordinary RPC did not progress after component history denial")
					}
					if worker != nil && len(worker.snapshot()) != 0 {
						t.Fatal("component history request or notification reached live proxy")
					}
					if len(manager.snapshot()) != 0 || len(workloads.frames()) != 0 {
						t.Fatal("component history caused engine/workload manager work")
					}
					manager.mu.Lock()
					projections := len(manager.projections)
					manager.mu.Unlock()
					if projections != 0 || b.workloads != nil || b.store != nil || b.telemetry != nil || b.routingForward.Load() {
						t.Fatal("component history changed ordinary caches or routing work")
					}
					h := b.routingHistory
					if h.accepted.count != 0 || h.stage.count != 0 || h.merge.count != 0 || h.initial.count != 0 || h.version != 0 || h.subscription != 0 || h.subscribed || h.initialBusy.Load() || h.ackSource != nil || h.refetchSource != nil || h.quarantined || h.refusal != (routingHistoryRefusal{}) || h.refusalActive || b.historyIncarnation != incarnation || b.historyRetiredThrough != retired {
						t.Fatal("component history changed J state, staging, apply, refetch or refusal storage")
					}
					for i := range h.readers.slots {
						if !reflect.ValueOf(&h.readers.slots[i].correlation).Elem().IsZero() {
							t.Fatal("owned component request copied history correlation")
						}
					}
					for _, state := range b.engineProxies {
						if state.subscribed {
							t.Fatal("component history enabled a facade subscription")
						}
					}
					logs.mu.Lock()
					captured := logs.buf.String()
					logs.mu.Unlock()
					if captured != "" {
						t.Fatal("component history request or notification reached logging")
					}
					// The live fixture must still relay a legitimate facade RPC after
					// the denial corpus, with the original params and ordinary ID.
					if worker != nil {
						b.handleMessage(routingBrokerRequest(`"ordinary-control"`, profile.ComponentName()+":nodes/list", `{"control":true}`))
						calls := worker.snapshot()
						if len(calls) != 1 || calls[0].method != profile.addressed("nodes/list") || string(calls[0].params) != `{"control":true}` || calls[0].notify {
							t.Fatal("legitimate facade RPC stopped relaying after history denial")
						}
						replies := out.frames()
						last := replies[len(replies)-1]
						if len(replies) != len(frames)+1 || last.ID == nil || string(*last.ID) != `"ordinary-control"` || last.Error != nil || len(last.Result) == 0 {
							t.Fatal("legitimate facade reply changed")
						}
					}
				})
			})
		}
	}
}

func TestRoutingHistoryLocalOwnedAliasesUseOrdinaryRefusal(t *testing.T) {
	cases := []struct {
		name, frame, wantID string
	}{
		{
			name:   "object ID",
			frame:  `{"jsonrpc":"2.0","id":{},"method":"ollama:routing/request","params":{}}`,
			wantID: `{}`,
		},
		{
			name:   "aliased duplicate ID keeps ordinary last-wins behavior",
			frame:  `{"jsonrpc":"2.0","id":"first","ID":"second","method":"ollama:routing/request","params":{}}`,
			wantID: `"second"`,
		},
		{
			name:   "literal HTML-sensitive ordinary ID",
			frame:  `{"jsonrpc":"2.0","id":"` + strings.Repeat("<", 86) + `","method":"ollama:routing/request","params":{"synthetic-private-key":"local-alias-payload-sentinel"}}`,
			wantID: `"` + strings.Repeat(`\u003c`, 86) + `"`,
		},
		{
			name:   "ordinary ID above history ID and error-frame bounds remains ordinary",
			frame:  `{"jsonrpc":"2.0","id":"` + strings.Repeat("<", 1024) + `","method":"ollama:routing/request","params":{"synthetic-private-key":"local-alias-payload-sentinel"}}`,
			wantID: `"` + strings.Repeat(`\u003c`, 1024) + `"`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				b, _, out := historyFixture(t)
				logs := &syncBuffer{}
				previous := slog.Default()
				slog.SetDefault(slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
				t.Cleanup(func() { slog.SetDefault(previous) })
				input, feed := io.Pipe()
				t.Cleanup(func() { _ = input.Close(); _ = feed.Close() })
				h := b.routingHistory
				codec, err := h.readers.acquire(historyLocalControl, readWriter{input, out}, func() { _ = input.Close() }, nil)
				if err != nil {
					t.Fatal(err)
				}
				// Inspect the live reader before retirement can clear its token.
				reader := reflect.ValueOf(&h.readers.slots[historyLocalControl].control).Elem().FieldByName("reader").Elem()
				b.codec = codec
				ctx, cancel := context.WithCancel(t.Context())
				t.Cleanup(cancel)
				done := make(chan error, 1)
				go func() { done <- b.readLoop(ctx) }()
				inputFrames := tc.frame + "\n" + `{"jsonrpc":"2.0","id":9,"method":"version","params":{}}` + "\n"
				if _, err := io.WriteString(feed, inputFrames); err != nil {
					t.Fatal(err)
				}
				synctest.Wait() // both replies completed; next physical read is blocked
				frames := out.frames()
				if len(frames) != 2 {
					t.Fatalf("ordinary refusal/progress replies = %d", len(frames))
				}
				refusal := frames[0]
				if refusal.ID == nil || string(*refusal.ID) != tc.wantID || refusal.Error == nil || refusal.Error.Code != -32601 || refusal.Error.Message != "method not found" || refusal.Error.Data != nil || refusal.Method != "" || len(refusal.Params) != 0 || len(refusal.Result) != 0 {
					t.Fatalf("ordinary fixed refusal changed: %+v", refusal)
				}
				progress := frames[1]
				if progress.ID == nil || string(*progress.ID) != "9" || progress.Error != nil || len(progress.Result) == 0 {
					t.Fatal("ordinary version RPC did not progress after private alias denial")
				}
				if h.accepted.count != 0 || h.stage.count != 0 || h.version != 0 || h.subscribed || h.initialBusy.Load() || h.ackSource != nil || h.refetchSource != nil || h.quarantined || h.refusal.source != nil || h.refusal.n != 0 || h.refusalActive || h.refusal.id != [jsonrpc.HistoryIDBytes]byte{} {
					t.Fatal("ordinary local alias changed J state, staging or refusal storage")
				}
				if !reader.FieldByName("idToken").IsZero() || !reflect.ValueOf(&h.readers.slots[historyLocalControl].correlation).Elem().IsZero() {
					t.Fatal("ordinary local alias copied a history control ID or changed correlation")
				}
				logs.mu.Lock()
				captured := logs.buf.String()
				logs.mu.Unlock()
				for _, sentinel := range []string{"ollama:routing/request", "first", "second", "synthetic-private-key", "local-alias-payload-sentinel", strings.Repeat("<", 86)} {
					if strings.Contains(captured, sentinel) {
						t.Fatalf("ordinary private rejection logged input sentinel %s", sentinel)
					}
				}
				cancel()
				if err := <-done; err != nil {
					t.Fatal(err)
				}
				historyAwaitReadSignal(t, codec.RetireRead())
			})
		})
	}
}

func TestRoutingHistoryOwnedAddressedRequestHasZeroHistoryEffects(t *testing.T) {
	b, p, _ := historyFixture(t)
	out := &syncBuffer{}
	frame := `{"jsonrpc":"2.0","id":"first","ID":"second","method":"ollama:routing/request","params":{}}` + "\n" +
		`{"jsonrpc":"2.0","method":"ordinary-barrier","params":{}}` + "\n"
	if err := p.attachRoutingHistoryPeer(b, readWriter{strings.NewReader(frame), out}, func() {}); err != nil {
		t.Fatal(err)
	}
	requests, failures, ordinary := 0, 0, 0
	if err := p.peer.SetHistoryReadHandlers(func(*Message) { requests++ }, func(jsonrpc.HistoryFailure) { failures++ }); err != nil {
		t.Fatal(err)
	}
	p.onNotify = func(method string, _ json.RawMessage) {
		if method == "ordinary-barrier" {
			ordinary++
		}
	}
	p.servePeer() // broker supplies no baseline ordinary worker request handler
	h := b.routingHistory
	if requests != 0 || failures != 0 || ordinary != 1 || len(out.frames()) != 0 || h.refusal.source != nil || h.refusal.n != 0 || h.refusal.id != [jsonrpc.HistoryIDBytes]byte{} || h.accepted.count != 0 || h.stage.count != 0 || h.refetchSource != nil || h.quarantined {
		t.Fatal("owned addressed/duplicate-ID request entered history handling")
	}
}

type historyFallbackOutput struct {
	calls                      int // writes are serialized by the codec's existing gate
	ownerEntered, ownerRelease chan struct{}
	errorEntered, errorRelease chan struct{}
	out                        syncBuffer
}

func (w *historyFallbackOutput) Write(p []byte) (int, error) {
	w.calls++
	switch w.calls {
	case 1:
		close(w.ownerEntered)
		<-w.ownerRelease
	case 2:
		return w.out.Write(p[:7])
	case 3:
		close(w.errorEntered)
		<-w.errorRelease
	}
	return w.out.Write(p)
}

func TestRoutingHistoryInitialFallbackWaitsForGateAndActualWrite(t *testing.T) {
	for _, size := range []int{86, 510} {
		t.Run(strconv.Itoa(size)+" literal less-than bytes", func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				b, _, _ := historyFixture(t)
				w := &historyFallbackOutput{ownerEntered: make(chan struct{}), ownerRelease: make(chan struct{}), errorEntered: make(chan struct{}), errorRelease: make(chan struct{})}
				var ownerOnce, errorOnce sync.Once
				releaseOwner := func() { ownerOnce.Do(func() { close(w.ownerRelease) }) }
				releaseError := func() { errorOnce.Do(func() { close(w.errorRelease) }) }
				t.Cleanup(releaseOwner)
				t.Cleanup(releaseError)
				codec := NewCodec(readWriter{strings.NewReader(""), w})
				ownerDone := make(chan error, 1)
				go func() { ownerDone <- codec.Notify("ordinary-owner", nil) }()
				<-w.ownerEntered
				token := `"` + strings.Repeat("<", size) + `"`
				var id [jsonrpc.HistoryIDBytes]byte
				idN := copy(id[:], token)
				b.routingHistory.initialBusy.Store(true)
				done := make(chan struct{})
				go func() { b.routingHistory.writeInitial(codec, id, idN); close(done) }()
				time.Sleep(300 * time.Millisecond) // virtual time: initial gate admission expired
				synctest.Wait()
				select {
				case <-done:
					t.Fatal("fallback dropped the mandatory busy reply at an occupied gate")
				default:
				}
				if !b.routingHistory.initialBusy.Load() {
					t.Fatal("fallback released initial ownership while waiting for the reply gate")
				}
				releaseOwner()
				if err := <-ownerDone; err != nil {
					t.Fatal(err)
				}
				<-w.errorEntered
				if !b.routingHistory.initialBusy.Load() {
					t.Fatal("fallback released initial ownership during partial error Write")
				}
				releaseError()
				<-done
				frames := w.out.frames()
				if len(frames) != 2 || frames[1].ID == nil || string(*frames[1].ID) != token || frames[1].Error == nil || frames[1].Error.Code != jsonrpc.HistoryBusyCode || b.routingHistory.initialBusy.Load() {
					t.Fatalf("required fallback changed ID/code/completion: %+v", frames)
				}
			})
		})
	}
}

func TestRoutingHistoryLocalRawErrorGateWaitKeepsBorrow(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, _, _ := historyFixture(t)
		w := &historyFallbackOutput{ownerEntered: make(chan struct{}), ownerRelease: make(chan struct{}), errorEntered: make(chan struct{}), errorRelease: make(chan struct{})}
		var ownerOnce, errorOnce sync.Once
		releaseOwner := func() { ownerOnce.Do(func() { close(w.ownerRelease) }) }
		releaseError := func() { errorOnce.Do(func() { close(w.errorRelease) }) }
		t.Cleanup(releaseOwner)
		t.Cleanup(releaseError)
		token := `"` + strings.Repeat("<", 510) + `"`
		input := `{"jsonrpc":"2.0","id":` + token + `,"method":"routing:subscribe","params":null}` + "\n"
		r := &b.routingHistory.readers
		codec, err := r.acquire(historyLocalControl, readWriter{strings.NewReader(input), w}, func() {}, nil)
		if err != nil {
			t.Fatal(err)
		}
		b.codec = codec
		ownerDone := make(chan error, 1)
		go func() { ownerDone <- codec.Notify("ordinary-owner", nil) }()
		<-w.ownerEntered
		ctx, cancel := context.WithCancel(t.Context())
		t.Cleanup(cancel)
		loopDone := make(chan error, 1)
		go func() { loopDone <- b.readLoop(ctx) }()
		synctest.Wait()
		cancel()
		retired := codec.RetireRead()
		historyRequireReadPending(t, retired)
		if _, err := r.acquire(historyLocalControl, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
			t.Fatalf("gate waiter released borrowed ID: %v", err)
		}
		releaseOwner()
		if err := <-ownerDone; err != nil {
			t.Fatal(err)
		}
		<-w.errorEntered
		historyRequireReadPending(t, retired)
		releaseError()
		if err := <-loopDone; err != nil {
			t.Fatal(err)
		}
		historyAwaitReadSignal(t, retired)
		frames := w.out.frames()
		if len(frames) != 2 || frames[1].ID == nil || string(*frames[1].ID) != token || frames[1].Error == nil || frames[1].Error.Code != jsonrpc.HistoryInvalidCode {
			t.Fatalf("gate/partial-write reply lost exact borrowed ID: %+v", frames)
		}
	})
}

func TestRoutingHistoryPrivacyGuard(t *testing.T) {
	b, p, out := historyFixture(t)
	logs := &syncBuffer{}
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	row := historyRecord("9", 10, 1)
	raw := historyMarshal(t, row)
	unknown := `"synthetic-secret-key":{"prompt":"prompt-sentinel","response":"response-sentinel","credential":"credential-sentinel","endpoint":"endpoint-sentinel","PIN":"pin-sentinel","key":"key-sentinel"},`
	withUnknown := append([]byte("{"+unknown), raw[1:]...)
	b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingHistoryWindow, historyMarshal(t, historyManifest(t, "1", row)))
	b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingRequest, withUnknown)
	b.forwardProxyNotificationFrom(p, 0, 0, nodepolicy.NotifyRoutingHistoryWindowEnd, historyMarshal(t, nodepolicy.RoutingHistoryWindowEnd{Schema: 1, Generation: "1"}))
	historyRequireRows(t, b, row)
	b.routingHistoryFailure(p, jsonrpc.HistoryFailure{Disposition: "quarantine", Category: "history-invalid", Count: 1, ScheduleRefetch: true})
	packed := b.routingHistory.accepted.cell(0)[:b.routingHistory.accepted.length[0]]
	logs.mu.Lock()
	captured := logs.buf.String()
	logs.mu.Unlock()
	for _, sentinel := range []string{"synthetic-secret-key", "prompt-sentinel", "response-sentinel", "credential-sentinel", "endpoint-sentinel", "pin-sentinel", "key-sentinel"} {
		if bytes.Contains(packed, []byte(sentinel)) || strings.Contains(captured, sentinel) {
			t.Fatalf("private sentinel retained: %s", sentinel)
		}
	}
	if !strings.Contains(captured, "history-invalid") {
		t.Fatal("missing fixed diagnostic positive control")
	}
	if len(out.frames()) != 0 {
		t.Fatal("private raw record entered generic client echo")
	}
}

func TestRoutingHistoryRawPrivateDispositions(t *testing.T) {
	test := func(name, fragment string) {
		t.Run(name, func(t *testing.T) {
			b, p, _ := historyFixture(t)
			row := historyRecord("9", 10, 1)
			begin := historyMarshal(t, Message{JSONRPC: "2.0", Method: nodepolicy.NotifyRoutingHistoryWindow, Params: historyMarshal(t, historyManifest(t, "1", row))})
			record := historyMarshal(t, row)
			bad := `{"jsonrpc":"2.0","method":"routing/request","params":{` + fragment + string(record[1:]) + "}\n"
			// One invalid private frame is followed by an ordinary notification
			// and a fresh complete replay on the very same existing read pump.
			end := historyMarshal(t, Message{JSONRPC: "2.0", Method: nodepolicy.NotifyRoutingHistoryWindowEnd, Params: historyMarshal(t, nodepolicy.RoutingHistoryWindowEnd{Schema: 1, Generation: "1"})})
			good := historyMarshal(t, Message{JSONRPC: "2.0", Method: nodepolicy.NotifyRoutingRequest, Params: record})
			input := string(begin) + "\n" + bad + `{"jsonrpc":"2.0","method":"ordinary-notification","params":{"ok":true}}` + "\n" + string(begin) + "\n" + string(good) + "\n" + string(end) + "\n"
			if err := p.attachRoutingHistoryPeer(b, readWriter{Reader: strings.NewReader(input), Writer: io.Discard}, func() {}); err != nil {
				t.Fatal(err)
			}
			ordinary, failures := 0, 0
			p.onNotify = func(method string, params json.RawMessage) {
				if method == "ordinary-notification" {
					ordinary++
					return
				}
				b.forwardProxyNotificationFrom(p, 0, 0, method, params)
			}
			if err := p.peer.SetHistoryReadHandlers(nil, func(f jsonrpc.HistoryFailure) {
				failures++
				if f.Disposition != "discard" {
					t.Errorf("private disposition = %+v", f)
				}
				b.routingHistoryFailure(p, f)
			}); err != nil {
				t.Fatal(err)
			}
			p.servePeer()
			if ordinary != 1 || failures != 1 {
				t.Fatalf("ordinary/failure callbacks = %d/%d", ordinary, failures)
			}
			historyRequireRows(t, b, row)
		})
	}
	test("duplicate escaped unknown key", `"synthetic-secret-key":1,"synthetic-secret-\u006bey":2,`)
	test("reverse duplicate escaped unknown key", `"synthetic-secret-\u006bey":1,"synthetic-secret-key":2,`)
	test("nested unknown duplicate", `"synthetic-secret-key":{"opaque":1,"opaque":2},`)
}

func TestRoutingHistoryPackedCapacity(t *testing.T) {
	b, _, _ := historyFixture(t)
	h := b.routingHistory
	for _, bank := range []*routingHistoryBank{h.accepted, h.stage, h.merge, h.initial} {
		if len(bank.data) != 16777216 || cap(bank.data) != 16777216 {
			t.Fatal("window backing is not exactly W")
		}
	}
	// The reader roles each own R in addition to the owner's R projector.
	// The fixed admission controller below owns/reuses those exact role banks.
	const want = 135086080
	if 4*historyWindowBytes+historyCellBytes+2*jsonrpc.HistoryRecordFrameBytes+3*jsonrpc.HistoryRecordBytes+2*jsonrpc.HistoryFrameBytes+historyControlBytes != want {
		t.Fatal("broker grant formula changed")
	}
	row := historyRecord("9223372036854775807", 100, historyMaxSequence)
	model := strings.Repeat(`"\`, 256)
	row.Requested.Model = model
	node, engine := strings.Repeat("n", 128), strings.Repeat("e", 128)
	for i := 0; i < 32; i++ {
		row.Attempts = append(row.Attempts, nodepolicy.RoutingHistoryAttempt{Index: int64(i), StartedAt: 100, Destination: nodepolicy.RoutingHistoryDestination{Kind: "local", Model: &model, NodeID: &node, Engine: &engine}, Outcome: "pending"})
	}
	if err := row.Validate(); err != nil {
		t.Fatal(err)
	}
	historyHeader(h.scratch[:], row.Request, row.CreatedAt, row.Seq, 1)
	n, err := historyPackRecord(h.scratch[:], row)
	if err != nil {
		t.Fatal(err)
	}
	if n > 65536 {
		t.Fatal("maximum metadata exceeded packed cell")
	}
	var decoded nodepolicy.RoutingHistoryRecord
	if err := historyUnpackRecord(h.scratch[:n], &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, row) {
		t.Fatal("packed UTF-8 metadata changed")
	}
	w := historyBoundedJSON{data: make([]byte, 0, jsonrpc.HistoryRecordBytes)}
	if err := historyRecordJSON(&w, h.scratch[:n]); err != nil {
		t.Fatal(err)
	}
	var wire nodepolicy.RoutingHistoryRecord
	if err := json.Unmarshal(w.data, &wire); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(wire, row) {
		t.Fatal("bounded encoding changed schema/nulls")
	}
	// A legal initial exceeds the old 8MiB cap without pagination or a
	// generic marshal/RawMessage/window materialization chain.
	initial := historyBoundedJSON{data: make([]byte, 0, jsonrpc.HistoryInitialBytes)}
	initial.literal(`{"requests":[`)
	for i := 0; i < 256; i++ {
		if i != 0 {
			initial.literal(",")
		}
		row.Request.RequestID = strconv.Itoa(i + 1)
		historyHeader(h.scratch[:], row.Request, row.CreatedAt, row.Seq, 1)
		n, err := historyPackRecord(h.scratch[:], row)
		if err != nil {
			t.Fatal(err)
		}
		if err := historyRecordJSON(&initial, h.scratch[:n]); err != nil {
			t.Fatal(err)
		}
	}
	initial.literal("]}")
	if initial.err != nil || len(initial.data) <= 8<<20 || len(initial.data) > 32<<20 {
		t.Fatalf("initial encoding bytes = %d / %v", len(initial.data), initial.err)
	}
}

func TestRoutingHistoryPackedOutputPreservesKnownFields(t *testing.T) {
	row := historyRecord("9223372036854775807", 100, 2)
	row.Requested.Model = "synthetic-\"model\\with-多字"
	var cell [historyCellBytes]byte
	historyHeader(cell[:], row.Request, row.CreatedAt, row.Seq, 1)
	n, err := historyPackRecord(cell[:], row)
	if err != nil {
		t.Fatalf("pack record: %v", err)
	}
	out := historyBoundedJSON{data: make([]byte, 0, jsonrpc.HistoryRecordBytes)}
	if err := historyRecordJSON(&out, cell[:n]); err != nil {
		t.Fatalf("encode packed record: %v", err)
	}
	var got nodepolicy.RoutingHistoryRecord
	if err := json.Unmarshal(out.data, &got); err != nil {
		t.Fatalf("decode encoded record: %v", err)
	}
	if !reflect.DeepEqual(got, row) {
		t.Fatal("packed output changed known fields, escaped text or nulls")
	}
}

func TestRoutingHistoryPackedOutputRejectsMalformedBacking(t *testing.T) {
	row := historyRecord("9", 100, 1)
	var cell [historyCellBytes]byte
	historyHeader(cell[:], row.Request, row.CreatedAt, row.Seq, 1)
	n, err := historyPackRecord(cell[:], row)
	if err != nil {
		t.Fatalf("pack record: %v", err)
	}
	tests := []struct {
		name string
		raw  []byte
	}{
		{name: "short header", raw: cell[:historySlotHeader-1]},
		{name: "short integer", raw: cell[:historySlotHeader+1]},
		{name: "missing attempt count", raw: cell[:n-1]},
		{name: "trailing bytes", raw: append(append([]byte(nil), cell[:n]...), 1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := historyBoundedJSON{data: make([]byte, 0, jsonrpc.HistoryRecordBytes)}
			if err := historyRecordJSON(&out, tt.raw); err == nil {
				t.Fatal("malformed packed record accepted")
			}
		})
	}
	t.Run("output grant exhausted", func(t *testing.T) {
		out := historyBoundedJSON{data: make([]byte, 0, 1)}
		if err := historyRecordJSON(&out, cell[:n]); err == nil {
			t.Fatal("output exceeded its fixed grant")
		}
	})
}

func TestRoutingHistoryProjectedPackingMatchesTypedPacking(t *testing.T) {
	row := historyRecord("9223372036854775807", 100, historyMaxSequence)
	row.Requested.Model = strings.Repeat(`"\`, 256)
	node, engine := strings.Repeat("n", 128), strings.Repeat("e", 128)
	for i := 0; i < nodepolicy.RoutingHistoryAttempts; i++ {
		row.Attempts = append(row.Attempts, nodepolicy.RoutingHistoryAttempt{
			Index: int64(i), StartedAt: 100, Outcome: "pending",
			Destination: nodepolicy.RoutingHistoryDestination{Kind: "local", Model: &row.Requested.Model, NodeID: &node, Engine: &engine},
		})
	}
	if err := row.Validate(); err != nil {
		t.Fatalf("validate fixture: %v", err)
	}
	schema, _ := nodepolicy.RoutingHistoryTransportSchema(nodepolicy.NotifyRoutingRequest, false)
	raw := historyMarshal(t, row)
	projected, _, err := jsonrpc.ProjectHistory(raw, make([]byte, 0, jsonrpc.HistoryRecordBytes), make([]uint32, jsonrpc.HistoryRecordBytes/4), schema)
	if err != nil {
		t.Fatalf("project record: %v", err)
	}
	view, err := nodepolicy.ReadProjectedRoutingHistoryRecord(projected)
	if err != nil {
		t.Fatalf("read projected view: %v", err)
	}
	var typed, direct [historyCellBytes]byte
	historyHeader(typed[:], row.Request, row.CreatedAt, row.Seq, 1)
	historyHeader(direct[:], view.Header.Request, view.Header.CreatedAt, view.Header.Seq, 1)
	want, err := historyPackRecord(typed[:], row)
	if err != nil {
		t.Fatalf("pack typed: %v", err)
	}
	got, err := historyPackProjectedRecord(direct[:], &view)
	if err != nil {
		t.Fatalf("pack projected: %v", err)
	}
	if got != want || !bytes.Equal(direct[:got], typed[:want]) {
		t.Fatal("direct projected packing changed the private cell schema")
	}
	output := historyBoundedJSON{data: make([]byte, 0, jsonrpc.HistoryRecordBytes)}
	if err := historyRecordJSON(&output, direct[:got]); err != nil {
		t.Fatalf("encode direct cell: %v", err)
	}
	var decoded nodepolicy.RoutingHistoryRecord
	if err := json.Unmarshal(output.data, &decoded); err != nil {
		t.Fatalf("decode direct cell output: %v", err)
	}
	if !reflect.DeepEqual(decoded, row) {
		t.Fatal("direct ingestion/output changed record facts")
	}
}

func TestRoutingHistoryProjectedManifestOwnsStageHeaders(t *testing.T) {
	b := &Broker{codec: NewCodec(&syncBuffer{}), nodeID: strings.Repeat("n", 128)}
	lifetime := newBrokerMainLifetime()
	t.Cleanup(lifetime.cancel)
	var err error
	b.routingHistory, err = newBrokerRoutingHistory(b, lifetime)
	if err != nil {
		t.Fatal(err)
	}
	p := historyActivate(t, b, strings.Repeat("r", 128))
	window := nodepolicy.RoutingHistoryWindow{Schema: 1, Generation: "1", Keys: make([]nodepolicy.RoutingHistoryWindowKey, nodepolicy.RoutingHistoryRecords)}
	for i := range window.Keys {
		window.Keys[i] = nodepolicy.RoutingHistoryWindowKey{
			Request:   nodepolicy.RoutingRequestKey{OriginNodeID: b.nodeID, ProxyRunID: b.historyRun, RequestID: strconv.FormatInt(9223372036854775000+int64(i), 10)},
			CreatedAt: int64(i), Seq: historyMaxSequence,
		}
	}
	window.Cutoff = &nodepolicy.RoutingHistoryCutoff{Request: window.Keys[0].Request, CreatedAt: 0}
	if err := window.Validate(); err != nil {
		t.Fatalf("validate maximum manifest: %v", err)
	}
	raw := historyMarshal(t, window)
	b.routeRoutingHistoryFrom(p, nodepolicy.NotifyRoutingHistoryWindow, raw)
	clear(raw)
	h := b.routingHistory
	clear(h.projected[:])
	if h.stage.count != len(window.Keys) || h.accepted.count != 0 || !h.stageHasCutoff {
		t.Fatalf("stage/accepted/cutoff = %d/%d/%v", h.stage.count, h.accepted.count, h.stageHasCutoff)
	}
	for i, key := range window.Keys {
		origin, run, id := historyKeyParts(h.stage.cell(i))
		if string(origin) != key.Request.OriginNodeID || string(run) != key.Request.ProxyRunID || string(id) != key.Request.RequestID ||
			historyWord(h.stage.cell(i), historySlotCreated) != uint64(key.CreatedAt) || historyWord(h.stage.cell(i), historySlotSequence) != uint64(key.Seq) || h.stage.length[i] != 0 {
			t.Fatalf("stage header %d changed or retained a borrowed manifest", i)
		}
	}
}

func TestRoutingHistoryControlScratchOwnership(t *testing.T) {
	b, _, _ := historyFixture(t)
	h := b.routingHistory
	// These are source allowances for simultaneously live application data,
	// not a compiled-layout or RSS claim. The fixed 10001-byte grammar stack
	// returns before MaterializeHistory decodes strings. A scalar string has
	// at most 3074 encoded bytes, 512 decoded bytes and 3080 unquote scratch;
	// invalid original JSON/unknown branches never enter that decoder.
	const semantic = 25600
	const writer = 12288 // grammar stack, fixed schema walk and scalar buffers
	const retirement = 47104
	const fixed = 20480 // indexes, reader/control objects, ID and generation copies
	const currentHeader = 3072
	const partialHeader = 4608 // workload text is checked after attempts
	const currentAttempt = 1280
	// Packed values contain the same validated text and fewer bytes than the
	// semantic object allowances. uint16 lengths cannot narrow a valid record.
	if historySlotHeader+currentHeader+nodepolicy.RoutingHistoryAttempts*currentAttempt >= 65535 || unsafe.Sizeof(h.accepted.length) != nodepolicy.RoutingHistoryRecords*2 {
		t.Fatal("packed record/index representation exceeded its proven bound")
	}
	// Valid header text is at most 2216 bytes, plus 552 inline/nullable
	// bytes. Before the late workload check, its four 128-byte IDs may
	// instead occupy 512 bytes each. A valid attempt uses at most 880 text
	// bytes plus 336 inline/nullable bytes (two kind-exclusive 128-byte IDs,
	// a 512-byte model, fixed enums/reason and two 19-digit money strings).
	const cursorControl = 2048
	const grammar = 10001
	const stringDecode = 3080 + 512 + 1024
	// RespondHistoryError consumes the already validated token without another
	// parser/grammar stack. Count its one local-dispatcher frame in fixed C,
	// even if an owned ordinary request allows the reader to proceed. Thus no
	// assumption that local read scratch has ended is needed. The independent
	// initial fallback retains both H grants and its own frame; its record
	// encoder has returned, so that frame/write control use writer scratch.
	const localFrame = jsonrpc.HistoryIDBytes + 128
	const initialFallback = localFrame + 1024
	if localFrame > jsonrpc.HistoryEnvelopeBytes || initialFallback > writer {
		t.Fatal("local required reply escaped its admitted role scratch")
	}
	// Count independently allocated nullable referents as well as inline
	// fields. This fails if a future semantic header/attempt grows an array.
	var referents func(reflect.Type) int
	referents = func(typ reflect.Type) int {
		switch typ.Kind() {
		case reflect.String:
			return 512
		case reflect.Pointer:
			return int(typ.Elem().Size()) + referents(typ.Elem())
		case reflect.Struct:
			n := 0
			for i := 0; i < typ.NumField(); i++ {
				n += referents(typ.Field(i).Type)
			}
			return n
		case reflect.Slice, reflect.Map, reflect.Array:
			t.Fatalf("unbounded semantic DTO member: %s", typ)
		}
		return 0
	}
	headerType := reflect.TypeOf(nodepolicy.RoutingHistoryRecordHeader{})
	attemptType := reflect.TypeOf(nodepolicy.RoutingHistoryAttempt{})
	malformedHeader := int(headerType.Size()) + referents(headerType)
	malformedAttempt := int(attemptType.Size()) + referents(attemptType)
	// Header failure precedes attempts. Current failure precedes related
	// reads. A related lookup keeps only a validated current and clears its
	// own slot immediately afterwards, including malformed related values.
	liveFacts := max(malformedHeader, partialHeader+malformedAttempt, partialHeader+currentAttempt+malformedAttempt)
	if liveFacts+max(grammar, stringDecode)+cursorControl > semantic {
		t.Fatalf("semantic facts/decoder/control exceed role allowance: %d", liveFacts+max(grammar, stringDecode)+cursorControl)
	}
	retiredRecord := int(unsafe.Sizeof(nodepolicy.RoutingHistoryRecord{})) + currentHeader - int(headerType.Size()) + nodepolicy.RoutingHistoryAttempts*currentAttempt + cursorControl
	if retiredRecord > retirement {
		t.Fatalf("retirement DTO and validation scratch exceed allowance: %d", retiredRecord)
	}
	// The actual role controller and its fixed refusal slot are inside h.
	// Subtract only its two separately funded R arrays, retaining every
	// controller mutex/header/generation field in this control census.
	control := int(unsafe.Sizeof(*h)) - historyCellBytes - 3*jsonrpc.HistoryRecordBytes - jsonrpc.HistoryRecordFrameBytes + 4*int(unsafe.Sizeof(routingHistoryBank{}))
	// ReadControl and HistoryCorrelation are embedded in the two slots and
	// already included in sizeof(h). Only admitted reader objects are extra.
	control += 2 * (int(unsafe.Sizeof(jsonrpc.Codec{})) + int(unsafe.Sizeof(jsonrpc.HistoryReader{})) + int(unsafe.Sizeof(bufio.Reader{})) + 256)
	control += 56                                                // Broker history owner pointer, incarnation/frontier and origin/run
	control += int(unsafe.Sizeof(jsonrpc.HistoryReader{})) + 128 // shared terminal reader and completed-channel sentinel
	control += 3*jsonrpc.HistoryIDBytes + 3*39                   // three logical launch-ID locations and generation referents
	control += localFrame                                        // synchronous local reply: validation, gate wait, partial/blocked Write and borrower completion
	// Named default-amd64 source representations, not emitted allocation or
	// placement measurements. Channel pointers/function slots in h are already
	// counted above. Zero-sized channel elements add no backing here.
	const roleChannels = (2 + 2 + 1 + 1 + 1 + 3 + 1) * 112  // gates, role-done, wake, handoff, consumed, read/write callback-done, main writer join
	const retry = 112 + 112 + 24                            // timeTimer, channel and physical Time element
	const currentReadValues = 2*128 + 2*32 + 32             // message/header alternatives, release captures, proxy adapter
	const roleHooks = 5*24 + 24 + 24                        // local/proxy read, fence, request/failure; Peer and local write interrupts
	const afterFunctions = 3 * (104 + 16)                   // local retirement and both active Writes, including stop captures
	const localLoopCaptures = 24 + 32 + 32 + 40 + 72        // retire, callback, join defer, pump launch, dispatch
	const handoffCopies = 2 * 40                            // current unbuffered readResult locations, no queue
	const proxyWriteCaptures = 2 * (10 + 40 + 24 + 56 + 48) // mutex/flags, interrupt, callback, finish and defer for initial and serial writer
	const initialLaunch = 40 + 24                           // initial launch/join; ID data remains in the three-location row
	control += roleChannels + retry + currentReadValues + roleHooks + afterFunctions + localLoopCaptures + handoffCopies + proxyWriteCaptures + initialLaunch
	// One production output-cancellation AfterFunc, its stop capture, fixed join
	// channel, callback/cleanup captures and a conservative joined waiter. It
	// registers once on history, independently of required reply Write contexts.
	const outputCancellation = 104 + 16 + 112 + 24 + 24 + 96
	control += outputCancellation
	// The private factory owns main, Serve and history standard cancelCtx
	// headers/Done/cancel captures. Only main and history get J child maps:
	// their lifetime maxima are two and five entries, respectively. A Swiss
	// small map retains its 48-byte header and one 136-byte interface-key group;
	// deletion/cancellation never credits that backing as reclaimed. Ordinary
	// RPC child graphs stay with Serve and are not duplicated as J registrations.
	const standardRoots = 3 * (80 + 112 + 16)
	const retainedParentMaps = 2 * (48 + 136)
	const cancelFanout = 24
	const mainLifetimeCaptures = 32 + 16 + 16 + 24 + 24 // signal launch, main/Serve cancellation, history launch and cancel/join
	control += int(unsafe.Sizeof(*h.lifetime)) + standardRoots + retainedParentMaps + cancelFanout + mainLifetimeCaptures
	// Initial acquisition is counted separately from the preceding existing
	// non-acquisition controls. One initialBusy admission owns one cancelCtx,
	// its lazy Done channel, one AfterFunc timeTimer, one deadlineDone channel,
	// a cancel function value and the timer/cleanup captures. Source-derived
	// default amd64 representations: cancelCtx 80, hchan 112, timeTimer 112,
	// cancel capture 16, timer capture 32, cleanup capture 32, acquire box 16.
	// The singleton cause pointer/errorString is 32 plus its 44-byte literal.
	// The instance barrier's
	// function slot is already in sizeof(h); its production value is nil.
	// Stop(true) excludes the unstarted callback; otherwise deadlineDone joins
	// its final application access before initialBusy clears, even if output
	// or mandatory fallback finished earlier. No started acquisition callback
	// can continue application work across initial admission reuse.
	const initialAcquireNamedControl = 80 + 2*112 + 112 + 16 + 32 + 32 + 16 + 32 + 44
	control += initialAcquireNamedControl
	// This named graph is not a full C/layout proof: active compiled frames,
	// launch/callee transfer areas, capture boxing/coalescing and retained timer
	// bookkeeping still need closure. These named source forms are not a new
	// catch-all allowance: reconcile stack-resident captures with actual frames
	// once, and do not infer retention bounds from Stop, delete or return.
	var offsets [2]uintptr
	for i := range offsets {
		codec, err := h.readers.acquire(routingHistoryReadRole(i), readWriter{Reader: strings.NewReader(""), Writer: io.Discard}, func() {}, nil)
		if err != nil {
			t.Fatalf("construct reader role: %v", err)
		}
		bank := reflect.ValueOf(&h.readers.slots[i].control).Elem().FieldByName("reader").Elem().FieldByName("offsets")
		fragments := reflect.ValueOf(&h.readers.slots[i].control).Elem().FieldByName("reader").Elem().FieldByName("r").Elem().FieldByName("buf")
		if fragments.Cap() != 256 {
			t.Fatal("private fragment backing does not match its actual constructor charge")
		}
		offsets[i] = bank.Pointer()
		if !codec.ReadRetirementEnabled() || bank.Cap()*4 != jsonrpc.HistoryRecordBytes || bank.Pointer() != reflect.ValueOf(h.readers.slots[i].offsets[:]).Pointer() || bank.Pointer() == reflect.ValueOf(h.guard[:]).Pointer() {
			t.Fatal("reader R is missing or aliases the active owner projector")
		}
	}
	if offsets[0] == offsets[1] || len(h.guard)*4 != jsonrpc.HistoryRecordBytes {
		t.Fatal("three simultaneous offset owners do not have distinct R backing")
	}
	// Ordinary callbacks can retain arbitrary old shells across rebind. They
	// must have zero added J fields/referents, rather than one budgeted sample.
	old := h.readers.slots[historyProxyPrivate].codec
	roleControl := &h.readers.slots[historyProxyPrivate].control
	historyAwaitReadSignal(t, h.readers.retire(historyProxyPrivate, old))
	replacement, err := h.readers.acquire(historyProxyPrivate, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil)
	if err != nil {
		t.Fatalf("capacity schedule rebind: %v", err)
	}
	if roleControl != &h.readers.slots[historyProxyPrivate].control || roleControl.Available() {
		t.Fatal("rebind replaced the fixed control or failed to admit replacement")
	}
	historyRequireReadDetached(t, old)
	if !replacement.ReadRetirementEnabled() {
		t.Fatal("replacement lost explicit read retirement")
	}
	if control > fixed {
		t.Fatalf("control/index/reader/ID and retained read-state bytes = %d, allowance = %d", control, fixed)
	}
	// Owner semantic ingestion can overlap both readers and both encoders.
	// Retirement excludes owner ingestion and the sole push encoder under
	// h.mu/run, but may overlap both readers and the independent initial.
	initialWriter := max(writer, initialFallback)
	live := fixed + max(3*semantic+writer+initialWriter, 2*semantic+retirement+initialWriter)
	if live > historyControlBytes || historyControlBytes != 131072 {
		t.Fatalf("simultaneous C ownership = %d, grant = %d", live, historyControlBytes)
	}
}

func TestRoutingHistoryRetiredProxyKeepsOrdinaryFootprint(t *testing.T) {
	// The pre-J process handle fields. Captured ordinary callbacks may keep
	// arbitrarily many old handles alive, so no J field may be added here.
	want := []string{"name", "cmd", "stdin", "peer", "done", "credentialRefs", "listenerOnly", "bringUpDone", "readyMu", "facadeState", "onNotify", "relayDir", "subMu", "subIDs"}
	typ := reflect.TypeOf(proxyProcess{})
	if typ.NumField() != len(want) {
		t.Fatal("old proxy handle gained history fields")
	}
	for i, name := range want {
		if typ.Field(i).Name != name {
			t.Fatalf("ordinary process field %d = %s, want %s", i, typ.Field(i).Name, name)
		}
	}
}

func historyAwaitReadSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatal("read owner did not reach its completion boundary")
	}
}

func historyRequireReadPending(t *testing.T, done <-chan struct{}) {
	t.Helper()
	if done == nil {
		t.Fatal("missing actual read-retirement completion")
	}
	select {
	case <-done:
		t.Fatal("reader backing released before actual consumption")
	default:
	}
}

func historyRequireReadDetached(t *testing.T, codec *Codec) {
	t.Helper()
	// The shared permanent terminal marker occupies the existing pointer;
	// its contents are checked in shared tests. No per-codec control field.
	if _, ok := reflect.TypeOf(codec).Elem().FieldByName("readState"); ok {
		t.Fatal("old codec shell retains a per-generation control")
	}
	if msg, release, err := codec.ReadWithRelease(); msg != nil || release != nil || !errors.Is(err, jsonrpc.ErrPeerClosed) {
		t.Fatalf("late leased read: message %v, release %v, error %v", msg, release != nil, err)
	}
	if msg, err := codec.Read(); msg != nil || !errors.Is(err, jsonrpc.ErrPeerClosed) {
		t.Fatalf("late plain read: message %v, error %v", msg, err)
	}
}

func historyCompleteRetiredRead(t *testing.T, codec *Codec, done chan struct{}) {
	t.Helper()
	msg, release, err := codec.ReadWithRelease()
	if msg != nil || release != nil || !errors.Is(err, jsonrpc.ErrPeerClosed) {
		t.Errorf("interrupted read returned message %v, release %v, error %v", msg, release != nil, err)
	}
	close(done)
}

func TestRoutingHistoryReadRoleAdmission(t *testing.T) {
	b, _, _ := historyFixture(t)
	r := &b.routingHistory.readers
	rw := readWriter{Reader: strings.NewReader(""), Writer: io.Discard}
	if _, err := r.acquire(historyProxyPrivate, rw, nil, nil); !errors.Is(err, jsonrpc.ErrHistoryInvalid) {
		t.Fatalf("missing read owner admitted: %v", err)
	}
	if _, err := r.acquire(routingHistoryReadRole(2), rw, func() {}, nil); !errors.Is(err, jsonrpc.ErrHistoryInvalid) {
		t.Fatalf("third role admitted: %v", err)
	}
	for _, role := range []routingHistoryReadRole{historyProxyPrivate, historyLocalControl} {
		codec, err := r.acquire(role, rw, func() {}, nil)
		if err != nil {
			t.Fatalf("admit role: %v", err)
		}
		if _, err := codec.Read(); !errors.Is(err, jsonrpc.ErrReadLeaseRequired) {
			t.Fatalf("plain read bypassed retirement: %v", err)
		}
		before := &r.slots[role].raw[:cap(r.slots[role].raw)][0]
		if _, err := r.acquire(role, rw, func() {}, nil); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
			t.Fatalf("duplicate role allocated backing: %v", err)
		}
		done := r.retire(role, codec)
		historyAwaitReadSignal(t, done)
		historyAwaitReadSignal(t, r.retire(role, codec))
		replacement, err := r.acquire(role, rw, func() {}, nil)
		if err != nil {
			t.Fatalf("rebind retired role: %v", err)
		}
		if &r.slots[role].raw[:cap(r.slots[role].raw)][0] != before {
			t.Fatal("replacement allocated another ordinary input bank")
		}
		// A late retirement from the old generation cannot release the new one.
		r.retire(role, codec)
		if _, err := r.acquire(role, rw, func() {}, nil); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
			t.Fatalf("late old retirement released replacement: %v", err)
		}
		historyRequireReadDetached(t, codec)
		r.retire(role, replacement)
	}
}

func TestRoutingHistoryReadRoleBorrowBlocksReuse(t *testing.T) {
	test := func(name, frame string, refusal bool) {
		t.Run(name, func(t *testing.T) {
			b, _, _ := historyFixture(t)
			r := &b.routingHistory.readers
			rw := readWriter{Reader: strings.NewReader(frame + frame), Writer: io.Discard}
			codec, err := r.acquire(historyLocalControl, rw, func() {}, nil)
			if err != nil {
				t.Fatal(err)
			}
			msg, release, err := codec.ReadWithRelease()
			if refusal {
				var failure *jsonrpc.HistoryFailure
				if !errors.As(err, &failure) || failure.Refusal == nil || msg != nil {
					t.Fatalf("missing borrowed refusal: %v", err)
				}
			} else if err != nil || msg == nil {
				t.Fatalf("read valid history control: %v", err)
			}
			if release == nil {
				t.Fatal("history borrow has no release")
			}
			done := r.retire(historyLocalControl, codec)
			historyRequireReadPending(t, done)
			if _, err := r.acquire(historyLocalControl, rw, func() {}, nil); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
				t.Fatalf("borrowed bank rebound: %v", err)
			}
			release()
			release()
			historyAwaitReadSignal(t, done)
			historyRequireReadDetached(t, codec)
			replacement, err := r.acquire(historyLocalControl, readWriter{strings.NewReader(frame), io.Discard}, func() {}, nil)
			if err != nil {
				t.Fatalf("released bank unavailable: %v", err)
			}
			nextMsg, nextRelease, nextErr := replacement.ReadWithRelease()
			if refusal {
				var failure *jsonrpc.HistoryFailure
				if nextMsg != nil || !errors.As(nextErr, &failure) || failure.Refusal == nil {
					t.Fatalf("replacement refusal changed: %v", nextErr)
				}
			} else if nextErr != nil || nextMsg == nil {
				t.Fatalf("replacement control changed: %v", nextErr)
			}
			if nextRelease == nil {
				t.Fatal("replacement history borrow is missing")
			}
			release() // A duplicate old release cannot release the new borrow.
			nextDone := r.retire(historyLocalControl, replacement)
			historyRequireReadPending(t, nextDone)
			nextRelease()
			historyAwaitReadSignal(t, nextDone)
		})
	}
	test("accepted control", `{"jsonrpc":"2.0","id":"old","method":"routing:subscribe","params":{}}`+"\n", false)
	test("refusal with nil message", `{"jsonrpc":"2.0","id":"old","method":"routing:subscribe","params":null}`+"\n", true)
}

// This wraps the precise pipe descriptor, retaining its real Close behavior.
// The test is a future Windows/native gate; source presence is not OS proof.
type historyObservedReadCloser struct {
	io.ReadCloser
	entered chan struct{}
	once    sync.Once
}

func (r *historyObservedReadCloser) Read(p []byte) (int, error) {
	r.once.Do(func() { close(r.entered) })
	return r.ReadCloser.Read(p)
}

func TestRoutingHistoryProxyPipeReadRetirement(t *testing.T) {
	input, output, err := os.Pipe()
	if err != nil {
		t.Fatalf("create actual stdout pipe: %v", err)
	}
	t.Cleanup(func() { _ = output.Close(); _ = input.Close() })
	b, p, _ := historyFixture(t)
	owned, err := pipeio.NewReader(input)
	if err != nil {
		t.Fatal(err)
	}
	observed := &historyObservedReadCloser{ReadCloser: owned, entered: make(chan struct{})}
	p.stdin = historyDiscardCloser{}
	if err := p.attachRoutingHistoryProxyPipe(b, observed); err != nil {
		t.Fatalf("attach actual stdout owner: %v", err)
	}
	codec := p.peer.Codec()
	readDone := make(chan struct{})
	go historyCompleteRetiredRead(t, codec, readDone)
	// Observe admission through the transport itself, not a scheduler delay.
	historyAwaitReadSignal(t, observed.entered)
	b.clearProxyHandles()
	historyAwaitReadSignal(t, codec.RetireRead())
	historyAwaitReadSignal(t, readDone)
	historyRequireReadDetached(t, codec)
	if b.historyRetiredThrough < b.historyIncarnation {
		t.Fatal("read retirement preceded source-authority fence")
	}
}

type historyDiscardCloser struct{}

func (historyDiscardCloser) Write(p []byte) (int, error) { return len(p), nil }
func (historyDiscardCloser) Close() error                { return nil }

type historyHeldRead struct {
	entered chan struct{}
	finish  chan struct{}
}

func (r *historyHeldRead) Read([]byte) (int, error) {
	close(r.entered)
	<-r.finish
	return 0, io.EOF
}

func TestRoutingHistoryReadRoleJoinsActiveRead(t *testing.T) {
	b, p, _ := historyFixture(t)
	r := &b.routingHistory.readers
	reader := &historyHeldRead{entered: make(chan struct{}), finish: make(chan struct{})}
	interruptEntered, interruptFinish := make(chan struct{}), make(chan struct{})
	var readOnce, interruptOnce sync.Once
	finishRead := func() { readOnce.Do(func() { close(reader.finish) }) }
	finishInterrupt := func() { interruptOnce.Do(func() { close(interruptFinish) }) }
	t.Cleanup(finishRead)
	t.Cleanup(finishInterrupt)
	if err := p.attachRoutingHistoryPeer(b, readWriter{reader, io.Discard}, func() {
		// Actual interruption runs after the source fence and outside both locks.
		b.workersMu.Lock()
		fenced := b.historyRetiredThrough >= b.historyIncarnation
		b.workersMu.Unlock()
		r.mu.Lock()
		r.mu.Unlock()
		if !fenced {
			t.Error("interrupt ran before source fence")
		}
		close(interruptEntered)
		<-interruptFinish
	}); err != nil {
		t.Fatal(err)
	}
	codec := p.peer.Codec()
	readDone := make(chan struct{})
	go historyCompleteRetiredRead(t, codec, readDone)
	historyAwaitReadSignal(t, reader.entered)
	withdrawn := make(chan struct{})
	go func() { b.clearProxyHandles(); close(withdrawn) }()
	historyAwaitReadSignal(t, interruptEntered)
	finishInterrupt()
	historyAwaitReadSignal(t, withdrawn)
	done := codec.RetireRead()
	historyRequireReadPending(t, done)
	if _, err := r.acquire(historyProxyPrivate, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
		t.Fatalf("active read rebound after interrupt returned: %v", err)
	}
	finishRead()
	historyAwaitReadSignal(t, readDone)
	historyAwaitReadSignal(t, done)
	historyRequireReadDetached(t, codec)
}

type historyHeldOutput struct {
	entered chan struct{}
	finish  chan struct{}
	once    sync.Once
	out     syncBuffer
}

func (w *historyHeldOutput) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	<-w.finish
	return w.out.Write(p)
}

func TestRoutingHistoryLocalRefusalCancellationKeepsBorrow(t *testing.T) {
	b, _, _ := historyFixture(t)
	w := &historyHeldOutput{entered: make(chan struct{}), finish: make(chan struct{})}
	var once sync.Once
	finish := func() { once.Do(func() { close(w.finish) }) }
	t.Cleanup(finish)
	input := `{"jsonrpc":"2.0","id":"old\u0031","method":"routing:subscribe","params":null}` + "\n" +
		`{"jsonrpc":"2.0","id":"must-not-read","method":"version","params":{}}` + "\n"
	r := &b.routingHistory.readers
	codec, err := r.acquire(historyLocalControl, readWriter{strings.NewReader(input), w}, func() {}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b.codec = codec
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	loopDone := make(chan struct{})
	go func() {
		if err := b.readLoop(ctx); err != nil {
			t.Errorf("local read loop: %v", err)
		}
		close(loopDone)
	}()
	historyAwaitReadSignal(t, w.entered)
	cancel()
	done := r.retire(historyLocalControl, codec)
	historyRequireReadPending(t, done)
	if _, err := r.acquire(historyLocalControl, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil); !errors.Is(err, jsonrpc.ErrHistoryBusy) {
		t.Fatalf("cancellation released handed-off refusal ID: %v", err)
	}
	finish()
	historyAwaitReadSignal(t, loopDone)
	historyAwaitReadSignal(t, done)
	historyRequireReadDetached(t, codec)
	frames := w.out.frames()
	if len(frames) != 1 || frames[0].ID == nil || string(*frames[0].ID) != `"old\u0031"` || frames[0].Error == nil || frames[0].Error.Code != jsonrpc.HistoryInvalidCode {
		t.Fatalf("consumed refusal response changed: %+v", frames)
	}
}

func TestRoutingHistoryLocalInitialCopiesBeforeCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, _, _ := historyFixture(t)
		input, feed := io.Pipe()
		t.Cleanup(func() { _ = input.Close(); _ = feed.Close() })
		w := &historyHeldOutput{entered: make(chan struct{}), finish: make(chan struct{})}
		var once sync.Once
		finish := func() { once.Do(func() { close(w.finish) }) }
		t.Cleanup(finish)
		r := &b.routingHistory.readers
		codec, err := r.acquire(historyLocalControl, readWriter{input, w}, func() { _ = input.Close() }, nil)
		if err != nil {
			t.Fatal(err)
		}
		b.codec = codec
		ctx, cancel := context.WithCancel(t.Context())
		t.Cleanup(cancel)
		loopDone := make(chan struct{})
		go func() {
			if err := b.readLoop(ctx); err != nil {
				t.Errorf("local read loop: %v", err)
			}
			close(loopDone)
		}()
		if _, err := io.WriteString(feed, `{"jsonrpc":"2.0","id":"initial\u0031","method":"routing:get-initial","params":{}}`+"\n"); err != nil {
			t.Fatal(err)
		}
		historyAwaitReadSignal(t, w.entered)
		cancel()
		historyAwaitReadSignal(t, loopDone)
		historyAwaitReadSignal(t, r.retire(historyLocalControl, codec))
		if !b.routingHistory.initialBusy.Load() {
			t.Fatal("initial output lease ended at cancellation")
		}
		replacement, err := r.acquire(historyLocalControl, readWriter{strings.NewReader(`{"jsonrpc":"2.0","id":"new","method":"routing:subscribe","params":{}}` + "\n"), io.Discard}, func() {}, nil)
		if err != nil {
			t.Fatalf("owned initial ID pinned retired reader: %v", err)
		}
		msg, release, err := replacement.ReadWithRelease()
		if err != nil || msg == nil || release == nil {
			t.Fatalf("replacement did not overwrite its own input: %v", err)
		}
		release()
		historyRequireReadDetached(t, codec)
		finish()
		synctest.Wait()
		if b.routingHistory.initialBusy.Load() {
			t.Fatal("initial writer did not release after actual output")
		}
		frames := w.out.frames()
		if len(frames) != 1 || frames[0].ID == nil || string(*frames[0].ID) != `"initial\u0031"` || frames[0].Error != nil {
			t.Fatalf("funded initial ID changed after reader reuse: %+v", frames)
		}
		r.retire(historyLocalControl, replacement)
	})
}

func TestRoutingHistoryRawLocalRefusalsPreserveOrdinaryRPC(t *testing.T) {
	b, _, out := historyFixture(t)
	input := `{"jsonrpc":"2.0","id":"private\u0031","method":"routing/request","params":{}}` + "\n" +
		`{"jsonrpc":"2.0","id":"invalid","method":"routing:subscribe","params":null}` + "\n" +
		`{"jsonrpc":"2.0","id":"` + strings.Repeat("x", jsonrpc.HistoryIDBytes) + `","method":"routing:get-initial","params":{}}` + "\n" +
		`{"jsonrpc":"2.0","id":7,"method":"version","params":{}}` + "\n"
	codec, err := b.routingHistory.readers.acquire(historyLocalControl, readWriter{strings.NewReader(input), out}, func() {}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b.codec = codec
	if err := b.readLoop(t.Context()); err != nil {
		t.Fatal(err)
	}
	frames := out.frames()
	if len(frames) != 4 {
		t.Fatalf("refusal/ordinary replies = %d", len(frames))
	}
	for i, want := range []struct {
		id   string
		code int
	}{
		{`"private\u0031"`, -32601}, {`"invalid"`, jsonrpc.HistoryInvalidCode}, {`null`, jsonrpc.HistoryInvalidCode},
	} {
		frame := frames[i]
		// encoding/json represents a literal null ID as a nil pointer.
		id := "null"
		if frame.ID != nil {
			id = string(*frame.ID)
		}
		if id != want.id || frame.Error == nil || frame.Error.Code != want.code || frame.Error.Data != nil {
			t.Fatalf("fixed refusal %d = %+v", i, frame)
		}
	}
	if frames[3].ID == nil || string(*frames[3].ID) != "7" || frames[3].Error != nil || len(frames[3].Result) == 0 {
		t.Fatal("ordinary version RPC did not survive history refusals")
	}
	if b.routingHistory.subscribed || b.routingHistory.initialBusy.Load() {
		t.Fatal("refused controls changed history state")
	}
}

func TestRoutingHistoryRawLocalPublicControlsKeepReplies(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, p, _ := historyFixture(t)
		row := historyRecord("9", 100, 1)
		historySend(t, b, p, "1", row)
		client, server := net.Pipe()
		t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
		codec, err := newRoutingHistoryLocalCodec(b, server, true)
		if err != nil {
			t.Fatal(err)
		}
		b.codec = codec
		done := make(chan struct{})
		go func() {
			if err := b.readLoop(t.Context()); err != nil {
				t.Errorf("local control read loop: %v", err)
			}
			close(done)
		}()
		local := NewCodec(client)
		for _, method := range []string{nodepolicy.MethodRoutingSubscribe, nodepolicy.MethodRoutingGetInitial, nodepolicy.MethodRoutingUnsubscribe} {
			historyFeedFrames(t, client, `{"jsonrpc":"2.0","id":"local\u0031","method":"`+method+`","params":{}}`+"\n")
			reply, err := local.Read()
			if err != nil {
				t.Fatalf("read %s reply: %v", method, err)
			}
			if reply.ID == nil || string(*reply.ID) != `"local\u0031"` || reply.Error != nil {
				t.Fatalf("%s reply = %+v", method, reply)
			}
			var initial nodepolicy.RoutingHistoryInitial
			if err := json.Unmarshal(reply.Result, &initial); err != nil {
				t.Fatalf("decode %s reply: %v", method, err)
			}
			if len(initial.Requests) != 1 || !reflect.DeepEqual(initial.Requests[0], row) {
				t.Fatalf("%s changed local public history", method)
			}
			synctest.Wait() // actual initial writer completion precedes the next call
		}
		if err := client.Close(); err != nil {
			t.Fatal(err)
		}
		historyAwaitReadSignal(t, done)
		if b.routingHistory.subscribed {
			t.Fatal("local unsubscribe failed")
		}
	})
}

func TestRoutingHistoryWorkerRefusalUsesExistingWriter(t *testing.T) {
	client, server := net.Pipe()
	t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
	b, p, _ := historyFixture(t)
	if err := p.attachRoutingHistoryPeer(b, client, func() { _ = client.Close() }); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	b.routingHistory.parent = ctx
	writerDone, pumpDone, staged := make(chan struct{}), make(chan struct{}), make(chan struct{})
	if err := p.peer.SetHistoryReadHandlers(nil, func(f jsonrpc.HistoryFailure) {
		b.routingHistoryFailure(p, f)
		// The worker has not read the reply yet. This boundary is unreachable
		// if the failure callback tries I/O on this peer itself.
		close(staged)
	}); err != nil {
		t.Fatal(err)
	}
	go func() { b.routingHistory.run(); close(writerDone) }()
	go func() { p.servePeer(); close(pumpDone) }()
	worker := NewCodec(server)
	callDone := make(chan struct{})
	go func() {
		raw, rpcErr, err := p.peer.Call(t.Context(), "ordinary-status", json.RawMessage(`{}`))
		var result struct {
			OK bool `json:"ok"`
		}
		if err != nil || rpcErr != nil {
			t.Errorf("ordinary Call after refusal: %v / %v", rpcErr, err)
		} else if err := json.Unmarshal(raw, &result); err != nil {
			t.Errorf("ordinary result: %v", err)
		} else if !result.OK {
			t.Error("ordinary result changed after refusal")
		}
		close(callDone)
	}()
	request, err := worker.Read()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(server, `{"jsonrpc":"2.0","id":"worker\u0031","method":"routing/request","params":{}}`+"\n"); err != nil {
		t.Fatal(err)
	}
	historyAwaitReadSignal(t, staged)
	refusal, err := worker.Read()
	if err != nil {
		t.Fatal(err)
	}
	if refusal.ID == nil || string(*refusal.ID) != `"worker\u0031"` || refusal.Error == nil || refusal.Error.Code != -32601 {
		t.Fatalf("worker private refusal = %+v", refusal)
	}
	if err := worker.Respond(request.ID, json.RawMessage(`{"ok":true}`)); err != nil {
		t.Fatal(err)
	}
	historyAwaitReadSignal(t, callDone)
	b.clearProxyHandles()
	historyAwaitReadSignal(t, pumpDone)
	cancel()
	historyAwaitReadSignal(t, writerDone)
}

type historyWriteStep struct {
	frame  Message
	finish chan struct{}
}

type historySteppedOutput struct {
	steps chan historyWriteStep
	stop  <-chan struct{}
	out   syncBuffer
}

func (w *historySteppedOutput) Write(p []byte) (int, error) {
	var frame Message
	if err := json.Unmarshal(p, &frame); err != nil {
		return 0, err
	}
	step := historyWriteStep{frame: frame, finish: make(chan struct{})}
	select {
	case w.steps <- step:
	case <-w.stop:
		return 0, io.ErrClosedPipe
	}
	select {
	case <-step.finish:
		return w.out.Write(p)
	case <-w.stop:
		return 0, io.ErrClosedPipe
	}
}

func historyNextWrite(t *testing.T, w *historySteppedOutput) historyWriteStep {
	t.Helper()
	select {
	case step := <-w.steps:
		return step
	case <-time.After(5 * time.Second):
		t.Fatal("history writer did not progress")
		return historyWriteStep{}
	}
}

func historyFeedFrames(t *testing.T, feed io.Writer, frames string) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		if _, err := io.WriteString(feed, frames); err != nil {
			t.Errorf("feed worker frames: %v", err)
		}
		close(done)
	}()
	historyAwaitReadSignal(t, done)
}

func TestRoutingHistoryOccupiedRefusalDoesNotBlockWorkerPump(t *testing.T) {
	b, p, _ := historyFixture(t)
	input, feed := io.Pipe()
	t.Cleanup(func() { _ = input.Close(); _ = feed.Close() })
	stop := make(chan struct{})
	t.Cleanup(func() { close(stop) })
	w := &historySteppedOutput{steps: make(chan historyWriteStep), stop: stop}
	if err := p.attachRoutingHistoryPeer(b, readWriter{input, w}, func() { _ = input.Close() }); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	b.routingHistory.parent = ctx
	writerDone, pumpDone, notified := make(chan struct{}), make(chan struct{}), make(chan struct{})
	p.onNotify = func(method string, params json.RawMessage) {
		if method == "ordinary-progress" {
			close(notified)
		}
	}
	logs := &syncBuffer{}
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	go func() { p.servePeer(); close(pumpDone) }()
	callDone := make(chan struct{})
	go func() {
		raw, rpcErr, err := p.peer.Call(t.Context(), "ordinary-status", json.RawMessage(`{}`))
		var result struct {
			OK bool `json:"ok"`
		}
		if err != nil || rpcErr != nil {
			t.Errorf("ordinary pending response = %v / %v", rpcErr, err)
		} else if err := json.Unmarshal(raw, &result); err != nil {
			t.Errorf("decode ordinary pending response: %v", err)
		} else if !result.OK {
			t.Error("ordinary pending response lost its result")
		}
		close(callDone)
	}()
	ordinary := historyNextWrite(t, w)
	if ordinary.frame.Method != "ordinary-status" || ordinary.frame.ID == nil {
		t.Fatal("ordinary request was not written")
	}
	close(ordinary.finish)
	manifest := historyManifest(t, "1", historyRecord("9", 100, 1))
	b.routeRoutingHistoryFrom(p, nodepolicy.NotifyRoutingHistoryWindow, historyMarshal(t, manifest))
	go func() { b.routingHistory.run(); close(writerDone) }()
	historyFeedFrames(t, feed, `{"jsonrpc":"2.0","id":"first\u0031","method":"routing/request","params":{}}`+"\n")
	first := historyNextWrite(t, w)
	if first.frame.ID == nil || string(*first.frame.ID) != `"first\u0031"` || first.frame.Error == nil || first.frame.Error.Code != -32601 || first.frame.Error.Data != nil {
		t.Fatalf("admitted worker refusal = %+v", first.frame)
	}
	// Both well-formed public controls and raw typed refusals must pass the
	// same nonwaiting admission. The second and subsequent requests cannot
	// pin another borrowed frame while the first actual Write remains held.
	var flood strings.Builder
	for i := 0; i < 64; i++ {
		flood.WriteString(`{"jsonrpc":"2.0","id":"synthetic-secret-id","method":"routing:subscribe","params":{}}` + "\n")
		flood.WriteString(`{"jsonrpc":"2.0","id":"synthetic-secret-id","method":"routing/request","params":{}}` + "\n")
	}
	flood.WriteString(`{"jsonrpc":"2.0","id":` + string(*ordinary.frame.ID) + `,"result":{"ok":true}}` + "\n")
	flood.WriteString(`{"jsonrpc":"2.0","method":"ordinary-progress","params":{}}` + "\n")
	historyFeedFrames(t, feed, flood.String())
	historyAwaitReadSignal(t, callDone)
	historyAwaitReadSignal(t, notified)
	h := b.routingHistory
	h.mu.Lock()
	if !h.refusalActive || h.refusal.source != p || string(h.refusal.id[:h.refusal.n]) != `"first\u0031"` || h.subscribed || h.refetchSource != nil || h.accepted.count != 0 || h.stageSource != p || h.stage.count != 1 || h.stageGeneration != "1" {
		t.Error("occupied slot was overwritten or wrong-direction requests changed history state")
	}
	h.mu.Unlock()
	logs.mu.Lock()
	captured := logs.buf.String()
	logs.mu.Unlock()
	decoder := json.NewDecoder(strings.NewReader(captured))
	discarded := 0
	for decoder.More() {
		var entry struct {
			Category string `json:"category"`
			Count    int    `json:"count"`
		}
		if err := decoder.Decode(&entry); err != nil {
			t.Fatalf("decode fixed diagnostic: %v", err)
		}
		if entry.Category == "history-worker-refusal-busy" {
			if entry.Count != 1 {
				t.Fatal("refusal discard diagnostic has a variable count")
			}
			discarded++
		}
	}
	if discarded != 128 || strings.Contains(captured, "synthetic-secret-id") {
		t.Fatalf("discard diagnostics = %d, want 128 without caller text", discarded)
	}
	withdrawn := make(chan struct{})
	go func() { b.clearProxyHandles(); close(withdrawn) }()
	historyAwaitReadSignal(t, withdrawn)
	historyAwaitReadSignal(t, pumpDone)
	historyAwaitReadSignal(t, p.peer.Codec().RetireRead())
	historyRequireReadDetached(t, p.peer.Codec())
	select {
	case <-writerDone:
		t.Fatal("reader retirement joined or interrupted history output")
	default:
	}
	if _, err := h.readers.acquire(historyProxyPrivate, readWriter{strings.NewReader(""), io.Discard}, func() {}, nil); err != nil {
		t.Fatalf("stalled old refusal output pinned reader role: %v", err)
	}
	replacement := historyActivate(t, b, "run-b")
	replacement.peer = NewPeer(NewCodec(&syncBuffer{}))
	h.stageRefusal(replacement, &jsonrpc.HistoryRequestRefusal{ReplyID: []byte(`"replacement"`), Code: -32601, Message: "method not found"})
	h.mu.Lock()
	h.retireLocked()
	if !h.refusalActive || h.refusal.source != p {
		t.Error("replacement/fence released an old active Write")
	}
	h.mu.Unlock()
	cancel()
	select {
	case <-writerDone:
		t.Fatal("cancellation released the refusal before actual Write returned")
	default:
	}
	close(first.finish)
	historyAwaitReadSignal(t, writerDone)
	h.mu.Lock()
	if h.refusal.source != nil || h.refusalActive || h.refusal.n != 0 || h.refusal.id != [jsonrpc.HistoryIDBytes]byte{} {
		t.Error("completed refusal retains its caller ID/source")
	}
	h.mu.Unlock()
}

func TestRoutingHistoryRefusalRefillPreservesAckAndPushTurns(t *testing.T) {
	b, p, _ := historyFixture(t)
	input, feed := io.Pipe()
	t.Cleanup(func() { _ = input.Close(); _ = feed.Close() })
	stop := make(chan struct{})
	t.Cleanup(func() { close(stop) })
	w := &historySteppedOutput{steps: make(chan historyWriteStep), stop: stop}
	b.codec = NewCodec(readWriter{strings.NewReader(""), w})
	if err := p.attachRoutingHistoryPeer(b, readWriter{input, w}, func() { _ = input.Close() }); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	h := b.routingHistory
	h.parent = ctx
	row := historyRecord("9", 100, 1)
	historySend(t, b, p, "1", row)
	h.subscribed = true
	pumpDone, writerDone := make(chan struct{}), make(chan struct{})
	consumed := make(chan struct{})
	p.onNotify = func(method string, params json.RawMessage) {
		if method == "ordinary-barrier" {
			consumed <- struct{}{}
		}
	}
	go func() { p.servePeer(); close(pumpDone) }()
	refill := func() {
		t.Helper()
		historyFeedFrames(t, feed, `{"jsonrpc":"2.0","id":"wrong-way","method":"routing:unsubscribe","params":{}}`+"\n"+`{"jsonrpc":"2.0","method":"ordinary-barrier","params":{}}`+"\n")
		historyAwaitReadSignal(t, consumed)
	}
	refill()
	go func() { h.run(); close(writerDone) }()
	// Refill during every actual Write. Each non-refusal turn must still
	// alternate acknowledgement and accepted data; no unbounded input race
	// can give prohibited requests unconditional writer priority.
	for _, want := range []string{"refusal", nodepolicy.NotifyRoutingHistoryAccepted, "refusal", nodepolicy.NotifyRoutingChanged} {
		step := historyNextWrite(t, w)
		if want == "refusal" {
			if step.frame.Error == nil || step.frame.Error.Code != -32601 {
				t.Fatalf("expected refusal, got %+v", step.frame)
			}
		} else if step.frame.Method != want {
			t.Fatalf("writer turn = %q, want %q", step.frame.Method, want)
		}
		refill()
		close(step.finish)
	}
	cancel()
	// A final already-selected refusal may own Write when cancellation wins.
	// The test output's stop is its explicit completion boundary.
	b.clearProxyHandles()
	historyAwaitReadSignal(t, pumpDone)
	select {
	case step := <-w.steps:
		close(step.finish)
	case <-writerDone:
	}
	historyAwaitReadSignal(t, writerDone)
}

func TestRoutingHistoryLocalTransportReadOwners(t *testing.T) {
	t.Run("IPC connection", func(t *testing.T) {
		client, server := net.Pipe()
		t.Cleanup(func() { _ = client.Close(); _ = server.Close() })
		b, _, _ := historyFixture(t)
		codec, err := newRoutingHistoryLocalCodec(b, client, true)
		if err != nil {
			t.Fatal(err)
		}
		readDone := make(chan struct{})
		go historyCompleteRetiredRead(t, codec, readDone)
		// The peer's first byte proves that Read is active without requiring
		// a completed frame or relying on a scheduling delay.
		if _, err := server.Write([]byte("{")); err != nil {
			t.Fatal(err)
		}
		historyAwaitReadSignal(t, b.routingHistory.readers.retire(historyLocalControl, codec))
		historyAwaitReadSignal(t, readDone)
		historyRequireReadDetached(t, codec)
	})
	t.Run("stdio no-op transport close", func(t *testing.T) {
		input, output, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		var readDone, retireDone chan struct{}
		t.Cleanup(func() {
			// Release the writer first so even a failed Windows read interrupt
			// cannot strand Close or a fixture goroutine during failure cleanup.
			_ = output.Close()
			_ = input.Close()
			if readDone != nil {
				historyAwaitReadSignal(t, readDone)
			}
			if retireDone != nil {
				historyAwaitReadSignal(t, retireDone)
			}
		})
		b, _, _ := historyFixture(t)
		owned, err := pipeio.NewReader(input)
		if err != nil {
			t.Fatal(err)
		}
		observed := &historyObservedReadCloser{ReadCloser: owned, entered: make(chan struct{})}
		transport := &stdioTransport{Reader: observed, Writer: historyDiscardCloser{}}
		// The seam closes the same captured reader that the codec reads through.
		codec, err := newRoutingHistoryLocalCodec(b, transport, false)
		if err != nil {
			t.Fatal(err)
		}
		readDone = make(chan struct{})
		go historyCompleteRetiredRead(t, codec, readDone)
		historyAwaitReadSignal(t, observed.entered)
		if err := transport.Close(); err != nil {
			t.Fatal(err)
		}
		select {
		case <-readDone:
			t.Fatal("fixture Close unexpectedly interrupted stdin")
		default:
		}
		retired := make(chan (<-chan struct{}), 1)
		retireDone = make(chan struct{})
		go func() {
			defer close(retireDone)
			retired <- b.routingHistory.readers.retire(historyLocalControl, codec)
		}()
		select {
		case done := <-retired:
			historyAwaitReadSignal(t, done)
		case <-time.After(5 * time.Second):
			t.Fatal("stdin read interruption did not return")
		}
		historyAwaitReadSignal(t, readDone)
		historyRequireReadDetached(t, codec)
	})
}
