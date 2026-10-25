// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package rpc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

// newPair wires a Client to an in-memory server side over a full-duplex
// net.Pipe, runs the client read loop, and returns the client, a codec
// for the server end, and that end's connection (close it to simulate the
// broker disconnecting).
func newPair(t *testing.T) (*Client, *Codec, net.Conn) {
	t.Helper()
	c1, c2 := net.Pipe()
	client := NewClient(c1, c1)
	ctx, cancel := context.WithCancel(context.Background())
	go client.Run(ctx)
	t.Cleanup(func() {
		cancel()
		c1.Close()
		c2.Close()
	})
	return client, NewCodec(c2, c2), c2
}

func TestCodecRoundTrip(t *testing.T) {
	c1, c2 := net.Pipe()
	defer c1.Close()
	defer c2.Close()
	a := NewCodec(c1, c1)
	b := NewCodec(c2, c2)

	go func() {
		_ = a.Write(&Message{JSONRPC: "2.0", Method: "hello", Params: json.RawMessage(`{"x":1}`)})
	}()

	msg, err := b.Read()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !msg.IsNotification() || msg.Method != "hello" {
		t.Fatalf("unexpected frame: %+v", msg)
	}
	if string(msg.Params) != `{"x":1}` {
		t.Fatalf("params = %s", msg.Params)
	}
}

func TestClientCallMatchesResponse(t *testing.T) {
	client, server, _ := newPair(t)

	// Server: read the request and echo a result tagged with its id.
	go func() {
		req, err := server.Read()
		if err != nil {
			return
		}
		_ = server.Write(&Message{JSONRPC: "2.0", ID: req.ID, Result: json.RawMessage(`{"pong":true}`)})
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	resp, err := client.Call(ctx, "ping", map[string]string{"a": "b"})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if string(resp.Result) != `{"pong":true}` {
		t.Fatalf("result = %s", resp.Result)
	}
}

func TestClientCallSurfacesRPCError(t *testing.T) {
	client, server, _ := newPair(t)
	go func() {
		req, err := server.Read()
		if err != nil {
			return
		}
		_ = server.Write(&Message{JSONRPC: "2.0", ID: req.ID, Error: &RPCError{Code: -32000, Message: "boom"}})
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err := client.Call(ctx, "explode", nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if rpcErr, ok := err.(*RPCError); !ok || rpcErr.Code != -32000 {
		t.Fatalf("expected *RPCError -32000, got %v", err)
	}
}

// recordingWriter counts writes, to show a refused request never reached the
// wire.
type recordingWriter struct{ writes int }

func (w *recordingWriter) Write(p []byte) (int, error) {
	w.writes++
	return len(p), nil
}

// TestClientCallRefusesBeforeWriting checks a request is not sent when its
// caller has already given up or the connection is gone. Sent, the broker
// would carry it out regardless — for a mutation, while the caller reported
// that it had not happened.
func TestClientCallRefusesBeforeWriting(t *testing.T) {
	w := &recordingWriter{}
	client := NewClient(strings.NewReader(""), w)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := client.Call(ctx, "engine:apply-settings", nil); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled call returned %v, want context.Canceled", err)
	}
	if w.writes != 0 {
		t.Errorf("a cancelled call wrote %d frame(s)", w.writes)
	}

	// Run returns at once on the empty stream, which closes the connection.
	_ = client.Run(context.Background())
	if _, err := client.Call(context.Background(), "engine:apply-settings", nil); err == nil {
		t.Error("a call on a closed connection returned no error")
	}
	if w.writes != 0 {
		t.Errorf("a call on a closed connection wrote %d frame(s)", w.writes)
	}
}

func TestClientDeliversNotifications(t *testing.T) {
	client, server, _ := newPair(t)
	go func() {
		_ = server.Write(&Message{JSONRPC: "2.0", Method: "errors:update", Params: json.RawMessage(`[]`)})
	}()

	select {
	case event, ok := <-client.Events():
		if !ok {
			t.Fatal("event stream closed")
		}
		if event.Kind != EventNotification || event.Generation != 1 || event.Message == nil || event.Message.Method != "errors:update" {
			t.Fatalf("notification event = %+v", event)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for notification")
	}
}

func TestNotificationFilterConsumesMatchingFrames(t *testing.T) {
	c1, c2 := net.Pipe()
	t.Cleanup(func() {
		c1.Close()
		c2.Close()
	})
	client := NewClient(c1, c1)
	consumed := make(chan string, 1)
	client.SetNotificationFilter(func(m *Message) bool {
		if m.Method == "service/log" {
			consumed <- string(m.Params)
			return true
		}
		return false
	})
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go client.Run(ctx)

	server := NewCodec(c2, c2)
	go func() {
		_ = server.Write(&Message{JSONRPC: "2.0", Method: "service/log", Params: json.RawMessage(`{"text":"x"}`)})
		_ = server.Write(&Message{JSONRPC: "2.0", Method: "app:ready", Params: json.RawMessage(`{}`)})
	}()

	select {
	case got := <-consumed:
		if got != `{"text":"x"}` {
			t.Fatalf("filter saw %s", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("filter never ran")
	}
	select {
	case event := <-client.Events():
		if event.Kind != EventNotification || event.Generation != 1 || event.Message == nil || event.Message.Method != "app:ready" {
			t.Fatalf("first queued notification = %+v; the filtered one leaked", event)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("unfiltered notification not delivered")
	}
}

func TestClientDisconnectClosesEvents(t *testing.T) {
	client, _, srv := newPair(t)
	srv.Close() // broker drops the connection -> client read loop hits EOF

	done := make(chan struct{})
	go func() {
		// Drain until the channel closes; a closed channel yields the
		// zero value with ok=false.
		for range client.Events() {
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("event stream was not closed after disconnect")
	}
}

func historyTestClient(t *testing.T, r io.Reader, w io.Writer, consume func(uint64, bool, []byte) error) *Client {
	t.Helper()
	c, err := NewHistoryClient(r, w, consume, func(uint64) {})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func historyTestReconnectingClient(t *testing.T, conn io.ReadWriteCloser, dial func(context.Context) (io.ReadWriteCloser, error), consume func(uint64, bool, []byte) error) *Client {
	t.Helper()
	c, err := NewReconnectingHistoryClient(conn, dial, consume, func(uint64) {})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func historyPair(t *testing.T) (*Client, *Codec, net.Conn) {
	t.Helper()
	a, b := sessionPipe(t)
	c := historyTestClient(t, a, a, func(uint64, bool, []byte) error { return nil })
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	return c, NewCodec(b, b), b
}

func historyRequest(t *testing.T, c *Client, s *Codec) *Message {
	t.Helper()
	c.SetHistorySubscription(true)
	request, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if request.Method != nodepolicy.MethodRoutingSubscribe || request.ID == nil {
		t.Fatalf("request = %+v", request)
	}
	var id string
	if err := json.Unmarshal(*request.ID, &id); err != nil || id != "jh:1" {
		t.Fatalf("generated first ID = %s, %v", *request.ID, err)
	}
	return request
}

func historyBarrier(t *testing.T, c *Client, s *Codec) {
	t.Helper()
	result := make(chan error, 1)
	go func() { _, err := c.Call(context.Background(), "ordinary-barrier", nil); result <- err }()
	request, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	var id int64
	if request.ID == nil || json.Unmarshal(*request.ID, &id) != nil || request.Method != "ordinary-barrier" {
		t.Fatalf("ordinary call lost numeric correlation: %+v", request)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{}`)}); err != nil {
		t.Fatal(err)
	}
	if err := sessionResult(t, result); err != nil {
		t.Fatal(err)
	}
}

func TestHistoryGeneratedQuarantineNoLateAOrdinarySurvivesB(t *testing.T) {
	test := func(name, candidates string) {
		t.Run(name, func(t *testing.T) {
			c, s, wire := historyPair(t)
			historyRequest(t, c, s)
			if _, err := wire.Write([]byte(`{"jsonrpc":"2.0",` + candidates + `,"result":{"requests":[]}}` + "\n")); err != nil {
				t.Fatal(err)
			}
			historyBarrier(t, c, s)
			b, err := s.Read()
			if err != nil {
				t.Fatal(err)
			}
			if b.ID == nil || string(*b.ID) != `"jh:2"` {
				t.Fatalf("B waited for absent A: %+v", b)
			}
			if err := s.Write(&Message{JSONRPC: "2.0", ID: b.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
				t.Fatal(err)
			}
			historyBarrier(t, c, s)
			state := c.HistoryState()
			if state.Loading || state.Quarantined {
				t.Fatalf("applied B did not restore history: %+v", state)
			}
			if _, err := wire.Write([]byte("{bad\n")); err != nil {
				t.Fatal(err)
			}
			historyBarrier(t, c, s)
			c.history.mu.Lock()
			c.history.retryAt = time.Time{}
			c.history.signal()
			c.history.mu.Unlock()
			newer, err := s.Read()
			if err != nil {
				t.Fatal(err)
			}
			if newer.ID == nil || string(*newer.ID) != `"jh:3"` {
				t.Fatalf("newer pending = %+v", newer)
			}
			for _, id := range []string{`"jh:1"`, `"jh:2"`, `"jh:99"`, `"jh:01"`} {
				if _, err := wire.Write([]byte(`{"jsonrpc":"2.0","id":` + id + `,"result":{"requests":[]}}` + "\n")); err != nil {
					t.Fatal(err)
				}
			}
			historyBarrier(t, c, s)
			if state := c.HistoryState(); !state.Quarantined || !state.Pending {
				t.Fatalf("late A/B changed newer pending/quarantine: %+v", state)
			}
			if err := s.Write(&Message{JSONRPC: "2.0", ID: newer.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
				t.Fatal(err)
			}
			historyBarrier(t, c, s)
		})
	}
	test("history ID first", `"id":"jh:1","\u0069d":7`)
	test("history ID last", `"id":7,"\u0069d":"\u006ah:1"`)
}

func TestHistoryBusyUsesOneLoadingRetry(t *testing.T) {
	c, s, _ := historyPair(t)
	request := historyRequest(t, c, s)
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Error: &RPCError{Code: jsonrpc.HistoryBusyCode, Message: jsonrpc.HistoryBusyMessage}}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	state := c.HistoryState()
	if !state.Loading || state.Pending || !state.Retry {
		t.Fatalf("busy state = %+v", state)
	}
	c.history.mu.Lock()
	deadline := c.history.retryAt
	c.history.mu.Unlock()
	c.SetHistorySubscription(true)
	c.history.mu.Lock()
	sameDeadline := c.history.retryAt == deadline
	c.history.mu.Unlock()
	if !sameDeadline {
		t.Fatal("repeated subscription bypassed busy backoff")
	}
	c.SetHistorySubscription(false)
	if c.HistoryState().Retry {
		t.Fatal("unsubscribe retained retry timer")
	}
	unsubscribe, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if unsubscribe.Method != nodepolicy.MethodRoutingUnsubscribe {
		t.Fatalf("unsubscribe = %+v", unsubscribe)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: unsubscribe.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	for _, tc := range []struct {
		attempt int
		want    time.Duration
	}{{0, 250 * time.Millisecond}, {1, 500 * time.Millisecond}, {2, time.Second}, {100, time.Second}} {
		if got := historyRetryDelay(tc.attempt); got != tc.want {
			t.Fatalf("retry %d = %v", tc.attempt, got)
		}
	}
}

func TestHistoryQuarantinePreservesPushBeforeDelayedInitial(t *testing.T) {
	a, b := sessionPipe(t)
	var newest int64
	pushes := 0
	// The real packed merge is covered in ui; this scalar consumer records
	// which validated revisions actually reach that synchronous seam.
	c := historyTestClient(t, a, a, func(_ uint64, initial bool, raw []byte) error {
		if initial {
			var baseline nodepolicy.RoutingHistoryInitial
			if err := json.Unmarshal(raw, &baseline); err != nil {
				return err
			}
			for _, row := range baseline.Requests {
				newest = max(newest, row.Seq)
			}
		} else {
			var row nodepolicy.RoutingHistoryRecord
			if err := json.Unmarshal(raw, &row); err != nil {
				return err
			}
			newest = max(newest, row.Seq)
			pushes++
		}
		return nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = c.Run(ctx) }()
	s := NewCodec(b, b)
	request := historyRequest(t, c, s)
	row := nodepolicy.RoutingHistoryRecord{Schema: 1, Request: nodepolicy.RoutingRequestKey{OriginNodeID: "synthetic-origin", ProxyRunID: "synthetic-run", RequestID: "1"}, Seq: 3, Requested: nodepolicy.RoutingHistoryRequested{Mode: "other"}, State: "routing", Spend: nodepolicy.RoutingHistorySpend{Currency: "USD", Accounting: "none"}, Attempts: []nodepolicy.RoutingHistoryAttempt{}}
	baseline, err := json.Marshal(nodepolicy.RoutingHistoryInitial{Requests: []nodepolicy.RoutingHistoryRecord{row}})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: baseline}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	if _, err := b.Write([]byte("{bad\n")); err != nil {
		t.Fatal(err)
	}
	refetch := historyReadMessage(t, s)
	if refetch.Method != nodepolicy.MethodRoutingGetInitial {
		t.Fatalf("refetch = %+v", refetch)
	}
	row.Seq = 4
	push, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", Method: nodepolicy.NotifyRoutingChanged, Params: push}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	if state := c.HistoryState(); !state.Quarantined || !state.Loading || !state.Pending {
		t.Fatalf("push prematurely restored history: %+v", state)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: refetch.ID, Result: baseline}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	if newest != row.Seq || pushes != 1 {
		t.Fatalf("delayed baseline lost intervening push: newest=%d pushes=%d", newest, pushes)
	}
	if state := c.HistoryState(); state.Quarantined || state.Loading || state.Pending || state.Retry {
		t.Fatalf("successful baseline did not restore history: %+v", state)
	}
}

func TestHistoryRepeatedSubscriptionKeepsLoadedBaseline(t *testing.T) {
	c, s, _ := historyPair(t)
	request := historyRequest(t, c, s)
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	c.SetHistorySubscription(true)
	historyBarrier(t, c, s)
	if state := c.HistoryState(); state.Loading || state.Pending || state.Retry {
		t.Fatalf("repeated subscription restarted a loaded baseline: %+v", state)
	}
}

func TestHistoryExhaustionStopsRetryOnly(t *testing.T) {
	c, s, _ := historyPair(t)
	c.history.mu.Lock()
	c.history.failed(jsonrpc.ErrHistoryExhausted)
	c.history.mu.Unlock()
	c.SetHistorySubscription(true)
	historyBarrier(t, c, s)
	state := c.HistoryState()
	if !state.Exhausted || state.Retry || state.Pending || state.Loading {
		t.Fatalf("exhaustion state = %+v", state)
	}
}

func TestHistoryResubscribeAfterUnsubscribeError(t *testing.T) {
	c, s, _ := historyPair(t)
	request := historyRequest(t, c, s)
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	c.SetHistorySubscription(false)
	unsubscribe, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if unsubscribe.Method != nodepolicy.MethodRoutingUnsubscribe {
		t.Fatalf("unsubscribe = %+v", unsubscribe)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: unsubscribe.ID, Error: &RPCError{Code: jsonrpc.HistoryInvalidCode, Message: "history-invalid"}}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	c.SetHistorySubscription(true)
	subscribe, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if subscribe.Method != nodepolicy.MethodRoutingSubscribe {
		t.Fatalf("uncertain unsubscribe skipped resubscription: %+v", subscribe)
	}
}

func TestHistoryIncomingRequestDoesNotSettleOutbound(t *testing.T) {
	c, s, wire := historyPair(t)
	request := historyRequest(t, c, s)
	if _, err := wire.Write([]byte("{\"jsonrpc\":\"2.0\",\"id\":\"jh:1\",\"method\":\"ordinary-call\",\"params\":{}}\n")); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	if !c.HistoryState().Pending {
		t.Fatal("incoming ordinary request settled history")
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	if c.HistoryState().Loading {
		t.Fatal("current result not applied")
	}
}

func TestHistoryMalformedValueFailsOnlyHistory(t *testing.T) {
	c, s, wire := historyPair(t)
	request := historyRequest(t, c, s)
	frame := `{"jsonrpc":"2.0","id":` + string(*request.ID) + `,"result":{"requests":[],"synthetic-secret":{"k":1,"\u006b":2}}}` + "\n"
	if _, err := wire.Write([]byte(frame)); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	if state := c.HistoryState(); state.Pending || !state.Loading || !state.Retry {
		t.Fatalf("history failure = %+v", state)
	}
}

func TestHistoryBusyDoesNotRestoreQuarantine(t *testing.T) {
	c, s, wire := historyPair(t)
	historyRequest(t, c, s)
	if _, err := wire.Write([]byte("{bad\n")); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	request, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Error: &RPCError{Code: jsonrpc.HistoryBusyCode, Message: jsonrpc.HistoryBusyMessage}}); err != nil {
		t.Fatal(err)
	}
	historyBarrier(t, c, s)
	if state := c.HistoryState(); !state.Quarantined || !state.Loading || !state.Retry {
		t.Fatalf("busy restored quarantine: %+v", state)
	}
}

func TestHistoryRefusalPreservesRawIncomingToken(t *testing.T) {
	c, s, wire := historyPair(t)
	if _, err := wire.Write([]byte(`{"jsonrpc":"2.0","id":"incoming\u002did","method":"routing/request","params":{}}` + "\n")); err != nil {
		t.Fatal(err)
	}
	reply, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	if reply.ID == nil || string(*reply.ID) != `"incoming\u002did"` || reply.Error == nil || reply.Error.Code != -32601 || reply.Error.Message != "method not found" {
		t.Fatalf("refusal = %+v", reply)
	}
	historyBarrier(t, c, s)
}

func TestHistoryDirtyTokensRemainCoalesced(t *testing.T) {
	c, s, _ := historyPair(t)
	request := historyRequest(t, c, s)
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	row := nodepolicy.RoutingHistoryRecord{Schema: 1, Request: nodepolicy.RoutingRequestKey{OriginNodeID: "synthetic-origin", ProxyRunID: "synthetic-run", RequestID: "1"}, Requested: nodepolicy.RoutingHistoryRequested{Mode: "other"}, State: "routing", Spend: nodepolicy.RoutingHistorySpend{Currency: "USD", Accounting: "none"}, Attempts: []nodepolicy.RoutingHistoryAttempt{}}
	for i := 1; i <= 500; i++ {
		row.Seq = int64(i)
		raw, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Write(&Message{JSONRPC: "2.0", Method: nodepolicy.NotifyRoutingChanged, Params: raw}); err != nil {
			t.Fatal(err)
		}
	}
	historyBarrier(t, c, s)
	if len(c.history.ready) != 1 || len(c.events) != 0 {
		t.Fatal("history accumulated payload events instead of one scalar token")
	}
}

func TestHistoryRefusalEncodedIDBounds(t *testing.T) {
	check := func(name, id string, wantCode int, nullID bool) {
		t.Run(name, func(t *testing.T) {
			c, s, wire := historyPair(t)
			if _, err := wire.Write([]byte(`{"jsonrpc":"2.0","id":` + id + `,"method":"routing/request","params":{}}` + "\n")); err != nil {
				t.Fatal(err)
			}
			reply, err := s.Read()
			if err != nil {
				t.Fatal(err)
			}
			wantID := id
			if nullID {
				wantID = "null"
			}
			// The ordinary JSON decoder represents an explicit null ID as nil.
			if nullID {
				if reply.ID != nil {
					t.Fatal("over-cap ID was echoed")
				}
			} else if reply.ID == nil || string(*reply.ID) != wantID {
				t.Fatal("admitted raw ID changed")
			}
			if reply.Error == nil || reply.Error.Code != wantCode {
				t.Fatalf("refusal code = %+v", reply.Error)
			}
			historyBarrier(t, c, s)
		})
	}
	check("encoded 512 preserved", `"`+strings.Repeat("a", 510)+`"`, -32601, false)
	check("encoded 513 null refusal", `"`+strings.Repeat("a", 511)+`"`, jsonrpc.HistoryInvalidCode, true)
	check("escaped encoded 512 preserved", `"`+strings.Repeat(`\u0061`, 85)+`"`, -32601, false)
	check("escaped encoded 513 null refusal", `"`+strings.Repeat(`\u0061`, 85)+`a"`, jsonrpc.HistoryInvalidCode, true)
}

func TestHistoryOrdinaryCallCannotBorrowHistory(t *testing.T) {
	w := &recordingWriter{}
	c := historyTestClient(t, strings.NewReader(""), w, func(uint64, bool, []byte) error { return nil })
	if _, err := c.Call(context.Background(), nodepolicy.MethodRoutingGetInitial, nil); !errors.Is(err, jsonrpc.ErrHistoryInvalid) {
		t.Fatalf("ordinary history call = %v", err)
	}
	if w.writes != 0 {
		t.Fatal("ordinary call transmitted history without a consume owner")
	}
}

func TestHistoryFactoryPreservesOrdinaryCodecShapeAndEagerRead(t *testing.T) {
	type baselineCodec struct {
		scanner *bufio.Scanner
		writer  io.Writer
		wmu     sync.Mutex
	}
	actual, baseline := reflect.TypeOf(Codec{}), reflect.TypeOf(baselineCodec{})
	if actual.Size() != baseline.Size() || actual.Align() != baseline.Align() || actual.NumField() != baseline.NumField() {
		t.Fatal("ordinary Codec gained a history tail")
	}
	for i := 0; i < baseline.NumField(); i++ {
		if actual.Field(i).Name != baseline.Field(i).Name || actual.Field(i).Type != baseline.Field(i).Type {
			t.Fatal("ordinary Codec field shape changed")
		}
	}
	c := NewCodec(strings.NewReader("{bad\n"+`{"jsonrpc":"2.0","method":"ordinary"}`+"\n"), discardWriter{})
	if c.scanner == nil {
		t.Fatal("ordinary reader is not eagerly constructed")
	}
	if _, err := c.Read(); err == nil {
		t.Fatal("malformed ordinary frame accepted")
	}
	msg, err := c.Read()
	if err != nil {
		t.Fatal(err)
	}
	if msg.Method != "ordinary" {
		t.Fatalf("ordinary framing did not continue: %+v", msg)
	}
}

type historyCountingReader struct {
	io.Reader
	reads int
}

func (r *historyCountingReader) Read(p []byte) (int, error) { r.reads++; return r.Reader.Read(p) }

func TestHistoryFactoryOwnsReaderBeforeFirstRead(t *testing.T) {
	r := &historyCountingReader{Reader: strings.NewReader(`{"jsonrpc":"2.0","method":"ordinary","params":{"owned":true}}` + "\n")}
	resets := 0
	c, err := NewHistoryClient(r, discardWriter{}, func(uint64, bool, []byte) error { return nil }, func(uint64) { resets++ })
	if err != nil {
		t.Fatal(err)
	}
	if c.current.codec.scanner != nil || c.history.readerGeneration != 1 || c.history.t != c.current || resets != 1 || r.reads != 0 {
		t.Fatal("history factory published a competing or incomplete reader")
	}
	if _, err := c.current.codec.Read(); !errors.Is(err, ErrStreamBroken) || r.reads != 0 {
		t.Fatalf("direct history Codec.Read touched backing: %v", err)
	}
	msg, err := c.readMessage(c.current)
	if err != nil {
		t.Fatal(err)
	}
	if msg == nil || msg.Method != "ordinary" || r.reads == 0 {
		t.Fatal("session did not own the actual initial input")
	}
	c.Close()
}

func TestHistoryFactoryRejectsMissingConsumerBeforeConstruction(t *testing.T) {
	r := &historyCountingReader{Reader: strings.NewReader("")}
	w := &recordingWriter{}
	if _, err := NewHistoryClient(r, w, nil, func(uint64) {}); !errors.Is(err, jsonrpc.ErrHistoryInvalid) {
		t.Fatalf("missing consume = %v", err)
	}
	if r.reads != 0 || w.writes != 0 {
		t.Fatal("invalid factory used transport backing")
	}
}

func TestHistoryOrdinaryFactoryKeepsNumericCalls(t *testing.T) {
	c, s, _ := newPair(t)
	result := make(chan error, 1)
	go func() { _, err := c.Call(context.Background(), nodepolicy.MethodRoutingGetInitial, nil); result <- err }()
	request, err := s.Read()
	if err != nil {
		t.Fatal(err)
	}
	var numeric int64
	if request.ID == nil || json.Unmarshal(*request.ID, &numeric) != nil {
		t.Fatal("ordinary constructor changed numeric calls")
	}
	if err := s.Write(&Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(`{"requests":[]}`)}); err != nil {
		t.Fatal(err)
	}
	if err := sessionResult(t, result); err != nil {
		t.Fatal(err)
	}
}
