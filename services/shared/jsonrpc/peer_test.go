// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package jsonrpc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// peerHistoryStream holds the terminal read until the test has observed live
// response delivery. It uses no socket or process and never retains writes.
type peerHistoryStream struct {
	io.Reader
	terminal chan struct{}
	written  chan struct{}
	err      error
}

func (s *peerHistoryStream) Read(b []byte) (int, error) {
	n, err := s.Reader.Read(b)
	if n == 0 && err == io.EOF {
		<-s.terminal
		return 0, s.err
	}
	return n, err
}

func (s *peerHistoryStream) Write(b []byte) (int, error) {
	select {
	case s.written <- struct{}{}:
	default:
	}
	return len(b), nil
}

func newPeerHistoryStream(t *testing.T, input string, terminalErr error) (*peerHistoryStream, func()) {
	t.Helper()
	s := &peerHistoryStream{Reader: strings.NewReader(input), terminal: make(chan struct{}), written: make(chan struct{}, 1), err: terminalErr}
	var once sync.Once
	release := func() { once.Do(func() { close(s.terminal) }) }
	t.Cleanup(release)
	return s, release
}

func newHistoryTestPeer(t *testing.T, stream io.ReadWriter, correlation *HistoryCorrelation) *Peer {
	t.Helper()
	codec, err := NewHistoryCodec(stream, 1<<20, make([]byte, 0, HistoryFrameBytes), make([]byte, 0, HistoryFrameBytes), make([]uint32, HistoryInitialBytes/4), correlation, historyTestSchema, nil)
	if err != nil {
		t.Fatal(err)
	}
	return NewPeer(codec)
}

func TestPeerHistoryFailuresPreserveOrdinaryCall(t *testing.T) {
	test := func(name, frame string, want HistoryFailure) {
		t.Run(name, func(t *testing.T) {
			var correlation HistoryCorrelation
			if err := correlation.Begin([]byte(`"history"`)); err != nil {
				t.Fatal(err)
			}
			input := frame + "\n" + `{"jsonrpc":"2.0","id":1,"result":true}` + "\n" + `{"jsonrpc":"2.0","method":"ordinary","params":null}` + "\n"
			stream, release := newPeerHistoryStream(t, input, io.EOF)
			peer := newHistoryTestPeer(t, stream, &correlation)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			call := make(chan error, 1)
			go func() {
				result, rpcErr, err := peer.Call(ctx, "ordinary", nil)
				if err == nil && (rpcErr != nil || string(result) != "true") {
					err = errors.New("ordinary result changed")
				}
				call <- err
			}()
			select {
			case <-stream.written:
			case <-ctx.Done():
				t.Fatal("ordinary request was not written")
			}
			failures := make(chan HistoryFailure, 2)
			notified := make(chan bool, 1)
			historyFailed := false
			go peer.ServeWithHistoryFailures(nil, func(method string, _ json.RawMessage) {
				notified <- method == "ordinary" && len(failures) == 1 && historyFailed == want.FailPending
			}, func(failure HistoryFailure) {
				// Taking mu here proves callbacks do not run under the peer lock.
				peer.mu.Lock()
				peer.mu.Unlock()
				if failure.FailPending {
					historyFailed = true
				}
				failures <- failure
			})
			select {
			case ok := <-notified:
				if !ok {
					t.Fatal("notification preceded history disposition handling")
				}
			case <-ctx.Done():
				t.Fatal("ordinary notification was not delivered")
			}
			if got := <-failures; got != want {
				t.Fatalf("history disposition = %+v, want %+v", got, want)
			}
			select {
			case err := <-call:
				if err != nil {
					t.Fatalf("ordinary Call failed: %v", err)
				}
			case <-ctx.Done():
				t.Fatal("ordinary Call did not complete")
			}
			select {
			case <-peer.Done():
				t.Fatal("history failure closed peer")
			default:
			}
			release()
		})
	}
	test("discard invalid private stage", `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":null}}`, HistoryFailure{Disposition: "discard", Category: "history-invalid", Count: 1})
	test("fail correlated history state", `{"jsonrpc":"2.0","id":"history","result":{"requests":null}}`, HistoryFailure{Disposition: "fail-call", Category: "history-invalid", Count: 1, FailPending: true})
	test("quarantine unclassifiable input", `{"synthetic-private-sentinel":`, HistoryFailure{Disposition: "quarantine", Category: "history-invalid", Count: 1, FailPending: true, ScheduleRefetch: true})
}

func TestPeerHistoryFailureRefetchSequencing(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte(`"history"`)); err != nil {
		t.Fatal(err)
	}
	malformed := "{\n"
	input := malformed + malformed + `{"jsonrpc":"2.0","method":"refetch-ready"}` + "\n" + `{"jsonrpc":"2.0","id":"refetch","result":{"requests":null}}` + "\n"
	peer := NewPeer(historyTestReader(t, input, &correlation))
	var failures []HistoryFailure
	scheduled := false
	peer.ServeWithHistoryFailures(nil, func(method string, _ json.RawMessage) {
		if method != "refetch-ready" || !scheduled || !correlation.Quarantined() {
			t.Fatal("refetch was not queued before next notification")
		}
		if err := correlation.Begin([]byte(`"refetch"`)); err != ErrHistoryBusy {
			t.Fatalf("retired transfer was reusable: %v", err)
		}
		correlation.ReleaseRetired()
		scheduled = false
		if err := correlation.Begin([]byte(`"refetch"`)); err != nil {
			t.Fatal(err)
		}
	}, func(failure HistoryFailure) {
		if failure.ScheduleRefetch {
			if scheduled {
				t.Fatal("duplicate outstanding refetch")
			}
			scheduled = true
		}
		failures = append(failures, failure)
	})
	want := []HistoryFailure{
		{Disposition: "quarantine", Category: "history-invalid", Count: 1, FailPending: true, ScheduleRefetch: true},
		{Disposition: "quarantine", Category: "history-invalid", Count: 1},
		{Disposition: "fail-call", Category: "history-invalid", Count: 1, FailPending: true},
	}
	if len(failures) != len(want) {
		t.Fatalf("disposition count = %d, want %d", len(failures), len(want))
	}
	for i := range want {
		if failures[i] != want[i] {
			t.Fatalf("disposition %d = %+v, want %+v", i, failures[i], want[i])
		}
	}
	if correlation.pendingN != 0 || correlation.retiredN == 0 || scheduled {
		t.Fatal("refetch failure lost retired ownership or queued extra work")
	}
}

func TestPeerHistoryTerminalReadsWakeOrdinaryCall(t *testing.T) {
	test := func(name, input string, terminalErr error, optIn bool) {
		t.Run(name, func(t *testing.T) {
			stream, release := newPeerHistoryStream(t, input, terminalErr)
			peer := newHistoryTestPeer(t, stream, &HistoryCorrelation{})
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			call := make(chan error, 1)
			go func() { _, _, err := peer.Call(ctx, "ordinary", nil); call <- err }()
			select {
			case <-stream.written:
			case <-ctx.Done():
				t.Fatal("request was not written")
			}
			callbacks := make(chan HistoryFailure, 1)
			if optIn {
				go peer.ServeWithHistoryFailures(nil, nil, func(failure HistoryFailure) { callbacks <- failure })
			} else {
				go peer.Serve(nil, nil)
			}
			release()
			select {
			case err := <-call:
				if err != ErrPeerClosed {
					t.Fatalf("terminal read returned %v, want ErrPeerClosed", err)
				}
			case <-ctx.Done():
				t.Fatal("terminal read did not wake Call")
			}
			if len(callbacks) != 0 {
				t.Fatal("terminal error was reported as recoverable history")
			}
		})
	}
	test("EOF", "", io.EOF, true)
	test("transport failure", "", errors.New("synthetic transport failure"), true)
	test("physical overflow", strings.Repeat("x", HistoryFrameBytes+1), io.EOF, true)
	test("default Serve keeps history failures terminal", "{\n", io.EOF, false)
}

func TestPeerHistoryBusyIsTerminal(t *testing.T) {
	var correlation HistoryCorrelation
	if err := correlation.Begin([]byte(`"history"`)); err != nil {
		t.Fatal(err)
	}
	// A private-record lease cannot receive an initial. Read returns busy
	// before consuming input, so recovering it would spin the read pump.
	codec, err := NewHistoryCodec(rw{strings.NewReader(""), io.Discard}, 1024, make([]byte, 0, HistoryRecordFrameBytes), nil, make([]uint32, HistoryRecordBytes/4), &correlation, historyTestSchema, nil)
	if err != nil {
		t.Fatal(err)
	}
	peer := NewPeer(codec)
	peer.ServeWithHistoryFailures(nil, nil, func(HistoryFailure) {
		t.Fatal("busy was reported as a consumed history failure")
	})
	select {
	case <-peer.Done():
	default:
		t.Fatal("busy did not terminate read pump")
	}
	if _, _, err := peer.Call(context.Background(), "ordinary", nil); err != ErrPeerClosed {
		t.Fatalf("Call after busy returned %v, want ErrPeerClosed", err)
	}
}

// pipePeers wires two Peers over a synchronous in-memory net.Pipe. Handlers
// that write a reply do so in a goroutine so a blocking write on the
// unbuffered pipe never stalls the read pump (real stdio pipes are buffered).
func pipePeers(t *testing.T) (a, b *Peer) {
	t.Helper()
	c1, c2 := net.Pipe()
	a = NewPeer(NewCodec(c1))
	b = NewPeer(NewCodec(c2))
	t.Cleanup(func() { _ = c1.Close(); _ = c2.Close() })
	return a, b
}

func TestPeerCallReturnsResult(t *testing.T) {
	a, b := pipePeers(t)
	go b.Serve(func(req *Message) {
		go b.Respond(req.ID, map[string]string{"echo": req.Method})
	}, nil)
	go a.Serve(nil, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	res, rpcErr, err := a.Call(ctx, "ping", json.RawMessage(`{"x":1}`))
	if err != nil || rpcErr != nil {
		t.Fatalf("Call err=%v rpcErr=%v", err, rpcErr)
	}
	var out map[string]string
	if json.Unmarshal(res, &out); out["echo"] != "ping" {
		t.Fatalf("unexpected result: %s", res)
	}
}

func TestPeerCallReturnsRPCError(t *testing.T) {
	a, b := pipePeers(t)
	go b.Serve(func(req *Message) {
		go b.RespondError(req.ID, -32601, "method not found")
	}, nil)
	go a.Serve(nil, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, rpcErr, err := a.Call(ctx, "nope", nil)
	if err != nil {
		t.Fatalf("transport err: %v", err)
	}
	if rpcErr == nil || rpcErr.Code != -32601 {
		t.Fatalf("want rpc error -32601, got %+v", rpcErr)
	}
}

func TestPeerSimultaneousInboundRequest(t *testing.T) {
	// Both peers Call each other while both are serving inbound requests. The
	// read pump is separate from Call, so neither side deadlocks.
	a, b := pipePeers(t)
	handler := func(self *Peer) func(*Message) {
		return func(req *Message) { go self.Respond(req.ID, map[string]string{"from": req.Method}) }
	}
	go a.Serve(handler(a), nil)
	go b.Serve(handler(b), nil)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	type res struct {
		raw json.RawMessage
		err error
	}
	ares := make(chan res, 1)
	bres := make(chan res, 1)
	go func() { r, _, e := a.Call(ctx, "a-calls-b", nil); ares <- res{r, e} }()
	go func() { r, _, e := b.Call(ctx, "b-calls-a", nil); bres <- res{r, e} }()

	for _, ch := range []chan res{ares, bres} {
		select {
		case r := <-ch:
			if r.err != nil {
				t.Fatalf("simultaneous call failed: %v", r.err)
			}
		case <-time.After(3 * time.Second):
			t.Fatal("simultaneous calls deadlocked")
		}
	}
}

func TestPeerNotificationOrdering(t *testing.T) {
	a, b := pipePeers(t)
	got := make(chan string, 16)
	go b.Serve(nil, func(method string, _ json.RawMessage) { got <- method })
	go a.Serve(nil, nil)

	const n = 8
	for i := 0; i < n; i++ {
		if err := a.Notify(fmt.Sprintf("n%d", i), nil); err != nil {
			t.Fatalf("Notify: %v", err)
		}
	}
	for i := 0; i < n; i++ {
		select {
		case m := <-got:
			if m != fmt.Sprintf("n%d", i) {
				t.Fatalf("out of order: got %s want n%d", m, i)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for notification %d", i)
		}
	}
}

func TestPeerCallCtxCancel(t *testing.T) {
	a, b := pipePeers(t)
	go b.Serve(func(*Message) {}, nil) // never responds
	go a.Serve(nil, nil)

	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(100 * time.Millisecond); cancel() }()
	_, _, err := a.Call(ctx, "hang", nil)
	if err != context.Canceled {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestPeerCloseWakesPendingCall(t *testing.T) {
	c1, c2 := net.Pipe()
	a := NewPeer(NewCodec(c1))
	b := NewPeer(NewCodec(c2))
	go b.Serve(func(*Message) {}, nil) // never responds
	go a.Serve(nil, nil)

	errc := make(chan error, 1)
	go func() { _, _, e := a.Call(context.Background(), "hang", nil); errc <- e }()
	time.Sleep(50 * time.Millisecond)
	_ = c1.Close() // a's read pump errors -> a.Close() -> pending Call wakes
	_ = c2.Close()

	select {
	case e := <-errc:
		if e != ErrPeerClosed {
			t.Fatalf("want ErrPeerClosed, got %v", e)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("pending Call was not woken on close")
	}
}

func TestPeerCallAfterCloseFailsFast(t *testing.T) {
	a, _ := pipePeers(t)
	a.Close()
	if _, _, err := a.Call(context.Background(), "m", nil); err != ErrPeerClosed {
		t.Fatalf("want ErrPeerClosed after Close, got %v", err)
	}
}

func TestPeerRelayRequest(t *testing.T) {
	a, b := pipePeers(t)
	go b.Serve(func(req *Message) {
		go b.Respond(req.ID, map[string]int{"ok": 1})
	}, nil)
	go a.Serve(nil, nil)

	type relayed struct {
		res json.RawMessage
		err error
	}
	done := make(chan relayed, 1)
	if err := a.RelayRequest("m", nil, func(res json.RawMessage, _ *RPCError, err error) {
		done <- relayed{res, err}
	}); err != nil {
		t.Fatalf("RelayRequest: %v", err)
	}
	select {
	case r := <-done:
		if r.err != nil {
			t.Fatalf("relay err: %v", r.err)
		}
		var out map[string]int
		if json.Unmarshal(r.res, &out); out["ok"] != 1 {
			t.Fatalf("unexpected relay result: %s", r.res)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("relay respond not invoked")
	}
}

// blockedPeerWriter optionally accepts a prefix, then holds one Write until
// released. All observable state is protected so race tests exercise callers.
type blockedPeerWriter struct {
	mu       sync.Mutex
	buf      bytes.Buffer
	calls    int
	prefix   int
	fail     bool
	entered  chan struct{}
	release  chan struct{}
	returned chan struct{}
	once     sync.Once
}

func newBlockedPeerWriter(t *testing.T, prefix int, fail bool) *blockedPeerWriter {
	t.Helper()
	w := &blockedPeerWriter{prefix: prefix, fail: fail, entered: make(chan struct{}), release: make(chan struct{}), returned: make(chan struct{})}
	t.Cleanup(w.unblock)
	return w
}

func (w *blockedPeerWriter) unblock() { w.once.Do(func() { close(w.release) }) }

func (w *blockedPeerWriter) Write(data []byte) (int, error) {
	w.mu.Lock()
	w.calls++
	call := w.calls
	if call == 1 && w.prefix > 0 {
		n, err := w.buf.Write(data[:w.prefix])
		w.mu.Unlock()
		return n, err
	}
	w.mu.Unlock()
	blockAt := 1
	if w.prefix > 0 {
		blockAt = 2
	}
	if call == blockAt {
		close(w.entered)
		<-w.release
		defer close(w.returned)
		if w.fail {
			return 0, io.ErrClosedPipe
		}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(data)
}

func (w *blockedPeerWriter) snapshot() ([]byte, int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return bytes.Clone(w.buf.Bytes()), w.calls
}

func awaitPeerSignal(t *testing.T, ch <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}

func awaitPeerError(t *testing.T, ch <-chan error) error {
	t.Helper()
	select {
	case err := <-ch:
		return err
	case <-time.After(time.Second):
		t.Fatal("Call did not return within one second")
		return nil
	}
}

func assertPeerPending(t *testing.T, p *Peer, want int) {
	t.Helper()
	p.mu.Lock()
	n := len(p.pending)
	p.mu.Unlock()
	if n != want {
		t.Fatalf("pending requests = %d, want %d", n, want)
	}
}

// Each notification case exercises both public send paths against the same
// synchronized writer, including raw codec rejection after peer retirement.
func notificationLifetimeEndpoints(t *testing.T, test func(*testing.T, func(io.Writer, func()) (*Codec, func(context.Context, string, any) error))) {
	t.Helper()
	t.Run("codec", func(t *testing.T) {
		test(t, func(w io.Writer, interrupt func()) (*Codec, func(context.Context, string, any) error) {
			c := NewCodecMaxFrameWithWriteInterrupt(rw{bytes.NewReader(nil), w}, WorkerFrameBytes, interrupt)
			return c, c.NotifyContext
		})
	})
	t.Run("peer", func(t *testing.T) {
		test(t, func(w io.Writer, interrupt func()) (*Codec, func(context.Context, string, any) error) {
			c := writeCodec(w)
			p := NewPeerWithWriteInterrupt(c, interrupt)
			return c, p.NotifyContext
		})
	})
}

func TestNotificationLifetimePreCancelled(t *testing.T) {
	notificationLifetimeEndpoints(t, func(t *testing.T, newEndpoint func(io.Writer, func()) (*Codec, func(context.Context, string, any) error)) {
		var buf bytes.Buffer
		var interrupts atomic.Int32
		_, send := newEndpoint(&buf, func() { interrupts.Add(1) })
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := send(ctx, "cancelled", nil); !errors.Is(err, context.Canceled) {
			t.Fatalf("send: %v", err)
		}
		if buf.Len() != 0 || interrupts.Load() != 0 {
			t.Fatalf("pre-cancelled send wrote %d bytes, interrupted %d times", buf.Len(), interrupts.Load())
		}
		if err := send(context.Background(), "healthy", nil); err != nil {
			t.Fatalf("healthy send: %v", err)
		}
	})
}

func TestNotificationLifetimeQueuedCancellation(t *testing.T) {
	notificationLifetimeEndpoints(t, func(t *testing.T, newEndpoint func(io.Writer, func()) (*Codec, func(context.Context, string, any) error)) {
		w := newBlockedPeerWriter(t, 0, false)
		var interrupts atomic.Int32
		_, send := newEndpoint(w, func() { interrupts.Add(1); w.unblock() })
		owner := make(chan error, 1)
		go func() { owner <- send(context.Background(), "owner", nil) }()
		awaitPeerSignal(t, w.entered, "active notification")
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		if err := send(ctx, "cancelled", nil); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("queued send: %v", err)
		}
		if interrupts.Load() != 0 {
			t.Fatal("queued send interrupted active owner")
		}
		w.unblock()
		if err := awaitPeerError(t, owner); err != nil {
			t.Fatalf("owner send: %v", err)
		}
		awaitPeerSignal(t, w.returned, "owner Write completion")
		if err := send(context.Background(), "healthy", nil); err != nil {
			t.Fatalf("later send: %v", err)
		}
		data, calls := w.snapshot()
		if calls != 2 {
			t.Fatalf("Write calls = %d, want 2", calls)
		}
		c := readCodec(bytes.NewReader(data))
		for _, want := range []string{"owner", "healthy"} {
			msg, err := c.Read()
			if err != nil || msg.Method != want {
				t.Fatalf("frame = %+v, error = %v, want %s", msg, err, want)
			}
		}
		if _, err := c.Read(); !errors.Is(err, io.EOF) {
			t.Fatalf("unexpected late bytes: %v", err)
		}
	})
}

func TestNotificationLifetimeActiveDeadline(t *testing.T) {
	notificationLifetimeEndpoints(t, func(t *testing.T, newEndpoint func(io.Writer, func()) (*Codec, func(context.Context, string, any) error)) {
		w := newBlockedPeerWriter(t, 0, true)
		var interrupts atomic.Int32
		c, send := newEndpoint(w, func() { interrupts.Add(1); w.unblock() })
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		result := make(chan error, 1)
		go func() { result <- send(ctx, "blocked", nil) }()
		awaitPeerSignal(t, w.entered, "active Write")
		if err := awaitPeerError(t, result); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("active send: %v", err)
		}
		awaitPeerSignal(t, w.returned, "physical Write completion")
		if err := send(context.Background(), "later", nil); !errors.Is(err, ErrPeerClosed) {
			t.Fatalf("later send: %v", err)
		}
		id := json.RawMessage("9")
		if err := c.Respond(&id, nil); !errors.Is(err, ErrPeerClosed) {
			t.Fatalf("raw response on retired stream: %v", err)
		}
		data, calls := w.snapshot()
		if len(data) != 0 || calls != 1 || interrupts.Load() != 1 {
			t.Fatalf("bytes=%d writes=%d interrupts=%d", len(data), calls, interrupts.Load())
		}
	})
}

func TestNotificationLifetimePartialFrameJoinsCallback(t *testing.T) {
	notificationLifetimeEndpoints(t, func(t *testing.T, newEndpoint func(io.Writer, func()) (*Codec, func(context.Context, string, any) error)) {
		w := newBlockedPeerWriter(t, 7, true)
		started, finish := make(chan struct{}), make(chan struct{})
		var once sync.Once
		release := func() { once.Do(func() { close(finish) }) }
		t.Cleanup(release)
		_, send := newEndpoint(w, func() { w.unblock(); close(started); <-finish })
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		result := make(chan error, 1)
		go func() { result <- send(ctx, "partial", nil) }()
		awaitPeerSignal(t, w.entered, "partial Write")
		cancel()
		awaitPeerSignal(t, started, "callback start")
		awaitPeerSignal(t, w.returned, "physical Write completion")
		select {
		case err := <-result:
			t.Fatalf("send returned before callback: %v", err)
		case <-time.After(20 * time.Millisecond):
		}
		release()
		if err := awaitPeerError(t, result); !errors.Is(err, context.Canceled) {
			t.Fatalf("partial send: %v", err)
		}
		if err := send(context.Background(), "later", nil); !errors.Is(err, ErrPeerClosed) {
			t.Fatalf("later send: %v", err)
		}
		data, calls := w.snapshot()
		if string(data) != `{"jsonr` || calls != 2 {
			t.Fatalf("partial bytes=%q writes=%d", data, calls)
		}
	})
}

func TestNotificationLifetimeShortWritesAndCompletedCancellation(t *testing.T) {
	notificationLifetimeEndpoints(t, func(t *testing.T, newEndpoint func(io.Writer, func()) (*Codec, func(context.Context, string, any) error)) {
		w := &oneByteWriter{}
		var interrupts atomic.Int32
		_, send := newEndpoint(w, func() { interrupts.Add(1) })
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if err := send(ctx, "complete", map[string]int{"value": 7}); err != nil {
			t.Fatalf("short-write send: %v", err)
		}
		cancel()
		if err := send(context.Background(), "after", nil); err != nil {
			t.Fatalf("completed cancellation damaged stream: %v", err)
		}
		if interrupts.Load() != 0 {
			t.Fatal("completed send interrupted transport")
		}
		c := readCodec(&w.buf)
		msg, err := c.Read()
		if err != nil || msg.Method != "complete" {
			t.Fatalf("complete frame=%+v error=%v", msg, err)
		}
		var params struct {
			Value int `json:"value"`
		}
		if err := json.Unmarshal(msg.Params, &params); err != nil || params.Value != 7 {
			t.Fatalf("params=%+v error=%v", params, err)
		}
		msg, err = c.Read()
		if err != nil || msg.Method != "after" {
			t.Fatalf("later frame=%+v error=%v", msg, err)
		}
	})
}

func TestNotificationLifetimeMarshalFailure(t *testing.T) {
	notificationLifetimeEndpoints(t, func(t *testing.T, newEndpoint func(io.Writer, func()) (*Codec, func(context.Context, string, any) error)) {
		var buf bytes.Buffer
		var interrupts atomic.Int32
		_, send := newEndpoint(&buf, func() { interrupts.Add(1) })
		if err := send(context.Background(), "invalid", make(chan int)); err == nil {
			t.Fatal("marshal succeeded")
		}
		if buf.Len() != 0 || interrupts.Load() != 0 {
			t.Fatal("marshal failure touched transport")
		}
		if err := send(context.Background(), "healthy", nil); err != nil {
			t.Fatalf("healthy send: %v", err)
		}
	})
}

type notificationCallWriter struct {
	first chan struct{}
	next  io.Writer
	once  sync.Once
}

func (w *notificationCallWriter) Write(data []byte) (int, error) {
	first := false
	w.once.Do(func() { first = true; close(w.first) })
	if first {
		return len(data), nil
	}
	return w.next.Write(data)
}

func TestNotificationLifetimePendingCallReleased(t *testing.T) {
	w := newBlockedPeerWriter(t, 0, true)
	writer := &notificationCallWriter{first: make(chan struct{}), next: w}
	p := NewPeerWithWriteInterrupt(writeCodec(writer), w.unblock)
	call := make(chan error, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _, _, err := p.Call(ctx, "pending", nil); call <- err }()
	awaitPeerSignal(t, writer.first, "Call frame")
	deadline, stop := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer stop()
	if err := p.NotifyContext(deadline, "blocked", nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("notification: %v", err)
	}
	awaitPeerSignal(t, w.returned, "notification Write completion")
	if err := awaitPeerError(t, call); !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("pending Call: %v", err)
	}
	assertPeerPending(t, p, 0)
}

func TestPeerWriteInterruptQueuedCancellation(t *testing.T) {
	w := newBlockedPeerWriter(t, 0, false)
	var interrupts atomic.Int32
	p := NewPeerWithWriteInterrupt(writeCodec(w), func() { interrupts.Add(1); w.unblock() })
	owner := make(chan error, 1)
	go func() { owner <- p.Notify("owner", nil) }()
	awaitPeerSignal(t, w.entered, "notification write ownership")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	result := make(chan error, 1)
	go func() { _, _, err := p.Call(ctx, "must-not-write", nil); result <- err }()
	if err := awaitPeerError(t, result); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued Call error = %v, want deadline exceeded", err)
	}
	assertPeerPending(t, p, 0)
	if interrupts.Load() != 0 {
		t.Fatalf("queued cancellation interrupted the owner %d times", interrupts.Load())
	}
	select {
	case <-p.Done():
		t.Fatal("queued cancellation closed the peer")
	default:
	}
	w.unblock()
	if err := awaitPeerError(t, owner); err != nil {
		t.Fatalf("owner Notify: %v", err)
	}
	if err := p.Notify("after", nil); err != nil {
		t.Fatalf("healthy peer Notify: %v", err)
	}
	data, calls := w.snapshot()
	if calls != 2 {
		t.Fatalf("writer calls = %d, want two notifications only", calls)
	}
	c := readCodec(bytes.NewReader(data))
	for _, want := range []string{"owner", "after"} {
		msg, err := c.Read()
		if err != nil || msg.Method != want || !msg.IsNotification() {
			t.Fatalf("notification = %+v, err=%v, want %s", msg, err, want)
		}
	}
	if _, err := c.Read(); !errors.Is(err, io.EOF) {
		t.Fatalf("cancelled request wrote a later frame: %v", err)
	}
}

func TestPeerWriteInterruptAlreadyCancelled(t *testing.T) {
	var data bytes.Buffer
	var interrupts atomic.Int32
	p := NewPeerWithWriteInterrupt(writeCodec(&data), func() { interrupts.Add(1) })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := p.Call(ctx, "must-not-write", nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("Call error = %v, want cancelled", err)
	}
	assertPeerPending(t, p, 0)
	if data.Len() != 0 || interrupts.Load() != 0 {
		t.Fatalf("cancelled Call wrote %d bytes and interrupted %d times", data.Len(), interrupts.Load())
	}
	select {
	case <-p.Done():
		t.Fatal("pre-cancelled Call closed the peer")
	default:
	}
}

func TestPeerWriteInterruptActiveDeadline(t *testing.T) {
	w := newBlockedPeerWriter(t, 0, true)
	var interrupts atomic.Int32
	p := NewPeerWithWriteInterrupt(writeCodec(w), func() { interrupts.Add(1); w.unblock() })
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	result := make(chan error, 1)
	go func() { _, _, err := p.Call(ctx, "blocked", nil); result <- err }()
	awaitPeerSignal(t, w.entered, "active Write")
	if err := awaitPeerError(t, result); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("active Call error = %v, want deadline exceeded", err)
	}
	select {
	case <-w.returned:
	default:
		t.Fatal("Call returned before its Write finished")
	}
	awaitPeerSignal(t, p.Done(), "peer invalidation")
	assertPeerPending(t, p, 0)
	if interrupts.Load() != 1 {
		t.Fatalf("interrupt calls = %d, want 1", interrupts.Load())
	}
	if _, _, err := p.Call(context.Background(), "later", nil); !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("Call after interrupted write = %v, want peer closed", err)
	}
	if err := p.Notify("later", nil); !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("Notify after interrupted write = %v, want peer closed", err)
	}
	data, calls := w.snapshot()
	if len(data) != 0 || calls != 1 {
		t.Fatalf("retired writer received %d bytes across %d calls", len(data), calls)
	}
}

func TestPeerWriteInterruptPartialFrameWaitsForCallback(t *testing.T) {
	w := newBlockedPeerWriter(t, 7, true)
	callbackStarted := make(chan struct{})
	callbackFinish := make(chan struct{})
	var finishOnce sync.Once
	finishCallback := func() { finishOnce.Do(func() { close(callbackFinish) }) }
	t.Cleanup(finishCallback)
	var interrupts atomic.Int32
	p := NewPeerWithWriteInterrupt(writeCodec(w), func() {
		interrupts.Add(1)
		w.unblock()
		close(callbackStarted)
		<-callbackFinish
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() { _, _, err := p.Call(ctx, "partial", nil); result <- err }()
	awaitPeerSignal(t, w.entered, "partial-frame Write")
	cancel()
	awaitPeerSignal(t, callbackStarted, "interruption callback")
	awaitPeerSignal(t, w.returned, "interrupted Write return")
	select {
	case err := <-result:
		t.Fatalf("Call returned before its interruption callback finished: %v", err)
	case <-time.After(20 * time.Millisecond):
	}
	finishCallback()
	if err := awaitPeerError(t, result); !errors.Is(err, context.Canceled) {
		t.Fatalf("partial Call error = %v, want cancelled", err)
	}
	assertPeerPending(t, p, 0)
	awaitPeerSignal(t, p.Done(), "partial-frame peer invalidation")
	id := json.RawMessage("9")
	for _, write := range []func() error{
		func() error { return p.Notify("later", nil) },
		func() error { return p.Respond(&id, nil) },
		func() error { return p.RespondError(&id, -32603, "later") },
		func() error { return p.RespondErrorData(&id, -32603, "later", nil) },
		func() error { return p.Codec().Notify("raw-later", nil) },
		func() error {
			return p.RelayRequest("later", nil, func(json.RawMessage, *RPCError, error) { t.Error("closed relay callback ran") })
		},
	} {
		if err := write(); !errors.Is(err, ErrPeerClosed) {
			t.Fatalf("write on retired frame error = %v, want peer closed", err)
		}
	}
	data, calls := w.snapshot()
	if string(data) != `{"jsonr` || calls != 2 || interrupts.Load() != 1 {
		t.Fatalf("partial frame = %q, writes=%d, interruptions=%d; want seven-byte prefix, two writes, one interruption", data, calls, interrupts.Load())
	}
}

func TestPeerWriteInterruptResponseTimeoutPreservesPeer(t *testing.T) {
	c1, c2 := net.Pipe()
	t.Cleanup(func() { _ = c1.Close(); _ = c2.Close() })
	var interrupts atomic.Int32
	a := NewPeerWithWriteInterrupt(NewCodec(c1), func() { interrupts.Add(1); _ = c1.Close() })
	b := NewPeer(NewCodec(c2))
	requests := make(chan *Message, 2)
	go b.Serve(func(req *Message) { requests <- req }, nil)
	go a.Serve(nil, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, _, err := a.Call(ctx, "no-response", nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("response-only timeout = %v, want deadline exceeded", err)
	}
	assertPeerPending(t, a, 0)
	if interrupts.Load() != 0 {
		t.Fatalf("response-only timeout interrupted transport %d times", interrupts.Load())
	}
	select {
	case <-a.Done():
		t.Fatal("response-only timeout closed healthy peer")
	default:
	}
	var late *Message
	select {
	case late = <-requests:
	case <-time.After(time.Second):
		t.Fatal("timed-out request did not reach remote peer")
	}
	if err := b.Respond(late.ID, map[string]string{"value": "late"}); err != nil {
		t.Fatalf("late reply: %v", err)
	}

	type callResult struct {
		raw    json.RawMessage
		rpcErr *RPCError
		err    error
	}
	result := make(chan callResult, 1)
	go func() {
		raw, rpcErr, err := a.Call(context.Background(), "reuse", nil)
		result <- callResult{raw, rpcErr, err}
	}()
	var req *Message
	select {
	case req = <-requests:
	case <-time.After(time.Second):
		t.Fatal("healthy peer could not send next request")
	}
	if req.Method != "reuse" || string(*req.ID) == string(*late.ID) {
		t.Fatalf("next request = %+v, want reuse with a distinct id", req)
	}
	if err := b.Respond(req.ID, map[string]string{"value": "fresh"}); err != nil {
		t.Fatalf("next reply: %v", err)
	}
	select {
	case got := <-result:
		if got.err != nil || got.rpcErr != nil {
			t.Fatalf("next Call err=%v rpcErr=%v", got.err, got.rpcErr)
		}
		var out map[string]string
		if err := json.Unmarshal(got.raw, &out); err != nil || out["value"] != "fresh" {
			t.Fatalf("next result = %v, decode error=%v, want fresh", out, err)
		}
	case <-time.After(time.Second):
		t.Fatal("next Call did not receive its own response")
	}
	assertPeerPending(t, a, 0)
	if interrupts.Load() != 0 {
		t.Fatalf("peer reuse interrupted transport %d times", interrupts.Load())
	}
}

func TestPeerWriteInterruptConcurrentCorrelation(t *testing.T) {
	c1, c2 := net.Pipe()
	t.Cleanup(func() { _ = c1.Close(); _ = c2.Close() })
	var interrupts atomic.Int32
	a := NewPeerWithWriteInterrupt(NewCodec(c1), func() { interrupts.Add(1); _ = c1.Close() })
	b := NewPeer(NewCodec(c2))
	requests := make(chan *Message, 3)
	go b.Serve(func(req *Message) { requests <- req }, nil)
	go a.Serve(nil, nil)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	type result struct {
		method string
		raw    json.RawMessage
		rpcErr *RPCError
		err    error
	}
	results := make(chan result, 3)
	for _, method := range []string{"first", "second"} {
		go func() {
			raw, rpcErr, err := a.Call(ctx, method, nil)
			results <- result{method, raw, rpcErr, err}
		}()
	}
	if err := a.RelayRequest("relay", nil, func(raw json.RawMessage, rpcErr *RPCError, err error) {
		results <- result{"relay", raw, rpcErr, err}
	}); err != nil {
		t.Fatalf("RelayRequest: %v", err)
	}
	received := make([]*Message, 0, 3)
	ids := make(map[string]bool)
	for range 3 {
		select {
		case req := <-requests:
			id := string(*req.ID)
			if ids[id] {
				t.Fatalf("duplicate concurrent request id %s", id)
			}
			ids[id] = true
			received = append(received, req)
		case <-time.After(time.Second):
			t.Fatal("concurrent request was not written")
		}
	}
	for i := len(received) - 1; i >= 0; i-- {
		req := received[i]
		if err := b.Respond(req.ID, map[string]string{"method": req.Method}); err != nil {
			t.Fatalf("out-of-order response: %v", err)
		}
	}
	seen := make(map[string]bool)
	for range 3 {
		select {
		case got := <-results:
			if got.err != nil || got.rpcErr != nil {
				t.Fatalf("%s result error=%v rpcErr=%v", got.method, got.err, got.rpcErr)
			}
			var out map[string]string
			if err := json.Unmarshal(got.raw, &out); err != nil || out["method"] != got.method {
				t.Fatalf("%s received %v, decode error=%v", got.method, out, err)
			}
			if seen[got.method] {
				t.Fatalf("duplicate result for %s", got.method)
			}
			seen[got.method] = true
		case <-time.After(time.Second):
			t.Fatal("concurrent request did not receive its response")
		}
	}
	for _, method := range []string{"first", "second", "relay"} {
		if !seen[method] {
			t.Fatalf("missing result for %s", method)
		}
	}
	if interrupts.Load() != 0 {
		t.Fatalf("successful concurrent requests interrupted transport %d times", interrupts.Load())
	}
}

// partialErrorWriter accepts a prefix with an error, but accepts future writes.
// Retirement must therefore come from the peer rather than this test transport.
type partialErrorWriter struct {
	buf   bytes.Buffer
	calls int
}

func (w *partialErrorWriter) Write(data []byte) (int, error) {
	w.calls++
	if w.calls == 1 {
		n, err := w.buf.Write(data[:7])
		if err != nil {
			return n, err
		}
		return n, io.ErrUnexpectedEOF
	}
	return w.buf.Write(data)
}

func TestPeerWriteInterruptNotificationAndResponseFailures(t *testing.T) {
	id := json.RawMessage("9")
	test := func(name string, send func(*Peer) error) {
		t.Run(name, func(t *testing.T) {
			var w partialErrorWriter
			interrupts := 0
			var p *Peer
			p = NewPeerWithWriteInterrupt(writeCodec(&w), func() {
				interrupts++
				select {
				case <-p.Done():
				default:
					t.Error("transport interruption ran before peer closure")
				}
			})
			if err := send(p); !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("partial %s error = %v, want unexpected EOF", name, err)
			}
			awaitPeerSignal(t, p.Done(), "failed-frame peer closure")
			assertPeerPending(t, p, 0)
			rejected := func(name string, write func() error) {
				t.Helper()
				if err := write(); !errors.Is(err, ErrPeerClosed) {
					t.Fatalf("%s after failed frame = %v, want peer closed", name, err)
				}
			}
			rejected("Call", func() error { _, _, err := p.Call(context.Background(), "later-call", nil); return err })
			rejected("RelayRequest", func() error {
				return p.RelayRequest("later-relay", nil, func(json.RawMessage, *RPCError, error) { t.Error("retired relay callback ran") })
			})
			rejected("Notify", func() error { return p.Notify("later-note", nil) })
			rejected("Respond", func() error { return p.Respond(&id, "later-response") })
			rejected("RespondError", func() error { return p.RespondError(&id, -32603, "later-error") })
			rejected("RespondErrorData", func() error { return p.RespondErrorData(&id, -32603, "later-error-data", map[string]int{"key": 1}) })
			rejected("direct Codec write", func() error { return p.Codec().Notify("later-codec", nil) })
			if w.buf.String() != `{"jsonr` || w.calls != 1 || interrupts != 1 {
				t.Fatalf("retired prefix=%q, writer calls=%d, interruptions=%d; want original prefix, one write, one interruption", w.buf.String(), w.calls, interrupts)
			}
		})
	}
	test("notification", func(p *Peer) error { return p.Notify("note", map[string]string{"value": "first"}) })
	test("successful response", func(p *Peer) error { return p.Respond(&id, map[string]string{"value": "first"}) })
	test("error response", func(p *Peer) error { return p.RespondError(&id, -32603, "first") })
	test("error response with data", func(p *Peer) error { return p.RespondErrorData(&id, -32603, "first", map[string]int{"key": 1}) })
}
