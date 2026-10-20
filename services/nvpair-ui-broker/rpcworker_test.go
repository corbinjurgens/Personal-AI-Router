// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/applog"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

const notificationHelperEnv = "PAIR_NOTIFICATION_LIFETIME_HELPER"

// This entry is inert without the test-owned environment marker. It uses only
// anonymous stdio pipes and files in the test's temporary directory.
func TestNotificationLifetimeHelper(t *testing.T) {
	mode := os.Getenv(notificationHelperEnv)
	if mode == "" {
		return
	}
	codec := NewCodecMaxFrame(readWriter{os.Stdin, os.Stdout}, workerFrameBytes)
	if err := codec.Notify("helper:ready", nil); err != nil {
		t.Fatal(err)
	}
	if mode == "exit" {
		notificationHelperResume(t)
		return
	}
	if mode == "stall" {
		msg, err := codec.Read()
		if err != nil || msg.Method != "pending" {
			t.Fatalf("pending frame=%+v error=%v", msg, err)
		}
		if err := codec.Notify("helper:pending", nil); err != nil {
			t.Fatal(err)
		}
		// Observe the first byte of the next frame before pausing. This is a
		// cross-process barrier proving the parent's large Write owns the gate.
		var first [1]byte
		if _, err := io.ReadFull(os.Stdin, first[:]); err != nil {
			t.Fatal(err)
		}
		if err := codec.Notify("helper:blocked", nil); err != nil {
			t.Fatal(err)
		}
		notificationHelperResume(t)
		codec = NewCodecMaxFrame(readWriter{io.MultiReader(bytes.NewReader(first[:]), os.Stdin), os.Stdout}, workerFrameBytes)
	}
	for {
		msg, err := codec.Read()
		if errors.Is(err, io.EOF) {
			return
		}
		if err != nil {
			var decode *jsonrpc.DecodeError
			if errors.As(err, &decode) {
				continue
			}
			t.Fatal(err)
		}
		if err := codec.Notify("helper:received", map[string]any{"method": msg.Method, "params": msg.Params}); err != nil {
			t.Fatal(err)
		}
	}
}

func notificationHelperResume(t *testing.T) {
	t.Helper()
	resume := filepath.Join(os.Getenv("PAIR_NOTIFICATION_LIFETIME_DIR"), "resume")
	deadline := time.Now().Add(15 * time.Second)
	for {
		if _, err := os.Stat(resume); err == nil {
			return
		} else if !errors.Is(err, os.ErrNotExist) {
			t.Fatal(err)
		}
		if time.Now().After(deadline) {
			t.Fatal("parent never released helper")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

type notificationWorkerFixture struct {
	w          *rpcWorker
	events     chan Message
	exited     chan struct{}
	dir        string
	resumeOnce sync.Once
}

func newNotificationWorker(t *testing.T, mode string, budget time.Duration) *notificationWorkerFixture {
	t.Helper()
	dir := t.TempDir()
	t.Setenv(notificationHelperEnv, mode)
	t.Setenv("PAIR_NOTIFICATION_LIFETIME_DIR", dir)
	f := &notificationWorkerFixture{dir: dir, events: make(chan Message, 32), exited: make(chan struct{})}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f.w, err = startRPCWorkerWithNotifyTimeout(engineManagerWorkerName, executable,
		[]string{"-test.run=^TestNotificationLifetimeHelper$"}, func(method string, params json.RawMessage) {
			f.events <- Message{Method: method, Params: params}
		}, budget, func(*rpcWorker) { close(f.exited) })
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		f.resume(t)
		f.w.Stop()
		if !waitClosed(f.w.Done(), 5*time.Second) {
			t.Error("helper cmd.Wait did not finish")
		}
	})
	f.event(t, "helper:ready")
	return f
}

func (f *notificationWorkerFixture) resume(t *testing.T) {
	t.Helper()
	f.resumeOnce.Do(func() {
		if err := os.WriteFile(filepath.Join(f.dir, "resume"), nil, 0600); err != nil {
			t.Error(err)
		}
	})
}

func (f *notificationWorkerFixture) event(t *testing.T, method string) Message {
	t.Helper()
	select {
	case msg := <-f.events:
		if msg.Method != method {
			t.Fatalf("notification=%s, want %s", msg.Method, method)
		}
		return msg
	case <-time.After(5 * time.Second):
		t.Fatalf("helper did not emit %s", method)
		return Message{}
	}
}

func notificationError(t *testing.T, result <-chan error) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("notification caller did not finish")
		return nil
	}
}

func (f *notificationWorkerFixture) pending(t *testing.T) <-chan error {
	t.Helper()
	result := make(chan error, 1)
	go func() { _, _, err := f.w.CallNoTimeout(context.Background(), "pending", nil); result <- err }()
	f.event(t, "helper:pending")
	return result
}

func TestNotificationLifetimeWorkerActiveDeadline(t *testing.T) {
	f := newNotificationWorker(t, "stall", 2*time.Second)
	pending := f.pending(t)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	// A frame larger than the anonymous pipe capacity physically blocks until
	// the production owner's hook closes StdinPipe. No cleanup has run yet.
	result := make(chan error, 1)
	go func() { result <- f.w.NotifyContext(ctx, "blocked", strings.Repeat("x", 4<<20)) }()
	f.event(t, "helper:blocked")
	err := notificationError(t, result)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("notification: %v", err)
	}
	if !errors.Is(notificationError(t, pending), errPeerClosed) {
		t.Fatal("pending Call was not released")
	}
	if _, err := f.w.stdin.Write([]byte("later\n")); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("native pipe still writable: %v", err)
	}
	if err := f.w.Notify("later", nil); !errors.Is(err, errPeerClosed) {
		t.Fatalf("retired notification: %v", err)
	}
	requireGateShutNow(t, f.w.Done(), "peer closure cannot manufacture process exit")
	requireGateShutNow(t, f.exited, "source exit fence must wait for cmd.Wait")
	f.w.interruptWrite()
	f.w.interruptWrite()
	f.resume(t)
	requireGateOpens(t, f.w.Done(), "real child exit after EOF")
	requireGateOpenNow(t, f.exited, "exit fence precedes Done")
}

func TestNotificationLifetimeWorkerWrappers(t *testing.T) {
	test := func(name string, send func(*rpcWorker) error) {
		t.Run(name, func(t *testing.T) {
			f := newNotificationWorker(t, "stall", time.Second)
			pending := f.pending(t)
			result := make(chan error, 1)
			go func() { result <- send(f.w) }()
			f.event(t, "helper:blocked")
			if err := notificationError(t, result); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("wrapper: %v", err)
			}
			if !errors.Is(notificationError(t, pending), errPeerClosed) {
				t.Fatal("pending call survived")
			}
			if _, err := f.w.stdin.Write(nil); !errors.Is(err, os.ErrClosed) {
				t.Fatalf("write completion/closure: %v", err)
			}
		})
	}
	test("Notify budget", func(w *rpcWorker) error { return w.Notify("blocked", strings.Repeat("x", 4<<20)) })
	test("SetLogLevel budget", func(w *rpcWorker) error { return w.SetLogLevel(strings.Repeat("x", 4<<20)) })
	if rpcWorkerNotifyTimeout != 60*time.Second {
		t.Fatalf("production budget=%s", rpcWorkerNotifyTimeout)
	}
}

func TestNotificationLifetimeWorkerQueuedCancellation(t *testing.T) {
	f := newNotificationWorker(t, "stall", 2*time.Second)
	pending := f.pending(t)
	active := make(chan error, 1)
	go func() { active <- f.w.Notify("active", strings.Repeat("x", 4<<20)) }()
	f.event(t, "helper:blocked")
	// The first send cannot complete while the child is paused. A cancelled
	// second send must neither close its pipe nor reach the child after resume.
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := f.w.NotifyContext(ctx, "queued", nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued send: %v", err)
	}
	requireGateShutNow(t, f.w.peer.Done(), "queued cancellation must preserve peer")
	f.resume(t)
	if err := notificationError(t, active); err != nil {
		t.Fatalf("active send: %v", err)
	}
	if err := f.w.Notify("after", nil); err != nil {
		t.Fatalf("later send: %v", err)
	}
	for _, want := range []string{"active", "after"} {
		msg := f.event(t, "helper:received")
		var got struct {
			Method string `json:"method"`
		}
		if err := json.Unmarshal(msg.Params, &got); err != nil || got.Method != want {
			t.Fatalf("received=%+v error=%v want=%s", got, err, want)
		}
	}
	f.w.Stop()
	if !errors.Is(notificationError(t, pending), errPeerClosed) {
		t.Fatal("pending Call survived exit")
	}
}

func TestNotificationLifetimeWorkerConcurrentStop(t *testing.T) {
	f := newNotificationWorker(t, "stall", 2*time.Second)
	pending := f.pending(t)
	active := make(chan error, 1)
	go func() { active <- f.w.Notify("active", strings.Repeat("x", 4<<20)) }()
	f.event(t, "helper:blocked")
	// Stop closes stdin before joining. The child resumes only after the send
	// has returned, proving no join in the interruption path is required.
	stopped := make(chan struct{})
	go func() { f.w.Stop(); close(stopped) }()
	if err := notificationError(t, active); err == nil {
		t.Fatal("closed pipe send succeeded")
	}
	if !errors.Is(notificationError(t, pending), errPeerClosed) {
		t.Fatal("pending call survived")
	}
	f.resume(t)
	requireGateOpens(t, stopped, "Stop join")
	f.w.Stop()
}

func TestNotificationLifetimeWorkerHealthyAndExit(t *testing.T) {
	t.Run("healthy delivery", func(t *testing.T) {
		f := newNotificationWorker(t, "read", time.Second)
		if err := f.w.SetLogLevel("debug"); err != nil {
			t.Fatal(err)
		}
		msg := f.event(t, "helper:received")
		var got struct {
			Method string                `json:"method"`
			Params applog.SetLevelParams `json:"params"`
		}
		if err := json.Unmarshal(msg.Params, &got); err != nil || got.Method != applog.SetLevelMethod || got.Params.Level != "debug" {
			t.Fatalf("received=%+v error=%v", got, err)
		}
	})
	t.Run("real process exit", func(t *testing.T) {
		f := newNotificationWorker(t, "exit", time.Second)
		f.resume(t)
		requireGateOpens(t, f.w.Done(), "child exit")
		requireGateOpenNow(t, f.exited, "source fence")
		if err := f.w.Notify("later", nil); !errors.Is(err, errPeerClosed) {
			t.Fatalf("after exit: %v", err)
		}
	})
}

type notificationLog struct {
	mu   sync.Mutex
	data bytes.Buffer
}

func (l *notificationLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.data.Write(p)
}
func (l *notificationLog) records(t *testing.T) []map[string]any {
	t.Helper()
	l.mu.Lock()
	data := bytes.Clone(l.data.Bytes())
	l.mu.Unlock()
	var records []map[string]any
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		var r map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		records = append(records, r)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return records
}

func TestNotificationLifetimeBoundSourceFailure(t *testing.T) {
	f := newNotificationWorker(t, "stall", time.Second)
	pending := f.pending(t)
	log := &notificationLog{}
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(log, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	b := &Broker{}
	source := b.activateEngineManagerSource()
	b.setEngineMgr(f.w)
	b.bindEngineMgr(f.w, source)
	req := nodepolicy.ModelMutationRequest{RequestID: "blocked-reply", Engine: "ollama"}
	b.replyModelMutation(source, f.w, req, nodepolicy.ModelMutationUnavailable, errors.New(strings.Repeat("x", 4<<20)))
	if !errors.Is(notificationError(t, pending), errPeerClosed) {
		t.Fatal("failed reply did not retire worker")
	}
	found := false
	for _, record := range log.records(t) {
		if record["msg"] == "worker notification failed" && record["worker"] == engineManagerWorkerName && record["method"] == nodepolicy.MethodModelMutationReply && record["err"] != nil {
			found = true
		}
		if _, ok := record["params"]; ok {
			t.Fatal("owner logged params")
		}
	}
	if !found {
		t.Fatal("missing worker/method/error diagnostic")
	}
	replacement, engine := newTestRPCWorkerPipe(t)
	next := b.activateEngineManagerSource()
	b.setEngineMgr(replacement)
	b.bindEngineMgr(replacement, next)
	b.replyModelMutation(source, f.w, req, 0, nil)
	// A subsequent frame must be the only one on the replacement stream.
	read := make(chan error, 1)
	go func() {
		msg, err := engine.Read()
		if err == nil && msg.Method != "replacement:probe" {
			err = fmt.Errorf("stale reply reached replacement: %s", msg.Method)
		}
		read <- err
	}()
	if err := replacement.Notify("replacement:probe", nil); err != nil {
		t.Fatal(err)
	}
	if err := notificationError(t, read); err != nil {
		t.Fatal(err)
	}
}

// The publisher models Task 2's owner policy without changing that branch's
// engine-manager files. The real broker reader and forwarding handler run here;
// these tests deliberately reproduce backpressure, not its daily-use frequency.
type notificationPublisher struct {
	codec      *Codec
	conn       net.Conn
	budget     time.Duration
	interrupts atomic.Int32
	terminal   chan struct{}
	once       sync.Once
}

func newNotificationPublisher(conn net.Conn, budget time.Duration) *notificationPublisher {
	p := &notificationPublisher{conn: conn, budget: budget, terminal: make(chan struct{})}
	p.codec = jsonrpc.NewCodecMaxFrameWithWriteInterrupt(conn, workerFrameBytes, func() { p.interrupts.Add(1); _ = conn.Close() })
	return p
}

func (p *notificationPublisher) notify(method string, params any) error {
	ctx, cancel := context.WithTimeout(context.Background(), p.budget)
	defer cancel()
	err := p.codec.NotifyContext(ctx, method, params)
	if err != nil {
		// Default publication failure also terminates on queued timeout, even
		// though the shared gate correctly does not call the active-write hook.
		p.once.Do(func() { close(p.terminal); _ = p.conn.Close() })
	}
	return err
}

type notificationForwardOutput struct {
	net.Conn
	entered chan struct{}
	once    sync.Once
}

func (w *notificationForwardOutput) Write(data []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	return w.Conn.Write(data)
}

type notificationForwardFixture struct {
	b         *Broker
	source    *engineManagerSource
	publisher *notificationPublisher
	client    *Codec
	output    *notificationForwardOutput
	pump      chan struct{}
	handled   chan string
}

func newNotificationForwardFixture(t *testing.T, budget time.Duration) *notificationForwardFixture {
	t.Helper()
	engine, reader := net.Pipe()
	downstream, client := net.Pipe()
	f := &notificationForwardFixture{publisher: newNotificationPublisher(engine, budget), client: NewCodecMaxFrame(client, workerFrameBytes),
		output: &notificationForwardOutput{Conn: downstream, entered: make(chan struct{})}, pump: make(chan struct{}), handled: make(chan string, 8)}
	f.b = &Broker{codec: NewCodec(f.output), engineSubscribed: true}
	f.source = f.b.activateEngineManagerSource()
	peer := NewPeer(NewCodecMaxFrame(reader, workerFrameBytes))
	go func() {
		defer close(f.pump)
		peer.Serve(nil, func(method string, params json.RawMessage) {
			f.b.forwardEngineNotificationFrom(f.source, method, params)
			f.handled <- method
		})
	}()
	t.Cleanup(func() {
		_ = engine.Close()
		_ = reader.Close()
		_ = downstream.Close()
		_ = client.Close()
		requireGateOpens(t, f.pump, "forwarding reader joined")
	})
	if err := f.publisher.notify("engine:pull-progress", map[string]string{"data": strings.Repeat("x", 1<<20)}); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, f.output.entered, "actual subscribed downstream write")
	return f
}

func (f *notificationForwardFixture) drain(t *testing.T) <-chan error {
	t.Helper()
	result := make(chan error, 1)
	go func() {
		for _, want := range []string{"engine:pull-progress", engineStateChangedMethod} {
			msg, err := f.client.Read()
			if err != nil {
				result <- err
				return
			}
			if msg.Method != want {
				result <- fmt.Errorf("forwarded method=%s, want %s", msg.Method, want)
				return
			}
		}
		result <- nil
	}()
	return result
}

func TestNotificationLifetimeForwardBackpressureRecovery(t *testing.T) {
	f := newNotificationForwardFixture(t, 2*time.Second)
	state, identity := make(chan error, 1), make(chan error, 1)
	go func() {
		state <- f.publisher.notify(engineStateChangedMethod, json.RawMessage(`{"engine":"ollama","running":true}`))
	}()
	go func() { identity <- f.publisher.notify(engineModelIdentityMethod, engineManagerSnapshot("ollama")) }()
	// The broker is still blocked in the actual forwarder, so neither frame
	// can be read. Both share the engine publisher's serialized write gate.
	select {
	case err := <-state:
		t.Fatalf("state escaped paused reader: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
	drained := f.drain(t)
	if err := notificationError(t, state); err != nil {
		t.Fatalf("state recovery: %v", err)
	}
	if err := notificationError(t, identity); err != nil {
		t.Fatalf("identity recovery: %v", err)
	}
	if err := notificationError(t, drained); err != nil {
		t.Fatal(err)
	}
	for range 3 {
		select {
		case <-f.handled:
		case <-time.After(2 * time.Second):
			t.Fatal("forwarder did not process every frame")
		}
	}
	r := f.b.nodePolicyRuntime()
	r.mu.Lock()
	run := r.engineIdentity["ollama"].EngineRunID
	r.mu.Unlock()
	if run != "run-ollama" {
		t.Fatalf("identity after recovery=%s", run)
	}
	if f.publisher.interrupts.Load() != 0 {
		t.Fatal("short pause interrupted publisher")
	}
	requireGateShutNow(t, f.publisher.terminal, "short pause preserves session")
}

func TestNotificationLifetimeForwardBackpressureTerminal(t *testing.T) {
	t.Run("active default deadline", func(t *testing.T) {
		f := newNotificationForwardFixture(t, 100*time.Millisecond)
		state, identity := make(chan error, 1), make(chan error, 1)
		go func() {
			state <- f.publisher.notify(engineStateChangedMethod, json.RawMessage(`{"engine":"ollama","running":true}`))
		}()
		go func() { identity <- f.publisher.notify(engineModelIdentityMethod, engineManagerSnapshot("ollama")) }()
		if err := notificationError(t, state); err == nil {
			t.Fatal("stalled state publication succeeded")
		}
		if err := notificationError(t, identity); err == nil {
			t.Fatal("stalled identity publication succeeded")
		}
		requireGateOpenNow(t, f.publisher.terminal, "default failure terminates session")
		if f.publisher.interrupts.Load() != 1 {
			t.Fatalf("active hook calls=%d", f.publisher.interrupts.Load())
		}
		if err := f.publisher.notify("later", nil); !errors.Is(err, errPeerClosed) {
			t.Fatalf("retired publication: %v", err)
		}
		requireGateShutNow(t, f.pump, "service-facing relay remains synchronous until downstream released")
	})
	t.Run("queued default deadline terminates separately", func(t *testing.T) {
		f := newNotificationForwardFixture(t, 100*time.Millisecond)
		active := make(chan error, 1)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		// The explicit send owns the gate, proven by the test transport's Write
		// barrier; the shorter default publication must time out while queued.
		probe := &notificationForwardOutput{Conn: f.publisher.conn, entered: make(chan struct{})}
		f.publisher.codec = jsonrpc.NewCodecMaxFrameWithWriteInterrupt(probe, workerFrameBytes, func() { f.publisher.interrupts.Add(1); _ = f.publisher.conn.Close() })
		go func() {
			active <- f.publisher.codec.NotifyContext(ctx, engineStateChangedMethod, json.RawMessage(`{"engine":"ollama","running":true}`))
		}()
		requireGateOpens(t, probe.entered, "explicit active engine write")
		if err := f.publisher.notify(engineModelIdentityMethod, engineManagerSnapshot("ollama")); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("queued default: %v", err)
		}
		if err := notificationError(t, active); err == nil {
			t.Fatal("terminal owner did not interrupt active frame")
		}
		requireGateOpenNow(t, f.publisher.terminal, "queued default failure policy")
		if f.publisher.interrupts.Load() != 1 {
			t.Fatalf("hook calls=%d, want active failed write only", f.publisher.interrupts.Load())
		}
	})
}
