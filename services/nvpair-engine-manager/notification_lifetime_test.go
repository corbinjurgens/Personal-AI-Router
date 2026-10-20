// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

// Each Write is synchronous, including while stalled. Only its transport owner
// (or an explicit healthy-reader resume) can finish it.
type notificationWriter struct {
	mu        sync.Mutex
	frames    [][]byte
	attempts  int
	blockFrom int
	entered   chan []byte
	written   chan []byte
	returned  chan struct{}
	closed    chan struct{}
	resume    chan struct{}
	once      sync.Once
}

func newNotificationWriter(blockFrom int) *notificationWriter {
	return &notificationWriter{blockFrom: blockFrom, entered: make(chan []byte, 16), written: make(chan []byte, 16), returned: make(chan struct{}, 16), closed: make(chan struct{}), resume: make(chan struct{})}
}

func (w *notificationWriter) Read([]byte) (int, error) {
	<-w.closed
	return 0, io.ErrClosedPipe
}

func (w *notificationWriter) Write(raw []byte) (int, error) {
	w.mu.Lock()
	w.attempts++
	block := w.attempts >= w.blockFrom
	w.mu.Unlock()
	if block {
		w.entered <- append([]byte(nil), raw...)
		defer func() { w.returned <- struct{}{} }()
		select {
		case <-w.closed:
			return 0, io.ErrClosedPipe
		case <-w.resume:
		}
	}
	w.mu.Lock()
	w.frames = append(w.frames, append([]byte(nil), raw...))
	w.mu.Unlock()
	w.written <- append([]byte(nil), raw...)
	return len(raw), nil
}

func (w *notificationWriter) Close() error {
	w.once.Do(func() { close(w.closed) })
	return nil
}

func lifetimeCodec(t *testing.T, w *notificationWriter, budget time.Duration) (context.Context, context.CancelFunc, *Codec) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	fail, join := newTerminalSession(ctx, cancel, w)
	t.Cleanup(join)
	return ctx, cancel, newOwnedCodec(ctx, w, fail, budget)
}

func lifetimeReceive[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-time.After(5 * time.Second):
		t.Fatal("notification owner did not finish")
		var zero T
		return zero
	}
}

func lifetimeMessage(t *testing.T, raw []byte) Message {
	t.Helper()
	var msg Message
	if err := json.Unmarshal(raw, &msg); err != nil {
		t.Fatal(err)
	}
	return msg
}

func TestNotificationLifetimeDefaultBudget(t *testing.T) {
	if notificationWriteBudget != 60*time.Second {
		t.Fatalf("production budget = %v", notificationWriteBudget)
	}
	w := newNotificationWriter(1)
	ctx, _, codec := lifetimeCodec(t, w, 100*time.Millisecond)
	done := make(chan error, 1)
	go func() { done <- codec.Notify("engine:state-changed", struct{}{}) }()
	lifetimeReceive(t, w.entered)
	if err := lifetimeReceive(t, done); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("default publication error = %v", err)
	}
	lifetimeReceive(t, w.returned)
	if ctx.Err() == nil {
		t.Fatal("default publication did not terminate the session")
	}
	if err := codec.Notify("engine:ready", nil); err == nil {
		t.Fatal("retired session accepted another notification")
	}
}

func TestNotificationLifetimeEarlierCallerDeadline(t *testing.T) {
	w := newNotificationWriter(1)
	root, _, codec := lifetimeCodec(t, w, notificationWriteBudget)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- codec.NotifyContext(ctx, "policy:request", nil) }()
	lifetimeReceive(t, w.entered)
	if err := lifetimeReceive(t, done); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("caller deadline = %v", err)
	}
	lifetimeReceive(t, w.returned)
	if root.Err() == nil {
		t.Fatal("active caller interruption did not terminate the session")
	}
}

func TestNotificationLifetimeQueuedCallerLeavesActiveWriter(t *testing.T) {
	w := newNotificationWriter(1)
	root, _, codec := lifetimeCodec(t, w, notificationWriteBudget)
	active := make(chan error, 1)
	go func() { active <- codec.Notify("engine:progress", nil) }()
	lifetimeReceive(t, w.entered)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if err := codec.NotifyContext(ctx, "queued-relay", nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued error = %v", err)
	}
	canceled, stop := context.WithCancel(context.Background())
	stop()
	if err := codec.NotifyContext(canceled, "pre-cancelled", nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled error = %v", err)
	}
	if root.Err() != nil {
		t.Fatal("queued caller cancellation closed the active session")
	}
	close(w.resume)
	if err := lifetimeReceive(t, active); err != nil {
		t.Fatal(err)
	}
	lifetimeReceive(t, w.returned)
	if err := codec.Notify("after-pause", nil); err != nil {
		t.Fatal(err)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.attempts != 2 || len(w.frames) != 2 {
		t.Fatalf("attempts=%d frames=%d; queued sends must emit nothing", w.attempts, len(w.frames))
	}
	if got := lifetimeMessage(t, w.frames[1]).Method; got != "after-pause" {
		t.Fatalf("late frame = %q", got)
	}
}

func TestNotificationLifetimeQueuedDefaultFailureClosesSession(t *testing.T) {
	w := newNotificationWriter(1)
	root, _, codec := lifetimeCodec(t, w, 40*time.Millisecond)
	// Responses retain the shared implementation's background context. The
	// queued default notification must terminate even behind such a frame.
	id := json.RawMessage("1")
	active := make(chan error, 1)
	go func() { active <- codec.Respond(&id, struct{}{}) }()
	lifetimeReceive(t, w.entered)
	if err := codec.Notify("queued-state", nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued default error = %v", err)
	}
	if err := lifetimeReceive(t, active); err == nil {
		t.Fatal("default failure left another active frame blocked")
	}
	lifetimeReceive(t, w.returned)
	if root.Err() == nil {
		t.Fatal("queued default failure did not terminate root work")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.attempts != 1 {
		t.Fatalf("queued default touched writer: %d attempts", w.attempts)
	}
}

type notificationRelayCase struct {
	name    string
	call    func(context.Context) error
	pending func() int
	late    func(*testing.T) func()
}

// Capture the real pending channel while the initial frame is in Write, then
// prove a late successful reply cannot be delivered after the call leaves.
func relayLifetimeProbe[T any](mu *sync.Mutex, pending *map[string]chan T, reply func(json.RawMessage), success func(string) T) (func() int, func(*testing.T) func()) {
	count := func() int { mu.Lock(); defer mu.Unlock(); return len(*pending) }
	late := func(t *testing.T) func() {
		t.Helper()
		mu.Lock()
		var id string
		var ch chan T
		for key, value := range *pending {
			id, ch = key, value
		}
		mu.Unlock()
		if id == "" || count() != 1 {
			t.Fatal("initial send did not register exactly one waiter")
		}
		raw, err := json.Marshal(success(id))
		if err != nil {
			t.Fatal(err)
		}
		return func() {
			reply(raw)
			if count() != 0 || len(ch) != 0 {
				t.Fatal("late reply reached a retired waiter")
			}
		}
	}
	return count, late
}

func notificationRelayCases(m *Manager) []notificationRelayCase {
	pendingPolicy, latePolicy := relayLifetimeProbe(&m.policyRelay.mu, &m.policyRelay.pending, m.policyRelay.reply, func(id string) policyRelayReply {
		return policyRelayReply{ID: id, Result: json.RawMessage(`{}`)}
	})
	pendingSettings, lateSettings := relayLifetimeProbe(&m.settingsRelay.mu, &m.settingsRelay.pending, m.settingsRelay.reply, func(id string) settingsReply {
		return settingsReply{ID: id, Result: json.RawMessage(`{}`)}
	})
	pendingSwitch, lateSwitch := relayLifetimeProbe(&m.switchRelay.mu, &m.switchRelay.pending, m.switchRelay.reply, func(id string) switchRelayReply {
		return switchRelayReply{ID: id, Result: json.RawMessage(`{}`)}
	})
	pendingMutation, lateMutation := relayLifetimeProbe(&m.modelMutation.mu, &m.modelMutation.pending, m.modelMutation.reply, func(id string) nodepolicy.ModelMutationReply {
		return nodepolicy.ModelMutationReply{RequestID: id, Engine: "fake", OK: true}
	})
	pendingResident, lateResident := relayLifetimeProbe(&m.residentDispatch.mu, &m.residentDispatch.pending, m.residentDispatch.reply, func(id string) residentDispatchReply {
		return residentDispatchReply{ID: id, Result: &nodepolicy.DispatchResidentResult{Dispatched: true}}
	})
	return []notificationRelayCase{
		{"policy", func(ctx context.Context) error {
			_, err := m.policyRelay.call(ctx, "get", json.RawMessage(`{}`), "")
			return err
		}, pendingPolicy, latePolicy},
		{"settings", func(ctx context.Context) error {
			_, err := m.settingsRelay.call(ctx, "get", settings.Request{}, "")
			return err
		}, pendingSettings, lateSettings},
		{"switch", func(ctx context.Context) error {
			_, err := m.switchRelay.call(ctx, settings.SwitchStatusMethod, "", json.RawMessage(`{}`))
			return err
		}, pendingSwitch, lateSwitch},
		{"mutation", func(ctx context.Context) error { return m.modelMutation.call(ctx, "fake") }, pendingMutation, lateMutation},
		{"resident", func(ctx context.Context) error {
			return m.residentDispatch.call(ctx, nodepolicy.DispatchResidentParams{})
		}, pendingResident, lateResident},
	}
}

func TestNotificationLifetimeRelayInitialSend(t *testing.T) {
	for index := range notificationRelayCases(&Manager{}) {
		// Each case gets a fresh stream; an interrupted frame retires it.
		w := newNotificationWriter(1)
		_, _, codec := lifetimeCodec(t, w, notificationWriteBudget)
		ex := newTestExecutor(t, mutationManifest(t, "fake"))
		m := NewManager(codec, ex, nil, "")
		c := notificationRelayCases(m)[index]
		t.Run(c.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			go func() { done <- c.call(ctx) }()
			lifetimeReceive(t, w.entered)
			late := c.late(t)
			cancel()
			if err := lifetimeReceive(t, done); !errors.Is(err, context.Canceled) {
				t.Fatalf("stalled send = %v", err)
			}
			lifetimeReceive(t, w.returned)
			late()
			w.mu.Lock()
			attempts := w.attempts
			w.mu.Unlock()
			if attempts != 1 {
				t.Fatalf("initial-send failure sent cleanup: %d writes", attempts)
			}
		})
	}
}

func TestNotificationLifetimeRelayPreCancelled(t *testing.T) {
	w := newNotificationWriter(1)
	_, _, codec := lifetimeCodec(t, w, notificationWriteBudget)
	m := NewManager(codec, newTestExecutor(t, mutationManifest(t, "fake")), nil, "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, c := range notificationRelayCases(m) {
		t.Run(c.name, func(t *testing.T) {
			if err := c.call(ctx); !errors.Is(err, context.Canceled) || c.pending() != 0 {
				t.Fatalf("cancelled relay = %v, pending=%d", err, c.pending())
			}
		})
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.attempts != 0 {
		t.Fatalf("pre-cancelled relays made %d writes", w.attempts)
	}
}

func TestNotificationLifetimeRelayQueuedCancellation(t *testing.T) {
	for index := range notificationRelayCases(&Manager{}) {
		w := newNotificationWriter(1)
		root, _, codec := lifetimeCodec(t, w, notificationWriteBudget)
		m := NewManager(codec, newTestExecutor(t, mutationManifest(t, "fake")), nil, "")
		c := notificationRelayCases(m)[index]
		t.Run(c.name, func(t *testing.T) {
			active := make(chan error, 1)
			go func() { active <- codec.Notify("engine:progress", nil) }()
			lifetimeReceive(t, w.entered)
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
			defer cancel()
			if err := c.call(ctx); !errors.Is(err, context.DeadlineExceeded) || c.pending() != 0 {
				t.Fatalf("queued relay = %v, pending=%d", err, c.pending())
			}
			if root.Err() != nil {
				t.Fatal("queued relay retired another active publication")
			}
			close(w.resume)
			if err := lifetimeReceive(t, active); err != nil {
				t.Fatal(err)
			}
			lifetimeReceive(t, w.returned)
			w.mu.Lock()
			attempts := w.attempts
			w.mu.Unlock()
			if attempts != 1 {
				t.Fatalf("queued relay emitted a late frame: %d attempts", attempts)
			}
		})
	}
}

func grantNotification(t *testing.T, m *Manager, method string, params any) {
	t.Helper()
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var request struct {
		ID        string `json:"id"`
		RequestID string `json:"requestId"`
		Engine    string `json:"engine"`
	}
	if err := json.Unmarshal(raw, &request); err != nil {
		t.Fatal(err)
	}
	var response any
	var reply func(json.RawMessage)
	switch method {
	case policyRequestMethod:
		response, reply = policyRelayReply{ID: request.ID, Result: json.RawMessage(`{}`)}, m.policyRelay.reply
	case "engine:settings-request":
		response, reply = settingsReply{ID: request.ID, Result: json.RawMessage(`{}`)}, m.settingsRelay.reply
	case "engine:switch-request":
		response, reply = switchRelayReply{ID: request.ID, Result: json.RawMessage(`{}`)}, m.switchRelay.reply
	case nodepolicy.MethodModelMutationRequest:
		response, reply = nodepolicy.ModelMutationReply{RequestID: request.RequestID, Engine: request.Engine, OK: true}, m.modelMutation.reply
	case residentDispatchRequestMethod:
		response, reply = residentDispatchReply{ID: request.ID, Result: &nodepolicy.DispatchResidentResult{Dispatched: true}}, m.residentDispatch.reply
	default:
		return // cancellation cleanup has no success reply
	}
	raw, err = json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	reply(raw)
}

func TestNotificationLifetimeCancelledReplyCannotSucceed(t *testing.T) {
	for index := range notificationRelayCases(&Manager{}) {
		w := newNotificationWriter(100)
		_, _, codec := lifetimeCodec(t, w, notificationWriteBudget)
		m := NewManager(codec, newTestExecutor(t, mutationManifest(t, "fake")), nil, "")
		c := notificationRelayCases(m)[index]
		t.Run(c.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			send := func(ctx context.Context, method string, params any) error {
				if err := codec.NotifyContext(ctx, method, params); err != nil {
					return err
				}
				cancel()
				grantNotification(t, m, method, params)
				return nil
			}
			m.policyRelay.send, m.settingsRelay.send, m.switchRelay.send = send, send, send
			m.modelMutation.send, m.residentDispatch.send = send, send
			if err := c.call(ctx); !errors.Is(err, context.Canceled) || c.pending() != 0 {
				t.Fatalf("cancelled queued grant = %v, pending=%d", err, c.pending())
			}
		})
	}
}

func TestNotificationLifetimeRelayCleanupTail(t *testing.T) {
	for index, name := range []string{"policy", "settings", "switch"} {
		t.Run(name, func(t *testing.T) {
			w := newNotificationWriter(2)
			_, _, codec := lifetimeCodec(t, w, notificationWriteBudget)
			m := NewManager(codec, newTestExecutor(t, mutationManifest(t, "fake")), nil, "")
			c := notificationRelayCases(m)[index]
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			// The first frame completes, then cancellation begins reply cleanup.
			// Observe that completion without replacing the actual send seam.
			sent := make(chan struct{})
			var initial sync.Once
			observe := func(ctx context.Context, method string, p any) error {
				err := codec.NotifyContext(ctx, method, p)
				initial.Do(func() { close(sent) })
				return err
			}
			m.policyRelay.send, m.settingsRelay.send, m.switchRelay.send = observe, observe, observe
			done := make(chan error, 1)
			go func() { done <- c.call(ctx) }()
			lifetimeReceive(t, sent)
			late := c.late(t)
			start := time.Now()
			cancel()
			msg := lifetimeMessage(t, lifetimeReceive(t, w.entered))
			if msg.Method != map[string]string{"policy": policyCancelMethod, "settings": "engine:settings-cancel", "switch": "engine:switch-cancel"}[name] {
				t.Fatalf("cleanup method = %q", msg.Method)
			}
			late() // pending entry must be gone before the stalled cleanup write
			if err := lifetimeReceive(t, done); !errors.Is(err, context.Canceled) {
				t.Fatalf("cleanup replaced request error: %v", err)
			}
			lifetimeReceive(t, w.returned)
			if elapsed := time.Since(start); elapsed < 900*time.Millisecond || elapsed > 2*time.Second {
				t.Fatalf("independent one-second cleanup tail = %v", elapsed)
			}
		})
	}
}

func TestNotificationLifetimePublicationLocks(t *testing.T) {
	test := func(name string, emit func(*Executor)) {
		t.Run(name, func(t *testing.T) {
			w := newNotificationWriter(1)
			_, cancel, codec := lifetimeCodec(t, w, notificationWriteBudget)
			ex := newTestExecutor(t, mutationManifest(t, "fake"))
			ex.emit = func(method string, p any) { _ = codec.Notify(method, p) }
			done := make(chan struct{})
			go func() { defer close(done); emit(ex) }()
			lifetimeReceive(t, w.entered)
			if ex.publishMu.TryLock() {
				ex.publishMu.Unlock()
				t.Fatal("publication did not retain ordering lock during send")
			}
			cancel()
			lifetimeReceive(t, done)
			lifetimeReceive(t, w.returned)
			if !ex.publishMu.TryLock() {
				t.Fatal("publication lock survived transport cancellation")
			}
			ex.publishMu.Unlock()
		})
	}
	test("state", func(ex *Executor) { ex.emitState("fake") })
	test("identity", func(ex *Executor) { ex.publishModelIdentity(nil) })
}

func TestNotificationLifetimeMutationLock(t *testing.T) {
	w := newNotificationWriter(1)
	_, _, codec := lifetimeCodec(t, w, notificationWriteBudget)
	ex, engine := mutationEngineFixture(t, "fake")
	m := NewManager(codec, ex, nil, "")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := ex.ModelDelete(ctx, "fake", "m:1"); done <- err }()
	lifetimeReceive(t, w.entered)
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	select {
	case st.contentMu <- struct{}{}:
		<-st.contentMu
		t.Fatal("mutation did not retain content guard during withdrawal")
	default:
	}
	cancel()
	if err := lifetimeReceive(t, done); !errors.Is(err, context.Canceled) {
		t.Fatalf("mutation = %v", err)
	}
	lifetimeReceive(t, w.returned)
	select {
	case st.contentMu <- struct{}{}:
		<-st.contentMu
	default:
		t.Fatal("content guard survived cancellation")
	}
	assertNoPending(t, &m.modelMutation)
	if effects := engine.seen(); len(effects) != 0 {
		t.Fatalf("failed withdrawal caused mutation effects: %v", effects)
	}
}

func TestNotificationLifetimeStopReleasesLifecycleLock(t *testing.T) {
	w := newNotificationWriter(1)
	_, cancel, codec := lifetimeCodec(t, w, notificationWriteBudget)
	ex := newTestExecutor(t, mutationManifest(t, "fake"))
	ex.emit = func(method string, p any) { _ = codec.Notify(method, p) }
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	// No actual engine: an owned-process-missing stop emits state and refuses.
	st.running = true
	done := make(chan error, 1)
	go func() { done <- ex.Stop("fake") }()
	lifetimeReceive(t, w.entered)
	if st.opMu.TryLock() {
		st.opMu.Unlock()
		t.Fatal("stop did not retain lifecycle lock while publishing")
	}
	cancel()
	if err := lifetimeReceive(t, done); err == nil {
		t.Fatal("missing process stop unexpectedly succeeded")
	}
	lifetimeReceive(t, w.returned)
	if !st.opMu.TryLock() {
		t.Fatal("lifecycle lock survived transport cancellation")
	}
	st.opMu.Unlock()
	stopped := make(chan struct{})
	go func() { defer close(stopped); ex.StopAll() }()
	lifetimeReceive(t, stopped)
}

func TestNotificationLifetimeManagerRunStopsAfterFailure(t *testing.T) {
	w := newNotificationWriter(100)
	ctx, _, codec := lifetimeCodec(t, w, 100*time.Millisecond)
	ex := newTestExecutor(t, testEngineManifest(fakeEngineBin))
	ex.loadedPollInterval = 0
	ex.emit = func(method string, p any) { _ = codec.Notify(method, p) }
	ex.reporter = NewReporter(codec)
	if err := ex.Start(ctx, "fake"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(ex.StopAll)
	st, err := ex.state("fake")
	if err != nil {
		t.Fatal(err)
	}
	st.mu.Lock()
	proc := st.proc
	st.mu.Unlock()
	if proc == nil {
		t.Fatal("fixture did not start an owned process")
	}
	w.mu.Lock()
	w.blockFrom = w.attempts + 3 // ready and subscription succeed before stall
	w.mu.Unlock()
	for len(w.written) > 0 {
		<-w.written
	}
	m := NewManager(codec, ex, nil, "")
	done := make(chan error, 1)
	go func() { done <- m.Run(ctx) }()
	lifetimeReceive(t, w.written)
	lifetimeReceive(t, w.written)
	// An ordinary publication shares the same production session owner.
	emitted := make(chan struct{})
	go func() { defer close(emitted); ex.emitState("fake") }()
	lifetimeReceive(t, w.entered)
	lifetimeReceive(t, emitted)
	lifetimeReceive(t, w.returned)
	if err := lifetimeReceive(t, done); err != nil {
		t.Fatalf("Run after terminal failure = %v", err)
	}
	if !ex.shuttingDown.Load() {
		t.Fatal("terminal failure bypassed Manager.Run/StopAll")
	}
	lifetimeReceive(t, proc.done)
	st.mu.Lock()
	running := st.running
	st.mu.Unlock()
	if running {
		t.Fatal("terminal failure left the owned engine running")
	}
}

func TestNotificationLifetimeReadyFailureReturned(t *testing.T) {
	w := newNotificationWriter(1)
	ctx, _, codec := lifetimeCodec(t, w, 40*time.Millisecond)
	ex := newTestExecutor(t, mutationManifest(t, "fake"))
	m := NewManager(codec, ex, nil, "")
	done := make(chan error, 1)
	go func() { done <- m.Run(ctx) }()
	lifetimeReceive(t, w.entered)
	if err := lifetimeReceive(t, done); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("ready send failure = %v", err)
	}
	lifetimeReceive(t, w.returned)
}
