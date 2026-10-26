// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

func startTestEngineNotifications(t *testing.T, b *Broker, disconnect func()) func() {
	t.Helper()
	b.disconnectClient = disconnect
	stop := b.startEngineNotificationRelay(context.Background())
	t.Cleanup(stop)
	return stop
}

func readEngineNotification(t *testing.T, client *Codec, method string, sequence int) {
	t.Helper()
	msg, err := client.Read()
	if err != nil {
		t.Fatalf("read engine notification: %v", err)
	}
	var params struct {
		Sequence int `json:"sequence"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		t.Fatalf("decode engine notification: %v", err)
	}
	if !msg.IsNotification() || msg.Method != method || params.Sequence != sequence {
		t.Fatalf("notification method=%s sequence=%d, want %s %d", msg.Method, params.Sequence, method, sequence)
	}
}

func readEngineNotificationReset(t *testing.T, client *Codec) {
	t.Helper()
	msg, err := client.Read()
	if err != nil {
		t.Fatal(err)
	}
	if !msg.IsNotification() || msg.Method != notifyEngineNotificationsReset || string(msg.Params) != "null" {
		t.Fatalf("private reset = %+v", msg)
	}
}

func TestEngineNotificationRelayPreservesOrderAndOwnsPayload(t *testing.T) {
	f := newNotificationForwardFixture(t, 2*time.Second)
	raw := json.RawMessage(`{"sequence":1}`)
	f.b.forwardEngineNotificationFrom(f.source, "engine:pull-progress", raw)
	// Both later events wait behind the first blocked service-facing Write.
	// Reusing the caller's bytes must not alter the queue's first payload.
	raw[len(raw)-2] = '9'
	f.b.forwardEngineNotificationFrom(f.source, "engine:install-progress", json.RawMessage(`{"sequence":2}`))
	if _, err := f.client.Read(); err != nil {
		t.Fatal(err)
	}
	readEngineNotification(t, f.client, "engine:pull-progress", 1)
	readEngineNotification(t, f.client, "engine:install-progress", 2)
}

func TestEngineNotificationRelayRejectsRetiredSources(t *testing.T) {
	f := newNotificationForwardFixture(t, 2*time.Second)
	f.b.forwardEngineNotificationFrom(f.source, "engine:pull-progress", json.RawMessage(`{"sequence":1}`))
	next := f.b.activateEngineManagerSource()
	// One old event was queued before retirement; another arrives afterwards.
	f.b.forwardEngineNotificationFrom(f.source, "engine:pull-progress", json.RawMessage(`{"sequence":2}`))
	f.b.forwardEngineNotificationFrom(next, "engine:pull-progress", json.RawMessage(`{"sequence":3}`))
	// The frame already in Write was accepted while its source was current.
	if _, err := f.client.Read(); err != nil {
		t.Fatal(err)
	}
	readEngineNotification(t, f.client, "engine:pull-progress", 3)
}

func TestEngineNotificationRelayPreservesSubscriptionAndPrivacy(t *testing.T) {
	f := newNotificationForwardFixture(t, 2*time.Second)
	f.b.engineMu.Lock()
	f.b.engineSubscribed = false
	f.b.engineMu.Unlock()
	f.b.forwardEngineNotificationFrom(f.source, "engine:pull-progress", json.RawMessage(`{"sequence":1}`))
	f.b.engineMu.Lock()
	f.b.engineSubscribed = true
	f.b.engineMu.Unlock()
	f.b.forwardEngineNotificationFrom(f.source, "routing:changed", json.RawMessage(`{"sequence":2}`))
	f.b.forwardEngineNotificationFrom(f.source, notifyEngineNotificationsReset, nil)
	f.b.forwardClusterManagerNotification(notifyEngineNotificationsReset, nil)
	f.b.forwardEngineNotificationFrom(f.source, "engine:pull-progress", json.RawMessage(`{"sequence":3}`))
	if _, err := f.client.Read(); err != nil {
		t.Fatal(err)
	}
	readEngineNotification(t, f.client, "engine:pull-progress", 3)
}

func TestEngineNotificationRelayByteLimitIncludesActiveFrame(t *testing.T) {
	f := newNotificationForwardFixture(t, 2*time.Second)
	r := f.b.engineNotifications
	const method = "engine:pull-progress"
	r.mu.Lock()
	activeBytes := r.bytes
	remaining := engineNotificationQueueBytes - r.bytes - len(method)
	r.mu.Unlock()
	if remaining < 2 {
		t.Fatal("fixture's first notification exhausted byte allowance")
	}
	f.b.forwardEngineNotificationFrom(f.source, method, json.RawMessage(`"`+strings.Repeat("x", remaining-2)+`"`))
	r.mu.Lock()
	used, queued := r.bytes, len(r.queue)
	r.mu.Unlock()
	if used != engineNotificationQueueBytes || queued != 1 {
		t.Fatalf("byte-bound queue used=%d queued=%d", used, queued)
	}
	f.b.forwardEngineNotificationFrom(f.source, method, json.RawMessage(`{}`))
	r.mu.Lock()
	used, queued = r.bytes, len(r.queue)
	dropped := r.dropped
	pending := r.resetPending
	r.mu.Unlock()
	if used != activeBytes || queued != 0 || !dropped || !pending {
		t.Fatalf("overflow retained pending payloads: bytes=%d queued=%d dropped=%v", used, queued, dropped)
	}
	requireGateShutNow(t, f.cancelled, "byte overflow preserves broker lifetime")
	requireGateShutNow(t, f.disconnected, "byte overflow preserves ordinary transport")
}

func TestEngineNotificationRelayExplicitSubscribeResumesAfterOverflow(t *testing.T) {
	f := newNotificationForwardFixture(t, 2*time.Second)
	for range engineNotificationQueueFrames + 1 {
		f.b.forwardEngineNotificationFrom(f.source, "engine:pull-progress", json.RawMessage(`{"sequence":1}`))
	}
	if _, err := f.client.Read(); err != nil {
		t.Fatal(err)
	}
	readEngineNotificationReset(t, f.client)
	id := json.RawMessage(`"resume"`)
	subscribed := make(chan struct{})
	go func() {
		f.b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "engine:subscribe", Params: json.RawMessage(`{}`)})
		close(subscribed)
	}()
	response, err := f.client.Read()
	if err != nil {
		t.Fatal(err)
	}
	var result SubscriptionResult
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatal(err)
	}
	if response.Error != nil || response.ID == nil || string(*response.ID) != string(id) || !result.Subscribed {
		t.Fatalf("explicit resubscription response=%+v result=%+v", response, result)
	}
	requireGateOpens(t, subscribed, "explicit subscription handler completed")
	f.b.forwardEngineNotificationFrom(f.source, "engine:pull-progress", json.RawMessage(`{"sequence":3}`))
	readEngineNotification(t, f.client, "engine:pull-progress", 3)
}

func TestEngineNotificationRelayResetPendingIsBounded(t *testing.T) {
	f := newNotificationForwardFixture(t, 2*time.Second)
	for range engineNotificationQueueFrames * 10 {
		f.b.queueEngineNotification(f.source, "engine:pull-progress", json.RawMessage(`{}`))
	}
	r := f.b.engineNotifications
	r.mu.Lock()
	pending, queued, wakes := r.resetPending, len(r.queue), len(r.wake)
	r.mu.Unlock()
	if !pending || queued != 0 || wakes > 1 {
		t.Fatalf("overflow ownership: pending=%v queued=%d wakes=%d", pending, queued, wakes)
	}
	requireGateShutNow(t, f.output.resetEntered, "reset waits for actual active write completion")
	if _, err := f.client.Read(); err != nil {
		t.Fatal(err)
	}
	readEngineNotificationReset(t, f.client)
	// The next frame is an ordinary barrier: no duplicate reset was queued.
	ordinary := make(chan error, 1)
	go func() { ordinary <- f.b.codec.Notify("ordinary:probe", nil) }()
	msg, err := f.client.Read()
	if err != nil || msg.Method != "ordinary:probe" {
		t.Fatalf("frame after one reset: %+v / %v", msg, err)
	}
	if err := notificationError(t, ordinary); err != nil {
		t.Fatal(err)
	}
}

func TestEngineNotificationRelayCancellationJoinsResetWriter(t *testing.T) {
	f := newNotificationForwardFixture(t, 2*time.Second)
	for range engineNotificationQueueFrames + 1 {
		f.b.queueEngineNotification(f.source, "engine:pull-progress", json.RawMessage(`{}`))
	}
	if _, err := f.client.Read(); err != nil {
		t.Fatal(err)
	}
	requireGateOpens(t, f.output.resetEntered, "private reset owns actual blocked Write")
	r := f.b.engineNotifications
	r.cancel()
	requireGateOpens(t, f.disconnected, "session cancellation interrupts reset Write")
	requireGateOpens(t, r.done, "reset writer joined")
	r.mu.Lock()
	retained := r.resetPending || r.bytes != 0 || len(r.queue) != 0
	r.mu.Unlock()
	if retained {
		t.Fatal("joined reset retained pending ownership")
	}
}

func TestEngineNotificationRelayCancellationJoinsBlockedWriter(t *testing.T) {
	f := newNotificationForwardFixture(t, 2*time.Second)
	r := f.b.engineNotifications
	r.cancel()
	requireGateOpens(t, f.disconnected, "session cancellation interrupts actual client Write")
	requireGateOpens(t, r.done, "session cancellation joins writer")
	// A late worker callback must neither enqueue nor resurrect a sender.
	f.b.forwardEngineNotificationFrom(f.source, "engine:pull-progress", json.RawMessage(`{}`))
	r.mu.Lock()
	used, queued := r.bytes, len(r.queue)
	r.mu.Unlock()
	if used != 0 || queued != 0 {
		t.Fatalf("cancelled relay retained late payload: bytes=%d queued=%d", used, queued)
	}
}

func TestEngineNotificationRelayCancellationInterruptsOrdinaryGateOwner(t *testing.T) {
	server, client := net.Pipe()
	t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
	output := &notificationForwardOutput{Conn: server, entered: make(chan struct{})}
	clientCancelled := make(chan struct{})
	b := &Broker{codec: NewCodec(output), cancel: func() { close(clientCancelled) }, engineSubscribed: true}
	source := b.activateEngineManagerSource()
	stop := startTestEngineNotifications(t, b, func() { _ = server.Close() })
	ordinary := make(chan error, 1)
	go func() { ordinary <- b.codec.Notify("ordinary", nil) }()
	requireGateOpens(t, output.entered, "ordinary frame owns client write gate")
	b.forwardEngineNotificationFrom(source, "engine:pull-progress", json.RawMessage(`{}`))
	stopped := make(chan struct{})
	go func() { stop(); close(stopped) }()
	requireGateOpens(t, stopped, "queued relay cancellation joins writer and interrupt callback")
	requireGateOpenNow(t, clientCancelled, "relay retires the owning client session")
	if err := notificationError(t, ordinary); err == nil {
		t.Fatal("cancelled client left an ordinary frame writing")
	}
}

func TestEngineNotificationRelayClientWriteFailureRetiresSession(t *testing.T) {
	server, client := net.Pipe()
	t.Cleanup(func() { _ = server.Close(); _ = client.Close() })
	clientCancelled := make(chan struct{})
	b := &Broker{codec: NewCodec(server), cancel: func() { close(clientCancelled) }, engineSubscribed: true}
	startTestEngineNotifications(t, b, func() { _ = server.Close() })
	source := b.activateEngineManagerSource()
	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	b.forwardEngineNotificationFrom(source, "engine:pull-progress", json.RawMessage(`{}`))
	requireGateOpens(t, clientCancelled, "failed relay write cancels the owning client session")
	requireGateOpens(t, b.engineNotifications.done, "failed writer joined")
}

func TestEngineNotificationRelayStopJoinsInterruptCallback(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		entered, release := make(chan struct{}), make(chan struct{})
		defer close(release)
		b := &Broker{disconnectClient: func() { close(entered); <-release }}
		parent, cancel := context.WithCancel(context.Background())
		defer cancel()
		stop := b.startEngineNotificationRelay(parent)
		cancel()
		<-entered
		stopped := make(chan struct{})
		go func() { stop(); close(stopped) }()
		synctest.Wait()
		select {
		case <-stopped:
			t.Fatal("relay stop returned before actual interrupt callback completion")
		default:
		}
		release <- struct{}{}
		<-stopped
		requireGateOpenNow(t, b.engineNotifications.done, "stop joins idle sender too")
	})
}

func TestEngineNotificationRelayConcurrentEnqueueAndStop(t *testing.T) {
	b := &Broker{codec: NewCodec(&syncBuffer{})}
	source := b.activateEngineManagerSource()
	stop := startTestEngineNotifications(t, b, func() {})
	start := make(chan struct{})
	var producers sync.WaitGroup
	for range 4 {
		producers.Add(1)
		go func() {
			defer producers.Done()
			<-start
			for range 100 {
				b.queueEngineNotification(source, "engine:pull-progress", json.RawMessage(`{}`))
			}
		}()
	}
	stopped := make(chan struct{})
	go func() { <-start; stop(); close(stopped) }()
	close(start)
	producers.Wait()
	requireGateOpens(t, stopped, "concurrent enqueue and stop complete without send/close race")
	r := b.engineNotifications
	r.mu.Lock()
	used, queued := r.bytes, len(r.queue)
	r.mu.Unlock()
	if used != 0 || queued != 0 {
		t.Fatalf("joined relay retained concurrent payloads: bytes=%d queued=%d", used, queued)
	}
}
