// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"sync"
)

const engineNotificationQueueFrames = 32

// Private parent control: nvpair-service consumes it before client broadcast.
const notifyEngineNotificationsReset = "service/engine-notifications-reset"

// The byte limit includes the frame in flight, not just the waiting channel.
// These are ordinary engine events; history never enters this queue.
const engineNotificationQueueBytes = workerFrameBytes

type engineNotification struct {
	source *engineManagerSource
	method string
	params json.RawMessage
}

type engineNotificationRelay struct {
	ctx          context.Context
	cancel       context.CancelFunc
	done         chan struct{}
	queue        chan engineNotification
	wake         chan struct{}
	mu           sync.Mutex
	bytes        int
	dropped      bool
	resetPending bool
}

// startEngineNotificationRelay runs once before worker publication. Broker has
// one service-facing client, so the relay shares that session's fate. Main's
// captured disconnectClient closes the actual input/output owners; cancellation
// alone cannot interrupt a plain codec Write or a different writer's gate.
func (b *Broker) startEngineNotificationRelay(parent context.Context) func() {
	ctx, cancel := context.WithCancel(parent)
	r := &engineNotificationRelay{ctx: ctx, cancel: cancel, done: make(chan struct{}),
		queue: make(chan engineNotification, engineNotificationQueueFrames), wake: make(chan struct{}, 1)}
	b.engineNotifications = r
	interrupted := make(chan struct{})
	context.AfterFunc(ctx, func() {
		if b.disconnectClient != nil {
			b.disconnectClient()
		}
		if b.cancel != nil {
			b.cancel()
		}
		close(interrupted)
	})
	go func() {
		defer close(r.done)
		defer func() {
			// Serialize with a producer admitted just before cancellation, then
			// release queued payloads before reporting actual sender completion.
			r.mu.Lock()
			defer r.mu.Unlock()
			for len(r.queue) != 0 {
				<-r.queue
			}
			r.bytes = 0
			r.resetPending = false
		}()
		for {
			select {
			case <-ctx.Done():
				return
			case <-r.wake:
			}
			for ctx.Err() == nil {
				event, reset, ok := r.next()
				if !ok {
					break
				}
				// A queued event from a retired manager cannot describe its
				// replacement. Never retain source locks across transport IO.
				var err error
				if reset {
					err = b.codec.NotifyContext(ctx, notifyEngineNotificationsReset, nil)
				} else if b.engineManagerSourceCurrent(event.source) {
					err = b.codec.NotifyContext(ctx, event.method, event.params)
				}
				r.mu.Lock()
				r.bytes -= len(event.method) + len(event.params)
				r.mu.Unlock()
				if err != nil {
					cancel()
					return
				}
			}
		}
	}()
	return func() {
		cancel()
		<-interrupted
		<-r.done
	}
}

// Only the sole sender takes work. A fixed pending bit takes priority over
// resumed engine events; reset never waits for space in the event queue and
// never acquires the codec gate until the previous actual Write has returned.
func (r *engineNotificationRelay) next() (engineNotification, bool, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.resetPending {
		r.resetPending = false
		return engineNotification{}, true, true
	}
	select {
	case event := <-r.queue:
		return event, false, true
	default:
		return engineNotification{}, false, false
	}
}

func (r *engineNotificationRelay) signal() {
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

// queueEngineNotification never waits for client IO or queue space. Overflow
// drops the slow client's engine subscription and pending events. The admitted
// event (already dequeued by the sole sender) remains owned until it finishes
// or ordinary session teardown interrupts it. The same sender then delivers
// one private reset to the service, which owns attached-client disconnection
// and recovery. Overflow must not cancel the broker or engine-manager.
// Params becomes queue-owned here; the engine read pump may immediately reuse
// or release its message. Per-client engine-event order is channel FIFO.
func (b *Broker) queueEngineNotification(source *engineManagerSource, method string, params json.RawMessage) {
	r := b.engineNotifications
	if r == nil || method == notifyEngineNotificationsReset || !b.engineManagerSourceCurrent(source) {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ctx.Err() != nil || r.dropped {
		return
	}
	size := len(method) + len(params)
	if len(r.queue) == cap(r.queue) || size > engineNotificationQueueBytes-r.bytes {
		r.dropped = true
		r.resetPending = true
		b.engineMu.Lock()
		b.engineSubscribed = false
		b.engineMu.Unlock()
		for {
			select {
			case event := <-r.queue:
				r.bytes -= len(event.method) + len(event.params)
			default:
				r.signal()
				slog.Warn("dropped slow service-facing engine subscription", "category", "engine-notification-overflow")
				return
			}
		}
	}
	r.bytes += size
	r.queue <- engineNotification{source: source, method: method, params: bytes.Clone(params)}
	r.signal()
}

// An explicit subscription can resume delivery after the client drains its
// transport. Serialize its queue reset and subscription flag with overflow.
func (b *Broker) resumeEngineNotifications() {
	if r := b.engineNotifications; r != nil {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.dropped = false
	}
	b.engineMu.Lock()
	b.engineSubscribed = true
	b.engineMu.Unlock()
}
