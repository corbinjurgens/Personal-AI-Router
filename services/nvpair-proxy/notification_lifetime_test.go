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
)

// The writer blocks synchronously until its owner closes it or the test
// releases a healthy reader. A context alone cannot interrupt Write.
type notificationWriter struct {
	mu        sync.Mutex
	frames    [][]byte
	attempts  int
	blockFrom int
	entered   chan struct{}
	returned  chan struct{}
	closed    chan struct{}
	resume    chan struct{}
	once      sync.Once
}

func newNotificationWriter(blockFrom int) *notificationWriter {
	return &notificationWriter{blockFrom: blockFrom, entered: make(chan struct{}, 2), returned: make(chan struct{}, 2), closed: make(chan struct{}), resume: make(chan struct{})}
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
		w.entered <- struct{}{}
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
	return len(raw), nil
}

func (w *notificationWriter) Close() error {
	w.once.Do(func() { close(w.closed) })
	return nil
}

func ownedNotificationCodec(t *testing.T, w *notificationWriter, budget time.Duration) (context.Context, *Codec) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	fail, join := newTerminalSession(ctx, cancel, w)
	t.Cleanup(join)
	return ctx, newOwnedCodec(ctx, w, fail, budget)
}

func TestNotificationLifetimeDefaultBudget(t *testing.T) {
	if notificationWriteBudget != 60*time.Second {
		t.Fatalf("production notification budget = %v", notificationWriteBudget)
	}
	w := newNotificationWriter(1)
	root, codec := ownedNotificationCodec(t, w, 100*time.Millisecond)
	done := make(chan error, 1)
	go func() { done <- codec.Notify("proxy/request", nil) }()
	lifetimeReceive(t, w.entered)
	if err := lifetimeReceive(t, done); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("default notification error = %v", err)
	}
	lifetimeReceive(t, w.returned)
	if root.Err() == nil {
		t.Fatal("default notification did not terminate the parent session")
	}
	if err := codec.Notify("workload:started", nil); err == nil {
		t.Fatal("retired session accepted another notification")
	}
}

func TestNotificationLifetimeCallerDeadline(t *testing.T) {
	w := newNotificationWriter(1)
	root, codec := ownedNotificationCodec(t, w, notificationWriteBudget)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- codec.NotifyContext(ctx, "performance:sample", nil) }()
	lifetimeReceive(t, w.entered)
	if err := lifetimeReceive(t, done); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("active caller deadline = %v", err)
	}
	lifetimeReceive(t, w.returned)
	if root.Err() == nil {
		t.Fatal("active write interruption did not terminate the parent session")
	}
}

func TestNotificationLifetimeQueuedCallerPreservesActiveWrite(t *testing.T) {
	w := newNotificationWriter(1)
	root, codec := ownedNotificationCodec(t, w, notificationWriteBudget)
	active := make(chan error, 1)
	go func() { active <- codec.Notify("proxy/request", nil) }()
	lifetimeReceive(t, w.entered)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if err := codec.NotifyContext(ctx, "performance:sample", nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued caller error = %v", err)
	}
	if root.Err() != nil {
		t.Fatal("queued cancellation retired an unrelated active write")
	}
	close(w.resume)
	if err := lifetimeReceive(t, active); err != nil {
		t.Fatalf("healthy active write: %v", err)
	}
	lifetimeReceive(t, w.returned)
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.attempts != 1 || len(w.frames) != 1 {
		t.Fatalf("queued notification leaked a frame: attempts=%d frames=%d", w.attempts, len(w.frames))
	}
}

func TestNotificationLifetimeQueuedDefaultFailureClosesSession(t *testing.T) {
	w := newNotificationWriter(1)
	root, codec := ownedNotificationCodec(t, w, 40*time.Millisecond)
	id := json.RawMessage("1")
	active := make(chan error, 1)
	go func() { active <- codec.Respond(&id, struct{}{}) }()
	lifetimeReceive(t, w.entered)
	if err := codec.Notify("workload:started", nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued default error = %v", err)
	}
	if err := lifetimeReceive(t, active); err == nil {
		t.Fatal("default notification failure left a response write blocked")
	}
	lifetimeReceive(t, w.returned)
	if root.Err() == nil {
		t.Fatal("queued default failure did not terminate the parent session")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.attempts != 1 {
		t.Fatalf("queued notification touched writer: attempts=%d", w.attempts)
	}
}
