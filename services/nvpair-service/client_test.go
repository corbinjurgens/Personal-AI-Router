// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"sync"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
)

type blockedClientConn struct {
	started   chan struct{}
	release   chan struct{}
	closed    chan struct{}
	startOnce sync.Once
	closeOnce sync.Once
}

func (c *blockedClientConn) Read([]byte) (int, error) { return 0, io.EOF }
func (c *blockedClientConn) Write(p []byte) (int, error) {
	c.startOnce.Do(func() { close(c.started) })
	select {
	case <-c.release:
		return len(p), nil
	case <-c.closed:
		return 0, io.ErrClosedPipe
	}
}
func (c *blockedClientConn) Close() error {
	c.closeOnce.Do(func() { close(c.closed) })
	return nil
}

func waitClientSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for client")
	}
}

func newBlockedTestClient(t *testing.T) (*client, *blockedClientConn) {
	t.Helper()
	conn := &blockedClientConn{started: make(chan struct{}), release: make(chan struct{}), closed: make(chan struct{})}
	c := newClient(1, conn)
	t.Cleanup(func() { c.closeNow(); waitClientSignal(t, c.written) })
	return c, conn
}

func TestClientByteBudgetDisconnects(t *testing.T) {
	c, conn := newBlockedTestClient(t)
	frame := make([]byte, 4*1024*1024)
	c.send(frame)
	waitClientSignal(t, conn.started)
	for i := 1; i < clientQueueBytes/len(frame); i++ {
		c.send(frame)
		if c.isClosing() {
			t.Fatalf("disconnected after %d frames within budget", i+1)
		}
	}
	c.mu.Lock()
	queued := c.queued
	c.mu.Unlock()
	if queued != clientQueueBytes {
		t.Fatalf("queued bytes = %d, want %d", queued, clientQueueBytes)
	}
	c.send(frame)
	if !c.isClosing() {
		t.Fatal("client accepted frame beyond byte budget")
	}
	waitClientSignal(t, conn.closed)
}

func TestClientAcceptsLargeFrameIntoEmptyQueue(t *testing.T) {
	c, conn := newBlockedTestClient(t)
	c.send(make([]byte, jsonrpc.WorkerFrameBytes))
	waitClientSignal(t, conn.started)
	if c.isClosing() {
		t.Fatal("client rejected valid large frame")
	}
	close(conn.release)
	c.closeGraceful()
	waitClientSignal(t, c.written)
}

func TestClientRejectsFrameLargerThanBudget(t *testing.T) {
	c, conn := newBlockedTestClient(t)
	c.send(make([]byte, clientQueueBytes+1))
	if !c.isClosing() {
		t.Fatal("client accepted oversized frame")
	}
	waitClientSignal(t, conn.closed)
}

func TestClientQueuedBytesReturnToZero(t *testing.T) {
	c, conn := newBlockedTestClient(t)
	frame := make([]byte, 1024)
	c.send(frame)
	waitClientSignal(t, conn.started)
	c.send(frame)
	close(conn.release)
	c.closeGraceful()
	waitClientSignal(t, c.written)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.queued != 0 {
		t.Fatalf("queued bytes = %d after drain, want 0", c.queued)
	}
}
