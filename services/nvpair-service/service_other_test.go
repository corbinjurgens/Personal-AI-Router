// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testEndpoint(dir string) string { return filepath.Join(dir, "s.sock") }

func TestStaleSocketIsRemoved(t *testing.T) {
	dir := shortTempDir(t)
	endpoint := filepath.Join(dir, "s.sock")
	// A socket file nobody listens on, as a crashed service leaves behind.
	l, err := net.Listen("unix", endpoint)
	if err != nil {
		t.Fatal(err)
	}
	l.(*net.UnixListener).SetUnlinkOnClose(false)
	_ = l.Close()
	if _, err := os.Stat(endpoint); err != nil {
		t.Fatalf("stale socket not created: %v", err)
	}

	brokers := make(chan *fakeBroker, 1)
	svc := newService(config{
		endpoint: endpoint,
		spawn: func() (*brokerProc, error) {
			b := newFakeBroker(1, false)
			go b.run()
			brokers <- b
			return b.proc(), nil
		},
		backoff:                defaultBackoff,
		stopGrace:              time.Second,
		shutdownRequestTimeout: time.Second,
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- svc.Run(ctx) }()
	defer func() {
		cancel()
		<-done
	}()
	waitUntil(t, "listening over the stale socket", func() bool { return alreadyAnswering(context.Background(), endpoint) })
	st, err := os.Stat(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	if mode := st.Mode().Perm(); mode != 0o600 {
		t.Fatalf("socket mode = %o, want 600", mode)
	}
}
