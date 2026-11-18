// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
)

type wakeChanges struct {
	mu       sync.Mutex
	ch       chan struct{}
	captured chan struct{}
}

func (s *wakeChanges) changed() <-chan struct{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ch == nil {
		s.ch = make(chan struct{})
	}
	select {
	case s.captured <- struct{}{}:
	default:
	}
	return s.ch
}

func (s *wakeChanges) notify() {
	s.mu.Lock()
	defer s.mu.Unlock()
	close(s.ch)
	s.ch = make(chan struct{})
}

func TestNetworkChangeResetsBackoffAndCoalescesProbe(t *testing.T) {
	m := newReverseTestManager(t)
	changes := &wakeChanges{captured: make(chan struct{}, 8)}
	oldStart, oldProbe := startNetworkMonitor, probeReverseSessions
	startNetworkMonitor = func(context.Context) (func() <-chan struct{}, error) { return changes.changed, nil }
	entered, release := make(chan struct{}, 8), make(chan struct{})
	probeReverseSessions = func(*clustertrust.ReverseManager) int { entered <- struct{}{}; <-release; return 0 }
	defer func() { startNetworkMonitor, probeReverseSessions = oldStart, oldProbe }()
	ctx, cancel := context.WithCancel(context.Background())
	m.startNetworkWatch(ctx)
	now := time.Now()
	m.reverseBackoff.Ended("peer", nil, 0, now)
	m.peerStateBackoff.Ended("peer", nil, 0, now)
	done := make(chan struct{})
	go func() { defer close(done); m.networkChangeLoop(ctx) }()
	defer func() { cancel(); close(release); <-done }()
	<-changes.captured
	changes.notify()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("no probe after notification")
	}
	if !m.reverseBackoff.Due("peer", nil, now) || !m.peerStateBackoff.Due("peer", nil, now) {
		t.Fatal("member remained in backoff")
	}
	select {
	case <-m.peerReachableWake:
	default:
		t.Fatal("peer-state loop not signaled")
	}
	changes.notify()
	changes.notify()
	release <- struct{}{}
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("no follow-up probe")
	}
	release <- struct{}{}
	select {
	case <-entered:
		t.Fatal("notifications did not coalesce")
	case <-time.After(100 * time.Millisecond):
	}
}
