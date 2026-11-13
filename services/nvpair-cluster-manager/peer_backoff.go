// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"time"
)

// peerBackoff schedules retries per peer UUID. It is safe for concurrent use.
type peerBackoff struct {
	mu      sync.Mutex
	initial time.Duration
	max     time.Duration
	jitter  func() float64 // returns a factor in [0.8, 1.2]; tests inject a constant
	entries map[string]peerBackoffEntry
}

type peerBackoffEntry struct {
	failures int
	due      time.Time
	addrs    string // sorted, comma-joined address set at the last failure
}

// newPeerBackoff retries after 1 s, doubling per consecutive failure up to
// 30 s, each delay scaled by a uniform factor in [0.8, 1.2].
func newPeerBackoff() *peerBackoff {
	return &peerBackoff{
		initial: time.Second,
		max:     30 * time.Second,
		jitter:  func() float64 { return 0.8 + 0.4*rand.Float64() },
		entries: make(map[string]peerBackoffEntry),
	}
}

func peerBackoffAddrs(addrs []string) string {
	sorted := slices.Clone(addrs)
	slices.Sort(sorted)
	return strings.Join(sorted, ",")
}

// due reports whether peer may be attempted at now. A changed address set
// clears the entry first, so a moved peer is retried at once.
func (b *peerBackoff) due(peer string, addrs []string, now time.Time) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	entry, ok := b.entries[peer]
	if !ok {
		return true
	}
	if entry.addrs != peerBackoffAddrs(addrs) {
		delete(b.entries, peer)
		return true
	}
	return !now.Before(entry.due)
}

// failed records a failed attempt at now:
// delay = min(initial << failures, max) * jitter(), then failures++.
func (b *peerBackoff) failed(peer string, addrs []string, now time.Time) {
	b.mu.Lock()
	defer b.mu.Unlock()
	entry := b.entries[peer]
	delay := b.initial
	for i := 0; i < entry.failures && delay < b.max; i++ {
		delay <<= 1
	}
	delay = min(delay, b.max)
	b.entries[peer] = peerBackoffEntry{
		failures: entry.failures + 1,
		due:      now.Add(time.Duration(float64(delay) * b.jitter())),
		addrs:    peerBackoffAddrs(addrs),
	}
}

// reset forgets peer (success, session present, or member gone).
func (b *peerBackoff) reset(peer string) {
	b.mu.Lock()
	delete(b.entries, peer)
	b.mu.Unlock()
}

// retain drops every entry whose peer is not in keep.
func (b *peerBackoff) retain(keep map[string]bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for peer := range b.entries {
		if !keep[peer] {
			delete(b.entries, peer)
		}
	}
}
