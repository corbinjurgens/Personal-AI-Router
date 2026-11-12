// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"slices"
	"testing"
	"time"
)

func testPeerBackoff(factor float64) *peerBackoff {
	b := newPeerBackoff()
	b.jitter = func() float64 { return factor }
	return b
}

func TestPeerBackoff(t *testing.T) {
	start := time.Unix(1000, 0)
	addrs := []string{"10.0.0.2:8080", "10.0.0.1:8080"}

	t.Run("a fresh peer is due", func(t *testing.T) {
		b := testPeerBackoff(1)
		if !b.due("p1", addrs, start) {
			t.Fatal("fresh peer is not due")
		}
	})

	t.Run("consecutive failures double the delay up to the cap", func(t *testing.T) {
		b := testPeerBackoff(1)
		want := []time.Duration{1, 2, 4, 8, 16, 30, 30}
		now := start
		for i, seconds := range want {
			b.failed("p1", addrs, now)
			delay := seconds * time.Second
			if b.due("p1", addrs, now.Add(delay-time.Nanosecond)) {
				t.Fatalf("failure %d: due before %v", i+1, delay)
			}
			if !b.due("p1", addrs, now.Add(delay)) {
				t.Fatalf("failure %d: not due at %v", i+1, delay)
			}
			now = now.Add(delay)
		}
	})

	t.Run("many failures stay at the cap", func(t *testing.T) {
		b := testPeerBackoff(1)
		for i := 0; i < 200; i++ {
			b.failed("p1", addrs, start)
		}
		if b.due("p1", addrs, start.Add(30*time.Second-time.Nanosecond)) || !b.due("p1", addrs, start.Add(30*time.Second)) {
			t.Fatal("delay after many failures is not the 30 s cap")
		}
	})

	t.Run("reset makes the peer due", func(t *testing.T) {
		b := testPeerBackoff(1)
		b.failed("p1", addrs, start)
		b.reset("p1")
		if !b.due("p1", addrs, start) {
			t.Fatal("peer is not due after reset")
		}
	})

	t.Run("a changed address set makes the peer due inside the delay", func(t *testing.T) {
		b := testPeerBackoff(1)
		b.failed("p1", addrs, start)
		if !b.due("p1", []string{"10.0.0.3:8080"}, start) {
			t.Fatal("moved peer is not due")
		}
		// The changed set cleared the entry: the next failure starts at 1 s.
		b.failed("p1", []string{"10.0.0.3:8080"}, start)
		if !b.due("p1", []string{"10.0.0.3:8080"}, start.Add(time.Second)) {
			t.Fatal("failure after an address change did not restart at 1 s")
		}
	})

	t.Run("address order does not count as a change and is not mutated", func(t *testing.T) {
		b := testPeerBackoff(1)
		given := slices.Clone(addrs)
		b.failed("p1", given, start)
		if !slices.Equal(given, addrs) {
			t.Fatalf("failed mutated the caller's slice: %v", given)
		}
		reordered := []string{addrs[1], addrs[0]}
		if b.due("p1", reordered, start) {
			t.Fatal("reordered address set reset the backoff")
		}
		if !slices.Equal(reordered, []string{addrs[1], addrs[0]}) {
			t.Fatalf("due mutated the caller's slice: %v", reordered)
		}
	})

	t.Run("jitter scales the delay", func(t *testing.T) {
		for _, factor := range []float64{0.8, 1.2} {
			b := testPeerBackoff(factor)
			b.failed("p1", addrs, start)
			b.failed("p1", addrs, start) // second delay: 2 s
			delay := time.Duration(float64(2*time.Second) * factor)
			if b.due("p1", addrs, start.Add(delay-time.Nanosecond)) || !b.due("p1", addrs, start.Add(delay)) {
				t.Fatalf("factor %v: delay is not %v", factor, delay)
			}
		}
	})

	t.Run("retain drops peers outside the keep set", func(t *testing.T) {
		b := testPeerBackoff(1)
		b.failed("p1", addrs, start)
		b.failed("p2", addrs, start)
		b.retain(map[string]bool{"p2": true})
		if !b.due("p1", addrs, start) {
			t.Fatal("dropped peer is still backed off")
		}
		if b.due("p2", addrs, start) {
			t.Fatal("kept peer lost its backoff")
		}
	})
}
