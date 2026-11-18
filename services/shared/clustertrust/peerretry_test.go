// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"context"
	"testing"
	"time"
)

func constantRetry() *PeerRetry {
	r := NewMemberRetry()
	r.Jitter = func() float64 { return 1 }
	return r
}

func TestPeerRetryLadders(t *testing.T) {
	t.Run("stream stays capped", func(t *testing.T) {
		r := NewStreamRetry()
		r.Jitter = func() float64 { return 1 }
		now := time.Now()
		for _, seconds := range []int{1, 2, 4, 8, 16, 30, 30} {
			if got := r.Next(0, now); got != time.Duration(seconds)*time.Second {
				t.Fatalf("delay = %v, want %ds", got, seconds)
			}
		}
		if got := r.Next(0, now.Add(15*time.Minute)); got != 30*time.Second {
			t.Fatalf("long delay = %v", got)
		}
	})
	t.Run("member grows from capped delay", func(t *testing.T) {
		r := constantRetry()
		now := time.Now()
		for range 6 {
			r.Next(0, now)
		}
		for _, seconds := range []int{60, 120, 240, 300, 300} {
			if got := r.Next(0, now.Add(10*time.Minute)); got != time.Duration(seconds)*time.Second {
				t.Fatalf("delay = %v, want %ds", got, seconds)
			}
		}
	})
	t.Run("only stable resets", func(t *testing.T) {
		r := constantRetry()
		now := time.Now()
		r.Next(0, now)
		if got := r.Next(29*time.Second, now); got != 2*time.Second {
			t.Fatalf("short delay = %v", got)
		}
		if got := r.Next(30*time.Second, now); got != time.Second {
			t.Fatalf("stable delay = %v", got)
		}
	})
}

func TestPeerRetryJitter(t *testing.T) {
	r := NewMemberRetry()
	now := time.Now()
	r.Next(0, now)
	for range 12 {
		r.Next(0, now.Add(15*time.Minute))
	}
	for range 1000 {
		if got := r.Next(0, now.Add(15*time.Minute)); got < 240*time.Second || got > 360*time.Second {
			t.Fatalf("jittered long cap = %v", got)
		}
	}
}

func TestPeerBackoff(t *testing.T) {
	b := NewPeerBackoff()
	b.NewRetry = constantRetry
	addrs := []string{"b", "a"}
	now := time.Now()
	b.Ended("peer", addrs, 0, now)
	b.Ended("peer", addrs, 0, now)
	b.Expedite("peer")
	if !b.Due("peer", []string{"a", "b"}, time.Now()) {
		t.Fatal("expedited peer not due")
	}
	b.Ended("peer", addrs, 0, now)
	if b.Due("peer", addrs, now.Add(3*time.Second)) || !b.Due("peer", addrs, now.Add(4*time.Second)) {
		t.Fatal("expedite erased ladder")
	}
	b.Ended("peer", addrs, 30*time.Second, now)
	if b.Due("peer", addrs, now) || !b.Due("peer", addrs, now.Add(time.Second)) {
		t.Fatal("stable outcome must wait initial delay")
	}
	b.Reset("peer")
	b.Ended("peer", addrs, 0, now)
	if !b.Due("peer", addrs, now.Add(time.Second)) {
		t.Fatal("reset did not restart ladder")
	}
	if !b.Due("peer", []string{"new"}, now) {
		t.Fatal("address change not due")
	}
	b.Ended("peer", addrs, 0, now)
	b.Ended("kept", addrs, 0, now)
	b.Retain(map[string]bool{"kept": true})
	if !b.Due("peer", addrs, now) || b.Due("kept", addrs, now) {
		t.Fatal("retain did not preserve only kept peer")
	}
	b.Expedite("absent")
	if !b.Due("absent", addrs, now) {
		t.Fatal("absent peer not due")
	}
}

func TestKeepPeerStream(t *testing.T) {
	t.Run("waits computed delays and stops", func(t *testing.T) {
		r := &PeerRetry{Initial: 10 * time.Millisecond, Cap: time.Second, StableAfter: time.Second, Jitter: func() float64 { return 1 }}
		var attempts []time.Time
		KeepPeerStream(context.Background(), PeerStreamLoop{Retry: r}, func(context.Context) (time.Duration, bool) {
			attempts = append(attempts, time.Now())
			return 0, len(attempts) < 3
		})
		for i, delay := range []time.Duration{10 * time.Millisecond, 20 * time.Millisecond} {
			if elapsed := attempts[i+1].Sub(attempts[i]); elapsed < delay {
				t.Fatalf("wait = %v, want >= %v", elapsed, delay)
			}
		}
	})
	t.Run("cancel interrupts wait", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		r := &PeerRetry{Initial: time.Hour, Cap: time.Hour, StableAfter: time.Second}
		calls := 0
		start := time.Now()
		KeepPeerStream(ctx, PeerStreamLoop{Retry: r}, func(context.Context) (time.Duration, bool) {
			calls++
			time.AfterFunc(10*time.Millisecond, cancel)
			return 0, true
		})
		if calls != 1 {
			t.Fatalf("calls = %d", calls)
		}
		if time.Since(start) > time.Second {
			t.Fatal("cancel did not interrupt the retry wait")
		}
	})
}
