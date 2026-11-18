// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"context"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"time"
)

const (
	PeerRetryInitial     = time.Second
	PeerRetryCap         = 30 * time.Second
	PeerRetryLongAfter   = 10 * time.Minute
	PeerRetryLongCap     = 5 * time.Minute
	PeerRetryStableAfter = 30 * time.Second
)

// PeerRetry schedules one peer connection. Initial, Cap and StableAfter must
// be positive. LongAfter == 0 disables the long cap. It is not concurrency safe.
type PeerRetry struct {
	Initial, Cap, LongAfter, LongCap, StableAfter time.Duration
	Jitter                                        func() float64 // nil means uniform [0.8, 1.2]
	base                                          time.Duration
	failingSince                                  time.Time
}

func NewStreamRetry() *PeerRetry {
	return &PeerRetry{Initial: PeerRetryInitial, Cap: PeerRetryCap, StableAfter: PeerRetryStableAfter}
}

func NewMemberRetry() *PeerRetry {
	r := NewStreamRetry()
	r.LongAfter, r.LongCap = PeerRetryLongAfter, PeerRetryLongCap
	return r
}

// Next records an attempt's healthy duration and returns the next delay.
func (r *PeerRetry) Next(up time.Duration, now time.Time) time.Duration {
	if up >= r.StableAfter {
		r.Reset()
	}
	if r.base == 0 {
		r.failingSince, r.base = now, r.Initial
	} else {
		cap := r.Cap
		if r.LongAfter > 0 && now.Sub(r.failingSince) >= r.LongAfter {
			cap = r.LongCap
		}
		r.base = min(2*r.base, cap)
	}
	factor := 0.8 + 0.4*rand.Float64()
	if r.Jitter != nil {
		factor = r.Jitter()
	}
	return time.Duration(float64(r.base) * factor)
}

func (r *PeerRetry) Reset() { r.base, r.failingSince = 0, time.Time{} }

// PeerBackoff holds member schedules for tick-driven supervisors. It is safe
// for concurrent use. NewRetry may be set before use to inject test timing.
type PeerBackoff struct {
	mu       sync.Mutex
	entries  map[string]peerBackoffEntry
	NewRetry func() *PeerRetry
}

type peerBackoffEntry struct {
	retry *PeerRetry
	due   time.Time
	addrs string
}

func NewPeerBackoff() *PeerBackoff {
	return &PeerBackoff{entries: make(map[string]peerBackoffEntry), NewRetry: NewMemberRetry}
}

func peerBackoffAddrs(addrs []string) string {
	sorted := slices.Clone(addrs)
	slices.Sort(sorted)
	return strings.Join(sorted, ",")
}

// Due clears an entry when its address set changes, retrying moved peers now.
func (b *PeerBackoff) Due(peer string, addrs []string, now time.Time) bool {
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

func (b *PeerBackoff) Ended(peer string, addrs []string, up time.Duration, now time.Time) {
	b.mu.Lock()
	defer b.mu.Unlock()
	entry := b.entries[peer]
	if entry.retry == nil {
		entry.retry = b.NewRetry()
	}
	entry.due = now.Add(entry.retry.Next(up, now))
	entry.addrs = peerBackoffAddrs(addrs)
	b.entries[peer] = entry
}

func (b *PeerBackoff) Reset(peer string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.entries, peer)
}

// ResetAll forgets every peer's retry history, making all peers due.
func (b *PeerBackoff) ResetAll() {
	b.mu.Lock()
	defer b.mu.Unlock()
	clear(b.entries)
}

// Expedite makes a known peer due now without erasing its failure history.
func (b *PeerBackoff) Expedite(peer string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if entry, ok := b.entries[peer]; ok {
		entry.due = time.Now()
		b.entries[peer] = entry
	}
}

func (b *PeerBackoff) Retain(keep map[string]bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for peer := range b.entries {
		if !keep[peer] {
			delete(b.entries, peer)
		}
	}
}

type PeerStreamLoop struct {
	Retry   *PeerRetry
	Changed func() <-chan struct{}
}

// KeepPeerStream retries until the context ends or attempt reports keep=false.
func KeepPeerStream(ctx context.Context, loop PeerStreamLoop, attempt func(context.Context) (up time.Duration, keep bool)) {
	for ctx.Err() == nil {
		var changed <-chan struct{}
		if loop.Changed != nil {
			changed = loop.Changed()
		}
		up, keep := attempt(ctx)
		if !keep || ctx.Err() != nil {
			return
		}
		select {
		case <-changed:
			loop.Retry.Reset()
			continue
		default:
		}
		timer := time.NewTimer(loop.Retry.Next(up, time.Now()))
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
			select {
			case <-changed:
				loop.Retry.Reset()
			default:
			}
		case <-changed:
			timer.Stop()
			loop.Retry.Reset()
		}
	}
}
