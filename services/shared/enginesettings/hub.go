// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package enginesettings

import "sync"

// Hub sends full snapshots, so coalescing a slow reader cannot lose a patch.
// Subscribe installs the reader and its baseline under the same lock.
type Hub struct {
	mu      sync.Mutex
	latest  []Snapshot
	readers map[chan []Snapshot]bool
}

func (h *Hub) Publish(value []Snapshot) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.latest = cloneSnapshots(value)
	for ch := range h.readers {
		select {
		case <-ch:
		default:
		}
		ch <- cloneSnapshots(value)
	}
}

// Each reader owns its snapshots, including nested serving records.
func cloneSnapshots(value []Snapshot) []Snapshot {
	out := make([]Snapshot, len(value))
	for i, s := range value {
		out[i] = s.Clone()
	}
	return out
}

func (h *Hub) Subscribe() (<-chan []Snapshot, func(), bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.readers) >= 64 {
		return nil, func() {}, false
	}
	if h.readers == nil {
		h.readers = make(map[chan []Snapshot]bool)
	}
	ch := make(chan []Snapshot, 1)
	h.readers[ch] = true
	ch <- cloneSnapshots(h.latest)
	return ch, func() { h.mu.Lock(); delete(h.readers, ch); h.mu.Unlock() }, true
}
