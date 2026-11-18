// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import "sync"

// SnapshotHub fans the latest published value out to snapshot-stream
// subscribers. Each reader holds at most one pending value: Publish replaces
// a value the reader has not taken yet, so a slow reader skips to the newest
// and never sees an older value after a newer one. The zero value is ready.
//
// Only a value that was actually published is ever handed out: a reader that
// subscribes before the first Publish receives nothing until then, because an
// unpublished zero value can encode as null, which subscribers read as a
// keepalive. Change detection belongs to the publisher.
type SnapshotHub[T any] struct {
	Limit int // maximum concurrent readers; 0 (or less) means 64
	// Clone is optional; when set, each reader receives its own copy. It runs
	// while the hub's lock is held, so it must not call back into the hub.
	Clone func(T) T
	// RetainWhileRead keeps the latest value only while at least one reader
	// is subscribed; Publish with no readers does not retain. The last cancel
	// drops the value and starts a new epoch, so a
	// reader joining after idle waits for a fresh publish. PublishAt rejects
	// values collected before that idle period.
	RetainWhileRead bool

	mu      sync.Mutex
	epoch   uint64
	latest  T
	has     bool
	readers map[chan T]struct{}
}

// copy returns value, cloned when Clone is set.
func (h *SnapshotHub[T]) copy(value T) T {
	if h.Clone == nil {
		return value
	}
	return h.Clone(value)
}

// Publish records value as the latest and offers it to every reader.
func (h *SnapshotHub[T]) Publish(value T) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.publishLocked(value)
}

// Epoch returns the current reader epoch. Capture it before collecting a
// value that will be published with PublishAt.
func (h *SnapshotHub[T]) Epoch() uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.epoch
}

// PublishAt publishes value only if no idle period started since epoch was
// captured. It reports whether it published.
func (h *SnapshotHub[T]) PublishAt(epoch uint64, value T) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.RetainWhileRead && h.epoch != epoch {
		return false
	}
	h.publishLocked(value)
	return true
}

// publishLocked retains and delivers value according to the reader rule.
func (h *SnapshotHub[T]) publishLocked(value T) {
	if h.RetainWhileRead && len(h.readers) == 0 {
		return
	}
	h.latest = h.copy(value)
	h.has = true
	for ch := range h.readers {
		select {
		case <-ch:
		default:
		}
		// Cannot block: the buffer is 1, it was just drained, and only
		// Publish and Subscribe send, both under mu.
		ch <- h.copy(value)
	}
}

// Latest returns the last published value and whether one exists.
func (h *SnapshotHub[T]) Latest() (T, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.has {
		var zero T
		return zero, false
	}
	return h.copy(h.latest), true
}

// Subscribe adds a reader. Its channel (buffer 1) holds the latest value at
// once when one has been published. It returns false, and a no-op cancel,
// when Limit readers already exist. Calling cancel removes the reader.
func (h *SnapshotHub[T]) Subscribe() (<-chan T, func(), bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	limit := h.Limit
	if limit <= 0 {
		limit = 64
	}
	if len(h.readers) >= limit {
		return nil, func() {}, false
	}
	if h.readers == nil {
		h.readers = make(map[chan T]struct{})
	}
	ch := make(chan T, 1)
	h.readers[ch] = struct{}{}
	if h.has {
		ch <- h.copy(h.latest)
	}
	return ch, func() {
		h.mu.Lock()
		if _, present := h.readers[ch]; present {
			delete(h.readers, ch)
			if h.RetainWhileRead && len(h.readers) == 0 {
				var zero T
				h.latest, h.has = zero, false
				h.epoch++
			}
		}
		h.mu.Unlock()
	}, true
}
