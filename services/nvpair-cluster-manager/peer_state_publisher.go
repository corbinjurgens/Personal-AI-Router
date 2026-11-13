// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"slices"
	"sync"
	"time"

	"nvpair-shared/noderec"
)

// peerStateRebuildDelay coalesces a burst of change signals into one rebuild.
const peerStateRebuildDelay = 250 * time.Millisecond

// peerStateRebuildEvery also rebuilds without a signal, so sources that raise
// no event (overlay addresses) reach peers within one interval.
const peerStateRebuildEvery = 10 * time.Second

// peerStateMaxReaders bounds concurrent peer-state streams served by one node.
const peerStateMaxReaders = 64

// peerStatePublisher holds this node's latest peer-state frame and forwards
// it to every serving stream, following enginesettings.Hub: each reader is a
// 1-buffered channel pre-filled on subscribe and replaced on publish, so a slow
// reader skips stale frames but never misses the latest. Published frames are
// never modified afterwards, so readers share them.
type peerStatePublisher struct {
	mu          sync.Mutex
	latest      noderec.PeerStateFrame
	isAvailable bool
	readers     map[chan noderec.PeerStateFrame]bool
}

// subscribe installs a reader, pre-filled with the latest frame when one is
// available. It returns false when the reader cap is reached.
func (p *peerStatePublisher) subscribe() (<-chan noderec.PeerStateFrame, func(), bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.readers) >= peerStateMaxReaders {
		return nil, func() {}, false
	}
	if p.readers == nil {
		p.readers = make(map[chan noderec.PeerStateFrame]bool)
	}
	ch := make(chan noderec.PeerStateFrame, 1)
	p.readers[ch] = true
	if p.isAvailable {
		ch <- p.latest
	}
	return ch, func() { p.mu.Lock(); delete(p.readers, ch); p.mu.Unlock() }, true
}

// publish makes frame the latest value and replaces any unread frame in every
// reader with it.
func (p *peerStatePublisher) publish(frame noderec.PeerStateFrame) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.publishLocked(frame)
}

func (p *peerStatePublisher) publishLocked(frame noderec.PeerStateFrame) {
	p.latest, p.isAvailable = frame, true
	for ch := range p.readers {
		select {
		case <-ch:
		default:
		}
		ch <- frame
	}
}

// offer publishes frame when the publisher is unavailable or frame differs
// from the latest one, and reports whether it published.
func (p *peerStatePublisher) offer(frame noderec.PeerStateFrame) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.isAvailable && peerStateFramesEqual(p.latest, frame) {
		return false
	}
	p.publishLocked(frame)
	return true
}

func (p *peerStatePublisher) markUnavailable() {
	p.mu.Lock()
	p.isAvailable = false
	p.mu.Unlock()
}

func (p *peerStatePublisher) available() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.isAvailable
}

func peerStateFramesEqual(a, b noderec.PeerStateFrame) bool {
	return a.RosterHash == b.RosterHash && a.Descriptor.HostUUID == b.Descriptor.HostUUID &&
		slices.Equal(a.Descriptor.Addresses, b.Descriptor.Addresses) &&
		slices.Equal(a.Descriptor.Services, b.Descriptor.Services)
}

// signalPeerState asks peerStatePublishLoop for a rebuild without blocking.
func (m *Manager) signalPeerState() {
	select {
	case m.peerStateSignal <- struct{}{}:
	default:
	}
}

// rebuildPeerState builds this node's frame and offers it to the publisher. It
// must run outside every manager lock: the descriptor and the roster read
// several owners. A descriptor that cannot be built marks the publisher
// unavailable, which refuses new streams and ends open ones.
func (m *Manager) rebuildPeerState() {
	descriptor, err := m.selfServicesDescriptor()
	if err != nil {
		m.peerState.markUnavailable()
		return
	}
	m.peerState.offer(noderec.PeerStateFrame{Descriptor: descriptor, RosterHash: m.rosterHash()})
}

// peerStatePublishLoop rebuilds the frame once at start, peerStateRebuildDelay
// after a change signal (draining further signals meanwhile) and every
// peerStateRebuildEvery.
func (m *Manager) peerStatePublishLoop(ctx context.Context) {
	m.rebuildPeerState()
	ticker := time.NewTicker(peerStateRebuildEvery)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		case <-m.peerStateSignal:
			delay := time.NewTimer(peerStateRebuildDelay)
		drain:
			for {
				select {
				case <-ctx.Done():
					delay.Stop()
					return
				case <-m.peerStateSignal:
				case <-delay.C:
					break drain
				}
			}
		}
		m.rebuildPeerState()
	}
}
