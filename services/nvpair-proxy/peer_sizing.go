// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/nodepolicy"
)

// peerSizingCache holds the broker's latest private peer sizing snapshot. It is
// ephemeral sizing configuration only: it grants no admission, capacity or
// speed authority, and it is separate from priority, reservations and the
// routing cost cache. Its zero value is empty and usable.
type peerSizingCache struct {
	mu       sync.RWMutex
	revision uint64
	received time.Time
	peers    map[string]nodepolicy.PeerSizing
	// generation mirrors revision for lock-free comparison by the chooser, which
	// must not take this mutex beneath the priority lock.
	generation atomic.Uint64
	// now is the monotonic receipt clock; nil means time.Now.
	now func() time.Time
}

// peerSizingQuery names one peer profile with the authority the caller read
// outside chooser locks: the live mesh admission and the discovery
// advertisement it is about to route on.
type peerSizingQuery struct {
	Host                 string
	AdmissionIdentity    string
	AdmissionIncarnation string
	PolicyRevision       string
	Key                  string
	Engine               string
	Model                string
}

// peerSizingToken is what a chooser keeps instead of the recipe: the snapshot
// generation it read and the instant the observation expires.
type peerSizingToken struct {
	Generation uint64
	Expires    time.Time
}

func (c *peerSizingCache) clock() time.Time {
	if c.now != nil {
		return c.now()
	}
	return time.Now()
}

// set replaces the whole snapshot. A lower revision is refused and an equal
// revision is a no-op that does not renew the receipt time.
func (c *peerSizingCache) set(s nodepolicy.PeerSizingSnapshot) error {
	if err := s.Validate(); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if s.Revision < c.revision {
		return errors.New("peer sizing revision is older than the current revision")
	}
	if s.Revision == c.revision {
		return nil
	}
	peers := make(map[string]nodepolicy.PeerSizing, len(s.Peers))
	for _, peer := range s.Clone().Peers {
		peers[peer.Host] = peer
	}
	c.peers, c.revision, c.received = peers, s.Revision, c.clock()
	c.generation.Store(s.Revision)
	return nil
}

func (c *peerSizingCache) count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.peers)
}

// recipe returns the destination recipe for q only while the observation is
// unexpired and its admission, policy revision and profile identity still match
// what the caller read. Any mismatch is unknown sizing, never a stale recipe.
func (c *peerSizingCache) recipe(q peerSizingQuery) (nodepolicy.PeerSizingProfile, peerSizingToken, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	peer, ok := c.peers[q.Host]
	if !ok || peer.AdmissionIdentity != q.AdmissionIdentity || peer.AdmissionIncarnation != q.AdmissionIncarnation || peer.PolicyRevision != q.PolicyRevision {
		return nodepolicy.PeerSizingProfile{}, peerSizingToken{}, false
	}
	expires := c.received.Add(time.Duration(nodepolicy.PeerSizingFreshnessMs-peer.AgeMs) * time.Millisecond)
	if !c.clock().Before(expires) {
		return nodepolicy.PeerSizingProfile{}, peerSizingToken{}, false
	}
	for _, profile := range peer.Profiles {
		if profile.Key != q.Key {
			continue
		}
		if profile.Engine != q.Engine || profile.Model != q.Model {
			return nodepolicy.PeerSizingProfile{}, peerSizingToken{}, false
		}
		profile.RequestOptions = append([]byte(nil), profile.RequestOptions...)
		return profile, peerSizingToken{Generation: c.revision, Expires: expires}, true
	}
	return nodepolicy.PeerSizingProfile{}, peerSizingToken{}, false
}
