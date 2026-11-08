// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func proxyPeerSizingFixture(revision uint64, ageMs int64) nodepolicy.PeerSizingSnapshot {
	return nodepolicy.PeerSizingSnapshot{Revision: revision, Peers: []nodepolicy.PeerSizing{{
		Host: "destination", AdmissionIdentity: "incarnation-d", AdmissionIncarnation: "3", PolicyRevision: "9", ProxyRunID: "run-9", AgeMs: ageMs,
		Profiles: []nodepolicy.PeerSizingProfile{
			{Key: "medium", Engine: "llamacpp", Model: "gemma", RequestOptions: json.RawMessage(`{"max_tokens":64}`), RequestOptionsMode: nodepolicy.RequestOptionsEnforced},
			{Key: "fast", Engine: "llamacpp", Model: "qwen"},
		},
	}}}
}

func peerSizingQueryFor(key, model string) peerSizingQuery {
	return peerSizingQuery{Host: "destination", AdmissionIdentity: "incarnation-d", AdmissionIncarnation: "3", PolicyRevision: "9", Key: key, Engine: "llamacpp", Model: model}
}

func TestPeerSizingCacheRevisionReplacement(t *testing.T) {
	clock := time.Unix(100, 0)
	c := peerSizingCache{now: func() time.Time { return clock }}
	if err := c.set(proxyPeerSizingFixture(4, 1000)); err != nil {
		t.Fatal(err)
	}
	recipe, token, ok := c.recipe(peerSizingQueryFor("medium", "gemma"))
	if !ok || string(recipe.RequestOptions) != `{"max_tokens":64}` || token.Generation != 4 || c.generation.Load() != 4 {
		t.Fatalf("recipe = %+v, %+v, %v", recipe, token, ok)
	}
	if want := clock.Add(89 * time.Second); !token.Expires.Equal(want) {
		t.Fatalf("expires = %v, want %v", token.Expires, want)
	}

	t.Run("lower revision refused", func(t *testing.T) {
		if err := c.set(nodepolicy.PeerSizingSnapshot{Revision: 3, Peers: []nodepolicy.PeerSizing{}}); err == nil {
			t.Fatal("accepted a lower revision")
		}
		if _, _, ok := c.recipe(peerSizingQueryFor("medium", "gemma")); !ok {
			t.Fatal("lower revision changed the cache")
		}
	})
	t.Run("equal revision does not renew", func(t *testing.T) {
		clock = clock.Add(10 * time.Second)
		if err := c.set(proxyPeerSizingFixture(4, 0)); err != nil {
			t.Fatal(err)
		}
		if _, token, _ := c.recipe(peerSizingQueryFor("medium", "gemma")); !token.Expires.Equal(time.Unix(100, 0).Add(89 * time.Second)) {
			t.Fatalf("equal revision renewed receipt: %v", token.Expires)
		}
	})
	t.Run("invalid frame leaves cache unchanged", func(t *testing.T) {
		bad := proxyPeerSizingFixture(5, 0)
		bad.Peers[0].AgeMs = nodepolicy.PeerSizingFreshnessMs
		if err := c.set(bad); err == nil {
			t.Fatal("accepted an expired observation")
		}
		if c.generation.Load() != 4 {
			t.Fatal("invalid frame changed the generation")
		}
	})
	t.Run("empty replacement withdraws everything", func(t *testing.T) {
		if err := c.set(nodepolicy.PeerSizingSnapshot{Revision: 6, Peers: []nodepolicy.PeerSizing{}}); err != nil {
			t.Fatal(err)
		}
		if _, _, ok := c.recipe(peerSizingQueryFor("medium", "gemma")); ok || c.count() != 0 {
			t.Fatal("empty snapshot did not withdraw")
		}
	})
}

func TestPeerSizingCacheExpiresAndChecksAuthorityAtLookup(t *testing.T) {
	clock := time.Unix(100, 0)
	c := peerSizingCache{now: func() time.Time { return clock }}
	if err := c.set(proxyPeerSizingFixture(1, 85000)); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name  string
		query func(peerSizingQuery) peerSizingQuery
	}{
		{"other admission identity", func(q peerSizingQuery) peerSizingQuery { q.AdmissionIdentity = "incarnation-x"; return q }},
		{"re-paired epoch", func(q peerSizingQuery) peerSizingQuery { q.AdmissionIncarnation = "4"; return q }},
		{"newer advertised revision", func(q peerSizingQuery) peerSizingQuery { q.PolicyRevision = "10"; return q }},
		{"profile engine changed", func(q peerSizingQuery) peerSizingQuery { q.Engine = "other"; return q }},
		{"profile model changed", func(q peerSizingQuery) peerSizingQuery { q.Model = "gemma-2"; return q }},
		{"absent profile", func(q peerSizingQuery) peerSizingQuery { q.Key = "large"; return q }},
		{"absent host", func(q peerSizingQuery) peerSizingQuery { q.Host = "elsewhere"; return q }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, ok := c.recipe(tc.query(peerSizingQueryFor("medium", "gemma"))); ok {
				t.Fatal("mismatched authority or profile returned a recipe")
			}
		})
	}
	if _, _, ok := c.recipe(peerSizingQueryFor("fast", "qwen")); !ok {
		t.Fatal("known no-op recipe missing before expiry")
	}
	clock = clock.Add(5 * time.Second)
	if _, _, ok := c.recipe(peerSizingQueryFor("fast", "qwen")); ok {
		t.Fatal("observation survived past the freshness window without a new snapshot")
	}
}

func TestPeerSizingCacheOwnsRecipeBytes(t *testing.T) {
	c := peerSizingCache{}
	s := proxyPeerSizingFixture(1, 0)
	if err := c.set(s); err != nil {
		t.Fatal(err)
	}
	s.Peers[0].Profiles[0].RequestOptions[2] = 'X'
	recipe, _, _ := c.recipe(peerSizingQueryFor("medium", "gemma"))
	if string(recipe.RequestOptions) != `{"max_tokens":64}` {
		t.Fatal("cache shares the sender's option bytes")
	}
	recipe.RequestOptions[2] = 'Y'
	again, _, _ := c.recipe(peerSizingQueryFor("medium", "gemma"))
	if string(again.RequestOptions) != `{"max_tokens":64}` {
		t.Fatal("lookup returned the cache's option bytes")
	}
}

func TestPeerSizingHandlerAcceptsOnlyUnaddressedValidSnapshots(t *testing.T) {
	p, w := policyProxy(t)
	response := rpc(t, p, w, nodepolicy.MethodSetPeerSizing, proxyPeerSizingFixture(2, 0))
	if response.Error != nil {
		t.Fatalf("valid snapshot rejected: %+v", response.Error)
	}
	var result struct {
		Peers int `json:"peers"`
	}
	if err := json.Unmarshal(response.Result, &result); err != nil || result.Peers != 1 {
		t.Fatalf("result = %s, %v", response.Result, err)
	}
	next := proxyPeerSizingFixture(3, 0)
	for name, call := range map[string]struct {
		method string
		params any
	}{
		"engine-addressed": {engines.AddressMethod("llamacpp", nodepolicy.MethodSetPeerSizing), next},
		"unknown field":    {nodepolicy.MethodSetPeerSizing, map[string]any{"revision": 3, "peers": next.Peers, "unknown": true}},
		"lower revision":   {nodepolicy.MethodSetPeerSizing, proxyPeerSizingFixture(1, 0)},
	} {
		t.Run(name, func(t *testing.T) {
			response := rpc(t, p, w, call.method, call.params)
			if response.Error == nil || response.Error.Code != -32602 {
				t.Fatalf("accepted: %+v", response)
			}
			if p.peerSizing.generation.Load() != 2 {
				t.Fatal("rejected RPC changed the cache")
			}
		})
	}
}

func TestPeerSizingIndependentOfPriorityAndReservations(t *testing.T) {
	p, _ := policyProxy(t)
	p.priority = []string{"destination"}
	p.priorityPending = map[string]int{"destination": 2}
	p.priorityReservations = map[string]int{"destination": 4}
	p.appliedPriorityGeneration = 7
	if err := p.peerSizing.set(proxyPeerSizingFixture(1, 0)); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p.priority, []string{"destination"}) || !reflect.DeepEqual(p.priorityPending, map[string]int{"destination": 2}) || !reflect.DeepEqual(p.priorityReservations, map[string]int{"destination": 4}) || p.appliedPriorityGeneration != 7 {
		t.Fatal("peer sizing changed scheduler or reservation state")
	}
}

func TestPeerSizingDiscoveryGenerationTracksPolicyOnlyChanges(t *testing.T) {
	d := NewDiscovery()
	base := Node{ID: "destination", Host: "dest.local", Port: 1, Addresses: []string{"10.0.0.2"}, ClusterUUID: "cluster-d", PolicyRevision: "9",
		Profiles: []nodepolicy.AdvertisedProfile{{Key: "medium", Engine: "llamacpp", Model: "gemma"}}}
	d.SetSubscribed([]Node{base})
	start := d.RoutingGeneration()
	if start == 0 {
		t.Fatal("discovery of a node did not advance the generation")
	}

	t.Run("unchanged replacement keeps generation", func(t *testing.T) {
		d.SetSubscribed([]Node{base})
		if d.RoutingGeneration() != start {
			t.Fatal("identical snapshot advanced the generation")
		}
	})
	steps := []struct {
		name   string
		change func(Node) Node
	}{
		{"policy revision only", func(n Node) Node { n.PolicyRevision = "10"; return n }},
		{"profile identity only", func(n Node) Node {
			n.Profiles = []nodepolicy.AdvertisedProfile{{Key: "medium", Engine: "llamacpp", Model: "gemma-2"}}
			return n
		}},
		{"cluster principal only", func(n Node) Node { n.ClusterUUID = "cluster-d2"; return n }},
	}
	current := base
	for _, step := range steps {
		t.Run(step.name, func(t *testing.T) {
			before := d.RoutingGeneration()
			current = step.change(current)
			_, updated, _ := d.SetSubscribed([]Node{current})
			if len(updated) != 0 {
				t.Fatalf("policy-only change announced a UI update: %+v", updated)
			}
			if d.RoutingGeneration() <= before {
				t.Fatal("policy-only change did not advance the generation")
			}
		})
	}
	t.Run("removal advances", func(t *testing.T) {
		before := d.RoutingGeneration()
		d.SetSubscribed(nil)
		if d.RoutingGeneration() <= before {
			t.Fatal("removal did not advance the generation")
		}
	})
	t.Run("nodes and generation captured together", func(t *testing.T) {
		d.AddManual(Node{ID: "manual", Host: "m.local", Port: 2})
		nodes, generation := d.NodesWithGeneration()
		if len(nodes) != 1 || generation != d.RoutingGeneration() {
			t.Fatalf("capture = %d nodes at %d, current %d", len(nodes), generation, d.RoutingGeneration())
		}
	})
}
