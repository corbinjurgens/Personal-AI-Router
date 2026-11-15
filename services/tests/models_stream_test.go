// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"net/http"
	"reflect"
	"slices"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

// TestPairedModelsStreamReplacesRefresh runs the production scanner against a
// paired peer whose em surface is the real nvpair-engine-manager (behind the
// fixture's counting observer, reached through the real reverse session) with a
// fixture-owned engine API. The scanner holds one GET /v1/models/events stream
// to the peer, and a residency change on the peer reaches the scanner's
// directory within 3 s, well under the old 15 s refresh, with no GET /v1/models
// in between.
//
// The residency change is the inventory change the fixture can drive: the
// peer's loaded-model watcher (1 s here) detects it and emits
// engine:models-changed, which triggers the publisher's 250 ms rebuild.
//
// Browse enrichment can still fetch GET /v1/models when an mDNS "updated"
// event (on this host they follow interface address changes) arrives before
// the stream is live or while it reconnects. So the test does not count
// fetches until the stream is live; a change is attributed to the stream only
// when no fetch reached the peer while it propagated, and is retried
// otherwise. Then a 16 s window, longer than the old refresh, must see no
// fetch unless a stream reconnect explains it.
func TestPairedModelsStreamReplacesRefresh(t *testing.T) {
	f := newTask2Fixture(t, task2WithLoadedPoll(1))
	em := f.b.observers[noderec.ServiceEngineManager]
	loaded := func() []string { return f.restricted(t).LoadedByEngine["llamacpp"] }
	streamCount := func() int { return em.count(f.a.uuid, http.MethodGet, "/v1/models/events") }
	fetchCount := func() int { return em.count(f.a.uuid, http.MethodGet, "/v1/models") }

	task1Eventually(t, "peer inventory", 35*time.Second, func() bool {
		node := f.restricted(t)
		return reflect.DeepEqual(node.Models, []string{"task2-model"}) && reflect.DeepEqual(node.LoadedByEngine["llamacpp"], []string{"task2-model"})
	})
	// Browse enrichment can fetch the first inventory before the stream
	// supervisor's next telemetry tick opens the stream.
	task1Eventually(t, "the scanner's models stream to the peer", 10*time.Second, func() bool {
		return streamCount() > 0
	})

	attributed := false
	for attempt := 1; attempt <= 3 && !attributed; attempt++ {
		f.mu.Lock()
		f.loaded = !f.loaded
		want := []string{}
		if f.loaded {
			want = []string{"task2-model"}
		}
		f.mu.Unlock()
		fetches := fetchCount()
		start := time.Now()
		task1Eventually(t, "residency change from the models stream", 3*time.Second, func() bool {
			got := loaded()
			return got != nil && slices.Equal(got, want)
		})
		elapsed := time.Since(start)
		attributed = fetchCount() == fetches
		t.Logf("attempt %d: residency change reached the scanner after %s; GET /v1/models during it: %d", attempt, elapsed, fetchCount()-fetches)
	}
	if !attributed {
		t.Fatal("every residency change coincided with a GET /v1/models; the stream's delivery could not be isolated")
	}
	// A quiet window longer than the old 15 s refresh: neither the refresh nor
	// browse enrichment fetches a peer whose stream is live.
	// A route change (an mDNS address update on this host) ends the stream,
	// and browse enrichment may fetch while it reconnects; only a fetch with
	// no reconnect to explain it fails.
	fetches, streams := fetchCount(), streamCount()
	time.Sleep(16 * time.Second)
	fetched, reconnected := fetchCount()-fetches, streamCount()-streams
	t.Logf("quiet window: GET /v1/models %d, stream reconnects %d", fetched, reconnected)
	if fetched > 0 && reconnected == 0 {
		t.Fatalf("live stream peer was fetched %d times over 16 s", fetched)
	}
}
