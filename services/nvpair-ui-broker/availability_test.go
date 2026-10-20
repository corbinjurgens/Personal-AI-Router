// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// paProjectionBarrier replies only after the fake engine's single reader has
// recorded every preceding projection on this connection.
func paProjectionBarrier(t *testing.T, b *Broker) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	raw, rpcErr, err := b.getEngineMgr().Call(ctx, "pa:projection-barrier", json.RawMessage(`{}`))
	if err != nil || rpcErr != nil {
		t.Fatalf("projection reader barrier: transport=%v rpc=%+v", err, rpcErr)
	}
	var reply struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal(raw, &reply); err != nil {
		t.Fatalf("decode projection reader barrier: %v", err)
	}
	if !reply.OK {
		t.Fatal("projection reader barrier was not acknowledged")
	}
}

func TestPauseProjectionCarriesProvenanceAndRevision(t *testing.T) {
	fixture := func(t *testing.T) (*activatedBroker, *fakeWorker) {
		t.Helper()
		ab := newActivatedBroker(t)
		_, err := ab.b.dispatchPolicy(context.Background(), methodPolicySet, policyParams{
			Policy: mustJSON(map[string]any{
				"version": nodepolicy.SchemaVersion,
				"roles":   map[string]any{"weak": map[string]any{"engine": "ollama", "model": "fixture-model"}},
			}),
			ExpectedPersistedRevision: mustJSON(ab.b.localPolicy().Policy.Revision),
		})
		if err != nil {
			t.Fatalf("save role fixture: %v", err)
		}
		return ab, attachFakeEngine(t, ab.b)
	}
	transition := func(t *testing.T, ab *activatedBroker, engine *fakeWorker, state nodepolicy.Availability, origin nodepolicy.PauseOrigin) {
		t.Helper()
		before := ab.b.localPolicy().Policy
		result, err := ab.b.setAvailability(context.Background(), state, origin)
		if err != nil {
			t.Fatalf("set availability %s from %s: %v", state, origin, err)
		}
		if result != state {
			t.Fatalf("availability result = %s, want %s", result, state)
		}
		saved := ab.b.localPolicy().Policy
		wantOrigin := origin
		if state == nodepolicy.Available {
			wantOrigin = ""
		}
		if saved.Availability != state || saved.PausedBy != wantOrigin || saved.Revision == before.Revision {
			t.Fatalf("saved policy = %+v; want %s/%s with a revision after %s", saved, state, wantOrigin, before.Revision)
		}
		want := saved.Advertisement()
		if !reflect.DeepEqual(want.Roles, before.Advertisement().Roles) || len(want.Roles) == 0 {
			t.Fatalf("availability changed declared roles: %+v", want.Roles)
		}
		var matched nodepolicy.Advertisement
		engine.waitFor("saved availability advertisement", func([]policyCall) bool {
			for _, got := range nodeProjections(t, engine) {
				// A pending local veto still carries the previous saved revision
				// and cannot satisfy this complete advertisement comparison.
				if reflect.DeepEqual(got, want) {
					matched = got
					return true
				}
			}
			return false
		})
		if stored := ab.b.localPolicy().Policy.Advertisement(); !reflect.DeepEqual(matched, stored) {
			t.Fatalf("matched advertisement = %+v, stored = %+v", matched, stored)
		}
	}
	emission := func(name string, state nodepolicy.Availability, origin nodepolicy.PauseOrigin) {
		t.Run(name, func(t *testing.T) {
			ab, engine := fixture(t)
			if state == nodepolicy.Available {
				transition(t, ab, engine, nodepolicy.Paused, origin)
			}
			transition(t, ab, engine, state, origin)
		})
	}
	emission("local pause", nodepolicy.Paused, nodepolicy.PausedLocally)
	emission("local resume", nodepolicy.Available, nodepolicy.PausedLocally)
	emission("peer pause", nodepolicy.Paused, nodepolicy.PausedByPeer)
	emission("peer resume", nodepolicy.Available, nodepolicy.PausedByPeer)

	t.Run("peer resume veto emits no projection", func(t *testing.T) {
		ab, engine := fixture(t)
		transition(t, ab, engine, nodepolicy.Paused, nodepolicy.PausedLocally)
		paProjectionBarrier(t, ab.b)
		before := nodeProjections(t, engine)
		saved := ab.b.localPolicy().Policy
		result, err := ab.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedByPeer)
		var veto *localVetoError
		if !errors.As(err, &veto) || result != "" {
			t.Fatalf("peer resume = %q, %v; want localVetoError and no result", result, err)
		}
		if got := ab.b.localPolicy().Policy; !reflect.DeepEqual(got, saved) {
			t.Fatalf("veto changed stored policy: got %+v, want %+v", got, saved)
		}
		paProjectionBarrier(t, ab.b)
		if got := nodeProjections(t, engine); !reflect.DeepEqual(got, before) {
			t.Fatalf("veto emitted a projection: before %+v, after %+v", before, got)
		}
	})
}
