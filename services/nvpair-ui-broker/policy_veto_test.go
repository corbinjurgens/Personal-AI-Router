// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// localAvailability sends node:set-availability as a local client does and
// returns the response.
func (rb *routedBroker) localAvailability(t *testing.T, id int, state nodepolicy.Availability) Message {
	t.Helper()
	rb.b.handleMessage(clientRequest(id, methodSetAvailability, map[string]any{"state": state}))
	return rb.out.waitFrame(t, "node:set-availability response", isResponse(id))
}

// requirePause fails unless the stored policy has availability and pausedBy.
func requirePause(t *testing.T, b *Broker, availability nodepolicy.Availability, pausedBy nodepolicy.PauseOrigin) {
	t.Helper()
	got := b.localPolicy().Policy
	if got.Availability != availability || got.PausedBy != pausedBy {
		t.Fatalf("stored availability %q pausedBy %q, want %q pausedBy %q", got.Availability, got.PausedBy, availability, pausedBy)
	}
}

// requirePausedLocally waits until the newest engine:node-projection the fake
// engine-manager has read carries pausedLocally want. A notification reaches
// the fake asynchronously, so the newest one may still be in flight.
func requirePausedLocally(t *testing.T, engine *fakeWorker, want bool, what string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		engine.mu.Lock()
		calls := append([]policyCall(nil), engine.projections...)
		engine.mu.Unlock()
		if len(calls) > 0 {
			var p nodeProjection
			if err := json.Unmarshal(calls[len(calls)-1].params, &p); err != nil {
				t.Fatalf("decode node projection: %v", err)
			}
			if p.PausedLocally == want {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("%s: the newest node projection never carried pausedLocally %t", what, want)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestPeerResumeOfALocalPauseIsVetoed(t *testing.T) {
	rb := newRoutedBroker(t)
	if resp := rb.localAvailability(t, 1, nodepolicy.Paused); resp.Error != nil {
		t.Fatalf("local pause: %+v", resp.Error)
	}
	requirePause(t, rb.b, nodepolicy.Paused, nodepolicy.PausedLocally)
	requirePausedLocally(t, rb.engine, true, "the local pause was not projected to engine-manager as local")
	paused := rb.b.localPolicy().Policy.Revision

	reply := rb.relayReply(t, "set-availability", map[string]any{"state": nodepolicy.Available})
	if reply.ErrorCode != codeLocalVeto || reply.Error != (&localVetoError{}).Error() {
		t.Fatalf("relayed resume of a local pause = code %d %q, want %d local veto", reply.ErrorCode, reply.Error, codeLocalVeto)
	}
	requirePause(t, rb.b, nodepolicy.Paused, nodepolicy.PausedLocally)
	if got := rb.b.localPolicy(); got.Policy.Revision != paused || got.Availability != nodepolicy.Paused {
		t.Fatalf("vetoed resume changed revision %s -> %s or live %q", paused, got.Policy.Revision, got.Availability)
	}

	if resp := rb.localAvailability(t, 2, nodepolicy.Available); resp.Error != nil {
		t.Fatalf("local resume after a vetoed peer resume: %+v", resp.Error)
	}
	requirePause(t, rb.b, nodepolicy.Available, "")
	requirePausedLocally(t, rb.engine, false, "the local resume left the projection paused locally")
}

func TestPeerCannotResumePeerPause(t *testing.T) {
	rb := newRoutedBroker(t)
	if _, errText := rb.relay(t, "set-availability", map[string]any{"state": nodepolicy.Paused}); errText != "" {
		t.Fatalf("relayed pause: %s", errText)
	}
	requirePause(t, rb.b, nodepolicy.Paused, nodepolicy.PausedByPeer)
	requirePausedLocally(t, rb.engine, false, "a peer pause was projected as local")
	for _, state := range []nodepolicy.Availability{nodepolicy.Available, nodepolicy.Paused} {
		before := rb.baseline(t)
		reply := rb.relayReply(t, "set-availability", map[string]any{"state": state})
		if reply.ErrorCode != codeLocalVeto || reply.Error != "This node is paused. Resume it on this node before managing it remotely." {
			t.Fatalf("peer %s after its pause = code %d %q, want the pause refusal", state, reply.ErrorCode, reply.Error)
		}
		before.requireUnchanged(t)
		requirePause(t, rb.b, nodepolicy.Paused, nodepolicy.PausedByPeer)
	}
	if resp := rb.localAvailability(t, 1, nodepolicy.Available); resp.Error != nil {
		t.Fatalf("local resume of a peer pause: %+v", resp.Error)
	}
	requirePause(t, rb.b, nodepolicy.Available, "")
}

func TestPauseProvenanceIsNeverDowngraded(t *testing.T) {
	t.Run("peer pause over a local pause is refused without changes", func(t *testing.T) {
		rb := newRoutedBroker(t)
		if resp := rb.localAvailability(t, 1, nodepolicy.Paused); resp.Error != nil {
			t.Fatalf("local pause: %+v", resp.Error)
		}
		before := rb.baseline(t)
		reply := rb.relayReply(t, "set-availability", map[string]any{"state": nodepolicy.Paused})
		if reply.ErrorCode != codeLocalVeto || reply.Error != "This node is paused. Resume it on this node before managing it remotely." {
			t.Fatalf("repeated peer pause = code %d %q, want the pause refusal", reply.ErrorCode, reply.Error)
		}
		before.requireUnchanged(t)
		requirePause(t, rb.b, nodepolicy.Paused, nodepolicy.PausedLocally)
		if reply := rb.relayReply(t, "set-availability", map[string]any{"state": nodepolicy.Available}); reply.ErrorCode != codeLocalVeto {
			t.Fatalf("relayed resume after a peer pause over a local one = code %d %q, want local veto", reply.ErrorCode, reply.Error)
		}
	})
	t.Run("local pause over a peer pause becomes local", func(t *testing.T) {
		rb := newRoutedBroker(t)
		if _, errText := rb.relay(t, "set-availability", map[string]any{"state": nodepolicy.Paused}); errText != "" {
			t.Fatalf("relayed pause: %s", errText)
		}
		if resp := rb.localAvailability(t, 1, nodepolicy.Paused); resp.Error != nil {
			t.Fatalf("local pause: %+v", resp.Error)
		}
		requirePause(t, rb.b, nodepolicy.Paused, nodepolicy.PausedLocally)
		requirePausedLocally(t, rb.engine, true, "the local pause over a peer pause was not projected as local")
	})
}

func TestPauseWithoutProvenanceCountsAsLocal(t *testing.T) {
	rb := newBrokerOnFile(t, `{"version":2,"revision":"12","availability":"paused"}`)
	requirePause(t, rb.b, nodepolicy.Paused, "")
	rb.b.publishNodeProjection()
	requirePausedLocally(t, rb.engine, true, "a pause saved before provenance existed was not projected as local")
	if reply := rb.relayReply(t, "set-availability", map[string]any{"state": nodepolicy.Available}); reply.ErrorCode != codeLocalVeto {
		t.Fatalf("relayed resume of a legacy pause = code %d %q, want local veto", reply.ErrorCode, reply.Error)
	}
	requirePause(t, rb.b, nodepolicy.Paused, "")
	if resp := rb.localAvailability(t, 1, nodepolicy.Available); resp.Error != nil {
		t.Fatalf("local resume of a legacy pause: %+v", resp.Error)
	}
	requirePause(t, rb.b, nodepolicy.Available, "")
}

// A local pause that is accepted but has not reached the stored policy yet
// already vetoes a peer resume, which therefore cannot supersede it.
func TestAcceptedLocalPauseVetoesPeerResumeBeforeItIsSaved(t *testing.T) {
	rb := newRoutedBroker(t)
	r := rb.b.nodePolicyRuntime()
	r.transitionMu.Lock()
	paused := make(chan error, 1)
	go func() {
		_, err := rb.b.setAvailability(context.Background(), nodepolicy.Paused, nodepolicy.PausedLocally)
		paused <- err
	}()
	deadline := time.Now().Add(3 * time.Second)
	for {
		r.mu.Lock()
		accepted := r.localPauseRequested
		r.mu.Unlock()
		if accepted {
			break
		}
		if time.Now().After(deadline) {
			r.transitionMu.Unlock()
			t.Fatal("the local pause was not accepted")
		}
		time.Sleep(5 * time.Millisecond)
	}
	requirePause(t, rb.b, nodepolicy.Available, "")
	_, err := rb.b.setAvailability(context.Background(), nodepolicy.Available, nodepolicy.PausedByPeer)
	r.transitionMu.Unlock()
	if policyErrorCode(err) != codeLocalVeto {
		t.Fatalf("peer resume during an accepted local pause = %v, want a local veto", err)
	}
	if err := <-paused; err != nil {
		t.Fatalf("the local pause was disturbed by the vetoed resume: %v", err)
	}
	requirePause(t, rb.b, nodepolicy.Paused, nodepolicy.PausedLocally)
	r.mu.Lock()
	pending := r.localPauseRequested
	r.mu.Unlock()
	if pending {
		t.Fatal("the finished local pause left its request pending")
	}
}

func TestConfigurationEditsStayAllowedWhileLocallyPaused(t *testing.T) {
	rb := newRoutedBroker(t)
	if resp := rb.localAvailability(t, 1, nodepolicy.Paused); resp.Error != nil {
		t.Fatalf("local pause: %+v", resp.Error)
	}
	t.Run("peer policy:set is refused without changes", func(t *testing.T) {
		doc := rb.peerSnapshot(t)
		doc["idle"].(map[string]any)["unloadAfterMinutes"] = 9
		before := rb.baseline(t)
		reply := rb.relayReply(t, "set", map[string]any{"policy": doc, "expectedPersistedRevision": doc["revision"]})
		if reply.ErrorCode != codeLocalVeto || reply.Error != "This node is paused. Resume it on this node before managing it remotely." {
			t.Fatalf("relayed policy:set = code %d %q, want the pause refusal", reply.ErrorCode, reply.Error)
		}
		before.requireUnchanged(t)
		requirePause(t, rb.b, nodepolicy.Paused, nodepolicy.PausedLocally)
	})
	t.Run("local policy:set ignores the pausedBy it sends", func(t *testing.T) {
		for _, sent := range []any{nodepolicy.PausedByPeer, "", "bogus"} {
			rb.localSet(t, func(doc map[string]any) {
				doc["pausedBy"] = sent
				doc["availability"] = nodepolicy.Available
				doc["idle"].(map[string]any)["unloadAfterMinutes"] = 11
			})
			requirePause(t, rb.b, nodepolicy.Paused, nodepolicy.PausedLocally)
		}
	})
}
