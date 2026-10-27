// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func TestStopPauseContractRefusalsHaveNoEffects(t *testing.T) {
	for _, tc := range []struct {
		name, state, mode    string
		peer, denied, fenced bool
		code                 int
		message              string
	}{
		{name: "local immediate unavailable", state: "paused", mode: `"immediate"`, code: -32000, message: errImmediatePauseUnavailable.Error()},
		{name: "local escalation unavailable", state: "paused", mode: `"immediate"`, fenced: true, code: -32000, message: errImmediatePauseUnavailable.Error()},
		{name: "remote immediate forged local target", state: "paused", mode: `"immediate"`, peer: true, code: -32602, message: "Immediate pause must be requested on this node."},
		{name: "revoked permission precedes immediate", state: "paused", mode: `"immediate"`, peer: true, denied: true, code: -32012, message: remoteManagementDeniedMessage},
		{name: "revoked permission precedes fence", state: "paused", mode: `"immediate"`, peer: true, denied: true, fenced: true, code: -32012, message: remoteManagementDeniedMessage},
		{name: "fence precedes immediate", state: "paused", mode: `"immediate"`, peer: true, fenced: true, code: -32011, message: (&localVetoError{}).Error()},
		{name: "initiator cannot repeat configured pause", state: "paused", mode: `"configured"`, peer: true, fenced: true, code: -32011},
		{name: "initiator cannot resume", state: "available", peer: true, fenced: true, code: -32011},
		{name: "permission precedes null mode", state: "paused", mode: `null`, peer: true, denied: true, code: -32012},
		{name: "null mode", state: "paused", mode: `null`, code: -32602},
		{name: "unknown mode", state: "paused", mode: `"fast"`, code: -32602},
		{name: "configured resume invalid", state: "available", mode: `"configured"`, code: -32602},
		{name: "immediate resume invalid", state: "available", mode: `"immediate"`, code: -32602},
		{name: "invalid state", state: "draining", code: -32602},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			if tc.denied {
				grantRemoteManagement(t, rb.b.clusterDir, "cluster-1", "self-node", "pinned-peer", false)
			}
			r := rb.b.nodePolicyRuntime()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r.mu.Lock()
			r.transitionGen = 17
			r.transitionCancel = cancel
			if tc.fenced {
				r.live, r.current.Availability, r.current.PausedBy = nodepolicy.Draining, nodepolicy.Paused, nodepolicy.PausedByPeer
			}
			live := r.live
			r.mu.Unlock()
			waitPushesSettled(t, rb.b)
			paProjectionBarrier(t, rb.b)
			before := rb.baseline(t)
			engine, proxy := rb.engine.snapshot(), rb.proxy.snapshot()
			projections := nodeProjections(t, rb.engine)
			availabilityNotices := func() []Message {
				var notices []Message
				for _, frame := range rb.out.frames() {
					if frame.Method == notifyAvailabilityChanged {
						notices = append(notices, frame)
					}
				}
				return notices
			}
			notices := availabilityNotices()
			params := policyParams{NodeID: rb.b.nodeID, State: tc.state, PauseMode: json.RawMessage(tc.mode)}
			var code int
			var message string
			if tc.peer {
				// Exercises the bound manager-source notification and verified caller,
				// even though the payload claims to target the broker's own node ID.
				reply := rb.relayReply(t, "set-availability", params)
				code, message = reply.ErrorCode, reply.Error
			} else {
				_, err := rb.b.dispatchPolicy(context.Background(), methodSetAvailability, params)
				if err == nil {
					t.Fatal("request unexpectedly succeeded")
				}
				code, message = policyErrorCode(err), err.Error()
			}
			if code != tc.code || (tc.message != "" && message != tc.message) {
				t.Fatalf("refusal = %d %q; want %d %q", code, message, tc.code, tc.message)
			}
			before.requireUnchanged(t)
			paProjectionBarrier(t, rb.b)
			if !reflect.DeepEqual(projections, nodeProjections(t, rb.engine)) || !reflect.DeepEqual(notices, availabilityNotices()) {
				t.Fatal("refused request published availability or engine projection")
			}
			if !reflect.DeepEqual(proxy, rb.proxy.snapshot()) {
				t.Fatal("refused request contacted proxy")
			}
			var effects []policyCall
			for _, call := range rb.engine.snapshot()[len(engine):] {
				if call.method != policyRelayReplyMethod && call.method != "pa:projection-barrier" {
					effects = append(effects, call)
				}
			}
			if len(effects) != 0 || ctx.Err() != nil {
				t.Fatalf("engine effects = %+v; previous transition error = %v", effects, ctx.Err())
			}
			r.mu.Lock()
			gen, pending, local, afterLive := r.transitionGen, r.pendingPauseRequested, r.localPauseRequested, r.live
			r.mu.Unlock()
			if gen != 17 || pending || local || live != afterLive {
				t.Fatalf("refused request registered transition: gen=%d pending=%t local=%t", gen, pending, local)
			}
		})
	}
}

func TestStopPauseContractConfiguredDrainPreservesPolicy(t *testing.T) {
	for _, mode := range []string{"", `"configured"`} {
		name := "omitted mode"
		if mode != "" {
			name = "explicit configured mode"
		}
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			before := rb.b.localPolicy().Policy
			rb.b.observeIntent(map[string]bool{"ollama": true, "llamacpp": false})
			rb.b.observeAdmission(nodepolicy.AdmissionState{Active: 1})
			type outcome struct {
				result any
				err    error
			}
			done := make(chan outcome, 1)
			go func() {
				result, err := rb.b.dispatchPolicy(context.Background(), methodSetAvailability, policyParams{State: "paused", PauseMode: json.RawMessage(mode)})
				done <- outcome{result, err}
			}()
			rb.proxy.waitFor("configured drain", func(calls []policyCall) bool {
				for _, push := range availabilityPushes(calls) {
					if push.State == nodepolicy.Draining && !push.CancelActive {
						return true
					}
				}
				return false
			})
			select {
			case got := <-done:
				t.Fatalf("drain finished with active work: %+v", got)
			default:
			}
			rb.b.observeAdmission(nodepolicy.AdmissionState{})
			select {
			case got := <-done:
				if got.err != nil {
					t.Fatal(got.err)
				}
				result, ok := got.result.(availabilitySetResult)
				if !ok || result.Availability != nodepolicy.Paused || result.PausedBy != nodepolicy.PausedLocally {
					t.Fatalf("local result = %+v", got.result)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("configured pause did not finish")
			}
			after := rb.b.localPolicy().Policy
			expected := before
			expected.Revision, expected.Availability, expected.PausedBy = after.Revision, nodepolicy.Paused, nodepolicy.PausedLocally
			if !reflect.DeepEqual(after, expected) {
				t.Fatal("configured pause changed saved options or declarations")
			}
			var disk nodepolicy.Policy
			if err := json.Unmarshal(rb.fileBytes(t), &disk); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(disk, after) {
				t.Fatal("saved policy differs from authoritative snapshot")
			}
			requirePolicyIntent(t, rb.b, map[string]bool{"ollama": true, "llamacpp": false})
			resumed, err := rb.b.dispatchPolicy(context.Background(), methodSetAvailability, policyParams{State: "available"})
			if err != nil {
				t.Fatal(err)
			}
			resume, ok := resumed.(availabilitySetResult)
			if !ok || resume.Availability != nodepolicy.Available || resume.PausedBy != "" {
				t.Fatalf("resume result = %+v", resumed)
			}
			if snapshot := rb.b.peerPolicy(); snapshot.Availability != nodepolicy.Available || snapshot.PausedBy != "" {
				t.Fatalf("peer resume readback = %+v", snapshot)
			}
		})
	}
}

func TestStopPauseContractProvenanceReadbackAndPush(t *testing.T) {
	rb := newRoutedBroker(t)
	decode := func(t *testing.T, value any, state nodepolicy.Availability, origin nodepolicy.PauseOrigin) {
		t.Helper()
		var got struct {
			Availability nodepolicy.Availability `json:"availability"`
			PausedBy     nodepolicy.PauseOrigin  `json:"pausedBy"`
		}
		if err := json.Unmarshal(rmJSON(t, value), &got); err != nil {
			t.Fatal(err)
		}
		if got.Availability != state || got.PausedBy != origin {
			t.Fatalf("snapshot = %+v; want %s/%s", got, state, origin)
		}
	}
	for _, tc := range []struct {
		name   string
		state  nodepolicy.Availability
		origin nodepolicy.PauseOrigin
	}{
		{"unknown origin", nodepolicy.Paused, ""},
		{"local drain", nodepolicy.Draining, nodepolicy.PausedLocally},
		{"peer pause", nodepolicy.Paused, nodepolicy.PausedByPeer},
		{"available clears provenance", nodepolicy.Available, nodepolicy.PausedByPeer},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := rb.b.nodePolicyRuntime()
			r.mu.Lock()
			r.live, r.current.PausedBy = tc.state, tc.origin
			r.mu.Unlock()
			want := tc.origin
			if tc.state == nodepolicy.Available {
				want = ""
			}
			decode(t, rb.b.localPolicy(), tc.state, want)
			decode(t, rb.b.peerPolicy(), tc.state, want)
			before := len(rb.out.frames())
			rb.b.notifyAvailability()
			for _, frame := range rb.out.frames()[before:] {
				if frame.Method == notifyAvailabilityChanged {
					decode(t, frame.Params, tc.state, want)
					return
				}
			}
			t.Fatal("no local availability push")
		})
	}
}

func TestStopPauseContractPeerResultAndAnnouncement(t *testing.T) {
	rb := newRoutedBroker(t)
	reply := rb.relayReply(t, "set-availability", policyParams{State: "paused", PauseMode: json.RawMessage(`"configured"`)})
	if reply.Error != "" {
		t.Fatalf("peer pause: %s", reply.Error)
	}
	var result map[string]json.RawMessage
	if err := json.Unmarshal(reply.Result, &result); err != nil {
		t.Fatal(err)
	}
	var origin nodepolicy.PauseOrigin
	if err := json.Unmarshal(result["pausedBy"], &origin); err != nil {
		t.Fatal(err)
	}
	if origin != nodepolicy.PausedByPeer || result["activation"] != nil {
		t.Fatalf("peer result = %s", reply.Result)
	}
	for _, tc := range []struct{ state, origin string }{{"paused", "peer"}, {"available", ""}} {
		before := len(rb.out.frames())
		rb.b.announceRemotePolicyChange(methodSetAvailability, "remote-node", rmJSON(t, map[string]string{"availability": tc.state, "pausedBy": "peer"}))
		frames := rb.out.frames()[before:]
		if len(frames) != 1 {
			t.Fatalf("remote announcements = %d", len(frames))
		}
		var got struct{ NodeID, Availability, PausedBy string }
		if err := json.Unmarshal(frames[0].Params, &got); err != nil {
			t.Fatal(err)
		}
		if got.NodeID != "remote-node" || got.Availability != tc.state || got.PausedBy != tc.origin {
			t.Fatalf("remote push = %+v", got)
		}
	}
}
