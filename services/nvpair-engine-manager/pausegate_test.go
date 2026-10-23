// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestPauseGateCoversEveryRegisteredMutationAndAllowsEveryRead(t *testing.T) {
	for _, scenario := range []struct {
		name       string
		projection *nodeProjection
		allowed    bool
		status     int
		code       string
	}{
		{"available", &nodeProjection{ManagementBlocked: false}, true, http.StatusOK, ""},
		{"unknown projection", nil, true, http.StatusServiceUnavailable, ""},
		{"local pause", &nodeProjection{PausedLocally: true, ManagementBlocked: true}, true, http.StatusConflict, "local_veto"},
		{"peer pause", &nodeProjection{Policy: nodepolicy.Advertisement{PausedBy: nodepolicy.PausedByPeer}, ManagementBlocked: true}, true, http.StatusConflict, "local_veto"},
		{"grant denial precedes pause", &nodeProjection{ManagementBlocked: true}, false, http.StatusConflict, "remote_management_denied"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			f := rmNewFixture(t, "uuid-self", "uuid-a")
			f.writePermissions("1", map[string]bool{"uuid-a": scenario.allowed})
			ex := newTestExecutor(t, lockLoadManifest())
			if err := ex.setEngineLocked("fake", true); err != nil {
				t.Fatal(err)
			}
			if scenario.projection != nil {
				ex.nodeProjection.set(*scenario.projection)
			}
			s := &controlServer{exec: ex, mesh: f.mesh}
			spy := newRMHandlerSpy()
			classes := rmExpectedClasses()
			for _, route := range s.routes() {
				t.Run(route.method+" "+route.path, func(t *testing.T) {
					class, ok := classes[route.path]
					if !ok || class != route.class {
						t.Fatalf("unclassified or changed route: %+v", route)
					}
					before := spy.count(route.family)
					handler := s.authorize(route, spy.handle(route.family))
					rec := httptest.NewRecorder()
					handler(rec, httptest.NewRequest(route.method, route.path, strings.NewReader(`{"engine":"fake"}`)), "uuid-a")
					want := http.StatusOK
					if class == controlMutation {
						want = scenario.status
					}
					if rec.Code != want {
						t.Fatalf("gate = %d %s, want %d", rec.Code, rec.Body.String(), want)
					}
					calls := spy.count(route.family) - before
					wantCalls := 0
					if want == http.StatusOK {
						wantCalls = 1
					}
					if calls != wantCalls {
						t.Fatalf("handler calls = %d, want %d", calls, wantCalls)
					}
					if class == controlMutation && scenario.code != "" {
						var body codedBody
						if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
							t.Fatal(err)
						}
						if body.Code != scenario.code {
							t.Fatalf("code = %q, want %q", body.Code, scenario.code)
						}
						if scenario.code == "local_veto" && body.Error != "This node is paused. Resume it on this node before managing it remotely." {
							t.Fatalf("pause message = %q", body.Error)
						}
					}
					if class == controlRead && route.method == http.MethodGet {
						rec := httptest.NewRecorder()
						handler(rec, httptest.NewRequest(http.MethodHead, route.path, nil), "uuid-a")
						if rec.Code != http.StatusOK || spy.count(route.family) != before+2 {
							t.Fatalf("HEAD blocked: %d %s", rec.Code, rec.Body.String())
						}
					}
				})
			}
		})
	}
}

func TestPeerPauseProjectionCancelsActivation(t *testing.T) {
	for _, origin := range []string{"local", "peer"} {
		t.Run(origin+" pause cancels accepted local start", func(t *testing.T) {
			ex := newTestExecutor(t, slowStartManifest())
			t.Cleanup(ex.StopAll)
			st, err := ex.state("fake")
			if err != nil {
				t.Fatal(err)
			}
			started := make(chan error, 1)
			go func() { started <- ex.Start(context.Background(), "fake") }()
			awaitLockCondition(t, "local readiness wait", func() bool { st.mu.Lock(); defer st.mu.Unlock(); return st.proc != nil })
			pauseOrigin := nodepolicy.PausedByPeer
			if origin == "local" {
				pauseOrigin = nodepolicy.PausedLocally
			}
			projection := nodeProjection{ManagementBlocked: true, PausedLocally: origin == "local", Policy: nodepolicy.Advertisement{PausedBy: pauseOrigin}}
			(&Manager{exec: ex}).applyNodeProjection(&Message{Params: json.RawMessage(mustJSONText(t, projection))})
			if err := awaitLockCommand(t, started); rpcCode(err) != -32011 {
				t.Fatalf("cancelled local start = %v, want -32011", err)
			}
		})
	}
	t.Run("accepted remote start", func(t *testing.T) {
		s, ex := vetoTestServer(t, slowStartManifest())
		ex.nodeProjection.set(nodeProjection{ManagementBlocked: false})
		done := startRemotely(t, s)
		(&Manager{exec: ex}).applyNodeProjection(&Message{Params: json.RawMessage(`{"policy":{"pausedBy":"peer"},"pausedLocally":false,"managementBlocked":true}`)})
		requireLocalVeto(t, "peer pause cancels start", awaitResponse(t, done))
		if enabled, known, err := ex.desired.get("fake"); err != nil || (known && enabled) {
			t.Fatalf("pause accepted late On: %v, %v, %v", enabled, known, err)
		}
	})
	t.Run("local and remote load waiters", func(t *testing.T) {
		ex := newTestExecutor(t, lockLoadManifest())
		ex.nodeProjection.set(nodeProjection{ManagementBlocked: false})
		remote, finish, err := ex.beginRemote(context.Background(), "fake")
		if err != nil {
			t.Fatal(err)
		}
		defer finish()
		st, err := ex.state("fake")
		if err != nil {
			t.Fatal(err)
		}
		st.opMu.Lock()
		unlock := sync.OnceFunc(st.opMu.Unlock)
		defer unlock()
		done := make(chan error, 2)
		for _, ctx := range []context.Context{context.Background(), remote} {
			go func() { _, err := ex.ModelLoad(ctx, "fake", "tiny"); done <- err }()
		}
		awaitLockCondition(t, "both waiting loads", func() bool { st.mu.Lock(); defer st.mu.Unlock(); return len(st.loads.active) == 2 })
		(&Manager{exec: ex}).applyNodeProjection(&Message{Params: json.RawMessage(`{"pausedLocally":false,"managementBlocked":true}`)})
		for i := 0; i < 2; i++ {
			if err := awaitLockCommand(t, done); rpcCode(err) != -32011 {
				t.Fatalf("cancelled local/remote load = %v, want -32011", err)
			}
		}
		if rpcCode(context.Cause(remote)) != -32011 {
			t.Fatalf("remote pause cause = %v", context.Cause(remote))
		}
		unlock()
		if err := ex.Start(context.Background(), "fake"); rpcCode(err) != -32011 {
			t.Fatalf("local start after peer pause = %v", err)
		}
		if _, err := ex.ModelLoad(context.Background(), "fake", "tiny"); rpcCode(err) != -32011 {
			t.Fatalf("local load after peer pause = %v", err)
		}
		if err := ex.setEngineLocked("fake", true); err != nil {
			t.Fatalf("local lock while paused: %v", err)
		}
		if err := ex.setEngineLocked("fake", false); err != nil {
			t.Fatalf("local unlock while paused: %v", err)
		}
		if err := ex.Stop("fake"); err != nil {
			t.Fatalf("local stop while paused: %v", err)
		}
	})
}
