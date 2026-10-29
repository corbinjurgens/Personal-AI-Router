// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

func TestHandleHTTP_PeerRefusalNeverStartsWorkload(t *testing.T) {
	setForTest(t, &retryBackoff, []time.Duration{time.Millisecond})
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := paNewFixture(t, tc.profile)
		peer := fx.peers(t, "paused-peer")[0]
		// Discovery still advertises availability when the destination pauses.
		peer.p.admission.setAvailability(nodepolicy.Paused, true)
		fx.replace(t, peer.node)
		response := fx.finish(t, fx.request(http.MethodPost, tc.inferencePath,
			fmt.Sprintf(`{"model":%q}`, paModel)), http.StatusServiceUnavailable)
		var refusal struct {
			Code   string `json:"code"`
			Reason string `json:"reason"`
		}
		if err := json.Unmarshal(response.body, &refusal); err != nil {
			t.Fatalf("decode refusal: %v", err)
		}
		if refusal.Code != "admission" || refusal.Reason != string(nodepolicy.RejectPaused) || response.header.Get(nodepolicy.AdmissionHeader) != string(nodepolicy.RejectPaused) {
			t.Fatalf("refusal = %+v, headers = %v", refusal, response.header)
		}
		if peer.ingress.Load() != 5 || peer.inference.Load() != 0 {
			t.Fatalf("peer ingress/engine calls = %d/%d, want 5/0", peer.ingress.Load(), peer.inference.Load())
		}
		var terminal *Workload
		for _, event := range frames(t, fx.w) {
			switch event.Method {
			case engines.AddressMethod(tc.profile.Name, "proxy/request-started"):
				t.Error("refusal published proxy/request-started")
			case workloadSubmittedMethod, workloadStartedMethod, workloadCompletedMethod, workloadErroredMethod:
				var params workloadParams
				if err := json.Unmarshal(event.Params, &params); err != nil {
					t.Fatalf("decode workload: %v", err)
				}
				wl := params.WorkloadInfo
				if event.Method == workloadStartedMethod || wl.State == "running" || wl.StartedAt != nil {
					t.Errorf("refusal published execution: method=%s workload=%+v", event.Method, wl)
				}
				if event.Method == workloadErroredMethod {
					if terminal != nil {
						t.Fatal("duplicate terminal event")
					}
					terminal = &wl
				}
			}
		}
		if terminal == nil || terminal.State != "failed" || terminal.CompletedAt == nil || terminal.Error == nil || *terminal.Error != "upstream returned HTTP 503" {
			t.Fatalf("terminal = %+v, want failed before start with HTTP 503 error", terminal)
		}
	})
}

// recRW is a thread-safe io.ReadWriter that records everything the codec writes
// (newline-delimited JSON-RPC frames) so a test can assert which notifications
// were emitted. Reads hit EOF immediately.
type recRW struct {
	mu sync.Mutex
	b  []byte
}

func (r *recRW) Read([]byte) (int, error) { return 0, io.EOF }

func (r *recRW) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.b = append(r.b, p...)
	return len(p), nil
}

func (r *recRW) has(s string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return strings.Contains(string(r.b), s)
}

// TestHandleHTTP_WorkloadVisibleBeforeFirstByte is a regression test: a job
// must be on the wire as soon as it is admitted, not when the upstream first
// responds. An engine serializes inference on a single GPU slot, so a burst of
// parallel submissions is dispatched concurrently but streams back one at a
// time; emitting only at first byte made the waiting submissions invisible and
// they appeared one card at a time. Here the upstream accepts the connection
// but never sends a byte until released, so a first-byte emission could not
// have fired — yet the job must already be visible.
//
// The event is workload:submitted, carrying state "queued". "running" is
// reserved for the commit point, when the engine is actually producing content:
// a job waiting its turn inside the engine is not running, and the broker's
// store rejects a backwards transition, so a job that claimed "running" at
// dispatch time could never return to "queued" for a retry.
//
// proxy/request-started also stays at the commit point, so it names the node
// that actually served after any failover.
func TestHandleHTTP_WorkloadVisibleBeforeFirstByte(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		received := make(chan struct{}, 1)
		release := make(chan struct{})
		var releaseOnce sync.Once
		doRelease := func() { releaseOnce.Do(func() { close(release) }) }

		slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case received <- struct{}{}:
			default:
			}
			<-release // no response byte until the test releases us
			w.WriteHeader(http.StatusOK)
			io.WriteString(w, `{"done":true}`)
		}))
		// Deferred order matters (LIFO): doRelease runs BEFORE slow.Close so
		// the blocked upstream handler is unblocked before the server is torn
		// down — otherwise Close hangs on the active connection, including on
		// t.Fatal.
		defer slow.Close()
		defer doRelease()

		rec := &recRW{}
		disc := NewDiscovery()
		disc.AddManual(nodeForModel(t, "busy-node", slow.URL, tc.advertisedModel))
		p := newTestProxy(tc.profile, NewCodec(rec), disc, tc.profile.FacadePort)

		done := make(chan struct{})
		go func() {
			defer close(done)
			p.soleFacade().handleHTTP(httptest.NewRecorder(), httptest.NewRequest(
				http.MethodPost, tc.inferencePath,
				strings.NewReader(fmt.Sprintf(`{"model":%q}`, tc.requestedModel))))
		}()

		select {
		case <-received:
		case <-time.After(5 * time.Second):
			t.Fatal("upstream never received the forwarded request")
		}

		// The upstream has the request but has sent no response byte, so a
		// first-byte emission could not have fired. The job must still be
		// visible, as queued.
		if !waitFor(t, rec, "workload:submitted") {
			t.Fatal("workload:submitted not emitted when the request was admitted")
		}
		if !rec.has(`"state":"queued"`) {
			t.Fatal("an admitted job must be queued until the engine produces content")
		}
		if rec.has("workload:started") {
			t.Fatal("workload:started emitted before the engine produced any content")
		}

		doRelease()
		<-done

		// Released: the engine produced content, so the job is now running and
		// then completes. queued -> running -> completed, never backwards.
		if !waitFor(t, rec, "workload:started") {
			t.Fatal("workload:started not emitted once the engine produced content")
		}
		if !rec.has(`"state":"running"`) {
			t.Fatal("the commit point must transition the job to running")
		}
	})
}
