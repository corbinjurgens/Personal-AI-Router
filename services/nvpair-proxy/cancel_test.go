// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// cancelFixture routes one request to node a (which stalls) with node b behind
// it, so a test can act while the request is pinned to a.
type cancelFixture struct {
	p    *Proxy
	w    *recordingWriter
	a    *stallingEngine
	b    *recordingEngine
	done chan *httptest.ResponseRecorder
}

func startCancelFixture(t *testing.T, tc engineCase, a *stallingEngine) *cancelFixture {
	t.Helper()
	fx := &cancelFixture{w: &recordingWriter{}, a: a, b: newRecordingEngine(t, http.StatusOK), done: make(chan *httptest.ResponseRecorder, 1)}
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "a", a.URL, tc.advertisedModel))
	disc.AddManual(nodeForModel(t, "b", fx.b.URL, tc.advertisedModel))
	fx.p = newTestProxy(tc.profile, NewCodec(fx.w), disc, tc.profile.FacadePort)
	fx.p.SetPriority([]string{"a", "b"})
	go func() {
		rec := httptest.NewRecorder()
		fx.p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
		fx.done <- rec
	}()
	waitChan(t, a.received, "the request at node a")
	return fx
}

func (fx *cancelFixture) cancel(t *testing.T, engine string, regenerate bool) nodepolicy.WorkloadCancelResult {
	t.Helper()
	f := rpc(t, fx.p, fx.w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{
		WorkloadID: "1", Engine: engine, Regenerate: regenerate,
	})
	if f.Error != nil {
		t.Fatalf("workload/cancel failed: %s", f.Error.Message)
	}
	var res nodepolicy.WorkloadCancelResult
	if err := json.Unmarshal(f.Result, &res); err != nil {
		t.Fatal(err)
	}
	return res
}

func (fx *cancelFixture) result(t *testing.T) *httptest.ResponseRecorder {
	t.Helper()
	select {
	case rec := <-fx.done:
		return rec
	case <-time.After(5 * time.Second):
		t.Fatal("the request never finished")
		return nil
	}
}

// terminal returns the workload's terminal event.
func (fx *cancelFixture) terminal(t *testing.T) Workload {
	t.Helper()
	var events []json.RawMessage
	waitForCond(t, 2*time.Second, "a terminal workload event", func() bool {
		events = append(notificationsOf(t, fx.w, workloadCompletedMethod), notificationsOf(t, fx.w, workloadErroredMethod)...)
		return len(events) > 0
	})
	if len(events) != 1 {
		t.Fatalf("got %d terminal workload events, want exactly 1", len(events))
	}
	var params workloadParams
	if err := json.Unmarshal(events[0], &params); err != nil {
		t.Fatal(err)
	}
	return params.WorkloadInfo
}

func TestCancel_BeforeCommitWithRegenerateMovesOn(t *testing.T) {
	forEachEngine(t, func(t *testing.T, tc engineCase) {
		fx := startCancelFixture(t, tc, newStallingEngine(t))
		if res := fx.cancel(t, tc.profile.Name, true); !res.Found {
			t.Fatal("workload/cancel did not find the in-flight workload")
		}
		waitChan(t, fx.a.cancelled, "the attempt on a to be aborted")
		rec := fx.result(t)
		if rec.Code != http.StatusOK || fx.b.hits() != 1 {
			t.Fatalf("status %d, b hits %d: regenerate must continue on the next candidate", rec.Code, fx.b.hits())
		}
		if got := rec.Header().Get(nodepolicy.NodeHeader); got != "b" {
			t.Fatalf("served by %q, want b", got)
		}
		wl := fx.terminal(t)
		if wl.State != "completed" || wl.ScheduledOn != "b" {
			t.Fatalf("terminal = %s on %q, want completed on b", wl.State, wl.ScheduledOn)
		}
		// The regenerated job was visible moving: queued on a, then on b.
		var placements []string
		for _, raw := range notificationsOf(t, fx.w, workloadSubmittedMethod) {
			var params workloadParams
			_ = json.Unmarshal(raw, &params)
			placements = append(placements, params.WorkloadInfo.ScheduledOn)
		}
		if strings.Join(placements, ",") != ",a,b" {
			t.Fatalf("queued placements = %v, want [\"\" a b]", placements)
		}
	})
}

// The excluded node stays excluded for the rest of the request, even when it
// is the only owner left: the request ends instead of waiting it out.
func TestCancel_RegenerateWithNoOtherNodeEnds(t *testing.T) {
	tc := anyCase(t)
	a := newStallingEngine(t)
	w := &recordingWriter{}
	disc := NewDiscovery()
	disc.AddManual(nodeForModel(t, "a", a.URL, tc.advertisedModel))
	p := newTestProxy(tc.profile, NewCodec(w), disc, tc.profile.FacadePort)
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		rec := httptest.NewRecorder()
		p.soleFacade().handleHTTP(rec, tc.inferenceRequest())
		done <- rec
	}()
	waitChan(t, a.received, "the request at a")
	rpcOK(t, p, w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{WorkloadID: "1", Regenerate: true})
	select {
	case rec := <-done:
		if rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), "excluded") {
			t.Fatalf("status %d body %s", rec.Code, rec.Body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a regenerate with no other node waited instead of ending")
	}
}

func TestCancel_BeforeCommitWithoutRegenerateCancels(t *testing.T) {
	tc := anyCase(t)
	fx := startCancelFixture(t, tc, newStallingEngine(t))
	if res := fx.cancel(t, "", false); !res.Found {
		t.Fatal("not found")
	}
	waitChan(t, fx.a.cancelled, "the attempt on a to be aborted")
	fx.result(t)
	if fx.b.hits() != 0 {
		t.Fatal("a plain cancel re-dispatched the request")
	}
	wl := fx.terminal(t)
	if wl.State != "cancelled" || wl.Error == nil || *wl.Error != "cancelled by request" {
		t.Fatalf("terminal = %+v, want cancelled by request", wl)
	}
	if res := fx.cancel(t, "", false); res.Found {
		t.Fatal("a finished workload was still found")
	}
}

// streamingStallEngine commits a response and then holds the stream open.
func newStreamingStallEngine(t *testing.T) *stallingEngine {
	t.Helper()
	e := &stallingEngine{received: make(chan struct{}, 8), cancelled: make(chan struct{}, 8), release: make(chan struct{})}
	e.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"token":"first"}`+"\n")
		w.(http.Flusher).Flush()
		e.received <- struct{}{}
		select {
		case <-r.Context().Done():
			e.cancelled <- struct{}{}
		case <-e.release:
		}
	}))
	t.Cleanup(func() {
		e.doRelease()
		e.Close()
	})
	return e
}

// After the commit a regenerate cannot re-dispatch — that would splice two
// answers — so the stream is aborted and the workload cancelled.
func TestCancel_AfterCommitAbortsAndCancels(t *testing.T) {
	tc := anyCase(t)
	fx := startCancelFixture(t, tc, newStreamingStallEngine(t))
	waitForCond(t, 2*time.Second, "the commit", func() bool {
		return len(notificationsOf(t, fx.w, workloadStartedMethod)) == 1
	})
	if res := fx.cancel(t, tc.profile.Name, true); !res.Found {
		t.Fatal("not found")
	}
	waitChan(t, fx.a.cancelled, "the committed stream to be aborted")
	fx.result(t)
	if fx.b.hits() != 0 {
		t.Fatal("a committed request was re-dispatched")
	}
	wl := fx.terminal(t)
	if wl.State != "cancelled" {
		t.Fatalf("terminal state = %s, want cancelled", wl.State)
	}
}

func TestCancel_LookupRules(t *testing.T) {
	p, w := policyProxy(t)
	f := rpc(t, p, w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{WorkloadID: "nope"})
	if f.Error != nil || !strings.Contains(string(f.Result), `"found":false`) {
		t.Fatalf("unknown workload: %+v %s", f.Error, f.Result)
	}
	rpcErr(t, p, w, nodepolicy.MethodWorkloadCancel, map[string]any{})
	rpcErr(t, p, w, nodepolicy.MethodWorkloadCancel, map[string]any{"workloadId": "1", "engine": "vllm"})

	// Two facades can hold the same id; without an engine that is ambiguous.
	a, b := newWorkloadCtl(func(string) {}), newWorkloadCtl(func(string) {})
	p.registerWorkload(workloadKey{engine: "ollama", id: "4"}, a)
	p.registerWorkload(workloadKey{engine: "lmstudio", id: "4"}, b)
	f = rpc(t, p, w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{WorkloadID: "4", Regenerate: true})
	if f.Error == nil || f.Error.Code != -32602 || f.Error.Message != "ambiguous workload id; pass engine" {
		t.Fatalf("ambiguous id answered %+v", f.Error)
	}
	if a.isExcluded("") || b.done || a.done {
		t.Fatal("an ambiguous cancel acted on a workload")
	}
	rpcOK(t, p, w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{WorkloadID: "4", Engine: "lmstudio"})

	// A runId from another proxy process finds nothing.
	f = rpc(t, p, w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{WorkloadID: "4", Engine: "ollama", RunID: "other"})
	if !strings.Contains(string(f.Result), `"found":false`) {
		t.Fatalf("stale runId matched: %s", f.Result)
	}
}
