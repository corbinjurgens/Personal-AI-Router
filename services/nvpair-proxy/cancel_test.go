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
	p       *Proxy
	w       *recordingWriter
	a       *stallingEngine
	b       *recordingEngine
	done    chan *httptest.ResponseRecorder
	control *workloadCtl
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
	controls := fx.p.workloadControls(tc.profile.Name, "1")
	if len(controls) != 1 {
		t.Fatal("request has no canonical workload controller")
	}
	fx.control = controls[0]
	return fx
}

func (fx *cancelFixture) cancel(t *testing.T, engine string, regenerate bool) nodepolicy.WorkloadCancelResult {
	t.Helper()
	if engine == "" {
		engine = fx.p.soleFacade().profile.Name
	}
	f := rpc(t, fx.p, fx.w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{
		OriginatedFrom: "test-origin", RunID: fx.p.runID, WorkloadID: "1", Engine: engine, Regenerate: regenerate,
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
		fx.control.mu.Lock()
		done := fx.control.done
		fx.control.mu.Unlock()
		if !done {
			t.Fatal("finished request retained live cancellation controller")
		}
		if len(fx.p.activeRoutingRequests()) != 0 || len(fx.p.workloadControls(fx.p.soleFacade().profile.Name, "1")) != 0 {
			t.Fatal("finished request retained registry or workload controller association")
		}
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
	rpcOK(t, p, w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{OriginatedFrom: "test-origin", RunID: p.runID, Engine: tc.profile.Name, WorkloadID: "1", Regenerate: true})
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
		w.Header().Set("Content-Type", "application/json")
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
	f := rpc(t, p, w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{OriginatedFrom: "test-origin", RunID: p.runID, Engine: "ollama", WorkloadID: "nope"})
	if f.Error != nil || !strings.Contains(string(f.Result), `"found":false`) {
		t.Fatalf("unknown workload: %+v %s", f.Error, f.Result)
	}
	rpcErr(t, p, w, nodepolicy.MethodWorkloadCancel, map[string]any{})
	rpcErr(t, p, w, nodepolicy.MethodWorkloadCancel, map[string]any{"workloadId": "1", "engine": "v/llm"})

	// Two facades can hold the same id; every complete target selects one.
	var cancelledA, cancelledB int
	a, b := newWorkloadCtl(func(string) { cancelledA++ }), newWorkloadCtl(func(string) { cancelledB++ })
	ra, err := p.registerRoutingRequest(a)
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(ra)
	rb, err := p.registerRoutingRequest(b)
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(rb)
	if err := p.registerWorkload(ra, workloadKey{engine: "ollama", id: "4"}); err != nil {
		t.Fatal(err)
	}
	if err := p.registerWorkload(rb, workloadKey{engine: "lmstudio", id: "4"}); err != nil {
		t.Fatal(err)
	}
	valid := nodepolicy.WorkloadCancelParams{OriginatedFrom: "test-origin", RunID: p.runID, WorkloadID: "4", Engine: "ollama"}
	test := func(name string, change func(*nodepolicy.WorkloadCancelParams), invalid bool) {
		t.Run(name, func(t *testing.T) {
			in := valid
			change(&in)
			response := rpc(t, p, w, nodepolicy.MethodWorkloadCancel, in)
			if invalid {
				if response.Error == nil || response.Error.Code != -32602 {
					t.Fatalf("incomplete target accepted: %+v", response)
				}
			} else {
				var result nodepolicy.WorkloadCancelResult
				if response.Error != nil || json.Unmarshal(response.Result, &result) != nil || result.Found {
					t.Fatalf("wrong target matched: %+v", response)
				}
			}
			if cancelledA != 0 || cancelledB != 0 {
				t.Fatalf("rejected target cancelled controls: a=%d b=%d", cancelledA, cancelledB)
			}
		})
	}
	test("missing origin", func(p *nodepolicy.WorkloadCancelParams) { p.OriginatedFrom = "" }, true)
	test("missing run", func(p *nodepolicy.WorkloadCancelParams) { p.RunID = "" }, true)
	test("missing engine", func(p *nodepolicy.WorkloadCancelParams) { p.Engine = "" }, true)
	test("wrong origin", func(p *nodepolicy.WorkloadCancelParams) { p.OriginatedFrom = "other" }, false)
	test("old run with reused workload number", func(p *nodepolicy.WorkloadCancelParams) { p.RunID = "old-run" }, false)
	test("wrong engine", func(p *nodepolicy.WorkloadCancelParams) { p.Engine = "vllm" }, false)
	valid.Engine = "lmstudio"
	rpcOK(t, p, w, nodepolicy.MethodWorkloadCancel, valid)
	if cancelledA != 0 || cancelledB != 1 {
		t.Fatalf("complete key selected wrong controller: a=%d b=%d", cancelledA, cancelledB)
	}
}

func TestAbortLocalIf_IgnoresEndedAttempt(t *testing.T) {
	test := func(name string, end func(*workloadCtl)) {
		t.Run(name, func(t *testing.T) {
			var requestCancelled, attemptCancelled bool
			ctl := newWorkloadCtl(func(string) { requestCancelled = true })
			att := &attemptCtl{
				claim:  &attemptClaim{},
				cancel: func() { attemptCancelled = true },
				nodeID: "self",
			}
			ctl.setAttempt(att)
			end(ctl)

			ctl.abortLocalIf(att)
			if requestCancelled || attemptCancelled || att.claim.abandoned() || ctl.isExcluded("self") {
				t.Fatalf("ended attempt callback acted: request cancelled=%t attempt cancelled=%t abandoned=%t excluded=%t",
					requestCancelled, attemptCancelled, att.claim.abandoned(), ctl.isExcluded("self"))
			}
		})
	}
	test("between attempts", func(ctl *workloadCtl) { ctl.setAttempt(nil) })
	test("finished request", func(ctl *workloadCtl) { ctl.finish() })
}
