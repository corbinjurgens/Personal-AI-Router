// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

func waitRoutingSnapshotIdle(t *testing.T, p *Proxy) {
	t.Helper()
	deadline := time.NewTimer(3 * time.Second)
	defer deadline.Stop()
	for p.routingSnapshot.Load() {
		select {
		case <-deadline.C:
			t.Fatal("snapshot worker did not release its gate")
		default:
			runtime.Gosched()
		}
	}
}

// The shared rpc fixture expects a synchronous response. Active snapshots own
// a worker, so wait for its response and deferred release before inspecting it.
func routingActiveRPC(t *testing.T, p *Proxy, w *recordingWriter, params any) frame {
	t.Helper()
	rpcSeq++
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	id := json.RawMessage(strconv.Itoa(rpcSeq))
	p.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: nodepolicy.MethodRoutingActive, Params: raw})
	waitRoutingSnapshotIdle(t, p)
	for _, out := range frames(t, w) {
		if out.ID != nil && string(*out.ID) == string(id) && out.Method == "" {
			return out
		}
	}
	t.Fatal("no active snapshot response")
	return frame{}
}

func routingCancelResultOf(t *testing.T, out frame) nodepolicy.RoutingCancelStatus {
	t.Helper()
	if out.Error != nil {
		t.Fatalf("cancel RPC: %+v", out.Error)
	}
	var result nodepolicy.RoutingCancelResult
	if err := json.Unmarshal(out.Result, &result); err != nil {
		t.Fatal(err)
	}
	return result.Status
}

func TestRoutingCancelIdentityAndLifetime(t *testing.T) {
	p, w := policyProxy(t)
	calls := 0
	owner := newRoutingRequest(func(reason string) {
		ownerReason := "cancelled by request"
		if reason != ownerReason {
			t.Errorf("reason=%q", reason)
		}
		calls++
	})
	request, err := p.registerRoutingRequest(owner)
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(request)
	check := func(name string, key nodepolicy.RoutingRequestKey, want nodepolicy.RoutingCancelStatus) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			got := routingCancelResultOf(t, rpc(t, p, w, nodepolicy.MethodRoutingCancel, nodepolicy.RoutingCancelParams{Request: key}))
			if got != want {
				t.Fatalf("status=%s want=%s", got, want)
			}
		})
	}
	other := request.key
	other.OriginNodeID = "other-origin"
	check("wrong origin", other, "not_found")
	other = request.key
	other.ProxyRunID = "old-run"
	check("wrong run", other, "not_found")
	if calls != 0 || owner.cancelled {
		t.Fatal("foreign key changed owner")
	}
	check("live", request.key, "cancelled")
	check("repeat live", request.key, "cancelled")
	if calls != 1 || owner.state != routingEnding || owner.reason != "cancelled by request" {
		t.Fatalf("calls=%d state=%s reason=%q", calls, owner.state, owner.reason)
	}
	owner.finish()
	check("ended but registered", request.key, "not_found")
	p.unregisterRoutingRequest(request)
	check("absent", request.key, "not_found")
}

func TestRoutingCancelStrictParams(t *testing.T) {
	p, w := policyProxy(t)
	for _, raw := range []string{
		`{}`, `null`, `{"request":null,"regenerate":false}`,
		`{"request":{"originNodeId":"test-origin","proxyRunId":"run","requestId":"01"},"regenerate":false}`,
		`{"request":{"originNodeId":"test-origin","proxyRunId":"run","requestId":"1","requestId":"2"},"regenerate":false}`,
		`{"request":{"originNodeId":"test-origin","requestId":"1"},"regenerate":false}`,
		`{"request":{"originNodeId":"test-origin","proxyRunId":"run","requestId":"1"}}`,
		`{"request":{"originNodeId":"test-origin","proxyRunId":"run","requestId":"1"},"regenerate":false,"regenerate":true}`,
		`{"request":{"originNodeId":"test-origin","proxyRunId":"run","requestId":"1"},"regenerate":null}`,
		`{"request":{"originNodeId":"test-origin","proxyRunId":"run","requestId":"1"},"regenerate":false,"extra":0}`,
	} {
		t.Run(raw, func(t *testing.T) {
			out := rpc(t, p, w, nodepolicy.MethodRoutingCancel, json.RawMessage(raw))
			if out.Error == nil || out.Error.Code != -32602 {
				t.Fatalf("invalid params accepted: %+v", out)
			}
		})
	}
	for _, raw := range []string{`null`, `[]`, `{"extra":0}`, `{"extra":0,"extra":1}`} {
		out := routingActiveRPC(t, p, w, json.RawMessage(raw))
		if out.Error == nil || out.Error.Code != -32602 {
			t.Fatalf("active accepted %s: %+v", raw, out)
		}
	}
}

func TestRoutingCancelCommittedConflictAndLegacyTransition(t *testing.T) {
	r := newRoutingRequest(nil)
	calls := 0
	r.onCancel = func(string) {
		// A callback may re-enter the owner: no callback can run under mu.
		r.mu.Lock()
		defer r.mu.Unlock()
		calls++
	}
	a, err := r.newDispatchAttempt(context.Background(), func() { calls++ }, "node-a")
	if err != nil {
		t.Fatal(err)
	}
	r.committed = true
	beforeState, beforeReason := r.state, r.reason
	if got := r.cancelRouting(true); got != "conflict_committed" {
		t.Fatalf("status=%s", got)
	}
	if calls != 0 || r.cancelled || a.aborted || a.redispatch || len(r.excluded) != 0 || r.active != a || r.state != beforeState || r.reason != beforeReason {
		t.Fatal("committed conflict had side effects")
	}
	if !r.cancelQualified(true, nil) || calls != 2 || !r.cancelled {
		t.Fatal("legacy workload regenerate did not cancel after commit")
	}
	if r.cancelRouting(false) != "cancelled" || r.cancelQualified(false, nil) || calls != 2 || r.reason != "cancelled by request" {
		t.Fatal("surfaces did not preserve the first cancellation")
	}
}

func TestRoutingCancelRegenerationBranches(t *testing.T) {
	test := func(name, node string, active, uncertain bool, want string) {
		t.Run(name, func(t *testing.T) {
			r := newRoutingRequest(nil)
			cancelled := 0
			if active {
				if _, err := r.newDispatchAttempt(context.Background(), func() { cancelled++ }, node); err != nil {
					t.Fatal(err)
				}
			} else {
				r.lastNode = node
			}
			r.uncertain = uncertain
			if got := r.cancelRouting(true); string(got) != want {
				t.Fatalf("status=%s", got)
			}
			wantOrdinal := uint64(0)
			if active {
				wantOrdinal = 1
			}
			if r.ordinal != wantOrdinal {
				t.Fatalf("attempt ordinal=%d want=%d", r.ordinal, wantOrdinal)
			}
			if want == "regenerating" {
				if r.cancelled || r.excluded[""] || (node != "" && !r.excluded[node]) {
					t.Fatalf("cancelled=%v exclusions=%v", r.cancelled, r.excluded)
				}
				if active && (cancelled != 1 || !r.active.aborted || !r.active.redispatch) {
					t.Fatal("active regeneration did not abandon attempt")
				}
			} else if !r.cancelled || !r.uncertain {
				t.Fatal("uncertain hold was cleared or not cancelled")
			}
		})
	}
	test("active concrete", "node-a", true, false, "regenerating")
	test("between concrete attempts", "node-a", false, false, "regenerating")
	test("before any attempt", "", false, false, "regenerating")
	test("uncertain hold", "node-a", true, true, "cancelled")
}

func TestRoutingCancelActiveOrderAndZeroFacade(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	w := &recordingWriter{}
	p.codec = NewCodec(w)
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
	if len(p.enabledFacades()) != 0 {
		t.Fatal("fixture has a facade")
	}
	for _, seq := range []int64{8, 9, math.MaxInt64 - 1} {
		p.requestSequence = seq
		owner := newRoutingRequest(nil)
		if _, err := owner.newAttempt(); err != nil {
			t.Fatal(err)
		}
		request, err := p.registerRoutingRequest(owner)
		if err != nil {
			t.Fatal(err)
		}
		defer p.unregisterRoutingRequest(request)
	}
	out := routingActiveRPC(t, p, w, nodepolicy.RoutingActiveParams{})
	if out.Error != nil {
		t.Fatal(out.Error)
	}
	var active nodepolicy.RoutingActiveResult
	if err := json.Unmarshal(out.Result, &active); err != nil {
		t.Fatal(err)
	}
	ids := []string{}
	for _, key := range active.Requests {
		ids = append(ids, key.RequestID)
	}
	if !reflect.DeepEqual(ids, []string{"9", "10", "9223372036854775807"}) {
		t.Fatalf("ids=%v", ids)
	}
	if got := routingCancelResultOf(t, rpc(t, p, w, nodepolicy.MethodRoutingCancel, nodepolicy.RoutingCancelParams{Request: active.Requests[2]})); got != "cancelled" {
		t.Fatalf("status=%s", got)
	}
	bufferAssertTotals(t, p.buffers, registryTestBytes(p), 4, 0)
	for _, request := range p.requests {
		p.unregisterRoutingRequest(request)
	}
	out = routingActiveRPC(t, p, w, nodepolicy.RoutingActiveParams{})
	if out.Error != nil || json.Unmarshal(out.Result, &active) != nil || active.Requests == nil || len(active.Requests) != 0 {
		t.Fatalf("empty snapshot=%s error=%+v", out.Result, out.Error)
	}
}

func TestRoutingCancelAvailableUnderSaturation(t *testing.T) {
	w := &recordingWriter{}
	p := newBoundProxy(NewCodec(w))
	first, err := p.registerRoutingRequest(newRoutingRequest(nil))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(first)
	metadata := registryTestBytes(p)
	lease := bufferMustAcquire(t, p.buffers, p.buffers.capacity-metadata)
	defer lease.close()
	body := &identityBody{}
	f := newFacade(p, anyProfile(t), NewDiscovery(), 0)
	rec := httptest.NewRecorder()
	f.handleHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", body))
	if rec.Code != http.StatusServiceUnavailable || body.reads != 0 || len(w.lines()) != 0 || p.requestSequence != 1 {
		t.Fatalf("overflow status=%d body reads=%d frames=%d sequence=%d", rec.Code, body.reads, len(w.lines()), p.requestSequence)
	}
	out := routingActiveRPC(t, p, w, nodepolicy.RoutingActiveParams{})
	if out.Error == nil || out.Error.Code != -32000 || out.Error.Message != "routing snapshot capacity unavailable; retry" {
		t.Fatalf("saturated snapshot should refuse before copying: %+v", out)
	}
	if got := routingCancelResultOf(t, rpc(t, p, w, nodepolicy.MethodRoutingCancel, nodepolicy.RoutingCancelParams{Request: first.key})); got != "cancelled" {
		t.Fatalf("saturated cancel=%s", got)
	}
	bufferAssertTotals(t, p.buffers, p.buffers.capacity, 3, 0)
	if !first.owner.cancelled {
		t.Fatal("full root hid cancellation")
	}
	id := json.RawMessage(`1`)
	keys, snapshotLease, snapshotErr := p.snapshotRoutingRequests(&id)
	if snapshotErr == nil || keys != nil || snapshotLease != nil {
		t.Fatal("failed snapshot retained an uncharged copy")
	}
	bufferAssertTotals(t, p.buffers, p.buffers.capacity, 3, 0)
	lease.close()
	if out := routingActiveRPC(t, p, w, nodepolicy.RoutingActiveParams{}); out.Error != nil {
		t.Fatalf("snapshot gate not reusable after exhaustion: %+v", out.Error)
	}
	bufferAssertTotals(t, p.buffers, metadata, 2, 0)
}

type routingChargedWrite struct {
	msg   Message
	bytes int
}

type routingChargedWriter struct {
	entered chan routingChargedWrite
	release chan struct{}
	fail    bool
}

func (*routingChargedWriter) Read([]byte) (int, error) { return 0, io.EOF }
func (w *routingChargedWriter) Write(data []byte) (int, error) {
	var msg Message
	if err := json.Unmarshal(data, &msg); err != nil {
		return 0, err
	}
	w.entered <- routingChargedWrite{msg: msg, bytes: len(data)}
	<-w.release
	if w.fail {
		return 0, errors.New("charged snapshot response failed")
	}
	return len(data), nil
}

func TestRoutingCancelSnapshotChargeHeldThroughResponse(t *testing.T) {
	test := func(name string, fail bool) {
		t.Run(name, func(t *testing.T) {
			writer := &routingChargedWriter{entered: make(chan routingChargedWrite, 1), release: make(chan struct{}), fail: fail}
			p := newBoundProxy(NewCodec(writer))
			request, err := p.registerRoutingRequest(newRoutingRequest(nil))
			if err != nil {
				t.Fatal(err)
			}
			metadata := registryTestBytes(p)
			var once sync.Once
			unblock := func() { once.Do(func() { close(writer.release) }) }
			t.Cleanup(func() { unblock(); waitRoutingSnapshotIdle(t, p); p.unregisterRoutingRequest(request) })
			// The actual outer message contains HTML-escaped caller-ID bytes,
			// independently of the inner result RawMessage representation.
			id := json.RawMessage(`"` + strings.Repeat("<", 4096) + `"`)
			p.handleMessage(&Message{ID: &id, Method: nodepolicy.MethodRoutingActive, Params: json.RawMessage(`{}`)})
			var written routingChargedWrite
			select {
			case written = <-writer.entered:
			case <-time.After(3 * time.Second):
				t.Fatal("snapshot response did not reach writer")
			}
			if written.msg.Error != nil {
				t.Fatalf("snapshot refused: %+v", written.msg.Error)
			}
			var result nodepolicy.RoutingActiveResult
			if err := json.Unmarshal(written.msg.Result, &result); err != nil {
				t.Fatal(err)
			}
			if len(result.Requests) != 1 || result.Requests[0] != request.key {
				t.Fatalf("snapshot=%+v", result)
			}
			p.buffers.mu.Lock()
			snapshotBytes := p.buffers.used - metadata
			p.buffers.mu.Unlock()
			minimum := int64(128 + len(request.key.OriginNodeID) + len(request.key.ProxyRunID) + len(request.key.RequestID) + len(written.msg.Result) + written.bytes)
			if snapshotBytes < minimum || !p.routingSnapshot.Load() {
				t.Fatalf("snapshot charge=%d minimum overlapping storage=%d gate=%v", snapshotBytes, minimum, p.routingSnapshot.Load())
			}
			bufferAssertTotals(t, p.buffers, metadata+snapshotBytes, 3, 0)
			// Entry cleanup races with the held response, but its independently
			// owned key copy and both encodings remain charged until I/O ends.
			p.unregisterRoutingRequest(request)
			bufferAssertTotals(t, p.buffers, snapshotBytes, 1, 0)
			unblock()
			waitRoutingSnapshotIdle(t, p)
			bufferAssertTotals(t, p.buffers, 0, 0, 0)
		})
	}
	test("successful write", false)
	test("failed write", true)
}

func TestRoutingCancelOversizedSnapshotIDBoundedRefusal(t *testing.T) {
	w := &recordingWriter{}
	p := newBoundProxy(NewCodec(w))
	id := json.RawMessage(`"` + strings.Repeat("<", (1<<20)/6) + `"`)
	p.handleMessage(&Message{ID: &id, Method: nodepolicy.MethodRoutingActive, Params: json.RawMessage(`{}`)})
	lines := w.lines()
	if len(lines) != 1 || len(lines[0]) >= 1<<20 {
		t.Fatalf("bounded error frame count=%d", len(lines))
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(lines[0], &envelope); err != nil {
		t.Fatal(err)
	}
	var controlError RPCError
	if err := json.Unmarshal(envelope["error"], &controlError); err != nil {
		t.Fatal(err)
	}
	if string(envelope["id"]) != "null" || controlError.Code != -32000 || p.routingSnapshot.Load() {
		t.Fatalf("oversized ID refusal id=%s error=%+v", envelope["id"], controlError)
	}
	bufferAssertTotals(t, p.buffers, 0, 0, 0)
}

func TestRoutingCancelPeerMethodUnavailable(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	r, err := p.registerRoutingRequest(newRoutingRequest(nil))
	if err != nil {
		t.Fatal(err)
	}
	defer p.unregisterRoutingRequest(r)
	f := newFacade(p, anyProfile(t), NewDiscovery(), 0)
	for _, method := range []string{"routing/cancel", "routing/active"} {
		rec := httptest.NewRecorder()
		f.serveIngress(rec, httptest.NewRequest(http.MethodPost, "/"+method, strings.NewReader(`{}`)), "peer")
		// Ingress has no routing RPC adapter; without an engine it cannot
		// answer the private process control call at all.
		if rec.Code != http.StatusServiceUnavailable || r.owner.cancelled || len(p.activeRoutingRequests()) != 1 {
			t.Fatalf("peer %s status=%d cancelled=%v", method, rec.Code, r.owner.cancelled)
		}
	}
}

// Only the first snapshot response can fail. Reads and other responses still
// use the actual duplex codec, so the production read loop can keep serving.
type routingSnapshotTestConn struct {
	net.Conn
	failFirst bool
	failed    chan struct{}
}

func (c *routingSnapshotTestConn) Write(data []byte) (int, error) {
	var msg Message
	if err := json.Unmarshal(data, &msg); err != nil {
		return 0, err
	}
	if c.failFirst && msg.ID != nil && string(*msg.ID) == "1" && msg.Error == nil {
		close(c.failed)
		return 0, errors.New("snapshot response write failed")
	}
	return c.Conn.Write(data)
}

func TestRoutingCancelSnapshotSingleFlight(t *testing.T) {
	test := func(name string, failFirst bool) {
		t.Run(name, func(t *testing.T) {
			server, client := net.Pipe()
			if err := client.SetDeadline(time.Now().Add(3 * time.Second)); err != nil {
				t.Fatal(err)
			}
			transport := &routingSnapshotTestConn{Conn: server, failFirst: failFirst, failed: make(chan struct{})}
			p := newBoundProxy(NewCodec(transport))
			request, err := p.registerRoutingRequest(newRoutingRequest(nil))
			if err != nil {
				t.Fatal(err)
			}
			p.identityMu.Lock()
			locked := true
			loopCtx, cancelLoop := context.WithCancel(context.Background())
			loopDone := make(chan error, 1)
			go func() { loopDone <- p.readLoop(loopCtx) }()
			t.Cleanup(func() {
				if locked {
					p.identityMu.Unlock()
				}
				cancelLoop()
				_ = client.Close()
				_ = server.Close()
				if err := <-loopDone; err != nil {
					t.Errorf("read loop: %v", err)
				}
				waitRoutingSnapshotIdle(t, p)
				p.unregisterRoutingRequest(request)
			})
			send := func(id string) {
				t.Helper()
				rawID := json.RawMessage(id)
				data, err := json.Marshal(&Message{JSONRPC: "2.0", ID: &rawID, Method: nodepolicy.MethodRoutingActive, Params: json.RawMessage(`{}`)})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := client.Write(append(data, '\n')); err != nil {
					t.Fatalf("send request %s: %v", id, err)
				}
			}
			reader := NewCodec(client)
			read := func(id string) *Message {
				t.Helper()
				out, err := reader.Read()
				if err != nil {
					t.Fatalf("read response %s: %v", id, err)
				}
				if out.ID == nil || string(*out.ID) != id {
					t.Fatalf("response ID=%v want=%s", out.ID, id)
				}
				return out
			}
			send("1")
			send("2")
			out := read("2")
			if out.Error == nil || out.Error.Code != -32000 || out.Error.Message != "routing snapshot busy; retry" || !p.routingSnapshot.Load() {
				t.Fatalf("second response before registry release=%+v gate=%v", out, p.routingSnapshot.Load())
			}
			// This response arrived while the first worker could not copy even
			// one key. A synchronous read loop or a contender copy cannot pass.
			p.identityMu.Unlock()
			locked = false
			checkSnapshot := func(out *Message) {
				t.Helper()
				if out.Error != nil {
					t.Fatalf("snapshot: %+v", out.Error)
				}
				var result nodepolicy.RoutingActiveResult
				if err := json.Unmarshal(out.Result, &result); err != nil {
					t.Fatal(err)
				}
				if len(result.Requests) != 1 || result.Requests[0] != request.key {
					t.Fatalf("snapshot=%+v", result)
				}
			}
			if failFirst {
				waitChan(t, transport.failed, "failed first snapshot response")
			} else {
				checkSnapshot(read("1"))
			}
			waitRoutingSnapshotIdle(t, p)
			send("3")
			checkSnapshot(read("3"))
			waitRoutingSnapshotIdle(t, p)
		})
	}
	test("response completes and gate is reused", false)
	test("response fails and gate is reused", true)
}

func TestRoutingCancelWorkloadAndRequestShareQualifiedTransition(t *testing.T) {
	test := func(name string, workloadFirst bool) {
		t.Run(name, func(t *testing.T) {
			p, w := policyProxy(t)
			calls := 0
			owner := newRoutingRequest(func(string) { calls++ })
			request, err := p.registerRoutingRequest(owner)
			if err != nil {
				t.Fatal(err)
			}
			defer p.unregisterRoutingRequest(request)
			key := workloadKey{engine: "ollama", id: "7"}
			if err := p.registerWorkload(request, key); err != nil {
				t.Fatal(err)
			}
			wrong := workloadKey{engine: "ollama", id: "8"}
			if owner.cancelQualified(false, &wrong) || calls != 0 {
				t.Fatal("unqualified workload changed owner")
			}
			workload := func() bool {
				out := rpc(t, p, w, nodepolicy.MethodWorkloadCancel, nodepolicy.WorkloadCancelParams{OriginatedFrom: request.key.OriginNodeID, RunID: request.key.ProxyRunID, Engine: key.engine, WorkloadID: key.id})
				var result nodepolicy.WorkloadCancelResult
				if out.Error != nil {
					t.Fatal(out.Error)
				}
				if err := json.Unmarshal(out.Result, &result); err != nil {
					t.Fatal(err)
				}
				return result.Found
			}
			if workloadFirst && !workload() {
				t.Fatal("first workload cancellation did not transition")
			}
			if got := routingCancelResultOf(t, rpc(t, p, w, nodepolicy.MethodRoutingCancel, nodepolicy.RoutingCancelParams{Request: request.key})); got != "cancelled" {
				t.Fatalf("status=%s", got)
			}
			if workload() || calls != 1 || owner.state != routingEnding || owner.reason != "cancelled by request" {
				t.Fatal("surfaces did not share one qualified transition")
			}
		})
	}
	test("workload first", true)
	test("request first", false)
}

func TestRoutingCancelSnapshotErrorRelease(t *testing.T) {
	p := newBoundProxy(NewCodec(bufferAckWriter(func([]byte) (int, error) { return 0, errors.New("write failed") })))
	id := json.RawMessage(`1`)
	p.handleMessage(&Message{ID: &id, Method: nodepolicy.MethodRoutingActive, Params: json.RawMessage(`{}`)})
	waitRoutingSnapshotIdle(t, p)
	if p.routingSnapshot.Load() {
		t.Fatal("response failure retained gate")
	}
	w := &recordingWriter{}
	p.codec = NewCodec(w)
	if out := routingActiveRPC(t, p, w, nodepolicy.RoutingActiveParams{}); out.Error != nil {
		t.Fatal(out.Error)
	}
}

func TestRoutingCancelConcurrentSurfacesHaveOneTransition(t *testing.T) {
	var callbacks atomic.Int64
	r := newRoutingRequest(func(string) { callbacks.Add(1) })
	key := workloadKey{engine: "ollama", id: "7"}
	r.workload = &key
	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); <-start; r.cancelQualified(false, &key) }()
	go func() { defer wg.Done(); <-start; r.cancelRouting(false) }()
	close(start)
	wg.Wait()
	if callbacks.Load() != 1 || r.state != routingEnding || r.reason != "cancelled by request" || !r.cancelled {
		t.Fatalf("callbacks=%d state=%s reason=%q cancelled=%v", callbacks.Load(), r.state, r.reason, r.cancelled)
	}
}
