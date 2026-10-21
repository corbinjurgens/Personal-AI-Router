// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

type identityBody struct{ reads int }

func (b *identityBody) Read([]byte) (int, error) { b.reads++; return 0, io.EOF }
func (b *identityBody) Close() error             { return nil }

func TestRoutingIdentityZeroFacadeBinding(t *testing.T) {
	w := &recordingWriter{}
	p := NewProxy(NewCodec(w))
	if p.routingIdentityReady() || len(p.runID) != 32 || len(p.enabledFacades()) != 0 {
		t.Fatal("new zero-facade proxy must have a fresh 128-bit run but no origin binding")
	}
	nonce, err := hex.DecodeString(p.runID)
	if err != nil || len(nonce) != 16 || p.runID != strings.ToLower(p.runID) {
		t.Fatalf("nonce is not 128-bit lowercase hex: %q", p.runID)
	}
	replacement := NewProxy(NewCodec(&recordingWriter{}))
	if replacement.runID == p.runID {
		t.Fatal("replacement reused the previous proxy run")
	}
	pol := nodepolicy.Default()
	f := rpc(t, p, w, nodepolicy.MethodSetPolicy, nodepolicy.SetPolicyParams{OriginNodeID: "origin-a", Policy: pol})
	var ack nodepolicy.PolicyAck
	if f.Error != nil || json.Unmarshal(f.Result, &ack) != nil || ack.OriginNodeID != "origin-a" || ack.RunID != p.runID || ack.Revision != pol.Revision {
		t.Fatalf("identity acknowledgement = %s, error %+v", f.Result, f.Error)
	}
	if !p.routingIdentityReady() {
		t.Fatal("zero-facade policy acknowledgement did not install identity")
	}
	pol.Revision = "1"
	rpcErr(t, p, w, nodepolicy.MethodSetPolicy, nodepolicy.SetPolicyParams{OriginNodeID: "origin-b", Policy: pol})
	if p.originNodeID != "origin-a" || p.admission.currentPolicy().Revision != "0" {
		t.Fatal("origin mutation changed the accepted binding or policy")
	}
	bare, err := json.Marshal(pol)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.applySetPolicy(bare); err == nil {
		t.Fatal("naked-policy IPC accepted")
	}
}

func TestRoutingIdentityMissingBindingHasZeroEffects(t *testing.T) {
	test := func(name string, ingress bool) {
		t.Run(name, func(t *testing.T) {
			w := &recordingWriter{}
			p := NewProxy(NewCodec(w))
			f := newFacade(p, anyProfile(t), NewDiscovery(), 0)
			body := &identityBody{}
			r := httptest.NewRequest(http.MethodPost, "/api/chat", body)
			ctx, cancel := context.WithTimeout(r.Context(), 100*time.Millisecond)
			defer cancel()
			r = r.WithContext(ctx)
			rec := httptest.NewRecorder()
			if ingress {
				f.serveIngress(rec, r, "peer")
			} else {
				f.handleHTTP(rec, r)
			}
			if rec.Code != http.StatusServiceUnavailable || body.reads != 0 || len(w.lines()) != 0 || len(p.requests) != 0 || len(p.workloadRequests) != 0 || p.requestSequence != 0 {
				t.Fatalf("unbound inference status=%d reads=%d notifications=%d requests=%d sequence=%d", rec.Code, body.reads, len(w.lines()), len(p.requests), p.requestSequence)
			}
		})
	}
	test("origin", false)
	test("authenticated ingress", true)
}

func TestRoutingIdentityReadinessWaitsForBinding(t *testing.T) {
	w := &recordingWriter{}
	p := NewProxy(NewCodec(w))
	// Exercise the identity fence independently of admission's revision fence.
	if err := p.admission.activatePolicy(nodepolicy.Default()); err != nil {
		t.Fatal(err)
	}
	rpcOK(t, p, w, nodepolicy.MethodSetModelIdentity, nodepolicy.SetModelIdentityParams{Engines: map[string]nodepolicy.EngineIdentity{}})
	if len(notificationsOf(t, w, nodepolicy.NotifyStrictReadiness)) != 0 {
		t.Fatal("unbound proxy emitted strict readiness")
	}
	rpcOK(t, p, w, nodepolicy.MethodSetPolicy, nodepolicy.SetPolicyParams{OriginNodeID: "origin", Policy: nodepolicy.Default()})
	if len(notificationsOf(t, w, nodepolicy.NotifyStrictReadiness)) != 1 {
		t.Fatal("bound proxy did not emit acknowledged readiness")
	}
}

func TestRoutingIdentityTerminalIngressPreservesRemoteEnvelope(t *testing.T) {
	p, engine := strictHTTPFixture(t)
	r := strictHTTPRequest(t)
	remote := strictRequest()
	remote.Request = nodepolicy.RoutingRequestKey{OriginNodeID: "remote-origin", ProxyRunID: "remote-run", RequestID: "73"}
	encoded, err := json.Marshal(remote)
	if err != nil {
		t.Fatal(err)
	}
	r.Header.Set(nodepolicy.StrictAdmissionHeader, string(encoded))
	rec := httptest.NewRecorder()
	p.soleFacade().serveIngress(rec, r, "remote-peer")
	if rec.Code != http.StatusOK || engine.hits() != 1 {
		t.Fatalf("strict terminal ingress status=%d hits=%d", rec.Code, engine.hits())
	}
	parsed, err := parseStrictAdmission(r.Header)
	if err != nil || parsed == nil || *parsed != remote {
		t.Fatalf("remote request envelope changed: %+v: %v", parsed, err)
	}
	if len(p.activeRoutingRequests()) != 0 || len(p.workloadRequests) != 0 || p.requestSequence != 0 {
		t.Fatal("terminal ingress minted a local origin request or workload")
	}
}

func TestRoutingIdentityProcessCounterAndExhaustion(t *testing.T) {
	p := newBoundProxy(NewCodec(&recordingWriter{}))
	for _, tc := range engineCases(t) {
		f := newFacade(p, tc.profile, NewDiscovery(), 0)
		rec := httptest.NewRecorder()
		f.handleHTTP(rec, httptest.NewRequest(http.MethodPost, tc.inferencePath, strings.NewReader(`{"model":"a","model":"b"}`)))
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("fixture status = %d", rec.Code)
		}
	}
	request, err := p.registerRoutingRequest(newRoutingRequest(func(string) {}))
	if err != nil || request.key.OriginNodeID != "test-origin" || request.key.ProxyRunID != p.runID || request.key.RequestID != "3" {
		t.Fatalf("process request = %+v: %v", request, err)
	}
	p.unregisterRoutingRequest(request)
	p.requestSequence = math.MaxInt64 - 1
	last, err := p.registerRoutingRequest(newRoutingRequest(func(string) {}))
	if err != nil || last.key.RequestID != "9223372036854775807" {
		t.Fatalf("last key = %+v: %v", last, err)
	}
	p.unregisterRoutingRequest(last)
	f := newFacade(p, anyProfile(t), NewDiscovery(), 0)
	for range 2 {
		body := &identityBody{}
		rec := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/api/chat", body)
		ctx, cancel := context.WithTimeout(r.Context(), 100*time.Millisecond)
		f.handleHTTP(rec, r.WithContext(ctx))
		cancel()
		if rec.Code != http.StatusServiceUnavailable || body.reads != 0 || p.requestSequence != math.MaxInt64 || len(p.requests) != 0 {
			t.Fatal("counter exhaustion wrapped or accepted inference")
		}
	}
}
