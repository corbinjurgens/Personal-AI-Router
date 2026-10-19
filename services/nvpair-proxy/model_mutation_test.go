// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"nvpair-shared/nodepolicy"
)

func newModelMutationProxy(t *testing.T) *Proxy {
	t.Helper()
	h := newStrictHarness(t)
	p := &Proxy{admission: h.a}
	params, err := json.Marshal(nodepolicy.SetModelIdentityParams{
		Engines: map[string]nodepolicy.EngineIdentity{"ollama": testIdentity("m")},
	})
	if err != nil {
		t.Fatalf("encode installed identity: %v", err)
	}
	if err := p.applySetModelIdentity(params); err != nil {
		t.Fatalf("install identity: %v", err)
	}
	return p
}

func TestModelMutationWithdrawalRejectsCachedEnvelopeBeforeDispatch(t *testing.T) {
	p := newModelMutationProxy(t)
	cached := strictRequest()
	body := []byte(strictBody("m"))
	validated, err := p.admission.qualifyStrict("ollama", "m", routeChatCompletions, body, cached.Identity)
	if err != nil {
		t.Fatalf("qualify cached envelope: %v", err)
	}
	ticket, err := p.admission.AdmitStrict(context.Background(), "ollama", "m", cached, validated.generation)
	if err != nil {
		t.Fatalf("admit before withdrawal: %v", err)
	}
	defer ticket.Finish(strictNotDispatched)
	if err := p.applySetModelIdentity(json.RawMessage(`{"engines":{}}`)); err != nil {
		t.Fatalf("accept withdrawal: %v", err)
	}
	if _, err := p.admission.qualifyStrict("ollama", "m", routeChatCompletions, body, cached.Identity); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("revalidate cached envelope: got %v, want unavailable", err)
	}
	if next, err := p.admission.AdmitStrict(context.Background(), "ollama", "m", cached, validated.generation); next != nil || !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("admit cached validation: ticket=%v error=%v, want unavailable", next, err)
	}
	if err := ticket.BeforeStrictDispatch(); !errors.Is(err, errStrictUnavailable) {
		t.Fatalf("dispatch held ticket: got %v, want unavailable", err)
	}
	if p.admission.currentActive() != 0 || len(p.admission.strict.markers) != 0 {
		t.Fatal("withdrawn held ticket retained capacity or a dispatch marker")
	}
}

func TestModelMutationWithdrawalPreservesDispatchedRequest(t *testing.T) {
	p := newModelMutationProxy(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ticket, err := p.admission.admitStrictNow(ctx, "ollama", "m", strictRequest())
	if err != nil {
		t.Fatalf("admit before withdrawal: %v", err)
	}
	defer ticket.Finish(strictQualifiedComplete)
	// serveStrictLocal installs this cancellation function before dispatch.
	ticket.(*strictTicket).base.cancel = cancel
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatalf("dispatch before withdrawal: %v", err)
	}
	if err := p.applySetModelIdentity(json.RawMessage(`{"engines":{}}`)); err != nil {
		t.Fatalf("accept withdrawal: %v", err)
	}
	if err := ctx.Err(); err != nil {
		t.Fatalf("withdrawal cancelled dispatched inference: %v", err)
	}
	if err := ticket.BeforeStrictDispatch(); err != nil {
		t.Fatalf("dispatched ownership changed: %v", err)
	}
	if p.admission.currentActive() != 1 || len(p.admission.strict.markers) != 1 {
		t.Fatal("withdrawal released dispatched capacity or its marker")
	}
	ticket.Finish(strictQualifiedComplete)
	if p.admission.currentActive() != 0 || len(p.admission.strict.markers) != 0 {
		t.Fatal("completed dispatched request retained capacity or its marker")
	}
}
