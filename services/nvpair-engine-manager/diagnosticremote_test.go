// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

func remoteDiagnosticPreview() modelconfig.DiagnosticPreview {
	checkpoint := diagnosticRecoveryCheckpointFixture()
	return modelconfig.DiagnosticPreview{
		DiagnosticPreviewRequest: modelconfig.DiagnosticPreviewRequest{NodeID: "host-peer", Engine: checkpoint.Run.Engine, ModelID: checkpoint.Run.ModelID, ProfileKey: checkpoint.Run.ProfileKey},
		DiagnosticIdentity:       checkpoint.Identity.Clone(), PreviewID: "consent", GenerateTokens: 128,
		LowPromptTokens: 440, HighPromptTokens: 880, TimeLimitSeconds: 1200,
		PlannedPerSlotContext: 1024, PlannedSlots: 1,
		DrainEngines: []string{"llamacpp"}, ExpiresAt: "2026-11-07T00:05:00Z",
	}
}

func diagnosticTestJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestRemoteDiagnosticRoutesClassifyConsentAndCleanup(t *testing.T) {
	want := map[string]controlClass{
		diagnosticPath + "preview": controlMutation,
		diagnosticPath + "start":   controlMutation,
		diagnosticPath + "status":  controlRead,
		diagnosticPath + "cancel":  controlMutation,
	}
	for _, route := range (&controlServer{}).routes() {
		class, exists := want[route.path]
		if !exists {
			continue
		}
		if route.class != class || route.method != http.MethodPost || route.handle == nil {
			t.Fatalf("route %+v, expected classified POST %s", route, class)
		}
		delete(want, route.path)
	}
	if len(want) != 0 {
		t.Fatalf("unregistered diagnostic routes: %v", want)
	}
}

func TestRemoteDiagnosticDirectAndReverseUseVerifiedCaller(t *testing.T) {
	test := func(name string, direct bool) {
		t.Run(name, func(t *testing.T) {
			preview := remoteDiagnosticPreview()
			run := diagnosticRecoveryCheckpointFixture().Run
			var calls atomic.Int32
			exec := &Executor{}
			// A configured pause is not the generic management veto. The
			// broker owns its saved pause gate, so the relay must reach it.
			exec.nodeProjection.set(nodeProjection{ManagementBlocked: true, Policy: nodepolicy.Advertisement{PauseMode: nodepolicy.PauseConfigured}})
			exec.diagnosticParent = func(_ context.Context, method, caller string, raw json.RawMessage) (json.RawMessage, error) {
				if caller != "uuid-self" {
					return nil, errors.New("unverified caller")
				}
				_, node, err := normalizeDiagnosticParams(method, raw, "")
				if err != nil || node != "" {
					return nil, errors.New("inbound request kept a forwarding target")
				}
				calls.Add(1)
				if method == "engine:diagnostic-preview" {
					return diagnosticTestJSON(t, preview), nil
				}
				return diagnosticTestJSON(t, run), nil
			}
			f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
				return (&controlServer{exec: exec, mesh: mesh}).mux()
			})
			f.directHealthy.Store(direct)
			grantRemoteManagement(t, f.peerDir, "1", map[string]bool{"uuid-self": true})
			ctx := remoteFixtureContext(t)
			call := func(method string, request any) json.RawMessage {
				t.Helper()
				raw, err := callRemoteDiagnostic(ctx, method, diagnosticTestJSON(t, request), f.m.peers.lookup, f.m.remoteClient)
				if err != nil {
					t.Fatalf("%s: %v", method, err)
				}
				return raw
			}
			var gotPreview modelconfig.DiagnosticPreview
			if err := modelconfig.DecodeStrict(call("engine:diagnostic-preview", preview.DiagnosticPreviewRequest), &gotPreview); err != nil {
				t.Fatal(err)
			}
			if gotPreview.NodeID != "host-peer" || gotPreview.PreviewID != preview.PreviewID {
				t.Fatalf("preview identity: %+v", gotPreview)
			}
			requests := []struct {
				method string
				params any
			}{
				{"engine:diagnostic-start", modelconfig.DiagnosticStartRequest{DiagnosticPreviewRequest: preview.DiagnosticPreviewRequest, PreviewID: preview.PreviewID, Consent: true}},
				{"engine:diagnostic-status", modelconfig.DiagnosticRunRef{NodeID: "host-peer"}},
				{"engine:diagnostic-cancel", modelconfig.DiagnosticRunRef{NodeID: "host-peer", RunID: run.RunID}},
			}
			for _, request := range requests {
				var got diagnosticPeerRun
				if err := modelconfig.DecodeStrict(call(request.method, request.params), &got); err != nil {
					t.Fatal(err)
				}
				if got.NodeID != "host-peer" || got.RunID != run.RunID {
					t.Fatalf("run identity: %+v", got)
				}
			}
			if calls.Load() != 4 || direct && f.reverseCalls.Load() != 0 || !direct && f.reverseCalls.Load() == 0 {
				t.Fatalf("broker=%d direct=%d reverse=%d", calls.Load(), f.directCalls.Load(), f.reverseCalls.Load())
			}
		})
	}
	test("direct pinned transport", true)
	test("reverse pinned transport", false)
}

func TestRemoteDiagnosticPermissionRefusesWithoutBrokerCall(t *testing.T) {
	var calls atomic.Int32
	exec := &Executor{diagnosticParent: func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls.Add(1)
		return json.RawMessage("null"), nil
	}}
	f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler { return (&controlServer{exec: exec, mesh: mesh}).mux() })
	grantRemoteManagement(t, f.peerDir, "1", map[string]bool{"uuid-self": false})
	ctx := remoteFixtureContext(t)
	for _, method := range []string{"preview", "start", "cancel"} {
		_, err := f.client(t, ctx).postDiagnostic(ctx, diagnosticPath+method, json.RawMessage(`{}`))
		var coded *codedError
		if !errors.As(err, &coded) || coded.code != codeRemoteManagementDenied {
			t.Fatalf("%s permission result=%v", method, err)
		}
	}
	if calls.Load() != 0 {
		t.Fatal("denied requests reached target broker")
	}
	if _, err := f.client(t, ctx).postDiagnostic(ctx, diagnosticPath+"status", json.RawMessage(`{}`)); err != nil {
		t.Fatalf("authenticated status read required management grant: %v", err)
	}
}

func TestRemoteDiagnosticResponseRejectsPrivateAuthorityAndDifferentRun(t *testing.T) {
	request := json.RawMessage(`{"nodeId":"host-peer","runId":"interrupted-run"}`)
	run := diagnosticRecoveryCheckpointFixture().Run
	if _, err := projectRemoteDiagnostic("engine:diagnostic-cancel", request, diagnosticTestJSON(t, run), "host-peer"); err != nil {
		t.Fatal(err)
	}
	test := func(name string, response any) {
		t.Run(name, func(t *testing.T) {
			if _, err := projectRemoteDiagnostic("engine:diagnostic-cancel", request, diagnosticTestJSON(t, response), "host-peer"); err == nil {
				t.Fatal("unsafe public response accepted")
			}
		})
	}
	other := run.Clone()
	other.RunID = "later-run"
	test("another run", other)
	test("private status envelope", map[string]any{"run": run, "token": "private"})
	fields := map[string]json.RawMessage{}
	if err := json.Unmarshal(diagnosticTestJSON(t, run), &fields); err != nil {
		t.Fatal(err)
	}
	fields["token"] = json.RawMessage(`"private"`)
	test("private field inside public run", fields)
	fields = map[string]json.RawMessage{}
	if err := json.Unmarshal(diagnosticTestJSON(t, run), &fields); err != nil {
		t.Fatal(err)
	}
	fields["nodeId"] = json.RawMessage(`"other-target"`)
	test("another target", fields)
}

func TestRemoteDiagnosticRequestStrictness(t *testing.T) {
	for _, raw := range []string{
		`{"nodeId":"peer","runId":"one","runId":"two"}`,
		`{"nodeId":"peer","runId":"one","caller":"forged"}`,
		`{"nodeId":"peer","runId":"one","token":"private"}`,
		`{"nodeId":"peer","runId":""}`,
	} {
		if _, _, err := normalizeDiagnosticParams("engine:diagnostic-cancel", []byte(raw), ""); err == nil {
			t.Fatalf("accepted invalid cancel: %s", raw)
		}
	}
	if _, _, err := normalizeDiagnosticParams("engine:diagnostic-status", []byte(`{}`), ""); err != nil {
		t.Fatalf("active-run lookup refused: %v", err)
	}
}

func TestRemoteDiagnosticEmptyStatusIsNullWithoutChangingSharedDecoder(t *testing.T) {
	for _, method := range []string{"engine:diagnostic-status", "engine:diagnostic-cancel"} {
		raw, err := projectRemoteDiagnostic(method, json.RawMessage(`{"runId":"exact"}`), json.RawMessage(" null\n"), "host-peer")
		if err != nil || string(raw) != "null" {
			t.Fatalf("%s empty status=%s error=%v", method, raw, err)
		}
	}
	if _, err := projectRemoteDiagnostic("engine:diagnostic-start", json.RawMessage(`{}`), json.RawMessage("null"), "host-peer"); err == nil {
		t.Fatal("missing start receipt accepted")
	}
}

func TestRemoteDiagnosticRelayCancellationRetiresOnlyWaiter(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var relay diagnosticRelay
	request := make(chan diagnosticRelayRequest, 1)
	cancelled := make(chan string, 1)
	relay.send = func(_ context.Context, method string, value any) error {
		switch method {
		case "engine:diagnostic-request":
			p, ok := value.(diagnosticRelayRequest)
			if !ok {
				return errors.New("unexpected relay request")
			}
			request <- p
		case "engine:diagnostic-request-cancel":
			var p struct {
				ID string `json:"id"`
			}
			if err := json.Unmarshal(diagnosticTestJSON(t, value), &p); err != nil {
				return err
			}
			cancelled <- p.ID
		default:
			return errors.New("wait cancellation issued execution command")
		}
		return nil
	}
	done := make(chan error, 1)
	go func() {
		_, err := relay.call(ctx, "engine:diagnostic-start", "verified-peer", json.RawMessage(`{}`))
		done <- err
	}()
	var accepted diagnosticRelayRequest
	select {
	case accepted = <-request:
	case <-time.After(3 * time.Second):
		t.Fatal("relay not dispatched")
	}
	if accepted.Caller != "verified-peer" || accepted.ID == "" {
		t.Fatalf("lost relay authority: %+v", accepted)
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("wait result=%v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("cancel did not join relay wait")
	}
	select {
	case id := <-cancelled:
		if id != accepted.ID {
			t.Fatal("cancel changed correlation")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("target waiter not cancelled")
	}
	relay.reply(diagnosticTestJSON(t, diagnosticRelayReply{ID: accepted.ID, Result: json.RawMessage(`null`)}))
	relay.mu.Lock()
	defer relay.mu.Unlock()
	if len(relay.pending) != 0 {
		t.Fatal("late reply recreated retired authority")
	}
}

func TestRemoteDiagnosticRelayPreservesTypedRefusal(t *testing.T) {
	var relay diagnosticRelay
	relay.send = func(_ context.Context, method string, value any) error {
		p, ok := value.(diagnosticRelayRequest)
		if method != "engine:diagnostic-request" || !ok {
			return errors.New("unexpected relay")
		}
		relay.reply(diagnosticTestJSON(t, diagnosticRelayReply{ID: p.ID, Error: "Halt", ErrorCode: codeLocalVeto}))
		return nil
	}
	_, err := relay.call(context.Background(), "engine:diagnostic-start", "peer", json.RawMessage(`{}`))
	var refusal *codedError
	if !errors.As(err, &refusal) || refusal.code != codeLocalVeto || !strings.Contains(err.Error(), "Halt") {
		t.Fatalf("refusal=%v", err)
	}
}
