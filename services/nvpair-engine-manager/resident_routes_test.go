// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"

	"nvpair-shared/clustertrust"
)

// These are transport fixtures, not qualification or resident-load evidence.
const residentRoutePolicy = `{"servedModels":[{"engine":"llamacpp","model":"fixture"}],"residentModels":[{"engine":"llamacpp","model":"fixture"}]}`

func TestResidentRoutesRegistration(t *testing.T) {
	s := &controlServer{}
	want := map[string]struct {
		class  controlClass
		family string
	}{
		controlPolicySetPath: {controlMutation, "policy-set"},
		controlPolicyGetPath: {controlRead, "policy-get"},
	}
	for _, route := range s.routes() {
		// The reviewed PG2 registry is exhaustive: adding any internal grant
		// endpoint, under any spelling, must fail this assertion.
		if _, ok := rmExpectedClasses()[route.path]; !ok {
			t.Errorf("unexpected ec route %q", route.path)
		}
		if expected, ok := want[route.path]; ok {
			if route.method != http.MethodPost || route.class != expected.class || route.family != expected.family || route.handle == nil {
				t.Errorf("route %q = method %s class %s family %s, want POST/%s/%s with handler", route.path, route.method, route.class, route.family, expected.class, expected.family)
			}
			delete(want, route.path)
		}
	}
	if len(want) != 0 {
		t.Fatalf("missing policy routes: %v", want)
	}
}

func TestResidentRoutesPermissionAndCaller(t *testing.T) {
	test := func(name string, direct bool) {
		t.Run(name, func(t *testing.T) {
			var sets, reads atomic.Int32
			exec := &Executor{policyParent: func(_ context.Context, method string, params json.RawMessage, caller string) (json.RawMessage, error) {
				if caller != "uuid-self" {
					t.Errorf("policy caller = %q, want authenticated uuid-self", caller)
				}
				var body policyBody
				if err := json.Unmarshal(params, &body); err != nil {
					t.Errorf("decode policy relay: %v", err)
				}
				if body.NodeID != "" {
					t.Errorf("forwarded nodeId = %q, want local target", body.NodeID)
				}
				switch method {
				case "set":
					sets.Add(1)
					var got, want any
					if err := json.Unmarshal(body.Policy, &got); err != nil {
						t.Errorf("decode served/resident policy: %v", err)
					}
					if err := json.Unmarshal([]byte(residentRoutePolicy), &want); err != nil {
						t.Errorf("decode expected policy: %v", err)
					}
					if !reflect.DeepEqual(got, want) || body.ExpectedPersistedRevision != "7" {
						t.Errorf("served/resident edit or CAS changed in transport")
					}
				case "get":
					reads.Add(1)
				default:
					t.Errorf("unexpected policy method %q", method)
				}
				return json.RawMessage(`{"ok":true}`), nil
			}}
			f := newRemoteReverseFixture(t, func(mesh *clustertrust.Mesh) http.Handler {
				return (&controlServer{exec: exec, mesh: mesh}).mux()
			})
			f.directHealthy.Store(direct)
			ctx := remoteFixtureContext(t)
			client := f.client(t, ctx)
			edit := policyBody{NodeID: "another-hop", Policy: json.RawMessage(residentRoutePolicy), ExpectedPersistedRevision: "7"}
			set := func(wantCode int, wantCalls int32) {
				t.Helper()
				_, err := client.postPolicy(ctx, controlPolicySetPath, edit)
				if wantCode == 0 && err != nil {
					t.Fatalf("authorized edit: %v", err)
				}
				if wantCode != 0 && (err == nil || rpcCode(err) != wantCode) {
					t.Fatalf("edit error = %v (code %d), want %d", err, rpcCode(err), wantCode)
				}
				if got := sets.Load(); got != wantCalls {
					t.Fatalf("mutation relay calls = %d, want %d", got, wantCalls)
				}
				if _, err := client.postPolicy(ctx, controlPolicyGetPath, policyBody{NodeID: "another-hop"}); err != nil {
					t.Fatalf("readback unavailable: %v", err)
				}
			}
			grantRemoteManagement(t, f.peerDir, "1", map[string]bool{"uuid-self": false})
			set(codeRemoteManagementDenied, 0)
			grantRemoteManagement(t, f.peerDir, "2", map[string]bool{"uuid-self": true})
			set(0, 1)
			// The same client and connection must re-read the target's grant.
			grantRemoteManagement(t, f.peerDir, "3", map[string]bool{"uuid-self": false})
			set(codeRemoteManagementDenied, 1)
			grantPath := filepath.Join(f.peerDir, clustertrust.RemoteManagementFile)
			if err := os.Remove(grantPath); err != nil {
				t.Fatal(err)
			}
			set(codeRemoteManagementDenied, 1)
			if err := os.WriteFile(grantPath, []byte(`{broken`), 0o600); err != nil {
				t.Fatal(err)
			}
			set(codeRemoteManagementDenied, 1)
			if reads.Load() != 5 {
				t.Fatalf("readback relay calls = %d, want 5", reads.Load())
			}
			if direct {
				if f.directCalls.Load() == 0 || f.reverseCalls.Load() != 0 {
					t.Fatalf("expected direct transport: direct/reverse = %d/%d", f.directCalls.Load(), f.reverseCalls.Load())
				}
			} else if f.reverseCalls.Load() == 0 {
				t.Fatal("expected reverse transport")
			}
		})
	}
	test("direct", true)
	test("reverse", false)
}

func TestResidentRoutesRejectInternalAndWrongMethods(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	f.writePermissions("1", map[string]bool{"uuid-a": true})
	var calls atomic.Int32
	exec := &Executor{policyParent: func(context.Context, string, json.RawMessage, string) (json.RawMessage, error) {
		calls.Add(1)
		return json.RawMessage(`{}`), nil
	}}
	ts := rmTLSServer(t, f.mesh, (&controlServer{exec: exec, mesh: f.mesh}).mux())
	client := f.client("uuid-a")
	for _, method := range []string{http.MethodGet, http.MethodHead, http.MethodPut, http.MethodPatch, http.MethodDelete, http.MethodOptions} {
		for _, path := range []string{controlPolicyGetPath, controlPolicySetPath} {
			status, headers, _ := rmRequest(t, client, ts.URL, method, path, `{}`)
			if status != http.StatusMethodNotAllowed || headers.Get("Allow") != http.MethodPost {
				t.Errorf("%s %s = %d Allow %q, want 405 POST", method, path, status, headers.Get("Allow"))
			}
		}
	}
	for _, method := range []string{"prepare-resident", "dispatch-resident", "finish-resident", "observe-resident", "reconcile-resident", "dispatch-resident-request", "dispatch-resident-reply"} {
		for _, prefix := range []string{"/v1/node/", "/v1/engine/", "/v1/node-policy/", "/node/", "/engine:"} {
			status, _, _ := rmRequest(t, client, ts.URL, http.MethodPost, prefix+method, `{}`)
			if status != http.StatusNotFound {
				t.Errorf("internal route %s%s = %d, want 404", prefix, method, status)
			}
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("invalid routes/methods reached policy relay %d times", calls.Load())
	}
}
