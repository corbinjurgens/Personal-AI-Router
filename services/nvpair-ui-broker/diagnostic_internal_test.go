// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

func TestDiagnosticInternalMethodsRefusePublicClients(t *testing.T) {
	methods := []string{
		"engine:diagnostic-reserve", "engine:diagnostic-release", "engine:diagnostic-run",
		"engine:diagnostic-run-status", "engine:diagnostic-abort",
		"engine:diagnostic-baseline-done", "engine:record-measured-memory",
		diagnosticObserveMethod,
		"engine:diagnostic-request", "engine:diagnostic-reply", "engine:diagnostic-request-cancel",
		"engine:remote-diagnostic-preview", "engine:remote-diagnostic-start",
		"engine:remote-diagnostic-status", "engine:remote-diagnostic-cancel",
	}
	check := func(method string) {
		t.Run(method, func(t *testing.T) {
			b, out := newPolicyBroker(t)
			manager := attachFakeEngine(t, b)
			params := map[string]string{"engine": "llamacpp", "runId": "public-run", "token": "public-token", "nodeId": "self-node"}
			b.handleMessage(clientRequest(1, method, params))
			response := out.waitFrame(t, "internal diagnostic refusal", isResponse(1))
			if response.Error == nil || response.Error.Code != -32601 || len(response.Result) != 0 {
				t.Fatalf("private method returned a non-refusal: %+v", response)
			}
			notification := clientRequest(2, method, params)
			notification.ID = nil
			b.handleMessage(notification)
			if calls := manager.snapshot(); len(calls) != 0 {
				t.Fatalf("private request or notification reached manager: %+v", calls)
			}
			if frames := out.frames(); len(frames) != 1 {
				t.Fatalf("notification generated a response: %+v", frames)
			}
		})
	}
	for _, method := range methods {
		check(method)
		for _, profile := range engineProxyProfiles {
			check(profile.ComponentName() + ":" + method)
		}
	}
}

func TestDiagnosticInternalMethodsLeavePublicStatusAvailable(t *testing.T) {
	if _, internal := internalClientMethods["engine:diagnostic-status"]; internal {
		t.Fatal("public diagnostic status is classified as worker-internal")
	}
}
