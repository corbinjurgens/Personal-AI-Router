// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestBoundEngineRequestStrictContract(t *testing.T) {
	valid := `{"requestId":"wait-1","proxyRunId":"run-1","policyRevision":"3","engine":"ollama","model":"a/b:tag","budgetMs":90000}`
	var got PrepareBoundEngineRequest
	if err := json.Unmarshal([]byte(valid), &got); err != nil {
		t.Fatal(err)
	}
	if got.Model != "a/b:tag" || got.BudgetMs != MaxBoundEngineBudgetMs || got.PeerID != "" {
		t.Fatalf("request=%+v", got)
	}
	test := func(name, raw string) {
		t.Run(name, func(t *testing.T) {
			var in PrepareBoundEngineRequest
			if err := json.Unmarshal([]byte(raw), &in); err == nil {
				t.Fatalf("accepted %s", raw)
			}
		})
	}
	test("zero budget", strings.Replace(valid, `90000`, `0`, 1))
	test("excess budget", strings.Replace(valid, `90000`, `90001`, 1))
	test("fractional budget", strings.Replace(valid, `90000`, `1.5`, 1))
	test("partial peer", strings.TrimSuffix(valid, "}")+`,"peerId":"peer"}`)
	test("partial incarnation", strings.TrimSuffix(valid, "}")+`,"peerIncarnation":"admission"}`)
	test("empty peer pair cannot substitute local authority", strings.TrimSuffix(valid, "}")+`,"peerId":"","peerIncarnation":""}`)
	test("duplicate", strings.TrimSuffix(valid, "}")+`,"budgetMs":1}`)
	test("case variant", strings.Replace(valid, `policyRevision`, `PolicyRevision`, 1))
	test("unknown body", strings.TrimSuffix(valid, "}")+`,"body":"secret"}`)
	test("null", strings.Replace(valid, `"run-1"`, `null`, 1))
	test("missing", strings.Replace(valid, `"proxyRunId":"run-1",`, "", 1))
	test("invalid raw UTF8", strings.Replace(valid, `a/b:tag`, string([]byte{255}), 1))
	peer := strings.TrimSuffix(valid, "}") + `,"peerId":"peer","peerIncarnation":"admission"}`
	if err := json.Unmarshal([]byte(peer), &got); err != nil || got.PeerID != "peer" {
		t.Fatalf("peer=%+v %v", got, err)
	}
}

func TestBoundEngineReplyStrictContract(t *testing.T) {
	for _, reply := range []BoundEnginePrepared{
		{RequestID: "wait", ProxyRunID: "run", Success: true},
		{RequestID: "wait", ProxyRunID: "run", Reason: "management-denied", Message: "Permission denied.", Code: -32012, SwitchRequestID: "binding:run:1"},
	} {
		raw, err := json.Marshal(reply)
		if err != nil {
			t.Fatal(err)
		}
		var got BoundEnginePrepared
		if err := json.Unmarshal(raw, &got); err != nil || got != reply {
			t.Fatalf("reply=%+v %v", got, err)
		}
	}
	test := func(name, raw string) {
		t.Run(name, func(t *testing.T) {
			var reply BoundEnginePrepared
			if err := json.Unmarshal([]byte(raw), &reply); err == nil {
				t.Fatal("malformed reply accepted")
			}
		})
	}
	test("missing success", `{"requestId":"wait","proxyRunId":"run"}`)
	test("success with error", `{"requestId":"wait","proxyRunId":"run","success":true,"code":-32012}`)
	test("unknown reason", `{"requestId":"wait","proxyRunId":"run","success":false,"reason":"native-secret","message":"refused","code":-32000}`)
	test("unbounded message", `{"requestId":"wait","proxyRunId":"run","success":false,"reason":"unavailable","message":"`+strings.Repeat("x", MaxBoundEngineMessageBytes+1)+`","code":-32000}`)
	test("duplicate", `{"requestId":"wait","proxyRunId":"run","success":true,"success":false}`)
}

func TestBoundEngineActivationAndCancellation(t *testing.T) {
	p := ActivateBoundEngineRequest{OperationID: "operation", Engine: "ollama", Model: "model", PolicyRevision: "1", ManagerIncarnation: "manager", Deadline: time.Now().UTC().Add(time.Minute), PeerID: "peer", PeerIncarnation: "admission"}
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var got ActivateBoundEngineRequest
	if err := json.Unmarshal(raw, &got); err != nil || got != p {
		t.Fatalf("activation=%+v %v", got, err)
	}
	if err := json.Unmarshal([]byte(strings.TrimSuffix(string(raw), "}")+`,"sourceEngine":"other"}`), &got); err == nil {
		t.Fatal("arbitrary lifecycle command accepted")
	}
	var cancel CancelBoundEngineRequest
	if err := json.Unmarshal([]byte(`{"requestId":"operation","proxyRunId":"run"}`), &cancel); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"requestId":"operation","proxyRunId":"run","engine":"ollama"}`), &cancel); err == nil {
		t.Fatal("cancellation accepted an engine command")
	}
}

func TestBoundEngineHTTPBudget(t *testing.T) {
	test := func(name string, values []string, want time.Duration) {
		t.Run(name, func(t *testing.T) {
			if got := EnginePrepareBudget(values); got != want {
				t.Fatalf("budget=%v want=%v", got, want)
			}
		})
	}
	test("absent uses total bound", nil, 90*time.Second)
	test("remaining milliseconds", []string{"123"}, 123*time.Millisecond)
	test("excess clamped", []string{"90001"}, 90*time.Second)
	test("largest integer clamped before duration conversion", []string{"18446744073709551615"}, 90*time.Second)
	test("overflow refuses", []string{"18446744073709551616"}, 0)
	test("empty refuses", []string{""}, 0)
	test("negative refuses", []string{"-1"}, 0)
	test("fraction refuses", []string{"1.5"}, 0)
	test("repeated refuses", []string{"1", "90000"}, 0)
	test("comma joined refuses", []string{"1,90000"}, 0)
	test("whitespace refuses", []string{" 90000"}, 0)
	test("noncanonical refuses", []string{"00090000"}, 0)
	test("exhausted refuses", []string{"0"}, 0)
}
