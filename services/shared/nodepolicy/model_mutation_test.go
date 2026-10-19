// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"testing"
)

const testMutationRequestID = "mutation-1"

func TestModelMutationRequestRoundTrips(t *testing.T) {
	raw, err := json.Marshal(ModelMutationRequest{RequestID: testMutationRequestID, Engine: "ollama"})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	var got ModelMutationRequest
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode request %s: %v", raw, err)
	}
	if got.RequestID != testMutationRequestID || got.Engine != "ollama" {
		t.Fatalf("request = %+v", got)
	}
	if MethodModelMutationRequest != "engine:model-mutation-request" || MethodModelMutationReply != "engine:model-mutation-reply" {
		t.Fatalf("methods = %q, %q", MethodModelMutationRequest, MethodModelMutationReply)
	}
}

func TestModelMutationRequestRejectsInvalidParams(t *testing.T) {
	test := func(name, params string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			var req ModelMutationRequest
			if err := json.Unmarshal([]byte(params), &req); err == nil {
				t.Fatalf("decode %s succeeded: %+v", params, req)
			}
		})
	}
	test("empty request id", `{"requestId":"","engine":"ollama"}`)
	test("request id with a space", `{"requestId":"mutation 1","engine":"ollama"}`)
	test("missing engine", `{"requestId":"mutation-1"}`)
	test("engine with a path separator", `{"requestId":"mutation-1","engine":"../ollama"}`)
	test("unknown field", `{"requestId":"mutation-1","engine":"ollama","model":"m:latest"}`)
	test("null request id", `{"requestId":null,"engine":"ollama"}`)
}

func TestModelMutationReplyRoundTrips(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reply ModelMutationReply
	}{
		{name: "accepted", reply: ModelMutationReply{RequestID: testMutationRequestID, Engine: "ollama", OK: true}},
		{name: "unavailable", reply: ModelMutationReply{RequestID: testMutationRequestID, Engine: "ollama", Error: "the worker retired", Code: ModelMutationUnavailable}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(tc.reply)
			if err != nil {
				t.Fatalf("marshal reply: %v", err)
			}
			var got ModelMutationReply
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("decode reply %s: %v", raw, err)
			}
			if got.RequestID != tc.reply.RequestID || got.Engine != tc.reply.Engine || got.OK != tc.reply.OK || got.Error != tc.reply.Error || got.Code != tc.reply.Code {
				t.Fatalf("reply = %+v, want %+v", got, tc.reply)
			}
		})
	}
}

func TestModelMutationReplyRejectsInvalidResults(t *testing.T) {
	test := func(name, params string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			var reply ModelMutationReply
			if err := json.Unmarshal([]byte(params), &reply); err == nil {
				t.Fatalf("decode %s succeeded: %+v", params, reply)
			}
		})
	}
	test("missing ok", `{"requestId":"mutation-1","engine":"ollama","error":"","code":0}`)
	test("null ok", `{"requestId":"mutation-1","engine":"ollama","ok":null,"error":"","code":0}`)
	test("missing error", `{"requestId":"mutation-1","engine":"ollama","ok":false,"code":1}`)
	test("missing code", `{"requestId":"mutation-1","engine":"ollama","ok":true,"error":""}`)
	test("success with an error", `{"requestId":"mutation-1","engine":"ollama","ok":true,"error":"refused","code":0}`)
	test("success with a code", `{"requestId":"mutation-1","engine":"ollama","ok":true,"error":"","code":1}`)
	test("failure without an error", `{"requestId":"mutation-1","engine":"ollama","ok":false,"error":"","code":1}`)
	test("blank failure error", `{"requestId":"mutation-1","engine":"ollama","ok":false,"error":"  ","code":1}`)
	test("unknown field", `{"requestId":"mutation-1","engine":"ollama","ok":true,"error":"","code":0,"result":"ok"}`)
}
