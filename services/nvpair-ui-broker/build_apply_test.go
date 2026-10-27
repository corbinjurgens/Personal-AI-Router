// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"nvpair-shared/buildapply"
)

func TestBuildApplyLocalAdmission(t *testing.T) {
	fences := buildapply.Fences{
		RuntimeGeneration: buildapply.Known("runtime"), ServiceInstance: buildapply.Known("service"),
		BrokerGeneration: buildapply.Known("broker"), ManagerIncarnation: buildapply.Known("manager"),
		PolicyRevision: buildapply.Known("1"), SettingsRevision: buildapply.Known("2"),
		IntentRevision: buildapply.Known("3"), LocksRevision: buildapply.Known("4"), AvailabilityRevision: buildapply.Known("5"),
	}
	cases := []struct {
		method string
		params any
	}{
		{"build:apply-impact", buildapply.ImpactRequest{}},
		{"build:apply-prepare", buildapply.PrepareRequest{OperationID: "operation", Fences: fences}},
		{"build:apply-finish", buildapply.FinishRequest{OperationID: "operation", Fences: fences}},
	}
	for _, test := range cases {
		t.Run(test.method, func(t *testing.T) {
			raw, err := json.Marshal(test.params)
			if err != nil {
				t.Fatal(err)
			}
			check := func(t *testing.T, params json.RawMessage, wantCode int) {
				t.Helper()
				var wire bytes.Buffer
				b := &Broker{codec: NewCodec(&wire)}
				id := json.RawMessage(`"request"`)
				b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: test.method, Params: params})
				response, err := b.codec.Read()
				if err != nil {
					t.Fatalf("read local reply: %v", err)
				}
				if wantCode != 0 {
					if response.Error == nil || response.Error.Code != wantCode {
						t.Fatalf("refusal = %+v", response)
					}
				} else {
					var result buildapply.Result[json.RawMessage]
					if response.Error != nil {
						t.Fatalf("unexpected RPC error: %+v", response.Error)
					}
					if err := json.Unmarshal(response.Result, &result); err != nil {
						t.Fatal(err)
					}
					if result.Available || result.Refusal == nil || result.Refusal.Code != buildapply.RefusalUnavailable {
						t.Fatalf("result = %+v", result)
					}
				}
			}
			t.Run("unavailable without workers", func(t *testing.T) { check(t, raw, 0) })
			for _, field := range []string{"nodeId", "remoteCaller", "command", "endpoint"} {
				t.Run("rejects "+field, func(t *testing.T) {
					var fields map[string]json.RawMessage
					if err := json.Unmarshal(raw, &fields); err != nil {
						t.Fatal(err)
					}
					fields[field] = json.RawMessage(`"untrusted"`)
					bad, err := json.Marshal(fields)
					if err != nil {
						t.Fatal(err)
					}
					check(t, bad, -32602)
				})
			}
		})
	}
}
