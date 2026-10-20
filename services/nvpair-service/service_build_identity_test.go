// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"

	"nvpair-shared/buildinfo"
	"nvpair-shared/jsonrpc"
)

func TestServiceStatusReportsCapturedServiceBuild(t *testing.T) {
	const captured = "git:0123456789abcdef0123456789abcdef01234567+dirty"
	svc := newService(config{version: "independent-version", serviceBuildID: buildinfo.ID(captured)})
	// Dispatch through the real request handler and response codec, without
	// starting a service, a broker, a listener, or a writer goroutine.
	c := &client{out: make(chan []byte, 1)}
	if !svc.handleClientFrame(c, []byte(`{"jsonrpc":"2.0","id":"build-check","method":"service/status"}`)) {
		t.Fatal("status request detached the client")
	}
	var response jsonrpc.Message
	select {
	case frame := <-c.out:
		if err := json.Unmarshal(frame, &response); err != nil {
			t.Fatalf("decode response: %v", err)
		}
	default:
		t.Fatal("status request produced no response")
	}
	if response.JSONRPC != "2.0" || response.ID == nil || string(*response.ID) != `"build-check"` || response.Error != nil {
		t.Fatalf("unexpected RPC envelope: %+v", response)
	}
	var result struct {
		Version string `json:"version"`
		Build   string `json:"serviceBuildId"`
	}
	if err := json.Unmarshal(response.Result, &result); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	if result.Build != captured || result.Version != "independent-version" {
		t.Fatalf("status = %+v, want captured service build and independent version", result)
	}
}

func TestServiceStatusOmitsUnknownBuild(t *testing.T) {
	svc := newService(config{version: "known-version"})
	encoded, err := json.Marshal(svc.status())
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &fields); err != nil {
		t.Fatal(err)
	}
	if _, present := fields["serviceBuildId"]; present {
		t.Fatalf("unknown build was emitted: %s", encoded)
	}
	if string(fields["version"]) != `"known-version"` {
		t.Fatalf("semantic version = %s", fields["version"])
	}
}
