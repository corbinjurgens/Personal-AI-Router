// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"
	"time"

	"nvpair-shared/buildinfo"
)

func TestNodeInfoReportsOwnServiceBuild(t *testing.T) {
	previous := serviceBuildID
	t.Cleanup(func() { serviceBuildID = previous })
	const captured = "git:0123456789abcdef0123456789abcdef01234567+dirty"
	serviceBuildID = buildinfo.ID(captured)
	body := buildResponseAt([]GPUInfo{{Name: "fixture GPU", VramBytes: 8192}}, &CPUInfo{Name: "fixture CPU", Cores: 8}, 16384, statsSnapshot{}, "host-a", nil, time.Unix(100, 0))
	var got NodeInfoResponse
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.ServiceBuildID != captured {
		t.Fatalf("build = %q, want captured node-info build %q", got.ServiceBuildID, captured)
	}
	if got.HostUUID != "host-a" || len(got.GPUs) != 1 || got.GPUs[0].Name != "fixture GPU" || got.CPU == nil || got.CPU.Cores != 8 || got.Memory == nil || got.Memory.TotalBytes != 16384 {
		t.Fatalf("identity/hardware changed: %+v", got)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode wire fields: %v", err)
	}
	_, present := wire["serviceBuildId"]
	if !present {
		t.Fatal("captured build missing from wire")
	}
}

func TestNodeInfoOmitsUnknownServiceBuild(t *testing.T) {
	previous := serviceBuildID
	t.Cleanup(func() { serviceBuildID = previous })
	serviceBuildID = ""
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(buildResponse(nil, nil, 0, statsSnapshot{}, "host-a", nil), &fields); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if _, present := fields["serviceBuildId"]; present {
		t.Fatal("unknown build must be absent")
	}
}
