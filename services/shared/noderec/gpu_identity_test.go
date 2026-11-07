// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestDirectoryGPUPhysicalIDRoundTrip(t *testing.T) {
	want := DirectoryNode{HostUUID: "fixture-host", GPUs: []GPUInfo{
		{Name: "GPU B", PhysicalID: "ioreg:b", VramBytes: 200},
		{Name: "GPU A", PhysicalID: "ioreg:a", VramBytes: 100},
		{Name: "unassociated GPU", VramBytes: 300},
	}}
	body, err := json.Marshal(NodeEvent{Node: want})
	if err != nil {
		t.Fatalf("encode directory event: %v", err)
	}
	var got NodeEvent
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode directory event: %v", err)
	}
	if !reflect.DeepEqual(got.Node.GPUs, want.GPUs) {
		t.Fatalf("directory GPUs = %+v, want %+v", got.Node.GPUs, want.GPUs)
	}
	var wire struct {
		Node struct {
			GPUs []map[string]json.RawMessage `json:"gpus"`
		} `json:"node"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode GPU wire fields: %v", err)
	}
	if len(wire.Node.GPUs) != len(want.GPUs) {
		t.Fatalf("GPU wire entries = %d, want %d", len(wire.Node.GPUs), len(want.GPUs))
	}
	var physicalID string
	if err := json.Unmarshal(wire.Node.GPUs[0]["physicalId"], &physicalID); err != nil {
		t.Fatalf("decode physicalId wire field: %v", err)
	}
	if physicalID != "ioreg:b" {
		t.Fatalf("physicalId = %q, want ioreg:b", physicalID)
	}
	if _, present := wire.Node.GPUs[2]["physicalId"]; present {
		t.Fatal("unknown physical association must be omitted")
	}
}
