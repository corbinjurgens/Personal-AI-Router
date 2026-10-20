// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"nvpair-shared/buildinfo"
)

const manualCleanBuild = "git:0123456789abcdef0123456789abcdef01234567"
const manualDirtyBuild = manualCleanBuild + "+dirty"

func TestManualBuildOnlyUpdateAndOptionalClearing(t *testing.T) {
	for _, tc := range []struct {
		name, field string
		want        buildinfo.ID
	}{
		{"dirty build", `,"serviceBuildId":"` + manualDirtyBuild + `"`, manualDirtyBuild},
		{"absent", "", ""},
		{"malformed", `,"serviceBuildId":"git:short"`, ""},
		{"oversize", `,"serviceBuildId":"` + strings.Repeat("a", 500) + `"`, ""},
		{"null", `,"serviceBuildId":null`, ""},
		{"number", `,"serviceBuildId":4`, ""},
		{"object", `,"serviceBuildId":{}`, ""},
		{"array", `,"serviceBuildId":[]`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, rw, rt := newTestManager()
			entry := ManualEntry{Address: "fixture.local", Name: "fixture"}
			id := nodeID(entry)
			m.nodes[id] = &trackedNode{entry: entry}
			base := `{"hostUuid":"host-a","GPUs":[{"name":"GPU"}],"cpu":{"cores":8},"memory":{"total_bytes":16384}`
			body := base + `,"serviceBuildId":"` + manualCleanBuild + `"}`
			rt.set(http.MethodGet, "fixture.local:14318", "/v1/node-info", func(*http.Request) (*http.Response, error) {
				return httpJSON(http.StatusOK, body)
			})
			m.probeNode(entry)
			if m.nodes[id].status.ServiceBuildID != manualCleanBuild {
				t.Fatalf("initial build = %q", m.nodes[id].status.ServiceBuildID)
			}
			for len(rw.frames) > 0 {
				<-rw.frames
			}
			body = base + tc.field + `}`
			m.probeNode(entry)
			var update *ManualNodeStatus
			for len(rw.frames) > 0 {
				frame := <-rw.frames
				var msg Message
				if err := json.Unmarshal(frame, &msg); err != nil {
					t.Fatalf("decode notification: %v", err)
				}
				if msg.Method == "node/updated" {
					var status ManualNodeStatus
					if err := json.Unmarshal(msg.Params, &status); err != nil {
						t.Fatalf("decode updated status: %v", err)
					}
					update = &status
				}
			}
			if update == nil {
				t.Fatal("build-only change did not emit node/updated")
			}
			if update.ServiceBuildID != tc.want || update.HostUUID != "host-a" || !update.NodeInfoUp || len(update.GPUs) != 1 || update.GPUs[0].Name != "GPU" || update.CPU == nil || update.CPU.Cores != 8 || update.Memory == nil || update.Memory.TotalBytes != 16384 {
				t.Fatalf("update = %+v, want build %q with intact identity/hardware", update, tc.want)
			}
			if m.nodes[id].status.ServiceBuildID != tc.want {
				t.Fatalf("stored build = %q, want %q", m.nodes[id].status.ServiceBuildID, tc.want)
			}
		})
	}
}

func TestManualBuildFetchFailureKeepsExistingFailureSemantics(t *testing.T) {
	m, _, rt := newTestManager()
	entry := ManualEntry{Address: "fixture.local", Name: "fixture"}
	id := nodeID(entry)
	m.nodes[id] = &trackedNode{entry: entry, status: ManualNodeStatus{HostUUID: "host-a", ServiceBuildID: manualCleanBuild, NodeInfoUp: true, GPUs: []GPUInfo{{Name: "GPU"}}}}
	rt.set(http.MethodGet, "fixture.local:14318", "/v1/node-info", func(*http.Request) (*http.Response, error) {
		return httpJSON(http.StatusServiceUnavailable, `{}`)
	})
	m.probeNode(entry)
	got := m.nodes[id].status
	if got.HostUUID != "host-a" || got.NodeInfoUp || got.ServiceBuildID != "" || len(got.GPUs) != 0 {
		t.Fatalf("failed manual probe = %+v, want retained identity and absent current inventory", got)
	}
}
