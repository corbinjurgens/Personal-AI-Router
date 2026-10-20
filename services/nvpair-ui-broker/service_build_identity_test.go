// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"reflect"
	"testing"

	"nvpair-shared/buildinfo"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func TestBrokerDirectoryBuildProjectionAndPush(t *testing.T) {
	const clean = "git:0123456789abcdef0123456789abcdef01234567"
	const dirty = clean + "+dirty"
	policy := nodepolicy.Default().Advertisement()
	node := noderec.DirectoryNode{HostUUID: "host-a", Name: "fixture", IP: "127.0.0.1", Trusted: true, ClusterUUID: "cluster-a", ServiceBuildID: clean, Models: []string{"model-a"}, Policy: &policy}
	store := newDiscoveryStore()
	var pushed []AvailableNode
	pushes := 0
	store.SetOnChange(func() {
		pushes++
		pushed = store.Snapshot()
	})
	for _, want := range []buildinfo.ID{clean, dirty, ""} {
		before := pushes
		node.ServiceBuildID = want
		store.Upsert(directoryToEnriched(node), sourceScanner)
		if pushes != before+1 || len(pushed) != 1 {
			t.Fatalf("build update did not push snapshot: pushes=%d snapshot=%+v", pushes, pushed)
		}
		got := pushed[0]
		if got.ServiceBuildID != want || got.HostUUID != "host-a" || got.Name != "fixture" || !got.Trusted || !got.Clustered || !reflect.DeepEqual(got.Models, []string{"model-a"}) || !reflect.DeepEqual(got.Policy, &policy) {
			t.Fatalf("snapshot = %+v, want build %q with identity/models/policy", got, want)
		}
		data, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("marshal snapshot: %v", err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			t.Fatalf("decode wire: %v", err)
		}
		_, present := fields["serviceBuildId"]
		if present != (want != "") {
			t.Fatalf("wire build presence=%v, want %v", present, want != "")
		}
	}
}

func TestBrokerManualBuildProjection(t *testing.T) {
	const dirty = "git:0123456789abcdef0123456789abcdef01234567+dirty"
	for _, tc := range []struct {
		name, field string
		want        buildinfo.ID
	}{
		{"dirty", `,"serviceBuildId":"` + dirty + `"`, dirty},
		{"absent", "", ""},
		{"invalid optional", `,"serviceBuildId":{}`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var status manualNodeStatus
			if err := json.Unmarshal([]byte(`{"id":"manual-a","hostUuid":"host-a","address":"127.0.0.1","gpus":[{"name":"GPU"}],"ollama_models":["model-a"]`+tc.field+`}`), &status); err != nil {
				t.Fatalf("decode manual status: %v", err)
			}
			enriched := manualToEnriched(status)
			if enriched.ServiceBuildID != tc.want || len(enriched.GPUs) != 1 || enriched.GPUs[0].Name != "GPU" {
				t.Fatalf("enriched = %+v, want build %q and hardware", enriched, tc.want)
			}
			store := newDiscoveryStore()
			store.Upsert(enriched, sourceManual)
			got := store.Snapshot()
			if len(got) != 1 || got[0].ServiceBuildID != tc.want || got[0].HostUUID != "host-a" || got[0].IPAddress != "127.0.0.1" || !reflect.DeepEqual(got[0].Models, []string{"model-a"}) {
				t.Fatalf("manual snapshot = %+v, want build %q and identity/models", got, tc.want)
			}
		})
	}
}
