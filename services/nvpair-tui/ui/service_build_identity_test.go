// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"nvpair-shared/buildinfo"
	"nvpair-shared/noderec"
	"nvpair-tui/rpc"
)

func TestServiceBuildWireToDetail(t *testing.T) {
	clean := "git:" + strings.Repeat("a", 40)
	long := "git:" + strings.Repeat("b", 64)
	for _, tc := range []struct {
		name, field, want string
	}{
		{"clean SHA-1", `,"serviceBuildId":"` + clean + `"`, clean},
		{"dirty SHA-1", `,"serviceBuildId":"` + clean + `+dirty"`, clean + "+dirty"},
		{"clean SHA-256", `,"serviceBuildId":"` + long + `"`, long},
		{"dirty SHA-256", `,"serviceBuildId":"` + long + `+dirty"`, long + "+dirty"},
		{"absent", "", ""},
		{"empty", `,"serviceBuildId":""`, ""},
		{"short revision", `,"serviceBuildId":"git:abcd"`, ""},
		{"uppercase revision", `,"serviceBuildId":"git:` + strings.Repeat("A", 40) + `"`, ""},
		{"oversize", `,"serviceBuildId":"git:` + strings.Repeat("a", 500) + `"`, ""},
		{"null", `,"serviceBuildId":null`, ""},
		{"number", `,"serviceBuildId":123`, ""},
		{"boolean", `,"serviceBuildId":true`, ""},
		{"object", `,"serviceBuildId":{"revision":"ignored"}`, ""},
		{"array", `,"serviceBuildId":["ignored"]`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			test := func(t *testing.T, feeds nodeFeeds) {
				t.Helper()
				rows := mergeNodes(feeds)
				if len(rows) != 1 {
					t.Fatalf("rows = %d, want one intact node", len(rows))
				}
				row := rows[0]
				if string(row.serviceBuildID) != tc.want {
					t.Errorf("build = %q, want %q", row.serviceBuildID, tc.want)
				}
				if row.name != "Peer" || row.presence != presenceOnline {
					t.Errorf("optional metadata damaged node: name %q, presence %v", row.name, row.presence)
				}
				d := newNodeDetail(nil, row)
				d.SetSize(180, 40)
				want := tc.want
				if want == "" {
					want = "--"
				}
				if view := d.View(); !strings.Contains(view, "Reported service build: "+want) {
					t.Errorf("detail lacks reported build %q: %s", want, view)
				}
			}
			t.Run("discovery", func(t *testing.T) {
				var nodes []availableNode
				data := `[{"hostUuid":"peer","name":"Peer","ipAddress":"192.0.2.1","port":14318,"models":["model"],"trusted":true` + tc.field + `}]`
				if err := json.Unmarshal([]byte(data), &nodes); err != nil {
					t.Fatal(err)
				}
				test(t, nodeFeeds{discovered: nodes})
				if nodes[0].HostUUID != "peer" || len(nodes[0].Models) != 1 || nodes[0].Models[0] != "model" || !nodes[0].Trusted {
					t.Fatalf("optional metadata damaged discovery: %+v", nodes[0])
				}
			})
			t.Run("manual", func(t *testing.T) {
				var nodes []manualNode
				data := `[{"id":"manual","hostUuid":"peer","name":"Peer","address":"192.0.2.1","node_info_up":true,"cpu":{},"memory":{}` + tc.field + `}]`
				if err := json.Unmarshal([]byte(data), &nodes); err != nil {
					t.Fatal(err)
				}
				test(t, nodeFeeds{manual: nodes})
				if nodes[0].HostUUID != "peer" || nodes[0].CPU == nil || nodes[0].Memory == nil {
					t.Fatalf("optional metadata damaged manual hardware/identity: %+v", nodes[0])
				}
			})
		})
	}
}

func TestServiceBuildDiscoveryUpdatesOpenDetail(t *testing.T) {
	clean := "git:" + strings.Repeat("a", 40)
	v := newNodesView(nil)
	v.SetSize(180, 40)
	push := func(field string) {
		v.Update(NotificationMsg{Msg: &rpc.Message{
			Method: "discovery:nodes-changed",
			Params: []byte(`[{"hostUuid":"peer","name":"Peer","ipAddress":"192.0.2.1","port":14318,"models":["model"],"modelsByEngine":{"ollama":["model"]},"trusted":true` + field + `}]`),
		}})
	}
	push(`,"serviceBuildId":"` + clean + `"`)
	if len(v.all) != 1 {
		t.Fatalf("rows = %d, want one", len(v.all))
	}
	v.detail = newNodeDetail(nil, v.all[0])
	v.detail.SetSize(180, 40)
	v.detail.policyFresh = true
	route := v.detail.telemetryRoute
	push(`,"serviceBuildId":"` + clean + `+dirty"`)
	if v.detail.node.serviceBuildID != buildinfo.ID(clean+"+dirty") || !strings.Contains(v.detail.View(), "Reported service build: "+clean+"+dirty") {
		t.Fatal("build-only discovery change did not reach the open detail")
	}
	if !v.detail.policyFresh || v.detail.telemetryRoute != route {
		t.Fatal("build-only change invalidated policy or telemetry route")
	}
	// An unrelated roster refresh must preserve the discovery-owned build.
	v.Update(clusterMembersMsg{nodes: []clusterNode{{NodeUUID: "peer", State: "joined"}}})
	if v.detail.node.serviceBuildID != buildinfo.ID(clean+"+dirty") {
		t.Fatal("roster refresh lost discovery build")
	}
	for _, field := range []string{"", `,"serviceBuildId":{}`} {
		push(`,"serviceBuildId":"` + clean + `"`)
		push(field)
		row := v.detail.node
		if row.serviceBuildID != "" || !strings.Contains(v.detail.View(), "Reported service build: --") {
			t.Fatalf("successful absent/invalid baseline retained build %q", row.serviceBuildID)
		}
		if row.key != "peer" || row.name != "Peer" || row.membership != membershipMember || len(row.models) != 1 || row.models[0] != "model" || !v.detail.policyFresh {
			t.Fatalf("build clearing damaged identity/models/policy: %+v", row)
		}
	}
}

func TestServiceBuildManualRefreshClearsOnlyBuild(t *testing.T) {
	v := newNodesView(nil)
	v.SetSize(180, 40)
	refresh := func(field string) {
		t.Helper()
		var nodes []manualNode
		if err := json.Unmarshal([]byte(`[{"id":"manual","hostUuid":"peer","name":"Peer","address":"192.0.2.1","node_info_up":true,"tls_enabled":true,"cpu":{},"memory":{}`+field+`}]`), &nodes); err != nil {
			t.Fatal(err)
		}
		v.Update(manualNodesMsg{nodes: nodes})
	}
	clean := "git:" + strings.Repeat("a", 40)
	refresh(`,"serviceBuildId":"` + clean + `"`)
	if len(v.all) != 1 {
		t.Fatalf("rows = %d, want one", len(v.all))
	}
	v.detail = newNodeDetail(nil, v.all[0])
	v.detail.SetSize(180, 40)
	if !strings.Contains(v.detail.View(), "Reported service build: "+clean) {
		t.Fatal("manual build missing from open detail")
	}
	v.Update(manualNodesMsg{err: errors.New("probe failed")})
	if v.detail.node.serviceBuildID != buildinfo.ID(clean) {
		t.Fatal("failed read discarded last report")
	}
	for _, field := range []string{"", `,"serviceBuildId":123`} {
		refresh(`,"serviceBuildId":"` + clean + `"`)
		refresh(field)
		row := v.detail.node
		if row.serviceBuildID != "" || !strings.Contains(v.detail.View(), "Reported service build: --") {
			t.Fatal("successful manual absence/invalid value did not clear build")
		}
		if row.manualID != "manual" || row.name != "Peer" || row.probedTelemetry == nil || row.probedTelemetry.CPU == nil || row.probedTelemetry.Memory == nil {
			t.Fatalf("build clearing damaged manual identity/hardware: %+v", row)
		}
	}
}

func TestServiceBuildDiscoveryBaselineOwnsMergedBuild(t *testing.T) {
	manualBuild := buildinfo.ID("git:" + strings.Repeat("b", 40))
	feeds := nodeFeeds{
		discovered: []availableNode{{HostUUID: "peer", Name: "Peer", IPAddress: "192.0.2.1", Port: 14318}},
		manual:     []manualNode{{ID: "manual", HostUUID: "peer", Address: "192.0.2.1", ServiceBuildID: manualBuild}},
	}
	rows := mergeNodes(feeds)
	if len(rows) != 1 || rows[0].serviceBuildID != "" || rows[0].manualID != "manual" {
		t.Fatalf("discovery absence must clear metadata without splitting identity: %+v", rows)
	}
	feeds.discovered[0].ServiceBuildID = buildinfo.ID("git:" + strings.Repeat("a", 40))
	rows = mergeNodes(feeds)
	if len(rows) != 1 || rows[0].serviceBuildID != buildinfo.ID("git:"+strings.Repeat("a", 40)) {
		t.Fatalf("manual report overrode discovery baseline: %+v", rows)
	}
}

func TestServiceBuildRetainedOfflineIsLabelledCached(t *testing.T) {
	clean := buildinfo.ID("git:" + strings.Repeat("a", 40))
	d := newNodeDetail(nil, nodeRow{key: "peer", name: "Peer", presence: presenceOnline, serviceBuildID: clean})
	d.SetSize(180, 40)
	d.followNode(nodeRow{}, false)
	if view := d.View(); !strings.Contains(view, "Reported service build: "+string(clean)+" (cached; node is not reachable)") {
		t.Fatalf("removed node's retained report lacks cached label: %s", view)
	}
	d.followNode(nodeRow{key: "peer", name: "Peer", presence: presenceOnline}, true)
	if view := d.View(); !strings.Contains(view, "Reported service build: --") || strings.Contains(view, string(clean)) {
		t.Fatalf("new baseline did not replace retained report: %s", view)
	}
}

func TestServiceBuildBrokerVersionLabels(t *testing.T) {
	m := newTestModel(&stubView{title: "Nodes"})
	m.ready, m.brokerVersion = true, "1.2.3"
	if view := m.headerView(); !strings.Contains(view, "broker ready  v1.2.3") || strings.Contains(view, "service ready") {
		t.Fatalf("header mislabels broker version: %s", view)
	}
	v := newServiceView(nil)
	v.SetSize(180, 40)
	v.brokerVersion = "1.2.3"
	if view := v.View(); !strings.Contains(view, "broker v1.2.3") || strings.Contains(view, "service v1.2.3") {
		t.Fatalf("service tab mislabels broker version: %s", view)
	}
	v.pingErr = errors.New("ping failed")
	if view := v.View(); !strings.Contains(view, "broker not responding: ping failed") {
		t.Fatalf("failed broker ping mislabelled: %s", view)
	}
}

func TestServiceBuildSelfDetailFollowsSnapshot(t *testing.T) {
	v := newNodesView(nil)
	v.SetSize(180, 40)
	v.feeds.selfUUID = "self"
	v.feeds.discovered = []availableNode{{HostUUID: "self", Name: "This host"}}
	v.rebuild()
	if len(v.all) != 1 || !v.all[0].self {
		t.Fatal("self fixture did not merge")
	}
	v.detail = newNodeDetail(nil, v.all[0])
	v.detail.SetSize(180, 40)
	build := "git:" + strings.Repeat("c", 40)
	v.Update(NotificationMsg{Msg: &rpc.Message{
		Method: "discovery:nodes-changed",
		Params: []byte(`[{"hostUuid":"self","name":"This host","serviceBuildId":"` + build + `"}]`),
	}})
	if view := v.detail.View(); !strings.Contains(view, "Reported service build: "+build) {
		t.Fatalf("self detail failed to follow node-info build: %s", view)
	}
}

func TestServiceBuildManualOfflineKeepsReportedLabel(t *testing.T) {
	build := buildinfo.ID("git:" + strings.Repeat("d", 40))
	rows := mergeNodes(nodeFeeds{manual: []manualNode{{ID: "manual", Name: "Peer", ServiceBuildID: build}}})
	if len(rows) != 1 || rows[0].presence != presenceOffline {
		t.Fatalf("manual offline fixture did not merge: %+v", rows)
	}
	d := newNodeDetail(nil, rows[0])
	d.SetSize(180, 40)
	if view := d.View(); !strings.Contains(view, "Reported service build: "+string(build)+" (cached; node is not reachable)") {
		t.Fatalf("manual offline report lacks cached label: %s", view)
	}
}

func TestServiceBuildAndActionStatusSurviveManyGPUFrameBudget(t *testing.T) {
	for _, tc := range []struct {
		name   string
		height int
		self   bool
		stale  bool
		editor bool
	}{
		{name: "self short", height: 12, self: true},
		{name: "self next height", height: 13, self: true},
		{name: "peer short", height: 12},
		{name: "peer stale short", height: 12, stale: true},
		{name: "peer stale next height", height: 13, stale: true},
		{name: "self editor short", height: 12, self: true, editor: true},
		{name: "peer stale editor short", height: 12, stale: true, editor: true},
		{name: "peer stale full height", height: 24, stale: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			budget := contentBudget(t, 80, tc.height)
			build := buildinfo.ID("git:" + strings.Repeat("a", 40))
			d := newNodeDetail(nil, nodeRow{
				key: "host", name: "Host", self: tc.self,
				presence: presenceOnline, serviceBuildID: build,
			})
			d.engines = []engineStatus{{Engine: "ollama", Installed: true, Running: true}}
			d.enginesStale = tc.stale
			gpus := make([]noderec.GPUInfo, 8)
			for i := range gpus {
				gpus[i] = noderec.GPUInfo{Name: "GPU", VramBytes: 1 << 30}
			}
			d.telemetryOK = true
			d.telemetry = nodeTelemetry{
				TelemetryValid: true, GPUs: gpus,
				CPU:    &noderec.CPUInfo{Name: "CPU", Cores: 8},
				Memory: &noderec.MemoryInfo{TotalBytes: 1 << 34, UsedBytes: 1 << 33},
			}
			d.SetSize(80, budget)
			d.refreshEngines()
			if tc.editor {
				d.mode = detailInputEnginePort
			}
			d.status.arm("stop engine - press y to confirm")
			view := d.View()
			if got := renderedRows(view); got > budget {
				t.Fatalf("rendered %d rows into budget %d: %s", got, budget, view)
			}
			frame := fitLines(view, budget)
			for _, want := range []string{
				"Reported service build: " + string(build),
				"press y to confirm", "more device(s)", "Engines",
			} {
				if !strings.Contains(frame, want) {
					t.Errorf("frame lost %q: %s", want, frame)
				}
			}
			if tc.editor && !strings.Contains(frame, d.inputLabel()) {
				t.Errorf("frame lost editor: %s", frame)
			}
			if tc.height == 24 && !strings.Contains(frame, "not answering") {
				t.Errorf("frame lost stale engine warning: %s", frame)
			}
		})
	}
}
