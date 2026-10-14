// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/noderec"
	"nvpair-tui/rpc"

	tea "github.com/charmbracelet/bubbletea"
)

const pairedNodeInfo = `{
	"GPUs":[{"name":"Peer GPU","vram_bytes":8589934592,"vram_used_bytes":1073741824,"utilization_percent":73}],
	"cpu":{"name":"Peer CPU","cores":8,"utilization_percent":42},
	"memory":{"total_bytes":34359738368,"used_bytes":8589934592},
	"telemetryValid":true,"msSince":100,"hostUuid":"physical-host","clusterUuid":"principal"
}`

func nodeInfoRPCFixture(t *testing.T) (*rpc.Client, *rpc.Codec) {
	t.Helper()
	local, remote := net.Pipe()
	if err := remote.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
		t.Fatalf("set fixture deadline: %v", err)
	}
	client := rpc.NewClient(local, local)
	done := make(chan error, 1)
	go func() { done <- client.Run(context.Background()) }()
	t.Cleanup(func() {
		if err := remote.Close(); err != nil {
			t.Errorf("close fixture: %v", err)
		}
		if err := local.Close(); err != nil {
			t.Errorf("close client: %v", err)
		}
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("client read loop did not stop")
		}
	})
	return client, rpc.NewCodec(remote, remote)
}

func assertNodeInfoRequest(t *testing.T, request *rpc.Message) {
	t.Helper()
	if request.Method != "discovery:get-node-info" || request.ID == nil {
		t.Fatalf("request method = %q, id = %v", request.Method, request.ID)
	}
	var params map[string]string
	if err := json.Unmarshal(request.Params, &params); err != nil {
		t.Fatalf("decode request params: %v", err)
	}
	if len(params) != 1 || params["hostUuid"] != "physical-host" {
		t.Fatalf("params = %v, want only hostUuid physical-host", params)
	}
}

func nodeInfoRPCPoll(t *testing.T, broker *rpc.Codec, d *nodeDetail, response *rpc.Message) nodeTelemetryMsg {
	t.Helper()
	cmd := d.telemetryCmd()
	if cmd == nil || !d.telemetryRunning {
		t.Fatal("paired detail did not start its telemetry chain")
	}
	result := make(chan tea.Msg, 1)
	go func() { result <- cmd() }()
	request, err := broker.Read()
	if err != nil {
		t.Fatalf("read request: %v", err)
	}
	assertNodeInfoRequest(t, request)
	response.JSONRPC = "2.0"
	response.ID = request.ID
	if err := broker.Write(response); err != nil {
		t.Fatalf("write response: %v", err)
	}
	select {
	case msg := <-result:
		reading, ok := msg.(nodeTelemetryMsg)
		if !ok {
			t.Fatalf("poll message = %T, want nodeTelemetryMsg", msg)
		}
		return reading
	case <-time.After(3 * time.Second):
		t.Fatal("poll did not return")
		return nodeTelemetryMsg{}
	}
}

func TestNodeInfoRPCPairedDetailWithoutDirectAddress(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "physical-host", membership: membershipMember})
	reading := nodeInfoRPCPoll(t, broker, d, &rpc.Message{Result: json.RawMessage(pairedNodeInfo)})
	if reading.err != nil {
		t.Fatalf("paired poll: %v", reading.err)
	}
	if reading.nodeKey != "physical-host" || reading.gen != d.telemetryGen || reading.address != "" {
		t.Fatalf("poll scope/address = %+v", reading)
	}
	tel := reading.telemetry
	if tel.HostUUID != "physical-host" || !tel.TelemetryValid || len(tel.GPUs) != 1 {
		t.Fatalf("decoded identity/GPU sample = %+v", tel)
	}
	gpu := tel.GPUs[0]
	if gpu.Name != "Peer GPU" || gpu.VramBytes != 8589934592 || gpu.VramUsedBytes != 1073741824 || gpu.UtilizationPercent != 73 {
		t.Fatalf("GPU = %+v", gpu)
	}
	if tel.CPU == nil || tel.CPU.Name != "Peer CPU" || tel.CPU.Cores != 8 || tel.CPU.UtilizationPercent != 42 {
		t.Fatalf("CPU = %+v", tel.CPU)
	}
	if tel.Memory == nil || tel.Memory.TotalBytes != 34359738368 || tel.Memory.UsedBytes != 8589934592 {
		t.Fatalf("memory = %+v", tel.Memory)
	}
	if next, open := d.update(reading); next == nil || !open || !d.telemetryOK {
		t.Fatal("successful paired result did not update and continue detail polling")
	}
}

func TestNodeInfoRPCTrustedDiscoveryUsesHostIdentity(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	rows := mergeNodes(nodeFeeds{discovered: []availableNode{{
		ID: "principal", HostUUID: "physical-host", Name: "peer", Trusted: true,
	}}})
	if len(rows) != 1 || rows[0].membership != membershipMember {
		t.Fatalf("trusted discovery rows = %+v", rows)
	}
	d := newNodeDetail(client, rows[0])
	reading := nodeInfoRPCPoll(t, broker, d, &rpc.Message{Result: json.RawMessage(pairedNodeInfo)})
	if reading.err != nil || reading.nodeKey != "physical-host" {
		t.Fatalf("trusted host poll = %+v", reading)
	}
}

func TestNodeInfoRPCOpenManualDetailAdoptsMergedHostIdentity(t *testing.T) {
	var directCalls atomic.Int32
	started := make(chan struct{}, 1)
	release := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		directCalls.Add(1)
		started <- struct{}{}
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		if _, err := w.Write([]byte(`{"hostUuid":"old-host","telemetryValid":true,"GPUs":[{"name":"Old GPU","utilization_percent":99}]}`)); err != nil {
			t.Errorf("write old direct sample: %v", err)
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	host, port := splitTestServer(t, srv.URL)
	client, broker := nodeInfoRPCFixture(t)
	v := newNodesView(client)
	v.SetSize(100, 30)
	v.feeds.manual = []manualNode{{ID: "one", Address: host, NodeInfoPort: port}}
	v.rebuild()
	v.selectedKey = "manual:one"
	v.restoreSelection()
	v.openDetail()
	d := v.detail
	if d == nil || d.node.key != "manual:one" || d.hostUUID != "" {
		t.Fatalf("opened manual detail = %+v", d)
	}
	gen := d.telemetryGen
	oldCmd := d.telemetryCmd()
	oldResult := make(chan tea.Msg, 1)
	go func() { oldResult <- oldCmd() }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("manual detail did not start direct polling")
	}
	v.feeds.discovered = []availableNode{{
		ID: "principal", HostUUID: "physical-host", Name: "peer", Trusted: true,
		IPAddress: host, Port: port,
	}}
	v.rebuild()
	if v.detail != d || d.node.key != "manual:one" || d.telemetryGen != gen || d.hostUUID != "physical-host" {
		t.Fatalf("detail lost screen scope or failed to adopt host identity: %+v", d)
	}
	if row := v.selectedRow(); row == nil || row.key != "physical-host" || row.manualID != "one" {
		t.Fatalf("selection after manual merge = %+v", row)
	}
	release <- struct{}{}
	var next tea.Cmd
	select {
	case msg := <-oldResult:
		reading, ok := msg.(nodeTelemetryMsg)
		if !ok || reading.err != nil || reading.telemetry.HostUUID != "old-host" {
			t.Fatalf("old direct result = %+v", msg)
		}
		next, _ = d.update(reading)
		if next == nil || d.telemetryOK || d.telemetry.HostUUID == "old-host" {
			t.Fatal("old direct sample overwrote the merged identity or stranded its chain")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("old direct command did not finish")
	}
	tick, ok := next().(nodeTelemetryTickMsg)
	if !ok || tick.nodeKey != "manual:one" || tick.gen != gen {
		t.Fatalf("continued chain tick = %+v", tick)
	}
	cmd, _ := d.update(tick)
	if cmd == nil {
		t.Fatal("old chain tick did not start the paired getter")
	}
	result := make(chan tea.Msg, 1)
	go func() { result <- cmd() }()
	request, err := broker.Read()
	if err != nil {
		t.Fatalf("read merged host request: %v", err)
	}
	assertNodeInfoRequest(t, request)
	if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(pairedNodeInfo)}); err != nil {
		t.Fatalf("write paired sample: %v", err)
	}
	select {
	case msg := <-result:
		reading, ok := msg.(nodeTelemetryMsg)
		if !ok || reading.err != nil || reading.nodeKey != "manual:one" || reading.gen != gen || reading.hostUUID != "physical-host" {
			t.Fatalf("paired result scope = %+v", msg)
		}
		if next, _ := d.update(reading); next == nil || !d.telemetryOK || d.telemetry.HostUUID != "physical-host" {
			t.Fatal("already-open detail did not display paired host telemetry and continue")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("paired command did not finish")
	}
	reading := nodeInfoRPCPoll(t, broker, d, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "node information unavailable"}})
	if reading.err == nil {
		t.Fatal("paired outage reported as successful")
	}
	if got := directCalls.Load(); got != 1 {
		t.Fatalf("direct calls after merge = %d, want only the original manual read", got)
	}
}

func TestNodeInfoRPCPairedWithoutHostIdentityIsUnavailable(t *testing.T) {
	var requests bytes.Buffer
	client := rpc.NewClient(strings.NewReader(""), &requests)
	d := newNodeDetail(client, nodeRow{key: "name:peer", membership: membershipMember})
	cmd := d.telemetryCmd()
	if cmd == nil || !d.telemetryRunning {
		t.Fatal("paired row without a host identity did not start unavailable/recovery chain")
	}
	msg := cmd()
	reading, ok := msg.(nodeTelemetryMsg)
	if !ok || reading.err == nil {
		t.Fatalf("missing host identity result = %+v", msg)
	}
	if requests.Len() != 0 {
		t.Fatalf("missing host identity wrote an RPC request: %q", requests.String())
	}
	if next, _ := d.update(reading); next == nil || d.telemetryOK {
		t.Fatal("missing host identity displayed as available or stopped recovery")
	}
}

func TestNodeInfoRPCOpenManualDetailAdoptsSelfLoopback(t *testing.T) {
	var manualCalls, loopbackCalls atomic.Int32
	started := make(chan struct{}, 1)
	release := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host, _, err := net.SplitHostPort(r.Host)
		if err != nil {
			t.Errorf("split requested host: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if r.Method != http.MethodGet || r.URL.Path != "/v1/node-info" {
			t.Errorf("node-info request = %s %s", r.Method, r.URL.Path)
		}
		body := pairedNodeInfo
		switch host {
		case "localhost":
			manualCalls.Add(1)
			started <- struct{}{}
			select {
			case <-release:
			case <-r.Context().Done():
				return
			}
			body = `{"hostUuid":"old-host","telemetryValid":true,"GPUs":[{"name":"Old GPU","utilization_percent":99}]}`
		case "127.0.0.1":
			loopbackCalls.Add(1)
		default:
			t.Errorf("unexpected direct target %q", host)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if _, err := w.Write([]byte(body)); err != nil {
			t.Errorf("write node-info: %v", err)
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	_, port := splitTestServer(t, srv.URL)
	var requests bytes.Buffer
	client := rpc.NewClient(strings.NewReader(""), &requests)
	v := newNodesView(client)
	v.SetSize(100, 30)
	v.feeds.manual = []manualNode{{ID: "self-entry", Address: "localhost", NodeInfoPort: port}}
	v.rebuild()
	v.selectedKey = "manual:self-entry"
	v.restoreSelection()
	v.openDetail()
	d := v.detail
	if d == nil || d.node.self || d.telemetrySelf || d.hostUUID != "" {
		t.Fatalf("initial manual detail = %+v", d)
	}
	d.engines = []engineStatus{{Engine: "ollama", Port: 11434}}
	d.refreshEngines()
	assertStructure := func(stage string) {
		t.Helper()
		if d.node.self || !d.remote() || d.proxy != nil {
			t.Fatalf("%s: telemetry transition changed the opening engine/proxy structure", stage)
		}
		columns := d.engineTable.Columns()
		want := []string{"ENGINE", "INSTALLED", "RUNNING", "HEALTHY", "ENGINE PORT"}
		if len(columns) != len(want) {
			t.Fatalf("%s: columns = %+v, want the opening remote layout", stage, columns)
		}
		for i, title := range want {
			if columns[i].Title != title {
				t.Fatalf("%s: column %d = %q, want %q", stage, i, columns[i].Title, title)
			}
		}
		rows := d.engineTable.Rows()
		if len(rows) != 1 || len(rows[0]) != 5 || rows[0][4] != "11434" {
			t.Fatalf("%s: engine rows = %+v, want one five-cell row with engine port 11434", stage, rows)
		}
	}
	assertStructure("before merge")
	gen := d.telemetryGen
	oldCmd := d.telemetryCmd()
	oldResult := make(chan tea.Msg, 1)
	go func() { oldResult <- oldCmd() }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("manual hostname read did not start")
	}
	v.feeds.selfUUID = "physical-host"
	v.feeds.manual[0].HostUUID = "physical-host"
	v.feeds.discovered = []availableNode{{
		HostUUID: "physical-host", Name: "local host", Trusted: true,
		IPAddress: "192.0.2.1", Port: port,
	}}
	v.rebuild()
	if v.detail != d || d.node.key != "manual:self-entry" || d.telemetryGen != gen || !d.telemetrySelf || d.hostUUID != "" {
		t.Fatalf("self merge changed scope or failed to adopt self intent: %+v", d)
	}
	d.refreshEngines()
	assertStructure("after merge before resize")
	d.SetSize(120, 40)
	d.refreshEngines()
	assertStructure("after resize")
	if row := v.selectedRow(); row == nil || !row.self || row.key != "physical-host" || row.manualID != "self-entry" {
		t.Fatalf("selected self row = %+v", row)
	}
	release <- struct{}{}
	var next tea.Cmd
	select {
	case msg := <-oldResult:
		reading, ok := msg.(nodeTelemetryMsg)
		if !ok || reading.err != nil || reading.self || reading.hostUUID != "" || reading.telemetry.HostUUID != "old-host" {
			t.Fatalf("old unchecked manual read = %+v", msg)
		}
		next, _ = d.update(reading)
		if next == nil || d.telemetryOK || d.telemetry.HostUUID == "old-host" || !d.telemetryRunning {
			t.Fatal("manual sample overwrote self telemetry or stranded the chain")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("old manual command did not finish")
	}
	if stray, _ := d.update(nodeTelemetryTickMsg{nodeKey: "manual:self-entry", gen: gen + 1}); stray != nil {
		t.Fatal("another generation started an extra self chain")
	}
	for i := 0; i < 2; i++ {
		tick, ok := next().(nodeTelemetryTickMsg)
		if !ok || tick.nodeKey != "manual:self-entry" || tick.gen != gen {
			t.Fatalf("self chain tick = %+v", tick)
		}
		cmd, _ := d.update(tick)
		if cmd == nil {
			t.Fatal("current chain tick did not start loopback read")
		}
		msg := cmd()
		reading, ok := msg.(nodeTelemetryMsg)
		if !ok || reading.err != nil || !reading.self || reading.address != "127.0.0.1" || reading.hostUUID != "" || reading.telemetry.HostUUID != "physical-host" {
			t.Fatalf("self loopback result = %+v", msg)
		}
		next, _ = d.update(reading)
		if next == nil || !d.telemetryOK || d.telemetryGen != gen {
			t.Fatal("self sample did not continue the original chain")
		}
	}
	if requests.Len() != 0 {
		t.Fatalf("self transition issued paired RPC: %q", requests.String())
	}
	if manualCalls.Load() != 1 || loopbackCalls.Load() != 2 {
		t.Fatalf("polls = manual %d, loopback %d; want one retired manual read and two self polls", manualCalls.Load(), loopbackCalls.Load())
	}
}

func TestNodeInfoRPCStaleSampleOmitsGPUUtilization(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "physical-host", membership: membershipMember})
	d.SetSize(100, 30)
	reading := nodeInfoRPCPoll(t, broker, d, &rpc.Message{Result: json.RawMessage(`{
		"hostUuid":"physical-host","telemetryValid":false,
		"GPUs":[{"name":"Stale GPU","utilization_percent":99}]
	}`)})
	if reading.err != nil {
		t.Fatalf("stale sample read failed: %v", reading.err)
	}
	d.update(reading)
	block := d.hardwareBlock()
	if !d.telemetryOK || !strings.Contains(block, "Stale GPU") || !strings.Contains(block, "--%") || strings.Contains(block, "99%") {
		t.Fatalf("stale paired presentation = %q", block)
	}
}

func TestNodeInfoRPCOutageRecoveryNeverDirectFallsBack(t *testing.T) {
	var directCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		directCalls.Add(1)
		if _, err := w.Write([]byte(pairedNodeInfo)); err != nil {
			t.Errorf("write direct response: %v", err)
		}
	}))
	t.Cleanup(srv.Close)
	host, port := splitTestServer(t, srv.URL)
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{
		key: "physical-host", address: host, port: port,
		membership: membershipMember, presence: presenceOnline,
	})
	d.SetSize(100, 30)
	d.status.info("operator message")
	status := d.status.render()
	reading := nodeInfoRPCPoll(t, broker, d, &rpc.Message{Result: json.RawMessage(pairedNodeInfo)})
	d.update(reading)
	for i := 0; i < 2; i++ {
		reading = nodeInfoRPCPoll(t, broker, d, &rpc.Message{Error: &rpc.RPCError{Code: -32000, Message: "node information unavailable"}})
		var rpcErr *rpc.RPCError
		if !errors.As(reading.err, &rpcErr) || rpcErr.Code != -32000 || rpcErr.Message != "node information unavailable" {
			t.Fatalf("unavailable result = %v", reading.err)
		}
		if next, _ := d.update(reading); next == nil {
			t.Fatal("outage stopped the recovery poll chain")
		}
		if d.telemetryOK || !strings.Contains(d.hardwareBlock(), "unavailable") || strings.Contains(d.hardwareBlock(), "73%") {
			t.Fatalf("outage presentation = %q", d.hardwareBlock())
		}
		if d.node.membership != membershipMember || d.node.presence != presenceOnline || d.status.render() != status {
			t.Fatal("telemetry failure changed membership, presence or operator status")
		}
	}
	reading = nodeInfoRPCPoll(t, broker, d, &rpc.Message{Result: json.RawMessage(pairedNodeInfo)})
	d.update(reading)
	if !d.telemetryOK || !strings.Contains(d.hardwareBlock(), "73%") {
		t.Fatalf("recovered telemetry = %q, error = %v", d.hardwareBlock(), reading.err)
	}
	if got := directCalls.Load(); got != 0 {
		t.Fatalf("paired polls sent %d direct HTTP requests", got)
	}
}

func TestNodeInfoRPCRejectsWrongOrMissingHostIdentity(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
	}{
		{"wrong host", `{"hostUuid":"principal","telemetryValid":true}`},
		{"missing host", `{"telemetryValid":true}`},
		{"empty host", `{"hostUuid":"","telemetryValid":true}`},
		{"malformed reply", `{"hostUuid":12}`},
		{"null reply", `null`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, broker := nodeInfoRPCFixture(t)
			d := newNodeDetail(client, nodeRow{key: "physical-host", membership: membershipMember})
			reading := nodeInfoRPCPoll(t, broker, d, &rpc.Message{Result: json.RawMessage(tc.body)})
			if reading.err == nil || reading.telemetry.HostUUID != "" || reading.telemetry.TelemetryValid {
				t.Fatalf("invalid identity accepted: %+v", reading)
			}
			d.update(reading)
			if d.telemetryOK {
				t.Fatal("invalid reply displayed as available telemetry")
			}
		})
	}
}

func TestNodeInfoRPCLateResponseAfterCloseReopenIsIgnored(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	node := nodeRow{key: "physical-host", membership: membershipMember}
	first := newNodeDetail(client, node)
	oldCmd := first.telemetryCmd()
	result := make(chan tea.Msg, 1)
	go func() { result <- oldCmd() }()
	request, err := broker.Read()
	if err != nil {
		t.Fatalf("read old request: %v", err)
	}
	assertNodeInfoRequest(t, request)
	second := newNodeDetail(client, node)
	if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(pairedNodeInfo)}); err != nil {
		t.Fatalf("write late response: %v", err)
	}
	select {
	case msg := <-result:
		if next, _ := second.update(msg); next != nil || second.telemetryOK {
			t.Fatal("old paired reply updated reopened detail or extended a second chain")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("late response did not finish the old command")
	}
	reading := nodeInfoRPCPoll(t, broker, second, &rpc.Message{Result: json.RawMessage(pairedNodeInfo)})
	if next, _ := second.update(reading); next == nil || !second.telemetryOK {
		t.Fatal("reopened detail's own poll did not continue")
	}
}

func TestNodeInfoRPCResponseTimeoutDisownsLateReply(t *testing.T) {
	client, broker := nodeInfoRPCFixture(t)
	d := newNodeDetail(client, nodeRow{key: "physical-host", membership: membershipMember})
	cmd := d.telemetryCmd()
	result := make(chan tea.Msg, 1)
	started := time.Now()
	go func() { result <- cmd() }()
	request, err := broker.Read()
	if err != nil {
		t.Fatalf("read timed request: %v", err)
	}
	assertNodeInfoRequest(t, request)
	select {
	case msg := <-result:
		reading, ok := msg.(nodeTelemetryMsg)
		if !ok || !errors.Is(reading.err, context.DeadlineExceeded) {
			t.Fatalf("deadline result = %+v", msg)
		}
		if elapsed := time.Since(started); elapsed < time.Second || elapsed > 3*time.Second {
			t.Fatalf("response deadline elapsed = %s, want about 1500 ms", elapsed)
		}
		if next, _ := d.update(reading); next == nil || d.telemetryOK {
			t.Fatal("timeout did not display unavailable and continue recovery")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("paired response wait exceeded deadline")
	}
	if err := broker.Write(&rpc.Message{JSONRPC: "2.0", ID: request.ID, Result: json.RawMessage(pairedNodeInfo)}); err != nil {
		t.Fatalf("write expired response: %v", err)
	}
	reading := nodeInfoRPCPoll(t, broker, d, &rpc.Message{Result: json.RawMessage(pairedNodeInfo)})
	if reading.err != nil || reading.telemetry.HostUUID != "physical-host" {
		t.Fatalf("fresh response after disowned reply = %+v", reading)
	}
}

func TestNodeInfoRPCSelfAndUnpairedRetainDirectPolling(t *testing.T) {
	var directCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		directCalls.Add(1)
		if r.Method != http.MethodGet || r.URL.Path != "/v1/node-info" {
			t.Errorf("direct request = %s %s", r.Method, r.URL.Path)
		}
		if _, err := w.Write([]byte(pairedNodeInfo)); err != nil {
			t.Errorf("write direct response: %v", err)
		}
	}))
	t.Cleanup(srv.Close)
	host, port := splitTestServer(t, srv.URL)
	for _, tc := range []struct {
		name string
		node nodeRow
	}{
		{"self member", nodeRow{key: "physical-host", self: true, membership: membershipMember, address: "192.0.2.1", port: port}},
		{"unpaired discovery", nodeRow{key: "physical-host", address: host, port: port}},
		{"foreign discovery", nodeRow{key: "physical-host", membership: membershipForeign, address: host, port: port}},
		{"pending discovery", nodeRow{key: "physical-host", membership: membershipPending, address: host, port: port}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A nil RPC client cannot satisfy a paired getter. Success therefore
			// pins the direct path, including loopback for this machine.
			d := newNodeDetail(nil, tc.node)
			cmd := d.telemetryCmd()
			if cmd == nil {
				t.Fatal("direct polling did not start")
			}
			msg := cmd()
			reading, ok := msg.(nodeTelemetryMsg)
			if !ok || reading.err != nil || reading.telemetry.HostUUID != "physical-host" {
				t.Fatalf("direct result = %+v", msg)
			}
			if reading.address != host {
				t.Fatalf("direct address = %q, want %q", reading.address, host)
			}
		})
	}
	if got := directCalls.Load(); got != 4 {
		t.Fatalf("direct HTTP calls = %d, want 4", got)
	}
}

func TestNodeInfoRPCManualTLSRetainsWorkerTelemetry(t *testing.T) {
	d := newNodeDetail(nil, nodeRow{
		key: "physical-host", membership: membershipMember, nodeInfoTLS: true,
		probedTelemetry: &nodeTelemetry{TelemetryValid: false,
			GPUs: []noderec.GPUInfo{{Name: "Worker GPU", UtilizationPercent: 99}}},
	})
	d.SetSize(100, 30)
	if cmd := d.telemetryCmd(); cmd != nil || d.telemetryRunning {
		t.Fatal("manual TLS started a client poll")
	}
	if !d.telemetryOK || !strings.Contains(d.hardwareBlock(), "Worker GPU") || strings.Contains(d.hardwareBlock(), "99%") {
		t.Fatalf("worker stale presentation = %q", d.hardwareBlock())
	}
	d.followNode(nodeRow{key: "physical-host", membership: membershipMember, nodeInfoTLS: true}, true)
	if d.telemetryOK {
		t.Fatal("missing worker reading displayed as available")
	}
}
