// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/applog"
	"nvpair-shared/noderec"
)

func init() { manualMemoryClock = func() (uint64, error) { return 1000000000, nil } }

func TestMemoryTelemetryManualFetchAndNotificationAge(t *testing.T) {
	old := manualMemoryClock
	t.Cleanup(func() { manualMemoryClock = old })
	boot := uint64(1000000000)
	manualMemoryClock = func() (uint64, error) { return boot, nil }
	m, rw, rt := newTestManager()
	f := noderec.MemoryFact{Bytes: "0", Source: "fixture", SampledAt: time.Now().UTC(), Valid: true}
	info := NodeInfoResponse{HostUUID: "peer", MemoryObservation: &noderec.MemoryObservation{RunID: "run", Domains: []noderec.MemoryDomain{{ID: "ram", Kind: "system", Total: f, Free: f}}}}
	body, err := json.Marshal(info)
	if err != nil {
		t.Fatal(err)
	}
	rt.set(http.MethodGet, "node.local:14318", "/v1/node-info", func(*http.Request) (*http.Response, error) {
		boot += 6000000000
		return httpJSON(http.StatusOK, string(body))
	})
	entry := ManualEntry{Name: "memory", Address: "node.local"}
	tn := &trackedNode{entry: entry}
	m.nodes[nodeID(entry)] = tn
	m.probeNode(tn)
	msg := readCaptureUntil(t, rw, methodIs("node/updated"))
	status := decodeParams[ManualNodeStatus](t, msg)
	if status.MemoryObservation == nil || status.MemoryObservation.Domains[0].Free.AgeMS != 6000 || status.MemoryHop == nil || status.MemoryHop.BootNS != "7000000000" {
		t.Fatalf("manual memory: %+v", status)
	}
}

type captureRW struct {
	frames chan []byte
}

func newCaptureRW() *captureRW {
	return &captureRW{frames: make(chan []byte, 64)}
}

func (rw *captureRW) Read(_ []byte) (int, error) {
	return 0, io.EOF
}

func (rw *captureRW) Write(p []byte) (int, error) {
	cp := append([]byte(nil), p...)
	rw.frames <- cp
	return len(p), nil
}

type fakeRoundTripper struct {
	mu       sync.RWMutex
	handlers map[string]func(*http.Request) (*http.Response, error)
}

func newFakeRoundTripper() *fakeRoundTripper {
	return &fakeRoundTripper{handlers: make(map[string]func(*http.Request) (*http.Response, error))}
}

func (rt *fakeRoundTripper) set(method, host, path string, fn func(*http.Request) (*http.Response, error)) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.handlers[method+" "+host+path] = fn
}

func (rt *fakeRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	key := req.Method + " " + req.URL.Host + req.URL.Path
	rt.mu.RLock()
	fn := rt.handlers[key]
	rt.mu.RUnlock()
	if fn == nil {
		return nil, errors.New("unexpected request: " + key)
	}
	return fn(req)
}

func httpJSON(status int, body string) (*http.Response, error) {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}, nil
}

func newTestManager() (*Manager, *captureRW, *fakeRoundTripper) {
	rw := newCaptureRW()
	m, err := NewManager(NewCodec(rw), tlsClientOptions{}, nil)
	if err != nil {
		panic(err)
	}
	rt := newFakeRoundTripper()
	m.client = &http.Client{Transport: rt, Timeout: time.Second}
	m.tlsClient = &http.Client{Transport: rt, Timeout: time.Second}
	return m, rw, rt
}

func configureHealthyNode(rt *fakeRoundTripper, addr string, info NodeInfoResponse) {
	infoHost := net.JoinHostPort(addr, "14318")
	rt.set(http.MethodGet, infoHost, "/v1/node-info", func(*http.Request) (*http.Response, error) {
		data, _ := json.Marshal(info)
		return httpJSON(http.StatusOK, string(data))
	})
}

func requestMessage(id int, method string, params any) *Message {
	idData, _ := json.Marshal(id)
	idRaw := json.RawMessage(idData)
	paramsRaw, _ := json.Marshal(params)
	return &Message{JSONRPC: "2.0", ID: &idRaw, Method: method, Params: paramsRaw}
}

func requestMessageRaw(id int, method string, params json.RawMessage) *Message {
	idData, _ := json.Marshal(id)
	idRaw := json.RawMessage(idData)
	return &Message{JSONRPC: "2.0", ID: &idRaw, Method: method, Params: params}
}

func notificationMessage(method string, params any) *Message {
	paramsRaw, _ := json.Marshal(params)
	return &Message{JSONRPC: "2.0", Method: method, Params: paramsRaw}
}

func readCaptureFrame(t *testing.T, rw *captureRW) Message {
	t.Helper()
	select {
	case data := <-rw.frames:
		var msg Message
		if err := json.Unmarshal(data, &msg); err != nil {
			t.Fatalf("decode frame %q: %v", data, err)
		}
		return msg
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for frame")
		return Message{}
	}
}

func readCaptureUntil(t *testing.T, rw *captureRW, match func(Message) bool) Message {
	t.Helper()
	deadline := time.After(2 * time.Second)
	for {
		select {
		case data := <-rw.frames:
			var msg Message
			if err := json.Unmarshal(data, &msg); err != nil {
				t.Fatalf("decode frame %q: %v", data, err)
			}
			if match(msg) {
				return msg
			}
		case <-deadline:
			t.Fatal("timed out waiting for matching frame")
			return Message{}
		}
	}
}

func assertNoCaptureMethod(t *testing.T, rw *captureRW, method string) {
	t.Helper()
	timer := time.NewTimer(150 * time.Millisecond)
	defer timer.Stop()
	for {
		select {
		case data := <-rw.frames:
			var msg Message
			if err := json.Unmarshal(data, &msg); err != nil {
				t.Fatalf("decode frame %q: %v", data, err)
			}
			if msg.Method == method {
				t.Fatalf("unexpected %s notification: %+v", method, msg)
			}
		case <-timer.C:
			return
		}
	}
}

func decodeResult[T any](t *testing.T, msg Message) T {
	t.Helper()
	if msg.Error != nil {
		t.Fatalf("unexpected RPC error: %+v", msg.Error)
	}
	var result T
	if err := json.Unmarshal(msg.Result, &result); err != nil {
		t.Fatalf("decode result %q: %v", msg.Result, err)
	}
	return result
}

func decodeParams[T any](t *testing.T, msg Message) T {
	t.Helper()
	var result T
	if err := json.Unmarshal(msg.Params, &result); err != nil {
		t.Fatalf("decode params %q: %v", msg.Params, err)
	}
	return result
}

func responseWithID(id int) func(Message) bool {
	return func(msg Message) bool {
		if msg.ID == nil {
			return false
		}
		var got int
		return json.Unmarshal(*msg.ID, &got) == nil && got == id
	}
}

func methodIs(method string) func(Message) bool {
	return func(msg Message) bool { return msg.Method == method }
}

func sampleInfo() NodeInfoResponse {
	return NodeInfoResponse{
		GPUs: []GPUInfo{{
			Name:               "RTX 6000",
			VramBytes:          48 << 30,
			VramUsedBytes:      12 << 30,
			UtilizationPercent: 42,
		}},
		CPU:            &CPUInfo{Name: "Threadripper", Cores: 64, UtilizationPercent: 7},
		Memory:         &MemoryInfo{TotalBytes: 128 << 30, UsedBytes: 32 << 30},
		TelemetryValid: true,
		MSSince:        137,
	}
}

func TestNodeID(t *testing.T) {
	if got := nodeID(ManualEntry{Name: "workstation", Address: "10.0.0.5"}); got != "manual:name:workstation" {
		t.Fatalf("named nodeID = %q", got)
	}
	if got := nodeID(ManualEntry{Address: "10.0.0.5"}); got != "manual:10.0.0.5" {
		t.Fatalf("unnamed nodeID = %q", got)
	}
}

func TestManualEntryValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		entry ManualEntry
	}{
		{"IPv4", ManualEntry{Address: "192.0.2.1"}},
		{"IPv6", ManualEntry{Address: "2001:db8::1"}},
		{"hostname", ManualEntry{Address: "lab.tailnet.ts.net."}},
		{"Unicode display name", ManualEntry{Address: "node.local", Name: "作業 PC"}},
		{"named reference at byte bound", ManualEntry{Address: "node.local", Name: strings.Repeat("n", manualNodeReferenceMaxBytes-len("manual:name:"))}},
		{"maximum TLS port", ManualEntry{Address: "node.local", TLSPort: 65535, MTLS: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.entry.validate(); err != nil {
				t.Fatalf("valid entry rejected: %v", err)
			}
		})
	}
}

func TestManualEntryRejectedBeforeEffects(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
	}{
		{"URL", `{"address":"http://node.local"}`},
		{"scoped IPv6", `{"address":"fe80::1%en0"}`},
		{"host and port", `{"address":"node.local:11434"}`},
		{"path", `{"address":"node.local/path"}`},
		{"credentials", `{"address":"user@node.local"}`},
		{"query", `{"address":"node.local?x"}`},
		{"whitespace", `{"address":" node.local"}`},
		{"empty DNS label", `{"address":"node..local"}`},
		{"leading hyphen", `{"address":"-node.local"}`},
		{"invalid UTF8", "{\"address\":\"node.local\",\"name\":\"bad\xff\"}"},
		{"unpaired surrogate", `{"address":"node.local","name":"\ud800"}`},
		{"format character", `{"address":"node.local","name":"bad\u200b"}`},
		{"control character", `{"address":"node.local","name":"bad\n"}`},
		{"negative port", `{"address":"node.local","tls_port":-1}`},
		{"over-bound port", `{"address":"node.local","tls_port":65536}`},
		{"mTLS without TLS", `{"address":"node.local","mtls":true}`},
		{"named reference over byte bound", `{"address":"node.local","name":"` + strings.Repeat("n", manualNodeReferenceMaxBytes-len("manual:name:")+1) + `"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, rw, _ := newTestManager()
			m.statePath = filepath.Join(t.TempDir(), stateFileName)
			m.handleMessage(requestMessageRaw(1, "node/add", json.RawMessage(tc.raw)))
			response := readCaptureFrame(t, rw)
			if response.Error == nil || response.Error.Code != -32602 || len(m.listNodes()) != 0 {
				t.Fatalf("invalid entry reached state/probes: response=%+v", response)
			}
			if _, err := os.Stat(m.statePath); !os.IsNotExist(err) {
				t.Fatalf("invalid entry reached persistence: %v", err)
			}
		})
	}
}

func TestManualNamesCannotClaimOtherKeyNamespaces(t *testing.T) {
	for _, name := range []string{"self", "peer-uuid", "manual:192.0.2.1", "192.0.2.1"} {
		t.Run(name, func(t *testing.T) {
			id := nodeID(ManualEntry{Address: "node.local", Name: name})
			if id == name || id == nodeID(ManualEntry{Address: "192.0.2.1"}) || !strings.HasPrefix(id, "manual:name:") {
				t.Fatalf("name escaped its namespace: %q", id)
			}
		})
	}
}

func TestManualNameUnicodeEscapes(t *testing.T) {
	for _, tc := range []struct {
		name, raw, want string
	}{
		{"paired surrogate", `"\ud83d\ude00"`, "😀"},
		{"literal replacement rune", `"�"`, "�"},
		{"escaped replacement rune", `"\ufffd"`, "�"},
		{"escaped backslash", `"\\ud800"`, `\ud800`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var entry ManualEntry
			if err := json.Unmarshal([]byte(`{"address":"node.local","name":`+tc.raw+`}`), &entry); err != nil {
				t.Fatal(err)
			}
			if entry.Name != tc.want {
				t.Fatalf("name=%q, want %q", entry.Name, tc.want)
			}
		})
	}
}

func TestNodeAddRespondsWithInitialStatusThenDiscoversProbeResult(t *testing.T) {
	m, rw, rt := newTestManager()
	configureHealthyNode(rt, "node.local", sampleInfo())

	m.handleMessage(requestMessage(1, "node/add", ManualEntry{Address: "node.local", Name: "lab"}))

	resp := readCaptureUntil(t, rw, responseWithID(1))
	initial := decodeResult[ManualNodeStatus](t, resp)
	if initial.ID != "manual:name:lab" || initial.Address != "node.local" {
		t.Fatalf("initial status = %+v", initial)
	}
	if initial.NodeInfoPort != 14318 {
		t.Fatalf("default ports not set: %+v", initial)
	}
	if initial.NodeInfoUp {
		t.Fatalf("initial status should be unprobed: %+v", initial)
	}

	discovered := readCaptureUntil(t, rw, methodIs("node/discovered"))
	status := decodeParams[ManualNodeStatus](t, discovered)
	if !status.NodeInfoUp {
		t.Fatalf("discovered status did not include healthy services: %+v", status)
	}
	if len(status.GPUs) != 1 || status.GPUs[0].Name != "RTX 6000" {
		t.Fatalf("gpus = %#v", status.GPUs)
	}
	if status.CPU == nil || status.CPU.Cores != 64 {
		t.Fatalf("cpu = %#v", status.CPU)
	}
	if status.Memory == nil || status.Memory.UsedBytes != 32<<30 {
		t.Fatalf("memory = %#v", status.Memory)
	}
	if !status.TelemetryValid || status.MSSince != 137 {
		t.Fatalf("telemetry = valid:%v age:%d, want true/137", status.TelemetryValid, status.MSSince)
	}
}

func TestNodeAddValidationErrors(t *testing.T) {
	m, rw, _ := newTestManager()

	m.handleMessage(requestMessageRaw(1, "node/add", json.RawMessage(`"bad"`)))
	resp := readCaptureFrame(t, rw)
	if resp.Error == nil || resp.Error.Code != -32602 {
		t.Fatalf("malformed params error = %+v", resp.Error)
	}

	m.handleMessage(requestMessage(2, "node/add", ManualEntry{}))
	resp = readCaptureFrame(t, rw)
	if resp.Error == nil || resp.Error.Code != -32602 {
		t.Fatalf("missing address error = %+v", resp.Error)
	}

	if got := m.listNodes(); len(got) != 0 {
		t.Fatalf("validation errors added nodes: %#v", got)
	}
}

func TestNodesListReturnsCurrentStatuses(t *testing.T) {
	m, rw, rt := newTestManager()
	configureHealthyNode(rt, "node.local", sampleInfo())

	m.handleMessage(requestMessage(1, "node/add", ManualEntry{Address: "node.local"}))
	_ = readCaptureUntil(t, rw, methodIs("node/discovered"))

	m.handleMessage(requestMessage(2, "nodes/list", nil))
	resp := readCaptureUntil(t, rw, responseWithID(2))
	var result struct {
		Nodes []ManualNodeStatus `json:"nodes"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		t.Fatalf("decode nodes/list: %v", err)
	}
	if len(result.Nodes) != 1 {
		t.Fatalf("nodes/list length = %d", len(result.Nodes))
	}
	if !result.Nodes[0].NodeInfoUp {
		t.Fatalf("nodes/list status = %+v", result.Nodes[0])
	}
}

func TestNodeRemoveReturnsRemovedAndNotifies(t *testing.T) {
	m, rw, _ := newTestManager()
	m.nodes["manual:name:lab"] = &trackedNode{
		entry:  ManualEntry{Name: "lab", Address: "node.local"},
		status: ManualNodeStatus{ID: "manual:name:lab", Address: "node.local", NodeInfoPort: 14318},
	}

	m.handleMessage(requestMessage(1, "node/remove", map[string]string{"id": "manual:name:lab"}))
	removed := readCaptureFrame(t, rw)
	if removed.Method != "node/removed" {
		t.Fatalf("first remove frame = %+v", removed)
	}
	status := decodeParams[ManualNodeStatus](t, removed)
	if status.ID != "manual:name:lab" {
		t.Fatalf("removed params = %+v", status)
	}
	resp := readCaptureUntil(t, rw, responseWithID(1))
	result := decodeResult[map[string]bool](t, resp)
	if !result["removed"] {
		t.Fatalf("removed result = %#v", result)
	}
	if got := m.listNodes(); len(got) != 0 {
		t.Fatalf("node still listed after removal: %#v", got)
	}

	m.handleMessage(requestMessage(2, "node/remove", map[string]string{"id": "manual:name:lab"}))
	resp = readCaptureUntil(t, rw, responseWithID(2))
	result = decodeResult[map[string]bool](t, resp)
	if result["removed"] {
		t.Fatalf("second removal result = %#v", result)
	}
	assertNoCaptureMethod(t, rw, "node/removed")
}

func TestAddThenRemoveBeforeInitialProbeDoesNotRediscover(t *testing.T) {
	m, rw, rt := newTestManager()
	started := make(chan struct{})
	release := make(chan struct{})
	rt.set(http.MethodGet, net.JoinHostPort("node.local", "14318"), "/v1/node-info", func(*http.Request) (*http.Response, error) {
		close(started)
		<-release
		return httpJSON(http.StatusOK, `{"GPUs":[]}`)
	})

	status := m.addNode(ManualEntry{Address: "node.local", Name: "lab"})
	if status.ID != "manual:name:lab" {
		t.Fatalf("add status = %+v", status)
	}

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("initial probe did not start")
	}

	if !m.removeNode("manual:name:lab") {
		t.Fatal("removeNode returned false")
	}
	_ = readCaptureUntil(t, rw, methodIs("node/removed"))
	close(release)

	assertNoCaptureMethod(t, rw, "node/discovered")
	if got := m.listNodes(); len(got) != 0 {
		t.Fatalf("node rediscovered in state: %#v", got)
	}
}

func TestReplacedManualNodeDropsOldProbeResultAndDiscovery(t *testing.T) {
	m, rw, rt := newTestManager()
	oldEntry := ManualEntry{Name: "lab", Address: "old.local"}
	oldInfo := sampleInfo()
	oldInfo.HostUUID = "old-host"
	oldInfo.ServiceBuildID = manualDirtyBuild
	configureHealthyNode(rt, oldEntry.Address, oldInfo)
	oldProbeStarted := make(chan struct{})
	releaseOldProbe := make(chan struct{})
	oldProbeDone := make(chan struct{})
	rt.set(http.MethodGet, net.JoinHostPort(oldEntry.Address, "14318"), "/v1/node-info", func(*http.Request) (*http.Response, error) {
		close(oldProbeStarted)
		<-releaseOldProbe
		data, err := json.Marshal(oldInfo)
		if err != nil {
			return nil, err
		}
		return httpJSON(http.StatusOK, string(data))
	})
	oldNode := &trackedNode{entry: oldEntry, status: ManualNodeStatus{ID: "manual:name:lab", Address: oldEntry.Address, NodeInfoPort: 14318}}
	m.nodes["manual:name:lab"] = oldNode
	go func() {
		m.probeAndDiscover(oldNode)
		close(oldProbeDone)
	}()
	select {
	case <-oldProbeStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("old probe did not start")
	}

	newEntry := ManualEntry{Name: "lab", Address: "new.local"}
	newInfo := sampleInfo()
	newInfo.HostUUID = "new-host"
	newInfo.ServiceBuildID = manualCleanBuild
	configureHealthyNode(rt, newEntry.Address, newInfo)
	m.addNode(newEntry)
	newDiscovery := decodeParams[ManualNodeStatus](t, readCaptureUntil(t, rw, methodIs("node/discovered")))
	if newDiscovery.Address != newEntry.Address || newDiscovery.HostUUID != "new-host" || newDiscovery.ServiceBuildID != manualCleanBuild {
		t.Fatalf("replacement discovery = %+v", newDiscovery)
	}
	for len(rw.frames) > 0 {
		<-rw.frames
	}

	close(releaseOldProbe)
	select {
	case <-oldProbeDone:
	case <-time.After(2 * time.Second):
		t.Fatal("old probe did not finish")
	}

	m.mu.RLock()
	got := m.nodes["manual:name:lab"].status
	m.mu.RUnlock()
	if got.Address != newEntry.Address || got.HostUUID != "new-host" || got.ServiceBuildID != manualCleanBuild {
		t.Fatalf("replacement status overwritten by old probe: %+v", got)
	}
	for len(rw.frames) > 0 {
		var msg Message
		if err := json.Unmarshal(<-rw.frames, &msg); err != nil {
			t.Fatalf("decode post-replacement notification: %v", err)
		}
		if msg.Method == "node/updated" || msg.Method == "node/discovered" {
			t.Fatalf("stale probe emitted %s notification: %+v", msg.Method, msg)
		}
	}
}

func TestProbeNodeEmitsUpdatedOnStateChange(t *testing.T) {
	m, rw, rt := newTestManager()
	entry := ManualEntry{Name: "lab", Address: "node.local"}
	m.nodes[nodeID(entry)] = &trackedNode{entry: entry}
	info := sampleInfo()
	configureHealthyNode(rt, "node.local", info)
	m.probeNode(m.nodes[nodeID(entry)])
	first := decodeParams[ManualNodeStatus](t, readCaptureUntil(t, rw, methodIs("node/updated")))
	if first.CPU == nil || first.CPU.Cores != 64 {
		t.Fatalf("first update CPU = %+v", first.CPU)
	}
	info.CPU = &CPUInfo{Name: "changed", Cores: 32}
	configureHealthyNode(rt, "node.local", info)
	m.probeNode(m.nodes[nodeID(entry)])
	second := decodeParams[ManualNodeStatus](t, readCaptureUntil(t, rw, methodIs("node/updated")))
	if second.CPU == nil || second.CPU.Cores != 32 || second.CPU.Name != "changed" {
		t.Fatalf("second update CPU = %+v", second.CPU)
	}
}

func TestProbeNodeNoUpdateWhenStable(t *testing.T) {
	m, rw, rt := newTestManager()
	entry := ManualEntry{Name: "lab", Address: "node.local"}
	m.nodes["manual:name:lab"] = &trackedNode{entry: entry, status: ManualNodeStatus{ID: "manual:name:lab", Address: "node.local", NodeInfoPort: 14318}}
	configureHealthyNode(rt, "node.local", sampleInfo())

	m.probeNode(m.nodes[nodeID(entry)])
	_ = readCaptureUntil(t, rw, methodIs("node/updated"))
	m.probeNode(m.nodes[nodeID(entry)])

	assertNoCaptureMethod(t, rw, "node/updated")
}

func TestProbeFailuresClearAvailability(t *testing.T) {
	m, rw, rt := newTestManager()
	entry := ManualEntry{Name: "lab", Address: "node.local"}
	m.nodes["manual:name:lab"] = &trackedNode{
		entry: entry,
		status: ManualNodeStatus{
			ID:           "manual:name:lab",
			Address:      "node.local",
			NodeInfoUp:   true,
			NodeInfoPort: 14318,
			GPUs:         sampleInfo().GPUs,
			CPU:          sampleInfo().CPU,
			Memory:       sampleInfo().Memory,
		},
	}
	rt.set(http.MethodGet, net.JoinHostPort("node.local", "14318"), "/v1/node-info", func(*http.Request) (*http.Response, error) {
		return httpJSON(http.StatusOK, `{not json`)
	})

	m.probeNode(m.nodes[nodeID(entry)])
	updated := decodeParams[ManualNodeStatus](t, readCaptureUntil(t, rw, methodIs("node/updated")))
	if updated.NodeInfoUp {
		t.Fatalf("services should be down: %+v", updated)
	}
	if len(updated.GPUs) != 0 || updated.CPU != nil || updated.Memory != nil {
		t.Fatalf("failed probe retained stale fields: %+v", updated)
	}
}

// TestProbeFailurePreservesHostUUID: a node-info blip
// must not blank the learned HostUUID, or the
// broker would rekey the live node UUID -> manual-id -> UUID.
func TestProbeFailurePreservesHostUUID(t *testing.T) {
	m, rw, rt := newTestManager()
	entry := ManualEntry{Name: "lab", Address: "node.local"}
	m.nodes["manual:name:lab"] = &trackedNode{entry: entry, status: ManualNodeStatus{ID: "manual:name:lab", Address: "node.local", NodeInfoPort: 14318}}

	info := sampleInfo()
	info.HostUUID = "node-uuid"
	configureHealthyNode(rt, "node.local", info)

	// First probe learns the UUID.
	m.probeNode(m.nodes[nodeID(entry)])
	first := decodeParams[ManualNodeStatus](t, readCaptureUntil(t, rw, methodIs("node/updated")))
	if first.HostUUID != "node-uuid" {
		t.Fatalf("first probe HostUUID = %q, want node-uuid", first.HostUUID)
	}

	// node-info goes down: the UUID must be preserved.
	rt.set(http.MethodGet, net.JoinHostPort("node.local", "14318"), "/v1/node-info", func(*http.Request) (*http.Response, error) {
		return nil, errors.New("node-info down")
	})
	m.probeNode(m.nodes[nodeID(entry)])
	down := decodeParams[ManualNodeStatus](t, readCaptureUntil(t, rw, methodIs("node/updated")))
	if down.NodeInfoUp {
		t.Fatal("node-info should be down")
	}
	if down.HostUUID != "node-uuid" {
		t.Fatalf("HostUUID dropped on node-info failure: %q", down.HostUUID)
	}

	// node-info recovers: still the same UUID (no flap).
	configureHealthyNode(rt, "node.local", info)
	m.probeNode(m.nodes[nodeID(entry)])
	up := decodeParams[ManualNodeStatus](t, readCaptureUntil(t, rw, methodIs("node/updated")))
	if up.HostUUID != "node-uuid" {
		t.Fatalf("HostUUID after recovery = %q, want node-uuid", up.HostUUID)
	}
}

func TestCPUAndMemoryNilAwareEquality(t *testing.T) {
	if !cpuEqual(nil, nil) || !memoryEqual(nil, nil) {
		t.Fatal("nil values should compare equal")
	}
	if cpuEqual(nil, &CPUInfo{}) || memoryEqual(nil, &MemoryInfo{}) {
		t.Fatal("nil and non-nil values should differ")
	}
	if !cpuEqual(&CPUInfo{Name: "cpu", Cores: 8}, &CPUInfo{Name: "cpu", Cores: 8}) {
		t.Fatal("equal CPU values differed")
	}
	if cpuEqual(&CPUInfo{Name: "cpu", Cores: 8}, &CPUInfo{Name: "cpu", Cores: 16}) {
		t.Fatal("different CPU values compared equal")
	}
	if !memoryEqual(&MemoryInfo{TotalBytes: 10, UsedBytes: 5}, &MemoryInfo{TotalBytes: 10, UsedBytes: 5}) {
		t.Fatal("equal memory values differed")
	}
	if memoryEqual(&MemoryInfo{TotalBytes: 10, UsedBytes: 5}, &MemoryInfo{TotalBytes: 10, UsedBytes: 6}) {
		t.Fatal("different memory values compared equal")
	}
}

func TestUnknownMethodReturnsMethodNotFound(t *testing.T) {
	m, rw, _ := newTestManager()
	m.handleMessage(requestMessage(1, "bogus", nil))
	resp := readCaptureFrame(t, rw)
	if resp.Error == nil || resp.Error.Code != -32601 {
		t.Fatalf("unknown method error = %+v", resp.Error)
	}
}

func TestLogSetLevelRequest(t *testing.T) {
	m, rw, _ := newTestManager()
	m.handleMessage(requestMessage(1, applog.SetLevelMethod, applog.SetLevelParams{Level: "debug"}))
	resp := readCaptureFrame(t, rw)
	result := decodeResult[map[string]string](t, resp)
	if result["level"] != "debug" {
		t.Fatalf("log/set-level result = %#v", result)
	}

	m.handleMessage(requestMessage(2, applog.SetLevelMethod, applog.SetLevelParams{Level: "not-a-level"}))
	resp = readCaptureFrame(t, rw)
	if resp.Error == nil || resp.Error.Code != -32602 {
		t.Fatalf("invalid log/set-level error = %+v", resp.Error)
	}
}

func TestShutdownRequestCancelsRun(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	mgr, err := NewManager(NewCodec(server), tlsClientOptions{}, nil)
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	done := make(chan error, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		done <- mgr.Run(ctx)
	}()

	reader := bufio.NewReader(client)
	ready := readPipeFrame(t, client, reader)
	if ready.Method != "ready" {
		t.Fatalf("first frame = %+v", ready)
	}

	writePipeRequest(t, client, 7, "shutdown", nil)
	resp := readPipeFrame(t, client, reader)
	if !responseWithID(7)(resp) || resp.Error != nil {
		t.Fatalf("shutdown response = %+v", resp)
	}

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not return after shutdown")
	}
}

func TestNotificationIsIgnored(t *testing.T) {
	m, rw, _ := newTestManager()
	m.handleMessage(notificationMessage("node/add", ManualEntry{Address: "node.local"}))
	if got := m.listNodes(); len(got) != 0 {
		t.Fatalf("notification mutated state: %#v", got)
	}
	assertNoCaptureMethod(t, rw, "")
}

func readPipeFrame(t *testing.T, conn net.Conn, reader *bufio.Reader) Message {
	t.Helper()
	if err := conn.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil {
		t.Fatalf("set read deadline: %v", err)
	}
	line, err := reader.ReadBytes('\n')
	if err != nil {
		t.Fatalf("read pipe frame: %v", err)
	}
	var msg Message
	if err := json.Unmarshal(line, &msg); err != nil {
		t.Fatalf("decode pipe frame %q: %v", line, err)
	}
	return msg
}

func writePipeRequest(t *testing.T, conn net.Conn, id int, method string, params any) {
	t.Helper()
	var raw json.RawMessage
	if params != nil {
		var err error
		raw, err = json.Marshal(params)
		if err != nil {
			t.Fatalf("marshal params: %v", err)
		}
	}
	msg := struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      int             `json:"id"`
		Method  string          `json:"method"`
		Params  json.RawMessage `json:"params,omitempty"`
	}{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  raw,
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	data = append(data, '\n')
	if _, err := conn.Write(data); err != nil {
		t.Fatalf("write request: %v", err)
	}
}
