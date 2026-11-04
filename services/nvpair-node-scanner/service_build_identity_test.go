// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/buildinfo"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

type serviceBuildTransport func(*http.Request) (*http.Response, error)

func (f serviceBuildTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func serviceBuildResponse(body string, status int) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

const scannerCleanBuild = "git:0123456789abcdef0123456789abcdef01234567"
const scannerDirtyBuild = scannerCleanBuild + "+dirty"

func TestScannerBuildDecodeAndEnrichmentCache(t *testing.T) {
	for _, tc := range []struct {
		name, field string
		want        buildinfo.ID
	}{
		{"clean", `,"serviceBuildId":"` + scannerCleanBuild + `"`, scannerCleanBuild},
		{"dirty", `,"serviceBuildId":"` + scannerDirtyBuild + `"`, scannerDirtyBuild},
		{"absent clears", "", ""},
		{"invalid clears", `,"serviceBuildId":"git:short"`, ""},
		{"oversize clears", `,"serviceBuildId":"` + strings.Repeat("a", 500) + `"`, ""},
		{"number clears", `,"serviceBuildId":7`, ""},
		{"object clears", `,"serviceBuildId":{}`, ""},
		{"array clears", `,"serviceBuildId":[]`, ""},
		{"null clears", `,"serviceBuildId":null`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"hostUuid":"host-a","GPUs":[{"name":"GPU"}],"cpu":{"cores":8},"memory":{"total_bytes":16384}` + tc.field + `}`
			status := http.StatusOK
			d := &daemon{lastInfo: map[string]NodeInfoResponse{"host-a": {ServiceBuildID: scannerDirtyBuild}}}
			d.http = &http.Client{Transport: serviceBuildTransport(func(*http.Request) (*http.Response, error) {
				return serviceBuildResponse(body, status), nil
			})}
			node := noderec.DirectoryNode{HostUUID: "host-a", ServiceBuildID: scannerDirtyBuild, Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 14318}}}
			d.enrichInfoAt(&node, "127.0.0.1")
			if node.ServiceBuildID != tc.want || len(node.GPUs) != 1 || node.GPUs[0].Name != "GPU" || node.CPU == nil || node.CPU.Cores != 8 || node.Memory == nil || node.Memory.TotalBytes != 16384 {
				t.Fatalf("successful node = %+v, want build %q with hardware", node, tc.want)
			}
			if d.lastInfo["host-a"].ServiceBuildID != tc.want {
				t.Fatalf("cache retained old build: %+v", d.lastInfo["host-a"])
			}
			status = http.StatusServiceUnavailable
			cachedNode := noderec.DirectoryNode{HostUUID: "host-a", Services: node.Services}
			d.enrichInfoAt(&cachedNode, "127.0.0.1")
			if cachedNode.ServiceBuildID != tc.want || !reflect.DeepEqual(cachedNode.GPUs, node.GPUs) || !reflect.DeepEqual(cachedNode.Memory, node.Memory) {
				t.Fatalf("failed fetch did not reuse last good inventory: %+v", cachedNode)
			}
		})
	}
}

func TestScannerBuildOnlyRefreshEmitsAndClears(t *testing.T) {
	var output bytes.Buffer
	policy := nodepolicy.Default().Advertisement()
	node := noderec.DirectoryNode{HostUUID: "host-a", IP: "127.0.0.1", Name: "fixture", ServiceBuildID: scannerCleanBuild, Models: []string{"model-a"}, Policy: &policy, GPUs: []noderec.GPUInfo{{Name: "GPU"}}, Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 14318}}}
	d := &daemon{codec: NewCodec(&output), dir: newDirectory(), lastInfo: map[string]NodeInfoResponse{"host-a": {ServiceBuildID: scannerCleanBuild}}}
	d.dir.upsert(node)
	body := `{"hostUuid":"host-a","serviceBuildId":"` + scannerDirtyBuild + `"}`
	status := http.StatusOK
	d.http = &http.Client{Transport: serviceBuildTransport(func(*http.Request) (*http.Response, error) {
		return serviceBuildResponse(body, status), nil
	})}
	refresh := func(want buildinfo.ID, wantUpdates int) {
		t.Helper()
		output.Reset()
		if !d.refreshNodeTelemetry(context.Background(), "host-a", "127.0.0.1", 14318) {
			t.Fatal("successful refresh rejected")
		}
		got, present := d.dir.get("host-a")
		if !present || got.ServiceBuildID != want || !reflect.DeepEqual(got.Models, []string{"model-a"}) || !reflect.DeepEqual(got.Policy, &policy) || !reflect.DeepEqual(got.GPUs, node.GPUs) {
			t.Fatalf("refresh clobbered node: %+v", got)
		}
		if d.lastInfo["host-a"].ServiceBuildID != want {
			t.Fatalf("cache build = %q, want %q", d.lastInfo["host-a"].ServiceBuildID, want)
		}
		updates := 0
		decoder := json.NewDecoder(&output)
		for {
			var msg Message
			err := decoder.Decode(&msg)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatalf("decode notification: %v", err)
			}
			if msg.Method == noderec.NotifyNodeUpdated {
				updates++
				var event noderec.NodeEvent
				if err := json.Unmarshal(msg.Params, &event); err != nil {
					t.Fatalf("decode build update: %v", err)
				}
				if event.Node.ServiceBuildID != want || event.Node.HostUUID != "host-a" {
					t.Fatalf("update = %+v, want build %q", event, want)
				}
			}
		}
		if updates != wantUpdates {
			t.Fatalf("build updates = %d, want %d", updates, wantUpdates)
		}
	}
	refresh(scannerDirtyBuild, 1)
	refresh(scannerDirtyBuild, 0)
	status = http.StatusServiceUnavailable
	output.Reset()
	if d.refreshNodeTelemetry(context.Background(), "host-a", "127.0.0.1", 14318) {
		t.Fatal("failed fetch accepted")
	}
	got, _ := d.dir.get("host-a")
	if got.ServiceBuildID != scannerDirtyBuild || output.Len() != 0 {
		t.Fatal("failed fetch cleared build or emitted update")
	}
	status = http.StatusOK
	body = `{"hostUuid":"host-a"}`
	refresh("", 1)
	body = `{"hostUuid":"host-a","serviceBuildId":"` + scannerCleanBuild + `"}`
	refresh(scannerCleanBuild, 1)
	body = `{"hostUuid":"host-a","serviceBuildId":{}}`
	refresh("", 1)
}

func TestScannerBuildRefreshRejectsReplacedEndpoints(t *testing.T) {
	for _, scenario := range []string{"removed", "removed and readded", "address replaced", "port replaced", "already stale target"} {
		t.Run(scenario, func(t *testing.T) {
			var output bytes.Buffer
			d := &daemon{codec: NewCodec(&output), dir: newDirectory()}
			node := noderec.DirectoryNode{HostUUID: "host-a", IP: "127.0.0.1", ServiceBuildID: scannerCleanBuild, Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 14318}}}
			d.dir.upsert(node)
			if scenario == "already stale target" {
				node.IP = "127.0.0.2"
				d.dir.upsert(node)
			}
			d.http = &http.Client{Transport: serviceBuildTransport(func(*http.Request) (*http.Response, error) {
				switch scenario {
				case "removed":
					d.dir.remove("host-a")
				case "removed and readded":
					d.dir.remove("host-a")
					d.dir.upsert(node)
				case "address replaced":
					node.IP = "127.0.0.2"
					d.dir.upsert(node)
				case "port replaced":
					node.Services = map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 14319}}
					d.dir.upsert(node)
				}
				return serviceBuildResponse(`{"hostUuid":"host-a","serviceBuildId":"`+scannerDirtyBuild+`"}`, http.StatusOK), nil
			})}
			d.refreshNodeTelemetry(context.Background(), "host-a", "127.0.0.1", 14318)
			got, present := d.dir.get("host-a")
			if scenario == "removed" {
				if present {
					t.Fatal("late build resurrected removed node")
				}
			} else if !present || got.ServiceBuildID != scannerCleanBuild {
				t.Fatalf("late build clobbered replacement: %+v", got)
			}
			decoder := json.NewDecoder(&output)
			for {
				var msg Message
				err := decoder.Decode(&msg)
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("decode notification: %v", err)
				}
				if msg.Method == noderec.NotifyNodeUpdated {
					t.Fatal("late build emitted directory update")
				}
			}
		})
	}
}

func TestScannerBuildFetchHonorsCurrentGuard(t *testing.T) {
	d := &daemon{}
	client := &http.Client{Transport: serviceBuildTransport(func(*http.Request) (*http.Response, error) {
		return serviceBuildResponse(`{"hostUuid":"host-a","GPUs":[{"name":"GPU"}],"serviceBuildId":"`+scannerCleanBuild+`"}`, http.StatusOK), nil
	})}
	got, ok := d.readNodeInfoWithin(context.Background(), client, "http", "127.0.0.1", 14318, func() bool { return false })
	if ok || got.HostUUID != "" || got.ServiceBuildID != "" || got.GPUs != nil {
		t.Fatalf("retired peer response accepted: %+v, ok=%v", got, ok)
	}
}

func TestDirectoryBuildRejectsReplacedAdmissionGeneration(t *testing.T) {
	dir := newDirectory()
	node := noderec.DirectoryNode{HostUUID: "host-a", IP: "127.0.0.1", ServiceBuildID: scannerCleanBuild}
	dir.installPeerServices(noderec.PeerServicesClaim{Incarnation: "old-admission"}, node)
	generation := dir.routeGeneration("host-a")
	dir.installPeerServices(noderec.PeerServicesClaim{Incarnation: "new-admission"}, node)
	_, changed, ok := dir.applyServiceBuild("host-a", generation, scannerDirtyBuild)
	if ok || changed {
		t.Fatalf("old admission build applied: ok=%v changed=%v", ok, changed)
	}
	got, present := dir.get("host-a")
	if !present || got.ServiceBuildID != scannerCleanBuild {
		t.Fatalf("replacement admission clobbered: %+v", got)
	}
}

func newServiceBuildPeerFixture(t *testing.T) (*daemon, noderec.PeerServicesClaim, noderec.DirectoryNode, *bytes.Buffer) {
	t.Helper()
	d, claim := newPeerServicesScanner(t)
	claim.Descriptor.Services = []noderec.PeerService{{Service: noderec.ServiceNodeInfo, Port: 14318}}
	policy := nodepolicy.Default().Advertisement()
	node := noderec.DirectoryNode{
		HostUUID: "peer", Name: "peer-name", ClusterUUID: "peer", Trusted: true,
		IP: "192.0.2.40", ServiceBuildID: scannerCleanBuild,
		Services:       map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 14318}},
		GPUs:           []GPUInfo{{Name: "previous GPU"}},
		Models:         []string{"model-a"},
		ModelsByEngine: map[string][]string{"ollama": {"model-a"}},
		LoadedByEngine: map[string][]string{"ollama": {"model-a"}},
		Policy:         &policy,
	}
	d.dir.installPeerServices(claim, node)
	d.lastInfo["peer"] = NodeInfoResponse{HostUUID: "peer", ServiceBuildID: scannerCleanBuild, GPUs: []GPUInfo{{Name: "previous GPU"}}}
	output := &bytes.Buffer{}
	d.codec = NewCodec(output)
	return d, claim, node, output
}

func TestAuthenticatedEnrichmentProjectsServiceBuildImmediately(t *testing.T) {
	for _, tc := range []struct {
		name, field string
		want        buildinfo.ID
	}{
		{"reported build", `,"serviceBuildId":"` + scannerDirtyBuild + `"`, scannerDirtyBuild},
		{"successful absence clears", "", ""},
		{"successful invalid clears", `,"serviceBuildId":{}`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, claim, node, output := newServiceBuildPeerFixture(t)
			fetches := 0
			d.tlsHTTP = &http.Client{Transport: serviceBuildTransport(func(*http.Request) (*http.Response, error) {
				fetches++
				return serviceBuildResponse(`{"hostUuid":"peer","GPUs":[{"name":"reported GPU"}],"cpu":{"cores":8},"memory":{"total_bytes":16384}`+tc.field+`}`, http.StatusOK), nil
			})}
			d.enrichPeerServices(claim, node, d.dir.routeGeneration("peer"))
			if fetches != 1 {
				t.Fatalf("node-info fetches = %d, want 1", fetches)
			}
			want := node
			want.ServiceBuildID = tc.want
			want.GPUs = []GPUInfo{{Name: "reported GPU"}}
			want.CPU = &CPUInfo{Cores: 8}
			want.Memory = &MemoryInfo{TotalBytes: 16384}
			if got := peerServicesNode(t, d); !reflect.DeepEqual(got, want) {
				t.Fatalf("immediate directory = %+v, want %+v with original models/policy", got, want)
			}
			cached := d.lastInfo["peer"]
			if cached.ServiceBuildID != tc.want || cached.HostUUID != "peer" || !reflect.DeepEqual(cached.GPUs, want.GPUs) {
				t.Fatalf("successful cached node-info = %+v, want build %q and fetched identity/hardware", cached, tc.want)
			}
			var msg Message
			decoder := json.NewDecoder(output)
			if err := decoder.Decode(&msg); err != nil {
				t.Fatalf("decode immediate update: %v", err)
			}
			if msg.Method != noderec.NotifyNodeUpdated {
				t.Fatalf("method = %q, want node-updated", msg.Method)
			}
			var event noderec.NodeEvent
			if err := json.Unmarshal(msg.Params, &event); err != nil {
				t.Fatalf("decode node event: %v", err)
			}
			if !reflect.DeepEqual(event.Node, want) {
				t.Fatalf("immediate event = %+v, want %+v", event.Node, want)
			}
			if err := decoder.Decode(&msg); err != io.EOF {
				t.Fatalf("unexpected trailing event or decode error: %v", err)
			}
		})
	}
}

func TestAuthenticatedBuildRejectsStaleOrUnadmittedEnrichment(t *testing.T) {
	for _, scenario := range []string{"stale generation", "stale incarnation", "admission withdrawn during fetch"} {
		t.Run(scenario, func(t *testing.T) {
			d, claim, node, output := newServiceBuildPeerFixture(t)
			generation := d.dir.routeGeneration("peer")
			cachedBefore := d.lastInfo["peer"]
			want := node
			if scenario == "stale incarnation" {
				claim.Incarnation = "obsolete-admission"
			}
			fetches := 0
			var fixtureErr error
			d.tlsHTTP = &http.Client{Transport: serviceBuildTransport(func(*http.Request) (*http.Response, error) {
				fetches++
				switch scenario {
				case "stale generation":
					want.IP = "192.0.2.41"
					d.dir.installPeerServices(claim, want)
				case "admission withdrawn during fetch":
					if err := os.Remove(filepath.Join(d.clusterDir, "trusted", "peer.json")); err != nil {
						fixtureErr = err
						return nil, err
					}
					d.mesh.Refresh()
					if _, admitted := d.mesh.PeerServiceIncarnation("peer"); admitted {
						fixtureErr = errors.New("fixture admission was not withdrawn")
						return nil, fixtureErr
					}
				}
				return serviceBuildResponse(`{"hostUuid":"peer","GPUs":[{"name":"stale GPU"}],"serviceBuildId":"`+scannerDirtyBuild+`"}`, http.StatusOK), nil
			})}
			d.enrichPeerServices(claim, node, generation)
			if fixtureErr != nil {
				t.Fatalf("withdraw fixture admission: %v", fixtureErr)
			}
			if fetches != 1 {
				t.Fatalf("node-info fetches = %d, want 1 before final authority guard", fetches)
			}
			if got := peerServicesNode(t, d); !reflect.DeepEqual(got, want) {
				t.Fatalf("stale enrichment changed current node/models/policy: got %+v, want %+v", got, want)
			}
			if !reflect.DeepEqual(d.lastInfo["peer"], cachedBefore) {
				t.Fatalf("stale enrichment changed cached node-info: %+v", d.lastInfo["peer"])
			}
			if output.Len() != 0 {
				t.Fatal("stale enrichment emitted a node update")
			}
		})
	}
}

type serviceBuildCloseBody struct {
	io.Reader
	onClose func() error
}

func (b *serviceBuildCloseBody) Close() error { return b.onClose() }

func TestTelemetryBuildRejectsAdmissionWithdrawnAfterFetchValidation(t *testing.T) {
	d, claim, node, output := newServiceBuildPeerFixture(t)
	generation := d.dir.routeGeneration("peer")
	cachedBefore := d.lastInfo["peer"]
	_, current, ok := d.peerServiceClient("peer", "peer", node.IP, 14318, noderec.ServiceNodeInfo, false)
	if !ok || !current() {
		t.Fatal("fixture has no current admitted node-info route")
	}
	t.Cleanup(d.closePeerClients)
	var seamErr error
	closes := 0
	d.tlsHTTP = &http.Client{Transport: serviceBuildTransport(func(*http.Request) (*http.Response, error) {
		body := &serviceBuildCloseBody{
			Reader: strings.NewReader(`{"hostUuid":"peer","serviceBuildId":"` + scannerDirtyBuild + `"}`),
			onClose: func() error {
				// readNodeInfoWithin defers Close until after decoding and its
				// current() validation, before the caller can apply the build.
				closes++
				if !current() {
					seamErr = errors.New("fixture route retired before the close seam")
					return seamErr
				}
				if err := os.Remove(filepath.Join(d.clusterDir, "trusted", "peer.json")); err != nil {
					seamErr = err
					return err
				}
				d.mesh.Refresh()
				if _, admitted := d.mesh.PeerServiceIncarnation("peer"); admitted || current() {
					seamErr = errors.New("fixture route remained admitted after pin withdrawal")
					return seamErr
				}
				return nil
			},
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: body}, nil
	})}

	d.refreshNodeTelemetryCandidates(context.Background(), "peer", node.CandidateIPs(), 14318)
	if seamErr != nil {
		t.Fatalf("post-validation admission withdrawal: %v", seamErr)
	}
	if closes != 1 {
		t.Fatalf("response closes = %d, want one deterministic withdrawal", closes)
	}
	if _, admitted := d.mesh.PeerServiceIncarnation("peer"); admitted {
		t.Fatal("fixture peer is still admitted after fetch")
	}
	if got := d.dir.routeGeneration("peer"); got != generation {
		t.Fatalf("fixture changed route generation %d -> %d, obscuring admission boundary", generation, got)
	}
	d.dir.mu.RLock()
	retained, exists := d.dir.peerServices["peer"]
	d.dir.mu.RUnlock()
	if !exists || retained.Incarnation != claim.Incarnation {
		t.Fatal("fixture retired the directory claim instead of only withdrawing admission")
	}
	if got := peerServicesNode(t, d); !reflect.DeepEqual(got, node) {
		t.Errorf("retired response changed directory build or unrelated node fields: got %+v, want %+v", got, node)
	}
	if got := d.lastInfo["peer"]; !reflect.DeepEqual(got, cachedBefore) {
		t.Errorf("retired response changed cached build or hardware: got %+v, want %+v", got, cachedBefore)
	}
	decoder := json.NewDecoder(output)
	for {
		var msg Message
		err := decoder.Decode(&msg)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("decode refresh notification: %v", err)
		}
		if msg.Method == noderec.NotifyNodeUpdated {
			var event noderec.NodeEvent
			if err := json.Unmarshal(msg.Params, &event); err != nil {
				t.Fatalf("decode directory update: %v", err)
			}
			t.Errorf("retired response emitted node-updated: %+v", event.Node)
		}
	}
}

func TestTelemetryBuildRechecksAuthenticatedClaimAtApply(t *testing.T) {
	for _, tc := range []struct {
		name, field string
		want        buildinfo.ID
		updates     int
	}{
		{"current claim", `,"serviceBuildId":"` + scannerDirtyBuild + `"`, scannerDirtyBuild, 1},
		{"current claim absent build", "", "", 1},
		{"expired claim", `,"serviceBuildId":"` + scannerDirtyBuild + `"`, scannerCleanBuild, 0},
		{"obsolete incarnation", `,"serviceBuildId":"` + scannerDirtyBuild + `"`, scannerCleanBuild, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, _, node, output := newServiceBuildPeerFixture(t)
			generation := d.dir.routeGeneration("peer")
			cachedBefore := d.lastInfo["peer"]
			closes := 0
			d.tlsHTTP = &http.Client{Transport: serviceBuildTransport(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body: &serviceBuildCloseBody{
						Reader: strings.NewReader(`{"hostUuid":"peer"` + tc.field + `}`),
						onClose: func() error {
							closes++
							d.peerServicesMu.Lock()
							d.dir.mu.Lock()
							claim := d.dir.peerServices["peer"]
							switch tc.name {
							case "expired claim":
								claim.ObservedAt = time.Now().Add(-peerServicesMaxAge - time.Second).UnixMilli()
							case "obsolete incarnation":
								claim.Incarnation = "obsolete-admission"
							}
							d.dir.peerServices["peer"] = claim
							d.dir.mu.Unlock()
							d.peerServicesMu.Unlock()
							return nil
						},
					},
				}, nil
			})}
			if !d.refreshNodeTelemetryCandidates(context.Background(), "peer", node.CandidateIPs(), 14318) {
				t.Fatal("claim revalidation suppressed otherwise valid telemetry")
			}
			if closes != 1 || d.dir.routeGeneration("peer") != generation {
				t.Fatalf("fixture missed final-claim boundary: closes=%d generation=%d, want 1 and %d", closes, d.dir.routeGeneration("peer"), generation)
			}
			wantNode := node
			wantNode.ServiceBuildID = tc.want
			if got := peerServicesNode(t, d); !reflect.DeepEqual(got, wantNode) {
				t.Fatalf("directory = %+v, want %+v with original identity/hardware/models/policy", got, wantNode)
			}
			wantCache := cachedBefore
			wantCache.ServiceBuildID = tc.want
			if got := d.lastInfo["peer"]; !reflect.DeepEqual(got, wantCache) {
				t.Fatalf("cache = %+v, want %+v", got, wantCache)
			}
			updates, telemetry := 0, 0
			decoder := json.NewDecoder(output)
			for {
				var msg Message
				err := decoder.Decode(&msg)
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("decode refresh notification: %v", err)
				}
				switch msg.Method {
				case noderec.NotifyNodeUpdated:
					updates++
					var event noderec.NodeEvent
					if err := json.Unmarshal(msg.Params, &event); err != nil {
						t.Fatalf("decode directory update: %v", err)
					}
					if !reflect.DeepEqual(event.Node, wantNode) {
						t.Fatalf("build event = %+v, want %+v", event.Node, wantNode)
					}
				case noderec.NotifyNodeTelemetry:
					telemetry++
				}
			}
			if updates != tc.updates || telemetry != 1 {
				t.Fatalf("events: updates=%d telemetry=%d, want %d and 1", updates, telemetry, tc.updates)
			}
		})
	}
}
