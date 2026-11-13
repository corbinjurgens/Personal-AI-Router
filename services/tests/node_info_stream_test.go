// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/mdns"
	"nvpair-shared/noderec"
)

// nodeInfoStreamNode is one side of TestPairedNodeInfoStreamReplacesTelemetryPull:
// its real node-info behind a counting proxy, and its real scanner.
type nodeInfoStreamNode struct {
	uuid, base, cert string
	// namespace is the private mDNS namespace this node's scanner browses;
	// only the peer advertises in it, so the scanner never polls itself
	// through the counting proxy.
	namespace noderec.MDNSNamespace
	scanner   *task1Process
	proxyPort int
	mu        sync.Mutex
	counts    map[string]int
}

func (n *nodeInfoStreamNode) count(path string) int {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.counts[path]
}

// peerTelemetry counts the discovery:node-telemetry notifications for hostUUID
// that the scanner has written to its broker link.
func (n *nodeInfoStreamNode) peerTelemetry(t *testing.T, hostUUID string) int {
	t.Helper()
	n.scanner.mu.Lock()
	defer n.scanner.mu.Unlock()
	if n.scanner.readErr != nil {
		t.Fatalf("scanner %s output: %v", n.uuid, n.scanner.readErr)
	}
	count := 0
	for _, frame := range n.scanner.frames {
		if frame.Method != noderec.NotifyNodeTelemetry {
			continue
		}
		var telemetry noderec.NodeTelemetry
		if err := json.Unmarshal(frame.Params, &telemetry); err != nil {
			t.Fatalf("scanner %s telemetry: %v", n.uuid, err)
		}
		if telemetry.HostUUID == hostUUID {
			count++
		}
	}
	return count
}

// TestPairedNodeInfoStreamReplacesTelemetryPull runs two paired nodes, each
// with the real nvpair-node-info behind a counting proxy and the real
// nvpair-node-scanner. Each scanner discovers the other node in its own private
// mDNS namespace and holds one GET /v1/node-info/events stream to it instead of
// pulling GET /v1/node-info every two seconds. In both directions, over a 10 s
// window:
//   - the scanner sends its broker discovery:node-telemetry for the peer at
//     least every 3 s. The scanner's stdout is the link its broker reads, so
//     the notifications counted there are the ones the broker receives;
//   - the peer's node-info serves the stream and no plain read;
//   - discovery:get-node-info for the peer answers from the live frame within
//     100 ms.
func TestPairedNodeInfoStreamReplacesTelemetryPull(t *testing.T) {
	a := &nodeInfoStreamNode{uuid: "ni-stream-a", namespace: newFixtureMDNSNamespace(t), counts: map[string]int{}}
	b := &nodeInfoStreamNode{uuid: "ni-stream-b", namespace: newFixtureMDNSNamespace(t), counts: map[string]int{}}
	nodes := []*nodeInfoStreamNode{a, b}
	peerOf := func(n *nodeInfoStreamNode) *nodeInfoStreamNode {
		if n == a {
			return b
		}
		return a
	}
	for _, n := range nodes {
		n.base = shortClusterTempDir(t)
		dir := filepath.Join(n.base, "cluster")
		if err := os.MkdirAll(filepath.Join(dir, "trusted"), 0o700); err != nil {
			t.Fatal(err)
		}
		n.cert = mintClusterIdentity(t, dir, n.uuid)
		task1WriteJSON(t, filepath.Join(dir, "identity.json"), map[string]any{"node_uuid": n.uuid, "created_at": 1})
		task1WriteJSON(t, filepath.Join(dir, "admission.json"), map[string]any{"counter": 1, "activated": 1, "clusterId": task1ClusterID, "epoch": 1})
	}
	// No cluster manager runs, but admission needs a member entry with a port.
	memberPorts := map[string]int{a.uuid: temporaryClusterPort(t), b.uuid: temporaryClusterPort(t)}
	for _, n := range nodes {
		peer := peerOf(n)
		dir := filepath.Join(n.base, "cluster")
		task1WriteJSON(t, filepath.Join(dir, "trusted", peer.uuid+".json"), map[string]any{"nodeUuid": peer.uuid, "nodeId": peer.uuid, "certPem": peer.cert, "clusterId": task1ClusterID, "admissionEpoch": 1})
		var members []task1Member
		for _, member := range nodes {
			members = append(members, task1Member{ID: member.uuid, NodeUUID: member.uuid, Name: member.uuid, IPAddress: "127.0.0.1", Port: memberPorts[member.uuid], ClusterID: task1ClusterID, AdmissionEpoch: 1, State: "member"})
		}
		task1WriteJSON(t, filepath.Join(dir, "members.json"), members)
		if _, admitted := clustertrust.Open(dir).PeerServiceIncarnation(peer.uuid); !admitted {
			t.Fatalf("%s does not admit %s", n.uuid, peer.uuid)
		}
	}
	for _, n := range nodes {
		port := freePort(t)
		info := startTask1Process(t, n.base, nodeInfoBin, "--port", strconv.Itoa(port), "--node-id", n.uuid)
		info.send(t, noderec.MethodSetClusterIdentity, noderec.ClusterIdentityParams{ClusterUUID: n.uuid}, false)
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 10*time.Second)
		upstream, err := url.Parse("http://127.0.0.1:" + strconv.Itoa(port))
		if err != nil {
			t.Fatal(err)
		}
		proxy := httputil.NewSingleHostReverseProxy(upstream)
		proxy.FlushInterval = -1
		server := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			n.mu.Lock()
			n.counts[r.Method+" "+r.URL.Path]++
			n.mu.Unlock()
			proxy.ServeHTTP(w, r)
		}))
		t.Cleanup(server.Close)
		n.proxyPort = server.Listener.Addr().(*net.TCPAddr).Port
	}
	for _, n := range nodes {
		args := append([]string{"--cluster-dir", filepath.Join(n.base, "cluster")}, fixtureMDNSScannerArgs(n.namespace)...)
		n.scanner = startTask1Process(t, n.base, scannerBin, args...)
		n.scanner.call(t, "discovery:get-nodes", nil)
		label := "ni-stream-" + filepath.Base(n.base)
		advertised := peerOf(n).namespace
		responder, err := mdns.NewResponder(label, label, advertised.Service, advertised.Domain, noderec.SRVPort, []string{
			"v=1", "uuid=" + n.uuid, "cluster-uuid=" + n.uuid, "ip=127.0.0.1", fmt.Sprintf("ni=%d", n.proxyPort),
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		advertiseDone := make(chan error, 1)
		go func() { advertiseDone <- responder.Run(ctx) }()
		t.Cleanup(func() {
			cancel()
			select {
			case err := <-advertiseDone:
				if err != nil {
					t.Error(err)
				}
			case <-time.After(5 * time.Second):
				t.Error("fixture advertiser did not stop")
			}
		})
	}

	getNodeInfo := func(n *nodeInfoStreamNode) (phase04NodeInfoSample, time.Duration, error) {
		start := time.Now()
		response := n.scanner.callResponse(t, "discovery:get-node-info", map[string]string{"hostUuid": peerOf(n).uuid})
		elapsed := time.Since(start)
		if response.Error != nil {
			return phase04NodeInfoSample{}, elapsed, fmt.Errorf("%+v", response.Error)
		}
		var got phase04NodeInfoSample
		task2Decode(t, response.Result, &got)
		return got, elapsed, nil
	}
	for _, n := range nodes {
		peer := peerOf(n)
		task1Eventually(t, n.uuid+" node information for "+peer.uuid+" from the node-info stream", 30*time.Second, func() bool {
			_, _, err := getNodeInfo(n)
			return err == nil && peer.count("GET /v1/node-info/events") > 0
		})
		// The first telemetry for a new stream peer follows its first frame.
		task1Eventually(t, n.uuid+" telemetry for "+peer.uuid, 10*time.Second, func() bool {
			return n.peerTelemetry(t, peer.uuid) > 0
		})
	}

	// A 10 s window: the 2 s pull would make about five plain reads here. The
	// loop samples each scanner's telemetry every 100 ms and asks each for the
	// peer's node information once a second.
	type direction struct {
		receiver, peer      *nodeInfoStreamNode
		plain, streams      int
		telemetry           int
		lastTelemetry       time.Time
		longestTelemetryGap time.Duration
	}
	directions := []*direction{{receiver: a, peer: b}, {receiver: b, peer: a}}
	windowStart := time.Now()
	for _, d := range directions {
		d.plain, d.streams = d.peer.count("GET /v1/node-info"), d.peer.count("GET /v1/node-info/events")
		d.telemetry, d.lastTelemetry = d.receiver.peerTelemetry(t, d.peer.uuid), windowStart
	}
	deadline := windowStart.Add(10 * time.Second)
	for tick := 0; time.Now().Before(deadline); tick++ {
		for _, d := range directions {
			now := time.Now()
			if count := d.receiver.peerTelemetry(t, d.peer.uuid); count > d.telemetry {
				d.longestTelemetryGap = max(d.longestTelemetryGap, now.Sub(d.lastTelemetry))
				d.telemetry, d.lastTelemetry = count, now
			}
			if tick%10 != 0 {
				continue
			}
			got, elapsed, err := getNodeInfo(d.receiver)
			if err != nil {
				t.Fatalf("%s node information for %s from a live stream failed: %v", d.receiver.uuid, d.peer.uuid, err)
			}
			if elapsed > 100*time.Millisecond {
				t.Fatalf("%s discovery:get-node-info for %s took %s, want an answer from the live frame within 100 ms", d.receiver.uuid, d.peer.uuid, elapsed)
			}
			if got.HostUUID != d.peer.uuid || got.ClusterUUID == nil || *got.ClusterUUID != d.peer.uuid {
				t.Fatalf("%s streamed node information for %s = %+v", d.receiver.uuid, d.peer.uuid, got)
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	for _, d := range directions {
		d.longestTelemetryGap = max(d.longestTelemetryGap, time.Since(d.lastTelemetry))
		t.Logf("%s -> %s: longest gap between peer telemetry notifications %s", d.peer.uuid, d.receiver.uuid, d.longestTelemetryGap)
		if d.longestTelemetryGap > 3*time.Second {
			t.Errorf("%s sent its broker no telemetry for %s for %s, want at least every 3 s", d.receiver.uuid, d.peer.uuid, d.longestTelemetryGap)
		}
		if plain := d.peer.count("GET /v1/node-info"); plain != d.plain {
			t.Errorf("stream peer %s was still pulled by %s: plain node-info reads %d -> %d over 10 s", d.peer.uuid, d.receiver.uuid, d.plain, plain)
		}
		if streams := d.peer.count("GET /v1/node-info/events"); streams != d.streams {
			t.Errorf("node-info stream from %s to %s reconnected during a quiet window: %d -> %d", d.peer.uuid, d.receiver.uuid, d.streams, streams)
		}
	}
}
