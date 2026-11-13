// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"net"
	"net/http/httputil"
	"net/url"
	"strconv"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

// TestPairedNodeInfoStreamReplacesTelemetryPull runs the production receiving
// stack (service, broker, scanner, cluster manager) against a paired peer whose
// node-info is the real nvpair-node-info binary behind a counting proxy. The
// receiving scanner holds one GET /v1/node-info/events stream to the peer and
// stops pulling GET /v1/node-info every two seconds; discovery:get-node-info is
// answered from that stream without a peer request.
//
// The fixture has one full stack, so only the receiving direction is covered.
// The broker relays scanner telemetry only to the job scheduler, which this
// fixture cannot observe; the keepalive re-emission cadence is covered by the
// scanner's unit tests.
func TestPairedNodeInfoStreamReplacesTelemetryPull(t *testing.T) {
	f := newPairedBootstrapFixture(t, false)
	port := freePort(t)
	info := startTask1Process(t, f.b.base, nodeInfoBin, "--port", strconv.Itoa(port), "--node-id", f.b.uuid)
	info.send(t, noderec.MethodSetClusterIdentity, noderec.ClusterIdentityParams{ClusterUUID: f.b.uuid}, false)
	waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 10*time.Second)
	upstream, err := url.Parse("http://127.0.0.1:" + strconv.Itoa(port))
	if err != nil {
		t.Fatal(err)
	}
	proxy := httputil.NewSingleHostReverseProxy(upstream)
	proxy.FlushInterval = -1
	f.nodeInfoUpstream.Store(proxy)
	f.bootstrap(t)

	params := map[string]string{"hostUuid": f.b.uuid}
	task1Eventually(t, "node information from the real node-info stream", 20*time.Second, func() bool {
		return f.call(t, "discovery:get-node-info", params).Error == nil
	})
	streams := f.count("upstream GET /v1/node-info/events")
	if streams < 1 {
		t.Fatal("real node-info never served its stream to the receiving scanner")
	}

	// A 10 s window: the 2 s pull would make about five plain reads here.
	plainBefore := f.count("upstream GET /v1/node-info")
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		start := time.Now()
		response := f.call(t, "discovery:get-node-info", params)
		elapsed := time.Since(start)
		if response.Error != nil {
			t.Fatalf("node information from a live stream failed: %+v", response.Error)
		}
		if elapsed > 100*time.Millisecond {
			t.Fatalf("discovery:get-node-info took %s, want an answer from the live frame within 100 ms", elapsed)
		}
		var got phase04NodeInfoSample
		task2Decode(t, response.Result, &got)
		if got.HostUUID != f.b.uuid || got.ClusterUUID == nil || *got.ClusterUUID != f.b.uuid {
			t.Fatalf("streamed node information = %+v", got)
		}
		time.Sleep(time.Second)
	}
	if plainAfter := f.count("upstream GET /v1/node-info"); plainAfter != plainBefore {
		t.Fatalf("stream peer was still pulled: plain node-info reads %d -> %d over 10 s", plainBefore, plainAfter)
	}
	if after := f.count("upstream GET /v1/node-info/events"); after != streams {
		t.Fatalf("node-info stream reconnected during a quiet window: %d -> %d", streams, after)
	}
}
