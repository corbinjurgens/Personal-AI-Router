// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/noderec"
)

func nodeInfoRPCTestClient(t *testing.T, d *daemon) *jsonrpc.Peer {
	t.Helper()
	clientConn, scannerConn := net.Pipe()
	d.codec = NewCodec(scannerConn)
	client := jsonrpc.NewPeer(NewCodec(clientConn))
	server := jsonrpc.NewPeer(d.codec)
	go client.Serve(nil, nil)
	go server.Serve(func(msg *Message) {
		if !d.handle(msg) {
			_ = d.codec.RespondError(msg.ID, -32601, "method not found")
		}
	}, nil)
	t.Cleanup(func() {
		client.Close()
		server.Close()
		_ = clientConn.Close()
		_ = scannerConn.Close()
	})
	return client
}

func callNodeInfoRPC(t *testing.T, client *jsonrpc.Peer, params string) (json.RawMessage, *jsonrpc.RPCError) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	result, rpcErr, err := client.Call(ctx, "discovery:get-node-info", json.RawMessage(params))
	if err != nil {
		t.Fatalf("node-info RPC transport: %v", err)
	}
	return result, rpcErr
}

func TestNodeInfoRPCRejectsInvalidParams(t *testing.T) {
	d := newSelfTestDaemon("self", "127.0.0.1")
	client := nodeInfoRPCTestClient(t, d)
	for _, params := range []string{`null`, `{}`, `[]`, `{"hostUuid":1}`, `{"hostUuid":" "}`, `{"hostUuid":"peer","url":"http://127.0.0.1"}`, `{"hostUuid":"peer","port":80}`, `{"hostUuid":"peer","principal":"other"}`} {
		t.Run(params, func(t *testing.T) {
			result, rpcErr := callNodeInfoRPC(t, client, params)
			if rpcErr == nil || rpcErr.Code != -32602 || len(result) != 0 {
				t.Fatalf("invalid params: result=%s error=%+v, want -32602", result, rpcErr)
			}
		})
	}
}

func TestNodeInfoRPCRejectsMissingAuthorityWithoutDirectFallback(t *testing.T) {
	test := func(name string, change func(*testing.T, *scannerReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			f.direct.Store(true)
			change(t, f)
			client := nodeInfoRPCTestClient(t, f.d)
			result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`)
			if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 {
				t.Fatalf("missing authority succeeded: result=%s error=%+v", result, rpcErr)
			}
			if f.directInfo.Load() != 0 || f.reverseInfo.Load() != 0 || f.infoCalls.Load() != 0 {
				t.Fatalf("missing authority reached network: %d/%d/%d", f.directInfo.Load(), f.reverseInfo.Load(), f.infoCalls.Load())
			}
		})
	}
	test("directory removed", func(_ *testing.T, f *scannerReverseFixture) { f.d.dir.remove("peer") })
	test("unpaired", func(_ *testing.T, f *scannerReverseFixture) {
		node := f.node
		node.ClusterUUID = ""
		f.d.dir.upsert(node)
	})
	test("pin removed before any adapter", func(t *testing.T, f *scannerReverseFixture) {
		clustertrusttest.RemovePeerPin(t, f.d.clusterDir, "peer")
	})
	test("admission removed", func(t *testing.T, f *scannerReverseFixture) {
		clustertrusttest.WriteAdmission(t, f.d.clusterDir, "", 2)
	})
	test("endpoint removed", func(_ *testing.T, f *scannerReverseFixture) {
		node := f.node
		node.Services = nil
		f.d.dir.upsert(node)
	})
}

// startFixtureStreams runs the scanner's node-info stream supervisor against
// the fixture until the test ends.
func startFixtureStreams(t *testing.T, f *scannerReverseFixture) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	var workers sync.WaitGroup
	t.Cleanup(func() { cancel(); workers.Wait() })
	f.d.superviseNodeInfoStreams(ctx, f.d.selfHostUUID(), &workers)
}

// waitNodeInfoRPC polls discovery:get-node-info until it answers, as a caller
// right after startup waits for the stream's first frame.
func waitNodeInfoRPC(t *testing.T, client *jsonrpc.Peer, host string) NodeInfoResponse {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"`+host+`"}`)
		if rpcErr == nil {
			var got NodeInfoResponse
			if err := json.Unmarshal(result, &got); err != nil {
				t.Fatal(err)
			}
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("node-info for %s never became available: %+v", host, rpcErr)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func requireNodeInfoUnavailable(t *testing.T, client *jsonrpc.Peer, host string) {
	t.Helper()
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"`+host+`"}`)
	if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 {
		t.Fatalf("node-info for %s: result=%s error=%+v, want unavailable", host, result, rpcErr)
	}
}

func TestNodeInfoRPCStreamPeerAnswersFromStreamThroughReverseRetry(t *testing.T) {
	f := newScannerReverseFixture(t)
	// The enrichment cache deliberately disagrees with the stream.
	f.d.lastInfo["peer"] = NodeInfoResponse{HostUUID: "peer", GPUs: []GPUInfo{{Name: "stale GPU"}}, MSSince: 99000}
	client := nodeInfoRPCTestClient(t, f.d)
	startFixtureStreams(t, f)
	got := waitNodeInfoRPC(t, client, "peer")
	cluster := "peer"
	want := NodeInfoResponse{HostUUID: "peer", ClusterUUID: &cluster,
		GPUs:   []GPUInfo{{Name: "reverse GPU", UtilizationPercent: 73}},
		CPU:    &CPUInfo{Name: "peer CPU", Cores: 8, UtilizationPercent: 42},
		Memory: &MemoryInfo{TotalBytes: 8192, UsedBytes: 2048}, TelemetryValid: true}
	if got.MSSince < 0 || got.MSSince > 1000 {
		t.Fatalf("live frame age = %d ms", got.MSSince)
	}
	got.MSSince = 0
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("full sample = %+v, want %+v", got, want)
	}
	if f.directInfo.Load() != 1 || f.reverseInfo.Load() != 1 || f.streamCalls.Load() != 1 || f.infoCalls.Load() != 0 {
		t.Fatalf("attempts direct/reverse/stream/plain = %d/%d/%d/%d, want 1/1/1/0", f.directInfo.Load(), f.reverseInfo.Load(), f.streamCalls.Load(), f.infoCalls.Load())
	}
	for range 3 {
		waitNodeInfoRPC(t, client, "peer")
	}
	if f.streamCalls.Load() != 1 || f.infoCalls.Load() != 0 {
		t.Fatalf("repeated answers reached the peer: stream=%d plain=%d", f.streamCalls.Load(), f.infoCalls.Load())
	}
	if f.d.lastInfo["peer"].GPUs[0].Name != "stale GPU" {
		t.Fatal("read-only getter changed enrichment cache")
	}
}

func TestNodeInfoRPCStreamPeerDirectUsesPairedAuthority(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	f.reverse.Store(false)
	client := nodeInfoRPCTestClient(t, f.d)
	startFixtureStreams(t, f)
	waitNodeInfoRPC(t, client, "peer")
	if f.directInfo.Load() != 1 || f.reverseInfo.Load() != 0 || f.streamCalls.Load() != 1 || f.infoCalls.Load() != 0 {
		t.Fatalf("paired direct stream: direct/reverse/stream/plain=%d/%d/%d/%d", f.directInfo.Load(), f.reverseInfo.Load(), f.streamCalls.Load(), f.infoCalls.Load())
	}
}

func TestNodeInfoRPCResolvesHostSeparatelyFromPrincipal(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.d.dir.remove("peer")
	node := f.node
	node.HostUUID = "physical-host"
	f.d.dir.upsert(node)
	f.reportedHost.Store("physical-host")
	client := nodeInfoRPCTestClient(t, f.d)
	startFixtureStreams(t, f)
	got := waitNodeInfoRPC(t, client, "physical-host")
	if got.HostUUID != "physical-host" || got.ClusterUUID == nil || *got.ClusterUUID != "peer" || f.reverseInfo.Load() != 1 {
		t.Fatalf("host/principal resolution: sample=%+v reverse=%d", got, f.reverseInfo.Load())
	}
}

func TestNodeInfoRPCRejectsConflictingPrincipal(t *testing.T) {
	for _, principal := range []string{"other-principal", ""} {
		t.Run("reported principal "+principal, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			f.reportedPrincipal.Store(&principal)
			client := nodeInfoRPCTestClient(t, f.d)
			startFixtureStreams(t, f)
			// The frame's principal is applied to the directory, after which the
			// node is no longer a paired stream peer.
			deadline := time.Now().Add(5 * time.Second)
			for mustClusterUUID(t, f.d, "peer") != principal {
				if time.Now().After(deadline) {
					t.Fatal("stream frame's principal was not applied")
				}
				time.Sleep(20 * time.Millisecond)
			}
			requireNodeInfoUnavailable(t, client, "peer")
			if f.infoCalls.Load() != 0 {
				t.Fatalf("conflicting principal caused %d plain reads", f.infoCalls.Load())
			}
		})
	}
}

func TestNodeInfoRPCAcceptsLegacyNullablePrincipal(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.reportedPrincipal.Store(nil)
	client := nodeInfoRPCTestClient(t, f.d)
	startFixtureStreams(t, f)
	got := waitNodeInfoRPC(t, client, "peer")
	if got.HostUUID != "peer" || got.ClusterUUID != nil || f.streamCalls.Load() != 1 {
		t.Fatalf("legacy principal sample=%+v streams=%d", got, f.streamCalls.Load())
	}
}

func TestNodeInfoRPCRejectsWrongIdentityAndInterruptedStream(t *testing.T) {
	test := func(name string, configure func(*scannerReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			configure(f)
			client := nodeInfoRPCTestClient(t, f.d)
			startFixtureStreams(t, f)
			deadline := time.Now().Add(3 * time.Second)
			for f.streamCalls.Load() == 0 && time.Now().Before(deadline) {
				time.Sleep(10 * time.Millisecond)
			}
			time.Sleep(100 * time.Millisecond)
			requireNodeInfoUnavailable(t, client, "peer")
			if f.streamCalls.Load() != 1 || f.infoCalls.Load() != 0 {
				t.Fatalf("stream/plain = %d/%d, want 1/0", f.streamCalls.Load(), f.infoCalls.Load())
			}
		})
	}
	test("wrong host", func(f *scannerReverseFixture) { f.reportedHost.Store("stranger") })
	test("interrupted stream is not replayed", func(f *scannerReverseFixture) { f.interrupt.Store(true) })
}

func TestNodeInfoRPCRejectsAuthorityLossWhileStreaming(t *testing.T) {
	test := func(name string, change func(*testing.T, *scannerReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			client := nodeInfoRPCTestClient(t, f.d)
			startFixtureStreams(t, f)
			waitNodeInfoRPC(t, client, "peer")
			change(t, f)
			// Refused at answer time, before the stream notices on its next line.
			requireNodeInfoUnavailable(t, client, "peer")
			if f.infoCalls.Load() != 0 {
				t.Fatalf("authority loss caused %d plain reads", f.infoCalls.Load())
			}
		})
	}
	test("pin", func(t *testing.T, f *scannerReverseFixture) {
		clustertrusttest.RemovePeerPin(t, f.d.clusterDir, "peer")
	})
	test("admission", func(t *testing.T, f *scannerReverseFixture) {
		clustertrusttest.WriteAdmission(t, f.d.clusterDir, "cluster", 2)
	})
	test("endpoint", func(_ *testing.T, f *scannerReverseFixture) {
		node := f.node
		node.Services = map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 1}}
		f.d.dir.upsert(node)
	})
	test("directory", func(_ *testing.T, f *scannerReverseFixture) { f.d.dir.remove("peer") })
}

func TestNodeInfoRPCSessionOutageAndRecoveryNeverReturnsCache(t *testing.T) {
	f := newScannerReverseFixture(t)
	client := nodeInfoRPCTestClient(t, f.d)
	startFixtureStreams(t, f)
	waitNodeInfoRPC(t, client, "peer")
	f.d.lastInfo["peer"] = NodeInfoResponse{HostUUID: "peer", TelemetryValid: true}
	f.reverse.Store(false)
	f.mu.Lock()
	for _, conn := range f.connections {
		conn.Close()
	}
	f.mu.Unlock()
	deadline := time.Now().Add(3 * time.Second)
	for {
		result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`)
		if rpcErr != nil {
			if rpcErr.Code != -32000 || len(result) != 0 {
				t.Fatalf("session outage: %s %+v", result, rpcErr)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("session outage kept answering")
		}
		time.Sleep(20 * time.Millisecond)
	}
	node, found := f.d.dir.get("peer")
	if !found || node.ClusterUUID != "peer" || !node.Trusted {
		t.Fatalf("session outage changed membership: %+v", node)
	}
	f.reverse.Store(true)
	waitNodeInfoRPC(t, client, "peer")
	if f.streamCalls.Load() != 2 || f.infoCalls.Load() != 0 {
		t.Fatalf("stream/plain handler calls = %d/%d, want 2/0", f.streamCalls.Load(), f.infoCalls.Load())
	}
}

func TestNodeInfoRPCStreamAnswerNeedsNoNetworkOrOriginGate(t *testing.T) {
	f := newScannerReverseFixture(t)
	client := nodeInfoRPCTestClient(t, f.d)
	startFixtureStreams(t, f)
	waitNodeInfoRPC(t, client, "peer")
	origin := "http://" + net.JoinHostPort("localhost", strconv.Itoa(f.node.Services[noderec.ServiceNodeInfo].Port))
	release, acquired := f.d.nodeInfoOrigins.acquire(context.Background(), origin)
	if !acquired {
		t.Fatal("stream held the origin gate")
	}
	defer release()
	started := time.Now()
	if _, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`); rpcErr != nil {
		t.Fatalf("stream answer behind a held origin gate: %+v", rpcErr)
	}
	if elapsed := time.Since(started); elapsed > 100*time.Millisecond {
		t.Fatalf("stream answer took %s", elapsed)
	}
	if f.infoCalls.Load() != 0 || f.streamCalls.Load() != 1 {
		t.Fatalf("stream answer reached the network: plain=%d stream=%d", f.infoCalls.Load(), f.streamCalls.Load())
	}
}

// liveFrameFixture installs a cached frame for the fixture peer without a
// stream, so the answer-time rules can be checked directly.
func liveFrameFixture(f *scannerReverseFixture, lineAt time.Time, current func() bool) {
	cluster := "peer"
	s := &f.d.nodeInfoStreams
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.live == nil {
		s.live = make(map[string]liveNodeInfo)
	}
	s.live["peer"] = liveNodeInfo{streamID: 1, info: NodeInfoResponse{HostUUID: "peer", ClusterUUID: &cluster, TelemetryValid: true, MSSince: 40,
		GPUs: []GPUInfo{{Name: "cached GPU", UtilizationPercent: 12}}}, frameAt: lineAt, lineAt: lineAt, current: current}
}

func TestNodeInfoRPCLiveFrameRules(t *testing.T) {
	t.Run("live frame", func(t *testing.T) {
		f := newScannerReverseFixture(t)
		liveFrameFixture(f, time.Now().Add(-time.Second), func() bool { return true })
		client := nodeInfoRPCTestClient(t, f.d)
		got := waitNodeInfoRPC(t, client, "peer")
		if got.GPUs[0].Name != "cached GPU" || got.MSSince < 1000 || got.MSSince > 2000 {
			t.Fatalf("live frame answer = %+v", got)
		}
		if f.infoCalls.Load() != 0 || f.streamCalls.Load() != 0 || f.directDials.Load() != 0 || f.reverseInfo.Load() != 0 {
			t.Fatal("live frame answer reached the network")
		}
	})
	t.Run("stale frame", func(t *testing.T) {
		f := newScannerReverseFixture(t)
		liveFrameFixture(f, time.Now().Add(-nodeInfoStreamIdle-time.Second), func() bool { return true })
		requireNodeInfoUnavailable(t, nodeInfoRPCTestClient(t, f.d), "peer")
		if f.infoCalls.Load() != 0 || f.directDials.Load() != 0 || f.reverseInfo.Load() != 0 {
			t.Fatal("stale frame fell back to a peer read")
		}
	})
	t.Run("route no longer current", func(t *testing.T) {
		f := newScannerReverseFixture(t)
		liveFrameFixture(f, time.Now(), func() bool { return false })
		requireNodeInfoUnavailable(t, nodeInfoRPCTestClient(t, f.d), "peer")
		if f.infoCalls.Load() != 0 || f.directDials.Load() != 0 || f.reverseInfo.Load() != 0 {
			t.Fatal("withdrawn route fell back to a peer read")
		}
	})
}

func TestClusterIdentitySweepSkipsLiveStreams(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	liveFrameFixture(f, time.Now(), func() bool { return true })
	f.d.refreshClusterIdentityOnce()
	if f.infoCalls.Load() != 0 {
		t.Fatalf("identity sweep read a peer with a live stream %d times", f.infoCalls.Load())
	}
	liveFrameFixture(f, time.Now().Add(-nodeInfoStreamIdle-time.Second), func() bool { return true })
	f.d.refreshClusterIdentityOnce()
	if f.infoCalls.Load() != 1 {
		t.Fatalf("identity sweep reads for a stream peer whose stream is down = %d, want 1", f.infoCalls.Load())
	}
}
