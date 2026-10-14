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

func TestNodeInfoRPCFreshFullSampleThroughReverseRetry(t *testing.T) {
	f := newScannerReverseFixture(t)
	// The cache deliberately disagrees with the live sample.
	f.d.lastInfo["peer"] = NodeInfoResponse{HostUUID: "peer", GPUs: []GPUInfo{{Name: "stale GPU"}}, MSSince: 99000}
	client := nodeInfoRPCTestClient(t, f.d)
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`)
	if rpcErr != nil {
		t.Fatalf("fresh getter rejected: %+v", rpcErr)
	}
	var got NodeInfoResponse
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatal(err)
	}
	cluster := "peer"
	want := NodeInfoResponse{HostUUID: "peer", ClusterUUID: &cluster,
		GPUs:   []GPUInfo{{Name: "reverse GPU", UtilizationPercent: 73}},
		CPU:    &CPUInfo{Name: "peer CPU", Cores: 8, UtilizationPercent: 42},
		Memory: &MemoryInfo{TotalBytes: 8192, UsedBytes: 2048}, TelemetryValid: true, MSSince: 100}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("full sample = %+v, want %+v", got, want)
	}
	if f.directInfo.Load() != 1 || f.reverseInfo.Load() != 1 || f.infoCalls.Load() != 1 {
		t.Fatalf("attempts direct/reverse/handler = %d/%d/%d, want 1/1/1", f.directInfo.Load(), f.reverseInfo.Load(), f.infoCalls.Load())
	}
	if f.d.lastInfo["peer"].GPUs[0].Name != "stale GPU" {
		t.Fatal("read-only getter changed enrichment cache")
	}
}

func TestNodeInfoRPCDirectUsesPairedAuthority(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	f.reverse.Store(false)
	client := nodeInfoRPCTestClient(t, f.d)
	_, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`)
	if rpcErr != nil || f.directInfo.Load() != 1 || f.reverseInfo.Load() != 0 || f.infoCalls.Load() != 1 {
		t.Fatalf("paired direct read: error=%+v direct/reverse/handler=%d/%d/%d", rpcErr, f.directInfo.Load(), f.reverseInfo.Load(), f.infoCalls.Load())
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
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"physical-host"}`)
	var got NodeInfoResponse
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatalf("decode result, RPC error=%+v: %v", rpcErr, err)
	}
	if rpcErr != nil || got.HostUUID != "physical-host" || got.ClusterUUID == nil || *got.ClusterUUID != "peer" || f.reverseInfo.Load() != 1 {
		t.Fatalf("host/principal resolution: sample=%+v error=%+v reverse=%d", got, rpcErr, f.reverseInfo.Load())
	}
}

func TestNodeInfoRPCRejectsConflictingPrincipal(t *testing.T) {
	for _, principal := range []string{"other-principal", ""} {
		t.Run("reported principal "+principal, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			f.reportedPrincipal.Store(&principal)
			client := nodeInfoRPCTestClient(t, f.d)
			result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`)
			if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 || f.infoCalls.Load() != 1 {
				t.Fatalf("conflicting principal accepted: result=%s error=%+v calls=%d", result, rpcErr, f.infoCalls.Load())
			}
		})
	}
}

func TestNodeInfoRPCAcceptsLegacyNullablePrincipal(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.reportedPrincipal.Store(nil)
	client := nodeInfoRPCTestClient(t, f.d)
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`)
	if rpcErr != nil {
		t.Fatalf("legacy principal rejected: %+v", rpcErr)
	}
	var got NodeInfoResponse
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatal(err)
	}
	if got.HostUUID != "peer" || got.ClusterUUID != nil || f.infoCalls.Load() != 1 {
		t.Fatalf("legacy principal sample=%+v calls=%d", got, f.infoCalls.Load())
	}
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

func TestNodeInfoRPCRejectsWrongIdentityAndIncompleteBody(t *testing.T) {
	test := func(name string, configure func(*scannerReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			configure(f)
			client := nodeInfoRPCTestClient(t, f.d)
			result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`)
			if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 || f.infoCalls.Load() != 1 {
				t.Fatalf("invalid response: result=%s error=%+v calls=%d", result, rpcErr, f.infoCalls.Load())
			}
		})
	}
	test("wrong host", func(f *scannerReverseFixture) { f.reportedHost.Store("stranger") })
	test("complete JSON in truncated HTTP body", func(f *scannerReverseFixture) { f.responseEnding.Store(scannerIncompleteFraming) })
	test("additional JSON value", func(f *scannerReverseFixture) { f.responseEnding.Store(scannerAdditionalJSON) })
	test("interrupted body is not replayed", func(f *scannerReverseFixture) { f.interrupt.Store(true) })
}

func TestNodeInfoRPCRejectsAuthorityLossDuringRead(t *testing.T) {
	test := func(name string, change func(*testing.T, *scannerReverseFixture)) {
		t.Run(name, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			entered, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			releaseReply := func() { once.Do(func() { close(release) }) }
			t.Cleanup(releaseReply)
			f.beforeReply = func(noderec.ServiceKey) { close(entered); <-release }
			client := nodeInfoRPCTestClient(t, f.d)
			type response struct {
				rpcErr *jsonrpc.RPCError
				err    error
			}
			done := make(chan response, 1)
			go func() {
				ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
				defer cancel()
				_, rpcErr, err := client.Call(ctx, "discovery:get-node-info", json.RawMessage(`{"hostUuid":"peer"}`))
				done <- response{rpcErr, err}
			}()
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("fetch did not reach handler")
			}
			change(t, f)
			releaseReply()
			select {
			case got := <-done:
				if got.err != nil || got.rpcErr == nil || got.rpcErr.Code != -32000 {
					t.Fatalf("authority loss response: %+v", got)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("authority loss did not finish")
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
	if _, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`); rpcErr != nil {
		t.Fatalf("initial sample: %+v", rpcErr)
	}
	f.d.lastInfo["peer"] = NodeInfoResponse{HostUUID: "peer", TelemetryValid: true}
	f.reverse.Store(false)
	f.d.closePeerClients()
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`)
	if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 {
		t.Fatalf("session outage returned success/cache: %s %+v", result, rpcErr)
	}
	node, found := f.d.dir.get("peer")
	if !found || node.ClusterUUID != "peer" || !node.Trusted {
		t.Fatalf("session outage changed membership: %+v", node)
	}
	f.reverse.Store(true)
	if _, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`); rpcErr != nil {
		t.Fatalf("recovered sample: %+v", rpcErr)
	}
	if f.infoCalls.Load() != 2 {
		t.Fatalf("successful fresh handler calls = %d, want 2", f.infoCalls.Load())
	}
}

func TestNodeInfoRPCDeadlineAndNonblockingReadPump(t *testing.T) {
	f := newScannerReverseFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	f.beforeReply = func(noderec.ServiceKey) { once.Do(func() { close(entered) }); <-release }
	t.Cleanup(func() { close(release) })
	client := nodeInfoRPCTestClient(t, f.d)
	type response struct {
		rpcErr *jsonrpc.RPCError
		err    error
	}
	done := make(chan response, 1)
	started := time.Now()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_, rpcErr, err := client.Call(ctx, "discovery:get-node-info", json.RawMessage(`{"hostUuid":"peer"}`))
		done <- response{rpcErr, err}
	}()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("fetch did not start")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	result, rpcErr, err := client.Call(ctx, "discovery:get-nodes", nil)
	if err != nil || rpcErr != nil {
		t.Fatalf("scanner read pump blocked: %v %+v", err, rpcErr)
	}
	var nodes noderec.GetNodesResult
	if err := json.Unmarshal(result, &nodes); err != nil || len(nodes.Nodes) != 1 || nodes.Nodes[0].HostUUID != "peer" {
		t.Fatalf("concurrent directory response = %s: %v", result, err)
	}
	select {
	case got := <-done:
		elapsed := time.Since(started)
		if got.err != nil || got.rpcErr == nil || got.rpcErr.Code != -32000 || elapsed < time.Second || elapsed > 2500*time.Millisecond {
			t.Fatalf("bounded read: response=%+v elapsed=%s", got, elapsed)
		}
	case <-time.After(2500 * time.Millisecond):
		t.Fatal("backend exceeded bounded read deadline")
	}
}

func TestNodeInfoRPCDeadlineIncludesOriginQueue(t *testing.T) {
	f := newScannerReverseFixture(t)
	origin := "http://" + net.JoinHostPort("localhost", strconv.Itoa(f.node.Services[noderec.ServiceNodeInfo].Port))
	release, acquired := f.d.nodeInfoOrigins.acquire(context.Background(), origin)
	if !acquired {
		t.Fatal("could not hold the origin gate")
	}
	defer release()
	client := nodeInfoRPCTestClient(t, f.d)
	started := time.Now()
	result, rpcErr := callNodeInfoRPC(t, client, `{"hostUuid":"peer"}`)
	elapsed := time.Since(started)
	if rpcErr == nil || rpcErr.Code != -32000 || len(result) != 0 || elapsed < time.Second || elapsed > 2500*time.Millisecond {
		t.Fatalf("queue deadline: result=%s error=%+v elapsed=%s", result, rpcErr, elapsed)
	}
	if f.directInfo.Load() != 0 || f.reverseInfo.Load() != 0 || f.infoCalls.Load() != 0 {
		t.Fatalf("queued request reached network: %d/%d/%d", f.directInfo.Load(), f.reverseInfo.Load(), f.infoCalls.Load())
	}
}
