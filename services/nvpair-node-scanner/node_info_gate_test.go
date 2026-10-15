// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"maps"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/noderec"
)

func TestScannerPeerRouteUsesDirectoryGeneration(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("initial authenticated snapshot rejected")
	}
	key := scannerPeerEndpoint{hostUUID: "peer", peerUUID: "peer", address: "192.0.2.40:14311", service: noderec.ServiceCluster}
	first, ok := d.peerServiceRoute(key, "192.0.2.40", 14311, true)
	if !ok || first.Generation == 0 || first.Generation != d.dir.routeGeneration("peer") || first.Incarnation != claim.Incarnation || first.Address != key.address || first.Protocol != clustertrust.ReverseTLS {
		t.Fatalf("initial route = %+v, ok=%v; want current nonzero directory generation and authenticated endpoint", first, ok)
	}
	_, current, ok := d.peerServiceClient("peer", "peer", "192.0.2.40", 14311, noderec.ServiceCluster, false)
	if !ok || !current() {
		t.Fatal("initial directory route unavailable")
	}
	claim.Sequence++
	claim.Descriptor.Name = "renamed peer"
	if !d.acceptPeerServices(claim) {
		t.Fatal("unchanged authenticated route rejected")
	}
	unchanged, ok := d.peerServiceRoute(key, "192.0.2.40", 14311, true)
	if !ok || unchanged != first || !current() {
		t.Fatalf("unchanged snapshot retired route: before=%+v after=%+v ok=%v", first, unchanged, ok)
	}
	claim.Sequence++
	claim.Descriptor.Services = []noderec.PeerService{}
	if !d.acceptPeerServices(claim) {
		t.Fatal("service withdrawal rejected")
	}
	if _, ok := d.peerServiceRoute(key, "192.0.2.40", 14311, true); ok || current() {
		t.Fatal("withdrawn service retained route authority")
	}
	withdrawnGeneration := d.dir.routeGeneration("peer")
	if withdrawnGeneration <= first.Generation {
		t.Fatalf("withdrawal generation = %d, want greater than %d", withdrawnGeneration, first.Generation)
	}
	claim.Sequence++
	claim.Descriptor.Services = []noderec.PeerService{{Service: noderec.ServiceCluster, Port: 14311}}
	if !d.acceptPeerServices(claim) {
		t.Fatal("identical endpoint re-add rejected")
	}
	readded, ok := d.peerServiceRoute(key, "192.0.2.40", 14311, true)
	if !ok || readded.Generation <= withdrawnGeneration || readded.Generation != d.dir.routeGeneration("peer") || current() {
		t.Fatalf("re-added route = %+v, ok=%v; want current directory generation greater than %d and retired old guard", readded, ok, withdrawnGeneration)
	}
	_, newCurrent, ok := d.peerServiceClient("peer", "peer", "192.0.2.40", 14311, noderec.ServiceCluster, false)
	if !ok || !newCurrent() {
		t.Fatal("fresh request did not capture re-added route")
	}
}

func TestScannerPeerRouteRejectsPreviousAdmissionClaimBeforeReconciliation(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("initial authenticated snapshot rejected")
	}
	_, previousCurrent, ok := d.peerServiceClient("peer", "peer", "192.0.2.40", 14311, noderec.ServiceCluster, false)
	if !ok || !previousCurrent() {
		t.Fatal("initial authenticated route unavailable")
	}
	generation := d.dir.routeGeneration("peer")
	// Complete the pin and durable-member replacement with the same certificate,
	// leaving scanner reconciliation and the old authenticated descriptor pending.
	writeScannerAdmissionPin(t, d.clusterDir, "peer", "cluster", 2)
	d.mesh.Refresh()
	incarnation, admitted := d.mesh.PeerServiceIncarnation("peer")
	if !admitted || incarnation == claim.Incarnation {
		t.Fatal("fixture did not complete same-certificate readmission")
	}
	if _, _, ok := d.peerServiceClient("peer", "peer", "192.0.2.40", 14311, noderec.ServiceCluster, false); ok {
		t.Fatal("new admission authorized a fresh request using the previous admission's descriptor")
	}
	if previousCurrent() {
		t.Fatal("previous request retained authority after readmission")
	}
	d.dir.mu.RLock()
	stored, present := d.dir.peerServices["peer"]
	d.dir.mu.RUnlock()
	if !present || stored.Incarnation != claim.Incarnation || d.dir.routeGeneration("peer") != generation {
		t.Fatal("test did not preserve the old claim before reconciliation")
	}
	claim.Incarnation = incarnation
	claim.Sequence++
	claim.ObservedAt = time.Now().UnixMilli()
	if !d.acceptPeerServices(claim) {
		t.Fatal("new admission's fresh descriptor rejected")
	}
	_, current, ok := d.peerServiceClient("peer", "peer", "192.0.2.40", 14311, noderec.ServiceCluster, false)
	if !ok || !current() {
		t.Fatal("fresh descriptor did not restore the current admission's route")
	}
}

func TestScannerPeerRouteRejectsExpiredClaimBeforeReconciliation(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("initial authenticated snapshot rejected")
	}
	client, previousCurrent, ok := d.peerServiceClient("peer", "peer", "192.0.2.40", 14311, noderec.ServiceCluster, false)
	if !ok || !previousCurrent() {
		t.Fatal("initial authenticated route unavailable")
	}
	generation := d.dir.routeGeneration("peer")
	expiredAt := time.Now().Add(-31 * time.Second).UnixMilli()
	d.dir.mu.Lock()
	expired := d.dir.peerServices["peer"]
	expired.ObservedAt = expiredAt
	d.dir.peerServices["peer"] = expired
	d.dir.mu.Unlock()
	if _, _, ok := d.peerServiceClient("peer", "peer", "192.0.2.40", 14311, noderec.ServiceCluster, false); ok {
		t.Fatal("expired descriptor authorized a fresh request before the sweeper")
	}
	if previousCurrent() {
		t.Fatal("previous request retained authority after descriptor expiry")
	}
	response, err := client.Get("https://192.0.2.40:14311/v1/cluster/services")
	if err == nil {
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		t.Fatal("cached client dispatched with expired descriptor authority")
	}
	d.dir.mu.RLock()
	stored, present := d.dir.peerServices["peer"]
	d.dir.mu.RUnlock()
	if !present || stored.ObservedAt != expiredAt || d.dir.routeGeneration("peer") != generation {
		t.Fatal("test did not preserve the expired claim before reconciliation")
	}
	claim.Sequence++
	claim.ObservedAt = time.Now().UnixMilli()
	if !d.acceptPeerServices(claim) {
		t.Fatal("fresh descriptor refresh rejected")
	}
	_, current, ok := d.peerServiceClient("peer", "peer", "192.0.2.40", 14311, noderec.ServiceCluster, false)
	if !ok || !current() {
		t.Fatal("fresh descriptor did not restore the current route")
	}
}

func TestScannerPeerRouteRejectsHeldResponseAfterDirectoryChange(t *testing.T) {
	test := func(name string, change func(*testing.T, *scannerReverseFixture), wantCurrent, wantGenerationChange, wantFresh bool) {
		t.Run(name, func(t *testing.T) {
			f := newScannerReverseFixture(t)
			port := f.node.Services[noderec.ServiceNodeInfo].Port
			_, current, ok := f.d.peerServiceClient("peer", "peer", "localhost", port, noderec.ServiceNodeInfo, false)
			if !ok || !current() {
				t.Fatal("initial route unavailable")
			}
			generation := f.d.dir.routeGeneration("peer")
			var output bytes.Buffer
			f.d.codec = NewCodec(&output)
			entered, release := make(chan struct{}), make(chan struct{})
			var once, releaseOnce sync.Once
			releaseReply := func() { releaseOnce.Do(func() { close(release) }) }
			t.Cleanup(releaseReply)
			f.beforeReply = func(service noderec.ServiceKey) {
				if service == noderec.ServiceNodeInfo {
					once.Do(func() { close(entered); <-release })
				}
			}
			done := make(chan bool, 1)
			go func() {
				done <- f.d.refreshNodeTelemetry(context.Background(), "peer", "localhost", port)
			}()
			select {
			case <-entered:
			case <-time.After(3 * time.Second):
				t.Fatal("request did not reach held response")
			}
			change(t, f)
			if got := f.d.dir.routeGeneration("peer"); (got > generation) != wantGenerationChange || got < generation {
				t.Fatalf("directory generation before=%d after=%d, want advancement=%v", generation, got, wantGenerationChange)
			}
			if got := current(); got != wantCurrent {
				t.Fatalf("captured authority current = %v, want %v", got, wantCurrent)
			}
			releaseReply()
			select {
			case published := <-done:
				if published != wantCurrent {
					t.Fatalf("held telemetry published = %v, want %v", published, wantCurrent)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("held response did not finish")
			}
			if !wantCurrent && output.Len() != 0 {
				t.Fatalf("stale response emitted %d bytes", output.Len())
			}
			// Every scenario with a current original tuple must allow a fresh read.
			if wantFresh {
				if _, ok := f.d.getPeerNodeInfo(context.Background(), "peer"); !ok {
					t.Fatal("fresh request could not use the current endpoint")
				}
			}
		})
	}
	update := func(f *scannerReverseFixture, change func(*noderec.DirectoryNode)) {
		node := f.node
		node.Services = maps.Clone(node.Services)
		change(&node)
		f.d.dir.upsert(node)
	}
	test("identical endpoint removed and re-added", func(t *testing.T, f *scannerReverseFixture) {
		if !f.d.dir.remove("peer") {
			t.Fatal("fixture endpoint missing before removal")
		}
		f.d.dir.upsert(f.node)
	}, false, true, true)
	test("identical service withdrawn and re-added", func(_ *testing.T, f *scannerReverseFixture) {
		update(f, func(node *noderec.DirectoryNode) { delete(node.Services, noderec.ServiceNodeInfo) })
		f.d.dir.upsert(f.node)
	}, false, true, true)
	test("port replacement", func(_ *testing.T, f *scannerReverseFixture) {
		update(f, func(node *noderec.DirectoryNode) {
			node.Services[noderec.ServiceNodeInfo] = noderec.ServiceStatus{Port: 1}
		})
	}, false, true, false)
	test("address replacement", func(_ *testing.T, f *scannerReverseFixture) {
		update(f, func(node *noderec.DirectoryNode) { node.IP = "192.0.2.41" })
	}, false, true, false)
	test("service withdrawal", func(_ *testing.T, f *scannerReverseFixture) {
		update(f, func(node *noderec.DirectoryNode) { delete(node.Services, noderec.ServiceNodeInfo) })
	}, false, true, false)
	test("principal replacement", func(_ *testing.T, f *scannerReverseFixture) {
		update(f, func(node *noderec.DirectoryNode) { node.ClusterUUID = "stranger" })
	}, false, true, false)
	test("new admission", func(t *testing.T, f *scannerReverseFixture) {
		clustertrusttest.WriteAdmission(t, f.d.clusterDir, "cluster", 2)
	}, false, false, true)
	test("unchanged endpoint", func(_ *testing.T, f *scannerReverseFixture) {
		update(f, func(node *noderec.DirectoryNode) { node.Name = "new display name" })
	}, true, false, true)
}

func TestScannerPeerRouteGenerationResetsCachedClientHealth(t *testing.T) {
	f := newScannerReverseFixture(t)
	port := f.node.Services[noderec.ServiceNodeInfo].Port
	oldClient, oldCurrent, ok := f.d.peerServiceClient("peer", "peer", "localhost", port, noderec.ServiceNodeInfo, false)
	if !ok {
		t.Fatal("initial route unavailable")
	}
	if _, ok := f.d.getPeerNodeInfo(context.Background(), "peer"); !ok {
		t.Fatal("initial reverse request failed")
	}
	incarnation, admitted := f.d.mesh.PeerServiceIncarnation("peer")
	if !admitted {
		t.Fatal("fixture peer not admitted")
	}
	key := scannerPeerEndpoint{hostUUID: "peer", peerUUID: "peer", address: net.JoinHostPort("localhost", strconv.Itoa(port)), service: noderec.ServiceNodeInfo, incarnation: incarnation, generation: f.d.dir.routeGeneration("peer")}
	f.d.peerMu.Lock()
	adapter := f.d.peerClients[key]
	f.d.peerMu.Unlock()
	if adapter == nil {
		t.Fatal("directory-backed route was not cached")
	}
	// A display-only refresh must retain failed-direct health, even if direct
	// connectivity has recovered. A later semantic generation resets it.
	f.direct.Store(true)
	node := f.node
	node.Name = "renamed peer"
	f.d.dir.upsert(node)
	if _, ok := f.d.getPeerNodeInfo(context.Background(), "peer"); !ok {
		t.Fatal("unchanged route request failed")
	}
	if got := f.directInfo.Load(); got != 1 {
		t.Fatalf("unchanged route reset failed-direct health: direct attempts=%d, want 1", got)
	}
	f.d.peerMu.Lock()
	unchangedAdapter := f.d.peerClients[key] == adapter
	f.d.peerMu.Unlock()
	if !unchangedAdapter || !oldCurrent() {
		t.Fatal("unchanged route retired cached authority")
	}
	if !f.d.dir.remove("peer") {
		t.Fatal("fixture endpoint missing before removal")
	}
	f.d.dir.upsert(node)
	response, err := oldClient.Get("http://" + key.address + "/v1/node-info")
	if err == nil {
		if err := response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		t.Fatal("old client dispatched after identical endpoint removal/re-add")
	}
	if oldCurrent() || f.directInfo.Load() != 1 || f.reverseInfo.Load() != 2 {
		t.Fatalf("old route retained authority or dispatched: direct=%d reverse=%d, want 1/2", f.directInfo.Load(), f.reverseInfo.Load())
	}
	if _, ok := f.d.getPeerNodeInfo(context.Background(), "peer"); !ok {
		t.Fatal("fresh generation request failed")
	}
	f.d.peerMu.Lock()
	_, staleCached := f.d.peerClients[key]
	cacheSize := len(f.d.peerClients)
	f.d.peerMu.Unlock()
	if staleCached || cacheSize != 1 || f.directInfo.Load() != 2 || f.reverseInfo.Load() != 2 {
		t.Fatalf("generation did not replace cached route health: stale cached=%v cache size=%d direct=%d reverse=%d, want false/1/2/2", staleCached, cacheSize, f.directInfo.Load(), f.reverseInfo.Load())
	}
}

func TestScannerDiscoveringPeerRouteRemainsTransient(t *testing.T) {
	f := newScannerReverseFixture(t)
	if !f.d.dir.remove("peer") {
		t.Fatal("fixture endpoint missing before removal")
	}
	port := f.node.Services[noderec.ServiceNodeInfo].Port
	if _, _, ok := f.d.peerServiceClient("peer", "peer", "localhost", port, noderec.ServiceNodeInfo, false); ok {
		t.Fatal("directory-backed lookup accepted an absent node")
	}
	client, current, ok := f.d.peerServiceClient("peer", "peer", "localhost", port, noderec.ServiceNodeInfo, true)
	if !ok || !current() {
		t.Fatal("pre-insertion browse route unavailable")
	}
	key := scannerPeerEndpoint{hostUUID: "peer", peerUUID: "peer", address: net.JoinHostPort("localhost", strconv.Itoa(port)), service: noderec.ServiceNodeInfo}
	route, ok := f.d.peerServiceRoute(key, "localhost", port, false)
	if !ok || route.Generation != 0 || route.Protocol != clustertrust.ReversePlain {
		t.Fatalf("transient route = %+v, ok=%v; want generation zero and node-info policy", route, ok)
	}
	if _, ok := f.d.readNodeInfoWithin(context.Background(), client, "http", "localhost", port, current); !ok {
		t.Fatal("transient browse request failed")
	}
	if f.d.hasPeerClient("peer") {
		t.Fatal("pre-insertion route installed a permanent client")
	}
	f.d.dir.upsert(f.node)
	if !current() {
		t.Fatal("browse insertion invented a generation boundary for transient work")
	}
	clustertrusttest.WriteAdmission(t, f.d.clusterDir, "cluster", 2)
	if current() {
		t.Fatal("transient route survived admission replacement")
	}
}
