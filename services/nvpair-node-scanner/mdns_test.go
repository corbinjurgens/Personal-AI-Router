// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/mdns"
	"nvpair-shared/noderec"
)

func newMDNSDisabledDaemon(t *testing.T) *daemon {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "cluster")
	return NewScanner(nil, clustertrust.Open(dir), dir, nil, false).daemon
}

func requireMDNSDisabled(t *testing.T, d *daemon) {
	t.Helper()
	if d.browser != nil {
		t.Fatal("mDNS disabled scanner constructed a browser")
	}
	if d.responder != nil {
		t.Fatal("mDNS disabled scanner constructed a responder")
	}
}

func startMDNSDisabledDaemon(t *testing.T, d *daemon) (context.CancelFunc, <-chan struct{}) {
	t.Helper()
	// Fail before run can open any multicast sockets if suppression regresses.
	requireMDNSDisabled(t, d)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		d.run(ctx)
		close(done)
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("mDNS disabled daemon did not stop after cancellation")
		}
	})
	return cancel, done
}

func waitMDNSCondition(t *testing.T, description string, condition func() bool) {
	t.Helper()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	for !condition() {
		select {
		case <-deadline.C:
			t.Fatalf("timed out waiting for %s", description)
		case <-tick.C:
		}
	}
}

func TestMDNSDisabledSuppressesConstruction(t *testing.T) {
	requireMDNSDisabled(t, newMDNSDisabledDaemon(t))
}

func TestMDNSEnabledConstructsDiscovery(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cluster")
	d := NewScanner(nil, clustertrust.Open(dir), dir, nil, true).daemon
	if d.browser == nil {
		t.Fatal("mDNS enabled scanner did not construct a browser")
	}
	// NewResponder only enumerates interfaces; Run and UpdateTXT send traffic.
	// Compare availability without starting either discovery component, including
	// on hosts without multicast-capable interfaces where construction must fail.
	_, err := mdns.NewResponder(d.instance, noderec.ServiceType, noderec.Domain, noderec.SRVPort, d.reg.txt())
	if got, want := d.responder != nil, err == nil; got != want {
		t.Fatalf("responder constructed = %v, want %v (interface availability: %v)", got, want, err)
	}
}

func TestMDNSDisabledDaemonPublishesSelfUntilCancellation(t *testing.T) {
	d := newMDNSDisabledDaemon(t)
	cancel, done := startMDNSDisabledDaemon(t, d)
	waitMDNSCondition(t, "self publication", func() bool {
		_, exists := d.dir.get(d.reg.record().HostUUID)
		return exists
	})
	select {
	case <-done:
		t.Fatal("daemon stopped without cancellation while mDNS was disabled")
	case <-time.After(50 * time.Millisecond):
	}
	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("daemon did not stop after cancellation")
	}
}

func TestMDNSDisabledPeerServicesRefreshAndMaintenance(t *testing.T) {
	fixture, claim := newPeerServicesScanner(t)
	d := NewScanner(nil, fixture.mesh, fixture.clusterDir, nil, false).daemon
	_, done := startMDNSDisabledDaemon(t, d)
	send := func(claim noderec.PeerServicesClaim) {
		t.Helper()
		params, err := json.Marshal(claim)
		if err != nil {
			t.Fatalf("marshal peer-services notification: %v", err)
		}
		if !d.handle(&Message{Method: "discovery:peer-services", Params: params}) {
			t.Fatal("private peer-services notification not handled")
		}
	}
	send(claim)
	if node := peerServicesNode(t, d); !node.Trusted || node.ClusterUUID != "peer" || node.Name != "peer-name" || node.Services[noderec.ServiceCluster].Port != 14311 || node.LastSeen != 0 {
		t.Fatalf("admitted peer was not ingested with existing authority: %+v", node)
	}
	refreshed := claim
	refreshed.Sequence = 2
	refreshed.Descriptor.Name = "refreshed-peer"
	refreshed.Descriptor.Services = []noderec.PeerService{{Service: noderec.ServiceCluster, Port: 15000}}
	send(refreshed)
	send(claim)
	if node := peerServicesNode(t, d); node.Name != "refreshed-peer" || node.Services[noderec.ServiceCluster].Port != 15000 || node.LastSeen != 0 {
		t.Fatalf("fresh metadata did not replace the peer or stale replay won: %+v", node)
	}
	// Age the accepted claim, then let the real daemon's one-second maintenance
	// retire it. No direct reconcile call can make this test pass.
	d.peerServicesMu.Lock()
	d.dir.mu.Lock()
	expired := d.dir.peerServices["peer"]
	expired.ObservedAt = time.Now().Add(-time.Minute).UnixMilli()
	d.dir.peerServices["peer"] = expired
	d.dir.mu.Unlock()
	d.peerServicesMu.Unlock()
	waitMDNSCondition(t, "periodic expiry of peer-services claim", func() bool {
		_, exists := d.dir.get("peer")
		return !exists
	})
	select {
	case <-done:
		t.Fatal("daemon stopped during mDNS-independent maintenance")
	default:
	}
}

func TestMDNSDisabledRefreshesInventory(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		timings := defaultScannerTimings()
		timings.peerRefreshInterval = 25 * time.Millisecond
		dir := filepath.Join(t.TempDir(), "cluster")
		d := newScannerWithTimings(nil, clustertrust.Open(dir), dir, nil, false, timings).daemon
		requireMDNSDisabled(t, d)
		const hostUUID = "peer-inventory"
		seedEMNode(d, hostUUID, "127.0.0.1", 18080)
		body := `{"models":["initial-model"]}`
		d.modelsHTTP = &http.Client{Transport: peerServicesTestTransport(func(r *http.Request) (*http.Response, error) {
			if r.Method != http.MethodGet || r.URL.Scheme != "http" || r.URL.Host != "127.0.0.1:18080" || r.URL.Path != "/v1/models" {
				return nil, fmt.Errorf("unexpected peer inventory request: %s %s", r.Method, r.URL)
			}
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
		})}
		// A peer-only engine registration prevents self address re-ranking from
		// refreshing this inventory through publishSelf instead of the sweep.
		d.refreshModelsOnce()
		if node, ok := d.dir.get(hostUUID); !ok || len(node.Models) != 1 || node.Models[0] != "initial-model" {
			t.Fatalf("initial peer inventory = %+v, present = %v", node, ok)
		}
		body = `{"models":["refreshed-model"]}`
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		done := make(chan struct{})
		go func() {
			d.refreshPeersLoop(ctx)
			close(done)
		}()
		synctest.Wait()
		time.Sleep(timings.peerRefreshInterval - time.Nanosecond)
		synctest.Wait()
		if node, ok := d.dir.get(hostUUID); !ok || len(node.Models) != 1 || node.Models[0] != "initial-model" {
			t.Fatalf("inventory refreshed before the configured tick: %+v, present = %v", node, ok)
		}
		time.Sleep(time.Nanosecond)
		synctest.Wait()
		if node, ok := d.dir.get(hostUUID); !ok || len(node.Models) != 1 || node.Models[0] != "refreshed-model" {
			t.Fatalf("inventory did not refresh on the configured tick: %+v, present = %v", node, ok)
		}
		cancel()
		<-done
		body = `{"models":["after-cancellation"]}`
		time.Sleep(timings.peerRefreshInterval)
		synctest.Wait()
		if node, ok := d.dir.get(hostUUID); !ok || len(node.Models) != 1 || node.Models[0] != "refreshed-model" {
			t.Fatalf("inventory refreshed after cancellation: %+v, present = %v", node, ok)
		}
	})
}
