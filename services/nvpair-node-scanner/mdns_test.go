// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/mdns"
	"nvpair-shared/noderec"
)

func newMDNSDisabledDaemon(t *testing.T) *daemon {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "cluster")
	scanner, err := NewScanner(nil, clustertrust.Open(dir), dir, nil, false, noderec.MDNSNamespace{Service: noderec.ServiceType, Domain: noderec.Domain})
	if err != nil {
		t.Fatal(err)
	}
	return scanner.daemon
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
	scanner, err := NewScanner(nil, clustertrust.Open(dir), dir, nil, true, noderec.MDNSNamespace{Service: noderec.ServiceType, Domain: noderec.Domain})
	if err != nil {
		t.Fatal(err)
	}
	d := scanner.daemon
	if d.browser == nil {
		t.Fatal("mDNS enabled scanner did not construct a browser")
	}
	// NewResponder only enumerates interfaces; Run and UpdateTXT send traffic.
	// Compare availability without starting either discovery component, including
	// on hosts without multicast-capable interfaces where construction must fail.
	_, err = mdns.NewResponder(d.instance, noderec.ServiceType, noderec.Domain, noderec.SRVPort, d.reg.txt())
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
	scanner, err := NewScanner(nil, fixture.mesh, fixture.clusterDir, nil, false, noderec.MDNSNamespace{Service: noderec.ServiceType, Domain: noderec.Domain})
	if err != nil {
		t.Fatal(err)
	}
	d := scanner.daemon
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

func TestScannerNamespaceReachesBrowserAndResponder(t *testing.T) {
	namespace := noderec.MDNSNamespace{Service: "_PAIR-Fixture._TCP.", Domain: "LOCAL."}
	dir := filepath.Join(t.TempDir(), "cluster")
	scanner, err := NewScanner(nil, clustertrust.Open(dir), dir, nil, true, namespace)
	if err != nil {
		t.Fatal(err)
	}
	d := scanner.daemon
	// Inspect the constructed owners without running them or emitting multicast.
	for field, want := range map[string]string{"service": "_pair-fixture._tcp", "domain": noderec.Domain} {
		if got := reflect.ValueOf(d.browser).Elem().FieldByName(field).String(); got != want {
			t.Fatalf("browser %s = %q, want %q", field, got, want)
		}
	}
	if d.responder == nil {
		if _, err := mdns.NewResponder(d.instance, "_pair-fixture._tcp", noderec.Domain, noderec.SRVPort, nil); err == nil {
			t.Fatal("configured responder missing despite available interfaces")
		}
		t.Skip("responder construction unavailable without a multicast-capable interface; browser checked")
	}
	for field, want := range map[string]string{
		"serviceName":  "_pair-fixture._tcp.local.",
		"instanceName": strings.Trim(d.instance, ".") + "._pair-fixture._tcp.local.",
		"hostName":     strings.Trim(d.instance, ".") + ".local.",
	} {
		if got := reflect.ValueOf(d.responder).Elem().FieldByName(field).String(); got != want {
			t.Fatalf("responder %s = %q, want %q", field, got, want)
		}
	}
}

func TestScannerRejectsNamespaceBeforeIdentityEffects(t *testing.T) {
	test := func(name string, enabled bool, namespace noderec.MDNSNamespace) {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "must-not-exist", "cluster")
			scanner, err := NewScanner(nil, nil, dir, nil, enabled, namespace)
			if err == nil || scanner != nil {
				t.Fatalf("invalid namespace constructed scanner: %v, %v", scanner, err)
			}
			if _, err := os.Stat(filepath.Dir(dir)); !os.IsNotExist(err) {
				t.Fatalf("invalid namespace touched identity directory: %v", err)
			}
		})
	}
	test("missing domain with multicast disabled", false, noderec.MDNSNamespace{Service: noderec.ServiceType})
	test("missing domain with multicast enabled", true, noderec.MDNSNamespace{Service: noderec.ServiceType})
	test("custom domain with multicast disabled", false, noderec.MDNSNamespace{Service: "_pair-fixture._tcp", Domain: "fixture.local"})
	test("custom domain with multicast enabled", true, noderec.MDNSNamespace{Service: "_pair-fixture._tcp", Domain: "fixture.local"})
}

func TestMDNSDisabledRefreshesInventory(t *testing.T) {
	d := newMDNSDisabledDaemon(t)
	requireMDNSDisabled(t, d)
	hostUUID := d.reg.record().HostUUID
	d.reg.setIdentity(hostUUID, "", []string{"192.0.2.10"})
	d.reg.register(noderec.RegisterParams{Service: noderec.ServiceEngineManager, Port: 18080})
	body := `{"models":["initial-model"]}`
	d.modelsHTTP = &http.Client{Transport: peerServicesTestTransport(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodGet || r.URL.Scheme != "http" || r.URL.Host != "127.0.0.1:18080" || r.URL.Path != "/v1/models" {
			return nil, fmt.Errorf("unexpected self inventory request: %s %s", r.Method, r.URL)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	d.publishSelf()
	if node, ok := d.dir.get(hostUUID); !ok || len(node.Models) != 1 || node.Models[0] != "initial-model" {
		t.Fatalf("initial self inventory = %+v, present = %v", node, ok)
	}
	body = `{"models":["refreshed-model"]}`
	// Drive the unchanged periodic sweep body without waiting its 15s interval.
	d.refreshModelsOnce()
	if node, ok := d.dir.get(hostUUID); !ok || len(node.Models) != 1 || node.Models[0] != "refreshed-model" {
		t.Fatalf("refreshed self inventory = %+v, present = %v", node, ok)
	}
}
