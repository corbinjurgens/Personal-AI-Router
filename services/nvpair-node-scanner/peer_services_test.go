// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/clustertrusttest"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

func newPeerServicesScanner(t *testing.T) (*daemon, noderec.PeerServicesClaim) {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "cluster")
	clustertrusttest.Join(t, dir, "cluster", "self", "peer")
	writeScannerAdmissionPin(t, dir, "peer", "cluster", 1)
	d := newMembershipTestDaemon(dir)
	incarnation, ok := d.mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("fixture has no current incarnation")
	}
	claim := noderec.PeerServicesClaim{Incarnation: incarnation, Source: "cm-first", Sequence: 1, ObservedAt: time.Now().UnixMilli(), Descriptor: noderec.PeerServicesDescriptor{HostUUID: "peer", Name: "peer-name", Addresses: []string{"192.0.2.40"}, Services: []noderec.PeerService{{Service: noderec.ServiceCluster, Port: 14311}}}}
	return d, claim
}

func peerServicesNode(t *testing.T, d *daemon) noderec.DirectoryNode {
	t.Helper()
	node, ok := d.dir.get("peer")
	if !ok {
		t.Fatal("peer missing from directory")
	}
	return node
}

func TestPeerServicesBootstrapsAdmittedPeer(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("current authenticated claim rejected")
	}
	node := peerServicesNode(t, d)
	if node.HostUUID != "peer" || node.ClusterUUID != "peer" || !node.Trusted || node.Name != "peer-name" || node.IP != "192.0.2.40" || node.Services[noderec.ServiceCluster].Port != 14311 {
		t.Fatalf("bootstrap node = %+v", node)
	}
	if node.LastSeen != 0 {
		t.Fatalf("descriptor minted liveness timestamp %d", node.LastSeen)
	}
}

func TestPeerServicesRejectsInvalidAuthorityAndMetadata(t *testing.T) {
	test := func(name string, change func(*noderec.PeerServicesClaim)) {
		t.Run(name, func(t *testing.T) {
			d, claim := newPeerServicesScanner(t)
			change(&claim)
			if d.acceptPeerServices(claim) {
				t.Fatal("invalid claim accepted")
			}
			if nodes := d.dir.snapshot(""); len(nodes) != 0 {
				t.Fatalf("invalid claim changed directory: %+v", nodes)
			}
		})
	}
	test("third party identity", func(c *noderec.PeerServicesClaim) { c.Descriptor.HostUUID = "stranger" })
	test("wrong incarnation", func(c *noderec.PeerServicesClaim) { c.Incarnation = "obsolete" })
	test("loopback address", func(c *noderec.PeerServicesClaim) { c.Descriptor.Addresses = []string{"127.0.0.1"} })
	test("gateway hostname", func(c *noderec.PeerServicesClaim) { c.Descriptor.Addresses = []string{"localhost"} })
	test("unknown service", func(c *noderec.PeerServicesClaim) { c.Descriptor.Services[0].Service = "arbitrary" })
	test("invalid port", func(c *noderec.PeerServicesClaim) { c.Descriptor.Services[0].Port = 65536 })
	test("expired observation", func(c *noderec.PeerServicesClaim) { c.ObservedAt = time.Now().Add(-31 * time.Second).UnixMilli() })
	test("future observation", func(c *noderec.PeerServicesClaim) { c.ObservedAt = time.Now().Add(time.Minute).UnixMilli() })
	test("missing source", func(c *noderec.PeerServicesClaim) { c.Source = "" })
	test("zero sequence", func(c *noderec.PeerServicesClaim) { c.Sequence = 0 })
}

func TestPeerServicesReconcilesBrowseClaims(t *testing.T) {
	test := func(name string, browseFirst bool) {
		t.Run(name, func(t *testing.T) {
			d, claim := newPeerServicesScanner(t)
			d.lastActivityAt = make(map[string]time.Time)
			d.noteActivity("peer", 0)
			browse := DiscoveryEvent{Type: "discovered", Node: RawNode{ID: "browse-name", Addresses: []string{"192.0.2.90"}, TXT: []string{"uuid=peer", "cluster-uuid=peer", "cl=14444"}}}
			if browseFirst {
				d.onBrowse(browse)
			}
			if !d.acceptPeerServices(claim) {
				t.Fatal("claim rejected")
			}
			if _, ok := d.activitySince("peer"); !ok {
				t.Fatal("route replacement cleared independently observed activity")
			}
			browse.Type = "updated"
			d.onBrowse(browse)
			if node := peerServicesNode(t, d); node.Name != "peer-name" || node.Services[noderec.ServiceCluster].Port != 14311 {
				t.Fatalf("browse overwrote authenticated snapshot: %+v", node)
			}
			if err := os.Remove(filepath.Join(d.clusterDir, "trusted", "peer.json")); err != nil {
				t.Fatal(err)
			}
			d.reloadTrust()
			if node := peerServicesNode(t, d); node.Name != "browse-name" || node.Services[noderec.ServiceCluster].Port != 14444 || node.Trusted {
				t.Fatalf("retirement did not restore live browse: %+v", node)
			}
			if _, ok := d.activitySince("peer"); !ok {
				t.Fatal("surviving browse claim lost independently observed activity")
			}
			browse.Type = "removed"
			d.onBrowse(browse)
			if _, ok := d.dir.get("peer"); ok {
				t.Fatal("last claim removal retained peer")
			}
			if _, ok := d.activitySince("peer"); ok {
				t.Fatal("last claim removal retained activity cache")
			}
		})
	}
	test("browse before authenticated", true)
	test("authenticated before browse", false)
}

func TestPeerServicesBrowseRemovalPreservesAuthenticatedSnapshot(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim rejected")
	}
	d.onBrowse(DiscoveryEvent{Type: "removed", Node: RawNode{TXT: []string{"uuid=peer"}}})
	if node := peerServicesNode(t, d); node.Services[noderec.ServiceCluster].Port != 14311 {
		t.Fatalf("browse removal changed authenticated node: %+v", node)
	}
}

func TestPeerServicesCompleteSnapshotsOrderAndWithdrawServices(t *testing.T) {
	d, first := newPeerServicesScanner(t)
	if !d.acceptPeerServices(first) {
		t.Fatal("first snapshot rejected")
	}
	second := first
	second.Sequence = 2
	second.Descriptor.Services = []noderec.PeerService{}
	if !d.acceptPeerServices(second) {
		t.Fatal("withdrawal rejected")
	}
	if node := peerServicesNode(t, d); len(node.Services) != 0 {
		t.Fatalf("withdrawn services survived: %+v", node.Services)
	}
	if d.acceptPeerServices(first) {
		t.Fatal("older response restored withdrawn service")
	}
	third := first
	third.Sequence = 3
	third.Descriptor.Services = []noderec.PeerService{{Service: noderec.ServiceCluster, Port: 15000}}
	if !d.acceptPeerServices(third) {
		t.Fatal("port update rejected")
	}
	if node := peerServicesNode(t, d); node.Services[noderec.ServiceCluster].Port != 15000 {
		t.Fatalf("port update not applied: %+v", node.Services)
	}
	restarted := third
	restarted.Source = "cm-restarted"
	restarted.Sequence = 1
	if !d.acceptPeerServices(restarted) {
		t.Fatal("process restart sequence reset rejected")
	}
	if d.acceptPeerServices(third) {
		t.Fatal("old source replay accepted")
	}
}

func TestPeerServicesSameCertificateReadmissionRetiresClaim(t *testing.T) {
	d, old := newPeerServicesScanner(t)
	if !d.acceptPeerServices(old) {
		t.Fatal("claim rejected")
	}
	writeScannerAdmissionPin(t, d.clusterDir, "peer", "cluster", 2)
	d.reloadTrust()
	if _, ok := d.dir.get("peer"); ok {
		t.Fatal("same-certificate re-pair retained old claim")
	}
	if d.acceptPeerServices(old) {
		t.Fatal("old admission restored peer")
	}
	d.enrichPeerServices(old, noderec.DirectoryNode{HostUUID: "peer"}, d.dir.routeGeneration("peer"))
	if _, ok := d.dir.get("peer"); ok {
		t.Fatal("delayed old enrichment resurrected peer")
	}
	current, ok := d.mesh.PeerServiceIncarnation("peer")
	if !ok || current == old.Incarnation {
		t.Fatal("readmission did not replace authority")
	}
	old.Incarnation = current
	if !d.acceptPeerServices(old) {
		t.Fatal("fresh readmission claim rejected")
	}
}

func TestPeerServicesLocalLeaveRetiresClaimAndPendingApply(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim rejected")
	}
	generation := d.dir.routeGeneration("peer")
	clustertrusttest.WriteAdmission(t, d.clusterDir, "", 2)
	d.reloadTrust()
	if _, ok := d.dir.get("peer"); ok {
		t.Fatal("local leave retained authenticated metadata")
	}
	if d.acceptPeerServices(claim) {
		t.Fatal("local leave accepted old claim")
	}
	d.enrichPeerServices(claim, noderec.DirectoryNode{HostUUID: "peer"}, generation)
	if _, ok := d.dir.get("peer"); ok {
		t.Fatal("pending enrichment survived local leave")
	}
}

func TestPeerServicesExpiryRemovesOnlyAuthenticatedClaim(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim rejected")
	}
	d.peerServicesMu.Lock()
	d.dir.mu.Lock()
	expired := d.dir.peerServices["peer"]
	expired.ObservedAt = time.Now().Add(-31 * time.Second).UnixMilli()
	d.dir.peerServices["peer"] = expired
	d.dir.mu.Unlock()
	d.peerServicesMu.Unlock()
	d.reconcilePeerServices()
	if _, ok := d.dir.get("peer"); ok {
		t.Fatal("expired metadata remained routable")
	}
	if d.acceptPeerServices(claim) {
		t.Fatal("replay after retirement accepted")
	}
}

func TestPeerServicesPrivateMethodRejectsRequests(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	params, err := json.Marshal(claim)
	if err != nil {
		t.Fatal(err)
	}
	id := json.RawMessage(`1`)
	request := &Message{Method: "discovery:peer-services", Params: params, ID: &id}
	if !d.handle(request) {
		t.Fatal("private method not consumed")
	}
	if _, ok := d.dir.get("peer"); ok {
		t.Fatal("ordinary request installed authority")
	}
	request.ID = nil
	if !d.handle(request) {
		t.Fatal("notification not consumed")
	}
	peerServicesNode(t, d)
}

func TestPeerServicesUnchangedSnapshotPreservesEnrichmentAndGeneration(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim rejected")
	}
	d.dir.mu.Lock()
	node := d.dir.nodes["peer"]
	node.Models = []string{"existing-model"}
	service := node.Services[noderec.ServiceCluster]
	service.Probe = noderec.ProbeStatus("reachable")
	node.Services[noderec.ServiceCluster] = service
	d.dir.nodes["peer"] = node
	d.dir.mu.Unlock()
	generation := d.dir.routeGeneration("peer")
	claim.Sequence++
	if !d.acceptPeerServices(claim) {
		t.Fatal("unchanged refresh rejected")
	}
	node = peerServicesNode(t, d)
	if generation != d.dir.routeGeneration("peer") || len(node.Models) != 1 || node.Models[0] != "existing-model" || node.Services[noderec.ServiceCluster].Probe != noderec.ProbeStatus("reachable") {
		t.Fatalf("unchanged metadata reset route or enrichment: generation %d -> %d, node %+v", generation, d.dir.routeGeneration("peer"), node)
	}
}

type peerServicesTestTransport func(*http.Request) (*http.Response, error)

func (f peerServicesTestTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func peerServicesEnrichmentClaim(t *testing.T, f *scannerReverseFixture) noderec.PeerServicesClaim {
	t.Helper()
	f.node.IP = "192.0.2.40"
	if _, existed := f.d.dir.removeBrowse("peer"); !existed {
		t.Fatal("fixture peer had no browse record to remove")
	}
	incarnation, ok := f.d.mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("fixture has no current incarnation")
	}
	return noderec.PeerServicesClaim{Incarnation: incarnation, Source: "cm-first", Sequence: 1, ObservedAt: time.Now().UnixMilli(),
		Descriptor: noderec.PeerServicesDescriptor{HostUUID: "peer", Name: "peer-name", Addresses: []string{"192.0.2.40"}, Services: []noderec.PeerService{
			{Service: noderec.ServiceNodeInfo, Port: f.node.Services[noderec.ServiceNodeInfo].Port},
			{Service: noderec.ServiceEngineManager, Port: f.node.Services[noderec.ServiceEngineManager].Port},
		}}}
}

func waitPeerServicesCondition(t *testing.T, description string, condition func() bool) {
	t.Helper()
	deadline := time.After(5 * time.Second)
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	for {
		if condition() {
			return
		}
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for %s", description)
		case <-tick.C:
		}
	}
}

func TestPeerServicesRetriesFullEnrichmentAfterFirstNodeInfoFailure(t *testing.T) {
	f := newScannerReverseFixture(t)
	claim := peerServicesEnrichmentClaim(t, f)
	var calls atomic.Int32
	f.d.tlsHTTP = &http.Client{Transport: peerServicesTestTransport(func(*http.Request) (*http.Response, error) {
		if calls.Add(1) == 1 {
			return &http.Response{StatusCode: http.StatusServiceUnavailable, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(""))}, nil
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"hostUuid":"peer","cpu":{"name":"recovered CPU","cores":8}}`))}, nil
	})}
	if !f.d.acceptPeerServices(claim) {
		t.Fatal("first claim rejected")
	}
	waitPeerServicesCondition(t, "successful first model fetch and failed node-info fetch", func() bool {
		f.d.peerServicesMu.Lock()
		busy := f.d.peerServicesEnriching["peer"]
		f.d.peerServicesMu.Unlock()
		return !busy && calls.Load() == 1 && f.modelCalls.Load() == 1
	})
	if node := peerServicesNode(t, f.d); node.CPU != nil || len(node.Models) != 1 || node.Models[0] != "model" {
		t.Fatalf("first enrichment = CPU %+v, models %v", node.CPU, node.Models)
	}
	if _, ok := f.d.lastInfoAt["peer"]; ok {
		t.Fatal("failed node-info fetch recorded success")
	}
	generation := f.d.dir.routeGeneration("peer")
	claim.Sequence++
	claim.ObservedAt = time.Now().UnixMilli()
	if !f.d.acceptPeerServices(claim) {
		t.Fatal("same-route retry claim rejected")
	}
	waitPeerServicesCondition(t, "recovered node-info fields", func() bool {
		f.d.infoMu.Lock()
		at := f.d.lastInfoAt["peer"]
		f.d.infoMu.Unlock()
		node := peerServicesNode(t, f.d)
		return !at.IsZero() && node.CPU != nil && node.CPU.Name == "recovered CPU"
	})
	if got := f.d.dir.routeGeneration("peer"); got != generation {
		t.Fatalf("retry changed route generation %d to %d", generation, got)
	}
}

func TestPeerServicesClaimsKeepOneEnrichmentInFlight(t *testing.T) {
	f := newScannerReverseFixture(t)
	claim := peerServicesEnrichmentClaim(t, f)
	started, release := make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	f.d.tlsHTTP = &http.Client{Transport: peerServicesTestTransport(func(r *http.Request) (*http.Response, error) {
		if calls.Add(1) == 1 {
			close(started)
		}
		select {
		case <-release:
		case <-r.Context().Done():
			return nil, r.Context().Err()
		}
		return &http.Response{StatusCode: http.StatusServiceUnavailable, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(""))}, nil
	})}
	if !f.d.acceptPeerServices(claim) {
		t.Fatal("first claim rejected")
	}
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("first node-info fetch did not start")
	}
	for range 2 {
		time.Sleep(2 * time.Second)
		claim.Sequence++
		claim.ObservedAt = time.Now().UnixMilli()
		if !f.d.acceptPeerServices(claim) {
			t.Fatal("later claim rejected")
		}
		if got := calls.Load(); got != 1 {
			t.Fatalf("%d node-info fetches overlapped", got)
		}
	}
	once.Do(func() { close(release) })
}

func TestPeerServicesNewIncarnationRejectsOldEnrichmentResult(t *testing.T) {
	f := newScannerReverseFixture(t)
	old := peerServicesEnrichmentClaim(t, f)
	started, release := make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	f.d.tlsHTTP = &http.Client{Transport: peerServicesTestTransport(func(r *http.Request) (*http.Response, error) {
		call := calls.Add(1)
		if call == 1 {
			close(started)
			select {
			case <-release:
			case <-r.Context().Done():
				return nil, r.Context().Err()
			}
		}
		name := "new CPU"
		if call == 1 {
			name = "obsolete CPU"
		}
		body, err := json.Marshal(NodeInfoResponse{HostUUID: "peer", CPU: &CPUInfo{Name: name, Cores: 8}})
		if err != nil {
			return nil, err
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(string(body)))}, nil
	})}
	if !f.d.acceptPeerServices(old) {
		t.Fatal("old claim rejected")
	}
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("old fetch did not start")
	}
	writeScannerAdmissionPin(t, f.d.clusterDir, "peer", "cluster", 2)
	f.d.reloadTrust()
	current, ok := f.d.mesh.PeerServiceIncarnation("peer")
	if !ok || current == old.Incarnation {
		t.Fatal("admission incarnation did not change")
	}
	claim := old
	claim.Incarnation = current
	claim.ObservedAt = time.Now().UnixMilli()
	if !f.d.acceptPeerServices(claim) {
		t.Fatal("new incarnation claim rejected")
	}
	once.Do(func() { close(release) })
	waitPeerServicesCondition(t, "obsolete enrichment completion", func() bool {
		f.d.peerServicesMu.Lock()
		busy := f.d.peerServicesEnriching["peer"]
		f.d.peerServicesMu.Unlock()
		return !busy
	})
	if node := peerServicesNode(t, f.d); node.CPU != nil {
		t.Fatalf("obsolete CPU published into new incarnation: %+v", node.CPU)
	}
	f.d.infoMu.Lock()
	_, cached := f.d.lastInfoAt["peer"]
	f.d.infoMu.Unlock()
	if cached {
		t.Fatal("obsolete node-info result cached for new incarnation")
	}
	claim.Sequence++
	claim.ObservedAt = time.Now().UnixMilli()
	if !f.d.acceptPeerServices(claim) {
		t.Fatal("new incarnation retry claim rejected")
	}
	waitPeerServicesCondition(t, "new incarnation enrichment", func() bool {
		node := peerServicesNode(t, f.d)
		return node.CPU != nil && node.CPU.Name == "new CPU"
	})
}

func TestPeerServicesHeldEnrichmentCannotRestoreWithdrawnMetadataOrCaches(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	claim.Descriptor.Services = []noderec.PeerService{{Service: noderec.ServiceNodeInfo, Port: 14318}}
	node := noderec.DirectoryNode{HostUUID: "peer", Name: "peer-name", ClusterUUID: "peer", IP: "192.0.2.40", IPs: []string{"192.0.2.40"}, Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 14318}}}
	d.dir.installPeerServices(claim, node)
	started, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	releaseFetch := func() { once.Do(func() { close(release) }) }
	t.Cleanup(releaseFetch)
	d.tlsHTTP = &http.Client{Transport: peerServicesTestTransport(func(r *http.Request) (*http.Response, error) {
		close(started)
		select {
		case <-release:
		case <-r.Context().Done():
			return nil, r.Context().Err()
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"hostUuid":"peer","GPUs":[{"name":"obsolete GPU"}]}`))}, nil
	})}
	generation := d.dir.routeGeneration("peer")
	go func() { d.enrichPeerServices(claim, node, generation); close(done) }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("enrichment did not begin")
	}
	withdrawal := claim
	withdrawal.Sequence++
	withdrawal.Descriptor.Services = []noderec.PeerService{}
	if !d.acceptPeerServices(withdrawal) {
		t.Fatal("withdrawal rejected while response held")
	}
	d.infoMu.Lock()
	d.lastInfo["peer"] = NodeInfoResponse{GPUs: []GPUInfo{{Name: "new cache"}}}
	d.infoMu.Unlock()
	releaseFetch()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("held enrichment did not complete")
	}
	if current := peerServicesNode(t, d); len(current.Services) != 0 || len(current.GPUs) != 0 {
		t.Fatalf("held response restored withdrawn metadata: %+v", current)
	}
	d.infoMu.Lock()
	cached := d.lastInfo["peer"]
	d.infoMu.Unlock()
	if len(cached.GPUs) != 1 || cached.GPUs[0].Name != "new cache" {
		t.Fatalf("obsolete completion changed current cache: %+v", cached)
	}
}

// cachedPolicy is the last-good policy advertisement cached for hostUUID.
func cachedPolicy(d *daemon, hostUUID string) *nodepolicy.Advertisement {
	d.infoMu.Lock()
	defer d.infoMu.Unlock()
	return d.lastPolicy[hostUUID]
}

// TestPinnedFetchCarriesPolicyAndRetainsItOnFailure follows a browsed peer's
// policy advertisement through the pinned mTLS /v1/models fetch, and checks a
// failed fetch keeps the last good value rather than making the peer look
// undeclared.
func TestPinnedFetchCarriesPolicyAndRetainsItOnFailure(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	node := f.node
	f.d.enrich(&node)
	if !reflect.DeepEqual(node.Policy, scannerFixturePolicy) {
		t.Fatalf("enriched policy = %+v, want %+v", node.Policy, scannerFixturePolicy)
	}
	if got := cachedPolicy(f.d, "peer"); !reflect.DeepEqual(got, scannerFixturePolicy) {
		t.Fatalf("cached policy = %+v, want %+v", got, scannerFixturePolicy)
	}

	f.direct.Store(false)
	f.reverse.Store(false)
	node = f.node
	f.d.enrich(&node)
	if !reflect.DeepEqual(node.Policy, scannerFixturePolicy) {
		t.Fatalf("policy after a failed fetch = %+v, want the last good %+v", node.Policy, scannerFixturePolicy)
	}
}

// TestPeerServicesBootstrapEnrichmentCarriesPolicy covers a peer known only
// from its authenticated claim, with no mDNS record: enrichPeerServices reuses
// the same pinned fetch and records the advertisement on the node and in the
// last-good cache.
func TestPeerServicesBootstrapEnrichmentCarriesPolicy(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	if _, existed := f.d.dir.removeBrowse("peer"); !existed {
		t.Fatal("fixture peer had no browse record to remove")
	}
	incarnation, ok := f.d.mesh.PeerServiceIncarnation("peer")
	if !ok {
		t.Fatal("fixture has no current incarnation")
	}
	claim := noderec.PeerServicesClaim{Incarnation: incarnation, Source: "cm-first", Sequence: 1, ObservedAt: time.Now().UnixMilli(),
		Descriptor: noderec.PeerServicesDescriptor{HostUUID: "peer", Name: "peer-name", Addresses: []string{"192.0.2.40"}}}
	// Installed directly, like the held-enrichment test: the descriptor
	// validation refuses the loopback address the fixture serves on.
	f.d.dir.installPeerServices(claim, f.node)
	if !f.d.dir.authenticated("peer") {
		t.Fatal("peer is not held by an authenticated claim")
	}
	f.d.enrichPeerServices(claim, f.node, f.d.dir.routeGeneration("peer"))
	node := peerServicesNode(t, f.d)
	if !reflect.DeepEqual(node.Policy, scannerFixturePolicy) || !reflect.DeepEqual(node.Models, []string{"model"}) {
		t.Fatalf("bootstrap enrichment = policy %+v models %v, want %+v and [model]", node.Policy, node.Models, scannerFixturePolicy)
	}
	if got := cachedPolicy(f.d, "peer"); !reflect.DeepEqual(got, scannerFixturePolicy) {
		t.Fatalf("cached policy = %+v, want %+v", got, scannerFixturePolicy)
	}
}
