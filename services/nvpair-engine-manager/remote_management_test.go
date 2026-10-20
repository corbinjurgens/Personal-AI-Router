// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// remote_management_test.go drives the ec authorization gate, the route
// registry, the caller-only permission read and the typed -32012 refusal.
//
// The permission store is written on disk in a t.TempDir cluster dir exactly as
// services/shared/clustertrust expects, and every request in the route-matrix
// tests travels the real pin gate over loopback mTLS so identity, method
// validation and the read/mutation decision are exercised together. No product
// binary, engine or network service is started.

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/noderec"
)

const (
	rmClusterID = "cluster-pg2"
	rmEpoch     = 3
)

// rmPeer is one pinned peer of the target fixture: its own cluster dir holds the
// keypair the test dials with.
type rmPeer struct {
	uuid string
	dir  string
}

// rmFixture is one target node's cluster directory plus the peers it pins. The
// permission document/marker and the peer's reverse pin live on disk, so
// Mesh.RemoteManagement and Mesh.VerifyClientPin read the same records they read
// in production.
type rmFixture struct {
	t        *testing.T
	dir      string
	selfUUID string
	peers    []rmPeer
	mesh     *clustertrust.Mesh
}

func rmNewFixture(t *testing.T, selfUUID string, peerUUIDs ...string) *rmFixture {
	t.Helper()
	dir := t.TempDir()
	clustertrusttest.WriteKeypair(t, dir, selfUUID)
	clustertrusttest.WriteAdmission(t, dir, rmClusterID, rmEpoch)
	f := &rmFixture{t: t, dir: dir, selfUUID: selfUUID}
	for _, uuid := range peerUUIDs {
		peerDir := t.TempDir()
		clustertrusttest.WriteKeypair(t, peerDir, uuid)
		clustertrusttest.WriteAdmission(t, peerDir, rmClusterID, rmEpoch)
		rmWritePin(t, dir, uuid, filepath.Join(peerDir, "node.crt"))
		// The reverse pin lets the peer verify this node's client certificate,
		// which the outbound-RPC test needs.
		rmWritePin(t, peerDir, selfUUID, filepath.Join(dir, "node.crt"))
		f.peers = append(f.peers, rmPeer{uuid: uuid, dir: peerDir})
	}
	rmWriteMembers(t, dir, rmClusterID, rmEpoch, peerUUIDs)
	f.mesh = clustertrust.Open(dir)
	if !f.mesh.Clustered() {
		t.Fatal("fixture target must read as clustered")
	}
	return f
}

func rmWritePin(t *testing.T, dir, uuid, certPath string) {
	t.Helper()
	certPEM, err := os.ReadFile(certPath)
	if err != nil {
		t.Fatal(err)
	}
	admission := rmReadAdmission(t, dir)
	trusted := filepath.Join(dir, "trusted")
	if err := os.MkdirAll(trusted, 0o700); err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{
		"nodeUuid": uuid, "certPem": string(certPEM),
		"clusterId": admission.ClusterID, "admissionEpoch": admission.Epoch,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(trusted, uuid+".json"), body, 0o600); err != nil {
		t.Fatal(err)
	}
}

func rmWriteMembers(t *testing.T, dir, clusterID string, epoch uint64, uuids []string) {
	t.Helper()
	members := make([]map[string]any, 0, len(uuids))
	for _, uuid := range uuids {
		members = append(members, map[string]any{
			"nodeUuid": uuid, "state": "member", "clusterId": clusterID, "admissionEpoch": epoch,
		})
	}
	body, err := json.Marshal(members)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "members.json"), body, 0o600); err != nil {
		t.Fatal(err)
	}
}

func rmReadAdmission(t *testing.T, dir string) struct {
	ClusterID string `json:"clusterId"`
	Epoch     uint64 `json:"epoch"`
} {
	t.Helper()
	var admission struct {
		ClusterID string `json:"clusterId"`
		Epoch     uint64 `json:"epoch"`
	}
	data, err := os.ReadFile(filepath.Join(dir, "admission.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &admission); err != nil {
		t.Fatal(err)
	}
	return admission
}

func (f *rmFixture) peer(uuid string) rmPeer {
	f.t.Helper()
	for _, p := range f.peers {
		if p.uuid == uuid {
			return p
		}
	}
	f.t.Fatalf("fixture has no peer %q", uuid)
	return rmPeer{}
}

func (f *rmFixture) certPEM(uuid string) []byte {
	f.t.Helper()
	data, err := os.ReadFile(filepath.Join(f.peer(uuid).dir, "node.crt"))
	if err != nil {
		f.t.Fatal(err)
	}
	return data
}

// serverDER is the target's own leaf, the only server certificate a test client
// accepts.
func (f *rmFixture) serverDER() []byte {
	f.t.Helper()
	id, err := clustertrust.LoadIdentity(f.dir)
	if err != nil {
		f.t.Fatal(err)
	}
	if len(id.Cert.Certificate) == 0 {
		f.t.Fatal("fixture target has no leaf certificate")
	}
	return id.Cert.Certificate[0]
}

// client dials the fixture's target as uuid, presenting that node's leaf. The
// target's own leaf is the only accepted server certificate.
func (f *rmFixture) client(uuid string) *http.Client {
	f.t.Helper()
	dir := f.dir
	if uuid != f.selfUUID {
		dir = f.peer(uuid).dir
	}
	id, err := clustertrust.LoadIdentity(dir)
	if err != nil {
		f.t.Fatal(err)
	}
	return &http.Client{Transport: &http.Transport{
		TLSClientConfig: clustertrust.ClientTLSConfig(id.Cert, f.serverDER()),
	}}
}

func rmFingerprint(der []byte) string {
	return fmt.Sprintf("sha256:%x", sha256.Sum256(der))
}

func (f *rmFixture) snapshot() clustertrust.RemoteManagementMigrationSnapshot {
	f.t.Helper()
	peers := make([]clustertrust.RemoteManagementMigrationAuthority, 0, len(f.peers))
	for _, p := range f.peers {
		peers = append(peers, clustertrust.RemoteManagementMigrationAuthority{
			PeerUUID: p.uuid, CertFingerprint: rmFingerprint(f.certPEM(p.uuid)),
			ClusterID: rmClusterID, AdmissionEpoch: rmEpoch,
		})
	}
	return clustertrust.RemoteManagementMigrationSnapshot{
		Local: clustertrust.RemoteManagementMigrationAuthority{
			PeerUUID: f.selfUUID, CertFingerprint: rmFingerprint(f.serverDER()),
			ClusterID: rmClusterID, AdmissionEpoch: rmEpoch,
		},
		Peers: peers,
	}
}

func (f *rmFixture) writeJSON(name string, value any) {
	rmWriteClusterJSON(f.t, f.dir, name, value)
}

// rmWriteClusterJSON lands one permission-store record in a cluster dir.
func rmWriteClusterJSON(t *testing.T, dir, name string, value any) {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode %s: %v", name, err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), body, 0o600); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

func (f *rmFixture) writeRaw(name string, body []byte) {
	f.t.Helper()
	if err := os.WriteFile(filepath.Join(f.dir, name), body, 0o600); err != nil {
		f.t.Fatal(err)
	}
}

// grantPin is one pinned peer as the permission fixture reads it back: the
// principal its pin file is named for and the certificate it pins.
type grantPin struct {
	uuid    string
	certPEM []byte
}

// grantMember is the members.json record the incarnation read pairs with a pin.
type grantMember struct {
	NodeUUID       string `json:"nodeUuid"`
	State          string `json:"state"`
	ClusterID      string `json:"clusterId"`
	AdmissionEpoch uint64 `json:"admissionEpoch"`
}

// grantRemoteManagement lands the complete granted permission state that the ec
// mutation gate reads in dir — the target node's cluster dir. allowed names the
// pinned peers and whether each may mutate this target, recorded at revision.
//
// A grant is pinned to the peer's current service incarnation, which the mesh
// reads from the live admission, the cluster-scoped pin and the members roster,
// so a fixture that only pinned a peer is completed here: missing records are
// written, held records are kept, and the result is verified through
// Mesh.RemoteManagement before the test continues, so an incomplete fixture
// fails at this call instead of at a later assertion. This is the one
// grant-fixture helper: rmFixture.writePermissions and the fixtures in the
// policy, settings, local-veto, pull and reverse tests all go through it.
func grantRemoteManagement(t *testing.T, dir, revision string, allowed map[string]bool) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(dir, "admission.json")); err != nil {
		// The incarnation read starts at a live admission, which a fixture
		// that pinned a peer on the strength of its pin alone never wrote.
		clustertrusttest.WriteAdmission(t, dir, "cluster-1", 1)
	}
	admission := rmReadAdmission(t, dir)
	if admission.Epoch == 0 {
		t.Fatalf("permission fixture target %s has no active admission epoch", dir)
	}
	peers := grantPinnedPeers(t, dir, admission.ClusterID, admission.Epoch)
	granted := make([]string, 0, len(allowed))
	for _, peer := range peers {
		if _, ok := allowed[peer.uuid]; ok {
			granted = append(granted, peer.uuid)
		}
	}
	if len(granted) != len(allowed) {
		t.Fatalf("permission fixture target %s does not pin every granted peer: %v", dir, allowed)
	}
	grantRoster(t, dir, granted, admission.ClusterID, admission.Epoch)

	mesh := clustertrust.Open(dir)
	selfUUID := mesh.NodeUUID()
	if selfUUID == "" {
		t.Fatalf("permission fixture target %s has no cluster identity", dir)
	}
	selfCert, err := os.ReadFile(filepath.Join(dir, "node.crt"))
	if err != nil {
		t.Fatalf("read permission fixture target certificate: %v", err)
	}
	// The snapshot's fingerprints are hashes of the stored certificates: the
	// permission read checks their shape, never their value.
	snapshot := clustertrust.RemoteManagementMigrationSnapshot{
		Local: clustertrust.RemoteManagementMigrationAuthority{
			PeerUUID: selfUUID, CertFingerprint: rmFingerprint(selfCert),
			ClusterID: admission.ClusterID, AdmissionEpoch: admission.Epoch,
		},
		Peers: make([]clustertrust.RemoteManagementMigrationAuthority, 0, len(peers)),
	}
	for _, peer := range peers {
		snapshot.Peers = append(snapshot.Peers, clustertrust.RemoteManagementMigrationAuthority{
			PeerUUID: peer.uuid, CertFingerprint: rmFingerprint(peer.certPEM),
			ClusterID: admission.ClusterID, AdmissionEpoch: admission.Epoch,
		})
	}
	migrationID, err := clustertrust.RemoteManagementMigrationID(snapshot)
	if err != nil {
		t.Fatalf("permission fixture migration snapshot: %v", err)
	}
	grants := make([]clustertrust.RemoteManagementGrant, 0, len(granted))
	for _, uuid := range granted {
		incarnation, ok := mesh.PeerServiceIncarnation(uuid)
		if !ok {
			t.Fatalf("permission fixture target %s has no current incarnation for %s", dir, uuid)
		}
		grants = append(grants, clustertrust.RemoteManagementGrant{
			PeerUUID: uuid, Incarnation: incarnation, Allowed: allowed[uuid], DecisionRevision: revision,
		})
	}
	rmWriteClusterJSON(t, dir, clustertrust.RemoteManagementMigrationFile, clustertrust.RemoteManagementMigration{
		Version: 1, ID: migrationID, State: "complete", Snapshot: snapshot,
	})
	rmWriteClusterJSON(t, dir, clustertrust.RemoteManagementFile, clustertrust.RemoteManagementDocument{
		Version: 1, Revision: revision, MigrationID: migrationID, Grants: grants,
	})
	for _, grant := range grants {
		permission := mesh.RemoteManagement(grant.PeerUUID)
		switch {
		case grant.Allowed && !permission.Allowed:
			t.Fatalf("permission fixture did not grant %s: state %q", grant.PeerUUID, permission.State)
		case !grant.Allowed && permission.State != "denied":
			t.Fatalf("permission fixture did not record a denial for %s: state %q", grant.PeerUUID, permission.State)
		}
	}
}

// grantPinnedPeers reads dir's pins in directory order, giving each pin the
// cluster scope the incarnation read requires: a fixture that pinned a peer
// before it had an active admission carries only the principal and the
// certificate. A pin scoped to another admission fails the fixture rather than
// being rewritten, so a roster that disagrees with the pins is never papered
// over.
func grantPinnedPeers(t *testing.T, dir, clusterID string, epoch uint64) []grantPin {
	t.Helper()
	trusted := filepath.Join(dir, "trusted")
	entries, err := os.ReadDir(trusted)
	if err != nil {
		t.Fatalf("read the pins of %s: %v", dir, err)
	}
	peers := make([]grantPin, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		uuid := strings.TrimSuffix(entry.Name(), ".json")
		path := filepath.Join(trusted, entry.Name())
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read pin %s: %v", entry.Name(), err)
		}
		var record struct {
			NodeUUID       string `json:"nodeUuid"`
			CertPEM        string `json:"certPem"`
			ClusterID      string `json:"clusterId"`
			AdmissionEpoch uint64 `json:"admissionEpoch"`
		}
		if err := json.Unmarshal(body, &record); err != nil {
			t.Fatalf("decode pin %s: %v", entry.Name(), err)
		}
		if record.NodeUUID != uuid || record.CertPEM == "" {
			t.Fatalf("pin %s in %s does not pin its own principal", entry.Name(), dir)
		}
		if record.ClusterID == "" || record.AdmissionEpoch == 0 {
			scoped, err := json.Marshal(map[string]any{
				"nodeUuid": record.NodeUUID, "certPem": record.CertPEM,
				"clusterId": clusterID, "admissionEpoch": epoch,
			})
			if err != nil {
				t.Fatalf("encode scoped pin %s: %v", entry.Name(), err)
			}
			if err := os.WriteFile(path, scoped, 0o600); err != nil {
				t.Fatalf("write scoped pin %s: %v", entry.Name(), err)
			}
		} else if record.ClusterID != clusterID || record.AdmissionEpoch != epoch {
			t.Fatalf("pin %s in %s records admission %s/%d, not this target's %s/%d",
				entry.Name(), dir, record.ClusterID, record.AdmissionEpoch, clusterID, epoch)
		}
		peers = append(peers, grantPin{uuid: uuid, certPEM: []byte(record.CertPEM)})
	}
	return peers
}

// grantRoster records every granted peer as a current member of the target's
// cluster at the pin's admission epoch — the record the incarnation read pairs
// with the pin — and keeps the records the fixture already holds.
func grantRoster(t *testing.T, dir string, uuids []string, clusterID string, epoch uint64) {
	t.Helper()
	path := filepath.Join(dir, "members.json")
	members := []grantMember{}
	if data, err := os.ReadFile(path); err == nil {
		if err := json.Unmarshal(data, &members); err != nil {
			t.Fatalf("decode the members roster of %s: %v", dir, err)
		}
	}
	held := make(map[string]grantMember, len(members))
	for _, member := range members {
		held[member.NodeUUID] = member
	}
	added := false
	for _, uuid := range uuids {
		member := grantMember{NodeUUID: uuid, State: "member", ClusterID: clusterID, AdmissionEpoch: epoch}
		switch current, ok := held[uuid]; {
		case !ok:
			members = append(members, member)
			added = true
		case current != member:
			t.Fatalf("the members roster of %s records %s as %+v, not a member of %s at epoch %d",
				dir, uuid, current, clusterID, epoch)
		}
	}
	if !added {
		return
	}
	body, err := json.Marshal(members)
	if err != nil {
		t.Fatalf("encode the members roster of %s: %v", dir, err)
	}
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatalf("write the members roster of %s: %v", dir, err)
	}
}

// writePermissions is grantRemoteManagement against this fixture's own target
// dir, so the grant fixture has one implementation no matter which test file
// built the cluster directory.
func (f *rmFixture) writePermissions(revision string, allowed map[string]bool) {
	grantRemoteManagement(f.t, f.dir, revision, allowed)
}

// writeGrantWithIncarnation lands a grant whose incarnation can never match the
// mesh's current token.
func (f *rmFixture) writeGrantWithIncarnation(revision, incarnation string, allowed bool, peerUUID string) {
	f.t.Helper()
	snapshot := f.snapshot()
	id, err := clustertrust.RemoteManagementMigrationID(snapshot)
	if err != nil {
		f.t.Fatal(err)
	}
	f.writeJSON(clustertrust.RemoteManagementMigrationFile, clustertrust.RemoteManagementMigration{
		Version: 1, ID: id, State: "complete", Snapshot: snapshot,
	})
	f.writeJSON(clustertrust.RemoteManagementFile, clustertrust.RemoteManagementDocument{
		Version: 1, Revision: revision, MigrationID: id,
		Grants: []clustertrust.RemoteManagementGrant{{
			PeerUUID: peerUUID, Incarnation: incarnation, Allowed: allowed, DecisionRevision: revision,
		}},
	})
}

func (f *rmFixture) setMarkerState(state string) {
	f.t.Helper()
	var marker clustertrust.RemoteManagementMigration
	data, err := os.ReadFile(filepath.Join(f.dir, clustertrust.RemoteManagementMigrationFile))
	if err != nil {
		f.t.Fatal(err)
	}
	if err := json.Unmarshal(data, &marker); err != nil {
		f.t.Fatal(err)
	}
	marker.State = state
	f.writeJSON(clustertrust.RemoteManagementMigrationFile, marker)
}

func (f *rmFixture) setDocumentMigrationID(id string) {
	f.t.Helper()
	var doc clustertrust.RemoteManagementDocument
	data, err := os.ReadFile(filepath.Join(f.dir, clustertrust.RemoteManagementFile))
	if err != nil {
		f.t.Fatal(err)
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		f.t.Fatal(err)
	}
	doc.MigrationID = id
	f.writeJSON(clustertrust.RemoteManagementFile, doc)
}

func (f *rmFixture) removeStore() {
	f.t.Helper()
	for _, name := range []string{clustertrust.RemoteManagementFile, clustertrust.RemoteManagementMigrationFile} {
		if err := os.Remove(filepath.Join(f.dir, name)); err != nil {
			f.t.Fatal(err)
		}
	}
}

// rmHandlerSpy stands in for a real route handler so a test can prove whether
// the gate reached it, without an engine fixture.
type rmHandlerSpy struct {
	mu    sync.Mutex
	calls map[string]int
}

func newRMHandlerSpy() *rmHandlerSpy {
	return &rmHandlerSpy{calls: map[string]int{}}
}

func (s *rmHandlerSpy) handle(family string) pinnedPeerHandler {
	return func(w http.ResponseWriter, r *http.Request, caller string) {
		s.mu.Lock()
		s.calls[family]++
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"handler": family, "method": r.Method, "caller": caller,
		})
	}
}

func (s *rmHandlerSpy) count(family string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls[family]
}

// rmRegistryMux registers every route from the production registry with a spy
// handler in place of the real one. The classification, method validation and
// grant decision all come from controlServer.authorize; only the side effect is
// replaced.
func rmRegistryMux(s *controlServer, spy *rmHandlerSpy) *http.ServeMux {
	mux := http.NewServeMux()
	for _, route := range s.routes() {
		route.handle = spy.handle(route.family)
		mux.HandleFunc(route.path, s.requirePinCaller(s.authorize(route, route.handle)))
	}
	return mux
}

func rmTLSServer(t *testing.T, mesh *clustertrust.Mesh, handler http.Handler) *httptest.Server {
	t.Helper()
	ts := httptest.NewUnstartedServer(handler)
	ts.TLS = mesh.ServerTLSConfig()
	ts.StartTLS()
	t.Cleanup(ts.Close)
	return ts
}

func rmRequest(t *testing.T, client *http.Client, base, method, path, body string) (int, http.Header, []byte) {
	t.Helper()
	var reader io.Reader
	if body != "" {
		reader = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, base+path, reader)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("%s %s: read response: %v", method, path, err)
	}
	return resp.StatusCode, resp.Header, data
}

func rmCodedBody(t *testing.T, data []byte) (code, message string) {
	t.Helper()
	var body codedBody
	if err := json.Unmarshal(data, &body); err != nil {
		t.Fatalf("decode typed error body %q: %v", data, err)
	}
	return body.Code, body.Error
}

// rmExpectedClasses is the registry's classification as reviewed against the
// plan: reads never change target state, mutations do. It is a specification
// list, so a route silently reclassified in controlServer.routes fails here.
func rmExpectedClasses() map[string]controlClass {
	return map[string]controlClass{
		controlEnginesPath:          controlRead,
		controlModelFilesPath:       controlRead,
		controlModelFilePath:        controlRead,
		controlRemoteManagementPath: controlRead,
		controlInstallPath:          controlMutation,
		controlPullPath:             controlMutation,
		controlLoadPath:             controlMutation,
		controlUnloadPath:           controlMutation,
		controlDeletePath:           controlMutation,
		controlStartPath:            controlMutation,
		controlStopPath:             controlMutation,
		settingsPath + "get":        controlRead,
		settingsPath + "preview":    controlRead,
		settingsPath + "events":     controlRead,
		settingsPath + "apply":      controlMutation,
		controlPolicyGetPath:        controlRead,
		controlPolicySetPath:        controlMutation,
		controlAvailabilitySetPath:  controlMutation,
		settings.SwitchStatusPath:   controlRead,
		settings.SwitchApplyPath:    controlMutation,
	}
}

func TestRemoteManagementRegistryParity(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	s := &controlServer{mesh: f.mesh}
	routes := s.routes()
	mux := s.mux()

	want := rmExpectedClasses()
	if len(routes) != len(want) {
		t.Fatalf("registry has %d routes, classification spec has %d", len(routes), len(want))
	}
	seen := map[string]controlClass{}
	for _, route := range routes {
		if route.path == "" || route.method == "" || route.class == controlUnclassified || route.family == "" {
			t.Fatalf("route %+v is not fully classified", route)
		}
		if _, dup := seen[route.path]; dup {
			t.Fatalf("path %q is registered more than once", route.path)
		}
		seen[route.path] = route.class
		if expected, ok := want[route.path]; !ok {
			t.Fatalf("route %q has no classification spec", route.path)
		} else if expected != route.class {
			t.Fatalf("route %q class = %s, want %s", route.path, route.class, expected)
		}
		// The mux must serve the exact pattern the registry names, so the
		// registry and the served surface cannot drift apart.
		_, pattern := mux.Handler(&http.Request{Method: route.method, URL: &url.URL{Path: route.path}})
		if pattern != route.path {
			t.Fatalf("mux pattern for %s = %q, want %q", route.path, pattern, route.path)
		}
	}
	if _, pattern := mux.Handler(&http.Request{Method: http.MethodGet, URL: &url.URL{Path: "/v1/not-a-control-route"}}); pattern != "" {
		t.Fatalf("unregistered path resolved to pattern %q", pattern)
	}
}

func TestRemoteManagementRouteMatrixDeniesUngrantedMutation(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a", "uuid-b")
	f.writePermissions("1", map[string]bool{"uuid-a": true, "uuid-b": false})
	s := &controlServer{mesh: f.mesh}
	spy := newRMHandlerSpy()
	ts := rmTLSServer(t, f.mesh, rmRegistryMux(s, spy))
	denied := f.client("uuid-b")

	// The assertions follow the independent classification spec, never the
	// registry's own answer, so a misclassified route is exercised as what it is
	// supposed to be.
	spec := rmExpectedClasses()
	for _, route := range s.routes() {
		class, ok := spec[route.path]
		if !ok {
			t.Fatalf("route %s has no classification spec", route.path)
		}
		if class != route.class {
			t.Errorf("registry classifies %s as %s, want %s", route.path, route.class, class)
		}
		t.Run(class.String()+"_"+route.method+"_"+route.path, func(t *testing.T) {
			before := spy.count(route.family)
			status, header, body := rmRequest(t, denied, ts.URL, route.method, route.path, `{"engine":"fixture"}`)
			reached := spy.count(route.family) - before
			if class == controlRead {
				if status != http.StatusOK {
					t.Fatalf("denied peer read %s = %d, want 200: %s", route.path, status, body)
				}
				if reached != 1 {
					t.Fatalf("denied peer read %s reached its handler %d times, want 1", route.path, reached)
				}
				return
			}
			if status != http.StatusConflict {
				t.Fatalf("ungranted mutation %s = %d, want 409: %s", route.path, status, body)
			}
			if reached != 0 {
				t.Fatalf("ungranted mutation %s reached its handler %d times, want 0", route.path, reached)
			}
			// A refused mutation must not open a streaming 200 body.
			if contentType := header.Get("Content-Type"); !strings.Contains(contentType, "application/json") ||
				strings.Contains(contentType, "ndjson") {
				t.Fatalf("ungranted mutation %s content type = %q, want a JSON refusal", route.path, contentType)
			}
			code, message := rmCodedBody(t, body)
			if code != remoteManagementWireCode {
				t.Fatalf("ungranted mutation %s code = %q, want %q", route.path, code, remoteManagementWireCode)
			}
			if message != remoteManagementDeniedMessage {
				t.Fatalf("ungranted mutation %s message = %q, want the bounded explanation", route.path, message)
			}
		})
	}
}

func TestRemoteManagementRouteMatrixGrantsMutation(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a", "uuid-b")
	f.writePermissions("1", map[string]bool{"uuid-a": true, "uuid-b": false})
	s := &controlServer{mesh: f.mesh}
	spy := newRMHandlerSpy()
	ts := rmTLSServer(t, f.mesh, rmRegistryMux(s, spy))
	allowed := f.client("uuid-a")

	spec := rmExpectedClasses()
	for _, route := range s.routes() {
		class, ok := spec[route.path]
		if !ok {
			t.Fatalf("route %s has no classification spec", route.path)
		}
		t.Run(class.String()+"_"+route.method+"_"+route.path, func(t *testing.T) {
			before := spy.count(route.family)
			status, _, body := rmRequest(t, allowed, ts.URL, route.method, route.path, `{"engine":"fixture"}`)
			if status != http.StatusOK {
				t.Fatalf("granted caller %s %s = %d, want 200: %s", class, route.path, status, body)
			}
			if reached := spy.count(route.family) - before; reached != 1 {
				t.Fatalf("granted caller did not reach %s exactly once (%d)", route.path, reached)
			}
			// Identity always reaches the handler from the verified certificate.
			var seen struct {
				Caller string `json:"caller"`
			}
			if err := json.Unmarshal(body, &seen); err != nil {
				t.Fatalf("decode handler body %q: %v", body, err)
			}
			if seen.Caller != "uuid-a" {
				t.Fatalf("handler saw caller %q, want uuid-a", seen.Caller)
			}
		})
	}
}

func TestRemoteManagementRouteMethodValidation(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	f.writePermissions("1", map[string]bool{"uuid-a": true})
	s := &controlServer{mesh: f.mesh}
	spy := newRMHandlerSpy()
	ts := rmTLSServer(t, f.mesh, rmRegistryMux(s, spy))
	client := f.client("uuid-a")

	// A wrong method on a mutation route is refused before its writer runs, even
	// for a granted caller.
	get := findRMRoute(t, s, controlLoadPath)
	before := spy.count(get.family)
	status, header, body := rmRequest(t, client, ts.URL, http.MethodGet, controlLoadPath, "")
	if status != http.StatusMethodNotAllowed {
		t.Fatalf("GET on a mutation route = %d, want 405: %s", status, body)
	}
	if allow := header.Get("Allow"); allow != http.MethodPost {
		t.Fatalf("Allow header = %q, want POST", allow)
	}
	if reached := spy.count(get.family) - before; reached != 0 {
		t.Fatalf("wrong-method request reached the mutation handler %d times", reached)
	}

	// HEAD is a read of a registered read route, folded to GET for the handler.
	read := findRMRoute(t, s, controlEnginesPath)
	beforeRead := spy.count(read.family)
	status, _, body = rmRequest(t, client, ts.URL, http.MethodHead, controlEnginesPath, "")
	if status != http.StatusOK {
		t.Fatalf("HEAD on a read route = %d, want 200: %s", status, body)
	}
	if reached := spy.count(read.family) - beforeRead; reached != 1 {
		t.Fatalf("HEAD did not reach the read handler exactly once (%d)", reached)
	}

	// HEAD on a mutation route is still a method error.
	before = spy.count(get.family)
	status, _, body = rmRequest(t, client, ts.URL, http.MethodHead, controlLoadPath, "")
	if status != http.StatusMethodNotAllowed {
		t.Fatalf("HEAD on a mutation route = %d, want 405: %s", status, body)
	}
	if reached := spy.count(get.family) - before; reached != 0 {
		t.Fatalf("HEAD reached the mutation handler %d times", reached)
	}

	// An unregistered path is not a control route at all.
	status, _, _ = rmRequest(t, client, ts.URL, http.MethodPost, "/v1/not-a-control-route", "{}")
	if status != http.StatusNotFound {
		t.Fatalf("unregistered path = %d, want 404", status)
	}
	if spy.count(read.family) == 0 {
		t.Fatalf("read handler %s never observed a call", read.family)
	}
}

func findRMRoute(t *testing.T, s *controlServer, path string) controlRoute {
	t.Helper()
	for _, route := range s.routes() {
		if route.path == path {
			return route
		}
	}
	t.Fatalf("registry has no route %q", path)
	return controlRoute{}
}

func TestRemoteManagementUnclassifiedRouteFailsClosed(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	f.writePermissions("1", map[string]bool{"uuid-a": true})
	s := &controlServer{mesh: f.mesh}
	spy := newRMHandlerSpy()

	cases := []controlRoute{
		{path: controlEnginesPath, method: http.MethodGet, class: controlUnclassified, family: "unclassified", handle: spy.handle("unclassified")},
		{path: controlEnginesPath, method: "", class: controlRead, family: "empty-method", handle: spy.handle("empty-method")},
		{path: "/v1/other", method: http.MethodGet, class: controlRead, family: "path-mismatch", handle: spy.handle("path-mismatch")},
		{path: controlEnginesPath, method: http.MethodGet, class: controlRead, family: "nil-handler", handle: nil},
	}
	for _, route := range cases {
		t.Run(route.family, func(t *testing.T) {
			rec := httptest.NewRecorder()
			// The request never matches a mismatched route path.
			req := httptest.NewRequest(http.MethodGet, controlEnginesPath, nil)
			s.authorize(route, route.handle)(rec, req, "uuid-a")
			if rec.Code != http.StatusInternalServerError {
				t.Fatalf("unclassified %s = %d, want 500", route.family, rec.Code)
			}
			if spy.count(route.family) != 0 {
				t.Fatalf("unclassified %s reached a handler", route.family)
			}
		})
	}
}

func TestRemoteManagementUnpinnedCallerForbidden(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	f.writePermissions("1", map[string]bool{"uuid-a": true})
	s := &controlServer{mesh: f.mesh}
	spy := newRMHandlerSpy()
	ts := rmTLSServer(t, f.mesh, rmRegistryMux(s, spy))

	// A keypair that was never pinned completes the handshake and is refused at
	// the HTTP layer.
	strangerDir := t.TempDir()
	clustertrusttest.WriteKeypair(t, strangerDir, "uuid-stranger")
	id, err := clustertrust.LoadIdentity(strangerDir)
	if err != nil {
		t.Fatal(err)
	}
	stranger := &http.Client{Transport: &http.Transport{
		TLSClientConfig: clustertrust.ClientTLSConfig(id.Cert, f.serverDER()),
	}}

	for _, tc := range []struct{ method, path string }{
		{http.MethodGet, controlEnginesPath},
		{http.MethodPost, controlLoadPath},
		{http.MethodGet, controlRemoteManagementPath},
		{http.MethodPost, settingsPath + "apply"},
		{http.MethodPost, controlPolicySetPath},
		{http.MethodPost, settings.SwitchApplyPath},
	} {
		status, _, body := rmRequest(t, stranger, ts.URL, tc.method, tc.path, "{}")
		if status != http.StatusForbidden {
			t.Fatalf("unpinned %s %s = %d, want 403: %s", tc.method, tc.path, status, body)
		}
	}
	if total := spy.count("engine-inventory") + spy.count("engine-load") + spy.count("settings-apply") +
		spy.count("policy-set") + spy.count("engine-switch"); total != 0 {
		t.Fatalf("unpinned caller reached %d handlers", total)
	}
}

func TestRemoteManagementForgedBodyIdentityCannotSelectGrant(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a", "uuid-b")
	f.writePermissions("1", map[string]bool{"uuid-a": true, "uuid-b": false})
	s := &controlServer{mesh: f.mesh}
	spy := newRMHandlerSpy()
	ts := rmTLSServer(t, f.mesh, rmRegistryMux(s, spy))

	forged := `{"engine":"fixture","nodeId":"host-a","uuid":"uuid-a","caller":"uuid-a","peerUuid":"uuid-a","local":true}`

	status, _, body := rmRequest(t, f.client("uuid-b"), ts.URL, http.MethodPost, controlLoadPath, forged)
	if status != http.StatusConflict {
		t.Fatalf("denied caller with a forged allowed body = %d, want 409: %s", status, body)
	}
	if code, _ := rmCodedBody(t, body); code != remoteManagementWireCode {
		t.Fatalf("forged-identity refusal code = %q, want %q", code, remoteManagementWireCode)
	}
	if spy.count("engine-load") != 0 {
		t.Fatal("forged identity reached the mutation handler")
	}

	// The reverse: the allowed caller keeps its own identity no matter what the
	// body claims.
	status, _, body = rmRequest(t, f.client("uuid-a"), ts.URL, http.MethodPost, controlLoadPath,
		`{"engine":"fixture","nodeId":"host-b","uuid":"uuid-b","caller":"uuid-b","local":true}`)
	if status != http.StatusOK {
		t.Fatalf("allowed caller with a forged denied body = %d, want 200: %s", status, body)
	}
	var seen struct {
		Caller string `json:"caller"`
	}
	if err := json.Unmarshal(body, &seen); err != nil {
		t.Fatal(err)
	}
	if seen.Caller != "uuid-a" {
		t.Fatalf("handler saw caller %q, want uuid-a", seen.Caller)
	}
}

func TestRemoteManagementRevocationBetweenRequestsHasNoStaleAllow(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	f.writePermissions("1", map[string]bool{"uuid-a": true})

	// Two independent Mesh readers over the one cluster directory: a cached
	// allow in either would survive the revocation on disk.
	firstMesh := f.mesh
	secondMesh := clustertrust.Open(f.dir)
	first := &controlServer{mesh: firstMesh}
	second := &controlServer{mesh: secondMesh}
	spyFirst := newRMHandlerSpy()
	spySecond := newRMHandlerSpy()
	tsFirst := rmTLSServer(t, firstMesh, rmRegistryMux(first, spyFirst))
	tsSecond := rmTLSServer(t, secondMesh, rmRegistryMux(second, spySecond))
	client := f.client("uuid-a")

	for name, ts := range map[string]*httptest.Server{"first reader": tsFirst, "second reader": tsSecond} {
		status, _, body := rmRequest(t, client, ts.URL, http.MethodPost, settings.SwitchApplyPath, switchApplyFixture)
		if status != http.StatusOK {
			t.Fatalf("%s before revoke = %d, want 200: %s", name, status, body)
		}
	}

	f.writePermissions("2", map[string]bool{"uuid-a": false})

	for name, ts := range map[string]*httptest.Server{"first reader": tsFirst, "second reader": tsSecond} {
		spy := spyFirst
		if name == "second reader" {
			spy = spySecond
		}
		before := spy.count("engine-switch")
		status, _, body := rmRequest(t, client, ts.URL, http.MethodPost, settings.SwitchApplyPath, switchApplyFixture)
		if status != http.StatusConflict {
			t.Fatalf("%s after revoke = %d, want 409: %s", name, status, body)
		}
		if code, _ := rmCodedBody(t, body); code != remoteManagementWireCode {
			t.Fatalf("%s after revoke code = %q, want %q", name, code, remoteManagementWireCode)
		}
		if spy.count("engine-switch") != before {
			t.Fatalf("%s reached the switch writer after revoke", name)
		}
	}
}

func TestRemoteManagementPermissionStoreStates(t *testing.T) {
	zeroIncarnation := strings.Repeat("0", 64)
	cases := []struct {
		name    string
		want    string
		prepare func(f *rmFixture)
	}{
		{name: "missing store", want: remoteManagementStateUnavailable, prepare: func(f *rmFixture) { f.removeStore() }},
		{name: "corrupt document", want: remoteManagementStateUnavailable, prepare: func(f *rmFixture) {
			f.writeRaw(clustertrust.RemoteManagementFile, []byte("{not json"))
		}},
		{name: "pending migration", want: remoteManagementStateUnavailable, prepare: func(f *rmFixture) { f.setMarkerState("pending") }},
		{name: "initializing guard", want: remoteManagementStateUnavailable, prepare: func(f *rmFixture) {
			f.writeRaw(clustertrust.RemoteManagementInitializingFile, []byte("{}"))
		}},
		{name: "document of another migration", want: remoteManagementStateUnavailable, prepare: func(f *rmFixture) {
			f.setDocumentMigrationID(zeroIncarnation)
		}},
		{name: "grant for another incarnation", want: remoteManagementStateUnreviewed, prepare: func(f *rmFixture) {
			f.writeGrantWithIncarnation("2", zeroIncarnation, true, "uuid-a")
		}},
		{name: "no grant recorded", want: remoteManagementStateUnreviewed, prepare: func(f *rmFixture) {
			f.writePermissions("2", map[string]bool{})
		}},
		{name: "explicit deny", want: remoteManagementStateDenied, prepare: func(f *rmFixture) {
			f.writePermissions("2", map[string]bool{"uuid-a": false})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := rmNewFixture(t, "uuid-self", "uuid-a")
			f.writePermissions("1", map[string]bool{"uuid-a": true})
			tc.prepare(f)
			if got := f.mesh.RemoteManagement("uuid-a").State; got != tc.want {
				t.Fatalf("permission state = %q, want %q", got, tc.want)
			}
			s := &controlServer{mesh: f.mesh}
			spy := newRMHandlerSpy()
			route := controlRoute{
				path: controlLoadPath, method: http.MethodPost, class: controlMutation,
				family: "engine-load", handle: spy.handle("engine-load"),
			}
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, controlLoadPath, strings.NewReader(`{"engine":"fixture"}`))
			s.authorize(route, route.handle)(rec, req, "uuid-a")
			if rec.Code != http.StatusConflict {
				t.Fatalf("state %q = %d, want a typed 409 refusal", tc.want, rec.Code)
			}
			code, message := rmCodedBody(t, rec.Body.Bytes())
			if code != remoteManagementWireCode {
				t.Fatalf("state %q code = %q, want %q", tc.want, code, remoteManagementWireCode)
			}
			if message != remoteManagementDeniedMessage {
				t.Fatalf("state %q message = %q, want the bounded explanation", tc.want, message)
			}
			if strings.Contains(message, f.dir) || strings.Contains(message, "json") {
				t.Fatalf("refusal carries storage detail: %q", message)
			}
			if spy.count("engine-load") != 0 {
				t.Fatalf("state %q reached the mutation handler", tc.want)
			}
		})
	}
}

func TestRemoteManagementPermissionGetIsCallerOnly(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a", "uuid-b")
	f.writePermissions("4", map[string]bool{"uuid-a": true, "uuid-b": false})
	s := &controlServer{mesh: f.mesh}
	ts := rmTLSServer(t, f.mesh, s.mux())

	cases := []struct {
		peer     string
		allowed  bool
		state    string
		revision string
	}{
		{peer: "uuid-a", allowed: true, state: remoteManagementStateAllowed, revision: "4"},
		{peer: "uuid-b", allowed: false, state: remoteManagementStateDenied, revision: "4"},
	}
	for _, tc := range cases {
		t.Run(tc.peer, func(t *testing.T) {
			status, header, body := rmRequest(t, f.client(tc.peer), ts.URL, http.MethodGet, controlRemoteManagementPath, "")
			if status != http.StatusOK {
				t.Fatalf("permission read = %d, want 200: %s", status, body)
			}
			if contentType := header.Get("Content-Type"); !strings.Contains(contentType, "application/json") {
				t.Fatalf("permission read content type = %q", contentType)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(body, &fields); err != nil {
				t.Fatalf("decode permission read %q: %v", body, err)
			}
			for _, key := range []string{"version", "allowed", "revision", "state"} {
				if _, ok := fields[key]; !ok {
					t.Fatalf("permission read %q is missing %q", body, key)
				}
			}
			if len(fields) != 4 {
				t.Fatalf("permission read exposes %d fields, want only the caller's four: %s", len(fields), body)
			}
			var view remoteManagementView
			if err := json.Unmarshal(body, &view); err != nil {
				t.Fatal(err)
			}
			if view.Version != remoteManagementVersion || view.Allowed != tc.allowed ||
				view.State != tc.state || view.Revision != tc.revision {
				t.Fatalf("permission read = %+v, want %+v", view, tc)
			}
		})
	}

	// A peer selector in the query never chooses whose permission is read.
	status, _, body := rmRequest(t, f.client("uuid-b"), ts.URL, http.MethodGet,
		controlRemoteManagementPath+"?peerUuid=uuid-a&caller=uuid-a&nodeId=host-a", "")
	if status != http.StatusOK {
		t.Fatalf("permission read with a selector = %d, want 200: %s", status, body)
	}
	var view remoteManagementView
	if err := json.Unmarshal(body, &view); err != nil {
		t.Fatal(err)
	}
	if view.Allowed || view.State != remoteManagementStateDenied {
		t.Fatalf("a query selector changed the caller's permission: %+v", view)
	}
}

func TestRemoteManagementSelfCertHasNoMutationBypass(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	f.writePermissions("1", map[string]bool{"uuid-a": true})
	s := &controlServer{mesh: f.mesh}
	spy := newRMHandlerSpy()
	ts := rmTLSServer(t, f.mesh, rmRegistryMux(s, spy))
	self := f.client("uuid-self")

	// The pin gate accepts this node's own leaf (VerifyClientPin self-trust), so
	// reads from the device itself keep working.
	status, _, body := rmRequest(t, self, ts.URL, http.MethodGet, controlEnginesPath, "")
	if status != http.StatusOK {
		t.Fatalf("self-cert read = %d, want 200: %s", status, body)
	}

	// There is no local-service bypass for a mutation: RemoteManagement refuses
	// the node's own UUID, so the device is denied like any ungranted peer.
	if permission := f.mesh.RemoteManagement(f.selfUUID); permission.Allowed {
		t.Fatalf("self permission unexpectedly allowed: %+v", permission)
	} else if permission.State != remoteManagementStateUnavailable {
		t.Fatalf("self permission state = %q, want unavailable", permission.State)
	}
	before := spy.count("engine-load")
	status, _, body = rmRequest(t, self, ts.URL, http.MethodPost, controlLoadPath, `{"engine":"fixture"}`)
	if status != http.StatusConflict {
		t.Fatalf("self-cert mutation = %d, want a typed 409 refusal: %s", status, body)
	}
	if code, _ := rmCodedBody(t, body); code != remoteManagementWireCode {
		t.Fatalf("self-cert mutation code = %q, want %q", code, remoteManagementWireCode)
	}
	if spy.count("engine-load") != before {
		t.Fatal("self-cert mutation reached the writer")
	}
}

func TestRemoteManagementDecisionAuditHasNoSecrets(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a", "uuid-b")
	f.writePermissions("1", map[string]bool{"uuid-a": true})
	s := &controlServer{mesh: f.mesh}
	spy := newRMHandlerSpy()

	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })

	const sentinel = "SENTINEL-PROMPT-BODY-SECRET"
	requests := []struct {
		route  controlRoute
		caller string
	}{
		{controlRoute{path: controlPolicySetPath, method: http.MethodPost, class: controlMutation,
			family: "policy-set", handle: spy.handle("policy-set")}, "uuid-a"},
		{controlRoute{path: controlPolicySetPath, method: http.MethodPost, class: controlMutation,
			family: "policy-set", handle: spy.handle("policy-set")}, "uuid-b"},
		{controlRoute{path: controlEnginesPath, method: http.MethodGet, class: controlRead,
			family: "engine-inventory", handle: spy.handle("engine-inventory")}, "uuid-b"},
	}
	for _, tc := range requests {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(tc.route.method, tc.route.path+"?secret="+sentinel,
			strings.NewReader(`{"prompt":"`+sentinel+`","policy":{}}`))
		s.authorize(tc.route, tc.route.handle)(rec, req, tc.caller)
	}

	out := logs.String()
	if !strings.Contains(out, "ec authorization decision") {
		t.Fatalf("no decision audit line was emitted: %s", out)
	}
	if strings.Contains(out, sentinel) {
		t.Fatalf("audit line leaked request content: %s", out)
	}
	for _, want := range []string{
		`"caller":"uuid-a"`, `"caller":"uuid-b"`,
		`"operation":"policy-set"`, `"operation":"engine-inventory"`,
		`"class":"mutation"`, `"class":"read"`,
		`"decision":"allowed"`, `"decision":"denied"`,
		`"permissionRevision":"1"`, `"reason":"Remote management has not been granted."`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("audit line is missing %s: %s", want, out)
		}
	}
}

func TestRemoteManagementRelaysCarryTypedDenial(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	f.writePermissions("1", map[string]bool{"uuid-a": true})

	denial := &codedError{code: codeRemoteManagementDenied, message: remoteManagementDeniedMessage}
	veto := &codedError{code: codeLocalVeto, message: "this node was paused on the node itself; resume it there"}

	// Policy relay: the inbound route writes the typed 409 body the origin maps
	// back to -32012, and keeps a local veto distinct.
	policy := func(err error) *controlServer {
		return &controlServer{mesh: f.mesh, exec: &Executor{policyParent: func(context.Context, string, json.RawMessage, string) (json.RawMessage, error) {
			return nil, err
		}}}
	}
	// Settings relay: same contract for the settings apply family.
	settingsServer := func(err error) *controlServer {
		return &controlServer{mesh: f.mesh, exec: newRelayExecutor(t, err)}
	}

	t.Run("policy", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, controlPolicySetPath, strings.NewReader(`{}`))
		policy(denial).policyHandler("set")(rec, req, "uuid-a")
		rmAssertTypedRefusal(t, rec, remoteManagementWireCode, codeRemoteManagementDenied)

		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodPost, controlAvailabilitySetPath, strings.NewReader(`{}`))
		policy(veto).policyHandler("set-availability")(rec, req, "uuid-a")
		rmAssertTypedRefusal(t, rec, "local_veto", codeLocalVeto)
	})

	t.Run("settings", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, settingsPath+"apply", strings.NewReader(`{"engine":"missing"}`))
		settingsServer(denial).settingsHandler("apply")(rec, req, "uuid-a")
		rmAssertTypedRefusal(t, rec, remoteManagementWireCode, codeRemoteManagementDenied)

		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodPost, settingsPath+"apply", strings.NewReader(`{"engine":"missing"}`))
		settingsServer(veto).settingsHandler("apply")(rec, req, "uuid-a")
		rmAssertTypedRefusal(t, rec, "local_veto", codeLocalVeto)
	})

	t.Run("switch", func(t *testing.T) {
		handler := switchHandler("target-self", switchTestGate, func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
			return nil, denial
		})
		rec := httptest.NewRecorder()
		handler(rec, httptest.NewRequest(http.MethodPost, settings.SwitchApplyPath, strings.NewReader(switchApplyFixture)))
		rmAssertTypedRefusal(t, rec, remoteManagementWireCode, codeRemoteManagementDenied)

		// A granted peer that also hits R4's local pause still gets -32011.
		vetoHandler := switchHandler("target-self", switchTestGate, func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
			return nil, &codedError{code: settings.SwitchRemoteVetoCode, message: veto.message}
		})
		rec = httptest.NewRecorder()
		vetoHandler(rec, httptest.NewRequest(http.MethodPost, settings.SwitchApplyPath, strings.NewReader(switchApplyFixture)))
		rmAssertTypedRefusal(t, rec, "local_veto", codeLocalVeto)
	})
}

// newRelayExecutor is an executor whose settings parent always fails the way the
// relay test needs; the registry's grant decision is not part of these relay
// assertions, which start after the gate.
func newRelayExecutor(t *testing.T, err error) *Executor {
	t.Helper()
	ex := NewExecutor(NewRegistry(), NewReporter(nil), func(string, any) {}, t.TempDir())
	ex.settingsParent = func(context.Context, string, settings.Request, string) (json.RawMessage, error) {
		return nil, err
	}
	return ex
}

func rmAssertTypedRefusal(t *testing.T, rec *httptest.ResponseRecorder, wire string, code int) {
	t.Helper()
	if rec.Code != http.StatusConflict {
		t.Fatalf("relay refusal = %d, want 409: %s", rec.Code, rec.Body.String())
	}
	bodyCode, message := rmCodedBody(t, rec.Body.Bytes())
	if bodyCode != wire {
		t.Fatalf("relay wire code = %q, want %q", bodyCode, wire)
	}
	if message == "" {
		t.Fatal("relay refusal carried no message")
	}
	// The origin turns the body back into the same JSON-RPC code.
	originErr := remoteStatusError("relay", http.StatusConflict, rec.Body.Bytes())
	if got := rpcCode(originErr); got != code {
		t.Fatalf("origin code from relay body = %d, want %d (%v)", got, code, originErr)
	}
}

func TestRemoteManagementGetRPCReadsPeerPermission(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	target := f.peer("uuid-a")
	peerMesh := clustertrust.Open(target.dir)

	view := remoteManagementView{Version: remoteManagementVersion, Allowed: true, Revision: "9", State: remoteManagementStateAllowed}
	handler := func(w http.ResponseWriter, r *http.Request) {
		if _, ok := peerMesh.VerifyClientPin(r); !ok {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		_ = json.NewEncoder(w).Encode(view)
	}
	ts := rmTLSServer(t, peerMesh, rmMuxFor(handler, controlRemoteManagementPath))

	m, managerConn, clientConn := rmManagerFor(t, f, ts)
	defer managerConn.Close()
	defer clientConn.Close()

	reply := rmRPCCall(t, m, clientConn, `{"nodeId":"host-a"}`)
	if reply.errCode != 0 {
		t.Fatalf("remote-management-get failed: %s", reply.line)
	}
	var capability remoteManagementCapability
	if err := json.Unmarshal(reply.result, &capability); err != nil {
		t.Fatalf("decode capability %s: %v", reply.result, err)
	}
	if capability != (remoteManagementCapability{Version: remoteManagementVersion, Allowed: true, Revision: "9", State: remoteManagementStateAllowed}) {
		t.Fatalf("capability = %+v", capability)
	}
}

func TestRemoteManagementGetRPCMapsMissingRouteUnsupported(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	target := f.peer("uuid-a")
	peerMesh := clustertrust.Open(target.dir)

	// A target with no permission route answers 404; that is "unsupported", never
	// an allowed peer.
	ts := rmTLSServer(t, peerMesh, http.NewServeMux())
	m, managerConn, clientConn := rmManagerFor(t, f, ts)
	defer managerConn.Close()
	defer clientConn.Close()

	reply := rmRPCCall(t, m, clientConn, `{"nodeId":"host-a"}`)
	if reply.errCode != 0 {
		t.Fatalf("remote-management-get on an old target failed: %s", reply.line)
	}
	var capability remoteManagementCapability
	if err := json.Unmarshal(reply.result, &capability); err != nil {
		t.Fatal(err)
	}
	if capability.State != remoteManagementStateUnsupported || capability.Allowed {
		t.Fatalf("capability = %+v, want unsupported and not allowed", capability)
	}
}

func TestRemoteManagementGetRPCValidatesNodeID(t *testing.T) {
	f := rmNewFixture(t, "uuid-self", "uuid-a")
	target := f.peer("uuid-a")
	peerMesh := clustertrust.Open(target.dir)
	ts := rmTLSServer(t, peerMesh, http.NewServeMux())
	m, managerConn, clientConn := rmManagerFor(t, f, ts)
	defer managerConn.Close()
	defer clientConn.Close()

	if reply := rmRPCCall(t, m, clientConn, `{}`); reply.errCode != -32602 {
		t.Fatalf("missing nodeId error code = %d, want -32602: %s", reply.errCode, reply.line)
	}
	if reply := rmRPCCall(t, m, clientConn, `{"nodeId":"host-unknown"}`); reply.errCode != -32000 {
		t.Fatalf("unknown nodeId error code = %d, want -32000: %s", reply.errCode, reply.line)
	}
}

func rmMuxFor(handler http.HandlerFunc, paths ...string) *http.ServeMux {
	mux := http.NewServeMux()
	for _, path := range paths {
		mux.HandleFunc(path, handler)
	}
	return mux
}

// rmManagerFor wires an engine-manager against the fixture with one discovered
// ec peer pointing at ts, exactly as discovery would report it.
func rmManagerFor(t *testing.T, f *rmFixture, ts *httptest.Server) (*Manager, net.Conn, net.Conn) {
	t.Helper()
	host, portStr, err := net.SplitHostPort(ts.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatal(err)
	}
	managerConn, clientConn := net.Pipe()
	ex := NewExecutor(NewRegistry(), NewReporter(nil), func(string, any) {}, t.TempDir())
	m := NewManager(NewCodec(managerConn), ex, f.mesh, f.dir)
	t.Cleanup(func() { m.controlHTTP.CloseIdle(); m.controlReadyHTTP.CloseIdle() })
	m.peers.set([]noderec.DirectoryNode{{
		HostUUID: "host-a", IP: host, ClusterUUID: "uuid-a",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineControl: {Port: port}},
	}})
	if _, ok := m.peers.lookup("host-a"); !ok {
		t.Fatal("fixture peer is not in the manager's directory")
	}
	return m, managerConn, clientConn
}

type rmRPCReply struct {
	line    string
	result  json.RawMessage
	errCode int
}

func rmRPCCall(t *testing.T, m *Manager, conn net.Conn, params string) rmRPCReply {
	t.Helper()
	id := json.RawMessage("1")
	reader := bufio.NewReader(conn)
	m.handleMessage(context.Background(), &Message{
		JSONRPC: "2.0", ID: &id, Method: remoteManagementGetMethod, Params: json.RawMessage(params),
	})
	conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	line, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("no response to %s: %v", params, err)
	}
	reply := rmRPCReply{line: strings.TrimSpace(line)}
	var frame struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code int `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(reply.line), &frame); err != nil {
		t.Fatalf("decode response %q: %v", reply.line, err)
	}
	reply.result = frame.Result
	if frame.Error != nil {
		reply.errCode = frame.Error.Code
	}
	return reply
}
