// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"reflect"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/noderec"
)

func TestDirectoryIdentityOnlyChangePublishes(t *testing.T) {
	d := newDirectory()
	node := noderec.DirectoryNode{HostUUID: "target", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineManager: {Port: 14317}}}
	d.upsert(node)
	inv := modelInventory{models: []string{"served-id"}, byEngine: map[string][]string{"llamacpp": {"served-id"}},
		identities: map[string]map[string]engines.ModelIdentity{"llamacpp": {"served-id": {
			Repository: "owner/repository", Quantization: "Q4_K_M", Strength: "declared",
			Files: []engines.ModelIdentityFile{{Role: "model", Name: "model.gguf", SizeBytes: 42}},
		}}}}
	if _, changed, valid := d.applyModels(node.HostUUID, node.IP, 14317, inv); !changed || !valid {
		t.Fatalf("initial inventory changed/valid = %v/%v", changed, valid)
	}
	if _, changed, valid := d.applyModels(node.HostUUID, node.IP, 14317, inv); changed || !valid {
		t.Fatalf("unchanged inventory changed/valid = %v/%v", changed, valid)
	}
	identity := inv.identities["llamacpp"]["served-id"]
	identity.Quantization = "Q8_0"
	inv.identities["llamacpp"]["served-id"] = identity
	published, changed, valid := d.applyModels(node.HostUUID, node.IP, 14317, inv)
	if !changed || !valid || published.ModelIdentities["llamacpp"]["served-id"].Quantization != "Q8_0" {
		t.Fatalf("identity-only change not published: %+v, changed/valid %v/%v", published, changed, valid)
	}
	published.ModelIdentities["llamacpp"]["served-id"].Files[0].Name = "changed.gguf"
	delete(published.ModelIdentities["llamacpp"], "served-id")
	delete(published.ModelIdentities, "llamacpp")
	stored, present := d.get(node.HostUUID)
	if !present || stored.ModelIdentities["llamacpp"]["served-id"].Files[0].Name != "model.gguf" || stored.ModelIdentities["llamacpp"]["served-id"].Quantization != "Q8_0" {
		t.Fatalf("publication exposed stored identity: %+v", stored.ModelIdentities)
	}
}

func TestDirectoryIdentitySnapshotsAreIndependent(t *testing.T) {
	d := newDirectory()
	node := noderec.DirectoryNode{HostUUID: "target", ModelIdentities: map[string]map[string]engines.ModelIdentity{"llamacpp": {"served-id": {
		Strength: "declared", Files: []engines.ModelIdentityFile{{Role: "model", Name: "model.gguf", SizeBytes: 42}},
	}}}}
	d.upsert(node)
	node.ModelIdentities["llamacpp"]["served-id"].Files[0].Name = "caller.gguf"
	got, present := d.get("target")
	if !present || got.ModelIdentities["llamacpp"]["served-id"].Files[0].Name != "model.gguf" {
		t.Fatal("upsert retained caller-owned identity files")
	}
	got.ModelIdentities["llamacpp"]["served-id"].Files[0].Name = "getter.gguf"
	rows := d.snapshot("")
	if len(rows) != 1 || rows[0].ModelIdentities["llamacpp"]["served-id"].Files[0].Name != "model.gguf" {
		t.Fatal("get exposed directory identity files")
	}
	delete(rows[0].ModelIdentities["llamacpp"], "served-id")
	last, _ := d.get("target")
	if len(last.ModelIdentities["llamacpp"]) != 1 {
		t.Fatal("snapshot exposed directory identity maps")
	}
	if !d.remove("target") || len(d.snapshot("")) != 0 {
		t.Fatal("removed identity-bearing node survived in the directory")
	}
}

func TestModelDownloadDirectoryLifecycleOnlyChanges(t *testing.T) {
	d := newDirectory()
	node := noderec.DirectoryNode{HostUUID: "target", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineManager: {Port: 14317}}}
	d.upsert(node)
	current := true
	inv := modelInventory{models: []string{"ready"}, loadedByEngine: map[string][]string{"llamacpp": {"org/model"}},
		downloadObservation: downloadObservation{snapshot: scannerDownloadSnapshot("1"), current: &current}}
	apply := func(wantChanged bool) noderec.DirectoryNode {
		t.Helper()
		got, changed, valid := d.applyModels(node.HostUUID, node.IP, 14317, inv)
		if !valid || changed != wantChanged {
			t.Fatalf("apply changed/valid = %v/%v, want %v/true", changed, valid, wantChanged)
		}
		if got.DownloadState == nil || !got.DownloadState.Equal(*inv.snapshot) || !reflect.DeepEqual(got.Models, inv.models) || !reflect.DeepEqual(got.LoadedByEngine, inv.loadedByEngine) {
			t.Fatalf("inventory/lifecycle/residency changed: %+v", got)
		}
		return got
	}
	apply(true)
	apply(false)
	inv.snapshot = scannerDownloadSnapshot("2")
	*inv.snapshot.Entries[0].Percent = 22
	apply(true)
	apply(false)
	if _, changed, valid := d.markDownloadsStale(node.HostUUID, node.IP, 14317); !valid || !changed {
		t.Fatalf("freshness loss not reported: %v/%v", changed, valid)
	}
	apply(true)
}

func TestModelDownloadDirectoryCopiesAreIndependent(t *testing.T) {
	d := newDirectory()
	current := true
	node := noderec.DirectoryNode{HostUUID: "target", DownloadState: scannerDownloadSnapshot("1"), DownloadStateCurrent: &current}
	d.upsert(node)
	*node.DownloadState.Entries[0].Percent = 99
	current = false
	first, present := d.get("target")
	if !present || *first.DownloadState.Entries[0].Percent != 21 || !*first.DownloadStateCurrent {
		t.Fatal("upsert retained caller-owned snapshot/percent/current pointers")
	}
	*first.DownloadState.Entries[0].Percent = 88
	*first.DownloadStateCurrent = false
	rows := d.snapshot("")
	if len(rows) != 1 || *rows[0].DownloadState.Entries[0].Percent != 21 || !*rows[0].DownloadStateCurrent {
		t.Fatal("get exposed directory-owned download fields")
	}
	*rows[0].DownloadState.Entries[0].Percent = 77
	last, _ := d.get("target")
	if *last.DownloadState.Entries[0].Percent != 21 {
		t.Fatal("snapshot exposed directory-owned download fields")
	}
}

func TestModelDownloadDirectoryRejectsLowerRevision(t *testing.T) {
	d := newDirectory()
	current := true
	node := noderec.DirectoryNode{HostUUID: "target", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineManager: {Port: 14317}},
		DownloadState: scannerDownloadSnapshot("10"), DownloadStateCurrent: &current, Models: []string{"ready"}}
	d.upsert(node)
	inv := modelInventory{models: []string{"obsolete"}, downloadObservation: downloadObservation{snapshot: scannerDownloadSnapshot("9"), current: &current}}
	if _, changed, valid := d.applyModels(node.HostUUID, node.IP, 14317, inv); changed || valid {
		t.Fatalf("lower revision accepted: %v/%v", changed, valid)
	}
	got, _ := d.get(node.HostUUID)
	if !got.DownloadState.Equal(*node.DownloadState) || !reflect.DeepEqual(got.Models, node.Models) || !*got.DownloadStateCurrent {
		t.Fatalf("lower revision changed canonical inventory or lifecycle: %+v", got)
	}
}

func TestDirectoryAuthenticatedClaimResistsBrowseSupersedeEviction(t *testing.T) {
	d := newDirectory()
	old := noderec.DirectoryNode{HostUUID: "admitted-peer", Name: "same-hostname", IP: "192.0.2.10", LastSeen: 1}
	d.installPeerServices(noderec.PeerServicesClaim{Incarnation: "current"}, old)
	arriving := noderec.DirectoryNode{HostUUID: "new-browse-peer", Name: "same-hostname", IP: "192.0.2.11", LastSeen: 1000}
	if evicted := d.upsertEvicting(arriving, []noderec.DirectoryNode{old}); len(evicted) != 0 {
		t.Fatalf("browse eviction removed authenticated authority: %+v", evicted)
	}
	if current, present := d.get("admitted-peer"); !present || current.IP != "192.0.2.10" {
		t.Fatalf("authenticated claim lost: %+v, %v", current, present)
	}
}

func TestToDirectoryNode(t *testing.T) {
	raw := RawNode{
		ID:        mdnsLabel("host-uuid"),
		Host:      mdnsLabel("host-uuid") + ".local.",
		Addresses: []string{"10.221.0.9", "192.168.1.10"},
		TXT:       []string{"v=1", "uuid=host-uuid", "cluster-uuid=clu", "name=  Desk A ", "ip=192.168.1.10", "ni=14318", "lc=8080"},
	}
	n, ok := toDirectoryNode(raw, true)
	if !ok {
		t.Fatal("a record with uuid= should project")
	}
	if n.HostUUID != "host-uuid" {
		t.Errorf("HostUUID = %q, want host-uuid", n.HostUUID)
	}
	if n.Name != "Desk A" {
		t.Errorf("Name = %q, want the normalized name= value", n.Name)
	}
	if n.IP != "192.168.1.10" {
		t.Errorf("IP = %q, want the ip= TXT value", n.IP)
	}
	if !n.Trusted {
		t.Error("Trusted should reflect the caller-supplied flag")
	}
	if !n.HasService(noderec.ServiceNodeInfo) || !n.HasService(noderec.ServiceLlamaCPP) {
		t.Errorf("services wrong: %+v", n.Services)
	}
	if n.Services[noderec.ServiceLlamaCPP].Port != 8080 {
		t.Errorf("ol port = %d", n.Services[noderec.ServiceLlamaCPP].Port)
	}
}

// TestToDirectoryNodeNameComesOnlyFromTXT pins the name source: the record's
// name= value after validation, never the instance label, and empty when the
// value is missing or invalid.
func TestToDirectoryNodeNameComesOnlyFromTXT(t *testing.T) {
	cases := []struct {
		name string
		txt  []string
		want string
	}{
		{"missing name", []string{"v=1", "uuid=host-uuid"}, ""},
		{"numeric name", []string{"v=1", "uuid=host-uuid", "name=1234"}, "1234"},
		{"non-ascii name", []string{"v=1", "uuid=host-uuid", "name=白いPC"}, "白いPC"},
		{"control character", []string{"v=1", "uuid=host-uuid", "name=bad\u0007name"}, ""},
		{"blank name", []string{"v=1", "uuid=host-uuid", "name=   "}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n, ok := toDirectoryNode(RawNode{ID: mdnsLabel("host-uuid"), Addresses: []string{"192.168.1.10"}, TXT: tc.txt}, false)
			if !ok {
				t.Fatal("a record with uuid= should project")
			}
			if n.Name != tc.want {
				t.Fatalf("Name = %q, want %q", n.Name, tc.want)
			}
		})
	}
}

// TestToDirectoryNodeIgnoresSRVPort asserts the locked-decision contract: the
// SRV port on a browsed _nvpair-node record is a fixed, NON-authoritative
// constant that consumers MUST ignore — every service port comes only from the
// TXT map. Here the record's SRV port is a bogus value while TXT says ni=14318,
// and the directory entry must report the TXT port, never the SRV one.
func TestToDirectoryNodeIgnoresSRVPort(t *testing.T) {
	raw := RawNode{
		ID:        "hostC",
		Host:      "hostC.local.",
		Port:      65000, // bogus SRV port — must be ignored
		Addresses: []string{"192.168.1.30"},
		TXT:       []string{"v=1", "uuid=host-c", "ip=192.168.1.30", "ni=14318", "lc=8080"},
	}
	n, ok := toDirectoryNode(raw, false)
	if !ok {
		t.Fatal("a record with uuid= should project")
	}
	if got := n.Services[noderec.ServiceNodeInfo].Port; got != 14318 {
		t.Errorf("ni port = %d, want 14318 from TXT (SRV port 65000 must be ignored)", got)
	}
	if got := n.Services[noderec.ServiceLlamaCPP].Port; got != 8080 {
		t.Errorf("ol port = %d, want 8080 from TXT", got)
	}
	for svc, st := range n.Services {
		if st.Port == 65000 {
			t.Errorf("the non-authoritative SRV port leaked into service %q", svc)
		}
	}
}

// TestToDirectoryNodeSkipsWithoutUUID: a record with no uuid= can't be keyed by
// a stable identity, so it's skipped (ok=false) rather than keyed by the
// hostname — node identity is the UUID everywhere, no name fallback.
func TestToDirectoryNodeSkipsWithoutUUID(t *testing.T) {
	raw := RawNode{
		ID:        "hostB",
		Addresses: []string{"10.221.0.9", "192.168.1.20"},
		TXT:       []string{"v=1", "er=14319"},
	}
	if _, ok := toDirectoryNode(raw, false); ok {
		t.Error("a record without uuid= must be skipped, not projected")
	}
}

// TestToDirectoryNodeIPFallback: with a uuid= but no ip=, every advertised address
// becomes a candidate and the canonical one is the ranker's head. The private
// blocks tie, so this is an arbitrary but stable choice — deliberately so. Which
// private block a node's real network uses is not knowable from the address, and
// the previous preference for 192.168 over 10.x is what made a multi-homed host
// publish a two-host direct-connect link instead of its LAN. Both addresses are
// kept, and a consumer that must connect resolves it by connecting.
func TestToDirectoryNodeIPFallback(t *testing.T) {
	raw := RawNode{
		ID:        "hostB",
		Addresses: []string{"10.221.0.9", "192.168.1.20"},
		TXT:       []string{"v=1", "uuid=host-b", "er=14319"},
	}
	n, ok := toDirectoryNode(raw, false)
	if !ok {
		t.Fatal("a record with uuid= should project")
	}
	if n.IP != "10.221.0.9" {
		t.Errorf("IP fallback = %q, want 10.221.0.9 (top-ranked)", n.IP)
	}
	if len(n.IPs) != 2 || n.IPs[0] != "10.221.0.9" || n.IPs[1] != "192.168.1.20" {
		t.Errorf("candidates = %v, want both advertised addresses ranked", n.IPs)
	}
	if n.Clustered() {
		t.Error("no cluster-uuid should mean not clustered")
	}
}

// TestToDirectoryNodePreservesPublishedOrder: a node's own ips= order survives
// projection, and its ip= stays canonical even though the advertised address list
// leads with something else. This is the multi-homed fix at the collapse point —
// the node ranked these with evidence no observer has.
func TestToDirectoryNodePreservesPublishedOrder(t *testing.T) {
	raw := RawNode{
		ID:        "spark",
		Addresses: []string{"192.168.240.2", "10.172.54.70"},
		TXT: []string{
			"v=1", "uuid=spark-1", "ip=10.172.54.70",
			"ips=10.172.54.70,192.168.240.2", "ni=14318",
		},
	}
	n, ok := toDirectoryNode(raw, false)
	if !ok {
		t.Fatal("a record with uuid= should project")
	}
	if n.IP != "10.172.54.70" {
		t.Errorf("canonical = %q, want the node's own ip= 10.172.54.70", n.IP)
	}
	if len(n.IPs) != 2 || n.IPs[0] != "10.172.54.70" || n.IPs[1] != "192.168.240.2" {
		t.Errorf("candidates = %v, want the node's published order", n.IPs)
	}
}

// TestToDirectoryNodeUnionsUnpublishedAddresses: an address the browse resolved
// but the node did not rank is kept as a fallback, appended rather than promoted.
func TestToDirectoryNodeUnionsUnpublishedAddresses(t *testing.T) {
	raw := RawNode{
		ID:        "hostC",
		Addresses: []string{"10.0.9.9"},
		TXT:       []string{"v=1", "uuid=host-c", "ip=10.0.0.5", "ips=10.0.0.5"},
	}
	n, ok := toDirectoryNode(raw, false)
	if !ok {
		t.Fatal("a record with uuid= should project")
	}
	want := []string{"10.0.0.5", "10.0.9.9"}
	if len(n.IPs) != len(want) || n.IPs[0] != want[0] || n.IPs[1] != want[1] {
		t.Errorf("candidates = %v, want %v", n.IPs, want)
	}
}

func TestDirectoryUpsertRemoveSnapshot(t *testing.T) {
	d := newDirectory()
	a := noderec.DirectoryNode{HostUUID: "a", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceLlamaCPP: {Port: 8080}}}
	b := noderec.DirectoryNode{HostUUID: "b", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceErrors: {Port: 14319}}}

	if isNew, _ := d.upsert(a); !isNew {
		t.Error("first upsert should be new")
	}
	if isNew, _ := d.upsert(a); isNew {
		t.Error("second upsert of same hostUuid should not be new")
	}
	d.upsert(b)

	if all := d.snapshot(""); len(all) != 2 || all[0].HostUUID != "a" || all[1].HostUUID != "b" {
		t.Fatalf("snapshot(all) = %+v, want [a b] sorted", all)
	}
	// Service filter.
	if ol := d.snapshot(noderec.ServiceLlamaCPP); len(ol) != 1 || ol[0].HostUUID != "a" {
		t.Fatalf("snapshot(ol) = %+v, want [a]", ol)
	}
	if !d.remove("a") {
		t.Error("remove(a) should report existed")
	}
	if d.remove("a") {
		t.Error("remove(a) again should report not existed")
	}
	if all := d.snapshot(""); len(all) != 1 || all[0].HostUUID != "b" {
		t.Fatalf("after remove, snapshot = %+v, want [b]", all)
	}
}
