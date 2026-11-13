// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"regexp"
	"slices"
	"testing"
	"time"
)

func rosterHashFixture() *Roster {
	return &Roster{
		ClusterID: "cluster-1",
		Members: []RosterEntry{
			{NodeUUID: "a", Name: "alpha", Addr: "10.0.0.1:1", AdmissionEpoch: 1, CertPem: "pem-a", CertFingerprint: "fp-a"},
			{NodeUUID: "b", Name: "bravo", Addr: "10.0.0.2:1", AdmissionEpoch: 2, CertPem: "pem-b", CertFingerprint: "fp-b",
				Endorsements: []Endorsement{{By: "a", Fingerprint: "fp-b", ClusterID: "cluster-1", IssuedAt: 1}, {By: "c", Fingerprint: "fp-b", ClusterID: "cluster-1", IssuedAt: 2}}},
			{NodeUUID: "c", Name: "charlie", Addr: "10.0.0.3:1", AdmissionEpoch: 3, CertPem: "pem-c", CertFingerprint: "fp-c"},
		},
		RemovalProofs: []RemovalProof{{Tombstone: Tombstone{NodeUUID: "d"}, SignerFingerprint: "fp-a"}},
	}
}

func cloneRosterForHashTest(r *Roster) *Roster {
	next := *r
	next.Members = slices.Clone(r.Members)
	for i := range next.Members {
		next.Members[i].Endorsements = slices.Clone(r.Members[i].Endorsements)
	}
	next.RemovalProofs = slices.Clone(r.RemovalProofs)
	return &next
}

func TestRosterHashCanonicalForm(t *testing.T) {
	base := rosterHashFixture()
	want := canonicalRosterHash(base)
	if !regexp.MustCompile(`^[0-9a-f]{64}$`).MatchString(want) {
		t.Fatalf("hash %q is not 64 lowercase hex characters", want)
	}

	t.Run("member and endorsement order does not matter", func(t *testing.T) {
		r := cloneRosterForHashTest(base)
		slices.Reverse(r.Members)
		for i := range r.Members {
			slices.Reverse(r.Members[i].Endorsements)
		}
		if got := canonicalRosterHash(r); got != want {
			t.Fatalf("reordered hash = %s, want %s", got, want)
		}
	})
	t.Run("address alone does not change the hash", func(t *testing.T) {
		r := cloneRosterForHashTest(base)
		r.Members[0].Addr = "192.168.1.9:7"
		r.Members[2].Addr = ""
		if got := canonicalRosterHash(r); got != want {
			t.Fatalf("address-only change altered hash")
		}
	})
	for name, edit := range map[string]func(*Roster){
		"name":          func(r *Roster) { r.Members[0].Name = "renamed" },
		"admission":     func(r *Roster) { r.Members[1].AdmissionEpoch++ },
		"endorsements":  func(r *Roster) { r.Members[1].Endorsements = r.Members[1].Endorsements[:1] },
		"removal proof": func(r *Roster) { r.RemovalProofs = nil },
		"member set":    func(r *Roster) { r.Members = r.Members[:2] },
	} {
		t.Run(name+" changes the hash", func(t *testing.T) {
			r := cloneRosterForHashTest(base)
			edit(r)
			if canonicalRosterHash(r) == want {
				t.Fatalf("%s change kept the hash", name)
			}
		})
	}
	t.Run("input is not modified", func(t *testing.T) {
		r := cloneRosterForHashTest(base)
		slices.Reverse(r.Members)
		before := cloneRosterForHashTest(r)
		canonicalRosterHash(r)
		for i := range r.Members {
			if r.Members[i].NodeUUID != before.Members[i].NodeUUID || r.Members[i].Addr != before.Members[i].Addr ||
				!slices.Equal(r.Members[i].Endorsements, before.Members[i].Endorsements) {
				t.Fatalf("member %d modified: %+v", i, r.Members[i])
			}
		}
	})
}

func TestRosterHashStableAcrossPinOrderAndAddress(t *testing.T) {
	type peer struct{ uuid, cert, fp string }
	var peers []peer
	for _, host := range []string{"node-a", "node-b", "node-c"} {
		uuid, cert, fp, _ := makeNode(t, host)
		peers = append(peers, peer{uuid, cert, fp})
	}
	forward, backward := newTestManager(t), newTestManager(t)
	for i := range peers {
		p, q := peers[i], peers[len(peers)-1-i]
		pinTrusted(t, forward, p.uuid, p.cert, p.fp)
		pinTrusted(t, backward, q.uuid, q.cert, q.fp)
	}
	// The self entries differ between managers, so compare peer entries only.
	peerHash := func(m *Manager) string {
		r := m.buildLocalRoster()
		r.Members = slices.DeleteFunc(r.Members, func(e RosterEntry) bool { return e.NodeUUID == m.identity.NodeUUID })
		return canonicalRosterHash(r)
	}
	if peerHash(forward) != peerHash(backward) {
		t.Fatal("pin order changed the peer roster hash")
	}

	first := forward.rosterHash()
	if second := forward.rosterHash(); second != first {
		t.Fatalf("consecutive hashes differ: %s then %s", first, second)
	}
	forward.upsertMember(&ClusterNode{NodeUUID: peers[0].uuid, IPAddress: "192.168.1.5", Port: 14999, ClusterID: "cluster-1", State: stateMember})
	if !forward.setMemberAddr(peers[0].uuid, "192.168.1.6", 14999) {
		t.Fatal("member address unchanged")
	}
	if got := forward.rosterHash(); got != first {
		t.Fatal("member address change altered the roster hash")
	}
	if changed, err := forward.trust.UpdateIdentity(peers[1].uuid, "renamed"); err != nil || !changed {
		t.Fatalf("rename pin: changed=%v err=%v", changed, err)
	}
	if forward.rosterHash() == first {
		t.Fatal("pin rename kept the roster hash")
	}
}

func TestShouldRunRequestedPass(t *testing.T) {
	last := time.Unix(1000, 0)
	if shouldRunRequestedPass(last, last.Add(requestedPassMinInterval-time.Millisecond)) {
		t.Fatal("requested pass allowed just under the interval")
	}
	if !shouldRunRequestedPass(last, last.Add(requestedPassMinInterval)) {
		t.Fatal("requested pass refused at the interval")
	}
	if !shouldRunRequestedPass(time.Time{}, time.Now()) {
		t.Fatal("first requested pass refused")
	}
}

func TestMaintainPassMakesNoNetworkCall(t *testing.T) {
	a, b := newClusterPeerFixture(t, false), newClusterPeerFixture(t, false)
	pairClusterFixtures(t, a, b)
	if !a.m.maintainPass() {
		t.Fatal("paired fixture is not clustered")
	}
	if got := b.rosters.Load(); got != 0 {
		t.Fatalf("maintainPass sent %d roster POSTs", got)
	}
	a.m.reconcilePass()
	if b.rosters.Load() == 0 {
		t.Fatal("reconcilePass sent no roster POST; the counting fixture is not reached")
	}
}

func TestRequestReconcilePassCoalesces(t *testing.T) {
	m := newTestManager(t)
	m.requestReconcilePass()
	m.requestReconcilePass()
	if len(m.reconcileRequest) != 1 {
		t.Fatalf("pending requests = %d, want 1", len(m.reconcileRequest))
	}
}
