// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"strings"
	"testing"
)

func TestInvalidSenderNameKeepsLastGoodName(t *testing.T) {
	for _, bad := range []string{"bad\u0007name", strings.Repeat("x", 129)} {
		m := newTestManager(t)
		aUUID, aCert, aFP, _ := makeNode(t, "a")
		pinTrusted(t, m, aUUID, aCert, aFP)
		m.upsertMember(&ClusterNode{NodeUUID: aUUID, Name: "good", State: stateMember})
		if _, err := m.trust.UpdateIdentity(aUUID, "good"); err != nil {
			t.Fatal(err)
		}
		roster := &Roster{ClusterID: "cluster-1", Members: []RosterEntry{
			{NodeUUID: aUUID, Name: bad, CertPem: aCert, CertFingerprint: aFP},
		}}
		if m.mergeRoster(roster, aUUID) {
			t.Fatalf("invalid name %q reported a change", bad)
		}
		if n, _ := m.memberByUUID(aUUID); n.Name != "good" {
			t.Fatalf("member name = %q after invalid %q, want good", n.Name, bad)
		}
		if pin, _ := m.trust.Get(aUUID); pin.Name != "good" {
			t.Fatalf("pin name = %q after invalid %q, want good", pin.Name, bad)
		}
	}
}

func TestFirstPinWithInvalidRelayedNameStoresFallback(t *testing.T) {
	m := newTestManager(t)
	cUUID, cCert, cFP, _ := makeNode(t, "c")
	if err := m.pinEntry(&RosterEntry{NodeUUID: cUUID, Name: "evil‮name", CertPem: cCert, CertFingerprint: cFP}, nil); err != nil {
		t.Fatalf("pinEntry: %v", err)
	}
	want := "PAIR " + strings.ToLower(cUUID[:8])
	if n, _ := m.memberByUUID(cUUID); n.Name != want {
		t.Fatalf("member name = %q, want %q", n.Name, want)
	}
	if pin, _ := m.trust.Get(cUUID); pin.Name != want {
		t.Fatalf("pin name = %q, want %q", pin.Name, want)
	}
}

func TestJoinRequestBidiNameShowsFallback(t *testing.T) {
	m := joinTestManager(t)
	if _, _, err := m.foundCluster("admitting cluster"); err != nil {
		t.Fatal(err)
	}
	p := 14321
	id := joinTestID(t)
	_, status, reason := m.applyJoinRequestWire("10.0.0.2", joinRequestWire{Version: 1, Operation: "request", RequestID: id,
		Requester: &joinRequester{"abcdef01-0000-4000-8000-000000000000", "desk‮kcab"}, Port: &p})
	if status != http.StatusOK || reason != "" {
		t.Fatalf("create: status %d reason %q", status, reason)
	}
	if got := m.joinSnapshot().Requests[0].PeerName; got != "PAIR abcdef01" {
		t.Fatalf("peerName = %q, want fallback", got)
	}
}

func TestDecomposedAccentArrivesNFC(t *testing.T) {
	m := newTestManager(t)
	aUUID, aCert, aFP, _ := makeNode(t, "a")
	pinTrusted(t, m, aUUID, aCert, aFP)
	m.upsertMember(&ClusterNode{NodeUUID: aUUID, Name: "old", State: stateMember})
	roster := &Roster{ClusterID: "cluster-1", Members: []RosterEntry{
		{NodeUUID: aUUID, Name: "Café", CertPem: aCert, CertFingerprint: aFP},
	}}
	if !m.mergeRoster(roster, aUUID) {
		t.Fatal("rename should report a change")
	}
	if n, _ := m.memberByUUID(aUUID); n.Name != "Café" {
		t.Fatalf("member name = %q, want NFC %q", n.Name, "Café")
	}
}
