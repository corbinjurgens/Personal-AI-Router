// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"os"
	"testing"
)

func TestMemberByUUIDNeverMatchesName(t *testing.T) {
	m := newTestManager(t)
	uuid := "aaaaaaaa-0000-4000-8000-000000000000"
	m.upsertMember(&ClusterNode{NodeUUID: uuid, Name: "office-pc", State: stateMember})
	if _, ok := m.memberByUUID("office-pc"); ok {
		t.Fatal("memberByUUID matched a member by name")
	}
	if n, ok := m.memberByUUID(uuid); !ok || n.Name != "office-pc" {
		t.Fatalf("memberByUUID(%s) = %+v, %v", uuid, n, ok)
	}
}

func TestNodesRemoveRequiresNodeUUID(t *testing.T) {
	m := newTestManager(t)
	for _, params := range []string{`{}`, `{"nodeId":"office-pc"}`} {
		out := captureCodec(m)
		id := json.RawMessage(`3`)
		m.handleNodesRemove(&Message{ID: &id, Params: json.RawMessage(params)})
		frames := out.lines(t)
		if len(frames) != 1 {
			t.Fatalf("%s: frames = %d, want 1", params, len(frames))
		}
		var rpcErr struct {
			Code int               `json:"code"`
			Data map[string]string `json:"data"`
		}
		if err := json.Unmarshal(frames[0]["error"], &rpcErr); err != nil {
			t.Fatalf("%s: decode error: %v", params, err)
		}
		if rpcErr.Code != codeInvalidParams || rpcErr.Data["field"] != "nodeUuid" {
			t.Fatalf("%s: error = %+v, want %d with field nodeUuid", params, rpcErr, codeInvalidParams)
		}
	}
}

func TestMembersWithSameNameSortByUUID(t *testing.T) {
	m := newTestManager(t)
	low, high := "11111111-0000-4000-8000-000000000000", "ffffffff-0000-4000-8000-000000000000"
	m.upsertMember(&ClusterNode{NodeUUID: high, Name: "twin", State: stateMember})
	m.upsertMember(&ClusterNode{NodeUUID: low, Name: "twin", State: stateMember})
	m.upsertMember(&ClusterNode{NodeUUID: "00000000-0000-4000-8000-000000000000", Name: "zeta", State: stateMember})
	nodes := m.snapshotNodes()
	if len(nodes) != 3 || nodes[0].NodeUUID != low || nodes[1].NodeUUID != high || nodes[2].Name != "zeta" {
		t.Fatalf("snapshot order = %+v, want twin(low), twin(high), zeta", nodes)
	}
	if err := m.persistMembersErr(); err != nil {
		t.Fatal(err)
	}
	var stored []ClusterNode
	data, err := os.ReadFile(m.membersPath())
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &stored); err != nil {
		t.Fatal(err)
	}
	if len(stored) != 3 || stored[0].NodeUUID != low || stored[1].NodeUUID != high {
		t.Fatalf("persisted order = %+v, want twin(low), twin(high), zeta", stored)
	}
}

func TestNewLeafHasNoDNSNames(t *testing.T) {
	uuid := "bbbbbbbb-0000-4000-8000-000000000000"
	certPEM, _, err := generateLeaf(uuid)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := pem.Decode(certPEM)
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	if len(cert.DNSNames) != 0 {
		t.Fatalf("DNSNames = %v, want none", cert.DNSNames)
	}
	if uuidFromCert(cert) != uuid || cert.Subject.CommonName != uuid {
		t.Fatalf("certificate principal = %q / CN %q, want %s", uuidFromCert(cert), cert.Subject.CommonName, uuid)
	}
}

func TestDuplicateNamedPeersAreBothMembers(t *testing.T) {
	m := newTestManager(t)
	aUUID, aCert, aFP, _ := makeNode(t, "a")
	bUUID, bCert, bFP, _ := makeNode(t, "b")
	pinTrusted(t, m, aUUID, aCert, aFP)
	pinTrusted(t, m, bUUID, bCert, bFP)
	m.upsertMember(&ClusterNode{NodeUUID: aUUID, Name: "same", State: stateMember})
	m.upsertMember(&ClusterNode{NodeUUID: bUUID, Name: "same", State: stateMember})
	for _, uuid := range []string{aUUID, bUUID} {
		roster := &Roster{ClusterID: "cluster-1", Members: []RosterEntry{
			{NodeUUID: aUUID, Name: "same", CertPem: aCert, CertFingerprint: aFP},
			{NodeUUID: bUUID, Name: "same", CertPem: bCert, CertFingerprint: bFP},
		}}
		m.mergeRoster(roster, uuid)
	}
	count := 0
	for _, n := range m.snapshotNodes() {
		if n.Name == "same" {
			count++
		}
	}
	if count != 2 {
		t.Fatalf("members named %q = %d, want 2", "same", count)
	}
	for _, uuid := range []string{aUUID, bUUID} {
		if _, ok := m.trust.Get(uuid); !ok {
			t.Fatalf("pin for %s dropped", uuid)
		}
	}
}
