// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"

	"nvpair-shared/noderec"
)

const (
	projectionMemberUUID   = "aaaaaaaa-0000-4000-8000-000000000001"
	projectionSelfUUID     = "bbbbbbbb-0000-4000-8000-000000000002"
	projectionStrangerUUID = "cccccccc-0000-4000-8000-000000000003"
)

func projectionScannerNode(uuid, name string) EnrichedNode {
	return directoryToEnriched(noderec.DirectoryNode{HostUUID: uuid, Name: name, IP: "192.0.2.10"})
}

func projectionNode(t *testing.T, s *discoveryStore, uuid string) AvailableNode {
	t.Helper()
	for _, node := range s.Snapshot() {
		if node.HostUUID == uuid {
			return node
		}
	}
	t.Fatalf("node %s missing from snapshot", uuid)
	return AvailableNode{}
}

func TestNodeNameProjectionPrecedence(t *testing.T) {
	test := func(name, uuid, mdnsName string, setup func(*discoveryStore), want string) {
		t.Run(name, func(t *testing.T) {
			s := newDiscoveryStore()
			s.Upsert(projectionScannerNode(uuid, mdnsName), sourceScanner)
			if setup != nil {
				setup(s)
			}
			node := projectionNode(t, s, uuid)
			if node.Name != want {
				t.Fatalf("shown name = %q, want %q", node.Name, want)
			}
			if node.ID != uuid {
				t.Fatalf("id = %q, want the node UUID %q", node.ID, uuid)
			}
		})
	}
	// The cluster manager's displayName already folds the nickname over the
	// member's own name; either way it wins over the scanner's name=.
	test("member nickname wins over the scanner name", projectionMemberUUID, "mdns-name", func(s *discoveryStore) {
		s.SetMemberNames(map[string]string{projectionMemberUUID: "My Nickname"})
	}, "My Nickname")
	test("member without nickname shows its own name over a different name=", projectionMemberUUID, "stale-mdns-name", func(s *discoveryStore) {
		s.SetMemberNames(map[string]string{projectionMemberUUID: "Member Own Name"})
	}, "Member Own Name")
	test("unpaired node shows its name=", projectionStrangerUUID, "Stranger Desk", func(s *discoveryStore) {
		s.SetMemberNames(map[string]string{projectionMemberUUID: "Member"})
	}, "Stranger Desk")
	test("self shows the manager name", projectionSelfUUID, "scanner-self-name", func(s *discoveryStore) {
		s.SetSelf(projectionSelfUUID, "Manager Name")
	}, "Manager Name")
	test("nameless unpaired node shows the fallback", projectionStrangerUUID, "", nil, "PAIR cccccccc")
}

func TestNodeNameProjectionManualNodeKeepsUserName(t *testing.T) {
	s := newDiscoveryStore()
	s.Upsert(manualToEnriched(manualNodeStatus{ID: "Garage box", Address: "192.0.2.20"}), sourceManual)
	node := projectionNode(t, s, "Garage box")
	if node.ID != "Garage box" || node.Name != "Garage box" {
		t.Fatalf("manual node = id %q name %q, want its user-entered id and name", node.ID, node.Name)
	}
}

func TestNodeNameProjectionSameNameSortsByUUID(t *testing.T) {
	s := newDiscoveryStore()
	s.Upsert(projectionScannerNode(projectionStrangerUUID, "Twin"), sourceScanner)
	s.Upsert(projectionScannerNode(projectionMemberUUID, "Twin"), sourceScanner)
	s.Upsert(projectionScannerNode(projectionSelfUUID, "Alpha"), sourceScanner)
	snap := s.Snapshot()
	var got []string
	for _, node := range snap {
		got = append(got, node.Name+"/"+node.HostUUID)
	}
	want := []string{"Alpha/" + projectionSelfUUID, "Twin/" + projectionMemberUUID, "Twin/" + projectionStrangerUUID}
	if len(got) != len(want) {
		t.Fatalf("snapshot = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("snapshot order = %v, want %v", got, want)
		}
	}
}

// A nodes:changed from the cluster manager re-publishes discovery:nodes-changed
// with the new member display name; pending invitees are not members.
func TestNodeNameProjectionNodesChangedRepublishes(t *testing.T) {
	var output bytes.Buffer
	store := newDiscoveryStore()
	b := &Broker{codec: NewCodec(struct {
		io.Reader
		io.Writer
	}{bytes.NewReader(nil), &output}), store: store}
	store.Upsert(projectionScannerNode(projectionMemberUUID, "mdns-member"), sourceScanner)
	store.Upsert(projectionScannerNode(projectionStrangerUUID, "mdns-pending"), sourceScanner)
	var published [][]AvailableNode
	store.SetOnChange(func() { published = append(published, store.Snapshot()) })

	b.forwardClusterManagerNotification("nodes:changed", json.RawMessage(`{"nodes":[
		{"nodeUuid":"`+projectionMemberUUID+`","name":"Own","displayName":"Renamed Member","state":"member"},
		{"nodeUuid":"`+projectionStrangerUUID+`","name":"Pending","displayName":"Pending","state":"pending-outbound"}]}`))

	if len(published) != 1 {
		t.Fatalf("nodes:changed published %d discovery snapshots, want 1", len(published))
	}
	names := map[string]string{}
	for _, node := range published[0] {
		names[node.HostUUID] = node.Name
	}
	if names[projectionMemberUUID] != "Renamed Member" || names[projectionStrangerUUID] != "mdns-pending" {
		t.Fatalf("published names = %v, want the member display name and the pending node's name=", names)
	}
	forwarded, err := NewCodec(bytes.NewBuffer(output.Bytes())).Read()
	if err != nil || forwarded.Method != "nodes:changed" {
		t.Fatalf("client did not receive nodes:changed: %v %+v", err, forwarded)
	}

	b.forwardClusterManagerNotification("nodes:changed", json.RawMessage(`{"nodes":[
		{"nodeUuid":"`+projectionMemberUUID+`","name":"Own","displayName":"Renamed Member","state":"member"}]}`))
	if len(published) != 1 {
		t.Fatalf("an unchanged member set republished (%d snapshots)", len(published))
	}
}

// The cached self name from the cluster manager reaches the projection.
func TestNodeNameProjectionSelfFromRecordedName(t *testing.T) {
	store := newDiscoveryStore()
	b := &Broker{store: store}
	store.Upsert(projectionScannerNode(projectionSelfUUID, ""), sourceScanner)
	b.recordSelfName(nodeNameIdentity{NodeUUID: projectionSelfUUID, Name: "From Manager"})
	if node := projectionNode(t, store, projectionSelfUUID); node.Name != "From Manager" {
		t.Fatalf("self shown name = %q, want From Manager", node.Name)
	}
}
