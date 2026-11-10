// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"slices"
	"testing"

	"nvpair-shared/noderec"
)

// nopReadWriter is a codec transport that reads EOF and discards writes, for
// tests that drive the manager's helpers directly.
type nopReadWriter struct{}

func (nopReadWriter) Read([]byte) (int, error)    { return 0, io.EOF }
func (nopReadWriter) Write(p []byte) (int, error) { return len(p), nil }

// TestDirectoryToPeerKeysByHostUUID: a peer is keyed by its stable hostUuid (the
// same identity it stamps as the origin of its errors and uses as its own
// localNodeID), so the self-check and EvictNode survive a peer's PC rename and
// never conflate two same-named peers. Host carries the node's display name and
// is never dialed.
func TestDirectoryToPeerKeysByHostUUID(t *testing.T) {
	n := noderec.DirectoryNode{
		HostUUID: "uuid-x",
		Name:     "host-x",
		IP:       "10.0.0.4",
		IPs:      []string{"10.0.0.4", "192.168.1.4"},
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceErrors: {Port: 14319}},
	}
	p, ok := directoryToPeer(n)
	if !ok {
		t.Fatal("node advertising er with an IP should project")
	}
	if p.ID != "uuid-x" {
		t.Fatalf("peer ID = %q, want hostUuid", p.ID)
	}
	if p.Host != "host-x" {
		t.Fatalf("peer Host = %q, want display name", p.Host)
	}
	if want := []string{"10.0.0.4", "192.168.1.4"}; !slices.Equal(p.Addresses, want) {
		t.Fatalf("peer addresses = %v, want %v", p.Addresses, want)
	}
}

// TestDirectoryPeerWithoutParseableAddressIsNotDialed: a display name is not a
// resolvable host, so a projected peer whose published address does not parse
// yields no host ports even though its Host is non-empty.
func TestDirectoryPeerWithoutParseableAddressIsNotDialed(t *testing.T) {
	n := noderec.DirectoryNode{
		HostUUID: "uuid-x",
		Name:     "Kitchen PC",
		IP:       "not-an-ip",
		Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceErrors: {Port: 14319}},
	}
	p, ok := directoryToPeer(n)
	if !ok {
		t.Fatal("node advertising er with an address entry should project")
	}
	if p.Host != "Kitchen PC" {
		t.Fatalf("peer Host = %q, want display name", p.Host)
	}
	if got := peerHostPorts(p); len(got) != 0 {
		t.Fatalf("peerHostPorts = %v, want none", got)
	}
}

// TestSetLocalNodeID: a new Manager has no node id (the hostname is not an
// identity), and the broker's --node-id sets it so local errors are attributed to
// this node's stable UUID.
func TestSetLocalNodeID(t *testing.T) {
	m := NewManager(NewCodec(nopReadWriter{}))
	if m.LocalNodeID() != "" {
		t.Fatalf("new Manager LocalNodeID = %q, want empty", m.LocalNodeID())
	}
	m.SetLocalNodeID("uuid-self")
	if m.LocalNodeID() != "uuid-self" {
		t.Fatalf("LocalNodeID = %q, want uuid-self", m.LocalNodeID())
	}
	// An empty override must not blank a known id.
	m.SetLocalNodeID("")
	if m.LocalNodeID() != "uuid-self" {
		t.Fatalf("empty override cleared the id: %q", m.LocalNodeID())
	}
}
