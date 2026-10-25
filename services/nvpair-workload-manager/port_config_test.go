// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"reflect"
	"testing"

	"nvpair-shared/noderec"
)

func TestWorkloadPeersRequireAdvertisedPort(t *testing.T) {
	for _, port := range []int{0, -1, 65536} {
		p := newPeerSet()
		p.Replace([]PeerNode{{ID: "fixture-peer", Addresses: []string{"127.0.0.1"}, Port: port}})
		if got := p.targets(); len(got) != 0 {
			t.Fatalf("invalid advertised port %d produced dial targets %+v", port, got)
		}
	}
}

func TestWorkloadRelayUsesPeerPortIndependentlyOfLocalListener(t *testing.T) {
	src := newRelayPeerSource("fixture-self")
	src.set([]noderec.DirectoryNode{
		{HostUUID: "fixture-peer", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceWorkload: {Port: 25121}}},
		{HostUUID: "missing-port", IP: "127.0.0.1", Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceWorkload: {Port: 0}}},
	})
	nodes, err := src.Nodes(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	m := NewManager(NewCodec(codecNop{}), 25120, "fixture-self", t.TempDir())
	m.peers.Replace(nodes)
	targets := m.peers.targets()
	if len(targets) != 1 || targets[0].id != "fixture-peer" || !reflect.DeepEqual(targets[0].candidates, []string{"127.0.0.1:25121"}) || m.server.port != 25120 {
		t.Fatalf("local=%d relay targets=%+v", m.server.port, targets)
	}
}

func TestWorkloadListenerPortValidation(t *testing.T) {
	for _, port := range []int{0, -1, 65536} {
		if err := validateListenerPort(port); err == nil {
			t.Fatalf("listener port %d accepted", port)
		}
	}
	for _, port := range []int{1, 25120, defaultPort, 65535} {
		if err := validateListenerPort(port); err != nil {
			t.Fatalf("listener port %d refused: %v", port, err)
		}
	}
}
