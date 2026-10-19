// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"slices"
	"strings"
	"testing"

	"nvpair-shared/noderec"
)

// A descriptor engine's service key reaches this node's own directory entry,
// which is what routes it locally, but never the mDNS TXT record.
func TestRegistryKeepsDynamicEngineKeysOutOfTXT(t *testing.T) {
	r := newRegistry("host-1", "clu-1", []string{"192.168.1.10"})
	r.register(noderec.RegisterParams{Service: noderec.ServiceLlamaCPP, Port: 8080})
	before := r.txt()
	if !r.register(noderec.RegisterParams{Service: "e-llamacpp-custom", Port: 18180}) {
		t.Fatal("dynamic engine registration not recorded")
	}
	if port, ok := r.record().Port("e-llamacpp-custom"); !ok || port != 18180 {
		t.Fatalf("self record port = %d, %v", port, ok)
	}
	after := r.txt()
	if !slices.Equal(before, after) {
		t.Fatalf("TXT changed by a dynamic engine:\nbefore %v\n after %v", before, after)
	}
	for _, entry := range after {
		if strings.HasPrefix(entry, "e-") {
			t.Fatalf("TXT carries %q", entry)
		}
	}
	if !slices.Contains(after, "lc=8080") {
		t.Fatalf("built-in engine key missing from TXT: %v", after)
	}
}

// A paired peer's descriptor engine arrives on the authenticated claim and is
// routable from the directory under its key; a malformed dynamic key rejects
// the claim.
func TestPeerServicesCarryDynamicEngineKeys(t *testing.T) {
	d, claim := newPeerServicesScanner(t)
	claim.Descriptor.Services = append(claim.Descriptor.Services, noderec.PeerService{Service: "e-llamacpp-custom", Port: 18180})
	if !d.acceptPeerServices(claim) {
		t.Fatal("claim with a dynamic engine rejected")
	}
	if node := peerServicesNode(t, d); node.Services["e-llamacpp-custom"].Port != 18180 {
		t.Fatalf("directory services = %+v", node.Services)
	}
	if nodes := d.dir.snapshot("e-llamacpp-custom"); len(nodes) != 1 || nodes[0].HostUUID != "peer" {
		t.Fatalf("snapshot filtered by the dynamic key = %+v", nodes)
	}

	d, claim = newPeerServicesScanner(t)
	claim.Descriptor.Services = append(claim.Descriptor.Services, noderec.PeerService{Service: "e-ollama", Port: 18180})
	if d.acceptPeerServices(claim) {
		t.Fatal("claim with a malformed dynamic key accepted")
	}
}
