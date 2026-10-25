// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"context"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grandcat/zeroconf"
	"nvpair-shared/noderec"
)

// TestBrokerAdvertisesNodeRecord exercises the register->advertise path
// end to end: the broker spawns node-info and the promoted daemon, registers
// node-info's service broker-locally, and the daemon folds that into its
// registry and advertises ONE _nvpair-node record carrying the schema key and
// the fixture's node-info port. This is the new consolidated discovery record
// (distinct from node-info's own transitional _nvpair-node-info advertisement).
//
// A private service/domain separates the fixture from installed PAIR. The
// record must also carry the fixture's own node UUID.
func TestBrokerAdvertisesNodeRecord(t *testing.T) {
	namespace := newFixtureMDNSNamespace(t)
	base := t.TempDir()
	hostUUID := fixtureNodeUUID(t, base)
	ports := reserveFixtureServicePorts(t)
	_, msgs, stderr, cleanup := startBrokerWithPorts(t, t.TempDir(), filepath.Join(base, "cluster"), nil, ports,
		"--scanner-mdns=true", "--scanner-mdns-service", namespace.Service, "--scanner-mdns-domain", namespace.Domain,
		"--node-info-path", nodeInfoBin)
	t.Cleanup(cleanup)
	// Drain the broker's stdout/stderr so it doesn't stall on a full pipe while
	// we browse. Both channels close on cleanup, ending these goroutines.
	go func() {
		for range msgs {
		}
	}()
	go func() {
		for range stderr {
		}
	}()

	host, err := os.Hostname()
	if err != nil {
		t.Fatalf("hostname: %v", err)
	}
	instance := strings.Trim(host, ".")
	wantNodeInfo := "ni=" + strconv.Itoa(ports.nodeInfo)

	// The scanner first announces the base node record, then re-announces after
	// node-info registers ni=. Startup ordering can therefore expose a valid
	// base record for a moment; wait for the enriched update instead of treating
	// the first packet as final.
	var entry []string
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if found := browseForNodeRecord(t, namespace, instance, hostUUID, 2*time.Second); found != nil {
			entry = found.Text
			if slices.Contains(entry, wantNodeInfo) {
				break
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	if entry == nil {
		t.Fatalf("did not discover this fixture's _nvpair-node record (uuid=%s) within timeout", hostUUID)
	}
	if !slices.Contains(entry, wantNodeInfo) {
		t.Errorf("_nvpair-node TXT missing %s (register->advertise path): %v", wantNodeInfo, entry)
	}
	if !slices.Contains(entry, "v=1") {
		t.Errorf("_nvpair-node TXT missing schema v=1: %v", entry)
	}
}

// fixtureNodeUUID seeds the node identity the fixture's scanner resolves from
// the parent of its cluster directory, so the fixture's record is told apart
// from any other node advertising on this host.
func fixtureNodeUUID(t *testing.T, base string) string {
	t.Helper()
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		t.Fatal(err)
	}
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	uuid := fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
	if err := os.WriteFile(filepath.Join(base, "node-id.json"), mustMarshal(t, map[string]any{"node_uuid": uuid}), 0o600); err != nil {
		t.Fatal(err)
	}
	return uuid
}

// browseForNodeRecord is browseForInstance restricted to the record whose
// uuid= TXT is hostUUID.
func browseForNodeRecord(t *testing.T, namespace noderec.MDNSNamespace, instance, hostUUID string, timeout time.Duration) *zeroconf.ServiceEntry {
	t.Helper()
	resolver, err := zeroconf.NewResolver(mdnsResolverOptions(nil)...)
	if err != nil {
		t.Fatalf("resolver: %v", err)
	}
	entries := make(chan *zeroconf.ServiceEntry)
	found := make(chan *zeroconf.ServiceEntry, 1)
	go func() {
		for e := range entries {
			if e.Instance == instance && slices.Contains(e.Text, "uuid="+hostUUID) {
				select {
				case found <- e:
				default:
				}
			}
		}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	if err := mdnsBrowse(t, resolver, ctx, namespace.Service, namespace.Domain, entries, nil); err != nil {
		t.Fatalf("browse: %v", err)
	}
	select {
	case e := <-found:
		return e
	case <-ctx.Done():
		return nil
	}
}
