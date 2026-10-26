// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net"
	"testing"
)

func TestTCPPortOccupancyRejectsBoundAddresses(t *testing.T) {
	test := func(name, network, address string, optional bool) {
		t.Run(name, func(t *testing.T) {
			owner, err := net.Listen(network, address)
			if err != nil {
				if optional {
					t.Skipf("IPv6 listener unavailable on this host: %v", err)
				}
				t.Fatalf("reserve occupied endpoint: %v", err)
			}
			t.Cleanup(func() {
				if err := owner.Close(); err != nil {
					t.Errorf("close occupied endpoint: %v", err)
				}
			})
			if port := owner.Addr().(*net.TCPAddr).Port; tcpPortAvailable(port) {
				t.Fatalf("preflight accepted port %d occupied at %s", port, owner.Addr())
			}
		})
	}
	test("IPv4 loopback", "tcp4", "127.0.0.1:0", false)
	test("IPv6 loopback", "tcp6", "[::1]:0", true)
	test("IPv4 wildcard", "tcp4", "0.0.0.0:0", false)
	test("IPv6 wildcard", "tcp6", "[::]:0", true)
}

func TestTCPPortOccupancyAcceptsEphemeralReservation(t *testing.T) {
	// Port zero exercises the free path without a reserve-release assumption.
	if !tcpPortAvailable(0) {
		t.Fatal("preflight refused an unallocated ephemeral port")
	}
}
