// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package tests

import (
	"context"
	"net"
	"testing"

	"github.com/grandcat/zeroconf"
)

func mdnsFixtureInterface(t *testing.T) net.Interface {
	t.Helper()
	ifaces, err := net.Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp != 0 && iface.Flags&net.FlagLoopback != 0 {
			t.Logf("mDNS fixture interface: %s index=%d flags=%s", iface.Name, iface.Index, iface.Flags)
			return iface
		}
	}
	t.Fatal("no active loopback interface for multicast fixture")
	return net.Interface{}
}

func mdnsResolverOptions(ifaces []net.Interface) []zeroconf.ClientOption {
	if ifaces == nil {
		return nil
	}
	return []zeroconf.ClientOption{zeroconf.SelectIfaces(ifaces)}
}

func registerMDNSFixture(fixture mdnsAdvertiserFixture, iface net.Interface) (func() error, error) {
	// Register's address enumeration omits loopback addresses. Keep the
	// existing proxy registration and both IP transports on Unix.
	server, err := zeroconf.RegisterProxy(fixture.Instance, fixture.Service, fixture.Domain, fixture.Port,
		fixture.Instance+"."+fixture.Domain+".", []string{"127.0.0.1"}, fixture.TXT, []net.Interface{iface})
	if err != nil {
		return nil, err
	}
	return func() error { server.Shutdown(); return nil }, nil
}

func mdnsBrowse(t *testing.T, resolver *zeroconf.Resolver, ctx context.Context, service, domain string, entries chan<- *zeroconf.ServiceEntry, _ []net.Interface) error {
	t.Helper()
	return resolver.Browse(ctx, service, domain, entries)
}

func mdnsFixtureLogLine(string) {}

func mdnsFixtureAdvertiserOutput(*testing.T, string) {}
