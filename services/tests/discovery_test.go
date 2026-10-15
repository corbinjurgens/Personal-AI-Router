// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/grandcat/zeroconf"
)

const testDomain = "local."

type mdnsAdvertiserFixture struct {
	Instance  string
	Service   string
	Port      int
	TXT       []string
	Interface string
}

func init() {
	if data := os.Getenv("NVPAIR_TEST_MDNS_ADVERTISER"); data != "" {
		var fixture mdnsAdvertiserFixture
		if err := json.Unmarshal([]byte(data), &fixture); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		iface, err := net.InterfaceByName(fixture.Interface)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		// Register's address enumeration omits loopback addresses. Supply the
		// fixture's loopback A record through the library's proxy registration.
		server, err := zeroconf.RegisterProxy(fixture.Instance, fixture.Service, testDomain, fixture.Port,
			fixture.Instance, []string{"127.0.0.1"}, fixture.TXT, []net.Interface{*iface})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stdout, "advertiser-ready")
		if _, err := io.Copy(io.Discard, os.Stdin); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		server.Shutdown()
		os.Exit(0)
	}
}

func mdnsFixtureInterface(t *testing.T) net.Interface {
	t.Helper()
	ifaces, err := net.Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "windows" {
		// x/net's Windows WriteTo ignores zeroconf's IfIndex control message.
		// Join the interface used by the OS multicast route so queries and
		// responses arrive on the same interface. DialUDP selects that route
		// without sending a packet; choosing the first adapter is ambiguous
		// on hosts with Ethernet, Wi-Fi and virtual adapters.
		conn, err := net.DialUDP("udp4", nil, &net.UDPAddr{IP: net.IPv4(224, 0, 0, 251), Port: 5353})
		if err != nil {
			t.Fatalf("select mDNS multicast route: %v; interfaces: %v", err, ifaces)
		}
		source := conn.LocalAddr().(*net.UDPAddr).IP
		if err := conn.Close(); err != nil {
			t.Fatalf("close mDNS route probe: %v", err)
		}
		for _, iface := range ifaces {
			if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagMulticast == 0 || iface.Flags&net.FlagLoopback != 0 {
				continue
			}
			addrs, err := iface.Addrs()
			if err != nil {
				t.Fatalf("addresses for mDNS interface %s: %v", iface.Name, err)
			}
			for _, addr := range addrs {
				ip, _, err := net.ParseCIDR(addr.String())
				if err != nil {
					t.Fatalf("parse mDNS interface %s address %q: %v", iface.Name, addr, err)
				}
				if ip.Equal(source) {
					t.Logf("mDNS fixture interface: %s index=%d flags=%s multicast source=%s", iface.Name, iface.Index, iface.Flags, source)
					return iface
				}
			}
		}
		t.Fatalf("no active multicast interface owns mDNS route source %s; interfaces: %v", source, ifaces)
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
	options := []zeroconf.ClientOption{zeroconf.SelectIfaces(ifaces)}
	if runtime.GOOS == "windows" {
		// The fixture selects Windows' IPv4 multicast route and publishes an
		// A record. IPv6 need not be enabled on that adapter. Unix retains
		// both transports, including IPv6 multicast over macOS loopback.
		options = append(options, zeroconf.SelectIPTraffic(zeroconf.IPv4))
	}
	return options
}

// Keep zeroconf's goodbye writes out of the test process. A stalled multicast
// interface must not prevent a failed assertion from returning to the suite.
func startMDNSAdvertiser(t *testing.T, fixture mdnsAdvertiserFixture) func() {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(executable)
	cmd.Env = append(os.Environ(), "NVPAIR_TEST_MDNS_ADVERTISER="+string(mustMarshal(t, fixture)))
	var output serviceOutput
	cmd.Stderr = &output
	cmd.WaitDelay = time.Second
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start advertiser %s: %v", fixture.Instance, err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			if err := stdin.Close(); err != nil {
				t.Errorf("close advertiser stdin: %v", err)
			}
			select {
			case err := <-done:
				if err != nil {
					t.Errorf("advertiser %s exit: %v; stderr: %s", fixture.Instance, err, output.String())
				}
			case <-time.After(3 * time.Second):
				t.Logf("advertiser %s goodbye exceeded grace; killing pid=%d", fixture.Instance, cmd.Process.Pid)
				if err := cmd.Process.Kill(); err != nil {
					t.Errorf("kill advertiser: %v", err)
				}
				select {
				case <-done:
				case <-time.After(2 * time.Second):
					t.Errorf("advertiser did not exit after kill; stderr: %s", output.String())
				}
			}
		})
	}
	t.Cleanup(stop)
	ready := make(chan bool, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		ready <- scanner.Scan() && scanner.Text() == "advertiser-ready"
	}()
	select {
	case ok := <-ready:
		if !ok {
			t.Fatalf("advertiser %s did not become ready; stderr: %s", fixture.Instance, output.String())
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("advertiser %s readiness timed out; stderr: %s", fixture.Instance, output.String())
	}
	return stop
}

func TestInProcessDiscovery(t *testing.T) {
	const (
		service  = "_test-disc._tcp"
		instance = "in-proc-test"
		port     = 55555
	)

	iface := mdnsFixtureInterface(t)
	startMDNSAdvertiser(t, mdnsAdvertiserFixture{Instance: instance, Service: service, Port: port, TXT: []string{"env=test"}, Interface: iface.Name})
	time.Sleep(500 * time.Millisecond)

	resolver, err := zeroconf.NewResolver(mdnsResolverOptions([]net.Interface{iface})...)
	if err != nil {
		t.Fatalf("resolver: %v", err)
	}

	entries := make(chan *zeroconf.ServiceEntry)
	found := make(chan *zeroconf.ServiceEntry, 1)
	go func() {
		for e := range entries {
			if e.Instance == instance {
				select {
				case found <- e:
				default:
				}
			}
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := resolver.Browse(ctx, service, testDomain, entries); err != nil {
		t.Fatalf("browse: %v", err)
	}

	select {
	case entry := <-found:
		if entry.Port != port {
			t.Errorf("port = %d, want %d", entry.Port, port)
		}
		if len(entry.Text) == 0 || entry.Text[0] != "env=test" {
			t.Errorf("txt = %v, want [env=test]", entry.Text)
		}
		if !hasAddress(entry) {
			t.Error("no addresses resolved")
		}
		t.Logf("OK: %s @ %s:%d addrs=%v txt=%v",
			entry.Instance, entry.HostName, entry.Port, entry.AddrIPv4, entry.Text)
	case <-ctx.Done():
		t.Fatal("timed out waiting for discovery")
	}
}

func TestInProcessMultipleInstances(t *testing.T) {
	const service = "_test-multi._tcp"
	instances := []string{"node-alpha", "node-beta", "node-gamma"}

	iface := mdnsFixtureInterface(t)
	for i, inst := range instances {
		startMDNSAdvertiser(t, mdnsAdvertiserFixture{Instance: inst, Service: service, Port: 50000 + i, Interface: iface.Name})
	}
	time.Sleep(500 * time.Millisecond)

	resolver, err := zeroconf.NewResolver(mdnsResolverOptions([]net.Interface{iface})...)
	if err != nil {
		t.Fatalf("resolver: %v", err)
	}

	entries := make(chan *zeroconf.ServiceEntry)
	foundSet := make(map[string]bool)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for e := range entries {
			for _, inst := range instances {
				if e.Instance == inst {
					foundSet[inst] = true
				}
			}
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := resolver.Browse(ctx, service, testDomain, entries); err != nil {
		t.Fatalf("browse: %v", err)
	}
	<-ctx.Done()
	<-done

	for _, inst := range instances {
		if !foundSet[inst] {
			t.Errorf("instance %q not discovered", inst)
		}
	}
	t.Logf("OK: discovered %d/%d instances", len(foundSet), len(instances))
}

func TestInProcessRemoval(t *testing.T) {
	const (
		service  = "_test-rmv._tcp"
		instance = "removable-node"
		port     = 55556
	)

	iface := mdnsFixtureInterface(t)
	stop := startMDNSAdvertiser(t, mdnsAdvertiserFixture{Instance: instance, Service: service, Port: port, Interface: iface.Name})

	// Phase 1: verify discoverable
	entry := browseForInstanceOnInterfaces(t, service, instance, 5*time.Second, []net.Interface{iface})
	if entry == nil {
		t.Fatal("service not found before removal")
	}
	t.Log("phase 1: service discovered")

	// Shut down the service
	stop()
	time.Sleep(time.Second)

	// Phase 2: verify no longer discoverable
	entry = browseForInstanceOnInterfaces(t, service, instance, 5*time.Second, []net.Interface{iface})
	if entry != nil {
		t.Error("service still discoverable after shutdown")
	} else {
		t.Log("phase 2: service correctly absent after removal")
	}
}

// --- helpers ---

func browseForInstance(t *testing.T, service, instance string, timeout time.Duration) *zeroconf.ServiceEntry {
	t.Helper()
	return browseForInstanceOnInterfaces(t, service, instance, timeout, nil)
}

func browseForInstanceOnInterfaces(t *testing.T, service, instance string, timeout time.Duration, ifaces []net.Interface) *zeroconf.ServiceEntry {
	t.Helper()
	resolver, err := zeroconf.NewResolver(mdnsResolverOptions(ifaces)...)
	if err != nil {
		t.Fatalf("resolver: %v", err)
	}

	entries := make(chan *zeroconf.ServiceEntry)
	found := make(chan *zeroconf.ServiceEntry, 1)
	go func() {
		for e := range entries {
			if e.Instance == instance {
				select {
				case found <- e:
				default:
				}
			}
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	if err := resolver.Browse(ctx, service, testDomain, entries); err != nil {
		t.Fatalf("browse: %v", err)
	}

	select {
	case e := <-found:
		return e
	case <-ctx.Done():
		return nil
	}
}

func hasAddress(e *zeroconf.ServiceEntry) bool {
	for _, ip := range e.AddrIPv4 {
		if ip != nil && !ip.IsUnspecified() {
			return true
		}
	}
	for _, ip := range e.AddrIPv6 {
		if ip != nil && !net.IP(ip).IsLinkLocalUnicast() {
			return true
		}
	}
	return false
}
