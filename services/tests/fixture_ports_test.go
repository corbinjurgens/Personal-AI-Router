// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"fmt"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"
)

// Fixture-owned brokers never use the product's default service ports. An
// installed PAIR holds those listeners on the same host, and a fixture that
// binds or dials them either fails to start or exchanges traffic with the
// installed service instead of its own workers.

// installedServicePorts are the default listeners of an installed PAIR: the
// broker's local services (ni, er, wl, cl, em, ec) and the engine
// compatibility facades. No fixture is ever assigned one of them.
var installedServicePorts = map[int]string{
	14318: "node-info",
	14319: "errors",
	14320: "workload-manager",
	14321: "cluster-manager",
	14322: "engine-manager model service",
	14323: "engine-manager control service",
	11434: "Ollama facade",
	1234:  "LM Studio facade",
	8080:  "llama.cpp facade",
}

// fixtureServicePorts is one fixture broker's complete local service plan,
// passed through the broker's per-service port flags.
type fixtureServicePorts struct {
	nodeInfo          int
	errors            int
	workload          int
	clusterManager    int
	engineManagerHTTP int
	engineControl     int
}

// reserveFixtureServicePorts reserves six distinct ephemeral ports for one
// fixture broker. All six listeners are held at once so the OS cannot hand the
// same port out twice, then released so the workers can bind them.
//
// The reserve-then-release window is a small race: another process may take a
// port between the release and the worker's bind. That makes the worker's bind
// fail and the test fail loudly; it never redirects fixture traffic to the
// installed service, whose ports are excluded here.
func reserveFixtureServicePorts(t *testing.T) fixtureServicePorts {
	t.Helper()
	ports := make([]int, 6)
	listeners := make([]net.Listener, 0, len(ports))
	defer func() {
		for _, ln := range listeners {
			if err := ln.Close(); err != nil {
				t.Errorf("release reserved fixture port: %v", err)
			}
		}
	}()
	for i := range ports {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("reserve fixture service port: %v", err)
		}
		listeners = append(listeners, ln)
		ports[i] = ln.Addr().(*net.TCPAddr).Port
		if owner, installed := installedServicePorts[ports[i]]; installed {
			t.Fatalf("OS selected installed PAIR port %d (%s) for a fixture", ports[i], owner)
		}
	}
	plan := fixtureServicePorts{
		nodeInfo:          ports[0],
		errors:            ports[1],
		workload:          ports[2],
		clusterManager:    ports[3],
		engineManagerHTTP: ports[4],
		engineControl:     ports[5],
	}
	t.Logf("fixture_service_ports %s", plan)
	return plan
}

func (p fixtureServicePorts) String() string {
	return fmt.Sprintf("ni=%d er=%d wl=%d cl=%d em=%d ec=%d",
		p.nodeInfo, p.errors, p.workload, p.clusterManager, p.engineManagerHTTP, p.engineControl)
}

// args are the broker flags that apply the plan.
func (p fixtureServicePorts) args() []string {
	return []string{
		"--node-info-port", strconv.Itoa(p.nodeInfo),
		"--errors-port", strconv.Itoa(p.errors),
		"--workload-manager-port", strconv.Itoa(p.workload),
		"--cluster-manager-port", strconv.Itoa(p.clusterManager),
		"--engine-manager-http-port", strconv.Itoa(p.engineManagerHTTP),
		"--engine-manager-control-port", strconv.Itoa(p.engineControl),
	}
}

// portOccupant names the first address on which port is already bound, or ""
// when it is free on IPv4 and IPv6. Wildcards are probed first: a probe never
// dials, so an occupant receives no connection, and probing loopback only after
// the wildcards bind means it cannot shadow a wildcard owner's loopback traffic.
// IPv6 is skipped only on a host without IPv6 loopback.
//
// Probing only 127.0.0.1 misses an IPv6-only owner, and on macOS a loopback
// bind can succeed beside another process's IPv4 wildcard listener. On Windows
// every family-specific probe binds beside a dual-stack wildcard owner, which
// only an identical dual-stack probe detects.
func portOccupant(port int) string {
	addrs := []struct{ network, host string }{
		{"tcp4", "0.0.0.0"},
		{"tcp6", "::"},
		{"tcp", ""},
		{"tcp4", "127.0.0.1"},
		{"tcp6", "::1"},
	}
	for _, addr := range addrs {
		if addr.network == "tcp6" && !ipv6LoopbackAvailable() {
			continue
		}
		address := net.JoinHostPort(addr.host, strconv.Itoa(port))
		ln, err := net.Listen(addr.network, address)
		if err != nil {
			return address
		}
		_ = ln.Close()
	}
	return ""
}

var ipv6LoopbackAvailable = sync.OnceValue(func() bool {
	ln, err := net.Listen("tcp6", "[::1]:0")
	if err != nil {
		return false
	}
	_ = ln.Close()
	return true
})

// portBusy reports whether port is bound on any IPv4 or IPv6 address. Tests
// that need a fixed product port skip on it rather than touch the owner.
func portBusy(port int) bool { return portOccupant(port) != "" }

// skipIfDefaultPortOccupied skips a test that deliberately needs a product
// default port when anything already holds it on either address family. The
// occupant is typically an installed PAIR; the test must not bind beside it or
// send it requests.
func skipIfDefaultPortOccupied(t *testing.T, port int, need string) {
	t.Helper()
	if address := portOccupant(port); address != "" {
		t.Skipf("default port %d is occupied (bind %s failed); %s needs it and must not touch the installed service", port, address, need)
	}
}

// acceptCounter is a synthetic occupant: wildcard listeners on one port that
// record the peer of every connection they accept.
type acceptCounter struct {
	listeners []net.Listener
	mu        sync.Mutex
	remotes   []net.Addr
	done      sync.WaitGroup
}

// occupyWildcard binds port on the given wildcard networks: "tcp4", "tcp6", or
// "tcp" for the dual-stack wildcard an installed PAIR worker binds.
func occupyWildcard(t *testing.T, port int, networks ...string) *acceptCounter {
	t.Helper()
	c := &acceptCounter{}
	for _, network := range networks {
		host := map[string]string{"tcp4": "0.0.0.0", "tcp6": "::", "tcp": ""}[network]
		ln, err := net.Listen(network, net.JoinHostPort(host, strconv.Itoa(port)))
		if err != nil {
			c.close()
			t.Fatalf("occupy %s port %d: %v", network, port, err)
		}
		c.listeners = append(c.listeners, ln)
		c.done.Add(1)
		go func() {
			defer c.done.Done()
			for {
				conn, err := ln.Accept()
				if err != nil {
					return
				}
				c.mu.Lock()
				c.remotes = append(c.remotes, conn.RemoteAddr())
				c.mu.Unlock()
				_ = conn.Close()
			}
		}()
	}
	t.Cleanup(c.close)
	return c
}

func (c *acceptCounter) close() {
	for _, ln := range c.listeners {
		_ = ln.Close()
	}
	c.done.Wait()
	c.listeners = nil
}

// assertNoAccepts fails on any connection that came from this host. A LAN peer
// that still remembers this host's default ports may dial in; that is logged,
// not counted, because it is not fixture traffic.
func (c *acceptCounter) assertNoAccepts(t *testing.T, what string) {
	t.Helper()
	local := map[string]bool{}
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		t.Fatalf("list local addresses: %v", err)
	}
	for _, addr := range addrs {
		if prefix, ok := addr.(*net.IPNet); ok {
			local[prefix.IP.String()] = true
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	var fromHost, fromPeers []string
	for _, remote := range c.remotes {
		tcp, ok := remote.(*net.TCPAddr)
		if ok && (tcp.IP.IsLoopback() || local[tcp.IP.String()]) {
			fromHost = append(fromHost, remote.String())
		} else {
			fromPeers = append(fromPeers, remote.String())
		}
	}
	if len(fromPeers) != 0 {
		t.Logf("%s accepted %d connection(s) from other hosts: %v", what, len(fromPeers), fromPeers)
	}
	if len(fromHost) != 0 {
		t.Errorf("%s accepted %d connection(s) from this host %v; fixtures must not reach it", what, len(fromHost), fromHost)
	}
}

func TestReserveFixtureServicePorts(t *testing.T) {
	first := reserveFixtureServicePorts(t)
	second := reserveFixtureServicePorts(t)
	for _, plan := range []fixtureServicePorts{first, second} {
		seen := map[int]bool{}
		for _, port := range []int{plan.nodeInfo, plan.errors, plan.workload, plan.clusterManager, plan.engineManagerHTTP, plan.engineControl} {
			if seen[port] {
				t.Fatalf("plan %s repeats port %d", plan, port)
			}
			seen[port] = true
			if _, installed := installedServicePorts[port]; installed {
				t.Fatalf("plan %s uses installed PAIR port %d", plan, port)
			}
			if address := portOccupant(port); address != "" {
				t.Fatalf("plan %s port %d is still held at %s after release", plan, port, address)
			}
		}
	}
	args := first.args()
	if len(args) != 12 {
		t.Fatalf("plan args = %q, want one flag and value per service", args)
	}
	for i, flag := range []string{"--node-info-port", "--errors-port", "--workload-manager-port", "--cluster-manager-port", "--engine-manager-http-port", "--engine-manager-control-port"} {
		if args[2*i] != flag {
			t.Fatalf("plan args = %q, want %s at %d", args, flag, 2*i)
		}
	}
}

// A synthetic wildcard owner on either address family is detected, and the
// detection itself never connects to it.
func TestPortOccupancyCoversBothAddressFamilies(t *testing.T) {
	cases := []struct {
		name     string
		networks []string
		ipv6     bool
	}{
		{name: "ipv4 wildcard", networks: []string{"tcp4"}},
		{name: "ipv6 wildcard", networks: []string{"tcp6"}, ipv6: true},
		{name: "ipv4 and ipv6 wildcards", networks: []string{"tcp4", "tcp6"}, ipv6: true},
		{name: "dual-stack wildcard", networks: []string{"tcp"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.ipv6 && !ipv6LoopbackAvailable() {
				t.Skip("host has no IPv6 loopback")
			}
			port := freePort(t)
			if address := portOccupant(port); address != "" {
				t.Fatalf("free port %d reported occupied at %s", port, address)
			}
			occupant := occupyWildcard(t, port, tc.networks...)
			if address := portOccupant(port); address == "" {
				t.Fatalf("port %d held on %v reported free", port, tc.networks)
			} else {
				t.Logf("port %d held on %v detected at %s", port, tc.networks, address)
			}
			if !portBusy(port) {
				t.Fatalf("portBusy(%d) = false with owners on %v", port, tc.networks)
			}
			// Any connection would arrive promptly; give the accept loops a moment.
			time.Sleep(100 * time.Millisecond)
			occupant.close()
			occupant.assertNoAccepts(t, "synthetic wildcard owner")
		})
	}
}

// TestFixtureBrokerLeavesDefaultServicePortsAlone runs a fixture broker with
// every port-owning service worker beside synthetic IPv4 and IPv6 wildcard
// owners of all six default service ports, and requires the workers to serve
// on the reserved plan while the synthetic owners accept nothing.
//
// It binds the default ports, so it runs only where they are all free and
// skips beside an installed PAIR.
func TestFixtureBrokerLeavesDefaultServicePortsAlone(t *testing.T) {
	defaults := []int{14318, 14319, 14320, 14321, 14322, 14323}
	for _, port := range defaults {
		skipIfDefaultPortOccupied(t, port, "the synthetic default-port owner")
	}
	networks := []string{"tcp4"}
	if ipv6LoopbackAvailable() {
		networks = append(networks, "tcp6")
	}
	owners := make(map[int]*acceptCounter, len(defaults))
	for _, port := range defaults {
		owners[port] = occupyWildcard(t, port, networks...)
	}

	ports := reserveFixtureServicePorts(t)
	_, msgs, _, stopBroker := startBrokerWithPorts(t, t.TempDir(), t.TempDir(), nil, ports,
		"--node-info-path", nodeInfoBin,
		"--errors-path", errorsBin,
		"--workload-manager-path", workloadMgrBin,
		"--cluster-manager-path", clusterMgrBin,
		"--engine-manager-path", engineMgrBin,
		"--proxy-engines=",
	)
	cleanup := sync.OnceFunc(stopBroker)
	t.Cleanup(cleanup)
	go func() {
		for range msgs {
		}
	}()
	for _, port := range []int{ports.nodeInfo, ports.errors, ports.workload, ports.clusterManager, ports.engineManagerHTTP, ports.engineControl} {
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 30*time.Second)
	}
	// Let the scanner enrich self and the workers settle on their plan.
	time.Sleep(3 * time.Second)
	cleanup()
	for _, port := range defaults {
		owners[port].close()
		owners[port].assertNoAccepts(t, fmt.Sprintf("synthetic owner of default port %d (%s)", port, installedServicePorts[port]))
	}
}

// TestFixtureBrokerWorkersListenOnReservedPorts checks that the plan reaches
// the workers: node-info, errors and workload-manager serve on the reserved
// ports. The scanner is the mDNS-free stub and no engine worker or proxy runs,
// so the fixture neither discovers nor dials another node or engine.
func TestFixtureBrokerWorkersListenOnReservedPorts(t *testing.T) {
	ports := reserveFixtureServicePorts(t)
	p, _ := startSubscriptionBroker(t, ports,
		"--node-info-path", nodeInfoBin,
		"--errors-path", errorsBin,
		"--workload-manager-path", workloadMgrBin,
		"--proxy-engines=",
	)
	for _, path := range []string{nodeInfoBin, errorsBin, workloadMgrBin} {
		canonical, err := canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		p.paths = append(p.paths, canonical)
	}
	waitForMethod(t, p.msgs, "app:ready", 10*time.Second)
	for _, port := range []int{ports.nodeInfo, ports.errors, ports.workload} {
		waitTCP(t, net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 15*time.Second)
		p.ports = append(p.ports, port)
	}
}
