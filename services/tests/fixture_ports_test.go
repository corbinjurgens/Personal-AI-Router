// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
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

// listenFixtureLoopback makes one bounded allocation attempt. A protected port
// is closed and rejected before a fixture can serve or dial it.
func listenFixtureLoopback(t *testing.T) net.Listener {
	t.Helper()
	ln, err := allocateFixtureLoopback(net.Listen)
	if err != nil {
		t.Fatalf("allocate fixture listener: %v", err)
	}
	return ln
}

func allocateFixtureLoopback(listen func(string, string) (net.Listener, error)) (net.Listener, error) {
	ln, err := listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if owner, installed := installedServicePorts[port]; installed {
		err := fmt.Errorf("OS selected installed PAIR port %d (%s) for a fixture", port, owner)
		if closeErr := ln.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close rejected fixture listener: %w", closeErr))
		}
		return nil, err
	}
	return ln, nil
}

func newFixtureHTTPServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	server := &httptest.Server{Listener: listenFixtureLoopback(t), Config: &http.Server{Handler: handler}}
	server.Start()
	return server
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
		ln := listenFixtureLoopback(t)
		listeners = append(listeners, ln)
		ports[i] = ln.Addr().(*net.TCPAddr).Port
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

// validate rejects unsafe synthetic plans as well as allocator mistakes before
// a fixture can launch a process or construct an inbound request.
func (p fixtureServicePorts) validate() error {
	seen := make(map[int]bool)
	for _, port := range []int{p.nodeInfo, p.errors, p.workload, p.clusterManager, p.engineManagerHTTP, p.engineControl} {
		if err := validateFixturePort(port); err != nil {
			return err
		}
		if seen[port] {
			return fmt.Errorf("fixture service plan repeats port %d", port)
		}
		seen[port] = true
	}
	return nil
}

func validateFixturePort(port int) error {
	if port < 1 || port > 65535 {
		return fmt.Errorf("fixture port %d must be between 1 and 65535", port)
	}
	if owner, installed := installedServicePorts[port]; installed {
		return fmt.Errorf("fixture port %d belongs to installed PAIR (%s)", port, owner)
	}
	return nil
}

// brokerArgs keeps the worker listeners and discovery registrations on the
// same explicit plan. Extra arguments cannot override one of its port flags.
func (p fixtureServicePorts) brokerArgs(clusterDir string, extraArgs ...string) ([]string, error) {
	if err := p.validate(); err != nil {
		return nil, err
	}
	portArgs := p.args()
	for _, arg := range extraArgs {
		flag, _, _ := strings.Cut(arg, "=")
		if !strings.HasPrefix(flag, "-") {
			continue
		}
		name := strings.TrimLeft(flag, "-")
		for i := 0; i < len(portArgs); i += 2 {
			if name == strings.TrimLeft(portArgs[i], "-") {
				return nil, fmt.Errorf("fixture service flag %s must come from its port plan", flag)
			}
		}
	}
	args := append([]string{"--scanner-path", scannerBin, "--cluster-dir", clusterDir}, portArgs...)
	return append(args, extraArgs...), nil
}

func (p fixtureServicePorts) workloadEventsURL() (string, error) {
	if err := p.validate(); err != nil {
		return "", err
	}
	return fmt.Sprintf("https://127.0.0.1:%d/v1/workloads/events", p.workload), nil
}

type fixtureAllocationListener struct {
	port     int
	closes   int
	closeErr error
}

func (l *fixtureAllocationListener) Accept() (net.Conn, error) {
	return nil, errors.New("allocation guard must not accept connections")
}

func (l *fixtureAllocationListener) Close() error {
	l.closes++
	return l.closeErr
}

func (l *fixtureAllocationListener) Addr() net.Addr {
	return &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: l.port}
}

func TestFixtureLoopbackRejectsInstalledPorts(t *testing.T) {
	for _, port := range []int{8080, 11434, 1234, 14318, 14319, 14320, 14321, 14322, 14323} {
		t.Run(strconv.Itoa(port), func(t *testing.T) {
			candidate := &fixtureAllocationListener{port: port}
			calls := 0
			ln, err := allocateFixtureLoopback(func(network, address string) (net.Listener, error) {
				calls++
				if network != "tcp" || address != "127.0.0.1:0" {
					t.Fatalf("allocation = %s %s, want tcp 127.0.0.1:0", network, address)
				}
				return candidate, nil
			})
			if err == nil || ln != nil || candidate.closes != 1 || calls != 1 {
				t.Fatalf("rejected allocation: listener=%v err=%v closes=%d calls=%d, want nil, error, 1, 1", ln, err, candidate.closes, calls)
			}
		})
	}
}

func TestFixtureLoopbackPreservesAllowedListener(t *testing.T) {
	candidate := &fixtureAllocationListener{port: 24001}
	ln, err := allocateFixtureLoopback(func(string, string) (net.Listener, error) { return candidate, nil })
	if err != nil || ln != candidate || candidate.closes != 0 {
		t.Fatalf("allowed allocation: listener=%v err=%v closes=%d, want original listener, nil, 0", ln, err, candidate.closes)
	}
}

func TestFixtureLoopbackReportsAllocationFailure(t *testing.T) {
	want := errors.New("allocation failed")
	ln, err := allocateFixtureLoopback(func(string, string) (net.Listener, error) { return nil, want })
	if ln != nil || !errors.Is(err, want) {
		t.Fatalf("failed allocation: listener=%v err=%v, want nil and allocation error", ln, err)
	}
}

func TestFixtureLoopbackReportsRejectedCloseFailure(t *testing.T) {
	want := errors.New("close failed")
	candidate := &fixtureAllocationListener{port: 14323, closeErr: want}
	ln, err := allocateFixtureLoopback(func(string, string) (net.Listener, error) { return candidate, nil })
	if ln != nil || !errors.Is(err, want) || candidate.closes != 1 {
		t.Fatalf("rejected close failure: listener=%v err=%v closes=%d, want nil, close error, 1", ln, err, candidate.closes)
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
