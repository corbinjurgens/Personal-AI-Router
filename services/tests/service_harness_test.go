// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/clustertrust"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/servicectl"
	"nvpair-shared/testbin"
)

const serviceCallTimeout = 10 * time.Second

// The service allows 2s for shutdown's reply and 18s for broker teardown.
const serviceStopTimeout = 35 * time.Second

// All current service-fixture callers are serial and hold one fixture at a time.
// Keep the slots aligned with the service-bundle roles in programs/tests.txt.
var serviceBundleSlots = struct {
	sync.Mutex
	used [1]bool
}{}

func acquireServiceBundle(t *testing.T) int {
	t.Helper()
	serviceBundleSlots.Lock()
	defer serviceBundleSlots.Unlock()
	for slot, used := range serviceBundleSlots.used {
		if !used {
			serviceBundleSlots.used[slot] = true
			return slot
		}
	}
	t.Fatalf("service fixture slots exhausted: manifest supports %d concurrent fixtures", len(serviceBundleSlots.used))
	return 0
}

type serviceOutput struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (o *serviceOutput) Write(p []byte) (int, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	n := len(p)
	if remaining := 128*1024 - o.buf.Len(); remaining > 0 {
		if len(p) > remaining {
			p = p[:remaining]
		}
		_, _ = o.buf.Write(p)
	}
	return n, nil
}
func (o *serviceOutput) String() string { o.mu.Lock(); defer o.mu.Unlock(); return o.buf.String() }

type serviceChild struct {
	cmd    *exec.Cmd
	done   chan struct{}
	err    error
	output *serviceOutput
}

type serviceFixture struct {
	t                                    *testing.T
	dir, configDir, clusterDir, endpoint string
	binDir                               string
	cleanupBins                          func()
	servicePath, brokerPath, scannerPath string
	proxyPath                            string
	facadePort                           int
	ports                                fixtureServicePorts
	env                                  []string
	inventory                            *serviceProcessInventory
	children                             []*serviceChild
	clients                              []*serviceRPCClient
	service                              *serviceChild
	cleanupVerified                      bool
}

func newServiceFixture(t *testing.T) *serviceFixture {
	t.Helper()
	dir, err := os.MkdirTemp("", "nvs-")
	if err != nil {
		t.Fatal(err)
	}
	dir, err = canonicalServicePath(dir)
	if err != nil {
		t.Fatal(err)
	}
	f := &serviceFixture{t: t, dir: dir, configDir: filepath.Join(dir, "config"), clusterDir: filepath.Join(dir, "cluster")}
	slot := acquireServiceBundle(t)
	programs := []string{"nvpair-service", "nvpair-ui-broker", "nvpair-node-scanner", "nvpair-proxy", "nvpair-cluster-manager", "nvpair-engine-manager"}
	t.Cleanup(func() {
		if err := f.removeBundle(); err != nil {
			t.Error(err)
			return // Retain the slot when ownership cleanup could not be verified.
		}
		serviceBundleSlots.Lock()
		serviceBundleSlots.used[slot] = false
		serviceBundleSlots.Unlock()
	})
	f.binDir, f.cleanupBins, err = testbin.MainDir("tests", fmt.Sprintf("service-bundle-%d", slot+1), "nvs-bin-", programs...)
	if err != nil {
		t.Fatalf("allocate service fixture binaries: %v", err)
	}
	// A fixed bundle may contain optional workers from an earlier fixture.
	// Remove those before the broker can discover stale siblings on startup.
	for _, program := range programs {
		path := filepath.Join(f.binDir, testbin.Exe(program))
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			t.Fatalf("remove previous fixture executable %q (a previous process may still be running): %v", path, err)
		}
	}
	f.endpoint = filepath.Join(dir, "s.sock")
	if runtime.GOOS == "windows" {
		f.endpoint = `\\.\pipe\nvpair-test-` + filepath.Base(dir)
	}
	for _, path := range []string{f.configDir, f.clusterDir, filepath.Join(dir, "logs")} {
		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	f.servicePath = f.copyBinary(serviceBin)
	f.brokerPath = f.copyBinary(brokerBin)
	f.scannerPath = f.copyBinary(scannerBin)
	f.inventory = &serviceProcessInventory{paths: []string{f.servicePath, f.brokerPath, f.scannerPath}, known: make(map[int32]serviceProcessIdentity)}
	// One plan for the fixture's lifetime: a restarted broker reuses it.
	f.ports = reserveFixtureServicePorts(t)
	f.env = append(os.Environ(), "HOME="+f.configDir, "XDG_CONFIG_HOME="+f.configDir, "APPDATA="+f.configDir, "LOCALAPPDATA="+f.configDir, servicectl.EndpointEnv+"="+f.endpoint)
	return f
}

// remoteServiceFixture is a serviceFixture whose bundle also runs
// engine-manager and cluster-manager, with an empty-directory scanner, inert
// engines, a pinned target identity and two pinned peers: one denied and one
// intended to be allowed after a target-local grant. Start with f.start().
type remoteServiceFixture struct {
	*serviceFixture
	enginePath, clusterPath string
	peer, allowed           *http.Client
	permissionPath          string
}

func newRemoteServiceFixture(t *testing.T) *remoteServiceFixture {
	t.Helper()
	f := newServiceFixture(t)
	// Reuse the registered bundle paths. Its scanner is the existing empty
	// directory fixture, so this service cannot discover installed peers.
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	jointCopyExecutable(t, self, f.scannerPath)
	enginePath := f.copyBinary(engineMgrBin)
	clusterPath := f.copyBinary(clusterMgrBin)
	f.inventory.paths = append(f.inventory.paths, enginePath, clusterPath)
	scannerPort := temporaryClusterPort(t)
	f.env = append(f.env, "NVPAIR_TEST_SUBSCRIPTION_SCANNER="+f.scannerPath,
		"NVPAIR_TEST_SUBSCRIPTION_PORT="+strconv.Itoa(scannerPort), "OLLAMA_HOST=", "USERPROFILE="+f.configDir)
	inert := newFixtureHTTPServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }))
	t.Cleanup(inert.Close)
	jointIsolateEngines(t, f.configDir, inert.Listener.Addr().(*net.TCPAddr).Port)
	peerDir := t.TempDir()
	peerCert := mintClusterIdentity(t, peerDir, "management-service-peer")
	allowedDir := t.TempDir()
	allowedCert := mintClusterIdentity(t, allowedDir, "management-service-allowed")
	targetCert := mintClusterIdentity(t, f.clusterDir, "management-service-target")
	for _, member := range []struct{ dir, id, peer, cert string }{
		{f.clusterDir, "management-service-target", "management-service-peer", peerCert},
		{peerDir, "management-service-peer", "management-service-target", targetCert},
		{allowedDir, "management-service-allowed", "management-service-target", targetCert},
	} {
		writeActiveAdmission(t, member.dir)
		writePin(t, member.dir, member.peer, member.cert)
		writeMembers(t, member.dir, member.id, member.peer)
		jointWriteJSON(t, filepath.Join(member.dir, "identity.json"), map[string]any{"node_uuid": member.id})
	}
	writePin(t, f.clusterDir, "management-service-allowed", allowedCert)
	writeMembers(t, f.clusterDir, "management-service-target", "management-service-peer", "management-service-allowed")
	policy := nodepolicy.Default()
	policy.Revision = "12"
	jointWriteJSON(t, f.policyPath(), policy)
	cfg, ok := clustertrust.Open(peerDir).ClientTLSConfig("management-service-target")
	if !ok {
		t.Fatal("service fixture has no peer TLS authority")
	}
	client := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: cfg}}
	t.Cleanup(client.CloseIdleConnections)
	allowedCFG, ok := clustertrust.Open(allowedDir).ClientTLSConfig("management-service-target")
	if !ok {
		t.Fatal("service fixture has no allowed peer TLS authority")
	}
	allowedClient := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{TLSClientConfig: allowedCFG}}
	t.Cleanup(allowedClient.CloseIdleConnections)
	return &remoteServiceFixture{
		serviceFixture: f,
		enginePath:     enginePath,
		clusterPath:    clusterPath,
		peer:           client,
		allowed:        allowedClient,
		permissionPath: filepath.Join(f.clusterDir, clustertrust.RemoteManagementFile),
	}
}

func (f *serviceFixture) copyBinary(src string) string {
	f.t.Helper()
	data, err := os.ReadFile(src)
	if err != nil {
		f.t.Fatal(err)
	}
	info, err := os.Stat(src)
	if err != nil {
		f.t.Fatal(err)
	}
	path := filepath.Join(f.binDir, filepath.Base(src))
	if err := os.WriteFile(path, data, info.Mode().Perm()); err != nil {
		f.t.Fatalf("copy fixture executable to %q (a previous process may still be running): %v", path, err)
	}
	path, err = canonicalServicePath(path)
	if err != nil {
		f.t.Fatal(err)
	}
	return path
}

func (f *serviceFixture) args() []string {
	args := []string{"--broker-path", f.brokerPath, "--endpoint", f.endpoint, "--log-dir", filepath.Join(f.dir, "logs"), "--", "--scanner-path", f.scannerPath, "--cluster-dir", f.clusterDir}
	args = append(args, f.ports.args()...)
	if f.proxyPath != "" {
		return append(args, "--proxy-path", f.proxyPath, "--proxy-engines", "llamacpp")
	}
	return append(args, "--proxy-engines=")
}

// newProxyServiceFixture is newServiceFixture with the broker fronting llama.cpp
// through a copied nvpair-proxy, so policy activation is acknowledged by a real
// proxy. Its facade restores an OS-assigned port from the fixture's app directory.
func newProxyServiceFixture(t *testing.T) *serviceFixture {
	t.Helper()
	f := newServiceFixture(t)
	f.proxyPath = f.copyBinary(proxyBin)
	f.inventory.paths = append(f.inventory.paths, f.proxyPath)
	f.facadePort = freePort(t)
	portPath := filepath.Join(filepath.Dir(f.policyPath()), "llamacpp-proxy-port.json")
	data, err := json.Marshal(struct {
		Port int `json:"port"`
	}{Port: f.facadePort})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(portPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(portPath, data, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("fixture_facade_config engine=llamacpp port=%d path=%q", f.facadePort, portPath)
	return f
}

func (f *serviceFixture) startChild(args ...string) *serviceChild {
	f.t.Helper()
	child := &serviceChild{cmd: exec.Command(f.servicePath, args...), done: make(chan struct{}), output: &serviceOutput{}}
	child.cmd.Dir = f.dir
	child.cmd.Env = f.env
	child.cmd.Stdout = child.output
	child.cmd.Stderr = child.output
	child.cmd.WaitDelay = 2 * time.Second
	if err := child.cmd.Start(); err != nil {
		f.t.Fatalf("start fixture child: %v", err)
	}
	f.children = append(f.children, child)
	f.cleanupVerified = false
	f.t.Cleanup(f.cleanup)
	go func() { child.err = child.cmd.Wait(); close(child.done) }()
	f.t.Logf("fixture_started direct_pid=%d endpoint=%q inventory=%+v", child.cmd.Process.Pid, f.endpoint, f.scan())
	return child
}

func (f *serviceFixture) start() { f.t.Helper(); f.service = f.startChild(f.args()...) }

func (f *serviceFixture) scan() []serviceProcessIdentity {
	f.t.Helper()
	owned, err := f.inventory.scan()
	if err != nil {
		f.t.Fatalf("fixture inventory: %v", err)
	}
	return owned
}

func (f *serviceFixture) dial() *serviceRPCClient {
	f.t.Helper()
	deadline := time.Now().Add(serviceCallTimeout)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		conn, err := servicectl.DialEndpoint(ctx, f.endpoint)
		cancel()
		if err == nil {
			c := newServiceRPCClient(conn)
			c.fixture = f
			f.clients = append(f.clients, c)
			return c
		}
		f.scan()
		if time.Now().After(deadline) {
			f.t.Fatalf("fixture endpoint did not listen: %v; output: %s", err, f.service.output.String())
		}
		time.Sleep(25 * time.Millisecond)
	}
}

func (f *serviceFixture) readyClient() *serviceRPCClient {
	f.t.Helper()
	c := f.dial()
	c.notification(f.t, "app:ready")
	owned := f.scan()
	for _, path := range f.inventory.paths {
		found := false
		for _, p := range owned {
			if p.Path == path {
				found = true
			}
		}
		if !found {
			f.t.Fatalf("ready fixture missing running executable %q: %+v", path, owned)
		}
	}
	servicePolicy(f.t, c)
	if f.facadePort != 0 {
		m := c.mustCall(f.t, "102", "llamacpp-proxy:get-status", nil)
		var status struct {
			Ready bool `json:"ready"`
			Port  int  `json:"port"`
		}
		if err := json.Unmarshal(m.Result, &status); err != nil {
			f.t.Fatal(err)
		}
		if !status.Ready || status.Port != f.facadePort {
			f.t.Fatalf("fixture facade ready=%t port=%d, want ready on ephemeral port %d", status.Ready, status.Port, f.facadePort)
		}
		f.t.Logf("fixture_facade_ready engine=llamacpp port=%d", status.Port)
	} else {
		f.t.Log("fixture_facade_ready engines=none")
	}
	f.t.Logf("fixture_ready endpoint=%q processes=%+v", f.endpoint, owned)
	return c
}

func (f *serviceFixture) waitChild(child *serviceChild, timeout time.Duration) {
	f.t.Helper()
	select {
	case <-child.done:
	case <-time.After(timeout):
		f.t.Fatalf("child pid %d did not exit in %s", child.cmd.Process.Pid, timeout)
	}
}

func (f *serviceFixture) cli(timeout time.Duration, args ...string) (string, int) {
	f.t.Helper()
	child := f.startChild(args...)
	f.waitChild(child, timeout)
	if child.err != nil {
		var exit *exec.ExitError
		if !errors.As(child.err, &exit) {
			f.t.Fatalf("CLI wait: %v", child.err)
		}
	}
	return child.output.String(), child.cmd.ProcessState.ExitCode()
}

func (f *serviceFixture) assertStopped() {
	f.t.Helper()
	owned := f.scan()
	if len(owned) != 0 {
		f.t.Fatalf("owned running processes remain: %+v", owned)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	conn, err := servicectl.DialEndpoint(ctx, f.endpoint)
	if err == nil {
		_ = conn.Close()
		f.t.Fatal("fixture endpoint still accepts connections")
	}
	f.t.Logf("fixture_cleanup endpoint=%q owned_running=0 endpoint_rejected=true", f.endpoint)
}

// Cleanup also works before app:ready or after the service has been killed.
// It never uses the service's two status PIDs as the ownership oracle.
func (f *serviceFixture) cleanup() {
	if err := f.cleanupChecked(); err != nil {
		f.t.Error(err)
	}
}

func (f *serviceFixture) removeBundle() error {
	if len(f.children) > 0 && !f.cleanupVerified {
		return fmt.Errorf("retaining fixture bundle after incomplete cleanup: %s", f.dir)
	}
	if err := os.RemoveAll(f.dir); err != nil {
		return err
	}
	if f.cleanupBins != nil {
		f.cleanupBins()
	}
	return nil
}

func (f *serviceFixture) cleanupChecked() error {
	if len(f.children) == 0 || f.cleanupVerified {
		return nil
	}
	var failures []error
	gracefulDeadline := time.Now().Add(3 * time.Second)
	for {
		ctx, cancel := context.WithDeadline(context.Background(), gracefulDeadline)
		conn, err := servicectl.DialEndpoint(ctx, f.endpoint)
		cancel()
		if err == nil {
			c := newServiceRPCClient(conn)
			m, callErr := c.call("900000", "service/stop", nil, serviceStopTimeout)
			err = callErr
			if err == nil && m.Error != nil {
				err = fmt.Errorf("service/stop RPC error: %+v", m.Error)
			}
			if closeErr := c.close(); closeErr != nil {
				failures = append(failures, closeErr)
			}
			if err != nil {
				f.t.Logf("graceful fixture stop unavailable: %v", err)
			}
			break
		}
		if time.Now().After(gracefulDeadline) {
			break
		}
		directRunning := false
		for _, child := range f.children {
			select {
			case <-child.done:
			default:
				directRunning = true
			}
		}
		if !directRunning {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	deadline := time.Now().Add(10 * time.Second)
	cleanupCtx, cleanupCancel := context.WithDeadline(context.Background(), deadline)
	defer cleanupCancel()
	for {
		owned, err := f.inventory.scanContext(cleanupCtx)
		if err != nil {
			failures = append(failures, fmt.Errorf("cleanup inventory: %w", err))
		}
		owned, recoveryErr := f.inventory.recoveryContext(cleanupCtx, owned)
		if recoveryErr != nil {
			failures = append(failures, fmt.Errorf("cleanup recorded identities: %w", recoveryErr))
		}
		// Unrecorded direct children remain ambiguous and prevent verification.
		f.inventory.mu.Lock()
		for _, child := range f.children {
			select {
			case <-child.done:
				continue
			default:
			}
			pid := int32(child.cmd.Process.Pid)
			identity, known := f.inventory.known[pid]
			if !known || identity.Path != f.servicePath {
				failures = append(failures, fmt.Errorf("direct child pid %d lacks a recorded copied-service identity", pid))
				continue
			}
		}
		f.inventory.mu.Unlock()
		if cleanupCtx.Err() != nil {
			failures = append(failures, cleanupCtx.Err())
			break
		}
		if len(owned) == 0 {
			break
		}
		// Killing the owner first prevents it from restarting the descendants.
		for _, path := range f.inventory.paths {
			for _, identity := range owned {
				if cleanupCtx.Err() != nil {
					failures = append(failures, cleanupCtx.Err())
					break
				}
				if identity.Path == path {
					if err := f.inventory.killContext(cleanupCtx, identity); err != nil {
						failures = append(failures, fmt.Errorf("kill owned fixture %+v: %w", identity, err))
					}
				}
			}
		}
		if time.Now().After(deadline) {
			failures = append(failures, fmt.Errorf("cleanup deadline: owned processes remain %+v", owned))
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	for _, c := range f.clients {
		if err := c.close(); err != nil {
			failures = append(failures, err)
		}
	}
	for _, child := range f.children {
		select {
		case <-child.done:
		case <-time.After(3 * time.Second):
			failures = append(failures, fmt.Errorf("direct child pid %d was not reaped", child.cmd.Process.Pid))
		}
	}
	// A second inventory catches late children before executable deletion.
	owned, err := f.inventory.scan()
	if err != nil {
		failures = append(failures, fmt.Errorf("final cleanup inventory: %w", err))
	} else if len(owned) != 0 {
		failures = append(failures, fmt.Errorf("final cleanup still owns %+v", owned))
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	conn, err := servicectl.DialEndpoint(ctx, f.endpoint)
	cancel()
	if err == nil {
		_ = conn.Close()
		failures = append(failures, errors.New("cleanup endpoint still accepts connections"))
	}
	f.cleanupVerified = len(failures) == 0
	if f.cleanupVerified {
		f.t.Logf("fixture_cleanup endpoint=%q owned_running=0 endpoint_rejected=true", f.endpoint)
	}
	return errors.Join(failures...)
}

type serviceRPCClient struct {
	*serviceRPCReader
	fixture *serviceFixture
	wmu     sync.Mutex
}

func newServiceRPCClient(conn io.ReadWriteCloser) *serviceRPCClient {
	return &serviceRPCClient{serviceRPCReader: newServiceRPCReader(conn)}
}

func (c *serviceRPCClient) next(timeout time.Duration, pred func(*jsonrpc.Message) bool) (*jsonrpc.Message, error) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		c.mu.Lock()
		for i, m := range c.frames {
			if pred(m) {
				c.frames = append(c.frames[:i], c.frames[i+1:]...)
				c.bytes -= serviceFrameBytes(m)
				c.mu.Unlock()
				return m, nil
			}
		}
		err := c.err
		c.mu.Unlock()
		if err != nil {
			return nil, err
		}
		select {
		case <-c.wake:
		case <-timer.C:
			return nil, fmt.Errorf("frame wait exceeded %s", timeout)
		}
	}
}

func (c *serviceRPCClient) request(id, method string, params json.RawMessage) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	rawID := json.RawMessage(id)
	data, err := json.Marshal(&jsonrpc.Message{JSONRPC: "2.0", ID: &rawID, Method: method, Params: params})
	if err != nil {
		return err
	}
	done := make(chan error, 1)
	go func() { _, err := c.conn.Write(append(data, '\n')); done <- err }()
	select {
	case err := <-done:
		return err
	case <-time.After(serviceCallTimeout):
		c.closeConnection()
		return fmt.Errorf("request write exceeded %s", serviceCallTimeout)
	}
}

func (c *serviceRPCClient) response(id string, timeout time.Duration) (*jsonrpc.Message, error) {
	return c.next(timeout, func(m *jsonrpc.Message) bool { return m.IsResponse() && string(*m.ID) == id })
}
func (c *serviceRPCClient) call(id, method string, params json.RawMessage, timeout time.Duration) (*jsonrpc.Message, error) {
	if err := c.request(id, method, params); err != nil {
		return nil, err
	}
	return c.response(id, timeout)
}
func (c *serviceRPCClient) mustCall(t *testing.T, id, method string, params json.RawMessage) *jsonrpc.Message {
	t.Helper()
	timeout := serviceCallTimeout
	if method == "service/stop" {
		timeout = serviceStopTimeout
	}
	m, err := c.call(id, method, params, timeout)
	if err != nil {
		t.Fatalf("%s: %v", method, err)
	}
	if m.Error != nil {
		t.Fatalf("%s RPC error: %+v", method, m.Error)
	}
	return m
}
func (c *serviceRPCClient) notification(t *testing.T, method string) *jsonrpc.Message {
	t.Helper()
	m, err := c.next(serviceCallTimeout, func(m *jsonrpc.Message) bool { return m.IsNotification() && m.Method == method })
	if err != nil {
		t.Fatalf("notification %s: %v", method, err)
	}
	return m
}
func (c *serviceRPCClient) mustClose(t *testing.T) {
	t.Helper()
	if err := c.close(); err != nil {
		t.Error(err)
	}
}

type serviceStatusSnapshot struct {
	PID            int    `json:"pid"`
	BrokerPID      int    `json:"brokerPid"`
	Clients        int    `json:"clients"`
	StartedAt      string `json:"startedAt"`
	Version        string `json:"version"`
	BrokerRestarts int    `json:"brokerRestarts"`
}

func serviceStatus(t *testing.T, c *serviceRPCClient) serviceStatusSnapshot {
	t.Helper()
	m := c.mustCall(t, "100", "service/status", nil)
	var s serviceStatusSnapshot
	if err := json.Unmarshal(m.Result, &s); err != nil {
		t.Fatal(err)
	}
	if s.PID <= 0 || s.BrokerPID <= 0 || s.Clients < 1 || s.Version == "" {
		t.Fatalf("invalid service status: %+v", s)
	}
	if _, err := time.Parse(time.RFC3339, s.StartedAt); err != nil {
		t.Fatalf("service startedAt: %v", err)
	}
	if c.fixture != nil {
		f := c.fixture
		if s.PID != f.service.cmd.Process.Pid {
			t.Fatalf("service/status pid=%d, launched service pid=%d", s.PID, f.service.cmd.Process.Pid)
		}
		owned := f.scan()
		brokerFound := false
		for _, identity := range owned {
			if int(identity.PID) == s.BrokerPID && identity.Path == f.brokerPath {
				brokerFound = true
			}
		}
		if !brokerFound {
			t.Fatalf("service/status broker pid=%d lacks a running copied-broker identity: %+v", s.BrokerPID, owned)
		}
	}
	t.Logf("service_status pid=%d broker_pid=%d clients=%d restarts=%d", s.PID, s.BrokerPID, s.Clients, s.BrokerRestarts)
	return s
}
func servicePolicy(t *testing.T, c *serviceRPCClient) nodepolicy.Policy {
	t.Helper()
	m := c.mustCall(t, "101", "policy:get", nil)
	return decodeServicePolicy(t, m)
}
func decodeServicePolicy(t *testing.T, m *jsonrpc.Message) nodepolicy.Policy {
	t.Helper()
	return decodeServicePolicySnapshot(t, m).Policy
}

type servicePolicySnapshot struct {
	Policy       nodepolicy.Policy       `json:"policy"`
	Availability nodepolicy.Availability `json:"availability"`
	// Activation and ActiveRevision say whether the proxy enforces Policy.Revision.
	Activation     string `json:"activation"`
	ActiveRevision string `json:"activeRevision"`
}

func decodeServicePolicySnapshot(t *testing.T, m *jsonrpc.Message) servicePolicySnapshot {
	t.Helper()
	if m.Error != nil {
		t.Fatalf("policy:get: %+v", m.Error)
	}
	var result servicePolicySnapshot
	if err := json.Unmarshal(m.Result, &result); err != nil {
		t.Fatal(err)
	}
	if err := result.Policy.Validate(); err != nil {
		t.Fatalf("policy:get invalid policy: %v", err)
	}
	if result.Policy.Version != 2 {
		t.Fatalf("unexpected policy version: %d", result.Policy.Version)
	}
	switch result.Availability {
	case nodepolicy.Available, nodepolicy.Draining, nodepolicy.Paused:
	default:
		t.Fatalf("invalid live availability: %q", result.Availability)
	}
	return result
}

// Resolve with the child's exact config environment, then restore the parent
// before accessing files. These fixtures and their callers remain serial.
func (f *serviceFixture) policyPath() string {
	f.t.Helper()
	for _, key := range []string{"HOME", "XDG_CONFIG_HOME", "APPDATA", "LOCALAPPDATA"} {
		original, existed := os.LookupEnv(key)
		defer func() {
			var err error
			if existed {
				err = os.Setenv(key, original)
			} else {
				err = os.Unsetenv(key)
			}
			if err != nil {
				f.t.Errorf("restore config environment %s: %v", key, err)
			}
		}()
		var value string
		found := false
		for _, entry := range f.env {
			if v, ok := strings.CutPrefix(entry, key+"="); ok {
				value, found = v, true
			}
		}
		if !found {
			f.t.Fatalf("child config environment missing %s", key)
		}
		if err := os.Setenv(key, value); err != nil {
			f.t.Fatalf("resolve child config environment %s: %v", key, err)
		}
	}
	path, err := appdir.Path(nodepolicy.FileName)
	if err != nil {
		f.t.Fatalf("resolve fixture policy path: %v", err)
	}
	rel, err := filepath.Rel(f.configDir, path)
	if err != nil || filepath.IsAbs(rel) || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		f.t.Fatalf("policy path %q escapes disposable config %q: %v", path, f.configDir, err)
	}
	f.t.Logf("fixture_policy_path path=%q disposable_config=%q", path, f.configDir)
	return path
}
