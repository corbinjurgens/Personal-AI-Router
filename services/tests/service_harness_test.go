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
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/servicectl"
)

const serviceCallTimeout = 10 * time.Second

// The service allows 2s for shutdown's reply and 18s for broker teardown.
const serviceStopTimeout = 35 * time.Second

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
	servicePath, brokerPath, scannerPath string
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
	t.Cleanup(func() {
		if err := f.removeBundle(); err != nil {
			t.Error(err)
		}
	})
	f.endpoint = filepath.Join(dir, "s.sock")
	if runtime.GOOS == "windows" {
		f.endpoint = `\\.\pipe\nvpair-test-` + filepath.Base(dir)
	}
	for _, path := range []string{f.configDir, f.clusterDir, filepath.Join(dir, "logs")} {
		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	copyBinary := func(src string) string {
		t.Helper()
		data, err := os.ReadFile(src)
		if err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(src)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, filepath.Base(src))
		if err := os.WriteFile(path, data, info.Mode().Perm()); err != nil {
			t.Fatal(err)
		}
		path, err = canonicalServicePath(path)
		if err != nil {
			t.Fatal(err)
		}
		return path
	}
	f.servicePath = copyBinary(serviceBin)
	f.brokerPath = copyBinary(brokerBin)
	f.scannerPath = copyBinary(scannerBin)
	f.inventory = &serviceProcessInventory{paths: []string{f.servicePath, f.brokerPath, f.scannerPath}, known: make(map[int32]serviceProcessIdentity)}
	f.env = append(os.Environ(), "HOME="+f.configDir, "XDG_CONFIG_HOME="+f.configDir, "APPDATA="+f.configDir, "LOCALAPPDATA="+f.configDir, servicectl.EndpointEnv+"="+f.endpoint)
	return f
}

func (f *serviceFixture) args() []string {
	return []string{"--broker-path", f.brokerPath, "--endpoint", f.endpoint, "--log-dir", filepath.Join(f.dir, "logs"), "--", "--scanner-path", f.scannerPath, "--cluster-dir", f.clusterDir, "--proxy-engines="}
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
	return os.RemoveAll(f.dir)
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
	if m.Error != nil {
		t.Fatalf("policy:get: %+v", m.Error)
	}
	var result struct {
		Policy       nodepolicy.Policy       `json:"policy"`
		Availability nodepolicy.Availability `json:"availability"`
	}
	if err := json.Unmarshal(m.Result, &result); err != nil {
		t.Fatal(err)
	}
	if err := result.Policy.Validate(); err != nil {
		t.Fatalf("policy:get invalid policy: %v", err)
	}
	if result.Policy.Version != 1 {
		t.Fatalf("unexpected policy version: %d", result.Policy.Version)
	}
	switch result.Availability {
	case nodepolicy.Available, nodepolicy.Draining, nodepolicy.Paused:
	default:
		t.Fatalf("invalid live availability: %q", result.Availability)
	}
	return result.Policy
}
