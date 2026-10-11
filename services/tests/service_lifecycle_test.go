// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

func TestServiceLifecycle(t *testing.T) {
	t.Run("independent policy calls sharing request id", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		first := f.readyClient()
		second := f.readyClient()
		if got := serviceStatus(t, first).Clients; got != 2 {
			t.Fatalf("attached clients=%d, want 2", got)
		}
		// Both requests are outstanding before either reader consumes a reply.
		for _, c := range []*serviceRPCClient{first, second} {
			if err := c.request("7", "policy:get", nil); err != nil {
				t.Fatal(err)
			}
		}
		var policies []nodepolicy.Policy
		for _, c := range []*serviceRPCClient{first, second} {
			m, err := c.response("7", serviceCallTimeout)
			if err != nil {
				t.Fatal(err)
			}
			policies = append(policies, decodeServicePolicy(t, m))
		}
		if !reflect.DeepEqual(policies[0], policies[1]) {
			t.Fatal("independent clients received different fresh policies")
		}
	})
	t.Run("one client detaches while the other remains functional", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		first := f.readyClient()
		second := f.readyClient()
		before := serviceStatus(t, second)
		first.mustClose(t)
		after := waitServiceClientCount(t, second, 1)
		assertServiceLifetime(t, before, after)
		servicePolicy(t, second)
	})
	t.Run("all clients detach and reconnect preserves process identities", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		first := f.readyClient()
		second := f.readyClient()
		before := serviceStatus(t, first)
		identities := f.scan()
		first.mustClose(t)
		second.mustClose(t)
		// Observe the lifetime through a fresh connection after both have left.
		third := f.readyClient()
		after := waitServiceClientCount(t, third, 1)
		assertServiceLifetime(t, before, after)
		current := f.scan()
		for _, identity := range identities {
			found := false
			for _, p := range current {
				if p == identity {
					found = true
				}
			}
			if !found {
				t.Fatalf("detach replaced process identity %+v; current=%+v", identity, current)
			}
		}
	})
	t.Run("RPC stop replies and the owned process exits", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		c := f.readyClient()
		assertServiceStoppedReply(t, c.mustCall(t, "8", "service/stop", nil))
		f.waitChild(f.service, serviceStopTimeout)
		if f.service.err != nil {
			t.Fatalf("service stop exit: %v; %s", f.service.err, f.service.output.String())
		}
		f.assertStopped()
	})
	t.Run("second service instance leaves the original running", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		c := f.readyClient()
		before := serviceStatus(t, c)
		second := f.startChild(f.args()...)
		f.waitChild(second, serviceCallTimeout)
		if second.err != nil || second.cmd.ProcessState.ExitCode() != 0 {
			t.Fatalf("second instance exit: %v; %s", second.err, second.output.String())
		}
		assertServiceLifetime(t, before, serviceStatus(t, c))
		servicePolicy(t, c)
		owned := f.scan()
		services := 0
		for _, p := range owned {
			if p.Path == f.servicePath {
				services++
			}
		}
		if services != 1 {
			t.Fatalf("running copied services=%d, want 1; identities=%+v", services, owned)
		}
	})
	t.Run("CLI status is structured and stopped status exits three", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		c := f.readyClient()
		before := serviceStatus(t, c)
		output, code := f.cli(serviceCallTimeout, "status")
		if code != 0 {
			t.Fatalf("status exit=%d output=%s", code, output)
		}
		var status serviceStatusSnapshot
		if err := json.Unmarshal([]byte(output), &status); err != nil {
			t.Fatalf("decode CLI status: %v", err)
		}
		assertServiceLifetime(t, before, status)
		if status.Clients != 2 {
			t.Fatalf("CLI status clients=%d, want caller plus attached observer", status.Clients)
		}
		assertServiceStoppedReply(t, c.mustCall(t, "8", "service/stop", nil))
		f.waitChild(f.service, serviceStopTimeout)
		output, code = f.cli(serviceCallTimeout, "status")
		if code != 3 {
			t.Fatalf("stopped status exit=%d, want 3; output=%s", code, output)
		}
		f.assertStopped()
	})
	t.Run("CLI stop allows the full broker shutdown grace", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		f.readyClient()
		output, code := f.cli(serviceStopTimeout, "stop")
		if code != 0 {
			t.Fatalf("CLI stop exit=%d output=%s", code, output)
		}
		f.waitChild(f.service, serviceStopTimeout)
		if f.service.err != nil {
			t.Fatalf("CLI stopped service exit: %v", f.service.err)
		}
		f.assertStopped()
	})
	t.Run("cleanup before awaiting readiness", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		f.cleanup()
		f.assertStopped()
	})
	t.Run("cleanup stops owned descendants despite an inventory error", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		f.readyClient()
		owned := f.scan()
		for _, path := range []string{f.brokerPath, f.scannerPath} {
			found := false
			for _, identity := range owned {
				if identity.Path == path {
					found = true
				}
			}
			if !found {
				t.Fatalf("fault injection lacks running owned descendant %q: %+v", path, owned)
			}
		}
		injected := errors.New("fixture injected inventory error")
		f.inventory.mu.Lock()
		f.inventory.scanError = injected
		f.inventory.mu.Unlock()
		// Force identity-based cleanup rather than letting graceful RPC stop
		// remove the descendants before the injected scan failure is observed.
		endpoint := f.endpoint
		f.endpoint += "-unavailable"
		defer func() {
			f.endpoint = endpoint
			f.inventory.mu.Lock()
			f.inventory.scanError = nil
			f.inventory.mu.Unlock()
			f.cleanup()
		}()
		if err := f.cleanupChecked(); !errors.Is(err, injected) {
			t.Fatalf("cleanup error=%v, want injected inventory error", err)
		}
		if f.cleanupVerified {
			t.Fatal("cleanup verified despite the injected inventory error")
		}
		if err := f.removeBundle(); err == nil {
			t.Fatal("bundle removal accepted failed inventory")
		}
		for _, path := range []string{f.dir, f.servicePath, f.brokerPath, f.scannerPath} {
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("failed inventory did not retain %q: %v", path, err)
			}
		}
		remaining, err := f.inventory.scan()
		if !errors.Is(err, injected) || len(remaining) != 0 {
			t.Fatalf("inventory error left owned processes: identities=%+v error=%v", remaining, err)
		}
		f.inventory.mu.Lock()
		f.inventory.scanError = nil
		f.inventory.mu.Unlock()
		f.endpoint = endpoint
		f.assertStopped()
		t.Log("cleanup_inventory_error_visible=true cleanup_verified=false bundle_retained=true owned_running=0 endpoint_rejected=true")
	})
	t.Run("cleanup after forced owned service termination", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		f.readyClient()
		owned := f.scan()
		var killed bool
		for _, identity := range owned {
			if int(identity.PID) == f.service.cmd.Process.Pid && identity.Path == f.servicePath {
				if err := f.inventory.kill(identity); err != nil {
					t.Fatal(err)
				}
				killed = true
			}
		}
		if !killed {
			t.Fatal("owned service identity unavailable for forced termination")
		}
		f.waitChild(f.service, serviceCallTimeout)
		if f.service.err == nil {
			t.Fatal("forced termination unexpectedly exited cleanly")
		}
		f.cleanup()
		f.assertStopped()
	})
	t.Run("cleanup recovers recorded descendants after empty enumeration fails", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		f.readyClient()
		recorded := f.scan()
		var owner serviceProcessIdentity
		for _, path := range f.inventory.paths {
			found := false
			for _, identity := range recorded {
				if identity.Path == path {
					found = true
					if path == f.servicePath && int(identity.PID) == f.service.cmd.Process.Pid {
						owner = identity
					}
				}
			}
			if !found {
				t.Fatalf("empty-enumeration fault lacks recorded copied identity %q: %+v", path, recorded)
			}
		}
		if owner.PID == 0 {
			t.Fatal("owned service identity unavailable for forced termination")
		}
		// Inject while the owner and descendants are still live. Killing the
		// owner first lets its broker exit on EOF before recovery is exercised.
		remaining := f.scan()
		for _, identity := range recorded {
			found := false
			for _, current := range remaining {
				if current == identity {
					found = true
				}
			}
			if !found {
				t.Fatalf("fault injection needs a still-live recorded identity %+v: %+v", identity, remaining)
			}
		}
		injected := errors.New("fixture injected empty enumeration error")
		endpoint := f.endpoint
		f.endpoint += "-unavailable"
		f.inventory.mu.Lock()
		f.inventory.enumerationError = injected
		f.inventory.mu.Unlock()
		defer func() {
			f.endpoint = endpoint
			f.inventory.mu.Lock()
			f.inventory.enumerationError = nil
			f.inventory.mu.Unlock()
			f.cleanup()
		}()
		if owned, err := f.inventory.scan(); len(owned) != 0 || !errors.Is(err, injected) {
			t.Fatalf("enumeration fault returned identities=%+v error=%v", owned, err)
		}
		if err := f.cleanupChecked(); !errors.Is(err, injected) {
			t.Fatalf("cleanup error=%v, want injected enumeration error", err)
		}
		f.waitChild(f.service, serviceCallTimeout)
		if f.service.err == nil {
			t.Fatal("forced termination unexpectedly exited cleanly")
		}
		if f.cleanupVerified {
			t.Fatal("cleanup verified despite failed enumeration")
		}
		if err := f.removeBundle(); err == nil {
			t.Fatal("bundle removal accepted failed enumeration")
		}
		for _, path := range []string{f.dir, f.servicePath, f.brokerPath, f.scannerPath} {
			if _, err := os.Stat(path); err != nil {
				t.Fatalf("failed enumeration did not retain %q: %v", path, err)
			}
		}
		// Restore actual enumeration before proving that descendants exited.
		f.inventory.mu.Lock()
		f.inventory.enumerationError = nil
		f.inventory.mu.Unlock()
		f.endpoint = endpoint
		f.assertStopped()
		t.Log("cleanup_empty_enumeration_error_visible=true cleanup_verified=false bundle_retained=true recorded_descendants_stopped=true owned_running=0 endpoint_rejected=true")
	})
	t.Run("cleanup retains the bundle after a malformed RPC frame", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		f.readyClient()
		originalClients := len(f.clients)
		r, w := io.Pipe()
		c := newServiceRPCClient(&serviceNotificationPipe{PipeReader: r})
		f.clients = append(f.clients, c)
		defer func() {
			if err := w.Close(); err != nil {
				t.Errorf("release malformed-frame writer: %v", err)
			}
			if err := r.Close(); err != nil {
				t.Errorf("release malformed-frame reader: %v", err)
			}
			if !waitServiceReader(t, c) {
				return
			}
			var decodeErr *jsonrpc.DecodeError
			if err := c.close(); !errors.As(err, &decodeErr) {
				t.Errorf("repeated close erased malformed-frame error: %v", err)
			}
			// Remove only the joined injected client; its error remains sticky.
			f.clients = f.clients[:originalClients]
			f.cleanup()
		}()
		if _, err := io.WriteString(w, "{\n"); err != nil {
			t.Fatal(err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		if !waitServiceReader(t, c) {
			t.FailNow()
		}
		var decodeErr *jsonrpc.DecodeError
		if err := c.close(); !errors.As(err, &decodeErr) {
			t.Fatalf("close erased malformed-frame error: %v", err)
		}
		assertServiceCleanupRetainsBundle(t, f, "invalid JSON-RPC message")
	})
	t.Run("cleanup retains the bundle when a reader cannot close", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		f.readyClient()
		originalClients := len(f.clients)
		blocked := &serviceBlockedReader{release: make(chan struct{})}
		c := newServiceRPCClient(blocked)
		f.clients = append(f.clients, c)
		defer func() {
			close(blocked.release)
			if !waitServiceReader(t, c) {
				return
			}
			if err := c.close(); err == nil || !strings.Contains(err.Error(), "fixture client reader did not exit") {
				t.Errorf("reader release erased its earlier cleanup failure: %v", err)
			}
			// Remove only the injected connection after its reader is joined.
			f.clients = f.clients[:originalClients]
			f.cleanup()
		}()

		assertServiceCleanupRetainsBundle(t, f, "fixture client reader did not exit")
		select {
		case <-c.done:
			t.Fatal("blocked reader unexpectedly finished before release")
		default:
		}
	})
	t.Run("cleanup retains the bundle after a notification budget error", func(t *testing.T) {
		f := newServiceFixture(t)
		f.start()
		f.readyClient()
		originalClients := len(f.clients)
		r, w := io.Pipe()
		c := newServiceRPCClient(&serviceNotificationPipe{PipeReader: r})
		f.clients = append(f.clients, c)
		written := make(chan error, 1)
		go func() {
			for i := 0; i < 4100; i++ {
				if _, err := io.WriteString(w, "{\"jsonrpc\":\"2.0\",\"method\":\"fixture:event\"}\n"); err != nil {
					written <- err
					return
				}
			}
			written <- w.Close()
		}()
		defer func() {
			if err := w.Close(); err != nil {
				t.Errorf("release notification writer: %v", err)
			}
			// Closing the reader also releases a writer on an earlier fatal path.
			if err := r.Close(); err != nil {
				t.Errorf("release notification reader: %v", err)
			}
			if !waitServiceReader(t, c) {
				return
			}
			f.clients = f.clients[:originalClients]
			f.cleanup()
		}()
		select {
		case err := <-written:
			if err != nil {
				t.Fatalf("notification flood: %v", err)
			}
		case <-time.After(serviceCallTimeout):
			t.Fatal("notification drain blocked the writer")
		}
		if !waitServiceReader(t, c) {
			t.FailNow()
		}
		assertServiceCleanupRetainsBundle(t, f, "fixture client retained-frame budget exceeded")
	})
}

type serviceBlockedReader struct{ release chan struct{} }

func (c *serviceBlockedReader) Read([]byte) (int, error)    { <-c.release; return 0, io.EOF }
func (c *serviceBlockedReader) Write(p []byte) (int, error) { return len(p), nil }
func (c *serviceBlockedReader) Close() error                { return nil }

type serviceNotificationPipe struct{ *io.PipeReader }

func (c *serviceNotificationPipe) Write(p []byte) (int, error) { return len(p), nil }

func waitServiceReader(t *testing.T, c *serviceRPCClient) bool {
	t.Helper()
	select {
	case <-c.done:
		return true
	case <-time.After(2 * time.Second):
		t.Error("fault-injected reader was not joined")
		return false
	}
}

func assertServiceCleanupRetainsBundle(t *testing.T, f *serviceFixture, wantError string) {
	t.Helper()
	err := f.cleanupChecked()
	if err == nil || !strings.Contains(err.Error(), wantError) {
		t.Fatalf("cleanup error=%v, want %q", err, wantError)
	}
	if f.cleanupVerified {
		t.Fatal("cleanup verified despite the injected reader failure")
	}
	if err := f.removeBundle(); err == nil {
		t.Fatal("bundle removal accepted failed cleanup")
	}
	for _, path := range []string{f.dir, f.servicePath, f.brokerPath, f.scannerPath} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("failed cleanup did not retain %q: %v", path, err)
		}
	}
	f.assertStopped()
	t.Logf("cleanup_failure_visible=%q cleanup_verified=false bundle_retained=true", err)
}

func assertServiceStoppedReply(t *testing.T, m *jsonrpc.Message) {
	t.Helper()
	var result struct {
		Stopped bool `json:"stopped"`
	}
	if err := json.Unmarshal(m.Result, &result); err != nil {
		t.Fatal(err)
	}
	if !result.Stopped {
		t.Fatalf("service/stop reply=%+v, want stopped=true", result)
	}
}

func assertServiceLifetime(t *testing.T, before, after serviceStatusSnapshot) {
	t.Helper()
	if after.PID != before.PID || after.BrokerPID != before.BrokerPID || after.StartedAt != before.StartedAt || after.BrokerRestarts != before.BrokerRestarts || after.Version != before.Version {
		t.Fatalf("service lifetime changed: before=%+v after=%+v", before, after)
	}
}

func waitServiceClientCount(t *testing.T, c *serviceRPCClient, want int) serviceStatusSnapshot {
	t.Helper()
	deadline := time.Now().Add(serviceCallTimeout)
	for {
		s := serviceStatus(t, c)
		if s.Clients == want {
			return s
		}
		if time.Now().After(deadline) {
			t.Fatalf("client count=%d, want %d", s.Clients, want)
		}
		time.Sleep(25 * time.Millisecond)
	}
}
