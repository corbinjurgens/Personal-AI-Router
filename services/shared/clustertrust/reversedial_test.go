// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clustertrust

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

func reverseDialConfigFixture(t *testing.T) string {
	t.Helper()
	base, err := os.MkdirTemp("", "pair-dial-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(base); err != nil {
			t.Errorf("remove config: %v", err)
		}
	})
	if err = os.Mkdir(filepath.Join(base, "cluster"), 0o700); err != nil {
		t.Fatal(err)
	}
	return base
}

func reverseDialFixture(t *testing.T) (string, net.Listener) {
	t.Helper()
	base := reverseDialConfigFixture(t)
	listener, err := ListenReversePrivate(ReverseDialPath(base))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { listener.Close() })
	return base, listener
}

func TestReverseDialPrivateEndpointRecoversStaleSocket(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix socket pathname recovery does not apply to named pipes")
	}
	if path := os.Getenv("PAIR_REVERSE_PRIVATE_CRASH_PATH"); path != "" {
		listener, err := ListenReversePrivate(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := fmt.Fprintln(os.Stdout, "ready"); err != nil {
			t.Fatal(err)
		}
		conn, err := listener.Accept()
		if err == nil {
			err = conn.Close()
		}
		t.Fatalf("crash fixture unexpectedly accepted or stopped: %v", err)
	}
	path := ReverseDialPath(reverseDialConfigFixture(t))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestReverseDialPrivateEndpointRecoversStaleSocket$")
	cmd.Env = append(os.Environ(), "PAIR_REVERSE_PRIVATE_CRASH_PATH="+path)
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	t.Cleanup(func() {
		if !waited {
			if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				t.Errorf("kill crash fixture: %v", err)
			}
			_ = cmd.Wait()
		}
	})
	if ready, err := bufio.NewReader(stdout).ReadString('\n'); err != nil || ready != "ready\n" {
		t.Fatalf("lock-aware listener did not become ready: %q, %v", ready, err)
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	err = cmd.Wait()
	waited = true
	var killed *exec.ExitError
	if !errors.As(err, &killed) {
		t.Fatalf("crash fixture did not exit by force: %v", err)
	}
	if info, err := os.Stat(path); err != nil || info.Mode()&os.ModeSocket == 0 {
		t.Fatalf("stale socket fixture missing: info=%v, err=%v", info, err)
	}
	record, err := os.ReadFile(path + ".lock")
	if err != nil {
		t.Fatal(err)
	}
	var identity struct {
		Version int    `json:"version"`
		Inode   uint64 `json:"inode"`
	}
	if err := json.Unmarshal(record, &identity); err != nil {
		t.Fatalf("crash left invalid ownership record: %v", err)
	}
	if identity.Version != 1 || identity.Inode == 0 {
		t.Fatalf("crash left no socket ownership evidence: %+v", identity)
	}
	listener, err := ListenReversePrivate(path)
	if err != nil {
		t.Fatalf("recover stale socket: %v", err)
	}
	t.Cleanup(func() {
		if err := listener.Close(); err != nil {
			t.Errorf("close recovered listener: %v", err)
		}
	})
	if info, err := os.Stat(path); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("recovered socket is not owner-only: info=%v, err=%v", info, err)
	}
	reverseDialAssertListenerReachable(t, path, listener)
}

func TestReverseDialPrivateEndpointPreservesUnrecordedStaleSocket(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix socket ownership records do not apply to named pipes")
	}
	path := ReverseDialPath(reverseDialConfigFixture(t))
	stale, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	stale.SetUnlinkOnClose(false)
	if err := stale.Close(); err != nil {
		t.Fatal(err)
	}
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	listener, err := ListenReversePrivate(path)
	if err == nil {
		closeErr := listener.Close()
		t.Fatalf("reclaimed socket without ownership evidence; close: %v", closeErr)
	}
	after, err := os.Stat(path)
	if err != nil || !os.SameFile(before, after) {
		t.Fatalf("unrecorded stale socket changed: info=%v, err=%v", after, err)
	}
}

func reverseDialAssertListenerReachable(t *testing.T, path string, listener net.Listener) {
	t.Helper()
	done := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			done <- err
			return
		}
		if err := conn.SetWriteDeadline(time.Now().Add(time.Second)); err != nil {
			done <- errors.Join(err, conn.Close())
			return
		}
		_, err = conn.Write([]byte("owner"))
		done <- errors.Join(err, conn.Close())
	}()
	conn, err := net.DialTimeout("unix", path, time.Second)
	if err != nil {
		t.Fatalf("dial original listener: %v", err)
	}
	defer func() {
		if err := conn.Close(); err != nil {
			t.Errorf("close listener probe: %v", err)
		}
	}()
	if err := conn.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	var reply [5]byte
	if _, err := io.ReadFull(conn, reply[:]); err != nil {
		t.Fatalf("read original listener reply: %v", err)
	}
	if string(reply[:]) != "owner" {
		t.Fatalf("original listener reply %q, want owner", reply)
	}
	if err := <-done; err != nil {
		t.Fatalf("original listener probe: %v", err)
	}
}

func TestReverseDialPrivateEndpointPreservesLiveOwner(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix lifetime locks do not apply to named pipes")
	}
	base, owner := reverseDialFixture(t)
	path := ReverseDialPath(base)
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	competitor, err := ListenReversePrivate(path)
	if err == nil {
		closeErr := competitor.Close()
		t.Fatalf("second live owner acquired endpoint; close: %v", closeErr)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatalf("competitor removed live socket: %v", err)
	}
	if !os.SameFile(before, after) {
		t.Fatal("competitor replaced live socket")
	}
	reverseDialAssertListenerReachable(t, path, owner)
}

func TestReverseDialPrivateEndpointPreservesUnlockedLiveOwner(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix stale socket detection does not apply to named pipes")
	}
	path := ReverseDialPath(reverseDialConfigFixture(t))
	owner, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := owner.Close(); err != nil {
			t.Errorf("close unlocked owner: %v", err)
		}
	})
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	competitor, err := ListenReversePrivate(path)
	if err == nil {
		closeErr := competitor.Close()
		t.Fatalf("replaced answering socket without lifetime lock; close: %v", closeErr)
	}
	after, err := os.Stat(path)
	if err != nil || !os.SameFile(before, after) {
		t.Fatalf("competitor changed answering socket: info=%v, err=%v", after, err)
	}
	reverseDialAssertListenerReachable(t, path, owner)
}

func TestReverseDialPrivateEndpointPreservesSaturatedUnlockedLiveOwner(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix listener backlog saturation does not apply to named pipes")
	}
	path := ReverseDialPath(reverseDialConfigFixture(t))
	owner, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := owner.Close(); err != nil {
			t.Errorf("close saturated owner: %v", err)
		}
	})
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	var queued []net.Conn
	t.Cleanup(func() {
		for _, conn := range queued {
			if err := conn.Close(); err != nil {
				t.Errorf("close queued client: %v", err)
			}
		}
	})
	var refused error
	for len(queued) < 8192 {
		conn, err := net.DialTimeout("unix", path, 20*time.Millisecond)
		if err != nil {
			refused = err
			break
		}
		queued = append(queued, conn)
	}
	if refused == nil || len(queued) == 0 {
		t.Fatalf("could not saturate live socket backlog: queued=%d, err=%v", len(queued), refused)
	}
	var dialErr net.Error
	if !errors.Is(refused, syscall.ECONNREFUSED) && !(errors.As(refused, &dialErr) && dialErr.Timeout()) {
		t.Fatalf("dial failed for a reason other than full backlog: %v", refused)
	}
	t.Logf("saturated backlog after %d queued connections: %v", len(queued), refused)
	competitor, err := ListenReversePrivate(path)
	if err == nil {
		closeErr := competitor.Close()
		t.Fatalf("replaced saturated live owner; close: %v", closeErr)
	}
	after, err := os.Stat(path)
	if err != nil || !os.SameFile(before, after) {
		t.Fatalf("competitor changed saturated socket: info=%v, err=%v", after, err)
	}
	if err := owner.SetDeadline(time.Now().Add(2 * time.Second)); err != nil {
		t.Fatal(err)
	}
	for range queued {
		conn, err := owner.Accept()
		if err != nil {
			t.Fatalf("drain saturated owner: %v", err)
		}
		if err := conn.Close(); err != nil {
			t.Fatal(err)
		}
	}
	reverseDialAssertListenerReachable(t, path, owner)
}

func TestReverseDialPrivateEndpointPreservesMismatchedOwnershipRecord(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix socket ownership records do not apply to named pipes")
	}
	path := ReverseDialPath(reverseDialConfigFixture(t))
	owner, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := owner.Close(); err != nil {
			t.Errorf("close mismatched-record owner: %v", err)
		}
	})
	if err := os.WriteFile(path+".lock", []byte(`{"version":1,"device":0,"inode":0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	competitor, err := ListenReversePrivate(path)
	if err == nil {
		closeErr := competitor.Close()
		t.Fatalf("accepted mismatched ownership record; close: %v", closeErr)
	}
	after, err := os.Stat(path)
	if err != nil || !os.SameFile(before, after) {
		t.Fatalf("mismatched record replaced owner: info=%v, err=%v", after, err)
	}
	reverseDialAssertListenerReachable(t, path, owner)
}

func TestReverseDialPrivateEndpointCloseReleasesOwnership(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix lifetime locks do not apply to named pipes")
	}
	base, first := reverseDialFixture(t)
	path := ReverseDialPath(base)
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if record, err := os.ReadFile(path + ".lock"); err != nil || len(record) != 0 {
		t.Fatalf("graceful close retained stale ownership evidence: %q, %v", record, err)
	}
	replacement, err := ListenReversePrivate(path)
	if err != nil {
		t.Fatalf("ownership not released on close: %v", err)
	}
	t.Cleanup(func() {
		if err := replacement.Close(); err != nil {
			t.Errorf("close replacement listener: %v", err)
		}
	})
	if err := first.Close(); err != nil {
		t.Fatalf("repeated close: %v", err)
	}
	competitor, err := ListenReversePrivate(path)
	if err == nil {
		closeErr := competitor.Close()
		t.Fatalf("repeated close released replacement ownership; close: %v", closeErr)
	}
	reverseDialAssertListenerReachable(t, path, replacement)
}

func TestReverseDialPrivateEndpointPrepareFailureReleasesOwnership(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix pathname preparation does not apply to named pipes")
	}
	path := ReverseDialPath(reverseDialConfigFixture(t))
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	child := filepath.Join(path, "occupied")
	if err := os.WriteFile(child, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	failed, err := ListenReversePrivate(path)
	if err == nil {
		closeErr := failed.Close()
		t.Fatalf("removed nonempty directory; close: %v", closeErr)
	}
	if err := os.Remove(child); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	listener, err := ListenReversePrivate(path)
	if err != nil {
		t.Fatalf("ownership not released after preparation failure: %v", err)
	}
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestReverseDialPrivateEndpointBindFailureReleasesOwnership(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix socket path length does not apply to named pipes")
	}
	path := filepath.Join(reverseDialConfigFixture(t), strings.Repeat("x", 200))
	for attempt := 0; attempt < 2; attempt++ {
		listener, err := ListenReversePrivate(path)
		if err == nil {
			closeErr := listener.Close()
			t.Fatalf("bound overlong Unix socket path; close: %v", closeErr)
		}
		var bindErr *net.OpError
		if !errors.As(err, &bindErr) || bindErr.Op != "listen" {
			t.Fatalf("attempt %d failed before bind, possibly retaining lock: %v", attempt, err)
		}
	}
}

func TestReverseDialPrivateEndpointPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows pipe ACL coverage is in TestReverseDialPrivatePipeCurrentUser")
	}
	base, _ := reverseDialFixture(t)
	info, err := os.Stat(ReverseDialPath(base))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("socket permissions %o, want 600", info.Mode().Perm())
	}
	if ReverseDialPath(base) != ReverseDialPath(filepath.Join(base, ".")) {
		t.Fatal("equivalent config path changed endpoint")
	}
}

func TestReverseDialHandshakeCancellation(t *testing.T) {
	base, listener := reverseDialFixture(t)
	done := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			done <- err
			return
		}
		defer conn.Close()
		_, err = io.Copy(io.Discard, conn)
		done <- err
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := DialReverse(ctx, base, "peer", "ni")
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deadline not preserved: %v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("private handshake exceeded caller deadline")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("canceled handshake left private connection open")
	}
}

// This models a socket timer firing while the matching context timer has not
// published Err yet. The deadline is already in the past, without a sleep.
type reverseDialPendingDeadlineContext struct {
	context.Context
	deadline time.Time
}

func (c reverseDialPendingDeadlineContext) Deadline() (time.Time, bool) { return c.deadline, true }

func TestReverseDialContextErrorPendingDeadlineTimer(t *testing.T) {
	ctx := reverseDialPendingDeadlineContext{Context: context.Background(), deadline: time.Now().Add(-time.Second)}
	if ctx.Err() != nil {
		t.Fatal("fixture context error already published")
	}
	if err := reverseDialContextError(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("elapsed deadline with pending context timer returned %v, want deadline exceeded", err)
	}
}

func TestReverseDialContextErrorBeforeDeadline(t *testing.T) {
	ctx := reverseDialPendingDeadlineContext{Context: context.Background(), deadline: time.Now().Add(time.Hour)}
	if err := reverseDialContextError(ctx); err != nil {
		t.Fatalf("future deadline returned %v", err)
	}
}

func TestReverseDialContextErrorPreservesCancellation(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	cancel()
	ctx := reverseDialPendingDeadlineContext{Context: parent, deadline: time.Now().Add(-time.Second)}
	if err := reverseDialContextError(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("published cancellation returned %v, want canceled", err)
	}
}

func acceptReverseDialWatch(listener net.Listener, callID string) (net.Conn, error) {
	conn, err := listener.Accept()
	if err != nil {
		return nil, err
	}
	header, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		conn.Close()
		return nil, err
	}
	var request ReverseDialRequest
	if err = json.Unmarshal(header, &request); err != nil {
		conn.Close()
		return nil, err
	}
	if !request.Watch || request.CallID != callID || request.PeerUUID != "" || request.Service != "" {
		conn.Close()
		return nil, errors.New("incorrect lifecycle attachment")
	}
	if _, err = conn.Write([]byte{0}); err != nil {
		conn.Close()
		return nil, err
	}
	return conn, nil
}

// refuseReverseDial answers one private dial request with a refusal byte, after
// the lifecycle connection attaches as DialReverse requires.
func refuseReverseDial(listener net.Listener, code byte) <-chan error {
	done := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			done <- err
			return
		}
		defer conn.Close()
		header, err := bufio.NewReader(conn).ReadBytes('\n')
		if err != nil {
			done <- err
			return
		}
		var request ReverseDialRequest
		if err = json.Unmarshal(header, &request); err != nil {
			done <- err
			return
		}
		watch, err := acceptReverseDialWatch(listener, request.CallID)
		if err != nil {
			done <- err
			return
		}
		defer watch.Close()
		_, err = conn.Write([]byte{code})
		done <- err
	}()
	return done
}

func TestReverseDialRefusalNamesItsReason(t *testing.T) {
	test := func(name string, code ReverseRefusal, wantCause error, wantText string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			base, listener := reverseDialFixture(t)
			done := refuseReverseDial(listener, byte(code))
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			_, err := DialReverse(ctx, base, "peer", "ni")
			var unavailable *ReverseUnavailableError
			if !errors.As(err, &unavailable) || unavailable.PeerUUID != "peer" || unavailable.Route != RouteReverse {
				t.Fatalf("refusal = %v, want a reverse-route ReverseUnavailableError for peer", err)
			}
			if wantCause != nil && !errors.Is(err, wantCause) {
				t.Fatalf("refusal = %v, want cause %v", err, wantCause)
			}
			if !strings.Contains(err.Error(), "private service dial refused: "+wantText) {
				t.Fatalf("refusal = %q, want reason %q", err, wantText)
			}
			if err = <-done; err != nil {
				t.Fatal(err)
			}
		})
	}
	test("no reverse session", ReverseRefusedNoSession, ErrReverseNoSession, "no current reverse session")
	test("capacity exhausted", ReverseRefusedCapacity, ErrReverseCapacity, "reverse capacity exhausted")
	test("admission failed", ReverseRefusedAdmission, ErrReverseAdmission, "peer admission failed")
	test("remote refused", ReverseRefusedRemote, ErrReverseRemoteRefused, "reverse service unavailable")
	test("open failed", ReverseRefusedFailed, nil, "reverse stream open failed")
	test("unrecognized code", ReverseRefusal(9), nil, "unrecognized refusal code 9")
}

func TestReverseDialPreservesReplyBufferedBytes(t *testing.T) {
	base, listener := reverseDialFixture(t)
	done := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			done <- err
			return
		}
		defer conn.Close()
		header, err := bufio.NewReader(conn).ReadBytes('\n')
		if err != nil {
			done <- err
			return
		}
		var request ReverseDialRequest
		if err = json.Unmarshal(header, &request); err != nil {
			done <- err
			return
		}
		if request.PeerUUID != "peer" || request.Service != noderec.ServiceNodeInfo || !ValidReverseSessionID(request.CallID) || request.Watch {
			done <- errors.New("incorrect request framing")
			return
		}
		watch, err := acceptReverseDialWatch(listener, request.CallID)
		if err != nil {
			done <- err
			return
		}
		defer watch.Close()
		_, err = conn.Write([]byte{0, 'b', 'o', 'd', 'y'})
		done <- err
	}()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	conn, err := DialReverse(ctx, base, "peer", "ni")
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	body, err := io.ReadAll(conn)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != "body" {
		t.Fatalf("reply bytes %q, want body", body)
	}
	if err = <-done; err != nil {
		t.Fatal(err)
	}
}

func TestReverseDialCloseClosesLifecycle(t *testing.T) {
	base, listener := reverseDialFixture(t)
	done := make(chan error, 1)
	go func() {
		first, err := listener.Accept()
		if err != nil {
			done <- err
			return
		}
		defer first.Close()
		header, err := bufio.NewReader(first).ReadBytes('\n')
		if err != nil {
			done <- err
			return
		}
		var request ReverseDialRequest
		if err = json.Unmarshal(header, &request); err != nil {
			done <- err
			return
		}
		control, err := acceptReverseDialWatch(listener, request.CallID)
		if err != nil {
			done <- err
			return
		}
		defer control.Close()
		if _, err = first.Write([]byte{0}); err != nil {
			done <- err
			return
		}
		var unexpected [1]byte
		if _, err = control.Read(unexpected[:]); err != io.EOF {
			done <- fmt.Errorf("lifecycle close returned %v, want EOF", err)
			return
		}
		done <- nil
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	conn, err := DialReverse(ctx, base, "peer", "ni")
	if err != nil {
		t.Fatal(err)
	}
	if err = conn.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err = <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal("retained lifecycle connection not closed")
	}
}

func TestReverseDialHeaderLimit(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, err := DialReverse(ctx, t.TempDir(), string(make([]byte, ReverseDialMaxHeader)), "ni")
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("oversized target not refused: %v", err)
	}
}
