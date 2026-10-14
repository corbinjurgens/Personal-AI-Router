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
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"nvpair-shared/noderec"
)

func reverseDialFixture(t *testing.T) (string, net.Listener) {
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
	listener, err := ListenReversePrivate(ReverseDialPath(base))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { listener.Close() })
	return base, listener
}

func TestReverseDialPrivateEndpointPermissions(t *testing.T) {
	base, _ := reverseDialFixture(t)
	if runtime.GOOS == "windows" {
		return
	} // Windows uses the current-user pipe ACL, without Unix mode bits.
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

func TestReverseDialRefusalIsTyped(t *testing.T) {
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
		watch, err := acceptReverseDialWatch(listener, request.CallID)
		if err != nil {
			done <- err
			return
		}
		defer watch.Close()
		_, err = conn.Write([]byte{1})
		done <- err
	}()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, err := DialReverse(ctx, base, "peer", "ni")
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) || unavailable.PeerUUID != "peer" {
		t.Fatalf("refusal not typed: %v", err)
	}
	if err = <-done; err != nil {
		t.Fatal(err)
	}
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
