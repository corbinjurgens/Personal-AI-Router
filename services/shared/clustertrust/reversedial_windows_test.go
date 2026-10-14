// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package clustertrust

import (
	"context"
	"errors"
	"io"
	"net"
	"path/filepath"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

func reverseDialPipePair(t *testing.T, path string, listener net.Listener) (net.Conn, net.Conn) {
	t.Helper()
	type result struct {
		conn net.Conn
		err  error
	}
	accepted := make(chan result)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	t.Cleanup(cancel)
	go func() {
		conn, err := listener.Accept()
		select {
		case accepted <- result{conn, err}:
		case <-ctx.Done():
			if conn != nil {
				conn.Close()
			}
		}
	}()
	client, err := dialReversePrivate(ctx, path)
	if err != nil {
		t.Fatalf("dial current-user pipe: %v", err)
	}
	t.Cleanup(func() { client.Close() })
	select {
	case peer := <-accepted:
		if peer.err != nil {
			t.Fatalf("accept current-user pipe: %v", peer.err)
		}
		t.Cleanup(func() { peer.conn.Close() })
		for _, conn := range []net.Conn{client, peer.conn} {
			if err := conn.SetDeadline(time.Now().Add(2 * time.Second)); err != nil {
				t.Fatalf("set pipe deadline: %v", err)
			}
		}
		return client, peer.conn
	case <-ctx.Done():
		t.Fatal("current-user pipe did not accept before deadline")
		return nil, nil
	}
}

func TestReverseDialPrivatePipeCurrentUser(t *testing.T) {
	base, listener := reverseDialFixture(t)
	path := ReverseDialPath(base)
	if path != ReverseDialPath(filepath.Join(base, ".")) {
		t.Fatal("equivalent config path changed pipe endpoint")
	}
	_, server := reverseDialPipePair(t, path, listener)
	handle, ok := server.(interface{ Fd() uintptr })
	if !ok {
		t.Fatal("accepted pipe does not expose its security handle")
	}
	sd, err := windows.GetSecurityInfo(windows.Handle(handle.Fd()), windows.SE_KERNEL_OBJECT, windows.DACL_SECURITY_INFORMATION)
	if err != nil {
		t.Fatalf("read live pipe security: %v", err)
	}
	control, _, err := sd.Control()
	if err != nil || control&windows.SE_DACL_PROTECTED == 0 {
		t.Fatalf("pipe DACL is not protected: control=%x err=%v", control, err)
	}
	dacl, defaulted, err := sd.DACL()
	if err != nil || dacl == nil || defaulted {
		t.Fatalf("pipe lacks an explicit DACL: dacl=%v defaulted=%v err=%v", dacl, defaulted, err)
	}
	if dacl.AceCount != 1 {
		t.Fatalf("pipe DACL entries=%d, want one current-user grant", dacl.AceCount)
	}
	var ace *windows.ACCESS_ALLOWED_ACE
	if err := windows.GetAce(dacl, 0, &ace); err != nil {
		t.Fatalf("read pipe access grant: %v", err)
	}
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		t.Fatalf("read current user: %v", err)
	}
	sid := (*windows.SID)(unsafe.Pointer(&ace.SidStart))
	// Named pipes map generic-all to FILE_ALL_ACCESS (standard rights,
	// synchronize and every file-specific right).
	const fileAllAccess windows.ACCESS_MASK = 0x001f01ff
	mask := ace.Mask
	if mask == windows.GENERIC_ALL {
		mask = fileAllAccess
	}
	if ace.Header.AceType != windows.ACCESS_ALLOWED_ACE_TYPE || ace.Header.AceFlags != 0 || mask != fileAllAccess || !sid.Equals(user.User.Sid) {
		t.Fatalf("pipe access grant: type=%d flags=%x mask=%x SID=%s, want current-user full control", ace.Header.AceType, ace.Header.AceFlags, ace.Mask, sid)
	}
}

func TestReverseDialPrivatePipeHalfClose(t *testing.T) {
	base, listener := reverseDialFixture(t)
	client, server := reverseDialPipePair(t, ReverseDialPath(base), listener)
	done := make(chan error, 1)
	go func() {
		body, err := io.ReadAll(server)
		if err != nil || string(body) != "request" {
			done <- errors.New("server did not receive request followed by half-close EOF")
			return
		}
		if _, err := io.WriteString(server, "reply after EOF"); err != nil {
			done <- err
			return
		}
		half, ok := server.(interface{ CloseWrite() error })
		if !ok {
			done <- errors.New("server pipe lacks CloseWrite")
			return
		}
		done <- half.CloseWrite()
	}()
	if _, err := io.WriteString(client, "request"); err != nil {
		t.Fatal(err)
	}
	half, ok := client.(interface{ CloseWrite() error })
	if !ok {
		t.Fatal("client pipe lacks CloseWrite")
	}
	if err := half.CloseWrite(); err != nil {
		t.Fatalf("client half-close: %v", err)
	}
	reply, err := io.ReadAll(client)
	if err != nil || string(reply) != "reply after EOF" {
		t.Fatalf("read after client half-close: reply=%q err=%v", reply, err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("server half-close: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("server half-close did not finish")
	}
}

func TestReverseDialPrivatePipeCompetingListenerAndRecreation(t *testing.T) {
	base, owner := reverseDialFixture(t)
	path := ReverseDialPath(base)
	competitor, err := ListenReversePrivate(path)
	if err == nil {
		competitor.Close()
		t.Fatal("competing listener acquired a live private pipe")
	}
	client, server := reverseDialPipePair(t, path, owner)
	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	if err := server.Close(); err != nil {
		t.Fatal(err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	replacement, err := ListenReversePrivate(path)
	if err != nil {
		t.Fatalf("recreate private pipe after close: %v", err)
	}
	t.Cleanup(func() { replacement.Close() })
	reverseDialPipePair(t, path, replacement)
}

func TestReverseDialPrivatePipeCancellation(t *testing.T) {
	base, _ := reverseDialFixture(t)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	conn, err := DialReverse(ctx, base, "peer", "ni")
	if conn != nil {
		conn.Close()
		t.Fatal("dial against unaccepted pipe returned a connection")
	}
	var unavailable *ReverseUnavailableError
	if !errors.As(err, &unavailable) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("pipe dial lost caller deadline: %v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("pipe dial outlived caller deadline")
	}
}
