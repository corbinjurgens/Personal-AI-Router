// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package pipeio

import (
	"errors"
	"io"
	"os"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func inheritedPipe(t *testing.T) (*os.File, *os.File) {
	t.Helper()
	input, output, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = input.Close(); _ = output.Close() })
	// exec inherits a blocking descriptor, then os.Stdin wraps it with NewFile.
	fd, err := unix.FcntlInt(input.Fd(), unix.F_DUPFD_CLOEXEC, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := unix.SetNonblock(fd, false); err != nil {
		_ = unix.Close(fd)
		t.Fatal(err)
	}
	inherited := os.NewFile(uintptr(fd), "inherited-stdin")
	t.Cleanup(func() { _ = inherited.Close() })
	return inherited, output
}

func TestReaderCancelsInheritedPipeRead(t *testing.T) {
	inherited, _ := inheritedPipe(t)
	owned, err := NewReader(inherited)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = owned.Close() })
	file, ok := owned.(*os.File)
	if !ok {
		t.Fatal("Unix pipe reader must use os.File's poller")
	}
	// A deadline is supported only when the descriptor joined the runtime
	// poller. This also bounds the read if cancellation regresses.
	if err := file.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatalf("inherited pipe is not pollable: %v", err)
	}
	started := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		close(started)
		var b [1]byte
		_, err := owned.Read(b[:])
		done <- err
	}()
	<-started
	if err := owned.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if !errors.Is(err, os.ErrClosed) {
			t.Fatalf("cancelled pipe read = %v, want ErrClosed", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("close did not interrupt inherited pipe read")
	}
}

func TestReaderInheritedPipeDataAndEOF(t *testing.T) {
	inherited, output := inheritedPipe(t)
	owned, err := NewReader(inherited)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = owned.Close() })
	const want = "inherited pipe data"
	if _, err := output.Write([]byte(want)); err != nil {
		t.Fatal(err)
	}
	if err := output.Close(); err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(owned)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("pipe data = %q, want %q", got, want)
	}
}
