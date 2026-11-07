// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package pipeio

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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
	done := make(chan error, 1)
	go func() {
		var b [1]byte
		_, err := owned.Read(b[:])
		done <- err
	}()
	awaitBlockedRead(t, done, "TestReaderCancelsInheritedPipeRead.func")
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

// Observe the read goroutine parked in the runtime poller or a native read,
// rather than a signal immediately before Read or an assumed scheduling delay.
func awaitBlockedRead(t *testing.T, done <-chan error, caller string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	stacks := make([]byte, 1<<20)
	for time.Now().Before(deadline) {
		n := runtime.Stack(stacks, true)
		for _, stack := range strings.Split(string(stacks[:n]), "\n\n") {
			if !strings.Contains(stack, caller) || !strings.Contains(stack, "os.(*File).Read(") {
				continue
			}
			if strings.Contains(stack, "[IO wait]") || strings.Contains(stack, "[syscall]") {
				t.Log("observed pending os.File.Read before triggering completion")
				return
			}
		}
		select {
		case err := <-done:
			t.Fatalf("read completed before blocked-read observation: %v", err)
		default:
			runtime.Gosched()
		}
	}
	t.Fatal("read did not block in os.File.Read")
}

func TestReaderFilesystemFIFOReceivesLastWriterEOF(t *testing.T) {
	test := func(name string, unlink bool) {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "input.fifo")
			if err := unix.Mkfifo(path, 0600); err != nil {
				t.Fatal(err)
			}
			fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NONBLOCK|unix.O_CLOEXEC, 0)
			if err != nil {
				t.Fatal(err)
			}
			output, err := os.OpenFile(path, os.O_WRONLY, 0)
			if err != nil {
				_ = unix.Close(fd)
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = output.Close() })
			if err := unix.SetNonblock(fd, false); err != nil {
				_ = unix.Close(fd)
				t.Fatal(err)
			}
			// Match inherited stdin: wrap only after clearing O_NONBLOCK.
			input := os.NewFile(uintptr(fd), path)
			t.Cleanup(func() { _ = input.Close() })
			if unlink {
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			}
			owned, err := NewReader(input)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = output.Close(); _ = owned.Close() })
			done := make(chan error, 1)
			go func() {
				var b [1]byte
				_, err := owned.Read(b[:])
				done <- err
			}()
			awaitBlockedRead(t, done, "TestReaderFilesystemFIFOReceivesLastWriterEOF.func")
			if err := output.Close(); err != nil {
				t.Fatal(err)
			}
			select {
			case err := <-done:
				if !errors.Is(err, io.EOF) {
					t.Fatalf("last-writer close read = %v, want EOF", err)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("pending FIFO read did not receive last-writer EOF")
			}
		})
	}
	test("linked FIFO", false)
	test("unlinked FIFO", true)
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
