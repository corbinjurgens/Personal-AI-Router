// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package pipeio

import (
	"errors"
	"io"
	"os"
	"runtime"
	"testing"
	"time"

	"golang.org/x/sys/windows"
)

func TestReaderCancelsNativeRead(t *testing.T) {
	input, output, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	owned, err := NewReader(input)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = output.Close(); _ = owned.Close() })
	r, ok := owned.(*reader)
	if !ok {
		t.Fatal("OS pipe did not get a synchronous read owner")
	}
	readDone := make(chan error, 1)
	go func() {
		var b [1]byte
		n, err := r.Read(b[:])
		if n != 0 {
			readDone <- errors.New("cancelled empty pipe returned bytes")
			return
		}
		readDone <- err
	}()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for {
		r.mu.Lock()
		active := r.thread != 0
		r.mu.Unlock()
		if active {
			break
		}
		select {
		case <-deadline.C:
			t.Fatal("read did not publish its cancellation owner")
		default:
			runtime.Gosched()
		}
	}
	closed := make(chan error, 1)
	go func() { closed <- r.Close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("cancel and close pipe: %v", err)
		}
	case <-deadline.C:
		t.Fatal("cancel did not join the native read")
	}
	select {
	case err := <-readDone:
		if !errors.Is(err, os.ErrClosed) || !errors.Is(err, windows.ERROR_OPERATION_ABORTED) {
			t.Fatalf("native cancelled read = %v, want ERROR_OPERATION_ABORTED and ErrClosed", err)
		}
		t.Log("ReadFile returned ERROR_OPERATION_ABORTED, mapped to ErrClosed")
	case <-deadline.C:
		t.Fatal("native read did not return")
	}
	if err := r.Close(); err != nil {
		t.Fatalf("duplicate close: %v", err)
	}
	var b [1]byte
	if _, err := r.Read(b[:]); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("read after close = %v", err)
	}
}

func TestReaderDataAndEOF(t *testing.T) {
	input, output, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	owned, err := NewReader(input)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = output.Close(); _ = owned.Close() })
	if _, err := output.Write([]byte("pipe data")); err != nil {
		t.Fatal(err)
	}
	if err := output.Close(); err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(owned)
	if err != nil || string(b) != "pipe data" {
		t.Fatalf("read data = %q, %v", b, err)
	}
}

func TestReaderClosedBeforeRead(t *testing.T) {
	input, output, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	owned, err := NewReader(input)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = output.Close(); _ = owned.Close() })
	if err := owned.Close(); err != nil {
		t.Fatal(err)
	}
	var b [1]byte
	if _, err := owned.Read(b[:]); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("read after close = %v", err)
	}
}
