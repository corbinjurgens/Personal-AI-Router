// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"
)

type notificationTransportProbe struct {
	io.ReadWriteCloser
}

func (p notificationTransportProbe) Write(raw []byte) (int, error) {
	fmt.Fprintln(os.Stderr, "notification-helper write-started")
	n, err := p.ReadWriteCloser.Write(raw)
	fmt.Fprintln(os.Stderr, "notification-helper write-finished")
	return n, err
}

// Dispatched before fixture builds by TestMain, using the existing package
// executable path on Windows as well as Unix. No product flag or env knob.
func runNotificationTransportHelper(args []string) int {
	var transport io.ReadWriteCloser
	var err error
	switch args[0] {
	case "stdio":
		transport, err = newStdioTransport()
	case "ipc":
		if len(args) != 2 {
			return 2
		}
		transport, err = dialIPC(args[1])
	default:
		return 2
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	fail, join := newTerminalSession(ctx, cancel, transport)
	defer join()
	codec := newOwnedCodec(ctx, notificationTransportProbe{transport}, fail, 150*time.Millisecond)
	read := make(chan error, 1)
	go func() { _, err := codec.Read(); read <- err }()
	// A finite frame well beyond anonymous pipe and socket buffering. The
	// parent holds the peer open and never drains it during this send.
	err = codec.Notify("proxy/request", strings.Repeat("x", 4<<20))
	if !errors.Is(err, context.DeadlineExceeded) {
		fmt.Fprintf(os.Stderr, "unexpected notify result: %v\n", err)
		return 3
	}
	select {
	case err := <-read:
		if err == nil {
			return 4
		}
		fmt.Fprintln(os.Stderr, "notification-helper read-finished")
	case <-time.After(3 * time.Second):
		fmt.Fprintln(os.Stderr, "read remained blocked after owner close")
		return 4
	}
	return 0
}

// Read stderr independently. cmd.Wait has exactly one owner; even failure
// cleanup joins both the process and this scanner before releasing the peer.
func startNotificationHelper(t *testing.T, mode, path string) (<-chan string, <-chan error, io.WriteCloser, io.ReadCloser) {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"notification-transport-helper", mode}
	if path != "" {
		args = append(args, path)
	}
	cmd := exec.Command(executable, args...)
	input, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	output, err := cmd.StdoutPipe()
	if err != nil {
		input.Close()
		t.Fatal(err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		input.Close()
		output.Close()
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		input.Close()
		output.Close()
		stderr.Close()
		t.Fatal(err)
	}
	events := make(chan string, 32)
	readerDone := make(chan error, 1)
	go func() {
		defer close(events)
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			events <- scanner.Text()
		}
		readerDone <- scanner.Err()
	}()
	processDone := make(chan struct{})
	waited := make(chan error, 1)
	go func() {
		// Stderr EOF precedes Wait, avoiding Wait closing an unread pipe.
		readErr := <-readerDone
		waitErr := cmd.Wait()
		close(processDone)
		waited <- errors.Join(readErr, waitErr)
	}()
	t.Cleanup(func() {
		select {
		case <-processDone:
		default:
			// Emergency cleanup fails the test; never count this as proof of
			// the child's own interruption. Keep both peers open until joined.
			t.Error("notification helper survived its completion assertions")
			if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
				t.Errorf("kill helper: %v", err)
			}
			<-processDone
		}
		input.Close()
		output.Close()
	})
	return events, waited, input, output
}

func assertNotificationHelperFinished(t *testing.T, events <-chan string, waited <-chan error) {
	t.Helper()
	want := []string{"notification-helper write-started", "notification-helper write-finished", "notification-helper read-finished"}
	var got []string
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	for {
		select {
		case event, ok := <-events:
			if !ok {
				if len(got) != len(want) {
					t.Fatalf("helper completions = %v, want %v", got, want)
				}
				for index := range want {
					if got[index] != want[index] {
						t.Fatalf("helper completions = %v, want %v", got, want)
					}
				}
				if err := lifetimeReceive(t, waited); err != nil {
					t.Fatalf("notification helper: %v", err)
				}
				return
			}
			if strings.HasPrefix(event, "notification-helper ") {
				got = append(got, event)
			} else {
				t.Log(event)
			}
		case <-timeout.C:
			t.Fatal("helper failed to finish while peers stayed open and undrained")
		}
	}
}

func lifetimeReceive[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-time.After(5 * time.Second):
		t.Fatal("notification owner did not finish")
		var zero T
		return zero
	}
}

func TestNotificationLifetimeInheritedStdio(t *testing.T) {
	events, waited, _, _ := startNotificationHelper(t, "stdio", "")
	assertNotificationHelperFinished(t, events, waited)
}

func TestNotificationLifetimeIPC(t *testing.T) {
	listener, path := ipcListener(t)
	t.Cleanup(func() { listener.Close() })
	events, waited, _, _ := startNotificationHelper(t, "ipc", path)
	// Accept can only wait until this test's finite deadline, even if the
	// helper fails startup. Closing the listener never closes an accepted peer.
	stop := time.AfterFunc(5*time.Second, func() { listener.Close() })
	conn, err := listener.Accept()
	stop.Stop()
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	assertNotificationHelperFinished(t, events, waited)
}

type notificationCloseProbe struct {
	name  string
	order *[]string
	err   error
}

func (p notificationCloseProbe) Read([]byte) (int, error)      { return 0, io.EOF }
func (p notificationCloseProbe) Write(raw []byte) (int, error) { return len(raw), nil }
func (p notificationCloseProbe) Close() error {
	*p.order = append(*p.order, p.name)
	return p.err
}

func TestNotificationLifetimeTransportClose(t *testing.T) {
	var order []string
	inputError, outputError := errors.New("input close"), errors.New("output close")
	transport := &stdioTransport{
		ReadCloser:  notificationCloseProbe{"input", &order, inputError},
		WriteCloser: notificationCloseProbe{"output", &order, outputError},
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := transport.Close(); !errors.Is(err, inputError) || !errors.Is(err, outputError) {
				t.Errorf("close lost errors: %v", err)
			}
		}()
	}
	wg.Wait()
	if len(order) != 2 || order[0] != "output" || order[1] != "input" {
		t.Fatalf("close order = %v", order)
	}
}
