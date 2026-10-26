// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package servicectl

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/windows"

	"nvpair-shared/ipc"
)

const windowsFakeServiceEnv = "NVPAIR_SERVICECTL_WINDOWS_FAKE"
const windowsFakePIDEnv = "NVPAIR_SERVICECTL_WINDOWS_FAKE_PID"

func TestMain(m *testing.M) {
	switch os.Getenv(windowsFakeServiceEnv) {
	case "exits without listening":
		data, err := json.Marshal(os.Getpid())
		if err != nil {
			os.Exit(8)
		}
		pidFile := os.Getenv(windowsFakePIDEnv)
		if err := os.WriteFile(pidFile+".tmp", data, 0o600); err != nil {
			os.Exit(9)
		}
		if err := os.Rename(pidFile+".tmp", pidFile); err != nil {
			os.Exit(10)
		}
		os.Exit(0)
	case "accepts":
		os.Exit(runWindowsFakeService())
	}
	os.Exit(m.Run())
}

func runWindowsFakeService() int {
	endpoint, err := Endpoint()
	if err != nil {
		return 2
	}
	listener, err := ipc.ListenPrivate(endpoint)
	if err != nil {
		return 3
	}
	defer listener.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	stop := context.AfterFunc(ctx, func() { _ = listener.Close() })
	defer stop()
	conn, err := listener.Accept()
	if err != nil {
		return 4
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		return 5
	}
	if err := json.NewEncoder(conn).Encode(os.Getpid()); err != nil {
		return 6
	}
	if _, err := io.Copy(io.Discard, conn); err != nil {
		return 7
	}
	return 0
}

func TestConnectOrStartBusyPipePreservesDeadline(t *testing.T) {
	dir, err := os.MkdirTemp(t.TempDir(), "sctl-busy-")
	if err != nil {
		t.Fatal(err)
	}
	endpoint := `\\.\pipe\` + filepath.Base(dir)
	t.Setenv(EndpointEnv, endpoint)
	listener, err := ipc.ListenPrivate(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := listener.Close(); err != nil {
			t.Errorf("close private listener: %v", err)
		}
	})

	// Without Accept, go-winio reserves the name with its disconnected first
	// handle. Verify the actual busy state before exercising the launcher.
	path, err := windows.UTF16PtrFromString(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := windows.CreateFile(path, windows.GENERIC_READ|windows.GENERIC_WRITE, 0, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_OVERLAPPED, 0)
	if err == nil {
		if err := windows.CloseHandle(handle); err != nil {
			t.Errorf("close unexpected pipe connection: %v", err)
		}
		t.Fatal("private pipe connected without Accept; want ERROR_PIPE_BUSY")
	}
	if !errors.Is(err, windows.ERROR_PIPE_BUSY) {
		t.Fatalf("private pipe open = %v, want ERROR_PIPE_BUSY", err)
	}
	t.Logf("private pipe precondition: %v", err)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	conn, err := ConnectOrStart(ctx, filepath.Join(dir, "must-not-start.exe"), nil)
	if conn != nil {
		if err := conn.Close(); err != nil {
			t.Errorf("close unexpected service connection: %v", err)
		}
		t.Fatal("connected to a pipe with no pending Accept")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("ConnectOrStart = %v, want context.DeadlineExceeded without a launch attempt", err)
	}
	if errors.Is(err, ErrNotRunning) {
		t.Errorf("busy pipe reported as absent: %v", err)
	}
	if errors.Is(err, os.ErrNotExist) {
		t.Errorf("busy pipe attempted the nonexistent service executable: %v", err)
	}
}

func TestDialEndpointAbsentPipePreservesCause(t *testing.T) {
	_, endpoint := windowsTestEndpoint(t)
	requireWindowsPipeOpenError(t, endpoint, windows.ERROR_FILE_NOT_FOUND)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	conn, err := DialEndpoint(ctx, endpoint)
	if conn != nil {
		_ = conn.Close()
		t.Fatal("connected to an absent private pipe")
	}
	if !errors.Is(err, ErrNotRunning) || !errors.Is(err, windows.ERROR_FILE_NOT_FOUND) {
		t.Fatalf("DialEndpoint = %v, want ErrNotRunning and the original missing-pipe cause", err)
	}
}

func TestDialEndpointBusyPipePreservesDeadline(t *testing.T) {
	_, endpoint := windowsTestEndpoint(t)
	holdWindowsTestPipe(t, endpoint, "")
	requireWindowsPipeOpenError(t, endpoint, windows.ERROR_PIPE_BUSY)
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	conn, err := DialEndpoint(ctx, endpoint)
	if conn != nil {
		_ = conn.Close()
		t.Fatal("connected to a private pipe without Accept")
	}
	if !errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrNotRunning) {
		t.Fatalf("DialEndpoint = %v, want deadline exceeded without ErrNotRunning", err)
	}
}

func TestConnectOrStartBusyPipePreservesCancellation(t *testing.T) {
	dir, endpoint := windowsTestEndpoint(t)
	holdWindowsTestPipe(t, endpoint, "")
	requireWindowsPipeOpenError(t, endpoint, windows.ERROR_PIPE_BUSY)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	timer := time.AfterFunc(100*time.Millisecond, cancel)
	defer timer.Stop()
	conn, err := ConnectOrStart(ctx, filepath.Join(dir, "must-not-start.exe"), nil)
	requireWindowsConnectError(t, conn, err, context.Canceled)
}

func TestConnectOrStartEndedContextDoesNotStart(t *testing.T) {
	test := func(name string, ctx context.Context, want error) {
		t.Run(name, func(t *testing.T) {
			dir, _ := windowsTestEndpoint(t)
			conn, err := ConnectOrStart(ctx, filepath.Join(dir, "must-not-start.exe"), nil)
			requireWindowsConnectError(t, conn, err, want)
		})
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	test("canceled", canceled, context.Canceled)
	expired, stop := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer stop()
	test("deadline expired", expired, context.DeadlineExceeded)
}

func TestConnectOrStartDeniedPipeDoesNotStart(t *testing.T) {
	dir, endpoint := windowsTestEndpoint(t)
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		t.Fatal(err)
	}
	// Allow the listener's SYNCHRONIZE-only reservation, but no client reads or
	// writes. This descriptor belongs only to the test's unique private pipe.
	holdWindowsTestPipe(t, endpoint, "D:P(A;;0x00100000;;;"+user.User.Sid.String()+")")
	requireWindowsPipeOpenError(t, endpoint, windows.ERROR_ACCESS_DENIED)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	conn, err := ConnectOrStart(ctx, filepath.Join(dir, "must-not-start.exe"), nil)
	requireWindowsConnectError(t, conn, err, windows.ERROR_ACCESS_DENIED)
}

func TestDialEndpointInvalidNamePreservesCause(t *testing.T) {
	_, endpoint := windowsTestEndpoint(t)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	// NUL cannot enter the Windows environment. Exercise argument validation
	// directly; the denied-pipe case separately proves ConnectOrStart refusal.
	conn, err := DialEndpoint(ctx, endpoint+"\x00")
	if conn != nil {
		_ = conn.Close()
		t.Fatal("connected using an endpoint containing NUL")
	}
	if !errors.Is(err, syscall.EINVAL) || errors.Is(err, ErrNotRunning) || errors.Is(err, os.ErrNotExist) {
		t.Fatalf("DialEndpoint = %v, want EINVAL without endpoint absence", err)
	}
}

func TestConnectOrStartStartsForAbsentPipe(t *testing.T) {
	_, endpoint := windowsTestEndpoint(t)
	requireWindowsPipeOpenError(t, endpoint, windows.ERROR_FILE_NOT_FOUND)
	t.Setenv(windowsFakeServiceEnv, "accepts")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, err := ConnectOrStart(ctx, os.Args[0], nil)
	if err != nil {
		t.Fatal(err)
	}
	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	defer stop()
	var process windows.Handle
	t.Cleanup(func() {
		_ = conn.Close()
		if process == 0 {
			return
		}
		defer windows.CloseHandle(process)
		event, err := windows.WaitForSingleObject(process, 3000)
		if err != nil || event != windows.WAIT_OBJECT_0 {
			t.Errorf("join detached fake service: event %d, error %v", event, err)
			if err := windows.TerminateProcess(process, 1); err != nil {
				t.Errorf("terminate detached fake service: %v", err)
			}
			if event, err := windows.WaitForSingleObject(process, 3000); err != nil || event != windows.WAIT_OBJECT_0 {
				t.Errorf("join terminated fake service: event %d, error %v", event, err)
			}
		}
	})
	var pid uint32
	if err := json.NewDecoder(conn).Decode(&pid); err != nil {
		t.Fatal(err)
	}
	if pid == 0 || pid == uint32(os.Getpid()) {
		t.Fatalf("fake service PID = %d, want a detached child", pid)
	}
	process, err = windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_TERMINATE, false, pid)
	if err != nil {
		t.Fatalf("own detached fake service: %v", err)
	}
}

func TestConnectOrStartPostStartDeadlinePreservesCause(t *testing.T) {
	dir, endpoint := windowsTestEndpoint(t)
	requireWindowsPipeOpenError(t, endpoint, windows.ERROR_FILE_NOT_FOUND)
	// The fixed-path test image exits immediately in this mode. A successful
	// process start must still bound its absent-endpoint wait by the caller.
	t.Setenv(windowsFakeServiceEnv, "exits without listening")
	pidFile := filepath.Join(dir, "fake-pid.json")
	t.Setenv(windowsFakePIDEnv, pidFile)
	t.Cleanup(func() { joinExitedWindowsFake(t, pidFile) })
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	conn, err := ConnectOrStart(ctx, os.Args[0], nil)
	requireWindowsConnectError(t, conn, err, context.DeadlineExceeded)
}

func joinExitedWindowsFake(t *testing.T, pidFile string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	var data []byte
	for {
		var err error
		data, err = os.ReadFile(pidFile)
		if err == nil {
			break
		}
		if !errors.Is(err, os.ErrNotExist) || time.Now().After(deadline) {
			t.Fatalf("read detached fake PID: %v", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	var pid uint32
	if err := json.Unmarshal(data, &pid); err != nil {
		t.Fatal(err)
	}
	if pid == 0 || pid == uint32(os.Getpid()) {
		t.Fatalf("invalid detached fake PID %d", pid)
	}
	process, err := windows.OpenProcess(windows.SYNCHRONIZE, false, pid)
	if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
		return // The self-terminating fake has already exited.
	}
	if err != nil {
		t.Fatal(err)
	}
	defer windows.CloseHandle(process)
	if event, err := windows.WaitForSingleObject(process, 3000); err != nil || event != windows.WAIT_OBJECT_0 {
		t.Fatalf("join exiting fake service: event %d, error %v", event, err)
	}
}

func windowsTestEndpoint(t *testing.T) (string, string) {
	t.Helper()
	dir, err := os.MkdirTemp(t.TempDir(), "sctl-pipe-")
	if err != nil {
		t.Fatal(err)
	}
	endpoint := `\\.\pipe\` + filepath.Base(dir)
	t.Setenv(EndpointEnv, endpoint)
	return dir, endpoint
}

func holdWindowsTestPipe(t *testing.T, endpoint, sddl string) {
	t.Helper()
	if sddl == "" {
		var err error
		sddl, err = ipc.CurrentUserSDDL()
		if err != nil {
			t.Fatal(err)
		}
	}
	listener, err := ipc.ListenSDDL(endpoint, sddl)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := listener.Close(); err != nil {
			t.Errorf("close private listener: %v", err)
		}
	})
}

func requireWindowsPipeOpenError(t *testing.T, endpoint string, want error) {
	t.Helper()
	path, err := windows.UTF16PtrFromString(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	handle, err := windows.CreateFile(path, windows.GENERIC_READ|windows.GENERIC_WRITE, 0, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_OVERLAPPED, 0)
	if err == nil {
		if err := windows.CloseHandle(handle); err != nil {
			t.Errorf("close unexpected pipe connection: %v", err)
		}
		t.Fatalf("private pipe connected, want %v", want)
	}
	if !errors.Is(err, want) {
		t.Fatalf("private pipe open = %v, want %v", err, want)
	}
}

func requireWindowsConnectError(t *testing.T, conn io.ReadWriteCloser, err, want error) {
	t.Helper()
	if conn != nil {
		_ = conn.Close()
		t.Fatal("unexpected service connection")
	}
	if !errors.Is(err, want) || errors.Is(err, ErrNotRunning) || errors.Is(err, os.ErrNotExist) {
		t.Fatalf("ConnectOrStart = %v, want %v without absence or launch errors", err, want)
	}
}
