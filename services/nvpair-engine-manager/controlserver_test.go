// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"net"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestControlHostDefault(t *testing.T) {
	if defaultControlHost != "0.0.0.0" {
		t.Fatalf("default control host = %q, want 0.0.0.0", defaultControlHost)
	}
}

func TestControlHostLiteralValidation(t *testing.T) {
	for _, host := range []string{"0.0.0.0", "127.0.0.1", "192.0.2.1", "::", "::1", "2001:db8::1"} {
		t.Run("accepts "+host, func(t *testing.T) {
			if err := validateControlHost(host); err != nil {
				t.Fatalf("literal control host rejected: %v", err)
			}
		})
	}
	for _, host := range []string{"", "localhost", "example.invalid", "127.0.0.1:14323", "[::1]", "::1%lo0", " 127.0.0.1", "127.0.0.1 ", "256.0.0.1"} {
		t.Run("rejects "+host, func(t *testing.T) {
			if err := validateControlHost(host); err == nil {
				t.Fatal("invalid control host accepted")
			}
			listener, err := listenControl(host, 0)
			if listener != nil {
				if closeErr := listener.Close(); closeErr != nil {
					t.Fatalf("close unexpected listener: %v", closeErr)
				}
				t.Fatal("invalid control host bound a listener")
			}
			if err == nil {
				t.Fatal("invalid control host did not return an error")
			}
		})
	}
}

func TestControlHostListenerBindsIPv4Loopback(t *testing.T) {
	listener, err := listenControl("127.0.0.1", 0)
	if err != nil {
		t.Fatalf("bind control listener: %v", err)
	}
	defer func() {
		if err := listener.Close(); err != nil {
			t.Errorf("close control listener: %v", err)
		}
	}()
	address, ok := listener.Addr().(*net.TCPAddr)
	if !ok || !address.IP.Equal(net.ParseIP("127.0.0.1")) || address.Port <= 0 {
		t.Fatalf("actual control listener = %v, want 127.0.0.1 on an allocated port", listener.Addr())
	}
}

func TestControlHostListenerBindsIPv6Loopback(t *testing.T) {
	probe, err := net.Listen("tcp6", "[::1]:0")
	if err != nil {
		t.Skipf("IPv6 loopback unavailable: %v", err)
	}
	if err := probe.Close(); err != nil {
		t.Fatalf("close IPv6 availability probe: %v", err)
	}
	listener, err := listenControl("::1", 0)
	if err != nil {
		t.Fatalf("bind IPv6 control listener: %v", err)
	}
	defer func() {
		if err := listener.Close(); err != nil {
			t.Errorf("close IPv6 control listener: %v", err)
		}
	}()
	address, ok := listener.Addr().(*net.TCPAddr)
	if !ok || !address.IP.Equal(net.ParseIP("::1")) || address.Port <= 0 {
		t.Fatalf("actual control listener = %v, want ::1 on an allocated port", listener.Addr())
	}
}

func TestControlHostCLIRejectsBeforeStartup(t *testing.T) {
	for _, host := range []string{"", "localhost", "127.0.0.1:14323", "[::1]", "256.0.0.1"} {
		t.Run("rejects "+host, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			root := t.TempDir()
			cmd := exec.CommandContext(ctx, managerBin, "--control-host", host, "--ipc", filepath.Join(root, "missing-ipc"))
			cmd.Env = isolatedManagerEnv(root, root)
			output, err := cmd.CombinedOutput()
			if ctx.Err() != nil || err == nil {
				t.Fatalf("invalid CLI host did not fail promptly: %v, %v, %s", ctx.Err(), err, output)
			}
			if !strings.Contains(string(output), "invalid --control-host:") || strings.Contains(string(output), "IPC endpoint") {
				t.Fatalf("diagnostic = %q, want host rejection before transport startup", output)
			}
		})
	}
}
