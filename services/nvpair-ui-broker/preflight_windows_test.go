// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"net"
	"syscall"
	"testing"
)

type preflightTestRawConn struct {
	syscall.RawConn
	controlErr error
}

func (raw *preflightTestRawConn) Control(f func(uintptr)) error {
	if raw.controlErr != nil {
		return raw.controlErr
	}
	f(17)
	return nil
}

func TestTCPPortPreflightAlwaysStopsBeforeListen(t *testing.T) {
	refused := errors.New("fixture socket operation refused")
	const port = 45678 // Synthetic only; the socket seam never opens a port.
	for _, tc := range []struct {
		name        string
		network     string
		controlErr  error
		optionErr   error
		bindErr     error
		wantOptions int
		wantBinds   int
		want        bool
	}{
		{name: "IPv4 wildcard is free", network: "tcp4", wantOptions: 1, wantBinds: 1, want: true},
		{name: "IPv6 wildcard is free", network: "tcp6", wantOptions: 1, wantBinds: 1, want: true},
		{name: "exclusive option fails", network: "tcp6", optionErr: refused, wantOptions: 1},
		{name: "occupied wildcard is refused", network: "tcp6", bindErr: refused, wantOptions: 1, wantBinds: 1},
		{name: "raw control fails", network: "tcp4", controlErr: refused},
		{name: "unexpected family is refused", network: "udp4"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			options, binds, nativeListens, closes := 0, 0, 0, 0
			ops := tcpPortPreflightOps{
				setOption: func(fd syscall.Handle, level, option, value int) error {
					options++
					if fd != 17 || level != syscall.SOL_SOCKET || option != ^syscall.SO_REUSEADDR || value != 1 {
						t.Fatalf("socket option = (%d, %d, %d, %d), want exclusive address use on the supplied socket", fd, level, option, value)
					}
					return tc.optionErr
				},
				bind: func(fd syscall.Handle, address syscall.Sockaddr) error {
					binds++
					if fd != 17 {
						t.Fatalf("bind socket = %d, want 17", fd)
					}
					switch address := address.(type) {
					case *syscall.SockaddrInet4:
						if tc.network != "tcp4" || address.Port != port || address.Addr != [4]byte{} {
							t.Fatalf("IPv4 bind address = %+v, want wildcard port %d for %s", address, port, tc.network)
						}
					case *syscall.SockaddrInet6:
						if tc.network != "tcp6" || address.Port != port || address.Addr != [16]byte{} || address.ZoneId != 0 {
							t.Fatalf("IPv6 bind address = %+v, want wildcard port %d for %s", address, port, tc.network)
						}
					default:
						t.Fatalf("unexpected sockaddr type %T", address)
					}
					return tc.bindErr
				},
				listen: func(config *net.ListenConfig, _ context.Context, network, address string) (net.Listener, error) {
					if network != "tcp" || address != ":45678" {
						t.Fatalf("socket request = (%q, %q), want the proxy's wildcard TCP address", network, address)
					}
					raw := &preflightTestRawConn{controlErr: tc.controlErr}
					controlErr := config.Control(tc.network, address, raw)
					// Model Go's control-before-bind/listen branch. A nil return
					// would let the normal listener path run and fails this test.
					if controlErr == nil {
						nativeListens++
					} else {
						closes++
					}
					if controlErr != errTCPPreflightStopped {
						t.Errorf("Control returned %v, want the unconditional stop sentinel", controlErr)
					}
					return nil, &net.OpError{Op: "listen", Net: network, Err: controlErr}
				},
			}
			if got := tcpPortAvailableWithOps(port, ops); got != tc.want {
				t.Errorf("available = %v, want %v", got, tc.want)
			}
			if options != tc.wantOptions || binds != tc.wantBinds {
				t.Errorf("socket calls = (%d options, %d binds), want (%d, %d)", options, binds, tc.wantOptions, tc.wantBinds)
			}
			if nativeListens != 0 || closes != 1 {
				t.Errorf("native listens = %d, closes = %d, want no listen and one close", nativeListens, closes)
			}
		})
	}
}

func TestTCPPortPreflightRejectsSocketSetupFailure(t *testing.T) {
	ops := tcpPortPreflightOps{
		setOption: func(syscall.Handle, int, int, int) error {
			t.Fatal("socket setup failure reached setsockopt")
			return nil
		},
		bind: func(syscall.Handle, syscall.Sockaddr) error {
			t.Fatal("socket setup failure reached bind")
			return nil
		},
		listen: func(*net.ListenConfig, context.Context, string, string) (net.Listener, error) {
			return nil, errors.New("fixture socket creation failed")
		},
	}
	if tcpPortAvailableWithOps(0, ops) {
		t.Fatal("socket setup failure reported the port available")
	}
}

type preflightUnexpectedListener struct {
	net.Listener
	closeCalls int
}

func (ln *preflightUnexpectedListener) Close() error {
	ln.closeCalls++
	return errors.New("fixture close failed")
}

func TestTCPPortPreflightClosesUnexpectedListener(t *testing.T) {
	ln := &preflightUnexpectedListener{}
	ops := tcpPortPreflightOps{
		listen: func(*net.ListenConfig, context.Context, string, string) (net.Listener, error) {
			return ln, nil
		},
	}
	if tcpPortAvailableWithOps(0, ops) {
		t.Fatal("unexpected listener reported the port available")
	}
	if ln.closeCalls != 1 {
		t.Fatalf("unexpected listener closed %d times, want once even on close failure", ln.closeCalls)
	}
}

func TestTCPPortPreflightGoClosesSocketAfterControlAbort(t *testing.T) {
	owner, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("allocate private occupied endpoint: %v", err)
	}
	t.Cleanup(func() { _ = owner.Close() })
	occupied := owner.Addr().(*net.TCPAddr).Port
	for _, tc := range []struct {
		name string
		port int
		want bool
	}{
		{name: "free ephemeral wildcard", port: 0, want: true},
		{name: "occupied loopback", port: occupied},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var raw syscall.RawConn
			ops := tcpPortPreflightOps{
				setOption: syscall.SetsockoptInt,
				bind:      syscall.Bind,
				listen: func(config *net.ListenConfig, ctx context.Context, network, address string) (net.Listener, error) {
					control := config.Control
					config.Control = func(network, address string, socket syscall.RawConn) error {
						raw = socket
						err := control(network, address, socket)
						if err != errTCPPreflightStopped {
							t.Fatal("Control did not stop Go before native bind/listen")
						}
						return err
					}
					return config.Listen(ctx, network, address)
				},
			}
			if got := tcpPortAvailableWithOps(tc.port, ops); got != tc.want {
				t.Fatalf("available = %v, want %v", got, tc.want)
			}
			if raw == nil {
				t.Fatal("Go did not supply a socket to Control")
			}
			if err := raw.Control(func(uintptr) {
				t.Error("Go left the preflight socket open after Control aborted")
			}); !errors.Is(err, net.ErrClosed) {
				t.Fatalf("control after preflight = %v, want closed socket", err)
			}
		})
	}
}

func TestBrokerPortAvailableRejectsOwnedLoopbackPort(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("allocate private occupied endpoint: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	b := &Broker{}
	if b.portAvailable(ln.Addr().(*net.TCPAddr).Port) {
		t.Fatal("production preflight accepted an occupied loopback port")
	}
	if !b.portAvailable(0) {
		t.Fatal("production preflight refused a free ephemeral port")
	}
	if err := ln.Close(); err != nil {
		t.Fatalf("close fixture owner after refused preflight: %v", err)
	}
}
