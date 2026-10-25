// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"syscall"
)

const preflightExclusiveAddrUse = ^syscall.SO_REUSEADDR

var errTCPPreflightStopped = errors.New("TCP port preflight stopped before listen")

type tcpPortPreflightOps struct {
	setOption func(syscall.Handle, int, int, int) error
	bind      func(syscall.Handle, syscall.Sockaddr) error
	listen    func(*net.ListenConfig, context.Context, string, string) (net.Listener, error)
}

// tcpPortAvailable checks the proxy's wildcard address without a native listen,
// which can request Windows firewall access. Go selects the same socket family
// and default dual-stack options as the proxy's net.Listen("tcp", ":port").
// Exclusive use also refuses concrete-address holders that Windows would allow
// to coexist with a default wildcard bind. The real proxy bind remains the
// authoritative, non-destructive check after this temporary socket is closed.
func tcpPortAvailable(port int) bool {
	return tcpPortAvailableWithOps(port, tcpPortPreflightOps{
		setOption: syscall.SetsockoptInt,
		bind:      syscall.Bind,
		listen:    (*net.ListenConfig).Listen,
	})
}

func tcpPortAvailableWithOps(port int, ops tcpPortPreflightOps) bool {
	bound := false
	config := net.ListenConfig{Control: func(network, _ string, raw syscall.RawConn) error {
		controlErr := raw.Control(func(fd uintptr) {
			var address syscall.Sockaddr
			switch network {
			case "tcp4":
				address = &syscall.SockaddrInet4{Port: port}
			case "tcp6":
				address = &syscall.SockaddrInet6{Port: port}
			default:
				return
			}
			socket := syscall.Handle(fd)
			if err := ops.setOption(socket, syscall.SOL_SOCKET, preflightExclusiveAddrUse, 1); err != nil {
				return
			}
			bound = ops.bind(socket, address) == nil
		})
		if controlErr != nil {
			bound = false
		}
		// Go's listenStream returns immediately on this error, before its Bind
		// or Listen, and socket closes its descriptor on that error path. Never
		// return nil here, including when the manual bind or RawConn fails.
		return errTCPPreflightStopped
	}}
	ln, err := ops.listen(&config, context.Background(), "tcp", fmt.Sprintf(":%d", port))
	if ln != nil {
		_ = ln.Close()
		return false
	}
	return bound && errors.Is(err, errTCPPreflightStopped)
}
