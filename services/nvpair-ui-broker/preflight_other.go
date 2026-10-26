// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"errors"
	"net"
	"strconv"
	"syscall"
)

// Facades can bind wildcards; managed launch settings require loopback binds
// (127.0.0.1 or ::1). On BSD a wildcard listener can coexist with an exact
// loopback listener, so a successful wildcard probe alone is insufficient.
// Probe both families explicitly, closing each temporary listener immediately.
// The real bind remains authoritative if another process wins the later race.
func tcpPortAvailable(port int) bool {
	for _, probe := range []struct{ network, host string }{
		{"tcp4", "0.0.0.0"}, {"tcp6", "::"},
		{"tcp4", "127.0.0.1"}, {"tcp6", "::1"},
	} {
		ln, err := net.Listen(probe.network, net.JoinHostPort(probe.host, strconv.Itoa(port)))
		if err != nil {
			// Hosts without IPv6 cannot launch an IPv6 listener. Other errors,
			// including address-in-use and descriptor exhaustion, fail closed.
			if probe.network == "tcp6" && (errors.Is(err, syscall.EAFNOSUPPORT) || errors.Is(err, syscall.EPROTONOSUPPORT) || errors.Is(err, syscall.EADDRNOTAVAIL)) {
				continue
			}
			return false
		}
		if err := ln.Close(); err != nil {
			return false
		}
	}
	return true
}
