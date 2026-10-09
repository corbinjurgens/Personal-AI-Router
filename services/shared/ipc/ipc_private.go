// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package ipc

import (
	"fmt"
	"net"
	"os"
	"syscall"
)

// ListenPrivate is Listen for an endpoint only the current user may open: the
// Unix socket is created with mode 0600.
//
// The umask is narrowed around the bind so the socket never exists with wider
// permissions, even briefly, and the mode is then set explicitly in case the
// platform ignores the umask for sockets. The umask is process-wide, so call
// this before starting anything else that creates files concurrently.
func ListenPrivate(path string) (net.Listener, error) {
	old := syscall.Umask(0o177)
	l, err := net.Listen("unix", path)
	syscall.Umask(old)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		_ = l.Close()
		return nil, fmt.Errorf("restrict %s to the current user: %w", path, err)
	}
	return l, nil
}
