// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package servicectl

import (
	"context"
	"io"
	"net"
	"os/exec"
	"syscall"

	"nvpair-shared/appdir"
)

// socketName is the service's Unix socket inside the per-user data directory.
const socketName = "service.sock"

func defaultEndpoint() (string, error) {
	return appdir.Path(socketName)
}

func dialEndpoint(ctx context.Context, endpoint string) (io.ReadWriteCloser, error) {
	var d net.Dialer
	return d.DialContext(ctx, "unix", endpoint)
}

// configureDetached puts the service in a new session, so it has no
// controlling terminal and neither a hangup on the caller's terminal nor a
// Ctrl+C delivered to the caller's process group reaches it.
func configureDetached(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}
