// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package servicectl

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"

	"nvpair-shared/appdir"
)

// socketName is the service's Unix socket inside the per-user data directory.
const socketName = "service.sock"

// maxSocketPath is the longest Unix socket path every supported platform
// accepts (sun_path is 104 bytes on macOS, 108 on Linux, including the NUL).
const maxSocketPath = 103

func defaultEndpoint() (string, error) {
	path, err := appdir.Path(socketName)
	if err != nil {
		return "", err
	}
	if len(path) <= maxSocketPath {
		return path, nil
	}
	return shortEndpoint()
}

// shortEndpoint is used when the app data path is too long for a socket, as it
// can be on macOS with a long user name: a private per-user directory under the
// system temp dir. The directory must be ours and closed to other users, so a
// directory another user pre-created there cannot capture the socket.
func shortEndpoint() (string, error) {
	dir := filepath.Join(os.TempDir(), fmt.Sprintf("nvpair-%d", os.Getuid()))
	if err := os.Mkdir(dir, 0o700); err != nil && !os.IsExist(err) {
		return "", err
	}
	fi, err := os.Lstat(dir)
	if err != nil {
		return "", err
	}
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !fi.IsDir() || fi.Mode().Perm()&0o077 != 0 || !ok || int(st.Uid) != os.Getuid() {
		return "", fmt.Errorf("service socket directory %s is not a private directory owned by this user", dir)
	}
	return filepath.Join(dir, socketName), nil
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
