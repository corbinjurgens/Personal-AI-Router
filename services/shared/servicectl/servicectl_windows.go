// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package servicectl

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"os/user"
	"strings"
	"syscall"

	"github.com/Microsoft/go-winio"
	"golang.org/x/sys/windows"
)

// Process creation flags for a service that must survive its launcher:
// no console of its own and none inherited, and a process group of its own so
// a console Ctrl+C sent to the launcher does not reach it.
const (
	detachedProcess       = 0x00000008
	createNewProcessGroup = 0x00000200
	createNoWindow        = 0x08000000
)

func defaultEndpoint() (string, error) {
	u, err := user.Current()
	if err != nil {
		return "", fmt.Errorf("resolve current user for the service pipe name: %w", err)
	}
	return `\\.\pipe\nvpair-service-` + pipeSafeUsername(u.Username), nil
}

// pipeSafeUsername reduces a Windows account name (often DOMAIN\user) to the
// user part, with any character a pipe name cannot carry replaced.
func pipeSafeUsername(name string) string {
	if i := strings.LastIndexByte(name, '\\'); i >= 0 {
		name = name[i+1:]
	}
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "user"
	}
	return b.String()
}

func dialEndpoint(ctx context.Context, endpoint string) (io.ReadWriteCloser, error) {
	conn, err := winio.DialPipeContext(ctx, endpoint)
	// A busy pipe is still owned by a listener. go-winio waits for an instance
	// until ctx ends; that timeout or cancellation must not authorize a start.
	if errors.Is(err, windows.ERROR_FILE_NOT_FOUND) {
		return nil, fmt.Errorf("%w: %w", ErrNotRunning, err)
	}
	return conn, err
}

func configureDetached(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: detachedProcess | createNewProcessGroup | createNoWindow,
	}
}
