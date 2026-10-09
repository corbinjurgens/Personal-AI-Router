// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package ipc

import (
	"fmt"
	"net"

	"github.com/Microsoft/go-winio"
	"golang.org/x/sys/windows"
)

// ListenPrivate is Listen for an endpoint only the current user may open.
//
// Listen passes no security descriptor, so its named pipe gets the system
// default DACL, which grants read access to Everyone and to anonymous logons on
// top of full control for SYSTEM, Administrators, and the creator. Read access
// is enough to receive whatever the server writes, so a per-user control
// endpoint needs an explicit descriptor: this one is protected (no inherited
// entries) and grants access to the current user's SID alone. Listen itself is
// unchanged for its existing callers.
//
// A path that is not a pipe path falls back to a Unix domain socket, as Listen
// does; Windows has no mode bits to restrict it with.
func ListenPrivate(path string) (net.Listener, error) {
	if !isPipePath(path) {
		return net.Listen("unix", path)
	}
	sddl, err := CurrentUserSDDL()
	if err != nil {
		return nil, err
	}
	return ListenSDDL(path, sddl)
}

// ListenSDDL creates a named-pipe listener at path whose security descriptor is
// the given SDDL string.
func ListenSDDL(path, sddl string) (net.Listener, error) {
	return winio.ListenPipe(path, &winio.PipeConfig{SecurityDescriptor: sddl})
}

// CurrentUserSDDL returns a protected SDDL descriptor granting generic-all to
// the current process token's user and to no one else.
func CurrentUserSDDL() (string, error) {
	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		return "", fmt.Errorf("read current user SID: %w", err)
	}
	return "D:P(A;;GA;;;" + user.User.Sid.String() + ")", nil
}
