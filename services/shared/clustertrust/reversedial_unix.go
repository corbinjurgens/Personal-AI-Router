// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package clustertrust

import (
	"context"
	"net"

	"nvpair-shared/ipc"
)

func dialReversePrivate(ctx context.Context, path string) (net.Conn, error) {
	return (&net.Dialer{}).DialContext(ctx, "unix", path)
}

// ListenReversePrivate binds the owner-only service gateway before goroutines.
func ListenReversePrivate(path string) (net.Listener, error) { return ipc.ListenPrivate(path) }
