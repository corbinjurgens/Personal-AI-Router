// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package clustertrust

import (
	"context"
	"net"

	"github.com/Microsoft/go-winio"
	"nvpair-shared/ipc"
)

func dialReversePrivate(ctx context.Context, path string) (net.Conn, error) {
	return winio.DialPipeContext(ctx, path)
}

// ListenReversePrivate adds pipe half-close without changing other private IPC.
func ListenReversePrivate(path string) (net.Listener, error) {
	return ipc.ListenPrivateMessagePipe(path)
}
