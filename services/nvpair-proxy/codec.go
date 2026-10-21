// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// The newline-delimited JSON-RPC 2.0 codec is single-sourced in
// nvpair-shared/jsonrpc. These local aliases keep this package's call sites and
// tests unchanged after removing the copy-pasted per-service codec.

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"time"

	"nvpair-shared/jsonrpc"
)

type (
	Message  = jsonrpc.Message
	RPCError = jsonrpc.RPCError
)

const notificationWriteBudget = 60 * time.Second

// Codec adds the parent session's notification lifetime to shared framing.
type Codec struct {
	*jsonrpc.Codec
	root   context.Context
	budget time.Duration
	fail   func(string, error)
}

func NewCodec(rw io.ReadWriter) *Codec {
	return &Codec{Codec: jsonrpc.NewCodec(rw), root: context.Background(), budget: notificationWriteBudget}
}

func newOwnedCodec(root context.Context, transport io.ReadWriteCloser, fail func(string, error), budget time.Duration) *Codec {
	return &Codec{
		Codec: jsonrpc.NewCodecMaxFrameWithWriteInterrupt(transport, 1<<20, func() {
			fail("active-write", errors.New("parent transport write failed or was interrupted"))
		}),
		root: root, budget: budget, fail: fail,
	}
}

func (c *Codec) Notify(method string, params any) error {
	err := c.NotifyContext(c.root, method, params)
	if err != nil && c.fail != nil {
		slog.Error("parent notification failed", "method", method, "err", err)
		c.fail(method, err)
	}
	return err
}

func (c *Codec) NotifyContext(ctx context.Context, method string, params any) error {
	ctx, cancel := context.WithTimeout(ctx, c.budget)
	defer cancel()
	return c.Codec.NotifyContext(ctx, method, params)
}
