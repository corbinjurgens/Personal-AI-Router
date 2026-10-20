// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"time"

	"nvpair-shared/jsonrpc"
)

type (
	Message     = jsonrpc.Message
	RPCError    = jsonrpc.RPCError
	DecodeError = jsonrpc.DecodeError
)

// maxFrameBytes caps a single inbound JSON-RPC frame. It is the shared
// worker-path cap so every hop agrees; the orchestrator only sends small
// engine:* requests inbound, but this side's *replies* can be megabytes
// (engine:catalog's model list), and a cap that differs per hop fails the whole
// path. An inbound frame larger than this surfaces as a terminal read error and
// manager.readLoop exits cleanly.
const maxFrameBytes = jsonrpc.WorkerFrameBytes

const notificationWriteBudget = 60 * time.Second

// Codec keeps the shared framing and write gate, adding the parent session's
// notification lifetime and default-publication failure policy.
type Codec struct {
	*jsonrpc.Codec
	root   context.Context
	budget time.Duration
	fail   func(string, error)
}

// NewCodec wraps rw with the shared codec at engine-manager's 8 MiB frame cap.
func NewCodec(rw io.ReadWriter) *Codec {
	return &Codec{Codec: jsonrpc.NewCodecMaxFrame(rw, maxFrameBytes), root: context.Background(), budget: notificationWriteBudget}
}

// budget is supplied only at construction; production uses notificationWriteBudget.
func newOwnedCodec(root context.Context, transport io.ReadWriteCloser, fail func(string, error), budget time.Duration) *Codec {
	return &Codec{
		Codec: jsonrpc.NewCodecMaxFrameWithWriteInterrupt(transport, maxFrameBytes, func() {
			fail("active-write", errors.New("parent transport write failed or was interrupted"))
		}),
		root: root, budget: budget, fail: fail,
	}
}

func (c *Codec) Notify(method string, params any) error {
	err := c.NotifyContext(c.root, method, params)
	if err != nil && c.fail != nil {
		slog.Error("parent notification failed", "method", method, "err", err)
		// Default publication failure is terminal even if this send timed out
		// in the gate and therefore never invoked the active-write hook.
		c.fail(method, err)
	}
	return err
}

func (c *Codec) NotifyContext(ctx context.Context, method string, params any) error {
	ctx, cancel := context.WithTimeout(ctx, c.budget)
	defer cancel()
	return c.Codec.NotifyContext(ctx, method, params)
}

// cancelRelayWait runs only after the pending entry has been removed. Cleanup
// has its own joined tail; it cannot inherit the expired request's context.
func cancelRelayWait(send func(context.Context, string, any) error, method, id string) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := send(ctx, method, map[string]string{"id": id}); err != nil {
		slog.Warn("relay cancellation send failed", "method", method, "id", id)
	}
}
