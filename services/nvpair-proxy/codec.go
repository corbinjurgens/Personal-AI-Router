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
	"nvpair-shared/nodepolicy"
)

type (
	Message  = jsonrpc.Message
	RPCError = jsonrpc.RPCError
)

const (
	notificationWriteBudget = 60 * time.Second
	proxyOrdinaryFrameBytes = 1 << 20
)

// Codec adds the parent session's notification lifetime to shared framing.
type Codec struct {
	*jsonrpc.Codec
	root   context.Context
	budget time.Duration
	fail   func(string, error)
	// Construction-only transport/hook references permit NewProxy to replace
	// the unused ordinary scanner before publishing any reader or writer.
	transport io.ReadWriter
	interrupt func()
	// The ordinary framing allowance, shared by all input. History output and
	// offsets borrow the sole history owner's existing P/R banks.
	raw []byte
}

func NewCodec(rw io.ReadWriter) *Codec {
	return &Codec{Codec: jsonrpc.NewCodec(rw), root: context.Background(), budget: notificationWriteBudget, transport: rw}
}

func newOwnedCodec(root context.Context, transport io.ReadWriteCloser, fail func(string, error), budget time.Duration) *Codec {
	interrupt := func() {
		fail("active-write", errors.New("parent transport write failed or was interrupted"))
	}
	return &Codec{
		Codec: jsonrpc.NewCodecMaxFrameWithWriteInterrupt(transport, proxyOrdinaryFrameBytes, interrupt),
		root:  root, budget: budget, fail: fail, transport: transport, interrupt: interrupt,
	}
}

// Called once during NewProxy, before IO or publication. Drop the unused
// scanner before allocating its replacement ordinary raw allowance; no second
// reader or gate remains reachable. This is not a runtime codec replacement.
func (c *Codec) attachRoutingHistory(h *proxyRoutingHistory) error {
	if c.transport == nil || c.raw != nil || h == nil {
		return jsonrpc.ErrHistoryInvalid
	}
	c.Codec = nil
	c.raw = make([]byte, 0, proxyOrdinaryFrameBytes)
	codec, err := jsonrpc.NewHistoryCodec(c.transport, proxyOrdinaryFrameBytes, c.raw,
		h.frames[1][:0], h.guard[:], &h.correlation, nodepolicy.RoutingHistoryTransportSchema, c.interrupt)
	if err != nil {
		return err
	}
	c.Codec = codec
	c.transport, c.interrupt = nil, nil
	return nil
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

// History acquisition tries once, with no gate ownership outside this one
// frame. The shared codec borrows frame through actual Write completion and
// uses the parent's lifetime and existing write-interrupt owner. Fully stopped
// output still couples ordinary notifications through their existing budget;
// history adds no gate lease across frames, acknowledgements or copier turns.
func (c *Codec) writeRoutingFrame(parent context.Context, frame []byte) error {
	return c.Codec.WritePreencoded(nil, parent, frame)
}
