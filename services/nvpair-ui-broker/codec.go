// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// The newline-delimited JSON-RPC 2.0 codec and the bidirectional Peer
// (id-correlation + read pump) are single-sourced in nvpair-shared/jsonrpc. These
// local aliases keep the broker's call sites unchanged. This covers the
// broker's client-facing Codec and the parent-side worker handles, which are
// built on Peer.

import (
	"io"
	"sync"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

type (
	Message  = jsonrpc.Message
	RPCError = jsonrpc.RPCError
	Codec    = jsonrpc.Codec
	Peer     = jsonrpc.Peer
)

var (
	NewCodec         = jsonrpc.NewCodec
	NewCodecMaxFrame = jsonrpc.NewCodecMaxFrame
	NewPeer          = jsonrpc.NewPeer
)

// workerFrameBytes is the inbound frame cap for a worker link. Single-sourced
// with every other hop on the same path, because a reply only arrives if all of
// them agree.
const workerFrameBytes = jsonrpc.WorkerFrameBytes

// errPeerClosed is the sentinel a worker handle's Call/RelayRequest returns
// once the child's transport has gone away.
var errPeerClosed = jsonrpc.ErrPeerClosed

// readWriter adapts a child's separate stdout (read) and stdin (write) into the
// io.ReadWriter the shared Codec expects.
type readWriter struct {
	io.Reader
	io.Writer
}

type routingHistoryReadRole uint8

const (
	historyProxyPrivate routingHistoryReadRole = iota
	historyLocalControl
)

// Exactly two admitted readers share fate with the broker history owner. Each
// slot keeps its original ordinary input backing and R offsets across restarts.
// Retired codec shells never own another bank. There is no client/worker pool.
type routingHistoryReadSlot struct {
	raw         []byte
	offsets     [jsonrpc.HistoryRecordBytes / 4]uint32
	control     jsonrpc.ReadControl
	correlation jsonrpc.HistoryCorrelation
	codec       *Codec
}

type routingHistoryReaders struct {
	mu    sync.Mutex
	slots [2]routingHistoryReadSlot
}

// acquire is a nonwaiting admission check. Only actual RetireRead completion
// permits rebinding; process Done, Close return and cancellation do not.
func (r *routingHistoryReaders) acquire(role routingHistoryReadRole, rw io.ReadWriter, interruptRead, interruptWrite func()) (*Codec, error) {
	if (role != historyProxyPrivate && role != historyLocalControl) || interruptRead == nil {
		return nil, jsonrpc.ErrHistoryInvalid
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	slot := &r.slots[role]
	if !slot.control.Available() {
		return nil, jsonrpc.ErrHistoryBusy
	}
	const ordinaryFrameBytes = 1 << 20
	if slot.raw == nil {
		// Replaces the ordinary reader's existing 1MiB input allocation. No
		// additional P/H frame is introduced; projection is in place.
		slot.raw = make([]byte, 0, ordinaryFrameBytes)
	}
	slot.correlation = jsonrpc.HistoryCorrelation{}
	codec, err := jsonrpc.NewHistoryCodec(rw, ordinaryFrameBytes, slot.raw, nil, slot.offsets[:],
		&slot.correlation, nodepolicy.RoutingHistoryTransportSchema, interruptWrite)
	if err != nil {
		return nil, err
	}
	if err := codec.EnableReadRetirement(&slot.control, interruptRead); err != nil {
		return nil, err
	}
	slot.codec = codec
	return codec, nil
}

// retire never holds admission or source-authority locks while interrupting
// the actual read owner. The fixed control admits rebind only after completion.
// Duplicate/late retirement cannot alter a replacement slot.
func (r *routingHistoryReaders) retire(role routingHistoryReadRole, codec *Codec) <-chan struct{} {
	return codec.RetireRead()
}
