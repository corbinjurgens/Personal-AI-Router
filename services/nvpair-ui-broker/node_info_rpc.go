// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"strings"
	"time"
)

const nodeInfoRPCBudget = 1500 * time.Millisecond

type nodeInfoRPCParams struct {
	HostUUID string `json:"hostUuid"`
}

func (s *scannerProcess) getNodeInfo(ctx context.Context, params json.RawMessage) (json.RawMessage, *RPCError, error) {
	return s.peer.Call(ctx, "discovery:get-node-info", params)
}

// handleNodeInfoRPC runs off the broker read pump. The scanner owns fetching,
// authority and parsing; the broker preserves its full result and RPC errors.
func (b *Broker) handleNodeInfoRPC(msg *Message) {
	var p nodeInfoRPCParams
	decoder := json.NewDecoder(bytes.NewReader(msg.Params))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&p); err != nil || strings.TrimSpace(p.HostUUID) == "" {
		_ = b.codec.RespondError(msg.ID, -32602, "expected {hostUuid: string}")
		return
	}
	if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
		_ = b.codec.RespondError(msg.ID, -32602, "expected one params object")
		return
	}
	scanner := b.getScanner()
	if scanner == nil || scanner.peer == nil {
		_ = b.codec.RespondError(msg.ID, -32000, "node scanner unavailable")
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), nodeInfoRPCBudget)
	defer cancel()
	result, rpcErr, err := scanner.getNodeInfo(ctx, msg.Params)
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32000, "node information unavailable")
	} else if rpcErr != nil {
		_ = b.codec.RespondErrorData(msg.ID, rpcErr.Code, rpcErr.Message, rpcErr.Data)
	} else {
		_ = b.codec.Respond(msg.ID, result)
	}
}
