// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"

	"nvpair-shared/nodepolicy"
)

func (b *Broker) handleRoutingControl(msg *Message) {
	method := nodepolicy.MethodRoutingActive
	if msg.Method == "routing:cancel" {
		var in nodepolicy.RoutingCancelParams
		if json.Unmarshal(msg.Params, &in) != nil || in.Request.OriginNodeID != b.nodeID {
			_ = b.codec.RespondError(msg.ID, -32602, "routing cancellation requires a complete origin-local key")
			return
		}
		method = nodepolicy.MethodRoutingCancel
	} else {
		var in nodepolicy.RoutingActiveParams
		if json.Unmarshal(msg.Params, &in) != nil {
			_ = b.codec.RespondError(msg.ID, -32602, "invalid routing active params")
			return
		}
	}
	if !b.routingForward.CompareAndSwap(false, true) {
		_ = b.codec.RespondError(msg.ID, -32000, "routing control busy; retry")
		return
	}
	go func() {
		defer b.routingForward.Store(false)
		proxy := b.policyProxy()
		if proxy == nil {
			_ = b.codec.RespondError(msg.ID, -32000, "routing proxy unavailable")
			return
		}
		result, rpcErr, err := proxy.Call(context.Background(), method, msg.Params)
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, "routing proxy call failed")
			return
		}
		if rpcErr != nil {
			// Forward only the bounded control vocabulary, never arbitrary worker text.
			code := -32000
			if rpcErr.Code == -32602 {
				code = -32602
			}
			message := "routing proxy refused control request"
			switch rpcErr.Message {
			case "routing snapshot busy; retry", "invalid routing active params", "invalid routing cancellation params":
				message = rpcErr.Message
			}
			_ = b.codec.RespondError(msg.ID, code, message)
			return
		}
		if method == nodepolicy.MethodRoutingCancel {
			var out nodepolicy.RoutingCancelResult
			err = json.Unmarshal(result, &out)
		} else {
			var out nodepolicy.RoutingActiveResult
			err = json.Unmarshal(result, &out)
			if err == nil {
				for _, key := range out.Requests {
					if key.OriginNodeID != b.nodeID {
						err = errors.New("foreign routing origin")
						break
					}
				}
			}
		}
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, "invalid routing proxy result")
			return
		}
		_ = b.codec.Respond(msg.ID, result)
	}()
}
