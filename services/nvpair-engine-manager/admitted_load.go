// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

const admittedLoadRequestMethod = "engine:admitted-load-request"
const admittedLoadReplyMethod = "engine:admitted-load-reply"

type admittedLoadReply struct {
	ID     string             `json:"id"`
	Result json.RawMessage    `json:"result,omitempty"`
	Error  *admittedLoadError `json:"error,omitempty"`
}

type admittedLoadError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// This transports a peer load to the broker's existing admission owner.
type admittedLoadRelay struct {
	mu      sync.Mutex
	pending map[string]chan admittedLoadReply
	send    func(context.Context, string, any) error
}

func (r *admittedLoadRelay) call(ctx context.Context, engine, model string) (json.RawMessage, error) {
	if r == nil || r.send == nil {
		return nil, fmt.Errorf("load admission coordinator unavailable")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	id := newOpID()
	ch := make(chan admittedLoadReply, 1)
	r.mu.Lock()
	if len(r.pending) >= maxPendingPolicy {
		r.mu.Unlock()
		return nil, fmt.Errorf("too many load admission requests")
	}
	if r.pending == nil {
		r.pending = make(map[string]chan admittedLoadReply)
	}
	r.pending[id] = ch
	r.mu.Unlock()
	defer func() { r.mu.Lock(); delete(r.pending, id); r.mu.Unlock() }()
	request := struct {
		ID     string `json:"id"`
		Params struct {
			Engine string `json:"engine"`
			Model  string `json:"model"`
		} `json:"params"`
	}{ID: id}
	request.Params.Engine, request.Params.Model = engine, model
	if err := r.send(ctx, admittedLoadRequestMethod, request); err != nil {
		return nil, err
	}
	select {
	case reply := <-ch:
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if reply.Error != nil {
			return nil, &codedError{code: reply.Error.Code, message: reply.Error.Message}
		}
		return reply.Result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (r *admittedLoadRelay) reply(raw json.RawMessage) {
	var reply admittedLoadReply
	if json.Unmarshal(raw, &reply) != nil || reply.ID == "" {
		return
	}
	r.mu.Lock()
	ch := r.pending[reply.ID]
	r.mu.Unlock()
	if ch != nil {
		select {
		case ch <- reply:
		default:
		}
	}
}
