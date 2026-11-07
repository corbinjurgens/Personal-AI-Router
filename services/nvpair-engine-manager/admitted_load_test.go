// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestAdmittedLoadRelayReturnsBrokerResult(t *testing.T) {
	r := &admittedLoadRelay{}
	r.send = func(_ context.Context, method string, value any) error {
		if method != admittedLoadRequestMethod {
			t.Fatalf("method=%s", method)
		}
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		var request struct {
			ID     string             `json:"id"`
			Params modelActionRequest `json:"params"`
		}
		if err := json.Unmarshal(raw, &request); err != nil {
			t.Fatal(err)
		}
		if request.Params.Engine != "fake" || request.Params.Model != "model" {
			t.Fatalf("params=%+v", request.Params)
		}
		reply, err := json.Marshal(admittedLoadReply{ID: request.ID, Result: json.RawMessage(`{"loaded":true}`)})
		if err != nil {
			t.Fatal(err)
		}
		r.reply(reply)
		return nil
	}
	raw, err := r.call(context.Background(), "fake", "model")
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Loaded bool `json:"loaded"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		t.Fatal(err)
	}
	if !result.Loaded || len(r.pending) != 0 {
		t.Fatalf("result=%+v pending=%v", result, r.pending)
	}
}

func TestAdmittedLoadRelayCancellationReleasesWaiter(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	r := &admittedLoadRelay{send: func(context.Context, string, any) error { cancel(); return nil }}
	_, err := r.call(ctx, "fake", "model")
	if !errors.Is(err, context.Canceled) || len(r.pending) != 0 {
		t.Fatalf("error=%v pending=%v", err, r.pending)
	}
}
