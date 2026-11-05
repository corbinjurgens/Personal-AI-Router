// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package jsonrpc

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"
)

func TestFrameBudgetUsesActualEnvelope(t *testing.T) {
	test := func(name string, response bool) {
		t.Run(name, func(t *testing.T) {
			var wire bytes.Buffer
			codec := NewCodec(&wire)
			payload := map[string]string{"model": "model with \"quotes\" and é"}
			id := json.RawMessage(`"correlation-with-escaped-\"quote"`)
			var err error
			var fits func(int) bool
			if response {
				err = codec.Respond(&id, payload)
				fits = func(limit int) bool { return ResponseFitsFrame(&id, payload, limit) }
			} else {
				err = codec.Notify("discovery:nodes", payload)
				fits = func(limit int) bool { return NotificationFitsFrame("discovery:nodes", payload, limit) }
			}
			if err != nil {
				t.Fatalf("encode real frame: %v", err)
			}
			if !fits(wire.Len()+1) || fits(wire.Len()) {
				t.Fatalf("frame budget disagrees with encoded envelope (%d bytes)", wire.Len())
			}
			if _, err := NewCodec(&wire).Read(); err != nil {
				t.Fatalf("read fitting frame: %v", err)
			}
		})
	}
	test("notification", false)
	test("response includes actual id", true)
}

func TestFrameBudgetRejectsUnencodablePayload(t *testing.T) {
	if NotificationFitsFrame("event", math.NaN(), DefaultFrameBytes) || ResponseFitsFrame(nil, math.NaN(), DefaultFrameBytes) {
		t.Fatal("unencodable payload was reported as fitting")
	}
}
