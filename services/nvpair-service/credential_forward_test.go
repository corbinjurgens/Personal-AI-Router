// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
)

// This fake broker deliberately answers credential pairs in reverse order.
// It exercises the real private service transport and ID remapping, while the
// broker's credential semantics remain independently tested at their owner.
func runCredentialForwardBroker(t *testing.T, b *fakeBroker) {
	t.Helper()
	defer b.exit(0)
	b.write(`{"jsonrpc":"2.0","method":"app:ready","params":{"version":"credential-fixture"}}`)
	var pending *jsonrpc.Message
	answer := func(message *jsonrpc.Message) {
		var params engines.SetCredentialParams
		if err := json.Unmarshal(message.Params, &params); err != nil {
			t.Error("decode private credential request")
			return
		}
		if params.Engine == "unavailable" {
			b.write(`{"jsonrpc":"2.0","id":%s,"error":{"code":-32000,"message":"Credential store is unavailable. Unlock the operating system credential store and try again."}}`, *message.ID)
			return
		}
		b.write(`{"jsonrpc":"2.0","id":%s,"result":{"supported":true,"configured":%t}}`, *message.ID, message.Method == engines.SetCredentialMethod && params.Credential != "")
	}
	scanner := bufio.NewScanner(b.stdinR)
	for scanner.Scan() {
		var message jsonrpc.Message
		if err := json.Unmarshal(scanner.Bytes(), &message); err != nil {
			t.Error("decode service frame")
			return
		}
		b.mu.Lock()
		b.frames = append(b.frames, message)
		b.mu.Unlock()
		if message.ID == nil {
			continue
		}
		switch message.Method {
		case engines.SetCredentialMethod, engines.CredentialStateMethod:
			if pending == nil {
				pending = &message
				continue
			}
			answer(&message)
			answer(pending)
			pending = nil
		case "shutdown":
			b.write(`{"jsonrpc":"2.0","id":%s,"result":null}`, *message.ID)
			return
		default:
			b.write(`{"jsonrpc":"2.0","id":%s,"result":null}`, *message.ID)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Error("read service frame")
	}
}

func TestCredentialForwardPrivateCorrelationAndNoReplay(t *testing.T) {
	const key = "test-key-123"
	h := newHarness(t, func(h *harness, cfg *config) {
		cfg.spawn = func() (*brokerProc, error) {
			b := newFakeBroker(1, false)
			go runCredentialForwardBroker(t, b)
			h.brokers <- b
			return b.proc(), nil
		}
	})
	broker := h.broker()
	one, two := h.dial(), h.dial()
	one.notification("app:ready")
	two.notification("app:ready")
	one.request("7", engines.SetCredentialMethod, `{"engine":"lmstudio","credential":"test-key-123"}`)
	two.request("7", engines.CredentialStateMethod, `{"engine":"lmstudio"}`)
	first, second := one.response("7"), two.response("7")
	var firstState, secondState engines.CredentialState
	if first.Error != nil || second.Error != nil || json.Unmarshal(first.Result, &firstState) != nil || json.Unmarshal(second.Result, &secondState) != nil || !firstState.Supported || !firstState.Configured || !secondState.Supported || secondState.Configured {
		t.Fatal("private credential responses crossed clients or lost sanitized state")
	}
	ids := append(broker.receivedIDs(engines.SetCredentialMethod), broker.receivedIDs(engines.CredentialStateMethod)...)
	if len(ids) != 2 || ids[0] == ids[1] {
		t.Fatal("service did not remap private client IDs")
	}
	// The next pair exercises an unavailable result alongside a successful
	// clear, again with identical client IDs and reverse broker completion.
	one.request("8", engines.SetCredentialMethod, `{"engine":"lmstudio","credential":""}`)
	two.request("8", engines.CredentialStateMethod, `{"engine":"unavailable"}`)
	clearReply, unavailable := one.response("8"), two.response("8")
	var cleared engines.CredentialState
	if clearReply.Error != nil || json.Unmarshal(clearReply.Result, &cleared) != nil || !cleared.Supported || cleared.Configured {
		t.Fatal("clear reply was not private sanitized state")
	}
	if unavailable.Error == nil || unavailable.Error.Code != -32000 || unavailable.Error.Message != "Credential store is unavailable. Unlock the operating system credential store and try again." || len(unavailable.Result) != 0 {
		t.Fatal("unavailable reply crossed clients or claimed absence")
	}
	for _, message := range []*jsonrpc.Message{first, second, clearReply, unavailable} {
		data, err := json.Marshal(message)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(data, []byte(key)) {
			t.Fatal("service reply exposed key")
		}
	}
	late := h.dial()
	late.notification("app:ready")
	for _, client := range []*testClient{one, two, late} {
		client.noMessage(50*time.Millisecond, func(message *jsonrpc.Message) bool {
			data, err := json.Marshal(message)
			if err != nil {
				t.Error("encode client frame")
				return true
			}
			return message.ID != nil || message.Method == engines.SetCredentialMethod || message.Method == engines.CredentialStateMethod || bytes.Contains(data, []byte(key)) || bytes.Contains(data, []byte("configured"))
		})
	}
	logData, err := os.ReadFile(filepath.Join(h.dir, "logs", "broker.log"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(logData, []byte(key)) || bytes.Contains(logData, []byte("credential")) {
		t.Fatal("private request body was logged")
	}
}
