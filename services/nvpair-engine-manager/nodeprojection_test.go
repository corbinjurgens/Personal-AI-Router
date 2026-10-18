// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"

	"net/http/httptest"
	"reflect"
	"testing"

	"nvpair-shared/nodepolicy"
)

// servedPolicy reads the policy object from both surfaces that serve the model
// inventory: GET /v1/models and the engine:models RPC. The HasKey results
// report whether each document carried a policy key at all.
func servedPolicy(t *testing.T, m *Manager, out *bytes.Buffer) (fromHTTP, fromRPC *nodepolicy.Advertisement, httpHasKey, rpcHasKey bool) {
	t.Helper()
	srv := httptest.NewServer(modelsHandler(m.exec))
	defer srv.Close()
	resp, err := srv.Client().Get(srv.URL + modelsPath)
	if err != nil {
		t.Fatalf("GET %s: %v", modelsPath, err)
	}
	defer resp.Body.Close()
	var doc map[string]json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		t.Fatalf("decode %s: %v", modelsPath, err)
	}
	fromHTTP, httpHasKey = decodePolicy(t, doc)

	out.Reset()
	id := json.RawMessage("7")
	m.runModels(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:models"})
	var reply struct {
		Result map[string]json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(out.Bytes(), &reply); err != nil {
		t.Fatalf("decode engine:models reply %q: %v", out.String(), err)
	}
	fromRPC, rpcHasKey = decodePolicy(t, reply.Result)
	return fromHTTP, fromRPC, httpHasKey, rpcHasKey
}

func decodePolicy(t *testing.T, doc map[string]json.RawMessage) (*nodepolicy.Advertisement, bool) {
	t.Helper()
	raw, ok := doc["policy"]
	if !ok {
		return nil, false
	}
	var adv nodepolicy.Advertisement
	if err := json.Unmarshal(raw, &adv); err != nil {
		t.Fatalf("decode policy %s: %v", raw, err)
	}
	return &adv, true
}

func pushProjection(t *testing.T, m *Manager, params string) {
	t.Helper()
	m.handleMessage(context.Background(), &Message{JSONRPC: "2.0", Method: nodeProjectionMethod, Params: json.RawMessage(params)})
}

func TestNodeProjectionServedWithTheModelInventory(t *testing.T) {
	var out bytes.Buffer
	ex := NewExecutor(NewRegistry(), NewReporter(nil), func(string, any) {}, t.TempDir())
	m := NewManager(NewCodec(&out), ex, nil, "")

	t.Run("absent before the first push", func(t *testing.T) {
		_, _, httpHasKey, rpcHasKey := servedPolicy(t, m, &out)
		if httpHasKey || rpcHasKey {
			t.Fatalf("policy served before any push: /v1/models %v, engine:models %v", httpHasKey, rpcHasKey)
		}
	})

	first := nodepolicy.Advertisement{
		Revision: "12",
		Roles: map[string]nodepolicy.AdvertisedRole{
			nodepolicy.TierStrong: {Engine: "ollama", Model: "qwen3:32b", ContextTokens: 32768, Capabilities: []nodepolicy.Capability{nodepolicy.CapTools}},
		},
	}
	t.Run("served after a push", func(t *testing.T) {
		pushProjection(t, m, string(mustMarshal(t, map[string]any{"policy": first})))
		fromHTTP, fromRPC, _, _ := servedPolicy(t, m, &out)
		if !reflect.DeepEqual(fromHTTP, &first) || !reflect.DeepEqual(fromRPC, &first) {
			t.Fatalf("served policy = /v1/models %+v, engine:models %+v; want %+v", fromHTTP, fromRPC, first)
		}
	})

	t.Run("an unreadable push keeps the previous projection", func(t *testing.T) {
		pushProjection(t, m, `{"policy":"not an object"}`)
		if fromHTTP, _, _, _ := servedPolicy(t, m, &out); !reflect.DeepEqual(fromHTTP, &first) {
			t.Fatalf("served policy after an unreadable push = %+v, want %+v", fromHTTP, first)
		}
	})

	t.Run("a later push replaces it whole", func(t *testing.T) {
		paused := nodepolicy.Advertisement{Revision: "13", Roles: map[string]nodepolicy.AdvertisedRole{}, PausedBy: nodepolicy.PausedLocally}
		pushProjection(t, m, string(mustMarshal(t, map[string]any{"policy": paused})))
		if fromHTTP, fromRPC, _, _ := servedPolicy(t, m, &out); !reflect.DeepEqual(fromHTTP, &paused) || !reflect.DeepEqual(fromRPC, &paused) {
			t.Fatalf("served policy = /v1/models %+v, engine:models %+v; want %+v", fromHTTP, fromRPC, paused)
		}
	})
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
