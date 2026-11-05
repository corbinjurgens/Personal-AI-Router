// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/noderec"
	"nvpair-ui-broker/relay"
	"nvpair-ui-broker/workloadstore"
)

func TestDiscoveryClientIdentityAggregateFitsFrame(t *testing.T) {
	test := func(name string) {
		t.Run(name, func(t *testing.T) {
			store := newDiscoveryStore()
			identity := engines.ModelIdentity{Strength: "declared"}
			for range 64 {
				identity.Files = append(identity.Files, engines.ModelIdentityFile{Role: "model", Name: strings.Repeat("a", 240) + ".gguf", SizeBytes: 42})
			}
			if err := identity.Validate(); err != nil {
				t.Fatalf("invalid specimen: %v", err)
			}
			for i := range 64 {
				name := fmt.Sprintf("peer-%d", i)
				store.Upsert(EnrichedNode{ID: name, HostUUID: name, Host: "127.0.0.1",
					ModelsByEngine:  map[string][]string{"llamacpp": {"model"}},
					ModelIdentities: map[string]map[string]engines.ModelIdentity{"llamacpp": {"model": identity}}}, sourceScanner)
			}
			var wire bytes.Buffer
			b := &Broker{codec: NewCodec(&wire), store: store}
			response := name == "get-nodes response"
			if name == "scheduler startup baseline" {
				b.workloads = workloadstore.New()
				b.telemetry = newTelemetryCache()
				b.initializeScheduler(&rpcWorker{peer: NewPeer(NewCodec(&wire))})
			} else if response {
				id := json.RawMessage(`"client-inventory"`)
				b.handleMessage(&Message{JSONRPC: "2.0", ID: &id, Method: "discovery:get-nodes"})
			} else {
				b.pushNodesChanged()
			}
			reader := jsonrpc.NewCodec(&wire)
			msg, err := reader.Read()
			if err != nil {
				t.Fatalf("read discovery client publication: %v", err)
			}
			if !response && msg.Method != "discovery:nodes-changed" {
				t.Fatalf("discovery publication method = %q", msg.Method)
			}
			var nodes []AvailableNode
			if response {
				var result GetNodesResult
				if err := json.Unmarshal(msg.Result, &result); err != nil {
					t.Fatalf("decode response: %v", err)
				}
				nodes = result.Nodes
			} else if err := json.Unmarshal(msg.Params, &nodes); err != nil {
				t.Fatalf("decode notification: %v", err)
			}
			if len(nodes) != 64 {
				t.Fatalf("routing node count = %d", len(nodes))
			}
			for _, node := range nodes {
				if node.ModelIdentities != nil || !reflect.DeepEqual(node.ModelsByEngine, map[string][]string{"llamacpp": {"model"}}) {
					t.Fatal("optional metadata displaced routing inventory")
				}
			}
			if len(store.Snapshot()[0].ModelIdentities) == 0 {
				t.Fatal("publication mutated stored identity facts")
			}
		})
	}
	test("subscription notification")
	test("get-nodes response")
	test("scheduler startup baseline")
}

func TestAddressedProxyIdentityPublicationFitsFrame(t *testing.T) {
	const targetBytes = jsonrpc.DefaultFrameBytes - 5
	identity := engines.ModelIdentity{Strength: "declared"}
	for range 64 {
		identity.Files = append(identity.Files, engines.ModelIdentityFile{Role: "model", Name: strings.Repeat("a", 240) + ".gguf", SizeBytes: 42})
	}
	nodes := make([]noderec.DirectoryNode, 2)
	for i, count := range []int{28, 29} {
		nodes[i] = noderec.DirectoryNode{HostUUID: fmt.Sprintf("peer-%d", i),
			Services:       map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceLlamaCPP: {Port: 8080}},
			ModelsByEngine: map[string][]string{"llamacpp": {}}, ModelIdentities: map[string]map[string]engines.ModelIdentity{"llamacpp": {}}}
		for j := range count {
			name := fmt.Sprintf("model-%d", j)
			nodes[i].ModelsByEngine["llamacpp"] = append(nodes[i].ModelsByEngine["llamacpp"], name)
			nodes[i].ModelIdentities["llamacpp"][name] = identity
		}
	}
	frameSize := func() int {
		t.Helper()
		var wire bytes.Buffer
		if err := NewCodec(&wire).Notify(noderec.NotifyNodes, noderec.GetNodesResult{Nodes: nodes}); err != nil {
			t.Fatalf("encode specimen: %v", err)
		}
		return wire.Len()
	}
	last := nodes[1].ModelIdentities["llamacpp"]["model-28"]
	for frameSize() > targetBytes {
		last.Files = last.Files[:len(last.Files)-1]
		nodes[1].ModelIdentities["llamacpp"]["model-28"] = last
	}
	// Emptying the last basename leaves enough room to tune the specimen with
	// bounded optional text while all identities still pass Validate.
	last.Files = append([]engines.ModelIdentityFile(nil), last.Files...)
	last.Files[len(last.Files)-1].Name = ""
	nodes[1].ModelIdentities["llamacpp"]["model-28"] = last
	gap := targetBytes - frameSize()
	if gap <= 273 {
		last.GeneralName = strings.Repeat("b", gap-len(`,"generalName":""`))
	} else {
		textBytes := gap - len(`,"generalName":"","architecture":""`)
		last.GeneralName = strings.Repeat("b", 256)
		last.Architecture = strings.Repeat("c", textBytes-256)
	}
	if err := last.Validate(); err != nil {
		t.Fatalf("invalid boundary specimen: %v", err)
	}
	nodes[1].ModelIdentities["llamacpp"]["model-28"] = last
	method := engines.AddressMethod("llamacpp", noderec.NotifyNodes)
	if frameSize() != targetBytes || jsonrpc.NotificationFitsFrame(method, noderec.GetNodesResult{Nodes: nodes}, jsonrpc.DefaultFrameBytes) {
		t.Fatalf("specimen does not isolate addressed envelope boundary: ordinary bytes %d", frameSize())
	}
	dir := relay.NewDirectory()
	for _, node := range nodes {
		dir.Apply(noderec.NotifyNodeDiscovered, node)
	}
	var wire bytes.Buffer
	p := &proxyProcess{peer: NewPeer(NewCodec(&wire)), relayDir: dir}
	p.handleSubscribe("llamacpp", json.RawMessage(`{"services":["lc"]}`))
	msg, err := jsonrpc.NewCodec(&wire).Read()
	if err != nil {
		t.Fatalf("read addressed proxy snapshot: %v", err)
	}
	var result noderec.GetNodesResult
	if err := json.Unmarshal(msg.Params, &result); err != nil {
		t.Fatalf("decode addressed snapshot: %v", err)
	}
	if msg.Method != method || len(result.Nodes) != len(nodes) {
		t.Fatalf("addressed snapshot lost routing nodes: method %q nodes %d", msg.Method, len(result.Nodes))
	}
	for i, node := range result.Nodes {
		if node.ModelIdentities != nil || !reflect.DeepEqual(node.ModelsByEngine, nodes[i].ModelsByEngine) {
			t.Fatal("addressed frame budget displaced routing inventory")
		}
	}
	if len(dir.Snapshot("")[0].ModelIdentities) == 0 {
		t.Fatal("addressed publication mutated stored identity facts")
	}
}
