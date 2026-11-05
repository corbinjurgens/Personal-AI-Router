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
)

func identityPublicationNode(t *testing.T, host string, modelCount int) noderec.DirectoryNode {
	t.Helper()
	identity := engines.ModelIdentity{Strength: "declared"}
	for range 64 {
		identity.Files = append(identity.Files, engines.ModelIdentityFile{Role: "model", Name: strings.Repeat("a", 240) + ".gguf", SizeBytes: 42})
	}
	if err := identity.Validate(); err != nil {
		t.Fatalf("invalid specimen identity: %v", err)
	}
	node := noderec.DirectoryNode{HostUUID: host, Name: host, IP: "127.0.0.1",
		Services:       map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceEngineManager: {Port: 14317}},
		ModelsByEngine: map[string][]string{"llamacpp": {}}, ModelIdentities: map[string]map[string]engines.ModelIdentity{"llamacpp": {}}}
	for i := range modelCount {
		name := fmt.Sprintf("model-%d", i)
		node.Models = append(node.Models, name)
		node.ModelsByEngine["llamacpp"] = append(node.ModelsByEngine["llamacpp"], name)
		node.ModelIdentities["llamacpp"][name] = identity
	}
	return node
}

func TestIdentityPublicationKeepsDiscoveryStreamReadable(t *testing.T) {
	node := identityPublicationNode(t, "peer", 64)
	var unbounded bytes.Buffer
	if err := jsonrpc.NewCodec(&unbounded).Notify(noderec.NotifyNodeUpdated, noderec.NodeEvent{Node: node}); err != nil {
		t.Fatalf("encode oversized specimen: %v", err)
	}
	if unbounded.Len() <= jsonrpc.DefaultFrameBytes || unbounded.Len() >= modelsResponseBytes {
		t.Fatalf("specimen must fit HTTP but exceed discovery: %d bytes", unbounded.Len())
	}
	if _, err := jsonrpc.NewCodec(&unbounded).Read(); err == nil {
		t.Fatal("ordinary codec accepted oversized specimen")
	}
	var wire bytes.Buffer
	d := &daemon{codec: NewCodec(&wire)}
	d.emit(noderec.NotifyNodeUpdated, node)
	if err := d.codec.Notify("next-event", map[string]bool{"alive": true}); err != nil {
		t.Fatalf("write following event: %v", err)
	}
	reader := jsonrpc.NewCodec(&wire)
	msg, err := reader.Read()
	if err != nil {
		t.Fatalf("read bounded discovery event: %v", err)
	}
	var event noderec.NodeEvent
	if err := json.Unmarshal(msg.Params, &event); err != nil {
		t.Fatalf("decode bounded discovery event: %v", err)
	}
	if event.Node.ModelIdentities != nil || !reflect.DeepEqual(event.Node.ModelsByEngine, node.ModelsByEngine) || !reflect.DeepEqual(event.Node.Models, node.Models) {
		t.Fatal("oversized optional metadata displaced routing inventory")
	}
	if len(node.ModelIdentities["llamacpp"]) != 64 {
		t.Fatal("publication mutated source identities")
	}
	if next, err := reader.Read(); err != nil || next.Method != "next-event" {
		t.Fatalf("discovery stream did not survive: message=%+v error=%v", next, err)
	}
}

func TestIdentityPublicationBoundsAggregateResponse(t *testing.T) {
	var wire bytes.Buffer
	d := &daemon{codec: NewCodec(&wire), dir: newDirectory()}
	for i := range 64 {
		d.dir.upsert(identityPublicationNode(t, fmt.Sprintf("peer-%d", i), 1))
	}
	id := json.RawMessage(`"inventory-request"`)
	if !d.handle(&Message{JSONRPC: "2.0", ID: &id, Method: noderec.MethodGetNodes}) {
		t.Fatal("inventory request was not handled")
	}
	msg, err := jsonrpc.NewCodec(&wire).Read()
	if err != nil {
		t.Fatalf("read aggregate response: %v", err)
	}
	var result noderec.GetNodesResult
	if err := json.Unmarshal(msg.Result, &result); err != nil {
		t.Fatalf("decode aggregate response: %v", err)
	}
	if len(result.Nodes) != 64 {
		t.Fatalf("routing nodes lost: got %d", len(result.Nodes))
	}
	for _, node := range result.Nodes {
		if node.ModelIdentities != nil || !reflect.DeepEqual(node.ModelsByEngine, map[string][]string{"llamacpp": {"model-0"}}) {
			t.Fatal("aggregate identity budget changed routing inventory")
		}
	}
	if len(d.dir.snapshot("")[0].ModelIdentities) == 0 {
		t.Fatal("response budget mutated stored directory")
	}
}
