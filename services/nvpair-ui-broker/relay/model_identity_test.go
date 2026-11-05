// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package relay

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

func relayIdentityNode(host string) noderec.DirectoryNode {
	identity := engines.ModelIdentity{Strength: "declared"}
	for range 64 {
		identity.Files = append(identity.Files, engines.ModelIdentityFile{Role: "model", Name: strings.Repeat("a", 240) + ".gguf", SizeBytes: 42})
	}
	return noderec.DirectoryNode{HostUUID: host,
		ModelsByEngine:  map[string][]string{"llamacpp": {"model"}},
		ModelIdentities: map[string]map[string]engines.ModelIdentity{"llamacpp": {"model": identity}}}
}

func TestDirectoryIdentityCopiesDoNotMutateBank(t *testing.T) {
	test := func(name string, mutate func(*Directory, noderec.DirectoryNode)) {
		t.Run(name, func(t *testing.T) {
			d := NewDirectory()
			node := relayIdentityNode("peer")
			d.Apply(noderec.NotifyNodeDiscovered, node)
			mutate(d, node)
			got := d.nodes["peer"].ModelIdentities["llamacpp"]["model"]
			if len(d.nodes["peer"].ModelIdentities["llamacpp"]) != 1 || len(got.Files) != 64 || got.Files[0].Name != strings.Repeat("a", 240)+".gguf" {
				t.Fatal("external identity mutation reached the relay bank")
			}
		})
	}
	mutate := func(node noderec.DirectoryNode) {
		node.ModelIdentities["llamacpp"]["model"].Files[0].Name = "changed.gguf"
		delete(node.ModelIdentities["llamacpp"], "model")
	}
	test("input", func(_ *Directory, node noderec.DirectoryNode) { mutate(node) })
	test("snapshot", func(d *Directory, _ noderec.DirectoryNode) { mutate(d.Snapshot("")[0]) })
	test("subscriber", func(d *Directory, _ noderec.DirectoryNode) {
		sub := &Subscriber{Send: func(nodes []noderec.DirectoryNode) { mutate(nodes[0]) }}
		d.Subscribe(sub)
		d.Deliver(sub)
	})
}

func TestDirectoryIdentityAggregatePublicationFitsFrame(t *testing.T) {
	d := NewDirectory()
	for i := range 64 {
		d.Apply(noderec.NotifyNodeDiscovered, relayIdentityNode(fmt.Sprintf("peer-%d", i)))
	}
	var wire bytes.Buffer
	sub := &Subscriber{Send: func(nodes []noderec.DirectoryNode) {
		if err := jsonrpc.NewCodec(&wire).Notify(noderec.NotifyNodes, noderec.GetNodesResult{Nodes: nodes}); err != nil {
			t.Fatalf("write subscriber snapshot: %v", err)
		}
	}}
	d.Subscribe(sub)
	d.Deliver(sub)
	msg, err := jsonrpc.NewCodec(&wire).Read()
	if err != nil {
		t.Fatalf("read aggregate subscriber snapshot: %v", err)
	}
	var result noderec.GetNodesResult
	if err := json.Unmarshal(msg.Params, &result); err != nil {
		t.Fatalf("decode subscriber snapshot: %v", err)
	}
	if len(result.Nodes) != 64 {
		t.Fatalf("routing node count = %d", len(result.Nodes))
	}
	for _, node := range result.Nodes {
		if node.ModelIdentities != nil || !reflect.DeepEqual(node.ModelsByEngine, map[string][]string{"llamacpp": {"model"}}) {
			t.Fatal("metadata budgeting displaced routing inventory")
		}
	}
	if len(d.nodes["peer-0"].ModelIdentities) == 0 {
		t.Fatal("publication removed stored identities")
	}
}
