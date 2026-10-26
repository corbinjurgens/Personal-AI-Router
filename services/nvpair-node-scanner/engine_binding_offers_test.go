// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"reflect"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestEngineBindingOffersPinnedEnrichmentAndLastGood(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	raw := json.RawMessage(`{"revision":"1","roles":{},"engineBindingOffers":{"version":1,"models":[{"engine":"lmstudio","model":"cold-model"}]}}`)
	f.policy.Store(&raw)
	node := f.node
	f.d.enrich(&node)
	want := &nodepolicy.EngineBindingOffers{Version: 1, Models: []nodepolicy.Role{{Engine: "lmstudio", Model: "cold-model"}}}
	if node.Policy == nil || !reflect.DeepEqual(node.Policy.EngineBindingOffers, want) || !reflect.DeepEqual(node.EngineModels("ollama"), []string{"model"}) || len(node.EngineModels("lmstudio")) != 0 {
		t.Fatalf("offer changed inventory or was lost: %+v", node)
	}
	f.direct.Store(false)
	f.reverse.Store(false)
	node = f.node
	f.d.enrich(&node)
	if node.Policy == nil || !reflect.DeepEqual(node.Policy.EngineBindingOffers, want) || !reflect.DeepEqual(cachedPolicy(f.d, "peer").EngineBindingOffers, want) {
		t.Fatalf("failed pinned fetch discarded last good offers: %+v", node.Policy)
	}
}

func TestEngineBindingOffersMalformedDoesNotDiscardInventory(t *testing.T) {
	f := newScannerReverseFixture(t)
	f.direct.Store(true)
	raw := json.RawMessage(`{"revision":"2","roles":{},"engineBindingOffers":{"version":1,"models":"invalid"}}`)
	f.policy.Store(&raw)
	node := f.node
	f.d.enrich(&node)
	if node.Policy == nil || node.Policy.Revision != "2" || node.Policy.EngineBindingOffers != nil || !reflect.DeepEqual(node.Models, []string{"model"}) {
		t.Fatalf("malformed extension cost canonical inventory: %+v", node)
	}
}
