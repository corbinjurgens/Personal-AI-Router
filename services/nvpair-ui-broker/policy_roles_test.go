// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
)

// pausedLocallyDocument is routedPolicyDocument saved while paused on the node
// itself.
var pausedLocallyDocument = strings.Replace(routedPolicyDocument, `"revision": "12",`,
	`"revision": "12", "availability": "paused", "pausedBy": "local",`, 1)

func TestPeerPolicySetRoundTripsRoles(t *testing.T) {
	rb := newRoutedBroker(t)
	beforeRouting := mustJSON(rb.b.localPolicy().Policy.Routing)
	beforeTiers := mustJSON(rb.b.localPolicy().Policy.Tiers)

	doc := rb.peerSnapshot(t)
	if roles, ok := doc["roles"].(map[string]any); !ok || len(roles) != 0 {
		t.Fatalf("relayed get roles = %#v, want an empty object", doc["roles"])
	}
	doc["roles"] = map[string]any{
		"weak":   map[string]any{"engine": "ollama", "model": "qwen3:8b"},
		"strong": map[string]any{"engine": "lmstudio", "model": "big"},
	}
	if _, errText := rb.relaySet(t, doc); errText != "" {
		t.Fatalf("relayed set with roles: %s", errText)
	}

	want := map[string]nodepolicy.Role{
		nodepolicy.TierWeak:   {Engine: "ollama", Model: "qwen3:8b"},
		nodepolicy.TierStrong: {Engine: "lmstudio", Model: "big"},
	}
	if got := rb.b.localPolicy().Policy.Roles; !reflect.DeepEqual(got, want) {
		t.Errorf("stored roles = %+v, want %+v", got, want)
	}
	sections := rb.fileSections(t)
	var fileRoles map[string]nodepolicy.Role
	if err := json.Unmarshal(sections["roles"], &fileRoles); err != nil || !reflect.DeepEqual(fileRoles, want) {
		t.Errorf("policy file roles = %s (%v), want %+v", sections["roles"], err, want)
	}
	var gotRouting, gotTiers bytes.Buffer
	if err := json.Compact(&gotRouting, sections["routing"]); err != nil {
		t.Fatalf("compact routing: %v", err)
	}
	if err := json.Compact(&gotTiers, sections["tiers"]); err != nil {
		t.Fatalf("compact tiers: %v", err)
	}
	if gotRouting.String() != string(beforeRouting) || gotTiers.String() != string(beforeTiers) {
		t.Errorf("a roles edit changed routing or tiers:\n routing %s\n tiers %s", gotRouting.String(), gotTiers.String())
	}

	result, errText := rb.relay(t, "get", map[string]any{})
	if errText != "" {
		t.Fatalf("relayed get: %s", errText)
	}
	var got struct {
		Policy nodepolicy.PeerPolicy `json:"policy"`
	}
	if err := json.Unmarshal(result, &got); err != nil {
		t.Fatalf("decode relayed get: %v", err)
	}
	if !reflect.DeepEqual(got.Policy.Roles, want) {
		t.Errorf("relayed get roles = %+v, want %+v", got.Policy.Roles, want)
	}
}

func TestPeerPolicySetRejectsInvalidRolesWithoutEffect(t *testing.T) {
	test := func(name string, roles map[string]any, wantErr string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			doc := rb.peerSnapshot(t)
			doc["roles"] = roles
			base := rb.baseline(t)
			reply := rb.relayReply(t, "set", map[string]any{"policy": doc, "expectedPersistedRevision": doc["revision"]})
			if reply.ErrorCode != -32602 || !strings.Contains(reply.Error, wantErr) {
				t.Fatalf("code %d, error %q; want -32602 containing %q", reply.ErrorCode, reply.Error, wantErr)
			}
			base.requireUnchanged(t)
		})
	}

	test("unknown tier", map[string]any{"huge": map[string]any{"engine": "ollama", "model": "m"}}, "unknown tier")
	test("reserved alias model", map[string]any{"weak": map[string]any{"engine": "ollama", "model": "weak-api"}}, "reserved alias")
	test("unknown engine", map[string]any{"weak": map[string]any{"engine": "no/pe", "model": "m"}}, "unknown engine")
}

// A peer reads who paused the node, but every policy write is fenced while paused.
func TestPeerPolicySetWhilePausedHasNoEffect(t *testing.T) {
	test := func(name string, edit func(doc map[string]any)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			rb := newBrokerOnFile(t, pausedLocallyDocument)
			if got := rb.b.localPolicy().Policy; got.Availability != nodepolicy.Paused || got.PausedBy != nodepolicy.PausedLocally {
				t.Fatalf("loaded availability/pausedBy = %q/%q, want paused/local (was the fixture rejected?)", got.Availability, got.PausedBy)
			}
			doc := rb.peerSnapshot(t)
			if doc["pausedBy"] != "local" {
				t.Fatalf("relayed get pausedBy = %#v, want \"local\"", doc["pausedBy"])
			}
			edit(doc)
			before := rb.baseline(t)
			reply := rb.relayReply(t, "set", map[string]any{"policy": doc, "expectedPersistedRevision": doc["revision"]})
			if reply.ErrorCode != codeLocalVeto || reply.Error != "This node is paused. Resume it on this node before managing it remotely." {
				t.Fatalf("relayed set = code %d %q, want the pause refusal", reply.ErrorCode, reply.Error)
			}
			before.requireUnchanged(t)
		})
	}

	test("provenance changed to peer", func(doc map[string]any) { doc["pausedBy"] = "peer" })
	test("resumed with provenance cleared", func(doc map[string]any) {
		doc["availability"] = "available"
		delete(doc, "pausedBy")
	})
	test("roles edit alongside", func(doc map[string]any) {
		doc["roles"] = map[string]any{"medium": map[string]any{"engine": "ollama", "model": "qwen3:8b"}}
		doc["pausedBy"] = "peer"
	})
}

func TestLocalRolesEditWhilePausedPreservesPause(t *testing.T) {
	rb := newBrokerOnFile(t, pausedLocallyDocument)
	before := rb.b.localPolicy().Policy.Revision
	rb.localSet(t, func(doc map[string]any) {
		doc["roles"] = map[string]any{"medium": map[string]any{"engine": "ollama", "model": "qwen3:8b"}}
	})
	got := rb.b.localPolicy().Policy
	want := map[string]nodepolicy.Role{nodepolicy.TierMedium: {Engine: "ollama", Model: "qwen3:8b"}}
	if !reflect.DeepEqual(got.Roles, want) || got.Revision == before || got.Availability != nodepolicy.Paused || got.PausedBy != nodepolicy.PausedLocally {
		t.Fatalf("local roles edit = %+v, want new revision and roles with the local pause preserved", got)
	}
	var saved map[string]nodepolicy.Role
	if err := json.Unmarshal(rb.fileSections(t)["roles"], &saved); err != nil {
		t.Fatalf("decode saved roles: %v", err)
	}
	if !reflect.DeepEqual(saved, want) {
		t.Fatalf("saved roles = %+v, want %+v", saved, want)
	}
}
