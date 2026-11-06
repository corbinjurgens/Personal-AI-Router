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
	if roles, ok := doc["roles"].([]any); !ok || len(roles) != 0 {
		t.Fatalf("relayed get roles = %#v, want an empty list", doc["roles"])
	}
	doc["profiles"] = []nodepolicy.Profile{
		{Key: "small-profile", DisplayName: "Small", Engine: "llamacpp", Model: "qwen3:8b"},
		{Key: "large-profile", DisplayName: "Large", Engine: "llamacpp-custom", Model: "big"},
	}
	doc["roles"] = []nodepolicy.Role{
		{Name: nodepolicy.TierStrong, Profiles: []string{"large-profile", "small-profile"}},
		{Name: nodepolicy.TierWeak, Profiles: []string{"small-profile"}},
		{Name: nodepolicy.TierMedium, Profiles: []string{}},
	}
	if _, errText := rb.relaySet(t, doc); errText != "" {
		t.Fatalf("relayed set with roles: %s", errText)
	}

	want := []nodepolicy.Role{
		{Name: nodepolicy.TierStrong, Profiles: []string{"large-profile", "small-profile"}},
		{Name: nodepolicy.TierWeak, Profiles: []string{"small-profile"}},
		{Name: nodepolicy.TierMedium, Profiles: []string{}},
	}
	if got := rb.b.localPolicy().Policy.Roles; !reflect.DeepEqual(got, want) {
		t.Errorf("stored roles = %+v, want %+v", got, want)
	}
	sections := rb.fileSections(t)
	var fileRoles []nodepolicy.Role
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
	test := func(name string, edit func(map[string]any), wantErr string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			rb := newRoutedBroker(t)
			doc := rb.peerSnapshot(t)
			doc["profiles"] = []nodepolicy.Profile{{Key: "fixture-profile", DisplayName: "Fixture", Engine: "llamacpp", Model: "m"}}
			doc["roles"] = []nodepolicy.Role{{Name: nodepolicy.TierWeak, Profiles: []string{"fixture-profile"}}}
			edit(doc)
			base := rb.baseline(t)
			reply := rb.relayReply(t, "set", map[string]any{"policy": doc, "expectedPersistedRevision": doc["revision"]})
			if reply.ErrorCode != -32602 || !strings.Contains(reply.Error, wantErr) {
				t.Fatalf("code %d, error %q; want -32602 containing %q", reply.ErrorCode, reply.Error, wantErr)
			}
			base.requireUnchanged(t)
		})
	}

	test("unknown tier", func(doc map[string]any) {
		doc["roles"] = []nodepolicy.Role{{Name: "huge", Profiles: []string{"fixture-profile"}}}
	}, "unknown tier")
	test("unknown profile", func(doc map[string]any) {
		doc["roles"] = []nodepolicy.Role{{Name: nodepolicy.TierWeak, Profiles: []string{"missing-profile"}}}
	}, "unknown profile key")
	test("reserved alias model", func(doc map[string]any) {
		doc["profiles"].([]nodepolicy.Profile)[0].Model = "weak-api"
	}, "reserved alias")
	test("unknown engine", func(doc map[string]any) {
		doc["profiles"].([]nodepolicy.Profile)[0].Engine = "no/pe"
	}, "unknown engine")
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
		doc["roles"] = []nodepolicy.Role{{Name: nodepolicy.TierMedium, Profiles: []string{}}}
		doc["pausedBy"] = "peer"
	})
}

func TestLocalRolesEditWhilePausedPreservesPause(t *testing.T) {
	rb := newBrokerOnFile(t, pausedLocallyDocument)
	before := rb.b.localPolicy().Policy.Revision
	rb.localSet(t, func(doc map[string]any) {
		doc["profiles"] = []nodepolicy.Profile{{Key: "small-profile", DisplayName: "Small", Engine: "llamacpp", Model: "qwen3:8b"}}
		doc["roles"] = []nodepolicy.Role{{Name: nodepolicy.TierMedium, Profiles: []string{"small-profile"}}}
	})
	got := rb.b.localPolicy().Policy
	want := []nodepolicy.Role{{Name: nodepolicy.TierMedium, Profiles: []string{"small-profile"}}}
	if !reflect.DeepEqual(got.Roles, want) || got.Revision == before || got.Availability != nodepolicy.Paused || got.PausedBy != nodepolicy.PausedLocally {
		t.Fatalf("local roles edit = %+v, want new revision and roles with the local pause preserved", got)
	}
	var saved []nodepolicy.Role
	if err := json.Unmarshal(rb.fileSections(t)["roles"], &saved); err != nil {
		t.Fatalf("decode saved roles: %v", err)
	}
	if !reflect.DeepEqual(saved, want) {
		t.Fatalf("saved roles = %+v, want %+v", saved, want)
	}
}
