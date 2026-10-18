// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestAdvertisementTakesBoundsFromProfiles(t *testing.T) {
	p := stagedPolicy(t)
	p.Profiles = []Profile{{
		Name: "qwen", Engine: "ollama", Model: "qwen3:8b", ContextTokens: 16384,
		Capabilities: []Capability{CapTools, CapVision}, MemoryBytes: 5000000000, MaxConcurrent: 2,
		RequestOptions: json.RawMessage(`{"keep_alive":"10m"}`),
	}}
	p.Roles = map[string]Role{
		TierWeak:   {Engine: "ollama", Model: "qwen3:8b"},
		TierMedium: {Engine: "ollama", Model: "qwen3:8b"},
		TierStrong: {Engine: "lmstudio", Model: "qwen3:8b"},
	}
	p.Availability = Paused
	p.PausedBy = PausedByPeer
	if err := p.Validate(); err != nil {
		t.Fatalf("fixture policy is invalid: %v", err)
	}

	got := p.Advertisement()
	bounded := AdvertisedRole{Engine: "ollama", Model: "qwen3:8b", ContextTokens: 16384, Capabilities: []Capability{CapTools, CapVision}}
	want := Advertisement{
		Revision: "12",
		Roles: map[string]AdvertisedRole{
			TierWeak:   bounded,
			TierMedium: bounded,
			// The only profile covers the same model on another engine.
			TierStrong: {Engine: "lmstudio", Model: "qwen3:8b"},
		},
		PausedBy: PausedByPeer,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Advertisement() = %+v\nwant %+v", got, want)
	}

	got.Roles[TierWeak].Capabilities[0] = CapEmbeddings
	if p.Profiles[0].Capabilities[0] != CapTools {
		t.Errorf("editing the advertisement changed the policy's profile")
	}

	data, err := json.Marshal(p.Advertisement())
	if err != nil {
		t.Fatalf("marshal advertisement: %v", err)
	}
	for _, leak := range []string{
		"routing", "NVPAIR_PROVIDER_SECRET", "free-acct", "free-small", "deployment", "allowedOrigins",
		"tiers", "profiles", "memoryBytes", "maxConcurrent", "keep_alive", "availability", "admission",
	} {
		if strings.Contains(string(data), leak) {
			t.Errorf("advertisement contains %q: %s", leak, data)
		}
	}
}

func TestAdvertisementWithoutProfileHasNoBounds(t *testing.T) {
	p := Default()
	p.Roles = map[string]Role{TierWeak: {Engine: "ollama", Model: "m"}}

	got := p.Advertisement()
	want := Advertisement{Revision: "0", Roles: map[string]AdvertisedRole{TierWeak: {Engine: "ollama", Model: "m"}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Advertisement() = %+v, want %+v", got, want)
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal advertisement: %v", err)
	}
	if want := `{"revision":"0","roles":{"weak":{"engine":"ollama","model":"m"}}}`; string(data) != want {
		t.Errorf("advertisement JSON = %s, want %s", data, want)
	}
}

func TestAdvertisementWithoutRoles(t *testing.T) {
	data, err := json.Marshal(Default().Advertisement())
	if err != nil {
		t.Fatalf("marshal advertisement: %v", err)
	}
	if want := `{"revision":"0","roles":{}}`; string(data) != want {
		t.Errorf("advertisement JSON = %s, want %s", data, want)
	}
}
