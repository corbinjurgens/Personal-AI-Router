// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDefaultIsValid(t *testing.T) {
	if err := Default().Validate(); err != nil {
		t.Fatalf("Default().Validate() = %v", err)
	}
}

func TestParseFillsOmittedFieldsFromDefault(t *testing.T) {
	p, err := Parse([]byte(`{"version":1,"availability":"paused"}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if p.Availability != Paused {
		t.Fatalf("availability = %q", p.Availability)
	}
	if p.Admission.MaxResidentModels != 1 || !p.Idle.StartOnDemand || p.Tiers == nil || p.Profiles == nil {
		t.Fatalf("defaults not applied: %+v", p)
	}
}

func TestValidateRejects(t *testing.T) {
	cases := map[string]func(*Policy){
		"draining saved": func(p *Policy) { p.Availability = Draining },
		"unknown engine": func(p *Policy) { p.Profiles = []Profile{{Name: "a", Engine: "nope", Model: "m"}} },
		"duplicate profile": func(p *Policy) {
			p.Profiles = []Profile{{Name: "a", Engine: "ollama", Model: "m"}, {Name: "a", Engine: "ollama", Model: "n"}}
		},
		"two profiles model": func(p *Policy) {
			p.Profiles = []Profile{{Name: "a", Engine: "ollama", Model: "m"}, {Name: "b", Engine: "ollama", Model: "m"}}
		},
		"unknown tier":  func(p *Policy) { p.Tiers = map[string][]TierEntry{"huge": {{Engine: "ollama", Model: "m"}}} },
		"tier as model": func(p *Policy) { p.Tiers = map[string][]TierEntry{"weak": {{Engine: "ollama", Model: "strong"}}} },
		"unknown capability": func(p *Policy) {
			p.Tiers = map[string][]TierEntry{"weak": {{Engine: "ollama", Model: "m", Capabilities: []Capability{"telepathy"}}}}
		},
		"options not object": func(p *Policy) {
			p.Profiles = []Profile{{Name: "a", Engine: "ollama", Model: "m", RequestOptions: json.RawMessage(`[1]`)}}
		},
		"options set model": func(p *Policy) {
			p.Profiles = []Profile{{Name: "a", Engine: "ollama", Model: "m", RequestOptions: json.RawMessage(`{"model":"x"}`)}}
		},
		"negative budget": func(p *Policy) { p.Admission.MemoryBudgetBytes = -1 },
		"bad preference":  func(p *Policy) { p.TierPolicy.PreferLoaded = "sometimes" },
	}
	for name, mutate := range cases {
		p := Default()
		mutate(&p)
		if err := p.Validate(); err == nil {
			t.Errorf("%s: Validate() = nil, want an error", name)
		}
	}
}

func TestProfileForAndIsTier(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{{Name: "q", Engine: "ollama", Model: "qwen3:8b", MaxConcurrent: 2}}
	if pr, ok := p.ProfileFor("ollama", "qwen3:8b"); !ok || pr.MaxConcurrent != 2 {
		t.Fatalf("ProfileFor = %+v, %v", pr, ok)
	}
	if _, ok := p.ProfileFor("lmstudio", "qwen3:8b"); ok {
		t.Fatal("ProfileFor matched the wrong engine")
	}
	if !IsTier("medium") || IsTier("qwen3:8b") {
		t.Fatal("IsTier misclassified")
	}
}

func TestParseRejectsGarbage(t *testing.T) {
	if _, err := Parse([]byte(`{`)); err == nil || !strings.Contains(err.Error(), "parse policy") {
		t.Fatalf("Parse(garbage) err = %v", err)
	}
}
