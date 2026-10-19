// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
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

// A descriptor engine ID is a policy identity like a built-in name: profiles,
// tiers and roles may name it whether or not this node runs it.
func TestValidateAcceptsDescriptorEngineIDs(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{{Name: "custom", Engine: "llamacpp-custom", Model: "org/model:Q4_K_M"}}
	p.Tiers = map[string][]TierEntry{"weak": {{Kind: EntryLocal, Engine: "strata", Model: "qwen-next"}}}
	p.Roles = map[string]Role{TierStrong: {Engine: "llamacpp-custom", Model: "org/model:Q4_K_M"}}
	if err := p.Validate(); err != nil {
		t.Fatalf("descriptor engine ids rejected: %v", err)
	}
}

func TestValidateRejects(t *testing.T) {
	cases := map[string]func(*Policy){
		"draining saved": func(p *Policy) { p.Availability = Draining },
		"unknown engine": func(p *Policy) { p.Profiles = []Profile{{Name: "a", Engine: "no/pe", Model: "m"}} },
		"overlong engine": func(p *Policy) {
			p.Profiles = []Profile{{Name: "a", Engine: strings.Repeat("e", engines.MaxDynamicIDLen+1), Model: "m"}}
		},
		"duplicate profile": func(p *Policy) {
			p.Profiles = []Profile{{Name: "a", Engine: "ollama", Model: "m"}, {Name: "a", Engine: "ollama", Model: "n"}}
		},
		"two profiles model": func(p *Policy) {
			p.Profiles = []Profile{{Name: "a", Engine: "ollama", Model: "m"}, {Name: "b", Engine: "ollama", Model: "m"}}
		},
		"unknown tier": func(p *Policy) {
			p.Tiers = map[string][]TierEntry{"huge": {{Kind: EntryLocal, Engine: "ollama", Model: "m"}}}
		},
		"tier as model": func(p *Policy) {
			p.Tiers = map[string][]TierEntry{"weak": {{Kind: EntryLocal, Engine: "ollama", Model: "strong"}}}
		},
		"unknown capability": func(p *Policy) {
			p.Tiers = map[string][]TierEntry{"weak": {{Kind: EntryLocal, Engine: "ollama", Model: "m", Capabilities: []Capability{"telepathy"}}}}
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

const v1Document = `{
  "version": 1,
  "availability": "paused",
  "pause": {"onActive": "cancel", "drainTimeoutSeconds": 7, "unloadModels": false, "stopEngines": true},
  "admission": {"maxResidentModels": 2, "maxConcurrentPerModel": 3, "queueTimeoutSeconds": 11,
    "memoryBudgetBytes": 8589934592, "switchModels": false, "switchTimeoutSeconds": 13},
  "idle": {"unloadAfterMinutes": 5, "stopEngineAfterMinutes": 9, "startOnDemand": false, "wakeTimeoutSeconds": 17},
  "profiles": [{"name": "qwen", "engine": "ollama", "model": "qwen3:8b", "quantization": "q4",
    "contextTokens": 16384, "capabilities": ["tools"], "memoryBytes": 5000000000, "maxConcurrent": 2,
    "requestOptions": {  "options": { "num_ctx": 16384 }  }}],
  "tiers": {
    "weak": [{"engine": "ollama", "model": "qwen3:8b", "contextTokens": 8192, "capabilities": ["tools", "structured"]}],
    "strong": [{"engine": "lmstudio", "model": "big"}, {"engine": "ollama", "model": "qwen3:8b"}]
  },
  "tierPolicy": {"allowStronger": false, "allowWeaker": true, "preferLoaded": "any"}
}`

func TestParseMigratesV1KeepingLocalSettings(t *testing.T) {
	got, err := Parse([]byte(v1Document))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	want := Policy{
		Version:      2,
		Revision:     "0",
		Availability: Paused,
		Pause:        PauseConfig{OnActive: CancelActive, DrainTimeoutSeconds: 7, UnloadModels: false, StopEngines: true},
		Admission: Admission{
			MaxResidentModels: 2, MaxConcurrentPerModel: 3, QueueTimeoutSeconds: 11,
			MemoryBudgetBytes: 8589934592, SwitchModels: false, SwitchTimeoutSeconds: 13,
		},
		Idle: IdleConfig{UnloadAfterMinutes: 5, StopEngineAfterMinutes: 9, StartOnDemand: false, WakeTimeoutSeconds: 17},
		Profiles: []Profile{{
			Name: "qwen", Engine: "ollama", Model: "qwen3:8b", Quantization: "q4", ContextTokens: 16384,
			Capabilities: []Capability{CapTools}, MemoryBytes: 5000000000, MaxConcurrent: 2,
			RequestOptions: json.RawMessage(`{"options":{"num_ctx":16384}}`),
		}},
		Roles: map[string]Role{},
		Tiers: map[string][]TierEntry{
			"weak": {{Kind: EntryLocal, Engine: "ollama", Model: "qwen3:8b", ContextTokens: 8192, Capabilities: []Capability{CapTools, CapStructured}}},
			"strong": {
				{Kind: EntryLocal, Engine: "lmstudio", Model: "big"},
				{Kind: EntryLocal, Engine: "ollama", Model: "qwen3:8b"},
			},
		},
		TierPolicy: TierPolicy{AllowStronger: false, AllowWeaker: true, PreferLoaded: "any"},
		Routing:    got.Routing,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("migrated policy = %+v\nwant %+v", got, want)
	}
}

func TestParseMigrationLeavesRemoteAndPaidOff(t *testing.T) {
	// A version 1 file that carries version 2 fields cannot start disclosing or
	// spending by upgrading.
	doc := `{"version":1,"revision":"41","routing":{
		"accounts":[{"id":"a","credential":"NVPAIR_PROVIDER_SECRET_A","billing":"paid",
			"perRequestCeiling":"1","dailyCeiling":"1","monthlyCeiling":"1"}],
		"tiers":{"weak":{"bareMode":"api","remoteAllowed":true,"paidAllowed":true}},
		"listener":{"enabled":true,"port":4000}}}`
	p, err := Parse([]byte(doc))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if p.Revision != "0" {
		t.Errorf("revision = %q, want 0", p.Revision)
	}
	if len(p.Routing.Accounts) != 0 || len(p.Routing.Deployments) != 0 {
		t.Errorf("accounts/deployments = %d/%d, want none", len(p.Routing.Accounts), len(p.Routing.Deployments))
	}
	if p.Routing.Listener.Enabled {
		t.Error("listener enabled by migration")
	}
	for _, tier := range []string{"weak", "medium", "strong"} {
		tr := p.Routing.Tiers[tier]
		if tr.RemoteAllowed || tr.PaidAllowed || tr.BareMode != BareHybrid {
			t.Errorf("tier %s = %+v, want hybrid with remote and paid off", tier, tr)
		}
	}
}

func TestParseIsIdempotent(t *testing.T) {
	inputs := map[string]string{
		"empty v2":           `{}`,
		"minimal v1":         `{"version":1}`,
		"full v1":            v1Document,
		"v1 with null tier":  `{"version":1,"tiers":{"weak":null,"medium":[]}}`,
		"v2 empty lists":     `{"version":2,"profiles":null,"tiers":{"weak":[{"kind":"local","engine":"ollama","model":"m","capabilities":[]}]}}`,
		"v2 spaced options":  `{"version":2,"profiles":[{"name":"a","engine":"ollama","model":"m","capabilities":[],"requestOptions": { "keep_alive" : "10m" }}]}`,
		"v2 with revision":   `{"version":2,"revision":"9007199254740993"}`,
		"v2 with api config": stagedRoutingDocument,
		"v1 html options":    `{"version":1,"profiles":[{"name":"a","engine":"ollama","model":"m","requestOptions":{"stop":["</s>"]}}]}`,
		"v2 html options":    `{"version":2,"profiles":[{"name":"a","engine":"ollama","model":"m","requestOptions":{"q":"a&b","c":"<x>"}}]}`,
		"v2 escaped options": `{"version":2,"profiles":[{"name":"a","engine":"ollama","model":"m","requestOptions":{"q":"a\u0026b \u003c"}}]}`,
		"v2 null roles":      `{"version":2,"roles":null}`,
		"v2 with roles":      `{"version":2,"roles":{"weak":{"engine":"ollama","model":"m"},"strong":{"engine":"lmstudio","model":"big"}}}`,
		"v2 paused by peer":  `{"version":2,"availability":"paused","pausedBy":"peer"}`,
	}
	marshalers := map[string]func(any) ([]byte, error){
		"Marshal":       json.Marshal,
		"MarshalIndent": func(v any) ([]byte, error) { return json.MarshalIndent(v, "", "  ") },
	}
	for name, doc := range inputs {
		for mname, marshal := range marshalers {
			t.Run(name+"/"+mname, func(t *testing.T) {
				first, err := Parse([]byte(doc))
				if err != nil {
					t.Fatalf("Parse: %v", err)
				}
				data, err := marshal(first)
				if err != nil {
					t.Fatalf("%s: %v", mname, err)
				}
				second, err := Parse(data)
				if err != nil {
					t.Fatalf("Parse(%s(Parse(x))): %v", mname, err)
				}
				if !reflect.DeepEqual(first, second) {
					t.Fatalf("not idempotent:\nfirst  %+v\nsecond %+v", first, second)
				}
			})
		}
	}
}

func TestParseMigrationKeepsFilesVersion1Accepted(t *testing.T) {
	t.Run("exact duplicate tier entry collapses to the first", func(t *testing.T) {
		doc := `{"version":1,"availability":"paused",
			"profiles":[{"name":"keep","engine":"ollama","model":"m"}],
			"tiers":{"weak":[
				{"engine":"ollama","model":"m","contextTokens":100},
				{"engine":"lmstudio","model":"m"},
				{"engine":"ollama","model":"m","contextTokens":100}]}}`
		p, err := Parse([]byte(doc))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		want := []TierEntry{
			{Kind: EntryLocal, Engine: "ollama", Model: "m", ContextTokens: 100},
			{Kind: EntryLocal, Engine: "lmstudio", Model: "m"},
		}
		if !reflect.DeepEqual(p.Tiers["weak"], want) {
			t.Errorf("weak entries = %+v, want %+v", p.Tiers["weak"], want)
		}
		if p.Availability != Paused || len(p.Profiles) != 1 || p.Profiles[0].Name != "keep" {
			t.Errorf("availability/profiles = %q/%+v, want paused with the profile kept", p.Availability, p.Profiles)
		}
	})
	t.Run("exact duplicate with reordered capabilities collapses", func(t *testing.T) {
		doc := `{"version":1,"tiers":{"weak":[
			{"engine":"ollama","model":"m","capabilities":["tools","structured"]},
			{"engine":"ollama","model":"m","capabilities":["structured","tools"]}]}}`
		p, err := Parse([]byte(doc))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		if len(p.Tiers["weak"]) != 1 {
			t.Errorf("weak entries = %+v, want one", p.Tiers["weak"])
		}
	})
	t.Run("entries differing in bounds stay separate", func(t *testing.T) {
		doc := `{"version":1,"tiers":{"weak":[
			{"engine":"ollama","model":"small:1b","contextTokens":100},
			{"engine":"ollama","model":"small:1b","contextTokens":32000,"capabilities":["tools"]}]}}`
		p, err := Parse([]byte(doc))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		want := []TierEntry{
			{Kind: EntryLocal, Engine: "ollama", Model: "small:1b", ContextTokens: 100},
			{Kind: EntryLocal, Engine: "ollama", Model: "small:1b", ContextTokens: 32000, Capabilities: []Capability{CapTools}},
		}
		if !reflect.DeepEqual(p.Tiers["weak"], want) {
			t.Errorf("weak entries = %+v, want %+v", p.Tiers["weak"], want)
		}
	})
	t.Run("profile named like an alias", func(t *testing.T) {
		p, err := Parse([]byte(`{"version":1,"profiles":[{"name":"weak","engine":"ollama","model":"m"}]}`))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		if len(p.Profiles) != 1 || p.Profiles[0].Name != "weak" {
			t.Errorf("profiles = %+v, want the profile named weak", p.Profiles)
		}
	})
}

func TestParseRejectsDuplicateTierEntryInV2(t *testing.T) {
	doc := `{"version":2,"tiers":{"weak":[
		{"kind":"local","engine":"ollama","model":"m"},{"kind":"local","engine":"ollama","model":"m"}]}}`
	if _, err := Parse([]byte(doc)); err == nil || !strings.Contains(err.Error(), "duplicate entry") {
		t.Fatalf("Parse(v2 duplicate) err = %v, want duplicate entry", err)
	}
}

func TestDefaultSurvivesJSONRoundTrip(t *testing.T) {
	data, err := json.Marshal(Default())
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Policy
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !reflect.DeepEqual(got, Default()) {
		t.Fatalf("round trip = %+v\nwant %+v", got, Default())
	}
}

func TestDefaultHybridValues(t *testing.T) {
	p := Default()
	if p.Version != 2 || p.Revision != "0" {
		t.Errorf("version/revision = %d/%q, want 2/0", p.Version, p.Revision)
	}
	wantLimits := Limits{QueueSeconds: 0, ColdStartSeconds: 0, MaxAttempts: 3, PreOutputBudgetSeconds: 30, MaxOutputTokens: 4096}
	if p.Routing.Limits != wantLimits {
		t.Errorf("limits = %+v, want %+v", p.Routing.Limits, wantLimits)
	}
	if p.Routing.Listener.Enabled {
		t.Error("listener enabled by default")
	}
	if len(p.Routing.Tiers) != 3 {
		t.Errorf("tier settings = %v, want weak, medium and strong", p.Routing.Tiers)
	}
	for tier, tr := range p.Routing.Tiers {
		if tr.RemoteAllowed || tr.PaidAllowed {
			t.Errorf("tier %s allows remote or paid by default", tier)
		}
	}
}

func TestParseRejectsUnsupportedVersions(t *testing.T) {
	for _, doc := range []string{`{"version":0}`, `{"version":3}`, `{"version":-1}`} {
		if _, err := Parse([]byte(doc)); err == nil || !strings.Contains(err.Error(), "unsupported policy version") {
			t.Errorf("Parse(%s) err = %v, want unsupported policy version", doc, err)
		}
	}
}

func TestParseRequiresEntryKindInV2(t *testing.T) {
	doc := `{"version":2,"tiers":{"weak":[{"engine":"ollama","model":"m"}]}}`
	if _, err := Parse([]byte(doc)); err == nil || !strings.Contains(err.Error(), "kind") {
		t.Fatalf("Parse(entry without kind) err = %v, want a kind error", err)
	}
}

func TestValidateRevision(t *testing.T) {
	test := func(rev string, wantOK bool) {
		t.Helper()
		t.Run(fmt.Sprintf("%q", rev), func(t *testing.T) {
			p := Default()
			p.Revision = rev
			if err := p.Validate(); (err == nil) != wantOK {
				t.Fatalf("Validate() with revision %q = %v, want ok=%v", rev, err, wantOK)
			}
		})
	}
	test("0", true)
	test("7", true)
	test("9007199254740993", true)
	test("9223372036854775807", true)
	test("", false)
	test("00", false)
	test("01", false)
	test("-1", false)
	test("+1", false)
	test("1.0", false)
	test("1e3", false)
	test(" 1", false)
	test("9223372036854775808", false)
}

func TestParseV2DuplicateEntryKey(t *testing.T) {
	parse := func(entries string) error {
		_, err := Parse([]byte(`{"version":2,"tiers":{"weak":[` + entries + `]}}`))
		return err
	}
	local := func(extra string) string {
		return `{"kind":"local","engine":"ollama","model":"m"` + extra + `}`
	}
	t.Run("near duplicates differing in context validate", func(t *testing.T) {
		if err := parse(local(`,"contextTokens":100`) + "," + local(`,"contextTokens":32000`)); err != nil {
			t.Fatalf("Parse: %v", err)
		}
	})
	t.Run("near duplicates differing in capabilities validate", func(t *testing.T) {
		if err := parse(local("") + "," + local(`,"capabilities":["tools"]`)); err != nil {
			t.Fatalf("Parse: %v", err)
		}
	})
	t.Run("exact duplicates are rejected", func(t *testing.T) {
		if err := parse(local(`,"contextTokens":100`) + "," + local(`,"contextTokens":100`)); err == nil || !strings.Contains(err.Error(), "duplicate entry") {
			t.Fatalf("Parse err = %v, want duplicate entry", err)
		}
	})
	t.Run("capability order does not distinguish entries", func(t *testing.T) {
		err := parse(local(`,"capabilities":["tools","structured"]`) + "," + local(`,"capabilities":["structured","tools"]`))
		if err == nil || !strings.Contains(err.Error(), "duplicate entry") {
			t.Fatalf("Parse err = %v, want duplicate entry", err)
		}
	})
}

func TestValidateRoles(t *testing.T) {
	test := func(name string, roles map[string]Role, wantErr string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			p := Default()
			p.Roles = roles
			err := p.Validate()
			if wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("Validate() = %v, want an error containing %q", err, wantErr)
			}
		})
	}

	test("no roles", map[string]Role{}, "")
	test("every tier declared", map[string]Role{
		TierWeak: {Engine: "ollama", Model: "qwen3:8b"}, TierMedium: {Engine: "lmstudio", Model: "mid"}, TierStrong: {Engine: "ollama", Model: "big"},
	}, "")
	test("same model serves two tiers", map[string]Role{
		TierWeak: {Engine: "ollama", Model: "qwen3:8b"}, TierMedium: {Engine: "ollama", Model: "qwen3:8b"},
	}, "")
	test("role without a profile", map[string]Role{TierStrong: {Engine: "ollama", Model: "not-profiled"}}, "")
	test("unknown tier", map[string]Role{"huge": {Engine: "ollama", Model: "m"}}, `unknown tier "huge"`)
	test("suffixed alias as tier", map[string]Role{"weak-local": {Engine: "ollama", Model: "m"}}, `unknown tier "weak-local"`)
	test("unknown engine", map[string]Role{TierWeak: {Engine: "no:pe", Model: "m"}}, `unknown engine "no:pe"`)
	test("descriptor engine", map[string]Role{TierWeak: {Engine: "llamacpp-custom", Model: "m"}}, "")
	test("missing model", map[string]Role{TierWeak: {Engine: "ollama"}}, "model is required")
	test("tier name as model", map[string]Role{TierWeak: {Engine: "ollama", Model: "strong"}}, "reserved alias")
	test("suffixed alias as model", map[string]Role{TierWeak: {Engine: "ollama", Model: "medium-api"}}, "reserved alias")
}

func TestValidatePausedBy(t *testing.T) {
	test := func(name string, availability Availability, pausedBy PauseOrigin, wantErr string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			p := Default()
			p.Availability = availability
			p.PausedBy = pausedBy
			err := p.Validate()
			if wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("Validate() = %v, want an error containing %q", err, wantErr)
			}
		})
	}

	test("available without provenance", Available, "", "")
	test("paused locally", Paused, PausedLocally, "")
	test("paused by a peer", Paused, PausedByPeer, "")
	// A pause saved before provenance existed counts as local.
	test("paused without provenance", Paused, "", "")
	test("available but paused locally", Available, PausedLocally, "pausedBy must be empty")
	test("available but paused by a peer", Available, PausedByPeer, "pausedBy must be empty")
	test("unknown provenance", Paused, "remote", `pausedBy must be "local" or "peer"`)
}

func TestParseOlderFileWithoutRolesOrPausedBy(t *testing.T) {
	p, err := Parse([]byte(`{"version":2,"availability":"paused","tiers":{"weak":[{"kind":"local","engine":"ollama","model":"m"}]}}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if p.Roles == nil || len(p.Roles) != 0 {
		t.Errorf("roles = %#v, want an empty map", p.Roles)
	}
	if p.PausedBy != "" {
		t.Errorf("pausedBy = %q, want empty", p.PausedBy)
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(data), `"roles":{}`) || strings.Contains(string(data), "pausedBy") {
		t.Errorf("marshaled policy = %s, want an empty roles map and no pausedBy", data)
	}
	again, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse(Marshal(Parse(x))): %v", err)
	}
	if !reflect.DeepEqual(again, p) {
		t.Errorf("round trip changed the policy:\n got %+v\nwant %+v", again, p)
	}
}

func TestParseKeepsRolesAndPausedBy(t *testing.T) {
	p, err := Parse([]byte(`{"version":2,"availability":"paused","pausedBy":"local",
		"roles":{"weak":{"engine":"ollama","model":"qwen3:8b"},"strong":{"engine":"lmstudio","model":"big"}}}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	want := map[string]Role{TierWeak: {Engine: "ollama", Model: "qwen3:8b"}, TierStrong: {Engine: "lmstudio", Model: "big"}}
	if !reflect.DeepEqual(p.Roles, want) {
		t.Errorf("roles = %+v, want %+v", p.Roles, want)
	}
	if p.PausedBy != PausedLocally {
		t.Errorf("pausedBy = %q, want %q", p.PausedBy, PausedLocally)
	}
}

func TestValidateProfileQualification(t *testing.T) {
	ref := &QualificationRef{ID: "msi-qwen3-8b", Revision: "3"}
	test := func(name string, pr Profile, wantErr string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			p := Default()
			pr.Name, pr.Engine, pr.Model = "m", "ollama", "m:latest"
			p.Profiles = []Profile{pr}
			err := p.Validate()
			if wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), wantErr) {
				t.Fatalf("Validate() = %v, want an error containing %q", err, wantErr)
			}
		})
	}

	test("no strict bounds", Profile{ContextTokens: 8192}, "")
	test("output ceiling below the context", Profile{ContextTokens: 8192, MaxOutputTokens: 2048}, "")
	test("qualified", Profile{ContextTokens: 8192, MaxOutputTokens: 2048, Qualification: ref}, "")
	test("negative output ceiling", Profile{ContextTokens: 8192, MaxOutputTokens: -1}, "maxOutputTokens must not be negative")
	test("output ceiling without a context", Profile{MaxOutputTokens: 2048}, "below a nonzero contextTokens")
	test("output ceiling filling the context", Profile{ContextTokens: 2048, MaxOutputTokens: 2048}, "below a nonzero contextTokens")
	test("qualified without an output ceiling", Profile{ContextTokens: 8192, Qualification: ref}, "nonzero contextTokens and maxOutputTokens")
	test("qualified without a context", Profile{Qualification: ref}, "nonzero contextTokens and maxOutputTokens")
	test("qualification without an id", Profile{ContextTokens: 8192, MaxOutputTokens: 1, Qualification: &QualificationRef{Revision: "1"}}, "qualification id")
	test("qualification id with a slash", Profile{ContextTokens: 8192, MaxOutputTokens: 1, Qualification: &QualificationRef{ID: "a/b", Revision: "1"}}, "qualification id")
	test("qualification revision zero", Profile{ContextTokens: 8192, MaxOutputTokens: 1, Qualification: &QualificationRef{ID: "q", Revision: "0"}}, "qualification revision")
	test("noncanonical qualification revision", Profile{ContextTokens: 8192, MaxOutputTokens: 1, Qualification: &QualificationRef{ID: "q", Revision: "01"}}, "qualification revision")
}

func TestParseKeepsProfileQualification(t *testing.T) {
	doc := `{"version":2,"profiles":[{"name":"m","engine":"ollama","model":"m:latest","contextTokens":8192,
		"maxOutputTokens":2048,"qualification":{"id":"msi-qwen3-8b","revision":"3"}}]}`
	p, err := Parse([]byte(doc))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	pr := p.Profiles[0]
	if pr.MaxOutputTokens != 2048 || pr.Qualification == nil || *pr.Qualification != (QualificationRef{ID: "msi-qwen3-8b", Revision: "3"}) {
		t.Fatalf("profile = %+v, want the output ceiling and qualification kept", pr)
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	again, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse(Marshal(Parse(x))): %v", err)
	}
	if !reflect.DeepEqual(again, p) {
		t.Errorf("round trip changed the policy:\n got %+v\nwant %+v", again, p)
	}
}
