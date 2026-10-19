// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestServedResidentAdvertisementRoundTrip(t *testing.T) {
	for _, name := range []string{"absent", "empty", "subset"} {
		t.Run(name, func(t *testing.T) {
			p := servedResidentPolicy()
			if name == "absent" {
				p.ServedModels = nil
			}
			if name == "empty" {
				p.ServedModels = ServedModels{}
				p.ResidentModels = nil
			}
			adv := p.Advertisement()
			for range 3 {
				raw, err := json.Marshal(adv)
				if err != nil {
					t.Fatal(err)
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(raw, &fields); err != nil {
					t.Fatal(err)
				}
				if (fields["servedModels"] != nil) != (p.ServedModels != nil) {
					t.Fatal("permission presence lost")
				}
				for _, private := range []string{"residentModels", "routing", "profiles", "tiers"} {
					if fields[private] != nil {
						t.Fatalf("advertisement exposed %s", private)
					}
				}
				var next Advertisement
				if err := json.Unmarshal(raw, &next); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(adv, next) {
					t.Fatal("unstable advertisement")
				}
				adv = next
			}
		})
	}
}

func TestServedResidentAdvertisementOwnedSlices(t *testing.T) {
	p := servedResidentPolicy()
	adv := p.Advertisement()
	p.ServedModels[0].Model = "caller change"
	if adv.ServedModels[0].Model != "m" {
		t.Fatal("advertisement aliases caller slices")
	}
	adv.ServedModels[0].Model = "advertisement change"
	if p.ServedModels[0].Model != "caller change" {
		t.Fatal("policy aliases advertisement")
	}
}

func TestServedResidentAdvertisementRawKeys(t *testing.T) {
	for _, raw := range []string{
		`{"servedModels":null}`, `{"servedModels":[],"servedModels":[]}`,
		`{"servedModels":[],"ServedModels":[]}`, `{"ServedModels":[]}`,
		`{"servedModels":[{"engine":"ollama","model":"m","model":"m"}]}`,
		`{"servedModels":[{"engine":"ollama","Model":"m"}]}`,
	} {
		var adv Advertisement
		if err := json.Unmarshal([]byte(raw), &adv); err == nil {
			t.Fatalf("advertisement accepted %s", raw)
		}
	}
}

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

func TestAdvertisementStrictCapabilityCompatibility(t *testing.T) {
	for _, original := range []string{
		`{"revision":"0","roles":{}}`,
		`{"revision":"12","roles":{"weak":{"engine":"ollama","model":"m","contextTokens":1024,"capabilities":["tools"]}},"pausedBy":"peer"}`,
		`{"revision":"0","roles":{},"strictAdmission":{"version":2,"future":{"newField":true}}}`,
	} {
		t.Run(original, func(t *testing.T) {
			var got Advertisement
			if err := json.Unmarshal([]byte(original), &got); err != nil {
				t.Fatal(err)
			}
			wantState := StrictCapabilityAbsent
			if strings.Contains(original, "strictAdmission") {
				wantState = StrictCapabilityUnsupported
			}
			if got.StrictAdmission.State() != wantState {
				t.Fatalf("state = %q, want %q", got.StrictAdmission.State(), wantState)
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			if string(encoded) != original {
				t.Fatalf("re-encoded = %s, want %s", encoded, original)
			}
		})
	}
}

func TestAdvertisementStrictCapabilityRejectsNull(t *testing.T) {
	var got Advertisement
	if err := json.Unmarshal([]byte(`{"revision":"0","roles":{},"strictAdmission":null}`), &got); err == nil {
		t.Fatal("accepted explicit null capability")
	}
}

// Both languages decode Document as raw JSON, preserving escape spellings until
// the actual Advertisement decoder runs (not a fixture mutation/re-encoding).
func TestAdvertisementStrictRawParityCorpus(t *testing.T) {
	data, err := os.ReadFile("testdata/strict-admission-raw.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name, Document, Error, Model string
		State                        StrictCapabilityState
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			var got Advertisement
			err := json.Unmarshal([]byte(tc.Document), &got)
			if tc.Error != "" {
				if err == nil || !strings.Contains(err.Error(), tc.Error) {
					t.Fatalf("decode error = %v, want %q", err, tc.Error)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.StrictAdmission.State() != tc.State {
				t.Fatalf("state = %q, want %q", got.StrictAdmission.State(), tc.State)
			}
			if tc.Model != "" && got.StrictAdmission.Models[0].Model != tc.Model {
				t.Fatalf("model = %q, want %q", got.StrictAdmission.Models[0].Model, tc.Model)
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			var again Advertisement
			if err := json.Unmarshal(encoded, &again); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, again) {
				t.Fatalf("unstable round trip: %+v / %+v", got, again)
			}
		})
	}
}

func TestAdvertisementLegacyCaseFolding(t *testing.T) {
	var got Advertisement
	input := `{"revision":"1","Revision":"2","ROLES":{"weak":{"ENGINE":"ollama","MODEL":"m"}},"PAUSEDBY":"peer"}`
	if err := json.Unmarshal([]byte(input), &got); err != nil {
		t.Fatal(err)
	}
	want := Advertisement{Revision: "2", Roles: map[string]AdvertisedRole{"weak": {Engine: "ollama", Model: "m"}}, PausedBy: PausedByPeer}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("decoded = %+v, want %+v", got, want)
	}
}

func TestAdvertisementCarriesProfileQualification(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{{
		Name: "q", Engine: "ollama", Model: "q:latest", ContextTokens: 8192, MaxOutputTokens: 2048,
		Qualification: &QualificationRef{ID: "msi-qwen3-8b", Revision: "3"},
	}}
	p.Roles = map[string]Role{TierWeak: {Engine: "ollama", Model: "q:latest"}, TierStrong: {Engine: "ollama", Model: "unprofiled"}}
	if err := p.Validate(); err != nil {
		t.Fatalf("fixture policy is invalid: %v", err)
	}

	got := p.Advertisement()
	weak := got.Roles[TierWeak]
	want := AdvertisedRole{Engine: "ollama", Model: "q:latest", ContextTokens: 8192, MaxOutputTokens: 2048,
		Qualification: &QualificationRef{ID: "msi-qwen3-8b", Revision: "3"}}
	if !reflect.DeepEqual(weak, want) {
		t.Fatalf("weak role = %+v, want %+v", weak, want)
	}
	if !weak.HybridQualified() {
		t.Errorf("the qualified role is not hybrid qualified")
	}
	if strong := got.Roles[TierStrong]; strong.Qualification != nil || strong.MaxOutputTokens != 0 || strong.HybridQualified() {
		t.Errorf("a role without a profile = %+v, want no qualification", strong)
	}

	weak.Qualification.Revision = "4"
	if p.Profiles[0].Qualification.Revision != "3" {
		t.Errorf("editing the advertisement changed the policy's profile")
	}

	data, err := json.Marshal(p.Advertisement().Roles[TierWeak])
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"engine":"ollama","model":"q:latest","contextTokens":8192,"maxOutputTokens":2048,"qualification":{"id":"msi-qwen3-8b","revision":"3"}}`; string(data) != want {
		t.Errorf("role JSON = %s, want %s", data, want)
	}
}

func TestAdvertisedRoleHybridQualified(t *testing.T) {
	ref := &QualificationRef{ID: "q", Revision: "1"}
	for _, tc := range []struct {
		name string
		role AdvertisedRole
		want bool
	}{
		{"finite and qualified", AdvertisedRole{ContextTokens: 4096, MaxOutputTokens: 1024, Qualification: ref}, true},
		{"no qualification", AdvertisedRole{ContextTokens: 4096, MaxOutputTokens: 1024}, false},
		{"no context", AdvertisedRole{MaxOutputTokens: 1024, Qualification: ref}, false},
		{"no output ceiling", AdvertisedRole{ContextTokens: 4096, Qualification: ref}, false},
		{"output filling the context", AdvertisedRole{ContextTokens: 1024, MaxOutputTokens: 1024, Qualification: ref}, false},
		{"negative output", AdvertisedRole{ContextTokens: 4096, MaxOutputTokens: -1, Qualification: ref}, false},
		{"malformed reference", AdvertisedRole{ContextTokens: 4096, MaxOutputTokens: 1024, Qualification: &QualificationRef{ID: "q", Revision: "x"}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.role.HybridQualified(); got != tc.want {
				t.Fatalf("HybridQualified() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestAdvertisementDecodesMalformedQualificationLeniently(t *testing.T) {
	// A peer's malformed qualification must not fail the inventory decode; it
	// only leaves the role ineligible for hybrid requests.
	for _, tc := range []struct {
		name, role string
	}{
		{"invalid values", `"maxOutputTokens":9999,"qualification":{"id":"","revision":"0","extra":true}`},
		{"output of the wrong type", `"maxOutputTokens":"many","qualification":{"id":"q","revision":"1"}`},
		{"qualification of the wrong type", `"maxOutputTokens":1024,"qualification":"q@1"`},
		{"qualification fields of the wrong type", `"maxOutputTokens":1024,"qualification":{"id":7,"revision":["1"]}`},
		{"null qualification", `"maxOutputTokens":1024,"qualification":null`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var adv Advertisement
			doc := `{"revision":"1","roles":{"weak":{"engine":"ollama","model":"m","contextTokens":4096,` + tc.role + `}}}`
			if err := json.Unmarshal([]byte(doc), &adv); err != nil {
				t.Fatalf("decode: %v", err)
			}
			role := adv.Roles[TierWeak]
			if role.Engine != "ollama" || role.Model != "m" || role.ContextTokens != 4096 {
				t.Fatalf("role = %+v, want its engine, model and context kept", role)
			}
			if role.HybridQualified() {
				t.Fatalf("malformed role = %+v counts as hybrid qualified", role)
			}
		})
	}
}

func TestAdvertisedRoleKeepsLegacyCaseFolding(t *testing.T) {
	var role AdvertisedRole
	if err := json.Unmarshal([]byte(`{"ENGINE":"ollama","Model":"m","CONTEXTTOKENS":4096,"maxoutputtokens":1024,"Qualification":{"ID":"q","REVISION":"1"}}`), &role); err != nil {
		t.Fatal(err)
	}
	want := AdvertisedRole{Engine: "ollama", Model: "m", ContextTokens: 4096, MaxOutputTokens: 1024, Qualification: &QualificationRef{ID: "q", Revision: "1"}}
	if !reflect.DeepEqual(role, want) {
		t.Fatalf("decoded = %+v, want %+v", role, want)
	}
}
