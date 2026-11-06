// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
)

func TestAdvertisementListsRoleProfiles(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{
		{Key: "z", DisplayName: "Last", Engine: "llamacpp", Model: "z", ContextTokens: 4096, MaxOutputTokens: 1024, Capabilities: []Capability{CapTools}, Qualification: &QualificationRef{ID: "q", Revision: "1"}, ModelIdentity: &engines.ModelIdentity{Strength: "declared", Files: []engines.ModelIdentityFile{{Role: "model", Name: "z.gguf", SizeBytes: 42}}}},
		{Key: "a", DisplayName: "First", Engine: "llamacpp", Model: "a"},
		{Key: "unused", DisplayName: "Unused", Engine: "llamacpp", Model: "unused"},
		{Key: "private", DisplayName: "Private", Engine: "llamacpp", Model: "private", LocalOnly: true},
	}
	p.Roles = []Role{{Name: TierStrong, Profiles: []string{"z", "a"}}, {Name: TierMedium, Profiles: []string{}}, {Name: TierWeak, Profiles: []string{"z", "private"}}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	a := p.Advertisement()
	wantRoles := []Role{{Name: TierStrong, Profiles: []string{"z", "a"}}, {Name: TierMedium, Profiles: []string{}}, {Name: TierWeak, Profiles: []string{"z", "private"}}}
	if !reflect.DeepEqual(a.Roles, wantRoles) {
		t.Fatalf("roles = %+v, want %+v", a.Roles, wantRoles)
	}
	var keys []string
	for _, pr := range a.Profiles {
		keys = append(keys, pr.Key)
	}
	if !reflect.DeepEqual(keys, []string{"a", "unused", "z"}) {
		t.Fatalf("profile keys = %v", keys)
	}
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	var fields struct {
		Roles []struct {
			Name     string
			Profiles json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	if string(fields.Roles[1].Profiles) != "[]" {
		t.Fatalf("empty membership = %s", fields.Roles[1].Profiles)
	}
	a.Roles[0].Profiles[0] = "changed"
	a.Profiles[2].Capabilities[0] = CapVision
	a.Profiles[2].Qualification.Revision = "2"
	a.Profiles[2].ModelIdentity.Files[0].Name = "changed.gguf"
	if p.Roles[0].Profiles[0] != "z" || p.Profiles[0].Capabilities[0] != CapTools || p.Profiles[0].Qualification.Revision != "1" || p.Profiles[0].ModelIdentity.Files[0].Name != "z.gguf" {
		t.Fatal("advertisement aliases policy nested state")
	}
	p.Roles = nil
	if got := p.Advertisement(); len(got.Profiles) != 3 || len(got.Roles) != 0 {
		t.Fatalf("profiles without roles = %+v", got)
	}
	t.Run("malformed profile prunes membership", func(t *testing.T) {
		var got Advertisement
		err := json.Unmarshal([]byte(`{"revision":"1","roles":[{"name":"strong","profiles":["bad"]},{"name":"weak","profiles":["good","bad"]}],"profiles":[{"key":"bad","displayName":"Profile","engine":"llamacpp","model":"bad\nmodel"},{"key":"good","displayName":"Profile","engine":"llamacpp","model":"good"}]}`), &got)
		if err != nil {
			t.Fatal(err)
		}
		want := []Role{{Name: TierStrong, Profiles: []string{}}, {Name: TierWeak, Profiles: []string{"good"}}}
		if !reflect.DeepEqual(got.Roles, want) || len(got.Profiles) != 1 || got.Profiles[0].Key != "good" {
			t.Fatalf("filtered advertisement = %+v", got)
		}
	})
}

func TestAdvertisementFiltersInvalidProfileMetadata(t *testing.T) {
	for name, metadata := range map[string]string{
		"missing display name":      `"icon":"code"`,
		"empty display name":        `"displayName":""`,
		"control display name":      `"displayName":"bad\nname"`,
		"oversized display name":    `"displayName":"` + strings.Repeat("x", 129) + `"`,
		"invalid display UTF-8":     `"displayName":"` + string([]byte{0xff}) + `"`,
		"invalid icon":              `"displayName":"Bad","icon":"unknown"`,
		"invalid identity strength": `"displayName":"Bad","modelIdentity":{"strength":"verified"}`,
		"invalid identity file":     `"displayName":"Bad","modelIdentity":{"strength":"declared","files":[{"role":"model","name":"../bad"}]}`,
		"invalid identity UTF-8":    `"displayName":"Bad","modelIdentity":{"strength":"declared","generalName":"` + string([]byte{0xff}) + `"}`,
	} {
		t.Run(name, func(t *testing.T) {
			raw := []byte(`{"profiles":[{"key":"bad","engine":"llamacpp","model":"bad",` + metadata + `},{"key":"good","displayName":"Good","engine":"llamacpp","model":"good"}],"roles":[{"name":"weak","profiles":["bad","good"]},{"name":"medium","profiles":["bad"]}]}`)
			var got Advertisement
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatal(err)
			}
			wantRoles := []Role{{Name: TierWeak, Profiles: []string{"good"}}, {Name: TierMedium, Profiles: []string{}}}
			if len(got.Profiles) != 1 || got.Profiles[0].Key != "good" || !reflect.DeepEqual(got.Roles, wantRoles) {
				t.Fatalf("malformed profile affected retained facts/declarations: %+v", got)
			}
		})
	}
}

func TestPauseModeAdvertisementRoundTrip(t *testing.T) {
	for _, mode := range []PauseMode{PauseConfigured, PauseImmediate} {
		t.Run(string(mode), func(t *testing.T) {
			p := Default()
			p.Availability, p.PausedBy, p.PauseMode = Paused, PausedLocally, mode
			raw, err := json.Marshal(p.Advertisement())
			if err != nil {
				t.Fatalf("marshal advertisement: %v", err)
			}
			var got Advertisement
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("decode advertisement: %v", err)
			}
			if got.PausedBy != PausedLocally || got.PauseMode != mode {
				t.Fatalf("advertisement = %+v, want local/%s", got, mode)
			}
		})
	}
}

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
				for _, private := range []string{"residentModels", "routing", "tiers"} {
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
		`{"servedModels":[{"engine":"llamacpp","model":"m","model":"m"}]}`,
		`{"servedModels":[{"engine":"llamacpp","Model":"m"}]}`,
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
		Key: "qwen", DisplayName: "qwen", Engine: "llamacpp", Model: "qwen3-8b", ContextTokens: 16384,
		Capabilities: []Capability{CapTools, CapVision}, MemoryBytes: 5000000000, MaxConcurrent: 2,
		RequestOptions: json.RawMessage(`{"keep_alive":"10m"}`),
	}}
	p.Roles = []Role{{Name: TierWeak, Profiles: []string{"qwen"}}, {Name: TierMedium, Profiles: []string{"qwen"}}, {Name: TierStrong, Profiles: []string{}}}
	p.Availability = Paused
	p.PausedBy = PausedByPeer
	if err := p.Validate(); err != nil {
		t.Fatalf("fixture policy is invalid: %v", err)
	}

	got := p.Advertisement()
	bounded := AdvertisedProfile{Key: "qwen", DisplayName: "qwen", Engine: "llamacpp", Model: "qwen3-8b", ContextTokens: 16384, Capabilities: []Capability{CapTools, CapVision}}
	want := Advertisement{Revision: "12", Roles: []Role{{Name: TierWeak, Profiles: []string{"qwen"}}, {Name: TierMedium, Profiles: []string{"qwen"}}, {Name: TierStrong, Profiles: []string{}}}, Profiles: []AdvertisedProfile{bounded}, PausedBy: PausedByPeer}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Advertisement() = %+v\nwant %+v", got, want)
	}

	got.Profiles[0].Capabilities[0] = CapEmbeddings
	if p.Profiles[0].Capabilities[0] != CapTools {
		t.Errorf("editing the advertisement changed the policy's profile")
	}

	data, err := json.Marshal(p.Advertisement())
	if err != nil {
		t.Fatalf("marshal advertisement: %v", err)
	}
	for _, leak := range []string{
		"routing", "NVPAIR_PROVIDER_SECRET", "free-acct", "free-small", "deployment", "allowedOrigins",
		"tiers", "memoryBytes", "maxConcurrent", "keep_alive", "availability", "admission",
	} {
		if strings.Contains(string(data), leak) {
			t.Errorf("advertisement contains %q: %s", leak, data)
		}
	}
}

func TestAdvertisementWithoutProfileHasNoBounds(t *testing.T) {
	p := Default()
	p.Roles = []Role{{Name: TierWeak, Profiles: []string{}}}
	got := p.Advertisement()
	want := Advertisement{Revision: "0", Roles: []Role{{Name: TierWeak, Profiles: []string{}}}, Profiles: []AdvertisedProfile{}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Advertisement()=%+v, want %+v", got, want)
	}
}

func TestAdvertisementWithoutRoles(t *testing.T) {
	data, err := json.Marshal(Default().Advertisement())
	if err != nil {
		t.Fatalf("marshal advertisement: %v", err)
	}
	var got map[string]json.RawMessage
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if string(got["roles"]) != "[]" || string(got["profiles"]) != "[]" || string(got["revision"]) != `"0"` || len(got) != 3 {
		t.Fatalf("empty advertisement = %s", data)
	}
}

func TestAdvertisementStrictCapabilityCompatibility(t *testing.T) {
	for _, original := range []string{
		`{"revision":"0","roles":[],"profiles":[]}`,
		`{"revision":"12","roles":[{"name":"weak","profiles":["m"]}],"profiles":[{"key":"m","displayName":"m","engine":"llamacpp","model":"m","contextTokens":1024,"capabilities":["tools"]}],"pausedBy":"peer"}`,
		`{"revision":"0","roles":[],"profiles":[],"strictAdmission":{"version":2,"future":{"newField":true}}}`,
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
			var encodedFields, originalFields map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &encodedFields); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(original), &originalFields); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(encodedFields, originalFields) {
				t.Fatalf("re-encoded = %s, want %s", encoded, original)
			}
		})
	}
}

func TestAdvertisementStrictCapabilityRejectsNull(t *testing.T) {
	var got Advertisement
	if err := json.Unmarshal([]byte(`{"revision":"0","roles":[],"profiles":[],"strictAdmission":null}`), &got); err == nil {
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
	input := `{"revision":"1","Revision":"2","ROLES":[{"NAME":"weak","PROFILES":["m"]}],"PROFILES":[{"KEY":"m","DISPLAYNAME":"Model","ENGINE":"llamacpp","MODEL":"m"}],"PAUSEDBY":"peer"}`
	if err := json.Unmarshal([]byte(input), &got); err != nil {
		t.Fatal(err)
	}
	want := Advertisement{Revision: "2", Roles: []Role{{Name: TierWeak, Profiles: []string{"m"}}}, Profiles: []AdvertisedProfile{{Key: "m", DisplayName: "Model", Engine: "llamacpp", Model: "m"}}, PausedBy: PausedByPeer}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("decoded = %+v, want %+v", got, want)
	}
}

func TestAdvertisementCarriesProfileQualification(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{{
		Key: "q", DisplayName: "q", Engine: "llamacpp", Model: "q", ContextTokens: 8192, MaxOutputTokens: 2048,
		Qualification: &QualificationRef{ID: "msi-qwen3-8b", Revision: "3"},
	}}
	p.Roles = []Role{{Name: TierWeak, Profiles: []string{"q"}}, {Name: TierStrong, Profiles: []string{}}}
	if err := p.Validate(); err != nil {
		t.Fatalf("fixture policy is invalid: %v", err)
	}

	got := p.Advertisement()
	weak := got.Profiles[0]
	want := AdvertisedProfile{Key: "q", DisplayName: "q", Engine: "llamacpp", Model: "q", ContextTokens: 8192, MaxOutputTokens: 2048,
		Qualification: &QualificationRef{ID: "msi-qwen3-8b", Revision: "3"}}
	if !reflect.DeepEqual(weak, want) {
		t.Fatalf("weak role = %+v, want %+v", weak, want)
	}
	if !weak.HybridQualified() {
		t.Errorf("the qualified role is not hybrid qualified")
	}
	if strong := got.Roles[1]; strong.Name != TierStrong || len(strong.Profiles) != 0 {
		t.Fatalf("empty declaration lost: %+v", strong)
	}

	weak.Qualification.Revision = "4"
	if p.Profiles[0].Qualification.Revision != "3" {
		t.Errorf("editing the advertisement changed the policy's profile")
	}

	data, err := json.Marshal(p.Advertisement().Profiles[0])
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"key":"q","displayName":"q","engine":"llamacpp","model":"q","contextTokens":8192,"maxOutputTokens":2048,"qualification":{"id":"msi-qwen3-8b","revision":"3"}}`; string(data) != want {
		t.Errorf("role JSON = %s, want %s", data, want)
	}
}

func TestAdvertisedProfileHybridQualified(t *testing.T) {
	ref := &QualificationRef{ID: "q", Revision: "1"}
	for _, tc := range []struct {
		name string
		role AdvertisedProfile
		want bool
	}{
		{"finite and qualified", AdvertisedProfile{ContextTokens: 4096, MaxOutputTokens: 1024, Qualification: ref}, true},
		{"no qualification", AdvertisedProfile{ContextTokens: 4096, MaxOutputTokens: 1024}, false},
		{"no context", AdvertisedProfile{MaxOutputTokens: 1024, Qualification: ref}, false},
		{"no output ceiling", AdvertisedProfile{ContextTokens: 4096, Qualification: ref}, false},
		{"output filling the context", AdvertisedProfile{ContextTokens: 1024, MaxOutputTokens: 1024, Qualification: ref}, false},
		{"negative output", AdvertisedProfile{ContextTokens: 4096, MaxOutputTokens: -1, Qualification: ref}, false},
		{"malformed reference", AdvertisedProfile{ContextTokens: 4096, MaxOutputTokens: 1024, Qualification: &QualificationRef{ID: "q", Revision: "x"}}, false},
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
			doc := `{"revision":"1","roles":[{"name":"weak","profiles":["m"]}],"profiles":[{"key":"m","displayName":"Profile","engine":"llamacpp","model":"m","contextTokens":4096,` + tc.role + `}]}`
			if err := json.Unmarshal([]byte(doc), &adv); err != nil {
				t.Fatalf("decode: %v", err)
			}
			role := adv.Profiles[0]
			if role.Engine != "llamacpp" || role.Model != "m" || role.ContextTokens != 4096 {
				t.Fatalf("role = %+v, want its engine, model and context kept", role)
			}
			if role.HybridQualified() {
				t.Fatalf("malformed role = %+v counts as hybrid qualified", role)
			}
		})
	}
}

func TestAdvertisedProfileKeepsLegacyCaseFolding(t *testing.T) {
	var role AdvertisedProfile
	if err := json.Unmarshal([]byte(`{"ENGINE":"llamacpp","Model":"m","CONTEXTTOKENS":4096,"maxoutputtokens":1024,"Qualification":{"ID":"q","REVISION":"1"}}`), &role); err != nil {
		t.Fatal(err)
	}
	want := AdvertisedProfile{Engine: "llamacpp", Model: "m", ContextTokens: 4096, MaxOutputTokens: 1024, Qualification: &QualificationRef{ID: "q", Revision: "1"}}
	if !reflect.DeepEqual(role, want) {
		t.Fatalf("decoded = %+v, want %+v", role, want)
	}
}
