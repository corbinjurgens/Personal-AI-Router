// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestParseStoredRetiresLocalEntriesAndOrphanRoutingRules(t *testing.T) {
	for _, name := range []string{"no API entries", "coder API sibling"} {
		t.Run(name, func(t *testing.T) {
			p := stagedPolicy(t)
			p.Tiers = map[string][]TierEntry{"weak": {{Kind: EntryLocal, Engine: "llamacpp", Model: "native"}}}
			p.Routing.Tiers = map[string]TierRouting{
				"weak": {BareMode: BareLocal, AllowedFamilies: []string{}}, "medium": {BareMode: BareHybrid, AllowedFamilies: []string{}}, "strong": {BareMode: BareLocal, AllowedFamilies: []string{}},
			}
			if name == "coder API sibling" {
				p.Tiers["coder"] = []TierEntry{{Kind: EntryAPI, Deployment: "paid-big"}, {Kind: EntryAPI, Deployment: "free-small"}}
				p.Routing.Tiers["coder"] = TierRouting{BareMode: BareAPI, AllowedFamilies: []string{"beta", "alpha"}}
				p.Tiers["weak"] = append(p.Tiers["weak"], TierEntry{Kind: EntryAPI, Deployment: "free-small"})
			}
			raw, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			before := bytes.Clone(raw)
			got, _, err := ParseStored(raw)
			if err != nil {
				t.Fatal(err)
			}
			want := p
			want.Tiers = map[string][]TierEntry{"weak": {}}
			want.Routing.Tiers = map[string]TierRouting{}
			if name == "coder API sibling" {
				want.Tiers["coder"] = p.Tiers["coder"]
				want.Tiers["weak"] = []TierEntry{{Kind: EntryAPI, Deployment: "free-small"}}
				want.Routing.Tiers["coder"] = p.Routing.Tiers["coder"]
				want.Routing.Tiers["weak"] = p.Routing.Tiers["weak"]
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("repaired policy = %+v, want %+v", got, want)
			}
			if !bytes.Equal(raw, before) {
				t.Fatal("repair mutated stored input")
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			again, _, err := ParseStored(encoded)
			if err != nil || !reflect.DeepEqual(again, got) {
				t.Fatalf("repair is not stable: %+v, %v", again, err)
			}
		})
	}
}

func TestStoredReservedRepairedProfileKey(t *testing.T) {
	_, _, err := ParseStored([]byte(`{"profiles":[{"engine":"llamacpp","model":"qwen3:8b"}]}`))
	const want = "profile key qwen3:8b is reserved: the text before its colon is a valid role name"
	if err == nil || err.Error() != want {
		t.Fatalf("stored repaired key error = %v, want %q", err, want)
	}
}

func TestStoredReservedExplicitProfileKeySurvivesInvalidModelSanitation(t *testing.T) {
	_, _, err := ParseStored([]byte(`{"profiles":[{"key":"coder:fast","displayName":"Bad","engine":"llamacpp","model":"bad\nmodel"}]}`))
	const want = "profile key coder:fast is reserved: the text before its colon is a valid role name"
	if err == nil || err.Error() != want {
		t.Fatalf("stored explicit key error = %v, want %q", err, want)
	}
}

func TestStoredObjectRolesDropWithoutReservingBareKey(t *testing.T) {
	p, _, err := ParseStored([]byte(`{"roles":{"medium":{"engine":"llamacpp","model":"medium"}},"profiles":[{"engine":"llamacpp","model":"medium"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Roles) != 0 || len(p.Profiles) != 1 || p.Profiles[0].Key != "medium" {
		t.Fatalf("stored policy = %+v", p)
	}
}

func TestParseStoredProfileIdentityUsesKeys(t *testing.T) {
	tests := []struct {
		name      string
		profiles  string
		want      []Profile
		roles     []Role
		wantRoles []Role
		dropped   int
	}{
		{
			name:      "final keys without old names",
			profiles:  `[{"key":"small","displayName":"Small","engine":"llamacpp","model":"small"},{"key":"large","displayName":"Large","engine":"llamacpp","model":"large"}]`,
			want:      []Profile{{Key: "small", DisplayName: "Small", Engine: "llamacpp", Model: "small"}, {Key: "large", DisplayName: "Large", Engine: "llamacpp", Model: "large"}},
			roles:     []Role{{Name: "weak", Profiles: []string{"large", "small"}}},
			wantRoles: []Role{{Name: "weak", Profiles: []string{"large", "small"}}},
		},
		{
			name:      "repeated labels and malformed sibling",
			profiles:  `[{"key":"small","displayName":"Shared","engine":"llamacpp","model":"small"},{"key":"bad","displayName":"Shared","engine":"llamacpp","model":"bad\nmodel"},{"key":"large","displayName":"Shared","engine":"llamacpp","model":"large"}]`,
			want:      []Profile{{Key: "small", DisplayName: "Shared", Engine: "llamacpp", Model: "small"}, {Key: "large", DisplayName: "Shared", Engine: "llamacpp", Model: "large"}},
			roles:     []Role{{Name: "weak", Profiles: []string{"large", "bad", "small"}}, {Name: "strong", Profiles: []string{"bad"}}},
			wantRoles: []Role{{Name: "weak", Profiles: []string{"large", "small"}}, {Name: "strong", Profiles: []string{}}},
			dropped:   1,
		},
		{
			name:      "legacy identity uses existing model rule",
			profiles:  `[{"name":"Shared","engine":"llamacpp","model":"small"},{"name":"Shared","engine":"llamacpp","model":"large"}]`,
			want:      []Profile{{Key: "small", DisplayName: "Shared", Engine: "llamacpp", Model: "small"}, {Key: "large", DisplayName: "Shared", Engine: "llamacpp", Model: "large"}},
			roles:     []Role{{Name: "weak", Profiles: []string{"large", "small"}}},
			wantRoles: []Role{{Name: "weak", Profiles: []string{"large", "small"}}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(map[string]any{"profiles": json.RawMessage(tc.profiles), "roles": tc.roles})
			if err != nil {
				t.Fatal(err)
			}
			before := bytes.Clone(raw)
			got, counts, err := ParseStored(raw)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Profiles, tc.want) || !reflect.DeepEqual(got.Roles, tc.wantRoles) || counts["profiles"] != tc.dropped {
				t.Fatalf("profiles = %+v, roles = %+v, counts = %v", got.Profiles, got.Roles, counts)
			}
			if !bytes.Equal(raw, before) {
				t.Fatal("stored repair mutated input")
			}
		})
	}
}

func TestParseStoredDropsOldRoles(t *testing.T) {
	t.Run("old role object and unkeyed profiles repair together", func(t *testing.T) {
		p := stagedPolicy(t)
		raw, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			t.Fatal(err)
		}
		fields["profiles"] = json.RawMessage(`[{"name":"small","engine":"llamacpp","model":"small-model"},{"name":"large","engine":"llamacpp","model":"large-model"}]`)
		fields["roles"] = json.RawMessage(`{"weak":{"engine":"llamacpp","model":"small-model"},"medium":{"engine":"llamacpp","model":"large-model"}}`)
		raw, err = json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		before := bytes.Clone(raw)
		got, _, err := ParseStored(raw)
		if err != nil {
			t.Fatal(err)
		}
		want := []Profile{{Key: "small-model", DisplayName: "small", Engine: "llamacpp", Model: "small-model"}, {Key: "large-model", DisplayName: "large", Engine: "llamacpp", Model: "large-model"}}
		if got.Roles == nil || len(got.Roles) != 0 || !reflect.DeepEqual(got.Profiles, want) || !reflect.DeepEqual(got.Tiers, p.Tiers) || !bytes.Equal(raw, before) {
			t.Fatalf("joint role/profile repair changed identity or API entries: %+v", got)
		}
	})
	t.Run("old declarations dropped without changing profiles or API tiers", func(t *testing.T) {
		original := stagedPolicy(t)
		original.Profiles = []Profile{{Key: "exact/model:Q4", DisplayName: "Exact", Engine: "llamacpp", Model: "exact/model:Q4"}}
		raw, err := json.Marshal(original)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			t.Fatal(err)
		}
		fields["roles"] = json.RawMessage(`{"medium":{"engine":"llamacpp","model":"exact/model:Q4"}}`)
		raw, err = json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		before := bytes.Clone(raw)
		got, _, err := ParseStored(raw)
		if err != nil {
			t.Fatal(err)
		}
		if got.Roles == nil || len(got.Roles) != 0 || !reflect.DeepEqual(got.Profiles, original.Profiles) || !reflect.DeepEqual(got.Tiers, original.Tiers) || !reflect.DeepEqual(got.Routing, original.Routing) {
			t.Fatalf("stored repair changed retained state: %+v", got)
		}
		if !bytes.Equal(raw, before) {
			t.Fatal("repair mutated stored bytes")
		}
	})
	t.Run("list roles preserved", func(t *testing.T) {
		p := Default()
		p.Profiles = []Profile{{Key: "exact", DisplayName: "Exact", Engine: "llamacpp", Model: "m"}}
		p.Roles = []Role{{Name: "strong", Profiles: []string{"exact"}}, {Name: "medium", Profiles: []string{}}}
		raw, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		got, _, err := ParseStored(raw)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Roles, p.Roles) {
			t.Fatalf("roles = %+v, want %+v", got.Roles, p.Roles)
		}
	})
	t.Run("reserved legacy model retains identity before validation", func(t *testing.T) {
		raw := []byte(`{"profiles":[{"name":"legacy","engine":"llamacpp","model":"medium"}]}`)
		repaired, err := repairStoredProfileKeys(raw)
		if err != nil {
			t.Fatal(err)
		}
		var fields struct{ Profiles []Profile }
		if err := json.Unmarshal(repaired, &fields); err != nil {
			t.Fatal(err)
		}
		if len(fields.Profiles) != 1 || fields.Profiles[0].Key != "medium" || fields.Profiles[0].Model != "medium" {
			t.Fatalf("repair renamed legacy identity: %+v", fields.Profiles)
		}
		p, _, err := ParseStored(raw)
		if err != nil || len(p.Roles) != 0 || len(p.Profiles) != 1 || p.Profiles[0].Key != "medium" {
			t.Fatalf("bare stored profile: %+v, %v", p, err)
		}
	})
}

func TestParseStoredGivesProfilesKeys(t *testing.T) {
	raw := []byte(`{"revision":"17","profiles":[{"name":"old-display","engine":"llamacpp","model":"unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL"}],"servedModels":[{"engine":"llamacpp","model":"unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL"}]}`)
	before := bytes.Clone(raw)
	p, _, err := ParseStored(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Profiles) != 1 || p.Profiles[0].Key != "unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL" || p.Profiles[0].DisplayName != "old-display" || p.Revision != "17" || p.ServedModels[0].Model != p.Profiles[0].Key {
		t.Fatalf("identity repair = %+v", p)
	}
	if !bytes.Equal(raw, before) {
		t.Fatal("stored repair changed source bytes")
	}
	encoded, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	again, _, err := ParseStored(encoded)
	if err != nil || !reflect.DeepEqual(again, p) {
		t.Fatalf("repaired policy is not stable: %v", err)
	}
	if _, err := Parse(raw); err == nil {
		t.Fatal("raw policy write acquired an implicit key")
	}
}

func TestParseStoredRefusesAmbiguousKeys(t *testing.T) {
	raw := []byte(`{"profiles":[{"name":"llama","engine":"llamacpp","model":"shared-model"},{"name":"strata","engine":"strata","model":"shared-model"}]}`)
	before := bytes.Clone(raw)
	p, counts, err := ParseStored(raw)
	want := "Profiles share a key after repair: llamacpp/shared-model, strata/shared-model. Rename one in this node's policy file before installing."
	if err == nil || err.Error() != want {
		t.Fatalf("collision error = %v, want %q", err, want)
	}
	if !reflect.DeepEqual(p, Policy{}) || counts != nil || !bytes.Equal(raw, before) {
		t.Fatal("ambiguous repair published a policy or changed source bytes")
	}
}

func TestStoredProfileFieldRepairDropsObsoleteMetadata(t *testing.T) {
	for _, tc := range []struct {
		name, profile, display string
		repairedKey            bool
	}{
		{"unkeyed", `{"name":"Old label","quantization":"q4","engine":"llamacpp","model":"m"}`, "Old label", true},
		{"already keyed", `{"key":"m","name":"Old label","quantization":"q4","engine":"llamacpp","model":"m"}`, "Old label", false},
		{"explicit display", `{"key":"m","displayName":"Chosen label","name":"Old label","quantization":"q4","engine":"llamacpp","model":"m"}`, "Chosen label", false},
		{"case insensitive fields", `{"Key":"m","Name":"Old label","Quantization":"q4","engine":"llamacpp","model":"m"}`, "Old label", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fields, err := modelObject([]byte(tc.profile))
			if err != nil {
				t.Fatal(err)
			}
			if got := repairStoredProfileFields(fields); got != tc.repairedKey {
				t.Fatalf("missing-key repair = %v, want %v", got, tc.repairedKey)
			}
			for key := range fields {
				if strings.EqualFold(key, "name") || strings.EqualFold(key, "quantization") {
					t.Fatalf("obsolete field retained: %s", key)
				}
			}
			raw := []byte(`{"profiles":[` + tc.profile + `]}`)
			before := bytes.Clone(raw)
			p, _, err := ParseStored(raw)
			if err != nil {
				t.Fatal(err)
			}
			want := []Profile{{Key: "m", DisplayName: tc.display, Engine: "llamacpp", Model: "m"}}
			if !reflect.DeepEqual(p.Profiles, want) || !bytes.Equal(raw, before) {
				t.Fatalf("repair changed identity or input: %+v", p.Profiles)
			}
			encoded, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			var saved struct{ Profiles []map[string]json.RawMessage }
			if err := json.Unmarshal(encoded, &saved); err != nil {
				t.Fatal(err)
			}
			if len(saved.Profiles) != 1 || saved.Profiles[0]["name"] != nil || saved.Profiles[0]["quantization"] != nil {
				t.Fatalf("serialized obsolete metadata: %s", encoded)
			}
		})
	}
}

func TestParseStoredPreservesExplicitProfileKeys(t *testing.T) {
	raw := []byte(`{"profiles":[{"key":"small","displayName":"Small context","icon":"code","name":"small","engine":"llamacpp","model":"small"},{"key":"large","displayName":"Large context","name":"large","engine":"llamacpp","model":"large"}]}`)
	p, _, err := ParseStored(raw)
	if err != nil {
		t.Fatal(err)
	}
	if p.Profiles[0].Key != "small" || p.Profiles[0].DisplayName != "Small context" || p.Profiles[0].Icon != "code" || p.Profiles[1].Key != "large" {
		t.Fatalf("explicit profile metadata changed: %+v", p.Profiles)
	}
}

func TestParseStoredRefusesRepairCollisionWithExplicitKey(t *testing.T) {
	raw := []byte(`{"profiles":[{"key":"shared-model","displayName":"First","name":"first","engine":"llamacpp","model":"other-model"},{"name":"second","engine":"strata","model":"shared-model"}]}`)
	_, _, err := ParseStored(raw)
	want := "Profiles share a key after repair: llamacpp/other-model, strata/shared-model. Rename one in this node's policy file before installing."
	if err == nil || err.Error() != want {
		t.Fatalf("mixed collision error = %v, want %q", err, want)
	}
}

func TestStoredPauseModeRepair(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		want      PauseMode
	}{
		{name: "old local pause", raw: `{"revision":"17","availability":"paused","pausedBy":"local"}`, want: PauseImmediate},
		{name: "old peer pause", raw: `{"revision":"17","availability":"paused","pausedBy":"peer"}`, want: PauseImmediate},
		{name: "configured pause", raw: `{"revision":"17","availability":"paused","pauseMode":"configured"}`, want: PauseConfigured},
		{name: "halt", raw: `{"revision":"17","availability":"paused","pauseMode":"immediate"}`, want: PauseImmediate},
		{name: "available", raw: `{"revision":"17","availability":"available"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, _, err := ParseStored([]byte(tc.raw))
			if err != nil {
				t.Fatalf("ParseStored: %v", err)
			}
			if p.PauseMode != tc.want || p.Revision != "17" {
				t.Fatalf("stored policy mode/revision = %q/%q, want %q/17", p.PauseMode, p.Revision, tc.want)
			}
		})
	}
}

func TestModelControlStoredPolicyPreservesValidSiblings(t *testing.T) {
	for _, version := range []string{"1", "2"} {
		t.Run("version "+version, func(t *testing.T) {
			kind := `"kind":"local",`
			if version == "1" {
				kind = ""
			}
			raw := []byte(`{"version":` + version + `,"revision":"42","availability":"available","admission":{"memoryBudgetBytes":12345},"profiles":[{"name":"good","engine":"llamacpp","model":"good"},{"key":"bad-key","displayName":"Bad","name":"bad","engine":"llamacpp","model":"RAW"}],"roles":[{"name":"weak","profiles":["good"]},{"name":"strong","profiles":["bad-key"]}],"tiers":{"weak":[{` + kind + `"engine":"llamacpp","model":"good"},{` + kind + `"engine":"llamacpp","model":"bad\u0085name"}]},"servedModels":[{"engine":"llamacpp","model":"bad\nname"}],"residentModels":[{"engine":"llamacpp","model":"bad\nname"}]}`)
			raw = bytes.ReplaceAll(raw, []byte("RAW"), []byte{0xff})
			routing, err := json.Marshal(stagedPolicy(t).Routing)
			if err != nil {
				t.Fatal(err)
			}
			if version == "2" {
				raw = bytes.Replace(raw, []byte(`"tiers":{"weak":[`), []byte(`"routing":`+string(routing)+`,"tiers":{"weak":[{"kind":"api","deployment":"free-small"},`), 1)
			}
			original := bytes.Clone(raw)
			p, counts, err := ParseStored(raw)
			if err != nil {
				t.Fatal(err)
			}
			wantRevision := "42"
			if version == "1" {
				wantRevision = "0"
			}
			if p.Version != 2 || p.Revision != wantRevision || p.Availability != Available || p.Admission.MemoryBudgetBytes != 12345 {
				t.Fatal("unrelated policy state changed")
			}
			if len(p.Profiles) != 1 || p.Profiles[0].Key != "good" || p.Profiles[0].Model != "good" || len(p.Roles) != 2 || !reflect.DeepEqual(p.Roles[0].Profiles, []string{"good"}) || len(p.Roles[1].Profiles) != 0 {
				t.Fatal("valid model siblings lost")
			}
			if version == "2" {
				if !reflect.DeepEqual(p.Tiers["weak"], []TierEntry{{Kind: EntryAPI, Deployment: "free-small"}}) {
					t.Fatal("valid sibling API entry lost")
				}
			} else if len(p.Tiers["weak"]) != 0 {
				t.Fatal("legacy local entries retained")
			}
			if p.ServedModels == nil || len(p.ServedModels) != 0 || len(p.ResidentModels) != 0 {
				t.Fatal("empty permission or residency changed")
			}
			if counts["tiers"] != 2 {
				t.Fatalf("retired local count=%d, want 2", counts["tiers"])
			}
			for _, category := range []string{"profiles", "servedModels", "residentModels"} {
				if counts[category] != 1 {
					t.Fatalf("%s count=%d, want 1", category, counts[category])
				}
			}
			if !bytes.Equal(raw, original) {
				t.Fatal("input bytes modified")
			}
		})
	}
}

func TestModelControlStoredPolicyRepairsDeploymentFamilies(t *testing.T) {
	original := stagedPolicy(t)
	for _, name := range []string{"disabled allowlist", "exhausted allowlist"} {
		t.Run(name, func(t *testing.T) {
			p := stagedPolicy(t)
			p.Routing.Deployments[0].Model = "bad\nname"
			if name == "exhausted allowlist" {
				p.Tiers["weak"] = append(p.Tiers["weak"], TierEntry{Kind: EntryAPI, Deployment: "paid-big"})
			}
			raw, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			got, counts, err := ParseStored(raw)
			if err != nil {
				t.Fatal(err)
			}
			if got.Revision != "12" || got.Availability != Available || !reflect.DeepEqual(got.Routing.Accounts, original.Routing.Accounts) || !reflect.DeepEqual(got.Routing.Limits, original.Routing.Limits) || !reflect.DeepEqual(got.Routing.Listener, original.Routing.Listener) {
				t.Fatal("unrelated routing configuration changed")
			}
			if len(got.Routing.Deployments) != 1 || got.Routing.Deployments[0].ID != "paid-big" {
				t.Fatal("deployment sibling lost")
			}
			if len(got.Tiers["weak"]) != 0 || len(got.Tiers["medium"]) != 0 || len(got.Tiers["strong"]) != 0 {
				t.Fatal("dependent or dormant API candidates retained")
			}
			if len(got.Routing.Tiers) != 0 {
				t.Fatal("routing rules without API entries retained")
			}
			if counts["deployments"] != 1 || counts["familyReferences"] != 1 || counts["deploymentReferences"] < 2 {
				t.Fatal("missing fixed removal counts")
			}
		})
	}
}

func TestModelControlStoredPolicyConservativeRecovery(t *testing.T) {
	for _, raw := range []string{
		`{"availability":"bad-secret","profiles":[{"key":"bad-key","name":"bad","displayName":"Bad","engine":"llamacpp","model":"bad\nname"}]}`,
		`{"profiles":[{"key":"bad","name":"bad","engine":"llamacpp","model":"good","Model":"bad\nname"}]}`,
		`{"roles":[{"name":"weak","profiles":[]}],"Roles":[]}`,
		`{"servedModels":[{"engine":"llamacpp","model":null}]}`,
		`{"servedModels":[{"engine":"llamacpp","model":"bad\nname","extra":true}]}`,
		`{"profiles":[{"name":"good","engine":"llamacpp","model":"good"},{"name":"bad","displayName":"bad\nname","engine":"llamacpp","model":"bad\nname"}]}`,
		`{"profiles":[{"name":"test","engine":"llamacpp","model":"bad\nname","memoryBytes":-1}]}`,
		`{"roles":[{"name":"Unknown","profiles":[]}]}`,
		`{"profiles":[{"name":"test","engine":"llamacpp","model":"bad\nname"}],"residentModels":[{"engine":"llamacpp","model":"unrelated"}]}`,
		`{"version":2,"tiers":{"coder":[{"engine":"llamacpp","model":"good"}]}}`,
		`{"version":2,"tiers":{"coder":[{"kind":"local","engine":"llamacpp","model":"good"},{"kind":"local","engine":"llamacpp","model":"good"}]}}`,
		`{"tiers":{"coder":[{"kind":"local","engine":"llamacpp","model":"good","contextTokens":-1}]}}`,
		`{"routing":{"tiers":{"Coder":{"bareMode":"local"}}}}`,
		`{"profiles":[{"engine":"llamacpp","model":"coder:fast\ninvalid"}]}`,
	} {
		data := []byte(raw)
		before := bytes.Clone(data)
		if _, _, err := ParseStored(data); err == nil {
			t.Fatal("unrelated malformed policy repaired")
		} else if err.Error() != "stored policy is invalid" {
			t.Fatalf("unsafe recovery diagnostic: %q", err.Error())
		}
		if !bytes.Equal(data, before) {
			t.Fatal("refused load changed input bytes")
		}
	}
}

func TestModelControlStoredPolicyLengthAndDecodedControls(t *testing.T) {
	for _, model := range []string{"", strings.Repeat("x", 513), "a\x00b", "a\x7fb", "a\u009fb", string([]byte{0xff})} {
		// Build the token separately to retain invalid encoding at the raw seam.
		token, err := json.Marshal(model)
		if err != nil {
			t.Fatal(err)
		}
		if model == string([]byte{0xff}) {
			token = []byte{'"', 0xff, '"'}
		}
		raw := append([]byte(`{"profiles":[{"key":"bad-key","name":"bad","displayName":"Bad","engine":"llamacpp","model":`), token...)
		raw = append(raw, []byte(`}]}`)...)
		p, counts, err := ParseStored(raw)
		if err != nil {
			t.Fatal(err)
		}
		if len(p.Profiles) != 0 || counts["profiles"] != 1 || p.Availability != Available {
			t.Fatal("invalid model did not receive entry-wise repair")
		}
	}
}

func TestModelControlStoredPolicyPreservesPausedState(t *testing.T) {
	p, counts, err := ParseStored([]byte(`{"version":2,"revision":"88","availability":"paused","pausedBy":"peer","pause":{"onActive":"cancel","drainTimeoutSeconds":19},"profiles":[{"key":"bad-key","name":"bad","displayName":"Bad","engine":"llamacpp","model":"bad\nname"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if p.Revision != "88" || p.Availability != Paused || p.PausedBy != PausedByPeer || p.Pause.OnActive != CancelActive || p.Pause.DrainTimeoutSeconds != 19 || counts["profiles"] != 1 {
		t.Fatal("model repair changed persisted pause state")
	}
}

func TestModelControlStoredPolicyRejectsPermissionsBeforeFamilyRepair(t *testing.T) {
	for _, tc := range []struct {
		name         string
		remote, paid bool
	}{
		{"paid without remote", false, true},
		{"prohibited remote", true, false},
		{"prohibited remote and paid", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := stagedPolicy(t)
			p.Routing.Deployments[0].Model = "bad\nmodel"
			tr := p.Routing.Tiers["weak"]
			tr.RemoteAllowed, tr.PaidAllowed = tc.remote, tc.paid
			p.Routing.Tiers["weak"] = tr
			raw, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			before := bytes.Clone(raw)
			if _, _, err := ParseStored(raw); err == nil || err.Error() != "stored policy is invalid" {
				t.Fatal("family exhaustion concealed invalid original permissions")
			}
			if !bytes.Equal(raw, before) {
				t.Fatal("refused load changed input")
			}
		})
	}
}

func TestModelControlStoredPolicyPreservesLegacyMemberOrder(t *testing.T) {
	for _, tc := range []struct{ name, revisions, want string }{
		{"canonical then alternate", `"revision":"11","Revision":"22"`, "22"},
		{"alternate then canonical", `"Revision":"22","revision":"11"`, "11"},
		{"same spelling", `"revision":"11","revision":"22"`, "22"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(`{` + tc.revisions + `,"availability":"paused","Availability":"available","idle":{"unloadAfterMinutes":7},"IDLE":{"wakeTimeoutSeconds":23},"routing":{"limits":{"queueSeconds":5},"Limits":{"maxAttempts":3}},"profiles":[{"name":"good","engine":"llamacpp","model":"good"},{"key":"bad-key","name":"bad","displayName":"Bad","engine":"llamacpp","model":"bad\nmodel"}],"roles":[{"name":"weak","profiles":["good"]},{"name":"strong","profiles":["bad-key"]}]}`)
			before := bytes.Clone(raw)
			p, counts, err := ParseStored(raw)
			if err != nil {
				t.Fatal(err)
			}
			if p.Revision != tc.want || p.Availability != Available {
				t.Fatal("sanitation reordered legacy scalar fields")
			}
			if p.Idle.UnloadAfterMinutes != 7 || p.Idle.WakeTimeoutSeconds != 23 {
				t.Fatal("sanitation collapsed legacy object merge")
			}
			if p.Routing.Limits.QueueSeconds != 5 || p.Routing.Limits.MaxAttempts != 3 {
				t.Fatal("sanitation collapsed routing object merge")
			}
			if len(p.Roles) != 2 || !reflect.DeepEqual(p.Roles[0].Profiles, []string{"good"}) || len(p.Roles[1].Profiles) != 0 || counts["profiles"] != 1 {
				t.Fatal("sanitation lost valid model sibling")
			}
			if !bytes.Equal(raw, before) {
				t.Fatal("load changed original document")
			}
		})
	}
}
