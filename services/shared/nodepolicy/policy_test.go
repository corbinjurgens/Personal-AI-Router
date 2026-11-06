// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
)

func TestValidateProfilesByKey(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{
		{Key: "lfm-8k", DisplayName: "LFM 8k", Engine: "llamacpp", Model: "lfm-8k", ContextTokens: 8192},
		{Key: "lfm-32k", DisplayName: "LFM 32k", Engine: "llamacpp", Model: "lfm-32k", ContextTokens: 32768},
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	got, ok := p.ProfileByKey("lfm-32k")
	if !ok || got.ContextTokens != 32768 || got.Model != "lfm-32k" {
		t.Fatalf("profile lookup = %+v, %v", got, ok)
	}
	if _, ok := p.ProfileByKey("missing"); ok {
		t.Fatal("unknown key matched a profile")
	}
	p.Profiles[1].Key = "lfm-8k"
	if err := p.Validate(); err == nil || !strings.Contains(err.Error(), "duplicate key") {
		t.Fatalf("duplicate key error = %v", err)
	}
}

func TestProfileMetadataValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*Profile)
		want string
	}{
		{"missing key", func(p *Profile) { p.Key = "" }, ".key"},
		{"empty display name", func(p *Profile) { p.DisplayName = "" }, ".displayName"},
		{"invalid icon", func(p *Profile) { p.Icon = "unknown" }, ".icon"},
		{"invalid model identity", func(p *Profile) { p.ModelIdentity = &engines.ModelIdentity{Strength: "invented"} }, ".modelIdentity"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := servedResidentPolicy()
			tc.edit(&p.Profiles[0])
			if err := p.Validate(); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("validation error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestProfileDisplayNamesMayRepeat(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{
		{Key: "first", DisplayName: "Shared display", Engine: "llamacpp", Model: "first-model"},
		{Key: "second", DisplayName: "Shared display", Engine: "llamacpp", Model: "second-model"},
	}
	if err := p.Validate(); err != nil {
		t.Fatalf("display names are presentation metadata: %v", err)
	}
}

func TestProfileDisplayNameDefaultsToKey(t *testing.T) {
	p, err := Parse([]byte(`{"profiles":[{"key":"model","engine":"llamacpp","model":"model"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Profiles) != 1 || p.Profiles[0].DisplayName != "model" {
		t.Fatalf("default display name = %+v", p.Profiles)
	}
}

func TestProfileMetadataRawValidation(t *testing.T) {
	for name, patch := range map[string]string{
		"null key":                 `"key":null,"displayName":"Model"`,
		"null display name":        `"key":"model","displayName":null`,
		"null icon":                `"key":"model","displayName":"Model","icon":null`,
		"invalid key encoding":     "\"key\":\"\xff\",\"displayName\":\"Model\"",
		"invalid display encoding": "\"key\":\"model\",\"displayName\":\"\xff\"",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse([]byte(`{"profiles":[{` + patch + `,"engine":"llamacpp","model":"model"}]}`)); err == nil {
				t.Fatal("accepted malformed profile metadata")
			}
		})
	}
}

func TestCopyProfilesDetachesModelIdentity(t *testing.T) {
	p := servedResidentPolicy()
	p.Profiles[0].ModelIdentity = &engines.ModelIdentity{Strength: "declared", GeneralName: "Original", Files: []engines.ModelIdentityFile{{Role: "model", Name: "model.gguf", SizeBytes: 42}}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	copy := copyProfiles(p.Profiles)
	copy[0].ModelIdentity.GeneralName = "Changed"
	copy[0].ModelIdentity.Files[0].Name = "changed.gguf"
	if p.Profiles[0].ModelIdentity.GeneralName != "Original" || p.Profiles[0].ModelIdentity.Files[0].Name != "model.gguf" {
		t.Fatal("profile copy aliases model identity")
	}
}

func TestSavedPauseModeValidation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		state   Availability
		mode    PauseMode
		invalid bool
	}{
		{name: "available without mode", state: Available},
		{name: "configured pause", state: Paused, mode: PauseConfigured},
		{name: "immediate pause", state: Paused, mode: PauseImmediate},
		{name: "available with mode", state: Available, mode: PauseImmediate, invalid: true},
		{name: "unknown mode", state: Paused, mode: "later", invalid: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := Default()
			p.Availability, p.PauseMode = tc.state, tc.mode
			if err := p.Validate(); (err != nil) != tc.invalid {
				t.Fatalf("Validate() = %v, want invalid %t", err, tc.invalid)
			}
		})
	}
}

func TestPauseModeRequestValidation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		state   Availability
		raw     string
		want    PauseMode
		invalid bool
	}{
		{name: "omitted pause", state: Paused, want: PauseConfigured},
		{name: "configured pause", state: Paused, raw: `"configured"`, want: PauseConfigured},
		{name: "immediate pause", state: Paused, raw: `"immediate"`, want: PauseImmediate},
		{name: "omitted resume", state: Available, want: PauseConfigured},
		{name: "null", state: Paused, raw: `null`, invalid: true},
		{name: "empty", state: Paused, raw: `""`, invalid: true},
		{name: "unknown", state: Paused, raw: `"fast"`, invalid: true},
		{name: "number", state: Paused, raw: `1`, invalid: true},
		{name: "object", state: Paused, raw: `{}`, invalid: true},
		{name: "configured resume", state: Available, raw: `"configured"`, invalid: true},
		{name: "immediate resume", state: Available, raw: `"immediate"`, invalid: true},
		{name: "null resume", state: Available, raw: `null`, invalid: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mode, err := ParsePauseMode(tc.state, json.RawMessage(tc.raw))
			if (err != nil) != tc.invalid || mode != tc.want {
				t.Fatalf("mode = %q, error = %v; want %q, invalid %t", mode, err, tc.want, tc.invalid)
			}
		})
	}
}

func TestProfileInferenceSettingsSchemaAndDefaults(t *testing.T) {
	for _, version := range []int{1, 2} {
		for _, mode := range []RequestOptionsMode{"", RequestOptionsDefaults, RequestOptionsEnforced} {
			t.Run(fmt.Sprintf("version %d mode %q", version, mode), func(t *testing.T) {
				p := Default()
				p.Version = version
				p.Profiles = []Profile{{Key: "sample", DisplayName: "sample", Engine: "llamacpp", Model: "sample", RequestOptions: json.RawMessage(`{"temperature":0.5}`), RequestOptionsMode: mode, LocalOnly: true}}
				raw, err := json.Marshal(p)
				if err != nil {
					t.Fatal(err)
				}
				got, err := Parse(raw)
				if err != nil {
					t.Fatal(err)
				}
				if got.Version != 2 || got.Profiles[0].RequestOptionsMode.Effective() != mode.Effective() || !got.Profiles[0].LocalOnly {
					t.Fatalf("settings lost: %+v", got.Profiles[0])
				}
				if mode != RequestOptionsEnforced && got.Profiles[0].RequestOptionsMode != "" {
					t.Fatal("default mode was not omitted")
				}
				encoded, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				again, err := Parse(encoded)
				if err != nil || !reflect.DeepEqual(got, again) {
					t.Fatalf("unstable roundtrip: %v", err)
				}
			})
		}
	}
	t.Run("omitted fields preserve client wins", func(t *testing.T) {
		got, err := Parse([]byte(`{"profiles":[{"key":"sample", "displayName":"sample", "engine":"llamacpp","model":"sample"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if got.Profiles[0].LocalOnly || got.Profiles[0].RequestOptionsMode.Effective() != RequestOptionsDefaults {
			t.Fatal("omitted fields changed behavior")
		}
		got.Profiles[0].LocalOnly = true
		if err := got.Validate(); err != nil {
			t.Fatal(err)
		}
		got.Profiles[0].LocalOnly = false
		if err := got.Validate(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestProfileInferenceSettingsValidation(t *testing.T) {
	test := func(name, settings string, wantError bool) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			_, err := Parse([]byte(`{"profiles":[{"key":"sample", "displayName":"sample", "engine":"llamacpp","model":"sample",` + settings + `}]}`))
			if (err != nil) != wantError {
				t.Fatalf("error=%v, want error=%v", err, wantError)
			}
		})
	}
	test("explicit defaults and false", `"requestOptionsMode":"defaults","localOnly":false`, false)
	test("unknown mode", `"requestOptionsMode":"force"`, true)
	test("mode number", `"requestOptionsMode":1`, true)
	test("mode null", `"requestOptionsMode":null`, true)
	test("flag string", `"localOnly":"true"`, true)
	test("flag number", `"localOnly":1`, true)
	test("flag null", `"localOnly":null`, true)
	test("enforced absent options", `"requestOptionsMode":"enforced"`, true)
	test("enforced empty options", `"requestOptionsMode":"enforced","requestOptions":{}`, true)
	test("enforced null options", `"requestOptionsMode":"enforced","requestOptions":null`, true)
	test("exact duplicate", `"requestOptionsMode":"enforced","requestOptions":{"temperature":1,"temperature":2}`, true)
	test("escaped duplicate", `"requestOptionsMode":"enforced","requestOptions":{"temperature":1,"temper\u0061ture":2}`, true)
	test("case folded duplicate", `"requestOptionsMode":"enforced","requestOptions":{"temperature":1,"TEMPERATURE":2}`, true)
	test("nested duplicate", `"requestOptionsMode":"enforced","requestOptions":{"options":{"seed":1,"SEED":2}}`, true)
	test("unicode folded duplicate", `"requestOptionsMode":"enforced","requestOptions":{"k":1,"\u212a":2}`, true)
	test("nested objects and array leaves", `"requestOptionsMode":"enforced","requestOptions":{"options":{"seed":1},"stop":["a","b"]}`, false)
	test("default duplicate behavior unchanged", `"requestOptions":{"temperature":1,"temperature":2}`, false)
	test("configured model forbidden", `"requestOptionsMode":"enforced","requestOptions":{"MODEL":"other"}`, true)
	test("bounded nesting", `"requestOptionsMode":"enforced","requestOptions":`+strings.Repeat(`{"x":`, 65)+`1`+strings.Repeat(`}`, 65), true)
}

func TestResidentContextBound(t *testing.T) {
	var maximum int
	if err := json.Unmarshal([]byte(fmt.Sprint(int(^uint(0)>>1))), &maximum); err != nil {
		t.Fatal(err)
	}
	test := func(name string, request, bound int, wantError bool) {
		t.Run(name, func(t *testing.T) {
			if err := ValidateResidentContext(request, bound); (err != nil) != wantError {
				t.Fatalf("ValidateResidentContext(%d, %d) = %v, want error %v", request, bound, err, wantError)
			}
		})
	}
	test("exact bound", 4096, 4096, false)
	test("below bound", 2048, 4096, false)
	test("bound plus one", 4097, 4096, true)
	test("absent bound", 4096, 0, true)
	test("negative bound", 4096, -1, true)
	test("absent request", 0, 4096, true)
	test("negative request", -1, 4096, true)
	test("maximum decoded request", maximum, 4096, true)
	test("maximum exact bound without overflow", maximum, maximum, false)
	test("below protocol ceiling above model bound", 8192, 4096, true)
}

func servedResidentPolicy() Policy {
	p := Default()
	p.ServedModels = ServedModels{{Engine: "llamacpp", Model: "m"}}
	p.ResidentModels = ResidentModels{{Engine: "llamacpp", Model: "m"}}
	p.Profiles = []Profile{{Key: "m", DisplayName: "m", Engine: "llamacpp", Model: "m", ContextTokens: 4096,
		MaxOutputTokens: 512, MaxConcurrent: 1, MemoryBytes: 1024,
		Qualification: &QualificationRef{ID: "q", Revision: "1"}}}
	return p
}

func TestReviewR2PolicyModelDefaults(t *testing.T) {
	for _, key := range []string{`model`, `Model`, `MODEL`, `mo\u0064el`, `Mo\u0064el`} {
		t.Run(key, func(t *testing.T) {
			p := servedResidentPolicy()
			if err := p.Validate(); err != nil {
				t.Fatal(err)
			}
			p.Profiles[0].RequestOptions = json.RawMessage(`{"` + key + `":"forbidden"}`)
			if err := p.Validate(); err == nil || !strings.Contains(err.Error(), "requestOptions may not set model") {
				t.Fatalf("model-equivalent defaults validation: %v", err)
			}
		})
	}
}

func TestServedResidentParityFixtures(t *testing.T) {
	data, err := os.ReadFile("testdata/served-resident.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Name          string          `json:"name"`
			Input         json.RawMessage `json:"input"`
			Raw           string          `json:"raw"`
			Valid         bool            `json:"valid"`
			ServedPresent bool            `json:"servedPresent"`
			Served        []ModelRef      `json:"served"`
			Residents     []ModelRef      `json:"residents"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("no parity cases")
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			input := c.Input
			if c.Raw != "" {
				input = []byte(c.Raw)
			}
			p, err := Parse(input)
			if !c.Valid {
				if err == nil {
					t.Fatal("invalid policy accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if (p.ServedModels != nil) != c.ServedPresent || !reflect.DeepEqual([]ModelRef(p.ServedModels), c.Served) || !reflect.DeepEqual([]ModelRef(p.ResidentModels), c.Residents) {
				t.Fatalf("unexpected permission/preference: %+v / %+v", p.ServedModels, p.ResidentModels)
			}
		})
	}
}

func TestServedResidentNull(t *testing.T) {
	for _, field := range []string{"servedModels", "residentModels"} {
		t.Run(field, func(t *testing.T) {
			if _, err := Parse([]byte(`{"` + field + `":null}`)); err == nil {
				t.Fatal("null accepted")
			}
		})
	}
}

func TestServedResidentPolicyOwnedSlices(t *testing.T) {
	caller := servedResidentPolicy()
	stored := caller
	if err := json.Unmarshal([]byte(`{}`), &stored); err != nil {
		t.Fatal(err)
	}
	caller.ServedModels[0].Model = "caller change"
	caller.ResidentModels[0].Model = "caller change"
	if stored.ServedModels[0].Model != "m" || stored.ResidentModels[0].Model != "m" {
		t.Fatal("decoded policy aliases caller slices")
	}
}

func TestServedResidentPolicyRawKeys(t *testing.T) {
	for _, raw := range []string{
		`{"servedModels":[],"servedModels":[]}`, `{"servedModels":[],"ServedModels":[]}`,
		`{"residentModels":[],"ResidentModels":[]}`, `{"residentModels":null}`,
		`{"servedModels":[{"engine":"llamacpp","engine":"llamacpp","model":"m"}]}`,
	} {
		var p Policy
		if err := json.Unmarshal([]byte(raw), &p); err == nil {
			t.Fatalf("policy decoder accepted %s", raw)
		}
	}
}

func TestServedResidentSubset(t *testing.T) {
	p := servedResidentPolicy()
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	p.ServedModels = nil
	if err := p.Validate(); err != nil {
		t.Fatalf("unrestricted residents: %v", err)
	}
	p.ServedModels = ServedModels{}
	if err := p.Validate(); err == nil {
		t.Fatal("resident outside empty served set accepted")
	}
	p.ServedModels = ServedModels{{Engine: "llamacpp", Model: "other"}}
	if err := p.Validate(); err == nil {
		t.Fatal("resident outside served set accepted")
	}
}

func TestServedResidentProfileBounds(t *testing.T) {
	reject := func(name string, edit func(*Policy)) {
		t.Run(name, func(t *testing.T) {
			p := servedResidentPolicy()
			edit(&p)
			if err := p.Validate(); err == nil {
				t.Fatal("invalid resident accepted")
			}
		})
	}
	reject("missing profile", func(p *Policy) { p.Profiles = nil })
	reject("wrong model", func(p *Policy) { p.Profiles[0].Model = "other" })
	reject("wrong engine", func(p *Policy) { p.Profiles[0].Engine = "llamacpp-custom" })
	reject("no context", func(p *Policy) { p.Profiles[0].ContextTokens = 0 })
	reject("no output", func(p *Policy) { p.Profiles[0].MaxOutputTokens = 0 })
	reject("output reaches context", func(p *Policy) { p.Profiles[0].MaxOutputTokens = 4096 })
	reject("no concurrency", func(p *Policy) { p.Profiles[0].MaxConcurrent = 0 })
	reject("unknown memory", func(p *Policy) { p.Profiles[0].MemoryBytes = 0 })
	reject("unqualified", func(p *Policy) { p.Profiles[0].Qualification = nil })
	reject("invalid qualification", func(p *Policy) { p.Profiles[0].Qualification.Revision = "0" })
}

func TestServedResidentListBounds(t *testing.T) {
	for _, field := range []string{"servedModels", "residentModels"} {
		t.Run(field, func(t *testing.T) {
			p := Default()
			var refs []ModelRef
			for i := range 256 {
				model := fmt.Sprintf("m%d", i)
				refs = append(refs, ModelRef{Engine: "llamacpp", Model: model})
				pr := servedResidentPolicy().Profiles[0]
				pr.Model, pr.Key, pr.DisplayName = model, model, model
				p.Profiles = append(p.Profiles, pr)
			}
			set := func() {
				if field == "servedModels" {
					p.ServedModels = ServedModels(refs)
				} else {
					p.ResidentModels = ResidentModels(refs)
				}
			}
			set()
			if err := p.Validate(); err != nil {
				t.Fatalf("256 entries: %v", err)
			}
			refs = append(refs, ModelRef{Engine: "llamacpp", Model: "extra"})
			set()
			if err := p.Validate(); err == nil {
				t.Fatal("257 entries accepted")
			}
			raw, err := json.Marshal(refs)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := Parse([]byte(`{"` + field + `":` + string(raw) + `}`)); err == nil {
				t.Fatal("oversized JSON list accepted")
			}
		})
	}
}

func TestServedResidentPolicyRoundTrip(t *testing.T) {
	test := func(name, input string, present bool) {
		t.Run(name, func(t *testing.T) {
			p, err := Parse([]byte(input))
			if err != nil {
				t.Fatal(err)
			}
			for range 3 {
				raw, err := json.Marshal(p)
				if err != nil {
					t.Fatal(err)
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(raw, &fields); err != nil {
					t.Fatal(err)
				}
				_, has := fields["servedModels"]
				if has != present {
					t.Fatalf("served presence = %v, want %v", has, present)
				}
				next, err := Parse(raw)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(p, next) {
					t.Fatal("unstable policy round trip")
				}
				p = next
			}
		})
	}
	test("absent", `{}`, false)
	test("empty", `{"servedModels":[],"residentModels":[]}`, true)
	test("restricted", `{"servedModels":[{"engine":"llamacpp","model":"m"}]}`, true)
	raw, err := json.Marshal(servedResidentPolicy())
	if err != nil {
		t.Fatal(err)
	}
	test("qualified subset", string(raw), true)
}

func TestServedResidentLegacyDefaults(t *testing.T) {
	for _, raw := range []string{`{}`, v1Document, `{"version":2}`} {
		p, err := Parse([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}
		if p.Version != 2 || p.ServedModels != nil || len(p.ResidentModels) != 0 {
			t.Fatal("legacy semantics changed")
		}
		data, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			t.Fatal(err)
		}
		if fields["servedModels"] != nil || fields["residentModels"] != nil {
			t.Fatal("legacy document gained fields")
		}
	}
}

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

// TestPauseOptionsNormalizeForOldDocumentsAndPeerSets accepts the obsolete
// stopEngines/unloadModels bools from files and older peers and reads both back
// as true, keeping onActive and the drain timeout as written.
func TestPauseOptionsNormalizeForOldDocumentsAndPeerSets(t *testing.T) {
	const legacyPause = `{"onActive":"cancel","drainTimeoutSeconds":7,"unloadModels":false,"stopEngines":false}`
	want := PauseConfig{OnActive: CancelActive, DrainTimeoutSeconds: 7, UnloadModels: true, StopEngines: true}
	requireCanonical := func(t *testing.T, got Policy, want PauseConfig) {
		t.Helper()
		if got.Pause != want {
			t.Fatalf("pause = %+v, want %+v", got.Pause, want)
		}
		data, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var doc struct {
			Pause map[string]json.RawMessage `json:"pause"`
		}
		if err := json.Unmarshal(data, &doc); err != nil {
			t.Fatalf("decode marshaled policy: %v", err)
		}
		for _, key := range []string{"unloadModels", "stopEngines"} {
			if string(doc.Pause[key]) != "true" {
				t.Fatalf("serialized pause.%s = %s, want true", key, doc.Pause[key])
			}
		}
	}
	parse := func(name, doc string, want PauseConfig) {
		t.Run(name, func(t *testing.T) {
			got, err := Parse([]byte(doc))
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			requireCanonical(t, got, want)
		})
	}
	parse("version 2 file", `{"version":2,"pause":`+legacyPause+`}`, want)
	parse("version 1 file", `{"version":1,"pause":`+legacyPause+`}`, want)
	parse("omitted pause", `{"version":2}`, Default().Pause)

	t.Run("direct Policy.UnmarshalJSON", func(t *testing.T) {
		var got Policy
		if err := json.Unmarshal([]byte(`{"version":2,"pause":`+legacyPause+`}`), &got); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		requireCanonical(t, got, want)
	})
	t.Run("older peer policy:set", func(t *testing.T) {
		stored := Default()
		merged, err := MergePeerSet(stored, peerDocument(t, stored, func(doc map[string]any) {
			doc["pause"] = map[string]any{"onActive": "cancel", "drainTimeoutSeconds": 7, "unloadModels": false, "stopEngines": false}
		}))
		if err != nil {
			t.Fatalf("MergePeerSet: %v", err)
		}
		requireCanonical(t, merged, want)
	})
	t.Run("default", func(t *testing.T) {
		if got := Default().Pause; !got.UnloadModels || !got.StopEngines {
			t.Fatalf("default pause = %+v, want both switches true", got)
		}
	})
	reject := func(name, pause string) {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse([]byte(`{"version":2,"pause":` + pause + `}`)); err == nil {
				t.Fatalf("Parse accepted pause %s", pause)
			}
		})
	}
	reject("rejects a non-bool switch", `{"onActive":"finish","drainTimeoutSeconds":7,"unloadModels":"no","stopEngines":true}`)
	reject("rejects an unknown onActive", `{"onActive":"later","drainTimeoutSeconds":7,"unloadModels":true,"stopEngines":true}`)
	reject("rejects a negative drain timeout", `{"onActive":"finish","drainTimeoutSeconds":-1,"unloadModels":true,"stopEngines":true}`)
}

// A descriptor engine ID is a policy identity like a built-in name: profiles
// and roles may name it whether or not this node runs it.
func TestValidateAcceptsDescriptorEngineIDs(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{{Key: "custom", DisplayName: "custom", Engine: "llamacpp-custom", Model: "org/model:Q4_K_M"}}
	p.Profiles = append(p.Profiles, Profile{Key: "strata", DisplayName: "Strata", Engine: "strata", Model: "qwen-next"})
	p.Roles = []Role{{Name: "strong", Profiles: []string{"custom", "strata"}}}
	if err := p.Validate(); err != nil {
		t.Fatalf("descriptor engine ids rejected: %v", err)
	}
}

func TestValidateRejects(t *testing.T) {
	cases := map[string]func(*Policy){
		"draining saved": func(p *Policy) { p.Availability = Draining },
		"unknown engine": func(p *Policy) {
			p.Profiles = []Profile{{Key: "a", DisplayName: "a", Engine: "no/pe", Model: "m"}}
		},
		"overlong engine": func(p *Policy) {
			p.Profiles = []Profile{{Key: "a", DisplayName: "a", Engine: strings.Repeat("e", engines.MaxDynamicIDLen+1), Model: "m"}}
		},
		"duplicate profile key": func(p *Policy) {
			p.Profiles = []Profile{{Key: "a", DisplayName: "a", Engine: "llamacpp", Model: "m"}, {Key: "a", DisplayName: "b", Engine: "llamacpp", Model: "n"}}
		},
		"two profiles model": func(p *Policy) {
			p.Profiles = []Profile{{Key: "a", DisplayName: "a", Engine: "llamacpp", Model: "m"}, {Key: "b", DisplayName: "b", Engine: "llamacpp", Model: "m"}}
		},
		"unknown tier": func(p *Policy) {
			*p = stagedPolicy(t)
			p.Tiers["Coder"] = []TierEntry{{Kind: EntryAPI, Deployment: "free-small"}}
		},
		"tier as model": func(p *Policy) {
			p.ServedModels = ServedModels{{Engine: "llamacpp", Model: "strong:local"}}
		},
		"unknown capability": func(p *Policy) {
			p.Profiles = []Profile{{Key: "m", DisplayName: "Model", Engine: "llamacpp", Model: "m", Capabilities: []Capability{"telepathy"}}}
		},
		"options not object": func(p *Policy) {
			p.Profiles = []Profile{{Key: "a", DisplayName: "a", Engine: "llamacpp", Model: "m", RequestOptions: json.RawMessage(`[1]`)}}
		},
		"options set model": func(p *Policy) {
			p.Profiles = []Profile{{Key: "a", DisplayName: "a", Engine: "llamacpp", Model: "m", RequestOptions: json.RawMessage(`{"model":"x"}`)}}
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

func TestProfileFor(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{{Key: "q", DisplayName: "q", Engine: "llamacpp", Model: "qwen3-8b", MaxConcurrent: 2}}
	if pr, ok := p.ProfileFor("llamacpp", "qwen3-8b"); !ok || pr.MaxConcurrent != 2 {
		t.Fatalf("ProfileFor = %+v, %v", pr, ok)
	}
	if _, ok := p.ProfileFor("llamacpp-custom", "qwen3-8b"); ok {
		t.Fatal("ProfileFor matched the wrong engine")
	}
}

func TestRoleNamesPreserveDeclarationOrder(t *testing.T) {
	p := Default()
	if len(p.RoleNames()) != 0 {
		t.Fatal("default supplied roles")
	}
	p.Roles = []Role{{Name: "deep"}, {Name: "coder"}, {Name: "fast"}}
	if got := p.RoleNames(); !reflect.DeepEqual(got, []string{"deep", "coder", "fast"}) {
		t.Fatalf("role order = %v", got)
	}
}

func TestNativeAndServedModelNamespaces(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{{Key: "lfm-8k", DisplayName: "LFM", Engine: "llamacpp", Model: "coder:local"}}
	if err := p.Validate(); err != nil {
		t.Fatalf("native target: %v", err)
	}
	p.ServedModels = []ModelRef{{Engine: "llamacpp", Model: "coder:local"}}
	if err := p.Validate(); err == nil || !strings.Contains(err.Error(), "reserved alias") {
		t.Fatalf("served target: %v", err)
	}
}

func TestBareFormerRoleIsValidProfileKey(t *testing.T) {
	p := Default()
	p.Profiles = []Profile{{Key: "weak", DisplayName: "Weak", Engine: "llamacpp", Model: "native"}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	p.Roles = []Role{{Name: "weak", Profiles: []string{}}}
	if err := p.Validate(); err == nil {
		t.Fatal("accepted same-node role and profile collision")
	}
}

func TestArbitraryAPIRoleKeys(t *testing.T) {
	for _, name := range []string{"coder", "Coder"} {
		t.Run(name, func(t *testing.T) {
			p := stagedPolicy(t)
			p.Tiers[name] = []TierEntry{{Kind: EntryAPI, Deployment: "free-small"}}
			p.Routing.Tiers["coder"] = TierRouting{BareMode: BareHybrid}
			if err := p.Validate(); (err == nil) != (name == "coder") {
				t.Fatalf("API role key: %v", err)
			}
		})
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
  "profiles": [{"key": "qwen", "displayName": "qwen", "engine": "llamacpp", "model": "qwen3-8b",
    "contextTokens": 16384, "capabilities": ["tools"], "memoryBytes": 5000000000, "maxConcurrent": 2,
    "requestOptions": {  "options": { "num_ctx": 16384 }  }}],
  "roles": [{"name":"weak","profiles":["qwen"]}],
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
		Pause:        PauseConfig{OnActive: CancelActive, DrainTimeoutSeconds: 7, UnloadModels: true, StopEngines: true},
		Admission: Admission{
			MaxResidentModels: 2, MaxConcurrentPerModel: 3, QueueTimeoutSeconds: 11,
			MemoryBudgetBytes: 8589934592, SwitchModels: false, SwitchTimeoutSeconds: 13,
		},
		Idle: IdleConfig{UnloadAfterMinutes: 5, StopEngineAfterMinutes: 9, StartOnDemand: false, WakeTimeoutSeconds: 17},
		Profiles: []Profile{{
			Key: "qwen", DisplayName: "qwen", Engine: "llamacpp", Model: "qwen3-8b", ContextTokens: 16384,
			Capabilities: []Capability{CapTools}, MemoryBytes: 5000000000, MaxConcurrent: 2,
			RequestOptions: json.RawMessage(`{"options":{"num_ctx":16384}}`),
		}},
		Roles:      []Role{{Name: "weak", Profiles: []string{"qwen"}}},
		Tiers:      map[string][]TierEntry{},
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
	if len(p.Routing.Tiers) != 0 {
		t.Fatalf("migration supplied role rules: %+v", p.Routing.Tiers)
	}
}

func TestParseIsIdempotent(t *testing.T) {
	inputs := map[string]string{
		"empty v2":           `{}`,
		"minimal v1":         `{"version":1}`,
		"full v1":            v1Document,
		"v1 with null tier":  `{"version":1,"tiers":{"weak":null,"medium":[]}}`,
		"v2 empty lists":     `{"version":2,"profiles":[{"key":"m","displayName":"Model","engine":"llamacpp","model":"m","capabilities":[]}],"roles":[{"name":"weak","profiles":["m"]}]}`,
		"v2 spaced options":  `{"version":2,"profiles":[{"key":"a", "displayName":"a", "engine":"llamacpp","model":"m","capabilities":[],"requestOptions": { "keep_alive" : "10m" }}]}`,
		"v2 with revision":   `{"version":2,"revision":"9007199254740993"}`,
		"v2 with api config": stagedRoutingDocument,
		"v1 html options":    `{"version":1,"profiles":[{"key":"a", "displayName":"a", "engine":"llamacpp","model":"m","requestOptions":{"stop":["</s>"]}}]}`,
		"v2 html options":    `{"version":2,"profiles":[{"key":"a", "displayName":"a", "engine":"llamacpp","model":"m","requestOptions":{"q":"a&b","c":"<x>"}}]}`,
		"v2 escaped options": `{"version":2,"profiles":[{"key":"a", "displayName":"a", "engine":"llamacpp","model":"m","requestOptions":{"q":"a\u0026b \u003c"}}]}`,
		"v2 null roles":      `{"version":2,"roles":null}`,
		"v2 with roles":      `{"version":2,"roles":[{"name":"weak","profiles":[]},{"name":"strong","profiles":[]}]}`,
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

func TestParseStoredRetiresLegacyLocalEntries(t *testing.T) {
	for _, entries := range []string{
		`{"engine":"llamacpp","model":"m","contextTokens":100},{"engine":"llamacpp-custom","model":"m"},{"engine":"llamacpp","model":"m","contextTokens":100}`,
		`{"engine":"llamacpp","model":"m","capabilities":["tools","structured"]},{"engine":"llamacpp","model":"m","capabilities":["structured","tools"]}`,
		`{"engine":"llamacpp","model":"m","contextTokens":100},{"engine":"llamacpp","model":"m","contextTokens":32000}`,
	} {
		t.Run(entries, func(t *testing.T) {
			raw := []byte(`{"version":1,"availability":"paused","profiles":[{"key":"keep","displayName":"weak","engine":"llamacpp","model":"m"}],"tiers":{"weak":[` + entries + `]}}`)
			p, _, err := ParseStored(raw)
			if err != nil {
				t.Fatal(err)
			}
			want := []Profile{{Key: "keep", DisplayName: "weak", Engine: "llamacpp", Model: "m"}}
			if len(p.Tiers["weak"]) != 0 || len(p.Roles) != 0 || !reflect.DeepEqual(p.Profiles, want) || p.Availability != Paused {
				t.Fatalf("retirement changed retained profile or invented declarations: %+v", p)
			}
		})
	}
}

func TestParseRejectsDuplicateTierEntryInV2(t *testing.T) {
	p := stagedPolicy(t)
	p.Tiers["coder"] = []TierEntry{{Kind: EntryAPI, Deployment: "free-small"}, {Kind: EntryAPI, Deployment: "free-small"}}
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(raw); err == nil || !strings.Contains(err.Error(), "duplicate entry") {
		t.Fatalf("duplicate API entry error = %v", err)
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
	if len(p.Routing.Tiers) != 0 {
		t.Errorf("tier settings = %v, want no role rules", p.Routing.Tiers)
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
	doc := `{"version":2,"tiers":{"weak":[{"engine":"llamacpp","model":"m"}]}}`
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

func TestParseV2APIEntryIdentity(t *testing.T) {
	t.Run("different deployments remain ordered", func(t *testing.T) {
		p := stagedPolicy(t)
		want := []TierEntry{{Kind: EntryAPI, Deployment: "paid-big"}, {Kind: EntryAPI, Deployment: "free-small"}}
		p.Tiers["coder"] = want
		raw, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		got, err := Parse(raw)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Tiers["coder"], want) {
			t.Fatalf("API entry order = %+v, want %+v", got.Tiers["coder"], want)
		}
	})
	t.Run("local bounds cannot disguise API entries", func(t *testing.T) {
		for _, entry := range []TierEntry{
			{Kind: EntryAPI, Deployment: "free-small", ContextTokens: 100},
			{Kind: EntryAPI, Deployment: "free-small", Capabilities: []Capability{CapTools}},
		} {
			p := stagedPolicy(t)
			p.Tiers["coder"] = []TierEntry{entry}
			if err := p.Validate(); err == nil || !strings.Contains(err.Error(), "names only a deployment") {
				t.Fatalf("API local fields error = %v", err)
			}
		}
	})
}

func TestValidateRoleLists(t *testing.T) {
	for _, tc := range []struct {
		name    string
		roles   []Role
		invalid bool
	}{
		{"no roles", []Role{}, false},
		{"valid empty medium", []Role{{Name: "medium", Profiles: []string{}}}, false},
		{"ordered shared keys", []Role{{Name: "strong", Profiles: []string{"second", "first"}}, {Name: "weak", Profiles: []string{"first"}}}, false},
		{"unknown key", []Role{{Name: "weak", Profiles: []string{"missing"}}}, true},
		{"duplicate key", []Role{{Name: "weak", Profiles: []string{"first", "first"}}}, true},
		{"duplicate name", []Role{{Name: "medium"}, {Name: "medium"}}, true},
		{"arbitrary empty role", []Role{{Name: "coder", Profiles: []string{}}}, false},
		{"duplicate arbitrary role", []Role{{Name: "coder"}, {Name: "coder"}}, true},
		{"uppercase role", []Role{{Name: "Coder"}}, true},
		{"profile key collision", []Role{{Name: "first"}}, true},
		{"suffixed alias", []Role{{Name: "weak:local"}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := Default()
			p.Profiles = []Profile{{Key: "first", DisplayName: "First", Engine: "llamacpp", Model: "m"}, {Key: "second", DisplayName: "Second", Engine: "llamacpp-custom", Model: "n"}}
			p.Roles = tc.roles
			if err := p.Validate(); (err != nil) != tc.invalid {
				t.Fatalf("Validate() = %v, invalid=%v", err, tc.invalid)
			}
		})
	}
	for _, tc := range []struct{ name, engine, model, diagnostic string }{
		{"unknown engine", "no:pe", "m", "unknown engine"},
		{"missing model", "llamacpp", "", "model is required"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := Default()
			p.Profiles = []Profile{{Key: "selected", DisplayName: "Selected", Engine: tc.engine, Model: tc.model}}
			p.Roles = []Role{{Name: "weak", Profiles: []string{"selected"}}}
			if err := p.Validate(); err == nil || !strings.Contains(err.Error(), tc.diagnostic) {
				t.Fatalf("Validate() = %v, want %q", err, tc.diagnostic)
			}
		})
	}
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
	p, _, err := ParseStored([]byte(`{"version":2,"availability":"paused","tiers":{"weak":[{"kind":"local","engine":"llamacpp","model":"m"}]}}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if p.Roles == nil || len(p.Roles) != 0 {
		t.Errorf("roles = %#v, want an empty list", p.Roles)
	}
	if p.PausedBy != "" {
		t.Errorf("pausedBy = %q, want empty", p.PausedBy)
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(data), `"roles":[]`) || strings.Contains(string(data), "pausedBy") {
		t.Errorf("marshaled policy = %s, want an empty roles list and no pausedBy", data)
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
		"roles":[{"name":"weak","profiles":[]},{"name":"strong","profiles":[]}]}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	want := []Role{{Name: "weak", Profiles: []string{}}, {Name: "strong", Profiles: []string{}}}
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
			pr.Key, pr.DisplayName = "m", "m"
			pr.Engine, pr.Model = "llamacpp", "m"
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
	doc := `{"version":2,"profiles":[{"key":"m", "displayName":"m", "engine":"llamacpp","model":"m","contextTokens":8192,
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
