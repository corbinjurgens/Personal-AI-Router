// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestModelControlPolicyWritesSelectedModels(t *testing.T) {
	for _, owner := range []string{"profiles", "roles", "tiers", "deployments"} {
		t.Run(owner, func(t *testing.T) {
			test := func(name, model string, wantOK bool) {
				t.Run(name, func(t *testing.T) {
					p := Default()
					switch owner {
					case "profiles":
						p.Profiles = []Profile{{Name: "test", Engine: "ollama", Model: model}}
					case "roles":
						p.Roles[TierWeak] = Role{Engine: "ollama", Model: model}
					case "tiers":
						p.Tiers[TierWeak] = []TierEntry{{Kind: EntryLocal, Engine: "ollama", Model: model}}
					case "deployments":
						p = stagedPolicy(t)
						p.Routing.Deployments[0].Model = model
					}
					if err := p.Validate(); (err == nil) != wantOK {
						t.Fatalf("typed validation accepted=%t, want %t", err == nil, wantOK)
					}
					raw, err := json.Marshal(p)
					if err != nil {
						t.Fatal(err)
					}
					if _, err := Parse(raw); (err == nil) != wantOK {
						t.Fatalf("Parse accepted=%t, want %t", err == nil, wantOK)
					}
				})
			}
			test("512 bytes", strings.Repeat("x", 512), true)
			test("513 bytes", strings.Repeat("x", 513), false)
			test("unknown Unicode name", "unknown/模型 with spaces:v1", true)
			test("control before trim", "\nmodel", false)
			for r := rune(0); r <= 0x9f; r++ {
				if r >= 0x20 && r < 0x7f {
					continue
				}
				test(fmt.Sprintf("control U+%04X", r), "sentinel"+string(r)+"model", false)
			}
		})
	}
}

func TestModelControlPolicyWritesIdentity(t *testing.T) {
	digest := strings.Repeat("a", 64)
	i := EngineIdentity{EngineRunID: "run", SettingsFingerprint: digest, LaunchDigest: digest, ModelDigests: map[string]string{"unknown/模型": digest}}
	if err := i.Validate(); err != nil {
		t.Fatal(err)
	}
	i.ModelDigests = map[string]string{"bad\nmodel": digest}
	if err := i.Validate(); err == nil {
		t.Fatal("typed identity accepted control-bearing model")
	}
	raw := []byte(`{"engineRunId":"run","modelDigests":{"RAW":"` + digest + `"}}`)
	raw = bytes.ReplaceAll(raw, []byte("RAW"), []byte{0xff})
	if err := json.Unmarshal(raw, &i); err == nil {
		t.Fatal("identity decoder replaced invalid model encoding")
	}
}

func TestModelControlAdvertisementStrictModelSiblings(t *testing.T) {
	id, err := json.Marshal(residentIdentity())
	if err != nil {
		t.Fatal(err)
	}
	entry := `{"engine":"ollama","model":"MODEL","maxConcurrent":1,"identity":IDENTITY}`
	entry = strings.Replace(entry, "IDENTITY", string(id), 1)
	good := strings.Replace(entry, "MODEL", "good", 1)
	bad := strings.Replace(entry, "MODEL", `bad\nmodel`, 1)
	var a Advertisement
	if err := json.Unmarshal([]byte(`{"strictAdmission":{"version":1,"models":[`+good+`,`+bad+`]}}`), &a); err != nil {
		t.Fatal(err)
	}
	if a.StrictAdmission.State() != StrictCapabilityValid || len(a.StrictAdmission.Models) != 1 || a.StrictAdmission.Models[0].Model != "good" {
		t.Fatal("strict model sanitation lost valid sibling")
	}
	if err := json.Unmarshal([]byte(`{"servedModels":[],"strictAdmission":{"version":1,"models":[`+bad+`]}}`), &a); err != nil {
		t.Fatal(err)
	}
	if a.StrictAdmission != nil || a.ServedModels == nil || len(a.ServedModels) != 0 {
		t.Fatal("empty capability remained eligible or permission broadened")
	}
}

func TestModelControlPolicyWritesRawEncoding(t *testing.T) {
	for _, raw := range []string{
		`{"roles":{"weak":{"engine":"ollama","model":"RAW"}}}`,
		`{"profiles":[{"name":"test","engine":"ollama","model":"RAW"}]}`,
		`{"tiers":{"weak":[{"kind":"local","engine":"ollama","model":"RAW"}]}}`,
		`{"servedModels":[{"engine":"ollama","model":"RAW"}]}`,
		`{"residentModels":[{"engine":"ollama","model":"RAW"}]}`,
		strings.Replace(stagedRoutingDocument, "small-1", "RAW", 1),
	} {
		input := bytes.ReplaceAll([]byte(raw), []byte("RAW"), []byte{0xff})
		if _, err := Parse(input); err == nil {
			t.Fatal("raw invalid encoding accepted by policy parser")
		}
	}
}

func TestModelControlAdvertisementFiltersBeforeStrictDecode(t *testing.T) {
	raw := []byte(`{"revision":"7","servedModels":[{"engine":"ollama","model":"good"},{"engine":"ollama","model":"bad\nname"}],"roles":{"weak":{"engine":"ollama","model":"good"},"strong":{"engine":"ollama","model":"RAW"}}}`)
	raw = bytes.ReplaceAll(raw, []byte("RAW"), []byte{0xff})
	var a Advertisement
	if err := json.Unmarshal(raw, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.ServedModels) != 1 || a.ServedModels[0].Model != "good" || len(a.Roles) != 1 || a.Roles[TierWeak].Model != "good" || a.Revision != "7" {
		t.Fatal("valid advertisement siblings lost")
	}
	if err := json.Unmarshal([]byte(`{"servedModels":[{"engine":"ollama","model":"bad\nname"}]}`), &a); err != nil {
		t.Fatal(err)
	}
	if a.ServedModels == nil || len(a.ServedModels) != 0 {
		t.Fatal("all-invalid permission became unrestricted")
	}
}

func TestModelControlAdvertisementUnsupportedCapabilityOpaque(t *testing.T) {
	var a Advertisement
	raw := []byte(`{"strictAdmission":{"version":999,"models":[{"model":"bad\nname"}],"future":true}}`)
	if err := json.Unmarshal(raw, &a); err != nil {
		t.Fatal(err)
	}
	if a.StrictAdmission.State() != StrictCapabilityUnsupported {
		t.Fatal("future version reinterpreted")
	}
	encoded, err := json.Marshal(a.StrictAdmission)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Version int
		Models  []struct{ Model string }
		Future  bool
	}
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	if got.Version != 999 || !got.Future || len(got.Models) != 1 || got.Models[0].Model != "bad\nname" {
		t.Fatal("opaque capability changed")
	}
}

func TestModelControlPolicyWritesDirectCapabilityRawTokens(t *testing.T) {
	id, err := json.Marshal(residentIdentity())
	if err != nil {
		t.Fatal(err)
	}
	entry := `{"engine":"ollama","model":"PRIVATE-MODEL","maxConcurrent":1,"identity":` + string(id) + `}`
	for _, tc := range []struct{ name, entry string }{
		{"invalid UTF-8", strings.Replace(entry, "PRIVATE-MODEL", string([]byte{0xff}), 1)},
		{"duplicate model before valid", strings.Replace(entry, `"model":"PRIVATE-MODEL"`, `"model":"PRIVATE-MODEL\n","model":"good"`, 1)},
		{"duplicate model after valid", strings.Replace(entry, `"model":"PRIVATE-MODEL"`, `"model":"good","model":"PRIVATE-MODEL\n"`, 1)},
		{"case-folded model", strings.Replace(entry, `"model":"PRIVATE-MODEL"`, `"model":"good","Model":"PRIVATE-MODEL"`, 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(`{"version":1,"models":[` + tc.entry + `]}`)
			var direct StrictAdmissionCapability
			if err := json.Unmarshal(raw, &direct); err == nil {
				t.Fatal("direct capability repaired invalid raw model")
			} else if strings.Contains(err.Error(), "PRIVATE-MODEL") {
				t.Fatal("raw model leaked in direct error")
			}
			var readiness StrictAdmissionReadiness
			envelope := append([]byte(`{"policyRevision":"1","proxyRunId":"proxy-1","capability":`), raw...)
			envelope = append(envelope, '}')
			if err := json.Unmarshal(envelope, &readiness); err == nil {
				t.Fatal("readiness decode repaired invalid raw model")
			}
		})
	}
}

func TestModelControlAdvertisementNumericVersions(t *testing.T) {
	id, err := json.Marshal(residentIdentity())
	if err != nil {
		t.Fatal(err)
	}
	for _, spelling := range []string{"1", "1.0", "1e0"} {
		t.Run(spelling, func(t *testing.T) {
			raw := []byte(`{"version":` + spelling + `,"models":[{"engine":"ollama","model":"good","maxConcurrent":1,"identity":` + string(id) + `}]}`)
			var direct StrictAdmissionCapability
			if err := json.Unmarshal(raw, &direct); err != nil {
				t.Fatal(err)
			}
			if direct.State() != StrictCapabilityValid || direct.Version != 1 {
				t.Fatal("integral version rejected")
			}
			var advertised Advertisement
			if err := json.Unmarshal(append(append([]byte(`{"strictAdmission":`), raw...), '}'), &advertised); err != nil {
				t.Fatal(err)
			}
			if advertised.StrictAdmission.State() != StrictCapabilityValid || advertised.StrictAdmission.Models[0].Model != "good" {
				t.Fatal("advertisement lost integral-version model")
			}
			encoded, err := json.Marshal(advertised.StrictAdmission)
			if err != nil {
				t.Fatal(err)
			}
			var canonical struct {
				Version int `json:"version"`
			}
			if err := json.Unmarshal(encoded, &canonical); err != nil {
				t.Fatal(err)
			}
			if canonical.Version != 1 {
				t.Fatal("supported version not canonicalized")
			}
		})
	}
	for _, spelling := range []string{"999", "999.0", "9.99e2"} {
		t.Run("opaque "+spelling, func(t *testing.T) {
			raw := []byte(`{"version":` + spelling + `,"models":"opaque","future":true}`)
			var advertised Advertisement
			if err := json.Unmarshal(append(append([]byte(`{"strictAdmission":`), raw...), '}'), &advertised); err != nil {
				t.Fatal(err)
			}
			if advertised.StrictAdmission.State() != StrictCapabilityUnsupported {
				t.Fatal("opaque capability interpreted")
			}
			encoded, err := advertised.StrictAdmission.MarshalJSON()
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(encoded, raw) {
				t.Fatal("opaque numeric spelling or payload changed")
			}
		})
	}
}

func TestModelControlAdvertisementLegacyOuterFields(t *testing.T) {
	var got Advertisement
	raw := []byte(`{"revision":"1","Revision":"2","ROLES":{"weak":{"ENGINE":"ollama","MODEL":"good"}},"PAUSEDBY":"peer"}`)
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.Revision != "2" || got.Roles[TierWeak].Model != "good" || got.PausedBy != PausedByPeer {
		t.Fatal("legacy case-fold or document-order semantics changed")
	}
	if err := json.Unmarshal([]byte(`{"strictAdmission":{"version":999},"StrictAdmission":{"version":1}}`), &got); err == nil || !strings.Contains(err.Error(), "must use the exact field name strictAdmission") {
		t.Fatal("exact strict field error lost precedence")
	}
}

func TestModelControlPolicyWritesIdentityWhitespace(t *testing.T) {
	digest := strings.Repeat("a", 64)
	for _, model := range []string{"", " ", "\u2003"} {
		i := EngineIdentity{EngineRunID: "run", SettingsFingerprint: digest, LaunchDigest: digest, ModelDigests: map[string]string{model: digest}}
		p := SetModelIdentityParams{Engines: map[string]EngineIdentity{"ollama": i}}
		if err := p.Validate(); err == nil || !strings.Contains(err.Error(), "modelDigests contains an empty model") {
			t.Fatal("identity whitespace-empty compatibility changed")
		}
	}
}

func TestModelControlPolicyWritesMissingModelCompatibility(t *testing.T) {
	_, err := Parse([]byte(`{"roles":{"weak":{"engine":"ollama"}}}`))
	if err == nil || !strings.Contains(err.Error(), "model is required") {
		t.Fatal("missing-model error compatibility changed")
	}
}

func TestModelControlPolicyWritesPermissionDuplicateDiagnosticFirst(t *testing.T) {
	for _, tc := range []struct{ name, raw, diagnostic string }{
		{"served null then empty", `{"servedModels":null,"servedModels":[]}`, "duplicate or case-variant servedModels key"},
		{"resident null then empty", `{"residentModels":null,"residentModels":[]}`, "duplicate or case-variant residentModels key"},
		{"served empty then permission", `{"servedModels":[],"servedModels":[{"engine":"ollama","model":"good"}]}`, "duplicate or case-variant servedModels key"},
		{"served diagnostic before invalid role", `{"roles":{"weak":{"model":false}},"servedModels":null,"servedModels":[]}`, "duplicate or case-variant servedModels key"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := CheckModelListKeys([]byte(tc.raw)); err == nil || err.Error() != tc.diagnostic {
				t.Fatal("preprocessing lost original permission-key diagnostic")
			}
			if _, err := Parse([]byte(tc.raw)); err == nil || !strings.Contains(err.Error(), tc.diagnostic) {
				t.Fatal("parser lost original permission-key diagnostic")
			}
		})
	}
}

func TestModelControlPolicyWritesLegacyRevisionOrder(t *testing.T) {
	for _, tc := range []struct{ name, revisions, want string }{
		{"canonical then alternate", `"revision":"11","Revision":"22"`, "22"},
		{"alternate then canonical", `"Revision":"22","revision":"11"`, "11"},
		{"same spelling", `"revision":"11","revision":"22"`, "22"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(`{` + tc.revisions + `,"roles":{"weak":{"engine":"ollama","model":"good"}}}`)
			if err := CheckModelListKeys(raw); err != nil {
				t.Fatal(err)
			}
			p, err := Parse(raw)
			if err != nil {
				t.Fatal(err)
			}
			if p.Revision != tc.want || p.Roles[TierWeak].Model != "good" {
				t.Fatal("legacy revision order or model sibling changed")
			}
		})
	}
}

func TestModelControlPolicyWritesAmbiguousModelContainersRemainStrict(t *testing.T) {
	for _, raw := range []string{
		`{"roles":{"weak":{"engine":"ollama","model":"bad\nmodel"}},"Roles":{}}`,
		`{"profiles":[],"Profiles":[]}`,
		`{"tiers":{},"Tiers":{}}`,
		`{"routing":{},"Routing":{}}`,
		`{"routing":{"deployments":[],"Deployments":[]}}`,
		`{"roles":{"weak":{"engine":"ollama","model":"bad\nmodel","Model":"good"}}}`,
	} {
		if _, err := Parse([]byte(raw)); err == nil {
			t.Fatal("strict write accepted ambiguous model container or field")
		}
		if _, _, err := ParseStored([]byte(raw)); err == nil || err.Error() != "stored policy is invalid" {
			t.Fatal("stored sanitation exposed shadowed model data")
		}
	}
}
