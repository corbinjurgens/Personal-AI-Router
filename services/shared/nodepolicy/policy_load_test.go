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

func TestModelControlStoredPolicyPreservesValidSiblings(t *testing.T) {
	for _, version := range []string{"1", "2"} {
		t.Run("version "+version, func(t *testing.T) {
			kind := `"kind":"local",`
			if version == "1" {
				kind = ""
			}
			raw := []byte(`{"version":` + version + `,"revision":"42","availability":"available","admission":{"memoryBudgetBytes":12345},"profiles":[{"name":"good","engine":"ollama","model":"good"},{"name":"bad","engine":"ollama","model":"RAW"}],"roles":{"weak":{"engine":"ollama","model":"good"},"strong":{"engine":"ollama","model":"bad\nname"}},"tiers":{"weak":[{` + kind + `"engine":"ollama","model":"good"},{` + kind + `"engine":"ollama","model":"bad\u0085name"}]},"servedModels":[{"engine":"ollama","model":"bad\nname"}],"residentModels":[{"engine":"ollama","model":"bad\nname"}]}`)
			raw = bytes.ReplaceAll(raw, []byte("RAW"), []byte{0xff})
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
			if len(p.Profiles) != 1 || p.Profiles[0].Model != "good" || len(p.Roles) != 1 || p.Roles[TierWeak].Model != "good" || len(p.Tiers[TierWeak]) != 1 || p.Tiers[TierWeak][0].Model != "good" {
				t.Fatal("valid model siblings lost")
			}
			if p.ServedModels == nil || len(p.ServedModels) != 0 || len(p.ResidentModels) != 0 {
				t.Fatal("empty permission or residency changed")
			}
			for _, category := range []string{"profiles", "roles", "tiers", "servedModels", "residentModels"} {
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
				p.Tiers[TierWeak] = append(p.Tiers[TierWeak], TierEntry{Kind: EntryAPI, Deployment: "paid-big"})
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
			if len(got.Tiers[TierWeak]) != 1 || got.Tiers[TierWeak][0].Kind != EntryLocal || len(got.Tiers[TierStrong]) != 0 {
				t.Fatal("dependent or dormant API candidates retained")
			}
			settings := got.Routing.Tiers[TierWeak]
			if settings.RemoteAllowed || settings.PaidAllowed || len(settings.AllowedFamilies) != 0 {
				t.Fatal("exhausted allowlist broadened")
			}
			if counts["deployments"] != 1 || counts["familyReferences"] != 1 || counts["deploymentReferences"] < 2 {
				t.Fatal("missing fixed removal counts")
			}
		})
	}
}

func TestModelControlStoredPolicyConservativeRecovery(t *testing.T) {
	for _, raw := range []string{
		`{"availability":"bad-secret","roles":{"weak":{"engine":"ollama","model":"bad\nname"}}}`,
		`{"roles":{"weak":{"engine":"ollama","model":"good","Model":"bad\nname"}}}`,
		`{"roles":{"weak":{"engine":"ollama","model":"bad\nname"}},"Roles":{}}`,
		`{"servedModels":[{"engine":"ollama","model":null}]}`,
		`{"servedModels":[{"engine":"ollama","model":"bad\nname","extra":true}]}`,
		`{"profiles":[{"name":"same","engine":"ollama","model":"good"},{"name":"same","engine":"ollama","model":"bad\nname"}]}`,
		`{"profiles":[{"name":"test","engine":"ollama","model":"bad\nname","memoryBytes":-1}]}`,
		`{"roles":{"unknown":{"engine":"ollama","model":"bad\nname"}}}`,
		`{"profiles":[{"name":"test","engine":"ollama","model":"bad\nname"}],"residentModels":[{"engine":"ollama","model":"unrelated"}]}`,
	} {
		if _, _, err := ParseStored([]byte(raw)); err == nil {
			t.Fatal("unrelated malformed policy repaired")
		} else if err.Error() != "stored policy is invalid" {
			t.Fatalf("unsafe recovery diagnostic: %q", err.Error())
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
		raw := append([]byte(`{"roles":{"weak":{"engine":"ollama","model":`), token...)
		raw = append(raw, []byte(`}}}`)...)
		p, counts, err := ParseStored(raw)
		if err != nil {
			t.Fatal(err)
		}
		if len(p.Roles) != 0 || counts["roles"] != 1 || p.Availability != Available {
			t.Fatal("invalid model did not receive entry-wise repair")
		}
	}
}

func TestModelControlStoredPolicyPreservesPausedState(t *testing.T) {
	p, counts, err := ParseStored([]byte(`{"version":2,"revision":"88","availability":"paused","pausedBy":"peer","pause":{"onActive":"cancel","drainTimeoutSeconds":19},"roles":{"weak":{"engine":"ollama","model":"bad\nname"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if p.Revision != "88" || p.Availability != Paused || p.PausedBy != PausedByPeer || p.Pause.OnActive != CancelActive || p.Pause.DrainTimeoutSeconds != 19 || counts["roles"] != 1 {
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
			tr := p.Routing.Tiers[TierWeak]
			tr.RemoteAllowed, tr.PaidAllowed = tc.remote, tc.paid
			p.Routing.Tiers[TierWeak] = tr
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
			raw := []byte(`{` + tc.revisions + `,"availability":"paused","Availability":"available","idle":{"unloadAfterMinutes":7},"IDLE":{"wakeTimeoutSeconds":23},"routing":{"limits":{"queueSeconds":5},"Limits":{"maxAttempts":3}},"roles":{"weak":{"engine":"ollama","model":"good"},"strong":{"engine":"ollama","model":"bad\nmodel"}}}`)
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
			if len(p.Roles) != 1 || p.Roles[TierWeak].Model != "good" || counts["roles"] != 1 {
				t.Fatal("sanitation lost valid model sibling")
			}
			if !bytes.Equal(raw, before) {
				t.Fatal("load changed original document")
			}
		})
	}
}
