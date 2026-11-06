// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

func TestModelPolicyGeneratedDisplayNamePreservesLongIdentifiers(t *testing.T) {
	test := func(name, model, wantDisplayName string) {
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			stored := nodepolicy.Default()
			stored.Profiles = []nodepolicy.Profile{{Key: "other", DisplayName: wantDisplayName, Engine: "llamacpp", Model: "other"}}
			value := json.Number("0.25")
			change := modelconfig.ModelChange{ModelID: model, Policy: &modelconfig.PolicyPatch{Generation: map[string]*json.Number{"temperature": &value}}}
			merged, err := h.b.mergeModelPolicy(stored, "llamacpp", change, "", settings.Preview{}, false)
			if err != nil {
				t.Fatalf("merge generation: %v", err)
			}
			profile, ok := merged.ProfileByKey(model)
			if !ok || profile.Model != model {
				t.Fatalf("model identity changed: %+v", merged.Profiles)
			}
			if err := nodepolicy.ValidateProfileDisplayName(profile.DisplayName); err != nil {
				t.Fatalf("invalid generated display name: %v", err)
			}
			if profile.DisplayName != wantDisplayName {
				t.Fatalf("display name = %q, want %q", profile.DisplayName, wantDisplayName)
			}
			if len(merged.Profiles) != 2 || merged.Profiles[0].DisplayName != wantDisplayName {
				t.Fatalf("repeated display label was not retained: %+v", merged.Profiles)
			}
			if len(stored.Profiles) != 1 || stored.Profiles[0].Key != "other" || stored.Profiles[0].DisplayName != wantDisplayName {
				t.Fatal("merge mutated stored profiles")
			}
		})
	}
	test("display bound", strings.Repeat("m", 128), strings.Repeat("m", 128))
	test("over display bound", strings.Repeat("m", 129), "model-ac4023a0494b27c9")
	test("model bound", strings.Repeat("m", 512), "model-861fd51c6c8148ff")
	test("multibyte display bound", strings.Repeat("é", 64), strings.Repeat("é", 64))
	test("over multibyte display bound", strings.Repeat("é", 65), "model-fb07b1b3783ab29c")
}

func TestModelPolicyRoleMembershipLists(t *testing.T) {
	test := func(name string, members []string, selected bool, want []string) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			h := newModelSettingsHarness(t)
			stored := nodepolicy.Default()
			stored.Profiles = []nodepolicy.Profile{
				{Key: "selected-profile", DisplayName: "Selected", Engine: "llamacpp", Model: "fixture/model", ContextTokens: 4096},
				{Key: "other-profile", DisplayName: "Other", Engine: "llamacpp", Model: "other-model"},
			}
			stored.Roles = []nodepolicy.Role{
				{Name: nodepolicy.TierStrong, Profiles: []string{"other-profile"}},
				{Name: nodepolicy.TierWeak, Profiles: members},
				{Name: nodepolicy.TierMedium, Profiles: []string{}},
			}
			before := settingsJSON(stored)
			change := modelconfig.ModelChange{ProfileKey: "selected-profile", ModelID: "fixture/model", Policy: &modelconfig.PolicyPatch{Roles: map[string]bool{nodepolicy.TierWeak: selected}}}
			merged, err := h.b.mergeModelPolicy(stored, "llamacpp", change, "", settings.Preview{}, false)
			if err != nil {
				t.Fatalf("merge role membership: %v", err)
			}
			wantRoles := []nodepolicy.Role{
				{Name: nodepolicy.TierStrong, Profiles: []string{"other-profile"}},
				{Name: nodepolicy.TierWeak, Profiles: want},
				{Name: nodepolicy.TierMedium, Profiles: []string{}},
			}
			if !reflect.DeepEqual(merged.Roles, wantRoles) {
				t.Fatalf("roles = %+v, want %+v", merged.Roles, wantRoles)
			}
			if !reflect.DeepEqual(merged.Profiles, stored.Profiles) {
				t.Fatalf("membership changed profile facts: %+v", merged.Profiles)
			}
			if !bytes.Equal(settingsJSON(stored), before) {
				t.Fatal("merge mutated the stored policy")
			}
		})
	}
	test("append preserves existing members", []string{"other-profile"}, true, []string{"other-profile", "selected-profile"})
	test("append to empty declaration", []string{}, true, []string{"selected-profile"})
	test("append is idempotent and keeps position", []string{"selected-profile", "other-profile"}, true, []string{"selected-profile", "other-profile"})
	test("remove preserves other members", []string{"other-profile", "selected-profile"}, false, []string{"other-profile"})
	test("remove final member retains declaration", []string{"selected-profile"}, false, []string{})
	test("remove absent member retains empty declaration", []string{}, false, []string{})
}

func TestModelPolicyRoleRemovalDoesNotDeclareRole(t *testing.T) {
	h := newModelSettingsHarness(t)
	stored := nodepolicy.Default()
	stored.Roles = []nodepolicy.Role{{Name: nodepolicy.TierMedium, Profiles: []string{}}}
	change := modelconfig.ModelChange{ModelID: "fixture/model", Policy: &modelconfig.PolicyPatch{Roles: map[string]bool{nodepolicy.TierWeak: false}}}
	merged, err := h.b.mergeModelPolicy(stored, "llamacpp", change, "", settings.Preview{}, false)
	if err != nil {
		t.Fatal(err)
	}
	want := []nodepolicy.Role{{Name: nodepolicy.TierMedium, Profiles: []string{}}}
	if !reflect.DeepEqual(merged.Roles, want) || len(merged.Profiles) != 1 || merged.Profiles[0].Key != "fixture/model" {
		t.Fatalf("membership edit changed declarations or omitted the profile: %+v", merged)
	}
}

func TestModelPolicyRoleAppendCreatesSelectedProfile(t *testing.T) {
	h := newModelSettingsHarness(t)
	stored := nodepolicy.Default()
	stored.Roles = []nodepolicy.Role{{Name: nodepolicy.TierWeak, Profiles: []string{}}}
	change := modelconfig.ModelChange{ModelID: "fixture/model", Policy: &modelconfig.PolicyPatch{Roles: map[string]bool{nodepolicy.TierWeak: true}}}
	merged, err := h.b.mergeModelPolicy(stored, "llamacpp", change, "", settings.Preview{}, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(merged.Profiles) != 1 || merged.Profiles[0].Key != "fixture/model" || !reflect.DeepEqual(merged.Roles, []nodepolicy.Role{{Name: nodepolicy.TierWeak, Profiles: []string{"fixture/model"}}}) {
		t.Fatalf("profile creation or membership = %+v", merged)
	}
	if len(stored.Profiles) != 0 || !reflect.DeepEqual(stored.Roles, []nodepolicy.Role{{Name: nodepolicy.TierWeak, Profiles: []string{}}}) {
		t.Fatal("membership patch created a profile or changed the stored declaration")
	}
}

func TestModelSettingsRoleMembershipPersistsWithNativeLoad(t *testing.T) {
	h := newModelSettingsHarness(t)
	policy := h.b.localPolicy().Policy
	policy.Profiles = []nodepolicy.Profile{
		{Key: "selected-profile", DisplayName: "Selected", Engine: "llamacpp", Model: "selected-profile", ContextTokens: 4096},
		{Key: "other-profile", DisplayName: "Other", Engine: "llamacpp", Model: "other-model"},
	}
	policy.Roles = []nodepolicy.Role{{Name: nodepolicy.TierStrong, Profiles: []string{"selected-profile"}}, {Name: nodepolicy.TierWeak, Profiles: []string{"other-profile"}}, {Name: nodepolicy.TierMedium, Profiles: []string{}}}
	if _, err := setCurrent(h.b, settingsJSON(policy)); err != nil {
		t.Fatalf("seed membership policy: %v", err)
	}
	p := h.request(t, true)
	p.ModelChange.ProfileKey = "selected-profile"
	p.ModelChange.Policy.Roles = map[string]bool{nodepolicy.TierWeak: true, nodepolicy.TierStrong: false}
	p = h.review(t, p, "")
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("apply membership and native load: %+v, %v", receipt, err)
	}
	wantRoles := []nodepolicy.Role{{Name: nodepolicy.TierStrong, Profiles: []string{}}, {Name: nodepolicy.TierWeak, Profiles: []string{"other-profile", "selected-profile"}}, {Name: nodepolicy.TierMedium, Profiles: []string{}}}
	data, err := os.ReadFile(h.path)
	if err != nil {
		t.Fatal(err)
	}
	saved, err := nodepolicy.Parse(data)
	if err != nil {
		t.Fatalf("parse saved membership policy: %v", err)
	}
	if !reflect.DeepEqual(saved.Roles, wantRoles) || !reflect.DeepEqual(h.b.localPolicy().Policy.Roles, wantRoles) {
		t.Fatalf("saved roles = %+v, want %+v", saved.Roles, wantRoles)
	}
	profile, ok := saved.ProfileByKey("selected-profile")
	if !ok || profile.Model != "selected-profile" || profile.ContextTokens != 8192 || profile.DisplayName != "Selected" {
		t.Fatalf("coupled edit changed target/display name or omitted derived context: %+v", profile)
	}
	record := h.record(t)
	if h.calls() != 1 || record.Model.Stage != "completed" || len(record.Snapshot.Settings.ModelLoads) != 1 || record.Snapshot.Settings.ModelLoads[0].ModelID != "fixture/model" || record.Snapshot.Settings.ModelLoads[0].Resolved == nil || record.Snapshot.Settings.ModelLoads[0].Resolved.PerSlotContext != 8192 {
		t.Fatalf("native load/journal = %+v, configure calls=%d", record, h.calls())
	}
	again, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || again != receipt || h.calls() != 1 || !reflect.DeepEqual(h.b.localPolicy().Policy.Roles, wantRoles) {
		t.Fatalf("retry changed membership or native effects: %+v, %v", again, err)
	}
}

func TestModelPolicyRoleMergeDetachesUnselectedMembership(t *testing.T) {
	h := newModelSettingsHarness(t)
	stored := nodepolicy.Default()
	stored.Profiles = []nodepolicy.Profile{{Key: "selected-profile", DisplayName: "Selected", Engine: "llamacpp", Model: "fixture/model"}}
	stored.Roles = []nodepolicy.Role{{Name: nodepolicy.TierStrong, Profiles: []string{"selected-profile"}}, {Name: nodepolicy.TierWeak, Profiles: []string{}}}
	change := modelconfig.ModelChange{ProfileKey: "selected-profile", ModelID: "fixture/model", Policy: &modelconfig.PolicyPatch{Roles: map[string]bool{nodepolicy.TierWeak: true}}}
	merged, err := h.b.mergeModelPolicy(stored, "llamacpp", change, "", settings.Preview{}, false)
	if err != nil {
		t.Fatal(err)
	}
	merged.Roles[0].Profiles[0] = "changed-key"
	merged.Roles[1].Profiles[0] = "changed-key"
	want := []nodepolicy.Role{{Name: nodepolicy.TierStrong, Profiles: []string{"selected-profile"}}, {Name: nodepolicy.TierWeak, Profiles: []string{}}}
	if !reflect.DeepEqual(stored.Roles, want) {
		t.Fatalf("merged roles alias stored membership: %+v", stored.Roles)
	}
}

func TestModelSettingsUndeclaredRoleRefusesBeforeWrites(t *testing.T) {
	h := newModelSettingsHarness(t)
	policy := h.b.localPolicy().Policy
	policy.Profiles = []nodepolicy.Profile{{Key: "selected-profile", DisplayName: "Selected", Engine: "llamacpp", Model: "fixture/model"}}
	policy.Roles = []nodepolicy.Role{{Name: nodepolicy.TierMedium, Profiles: []string{}}}
	if _, err := setCurrent(h.b, settingsJSON(policy)); err != nil {
		t.Fatalf("seed declared role: %v", err)
	}
	p := h.request(t, true)
	p.ModelChange.Policy.Roles = map[string]bool{nodepolicy.TierWeak: true}
	beforePolicy, beforeSettings := h.b.localPolicy().Policy, h.record(t)
	settingsPath, err := h.b.engineSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	policyBytes, err := os.ReadFile(h.path)
	if err != nil {
		t.Fatal(err)
	}
	settingsBytes, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	writes := 0
	replacer := durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
		writes++
		return nil, os.ErrPermission
	}}
	h.b.engineSettingsReplacer = replacer
	setReplacer(h.b, replacer)
	proxyCalls := len(h.proxy.snapshot())
	_, err = h.b.previewEngineSettings(context.Background(), p, "")
	const want = "Role weak is not declared on this node; create it in node settings."
	if err == nil || err.Error() != want || engineSettingsErrorCode(err) != -32602 {
		t.Fatalf("undeclared role refusal = %v, want invalid params %q", err, want)
	}
	if writes != 0 || h.calls() != 0 || !reflect.DeepEqual(h.record(t), beforeSettings) || !reflect.DeepEqual(h.b.localPolicy().Policy, beforePolicy) {
		t.Fatal("undeclared role reached journal, native or policy writes")
	}
	for _, call := range h.proxy.snapshot()[proxyCalls:] {
		if call.method == nodepolicy.MethodSetEngineDrain {
			t.Fatal("undeclared role drained an engine")
		}
	}
	for path, want := range map[string][]byte{h.path: policyBytes, settingsPath: settingsBytes} {
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("undeclared role changed %s", path)
		}
	}
}
