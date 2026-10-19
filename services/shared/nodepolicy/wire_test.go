// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestStrictReadinessFences(t *testing.T) {
	for _, tc := range []struct {
		name      string
		value     StrictAdmissionReadiness
		wantError bool
	}{
		{"withdrawal", StrictAdmissionReadiness{PolicyRevision: "0", ProxyRunID: "p-1"}, false},
		{"missing revision", StrictAdmissionReadiness{ProxyRunID: "p-1"}, true},
		{"invalid revision", StrictAdmissionReadiness{PolicyRevision: "01", ProxyRunID: "p-1"}, true},
		{"missing run", StrictAdmissionReadiness{PolicyRevision: "1"}, true},
		{"invalid run", StrictAdmissionReadiness{PolicyRevision: "1", ProxyRunID: "p/1"}, true},
		{"invalid capability", StrictAdmissionReadiness{PolicyRevision: "1", ProxyRunID: "p-1", Capability: &StrictAdmissionCapability{Version: 1}}, true},
		{"unsupported capability", StrictAdmissionReadiness{PolicyRevision: "1", ProxyRunID: "p-1", Capability: &StrictAdmissionCapability{Version: 2}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.value.Validate()
			if (err != nil) != tc.wantError {
				t.Fatalf("Validate() = %v, want error %v", err, tc.wantError)
			}
		})
	}
}

func TestStrictReadinessWithdrawalOmitsCapability(t *testing.T) {
	r := StrictAdmissionReadiness{PolicyRevision: "12", ProxyRunID: "p-1"}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"policyRevision":"12","proxyRunId":"p-1"}` {
		t.Fatalf("encoded = %s", data)
	}
	var again StrictAdmissionReadiness
	if err := json.Unmarshal(data, &again); err != nil {
		t.Fatal(err)
	}
	if again != r {
		t.Fatalf("round trip = %+v, want %+v", again, r)
	}
}

func TestSetModelIdentityParamsValidate(t *testing.T) {
	digest := strings.Repeat("a", 64)
	valid := func() EngineIdentity {
		return EngineIdentity{EngineRunID: "run-1", SettingsFingerprint: digest, LaunchDigest: digest, ModelDigests: map[string]string{"m:latest": digest}}
	}
	for _, tc := range []struct {
		name    string
		edit    func(*EngineIdentity)
		params  *SetModelIdentityParams
		wantErr string
	}{
		{name: "valid", edit: func(*EngineIdentity) {}},
		{name: "no models", edit: func(i *EngineIdentity) { i.ModelDigests = map[string]string{} }},
		{name: "no engines", params: &SetModelIdentityParams{Engines: map[string]EngineIdentity{}}},
		{name: "missing engines", params: &SetModelIdentityParams{}, wantErr: "engines is required"},
		{name: "missing run", edit: func(i *EngineIdentity) { i.EngineRunID = "" }, wantErr: "engineRunId"},
		{name: "invalid run", edit: func(i *EngineIdentity) { i.EngineRunID = "run/1" }, wantErr: "engineRunId"},
		{name: "uppercase fingerprint", edit: func(i *EngineIdentity) { i.SettingsFingerprint = strings.Repeat("A", 64) }, wantErr: "settingsFingerprint"},
		{name: "short launch digest", edit: func(i *EngineIdentity) { i.LaunchDigest = "abc" }, wantErr: "launchDigest"},
		{name: "missing model map", edit: func(i *EngineIdentity) { i.ModelDigests = nil }, wantErr: "modelDigests is required"},
		{name: "empty model", edit: func(i *EngineIdentity) { i.ModelDigests = map[string]string{" ": digest} }, wantErr: "empty model"},
		{name: "prefixed model digest", edit: func(i *EngineIdentity) { i.ModelDigests = map[string]string{"m": "sha256:" + digest} }, wantErr: "modelDigests"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			params := tc.params
			if params == nil {
				id := valid()
				tc.edit(&id)
				params = &SetModelIdentityParams{Engines: map[string]EngineIdentity{"ollama": id}}
			}
			err := params.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Validate() = %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}
}
