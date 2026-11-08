// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// The real authenticated engine:remote-policy-get route carries a destination
// profile's saved request options and the activation attestation. The joint
// target has no proxy, so nothing is activated and the attestation is null: an
// origin must treat that read as unknown sizing.
func TestPeerSizingAuthenticatedReadCarriesOptionsAndNullAttestation(t *testing.T) {
	f := newPolicyJointFixture(t)
	draft := f.want
	draft.Profiles = append([]nodepolicy.Profile(nil), f.want.Profiles...)
	draft.Profiles[0].RequestOptions = json.RawMessage(`{"max_tokens":64,"options":{"num_predict":64}}`)
	draft.Profiles[0].RequestOptionsMode = nodepolicy.RequestOptionsEnforced
	set := callRPC(t, f.target.in, f.target.msgs, "policy:set", map[string]any{"policy": draft, "expectedPersistedRevision": f.want.Revision}, 10*time.Second)
	if set.Error != nil {
		t.Fatalf("local policy:set failed: %+v", set.Error)
	}
	var saved struct {
		Policy nodepolicy.Policy `json:"policy"`
	}
	if err := json.Unmarshal(set.Result, &saved); err != nil {
		t.Fatal("decode saved policy")
	}

	read := f.rpc(t, "engine:remote-policy-get", map[string]any{}).Result
	var envelope struct {
		Policy struct {
			Revision string               `json:"revision"`
			Profiles []nodepolicy.Profile `json:"profiles"`
		} `json:"policy"`
		ActiveRequestPolicy json.RawMessage `json:"activeRequestPolicy"`
	}
	if err := json.Unmarshal(read, &envelope); err != nil {
		t.Fatal("decode authenticated peer read")
	}
	if envelope.Policy.Revision != saved.Policy.Revision || len(envelope.Policy.Profiles) != 1 {
		t.Fatalf("peer read revision %q profiles %d, want saved revision %q", envelope.Policy.Revision, len(envelope.Policy.Profiles), saved.Policy.Revision)
	}
	profile := envelope.Policy.Profiles[0]
	if string(profile.RequestOptions) != `{"max_tokens":64,"options":{"num_predict":64}}` || profile.RequestOptionsMode != nodepolicy.RequestOptionsEnforced {
		t.Fatalf("peer read options = %s mode %q", profile.RequestOptions, profile.RequestOptionsMode)
	}
	if string(envelope.ActiveRequestPolicy) != "null" {
		t.Fatalf("activeRequestPolicy = %s, want explicit null with no live proxy", envelope.ActiveRequestPolicy)
	}
}
