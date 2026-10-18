// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestStrictPolicyWireCapabilityStates(t *testing.T) {
	for _, tc := range []struct {
		name, input string
		state       nodepolicy.StrictCapabilityState
	}{
		{"old inventory", `{"revision":"0","roles":{}}`, nodepolicy.StrictCapabilityAbsent},
		{"future inventory", `{"revision":"0","roles":{},"strictAdmission":{"version":2,"future":true}}`, nodepolicy.StrictCapabilityUnsupported},
		{"valid inventory", `{"revision":"0","roles":{},"strictAdmission":{"version":1,"models":[{"engine":"ollama","model":"m:latest","maxConcurrent":1,"identity":{"engineRunId":"e-1","settingsFingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","modelDigest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}]}}`, nodepolicy.StrictCapabilityValid},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := decodePolicyAdvertisement([]byte(tc.input))
			if err != nil {
				t.Fatal(err)
			}
			if got.StrictAdmission.State() != tc.state {
				t.Fatalf("state = %q, want %q", got.StrictAdmission.State(), tc.state)
			}
		})
	}
}

func TestStrictPolicyWireRejectsMalformedCapability(t *testing.T) {
	for _, tc := range []struct{ name, input string }{
		{"null capability", `{"strictAdmission":null}`},
		{"alternate-case null capability", `{"StrictAdmission":null}`},
		{"mixed capability aliases", `{"strictAdmission":{"version":2},"StrictAdmission":{"version":2}}`},
		{"empty model list", `{"strictAdmission":{"version":1,"models":[]}}`},
		{"unknown capability field", `{"strictAdmission":{"version":1,"models":[],"extra":true}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := decodePolicyAdvertisement([]byte(tc.input)); err == nil {
				t.Fatal("accepted malformed capability")
			}
		})
	}
}
