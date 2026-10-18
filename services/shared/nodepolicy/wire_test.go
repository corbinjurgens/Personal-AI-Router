// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"encoding/json"
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
