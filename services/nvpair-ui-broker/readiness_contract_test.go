// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestStrictReadinessContractPolicyDoesNotAdvertise(t *testing.T) {
	p := nodepolicy.Default()
	p.Admission.MaxConcurrentPerModel = 2
	advertisement := p.Advertisement()
	if advertisement.StrictAdmission.State() != nodepolicy.StrictCapabilityAbsent {
		t.Fatal("finite policy alone advertised strict readiness")
	}
	data, err := json.Marshal(advertisement)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	if _, exists := fields["strictAdmission"]; exists {
		t.Fatal("unset capability was encoded")
	}
}
