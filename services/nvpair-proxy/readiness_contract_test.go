// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestStrictDestinationContractRejectsLegacyWaitAsStrict(t *testing.T) {
	for _, tc := range []struct{ name, input string }{
		{"legacy wait header", `0`},
		{"incomplete envelope", `{"queueBudgetMs":0}`},
		{"unsupported version", `{"version":2,"queueBudgetMs":0,"coldStartBudgetMs":0,"preOutputBudgetMs":1,"request":{},"identity":{}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var request nodepolicy.StrictAdmissionRequest
			if err := json.Unmarshal([]byte(tc.input), &request); err == nil {
				t.Fatal("legacy/incomplete/unsupported request accepted as strict")
			}
		})
	}
}
