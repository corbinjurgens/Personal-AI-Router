// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy_test

import (
	"encoding/json"
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestAdmissionSnapshotParams(t *testing.T) {
	if err := json.Unmarshal([]byte(`{}`), &nodepolicy.AdmissionSnapshotParams{}); err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string]string{"null": `null`, "array": `[]`, "unknown field": `{"engine":"llamacpp"}`, "trailing object": `{}{}`} {
		t.Run(name, func(t *testing.T) {
			if err := json.Unmarshal([]byte(input), &nodepolicy.AdmissionSnapshotParams{}); err == nil {
				t.Fatal("accepted invalid snapshot params")
			}
		})
	}
}
