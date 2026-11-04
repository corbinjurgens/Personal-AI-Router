// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"nvpair-shared/nodepolicy"
)

func TestModelPolicyFingerprintIgnoresPauseMetadata(t *testing.T) {
	policy := nodepolicy.Default()
	before := modelPolicyFingerprint(policy)
	for _, mode := range []nodepolicy.PauseMode{nodepolicy.PauseConfigured, nodepolicy.PauseImmediate} {
		t.Run(string(mode), func(t *testing.T) {
			paused := policy
			paused.Revision = "7"
			paused.Availability = nodepolicy.Paused
			paused.PausedBy = nodepolicy.PausedLocally
			paused.PauseMode = mode
			if got := modelPolicyFingerprint(paused); got != before {
				t.Fatalf("pause changed model settings fingerprint: got %s, want %s", got, before)
			}
			paused.Idle.UnloadAfterMinutes++
			if modelPolicyFingerprint(paused) == before {
				t.Fatal("model settings fingerprint ignored a policy content change")
			}
		})
	}
}
