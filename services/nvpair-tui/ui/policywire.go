// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"encoding/json"

	"nvpair-shared/nodepolicy"
)

// decodePolicyAdvertisement is the minimal inventory-policy decoder for A2's
// future TUI diagnostics. The current TUI decodes availability only; A0 does not
// subscribe to an inventory transport that has not landed yet. Shared decoding
// preserves unsupported versions without interpreting them as readiness.
func decodePolicyAdvertisement(data []byte) (nodepolicy.Advertisement, error) {
	var policy nodepolicy.Advertisement
	err := json.Unmarshal(data, &policy)
	return policy, err
}
