// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "nvpair-shared/modelconfig"

// Task 5 replaces this body with the bounded stopped-engine resolver. Do not
// call export/hash, live inventory, download or native initialization to fill it.
func inspectModelArtifacts(p modelconfig.InspectRequest) modelconfig.InspectResult {
	return modelconfig.InspectResult{
		Engine: p.Engine, ModelID: p.ModelID, Status: "unavailable",
		Reason: "bounded artifact inspection is not ready",
		Facts:  &modelconfig.Facts{Status: "unavailable", Unknowns: []string{"exact target artifacts and bounded metadata unavailable"}},
	}
}
