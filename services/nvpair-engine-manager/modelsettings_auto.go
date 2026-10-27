// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"

	"nvpair-shared/modelconfig"
)

// Task 11 replaces this refusal after O2 is answered. No candidate, default,
// fallback, native probe or saved resolution is selected by this contract body.
func resolveModelSettingsAuto(ctx context.Context, p modelconfig.InspectRequest, requested modelconfig.RequestedLoad) (modelconfig.AutoProposal, error) {
	estimate, err := estimateModelSettings(ctx, p, requested)
	if err != nil {
		return modelconfig.AutoProposal{}, err
	}
	if (requested.Context == nil || requested.Context.Mode != "auto") && (requested.GPU == nil || requested.GPU.Mode != "auto") {
		return modelconfig.AutoProposal{}, modelSettingsParamsError(fmt.Errorf("modelChange.load: select explicit Auto preferences"))
	}
	return modelconfig.AutoProposal{Status: "unavailable", Requested: requested.Clone(), Estimate: estimate},
		modelSettingsParamsError(fmt.Errorf("modelChange.load: reviewed Auto resolution is not ready; owner trade choices and target inputs are required"))
}
