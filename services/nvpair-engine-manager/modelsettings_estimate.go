// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"

	"nvpair-shared/modelconfig"
)

// Task 10 replaces this advisory body. Inputs carry preferences only, never
// caller telemetry, native build claims, device mappings or allocation proof.
func estimateModelSettings(ctx context.Context, p modelconfig.InspectRequest, requested modelconfig.RequestedLoad) (modelconfig.Estimate, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.Estimate{}, modelSettingsParamsError(err)
	}
	if err := requested.Validate(); err != nil {
		return modelconfig.Estimate{}, modelSettingsParamsError(err)
	}
	if err := ctx.Err(); err != nil {
		return modelconfig.Estimate{}, err
	}
	return modelconfig.Estimate{Status: "unavailable", Unknowns: []string{"bounded artifact facts and target memory mapping unavailable; estimator not ready"}}, nil
}
