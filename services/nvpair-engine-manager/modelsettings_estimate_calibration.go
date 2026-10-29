// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

type modelEstimateCalibrationKey struct {
	Component, Class, Backend string
}

type modelEstimateCalibration struct {
	LowNum, LowDen, HighNum, HighDen uint64
	Provenance                       []string // Reviewed fit fixture names, never runtime samples.
}

// Populate only from reviewed calibration evidence. Synthetic tests are not evidence.
var modelEstimateCalibrations = map[modelEstimateCalibrationKey]modelEstimateCalibration{}

// Editable uncertainty policy, not a measured accuracy claim.
const modelEstimateDefaultLowNum, modelEstimateDefaultLowDen = 3, 4
const modelEstimateDefaultHighNum, modelEstimateDefaultHighDen = 3, 2

func modelEstimateBand(component, class, backend string, expected uint64) (low, high uint64, basis string) {
	band, ok := modelEstimateCalibrations[modelEstimateCalibrationKey{component, class, backend}]
	basis = "calibrated"
	if !ok {
		band = modelEstimateCalibration{LowNum: modelEstimateDefaultLowNum, LowDen: modelEstimateDefaultLowDen, HighNum: modelEstimateDefaultHighNum, HighDen: modelEstimateDefaultHighDen}
		basis = "heuristic"
	}
	scale := func(num, den uint64, ceil bool) uint64 {
		if den == 0 {
			panic(errModelEstimateOverflow)
		}
		whole := modelEstimateMul(expected/den, num)
		remainder := modelEstimateMul(expected%den, num)
		value := modelEstimateAdd(whole, remainder/den)
		if ceil && remainder%den != 0 {
			value = modelEstimateAdd(value, 1)
		}
		return value
	}
	return scale(band.LowNum, band.LowDen, false), scale(band.HighNum, band.HighDen, true), basis
}
