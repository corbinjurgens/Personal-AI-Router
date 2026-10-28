//go:build !linux && !darwin && !windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "errors"

func openModelDeviceLibraries(modelDeviceRuntime) (modelDeviceSymbols, modelDeviceRegister, error) {
	return nil, nil, errors.New("native metadata unsupported on this platform")
}
func openModelCUDADriver() (modelDeviceSymbols, modelDeviceRegister, error) {
	return nil, nil, errors.New("CUDA unsupported on this platform")
}
