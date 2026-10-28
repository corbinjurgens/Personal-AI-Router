//go:build linux || darwin

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ebitengine/purego"
)

func openModelDeviceLibraries(selected modelDeviceRuntime) (modelDeviceSymbols, modelDeviceRegister, error) {
	var handles []uintptr
	// Preload only the selected base and registry libraries. Never load_all:
	// that API searches ambient directories and can initialize other engines.
	for _, prefix := range []string{"libggml-base.", "libggml."} {
		found := false
		for _, lib := range selected.Libraries {
			if !strings.HasPrefix(lib.Name, prefix) {
				continue
			}
			h, err := purego.Dlopen(filepath.Join(selected.Root, lib.Name), purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if err != nil {
				return nil, nil, err
			}
			handles = append(handles, h)
			found = true
			break
		}
		if !found {
			return nil, nil, errors.New("missing selected ggml library")
		}
	}
	lookup := func(name string) (uintptr, error) {
		for i := len(handles) - 1; i >= 0; i-- {
			if p, e := purego.Dlsym(handles[i], name); e == nil && p != 0 {
				return p, nil
			}
		}
		return 0, errors.New("unsupported ggml export")
	}
	if err := loadModelDeviceCUDA(selected, lookup, purego.RegisterFunc); err != nil {
		return nil, nil, err
	}
	// Native resources intentionally live until this private process exits.
	return lookup, purego.RegisterFunc, nil
}
func openModelCUDADriver() (modelDeviceSymbols, modelDeviceRegister, error) {
	if runtime.GOOS != "linux" {
		return nil, nil, errors.New("CUDA unsupported on this platform")
	}
	h, err := purego.Dlopen("libcuda.so.1", purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, nil, err
	}
	return func(name string) (uintptr, error) { return purego.Dlsym(h, name) }, purego.RegisterFunc, nil
}
