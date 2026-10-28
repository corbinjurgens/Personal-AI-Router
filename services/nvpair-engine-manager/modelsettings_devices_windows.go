// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"path/filepath"
	"strings"

	"github.com/ebitengine/purego"
	"golang.org/x/sys/windows"
)

func openModelDeviceLibraries(selected modelDeviceRuntime) (modelDeviceSymbols, modelDeviceRegister, error) {
	var handles []windows.Handle
	for _, name := range []string{"ggml-base.dll", "ggml.dll"} {
		found := false
		for _, lib := range selected.Libraries {
			found = found || strings.EqualFold(name, lib.Name)
		}
		if !found {
			return nil, nil, errors.New("missing selected ggml library")
		}
		h, err := windows.LoadLibraryEx(filepath.Join(selected.Root, name), 0, windows.LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|windows.LOAD_LIBRARY_SEARCH_SYSTEM32)
		if err != nil {
			return nil, nil, err
		}
		handles = append(handles, h)
	}
	// Preload selected CUDA dependencies with an isolated dependency search.
	for _, prefix := range []string{"cublaslt", "cublas64", "cudart", "ggml-cuda"} {
		for _, lib := range selected.Libraries {
			if !strings.HasPrefix(strings.ToLower(lib.Name), prefix) {
				continue
			}
			h, err := windows.LoadLibraryEx(filepath.Join(selected.Root, lib.Name), 0, windows.LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|windows.LOAD_LIBRARY_SEARCH_SYSTEM32)
			if err != nil {
				return nil, nil, err
			}
			handles = append(handles, h)
		}
	}
	lookup := func(name string) (uintptr, error) {
		for i := len(handles) - 1; i >= 0; i-- {
			if p, e := windows.GetProcAddress(handles[i], name); e == nil && p != 0 {
				return p, nil
			}
		}
		return 0, errors.New("unsupported ggml export")
	}
	if err := loadModelDeviceCUDA(selected, lookup, purego.RegisterFunc); err != nil {
		return nil, nil, err
	}
	return lookup, purego.RegisterFunc, nil
}
func openModelCUDADriver() (modelDeviceSymbols, modelDeviceRegister, error) {
	h, err := windows.LoadLibraryEx("nvcuda.dll", 0, windows.LOAD_LIBRARY_SEARCH_SYSTEM32)
	if err != nil {
		return nil, nil, err
	}
	return func(name string) (uintptr, error) { return windows.GetProcAddress(h, name) }, purego.RegisterFunc, nil
}
