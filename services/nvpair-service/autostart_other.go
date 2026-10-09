// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !linux && !darwin && !windows

package main

import (
	"errors"
	"runtime"
)

var errAutostartUnsupported = errors.New("autostart is not supported on " + runtime.GOOS)

func autostartLocation() (string, error) { return "", errAutostartUnsupported }

func writeAutostart(autostartEntry) error { return errAutostartUnsupported }

func removeAutostart() (bool, error) { return false, errAutostartUnsupported }

func readAutostart() (string, bool, error) { return "", false, errAutostartUnsupported }
