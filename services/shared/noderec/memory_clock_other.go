// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !linux && !darwin && !windows

package noderec

import "errors"

// MemoryBootClock reports unsupported clocks as unavailable.
func MemoryBootClock() (uint64, error) {
	return 0, errors.New("boot clock unsupported")
}
