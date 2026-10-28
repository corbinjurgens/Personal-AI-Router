// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"errors"

	"golang.org/x/sys/windows"
)

// MemoryBootClock returns suspend-aware nanoseconds since boot.
func MemoryBootClock() (uint64, error) {
	d := windows.DurationSinceBoot()
	if d < 0 {
		return 0, errors.New("invalid boot clock")
	}
	return uint64(d), nil
}
