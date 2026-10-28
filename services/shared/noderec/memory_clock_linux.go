// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import "golang.org/x/sys/unix"

// MemoryBootClock returns suspend-aware nanoseconds since boot.
func MemoryBootClock() (uint64, error) {
	var ts unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_BOOTTIME, &ts); err != nil {
		return 0, err
	}
	return memoryClockTimespec(ts.Sec, ts.Nsec)
}
