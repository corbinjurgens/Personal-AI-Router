// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import "testing"

func TestMemoryTelemetryClockTimespec(t *testing.T) {
	got, err := memoryClockTimespec(2, 3)
	if err != nil || got != 2000000003 {
		t.Fatalf("clock = %d, %v", got, err)
	}
	for _, tc := range []struct{ sec, nsec int64 }{{-1, 0}, {0, -1}, {0, 1000000000}, {1 << 62, 0}} {
		if _, err := memoryClockTimespec(tc.sec, tc.nsec); err == nil {
			t.Fatalf("accepted invalid timespec %+v", tc)
		}
	}
}

func TestMemoryTelemetryHop(t *testing.T) {
	hop := MemoryHop{ForwarderID: "run", BootNS: "100"}
	got, err := hop.ElapsedAt(1000101)
	if err != nil || got != 2 {
		t.Fatalf("elapsed = %d, %v", got, err)
	}
	if _, err := hop.ElapsedAt(99); err == nil {
		t.Fatal("accepted future stamp")
	}
	hop.BootNS = "0100"
	if _, err := hop.ElapsedAt(200); err == nil {
		t.Fatal("accepted noncanonical stamp")
	}
}
