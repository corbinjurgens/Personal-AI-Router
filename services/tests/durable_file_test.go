// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"os"
	"runtime"
	"syscall"
	"testing"
	"time"
)

func isWindowsSharingError(err error) bool {
	var errno syscall.Errno
	return runtime.GOOS == "windows" && errors.As(err, &errno) && (errno == 5 || errno == 32)
}

// readLiveDurableFile reads a complete old or new snapshot while a product
// process may atomically replace the file. Windows opens can conflict with that
// replacement; retry only ERROR_ACCESS_DENIED (5) / ERROR_SHARING_VIOLATION (32).
func readLiveDurableFile(t *testing.T, path string) []byte {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		data, err := os.ReadFile(path)
		if err == nil {
			return data
		}
		if !isWindowsSharingError(err) {
			t.Fatalf("read %s: %v", path, err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("read %s after transient Windows sharing conflicts: %v", path, err)
		}
		time.Sleep(25 * time.Millisecond)
	}
}
