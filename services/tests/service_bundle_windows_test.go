// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"os"
	"testing"

	"golang.org/x/sys/windows"
)

func serviceImageReleasePending(err error) bool {
	return errors.Is(err, windows.ERROR_SHARING_VIOLATION) || errors.Is(err, windows.ERROR_ACCESS_DENIED)
}

func TestServiceBundleWindowsImageReleaseErrors(t *testing.T) {
	for _, code := range []error{windows.ERROR_SHARING_VIOLATION, windows.ERROR_ACCESS_DENIED} {
		if err := (&os.PathError{Op: "open", Path: "scanner.exe", Err: code}); !serviceImageReleasePending(err) {
			t.Fatalf("stopped-process image error is not retryable: %v", err)
		}
	}
	if err := (&os.PathError{Op: "open", Path: "scanner.exe", Err: windows.ERROR_DISK_FULL}); serviceImageReleasePending(err) {
		t.Fatalf("unrelated filesystem failure is retryable: %v", err)
	}
}
