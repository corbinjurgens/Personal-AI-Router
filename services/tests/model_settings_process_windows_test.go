//go:build windows

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"errors"
	"fmt"
	"time"

	"golang.org/x/sys/windows"
)

func modelSettingsReplaceProcess(expected serviceProcessIdentity, kill func() error, timeout time.Duration) (exited bool, err error) {
	milliseconds := timeout.Milliseconds()
	if milliseconds <= 0 || milliseconds >= int64(windows.INFINITE) {
		return false, errors.New("manager replacement requires a finite positive wait")
	}
	handle, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(expected.PID))
	if err != nil {
		return false, fmt.Errorf("retain manager lifetime: %w", err)
	}
	defer func() {
		if closeErr := windows.CloseHandle(handle); closeErr != nil {
			exited = false
			err = errors.Join(err, fmt.Errorf("close manager lifetime: %w", closeErr))
		}
	}()
	// Read identity from the same retained handle that will prove exit. Neither
	// PID reuse nor an inaccessible exiting process can substitute that proof.
	var created, end, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(handle, &created, &end, &kernel, &user); err != nil {
		return false, err
	}
	image := make([]uint16, 32768)
	size := uint32(len(image))
	if err := windows.QueryFullProcessImageName(handle, 0, &image[0], &size); err != nil {
		return false, err
	}
	path, err := serviceExecutablePath(windows.UTF16ToString(image[:size]))
	if err != nil {
		return false, err
	}
	if path != expected.Path || created.Nanoseconds()/int64(time.Millisecond) != expected.Created {
		return false, errors.New("manager identity changed before replacement")
	}
	if err := kill(); err != nil {
		return false, err
	}
	result, err := windows.WaitForSingleObject(handle, uint32(milliseconds))
	if err != nil {
		return false, err
	}
	switch result {
	case windows.WAIT_OBJECT_0:
		return true, nil
	case uint32(windows.WAIT_TIMEOUT):
		return false, nil
	default:
		return false, fmt.Errorf("unexpected manager lifetime wait result: %d", result)
	}
}
