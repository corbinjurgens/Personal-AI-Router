// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package tests

import (
	"errors"
	"time"
)

func modelSettingsReplaceProcess(expected serviceProcessIdentity, kill func() error, timeout time.Duration) (exited bool, err error) {
	proof, err := retainFixtureProcessLifetime(expected)
	if err != nil {
		return false, err
	}
	defer func() {
		if closeErr := proof.close(); closeErr != nil {
			exited = false
			err = errors.Join(err, closeErr)
		}
	}()
	if err := kill(); err != nil {
		return false, err
	}
	return proof.wait(timeout)
}
