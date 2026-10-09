// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !linux

package main

import (
	"errors"
	"io"
)

// systemd user units exist only on Linux; runAutostart rejects --systemd
// before reaching these.
func enableSystemd(autostartEntry, io.Writer) error {
	return errors.New("--systemd is only supported on Linux")
}

func disableSystemd(io.Writer) (bool, error) { return false, nil }

func statusSystemd(io.Writer) (bool, error) { return false, nil }
