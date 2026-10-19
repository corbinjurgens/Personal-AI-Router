// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package main

import (
	"errors"
	"os"
	"os/exec"
)

func lockBusy(err error) bool { return errors.Is(err, os.ErrExist) }

func runChild(cmd *exec.Cmd) error { return cmd.Run() }
