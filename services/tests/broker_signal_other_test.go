// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package tests

import (
	"os"
	"os/exec"
)

func configureBrokerSignalTarget(*exec.Cmd) {}

func sendBrokerInterrupt(process *os.Process) error {
	return process.Signal(os.Interrupt)
}
