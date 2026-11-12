// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package main

import (
	"errors"
	"os"
)

// platformNameCandidates returns the Linux sources for the generated name:
// the systemd pretty hostname, then the static hostname, then the kernel
// hostname.
func platformNameCandidates() []nameCandidate {
	return []nameCandidate{
		func() (string, error) {
			data, err := os.ReadFile("/etc/machine-info")
			if err != nil {
				return "", err
			}
			return prettyHostnameFromMachineInfo(data), nil
		},
		func() (string, error) {
			data, err := os.ReadFile("/etc/hostname")
			if err != nil {
				return "", err
			}
			if line := firstHostnameLine(data); line != "" {
				return line, nil
			}
			return "", errors.New("/etc/hostname has no name")
		},
		os.Hostname,
	}
}
