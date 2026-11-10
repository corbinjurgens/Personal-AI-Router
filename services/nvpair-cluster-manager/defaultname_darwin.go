// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package main

import (
	"bytes"
	"context"
	"errors"
	"os/exec"
	"strings"
	"time"
)

const (
	scutilPath      = "/usr/sbin/scutil"
	scutilTimeout   = 2 * time.Second
	scutilMaxOutput = 4096
)

// platformNameCandidates returns the macOS sources for the generated name:
// the user-facing ComputerName, then the Bonjour LocalHostName.
func platformNameCandidates() []nameCandidate {
	return []nameCandidate{
		func() (string, error) { return scutilGet("ComputerName") },
		func() (string, error) { return scutilGet("LocalHostName") },
	}
}

// scutilGet reads one System Configuration preference with scutil --get.
// A non-zero exit means the key is not set.
func scutilGet(key string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), scutilTimeout)
	defer cancel()
	var stdout bytes.Buffer
	cmd := exec.CommandContext(ctx, scutilPath, "--get", key)
	cmd.Stdout = &stdout
	if err := cmd.Run(); err != nil {
		return "", err
	}
	if stdout.Len() > scutilMaxOutput {
		return "", errors.New("scutil output too long")
	}
	return strings.TrimSuffix(stdout.String(), "\n"), nil
}
