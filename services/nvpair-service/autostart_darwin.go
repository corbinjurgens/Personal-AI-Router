// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// autostartLocation is the per-user LaunchAgent plist. launchd reads it at the
// next login; nothing is loaded into the current session.
func autostartLocation() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	return filepath.Join(home, "Library", "LaunchAgents", launchAgentLabel+".plist"), nil
}

func writeAutostart(e autostartEntry) error {
	path, err := autostartLocation()
	if err != nil {
		return err
	}
	return writeFileAtomic(path, []byte(launchAgentPlist(e)), 0o644)
}

func removeAutostart() (bool, error) {
	path, err := autostartLocation()
	if err != nil {
		return false, err
	}
	if err := os.Remove(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func readAutostart() (string, bool, error) {
	path, err := autostartLocation()
	if err != nil {
		return "", false, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", false, nil
		}
		return "", false, err
	}
	args, err := launchAgentArgs(data)
	if err != nil {
		return "", false, fmt.Errorf("read %s: %w", path, err)
	}
	return strings.Join(args, " "), true, nil
}
