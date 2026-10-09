// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// autostartLocation is the XDG autostart entry:
// $XDG_CONFIG_HOME/autostart/nvpair-service.desktop, or ~/.config when the
// variable is unset or not absolute (as the XDG Base Directory spec requires).
func autostartLocation() (string, error) {
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" || !filepath.IsAbs(base) {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home directory: %w", err)
		}
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "autostart", desktopFileName), nil
}

func writeAutostart(e autostartEntry) error {
	path, err := autostartLocation()
	if err != nil {
		return err
	}
	content, err := desktopEntry(e)
	if err != nil {
		return err
	}
	return writeFileAtomic(path, []byte(content), 0o644)
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
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", false, nil
		}
		return "", false, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if v, ok := strings.CutPrefix(sc.Text(), "Exec="); ok {
			return strings.ReplaceAll(v, `\\`, `\`), true, nil
		}
	}
	return "", false, sc.Err()
}
