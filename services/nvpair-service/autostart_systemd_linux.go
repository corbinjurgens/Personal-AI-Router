// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// systemdUnitName is the systemd user unit that `autostart enable --systemd`
// writes, for machines that have no desktop session to run an XDG entry.
const systemdUnitName = "nvpair-service.service"

// systemctlRun runs `systemctl --user args...`. Tests replace it so they never
// touch the real user manager.
var systemctlRun = func(args ...string) ([]byte, error) {
	return exec.Command("systemctl", append([]string{"--user"}, args...)...).CombinedOutput()
}

// systemdUnitPath is $XDG_CONFIG_HOME/systemd/user/nvpair-service.service,
// with the same fallback to ~/.config as the XDG autostart entry.
func systemdUnitPath() (string, error) {
	xdg, err := autostartLocation()
	if err != nil {
		return "", err
	}
	// autostartLocation is <config>/autostart/<file>.
	return filepath.Join(filepath.Dir(filepath.Dir(xdg)), "systemd", "user", systemdUnitName), nil
}

// systemdQuote quotes one ExecStart word per systemd.service(5): a word with
// whitespace, quotes, or backslashes is double-quoted with C-style escapes,
// and % and $ are doubled so they are not read as specifiers or variables.
func systemdQuote(arg string) string {
	arg = strings.ReplaceAll(arg, "%", "%%")
	arg = strings.ReplaceAll(arg, "$", "$$")
	if arg != "" && !strings.ContainsAny(arg, " \t\"'\\;") {
		return arg
	}
	arg = strings.ReplaceAll(arg, `\`, `\\`)
	arg = strings.ReplaceAll(arg, `"`, `\"`)
	return `"` + arg + `"`
}

// systemdUnit renders the user unit for e.
func systemdUnit(e autostartEntry) (string, error) {
	if !filepath.IsAbs(e.Program) {
		return "", fmt.Errorf("service path %q is not absolute", e.Program)
	}
	parts := make([]string, 0, len(e.Args)+1)
	for _, a := range append([]string{e.Program}, e.Args...) {
		if strings.ContainsAny(a, "\n\r") {
			return "", fmt.Errorf("argument %q contains a line break", a)
		}
		parts = append(parts, systemdQuote(a))
	}
	var b strings.Builder
	b.WriteString("[Unit]\n")
	b.WriteString("Description=" + autostartDisplayName + "\n\n")
	b.WriteString("[Service]\n")
	b.WriteString("ExecStart=" + strings.Join(parts, " ") + "\n")
	b.WriteString("Restart=on-failure\n\n")
	b.WriteString("[Install]\n")
	b.WriteString("WantedBy=default.target\n")
	return b.String(), nil
}

// enableSystemd writes the unit and enables it. A missing or failing systemctl
// is not fatal: the unit stays and the commands to run by hand are printed.
func enableSystemd(e autostartEntry, out io.Writer) error {
	path, err := systemdUnitPath()
	if err != nil {
		return err
	}
	content, err := systemdUnit(e)
	if err != nil {
		return err
	}
	if err := writeFileAtomic(path, []byte(content), 0o644); err != nil {
		return err
	}
	fmt.Fprintf(out, "wrote systemd user unit: %s\n", path)
	steps := [][]string{{"daemon-reload"}, {"enable", systemdUnitName}}
	for _, step := range steps {
		if output, err := systemctlRun(step...); err != nil {
			fmt.Fprintf(out, "systemctl --user %s failed: %v\n", strings.Join(step, " "), err)
			if msg := strings.TrimSpace(string(output)); msg != "" {
				fmt.Fprintln(out, msg)
			}
			fmt.Fprintln(out, "The unit file is in place. Run these by hand:")
			for _, s := range steps {
				fmt.Fprintf(out, "  systemctl --user %s\n", strings.Join(s, " "))
			}
			printLingerHint(out)
			return nil
		}
	}
	fmt.Fprintf(out, "systemd user unit enabled: %s\n", systemdUnitName)
	fmt.Fprintf(out, "Start it now with: systemctl --user start %s\n", systemdUnitName)
	printLingerHint(out)
	return nil
}

func printLingerHint(out io.Writer) {
	fmt.Fprintln(out, "To start the service at boot without a login, run once: loginctl enable-linger $USER")
}

// disableSystemd disables and removes the unit if it exists. systemctl is run
// best-effort; the file is removed either way.
func disableSystemd(out io.Writer) (bool, error) {
	path, err := systemdUnitPath()
	if err != nil {
		return false, err
	}
	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	if output, err := systemctlRun("disable", systemdUnitName); err != nil {
		fmt.Fprintf(out, "systemctl --user disable %s failed (continuing): %v %s\n", systemdUnitName, err, strings.TrimSpace(string(output)))
	}
	if err := os.Remove(path); err != nil {
		return false, err
	}
	_, _ = systemctlRun("daemon-reload")
	fmt.Fprintf(out, "autostart disabled: removed %s\n", path)
	return true, nil
}

// statusSystemd reports the unit if present.
func statusSystemd(out io.Writer) (bool, error) {
	path, err := systemdUnitPath()
	if err != nil {
		return false, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	command := ""
	for _, line := range strings.Split(string(data), "\n") {
		if v, ok := strings.CutPrefix(line, "ExecStart="); ok {
			command = v
			break
		}
	}
	state := "unknown"
	if output, err := systemctlRun("is-enabled", systemdUnitName); err == nil || len(output) > 0 {
		state = strings.TrimSpace(string(output))
	}
	fmt.Fprintf(out, "autostart: enabled (systemd user unit %s, %s)\n  %s\n", path, state, command)
	return true, nil
}
