// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Names under which the service registers itself to start at login.
const (
	// launchAgentLabel is the macOS LaunchAgent label and plist file stem.
	launchAgentLabel = "com.nvidia.pair.service"
	// desktopFileName is the XDG autostart entry on Linux.
	desktopFileName = "nvpair-service.desktop"
	// runValueName is the value under HKCU\...\CurrentVersion\Run on Windows.
	runValueName = "NVPAIRService"
	// autostartDisplayName labels the entry where a desktop shows one.
	autostartDisplayName = "NVIDIA Personal AI Router service"
)

// autostartEntry is the command registered to run at login.
type autostartEntry struct {
	Program string
	Args    []string
}

// runAutostart implements `nvpair-service autostart enable|disable|status`.
// Everything after "enable" is recorded verbatim as the service's own command
// line, so `autostart enable -- --log-level debug` starts the service at login
// with those broker arguments.
func runAutostart(args []string, out io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: nvpair-service autostart enable|disable|status [--systemd] [service flags] [-- broker args]")
	}
	args, useSystemd := extractSystemdFlag(args)
	if useSystemd && runtime.GOOS != "linux" {
		return errors.New("--systemd is only supported on Linux")
	}
	if len(args) == 0 {
		return errors.New("usage: nvpair-service autostart enable|disable|status [--systemd] [service flags] [-- broker args]")
	}
	location, err := autostartLocation()
	if err != nil {
		return err
	}
	switch args[0] {
	case "enable":
		if err := validateRunArgs(args[1:]); err != nil {
			return err
		}
		exe, err := os.Executable()
		if err != nil {
			return fmt.Errorf("locate own executable: %w", err)
		}
		if resolved, err := filepath.EvalSymlinks(exe); err == nil {
			exe = resolved
		}
		entry := autostartEntry{Program: exe, Args: args[1:]}
		if useSystemd {
			return enableSystemd(entry, out)
		}
		if err := writeAutostart(entry); err != nil {
			return err
		}
		fmt.Fprintf(out, "autostart enabled: %s\n", location)
		fmt.Fprintln(out, "nvpair-service will start at your next login.")
	case "disable":
		removed, err := removeAutostart()
		if err != nil {
			return err
		}
		if removed {
			fmt.Fprintf(out, "autostart disabled: removed %s\n", location)
		}
		unitRemoved, err := disableSystemd(out)
		if err != nil {
			return err
		}
		if !removed && !unitRemoved {
			fmt.Fprintln(out, "autostart was not enabled")
		}
	case "status":
		command, ok, err := readAutostart()
		if err != nil {
			return err
		}
		if ok {
			fmt.Fprintf(out, "autostart: enabled (%s)\n  %s\n", location, command)
		}
		unitOK, err := statusSystemd(out)
		if err != nil {
			return err
		}
		if !ok && !unitOK {
			fmt.Fprintln(out, "autostart: disabled")
		}
	default:
		return fmt.Errorf("unknown autostart command %q: use enable, disable, or status", args[0])
	}
	return nil
}

// extractSystemdFlag removes a --systemd flag from the service-flag part of
// args (before any "--") and reports whether it was present.
func extractSystemdFlag(args []string) ([]string, bool) {
	found := false
	out := make([]string, 0, len(args))
	for i, a := range args {
		if a == "--" {
			out = append(out, args[i:]...)
			break
		}
		if a == "--systemd" || a == "-systemd" {
			found = true
			continue
		}
		out = append(out, a)
	}
	return out, found
}

// desktopEntry renders an XDG autostart .desktop file for e.
func desktopEntry(e autostartEntry) (string, error) {
	parts := make([]string, 0, len(e.Args)+1)
	for _, a := range append([]string{e.Program}, e.Args...) {
		if strings.ContainsAny(a, "\n\r") {
			return "", fmt.Errorf("argument %q contains a line break", a)
		}
		parts = append(parts, desktopExecQuote(a))
	}
	// The quoting rule runs on the value after string unescaping, so the
	// string-level backslash escape is applied last when writing.
	exec := strings.ReplaceAll(strings.Join(parts, " "), `\`, `\\`)
	var b strings.Builder
	b.WriteString("[Desktop Entry]\n")
	b.WriteString("Type=Application\n")
	b.WriteString("Name=" + autostartDisplayName + "\n")
	b.WriteString("Comment=Keeps PAIR inference running in the background\n")
	b.WriteString("Exec=" + exec + "\n")
	b.WriteString("Path=" + strings.ReplaceAll(filepath.Dir(e.Program), `\`, `\\`) + "\n")
	b.WriteString("Terminal=false\n")
	b.WriteString("NoDisplay=true\n")
	b.WriteString("X-GNOME-Autostart-enabled=true\n")
	return b.String(), nil
}

// desktopExecQuote quotes one Exec argument per the Desktop Entry
// Specification: an argument with a reserved character is double-quoted with
// ", `, $, and \ backslash-escaped inside, and a literal % is doubled so it is
// not read as a field code.
func desktopExecQuote(arg string) string {
	arg = strings.ReplaceAll(arg, "%", "%%")
	if arg != "" && !strings.ContainsAny(arg, " \t\"'\\><~|&;$*?#()`") {
		return arg
	}
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range arg {
		switch r {
		case '"', '`', '$', '\\':
			b.WriteByte('\\')
		}
		b.WriteRune(r)
	}
	b.WriteByte('"')
	return b.String()
}

// launchAgentPlist renders a per-user LaunchAgent that starts e once at login.
// KeepAlive is off: service/stop is an explicit request that must stick until
// the next login.
func launchAgentPlist(e autostartEntry) string {
	esc := func(s string) string {
		var b strings.Builder
		_ = xml.EscapeText(&b, []byte(s))
		return b.String()
	}
	var b strings.Builder
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	b.WriteString(`<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">` + "\n")
	b.WriteString(`<plist version="1.0">` + "\n<dict>\n")
	b.WriteString("\t<key>Label</key>\n\t<string>" + launchAgentLabel + "</string>\n")
	b.WriteString("\t<key>ProgramArguments</key>\n\t<array>\n")
	for _, a := range append([]string{e.Program}, e.Args...) {
		b.WriteString("\t\t<string>" + esc(a) + "</string>\n")
	}
	b.WriteString("\t</array>\n")
	b.WriteString("\t<key>WorkingDirectory</key>\n\t<string>" + esc(filepath.Dir(e.Program)) + "</string>\n")
	b.WriteString("\t<key>RunAtLoad</key>\n\t<true/>\n")
	b.WriteString("\t<key>KeepAlive</key>\n\t<false/>\n")
	b.WriteString("</dict>\n</plist>\n")
	return b.String()
}

// launchAgentArgs reads ProgramArguments back out of a plist written by
// launchAgentPlist.
func launchAgentArgs(data []byte) ([]string, error) {
	dec := xml.NewDecoder(strings.NewReader(string(data)))
	var lastKey string
	inArgs := false
	var args []string
	for {
		tok, err := dec.Token()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		start, ok := tok.(xml.StartElement)
		if !ok {
			if end, isEnd := tok.(xml.EndElement); isEnd && end.Name.Local == "array" && inArgs {
				return args, nil
			}
			continue
		}
		switch start.Name.Local {
		case "key":
			var k string
			if err := dec.DecodeElement(&k, &start); err != nil {
				return nil, err
			}
			lastKey = k
		case "array":
			inArgs = lastKey == "ProgramArguments"
		case "string":
			var v string
			if err := dec.DecodeElement(&v, &start); err != nil {
				return nil, err
			}
			if inArgs {
				args = append(args, v)
			}
		}
	}
	if len(args) == 0 {
		return nil, errors.New("no ProgramArguments in LaunchAgent")
	}
	return args, nil
}

// windowsCommandLine joins e into a command line that CommandLineToArgvW (and
// so Go's os.Args on Windows) splits back into the same arguments.
func windowsCommandLine(e autostartEntry) string {
	parts := []string{`"` + e.Program + `"`}
	for _, a := range e.Args {
		parts = append(parts, windowsQuoteArg(a))
	}
	return strings.Join(parts, " ")
}

func windowsQuoteArg(s string) string {
	if s != "" && !strings.ContainsAny(s, " \t\"") {
		return s
	}
	var b strings.Builder
	b.WriteByte('"')
	slashes := 0
	for i := 0; i < len(s); i++ {
		switch c := s[i]; c {
		case '\\':
			slashes++
		case '"':
			b.WriteString(strings.Repeat(`\`, slashes*2+1))
			b.WriteByte('"')
			slashes = 0
		default:
			b.WriteString(strings.Repeat(`\`, slashes))
			slashes = 0
			b.WriteByte(c)
		}
	}
	b.WriteString(strings.Repeat(`\`, slashes*2))
	b.WriteByte('"')
	return b.String()
}

// writeFileAtomic writes data to path through a temporary file in the same
// directory, so a crash never leaves a half-written entry behind.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Chmod(perm); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}
