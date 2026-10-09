// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestDesktopEntryQuotesExec(t *testing.T) {
	got, err := desktopEntry(autostartEntry{
		Program: "/opt/My Apps/nvpair-service",
		Args:    []string{"--", "--log-level", "debug", `a"b`, "50%", `c\d`},
	})
	if err != nil {
		t.Fatal(err)
	}
	wantExec := `Exec="/opt/My Apps/nvpair-service" -- --log-level debug "a\\"b" 50%% "c\\\\d"`
	if !strings.Contains(got, wantExec+"\n") {
		t.Fatalf("desktop entry:\n%s\nwant line:\n%s", got, wantExec)
	}
	for _, want := range []string{"[Desktop Entry]\n", "Type=Application\n", "Path=/opt/My Apps\n", "X-GNOME-Autostart-enabled=true\n"} {
		if !strings.Contains(got, want) {
			t.Fatalf("desktop entry missing %q:\n%s", want, got)
		}
	}
	if _, err := desktopEntry(autostartEntry{Program: "/x", Args: []string{"a\nb"}}); err == nil {
		t.Fatal("line break in an argument accepted")
	}
}

func TestLaunchAgentPlistRoundTrips(t *testing.T) {
	e := autostartEntry{Program: "/Applications/PAIR & Co/nvpair-service", Args: []string{"--", "--log-level", "<debug>"}}
	plist := launchAgentPlist(e)
	if !strings.Contains(plist, "<string>/Applications/PAIR &amp; Co/nvpair-service</string>") {
		t.Fatalf("program not escaped:\n%s", plist)
	}
	if !strings.Contains(plist, "<key>RunAtLoad</key>\n\t<true/>") || !strings.Contains(plist, "<key>KeepAlive</key>\n\t<false/>") {
		t.Fatalf("plist lifecycle keys wrong:\n%s", plist)
	}
	args, err := launchAgentArgs([]byte(plist))
	if err != nil {
		t.Fatal(err)
	}
	want := append([]string{e.Program}, e.Args...)
	if strings.Join(args, "|") != strings.Join(want, "|") {
		t.Fatalf("args = %q, want %q", args, want)
	}
}

func TestWindowsCommandLine(t *testing.T) {
	got := windowsCommandLine(autostartEntry{
		Program: `C:\Program Files\PAIR\nvpair-service.exe`,
		Args:    []string{"--", "--log-level", "debug", "", "a b", `say "hi"`, `trail\`, `dir\ x\`},
	})
	want := `"C:\Program Files\PAIR\nvpair-service.exe" -- --log-level debug "" "a b" "say \"hi\"" trail\ "dir\ x\\"`
	if got != want {
		t.Fatalf("command line\n got %s\nwant %s", got, want)
	}
}

func TestAutostartRejectsBadCommands(t *testing.T) {
	var out bytes.Buffer
	if err := runAutostart(nil, &out); err == nil {
		t.Fatal("no subcommand accepted")
	}
	if err := runAutostart([]string{"bogus"}, &out); err == nil {
		t.Fatal("unknown subcommand accepted")
	}
	if err := runAutostart([]string{"enable", "--no-such-flag"}, &out); err == nil {
		t.Fatal("invalid service flag accepted for autostart")
	}
}
