// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"testing"
)

const testNameUUID = "0123abcd-4567-89ef-0123-456789abcdef"

func TestDefaultNodeNameCandidates(t *testing.T) {
	fixed := func(value string) nameCandidate {
		return func() (string, error) { return value, nil }
	}
	failing := func() (string, error) { return "", errors.New("unavailable") }

	tests := []struct {
		name       string
		candidates []nameCandidate
		want       string
	}{
		{name: "first valid wins", candidates: []nameCandidate{fixed("first"), fixed("second")}, want: "first"},
		{name: "erroring candidate skipped", candidates: []nameCandidate{failing, fixed("second")}, want: "second"},
		{name: "invalid candidate skipped", candidates: []nameCandidate{fixed("  "), fixed("bad\x07name"), fixed("third")}, want: "third"},
		{name: "value is normalized", candidates: []nameCandidate{fixed("  Café PC\n")}, want: "Café PC"},
		{name: "all failing gives fallback", candidates: []nameCandidate{failing, fixed("")}, want: "PAIR 0123abcd"},
		{name: "no candidates gives fallback", candidates: nil, want: "PAIR 0123abcd"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := firstValidName(tt.candidates, testNameUUID); got != tt.want {
				t.Fatalf("firstValidName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestPrettyHostnameFromMachineInfo(t *testing.T) {
	tests := []struct {
		name string
		data string
		want string
	}{
		{name: "double quoted", data: "PRETTY_HOSTNAME=\"Office PC\"\n", want: "Office PC"},
		{name: "single quoted", data: "PRETTY_HOSTNAME='Office PC'\n", want: "Office PC"},
		{name: "unquoted", data: "PRETTY_HOSTNAME=office\n", want: "office"},
		{name: "escaped quote", data: `PRETTY_HOSTNAME="Bob\"s \$PC \\ \` + "`" + `x"` + "\n", want: "Bob\"s $PC \\ `x"},
		{name: "missing key", data: "CHASSIS=desktop\nDEPLOYMENT=home\n", want: ""},
		{name: "comment lines skipped", data: "# PRETTY_HOSTNAME=\"commented\"\n\nCHASSIS=laptop\nPRETTY_HOSTNAME=\"Real\"\n", want: "Real"},
		{name: "mismatched quotes kept", data: "PRETTY_HOSTNAME=\"half'\n", want: "\"half'"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := prettyHostnameFromMachineInfo([]byte(tt.data)); got != tt.want {
				t.Fatalf("prettyHostnameFromMachineInfo() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestFirstHostnameLine(t *testing.T) {
	tests := []struct {
		name string
		data string
		want string
	}{
		{name: "single line", data: "box\n", want: "box"},
		{name: "comments and blanks skipped", data: "# managed\n\n  box2  \nother\n", want: "box2"},
		{name: "empty", data: "\n# only comment\n", want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := firstHostnameLine([]byte(tt.data)); got != tt.want {
				t.Fatalf("firstHostnameLine() = %q, want %q", got, tt.want)
			}
		})
	}
}
