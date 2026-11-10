// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"strings"

	"nvpair-shared/nodename"
)

// nameCandidate returns one possible generated node name, or an error when the
// platform source is unavailable.
type nameCandidate func() (string, error)

// defaultNodeName returns this node's generated display name: the first
// platform candidate that nodename.Normalize accepts, else the UUID fallback.
// It is computed once per service start; an OS rename while PAIR runs is not
// followed.
func defaultNodeName(nodeUUID string) string {
	return firstValidName(platformNameCandidates(), nodeUUID)
}

// firstValidName is the candidate loop behind defaultNodeName, separated so
// tests can inject candidates without touching the OS.
func firstValidName(candidates []nameCandidate, nodeUUID string) string {
	for _, candidate := range candidates {
		raw, err := candidate()
		if err != nil {
			continue
		}
		if name, err := nodename.Normalize(raw); err == nil {
			return name
		}
	}
	return nodename.Fallback(nodeUUID)
}

// prettyHostnameFromMachineInfo returns PRETTY_HOSTNAME from the contents of
// /etc/machine-info (systemd's KEY=VALUE format with shell-style quoting), or
// "" when the key is absent.
func prettyHostnameFromMachineInfo(data []byte) string {
	value := ""
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, raw, ok := strings.Cut(line, "=")
		if !ok || strings.TrimSpace(key) != "PRETTY_HOSTNAME" {
			continue
		}
		value = unquoteMachineInfoValue(strings.TrimSpace(raw))
	}
	return value
}

// unquoteMachineInfoValue strips one pair of matching surrounding quotes and
// resolves the backslash escapes systemd writes: \\, \", \', \$ and \`.
func unquoteMachineInfoValue(raw string) string {
	if len(raw) >= 2 && (raw[0] == '"' || raw[0] == '\'') && raw[len(raw)-1] == raw[0] {
		raw = raw[1 : len(raw)-1]
	}
	var b strings.Builder
	for i := 0; i < len(raw); i++ {
		if raw[i] == '\\' && i+1 < len(raw) {
			switch raw[i+1] {
			case '\\', '"', '\'', '$', '`':
				b.WriteByte(raw[i+1])
				i++
				continue
			}
		}
		b.WriteByte(raw[i])
	}
	return b.String()
}

// firstHostnameLine returns the first line of /etc/hostname that is not blank
// and not a comment, trimmed; "" when there is none.
func firstHostnameLine(data []byte) string {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		return line
	}
	return ""
}
