// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestConfiguredEngineIDsAreAccepted checks a backend need not be llamacpp:
// any engine-manager engine ID is driven on the port the caller takes from its
// facade.
func TestConfiguredEngineIDsAreAccepted(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		port int
	}{
		{
			name: "fork on its own port",
			args: []string{"--backend", "llamacpp-custom", "--port", "8090"},
			port: 8090,
		},
		{
			name: "mixed-case descriptor engine",
			args: []string{"--backend", "Strata", "--port", "9000"},
			port: 9000,
		},
		{
			name: "default backend and port",
			args: nil,
			port: defaultLlamaCPPPort,
		},
		{
			name: "llamacpp may override its port",
			args: []string{"--backend", "llamacpp", "--port", "8181"},
			port: 8181,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stderr bytes.Buffer
			cfg, err := parseConfig(tc.args, &stderr)
			if err != nil {
				t.Fatalf("parse config: %v", err)
			}
			if got := effectivePort(cfg); got != tc.port {
				t.Errorf("port = %d, want %d", got, tc.port)
			}
		})
	}
}

// TestConfiguredEngineNeedsItsPort checks nothing is guessed for an engine the
// dispatcher does not know: without a port it would fall back to llama.cpp's,
// which belongs to a different engine.
func TestConfiguredEngineNeedsItsPort(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"no port", []string{"--backend", "strata"}, `--port is required for engine "strata"`},
		{"invalid id", []string{"--backend", "../strata", "--port", "9000"}, "--backend must be"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stderr bytes.Buffer
			_, err := parseConfig(tc.args, &stderr)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want %q", err, tc.want)
			}
		})
	}
}
