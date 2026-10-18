// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestConfiguredEngineIDsAreAccepted checks a backend need not be one of the
// built-in three: any engine-manager engine ID is driven through the protocol
// profile and port the caller takes from its descriptor and facade.
func TestConfiguredEngineIDsAreAccepted(t *testing.T) {
	for _, tc := range []struct {
		name     string
		args     []string
		protocol string
		port     int
	}{
		{
			name:     "fork on openai-v1",
			args:     []string{"--backend", "llamacpp-custom", "--protocol", "openai-v1", "--port", "8090"},
			protocol: protocolOpenAIV1,
			port:     8090,
		},
		{
			name:     "variant on ollama-api with thinking off",
			args:     []string{"--backend", "Ollama-Nightly", "--protocol", "ollama-api", "--port", "11500", "--ollama-think", "false"},
			protocol: protocolOllamaAPI,
			port:     11500,
		},
		{
			name:     "built-in keeps its own protocol and default port",
			args:     []string{"--backend", "llamacpp"},
			protocol: protocolOpenAIV1,
			port:     defaultLlamaCPPPort,
		},
		{
			name:     "built-in may name its own protocol",
			args:     []string{"--backend", "ollama", "--protocol", "ollama-api"},
			protocol: protocolOllamaAPI,
			port:     defaultOllamaPort,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stderr bytes.Buffer
			cfg, err := parseConfig(tc.args, &stderr)
			if err != nil {
				t.Fatalf("parse config: %v", err)
			}
			if got := effectiveProtocol(cfg); got != tc.protocol {
				t.Errorf("protocol = %q, want %q", got, tc.protocol)
			}
			if got := effectivePort(cfg); got != tc.port {
				t.Errorf("port = %d, want %d", got, tc.port)
			}
		})
	}
}

// TestConfiguredEngineNeedsItsProtocolAndPort checks nothing is guessed for an
// engine the dispatcher does not know: without a protocol it would fall back to
// Ollama's API, and without a port to Ollama's, which is how an unknown name
// used to be treated by every path that was not the validation.
func TestConfiguredEngineNeedsItsProtocolAndPort(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"no protocol", []string{"--backend", "strata", "--port", "9000"}, `--protocol is required for engine "strata"`},
		{"no port", []string{"--backend", "strata", "--protocol", "openai-v1"}, `--port is required for engine "strata"`},
		{"unknown protocol", []string{"--backend", "strata", "--protocol", "anthropic-messages", "--port", "9000"}, "--protocol must be ollama-api or openai-v1"},
		{"invalid id", []string{"--backend", "../strata", "--protocol", "openai-v1", "--port", "9000"}, "--backend must be"},
		{"built-in with another protocol", []string{"--backend", "lmstudio", "--protocol", "ollama-api"}, "--protocol for lmstudio must be openai-v1"},
		{"thinking on openai-v1", []string{"--backend", "strata", "--protocol", "openai-v1", "--port", "9000", "--ollama-think", "false"}, "--ollama-think is only valid with the ollama-api protocol"},
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
