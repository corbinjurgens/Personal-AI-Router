// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// TestConfiguredEngineUsesItsProtocolRoutes checks a configured engine is
// driven through exactly the routes of the protocol profile it was given: its
// inventory and its inference both, so a fork serving only the OpenAI-compatible
// surface is never sent Ollama's /api requests.
func TestConfiguredEngineUsesItsProtocolRoutes(t *testing.T) {
	for _, tc := range []struct {
		protocol string
		models   string
		answer   string
		want     []string
	}{
		{
			protocol: protocolOpenAIV1,
			models:   `{"data":[{"id":"fork-model"}]}`,
			answer:   `{"choices":[{"message":{"content":"done"}}]}`,
			want:     []string{"GET /v1/models", "POST /v1/chat/completions"},
		},
		{
			protocol: protocolOllamaAPI,
			models:   `{"models":[{"name":"fork-model"}]}`,
			answer:   `{"response":"done"}`,
			want:     []string{"GET /api/tags", "POST /api/generate"},
		},
	} {
		t.Run(tc.protocol, func(t *testing.T) {
			var mu sync.Mutex
			var seen []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				seen = append(seen, r.Method+" "+r.URL.Path)
				mu.Unlock()
				switch r.URL.Path {
				case "/v1/models", "/api/tags":
					_, _ = w.Write([]byte(tc.models))
				case "/v1/chat/completions", "/api/generate":
					_, _ = w.Write([]byte(tc.answer))
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			var stdout, stderr bytes.Buffer
			exit := runAgainstServer(
				t,
				context.Background(),
				[]string{"--backend", "llamacpp-custom", "--protocol", tc.protocol, "--port", "8090", "--prompt", "test"},
				server.URL,
				&stdout,
				&stderr,
			)
			if exit != 0 {
				t.Fatalf("exit=%d stderr=%s", exit, stderr.String())
			}
			mu.Lock()
			defer mu.Unlock()
			if len(seen) != len(tc.want) {
				t.Fatalf("requests = %v, want %v", seen, tc.want)
			}
			for i := range tc.want {
				if seen[i] != tc.want[i] {
					t.Errorf("request %d = %q, want %q", i, seen[i], tc.want[i])
				}
			}
		})
	}
}
