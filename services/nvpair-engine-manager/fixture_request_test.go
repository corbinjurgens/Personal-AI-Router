// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"testing"
	"time"
)

// fixtureRequestHandler rejects unrelated loopback traffic before fixture effects.
func fixtureRequestHandler(method, path string, handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method || r.URL.Path != path {
			http.NotFound(w, r)
			return
		}
		handler(w, r)
	}
}

func assertFixtureRejectsProbe(t *testing.T, serverURL string) {
	t.Helper()
	assertFixtureRejectsRequest(t, serverURL, http.MethodGet, "/")
}

func assertFixtureRejectsRequest(t *testing.T, serverURL, method, path string) {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), method, serverURL+path, nil)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Timeout: time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("fixture probe: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("fixture probe status = %d, want 404", resp.StatusCode)
	}
	t.Logf("injected %s %s%s: status %d", method, serverURL, path, resp.StatusCode)
}
