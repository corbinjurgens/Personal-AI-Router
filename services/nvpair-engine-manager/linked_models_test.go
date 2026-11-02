// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"testing"
)

func TestLinkedModelPeerBoundary(t *testing.T) {
	t.Run("native file request without a listing is refused", func(t *testing.T) {
		root := t.TempDir()
		ex := copyTestExecutor(t, "ollama", root, true)
		fixture := writeOllamaModel(t, filepath.Join(root, "models"), "tiny", 128, 3)
		query := url.Values{"engine": {"ollama"}, "model": {"tiny"}, "path": {fixture.manifestRel}}
		response := httptest.NewRecorder()
		(&controlServer{exec: ex}).handleModelFile(response, httptest.NewRequest(http.MethodGet, controlModelFilePath+"?"+query.Encode(), nil))
		if response.Code == http.StatusOK {
			t.Fatal("served native model file without an issued listing")
		}
	})
}
