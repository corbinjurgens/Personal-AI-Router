// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"nvpair-shared/engines"
)

func TestModelDownloadHTTPAndRPCCarryCanonicalSnapshot(t *testing.T) {
	test := func(name string, reserve bool, wantModels []string) {
		t.Run(name, func(t *testing.T) {
			ex := modelDownloadInventoryExecutor(t)
			if reserve {
				reserveModelDownload(t, ex.downloads, "ollama", "demo", engines.ModelNamingImpliedLatestTag)
			}
			request := httptest.NewRequest(http.MethodGet, modelsPath, nil)
			recorder := httptest.NewRecorder()
			modelsHandler(ex).ServeHTTP(recorder, request)
			if recorder.Code != http.StatusOK {
				t.Fatalf("HTTP model status = %d", recorder.Code)
			}
			var served ModelsResult
			if err := json.Unmarshal(recorder.Body.Bytes(), &served); err != nil {
				t.Fatalf("decode HTTP inventory: %v", err)
			}
			if served.DownloadState == nil {
				t.Fatal("own HTTP producer omitted downloadState")
			}
			if err := served.DownloadState.Validate(); err != nil {
				t.Fatalf("invalid HTTP lifecycle: %v", err)
			}
			var output bytes.Buffer
			manager := &Manager{exec: ex, codec: NewCodec(&output)}
			id := json.RawMessage(`1`)
			manager.runModels(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:models"})
			var message Message
			if err := json.Unmarshal(output.Bytes(), &message); err != nil {
				t.Fatalf("decode model RPC envelope: %v", err)
			}
			var rpc ModelsResult
			if err := json.Unmarshal(message.Result, &rpc); err != nil {
				t.Fatalf("decode model RPC inventory: %v", err)
			}
			if rpc.DownloadState == nil || !rpc.DownloadState.Equal(*served.DownloadState) ||
				!reflect.DeepEqual(rpc.Models, wantModels) || !reflect.DeepEqual(served.Models, wantModels) ||
				!reflect.DeepEqual(rpc.ByEngine, served.ByEngine) ||
				!reflect.DeepEqual(rpc.LoadedByEngine, served.LoadedByEngine) {
				t.Fatalf("HTTP/RPC disagree on canonical inventory/lifecycle: HTTP=%+v RPC=%+v", served, rpc)
			}
		})
	}
	test("explicit empty lifecycle", false, []string{"demo:latest", "ready:latest"})
	test("queued model remains loaded but unavailable", true, []string{"ready:latest"})
}
