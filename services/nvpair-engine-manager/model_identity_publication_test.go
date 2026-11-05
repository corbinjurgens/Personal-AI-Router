// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
)

func identityPublicationExecutor(t *testing.T, count int) (*Executor, []string) {
	t.Helper()
	ex := modelDownloadInventoryExecutor(t)
	identity := engines.ModelIdentity{Strength: "declared"}
	for range 64 {
		identity.Files = append(identity.Files, engines.ModelIdentityFile{Role: "model", Name: strings.Repeat("a", 240) + ".gguf", SizeBytes: 42})
	}
	if err := identity.Validate(); err != nil {
		t.Fatalf("invalid aggregate specimen: %v", err)
	}
	names := make([]string, count)
	models := make([]map[string]string, count)
	identities := make(map[string]engines.ModelIdentity, count)
	for i := range count {
		name := fmt.Sprintf("model-%d", i)
		names[i], models[i], identities[name] = name, map[string]string{"id": name}, identity
	}
	data, err := json.Marshal(map[string]any{"data": models})
	if err != nil {
		t.Fatalf("encode inventory fixture: %v", err)
	}
	ex.client = &http.Client{Transport: memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
		if request.Method != http.MethodGet || (request.URL.Path != "/models" && request.URL.Path != "/v1/models") {
			t.Errorf("unexpected engine effect: %s %s", request.Method, request.URL.Path)
			return nil, fmt.Errorf("fixture refuses engine effects")
		}
		body := data
		if request.URL.Path == "/v1/models" {
			body = []byte(`{"data":[]}`)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(bytes.NewReader(body))}, nil
	})}
	ex.modelIdentities.values = map[string]map[string]engines.ModelIdentity{"llamacpp": identities}
	return ex, names
}

func assertPublishedIdentityInventory(t *testing.T, result ModelsResult, names []string) {
	t.Helper()
	if result.Identities != nil || !reflect.DeepEqual(result.Models, names) || !reflect.DeepEqual(result.ByEngine, map[string][]string{"llamacpp": names}) {
		t.Fatalf("publication identities=%d models=%d engineModels=%d, want models=%d (identities nil=%v)", len(result.Identities), len(result.Models), len(result.ByEngine["llamacpp"]), len(names), result.Identities == nil)
	}
}

func TestModelIdentityHTTPAggregatePreservesInventory(t *testing.T) {
	ex, names := identityPublicationExecutor(t, 512)
	unbounded := ex.modelsSweep(context.Background(), true)
	ex.attachModelIdentities(&unbounded)
	data, err := json.Marshal(unbounded)
	if err != nil || len(data)+1 <= jsonrpc.WorkerFrameBytes {
		t.Fatalf("specimen must exceed peer HTTP limit: bytes=%d error=%v", len(data)+1, err)
	}
	recorder := httptest.NewRecorder()
	modelsHandler(ex).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, modelsPath, nil))
	if recorder.Code != http.StatusOK || recorder.Body.Len() > jsonrpc.WorkerFrameBytes {
		t.Fatalf("HTTP inventory status=%d bytes=%d", recorder.Code, recorder.Body.Len())
	}
	var result ModelsResult
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode actual HTTP inventory: %v", err)
	}
	assertPublishedIdentityInventory(t, result, names)
	if len(ex.modelIdentities.snapshot()["llamacpp"]) != len(names) {
		t.Fatal("HTTP publication mutated stored identity facts")
	}
}

func TestModelIdentityResponseBudgetsActualID(t *testing.T) {
	ex, names := identityPublicationExecutor(t, 450)
	result := ex.ModelsResult(context.Background())
	idBytes, err := json.Marshal(strings.Repeat("i", 128*1024))
	if err != nil {
		t.Fatalf("encode correlation ID: %v", err)
	}
	id := json.RawMessage(idBytes)
	if result.Identities == nil || jsonrpc.ResponseFitsFrame(&id, result, jsonrpc.WorkerFrameBytes) {
		t.Fatal("specimen must fit HTTP and exceed the actual response envelope")
	}
	var wire bytes.Buffer
	m := &Manager{exec: ex, codec: NewCodec(&wire)}
	m.runModels(context.Background(), &Message{JSONRPC: "2.0", ID: &id, Method: "engine:models"})
	msg, err := jsonrpc.NewCodecMaxFrame(&wire, jsonrpc.WorkerFrameBytes).Read()
	if err != nil {
		t.Fatalf("read actual engine:models response: %v", err)
	}
	if msg.ID == nil || !bytes.Equal(*msg.ID, id) {
		t.Fatal("response correlation ID changed")
	}
	result = ModelsResult{}
	if err := json.Unmarshal(msg.Result, &result); err != nil {
		t.Fatalf("decode response inventory: %v", err)
	}
	assertPublishedIdentityInventory(t, result, names)
	if len(ex.modelIdentities.snapshot()["llamacpp"]) != len(names) {
		t.Fatal("response publication mutated stored identity facts")
	}
}

func TestModelIdentityNotificationAggregateWithdrawsWithoutLosingInventory(t *testing.T) {
	ex, names := identityPublicationExecutor(t, 512)
	heavy := ex.modelIdentities.snapshot()
	ex.modelIdentities.values = map[string]map[string]engines.ModelIdentity{"llamacpp": {"model-0": heavy["llamacpp"]["model-0"]}}
	updates := modelDownloadWatch(t, ex)
	ex.pokeLoaded()
	first := nextModelDownloadUpdate(t, updates)
	if len(first.Models.Identities["llamacpp"]) != 1 {
		t.Fatal("fitting initial metadata was not published")
	}
	ex.modelIdentities.mu.Lock()
	ex.modelIdentities.values = heavy
	ex.modelIdentities.mu.Unlock()
	ex.pokeLoaded()
	withdrawn := nextModelDownloadUpdate(t, updates)
	assertPublishedIdentityInventory(t, withdrawn.Models, names)
	var wire bytes.Buffer
	codec := jsonrpc.NewCodecMaxFrame(&wire, jsonrpc.WorkerFrameBytes)
	if err := codec.Notify(modelsChangedMethod, withdrawn); err != nil {
		t.Fatalf("encode watcher publication: %v", err)
	}
	if err := codec.Notify("next-event", nil); err != nil {
		t.Fatalf("encode following event: %v", err)
	}
	reader := jsonrpc.NewCodecMaxFrame(&wire, jsonrpc.WorkerFrameBytes)
	if _, err := reader.Read(); err != nil {
		t.Fatalf("read watcher publication: %v", err)
	}
	if msg, err := reader.Read(); err != nil || msg.Method != "next-event" {
		t.Fatalf("worker stream did not survive: message=%+v error=%v", msg, err)
	}
	if len(ex.modelIdentities.snapshot()["llamacpp"]) != len(names) {
		t.Fatal("notification budget mutated stored identity facts")
	}
}
