// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package tests

import (
	"encoding/json"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestStrataContractModelsAndOperations(t *testing.T) {
	strataContractPlatform(t)
	wantInventory := []string{
		"Org/Strata-Loaded:Q4_K_M",
		"Org/Strata-Unloaded:Q5_K_M",
		"Org/Strata-Loading:Q6_K",
		"Org/Strata-NoStatus:Q8_0",
	}
	slices.Sort(wantInventory)
	assertModels := func(t *testing.T, got strataContractModels, wantLoaded []string) {
		t.Helper()
		inventory, present := got.ModelsByEngine["strata"]
		inventory = slices.Sorted(slices.Values(inventory))
		if !present || !slices.Equal(inventory, wantInventory) {
			t.Fatalf("modelsByEngine.strata=%v (present=%t), want %v", inventory, present, wantInventory)
		}
		loaded, present := got.LoadedByEngine["strata"]
		loaded = slices.Sorted(slices.Values(loaded))
		wantLoaded = slices.Sorted(slices.Values(wantLoaded))
		if !present || !slices.Equal(loaded, wantLoaded) {
			t.Fatalf("loadedByEngine.strata=%v (present=%t), want %v", loaded, present, wantLoaded)
		}
	}
	assertNotification := func(t *testing.T, fixture *strataContractFixture, wantLoaded []string) {
		t.Helper()
		message := fixture.Manager.strataContractNotification(t, "engine:models-changed", func(raw json.RawMessage) bool {
			var value struct {
				Engine string               `json:"engine"`
				Models strataContractModels `json:"models"`
			}
			strataContractDecode(t, raw, &value)
			return value.Engine == "strata" && slices.Equal(
				slices.Sorted(slices.Values(value.Models.LoadedByEngine["strata"])),
				slices.Sorted(slices.Values(wantLoaded)),
			)
		})
		var value struct {
			Engine string               `json:"engine"`
			Models strataContractModels `json:"models"`
		}
		strataContractDecode(t, message.Params, &value)
		assertModels(t, value.Models, wantLoaded)
	}
	assertSnapshot := func(t *testing.T, fixture *strataContractFixture, wantLoaded []string) {
		t.Helper()
		var got strataContractModels
		fixture.Manager.strataContractOK(t, "engine:models", nil, &got)
		assertModels(t, got, wantLoaded)
	}
	start := func(t *testing.T) *strataContractFixture {
		t.Helper()
		fixture := strataContractNew(t, true)
		fixture.strataContractStartManager(t)
		// The first watcher sweep seeds its baseline without a models push.
		// Fence it while stopped so starting the runtime produces an observable
		// initial residency push before either model operation can run.
		fixture.Manager.strataContractNotification(t, "engine:model-identity", func(raw json.RawMessage) bool {
			var value strataContractIdentity
			strataContractDecode(t, raw, &value)
			_, present := value.Engines["strata"]
			return !present
		})
		fixture.strataContractStartRuntime(t)
		assertNotification(t, fixture, []string{"Org/Strata-Loaded:Q4_K_M"})
		assertSnapshot(t, fixture, []string{"Org/Strata-Loaded:Q4_K_M"})
		return fixture
	}
	assertRuntime := func(t *testing.T, fixture *strataContractFixture, wantStatus string, wantRequests []strataContractRequest) {
		t.Helper()
		control := fixture.strataContractControl(t, 0)
		wantModels := map[string]string{
			"Org/Strata-Loaded:Q4_K_M":   "loaded",
			"Org/Strata-Unloaded:Q5_K_M": wantStatus,
			"Org/Strata-Loading:Q6_K":    "loading",
			"Org/Strata-NoStatus:Q8_0":   "",
		}
		if !reflect.DeepEqual(control.Models, wantModels) {
			t.Fatalf("runtime residency=%v, want %v", control.Models, wantModels)
		}
		var mutations []strataContractRequest
		for _, request := range control.Requests {
			// Health and inventory GET counts vary with background sweeps.
			if request.Method != http.MethodGet {
				mutations = append(mutations, request)
			}
		}
		if !reflect.DeepEqual(mutations, wantRequests) {
			t.Fatalf("runtime mutation method/path/JSON model counts=%+v, want %+v", mutations, wantRequests)
		}
		if control.NextLoadStatus != 0 {
			t.Fatalf("next load failure was not consumed: %d", control.NextLoadStatus)
		}
	}

	t.Run("exact ID load and unload update snapshots and notifications", func(t *testing.T) {
		fixture := start(t)
		assertRuntime(t, fixture, "unloaded", nil)
		params := map[string]string{"engine": "strata", "model": "Org/Strata-Unloaded:Q5_K_M"}
		var loaded struct {
			Model  string `json:"model"`
			Status string `json:"status"`
			OK     bool   `json:"ok"`
		}
		fixture.Manager.strataContractOK(t, "engine:load-model", params, &loaded)
		if loaded.Model != "Org/Strata-Unloaded:Q5_K_M" || loaded.Status != "loaded" || !loaded.OK {
			t.Fatalf("load response=%+v, want exact model, status loaded and ok true", loaded)
		}
		assertNotification(t, fixture, []string{"Org/Strata-Loaded:Q4_K_M", "Org/Strata-Unloaded:Q5_K_M"})
		assertSnapshot(t, fixture, []string{"Org/Strata-Loaded:Q4_K_M", "Org/Strata-Unloaded:Q5_K_M"})
		assertRuntime(t, fixture, "loaded", []strataContractRequest{
			{Method: http.MethodPost, Path: "/v1/load", Model: "Org/Strata-Unloaded:Q5_K_M", Count: 1},
		})

		var unloaded struct {
			Model  string `json:"model"`
			Status string `json:"status"`
			OK     bool   `json:"ok"`
		}
		fixture.Manager.strataContractOK(t, "engine:unload-model", params, &unloaded)
		if unloaded.Model != "Org/Strata-Unloaded:Q5_K_M" || unloaded.Status != "unloaded" || !unloaded.OK {
			t.Fatalf("unload response=%+v, want exact model, status unloaded and ok true", unloaded)
		}
		assertNotification(t, fixture, []string{"Org/Strata-Loaded:Q4_K_M"})
		assertSnapshot(t, fixture, []string{"Org/Strata-Loaded:Q4_K_M"})
		assertRuntime(t, fixture, "unloaded", []strataContractRequest{
			{Method: http.MethodPost, Path: "/v1/load", Model: "Org/Strata-Unloaded:Q5_K_M", Count: 1},
			{Method: http.MethodPost, Path: "/v1/unload", Model: "Org/Strata-Unloaded:Q5_K_M", Count: 1},
		})
	})

	t.Run("HTTP conflict load propagates once without changing residency", func(t *testing.T) {
		fixture := start(t)
		assertRuntime(t, fixture, "unloaded", nil)
		fixture.strataContractControl(t, http.StatusConflict)
		response := fixture.Manager.strataContractCall(t, "engine:load-model", map[string]string{
			"engine": "strata", "model": "Org/Strata-Unloaded:Q5_K_M",
		})
		if response.Error == nil || response.Error.Code != -32000 {
			t.Fatalf("load error=%+v, want RPC -32000", response.Error)
		}
		body, ok := strings.CutPrefix(response.Error.Message, `action "load_model": engine returned HTTP 409: `)
		if !ok {
			t.Fatalf("load diagnostic=%q, want load_model action and HTTP 409", response.Error.Message)
		}
		var diagnostic map[string]string
		strataContractDecode(t, json.RawMessage(body), &diagnostic)
		if !reflect.DeepEqual(diagnostic, map[string]string{"error": "fixture load refused"}) {
			t.Fatalf("load error body=%v, want fixture load refused", diagnostic)
		}
		assertSnapshot(t, fixture, []string{"Org/Strata-Loaded:Q4_K_M"})
		assertRuntime(t, fixture, "unloaded", []strataContractRequest{
			{Method: http.MethodPost, Path: "/v1/load", Model: "Org/Strata-Unloaded:Q5_K_M", Count: 1},
		})
	})
}
