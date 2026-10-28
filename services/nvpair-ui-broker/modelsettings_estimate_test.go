// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
)

func TestModelSettingsEstimateDoesNotChangeReviewOrApply(t *testing.T) {
	h := newModelSettingsHarness(t)
	var mu sync.Mutex
	var estimate *modelconfig.Estimate
	setEstimate := func(value *modelconfig.Estimate) {
		mu.Lock()
		defer mu.Unlock()
		estimate = value
	}
	h.engine.setAnswer(func(method string, raw json.RawMessage) (any, string) {
		result, failure := h.answer(method, raw)
		if method == "engine:preview-launch" && failure == "" {
			preview := result.(settings.Preview)
			mu.Lock()
			preview.Estimate = estimate
			mu.Unlock()
			return preview, ""
		}
		return result, failure
	})
	p := h.request(t, false)
	setEstimate(&modelconfig.Estimate{Status: "partial", Unknowns: []string{"synthetic"}})
	first, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || first.Estimate == nil || first.PreviewDigest == "" {
		t.Fatalf("first preview = %+v, %v", first, err)
	}
	setEstimate(&modelconfig.Estimate{Status: "unknown", Unknowns: []string{"telemetry unavailable"}})
	second, err := h.b.previewEngineSettings(context.Background(), p, "")
	if err != nil || second.Estimate == nil || second.Estimate.Status != "unknown" || second.PreviewDigest != first.PreviewDigest {
		t.Fatalf("second preview = %+v, %v; first digest = %q", second, err, first.PreviewDigest)
	}
	setEstimate(nil)
	p.ModelChange.PreviewDigest = first.PreviewDigest
	receipt, err := h.b.applyEngineSettings(context.Background(), p, "")
	if err != nil || receipt.Phase != "succeeded" {
		t.Fatalf("apply after estimate disappeared = %+v, %v", receipt, err)
	}
}

func TestModelSettingsEstimateRejectsClientMemoryInput(t *testing.T) {
	_, err := modelconfig.DecodeInspectRequest([]byte(`{"engine":"llamacpp","modelId":"fixture/model","memory":{"source":"broker","capacityBytes":"999999"}}`))
	if err == nil {
		t.Fatal("inspection accepted client-supplied memory")
	}
}
