// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os"
	"reflect"
	"slices"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
)

func TestDraftExtraReportedBuildRoundTrip(t *testing.T) {
	var marker string
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		marker, _ = nativeOptionsFixture(t, f.ex, f.st)
		writeTestFile(t, os.Getenv("PAIR_NATIVE_HELP_FILE"), []byte(nativeOptionsFixtureHelp+"--spec-draft-n-max N  maximum draft tokens\n"))
	})
	extra := map[string]string{"spec-draft-n-max": "8"}
	preview, err := f.ex.PreviewLaunch(nativeOptionsPreviewRequest(t, f, extra))
	if err != nil || len(preview.Errors) != 0 {
		t.Fatalf("reported draft extra preview = %+v, err=%v", preview, err)
	}
	i := slices.IndexFunc(preview.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ModelID == f.request.ModelID })
	if i < 0 || !reflect.DeepEqual(preview.Settings.ModelLoads[i].Requested.ExtraOptions, extra) {
		t.Fatalf("preview lost draft extra: %+v", preview.Settings.ModelLoads)
	}
	requireNativeHelpSpawns(t, marker, 1)

	request := modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "cpu"}, ExtraOptions: extra}
	options, _, err := typedLlamaModelOptions("model", request, nil)
	if err != nil {
		t.Fatal(err)
	}
	data, err := typedLlamaPreset(map[string]map[string]string{"model": options})
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parseLlamaServingPresetAllowing(data, modelLoadAllowedPresetKeys([]modelconfig.ModelLoad{{ModelID: "model", Requested: request}}))
	if err != nil || parsed["model"]["spec-draft-n-max"] != "8" {
		t.Fatalf("generated draft option = %v, err=%v", parsed, err)
	}
	converted, err := convertTypedLlamaOptions("model", extra)
	if err != nil || !reflect.DeepEqual(converted.ExtraOptions, extra) {
		t.Fatalf("converted draft extra = %+v, err=%v", converted, err)
	}
	if key, values := llamaLaunchArgument("--spec-draft-n-max"); key != "spec-draft-n-max" || values != 1 {
		t.Fatalf("draft launch boundary = %q, %d", key, values)
	}
}

func TestDraftExtraMissingFromKnownBuildRefused(t *testing.T) {
	var marker string
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		marker, _ = nativeOptionsFixture(t, f.ex, f.st)
	})
	preview, err := f.ex.PreviewLaunch(nativeOptionsPreviewRequest(t, f, map[string]string{"spec-draft-n-max": "8"}))
	if err != nil {
		t.Fatal(err)
	}
	want := "modelLoads.extraOptions: model \"" + f.request.ModelID + "\": This llama.cpp build does not report --spec-draft-n-max."
	if got := preview.Errors["modelLoads.extraOptions"]; got != want {
		t.Fatalf("missing draft option refusal = %q, want %q", got, want)
	}
	requireNativeHelpSpawns(t, marker, 1)
}

func TestDraftTokenUnknownNeverFits(t *testing.T) {
	now := time.Now()
	in := autoTestInput()
	in.Memory = autoTestMemory(now)
	state := modelEstimateState{now: func() time.Time { return now }, run: autoTestEstimate}
	control := state.estimate(context.Background(), in)
	if control.CapacityFit != "fits" || control.FreeFit != "fits" {
		t.Fatalf("finite control does not fit: %+v", control)
	}
	in.Allocation.Options.ExtraOptions = map[string]string{"spec-draft-n-max": "8"}
	estimate := state.estimate(context.Background(), in)
	if estimate.Status != "unknown" || estimate.CapacityFit != "unknown" || estimate.FreeFit != "unknown" || !slices.Contains(estimate.Unknowns, "Options not priced: spec-draft-n-max.") {
		t.Fatalf("draft token cost was priced: %+v", estimate)
	}
}

func TestSpecDraftPreservesStrictOptionBoundary(t *testing.T) {
	t.Run("operator preset", func(t *testing.T) {
		_, err := parseLlamaServingPreset([]byte("[model]\nspec-draft-n-max = 8\n"))
		var field *ServingFieldError
		if !errors.As(err, &field) || field.Field != "serving.models.spec-draft-n-max" || field.Reason != "Unsupported envelope-affecting preset option." {
			t.Fatalf("operator draft option refusal = %v", err)
		}
	})
	t.Run("other unsupported extras", func(t *testing.T) {
		for _, option := range llamaServingOptions {
			if option.key == "spec-draft-n-max" {
				if !option.unsupported || !option.unpricedExtra {
					t.Fatalf("draft boundary metadata = %+v", option)
				}
				continue
			}
			if option.unpricedExtra {
				t.Fatalf("another option gained unpriced acceptance: %s", option.key)
			}
			if option.unsupported {
				t.Run(option.key, func(t *testing.T) {
					if err := validateModelExtraOptions("model", map[string]string{option.key: "false"}); err == nil {
						t.Fatalf("unsupported extra accepted: %s", option.key)
					}
				})
			}
		}
	})
	t.Run("option projection metadata", func(t *testing.T) {
		options, err := llamaModelSettingsOptions()
		if err != nil {
			t.Fatal(err)
		}
		if err := validateModelSettingsOptionCoverage(llamaServingOptions, options); err != nil {
			t.Fatal(err)
		}
		i := slices.IndexFunc(options, func(option modelSettingsOption) bool { return option.Native.key == "spec-draft-n-max" })
		if i < 0 {
			t.Fatal("draft option missing from projection")
		}
		options[i].Native.unpricedExtra = false
		if err := validateModelSettingsOptionCoverage(llamaServingOptions, options); err == nil {
			t.Fatal("coverage accepted mismatched unpriced-extra metadata")
		}
	})
}
