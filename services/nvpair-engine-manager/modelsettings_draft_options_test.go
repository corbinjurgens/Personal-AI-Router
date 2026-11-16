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

// Options accepted only as unpriced extras (owner decision D9 admits
// ctx-checkpoints). Every other unsupported option stays refused.
type unpricedExtraTestOption struct{ key, value, help string }

var unpricedExtraTestOptions = []unpricedExtraTestOption{
	{"spec-draft-n-max", "8", "--spec-draft-n-max N  maximum draft tokens\n"},
	{"ctx-checkpoints", "6", "-ctxcp, --ctx-checkpoints, --swa-checkpoints N  max context checkpoints per slot\n"},
}

func TestDraftExtraReportedBuildRoundTrip(t *testing.T) {
	for _, tc := range unpricedExtraTestOptions {
		t.Run(tc.key, func(t *testing.T) {
			var marker string
			f := newAppliedKeyFixture(t, func(f inspectionFixture) {
				marker, _ = nativeOptionsFixture(t, f.ex, f.st)
				writeTestFile(t, os.Getenv("PAIR_NATIVE_HELP_FILE"), []byte(nativeOptionsFixtureHelp+tc.help))
			})
			extra := map[string]string{tc.key: tc.value}
			preview, err := f.ex.PreviewLaunch(nativeOptionsPreviewRequest(t, f, extra))
			if err != nil || len(preview.Errors) != 0 {
				t.Fatalf("reported extra preview = %+v, err=%v", preview, err)
			}
			i := slices.IndexFunc(preview.Settings.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ModelID == f.request.ModelID })
			if i < 0 || !reflect.DeepEqual(preview.Settings.ModelLoads[i].Requested.ExtraOptions, extra) {
				t.Fatalf("preview lost extra: %+v", preview.Settings.ModelLoads)
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
			parsed, err := parseLlamaServingPresetAllowing(data, modelLoadAllowedPresetKeys([]modelconfig.ModelLoad{{ProfileKey: "model", ModelID: "model", Requested: request}}))
			if err != nil || parsed["model"][tc.key] != tc.value {
				t.Fatalf("generated option = %v, err=%v", parsed, err)
			}
			converted, err := convertTypedLlamaOptions("model", extra)
			if err != nil || !reflect.DeepEqual(converted.ExtraOptions, extra) {
				t.Fatalf("converted extra = %+v, err=%v", converted, err)
			}
			if key, values := llamaLaunchArgument("--" + tc.key); key != tc.key || values != 1 {
				t.Fatalf("launch boundary = %q, %d", key, values)
			}
		})
	}
}

func TestConvertTypedLlamaCarriesContextCheckpoints(t *testing.T) {
	f := newModelLoadFixture(t)
	writeTestFile(t, f.source, []byte("[*]\nfit = off\nload-on-startup = false\n[weak-model]\nmodel = weak.gguf\nctx-size = 4096\nparallel = 1\ngpu-layers = 0\nctx-checkpoints = 6\n[medium-model]\nmodel = medium.gguf\nctx-size = 16384\nparallel = 2\ngpu-layers = 0\n"))
	rt := convertModelLoadFixture(t, f)
	for _, load := range rt.ModelLoads {
		var want map[string]string
		if load.ModelID == "weak-model" {
			want = map[string]string{"ctx-checkpoints": "6"}
		}
		if !reflect.DeepEqual(load.Requested.ExtraOptions, want) {
			t.Fatalf("%s extra options = %v, want %v", load.ModelID, load.Requested.ExtraOptions, want)
		}
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
	for _, tc := range unpricedExtraTestOptions {
		t.Run(tc.key, func(t *testing.T) {
			in.Allocation.Options.ExtraOptions = map[string]string{tc.key: tc.value}
			estimate := state.estimate(context.Background(), in)
			if estimate.Status != "unknown" || estimate.CapacityFit != "unknown" || estimate.FreeFit != "unknown" || !slices.Contains(estimate.Unknowns, "Options not priced: "+tc.key+".") {
				t.Fatalf("unpriced option cost was priced: %+v", estimate)
			}
		})
	}
}

func TestSpecDraftPreservesStrictOptionBoundary(t *testing.T) {
	t.Run("operator preset", func(t *testing.T) {
		for _, tc := range unpricedExtraTestOptions {
			_, err := parseLlamaServingPreset([]byte("[model]\n" + tc.key + " = " + tc.value + "\n"))
			var field *ServingFieldError
			if !errors.As(err, &field) || field.Field != "serving.models."+tc.key || field.Reason != "Unsupported envelope-affecting preset option." {
				t.Fatalf("operator %s refusal = %v", tc.key, err)
			}
		}
	})
	t.Run("other unsupported extras", func(t *testing.T) {
		for _, option := range llamaServingOptions {
			if slices.ContainsFunc(unpricedExtraTestOptions, func(tc unpricedExtraTestOption) bool { return tc.key == option.key }) {
				if !option.unsupported || !option.unpricedExtra {
					t.Fatalf("unpriced boundary metadata = %+v", option)
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
		for _, tc := range unpricedExtraTestOptions {
			i := slices.IndexFunc(options, func(option modelSettingsOption) bool { return option.Native.key == tc.key })
			if i < 0 {
				t.Fatalf("%s missing from projection", tc.key)
			}
			options[i].Native.unpricedExtra = false
			if err := validateModelSettingsOptionCoverage(llamaServingOptions, options); err == nil {
				t.Fatalf("coverage accepted mismatched unpriced-extra metadata for %s", tc.key)
			}
			options[i].Native.unpricedExtra = true
		}
	})
}
