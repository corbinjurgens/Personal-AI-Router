// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/modelconfig"
)

func savedNativeFitRequested() modelconfig.RequestedLoad {
	on, target, ctx := "on", "512", int32(4096)
	return modelconfig.RequestedLoad{Fit: &on, FitTarget: &target, FitCtx: &ctx}
}

func appliedNativeFitPatchFixture(t *testing.T) inspectionFixture {
	t.Helper()
	f := newAppliedKeyFixture(t, func(f inspectionFixture) {
		nativeOptionsFixture(t, f.ex, f.st)
	})
	rt := f.st.plat.Runtime
	for i := range rt.ModelLoads {
		if rt.ModelLoads[i].ModelID == f.request.ModelID {
			fit := savedNativeFitRequested()
			rt.ModelLoads[i].Requested.Fit = fit.Fit
			rt.ModelLoads[i].Requested.FitTarget = fit.FitTarget
			rt.ModelLoads[i].Requested.FitCtx = fit.FitCtx
		}
	}
	command, err := launchForState(f.st, rt.Port)
	if err != nil {
		t.Fatal(err)
	}
	// Preserve the alias's checked artifact binding while regenerating a
	// resolution that agrees with the saved native-fit intent.
	prepared, err := compileTypedLlamaInputs(context.Background(), f.st, command, rt, false)
	if err != nil {
		t.Fatalf("compile saved native-fit fixture: %v", err)
	}
	rt.ModelLoads = prepared.ModelLoads
	f.st.plat.Runtime = rt
	for _, load := range rt.ModelLoads {
		if load.ModelID == f.request.ModelID {
			if err := load.Validate(); err != nil || load.Resolved == nil {
				t.Fatalf("invalid applied native-fit fixture: %+v, err=%v", load, err)
			}
			return f
		}
	}
	t.Fatal("native-fit fixture target missing")
	return f
}

func requireSavedNativeFitFields(t *testing.T, loads []modelconfig.ModelLoad, model, target string, ctx int32) {
	t.Helper()
	for _, load := range loads {
		if load.ModelID != model {
			continue
		}
		got := load.Requested
		if got.Fit == nil || *got.Fit != "on" || got.FitTarget == nil || *got.FitTarget != target || got.FitCtx == nil || *got.FitCtx != ctx {
			t.Fatalf("saved native-fit fields = %+v, want fit=on fitTarget=%s fitCtx=%d", got, target, ctx)
		}
		return
	}
	t.Fatalf("model %q missing from loads: %+v", model, loads)
}

func TestNativeFitSavedFieldPatchDirect(t *testing.T) {
	target, ctx := "1024", int32(8192)
	test := func(name string, set modelconfig.RequestedLoad, wantTarget string, wantCtx int32) {
		t.Run(name, func(t *testing.T) {
			saved := []modelconfig.ModelLoad{{ModelID: "model", Requested: savedNativeFitRequested()}}
			if err := saved[0].Requested.Validate(); err != nil {
				t.Fatalf("saved native-fit setup: %v", err)
			}
			got, err := patchModelLoad(saved, "model", modelconfig.LoadPatch{Set: set})
			if err != nil {
				t.Fatalf("single-leaf patch of saved native fit: %v", err)
			}
			requireSavedNativeFitFields(t, got, "model", wantTarget, wantCtx)
			requireSavedNativeFitFields(t, saved, "model", "512", 4096)
			if set.Fit != nil {
				t.Fatal("sparse patch acquired a synthetic fit selection")
			}
		})
	}
	test("fitTarget preserves saved fit and context", modelconfig.RequestedLoad{FitTarget: &target}, "1024", 4096)
	test("fitCtx preserves saved fit and target", modelconfig.RequestedLoad{FitCtx: &ctx}, "512", 8192)
}

func TestNativeFitSavedFieldPatchPreviewLaunch(t *testing.T) {
	target, ctx := "1024", int32(8192)
	test := func(name string, set modelconfig.RequestedLoad, wantTarget string, wantCtx int32) {
		t.Run(name, func(t *testing.T) {
			f := appliedNativeFitPatchFixture(t)
			request := nativeOptionsPreviewRequest(t, f, nil)
			if _, err := request.Settings.Canonical(); err != nil {
				t.Fatalf("saved native-fit baseline request: %v", err)
			}
			request.ModelChange.Load = &modelconfig.LoadPatch{Set: set}
			preview, err := f.ex.PreviewLaunch(request)
			if err != nil {
				t.Fatalf("PreviewLaunch: %v", err)
			}
			if len(preview.Errors) != 0 {
				t.Fatalf("single-leaf preview of saved native fit has field errors: %v", preview.Errors)
			}
			requireSavedNativeFitFields(t, preview.Settings.ModelLoads, f.request.ModelID, wantTarget, wantCtx)
			requireSavedNativeFitFields(t, f.st.plat.Runtime.ModelLoads, f.request.ModelID, "512", 4096)
			if !reflect.DeepEqual(request.ModelChange.Load.Set, set) || request.ModelChange.Load.Set.Fit != nil {
				t.Fatalf("preview changed sparse Set: %+v", request.ModelChange.Load.Set)
			}
		})
	}
	test("fitTarget preserves saved fit and context", modelconfig.RequestedLoad{FitTarget: &target}, "1024", 4096)
	test("fitCtx preserves saved fit and target", modelconfig.RequestedLoad{FitCtx: &ctx}, "512", 8192)
}

func nativeFitPatchRefusalCases(t *testing.T, check func(*testing.T, modelconfig.RequestedLoad, modelconfig.LoadPatch, string)) {
	t.Helper()
	on, off, target, ctx := "on", "off", "512", int32(4096)
	test := func(name string, saved modelconfig.RequestedLoad, patch modelconfig.LoadPatch, field string) {
		t.Run(name, func(t *testing.T) {
			if err := saved.Validate(); err != nil {
				t.Fatalf("saved setup: %v", err)
			}
			if err := patch.Validate(); err != nil {
				t.Fatalf("valid patch shape refused before merge: %v", err)
			}
			check(t, saved.Clone(), patch, field)
		})
	}
	test("fit off retains saved target", modelconfig.RequestedLoad{Fit: &on, FitTarget: &target}, modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Fit: &off}}, "fitTarget")
	test("fit clear retains saved target", modelconfig.RequestedLoad{Fit: &on, FitTarget: &target}, modelconfig.LoadPatch{Clear: []string{"fit"}}, "fitTarget")
	test("fit off retains saved context", modelconfig.RequestedLoad{Fit: &on, FitCtx: &ctx}, modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Fit: &off}}, "fitCtx")
	test("fit clear retains saved context", modelconfig.RequestedLoad{Fit: &on, FitCtx: &ctx}, modelconfig.LoadPatch{Clear: []string{"fit"}}, "fitCtx")
	test("fit off clears only target", savedNativeFitRequested(), modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Fit: &off}, Clear: []string{"fitTarget"}}, "fitCtx")
	test("fit clear clears only context", savedNativeFitRequested(), modelconfig.LoadPatch{Clear: []string{"fit", "fitCtx"}}, "fitTarget")
	test("saved fit versus GPU Auto patch", savedNativeFitRequested(), modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "auto"}}}, "fit")
	test("saved GPU Auto versus fit patch", modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "auto"}}, modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Fit: &on}}, "fit")
}

func TestNativeFitSavedFieldPatchDirectRefusesMergedConflicts(t *testing.T) {
	nativeFitPatchRefusalCases(t, func(t *testing.T, requested modelconfig.RequestedLoad, patch modelconfig.LoadPatch, field string) {
		saved := []modelconfig.ModelLoad{{ModelID: "model", Requested: requested}}
		before, patchBefore := modelconfig.CloneLoads(saved), patch.Set.Clone()
		got, err := patchModelLoad(saved, "model", patch)
		var typed *ServingFieldError
		if !errors.As(err, &typed) || typed.Field != "modelLoads."+field || typed.ModelID != "model" {
			t.Fatalf("merged refusal = %v, want named field modelLoads.%s for model", err, field)
		}
		if got != nil || !reflect.DeepEqual(saved, before) || !reflect.DeepEqual(patch.Set, patchBefore) {
			t.Fatalf("refused patch changed saved state or Set: loads=%+v set=%+v", saved, patch.Set)
		}
	})
}

func TestNativeFitSavedFieldPatchPreviewRefusesBeforeHelp(t *testing.T) {
	nativeFitPatchRefusalCases(t, func(t *testing.T, saved modelconfig.RequestedLoad, patch modelconfig.LoadPatch, field string) {
		f := newModelLoadFixture(t)
		rt := convertModelLoadFixture(t, f)
		load := &rt.ModelLoads[0]
		load.Requested, load.Resolved = saved, nil
		f.st.plat.Runtime = rt
		marker, _ := nativeOptionsFixture(t, f.ex, f.st)
		request := settingsRequest(t, f.ex)
		request.ModelChange = &modelconfig.ModelChange{ModelID: load.ModelID, ExpectedPolicyRevision: "0", Load: &patch}
		before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
		memory := &autoTestMemoryOwner{}
		ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: memory})
		preview, err := f.ex.PreviewLaunch(request, ctx)
		if err != nil {
			t.Fatalf("PreviewLaunch: %v", err)
		}
		message := preview.Errors["modelLoads."+field]
		if message == "" || !strings.Contains(message, field) {
			t.Fatalf("preview errors = %v, want named modelLoads.%s refusal", preview.Errors, field)
		}
		if memory.calls != 0 {
			t.Fatalf("invalid merged load read memory telemetry %d times", memory.calls)
		}
		requireNativeHelpSpawns(t, marker, 0)
		if !reflect.DeepEqual(f.st.plat.Runtime.ModelLoads, before) || !modelconfig.EqualLoads(preview.Settings.ModelLoads, before) {
			t.Fatal("refused preview changed saved/applied loads or its baseline reply")
		}
	})
}

func TestNativeFitSavedFieldPatchExplicitClears(t *testing.T) {
	off := "off"
	test := func(name string, patch modelconfig.LoadPatch, wantFit *string) {
		t.Run(name, func(t *testing.T) {
			saved := []modelconfig.ModelLoad{{ModelID: "model", Requested: savedNativeFitRequested()}}
			got, err := patchModelLoad(saved, "model", patch)
			if err != nil {
				t.Fatalf("atomic clear: %v", err)
			}
			want := modelconfig.RequestedLoad{Fit: wantFit}
			if len(got) != 1 || !reflect.DeepEqual(got[0].Requested, want) {
				t.Fatalf("cleared load = %+v, want %+v", got, want)
			}
			requireSavedNativeFitFields(t, saved, "model", "512", 4096)
		})
	}
	test("fit off and both dependent clears", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Fit: &off}, Clear: []string{"fitTarget", "fitCtx"}}, &off)
	test("fit and both dependents cleared", modelconfig.LoadPatch{Clear: []string{"fit", "fitTarget", "fitCtx"}}, nil)
}

func TestNativeFitSavedFieldPatchPreviewExplicitClears(t *testing.T) {
	off := "off"
	test := func(name string, patch modelconfig.LoadPatch, wantFit *string) {
		t.Run(name, func(t *testing.T) {
			f := appliedNativeFitPatchFixture(t)
			requireSavedNativeFitFields(t, f.st.plat.Runtime.ModelLoads, f.request.ModelID, "512", 4096)
			request := nativeOptionsPreviewRequest(t, f, nil)
			request.ModelChange.Load = &patch
			before := modelconfig.CloneLoads(f.st.plat.Runtime.ModelLoads)
			preview, err := f.ex.PreviewLaunch(request)
			if err != nil || len(preview.Errors) != 0 {
				t.Fatalf("atomic clear preview: err=%v fields=%v", err, preview.Errors)
			}
			found := false
			for _, load := range preview.Settings.ModelLoads {
				if load.ModelID == f.request.ModelID {
					found = true
					if !reflect.DeepEqual(load.Requested.Fit, wantFit) || load.Requested.FitTarget != nil || load.Requested.FitCtx != nil {
						t.Fatalf("explicit clears not preserved: %+v", load.Requested)
					}
				}
			}
			if !found || !reflect.DeepEqual(f.st.plat.Runtime.ModelLoads, before) {
				t.Fatal("preview omitted target or changed saved state")
			}
		})
	}
	test("fit off and both dependent clears", modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Fit: &off}, Clear: []string{"fitTarget", "fitCtx"}}, &off)
	test("fit and both dependents cleared", modelconfig.LoadPatch{Clear: []string{"fit", "fitTarget", "fitCtx"}}, nil)
}

func TestNativeFitSavedFieldPatchInvalidLeaves(t *testing.T) {
	test := func(name string, set modelconfig.RequestedLoad, field string) {
		t.Run(name, func(t *testing.T) {
			saved := []modelconfig.ModelLoad{{ModelID: "model", Requested: savedNativeFitRequested()}}
			if got, err := patchModelLoad(saved, "model", modelconfig.LoadPatch{Set: set}); err == nil || got != nil || !strings.Contains(err.Error(), field) {
				t.Fatalf("invalid leaf patch = %+v, err=%v, want %s refusal", got, err, field)
			}
			requireSavedNativeFitFields(t, saved, "model", "512", 4096)
			f := newModelLoadFixture(t)
			rt := convertModelLoadFixture(t, f)
			load := &rt.ModelLoads[0]
			load.Requested, load.Resolved = savedNativeFitRequested(), nil
			f.st.plat.Runtime = rt
			marker, _ := nativeOptionsFixture(t, f.ex, f.st)
			request := settingsRequest(t, f.ex)
			request.ModelChange = &modelconfig.ModelChange{ModelID: load.ModelID, ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: set}}
			memory := &autoTestMemoryOwner{}
			ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: memory})
			preview, err := f.ex.PreviewLaunch(request, ctx)
			if err != nil || !strings.Contains(preview.Errors["modelLoads"], field) {
				t.Fatalf("invalid leaf preview = %v, err=%v, want %s refusal", preview.Errors, err, field)
			}
			if memory.calls != 0 {
				t.Fatalf("invalid leaf read telemetry %d times", memory.calls)
			}
			requireNativeHelpSpawns(t, marker, 0)
			requireSavedNativeFitFields(t, f.st.plat.Runtime.ModelLoads, load.ModelID, "512", 4096)
		})
	}
	badTarget, largeTarget, negative, large := "512,", "10000000", int32(-1), int32(16777217)
	test("malformed target", modelconfig.RequestedLoad{FitTarget: &badTarget}, "fitTarget")
	test("target bound", modelconfig.RequestedLoad{FitTarget: &largeTarget}, "fitTarget")
	test("context lower bound", modelconfig.RequestedLoad{FitCtx: &negative}, "fitCtx")
	test("context upper bound", modelconfig.RequestedLoad{FitCtx: &large}, "fitCtx")
}

func TestNativeFitSavedFieldPatchRetainsOtherMergedRelationships(t *testing.T) {
	test := func(name string, editSaved func(*modelconfig.RequestedLoad), set modelconfig.RequestedLoad, wantError string) {
		t.Run(name, func(t *testing.T) {
			requested := savedNativeFitRequested()
			editSaved(&requested)
			if err := requested.Validate(); err != nil {
				t.Fatalf("saved setup: %v", err)
			}
			saved := []modelconfig.ModelLoad{{ModelID: "model", Requested: requested}}
			before := modelconfig.CloneLoads(saved)
			if err := (modelconfig.LoadPatch{Set: set}).Validate(); err != nil {
				t.Fatalf("valid patch shape refused before merge: %v", err)
			}
			got, err := patchModelLoad(saved, "model", modelconfig.LoadPatch{Set: set})
			if err == nil || !strings.Contains(err.Error(), wantError) || got != nil {
				t.Fatalf("merged relationship = %+v, err=%v, want %s refusal", got, err, wantError)
			}
			if !reflect.DeepEqual(saved, before) {
				t.Fatal("refused merge changed saved load")
			}
		})
	}
	batch, ubatch, slots := int32(32), int32(64), int32(2)
	test("microbatch inherits smaller batch", func(r *modelconfig.RequestedLoad) { r.BatchSize = &batch }, modelconfig.RequestedLoad{UBatchSize: &ubatch}, "ubatchSize")
	test("batch inherits larger microbatch", func(r *modelconfig.RequestedLoad) { r.UBatchSize = &ubatch }, modelconfig.RequestedLoad{BatchSize: &batch}, "ubatchSize")
	test("slots inherit large context", func(r *modelconfig.RequestedLoad) { r.Context = &modelconfig.Context{Mode: "fixed", Tokens: 1 << 30} }, modelconfig.RequestedLoad{Slots: &slots}, "total context exceeds int32")
	test("context inherits multiple slots", func(r *modelconfig.RequestedLoad) { r.Slots = &slots }, modelconfig.RequestedLoad{Context: &modelconfig.Context{Mode: "fixed", Tokens: 1 << 30}}, "total context exceeds int32")
}
