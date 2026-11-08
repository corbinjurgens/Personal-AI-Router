// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"maps"
	"path/filepath"
	"strings"

	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

type modelConfigurationArtifact struct{ Role, Identity string }

func modelConfigurationArtifacts(bindings []modelconfig.ArtifactSummary) []modelConfigurationArtifact {
	artifacts := make([]modelConfigurationArtifact, 0, len(bindings))
	for _, binding := range bindings {
		artifacts = append(artifacts, modelConfigurationArtifact{binding.Role, binding.Identity})
	}
	return artifacts
}

// These identities scope advisory records only. Telemetry and resolver evidence
// never become configuration identity or launch authority.
func modelConfigurationPools(in modelEstimateInput) []string {
	if in.Allocation != nil && modelLoadUnpriced(in.Allocation.Options) {
		return nil
	}
	// Missing or expired topology cannot prove a non-UMA host. In particular,
	// never mint a second host/device record scope when shared evidence expires.
	if in.Memory == nil || in.Memory.Target == nil {
		return nil
	}
	observed := in.Memory.Target.At(noderec.MemoryBootClock)
	if observed == nil {
		return nil
	}
	hostKind := ""
	for _, pool := range observed.Domains {
		if (pool.Kind == "system" || pool.Kind == "shared") && (pool.Total.Valid || pool.Free.Valid) {
			hostKind = pool.Kind
		}
	}
	if hostKind == "" {
		return nil
	}
	if hostKind == "shared" {
		return []string{"shared"}
	}
	if in.Allocation != nil && (in.Allocation.GPU.Mode == "cpu" || in.Allocation.GPU.Mode == "layers" && in.Allocation.GPU.Layers != nil && *in.Allocation.GPU.Layers == 0) {
		return []string{"host"}
	}
	return []string{"host", "device"}
}

func modelConfigurationOptions(a modelconfig.ResolvedLoad) modelconfig.RequestedLoad {
	options := a.Options.Clone()
	options.Context = &modelconfig.Context{Mode: "fixed", Tokens: a.PerSlotContext}
	slots := a.Slots
	options.Slots = &slots
	gpu := a.Clone().GPU
	options.GPU = &gpu
	return options
}

func modelConfigurationOptionsID(options modelconfig.RequestedLoad, snapshot []byte) string {
	return modelBufferHash(struct {
		Options modelconfig.RequestedLoad
		Preset  string
	}{options, modelLoadDigest(snapshot)})
}

// source binds the content-based native build ID and filtered launch/native
// environment. Build paths and mtimes are evidence checks, not key inputs.
func modelConfigurationKey(in modelEstimateInput, source, optionsID string) string {
	return modelConfigurationKeyForPools(in, source, optionsID, modelConfigurationPools(in))
}

// A diagnostic retains accepted pool kinds while it holds exclusive execution.
// This overload carries identity only, never renewed telemetry or fit authority.
func modelConfigurationKeyForPools(in modelEstimateInput, source, optionsID string, pools []string) string {
	a := in.Allocation
	if a != nil && modelLoadUnpriced(a.Options) {
		return ""
	}
	if a == nil || source == "" || optionsID == "" || a.PerSlotContext <= 0 || a.Slots <= 0 || len(in.Bindings) == 0 {
		return ""
	}
	if len(pools) == 0 {
		return ""
	}
	placement := "cpu"
	if a.GPU.Mode != "cpu" {
		if a.GPU.Mode != "layers" || a.GPU.Layers == nil {
			return ""
		}
		if *a.GPU.Layers > 0 {
			placement = "unbound-gpu"
			if a.DeviceBinding != nil {
				placement = a.DeviceBinding.DeviceID
			}
		}
	}
	return modelBufferHash(struct {
		Artifacts                  []modelConfigurationArtifact
		Source, Options, Placement string
		Pools                      []string
	}{modelConfigurationArtifacts(in.Bindings), source, optionsID, placement, pools})
}

// Execution identity names an applied unpriced launch, never finite capacity.
func modelExecutionKey(in modelEstimateInput, source, optionsID string) string {
	if in.Allocation == nil || !modelLoadUnpriced(in.Allocation.Options) || in.Allocation.NativeBuildID == "" || source == "" || optionsID == "" || len(in.Bindings) == 0 {
		return ""
	}
	return modelBufferHash(struct {
		Artifacts              []modelConfigurationArtifact
		Build, Source, Options string
	}{modelConfigurationArtifacts(in.Bindings), in.Allocation.NativeBuildID, source, optionsID})
}

// Measurement shares applied option and native-source construction with the
// advisory keys, but excludes file locations/stat evidence and process endpoints.
// Artifact identity retains the existing sampled-content assurance, not a full
// model hash. Unrepresented file-valued options remain unknown.
func modelMeasurementSemantic(ctx context.Context, st *engineState, command launchCommand, prepared *llamaServingLaunch, in modelEstimateInput, key string) string {
	if in.Allocation == nil || prepared == nil || len(in.Allocation.Artifacts) == 0 {
		return ""
	}
	sections, err := parseLlamaServingPresetAllowing(prepared.Snapshot, modelLoadAllowedPresetKeys(prepared.ModelLoads))
	if err != nil || sections[key] == nil {
		return ""
	}
	options := maps.Clone(sections[key])
	for _, name := range []string{"model", "model-draft", "mmproj"} {
		delete(options, name)
	}
	for name, value := range options {
		if modelMeasurementFileOption(name, value) {
			return ""
		}
	}
	type artifact struct{ Role, Sample, Size, Assurance string }
	artifacts := make([]artifact, 0, len(in.Allocation.Artifacts))
	for _, binding := range in.Allocation.Artifacts {
		if binding.Validate() != nil || binding.SampleSHA256 == "" {
			return ""
		}
		artifacts = append(artifacts, artifact{binding.Role, binding.SampleSHA256, binding.Size, "sampled-local-binding"})
	}
	semantic := launchCommand{Bin: command.Bin, Env: maps.Clone(command.Env)}
	if semantic.Env == nil {
		semantic.Env = map[string]string{}
	}
	rt := st.plat.Runtime
	if rt.EditableLaunch != nil {
		for _, control := range rt.EditableLaunch.Controls {
			for _, name := range control.Env {
				semantic.Env[name] = ""
			}
		}
	}
	// Override inherited native endpoints and preset paths in this identity-only
	// command too. The source owner still validates the real executable/libraries.
	for _, name := range []string{"LLAMA_ARG_PORT", "LLAMA_ARG_HOST", "LLAMA_ARG_MODELS_PRESET"} {
		semantic.Env[name] = ""
	}
	for _, assignment := range modelEffectiveEnvironment(semantic.Env) {
		name, value, _ := strings.Cut(assignment, "=")
		_, explicit := semantic.Env[name]
		if !explicit && !strings.HasPrefix(name, "LLAMA_") && !strings.HasPrefix(name, "GGML_") && !strings.HasPrefix(name, "CUDA_") {
			continue
		}
		// The native-source owner validates and normalizes its confined loader
		// directory. Other file-valued environment inputs have no artifact proof.
		if name != "LD_LIBRARY_PATH" && modelMeasurementFileOption(name, value) {
			return ""
		}
	}
	for i := 0; i < len(command.Args); {
		if rt.EditableLaunch != nil {
			control, _, last, err := rt.EditableLaunch.readControl(command.Args, i)
			if err != nil {
				return ""
			}
			if control != nil {
				i = last + 1
				continue
			}
		}
		name, values := llamaLaunchArgument(command.Args[i])
		end := i + 1 + values
		if end > len(command.Args) {
			return ""
		}
		if name != "models-preset" {
			for _, value := range command.Args[i:end] {
				if modelMeasurementFileOption(name, value) {
					return ""
				}
			}
			semantic.Args = append(semantic.Args, command.Args[i:end]...)
		}
		i = end
	}
	source, err := modelReviewedNativeSource(ctx, st, semantic)
	if err != nil {
		return ""
	}
	return modelBufferHash(struct {
		Artifacts          []artifact
		Source             string
		Load               modelconfig.RequestedLoad
		Options            map[string]string
		Arguments          []string
		WorkingSetLimitMiB *int
	}{artifacts, source, modelConfigurationOptions(*in.Allocation), options, semantic.Args, rt.WorkingSetLimitMiB})
}

func modelMeasurementFileOption(name, value string) bool {
	return filepath.IsAbs(value) || strings.ContainsAny(value, `/\`) || strings.HasSuffix(name, "-file") || strings.HasSuffix(name, "-path")
}
