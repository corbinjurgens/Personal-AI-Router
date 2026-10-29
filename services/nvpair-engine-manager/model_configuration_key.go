// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

// These identities scope advisory records only. Telemetry and resolver evidence
// never become configuration identity or launch authority.
func modelConfigurationPools(in modelEstimateInput) []string {
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

func modelConfigurationKey(in modelEstimateInput, source, optionsID string) string {
	a := in.Allocation
	if a == nil || source == "" || optionsID == "" || a.PerSlotContext <= 0 || a.Slots <= 0 || len(in.Bindings) == 0 {
		return ""
	}
	pools := modelConfigurationPools(in)
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
	type artifact struct{ Role, Identity string }
	artifacts := make([]artifact, 0, len(in.Bindings))
	for _, binding := range in.Bindings {
		artifacts = append(artifacts, artifact{binding.Role, binding.Identity})
	}
	return modelBufferHash(struct {
		Artifacts                  []artifact
		Source, Options, Placement string
		Pools                      []string
	}{artifacts, source, optionsID, placement, pools})
}
