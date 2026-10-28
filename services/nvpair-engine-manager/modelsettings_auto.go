// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"nvpair-shared/modelconfig"
)

const modelAutoRevision = "auto/finite-llama/1"
const modelAutoBudget = 128
const modelAutoReserve = "Advisory uncalibrated heuristic: reserve 10 percent of capacity (rounded up) plus 512 MiB per independent pool; compare against snapshot free bytes. No qualification or admission guarantee."

// Only a target owner may supply capacity. There is deliberately no production
// implementation or RPC input: node-info utilization is not memory telemetry.
type modelAutoMemoryOwner interface {
	ModelAutoMemory(context.Context) (*modelEstimateMemory, error)
}

// A cost owner may prove the narrow dense transfer optimization by deriving a
// complete, layer-independent non-weight host cost for this exact context and
// options. This is a cost-model contract, never inferred from sampled estimates
// or architecture names. No production estimator currently supplies this proof.
type modelAutoLayerCostOwner interface {
	ModelAutoFixedHostCost(context.Context, modelEstimateInput) (uint64, bool)
}

// Private operation dependencies, never serialized or retained in Runtime.
type modelAutoOperation struct {
	Memory           modelAutoMemoryOwner
	LayerCosts       modelAutoLayerCostOwner
	ConversionSource string
	RecalculateModel string
	Proposals        map[string]modelconfig.AutoProposal
}
type modelAutoOperationKey struct{}

func modelLoadHasAuto(r modelconfig.RequestedLoad) bool {
	return r.Context != nil && r.Context.Mode == "auto" || r.GPU != nil && r.GPU.Mode == "auto"
}

// The no-facts entry point cannot recommend anything without owner inputs.
func resolveModelSettingsAuto(ctx context.Context, p modelconfig.InspectRequest, requested modelconfig.RequestedLoad) (modelconfig.AutoProposal, error) {
	estimate, err := estimateModelSettings(ctx, p, requested)
	if err != nil {
		return modelconfig.AutoProposal{}, err
	}
	if !modelLoadHasAuto(requested) {
		return modelconfig.AutoProposal{}, modelSettingsParamsError(fmt.Errorf("modelChange.load: select explicit Auto preferences"))
	}
	return modelconfig.AutoProposal{Status: "unavailable", Requested: requested.Clone(), Estimate: estimate},
		modelSettingsParamsError(fmt.Errorf("target memory telemetry unavailable"))
}

// Auto rounds down its cap and up its explicit floor. Fixed context uses native
// padding and is never reduced. No automatic RoPE extension is enabled.
func modelAutoContexts(r modelconfig.RequestedLoad, supported *int32) (int32, int32, error) {
	cap, floor := int64(8192), int64(4096)
	auto := r.Context == nil || r.Context.Mode == "auto"
	if r.Context != nil && auto {
		cap, floor = int64(r.Context.Cap), int64(r.Context.Floor)
	} else if r.Context != nil {
		cap, floor = int64(r.Context.Tokens), int64(r.Context.Tokens)
	}
	if auto {
		floor = (floor + 255) / 256 * 256
		cap = cap / 256 * 256
	} else {
		cap = (cap + 255) / 256 * 256
		floor = cap
	}
	if supported == nil || *supported <= 0 {
		return 0, 0, fmt.Errorf("model supported context unavailable; automatic RoPE extension is disabled")
	}
	if cap > int64(*supported) {
		if !auto {
			return 0, 0, fmt.Errorf("fixed context exceeds known model context; automatic RoPE extension is disabled")
		}
		cap = int64(*supported) / 256 * 256
	}
	if floor <= 0 || cap < floor || cap > math.MaxInt32 {
		return 0, 0, fmt.Errorf("context: no valid 256-token candidate within the requested range and known model context")
	}
	return int32(cap), int32(floor), nil
}

// This proves only dense output-first WEIGHT transfer, not total cost. Bisection
// additionally requires a complete fixed host cost from the cost owner, checked
// against each evaluated allocation. Without that proof, enumerate honestly.
func modelAutoWeightTransferMonotonic(in modelEstimateInput, r modelconfig.RequestedLoad) (proven bool) {
	defer func() {
		if recover() != nil {
			proven = false
		}
	}()
	m, reason := modelEstimateAdapt(context.Background(), in.Facts)
	if reason != "" || m.weightsUnknown || m.companions || !slices.Contains([]string{"llama", "qwen2", "qwen3"}, m.arch) || r.KVOffload == nil || *r.KVOffload || r.OpOffload == nil || *r.OpOffload || in.Memory == nil || in.Memory.UMA {
		return false
	}
	for key := range m.numeric {
		if strings.Contains(key, ".ssm.") || strings.Contains(key, ".shortconv.") || strings.Contains(key, ".wkv.") {
			return false
		}
	}
	for _, key := range []string{"expert_count", "attention.sliding_window", "attention.kv_lora_rank", "full_attention_interval", "attention.shared_kv_layers", "nextn_predict_layers"} {
		if n, exists := m.numeric[m.arch+"."+key]; exists {
			v, ok := modelEstimateUint(n)
			if !ok || v != 0 {
				return false
			}
		}
	}
	return true
}

// Missing components, unknown domains and incomplete multi-device allocation
// cannot fit. Direction is meaningful only with the complete total-cost proof:
// host needs more offload; device needs less (device upper bound wins).
func modelAutoFit(out modelconfig.Estimate, memory *modelEstimateMemory, now time.Time, gpu modelconfig.GPU) (known, fits bool, direction int) {
	if memory == nil || !modelEstimateMemoryFresh(memory, now) || out.Status != "estimated" || len(out.Unknowns) != 0 || len(out.Domains) == 0 || len(out.Components) == 0 || out.Validate() != nil {
		return false, false, 0
	}
	if len(memory.Domains) == 0 || len(memory.Domains) > 64 || (modelconfig.Estimate{Status: "unknown", Domains: memory.Domains}).Validate() != nil {
		return false, false, 0
	}
	cpu := gpu.Mode == "cpu" || gpu.Mode == "layers" && gpu.Layers != nil && *gpu.Layers == 0
	if memory.UMA {
		if len(memory.Domains) != 1 || memory.Domains[0].Kind != "shared" || len(out.Domains) != 1 {
			return false, false, 0
		}
	} else {
		hosts, devices := 0, 0
		for _, d := range memory.Domains {
			switch d.Kind {
			case "host":
				hosts++
			case "device":
				devices++
				if !cpu && (d.DeviceID == "" || d.DeviceID != memory.NativeDeviceID) {
					return false, false, 0
				}
			default:
				return false, false, 0
			}
		}
		if hosts != 1 || !cpu && devices != 1 {
			return false, false, 0
		}
	}
	seen := map[string]bool{}
	for _, d := range out.Domains {
		index := slices.IndexFunc(memory.Domains, func(m modelconfig.MemoryDomain) bool {
			return m.ID == d.ID && m.Kind == d.Kind && m.DeviceID == d.DeviceID
		})
		if index < 0 || seen[d.ID] {
			return false, false, 0
		}
		seen[d.ID] = true
		pool := memory.Domains[index]
		capacity, ce := strconv.ParseUint(pool.CapacityBytes, 10, 64)
		free, fe := strconv.ParseUint(pool.FreeBytes, 10, 64)
		required, re := strconv.ParseUint(d.RequiredBytes, 10, 64)
		if ce != nil || fe != nil || re != nil || capacity == 0 || free > capacity || cpu && d.Kind == "device" && required != 0 {
			return false, false, 0
		}
		reserve := capacity/10 + uint64(512<<20)
		if capacity%10 != 0 {
			reserve++
		}
		if reserve > free || required > free-reserve {
			if d.Kind == "device" {
				direction = -1
			} else if direction == 0 {
				direction = 1
			}
		}
	}
	for _, d := range memory.Domains {
		if (d.Kind == "host" || d.Kind == "shared" || !cpu && d.Kind == "device") && !seen[d.ID] {
			return false, false, 0
		}
	}
	components := map[string]bool{}
	sums := map[string]uint64{}
	for _, c := range out.Components {
		components[c.Name] = true
		if c.Status == "excluded" {
			if c.Name != "companions" && c.Name != "recurrent-state" && c.Name != "known-exclusions" {
				return false, false, 0
			}
			continue
		}
		if c.Status != "estimated" || c.Bytes == "" || !seen[c.DomainID] {
			return false, false, 0
		}
		n, err := strconv.ParseUint(c.Bytes, 10, 64)
		if err != nil || n > math.MaxUint64-sums[c.DomainID] {
			return false, false, 0
		}
		sums[c.DomainID] += n
	}
	for _, name := range []string{"weights", "attention-cache", "recurrent-state", "compute-workspace", "prompt-cache", "companions"} {
		if !components[name] {
			return false, false, 0
		}
	}
	for _, d := range out.Domains {
		if strconv.FormatUint(sums[d.ID], 10) != d.RequiredBytes {
			return false, false, 0
		}
	}
	return true, direction == 0, direction
}

// Verify the priced transfer against the existing estimator's output-first
// mapping, rather than using architecture names to certify arbitrary costs.
func modelAutoTransfer(in modelEstimateInput, out modelconfig.Estimate, fixedHostCost uint64) (proven bool) {
	defer func() {
		if recover() != nil {
			proven = false
		}
	}()
	m, reason := modelEstimateAdapt(context.Background(), in.Facts)
	if reason != "" {
		return false
	}
	projection := modelconfig.Estimate{Components: []modelconfig.EstimateComponent{estimateComponent("weights", m.weights, "")}}
	modelEstimatePlacement(&projection, m, in)
	weights := map[string]uint64{}
	var hostCost uint64
	for _, c := range out.Components {
		if c.Status == "excluded" {
			continue
		}
		i := slices.IndexFunc(out.Domains, func(d modelconfig.MemoryDomain) bool { return d.ID == c.DomainID })
		if i < 0 {
			return false
		}
		if c.Name != "weights" {
			if out.Domains[i].Kind != "host" {
				return false
			}
			n, err := strconv.ParseUint(c.Bytes, 10, 64)
			if err != nil || n > math.MaxUint64-hostCost {
				return false
			}
			hostCost += n
			continue
		}
		n, err := strconv.ParseUint(c.Bytes, 10, 64)
		if err != nil || n > math.MaxUint64-weights[out.Domains[i].Kind] {
			return false
		}
		weights[out.Domains[i].Kind] += n
	}
	return hostCost == fixedHostCost && strconv.FormatUint(weights["host"], 10) == projection.HostBytes && strconv.FormatUint(weights["device"], 10) == projection.DeviceBytes
}

// Resolves Preview from already validated bounded facts. Only an explicit
// preview/recalculate reaches here; Apply and restart use modelAutoFrozen.
func resolveModelSettingsAutoInput(ctx context.Context, model string, requested modelconfig.RequestedLoad, in modelEstimateInput, op modelAutoOperation, estimates *modelEstimateState) (modelconfig.AutoProposal, error) {
	out := modelconfig.AutoProposal{Status: "unavailable", Requested: requested.Clone(), ResolverRevision: modelAutoRevision, Estimate: modelEstimateUnknown("target memory telemetry unavailable"), Warnings: []string{modelAutoReserve}}
	fail := func(reason string) (modelconfig.AutoProposal, error) {
		return out, modelLoadError(model, "auto", reason)
	}
	if !modelLoadHasAuto(requested) || requested.Validate() != nil {
		return fail("Select valid explicit Auto preferences.")
	}
	if op.Memory == nil {
		return out, modelSettingsParamsError(fmt.Errorf("target memory telemetry unavailable"))
	}
	memory, err := op.Memory.ModelAutoMemory(ctx)
	if err != nil {
		return fail(err.Error())
	}
	if memory == nil {
		return out, modelSettingsParamsError(fmt.Errorf("target memory telemetry unavailable"))
	}
	in.Memory = memory
	if !modelEstimateMemoryFresh(memory, estimates.clock()) {
		return fail("Target memory snapshot is stale or incomplete.")
	}
	if in.Allocation == nil {
		return fail("Bounded artifact allocation unavailable.")
	}
	bounded := in
	bounded.Allocation = nil // candidates are compiled/checked separately below
	if reason := modelEstimateBounds(ctx, bounded); reason != "" {
		return fail(reason)
	}
	facts, _ := projectInspectionFacts(in.Facts)
	cap, floor, err := modelAutoContexts(requested, facts.ContextTokens)
	if err != nil {
		return fail(err.Error())
	}
	if facts.LayerCount == nil || *facts.LayerCount < 0 || *facts.LayerCount > modelEstimateLayers {
		return fail("Bounded model layer count unavailable.")
	}
	maxLayers := *facts.LayerCount + 1
	monotonic := op.LayerCosts != nil && requested.GPU != nil && requested.GPU.Mode == "auto" && modelAutoWeightTransferMonotonic(in, requested)
	var fixedHostCost uint64
	proofRead, proofKnown := false, false
	evaluations, unknown := 0, false
	var selected *modelconfig.ResolvedLoad
	evaluate := func(tokens int32, gpu modelconfig.GPU) (bool, bool, int, error) {
		if err := ctx.Err(); err != nil {
			return false, false, 0, err
		}
		if evaluations >= modelAutoBudget {
			out.Status = "budget-exhausted"
			return false, false, 0, fmt.Errorf("budget-exhausted: unsearched context %d..%d, GPU layers 0..%d (128 evaluations)", floor, tokens, maxLayers)
		}
		candidate := requested.Clone()
		candidate.Context, candidate.GPU = &modelconfig.Context{Mode: "fixed", Tokens: tokens}, &gpu
		options, allocation, err := typedLlamaModelOptions(model, candidate, facts.LayerCount)
		if err != nil {
			return false, false, 0, err
		}
		allocation.Artifacts, allocation.NativeBuildID = slices.Clone(in.Allocation.Artifacts), in.BuildID
		effective := allocation.Clone()
		if effective.GPU.Mode == "layers" && effective.GPU.Layers != nil && *effective.GPU.Layers == 0 {
			effective.GPU = modelconfig.GPU{Mode: "cpu"}
			effective.Options.GPU = &effective.GPU
		}
		if cache, err := strconv.ParseInt(options["cache-ram"], 10, 32); err == nil {
			n := int32(cache)
			effective.Options.CacheRAM = &n
		}
		candidateInput := in
		candidateInput.Allocation = &effective
		evaluations++
		out.Estimate = estimates.estimate(ctx, candidateInput)
		known, fits, direction := modelAutoFit(out.Estimate, memory, estimates.clock(), allocation.GPU)
		if monotonic && known {
			if !proofRead {
				// The proof is derived once per context from CPU options, not
				// by observing an arbitrary offloaded workspace sample.
				proofInput := candidateInput
				cpu := effective.Clone()
				cpu.GPU = modelconfig.GPU{Mode: "cpu"}
				cpu.Options.GPU = &cpu.GPU
				proofInput.Allocation = &cpu
				fixedHostCost, proofKnown = op.LayerCosts.ModelAutoFixedHostCost(ctx, proofInput)
				proofRead = true
			}
			if !proofKnown || !modelAutoTransfer(candidateInput, out.Estimate, fixedHostCost) {
				monotonic = false
			}
		}
		unknown = unknown || !known
		if fits {
			selected = allocation
		}
		return known, fits, direction, nil
	}
	for tokens := cap; tokens >= floor; tokens -= 256 {
		proofRead, proofKnown = false, false
		gpu := modelconfig.GPU{Mode: "native-default"}
		if requested.GPU != nil {
			gpu = *requested.GPU
		}
		if gpu.Mode != "auto" {
			_, fits, _, err := evaluate(tokens, gpu)
			if err != nil {
				return fail(err.Error())
			}
			if fits {
				break
			}
		} else {
			lo, hi := int32(0), maxLayers
			var best *modelconfig.ResolvedLoad
			var bestEstimate modelconfig.Estimate
			for lo <= hi {
				layers := hi
				if monotonic {
					layers = lo + (hi-lo)/2
				}
				placement := modelconfig.GPU{Mode: "layers", Layers: &layers}
				if layers == 0 {
					placement = modelconfig.GPU{Mode: "cpu"}
				}
				bisecting := monotonic
				known, fits, direction, err := evaluate(tokens, placement)
				if err != nil {
					return fail(err.Error())
				}
				if bisecting && !monotonic {
					// No valid total-cost bound was established. Discard the
					// bisection range and enumerate all placements, including CPU.
					lo, hi, best, selected = 0, maxLayers, nil, nil
					continue
				}
				if fits {
					best, bestEstimate = selected, out.Estimate
					if !monotonic {
						break
					}
					lo = layers + 1
				} else if monotonic && known {
					if direction > 0 {
						lo = layers + 1
					} else {
						hi = layers - 1
					}
				} else {
					// Unknown midpoint gives no bound. Enumerate the entire
					// context under the same budget instead of assuming failure.
					if monotonic {
						monotonic, lo, hi = false, 0, maxLayers
					} else {
						hi--
					}
				}
			}
			if best != nil {
				selected, out.Estimate = best, bestEstimate
				break
			}
		}
		if requested.Context != nil && requested.Context.Mode != "auto" {
			break
		}
	}
	if selected == nil {
		out.Status = "insufficient-memory"
		if unknown {
			out.Status = "unavailable"
		}
		return fail(out.Status + ": no complete fitting candidate in the searched range")
	}
	selected.ResolverRevision = modelAutoRevision
	selected.ReviewedInputID, err = modelAutoReviewedID(model, requested, selected, op.ConversionSource, memory)
	if err != nil {
		return fail(err.Error())
	}
	out.Status, out.Resolved, out.ReviewedInputID = "fits", selected, selected.ReviewedInputID
	out.MemorySource, out.MemoryTimestamp = memory.Source, memory.At.UTC().Format(time.RFC3339Nano)
	out.Estimate.Assumptions = append(out.Estimate.Assumptions, modelAutoReserve)
	if selected.GPU.Mode == "cpu" || selected.GPU.Layers != nil && *selected.GPU.Layers == 0 {
		out.Warnings = append(out.Warnings, "CPU-only placement is slower; no GPU capacity is credited.")
		out.Estimate.Assumptions = append(out.Estimate.Assumptions, "CPU-only placement is slower; no GPU capacity is credited.")
	}
	return out, nil
}

// Compact review envelope: conversion source, memory snapshot hash, and a hash
// binding both to intent, finite options, sampled artifacts and native build.
// Telemetry itself is not needed to compile an accepted resolution again.
func modelAutoReviewedID(model string, requested modelconfig.RequestedLoad, resolved *modelconfig.ResolvedLoad, conversion string, memory *modelEstimateMemory) (string, error) {
	data, err := json.Marshal(memory)
	if err != nil {
		return "", err
	}
	return modelAutoReviewHash(model, requested, resolved, conversion, modelLoadDigest(data))
}

func modelAutoReviewHash(model string, requested modelconfig.RequestedLoad, resolved *modelconfig.ResolvedLoad, conversion, memory string) (string, error) {
	if conversion == "" {
		conversion = "-"
	}
	finite := resolved.Clone()
	finite.SourceInputDigest, finite.PresetDigest, finite.ReviewedInputID = "", "", ""
	finite.ResolverRevision = modelAutoRevision
	data, err := json.Marshal(struct {
		Model, Conversion, Memory, Estimator string
		Requested                            modelconfig.RequestedLoad
		Finite                               modelconfig.ResolvedLoad
	}{model, conversion, memory, modelEstimatorRevision, requested, finite})
	if err != nil {
		return "", err
	}
	return conversion + ":" + memory + ":" + modelLoadDigest(data), nil
}

func modelAutoReviewParts(id string) (string, string, bool) {
	parts := strings.Split(id, ":")
	validDigest := func(s string) bool {
		if len(s) != 64 {
			return false
		}
		for _, c := range s {
			if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
				return false
			}
		}
		return true
	}
	if len(parts) != 3 || parts[0] != "-" && !validDigest(parts[0]) || !validDigest(parts[1]) || !validDigest(parts[2]) {
		return "", "", false
	}
	conversion := parts[0]
	if conversion == "-" {
		conversion = ""
	}
	return conversion, parts[1], true
}

func modelAutoFrozen(model string, requested modelconfig.RequestedLoad, resolved *modelconfig.ResolvedLoad) error {
	if resolved == nil || resolved.ResolverRevision != modelAutoRevision {
		return modelLoadError(model, "auto", "Reviewed finite Auto resolution required; explicitly Recalculate.")
	}
	conversion, memory, ok := modelAutoReviewParts(resolved.ReviewedInputID)
	if !ok {
		return modelLoadError(model, "auto", "Missing or invalid Auto reviewed input identity.")
	}
	id, err := modelAutoReviewHash(model, requested, resolved, conversion, memory)
	if err != nil || id != resolved.ReviewedInputID {
		return modelLoadError(model, "auto", "Auto inputs changed since preview; explicitly Recalculate.")
	}
	if err := (modelconfig.ModelLoad{ModelID: model, Requested: requested, Resolved: resolved}).Validate(); err != nil {
		return modelLoadError(model, "auto", err.Error())
	}
	return nil
}

func modelAutoConversionSource(resolved *modelconfig.ResolvedLoad) string {
	if resolved == nil {
		return ""
	}
	if resolved.ResolverRevision == externalConversionResolver {
		return resolved.ReviewedInputID
	}
	if resolved.ResolverRevision != modelAutoRevision {
		return ""
	}
	conversion, _, _ := modelAutoReviewParts(resolved.ReviewedInputID)
	return conversion
}

// Never consult capacity on finite compilation, including restart/recovery.
func modelAutoCompileFrozen(model string, requested modelconfig.RequestedLoad, previous *modelconfig.ResolvedLoad, layers *int32, artifacts []modelconfig.ArtifactBinding, build string) (map[string]string, *modelconfig.ResolvedLoad, error) {
	if err := modelAutoFrozen(model, requested, previous); err != nil {
		return nil, nil, err
	}
	options, resolved, err := typedLlamaModelOptions(model, previous.Options, layers)
	if err != nil {
		return nil, nil, err
	}
	resolved.Artifacts, resolved.NativeBuildID = artifacts, build
	resolved.ResolverRevision, resolved.ReviewedInputID = previous.ResolverRevision, previous.ReviewedInputID
	if !reflect.DeepEqual(resolved.Artifacts, previous.Artifacts) || resolved.NativeBuildID != previous.NativeBuildID {
		return nil, nil, modelLoadError(model, "auto", "Auto artifact or native build changed; obtain a fresh preview.")
	}
	checked := resolved.Clone()
	checked.SourceInputDigest, checked.PresetDigest = previous.SourceInputDigest, previous.PresetDigest
	if err := modelAutoFrozen(model, requested, &checked); err != nil {
		return nil, nil, err
	}
	return options, resolved, nil
}

func modelAutoCompileCandidate(model string, reviewed *modelconfig.ResolvedLoad, layers *int32) (map[string]string, *modelconfig.ResolvedLoad, error) {
	options, _, err := typedLlamaModelOptions(model, reviewed.Options, layers)
	return options, reviewed, err
}
