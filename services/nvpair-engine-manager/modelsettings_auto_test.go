// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/gguffacts"
	"nvpair-shared/modelconfig"
)

func TestModelSettingsAutoMissingTelemetry(t *testing.T) {
	request := modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "auto"}}
	proposal, err := resolveModelSettingsAuto(context.Background(), modelconfig.InspectRequest{Engine: "llamacpp", ModelID: "example/model:Q4_K_M"}, request)
	if err == nil || err.Error() != "target memory telemetry unavailable" {
		t.Fatalf("error = %v, want target memory telemetry unavailable", err)
	}
	if proposal.Status != "unavailable" || proposal.Resolved != nil {
		t.Fatalf("unexpected proposal: %+v", proposal)
	}
}

type autoTestMemoryOwner struct {
	memory *modelEstimateMemory
	calls  int
}

func (o *autoTestMemoryOwner) ModelAutoMemory(context.Context) (*modelEstimateMemory, error) {
	o.calls++
	return o.memory, nil
}

// This synthetic cost formula has no layer argument: every non-weight component
// is host-only and invariant under offload at a fixed context. The production
// estimator has no such complete cost proof.
type autoTestLayerCosts struct {
	contextWorkspace func(int32) uint64
}

func (o autoTestLayerCosts) ModelAutoFixedHostCost(_ context.Context, in modelEstimateInput) (uint64, bool) {
	cost := uint64(in.Allocation.TotalContext)*32 + 1024 + 1048576
	if o.contextWorkspace != nil {
		cost += o.contextWorkspace(in.Allocation.TotalContext)
	}
	return cost, true
}

func autoTestMemory(now time.Time) *modelEstimateMemory {
	return &modelEstimateMemory{Identity: "synthetic-snapshot", Source: "synthetic-owner", At: now, NativeDeviceID: "physical-gpu", Domains: []modelconfig.MemoryDomain{
		{ID: "host", Kind: "host", CapacityBytes: "8589934592", FreeBytes: "8589934592"},
		{ID: "gpu", Kind: "device", DeviceID: "physical-gpu", CapacityBytes: "8589934592", FreeBytes: "8589934592"},
	}}
}

// A synthetic complete cost owner, not an assertion that Task 10 prices a full
// native graph. It uses the REAL bounded tensor transfer and supplies explicit
// host-only layer-independent costs for the otherwise missing graph components.
func autoTestEstimate(ctx context.Context, in modelEstimateInput) modelconfig.Estimate {
	m, reason := modelEstimateAdapt(ctx, in.Facts)
	if reason != "" {
		return modelEstimateUnknown(reason)
	}
	projection := modelconfig.Estimate{Components: []modelconfig.EstimateComponent{estimateComponent("weights", m.weights, "synthetic")}}
	modelEstimatePlacement(&projection, m, in)
	host, he := strconv.ParseUint(projection.HostBytes, 10, 64)
	device, de := strconv.ParseUint(projection.DeviceBytes, 10, 64)
	if he != nil || de != nil {
		return modelEstimateUnknown("synthetic placement unavailable")
	}
	extra := uint64(in.Allocation.TotalContext)*32 + 1024 + 1048576
	out := modelconfig.Estimate{Status: "estimated", Domains: []modelconfig.MemoryDomain{{ID: "host", Kind: "host", RequiredBytes: strconv.FormatUint(host+extra, 10)}}, Components: []modelconfig.EstimateComponent{
		{Name: "weights", DomainID: "host", Status: "estimated", Bytes: strconv.FormatUint(host, 10)},
		{Name: "attention-cache", DomainID: "host", Status: "estimated", Bytes: strconv.FormatUint(uint64(in.Allocation.TotalContext)*32, 10)},
		{Name: "compute-workspace", DomainID: "host", Status: "estimated", Bytes: "1024"},
		{Name: "prompt-cache", DomainID: "host", Status: "estimated", Bytes: "1048576"},
		{Name: "recurrent-state", Status: "excluded"}, {Name: "companions", Status: "excluded"},
	}}
	if in.Allocation.GPU.Mode != "cpu" {
		out.Domains = append(out.Domains, modelconfig.MemoryDomain{ID: "offload-unmapped", Kind: "device", RequiredBytes: strconv.FormatUint(device, 10)})
		out.Components = append(out.Components, modelconfig.EstimateComponent{Name: "weights", DomainID: "offload-unmapped", Status: "estimated", Bytes: strconv.FormatUint(device, 10)})
	}
	return out
}

func autoTestInput() modelEstimateInput {
	in := estimateFixture("llama")
	in.Facts.Shards[0].Numeric["llama.context_length"] = gguffacts.Numeric{Type: 10, Bits: 32768}
	in.Allocation.Artifacts = []modelconfig.ArtifactBinding{{Role: "model", Path: "synthetic.gguf", ResolvedTarget: "synthetic.gguf", Size: "327936", MtimeNS: "1", SampleSHA256: strings.Repeat("a", 64)}}
	return in
}

func autoTestRequest() modelconfig.RequestedLoad {
	off := false
	f16 := "f16"
	return modelconfig.RequestedLoad{GPU: &modelconfig.GPU{Mode: "auto"}, KVOffload: &off, OpOffload: &off, CacheTypeK: &f16, CacheTypeV: &f16}
}

func TestModelSettingsAutoContextRange(t *testing.T) {
	known := int32(32768)
	test := func(name string, context *modelconfig.Context, cap, floor int32, refusal bool) {
		t.Run(name, func(t *testing.T) {
			r := autoTestRequest()
			r.Context = context
			gotCap, gotFloor, err := modelAutoContexts(r, &known)
			if refusal {
				if err == nil {
					t.Fatalf("accepted range %d..%d", gotFloor, gotCap)
				}
				return
			}
			if err != nil || gotCap != cap || gotFloor != floor {
				t.Fatalf("range=%d..%d err=%v, want %d..%d", gotFloor, gotCap, err, floor, cap)
			}
		})
	}
	test("default", nil, 8192, 4096, false)
	test("explicit cap below default floor", &modelconfig.Context{Mode: "auto", Floor: 2048, Cap: 2048}, 2048, 2048, false)
	test("explicit override", &modelconfig.Context{Mode: "auto", Floor: 4353, Cap: 10000}, 9984, 4608, false)
	test("no grid point", &modelconfig.Context{Mode: "auto", Floor: 100, Cap: 200}, 0, 0, true)
	test("fixed padding never shrinks", &modelconfig.Context{Mode: "fixed", Tokens: 4097}, 4352, 4352, false)
	test("fixed exceeds model", &modelconfig.Context{Mode: "fixed", Tokens: 65536}, 0, 0, true)
	test("no automatic extension", &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 65536}, 32768, 4096, false)
}

func TestModelSettingsAutoPreservesExplicitInputs(t *testing.T) {
	now := time.Now()
	owner := &autoTestMemoryOwner{memory: autoTestMemory(now)}
	state := modelEstimateState{now: func() time.Time { return now }, run: autoTestEstimate}
	r := autoTestRequest()
	three, threads, batch, ubatch := int32(3), int32(6), int32(64), int32(32)
	r.Context = &modelconfig.Context{Mode: "fixed", Tokens: 4097}
	r.Slots, r.Threads, r.BatchSize, r.UBatchSize = &three, &threads, &batch, &ubatch
	before := r.Clone()
	p, err := resolveModelSettingsAutoInput(context.Background(), "exact", r, autoTestInput(), modelAutoOperation{Memory: owner}, &state)
	if err != nil || p.Status != "fits" {
		t.Fatalf("proposal=%+v err=%v", p, err)
	}
	if !reflect.DeepEqual(before, r) || p.Resolved.PerSlotContext != 4352 || p.Resolved.TotalContext != 13056 || p.Resolved.Slots != 3 {
		t.Fatalf("changed context/slots: %+v", p.Resolved)
	}
	got := p.Resolved.Options
	if !reflect.DeepEqual(got.CacheTypeK, r.CacheTypeK) || !reflect.DeepEqual(got.CacheTypeV, r.CacheTypeV) || !reflect.DeepEqual(got.Threads, r.Threads) || !reflect.DeepEqual(got.BatchSize, r.BatchSize) || !reflect.DeepEqual(got.KVOffload, r.KVOffload) {
		t.Fatal("manual/quality inputs changed")
	}
	layers := int32(2)
	options, _, err := modelAutoCompileCandidate("exact", p.Resolved, &layers)
	if err != nil || options["fit"] != "off" || options["parallel"] != "3" || options["ctx-size"] != "13056" {
		t.Fatalf("native options=%v err=%v", options, err)
	}
}

func TestModelSettingsAutoDenseTransferMonotonic(t *testing.T) {
	for _, arch := range []string{"llama", "qwen2", "qwen3"} {
		t.Run(arch, func(t *testing.T) {
			in := estimateFixture(arch)
			in.Memory = autoTestMemory(time.Now())
			r := autoTestRequest()
			if !modelAutoWeightTransferMonotonic(in, r) {
				t.Fatal("dense transfer proof unavailable")
			}
			previousHost, previousDevice := uint64(^uint64(0)), uint64(0)
			m, reason := modelEstimateAdapt(context.Background(), in.Facts)
			if reason != "" {
				t.Fatal(reason)
			}
			for layers := int32(0); layers <= 3; layers++ {
				in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
				out := modelconfig.Estimate{Components: []modelconfig.EstimateComponent{estimateComponent("weights", m.weights, "")}}
				modelEstimatePlacement(&out, m, in)
				h, he := strconv.ParseUint(out.HostBytes, 10, 64)
				d, de := strconv.ParseUint(out.DeviceBytes, 10, 64)
				if he != nil || de != nil || h > previousHost || d < previousDevice || h+d != m.weights {
					t.Fatalf("nonmonotonic transfer layers=%d host=%s device=%s", layers, out.HostBytes, out.DeviceBytes)
				}
				previousHost, previousDevice = h, d
			}
		})
	}
	in := estimateFixture("lfm2")
	in.Memory = autoTestMemory(time.Now())
	if modelAutoWeightTransferMonotonic(in, autoTestRequest()) {
		t.Fatal("unproven recurrent architecture used bisection")
	}
	in = estimateFixture("llama")
	in.Memory = autoTestMemory(time.Now())
	r := autoTestRequest()
	on := true
	r.KVOffload = &on
	if modelAutoWeightTransferMonotonic(in, r) {
		t.Fatal("per-layer KV cost used bisection")
	}
}

func TestModelSettingsAutoFloorOnlyFitWithinBudget(t *testing.T) {
	now := time.Now()
	owner := &autoTestMemoryOwner{memory: autoTestMemory(now)}
	owner.memory.Domains[0].CapacityBytes, owner.memory.Domains[0].FreeBytes = "1073741824", "1073741824"
	in := autoTestInput()
	in.Facts.Shards[0].Numeric["llama.block_count"] = gguffacts.Numeric{Type: 10, Bits: 80}
	workspace := func(tokens int32) uint64 { return uint64(tokens-4096) * (4 << 20) }
	proof := autoTestLayerCosts{contextWorkspace: workspace}
	calls := 0
	state := modelEstimateState{now: func() time.Time { return now }, run: func(ctx context.Context, in modelEstimateInput) modelconfig.Estimate {
		calls++
		out := autoTestEstimate(ctx, in)
		extra := workspace(in.Allocation.TotalContext)
		h, err := strconv.ParseUint(out.Domains[0].RequiredBytes, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		out.Domains[0].RequiredBytes = strconv.FormatUint(h+extra, 10)
		out.Components = append(out.Components, modelconfig.EstimateComponent{Name: "synthetic-context-workspace", Status: "estimated", DomainID: "host", Bytes: strconv.FormatUint(extra, 10)})
		return out
	}}
	// Assert the complete total-cost proof, not just monotonic tensor weights.
	previousHost, previousDevice := ^uint64(0), uint64(0)
	for layers := int32(0); layers <= 81; layers++ {
		candidate := in
		allocation := in.Allocation.Clone()
		allocation.TotalContext = 4096
		allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
		candidate.Allocation = &allocation
		out := autoTestEstimate(context.Background(), candidate)
		cost, ok := proof.ModelAutoFixedHostCost(context.Background(), candidate)
		if !ok || !modelAutoTransfer(candidate, out, cost) {
			t.Fatalf("complete cost proof failed at %d layers", layers)
		}
		host, he := strconv.ParseUint(out.Domains[0].RequiredBytes, 10, 64)
		device, de := strconv.ParseUint(out.Domains[1].RequiredBytes, 10, 64)
		if he != nil || de != nil || host > previousHost || device < previousDevice {
			t.Fatalf("nonmonotonic TOTAL cost at %d layers", layers)
		}
		previousHost, previousDevice = host, device
	}
	p, err := resolveModelSettingsAutoInput(context.Background(), "exact", autoTestRequest(), in, modelAutoOperation{Memory: owner, LayerCosts: proof}, &state)
	if err != nil || p.Status != "fits" || p.Resolved.PerSlotContext != 4096 || p.Resolved.GPU.Layers == nil || *p.Resolved.GPU.Layers != 81 || calls > modelAutoBudget {
		t.Fatalf("floor-only search: proposal=%+v calls=%d err=%v", p, calls, err)
	}
}

func TestModelSettingsAutoBudgetAndInsufficientAreDistinct(t *testing.T) {
	test := func(name string, layers uint64, want string) {
		t.Run(name, func(t *testing.T) {
			now := time.Now()
			memory := autoTestMemory(now)
			memory.Domains[0].FreeBytes = "1"
			in := autoTestInput()
			in.Facts.Shards[0].Numeric["llama.block_count"] = gguffacts.Numeric{Type: 10, Bits: layers}
			r := autoTestRequest()
			r.KVOffload = nil // unproven: enumerate
			calls := 0
			state := modelEstimateState{now: func() time.Time { return now }, run: func(ctx context.Context, in modelEstimateInput) modelconfig.Estimate {
				calls++
				return autoTestEstimate(ctx, in)
			}}
			p, err := resolveModelSettingsAutoInput(context.Background(), "exact", r, in, modelAutoOperation{Memory: &autoTestMemoryOwner{memory: memory}}, &state)
			if err == nil || p.Status != want || p.Resolved != nil || calls > modelAutoBudget {
				t.Fatalf("status=%s calls=%d err=%v", p.Status, calls, err)
			}
			if want == "budget-exhausted" && !strings.Contains(err.Error(), "unsearched context") {
				t.Fatal("missing unsearched range", err)
			}
		})
	}
	test("unproven range exceeds budget", 80, "budget-exhausted")
	test("searched complete range cannot fit", 2, "insufficient-memory")
}

func TestModelSettingsAutoUnknownCannotFit(t *testing.T) {
	test := func(name string, mutate func(*modelEstimateMemory, *modelconfig.Estimate)) {
		t.Run(name, func(t *testing.T) {
			now := time.Now()
			memory := autoTestMemory(now)
			in := autoTestInput()
			in.Memory = memory
			out := autoTestEstimate(context.Background(), in)
			modelEstimateDomains(&out, in, now)
			mutate(memory, &out)
			known, fits, _ := modelAutoFit(out, memory, now, in.Allocation.GPU)
			if known || fits {
				t.Fatalf("unknown accepted: %+v", out)
			}
		})
	}
	test("stale snapshot", func(m *modelEstimateMemory, _ *modelconfig.Estimate) { m.At = m.At.Add(-6 * time.Second) })
	test("unknown cost", func(_ *modelEstimateMemory, e *modelconfig.Estimate) { e.Components[2].Status = "unknown" })
	test("missing workspace", func(_ *modelEstimateMemory, e *modelconfig.Estimate) {
		e.Components = slices.Delete(e.Components, 2, 3)
	})
	test("unmapped domain", func(_ *modelEstimateMemory, e *modelconfig.Estimate) { e.Domains[0].ID = "unmapped" })
	test("incomplete allocation", func(_ *modelEstimateMemory, e *modelconfig.Estimate) { e.Domains[0].RequiredBytes = "" })
	test("missing capacity", func(m *modelEstimateMemory, _ *modelconfig.Estimate) { m.Domains[0].CapacityBytes = "" })
	test("partial totals", func(_ *modelEstimateMemory, e *modelconfig.Estimate) { e.Status = "partial" })
}

func TestModelSettingsAutoUnknownMultiGPUAndCPUDisclosure(t *testing.T) {
	now := time.Now()
	memory := autoTestMemory(now)
	memory.Domains = append(memory.Domains, modelconfig.MemoryDomain{ID: "second", Kind: "device", DeviceID: "unmapped", CapacityBytes: "8589934592", FreeBytes: "8589934592"})
	r := autoTestRequest()
	r.Context = &modelconfig.Context{Mode: "fixed", Tokens: 4096}
	in := autoTestInput()
	state := modelEstimateState{now: func() time.Time { return now }, run: autoTestEstimate}
	p, err := resolveModelSettingsAutoInput(context.Background(), "exact", r, in, modelAutoOperation{Memory: &autoTestMemoryOwner{memory: memory}}, &state)
	if err != nil || p.Resolved == nil || p.Resolved.GPU.Mode != "cpu" || !slices.Contains(p.Warnings, "CPU-only placement is slower; no GPU capacity is credited.") {
		t.Fatalf("CPU fallback=%+v err=%v", p, err)
	}
	layers := int32(2)
	r.GPU = &modelconfig.GPU{Mode: "layers", Layers: &layers}
	r.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 4096}
	p, err = resolveModelSettingsAutoInput(context.Background(), "exact", r, in, modelAutoOperation{Memory: &autoTestMemoryOwner{memory: memory}}, &state)
	if err == nil || p.Status != "unavailable" || p.Resolved != nil {
		t.Fatalf("unmapped multi-GPU passed: %+v %v", p, err)
	}
}

func TestModelSettingsAutoUnsupportedPrecisionIsFieldError(t *testing.T) {
	now := time.Now()
	r := autoTestRequest()
	q8 := "q8_0"
	r.CacheTypeV = &q8
	state := modelEstimateState{now: func() time.Time { return now }, run: func(context.Context, modelEstimateInput) modelconfig.Estimate {
		t.Fatal("unsupported native control reached estimator")
		return modelconfig.Estimate{}
	}}
	p, err := resolveModelSettingsAutoInput(context.Background(), "exact", r, autoTestInput(), modelAutoOperation{Memory: &autoTestMemoryOwner{memory: autoTestMemory(now)}}, &state)
	if err == nil || !strings.Contains(err.Error(), "precision") || p.Resolved != nil {
		t.Fatalf("unsupported precision: %+v %v", p, err)
	}
}

func TestModelSettingsAutoRecalculateChangesReviewOnlyExplicitly(t *testing.T) {
	now := time.Now()
	owner := &autoTestMemoryOwner{memory: autoTestMemory(now)}
	state := modelEstimateState{now: func() time.Time { return now }, run: autoTestEstimate}
	r := autoTestRequest()
	in := autoTestInput()
	first, err := resolveModelSettingsAutoInput(context.Background(), "exact", r, in, modelAutoOperation{Memory: owner, ConversionSource: strings.Repeat("c", 64)}, &state)
	if err != nil {
		t.Fatal(err)
	}
	owner.memory.Identity = "new-snapshot"
	second, err := resolveModelSettingsAutoInput(context.Background(), "exact", r, in, modelAutoOperation{Memory: owner, ConversionSource: strings.Repeat("c", 64)}, &state)
	if err != nil {
		t.Fatal(err)
	}
	if first.ReviewedInputID == second.ReviewedInputID || !reflect.DeepEqual(first.Resolved.GPU, second.Resolved.GPU) {
		t.Fatal("new memory snapshot did not produce independent review identity")
	}
	conversion, _, ok := modelAutoReviewParts(first.ReviewedInputID)
	if !ok || conversion != strings.Repeat("c", 64) {
		t.Fatal("Auto review lost conversion source")
	}
}

func TestModelSettingsAutoReservePerIndependentPool(t *testing.T) {
	now := time.Now()
	in := autoTestInput()
	in.Memory = autoTestMemory(now)
	out := autoTestEstimate(context.Background(), in)
	modelEstimateDomains(&out, in, now)
	capacity := uint64(8589934592)
	reserve := capacity/10 + 1 + (512 << 20)
	required, err := strconv.ParseUint(out.Domains[0].RequiredBytes, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	in.Memory.Domains[0].FreeBytes = strconv.FormatUint(required+reserve, 10)
	known, fits, _ := modelAutoFit(out, in.Memory, now, in.Allocation.GPU)
	if !known || !fits {
		t.Fatal("exact reserve boundary should fit")
	}
	in.Memory.Domains[0].FreeBytes = strconv.FormatUint(required+reserve-1, 10)
	known, fits, _ = modelAutoFit(out, in.Memory, now, in.Allocation.GPU)
	if !known || fits {
		t.Fatal("one byte below reserve should fail")
	}
}

// Keep these fixtures on the real worker validation path, not just hash helpers.
func autoTestFrozenLaunch(t *testing.T) (modelLoadFixture, settings.Request, *autoTestMemoryOwner) {
	t.Helper()
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	// A replaceable binding, never an executable launched by this fixture.
	f.st.plat.Runtime.Bin = filepath.Join(f.root, "synthetic-auto-native")
	f.st.binPath = f.st.plat.Runtime.Bin
	writeTestFile(t, f.st.plat.Runtime.Bin, []byte("first synthetic Auto native build"))
	command, err := launchForState(f.st, f.st.plat.Runtime.Port)
	if err != nil {
		t.Fatal(err)
	}
	if command.Bin != f.st.plat.Runtime.Bin {
		t.Fatal("public preview does not use the replaceable synthetic native binding")
	}
	now := time.Now()
	owner := &autoTestMemoryOwner{memory: autoTestMemory(now)}
	f.ex.modelEstimates.now = func() time.Time { return now }
	f.ex.modelEstimates.run = autoTestEstimate
	r := autoTestRequest()
	r.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 8192}
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: r}}
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	if err := f.ex.validateModelSettingsLaunch(ctx, request, false); err != nil {
		t.Fatal("public preview validation rejected owner-injected Auto", err)
	}
	if owner.calls != 0 {
		t.Fatal("public validation resolved Auto before the compiler")
	}
	preview, err := f.ex.PreviewLaunch(request, ctx)
	if err != nil || len(preview.Errors) != 0 || preview.Auto == nil || preview.Auto.Status != "fits" || preview.Estimate == nil || preview.Estimate.MemorySource != "synthetic-owner" {
		t.Fatalf("Auto preview=%+v err=%v", preview, err)
	}
	if preview.Settings.Serving != nil || f.st.running || f.st.proc != nil {
		t.Fatal("preview created declaration or activated")
	}
	assertNoSettingsOverride(t, f.ex)
	return f, settings.Request{Engine: "fake", Settings: preview.Settings}, owner
}

func TestModelSettingsAutoFrozenApplyValidation(t *testing.T) {
	test := func(name string, mutate func(*settings.Request), refused bool) {
		t.Run(name, func(t *testing.T) {
			f, request, owner := autoTestFrozenLaunch(t)
			mutate(&request)
			owner.calls = 0
			owner.memory = nil
			ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
			err := f.ex.validateModelSettingsLaunch(ctx, request, true)
			if (err != nil) != refused {
				t.Fatalf("validation error=%v, want refusal=%v", err, refused)
			}
			if owner.calls != 0 {
				t.Fatalf("Apply called resolver memory owner %d times", owner.calls)
			}
		})
	}
	selected := func(p *settings.Request) *modelconfig.ModelLoad {
		for i := range p.Settings.ModelLoads {
			if p.Settings.ModelLoads[i].ModelID == "weak-model" {
				return &p.Settings.ModelLoads[i]
			}
		}
		panic("missing synthetic weak-model")
	}
	test("frozen Auto accepted without telemetry", func(*settings.Request) {}, false)
	test("conversion label cannot authorize Auto", func(p *settings.Request) { selected(p).Resolved.ResolverRevision = externalConversionResolver }, true)
	test("missing reviewed identity", func(p *settings.Request) { selected(p).Resolved.ReviewedInputID = "" }, true)
	test("tampered reviewed identity", func(p *settings.Request) {
		r := selected(p).Resolved
		r.ReviewedInputID = strings.Repeat("a", len(r.ReviewedInputID))
	}, true)
	test("stale requested preview", func(p *settings.Request) { selected(p).Requested.Context.Cap = 12288 }, true)
	test("stale patch preview refused at owner boundary", func(p *settings.Request) {
		p.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", PreviewDigest: strings.Repeat("b", 64), Load: &modelconfig.LoadPatch{Set: autoTestRequest()}}
	}, true)
}

func TestModelSettingsAutoCurrentEstimatorCannotInventFits(t *testing.T) {
	now := time.Now()
	r := autoTestRequest()
	r.Context = &modelconfig.Context{Mode: "auto", Floor: 4096, Cap: 4096}
	r.GPU = &modelconfig.GPU{Mode: "cpu"}
	state := modelEstimateState{now: func() time.Time { return now }}
	p, err := resolveModelSettingsAutoInput(context.Background(), "exact", r, autoTestInput(), modelAutoOperation{Memory: &autoTestMemoryOwner{memory: autoTestMemory(now)}}, &state)
	if err == nil || p.Status != "unavailable" || p.Resolved != nil {
		t.Fatalf("partial estimate invented Fits: %+v %v", p, err)
	}
}

func TestModelSettingsAutoGreatestFittingOffload(t *testing.T) {
	now := time.Now()
	in := autoTestInput()
	memory := autoTestMemory(now)
	layers := int32(2)
	in.Allocation.GPU = modelconfig.GPU{Mode: "layers", Layers: &layers}
	m, reason := modelEstimateAdapt(context.Background(), in.Facts)
	if reason != "" {
		t.Fatal(reason)
	}
	projection := modelconfig.Estimate{Components: []modelconfig.EstimateComponent{estimateComponent("weights", m.weights, "")}}
	modelEstimatePlacement(&projection, m, in)
	limit, err := strconv.ParseUint(projection.DeviceBytes, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	capacity := uint64(8589934592)
	memory.Domains[1].FreeBytes = strconv.FormatUint(capacity/10+1+(512<<20)+limit, 10)
	state := modelEstimateState{now: func() time.Time { return now }, run: autoTestEstimate}
	p, err := resolveModelSettingsAutoInput(context.Background(), "exact", autoTestRequest(), in, modelAutoOperation{Memory: &autoTestMemoryOwner{memory: memory}}, &state)
	if err != nil || p.Resolved == nil || p.Resolved.PerSlotContext != 8192 || p.Resolved.GPU.Layers == nil || *p.Resolved.GPU.Layers != 2 {
		t.Fatalf("greatest fitting offload: %+v %v", p, err)
	}
}

func TestModelSettingsAutoManualPlacementRemainsConstraint(t *testing.T) {
	test := func(name string, gpu modelconfig.GPU, want int32) {
		t.Run(name, func(t *testing.T) {
			now := time.Now()
			r := autoTestRequest()
			r.Context = &modelconfig.Context{Mode: "auto", Floor: 2048, Cap: 2048}
			r.GPU = &gpu
			state := modelEstimateState{now: func() time.Time { return now }, run: autoTestEstimate}
			p, err := resolveModelSettingsAutoInput(context.Background(), "exact", r, autoTestInput(), modelAutoOperation{Memory: &autoTestMemoryOwner{memory: autoTestMemory(now)}}, &state)
			if err != nil || p.Resolved == nil || p.Resolved.PerSlotContext != 2048 {
				t.Fatalf("manual placement: %+v %v", p, err)
			}
			if gpu.Mode == "cpu" {
				if p.Resolved.GPU.Mode != "cpu" {
					t.Fatal("CPU request was offloaded")
				}
			} else if p.Resolved.GPU.Layers == nil || *p.Resolved.GPU.Layers != want {
				t.Fatalf("placement changed: %+v", p.Resolved.GPU)
			}
		})
	}
	one := int32(1)
	test("explicit CPU", modelconfig.GPU{Mode: "cpu"}, 0)
	test("native default finite", modelconfig.GPU{Mode: "native-default"}, 3)
	test("explicit layer count", modelconfig.GPU{Mode: "layers", Layers: &one}, 1)
}

func TestModelSettingsAutoUMASingleReserve(t *testing.T) {
	now := time.Now()
	const required = uint64(1024)
	const capacity = uint64(8589934592)
	memory := &modelEstimateMemory{Identity: "UMA", Source: "synthetic-owner", At: now, UMA: true, Domains: []modelconfig.MemoryDomain{{ID: "shared", Kind: "shared", CapacityBytes: strconv.FormatUint(capacity, 10), FreeBytes: strconv.FormatUint(required+capacity/10+1+(512<<20), 10)}}}
	out := modelconfig.Estimate{Status: "estimated", Domains: []modelconfig.MemoryDomain{{ID: "shared", Kind: "shared", RequiredBytes: "1024"}}, Components: []modelconfig.EstimateComponent{
		{Name: "weights", DomainID: "shared", Status: "estimated", Bytes: "512"},
		{Name: "attention-cache", DomainID: "shared", Status: "estimated", Bytes: "256"},
		{Name: "compute-workspace", DomainID: "shared", Status: "estimated", Bytes: "128"},
		{Name: "prompt-cache", DomainID: "shared", Status: "estimated", Bytes: "128"},
		{Name: "recurrent-state", Status: "excluded"}, {Name: "companions", Status: "excluded"},
	}}
	known, fits, _ := modelAutoFit(out, memory, now, modelconfig.GPU{Mode: "cpu"})
	if !known || !fits {
		t.Fatal("UMA capacity was double-reserved or rejected")
	}
}

func TestModelSettingsAutoPreviewRecalculateIsExplicit(t *testing.T) {
	f, frozen, owner := autoTestFrozenLaunch(t)
	f.st.plat.Runtime.ModelLoads = modelconfig.CloneLoads(frozen.Settings.ModelLoads)
	owner.memory.Identity = "changed-snapshot"
	owner.calls = 0
	ctx := context.WithValue(context.Background(), modelAutoOperationKey{}, modelAutoOperation{Memory: owner})
	request := settingsRequest(t, f.ex)
	threads := int32(6)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: modelconfig.RequestedLoad{Threads: &threads}}}
	manual, err := f.ex.previewModelLoadChange(ctx, f.st, request)
	if err != nil || len(manual.Errors) == 0 || owner.calls != 0 {
		t.Fatalf("manual edit silently recalculated: errors=%v owner calls=%d err=%v", manual.Errors, owner.calls, err)
	}
	request.ModelChange.Load.Set = autoTestRequest()
	recalculated, err := f.ex.previewModelLoadChange(ctx, f.st, request)
	if err != nil || len(recalculated.Errors) != 0 || recalculated.Auto == nil || owner.calls != 1 {
		t.Fatalf("explicit Recalculate: %+v calls=%d err=%v", recalculated, owner.calls, err)
	}
	previous := rtLoadResolution(frozen.Settings.ModelLoads, "weak-model")
	if recalculated.Auto.ReviewedInputID == previous.ReviewedInputID || modelAutoConversionSource(recalculated.Auto.Resolved) != modelAutoConversionSource(previous) {
		t.Fatal("Recalculate did not bind fresh inputs and original conversion source")
	}
	if !modelconfig.EqualLoads(f.st.plat.Runtime.ModelLoads, frozen.Settings.ModelLoads) || f.st.running || f.st.proc != nil {
		t.Fatal("Recalculate changed accepted runtime")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsAutoPublicPreviewUsesOwnerCompiler(t *testing.T) {
	f, request, owner := autoTestFrozenLaunch(t) // real validation then PreviewLaunch
	if owner.calls != 1 {
		t.Fatalf("preview owner reads = %d, want one compiler snapshot", owner.calls)
	}
	resolved := rtLoadResolution(request.Settings.ModelLoads, "weak-model")
	if resolved == nil || resolved.ResolverRevision != modelAutoRevision || resolved.ReviewedInputID == "" {
		t.Fatal("public preview did not return reviewed Auto")
	}
	if f.st.running || f.st.proc != nil {
		t.Fatal("public preview activated an engine")
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsAutoPublicPreviewMissingOwnerRefusal(t *testing.T) {
	f := newModelLoadFixture(t)
	f.st.plat.Runtime = convertModelLoadFixture(t, f)
	request := settingsRequest(t, f.ex)
	request.ModelChange = &modelconfig.ModelChange{ModelID: "weak-model", ExpectedPolicyRevision: "0", Load: &modelconfig.LoadPatch{Set: autoTestRequest()}}
	err := f.ex.validateModelSettingsLaunch(context.Background(), request, false)
	if err == nil || err.Error() != "target memory telemetry unavailable" {
		t.Fatalf("public missing-owner refusal = %v", err)
	}
	preview, err := f.ex.PreviewLaunch(request)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, reason := range preview.Errors {
		if reason == "target memory telemetry unavailable" {
			found = true
		}
	}
	if !found || preview.Auto != nil && preview.Auto.Resolved != nil {
		t.Fatalf("compiler missing-owner result = %+v", preview)
	}
	assertNoSettingsOverride(t, f.ex)
}

func TestModelSettingsAutoLayerDependentWorkspaceKeepsCPU(t *testing.T) {
	test := func(name string, proof modelAutoLayerCostOwner) {
		t.Run(name, func(t *testing.T) {
			now := time.Now()
			in := autoTestInput()
			in.Facts.Shards[0].Numeric["llama.block_count"] = gguffacts.Numeric{Type: 10, Bits: 80}
			r := autoTestRequest()
			r.Context = &modelconfig.Context{Mode: "fixed", Tokens: 4096}
			calls := 0
			state := modelEstimateState{now: func() time.Time { return now }, run: func(ctx context.Context, in modelEstimateInput) modelconfig.Estimate {
				calls++
				out := autoTestEstimate(ctx, in)
				if in.Allocation.GPU.Mode != "cpu" {
					const extra = uint64(16 << 30)
					host, err := strconv.ParseUint(out.Domains[0].RequiredBytes, 10, 64)
					if err != nil {
						t.Fatal(err)
					}
					out.Domains[0].RequiredBytes = strconv.FormatUint(host+extra, 10)
					out.Components = append(out.Components, modelconfig.EstimateComponent{Name: "offloaded-workspace", Status: "estimated", DomainID: "host", Bytes: strconv.FormatUint(extra, 10)})
				}
				return out
			}}
			p, err := resolveModelSettingsAutoInput(context.Background(), "exact", r, in, modelAutoOperation{Memory: &autoTestMemoryOwner{memory: autoTestMemory(now)}, LayerCosts: proof}, &state)
			if err != nil || p.Status != "fits" || p.Resolved == nil || p.Resolved.GPU.Mode != "cpu" || calls > modelAutoBudget {
				t.Fatalf("layer-dependent workspace skipped CPU: status=%s calls=%d err=%v", p.Status, calls, err)
			}
		})
	}
	test("no total-cost proof enumerates", nil)
	test("estimate disagrees with proof falls back", autoTestLayerCosts{})
}
