// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestMemoryPlanSourceClone(t *testing.T) {
	original := Estimate{Status: "partial", EstimateSource: &EstimateSource{Kind: "engine-report", Provider: "fixture", Revision: "1", Basis: "synthetic", UnpricedCosts: []string{"transient peak"}}}
	clone := original.Clone()
	clone.EstimateSource.UnpricedCosts[0] = "changed"
	if original.EstimateSource.UnpricedCosts[0] != "transient peak" {
		t.Fatal("clone shares source costs")
	}
	if err := original.Validate(); err != nil {
		t.Fatalf("validate source: %v", err)
	}
}

func memoryPlanFixture() MemoryPlan {
	config := MemoryPlanConfig{
		ContextTokens: 8192, KVFormat: "int8", BatchSize: testPointer(int32(1)),
		Microbatch:    &MemoryPlanMicrobatch{Mode: "fixed", Tokens: testPointer(int32(1024))},
		ParallelSlots: testPointer(int32(1)), Devices: []MemoryPlanSelector{{Backend: "cuda", Selector: "cuda:0"}}, Placement: "hybrid",
	}
	resource := func(kind, total string) MemoryPlanResource {
		r := MemoryPlanResource{
			Kind: kind, Mapping: "not-applicable", MappingBasis: "Synthetic capacity resource",
			PricedTotalBytes: total, PricedTotalBasis: "Synthetic once-counted total",
			RequiredWithAllowancesBytes: total, UncoveredCosts: []string{"runtime_driver_overhead", "transient_peak"},
			Assumptions:      []MemoryPlanAmount{{Name: "protected reserve", Bytes: "0", Basis: "Synthetic protected amount"}},
			CallerAllowances: []MemoryPlanAmount{{Name: "asserted-extra", Bytes: "0", Provenance: "caller-asserted"}},
		}
		for _, name := range memoryPlanGroupNames {
			r.Groups = append(r.Groups, MemoryPlanGroup{Name: name, Quality: "unknown", Basis: "Synthetic unpartitioned cost", Additive: name != "transient_peak"})
		}
		r.Groups[0].Quality, r.Groups[0].Bytes = "exact", "0"
		r.Groups[1].CombinedMembership = []string{"kv_cache", "compute_scratch"}
		r.CombinedAllocations = []MemoryPlanCombinedAllocation{{Members: []string{"kv_cache", "compute_scratch"}, Bytes: "7", Quality: "estimated", Basis: "Synthetic overlapping view"}}
		return r
	}
	device := resource("device", "9007199254740993")
	device.NativeSelector, device.Mapping = "cuda:0", "unproven"
	device.MappingBasis = "Native ordinal has no proven physical mapping"
	device.AllocatedBytes, device.MappedViewBytes = "0", "0"
	return MemoryPlan{
		Identity: MemoryPlanIdentity{
			Engine: "strata", Provider: "synthetic-provider", Revision: "fixture-revision",
			ExecutableSHA256: strings.Repeat("a", 64), RuntimeSourceSHA256: strings.Repeat("b", 64), ArtifactIdentitySHA256: strings.Repeat("c", 64),
			Build:   MemoryPlanBuild{Version: "fixture", Dirty: testPointer(false), Basis: "Synthetic embedded build summary"},
			ModelID: "test/model", AssertedContentSHA256: strings.Repeat("d", 64), ContentAssurance: "Sampled identity, caller content assertion unverified",
			Cacheable: false, CacheableReason: "Full model content is unverified",
		},
		Requested: config, Resolved: testPointer(config.Clone()), ConfigurationComplete: true,
		PricingScope: "resolved_configuration", PricingQualification: "non-replayable-pricing-inputs",
		Devices: []MemoryPlanResource{device}, HostRAM: resource("host_ram", "0"), HostCommit: resource("host_commit", "18446744073709551615"),
		Verdict:  MemoryPlanVerdict{Status: "not_evaluated", Conditional: true, Basis: "Synthetic provider verdict, no live observation"},
		Warnings: []string{"Synthetic kernel envelope is not a replayable launch configuration"},
	}
}

func memoryPlanEstimateFixture() Estimate {
	p := memoryPlanFixture()
	return Estimate{Status: "partial", EstimateSource: &EstimateSource{Kind: "engine-report", Provider: p.Identity.Provider, Revision: p.Identity.Revision, Basis: "Synthetic pricing", UnpricedCosts: []string{"runtime_driver_overhead", "transient_peak"}}, MemoryPlan: &p}
}

func TestMemoryPlanInspectRoundTripPreservesAccounting(t *testing.T) {
	original := InspectResult{Engine: "strata", ModelID: "test/model", Status: "unknown", Estimate: memoryPlanEstimateFixture()}
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("encode inspection: %v", err)
	}
	got, err := DecodeInspectResult(data)
	if err != nil {
		t.Fatalf("decode inspection: %v", err)
	}
	if !reflect.DeepEqual(got, original) {
		t.Fatalf("inspection changed accounting: got %+v want %+v", got, original)
	}
	p := got.Estimate.MemoryPlan
	if p.Devices[0].PricedTotalBytes != "9007199254740993" || p.HostCommit.PricedTotalBytes != "18446744073709551615" || p.Devices[0].Groups[0].Bytes != "0" || p.Devices[0].Groups[1].Bytes != "" || got.Estimate.Status != "partial" || p.Verdict.Status != "not_evaluated" {
		t.Fatalf("lost exact bytes, unknowns or provider status: %+v", p)
	}
}

func TestMemoryPlanSourceKindsSurviveAutoRoundTrip(t *testing.T) {
	for _, kind := range []string{"engine-report", "pair-estimate", "measured", "user-stated"} {
		t.Run(kind, func(t *testing.T) {
			e := memoryPlanEstimateFixture()
			if kind != "engine-report" {
				e.MemoryPlan = nil
			}
			e.EstimateSource.Kind = kind
			original := AutoProposal{Status: "unavailable", Estimate: e, Warnings: []string{"Synthetic advisory only"}}
			data, err := json.Marshal(original)
			if err != nil {
				t.Fatalf("encode proposal: %v", err)
			}
			var got AutoProposal
			if err := DecodeStrict(data, &got); err != nil {
				t.Fatalf("decode proposal: %v", err)
			}
			if err := got.Estimate.Validate(); err != nil {
				t.Fatalf("validate estimate: %v", err)
			}
			if !reflect.DeepEqual(got, original) {
				t.Fatalf("proposal changed source or accounting: got %+v want %+v", got, original)
			}
		})
	}
}

func TestMemoryPlanCloneDetachesNestedData(t *testing.T) {
	original := InspectResult{Estimate: memoryPlanEstimateFixture()}
	before := original.Clone()
	clone := original.Clone()
	p := clone.Estimate.MemoryPlan
	*p.Identity.Build.Dirty = true
	*p.Requested.BatchSize = 2
	*p.Requested.ParallelSlots = 2
	*p.Requested.Microbatch.Tokens = 512
	p.Requested.Devices[0].Selector = "changed"
	*p.Resolved.Microbatch.Tokens = 512
	p.Devices[0].Groups[1].CombinedMembership[0] = "changed"
	p.Devices[0].CombinedAllocations[0].Members[0] = "changed"
	p.Devices[0].Assumptions[0].Name = "changed"
	p.Devices[0].CallerAllowances[0].Name = "changed"
	p.Devices[0].UncoveredCosts[0] = "changed"
	p.HostRAM.Groups[0].Bytes = "9"
	p.HostCommit.CombinedAllocations[0].Members[0] = "changed"
	p.Warnings[0] = "changed"
	clone.Estimate.EstimateSource.UnpricedCosts[0] = "changed"
	if !reflect.DeepEqual(original, before) {
		t.Fatal("inspection clone shares nested plan data")
	}
	auto := AutoProposal{Estimate: original.Estimate}
	autoClone := auto.Clone()
	autoClone.Estimate.MemoryPlan.Devices[0].Groups[0].Name = "changed"
	if !reflect.DeepEqual(auto.Estimate, before.Estimate) {
		t.Fatal("Auto clone shares nested plan data")
	}
}

func TestMemoryPlanPAIRSourceSurvivesInspection(t *testing.T) {
	estimate := memoryPlanEstimateFixture()
	estimate.MemoryPlan = nil
	estimate.EstimateSource.Kind = "pair-estimate"
	original := InspectResult{Engine: "llamacpp", ModelID: "test/model", Status: "unknown", Estimate: estimate}
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("encode inspection: %v", err)
	}
	got, err := DecodeInspectResult(data)
	if err != nil {
		t.Fatalf("decode inspection: %v", err)
	}
	if !reflect.DeepEqual(got, original) {
		t.Fatalf("PAIR inspection changed provenance: got %+v want %+v", got, original)
	}
}

func TestMemoryPlanConfigurationPreservesEngineResolution(t *testing.T) {
	test := func(name string, mutate func(*MemoryPlan)) {
		t.Run(name, func(t *testing.T) {
			plan := memoryPlanFixture()
			mutate(&plan)
			if err := plan.Validate(); err != nil {
				t.Fatalf("reject engine resolution: %v", err)
			}
		})
	}
	test("minimum layout has no replay settings", func(p *MemoryPlan) {
		p.ConfigurationComplete, p.Resolved, p.PricingScope = false, nil, "minimum_supported_layout"
	})
	test("omitted requested defaults remain omitted", func(p *MemoryPlan) {
		p.Requested.KVFormat, p.Requested.BatchSize, p.Requested.ParallelSlots, p.Requested.Microbatch = "", nil, nil, nil
	})
	test("requested alias resolves canonically", func(p *MemoryPlan) { p.Requested.KVFormat = "q8_0" })
	test("automatic marker resolves within explicit limit", func(p *MemoryPlan) {
		p.Requested.Microbatch = &MemoryPlanMicrobatch{Mode: "auto", Limit: testPointer(int32(8192))}
	})
}

func TestMemoryPlanValidationRejectsContradictions(t *testing.T) {
	test := func(name string, mutate func(*MemoryPlan)) {
		t.Run(name, func(t *testing.T) {
			plan := memoryPlanFixture()
			mutate(&plan)
			if err := plan.Validate(); err == nil {
				t.Fatal("accepted contradictory memory plan")
			}
		})
	}
	test("duplicate device", func(p *MemoryPlan) {
		p.Devices = append(p.Devices, p.Devices[0])
		p.Requested.Devices = append(p.Requested.Devices, p.Requested.Devices[0])
	})
	test("duplicate group", func(p *MemoryPlan) { p.Devices[0].Groups[1].Name = "weights" })
	test("missing group", func(p *MemoryPlan) { p.HostRAM.Groups = p.HostRAM.Groups[:5] })
	test("unknown group", func(p *MemoryPlan) { p.Devices[0].Groups[0].Name = "native_component" })
	test("invalid quality", func(p *MemoryPlan) { p.Devices[0].Groups[0].Quality = "measured" })
	test("unknown with zero", func(p *MemoryPlan) { p.Devices[0].Groups[1].Bytes = "0" })
	test("known without bytes", func(p *MemoryPlan) { p.Devices[0].Groups[0].Bytes = "" })
	test("noncanonical bytes", func(p *MemoryPlan) { p.Devices[0].AllocatedBytes = "01" })
	test("negative bytes", func(p *MemoryPlan) { p.Devices[0].AllocatedBytes = "-1" })
	test("floating bytes", func(p *MemoryPlan) { p.Devices[0].AllocatedBytes = "1.5" })
	test("overflow bytes", func(p *MemoryPlan) { p.HostRAM.PricedTotalBytes = "18446744073709551616" })
	test("overflow allowances", func(p *MemoryPlan) {
		p.HostCommit.CallerAllowances = []MemoryPlanAmount{{Name: "extra", Bytes: "1", Provenance: "caller-asserted"}}
	})
	test("unknown membership", func(p *MemoryPlan) { p.Devices[0].Groups[1].CombinedMembership[0] = "native_component" })
	test("duplicate membership", func(p *MemoryPlan) { p.Devices[0].CombinedAllocations[0].Members = []string{"kv_cache", "kv_cache"} })
	test("additive transient peak", func(p *MemoryPlan) { p.Devices[0].Groups[5].Additive = true })
	test("inconsistent resolved scope", func(p *MemoryPlan) { p.PricingScope = "minimum_supported_layout" })
	test("incomplete with resolution", func(p *MemoryPlan) { p.ConfigurationComplete = false })
	test("replay approval", func(p *MemoryPlan) { p.PricingQualification = "replayable" })
	test("changed exact context", func(p *MemoryPlan) { p.Resolved.ContextTokens = 4096 })
	test("changed explicit KV", func(p *MemoryPlan) { p.Resolved.KVFormat = "fp16" })
	test("changed explicit batch", func(p *MemoryPlan) { *p.Resolved.BatchSize = 2 })
	test("changed explicit microbatch", func(p *MemoryPlan) { *p.Resolved.Microbatch.Tokens = 512 })
	test("unknown microbatch mode", func(p *MemoryPlan) { p.Requested.Microbatch.Mode = "native" })
	test("unproven physical identity", func(p *MemoryPlan) { p.Devices[0].PhysicalID = "GPU-fixture" })
	test("cache assertion promoted", func(p *MemoryPlan) { p.Identity.Cacheable = true })
	test("missing budget", func(p *MemoryPlan) {
		p.Verdict.Status = "indeterminate"
		p.Verdict.BudgetBasis = "before_runtime_initialization"
	})
	test("shortfall without budget", func(p *MemoryPlan) { p.Devices[0].ShortfallBytes = "0"; p.Devices[0].ShortfallBasis = "synthetic" })
	test("duplicate uncovered cost", func(p *MemoryPlan) { p.Devices[0].UncoveredCosts = []string{"peak", "peak"} })
	test("oversized warning", func(p *MemoryPlan) { p.Warnings = []string{strings.Repeat("x", 1025)} })
}

func memoryPlanWithBudget() MemoryPlan {
	p := memoryPlanFixture()
	p.Verdict.Status, p.Verdict.BudgetBasis = "indeterminate", "before_runtime_initialization"
	setBudget := func(r *MemoryPlanResource) {
		r.BudgetBytes, r.ShortfallBytes, r.ShortfallBasis = r.RequiredWithAllowancesBytes, "0", "Synthetic lower bound"
	}
	setBudget(&p.Devices[0])
	setBudget(&p.HostRAM)
	setBudget(&p.HostCommit)
	return p
}

func TestMemoryPlanBudgetUsesNonadditiveThreshold(t *testing.T) {
	p := memoryPlanWithBudget()
	r := &p.Devices[0]
	r.PricedTotalBytes = "100"
	r.MinimumSelectionCapacityBytes, r.MinimumSelectionCapacityBasis = "200", "Synthetic selection threshold"
	r.CallerAllowances = []MemoryPlanAmount{{Name: "runtime", Bytes: "10", Provenance: "caller-asserted"}}
	r.RequiredWithAllowancesBytes, r.BudgetBytes, r.ShortfallBytes = "210", "205", "5"
	p.Verdict.Status = "does_not_fit"
	if err := p.Validate(); err != nil {
		t.Fatalf("reject valid threshold shortfall: %v", err)
	}
	r.RequiredWithAllowancesBytes, r.ShortfallBytes = "310", "105"
	if err := p.Validate(); err == nil {
		t.Fatal("added selection threshold to priced allocations")
	}
}

func TestMemoryPlanBudgetVerdictsPreserveConditionalCosts(t *testing.T) {
	test := func(name string, mutate func(*MemoryPlan), wantError bool) {
		t.Run(name, func(t *testing.T) {
			p := memoryPlanWithBudget()
			mutate(&p)
			err := p.Validate()
			if (err != nil) != wantError {
				t.Fatalf("validate=%v wantError=%v", err, wantError)
			}
		})
	}
	test("indeterminate categories remain valid", func(*MemoryPlan) {}, false)
	test("explicit zero caller assertion", func(p *MemoryPlan) {
		p.Devices[0].CallerAllowances = []MemoryPlanAmount{{Name: "runtime", Bytes: "0", Provenance: "caller-asserted"}}
	}, false)
	test("shortfall must match arithmetic", func(p *MemoryPlan) { p.Devices[0].ShortfallBytes = "1" }, true)
	test("uncovered costs prevent conditional provider verdict", func(p *MemoryPlan) { p.Verdict.Status = "fits_within_allowances" }, true)
	test("negative provider verdict needs shortfall", func(p *MemoryPlan) { p.Verdict.Status = "does_not_fit" }, true)
	test("conditional fit retains unknown partitions", func(p *MemoryPlan) {
		p.Devices[0].UncoveredCosts, p.HostRAM.UncoveredCosts, p.HostCommit.UncoveredCosts = nil, nil, nil
		p.Verdict.Status = "fits_within_allowances"
	}, false)
	test("conditional fit cannot become unconditional", func(p *MemoryPlan) {
		p.Devices[0].UncoveredCosts, p.HostRAM.UncoveredCosts, p.HostCommit.UncoveredCosts = nil, nil, nil
		p.Verdict.Status, p.Verdict.Conditional = "fits_within_allowances", false
	}, true)
}

func TestMemoryPlanStrictWireRejectsPrivateAndMissingFields(t *testing.T) {
	original, err := json.Marshal(memoryPlanFixture())
	if err != nil {
		t.Fatalf("encode fixture: %v", err)
	}
	resolved, err := json.Marshal(memoryPlanFixture().Resolved)
	if err != nil {
		t.Fatalf("encode resolved fixture: %v", err)
	}
	test := func(name, old, replacement string) {
		t.Run(name, func(t *testing.T) {
			input := strings.Replace(string(original), old, replacement, 1)
			if input == string(original) {
				t.Fatal("fixture replacement did not match")
			}
			var plan MemoryPlan
			if err := DecodeStrict([]byte(input), &plan); err == nil {
				t.Fatal("accepted malformed or private plan field")
			}
		})
	}
	test("private extension", `"identity":`, `"extensions":{"strata":{"argv":["private"]}},"identity":`)
	test("private resource path", `"nativeSelector":"cuda:0"`, `"nativeSelector":"cuda:0","path":"/private/model"`)
	test("numeric bytes", `"pricedTotalBytes":"9007199254740993"`, `"pricedTotalBytes":9007199254740993`)
	test("null unknown bytes", `"quality":"unknown"`, `"bytes":null,"quality":"unknown"`)
	test("null resolved config", `"resolved":`+string(resolved), `"resolved":null`)
	test("missing configuration complete", `"configurationComplete":true,`, ``)
	test("missing cacheable", `"cacheable":false,`, ``)
	test("missing conditional", `"conditional":true,`, ``)
	test("missing additive", `,"additive":true`, ``)
	test("duplicate group key", `"name":"weights"`, `"name":"weights","name":"weights"`)
}

func TestMemoryPlanCompleteEnvelopeSizeCeiling(t *testing.T) {
	p := memoryPlanFixture()
	for i := 0; i < 80; i++ {
		p.Warnings = append(p.Warnings, fmt.Sprintf("%03d%s", i, strings.Repeat("x", 997)))
	}
	if err := p.Validate(); err == nil || !strings.Contains(err.Error(), "64 KiB") {
		t.Fatalf("oversized plan did not reach payload ceiling: %v", err)
	}
	inspection := InspectResult{Engine: "strata", ModelID: "test/model", Status: "unknown", Estimate: memoryPlanEstimateFixture()}
	inspection.Reason = strings.Repeat("x", MaxPayloadBytes)
	if err := inspection.Validate(); err == nil {
		t.Fatal("accepted oversized enclosing inspection")
	}
}

func TestMemoryPlanEstimateRejectsMixedAccountingAndIdentity(t *testing.T) {
	test := func(name string, mutate func(*Estimate)) {
		t.Run(name, func(t *testing.T) {
			e := memoryPlanEstimateFixture()
			mutate(&e)
			if err := e.Validate(); err == nil {
				t.Fatal("accepted conflicting estimate provenance or accounting")
			}
		})
	}
	test("missing source", func(e *Estimate) { e.EstimateSource = nil })
	test("wrong source", func(e *Estimate) { e.EstimateSource.Kind = "pair-estimate" })
	test("wrong provider", func(e *Estimate) { e.EstimateSource.Provider = "other" })
	test("wrong revision", func(e *Estimate) { e.EstimateSource.Revision = "other" })
	test("partial report relabeled complete", func(e *Estimate) { e.Status = "estimated" })
	test("priced report relabeled absent", func(e *Estimate) { e.Status = "unknown" })
	test("legacy additive components", func(e *Estimate) {
		e.Components = []EstimateComponent{{Name: "weights", Status: "estimated", Bytes: "0"}}
	})
	inspection := InspectResult{Engine: "strata", ModelID: "other-model", Status: "unknown", Estimate: memoryPlanEstimateFixture()}
	if err := inspection.Validate(); err == nil {
		t.Fatal("accepted mismatched inspection/model identity")
	}
}
