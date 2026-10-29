// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"math"
	"nvpair-shared/noderec"
	"slices"
	"strconv"
	"testing"
	"time"
)

func completeMemoryArithmeticFixture() Estimate {
	e := Estimate{Status: "estimated", Domains: []MemoryDomain{{ID: "ram", Kind: "host", RequiredBytes: "100", CapacityBytes: "10737418240", FreeBytes: "10737418240", SampleID: "synthetic", CapacitySource: "fixture", CapacityTimestamp: "2026-10-28T00:00:00Z", FreeSource: "fixture", FreeTimestamp: "2026-10-28T00:00:00Z", RemainingValidityMS: 1000}}}
	for _, name := range MemoryComponentNames {
		c := EstimateComponent{Name: name, Status: "excluded", Reason: "synthetic absence"}
		if name == "weights" {
			c.Status = "estimated"
			c.Bytes = "100"
			c.LowBytes, c.HighBytes, c.Basis = "100", "100", "derived"
			c.DomainID = "ram"
		}
		e.Components = append(e.Components, c)
	}
	return e
}

func TestMemoryUncertaintyNeverImprovesVerdict(t *testing.T) {
	for _, available := range []string{"10737418240", "100"} {
		for _, status := range []string{"unknown", "partial"} {
			for _, domain := range []string{"ram", ""} {
				t.Run(available+"/"+status+"/"+domain, func(t *testing.T) {
					e := completeMemoryArithmeticFixture()
					e.Domains[0].FreeBytes = available
					want := CompareEstimate(e).FreeFit
					c := EstimateComponent{Name: "runtime-overhead", Status: status, DomainID: domain, Reason: "unpriced residual"}
					if status == "partial" {
						c.Bytes, c.LowBytes, c.HighBytes, c.Basis = "0", "0", "0", "heuristic"
					}
					e.Components = append(e.Components, c)
					got := CompareEstimate(e)
					warning := "not priced: runtime-overhead"
					if domain == "" {
						warning = "not priced (any pool): runtime-overhead"
					}
					if got.FreeFit != want || !slices.Contains(got.Domains[0].Unknowns, warning) {
						t.Fatalf("verdict/warnings = %+v, want %s with warning", got, want)
					}
				})
			}
		}
	}
}

func TestMemoryRangesAndSourceSlots(t *testing.T) {
	t.Run("missing weights cannot form summed requirement", func(t *testing.T) {
		e := completeMemoryArithmeticFixture()
		e.Components = e.Components[1:]
		if got := CompareEstimate(e); got.FreeFit != "unknown" {
			t.Fatalf("missing weights=%+v", got)
		}
	})
	t.Run("partial retains priced upper band", func(t *testing.T) {
		e := completeMemoryArithmeticFixture()
		e.Components[0].Status, e.Components[0].Reason, e.Components[0].HighBytes = "partial", "unpriced residual", "200"
		e.Domains[0].FreeBytes = strconv.FormatUint(199+MemoryReserve(10<<30), 10)
		got := CompareEstimate(e)
		if got.FreeFit != "tight" || !slices.Contains(got.Domains[0].Unknowns, "not priced: weights") {
			t.Fatalf("partial=%+v", got)
		}
	})
	t.Run("unknown weights", func(t *testing.T) {
		e := completeMemoryArithmeticFixture()
		e.Components[0] = EstimateComponent{Name: "weights", DomainID: "ram", Status: "unknown", Reason: "unavailable"}
		if got := CompareEstimate(e); got.FreeFit != "unknown" {
			t.Fatalf("unknown weights=%+v", got)
		}
	})
	for _, tc := range []struct{ name, available, want string }{
		{"upper reserve boundary", strconv.FormatUint(200+MemoryReserve(10<<30), 10), "fits"},
		{"below upper reserve", strconv.FormatUint(199+MemoryReserve(10<<30), 10), "tight"},
		{"expected boundary", "100", "tight"},
		{"upper alone exceeds", "150", "tight"},
		{"expected exceeds", "99", "exceeds"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := completeMemoryArithmeticFixture()
			e.Components[0].HighBytes = "200"
			e.Domains[0].FreeBytes = tc.available
			if got := CompareEstimate(e); got.FreeFit != tc.want {
				t.Fatalf("comparison=%+v want %s", got, tc.want)
			}
		})
	}
	for _, kind := range []string{"user-stated", "measured", "engine-report"} {
		t.Run(kind, func(t *testing.T) {
			e := completeMemoryArithmeticFixture()
			e.Components = nil
			e.EstimateSource = &EstimateSource{Kind: kind, Provider: "fixture", Revision: "1", Basis: "synthetic", UnpricedCosts: []string{"transient peaks"}}
			got := CompareEstimate(e)
			if got.FreeFit != "fits" || !slices.Contains(got.Domains[0].Unknowns, "not priced (any pool): transient peaks") {
				t.Fatalf("slot comparison=%+v", got)
			}
		})
	}
}
func TestMemoryCompleteSyntheticComparisons(t *testing.T) {
	for _, tc := range []struct{ name, available, want string }{{"sufficient", "10737418240", "fits"}, {"no reserve", "100", "tight"}, {"shortfall", "99", "exceeds"}, {"zero available", "0", "exceeds"}, {"missing available", "", "unknown"}} {
		t.Run(tc.name, func(t *testing.T) {
			e := completeMemoryArithmeticFixture()
			e.Domains[0].FreeBytes = tc.available
			got := CompareEstimate(e)
			if got.FreeFit != tc.want {
				t.Fatalf("freeFit=%s want %s: %+v", got.FreeFit, tc.want, got)
			}
			if got.CapacityFit != "fits" {
				t.Fatalf("capacity=%s", got.CapacityFit)
			}
		})
	}
}
func TestMemoryIncompleteSyntheticComparisons(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Estimate)
	}{
		{"partial workspace", func(e *Estimate) { e.Components[3].Status = "partial" }},
		{"expired", func(e *Estimate) { e.Domains[0].RemainingValidityMS = 0 }},
		{"unmapped", func(e *Estimate) { e.Components[0].DomainID = "" }},
		{"unmapped physical device", func(e *Estimate) { e.Domains[0].Kind = "device"; e.Domains[0].DeviceID = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := completeMemoryArithmeticFixture()
			tc.mutate(&e)
			got := CompareEstimate(e)
			if got.FreeFit != "unknown" || got.CapacityFit != "unknown" {
				t.Fatalf("comparison=%+v", got)
			}
		})
	}
}
func TestMemoryIndependentPoolShortfall(t *testing.T) {
	e := completeMemoryArithmeticFixture()
	d := e.Domains[0]
	d.ID = "gpu"
	d.Kind = "device"
	d.DeviceID = "uuid"
	d.RequiredBytes = "100"
	d.FreeBytes = "99"
	e.Domains = append(e.Domains, d)
	c := e.Components[0]
	c.DomainID = "gpu"
	e.Components = append(e.Components, c)
	got := CompareEstimate(e)
	if got.FreeFit != "exceeds" || got.Domains[0].FreeFit != "fits" {
		t.Fatalf("independent comparison=%+v", got)
	}
}
func TestMemoryWorkerHopAgesWithoutRenewal(t *testing.T) {
	fact := noderec.MemoryFact{Valid: true, Bytes: "100", Source: "fixture", SampledAt: time.Date(2026, 10, 28, 0, 0, 0, 0, time.UTC), AgeMS: 4998}
	m := &TargetMemory{HostID: "host", ObservationID: "sample", Observation: &noderec.MemoryObservation{RunID: "run", Domains: []noderec.MemoryDomain{{ID: "ram", Kind: "system", Total: fact, Free: fact}}}, Hop: &noderec.MemoryHop{ForwarderID: "broker", BootNS: "1000000"}}
	for _, tc := range []struct {
		name  string
		now   uint64
		valid bool
	}{{"fresh", 2000000, true}, {"expired", 4000000, false}, {"regressed", 0, false}} {
		t.Run(tc.name, func(t *testing.T) {
			got := m.At(func() (uint64, error) { return tc.now, nil })
			if got == nil || got.Domains[0].Total.Valid != tc.valid {
				t.Fatalf("observation=%+v", got)
			}
		})
	}
	request := WorkerInspectRequest{Request: InspectRequest{Engine: "llamacpp", ModelID: "model"}, Memory: m}
	data, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeWorkerInspectRequest(data); err != nil {
		t.Fatal(err)
	}
	if m.Observation.Domains[0].Total.AgeMS != 4998 {
		t.Fatal("input mutated")
	}
}

func TestMemoryRejectsUnsourcedExclusion(t *testing.T) {
	e := Estimate{Status: "estimated", Components: []EstimateComponent{{Name: "weights", Status: "excluded"}}}
	if err := e.Validate(); err == nil {
		t.Fatal("unsourced exclusion was accepted as a complete cost")
	}
}

func TestMemoryComparisonBoundaries(t *testing.T) {
	const capacity = uint64(10 << 30)
	reserve := capacity/10 + 512<<20
	for _, tc := range []struct {
		name                string
		required, available uint64
		want                string
	}{
		{"fits at reserve boundary", 100, 100 + reserve, "fits"},
		{"tight below reserve boundary", 100, 99 + reserve, "tight"},
		{"tight at required boundary", 100, 100, "tight"},
		{"exceeds available", 101, 100, "exceeds"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := CompareMemory(tc.required, capacity, tc.available); got != tc.want {
				t.Fatalf("comparison = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestMemoryComparisonRejectsInvalidFacts(t *testing.T) {
	if got := CompareMemory(1, 0, 1); got != "unknown" {
		t.Fatalf("comparison = %q, want unknown", got)
	}
}

func TestMemoryComparisonOverflowIsUnknown(t *testing.T) {
	if got := CompareMemory(math.MaxUint64, math.MaxUint64, math.MaxUint64); got != "unknown" {
		t.Fatalf("overflow comparison=%q, want unknown", got)
	}
	e := completeMemoryArithmeticFixture()
	e.Domains[0].RequiredBytes, e.Domains[0].CapacityBytes, e.Domains[0].FreeBytes = "18446744073709551615", "18446744073709551615", "18446744073709551615"
	e.Components[0].Bytes = "18446744073709551615"
	e.Components[0].LowBytes, e.Components[0].HighBytes = e.Components[0].Bytes, e.Components[0].Bytes
	got := CompareEstimate(e)
	if got.CapacityFit != "unknown" || got.FreeFit != "unknown" {
		t.Fatal("overflow acquired a sized comparison verdict")
	}
}
