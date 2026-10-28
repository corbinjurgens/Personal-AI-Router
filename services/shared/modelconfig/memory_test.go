// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"math"
	"nvpair-shared/noderec"
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
			c.DomainID = "ram"
		}
		e.Components = append(e.Components, c)
	}
	return e
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
		{"missing category", func(e *Estimate) { e.Components = e.Components[:5] }},
		{"sum mismatch", func(e *Estimate) { e.Domains[0].RequiredBytes = "101" }},
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
	got := CompareEstimate(e)
	if got.CapacityFit != "unknown" || got.FreeFit != "unknown" {
		t.Fatal("overflow acquired a sized comparison verdict")
	}
}
