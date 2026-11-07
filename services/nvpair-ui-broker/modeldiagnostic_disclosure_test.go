// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

func TestModelDiagnosticBaselineRetainsObservedDisturbance(t *testing.T) {
	use := func(value uint32) *uint32 { return &value }
	test := func(name string, utilization []*uint32, wantKnown, wantUnknown bool, low, high uint32) {
		t.Run(name, func(t *testing.T) {
			at := time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC)
			var samples diagnosticSamples
			for i, free := range []uint64{9000, 4000, 8000} {
				now := at.Add(time.Duration(i) * time.Second)
				memory := diagnosticMemoryFixture(now, free)
				gpu := memory.Domains[0]
				gpu.ID, gpu.Kind, gpu.DeviceIDs = "gpu-unused", "dedicated", []string{"gpu-unused"}
				gpu.Free.Bytes = "8000"
				memory.Domains = append(memory.Domains, gpu)
				if err := samples.observe(memory, now, utilization[i]); err != nil {
					t.Fatal(err)
				}
			}
			// A busy post-load sample cannot become baseline disclosure.
			memory := diagnosticMemoryFixture(at.Add(4*time.Second), 1000)
			gpu := memory.Domains[0]
			gpu.ID, gpu.Kind, gpu.DeviceIDs = "gpu-unused", "dedicated", []string{"gpu-unused"}
			memory.Domains = append(memory.Domains, gpu)
			if err := samples.observe(memory, at.Add(4*time.Second), use(99)); err != nil {
				t.Fatal(err)
			}
			identity := diagnosticIdentityFixture()
			identity.Placement = identity.Placement[:1]
			status := diagnosticManagerStatus{Run: &modelconfig.DiagnosticRun{RunID: "baseline", Outcome: "failed"}, Markers: map[string]time.Time{"baseline-ready": at, "loaded": at.Add(3 * time.Second)}}
			profile, err := samples.measured(identity, status, at.Add(5*time.Second))
			if err != nil {
				t.Fatal(err)
			}
			if len(profile.Pools) != 1 || profile.Pools[0].BaselineAvailableBytes != "8000" {
				t.Fatalf("cold median changed: %+v", profile.Pools)
			}
			for _, want := range []string{
				"Cold baseline: RAM use above half capacity (pools 1; used bytes 6000-6000; capacity bytes 10000-10000); the diagnostic may run out of memory.",
				"Cold baseline: GPU memory use above 5 percent (pools 1; used bytes 2000-2000; capacity bytes 10000-10000); activity included; source unidentified.",
			} {
				if !slices.Contains(profile.Notes, want) {
					t.Fatalf("baseline evidence missing %q: %v", want, profile.Notes)
				}
			}
			known := false
			for _, note := range profile.Notes {
				if strings.HasPrefix(note, "Cold baseline GPU utilization (node maximum):") {
					known = true
					want := fmt.Sprintf("Cold baseline GPU utilization (node maximum): %d-%d percent. Activity is included; its source is not identified.", low, high)
					if note != want {
						t.Fatalf("baseline utilization=%q, want %q", note, want)
					}
				}
				if strings.Contains(note, "9000 of 10000") {
					t.Fatalf("post-load disturbance leaked into baseline: %q", note)
				}
			}
			unknown := slices.Contains(profile.Notes, "Cold baseline GPU utilization is unknown for some or all samples; missing or stale telemetry is not zero.")
			if known != wantKnown || unknown != wantUnknown {
				t.Fatalf("known=%v unknown=%v notes=%v", known, unknown, profile.Notes)
			}
		})
	}
	test("nonzero utilization", []*uint32{use(7), use(42), use(11)}, true, false, 7, 42)
	test("unknown utilization", []*uint32{nil, nil, nil}, false, true, 0, 0)
	test("observed zero utilization", []*uint32{use(0), use(0), use(0)}, true, false, 0, 0)
	test("partly unknown utilization", []*uint32{nil, use(42), nil}, true, true, 42, 42)
}

func TestModelDiagnosticBaselineDisclosureFitsRecordBounds(t *testing.T) {
	test := func(name string, gpuPools, idLength int) {
		t.Run(name, func(t *testing.T) {
			at := time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC)
			var samples diagnosticSamples
			identity := diagnosticIdentityFixture()
			identity.Placement = identity.Placement[:1]
			identity.GPUs = nil
			for i := range 3 {
				now := at.Add(time.Duration(i) * time.Second)
				memory := diagnosticMemoryFixture(now, 4000)
				for j := range gpuPools {
					gpu := memory.Domains[0]
					gpu.ID = fmt.Sprintf("gpu-%02d-%s", j, strings.Repeat("x", idLength))
					gpu.Kind, gpu.DeviceIDs, gpu.Free.Bytes = "dedicated", []string{gpu.ID}, "8000"
					memory.Domains = append(memory.Domains, gpu)
					if i == 0 {
						identity.Placement = append(identity.Placement, modelconfig.DiagnosticPlacement{DomainID: gpu.ID, Kind: "device", DeviceID: gpu.ID})
						identity.GPUs = append(identity.GPUs, modelconfig.DiagnosticGPU{UUID: gpu.ID, Name: "synthetic GPU", TotalMemoryBytes: "10000"})
					}
				}
				if err := samples.observe(memory, now, nil); err != nil {
					t.Fatal(err)
				}
			}
			status := diagnosticManagerStatus{Run: &modelconfig.DiagnosticRun{RunID: "bounded-disclosure", Outcome: "failed"}, Markers: map[string]time.Time{"baseline-ready": at, "loaded": at.Add(3 * time.Second)}}
			profile, err := samples.measured(identity, status, at.Add(5*time.Second))
			if err != nil {
				t.Fatalf("valid baseline evidence prevented recording: %v", err)
			}
			gpuEvidence, ramEvidence, unknownUtilization := false, false, false
			for _, note := range profile.Notes {
				if strings.Contains(note, "GPU memory use") && !strings.Contains(note, fmt.Sprintf("pools %d;", gpuPools)) {
					t.Fatalf("baseline threshold count lost: %q", note)
				}
				gpuEvidence = gpuEvidence || strings.Contains(note, "GPU memory use") && strings.Contains(note, "5 percent") && strings.Contains(note, "2000") && strings.Contains(note, "10000")
				ramEvidence = ramEvidence || strings.Contains(note, "RAM use") && strings.Contains(note, "half") && strings.Contains(note, "6000") && strings.Contains(note, "10000")
				unknownUtilization = unknownUtilization || strings.Contains(note, "utilization is unknown")
			}
			if !gpuEvidence || !ramEvidence || !unknownUtilization {
				t.Fatalf("bounded notes lost baseline evidence: %v", profile.Notes)
			}
		})
	}
	test("pool count at wire limit", 63, 0)
	test("long physical pool name", 1, 200)
}

func TestModelDiagnosticIncompleteLongPoolRecordReleasesAfterStop(t *testing.T) {
	b, manager, op := diagnosticBrokerFixture(t)
	identity := op.review.preview.DiagnosticIdentity.Clone()
	identity.Placement = []modelconfig.DiagnosticPlacement{{DomainID: strings.Repeat("x", 200), Kind: "host"}}
	op.review.preview.DiagnosticIdentity = identity.Clone()
	at := time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC)
	for i := range 3 {
		now := at.Add(time.Duration(i) * time.Second)
		memory := diagnosticMemoryFixture(now, 4000)
		memory.Domains[0].ID = identity.Placement[0].DomainID
		if err := op.samples.observe(memory, now, nil); err != nil {
			t.Fatal(err)
		}
	}
	run := op.run.Clone()
	run.Phase, run.Outcome, run.FailedPhase, run.Reason = "done", "failed", "loading", "incomplete allocation evidence"
	setDiagnosticContractAnswer(t, manager, op, func(method string, _ json.RawMessage) (any, string) {
		if method == "engine:diagnostic-run-status" {
			return diagnosticStatusReply{Run: run, State: "cleanup-complete", Token: op.token, WorkStopped: true, Identity: &identity,
				Markers: map[string]time.Time{"baseline-ready": at, "loaded": at.Add(3 * time.Second)}}, ""
		}
		return nil, "unexpected method"
	})
	if err := b.cleanupDiagnostic(context.Background(), op); err != nil {
		t.Fatalf("stopped incomplete run retained resources: %v", err)
	}
	if !op.recorded || !op.reservationReleased || op.run.Result == nil || op.run.Result.Outcome != "failed" || len(op.run.Result.Pools) != 1 || op.run.Result.Pools[0].DomainID != identity.Placement[0].DomainID {
		t.Fatalf("failure evidence or release missing: %+v", op.run.Result)
	}
	requireDiagnosticReleased(t, b)
}

func TestModelDiagnosticUtilizationRequiresFreshMatchingScannerTelemetry(t *testing.T) {
	test := func(name string, mutate func(*noderec.NodeTelemetry), elapsed time.Duration, want *uint32) {
		t.Run(name, func(t *testing.T) {
			b, _ := newPolicyBroker(t)
			at := time.Now().UTC()
			memory := diagnosticMemoryFixture(at, 9000)
			seedDiagnosticTelemetry(t, b, at, memory)
			b.telemetry.mu.Lock()
			value := &b.telemetry.nodes[b.nodeID].scanner.value
			value.TelemetryValid, value.GPUUtilizationPct = true, 37
			mutate(value)
			b.telemetry.mu.Unlock()
			got := b.diagnosticGPUUtilization(memory, at.Add(elapsed))
			if want == nil {
				if got != nil {
					t.Fatalf("unknown utilization became %d", *got)
				}
			} else if got == nil || *got != *want {
				t.Fatalf("utilization=%v, want %d", got, *want)
			}
		})
	}
	value, zero := uint32(37), uint32(0)
	test("fresh nonzero", func(*noderec.NodeTelemetry) {}, 0, &value)
	test("fresh zero", func(v *noderec.NodeTelemetry) { v.GPUUtilizationPct = 0 }, 0, &zero)
	test("invalid is unknown", func(v *noderec.NodeTelemetry) { v.TelemetryValid = false }, 0, nil)
	test("stale age is unknown", func(v *noderec.NodeTelemetry) { v.MSSince = noderec.MemoryFreshMS + 1 }, 0, nil)
	test("aged receipt is unknown", func(*noderec.NodeTelemetry) {}, 6*time.Second, nil)
	test("negative age is unknown", func(v *noderec.NodeTelemetry) { v.MSSince = -1 }, 0, nil)
	test("out of range is unknown", func(v *noderec.NodeTelemetry) { v.GPUUtilizationPct = 101 }, 0, nil)
	test("future receipt is unknown", func(*noderec.NodeTelemetry) {}, -time.Second, nil)
	test("different memory producer is unknown", func(v *noderec.NodeTelemetry) { v.Memory.RunID = "other" }, 0, nil)
}

func TestModelDiagnosticPreviewDisclosesUtilizationWithoutApplicationAttribution(t *testing.T) {
	for _, valid := range []bool{true, false} {
		name := "unknown utilization"
		if valid {
			name = "observed utilization"
		}
		t.Run(name, func(t *testing.T) {
			b, _, request, _ := diagnosticPreviewFixture(t)
			b.telemetry.mu.Lock()
			value := &b.telemetry.nodes[b.nodeID].scanner.value
			value.TelemetryValid, value.GPUUtilizationPct = valid, 37
			b.telemetry.mu.Unlock()
			preview, err := b.previewDiagnostic(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			want := "Preview GPU utilization is unknown for some or all samples; missing or stale telemetry is not zero."
			if valid {
				want = "Preview GPU utilization (node maximum): 37-37 percent. Activity is included; its source is not identified."
			}
			if !slices.Contains(preview.Warnings, want) {
				t.Fatalf("preview disclosure missing %q: %v", want, preview.Warnings)
			}
		})
	}
}
