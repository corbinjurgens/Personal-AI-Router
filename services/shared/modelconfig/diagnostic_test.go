// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"strings"
	"testing"
)

func diagnosticFixture() MeasuredProfile {
	return MeasuredProfile{
		DiagnosticIdentity: DiagnosticIdentity{
			ConfigurationKey: strings.Repeat("a", 64), NativeBuildID: "sha256:" + strings.Repeat("b", 64), BuildInfo: "b11146", CPUModel: "CPU", TotalRAMBytes: "68719476736",
			GPUs:      []DiagnosticGPU{{UUID: "gpu-a", Name: "GPU A", TotalMemoryBytes: "17179869184"}, {UUID: "gpu-b", Name: "GPU B", TotalMemoryBytes: "17179869184"}},
			Artifacts: []ArtifactSummary{{Role: "model", Identity: "artifact", Size: "100", Assurance: "sampled"}}, PerSlotContext: 1024, Slots: 1,
			Placement: []DiagnosticPlacement{{DomainID: "host", Kind: "host"}, {DomainID: "device-a", Kind: "device", DeviceID: "gpu-a"}, {DomainID: "device-b", Kind: "device", DeviceID: "gpu-b"}},
		}, RunID: "run", RecordedAt: "2026-11-07T00:00:00Z", Outcome: "failed", Reason: "cancelled before sampling",
	}
}

func TestDiagnosticTwoGPUIdentity(t *testing.T) {
	p := diagnosticFixture()
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	t.Run("GPU enumeration order preserves currentness", func(t *testing.T) {
		current := p.DiagnosticIdentity.Clone()
		current.GPUs[0], current.GPUs[1] = current.GPUs[1], current.GPUs[0]
		if !p.Matches(current) {
			t.Fatal("enumeration order changed hardware identity")
		}
	})
	t.Run("changed second GPU invalidates currentness", func(t *testing.T) {
		current := p.DiagnosticIdentity.Clone()
		current.GPUs[1].TotalMemoryBytes = "8589934592"
		if p.Matches(current) {
			t.Fatal("accepted changed hardware")
		}
	})
	t.Run("unknown placement device rejected", func(t *testing.T) {
		p := p.Clone()
		p.Placement[2].DeviceID = "missing"
		if p.Validate() == nil {
			t.Fatal("accepted unmapped device")
		}
	})
	t.Run("unpriced selection stays diagnostic only", func(t *testing.T) {
		p := p.Clone()
		p.ConfigurationKey = ""
		p.ExecutionKey = strings.Repeat("c", 64)
		p.ConfigurationKeyReason = "gpu-selection"
		if err := p.Validate(); err != nil {
			t.Fatal(err)
		}
		if p.OverlayEligible(p.DiagnosticIdentity, []string{"host", "device"}) || p.AutoEligible(p.DiagnosticIdentity, []string{"host", "device"}) {
			t.Fatal("unpriced record became eligible")
		}
	})
}

func TestDiagnosticOverlayRequiresCompletePoolAndContextEvidence(t *testing.T) {
	p := diagnosticFixture()
	p.Outcome = "completed"
	for _, placement := range p.Placement {
		p.Pools = append(p.Pools, MeasuredPool{
			DomainID: placement.DomainID, Kind: placement.Kind, DeviceID: placement.DeviceID,
			BaselineAvailableBytes: "1000", AfterLoadAvailableBytes: "900", LowMinAvailableBytes: "800", HighMinAvailableBytes: "700", PeakUsedBytes: "300", Source: "fixture", Samples: 6, IntervalMS: 2000,
		})
	}
	prompt, cached, generated := int32(880), int32(0), int32(128)
	first, end, overlap := 10.0, 100.0, true
	p.High = &MeasuredPhase{Coverage: "complete", Overlapped: &overlap, OverlapWindowSampled: &overlap, WallMS: 100,
		Requests: []MeasuredRequest{{TargetPromptTokens: prompt, PromptTokens: &prompt, CachedTokens: &cached, GeneratedTokens: &generated, FirstTokenMS: &first, EndMS: &end}}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if !p.OverlayEligible(p.DiagnosticIdentity, []string{"host", "device"}) {
		t.Fatal("complete current evidence was refused")
	}
	test := func(name string, change func(*MeasuredProfile)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			candidate := p.Clone()
			change(&candidate)
			if candidate.OverlayEligible(p.DiagnosticIdentity, []string{"host", "device"}) {
				t.Fatal("incomplete evidence became an overlay")
			}
		})
	}
	test("missing device pool", func(p *MeasuredProfile) { p.Pools = p.Pools[:2] })
	test("missing phase sample", func(p *MeasuredProfile) { p.Pools[0].LowMinAvailableBytes = "" })
	test("partial high context", func(p *MeasuredProfile) { p.High.Coverage = "partial" })
	test("unsampled overlap", func(p *MeasuredProfile) { *p.High.OverlapWindowSampled = false })
	test("changed build", func(p *MeasuredProfile) { p.NativeBuildID = "sha256:" + strings.Repeat("d", 64) })
}

func TestDiagnosticRequestConsent(t *testing.T) {
	r := DiagnosticStartRequest{DiagnosticPreviewRequest: DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "model", ProfileKey: "profile"}, PreviewID: "preview"}
	if r.Validate() == nil {
		t.Fatal("accepted missing consent")
	}
	r.Consent = true
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestDiagnosticPreviewAllowsUnobservedAllocationBeforeConsent(t *testing.T) {
	identity := diagnosticFixture().DiagnosticIdentity
	identity.ConfigurationKey = ""
	identity.ExecutionKey = strings.Repeat("c", 64)
	identity.ConfigurationKeyReason = "native-fit"
	identity.BuildInfo = ""
	identity.PerSlotContext = 0
	identity.Slots = 0
	identity.Placement = nil
	p := DiagnosticPreview{DiagnosticPreviewRequest: DiagnosticPreviewRequest{Engine: "llamacpp", ModelID: "model", ProfileKey: "profile"}, DiagnosticIdentity: identity, PreviewID: "preview", GenerateTokens: 128, TimeLimitSeconds: 1200, ExpiresAt: "2026-11-07T01:00:00Z"}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if identity.Validate() == nil {
		t.Fatal("unobserved allocation accepted as realized evidence")
	}
}

func TestDiagnosticMeasuredValidation(t *testing.T) {
	test := func(name string, change func(*MeasuredProfile)) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			p := diagnosticFixture()
			change(&p)
			if p.Validate() == nil {
				t.Fatal("accepted invalid result")
			}
		})
	}
	test("unknown outcome", func(p *MeasuredProfile) { p.Outcome = "ok" })
	test("unknown failed phase", func(p *MeasuredProfile) { p.FailedPhase = "other" })
	test("oversized notes", func(p *MeasuredProfile) { p.Notes = make([]string, 65) })
	test("oversized reason", func(p *MeasuredProfile) { p.Reason = strings.Repeat("x", 257) })
	test("noncanonical RAM", func(p *MeasuredProfile) { p.TotalRAMBytes = "01" })
	test("noncanonical pool", func(p *MeasuredProfile) {
		p.Pools = []MeasuredPool{{DomainID: "host", Kind: "host", Source: "test", IntervalMS: 2000, BaselineAvailableBytes: "01"}}
	})
}

func TestDiagnosticCloneDetachesEvidence(t *testing.T) {
	p := diagnosticFixture()
	n := int32(1)
	f := 1.0
	b := true
	p.Requested.Slots = &n
	p.Requested.ExtraOptions = map[string]string{"device": "gpu-a"}
	p.Notes = []string{"note"}
	p.Pools = []MeasuredPool{{DomainID: "host", Kind: "host", Source: "test", IntervalMS: 2000}}
	p.High = &MeasuredPhase{Coverage: "partial", Requests: []MeasuredRequest{{TargetPromptTokens: 10, PromptTokens: &n, CachedTokens: &n, GeneratedTokens: &n, PromptMS: &f, PromptPerSecond: &f, DecodeMS: &f, DecodePerSecond: &f, FirstTokenMS: &f, EndMS: &f}}, Overlapped: &b, OverlapWindowSampled: &b, CombinedGeneratedPerSecond: &f}
	p.Low = p.High
	c := p.Clone()
	c.GPUs[0].Name = "changed"
	c.Artifacts[0].Identity = "changed"
	c.Placement[0].DomainID = "changed"
	*c.Requested.Slots = 2
	c.Requested.ExtraOptions["device"] = "changed"
	c.Notes[0] = "changed"
	c.Pools[0].Source = "changed"
	*c.High.Requests[0].PromptTokens = 2
	*c.High.Requests[0].CachedTokens = 2
	*c.High.Requests[0].GeneratedTokens = 2
	*c.High.Requests[0].PromptMS = 2
	*c.High.Requests[0].PromptPerSecond = 2
	*c.High.Requests[0].DecodeMS = 2
	*c.High.Requests[0].DecodePerSecond = 2
	*c.High.Requests[0].FirstTokenMS = 2
	*c.High.Requests[0].EndMS = 2
	*c.High.Overlapped = false
	*c.High.OverlapWindowSampled = false
	*c.High.CombinedGeneratedPerSecond = 2
	if p.GPUs[0].Name != "GPU A" || p.Artifacts[0].Identity != "artifact" || p.Placement[0].DomainID != "host" || *p.Requested.Slots != 1 || p.Requested.ExtraOptions["device"] != "gpu-a" || p.Notes[0] != "note" || p.Pools[0].Source != "test" || *p.High.Requests[0].PromptTokens != 1 || *p.High.Requests[0].DecodeMS != 1 || !*p.High.Overlapped || *p.High.CombinedGeneratedPerSecond != 1 {
		t.Fatal("clone aliases source evidence")
	}
	if *c.Low.Requests[0].PromptTokens != 1 {
		t.Fatal("cloned phases alias each other")
	}
}

func TestDiagnosticUnknownRateRoundTrip(t *testing.T) {
	p := MeasuredRequest{TargetPromptTokens: 100}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var got MeasuredRequest
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.DecodePerSecond != nil || got.PromptPerSecond != nil {
		t.Fatal("unknown rate became zero")
	}
}

func TestDiagnosticStrictRoundTripFlattensIdentity(t *testing.T) {
	p := diagnosticFixture()
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var got MeasuredProfile
	if err := DecodeStrict(data, &got); err != nil {
		t.Fatal(err)
	}
	if !got.Matches(p.DiagnosticIdentity) || got.RunID != p.RunID {
		t.Fatal("flattened identity did not round trip")
	}
	for _, change := range []string{`,"cpuModel":"other"`, `,"unrecognized":true`} {
		raw := strings.TrimSuffix(string(data), "}") + change + "}"
		if err := DecodeStrict([]byte(raw), &got); err == nil {
			t.Fatal("strict decode accepted duplicate or unknown embedded field")
		}
	}
}
