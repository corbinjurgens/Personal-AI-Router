// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMemoryProfileExactPools(t *testing.T) {
	p := MemoryProfile{ConfigurationKey: strings.Repeat("a", 64), Kind: "stated", RecordedAt: "2026-10-30T00:00:00Z", HostBytes: "1024"}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := p.ValidatePools([]string{"host"}); err != nil {
		t.Fatal(err)
	}
	t.Run("missing device", func(t *testing.T) {
		if p.ValidatePools([]string{"host", "device"}) == nil {
			t.Fatal("accepted missing device")
		}
	})
	t.Run("extra host", func(t *testing.T) {
		if p.ValidatePools([]string{"shared"}) == nil {
			t.Fatal("accepted extra host")
		}
	})
	t.Run("canonical decimal", func(t *testing.T) {
		p.HostBytes = "01"
		if p.Validate() == nil {
			t.Fatal("accepted leading zero")
		}
	})
}

func TestMemoryProfileMeasuredDetailRequired(t *testing.T) {
	p := MemoryProfile{Kind: "measured", ConfigurationKey: strings.Repeat("a", 64), RecordedAt: "2026-11-07T00:00:00Z"}
	if p.Validate() == nil {
		t.Fatal("accepted measured record without detail")
	}
}

func TestMemoryProfileUnpricedResultRetained(t *testing.T) {
	m := diagnosticFixture()
	m.ConfigurationKey = ""
	m.ExecutionKey = strings.Repeat("c", 64)
	m.ConfigurationKeyReason = "gpu-selection"
	p := MemoryProfile{Kind: "measured", ExecutionKey: m.ExecutionKey, NativeBuildID: m.NativeBuildID, RecordedAt: m.RecordedAt, Measured: &m}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	c := p.Clone()
	c.Measured.GPUs[1].Name = "changed"
	if p.Measured.GPUs[1].Name != "GPU B" {
		t.Fatal("profile clone aliases measured detail")
	}
	if p.OverlayEligible(m.DiagnosticIdentity, []string{"host", "device"}) {
		t.Fatal("unpriced record eligible")
	}
}

func TestMemoryProfileEarlierBuildLabel(t *testing.T) {
	m := diagnosticFixture()
	if got := m.BuildLabel("sha256:" + strings.Repeat("c", 64)); got != "measured on "+m.NativeBuildID {
		t.Fatalf("label = %q", got)
	}
	current := m.DiagnosticIdentity.Clone()
	current.NativeBuildID = "sha256:" + strings.Repeat("c", 64)
	if m.Matches(current) || m.AutoEligible(current, []string{"host", "device"}) {
		t.Fatal("earlier build counted as current")
	}
}

func TestMemoryProfileRequestCanonicalDecoding(t *testing.T) {
	valid := `{"engine":"llamacpp","modelId":"model","configurationKey":"` + strings.Repeat("a", 64) + `","hostBytes":"18446744073709551615"}`
	if _, err := DecodeSetStatedMemoryRequest([]byte(valid)); err != nil {
		t.Fatal(err)
	}
	for _, suffix := range []string{`,"hostBytes":"1"`, `,"memory":{}`, `,"current":true`, `,"nativeBuildId":"sha256:` + strings.Repeat("b", 64) + `"`} {
		t.Run(suffix, func(t *testing.T) {
			raw := strings.TrimSuffix(valid, "}") + suffix + "}"
			if _, err := DecodeSetStatedMemoryRequest([]byte(raw)); err == nil {
				t.Fatal("accepted duplicate or authority field")
			}
		})
	}
	for _, bytes := range []string{"", "01", "-1", "1.0", "18446744073709551616"} {
		t.Run(bytes, func(t *testing.T) {
			if _, err := DecodeSetStatedMemoryRequest([]byte(strings.Replace(valid, "18446744073709551615", bytes, 1))); err == nil {
				t.Fatal("accepted noncanonical bytes")
			}
		})
	}
}

func TestMemoryProfileBuildProvenance(t *testing.T) {
	p := MemoryProfile{ConfigurationKey: strings.Repeat("a", 64), Kind: "stated", RecordedAt: "2026-11-07T00:00:00Z", HostBytes: "1024", NativeBuildID: "sha256:" + strings.Repeat("b", 64)}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var decoded MemoryProfile
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded != p {
		t.Fatal("record lost build provenance on round trip")
	}
	for _, id := range []string{"local-build-path", "sha256:", "sha256:" + strings.Repeat("B", 64)} {
		t.Run(id, func(t *testing.T) {
			candidate := p
			candidate.NativeBuildID = id
			if candidate.Validate() == nil {
				t.Fatal("accepted non-content build identity")
			}
		})
	}
}

func TestMemoryProfileInspectionBoundsAndClone(t *testing.T) {
	p := MemoryProfile{ConfigurationKey: strings.Repeat("a", 64), Kind: "stated", RecordedAt: "2026-10-30T00:00:00Z", HostBytes: "1", Current: true}
	r := InspectResult{Engine: "llamacpp", ModelID: "model", Status: "unavailable", ConfigurationKey: p.ConfigurationKey, ConfigurationPools: []string{"host"}, Estimate: Estimate{Status: "unknown"}, MemoryProfiles: []MemoryProfile{p}}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeInspectResult(data)
	if err != nil {
		t.Fatal(err)
	}
	clone := decoded.Clone()
	clone.MemoryProfiles[0].HostBytes = "2"
	if decoded.MemoryProfiles[0].HostBytes != "1" {
		t.Fatal("clone aliases profiles")
	}
	r.MemoryProfiles = make([]MemoryProfile, 9)
	if r.Validate() == nil {
		t.Fatal("accepted oversized profile array")
	}
}
