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

func TestMemoryProfileRequestCanonicalDecoding(t *testing.T) {
	valid := `{"engine":"llamacpp","modelId":"model","configurationKey":"` + strings.Repeat("a", 64) + `","hostBytes":"18446744073709551615"}`
	if _, err := DecodeSetStatedMemoryRequest([]byte(valid)); err != nil {
		t.Fatal(err)
	}
	for _, suffix := range []string{`,"hostBytes":"1"`, `,"memory":{}`, `,"current":true`} {
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
