// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package performancewire

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

func routingCostsFixture() RoutingCosts {
	return RoutingCosts{Revision: 1, Profiles: []RoutingProfile{{Host: "host", Engine: "llamacpp-custom", Model: "profile/key", Fingerprint: "fingerprint",
		Family: "llamacpp", APIPath: "/v1/chat/completions", Schema: "llamacpp-b12020-0ec4027a", AgeMs: 23,
		Decode: &RoutingRate{Samples: 2, TokensPerSecond: 79, Source: "engine"}}}}
}

func TestRoutingCostsValidation(t *testing.T) {
	tests := []struct {
		name string
		edit func(*RoutingCosts)
	}{
		{"zero revision", func(c *RoutingCosts) { c.Revision = 0 }},
		{"negative revision", func(c *RoutingCosts) { c.Revision = -1 }},
		{"missing profiles", func(c *RoutingCosts) { c.Profiles = nil }},
		{"too many profiles", func(c *RoutingCosts) { c.Profiles = make([]RoutingProfile, MaxEntries+1) }},
		{"duplicate tuple", func(c *RoutingCosts) {
			c.Profiles = append(c.Profiles, c.Profiles[0])
			c.Profiles[1].Fingerprint = "other"
		}},
		{"invalid host", func(c *RoutingCosts) { c.Profiles[0].Host = "bad host" }},
		{"oversized engine", func(c *RoutingCosts) { c.Profiles[0].Engine = strings.Repeat("e", MaxIDBytes+1) }},
		{"invalid model", func(c *RoutingCosts) { c.Profiles[0].Model = "model\n" }},
		{"missing fingerprint", func(c *RoutingCosts) { c.Profiles[0].Fingerprint = "" }},
		{"negative age", func(c *RoutingCosts) { c.Profiles[0].AgeMs = -1 }},
		{"unsupported family", func(c *RoutingCosts) { c.Profiles[0].Family = "other" }},
		{"unsupported API", func(c *RoutingCosts) { c.Profiles[0].APIPath = "/v1/completions" }},
		{"unsupported schema", func(c *RoutingCosts) { c.Profiles[0].Schema = "strata-f6726098" }},
		{"missing rates", func(c *RoutingCosts) { c.Profiles[0].Decode = nil }},
		{"zero rate", func(c *RoutingCosts) { c.Profiles[0].Decode.TokensPerSecond = 0 }},
		{"negative rate", func(c *RoutingCosts) { c.Profiles[0].Decode.TokensPerSecond = -1 }},
		{"nan rate", func(c *RoutingCosts) { c.Profiles[0].Decode.TokensPerSecond = math.NaN() }},
		{"infinite rate", func(c *RoutingCosts) { c.Profiles[0].Decode.TokensPerSecond = math.Inf(1) }},
		{"zero samples", func(c *RoutingCosts) { c.Profiles[0].Decode.Samples = 0 }},
		{"unknown source", func(c *RoutingCosts) { c.Profiles[0].Decode.Source = "estimate" }},
		{"load without duration", func(c *RoutingCosts) { c.Profiles[0].LoadSamples = 1 }},
		{"load without samples", func(c *RoutingCosts) { ms := int64(10); c.Profiles[0].LoadP95Ms = &ms }},
		{"too many extras", func(c *RoutingCosts) {
			c.Profiles[0].Extra = map[string]float64{}
			for i := 0; i < 9; i++ {
				c.Profiles[0].Extra[fmt.Sprint(i)] = 1
			}
		}},
		{"invalid extra", func(c *RoutingCosts) { c.Profiles[0].Extra = map[string]float64{"future": math.Inf(-1)} }},
		{"oversized payload", func(c *RoutingCosts) {
			for i := 1; i < MaxEntries; i++ {
				p := c.Profiles[0]
				p.Host = fmt.Sprint(i)
				p.Extra = map[string]float64{}
				for j := 0; j < 8; j++ {
					p.Extra[fmt.Sprint(j)+strings.Repeat("x", MaxLabelBytes-1)] = 1
				}
				p.Model = strings.Repeat("m", MaxModelBytes)
				p.Engine = strings.Repeat("e", MaxIDBytes)
				p.Fingerprint = strings.Repeat("f", MaxIDBytes)
				c.Profiles = append(c.Profiles, p)
			}
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := routingCostsFixture()
			tc.edit(&c)
			if err := c.Validate(); err == nil {
				t.Fatal("invalid routing costs accepted")
			}
		})
	}
}

func TestRoutingCostsStrictDecoder(t *testing.T) {
	c := routingCostsFixture()
	raw, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeRoutingCosts(raw)
	if err != nil || !reflect.DeepEqual(got, c) {
		t.Fatalf("roundtrip = %+v, %v", got, err)
	}
	for name, raw := range map[string]string{
		"unknown envelope": `{"revision":1,"profiles":[],"other":1}`,
		"trailing object":  string(raw) + ` {}`,
		"unknown profile":  strings.Replace(string(raw), `"ageMs":23`, `"ageMs":23,"other":1`, 1),
		"unknown rate":     strings.Replace(string(raw), `"source":"engine"`, `"source":"engine","other":1`, 1),
		"oversized bytes":  strings.Repeat(" ", MaxPayloadBytes+1),
		"null":             `null`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeRoutingCosts([]byte(raw)); err == nil {
				t.Fatal("invalid payload accepted")
			}
		})
	}
}

func TestRoutingCostsSupportedTuplesAndEmptyReplacement(t *testing.T) {
	c := RoutingCosts{Revision: 2, Profiles: []RoutingProfile{}}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	c = routingCostsFixture()
	p := c.Profiles[0]
	p.Family, p.Schema = "strata", "strata-f6726098"
	c.Profiles = append(c.Profiles, p)
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}
