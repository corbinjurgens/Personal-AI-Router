// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package performancewire

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
)

const MethodRoutingCosts = "node/set-routing-costs"

// RoutingCosts is the broker's full replacement of the speed evidence a
// proxy may use for tier scoring. It is never a delta.
type RoutingCosts struct {
	Revision int64            `json:"revision"`
	Profiles []RoutingProfile `json:"profiles"`
}

// RoutingProfile is one profile's newest compatible current evidence. Model
// is the normalized dispatched profile key, not the native artifact name.
type RoutingProfile struct {
	Host        string             `json:"host"`
	Engine      string             `json:"engine"`
	Model       string             `json:"model"`
	Fingerprint string             `json:"fingerprint"`
	APIPath     string             `json:"apiPath"`
	Schema      string             `json:"schema"`
	Family      string             `json:"family"`
	AgeMs       int64              `json:"ageMs"`
	Prefill     *RoutingRate       `json:"prefill,omitempty"`
	Decode      *RoutingRate       `json:"decode,omitempty"`
	LoadP95Ms   *int64             `json:"loadP95Ms,omitempty"`
	LoadSamples int                `json:"loadSamples"`
	Extra       map[string]float64 `json:"extra,omitempty"`
}

type RoutingRate struct {
	TokensPerSecond float64 `json:"tokensPerSecond"`
	Samples         int     `json:"samples"`
	Source          string  `json:"source"`
}

// RoutingCompatible recognizes the engine-facing extractors that produce
// qualified phase evidence. Family, API and schema are independent dimensions.
func RoutingCompatible(family, api, schema string) bool {
	return api == "/v1/chat/completions" &&
		((family == "llamacpp" && schema == "llamacpp-b12020-0ec4027a") ||
			(family == "strata" && schema == "strata-f6726098"))
}

func (c RoutingCosts) Validate() error {
	if c.Revision <= 0 {
		return errors.New("revision must be positive")
	}
	if c.Profiles == nil || len(c.Profiles) > MaxEntries {
		return fmt.Errorf("profiles must be an array of at most %d entries", MaxEntries)
	}
	seen := make(map[[6]string]bool, len(c.Profiles))
	for i, p := range c.Profiles {
		if err := p.Validate(); err != nil {
			return fmt.Errorf("profile %d: %w", i, err)
		}
		key := [6]string{p.Host, p.Engine, p.Model, p.Family, p.APIPath, p.Schema}
		if seen[key] {
			return fmt.Errorf("profile %d repeats a key", i)
		}
		seen[key] = true
	}
	data, err := json.Marshal(c)
	if err != nil {
		return err
	}
	if len(data) > MaxPayloadBytes {
		return fmt.Errorf("routing costs exceed %d bytes", MaxPayloadBytes)
	}
	return nil
}

func (p RoutingProfile) Validate() error {
	if err := (ProfileKey{Host: p.Host, Engine: p.Engine, Model: p.Model, Fingerprint: p.Fingerprint}).Validate(); err != nil {
		return err
	}
	if err := checkID("family", p.Family); err != nil {
		return err
	}
	if err := checkID("schema", p.Schema); err != nil {
		return err
	}
	if !RoutingCompatible(p.Family, p.APIPath, p.Schema) {
		return errors.New("unsupported routing measurement tuple")
	}
	if p.AgeMs < 0 {
		return errors.New("ageMs must be non-negative")
	}
	if p.Prefill == nil && p.Decode == nil {
		return errors.New("routing evidence requires a usable phase")
	}
	for _, r := range []*RoutingRate{p.Prefill, p.Decode} {
		if r == nil {
			continue
		}
		if math.IsNaN(r.TokensPerSecond) || math.IsInf(r.TokensPerSecond, 0) || r.TokensPerSecond <= 0 || r.TokensPerSecond > MaxTokensPerSecond || r.Samples <= 0 || r.Samples > MaxSamplesPerKey {
			return errors.New("routing rate requires finite positive speed and samples within bounds")
		}
		if r.Source != "engine" && r.Source != "count" {
			return errors.New("unsupported routing rate source")
		}
	}
	if p.LoadSamples < 0 || p.LoadSamples > MaxSamplesPerKey || (p.LoadP95Ms == nil) != (p.LoadSamples == 0) {
		return errors.New("load duration and samples must be present together")
	}
	if p.LoadP95Ms != nil && (*p.LoadP95Ms < 0 || *p.LoadP95Ms > MaxLatencyMs) {
		return errors.New("loadP95Ms is out of range")
	}
	if len(p.Extra) > 8 {
		return errors.New("extra exceeds 8 keys")
	}
	for key, value := range p.Extra {
		if err := checkText("extra key", key, MaxLabelBytes, true); err != nil {
			return err
		}
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return errors.New("extra value must be finite")
		}
	}
	return nil
}

// DecodeRoutingCosts refuses oversized payloads, unknown fields and trailing data.
func DecodeRoutingCosts(data []byte) (RoutingCosts, error) {
	if len(data) > MaxPayloadBytes {
		return RoutingCosts{}, fmt.Errorf("routing costs exceed %d bytes", MaxPayloadBytes)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var c RoutingCosts
	if err := dec.Decode(&c); err != nil {
		return RoutingCosts{}, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return RoutingCosts{}, errors.New("routing costs: trailing data")
	}
	if err := c.Validate(); err != nil {
		return RoutingCosts{}, err
	}
	return c, nil
}
