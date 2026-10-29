// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"encoding/json"
	"fmt"
	"slices"
	"time"
)

// MemoryProfile is an advisory statement for one applied configuration.
type MemoryProfile struct {
	ConfigurationKey string `json:"configurationKey"`
	Kind             string `json:"kind"`
	RecordedAt       string `json:"recordedAt"`
	HostBytes        string `json:"hostBytes,omitempty"`
	DeviceBytes      string `json:"deviceBytes,omitempty"`
	SharedBytes      string `json:"sharedBytes,omitempty"`
	Current          bool   `json:"current"`
}

func (p MemoryProfile) Clone() MemoryProfile { return p }
func profileByteFields(data []byte) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for _, name := range []string{"hostBytes", "deviceBytes", "sharedBytes"} {
		if raw, ok := fields[name]; ok {
			var value string
			if err := json.Unmarshal(raw, &value); err != nil || ValidateUnsignedDecimal(value) != nil {
				return fmt.Errorf("%s: canonical byte decimal required", name)
			}
		}
	}
	return nil
}
func (p *MemoryProfile) UnmarshalJSON(data []byte) error {
	type wire MemoryProfile
	var value wire
	if err := DecodeStrict(data, &value); err != nil {
		return err
	}
	if err := profileByteFields(data); err != nil {
		return err
	}
	*p = MemoryProfile(value)
	return nil
}
func (p MemoryProfile) Validate() error {
	if !digest(p.ConfigurationKey) || p.Kind != "stated" {
		return fmt.Errorf("memoryProfile: invalid configuration identity or kind")
	}
	at, err := time.Parse(time.RFC3339Nano, p.RecordedAt)
	if err != nil || at.IsZero() || len(p.RecordedAt) > 64 {
		return fmt.Errorf("memoryProfile.recordedAt: valid timestamp required")
	}
	if p.HostBytes == "" && p.DeviceBytes == "" && p.SharedBytes == "" {
		return fmt.Errorf("memoryProfile: pool bytes required")
	}
	for _, v := range []string{p.HostBytes, p.DeviceBytes, p.SharedBytes} {
		if v != "" {
			if err := ValidateUnsignedDecimal(v); err != nil {
				return fmt.Errorf("memoryProfile: canonical byte decimal required")
			}
		}
	}
	if p.SharedBytes != "" && (p.HostBytes != "" || p.DeviceBytes != "") {
		return fmt.Errorf("memoryProfile: shared pool is exclusive")
	}
	return nil
}
func (p MemoryProfile) ValidatePools(pools []string) error {
	if err := p.Validate(); err != nil {
		return err
	}
	if !slices.Equal(pools, []string{"host"}) && !slices.Equal(pools, []string{"host", "device"}) && !slices.Equal(pools, []string{"shared"}) {
		return fmt.Errorf("memoryProfile: applied pools unavailable")
	}
	if (p.HostBytes != "") != slices.Contains(pools, "host") || (p.DeviceBytes != "") != slices.Contains(pools, "device") || (p.SharedBytes != "") != slices.Contains(pools, "shared") {
		return fmt.Errorf("memoryProfile: bytes must match every applied configuration pool exactly")
	}
	return nil
}

type ClearStatedMemoryRequest struct {
	Engine           string `json:"engine"`
	NodeID           string `json:"nodeId,omitempty"`
	ModelID          string `json:"modelId"`
	ConfigurationKey string `json:"configurationKey"`
}

func (p ClearStatedMemoryRequest) Clone() ClearStatedMemoryRequest { return p }
func (p ClearStatedMemoryRequest) Validate() error {
	if err := (InspectRequest{Engine: p.Engine, NodeID: p.NodeID, ModelID: p.ModelID}).Validate(); err != nil {
		return err
	}
	if !digest(p.ConfigurationKey) {
		return fmt.Errorf("configurationKey: applied configuration digest required")
	}
	return CheckPayloadSize(p)
}

type SetStatedMemoryRequest struct {
	Engine           string `json:"engine"`
	NodeID           string `json:"nodeId,omitempty"`
	ModelID          string `json:"modelId"`
	ConfigurationKey string `json:"configurationKey"`
	HostBytes        string `json:"hostBytes,omitempty"`
	DeviceBytes      string `json:"deviceBytes,omitempty"`
	SharedBytes      string `json:"sharedBytes,omitempty"`
}

func (p SetStatedMemoryRequest) Clone() SetStatedMemoryRequest { return p }
func (p SetStatedMemoryRequest) Validate() error {
	if err := (ClearStatedMemoryRequest{p.Engine, p.NodeID, p.ModelID, p.ConfigurationKey}).Validate(); err != nil {
		return err
	}
	return (MemoryProfile{ConfigurationKey: p.ConfigurationKey, Kind: "stated", RecordedAt: "2000-01-01T00:00:00Z", HostBytes: p.HostBytes, DeviceBytes: p.DeviceBytes, SharedBytes: p.SharedBytes}).Validate()
}
func DecodeSetStatedMemoryRequest(data []byte) (SetStatedMemoryRequest, error) {
	var p SetStatedMemoryRequest
	if err := DecodeStrict(data, &p); err != nil {
		return p, err
	}
	if err := profileByteFields(data); err != nil {
		return p, err
	}
	return p, p.Validate()
}
func DecodeClearStatedMemoryRequest(data []byte) (ClearStatedMemoryRequest, error) {
	var p ClearStatedMemoryRequest
	if err := DecodeStrict(data, &p); err != nil {
		return p, err
	}
	return p, p.Validate()
}
