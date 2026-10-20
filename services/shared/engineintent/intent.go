// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package engineintent defines the saved-intent wire contract shared by the
// engine manager and its clients. It describes intent, not runtime status.
package engineintent

import (
	"encoding/json"
	"fmt"
)

// Origin identifies who saved Off relative to the engine-owning device.
type Origin string

const (
	Local Origin = "local"
	Peer  Origin = "peer"
)

// Snapshot is the engine:intent result and engine:intent-changed payload.
// EnabledByEngine reports every registered engine; unknown intent is false.
// OffByEngine contains only explicitly saved Off engines. Legacy saved Off
// without stored provenance is Local; unknown and On engines have no origin.
// Both maps are required, including when empty, and come from one store read.
type Snapshot struct {
	EnabledByEngine map[string]bool   `json:"enabledByEngine"`
	OffByEngine     map[string]Origin `json:"offByEngine"`
}

// UnmarshalJSON rejects null enabled values instead of treating them as false.
func (s *Snapshot) UnmarshalJSON(data []byte) error {
	type snapshot Snapshot
	var decoded Snapshot
	wire := struct {
		*snapshot
		EnabledByEngine map[string]*bool `json:"enabledByEngine"`
	}{snapshot: (*snapshot)(&decoded)}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	if wire.EnabledByEngine != nil {
		decoded.EnabledByEngine = make(map[string]bool, len(wire.EnabledByEngine))
		for engine, enabled := range wire.EnabledByEngine {
			if enabled == nil {
				return fmt.Errorf("intent enabled value for %s must be a boolean", engine)
			}
			decoded.EnabledByEngine[engine] = *enabled
		}
	}
	*s = decoded
	return nil
}

// Validate rejects incomplete snapshots and origins inconsistent with the
// reported engines. Absence from OffByEngine is valid for unknown intent.
func (s Snapshot) Validate() error {
	if s.EnabledByEngine == nil || s.OffByEngine == nil {
		return fmt.Errorf("intent snapshot requires enabledByEngine and offByEngine objects")
	}
	for engine, origin := range s.OffByEngine {
		if origin != Local && origin != Peer {
			return fmt.Errorf("invalid Off origin for %s: %q", engine, origin)
		}
		enabled, reported := s.EnabledByEngine[engine]
		if !reported || enabled {
			return fmt.Errorf("Off origin for %s requires a reported disabled engine", engine)
		}
	}
	return nil
}
