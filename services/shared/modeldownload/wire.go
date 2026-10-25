// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package modeldownload defines target-owned download observations. Operation
// outcomes do not establish content readiness or model residency.
package modeldownload

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"strconv"
	"unicode/utf8"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

const (
	Version          = 1
	MaxEntries       = 64
	MaxSnapshotBytes = 256 * 1024
	MaxUnsafeEntries = 32
	MaxRecentEntries = 32
	MaxJournalBytes  = 64 * 1024
)

const (
	PhaseQueued     = "queued"
	PhaseInProgress = "in_progress"
	PhaseFailed     = "failed"
	PhaseComplete   = "complete"
	PhaseUnknown    = "unknown"
)

const (
	ReasonBusy              = "busy"
	ReasonCapacity          = "capacity"
	ReasonDownloadFailed    = "download_failed"
	ReasonInterrupted       = "interrupted"
	ReasonProgressTimeout   = "progress_timeout"
	ReasonAcceptanceUnknown = "acceptance_unknown"
	ReasonCleanupUnknown    = "cleanup_unknown"
	ReasonPersistenceError  = "persistence_error"
	ReasonUnsupported       = "unsupported"

	PersistenceUncertain = "uncertain"
)

// Snapshot is a full observation scoped to one manager incarnation. An absent
// snapshot is unsupported; a current snapshot with entries:[] is known empty.
// Observation currentness belongs to the receiver, outside this DTO.
type Snapshot struct {
	Version          int     `json:"version"`
	SourceID         string  `json:"sourceId"`
	Revision         string  `json:"revision"`
	RecoveryRequired bool    `json:"recoveryRequired"`
	Entries          []Entry `json:"entries"`
}

// Entry preserves the request's exact model spelling. Only the target owner
// knows descriptor-selected naming and can enforce normalized content-key
// uniqueness; this wire contract rejects repeated exact engine/model pairs.
type Entry struct {
	ID              string `json:"id"`
	Engine          string `json:"engine"`
	Model           string `json:"model"`
	Phase           string `json:"phase"`
	BlocksReadiness bool   `json:"blocksReadiness"`
	Percent         *int   `json:"percent,omitempty"`
	Reason          string `json:"reason,omitempty"`
	// Persistence qualifies a terminal outcome whose marker removal landed
	// without confirmed directory durability. It does not certify model files.
	Persistence string `json:"persistence,omitempty"`
}

// Validate checks identities, bounds and lifecycle invariants without returning
// caller-controlled text in diagnostics.
func (s Snapshot) Validate() error {
	if s.Version != Version {
		return errors.New("unsupported model download version")
	}
	if !validID(s.SourceID) {
		return errors.New("invalid model download source ID")
	}
	revision, err := strconv.ParseUint(s.Revision, 10, 64)
	if err != nil || strconv.FormatUint(revision, 10) != s.Revision {
		return errors.New("invalid model download revision")
	}
	if s.Entries == nil || len(s.Entries) > MaxEntries {
		return errors.New("model download entries must be an array within the entry limit")
	}
	ids := make(map[string]bool, len(s.Entries))
	type key struct{ engine, model string }
	keys := make(map[key]bool, len(s.Entries))
	unsafe, recent := 0, 0
	for _, entry := range s.Entries {
		if err := entry.validate(); err != nil {
			return err
		}
		k := key{entry.Engine, entry.Model}
		if ids[entry.ID] || keys[k] {
			return errors.New("duplicate model download identity or key")
		}
		ids[entry.ID], keys[k] = true, true
		if entry.BlocksReadiness {
			unsafe++
		} else {
			recent++
		}
	}
	if unsafe > MaxUnsafeEntries || recent > MaxRecentEntries {
		return errors.New("model download unsafe or recent entry limit exceeded")
	}
	data, err := json.Marshal(s)
	if err != nil || len(data) > MaxSnapshotBytes {
		return errors.New("model download snapshot exceeds byte limit")
	}
	return nil
}

func (e Entry) validate() error {
	if !validID(e.ID) {
		return errors.New("invalid model download ID")
	}
	if len(e.Engine) > engines.MaxDynamicIDLen || !engines.ValidID(e.Engine) {
		return errors.New("invalid model download engine")
	}
	if nodepolicy.ValidateModelIdentifier(e.Model) != nil {
		return errors.New("invalid model download model")
	}
	switch e.Phase {
	case PhaseQueued, PhaseInProgress:
		if !e.BlocksReadiness {
			return errors.New("active model download must block readiness")
		}
	case PhaseComplete:
		if e.BlocksReadiness {
			return errors.New("complete model download must not block readiness")
		}
	case PhaseFailed, PhaseUnknown:
	default:
		return errors.New("invalid model download phase")
	}
	if e.Percent != nil && (*e.Percent < 0 || *e.Percent > 100) {
		return errors.New("invalid model download percent")
	}
	switch e.Reason {
	case "":
	case ReasonBusy, ReasonCapacity, ReasonDownloadFailed, ReasonInterrupted,
		ReasonProgressTimeout, ReasonAcceptanceUnknown, ReasonCleanupUnknown,
		ReasonPersistenceError, ReasonUnsupported:
		if e.Phase != PhaseFailed && e.Phase != PhaseUnknown {
			return errors.New("model download reason requires a failed or unknown outcome")
		}
	default:
		return errors.New("invalid model download reason")
	}
	if e.Persistence != "" {
		if e.Persistence != PersistenceUncertain || e.Phase == PhaseQueued || e.Phase == PhaseInProgress {
			return errors.New("invalid model download persistence qualification")
		}
	}
	return nil
}

func validID(id string) bool {
	if len(id) != 32 {
		return false
	}
	for _, c := range id {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// Clone returns an independent copy, including optional measured percentages.
func (s Snapshot) Clone() Snapshot {
	s.Entries = slices.Clone(s.Entries)
	for i := range s.Entries {
		if percent := s.Entries[i].Percent; percent != nil {
			value := *percent
			s.Entries[i].Percent = &value
		}
	}
	return s
}

// Equal compares all observation fields and entry order, distinguishing an
// absent entries array from a known-empty array and unknown percent from zero.
func (s Snapshot) Equal(other Snapshot) bool {
	if s.Version != other.Version || s.SourceID != other.SourceID || s.Revision != other.Revision ||
		s.RecoveryRequired != other.RecoveryRequired || (s.Entries == nil) != (other.Entries == nil) {
		return false
	}
	return slices.EqualFunc(s.Entries, other.Entries, func(a, b Entry) bool {
		if a.Percent != nil && b.Percent != nil {
			if *a.Percent != *b.Percent {
				return false
			}
		} else if a.Percent != nil || b.Percent != nil {
			return false
		}
		a.Percent, b.Percent = nil, nil
		return a == b
	})
}

// Parse bounds raw bytes before decoding. A malformed extension returns no
// usable snapshot, rather than manufacturing a successful empty observation.
func Parse(data []byte) (Snapshot, error) {
	if len(data) > MaxSnapshotBytes || !utf8.Valid(data) {
		return Snapshot{}, errors.New("model download snapshot exceeds byte limit or is not valid UTF-8")
	}
	var snapshot Snapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return Snapshot{}, errors.New("invalid model download snapshot")
	}
	return snapshot, nil
}

// UnmarshalJSON enforces the same bounded contract when nested in another DTO.
func (s *Snapshot) UnmarshalJSON(data []byte) error {
	type plain Snapshot
	var next plain
	if err := decodeObject(data, &next,
		[]string{"version", "sourceId", "revision", "recoveryRequired", "entries"}, nil); err != nil {
		return err
	}
	if err := Snapshot(next).Validate(); err != nil {
		return err
	}
	*s = Snapshot(next)
	return nil
}

// UnmarshalJSON requires an explicit blocksReadiness boolean, including false.
func (e *Entry) UnmarshalJSON(data []byte) error {
	type plain Entry
	var next plain
	if err := decodeObject(data, &next,
		[]string{"id", "engine", "model", "phase", "blocksReadiness"},
		[]string{"percent", "reason", "persistence"}); err != nil {
		return err
	}
	if err := Entry(next).validate(); err != nil {
		return err
	}
	*e = Entry(next)
	return nil
}

// decodeObject rejects unknown, repeated, missing and null fields before typed
// decoding. Errors deliberately omit decoder details and input field names.
func decodeObject(data []byte, dst any, required, optional []string) error {
	if len(data) > MaxSnapshotBytes || !utf8.Valid(data) {
		return errors.New("model download JSON exceeds byte limit or is not valid UTF-8")
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	start, err := dec.Token()
	if err != nil || start != json.Delim('{') {
		return errors.New("model download JSON must be an object")
	}
	seen := make(map[string]bool, len(required)+len(optional))
	for dec.More() {
		token, err := dec.Token()
		if err != nil {
			return errors.New("invalid model download JSON field")
		}
		name, ok := token.(string)
		if !ok || seen[name] || !slices.Contains(required, name) && !slices.Contains(optional, name) {
			return errors.New("unknown or duplicate model download JSON field")
		}
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return errors.New("invalid or null model download JSON field")
		}
		seen[name] = true
	}
	if _, err := dec.Token(); err != nil {
		return errors.New("invalid model download JSON object")
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("trailing model download JSON data")
	}
	for _, name := range required {
		if !seen[name] {
			return errors.New("missing required model download JSON field")
		}
	}
	if err := json.Unmarshal(data, dst); err != nil {
		return errors.New("invalid model download JSON value")
	}
	return nil
}
