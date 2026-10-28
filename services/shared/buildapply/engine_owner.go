// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"sort"
	"strings"

	"nvpair-shared/engines"
)

const EngineOwnerEvidenceMethod = "engine:owner-evidence"

// EngineOwnerEvidenceRequest has no selectors or mutation authority.
type EngineOwnerEvidenceRequest struct{}

type EngineOwnerIntent struct {
	Revision        string            `json:"revision"`
	EnabledByEngine map[string]bool   `json:"enabledByEngine"`
	OffByEngine     map[string]string `json:"offByEngine"`
}

type EngineOwnerLocks struct {
	Revision       string          `json:"revision"`
	LockedByEngine map[string]bool `json:"lockedByEngine"`
}

type EngineOwnerUniverseEntry struct {
	Engine              string                `json:"engine"`
	Registered          bool                  `json:"registered"`
	Supported           Observation[bool]     `json:"supported"`
	Installed           Observation[bool]     `json:"installed"`
	Running             Observation[bool]     `json:"running"`
	EngineRunID         Observation[string]   `json:"engineRunId"`
	TerminationVerified Observation[bool]     `json:"terminationVerified"`
	ModelRoots          Observation[[]string] `json:"modelRoots"`
	DataRoots           Observation[[]string] `json:"dataRoots"`
	RootsComplete       bool                  `json:"rootsComplete"`
}

// The run identity is issued by the engine owner. PID is known only from its
// retained process object; neither PID lookup nor a serialized pointer is proof.
type EngineOwnerProcessLifetime struct {
	EngineRunID string           `json:"engineRunId"`
	PID         Observation[int] `json:"pid"`
}

type EngineOwnerStopAuthority struct {
	Engine          string                                  `json:"engine"`
	Scoped          Observation[bool]                       `json:"scoped"`
	Adopted         Observation[bool]                       `json:"adopted"`
	ReclaimPending  Observation[bool]                       `json:"reclaimPending"`
	ProcessLifetime Observation[EngineOwnerProcessLifetime] `json:"processLifetime"`
	Reason          string                                  `json:"reason"`
}

type EngineOwnerOperation struct {
	ID                string              `json:"id"`
	Engine            string              `json:"engine"`
	Kind              string              `json:"kind"`
	Phase             string              `json:"phase"`
	SourceIncarnation Observation[string] `json:"sourceIncarnation"`
	Retained          bool                `json:"retained"`
	Unsafe            bool                `json:"unsafe"`
}

type EngineOwnerReservations struct {
	Revision             string                 `json:"revision"`
	BlocksApply          Observation[bool]      `json:"blocksApply"`
	UnsafeCopiesComplete bool                   `json:"unsafeCopiesComplete"`
	Operations           []EngineOwnerOperation `json:"operations"`
}

type EngineOwnerReason struct {
	Section string `json:"section"`
	Engine  string `json:"engine"`
	Code    string `json:"code"`
}

// Private parent evidence. Complete describes knowledge, never permission to
// stop, reserve or apply. Explicit false intent/locks and unknown provenance
// survive transport. No field contains commands, native errors or credentials.
type EngineOwnerEvidence struct {
	ManagerIncarnation string                                  `json:"managerIncarnation"`
	CaptureToken       string                                  `json:"captureToken"`
	Intent             Observation[EngineOwnerIntent]          `json:"intent"`
	Locks              Observation[EngineOwnerLocks]           `json:"locks"`
	Universe           Observation[[]EngineOwnerUniverseEntry] `json:"universe"`
	StopAuthority      Observation[[]EngineOwnerStopAuthority] `json:"stopAuthority"`
	Reservations       Observation[EngineOwnerReservations]    `json:"reservations"`
	Complete           bool                                    `json:"complete"`
	Reasons            []EngineOwnerReason                     `json:"reasons"`
}

// ImpactEnvelope is the private broker-to-service result. Only Impact may be
// projected into public preview output. An unavailable manager read is unknown,
// never a zero-valued manager document. Service profile roots stay service-owned.
type ImpactEnvelope struct {
	Impact          Impact                           `json:"impact"`
	ManagerEvidence Observation[EngineOwnerEvidence] `json:"managerEvidence"`
}

func (v ImpactEnvelope) MarshalJSON() ([]byte, error) {
	type wire ImpactEnvelope
	data, err := json.Marshal(wire(v))
	if err == nil && len(data) > MaxStatusBytes {
		return nil, errors.New("build apply impact exceeds limit")
	}
	return data, err
}

func (v *ImpactEnvelope) UnmarshalJSON(data []byte) error {
	type wire ImpactEnvelope
	if len(data) == 0 || len(data) > MaxStatusBytes {
		return errors.New("invalid build apply impact size")
	}
	var result wire
	if err := decodeResponse(data, &result, "impact", "managerEvidence"); err != nil {
		return err
	}
	*v = ImpactEnvelope(result)
	return nil
}

func ownerEngineID(value string) bool {
	return len(value) <= engines.MaxDynamicIDLen && engines.ValidID(value)
}

func ownerRevision(value string) bool {
	return strings.HasPrefix(value, "sha256:") && len(value) == len("sha256:")+64 && wireRevision(strings.TrimPrefix(value, "sha256:"))
}
func ownerIdentity(value string) bool { return token(value, false) }

func ownerRoots(value []string) bool {
	if value == nil {
		return false
	}
	for i, path := range value {
		if !localPath(path) || strings.ContainsAny(path, "{}$%~") || i > 0 && value[i-1] >= path {
			return false
		}
	}
	return true
}

func ownerReason(code string) bool {
	switch code {
	case "busy", "saved_state_unknown", "changed", "unmodeled_data_roots", "unobserved",
		"reclaim_unknown", "process_lifetime_closed", "external_management", "no_live_process",
		"owned_process_lifetime", "scoped_lifetime_unknown", "content_authority_unknown", "source_mismatch", "unavailable":
		return true
	}
	return false
}

func (e EngineOwnerEvidence) knowledgeComplete() bool {
	if !e.Intent.Known || !e.Locks.Known || !e.Universe.Known || !e.StopAuthority.Known || !e.Reservations.Known {
		return false
	}
	for _, entry := range *e.Universe.Value {
		if !entry.Supported.Known || !entry.Installed.Known || !entry.Running.Known || !entry.TerminationVerified.Known || !entry.RootsComplete {
			return false
		}
	}
	for _, entry := range *e.StopAuthority.Value {
		if !entry.Scoped.Known || !entry.Adopted.Known || !entry.ReclaimPending.Known {
			return false
		}
		if entry.ProcessLifetime.Known && !entry.ProcessLifetime.Value.PID.Known {
			return false
		}
	}
	return e.Reservations.Value.BlocksApply.Known && e.Reservations.Value.UnsafeCopiesComplete
}

func (e EngineOwnerEvidence) validateSections() error {
	invalid := errors.New("invalid engine owner evidence")
	if !ownerIdentity(e.ManagerIncarnation) || e.Reasons == nil {
		return invalid
	}
	if !validObservation(e.Intent, func(v EngineOwnerIntent) bool {
		if !ownerRevision(v.Revision) || v.EnabledByEngine == nil || v.OffByEngine == nil {
			return false
		}
		for engine, enabled := range v.EnabledByEngine {
			if !ownerEngineID(engine) {
				return false
			}
			origin, exists := v.OffByEngine[engine]
			if enabled && exists || !enabled && (!exists || origin != "local" && origin != "peer") {
				return false
			}
		}
		for engine := range v.OffByEngine {
			if enabled, exists := v.EnabledByEngine[engine]; !exists || enabled {
				return false
			}
		}
		return true
	}) || !validObservation(e.Locks, func(v EngineOwnerLocks) bool {
		if !ownerRevision(v.Revision) || v.LockedByEngine == nil {
			return false
		}
		for engine := range v.LockedByEngine {
			if !ownerEngineID(engine) {
				return false
			}
		}
		return true
	}) {
		return invalid
	}
	names := map[string]bool{}
	if !validObservation(e.Universe, func(entries []EngineOwnerUniverseEntry) bool {
		if entries == nil {
			return false
		}
		for i, v := range entries {
			if !ownerEngineID(v.Engine) || i > 0 && entries[i-1].Engine >= v.Engine {
				return false
			}
			names[v.Engine] = true
			for _, b := range []Observation[bool]{v.Supported, v.Installed, v.Running, v.TerminationVerified} {
				if !validObservation(b, func(bool) bool { return true }) {
					return false
				}
			}
			if !validObservation(v.EngineRunID, ownerIdentity) || !validObservation(v.ModelRoots, ownerRoots) || !validObservation(v.DataRoots, ownerRoots) || v.RootsComplete && (!v.ModelRoots.Known || !v.DataRoots.Known) {
				return false
			}
			if !v.Registered && v.Supported.Known || v.Running.Known && *v.Running.Value && v.TerminationVerified.Known && *v.TerminationVerified.Value {
				return false
			}
		}
		return true
	}) {
		return invalid
	}
	if !validObservation(e.StopAuthority, func(entries []EngineOwnerStopAuthority) bool {
		if entries == nil || !e.Universe.Known || len(entries) != len(names) {
			return false
		}
		for i, v := range entries {
			if !names[v.Engine] || i > 0 && entries[i-1].Engine >= v.Engine || !ownerReason(v.Reason) {
				return false
			}
			for _, b := range []Observation[bool]{v.Scoped, v.Adopted, v.ReclaimPending} {
				if !validObservation(b, func(bool) bool { return true }) {
					return false
				}
			}
			if !validObservation(v.ProcessLifetime, func(p EngineOwnerProcessLifetime) bool {
				return ownerIdentity(p.EngineRunID) && validObservation(p.PID, func(pid int) bool { return pid > 0 && pid <= 2147483647 })
			}) {
				return false
			}
			universe := (*e.Universe.Value)[i]
			if v.ProcessLifetime.Known && (!universe.EngineRunID.Known || *universe.EngineRunID.Value != v.ProcessLifetime.Value.EngineRunID || !v.Scoped.Known || !*v.Scoped.Value || v.Reason != "owned_process_lifetime") {
				return false
			}
			if v.Scoped.Known {
				if !v.Adopted.Known || !v.ReclaimPending.Known {
					return false
				}
				if *v.Scoped.Value {
					if *v.Adopted.Value || *v.ReclaimPending.Value || !universe.Running.Known {
						return false
					}
					switch v.Reason {
					case "owned_process_lifetime":
						if !v.ProcessLifetime.Known || !*universe.Running.Value {
							return false
						}
					case "no_live_process":
						if v.ProcessLifetime.Known || *universe.Running.Value || !universe.TerminationVerified.Known || !*universe.TerminationVerified.Value {
							return false
						}
					default:
						return false
					}
				} else if !*v.Adopted.Value || v.Reason != "external_management" {
					return false
				}
			}
		}
		return true
	}) {
		return invalid
	}
	if e.Universe.Known {
		if e.Intent.Known {
			for engine := range e.Intent.Value.EnabledByEngine {
				if !names[engine] {
					return invalid
				}
			}
		}
		if e.Locks.Known {
			for engine := range e.Locks.Value.LockedByEngine {
				if !names[engine] {
					return invalid
				}
			}
		}
	}
	if e.Intent.Known && e.Locks.Known {
		for engine, locked := range e.Locks.Value.LockedByEngine {
			if locked {
				enabled, exists := e.Intent.Value.EnabledByEngine[engine]
				if !exists || enabled || e.Intent.Value.OffByEngine[engine] != "local" {
					return invalid
				}
			}
		}
	}
	if !validObservation(e.Reservations, func(v EngineOwnerReservations) bool {
		if !ownerRevision(v.Revision) || v.Operations == nil || !validObservation(v.BlocksApply, func(bool) bool { return true }) {
			return false
		}
		ids := map[string]bool{}
		for i, op := range v.Operations {
			if !ownerIdentity(op.ID) || ids[op.ID] || !ownerEngineID(op.Engine) || e.Universe.Known && !names[op.Engine] || i > 0 && v.Operations[i-1].ID >= op.ID || !validObservation(op.SourceIncarnation, ownerIdentity) {
				return false
			}
			ids[op.ID] = true
			switch op.Kind {
			case "download", "copy", "content_reader", "content_writer":
			default:
				return false
			}
			switch op.Phase {
			case "queued", "in_progress", "unsafe", "uncertain", "unknown", "active":
			default:
				return false
			}
			if !op.Retained && (!op.SourceIncarnation.Known || *op.SourceIncarnation.Value != e.ManagerIncarnation) {
				return false
			}
			if op.Unsafe && !op.Retained || v.UnsafeCopiesComplete && (!op.SourceIncarnation.Known || *op.SourceIncarnation.Value != e.ManagerIncarnation) {
				return false
			}
		}
		if len(v.Operations) != 0 && (!v.BlocksApply.Known || !*v.BlocksApply.Value) {
			return false
		}
		return !v.UnsafeCopiesComplete || v.BlocksApply.Known
	}) {
		return invalid
	}
	for i, reason := range e.Reasons {
		if !ownerReason(reason.Code) || reason.Engine != "" && !ownerEngineID(reason.Engine) {
			return invalid
		}
		switch reason.Section {
		case "intent", "locks", "universe", "stopAuthority", "reservations":
		default:
			return invalid
		}
		if i > 0 && ownerReasonKey(e.Reasons[i-1]) >= ownerReasonKey(reason) {
			return invalid
		}
	}
	for section, known := range map[string]bool{"intent": e.Intent.Known, "locks": e.Locks.Known, "universe": e.Universe.Known, "stopAuthority": e.StopAuthority.Known, "reservations": e.Reservations.Known} {
		if known {
			continue
		}
		explained := false
		for _, reason := range e.Reasons {
			if reason.Section == section && reason.Engine == "" {
				explained = true
			}
		}
		if !explained {
			return invalid
		}
	}
	return nil
}

func ownerReasonKey(r EngineOwnerReason) string { return r.Section + "/" + r.Engine + "/" + r.Code }

func (e EngineOwnerEvidence) digest() (string, error) {
	type wire EngineOwnerEvidence
	e.CaptureToken = ""
	data, err := json.Marshal(wire(e))
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// Finalize derives completeness and the canonical comparison token. It does
// not reserve any owner, and changes to unknown/known sections change the token.
func (e *EngineOwnerEvidence) Finalize() error {
	sort.Slice(e.Reasons, func(i, j int) bool { return ownerReasonKey(e.Reasons[i]) < ownerReasonKey(e.Reasons[j]) })
	if err := e.validateSections(); err != nil {
		return err
	}
	e.Complete = e.knowledgeComplete()
	var err error
	e.CaptureToken, err = e.digest()
	return err
}

func (e EngineOwnerEvidence) validate() error {
	if err := e.validateSections(); err != nil {
		return err
	}
	digest, err := e.digest()
	if err != nil {
		return err
	}
	if e.Complete != e.knowledgeComplete() || e.CaptureToken != digest {
		return errors.New("inconsistent engine owner capture")
	}
	return nil
}

func (e EngineOwnerEvidence) MarshalJSON() ([]byte, error) {
	type wire EngineOwnerEvidence
	if err := e.validate(); err != nil {
		return nil, err
	}
	data, err := json.Marshal(wire(e))
	if err == nil && len(data) > MaxStatusBytes {
		return nil, errors.New("engine owner evidence exceeds limit")
	}
	return data, err
}

func (e *EngineOwnerEvidence) UnmarshalJSON(data []byte) error {
	type wire EngineOwnerEvidence
	var value wire
	if len(data) == 0 || len(data) > MaxStatusBytes {
		return errors.New("invalid engine owner evidence size")
	}
	if err := decodeResponse(data, &value, "managerIncarnation", "captureToken", "intent", "locks", "universe", "stopAuthority", "reservations", "complete", "reasons"); err != nil {
		return err
	}
	result := EngineOwnerEvidence(value)
	if err := result.validate(); err != nil {
		return err
	}
	*e = result
	return nil
}

func (v *EngineOwnerIntent) UnmarshalJSON(data []byte) error {
	type wire EngineOwnerIntent
	if err := decodeResponse(data, (*wire)(v), "revision", "enabledByEngine", "offByEngine"); err != nil {
		return err
	}
	return ownerBooleanMap(data, "enabledByEngine")
}
func (v *EngineOwnerLocks) UnmarshalJSON(data []byte) error {
	type wire EngineOwnerLocks
	if err := decodeResponse(data, (*wire)(v), "revision", "lockedByEngine"); err != nil {
		return err
	}
	return ownerBooleanMap(data, "lockedByEngine")
}

func ownerBooleanMap(data []byte, field string) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	var values map[string]json.RawMessage
	if err := json.Unmarshal(fields[field], &values); err != nil {
		return err
	}
	for _, raw := range values {
		raw = bytes.TrimSpace(raw)
		if !bytes.Equal(raw, []byte("true")) && !bytes.Equal(raw, []byte("false")) {
			return errors.New("invalid engine owner boolean")
		}
	}
	return nil
}
func (v *EngineOwnerUniverseEntry) UnmarshalJSON(data []byte) error {
	type wire EngineOwnerUniverseEntry
	return decodeResponse(data, (*wire)(v), "engine", "registered", "supported", "installed", "running", "engineRunId", "terminationVerified", "modelRoots", "dataRoots", "rootsComplete")
}
func (v *EngineOwnerStopAuthority) UnmarshalJSON(data []byte) error {
	type wire EngineOwnerStopAuthority
	return decodeResponse(data, (*wire)(v), "engine", "scoped", "adopted", "reclaimPending", "processLifetime", "reason")
}
func (v *EngineOwnerProcessLifetime) UnmarshalJSON(data []byte) error {
	type wire EngineOwnerProcessLifetime
	return decodeResponse(data, (*wire)(v), "engineRunId", "pid")
}
func (v *EngineOwnerReservations) UnmarshalJSON(data []byte) error {
	type wire EngineOwnerReservations
	return decodeResponse(data, (*wire)(v), "revision", "blocksApply", "unsafeCopiesComplete", "operations")
}
func (v *EngineOwnerOperation) UnmarshalJSON(data []byte) error {
	type wire EngineOwnerOperation
	return decodeResponse(data, (*wire)(v), "id", "engine", "kind", "phase", "sourceIncarnation", "retained", "unsafe")
}
func (v *EngineOwnerReason) UnmarshalJSON(data []byte) error {
	type wire EngineOwnerReason
	return decodeResponse(data, (*wire)(v), "section", "engine", "code")
}
