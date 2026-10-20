// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package performancewire

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// This file is the standalone local memory-evidence contract. It does not add
// memory fields to performance:summary or change that message's schema.
//
// Every bound is nullable. A nil pointer means the producer could not establish
// the value; a pointer to zero is a measured zero. Consumers must never read a
// missing value as zero or as an upper bound. An observed allocation is not a
// qualified upper bound.

const (
	// MemoryEvidenceVersion is the memory-evidence envelope version.
	MemoryEvidenceVersion = 1

	// MaxMemoryEvidenceBytes bounds one encoded MemoryEvidence, matching the
	// existing telemetry payload bound.
	MaxMemoryEvidenceBytes = MaxPayloadBytes
	// MaxMemoryEvidenceEntries bounds the retained entries in one envelope.
	// OmittedEntries is a separate nonnegative count and is not bounded by it.
	MaxMemoryEvidenceEntries = MaxEntries

	// MemoryReadingValidForMs is the fixed freshness window of an allocation
	// reading: three default watcher periods.
	MemoryReadingValidForMs int64 = 15_000

	// MaxMemoryBytes bounds one observed or qualified byte count: one PiB.
	MaxMemoryBytes int64 = 1 << 50
	// MaxMemoryConcurrentExecutions bounds a qualified concurrency envelope.
	MaxMemoryConcurrentExecutions int64 = 1024
)

// MemoryState is how far an entry's memory evidence has been established.
type MemoryState string

const (
	// MemoryStateUnknown means evidence is incomplete. Known partial metadata
	// may remain present, but there is no qualified upper bound.
	MemoryStateUnknown MemoryState = "unknown"
	// MemoryStateAdvisory means the entry carries observations but no qualified
	// upper bound.
	MemoryStateAdvisory MemoryState = "advisory"
	// MemoryStateQualified means the entry carries a current, fully bound upper
	// bound and a fresh reading within it.
	MemoryStateQualified MemoryState = "qualified"
)

// MemoryUncertainty is a bounded reason an entry is not qualified.
type MemoryUncertainty string

const (
	MemoryUncertaintyUnsupported           MemoryUncertainty = "unsupported"
	MemoryUncertaintyMissingReading        MemoryUncertainty = "missing-reading"
	MemoryUncertaintyIdentityUnknown       MemoryUncertainty = "identity-unknown"
	MemoryUncertaintyIdentityChanged       MemoryUncertainty = "identity-changed"
	MemoryUncertaintySettingsUnapplied     MemoryUncertainty = "settings-unapplied"
	MemoryUncertaintyContextUnknown        MemoryUncertainty = "context-unknown"
	MemoryUncertaintyConcurrencyUnknown    MemoryUncertainty = "concurrency-unknown"
	MemoryUncertaintyHardwareUnqualified   MemoryUncertainty = "hardware-unqualified"
	MemoryUncertaintyQualificationMissing  MemoryUncertainty = "qualification-missing"
	MemoryUncertaintyQualificationMismatch MemoryUncertainty = "qualification-mismatch"
	MemoryUncertaintyClockUncertain        MemoryUncertainty = "clock-uncertain"
	MemoryUncertaintyStale                 MemoryUncertainty = "stale"
	MemoryUncertaintyBoundExceeded         MemoryUncertainty = "bound-exceeded"
)

// MemoryDomain names the memory pool a qualification covers. Domains are not
// summed: a dedicated device, host physical memory and unified physical memory
// are separate budgets.
type MemoryDomain string

const (
	MemoryDomainDedicatedDevice MemoryDomain = "dedicated-device"
	MemoryDomainHostPhysical    MemoryDomain = "host-physical"
	MemoryDomainUnifiedPhysical MemoryDomain = "unified-physical"
)

// MemoryQualificationScope names who attested a qualification. The only
// supported scope is an operator-attested measurement of the current run; it
// is not independent runtime hardware verification.
type MemoryQualificationScope string

const (
	MemoryQualificationScopeOperatorAttestedCurrentRun MemoryQualificationScope = "operator-attested-current-run"
)

// MemoryEvidence is one bounded set of memory evidence entries. Entries is
// required and may be empty; OmittedEntries is a nonnegative count of entries
// left out, which may exceed the retained-entry cap.
type MemoryEvidence struct {
	Version        int                   `json:"version"`
	Entries        []MemoryEvidenceEntry `json:"entries"`
	OmittedEntries int                   `json:"omittedEntries"`
}

// MemoryEvidenceEntry is one engine/model/domain memory observation. Every
// group is structurally required; individual unknown values are explicit nulls.
type MemoryEvidenceEntry struct {
	Engine          string                `json:"engine"`
	Model           string                `json:"model"`
	State           MemoryState           `json:"state"`
	Uncertainty     []MemoryUncertainty   `json:"uncertainty"`
	Source          MemorySource          `json:"source"`
	AppliedSettings MemoryAppliedSettings `json:"appliedSettings"`
	Envelope        MemoryEnvelope        `json:"envelope"`
	Observed        MemoryObservation     `json:"observed"`
	Qualification   MemoryQualification   `json:"qualification"`
	Provenance      MemoryProvenance      `json:"provenance"`
	Freshness       MemoryFreshness       `json:"freshness"`
}

// MemorySource binds evidence to a current process, engine run, launch and
// concrete model. A qualified entry requires every member.
type MemorySource struct {
	HostUUID           *string `json:"hostUuid"`
	ManagerIncarnation *string `json:"managerIncarnation"`
	EngineRunID        *string `json:"engineRunId"`
	LaunchDigest       *string `json:"launchDigest"`
	ModelDigest        *string `json:"modelDigest"`
}

// MemoryAppliedSettings identifies the fully applied settings snapshot. It is
// not the broker's SettingsFingerprint and carries no second hashing formula.
type MemoryAppliedSettings struct {
	Epoch    *string `json:"epoch"`
	Revision *int64  `json:"revision"`
}

// MemoryEnvelope is the tested serving envelope a qualification covers. A
// qualified entry requires every member, including the offload/runtime mode.
type MemoryEnvelope struct {
	Domain                 *MemoryDomain `json:"domain"`
	EffectiveContextTokens *int64        `json:"effectiveContextTokens"`
	ConcurrentExecutions   *int64        `json:"concurrentExecutions"`
	OffloadMode            *string       `json:"offloadMode"`
}

// MemoryObservation is a current allocation reading. AllocatedBytes is
// nullable and may be a measured zero. ExtractorRevision names the exact
// extractor/runtime revision that produced it.
type MemoryObservation struct {
	AllocatedBytes    *int64  `json:"allocatedBytes"`
	ExtractorRevision *string `json:"extractorRevision"`
}

// MemoryQualification is the reviewed upper bound and the document identity it
// came from. UpperBytes is nullable; only a qualified entry may set it.
type MemoryQualification struct {
	UpperBytes      *int64  `json:"upperBytes"`
	ID              *string `json:"id"`
	Revision        *string `json:"revision"`
	ArtifactDigest  *string `json:"artifactDigest"`
	ProcedureDigest *string `json:"procedureDigest"`
}

// MemoryProvenance names the measurement procedure, hardware fingerprint,
// observed runtime/hardware/driver descriptions and attestation scope. The
// descriptions are operator-attested, not runtime hardware authentication. A
// qualified entry requires every member.
type MemoryProvenance struct {
	ProcedureID         *string                   `json:"procedureId"`
	ProcedureRevision   *string                   `json:"procedureRevision"`
	HardwareFingerprint *string                   `json:"hardwareFingerprint"`
	RuntimeRevision     *string                   `json:"runtimeRevision"`
	HardwareDescription *string                   `json:"hardwareDescription"`
	DriverRevision      *string                   `json:"driverRevision"`
	Scope               *MemoryQualificationScope `json:"scope"`
}

// MemoryFreshness carries monotonic ages and the source observation
// generation. SampleAgeMs is limited to MemoryReadingValidForMs only for a
// qualified entry; advisory and unknown states may represent a stale reading.
// QualificationAgeMs is separate and is never refreshed by a new allocation
// read, and it has no wall-age expiration. ObservationGeneration identifies the
// source observation and is positive when present. Unknown values are null.
type MemoryFreshness struct {
	SampleAgeMs        *int64  `json:"sampleAgeMs"`
	QualificationAgeMs *int64  `json:"qualificationAgeMs"`
	Generation         *uint64 `json:"observationGeneration"`
	ReadingValidForMs  int64   `json:"readingValidForMs"`
}

// Validate checks the envelope, its limits, entry keys and every entry.
func (m MemoryEvidence) Validate() error {
	if m.Version != MemoryEvidenceVersion {
		return fmt.Errorf("memory evidence version %d is not supported", m.Version)
	}
	if m.Entries == nil {
		return errors.New("entries is required and must not be null")
	}
	if len(m.Entries) > MaxMemoryEvidenceEntries {
		return fmt.Errorf("%d entries exceed %d", len(m.Entries), MaxMemoryEvidenceEntries)
	}
	if m.OmittedEntries < 0 {
		return fmt.Errorf("omittedEntries %d must be nonnegative", m.OmittedEntries)
	}
	type entryKey struct {
		engine string
		model  string
		domain MemoryDomain
	}
	seen := make(map[entryKey]bool, len(m.Entries))
	for i, e := range m.Entries {
		if err := e.Validate(); err != nil {
			return fmt.Errorf("entry %d: %w", i, err)
		}
		var domain MemoryDomain
		if e.Envelope.Domain != nil {
			domain = *e.Envelope.Domain
		}
		key := entryKey{engine: e.Engine, model: e.Model, domain: domain}
		if seen[key] {
			return fmt.Errorf("entry %d repeats an engine/model/domain key", i)
		}
		seen[key] = true
	}
	return nil
}

// Validate checks one entry's strings, groups and state-specific requirements.
func (e MemoryEvidenceEntry) Validate() error {
	if err := checkID("engine", e.Engine); err != nil {
		return err
	}
	if err := checkText("model", e.Model, MaxModelBytes, true); err != nil {
		return err
	}
	switch e.State {
	case MemoryStateUnknown, MemoryStateAdvisory, MemoryStateQualified:
	default:
		return fmt.Errorf("state %q is not supported", e.State)
	}
	if e.Uncertainty == nil {
		return errors.New("uncertainty is required and must not be null")
	}
	seen := make(map[MemoryUncertainty]bool, len(e.Uncertainty))
	for _, u := range e.Uncertainty {
		if !validMemoryUncertainty(u) {
			return fmt.Errorf("uncertainty %q is not supported", u)
		}
		if seen[u] {
			return fmt.Errorf("uncertainty %q is repeated", u)
		}
		seen[u] = true
	}
	if err := e.Source.Validate(); err != nil {
		return fmt.Errorf("source: %w", err)
	}
	if err := e.AppliedSettings.Validate(); err != nil {
		return fmt.Errorf("appliedSettings: %w", err)
	}
	if err := e.Envelope.Validate(); err != nil {
		return fmt.Errorf("envelope: %w", err)
	}
	if err := e.Observed.Validate(); err != nil {
		return fmt.Errorf("observed: %w", err)
	}
	if err := e.Qualification.Validate(); err != nil {
		return fmt.Errorf("qualification: %w", err)
	}
	if err := e.Provenance.Validate(); err != nil {
		return fmt.Errorf("provenance: %w", err)
	}
	if err := e.Freshness.Validate(); err != nil {
		return fmt.Errorf("freshness: %w", err)
	}
	if e.State != MemoryStateQualified {
		if e.Qualification.UpperBytes != nil {
			return fmt.Errorf("state %q must not carry a qualified upper bound", e.State)
		}
		return nil
	}
	return e.validateQualified()
}

// validateQualified requires every current binding, the complete measured
// envelope, positive provenance, no uncertainty and a fresh reading within the
// bound.
func (e MemoryEvidenceEntry) validateQualified() error {
	if len(e.Uncertainty) != 0 {
		return errors.New("qualified entry must not carry uncertainty")
	}
	for _, f := range []struct {
		name string
		v    *string
	}{
		{"source hostUuid", e.Source.HostUUID},
		{"source managerIncarnation", e.Source.ManagerIncarnation},
		{"source engineRunId", e.Source.EngineRunID},
		{"source launchDigest", e.Source.LaunchDigest},
		{"source modelDigest", e.Source.ModelDigest},
		{"appliedSettings epoch", e.AppliedSettings.Epoch},
		{"observed extractorRevision", e.Observed.ExtractorRevision},
		{"qualification id", e.Qualification.ID},
		{"qualification revision", e.Qualification.Revision},
		{"qualification artifactDigest", e.Qualification.ArtifactDigest},
		{"qualification procedureDigest", e.Qualification.ProcedureDigest},
		{"provenance procedureId", e.Provenance.ProcedureID},
		{"provenance procedureRevision", e.Provenance.ProcedureRevision},
		{"provenance hardwareFingerprint", e.Provenance.HardwareFingerprint},
		{"provenance runtimeRevision", e.Provenance.RuntimeRevision},
		{"provenance hardwareDescription", e.Provenance.HardwareDescription},
		{"provenance driverRevision", e.Provenance.DriverRevision},
	} {
		if f.v == nil {
			return fmt.Errorf("%s is required for a qualified entry", f.name)
		}
	}
	if e.AppliedSettings.Revision == nil {
		return errors.New("appliedSettings revision is required for a qualified entry")
	}
	if e.Envelope.Domain == nil {
		return errors.New("envelope domain is required for a qualified entry")
	}
	if e.Envelope.EffectiveContextTokens == nil {
		return errors.New("envelope effectiveContextTokens is required for a qualified entry")
	}
	if e.Envelope.ConcurrentExecutions == nil {
		return errors.New("envelope concurrentExecutions is required for a qualified entry")
	}
	if e.Envelope.OffloadMode == nil {
		return errors.New("envelope offloadMode is required for a qualified entry")
	}
	if e.Observed.AllocatedBytes == nil {
		return errors.New("observed allocatedBytes is required for a qualified entry")
	}
	if e.Qualification.UpperBytes == nil {
		return errors.New("qualification upperBytes is required for a qualified entry")
	}
	if e.Provenance.Scope == nil || *e.Provenance.Scope != MemoryQualificationScopeOperatorAttestedCurrentRun {
		return errors.New("provenance scope must be operator-attested-current-run for a qualified entry")
	}
	if e.Freshness.SampleAgeMs == nil {
		return errors.New("freshness sampleAgeMs is required for a qualified entry")
	}
	if *e.Freshness.SampleAgeMs > MemoryReadingValidForMs {
		return fmt.Errorf("sampleAgeMs %d is outside 0..%d for a qualified entry", *e.Freshness.SampleAgeMs, MemoryReadingValidForMs)
	}
	if e.Freshness.QualificationAgeMs == nil {
		return errors.New("freshness qualificationAgeMs is required for a qualified entry")
	}
	if e.Freshness.Generation == nil {
		return errors.New("freshness observationGeneration is required for a qualified entry")
	}
	if *e.Observed.AllocatedBytes > *e.Qualification.UpperBytes {
		return fmt.Errorf("observed allocation %d exceeds qualified upper %d", *e.Observed.AllocatedBytes, *e.Qualification.UpperBytes)
	}
	return nil
}

// Validate checks every present source member.
func (s MemorySource) Validate() error {
	for _, f := range []struct {
		name string
		v    *string
	}{
		{"hostUuid", s.HostUUID},
		{"managerIncarnation", s.ManagerIncarnation},
		{"engineRunId", s.EngineRunID},
	} {
		if f.v != nil {
			if err := checkID(f.name, *f.v); err != nil {
				return err
			}
		}
	}
	for _, f := range []struct {
		name string
		v    *string
	}{
		{"launchDigest", s.LaunchDigest},
		{"modelDigest", s.ModelDigest},
	} {
		if f.v != nil {
			if err := checkDigest(f.name, *f.v); err != nil {
				return err
			}
		}
	}
	return nil
}

// Validate checks every present applied-settings member.
func (a MemoryAppliedSettings) Validate() error {
	if a.Epoch != nil {
		if err := checkID("epoch", *a.Epoch); err != nil {
			return err
		}
	}
	if a.Revision != nil && *a.Revision <= 0 {
		return errors.New("revision must be positive")
	}
	return nil
}

// Validate checks every present envelope member.
func (v MemoryEnvelope) Validate() error {
	if v.Domain != nil {
		switch *v.Domain {
		case MemoryDomainDedicatedDevice, MemoryDomainHostPhysical, MemoryDomainUnifiedPhysical:
		default:
			return fmt.Errorf("domain %q is not supported", *v.Domain)
		}
	}
	if v.EffectiveContextTokens != nil && (*v.EffectiveContextTokens < 1 || *v.EffectiveContextTokens > MaxTokens) {
		return fmt.Errorf("effectiveContextTokens %d is outside 1..%d", *v.EffectiveContextTokens, MaxTokens)
	}
	if v.ConcurrentExecutions != nil && (*v.ConcurrentExecutions < 1 || *v.ConcurrentExecutions > MaxMemoryConcurrentExecutions) {
		return fmt.Errorf("concurrentExecutions %d is outside 1..%d", *v.ConcurrentExecutions, MaxMemoryConcurrentExecutions)
	}
	if v.OffloadMode != nil {
		if err := checkText("offloadMode", *v.OffloadMode, MaxIDBytes, true); err != nil {
			return err
		}
	}
	return nil
}

// Validate checks every present observation member.
func (o MemoryObservation) Validate() error {
	if o.AllocatedBytes != nil && (*o.AllocatedBytes < 0 || *o.AllocatedBytes > MaxMemoryBytes) {
		return fmt.Errorf("allocatedBytes %d is outside 0..%d", *o.AllocatedBytes, MaxMemoryBytes)
	}
	if o.ExtractorRevision != nil {
		if err := checkText("extractorRevision", *o.ExtractorRevision, MaxIDBytes, true); err != nil {
			return err
		}
	}
	return nil
}

// Validate checks every present qualification member.
func (q MemoryQualification) Validate() error {
	if q.UpperBytes != nil && (*q.UpperBytes < 1 || *q.UpperBytes > MaxMemoryBytes) {
		return fmt.Errorf("upperBytes %d is outside 1..%d", *q.UpperBytes, MaxMemoryBytes)
	}
	if q.ID != nil {
		if err := checkID("id", *q.ID); err != nil {
			return err
		}
	}
	if q.Revision != nil {
		if err := checkCanonicalPositiveDecimal("revision", *q.Revision); err != nil {
			return err
		}
	}
	for _, f := range []struct {
		name string
		v    *string
	}{
		{"artifactDigest", q.ArtifactDigest},
		{"procedureDigest", q.ProcedureDigest},
	} {
		if f.v != nil {
			if err := checkDigest(f.name, *f.v); err != nil {
				return err
			}
		}
	}
	return nil
}

// Validate checks every present provenance member.
func (p MemoryProvenance) Validate() error {
	if p.ProcedureID != nil {
		if err := checkID("procedureId", *p.ProcedureID); err != nil {
			return err
		}
	}
	if p.ProcedureRevision != nil {
		if err := checkCanonicalPositiveDecimal("procedureRevision", *p.ProcedureRevision); err != nil {
			return err
		}
	}
	if p.HardwareFingerprint != nil {
		if err := checkID("hardwareFingerprint", *p.HardwareFingerprint); err != nil {
			return err
		}
	}
	for _, f := range []struct {
		name string
		v    *string
	}{
		{"runtimeRevision", p.RuntimeRevision},
		{"hardwareDescription", p.HardwareDescription},
		{"driverRevision", p.DriverRevision},
	} {
		if f.v != nil {
			if err := checkText(f.name, *f.v, MaxModelBytes, true); err != nil {
				return err
			}
		}
	}
	if p.Scope != nil && *p.Scope != MemoryQualificationScopeOperatorAttestedCurrentRun {
		return fmt.Errorf("scope %q is not supported", *p.Scope)
	}
	return nil
}

// Validate checks the fixed reading window, both nonnegative ages and the
// observation generation. The reading-age upper bound is state-specific and is
// enforced by the entry.
func (f MemoryFreshness) Validate() error {
	if f.ReadingValidForMs != MemoryReadingValidForMs {
		return fmt.Errorf("readingValidForMs must be %d", MemoryReadingValidForMs)
	}
	if f.SampleAgeMs != nil && *f.SampleAgeMs < 0 {
		return fmt.Errorf("sampleAgeMs %d must be nonnegative", *f.SampleAgeMs)
	}
	if f.QualificationAgeMs != nil && *f.QualificationAgeMs < 0 {
		return fmt.Errorf("qualificationAgeMs %d must be nonnegative", *f.QualificationAgeMs)
	}
	if f.Generation != nil && *f.Generation == 0 {
		return errors.New("observationGeneration must be positive")
	}
	return nil
}

// EncodeMemoryEvidence validates m and encodes it within MaxMemoryEvidenceBytes.
func EncodeMemoryEvidence(m MemoryEvidence) ([]byte, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	data, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	if len(data) > MaxMemoryEvidenceBytes {
		return nil, fmt.Errorf("memory evidence is %d bytes, over %d", len(data), MaxMemoryEvidenceBytes)
	}
	return data, nil
}

// DecodeMemoryEvidence strictly decodes and validates one MemoryEvidence. It
// refuses a payload over MaxMemoryEvidenceBytes before parsing, duplicate,
// case-variant or unknown keys at every object depth, required nulls and
// trailing data.
func DecodeMemoryEvidence(data []byte) (MemoryEvidence, error) {
	if err := checkMemoryPayload(data); err != nil {
		return MemoryEvidence{}, err
	}
	var m MemoryEvidence
	if err := json.Unmarshal(data, &m); err != nil {
		return MemoryEvidence{}, fmt.Errorf("decode memory evidence: %w", err)
	}
	return m, nil
}

func (m *MemoryEvidence) UnmarshalJSON(data []byte) error {
	if err := checkMemoryPayload(data); err != nil {
		return err
	}
	keys := []string{"version", "entries", "omittedEntries"}
	if err := checkMemoryFields(data, keys, keys, keys); err != nil {
		return err
	}
	type wire MemoryEvidence
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*m = MemoryEvidence(next)
	return m.Validate()
}

func (e *MemoryEvidenceEntry) UnmarshalJSON(data []byte) error {
	keys := []string{
		"engine", "model", "state", "uncertainty", "source", "appliedSettings",
		"envelope", "observed", "qualification", "provenance", "freshness",
	}
	if err := checkMemoryFields(data, keys, keys, keys); err != nil {
		return err
	}
	type wire MemoryEvidenceEntry
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*e = MemoryEvidenceEntry(next)
	return e.Validate()
}

func (s *MemorySource) UnmarshalJSON(data []byte) error {
	keys := []string{"hostUuid", "managerIncarnation", "engineRunId", "launchDigest", "modelDigest"}
	if err := checkMemoryFields(data, keys, keys, nil); err != nil {
		return err
	}
	type wire MemorySource
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*s = MemorySource(next)
	return s.Validate()
}

func (a *MemoryAppliedSettings) UnmarshalJSON(data []byte) error {
	keys := []string{"epoch", "revision"}
	if err := checkMemoryFields(data, keys, keys, nil); err != nil {
		return err
	}
	type wire MemoryAppliedSettings
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*a = MemoryAppliedSettings(next)
	return a.Validate()
}

func (v *MemoryEnvelope) UnmarshalJSON(data []byte) error {
	keys := []string{"domain", "effectiveContextTokens", "concurrentExecutions", "offloadMode"}
	if err := checkMemoryFields(data, keys, keys, nil); err != nil {
		return err
	}
	type wire MemoryEnvelope
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*v = MemoryEnvelope(next)
	return v.Validate()
}

func (o *MemoryObservation) UnmarshalJSON(data []byte) error {
	keys := []string{"allocatedBytes", "extractorRevision"}
	if err := checkMemoryFields(data, keys, keys, nil); err != nil {
		return err
	}
	type wire MemoryObservation
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*o = MemoryObservation(next)
	return o.Validate()
}

func (q *MemoryQualification) UnmarshalJSON(data []byte) error {
	keys := []string{"upperBytes", "id", "revision", "artifactDigest", "procedureDigest"}
	if err := checkMemoryFields(data, keys, keys, nil); err != nil {
		return err
	}
	type wire MemoryQualification
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*q = MemoryQualification(next)
	return q.Validate()
}

func (p *MemoryProvenance) UnmarshalJSON(data []byte) error {
	keys := []string{
		"procedureId", "procedureRevision", "hardwareFingerprint",
		"runtimeRevision", "hardwareDescription", "driverRevision", "scope",
	}
	if err := checkMemoryFields(data, keys, keys, nil); err != nil {
		return err
	}
	type wire MemoryProvenance
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*p = MemoryProvenance(next)
	return p.Validate()
}

func (f *MemoryFreshness) UnmarshalJSON(data []byte) error {
	keys := []string{"sampleAgeMs", "qualificationAgeMs", "observationGeneration", "readingValidForMs"}
	if err := checkMemoryFields(data, keys, keys, []string{"readingValidForMs"}); err != nil {
		return err
	}
	type wire MemoryFreshness
	var next wire
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	*f = MemoryFreshness(next)
	return f.Validate()
}

func checkMemoryPayload(data []byte) error {
	if len(data) > MaxMemoryEvidenceBytes {
		return fmt.Errorf("memory evidence is %d bytes, over %d", len(data), MaxMemoryEvidenceBytes)
	}
	if !utf8.Valid(data) {
		return errors.New("memory evidence must be valid UTF-8")
	}
	return nil
}

// checkMemoryFields rejects unknown keys and missing required keys. Allowed,
// present and non-null are separate so a nullable member is still structurally
// required. It inspects keys before decoding so a duplicate cannot collapse.
func checkMemoryFields(data []byte, allowed, present, nonNull []string) error {
	fields, err := memoryObjectFields(data)
	if err != nil {
		return err
	}
	for key := range fields {
		if !slices.Contains(allowed, key) {
			return fmt.Errorf("unknown field %q", key)
		}
	}
	for _, key := range present {
		if _, ok := fields[key]; !ok {
			return fmt.Errorf("%s is required", key)
		}
	}
	for _, key := range nonNull {
		value, ok := fields[key]
		if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("%s is required and must not be null", key)
		}
	}
	return nil
}

// memoryObjectFields reads one JSON object, rejecting duplicate keys including
// case variants, and requires the object to consume the whole input.
func memoryObjectFields(raw []byte) (map[string]json.RawMessage, error) {
	notObject := errors.New("memory evidence must be a JSON object")
	dec := json.NewDecoder(bytes.NewReader(raw))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil, notObject
	}
	fields := map[string]json.RawMessage{}
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, notObject
		}
		key, err := memoryTokenString(tok)
		if err != nil {
			return nil, notObject
		}
		for seen := range fields {
			if strings.EqualFold(seen, key) {
				return nil, fmt.Errorf("duplicate field %q", key)
			}
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return nil, notObject
		}
		fields[key] = value
	}
	if _, err := dec.Token(); err != nil {
		return nil, notObject
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, notObject
	}
	return fields, nil
}

// memoryTokenString converts an object key token without a type assertion.
func memoryTokenString(tok json.Token) (string, error) {
	raw, err := json.Marshal(tok)
	if err != nil {
		return "", err
	}
	var key string
	if err := json.Unmarshal(raw, &key); err != nil {
		return "", err
	}
	return key, nil
}

func checkDigest(name, v string) error {
	if len(v) != 64 {
		return fmt.Errorf("%s must be 64 lowercase SHA-256 hex characters", name)
	}
	for _, r := range v {
		if !(r >= '0' && r <= '9') && !(r >= 'a' && r <= 'f') {
			return fmt.Errorf("%s must be 64 lowercase SHA-256 hex characters", name)
		}
	}
	return nil
}

func checkCanonicalPositiveDecimal(name, v string) error {
	if v == "" || v[0] == '0' {
		return fmt.Errorf("%s must be a canonical positive decimal string", name)
	}
	for _, r := range v {
		if r < '0' || r > '9' {
			return fmt.Errorf("%s must be a canonical positive decimal string", name)
		}
	}
	n, err := strconv.ParseUint(v, 10, 64)
	if err != nil || strconv.FormatUint(n, 10) != v {
		return fmt.Errorf("%s must be a canonical positive decimal string", name)
	}
	return nil
}

func validMemoryUncertainty(u MemoryUncertainty) bool {
	switch u {
	case MemoryUncertaintyUnsupported,
		MemoryUncertaintyMissingReading,
		MemoryUncertaintyIdentityUnknown,
		MemoryUncertaintyIdentityChanged,
		MemoryUncertaintySettingsUnapplied,
		MemoryUncertaintyContextUnknown,
		MemoryUncertaintyConcurrencyUnknown,
		MemoryUncertaintyHardwareUnqualified,
		MemoryUncertaintyQualificationMissing,
		MemoryUncertaintyQualificationMismatch,
		MemoryUncertaintyClockUncertain,
		MemoryUncertaintyStale,
		MemoryUncertaintyBoundExceeded:
		return true
	}
	return false
}
