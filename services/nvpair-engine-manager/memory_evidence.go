// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// memory_evidence.go is the engine-manager half of the local memory-evidence
// contract: a bounded, source-bound producer that reads the fixed, read-only
// operator qualification document and reports one entry per documented model.
// No engine has a supported loaded-memory extractor, so every entry is
// unsupported with no observed allocation and the qualified upper stays nil.
// The producer never queries an engine, starts a process, or changes admission.
// A qualification would be scoped to the current manager incarnation, engine
// run, launch, model content and fully applied settings.

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"nvpair-shared/engines"
	"nvpair-shared/enginesettings"
	"nvpair-shared/performancewire"
)

const (
	// memoryQualificationFile is the one fixed, read-only qualification
	// document under Executor.baseDir. It is absent by default and has no
	// environment or alternate-path fallback.
	memoryQualificationFile = "memory-qualification.json"
	// maxMemoryQualificationBytes bounds one qualification document.
	maxMemoryQualificationBytes = 256 << 10
	// maxMemoryQualificationRecords bounds the records in one document.
	maxMemoryQualificationRecords = 128
)

// memoryEvidenceProducer is the zero-value-safe producer state. Its zero value
// performs no I/O and starts no goroutine. Tests inject now and hostUUID; the
// later Manager hook sets hostUUID from clustertrust.Mesh.NodeUUID, and a nil
// provider means the local host identity is unknown.
type memoryEvidenceProducer struct {
	mu         sync.Mutex
	now        func() time.Time
	hostUUID   func() string
	generation uint64
	authority  memoryEvidenceAuthority
	artifact   *memoryQualificationState
}

// memoryEvidenceAuthority is the one document-level high-water record retained
// for the manager lifetime. It is never keyed per model and never grows.
type memoryEvidenceAuthority struct {
	present        bool
	revision       string
	revisionNumber uint64
	digest         string
	withdrawn      bool
	clockUncertain bool
}

// memoryQualificationState is the currently accepted document plus its age
// anchor. A duplicate read of the same bytes preserves both.
type memoryQualificationState struct {
	digest         string
	revision       string
	revisionNumber uint64
	records        []memoryQualificationRecord
	acceptedAt     time.Time
	baseAgeMs      int64
	lastAgeMs      int64
	lastWallMs     int64
}

// memoryEvidenceCapture is taken before the existing loaded-model read. The
// model owner passes it to memoryEvidenceEntries afterwards, which rechecks this
// same source before publishing anything. startedAt is a
// conservative monotonic age anchor: source capture, the whole sweep's I/O and
// buffering, and verification all count toward sample age.
type memoryEvidenceCapture struct {
	generation             uint64
	startedAt              time.Time
	source                 memoryEvidenceSourceSnapshot
	artifactDigest         string
	retainedArtifactDigest string
}

type memoryEvidenceSourceSnapshot struct {
	hostUUID           string
	managerIncarnation string
	engines            map[string]engineModelIdentity
	epochs             map[string]uint64
	settings           map[string]memoryAppliedSettings
	// naming comes from the live registry descriptor, not from the artifact.
	// It decides canonical model names.
	naming map[string]string
}

type memoryAppliedSettings struct {
	epoch    string
	revision int64
	ok       bool
}

// memoryQualificationArtifact is the strict local document. It deliberately has
// no artifact digest member: ArtifactDigest in published evidence is the
// SHA-256 of these bytes, so the document never claims a hash of itself.
type memoryQualificationArtifact struct {
	Version    *int                         `json:"version"`
	Revision   *string                      `json:"revision"`
	MeasuredAt *int64                       `json:"measuredAt"`
	Records    *[]memoryQualificationRecord `json:"records"`
}

type memoryQualificationRecord struct {
	Source        *memoryArtifactSource        `json:"source"`
	Config        *memoryArtifactConfig        `json:"config"`
	Envelope      *memoryArtifactEnvelope      `json:"envelope"`
	Qualification *memoryArtifactQualification `json:"qualification"`
	Procedure     *memoryArtifactProcedure     `json:"procedure"`
	Hardware      *memoryArtifactHardware      `json:"hardware"`
	Runtime       *memoryArtifactRuntime       `json:"runtime"`
	Driver        *memoryArtifactDriver        `json:"driver"`
}

type memoryArtifactSource struct {
	HostUUID           *string `json:"hostUuid"`
	ManagerIncarnation *string `json:"managerIncarnation"`
	Engine             *string `json:"engine"`
	EngineRunID        *string `json:"engineRunId"`
	LaunchDigest       *string `json:"launchDigest"`
	Model              *string `json:"model"`
	ModelDigest        *string `json:"modelDigest"`
}

type memoryArtifactConfig struct {
	Epoch    *string `json:"epoch"`
	Revision *int64  `json:"revision"`
}

type memoryArtifactEnvelope struct {
	Domain                 *performancewire.MemoryDomain `json:"domain"`
	EffectiveContextTokens *int64                        `json:"effectiveContextTokens"`
	ConcurrentExecutions   *int64                        `json:"concurrentExecutions"`
	OffloadMode            *string                       `json:"offloadMode"`
	ExtractorRevision      *string                       `json:"extractorRevision"`
}

type memoryArtifactQualification struct {
	ID         *string                                   `json:"id"`
	UpperBytes *int64                                    `json:"upperBytes"`
	Scope      *performancewire.MemoryQualificationScope `json:"scope"`
}

type memoryArtifactProcedure struct {
	ID       *string `json:"id"`
	Revision *string `json:"revision"`
	Digest   *string `json:"digest"`
}

type memoryArtifactHardware struct {
	Fingerprint *string `json:"fingerprint"`
	Description *string `json:"description"`
}

type memoryArtifactRuntime struct {
	Revision *string `json:"revision"`
}

type memoryArtifactDriver struct {
	Revision *string `json:"revision"`
}

// captureMemoryEvidence takes the immutable pre-read source snapshot and time
// anchor before the whole model sweep begins engine I/O. Pass the same capture
// to memoryEvidenceEntries for each existing loaded_models response.
func (e *Executor) captureMemoryEvidence() memoryEvidenceCapture {
	p := &e.memoryEvidence
	provider := e.memoryEvidenceHostProvider()
	p.mu.Lock()
	p.generation++
	generation := p.generation
	startedAt := p.clock()
	retainedArtifactDigest := ""
	if p.artifact != nil {
		retainedArtifactDigest = p.artifact.digest
	}
	p.mu.Unlock()
	source := e.memoryEvidenceSource(provider)
	artifactDigest := ""
	// Accept authority before engine I/O so even the first contradictory
	// observation can withdraw it if the document disappears during the read.
	// A document introduced during engine I/O waits for the next capture.
	if data, present, err := readBoundedMemoryQualification(filepath.Join(e.baseDir, memoryQualificationFile)); present && err == nil {
		sum := sha256.Sum256(data)
		artifactDigest = hex.EncodeToString(sum[:])
		if document, err := decodeMemoryQualification(data); err == nil && len(*document.Records) > 0 {
			current := e.memoryEvidenceSource(e.memoryEvidenceHostProvider())
			if sameMemoryEvidenceSource(source, current) {
				p.mu.Lock()
				p.acceptMemoryQualification(document, artifactDigest, recordEngine((*document.Records)[0]), source, p.clock())
				p.mu.Unlock()
			}
		}
	}
	return memoryEvidenceCapture{generation: generation, startedAt: startedAt, source: source, artifactDigest: artifactDigest, retainedArtifactDigest: retainedArtifactDigest}
}

// memoryEvidenceEntries is the post-read verifier/producer. It rechecks the
// capture's source, reads and validates the bounded fixed document outside the
// state lock, and returns at most one entry per documented model for the engine.
// No engine has a supported loaded-memory extractor, so the engine's loaded_models
// response is not read and every entry is unsupported with no observed
// allocation. This function performs no query. A memory failure never removes
// model names and never widens the bound.
func (e *Executor) memoryEvidenceEntries(capture memoryEvidenceCapture, engine string) []performancewire.MemoryEvidenceEntry {
	path := filepath.Join(e.baseDir, memoryQualificationFile)
	data, present, readErr := readBoundedMemoryQualification(path)

	documentOK := false
	if readErr == nil && present {
		_, err := decodeMemoryQualification(data)
		if err == nil {
			documentOK = true
		}
	}
	digest := ""
	if present && readErr == nil {
		sum := sha256.Sum256(data)
		digest = hex.EncodeToString(sum[:])
	}
	current := e.memoryEvidenceSource(e.memoryEvidenceHostProvider())
	sourceUnchanged := sameMemoryEvidenceSource(capture.source, current)

	p := &e.memoryEvidence
	p.mu.Lock()
	defer p.mu.Unlock()
	now := p.clock()

	if capture.startedAt.IsZero() || now.Before(capture.startedAt) || now.UnixMilli() < capture.startedAt.UnixMilli() {
		p.authority.clockUncertain = true
	}

	state := p.artifact
	records := []memoryQualificationRecord(nil)
	if state != nil {
		records = state.records
	}
	artifactCurrent := state != nil && present && readErr == nil && documentOK && state.digest == digest
	qualificationAge := (*int64)(nil)
	if state != nil {
		if age, ok := p.qualificationAge(state, now); ok {
			qualificationAge = &age
		}
	}

	entries := make([]performancewire.MemoryEvidenceEntry, 0, len(records))
	for _, record := range records {
		if record.Source != nil && record.Source.Engine != nil && *record.Source.Engine == engine {
			entries = append(entries, p.recordEntry(record, capture, engine, sourceUnchanged, artifactCurrent, qualificationAge))
		}
	}
	if len(entries) == 0 {
		return nil
	}
	return entries
}

// finalizeMemoryEvidence is the whole-sweep fence for the model publication
// owner. Call it once, after every engine's raw response for one capture has
// been passed to memoryEvidenceEntries and before freezing the payload. A
// withdrawal latched by any engine, a changed source/config/content, or a
// newer capture invalidates every positive upper in the combined projection,
// including entries that were built earlier in the same sweep. Sample ages and
// source bindings are fenced as well. It reads the current source outside the
// producer lock and performs no I/O while holding it.
func (e *Executor) finalizeMemoryEvidence(capture memoryEvidenceCapture, entries []performancewire.MemoryEvidenceEntry) []performancewire.MemoryEvidenceEntry {
	if len(entries) == 0 {
		return entries
	}
	data, present, err := readBoundedMemoryQualification(filepath.Join(e.baseDir, memoryQualificationFile))
	sum := sha256.Sum256(data)
	digest := hex.EncodeToString(sum[:])
	current := e.memoryEvidenceSource(e.memoryEvidenceHostProvider())
	sourceUnchanged := sameMemoryEvidenceSource(capture.source, current)

	p := &e.memoryEvidence
	p.mu.Lock()
	stale := capture.generation < p.generation
	withdrawn := p.authority.withdrawn
	clockUncertain := p.authority.clockUncertain
	available := p.artifact != nil && present && err == nil && p.artifact.digest == digest
	qualificationAge := (*int64)(nil)
	now := p.clock()
	if capture.startedAt.IsZero() || now.Before(capture.startedAt) || now.UnixMilli() < capture.startedAt.UnixMilli() {
		p.authority.clockUncertain = true
	}
	if p.artifact != nil {
		if age, ok := p.qualificationAge(p.artifact, now); ok {
			qualificationAge = &age
		}
	}
	clockUncertain = p.authority.clockUncertain
	p.mu.Unlock()

	out := make([]performancewire.MemoryEvidenceEntry, len(entries))
	copy(out, entries)
	for i := range out {
		entryCurrent := available && (out[i].Qualification.UpperBytes == nil ||
			(out[i].Qualification.ArtifactDigest != nil && *out[i].Qualification.ArtifactDigest == digest))
		if sourceUnchanged && !stale {
			if out[i].Freshness.Generation != nil && *out[i].Freshness.Generation == capture.generation {
				age := now.Sub(capture.startedAt).Milliseconds()
				if age >= 0 && !clockUncertain {
					out[i].Freshness.SampleAgeMs = &age
				} else {
					out[i].Freshness.SampleAgeMs = nil
				}
			}
		}
		if out[i].Qualification.UpperBytes != nil && entryCurrent {
			out[i].Freshness.QualificationAgeMs = qualificationAge
		}
		readingFresh := out[i].Freshness.SampleAgeMs != nil && *out[i].Freshness.SampleAgeMs <= performancewire.MemoryReadingValidForMs
		if out[i].Freshness.SampleAgeMs != nil && *out[i].Freshness.SampleAgeMs > performancewire.MemoryReadingValidForMs {
			out[i].Uncertainty = appendMemoryUncertainty(out[i].Uncertainty, performancewire.MemoryUncertaintyStale)
		}
		if sourceUnchanged && !stale && !withdrawn && !clockUncertain && entryCurrent &&
			(out[i].Qualification.UpperBytes == nil || readingFresh) {
			continue
		}
		if !sourceUnchanged {
			out[i].Uncertainty = appendMemoryUncertainty(out[i].Uncertainty, performancewire.MemoryUncertaintyIdentityChanged)
			// A response spanning a source change cannot be attributed to
			// either run, including advisory observations.
			out[i].Source = performancewire.MemorySource{}
			out[i].AppliedSettings = performancewire.MemoryAppliedSettings{}
			out[i].Provenance = performancewire.MemoryProvenance{}
			out[i].Envelope = performancewire.MemoryEnvelope{}
			out[i].Freshness.SampleAgeMs = nil
			out[i].Freshness.Generation = nil
		}
		if stale {
			out[i].Uncertainty = appendMemoryUncertainty(out[i].Uncertainty, performancewire.MemoryUncertaintyStale)
		}
		if withdrawn {
			out[i].Uncertainty = appendMemoryUncertainty(out[i].Uncertainty, performancewire.MemoryUncertaintyBoundExceeded)
		}
		if clockUncertain {
			out[i].Uncertainty = appendMemoryUncertainty(out[i].Uncertainty, performancewire.MemoryUncertaintyClockUncertain)
			out[i].Freshness.SampleAgeMs = nil
			out[i].Freshness.QualificationAgeMs = nil
		}
		if !entryCurrent {
			out[i].Uncertainty = appendMemoryUncertainty(out[i].Uncertainty, performancewire.MemoryUncertaintyQualificationMissing)
		}
		if out[i].Qualification.UpperBytes != nil && !readingFresh {
			out[i].Uncertainty = appendMemoryUncertainty(out[i].Uncertainty, performancewire.MemoryUncertaintyStale)
		}
		out[i].Qualification.UpperBytes = nil
		if out[i].Observed.AllocatedBytes != nil {
			out[i].State = performancewire.MemoryStateAdvisory
		} else {
			out[i].State = performancewire.MemoryStateUnknown
		}
	}
	return out
}

// memoryEvidenceHostProvider returns the installed host UUID provider under the
// short state lock, then calls it outside the lock.
func (e *Executor) memoryEvidenceHostProvider() func() string {
	e.memoryEvidence.mu.Lock()
	defer e.memoryEvidence.mu.Unlock()
	return e.memoryEvidence.hostUUID
}

func (p *memoryEvidenceProducer) clock() time.Time {
	if p.now != nil {
		return p.now()
	}
	return time.Now()
}

func (e *Executor) memoryEvidenceSource(provider func() string) memoryEvidenceSourceSnapshot {
	host := ""
	if provider != nil {
		host = provider()
	}
	identity := e.modelIdentity()
	naming := map[string]string{}
	for _, name := range e.reg.Names() {
		manifest, ok := e.reg.Get(name)
		if !ok || manifest == nil {
			continue
		}
		naming[name] = manifest.ModelNamingOrDefault()
	}
	return memoryEvidenceSourceSnapshot{
		hostUUID:           host,
		managerIncarnation: e.switchIncarnation,
		engines:            identity.Engines,
		epochs:             identity.epochs,
		settings:           e.memoryAppliedSettings(),
		naming:             naming,
	}
}

// memoryAppliedSettings reads only the atomic Subscribe baseline and
// unsubscribes immediately. No broker fingerprint is copied. A concurrent
// subscribe failure is unknown.
func (e *Executor) memoryAppliedSettings() map[string]memoryAppliedSettings {
	ch, cancel, ok := e.settingsHub.Subscribe()
	if !ok {
		return nil
	}
	var baseline []enginesettings.Snapshot
	select {
	case baseline = <-ch:
	default:
	}
	cancel()
	out := make(map[string]memoryAppliedSettings, len(baseline))
	for _, snapshot := range baseline {
		if snapshot.Engine == "" || snapshot.Epoch == "" || snapshot.Revision == 0 ||
			snapshot.AppliedRevision != snapshot.Revision ||
			snapshot.Phase == "applying" || snapshot.Phase == "failed" ||
			snapshot.Revision > math.MaxInt64 {
			out[snapshot.Engine] = memoryAppliedSettings{}
			continue
		}
		out[snapshot.Engine] = memoryAppliedSettings{epoch: snapshot.Epoch, revision: int64(snapshot.Revision), ok: true}
	}
	return out
}

func sameMemoryEvidenceSource(a, b memoryEvidenceSourceSnapshot) bool {
	if a.hostUUID != b.hostUUID || a.managerIncarnation != b.managerIncarnation {
		return false
	}
	if len(a.engines) != len(b.engines) || len(a.settings) != len(b.settings) {
		return false
	}
	if len(a.epochs) != len(b.epochs) {
		return false
	}
	for name, epoch := range a.epochs {
		if b.epochs[name] != epoch {
			return false
		}
	}
	for name, id := range a.engines {
		other, ok := b.engines[name]
		if !ok || id.EngineRunID != other.EngineRunID || id.LaunchDigest != other.LaunchDigest || a.epochs[name] != b.epochs[name] {
			return false
		}
		if len(id.ModelDigests) != len(other.ModelDigests) {
			return false
		}
		for model, digest := range id.ModelDigests {
			if other.ModelDigests[model] != digest {
				return false
			}
		}
	}
	for name, applied := range a.settings {
		if other, ok := b.settings[name]; !ok || other != applied {
			return false
		}
	}
	if len(a.naming) != len(b.naming) {
		return false
	}
	for name, naming := range a.naming {
		if b.naming[name] != naming {
			return false
		}
	}
	return true
}

func readBoundedMemoryQualification(path string) (data []byte, present bool, err error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, err
	}
	defer f.Close()
	limited := io.LimitReader(f, maxMemoryQualificationBytes+1)
	data, err = io.ReadAll(limited)
	if err != nil {
		return nil, true, err
	}
	if len(data) > maxMemoryQualificationBytes {
		return nil, true, errors.New("memory qualification document is over its byte limit")
	}
	return data, true, nil
}

func decodeMemoryQualification(data []byte) (memoryQualificationArtifact, error) {
	if len(data) > maxMemoryQualificationBytes {
		return memoryQualificationArtifact{}, errors.New("memory qualification document is over its byte limit")
	}
	if !utf8.Valid(data) {
		return memoryQualificationArtifact{}, errors.New("memory qualification document must be valid UTF-8")
	}
	if err := checkStrictJSONStructure(data); err != nil {
		return memoryQualificationArtifact{}, err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var document memoryQualificationArtifact
	if err := dec.Decode(&document); err != nil {
		return memoryQualificationArtifact{}, err
	}
	if err := dec.Decode(new(json.RawMessage)); err != io.EOF {
		return memoryQualificationArtifact{}, errors.New("memory qualification document has trailing data")
	}
	if err := validateMemoryQualification(document); err != nil {
		return memoryQualificationArtifact{}, err
	}
	return document, nil
}

// strictJSONSpec is the exact, case-sensitive shape of one artifact object or
// array element. A leaf has neither object nor array.
type strictJSONSpec struct {
	object map[string]*strictJSONSpec
	array  *strictJSONSpec
}

func memoryQualificationShape() strictJSONSpec {
	leaf := func() *strictJSONSpec { return &strictJSONSpec{} }
	object := func(fields ...string) *strictJSONSpec {
		spec := &strictJSONSpec{object: make(map[string]*strictJSONSpec, len(fields))}
		for _, field := range fields {
			spec.object[field] = leaf()
		}
		return spec
	}
	record := &strictJSONSpec{object: map[string]*strictJSONSpec{
		"source": object("hostUuid", "managerIncarnation", "engine", "engineRunId", "launchDigest", "model", "modelDigest"),
		"config": object("epoch", "revision"),
		"envelope": object("domain", "effectiveContextTokens", "concurrentExecutions",
			"offloadMode", "extractorRevision"),
		"qualification": object("id", "upperBytes", "scope"),
		"procedure":     object("id", "revision", "digest"),
		"hardware":      object("fingerprint", "description"),
		"runtime":       object("revision"),
		"driver":        object("revision"),
	}}
	return strictJSONSpec{object: map[string]*strictJSONSpec{
		"version": leaf(), "revision": leaf(), "measuredAt": leaf(),
		"records": &strictJSONSpec{array: record},
	}}
}

// checkStrictJSONStructure rejects unknown, duplicate and case-variant keys at
// every object depth before a decoder can collapse or case-match them.
func checkStrictJSONStructure(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	spec := memoryQualificationShape()
	if err := strictJSONValue(dec, &spec); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("memory qualification document has trailing data")
	}
	return nil
}

func strictJSONValue(dec *json.Decoder, spec *strictJSONSpec) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	switch tok {
	case json.Delim('{'):
		if spec.object == nil {
			return errors.New("memory qualification document has an unexpected object")
		}
		seen := make(map[string]bool)
		for dec.More() {
			keyTok, err := dec.Token()
			if err != nil {
				return err
			}
			raw, err := json.Marshal(keyTok)
			if err != nil {
				return err
			}
			var key string
			if err := json.Unmarshal(raw, &key); err != nil {
				return err
			}
			for prior := range seen {
				if strings.EqualFold(prior, key) {
					return errors.New("memory qualification document has a duplicate field")
				}
			}
			seen[key] = true
			child, known := spec.object[key]
			if !known {
				return errors.New("memory qualification document has an unknown or case-variant field")
			}
			if err := strictJSONValue(dec, child); err != nil {
				return err
			}
		}
		_, err := dec.Token()
		return err
	case json.Delim('['):
		if spec.array == nil {
			return errors.New("memory qualification document has an unexpected array")
		}
		for dec.More() {
			if err := strictJSONValue(dec, spec.array); err != nil {
				return err
			}
		}
		_, err := dec.Token()
		return err
	}
	return nil
}

func validateMemoryQualification(document memoryQualificationArtifact) error {
	if document.Version == nil || *document.Version != performancewire.MemoryEvidenceVersion {
		return errors.New("memory qualification version must be 1")
	}
	if document.Revision == nil {
		return errors.New("memory qualification revision is required")
	}
	if err := (performancewire.MemoryQualification{Revision: document.Revision}).Validate(); err != nil {
		return err
	}
	if document.MeasuredAt == nil || *document.MeasuredAt <= 0 {
		return errors.New("memory qualification measuredAt must be positive wall milliseconds")
	}
	if document.Records == nil {
		return errors.New("memory qualification records is required")
	}
	records := *document.Records
	if len(records) > maxMemoryQualificationRecords {
		return errors.New("memory qualification has too many records")
	}
	seen := make(map[string]bool, len(records))
	for i := range records {
		if err := validateMemoryQualificationRecord(records[i]); err != nil {
			return err
		}
		source := records[i].Source
		domain := ""
		if records[i].Envelope.Domain != nil {
			domain = string(*records[i].Envelope.Domain)
		}
		key := *source.Engine + "\x00" + *source.Model + "\x00" + domain
		if seen[key] {
			return errors.New("memory qualification repeats an engine/model/domain record")
		}
		seen[key] = true
	}
	return nil
}

func validateMemoryQualificationRecord(record memoryQualificationRecord) error {
	if record.Source == nil || record.Config == nil || record.Envelope == nil ||
		record.Qualification == nil || record.Procedure == nil || record.Hardware == nil ||
		record.Runtime == nil || record.Driver == nil {
		return errors.New("memory qualification record group is required")
	}
	source := record.Source
	for _, field := range []*string{source.HostUUID, source.ManagerIncarnation, source.Engine, source.EngineRunID, source.LaunchDigest, source.Model, source.ModelDigest} {
		if field == nil {
			return errors.New("memory qualification source member is required")
		}
	}
	if err := (performancewire.MemorySource{
		HostUUID: source.HostUUID, ManagerIncarnation: source.ManagerIncarnation,
		EngineRunID: source.EngineRunID, LaunchDigest: source.LaunchDigest, ModelDigest: source.ModelDigest,
	}).Validate(); err != nil {
		return err
	}
	if record.Config.Epoch == nil || record.Config.Revision == nil || *record.Config.Revision <= 0 {
		return errors.New("memory qualification applied settings are required")
	}
	if err := (performancewire.MemoryAppliedSettings{Epoch: record.Config.Epoch, Revision: record.Config.Revision}).Validate(); err != nil {
		return err
	}
	envelope := record.Envelope
	if envelope.Domain == nil || envelope.EffectiveContextTokens == nil || envelope.ConcurrentExecutions == nil ||
		envelope.OffloadMode == nil || envelope.ExtractorRevision == nil {
		return errors.New("memory qualification envelope member is required")
	}
	if err := (performancewire.MemoryEnvelope{
		Domain: envelope.Domain, EffectiveContextTokens: envelope.EffectiveContextTokens,
		ConcurrentExecutions: envelope.ConcurrentExecutions, OffloadMode: envelope.OffloadMode,
	}).Validate(); err != nil {
		return err
	}
	if err := (performancewire.MemoryObservation{ExtractorRevision: envelope.ExtractorRevision}).Validate(); err != nil {
		return err
	}
	qualification := record.Qualification
	if qualification.ID == nil || qualification.UpperBytes == nil || qualification.Scope == nil {
		return errors.New("memory qualification member is required")
	}
	procedure := record.Procedure
	if procedure.ID == nil || procedure.Revision == nil || procedure.Digest == nil ||
		record.Hardware.Fingerprint == nil || record.Hardware.Description == nil ||
		record.Runtime.Revision == nil || record.Driver.Revision == nil {
		return errors.New("memory qualification provenance member is required")
	}
	if err := (performancewire.MemoryQualification{
		UpperBytes: qualification.UpperBytes, ID: qualification.ID,
	}).Validate(); err != nil {
		return err
	}
	if *qualification.Scope != performancewire.MemoryQualificationScopeOperatorAttestedCurrentRun {
		return errors.New("memory qualification scope is not supported")
	}
	// Validate the complete metadata as an unknown-state template. It carries
	// no reading, age, generation or artifact digest, and is never emitted as
	// an observation; it exists only to reuse the wire field checks.
	template := performancewire.MemoryEvidenceEntry{
		Engine: *source.Engine, Model: *source.Model, State: performancewire.MemoryStateUnknown,
		Uncertainty: []performancewire.MemoryUncertainty{},
		Source: performancewire.MemorySource{
			HostUUID: source.HostUUID, ManagerIncarnation: source.ManagerIncarnation,
			EngineRunID: source.EngineRunID, LaunchDigest: source.LaunchDigest, ModelDigest: source.ModelDigest,
		},
		AppliedSettings: performancewire.MemoryAppliedSettings{Epoch: record.Config.Epoch, Revision: record.Config.Revision},
		Envelope: performancewire.MemoryEnvelope{
			Domain: envelope.Domain, EffectiveContextTokens: envelope.EffectiveContextTokens,
			ConcurrentExecutions: envelope.ConcurrentExecutions, OffloadMode: envelope.OffloadMode,
		},
		Observed:      performancewire.MemoryObservation{ExtractorRevision: envelope.ExtractorRevision},
		Qualification: performancewire.MemoryQualification{},
		Provenance: performancewire.MemoryProvenance{
			ProcedureID: procedure.ID, ProcedureRevision: procedure.Revision,
			HardwareFingerprint: record.Hardware.Fingerprint, RuntimeRevision: record.Runtime.Revision,
			HardwareDescription: record.Hardware.Description, DriverRevision: record.Driver.Revision,
			Scope: qualification.Scope,
		},
		Freshness: performancewire.MemoryFreshness{ReadingValidForMs: performancewire.MemoryReadingValidForMs},
	}
	return template.Validate()
}

func (p *memoryEvidenceProducer) acceptMemoryQualification(document memoryQualificationArtifact, digest, engine string, source memoryEvidenceSourceSnapshot, now time.Time) {
	if document.Revision == nil {
		return
	}
	revision := *document.Revision
	number, err := parseCanonicalRevision(revision)
	if err != nil {
		return
	}
	if p.authority.present {
		switch {
		case number < p.authority.revisionNumber:
			return
		case number == p.authority.revisionNumber:
			if digest != p.authority.digest {
				return
			}
			return
		}
	}
	if !documentMatchesSource(document, engine, source) {
		return
	}
	if memoryDocumentAliasesAmbiguous(document, source) {
		return
	}
	age := int64(0)
	uncertain := false
	nowMs := now.UnixMilli()
	if *document.MeasuredAt > nowMs {
		uncertain = true
	} else {
		age = nowMs - *document.MeasuredAt
	}
	p.authority = memoryEvidenceAuthority{
		present: true, revision: revision, revisionNumber: number, digest: digest,
		withdrawn: false, clockUncertain: uncertain,
	}
	p.artifact = &memoryQualificationState{
		digest: digest, revision: revision, revisionNumber: number,
		records: *document.Records, acceptedAt: now,
		baseAgeMs: age, lastAgeMs: age,
		lastWallMs: nowMs,
	}
}

func parseCanonicalRevision(revision string) (uint64, error) {
	if revision == "" || revision[0] == '0' {
		return 0, errors.New("revision is not canonical")
	}
	var value uint64
	for _, r := range revision {
		if r < '0' || r > '9' {
			return 0, errors.New("revision is not canonical")
		}
		digit := uint64(r - '0')
		if value > (math.MaxUint64-digit)/10 {
			return 0, errors.New("revision overflows")
		}
		value = value*10 + digit
	}
	return value, nil
}

func documentMatchesSource(document memoryQualificationArtifact, engine string, source memoryEvidenceSourceSnapshot) bool {
	if document.Records == nil || len(*document.Records) == 0 {
		return false
	}
	relevant := false
	for _, record := range *document.Records {
		if !recordMatchesCapture(record, source, recordEngine(record)) {
			return false
		}
		relevant = relevant || recordEngine(record) == engine
	}
	return relevant
}

func recordMatchesCapture(record memoryQualificationRecord, source memoryEvidenceSourceSnapshot, engine string) bool {
	if record.Source == nil || record.Config == nil || record.Source.Engine == nil || record.Source.Model == nil {
		return false
	}
	if *record.Source.Engine != engine {
		return false
	}
	identity, ok := source.engines[engine]
	if !ok {
		return false
	}
	applied, ok := source.settings[engine]
	if !ok || !applied.ok {
		return false
	}
	model := engines.NormalizeModelName(*record.Source.Model, source.naming[engine])
	digest, ok := memoryInventoryDigest(identity.ModelDigests, source.naming[engine], model)
	if !ok || digest != recordModelDigest(record) {
		return false
	}
	return record.Source.HostUUID != nil && *record.Source.HostUUID == source.hostUUID &&
		record.Source.ManagerIncarnation != nil && *record.Source.ManagerIncarnation == source.managerIncarnation &&
		record.Source.EngineRunID != nil && *record.Source.EngineRunID == identity.EngineRunID &&
		record.Source.LaunchDigest != nil && *record.Source.LaunchDigest == identity.LaunchDigest &&
		record.Config.Epoch != nil && *record.Config.Epoch == applied.epoch &&
		record.Config.Revision != nil && *record.Config.Revision == applied.revision
}

// memoryInventoryDigest resolves a canonical model name against the current
// inventory. Two raw inventory names that normalize to the same canonical name
// are ambiguous and resolve to nothing, as does a differing duplicate digest.
func memoryInventoryDigest(digests map[string]string, naming, model string) (string, bool) {
	found := ""
	matched := false
	for name, digest := range digests {
		if memoryCanonicalModel(name, naming) != model {
			continue
		}
		if matched {
			return "", false
		}
		found, matched = digest, true
	}
	return found, matched
}

func memoryCanonicalModel(model, naming string) string {
	return engines.NormalizeModelName(model, naming)
}

// memoryDocumentAliasesAmbiguous rejects a document whose records collide once
// the live descriptor's naming rule is applied, so no alias can widen a bound.
func memoryDocumentAliasesAmbiguous(document memoryQualificationArtifact, source memoryEvidenceSourceSnapshot) bool {
	if document.Records == nil {
		return false
	}
	seen := make(map[string]bool, len(*document.Records))
	for _, record := range *document.Records {
		if record.Source == nil || record.Source.Engine == nil || record.Source.Model == nil {
			return true
		}
		key := memoryReadingKey(*record.Source.Engine, memoryCanonicalModel(*record.Source.Model, source.naming[*record.Source.Engine]))
		if seen[key] {
			return true
		}
		seen[key] = true
	}
	return false
}

func (p *memoryEvidenceProducer) qualificationAge(state *memoryQualificationState, now time.Time) (int64, bool) {
	if p.authority.clockUncertain {
		return 0, false
	}
	if now.UnixMilli() < state.lastWallMs {
		p.authority.clockUncertain = true
		return 0, false
	}
	state.lastWallMs = now.UnixMilli()
	elapsed := now.Sub(state.acceptedAt).Milliseconds()
	if elapsed < 0 {
		p.authority.clockUncertain = true
		return 0, false
	}
	age := state.baseAgeMs + elapsed
	if age < state.lastAgeMs {
		age = state.lastAgeMs
	}
	state.lastAgeMs = age
	return age, true
}

// recordEntry projects one documented record. With no supported extractor the
// entry never carries an allocation or a qualified upper: it is unknown, marked
// unsupported with a missing reading, plus any identity, document or clock
// reasons.
func (p *memoryEvidenceProducer) recordEntry(record memoryQualificationRecord, capture memoryEvidenceCapture, engine string, sourceUnchanged, artifactCurrent bool, qualificationAge *int64) performancewire.MemoryEvidenceEntry {
	model := memoryCanonicalModel(recordModel(record), capture.source.naming[engine])
	reasons := make([]performancewire.MemoryUncertainty, 0, 6)
	entry := baseMemoryEvidenceEntry(record, capture, model)
	recordMatches := recordMatchesCapture(record, capture.source, engine)
	if !recordMatches {
		// An old operator document is never current-run provenance for a
		// replacement engine or model.
		entry.Provenance = performancewire.MemoryProvenance{}
		entry.Envelope = performancewire.MemoryEnvelope{}
	}
	entry.Freshness.QualificationAgeMs = qualificationAge
	if !sourceUnchanged {
		entry.Source = performancewire.MemorySource{}
		entry.AppliedSettings = performancewire.MemoryAppliedSettings{}
		entry.Provenance = performancewire.MemoryProvenance{}
		entry.Envelope = performancewire.MemoryEnvelope{}
		entry.Freshness.SampleAgeMs = nil
		entry.Freshness.Generation = nil
	}

	if !sourceUnchanged || !recordMatches {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyIdentityChanged)
	}
	if !artifactCurrent {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyQualificationMissing)
	}
	if p.authority.clockUncertain {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyClockUncertain)
	}
	if p.authority.withdrawn {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyBoundExceeded)
	}
	reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyUnsupported)
	reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyMissingReading)
	entry.State = performancewire.MemoryStateUnknown
	entry.Uncertainty = reasons
	return entry
}

func baseMemoryEvidenceEntry(record memoryQualificationRecord, capture memoryEvidenceCapture, model string) performancewire.MemoryEvidenceEntry {
	entry := performancewire.MemoryEvidenceEntry{
		Engine: recordEngine(record), Model: model, State: performancewire.MemoryStateUnknown,
		Uncertainty:     []performancewire.MemoryUncertainty{},
		Source:          performancewire.MemorySource{},
		AppliedSettings: performancewire.MemoryAppliedSettings{},
		Envelope:        performancewire.MemoryEnvelope{},
		Observed:        performancewire.MemoryObservation{},
		Qualification:   performancewire.MemoryQualification{},
		Provenance:      performancewire.MemoryProvenance{},
		Freshness: performancewire.MemoryFreshness{
			Generation: memoryValue(capture.generation), ReadingValidForMs: performancewire.MemoryReadingValidForMs,
		},
	}
	bindMemoryObservation(&entry, capture, recordEngine(record), model)
	if record.Envelope != nil {
		entry.Envelope = performancewire.MemoryEnvelope{
			Domain: record.Envelope.Domain, EffectiveContextTokens: record.Envelope.EffectiveContextTokens,
			ConcurrentExecutions: record.Envelope.ConcurrentExecutions, OffloadMode: record.Envelope.OffloadMode,
		}
	}
	if record.Qualification != nil {
		entry.Provenance.Scope = record.Qualification.Scope
	}
	if record.Procedure != nil {
		entry.Provenance.ProcedureID = record.Procedure.ID
		entry.Provenance.ProcedureRevision = record.Procedure.Revision
	}
	if record.Hardware != nil {
		entry.Provenance.HardwareFingerprint = record.Hardware.Fingerprint
		entry.Provenance.HardwareDescription = record.Hardware.Description
	}
	if record.Runtime != nil {
		entry.Provenance.RuntimeRevision = record.Runtime.Revision
	}
	if record.Driver != nil {
		entry.Provenance.DriverRevision = record.Driver.Revision
	}
	return entry
}

// bindMemoryObservation labels a reading with the source captured before its
// request. Operator authority is deliberately not an identity provider.
func bindMemoryObservation(entry *performancewire.MemoryEvidenceEntry, capture memoryEvidenceCapture, engine, model string) {
	entry.Source = performancewire.MemorySource{
		HostUUID:           memoryOptionalID(capture.source.hostUUID),
		ManagerIncarnation: memoryOptionalID(capture.source.managerIncarnation),
	}
	if identity, ok := capture.source.engines[engine]; ok {
		entry.Source.EngineRunID = memoryOptionalID(identity.EngineRunID)
		entry.Source.LaunchDigest = memoryOptionalDigest(identity.LaunchDigest)
		if digest, ok := memoryInventoryDigest(identity.ModelDigests, capture.source.naming[engine], model); ok {
			entry.Source.ModelDigest = memoryOptionalDigest(digest)
		}
	}
	if applied, ok := capture.source.settings[engine]; ok && applied.ok {
		entry.AppliedSettings = performancewire.MemoryAppliedSettings{Epoch: memoryValue(applied.epoch), Revision: memoryValue(applied.revision)}
	}
}

// boundedMemoryEvidence keeps a deterministic prefix of valid entries. Memory
// failures never change the separate installed/loaded model inventory.
func boundedMemoryEvidence(entries []performancewire.MemoryEvidenceEntry) *performancewire.MemoryEvidence {
	slices.SortFunc(entries, func(a, b performancewire.MemoryEvidenceEntry) int {
		if order := strings.Compare(a.Engine, b.Engine); order != 0 {
			return order
		}
		if order := strings.Compare(a.Model, b.Model); order != 0 {
			return order
		}
		return strings.Compare(memoryDomainKey(a), memoryDomainKey(b))
	})
	result := &performancewire.MemoryEvidence{Version: performancewire.MemoryEvidenceVersion, Entries: []performancewire.MemoryEvidenceEntry{}}
	seen := make(map[string]bool)
	for _, entry := range entries {
		key := memoryReadingKey(entry.Engine, entry.Model) + "\x00" + memoryDomainKey(entry)
		if seen[key] {
			continue
		}
		seen[key] = true
		if entry.Validate() != nil || len(result.Entries) >= performancewire.MaxMemoryEvidenceEntries {
			result.OmittedEntries++
			continue
		}
		result.Entries = append(result.Entries, entry)
	}
	for {
		data, err := json.Marshal(result)
		if err == nil && len(data) <= performancewire.MaxMemoryEvidenceBytes {
			return result
		}
		if len(result.Entries) == 0 {
			return nil
		}
		result.Entries = result.Entries[:len(result.Entries)-1]
		result.OmittedEntries++
	}
}

func memoryDomainKey(entry performancewire.MemoryEvidenceEntry) string {
	if entry.Envelope.Domain == nil {
		return ""
	}
	return string(*entry.Envelope.Domain)
}

func memoryUnknownLoadedNames(capture memoryEvidenceCapture, engine string, names []string, entries []performancewire.MemoryEvidenceEntry) []performancewire.MemoryEvidenceEntry {
	covered := make(map[string]bool, len(entries))
	for _, entry := range entries {
		covered[entry.Model] = true
	}
	for _, rawName := range names {
		name := memoryCanonicalModel(rawName, capture.source.naming[engine])
		if name == "" || covered[name] {
			continue
		}
		covered[name] = true
		record := memoryQualificationRecord{Source: &memoryArtifactSource{Engine: memoryValue(engine)}}
		entry := baseMemoryEvidenceEntry(record, capture, name)
		entry.Uncertainty = []performancewire.MemoryUncertainty{performancewire.MemoryUncertaintyUnsupported}
		entry.Freshness.Generation = nil
		entries = append(entries, entry)
	}
	return entries
}

func recordEngine(record memoryQualificationRecord) string {
	if record.Source != nil && record.Source.Engine != nil {
		return *record.Source.Engine
	}
	return ""
}

func recordModel(record memoryQualificationRecord) string {
	if record.Source != nil && record.Source.Model != nil {
		return *record.Source.Model
	}
	return ""
}

func recordModelDigest(record memoryQualificationRecord) string {
	if record.Source != nil && record.Source.ModelDigest != nil {
		return *record.Source.ModelDigest
	}
	return ""
}

func memoryReadingKey(engine, model string) string {
	return engine + "\x00" + model
}

func appendMemoryUncertainty(list []performancewire.MemoryUncertainty, value performancewire.MemoryUncertainty) []performancewire.MemoryUncertainty {
	if slices.Contains(list, value) {
		return list
	}
	return append(list, value)
}

// memoryPrimitive bounds the pointer helper to the wire field types this file
// actually needs.
type memoryPrimitive interface {
	~string | int | int64 | uint64
}

func memoryValue[T memoryPrimitive](value T) *T {
	return &value
}

func memoryOptionalID(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func memoryOptionalDigest(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}
