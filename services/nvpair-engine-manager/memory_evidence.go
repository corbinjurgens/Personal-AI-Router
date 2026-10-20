// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// memory_evidence.go is the engine-manager half of the local memory-evidence
// contract: a bounded, source-bound producer that reads the fixed, read-only
// operator qualification document and compares it with the same loaded-model
// response the model sweep already fetches. It never queries an engine, starts
// a process, or changes admission. A qualification is scoped to the current
// manager incarnation, engine run, launch, model content and fully applied
// settings; absent or mismatched authority leaves the qualified upper nil.

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
	// memoryExtractorOllamaPSRevision is the one supported loaded-memory
	// extractor: the official Ollama /api/ps shape. It is not a live runtime
	// version acceptance claim.
	memoryExtractorOllamaPSRevision = "ollama-ps-v1"
	// memoryFamilyOllama is the registry descriptor family that owns the
	// loaded-model memory extractor. An engine ID never grants support by
	// itself.
	memoryFamilyOllama = "ollama"
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
	readings   map[string]memoryReading
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

// memoryReading is the last completed allocation observation for one source
// model. observedAt retains the conservative pre-I/O capture anchor, rather
// than the time this reading was projected. generation orders overlapping
// sweeps; a late older completion never replaces a newer reading.
type memoryReading struct {
	generation uint64
	observedAt time.Time
	allocated  *int64
	context    *int64
	digest     string
	ambiguous  bool
}

// memoryEvidenceCapture is taken before the existing loaded-model read. The
// model owner passes the raw response to memoryEvidenceEntries afterwards, which
// rechecks this same source before publishing anything. startedAt is a
// conservative monotonic allocation-age anchor: source capture, the whole
// sweep's I/O and buffering, and verification all count toward reading age.
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
	// families and naming come from the live registry descriptor, not from
	// the artifact. They decide extractor support and canonical model names.
	families map[string]string
	naming   map[string]string
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
// Raw is the existing loaded_models response; this function performs no query.
// A memory failure never removes model names and never widens the bound.
func (e *Executor) memoryEvidenceEntries(capture memoryEvidenceCapture, engine string, raw json.RawMessage) []performancewire.MemoryEvidenceEntry {
	naming := capture.source.naming[engine]
	rows, rowsOK := parseOllamaLoadedMemory(raw, naming)
	if capture.source.families[engine] != memoryFamilyOllama {
		rowsOK = false
	}

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

	qualifiedUpper := p.authority.withdrawn == false && p.authority.clockUncertain == false
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
		} else {
			qualifiedUpper = false
		}
	}

	if len(records) == 0 {
		return memoryEvidenceLoadedUnknown(capture, engine, rows, rowsOK, sourceUnchanged, qualificationAge, now)
	}

	relevant := make([]memoryQualificationRecord, 0, len(records))
	for _, record := range records {
		if record.Source != nil && record.Source.Engine != nil && *record.Source.Engine == engine {
			relevant = append(relevant, record)
		}
	}
	if len(relevant) == 0 {
		return memoryEvidenceLoadedUnknown(capture, engine, rows, rowsOK, sourceUnchanged, qualificationAge, now)
	}

	// Contradictions survive overlapping sweeps even when their observations
	// are too old to publish. Bind them to authority retained or read before
	// I/O so a late response cannot withdraw replacement authority.
	if sourceUnchanged && state != nil && (capture.artifactDigest == state.digest || capture.retainedArtifactDigest == state.digest) && !p.authority.withdrawn && rowsOK {
		for _, record := range relevant {
			if !recordMatchesCapture(record, capture.source, engine) || !memoryRecordSupported(record, capture.source) {
				continue
			}
			model := memoryCanonicalModel(recordModel(record), naming)
			reading, ok := rows[model]
			if !ok || reading.ambiguous || reading.allocated == nil || reading.digest != recordModelDigest(record) {
				continue
			}
			if *reading.allocated > *record.Qualification.UpperBytes {
				p.authority.withdrawn = true
				qualifiedUpper = false
			}
		}
	}

	entries := make([]performancewire.MemoryEvidenceEntry, 0, len(relevant))
	covered := make(map[string]bool, len(relevant))
	for _, record := range relevant {
		entry := p.recordEntry(record, capture, engine, rows, rowsOK, sourceUnchanged, artifactCurrent, qualificationAge, qualifiedUpper)
		entries = append(entries, entry)
		covered[entry.Model] = true
	}
	for _, entry := range memoryEvidenceLoadedUnknown(capture, engine, rows, rowsOK, sourceUnchanged, nil, now) {
		if !covered[entry.Model] {
			entries = append(entries, entry)
		}
	}
	return entries
}

// finalizeMemoryEvidence is the whole-sweep fence for the model publication
// owner. Call it once, after every engine's raw response for one capture has
// been passed to memoryEvidenceEntries and before freezing the payload. A
// contradiction latched by any engine, a changed source/config/content, or a
// newer capture invalidates every positive upper in the combined projection,
// including entries that were built earlier in the same sweep. Advisory
// observation ages and source bindings are fenced as well. It reads the current
// source outside the producer lock and performs no I/O while holding it.
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
	readings := make(map[string]memoryReading, len(p.readings))
	for key, reading := range p.readings {
		readings[key] = reading
		if now.Before(reading.observedAt) {
			p.authority.clockUncertain = true
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
			anchor := time.Time{}
			if out[i].Freshness.Generation != nil {
				if *out[i].Freshness.Generation == capture.generation {
					anchor = capture.startedAt
				}
				if reading, ok := readings[memoryReadingKey(out[i].Engine, out[i].Model)]; ok &&
					*out[i].Freshness.Generation == reading.generation {
					anchor = reading.observedAt
				}
			}
			if !anchor.IsZero() {
				age := now.Sub(anchor).Milliseconds()
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
	families := map[string]string{}
	naming := map[string]string{}
	for _, name := range e.reg.Names() {
		manifest, ok := e.reg.Get(name)
		if !ok || manifest == nil {
			continue
		}
		families[name] = manifest.FamilyOrDefault()
		naming[name] = manifest.ModelNamingOrDefault()
	}
	return memoryEvidenceSourceSnapshot{
		hostUUID:           host,
		managerIncarnation: e.switchIncarnation,
		engines:            identity.Engines,
		epochs:             identity.epochs,
		settings:           e.memoryAppliedSettings(),
		families:           families,
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
	if len(a.families) != len(b.families) || len(a.naming) != len(b.naming) {
		return false
	}
	for name, family := range a.families {
		if b.families[name] != family || b.naming[name] != a.naming[name] {
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
	// Readings survive a new document; they are observations, not authority.
	// Prune anything the new document no longer names so the map stays bounded.
	if p.readings == nil {
		p.readings = make(map[string]memoryReading)
	}
	allowed := make(map[string]bool, len(*document.Records))
	for _, record := range *document.Records {
		if record.Source != nil && record.Source.Engine != nil && record.Source.Model != nil {
			naming := source.naming[*record.Source.Engine]
			allowed[memoryReadingKey(*record.Source.Engine, memoryCanonicalModel(*record.Source.Model, naming))] = true
		}
	}
	for key := range p.readings {
		if !allowed[key] {
			delete(p.readings, key)
		}
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

func (p *memoryEvidenceProducer) recordEntry(record memoryQualificationRecord, capture memoryEvidenceCapture, engine string, rows map[string]memoryReading, rowsOK, sourceUnchanged, artifactCurrent bool, qualificationAge *int64, qualifiedUpper bool) performancewire.MemoryEvidenceEntry {
	model := memoryCanonicalModel(recordModel(record), capture.source.naming[engine])
	reasons := make([]performancewire.MemoryUncertainty, 0, 4)
	reading, hasReading := rows[model]
	if !rowsOK {
		hasReading = false
	}
	now := p.clock()
	staleGeneration := capture.generation < p.generation
	entry := baseMemoryEvidenceEntry(record, capture, model)
	if !recordMatchesCapture(record, capture.source, engine) || (hasReading && reading.digest != recordModelDigest(record)) {
		// An old operator document is never current-run provenance for a
		// fresh allocation from a replacement engine or model.
		entry.Provenance = performancewire.MemoryProvenance{}
		entry.Envelope = performancewire.MemoryEnvelope{}
	}
	entry.Observed = performancewire.MemoryObservation{
		ExtractorRevision: memoryValue(memoryExtractorOllamaPSRevision),
	}
	if !memoryRecordSupported(record, capture.source) {
		hasReading = false
		entry.Observed.ExtractorRevision = nil
	}
	if hasReading {
		key := memoryReadingKey(engine, model)
		prior, seen := p.readings[key]
		switch {
		case staleGeneration:
			// Keep the locally read values but publish no age or upper: this
			// completion lost to a newer sweep.
			reading.observedAt = time.Time{}
		case seen && now.Before(prior.observedAt):
			// A backward sample time makes every age unprovable.
			p.authority.clockUncertain = true
			reading = prior
		case seen && capture.generation <= prior.generation:
			// The same capture/generation is a duplicate, not a new read.
			reading = prior
		default:
			if p.readings == nil {
				p.readings = make(map[string]memoryReading)
			}
			p.readings[key] = memoryReading{
				generation: capture.generation, observedAt: capture.startedAt, allocated: reading.allocated,
				context: reading.context, digest: reading.digest, ambiguous: reading.ambiguous,
			}
			reading = p.readings[key]
		}
		entry.Observed.AllocatedBytes = reading.allocated
		entry.Envelope.Domain = memoryValue(performancewire.MemoryDomainDedicatedDevice)
		if entry.Source.ModelDigest != nil && *entry.Source.ModelDigest != reading.digest {
			entry.Source.ModelDigest = nil
		}
		entry.Freshness.Generation = memoryValue(reading.generation)
		if reading.generation == 0 {
			entry.Freshness.Generation = nil
		}
		if !reading.observedAt.IsZero() {
			sampleAge := now.Sub(reading.observedAt).Milliseconds()
			if sampleAge < 0 {
				p.authority.clockUncertain = true
				reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyClockUncertain)
				qualifiedUpper = false
			} else {
				entry.Freshness.SampleAgeMs = &sampleAge
				if sampleAge > performancewire.MemoryReadingValidForMs {
					reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyStale)
					qualifiedUpper = false
				}
			}
		}
		if staleGeneration {
			reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyStale)
			qualifiedUpper = false
		}
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

	if !sourceUnchanged || !recordMatchesCapture(record, capture.source, engine) {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyIdentityChanged)
		qualifiedUpper = false
	}
	if !artifactCurrent {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyQualificationMissing)
		qualifiedUpper = false
	}
	if p.authority.clockUncertain {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyClockUncertain)
		qualifiedUpper = false
	}
	if p.authority.withdrawn {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyBoundExceeded)
		qualifiedUpper = false
	}
	if !memoryRecordSupported(record, capture.source) {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyUnsupported)
		qualifiedUpper = false
	}
	if !hasReading {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyMissingReading)
		qualifiedUpper = false
	}
	if hasReading && reading.ambiguous {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyQualificationMismatch)
		qualifiedUpper = false
	}
	if hasReading && reading.allocated == nil {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyMissingReading)
		qualifiedUpper = false
	}
	if hasReading && reading.digest != recordModelDigest(record) {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyQualificationMismatch)
		qualifiedUpper = false
	}
	if hasReading && reading.context == nil {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyContextUnknown)
		qualifiedUpper = false
	}
	if hasReading && reading.context != nil && record.Envelope != nil && record.Envelope.EffectiveContextTokens != nil && *reading.context != *record.Envelope.EffectiveContextTokens {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyQualificationMismatch)
		qualifiedUpper = false
	}
	if hasReading && reading.allocated != nil && record.Qualification != nil && record.Qualification.UpperBytes != nil &&
		*reading.allocated > *record.Qualification.UpperBytes {
		reasons = appendMemoryUncertainty(reasons, performancewire.MemoryUncertaintyBoundExceeded)
		qualifiedUpper = false
	}
	if qualifiedUpper && len(reasons) == 0 && hasReading && reading.allocated != nil &&
		entry.Freshness.SampleAgeMs != nil && entry.Freshness.QualificationAgeMs != nil {
		entry.State = performancewire.MemoryStateQualified
		entry.Uncertainty = []performancewire.MemoryUncertainty{}
		entry.Qualification = performancewire.MemoryQualification{
			UpperBytes: record.Qualification.UpperBytes, ID: record.Qualification.ID,
			Revision: memoryValue(p.artifact.revision), ArtifactDigest: memoryValue(p.artifact.digest),
			ProcedureDigest: record.Procedure.Digest,
		}
		return entry
	}
	if hasReading && reading.allocated != nil {
		entry.State = performancewire.MemoryStateAdvisory
	} else {
		entry.State = performancewire.MemoryStateUnknown
	}
	entry.Uncertainty = reasons
	if entry.Uncertainty == nil {
		entry.Uncertainty = []performancewire.MemoryUncertainty{}
	}
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
		entry.Uncertainty = []performancewire.MemoryUncertainty{performancewire.MemoryUncertaintyMissingReading}
		entry.Freshness.Generation = nil
		if capture.source.families[engine] != memoryFamilyOllama {
			entry.Uncertainty = []performancewire.MemoryUncertainty{performancewire.MemoryUncertaintyUnsupported}
		}
		entries = append(entries, entry)
	}
	return entries
}

func memoryEvidenceLoadedUnknown(capture memoryEvidenceCapture, engine string, rows map[string]memoryReading, rowsOK, sourceUnchanged bool, qualificationAge *int64, now time.Time) []performancewire.MemoryEvidenceEntry {
	if !rowsOK || len(rows) == 0 || capture.source.families[engine] != memoryFamilyOllama {
		return nil
	}
	names := make([]string, 0, len(rows))
	for name := range rows {
		names = append(names, name)
	}
	slices.Sort(names)
	if len(names) > maxMemoryQualificationRecords {
		names = names[:maxMemoryQualificationRecords]
	}
	out := make([]performancewire.MemoryEvidenceEntry, 0, len(names))
	for _, name := range names {
		reading := rows[name]
		entry := performancewire.MemoryEvidenceEntry{
			Engine: engine, Model: name, State: performancewire.MemoryStateUnknown,
			Uncertainty: []performancewire.MemoryUncertainty{performancewire.MemoryUncertaintyQualificationMissing},
			Source: performancewire.MemorySource{
				HostUUID: memoryOptionalID(capture.source.hostUUID), ManagerIncarnation: memoryOptionalID(capture.source.managerIncarnation),
			},
			AppliedSettings: performancewire.MemoryAppliedSettings{},
			Envelope:        performancewire.MemoryEnvelope{},
			Observed: performancewire.MemoryObservation{
				AllocatedBytes: reading.allocated, ExtractorRevision: memoryValue(memoryExtractorOllamaPSRevision),
			},
			Qualification: performancewire.MemoryQualification{},
			Provenance:    performancewire.MemoryProvenance{},
			Freshness: performancewire.MemoryFreshness{
				QualificationAgeMs: qualificationAge, Generation: memoryValue(capture.generation),
				ReadingValidForMs: performancewire.MemoryReadingValidForMs,
			},
		}
		entry.Envelope.Domain = memoryValue(performancewire.MemoryDomainDedicatedDevice)
		if !capture.startedAt.IsZero() && !now.Before(capture.startedAt) && now.UnixMilli() >= capture.startedAt.UnixMilli() {
			age := now.Sub(capture.startedAt).Milliseconds()
			entry.Freshness.SampleAgeMs = &age
			if age > performancewire.MemoryReadingValidForMs {
				entry.Uncertainty = appendMemoryUncertainty(entry.Uncertainty, performancewire.MemoryUncertaintyStale)
			}
		} else {
			entry.Uncertainty = appendMemoryUncertainty(entry.Uncertainty, performancewire.MemoryUncertaintyClockUncertain)
		}
		if identity, ok := capture.source.engines[engine]; ok {
			entry.Source.EngineRunID = memoryOptionalID(identity.EngineRunID)
			entry.Source.LaunchDigest = memoryOptionalDigest(identity.LaunchDigest)
			if digest, ok := memoryInventoryDigest(identity.ModelDigests, capture.source.naming[engine], name); ok && digest == reading.digest {
				entry.Source.ModelDigest = memoryOptionalDigest(digest)
			}
		}
		if applied, ok := capture.source.settings[engine]; ok && applied.ok {
			entry.AppliedSettings = performancewire.MemoryAppliedSettings{Epoch: memoryValue(applied.epoch), Revision: memoryValue(applied.revision)}
		}
		if reading.allocated != nil {
			entry.State = performancewire.MemoryStateAdvisory
		}
		if !sourceUnchanged {
			entry.Source = performancewire.MemorySource{}
			entry.AppliedSettings = performancewire.MemoryAppliedSettings{}
			entry.Envelope = performancewire.MemoryEnvelope{}
			entry.Freshness.SampleAgeMs = nil
			entry.Freshness.Generation = nil
			entry.Uncertainty = appendMemoryUncertainty(entry.Uncertainty, performancewire.MemoryUncertaintyIdentityChanged)
		}
		out = append(out, entry)
	}
	return out
}

func memoryRecordSupported(record memoryQualificationRecord, source memoryEvidenceSourceSnapshot) bool {
	if record.Source == nil || record.Envelope == nil || record.Envelope.Domain == nil ||
		record.Envelope.ExtractorRevision == nil || record.Source.Engine == nil {
		return false
	}
	return source.families[*record.Source.Engine] == memoryFamilyOllama &&
		*record.Envelope.Domain == performancewire.MemoryDomainDedicatedDevice &&
		*record.Envelope.ExtractorRevision == memoryExtractorOllamaPSRevision
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

// parseOllamaLoadedMemory reads only the official /api/ps members this
// extractor supports. Additional native fields are preserved from the response
// but ignored here. A bounded structural check rejects duplicate or
// case-variant keys in the response envelope and in every row before a decoder
// can silently keep the last value. Any ambiguous field, normalized name or
// digest ownership rejects the allocation response as a whole: dropping a row
// must not hide a conflict with a clean sibling. size_vram is the
// dedicated-device observation; missing, negative, fractional, overflowing or
// null is unknown, never zero. context_length missing stays unknown.
func parseOllamaLoadedMemory(raw json.RawMessage, naming string) (map[string]memoryReading, bool) {
	if len(raw) == 0 || len(raw) > performancewire.MaxMemoryEvidenceBytes || !utf8.Valid(raw) {
		return nil, false
	}
	object, duplicate, err := memoryStrictObjectFields(raw)
	if err != nil || duplicate {
		return nil, false
	}
	array, ok := object["models"]
	if !ok || bytes.Equal(bytes.TrimSpace(array), []byte("null")) {
		return nil, false
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(array, &elements); err != nil || len(elements) > maxMemoryQualificationRecords {
		return nil, false
	}
	rows := make(map[string]memoryReading, len(elements))
	digests := make(map[string]bool, len(elements))
	for _, element := range elements {
		elementFields, ambiguous, err := memoryStrictObjectFields(element)
		if err != nil || ambiguous {
			return nil, false
		}
		rawName := memoryJSONString(elementFields["name"])
		rawModel := memoryJSONString(elementFields["model"])
		digest := memoryJSONString(elementFields["digest"])
		if rawName == "" || rawModel == "" || digest == "" {
			return nil, false
		}
		name := memoryCanonicalModel(rawName, naming)
		model := memoryCanonicalModel(rawModel, naming)
		if name == "" || name != model {
			return nil, false
		}
		if _, seen := rows[name]; seen {
			return nil, false
		}
		if digests[digest] {
			return nil, false
		}
		digests[digest] = true
		rows[name] = memoryReading{
			allocated: memoryJSONNonnegative(elementFields["size_vram"]),
			context:   memoryJSONPositive(elementFields["context_length"]),
			digest:    digest,
		}
	}
	return rows, true
}

// memoryStrictObjectFields reads one JSON object, preserving additional native
// fields but reporting any duplicate key, including a case variant. The first
// value is retained only to identify the affected row; a duplicate is never
// resolved by keeping the last value.
func memoryStrictObjectFields(data []byte) (map[string]json.RawMessage, bool, error) {
	notObject := errors.New("loaded model response is not a JSON object")
	dec := json.NewDecoder(bytes.NewReader(data))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil, false, notObject
	}
	fields := map[string]json.RawMessage{}
	seen := map[string]string{}
	duplicate := false
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, false, notObject
		}
		rawKey, err := json.Marshal(tok)
		if err != nil {
			return nil, false, notObject
		}
		var key string
		if err := json.Unmarshal(rawKey, &key); err != nil {
			return nil, false, notObject
		}
		for prior := range seen {
			if strings.EqualFold(prior, key) {
				duplicate = true
			}
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return nil, false, notObject
		}
		if _, exists := fields[key]; !exists {
			fields[key] = value
			seen[key] = key
		}
	}
	if _, err := dec.Token(); err != nil {
		return nil, false, notObject
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, false, notObject
	}
	return fields, duplicate, nil
}

func memoryJSONString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return ""
	}
	return value
}

func memoryJSONNonnegative(raw json.RawMessage) *int64 {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var value int64
	if err := json.Unmarshal(raw, &value); err != nil || value < 0 || value > performancewire.MaxMemoryBytes {
		return nil
	}
	return &value
}

func memoryJSONPositive(raw json.RawMessage) *int64 {
	value := memoryJSONNonnegative(raw)
	if value == nil || *value <= 0 {
		return nil
	}
	return value
}
