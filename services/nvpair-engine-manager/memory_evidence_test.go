// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nvpair-shared/clustertrust"
	"nvpair-shared/clustertrusttest"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/performancewire"
)

const (
	memoryTestEngine = "memory-engine"
	memoryTestModel  = "memory-model.gguf"
	memoryTestRun    = "run-current"
	memoryTestLaunch = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	memoryTestDigest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	// memoryTestExtractorRevision is a document-supplied label. No engine
	// supports an extractor, so it never makes an entry supported.
	memoryTestExtractorRevision = "test-extractor-v1"
)

type memoryEvidenceHarness struct {
	ex       *Executor
	engine   string
	host     string
	clock    time.Time
	runID    string
	launch   string
	digest   string
	settings settings.Snapshot
}

func newMemoryEvidenceHarness(t *testing.T) *memoryEvidenceHarness {
	t.Helper()
	return newMemoryEvidenceHarnessFor(t, memoryTestEngine)
}

func newMemoryEvidenceHarnessFor(t *testing.T, engine string) *memoryEvidenceHarness {
	t.Helper()
	h := &memoryEvidenceHarness{
		engine: engine,
		host:   "host-a",
		clock:  time.UnixMilli(1_760_000_000_000).UTC(),
		runID:  memoryTestRun,
		launch: memoryTestLaunch,
		digest: memoryTestDigest,
		settings: settings.Snapshot{
			Engine: engine, Revision: 7, AppliedRevision: 7, Epoch: "epoch-1", Phase: "succeeded",
		},
	}
	manifest := &Manifest{Engine: engine, DisplayName: "Memory fixture", ModelNaming: engines.ModelNamingExact}
	reg := NewRegistry()
	reg.engines[engine] = manifest
	h.ex = NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	h.ex.engines[engine] = &engineState{
		manifest: manifest, plat: &Platform{}, running: true, runID: h.runID, launchDigest: h.launch,
		proc: &managedProc{}, contentMu: make(chan struct{}, 1),
	}
	h.ex.inventoryDigests = map[string]observedDigests{
		engine: {runID: h.runID, models: map[string]string{memoryTestModel: h.digest}, epoch: 9},
	}
	h.ex.settingsHub.Publish([]settings.Snapshot{h.settings})
	h.ex.memoryEvidence.hostUUID = func() string { return h.host }
	h.ex.memoryEvidence.now = func() time.Time { return h.clock }
	return h
}

func (h *memoryEvidenceHarness) record() memoryQualificationRecord {
	engine := h.engine
	return memoryQualificationRecord{
		Source: &memoryArtifactSource{
			HostUUID: memoryValue(h.host), ManagerIncarnation: memoryValue(h.ex.managerIncarnation),
			Engine: &engine, EngineRunID: memoryValue(h.runID), LaunchDigest: memoryValue(h.launch),
			Model: memoryValue(memoryTestModel), ModelDigest: memoryValue(h.digest),
		},
		Config:        &memoryArtifactConfig{Epoch: memoryValue(h.settings.Epoch), Revision: memoryValue(int64(h.settings.Revision))},
		Envelope:      memoryTestEnvelope(),
		Qualification: memoryTestQualification(10_000),
		Procedure:     memoryTestProcedure(),
		Hardware:      &memoryArtifactHardware{Fingerprint: memoryValue("gpu-a"), Description: memoryValue("test GPU")},
		Runtime:       &memoryArtifactRuntime{Revision: memoryValue("engine-1.0")},
		Driver:        &memoryArtifactDriver{Revision: memoryValue("driver-1")},
	}
}

func memoryTestEnvelope() *memoryArtifactEnvelope {
	domain := performancewire.MemoryDomainDedicatedDevice
	return &memoryArtifactEnvelope{
		Domain: &domain, EffectiveContextTokens: memoryValue[int64](8192),
		ConcurrentExecutions: memoryValue[int64](2), OffloadMode: memoryValue("gpu-full"),
		ExtractorRevision: memoryValue(memoryTestExtractorRevision),
	}
}

func memoryTestQualification(upper int64) *memoryArtifactQualification {
	scope := performancewire.MemoryQualificationScopeOperatorAttestedCurrentRun
	return &memoryArtifactQualification{ID: memoryValue("mq-1"), UpperBytes: &upper, Scope: &scope}
}

func memoryTestProcedure() *memoryArtifactProcedure {
	return &memoryArtifactProcedure{
		ID: memoryValue("mem-proc"), Revision: memoryValue("2"),
		Digest: memoryValue("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"),
	}
}

func (h *memoryEvidenceHarness) artifact(revision string) memoryQualificationArtifact {
	measured := h.clock.Add(-5 * time.Minute).UnixMilli()
	records := []memoryQualificationRecord{h.record()}
	return memoryQualificationArtifact{
		Version: memoryValue(1), Revision: &revision, MeasuredAt: &measured, Records: &records,
	}
}

func (h *memoryEvidenceHarness) writeArtifact(t *testing.T, document memoryQualificationArtifact) []byte {
	t.Helper()
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatalf("marshal setup artifact: %v", err)
	}
	h.writeArtifactBytes(t, data)
	return data
}

func (h *memoryEvidenceHarness) writeArtifactBytes(t *testing.T, data []byte) {
	t.Helper()
	path := filepath.Join(h.ex.baseDir, memoryQualificationFile)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write setup artifact: %v", err)
	}
}

func (h *memoryEvidenceHarness) removeArtifact(t *testing.T) {
	t.Helper()
	if err := os.Remove(filepath.Join(h.ex.baseDir, memoryQualificationFile)); err != nil {
		t.Fatalf("remove setup artifact: %v", err)
	}
}

func (h *memoryEvidenceHarness) entries() []performancewire.MemoryEvidenceEntry {
	capture := h.ex.captureMemoryEvidence()
	return h.ex.memoryEvidenceEntries(capture, h.engine)
}

func memoryEntry(t *testing.T, entries []performancewire.MemoryEvidenceEntry) performancewire.MemoryEvidenceEntry {
	t.Helper()
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(entries))
	}
	return entries[0]
}

func memoryUpper(entry performancewire.MemoryEvidenceEntry) *int64 {
	return entry.Qualification.UpperBytes
}

func slicesContain(list []performancewire.MemoryUncertainty, want performancewire.MemoryUncertainty) bool {
	for _, item := range list {
		if item == want {
			return true
		}
	}
	return false
}

// requireUnsupportedUnknown fails unless the entry is the unsupported outcome:
// an unknown state with no allocation, extractor revision or qualified upper.
func requireUnsupportedUnknown(t *testing.T, entry performancewire.MemoryEvidenceEntry) {
	t.Helper()
	if entry.State != performancewire.MemoryStateUnknown {
		t.Fatalf("state = %q, want unknown: %+v", entry.State, entry)
	}
	if !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyUnsupported) {
		t.Fatalf("uncertainty = %v, want unsupported", entry.Uncertainty)
	}
	if memoryUpper(entry) != nil || entry.Observed.AllocatedBytes != nil || entry.Observed.ExtractorRevision != nil {
		t.Fatalf("unsupported entry carries a reading or upper: %+v", entry)
	}
	if err := entry.Validate(); err != nil {
		t.Fatalf("unsupported entry is invalid: %v", err)
	}
}

func TestManagerMemoryEvidenceUsesLiveMeshIdentity(t *testing.T) {
	h := memoryModelsFixture(t)
	clusterDir := t.TempDir()
	clustertrusttest.WriteKeypair(t, clusterDir, "uuid-memory-first")
	mesh := clustertrust.Open(clusterDir)
	NewManager(NewCodec(&bytes.Buffer{}), h.ex, mesh, clusterDir)
	h.host = "uuid-memory-first"
	h.writeArtifact(t, h.artifact("2"))
	entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
	requireUnsupportedUnknown(t, entry)
	if entry.Source.HostUUID == nil || *entry.Source.HostUUID != mesh.NodeUUID() {
		t.Fatalf("manager did not bind the entry to the mesh identity: %+v", entry)
	}
	if slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyIdentityChanged) {
		t.Fatalf("document for the current mesh identity reported an identity change: %+v", entry)
	}

	clustertrusttest.WriteKeypair(t, clusterDir, "uuid-memory-second")
	mesh.Refresh()
	if got := mesh.NodeUUID(); got != "uuid-memory-second" {
		t.Fatalf("refreshed mesh UUID = %q, want uuid-memory-second", got)
	}
	entry = modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
	if !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyIdentityChanged) || entry.Provenance.Scope != nil {
		t.Fatalf("old host authority survived mesh identity refresh: %+v", entry)
	}
	h.host = "uuid-memory-second"
	h.writeArtifact(t, h.artifact("3"))
	entry = modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
	if entry.Source.HostUUID == nil || *entry.Source.HostUUID != mesh.NodeUUID() ||
		slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyIdentityChanged) {
		t.Fatalf("manager did not read refreshed mesh identity: %+v", entry)
	}
}

func TestManagerMemoryEvidenceNilMeshOverridesInjectedIdentity(t *testing.T) {
	h := memoryModelsFixture(t)
	h.writeArtifact(t, h.artifact("2"))
	NewManager(NewCodec(&bytes.Buffer{}), h.ex, nil, "")
	entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
	if entry.Source.HostUUID != nil {
		t.Fatalf("nil mesh retained injected host identity: %+v", entry)
	}
}

func TestMemoryEvidenceDocumentedModelIsUnsupported(t *testing.T) {
	test := func(name, engine string) {
		t.Run(name, func(t *testing.T) {
			h := newMemoryEvidenceHarnessFor(t, engine)
			h.writeArtifact(t, h.artifact("2"))

			entry := memoryEntry(t, h.entries())
			requireUnsupportedUnknown(t, entry)
			if !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyMissingReading) {
				t.Fatalf("uncertainty = %v, want missing reading", entry.Uncertainty)
			}
			if slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyIdentityChanged) ||
				slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyQualificationMissing) {
				t.Fatalf("a matching, current document reported %v", entry.Uncertainty)
			}
			if entry.Freshness.QualificationAgeMs == nil || *entry.Freshness.QualificationAgeMs < 5*60*1000 {
				t.Fatalf("qualificationAgeMs = %v, want at least the measured age", entry.Freshness.QualificationAgeMs)
			}
			if entry.Freshness.ReadingValidForMs != performancewire.MemoryReadingValidForMs {
				t.Fatalf("readingValidForMs = %d, want %d", entry.Freshness.ReadingValidForMs, performancewire.MemoryReadingValidForMs)
			}
		})
	}
	test("descriptor engine", memoryTestEngine)
	test("llama.cpp", engines.NameLlamaCPP)
}

func TestMemoryEvidenceNoEntryWithoutCurrentAuthority(t *testing.T) {
	test := func(name string, writeDocument bool, setup func(h *memoryEvidenceHarness)) {
		t.Run(name, func(t *testing.T) {
			h := newMemoryEvidenceHarness(t)
			if writeDocument {
				h.writeArtifact(t, h.artifact("2"))
			}
			setup(h)
			if entries := h.entries(); len(entries) != 0 {
				t.Fatalf("entries = %+v, want none", entries)
			}
		})
	}
	test("missing document", false, func(h *memoryEvidenceHarness) {})
	test("no host identity", true, func(h *memoryEvidenceHarness) { h.host = "" })
	test("no owned identity", true, func(h *memoryEvidenceHarness) { h.ex.engines[h.engine].running = false })
	test("unapplied settings", true, func(h *memoryEvidenceHarness) {
		applying := h.settings
		applying.Phase = "applying"
		h.ex.settingsHub.Publish([]settings.Snapshot{applying})
	})
}

func TestMemoryEvidenceMalformedArtifactRejectsWholeSubset(t *testing.T) {
	cases := []struct {
		name string
		data func(t *testing.T, h *memoryEvidenceHarness) []byte
	}{
		{
			name: "invalid json",
			data: func(t *testing.T, h *memoryEvidenceHarness) []byte { return []byte(`{"version":1,`) },
		},
		{
			name: "duplicate key",
			data: func(t *testing.T, h *memoryEvidenceHarness) []byte {
				return []byte(`{"version":1,"revision":"2","revision":"2","measuredAt":1,"records":[]}`)
			},
		},
		{
			name: "case variant key",
			data: func(t *testing.T, h *memoryEvidenceHarness) []byte {
				return []byte(`{"Version":1,"version":1,"revision":"2","measuredAt":1,"records":[]}`)
			},
		},
		{
			name: "oversize",
			data: func(t *testing.T, h *memoryEvidenceHarness) []byte {
				return []byte(strings.Repeat("x", maxMemoryQualificationBytes+1))
			},
		},
		{
			name: "one invalid record invalidates the valid one",
			data: func(t *testing.T, h *memoryEvidenceHarness) []byte {
				document := h.artifact("2")
				first := h.record()
				second := h.record()
				second.Qualification.UpperBytes = nil
				document.Records = &[]memoryQualificationRecord{first, second}
				data, err := json.Marshal(document)
				if err != nil {
					t.Fatalf("marshal setup: %v", err)
				}
				return data
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newMemoryEvidenceHarness(t)
			h.writeArtifactBytes(t, tc.data(t, h))
			if entries := h.entries(); len(entries) != 0 {
				t.Fatalf("entries = %+v, want none when the document is rejected as a whole", entries)
			}
		})
	}
	t.Run("read error", func(t *testing.T) {
		h := newMemoryEvidenceHarness(t)
		if err := os.Mkdir(filepath.Join(h.ex.baseDir, memoryQualificationFile), 0o700); err != nil {
			t.Fatalf("create unreadable setup path: %v", err)
		}
		if entries := h.entries(); len(entries) != 0 {
			t.Fatalf("entries = %+v, want none after a read error", entries)
		}
	})
}

func TestMemoryEvidenceBindingMismatchesRejectDocument(t *testing.T) {
	test := func(name string, mutate func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact)) {
		t.Run(name, func(t *testing.T) {
			h := newMemoryEvidenceHarness(t)
			document := h.artifact("2")
			mutate(t, h, &document)
			h.writeArtifact(t, document)
			if entries := h.entries(); len(entries) != 0 {
				t.Fatalf("entries = %+v, want none for a document bound to another source (%s)", entries, name)
			}
			if h.ex.memoryEvidence.authority.present {
				t.Fatalf("a mismatched document became authority (%s)", name)
			}
		})
	}
	test("host", func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) {
		(*document.Records)[0].Source.HostUUID = memoryValue("host-other")
	})
	test("manager incarnation", func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) {
		(*document.Records)[0].Source.ManagerIncarnation = memoryValue("mgr-other")
	})
	test("engine run", func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) {
		h.runID = "run-new"
		h.ex.engines[h.engine].runID = h.runID
		h.ex.inventoryDigests[h.engine] = observedDigests{runID: h.runID, models: map[string]string{memoryTestModel: h.digest}, epoch: 9}
	})
	test("launch", func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) {
		h.launch = strings.Repeat("d", 64)
		h.ex.engines[h.engine].launchDigest = h.launch
	})
	test("model content", func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) {
		h.digest = strings.Repeat("e", 64)
		h.ex.inventoryDigests[h.engine] = observedDigests{runID: h.runID, models: map[string]string{memoryTestModel: h.digest}, epoch: 9}
	})
	test("applied settings revision", func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) {
		next := h.settings
		next.Revision, next.AppliedRevision = 8, 8
		h.ex.settingsHub.Publish([]settings.Snapshot{next})
	})
	test("copied RTX record on GTX host", func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) {
		(*document.Records)[0].Source.HostUUID = memoryValue("rtx-host")
		h.host = "gtx-host"
	})
	test("same host new run", func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) {
		(*document.Records)[0].Source.EngineRunID = memoryValue("run-old")
	})
	test("replaced model content", func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) {
		h.ex.inventoryMu.Lock()
		h.ex.inventoryDigests[h.engine].models[memoryTestModel] = strings.Repeat("d", 64)
		h.ex.inventoryMu.Unlock()
	})
}

func TestMemoryEvidenceOlderCompletionIsStale(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	older := h.ex.captureMemoryEvidence()
	newer := h.ex.captureMemoryEvidence()

	first := memoryEntry(t, h.ex.finalizeMemoryEvidence(newer, h.ex.memoryEvidenceEntries(newer, h.engine)))
	if slicesContain(first.Uncertainty, performancewire.MemoryUncertaintyStale) {
		t.Fatalf("the newer completion is stale: %v", first.Uncertainty)
	}
	late := memoryEntry(t, h.ex.finalizeMemoryEvidence(older, h.ex.memoryEvidenceEntries(older, h.engine)))
	requireUnsupportedUnknown(t, late)
	if !slicesContain(late.Uncertainty, performancewire.MemoryUncertaintyStale) {
		t.Fatalf("uncertainty = %v, want stale", late.Uncertainty)
	}
}

func TestMemoryEvidenceQualificationAgeAndClockUncertainty(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	original := h.writeArtifact(t, h.artifact("2"))
	capture := h.ex.captureMemoryEvidence()
	duplicateObservation := func() performancewire.MemoryEvidenceEntry {
		return memoryEntry(t, h.ex.memoryEvidenceEntries(capture, h.engine))
	}

	first := duplicateObservation()
	if first.Freshness.QualificationAgeMs == nil || *first.Freshness.QualificationAgeMs < 5*60*1000 {
		t.Fatalf("qualificationAgeMs = %v, want at least the measured age", first.Freshness.QualificationAgeMs)
	}

	h.clock = h.clock.Add(2 * time.Second)
	second := duplicateObservation()
	if *second.Freshness.QualificationAgeMs <= *first.Freshness.QualificationAgeMs {
		t.Fatalf("qualification age did not advance: %d then %d", *first.Freshness.QualificationAgeMs, *second.Freshness.QualificationAgeMs)
	}

	// A duplicate artifact must not rejuvenate the age.
	h.writeArtifactBytes(t, original)
	h.clock = h.clock.Add(time.Second)
	duplicate := duplicateObservation()
	if *duplicate.Freshness.QualificationAgeMs <= *second.Freshness.QualificationAgeMs {
		t.Fatalf("duplicate artifact rejuvenated qualification age: %d then %d", *second.Freshness.QualificationAgeMs, *duplicate.Freshness.QualificationAgeMs)
	}

	// A rollback makes every age unprovable and latches clock uncertainty.
	h.clock = h.clock.Add(-10 * time.Second)
	rolled := duplicateObservation()
	if !slicesContain(rolled.Uncertainty, performancewire.MemoryUncertaintyClockUncertain) || rolled.Freshness.QualificationAgeMs != nil {
		t.Fatalf("clock rollback left a provable age: %+v", rolled)
	}
	duplicateRolled := duplicateObservation()
	if !slicesContain(duplicateRolled.Uncertainty, performancewire.MemoryUncertaintyClockUncertain) {
		t.Fatal("a duplicate artifact cleared latched clock uncertainty")
	}

	// A new, higher reviewed document can recover.
	h.clock = h.clock.Add(20 * time.Second)
	recovered := h.artifact("3")
	measured := h.clock.Add(-time.Second).UnixMilli()
	recovered.MeasuredAt = &measured
	h.writeArtifact(t, recovered)
	entry := memoryEntry(t, h.entries())
	if slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyClockUncertain) || entry.Freshness.QualificationAgeMs == nil {
		t.Fatalf("a higher reviewed revision did not recover: %+v", entry)
	}
}

func TestMemoryEvidenceDocumentRevisionHighWater(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	current := func(t *testing.T) bool {
		t.Helper()
		entry := memoryEntry(t, h.entries())
		requireUnsupportedUnknown(t, entry)
		return !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyQualificationMissing)
	}
	if !current(t) {
		t.Fatal("setup: the accepted document is not current")
	}

	t.Run("lower revision rejected", func(t *testing.T) {
		h.writeArtifact(t, h.artifact("1"))
		if current(t) {
			t.Fatal("a lower revision replaced the accepted document")
		}
	})
	t.Run("same revision different bytes rejected", func(t *testing.T) {
		document := h.artifact("2")
		(*document.Records)[0].Qualification = memoryTestQualification(20_000)
		h.writeArtifact(t, document)
		if current(t) {
			t.Fatal("different bytes at the same revision replaced the accepted document")
		}
	})
	t.Run("deletion leaves the document not current", func(t *testing.T) {
		h.removeArtifact(t)
		if current(t) {
			t.Fatal("a deleted document is current")
		}
	})
	t.Run("identical document is current again", func(t *testing.T) {
		h.writeArtifact(t, h.artifact("2"))
		if !current(t) {
			t.Fatal("the identical document is not current after reappearing")
		}
	})
	t.Run("higher revision replaces authority", func(t *testing.T) {
		document := h.artifact("3")
		(*document.Records)[0].Qualification = memoryTestQualification(12_000)
		h.writeArtifact(t, document)
		if !current(t) {
			t.Fatal("a higher revision was not accepted")
		}
		if got := h.ex.memoryEvidence.authority.revision; got != "3" {
			t.Fatalf("authority revision = %q, want 3", got)
		}
	})
}

func TestMemoryEvidenceRecordLimitRejectsWholeDocument(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	document := h.artifact("2")
	records := make([]memoryQualificationRecord, 0, maxMemoryQualificationRecords+1)
	for i := 0; i < maxMemoryQualificationRecords+1; i++ {
		record := h.record()
		record.Source.Model = memoryValue(fmt.Sprintf("model-%03d", i))
		records = append(records, record)
	}
	document.Records = &records
	h.writeArtifact(t, document)
	if entries := h.entries(); len(entries) != 0 {
		t.Fatalf("entries = %d, want none for an over-limit document", len(entries))
	}
}

func TestMemoryEvidenceReviewRejectsCaseVariantAuthorityKeys(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	data, err := json.Marshal(h.artifact("2"))
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"version", "hostUuid", "upperBytes"} {
		t.Run(key, func(t *testing.T) {
			old := `"` + key + `":`
			changed := strings.Replace(string(data), old, `"`+strings.ToUpper(key)+`":`, 1)
			if changed == string(data) {
				t.Fatal("case-variant probe did not change its key")
			}
			if _, err := decodeMemoryQualification([]byte(changed)); err == nil {
				t.Fatal("case-variant authority key was accepted")
			}
		})
	}
}

func TestMemoryEvidenceReviewDoesNotRelabelNewRunWithOldAuthority(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	original := memoryEntry(t, h.entries())
	if original.Provenance.Scope == nil {
		t.Fatal("setup did not retain the original run's attestation")
	}
	const nextRun = "run-restarted"
	state := h.ex.engines[h.engine]
	state.mu.Lock()
	state.runID = nextRun
	state.mu.Unlock()
	h.ex.inventoryMu.Lock()
	digests := h.ex.inventoryDigests[h.engine]
	digests.runID = nextRun
	h.ex.inventoryDigests[h.engine] = digests
	h.ex.inventoryMu.Unlock()
	capture := h.ex.captureMemoryEvidence()
	entries := h.ex.memoryEvidenceEntries(capture, h.engine)
	entry := memoryEntry(t, h.ex.finalizeMemoryEvidence(capture, entries))
	requireUnsupportedUnknown(t, entry)
	if !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyIdentityChanged) {
		t.Fatalf("uncertainty = %v, want identity changed", entry.Uncertainty)
	}
	if entry.Source.EngineRunID == nil || *entry.Source.EngineRunID != nextRun {
		t.Fatalf("new run entry was attributed to run %v, want %s", entry.Source.EngineRunID, nextRun)
	}
	if entry.Provenance.Scope != nil {
		t.Fatal("old run's operator attestation was inherited by the new run")
	}
}

func TestMemoryEvidenceFinalFenceRechecksDocumentBytes(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	capture := h.ex.captureMemoryEvidence()
	entries := h.ex.memoryEvidenceEntries(capture, h.engine)
	if slicesContain(memoryEntry(t, entries).Uncertainty, performancewire.MemoryUncertaintyQualificationMissing) {
		t.Fatal("positive control reported a missing qualification")
	}
	h.removeArtifact(t)
	fenced := memoryEntry(t, h.ex.finalizeMemoryEvidence(capture, entries))
	if !slicesContain(fenced.Uncertainty, performancewire.MemoryUncertaintyQualificationMissing) {
		t.Fatalf("deleted document survived the final publication fence: %v", fenced.Uncertainty)
	}
}

func TestMemoryEvidencePublicationBoundsAndOrdering(t *testing.T) {
	entries := make([]performancewire.MemoryEvidenceEntry, 0, 140)
	for i := 139; i >= 0; i-- {
		entry := performancewire.MemoryEvidenceEntry{
			Engine: "llamacpp", Model: fmt.Sprintf("model-%03d", i), State: performancewire.MemoryStateUnknown,
			Uncertainty: []performancewire.MemoryUncertainty{},
			Freshness:   performancewire.MemoryFreshness{ReadingValidForMs: performancewire.MemoryReadingValidForMs},
		}
		entries = append(entries, entry)
	}
	result := boundedMemoryEvidence(entries)
	if result == nil {
		t.Fatal("bounded evidence missing")
	}
	if err := result.Validate(); err != nil {
		t.Fatalf("bounded evidence invalid: %v", err)
	}
	if len(result.Entries) != 128 || result.OmittedEntries != 12 || result.Entries[0].Model != "model-000" || result.Entries[127].Model != "model-127" {
		t.Fatalf("wrong deterministic prefix/count: entries=%d omitted=%d", len(result.Entries), result.OmittedEntries)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("encode bounded evidence: %v", err)
	}
	if len(encoded) > performancewire.MaxMemoryEvidenceBytes {
		t.Fatal("encoded evidence exceeds byte bound")
	}
}

func TestMemoryEvidenceReversedPreIOClockRemainsUnknown(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	capture := h.ex.captureMemoryEvidence()
	h.clock = h.clock.Add(-time.Second)
	entries := h.ex.memoryEvidenceEntries(capture, h.engine)
	entry := memoryEntry(t, h.ex.finalizeMemoryEvidence(capture, entries))
	requireUnsupportedUnknown(t, entry)
	if entry.Freshness.SampleAgeMs != nil {
		t.Fatalf("reversed clock fabricated sample freshness: %+v", entry)
	}
	if !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyClockUncertain) {
		t.Fatalf("uncertainty = %v, want clock-uncertain", entry.Uncertainty)
	}
}
