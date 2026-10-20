// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/performancewire"
)

const (
	memoryTestModel  = "llama3:8b"
	memoryTestRun    = "run-current"
	memoryTestLaunch = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	memoryTestDigest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

type memoryEvidenceHarness struct {
	ex       *Executor
	host     string
	clock    time.Time
	runID    string
	launch   string
	digest   string
	settings settings.Snapshot
}

func newMemoryEvidenceHarness(t *testing.T) *memoryEvidenceHarness {
	t.Helper()
	h := &memoryEvidenceHarness{
		host:   "host-a",
		clock:  time.UnixMilli(1_760_000_000_000).UTC(),
		runID:  memoryTestRun,
		launch: memoryTestLaunch,
		digest: memoryTestDigest,
		settings: settings.Snapshot{
			Engine: "ollama", Revision: 7, AppliedRevision: 7, Epoch: "epoch-1", Phase: "succeeded",
		},
	}
	manifest := &Manifest{Engine: "ollama", DisplayName: "Ollama", Family: "ollama", ModelNaming: engines.ModelNamingImpliedLatestTag}
	reg := NewRegistry()
	reg.engines["ollama"] = manifest
	h.ex = NewExecutor(reg, NewReporter(nil), func(string, any) {}, t.TempDir())
	h.ex.engines["ollama"] = &engineState{
		manifest: manifest, plat: &Platform{}, running: true, runID: h.runID, launchDigest: h.launch,
		proc: &managedProc{}, contentMu: make(chan struct{}, 1),
	}
	h.ex.inventoryDigests = map[string]observedDigests{
		"ollama": {runID: h.runID, models: map[string]string{memoryTestModel: h.digest}, epoch: 9},
	}
	h.ex.settingsHub.Publish([]settings.Snapshot{h.settings})
	h.ex.memoryEvidence.hostUUID = func() string { return h.host }
	h.ex.memoryEvidence.now = func() time.Time { return h.clock }
	return h
}

func (h *memoryEvidenceHarness) record() memoryQualificationRecord {
	engine := "ollama"
	return memoryQualificationRecord{
		Source: &memoryArtifactSource{
			HostUUID: memoryValue(h.host), ManagerIncarnation: memoryValue(h.ex.switchIncarnation),
			Engine: &engine, EngineRunID: memoryValue(h.runID), LaunchDigest: memoryValue(h.launch),
			Model: memoryValue(memoryTestModel), ModelDigest: memoryValue(h.digest),
		},
		Config:        &memoryArtifactConfig{Epoch: memoryValue(h.settings.Epoch), Revision: memoryValue(int64(h.settings.Revision))},
		Envelope:      memoryTestEnvelope(),
		Qualification: memoryTestQualification(10_000),
		Procedure:     memoryTestProcedure(),
		Hardware:      &memoryArtifactHardware{Fingerprint: memoryValue("gpu-a"), Description: memoryValue("test GPU")},
		Runtime:       &memoryArtifactRuntime{Revision: memoryValue("ollama-0.12.3")},
		Driver:        &memoryArtifactDriver{Revision: memoryValue("driver-1")},
	}
}

func memoryTestEnvelope() *memoryArtifactEnvelope {
	domain := performancewire.MemoryDomainDedicatedDevice
	return &memoryArtifactEnvelope{
		Domain: &domain, EffectiveContextTokens: memoryValue[int64](8192),
		ConcurrentExecutions: memoryValue[int64](2), OffloadMode: memoryValue("gpu-full"),
		ExtractorRevision: memoryValue(memoryExtractorOllamaPSRevision),
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

func (h *memoryEvidenceHarness) entries(raw string) []performancewire.MemoryEvidenceEntry {
	capture := h.ex.captureMemoryEvidence()
	return h.ex.memoryEvidenceEntries(capture, "ollama", json.RawMessage(raw))
}

func memoryPS(name, digest string, sizeVRAM, context any) string {
	return fmt.Sprintf(`{"models":[{"name":%q,"model":%q,"digest":%q,"size_vram":%v,"context_length":%v}]}`,
		name, name, digest, sizeVRAM, context)
}

func memoryPSWithoutSize(name, digest string) string {
	return fmt.Sprintf(`{"models":[{"name":%q,"model":%q,"digest":%q,"context_length":8192}]}`, name, name, digest)
}

func memoryPSWithoutContext(name, digest string) string {
	return fmt.Sprintf(`{"models":[{"name":%q,"model":%q,"digest":%q,"size_vram":9000}]}`, name, name, digest)
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

func sameMemoryInt64(a, b *int64) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func TestModelsMemoryEvidenceFirstAuthorityDisappearsDuringContradiction(t *testing.T) {
	h, loaded := memoryModelsFixture(t)
	original := h.writeArtifact(t, h.artifact("2"))
	transport := h.ex.client.Transport
	var removed atomic.Bool
	h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path == "/api/ps" && removed.CompareAndSwap(false, true) {
			h.removeArtifact(t)
		}
		return transport.RoundTrip(request)
	})
	*loaded = memoryPS(memoryTestModel, memoryTestDigest, 11000, 8192)
	if entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background())); memoryUpper(entry) != nil {
		t.Fatalf("contradiction with missing document qualified: %+v", entry)
	}
	h.writeArtifactBytes(t, original)
	*loaded = memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
	entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
	if memoryUpper(entry) != nil || !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyBoundExceeded) {
		t.Fatalf("restored first authority forgot contradiction: %+v", entry)
	}
}

func TestModelsMemoryEvidenceOlderContradictionCompletesBeforeNewerRead(t *testing.T) {
	h, _ := memoryModelsFixture(t)
	h.writeArtifact(t, h.artifact("2"))
	entered := make(chan struct{}, 2)
	olderRelease, newerRelease := make(chan struct{}), make(chan struct{})
	var olderOnce, newerOnce sync.Once
	unblockOlder := func() { olderOnce.Do(func() { close(olderRelease) }) }
	unblockNewer := func() { newerOnce.Do(func() { close(newerRelease) }) }
	t.Cleanup(unblockOlder)
	t.Cleanup(unblockNewer)
	transport := h.ex.client.Transport
	var requests atomic.Int64
	h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path != "/api/ps" {
			return transport.RoundTrip(request)
		}
		n := requests.Add(1)
		allocation := 9000
		if n <= 2 {
			release := newerRelease
			if n == 1 {
				release, allocation = olderRelease, 11000
			}
			entered <- struct{}{}
			select {
			case <-release:
			case <-request.Context().Done():
				return nil, request.Context().Err()
			}
		}
		body := memoryPS(memoryTestModel, memoryTestDigest, allocation, 8192)
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
	})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	start := func() chan ModelsResult {
		result := make(chan ModelsResult, 1)
		go func() { result <- h.ex.ModelsResult(ctx) }()
		select {
		case <-entered:
		case <-ctx.Done():
			t.Fatal("loaded-model request did not enter fixture")
		}
		return result
	}
	older, newer := start(), start()
	unblockOlder()
	if entry := modelsMemoryEntry(t, <-older); memoryUpper(entry) != nil {
		t.Fatal("older completion published a qualified upper")
	}
	unblockNewer()
	for _, result := range []ModelsResult{<-newer, h.ex.ModelsResult(ctx)} {
		entry := modelsMemoryEntry(t, result)
		if memoryUpper(entry) != nil || !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyBoundExceeded) {
			t.Fatalf("lower reading forgot earlier completed contradiction: %+v", entry)
		}
	}
}

func TestModelsMemoryEvidenceRetainedAuthoritySurvivesUnavailableCapture(t *testing.T) {
	test := func(name string, corrupt bool) {
		t.Run(name, func(t *testing.T) {
			h, loaded := memoryModelsFixture(t)
			original := h.writeArtifact(t, h.artifact("2"))
			if memoryUpper(modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))) == nil {
				t.Fatal("initial authority did not qualify")
			}
			if corrupt {
				h.writeArtifactBytes(t, []byte(`{"version":`))
			} else {
				h.removeArtifact(t)
			}
			transport := h.ex.client.Transport
			var restored atomic.Bool
			h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
				if request.URL.Path == "/api/ps" && restored.CompareAndSwap(false, true) {
					h.writeArtifactBytes(t, original)
				}
				return transport.RoundTrip(request)
			})
			*loaded = memoryPS(memoryTestModel, memoryTestDigest, 11000, 8192)
			if entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background())); memoryUpper(entry) != nil {
				t.Fatalf("contradicting retained authority qualified: %+v", entry)
			}
			*loaded = memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
			if memoryUpper(entry) != nil || !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyBoundExceeded) {
				t.Fatalf("restored identical document forgot retained contradiction: %+v", entry)
			}
		})
	}
	test("missing document at capture", false)
	test("corrupt document at capture", true)
}

func TestModelsMemoryEvidenceDefersAuthorityChangedDuringRead(t *testing.T) {
	test := func(name string, initiallyPresent bool) {
		t.Run(name, func(t *testing.T) {
			h, loaded := memoryModelsFixture(t)
			if initiallyPresent {
				h.writeArtifact(t, h.artifact("2"))
				if memoryUpper(modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))) == nil {
					t.Fatal("initial authority did not qualify")
				}
			}
			prior := h.ex.memoryEvidence.authority
			transport := h.ex.client.Transport
			var changed atomic.Bool
			h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
				if request.URL.Path == "/api/ps" && changed.CompareAndSwap(false, true) {
					h.writeArtifact(t, h.artifact("3"))
				}
				return transport.RoundTrip(request)
			})
			*loaded = memoryPS(memoryTestModel, memoryTestDigest, 11000, 8192)
			entry := modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))
			if memoryUpper(entry) != nil || entry.Qualification.ID != nil || entry.Qualification.ArtifactDigest != nil || entry.State == performancewire.MemoryStateQualified {
				t.Fatalf("document changed during I/O supplied authority: %+v", entry)
			}
			// A same-capture contradiction may withdraw the retained document,
			// but cannot accept the replacement document observed only after I/O.
			prior.withdrawn = initiallyPresent
			if got := h.ex.memoryEvidence.authority; got != prior {
				t.Fatalf("document changed during I/O advanced authority: got %+v, want %+v", got, prior)
			}
			*loaded = memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			if memoryUpper(modelsMemoryEntry(t, h.ex.ModelsResult(context.Background()))) == nil {
				t.Fatal("fresh capture of replacement document did not qualify")
			}
		})
	}
	test("document appeared after capture", false)
	test("accepted document replaced after capture", true)
}

func TestModelsMemoryEvidenceOverlappingContradictionBinding(t *testing.T) {
	test := func(name string, replaceArtifact, replaceSource, removeArtifact bool, wantWithdrawn bool) {
		t.Run(name, func(t *testing.T) {
			h, _ := memoryModelsFixture(t)
			original := h.writeArtifact(t, h.artifact("2"))
			entered := make(chan struct{})
			release := make(chan struct{})
			var releaseOnce sync.Once
			unblock := func() { releaseOnce.Do(func() { close(release) }) }
			t.Cleanup(unblock)
			var requests atomic.Int64
			h.ex.client.Transport = memoryModelsTransport(func(request *http.Request) (*http.Response, error) {
				body := ""
				switch request.URL.Path {
				case "/api/tags":
					body = fmt.Sprintf(`{"models":[{"name":%q,"digest":%q}]}`, memoryTestModel, memoryTestDigest)
				case "/api/ps":
					allocation := 9000
					if requests.Add(1) == 1 {
						close(entered)
						select {
						case <-release:
						case <-request.Context().Done():
							return nil, request.Context().Err()
						}
						allocation = 11000
					}
					body = memoryPS(memoryTestModel, memoryTestDigest, allocation, 8192)
				default:
					return nil, fmt.Errorf("unexpected endpoint %s", request.URL.Path)
				}
				if request.Method != http.MethodGet {
					return nil, fmt.Errorf("unexpected method %s", request.Method)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
			})
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			older := make(chan ModelsResult, 1)
			go func() { older <- h.ex.ModelsResult(ctx) }()
			select {
			case <-entered:
			case <-ctx.Done():
				t.Fatal("older loaded-model request did not enter fixture")
			}
			if replaceSource {
				state := h.ex.engines["ollama"]
				state.mu.Lock()
				state.launchDigest = strings.Repeat("d", 64)
				state.mu.Unlock()
			}
			if replaceArtifact {
				h.writeArtifact(t, h.artifact("3"))
			}
			newer := modelsMemoryEntry(t, h.ex.ModelsResult(ctx))
			if got := memoryUpper(newer) != nil; got == replaceSource {
				t.Fatalf("newer lower reading qualified = %v, want %v", got, !replaceSource)
			}
			if removeArtifact {
				h.removeArtifact(t)
			}
			unblock()
			select {
			case result := <-older:
				if entry := modelsMemoryEntry(t, result); memoryUpper(entry) != nil {
					t.Fatal("older completion published a positive upper")
				}
			case <-ctx.Done():
				t.Fatal("older sweep did not complete")
			}
			if removeArtifact {
				h.writeArtifactBytes(t, original)
			}
			if replaceSource {
				state := h.ex.engines["ollama"]
				state.mu.Lock()
				state.launchDigest = memoryTestLaunch
				state.mu.Unlock()
			}
			last := modelsMemoryEntry(t, h.ex.ModelsResult(ctx))
			if got := memoryUpper(last) == nil; got != wantWithdrawn {
				t.Fatalf("subsequent lower reading withdrawn = %v, want %v: %+v", got, wantWithdrawn, last)
			}
			if got := slicesContain(last.Uncertainty, performancewire.MemoryUncertaintyBoundExceeded); got != wantWithdrawn {
				t.Fatalf("bound-exceeded = %v, want %v: %+v", got, wantWithdrawn, last)
			}
		})
	}
	test("same source and artifact retain contradiction", false, false, false, true)
	test("absent artifact cannot erase contradiction", false, false, true, true)
	test("replacement artifact excludes old contradiction", true, false, false, false)
	test("replacement source excludes old contradiction", false, true, false, false)
}

func TestMemoryEvidenceQualifiedSyntheticPositive(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	document := h.artifact("2")
	data := h.writeArtifact(t, document)

	entry := memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))
	if entry.State != performancewire.MemoryStateQualified {
		t.Fatalf("state = %q, want qualified: %+v", entry.State, entry.Uncertainty)
	}
	if entry.Qualification.UpperBytes == nil || *entry.Qualification.UpperBytes != 10_000 {
		t.Fatalf("upperBytes = %v, want 10000", entry.Qualification.UpperBytes)
	}
	if entry.Observed.AllocatedBytes == nil || *entry.Observed.AllocatedBytes != 9000 {
		t.Fatalf("allocatedBytes = %v, want 9000", entry.Observed.AllocatedBytes)
	}
	sum := sha256.Sum256(data)
	if entry.Qualification.ArtifactDigest == nil || *entry.Qualification.ArtifactDigest != hex.EncodeToString(sum[:]) {
		t.Fatalf("artifactDigest = %v, want SHA-256 of the document bytes", entry.Qualification.ArtifactDigest)
	}
	if entry.Qualification.Revision == nil || *entry.Qualification.Revision != "2" {
		t.Fatalf("revision = %v, want 2", entry.Qualification.Revision)
	}
	if entry.Freshness.ReadingValidForMs != performancewire.MemoryReadingValidForMs {
		t.Fatalf("readingValidForMs = %d, want %d", entry.Freshness.ReadingValidForMs, performancewire.MemoryReadingValidForMs)
	}
	if err := entry.Validate(); err != nil {
		t.Fatalf("produced qualified entry is invalid: %v", err)
	}
}

func TestMemoryEvidenceSafeNilDefaults(t *testing.T) {
	cases := []struct {
		name  string
		setup func(t *testing.T, h *memoryEvidenceHarness) string
	}{
		{
			name: "missing artifact",
			setup: func(t *testing.T, h *memoryEvidenceHarness) string {
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "no host identity",
			setup: func(t *testing.T, h *memoryEvidenceHarness) string {
				h.writeArtifact(t, h.artifact("2"))
				h.host = ""
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "no owned identity",
			setup: func(t *testing.T, h *memoryEvidenceHarness) string {
				h.writeArtifact(t, h.artifact("2"))
				h.ex.engines["ollama"].running = false
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "unapplied settings",
			setup: func(t *testing.T, h *memoryEvidenceHarness) string {
				h.writeArtifact(t, h.artifact("2"))
				applying := h.settings
				applying.Phase = "applying"
				h.ex.settingsHub.Publish([]settings.Snapshot{applying})
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "unknown allocation",
			setup: func(t *testing.T, h *memoryEvidenceHarness) string {
				h.writeArtifact(t, h.artifact("2"))
				return memoryPSWithoutSize(memoryTestModel, memoryTestDigest)
			},
		},
		{
			name: "unknown context",
			setup: func(t *testing.T, h *memoryEvidenceHarness) string {
				h.writeArtifact(t, h.artifact("2"))
				return memoryPSWithoutContext(memoryTestModel, memoryTestDigest)
			},
		},
		{
			name: "unsupported domain",
			setup: func(t *testing.T, h *memoryEvidenceHarness) string {
				document := h.artifact("2")
				domain := performancewire.MemoryDomainHostPhysical
				(*document.Records)[0].Envelope.Domain = &domain
				h.writeArtifact(t, document)
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "zero allocation stays measured zero",
			setup: func(t *testing.T, h *memoryEvidenceHarness) string {
				h.writeArtifact(t, h.artifact("2"))
				return memoryPS(memoryTestModel, memoryTestDigest, 0, 8192)
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newMemoryEvidenceHarness(t)
			raw := tc.setup(t, h)
			entry := memoryEntry(t, h.entries(raw))
			if tc.name == "zero allocation stays measured zero" {
				if memoryUpper(entry) == nil || entry.State != performancewire.MemoryStateQualified {
					t.Fatalf("a measured zero must remain qualified under a positive bound: %+v", entry)
				}
				if entry.Observed.AllocatedBytes == nil || *entry.Observed.AllocatedBytes != 0 {
					t.Fatalf("allocatedBytes = %v, want measured 0", entry.Observed.AllocatedBytes)
				}
				return
			}
			if memoryUpper(entry) != nil {
				t.Fatalf("upperBytes = %d, want nil for %s", *memoryUpper(entry), tc.name)
			}
		})
	}
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
			entry := memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))
			if memoryUpper(entry) != nil {
				t.Fatalf("upperBytes = %d, want nil when the document is rejected as a whole", *memoryUpper(entry))
			}
		})
	}
	t.Run("read error", func(t *testing.T) {
		h := newMemoryEvidenceHarness(t)
		if err := os.Mkdir(filepath.Join(h.ex.baseDir, memoryQualificationFile), 0o700); err != nil {
			t.Fatalf("create unreadable setup path: %v", err)
		}
		entry := memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))
		if memoryUpper(entry) != nil {
			t.Fatalf("upperBytes = %d, want nil after a read error", *memoryUpper(entry))
		}
	})
}

func TestMemoryEvidenceBindingMismatches(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string
	}{
		{
			name: "host",
			mutate: func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string {
				(*document.Records)[0].Source.HostUUID = memoryValue("host-other")
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "manager incarnation",
			mutate: func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string {
				(*document.Records)[0].Source.ManagerIncarnation = memoryValue("mgr-other")
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "engine run",
			mutate: func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string {
				h.runID = "run-new"
				h.ex.engines["ollama"].runID = h.runID
				h.ex.inventoryDigests["ollama"] = observedDigests{runID: h.runID, models: map[string]string{memoryTestModel: h.digest}, epoch: 9}
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "launch",
			mutate: func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string {
				h.launch = strings.Repeat("d", 64)
				h.ex.engines["ollama"].launchDigest = h.launch
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "model content",
			mutate: func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string {
				h.digest = strings.Repeat("e", 64)
				h.ex.inventoryDigests["ollama"] = observedDigests{runID: h.runID, models: map[string]string{memoryTestModel: h.digest}, epoch: 9}
				return memoryPS(memoryTestModel, h.digest, 9000, 8192)
			},
		},
		{
			name: "applied settings revision",
			mutate: func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string {
				next := h.settings
				next.Revision, next.AppliedRevision = 8, 8
				h.ex.settingsHub.Publish([]settings.Snapshot{next})
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "envelope context",
			mutate: func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string {
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 4096)
			},
		},
		{
			name: "copied RTX record on GTX host",
			mutate: func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string {
				(*document.Records)[0].Source.HostUUID = memoryValue("rtx-host")
				h.host = "gtx-host"
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
		{
			name: "same host new run",
			mutate: func(t *testing.T, h *memoryEvidenceHarness, document *memoryQualificationArtifact) string {
				(*document.Records)[0].Source.EngineRunID = memoryValue("run-old")
				return memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newMemoryEvidenceHarness(t)
			document := h.artifact("2")
			raw := tc.mutate(t, h, &document)
			h.writeArtifact(t, document)
			entry := memoryEntry(t, h.entries(raw))
			if memoryUpper(entry) != nil {
				t.Fatalf("upperBytes = %d, want nil for %s", *memoryUpper(entry), tc.name)
			}
		})
	}
}

func TestMemoryEvidenceOlderCompletionIsStale(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	raw := memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
	older := h.ex.captureMemoryEvidence()
	newer := h.ex.captureMemoryEvidence()

	first := memoryEntry(t, h.ex.memoryEvidenceEntries(newer, "ollama", json.RawMessage(raw)))
	if memoryUpper(first) == nil {
		t.Fatal("setup: the newer completion should qualify")
	}
	late := memoryEntry(t, h.ex.memoryEvidenceEntries(older, "ollama", json.RawMessage(raw)))
	if memoryUpper(late) != nil {
		t.Fatalf("an older completion published upper %d, want nil", *memoryUpper(late))
	}
	if !slicesContain(late.Uncertainty, performancewire.MemoryUncertaintyStale) {
		t.Fatalf("uncertainty = %v, want stale", late.Uncertainty)
	}
}

func TestMemoryEvidenceAgesAndClockUncertainty(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	original := h.writeArtifact(t, h.artifact("2"))
	raw := memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)
	capture := h.ex.captureMemoryEvidence()
	duplicateObservation := func() []performancewire.MemoryEvidenceEntry {
		return h.ex.memoryEvidenceEntries(capture, "ollama", json.RawMessage(raw))
	}

	first := memoryEntry(t, duplicateObservation())
	if first.Freshness.QualificationAgeMs == nil || *first.Freshness.QualificationAgeMs < 5*60*1000 {
		t.Fatalf("qualificationAgeMs = %v, want at least the measured age", first.Freshness.QualificationAgeMs)
	}
	if first.Freshness.SampleAgeMs == nil || *first.Freshness.SampleAgeMs != 0 {
		t.Fatalf("sampleAgeMs = %v, want 0", first.Freshness.SampleAgeMs)
	}

	h.clock = h.clock.Add(2 * time.Second)
	second := memoryEntry(t, duplicateObservation())
	if *second.Freshness.SampleAgeMs != 2000 {
		t.Fatalf("sampleAgeMs = %d, want 2000", *second.Freshness.SampleAgeMs)
	}
	if *second.Freshness.QualificationAgeMs <= *first.Freshness.QualificationAgeMs {
		t.Fatalf("qualification age did not advance: %d then %d", *first.Freshness.QualificationAgeMs, *second.Freshness.QualificationAgeMs)
	}

	// A duplicate artifact must not rejuvenate either age.
	h.writeArtifactBytes(t, original)
	h.clock = h.clock.Add(time.Second)
	duplicate := memoryEntry(t, duplicateObservation())
	if *duplicate.Freshness.QualificationAgeMs <= *second.Freshness.QualificationAgeMs {
		t.Fatalf("duplicate artifact rejuvenated qualification age: %d then %d", *second.Freshness.QualificationAgeMs, *duplicate.Freshness.QualificationAgeMs)
	}

	// A rollback makes every age unprovable and latches clock uncertainty.
	h.clock = h.clock.Add(-10 * time.Second)
	rolled := memoryEntry(t, duplicateObservation())
	if memoryUpper(rolled) != nil {
		t.Fatalf("upperBytes = %d, want nil after a clock rollback", *memoryUpper(rolled))
	}
	if !slicesContain(rolled.Uncertainty, performancewire.MemoryUncertaintyClockUncertain) {
		t.Fatalf("uncertainty = %v, want clock-uncertain", rolled.Uncertainty)
	}
	duplicateRolled := memoryEntry(t, duplicateObservation())
	if memoryUpper(duplicateRolled) != nil {
		t.Fatal("a duplicate artifact cleared latched clock uncertainty")
	}

	// A new, higher reviewed document can recover.
	h.clock = h.clock.Add(20 * time.Second)
	recovered := h.artifact("3")
	measured := h.clock.Add(-time.Second).UnixMilli()
	recovered.MeasuredAt = &measured
	h.writeArtifact(t, recovered)
	entry := memoryEntry(t, h.entries(raw))
	if memoryUpper(entry) == nil {
		t.Fatalf("a higher reviewed revision did not recover qualification: %+v", entry.Uncertainty)
	}
}

func TestMemoryEvidenceHighWaterWithdrawalAndRecovery(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	qualified := memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))
	if memoryUpper(qualified) == nil {
		t.Fatal("setup: expected a qualified entry")
	}

	t.Run("lower revision rejected", func(t *testing.T) {
		h.writeArtifact(t, h.artifact("1"))
		if got := memoryUpper(memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))); got != nil {
			t.Fatalf("upperBytes = %d, want nil for a lower revision", *got)
		}
	})

	t.Run("same revision different bytes rejected", func(t *testing.T) {
		document := h.artifact("2")
		(*document.Records)[0].Qualification = memoryTestQualification(20_000)
		h.writeArtifact(t, document)
		if got := memoryUpper(memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))); got != nil {
			t.Fatalf("upperBytes = %d, want nil for different bytes at the same revision", *got)
		}
	})

	h.writeArtifact(t, h.artifact("2"))
	exceeded := memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 11_000, 8192)))
	if memoryUpper(exceeded) != nil {
		t.Fatalf("upperBytes = %d, want nil after an exceedance", *memoryUpper(exceeded))
	}
	if !slicesContain(exceeded.Uncertainty, performancewire.MemoryUncertaintyBoundExceeded) {
		t.Fatalf("uncertainty = %v, want bound-exceeded", exceeded.Uncertainty)
	}

	t.Run("lower reading does not recover", func(t *testing.T) {
		if got := memoryUpper(memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))); got != nil {
			t.Fatalf("upperBytes = %d, want nil after withdrawal", *got)
		}
	})
	t.Run("deletion does not recover", func(t *testing.T) {
		h.removeArtifact(t)
		if got := memoryUpper(memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))); got != nil {
			t.Fatalf("upperBytes = %d, want nil after deletion", *got)
		}
	})
	t.Run("reappearance does not recover", func(t *testing.T) {
		h.writeArtifact(t, h.artifact("2"))
		if got := memoryUpper(memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))); got != nil {
			t.Fatalf("upperBytes = %d, want nil after reappearance", *got)
		}
	})
	t.Run("entry eviction does not recover", func(t *testing.T) {
		if got := memoryUpper(memoryEntry(t, h.entries(`{"models":[]}`))); got != nil {
			t.Fatalf("upperBytes = %d, want nil after eviction", *got)
		}
	})
	t.Run("higher reviewed revision recovers", func(t *testing.T) {
		document := h.artifact("3")
		(*document.Records)[0].Qualification = memoryTestQualification(12_000)
		h.writeArtifact(t, document)
		entry := memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))
		if memoryUpper(entry) == nil || *memoryUpper(entry) != 12_000 {
			t.Fatalf("upperBytes = %v, want 12000 recovery", memoryUpper(entry))
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
	entry := memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))
	if memoryUpper(entry) != nil {
		t.Fatalf("upperBytes = %d, want nil for an over-limit document", *memoryUpper(entry))
	}
}

func slicesContain(list []performancewire.MemoryUncertainty, want performancewire.MemoryUncertainty) bool {
	for _, item := range list {
		if item == want {
			return true
		}
	}
	return false
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

func TestMemoryEvidenceReviewRejectsReplacedContent(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	h.ex.inventoryMu.Lock()
	h.ex.inventoryDigests["ollama"].models[memoryTestModel] = strings.Repeat("d", 64)
	h.ex.inventoryMu.Unlock()
	entry := memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))
	if memoryUpper(entry) != nil {
		t.Fatal("old content qualification survived current inventory digest replacement")
	}
}

func TestMemoryEvidenceReviewRejectsAmbiguousAllocation(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	raw := strings.Replace(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192), `"size_vram":9000`, `"size_vram":20000,"size_vram":9000`, 1)
	entry := memoryEntry(t, h.entries(raw))
	if memoryUpper(entry) != nil {
		t.Fatal("duplicate allocation fields yielded a qualified upper")
	}
}

func TestMemoryEvidenceReviewDoesNotRelabelNewRunWithOldAuthority(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	if memoryUpper(memoryEntry(t, h.entries(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))) == nil {
		t.Fatal("setup did not qualify the original run")
	}
	const nextRun = "run-restarted"
	state := h.ex.engines["ollama"]
	state.mu.Lock()
	state.runID = nextRun
	state.mu.Unlock()
	h.ex.inventoryMu.Lock()
	digests := h.ex.inventoryDigests["ollama"]
	digests.runID = nextRun
	h.ex.inventoryDigests["ollama"] = digests
	h.ex.inventoryMu.Unlock()
	capture := h.ex.captureMemoryEvidence()
	entries := h.ex.memoryEvidenceEntries(capture, "ollama", json.RawMessage(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))
	entry := memoryEntry(t, h.ex.finalizeMemoryEvidence(capture, entries))
	if memoryUpper(entry) != nil {
		t.Fatal("old artifact qualified the new run")
	}
	if entry.Observed.AllocatedBytes != nil && (entry.Source.EngineRunID == nil || *entry.Source.EngineRunID != nextRun) {
		t.Fatal("new run's allocation was attributed to the old artifact's run")
	}
	if entry.Provenance.Scope != nil {
		t.Fatal("old run's operator attestation was inherited by the new run")
	}
}

func TestMemoryEvidenceNativeAllocationRejectsAmbiguousOrNonintegerValues(t *testing.T) {
	test := func(name, size string) {
		t.Run(name, func(t *testing.T) {
			h := newMemoryEvidenceHarness(t)
			h.writeArtifact(t, h.artifact("2"))
			raw := memoryPS(memoryTestModel, memoryTestDigest, size, 8192)
			entry := memoryEntry(t, h.entries(raw))
			if memoryUpper(entry) != nil || entry.Observed.AllocatedBytes != nil {
				t.Fatalf("invalid allocation accepted: %+v", entry)
			}
		})
	}
	test("negative", "-1")
	test("fractional", "1.5")
	test("overflow", "9223372036854775808")
	test("null", "null")
	test("quoted number", `"9000"`)
	t.Run("case-variant allocation duplicate", func(t *testing.T) {
		h := newMemoryEvidenceHarness(t)
		h.writeArtifact(t, h.artifact("2"))
		raw := strings.Replace(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192), `"size_vram":9000`, `"SIZE_VRAM":20000,"size_vram":9000`, 1)
		if memoryUpper(memoryEntry(t, h.entries(raw))) != nil {
			t.Fatal("case-variant duplicate qualified")
		}
	})
}

func TestMemoryEvidenceAliasedRowsCannotQualify(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	manifest := h.ex.engines["ollama"].manifest
	manifest.ModelNaming = engines.ModelNamingImpliedLatestTag
	const model = "llama3:latest"
	h.ex.inventoryDigests["ollama"] = observedDigests{runID: h.runID, models: map[string]string{model: h.digest}, epoch: 9}
	document := h.artifact("2")
	(*document.Records)[0].Source.Model = memoryValue(model)
	h.writeArtifact(t, document)
	raw := fmt.Sprintf(`{"models":[{"name":"llama3","model":"llama3:latest","digest":%q,"size_vram":9000,"context_length":8192},{"name":"llama3:latest","model":"llama3","digest":%q,"size_vram":9000,"context_length":8192}]}`, h.digest, h.digest)
	if memoryUpper(memoryEntry(t, h.entries(raw))) != nil {
		t.Fatal("normalized duplicate rows qualified")
	}
}

func TestMemoryEvidenceFinalFenceRechecksDocumentBytes(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	capture := h.ex.captureMemoryEvidence()
	entries := h.ex.memoryEvidenceEntries(capture, "ollama", json.RawMessage(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192)))
	if memoryUpper(memoryEntry(t, entries)) == nil {
		t.Fatal("positive control did not qualify")
	}
	h.removeArtifact(t)
	if memoryUpper(memoryEntry(t, h.ex.finalizeMemoryEvidence(capture, entries))) != nil {
		t.Fatal("deleted document survived final publication fence")
	}
}

func TestMemoryEvidencePublicationBoundsAndOrdering(t *testing.T) {
	entries := make([]performancewire.MemoryEvidenceEntry, 0, 140)
	for i := 139; i >= 0; i-- {
		entry := performancewire.MemoryEvidenceEntry{
			Engine: "ollama", Model: fmt.Sprintf("model-%03d", i), State: performancewire.MemoryStateUnknown,
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

func TestMemoryEvidenceReviewDelayedBufferedReadingAge(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	capture := h.ex.captureMemoryEvidence()
	raw := json.RawMessage(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192))
	// The response is already buffered. Delaying projection cannot make it
	// younger than the conservative pre-I/O capture anchor.
	h.clock = h.clock.Add(16 * time.Second)
	entries := h.ex.memoryEvidenceEntries(capture, "ollama", raw)
	entry := memoryEntry(t, h.ex.finalizeMemoryEvidence(capture, entries))
	if memoryUpper(entry) != nil {
		t.Fatalf("buffered reading older than validity window qualified with sample age %v ms", entry.Freshness.SampleAgeMs)
	}
	if entry.Freshness.SampleAgeMs != nil && *entry.Freshness.SampleAgeMs < 16000 {
		t.Fatalf("buffered reading was rejuvenated to %d ms", *entry.Freshness.SampleAgeMs)
	}
	if entry.Freshness.SampleAgeMs == nil || *entry.Freshness.SampleAgeMs != 16000 {
		t.Fatalf("sample age = %v, want known 16000 ms", entry.Freshness.SampleAgeMs)
	}
	if entry.Observed.AllocatedBytes == nil || *entry.Observed.AllocatedBytes != 9000 || entry.State != performancewire.MemoryStateAdvisory {
		t.Fatalf("delayed reading lost its advisory allocation: %+v", entry)
	}
}

func TestMemoryEvidenceFinalizationDelayCountsTowardReadingAge(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	capture := h.ex.captureMemoryEvidence()
	raw := json.RawMessage(memoryPS(memoryTestModel, memoryTestDigest, 9000, 8192))
	entries := h.ex.memoryEvidenceEntries(capture, "ollama", raw)
	if memoryUpper(memoryEntry(t, entries)) == nil {
		t.Fatal("fresh positive control did not qualify")
	}
	h.clock = h.clock.Add(16 * time.Second)
	entry := memoryEntry(t, h.ex.finalizeMemoryEvidence(capture, entries))
	if memoryUpper(entry) != nil || entry.Freshness.SampleAgeMs == nil || *entry.Freshness.SampleAgeMs != 16000 {
		t.Fatalf("finalization delay did not expire the reading: %+v", entry)
	}
	if !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyStale) {
		t.Fatalf("uncertainty = %v, want stale", entry.Uncertainty)
	}
}

func TestMemoryEvidenceReversedPreIOClockRemainsUnknown(t *testing.T) {
	h := newMemoryEvidenceHarness(t)
	h.writeArtifact(t, h.artifact("2"))
	capture := h.ex.captureMemoryEvidence()
	h.clock = h.clock.Add(-time.Second)
	entries := h.ex.memoryEvidenceEntries(capture, "ollama", json.RawMessage(memoryPS(memoryTestModel, memoryTestDigest, 0, 8192)))
	entry := memoryEntry(t, h.ex.finalizeMemoryEvidence(capture, entries))
	if memoryUpper(entry) != nil || entry.Freshness.SampleAgeMs != nil {
		t.Fatalf("reversed clock fabricated reading freshness: %+v", entry)
	}
	if entry.Observed.AllocatedBytes == nil || *entry.Observed.AllocatedBytes != 0 {
		t.Fatal("actual measured zero was lost when its age became unknown")
	}
	if !slicesContain(entry.Uncertainty, performancewire.MemoryUncertaintyClockUncertain) {
		t.Fatalf("uncertainty = %v, want clock-uncertain", entry.Uncertainty)
	}
}
