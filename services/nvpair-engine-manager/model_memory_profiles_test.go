// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
)

func memoryProfilesFixture(t *testing.T) (inspectionFixture, modelconfig.ClearStatedMemoryRequest) {
	t.Helper()
	f := newAppliedKeyFixture(t)
	base := t.TempDir()
	if err := os.Chmod(base, 0700); err != nil {
		t.Fatal(err)
	}
	f.ex.modelMemoryProfiles = newModelMemoryProfiles(base)
	f.ex.modelEstimates.run = func(context.Context, modelEstimateInput) modelconfig.Estimate {
		return modelconfig.Estimate{Status: "unknown"}
	}
	result := inspectAppliedKeyFixture(t, f)
	if result.ConfigurationKey == "" || result.Estimate.Status != "unknown" || len(result.Estimate.Domains) != 0 {
		t.Fatalf("independent key unavailable: %+v", result)
	}
	return f, modelconfig.ClearStatedMemoryRequest{Engine: f.request.Engine, ModelID: f.request.ModelID, ConfigurationKey: result.ConfigurationKey}
}

func measuredStoreFixture() modelconfig.MeasuredProfile {
	prompt, cached, generated := int32(880), int32(0), int32(128)
	first, end, overlap := 10.0, 100.0, true
	return modelconfig.MeasuredProfile{
		DiagnosticIdentity: modelconfig.DiagnosticIdentity{
			ConfigurationKey: strings.Repeat("a", 64), NativeBuildID: "sha256:" + strings.Repeat("b", 64), BuildInfo: "fixture-build", CPUModel: "fixture-cpu", TotalRAMBytes: "8192",
			GPUs: []modelconfig.DiagnosticGPU{}, Artifacts: []modelconfig.ArtifactSummary{{Role: "model", Identity: "fixture-artifact", Size: "100", Assurance: "declared"}},
			PerSlotContext: 1024, Slots: 1, Placement: []modelconfig.DiagnosticPlacement{{DomainID: "host", Kind: "host"}},
		},
		RunID: "complete-one", RecordedAt: "2026-11-07T00:00:00Z", Outcome: "completed", Notes: []string{},
		Pools: []modelconfig.MeasuredPool{{DomainID: "host", Kind: "host", BaselineAvailableBytes: "1000", AfterLoadAvailableBytes: "900", LowMinAvailableBytes: "800", HighMinAvailableBytes: "700", PeakUsedBytes: "300", Source: "fixture", Samples: 6, IntervalMS: 2000}},
		High:  &modelconfig.MeasuredPhase{Coverage: "complete", Overlapped: &overlap, OverlapWindowSampled: &overlap, WallMS: 100, Requests: []modelconfig.MeasuredRequest{{TargetPromptTokens: prompt, PromptTokens: &prompt, CachedTokens: &cached, GeneratedTokens: &generated, FirstTokenMS: &first, EndMS: &end}}},
	}
}

func TestModelMemoryProfilesMeasuredRetentionAndReopen(t *testing.T) {
	base := t.TempDir()
	if err := os.Chmod(base, 0700); err != nil {
		t.Fatal(err)
	}
	s := newModelMemoryProfiles(base)
	m := measuredStoreFixture()
	stated := modelconfig.MemoryProfile{ConfigurationKey: m.ConfigurationKey, NativeBuildID: m.NativeBuildID, Kind: "stated", RecordedAt: m.RecordedAt, HostBytes: "400"}
	if err := s.update("llamacpp", "model", m.ConfigurationKey, &stated); err != nil {
		t.Fatal(err)
	}
	for _, outcome := range []string{"completed", "failed"} {
		for _, stamp := range []string{"01", "02"} {
			candidate := m.Clone()
			candidate.RunID = outcome + stamp
			candidate.Outcome = outcome
			candidate.RecordedAt = "2026-11-07T00:00:" + stamp + "Z"
			if err := s.recordMeasured("llamacpp", "model", "profile", candidate); err != nil {
				t.Fatal(err)
			}
		}
	}
	reopened := newModelMemoryProfiles(base)
	if reopened.unavailable || len(reopened.records) != 3 {
		t.Fatalf("reopen lost retention buckets: unavailable=%v records=%+v", reopened.unavailable, reopened.records)
	}
	want := map[string]bool{"completed02": false, "failed02": false}
	for _, record := range reopened.records {
		if record.Profile.Kind == "stated" {
			if record.Profile.HostBytes != "400" {
				t.Fatal("measured write changed statement")
			}
			continue
		}
		if _, ok := want[record.Profile.Measured.RunID]; !ok {
			t.Fatalf("retained obsolete run %q", record.Profile.Measured.RunID)
		}
		want[record.Profile.Measured.RunID] = true
	}
	for runID, found := range want {
		if !found {
			t.Fatalf("lost run identity %q", runID)
		}
	}
}

func TestModelMemoryProfilesMeasuredCurrentnessRequiresFullEvidence(t *testing.T) {
	base := t.TempDir()
	if err := os.Chmod(base, 0700); err != nil {
		t.Fatal(err)
	}
	s := newModelMemoryProfiles(base)
	m := measuredStoreFixture()
	if err := s.recordMeasured("llamacpp", "model", "profile", m); err != nil {
		t.Fatal(err)
	}
	r := modelconfig.InspectResult{Engine: "llamacpp", ModelID: "model", ConfigurationKey: m.ConfigurationKey, ConfigurationPools: []string{"host"}, NativeBuildID: m.NativeBuildID, Artifacts: m.Artifacts}
	s.attachWithIdentity(&r, "profile", nil)
	if len(r.MemoryProfiles) != 1 || r.MemoryProfiles[0].Current {
		t.Fatal("key alone marked measurement current")
	}
	s.attachWithIdentity(&r, "profile", &m.DiagnosticIdentity)
	if !r.MemoryProfiles[0].Current {
		t.Fatal("full matching identity was refused")
	}
	r.MemoryProfiles[0].Measured.Pools[0].PeakUsedBytes = "1"
	s.attachWithIdentity(&r, "profile", &m.DiagnosticIdentity)
	if r.MemoryProfiles[0].Measured.Pools[0].PeakUsedBytes != "300" {
		t.Fatal("inspection mutated stored evidence")
	}
	changed := m.DiagnosticIdentity.Clone()
	changed.TotalRAMBytes = "16384"
	s.attachWithIdentity(&r, "profile", &changed)
	if r.MemoryProfiles[0].Current {
		t.Fatal("changed hardware retained validation")
	}
}
func TestModelMemoryProfilesFirstSaveWithoutEstimate(t *testing.T) {
	f, q := memoryProfilesFixture(t)
	set := modelconfig.SetStatedMemoryRequest{Engine: q.Engine, ModelID: q.ModelID, ConfigurationKey: q.ConfigurationKey, HostBytes: "12345"}
	result, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, &set)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.MemoryProfiles) != 1 || !result.MemoryProfiles[0].Current || result.MemoryProfiles[0].HostBytes != "12345" || result.Estimate.EstimateSource == nil || result.Estimate.EstimateSource.Kind != "user-stated" || result.Estimate.EstimateSource.Provider != "pair-user-statement" || len(result.Estimate.Domains) != 1 || result.Estimate.Domains[0].RequiredBytes != "12345" {
		t.Fatalf("statement/estimate: %+v", result)
	}
	result.MemoryProfiles[0].HostBytes = "99"
	next := inspectAppliedKeyFixture(t, f)
	if next.MemoryProfiles[0].HostBytes != "12345" {
		t.Fatal("reply aliases store")
	}
}

func TestModelMemoryProfilesLocalRecordWhilePaused(t *testing.T) {
	f, q := memoryProfilesFixture(t)
	f.ex.nodeProjection.set(nodeProjection{ManagementBlocked: true})
	set := modelconfig.SetStatedMemoryRequest{Engine: q.Engine, ModelID: q.ModelID, ConfigurationKey: q.ConfigurationKey, HostBytes: "1"}
	if _, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, &set); err != nil {
		t.Fatal(err)
	}
	if _, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, nil); err != nil {
		t.Fatal(err)
	}
}

func TestModelMemoryProfilesEarlierBuildRemainsHistory(t *testing.T) {
	f, q := memoryProfilesFixture(t)
	set := modelconfig.SetStatedMemoryRequest{Engine: q.Engine, ModelID: q.ModelID, ConfigurationKey: q.ConfigurationKey, HostBytes: "12345"}
	first, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, &set)
	if err != nil {
		t.Fatal(err)
	}
	if len(first.MemoryProfiles) != 1 || first.MemoryProfiles[0].NativeBuildID == "" || first.MemoryProfiles[0].NativeBuildID != first.NativeBuildID {
		t.Fatal("statement lost content-based native build provenance")
	}
	writeTestFile(t, filepath.Join(f.st.installDir, "libggml-base.dylib"), []byte("another private native build"))
	next := inspectAppliedKeyFixture(t, f)
	if next.ConfigurationKey == q.ConfigurationKey || len(next.MemoryProfiles) != 1 || next.MemoryProfiles[0].Current || next.MemoryProfiles[0].NativeBuildID != first.NativeBuildID || next.NativeBuildID == first.NativeBuildID {
		t.Fatal("earlier-build statement was lost or marked current")
	}
	if next.Estimate.EstimateSource != nil && (next.Estimate.EstimateSource.Kind == "user-stated" || next.Estimate.EstimateSource.Kind == "measured") {
		t.Fatal("earlier-build history lifted current-build estimate uncertainty")
	}
}
func TestModelMemoryProfilesExactPools(t *testing.T) {
	test := func(name string, host, device, shared string) {
		t.Run(name, func(t *testing.T) {
			f, q := memoryProfilesFixture(t)
			set := modelconfig.SetStatedMemoryRequest{Engine: q.Engine, ModelID: q.ModelID, ConfigurationKey: q.ConfigurationKey, HostBytes: host, DeviceBytes: device, SharedBytes: shared}
			if _, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, &set); err == nil {
				t.Fatal("accepted wrong pool set")
			}
			if len(f.ex.modelMemoryProfiles.records) != 0 {
				t.Fatal("rejected statement persisted")
			}
		})
	}
	test("missing host", "", "1", "")
	test("extra device", "1", "2", "")
	test("wrong shared pool", "", "", "1")
}
func TestModelMemoryProfilesStaleKey(t *testing.T) {
	f, q := memoryProfilesFixture(t)
	q.ConfigurationKey = strings.Repeat("f", 64)
	_, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, nil)
	if err == nil || rpcCode(err) != -32010 || !strings.Contains(err.Error(), "reinspect") {
		t.Fatalf("conflict: %v", err)
	}
	set := modelconfig.SetStatedMemoryRequest{Engine: q.Engine, ModelID: q.ModelID, ConfigurationKey: q.ConfigurationKey, HostBytes: "1"}
	if _, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, &set); rpcCode(err) != -32010 {
		t.Fatalf("stale set conflict: %v", err)
	}
}
func TestModelMemoryProfilesRestartSetAndClear(t *testing.T) {
	f, q := memoryProfilesFixture(t)
	s := f.ex.modelMemoryProfiles
	p := modelconfig.MemoryProfile{ConfigurationKey: q.ConfigurationKey, Kind: "stated", RecordedAt: "2026-10-30T01:00:00Z", HostBytes: "2048"}
	if err := s.update(q.Engine, q.ModelID, q.ConfigurationKey, &p); err != nil {
		t.Fatal(err)
	}
	restarted := newModelMemoryProfiles(filepath.Dir(s.path))
	if restarted.unavailable || len(restarted.records) != 1 || restarted.records[0].Profile.HostBytes != "2048" {
		t.Fatalf("restart: %+v", restarted)
	}
	if err := restarted.update(q.Engine, q.ModelID, q.ConfigurationKey, nil); err != nil {
		t.Fatal(err)
	}
	cleared := newModelMemoryProfiles(filepath.Dir(s.path))
	if cleared.unavailable || len(cleared.records) != 0 {
		t.Fatalf("clear restart: %+v", cleared)
	}
	info, err := os.Stat(s.path)
	if err != nil {
		t.Fatal(err)
	}
	// Windows reports read-only attributes through mode bits, not file ACLs.
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0600 {
		t.Fatalf("permissions: %v", info.Mode())
	}
}
func TestModelMemoryProfilesCorruptPreserved(t *testing.T) {
	for _, name := range []string{"invalid JSON", "oversize", "duplicate"} {
		t.Run(name, func(t *testing.T) {
			base := t.TempDir()
			if err := os.Chmod(base, 0700); err != nil {
				t.Fatal(err)
			}
			data := []byte("broken")
			if name == "oversize" {
				data = []byte(strings.Repeat(" ", modelMemoryProfilesBytes+1))
			}
			if name == "duplicate" {
				r := modelMemoryProfileRecord{Engine: "llamacpp", ModelID: "model", Profile: modelconfig.MemoryProfile{ConfigurationKey: strings.Repeat("a", 64), Kind: "stated", RecordedAt: "2026-10-30T00:00:00Z", HostBytes: "1"}}
				var err error
				data, err = json.Marshal([]modelMemoryProfileRecord{r, r})
				if err != nil {
					t.Fatal(err)
				}
			}
			path := filepath.Join(base, "model-memory-profiles.json")
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			s := newModelMemoryProfiles(base)
			r := modelconfig.InspectResult{}
			s.attach(&r)
			if !r.MemoryProfilesUnavailable {
				t.Fatal("corruption not reported")
			}
			if err := s.update("llamacpp", "model", strings.Repeat("a", 64), nil); err == nil {
				t.Fatal("corruption overwritten")
			}
			after, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if string(after) != string(data) {
				t.Fatal("corrupt file changed")
			}
		})
	}
}
func TestModelMemoryProfilesCurrentFirstHistoryBound(t *testing.T) {
	f, q := memoryProfilesFixture(t)
	s := f.ex.modelMemoryProfiles
	for i := 0; i < 270; i++ {
		p := modelconfig.MemoryProfile{ConfigurationKey: fmt.Sprintf("%064x", i), Kind: "stated", RecordedAt: fmt.Sprintf("2026-10-30T01:%02d:%02dZ", i/60, i%60), HostBytes: "1"}
		if err := s.update(q.Engine, q.ModelID, p.ConfigurationKey, &p); err != nil {
			t.Fatal(err)
		}
	}
	if len(s.records) != 256 {
		t.Fatalf("retained=%d", len(s.records))
	}
	current := s.records[0].Profile.ConfigurationKey
	r := modelconfig.InspectResult{Engine: q.Engine, ModelID: q.ModelID, ConfigurationKey: current, ConfigurationPools: []string{"host"}}
	s.attach(&r)
	if len(r.MemoryProfiles) != 8 || !r.MemoryProfiles[0].Current || r.MemoryProfiles[0].ConfigurationKey != current {
		t.Fatalf("history=%+v", r.MemoryProfiles)
	}
}

func TestModelMemoryProfilesReplyBudgetKeepsCurrent(t *testing.T) {
	f, q := memoryProfilesFixture(t)
	s := f.ex.modelMemoryProfiles
	for i := 0; i < 8; i++ {
		p := modelconfig.MemoryProfile{ConfigurationKey: fmt.Sprintf("%064x", i), Kind: "stated", RecordedAt: "2026-10-30T00:00:00Z", HostBytes: "1"}
		if err := s.update(q.Engine, q.ModelID, p.ConfigurationKey, &p); err != nil {
			t.Fatal(err)
		}
	}
	r := modelconfig.InspectResult{Engine: q.Engine, ModelID: q.ModelID, Status: "resolved", Estimate: modelconfig.Estimate{Status: "unknown"}, ConfigurationKey: s.records[0].Profile.ConfigurationKey, ConfigurationPools: []string{"host"}}
	a := modelconfig.ArtifactSummary{Role: "model", Identity: strings.Repeat("a", 256), Assurance: strings.Repeat("b", 256), Size: "1"}
	for {
		r.Artifacts = append(r.Artifacts, a)
		data, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) > modelconfig.MaxPayloadBytes-400 {
			r.Artifacts = r.Artifacts[:len(r.Artifacts)-1]
			break
		}
	}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	r.Reason = strings.Repeat("x", max(0, modelconfig.MaxPayloadBytes-len(data)-430))
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	s.attach(&r)
	if len(r.MemoryProfiles) == 0 || len(r.MemoryProfiles) >= 8 || !r.MemoryProfiles[0].Current {
		t.Fatalf("profile trimming: %+v", r.MemoryProfiles)
	}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestModelMemoryProfilesUnsupportedFilePreserved(t *testing.T) {
	base := t.TempDir()
	if err := os.Chmod(base, 0700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(base, "model-memory-profiles.json")
	target := filepath.Join(base, "missing.json")
	if err := os.Symlink(target, path); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("file-symlink fixture unavailable: %v", err)
		}
		t.Fatal(err)
	}
	s := newModelMemoryProfiles(base)
	p := modelconfig.MemoryProfile{ConfigurationKey: strings.Repeat("a", 64), Kind: "stated", RecordedAt: "2026-10-30T00:00:00Z", HostBytes: "1"}
	err := s.update("llamacpp", "model", p.ConfigurationKey, &p)
	info, statErr := os.Lstat(path)
	if statErr != nil {
		t.Fatal(statErr)
	}
	if !s.unavailable || err == nil || info.Mode()&os.ModeSymlink == 0 {
		t.Fatal("unsupported profile entry was treated as absent or replaced")
	}
	got, err := os.Readlink(path)
	if err != nil || got != target {
		t.Fatal("unsupported profile entry target changed")
	}
}

func TestModelMemoryProfilesLandedReplacementAdopted(t *testing.T) {
	f, q := memoryProfilesFixture(t)
	previous := engineFileReplacer
	t.Cleanup(func() { engineFileReplacer = previous })
	engineFileReplacer = durablefile.Replacer{SyncDir: func(string) error { return errors.New("synthetic directory sync failure") }}
	set := modelconfig.SetStatedMemoryRequest{Engine: q.Engine, ModelID: q.ModelID, ConfigurationKey: q.ConfigurationKey, HostBytes: "123"}
	_, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, &set)
	if !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("landed write outcome was erased: %v", err)
	}
	live := inspectAppliedKeyFixture(t, f)
	reopened := newModelMemoryProfiles(filepath.Dir(f.ex.modelMemoryProfiles.path))
	if live.MemoryProfilesUnavailable || len(live.MemoryProfiles) != 1 || live.MemoryProfiles[0].HostBytes != "123" || reopened.unavailable || len(reopened.records) != 1 {
		t.Fatal("landed record missing from live or restarted inspection")
	}
	engineFileReplacer = previous
	if _, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, nil); err != nil {
		t.Fatalf("landed outcome disabled subsequent recording: %v", err)
	}
}

func TestModelMemoryProfilesMutationReplyBudget(t *testing.T) {
	f, q := memoryProfilesFixture(t)
	padding := ""
	f.ex.modelEstimates.run = func(context.Context, modelEstimateInput) modelconfig.Estimate {
		return modelconfig.Estimate{Status: "unknown", Unknowns: []string{padding}}
	}
	f.ex.modelEstimates.entries, f.ex.modelEstimates.bytes = nil, 0
	initial := inspectAppliedKeyFixture(t, f)
	data, err := json.Marshal(initial)
	if err != nil {
		t.Fatal(err)
	}
	padding = strings.Repeat("x", modelconfig.MaxPayloadBytes-len(data)-80)
	f.ex.modelEstimates.entries, f.ex.modelEstimates.bytes = nil, 0
	before := inspectAppliedKeyFixture(t, f)
	if err := before.Validate(); err != nil {
		t.Fatal(err)
	}
	set := modelconfig.SetStatedMemoryRequest{Engine: q.Engine, ModelID: q.ModelID, ConfigurationKey: q.ConfigurationKey, HostBytes: "123"}
	result, err := f.ex.recordStatedMemory(keyTestMemoryContext(), q, &set)
	if err != nil {
		t.Fatalf("valid statement failed at reply budget: %v", err)
	}
	if err := result.Validate(); err != nil {
		t.Fatal(err)
	}
	reopened := newModelMemoryProfiles(filepath.Dir(f.ex.modelMemoryProfiles.path))
	if len(result.MemoryProfiles) != 1 || !result.MemoryProfiles[0].Current || result.MemoryProfiles[0].HostBytes != "123" || len(reopened.records) != 1 {
		t.Fatal("bounded response or persisted current statement missing")
	}
}
