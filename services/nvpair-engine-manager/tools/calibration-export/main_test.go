// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	"nvpair-engine-manager/internal/artifactbinding"
	"nvpair-shared/modelconfig"
)

func syntheticStore(t *testing.T) ([]byte, []namedArtifact, storedSample) {
	t.Helper()
	data, err := os.ReadFile("../../testdata/estimate-calibration/synthetic-dense-cpu.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture estimateCalibrationFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	write := func(v interface{}) {
		t.Helper()
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	text := func(s string) { write(uint64(len(s))); b.WriteString(s) }
	s := fixture.Facts.Shards[0]
	b.WriteString("GGUF")
	write(uint32(3))
	write(uint64(len(s.Tensors)))
	write(uint64(len(s.Numeric) + 1))
	text("general.architecture")
	write(uint32(8))
	text(s.Architecture)
	keys := make([]string, 0, len(s.Numeric))
	for k := range s.Numeric {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		text(k)
		write(uint32(4))
		write(uint32(s.Numeric[k].Bits))
	}
	for _, tensor := range s.Tensors {
		text(tensor.Name)
		write(uint32(len(tensor.Dimensions)))
		for _, d := range tensor.Dimensions {
			write(d)
		}
		write(tensor.Type)
		write(tensor.Offset)
	}
	b.Write(make([]byte, (1<<20)-b.Len()))
	path := filepath.Join(t.TempDir(), "synthetic.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	e, err := artifactbinding.Capture(path)
	if err != nil {
		t.Fatal(err)
	}
	binding, err := (artifactbinding.Access{}).Sample(context.Background(), e, false)
	if err != nil {
		t.Fatal(err)
	}
	summaries := artifactbinding.Summaries([]modelconfig.ArtifactBinding{{Role: "model", Path: binding.SourcePath, ResolvedTarget: binding.ResolvedTarget, Size: strconv.FormatInt(binding.SizeBytes, 10), MtimeNS: binding.MtimeNS, SampleSHA256: binding.SampleSHA256}})
	summaries[0].Assurance = "sampled-local-binding"
	opts := fixture.Allocation.Options.Clone()
	opts.GPU = &fixture.Allocation.GPU
	slots := fixture.Allocation.Slots
	opts.Slots = &slots
	opts.Context = &modelconfig.Context{Mode: "fixed", Tokens: 1024}
	lo, hi, err := modelconfig.BufferReportInterval("0.06")
	if err != nil {
		t.Fatal(err)
	}
	l, err := strconv.ParseInt(lo, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	h, err := strconv.ParseInt(hi, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	sample := storedSample{ModelIdentity: strings.Repeat("a", 64), Observation: modelconfig.BufferObservation{Status: "reported", SampleID: strings.Repeat("b", 64), RecordedAt: "2026-10-30T00:00:00Z", ConfigurationKey: strings.Repeat("c", 64), UpstreamPin: fixture.NativePin, BuildAssurance: "recognized-self-reported-pin", TopologyIdentity: strings.Repeat("d", 64), PerSlotContext: 1024, Slots: 1, CapturedOptions: &opts, Artifacts: summaries, Baseline: []modelconfig.BufferBaseline{{Component: "weights", Status: "estimated", Bytes: "65536"}}, Comparisons: []modelconfig.BufferComparison{{Component: "weights", Status: "compatible", EstimatedBytes: "65536", LowerBytes: lo, UpperBytes: hi, DeltaLowerBytes: fmt.Sprint(l - 65536), DeltaUpperBytes: fmt.Sprint(h - 65536)}}, Rows: []modelconfig.BufferReportRow{{Component: "model", Label: "CPU", Phase: "load", Sequence: "1", Role: "model", ReportedMiB: "0.06", Unit: "MiB", Precision: "0.01", SourceKind: "log-scraped", IntervalStatus: "bounded", LowerBytes: lo, UpperBytes: hi, PoolIdentity: strings.Repeat("e", 64), Coverage: "compatible"}}}}
	if err := sample.Observation.Validate(); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal([]storedSample{sample})
	if err != nil {
		t.Fatal(err)
	}
	return encoded, []namedArtifact{{"model", path}}, sample
}

func TestExportSyntheticStore(t *testing.T) {
	store, paths, sample := syntheticStore(t)
	fixtures, err := exportFixtures(context.Background(), bytes.NewReader(store), paths, "synthetic", "synthetic", artifactbinding.Access{})
	if err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 1 {
		t.Fatalf("fixtures=%d", len(fixtures))
	}
	f := fixtures[0]
	if f.ConfigurationKey != sample.Observation.ConfigurationKey || !reflect.DeepEqual(f.Artifacts, sample.Observation.Artifacts) || f.Class != "" || f.Backend != "cpu" || f.Role != "synthetic" || len(f.Observed) != 1 || f.Observed[0].Correspondence != "supported" || f.Observed[0].Comparison.LowerBytes != sample.Observation.Comparisons[0].LowerBytes {
		t.Fatalf("bad fixture: %+v", f)
	}
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(paths[0].Path)) || bytes.Contains(data, []byte("sampleSha256")) {
		t.Fatal("private file evidence leaked")
	}
	root := t.TempDir()
	storePath := filepath.Join(root, "store.json")
	if err := os.WriteFile(storePath, store, 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(root, "output")
	if err := run(context.Background(), storePath, out, "synthetic", "synthetic", paths); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(filepath.Join(out, "sample-000.json"))
	if err != nil {
		t.Fatal(err)
	}
	var written estimateCalibrationFixture
	if err := json.Unmarshal(data, &written); err != nil {
		t.Fatal(err)
	}
	if written.ConfigurationKey != f.ConfigurationKey || written.Provenance.Machine != "synthetic" {
		t.Fatalf("written fixture=%+v", written)
	}
}

func TestExportRejectsUnjoinableStore(t *testing.T) {
	test := func(name string, edit func(*storedSample, *[]namedArtifact)) {
		t.Run(name, func(t *testing.T) {
			_, paths, sample := syntheticStore(t)
			edit(&sample, &paths)
			store, err := json.Marshal([]storedSample{sample})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := exportFixtures(context.Background(), bytes.NewReader(store), paths, "synthetic", "synthetic", artifactbinding.Access{}); err == nil {
				t.Fatal("unjoinable sample accepted")
			}
		})
	}
	test("keyless", func(s *storedSample, _ *[]namedArtifact) { s.Observation.ConfigurationKey = "" })
	test("wrong identity", func(s *storedSample, _ *[]namedArtifact) {
		s.Observation.Artifacts[0].Identity = strings.Repeat("f", 64)
	})
	test("wrong size", func(s *storedSample, _ *[]namedArtifact) { s.Observation.Artifacts[0].Size = "123" })
	test("incomplete role set", func(s *storedSample, _ *[]namedArtifact) {
		a := s.Observation.Artifacts[0]
		a.Role = "draft"
		s.Observation.Artifacts = append(s.Observation.Artifacts, a)
	})
	test("missing selectors", func(_ *storedSample, p *[]namedArtifact) { *p = nil })
	test("no finite options", func(s *storedSample, _ *[]namedArtifact) { s.Observation.CapturedOptions = nil })
	test("source changed", func(_ *storedSample, p *[]namedArtifact) {
		if err := os.WriteFile((*p)[0].Path, []byte("changed"), 0600); err != nil {
			t.Fatal(err)
		}
	})
}

func TestExportRejectsCanceledRead(t *testing.T) {
	store, paths, _ := syntheticStore(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := exportFixtures(ctx, bytes.NewReader(store), paths, "synthetic", "synthetic", artifactbinding.Access{}); err == nil {
		t.Fatal("canceled export accepted")
	}
}

func TestExportDoesNotSupportNonCPUWeights(t *testing.T) {
	_, paths, sample := syntheticStore(t)
	sample.Observation.Rows[0].Label = "CUDA0"
	store, err := json.Marshal([]storedSample{sample})
	if err != nil {
		t.Fatal(err)
	}
	fixtures, err := exportFixtures(context.Background(), bytes.NewReader(store), paths, "synthetic", "synthetic", artifactbinding.Access{})
	if err != nil {
		t.Fatal(err)
	}
	if fixtures[0].Observed[0].Correspondence != "diagnostic" {
		t.Fatal("non-CPU weight row incorrectly became supported CPU weight evidence")
	}
}

type trackedExportFile struct {
	artifactbinding.File
	closed *int
}

func (f trackedExportFile) Close() error { *f.closed++; return f.File.Close() }

func TestExportRechecksAfterFactsAndCloses(t *testing.T) {
	store, paths, _ := syntheticStore(t)
	opens, closes := 0, 0
	access := artifactbinding.Access{Open: func(p string) (artifactbinding.File, error) {
		opens++
		if opens == 3 {
			if err := os.WriteFile(p, []byte("changed after decode"), 0600); err != nil {
				t.Fatal(err)
			}
		}
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		return trackedExportFile{f, &closes}, nil
	}}
	fixtures, err := exportFixtures(context.Background(), bytes.NewReader(store), paths, "synthetic", "synthetic", access)
	if err == nil || fixtures != nil || opens != closes {
		t.Fatalf("post-facts drift accepted or handles leaked: fixtures=%d err=%v opens=%d closes=%d", len(fixtures), err, opens, closes)
	}
}

func TestExportReadsBoundedHistoryBeyondOneWirePayload(t *testing.T) {
	_, paths, sample := syntheticStore(t)
	var samples []storedSample
	for i := range 128 {
		copy := sample
		copy.Observation.SampleID = fmt.Sprintf("%064x", i+1)
		samples = append(samples, copy)
	}
	store, err := json.Marshal(samples)
	if err != nil {
		t.Fatal(err)
	}
	if len(store) <= modelconfig.MaxPayloadBytes {
		t.Fatal("setup did not exceed a single wire payload")
	}
	fixtures, err := exportFixtures(context.Background(), bytes.NewReader(store), paths, "synthetic", "synthetic", artifactbinding.Access{})
	if err != nil || len(fixtures) != len(samples) {
		t.Fatalf("bounded history rejected: fixtures=%d samples=%d err=%v", len(fixtures), len(samples), err)
	}
}
