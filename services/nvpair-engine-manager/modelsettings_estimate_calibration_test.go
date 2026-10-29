// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"nvpair-shared/gguffacts"
	"nvpair-shared/modelconfig"
)

// Test-only fixture contract shared in shape with the standalone exporter;
// this is deliberately not a new manager startup or RPC surface.
type estimateCalibrationFixture struct {
	Version          int                           `json:"version"`
	Role             string                        `json:"role"`
	Class            string                        `json:"class,omitempty"`
	Backend          string                        `json:"backend"`
	NativePin        string                        `json:"nativePin"`
	ConfigurationKey string                        `json:"configurationKey"`
	Facts            gguffacts.Facts               `json:"facts"`
	Artifacts        []modelconfig.ArtifactSummary `json:"artifacts"`
	Allocation       estimateCalibrationAllocation `json:"allocation"`
	Sample           struct {
		ConfigurationKey string                        `json:"configurationKey"`
		Artifacts        []modelconfig.ArtifactSummary `json:"artifacts"`
	} `json:"sample"`
	Observed   []estimateCalibrationInterval `json:"observed"`
	PoolPeaks  []estimateCalibrationPeak     `json:"poolPeaks,omitempty"`
	Provenance struct {
		Machine string `json:"machine"`
		Date    string `json:"date"`
		Build   string `json:"build"`
	} `json:"provenance"`
}

// Only finite numerical/enum/bool options, no paths or native args/digests.
type estimateCalibrationAllocation struct {
	PerSlotContext int32                     `json:"perSlotContext"`
	TotalContext   int32                     `json:"totalContext"`
	Slots          int32                     `json:"slots"`
	GPU            modelconfig.GPU           `json:"gpu"`
	Options        modelconfig.RequestedLoad `json:"options"`
	SWAFull        *bool                     `json:"swaFull,omitempty"`
}

type estimateCalibrationInterval struct {
	Correspondence string                       `json:"correspondence"`
	Comparison     modelconfig.BufferComparison `json:"comparison"`
}

type estimateCalibrationPeak struct {
	Pool           string `json:"pool"`
	Bytes          string `json:"bytes"`
	Correspondence string `json:"correspondence"`
}

func (f estimateCalibrationFixture) input() modelEstimateInput {
	unified := false
	return modelEstimateInput{Facts: f.Facts, Bindings: f.Artifacts,
		Allocation: &modelconfig.ResolvedLoad{PerSlotContext: f.Allocation.PerSlotContext,
			TotalContext: f.Allocation.TotalContext, Slots: f.Allocation.Slots, KVUnified: &unified,
			GPU: f.Allocation.GPU, Options: f.Allocation.Options},
		BuildID: f.Provenance.Build, Support: modelEstimatorSupport, SWAFull: f.Allocation.SWAFull}
}

var estimateCalibrationDigest = regexp.MustCompile(`^[0-9a-f]{64}$`)
var estimateCalibrationMachine = regexp.MustCompile(`^(synthetic|pc3|msi)$`)
var estimateCalibrationTensor = regexp.MustCompile(`^(token_embd|output|output_norm|blk\.[0-9]+\.(attn_q|attn_k|attn_v|attn_output|attn_norm|ffn_norm|ffn_gate|ffn_up|ffn_down|ffn_gate_exps|ffn_up_exps|ffn_down_exps))\.(weight|bias)$`)

func readEstimateCalibration(r io.Reader) (estimateCalibrationFixture, error) {
	var f estimateCalibrationFixture
	data, err := io.ReadAll(io.LimitReader(r, (8<<20)+1))
	if err != nil {
		return f, err
	}
	if len(data) > 8<<20 {
		return f, fmt.Errorf("fixture exceeds byte budget")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&f); err != nil {
		return f, fmt.Errorf("fixture JSON rejected")
	}
	if err := d.Decode(new(json.RawMessage)); err != io.EOF {
		return f, fmt.Errorf("fixture trailing JSON")
	}
	// Exporters need not duplicate the private estimator's classifier.
	if f.Class == "" {
		m, reason := modelEstimateAdapt(context.Background(), f.Facts)
		if reason != "" {
			return f, fmt.Errorf("fixture source facts")
		}
		f.Class = modelEstimateClass(m)
	}
	return f, f.validate()
}

func (f estimateCalibrationFixture) validate() error {
	if f.Version != 1 || !slices.Contains([]string{"synthetic", "fit", "check"}, f.Role) {
		return fmt.Errorf("fixture version/role")
	}
	if !slices.Contains([]string{"cpu", "cuda", "metal", "vulkan", "hip"}, f.Backend) || f.NativePin != modelBufferPin {
		return fmt.Errorf("fixture backend/native pin")
	}
	if !estimateCalibrationMachine.MatchString(f.Provenance.Machine) || !modelBufferBuildSupported(f.Provenance.Build) {
		return fmt.Errorf("fixture provenance machine/build")
	}
	if _, err := time.Parse("2006-01-02", f.Provenance.Date); err != nil {
		return fmt.Errorf("fixture provenance date")
	}
	if !estimateCalibrationDigest.MatchString(f.ConfigurationKey) || f.ConfigurationKey != f.Sample.ConfigurationKey || !reflect.DeepEqual(f.Artifacts, f.Sample.Artifacts) {
		return fmt.Errorf("fixture configuration/artifact join")
	}
	if len(f.Artifacts) != len(f.Facts.Shards) {
		return fmt.Errorf("fixture shard/artifact join")
	}
	for i, a := range f.Artifacts {
		role := map[gguffacts.Role]string{gguffacts.RoleMain: "model", gguffacts.RoleDraft: "draft", gguffacts.RoleCompanion: "companion"}[f.Facts.Shards[i].Role]
		size, err := strconv.ParseUint(a.Size, 10, 63)
		if role == "" || a.Role != role || !estimateCalibrationDigest.MatchString(a.Identity) || err != nil || size == 0 || a.Assurance != "sampled-local-binding" {
			return fmt.Errorf("fixture artifact summary")
		}
	}
	if reason := modelEstimateBounds(context.Background(), f.input()); reason != "" {
		return fmt.Errorf("fixture exceeds estimator bounds")
	}
	if f.Allocation.GPU.Mode != "cpu" && (f.Allocation.GPU.Mode != "layers" || f.Allocation.GPU.Layers == nil || *f.Allocation.GPU.Layers < 0) {
		return fmt.Errorf("fixture finite GPU allocation")
	}
	if f.Allocation.GPU.Mode == "cpu" && (f.Backend != "cpu" || f.Allocation.GPU.Layers != nil) || f.Allocation.GPU.Mode == "layers" && f.Backend == "cpu" {
		return fmt.Errorf("fixture backend/allocation mismatch")
	}
	// Facts are decoded by gguffacts, never by another metadata parser. Reject
	// retained strings not in this deliberately narrow export allowlist. Unknown
	// metadata/tensor names require a reviewed schema extension, never passthrough.
	for _, s := range f.Facts.Shards {
		if !slices.Contains([]string{"llama", "qwen2", "qwen3", "qwen2moe", "qwen3moe"}, s.Architecture) || len(s.Templates) != 0 {
			return fmt.Errorf("fixture facts privacy allowlist")
		}
		for key := range s.Numeric {
			field := strings.TrimPrefix(key, s.Architecture+".")
			if key == field || !slices.Contains([]string{"block_count", "embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv", "vocab_size", "expert_count", "expert_used_count", "expert_feed_forward_length", "expert_shared_count"}, field) {
				return fmt.Errorf("fixture numeric privacy allowlist")
			}
		}
		for _, tensor := range s.Tensors {
			if !estimateCalibrationTensor.MatchString(tensor.Name) {
				return fmt.Errorf("fixture tensor privacy allowlist")
			}
		}
	}
	m, reason := modelEstimateAdapt(context.Background(), f.Facts)
	if reason != "" || f.Class != modelEstimateClass(m) {
		return fmt.Errorf("fixture source class mismatch")
	}
	if len(f.Observed) == 0 || len(f.Observed) > len(modelconfig.MemoryComponentNames) || len(f.PoolPeaks) > 2 {
		return fmt.Errorf("fixture observation budget")
	}
	seen := map[string]bool{}
	for _, o := range f.Observed {
		c := o.Comparison
		if seen[c.Component] || !slices.Contains(modelconfig.MemoryComponentNames, c.Component) {
			return fmt.Errorf("fixture component")
		}
		seen[c.Component] = true
		if _, _, err := estimateCalibrationIntervalBounds(c.LowerBytes, c.UpperBytes); err != nil {
			return err
		}
		if c.EstimatedBytes != "" || c.DeltaLowerBytes != "" || c.DeltaUpperBytes != "" {
			return fmt.Errorf("fixture comparison contains redundant estimates")
		}
		if o.Correspondence == "supported" {
			if c.Component != "weights" || c.Status != "compatible" || f.Backend != "cpu" || slices.ContainsFunc(f.Artifacts, func(a modelconfig.ArtifactSummary) bool { return a.Role != "model" }) {
				return fmt.Errorf("fixture unsupported correspondence component=%s class=%s", c.Component, f.Class)
			}
		} else if o.Correspondence != "diagnostic" || !slices.Contains([]string{"unknown", "compatible"}, c.Status) {
			return fmt.Errorf("fixture correspondence")
		}
	}
	for _, p := range f.PoolPeaks {
		if !slices.Contains([]string{"host", "device", "shared"}, p.Pool) || p.Correspondence != "diagnostic" {
			return fmt.Errorf("fixture peak is diagnostic only")
		}
		if _, _, err := estimateCalibrationIntervalBounds(p.Bytes, p.Bytes); err != nil {
			return err
		}
	}
	return nil
}

func estimateCalibrationIntervalBounds(low, high string) (uint64, uint64, error) {
	l, le := strconv.ParseUint(low, 10, 64)
	h, he := strconv.ParseUint(high, 10, 64)
	if le != nil || he != nil || l > h {
		return 0, 0, fmt.Errorf("fixture invalid observed interval")
	}
	return l, h, nil
}

type estimateCalibrationReport struct {
	Component, Class, Correspondence, Basis string
	ErrorLow, ErrorHigh                     string // observed minus expected, never unsigned subtraction
	Inside                                  bool
}

func reportEstimateCalibration(f estimateCalibrationFixture) ([]estimateCalibrationReport, error) {
	if err := f.validate(); err != nil {
		return nil, err
	}
	var state modelEstimateState
	estimate := state.estimate(context.Background(), f.input())
	if err := estimate.Validate(); err != nil {
		return nil, err
	}
	var reports []estimateCalibrationReport
	for _, o := range f.Observed {
		var expected, low, high uint64
		basis := "unpriced"
		found := false
		for _, c := range estimate.Components {
			if c.Name != o.Comparison.Component || c.Bytes == "" {
				continue
			}
			n, err := strconv.ParseUint(c.Bytes, 10, 64)
			if err != nil {
				return nil, err
			}
			l, h, err := estimateCalibrationIntervalBounds(c.LowBytes, c.HighBytes)
			if err != nil {
				return nil, err
			}
			expected = modelEstimateAdd(expected, n)
			low = modelEstimateAdd(low, l)
			high = modelEstimateAdd(high, h)
			basis, found = c.Basis, true
		}
		l, h, err := estimateCalibrationIntervalBounds(o.Comparison.LowerBytes, o.Comparison.UpperBytes)
		if err != nil {
			return nil, err
		}
		delta := func(n uint64) string {
			return new(big.Int).Sub(new(big.Int).SetUint64(n), new(big.Int).SetUint64(expected)).String()
		}
		r := estimateCalibrationReport{Component: o.Comparison.Component, Class: f.Class, Correspondence: o.Correspondence, Basis: basis, Inside: found && l >= low && h <= high}
		if found {
			r.ErrorLow, r.ErrorHigh = delta(l), delta(h)
		}
		reports = append(reports, r)
		_, calibrated := modelEstimateCalibrations[modelEstimateCalibrationKey{r.Component, f.Class, f.Backend}]
		if f.Role != "synthetic" && o.Correspondence == "supported" && calibrated && (basis != "calibrated" || !r.Inside) {
			return reports, fmt.Errorf("out of band component=%s class=%s", r.Component, r.Class)
		}
	}
	return reports, nil
}

func validateEstimateCalibrationEvidence(fixtures map[string]estimateCalibrationFixture) error {
	for key, band := range modelEstimateCalibrations {
		fail := func(reason string) error {
			return fmt.Errorf("%s component=%s class=%s backend=%s", reason, key.Component, key.Class, key.Backend)
		}
		if band.LowDen == 0 || band.HighDen == 0 || band.LowNum > band.LowDen || band.HighNum < band.HighDen {
			return fail("invalid band")
		}
		eligible := func(f estimateCalibrationFixture, role string) bool {
			if f.Role != role || f.Provenance.Machine == "synthetic" || f.Class != key.Class || f.Backend != key.Backend || f.validate() != nil {
				return false
			}
			return slices.ContainsFunc(f.Observed, func(o estimateCalibrationInterval) bool {
				return o.Comparison.Component == key.Component && o.Correspondence == "supported"
			})
		}
		seen := map[string]bool{}
		for _, name := range band.Provenance {
			f, exists := fixtures[name]
			if seen[name] || strings.HasPrefix(name, "synthetic-") || !exists || !eligible(f, "fit") {
				return fail("ineligible fit evidence")
			}
			seen[name] = true
			if _, err := reportEstimateCalibration(f); err != nil {
				return err
			}
		}
		if len(seen) < 3 {
			return fail("requires at least three fit fixtures")
		}
		checks := false
		for name, f := range fixtures {
			if !strings.HasPrefix(name, "synthetic-") && eligible(f, "check") {
				checks = true
				if _, err := reportEstimateCalibration(f); err != nil {
					return err
				}
			}
		}
		if !checks {
			return fail("missing held-out check coverage")
		}
	}
	return nil
}

func loadEstimateCalibrationFixtures(t *testing.T) map[string]estimateCalibrationFixture {
	t.Helper()
	paths, err := filepath.Glob("testdata/estimate-calibration/*.json")
	if err != nil || len(paths) == 0 {
		t.Fatalf("fixture inventory: paths=%v err=%v", paths, err)
	}
	fixtures := map[string]estimateCalibrationFixture{}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		f, err := readEstimateCalibration(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if strings.HasPrefix(filepath.Base(path), "synthetic-") && f.Role != "synthetic" {
			t.Fatalf("%s: checked-in synthetic fixture cannot be fit/check evidence", path)
		}
		fixtures[filepath.Base(path)] = f
	}
	return fixtures
}

func TestModelSettingsEstimateCalibration(t *testing.T) {
	fixtures := loadEstimateCalibrationFixtures(t)
	if err := validateEstimateCalibrationEvidence(fixtures); err != nil {
		t.Fatal(err)
	}
	for name, f := range fixtures {
		t.Run(name, func(t *testing.T) {
			reports, err := reportEstimateCalibration(f)
			if err != nil {
				t.Fatal(err)
			}
			for _, r := range reports {
				t.Logf("component=%s class=%s correspondence=%s basis=%s signed-error=[%s,%s] inside=%t", r.Component, r.Class, r.Correspondence, r.Basis, r.ErrorLow, r.ErrorHigh, r.Inside)
			}
			for _, p := range f.PoolPeaks {
				t.Logf("diagnostic pool=%s peak=%s (no component oracle)", p.Pool, p.Bytes)
			}
		})
	}
}

// Construct only disposable GGUF headers and zero tensor storage. Decode is the
// sole metadata reader; this writer is not an exporter or binding sampler.
func TestModelSettingsEstimateCalibrationDecodedFacts(t *testing.T) {
	fixtures := loadEstimateCalibrationFixtures(t)
	multi := fixtures["synthetic-dense-cpu.json"]
	second := multi.Facts.Shards[0]
	second.Tensors = slices.Clone(second.Tensors)
	second.Tensors[0].Name = "output.weight"
	multi.Facts.Shards = append(slices.Clone(multi.Facts.Shards), second)
	artifact := multi.Artifacts[0]
	artifact.Identity = strings.Repeat("d", 64)
	multi.Artifacts = append(slices.Clone(multi.Artifacts), artifact)
	multi.Sample.Artifacts = slices.Clone(multi.Artifacts)
	fixtures["constructed-multishard"] = multi
	for name, f := range fixtures {
		t.Run(name, func(t *testing.T) {
			if _, err := reportEstimateCalibration(f); err != nil {
				t.Fatal(err)
			}
			var artifacts []gguffacts.Artifact
			for i, s := range f.Facts.Shards {
				var b bytes.Buffer
				write := func(v interface{}) {
					t.Helper()
					if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
						t.Fatal(err)
					}
				}
				text := func(v string) { write(uint64(len(v))); b.WriteString(v) }
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
					n := s.Numeric[k]
					if n.Type != 4 || n.Array != nil || n.Bits > 1<<32-1 {
						t.Fatal("synthetic writer accepts only uint32 scalars")
					}
					text(k)
					write(uint32(4))
					write(uint32(n.Bits))
				}
				for _, tensor := range s.Tensors {
					text(tensor.Name)
					write(uint32(len(tensor.Dimensions)))
					for _, dim := range tensor.Dimensions {
						write(dim)
					}
					write(tensor.Type)
					write(tensor.Offset)
				}
				size, err := strconv.ParseInt(f.Artifacts[i].Size, 10, 64)
				if err != nil {
					t.Fatal(err)
				}
				b.Write(make([]byte, int(size)-b.Len()))
				artifacts = append(artifacts, gguffacts.Artifact{Role: s.Role, Reader: bytes.NewReader(b.Bytes()), Size: size})
			}
			facts, err := gguffacts.Decode(context.Background(), artifacts)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(facts.Shards, f.Facts.Shards) {
				t.Fatalf("decoded shards differ: got %+v want %+v", facts.Shards, f.Facts.Shards)
			}
		})
	}
}

func calibrationConstructedEvidence(t *testing.T) (map[string]estimateCalibrationFixture, modelEstimateCalibrationKey) {
	t.Helper()
	f := loadEstimateCalibrationFixtures(t)["synthetic-dense-cpu.json"]
	// Invented provenance only for unit eligibility checks. Nothing is saved,
	// measured, loaded on PC3, or offered as production calibration evidence.
	f.Provenance.Machine = "pc3"
	fixtures := map[string]estimateCalibrationFixture{}
	for _, name := range []string{"fit-a", "fit-b", "fit-c", "check-a"} {
		copy := f
		copy.Observed = append([]estimateCalibrationInterval(nil), f.Observed[:1]...)
		copy.Role = "fit"
		if name == "check-a" {
			copy.Role = "check"
		}
		fixtures[name] = copy
	}
	key := modelEstimateCalibrationKey{"weights", "dense", "cpu"}
	prior := modelEstimateCalibrations
	modelEstimateCalibrations = map[modelEstimateCalibrationKey]modelEstimateCalibration{key: {LowNum: 3, LowDen: 4, HighNum: 3, HighDen: 2, Provenance: []string{"fit-a", "fit-b", "fit-c"}}}
	t.Cleanup(func() { modelEstimateCalibrations = prior })
	return fixtures, key
}

func TestModelSettingsEstimateCalibrationEmittedWeights(t *testing.T) {
	f := loadEstimateCalibrationFixtures(t)["synthetic-dense-cpu.json"]
	before := estimateTestComponent(t, estimateFixtureResult(t, f.input()), "weights")
	fixtures, _ := calibrationConstructedEvidence(t)
	if err := validateEstimateCalibrationEvidence(fixtures); err != nil {
		t.Fatal(err)
	}
	after := estimateTestComponent(t, estimateFixtureResult(t, f.input()), "weights")
	if before.Basis != "derived" || after.Basis != "calibrated" || after.LowBytes != "49152" || after.HighBytes != "98304" || after.LowBytes == before.LowBytes || after.HighBytes == before.HighBytes || after.Bytes != before.Bytes {
		t.Fatalf("injected band did not change emitted weight range: before=%+v after=%+v", before, after)
	}
}

func TestModelSettingsEstimateCalibrationUncalibratedSignedError(t *testing.T) {
	f := loadEstimateCalibrationFixtures(t)["synthetic-dense-cpu.json"]
	reports, err := reportEstimateCalibration(f)
	if err != nil {
		t.Fatal(err)
	}
	r := reports[0]
	if r.Component != "weights" || r.Basis != "derived" || r.Inside || r.ErrorLow != "-7865" || r.ErrorHigh != "2622" {
		t.Fatalf("uncalibrated signed interval: %+v", r)
	}
	low, high, err := modelconfig.BufferReportInterval("0.06")
	if err != nil {
		t.Fatal(err)
	}
	if f.Observed[0].Comparison.LowerBytes != low || f.Observed[0].Comparison.UpperBytes != high {
		t.Fatal("synthetic supported interval does not preserve native outward rounding")
	}
}

func TestModelSettingsEstimateCalibrationRejectsInvalidEvidence(t *testing.T) {
	test := func(name, want string, edit func(map[string]estimateCalibrationFixture, modelEstimateCalibrationKey)) {
		t.Run(name, func(t *testing.T) {
			fixtures, key := calibrationConstructedEvidence(t)
			edit(fixtures, key)
			err := validateEstimateCalibrationEvidence(fixtures)
			if err == nil || !strings.Contains(err.Error(), want) || !strings.Contains(err.Error(), "component=weights") || !strings.Contains(err.Error(), "class=dense") {
				t.Fatalf("got %v, want %s naming component/class", err, want)
			}
		})
	}
	test("out of band check", "out of band", func(fs map[string]estimateCalibrationFixture, _ modelEstimateCalibrationKey) {
		f := fs["check-a"]
		f.Observed[0].Comparison.LowerBytes = "9000000"
		f.Observed[0].Comparison.UpperBytes = "9000001"
		fs["check-a"] = f
	})
	test("two fit fixtures", "three fit", func(_ map[string]estimateCalibrationFixture, key modelEstimateCalibrationKey) {
		b := modelEstimateCalibrations[key]
		b.Provenance = b.Provenance[:2]
		modelEstimateCalibrations[key] = b
	})
	test("diagnostic fit", "ineligible fit", func(fs map[string]estimateCalibrationFixture, _ modelEstimateCalibrationKey) {
		f := fs["fit-a"]
		f.Observed[0].Correspondence = "diagnostic"
		fs["fit-a"] = f
	})
	test("synthetic fit", "ineligible fit", func(fs map[string]estimateCalibrationFixture, _ modelEstimateCalibrationKey) {
		f := fs["fit-a"]
		f.Role = "synthetic"
		fs["fit-a"] = f
	})
	test("synthetic provenance", "ineligible fit", func(fs map[string]estimateCalibrationFixture, _ modelEstimateCalibrationKey) {
		f := fs["fit-a"]
		f.Provenance.Machine = "synthetic"
		fs["fit-a"] = f
	})
	test("missing check", "missing held-out", func(fs map[string]estimateCalibrationFixture, _ modelEstimateCalibrationKey) { delete(fs, "check-a") })
	test("duplicate fit", "ineligible fit", func(_ map[string]estimateCalibrationFixture, key modelEstimateCalibrationKey) {
		b := modelEstimateCalibrations[key]
		b.Provenance[2] = b.Provenance[0]
		modelEstimateCalibrations[key] = b
	})
}

func TestModelSettingsEstimateCalibrationRejectsInvalidFixtures(t *testing.T) {
	test := func(name, want string, edit func(*estimateCalibrationFixture)) {
		t.Run(name, func(t *testing.T) {
			f := loadEstimateCalibrationFixtures(t)["synthetic-dense-cpu.json"]
			edit(&f)
			data, err := json.Marshal(f)
			if err != nil {
				t.Fatal(err)
			}
			_, err = readEstimateCalibration(bytes.NewReader(data))
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("got %v, want %s", err, want)
			}
		})
	}
	test("configuration join", "configuration/artifact join", func(f *estimateCalibrationFixture) { f.Sample.ConfigurationKey = strings.Repeat("e", 64) })
	test("artifact join", "configuration/artifact join", func(f *estimateCalibrationFixture) { f.Sample.Artifacts[0].Identity = strings.Repeat("e", 64) })
	test("wrong source class", "source class", func(f *estimateCalibrationFixture) { f.Class = "moe" })
	test("unbounded native mode", "finite GPU", func(f *estimateCalibrationFixture) { f.Allocation.GPU.Mode = "auto" })
	test("untrusted machine text", "provenance", func(f *estimateCalibrationFixture) { f.Provenance.Machine = "/private/secret" })
	test("untrusted numeric metadata", "numeric privacy", func(f *estimateCalibrationFixture) { f.Facts.Shards[0].Numeric["private.name"] = gguffacts.Numeric{} })
	test("untrusted tensor name", "tensor privacy", func(f *estimateCalibrationFixture) { f.Facts.Shards[0].Tensors[0].Name = "/private/secret" })
	test("template digest excluded", "facts privacy", func(f *estimateCalibrationFixture) {
		f.Facts.Shards[0].Templates = []gguffacts.Template{{Key: "private"}}
	})
	test("KV cannot be supported", "unsupported correspondence", func(f *estimateCalibrationFixture) {
		f.Observed[0].Comparison.Component = "attention-cache"
		f.Observed = f.Observed[:1]
	})
	test("unknown comparison cannot be supported", "unsupported correspondence", func(f *estimateCalibrationFixture) { f.Observed[0].Comparison.Status = "unknown" })
	test("peak cannot be supported", "diagnostic only", func(f *estimateCalibrationFixture) { f.PoolPeaks[0].Correspondence = "supported" })
	test("reversed interval", "invalid observed interval", func(f *estimateCalibrationFixture) { f.Observed[0].Comparison.LowerBytes = "9999999" })
}

func TestModelSettingsEstimateCalibrationJSONBoundary(t *testing.T) {
	for _, data := range []string{`{"path":"private"}`, `{}`, `{} {}`, strings.Repeat(" ", (8<<20)+1)} {
		if _, err := readEstimateCalibration(strings.NewReader(data)); err == nil {
			t.Fatal("accepted invalid or oversized fixture")
		}
	}
}

func TestModelSettingsEstimateCalibrationDerivesExporterClass(t *testing.T) {
	f := loadEstimateCalibrationFixtures(t)["synthetic-dense-cpu.json"]
	f.Class = ""
	f.Allocation.SWAFull = nil
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := readEstimateCalibration(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Class != "dense" || decoded.input().SWAFull != nil {
		t.Fatalf("source class or unknown SWA choice lost: %+v", decoded)
	}
}

func TestModelSettingsEstimateCalibrationDiagnosticsNeverValidateBands(t *testing.T) {
	fixtures, _ := calibrationConstructedEvidence(t)
	f := fixtures["check-a"]
	f.Observed[0].Correspondence = "diagnostic"
	f.Observed[0].Comparison.LowerBytes = "9000000"
	f.Observed[0].Comparison.UpperBytes = "9000001"
	if reports, err := reportEstimateCalibration(f); err != nil || len(reports) != 1 || reports[0].Inside {
		t.Fatalf("diagnostic report=%+v err=%v", reports, err)
	}
	fixtures["check-a"] = f
	if err := validateEstimateCalibrationEvidence(fixtures); err == nil || !strings.Contains(err.Error(), "missing held-out") {
		t.Fatalf("diagnostic used for check coverage: %v", err)
	}
}
