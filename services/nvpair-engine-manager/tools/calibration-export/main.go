// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"nvpair-engine-manager/internal/artifactbinding"
	"nvpair-shared/gguffacts"
	"nvpair-shared/modelconfig"
)

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

type storedSample struct {
	ModelIdentity string                        `json:"modelIdentity"`
	Observation   modelconfig.BufferObservation `json:"observation"`
}
type namedArtifact struct{ Role, Path string }
type artifactFlags []namedArtifact

func (a *artifactFlags) String() string { return "ordered role=path selectors" }
func (a *artifactFlags) Set(value string) error {
	role, path, ok := strings.Cut(value, "=")
	if !ok || path == "" || !slices.Contains([]string{"model", "draft", "companion"}, role) {
		return errors.New("artifact requires model|draft|companion=path")
	}
	*a = append(*a, namedArtifact{role, path})
	return nil
}

var digest = regexp.MustCompile(`^[0-9a-f]{64}$`)
var tensorName = regexp.MustCompile(`^(token_embd|output|output_norm|blk\.[0-9]+\.(attn_q|attn_k|attn_v|attn_output|attn_norm|ffn_norm|ffn_gate|ffn_up|ffn_down|ffn_gate_exps|ffn_up_exps|ffn_down_exps))\.(weight|bias)$`)

func main() {
	store := flag.String("store", "", "sample store file")
	out := flag.String("out", "", "new output directory")
	machine := flag.String("machine", "synthetic", "synthetic, pc3 or msi")
	role := flag.String("role", "synthetic", "synthetic, fit or check")
	var paths artifactFlags
	flag.Var(&paths, "artifact", "ordered model|draft|companion=path (repeat)")
	flag.Parse()
	if err := run(context.Background(), *store, *out, *machine, *role, paths); err != nil {
		fmt.Fprintln(os.Stderr, "calibration export refused:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, store, out, machine, role string, paths []namedArtifact) error {
	f, err := os.Open(store)
	if err != nil {
		return errors.New("sample store unavailable")
	}
	fixtures, err := exportFixtures(ctx, f, paths, machine, role, artifactbinding.Access{})
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return errors.New("sample store close failed")
	}
	if out == "" {
		return errors.New("output directory required")
	}
	// Nothing is published until the complete store and artifact set join.
	if err := os.Mkdir(out, 0700); err != nil {
		return errors.New("output directory must be new")
	}
	ok := false
	defer func() {
		if !ok {
			for i := range fixtures {
				_ = os.Remove(filepath.Join(out, fmt.Sprintf("sample-%03d.json", i)))
			}
			_ = os.Remove(out)
		}
	}()
	for i, f := range fixtures {
		data, err := json.MarshalIndent(f, "", "  ")
		if err != nil {
			return errors.New("fixture encoding failed")
		}
		if len(data) > 8<<20 {
			return errors.New("fixture exceeds byte budget")
		}
		if err := os.WriteFile(filepath.Join(out, fmt.Sprintf("sample-%03d.json", i)), append(data, '\n'), 0600); err != nil {
			return errors.New("fixture write failed")
		}
	}
	ok = true
	return nil
}

func exportFixtures(ctx context.Context, store io.Reader, paths []namedArtifact, machine, role string, access artifactbinding.Access) ([]estimateCalibrationFixture, error) {
	if !slices.Contains([]string{"synthetic", "pc3", "msi"}, machine) || !slices.Contains([]string{"synthetic", "fit", "check"}, role) || machine == "synthetic" && role != "synthetic" {
		return nil, errors.New("invalid provenance/role")
	}
	data, err := io.ReadAll(io.LimitReader(store, (4<<20)+1))
	if err != nil || len(data) > 4<<20 {
		return nil, errors.New("sample store exceeds budget or read failed")
	}
	var rawSamples []json.RawMessage
	d := json.NewDecoder(bytes.NewReader(data))
	if d.Decode(&rawSamples) != nil || len(rawSamples) == 0 || len(rawSamples) > 128 {
		return nil, errors.New("invalid sample store")
	}
	if d.Decode(new(json.RawMessage)) != io.EOF {
		return nil, errors.New("sample store trailing data")
	}
	samples := make([]storedSample, len(rawSamples))
	for i, raw := range rawSamples {
		if modelconfig.DecodeStrict(raw, &samples[i]) != nil {
			return nil, errors.New("invalid stored sample")
		}
	}
	if len(paths) == 0 || len(paths) > gguffacts.MaxArtifacts {
		return nil, errors.New("artifact selection required")
	}
	var evidence []artifactbinding.Evidence
	var bindings []modelconfig.ArtifactBinding
	var artifacts []gguffacts.Artifact
	var opened []artifactbinding.File
	defer func() {
		for _, f := range opened {
			_ = f.Close()
		}
	}()
	lastRank := 0
	for _, path := range paths {
		rank := map[string]int{"model": 1, "draft": 2, "companion": 3}[path.Role]
		if rank == 0 || rank < lastRank {
			return nil, errors.New("invalid artifact role order")
		}
		lastRank = rank
		e, err := artifactbinding.Capture(path.Path)
		if err != nil {
			return nil, errors.New("artifact evidence unavailable")
		}
		binding, err := access.Sample(ctx, e, false)
		if err != nil {
			return nil, errors.New("artifact sampling failed")
		}
		f, err := access.CheckedOpen(ctx, e)
		if err != nil {
			return nil, errors.New("artifact checked open failed")
		}
		opened = append(opened, f)
		evidence = append(evidence, e)
		bindings = append(bindings, modelconfig.ArtifactBinding{Role: path.Role, Path: binding.SourcePath, ResolvedTarget: binding.ResolvedTarget, Size: strconv.FormatInt(binding.SizeBytes, 10), MtimeNS: binding.MtimeNS, SampleSHA256: binding.SampleSHA256})
		ggufRole := map[string]gguffacts.Role{"model": gguffacts.RoleMain, "draft": gguffacts.RoleDraft, "companion": gguffacts.RoleCompanion}[path.Role]
		artifacts = append(artifacts, gguffacts.Artifact{Role: ggufRole, Reader: artifactbinding.Reader{Context: ctx, File: f, Expected: e}, Size: e.Size})
	}
	facts, err := gguffacts.Decode(ctx, artifacts)
	if err != nil {
		return nil, errors.New("bounded GGUF decode failed")
	}
	err = validateExportFacts(facts)
	if err != nil {
		return nil, err
	}
	summaries := artifactbinding.Summaries(bindings)
	for i := range summaries {
		summaries[i].Assurance = "sampled-local-binding"
	}
	seen := map[string]bool{}
	var fixtures []estimateCalibrationFixture
	for _, sample := range samples {
		o := sample.Observation
		encoded, err := json.Marshal(sample)
		if err != nil || len(encoded) > 64<<10 || !digest.MatchString(sample.ModelIdentity) || !digest.MatchString(o.SampleID) || seen[o.SampleID] || o.Validate() != nil {
			return nil, errors.New("invalid stored sample")
		}
		seen[o.SampleID] = true
		if !digest.MatchString(o.ConfigurationKey) || !reflect.DeepEqual(o.Artifacts, summaries) {
			return nil, errors.New("unjoinable configuration/artifact set")
		}
		if o.UpstreamPin == "" || o.BuildAssurance != "recognized-self-reported-pin" || o.CapturedOptions == nil {
			return nil, errors.New("sample lacks native provenance/effective allocation")
		}
		opts := o.CapturedOptions
		if o.PerSlotContext <= 0 || o.PerSlotContext%256 != 0 || o.Slots <= 0 || o.Slots > 256 || int64(o.PerSlotContext)*int64(o.Slots) > 1<<20 || opts.Context == nil || opts.Context.Mode != "fixed" || opts.Context.Tokens != o.PerSlotContext || opts.Slots == nil || *opts.Slots != o.Slots || opts.GPU == nil || opts.CacheTypeK == nil || opts.CacheTypeV == nil || opts.FlashAttention == nil || *opts.FlashAttention == "inherit" || opts.BatchSize == nil || opts.UBatchSize == nil || opts.KVOffload == nil || opts.OpOffload == nil || opts.CacheRAM == nil {
			return nil, errors.New("sample lacks finite effective allocation")
		}
		backend := "cpu"
		if opts.GPU.Mode != "cpu" {
			if opts.GPU.Mode != "layers" || opts.GPU.Layers == nil || *opts.GPU.Layers <= 0 {
				return nil, errors.New("sample lacks finite GPU allocation")
			}
			backend = ""
			for _, r := range o.Rows {
				if strings.HasPrefix(r.Label, "CUDA") {
					backend = "cuda"
				}
			}
			if backend == "" {
				return nil, errors.New("sample backend family unavailable")
			}
		}
		at, err := time.Parse(time.RFC3339Nano, o.RecordedAt)
		if err != nil {
			return nil, errors.New("sample date unavailable")
		}
		f := estimateCalibrationFixture{Version: 1, Role: role, Backend: backend, NativePin: o.UpstreamPin, ConfigurationKey: o.ConfigurationKey, Facts: facts, Artifacts: summaries}
		f.Allocation = estimateCalibrationAllocation{PerSlotContext: o.PerSlotContext, TotalContext: o.PerSlotContext * o.Slots, Slots: o.Slots, GPU: *opts.GPU, Options: opts.Clone()}
		f.Sample.ConfigurationKey = o.ConfigurationKey
		f.Sample.Artifacts = summaries
		f.Provenance.Machine = machine
		f.Provenance.Date = at.UTC().Format("2006-01-02")
		f.Provenance.Build = "b12020-" + o.UpstreamPin
		f.Observed, err = exportIntervals(o, backend)
		if err != nil {
			return nil, err
		}
		fixtures = append(fixtures, f)
	}
	// Recheck the ORIGINAL evidence and sampled identity after all facts and
	// projections, never replacing it with a recaptured object.
	for i, e := range evidence {
		if artifactbinding.Validate(e) != nil {
			return nil, errors.New("artifact changed after facts")
		}
		b, err := access.Sample(ctx, e, false)
		if err != nil || b.SampleSHA256 != bindings[i].SampleSHA256 {
			return nil, errors.New("artifact changed after facts")
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return fixtures, nil
}

func validateExportFacts(f gguffacts.Facts) error {
	arch := ""
	tensors := 0
	for _, s := range f.Shards {
		tensors += len(s.Tensors)
		if len(s.Numeric) > 256 || tensors > 16384 {
			return errors.New("facts exceed calibration estimator budget")
		}
		if (s.Role == gguffacts.RoleMain && arch != "" && arch != s.Architecture) || len(s.Templates) > 0 || !slices.Contains([]string{"llama", "qwen2", "qwen3", "qwen2moe", "qwen3moe"}, s.Architecture) {
			return errors.New("facts outside initial calibration privacy/class allowlist")
		}
		if s.Role == gguffacts.RoleMain {
			arch = s.Architecture
		}
		for key, n := range s.Numeric {
			if len(n.Array) > 512 {
				return errors.New("facts exceed calibration numeric budget")
			}
			field := strings.TrimPrefix(key, s.Architecture+".")
			if key == field || !slices.Contains([]string{"block_count", "embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv", "vocab_size", "expert_count", "expert_used_count", "expert_feed_forward_length", "expert_shared_count"}, field) {
				return errors.New("facts numeric privacy allowlist")
			}
		}
		for _, tensor := range s.Tensors {
			if !tensorName.MatchString(tensor.Name) {
				return errors.New("facts tensor privacy allowlist")
			}
		}
	}
	if arch == "" {
		return errors.New("main facts unavailable")
	}
	return nil
}

func exportIntervals(o modelconfig.BufferObservation, backend string) ([]estimateCalibrationInterval, error) {
	var out []estimateCalibrationInterval
	for _, component := range modelconfig.MemoryComponentNames {
		var low, high uint64
		found := false
		cpuWeights := true
		for _, row := range o.Rows {
			name := map[string]string{"model": "weights", "kv": "attention-cache", "compute": "compute-workspace", "recurrent": "recurrent-state"}[row.Component]
			if name != component || row.LowerBytes == "" || row.UpperBytes == "" {
				if name == "weights" && component == "weights" {
					cpuWeights = false
				}
				continue
			}
			if component == "weights" && (row.Label != "CPU" || row.Role != "model" || row.PoolIdentity == "") {
				cpuWeights = false
			}
			l, le := strconv.ParseUint(row.LowerBytes, 10, 64)
			h, he := strconv.ParseUint(row.UpperBytes, 10, 64)
			if le != nil || he != nil || l > h || low > math.MaxUint64-l || high > math.MaxUint64-h {
				return nil, errors.New("invalid diagnostic interval")
			}
			low += l
			high += h
			found = true
		}
		interval := estimateCalibrationInterval{Correspondence: "diagnostic", Comparison: modelconfig.BufferComparison{Component: component, Status: "unknown", LowerBytes: strconv.FormatUint(low, 10), UpperBytes: strconv.FormatUint(high, 10)}}
		for _, c := range o.Comparisons {
			if component == "weights" && found && cpuWeights && c.Component == component && c.Status == "compatible" && c.LowerBytes == strconv.FormatUint(low, 10) && c.UpperBytes == strconv.FormatUint(high, 10) && backend == "cpu" && len(o.Reasons) == 0 && o.TopologyIdentity != "" && !slices.ContainsFunc(o.Artifacts, func(a modelconfig.ArtifactSummary) bool { return a.Role != "model" }) {
				interval.Correspondence = "supported"
				interval.Comparison.Status = "compatible"
				interval.Comparison.LowerBytes = c.LowerBytes
				interval.Comparison.UpperBytes = c.UpperBytes
				found = true
			}
		}
		if found {
			out = append(out, interval)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("sample has no bounded observed intervals")
	}
	return out, nil
}
