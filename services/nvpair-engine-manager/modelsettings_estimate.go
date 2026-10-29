// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-engine-manager/internal/artifactbinding"

	gguf "github.com/gpustack/gguf-parser-go"
	"nvpair-shared/gguffacts"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

const modelEstimatorRevision = "pair-gguf/2;gguf-parser-go/v0.27.0@9257743a3c1775b882ba1b58ec7f5a856c2cc385"
const modelEstimatorSupport = "finite-llama/non-unified/1"
const modelEstimateRecords = 128
const modelEstimateBytes = 1 << 20
const modelEstimateSlotBytes = modelEstimateRecords * 512
const modelEstimateTTL = 30 * time.Second
const modelEstimateTensors = 16384
const modelEstimateLayers = 512

// Borrowed read-only facts, outside compiler results and all authority/digests.
// Ordered bindings have already passed the existing artifact owner's rechecks.
type modelEstimateInput struct {
	Facts            gguffacts.Facts
	Bindings         []modelconfig.ArtifactSummary
	Allocation       *modelconfig.ResolvedLoad
	BuildID, Support string
	Memory           *modelEstimateMemory
	SWAFull          *bool
}

func modelEstimateBindings(bindings []modelconfig.ArtifactBinding) []modelconfig.ArtifactSummary {
	return artifactbinding.Summaries(bindings)
}

// Internal future broker input, never accepted from worker RPC callers. One
// offload device is representable; array positions do not identify devices.
type modelEstimateMemory struct {
	Target           *modelconfig.TargetMemory
	Map              *modelDeviceMap
	Identity, Source string
	At               time.Time
	Domains          []modelconfig.MemoryDomain
	NativeDeviceID   string
	UMA              bool
}

type modelEstimateRecord struct {
	key     string
	created time.Time
	bytes   int
	value   modelconfig.Estimate
}

// Separate from Task 5's facts cache. Zero value ready; no queue or goroutine.
type modelEstimateState struct {
	mu      sync.Mutex
	entries []modelEstimateRecord // least recently used first
	bytes   int
	now     func() time.Time
	run     func(context.Context, modelEstimateInput) modelconfig.Estimate
}

func (s *modelEstimateState) clock() time.Time {
	if s.now != nil {
		return s.now()
	}
	return time.Now()
}

func modelEstimateUnknown(reason string) modelconfig.Estimate {
	return modelconfig.Estimate{Status: "unknown", Source: modelEstimatorRevision, CapacityFit: "unknown", FreeFit: "unknown", Unknowns: []string{reason}}
}

// Preferences alone establish neither artifacts nor native allocation. Auto's
// existing no-facts helper stays honest until its own resolver is implemented.
func estimateModelSettings(ctx context.Context, p modelconfig.InspectRequest, requested modelconfig.RequestedLoad) (modelconfig.Estimate, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.Estimate{}, modelSettingsParamsError(err)
	}
	if err := requested.Validate(); err != nil {
		return modelconfig.Estimate{}, modelSettingsParamsError(err)
	}
	if err := ctx.Err(); err != nil {
		return modelconfig.Estimate{}, err
	}
	return modelEstimateUnknown("bounded artifact facts and effective native allocation unavailable"), nil
}

func (s *modelEstimateState) remove(i int) {
	s.bytes -= s.entries[i].bytes
	copy(s.entries[i:], s.entries[i+1:])
	s.entries[len(s.entries)-1] = modelEstimateRecord{}
	s.entries = s.entries[:len(s.entries)-1]
}

// Failures are advisory data, including panic, overflow and budget exhaustion.
// The caller's operation/context is never cancelled by this boundary.
func (s *modelEstimateState) estimate(ctx context.Context, in modelEstimateInput) (out modelconfig.Estimate) {
	defer func() {
		if r := recover(); r != nil {
			reason := "estimator library panic"
			if err, ok := r.(error); ok && errors.Is(err, errModelEstimateOverflow) {
				reason = "estimator arithmetic overflow"
			}
			out = modelEstimateUnknown(reason)
		}
	}()
	ctx, cancel := context.WithTimeout(ctx, modelInspectionBudget)
	defer cancel()
	if ctx.Err() != nil {
		return modelEstimateUnknown("estimator operation budget exhausted")
	}
	if !s.mu.TryLock() {
		return modelEstimateUnknown("estimator busy")
	}
	defer s.mu.Unlock()
	if reason := modelEstimateBounds(ctx, in); reason != "" {
		return modelEstimateUnknown(reason)
	}
	now := s.clock()
	memoryIdentity := "unavailable: no broker memory input"
	if in.Memory != nil {
		memoryIdentity = in.Memory.Identity
	}
	keyBytes, err := json.Marshal(struct {
		Bindings                                 []modelconfig.ArtifactSummary
		Allocation                               *modelconfig.ResolvedLoad
		Revision, Build, Support, MemoryIdentity string
		Memory                                   *modelEstimateMemory
		SWAFull                                  *bool
	}{in.Bindings, in.Allocation, modelEstimatorRevision, in.BuildID, in.Support, memoryIdentity, in.Memory, in.SWAFull})
	if err != nil || len(keyBytes) > 64<<10 {
		return modelEstimateUnknown("estimator cache identity exceeds budget")
	}
	key := modelLoadDigest(keyBytes)
	for i := 0; i < len(s.entries); {
		entry := s.entries[i]
		if now.Before(entry.created) || now.Sub(entry.created) >= modelEstimateTTL {
			s.remove(i)
			continue
		}
		if entry.key == key {
			if ctx.Err() != nil {
				return modelEstimateUnknown("estimator operation budget exhausted")
			}
			copy(s.entries[i:], s.entries[i+1:])
			s.entries[len(s.entries)-1] = entry
			cached := entry.value.Clone()
			modelEstimateDomains(&cached, in, s.clock())
			return cached
		}
		i++
	}
	if s.run != nil {
		out = s.run(ctx, in)
	} else {
		out = calculateModelEstimate(ctx, in)
	}
	if ctx.Err() != nil {
		return modelEstimateUnknown("estimator operation budget exhausted")
	}
	out.Source, out.BuildID, out.Timestamp = modelEstimatorRevision, in.BuildID, now.UTC().Format(time.RFC3339Nano)
	out.EstimateSource = &modelconfig.EstimateSource{Kind: "pair-estimate", Provider: "PAIR", Revision: modelEstimatorRevision, Basis: "bounded GGUF facts and pinned graph estimates; see component range bases"}
	raw := out.Clone()
	modelEstimateDomains(&out, in, s.clock())
	if err := out.Validate(); err != nil {
		return modelEstimateUnknown("estimator result exceeds contract or size budget")
	}
	data, err := json.Marshal(out)
	if err != nil {
		return modelEstimateUnknown("estimator result encoding unavailable")
	}
	if ctx.Err() != nil {
		return modelEstimateUnknown("estimator operation budget exhausted")
	}
	// Charge retained strings, record/slice overhead and reserve the backing array.
	size := 4*len(data) + len(key) + 512
	if size <= modelEstimateBytes-modelEstimateSlotBytes {
		for i := range s.entries {
			if s.entries[i].key == key {
				s.remove(i)
				break
			}
		}
		for len(s.entries) >= modelEstimateRecords || s.bytes > modelEstimateBytes-modelEstimateSlotBytes-size {
			s.remove(0)
		}
		if s.entries == nil {
			s.entries = make([]modelEstimateRecord, 0, modelEstimateRecords)
		}
		s.entries = append(s.entries, modelEstimateRecord{key, now, size, modelEstimateDetach(raw)})
		s.bytes += size
	}
	out = raw
	modelEstimateDomains(&out, in, s.clock())
	return out
}

// Detach string backing storage too; a bounded substring must not retain a
// larger input allocation. The wire clone already detaches every slice.
func modelEstimateDetach(out modelconfig.Estimate) modelconfig.Estimate {
	out = out.Clone()
	for _, value := range []*string{&out.Status, &out.Source, &out.BuildID, &out.Timestamp, &out.MemorySource, &out.MemoryTimestamp, &out.HostBytes, &out.DeviceBytes} {
		*value = strings.Clone(*value)
	}
	for i := range out.Components {
		c := &out.Components[i]
		for _, value := range []*string{&c.Name, &c.DomainID, &c.Status, &c.Bytes, &c.Reason} {
			*value = strings.Clone(*value)
		}
	}
	for i := range out.Domains {
		d := &out.Domains[i]
		for _, value := range []*string{&d.ID, &d.Kind, &d.DeviceID, &d.CapacityBytes, &d.FreeBytes, &d.RequiredBytes} {
			*value = strings.Clone(*value)
		}
	}
	for i := range out.Unknowns {
		out.Unknowns[i] = strings.Clone(out.Unknowns[i])
	}
	for i := range out.Assumptions {
		out.Assumptions[i] = strings.Clone(out.Assumptions[i])
	}
	return out
}

func modelEstimateBounds(ctx context.Context, in modelEstimateInput) string {
	if len(in.Facts.Shards) == 0 || len(in.Bindings) == 0 {
		return "bounded artifact facts or ordered bindings unavailable"
	}
	if len(in.Facts.Shards) > gguffacts.MaxArtifacts || len(in.Bindings) > gguffacts.MaxArtifacts || len(in.BuildID) > 256 || len(in.Support) > 256 {
		return "estimator artifact/build limit"
	}
	var tensors, retained int
	for _, b := range in.Bindings {
		retained += len(b.Role) + len(b.Identity) + len(b.Size) + len(b.Assurance)
		if retained > 64<<10 || b.Identity == "" {
			return "estimator ordered binding limit or missing identity"
		}
	}
	for _, shard := range in.Facts.Shards {
		if ctx.Err() != nil {
			return "estimator operation budget exhausted"
		}
		tensors += len(shard.Tensors)
		if tensors > modelEstimateTensors || len(shard.Numeric) > 256 || len(shard.Architecture) > 256 {
			return "estimator tensor/metadata limit"
		}
		for key, n := range shard.Numeric {
			if len(key) > 256 || len(n.Array) > modelEstimateLayers {
				return "estimator numeric limit"
			}
			retained += len(key) + len(n.Array)*8 + 64
		}
		for _, tensor := range shard.Tensors {
			if len(tensor.Name) > gguffacts.MaxNameBytes || len(tensor.Dimensions) > 4 {
				return "estimator tensor dimension/name limit"
			}
			retained += len(tensor.Name) + len(tensor.Dimensions)*8 + 128
		}
		if retained > 8<<20 {
			return "estimator retained limit"
		}
	}
	if a := in.Allocation; a != nil {
		if a.TotalContext <= 0 || a.TotalContext > 1<<20 || a.Slots <= 0 || a.Slots > 256 || a.PerSlotContext <= 0 || a.PerSlotContext%256 != 0 || int64(a.PerSlotContext)*int64(a.Slots) != int64(a.TotalContext) || a.KVUnified == nil || *a.KVUnified {
			return "estimator context/slots limit or unsupported unified cache"
		}
		if len(a.Artifacts) > gguffacts.MaxArtifacts {
			return "estimator allocation artifact limit"
		}
		allocationText := len(a.NativeBuildID) + len(a.SourceInputDigest) + len(a.PresetDigest) + len(a.ResolverRevision) + len(a.ReviewedInputID) + len(a.GPU.Mode)
		for _, b := range a.Artifacts {
			allocationText += len(b.Role) + len(b.Path) + len(b.ResolvedTarget) + len(b.Size) + len(b.MtimeNS) + len(b.SampleSHA256)
		}
		if allocationText > 64<<10 {
			return "estimator allocation identity limit"
		}
		if err := a.Options.Validate(); err != nil {
			return "estimator malformed effective options"
		}
	}
	if in.Memory != nil && (len(in.Memory.Domains) > 64 || len(in.Memory.Identity) > 256 || len(in.Memory.Source) > 256 || len(in.Memory.NativeDeviceID) > 256) {
		return "estimator memory input limit"
	}
	if in.Memory != nil {
		for _, d := range in.Memory.Domains {
			if len(d.ID) > 256 || len(d.Kind) > 16 || len(d.DeviceID) > 256 || len(d.CapacityBytes) > 20 || len(d.FreeBytes) > 20 || len(d.RequiredBytes) > 20 {
				return "estimator memory domain field limit"
			}
		}
	}
	return ""
}

var errModelEstimateOverflow = errors.New("estimate arithmetic overflow")

func modelEstimateAdd(a, b uint64) uint64 {
	if b > math.MaxUint64-a {
		panic(errModelEstimateOverflow)
	}
	return a + b
}

func modelEstimateMul(a, b uint64) uint64 {
	if a != 0 && b > math.MaxUint64/a {
		panic(errModelEstimateOverflow)
	}
	return a * b
}

// Facts preserve raw integer representations. Check their declared width
// before conversion, including synthetic/internal callers outside the decoder.
func modelEstimateUint(n gguffacts.Numeric) (uint64, bool) {
	if n.Array != nil {
		return 0, false
	}
	switch n.Type {
	case 0, 1:
		if n.Bits > math.MaxUint8 {
			return 0, false
		}
	case 2, 3:
		if n.Bits > math.MaxUint16 {
			return 0, false
		}
	case 4, 5:
		if n.Bits > math.MaxUint32 {
			return 0, false
		}
	}
	return n.Uint64()
}

// Validate the library's intermediate multiply, block alignment and products.
// Unknown/deprecated types never become f16 by fallback.
func modelEstimateRow(typ gguf.GGMLType, dims ...uint64) (uint64, bool) {
	trait, ok := typ.Trait()
	if !ok || trait.BlockSize == 0 || trait.TypeSize == 0 || len(dims) == 0 || dims[0] == 0 || dims[0]%trait.BlockSize != 0 {
		return 0, false
	}
	n := modelEstimateMul(trait.TypeSize, dims[0]) / trait.BlockSize
	for _, dim := range dims[1:] {
		if dim == 0 {
			return 0, false
		}
		n = modelEstimateMul(n, dim)
	}
	return typ.RowSizeOf(dims), true
}

type modelEstimateModel struct {
	file                       gguf.GGUFFile
	numeric                    map[string]gguffacts.Numeric
	arch                       string
	weights, companionWeights  uint64
	companions, weightsUnknown bool
}

// In-memory adapter only: no file/parser/native APIs. Sampling recommendations
// and template information never enter this projection or change saved policy.
func modelEstimateAdapt(ctx context.Context, facts gguffacts.Facts) (modelEstimateModel, string) {
	m := modelEstimateModel{numeric: map[string]gguffacts.Numeric{}}
	seen := map[string]bool{}
	for _, shard := range facts.Shards {
		if shard.Role == gguffacts.RoleMain {
			if !inspectionArchitectureName(shard.Architecture) || m.arch != "" && m.arch != shard.Architecture {
				return m, "conflicting or missing main architecture"
			}
			m.arch = shard.Architecture
			for key, value := range shard.Numeric {
				if !strings.HasPrefix(key, m.arch+".") {
					continue
				}
				if old, ok := m.numeric[key]; ok && !reflect.DeepEqual(old, value) {
					return m, "conflicting main numeric dimensions"
				}
				m.numeric[key] = value
			}
		} else {
			m.companions = true
		}
		for _, tensor := range shard.Tensors {
			if ctx.Err() != nil {
				return m, "estimator operation budget exhausted"
			}
			bytes, ok := modelEstimateRow(gguf.GGMLType(tensor.Type), tensor.Dimensions...)
			if !ok || bytes != tensor.Bytes {
				m.weightsUnknown = true
				continue
			}
			if shard.Role != gguffacts.RoleMain {
				m.companionWeights = modelEstimateAdd(m.companionWeights, bytes)
				continue
			}
			if seen[tensor.Name] || tensor.Name == "" {
				return m, "duplicate or missing main tensor name"
			}
			seen[tensor.Name] = true
			m.weights = modelEstimateAdd(m.weights, bytes)
			m.file.TensorInfos = append(m.file.TensorInfos, gguf.GGUFTensorInfo{Name: tensor.Name, Type: gguf.GGMLType(tensor.Type), NDimensions: uint32(len(tensor.Dimensions)), Dimensions: slices.Clone(tensor.Dimensions)})
		}
	}
	if m.arch == "" {
		return m, "main architecture unavailable"
	}
	m.file.Header.MetadataKV = append(m.file.Header.MetadataKV, gguf.GGUFMetadataKV{Key: "general.architecture", ValueType: gguf.GGUFMetadataValueTypeString, Value: m.arch})
	for _, key := range sortedKeys(m.numeric) {
		n := m.numeric[key]
		if n.Array != nil {
			values := make([]any, len(n.Array))
			valid := true
			for i, bits := range n.Array {
				v, ok := modelEstimateUint(gguffacts.Numeric{Type: n.Type, Bits: bits})
				if !ok || v > 1<<20 {
					valid = false
					break
				}
				values[i] = v
			}
			if !valid {
				continue
			} // dependent components retain their own Unknowns
			m.file.Header.MetadataKV = append(m.file.Header.MetadataKV, gguf.GGUFMetadataKV{Key: key, ValueType: gguf.GGUFMetadataValueTypeArray, Value: gguf.GGUFMetadataKVArrayValue{Type: gguf.GGUFMetadataValueTypeUint64, Len: uint64(len(values)), Array: values}})
		} else if v, ok := modelEstimateUint(n); ok && v <= 1<<20 {
			m.file.Header.MetadataKV = append(m.file.Header.MetadataKV, gguf.GGUFMetadataKV{Key: key, ValueType: gguf.GGUFMetadataValueTypeUint64, Value: v})
		}
	}
	m.file.Header.TensorCount = uint64(len(m.file.TensorInfos))
	m.file.Header.MetadataKVCount = uint64(len(m.file.Header.MetadataKV))
	m.file.ModelSize = gguf.GGUFBytesScalar(m.weights)
	// No file metadata size is invented. Equal values avoid the library's
	// unsigned (Size-ModelSize) subtraction underflow in unused bootstrap costs.
	m.file.Size = m.file.ModelSize
	return m, ""
}

func (m modelEstimateModel) number(name string) (uint64, bool) {
	n, exists := m.numeric[m.arch+"."+name]
	v, ok := modelEstimateUint(n)
	return v, exists && ok && v <= 1<<20
}

func estimateComponent(name string, bytes uint64, reason string) modelconfig.EstimateComponent {
	amount := strconv.FormatUint(bytes, 10)
	return modelconfig.EstimateComponent{Name: name, Status: "estimated", Bytes: amount, LowBytes: amount, HighBytes: amount, Basis: "derived", Reason: reason}
}

func unknownEstimateComponent(name, reason string) modelconfig.EstimateComponent {
	return modelconfig.EstimateComponent{Name: name, Status: "unknown", Reason: reason}
}

func absentEstimateComponent(name string) modelconfig.EstimateComponent {
	return modelconfig.EstimateComponent{Name: name, Status: "excluded", Reason: "absent from validated selected artifact set"}
}

func calculateModelEstimate(ctx context.Context, in modelEstimateInput) modelconfig.Estimate {
	m, reason := modelEstimateAdapt(ctx, in.Facts)
	if reason != "" {
		return modelEstimateUnknown(reason)
	}
	out := modelconfig.Estimate{Status: "partial", Assumptions: []string{
		"Advisory; tensor storage and pinned library graph costs are not measured native allocations.",
		"Simultaneous resident models are not included.",
		"Known totals are partial component sums; advisory fit verdicts retain warnings for unpriced costs and are not complete peak bounds.",
	}}
	weights := estimateComponent("weights", m.weights, "bounded tensor directory; all resident experts retained, never active-expert fraction")
	if m.weightsUnknown || m.weights == 0 {
		weights.Status, weights.Reason = "partial", "tensor types/dimensions incomplete; known tensor bytes only"
	}
	out.Components = append(out.Components, weights)
	if m.companions {
		c := estimateComponent("companions", m.companionWeights, "draft/projector tensor storage only; independent cache, workspace and placement unsized")
		c.Status, c.Basis = "partial", "heuristic"
		out.Components = append(out.Components, c)
	} else {
		out.Components = append(out.Components, absentEstimateComponent("companions"))
	}
	kv, recurrent := modelEstimateCaches(ctx, m, in)
	out.Components = append(out.Components, kv, recurrent, modelEstimateCompute(ctx, m, in))
	if in.Allocation != nil && in.Allocation.GPU.Mode == "layers" && in.Allocation.GPU.Layers != nil && *in.Allocation.GPU.Layers > 0 {
		out.Components = append(out.Components, modelEstimateComputePortion(ctx, m, in, true))
	}
	if in.Allocation != nil && in.Allocation.Options.CacheRAM != nil && *in.Allocation.Options.CacheRAM > 0 {
		c := estimateComponent("prompt-cache", modelEstimateMul(uint64(*in.Allocation.Options.CacheRAM), 1<<20), "configured host prompt-cache ceiling; usage is demand-dependent")
		c.LowBytes = "0"
		out.Components = append(out.Components, c)
	} else {
		out.Components = append(out.Components, unknownEstimateComponent("prompt-cache", "effective native cache-ram unavailable"))
	}
	out.Components = append(out.Components, unknownEstimateComponent("runtime-overhead", "backend context, driver and allocator costs not yet calibrated or measured"))
	out.Unknowns = append(out.Unknowns, "mmap residency unsized", "transient peaks unsized")
	modelEstimatePlacement(&out, m, in)
	// Band each placed portion, so split weights retain their calibrated ranges.
	class := modelEstimateClass(m)
	backend := modelEstimateBackend(in)
	for i := range out.Components {
		c := &out.Components[i]
		if c.Bytes == "" {
			continue
		}
		_, calibrated := modelEstimateCalibrations[modelEstimateCalibrationKey{c.Name, class, backend}]
		if c.Basis != "heuristic" && !calibrated {
			continue
		}
		n, err := strconv.ParseUint(c.Bytes, 10, 64)
		if err != nil {
			panic(errModelEstimateOverflow)
		}
		low, high, basis := modelEstimateBand(c.Name, class, backend, n)
		c.LowBytes, c.HighBytes, c.Basis = strconv.FormatUint(low, 10), strconv.FormatUint(high, 10), basis
	}
	modelEstimateDomainAmounts(&out)
	for _, c := range out.Components {
		if c.Status == "unknown" || c.Status == "partial" {
			out.Unknowns = append(out.Unknowns, c.Name+": "+c.Reason)
		}
	}
	return out
}

// Family evidence is advisory only. Missing or mixed native inventory is not a
// reason to infer a family from the operating system or a physical GPU brand.
func modelEstimateBackend(in modelEstimateInput) string {
	if in.Allocation == nil {
		return ""
	}
	a := in.Allocation
	if a.GPU.Mode == "cpu" || a.GPU.Mode == "layers" && a.GPU.Layers != nil && *a.GPU.Layers == 0 {
		return "cpu"
	}
	if in.Memory == nil || in.Memory.Map == nil || in.Memory.Map.Unknown != "" {
		return ""
	}
	family := ""
	for _, d := range in.Memory.Map.Devices {
		if d.Kind == 0 {
			continue
		}
		next := ""
		if d.Unknown == "" && strings.HasPrefix(d.Selector, "CUDA") {
			next = "cuda"
		}
		if d.Unknown == "" && strings.HasPrefix(d.Selector, "Metal") {
			next = "metal"
		}
		if next == "" || family != "" && family != next {
			return ""
		}
		family = next
	}
	return family
}

func modelEstimateDomainAmounts(out *modelconfig.Estimate) {
	for i := range out.Domains {
		d := &out.Domains[i]
		var expected, low, high uint64
		for _, c := range out.Components {
			if c.DomainID != d.ID || c.Bytes == "" {
				continue
			}
			for _, pair := range []struct {
				text string
				sum  *uint64
			}{{c.Bytes, &expected}, {c.LowBytes, &low}, {c.HighBytes, &high}} {
				n, err := strconv.ParseUint(pair.text, 10, 64)
				if err != nil {
					panic(errModelEstimateOverflow)
				}
				*pair.sum = modelEstimateAdd(*pair.sum, n)
			}
		}
		d.RequiredBytes, d.RequiredLowBytes, d.RequiredHighBytes = strconv.FormatUint(expected, 10), strconv.FormatUint(low, 10), strconv.FormatUint(high, 10)
	}
}

// Classification groups calibration evidence, never grants formula support.
// The more specific native memory layout takes precedence over expert count.
func modelEstimateClass(m modelEstimateModel) string {
	if slices.Contains([]string{"lfm2", "lfm2moe", "qwen3next", "qwen35", "qwen35moe", "mamba", "mamba2", "jamba", "rwkv6", "rwkv7"}, m.arch) {
		return "hybrid"
	}
	if slices.Contains([]string{"deepseek2", "deepseek32", "glm-dsa", "kimi-k3"}, m.arch) {
		return "mla"
	}
	if window, ok := m.number("attention.sliding_window"); ok && window > 0 || slices.Contains([]string{"gemma2", "gemma3", "gemma3n", "gemma4"}, m.arch) {
		return "swa"
	}
	if experts, ok := m.number("expert_count"); ok && experts > 0 || slices.Contains([]string{"qwen2moe", "qwen3moe"}, m.arch) {
		return "moe"
	}
	if slices.Contains([]string{"llama", "qwen2", "qwen3", "mistral"}, m.arch) {
		return "dense"
	}
	return m.arch
}

func modelEstimateCacheType(value *string) (gguf.GGMLType, bool) {
	if value != nil {
		switch *value {
		case "f16":
			return gguf.GGMLTypeF16, true
		case "bf16":
			return gguf.GGMLTypeBF16, true
		case "q8_0":
			return gguf.GGMLTypeQ8_0, true
		case "q4_0":
			return gguf.GGMLTypeQ4_0, true
		}
	}
	return 0, false
}

// Recurrent state does not depend on attention precision or context length.
// Size it independently so an unresolved KV option cannot erase known state.
func modelEstimateRecurrent(m modelEstimateModel, in modelEstimateInput) modelconfig.EstimateComponent {
	return modelEstimateRecurrentForLayers(m, in, false)
}

func modelEstimateRecurrentForLayers(m modelEstimateModel, in modelEstimateInput, deviceOnly bool) modelconfig.EstimateComponent {
	u := unknownEstimateComponent("recurrent-state", "recurrent dimensions or per-layer native mapping unavailable")
	if slices.Contains([]string{"mamba", "mamba2", "rwkv6", "rwkv7", "qwen3next", "qwen35", "qwen35moe"}, m.arch) {
		return modelEstimateNativeRecurrent(m, in, deviceOnly)
	}
	if m.arch != "lfm2" && m.arch != "lfm2moe" {
		if !slices.Contains([]string{"llama", "llama4", "gpt-oss", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "mistral", "gemma2", "gemma3", "gemma4", "deepseek2", "deepseek3"}, m.arch) {
			u.Reason = "native recurrent layer mapping unconfirmed for architecture " + m.arch
			return u
		}
		for _, field := range []string{"attention.recurrent_layers", "full_attention_interval", "shortconv.l_cache", "ssm.state_size", "ssm.conv_kernel", "wkv.head_size"} {
			if n, exists := m.numeric[m.arch+"."+field]; exists {
				v, ok := modelEstimateUint(n)
				if !ok || v > 0 {
					return u
				}
			}
		}
		if n, exists := m.numeric[m.arch+".attention.head_count_kv"]; exists && n.Array != nil {
			for _, bits := range n.Array {
				v, ok := modelEstimateUint(gguffacts.Numeric{Type: n.Type, Bits: bits})
				if !ok || v == 0 {
					return u
				}
			}
		}
		return absentEstimateComponent("recurrent-state")
	}
	if in.Allocation == nil || in.Support != modelEstimatorSupport || in.BuildID == "" {
		return u
	}
	layers, lok := m.number("block_count")
	emb, eok := m.number("embedding_length")
	mask, exists := m.numeric[m.arch+".attention.head_count_kv"]
	if !lok || layers == 0 || layers > modelEstimateLayers || !eok || emb == 0 || !exists || mask.Array == nil || len(mask.Array) != int(layers) {
		return u
	}
	var count uint64
	for i, bits := range mask.Array {
		v, ok := modelEstimateUint(gguffacts.Numeric{Type: mask.Type, Bits: bits})
		if !ok {
			return u
		}
		if v == 0 && (!deviceOnly || modelEstimateLayerOnDevice(in, layers, uint64(i))) {
			count++
		}
	}
	if count == 0 {
		if deviceOnly {
			return estimateComponent("recurrent-state", 0, "no recurrent layers on device")
		}
		return absentEstimateComponent("recurrent-state")
	}
	length, ok := m.number("shortconv.l_cache")
	if !ok || length == 0 {
		return u
	}
	// Native llama-hparams shortconv contract referenced by the pinned library:
	// F32[(l_cache-1)*embedding, sequences] per recurrent layer.
	elements := modelEstimateMul(modelEstimateMul(length-1, emb), uint64(in.Allocation.Slots))
	var bytes uint64
	if elements > 0 {
		bytes, _ = modelEstimateRow(gguf.GGMLTypeF32, elements)
	}
	return estimateComponent("recurrent-state", modelEstimateMul(bytes, count), "LFM2 short-convolution state per recurrent layer and slot; no dense KV for those layers")
}

// Reuse Task 4's output-first layer owner for both cache kinds.
func modelEstimateLayerOnDevice(in modelEstimateInput, layers, layer uint64) bool {
	a := in.Allocation
	return a != nil && a.GPU.Mode == "layers" && a.GPU.Layers != nil && *a.GPU.Layers > 0 && uint64(*a.GPU.Layers) <= layers+1 && layer >= layers+1-uint64(*a.GPU.Layers)
}

func modelEstimateNativeRecurrent(m modelEstimateModel, in modelEstimateInput, deviceOnly bool) modelconfig.EstimateComponent {
	u := unknownEstimateComponent("recurrent-state", "native recurrent dimensions, mask or main-context mapping unavailable")
	if in.Allocation == nil || in.Support != modelEstimatorSupport || in.BuildID == "" || in.Allocation.Slots < 1 {
		return u
	}
	// Native hparams widths are uint32_t, not arbitrary precision dimensions.
	number := func(key string) (uint64, bool) {
		// llama-model-loader.cpp:122,167 requires GGUF uint32 for these
		// scalar hparams; a numerically equal signed/wider type is not valid.
		n := m.numeric[m.arch+"."+key]
		v, ok := m.number(key)
		return v, ok && n.Type == 4 && n.Array == nil && v > 0 && v <= math.MaxUint32
	}
	layers, lok := number("block_count")
	emb, eok := number("embedding_length")
	if !lok || layers > modelEstimateLayers || !eok {
		return u
	}
	hybrid := slices.Contains([]string{"qwen3next", "qwen35", "qwen35moe"}, m.arch)
	trunk := layers
	if n, exists := m.numeric[m.arch+".nextn_predict_layers"]; exists {
		next, ok := modelEstimateUint(n)
		if n.Type != 4 || n.Array != nil || !ok || next >= layers || next > 0 && !hybrid {
			return u
		}
		trunk -= next
	}
	var r, s uint64
	if m.arch == "rwkv6" || m.arch == "rwkv7" {
		head, ok := number("wkv.head_size")
		shift := uint64(2) // llama-hparams.h:143; only absent keys use this default.
		if _, exists := m.numeric[m.arch+".token_shift_count"]; exists {
			var valid bool
			shift, valid = number("token_shift_count")
			if !valid {
				return u
			}
		}
		if !ok || emb%head != 0 {
			return u
		}
		r, s = modelEstimateMul(shift, emb), modelEstimateMul(emb, head)
	} else {
		conv, cok := number("ssm.conv_kernel")
		inner, iok := number("ssm.inner_size")
		state, sok := number("ssm.state_size")
		rank, rok := number("ssm.time_step_rank")
		group := uint64(0) // Mamba loader leaves ssm_n_group at its native zero.
		if m.arch != "mamba" {
			var gok bool
			group, gok = number("ssm.group_count")
			if !gok {
				return u
			}
		}
		if !cok || !iok || !sok || !rok || m.arch == "mamba" && inner != 2*emb || m.arch != "mamba" && inner%rank != 0 {
			return u
		}
		if m.arch != "mamba" && (inner%group != 0 || rank%group != 0) {
			return u
		}
		// Qwen tensor loaders use value_dim=state*rank, and grouped value
		// heads must divide by key heads (models/qwen35.cpp:55-60,439).
		if hybrid && inner != modelEstimateMul(state, rank) {
			return u
		}
		// llama-hparams.cpp:208-261: R=(conv-1)*(inner+2*group*state),
		// S=state*inner, also for Qwen gated delta-net at this pin.
		channels := modelEstimateAdd(inner, modelEstimateMul(2, modelEstimateMul(group, state)))
		if channels > math.MaxUint32 {
			return u
		}
		r = modelEstimateMul(conv-1, channels)
		s = modelEstimateMul(state, inner)
	}
	if r > math.MaxUint32 || s > math.MaxUint32 {
		return u
	}
	mask, explicit := m.numeric[m.arch+".attention.recurrent_layers"]
	interval := uint64(4)
	if hybrid && !explicit {
		if _, exists := m.numeric[m.arch+".full_attention_interval"]; exists {
			var ok bool
			interval, ok = number("full_attention_interval")
			if !ok {
				return u
			}
		}
	}
	if explicit && (!hybrid || mask.Type != 4 && !(mask.Type == 7 && mask.Array != nil) || mask.Array != nil && len(mask.Array) != int(layers)) {
		return u
	}
	var count uint64
	for i := uint64(0); i < layers; i++ {
		recurrent := i < trunk
		if hybrid {
			if explicit {
				bit := mask.Bits
				if mask.Array != nil {
					bit = mask.Array[i]
				}
				if bit > 1 || i >= trunk && bit != 0 {
					return u
				}
				recurrent = recurrent && bit == 1
			} else {
				recurrent = recurrent && (i+1)%interval != 0
			}
		}
		if recurrent && (!deviceOnly || modelEstimateLayerOnDevice(in, layers, i)) {
			count++
		}
	}
	// llama-model.cpp:2547-2648 passes F32 R/S, max(1,n_seq_max),
	// n_rs_seq (default zero, llama-context.cpp:3701). No context padding:
	// recurrent.cpp:101-103 uses rows=mem_size*(1+n_rs_seq). Backend buffer
	// alignment/context overhead remains in the unpriced runtime-overhead row.
	bytes := modelEstimateMul(modelEstimateMul(modelEstimateAdd(r, s), 4), uint64(in.Allocation.Slots))
	return estimateComponent("recurrent-state", modelEstimateMul(bytes, count), "pinned native F32 R/S recurrent tensors per main-context layer and sequence slot; no rollback planes in supported allocation")
}

// Explicit support matrix, using native hparams contracts referenced by the
// pinned MIT library. Family names never substitute for recurrent/MLA mappings.
func modelEstimateCaches(ctx context.Context, m modelEstimateModel, in modelEstimateInput) (attention, state modelconfig.EstimateComponent) {
	return modelEstimateCachesForLayers(ctx, m, in, false)
}

// deviceOnly selects native layers, never a fraction of aggregate cache bytes.
// llama-model.cpp:1521-1546 and llama-kv-cache.cpp:214-220 at 7fe450e1.
func modelEstimateCachesForLayers(ctx context.Context, m modelEstimateModel, in modelEstimateInput, deviceOnly bool) (attention, state modelconfig.EstimateComponent) {
	defer func() {
		if recovered := recover(); recovered != nil {
			reason := "pinned library cache panic"
			if err, ok := recovered.(error); ok && errors.Is(err, errModelEstimateOverflow) {
				reason = "cache arithmetic overflow"
			}
			attention = unknownEstimateComponent("attention-cache", reason)
			state = unknownEstimateComponent("recurrent-state", reason)
		}
	}()
	kv := unknownEstimateComponent("attention-cache", "architecture or effective allocation unmapped")
	recurrent := modelEstimateRecurrent(m, in)
	if slices.Contains([]string{"mamba", "mamba2", "rwkv6", "rwkv7"}, m.arch) && recurrent.Status == "estimated" {
		return absentEstimateComponent("attention-cache"), recurrent
	}
	hybrid := slices.Contains([]string{"lfm2", "lfm2moe", "qwen3next", "qwen35", "qwen35moe"}, m.arch)
	if slices.Contains([]string{"qwen3next", "qwen35", "qwen35moe"}, m.arch) {
		if _, explicit := m.numeric[m.arch+".attention.recurrent_layers"]; explicit {
			// models/qwen35.cpp:17-21 gives this mask precedence over
			// the interval. Keep attention unpriced until its mapping joins.
			kv.Reason = "attention cache explicit recurrent layer mapping unconfirmed"
			return kv, recurrent
		}
	}
	mla := m.arch == "deepseek2"
	shared := m.arch == "gemma4"
	// llama-arch.cpp:127 names LLM_ARCH_OPENAI_MOE as canonical "gpt-oss".
	dense := slices.Contains([]string{"llama", "llama4", "gpt-oss", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "mistral", "gemma2", "gemma3", "deepseek2", "gemma4"}, m.arch)
	if !dense && !hybrid {
		kv.Reason = "architecture unmapped; MLA/compressed caches are not dense KV"
		return kv, recurrent
	}
	if dense && recurrent.Status == "unknown" {
		kv.Reason = "nominally dense architecture contains unmapped recurrent metadata"
		return kv, recurrent
	}
	if in.Allocation == nil || in.Support != modelEstimatorSupport || in.BuildID == "" {
		kv.Reason = "effective native allocation/build support unavailable"
		return kv, recurrent
	}
	a := in.Allocation
	layers, ok := m.number("block_count")
	if !ok || layers == 0 || layers > modelEstimateLayers {
		kv.Reason = "block_count missing, malformed or exceeds estimator limit"
		return kv, recurrent
	}
	heads, headsOK := m.number("attention.head_count")
	emb, embOK := m.number("embedding_length")
	if !headsOK || heads == 0 || !embOK || emb == 0 || emb%heads != 0 {
		kv.Reason = "attention heads/embedding missing or indivisible"
		return kv, recurrent
	}
	key, keyOK := m.number("attention.key_length")
	value, valueOK := m.number("attention.value_length")
	if _, exists := m.numeric[m.arch+".attention.key_length"]; !exists {
		key, keyOK = emb/heads, true
	}
	if _, exists := m.numeric[m.arch+".attention.value_length"]; !exists {
		value, valueOK = emb/heads, true
	}
	if !keyOK || !valueOK || key == 0 || value == 0 {
		kv.Reason = "key/value widths malformed"
		return kv, recurrent
	}
	kt, kOK := modelEstimateCacheType(a.Options.CacheTypeK)
	vt, vOK := modelEstimateCacheType(a.Options.CacheTypeV)
	if !kOK || !vOK || a.Options.FlashAttention == nil || *a.Options.FlashAttention == "inherit" {
		kv.Reason = "effective K/V precision or Flash Attention unresolved"
		return kv, recurrent
	}
	if vt.IsQuantized() && *a.Options.FlashAttention != "on" {
		kv.Reason = "quantized V requires mapped Flash Attention; no precision fallback"
		return kv, recurrent
	}
	hkv, scalarOK := m.number("attention.head_count_kv")
	perHeads, present := m.numeric[m.arch+".attention.head_count_kv"]
	isArray := present && perHeads.Array != nil
	if isArray && len(perHeads.Array) != int(layers) || !isArray && (!scalarOK || hkv == 0) {
		kv.Reason = "KV head count or complete per-layer recurrent mask unavailable"
		return kv, recurrent
	}
	interval, intervalOK := m.number("full_attention_interval")
	if hybrid && !isArray && (!intervalOK || interval < 2 || interval > layers || !slices.Contains([]string{"qwen3next", "qwen35", "qwen35moe"}, m.arch)) {
		kv.Reason = "hybrid attention/recurrent layer mask unavailable"
		return kv, recurrent
	}
	for _, field := range []string{"attention.kv_lora_rank", "attention.key_length_mla", "attention.value_length_mla"} {
		if n, exists := m.numeric[m.arch+"."+field]; exists {
			v, valid := modelEstimateUint(n)
			if !valid || v != 0 && !mla {
				kv.Reason = "MLA metadata on unmapped architecture"
				return kv, recurrent
			}
		}
	}
	if mla {
		// llama.cpp 7fe450e1: models/deepseek2.cpp:15-17 loads MLA query
		// widths separately. llama-model.cpp:1371-1380 loads stored key width
		// and rotary width; llama-hparams.cpp:156-159 multiplies by KV heads.
		// There is no FA-dependent width rewrite at this pin. Only accept the
		// confirmed latent+rotary layout, not a guessed conversion of query heads.
		// gguf-parser-go v0.27.0 file_architecture.go:1321-1330 retains both
		// widths, but file_estimate__llamacpp.go:1121-1123 substitutes query
		// widths. Do not use that approximation for the native cache allocation.
		latent, lok := m.number("attention.kv_lora_rank")
		rope, rok := m.number("rope.dimension_count")
		mk, mkok := m.number("attention.key_length_mla")
		mv, mvok := m.number("attention.value_length_mla")
		if !lok || latent == 0 || !rok || rope == 0 || !mkok || mk <= rope || !mvok || mv == 0 || isArray || hkv != 1 || key != modelEstimateAdd(latent, rope) || value != latent || kt != vt {
			kv.Reason = "MLA latent/rotary stored widths, scalar KV head or matching K/V types unconfirmed"
			return kv, recurrent
		}
	}
	cacheLayers := layers
	if n, exists := m.numeric[m.arch+".nextn_predict_layers"]; exists {
		next, valid := modelEstimateUint(n)
		// Main serving context only, not an MTP context: llama-model.cpp:
		// 2676-2681 and 2577-2584. Router and all-NextN contexts stay unmapped.
		if !valid || next >= layers || next > 0 && !mla && !slices.Contains([]string{"qwen3next", "qwen35", "qwen35moe"}, m.arch) {
			kv.Reason = "NextN architecture or main-context layer mapping unconfirmed"
			return kv, recurrent
		}
		if _, explicit := m.numeric[m.arch+".attention.recurrent_layers"]; next > 0 && hybrid && explicit {
			// models/qwen35.cpp:17-21 gives an explicit recurrent mask
			// precedence over the interval; that mapping is not priced here.
			kv.Reason = "NextN explicit recurrent layer mapping unconfirmed"
			return kv, recurrent
		}
		cacheLayers -= next
	}
	if n, exists := m.numeric[m.arch+".attention.shared_kv_layers"]; exists {
		count, valid := modelEstimateUint(n)
		// models/gemma4.cpp:7-10, llama-hparams.cpp:320-327 and
		// llama-kv-cache.cpp:168-171: trailing shared layers allocate no K/V.
		// Reuse requires two leading caches (llama-model.cpp:2661-2669).
		if !valid || count > 0 && (!shared || layers < 2 || count > layers-2 || cacheLayers != layers) {
			kv.Reason = "shared-KV architecture or leading cache mapping unconfirmed"
			return kv, recurrent
		}
		cacheLayers -= count
	}
	window, hasWindow := m.number("attention.sliding_window")
	// models/llama4.cpp:8-19 selects chunked iSWA and overrides a
	// nonzero (or absent) window with 8192. Malformed is not absent.
	if m.arch == "llama4" {
		if _, exists := m.numeric[m.arch+".attention.sliding_window"]; !exists || hasWindow && window > 0 {
			window, hasWindow = 8192, true
		}
	}
	if mla && (window > 0 || m.numeric[m.arch+".attention.sliding_window_pattern"].Array != nil) {
		kv.Reason = "MLA SWA combination unmapped"
		return kv, recurrent
	}
	if shared && (!hasWindow || window == 0) {
		kv.Reason = "shared-KV SWA window unavailable"
		return kv, recurrent
	}
	if m.arch == "gpt-oss" && (!hasWindow || window == 0) {
		// models/openai-moe.cpp:6-9 requires a scalar window and keeps
		// standard iSWA even for zero. Neither case is ordinary dense KV.
		kv.Reason = "GPT-OSS native SWA window missing or zero; iSWA mapping unconfirmed"
		return kv, recurrent
	}
	if _, exists := m.numeric[m.arch+".attention.sliding_window"]; exists && !hasWindow {
		kv.Reason = "malformed SWA window"
		if m.numeric[m.arch+".attention.sliding_window"].Array != nil {
			kv.Reason = "per-layer SWA window lengths unsupported by pinned scalar n_swa"
		}
		return kv, recurrent
	}
	_, hasPattern := m.numeric[m.arch+".attention.sliding_window_pattern"]
	swaLayers := make([]bool, layers)
	swaKey, swaValue := key, value
	hasWidths := false
	for _, field := range []string{"attention.key_length_swa", "attention.value_length_swa"} {
		if _, exists := m.numeric[m.arch+"."+field]; !exists {
			continue
		}
		hasWidths = true
		n, valid := m.number(field)
		if !valid || n == 0 {
			kv.Reason = "malformed SWA key/value widths"
			return kv, recurrent
		}
		if field == "attention.key_length_swa" {
			swaKey = n
		} else {
			swaValue = n
		}
	}
	// Gemma's native SWA layout must not silently become full attention when
	// bounded metadata omits a window or the library supplies its own default.
	if window == 0 && (hasPattern || hasWidths || m.arch == "gemma2" || m.arch == "gemma3") {
		kv.Reason = "SWA window unavailable for the native layer mapping"
		return kv, recurrent
	}
	if window > 0 {
		if shared {
			// Bounded Task3 subset: full cache with equal global/SWA widths.
			// llama-kv-cache-iswa.cpp:74-81 makes both cache sizes identical.
			// Compact shared-KV and unequal widths remain unconfirmed here.
			pattern, exists := m.numeric[m.arch+".attention.sliding_window_pattern"]
			if !exists || len(pattern.Array) != int(layers) || in.SWAFull == nil || !*in.SWAFull || key != value || swaKey != key || swaValue != value {
				kv.Reason = "shared-KV requires full cache and equal widths; SWA variant mapping deferred"
				return kv, recurrent
			}
			if *a.Options.FlashAttention == "on" {
				// llama-context.cpp:3792-3809 validates quantization blocks
				// per head, before GQA combines multiple heads into a row.
				_, validK := modelEstimateRow(kt, key)
				_, validV := modelEstimateRow(vt, value)
				if !validK || !validV {
					kv.Reason = "shared-KV per-head cache precision incompatible with native block size"
					return kv, recurrent
				}
			}
			// llama-model-loader.cpp:365-380 accepts bool/uint32/int32
			// arrays for the native integer pattern, not float bit patterns.
			if pattern.Type != 7 && pattern.Type != 4 && pattern.Type != 5 {
				kv.Reason = "shared-KV SWA pattern element type unsupported"
				return kv, recurrent
			}
			for _, bit := range pattern.Array {
				if bit > 1 {
					kv.Reason = "shared-KV SWA pattern malformed"
					return kv, recurrent
				}
			}
			if cacheLayers < 2 || pattern.Array[cacheLayers-2] != 1 || pattern.Array[cacheLayers-1] != 0 {
				kv.Reason = "shared-KV leading reuse sources do not match native SWA/global order"
				return kv, recurrent
			}
			if *a.Options.FlashAttention == "off" && isArray {
				// llama-kv-cache.cpp:209-210 pads transposed V to the
				// maximum width across all layers. Keep that variant unpriced.
				for _, heads := range perHeads.Array {
					if heads != perHeads.Array[0] {
						kv.Reason = "shared-KV transposed V padding for variable heads unconfirmed"
						return kv, recurrent
					}
				}
			}
			// Uniform full-size caches do not depend on the SWA layer mask.
			window = 0
		} else {
			// models/lfm2moe.cpp does not enable SWA, unlike lfm2.cpp:26-30.
			// Do not reuse LFM2's all-attention-layer mapping for that family.
			if !slices.Contains([]string{"gemma2", "gemma3", "llama4", "gpt-oss", "lfm2"}, m.arch) {
				kv.Reason = "SWA native architecture mapping unconfirmed"
				return kv, recurrent
			}
			if hasPattern && m.arch == "lfm2" {
				kv.Reason = "LFM2 SWA pattern override mapping unconfirmed; native marks every attention layer"
				return kv, recurrent
			}
			if _, exists := m.numeric[m.arch+".attention.sliding_window_pattern"]; !exists && m.arch != "lfm2" {
				kv.Reason = "SWA per-layer pattern unavailable"
				return kv, recurrent
			}
			if in.SWAFull == nil {
				kv.Reason = "effective SWA full-cache mode unavailable"
				return kv, recurrent
			}
			if !*in.SWAFull && (a.Options.UBatchSize == nil || *a.Options.UBatchSize <= 0 || a.Options.BatchSize == nil || *a.Options.BatchSize <= 0 || *a.Options.UBatchSize > *a.Options.BatchSize || *a.Options.BatchSize > 4096) {
				kv.Reason = "SWA effective microbatch unavailable"
				return kv, recurrent
			}
			if pattern, exists := m.numeric[m.arch+".attention.sliding_window_pattern"]; exists {
				// llama-model.cpp:3308-3314 reads arrays before scalars;
				// llama-model-loader.cpp:365-380 accepts bool/u32/i32.
				if pattern.Array != nil {
					if len(pattern.Array) != int(layers) || pattern.Type != 7 && pattern.Type != 4 && pattern.Type != 5 {
						kv.Reason = "SWA native per-layer pattern length or element type unsupported"
						return kv, recurrent
					}
					for i, bit := range pattern.Array {
						if bit > 1 {
							kv.Reason = "SWA native per-layer pattern malformed"
							return kv, recurrent
						}
						swaLayers[i] = bit == 1
					}
				} else {
					period, valid := modelEstimateUint(pattern)
					if !valid || period > layers {
						kv.Reason = "SWA declared pattern malformed or unmapped"
						return kv, recurrent
					}
					// llama-hparams.cpp:8-21, dense_first=false.
					for i := range swaLayers {
						swaLayers[i] = period == 0 || uint64(i)%period < period-1
					}
				}
			} else {
				// models/lfm2.cpp:26-30: all attention layers use SWA.
				for i := range swaLayers {
					swaLayers[i] = true
				}
			}
			if !slices.Contains(swaLayers, true) {
				kv.Reason = "SWA pattern has no windowed layer; native iSWA assertion would fail"
				return kv, recurrent
			}
		}
	}
	// Native transposed V pads to the maximum GQA width across all layers,
	// including layers filtered into the other iSWA cache (kv-cache.cpp:209).
	if *a.Options.FlashAttention == "off" {
		variable := window > 0 && swaValue != value
		if isArray {
			var first uint64
			for _, h := range perHeads.Array {
				if h == 0 && hybrid {
					continue
				}
				if first == 0 {
					first = h
				}
				variable = variable || h != first
			}
		}
		if variable {
			kv.Reason = "transposed V padding for variable per-layer widths unconfirmed"
			return kv, recurrent
		}
	}
	var cacheBytes uint64
	for i := uint64(0); i < cacheLayers; i++ {
		if ctx.Err() != nil {
			kv.Reason = "estimator operation budget exhausted"
			return kv, recurrent
		}
		h := hkv
		if isArray {
			var valid bool
			h, valid = modelEstimateUint(gguffacts.Numeric{Type: perHeads.Type, Bits: perHeads.Array[i]})
			if !valid || h > heads {
				kv.Reason = "invalid per-layer KV heads"
				return kv, recurrent
			}
		} else if hybrid && (i+1)%interval != 0 {
			h = 0
		}
		if h > heads {
			kv.Reason = "KV heads exceed attention heads"
			return kv, recurrent
		}
		if hybrid && h == 0 {
			continue
		}
		if h == 0 {
			kv.Reason = "zero KV heads without mapped recurrent layer"
			return kv, recurrent
		}
		cells := uint64(a.TotalContext)
		k, v := key, value
		if window > 0 && swaLayers[i] {
			perStream := uint64(a.PerSlotContext)
			if !*in.SWAFull {
				perStream = gguf.GGMLPadding(min(perStream, modelEstimateAdd(window, uint64(*a.Options.UBatchSize))), 256)
			}
			cells = modelEstimateMul(perStream, uint64(a.Slots))
			k, v = swaKey, swaValue
		}
		if *a.Options.FlashAttention == "on" {
			// llama-context.cpp:3792-3809 checks block alignment per head.
			_, validK := modelEstimateRow(kt, k)
			_, validV := modelEstimateRow(vt, v)
			if !validK || !validV {
				kv.Reason = "per-head cache precision incompatible with native block size"
				return kv, recurrent
			}
		}
		kb, validK := modelEstimateRow(kt, modelEstimateMul(k, h), cells)
		vb, validV := modelEstimateRow(vt, modelEstimateMul(v, h), cells)
		// llama-kv-cache.cpp:229-234: MLA has K only. FA changes the
		// attention operation, not this allocation (llama-model.cpp:2742).
		if mla {
			// Context validation still checks V precision/block divisibility
			// for MLA (llama-context.cpp:3803-3809), despite no V storage.
			vb = 0
		}
		if !validK || !validV {
			kv.Reason = "K/V row width incompatible with cache block size"
			return kv, recurrent
		}
		if !deviceOnly || modelEstimateLayerOnDevice(in, layers, i) {
			cacheBytes = modelEstimateAdd(cacheBytes, modelEstimateAdd(kb, vb))
		}
	}
	kv = estimateComponent("attention-cache", cacheBytes, "per-layer native head widths; independently block-sized K/V; total context already includes slots")
	if mla {
		kv.Reason = "native MLA latent+rotary K only; main-context NextN filter; total context already includes slots"
	} else if shared {
		kv.Reason = "native leading shared-KV caches only; full SWA cache with equal head widths"
	}
	return kv, recurrent
}

// Validate only graph inputs consumed by the heuristic, not native loadability.
// Tensor owners: 7fe450e1 src/models/{qwen2moe,qwen3moe,deepseek2,lfm2}.cpp.
func modelEstimateSpecialComputeLayout(m *modelEstimateModel) string {
	emb, _ := m.number("embedding_length")
	ffn, _ := m.number("feed_forward_length")
	layers, _ := m.number("block_count")
	heads, _ := m.number("attention.head_count")
	shape := func(name string, dims ...uint64) bool {
		t, ok := m.file.TensorInfos.Get(name)
		return ok && slices.Equal(t.Dimensions, dims)
	}
	experts, moe := m.number("expert_count")
	used, usedOK := m.number("expert_used_count")
	width, widthOK := m.number("expert_feed_forward_length")
	shared, _ := m.number("expert_shared_feed_forward_length")
	// Native qwen{2,3}moe.cpp uses n_ff/n_expert_used when width is
	// omitted. The parser's Architecture does not apply that default.
	// Pass the proven native width, then require matching expert tensors below.
	set := func(key string, n uint64) {
		m.file.Header.MetadataKV = append(m.file.Header.MetadataKV, gguf.GGUFMetadataKV{Key: m.arch + "." + key, ValueType: gguf.GGUFMetadataValueTypeUint64, Value: n})
	}
	if _, exists := m.numeric[m.arch+".expert_feed_forward_length"]; !exists && (m.arch == "qwen2moe" || m.arch == "qwen3moe") && usedOK && used > 0 && ffn%used == 0 {
		width, widthOK = ffn/used, true
		set("expert_feed_forward_length", width)
	}
	if _, exists := m.numeric[m.arch+".expert_shared_feed_forward_length"]; !exists && m.arch == "qwen2moe" {
		shared = ffn
		set("expert_shared_feed_forward_length", shared)
	}
	if m.arch == "lfm2" && (experts != 0 || used != 0) {
		return "LFM2 dense graph contains unconfirmed experts"
	}
	if moe && experts > 0 {
		if !usedOK || used == 0 || used > experts || experts > 65536 {
			return "MoE expert_used_count missing or inconsistent"
		}
		if !widthOK || width == 0 || width > 65536 {
			return "MoE expert_feed_forward_length missing or unconfirmed"
		}
	} else if m.arch != "lfm2" {
		return "MoE expert_count unavailable"
	}
	if m.arch == "qwen2moe" && shared == 0 {
		return "MoE expert_shared_feed_forward_length missing or unconfirmed"
	}
	if m.arch == "deepseek2" {
		count, ok := m.number("expert_shared_count")
		if _, exists := m.numeric[m.arch+".expert_shared_feed_forward_length"]; !exists && ok {
			shared = modelEstimateMul(count, width)
			set("expert_shared_feed_forward_length", shared)
		}
		if !ok || shared != modelEstimateMul(count, width) {
			return "MLA shared expert width/count unconfirmed"
		}
	}
	for i := uint64(0); i < layers; i++ {
		p := "blk." + strconv.FormatUint(i, 10) + "."
		if !shape(p+"attn_norm.weight", emb) || !shape(p+"ffn_norm.weight", emb) {
			return "graph layer norms incomplete: " + p
		}
		if moe && experts > 0 {
			if !shape(p+"ffn_gate_inp.weight", emb, experts) || !shape(p+"ffn_gate_exps.weight", emb, width, experts) || !shape(p+"ffn_up_exps.weight", emb, width, experts) || !shape(p+"ffn_down_exps.weight", width, emb, experts) {
				return "MoE active expert tensor width incomplete: " + p
			}
			if shared > 0 && (!shape(p+"ffn_gate_shexp.weight", emb, shared) || !shape(p+"ffn_up_shexp.weight", emb, shared) || !shape(p+"ffn_down_shexp.weight", shared, emb)) {
				return "MoE shared expert tensors incomplete: " + p
			}
		} else if !shape(p+"ffn_gate.weight", emb, ffn) || !shape(p+"ffn_up.weight", emb, ffn) || !shape(p+"ffn_down.weight", ffn, emb) {
			return "hybrid FFN tensors incomplete: " + p
		}
		if m.arch == "deepseek2" {
			q, qok := m.number("attention.q_lora_rank")
			kv, kok := m.number("attention.kv_lora_rank")
			k, keyOK := m.number("attention.key_length_mla")
			v, valueOK := m.number("attention.value_length_mla")
			rope, rok := m.number("rope.dimension_count")
			// deepseek2.cpp:114-119 selects split K/V projections for MLA;
			// the legacy unsplit attn_kv_b tensor belongs to its non-MLA branch.
			if !qok || !kok || !keyOK || !valueOK || !rok || q == 0 || kv == 0 || k <= rope || v == 0 || k > 65536 || v > 65536 || q > 65536 || kv > 65536 || !shape(p+"attn_q_a.weight", emb, q) || !shape(p+"attn_q_a_norm.weight", q) || !shape(p+"attn_q_b.weight", q, heads*k) || !shape(p+"attn_kv_a_mqa.weight", emb, kv+rope) || !shape(p+"attn_kv_a_norm.weight", kv) || !shape(p+"attn_k_b.weight", k-rope, kv, heads) || !shape(p+"attn_v_b.weight", kv, v, heads) || !shape(p+"attn_output.weight", heads*v, emb) {
				return "MLA query/latent/rotary graph inputs unconfirmed: " + p
			}
			continue
		}
		hkv, ok := m.number("attention.head_count_kv")
		if m.arch == "lfm2" {
			n := m.numeric[m.arch+".attention.head_count_kv"]
			if len(n.Array) != int(layers) {
				return "hybrid per-layer KV heads unavailable"
			}
			hkv, ok = modelEstimateUint(gguffacts.Numeric{Type: n.Type, Bits: n.Array[i]})
			if ok && hkv == 0 {
				cache, valid := m.number("shortconv.l_cache")
				if !valid || cache == 0 || cache > 65536 || !shape(p+"shortconv.conv.weight", cache, emb) || !shape(p+"shortconv.in_proj.weight", emb, 3*emb) || !shape(p+"shortconv.out_proj.weight", emb, emb) {
					return "hybrid shortconv graph inputs unavailable: " + p
				}
				continue
			}
			// lfm2.cpp:69-72 creates per-head Q/K normalization weights.
			if !shape(p+"attn_q_norm.weight", emb/heads) || !shape(p+"attn_k_norm.weight", emb/heads) {
				return "hybrid attention normalization tensors incomplete: " + p
			}
		}
		if !ok || hkv == 0 || hkv > heads || !shape(p+"attn_q.weight", emb, emb) || !shape(p+"attn_k.weight", emb, emb/heads*hkv) || !shape(p+"attn_v.weight", emb, emb/heads*hkv) || !shape(p+"attn_output.weight", emb, emb) {
			return "attention graph tensors incomplete: " + p
		}
	}
	return ""
}

func modelEstimateCompute(ctx context.Context, m modelEstimateModel, in modelEstimateInput) (out modelconfig.EstimateComponent) {
	return modelEstimateComputePortion(ctx, m, in, false)
}

func modelEstimateComputePortion(ctx context.Context, m modelEstimateModel, in modelEstimateInput, device bool) (out modelconfig.EstimateComponent) {
	defer func() {
		if recovered := recover(); recovered != nil {
			reason := "pinned library compute panic"
			if err, ok := recovered.(error); ok && errors.Is(err, errModelEstimateOverflow) {
				reason = "compute arithmetic overflow"
			}
			out = unknownEstimateComponent("compute-workspace", reason)
		}
	}()
	u := unknownEstimateComponent("compute-workspace", "architecture/native graph mapping or effective batch/FA options unavailable")
	if in.Allocation == nil || in.Support != modelEstimatorSupport || in.BuildID == "" || !slices.Contains([]string{"llama", "qwen2", "qwen3", "qwen2moe", "qwen3moe", "deepseek2", "lfm2"}, m.arch) {
		return u
	}
	a := in.Allocation
	if a.Options.BatchSize == nil || a.Options.UBatchSize == nil || a.Options.FlashAttention == nil || *a.Options.FlashAttention == "inherit" {
		return u
	}
	batch, ubatch := *a.Options.BatchSize, *a.Options.UBatchSize
	if batch < 32 || batch > 4096 || ubatch < 1 || ubatch > batch {
		u.Reason = "batch limits or library minimum batch mismatch"
		return u
	}
	special := slices.Contains([]string{"qwen2moe", "qwen3moe", "deepseek2", "lfm2"}, m.arch)
	for _, field := range []string{"block_count", "embedding_length", "feed_forward_length", "attention.head_count", "vocab_size"} {
		v, ok := m.number(field)
		if !ok || v == 0 || v > 65536 || field == "block_count" && v > modelEstimateLayers {
			u.Reason = "dense graph required scalar dimension missing/outside range: " + field
			return u
		}
	}
	for _, field := range []string{"expert_count", "attention.sliding_window", "attention.kv_lora_rank", "attention.key_length_mla", "attention.value_length_mla", "full_attention_interval", "attention.shared_kv_layers", "nextn_predict_layers"} {
		if special && slices.Contains([]string{"expert_count", "attention.kv_lora_rank", "attention.key_length_mla", "attention.value_length_mla"}, field) {
			continue
		}
		if n, exists := m.numeric[m.arch+"."+field]; exists {
			v, ok := modelEstimateUint(n)
			if !ok || v != 0 {
				u.Reason = "dense graph excludes MoE/SWA/MLA/hybrid/shared/NextN"
				return u
			}
		}
	}
	heads, _ := m.number("attention.head_count")
	emb, _ := m.number("embedding_length")
	hkv, _ := m.number("attention.head_count_kv")
	if emb%heads != 0 || hkv > heads || !special && hkv == 0 {
		u.Reason = "dense graph attention dimensions inconsistent"
		return u
	}
	for _, field := range []string{"attention.key_length", "attention.value_length"} {
		if _, exists := m.numeric[m.arch+"."+field]; exists {
			v, ok := m.number(field)
			if !ok || v == 0 || v > 65536 {
				u.Reason = "dense graph key/value width malformed"
				return u
			}
		}
	}
	// Refuse malformed/special state metadata rather than letting the library
	// infer a recurrent graph under a nominally dense architecture.
	for key := range m.numeric {
		if strings.Contains(key, ".ssm.") || strings.Contains(key, ".shortconv.") && m.arch != "lfm2" || strings.Contains(key, ".wkv.") {
			u.Reason = "dense graph contains unmapped recurrent metadata"
			return u
		}
	}
	// Preflight library element sums and a conservative arithmetic envelope.
	var elements uint64
	for _, tensor := range m.file.TensorInfos {
		n := uint64(1)
		for _, dim := range tensor.Dimensions {
			if dim > 1<<20 {
				u.Reason = "dense graph tensor dimension outside arithmetic envelope"
				return u
			}
			n = modelEstimateMul(n, dim)
		}
		elements = modelEstimateAdd(elements, n)
	}
	if elements > 1<<40 || m.weightsUnknown || len(m.file.TensorInfos) == 0 {
		u.Reason = "dense graph tensor mapping incomplete or outside arithmetic envelope"
		return u
	}
	// The graph heuristic must see complete conventional dense tensor shapes;
	// a directory containing only one recognizable projection is insufficient.
	shape := func(name string, dims ...uint64) bool {
		tensor, ok := m.file.TensorInfos.Get(name)
		return ok && slices.Equal(tensor.Dimensions, dims)
	}
	ffn, _ := m.number("feed_forward_length")
	vocab, _ := m.number("vocab_size")
	layers, _ := m.number("block_count")
	if special {
		if reason := modelEstimateSpecialComputeLayout(&m); reason != "" {
			u.Reason = reason
			return u
		}
	}
	norm := "output_norm.weight"
	if m.arch == "lfm2" {
		// lfm2.cpp:39 / llama-arch.cpp:436 names OUTPUT_NORM_LFM2.
		norm = "token_embd_norm.weight"
	}
	if !shape("token_embd.weight", emb, vocab) || !shape(norm, emb) {
		u.Reason = "dense graph embedding/output shapes incomplete"
		return u
	}
	for _, field := range []string{"attention.key_length", "attention.value_length"} {
		if _, exists := m.numeric[m.arch+"."+field]; exists {
			width, _ := m.number(field)
			if width != emb/heads && m.arch != "deepseek2" {
				u.Reason = "dense graph unequal head widths unmapped"
				return u
			}
		}
	}
	for i := uint64(0); i < layers; i++ {
		if special {
			break
		}
		prefix := "blk." + strconv.FormatUint(i, 10) + "."
		if !shape(prefix+"attn_norm.weight", emb) || !shape(prefix+"attn_q.weight", emb, emb) || !shape(prefix+"attn_k.weight", emb, emb/heads*hkv) || !shape(prefix+"attn_v.weight", emb, emb/heads*hkv) || !shape(prefix+"attn_output.weight", emb, emb) || !shape(prefix+"ffn_norm.weight", emb) || !shape(prefix+"ffn_gate.weight", emb, ffn) || !shape(prefix+"ffn_up.weight", emb, ffn) || !shape(prefix+"ffn_down.weight", ffn, emb) {
			u.Reason = "dense graph required tensor shape incomplete: " + prefix
			return u
		}
	}
	kt, kok := modelEstimateCacheType(a.Options.CacheTypeK)
	vt, vok := modelEstimateCacheType(a.Options.CacheTypeV)
	if !kok || !vok || vt.IsQuantized() && *a.Options.FlashAttention != "on" {
		return u
	}
	if ctx.Err() != nil {
		u.Reason = "estimator operation budget exhausted"
		return u
	}
	offload := int32(0)
	if a.GPU.Mode == "layers" && a.GPU.Layers != nil && *a.GPU.Layers >= 0 && uint64(*a.GPU.Layers) <= layers+1 {
		offload = *a.GPU.Layers
	} else if a.GPU.Mode != "cpu" {
		u.Reason = "finite single-device layer placement unavailable; multi-device splits unsupported"
		return u
	}
	if offload == 0 && (device || a.Options.OpOffload == nil || *a.Options.OpOffload) {
		u.Reason = "CPU graph requires explicit op-offload=false; native offload mapping unavailable"
		return u
	}
	if offload > 0 && modelEstimateBackend(in) == "" {
		u.Reason = "single-device native inventory unavailable"
		return u
	}
	if offload > 0 {
		count := 0
		for _, d := range in.Memory.Map.Devices {
			if d.Kind != 0 {
				count++
			}
		}
		if count != 1 {
			u.Reason = "multi-device compute split unconfirmed"
			return u
		}
	}
	opts := []gguf.GGUFRunEstimateOption{gguf.WithLLaMACppContextSize(a.TotalContext), gguf.WithParallelSize(a.Slots), gguf.WithLLaMACppLogicalBatchSize(batch), gguf.WithLLaMACppPhysicalBatchSize(ubatch), gguf.WithLLaMACppCacheKeyType(kt), gguf.WithLLaMACppCacheValueType(vt), gguf.WithLLaMACppOffloadLayers(uint64(offload))}
	if *a.Options.FlashAttention == "on" {
		opts = append(opts, gguf.WithFlashAttention())
	}
	result := m.file.EstimateLLaMACppRun(opts...)
	if len(result.Devices) > 2 {
		u.Reason = "library device limit"
		return u
	}
	if len(result.Devices) == 0 {
		return u
	}
	// The pinned estimator reserves a hypothetical GPU scratch buffer even at
	// zero GPU layers and has no op-offload=false option. Do not relabel that
	// buffer as host memory. Only its CPU portion is mapped here; full CPU
	// scratch remains unsized rather than summing fictitious device allocations.
	c := result.Devices[0].Computation
	if device {
		if len(result.Devices) != 2 {
			u.Reason = "library device computation unavailable"
			return u
		}
		c = result.Devices[1].Computation
	}
	// Computation.Sum reuses compute/output storage, taking their maximum.
	total := modelEstimateAdd(modelEstimateAdd(uint64(c.Footprint), uint64(c.Input)), max(uint64(c.Compute), uint64(c.Output)))
	if total == 0 {
		return u
	}
	component := estimateComponent("compute-workspace", total, "pinned library graph portion only; native scratch, allocator and transient peaks unpriced; no hypothetical device buffer charged at zero GPU layers")
	component.Status = "partial"
	component.Basis = "heuristic"
	component.DomainID = "host"
	if device {
		component.DomainID = "offload-unmapped"
	}
	return component
}

func modelEstimatePlacement(out *modelconfig.Estimate, m modelEstimateModel, in modelEstimateInput) {
	out.Domains = []modelconfig.MemoryDomain{{ID: "host", Kind: "host"}, {ID: "offload-unmapped", Kind: "device"}}
	if in.Allocation == nil {
		out.Unknowns = append(out.Unknowns, "placement: effective allocation unavailable")
		return
	}
	a := in.Allocation
	cpu := a.GPU.Mode == "cpu" || a.GPU.Mode == "layers" && a.GPU.Layers != nil && *a.GPU.Layers == 0
	layers, valid := m.number("block_count")
	if !cpu && (!valid || layers == 0 || layers > modelEstimateLayers || a.GPU.Mode != "layers" || a.GPU.Layers == nil || *a.GPU.Layers < 0 || uint64(*a.GPU.Layers) > layers+1) {
		out.Unknowns = append(out.Unknowns, "placement: finite layer mapping unavailable")
		return
	}
	var host, device uint64
	for i := range out.Components {
		c := &out.Components[i]
		if c.Bytes == "" {
			continue
		}
		n, err := strconv.ParseUint(c.Bytes, 10, 64)
		if err != nil {
			panic(errModelEstimateOverflow)
		}
		if c.Name == "companions" {
			continue
		}
		if c.Name == "compute-workspace" && c.DomainID != "" {
			if c.DomainID == "host" {
				host = modelEstimateAdd(host, n)
			} else {
				device = modelEstimateAdd(device, n)
			}
			continue
		}
		if cpu || c.Name == "prompt-cache" {
			c.DomainID = "host"
			host = modelEstimateAdd(host, n)
			continue
		}
		if c.Name == "weights" {
			// llama-model.cpp:1521-1546 at 7fe450e1: start = blocks+1-ngl;
			// output has index blocks. Thus ngl=1 selects output, then trailing
			// blocks. Embeddings and unclassified tensors remain on host.
			gpuLayers := uint64(*a.GPU.Layers)
			var offloadBlocks uint64
			if gpuLayers > 0 {
				offloadBlocks = gpuLayers - 1
			}
			var h, d uint64
			var embedding uint64
			hasOutput := false
			for _, shard := range in.Facts.Shards {
				if shard.Role == gguffacts.RoleMain {
					for _, tensor := range shard.Tensors {
						if tensor.Name == "output.weight" {
							hasOutput = true
						}
					}
				}
			}
			for _, tensor := range m.file.TensorInfos {
				bytes, ok := modelEstimateRow(tensor.Type, tensor.Dimensions...)
				if !ok {
					continue
				}
				if tensor.Name == "token_embd.weight" {
					embedding = bytes
				}
				onGPU := gpuLayers > 0 && (tensor.Name == "output.weight" || tensor.Name == "output_norm.weight")
				if strings.HasPrefix(tensor.Name, "blk.") {
					rest := strings.TrimPrefix(tensor.Name, "blk.")
					id, _, found := strings.Cut(rest, ".")
					layer, err := strconv.ParseUint(id, 10, 64)
					if found && err == nil && layer < layers {
						onGPU = layer >= layers-offloadBlocks
					}
				}
				if onGPU {
					d = modelEstimateAdd(d, bytes)
				} else {
					h = modelEstimateAdd(h, bytes)
				}
			}
			if gpuLayers > 0 && !hasOutput && (m.arch == "qwen2" || m.arch == "gemma2") && embedding > 0 {
				d = modelEstimateAdd(d, embedding)
				c.Reason += "; tied output device copy is a projection; buffer-type selection and fallback unverified"
				out.Unknowns = append(out.Unknowns, "tied output: device buffer-type selection and fallback unverified")
			}
			host = modelEstimateAdd(host, h)
			device = modelEstimateAdd(device, d)
			c.Reason += "; output-first tensor placement projection; embedding/unclassified tensors on host"
			c.DomainID, c.Bytes = "host", strconv.FormatUint(h, 10)
			c.LowBytes, c.HighBytes = c.Bytes, c.Bytes
			if d > 0 {
				part := *c
				part.DomainID, part.Bytes = "offload-unmapped", strconv.FormatUint(d, 10)
				part.LowBytes, part.HighBytes = part.Bytes, part.Bytes
				out.Components = append(out.Components, part)
			}
		} else if c.Name == "attention-cache" || c.Name == "recurrent-state" {
			if a.Options.KVOffload != nil && !*a.Options.KVOffload {
				c.DomainID = "host"
				host = modelEstimateAdd(host, n)
			} else if a.Options.KVOffload != nil {
				var part modelconfig.EstimateComponent
				if c.Name == "recurrent-state" {
					part = modelEstimateRecurrentForLayers(m, in, true)
				} else {
					part, _ = modelEstimateCachesForLayers(context.Background(), m, in, true)
				}
				d, err := strconv.ParseUint(part.Bytes, 10, 64)
				if err != nil || part.Status != "estimated" || d > n {
					out.Unknowns = append(out.Unknowns, c.Name+": native per-layer device placement unavailable")
					continue
				}
				h := n - d
				host, device = modelEstimateAdd(host, h), modelEstimateAdd(device, d)
				c.DomainID, c.Bytes = "host", strconv.FormatUint(h, 10)
				c.LowBytes, c.HighBytes = c.Bytes, c.Bytes
				if h == 0 {
					c.DomainID, c.Bytes = "offload-unmapped", strconv.FormatUint(d, 10)
					c.LowBytes, c.HighBytes = c.Bytes, c.Bytes
				} else if d > 0 {
					part := *c
					part.DomainID, part.Bytes = "offload-unmapped", strconv.FormatUint(d, 10)
					part.LowBytes, part.HighBytes = part.Bytes, part.Bytes
					out.Components = append(out.Components, part)
				}
			} else {
				out.Unknowns = append(out.Unknowns, c.Name+": per-layer device placement unresolved; no layer-fraction approximation used")
			}
		}
	}
	out.HostBytes, out.DeviceBytes = strconv.FormatUint(host, 10), strconv.FormatUint(device, 10)
	modelEstimateDomainAmounts(out)
	if !cpu {
		out.Unknowns = append(out.Unknowns, "native offload device identity unavailable; device bytes are an unmapped aggregate")
	}
}

func modelEstimateMemoryFresh(m *modelEstimateMemory, now time.Time) bool {
	return m.Identity != "" && len(m.Identity) <= 256 && m.Source != "" && len(m.Source) <= 256 && !now.Before(m.At) && now.Sub(m.At) <= 5*time.Second
}

func modelEstimateDomains(out *modelconfig.Estimate, in modelEstimateInput, now time.Time) {
	// Cached computation is reusable, observation validity is always rechecked.
	for i := range out.Domains {
		d := &out.Domains[i]
		d.CapacityFit, d.FreeFit = "unknown", "unknown"
		d.RemainingValidityMS = 0
		d.ReserveBytes = ""
	}
	if in.Memory == nil || in.Memory.Target == nil {
		modelEstimateDomainsLegacy(out, in, now)
		if in.Memory != nil && modelEstimateMemoryFresh(in.Memory, now) {
			remaining := max(int64(0), noderec.MemoryFreshMS-int64((now.Sub(in.Memory.At)+time.Millisecond-1)/time.Millisecond))
			for i := range out.Domains {
				out.Domains[i].RemainingValidityMS = min(out.Domains[i].RemainingValidityMS, int64(remaining))
			}
		}
		for i := range out.Domains {
			d := &out.Domains[i]
			if d.Kind != "device" {
				continue
			}
			if in.Allocation == nil || in.Allocation.DeviceBinding == nil || in.Allocation.DeviceBinding.Validate() != nil || in.Allocation.DeviceBinding.DomainID != d.ID || in.Allocation.DeviceBinding.DeviceID != d.DeviceID {
				d.RemainingValidityMS = 0
				d.Unknowns = append(d.Unknowns, "GPU allocation has no matching reviewed physical binding")
			}
		}
		*out = modelconfig.CompareEstimate(*out)
		return
	}
	m := in.Memory
	observation := m.Target.At(noderec.MemoryBootClock)
	cpu := in.Allocation != nil && (in.Allocation.GPU.Mode == "cpu" || in.Allocation.GPU.Mode == "layers" && in.Allocation.GPU.Layers != nil && *in.Allocation.GPU.Layers == 0)
	if observation == nil {
		out.Unknowns = append(out.Unknowns, "target observation or native map unavailable")
		*out = modelconfig.CompareEstimate(*out)
		return
	}
	mapKnown := m.Map != nil && m.Map.Unknown == "" && m.Map.ObservationID != "" && m.Map.ObservationID == m.Target.MapID
	if !cpu && !mapKnown {
		out.Unknowns = append(out.Unknowns, "native map unavailable; device placement unproved")
	}
	// CPU placement needs only the independently identified host/shared pool.
	// It never needs GPU enumeration, nor charges an empty GPU domain.
	hostPool := ""
	if cpu {
		for _, d := range observation.Domains {
			if d.Kind == "system" || d.Kind == "shared" {
				hostPool = d.ID
			}
		}
		out.Domains = slices.DeleteFunc(out.Domains, func(d modelconfig.MemoryDomain) bool { return d.Kind == "device" && d.RequiredBytes == "0" })
	} else if mapKnown {
		hostPool = m.Map.HostPool
	} else {
		// Host observations are independent of the unavailable device map.
		for _, d := range observation.Domains {
			if d.Kind == "system" || d.Kind == "shared" {
				hostPool = d.ID
			}
		}
	}
	// A trusted, unmixed UMA topology has one physical pool. Collapse before
	// attaching capacity, preserving each component's charge and band once.
	if !cpu && mapKnown && len(observation.Domains) == 1 && observation.Domains[0].Kind == "shared" && observation.Domains[0].ID == hostPool {
		for i := range out.Components {
			if out.Components[i].DomainID == "host" || out.Components[i].DomainID == "offload-unmapped" {
				out.Components[i].DomainID = hostPool
			}
		}
		out.Domains = []modelconfig.MemoryDomain{{ID: hostPool, Kind: "shared"}}
		modelEstimateDomainAmounts(out)
		out.HostBytes, out.DeviceBytes = "", ""
		out.Assumptions = append(out.Assumptions, "UMA is one shared pool; host and GPU capacity are never added.")
	}
	var assumed *modelNativeDevice
	if !cpu && mapKnown && in.Allocation != nil && in.Allocation.DeviceBinding == nil && modelEstimateBackend(in) == "cuda" {
		// QA1 is a display comparison, never a synthesized launch binding.
		var devices []modelNativeDevice
		for _, device := range m.Map.Devices {
			if device.Kind != 0 {
				devices = append(devices, device)
			}
		}
		if len(devices) == 1 && devices[0].Kind == 1 && devices[0].Unknown == "" && canonicalModelDeviceUUID(devices[0].UUID) != "" {
			var dedicated []noderec.MemoryDomain
			for _, fact := range observation.Domains {
				if fact.Kind == "dedicated" {
					dedicated = append(dedicated, fact)
				}
			}
			if len(dedicated) == 1 && dedicated[0].ID == devices[0].Pool && slices.Contains(dedicated[0].DeviceIDs, devices[0].UUID) {
				assumed = &devices[0]
			}
		}
	}
	for i := range out.Domains {
		d := &out.Domains[i]
		if !cpu && !mapKnown && (d.Kind == "device" || d.ID == "offload-unmapped") {
			d.Unknowns = append(d.Unknowns, "native map unavailable; device placement unproved")
			continue
		}
		pool := d.ID
		if d.Kind == "host" {
			pool = hostPool
		}
		if d.ID == "offload-unmapped" {
			if assumed != nil {
				pool = assumed.Pool
				d.Unknowns = append(d.Unknowns, "device mapping assumed: only one GPU present")
			} else {
				b := m.Map.Binding
				if b == nil || in.Allocation == nil || in.Allocation.DeviceBinding == nil {
					continue
				}
				a := in.Allocation.DeviceBinding
				if a.Validate() != nil || a.HostID != m.Target.HostID || a.Digest != b.Digest || a.DeviceID != b.UUID || a.DomainID != b.Pool || a.HostID != b.Host || a.Selector != b.Selector {
					continue
				}
				pool = b.Pool
			}
		}
		for _, fact := range observation.Domains {
			if fact.ID != pool {
				continue
			}
			if fact.Kind == "unknown" {
				continue
			}
			if d.Kind == "device" && fact.Kind != "dedicated" || d.Kind == "host" && fact.Kind != "system" && fact.Kind != "shared" {
				continue
			}
			if fact.Kind == "dedicated" && assumed == nil && (m.Map == nil || m.Map.Binding == nil || !slices.Contains(fact.DeviceIDs, m.Map.Binding.UUID)) {
				continue
			}
			old := d.ID
			d.ID = pool
			if fact.Kind == "shared" {
				d.Kind = "shared"
				out.HostBytes, out.DeviceBytes = "", ""
			}
			if fact.Kind == "dedicated" {
				if assumed != nil {
					d.DeviceID = assumed.UUID
				} else {
					d.DeviceID = m.Map.Binding.UUID
				}
			}
			for j := range out.Components {
				if out.Components[j].DomainID == old {
					out.Components[j].DomainID = pool
				}
			}
			d.Source, d.SampleID = "target-node-info", m.Target.ObservationID+":"+observation.RunID
			d.CapacityBytes, d.FreeBytes = "", ""
			remaining := noderec.MemoryFreshMS
			if fact.Total.Valid {
				d.CapacityBytes = fact.Total.Bytes
				d.CapacitySource = fact.Total.Source
				d.CapacityTimestamp = fact.Total.SampledAt.Format(time.RFC3339Nano)
				remaining = min(remaining, noderec.MemoryFreshMS-fact.Total.AgeMS)
			} else {
				remaining = 0
			}
			if fact.Free.Valid {
				d.FreeBytes = fact.Free.Bytes
				d.FreeSource = fact.Free.Source
				d.FreeTimestamp = fact.Free.SampledAt.Format(time.RFC3339Nano)
				remaining = min(remaining, noderec.MemoryFreshMS-fact.Free.AgeMS)
			}
			d.RemainingValidityMS = max(0, remaining)
		}
	}
	*out = modelconfig.CompareEstimate(*out)
}

func modelEstimateDomainsLegacy(out *modelconfig.Estimate, in modelEstimateInput, now time.Time) {
	m := in.Memory
	if m == nil {
		out.Unknowns = append(out.Unknowns, "target memory telemetry unavailable")
		return
	}
	if !modelEstimateMemoryFresh(m, now) || len(m.Domains) == 0 || len(m.Domains) > 64 {
		out.Unknowns = append(out.Unknowns, "memory snapshot stale, missing source/identity or outside domain limit")
		return
	}
	check := modelconfig.Estimate{Status: "unknown", Domains: m.Domains}
	if err := check.Validate(); err != nil {
		out.Unknowns = append(out.Unknowns, "invalid memory domains")
		return
	}
	for _, d := range m.Domains {
		for _, v := range []string{d.CapacityBytes, d.FreeBytes} {
			if v != "" {
				if _, err := strconv.ParseUint(v, 10, 64); err != nil {
					out.Unknowns = append(out.Unknowns, "memory bytes exceed uint64 range")
					return
				}
			}
		}
		if d.CapacityBytes != "" && d.FreeBytes != "" {
			capacity, _ := strconv.ParseUint(d.CapacityBytes, 10, 64)
			free, _ := strconv.ParseUint(d.FreeBytes, 10, 64)
			if free > capacity {
				out.Unknowns = append(out.Unknowns, "free memory exceeds capacity")
				return
			}
		}
	}
	if m.UMA {
		if len(m.Domains) != 1 || m.Domains[0].Kind != "shared" {
			out.Unknowns = append(out.Unknowns, "UMA requires exactly one shared pool")
			return
		}
		out.MemorySource, out.MemoryTimestamp = m.Source, m.At.UTC().Format(time.RFC3339Nano)
		d := m.Domains[0]
		d.RequiredBytes = ""
		if out.HostBytes != "" && out.DeviceBytes != "" {
			h, _ := strconv.ParseUint(out.HostBytes, 10, 64)
			v, _ := strconv.ParseUint(out.DeviceBytes, 10, 64)
			d.RequiredBytes = strconv.FormatUint(modelEstimateAdd(h, v), 10)
		}
		for i := range out.Components {
			if out.Components[i].DomainID != "" {
				out.Components[i].DomainID = d.ID
			}
		}
		out.Domains = []modelconfig.MemoryDomain{d}
		out.HostBytes, out.DeviceBytes = "", ""
		out.Assumptions = append(out.Assumptions, "UMA is one shared pool; host and GPU capacity are never added.")
		return
	}
	var hosts int
	matchedDevices := 0
	for _, d := range m.Domains {
		if d.Kind == "shared" {
			out.Unknowns = append(out.Unknowns, "shared memory requires explicit UMA pool semantics")
			return
		}
		if d.Kind == "host" {
			hosts++
		}
		if d.Kind == "device" && m.NativeDeviceID != "" && d.DeviceID == m.NativeDeviceID {
			matchedDevices++
		}
	}
	if hosts > 1 || matchedDevices > 1 {
		out.Unknowns = append(out.Unknowns, "ambiguous host/native device memory mapping")
		return
	}
	out.MemorySource, out.MemoryTimestamp = m.Source, m.At.UTC().Format(time.RFC3339Nano)
	for i := range out.Domains {
		target := &out.Domains[i]
		for _, d := range m.Domains {
			if target.Kind == "host" && d.Kind == "host" || target.ID == "offload-unmapped" && d.Kind == "device" && m.NativeDeviceID != "" && d.DeviceID == m.NativeDeviceID {
				old, required := target.ID, target.RequiredBytes
				*target = d
				target.RequiredBytes = required
				for j := range out.Components {
					if out.Components[j].DomainID == old {
						out.Components[j].DomainID = d.ID
					}
				}
				break
			}
		}
	}
}
