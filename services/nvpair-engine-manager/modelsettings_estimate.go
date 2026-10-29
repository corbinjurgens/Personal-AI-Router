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
	ordered := make([]modelconfig.ArtifactSummary, 0, len(bindings))
	for _, binding := range bindings {
		data, err := json.Marshal(binding)
		if err != nil {
			return nil
		}
		ordered = append(ordered, modelconfig.ArtifactSummary{Role: binding.Role, Identity: modelLoadDigest(data), Size: binding.Size})
	}
	return ordered
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
	out.EstimateSource = &modelconfig.EstimateSource{Kind: "pair-estimate", Provider: "PAIR", Revision: modelEstimatorRevision, Basis: "bounded GGUF facts and pinned graph estimates; uncalibrated"}
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
		"Advisory and uncalibrated; tensor storage and pinned library graph costs are not measured native allocations.",
		"Selected model scenario only; simultaneous residents are unsized and no reclaimed memory is credited.",
		"Known totals are partial component sums, not a complete peak or a Fits decision.",
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
	if in.Allocation != nil && in.Allocation.Options.CacheRAM != nil && *in.Allocation.Options.CacheRAM > 0 {
		c := estimateComponent("prompt-cache", modelEstimateMul(uint64(*in.Allocation.Options.CacheRAM), 1<<20), "configured host prompt-cache ceiling; usage is demand-dependent")
		c.LowBytes = "0"
		out.Components = append(out.Components, c)
	} else {
		out.Components = append(out.Components, unknownEstimateComponent("prompt-cache", "effective native cache-ram unavailable"))
	}
	out.Unknowns = append(out.Unknowns, "driver/runtime, allocator alignment/fragmentation, mmap residency and simultaneous resident model costs unsized")
	modelEstimatePlacement(&out, m, in)
	for _, c := range out.Components {
		if c.Status == "unknown" || c.Status == "partial" {
			out.Unknowns = append(out.Unknowns, c.Name+": "+c.Reason)
		}
	}
	return out
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
	u := unknownEstimateComponent("recurrent-state", "recurrent dimensions or per-layer native mapping unavailable")
	if m.arch != "lfm2" && m.arch != "lfm2moe" {
		if !slices.Contains([]string{"llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "mistral", "gemma2", "gemma3", "deepseek2", "deepseek3"}, m.arch) {
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
	for _, bits := range mask.Array {
		v, ok := modelEstimateUint(gguffacts.Numeric{Type: mask.Type, Bits: bits})
		if !ok {
			return u
		}
		if v == 0 {
			count++
		}
	}
	if count == 0 {
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

// Explicit support matrix, using native hparams contracts referenced by the
// pinned MIT library. Family names never substitute for recurrent/MLA mappings.
func modelEstimateCaches(ctx context.Context, m modelEstimateModel, in modelEstimateInput) (attention, state modelconfig.EstimateComponent) {
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
	hybrid := slices.Contains([]string{"lfm2", "lfm2moe", "qwen3next", "qwen35", "qwen35moe"}, m.arch)
	dense := slices.Contains([]string{"llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "mistral", "gemma2", "gemma3"}, m.arch)
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
	for _, field := range []string{"attention.kv_lora_rank", "attention.key_length_mla", "attention.value_length_mla", "attention.shared_kv_layers", "nextn_predict_layers"} {
		if n, exists := m.numeric[m.arch+"."+field]; exists {
			v, valid := modelEstimateUint(n)
			if !valid || v != 0 {
				kv.Reason = "compressed/shared/NextN cache mapping unavailable"
				return kv, recurrent
			}
		}
	}
	window, hasWindow := m.number("attention.sliding_window")
	if _, exists := m.numeric[m.arch+".attention.sliding_window"]; exists && !hasWindow {
		kv.Reason = "malformed SWA window"
		return kv, recurrent
	}
	_, hasPattern := m.numeric[m.arch+".attention.sliding_window_pattern"]
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
		if _, exists := m.numeric[m.arch+".attention.sliding_window_pattern"]; !exists && m.arch != "lfm2" && m.arch != "lfm2moe" {
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
			period, ok := modelEstimateUint(pattern)
			if !ok || period > layers {
				kv.Reason = "SWA declared pattern malformed or unmapped"
				return kv, recurrent
			}
		}
	}
	arch := m.file.Architecture()
	var cacheBytes uint64
	for i := uint64(0); i < layers; i++ {
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
		if window > 0 && i < uint64(len(arch.AttentionSlidingWindowLayers)) && arch.AttentionSlidingWindowLayers[i] {
			perStream := uint64(a.PerSlotContext)
			if !*in.SWAFull {
				perStream = gguf.GGMLPadding(min(perStream, modelEstimateAdd(window, uint64(*a.Options.UBatchSize))), 256)
			}
			cells = modelEstimateMul(perStream, uint64(a.Slots))
			k, v = swaKey, swaValue
		}
		kb, validK := modelEstimateRow(kt, modelEstimateMul(k, h), cells)
		vb, validV := modelEstimateRow(vt, modelEstimateMul(v, h), cells)
		if !validK || !validV {
			kv.Reason = "K/V row width incompatible with cache block size"
			return kv, recurrent
		}
		cacheBytes = modelEstimateAdd(cacheBytes, modelEstimateAdd(kb, vb))
	}
	kv = estimateComponent("attention-cache", cacheBytes, "per-layer native head widths; independently block-sized K/V; total context already includes slots")
	return kv, recurrent
}

// Only an ordinary dense CPU graph is admitted to the full library heuristic.
// Missing dimensions do not trigger heuristic defaults; complex graphs retain
// their independent weight/cache results and an Unknown workspace.
func modelEstimateCompute(ctx context.Context, m modelEstimateModel, in modelEstimateInput) (out modelconfig.EstimateComponent) {
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
	if in.Allocation == nil || in.Support != modelEstimatorSupport || in.BuildID == "" || !slices.Contains([]string{"llama", "qwen2", "qwen3"}, m.arch) {
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
	for _, field := range []string{"block_count", "embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv", "vocab_size"} {
		v, ok := m.number(field)
		if !ok || v == 0 || v > 65536 || field == "block_count" && v > modelEstimateLayers {
			u.Reason = "dense graph required scalar dimension missing/outside range: " + field
			return u
		}
	}
	for _, field := range []string{"expert_count", "attention.sliding_window", "attention.kv_lora_rank", "attention.key_length_mla", "attention.value_length_mla", "full_attention_interval", "attention.shared_kv_layers", "nextn_predict_layers"} {
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
	if emb%heads != 0 || hkv > heads {
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
		if strings.Contains(key, ".ssm.") || strings.Contains(key, ".shortconv.") || strings.Contains(key, ".wkv.") {
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
	if !shape("token_embd.weight", emb, vocab) || !shape("output_norm.weight", emb) {
		u.Reason = "dense graph embedding/output shapes incomplete"
		return u
	}
	for _, field := range []string{"attention.key_length", "attention.value_length"} {
		if _, exists := m.numeric[m.arch+"."+field]; exists {
			width, _ := m.number(field)
			if width != emb/heads {
				u.Reason = "dense graph unequal head widths unmapped"
				return u
			}
		}
	}
	for i := uint64(0); i < layers; i++ {
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
	if a.GPU.Mode != "cpu" {
		u.Reason = "GPU graph/device mapping unavailable; CPU workspace cannot price offloaded graph"
		return u
	}
	if a.Options.OpOffload == nil || *a.Options.OpOffload {
		u.Reason = "CPU graph requires explicit op-offload=false; native offload mapping unavailable"
		return u
	}
	opts := []gguf.GGUFRunEstimateOption{gguf.WithLLaMACppContextSize(a.TotalContext), gguf.WithParallelSize(a.Slots), gguf.WithLLaMACppLogicalBatchSize(batch), gguf.WithLLaMACppPhysicalBatchSize(ubatch), gguf.WithLLaMACppCacheKeyType(kt), gguf.WithLLaMACppCacheValueType(vt), gguf.WithLLaMACppOffloadLayers(0)}
	if *a.Options.FlashAttention == "on" {
		opts = append(opts, gguf.WithFlashAttention())
	}
	result := m.file.EstimateLLaMACppRun(opts...)
	if len(result.Devices) > 64 {
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
	total := modelEstimateAdd(modelEstimateAdd(uint64(c.Footprint), uint64(c.Input)), modelEstimateAdd(uint64(c.Compute), uint64(c.Output)))
	if total == 0 {
		return u
	}
	component := estimateComponent("compute-workspace", total, "pinned library dense CPU graph portion only; complete CPU scratch/op-offload mapping unavailable; no hypothetical GPU buffer charged as host")
	component.Status = "partial"
	component.Basis = "heuristic"
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
		if cpu || c.Name == "prompt-cache" {
			c.DomainID = "host"
			host = modelEstimateAdd(host, n)
			continue
		}
		if c.Name == "weights" {
			// Pinned native output-first layer convention. Tensor placement is a
			// projection; embeddings and unclassified tensors remain on host.
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
			} else if a.Options.KVOffload != nil && uint64(*a.GPU.Layers) == layers+1 {
				c.DomainID = "offload-unmapped"
				device = modelEstimateAdd(device, n)
			} else {
				out.Unknowns = append(out.Unknowns, c.Name+": per-layer device placement unresolved; no layer-fraction approximation used")
			}
		}
	}
	out.HostBytes, out.DeviceBytes = strconv.FormatUint(host, 10), strconv.FormatUint(device, 10)
	out.Domains[0].RequiredBytes = out.HostBytes
	out.Domains[1].RequiredBytes = out.DeviceBytes
	for i := range out.Domains {
		d := &out.Domains[i]
		var low, high uint64
		for _, c := range out.Components {
			if c.DomainID == d.ID && c.Bytes != "" {
				l, _ := strconv.ParseUint(c.LowBytes, 10, 64)
				h, _ := strconv.ParseUint(c.HighBytes, 10, 64)
				low, high = modelEstimateAdd(low, l), modelEstimateAdd(high, h)
			}
		}
		d.RequiredLowBytes, d.RequiredHighBytes = strconv.FormatUint(low, 10), strconv.FormatUint(high, 10)
	}
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
			if fact.Kind == "dedicated" && (m.Map == nil || m.Map.Binding == nil || !slices.Contains(fact.DeviceIDs, m.Map.Binding.UUID)) {
				continue
			}
			old := d.ID
			d.ID = pool
			if fact.Kind == "shared" {
				d.Kind = "shared"
				out.HostBytes, out.DeviceBytes = "", ""
			}
			if fact.Kind == "dedicated" && m.Map.Binding != nil {
				d.DeviceID = m.Map.Binding.UUID
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
