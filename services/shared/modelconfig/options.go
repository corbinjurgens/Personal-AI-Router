// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package modelconfig contains portable model configuration contracts.
package modelconfig

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

// OptionDescriptor describes a service-owned option, without native flag names.
type OptionDescriptor struct {
	Field          string          `json:"field"`
	Label          string          `json:"label"`
	Kind           string          `json:"kind"`
	Minimum        *json.Number    `json:"minimum,omitempty"`
	Maximum        *json.Number    `json:"maximum,omitempty"`
	Values         []string        `json:"values,omitempty"`
	Default        json.RawMessage `json:"default,omitempty"`
	DefaultSource  string          `json:"defaultSource,omitempty"`
	Supported      bool            `json:"supported"`
	Reason         string          `json:"reason,omitempty"`
	Portable       bool            `json:"portable"`
	Owner          string          `json:"owner,omitempty"`
	Phase          string          `json:"phase,omitempty"`
	IdentityEffect string          `json:"identityEffect,omitempty"`
	EstimateEffect string          `json:"estimateEffect,omitempty"`
}

func (o OptionDescriptor) Clone() OptionDescriptor {
	o.Minimum, o.Maximum = clonePointer(o.Minimum), clonePointer(o.Maximum)
	o.Values = slices.Clone(o.Values)
	o.Default = slices.Clone(o.Default)
	return o
}

// Validate checks descriptor transport shape, not native support. The manager
// builds these from its existing option table and build/architecture facts.
func (o OptionDescriptor) Validate() error {
	if boundedText(o.Field, 128, true) != nil || boundedText(o.Label, 256, true) != nil || !slices.Contains([]string{"integer", "number", "boolean", "enum", "string"}, o.Kind) {
		return fmt.Errorf("option: invalid descriptor identity or kind")
	}
	if len(o.Values) > 64 || !o.Supported && o.Reason == "" {
		return fmt.Errorf("option: bounded values and unsupported reason required")
	}
	for _, value := range o.Values {
		if err := boundedText(value, 256, true); err != nil {
			return fmt.Errorf("option.values: %w", err)
		}
	}
	var bounds [2]*big.Rat
	for i, value := range []*json.Number{o.Minimum, o.Maximum} {
		if value == nil {
			continue
		}
		n, _, err := boundedNumber(*value)
		if err != nil {
			return fmt.Errorf("option: invalid numeric bound: %w", err)
		}
		bounds[i] = n
	}
	if bounds[0] != nil && bounds[1] != nil && bounds[0].Cmp(bounds[1]) > 0 {
		return fmt.Errorf("option: inverted bounds")
	}
	if o.Default != nil {
		var raw json.RawMessage
		if err := DecodeStrict(o.Default, &raw); err != nil {
			return fmt.Errorf("option.default: %w", err)
		}
		if o.DefaultSource == "" {
			return fmt.Errorf("option.defaultSource: required for a projected default")
		}
	}
	return CheckPayloadSize(o)
}

// ValidateRequested checks portable constraints only. Build, artifact and
// architecture-dependent checks belong to the target's preview owner.
func (r RequestedLoad) Validate() error {
	if r.Context != nil {
		c := r.Context
		switch c.Mode {
		case "fixed":
			if c.Tokens <= 0 || c.Floor != 0 || c.Cap != 0 {
				return fmt.Errorf("context: fixed requires positive tokens only")
			}
		case "auto":
			if c.Tokens != 0 || c.Floor <= 0 || c.Cap < c.Floor {
				return fmt.Errorf("context: auto requires a positive floor and cap")
			}
		default:
			return fmt.Errorf("context.mode: invalid value")
		}
	}
	if r.Slots != nil && *r.Slots <= 0 {
		return fmt.Errorf("slots: must be positive")
	}
	if r.GPU != nil {
		switch r.GPU.Mode {
		case "native-default", "cpu", "auto":
			if r.GPU.Layers != nil {
				return fmt.Errorf("gpu.layers: only valid for layers")
			}
		case "layers":
			if r.GPU.Layers == nil || *r.GPU.Layers < 0 {
				return fmt.Errorf("gpu.layers: must be nonnegative")
			}
		default:
			return fmt.Errorf("gpu.mode: invalid value")
		}
	}
	for name, value := range map[string]*int32{"batchSize": r.BatchSize, "ubatchSize": r.UBatchSize, "threads": r.Threads, "threadsBatch": r.ThreadsBatch, "cacheRAM": r.CacheRAM} {
		if value != nil && *value <= 0 {
			return fmt.Errorf("%s: must be positive", name)
		}
	}
	if r.BatchSize != nil && r.UBatchSize != nil && *r.UBatchSize > *r.BatchSize {
		return fmt.Errorf("ubatchSize: exceeds batchSize")
	}
	if r.SleepIdleSeconds != nil && *r.SleepIdleSeconds != -1 && *r.SleepIdleSeconds <= 0 {
		return fmt.Errorf("sleepIdleSeconds: must be -1 or positive")
	}
	for name, value := range map[string]*string{"cacheTypeK": r.CacheTypeK, "cacheTypeV": r.CacheTypeV} {
		if value != nil && *value != "f16" && *value != "bf16" && *value != "q8_0" && *value != "q4_0" {
			return fmt.Errorf("%s: invalid value", name)
		}
	}
	if r.FlashAttention != nil && *r.FlashAttention != "inherit" && *r.FlashAttention != "on" && *r.FlashAttention != "off" {
		return fmt.Errorf("flashAttention: invalid value")
	}
	if r.Context != nil && r.Context.Mode == "fixed" {
		slots := int32(1) // arithmetic check only, not a saved/native default
		if r.Slots != nil {
			slots = *r.Slots
		}
		if _, _, err := TotalContext(r.Context.Tokens, slots); err != nil {
			return err
		}
	}
	return nil
}

func (r RequestedLoad) selectedFields() map[string]bool {
	return map[string]bool{
		"context": r.Context != nil, "slots": r.Slots != nil, "gpu": r.GPU != nil,
		"cacheTypeK": r.CacheTypeK != nil, "cacheTypeV": r.CacheTypeV != nil,
		"flashAttention": r.FlashAttention != nil, "batchSize": r.BatchSize != nil,
		"ubatchSize": r.UBatchSize != nil, "threads": r.Threads != nil,
		"threadsBatch": r.ThreadsBatch != nil, "kvOffload": r.KVOffload != nil,
		"opOffload": r.OpOffload != nil, "sleepIdleSeconds": r.SleepIdleSeconds != nil,
		"cacheRAM": r.CacheRAM != nil,
	}
}

func (r RequestedLoad) Clone() RequestedLoad {
	r.Context = clonePointer(r.Context)
	r.Slots = clonePointer(r.Slots)
	r.GPU = clonePointer(r.GPU)
	if r.GPU != nil {
		r.GPU.Layers = clonePointer(r.GPU.Layers)
	}
	r.CacheTypeK = clonePointer(r.CacheTypeK)
	r.CacheTypeV = clonePointer(r.CacheTypeV)
	r.FlashAttention = clonePointer(r.FlashAttention)
	r.BatchSize = clonePointer(r.BatchSize)
	r.UBatchSize = clonePointer(r.UBatchSize)
	r.Threads = clonePointer(r.Threads)
	r.ThreadsBatch = clonePointer(r.ThreadsBatch)
	r.KVOffload = clonePointer(r.KVOffload)
	r.OpOffload = clonePointer(r.OpOffload)
	r.SleepIdleSeconds = clonePointer(r.SleepIdleSeconds)
	r.CacheRAM = clonePointer(r.CacheRAM)
	return r
}

// validateGeneration mirrors MODEL_OPTION_FIELDS, rather than imposing new
// request-dialect semantics. Exact rational comparisons prevent a decimal just
// outside a bound from rounding into it. Lexeme/exponent bounds keep this work
// bounded and the original json.Number is retained without float conversion.
func validateGeneration(key string, value *json.Number) error {
	switch key {
	case "temperature", "top_p", "top_k", "min_p", "repeat_penalty", "frequency_penalty", "presence_penalty", "seed":
	default:
		return fmt.Errorf("generation: unsupported field %q", key)
	}
	if value == nil { // explicit clear
		return nil
	}
	n, f, err := boundedNumber(*value)
	if err != nil {
		return fmt.Errorf("generation.%s: %w", key, err)
	}
	zero, one, two := big.NewRat(0, 1), big.NewRat(1, 1), big.NewRat(2, 1)
	valid := false
	switch key {
	case "temperature":
		valid = n.Cmp(zero) >= 0
	case "top_p", "min_p":
		valid = n.Cmp(zero) >= 0 && n.Cmp(one) <= 0
	case "repeat_penalty":
		valid = n.Cmp(zero) > 0 && f > 0
	case "frequency_penalty", "presence_penalty":
		valid = n.Cmp(big.NewRat(-2, 1)) >= 0 && n.Cmp(two) <= 0
	case "top_k", "seed":
		limit := big.NewRat(1<<53-1, 1)
		valid = n.IsInt() && n.Cmp(limit) <= 0 && n.Cmp(new(big.Rat).Neg(limit)) >= 0
		if key == "top_k" {
			valid = valid && n.Sign() >= 0
		}
	}
	if !valid {
		return fmt.Errorf("generation.%s: outside the existing field range", key)
	}
	return nil
}

func boundedNumber(value json.Number) (*big.Rat, float64, error) {
	s := value.String()
	if len(s) == 0 || len(s) > 128 || !json.Valid([]byte(s)) {
		return nil, 0, fmt.Errorf("invalid numeric lexeme")
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) {
		return nil, 0, fmt.Errorf("number must be finite")
	}
	if i := strings.IndexAny(s, "eE"); i >= 0 {
		exponent, err := strconv.ParseInt(s[i+1:], 10, 32)
		if err != nil || exponent < -324 || exponent > 308 {
			return nil, 0, fmt.Errorf("exponent exceeds numeric budget")
		}
	}
	n, ok := new(big.Rat).SetString(s)
	if !ok {
		return nil, 0, fmt.Errorf("invalid decimal")
	}
	return n, f, nil
}

func (a ArtifactBinding) Validate() error {
	if a.Role != "model" && a.Role != "draft" {
		return fmt.Errorf("artifact.role: unsupported consumed role")
	}
	if err := boundedText(a.Path, 4096, true); err != nil {
		return fmt.Errorf("artifact.path: %w", err)
	}
	if err := boundedText(a.ResolvedTarget, 4096, true); err != nil {
		return fmt.Errorf("artifact.resolvedTarget: %w", err)
	}
	if err := ValidateRevision(a.Size); err != nil || a.Size == "0" {
		return fmt.Errorf("artifact.size: expected positive int64 decimal string")
	}
	mtime, err := strconv.ParseInt(a.MtimeNS, 10, 64)
	if err != nil || strconv.FormatInt(mtime, 10) != a.MtimeNS {
		return fmt.Errorf("artifact.mtimeNs: expected canonical signed int64 decimal string")
	}
	if !digest(a.SampleSHA256) {
		return fmt.Errorf("artifact.sampleSha256: expected SHA256 hex")
	}
	return nil
}

func (r ResolvedLoad) Validate() error {
	per, total, err := TotalContext(r.PerSlotContext, r.Slots)
	unified := r.KVUnified != nil && *r.KVUnified
	if err != nil || per != r.PerSlotContext || r.TotalContext <= 0 || r.TotalContext%256 != 0 || (!unified && total != r.TotalContext) || (unified && total > r.TotalContext) {
		return fmt.Errorf("resolved.context: expected padded per-slot and checked total allocation")
	}
	if r.GPU.Mode != "cpu" && r.GPU.Mode != "layers" {
		return fmt.Errorf("resolved.gpu: finite placement required")
	}
	if err := (RequestedLoad{GPU: &r.GPU}).Validate(); err != nil {
		return err
	}
	if err := r.Options.Validate(); err != nil {
		return fmt.Errorf("resolved.options: %w", err)
	}
	if r.Options.Context != nil && (r.Options.Context.Mode != "fixed" || r.Options.Context.Tokens != r.PerSlotContext) || r.Options.Slots != nil && *r.Options.Slots != r.Slots || r.Options.GPU != nil && !reflect.DeepEqual(*r.Options.GPU, r.GPU) {
		return fmt.Errorf("resolved.options: allocation must agree with finite resolution")
	}
	if len(r.Artifacts) == 0 || len(r.Artifacts) > MaxArtifacts || r.Artifacts[0].Role != "model" {
		return fmt.Errorf("resolved.artifacts: bounded ordered model shards required")
	}
	draft := false
	seen, sources := map[string]bool{}, map[string]bool{}
	for _, a := range r.Artifacts {
		if err := a.Validate(); err != nil {
			return err
		}
		if a.Role == "draft" {
			draft = true
		} else if draft {
			return fmt.Errorf("resolved.artifacts: model shards must precede draft shards")
		}
		if seen[a.ResolvedTarget] || sources[a.Path] {
			return fmt.Errorf("resolved.artifacts: duplicate source or resolved target")
		}
		seen[a.ResolvedTarget] = true
		sources[a.Path] = true
	}
	if err := boundedText(r.NativeBuildID, 256, true); err != nil {
		return fmt.Errorf("resolved.nativeBuildId: %w", err)
	}
	if !digest(r.SourceInputDigest) || !digest(r.PresetDigest) {
		return fmt.Errorf("resolved: source input and preset SHA256 digests required")
	}
	if err := boundedText(r.ResolverRevision, 256, false); err != nil {
		return fmt.Errorf("resolved.resolverRevision: %w", err)
	}
	if err := boundedText(r.ReviewedInputID, 256, false); err != nil {
		return fmt.Errorf("resolved.reviewedInputId: %w", err)
	}
	if (r.ResolverRevision == "") != (r.ReviewedInputID == "") {
		return fmt.Errorf("resolved: resolver revision and reviewed input identity must occur together")
	}
	return CheckPayloadSize(r)
}

func (m ModelLoad) Validate() error {
	if err := validateModelID(m.ModelID); err != nil {
		return err
	}
	if err := m.Requested.Validate(); err != nil {
		return err
	}
	if m.Resolved == nil {
		return nil // saved intent is distinguishable from an accepted resolution
	}
	r := m.Resolved
	if err := r.Validate(); err != nil {
		return err
	}
	if m.Requested.Slots != nil && *m.Requested.Slots != r.Slots {
		return fmt.Errorf("resolved.slots: differs from requested slots")
	}
	if c := m.Requested.Context; c != nil {
		if c.Mode == "fixed" {
			per, _, err := TotalContext(c.Tokens, r.Slots)
			if err != nil || per != r.PerSlotContext {
				return fmt.Errorf("resolved.context: differs from requested fixed context")
			}
		} else if r.PerSlotContext < c.Floor || r.PerSlotContext > c.Cap {
			return fmt.Errorf("resolved.context: outside requested Auto range")
		}
	}
	if g := m.Requested.GPU; g != nil && g.Mode != "auto" && g.Mode != "native-default" && !reflect.DeepEqual(*g, r.GPU) {
		return fmt.Errorf("resolved.gpu: differs from requested placement")
	}
	if m.Requested.Context != nil && m.Requested.Context.Mode == "auto" || m.Requested.GPU != nil && m.Requested.GPU.Mode == "auto" {
		if r.ResolverRevision == "" || r.ReviewedInputID == "" {
			return fmt.Errorf("resolved: Auto requires resolver revision and reviewed input identity")
		}
	}
	// Effective explicit options must preserve requested quality/retention fields.
	want, got := m.Requested.Clone(), r.Options.Clone()
	want.Context, want.Slots, want.GPU = nil, nil, nil
	got.Context, got.Slots, got.GPU = nil, nil, nil
	// Resolutions may materialize effective native choices for inherited fields;
	// each explicitly requested leaf must still match exactly.
	wv, gv := reflect.ValueOf(want), reflect.ValueOf(got)
	for i := 0; i < wv.NumField(); i++ {
		if !wv.Field(i).IsNil() && !reflect.DeepEqual(wv.Field(i).Interface(), gv.Field(i).Interface()) {
			return fmt.Errorf("resolved.options: changed an explicit requested option")
		}
	}
	return nil
}

// TotalContext applies the native 256-token padding and checks int32 overflow.
func TotalContext(tokens, slots int32) (int32, int32, error) {
	if tokens <= 0 || slots <= 0 {
		return 0, 0, fmt.Errorf("context and slots must be positive")
	}
	padded := (int64(tokens) + 255) / 256 * 256
	total := padded * int64(slots)
	if total > math.MaxInt32 {
		return 0, 0, fmt.Errorf("total context exceeds int32")
	}
	return int32(padded), int32(total), nil
}
