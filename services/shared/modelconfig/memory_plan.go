// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
)

// EstimateSource identifies the owner of an advisory estimate and its omissions.
type EstimateSource struct {
	Kind          string   `json:"kind"`
	Provider      string   `json:"provider"`
	Revision      string   `json:"revision"`
	Basis         string   `json:"basis"`
	UnpricedCosts []string `json:"unpricedCosts,omitempty"`
}

// MemoryPlan carries accounting views, not launch settings or measured usage.
// Empty byte strings are unknown; "0" is an explicitly known zero.
type MemoryPlan struct {
	Identity              MemoryPlanIdentity   `json:"identity"`
	Requested             MemoryPlanConfig     `json:"requested"`
	Resolved              *MemoryPlanConfig    `json:"resolved,omitempty"`
	ConfigurationComplete bool                 `json:"configurationComplete"`
	PricingScope          string               `json:"pricingScope"`
	PricingQualification  string               `json:"pricingQualification"`
	Devices               []MemoryPlanResource `json:"devices"`
	HostRAM               MemoryPlanResource   `json:"hostRAM"`
	HostCommit            MemoryPlanResource   `json:"hostCommit"`
	Verdict               MemoryPlanVerdict    `json:"verdict"`
	Warnings              []string             `json:"warnings,omitempty"`
}

type MemoryPlanIdentity struct {
	Engine                 string          `json:"engine"`
	Provider               string          `json:"provider"`
	Revision               string          `json:"revision"`
	ExecutableSHA256       string          `json:"executableSHA256"`
	Build                  MemoryPlanBuild `json:"build"`
	RuntimeSourceSHA256    string          `json:"runtimeSourceSHA256"`
	ModelID                string          `json:"modelId"`
	ContentSHA256          string          `json:"contentSHA256,omitempty"`
	ContentAssurance       string          `json:"contentAssurance"`
	AssertedContentSHA256  string          `json:"assertedContentSHA256,omitempty"`
	ArtifactIdentitySHA256 string          `json:"artifactIdentitySHA256"`
	Cacheable              bool            `json:"cacheable"`
	CacheableReason        string          `json:"cacheableReason,omitempty"`
}

// Build is a safe summary; private build records stay with the adapter.
type MemoryPlanBuild struct {
	Version   string `json:"version,omitempty"`
	Revision  string `json:"revision,omitempty"`
	Dirty     *bool  `json:"dirty,omitempty"`
	Timestamp string `json:"timestamp,omitempty"`
	Basis     string `json:"basis"`
}

type MemoryPlanConfig struct {
	ContextTokens int32                 `json:"contextTokens"`
	KVFormat      string                `json:"kvFormat,omitempty"`
	BatchSize     *int32                `json:"batchSize,omitempty"`
	Microbatch    *MemoryPlanMicrobatch `json:"microbatch,omitempty"`
	ParallelSlots *int32                `json:"parallelSlots,omitempty"`
	Devices       []MemoryPlanSelector  `json:"devices"`
	Placement     string                `json:"placement"`
}

type MemoryPlanMicrobatch struct {
	Mode   string `json:"mode"`
	Tokens *int32 `json:"tokens,omitempty"`
	Limit  *int32 `json:"limit,omitempty"`
}

type MemoryPlanSelector struct {
	Backend  string `json:"backend"`
	Selector string `json:"selector"`
}

type MemoryPlanResource struct {
	Kind                          string                         `json:"kind"`
	NativeSelector                string                         `json:"nativeSelector,omitempty"`
	PhysicalID                    string                         `json:"physicalId,omitempty"`
	PoolID                        string                         `json:"poolId,omitempty"`
	Mapping                       string                         `json:"mapping"`
	MappingBasis                  string                         `json:"mappingBasis"`
	Groups                        []MemoryPlanGroup              `json:"groups"`
	CombinedAllocations           []MemoryPlanCombinedAllocation `json:"combinedAllocations,omitempty"`
	AllocatedBytes                string                         `json:"allocatedBytes,omitempty"`
	MappedViewBytes               string                         `json:"mappedViewBytes,omitempty"`
	PricedTotalBytes              string                         `json:"pricedTotalBytes"`
	PricedTotalBasis              string                         `json:"pricedTotalBasis"`
	MinimumSelectionCapacityBytes string                         `json:"minimumSelectionCapacityBytes,omitempty"`
	MinimumSelectionCapacityBasis string                         `json:"minimumSelectionCapacityBasis,omitempty"`
	Assumptions                   []MemoryPlanAmount             `json:"assumptions,omitempty"`
	RequiredWithAllowancesBytes   string                         `json:"requiredWithAllowancesBytes"`
	BudgetBytes                   string                         `json:"budgetBytes,omitempty"`
	ShortfallBytes                string                         `json:"shortfallBytes,omitempty"`
	ShortfallBasis                string                         `json:"shortfallBasis,omitempty"`
	CallerAllowances              []MemoryPlanAmount             `json:"callerAllowances,omitempty"`
	UncoveredCosts                []string                       `json:"uncoveredCosts,omitempty"`
}

type MemoryPlanGroup struct {
	Name               string   `json:"name"`
	Bytes              string   `json:"bytes,omitempty"`
	Quality            string   `json:"quality"`
	Basis              string   `json:"basis"`
	Additive           bool     `json:"additive"`
	CombinedMembership []string `json:"combinedMembership,omitempty"`
}

type MemoryPlanCombinedAllocation struct {
	Members []string `json:"members"`
	Bytes   string   `json:"bytes,omitempty"`
	Quality string   `json:"quality"`
	Basis   string   `json:"basis"`
}

type MemoryPlanAmount struct {
	Name       string `json:"name"`
	Bytes      string `json:"bytes"`
	Basis      string `json:"basis,omitempty"`
	Provenance string `json:"provenance,omitempty"`
}

type MemoryPlanVerdict struct {
	Status      string `json:"status"`
	Conditional bool   `json:"conditional"`
	BudgetBasis string `json:"budgetBasis,omitempty"`
	Basis       string `json:"basis"`
}

var memoryPlanGroupNames = []string{"weights", "kv_cache", "compute_scratch", "output_buffers", "runtime_driver_overhead", "transient_peak"}

// Presence checks keep required false values distinct from omitted fields.
// DecodeStrict still owns duplicate, shape, null and envelope validation.
func decodeMemoryPlanRequired(data []byte, out any, fields ...string) error {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(data, &object); err != nil {
		return err
	}
	for _, field := range fields {
		value, ok := object[field]
		if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("memory plan: required field %s", field)
		}
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	return d.Decode(out)
}

func (p *MemoryPlan) UnmarshalJSON(data []byte) error {
	type plain MemoryPlan
	return decodeMemoryPlanRequired(data, (*plain)(p), "configurationComplete")
}
func (i *MemoryPlanIdentity) UnmarshalJSON(data []byte) error {
	type plain MemoryPlanIdentity
	return decodeMemoryPlanRequired(data, (*plain)(i), "cacheable")
}
func (g *MemoryPlanGroup) UnmarshalJSON(data []byte) error {
	type plain MemoryPlanGroup
	return decodeMemoryPlanRequired(data, (*plain)(g), "additive")
}
func (v *MemoryPlanVerdict) UnmarshalJSON(data []byte) error {
	type plain MemoryPlanVerdict
	return decodeMemoryPlanRequired(data, (*plain)(v), "conditional")
}

func memoryPlanTexts(values []string, limit int) error {
	if len(values) > 256 {
		return fmt.Errorf("memory plan: collection exceeds budget")
	}
	seen := map[string]bool{}
	for _, value := range values {
		if boundedText(value, limit, true) != nil || seen[value] {
			return fmt.Errorf("memory plan: invalid or duplicate text")
		}
		seen[value] = true
	}
	return nil
}

func (c MemoryPlanConfig) validate(resolved bool) error {
	if c.ContextTokens <= 0 || !slices.Contains([]string{"cpu", "device", "hybrid"}, c.Placement) || len(c.Devices) > 64 {
		return fmt.Errorf("memory plan config: invalid context or placement")
	}
	if c.KVFormat != "" && !slices.Contains([]string{"fp16", "bf16", "int8", "q4_0", "k8v4", "f16", "q8_0"}, c.KVFormat) {
		return fmt.Errorf("memory plan config: invalid KV format")
	}
	if resolved && (c.KVFormat == "" || c.BatchSize == nil || c.ParallelSlots == nil || c.Microbatch == nil) {
		return fmt.Errorf("memory plan config: resolved values required")
	}
	if c.BatchSize != nil && *c.BatchSize <= 0 || c.ParallelSlots != nil && *c.ParallelSlots <= 0 {
		return fmt.Errorf("memory plan config: invalid batch or parallelism")
	}
	if c.Microbatch != nil {
		m := c.Microbatch
		switch m.Mode {
		case "fixed":
			if m.Tokens == nil || *m.Tokens <= 0 || *m.Tokens > c.ContextTokens || m.Limit != nil {
				return fmt.Errorf("memory plan config: invalid fixed microbatch")
			}
		case "auto":
			if resolved || m.Tokens != nil || m.Limit != nil && *m.Limit <= 0 {
				return fmt.Errorf("memory plan config: invalid automatic microbatch")
			}
		default:
			return fmt.Errorf("memory plan config: invalid microbatch mode")
		}
	}
	seen := map[string]bool{}
	for _, d := range c.Devices {
		if boundedText(d.Backend, 256, true) != nil || boundedText(d.Selector, 256, true) != nil || seen[d.Selector] {
			return fmt.Errorf("memory plan config: invalid or duplicate selector")
		}
		seen[d.Selector] = true
	}
	if c.Placement != "cpu" && len(c.Devices) == 0 || c.Placement == "cpu" && len(c.Devices) != 0 {
		return fmt.Errorf("memory plan config: selectors contradict placement")
	}
	return nil
}

func validateMemoryPlanQuantity(value, quality, basis string) error {
	if !slices.Contains([]string{"exact", "estimated", "unknown"}, quality) || boundedText(basis, 1024, true) != nil {
		return fmt.Errorf("memory plan: invalid quality or basis")
	}
	if quality == "unknown" {
		if value != "" {
			return fmt.Errorf("memory plan: unknown quantity has bytes")
		}
		return nil
	}
	return ValidateUnsignedDecimal(value)
}

func validateMemoryPlanMembers(members []string) error {
	if len(members) > len(memoryPlanGroupNames) {
		return fmt.Errorf("memory plan: too many group members")
	}
	seen := map[string]bool{}
	for _, name := range members {
		if !slices.Contains(memoryPlanGroupNames, name) || seen[name] {
			return fmt.Errorf("memory plan: invalid or duplicate group member")
		}
		seen[name] = true
	}
	return nil
}

func validateMemoryPlanAmounts(amounts []MemoryPlanAmount, allowance bool) (uint64, error) {
	if len(amounts) > 256 {
		return 0, fmt.Errorf("memory plan: too many amounts")
	}
	var total uint64
	seen := map[string]bool{}
	for _, amount := range amounts {
		if boundedText(amount.Name, 256, true) != nil || seen[amount.Name] || boundedText(amount.Basis, 1024, false) != nil || boundedText(amount.Provenance, 256, allowance) != nil || ValidateUnsignedDecimal(amount.Bytes) != nil {
			return 0, fmt.Errorf("memory plan: invalid or duplicate amount")
		}
		seen[amount.Name] = true
		if allowance {
			value, err := strconv.ParseUint(amount.Bytes, 10, 64)
			if err != nil || value > math.MaxUint64-total {
				return 0, fmt.Errorf("memory plan: allowance overflow")
			}
			total += value
		}
	}
	return total, nil
}

func (r MemoryPlanResource) validate(kind string) error {
	if r.Kind != kind || !slices.Contains([]string{"unproven", "proven", "not-applicable"}, r.Mapping) || boundedText(r.MappingBasis, 1024, true) != nil {
		return fmt.Errorf("memory plan resource: invalid kind or mapping")
	}
	for _, text := range []string{r.NativeSelector, r.PhysicalID, r.PoolID} {
		if boundedText(text, 256, false) != nil {
			return fmt.Errorf("memory plan resource: invalid identity")
		}
	}
	if kind == "device" && (r.NativeSelector == "" || r.Mapping == "not-applicable") || kind != "device" && r.NativeSelector != "" || r.Mapping == "proven" && r.PhysicalID == "" && r.PoolID == "" || r.Mapping != "proven" && (r.PhysicalID != "" || r.PoolID != "") {
		return fmt.Errorf("memory plan resource: contradictory mapping")
	}
	if len(r.Groups) != len(memoryPlanGroupNames) || len(r.CombinedAllocations) > 256 {
		return fmt.Errorf("memory plan resource: invalid group or combined collection")
	}
	seen := map[string]bool{}
	for _, group := range r.Groups {
		if !slices.Contains(memoryPlanGroupNames, group.Name) || seen[group.Name] || validateMemoryPlanMembers(group.CombinedMembership) != nil {
			return fmt.Errorf("memory plan resource: invalid or duplicate group")
		}
		seen[group.Name] = true
		if err := validateMemoryPlanQuantity(group.Bytes, group.Quality, group.Basis); err != nil {
			return err
		}
		if group.Name == "transient_peak" && group.Additive {
			return fmt.Errorf("memory plan resource: transient peak must be nonadditive")
		}
	}
	for _, combined := range r.CombinedAllocations {
		if len(combined.Members) == 0 || validateMemoryPlanMembers(combined.Members) != nil {
			return fmt.Errorf("memory plan resource: invalid combined membership")
		}
		if err := validateMemoryPlanQuantity(combined.Bytes, combined.Quality, combined.Basis); err != nil {
			return err
		}
	}
	for _, value := range []string{r.AllocatedBytes, r.MappedViewBytes, r.MinimumSelectionCapacityBytes, r.BudgetBytes, r.ShortfallBytes} {
		if value != "" && ValidateUnsignedDecimal(value) != nil {
			return fmt.Errorf("memory plan resource: invalid byte quantity")
		}
	}
	if ValidateUnsignedDecimal(r.PricedTotalBytes) != nil || ValidateUnsignedDecimal(r.RequiredWithAllowancesBytes) != nil || boundedText(r.PricedTotalBasis, 1024, true) != nil || boundedText(r.MinimumSelectionCapacityBasis, 1024, r.MinimumSelectionCapacityBytes != "") != nil || boundedText(r.ShortfallBasis, 1024, r.ShortfallBytes != "") != nil {
		return fmt.Errorf("memory plan resource: invalid total or basis")
	}
	if r.MinimumSelectionCapacityBytes == "" && r.MinimumSelectionCapacityBasis != "" || r.ShortfallBytes == "" && r.ShortfallBasis != "" || (r.BudgetBytes == "") != (r.ShortfallBytes == "") {
		return fmt.Errorf("memory plan resource: inconsistent budget or threshold")
	}
	if _, err := validateMemoryPlanAmounts(r.Assumptions, false); err != nil {
		return err
	}
	allowances, err := validateMemoryPlanAmounts(r.CallerAllowances, true)
	if err != nil {
		return err
	}
	for _, allowance := range r.CallerAllowances {
		if slices.Contains(r.UncoveredCosts, allowance.Name) {
			return fmt.Errorf("memory plan resource: covered cost is also uncovered")
		}
	}
	if err := memoryPlanTexts(r.UncoveredCosts, 256); err != nil {
		return err
	}
	total, err := strconv.ParseUint(r.PricedTotalBytes, 10, 64)
	if err != nil {
		return err
	}
	if r.MinimumSelectionCapacityBytes != "" {
		threshold, parseErr := strconv.ParseUint(r.MinimumSelectionCapacityBytes, 10, 64)
		if parseErr != nil {
			return parseErr
		}
		total = max(total, threshold)
	}
	if allowances > math.MaxUint64-total {
		return fmt.Errorf("memory plan resource: required total overflow")
	}
	required := total + allowances
	if r.RequiredWithAllowancesBytes != strconv.FormatUint(required, 10) {
		return fmt.Errorf("memory plan resource: inconsistent required total")
	}
	if r.BudgetBytes != "" {
		budget, parseErr := strconv.ParseUint(r.BudgetBytes, 10, 64)
		if parseErr != nil {
			return parseErr
		}
		var shortfall uint64
		if required > budget {
			shortfall = required - budget
		}
		if r.ShortfallBytes != strconv.FormatUint(shortfall, 10) {
			return fmt.Errorf("memory plan resource: inconsistent shortfall")
		}
	}
	return nil
}

func (p MemoryPlan) Validate() error {
	i := p.Identity
	for _, text := range []string{i.Engine, i.Provider, i.Revision, i.ContentAssurance} {
		if boundedText(text, 256, true) != nil {
			return fmt.Errorf("memory plan identity: invalid provenance")
		}
	}
	if validateModelID(i.ModelID) != nil || !digest(i.ExecutableSHA256) || !digest(i.RuntimeSourceSHA256) || !digest(i.ArtifactIdentitySHA256) || i.ContentSHA256 != "" && !digest(i.ContentSHA256) || i.AssertedContentSHA256 != "" && !digest(i.AssertedContentSHA256) || boundedText(i.CacheableReason, 1024, !i.Cacheable) != nil || i.Cacheable && i.ContentSHA256 == "" {
		return fmt.Errorf("memory plan identity: invalid digest or cacheability")
	}
	for _, text := range []string{i.Build.Version, i.Build.Revision, i.Build.Timestamp} {
		if boundedText(text, 256, false) != nil {
			return fmt.Errorf("memory plan build: invalid provenance")
		}
	}
	if boundedText(i.Build.Basis, 1024, true) != nil {
		return fmt.Errorf("memory plan build: basis required")
	}
	if err := p.Requested.validate(false); err != nil {
		return err
	}
	if p.PricingQualification != "non-replayable-pricing-inputs" || p.ConfigurationComplete != (p.Resolved != nil) || p.ConfigurationComplete && p.PricingScope != "resolved_configuration" || !p.ConfigurationComplete && p.PricingScope != "minimum_supported_layout" {
		return fmt.Errorf("memory plan: inconsistent configuration or pricing scope")
	}
	if p.Resolved != nil {
		if err := p.Resolved.validate(true); err != nil {
			return err
		}
		if p.Resolved.ContextTokens != p.Requested.ContextTokens || p.Resolved.Placement != p.Requested.Placement || !slices.Equal(p.Resolved.Devices, p.Requested.Devices) {
			return fmt.Errorf("memory plan: resolved configuration changed exact selection")
		}
		canonicalKV := func(value string) string {
			switch value {
			case "f16":
				return "fp16"
			case "q8_0":
				return "int8"
			default:
				return value
			}
		}
		requested, resolved := p.Requested, p.Resolved
		if requested.KVFormat != "" && canonicalKV(requested.KVFormat) != canonicalKV(resolved.KVFormat) || requested.BatchSize != nil && *requested.BatchSize != *resolved.BatchSize || requested.ParallelSlots != nil && *requested.ParallelSlots != *resolved.ParallelSlots {
			return fmt.Errorf("memory plan: resolved configuration changed explicit settings")
		}
		if requested.Microbatch != nil {
			m := requested.Microbatch
			if m.Mode == "fixed" && *m.Tokens != *resolved.Microbatch.Tokens || m.Mode == "auto" && m.Limit != nil && *resolved.Microbatch.Tokens > *m.Limit {
				return fmt.Errorf("memory plan: resolved microbatch changed selection")
			}
		}
	}
	if len(p.Devices) > 64 || len(p.Devices) != len(p.Requested.Devices) {
		return fmt.Errorf("memory plan: invalid device collection")
	}
	selectors, physical := map[string]bool{}, map[string]bool{}
	resources := make([]MemoryPlanResource, 0, len(p.Devices)+2)
	for index, device := range p.Devices {
		if err := device.validate("device"); err != nil {
			return err
		}
		if selectors[device.NativeSelector] || device.NativeSelector != p.Requested.Devices[index].Selector || device.PhysicalID != "" && physical[device.PhysicalID] {
			return fmt.Errorf("memory plan: duplicate or mismatched device")
		}
		selectors[device.NativeSelector] = true
		if device.PhysicalID != "" {
			physical[device.PhysicalID] = true
		}
		resources = append(resources, device)
	}
	if err := p.HostRAM.validate("host_ram"); err != nil {
		return err
	}
	if err := p.HostCommit.validate("host_commit"); err != nil {
		return err
	}
	resources = append(resources, p.HostRAM, p.HostCommit)
	if boundedText(p.Verdict.Basis, 1024, true) != nil || !slices.Contains([]string{"not_evaluated", "does_not_fit", "indeterminate", "fits_within_allowances"}, p.Verdict.Status) {
		return fmt.Errorf("memory plan verdict: invalid status or basis")
	}
	budget := p.Verdict.Status != "not_evaluated"
	if budget && !slices.Contains([]string{"before_runtime_initialization", "after_runtime_initialization"}, p.Verdict.BudgetBasis) || !budget && p.Verdict.BudgetBasis != "" {
		return fmt.Errorf("memory plan verdict: invalid budget basis")
	}
	shortfall, uncovered := false, false
	for _, resource := range resources {
		if (resource.BudgetBytes != "") != budget {
			return fmt.Errorf("memory plan verdict: incomplete budget")
		}
		shortfall = shortfall || resource.ShortfallBytes != "" && resource.ShortfallBytes != "0"
		uncovered = uncovered || len(resource.UncoveredCosts) > 0
	}
	if budget && (shortfall && p.Verdict.Status != "does_not_fit" || !shortfall && uncovered && p.Verdict.Status != "indeterminate" || !shortfall && !uncovered && p.Verdict.Status != "fits_within_allowances") || p.Verdict.Status == "fits_within_allowances" && !p.Verdict.Conditional {
		return fmt.Errorf("memory plan verdict: contradicts resource accounting")
	}
	if err := memoryPlanTexts(p.Warnings, 1024); err != nil {
		return err
	}
	return CheckPayloadSize(p)
}

// Partial describes accounting completeness, independently of advisory fit.
func (p MemoryPlan) partial() bool {
	if !p.ConfigurationComplete || p.Verdict.Conditional {
		return true
	}
	resources := append([]MemoryPlanResource{p.HostRAM, p.HostCommit}, p.Devices...)
	for _, resource := range resources {
		if len(resource.UncoveredCosts) != 0 {
			return true
		}
		for _, group := range resource.Groups {
			if group.Quality == "unknown" {
				return true
			}
		}
		for _, combined := range resource.CombinedAllocations {
			if combined.Quality == "unknown" {
				return true
			}
		}
	}
	return false
}

func (c MemoryPlanConfig) Clone() MemoryPlanConfig {
	c.BatchSize, c.ParallelSlots = clonePointer(c.BatchSize), clonePointer(c.ParallelSlots)
	c.Devices = slices.Clone(c.Devices)
	if c.Microbatch != nil {
		m := *c.Microbatch
		m.Tokens, m.Limit = clonePointer(m.Tokens), clonePointer(m.Limit)
		c.Microbatch = &m
	}
	return c
}

func (r MemoryPlanResource) Clone() MemoryPlanResource {
	r.Groups = slices.Clone(r.Groups)
	for i := range r.Groups {
		r.Groups[i].CombinedMembership = slices.Clone(r.Groups[i].CombinedMembership)
	}
	r.CombinedAllocations = slices.Clone(r.CombinedAllocations)
	for i := range r.CombinedAllocations {
		r.CombinedAllocations[i].Members = slices.Clone(r.CombinedAllocations[i].Members)
	}
	r.Assumptions = slices.Clone(r.Assumptions)
	r.CallerAllowances = slices.Clone(r.CallerAllowances)
	r.UncoveredCosts = slices.Clone(r.UncoveredCosts)
	return r
}

func (p MemoryPlan) Clone() MemoryPlan {
	p.Identity.Build.Dirty = clonePointer(p.Identity.Build.Dirty)
	p.Requested = p.Requested.Clone()
	if p.Resolved != nil {
		resolved := p.Resolved.Clone()
		p.Resolved = &resolved
	}
	p.Devices = slices.Clone(p.Devices)
	for i := range p.Devices {
		p.Devices[i] = p.Devices[i].Clone()
	}
	p.HostRAM, p.HostCommit = p.HostRAM.Clone(), p.HostCommit.Clone()
	p.Warnings = slices.Clone(p.Warnings)
	return p
}

func (s EstimateSource) validate() error {
	if !slices.Contains([]string{"engine-report", "pair-estimate", "measured", "user-stated"}, s.Kind) || boundedText(s.Provider, 256, true) != nil || boundedText(s.Revision, 256, true) != nil || boundedText(s.Basis, 1024, true) != nil || len(s.UnpricedCosts) > 256 {
		return fmt.Errorf("estimate source: invalid provenance")
	}
	seen := map[string]bool{}
	for _, cost := range s.UnpricedCosts {
		if boundedText(cost, 256, true) != nil || seen[cost] {
			return fmt.Errorf("estimate source: invalid unpriced cost")
		}
		seen[cost] = true
	}
	return nil
}
