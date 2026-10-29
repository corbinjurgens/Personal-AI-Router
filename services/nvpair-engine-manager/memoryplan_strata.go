// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/big"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"nvpair-shared/modelconfig"
)

const strataMemoryPlanRevision = "37115e0bfbd4ffd051984b3176d6b56011e7f848"

// These types are private transport, never public settings or replay authority.
type strataMemoryPlanChoice struct {
	Value uint64
	Auto  string
}

func (c strataMemoryPlanChoice) MarshalJSON() ([]byte, error) {
	if c.Auto != "" {
		return json.Marshal(c.Auto)
	}
	return json.Marshal(c.Value)
}
func (c *strataMemoryPlanChoice) UnmarshalJSON(b []byte) error {
	*c = strataMemoryPlanChoice{}
	if len(b) > 0 && b[0] == '"' {
		return json.Unmarshal(b, &c.Auto)
	}
	return json.Unmarshal(b, &c.Value)
}

type strataMemoryPlanDevice struct {
	ID      uint64 `json:"id"`
	Backend string `json:"backend"`
}
type strataMemoryPlanConfig struct {
	Context        uint64                   `json:"context"`
	KVType         string                   `json:"kv_type,omitempty"`
	BatchSize      *uint64                  `json:"batch_size,omitempty"`
	MicrobatchSize *strataMemoryPlanChoice  `json:"microbatch_size,omitempty"`
	ParallelSlots  *uint64                  `json:"parallel_slots,omitempty"`
	Devices        []strataMemoryPlanDevice `json:"devices"`
	Offload        struct {
		Mode string `json:"mode"`
	} `json:"offload"`
}
type strataMemoryPlanModelRequest struct {
	ID            string  `json:"id"`
	ContentSHA256 *string `json:"content_sha256,omitempty"`
}
type strataMemoryPlanDeviceBudget struct {
	ID    uint64 `json:"id"`
	Bytes uint64 `json:"bytes"`
}
type strataMemoryPlanBudget struct {
	Basis           string                         `json:"basis"`
	Devices         []strataMemoryPlanDeviceBudget `json:"devices"`
	HostRAMBytes    uint64                         `json:"host_ram_bytes"`
	HostCommitBytes uint64                         `json:"host_commit_bytes"`
}
type strataMemoryPlanOptions struct {
	Executable          string                 `json:"executable"`
	WorkingDirectory    string                 `json:"working_directory"`
	Pack                string                 `json:"pack"`
	NativeModel         string                 `json:"native_model"`
	ExpertProfile       string                 `json:"expert_profile"`
	ResidentBudgetBytes uint64                 `json:"resident_budget_bytes"`
	FileCacheBytes      uint64                 `json:"file_cache_bytes"`
	HostHeadroomBytes   uint64                 `json:"host_headroom_bytes"`
	VRAMReserveBytes    uint64                 `json:"vram_reserve_bytes"`
	CacheSlots          strataMemoryPlanChoice `json:"cache_slots"`
	KVResidentCells     uint64                 `json:"kv_resident_cells"`
	Spec                uint64                 `json:"spec"`
	NoPrefillBorrow     bool                   `json:"no_prefill_borrow"`
	Environment         map[string]string      `json:"environment,omitempty"`
	Allowances          map[string]uint64      `json:"allowances,omitempty"`
}
type strataMemoryPlanRequestExtensions struct {
	Strata strataMemoryPlanOptions `json:"strata"`
}
type strataMemoryPlanRequest struct {
	SchemaVersion int                               `json:"schema_version"`
	Model         strataMemoryPlanModelRequest      `json:"model"`
	Config        strataMemoryPlanConfig            `json:"config"`
	Budget        *strataMemoryPlanBudget           `json:"budget,omitempty"`
	Extensions    strataMemoryPlanRequestExtensions `json:"extensions"`
}
type strataMemoryPlanBuild struct {
	SchemaVersion   int     `json:"schema_version"`
	Version         string  `json:"version"`
	Backend         string  `json:"backend"`
	Commit          *string `json:"commit"`
	Dirty           *bool   `json:"dirty"`
	BuiltAt         *string `json:"built_at,omitempty"`
	CUDA            *string `json:"cuda"`
	Configuration   string  `json:"configuration,omitempty"`
	Architectures   string  `json:"architectures,omitempty"`
	GitReason       *string `json:"git_reason,omitempty"`
	CUDAReason      *string `json:"cuda_reason,omitempty"`
	SourceSHA256    string  `json:"source_sha256,omitempty"`
	ConfigSHA256    string  `json:"config_sha256,omitempty"`
	ToolchainSHA256 string  `json:"toolchain_sha256,omitempty"`
}
type strataMemoryPlanInputIdentity struct {
	Path           string   `json:"path"`
	SizeBytes      uint64   `json:"size_bytes"`
	FileID         []uint64 `json:"file_id"`
	ContentSHA256  *string  `json:"content_sha256"`
	MetadataBytes  *uint64  `json:"metadata_bytes"`
	MetadataSHA256 *string  `json:"metadata_sha256"`
}
type strataMemoryPlanQuantity struct {
	Bytes              *uint64  `json:"bytes"`
	Quality            string   `json:"quality"`
	Basis              string   `json:"basis"`
	Additive           *bool    `json:"additive,omitempty"`
	CombinedMembership []string `json:"combined_membership,omitempty"`
}
type strataMemoryPlanCombined struct {
	Bytes   *uint64  `json:"bytes"`
	Quality string   `json:"quality"`
	Basis   string   `json:"basis"`
	Members []string `json:"members"`
}
type strataMemoryPlanAmount struct {
	Name  string `json:"name"`
	Bytes uint64 `json:"bytes"`
	Basis string `json:"basis,omitempty"`
}
type strataMemoryPlanResource struct {
	ID                            *uint64                             `json:"id,omitempty"`
	Backend                       string                              `json:"backend,omitempty"`
	Groups                        map[string]strataMemoryPlanQuantity `json:"groups"`
	CombinedAllocations           []strataMemoryPlanCombined          `json:"combined_allocations"`
	AllocatedBytes                *uint64                             `json:"allocated_bytes"`
	MappedViewBytes               *uint64                             `json:"mapped_view_bytes"`
	Assumptions                   []strataMemoryPlanAmount            `json:"assumptions"`
	PricedTotalBytes              uint64                              `json:"priced_total_bytes"`
	PricedTotalBasis              string                              `json:"priced_total_basis"`
	MinimumSelectionCapacityBytes uint64                              `json:"minimum_selection_capacity_bytes"`
	MinimumSelectionCapacityBasis string                              `json:"minimum_selection_capacity_basis"`
	CallerAllowances              map[string]uint64                   `json:"caller_allowances"`
	RequiredWithAllowancesBytes   uint64                              `json:"required_with_allowances_bytes"`
	BudgetBytes                   *uint64                             `json:"budget_bytes"`
	ShortfallBytes                *uint64                             `json:"shortfall_bytes"`
	ShortfallBasis                string                              `json:"shortfall_basis"`
	UncoveredCosts                []string                            `json:"uncovered_costs"`
}
type strataMemoryPlanNativeComponents struct {
	VRAM                       map[string]uint64 `json:"vram"`
	RAM                        map[string]uint64 `json:"ram"`
	VRAMRequired               uint64            `json:"vram_required"`
	RAMRequired                uint64            `json:"ram_required"`
	CommitRequired             uint64            `json:"commit_required"`
	UnknownCosts               string            `json:"unknown_costs"`
	MinimumPolicyVRAMFreeBytes uint64            `json:"minimum_policy_vram_free_bytes,omitempty"`
}
type strataMemoryPlanNativeFit struct {
	Status         string `json:"status"`
	ShortfallBytes struct {
		VRAM   uint64 `json:"vram"`
		RAM    uint64 `json:"ram"`
		Commit uint64 `json:"commit"`
	} `json:"shortfall_bytes"`
}
type strataMemoryPlanPrivate struct {
	Requested        strataMemoryPlanOptions          `json:"requested"`
	Resolved         *strataMemoryPlanNativeSettings  `json:"resolved"`
	NativeComponents strataMemoryPlanNativeComponents `json:"native_components"`
	NativeFit        strataMemoryPlanNativeFit        `json:"native_fit"`
	PricingBudget    *strataMemoryPlanBudget          `json:"pricing_budget"`
	Inputs           []strataMemoryPlanInputIdentity  `json:"inputs"`
	EstimateBasis    string                           `json:"estimate_basis"`
	Query            struct {
		Argv             []string          `json:"argv"`
		Environment      map[string]string `json:"environment"`
		WorkingDirectory string            `json:"working_directory"`
	} `json:"query"`
}
type strataMemoryPlanResponse struct {
	Kind          string `json:"kind"`
	SchemaVersion int    `json:"schema_version"`
	Engine        struct {
		ID               string                `json:"id"`
		Build            strataMemoryPlanBuild `json:"build"`
		ExecutableSHA256 string                `json:"executable_sha256"`
		IdentityBasis    string                `json:"identity_basis"`
	} `json:"engine"`
	Model struct {
		ID                    string  `json:"id"`
		ContentSHA256         *string `json:"content_sha256"`
		AssertedContentSHA256 *string `json:"asserted_content_sha256"`
		IdentityBasis         string  `json:"identity_basis"`
	} `json:"model"`
	RequestedConfig       strataMemoryPlanConfig  `json:"requested_config"`
	ResolvedConfig        *strataMemoryPlanConfig `json:"resolved_config"`
	ConfigurationComplete bool                    `json:"configuration_complete"`
	BudgetBasis           *string                 `json:"budget_basis"`
	PricingScope          string                  `json:"pricing_scope"`
	Resources             struct {
		Devices    []strataMemoryPlanResource `json:"devices"`
		HostRAM    strataMemoryPlanResource   `json:"host_ram"`
		HostCommit strataMemoryPlanResource   `json:"host_commit"`
	} `json:"resources"`
	Fit struct {
		Status      string `json:"status"`
		Conditional bool   `json:"conditional"`
		Basis       string `json:"basis"`
	} `json:"fit"`
	Cacheable       bool     `json:"cacheable"`
	CacheableReason string   `json:"cacheable_reason"`
	Warnings        []string `json:"warnings"`
	Capabilities    struct {
		LiveObservation    bool     `json:"live_observation"`
		MaxContextSearch   bool     `json:"max_context_search"`
		AutomaticCacheFill bool     `json:"automatic_cache_fill"`
		SupportedBackends  []string `json:"supported_backends"`
		SerialOnly         bool     `json:"serial_only"`
	} `json:"capabilities"`
	Extensions struct {
		Strata strataMemoryPlanPrivate `json:"strata"`
	} `json:"extensions"`
}

// The source owner supplies independently admitted identities. The decoder never
// constructs expected identities from a response in production.
type strataMemoryPlanBinding struct {
	Request             strataMemoryPlanRequest
	ExecutableSHA256    string
	Build               strataMemoryPlanBuild
	RuntimeSourceSHA256 string
	Inputs              []strataMemoryPlanInputIdentity
}

var errStrataMemoryPlan = errors.New("Strata memory report unavailable or invalid")
var errStrataMemoryPlanCleanup = errors.New("Strata memory query cleanup unconfirmed")

// Required nulls and required false/zero values must not disappear in decoding.
// The private transport has its own size ceiling; the public plan is smaller.
func strataPlanRequired(b []byte, t reflect.Type) error {
	b = bytes.TrimSpace(b)
	if t.Kind() == reflect.Pointer {
		if bytes.Equal(b, []byte("null")) {
			return nil
		}
		return strataPlanRequired(b, t.Elem())
	}
	if bytes.Equal(b, []byte("null")) {
		return errStrataMemoryPlan
	}
	if t == reflect.TypeFor[strataMemoryPlanChoice]() || t == reflect.TypeFor[strataMemoryPlanNativeSettings]() {
		return nil
	}
	switch t.Kind() {
	case reflect.Struct:
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(b, &obj); err != nil {
			return err
		}
		allowed := make(map[string]bool, t.NumField())
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			tag := strings.Split(f.Tag.Get("json"), ",")
			allowed[tag[0]] = true
			v, ok := obj[tag[0]]
			if !ok {
				if len(tag) > 1 && tag[1] == "omitempty" {
					continue
				}
				return errStrataMemoryPlan
			}
			if err := strataPlanRequired(v, f.Type); err != nil {
				return err
			}
		}
		for key := range obj {
			if !allowed[key] {
				return errStrataMemoryPlan
			}
		}
	case reflect.Slice:
		var list []json.RawMessage
		if err := json.Unmarshal(b, &list); err != nil {
			return err
		}
		for _, v := range list {
			if err := strataPlanRequired(v, t.Elem()); err != nil {
				return err
			}
		}
	case reflect.Map:
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(b, &obj); err != nil {
			return err
		}
		for _, v := range obj {
			if err := strataPlanRequired(v, t.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}
func strataPlanDecode(b []byte, out any) error {
	if len(b) > memoryPlanOutputLimit || !utf8.Valid(b) {
		return errStrataMemoryPlan
	}
	scan := json.NewDecoder(bytes.NewReader(b))
	scan.UseNumber()
	if strataPlanScan(scan, 0) != nil {
		return errStrataMemoryPlan
	}
	if _, err := scan.Token(); err != io.EOF {
		return errStrataMemoryPlan
	}
	if err := strataPlanRequired(b, reflect.TypeOf(out).Elem()); err != nil {
		return errStrataMemoryPlan
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if d.Decode(out) != nil {
		return errStrataMemoryPlan
	}
	return nil
}

func strataPlanScan(d *json.Decoder, depth int) error {
	if depth > 32 {
		return errStrataMemoryPlan
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[strings.ToLower(name)] {
				return errStrataMemoryPlan
			}
			seen[strings.ToLower(name)] = true
			if err := strataPlanScan(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := strataPlanScan(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return errStrataMemoryPlan
	}
	_, err = d.Token()
	return err
}
func strataPlanBytes(n *uint64) string {
	if n == nil {
		return ""
	}
	return strconv.FormatUint(*n, 10)
}
func strataPlanNumber(n uint64) string { return strconv.FormatUint(n, 10) }
func strataPlanDigest(s string) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == sha256.Size && strings.ToLower(s) == s
}

func (c strataMemoryPlanConfig) neutral() (modelconfig.MemoryPlanConfig, error) {
	if c.Context == 0 || c.Context > 1048576 || len(c.Devices) != 1 || c.Devices[0].ID > 65535 || c.Devices[0].Backend != "cuda" || c.Offload.Mode != "hybrid" {
		return modelconfig.MemoryPlanConfig{}, errStrataMemoryPlan
	}
	n := modelconfig.MemoryPlanConfig{ContextTokens: int32(c.Context), KVFormat: c.KVType, Placement: c.Offload.Mode, Devices: []modelconfig.MemoryPlanSelector{{Backend: "cuda", Selector: strataPlanNumber(c.Devices[0].ID)}}}
	if !slices.Contains([]string{"", "fp16", "int8", "q4_0", "k8v4", "f16", "q8_0"}, c.KVType) {
		return n, errStrataMemoryPlan
	}
	for _, v := range []*uint64{c.BatchSize, c.ParallelSlots} {
		if v != nil && *v != 1 {
			return n, errStrataMemoryPlan
		}
	}
	one := int32(1)
	if c.BatchSize != nil {
		v := one
		n.BatchSize = &v
	}
	if c.ParallelSlots != nil {
		v := one
		n.ParallelSlots = &v
	}
	if m := c.MicrobatchSize; m != nil {
		n.Microbatch = &modelconfig.MemoryPlanMicrobatch{}
		if m.Auto != "" {
			n.Microbatch.Mode = "auto"
			switch m.Auto {
			case "auto":
			case "auto:8192", "auto:16384", "auto:32768":
				v, _ := strconv.ParseInt(strings.TrimPrefix(m.Auto, "auto:"), 10, 32)
				limit := int32(v)
				n.Microbatch.Limit = &limit
			default:
				return n, errStrataMemoryPlan
			}
		} else {
			if m.Value == 0 || m.Value > min(c.Context, 32768) {
				return n, errStrataMemoryPlan
			}
			v := int32(m.Value)
			n.Microbatch.Mode = "fixed"
			n.Microbatch.Tokens = &v
		}
	}
	return n, nil
}

var strataPlanAllowanceOwners = map[string]string{"device_runtime_context": "device", "device_late_libraries": "device", "device_transient_peak": "device", "host_ram_runtime": "host_ram", "host_ram_transient_peak": "host_ram", "host_commit_runtime": "host_commit", "host_commit_transient_peak": "host_commit"}

func (r strataMemoryPlanRequest) validate() error {
	if r.SchemaVersion != 1 || r.Model.ID == "" || r.Model.ContentSHA256 != nil && !strataPlanDigest(*r.Model.ContentSHA256) {
		return errStrataMemoryPlan
	}
	if _, err := r.Config.neutral(); err != nil {
		return err
	}
	s := r.Extensions.Strata
	for _, p := range []string{s.Executable, s.WorkingDirectory, s.Pack, s.NativeModel, s.ExpertProfile} {
		if p == "" || len(p) > 4096 || strings.ContainsAny(p, "\x00\r\n") {
			return errStrataMemoryPlan
		}
	}
	if s.ResidentBudgetBytes == 0 || s.ResidentBudgetBytes > 1<<40 || s.FileCacheBytes > 1<<40 || s.HostHeadroomBytes > 1<<40 || s.VRAMReserveBytes > 1<<40 || s.FileCacheBytes%(1<<20) != 0 || s.VRAMReserveBytes%(1<<20) != 0 || s.Spec != 2 || !s.NoPrefillBorrow || s.KVResidentCells >= r.Config.Context {
		return errStrataMemoryPlan
	}
	if s.CacheSlots.Auto != "auto" && (s.CacheSlots.Auto != "" || s.CacheSlots.Value == 0 || s.CacheSlots.Value > 24576) {
		return errStrataMemoryPlan
	}
	if r.Budget == nil && (s.CacheSlots.Auto != "" || r.Config.MicrobatchSize == nil || r.Config.MicrobatchSize.Auto != "") {
		return errStrataMemoryPlan
	}
	for k := range s.Allowances {
		if _, ok := strataPlanAllowanceOwners[k]; !ok {
			return errStrataMemoryPlan
		}
	}
	for k, v := range s.Environment {
		if !strataPlanEnvironment(k, v) {
			return errStrataMemoryPlan
		}
	}
	if b := r.Budget; b != nil {
		if !slices.Contains([]string{"before_runtime_initialization", "after_runtime_initialization"}, b.Basis) || len(b.Devices) != 1 || b.Devices[0].ID != r.Config.Devices[0].ID {
			return errStrataMemoryPlan
		}
	}
	return nil
}
func strataPlanEnvironment(k, v string) bool {
	switch k {
	case "STRATA_FILE_CACHE_COMPACT", "STRATA_PREFILL_MMQ", "STRATA_PREFILL_RESERVE_BO":
		return v == "0" || v == "1"
	case "STRATA_OWNED_PRICE":
		return v == "exact"
	case "CUDA_MODULE_LOADING":
		return v == "EAGER" || v == "LAZY"
	case "CUDA_DEVICE_ORDER":
		return v == "PCI_BUS_ID"
	}
	low, high := uint64(1), uint64(256)
	switch k {
	case "STRATA_PREFILL_RING":
		low, high = 8, 512
	case "STRATA_STAGER_THREADS", "STRATA_STAGER_BATCH":
	case "STRATA_STAGER_RING":
		low = 2
	default:
		return false
	}
	n, e := strconv.ParseUint(v, 10, 64)
	return e == nil && strataPlanNumber(n) == v && n >= low && n <= high && !(k == "STRATA_PREFILL_RING" && n > 8 && n < 16)
}

// Production uses this finite, exact-setting conversion. Budgets and allowances
// remain private mapper-test inputs until their later target-evidence owner exists.
func encodeStrataMemoryPlanRequest(c modelconfig.MemoryPlanConfig, model strataMemoryPlanModelRequest, s strataMemoryPlanOptions) ([]byte, error) {
	r := strataMemoryPlanRequest{SchemaVersion: 1, Model: model, Extensions: strataMemoryPlanRequestExtensions{Strata: s}}
	if c.ContextTokens <= 0 || c.KVFormat == "" || c.BatchSize == nil || c.ParallelSlots == nil || c.Placement != "hybrid" || len(c.Devices) != 1 || c.Microbatch == nil || c.Microbatch.Mode != "fixed" || c.Microbatch.Tokens == nil || *c.Microbatch.Tokens <= 0 || len(s.Allowances) != 0 || s.CacheSlots.Auto != "" {
		return nil, errStrataMemoryPlan
	}
	id, err := strconv.ParseUint(c.Devices[0].Selector, 10, 64)
	if err != nil || strataPlanNumber(id) != c.Devices[0].Selector {
		return nil, errStrataMemoryPlan
	}
	r.Config.Context = uint64(c.ContextTokens)
	r.Config.KVType = c.KVFormat
	r.Config.Devices = []strataMemoryPlanDevice{{ID: id, Backend: c.Devices[0].Backend}}
	r.Config.Offload.Mode = c.Placement
	one := uint64(1)
	r.Config.BatchSize = &one
	r.Config.ParallelSlots = &one
	if c.BatchSize != nil && *c.BatchSize != 1 || c.ParallelSlots != nil && *c.ParallelSlots != 1 || c.Microbatch.Limit != nil {
		return nil, errStrataMemoryPlan
	}
	r.Config.MicrobatchSize = &strataMemoryPlanChoice{Value: uint64(*c.Microbatch.Tokens)}
	if err := r.validate(); err != nil {
		return nil, err
	}
	return json.Marshal(r)
}

func (r strataMemoryPlanResource) neutral(kind string) modelconfig.MemoryPlanResource {
	n := modelconfig.MemoryPlanResource{Kind: kind, Mapping: "not-applicable", MappingBasis: "Independent engine capacity constraint; not observed usage.", AllocatedBytes: strataPlanBytes(r.AllocatedBytes), MappedViewBytes: strataPlanBytes(r.MappedViewBytes), PricedTotalBytes: strataPlanNumber(r.PricedTotalBytes), PricedTotalBasis: r.PricedTotalBasis, MinimumSelectionCapacityBytes: strataPlanNumber(r.MinimumSelectionCapacityBytes), MinimumSelectionCapacityBasis: r.MinimumSelectionCapacityBasis, RequiredWithAllowancesBytes: strataPlanNumber(r.RequiredWithAllowancesBytes), BudgetBytes: strataPlanBytes(r.BudgetBytes), ShortfallBytes: strataPlanBytes(r.ShortfallBytes), UncoveredCosts: slices.Clone(r.UncoveredCosts)}
	if r.ShortfallBytes != nil {
		n.ShortfallBasis = r.ShortfallBasis
	}
	if kind == "device" && r.ID != nil {
		n.NativeSelector = strataPlanNumber(*r.ID)
		n.Mapping = "unproven"
		n.MappingBasis = "Native CUDA ordinal has no proven physical mapping."
	}
	for _, name := range []string{"weights", "kv_cache", "compute_scratch", "output_buffers", "runtime_driver_overhead", "transient_peak"} {
		g := r.Groups[name]
		n.Groups = append(n.Groups, modelconfig.MemoryPlanGroup{Name: name, Bytes: strataPlanBytes(g.Bytes), Quality: g.Quality, Basis: g.Basis, Additive: g.Additive != nil && *g.Additive, CombinedMembership: slices.Clone(g.CombinedMembership)})
	}
	for _, g := range r.CombinedAllocations {
		n.CombinedAllocations = append(n.CombinedAllocations, modelconfig.MemoryPlanCombinedAllocation{Bytes: strataPlanBytes(g.Bytes), Quality: g.Quality, Basis: g.Basis, Members: slices.Clone(g.Members)})
	}
	for _, a := range r.Assumptions {
		n.Assumptions = append(n.Assumptions, modelconfig.MemoryPlanAmount{Name: a.Name, Bytes: strataPlanNumber(a.Bytes), Basis: a.Basis})
	}
	keys := make([]string, 0, len(r.CallerAllowances))
	for k := range r.CallerAllowances {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		n.CallerAllowances = append(n.CallerAllowances, modelconfig.MemoryPlanAmount{Name: k, Bytes: strataPlanNumber(r.CallerAllowances[k]), Provenance: "caller-asserted-upper-bound"})
	}
	return n
}

func decodeStrataMemoryPlan(b []byte, exit int, binding strataMemoryPlanBinding) (modelconfig.MemoryPlan, error) {
	var p modelconfig.MemoryPlan
	if exit != 0 {
		var failure struct {
			Error struct {
				Code    string  `json:"code"`
				Message string  `json:"message"`
				Field   *string `json:"field,omitempty"`
				Cleanup string  `json:"cleanup,omitempty"`
			} `json:"error"`
		}
		if exit == 2 && strataPlanDecode(b, &failure) == nil && failure.Error.Cleanup == "unconfirmed" {
			return p, errStrataMemoryPlanCleanup
		}
		return p, errStrataMemoryPlan
	}
	var r strataMemoryPlanResponse
	if strataPlanDecode(b, &r) != nil || binding.Request.validate() != nil {
		return p, errStrataMemoryPlan
	}
	s := r.Extensions.Strata
	if r.Kind != "engine_memory_plan" || r.SchemaVersion != 1 || r.Engine.ID != "strata" || r.Engine.Build.Backend != "cuda" || r.Engine.Build.SchemaVersion != 1 || r.Cacheable || r.Model.ContentSHA256 != nil || !r.Fit.Conditional || r.Capabilities.LiveObservation || r.Capabilities.MaxContextSearch || !r.Capabilities.SerialOnly || !r.Capabilities.AutomaticCacheFill || !slices.Equal(r.Capabilities.SupportedBackends, []string{"cuda"}) {
		return p, errStrataMemoryPlan
	}
	if r.Model.ID != binding.Request.Model.ID || !reflect.DeepEqual(r.Model.AssertedContentSHA256, binding.Request.Model.ContentSHA256) || r.Engine.ExecutableSHA256 != binding.ExecutableSHA256 || !reflect.DeepEqual(r.Engine.Build, binding.Build) || !reflect.DeepEqual(r.RequestedConfig, binding.Request.Config) || !reflect.DeepEqual(s.Requested, binding.Request.Extensions.Strata) || !reflect.DeepEqual(s.Inputs, binding.Inputs) || len(s.Inputs) == 0 || len(s.Inputs) > 256 {
		return p, errStrataMemoryPlan
	}
	if err := validateStrataPlanAccounting(r, binding.Request); err != nil {
		return p, err
	}
	identityBytes, err := json.Marshal(s.Inputs)
	if err != nil {
		return p, errStrataMemoryPlan
	}
	digest := sha256.Sum256(identityBytes)
	p.Identity = modelconfig.MemoryPlanIdentity{Engine: "strata", Provider: "strata", Revision: strataMemoryPlanRevision, ExecutableSHA256: r.Engine.ExecutableSHA256, RuntimeSourceSHA256: binding.RuntimeSourceSHA256, ModelID: r.Model.ID, ContentAssurance: "sampled-metadata-and-stat-only", ArtifactIdentitySHA256: hex.EncodeToString(digest[:]), Cacheable: false, CacheableReason: r.CacheableReason, Build: modelconfig.MemoryPlanBuild{Version: r.Engine.Build.Version, Dirty: r.Engine.Build.Dirty, Basis: r.Engine.IdentityBasis}}
	if r.Engine.Build.Commit != nil {
		p.Identity.Build.Revision = *r.Engine.Build.Commit
	}
	if r.Engine.Build.BuiltAt != nil {
		p.Identity.Build.Timestamp = *r.Engine.Build.BuiltAt
	}
	if r.Model.AssertedContentSHA256 != nil {
		p.Identity.AssertedContentSHA256 = *r.Model.AssertedContentSHA256
	}
	p.Requested, err = r.RequestedConfig.neutral()
	if err != nil {
		return modelconfig.MemoryPlan{}, err
	}
	if r.ResolvedConfig != nil {
		v, e := r.ResolvedConfig.neutral()
		if e != nil {
			return modelconfig.MemoryPlan{}, e
		}
		p.Resolved = &v
	}
	p.ConfigurationComplete = r.ConfigurationComplete
	p.PricingScope = r.PricingScope
	p.PricingQualification = "non-replayable-pricing-inputs"
	for _, d := range r.Resources.Devices {
		p.Devices = append(p.Devices, d.neutral("device"))
	}
	p.HostRAM = r.Resources.HostRAM.neutral("host_ram")
	p.HostCommit = r.Resources.HostCommit.neutral("host_commit")
	p.Verdict = modelconfig.MemoryPlanVerdict{Status: r.Fit.Status, Conditional: r.Fit.Conditional, Basis: r.Fit.Basis}
	if r.BudgetBasis != nil {
		p.Verdict.BudgetBasis = *r.BudgetBasis
	}
	p.Warnings = slices.Clone(r.Warnings)
	p.Warnings = append(p.Warnings, "Pricing envelope inputs are not replayable launch settings.")
	if p.Validate() != nil {
		return modelconfig.MemoryPlan{}, errStrataMemoryPlan
	}
	wire, err := json.Marshal(p)
	if err != nil || len(wire) > modelconfig.MaxPayloadBytes {
		return modelconfig.MemoryPlan{}, errStrataMemoryPlan
	}
	for _, path := range []string{s.Requested.Executable, s.Requested.WorkingDirectory, s.Requested.Pack, s.Requested.NativeModel, s.Requested.ExpertProfile} {
		quoted, err := json.Marshal(path)
		if err != nil || path != "" && bytes.Contains(wire, quoted[1:len(quoted)-1]) {
			return modelconfig.MemoryPlan{}, errStrataMemoryPlan
		}
	}
	for _, in := range s.Inputs {
		quoted, err := json.Marshal(in.Path)
		if err != nil || bytes.Contains(wire, quoted[1:len(quoted)-1]) {
			return modelconfig.MemoryPlan{}, errStrataMemoryPlan
		}
	}
	return p, nil
}

var strataPlanDeviceComponents = strings.Fields("weights_canonical weights_native_dense weights_native_head head_scratch_logits session_including_resident_kv verifier prefill_scratch_ring_stage prefill_token_ids prefill_kv_identity cache_tables decode_hit_scratch expert_cache overhead_allowance")
var strataPlanHostComponents = strings.Fields("resident_experts file_cache_target mapped_embedding verifier_mapped streamed_kv host_headroom_allowance startup_scratch_allowance prefill_host_stager")
var strataPlanScalars = strings.Fields("context prefill spec kv kv_resident batch batch_groups parallel auto_cache per_layer no_prefill_borrow resident_budget resident_bytes resident_experts resident_pin file_cache host_headroom vram_reserve shared_stage ring workspace weight_scratch common region prefill_mmq prefill_stage_own prefill_reserve_bo prefill_ring_enabled experts top_k vocab max_blob cache_request environment file_cache_effective file_cache_compact vram_required ram_required commit_required stager_threads stager_ring stager_batch")

type strataMemoryPlanNativeSettings struct {
	Scalars       map[string]uint64
	LayerBlobs    []uint64
	KVStageRuns   []uint64
	ResidentOrder []uint64
	Slots         [][]uint64
}

func (s *strataMemoryPlanNativeSettings) UnmarshalJSON(b []byte) error {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil {
		return err
	}
	if len(obj) != len(strataPlanScalars)+4 {
		return errStrataMemoryPlan
	}
	s.Scalars = map[string]uint64{}
	for _, k := range strataPlanScalars {
		var n uint64
		v, ok := obj[k]
		if !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) || json.Unmarshal(v, &n) != nil {
			return errStrataMemoryPlan
		}
		s.Scalars[k] = n
	}
	for k, dst := range map[string]any{"layer_blobs": &s.LayerBlobs, "kv_stage_runs": &s.KVStageRuns, "resident_order": &s.ResidentOrder, "slots": &s.Slots} {
		v, ok := obj[k]
		if !ok || strataPlanRequired(v, reflect.TypeOf(dst).Elem()) != nil || json.Unmarshal(v, dst) != nil {
			return errStrataMemoryPlan
		}
	}
	if len(s.LayerBlobs) > 256 || len(s.KVStageRuns) > 256 || len(s.ResidentOrder) > 24576 || len(s.Slots) > 24576 {
		return errStrataMemoryPlan
	}
	for _, row := range s.Slots {
		if len(row) != 4 {
			return errStrataMemoryPlan
		}
	}
	return nil
}
func strataPlanSum(m map[string]uint64, keys []string) (uint64, error) {
	if len(m) != len(keys) {
		return 0, errStrataMemoryPlan
	}
	var total uint64
	for _, k := range keys {
		v, ok := m[k]
		if !ok || v > math.MaxUint64-total {
			return 0, errStrataMemoryPlan
		}
		total += v
	}
	return total, nil
}
func validateStrataPlanAccounting(r strataMemoryPlanResponse, request strataMemoryPlanRequest) error {
	s := r.Extensions.Strata
	e := s.NativeComponents
	v, err := strataPlanSum(e.VRAM, strataPlanDeviceComponents)
	if err != nil {
		return err
	}
	h, err := strataPlanSum(e.RAM, strataPlanHostComponents)
	if err != nil {
		return err
	}
	if v != e.VRAMRequired || h != e.RAMRequired || h > math.MaxUint64-v || h+v != e.CommitRequired || len(r.Resources.Devices) != 1 {
		return errStrataMemoryPlan
	}
	d := r.Resources.Devices[0]
	if d.ID == nil || *d.ID != request.Config.Devices[0].ID || d.Backend != "cuda" || r.Resources.HostRAM.ID != nil || r.Resources.HostRAM.Backend != "" || r.Resources.HostCommit.ID != nil || r.Resources.HostCommit.Backend != "" || d.PricedTotalBytes != v || r.Resources.HostRAM.PricedTotalBytes != h || r.Resources.HostCommit.PricedTotalBytes != h+v || d.MinimumSelectionCapacityBytes != e.MinimumPolicyVRAMFreeBytes || r.Resources.HostRAM.MinimumSelectionCapacityBytes != 0 || r.Resources.HostCommit.MinimumSelectionCapacityBytes != 0 {
		return errStrataMemoryPlan
	}
	if r.ConfigurationComplete != (s.Resolved != nil) || r.ConfigurationComplete != (r.ResolvedConfig != nil) {
		return errStrataMemoryPlan
	}
	if r.ConfigurationComplete {
		cfg := r.ResolvedConfig
		n := s.Resolved.Scalars
		o := request.Extensions.Strata
		kv := map[string]uint64{"fp16": 0, "int8": 1, "q4_0": 2, "k8v4": 3}
		if n["context"] != cfg.Context || n["prefill"] == 0 || cfg.MicrobatchSize == nil || cfg.MicrobatchSize.Auto != "" || n["prefill"] != cfg.MicrobatchSize.Value || n["kv"] != kv[cfg.KVType] || n["spec"] != 2 || n["batch"] != 0 || n["parallel"] != 1 || n["batch_groups"] != 1 || n["no_prefill_borrow"] != 1 || n["resident_budget"] != o.ResidentBudgetBytes || n["file_cache"] != o.FileCacheBytes || n["host_headroom"] != o.HostHeadroomBytes || n["vram_reserve"] != o.VRAMReserveBytes || n["kv_resident"] != o.KVResidentCells || o.CacheSlots.Auto == "" && n["cache_request"] != o.CacheSlots.Value {
			return errStrataMemoryPlan
		}
	}
	var basis string
	if request.Budget != nil {
		basis = request.Budget.Basis
	}
	if (r.BudgetBasis == nil) != (request.Budget == nil) || r.BudgetBasis != nil && *r.BudgetBasis != basis {
		return errStrataMemoryPlan
	}
	resources := []strataMemoryPlanResource{d, r.Resources.HostRAM, r.Resources.HostCommit}
	kinds := []string{"device", "host_ram", "host_commit"}
	pricing := request.Budget
	if pricing != nil {
		copyBudget := *pricing
		copyBudget.Devices = slices.Clone(pricing.Devices)
		pricing = &copyBudget
	}
	for i, res := range resources {
		if res.AllocatedBytes != nil && *res.AllocatedBytes > res.PricedTotalBytes || res.MappedViewBytes != nil && *res.MappedViewBytes > res.PricedTotalBytes {
			return errStrataMemoryPlan
		}
		if len(res.Groups) != 6 {
			return errStrataMemoryPlan
		}
		for _, k := range []string{"weights", "kv_cache", "compute_scratch", "output_buffers", "runtime_driver_overhead", "transient_peak"} {
			g, ok := res.Groups[k]
			if !ok || g.Additive != nil && *g.Additive {
				return errStrataMemoryPlan
			}
			known := i == 0 && k == "weights" || i == 1 && (k == "weights" || k == "kv_cache")
			if !known && (g.Bytes != nil || g.Quality != "unknown") {
				return errStrataMemoryPlan
			}
		}
		expected := map[string]uint64{}
		uncovered := []string{}
		keys := make([]string, 0, len(strataPlanAllowanceOwners))
		for k := range strataPlanAllowanceOwners {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		for _, k := range keys {
			if strataPlanAllowanceOwners[k] != kinds[i] || k == "device_runtime_context" && basis == "after_runtime_initialization" {
				continue
			}
			if value, ok := request.Extensions.Strata.Allowances[k]; ok {
				expected[k] = value
			} else {
				uncovered = append(uncovered, k)
			}
		}
		nativeUnknown := "CUDA context/modules, graph capture, CPU TLS/request scratch, driver and OS residency"
		if e.UnknownCosts != nativeUnknown && e.UnknownCosts != nativeUnknown+"; prefill_scratch_ring_stage is a conservative CUDA kernel-capability envelope (possible MMQ plus FP16 fallback, fused/nonfused carve maxima and native ring bound), not observed runtime allocation; prefill settings are envelope count inputs, not replayable runtime settings" {
			uncovered = append(uncovered, "unclassified_native_costs")
		}
		actual := slices.Clone(res.UncoveredCosts)
		slices.Sort(actual)
		slices.Sort(uncovered)
		if !reflect.DeepEqual(res.CallerAllowances, expected) || !slices.Equal(actual, uncovered) {
			return errStrataMemoryPlan
		}
		var allowance uint64
		for _, v := range expected {
			if v > math.MaxUint64-allowance {
				return errStrataMemoryPlan
			}
			allowance += v
		}
		if pricing != nil {
			var budget uint64
			switch i {
			case 0:
				budget = request.Budget.Devices[0].Bytes
				pricing.Devices[0].Bytes = budget - min(budget, allowance)
			case 1:
				budget = request.Budget.HostRAMBytes
				pricing.HostRAMBytes = budget - min(budget, allowance)
			case 2:
				budget = request.Budget.HostCommitBytes
				pricing.HostCommitBytes = budget - min(budget, allowance)
			}
			if res.BudgetBytes == nil || *res.BudgetBytes != budget {
				return errStrataMemoryPlan
			}
		} else if res.BudgetBytes != nil || res.ShortfallBytes != nil {
			return errStrataMemoryPlan
		}
	}
	if !reflect.DeepEqual(s.PricingBudget, pricing) {
		return errStrataMemoryPlan
	}
	if !validateStrataPlanQuery(s, request) {
		return errStrataMemoryPlan
	}
	totals := []uint64{max(v, e.MinimumPolicyVRAMFreeBytes), h, h + v}
	shorts := []uint64{s.NativeFit.ShortfallBytes.VRAM, s.NativeFit.ShortfallBytes.RAM, s.NativeFit.ShortfallBytes.Commit}
	anyShort := false
	for i, total := range totals {
		var short uint64
		if pricing != nil {
			budget := []uint64{pricing.Devices[0].Bytes, pricing.HostRAMBytes, pricing.HostCommitBytes}[i]
			short = total - min(total, budget)
		}
		if shorts[i] != short {
			return errStrataMemoryPlan
		}
		anyShort = anyShort || short > 0
	}
	status := "not_evaluated"
	if pricing != nil {
		status = "within_priced_budget"
		if anyShort {
			status = "does_not_fit"
		}
	}
	if s.NativeFit.Status != status || !r.ConfigurationComplete && status != "does_not_fit" {
		return errStrataMemoryPlan
	}
	// Validate identity shape even though the exact admitted closure is compared above.
	paths := map[string]bool{}
	for _, in := range s.Inputs {
		if in.Path == "" || len(in.Path) > 4096 || strings.ContainsAny(in.Path, "\x00\r\n") || paths[in.Path] || len(in.FileID) != 2 || in.MetadataBytes != nil && *in.MetadataBytes > in.SizeBytes || (in.MetadataBytes == nil) != (in.MetadataSHA256 == nil) {
			return errStrataMemoryPlan
		}
		paths[in.Path] = true
		for _, hash := range []*string{in.ContentSHA256, in.MetadataSHA256} {
			if hash != nil && !strataPlanDigest(*hash) {
				return errStrataMemoryPlan
			}
		}
	}
	return nil
}

func validateStrataPlanQuery(s strataMemoryPlanPrivate, r strataMemoryPlanRequest) bool {
	o := r.Extensions.Strata
	c := r.Config
	gib := func(n uint64) string {
		// Decimal division in Strata uses 28 significant digits, half-even.
		integer := n / (1 << 30)
		digits := len(strconv.FormatUint(integer, 10))
		if integer == 0 {
			digits = 0
			if n != 0 {
				for value := n; value < 1<<30; value *= 10 {
					digits--
				}
				digits++
			}
		}
		places := 28 - digits
		scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(places)), nil)
		numerator := new(big.Int).Mul(new(big.Int).SetUint64(n), scale)
		q, rem := new(big.Int), new(big.Int)
		q.QuoRem(numerator, big.NewInt(1<<30), rem)
		cmp := new(big.Int).Lsh(rem, 1).Cmp(big.NewInt(1 << 30))
		if cmp > 0 || cmp == 0 && q.Bit(0) == 1 {
			q.Add(q, big.NewInt(1))
		}
		v := new(big.Rat).SetFrac(q, scale).FloatString(places)
		return strings.TrimRight(strings.TrimRight(v, "0"), ".")
	}
	kv := c.KVType
	switch kv {
	case "", "q8_0":
		kv = "int8"
	case "f16":
		kv = "fp16"
	}
	choice := func(v strataMemoryPlanChoice) string {
		if v.Auto != "" {
			return v.Auto
		}
		return strataPlanNumber(v.Value)
	}
	micro := "auto"
	if c.MicrobatchSize != nil {
		micro = choice(*c.MicrobatchSize)
	}
	argv := []string{o.Executable, "--resolve-launch", "--price-only", "--pack", o.Pack, "--native", o.NativeModel, "--expert-profile", o.ExpertProfile, "--max-context", strataPlanNumber(c.Context), "--prefill", micro, "--kv", kv, "--spec", "2", "--kv-resident", strataPlanNumber(o.KVResidentCells), "--resident-budget-gib", gib(o.ResidentBudgetBytes), "--expert-cache", choice(o.CacheSlots), "--vram-reserve-mib", strataPlanNumber(o.VRAMReserveBytes / (1 << 20)), "--no-prefill-borrow"}
	if b := s.PricingBudget; b != nil {
		argv = append(argv, "--plan-vram-free-bytes", strataPlanNumber(b.Devices[0].Bytes), "--plan-ram-available-bytes", strataPlanNumber(b.HostRAMBytes), "--plan-commit-available-bytes", strataPlanNumber(b.HostCommitBytes))
	}
	env := map[string]string{}
	for k, v := range o.Environment {
		env[k] = v
	}
	env["CUDA_VISIBLE_DEVICES"] = strataPlanNumber(c.Devices[0].ID)
	env["STRATA_FILE_CACHE_MIB"] = strataPlanNumber(o.FileCacheBytes / (1 << 20))
	env["STRATA_RESIDENT_HEADROOM_GIB"] = gib(o.HostHeadroomBytes)
	return s.Query.WorkingDirectory == o.WorkingDirectory && slices.Equal(s.Query.Argv, argv) && reflect.DeepEqual(s.Query.Environment, env)
}
