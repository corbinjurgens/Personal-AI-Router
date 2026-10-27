// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package modelconfig defines portable preferences and target-owned readback.
// It performs no artifact I/O, native resolution, policy mutation or estimation.
package modelconfig

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"nvpair-shared/engines"
)

const MaxPayloadBytes = 64 * 1024
const MaxModels = 256

// MaxArtifacts matches the existing manager copy/resolution budget. Transport
// size is an additional limit, not permission to raise the settings budget.
const MaxArtifacts = 1024

type Context struct {
	Mode   string `json:"mode"`
	Tokens int32  `json:"tokens,omitempty"`
	Floor  int32  `json:"floor,omitempty"`
	Cap    int32  `json:"cap,omitempty"`
}
type GPU struct {
	Mode   string `json:"mode"`
	Layers *int32 `json:"layers,omitempty"`
}

// RequestedLoad contains only portable intent. Nil means inherited, including
// explicit false versus absent booleans. No Auto defaults are selected here.
type RequestedLoad struct {
	Context          *Context `json:"context,omitempty"`
	Slots            *int32   `json:"slots,omitempty"`
	GPU              *GPU     `json:"gpu,omitempty"`
	CacheTypeK       *string  `json:"cacheTypeK,omitempty"`
	CacheTypeV       *string  `json:"cacheTypeV,omitempty"`
	FlashAttention   *string  `json:"flashAttention,omitempty"`
	BatchSize        *int32   `json:"batchSize,omitempty"`
	UBatchSize       *int32   `json:"ubatchSize,omitempty"`
	Threads          *int32   `json:"threads,omitempty"`
	ThreadsBatch     *int32   `json:"threadsBatch,omitempty"`
	KVOffload        *bool    `json:"kvOffload,omitempty"`
	OpOffload        *bool    `json:"opOffload,omitempty"`
	SleepIdleSeconds *int32   `json:"sleepIdleSeconds,omitempty"`
	CacheRAM         *int32   `json:"cacheRAM,omitempty"` // MiB, not bytes
}

// ArtifactBinding is target-owned sampled local identity, not a verified full
// content digest. Path/ResolvedTarget correspond to servingFileBinding's source
// and symlink target; size is positive signed-int64 decimal text and MtimeNS is
// signed-int64 decimal text. Same-stat changes outside the samples are undetected.
// Role order for resolved load bindings is model shards followed by optional
// draft shards, in native order. Read-only ArtifactSummary also permits a
// companion phase after model and draft artifacts.
// The resolver must additionally prove completeness, regular files and confinement.
type ArtifactBinding struct {
	Role           string `json:"role"`
	Path           string `json:"path"`
	ResolvedTarget string `json:"resolvedTarget"`
	Size           string `json:"size"`
	MtimeNS        string `json:"mtimeNs"`
	SampleSHA256   string `json:"sampleSha256"`
}

// ResolvedLoad is produced only by target Preview/acceptance and persisted by
// the manager. It is never portable input. GPU must be CPU or finite layers;
// native-default and Auto must be resolved before acceptance. Options contains
// effective explicit choices, never estimates, telemetry or implicit defaults.
type ResolvedLoad struct {
	PerSlotContext    int32             `json:"perSlotContext"`
	TotalContext      int32             `json:"totalContext"`
	Slots             int32             `json:"slots"`
	KVUnified         *bool             `json:"kvUnified,omitempty"`
	GPU               GPU               `json:"gpu"`
	Options           RequestedLoad     `json:"options"`
	Artifacts         []ArtifactBinding `json:"artifacts"`
	NativeBuildID     string            `json:"nativeBuildId"`
	SourceInputDigest string            `json:"sourceInputDigest"`
	PresetDigest      string            `json:"presetDigest"`
	ResolverRevision  string            `json:"resolverRevision,omitempty"`
	ReviewedInputID   string            `json:"reviewedInputId,omitempty"`
}

// ModelLoad is storage/readback, not a caller patch. Full settings requests may
// echo it only as an unchanged target baseline. Consumers must compare against
// fresh target state and reject caller changes to this authority before effects.
type ModelLoad struct {
	ModelID   string        `json:"modelId"`
	Requested RequestedLoad `json:"requested"`
	Resolved  *ResolvedLoad `json:"resolved,omitempty"`
}

// LoadPatch is the public load-edit boundary. There is deliberately no resolved,
// artifacts, native build, device identity or telemetry input.
type LoadPatch struct {
	Set   RequestedLoad `json:"set"`
	Clear []string      `json:"clear,omitempty"`
}

// Generation selects existing numeric leaves: a number sets, null clears, and
// absence preserves. Roles selects weak/medium/strong membership of this exact
// engine/model: true assigns it (displacement requires preview); false removes
// only its own assignment, never a different model. No profile is copied.
// Served changes only selected membership, preserving nil/unrestricted versus
// explicit empty permission; residency still needs existing qualified bounds.
type PolicyPatch struct {
	Generation         map[string]*json.Number `json:"generation,omitempty"`
	RequestOptionsMode *string                 `json:"requestOptionsMode,omitempty"`
	Roles              map[string]bool         `json:"roles,omitempty"`
	Served             *bool                   `json:"served,omitempty"`
	Resident           *bool                   `json:"resident,omitempty"`
	Binding            *bool                   `json:"binding,omitempty"`
}
type ModelChange struct {
	ModelID                string       `json:"modelId"`
	ExpectedPolicyRevision string       `json:"expectedPolicyRevision"`
	Load                   *LoadPatch   `json:"load,omitempty"`
	Policy                 *PolicyPatch `json:"policy,omitempty"`
	PreviewDigest          string       `json:"previewDigest,omitempty"`
	SupersedesRequestID    string       `json:"supersedesRequestId,omitempty"`
}
type InspectRequest struct {
	Engine  string `json:"engine"`
	NodeID  string `json:"nodeId,omitempty"`
	ModelID string `json:"modelId"`
}

// ArtifactSummary is path-free. Identity is an opaque resolver binding identity,
// not a claim of fresh full hashing. Assurance names the actual source guarantee.
type ArtifactSummary struct {
	Role      string `json:"role"`
	Identity  string `json:"identity"`
	Size      string `json:"size"`
	Assurance string `json:"assurance"`
}
type Facts struct {
	Status          string `json:"status"`
	Source          string `json:"source,omitempty"`
	Architecture    string `json:"architecture,omitempty"`
	ContextTokens   *int32 `json:"contextTokens,omitempty"`
	LayerCount      *int32 `json:"layerCount,omitempty"`
	TemplatePresent *bool  `json:"templatePresent,omitempty"`
	TemplateDigest  string `json:"templateDigest,omitempty"`
	// Bounded numeric dimensions/counts only, never templates or token strings.
	Dimensions map[string]string `json:"dimensions,omitempty"`
	Unknowns   []string          `json:"unknowns,omitempty"`
}

// MemoryDomain identifies one independent pool. UMA is one shared pool, not
// host plus GPU. DeviceID is target-mapped identity, never an array ordinal.
// All byte fields are canonical uint64 decimal strings; absence means unknown.
type MemoryDomain struct {
	ID            string `json:"id"`
	Kind          string `json:"kind"` // host, device or shared
	DeviceID      string `json:"deviceId,omitempty"`
	CapacityBytes string `json:"capacityBytes,omitempty"`
	FreeBytes     string `json:"freeBytes,omitempty"`
	RequiredBytes string `json:"requiredBytes,omitempty"`
}
type EstimateComponent struct {
	Name     string `json:"name"`
	DomainID string `json:"domainId,omitempty"`
	Status   string `json:"status"`
	Bytes    string `json:"bytes,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

// Estimate is advisory, never allocation/admission/qualification authority.
type Estimate struct {
	Status          string              `json:"status"`
	Source          string              `json:"source,omitempty"`
	BuildID         string              `json:"buildId,omitempty"`
	Timestamp       string              `json:"timestamp,omitempty"`
	MemorySource    string              `json:"memorySource,omitempty"`
	MemoryTimestamp string              `json:"memoryTimestamp,omitempty"`
	HostBytes       string              `json:"hostBytes,omitempty"`
	DeviceBytes     string              `json:"deviceBytes,omitempty"`
	Domains         []MemoryDomain      `json:"domains,omitempty"`
	Components      []EstimateComponent `json:"components,omitempty"`
	Assumptions     []string            `json:"assumptions,omitempty"`
	Unknowns        []string            `json:"unknowns,omitempty"`
}
type InspectResult struct {
	Engine    string             `json:"engine"`
	NodeID    string             `json:"nodeId,omitempty"`
	ModelID   string             `json:"modelId"`
	Status    string             `json:"status"`
	Reason    string             `json:"reason,omitempty"`
	Source    string             `json:"source,omitempty"`
	Assurance string             `json:"assurance,omitempty"`
	Artifacts []ArtifactSummary  `json:"artifacts,omitempty"`
	Facts     *Facts             `json:"facts,omitempty"`
	Options   []OptionDescriptor `json:"options,omitempty"`
	Estimate  Estimate           `json:"estimate"`
}

// Impact represents unresolved owner choices without manufacturing a timeout,
// restart policy or false boolean. Nil means not yet determined. Policy-only
// previews use restartScope none. No defaults are installed by this package.
type Impact struct {
	RestartScope        string   `json:"restartScope"`
	AffectedModels      []string `json:"affectedModels,omitempty"`
	DrainTimeoutSeconds *int32   `json:"drainTimeoutSeconds,omitempty"`
	MayInterruptActive  *bool    `json:"mayInterruptActive,omitempty"`
	StaysOff            *bool    `json:"staysOff,omitempty"`
	ActiveWorkPolicy    string   `json:"activeWorkPolicy,omitempty"`
	Unknowns            []string `json:"unknowns,omitempty"`
}
type FieldDiff struct {
	Field  string          `json:"field"`
	Before json.RawMessage `json:"before,omitempty"`
	After  json.RawMessage `json:"after,omitempty"`
}
type AutoProposal struct {
	Status           string        `json:"status"`
	Requested        RequestedLoad `json:"requested"`
	Resolved         *ResolvedLoad `json:"resolved,omitempty"`
	Estimate         Estimate      `json:"estimate"`
	MemorySource     string        `json:"memorySource,omitempty"`
	MemoryTimestamp  string        `json:"memoryTimestamp,omitempty"`
	ResolverRevision string        `json:"resolverRevision,omitempty"`
	ReviewedInputID  string        `json:"reviewedInputId,omitempty"`
	Warnings         []string      `json:"warnings,omitempty"`
}

// DecodeStrict checks the entire envelope before any custom UnmarshalJSON can
// collapse duplicates or ignore fields. It checks exact field spelling, rejects
// unknown keys, null selected leaves (except explicit generation clears), invalid
// UTF-8, excessive nesting, trailing data and the complete transport byte budget.
// It is decoding, not semantic validation: use the exported typed decode hooks
// at transport boundaries. Decode into a fresh value and publish only on success.
func DecodeStrict(data []byte, dst any) error {
	if len(data) > MaxPayloadBytes {
		return fmt.Errorf("payload: exceeds 64 KiB")
	}
	if !utf8.Valid(data) {
		return fmt.Errorf("payload: invalid UTF-8")
	}
	t := reflect.TypeOf(dst)
	if t == nil || t.Kind() != reflect.Pointer || reflect.ValueOf(dst).IsNil() {
		return fmt.Errorf("payload: destination must be a nonnil pointer")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	if err := scanValue(d, 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("payload: trailing data")
	}
	if err := checkShape(data, t.Elem(), "payload"); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	d.UseNumber()
	return d.Decode(dst)
}

func scanValue(d *json.Decoder, depth int) error {
	if depth > 32 {
		return fmt.Errorf("payload: nesting exceeds limit")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			k, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := k.(string)
			folded := strings.ToLower(key)
			if !ok || seen[folded] {
				return fmt.Errorf("payload: duplicate or invalid key")
			}
			seen[folded] = true
			if err := scanValue(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := scanValue(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("payload: invalid delimiter")
	}
	_, err = d.Token()
	return err
}

// Reflection is limited to JSON field shape, including structs with custom
// decoders such as ServingConfig. No authority is inferred from Go field types.
func checkShape(data []byte, t reflect.Type, path string) error {
	data = bytes.TrimSpace(data)
	if bytes.Equal(data, []byte("null")) {
		// Null generation values are the sole selected-leaf clear encoding.
		if strings.HasPrefix(path, "payload.") && strings.Contains(path, ".generation.") && t == reflect.TypeOf((*json.Number)(nil)) {
			return nil
		}
		// Nil collections/readback pointers may occur in stored snapshots.
		if t.Kind() == reflect.Slice || t.Kind() == reflect.Map {
			return nil
		}
		return fmt.Errorf("%s: null is not a value; omit inherited fields", path)
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == reflect.TypeOf(json.RawMessage{}) {
		return nil
	}
	if t == reflect.TypeOf(json.Number("")) {
		if len(data) == 0 || !(data[0] == '-' || data[0] >= '0' && data[0] <= '9') || !json.Valid(data) {
			return fmt.Errorf("%s: expected JSON number", path)
		}
		return nil
	}
	switch t.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			return err
		}
		allowed := map[string]reflect.Type{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if f.PkgPath != "" { // unexported implementation detail
				continue
			}
			name := strings.Split(f.Tag.Get("json"), ",")[0]
			if name == "-" {
				continue
			}
			if name == "" {
				name = f.Name
			}
			allowed[name] = f.Type
		}
		for name, raw := range fields {
			field, ok := allowed[name]
			if !ok {
				return fmt.Errorf("%s: unknown field %q", path, name)
			}
			// encoding/json's ,string accepts noncanonical numeric spellings.
			// Revisions/sequences keep uint64 in Go and decimal strings on wire.
			for i := 0; i < t.NumField(); i++ {
				f := t.Field(i)
				tag := strings.Split(f.Tag.Get("json"), ",")
				if tag[0] == name && slices.Contains(tag[1:], "string") && f.Type.Kind() == reflect.Uint64 {
					var text string
					if json.Unmarshal(raw, &text) != nil || ValidateUnsignedDecimal(text) != nil {
						return fmt.Errorf("%s.%s: expected canonical uint64 decimal string", path, name)
					}
				}
			}
			if err := checkShape(raw, field, path+"."+name); err != nil {
				return err
			}
		}
	case reflect.Slice, reflect.Array:
		var entries []json.RawMessage
		if err := json.Unmarshal(data, &entries); err != nil {
			return err
		}
		for i, entry := range entries {
			if err := checkShape(entry, t.Elem(), fmt.Sprintf("%s.%d", path, i)); err != nil {
				return err
			}
		}
	case reflect.Map:
		var entries map[string]json.RawMessage
		if err := json.Unmarshal(data, &entries); err != nil {
			return err
		}
		for key, entry := range entries {
			if err := checkShape(entry, t.Elem(), path+"."+key); err != nil {
				return err
			}
		}
	}
	return nil
}

// CheckPayloadSize must be applied to the complete request/response, not only
// modelLoads. It also propagates malformed RawMessage/Number encoding errors.
func CheckPayloadSize(value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("payload: %w", err)
	}
	if len(data) > MaxPayloadBytes {
		return fmt.Errorf("payload: exceeds 64 KiB")
	}
	return nil
}

// ValidateRevision follows nodepolicy's canonical nonnegative signed-int64
// revision range. Engine uint64 revisions use ValidateUnsignedDecimal instead.
func ValidateRevision(s string) error {
	if err := ValidateUnsignedDecimal(s); err != nil {
		return fmt.Errorf("revision: %w", err)
	}
	if _, err := strconv.ParseInt(s, 10, 64); err != nil {
		return fmt.Errorf("revision: exceeds signed int64")
	}
	return nil
}
func ValidateUnsignedDecimal(s string) error {
	if s == "" || len(s) > 20 || len(s) > 1 && s[0] == '0' {
		return fmt.Errorf("expected canonical uint64 decimal string")
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return fmt.Errorf("expected canonical uint64 decimal string")
		}
	}
	if _, err := strconv.ParseUint(s, 10, 64); err != nil {
		return fmt.Errorf("decimal value exceeds uint64")
	}
	return nil
}
func validateModelID(id string) error {
	// Keep this exact rule aligned with nodepolicy.ValidateModelIdentifier and
	// checkModelRef (parity tested). Importing nodepolicy here would cycle its
	// same-package engine-invalidation tests through enginesettings.
	if id == "" || len(id) > 512 || !utf8.ValidString(id) {
		return fmt.Errorf("modelId: expected nonempty valid UTF-8 of at most 512 bytes")
	}
	for _, c := range id {
		if c < 0x20 || c >= 0x7f && c <= 0x9f {
			return fmt.Errorf("modelId: control character")
		}
	}
	tier := id
	// Strip only one suffix, as nodepolicy.ParseAlias does.
	if strings.HasSuffix(id, "-api") {
		tier = strings.TrimSuffix(id, "-api")
	} else if strings.HasSuffix(id, "-local") {
		tier = strings.TrimSuffix(id, "-local")
	}
	if isTier(tier) {
		return fmt.Errorf("modelId: reserved routing alias")
	}
	return nil
}
func isTier(s string) bool {
	return s == "weak" || s == "medium" || s == "strong"
}
func boundedText(s string, limit int, required bool) error {
	if required && s == "" || len(s) > limit || !utf8.ValidString(s) {
		return fmt.Errorf("invalid or oversized text")
	}
	for _, c := range s {
		if c < 0x20 || c >= 0x7f && c <= 0x9f {
			return fmt.Errorf("control character in text")
		}
	}
	return nil
}
func digest(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func DecodeModelChange(data []byte) (ModelChange, error) {
	var c ModelChange
	if err := DecodeStrict(data, &c); err != nil {
		return ModelChange{}, err
	}
	if err := c.Validate(); err != nil {
		return ModelChange{}, err
	}
	c = c.Clone()
	if c.Load != nil {
		sort.Strings(c.Load.Clear)
	}
	return c, nil
}
func DecodeInspectRequest(data []byte) (InspectRequest, error) {
	var r InspectRequest
	if err := DecodeStrict(data, &r); err != nil {
		return InspectRequest{}, err
	}
	if err := r.Validate(); err != nil {
		return InspectRequest{}, err
	}
	return r, nil
}
func (r InspectRequest) Validate() error {
	if !engines.ValidRoutingID(r.Engine) {
		return fmt.Errorf("engine: invalid routing ID")
	}
	if err := boundedText(r.NodeID, 256, false); err != nil {
		return fmt.Errorf("nodeId: %w", err)
	}
	return validateModelID(r.ModelID)
}
func (c ModelChange) Validate() error {
	if err := validateModelID(c.ModelID); err != nil {
		return fmt.Errorf("modelId: %w", err)
	}
	if err := ValidateRevision(c.ExpectedPolicyRevision); err != nil {
		return err
	}
	if c.Load == nil && c.Policy == nil {
		return fmt.Errorf("modelChange: select an edit")
	}
	if c.PreviewDigest != "" && !digest(c.PreviewDigest) {
		return fmt.Errorf("previewDigest: expected SHA256 hex")
	}
	if err := boundedText(c.SupersedesRequestID, 128, false); err != nil {
		return fmt.Errorf("supersedesRequestId: %w", err)
	}
	if c.Load != nil {
		if err := c.Load.Validate(); err != nil {
			return err
		}
	}
	if c.Policy != nil {
		if err := c.Policy.Validate(); err != nil {
			return err
		}
	}
	return CheckPayloadSize(c)
}
func (p LoadPatch) Validate() error {
	if err := p.Set.Validate(); err != nil {
		return err
	}
	fields := p.Set.selectedFields()
	seen := map[string]bool{}
	for _, key := range p.Clear {
		_, allowed := fields[key]
		if !allowed || seen[key] || fields[key] {
			return fmt.Errorf("load.clear: invalid, repeated or set field %q", key)
		}
		seen[key] = true
	}
	if len(p.Clear) == 0 {
		selected := false
		for _, set := range fields {
			selected = selected || set
		}
		if !selected {
			return fmt.Errorf("load: select an edit")
		}
	}
	return nil
}
func (p PolicyPatch) Validate() error {
	for k, v := range p.Generation {
		if err := validateGeneration(k, v); err != nil {
			return err
		}
	}
	for tier := range p.Roles {
		if !isTier(tier) {
			return fmt.Errorf("roles: unknown tier")
		}
	}
	if p.RequestOptionsMode != nil && *p.RequestOptionsMode != "defaults" && *p.RequestOptionsMode != "enforced" {
		return fmt.Errorf("requestOptionsMode: invalid value")
	}
	if len(p.Generation) == 0 && p.RequestOptionsMode == nil && len(p.Roles) == 0 && p.Served == nil && p.Resident == nil && p.Binding == nil {
		return fmt.Errorf("policy: select an edit")
	}
	// Enforced-empty and residency prerequisites require merging the selected
	// leaves with fresh target policy, then running nodepolicy validation.
	return CheckPayloadSize(p)
}

// CanonicalLoads validates storage/readback, clones all fields and sorts only
// exact model IDs. Artifact order is native semantic order, never sorted here.
func CanonicalLoads(loads []ModelLoad) ([]ModelLoad, error) {
	if len(loads) > MaxModels {
		return nil, fmt.Errorf("modelLoads: too many entries")
	}
	if err := CheckPayloadSize(loads); err != nil {
		return nil, err
	}
	out := CloneLoads(loads)
	sort.Slice(out, func(i, j int) bool { return out[i].ModelID < out[j].ModelID })
	for i, m := range out {
		if err := m.Validate(); err != nil {
			return nil, fmt.Errorf("modelLoads.%d: %w", i, err)
		}
		if i > 0 && out[i-1].ModelID == m.ModelID {
			return nil, fmt.Errorf("modelLoads.%d.modelId: duplicate", i)
		}
	}
	return out, nil
}

// DecodeLoads reads target-owned persisted/readback entries, not a public edit.
func DecodeLoads(data []byte) ([]ModelLoad, error) {
	var loads []ModelLoad
	if err := DecodeStrict(data, &loads); err != nil {
		return nil, err
	}
	return CanonicalLoads(loads)
}

// CloneLoads uses typed copies so invalid values cannot silently disappear via
// an ignored marshal/unmarshal error. Nil and explicit empty slices survive.
func CloneLoads(loads []ModelLoad) []ModelLoad {
	out := slices.Clone(loads)
	for i := range out {
		out[i].Requested = out[i].Requested.Clone()
		if out[i].Resolved != nil {
			r := out[i].Resolved.Clone()
			out[i].Resolved = &r
		}
	}
	return out
}
func (r ResolvedLoad) Clone() ResolvedLoad {
	r.KVUnified = clonePointer(r.KVUnified)
	r.GPU.Layers = clonePointer(r.GPU.Layers)
	r.Options = r.Options.Clone()
	r.Artifacts = slices.Clone(r.Artifacts)
	return r
}
func clonePointer[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}
func EqualLoads(a, b []ModelLoad) bool {
	if len(a) != len(b) {
		return false
	}
	a, b = CloneLoads(a), CloneLoads(b)
	sort.Slice(a, func(i, j int) bool { return a[i].ModelID < a[j].ModelID })
	sort.Slice(b, func(i, j int) bool { return b[i].ModelID < b[j].ModelID })
	for i := range a {
		if !reflect.DeepEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}

func (p PolicyPatch) Clone() PolicyPatch {
	p.Generation = maps.Clone(p.Generation)
	for key, value := range p.Generation {
		p.Generation[key] = clonePointer(value)
	}
	p.RequestOptionsMode = clonePointer(p.RequestOptionsMode)
	p.Roles = maps.Clone(p.Roles)
	p.Served, p.Resident, p.Binding = clonePointer(p.Served), clonePointer(p.Resident), clonePointer(p.Binding)
	return p
}
func (c ModelChange) Clone() ModelChange {
	if c.Load != nil {
		p := LoadPatch{Set: c.Load.Set.Clone(), Clear: slices.Clone(c.Load.Clear)}
		c.Load = &p
	}
	if c.Policy != nil {
		p := c.Policy.Clone()
		c.Policy = &p
	}
	return c
}
func (e Estimate) Clone() Estimate {
	e.Domains = slices.Clone(e.Domains)
	e.Components = slices.Clone(e.Components)
	e.Assumptions = slices.Clone(e.Assumptions)
	e.Unknowns = slices.Clone(e.Unknowns)
	return e
}
func (i Impact) Clone() Impact {
	i.AffectedModels = slices.Clone(i.AffectedModels)
	i.DrainTimeoutSeconds = clonePointer(i.DrainTimeoutSeconds)
	i.MayInterruptActive = clonePointer(i.MayInterruptActive)
	i.StaysOff = clonePointer(i.StaysOff)
	i.Unknowns = slices.Clone(i.Unknowns)
	return i
}
func (a AutoProposal) Clone() AutoProposal {
	a.Requested = a.Requested.Clone()
	if a.Resolved != nil {
		r := a.Resolved.Clone()
		a.Resolved = &r
	}
	a.Estimate = a.Estimate.Clone()
	a.Warnings = slices.Clone(a.Warnings)
	return a
}
func (r InspectResult) Clone() InspectResult {
	r.Artifacts = slices.Clone(r.Artifacts)
	if r.Facts != nil {
		f := *r.Facts
		f.ContextTokens = clonePointer(f.ContextTokens)
		f.LayerCount = clonePointer(f.LayerCount)
		f.TemplatePresent = clonePointer(f.TemplatePresent)
		f.Dimensions = maps.Clone(f.Dimensions)
		f.Unknowns = slices.Clone(f.Unknowns)
		r.Facts = &f
	}
	r.Options = slices.Clone(r.Options)
	for i := range r.Options {
		r.Options[i] = r.Options[i].Clone()
	}
	r.Estimate = r.Estimate.Clone()
	return r
}

func DecodeInspectResult(data []byte) (InspectResult, error) {
	var r InspectResult
	if err := DecodeStrict(data, &r); err != nil {
		return InspectResult{}, err
	}
	if err := r.Validate(); err != nil {
		return InspectResult{}, err
	}
	return r, nil
}
func (r InspectResult) Validate() error {
	if err := (InspectRequest{Engine: r.Engine, NodeID: r.NodeID, ModelID: r.ModelID}).Validate(); err != nil {
		return err
	}
	if !slices.Contains([]string{"resolved", "missing", "ambiguous", "busy", "unknown", "unavailable"}, r.Status) {
		return fmt.Errorf("inspect.status: invalid value")
	}
	if len(r.Artifacts) > MaxArtifacts || r.Status == "resolved" && len(r.Artifacts) == 0 {
		return fmt.Errorf("inspect.artifacts: bounded resolved artifacts required")
	}
	phase := 0
	for _, a := range r.Artifacts {
		switch a.Role {
		case "model":
			if phase != 0 {
				return fmt.Errorf("inspect.artifacts: invalid role order")
			}
		case "draft":
			if phase > 1 {
				return fmt.Errorf("inspect.artifacts: invalid role order")
			}
			phase = 1
		case "companion":
			phase = 2
		default:
			return fmt.Errorf("inspect.artifacts: invalid role order")
		}
		if err := ValidateRevision(a.Size); err != nil || a.Size == "0" {
			return fmt.Errorf("inspect.artifacts.size: expected positive int64 decimal string")
		}
		if boundedText(a.Identity, 256, true) != nil || boundedText(a.Assurance, 256, true) != nil {
			return fmt.Errorf("inspect.artifacts: identity and assurance required")
		}
	}
	if len(r.Artifacts) > 0 && r.Artifacts[0].Role != "model" {
		return fmt.Errorf("inspect.artifacts: model shards required first")
	}
	if r.Facts != nil {
		f := r.Facts
		if len(f.Dimensions) > 256 || f.ContextTokens != nil && *f.ContextTokens <= 0 || f.LayerCount != nil && *f.LayerCount < 0 {
			return fmt.Errorf("facts: invalid bounded dimensions")
		}
		for key, value := range f.Dimensions {
			if boundedText(key, 256, true) != nil || ValidateUnsignedDecimal(value) != nil {
				return fmt.Errorf("facts.dimensions: expected bounded names and uint64 decimal values")
			}
		}
		if f.TemplateDigest != "" && !digest(f.TemplateDigest) {
			return fmt.Errorf("facts.templateDigest: expected SHA256 hex")
		}
	}
	if len(r.Options) > 256 {
		return fmt.Errorf("inspect.options: too many descriptors")
	}
	seen := map[string]bool{}
	for _, o := range r.Options {
		if seen[o.Field] {
			return fmt.Errorf("inspect.options: duplicate descriptor")
		}
		seen[o.Field] = true
		if err := o.Validate(); err != nil {
			return err
		}
	}
	if err := r.Estimate.Validate(); err != nil {
		return err
	}
	return CheckPayloadSize(r)
}

func (e Estimate) Validate() error {
	if !slices.Contains([]string{"estimated", "partial", "unknown", "unavailable"}, e.Status) {
		return fmt.Errorf("estimate.status: invalid advisory state")
	}
	for _, value := range []string{e.HostBytes, e.DeviceBytes} {
		if value != "" && ValidateUnsignedDecimal(value) != nil {
			return fmt.Errorf("estimate: expected canonical byte strings")
		}
	}
	if len(e.Domains) > 64 || len(e.Components) > 256 || len(e.Unknowns) > 256 || len(e.Assumptions) > 256 {
		return fmt.Errorf("estimate: collection exceeds budget")
	}
	seen := map[string]bool{}
	for _, d := range e.Domains {
		if boundedText(d.ID, 256, true) != nil || seen[d.ID] || !slices.Contains([]string{"host", "device", "shared"}, d.Kind) {
			return fmt.Errorf("estimate.domains: invalid or duplicate domain")
		}
		seen[d.ID] = true
		for _, value := range []string{d.CapacityBytes, d.FreeBytes, d.RequiredBytes} {
			if value != "" && ValidateUnsignedDecimal(value) != nil {
				return fmt.Errorf("estimate.domains: expected canonical byte strings")
			}
		}
	}
	for _, c := range e.Components {
		if boundedText(c.Name, 256, true) != nil || !slices.Contains([]string{"estimated", "partial", "unknown", "excluded"}, c.Status) || c.DomainID != "" && !seen[c.DomainID] {
			return fmt.Errorf("estimate.components: invalid component or domain")
		}
		if c.Bytes != "" && ValidateUnsignedDecimal(c.Bytes) != nil {
			return fmt.Errorf("estimate.components: expected canonical byte string")
		}
	}
	return CheckPayloadSize(e)
}
