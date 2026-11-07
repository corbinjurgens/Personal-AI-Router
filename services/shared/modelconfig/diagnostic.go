// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package modelconfig

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"
	"time"

	"nvpair-shared/nodepolicy"
)

type DiagnosticPreviewRequest struct {
	Engine     string `json:"engine"`
	NodeID     string `json:"nodeId,omitempty"`
	ModelID    string `json:"modelId"`
	ProfileKey string `json:"profileKey"`
}

func (r DiagnosticPreviewRequest) Clone() DiagnosticPreviewRequest { return r }
func (r DiagnosticPreviewRequest) Validate() error {
	if err := (InspectRequest{Engine: r.Engine, NodeID: r.NodeID, ModelID: r.ModelID, ProfileKey: r.ProfileKey}).Validate(); err != nil {
		return err
	}
	if err := nodepolicy.ValidateProfileKey(r.ProfileKey); err != nil {
		return err
	}
	return diagnosticText(r.Engine, r.NodeID, r.ModelID, r.ProfileKey)
}

type DiagnosticModelRef struct {
	Engine     string `json:"engine"`
	ModelID    string `json:"modelId"`
	ProfileKey string `json:"profileKey"`
}

func (r DiagnosticModelRef) Clone() DiagnosticModelRef { return r }
func (r DiagnosticModelRef) Validate() error {
	if r.Engine == "" || r.ModelID == "" {
		return fmt.Errorf("diagnostic: model identity required")
	}
	if err := nodepolicy.ValidateProfileKey(r.ProfileKey); err != nil {
		return err
	}
	return diagnosticText(r.Engine, r.ModelID, r.ProfileKey)
}

// DiagnosticIdentity binds observed hardware, artifacts, build and realized
// placement. GPUs includes every usable GPU, including unused devices.
type DiagnosticIdentity struct {
	ConfigurationKey       string                `json:"configurationKey,omitempty"`
	ExecutionKey           string                `json:"executionKey,omitempty"`
	ConfigurationKeyReason string                `json:"configurationKeyReason,omitempty"`
	NativeBuildID          string                `json:"nativeBuildId"`
	BuildInfo              string                `json:"buildInfo"`
	CPUModel               string                `json:"cpuModel"`
	TotalRAMBytes          string                `json:"totalRamBytes"`
	GPUs                   []DiagnosticGPU       `json:"gpus"`
	Artifacts              []ArtifactSummary     `json:"artifacts"`
	Requested              RequestedLoad         `json:"requested"`
	PerSlotContext         int32                 `json:"perSlotContext"`
	Slots                  int32                 `json:"slots"`
	Placement              []DiagnosticPlacement `json:"placement"`
}
type DiagnosticGPU struct {
	UUID             string `json:"uuid"`
	Name             string `json:"name"`
	TotalMemoryBytes string `json:"totalMemoryBytes"`
}
type DiagnosticPlacement struct {
	DomainID string `json:"domainId"`
	Kind     string `json:"kind"`
	DeviceID string `json:"deviceId,omitempty"`
}

func (i DiagnosticIdentity) Clone() DiagnosticIdentity {
	i.GPUs = slices.Clone(i.GPUs)
	i.Artifacts = slices.Clone(i.Artifacts)
	i.Placement = slices.Clone(i.Placement)
	i.Requested = i.Requested.Clone()
	return i
}
func (i DiagnosticIdentity) Validate() error {
	return i.validate(true)
}

// ValidateRequested checks the consent binding before the child has run.
// Runtime allocation and build claims can be absent until the authorized load.
func (i DiagnosticIdentity) ValidateRequested() error {
	return i.validate(false)
}

func (i DiagnosticIdentity) validate(realized bool) error {
	if (i.ConfigurationKey == "") == (i.ExecutionKey == "") {
		return fmt.Errorf("diagnostic: exactly one configuration or execution key required")
	}
	if i.ConfigurationKey != "" && (!digest(i.ConfigurationKey) || i.ConfigurationKeyReason != "") {
		return fmt.Errorf("diagnostic: invalid configuration key")
	}
	if i.ExecutionKey != "" && (!digest(i.ExecutionKey) || (i.ConfigurationKeyReason != "native-fit" && i.ConfigurationKeyReason != "gpu-selection")) {
		return fmt.Errorf("diagnostic: invalid execution key")
	}
	if len(i.NativeBuildID) != 71 || i.NativeBuildID[:7] != "sha256:" || !digest(i.NativeBuildID[7:]) {
		return fmt.Errorf("diagnostic: native build identity required")
	}
	if i.CPUModel == "" || i.PerSlotContext < 0 || i.Slots < 0 || i.Slots > 8 || len(i.GPUs) > 64 || len(i.Artifacts) == 0 || len(i.Artifacts) > 64 || len(i.Placement) > 64 {
		return fmt.Errorf("diagnostic: incomplete or oversized identity")
	}
	if realized && (i.PerSlotContext < 208 || i.Slots < 1 || len(i.Placement) == 0) {
		return fmt.Errorf("diagnostic: realized allocation required")
	}
	if err := diagnosticText(i.BuildInfo, i.CPUModel); err != nil {
		return err
	}
	if err := ValidateUnsignedDecimal(i.TotalRAMBytes); err != nil {
		return err
	}
	seen := map[string]bool{}
	for _, g := range i.GPUs {
		if g.UUID == "" || g.Name == "" || seen[g.UUID] {
			return fmt.Errorf("diagnostic: invalid GPU identity")
		}
		seen[g.UUID] = true
		if err := diagnosticText(g.UUID, g.Name); err != nil {
			return err
		}
		if err := ValidateUnsignedDecimal(g.TotalMemoryBytes); err != nil {
			return err
		}
	}
	for _, a := range i.Artifacts {
		if a.Identity == "" || a.Role == "" || a.Assurance == "" {
			return fmt.Errorf("diagnostic: artifact identity required")
		}
		if err := diagnosticText(a.Identity, a.Role, a.Assurance); err != nil {
			return err
		}
		if err := ValidateUnsignedDecimal(a.Size); err != nil {
			return err
		}
	}
	domains := map[string]bool{}
	for _, p := range i.Placement {
		if p.DomainID == "" || domains[p.DomainID] || !slices.Contains([]string{"host", "device", "shared"}, p.Kind) || (p.Kind == "device" && !seen[p.DeviceID]) || (p.Kind != "device" && p.DeviceID != "") {
			return fmt.Errorf("diagnostic: invalid placement")
		}
		domains[p.DomainID] = true
		if err := diagnosticText(p.DomainID, p.DeviceID); err != nil {
			return err
		}
	}
	if err := i.Requested.Validate(); err != nil {
		return err
	}
	return CheckPayloadSize(i)
}

type DiagnosticPreview struct {
	DiagnosticPreviewRequest
	DiagnosticIdentity
	PreviewID             string               `json:"previewId"`
	PlannedPerSlotContext int32                `json:"plannedPerSlotContext"`
	PlannedSlots          int32                `json:"plannedSlots"`
	LowPromptTokens       int32                `json:"lowPromptTokens"`
	HighPromptTokens      int32                `json:"highPromptTokens"`
	GenerateTokens        int32                `json:"generateTokens"`
	TimeLimitSeconds      int32                `json:"timeLimitSeconds"`
	UnloadModels          []DiagnosticModelRef `json:"unloadModels"`
	ReloadedAfter         []DiagnosticModelRef `json:"reloadedAfter"`
	DrainEngines          []string             `json:"drainEngines"`
	StartsEngine          bool                 `json:"startsEngine"`
	Warnings              []string             `json:"warnings"`
	ExpiresAt             string               `json:"expiresAt"`
}

func (p DiagnosticPreview) Clone() DiagnosticPreview {
	p.DiagnosticIdentity = p.DiagnosticIdentity.Clone()
	p.UnloadModels = slices.Clone(p.UnloadModels)
	p.ReloadedAfter = slices.Clone(p.ReloadedAfter)
	p.DrainEngines = slices.Clone(p.DrainEngines)
	p.Warnings = slices.Clone(p.Warnings)
	return p
}
func (p DiagnosticPreview) Validate() error {
	if err := p.DiagnosticPreviewRequest.Validate(); err != nil {
		return err
	}
	if err := p.DiagnosticIdentity.ValidateRequested(); err != nil {
		return err
	}
	knownTargets := p.PlannedPerSlotContext >= 208 && p.PlannedSlots >= 1 && p.HighPromptTokens == p.PlannedPerSlotContext-144 && p.LowPromptTokens == min(int32(512), p.HighPromptTokens/2)
	unknownTargets := p.PlannedPerSlotContext == 0 && p.LowPromptTokens == 0 && p.HighPromptTokens == 0
	if p.PlannedSlots < 0 || p.PlannedSlots > 8 {
		return fmt.Errorf("diagnostic: unsupported planned slots")
	}
	if p.PreviewID == "" || p.GenerateTokens != 128 || (!knownTargets && !unknownTargets) || p.TimeLimitSeconds <= 0 || p.TimeLimitSeconds > 1200 {
		return fmt.Errorf("diagnostic: invalid preview targets")
	}
	if err := diagnosticTimestamp(p.ExpiresAt); err != nil {
		return err
	}
	if err := diagnosticText(p.PreviewID); err != nil {
		return err
	}
	for _, list := range [][]DiagnosticModelRef{p.UnloadModels, p.ReloadedAfter} {
		if len(list) > 64 {
			return fmt.Errorf("diagnostic: too many models")
		}
		for _, m := range list {
			if err := m.Validate(); err != nil {
				return err
			}
		}
	}
	if err := diagnosticList(p.DrainEngines); err != nil {
		return err
	}
	if err := diagnosticList(p.Warnings); err != nil {
		return err
	}
	return CheckPayloadSize(p)
}

type DiagnosticStartRequest struct {
	DiagnosticPreviewRequest
	PreviewID string `json:"previewId"`
	Consent   bool   `json:"consent"`
}

func (r DiagnosticStartRequest) Clone() DiagnosticStartRequest { return r }
func (r DiagnosticStartRequest) Validate() error {
	if err := r.DiagnosticPreviewRequest.Validate(); err != nil {
		return err
	}
	if !r.Consent || r.PreviewID == "" {
		return fmt.Errorf("diagnostic: consent and preview required")
	}
	return diagnosticText(r.PreviewID)
}

type DiagnosticRunRef struct {
	NodeID string `json:"nodeId,omitempty"`
	RunID  string `json:"runId"`
}

func (r DiagnosticRunRef) Clone() DiagnosticRunRef { return r }
func (r DiagnosticRunRef) Validate() error {
	if r.RunID == "" {
		return fmt.Errorf("diagnostic: run ID required")
	}
	return diagnosticText(r.NodeID, r.RunID)
}

type DiagnosticRun struct {
	RunID            string           `json:"runId"`
	Engine           string           `json:"engine"`
	ModelID          string           `json:"modelId"`
	ProfileKey       string           `json:"profileKey"`
	ConfigurationKey string           `json:"configurationKey,omitempty"`
	ExecutionKey     string           `json:"executionKey,omitempty"`
	Phase            string           `json:"phase"`
	Outcome          string           `json:"outcome"`
	FailedPhase      string           `json:"failedPhase,omitempty"`
	Reason           string           `json:"reason,omitempty"`
	StartedAt        string           `json:"startedAt"`
	ElapsedMS        int64            `json:"elapsedMs"`
	Result           *MeasuredProfile `json:"result,omitempty"`
}

func (r DiagnosticRun) Clone() DiagnosticRun {
	if r.Result != nil {
		v := r.Result.Clone()
		r.Result = &v
	}
	return r
}
func (r DiagnosticRun) Validate() error {
	if err := (DiagnosticPreviewRequest{Engine: r.Engine, ModelID: r.ModelID, ProfileKey: r.ProfileKey}).Validate(); err != nil {
		return err
	}
	if r.RunID == "" || !diagnosticPhase(r.Phase) || !diagnosticOutcome(r.Outcome) || (r.FailedPhase != "" && !diagnosticPhase(r.FailedPhase)) || r.ElapsedMS < 0 || (r.ConfigurationKey == "") == (r.ExecutionKey == "") || (r.ConfigurationKey != "" && !digest(r.ConfigurationKey)) || (r.ExecutionKey != "" && !digest(r.ExecutionKey)) {
		return fmt.Errorf("diagnostic: invalid run")
	}
	if err := diagnosticTimestamp(r.StartedAt); err != nil {
		return err
	}
	if err := diagnosticText(r.RunID, r.Reason); err != nil {
		return err
	}
	if r.Result != nil {
		if err := r.Result.Validate(); err != nil {
			return err
		}
		if r.Result.RunID != r.RunID || r.Result.ConfigurationKey != r.ConfigurationKey || r.Result.ExecutionKey != r.ExecutionKey || r.Result.Outcome != r.Outcome {
			return fmt.Errorf("diagnostic: result identity mismatch")
		}
	}
	return CheckPayloadSize(r)
}

type MeasuredProfile struct {
	DiagnosticIdentity
	RunID          string         `json:"runId"`
	RecordedAt     string         `json:"recordedAt"`
	Outcome        string         `json:"outcome"`
	FailedPhase    string         `json:"failedPhase,omitempty"`
	Reason         string         `json:"reason,omitempty"`
	BuildPin       string         `json:"buildPin"`
	Pools          []MeasuredPool `json:"pools"`
	Low            *MeasuredPhase `json:"low,omitempty"`
	High           *MeasuredPhase `json:"high,omitempty"`
	BufferSampleID string         `json:"bufferSampleId,omitempty"`
	Notes          []string       `json:"notes"`
}
type MeasuredPool struct {
	DomainID                string `json:"domainId"`
	Kind                    string `json:"kind"`
	DeviceID                string `json:"deviceId,omitempty"`
	BaselineAvailableBytes  string `json:"baselineAvailableBytes,omitempty"`
	AfterLoadAvailableBytes string `json:"afterLoadAvailableBytes,omitempty"`
	LowMinAvailableBytes    string `json:"lowMinAvailableBytes,omitempty"`
	HighMinAvailableBytes   string `json:"highMinAvailableBytes,omitempty"`
	PeakUsedBytes           string `json:"peakUsedBytes,omitempty"`
	Source                  string `json:"source"`
	Samples                 int32  `json:"samples"`
	IntervalMS              int64  `json:"intervalMs"`
}
type MeasuredPhase struct {
	Coverage                   string            `json:"coverage"`
	Requests                   []MeasuredRequest `json:"requests"`
	Overlapped                 *bool             `json:"overlapped,omitempty"`
	OverlapWindowSampled       *bool             `json:"overlapWindowSampled,omitempty"`
	CombinedGeneratedPerSecond *float64          `json:"combinedGeneratedPerSecond,omitempty"`
	WallMS                     int64             `json:"wallMs"`
}
type MeasuredRequest struct {
	TargetPromptTokens int32    `json:"targetPromptTokens"`
	PromptTokens       *int32   `json:"promptTokens,omitempty"`
	CachedTokens       *int32   `json:"cachedTokens,omitempty"`
	PromptMS           *float64 `json:"promptMs,omitempty"`
	PromptPerSecond    *float64 `json:"promptPerSecond,omitempty"`
	GeneratedTokens    *int32   `json:"generatedTokens,omitempty"`
	DecodeMS           *float64 `json:"decodeMs,omitempty"`
	DecodePerSecond    *float64 `json:"decodePerSecond,omitempty"`
	FirstTokenMS       *float64 `json:"firstTokenMs,omitempty"`
	EndMS              *float64 `json:"endMs,omitempty"`
}

func (r MeasuredRequest) Clone() MeasuredRequest {
	r.PromptTokens = clonePointer(r.PromptTokens)
	r.CachedTokens = clonePointer(r.CachedTokens)
	r.GeneratedTokens = clonePointer(r.GeneratedTokens)
	r.PromptMS = clonePointer(r.PromptMS)
	r.PromptPerSecond = clonePointer(r.PromptPerSecond)
	r.DecodeMS = clonePointer(r.DecodeMS)
	r.DecodePerSecond = clonePointer(r.DecodePerSecond)
	r.FirstTokenMS = clonePointer(r.FirstTokenMS)
	r.EndMS = clonePointer(r.EndMS)
	return r
}
func (r MeasuredRequest) Validate() error {
	if r.TargetPromptTokens <= 0 {
		return fmt.Errorf("diagnostic: invalid prompt target")
	}
	for _, v := range []*int32{r.PromptTokens, r.CachedTokens, r.GeneratedTokens} {
		if v != nil && *v < 0 {
			return fmt.Errorf("diagnostic: negative token count")
		}
	}
	for _, v := range []*float64{r.PromptMS, r.PromptPerSecond, r.DecodeMS, r.DecodePerSecond, r.FirstTokenMS, r.EndMS} {
		if v != nil && (math.IsNaN(*v) || math.IsInf(*v, 0) || *v < 0) {
			return fmt.Errorf("diagnostic: invalid numeric observation")
		}
	}
	if r.FirstTokenMS != nil && r.EndMS != nil && *r.EndMS < *r.FirstTokenMS {
		return fmt.Errorf("diagnostic: reversed request interval")
	}
	return nil
}
func (p MeasuredPhase) Clone() MeasuredPhase {
	p.Requests = slices.Clone(p.Requests)
	for i := range p.Requests {
		p.Requests[i] = p.Requests[i].Clone()
	}
	p.Overlapped = clonePointer(p.Overlapped)
	p.OverlapWindowSampled = clonePointer(p.OverlapWindowSampled)
	p.CombinedGeneratedPerSecond = clonePointer(p.CombinedGeneratedPerSecond)
	return p
}
func (p MeasuredPhase) Validate() error {
	if !slices.Contains([]string{"complete", "partial"}, p.Coverage) || len(p.Requests) == 0 || len(p.Requests) > 8 || p.WallMS < 0 {
		return fmt.Errorf("diagnostic: invalid phase")
	}
	if v := p.CombinedGeneratedPerSecond; v != nil && (math.IsNaN(*v) || math.IsInf(*v, 0) || *v < 0) {
		return fmt.Errorf("diagnostic: invalid throughput")
	}
	for _, r := range p.Requests {
		if err := r.Validate(); err != nil {
			return err
		}
		if p.Coverage == "complete" && (r.PromptTokens == nil || int64(*r.PromptTokens)*100 < int64(r.TargetPromptTokens)*95 || r.CachedTokens == nil || *r.CachedTokens != 0) {
			return fmt.Errorf("diagnostic: incomplete prompt coverage")
		}
	}
	return nil
}
func (p MeasuredPool) Clone() MeasuredPool { return p }
func (p MeasuredPool) Validate() error {
	if p.DomainID == "" || p.Source == "" || !slices.Contains([]string{"host", "device", "shared"}, p.Kind) || (p.Kind == "device") != (p.DeviceID != "") || p.Samples < 0 || p.IntervalMS <= 0 {
		return fmt.Errorf("diagnostic: invalid sampled pool")
	}
	if err := diagnosticText(p.DomainID, p.DeviceID, p.Source); err != nil {
		return err
	}
	for _, v := range []string{p.BaselineAvailableBytes, p.AfterLoadAvailableBytes, p.LowMinAvailableBytes, p.HighMinAvailableBytes, p.PeakUsedBytes} {
		if v != "" {
			if err := ValidateUnsignedDecimal(v); err != nil {
				return err
			}
		}
	}
	return nil
}
func (p MeasuredProfile) Clone() MeasuredProfile {
	p.DiagnosticIdentity = p.DiagnosticIdentity.Clone()
	p.Pools = slices.Clone(p.Pools)
	p.Notes = slices.Clone(p.Notes)
	if p.Low != nil {
		v := p.Low.Clone()
		p.Low = &v
	}
	if p.High != nil {
		v := p.High.Clone()
		p.High = &v
	}
	return p
}
func (p MeasuredProfile) Validate() error {
	if err := p.DiagnosticIdentity.validate(p.Outcome == "completed"); err != nil {
		return err
	}
	if p.RunID == "" || !diagnosticOutcome(p.Outcome) || p.Outcome == "running" || (p.FailedPhase != "" && !diagnosticPhase(p.FailedPhase)) || len(p.Pools) > 64 {
		return fmt.Errorf("diagnostic: invalid measured result")
	}
	if err := diagnosticTimestamp(p.RecordedAt); err != nil {
		return err
	}
	if err := diagnosticText(p.RunID, p.Reason, p.BuildPin, p.BufferSampleID); err != nil {
		return err
	}
	if err := diagnosticList(p.Notes); err != nil {
		return err
	}
	seen := map[string]bool{}
	for _, pool := range p.Pools {
		if err := pool.Validate(); err != nil {
			return err
		}
		if seen[pool.DomainID] || !slices.Contains(p.Placement, DiagnosticPlacement{pool.DomainID, pool.Kind, pool.DeviceID}) {
			return fmt.Errorf("diagnostic: pool placement mismatch")
		}
		seen[pool.DomainID] = true
	}
	for _, phase := range []*MeasuredPhase{p.Low, p.High} {
		if phase != nil {
			if err := phase.Validate(); err != nil {
				return err
			}
		}
	}
	if p.ConfigurationKeyReason == "gpu-selection" && p.BufferSampleID != "" {
		return fmt.Errorf("diagnostic: GPU selection has no buffer sample")
	}
	return CheckPayloadSize(p)
}

// Matches requires the complete identity, never just a caller-supplied Current flag.
func (p MeasuredProfile) Matches(current DiagnosticIdentity) bool {
	if p.Validate() != nil || current.Validate() != nil {
		return false
	}
	before := p.DiagnosticIdentity.Clone()
	after := current.Clone()
	byUUID := func(a, b DiagnosticGPU) int { return strings.Compare(a.UUID, b.UUID) }
	slices.SortFunc(before.GPUs, byUUID)
	slices.SortFunc(after.GPUs, byUUID)
	return reflect.DeepEqual(before, after)
}
func (p MeasuredProfile) Covers(pools []string) bool {
	if len(pools) == 0 {
		return false
	}
	for _, kind := range pools {
		found := false
		for _, pool := range p.Pools {
			if pool.Kind == kind && pool.PeakUsedBytes != "" {
				found = true
			}
		}
		if !found {
			return false
		}
	}
	for _, placement := range p.Placement {
		found := false
		for _, pool := range p.Pools {
			if pool.DomainID == placement.DomainID && pool.PeakUsedBytes != "" {
				found = true
			}
		}
		if !found {
			return false
		}
	}
	return true
}
func (p MeasuredProfile) OverlayEligible(current DiagnosticIdentity, pools []string) bool {
	for _, pool := range p.Pools {
		if pool.BaselineAvailableBytes == "" || pool.AfterLoadAvailableBytes == "" || pool.LowMinAvailableBytes == "" || pool.HighMinAvailableBytes == "" || pool.Samples == 0 {
			return false
		}
	}
	if p.High != nil {
		latestFirst, earliestEnd := float64(0), math.Inf(1)
		for _, r := range p.High.Requests {
			if r.TargetPromptTokens != p.PerSlotContext-144 || r.FirstTokenMS == nil || r.EndMS == nil || r.GeneratedTokens == nil || *r.GeneratedTokens < 1 {
				return false
			}
			latestFirst = max(latestFirst, *r.FirstTokenMS)
			earliestEnd = min(earliestEnd, *r.EndMS)
		}
		if latestFirst >= earliestEnd {
			return false
		}
	}
	return p.ExecutionKey == "" && p.Matches(current) && p.Outcome == "completed" && p.Covers(pools) && p.High != nil && p.High.Coverage == "complete" && len(p.High.Requests) == int(p.Slots) && p.High.Overlapped != nil && *p.High.Overlapped && p.High.OverlapWindowSampled != nil && *p.High.OverlapWindowSampled
}
func (p MeasuredProfile) AutoEligible(current DiagnosticIdentity, pools []string) bool {
	return p.OverlayEligible(current, pools)
}
func (p MeasuredProfile) BuildLabel(currentBuildID string) string {
	if p.NativeBuildID != currentBuildID {
		return "measured on " + p.NativeBuildID
	}
	return "Measured on this machine"
}
func diagnosticText(values ...string) error {
	for _, v := range values {
		if err := boundedText(v, 256, false); err != nil {
			return err
		}
	}
	return nil
}
func diagnosticList(values []string) error {
	if len(values) > 64 {
		return fmt.Errorf("diagnostic: oversized list")
	}
	return diagnosticText(values...)
}
func diagnosticTimestamp(v string) error {
	at, err := time.Parse(time.RFC3339Nano, v)
	if err != nil || at.IsZero() || len(v) > 64 {
		return fmt.Errorf("diagnostic: invalid timestamp")
	}
	return nil
}
func diagnosticPhase(v string) bool {
	return slices.Contains([]string{"reserving", "draining", "unloading", "baseline", "starting", "loading", "low", "high", "restoring", "done"}, v)
}
func diagnosticOutcome(v string) bool {
	return slices.Contains([]string{"running", "completed", "failed", "cancelled", "cleanup-failed"}, v)
}
