// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

const diagnosticObserveMethod = "internal:diagnostic-observe"
const diagnosticRecordMethod = "engine:record-measured-memory"

// Hardware is target node-info evidence supplied by the private parent, never
// public settings or a native enumeration request. Physical IDs must also occur
// in the fresh memory observation. Include unused GPUs in this identity.
type diagnosticHardware struct {
	CPUModel      string                      `json:"cpuModel"`
	TotalRAMBytes string                      `json:"totalRamBytes"`
	GPUs          []modelconfig.DiagnosticGPU `json:"gpus"`
}

func (h *diagnosticHardware) clone() *diagnosticHardware {
	if h == nil {
		return nil
	}
	c := *h
	c.GPUs = slices.Clone(h.GPUs)
	slices.SortFunc(c.GPUs, func(a, b modelconfig.DiagnosticGPU) int { return strings.Compare(a.UUID, b.UUID) })
	return &c
}

type diagnosticHardwareKey struct{}
type diagnosticCapturedHardwareKey struct{}
type diagnosticConfigurationPoolsKey struct{}

func (e *Executor) diagnosticInspectionPools(ctx context.Context, engine string) []string {
	pools, _ := ctx.Value(diagnosticConfigurationPoolsKey{}).([]string)
	if len(pools) == 0 {
		return nil
	}
	e.mutationAdmission.Lock()
	authorized := e.diagnosticAuthorityLocked(ctx, engine)
	e.mutationAdmission.Unlock()
	if !authorized {
		return nil
	}
	return slices.Clone(pools)
}

func withDiagnosticHardware(ctx context.Context, hardware *diagnosticHardware) context.Context {
	return context.WithValue(ctx, diagnosticHardwareKey{}, hardware.clone())
}

func diagnosticMemory(ctx context.Context) *modelconfig.TargetMemory {
	op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
	if owner, ok := op.Memory.(modelTargetMemoryOwner); ok {
		return owner.target
	}
	return nil
}

func diagnosticHardwareMatchesAt(h *diagnosticHardware, memory *modelconfig.TargetMemory, fresh bool) error {
	if h == nil || memory == nil {
		return diagnosticConflict("target hardware evidence unavailable")
	}
	if err := memory.Validate(); err != nil {
		return err
	}
	observation := memory.Observation
	if fresh {
		observation = memory.At(noderec.MemoryBootClock)
	}
	if observation == nil {
		return diagnosticConflict("fresh target memory evidence unavailable")
	}
	host := false
	matched := map[string]bool{}
	for _, pool := range observation.Domains {
		if !pool.Total.Valid {
			return diagnosticConflict("target pool capacity unavailable")
		}
		if pool.Kind == "system" || pool.Kind == "shared" {
			host = pool.Total.Bytes == h.TotalRAMBytes
			if pool.Kind == "system" {
				continue
			}
		}
		if pool.Kind != "dedicated" && pool.Kind != "shared" {
			return diagnosticConflict("unsupported target pool")
		}
		found := false
		for _, gpu := range h.GPUs {
			if slices.Contains(pool.DeviceIDs, gpu.UUID) && pool.Total.Bytes == gpu.TotalMemoryBytes {
				if matched[gpu.UUID] {
					return diagnosticConflict("ambiguous physical GPU pools")
				}
				matched[gpu.UUID], found = true, true
			}
		}
		if !found {
			return diagnosticConflict("target GPU identity unavailable")
		}
	}
	if !host || len(matched) != len(h.GPUs) {
		return diagnosticConflict("target hardware and memory disagree")
	}
	return nil
}

func (e *Executor) diagnosticRequestedIdentity(ctx context.Context, p modelconfig.InspectRequest, result modelconfig.InspectResult) (modelconfig.DiagnosticIdentity, error) {
	hardware, _ := ctx.Value(diagnosticHardwareKey{}).(*diagnosticHardware)
	captured, _ := ctx.Value(diagnosticCapturedHardwareKey{}).(bool)
	if err := diagnosticHardwareMatchesAt(hardware, diagnosticMemory(ctx), !captured); err != nil {
		return modelconfig.DiagnosticIdentity{}, err
	}
	st, err := e.state(p.Engine)
	if err != nil {
		return modelconfig.DiagnosticIdentity{}, err
	}
	st.mu.Lock()
	var selected *modelconfig.ModelLoad
	for _, load := range st.plat.Runtime.ModelLoads {
		if load.ProfileKey == p.ProfileKey && load.ModelID == p.ModelID {
			copy := modelconfig.CloneLoads([]modelconfig.ModelLoad{load})[0]
			selected = &copy
		}
	}
	st.mu.Unlock()
	if selected == nil || selected.Resolved == nil {
		return modelconfig.DiagnosticIdentity{}, errModelDiagnosticIdentity
	}
	identity := modelconfig.DiagnosticIdentity{
		ConfigurationKey: result.ConfigurationKey, ExecutionKey: result.ExecutionKey, ConfigurationKeyReason: result.ConfigurationKeyReason,
		NativeBuildID: result.NativeBuildID, CPUModel: hardware.CPUModel, TotalRAMBytes: hardware.TotalRAMBytes,
		GPUs: slices.Clone(hardware.GPUs), Artifacts: slices.Clone(result.Artifacts), Requested: selected.Requested.Clone(),
	}
	slices.SortFunc(identity.GPUs, func(a, b modelconfig.DiagnosticGPU) int { return strings.Compare(a.UUID, b.UUID) })
	if err := identity.ValidateRequested(); err != nil {
		return identity, err
	}
	return identity, nil
}

type diagnosticObservation struct {
	Identity              modelconfig.DiagnosticIdentity   `json:"identity"`
	PlannedPerSlotContext int32                            `json:"plannedPerSlotContext"`
	PlannedSlots          int32                            `json:"plannedSlots"`
	EffectsFence          string                           `json:"effectsFence"`
	UnloadModels          []modelconfig.DiagnosticModelRef `json:"unloadModels"`
	ReloadedAfter         []modelconfig.DiagnosticModelRef `json:"reloadedAfter"`
	DrainEngines          []string                         `json:"drainEngines"`
	StartsEngine          bool                             `json:"startsEngine"`
	Supported             bool                             `json:"supported"`
}

func (e *Executor) observeDiagnostic(ctx context.Context, p modelconfig.DiagnosticPreviewRequest, memory *modelconfig.TargetMemory, hardware *diagnosticHardware) (diagnosticObservation, error) {
	if err := p.Validate(); err != nil {
		return diagnosticObservation{}, modelSettingsParamsError(err)
	}
	if p.Engine != "llamacpp" || p.NodeID != "" && (memory == nil || p.NodeID != memory.HostID) {
		return diagnosticObservation{}, diagnosticConflict("local llama.cpp profile required")
	}
	if err := e.diagnosticPauseVeto(); err != nil {
		return diagnosticObservation{}, err
	}
	ctx = withDiagnosticHardware(e.modelMemoryContext(ctx, p.Engine, memory, false), hardware)
	request := modelconfig.InspectRequest{Engine: p.Engine, ModelID: p.ModelID, ProfileKey: p.ProfileKey}
	before, err := e.diagnosticEffectsFence(ctx)
	if err != nil {
		return diagnosticObservation{}, err
	}
	inspection, err := e.inspectModelArtifacts(ctx, request)
	if err != nil {
		return diagnosticObservation{}, err
	}
	identity, err := e.diagnosticRequestedIdentity(ctx, request, inspection)
	if err != nil {
		return diagnosticObservation{}, err
	}
	result := diagnosticObservation{Identity: identity, EffectsFence: before, Supported: true,
		UnloadModels: []modelconfig.DiagnosticModelRef{}, ReloadedAfter: []modelconfig.DiagnosticModelRef{}}
	for _, engine := range e.reg.Names() {
		st, err := e.state(engine)
		if err != nil {
			return diagnosticObservation{}, err
		}
		st.mu.Lock()
		running := st.running
		loads := modelconfig.CloneLoads(st.plat.Runtime.ModelLoads)
		st.mu.Unlock()
		result.DrainEngines = append(result.DrainEngines, engine)
		if engine == p.Engine {
			result.StartsEngine = !running
			for _, load := range loads {
				if load.ProfileKey != p.ProfileKey || load.ModelID != p.ModelID || load.Resolved == nil {
					continue
				}
				if load.Requested.Slots != nil && *load.Requested.Slots > 8 || load.Resolved.Slots > 8 {
					return diagnosticObservation{}, diagnosticConflict("more than eight slots are unsupported")
				}
				result.PlannedSlots = load.Resolved.Slots
				if identity.ConfigurationKey != "" {
					if _, _, err := modelDiagnosticTargets(load.Resolved.PerSlotContext, load.Resolved.Slots); err != nil {
						return diagnosticObservation{}, err
					}
					result.PlannedPerSlotContext = load.Resolved.PerSlotContext
				}
			}
		}
		if !running {
			continue
		}
		loaded, err := e.diagnosticLoaded(ctx, engine)
		if err != nil {
			return diagnosticObservation{}, err
		}
		for _, model := range loaded {
			ref := modelconfig.DiagnosticModelRef{Engine: engine, ModelID: model, ProfileKey: model}
			for _, load := range loads {
				if load.ProfileKey == model {
					ref.ModelID = load.ModelID
				}
			}
			if err := ref.Validate(); err != nil {
				return diagnosticObservation{}, err
			}
			result.UnloadModels = append(result.UnloadModels, ref)
		}
	}
	// Resident reload is policy-owned after drains lift. Do not claim that all
	// previously loaded models will be reloaded merely because they were present.
	after, err := e.diagnosticEffectsFence(ctx)
	if err != nil {
		return diagnosticObservation{}, err
	}
	if before != after {
		return diagnosticObservation{}, diagnosticConflict("effects changed during observation")
	}
	return result, modelconfig.CheckPayloadSize(result)
}

func diagnosticRequestedMatches(a, b modelconfig.DiagnosticIdentity) bool {
	a, b = a.Clone(), b.Clone()
	a.BuildInfo, b.BuildInfo = "", ""
	a.PerSlotContext, b.PerSlotContext = 0, 0
	a.Slots, b.Slots = 0, 0
	a.Placement, b.Placement = nil, nil
	slices.SortFunc(a.GPUs, func(a, b modelconfig.DiagnosticGPU) int { return strings.Compare(a.UUID, b.UUID) })
	slices.SortFunc(b.GPUs, func(a, b modelconfig.DiagnosticGPU) int { return strings.Compare(a.UUID, b.UUID) })
	return reflect.DeepEqual(a, b)
}

type diagnosticRecordParams struct {
	RunID   string                      `json:"runId"`
	Token   string                      `json:"token"`
	Profile modelconfig.MeasuredProfile `json:"profile"`
}

func (e *Executor) recordDiagnostic(p diagnosticRecordParams) error {
	if err := p.Profile.Validate(); err != nil {
		return modelSettingsParamsError(err)
	}
	done, ok := e.completedDiagnosticRun(p.RunID)
	if !ok || p.Token == "" || done.token != p.Token || !done.Status.WorkStopped || done.Status.Identity == nil ||
		p.Profile.RunID != p.RunID || p.Profile.ConfigurationKey != done.Reservation.ConfigurationKey || p.Profile.ExecutionKey != done.Reservation.ExecutionKey ||
		!reflect.DeepEqual(p.Profile.DiagnosticIdentity, *done.Status.Identity) || p.Profile.Outcome != done.Status.Outcome ||
		p.Profile.FailedPhase != done.Status.FailedPhase || p.Profile.Reason != done.Status.Reason {
		return diagnosticConflict("record does not match the completed producing run")
	}
	if p.Profile.BuildPin != done.Status.Identity.NativeBuildID {
		return diagnosticConflict("record build pin mismatch")
	}
	// The sampler classifies overlap only; it cannot substitute native token,
	// timing or cache evidence supplied by the manager's joined exercise.
	for _, pair := range [][2]*modelconfig.MeasuredPhase{{p.Profile.Low, done.Status.Low}, {p.Profile.High, done.Status.High}} {
		if pair[0] == nil || pair[1] == nil {
			if pair[0] != nil || pair[1] != nil {
				return diagnosticConflict("exercise evidence mismatch")
			}
			continue
		}
		if pair[1].Coverage == "partial" && pair[0].Coverage != "partial" {
			return diagnosticConflict("native coverage cannot be promoted")
		}
		a, b := pair[0].Clone(), pair[1].Clone()
		a.Coverage, b.Coverage = "", ""
		a.OverlapWindowSampled, b.OverlapWindowSampled = nil, nil
		if !reflect.DeepEqual(a, b) {
			return diagnosticConflict("exercise evidence mismatch")
		}
	}
	if p.Profile.BufferSampleID != "" && (done.Inspection.LoadObservation == nil || !done.Inspection.LoadObservation.Current || done.Inspection.LoadObservation.SampleID != p.Profile.BufferSampleID) {
		return diagnosticConflict("buffer evidence does not belong to the producing load")
	}
	return e.modelMemoryProfiles.recordMeasured(done.Reservation.Engine, done.Reservation.ModelID, done.Reservation.ProfileKey, p.Profile)
}

func (m *Manager) handleDiagnosticEvidenceRequest(ctx context.Context, msg *Message) bool {
	if !msg.IsRequest() {
		return false
	}
	switch msg.Method {
	case diagnosticObserveMethod:
		go func() {
			var p struct {
				Request  modelconfig.DiagnosticPreviewRequest `json:"request"`
				Memory   *modelconfig.TargetMemory            `json:"memory"`
				Hardware *diagnosticHardware                  `json:"hardware"`
			}
			if err := modelconfig.DecodeStrict(msg.Params, &p); err != nil {
				m.respondOrErr(msg, nil, modelSettingsParamsError(err))
				return
			}
			result, err := m.exec.observeDiagnostic(ctx, p.Request, p.Memory, p.Hardware)
			m.respondOrErr(msg, result, err)
		}()
		return true
	case diagnosticRecordMethod:
		go func() {
			var p diagnosticRecordParams
			if err := modelconfig.DecodeStrict(msg.Params, &p); err != nil {
				m.respondOrErr(msg, nil, modelSettingsParamsError(err))
				return
			}
			m.respondOrErr(msg, map[string]bool{"ok": true}, m.exec.recordDiagnostic(p))
		}()
		return true
	}
	return false
}

// Add hardware solely to the private inspection envelope. Reuse the canonical
// worker decoder for all existing envelope semantics after strict field decode.
func decodeDiagnosticInspectEnvelope(raw []byte) (settings.WorkerRequest[modelconfig.InspectRequest], *diagnosticHardware, error) {
	var p struct {
		settings.WorkerRequest[modelconfig.InspectRequest]
		Hardware *diagnosticHardware `json:"hardware,omitempty"`
	}
	if err := modelconfig.DecodeStrict(raw, &p); err != nil {
		return p.WorkerRequest, nil, err
	}
	base, err := json.Marshal(p.WorkerRequest)
	if err != nil {
		return p.WorkerRequest, nil, err
	}
	worker, err := settings.DecodeWorkerRequest[modelconfig.InspectRequest](base)
	return worker, p.Hardware, err
}

// Records with a complete realization contain authenticated post-load evidence.
// Failed runs can establish currentness for a warning, never estimate bytes.
// Reuse it only
// after re-reading current files, settings, build and every physical pool. A key
// or the persisted Current flag alone never establishes this identity.
func (e *Executor) inspectionDiagnosticIdentity(ctx context.Context, p modelconfig.InspectRequest, result modelconfig.InspectResult, liveRealization ...*bool) *modelconfig.DiagnosticIdentity {
	requested, err := e.diagnosticRequestedIdentity(ctx, p, result)
	if err != nil || e.modelMemoryProfiles == nil {
		return nil
	}
	if live, observed := e.inspectionLoadedDiagnosticIdentity(ctx, p, requested); observed {
		for _, target := range liveRealization {
			*target = live != nil
		}
		return live
	}
	e.modelMemoryProfiles.mu.Lock()
	defer e.modelMemoryProfiles.mu.Unlock()
	for i := len(e.modelMemoryProfiles.records) - 1; i >= 0; i-- {
		record := e.modelMemoryProfiles.records[i]
		m := record.Profile.Measured
		if record.Engine != p.Engine || record.ModelID != p.ModelID || record.ProfileKey != p.ProfileKey || m == nil || !diagnosticRequestedMatches(m.DiagnosticIdentity, requested) {
			continue
		}
		identity := m.DiagnosticIdentity.Clone()
		if identity.Validate() != nil {
			continue
		}
		st, stateErr := e.state(p.Engine)
		if stateErr != nil {
			continue
		}
		st.mu.Lock()
		allocationMatches := false
		for _, load := range st.plat.Runtime.ModelLoads {
			if load.ProfileKey == p.ProfileKey && load.Resolved != nil {
				// Native fit chooses context at load time. Its authenticated
				// realization survives an unchanged restart; a finite GPU-
				// selection load must still match its resolved context/slots.
				allocationMatches = load.Resolved.Slots == identity.Slots && (modelLoadNativeFit(load.Requested) || load.Resolved.PerSlotContext == identity.PerSlotContext)
			}
		}
		st.mu.Unlock()
		if !allocationMatches {
			continue
		}
		memory := diagnosticMemory(ctx)
		valid := true
		for _, placement := range identity.Placement {
			found := false
			for _, pool := range memory.Observation.Domains {
				kind := pool.Kind
				if kind == "system" {
					kind = "host"
				}
				if kind == "dedicated" {
					kind = "device"
				}
				if pool.ID == placement.DomainID && kind == placement.Kind && (kind != "device" || slices.Contains(pool.DeviceIDs, placement.DeviceID)) {
					found = true
				}
			}
			valid = valid && found
		}
		if valid {
			return &identity
		}
	}
	return nil
}

func inspectionMeasurementIdentity(semantic string, diagnostic *modelconfig.DiagnosticIdentity, live bool) modelconfig.MeasurementIdentity {
	if diagnostic == nil || diagnostic.ConfigurationKeyReason == "native-fit" && !live {
		return modelconfig.MeasurementIdentity{}
	}
	return modelconfig.MeasurementFromDiagnostic(semantic, *diagnostic)
}

// A present child supersedes retained realization. If its props cannot be
// authenticated, inspection must not replace that uncertainty with past evidence.
func (e *Executor) inspectionLoadedDiagnosticIdentity(ctx context.Context, p modelconfig.InspectRequest, requested modelconfig.DiagnosticIdentity) (*modelconfig.DiagnosticIdentity, bool) {
	st, err := e.state(p.Engine)
	if err != nil {
		return nil, false
	}
	st.mu.Lock()
	running, proc, run, port, config := st.running, st.proc, st.runID, st.port, st.ownerConfigLocked()
	uncertain := st.adopted || st.stopping && st.running || st.reclaimFailed || st.reclaimPID != 0
	owned := running && proc != nil && !uncertain
	st.mu.Unlock()
	if !running {
		return nil, uncertain || proc != nil
	}
	if !owned {
		return nil, true
	}
	loaded, err := e.diagnosticLoaded(ctx, p.Engine)
	if err != nil {
		return nil, true
	}
	if !slices.Contains(loaded, p.ProfileKey) {
		return nil, false
	}
	observed, err := e.diagnosticReadLoad(ctx, st, port, p.ProfileKey, diagnosticMemory(ctx), time.Time{})
	if err != nil || observed.BuildInfo == "" {
		return nil, true
	}
	st.mu.Lock()
	current := st.running && st.proc == proc && st.runID == run && st.port == port && !st.adopted && !st.stopping && st.ownerConfigLocked() == config
	st.mu.Unlock()
	if !current {
		return nil, true
	}
	requested.BuildInfo, requested.PerSlotContext, requested.Slots, requested.Placement = observed.BuildInfo, observed.PerSlotContext, observed.Slots, observed.PoolPlacement
	if requested.Validate() != nil {
		return nil, true
	}
	return &requested, true
}
