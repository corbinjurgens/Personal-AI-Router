// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

type modelTargetMemoryOwner struct {
	target    *modelconfig.TargetMemory
	deviceMap *modelDeviceMap
}

func (o modelTargetMemoryOwner) ModelAutoMemory(ctx context.Context) (*modelEstimateMemory, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	observation := o.target.At(noderec.MemoryBootClock)
	if observation == nil {
		return nil, nil
	}
	m := &modelEstimateMemory{Target: o.target.Clone(), Map: o.deviceMap, Identity: o.target.ObservationID, Source: "target-node-info", At: time.Now()}
	var age int64
	for _, pool := range observation.Domains {
		d := modelconfig.MemoryDomain{ID: pool.ID}
		switch pool.Kind {
		case "system":
			d.Kind = "host"
		case "shared":
			d.Kind = "shared"
			m.UMA = true
		case "dedicated":
			d.Kind = "device"
		default:
			continue
		}
		if pool.Total.Valid {
			d.CapacityBytes = pool.Total.Bytes
			age = max(age, pool.Total.AgeMS)
		}
		if pool.Free.Valid {
			d.FreeBytes = pool.Free.Bytes
			age = max(age, pool.Free.AgeMS)
		}
		if o.deviceMap != nil && o.deviceMap.Binding != nil && o.deviceMap.Binding.Pool == pool.ID {
			d.DeviceID = o.deviceMap.Binding.UUID
			m.NativeDeviceID = d.DeviceID
		}
		m.Domains = append(m.Domains, d)
	}
	m.At = m.At.Add(-time.Duration(age) * time.Millisecond)
	return m, nil
}

// Only explicit inspection and portable load preview call this adapter. Frozen
// validation uses the observation but never invokes the native metadata helper.
func (e *Executor) modelMemoryContext(ctx context.Context, engine string, target *modelconfig.TargetMemory, native bool) context.Context {
	if target == nil {
		return ctx
	}
	op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
	owner := modelTargetMemoryOwner{target: target.Clone()}
	op.Memory = owner
	observation := target.At(noderec.MemoryBootClock)
	if observation == nil {
		return context.WithValue(ctx, modelAutoOperationKey{}, op)
	}
	topology := modelDeviceTopology{Host: target.HostID, Run: observation.RunID, Observation: target.ObservationID}
	uuid := ""
	for _, pool := range observation.Domains {
		p := modelDevicePool{ID: pool.ID, Kind: pool.Kind}
		if p.Kind == "system" {
			p.Kind = "host"
		}
		for _, alias := range pool.DeviceIDs {
			switch {
			case strings.HasPrefix(alias, "GPU-"):
				value := canonicalModelDeviceUUID(alias)
				if p.UUID != "" && p.UUID != value {
					p.Unsupported = "ambiguous physical UUID aliases"
				}
				p.UUID = value
			case strings.HasPrefix(alias, "pci:"):
				value := canonicalModelDevicePCI(strings.TrimPrefix(alias, "pci:"))
				if p.PCI != "" && p.PCI != value {
					p.Unsupported = "ambiguous physical PCI aliases"
				}
				p.PCI = value
			case strings.HasPrefix(alias, "luid:"):
				if p.LUID != "" && !strings.EqualFold(p.LUID, strings.TrimPrefix(alias, "luid:")) {
					p.Unsupported = "ambiguous physical LUID aliases"
				}
				p.LUID = strings.TrimPrefix(alias, "luid:")
			case strings.HasPrefix(alias, "luid_"):
				if p.LUID != "" && !strings.EqualFold(p.LUID, alias) {
					p.Unsupported = "ambiguous physical LUID aliases"
				}
				p.LUID = alias
			default:
				if pool.Kind == "dedicated" {
					p.Unsupported = "unsupported physical alias"
				}
			}
		}
		if p.Kind == "dedicated" {
			if uuid != "" || p.UUID == "" || p.PCI == "" || p.Unsupported != "" {
				native = false
			}
			uuid = p.UUID
		}
		topology.Pools = append(topology.Pools, p)
	}
	deviceOp := modelDeviceOperation{Topology: topology, Selector: "CPU"}
	current := func() bool { return true }
	if uuid != "" {
		deviceOp.Selector = "CUDA0"
		st, err := e.state(engine)
		if err == nil && native && e.activationVeto(engine) == nil {
			st.mu.Lock()
			permitted := !st.adopted && st.plat.Runtime.modeOrDefault() == "process"
			st.mu.Unlock()
			snapshot, snapshotErr := inspectionSnapshot(st, engine)
			if permitted && snapshotErr == nil {
				current = func() bool {
					after, err := inspectionSnapshot(st, engine)
					return err == nil && snapshot.equal(after) && e.activationVeto(engine) == nil
				}
				runtime, runtimeErr := modelDeviceRuntimeSnapshot(ctx, st, modelDevicesBuild)
				if runtimeErr == nil {
					effective, envErr := modelNativeEnvironment(runtime.Root, snapshot.command.Env)
					if envErr == nil {
						effective, envErr = modelDeviceEnvironment(effective, "")
					}
					if envErr == nil {
						for _, assignment := range effective {
							key, value, _ := strings.Cut(assignment, "=")
							if key == "CUDA_VISIBLE_DEVICES" && value != uuid {
								envErr = servingError("CUDA visibility conflicts with the reviewed full UUID.")
							}
						}
					}
					if envErr == nil {
						effective, envErr = modelDeviceEnvironment(effective, uuid)
					}
					if envErr == nil {
						deviceOp.Runtime = runtime
						deviceOp.Environment = effective
						deviceOp.NativeInitAllowed = true
					}
				}
			}
		}
	}
	mapped := mapModelSettingsDevices(ctx, deviceOp, op.DeviceRunner)
	if !current() || ctx.Err() != nil {
		mapped.Binding = nil
		mapped.Unknown = "target changed during native inspection"
	}
	owner.deviceMap = &mapped
	owner.target.MapID = mapped.ObservationID
	op.Memory = owner
	return context.WithValue(ctx, modelAutoOperationKey{}, op)
}

func (e *Executor) validateUnacceptedAutoMemory(ctx context.Context, p settings.Request, selected string) error {
	hasAuto := false
	for _, load := range p.Settings.ModelLoads {
		hasAuto = hasAuto || load.ModelID == selected && modelLoadHasAuto(load.Requested)
	}
	if !hasAuto {
		return modelLoadError(selected, "auto", "Resource validation requires the selected unaccepted Auto allocation.")
	}
	op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
	if op.Memory == nil {
		return modelLoadError("", "auto", "Fresh target memory unavailable; explicitly Recalculate.")
	}
	st, err := e.state(p.Engine)
	if err != nil {
		return err
	}
	snapshot, err := inspectionSnapshot(st, p.Engine)
	if err != nil {
		return err
	}
	var inputs map[string]modelEstimateInput
	var compileErr error
	validated := e.previewLaunch(snapshot.state, p, func(command launchCommand, rt Runtime) (*llamaServingLaunch, error) {
		prepared, err := compileTypedLlamaInputs(ctx, snapshot.state, command, rt, true, &inputs)
		compileErr = err
		return prepared, err
	})
	if compileErr != nil {
		return compileErr
	}
	if len(validated.Errors) != 0 || validated.Conflict != nil {
		return modelLoadError("", "auto", "Frozen launch validation failed; explicitly Recalculate.")
	}
	for _, load := range p.Settings.ModelLoads {
		if load.ModelID != selected || !modelLoadHasAuto(load.Requested) {
			continue
		}
		in, ok := inputs[load.ModelID]
		if ok {
			in.Memory, err = op.Memory.ModelAutoMemory(ctx)
			if err != nil {
				return err
			}
		}
		if !ok || in.Memory == nil {
			return modelLoadError(load.ModelID, "auto", "Fresh target evidence unavailable; explicitly Recalculate.")
		}
		// The map digest remains opaque. Replaying its exact reviewed projection
		// is permitted only after native/file/environment source equality above;
		// fresh physical facts must still contain this exact pool and full UUID.
		if binding := load.Resolved.DeviceBinding; binding != nil && in.Memory.Target != nil {
			observation := in.Memory.Target.At(noderec.MemoryBootClock)
			matched := false
			if observation != nil {
				for _, pool := range observation.Domains {
					if pool.ID == binding.DomainID && pool.Kind == "dedicated" {
						for _, id := range pool.DeviceIDs {
							matched = matched || id == binding.DeviceID
						}
					}
				}
			}
			if !matched {
				return modelLoadError(load.ModelID, "auto", "Reviewed physical device no longer matches target facts; explicitly Recalculate.")
			}
			mapped := modelDeviceMap{ObservationID: in.Memory.Target.MapID, Binding: &modelDeviceBinding{Host: binding.HostID, Pool: binding.DomainID, UUID: binding.DeviceID, Selector: binding.Selector, Digest: binding.Digest}}
			for _, pool := range observation.Domains {
				if pool.Kind == "system" {
					mapped.HostPool = pool.ID
				}
			}
			in.Memory.Map = &mapped
			in.Memory.NativeDeviceID = binding.DeviceID
			for i := range in.Memory.Domains {
				if in.Memory.Domains[i].ID == binding.DomainID {
					in.Memory.Domains[i].DeviceID = binding.DeviceID
				}
			}
		}
		estimate := e.modelEstimates.estimate(ctx, in)
		known, fits, _ := modelAutoFit(estimate, in.Memory, e.modelEstimates.clock(), load.Resolved.GPU)
		if !known || !fits {
			return modelLoadError(load.ModelID, "auto", "Reviewed allocation no longer fits fresh complete target evidence; explicitly Recalculate.")
		}
	}
	return ctx.Err()
}

func modelSettingsParamsError(err error) error {
	return &codedError{code: -32602, message: err.Error()}
}

type modelMemoryRecordRemoteKey struct{}

func (e *Executor) recordStatedMemory(ctx context.Context, p modelconfig.ClearStatedMemoryRequest, set *modelconfig.SetStatedMemoryRequest) (modelconfig.InspectResult, error) {
	if err := p.Validate(); err != nil {
		return modelconfig.InspectResult{}, modelSettingsParamsError(err)
	}
	if p.NodeID != "" {
		return modelconfig.InspectResult{}, modelSettingsParamsError(fmt.Errorf("nodeId: local target required"))
	}
	st, err := e.state(p.Engine)
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	before, err := inspectionSnapshot(st, p.Engine)
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	result, err := e.InspectModel(ctx, modelconfig.InspectRequest{Engine: p.Engine, ModelID: p.ModelID})
	if err != nil {
		return result, err
	}
	if result.ConfigurationKey == "" || result.ConfigurationKey != p.ConfigurationKey {
		return result, &codedError{code: -32010, message: "Settings changed; reinspect and try again."}
	}
	var profile *modelconfig.MemoryProfile
	if set != nil {
		if err := set.Validate(); err != nil {
			return result, modelSettingsParamsError(err)
		}
		value := modelconfig.MemoryProfile{ConfigurationKey: p.ConfigurationKey, Kind: "stated", RecordedAt: time.Now().UTC().Format(time.RFC3339Nano), HostBytes: set.HostBytes, DeviceBytes: set.DeviceBytes, SharedBytes: set.SharedBytes}
		if err := value.ValidatePools(result.ConfigurationPools); err != nil {
			return result, modelSettingsParamsError(err)
		}
		profile = &value
	}
	// Serialize the statement with finite configuration writers, without holding
	// the engine mutex during inspection or estimation.
	if !st.opMu.TryLock() {
		return result, &codedError{code: -32010, message: "Settings changed; reinspect and try again."}
	}
	defer st.opMu.Unlock()
	state := snapshotLlamaState(st)
	command, err := launchForState(state, state.plat.Runtime.Port)
	if err != nil {
		return result, err
	}
	after := modelInspectionSnapshot{state: state, command: command, store: before.store}
	if !before.equal(after) {
		return result, &codedError{code: -32010, message: "Settings changed; reinspect and try again."}
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if remote, _ := ctx.Value(modelMemoryRecordRemoteKey{}).(bool); remote {
		if err := e.remoteVeto(p.Engine); err != nil {
			return result, err
		}
	}
	if err := e.modelMemoryProfiles.updateInspection(p.Engine, p.ModelID, p.ConfigurationKey, profile, &result); err != nil {
		return result, err
	}
	return result, nil
}

func (m *Manager) statedMemory(ctx context.Context, msg *Message) {
	if msg.Method == "engine:set-stated-memory" {
		p, err := settings.DecodeWorkerRequest[modelconfig.SetStatedMemoryRequest](msg.Params)
		if err != nil {
			m.respondOrErr(msg, nil, modelSettingsParamsError(err))
			return
		}
		ctx = m.exec.modelMemoryContext(ctx, p.Request.Engine, p.Memory, false)
		q := p.Request
		if err := q.Validate(); err != nil {
			m.respondOrErr(msg, nil, modelSettingsParamsError(err))
			return
		}
		if p.RemoteMemoryRecord {
			ctx = context.WithValue(ctx, modelMemoryRecordRemoteKey{}, true)
			if err := m.exec.remoteVeto(q.Engine); err != nil {
				m.respondOrErr(msg, nil, err)
				return
			}
		}
		result, err := m.exec.recordStatedMemory(ctx, modelconfig.ClearStatedMemoryRequest{Engine: q.Engine, NodeID: q.NodeID, ModelID: q.ModelID, ConfigurationKey: q.ConfigurationKey}, &q)
		m.respondOrErr(msg, result, err)
	} else {
		p, err := settings.DecodeWorkerRequest[modelconfig.ClearStatedMemoryRequest](msg.Params)
		if err != nil {
			m.respondOrErr(msg, nil, modelSettingsParamsError(err))
			return
		}
		ctx = m.exec.modelMemoryContext(ctx, p.Request.Engine, p.Memory, false)
		if p.RemoteMemoryRecord {
			ctx = context.WithValue(ctx, modelMemoryRecordRemoteKey{}, true)
			if err := m.exec.remoteVeto(p.Request.Engine); err != nil {
				m.respondOrErr(msg, nil, err)
				return
			}
		}
		result, err := m.exec.recordStatedMemory(ctx, p.Request, nil)
		m.respondOrErr(msg, result, err)
	}
}

// InspectModel is a target-local read. The broker supplies authenticated node
// identity after this reply; a worker never echoes a caller's target as proof.
// Artifact inspection reads bounded local metadata and samples without native,
// network, lifecycle, qualification or settings effects.
func (e *Executor) InspectModel(ctx context.Context, p modelconfig.InspectRequest) (modelconfig.InspectResult, error) {
	// Artifact inspection and advisory estimation share the operation budget.
	ctx, cancel := context.WithTimeout(ctx, modelInspectionBudget)
	defer cancel()
	if err := p.Validate(); err != nil {
		return modelconfig.InspectResult{}, modelSettingsParamsError(err)
	}
	if p.NodeID != "" {
		return modelconfig.InspectResult{}, modelSettingsParamsError(fmt.Errorf("nodeId: worker inspection must address the local target"))
	}
	if err := ctx.Err(); err != nil {
		return modelconfig.InspectResult{}, err
	}
	var advisory modelEstimateInput
	result, err := e.inspectModelArtifacts(ctx, p, &advisory)
	if err != nil {
		return modelconfig.InspectResult{}, err
	}
	if p.Engine == "llamacpp" {
		result.Options, err = modelInspectionOptions(result.Options)
		if err != nil {
			return modelconfig.InspectResult{}, err
		}
	}
	if op, ok := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation); ok && op.Memory != nil {
		advisory.Memory, _ = op.Memory.ModelAutoMemory(ctx)
	}
	result.Estimate = e.modelEstimates.estimate(ctx, advisory)
	e.modelMemoryProfiles.attach(&result)
	if err := completeModelInspectionReply(&result); err != nil {
		return modelconfig.InspectResult{}, err
	}
	return result, nil
}

func completeModelInspectionReply(result *modelconfig.InspectResult) error {
	// The final public reply keeps its existing ceiling. An advisory result
	// cannot turn a successful artifact inspection into a payload-size failure.
	if err := modelconfig.CheckPayloadSize(*result); err != nil {
		result.Estimate = modelEstimateUnknown("estimate omitted: complete inspection reply size limit")
	}
	return result.Validate()
}

func (m *Manager) inspectModel(ctx context.Context, msg *Message) {
	envelope, err := settings.DecodeWorkerRequest[modelconfig.InspectRequest](msg.Params)
	if err != nil {
		m.respondOrErr(msg, nil, modelSettingsParamsError(err))
		return
	}
	p := envelope.Request
	if err := p.Validate(); err != nil {
		m.respondOrErr(msg, nil, modelSettingsParamsError(err))
		return
	}
	ctx = m.exec.modelMemoryContext(ctx, p.Engine, envelope.Memory, envelope.NativeInitAllowed)
	result, err := m.exec.InspectModel(ctx, p)
	m.respondOrErr(msg, result, err)
}

func (m *Manager) remoteInspectModel(ctx context.Context, msg *Message) {
	p, err := modelconfig.DecodeInspectRequest(msg.Params)
	if err != nil {
		m.respondOrErr(msg, nil, modelSettingsParamsError(err))
		return
	}
	if p.NodeID == "" {
		m.respondOrErr(msg, nil, modelSettingsParamsError(fmt.Errorf("nodeId: remote target required")))
		return
	}
	peer, ok := m.peers.lookup(p.NodeID)
	if !ok || m.mesh == nil {
		m.respondOrErr(msg, nil, fmt.Errorf("model inspection unavailable: device is offline"))
		return
	}
	client, err := m.remoteClient(ctx, peer)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	target := p.NodeID
	p.NodeID = ""
	raw, err := client.postJSON(ctx, settingsPath+"inspect", p.Engine, p)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	result, err := modelconfig.DecodeInspectResult(raw)
	if err == nil && (result.Engine != p.Engine || result.ModelID != p.ModelID || result.NodeID != target) {
		err = fmt.Errorf("model inspection: target reply identity mismatch")
	}
	m.respondOrErr(msg, result, err)
}

// Parent launch traffic permits the broker-owned PreserveCORS guard. Public
// settings.DecodeRequest deliberately rejects it, so it is not this decoder.
func decodeModelSettingsLaunch(raw []byte) (settings.Request, error) {
	if err := validateServingModelsJSON(raw, "settings", "serving"); err != nil {
		return settings.Request{}, modelSettingsParamsError(err)
	}
	var p settings.Request
	if err := modelconfig.DecodeStrict(raw, &p); err != nil {
		return settings.Request{}, modelSettingsParamsError(err)
	}
	if err := p.Validate(); err != nil {
		return settings.Request{}, modelSettingsParamsError(err)
	}
	c, err := p.Settings.Canonical()
	if err != nil {
		return settings.Request{}, modelSettingsParamsError(err)
	}
	p.Settings = c
	return p, nil
}

// The worker accepts frozen native settings from its parent. Coupled policy
// acceptance belongs to the broker. Apply must never invoke an Auto resolver.
func (e *Executor) validateModelSettingsLaunch(ctx context.Context, p settings.Request, applying bool) error {
	if err := p.Validate(); err != nil {
		return modelSettingsParamsError(err)
	}
	if p.ModelChange != nil {
		if applying && p.ModelChange.PreviewDigest == "" {
			return modelSettingsParamsError(fmt.Errorf("modelChange.previewDigest: reviewed preview required"))
		}
		if !applying && p.ModelChange.Load != nil {
			r := p.ModelChange.Load.Set
			// The owner-aware compiler resolves Auto after target baseline and
			// artifact validation. This guard neither resolves nor reads capacity.
			// Keep the exact production refusal when no target owner exists.
			op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
			if modelLoadHasAuto(r) && op.Memory == nil {
				return modelSettingsParamsError(fmt.Errorf("target memory telemetry unavailable"))
			}
		}
		if !applying && p.ModelChange.Load != nil && p.ModelChange.Policy == nil {
			return ctx.Err()
		}
		return modelSettingsParamsError(fmt.Errorf("modelChange: reviewed model settings application is not ready"))
	}
	if applying {
		for _, load := range p.Settings.ModelLoads {
			if load.Resolved == nil {
				return modelSettingsParamsError(fmt.Errorf("settings.modelLoads: reviewed finite resolution required"))
			}
			if modelLoadHasAuto(load.Requested) {
				if err := modelAutoFrozen(load.ModelID, load.Requested, load.Resolved); err != nil {
					return err
				}
			}
		}
	}
	return ctx.Err()
}
