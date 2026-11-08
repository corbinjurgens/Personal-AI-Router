// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"reflect"
	"slices"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
)

// Stored only in policyRuntime. These are observations, never persisted current
// flags or engine authority. Each replay keeps the original boot-clock budget.
type performanceConfiguration struct {
	broker                    *Broker
	worker                    *rpcWorker
	source                    *engineManagerSource
	generation                uint64
	settings, revision, model string
	engineEvidence            string
	memory                    *modelconfig.TargetMemory
	state                     nodepolicy.MeasurementState
}

func (b *Broker) wakePerformanceConfiguration() {
	if p := b.performance; p != nil {
		performanceSignal(p.configurationWake)
	}
}

func measurementMemoryTopology(memory *modelconfig.TargetMemory) *noderec.MemoryObservation {
	if memory == nil {
		return nil
	}
	observed := memory.At(noderec.MemoryBootClock)
	if observed == nil || len(observed.Domains) == 0 {
		return nil
	}
	for i := range observed.Domains {
		pool := &observed.Domains[i]
		if !pool.Total.Valid {
			return nil
		}
		pool.Total = noderec.MemoryFact{Bytes: pool.Total.Bytes, Valid: true}
		pool.Free = noderec.MemoryFact{}
		slices.Sort(pool.DeviceIDs)
	}
	slices.SortFunc(observed.Domains, func(a, b noderec.MemoryDomain) int {
		if a.ID < b.ID {
			return -1
		}
		if a.ID > b.ID {
			return 1
		}
		return 0
	})
	return observed
}

func measurementMemoryCurrent(before, after *modelconfig.TargetMemory) bool {
	if before == nil || after == nil || before.HostID != after.HostID {
		return false
	}
	a, b := measurementMemoryTopology(before), measurementMemoryTopology(after)
	return a != nil && b != nil && reflect.DeepEqual(a, b)
}

func (c performanceConfiguration) currentLocked(r *policyRuntime, engine, key string) bool {
	if c.source == nil || c.source != r.engineManagerSource || c.generation != r.measurementGeneration || c.settings == "" || c.settings != r.settingsIdentity[engine] || c.revision != r.current.Revision {
		return false
	}
	identity, known := r.engineIdentity[engine]
	if !known || c.engineEvidence != modelSettingsHash(identity) {
		return false
	}
	profile, ok := r.current.ProfileByKey(key)
	if !ok || profile.Engine != engine || profile.Model != c.model || c.broker == nil {
		return false
	}
	return c.broker.modelWorkerCurrent(c.worker, c.source) && measurementMemoryCurrent(c.memory, c.broker.modelTargetMemory())
}

// Scanner lifecycle callbacks retire the observation generation even if the
// replacement publishes identical memory before an outstanding read returns.
func (b *Broker) invalidateMeasurementConfigurations() {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.measurementGeneration++
	r.measurementConfigurations = nil
	r.mu.Unlock()
	b.wakePerformanceConfiguration()
	b.schedulePolicyPush(pushIdentity)
}

// Caller holds r.mu. Validation reads existing local memory owners only; worker
// RPC and hardware inspection are confined to the asynchronous refresh below.
func (r *policyRuntime) measurementStatesLocked() map[string]map[string]nodepolicy.MeasurementState {
	var out map[string]map[string]nodepolicy.MeasurementState
	for engine, profiles := range r.measurementConfigurations {
		for key, configuration := range profiles {
			if !configuration.currentLocked(r, engine, key) {
				delete(profiles, key)
				continue
			}
			if out == nil {
				out = map[string]map[string]nodepolicy.MeasurementState{}
			}
			if out[engine] == nil {
				out[engine] = map[string]nodepolicy.MeasurementState{}
			}
			out[engine][key] = configuration.state
		}
	}
	return out
}

func measurementState(identity modelconfig.MeasurementIdentity, memory *modelconfig.TargetMemory) (nodepolicy.MeasurementState, bool) {
	if identity.Validate() != nil || !identity.Complete || memory == nil {
		return nodepolicy.MeasurementState{}, false
	}
	observation := memory.At(noderec.MemoryBootClock)
	if observation == nil {
		return nodepolicy.MeasurementState{}, false
	}
	remaining := noderec.MemoryFreshMS
	for _, pool := range memory.Observation.Domains {
		if !pool.Total.Valid {
			return nodepolicy.MeasurementState{}, false
		}
		remaining = min(remaining, noderec.MemoryFreshMS-pool.Total.AgeMS)
	}
	// Age the budget from the original hop, not receipt or a later replay.
	if remaining <= 0 {
		return nodepolicy.MeasurementState{}, false
	}
	state := nodepolicy.MeasurementState{Identity: identity, ObservedBootNS: memory.Hop.BootNS, ValidityMS: remaining}
	return state, state.Validate() == nil
}

// Reuse the performance sweep's one serialized refresh. A whole pass has a
// bounded budget and profile ceiling, independent of desktop subscriptions.
func (b *Broker) refreshPerformanceConfigurations(parent context.Context) {
	ctx, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()
	b.workersMu.Lock()
	worker, source := b.engineMgr, b.engineMgrSource
	b.workersMu.Unlock()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	profiles := slices.Clone(r.current.Profiles)
	revision := r.current.Revision
	generation := r.measurementGeneration
	engineEvidence := modelSettingsHash(r.engineIdentity)
	engines := make(map[string]string, len(r.engineIdentity))
	for engine, identity := range r.engineIdentity {
		engines[engine] = modelSettingsHash(identity)
	}
	fingerprints := make(map[string]string, len(r.settingsIdentity))
	for engine, value := range r.settingsIdentity {
		fingerprints[engine] = value
	}
	before := r.measurementStatesLocked()
	r.mu.Unlock()
	if len(profiles) > 128 {
		profiles = nil
	}
	next := map[string]map[string]performanceConfiguration{}
	if worker != nil && source != nil {
		for _, profile := range profiles {
			if ctx.Err() != nil {
				break
			}
			if profile.Engine != "llamacpp" || fingerprints[profile.Engine] == "" || engines[profile.Engine] == "" {
				continue
			}
			memory := b.modelTargetMemory()
			hardware, err := b.diagnosticTargetHardware(ctx, memory)
			if err != nil {
				continue
			}
			request := modelconfig.InspectRequest{Engine: profile.Engine, ModelID: profile.Model, ProfileKey: profile.Key}
			envelope := diagnosticInspectEnvelope{WorkerRequest: settings.WorkerRequest[modelconfig.InspectRequest]{Request: request, Memory: memory, NativeInitAllowed: false}, Hardware: hardware}
			raw, rpcErr, err := worker.CallNoTimeout(ctx, "engine:inspect-model", settingsJSON(envelope))
			if err != nil || rpcErr != nil {
				continue
			}
			result, err := modelconfig.DecodeInspectResult(raw)
			if err != nil || result.Engine != profile.Engine || result.ModelID != profile.Model || result.NodeID != "" || result.MeasurementIdentity == nil {
				continue
			}
			state, ok := measurementState(*result.MeasurementIdentity, memory)
			if !ok {
				continue
			}
			if next[profile.Engine] == nil {
				next[profile.Engine] = map[string]performanceConfiguration{}
			}
			next[profile.Engine][profile.Key] = performanceConfiguration{broker: b, worker: worker, source: source, generation: generation, settings: fingerprints[profile.Engine], engineEvidence: engines[profile.Engine], revision: revision, model: profile.Model, memory: memory.Clone(), state: state}
		}
	}
	r.mu.Lock()
	// A stale pass cannot erase a newer owner/settings/policy observation.
	if source == r.engineManagerSource && generation == r.measurementGeneration && engineEvidence == modelSettingsHash(r.engineIdentity) && revision == r.current.Revision && sameStrings(fingerprints, r.settingsIdentity) {
		r.measurementConfigurations = next
	}
	after := r.measurementStatesLocked()
	r.mu.Unlock()
	if !reflect.DeepEqual(before, after) {
		b.schedulePolicyPush(pushIdentity)
	}
}

// Keep JSON serialization at the existing identity-control owner; this helper
// exposes a detached snapshot to the following sample/summary foundation unit.
func (b *Broker) currentMeasurementIdentities() map[string]map[string]nodepolicy.MeasurementState {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.measurementStatesLocked()
}
