// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

const modelReviewTTL = 10 * time.Minute
const modelReviewCount = 64
const modelReviewBytes = 4 << 20

func modelWorkerElapsed(start uint64, clockErr error) int64 {
	end, err := noderec.MemoryBootClock()
	if clockErr != nil || err != nil {
		return -1
	}
	ms, err := noderec.ElapsedMemoryMS(start, end)
	if err != nil {
		return -1
	}
	return ms
}

func modelAgeAdvice(estimate *modelconfig.Estimate, elapsed int64) {
	if estimate == nil {
		return
	}
	expired := elapsed < 0
	for i := range estimate.Domains {
		d := &estimate.Domains[i]
		if elapsed < 0 {
			d.RemainingValidityMS = 0
		} else {
			d.RemainingValidityMS = max(0, d.RemainingValidityMS-elapsed)
		}
		if d.RemainingValidityMS == 0 {
			d.CapacityFit, d.FreeFit = "unknown", "unknown"
			expired = true
		}
	}
	if expired {
		estimate.CapacityFit, estimate.FreeFit = "unknown", "unknown"
	}
}

// Existing scanner/cache owners retire observations on process and directory
// replacement. This lookup never creates an observation or renews its age.
func (b *Broker) modelTargetMemory() *modelconfig.TargetMemory {
	if b.telemetry == nil {
		return nil
	}
	b.telemetry.mu.RLock()
	observation := b.telemetry.nodes[b.nodeID].scanner
	run := ""
	if observation != nil && observation.value.Memory != nil {
		run = observation.value.Memory.RunID
	}
	b.telemetry.mu.RUnlock()
	view := b.lookupLocalMemory(b.nodeID, run, time.Now())
	if view.Observation == nil {
		return nil
	}
	identity := modelSettingsHash(struct {
		Host, Run string
		Domains   []noderec.MemoryDomain
	}{b.nodeID, run, view.Observation.Domains})
	hop := noderec.NewMemoryHop("model-settings-broker", noderec.MemoryBootClock)
	if hop == nil {
		return nil
	}
	return &modelconfig.TargetMemory{HostID: b.nodeID, ObservationID: identity, Observation: view.Observation, Hop: hop}
}

func modelWorkerRequest[T any](b *Broker, request T, caller string, native bool) settings.WorkerRequest[T] {
	b.engineConfigMu.Lock()
	err := b.authorizeSettingsCallerLocked(caller)
	if err == nil {
		err = b.authorizeRemoteManagement(caller, "settings", "")
	}
	runtime := b.nodePolicyRuntime()
	runtime.mu.Lock()
	blocked := runtime.managementBlockedLocked() || b.buildApplyMaintenance.active != nil
	runtime.mu.Unlock()
	b.engineConfigMu.Unlock()
	envelope := settings.WorkerRequest[T]{Request: request, Memory: b.modelTargetMemory(), NativeInitAllowed: native && err == nil && !blocked}
	// Advisory observations cannot consume the portable request's transport
	// budget. Missing input remains Unknown and cannot authorize Auto.
	if modelconfig.CheckPayloadSize(envelope) != nil {
		envelope.Memory = nil
		envelope.NativeInitAllowed = false
	}
	return envelope
}

func modelReviewMemoryCurrent(before, after *modelconfig.TargetMemory, loads []modelconfig.ModelLoad, selected string) bool {
	if before == nil || after == nil || before.HostID != after.HostID {
		return false
	}
	a, b := before.At(noderec.MemoryBootClock), after.At(noderec.MemoryBootClock)
	if a == nil || b == nil {
		return false
	}
	for _, observation := range []*noderec.MemoryObservation{a, b} {
		for i := range observation.Domains {
			pool := &observation.Domains[i]
			required := pool.Kind == "system" || pool.Kind == "shared"
			for _, load := range loads {
				if load.ModelID != selected {
					continue
				}
				if load.Resolved != nil && load.Resolved.DeviceBinding != nil && load.Resolved.DeviceBinding.DomainID == pool.ID {
					required = true
				}
			}
			if !required {
				pool.Total = noderec.MemoryFact{}
				pool.Free = noderec.MemoryFact{}
				continue
			}
			for _, fact := range []*noderec.MemoryFact{&observation.Domains[i].Total, &observation.Domains[i].Free} {
				if !fact.Valid {
					return false
				}
				fact.AgeMS = 0
			}
		}
	}
	return reflect.DeepEqual(a, b)
}

type modelSettingsRetainedReview struct {
	Target  string
	Request settings.Request
	Caller  string
	Preview settings.Preview
	Source  modelSettingsReviewSource
	Created time.Time
	Bytes   int
}

func modelReviewRequest(p settings.Request) settings.Request {
	p = p.Clone()
	p.RequestID = ""
	p.ModelChange.PreviewDigest = ""
	return p
}

func (b *Broker) modelReviewRetainedLocked(p settings.Request, caller string, created time.Time) bool {
	for _, record := range b.modelReviews {
		if record.Target == b.nodeID && record.Caller == caller && record.Preview.PreviewDigest == p.ModelChange.PreviewDigest && record.Created.Equal(created) && reflect.DeepEqual(record.Request, modelReviewRequest(p)) {
			return true
		}
	}
	return false
}

func (b *Broker) releaseAcceptedModelReviewLocked(p settings.Request, caller string) {
	for i, record := range b.modelReviews {
		if record.Target == b.nodeID && record.Caller == caller && record.Preview.PreviewDigest == p.ModelChange.PreviewDigest {
			copy(b.modelReviews[i:], b.modelReviews[i+1:])
			b.modelReviews[len(b.modelReviews)-1] = modelSettingsRetainedReview{}
			b.modelReviews = b.modelReviews[:len(b.modelReviews)-1]
			return
		}
	}
}

// Caller holds engineConfigMu. This table retains unaccepted authority only;
// receipt retries and recovery remain exclusively journal-owned.
func (b *Broker) retainModelReviewLocked(p settings.Request, caller string, preview settings.Preview, source modelSettingsReviewSource) error {
	now := time.Now()
	record := modelSettingsRetainedReview{Target: b.nodeID, Request: modelReviewRequest(p), Caller: caller, Preview: preview.Clone(), Source: source, Created: now}
	encoded, err := json.Marshal(struct {
		Request               settings.Request
		Caller                string
		Preview               settings.Preview
		Launch                settings.LaunchState
		Target, Policy, Epoch string
		Created               time.Time
	}{record.Request, caller, record.Preview, source.Launch, record.Target, source.Policy, source.Source.settingsEpoch, record.Created})
	if err != nil || len(encoded) > modelReviewBytes {
		return modelSettingsConflict("review exceeds retention budget; reduce settings and Recalculate")
	}
	record.Bytes = len(encoded)
	kept := make([]modelSettingsRetainedReview, 0, min(len(b.modelReviews), modelReviewCount))
	bytes := 0
	for _, old := range b.modelReviews {
		if now.Sub(old.Created) >= modelReviewTTL || old.Caller == caller && old.Request.Engine == p.Engine && old.Request.ModelChange.ModelID == p.ModelChange.ModelID {
			continue
		}
		kept = append(kept, old)
		bytes += old.Bytes
	}
	for len(kept) >= modelReviewCount || bytes+record.Bytes > modelReviewBytes {
		bytes -= kept[0].Bytes
		kept = kept[1:]
	}
	b.modelReviews = append(kept, record)
	return nil
}

func (b *Broker) retainedModelReview(ctx context.Context, p settings.Request, caller string) (settings.Preview, modelSettingsReviewSource, error) {
	b.engineConfigMu.Lock()
	var found *modelSettingsRetainedReview
	for _, record := range b.modelReviews {
		if record.Target == b.nodeID && record.Caller == caller && record.Preview.PreviewDigest == p.ModelChange.PreviewDigest {
			if err := p.ValidateModelChange(record.Request.Settings, true); err != nil {
				b.engineConfigMu.Unlock()
				return settings.Preview{}, modelSettingsReviewSource{}, &invalidParamsError{err: err}
			}
		}
		if record.Target == b.nodeID && record.Caller == caller && record.Preview.PreviewDigest == p.ModelChange.PreviewDigest && reflect.DeepEqual(record.Request, modelReviewRequest(p)) && time.Since(record.Created) < modelReviewTTL {
			copy := record
			found = &copy
			break
		}
	}
	b.engineConfigMu.Unlock()
	if found == nil {
		return settings.Preview{}, modelSettingsReviewSource{}, modelSettingsConflict("review expired, superseded or changed; explicitly Recalculate")
	}
	if !b.modelWorkerCurrent(found.Source.Worker, found.Source.Source) {
		return settings.Preview{}, modelSettingsReviewSource{}, modelSettingsConflict("review worker changed; explicitly Recalculate")
	}
	if p.ModelChange.Load != nil {
		request := p.Clone()
		request.Settings, request.ModelChange = found.Preview.Settings.Clone(), nil
		envelope := modelWorkerRequest(b, request, caller, false)
		for _, load := range request.Settings.ModelLoads {
			if load.ModelID == p.ModelChange.ModelID && load.Resolved != nil && strings.HasPrefix(load.Resolved.ResolverRevision, "auto/") {
				envelope.ValidateResources = true
				envelope.ResourceModel = load.ModelID
			}
		}
		var validated settings.Preview
		if err := modelSettingsCall(ctx, found.Source.Worker, "engine:preview-launch", envelope, &validated); err != nil {
			code := engineSettingsErrorCode(err)
			if code == -32011 || code == -32012 {
				return settings.Preview{}, modelSettingsReviewSource{}, err
			}
			return settings.Preview{}, modelSettingsReviewSource{}, modelSettingsConflict("frozen native or resource validation failed; explicitly Recalculate: " + err.Error())
		}
		if time.Since(found.Created) >= modelReviewTTL {
			return settings.Preview{}, modelSettingsReviewSource{}, modelSettingsConflict("review expired during validation; explicitly Recalculate")
		}
		if len(validated.Errors) != 0 || validated.Conflict != nil || !validated.Settings.Equal(found.Preview.Settings) {
			return settings.Preview{}, modelSettingsReviewSource{}, modelSettingsConflict("frozen native inputs changed; explicitly Recalculate")
		}
		if envelope.ValidateResources {
			found.Source.Memory = envelope.Memory
		}
	}
	found.Source.ReviewedAt = found.Created
	return found.Preview.Clone(), found.Source, nil
}
