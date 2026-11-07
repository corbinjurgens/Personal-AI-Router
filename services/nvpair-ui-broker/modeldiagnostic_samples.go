// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"time"

	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/noderec"
)

type diagnosticPoolSample struct {
	at        time.Time
	available uint64
	source    string
}

type diagnosticSamples struct {
	producer      string
	seen          map[string]bool
	pools         map[string][]diagnosticPoolSample
	baselineDone  bool
	baselinePools []string
	bindings      map[string]string
}

func (b *Broker) diagnosticMemory() (*noderec.MemoryObservation, error) {
	memory := b.modelTargetMemory()
	if memory == nil {
		return nil, diagnosticError("memory-telemetry-unavailable")
	}
	observation := memory.At(noderec.MemoryBootClock)
	if observation == nil || observation.Validate() != nil || len(observation.Domains) == 0 {
		return nil, diagnosticError("memory-telemetry-unavailable")
	}
	for _, domain := range observation.Domains {
		if !domain.Free.Valid || !domain.Total.Valid {
			return nil, diagnosticError("memory-telemetry-unavailable")
		}
	}
	return observation, nil
}

func (b *Broker) diagnosticMemoryWarnings() []string {
	observation, err := b.diagnosticMemory()
	if err != nil {
		return nil
	}
	warnings := []string{}
	for _, domain := range observation.Domains {
		total, totalErr := strconv.ParseUint(domain.Total.Bytes, 10, 64)
		free, freeErr := strconv.ParseUint(domain.Free.Bytes, 10, 64)
		if totalErr != nil || freeErr != nil || total == 0 || free > total {
			continue
		}
		used := total - free
		if domain.Kind == "dedicated" && used > total/20 {
			warnings = append(warnings, "GPU memory use is above 5 percent on "+domain.ID+"; other programs may affect the result.")
		}
		if (domain.Kind == "system" || domain.Kind == "shared") && used > total/2 {
			warnings = append(warnings, "RAM use is above half capacity on "+domain.ID+"; the diagnostic may run out of memory.")
		}
	}
	return warnings
}

// Sample identity excludes carried age and receipt time. Seeing the same source
// fact on another tick cannot turn it into another baseline or overlap sample.
func (s *diagnosticSamples) observe(observation *noderec.MemoryObservation, now time.Time) error {
	if observation == nil || observation.Validate() != nil {
		return diagnosticError("invalid diagnostic telemetry")
	}
	if s.producer != "" && s.producer != observation.RunID {
		return diagnosticError("memory telemetry producer changed during diagnostic")
	}
	s.producer = observation.RunID
	if s.seen == nil {
		s.seen = make(map[string]bool)
		s.pools = make(map[string][]diagnosticPoolSample)
		s.bindings = make(map[string]string)
	}
	if len(s.baselinePools) == 0 {
		for _, domain := range observation.Domains {
			s.baselinePools = append(s.baselinePools, domain.ID)
		}
	}
	if len(observation.Domains) != len(s.baselinePools) {
		return diagnosticError("memory pool set changed during diagnostic")
	}
	for _, domain := range observation.Domains {
		devices := slices.Clone(domain.DeviceIDs)
		slices.Sort(devices)
		binding := modelSettingsHash(struct {
			Kind, Capacity, TotalSource, FreeSource string
			Devices                                 []string
		}{domain.Kind, domain.Total.Bytes, domain.Total.Source, domain.Free.Source, devices})
		if previous, exists := s.bindings[domain.ID]; exists && previous != binding {
			return diagnosticError("memory pool identity or source changed during diagnostic")
		}
		if !slices.Contains(s.baselinePools, domain.ID) {
			return diagnosticError("memory pool set changed during diagnostic")
		}
		s.bindings[domain.ID] = binding
		f := domain.Free
		if !f.Valid || !domain.Total.Valid || f.AgeMS < 0 || f.AgeMS > noderec.MemoryFreshMS || f.SampledAt.After(now) || now.Sub(f.SampledAt) > time.Duration(noderec.MemoryFreshMS)*time.Millisecond {
			continue
		}
		value, err := strconv.ParseUint(f.Bytes, 10, 64)
		capacity, capacityErr := strconv.ParseUint(domain.Total.Bytes, 10, 64)
		if err != nil || capacityErr != nil || value > capacity {
			continue
		}
		key := modelSettingsHash(struct {
			Run, Domain, Source string
			At                  time.Time
		}{observation.RunID, domain.ID, f.Source, f.SampledAt})
		if s.seen[key] {
			continue
		}
		s.seen[key] = true
		s.pools[domain.ID] = append(s.pools[domain.ID], diagnosticPoolSample{f.SampledAt, value, f.Source})
	}
	return nil
}

func diagnosticInWindow(at, start, end time.Time) bool {
	return !start.IsZero() && !at.Before(start) && (end.IsZero() || !at.After(end))
}

func (s *diagnosticSamples) baselineReady(at time.Time) bool {
	if at.IsZero() || len(s.baselinePools) == 0 {
		return false
	}
	for _, domain := range s.baselinePools {
		count := 0
		for _, sample := range s.pools[domain] {
			if !sample.at.Before(at) {
				count++
			}
		}
		if count < 3 {
			return false
		}
	}
	return true
}

func (b *Broker) sampleDiagnostic(ctx context.Context, op *brokerDiagnostic) error {
	// The ticker belongs to this operation and ends before done closes, including
	// cancellation and cleanup-failed. Notifications only trigger status reads.
	ticker := time.NewTicker(diagnosticSampleInterval)
	defer ticker.Stop()
	for {
		status, err := b.readDiagnosticManager(ctx, op)
		if err != nil {
			return err
		}
		op.status = status
		op.status.Markers = b.diagnosticMarkers(op)
		if status.Run != nil {
			b.setDiagnosticRun(op, *status.Run)
		}
		if status.State == "cleanup-failed" {
			return diagnosticError("manager cleanup failed")
		}
		if status.WorkStopped {
			op.stopped = true
			return nil
		}
		if status.Run == nil || status.State == "none" {
			return diagnosticError("accepted diagnostic status unavailable")
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-op.wake:
			continue
		case <-ticker.C:
			observation, err := b.diagnosticMemory()
			if err != nil {
				return err
			}
			if err := op.samples.observe(observation, b.diagnosticNow()); err != nil {
				return err
			}
			if !op.samples.baselineDone && op.samples.baselineReady(b.diagnosticMarkers(op)["baseline-ready"]) {
				if err := b.diagnosticCall(ctx, op, "engine:diagnostic-baseline-done", struct {
					RunID string `json:"runId"`
					Token string `json:"token"`
					Phase string `json:"phase"`
				}{op.runID, op.token, "baseline"}, nil); err != nil {
					return err
				}
				op.samples.baselineDone = true
			}
		}
	}
}

func diagnosticMinimum(samples []diagnosticPoolSample, start, end time.Time) (string, int) {
	var minimum uint64
	count := 0
	for _, sample := range samples {
		if diagnosticInWindow(sample.at, start, end) {
			if count == 0 || sample.available < minimum {
				minimum = sample.available
			}
			count++
		}
	}
	if count < 2 {
		return "", count
	}
	return strconv.FormatUint(minimum, 10), count
}

func (s *diagnosticSamples) measured(identity modelconfig.DiagnosticIdentity, status diagnosticManagerStatus, now time.Time) (modelconfig.MeasuredProfile, error) {
	if status.Run == nil {
		return modelconfig.MeasuredProfile{}, diagnosticError("run identity missing for record")
	}
	result := modelconfig.MeasuredProfile{DiagnosticIdentity: identity.Clone(), RunID: status.Run.RunID, RecordedAt: now.Format(time.RFC3339Nano), Outcome: status.Run.Outcome, FailedPhase: status.Run.FailedPhase, Reason: status.Run.Reason, BuildPin: identity.NativeBuildID,
		Notes: []string{"Measured by sampling OS memory pools every ~2 s; short peaks may be missed. Other programs are included; this is not an exact process allocation."}}
	if status.Low != nil {
		v := status.Low.Clone()
		result.Low = &v
	}
	if status.High != nil {
		v := status.High.Clone()
		result.High = &v
	}
	markers := status.Markers
	for _, placement := range identity.Placement {
		pool := modelconfig.MeasuredPool{DomainID: placement.DomainID, Kind: placement.Kind, DeviceID: placement.DeviceID, Source: "OS pool telemetry", IntervalMS: diagnosticSampleInterval.Milliseconds()}
		samples := s.pools[placement.DomainID]
		baseline := []uint64{}
		for _, sample := range samples {
			if diagnosticInWindow(sample.at, markers["baseline-ready"], markers["loaded"]) && len(baseline) < 3 {
				baseline = append(baseline, sample.available)
			}
		}
		if len(baseline) == 3 {
			slices.Sort(baseline)
			pool.BaselineAvailableBytes = strconv.FormatUint(baseline[1], 10)
		} else {
			result.Notes = append(result.Notes, "Insufficient cold baseline samples for "+placement.DomainID+".")
		}
		pool.AfterLoadAvailableBytes, _ = diagnosticMinimum(samples, markers["loaded"], markers["low-start"])
		pool.LowMinAvailableBytes, _ = diagnosticMinimum(samples, markers["low-start"], markers["low-end"])
		pool.HighMinAvailableBytes, _ = diagnosticMinimum(samples, markers["high-start"], markers["high-end"])
		if pool.AfterLoadAvailableBytes == "" || pool.LowMinAvailableBytes == "" || pool.HighMinAvailableBytes == "" {
			result.Notes = append(result.Notes, "Too few fresh samples in an allocation or exercise phase for "+placement.DomainID+"; minimum absent.")
		}
		for _, sample := range samples {
			if diagnosticInWindow(sample.at, markers["baseline-ready"], markers["high-end"]) {
				pool.Samples++
			}
		}
		// An uncovered phase has an open peak, not a zero-byte result.
		if pool.BaselineAvailableBytes != "" && pool.AfterLoadAvailableBytes != "" && pool.LowMinAvailableBytes != "" && pool.HighMinAvailableBytes != "" {
			base, _ := strconv.ParseUint(pool.BaselineAvailableBytes, 10, 64)
			minimum := base
			for _, value := range []string{pool.AfterLoadAvailableBytes, pool.LowMinAvailableBytes, pool.HighMinAvailableBytes} {
				n, err := strconv.ParseUint(value, 10, 64)
				if err != nil {
					return result, err
				}
				minimum = min(minimum, n)
			}
			pool.PeakUsedBytes = strconv.FormatUint(base-minimum, 10)
		}
		result.Pools = append(result.Pools, pool)
	}
	if result.High != nil {
		sampled := diagnosticOverlapSampled(*result.High, markers["high-start"], identity.Placement, s.pools)
		result.High.OverlapWindowSampled = &sampled
		if !sampled {
			result.High.Coverage = "partial"
			result.Notes = append(result.Notes, "Full-context overlap was not covered by fresh memory samples in every pool.")
		}
	}
	return result, result.Validate()
}

func diagnosticOverlapSampled(high modelconfig.MeasuredPhase, start time.Time, placements []modelconfig.DiagnosticPlacement, pools map[string][]diagnosticPoolSample) bool {
	if start.IsZero() || len(high.Requests) == 0 || high.Overlapped == nil || !*high.Overlapped || len(placements) == 0 {
		return false
	}
	latest, earliest := float64(0), math.Inf(1)
	for _, request := range high.Requests {
		if request.FirstTokenMS == nil || request.EndMS == nil {
			return false
		}
		latest = max(latest, *request.FirstTokenMS)
		earliest = min(earliest, *request.EndMS)
	}
	if latest >= earliest || math.IsInf(earliest, 0) || earliest > float64(diagnosticRunLimit.Milliseconds()) {
		return false
	}
	from, to := start.Add(time.Duration(latest*float64(time.Millisecond))), start.Add(time.Duration(earliest*float64(time.Millisecond)))
	for _, placement := range placements {
		found := false
		for _, sample := range pools[placement.DomainID] {
			if diagnosticInWindow(sample.at, from, to) {
				found = true
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (b *Broker) recordDiagnostic(ctx context.Context, op *brokerDiagnostic) error {
	identity := op.status.Identity
	if identity == nil && op.status.Observation != nil {
		observed := op.review.preview.DiagnosticIdentity.Clone()
		observed.PerSlotContext, observed.Slots = op.status.Observation.PerSlotContext, op.status.Observation.Slots
		observed.BuildInfo = op.status.Observation.BuildInfo
		if op.status.Observation.NativeBuildID != "" {
			observed.NativeBuildID = op.status.Observation.NativeBuildID
		}
		observed.Placement = slices.Clone(op.status.Observation.PoolPlacement)
		identity = &observed
	}
	if identity == nil {
		if op.status.Run == nil || op.status.Run.Outcome == "completed" {
			return diagnosticError("manager has not supplied observed diagnostic identity")
		}
		known := op.review.preview.DiagnosticIdentity.Clone()
		identity = &known // Failed pre-load evidence retains unknown runtime fields.
	}
	// Requested evidence must still match consent. Realized allocation is
	// supplied by the native observation and can differ for unpriced profiles.
	if !diagnosticRequestedIdentityMatches(*identity, op.review.preview.DiagnosticIdentity) {
		return diagnosticError("observed identity differs from consent")
	}
	op.status.Markers = b.diagnosticMarkers(op)
	profile, err := op.samples.measured(*identity, op.status, b.diagnosticNow())
	if err != nil {
		return err
	}
	// Fresh target inspection links only a buffer sample for this exact load.
	// GPU-selection records intentionally never attach buffer evidence.
	if identity.ConfigurationKeyReason != "gpu-selection" {
		var inspection modelconfig.InspectResult
		preview := op.review.preview
		request := settings.WorkerRequest[modelconfig.InspectRequest]{Request: modelconfig.InspectRequest{Engine: preview.Engine, ModelID: preview.ModelID, ProfileKey: preview.ProfileKey}, Memory: b.modelTargetMemory(), NativeInitAllowed: false}
		if err := b.diagnosticCall(ctx, op, "engine:inspect-model", request, &inspection); err != nil {
			return err
		}
		if inspection.ConfigurationKey != identity.ConfigurationKey || inspection.ExecutionKey != identity.ExecutionKey || inspection.NativeBuildID != identity.NativeBuildID {
			return diagnosticError("post-run inspection identity changed")
		}
		if observation := inspection.LoadObservation; observation != nil && observation.Current {
			profile.BufferSampleID = observation.SampleID
		}
	}
	if err := profile.Validate(); err != nil {
		return fmt.Errorf("measured record: %w", err)
	}
	if err := b.diagnosticCall(ctx, op, "engine:record-measured-memory", struct {
		RunID   string                      `json:"runId"`
		Token   string                      `json:"token"`
		Profile modelconfig.MeasuredProfile `json:"profile"`
	}{op.runID, op.token, profile}, nil); err != nil {
		return err
	}
	b.diagnostics.mu.Lock()
	op.run.Result = &profile
	b.diagnostics.mu.Unlock()
	if run, _ := b.diagnosticStatus(op.runID); run != nil {
		b.publishDiagnostic(*run)
	}
	return nil
}

func diagnosticRequestedIdentityMatches(actual, expected modelconfig.DiagnosticIdentity) bool {
	actual, expected = actual.Clone(), expected.Clone()
	actual.BuildInfo, expected.BuildInfo = "", ""
	actual.PerSlotContext, expected.PerSlotContext = 0, 0
	actual.Slots, expected.Slots = 0, 0
	actual.Placement, expected.Placement = nil, nil
	byID := func(a, b modelconfig.DiagnosticGPU) int { return cmp.Compare(a.UUID, b.UUID) }
	slices.SortFunc(actual.GPUs, byID)
	slices.SortFunc(expected.GPUs, byID)
	return reflect.DeepEqual(actual, expected)
}
