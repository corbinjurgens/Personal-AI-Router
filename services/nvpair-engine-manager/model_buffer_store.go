// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
)

var errModelBufferRealizationPending = errors.New("model buffer load frames pending")

// Realization is read from the live capture, never from persisted samples. The
// parser must have joined a complete load/reservation on one exact child. This
// private path also serves unpriced loads without publishing a buffer sample.
func (e *Executor) modelBufferRealization(ctx context.Context, st *engineState, model string, notBefore time.Time) (string, []modelconfig.BufferReportRow, []modelNativeDevice, error) {
	st.mu.Lock()
	c := st.bufferCapture
	current := c != nil && c.currentLocked(st)
	command := st.plat.Runtime.servingCommand
	st.mu.Unlock()
	if !current || c.sourceID == "" || c.buildID == "" {
		return "", nil, nil, errModelDiagnosticIdentity
	}
	command, err := modelConstrainCommand(command, c.prepared.ModelLoads)
	if err != nil {
		return "", nil, nil, err
	}
	source, err := modelReviewedNativeSource(ctx, st, command)
	build, buildErr := nativeBuildID(ctx, st)
	if err != nil || buildErr != nil || source != c.sourceID || build != c.buildID || !modelBufferBindingsCurrent(ctx, modelBufferPreparedLoads(c.prepared)) {
		return "", nil, nil, errModelDiagnosticIdentity
	}
	c.mu.Lock()
	var selected *modelBufferChild
	for _, child := range c.children {
		if child.model == model {
			if selected != nil {
				c.mu.Unlock()
				return "", nil, nil, errModelDiagnosticIdentity
			}
			selected = child
		}
	}
	if c.retired || c.clockUncertain || time.Now().UnixNano() < c.wallHigh {
		c.mu.Unlock()
		return "", nil, nil, errModelDiagnosticIdentity
	}
	if selected == nil && len(c.children) != 0 {
		c.mu.Unlock()
		return "", nil, nil, errModelDiagnosticIdentity
	}
	if selected == nil || !selected.poisoned && !selected.sleeping && (!selected.loading || !selected.ready) {
		c.mu.Unlock()
		return "", nil, nil, errModelBufferRealizationPending
	}
	if selected.startedAt.Before(notBefore) || selected.poisoned || selected.sleeping || !modelBufferPlacementComplete(selected.parser) {
		c.mu.Unlock()
		return "", nil, nil, errModelDiagnosticIdentity
	}
	rows := slices.Clone(selected.parser.rows)
	identity := modelBufferHash(struct {
		Token, Run, Model, Phase string
		Port                     int
		Generation               uint64
		Rows                     []modelconfig.BufferReportRow
	}{c.token, c.run, model, selected.phase, selected.port, selected.generation, rows})
	generation := selected.generation
	devices := slices.Clone(selected.devices)
	environment := slices.Clone(c.environment)
	c.mu.Unlock()
	needsDevices := slices.ContainsFunc(rows, func(row modelconfig.BufferReportRow) bool {
		return strings.HasPrefix(row.Label, "CUDA") && row.Label != "CUDA_Host"
	})
	if needsDevices && len(devices) == 0 {
		e.mutationAdmission.Lock()
		authorized := e.diagnosticAuthorityLocked(ctx, "llamacpp")
		e.mutationAdmission.Unlock()
		if !authorized || e.diagnosticActivationVeto(ctx, "llamacpp") != nil {
			return "", nil, nil, errModelDiagnosticIdentity
		}
		runtime, err := modelDeviceRuntimeSnapshot(ctx, st)
		if err != nil || runtime.Build != build {
			return "", nil, nil, errModelDiagnosticIdentity
		}
		environment, err = modelDeviceEnvironment(environment, "")
		if err != nil {
			return "", nil, nil, err
		}
		op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
		run := op.DeviceRunner
		if run == nil {
			run = runModelDeviceProcess
		}
		helperCtx, cancel := context.WithTimeout(ctx, modelDevicesBudget)
		reply, err := run(helperCtx, runtime, environment)
		failed := err != nil || helperCtx.Err() != nil || reply.Unknown != "" || reply.Commit == "" || len(reply.Devices) == 0 || len(reply.Devices) > modelDevicesLimit
		cancel()
		if failed {
			return "", nil, nil, errModelDiagnosticIdentity
		}
		devices = slices.Clone(reply.Devices)
	}
	st.mu.Lock()
	current = c.currentLocked(st)
	st.mu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	if !current || c.retired || selected != c.children[selected.port] || selected.generation != generation || selected.poisoned || selected.sleeping || !selected.ready || !modelBufferPlacementComplete(selected.parser) || modelBufferHash(selected.parser.rows) != modelBufferHash(rows) || ctx.Err() != nil {
		return "", nil, nil, errModelDiagnosticIdentity
	}
	selected.devices = slices.Clone(devices)
	return identity, rows, devices, nil
}

const modelBufferHistoryBytes = 4 << 20
const modelBufferHistoryAge = 30 * 24 * time.Hour

type modelBufferStoredSample struct {
	ModelIdentity string                        `json:"modelIdentity"`
	Observation   modelconfig.BufferObservation `json:"observation"`
}

// All disk access belongs to this owner, outside engine and output-pump locks.
// Recovery never reconstructs a live child or a current sample.
type modelBufferStore struct {
	mu            sync.Mutex
	path          string
	replacer      durablefile.Replacer
	samples       []modelBufferStoredSample
	writeDisabled bool
	now           func() time.Time
	view          atomic.Pointer[modelBufferHistoryView]
}

type modelBufferHistoryView struct {
	samples     []modelBufferStoredSample
	unavailable bool
}

func (s *modelBufferStore) publishView() {
	s.view.Store(&modelBufferHistoryView{samples: s.samples, unavailable: s.writeDisabled})
}

func newModelBufferStore(base string) *modelBufferStore {
	s := &modelBufferStore{path: filepath.Join(base, "model-buffer-samples.json"), replacer: engineFileReplacer, now: time.Now}
	if base == "" {
		s.writeDisabled = true
		s.publishView()
		return s
	}
	// The fixed history file shares the executor's user-scoped state root.
	// On POSIX make that root private before reading or publishing history.
	// Windows uses the existing profile directory's inherited user ACL.
	info, err := os.Lstat(base)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 || runtime.GOOS != "windows" && os.Chmod(base, 0o700) != nil {
		s.writeDisabled = true
		s.publishView()
		return s
	}
	s.load()
	return s
}

func validateModelBufferSample(sample modelBufferStoredSample) error {
	encoded, err := json.Marshal(sample)
	if err != nil || len(encoded) > 64<<10 {
		return fmt.Errorf("sample exceeds encoded budget")
	}
	if len(sample.ModelIdentity) != 64 || len(sample.Observation.SampleID) != 64 {
		return fmt.Errorf("invalid sample identity")
	}
	for _, id := range []string{sample.ModelIdentity, sample.Observation.SampleID} {
		for _, c := range id {
			if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
				return fmt.Errorf("invalid sample identity")
			}
		}
	}
	if _, err := time.Parse(time.RFC3339Nano, sample.Observation.RecordedAt); err != nil {
		return fmt.Errorf("invalid sample time")
	}
	return sample.Observation.Validate()
}

func (s *modelBufferStore) load() {
	s.mu.Lock()
	defer s.mu.Unlock()
	defer s.publishView()
	f, err := durablefile.Open(s.path)
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		s.writeDisabled = true
		return
	}
	data, err := io.ReadAll(io.LimitReader(f, modelBufferHistoryBytes+1))
	closeErr := f.Close()
	if err != nil || closeErr != nil || len(data) > modelBufferHistoryBytes {
		s.writeDisabled = true
		return
	}
	var encoded []json.RawMessage
	d := json.NewDecoder(bytes.NewReader(data))
	if d.Decode(&encoded) != nil || encoded == nil || len(encoded) > 128 {
		s.writeDisabled = true
		return
	}
	var trailing json.RawMessage
	if d.Decode(&trailing) != io.EOF {
		s.writeDisabled = true
		return
	}
	seen := map[string]bool{}
	var recovered []modelBufferStoredSample
	for _, raw := range encoded {
		var sample modelBufferStoredSample
		// Fold the former on-disk pin at this existing decode boundary only.
		// It is a reported build claim, never a content identity.
		var stored struct {
			ModelIdentity string          `json:"modelIdentity"`
			Observation   json.RawMessage `json:"observation"`
		}
		if modelconfig.DecodeStrict(raw, &stored) != nil {
			s.writeDisabled = true
			return
		}
		sample.ModelIdentity = stored.ModelIdentity
		var fields map[string]json.RawMessage
		if json.Unmarshal(stored.Observation, &fields) != nil || fields == nil {
			s.writeDisabled = true
			return
		}
		if previous, ok := fields["upstreamPin"]; ok {
			var pin, assurance string
			if modelconfig.DecodeStrict(previous, &pin) != nil || pin != "0ec4027a2a0183de87c007069adddc06e665f7de" || fields["buildId"] != nil || fields["reportedBuild"] != nil || modelconfig.DecodeStrict(fields["buildAssurance"], &assurance) != nil || assurance != "recognized-self-reported-pin" {
				s.writeDisabled = true
				return
			}
			delete(fields, "upstreamPin")
			fields["reportedBuild"] = json.RawMessage(`"b12020-0ec4027a2"`)
			fields["buildAssurance"] = json.RawMessage(`"recognized-output"`)
			folded, err := json.Marshal(fields)
			if err != nil {
				s.writeDisabled = true
				return
			}
			stored.Observation = folded
		}
		if modelconfig.DecodeStrict(stored.Observation, &sample.Observation) != nil || validateModelBufferSample(sample) != nil || seen[sample.Observation.SampleID] {
			s.writeDisabled = true
			return
		}
		seen[sample.Observation.SampleID] = true
		sample.Observation.Current = false
		recovered = append(recovered, sample)
	}
	s.samples = s.retained(recovered, s.now())
}

func (s *modelBufferStore) retained(samples []modelBufferStoredSample, now time.Time) []modelBufferStoredSample {
	retained := make([]modelBufferStoredSample, 0, len(samples))
	for _, sample := range samples {
		at, err := time.Parse(time.RFC3339Nano, sample.Observation.RecordedAt)
		if err != nil || now.Sub(at) > modelBufferHistoryAge {
			continue
		}
		sample.Observation = sample.Observation.Clone()
		sample.Observation.Current = false
		retained = append(retained, sample)
	}
	slices.SortStableFunc(retained, func(a, b modelBufferStoredSample) int {
		at, _ := time.Parse(time.RFC3339Nano, a.Observation.RecordedAt)
		bt, _ := time.Parse(time.RFC3339Nano, b.Observation.RecordedAt)
		if n := at.Compare(bt); n != 0 {
			return n
		}
		if a.Observation.SampleID < b.Observation.SampleID {
			return -1
		}
		if a.Observation.SampleID > b.Observation.SampleID {
			return 1
		}
		return 0
	})
	for len(retained) > 0 {
		data, err := json.Marshal(retained)
		// Four times encoded bytes conservatively charges retained strings,
		// slices and records, alongside the independent encoded file budget.
		if err == nil && len(retained) <= 128 && len(data) <= modelBufferHistoryBytes && len(data)*4 <= 16<<20 {
			break
		}
		retained[0] = modelBufferStoredSample{}
		retained = retained[1:]
	}
	return slices.Clone(retained)
}

func (s *modelBufferStore) publish(sample modelBufferStoredSample) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.writeDisabled {
		return "history-unavailable"
	}
	if validateModelBufferSample(sample) != nil {
		return "failed"
	}
	candidate := make([]modelBufferStoredSample, 0, len(s.samples)+1)
	for _, old := range s.samples {
		if old.Observation.SampleID != sample.Observation.SampleID {
			candidate = append(candidate, old)
		}
	}
	sample.Observation.Current = false
	sample.Observation.Persistence = "saved"
	candidate = append(candidate, sample)
	candidate = s.retained(candidate, s.now())
	disk := make([]modelBufferStoredSample, len(candidate))
	for i, v := range candidate {
		disk[i] = v
		disk[i].Observation = v.Observation.Clone()
		disk[i].Observation.Persistence = "uncertain"
	}
	data, err := json.Marshal(disk)
	if err != nil || len(data) > modelBufferHistoryBytes {
		return "failed"
	}
	// The existing owner creates a private base directory. Refuse a parent
	// with group/other access instead of silently changing unrelated state.
	info, err := os.Stat(filepath.Dir(s.path))
	if err != nil || !info.IsDir() || runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		return "failed"
	}
	err = s.replacer.WriteFile(s.path, data, 0o600)
	if err != nil && !errors.Is(err, durablefile.ErrReplaced) {
		return "failed"
	}
	s.samples = candidate
	defer s.publishView()
	if err != nil {
		for i := range s.samples {
			if s.samples[i].Observation.SampleID == sample.Observation.SampleID {
				s.samples[i].Observation.Persistence = "uncertain"
			}
		}
		return "uncertain"
	}
	return "saved"
}

func modelBufferHistoryBuildReason(recorded, current string) string {
	if recorded != "" && recorded == current {
		return ""
	}
	if current == "" && recorded != "" {
		return "source-unavailable"
	}
	return "earlier-build"
}

// Read only the immutable view. Diagnostic history's earlier-build fallback
// cannot supply a replacement for the current applied configuration.
func (s *modelBufferStore) compatibleCPUWeights(key, buildID, hostPool string) (low, high uint64, sampleID string, ok bool) {
	if s == nil || key == "" || buildID == "" || hostPool == "" {
		return
	}
	view := s.view.Load()
	if view == nil || view.unavailable {
		return
	}
	now := time.Now()
	if s.now != nil {
		now = s.now()
	}
	var bestAt time.Time
	for _, sample := range view.samples {
		o := sample.Observation
		if o.ConfigurationKey != key || o.BuildID != buildID || validateModelBufferSample(sample) != nil {
			continue
		}
		at, err := time.Parse(time.RFC3339Nano, o.RecordedAt)
		if err != nil || now.Before(at) || now.Sub(at) > modelBufferHistoryAge {
			continue
		}
		l, h, compatible := modelBufferCPUWeightInterval(o, hostPool)
		if !compatible {
			continue
		}
		if !ok || at.After(bestAt) || at.Equal(bestAt) && o.SampleID > sampleID {
			low, high, sampleID, ok, bestAt = l, h, o.SampleID, true, at
		}
	}
	return
}

// The caller validates the stored sample once. Reuse Task14's correspondence,
// then check the CPU interval and current physical host independently.
func modelBufferCPUWeightInterval(o modelconfig.BufferObservation, hostPool string) (uint64, uint64, bool) {
	// Capture's device-diagnostic reason does not invalidate its independently
	// compatible CPU weights; correspondence and physical host are checked below.
	if hostPool == "" || o.Status == "unknown" || o.Reason != "" && o.Reason != "unmapped-device" || len(o.Reasons) != 0 || o.BuildAssurance != "recognized-output" || o.Persistence == "history-unavailable" {
		return 0, 0, false
	}
	var stored modelconfig.BufferComparison
	count := 0
	for _, c := range o.Comparisons {
		if c.Component == "weights" {
			stored = c
			count++
		}
	}
	if count != 1 || stored.Status != "compatible" {
		return 0, 0, false
	}
	if !slices.Contains(modelBufferComparisons(o), stored) {
		return 0, 0, false
	}
	var low, high uint64
	found := false
	for _, row := range o.Rows {
		if row.Component != "model" {
			continue
		}
		if row.Label != "CPU" || row.Role != "model" || row.PoolIdentity != hostPool || row.DeviceBinding != "" || row.Phase != "load" || row.IntervalStatus != "bounded" {
			return 0, 0, false
		}
		l, le := strconv.ParseUint(row.LowerBytes, 10, 64)
		h, he := strconv.ParseUint(row.UpperBytes, 10, 64)
		if le != nil || he != nil || l > h || l > ^uint64(0)-low || h > ^uint64(0)-high {
			return 0, 0, false
		}
		low, high, found = low+l, high+h, true
	}
	return low, high, found && stored.LowerBytes == strconv.FormatUint(low, 10) && stored.UpperBytes == strconv.FormatUint(high, 10)
}

// reportedBuild is the build that the router itself reported (props
// build_info) in the newest retained sample for this exact native content, or
// "" when no retained sample has one.
func (s *modelBufferStore) reportedBuild(buildID string) string {
	if s == nil || buildID == "" {
		return ""
	}
	view := s.view.Load()
	if view == nil || view.unavailable {
		return ""
	}
	for i := len(view.samples) - 1; i >= 0; i-- {
		if o := view.samples[i].Observation; o.BuildID == buildID && o.ReportedBuild != "" {
			return o.ReportedBuild
		}
	}
	return ""
}

func (s *modelBufferStore) newest(model, buildID string) modelconfig.BufferObservation {
	view := s.view.Load()
	if view == nil || view.unavailable {
		return modelconfig.BufferObservation{Status: "unknown", Reason: "history-unavailable", Persistence: "history-unavailable"}
	}
	now := s.now()
	var earlier *modelconfig.BufferObservation
	for i := len(view.samples) - 1; i >= 0; i-- {
		sample := view.samples[i]
		at, err := time.Parse(time.RFC3339Nano, sample.Observation.RecordedAt)
		if sample.ModelIdentity != model || err != nil || now.Sub(at) > modelBufferHistoryAge {
			continue
		}
		o := sample.Observation.Clone()
		o.Current = false
		if now.Before(at) {
			o.Status = "partial"
			o.Reason = "clock-reversal"
		}
		reason := modelBufferHistoryBuildReason(o.BuildID, buildID)
		if reason == "" {
			return o
		}
		if earlier == nil {
			if o.Reason != "clock-reversal" {
				o.Reason = reason
			}
			earlier = &o
		}
	}
	if earlier != nil {
		return *earlier
	}
	return modelconfig.BufferObservation{Status: "unknown", Reason: "no-matching-sample"}
}
