// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
)

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
		if modelconfig.DecodeStrict(raw, &sample) != nil || validateModelBufferSample(sample) != nil || seen[sample.Observation.SampleID] {
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

func (s *modelBufferStore) newest(model string) modelconfig.BufferObservation {
	view := s.view.Load()
	if view == nil || view.unavailable {
		return modelconfig.BufferObservation{Status: "unknown", Reason: "history-unavailable", Persistence: "history-unavailable"}
	}
	now := s.now()
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
		return o
	}
	return modelconfig.BufferObservation{Status: "unknown", Reason: "no-matching-sample"}
}
