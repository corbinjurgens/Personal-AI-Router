// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
)

const modelMemoryProfilesBytes = 1 << 20

type modelMemoryProfileRecord struct {
	Engine  string                    `json:"engine"`
	ModelID string                    `json:"modelId"`
	Profile modelconfig.MemoryProfile `json:"profile"`
}
type modelMemoryProfiles struct {
	mu          sync.Mutex
	path        string
	records     []modelMemoryProfileRecord
	unavailable bool
}

func newModelMemoryProfiles(base string) *modelMemoryProfiles {
	s := &modelMemoryProfiles{path: filepath.Join(base, "model-memory-profiles.json")}
	info, err := os.Lstat(base)
	if base == "" || err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 || runtime.GOOS != "windows" && info.Mode().Perm()&0077 != 0 {
		s.unavailable = true
		return s
	}
	entry, err := os.Lstat(s.path)
	if os.IsNotExist(err) {
		return s
	}
	if err != nil || !entry.Mode().IsRegular() {
		s.unavailable = true
		return s
	}
	f, err := durablefile.Open(s.path)
	if err != nil {
		s.unavailable = true
		return s
	}
	data, err := io.ReadAll(io.LimitReader(f, modelMemoryProfilesBytes+1))
	closeErr := f.Close()
	if err != nil || closeErr != nil || len(data) > modelMemoryProfilesBytes {
		s.unavailable = true
		return s
	}
	// DecodeStrict's public frame ceiling is smaller than this store's ceiling.
	var raw []json.RawMessage
	if json.Unmarshal(data, &raw) != nil || raw == nil || len(raw) > 256 {
		s.unavailable = true
		return s
	}
	seen := map[string]bool{}
	for _, v := range raw {
		var r modelMemoryProfileRecord
		if modelconfig.DecodeStrict(v, &r) != nil || r.Profile.Validate() != nil || (modelconfig.InspectRequest{Engine: r.Engine, ModelID: r.ModelID}).Validate() != nil || r.Profile.Current {
			s.unavailable = true
			return s
		}
		key := modelMemoryProfileKey(r)
		if seen[key] {
			s.unavailable = true
			return s
		}
		seen[key] = true
		s.records = append(s.records, r)
	}
	return s
}
func modelMemoryProfileKey(r modelMemoryProfileRecord) string {
	return r.Engine + "\x00" + r.ModelID + "\x00" + r.Profile.ConfigurationKey + "\x00" + r.Profile.Kind
}
func (s *modelMemoryProfiles) update(engine, model, key string, p *modelconfig.MemoryProfile) error {
	return s.updateInspection(engine, model, key, p, nil)
}

// Prepare the exact bounded reply from the same candidate before publishing.
// A post-publication payload failure must never masquerade as a rejected save.
func (s *modelMemoryProfiles) updateInspection(engine, model, key string, p *modelconfig.MemoryProfile, reply *modelconfig.InspectResult) error {
	if s == nil {
		return fmt.Errorf("memory profile store unavailable")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.unavailable {
		return fmt.Errorf("memory profile store unavailable; existing file preserved")
	}
	entry, entryErr := os.Lstat(s.path)
	if entryErr != nil && !os.IsNotExist(entryErr) || entryErr == nil && !entry.Mode().IsRegular() {
		s.unavailable = true
		return fmt.Errorf("memory profile store unavailable; existing file preserved")
	}
	candidate := make([]modelMemoryProfileRecord, 0, len(s.records)+1)
	identity := modelMemoryProfileKey(modelMemoryProfileRecord{Engine: engine, ModelID: model, Profile: modelconfig.MemoryProfile{ConfigurationKey: key, Kind: "stated"}})
	for _, r := range s.records {
		if modelMemoryProfileKey(r) != identity {
			candidate = append(candidate, r)
		}
	}
	if p != nil {
		r := modelMemoryProfileRecord{Engine: engine, ModelID: model, Profile: p.Clone()}
		r.Profile.Current = false
		candidate = append(candidate, r)
	}
	// Oldest history is evicted; the explicit current write is retained.
	slices.SortStableFunc(candidate, func(a, b modelMemoryProfileRecord) int {
		at, _ := time.Parse(time.RFC3339Nano, a.Profile.RecordedAt)
		bt, _ := time.Parse(time.RFC3339Nano, b.Profile.RecordedAt)
		return at.Compare(bt)
	})
	for len(candidate) > 256 {
		index := 0
		if p != nil && modelMemoryProfileKey(candidate[0]) == identity {
			index = 1
		}
		candidate = slices.Delete(candidate, index, index+1)
	}
	data, err := json.Marshal(candidate)
	if err != nil || len(data) > modelMemoryProfilesBytes {
		return fmt.Errorf("memory profile store size limit")
	}
	var projected modelconfig.InspectResult
	if reply != nil {
		projected = reply.Clone()
		attachModelMemoryProfiles(&projected, candidate, false)
		if err := completeModelInspectionReply(&projected); err != nil {
			return err
		}
	}
	if err := engineFileReplacer.WriteFile(s.path, data, 0600); err != nil {
		if errors.Is(err, durablefile.ErrReplaced) {
			s.records = candidate
			if reply != nil {
				*reply = projected
			}
			return fmt.Errorf("memory profile saved but durability uncertain; reinspect: %w", durablefile.ErrReplaced)
		}
		s.unavailable = true
		return fmt.Errorf("memory profile persistence unavailable")
	}
	s.records = candidate
	if reply != nil {
		*reply = projected
	}
	return nil
}
func (s *modelMemoryProfiles) attach(r *modelconfig.InspectResult) {
	if s == nil {
		r.MemoryProfiles = nil
		r.MemoryProfilesUnavailable = true
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	attachModelMemoryProfiles(r, s.records, s.unavailable)
}

func attachModelMemoryProfiles(r *modelconfig.InspectResult, records []modelMemoryProfileRecord, unavailable bool) {
	r.MemoryProfiles = nil
	r.MemoryProfilesUnavailable = unavailable
	if unavailable {
		return
	}
	var history []modelconfig.MemoryProfile
	for i := len(records) - 1; i >= 0; i-- {
		v := records[i]
		if v.Engine == r.Engine && v.ModelID == r.ModelID {
			p := v.Profile.Clone()
			p.Current = p.ConfigurationKey == r.ConfigurationKey && r.ConfigurationKey != ""
			if p.Current {
				r.MemoryProfiles = append(r.MemoryProfiles, p)
			} else {
				history = append(history, p)
			}
		}
	}
	r.MemoryProfiles = append(r.MemoryProfiles, history...)
	if len(r.MemoryProfiles) > 8 {
		r.MemoryProfiles = r.MemoryProfiles[:8]
	}
	for modelconfig.CheckPayloadSize(*r) != nil && len(r.MemoryProfiles) > 0 && !r.MemoryProfiles[len(r.MemoryProfiles)-1].Current {
		r.MemoryProfiles = r.MemoryProfiles[:len(r.MemoryProfiles)-1]
	}
}
