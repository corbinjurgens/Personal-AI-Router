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
	Engine     string                    `json:"engine"`
	ModelID    string                    `json:"modelId"`
	ProfileKey string                    `json:"profileKey,omitempty"`
	Profile    modelconfig.MemoryProfile `json:"profile"`
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
		if modelconfig.DecodeStrict(v, &r) != nil || validateModelMemoryProfileRecord(r) != nil {
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
	profile := r.ModelID
	identity := "configuration:" + r.Profile.ConfigurationKey
	bucket := r.Profile.Kind
	if r.Profile.Kind == "measured" {
		profile = r.ProfileKey
		if r.Profile.ExecutionKey != "" {
			identity = "execution:" + r.Profile.ExecutionKey
		}
		if r.Profile.Measured != nil && r.Profile.Measured.Outcome != "completed" {
			bucket += ":unsuccessful"
		} else {
			bucket += ":completed"
		}
	}
	return r.Engine + "\x00" + profile + "\x00" + identity + "\x00" + bucket
}

func validateModelMemoryProfileRecord(r modelMemoryProfileRecord) error {
	if err := (modelconfig.InspectRequest{Engine: r.Engine, ModelID: r.ModelID, ProfileKey: r.ProfileKey}).Validate(); err != nil {
		return err
	}
	if err := r.Profile.Validate(); err != nil {
		return err
	}
	if r.Profile.Current || r.Profile.Kind == "measured" && (r.Engine != "llamacpp" || r.ProfileKey == "") {
		return fmt.Errorf("memory profile: invalid stored scope or current flag")
	}
	return modelconfig.CheckPayloadSize(r)
}

// recordMeasured persists detached evidence only. The diagnostic operation must
// first authenticate the producing run and its complete identity; this store
// does not confer run authority and is not a public record method.
func (s *modelMemoryProfiles) recordMeasured(engine, model, profileKey string, measured modelconfig.MeasuredProfile) error {
	p := modelconfig.MemoryProfile{
		ConfigurationKey: measured.ConfigurationKey, ExecutionKey: measured.ExecutionKey,
		Kind: "measured", RecordedAt: measured.RecordedAt, NativeBuildID: measured.NativeBuildID,
	}
	detail := measured.Clone()
	p.Measured = &detail
	if measured.ExecutionKey == "" && measured.Outcome == "completed" {
		for _, pool := range measured.Pools {
			switch pool.Kind {
			case "host":
				p.HostBytes = pool.PeakUsedBytes
			case "device":
				p.DeviceBytes = pool.PeakUsedBytes
			case "shared":
				p.SharedBytes = pool.PeakUsedBytes
			}
		}
		var pools []string
		for _, placement := range measured.Placement {
			pools = append(pools, placement.Kind)
		}
		p.CoversPools = measured.Covers(pools)
	}
	r := modelMemoryProfileRecord{Engine: engine, ModelID: model, ProfileKey: profileKey, Profile: p}
	if err := validateModelMemoryProfileRecord(r); err != nil {
		return err
	}
	if s == nil {
		return fmt.Errorf("memory profile store unavailable")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.unavailable {
		return fmt.Errorf("memory profile store unavailable; existing file preserved")
	}
	key := modelMemoryProfileKey(r)
	candidate := make([]modelMemoryProfileRecord, 0, len(s.records)+1)
	for _, existing := range s.records {
		if modelMemoryProfileKey(existing) == key {
			before, _ := time.Parse(time.RFC3339Nano, existing.Profile.RecordedAt)
			after, _ := time.Parse(time.RFC3339Nano, p.RecordedAt)
			if after.Before(before) {
				return nil
			}
			continue
		}
		candidate = append(candidate, existing)
	}
	candidate = append(candidate, r)
	return s.publishLocked(candidate, key, nil, true)
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
		if p.Kind != "stated" || p.ConfigurationKey != key {
			return fmt.Errorf("memory profile: stated configuration mismatch")
		}
		candidate = append(candidate, r)
	}
	return s.publishLocked(candidate, identity, reply, false)
}

func (s *modelMemoryProfiles) publishLocked(candidate []modelMemoryProfileRecord, identity string, reply *modelconfig.InspectResult, preserveStated bool) error {
	if s.unavailable {
		return fmt.Errorf("memory profile store unavailable; existing file preserved")
	}
	entry, entryErr := os.Lstat(s.path)
	if entryErr != nil && !os.IsNotExist(entryErr) || entryErr == nil && !entry.Mode().IsRegular() {
		s.unavailable = true
		return fmt.Errorf("memory profile store unavailable; existing file preserved")
	}
	for _, r := range candidate {
		if err := validateModelMemoryProfileRecord(r); err != nil {
			return err
		}
	}
	// Oldest history is evicted; the explicit current write is retained.
	slices.SortStableFunc(candidate, func(a, b modelMemoryProfileRecord) int {
		at, _ := time.Parse(time.RFC3339Nano, a.Profile.RecordedAt)
		bt, _ := time.Parse(time.RFC3339Nano, b.Profile.RecordedAt)
		return at.Compare(bt)
	})
	for len(candidate) > 256 {
		index := -1
		for i, r := range candidate {
			if modelMemoryProfileKey(r) != identity && (!preserveStated || r.Profile.Kind != "stated") {
				index = i
				break
			}
		}
		if index < 0 {
			return fmt.Errorf("memory profile store retention limit")
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
	s.attachWithIdentity(r, "", nil)
}

// attachWithIdentity accepts freshly proved complete target identity from the
// manager. A missing identity leaves measured history visible but never current.
func (s *modelMemoryProfiles) attachWithIdentity(r *modelconfig.InspectResult, profileKey string, current *modelconfig.DiagnosticIdentity) {
	if s == nil {
		r.MemoryProfiles = nil
		r.MemoryProfilesUnavailable = true
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	attachModelMemoryProfilesWithIdentity(r, s.records, s.unavailable, profileKey, current)
}

func attachModelMemoryProfiles(r *modelconfig.InspectResult, records []modelMemoryProfileRecord, unavailable bool) {
	attachModelMemoryProfilesWithIdentity(r, records, unavailable, "", nil)
}

func attachModelMemoryProfilesWithIdentity(r *modelconfig.InspectResult, records []modelMemoryProfileRecord, unavailable bool, profileKey string, current *modelconfig.DiagnosticIdentity) {
	r.MemoryProfiles = nil
	r.MemoryProfilesUnavailable = unavailable
	if unavailable {
		return
	}
	if profileKey == "" {
		profileKey = r.ModelID
	}
	var history []modelconfig.MemoryProfile
	for i := len(records) - 1; i >= 0; i-- {
		v := records[i]
		if v.Engine == r.Engine && v.ModelID == r.ModelID {
			if v.Profile.Kind == "measured" && v.ProfileKey != profileKey {
				continue
			}
			p := v.Profile.Clone()
			p.Current = p.ConfigurationKey == r.ConfigurationKey && r.ConfigurationKey != ""
			if p.Kind == "measured" {
				p.Current = current != nil && p.Measured.Matches(*current) &&
					p.ConfigurationKey == r.ConfigurationKey && p.ExecutionKey == r.ExecutionKey &&
					p.NativeBuildID == r.NativeBuildID && slices.Equal(p.Measured.Artifacts, r.Artifacts)
			}
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
	for modelconfig.CheckPayloadSize(*r) != nil && len(r.MemoryProfiles) > 0 {
		index := len(r.MemoryProfiles) - 1
		// Measured detail can occupy almost an entire frame. Omit advisory
		// evidence before allowing it to break inspection, retaining current
		// stated figures just as the existing statement path does.
		for index >= 0 && r.MemoryProfiles[index].Current && r.MemoryProfiles[index].Kind == "stated" {
			index--
		}
		if index < 0 {
			break
		}
		r.MemoryProfiles = slices.Delete(r.MemoryProfiles, index, index+1)
	}
}
