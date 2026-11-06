// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"nvpair-shared/engines"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

// deriveProfileContext returns a detached policy. Only target-resolved per-slot
// context is authority; a profile without a typed load keeps its descriptor bound.
func deriveProfileContext(policy nodepolicy.Policy, loadsByEngine map[string][]modelconfig.ModelLoad) (nodepolicy.Policy, error) {
	data, err := json.Marshal(policy)
	if err != nil {
		return nodepolicy.Policy{}, err
	}
	// Copy without invoking Policy.UnmarshalJSON's unrelated pause normalization.
	type policyCopy nodepolicy.Policy
	var copied policyCopy
	if err := json.Unmarshal(data, &copied); err != nil {
		return nodepolicy.Policy{}, err
	}
	detached := nodepolicy.Policy(copied)
	for i := range detached.Profiles {
		profile := &detached.Profiles[i]
		load := profileLoad(loadsByEngine, profile.Engine, profile.Key)
		if load == nil {
			continue
		}
		if load.Resolved == nil || load.Resolved.PerSlotContext <= 0 {
			return nodepolicy.Policy{}, fmt.Errorf("profile %q has no resolved per-slot context", profile.Key)
		}
		profile.ContextTokens = int(load.Resolved.PerSlotContext)
	}
	return detached, nil
}

func profileLoad(loadsByEngine map[string][]modelconfig.ModelLoad, engine, key string) *modelconfig.ModelLoad {
	loads := loadsByEngine[engine]
	i := slices.IndexFunc(loads, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == key })
	if i < 0 {
		return nil
	}
	return &loads[i]
}

// Caller holds engineConfigMu. These are the same saved settings snapshots the
// coupled operation uses, never running-model observations or a second cache.
func (b *Broker) profileLoadsByEngineLocked() map[string][]modelconfig.ModelLoad {
	loads := make(map[string][]modelconfig.ModelLoad, len(b.engineSettings))
	for engine, record := range b.engineSettings {
		if record != nil {
			loads[engine] = modelconfig.CloneLoads(record.Snapshot.Settings.ModelLoads)
		}
	}
	return loads
}

// prepareProfileLoadsLocked fills gaps through the settings snapshot owner,
// without attaching or saving its private initialization/drift records. The
// caller holds engineConfigMu, but not policy.mu, throughout these worker reads.
// Existing journal records remain the accepted desired-settings authority.
// The returned source check runs under policy.mu immediately before publication.
func (b *Broker) prepareProfileLoadsLocked(ctx context.Context, raw json.RawMessage, stored nodepolicy.Policy) (map[string][]modelconfig.ModelLoad, func() error, error) {
	if err := b.loadEngineSettingsLocked(); err != nil {
		return nil, nil, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return nil, nil, &invalidParamsError{fmt.Errorf("policy must be a JSON object")}
	}
	var engineNames []string
	for _, profile := range stored.Profiles {
		engineNames = append(engineNames, profile.Engine)
	}
	for field, value := range fields {
		if !strings.EqualFold(field, "profiles") {
			continue
		}
		// Read selectors alone: caller-owned metadata and managed fields have
		// not been normalized or validated yet.
		var profiles []struct {
			Engine string `json:"engine"`
		}
		if err := json.Unmarshal(value, &profiles); err != nil {
			return nil, nil, &invalidParamsError{err}
		}
		for _, profile := range profiles {
			engineNames = append(engineNames, profile.Engine)
		}
	}
	slices.Sort(engineNames)
	engineNames = slices.Compact(engineNames)
	loads := b.profileLoadsByEngineLocked()
	b.workersMu.Lock()
	worker, source := b.engineMgr, b.engineMgrSource
	b.workersMu.Unlock()
	prepared := make(map[string]*engineSettingsRecord)
	for _, engine := range engineNames {
		if !engines.ValidRoutingID(engine) {
			return nil, nil, &invalidParamsError{fmt.Errorf("unknown engine %q", engine)}
		}
		if record := b.engineSettings[engine]; record != nil {
			if record.Snapshot.Engine != engine || record.Snapshot.Revision == 0 {
				return nil, nil, fmt.Errorf("profile engine %q snapshot authority is invalid", engine)
			}
			config, err := record.Snapshot.Settings.Canonical()
			if err != nil {
				return nil, nil, fmt.Errorf("profile engine %q snapshot authority is invalid: %w", engine, err)
			}
			loads[engine] = config.ModelLoads
			continue
		}
		if _, ok := b.frontedEngineProfile(engine); !ok {
			return nil, nil, &invalidParamsError{fmt.Errorf("profile engine %q settings authority is unavailable", engine)}
		}
		record, _, err := b.prepareSettingsSnapshotLocked(ctx, engine)
		if err != nil {
			return nil, nil, err
		}
		loads[engine] = modelconfig.CloneLoads(record.Snapshot.Settings.ModelLoads)
		prepared[engine] = record
	}
	// A native configuration may change independently of the broker journal.
	// Compare the complete saved declaration again, ignoring runtime observations.
	for _, engine := range engineNames {
		before := prepared[engine]
		if before == nil {
			continue
		}
		record, _, err := b.prepareSettingsSnapshotLocked(ctx, engine)
		if err != nil {
			return nil, nil, err
		}
		if !before.Snapshot.Settings.Equal(record.Snapshot.Settings) {
			return nil, nil, modelSettingsConflict("native source changed during policy preparation; reload and try again")
		}
	}
	r := &b.nodePolicy
	current := func() error {
		if len(prepared) == 0 {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if !b.engineMgrPairCurrent(worker, source) || r.engineManagerSource != source {
			return modelSettingsConflict("engine source changed during policy preparation; reload and try again")
		}
		select {
		case <-worker.Done():
			return modelSettingsConflict("engine source stopped during policy preparation; reload and try again")
		default:
			return nil
		}
	}
	r.mu.Lock()
	err := current()
	r.mu.Unlock()
	if err != nil {
		return nil, nil, err
	}
	return loads, current, nil
}

// normalizeRawPolicyProfiles runs before Parse or MergePeerSet validates the
// document. Managed fields must not gain authority, or cause validation refusal,
// just because a caller supplied them. Shared nodepolicy still owns validation.
// The caller checks CheckModelListKeys on the original bytes before preprocessing.
func normalizeRawPolicyProfiles(raw json.RawMessage, stored nodepolicy.Policy, loadsByEngine map[string][]modelconfig.ModelLoad) (json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return nil, fmt.Errorf("policy must be a JSON object")
	}
	for key, value := range fields {
		if !strings.EqualFold(key, "profiles") {
			continue
		}
		var entries []map[string]json.RawMessage
		if err := json.Unmarshal(value, &entries); err != nil {
			return nil, err
		}
		profiles := make([]nodepolicy.Profile, len(entries))
		for i, entry := range entries {
			// Decode only the selectors before stripping managed values. This also
			// ignores malformed incoming identity and typed context field types.
			var target struct {
				Key    string `json:"key"`
				Engine string `json:"engine"`
			}
			data, err := json.Marshal(entry)
			if err != nil {
				return nil, err
			}
			if err := json.Unmarshal(data, &target); err != nil {
				return nil, err
			}
			dropFields(entry, "modelIdentity")
			if profileLoad(loadsByEngine, target.Engine, target.Key) != nil {
				dropFields(entry, "contextTokens")
			}
			data, err = json.Marshal(entry)
			if err != nil {
				return nil, err
			}
			if err := json.Unmarshal(data, &profiles[i]); err != nil {
				return nil, err
			}
			for _, previous := range stored.Profiles {
				if previous.Key == profiles[i].Key {
					profiles[i].ModelIdentity = previous.ModelIdentity
					break
				}
			}
		}
		normalized, err := deriveProfileContext(nodepolicy.Policy{Profiles: profiles}, loadsByEngine)
		if err != nil {
			return nil, err
		}
		if entries == nil {
			normalized.Profiles = nil
		}
		fields[key], err = json.Marshal(normalized.Profiles)
		if err != nil {
			return nil, err
		}
	}
	return json.Marshal(fields)
}

// Run after MergePeerSet so its inference-lock refusal retains precedence.
func protectProfileTargets(stored, candidate nodepolicy.Policy, loadsByEngine map[string][]modelconfig.ModelLoad) error {
	for _, profile := range candidate.Profiles {
		for _, previous := range stored.Profiles {
			if previous.Key != profile.Key {
				continue
			}
			if previous.Engine != profile.Engine || profileLoad(loadsByEngine, previous.Engine, previous.Key) != nil && previous.Model != profile.Model {
				return fmt.Errorf("A profile keeps its model; create a new profile.")
			}
		}
	}
	return nil
}
