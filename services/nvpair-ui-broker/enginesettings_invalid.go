// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
)

// codeSettingsRecordInvalid refuses settings work for an engine whose saved
// settings journal entry is invalid. engine:reset-settings-record recovers it.
const codeSettingsRecordInvalid = -32016

// invalidEngineSettings is one journal entry that failed to decode or validate.
// Every journal save writes Raw back with the same content (compacted JSON). Model reports that the
// entry may hold an unfinished model operation, whose policy reservation,
// drain and startup-restoration fences stay in force until the owner resets it.
type invalidEngineSettings struct {
	Raw   json.RawMessage
	Cause string
	Model bool
}

type invalidEngineSettingsError struct {
	engine string
	cause  string
}

func (e *invalidEngineSettingsError) Error() string {
	return fmt.Sprintf("saved settings for engine %q are invalid (%s); settings for this engine are blocked until its saved settings record is reset on this device", e.engine, e.cause)
}

func decodeStoredEngineSettings(raw json.RawMessage) (*engineSettingsRecord, error) {
	var record *engineSettingsRecord
	if err := json.Unmarshal(raw, &record); err != nil {
		return nil, fmt.Errorf("entry could not be decoded: %w", err)
	}
	if record == nil {
		return nil, fmt.Errorf("entry is empty")
	}
	if err := validateStoredModelSettings(record); err != nil {
		return nil, fmt.Errorf("invalid model settings: %w", err)
	}
	return record, nil
}

// storedModelOperationPresent reads an invalid entry only far enough to tell
// whether it carries an unfinished model operation. A model subrecord whose
// stage reads "completed" holds no reservation; anything it cannot read counts
// as unfinished.
func storedModelOperationPresent(raw json.RawMessage) bool {
	var probe struct {
		Model json.RawMessage `json:"model"`
	}
	if json.Unmarshal(raw, &probe) != nil {
		return true
	}
	model := bytes.TrimSpace(probe.Model)
	if len(model) == 0 || bytes.Equal(model, []byte("null")) {
		return false
	}
	var stage struct {
		Stage string `json:"stage"`
	}
	return json.Unmarshal(model, &stage) != nil || stage.Stage != "completed"
}

// engineSettingsEntryErrorLocked refuses settings work for one engine whose
// journal entry is invalid. The caller holds engineConfigMu after a load.
func (b *Broker) engineSettingsEntryErrorLocked(engine string) error {
	if entry, ok := b.engineSettingsInvalid[engine]; ok {
		return &invalidEngineSettingsError{engine: engine, cause: entry.Cause}
	}
	return nil
}

// invalidModelSettingsLocked returns the first invalid entry, by engine name,
// that may hold model work. Node-wide model fences treat it as pending work.
func (b *Broker) invalidModelSettingsLocked() error {
	names := make([]string, 0, len(b.engineSettingsInvalid))
	for engine, entry := range b.engineSettingsInvalid {
		if entry.Model {
			names = append(names, engine)
		}
	}
	if len(names) == 0 {
		return nil
	}
	slices.Sort(names)
	return b.engineSettingsEntryErrorLocked(names[0])
}

type resetSettingsRecordParams struct {
	NodeID string `json:"nodeId,omitempty"`
	Engine string `json:"engine"`
}

type resetSettingsRecordResult struct {
	Engine string `json:"engine"`
	// Backup is the file that keeps the discarded entry.
	Backup string `json:"backup"`
}

func (b *Broker) handleResetSettingsRecord(msg *Message) {
	var p resetSettingsRecordParams
	if err := json.Unmarshal(msg.Params, &p); err != nil || !engines.ValidRoutingID(p.Engine) {
		_ = b.codec.RespondError(msg.ID, -32602, "invalid reset settings record request")
		return
	}
	result, err := b.resetSettingsRecord(p)
	if err != nil {
		_ = b.codec.RespondError(msg.ID, engineSettingsErrorCode(err), err.Error())
		return
	}
	_ = b.codec.Respond(msg.ID, result)
}

// resetSettingsRecord discards one invalid journal entry at the owner's
// request. The entry is first copied to a backup beside the journal; native
// engine configuration and node policy are not changed. The next settings read
// starts a fresh record from the engine's current native configuration.
func (b *Broker) resetSettingsRecord(p resetSettingsRecordParams) (resetSettingsRecordResult, error) {
	if p.NodeID != "" && p.NodeID != b.nodeID {
		return resetSettingsRecordResult{}, &invalidParamsError{fmt.Errorf("saved settings records can be reset only on their own device")}
	}
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	b.engineConfigMu.Lock()
	result, restore, err := b.resetSettingsRecordLocked(p.Engine)
	b.engineConfigMu.Unlock()
	if err != nil {
		return resetSettingsRecordResult{}, err
	}
	if restore && b.facadeStartupReady() {
		// The entry fenced startup restoration. Run it now, as startup would
		// have; before facade startup, startup itself still restores.
		go b.restoreInstalledEngines(b.getEngineMgr())
	}
	return result, nil
}

func (b *Broker) resetSettingsRecordLocked(engine string) (resetSettingsRecordResult, bool, error) {
	if err := b.loadEngineSettingsLocked(); err != nil {
		return resetSettingsRecordResult{}, false, err
	}
	entry, ok := b.engineSettingsInvalid[engine]
	if !ok {
		return resetSettingsRecordResult{}, false, &invalidParamsError{fmt.Errorf("engine %q has no invalid saved settings record", engine)}
	}
	path, err := b.engineSettingsPath()
	if err != nil {
		return resetSettingsRecordResult{}, false, err
	}
	stamp := time.Now().UTC().Format("20060102T150405.000000000Z")
	backup := strings.TrimSuffix(path, filepath.Ext(path)) + "." + engine + "." + stamp + ".invalid.json"
	data, err := json.Marshal(struct {
		Engine string          `json:"engine"`
		Cause  string          `json:"cause"`
		Reset  string          `json:"resetAt"`
		Entry  json.RawMessage `json:"entry"`
	}{engine, entry.Cause, stamp, entry.Raw})
	if err != nil {
		return resetSettingsRecordResult{}, false, err
	}
	if err := os.MkdirAll(filepath.Dir(backup), 0o700); err != nil {
		return resetSettingsRecordResult{}, false, err
	}
	if err := b.engineSettingsReplacer.WriteFile(backup, data, 0o600); err != nil && !errors.Is(err, durablefile.ErrReplaced) {
		return resetSettingsRecordResult{}, false, fmt.Errorf("saved settings record was kept; its backup could not be written: %w", err)
	}
	delete(b.engineSettingsInvalid, engine)
	if err := b.saveEngineSettingsLocked(); err != nil && !errors.Is(err, durablefile.ErrReplaced) {
		b.engineSettingsInvalid[engine] = entry
		return resetSettingsRecordResult{}, false, fmt.Errorf("saved settings record was kept; the journal could not be saved: %w", err)
	}
	slog.Warn("invalid settings journal entry reset by the owner", "engine", engine, "backup", backup)
	restore := false
	if entry.Model {
		r := b.nodePolicyRuntime()
		r.mu.Lock()
		delete(r.drains, engine)
		r.mu.Unlock()
		b.schedulePolicyPush(pushDrainPrefix + engine)
		b.signalModelConfig()
		restore = b.invalidModelSettingsLocked() == nil
		for name := range b.engineSettings {
			restore = restore && !b.modelSettingsPendingLocked(name)
		}
	}
	return resetSettingsRecordResult{Engine: engine, Backup: backup}, restore, nil
}
