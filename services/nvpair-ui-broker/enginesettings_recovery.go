// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"time"

	settings "nvpair-shared/enginesettings"
)

// Convert pending/failed full-command settings before replay, preserving the
// accepted arguments and proxy port. Component launch_args/launch_env need no
// migration because they have always stored only literal user overrides.
func (b *Broker) migrateSettingsArgumentsLocked(ctx context.Context, engine string, record *engineSettingsRecord) error {
	if record == nil {
		return nil
	}
	previous := record.Snapshot.Clone()
	changed, err := b.prepareSettingsArgumentsLocked(ctx, engine, record)
	if err != nil || !changed {
		return err
	}
	if err := b.saveEngineSettingsLocked(); err != nil {
		record.Snapshot = previous
		return err
	}
	return nil
}

// Preparation updates only the supplied record. Apply uses a private copy so
// migration cannot save anything before its final pause admission check.
func (b *Broker) prepareSettingsArgumentsLocked(ctx context.Context, engine string, record *engineSettingsRecord) (bool, error) {
	if record == nil || record.Snapshot.Format != "pair-launch-v1" {
		return false, nil
	}
	var preview settings.Preview
	if err := b.settingsWorkerCall(ctx, "engine:preview-launch", settings.Request{
		Engine: engine, Settings: record.Snapshot.Settings.Clone(), Format: "pair-launch-v1",
	}, &preview); err != nil {
		return false, err
	}
	if len(preview.Errors) != 0 || preview.Conflict != nil {
		return false, fmt.Errorf("saved full-command settings could not be converted; existing settings were preserved")
	}
	previous := record.Snapshot.Clone()
	record.Snapshot.Settings = preview.Settings.Clone()
	record.Snapshot.Format = "pair-arguments-v1"
	record.Snapshot.Revision++
	if previous.AppliedRevision == previous.Revision {
		record.Snapshot.AppliedRevision = record.Snapshot.Revision
	}
	return true, nil
}

func (b *Broker) explicitEngineSettings(engine string) (settings.Config, bool) {
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	return b.explicitEngineSettingsLocked(engine)
}

func (b *Broker) explicitEngineSettingsLocked(engine string) (settings.Config, bool) {
	if err := b.loadEngineSettingsLocked(); err != nil {
		return settings.Config{}, false
	}
	if record := b.engineSettings[engine]; record != nil && record.Explicit {
		return record.Snapshot.Settings.Clone(), true
	}
	return settings.Config{}, false
}

func (b *Broker) prepareExplicitEngineSettings(engine string) bool {
	b.engineConfigMu.Lock()
	loadErr := b.loadEngineSettingsLocked()
	b.engineConfigMu.Unlock()
	if loadErr != nil {
		return true
	} // preserve component stores; recovery reports the journal error
	config, ok := b.explicitEngineSettings(engine)
	if !ok {
		return false
	}
	profile, _ := engineProxyProfileFor(engine)
	state := b.engineProxy(profile)
	state.explicitSettings.Store(true)
	state.managedFacade.Store(false)
	state.backendPort.Store(int32(config.ServerPort))
	state.startupPort.Store(int32(config.ProxyPort))
	if engine == "ollama" {
		b.managedOllamaBackend.Store(0)
		b.syncCurrentEngineOllamaHostAliasReservation()
	}
	return true
}

// Legacy proxy stores did not distinguish user choices from automatic moves.
// Preserve a valid non-colliding saved choice as explicit on first upgrade.
// LM Studio's old 1235 default is excluded, matching its existing migration.
func (b *Broker) migrateLegacyEngineSettings() {
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if b.loadEngineSettingsLocked() != nil || b.getEngineMgr() == nil {
		return
	}
	journal, err := b.engineSettingsPath()
	if err != nil {
		return
	}
	for _, engine := range []string{"ollama", "lmstudio"} {
		name := "proxy-port.json"
		if engine == "lmstudio" {
			name = "lmstudio-proxy-port.json"
		}
		if b.engineSettings[engine] != nil {
			continue
		}
		data, err := os.ReadFile(filepath.Join(filepath.Dir(journal), name))
		if err != nil {
			continue
		}
		var saved struct {
			Port int `json:"port"`
		}
		if json.Unmarshal(data, &saved) != nil || saved.Port < 1 || saved.Port > 65535 || (engine == "lmstudio" && saved.Port == 1235) {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_ = b.settingsWorkerCall(ctx, "engine:status", map[string]string{"engine": engine}, nil)
		var launch settings.LaunchState
		err = b.settingsWorkerCall(ctx, "engine:get-launch", map[string]string{"engine": engine}, &launch)
		if err != nil || !launch.Editable || launch.ServerPort == saved.Port {
			cancel()
			continue
		}
		config := settings.ConfigFromLaunch(launch, saved.Port)
		current := settings.Snapshot{Running: launch.Running, EffectiveServerPort: launch.EffectivePort}
		if proxy := b.settingsProxy(engine); proxy != nil {
			_, current.EffectiveProxyPort = proxy.Status(engine)
		}
		err = b.validateSettingsPortsLocked(ctx, engine, config, current)
		cancel()
		if err != nil {
			continue
		}
		b.engineSettings[engine] = &engineSettingsRecord{Explicit: true, Receipts: map[string]settingsReceipt{}, Snapshot: settings.Snapshot{NodeID: b.nodeID, Engine: engine, Revision: 1, AppliedRevision: 1, Epoch: b.engineSettingsEpoch, Phase: "idle", Settings: config.Clone()}}
		if err := b.saveEngineSettingsLocked(); err != nil {
			delete(b.engineSettings, engine)
			b.engineSettingsError = err
			return
		}
	}
}

func (b *Broker) replayLoadedEngineSettings() {
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if b.engineSettingsLoaded && b.engineSettingsError == nil && len(b.engineSettings) > 0 {
		b.publishSettingsLocked()
	}
}

// Replay accepted operations before automatic engine startup. Component stores
// can be at any phase after a crash; configure revalidates and converges them
// under the same engine lock used by a live Apply, then records its outcome.
func (b *Broker) recoverEngineSettings() bool {
	// Same order a live Apply uses: the operation lock, then the journal lock
	// that runSettingsOperationLocked releases around the engine restart.
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if err := b.loadEngineSettingsLocked(); err != nil {
		if b.knownModelSettingsFenceLocked() {
			slog.Error("engine settings recovery blocked by known model operation", "err", err)
			return false
		}
		slog.Error("settings journal unreadable; model involvement is unknown; preserving ordinary enabled-intent restoration", "err", err)
		return true
	}
	b.reconstructModelSettingsLocked()
	for engine, record := range b.engineSettings {
		if b.modelSettingsPendingLocked(engine) {
			if modelSettingsNeedsReconciliation(record) {
				// Failed/uncertain acceptance and activation-pending work have
				// an explicit fresh local repair path. Keep their fences here.
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), settings.OperationBudget)
			_ = b.runModelSettingsOperationLocked(ctx, engine, record)
			cancel()
			continue
		}
		if record.Snapshot.Phase != "applying" {
			continue
		}
		if err := b.pendingSwitchLocked(engine); err != nil {
			slog.Error("settings replay conflicts with engine switch recovery", "engine", engine, "err", err)
			return false
		}
		ctx, cancel := context.WithTimeout(context.Background(), settings.OperationBudget)
		_ = b.runSettingsOperationLocked(ctx, engine, record)
		cancel()
	}
	// Readiness may arrive before the journal is loaded. Seed the applied
	// identity even when recovery and later live readbacks change no settings.
	b.publishSettingsLocked()
	for engine := range b.engineSettings {
		if b.modelSettingsPendingLocked(engine) {
			return false
		}
	}
	return true
}

// Legacy lifecycle/port RPCs share the node lock. Reconcile their component
// changes into the same revision stream before allowing another Apply.
func (b *Broker) reconcileLegacySettingsLocked() {
	if !b.engineSettingsLoaded || b.engineSettingsError != nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	changed := false
	for engine, record := range b.engineSettings {
		before := record.Snapshot.Clone()
		after, err := b.settingsSnapshotLocked(ctx, engine)
		if err == nil && !reflect.DeepEqual(before, after) {
			changed = true
		}
	}
	if changed {
		b.publishSettingsLocked()
	}
}
