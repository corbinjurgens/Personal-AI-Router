// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"log/slog"
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
	if err := b.settingsWorkerCall(ctx, "engine:preview-launch", settings.WorkerRequest[settings.Request]{Request: settings.Request{
		Engine: engine, Settings: record.Snapshot.Settings.Clone(), Format: "pair-launch-v1",
	}}, &preview); err != nil {
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
	profile, ok := b.frontedEngineProfile(engine)
	if !ok {
		return false
	}
	state := b.engineProxy(profile)
	state.backendPort.Store(int32(config.ServerPort))
	b.setExplicitSettingsProxyPort(profile, config.ProxyPort)
	return true
}

func (b *Broker) setExplicitSettingsProxyPort(profile engineProxyProfile, port int) {
	state := b.engineProxy(profile)
	state.explicitSettings.Store(true)
	state.startupPort.Store(int32(port))
}

// recoverSettingsProxy repairs stopped-engine availability through the same
// supervisor used by Serve and policy changes. Call off engineConfigMu and
// before Configure: facade/policy replay may need that mutex. A recovering
// supervisor keeps its budget and backoff; a fresh request may renew exhaustion.
func (b *Broker) recoverSettingsProxy(ctx context.Context, profile engineProxyProfile, port int) error {
	if proxy := b.settingsProxy(profile.Name); proxy != nil && b.proxyProcessHandle() == proxy {
		if ready, _ := proxy.Status(profile.Name); ready {
			return nil // Configure retains normal stop, swap, Resume ordering.
		}
	}
	b.proxySupMu.Lock()
	defer b.proxySupMu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if b.proxySupClosing.Load() || b.proxySupPhase != proxySupLive {
		return fmt.Errorf("proxy supervision is not available for settings recovery")
	}
	if !b.proxyEnabled(profile) {
		return fmt.Errorf("proxy is disabled or its binary is unresolved")
	}
	pp := b.proxyProcessHandle()
	sup := b.getProxySupervisor()
	if sup != nil && !b.proxySupExhausted.Load() && (pp == nil || !pp.bringUpDone.Load()) {
		return fmt.Errorf("proxy is recovering; retry after supervision completes")
	}
	if pp == nil {
		if sup != nil {
			b.stopProxySupervisorLocked()
		}
		if err := b.startProxySupervisorLocked(); err != nil {
			return fmt.Errorf("recover proxy: %w", err)
		}
		pp = b.proxyProcessHandle()
	}
	if pp == nil {
		return fmt.Errorf("proxy unavailable after recovery")
	}
	if ready, _ := pp.Status(profile.Name); !ready || b.settingsProxy(profile.Name) != pp {
		// A live shared process keeps its sibling facades and policy state.
		// Publish before enable so its ready event finds the proper handle.
		b.workersMu.Lock()
		if b.proxyProc != pp {
			b.workersMu.Unlock()
			return fmt.Errorf("proxy changed before facade recovery")
		}
		b.engineProxy(profile).proxy = pp
		b.workersMu.Unlock()
		bringUp, cancel := b.proxyBringUpContext()
		stop := context.AfterFunc(ctx, cancel)
		err := b.enableEngineFacade(bringUp, pp, profile)
		stop()
		cancel()
		if err != nil && !facadeCameUpAnyway(pp, profile.Name, err) {
			pp.readyMu.Lock()
			delete(pp.facadeState, profile.Name)
			pp.readyMu.Unlock()
			b.workersMu.Lock()
			if b.proxyProc == pp && b.engineProxy(profile).proxy == pp {
				b.engineProxy(profile).proxy = nil
			}
			b.workersMu.Unlock()
			return err
		}
	}
	if ready, _ := pp.Status(profile.Name); ready && b.settingsProxy(profile.Name) == pp && b.proxyProcessHandle() == pp {
		// The process now hosts a facade even if its subsequent durable port
		// write fails. A listener policy change must not stop this process.
		pp.listenerOnly.Store(false)
	}
	// Enable may be idempotent on a different port and does not persist a
	// built-in port. The addressed setter both converges and saves the choice,
	// and checks readiness and process identity before reporting success.
	return b.rebindSettingsProxy(profile.Name, port)
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
			if modelSettingsNeedsReconciliation(record) && (record.Model.Stage != "activation-pending" || record.Receipts[record.Snapshot.RequestID].Phase == "failed") {
				// Failed/uncertain work requires fresh local repair. A pending
				// activation can retry the existing frozen operation on startup.
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
