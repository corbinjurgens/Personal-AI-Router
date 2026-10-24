// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/noderec"
)

type settingsReceipt struct {
	Hash     string `json:"hash"`
	Revision uint64 `json:"revision"`
	Phase    string `json:"phase"`
}
type engineSettingsRecord struct {
	Snapshot settings.Snapshot          `json:"snapshot"`
	Previous settings.Config            `json:"previous"`
	Resume   bool                       `json:"resume"`
	Explicit bool                       `json:"explicit"`
	Receipts map[string]settingsReceipt `json:"receipts"`
}

func (r *engineSettingsRecord) clone() *engineSettingsRecord {
	if r == nil {
		return nil
	}
	copy := *r
	copy.Snapshot = r.Snapshot.Clone()
	copy.Previous = r.Previous.Clone()
	copy.Receipts = make(map[string]settingsReceipt, len(r.Receipts))
	for id, receipt := range r.Receipts {
		copy.Receipts[id] = receipt
	}
	return &copy
}

type activeSettingsOperation struct {
	Engine string
	Port   int
}

func settingsID() string                     { var id [16]byte; _, _ = rand.Read(id[:]); return hex.EncodeToString(id[:]) }
func settingsJSON(value any) json.RawMessage { data, _ := json.Marshal(value); return data }

func (b *Broker) engineSettingsPath() (string, error) {
	if base := b.clusterManagerConfigDir(); base != "" {
		return filepath.Join(base, "engine-settings-operations.json"), nil
	}
	return appdir.Path("engine-settings-operations.json")
}

// The journal records accepted desired configuration and operation outcomes.
// Component stores still supply the normal settings baseline. Interrupted or
// failed operations retain the journal's desired values until reconciled.
func (b *Broker) loadEngineSettingsLocked() error {
	if b.engineSettingsLoaded {
		return b.engineSettingsError
	}
	b.engineSettingsLoaded = true
	b.engineSettingsEpoch = settingsID()
	b.engineSettings = make(map[string]*engineSettingsRecord)
	path, err := b.engineSettingsPath()
	if err != nil {
		b.engineSettingsError = err
		return err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err == nil {
		err = json.Unmarshal(data, &b.engineSettings)
	}
	if err != nil || b.engineSettings == nil {
		b.engineSettingsError = fmt.Errorf("settings operation journal could not be read; existing settings were preserved")
		return b.engineSettingsError
	}
	for _, record := range b.engineSettings {
		if record == nil {
			b.engineSettingsError = fmt.Errorf("invalid settings journal")
			return b.engineSettingsError
		}
		record.Snapshot.NodeID = b.nodeID
		record.Snapshot.Epoch = b.engineSettingsEpoch
		record.Snapshot.Sequence = 0
		if record.Receipts == nil {
			record.Receipts = make(map[string]settingsReceipt)
		}
	}
	return nil
}

func (b *Broker) saveEngineSettingsLocked() error {
	// Bound recovery metadata. An evicted retry still has its original stale
	// expected revision, so it cannot overwrite a later configuration.
	for _, record := range b.engineSettings {
		for len(record.Receipts) > 256 {
			oldest := ""
			var revision uint64 = ^uint64(0)
			for id, receipt := range record.Receipts {
				if id != record.Snapshot.RequestID && receipt.Phase != "applying" && receipt.Revision <= revision {
					oldest = id
					revision = receipt.Revision
				}
			}
			if oldest == "" {
				break
			}
			delete(record.Receipts, oldest)
		}
	}
	path, err := b.engineSettingsPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.Marshal(b.engineSettings)
	if err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".engine-settings-*")
	if err != nil {
		return err
	}
	tmp := file.Name()
	defer os.Remove(tmp)
	if err = file.Chmod(0o600); err == nil {
		_, err = file.Write(data)
	}
	if err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (b *Broker) settingsWorkerCall(ctx context.Context, method string, params any, target any) error {
	w := b.getEngineMgr()
	if w == nil {
		return fmt.Errorf("engine manager unavailable")
	}
	result, rpcErr, err := w.CallNoTimeout(ctx, method, settingsJSON(params))
	if err != nil {
		return err
	}
	if rpcErr != nil {
		return fmt.Errorf("%s", rpcErr.Message)
	}
	if target != nil {
		return json.Unmarshal(result, target)
	}
	return nil
}

func (b *Broker) settingsProxy(engine string) *proxyProcess {
	profile, ok := b.frontedEngineProfile(engine)
	if !ok {
		return nil
	}
	return b.engineProxyHandle(profile)
}

func isEngineDiscoveryService(service noderec.ServiceKey) bool {
	for _, profile := range engineProxyProfiles {
		if profile.DiscoveryService == service {
			return true
		}
	}
	return false
}

func (b *Broker) settingsSnapshotLocked(ctx context.Context, engine string) (settings.Snapshot, error) {
	record, save, err := b.prepareSettingsSnapshotLocked(ctx, engine)
	if err != nil {
		return settings.Snapshot{}, err
	}
	previous := b.engineSettings[engine].clone()
	if current := b.engineSettings[engine]; current != nil {
		// Readers can run while an accepted apply waits on the worker. Keep
		// that operation's record pointer attached to the journal.
		*current = *record
		record = current
	} else {
		b.engineSettings[engine] = record
	}
	if save {
		if err := b.saveEngineSettingsLocked(); err != nil {
			if previous == nil {
				delete(b.engineSettings, engine)
			} else {
				*record = *previous
			}
			return settings.Snapshot{}, err
		}
	}
	return record.Snapshot.Clone(), nil
}

// Build initialization, migration and drift readback on a private record.
// Apply must finish worker IO before it can atomically authorize and save.
func (b *Broker) prepareSettingsSnapshotLocked(ctx context.Context, engine string) (*engineSettingsRecord, bool, error) {
	if _, ok := engineProxyProfileFor(engine); !ok {
		return nil, false, fmt.Errorf("this engine does not support settings")
	}
	if err := b.loadEngineSettingsLocked(); err != nil {
		return nil, false, err
	}
	var launch settings.LaunchState
	if err := b.settingsWorkerCall(ctx, "engine:get-launch", map[string]string{"engine": engine}, &launch); err != nil {
		return nil, false, err
	}
	proxyPort := 0
	ready := false
	if proxy := b.settingsProxy(engine); proxy != nil {
		ready, proxyPort = proxy.Status(engine)
	}
	record := b.engineSettings[engine].clone()
	save, err := b.prepareSettingsArgumentsLocked(ctx, engine, record)
	if err != nil {
		return nil, false, err
	}
	configuredProxy := proxyPort
	if !ready && record != nil {
		configuredProxy = record.Snapshot.Settings.ProxyPort
	}
	current := settings.ConfigFromLaunch(launch, configuredProxy)
	if record == nil {
		record = &engineSettingsRecord{Snapshot: settings.Snapshot{NodeID: b.nodeID, Engine: engine, Revision: 1, AppliedRevision: 1, Epoch: b.engineSettingsEpoch, Phase: "idle", Settings: current}, Receipts: make(map[string]settingsReceipt)}
		save = true
	} else if record.Snapshot.Phase != "applying" && record.Snapshot.Phase != "failed" && !record.Snapshot.Settings.Equal(current) {
		record.Snapshot.Settings = current
		record.Snapshot.Revision++
		record.Snapshot.AppliedRevision = record.Snapshot.Revision
		save = true
	}
	snapshot := record.Snapshot.Clone()
	snapshot.Serving = launch.ServingReadback.Clone()
	snapshot.EffectiveServerPort = launch.EffectivePort
	snapshot.EffectiveProxyPort = proxyPort
	snapshot.Running = launch.Running
	snapshot.Adopted = launch.Adopted
	snapshot.Format = launch.Format
	snapshot.Editable = launch.Editable && ready
	snapshot.Reason = launch.Reason
	if !ready {
		snapshot.Reason = "The proxy service is unavailable."
	}
	if snapshot.Phase == "applying" {
		snapshot.Editable = false
		snapshot.Reason = "Settings are being applied."
	}
	if !reflect.DeepEqual(snapshot, record.Snapshot) {
		snapshot.Sequence++
		record.Snapshot = snapshot.Clone()
	}
	return record, save, nil
}

func (b *Broker) publishSettingsLocked() {
	all := make([]settings.Snapshot, 0, len(b.engineSettings))
	for _, profile := range engineProxyProfiles {
		if record := b.engineSettings[profile.Name]; record != nil {
			record.Snapshot.Sequence++
			all = append(all, record.Snapshot.Clone())
			if b.codec != nil {
				_ = b.codec.Notify("engine:settings-changed", record.Snapshot)
			}
		}
	}
	// This is a state projection, not permission to execute forward work.
	b.workersMu.Lock()
	worker := b.engineMgr
	b.workersMu.Unlock()
	if worker != nil {
		_ = worker.Notify("engine:settings-projection", all)
	}
	identities := make([]settings.Snapshot, 0, len(all))
	c := b.switchCoordinator()
	for _, snapshot := range all {
		// A settings refresh must not undo an acknowledged switch withdrawal
		// while forward work or conservative cleanup still reserves the engine.
		if c.store == nil || b.pendingSwitchLocked(snapshot.Engine) == nil {
			identities = append(identities, snapshot)
		}
	}
	b.observeSettingsIdentity(identities)
}

func (b *Broker) validateSettingsPortsLocked(ctx context.Context, engine string, config settings.Config, current settings.Snapshot) error {
	if config.ServerPort < 1 || config.ServerPort > 65535 || config.ProxyPort < 1 || config.ProxyPort > 65535 {
		return fmt.Errorf("ports must be between 1 and 65535")
	}
	if config.ServerPort == config.ProxyPort {
		return fmt.Errorf("Server port and proxy port collide: both are %d. Choose different ports.", config.ServerPort)
	}
	reserved := map[int]bool{}
	for _, service := range b.servicePorts().entries() {
		reserved[service.port] = true
	}
	if alias := b.currentOllamaHostAlias().Port; alias > 0 {
		reserved[alias] = true
	}
	if listener := b.routingListenerPort(); listener > 0 {
		reserved[listener] = true
	}
	// Include every registered PAIR listener, including services added later.
	if b.regCache != nil {
		for _, service := range b.regCache.Snapshot() {
			if service.Port > 0 && !isEngineDiscoveryService(service.Service) {
				reserved[service.Port] = true
			}
		}
	}
	if reserved[config.ServerPort] || reserved[config.ProxyPort] {
		return fmt.Errorf("a selected port is reserved by a PAIR service or proxy alias")
	}
	var installed struct {
		Engines []struct {
			Engine string `json:"engine"`
			Port   int    `json:"port"`
		} `json:"engines"`
	}
	if err := b.settingsWorkerCall(ctx, "engine:configured-ports", nil, &installed); err != nil {
		return err
	}
	for _, other := range installed.Engines {
		if other.Engine != engine && (other.Port == config.ServerPort || other.Port == config.ProxyPort) {
			return fmt.Errorf("a selected port is reserved by another configured engine")
		}
	}
	for _, profile := range engineProxyProfiles {
		if profile.Name == engine {
			continue
		}
		if record := b.engineSettings[profile.Name]; record != nil {
			port := record.Snapshot.Settings.ProxyPort
			if port == config.ServerPort || port == config.ProxyPort {
				return fmt.Errorf("a selected port is reserved by another configured proxy")
			}
		}
		if proxy := b.settingsProxy(profile.Name); proxy != nil {
			_, port := proxy.Status(profile.Name)
			if port == config.ServerPort || port == config.ProxyPort {
				return fmt.Errorf("a selected port is already used by another proxy")
			}
		}
	}
	for _, port := range []int{config.ServerPort, config.ProxyPort} {
		if port == current.EffectiveProxyPort || (current.Running && port == current.EffectiveServerPort) {
			continue
		}
		if !b.portAvailable(port) {
			return fmt.Errorf("port %d is already in use", port)
		}
	}
	return nil
}

func (b *Broker) previewSettingsLocked(ctx context.Context, p settings.Request) (settings.Preview, error) {
	current, err := b.settingsSnapshotLocked(ctx, p.Engine)
	if err != nil {
		return settings.Preview{}, err
	}
	return b.previewSettingsSnapshotLocked(ctx, p, current)
}

func (b *Broker) previewSettingsSnapshotLocked(ctx context.Context, p settings.Request, current settings.Snapshot) (settings.Preview, error) {
	if current.Revision != p.ExpectedRevision {
		return settings.Preview{}, fmt.Errorf("settings changed on this device; reload before applying")
	}
	var preview settings.Preview
	if err := b.settingsWorkerCall(ctx, "engine:preview-launch", p, &preview); err != nil {
		return preview, err
	}
	preview.Revision = current.Revision
	preview.Rebind = preview.Settings.ProxyPort != current.EffectiveProxyPort
	if preview.Errors == nil {
		preview.Errors = make(map[string]string)
	}
	if !current.Editable {
		preview.Errors["settings"] = current.Reason
	}
	if len(preview.Errors) == 0 && preview.Conflict == nil {
		if err := b.validateSettingsPortsLocked(ctx, p.Engine, preview.Settings, current); err != nil {
			preview.Errors["ports"] = err.Error()
		}
	}
	return preview, nil
}

// runSettingsOperationLocked applies an already-accepted record. The caller
// holds engineConfigMu; it is released around the engine-manager call, which
// stops and restarts the engine and can run for minutes, and re-acquired to
// record the outcome. Holding the journal lock across that call froze every
// other reader — the other engine's editor, the refresh poller — for the whole
// restart. settingsApplyMu, which the caller also holds, is what keeps a
// second apply out of this window.
func (b *Broker) runSettingsOperationLocked(ctx context.Context, engine string, record *engineSettingsRecord) error {
	if err := b.migrateSettingsArgumentsLocked(ctx, engine, record); err != nil {
		return err
	}
	profile, _ := engineProxyProfileFor(engine)
	service := profile.DiscoveryService
	if b.regCache != nil {
		b.unregisterService(service)
	}
	b.setProxyLocalBackend(b.settingsProxy(engine), engine, record.Snapshot.EffectiveServerPort, false)
	id := settingsID()
	b.settingsRelayMu.Lock()
	if b.activeSettings == nil {
		b.activeSettings = make(map[string]activeSettingsOperation)
	}
	b.activeSettings[id] = activeSettingsOperation{Engine: engine, Port: record.Snapshot.Settings.ProxyPort}
	b.settingsRelayMu.Unlock()
	defer func() { b.settingsRelayMu.Lock(); delete(b.activeSettings, id); b.settingsRelayMu.Unlock() }()
	// A paused node stays paused: an apply never resumes a stopped engine
	// while it is, from any caller or from recovery; resuming the node does.
	// A running engine whose launch changed is still restarted.
	paused, _ := b.settingsPause()
	configure := settings.Configure{Engine: engine, Settings: record.Snapshot.Settings.Clone(), OperationID: id, Resume: record.Resume && !paused}

	launch, err := func() (settings.LaunchState, error) {
		b.engineConfigMu.Unlock()
		defer b.engineConfigMu.Lock()
		var launch settings.LaunchState
		// No strict request may be admitted under settings about to change.
		err := b.withdrawSettingsIdentity(ctx, engine)
		if err == nil {
			err = b.settingsWorkerCall(ctx, "engine:configure-launch", configure, &launch)
		}
		if err != nil {
			// A failed RPC carries no result, so read the observed facts back
			// here too and keep the whole worker round trip off the lock.
			launch = settings.LaunchState{}
			_ = b.settingsWorkerCall(ctx, "engine:get-launch", map[string]string{"engine": engine}, &launch)
		}
		return launch, err
	}()

	if err != nil {
		record.Snapshot.Phase = "failed"
		record.Snapshot.Error = err.Error()
	} else {
		record.Snapshot.Phase = "succeeded"
		record.Snapshot.Error = ""
		record.Snapshot.AppliedRevision = record.Snapshot.Revision
	}
	// Refresh observed facts without replacing the accepted desired revision;
	// never publish the pre-operation running state.
	if launch.Engine != "" {
		record.Snapshot.Serving = launch.ServingReadback.Clone()
		record.Snapshot.Running = launch.Running
		record.Snapshot.Adopted = launch.Adopted
		record.Snapshot.EffectiveServerPort = launch.EffectivePort
		record.Snapshot.Editable = launch.Editable
		record.Snapshot.Reason = launch.Reason
		b.engineProxy(profile).backendPort.Store(int32(launch.EffectivePort))
	}
	proxyReady := false
	if proxy := b.settingsProxy(engine); proxy != nil {
		proxyReady, record.Snapshot.EffectiveProxyPort = proxy.Status(engine)
	}
	// An unsuccessful apply can leave the previous engine running. Restore
	// discovery from observed runtime state, independent of the apply result.
	if proxyReady && launch.Running && launch.EffectivePort > 0 && record.Snapshot.EffectiveProxyPort > 0 && launch.EffectivePort != record.Snapshot.EffectiveProxyPort {
		b.setProxyLocalBackend(b.settingsProxy(engine), engine, launch.EffectivePort, true)
		if b.regCache != nil {
			b.registerService(noderec.RegisterParams{Service: service, Port: record.Snapshot.EffectiveProxyPort})
		}
	}
	if err == nil {
		record.Resume = false
	}
	if receipt, exists := record.Receipts[record.Snapshot.RequestID]; exists {
		receipt.Phase = record.Snapshot.Phase
		record.Receipts[record.Snapshot.RequestID] = receipt
	}
	if saveErr := b.saveEngineSettingsLocked(); saveErr != nil {
		record.Snapshot.Phase = "failed"
		record.Snapshot.Error = "The operation result could not be saved. Reload to reconcile the device."
		if receipt, exists := record.Receipts[record.Snapshot.RequestID]; exists {
			receipt.Phase = "failed"
			record.Receipts[record.Snapshot.RequestID] = receipt
		}
		err = saveErr
	}
	b.publishSettingsLocked()
	return err
}

// settingsPause reports whether this node is paused, and whether by itself. A
// local pause accepted and not yet saved already counts as local, as it does
// for a paired node's resume (setAvailability).
func (b *Broker) settingsPause() (paused, local bool) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	local = pausedLocally(r.current) || r.localPauseRequested
	return r.managementBlockedLocked(), local
}

// authorizeSettingsCallerLocked rejects a peer this node no longer pins. A
// local caller passes an empty caller and is always allowed.
func (b *Broker) authorizeSettingsCallerLocked(caller string) error {
	if caller == "" {
		return nil
	}
	if !clustertrust.Open(b.clusterDir).HasPin(caller) {
		return fmt.Errorf("the requesting device is no longer paired")
	}
	return nil
}

// getEngineSettings reads the authoritative snapshot for one engine and
// republishes it, so a reader and every subscriber see the same revision.
func (b *Broker) getEngineSettings(ctx context.Context, p settings.Request, caller string) (settings.Snapshot, error) {
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if err := ctx.Err(); err != nil {
		return settings.Snapshot{}, err
	}
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		return settings.Snapshot{}, err
	}
	snapshot, err := b.settingsSnapshotLocked(ctx, p.Engine)
	if err != nil {
		return settings.Snapshot{}, err
	}
	b.publishSettingsLocked()
	return snapshot, nil
}

// previewEngineSettings validates a proposed change and reports what applying
// it would do, without touching the journal or the engine.
func (b *Broker) previewEngineSettings(ctx context.Context, p settings.Request, caller string) (settings.Preview, error) {
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if err := ctx.Err(); err != nil {
		return settings.Preview{}, err
	}
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		return settings.Preview{}, err
	}
	p.PreserveCORS = caller != ""
	return b.previewSettingsLocked(ctx, p)
}

// applyEngineSettings accepts a change into the journal and then converges the
// engine onto it. RequestID makes it idempotent: replaying one returns the
// original receipt, and reusing it for different settings is refused.
func (b *Broker) applyEngineSettings(ctx context.Context, p settings.Request, caller string) (settingsReceipt, error) {
	ctx, cancel := context.WithTimeout(ctx, settings.CallBudget)
	defer cancel()
	// Serialize whole applies. Taken before engineConfigMu; see its comment.
	b.settingsApplyMu.Lock()
	defer b.settingsApplyMu.Unlock()
	b.engineConfigMu.Lock()
	defer b.engineConfigMu.Unlock()
	if err := ctx.Err(); err != nil {
		return settingsReceipt{}, err
	}
	if err := b.authorizeSettingsCallerLocked(caller); err != nil {
		return settingsReceipt{}, err
	}
	// The acceptance recheck under the journal locks: the permission may have
	// been revoked since the relay was read.
	if err := b.authorizeRemoteManagement(caller, "settings", p.RequestID); err != nil {
		return settingsReceipt{}, err
	}
	if p.RequestID == "" || len(p.RequestID) > 128 {
		return settingsReceipt{}, fmt.Errorf("a request identifier is required")
	}
	if err := b.loadEngineSettingsLocked(); err != nil {
		return settingsReceipt{}, err
	}
	fingerprint := sha256.Sum256(settingsJSON(struct {
		Caller  string
		Request settings.Request
	}{caller, p}))
	hash := hex.EncodeToString(fingerprint[:])
	if record := b.engineSettings[p.Engine]; record != nil {
		if receipt, exists := record.Receipts[p.RequestID]; exists {
			if err := b.authorizeRemoteManagement(caller, "settings", p.RequestID); err != nil {
				return settingsReceipt{}, err
			}
			if receipt.Hash != hash {
				return settingsReceipt{}, fmt.Errorf("request identifier was already used for different settings")
			}
			return receipt, nil
		}
	}
	if err := b.pendingSwitchLocked(p.Engine); err != nil {
		return settingsReceipt{}, err
	}
	// Recheck CORS after acquiring the target's apply lock. An HTTP ingress
	// check alone can become stale while another accepted apply is running.
	p.PreserveCORS = caller != ""
	record, _, err := b.prepareSettingsSnapshotLocked(ctx, p.Engine)
	if err != nil {
		return settingsReceipt{}, err
	}
	preview, err := b.previewSettingsSnapshotLocked(ctx, p, record.Snapshot)
	if err != nil {
		return settingsReceipt{}, err
	}
	// Preview can wait on a worker. Order the final decision and journal write
	// with pause acceptance, without holding r.mu across worker or client IO.
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if err := b.authorizeRemoteManagementLocked(r, caller, "settings", p.RequestID); err != nil {
		r.mu.Unlock()
		return settingsReceipt{}, err
	}
	if len(preview.Errors) > 0 || preview.Conflict != nil {
		r.mu.Unlock()
		return settingsReceipt{}, fmt.Errorf("resolve settings errors and port conflicts before applying")
	}
	previous := b.engineSettings[p.Engine]
	b.engineSettings[p.Engine] = record
	rollback := func() {
		if previous == nil {
			delete(b.engineSettings, p.Engine)
		} else {
			b.engineSettings[p.Engine] = previous
		}
	}
	if preview.Settings.Equal(record.Snapshot.Settings) && record.Snapshot.Phase != "failed" {
		record.Snapshot.RequestID = p.RequestID
		record.Snapshot.Phase = "succeeded"
		record.Receipts[p.RequestID] = settingsReceipt{Hash: hash, Revision: record.Snapshot.Revision, Phase: "succeeded"}
		if err := b.saveEngineSettingsLocked(); err != nil {
			rollback()
			r.mu.Unlock()
			return settingsReceipt{}, err
		}
		r.mu.Unlock()
		b.publishSettingsLocked()
		return record.Receipts[p.RequestID], nil
	}
	record.Previous = record.Snapshot.Settings.Clone()
	record.Resume = record.Resume || preview.Restart
	record.Explicit = true
	record.Snapshot.Settings = preview.Settings.Clone()
	record.Snapshot.Revision++
	record.Snapshot.RequestID = p.RequestID
	record.Snapshot.Phase = "applying"
	record.Snapshot.Error = ""
	record.Receipts[p.RequestID] = settingsReceipt{Hash: hash, Revision: record.Snapshot.Revision, Phase: "applying"}
	if err := b.saveEngineSettingsLocked(); err != nil {
		rollback()
		r.mu.Unlock()
		return settingsReceipt{}, fmt.Errorf("settings could not be saved; no runtime changes were made")
	}
	r.mu.Unlock()
	b.publishSettingsLocked()
	// Once accepted, completion is target-owned. Losing the requesting UI or
	// peer cannot abandon half a port swap. The operation has its own deadline.
	operationCtx, operationCancel := context.WithTimeout(context.Background(), settings.OperationBudget)
	defer operationCancel()
	// A restart of a running engine first drains it at the proxy (settingsdrain.go).
	if preview.Restart {
		release := b.drainEngineForSettingsLocked(operationCtx, p.Engine)
		defer release()
	}
	_ = b.runSettingsOperationLocked(operationCtx, p.Engine, record)
	return record.Receipts[p.RequestID], nil
}

// dispatchEngineSettings routes an RPC method name to its typed entry point.
// It exists only for the two edges that receive the method as a string — the
// client RPC surface and the peer relay — so no in-process caller has to
// recover a concrete result from an interface.
func (b *Broker) dispatchEngineSettings(ctx context.Context, method string, p settings.Request, caller string) (any, error) {
	switch method {
	case "get":
		return b.getEngineSettings(ctx, p, caller)
	case "preview":
		return b.previewEngineSettings(ctx, p, caller)
	case "apply":
		return b.applyEngineSettings(ctx, p, caller)
	default:
		return nil, fmt.Errorf("unsupported settings method")
	}
}

func (b *Broker) handleEngineSettings(msg *Message) {
	var p settings.Request
	if json.Unmarshal(msg.Params, &p) != nil {
		_ = b.codec.RespondError(msg.ID, -32602, "invalid settings request")
		return
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		remote := "engine:remote-" + strings.TrimPrefix(msg.Method, "engine:")
		worker := b.getEngineMgr()
		if worker == nil {
			_ = b.codec.RespondError(msg.ID, -32000, "engine manager unavailable")
			return
		}
		err := worker.RelayRequest(remote, msg.Params, func(result json.RawMessage, rpcErr *RPCError, err error) {
			if err != nil {
				_ = b.codec.RespondError(msg.ID, -32000, err.Error())
			} else if rpcErr != nil {
				_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
			} else {
				_ = b.codec.Respond(msg.ID, result)
			}
		})
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, err.Error())
		}
		return
	}
	method := strings.TrimSuffix(strings.TrimPrefix(msg.Method, "engine:"), "-settings")
	result, err := b.dispatchEngineSettings(context.Background(), method, p, "")
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32000, err.Error())
	} else {
		_ = b.codec.Respond(msg.ID, result)
	}
}

func (b *Broker) handleSettingsRelay(source *engineManagerSource, raw json.RawMessage) {
	var relay settings.Relay
	if json.Unmarshal(raw, &relay) != nil || relay.ID == "" {
		return
	}
	if !b.engineManagerSourceCurrent(source) {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), settings.RelayBudget)
	b.settingsRelayMu.Lock()
	if b.settingsCancels == nil {
		b.settingsCancels = make(map[string]context.CancelFunc)
	}
	if len(b.settingsCancels) >= 128 {
		b.settingsRelayMu.Unlock()
		cancel()
		b.replyPeerRelay(source, b.getEngineMgr(), "engine:settings-reply", map[string]string{"id": relay.ID, "error": "too many settings requests"})
		return
	}
	b.settingsCancels[relay.ID] = cancel
	b.settingsRelayMu.Unlock()
	go func() {
		defer cancel()
		defer func() { b.settingsRelayMu.Lock(); delete(b.settingsCancels, relay.ID); b.settingsRelayMu.Unlock() }()
		worker, pairErr := b.awaitEngineMgrPair(ctx, source)
		if pairErr != nil || !b.engineManagerSourceCurrent(source) {
			return
		}
		var result any
		var err error
		if relay.Method == "rebind" {
			b.settingsRelayMu.Lock()
			operation, ok := b.activeSettings[relay.Request.RequestID]
			b.settingsRelayMu.Unlock()
			if !ok || operation.Engine != relay.Request.Engine || operation.Port != relay.Request.Settings.ProxyPort {
				err = fmt.Errorf("unknown settings operation")
			} else {
				err = b.rebindSettingsProxy(operation.Engine, operation.Port)
			}
		} else if callerErr := requireRemoteCaller(relay.Caller); callerErr != nil {
			// A peer relay must carry the caller engine-manager verified;
			// otherwise it would take a local caller's empty-caller path.
			err = callerErr
		} else {
			result, err = b.dispatchEngineSettings(ctx, relay.Method, relay.Request, relay.Caller)
		}
		reply := struct {
			ID        string `json:"id"`
			Result    any    `json:"result"`
			Error     string `json:"error,omitempty"`
			ErrorCode int    `json:"errorCode,omitempty"`
		}{ID: relay.ID, Result: result}
		if err != nil {
			reply.Error = err.Error()
			reply.ErrorCode = policyErrorCode(err)
		}
		b.replyPeerRelay(source, worker, "engine:settings-reply", reply)
	}()
}

func (b *Broker) rebindSettingsProxy(engine string, port int) error {
	p := b.settingsProxy(engine)
	if p == nil {
		return fmt.Errorf("proxy unavailable")
	}
	// Disable automatic facade takeover before the ready event can race the
	// explicit rebind. The accepted journal restores these choices after restart.
	profile, _ := engineProxyProfileFor(engine)
	state := b.engineProxy(profile)
	state.explicitSettings.Store(true)
	state.managedFacade.Store(false)
	_, rpcErr, err := p.Call(context.Background(), engine+":set-port", settingsJSON(map[string]int{"port": port}))
	if err != nil {
		return err
	}
	if rpcErr != nil {
		return fmt.Errorf("%s", rpcErr.Message)
	}
	state.managedFacade.Store(false)
	state.startupPort.Store(int32(port))
	return nil
}

func (b *Broker) refreshEngineSettings(ctx context.Context) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !b.managedPortOwnershipReady() {
				continue
			}
			func() {
				b.engineConfigMu.Lock()
				defer b.engineConfigMu.Unlock()
				readCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
				defer cancel()
				changed := false
				for _, engine := range engines.Names() {
					var before settings.Snapshot
					if r := b.engineSettings[engine]; r != nil {
						before = r.Snapshot.Clone()
					}
					after, err := b.settingsSnapshotLocked(readCtx, engine)
					if err == nil && !reflect.DeepEqual(before, after) {
						changed = true
					}
				}
				if changed {
					b.publishSettingsLocked()
				}
			}()
		}
	}
}
