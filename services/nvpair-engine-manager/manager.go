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
	"log"
	"log/slog"
	"path/filepath"
	"sync"

	"nvpair-shared/applog"
	"nvpair-shared/buildapply"
	"nvpair-shared/clustertrust"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/noderec"
	"nvpair-shared/reach"
)

// Version is stamped at build time via -ldflags "-X main.Version=...".
// See versions.json at the repo root for the source of truth.
var Version = "dev"

// ReadyParams is the payload of the one-shot "engine:ready" notification.
type ReadyParams struct {
	Version string `json:"version"`
}

type engineParam struct {
	Engine string `json:"engine"`
	Port   int    `json:"port,omitempty"`
}

// opParam is the install/uninstall input.
type opParam struct {
	Engine string `json:"engine"`
}

func (p *opParam) UnmarshalJSON(data []byte) error {
	type wire opParam
	var value wire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return err
	}
	*p = opParam(value)
	return nil
}

type actionParam struct {
	Engine string          `json:"engine"`
	Action string          `json:"action"`
	Params json.RawMessage `json:"params,omitempty"`
}

// setPortParam is the engine:set-port input: the engine whose server port to
// change and the new port. The chosen port is persisted as a manifest
// override and applied (the engine is bounced onto it if running).
type setPortParam struct {
	Engine string `json:"engine"`
	Port   int    `json:"port"`
}

// Manager is the engine-manager's JSON-RPC front end. It dispatches the
// engine:* surface to the Executor. Long-running operations (install,
// uninstall, action) run in their own goroutine so the read
// loop never blocks; the codec serializes the concurrent responses.
type Manager struct {
	diagnosticRecoveryMu sync.Mutex
	copyMu               sync.Mutex
	copies               map[string]*activeCopy
	copiesClosed         bool
	// Guarded by copyMu. Held steps include each live copy's retirement;
	// one further step is always left for shutdown closure.
	copyRevision uint64
	copyHeld     uint64
	// Shares the Executor's durable content owner, including on controllers
	// that never access a destination engine or model store.
	copyJournal  *modelDownloads
	copyRegistry *Registry
	// Preparation is read-only; its returned effect runs inside durable evidence.
	copyPrepare      func(context.Context, copyRequest, string, func(copyProgress)) (func() (copyModelResult, error), copyModelResult, error)
	settingsRelay    settingsRelay
	diagnosticRelay  diagnosticRelay
	policyRelay      policyRelay
	modelMutation    modelMutationRelay
	residentDispatch residentDispatchRelay
	admittedLoad     admittedLoadRelay
	codec            *Codec
	exec             *Executor
	peers            *peerDirectory
	addrs            *reach.Chooser
	mesh             *clustertrust.Mesh // cluster identity + pins for dialing peers' ec surfaces
	controlHTTP      *clustertrust.PeerServiceClient
	controlReadyHTTP *clustertrust.PeerServiceClient
	// remoteEngines caches each peer's streamed engine status list
	// (engines_stream.go).
	remoteEngines remoteEngineCache
	// catalog answers engine:catalog: the models an engine can download, as
	// opposed to the ones it already holds.
	catalog *catalogService
	cancel  context.CancelFunc
}

func NewManager(codec *Codec, exec *Executor, mesh *clustertrust.Mesh, clusterDir string) *Manager {
	exec.memoryEvidence.mu.Lock()
	exec.memoryEvidence.hostUUID = mesh.NodeUUID
	exec.memoryEvidence.mu.Unlock()
	// cancel defaults to a no-op so the "shutdown" handler is safe even if
	// handleMessage is reached before Run() installs the real CancelFunc
	// (e.g. a unit test calling it directly); Run overwrites it.
	m := &Manager{
		codec:        codec,
		exec:         exec,
		copyJournal:  exec.downloads,
		copyRegistry: exec.reg,
		peers:        newPeerDirectory(),
		addrs:        reach.NewChooser(),
		mesh:         mesh,
		catalog:      newCatalogService(),
		cancel:       func() {},
	}
	m.copyPrepare = m.prepareCopy
	m.catalog.familyOf = exec.reg.familyOf
	baseDir := ""
	if clusterDir != "" {
		baseDir = filepath.Dir(filepath.Clean(clusterDir))
	}
	m.controlHTTP = clustertrust.NewPeerServiceClient(mesh, clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{ResponseHeaderTimeout: remoteResponseHeaderTimeout},
		BaseConfigDir:     baseDir, ResolveRoute: m.resolveControlRoute,
	})
	m.controlReadyHTTP = clustertrust.NewPeerServiceClient(mesh, clustertrust.PeerServiceClientOptions{
		PeerClientOptions: clustertrust.PeerClientOptions{ResponseHeaderTimeout: remoteReadyResponseHeaderTimeout},
		BaseConfigDir:     baseDir, ResolveRoute: m.resolveControlRoute,
	})
	m.admittedLoad.send = codec.NotifyContext
	exec.admittedLoadParent = m.admittedLoad.call
	m.settingsRelay.send = codec.NotifyContext
	m.diagnosticRelay.send = codec.NotifyContext
	exec.diagnosticParent = m.diagnosticRelay.call
	exec.settingsParent = m.settingsRelay.call
	exec.inspectParent = m.settingsRelay.callInspect
	exec.memoryProfilesParent = m.settingsRelay.callRelay
	m.policyRelay.send = codec.NotifyContext
	exec.policyParent = m.policyRelay.call
	m.modelMutation.send = codec.NotifyContext
	exec.modelMutationParent = m.modelMutation.call
	m.residentDispatch.send = codec.NotifyContext
	exec.residentParent = m.residentDispatch.call
	return m
}

func (m *Manager) Run(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	m.cancel = cancel
	defer cancel()
	defer m.stopCopies()
	// Restore exclusion before publishing readiness or accepting any activation.
	if _, err := m.exec.restoreDiagnosticCheckpoint(); err != nil {
		return fmt.Errorf("restore diagnostic cleanup hold: %w", err)
	}

	if err := m.codec.Notify("engine:ready", ReadyParams{Version: Version}); err != nil {
		return fmt.Errorf("failed to send ready notification: %w", err)
	}

	// Subscribe to the broker's discovery relay for ec peers (nodes exposing the
	// remote-control surface). They arrive as discovery:nodes snapshots handled
	// in handleMessage. Non-fatal: the directory just stays empty if the parent
	// isn't a relay-aware broker, and remote calls then resolve to no target.
	if err := m.codec.Notify(noderec.MethodSubscribe, noderec.SubscribeParams{Services: []noderec.ServiceKey{noderec.ServiceEngineControl}}); err != nil {
		slog.Warn("failed to subscribe to discovery relay for ec peers", "err", err)
	}

	// Watch each running engine's loaded-model set and emit engine:models-changed
	// on change (load/unload, JIT auto-load, TTL eviction). Bound to ctx: stops
	// when the read loop exits below.
	go m.exec.watchLoaded(ctx)
	go m.exec.modelIdentities.run(ctx)
	go m.exec.enginesEvents.run(ctx, m.exec.publishedStatuses)
	go m.watchPeerSettings(ctx)
	startupRecoveryDone := make(chan struct{})
	go func() {
		defer close(startupRecoveryDone)
		if err := m.recoverDiagnosticBeforeRestore(ctx); err != nil {
			slog.Warn("diagnostic startup cleanup remains held", "err", err)
		}
	}()

	err := m.readLoop(ctx)
	// Cancel first so any in-flight start aborts at its readiness wait,
	// then stop engines so the parent's exit doesn't orphan them.
	cancel()
	m.stopCopies()
	m.exec.abortDiagnosticAndJoin()
	<-startupRecoveryDone
	m.exec.StopAll()
	m.controlHTTP.CloseIdle()
	m.controlReadyHTTP.CloseIdle()
	return err
}

func (m *Manager) readLoop(ctx context.Context) error {
	for {
		msg, err := m.codec.Read()
		if err != nil {
			if err == io.EOF || ctx.Err() != nil {
				return nil
			}
			var de *DecodeError
			if errors.As(err, &de) {
				log.Printf("JSON-RPC decode error (skipping frame): %v", err)
				continue
			}
			// Terminal transport/scanner error — stop instead of spinning.
			log.Printf("JSON-RPC read error (terminal): %v", err)
			return err
		}
		m.handleMessage(ctx, msg)
		if ctx.Err() != nil {
			return nil
		}
	}
}

func (m *Manager) handleMessage(ctx context.Context, msg *Message) {
	if msg.Method == residentDispatchReplyMethod && msg.IsNotification() {
		m.residentDispatch.reply(msg.Params)
		return
	}
	if m.handleDiagnosticRelayMessage(ctx, msg) || m.handleSettingsMessage(ctx, msg) || m.handlePolicyMessage(ctx, msg) {
		return
	}
	if msg.Method == applog.SetLevelMethod {
		resolved, err := applog.HandleSetLevelParams(msg.Params)
		if msg.IsRequest() {
			if err != nil {
				m.codec.RespondError(msg.ID, -32602, err.Error())
				return
			}
			m.codec.Respond(msg.ID, map[string]string{"level": resolved})
		}
		if err != nil {
			slog.Warn("log/set-level rejected", "err", err)
		} else {
			slog.Info("log level changed", "level", resolved)
		}
		return
	}

	if msg.Method == noderec.NotifyNodes {
		m.applyDiscovery(msg)
		return
	}
	// The broker's answer to this service's model-mutation request
	// (model_mutation.go). It is consumed as an internal notification here and
	// never treated as a client request, so a public request with the same name
	// cannot complete a pending wait.
	if msg.Method == admittedLoadReplyMethod && msg.IsNotification() {
		m.admittedLoad.reply(msg.Params)
		return
	}
	if msg.Method == nodepolicy.MethodModelMutationReply && msg.IsNotification() {
		m.modelMutation.reply(msg.Params)
		return
	}
	if msg.Method == nodeProjectionMethod && msg.IsNotification() {
		m.applyNodeProjection(msg)
		return
	}
	if msg.Method == restoreInstalledMethod && msg.IsNotification() {
		go func() {
			if err := m.recoverDiagnosticBeforeRestore(ctx); err != nil {
				slog.Warn("installed restore held by diagnostic cleanup", "err", err)
				return
			}
			if err := m.exec.RestoreInstalled(ctx); err != nil {
				slog.Warn("one or more installed engines could not be restored", "err", err)
			}
		}()
		return
	}

	if !msg.IsRequest() {
		if msg.IsNotification() {
			log.Printf("ignoring incoming notification: %s", msg.Method)
		}
		return
	}

	switch msg.Method {
	case "engine:owner-evidence":
		var request buildapply.EngineOwnerEvidenceRequest
		if err := buildapply.DecodeRequest(msg.Params, &request); err != nil {
			m.codec.RespondError(msg.ID, -32602, "Invalid engine owner evidence request.")
			return
		}
		evidence, err := m.engineOwnerEvidence()
		if err != nil {
			m.codec.RespondError(msg.ID, -32000, "Engine owner evidence is unavailable.")
			return
		}
		// Marshal before writing so overflow/invalid evidence produces one fixed
		// unavailable response, never a truncated authority-bearing section.
		data, err := json.Marshal(evidence)
		if err != nil {
			m.codec.RespondError(msg.ID, -32000, "Engine owner evidence is unavailable.")
			return
		}
		m.codec.Respond(msg.ID, json.RawMessage(data))

	case nodepolicy.MethodReconcileResident:
		go m.runResident(ctx, msg)

	case "shutdown":
		if err := m.codec.Respond(msg.ID, nil); err != nil {
			log.Printf("failed to respond to shutdown: %v", err)
		}
		log.Println("shutdown requested via JSON-RPC")
		m.cancel()

	case "engine:get-installed":
		go func() {
			m.codec.Respond(msg.ID, map[string]any{"engines": m.exec.GetInstalled()})
		}()

	// Removes the engines PAIR installed. Long-running — each engine goes
	// through the full uninstall — so it runs off the read loop like the other
	// lifecycle operations.
	case "engine:uninstall-managed":
		go func() {
			m.codec.Respond(msg.ID, map[string]any{"engines": m.exec.UninstallManaged(ctx)})
		}()

	case prepareShutdownMethod:
		go func() {
			m.exec.abortDiagnosticAndJoin()
			m.exec.StopAll()
			m.codec.Respond(msg.ID, nil)
		}()

	case "engine:describe":
		var p engineParam
		if !m.parse(msg, &p) {
			return
		}
		mf, ok := m.exec.reg.Get(p.Engine)
		if !ok {
			m.codec.RespondError(msg.ID, -32602, fmt.Sprintf("unknown engine %q", p.Engine))
			return
		}
		m.codec.Respond(msg.ID, mf)

	case descriptorsMethod:
		m.codec.Respond(msg.ID, map[string]any{"engines": m.exec.reg.Descriptors()})

	case "engine:status":
		var p engineParam
		if !m.parse(msg, &p) {
			return
		}
		go func() {
			st, err := m.exec.StatusAtPort(p.Engine, p.Port)
			m.respondOrErr(msg, st, err)
		}()

	// Only the broker calls this; it refuses the method from its clients.
	case engines.LocalCredentialMethod:
		var p engineParam
		if !m.parse(msg, &p) {
			return
		}
		credential, err := m.exec.LocalCredential(p.Engine)
		m.respondOrErr(msg, credential, err)

	case "engine:logs":
		var p engineParam
		if !m.parse(msg, &p) {
			return
		}
		lines, err := m.exec.Logs(p.Engine)
		m.respondOrErr(msg, map[string]any{"lines": lines}, err)

	case "engine:errors":
		m.codec.Respond(msg.ID, map[string]any{"errors": m.exec.Errors()})

	case "engine:models":
		go m.runModels(ctx, msg)

	case "engine:catalog":
		go m.runCatalog(ctx, msg)

	case "engine:install", "engine:uninstall":
		go m.runOp(ctx, msg)

	case "engine:set-port":
		go m.runSetPort(ctx, msg)

	case "internal:set-reserved-port":
		var p struct {
			Port int `json:"port"`
		}
		if !m.parse(msg, &p) {
			return
		}
		if err := m.exec.SetReservedPort(p.Port); err != nil {
			m.codec.RespondError(msg.ID, -32602, err.Error())
			return
		}
		m.codec.Respond(msg.ID, map[string]int{"port": p.Port})

	case "engine:action":
		go m.runAction(ctx, msg)

	case loadModelMethod, unloadModelMethod, deleteModelMethod:
		go m.runModelOperation(ctx, msg)

	case "engine:remote-get-installed", "engine:remote-install", "engine:remote-pull-model",
		"engine:remote-load-model", "engine:remote-unload-model", "engine:remote-delete-model":
		go m.runRemote(ctx, msg)

	case "engine:remote-copy-model":
		go m.runRemoteCopy(ctx, msg)
	case "engine:copy-model":
		go m.runCopy(ctx, msg)
	case "engine:cancel-model-copy":
		m.copyState(msg, true)
	case "engine:active-model-copies":
		m.copyState(msg, false)
	case "engine:model-copy-capabilities":
		go m.runCopyCapabilities(ctx, msg)

	case remoteManagementGetMethod:
		go m.remoteManagementGet(ctx, msg)

	default:
		m.codec.RespondError(msg.ID, -32601, fmt.Sprintf("method not found: %s", msg.Method))
	}
}

// runOp executes a lifecycle op asynchronously and responds with the
// engine's resulting status (or an error).
func (m *Manager) runOp(ctx context.Context, msg *Message) error {
	if msg.Method == diagnosticRunMethod {
		m.startDiagnostic(msg)
		return nil
	}
	var p opParam
	if !m.parse(msg, &p) {
		return fmt.Errorf("invalid lifecycle parameters")
	}
	if p.Engine == "" {
		m.codec.RespondError(msg.ID, -32602, "engine is required")
		return fmt.Errorf("engine is required")
	}
	var err error
	switch msg.Method {
	case "engine:install":
		err = m.exec.Install(ctx, p.Engine)
	case "engine:uninstall":
		err = m.exec.Uninstall(ctx, p.Engine)
	case wakeMethod:
		err = m.exec.Wake(ctx, p.Engine)
	case sleepMethod:
		err = m.exec.sleepContext(ctx, p.Engine)
	}
	// Diagnostic dispatch is manager-internal and synchronous, with a scoped
	// context. No JSON-RPC response is emitted for this joined lifecycle call.
	if msg.ID == nil {
		return err
	}
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return err
	}
	st, err := m.exec.Status(p.Engine)
	m.respondOrErr(msg, st, err)
	return err
}

// runSetPort persists an engine's server port (as a manifest override) and
// applies it, responding with the engine's resulting status.
func (m *Manager) runSetPort(ctx context.Context, msg *Message) {
	var p setPortParam
	if !m.parse(msg, &p) {
		return
	}
	if p.Engine == "" {
		m.codec.RespondError(msg.ID, -32602, "engine is required")
		return
	}
	if p.Port < 1 || p.Port > 65535 {
		m.codec.RespondError(msg.ID, -32602, "port must be between 1 and 65535")
		return
	}
	st, err := m.exec.SetPort(ctx, p.Engine, p.Port)
	m.respondOrErr(msg, st, err)
}

// runModels answers engine:models with the union of running engines' model
// lists plus per-engine attribution (modelsByEngine). It runs async (like
// engine:action) because it makes one loopback HTTP call per running engine and
// must not block the read loop.
func (m *Manager) runModels(ctx context.Context, msg *Message) {
	result := m.exec.ModelsResult(ctx)
	if result.Identities != nil && !jsonrpc.ResponseFitsFrame(msg.ID, result, jsonrpc.WorkerFrameBytes) {
		result.Identities = nil
	}
	m.codec.Respond(msg.ID, result)
}

// runCatalog answers engine:catalog. It is off the request goroutine because the
// the llama.cpp source is a live fetch on a cold cache.
func (m *Manager) runCatalog(ctx context.Context, msg *Message) {
	var p catalogParams
	if !m.parse(msg, &p) {
		return
	}
	if p.Engine == "" {
		m.codec.RespondError(msg.ID, -32602, "engine is required")
		return
	}
	res, err := m.catalog.Catalog(ctx, p.Engine, p.Platform, p.Arch, p.Query)
	if err != nil {
		m.codec.RespondError(msg.ID, -32603, err.Error())
		return
	}
	m.codec.Respond(msg.ID, res)
}

func (m *Manager) runAction(ctx context.Context, msg *Message) {
	var p actionParam
	if !m.parse(msg, &p) {
		return
	}
	if p.Engine == "" || p.Action == "" {
		m.codec.RespondError(msg.ID, -32602, "engine and action are required")
		return
	}
	manifest, known := m.exec.reg.Get(p.Engine)
	if !known || !manifest.clientActionAllowed(p.Action) {
		m.codec.RespondError(msg.ID, -32602, "Use admitted model loads or proxy inference; this raw action is unavailable.")
		return
	}
	// A model pull is routed through the streaming path so a local pull emits
	// live engine:pull-progress just like a remote pull relays
	// engine:remote-progress; the response is still the action's terminal result
	// rather than a buffered progress blob.
	var res json.RawMessage
	var err error
	if p.Action == pullModelAction {
		manifest, ok := m.exec.reg.Get(p.Engine)
		if !ok {
			manifest = &Manifest{}
		}
		model, validationErr := manifest.pullProgressModel(p.Params)
		if validationErr != nil {
			m.codec.RespondError(msg.ID, -32602, validationErr.Error())
			return
		}
		var pull *modelPullReservation
		pull, err = m.exec.reserveModelPull(ctx, p.Engine, model, p.Params)
		if err == nil {
			res, err = pull.run(ctx)
		}
		if err != nil {
			userMsg := m.exec.reportPullFailed(p.Engine, model, err)
			if pull == nil {
				// Refusal before reservation has no lifecycle token to publish its
				// terminal. Preserve the local pull contract without inventing a
				// download attempt or duplicating a reserved attempt's terminal.
				m.exec.emitPullProgress(ProgressEvent{Engine: p.Engine, Op: "pull", Stage: "error", Percent: -1, Message: userMsg})
			}
			m.codec.RespondError(msg.ID, -32000, userMsg)
			return
		}
	} else {
		res, err = m.exec.Action(ctx, p.Engine, p.Action, p.Params)
	}
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	// A permitted mutation may have changed what's resident; nudge the loaded
	// watcher so engine:models-changed
	// fires promptly instead of waiting for the next poll tick. Pure reads
	// (list_models/loaded_models/list_downloaded) can't move residency, so skip
	// the nudge and its wasted sweep for them.
	if !residencyNeutralActions[p.Action] {
		m.exec.pokeLoaded()
	}
	m.codec.Respond(msg.ID, res)
}

// applyDiscovery folds a discovery:nodes snapshot (the broker relay's full
// filtered ec set) into the peer directory, replacing it wholesale.
func (m *Manager) applyDiscovery(msg *Message) {
	var res noderec.GetNodesResult
	if err := json.Unmarshal(msg.Params, &res); err != nil {
		slog.Debug("invalid discovery:nodes snapshot", "err", err)
		return
	}
	m.peers.set(res.Nodes)
}

func (m *Manager) parse(msg *Message, v any) bool {
	switch v.(type) {
	case *settings.Request, *settings.Configure:
		if err := validateServingModelsJSON(msg.Params, "settings", "serving"); err != nil {
			m.codec.RespondError(msg.ID, -32602, err.Error())
			return false
		}
	case *remoteParam:
		// Remote model RPCs must refuse before peer lookup/dial or progress
		// publication. Non-model remote operations retain their ignored fields.
		switch msg.Method {
		case "engine:remote-load-model", "engine:remote-unload-model", "engine:remote-delete-model", "engine:remote-copy-model", "engine:remote-pull-model":
			// Copy's node/engine/model structural checks already precede lookup.
			// Other remote typed operations check required models here because
			// runRemote constructs the peer client before its method guards.
			requireModel := msg.Method != "engine:remote-pull-model" && msg.Method != "engine:remote-copy-model"
			if err := validateModelRequest(msg.Params, requireModel); err != nil {
				m.codec.RespondError(msg.ID, -32602, err.Error())
				return false
			}
			var raw struct {
				Model  json.RawMessage `json:"model"`
				Params json.RawMessage `json:"params"`
			}
			if json.Unmarshal(msg.Params, &raw) == nil {
				if msg.Method == "engine:remote-pull-model" {
					if len(raw.Model) == 0 && (len(raw.Params) == 0 || string(raw.Params) == "null") {
						if err := validateModelRequest(msg.Params, true); err != nil {
							m.codec.RespondError(msg.ID, -32602, err.Error())
							return false
						}
					}
					// The local manifest does not establish a peer's binding.
					// Only common pull fields are identifiers here; the destination
					// checks its custom fields before subscribing or dispatching.
					if err := (&Manifest{}).validateActionModelParams(pullModelAction, raw.Params, false); err != nil {
						m.codec.RespondError(msg.ID, -32602, err.Error())
						return false
					}
				}
			}
		}
	}
	if err := json.Unmarshal(msg.Params, v); err != nil {
		m.codec.RespondError(msg.ID, -32602, "invalid params: "+err.Error())
		return false
	}
	return true
}

func (m *Manager) respondOrErr(msg *Message, result any, err error) {
	if err != nil {
		var busy *mutationBusyError
		if errors.As(err, &busy) {
			m.codec.RespondErrorData(msg.ID, -32000, err.Error(), map[string]string{"kind": "busy"})
			return
		}
		m.codec.RespondError(msg.ID, rpcCode(err), err.Error())
		return
	}
	m.codec.Respond(msg.ID, result)
}
