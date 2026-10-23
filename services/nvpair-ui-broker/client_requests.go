// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"strings"
	"time"

	"nvpair-shared/applog"
	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

// internalClientMethods are worker- and broker-internal methods a public
// client must never call. The generic engine:* / cluster:* relays below would
// otherwise hand them to engine-manager, which serves several of them as its
// own broker-side surface. One list so a new internal method is added once.
var internalClientMethods = map[string]struct{}{
	// Peer relay notifications: only engine-manager relays these, with a
	// verified caller.
	policyRelayRequestMethod:       {},
	policyRelayReplyMethod:         {},
	policyRelayCancelMethod:        {},
	"engine:settings-request":      {},
	"engine:settings-reply":        {},
	"engine:settings-cancel":       {},
	"engine:settings-projection":   {},
	"engine:switch-request":        {},
	"engine:switch-reply":          {},
	"engine:switch-cancel":         {},
	"engine:switch-changed":        {},
	"engine:switch-recovery-ready": {},
	// Switch executor calls belong to the coordinator, which retains the
	// operation lock. Relaying a client's target-start step would bypass it.
	settings.SwitchPrepareMethod:   {},
	settings.SwitchStepMethod:      {},
	settings.SwitchReconcileMethod: {},
	"engine:set-reserved-port":     {},
	"internal:set-reserved-port":   {},
	// Broker/engine-manager coordination.
	nodepolicy.MethodModelMutationRequest: {},
	nodepolicy.MethodModelMutationReply:   {},
	nodepolicy.MethodReconcileResident:    {},
	"engine:dispatch-resident-request":    {},
	"engine:dispatch-resident-reply":      {},
	"engine:node-projection":              {},
	"engine:restore-enabled":              {},
	"engine:prepare-shutdown":             {},
	"engine:get-launch":                   {},
	"engine:preview-launch":               {},
	"engine:configured-ports":             {},
	"engine:configure-launch":             {},
	// The native API key of a local engine. It is outside every relayed
	// namespace already; listing it keeps a future relay rule from reaching it.
	engines.LocalCredentialMethod: {},
}

// forwardLogLevel pushes an already-validated log level down to every
// running child subprocess over its stdin. Each child applies it through
// the shared applog stdin handler. Failures are logged per-child but
// never propagated: a child that won't take the new level (e.g. its pipe
// is mid-teardown, or it never started) must not fail the caller's
// log/set-level request, which has already succeeded for the broker
// itself. Each handle is read through a workersMu-guarded getter or snapshot because
// a supervisor may swap it (on a restart) concurrently with this call.
func (b *Broker) forwardLogLevel(level string) {
	if s := b.getScanner(); s != nil {
		if err := s.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to scanner", "err", err)
		}
	}
	if n := b.getNodeInfo(); n != nil {
		if err := n.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to node-info", "err", err)
		}
	}
	// One iteration over distinct handles, not one per engine: every engine's
	// facade lives in the same process now, so both engine slots resolve to the
	// same handle and a per-engine loop would send this process-scoped
	// notification twice.
	for _, target := range b.livePriorityTargets() {
		if err := target.proxy.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to proxy", "err", err)
		}
	}
	if wm := b.getWorkloadMgr(); wm != nil {
		if err := wm.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to workload-manager", "err", err)
		}
	}
	if ep := b.getErrors(); ep != nil {
		if err := ep.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to nvpair-errors", "err", err)
		}
	}
	// Configuration must also reach a current manager hidden by recovery.
	b.workersMu.Lock()
	em := b.engineMgr
	b.workersMu.Unlock()
	if em != nil {
		if err := em.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to engine-manager", "err", err)
		}
	}
	if mn := b.getManualNodes(); mn != nil {
		if err := mn.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to manual-nodes", "err", err)
		}
	}
	if st := b.getSettings(); st != nil {
		if err := st.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to settings", "err", err)
		}
	}
	if cm := b.getClusterMgr(); cm != nil {
		if err := cm.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to cluster-manager", "err", err)
		}
	}
	if sc := b.getScheduler(); sc != nil {
		if err := sc.SetLogLevel(level); err != nil {
			slog.Warn("failed to forward log/set-level to scheduler", "err", err)
		}
	}
}

func (b *Broker) handleMessage(msg *Message) {
	if msg.Method == applog.SetLevelMethod {
		resolved, err := applog.HandleSetLevelParams(msg.Params)
		if msg.IsRequest() {
			if err != nil {
				b.codec.RespondError(msg.ID, -32602, err.Error())
				return
			}
			b.codec.Respond(msg.ID, map[string]string{"level": resolved})
		}
		if err != nil {
			slog.Warn("log/set-level rejected", "err", err)
		} else {
			slog.Info("log level changed", "level", resolved)
			// The broker fans the (validated) level out to every child it
			// supervises so a single log/set-level call adjusts the whole
			// process tree, not just the broker. Done after the caller's
			// response is on the wire; child forwarding is best-effort.
			b.forwardLogLevel(resolved)
		}
		return
	}

	if !msg.IsRequest() {
		if msg.IsNotification() {
			// A client can inject an errors:report the same way a supervised
			// worker emits one on its stdout — surfacing its own operational
			// errors (e.g. a model-pull failure the engine itself doesn't
			// report) through the shared registry that drives errors:update.
			// dispatchErrorsNotif stamps the origin nodeId / timestamp
			// exactly as it does for workers. Every other inbound
			// notification is dropped.
			if msg.Method == methodErrorsReport && b.dispatchErrorsNotif("client", msg.Method, msg.Params) {
				return
			}
			slog.Debug("ignoring incoming notification", "method", msg.Method)
		}
		return
	}

	if profile, ok := engineProxyProfileForMethod(msg.Method); ok {
		method := strings.TrimPrefix(msg.Method, profile.ComponentName()+":")
		if b.handleEngineProxyBrokerRequest(profile, method, msg) {
			return
		}
	}
	// Reject worker-internal methods before any generic relay can reach a
	// worker that would serve them.
	if _, internal := internalClientMethods[msg.Method]; internal {
		message := "method not found: " + msg.Method
		if msg.Method == "engine:configure-launch" {
			message = "use engine:apply-settings for launch configuration"
		}
		if err := b.codec.RespondError(msg.ID, -32601, message); err != nil {
			log.Printf("failed to reject private method %s: %v", msg.Method, err)
		}
		return
	}
	if b.handlePolicyRequest(msg) {
		return
	}

	switch msg.Method {
	case "engine:switch", "engine:switch-status":
		go b.handleEngineSwitch(msg)
	case "engine:get-settings", "engine:preview-settings", "engine:apply-settings":
		go b.handleEngineSettings(msg)
	case "ping":
		if err := b.codec.Respond(msg.ID, PingResult{
			Pong:     true,
			Version:  Version,
			UptimeMS: time.Since(b.startedAt).Milliseconds(),
		}); err != nil {
			log.Printf("failed to respond to ping: %v", err)
		}

	case "version":
		if err := b.codec.Respond(msg.ID, VersionResult{Version: Version}); err != nil {
			log.Printf("failed to respond to version: %v", err)
		}

	case "discovery:get-nodes":
		if err := b.codec.Respond(msg.ID, GetNodesResult{Nodes: b.store.Snapshot()}); err != nil {
			log.Printf("failed to respond to discovery:get-nodes: %v", err)
		}

	case "discovery:get-node-info":
		go b.handleNodeInfoRPC(msg)

	case "discovery:subscribe":
		b.subMu.Lock()
		wasSubscribed := b.subscribed
		b.subscribed = true
		b.subMu.Unlock()
		if err := b.codec.Respond(msg.ID, SubscriptionResult{Subscribed: true}); err != nil {
			log.Printf("failed to respond to discovery:subscribe: %v", err)
		}
		// On a fresh subscription, push the current snapshot once so the
		// subscriber gets an immediate baseline without a separate
		// discovery:get-nodes round-trip. Sent AFTER the response so the
		// caller sees the ack first. A redundant re-subscribe is a no-op
		// and does not re-emit — the stream is already flowing.
		if !wasSubscribed {
			b.pushNodesChanged()
		}

	case "discovery:unsubscribe":
		b.subMu.Lock()
		b.subscribed = false
		b.subMu.Unlock()
		if err := b.codec.Respond(msg.ID, SubscriptionResult{Subscribed: false}); err != nil {
			log.Printf("failed to respond to discovery:unsubscribe: %v", err)
		}

	case "engine:set-port":
		go b.handleSettingsPortRPC(msg, "")

	case "workloads:subscribe":
		b.workloadsMu.Lock()
		b.workloadsSubscribed = true
		b.workloadsMu.Unlock()
		// No baseline: the broker keeps no workload catalog (it's a relay,
		// not the source of truth), so there's nothing to replay. The
		// subscriber sees events from the next local-origin or peer-origin
		// transition onward.
		if err := b.codec.Respond(msg.ID, SubscriptionResult{Subscribed: true}); err != nil {
			log.Printf("failed to respond to workloads:subscribe: %v", err)
		}

	case "workloads:unsubscribe":
		b.workloadsMu.Lock()
		b.workloadsSubscribed = false
		b.workloadsMu.Unlock()
		if err := b.codec.Respond(msg.ID, SubscriptionResult{Subscribed: false}); err != nil {
			log.Printf("failed to respond to workloads:unsubscribe: %v", err)
		}

	case "workloads:get-initial":
		// Authoritative baseline from the durable store: every current +
		// historic workload as its full workloadInfo, so a (re)connecting or
		// late-subscribing client can seed its view without waiting for the
		// next transition. Ordered by createdAt (see Store.Snapshot).
		snapshot := projectWorkloadSnapshot(b.workloads.Snapshot())
		if snapshot == nil {
			snapshot = []json.RawMessage{}
		}
		if err := b.codec.Respond(msg.ID, workloadsInitialResult{Workloads: snapshot}); err != nil {
			log.Printf("failed to respond to workloads:get-initial: %v", err)
		}

	case "node/add", "node/remove", "nodes/list":
		b.relayToManualNodes(msg)

	case "engine:subscribe":
		b.engineMu.Lock()
		b.engineSubscribed = true
		b.engineMu.Unlock()
		if err := b.codec.Respond(msg.ID, SubscriptionResult{Subscribed: true}); err != nil {
			log.Printf("failed to respond to engine:subscribe: %v", err)
		}

	case "engine:unsubscribe":
		b.engineMu.Lock()
		b.engineSubscribed = false
		b.engineMu.Unlock()
		if err := b.codec.Respond(msg.ID, SubscriptionResult{Subscribed: false}); err != nil {
			log.Printf("failed to respond to engine:unsubscribe: %v", err)
		}

	case methodErrorsGetInitial:
		b.handleErrorsGetInitial(msg)

	case methodErrorsClear:
		b.handleErrorsClear(msg)

	case methodErrorsReport:
		b.handleErrorsReport(msg)

	case "shutdown":
		if err := b.codec.Respond(msg.ID, nil); err != nil {
			log.Printf("failed to respond to shutdown: %v", err)
		}
		slog.Info("shutdown requested via JSON-RPC")
		b.cancel()

	default:
		// Any remaining method under an engine's <component>: prefix is
		// relayed verbatim to that engine's proxy (the reserved broker-local
		// ones — get-status, set-port and the subscription methods — are handled by
		// the profile-driven block above). This makes the broker a thin pass-through
		// for each proxy's whole control plane without enumerating methods.
		//
		// The prefixes are the engines' ComponentName values, so this loop
		// gains a new engine for free rather than needing another branch.
		if profile, ok := engineProxyProfileForMethod(msg.Method); ok {
			b.relayToEngineProxy(profile, msg)
			return
		}
		if strings.HasPrefix(msg.Method, "engine:") {
			b.relayToEngine(msg)
			return
		}
		if strings.HasPrefix(msg.Method, "settings/") {
			b.relayToSettings(msg)
			return
		}
		// cluster:* and nodes:* are the cluster-manager's namespaces; relay
		// them verbatim to nvpair-cluster-manager.
		if strings.HasPrefix(msg.Method, "cluster:") || strings.HasPrefix(msg.Method, "nodes:") {
			b.relayToClusterManager(msg)
			return
		}
		if err := b.codec.RespondError(msg.ID, -32601, fmt.Sprintf("method not found: %s", msg.Method)); err != nil {
			log.Printf("failed to send error response: %v", err)
		}
	}
}

// relayToEngine forwards an engine:<method> request to nvpair-engine-manager
// and relays its eventual response to the client. engine-manager's methods
// are already engine:-prefixed, so the method is forwarded verbatim (no
// prefix translation). Unlike the proxy relay it imposes no broker-side
// timeout: engine lifecycle ops (install, model pull, ...) run for minutes
// and report progress via push events, so the broker waits for the real
// response asynchronously rather than fabricating a timeout — meanwhile
// other client requests keep being served on the read-loop goroutine.
func (b *Broker) relayToEngine(msg *Message) {
	go b.relayToEngineNow(msg)
}

// Explicit local On must serialize with switch obligation discharge even when
// the caller leaves the engine's port unchanged.
func engineLocalOnRequest(method string, params json.RawMessage) bool {
	switch method {
	case "engine:start", "engine:restart":
		return true
	case "engine:install":
		var request struct {
			Start bool `json:"start"`
		}
		return json.Unmarshal(params, &request) == nil && request.Start
	default:
		return false
	}
}

type pendingInstallStart struct {
	engine     string
	superseded bool
}

// Called only after local Off (one engine) or local pause (engine "", every
// engine) admission succeeds, under engineConfigMu.
func (b *Broker) supersedeInstallStartsLocked(engine string) {
	for pending := range b.pendingInstallStarts {
		if engine == "" || pending.engine == engine {
			pending.superseded = true
		}
	}
}

func (b *Broker) relayToEngineNow(msg *Message) {
	if needsOllamaPortGate(msg.Method, msg.Params) && b.ollamaFacadeIsPendingBackend() {
		go func() {
			select {
			case <-b.ollamaPortReady:
				// Recovery opens the general port gate before a best-effort live
				// rebind can finish. Never probe if the proxy still owns :11434.
				if b.ollamaFacadeIsPendingBackend() {
					_ = b.codec.RespondError(msg.ID, -32000, "Ollama port setup is still reconciling; retry")
					return
				}
				b.relayToEngine(msg)
			case <-time.After(rpcWorkerCallTimeout):
				_ = b.codec.RespondError(msg.ID, -32000, "Ollama port setup did not finish; retry")
			}
		}()
		return
	}
	if needsLMStudioPortGate(msg.Method, msg.Params) && b.lmstudioPortOwnershipPending() {
		go func() {
			select {
			case <-b.lmstudioPortReady:
				b.relayToEngine(msg)
			case <-time.After(rpcWorkerCallTimeout):
				_ = b.codec.RespondError(msg.ID, -32000, "LM Studio port setup did not finish; retry")
			}
		}()
		return
	}

	var installStart *pendingInstallStart
	if msg.Method == "engine:install" && engineLocalOnRequest(msg.Method, msg.Params) {
		// Preserve early proxy-port refusal before doing installation work.
		// Normal start admission rechecks after preparation has completed.
		if engine, port, assigned := enginePortAssignmentRequest(msg.Method, msg.Params); assigned && b.rejectProxyHeldPort(msg, port, engine) {
			return
		}
		// Preparation can take minutes. It must not hold either global lock,
		// and must never leave a manager-owned unguarded start continuation.
		var params map[string]json.RawMessage
		var engine string
		if err := json.Unmarshal(msg.Params, &params); err != nil {
			_ = b.codec.RespondError(msg.ID, -32602, err.Error())
			return
		}
		if err := json.Unmarshal(params["engine"], &engine); err != nil {
			_ = b.codec.RespondError(msg.ID, -32602, "engine is required")
			return
		}
		installStart = &pendingInstallStart{engine: engine}
		b.engineConfigMu.Lock()
		if b.pendingInstallStarts == nil {
			b.pendingInstallStarts = make(map[*pendingInstallStart]struct{})
		}
		b.pendingInstallStarts[installStart] = struct{}{}
		b.engineConfigMu.Unlock()
		defer func() {
			b.engineConfigMu.Lock()
			delete(b.pendingInstallStarts, installStart)
			b.engineConfigMu.Unlock()
		}()
		params["start"] = json.RawMessage(`false`)
		preparation, err := json.Marshal(params)
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32602, err.Error())
			return
		}
		em := b.getEngineMgr()
		if em == nil {
			_ = b.codec.RespondError(msg.ID, -32000, "engine-manager not available")
			return
		}
		_, rpcErr, err := em.CallNoTimeout(context.Background(), "engine:install", preparation)
		if err != nil {
			_ = b.codec.RespondError(msg.ID, -32000, fmt.Sprintf("engine call failed: %v", err))
			return
		}
		if rpcErr != nil {
			_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
			return
		}
		// Start and combined install/start both return Manager.runOp's Status.
		// Preserve every original option (including port and bind) for start.
		start := *msg
		start.Method = "engine:start"
		msg = &start
	}
	requestedEngine, requestedPort, isEnginePortAssignment := enginePortAssignmentRequest(msg.Method, msg.Params)
	lifecycleLocked := false
	localOffEngine := ""
	localOffOperationID := ""
	offOperationLocked := false
	if msg.Method == "engine:stop" {
		var target struct {
			Engine string `json:"engine"`
		}
		if json.Unmarshal(msg.Params, &target) == nil && target.Engine != "" {
			remote, reserved, operationID, err := b.admitLocalEngineOff(target.Engine)
			if err != nil {
				_ = b.codec.RespondError(msg.ID, -32000, err.Error())
				return
			}
			if reserved && !remote {
				_ = b.codec.RespondError(msg.ID, -32000, "Engine switch reserves this engine (busy).")
				return
			}
			if remote {
				localOffEngine = target.Engine
				localOffOperationID = operationID
			} else {
				offOperationLocked = true
				defer b.settingsApplyMu.Unlock()
			}
		}
	}
	if localOffEngine == "" && (isEnginePortAssignment || msg.Method == "engine:stop" || engineLocalOnRequest(msg.Method, msg.Params)) {
		if !offOperationLocked {
			b.settingsApplyMu.Lock()
			defer b.settingsApplyMu.Unlock()
		}
		b.engineConfigMu.Lock()
		defer b.engineConfigMu.Unlock()
		lifecycleLocked = true
		var target struct {
			Engine string `json:"engine"`
		}
		if err := json.Unmarshal(msg.Params, &target); err != nil || b.pendingSwitchLocked(target.Engine) != nil {
			_ = b.codec.RespondError(msg.ID, -32000, "Engine switch recovery reserves this engine.")
			return
		}
		if installStart != nil && installStart.superseded {
			_ = b.codec.RespondError(msg.ID, -32000, "Engine install start superseded by local Off or pause (busy).")
			return
		}
		defer b.reconcileLegacySettingsLocked()
	}
	isOllamaPortAssignment := isEnginePortAssignment && requestedEngine == "ollama"
	if isEnginePortAssignment && b.rejectProxyHeldPort(msg, requestedPort, requestedEngine) {
		return
	}
	if isOllamaPortAssignment && requestedPort == managedOllamaFacadePort && b.ollamaState().managedFacade.Load() {
		if err := b.codec.RespondError(msg.ID, -32000, "port 11434 is reserved by the managed Ollama proxy; choose another backend port or disable managed port ownership and restart NVPAIR"); err != nil {
			log.Printf("failed to reject conflicting Ollama port assignment: %v", err)
		}
		return
	}
	requestedLMStudioPort, isLMStudioSetPort := lmstudioSetPortRequest(msg.Method, msg.Params)
	if isLMStudioSetPort && requestedLMStudioPort == managedLMStudioFacadePort && b.lmstudioState().managedFacade.Load() {
		if err := b.codec.RespondError(msg.ID, -32000, "port 1234 is reserved by the managed LM Studio proxy; choose another backend port or disable managed port ownership and restart NVPAIR"); err != nil {
			log.Printf("failed to reject conflicting LM Studio engine:set-port: %v", err)
		}
		return
	}

	em := b.getEngineMgr()
	if localOffEngine != "" {
		// Recovery hides the replacement from forward callers; local Off must
		// still reach its durable veto owner while that reservation is held.
		b.workersMu.Lock()
		em = b.engineMgr
		b.workersMu.Unlock()
	}
	if em == nil {
		if err := b.codec.RespondError(msg.ID, -32000, "engine-manager not available"); err != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, err)
		}
		return
	}
	b.workersMu.Lock()
	source := b.engineMgrSource
	if b.engineMgr != em {
		source = nil
	}
	b.workersMu.Unlock()
	// Capture the client's id/method for the async response closure; the
	// read loop allocates a fresh Message per request so these are stable.
	id := msg.ID
	method := msg.Method
	params := msg.Params
	if localOffOperationID != "" {
		// This Off was admitted against this remote reservation. Once it has
		// settled, the delayed request must not stop a later local On or switch.
		var err error
		params, err = json.Marshal(struct {
			Engine            string `json:"engine"`
			SwitchOperationID string `json:"switchOperationId"`
		}{localOffEngine, localOffOperationID})
		if err != nil {
			_ = b.codec.RespondError(id, -32000, err.Error())
			return
		}
		b.engineConfigMu.Lock()
		// A delayed scoped stop is a no-op after durable discharge. Do not
		// withdraw the identity of a later local On or switch in that case.
		c := b.switchCoordinator()
		withdrawn := c.withdrawnIdentityID == localOffOperationID && c.withdrawnIdentityProxy == b.policyProxy()
		confirmWithdrawal := false
		if obligation, ok := c.store.journal.LocalObligations[localOffOperationID]; ok && !obligation.Discharged && !withdrawn {
			b.forgetSettingsIdentity(localOffEngine)
			confirmWithdrawal = true
		}
		b.engineConfigMu.Unlock()
		if confirmWithdrawal {
			err = b.pushModelIdentityConfirmed(context.Background())
		}
		if err != nil {
			_ = b.codec.RespondError(id, -32000, err.Error())
			return
		}
	}
	if lifecycleLocked {
		b.engineConfigMu.Unlock()
	}
	result, rpcErr, err := em.CallNoTimeout(context.Background(), method, params)
	if lifecycleLocked {
		b.engineConfigMu.Lock()
	}
	if localOffEngine != "" {
		// The manager has now persisted the veto and cancelled its startup;
		// only then cancel the coordinator's drain/response wait.
		b.cancelRemoteEngineSwitchOperation(localOffEngine, localOffOperationID)
	}
	switch {
	case err != nil:
		if e := b.codec.RespondError(id, -32000, fmt.Sprintf("engine call failed: %v", err)); e != nil {
			log.Printf("failed to relay engine error for %s: %v", method, e)
		}
	case rpcErr != nil:
		if e := b.codec.RespondError(id, rpcErr.Code, rpcErr.Message); e != nil {
			log.Printf("failed to relay engine error for %s: %v", method, e)
		}
	default:
		// A manager can return an unchanged running status without emitting a
		// state-change notification. Seed the policy from the authoritative
		// response, fenced to the manager that answered this request.
		switch method {
		case "engine:status", "engine:start", "engine:stop", "engine:restart", "engine:install", "engine:uninstall":
			var status struct {
				Engine  string `json:"engine"`
				Running *bool  `json:"running"`
			}
			if json.Unmarshal(result, &status) == nil && status.Engine != "" && status.Running != nil {
				b.observeEngineRunningFrom(source, status.Engine, *status.Running)
			}
		}
		if isOllamaPortAssignment {
			var status struct {
				Port int `json:"port"`
			}
			if json.Unmarshal(result, &status) == nil && status.Port > 0 {
				b.ollamaState().backendPort.Store(int32(status.Port))
			}
		}
		if isLMStudioSetPort {
			b.lmstudioState().backendPort.Store(int32(requestedLMStudioPort))
		}
		if e := b.codec.Respond(id, result); e != nil {
			log.Printf("failed to relay engine result for %s: %v", method, e)
		}
	}
}

// relayToSettings forwards a settings/<method> request to nvpair-node-settings
// and relays its response straight back. Settings get/set ops are local and
// fast, so unlike the engine relay this uses the bounded synchronous Call
// (the same shape as the proxy relay). The method is forwarded verbatim —
// nvpair-node-settings' methods are already settings/-prefixed.
func (b *Broker) relayToSettings(msg *Message) {
	st := b.getSettings()
	if st == nil {
		if err := b.codec.RespondError(msg.ID, -32000, "node-settings not available"); err != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, err)
		}
		return
	}
	result, rpcErr, err := st.Call(context.Background(), msg.Method, msg.Params)
	switch {
	case err != nil:
		if err := b.codec.RespondError(msg.ID, -32000, fmt.Sprintf("settings call failed: %v", err)); err != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, err)
		}
	case rpcErr != nil:
		if err := b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message); err != nil {
			log.Printf("failed to relay settings error for %s: %v", msg.Method, err)
		}
	default:
		if err := b.codec.Respond(msg.ID, result); err != nil {
			log.Printf("failed to relay settings result for %s: %v", msg.Method, err)
		}
	}
}

// relayToManualNodes forwards a manual-node request (node/add, node/remove,
// nodes/list) to nvpair-manual-nodes and relays its response back. node/add
// kicks off an initial probe, so the relay uses the no-timeout async
// RelayRequest rather than the bounded Call. The resulting node/discovered
// notification is merged into the discovery store by
// forwardManualNodesNotification, so the snapshot stays in sync.
func (b *Broker) relayToManualNodes(msg *Message) {
	mn := b.getManualNodes()
	if mn == nil {
		if err := b.codec.RespondError(msg.ID, -32000, "manual-nodes not available"); err != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, err)
		}
		return
	}
	id := msg.ID
	method := msg.Method
	relayErr := mn.RelayRequest(method, msg.Params, func(result json.RawMessage, rpcErr *RPCError, err error) {
		switch {
		case err != nil:
			if e := b.codec.RespondError(id, -32000, fmt.Sprintf("manual-nodes call failed: %v", err)); e != nil {
				log.Printf("failed to relay manual-nodes error for %s: %v", method, e)
			}
		case rpcErr != nil:
			if e := b.codec.RespondError(id, rpcErr.Code, rpcErr.Message); e != nil {
				log.Printf("failed to relay manual-nodes error for %s: %v", method, e)
			}
		default:
			if e := b.codec.Respond(id, result); e != nil {
				log.Printf("failed to relay manual-nodes result for %s: %v", method, e)
			}
		}
	})
	if relayErr != nil {
		if err := b.codec.RespondError(msg.ID, -32000, fmt.Sprintf("manual-nodes call failed: %v", relayErr)); err != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, err)
		}
	}
}

// relayToClusterManager forwards a cluster:* / nodes:* request to
// nvpair-cluster-manager verbatim and maps its response (result or JSON-RPC
// error) straight back to the client. Cluster get/set + pairing ops are
// local request/response, so it uses the bounded synchronous Call (the same
// shape as the proxy/settings relays). If no cluster-manager is supervised
// the client gets a clear error rather than a silent hang.
func (b *Broker) relayToClusterManager(msg *Message) {
	if msg.Method == setLocalServicesMethod {
		if err := b.codec.RespondError(msg.ID, -32601, "method not found: "+msg.Method); err != nil {
			log.Printf("failed to reject private cluster-manager method: %v", err)
		}
		return
	}
	cm := b.getClusterMgr()
	if cm == nil {
		if err := b.codec.RespondError(msg.ID, -32000, "nvpair-cluster-manager not available"); err != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, err)
		}
		return
	}
	result, rpcErr, err := cm.Call(context.Background(), msg.Method, msg.Params)
	switch {
	case err != nil:
		if err := b.codec.RespondError(msg.ID, -32000, fmt.Sprintf("cluster-manager call failed: %v", err)); err != nil {
			log.Printf("failed to respond to %s: %v", msg.Method, err)
		}
	case rpcErr != nil:
		if err := b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message); err != nil {
			log.Printf("failed to relay cluster-manager error for %s: %v", msg.Method, err)
		}
	default:
		if err := b.codec.Respond(msg.ID, result); err != nil {
			log.Printf("failed to relay cluster-manager result for %s: %v", msg.Method, err)
		}
	}
}
