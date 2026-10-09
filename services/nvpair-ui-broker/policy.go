// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// policy.go is the broker side of the fork's node policy (FORK_DESIGN.md §3.1,
// §3.2). The broker owns <appdir>/node-policy.json and is its only writer; the
// proxy enforces it. This file holds the store, the client methods
// policy:get / policy:set, the state mirrored into the proxy (policy,
// availability, residency, saved engine intent, settings drains), and the
// remote relay that lets a paired node read or change this node's policy.
//
// Availability orchestration lives in availability.go, the idle policy and
// start-on-demand in idlepolicy.go, settings drains in settingsdrain.go.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/clustertrust"
	"nvpair-shared/nodepolicy"
)

// Client-facing methods and notifications.
const (
	methodPolicyGet           = "policy:get"
	methodPolicySet           = "policy:set"
	methodSetAvailability     = "node:set-availability"
	notifyPolicyChanged       = "policy:changed"
	notifyAvailabilityChanged = "node:availability-changed"
)

// Engine-manager methods and notifications the policy uses.
const (
	engineWakeMethod          = "engine:wake"
	engineSleepMethod         = "engine:sleep"
	engineIntentMethod        = "engine:intent"
	engineIntentChanged       = "engine:intent-changed"
	engineUnloadModelMethod   = "engine:unload-model"
	engineStateChangedMethod  = "engine:state-changed"
	engineModelsChangedMethod = "engine:models-changed"
	policyRelayRequestMethod  = "policy:request"
	policyRelayReplyMethod    = "policy:reply"
	policyRelayCancelMethod   = "policy:cancel"
)

// remotePolicyMethods maps a client method aimed at another node to the
// engine-manager method that carries it over the pinned ec surface.
var remotePolicyMethods = map[string]string{
	methodPolicyGet:       "engine:remote-policy-get",
	methodPolicySet:       "engine:remote-policy-set",
	methodSetAvailability: "engine:remote-availability-set",
}

// engineCallTimeout bounds one engine-manager call made on the policy's behalf
// (an unload, a sleep, a wake). A wake waits for engine readiness, which can
// take minutes for a large model server.
const engineCallTimeout = 5 * time.Minute

// policyRuntime is everything the node policy tracks. One mutex guards the
// state; pushMu serializes what is sent to the proxy, and every push reads the
// state at send time, so a later change can never be overwritten by an earlier
// one that happened to be sent second.
type policyRuntime struct {
	mu sync.Mutex
	// path is the policy file. Empty keeps the policy in memory only, which is
	// what a Broker built by a test gets unless it asks for a file.
	path    string
	current nodepolicy.Policy
	ready   bool
	// live is the availability the node is actually in. It differs from
	// current.Availability only while draining.
	live         nodepolicy.Availability
	cancelActive bool

	// Admission state from the proxy, and a channel closed on every update so
	// waiters can block on a change without polling.
	admission     nodepolicy.AdmissionState
	haveAdmission bool
	admissionCh   chan struct{}

	// What engine-manager reports: which engines run, what each has loaded,
	// and each engine's saved intent.
	running map[string]bool
	loaded  map[string][]string
	intent  map[string]bool
	drains  map[string]bool

	// Idle bookkeeping: when an engine was seen to start, when it last gained a
	// loaded model, and the baseline the idle policy last acted on.
	runningSince   map[string]time.Time
	residencyAt    map[string]time.Time
	idleUnloadedAt map[string]time.Time
	idleSleptAt    map[string]time.Time
	now            func() time.Time

	// Availability transitions: transitionMu runs one at a time; the newest
	// request bumps transitionGen and cancels the one in progress.
	transitionMu     sync.Mutex
	transitionGen    uint64
	transitionCancel context.CancelFunc
	pauseSlept       map[string]bool
	restoreSkipped   bool

	waking map[string]bool

	pushMu      sync.Mutex
	pendingPush map[string]bool

	relayCancels map[string]context.CancelFunc
}

// nodePolicyRuntime returns the runtime, initializing it with the default
// policy the first time. Serve calls loadNodePolicy before anything reads it.
func (b *Broker) nodePolicyRuntime() *policyRuntime {
	r := &b.nodePolicy
	r.mu.Lock()
	r.initLocked()
	r.mu.Unlock()
	return r
}

func (r *policyRuntime) initLocked() {
	if r.ready {
		return
	}
	r.ready = true
	r.current = nodepolicy.Default()
	r.live = r.current.Availability
	r.admissionCh = make(chan struct{})
	r.running = map[string]bool{}
	r.loaded = map[string][]string{}
	r.intent = map[string]bool{}
	r.drains = map[string]bool{}
	r.runningSince = map[string]time.Time{}
	r.residencyAt = map[string]time.Time{}
	r.idleUnloadedAt = map[string]time.Time{}
	r.idleSleptAt = map[string]time.Time{}
	r.pauseSlept = map[string]bool{}
	r.waking = map[string]bool{}
	r.pendingPush = map[string]bool{}
	r.relayCancels = map[string]context.CancelFunc{}
	if r.now == nil {
		r.now = time.Now
	}
}

// loadNodePolicy reads the policy file at startup. A missing file means the
// default policy. A file that cannot be parsed or validated is kept as
// <file>.bad for the user to inspect and the default is used, so a bad edit
// can never stop the node from starting.
func (b *Broker) loadNodePolicy(path string) {
	r := b.nodePolicyRuntime()
	policy := nodepolicy.Default()
	data, err := os.ReadFile(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		slog.Warn("node policy unreadable; using the default policy", "path", path, "err", err)
	default:
		parsed, perr := nodepolicy.Parse(data)
		if perr != nil {
			bad := path + ".bad"
			if werr := os.WriteFile(bad, data, 0o600); werr != nil {
				slog.Warn("could not keep a copy of the invalid node policy", "path", bad, "err", werr)
			}
			slog.Warn("node policy invalid; kept a copy and using the default policy", "path", path, "copy", bad, "err", perr)
		} else {
			policy = parsed
		}
	}
	r.mu.Lock()
	r.path = path
	r.current = policy
	r.live = policy.Availability
	r.mu.Unlock()
	slog.Info("node policy loaded", "path", path, "availability", policy.Availability)
}

// defaultNodePolicyPath is <appdir>/node-policy.json.
func defaultNodePolicyPath() (string, error) {
	return appdir.Path(nodepolicy.FileName)
}

// saveLocked writes the policy atomically. Caller holds r.mu.
func (r *policyRuntime) saveLocked(policy nodepolicy.Policy) error {
	if r.path == "" {
		return nil
	}
	data, err := json.MarshalIndent(policy, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(r.path), 0o700); err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(r.path), ".node-policy-*")
	if err != nil {
		return err
	}
	tmp := file.Name()
	defer os.Remove(tmp)
	if err = file.Chmod(0o600); err == nil {
		_, err = file.Write(append(data, '\n'))
	}
	if err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmp, r.path)
}

// policyGetResult is the policy:get result.
type policyGetResult struct {
	Policy       nodepolicy.Policy       `json:"policy"`
	Availability nodepolicy.Availability `json:"availability"`
}

// policyParams is the params shape of every policy method.
type policyParams struct {
	NodeID string          `json:"nodeId,omitempty"`
	Policy json.RawMessage `json:"policy,omitempty"`
	State  string          `json:"state,omitempty"`
}

func (b *Broker) localPolicy() policyGetResult {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return policyGetResult{Policy: r.current, Availability: r.live}
}

// setLocalPolicy validates, persists and applies a new policy. Availability is
// not part of what policy:set changes — node:set-availability owns it — so the
// incoming value is ignored and the persisted one kept. Omitted fields take
// their defaults, exactly as when the file is loaded.
func (b *Broker) setLocalPolicy(raw json.RawMessage) (nodepolicy.Policy, error) {
	if len(raw) == 0 {
		return nodepolicy.Policy{}, fmt.Errorf("policy is required")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return nodepolicy.Policy{}, fmt.Errorf("policy must be a JSON object")
	}
	delete(fields, "availability")
	stripped, err := json.Marshal(fields)
	if err != nil {
		return nodepolicy.Policy{}, err
	}
	policy, err := nodepolicy.Parse(stripped)
	if err != nil {
		return nodepolicy.Policy{}, err
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	policy.Availability = r.current.Availability
	if err := policy.Validate(); err != nil {
		r.mu.Unlock()
		return nodepolicy.Policy{}, err
	}
	if err := r.saveLocked(policy); err != nil {
		r.mu.Unlock()
		return nodepolicy.Policy{}, fmt.Errorf("the policy could not be saved: %w", err)
	}
	r.current = policy
	r.mu.Unlock()
	b.schedulePolicyPush(pushPolicy)
	b.notifyClients(notifyPolicyChanged, map[string]any{"nodeId": b.nodeID, "policy": policy})
	return policy, nil
}

// notifyClients sends a notification to every client. These are low-volume
// state changes, so they need no subscription.
func (b *Broker) notifyClients(method string, params any) {
	if b.codec == nil {
		return
	}
	if err := b.codec.Notify(method, params); err != nil {
		slog.Warn("notify clients failed", "method", method, "err", err)
	}
}

// notifyAvailability tells clients the live availability and active count.
func (b *Broker) notifyAvailability() {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	live, active := r.live, r.admission.Active
	r.mu.Unlock()
	b.notifyClients(notifyAvailabilityChanged, map[string]any{"nodeId": b.nodeID, "availability": live, "active": active})
}

// handlePolicyRequest serves the broker-owned node policy methods, and holds
// back engine:wake while the node is not available. It reports whether it
// answered the message; an engine:wake that may proceed falls through to the
// ordinary engine relay.
func (b *Broker) handlePolicyRequest(msg *Message) bool {
	switch msg.Method {
	case methodPolicyGet, methodPolicySet, methodSetAvailability:
		go b.servePolicyRequest(msg)
		return true
	case engineWakeMethod:
		if reason := b.wakeRefusal(); reason != "" {
			_ = b.codec.RespondError(msg.ID, -32000, reason)
			return true
		}
	}
	return false
}

// wakeRefusal is why nothing may be started right now, or "" when it may.
func (b *Broker) wakeRefusal() string {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	live := r.live
	r.mu.Unlock()
	if live != nodepolicy.Available {
		return fmt.Sprintf("this node is %s; engines are not started on demand", live)
	}
	return ""
}

func (b *Broker) servePolicyRequest(msg *Message) {
	var p policyParams
	if len(msg.Params) > 0 && json.Unmarshal(msg.Params, &p) != nil {
		_ = b.codec.RespondError(msg.ID, -32602, "invalid params")
		return
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		b.relayPolicyToPeer(msg, p)
		return
	}
	result, err := b.dispatchPolicy(context.Background(), msg.Method, p)
	if err != nil {
		code := -32000
		var invalid *invalidParamsError
		if errors.As(err, &invalid) {
			code = -32602
		}
		_ = b.codec.RespondError(msg.ID, code, err.Error())
		return
	}
	_ = b.codec.Respond(msg.ID, result)
}

// invalidParamsError marks a request the caller must change before retrying.
type invalidParamsError struct{ err error }

func (e *invalidParamsError) Error() string { return e.err.Error() }

// dispatchPolicy runs one policy method against this node. It is shared by the
// client surface and the relay from paired nodes, neither of which can reach
// another node through it.
func (b *Broker) dispatchPolicy(ctx context.Context, method string, p policyParams) (any, error) {
	switch method {
	case methodPolicyGet:
		return b.localPolicy(), nil
	case methodPolicySet:
		policy, err := b.setLocalPolicy(p.Policy)
		if err != nil {
			return nil, &invalidParamsError{err}
		}
		return map[string]any{"policy": policy}, nil
	case methodSetAvailability:
		state := nodepolicy.Availability(p.State)
		if state != nodepolicy.Available && state != nodepolicy.Paused {
			return nil, &invalidParamsError{fmt.Errorf("state must be %q or %q", nodepolicy.Available, nodepolicy.Paused)}
		}
		availability, err := b.setAvailability(ctx, state)
		if err != nil {
			return nil, err
		}
		return map[string]any{"availability": availability}, nil
	default:
		return nil, fmt.Errorf("unsupported policy method %q", method)
	}
}

// relayPolicyToPeer forwards a policy call aimed at another node through
// engine-manager's pinned ec surface. On success the change is announced to
// this node's clients under the target's nodeId, so a UI watching the cluster
// sees it without polling.
func (b *Broker) relayPolicyToPeer(msg *Message, p policyParams) {
	worker := b.getEngineMgr()
	if worker == nil {
		_ = b.codec.RespondError(msg.ID, -32000, "engine manager unavailable")
		return
	}
	remote := remotePolicyMethods[msg.Method]
	err := worker.RelayRequest(remote, msg.Params, func(result json.RawMessage, rpcErr *RPCError, err error) {
		switch {
		case err != nil:
			_ = b.codec.RespondError(msg.ID, -32000, err.Error())
		case rpcErr != nil:
			_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
		default:
			_ = b.codec.Respond(msg.ID, result)
			b.announceRemotePolicyChange(msg.Method, p.NodeID, result)
		}
	})
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32000, err.Error())
	}
}

func (b *Broker) announceRemotePolicyChange(method, nodeID string, result json.RawMessage) {
	var r struct {
		Policy       json.RawMessage `json:"policy"`
		Availability string          `json:"availability"`
	}
	if json.Unmarshal(result, &r) != nil {
		return
	}
	switch method {
	case methodPolicySet:
		if len(r.Policy) > 0 {
			b.notifyClients(notifyPolicyChanged, map[string]any{"nodeId": nodeID, "policy": r.Policy})
		}
	case methodSetAvailability:
		if r.Availability != "" {
			b.notifyClients(notifyAvailabilityChanged, map[string]any{"nodeId": nodeID, "availability": r.Availability})
		}
	}
}

// policyRelayRequest is a paired node's policy call, relayed by engine-manager.
type policyRelayRequest struct {
	ID     string          `json:"id"`
	Method string          `json:"method"`
	Caller string          `json:"caller,omitempty"`
	Params json.RawMessage `json:"params"`
}

// relayedPolicyMethods maps the relay's method names to the client methods.
var relayedPolicyMethods = map[string]string{
	"get":              methodPolicyGet,
	"set":              methodPolicySet,
	"set-availability": methodSetAvailability,
}

// handlePolicyRelay serves a paired node's policy call. The caller must still
// be pinned, and the call always targets this node: a relayed request never
// reaches relayPolicyToPeer, so it cannot be forwarded again.
func (b *Broker) handlePolicyRelay(raw json.RawMessage) {
	var req policyRelayRequest
	if json.Unmarshal(raw, &req) != nil || req.ID == "" {
		return
	}
	worker := b.getEngineMgr()
	if worker == nil {
		return
	}
	r := b.nodePolicyRuntime()
	ctx, cancel := context.WithCancel(context.Background())
	r.mu.Lock()
	if len(r.relayCancels) >= 64 {
		r.mu.Unlock()
		cancel()
		_ = worker.Notify(policyRelayReplyMethod, map[string]string{"id": req.ID, "error": "too many policy requests"})
		return
	}
	r.relayCancels[req.ID] = cancel
	r.mu.Unlock()
	go func() {
		defer cancel()
		defer func() { r.mu.Lock(); delete(r.relayCancels, req.ID); r.mu.Unlock() }()
		reply := struct {
			ID     string `json:"id"`
			Result any    `json:"result"`
			Error  string `json:"error,omitempty"`
		}{ID: req.ID}
		result, err := b.servePolicyRelay(ctx, req)
		if err != nil {
			reply.Error = err.Error()
		} else {
			reply.Result = result
		}
		_ = worker.Notify(policyRelayReplyMethod, reply)
	}()
}

func (b *Broker) servePolicyRelay(ctx context.Context, req policyRelayRequest) (any, error) {
	method, ok := relayedPolicyMethods[req.Method]
	if !ok {
		return nil, fmt.Errorf("unsupported policy method")
	}
	if req.Caller != "" && !clustertrust.Open(b.clusterDir).HasPin(req.Caller) {
		return nil, fmt.Errorf("the requesting device is no longer paired")
	}
	var p policyParams
	if len(req.Params) > 0 && json.Unmarshal(req.Params, &p) != nil {
		return nil, fmt.Errorf("invalid policy request")
	}
	p.NodeID = ""
	slog.Info("serving a paired node's policy request", "method", method, "caller", req.Caller)
	return b.dispatchPolicy(ctx, method, p)
}

func (b *Broker) cancelPolicyRelay(raw json.RawMessage) {
	var p struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(raw, &p) != nil {
		return
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	cancel := r.relayCancels[p.ID]
	r.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// handlePolicyEngineNotification observes engine-manager's stream for the
// policy. It consumes the policy relay's own notifications and reports true for
// them; everything else is only observed and continues to subscribed clients.
func (b *Broker) handlePolicyEngineNotification(method string, params json.RawMessage) bool {
	switch method {
	case policyRelayRequestMethod:
		b.handlePolicyRelay(params)
		return true
	case policyRelayCancelMethod:
		b.cancelPolicyRelay(params)
		return true
	case engineStateChangedMethod:
		var st struct {
			Engine  string `json:"engine"`
			Running bool   `json:"running"`
		}
		if json.Unmarshal(params, &st) == nil && st.Engine != "" {
			b.observeEngineRunning(st.Engine, st.Running)
		}
	case engineModelsChangedMethod:
		var p struct {
			Models struct {
				LoadedByEngine map[string][]string `json:"loadedByEngine"`
			} `json:"models"`
		}
		if json.Unmarshal(params, &p) == nil {
			b.observeLoaded(p.Models.LoadedByEngine)
		}
	case engineIntentChanged:
		var p struct {
			EnabledByEngine map[string]bool `json:"enabledByEngine"`
		}
		if json.Unmarshal(params, &p) == nil && p.EnabledByEngine != nil {
			b.observeIntent(p.EnabledByEngine)
		}
	}
	return false
}

// observeEngineRunning records a running/stopped transition. A stopped engine
// has nothing loaded, which is what the proxy must see.
func (b *Broker) observeEngineRunning(engine string, running bool) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	was := r.running[engine]
	if running {
		r.running[engine] = true
		if !was {
			r.runningSince[engine] = r.now()
		}
		if _, ok := r.loaded[engine]; !ok {
			r.loaded[engine] = []string{}
		}
	} else {
		delete(r.running, engine)
		delete(r.loaded, engine)
	}
	changed := was != running
	r.mu.Unlock()
	if changed {
		b.schedulePolicyPush(pushResidency)
	}
}

// observeLoaded folds engine-manager's loadedByEngine into the residency. Only
// engines that answered are present, so an absent engine keeps what it had; a
// stop arrives separately as engine:state-changed.
func (b *Broker) observeLoaded(loadedByEngine map[string][]string) {
	if len(loadedByEngine) == 0 {
		return
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	now := r.now()
	for engine, models := range loadedByEngine {
		if !r.running[engine] {
			r.running[engine] = true
			r.runningSince[engine] = now
		}
		prev := r.loaded[engine]
		for _, m := range models {
			if !containsString(prev, m) {
				r.residencyAt[engine] = now
				break
			}
		}
		r.loaded[engine] = append([]string{}, models...)
	}
	r.mu.Unlock()
	b.schedulePolicyPush(pushResidency)
}

func (b *Broker) observeIntent(enabledByEngine map[string]bool) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.intent = make(map[string]bool, len(enabledByEngine))
	for k, v := range enabledByEngine {
		r.intent[k] = v
	}
	r.mu.Unlock()
	b.schedulePolicyPush(pushIntent)
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// refreshEnginePolicySnapshot seeds running engines, residency and saved
// intent from a (re)started engine-manager, whose watcher pushes only changes.
func (b *Broker) refreshEnginePolicySnapshot(w *rpcWorker) {
	if w == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if result, rpcErr, err := w.CallNoTimeout(ctx, engineIntentMethod, nil); err == nil && rpcErr == nil {
		var p struct {
			EnabledByEngine map[string]bool `json:"enabledByEngine"`
		}
		if json.Unmarshal(result, &p) == nil && p.EnabledByEngine != nil {
			b.observeIntent(p.EnabledByEngine)
		}
	}
	if result, rpcErr, err := w.CallNoTimeout(ctx, "engine:get-installed", nil); err == nil && rpcErr == nil {
		var p struct {
			Engines []struct {
				Engine  string `json:"engine"`
				Running bool   `json:"running"`
			} `json:"engines"`
		}
		if json.Unmarshal(result, &p) == nil {
			for _, e := range p.Engines {
				b.observeEngineRunning(e.Engine, e.Running)
			}
		}
	}
	if result, rpcErr, err := w.CallNoTimeout(ctx, "engine:models", nil); err == nil && rpcErr == nil {
		var p struct {
			LoadedByEngine map[string][]string `json:"loadedByEngine"`
		}
		if json.Unmarshal(result, &p) == nil {
			b.observeLoaded(p.LoadedByEngine)
		}
	}
}

// Proxy pushes. Each kind is coalesced: scheduling one that is already pending
// is a no-op, and the push reads the state when it is sent.
const (
	pushPolicy       = "policy"
	pushAvailability = "availability"
	pushResidency    = "residency"
	pushIntent       = "intent"
	pushDrainPrefix  = "drain:"
)

// policyProxy is the proxy process. Every engine's facade lives in it, so any
// published handle is the same process.
func (b *Broker) policyProxy() *proxyProcess {
	for _, profile := range engineProxyProfiles {
		if p := b.engineProxyHandle(profile); p != nil {
			return p
		}
	}
	return nil
}

// schedulePolicyPush sends one kind of state to the proxy in the background.
func (b *Broker) schedulePolicyPush(kind string) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if r.pendingPush[kind] {
		r.mu.Unlock()
		return
	}
	r.pendingPush[kind] = true
	r.mu.Unlock()
	go func() {
		r.pushMu.Lock()
		defer r.pushMu.Unlock()
		r.mu.Lock()
		delete(r.pendingPush, kind)
		r.mu.Unlock()
		b.sendPolicyStateLocked(context.Background(), b.policyProxy(), kind)
	}()
}

// pushPolicyNow sends one kind of state and returns once the proxy answered,
// for the steps that must not continue before the proxy knows.
func (b *Broker) pushPolicyNow(ctx context.Context, kind string) {
	r := b.nodePolicyRuntime()
	r.pushMu.Lock()
	defer r.pushMu.Unlock()
	b.sendPolicyStateLocked(ctx, b.policyProxy(), kind)
}

// replayPolicyToProxy re-sends everything to a freshly started proxy. It runs
// before the proxy's facades are enabled, so a paused node is paused before it
// can admit anything. Availability goes first for the same reason.
func (b *Broker) replayPolicyToProxy(ctx context.Context, pp *proxyProcess) {
	r := b.nodePolicyRuntime()
	r.pushMu.Lock()
	defer r.pushMu.Unlock()
	r.mu.Lock()
	// Work in flight on the old process died with it.
	r.admission = nodepolicy.AdmissionState{}
	r.haveAdmission = false
	r.signalAdmissionLocked()
	kinds := []string{pushAvailability, pushPolicy, pushIntent, pushResidency}
	for engine := range r.drains {
		kinds = append(kinds, pushDrainPrefix+engine)
	}
	r.mu.Unlock()
	for _, kind := range kinds {
		b.sendPolicyStateLocked(ctx, pp, kind)
	}
}

// sendPolicyStateLocked sends the current state of one kind. Caller holds
// pushMu. An older proxy without the method only logs.
func (b *Broker) sendPolicyStateLocked(ctx context.Context, pp *proxyProcess, kind string) {
	if pp == nil {
		return
	}
	r := &b.nodePolicy
	r.mu.Lock()
	var method string
	var params any
	switch {
	case kind == pushPolicy:
		method, params = nodepolicy.MethodSetPolicy, r.current
	case kind == pushAvailability:
		method, params = nodepolicy.MethodSetAvailability, nodepolicy.SetAvailabilityParams{State: r.live, CancelActive: r.cancelActive}
	case kind == pushResidency:
		loaded := make(map[string][]string, len(r.running))
		for engine := range r.running {
			models := append([]string{}, r.loaded[engine]...)
			sort.Strings(models)
			loaded[engine] = models
		}
		method, params = nodepolicy.MethodSetResidency, nodepolicy.SetResidencyParams{LoadedByEngine: loaded}
	case kind == pushIntent:
		intent := make(map[string]bool, len(r.intent))
		for k, v := range r.intent {
			intent[k] = v
		}
		method, params = nodepolicy.MethodSetEngineIntent, nodepolicy.SetEngineIntentParams{EnabledByEngine: intent}
	case len(kind) > len(pushDrainPrefix) && kind[:len(pushDrainPrefix)] == pushDrainPrefix:
		engine := kind[len(pushDrainPrefix):]
		method, params = nodepolicy.MethodSetEngineDrain, nodepolicy.SetEngineDrainParams{Engine: engine, Drain: r.drains[engine]}
	default:
		r.mu.Unlock()
		return
	}
	data, err := json.Marshal(params)
	r.mu.Unlock()
	if err != nil {
		return
	}
	_, rpcErr, err := pp.Call(ctx, method, data)
	switch {
	case err != nil:
		slog.Warn("could not push node policy state to the proxy", "method", method, "err", err)
	case rpcErr != nil:
		slog.Warn("proxy refused node policy state", "method", method, "code", rpcErr.Code, "err", rpcErr.Message)
	}
}

// handleAdmissionNotification takes the proxy's admission/* notifications and
// reports whether method was one of them.
func (b *Broker) handleAdmissionNotification(method string, params json.RawMessage) bool {
	switch method {
	case nodepolicy.NotifyAdmissionState:
		var st nodepolicy.AdmissionState
		if json.Unmarshal(params, &st) != nil {
			slog.Warn("proxy sent an invalid admission state")
			return true
		}
		b.observeAdmission(st)
		return true
	case nodepolicy.NotifyAdmissionUnload:
		var req nodepolicy.UnloadRequest
		if json.Unmarshal(params, &req) != nil || req.Engine == "" {
			slog.Warn("proxy sent an invalid unload request")
			return true
		}
		go b.unloadModels(req.Engine, req.Models, "admission")
		return true
	case nodepolicy.NotifyAdmissionWake:
		var req nodepolicy.WakeRequest
		if json.Unmarshal(params, &req) != nil || req.Engine == "" {
			slog.Warn("proxy sent an invalid wake request")
			return true
		}
		go b.wakeOnDemand(req.Engine)
		return true
	}
	return false
}

func (b *Broker) observeAdmission(st nodepolicy.AdmissionState) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	activeChanged := r.admission.Active != st.Active
	r.admission = st
	r.haveAdmission = true
	r.signalAdmissionLocked()
	draining := r.live == nodepolicy.Draining
	r.mu.Unlock()
	// While draining, clients follow the count down to zero.
	if draining && activeChanged {
		b.notifyAvailability()
	}
}

// signalAdmissionLocked wakes everything waiting on an admission change.
func (r *policyRuntime) signalAdmissionLocked() {
	close(r.admissionCh)
	r.admissionCh = make(chan struct{})
}

// waitAdmission blocks until pred holds for the admission state, timeout
// passes, or ctx ends, and reports whether pred held. Without a proxy, or
// before it has reported anything, nothing is running here.
func (b *Broker) waitAdmission(ctx context.Context, timeout time.Duration, pred func(nodepolicy.AdmissionState) bool) bool {
	r := b.nodePolicyRuntime()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		noProxy := b.policyProxy() == nil
		r.mu.Lock()
		ok := noProxy || !r.haveAdmission || pred(r.admission)
		ch := r.admissionCh
		r.mu.Unlock()
		if ok {
			return true
		}
		select {
		case <-ch:
		case <-timer.C:
			return false
		case <-ctx.Done():
			return false
		}
	}
}

// callEngine makes one engine-manager call on the policy's behalf.
func (b *Broker) callEngine(ctx context.Context, method string, params any) (json.RawMessage, error) {
	w := b.getEngineMgr()
	if w == nil {
		return nil, fmt.Errorf("engine manager unavailable")
	}
	data, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, engineCallTimeout)
	defer cancel()
	result, rpcErr, err := w.CallNoTimeout(ctx, method, data)
	if err != nil {
		return nil, err
	}
	if rpcErr != nil {
		return nil, fmt.Errorf("%s", rpcErr.Message)
	}
	return result, nil
}

// unloadModels unloads models from one engine, one at a time.
func (b *Broker) unloadModels(engine string, models []string, reason string) {
	for _, model := range models {
		if model == "" {
			continue
		}
		if _, err := b.callEngine(context.Background(), engineUnloadModelMethod, map[string]string{"engine": engine, "model": model}); err != nil {
			slog.Warn("model unload failed", "engine", engine, "model", model, "reason", reason, "err", err)
			continue
		}
		slog.Info("model unloaded", "engine", engine, "model", model, "reason", reason)
	}
}
