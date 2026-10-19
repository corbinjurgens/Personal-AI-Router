// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"reflect"
	"sync"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

const (
	residentDispatchRequestMethod = "engine:dispatch-resident-request"
	residentDispatchReplyMethod   = "engine:dispatch-resident-reply"
	maxResidentPending            = 32
	maxResidentOperations         = 256
)

type residentDispatchRequest struct {
	ID     string                            `json:"id"`
	Params nodepolicy.DispatchResidentParams `json:"params"`
}

type residentDispatchReply struct {
	ID     string                             `json:"id"`
	Result *nodepolicy.DispatchResidentResult `json:"result,omitempty"`
	Error  *RPCError                          `json:"error,omitempty"`
}

// Keep validated final-grant reasons internal; the reconcile wire is unchanged.
type residentDispatchRefusal struct {
	reason nodepolicy.DispatchResidentReason
}

func (residentDispatchRefusal) Error() string { return "dispatch-refused" }

// The parent codec sends synchronously. No inventory, publication or content
// lock is held while it sends or waits. Only an internal notification can reply.
type residentDispatchRelay struct {
	mu      sync.Mutex
	pending map[string]chan residentDispatchReply
	send    func(string, any) error
}

func (r *residentDispatchRelay) call(ctx context.Context, p nodepolicy.DispatchResidentParams) error {
	if r == nil || r.send == nil {
		return errors.New("dispatch-unavailable")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	id := newOpID()
	ch := make(chan residentDispatchReply, 1)
	r.mu.Lock()
	if len(r.pending) >= maxResidentPending {
		r.mu.Unlock()
		return errors.New("dispatch-busy")
	}
	if r.pending == nil {
		r.pending = make(map[string]chan residentDispatchReply)
	}
	r.pending[id] = ch
	r.mu.Unlock()
	defer func() { r.mu.Lock(); delete(r.pending, id); r.mu.Unlock() }()
	if err := r.send(residentDispatchRequestMethod, residentDispatchRequest{ID: id, Params: p}); err != nil {
		return err
	}
	select {
	case reply := <-ch:
		if reply.Error != nil || reply.Result == nil || reply.Result.Validate() != nil || !reply.Result.Dispatched {
			if reply.Error == nil && reply.Result != nil && reply.Result.Validate() == nil {
				return residentDispatchRefusal{reason: reply.Result.Reason}
			}
			return errors.New("dispatch-refused")
		}
		// A queued grant cannot authorize a cancelled or expired operation.
		return ctx.Err()
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *residentDispatchRelay) reply(raw json.RawMessage) {
	var p residentDispatchReply
	d := json.NewDecoder(bytes.NewReader(raw))
	// encoding/json otherwise accepts duplicate and case-variant envelope keys.
	start, err := d.Token()
	if err != nil || start != json.Delim('{') {
		return
	}
	seen := map[string]bool{}
	for d.More() {
		token, err := d.Token()
		key, ok := token.(string)
		if err != nil || !ok || seen[key] || (key != "id" && key != "result" && key != "error") {
			return
		}
		seen[key] = true
		var value json.RawMessage
		if d.Decode(&value) != nil || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return
		}
	}
	if _, err := d.Token(); err != nil {
		return
	}
	d = json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if d.Decode(&p) != nil || d.Decode(&struct{}{}) != io.EOF || p.ID == "" {
		return
	}
	// Invalid envelopes can only refuse, never authorize.
	if (p.Result == nil) == (p.Error == nil) {
		p.Result = nil
		p.Error = &RPCError{Code: -32602, Message: "invalid dispatch reply"}
	}
	r.mu.Lock()
	ch := r.pending[p.ID]
	r.mu.Unlock()
	if ch != nil {
		select {
		case ch <- p:
		default:
		}
	}
}

type residentOperation struct {
	params   nodepolicy.ReconcileResidentParams
	deadline time.Time
	done     chan struct{}
	result   nodepolicy.ReconcileResidentResult
}

type residentOperations struct {
	mu  sync.Mutex
	ops map[string]*residentOperation
}

// Entries coalesce for the request's original lifetime, including completion.
// Changed payloads conflict; a full table refuses rather than evicting live IDs.
func (e *Executor) reconcileResident(ctx context.Context, p nodepolicy.ReconcileResidentParams) (nodepolicy.ReconcileResidentResult, error) {
	if err := p.Validate(); err != nil {
		return nodepolicy.ReconcileResidentResult{}, err
	}
	r := &e.residents
	r.mu.Lock()
	now := time.Now()
	for id, op := range r.ops {
		select {
		case <-op.done:
			if !now.Before(op.deadline) {
				delete(r.ops, id)
			}
		default:
		}
	}
	if op := r.ops[p.OperationID]; op != nil {
		if !reflect.DeepEqual(op.params, p) {
			r.mu.Unlock()
			return nodepolicy.ReconcileResidentResult{}, errors.New("resident-operation-conflict")
		}
		r.mu.Unlock()
		<-op.done
		return op.result, nil
	}
	if len(r.ops) >= maxResidentOperations {
		r.mu.Unlock()
		return nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}, nil
	}
	if r.ops == nil {
		r.ops = make(map[string]*residentOperation)
	}
	op := &residentOperation{params: p, deadline: now.Add(time.Duration(p.TimeoutMs) * time.Millisecond), done: make(chan struct{})}
	r.ops[p.OperationID] = op
	r.mu.Unlock()
	ctx, cancel := context.WithDeadline(ctx, op.deadline)
	defer cancel()
	op.result = e.executeResident(ctx, p)
	close(op.done)
	return op.result, nil
}

func (m *Manager) runResident(ctx context.Context, msg *Message) {
	var p nodepolicy.ReconcileResidentParams
	// The shared decoder rejects duplicate, case-variant, null and unknown keys.
	if json.Unmarshal(msg.Params, &p) != nil || p.Validate() != nil {
		m.codec.RespondError(msg.ID, -32602, "invalid resident request")
		return
	}
	result, err := m.exec.reconcileResident(ctx, p)
	if err != nil {
		m.codec.RespondError(msg.ID, -32010, "resident operation conflicts")
		return
	}
	m.codec.Respond(msg.ID, result)
}

// residentLoadCall accepts only the explicit, prompt-free Ollama load protocol.
// Fixed extras cannot silently override the profile or add inference input.
func (e *Executor) residentLoadCall(p nodepolicy.ReconcileResidentParams) (Action, string, json.RawMessage, bool) {
	m, ok := e.reg.Get(p.Engine)
	if !ok || m.FamilyOrDefault() != "ollama" || p.Load.Validate() != nil {
		return Action{}, "", nil, false
	}
	b, ok := m.Operations[engines.OperationLoad]
	if !ok || b.modelParam() != "model" {
		return Action{}, "", nil, false
	}
	a, ok := m.Actions[b.Action]
	if !ok || a.HTTP == nil || a.HTTP.Method != http.MethodPost || a.HTTP.Path != "/api/generate" || a.HTTP.ParamsIn != "" ||
		a.RestartAfter || len(a.Cmd) != 0 || a.RemovePath != nil || a.ModelResolution != "" || a.ProgressProtocol != "" || m.mutatesModelContent(b.Action) {
		return Action{}, "", nil, false
	}
	for key, value := range b.Params {
		if key != "stream" || !bytes.Equal(bytes.TrimSpace(value), []byte("false")) {
			return Action{}, "", nil, false
		}
	}
	// Ollama accepts numeric keep_alive as seconds, and num_ctx as an integer.
	body, err := json.Marshal(struct {
		Model     string `json:"model"`
		Stream    bool   `json:"stream"`
		KeepAlive int    `json:"keep_alive"`
		Options   struct {
			Context int `json:"num_ctx"`
		} `json:"options"`
	}{Model: engines.NormalizeModelName(p.Model, m.ModelNamingOrDefault()), KeepAlive: p.Load.KeepAliveSeconds, Options: struct {
		Context int `json:"num_ctx"`
	}{p.Load.ContextLength}})
	return a, b.Action, body, err == nil
}

func (e *Executor) residentIdentity(p nodepolicy.ReconcileResidentParams) bool {
	m, ok := e.reg.Get(p.Engine)
	if !ok {
		return false
	}
	model := engines.NormalizeModelName(p.Model, m.ModelNamingOrDefault())
	id, ok := e.modelIdentity().Engines[p.Engine]
	if !ok || id.EngineRunID != p.Identity.EngineRunID || id.LaunchDigest == "" {
		return false
	}
	found := false
	for name, digest := range id.ModelDigests {
		if engines.NormalizeModelName(name, m.ModelNamingOrDefault()) == model {
			if found || digest != p.Identity.ModelDigest {
				return false
			}
			found = true
		}
	}
	// The broker owns SettingsFingerprint. StrictModelIdentity has no separate
	// launch field: the final proxy grant checks qualification against the
	// engine's full identity. Here manager-owned run/content identity is compared.
	return found
}

func (e *Executor) residentCheck(p nodepolicy.ReconcileResidentParams) string {
	if p.Reservation.ManagerIncarnation != e.switchIncarnation {
		return "unqualified"
	}
	if err := e.remoteVeto(p.Engine); err != nil {
		if errors.Is(err, errNoProjection) {
			return "projection-unknown"
		}
		paused, _ := e.nodeProjection.pausedLocally()
		if paused {
			return "paused-local"
		}
		return "engine-off-local"
	}
	e.desired.mu.Lock()
	state, err := e.desired.load()
	e.desired.mu.Unlock()
	if err != nil {
		return "observation-unknown"
	}
	if cp := state.PendingSwitch; cp != nil && cp.Phase != "released" && (cp.SourceEngine == p.Engine || cp.TargetEngine == p.Engine) {
		return "switch-busy"
	}
	if !state.Engines[p.Engine] {
		return "engine-off"
	}
	m, ok := e.reg.Get(p.Engine)
	if !ok {
		return "unsupported-load"
	}
	st, err := e.state(p.Engine)
	if err != nil {
		return "engine-off"
	}
	st.mu.Lock()
	running := st.running
	st.mu.Unlock()
	if !running {
		return "engine-off"
	}
	model := engines.NormalizeModelName(p.Model, m.ModelNamingOrDefault())
	installed := false
	for _, name := range e.cachedInventory()[p.Engine] {
		if engines.NormalizeModelName(name, m.ModelNamingOrDefault()) == model {
			installed = true
		}
	}
	if !installed {
		return "not-installed"
	}
	if !e.residentIdentity(p) {
		return "unqualified"
	}
	if _, _, _, ok := e.residentLoadCall(p); !ok {
		return "unsupported-load"
	}
	read, ok := m.Actions["loaded_models"]
	if !ok || read.Result == nil || read.HTTP == nil || read.HTTP.Method != http.MethodGet || read.HTTP.Path != "/api/ps" ||
		read.RestartAfter || len(read.Cmd) != 0 || read.RemovePath != nil || read.ModelResolution != "" || read.ProgressProtocol != "" || m.mutatesModelContent("loaded_models") {
		return "unsupported-load"
	}
	return ""
}

// Join opMu without extending the caller's absolute deadline. A cancelled
// waiter releases the mutex when the earlier lifecycle operation completes.
func residentLock(ctx context.Context, mu *sync.Mutex) bool {
	if mu.TryLock() {
		return true
	}
	joined := make(chan struct{})
	go func() {
		mu.Lock()
		select {
		case joined <- struct{}{}:
		case <-ctx.Done():
			mu.Unlock()
		}
	}()
	select {
	case <-joined:
		return true
	case <-ctx.Done():
		return false
	}
}

func (e *Executor) executeResident(ctx context.Context, p nodepolicy.ReconcileResidentParams) nodepolicy.ReconcileResidentResult {
	refused := nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentNotDispatched}
	logRefusal := func(reason string, level slog.Level) {
		slog.Log(ctx, level, "resident operation refused", "operationId", p.OperationID, "engine", p.Engine, "model", p.Model, "reason", reason)
	}
	contextReason := func() string {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return string(nodepolicy.ModelReasonOperationTimeout)
		}
		return "operation-cancelled"
	}
	if ctx.Err() != nil {
		logRefusal(contextReason(), slog.LevelInfo)
		return refused
	}
	if reason := e.residentCheck(p); reason != "" {
		logRefusal(reason, slog.LevelInfo)
		return refused
	}
	ctx, end, err := e.beginRemote(ctx, p.Engine)
	if err != nil {
		logRefusal("remote-veto", slog.LevelInfo)
		return refused
	}
	defer end()
	done, err := e.admitMutation(p.Engine)
	if err != nil {
		logRefusal("mutation-refused", slog.LevelInfo)
		return refused
	}
	defer done()
	st, err := e.state(p.Engine)
	if err != nil {
		logRefusal(string(nodepolicy.ModelReasonEngineOff), slog.LevelInfo)
		return refused
	}
	// E3 tracks us while waiting for the lifecycle lock. No content guard:
	// residency never changes installed content or withdraws B2 identity.
	if !residentLock(ctx, &st.opMu) {
		logRefusal(contextReason(), slog.LevelInfo)
		return refused
	}
	defer st.opMu.Unlock()
	if reason := e.residentCheck(p); reason != "" {
		logRefusal(reason, slog.LevelInfo)
		return refused
	}
	if ctx.Err() != nil {
		logRefusal(contextReason(), slog.LevelInfo)
		return refused
	}
	a, action, body, ok := e.residentLoadCall(p)
	if !ok || e.residentParent == nil {
		if !ok {
			logRefusal(string(nodepolicy.ModelReasonUnsupportedLoad), slog.LevelInfo)
		} else {
			logRefusal("dispatch-unavailable", slog.LevelWarn)
		}
		return refused
	}
	grant := nodepolicy.DispatchResidentParams{ReservationID: p.Reservation.ReservationID, ProxyRunID: p.Reservation.ProxyRunID,
		ManagerIncarnation: e.switchIncarnation, OperationID: p.OperationID, PolicyRevision: p.PolicyRevision, Engine: p.Engine, Model: p.Model, Identity: p.Identity}
	if err := e.residentParent(ctx, grant); err != nil {
		var refusal residentDispatchRefusal
		switch {
		case errors.As(err, &refusal):
			logRefusal(string(refusal.reason), slog.LevelInfo)
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
			logRefusal(contextReason(), slog.LevelInfo)
		default:
			logRefusal("dispatch-relay-failed", slog.LevelWarn)
		}
		return refused
	}
	if ctx.Err() != nil {
		logRefusal(contextReason(), slog.LevelInfo)
		return refused
	}
	// Veto may have arrived during the relay. Never use a queued grant to
	// bypass the local authority or dispatch under withdrawn identity.
	if reason := e.residentCheck(p); reason != "" {
		logRefusal(reason, slog.LevelInfo)
		return refused
	}
	_, loadErr := e.dispatchAction(ctx, st, p.Engine, action, a, body)
	e.pokeLoaded()
	result := nodepolicy.ReconcileResidentResult{Outcome: nodepolicy.ResidentUnknown}
	// Readback starts after the call, never from the watcher's last-good cache.
	// The same absolute deadline bounds readback even after a lost response.
	loadedAction, haveLoaded := st.manifest.Actions["loaded_models"]
	if ctx.Err() == nil && haveLoaded && loadedAction.Result != nil && e.residentIdentity(p) {
		raw, readErr := e.Action(ctx, p.Engine, "loaded_models", nil)
		if readErr == nil && ctx.Err() == nil && e.residentIdentity(p) {
			if names, known := extractStringsResult(raw, loadedAction.Result); known {
				model := engines.NormalizeModelName(p.Model, st.manifest.ModelNamingOrDefault())
				loaded := false
				for _, name := range names {
					if engines.NormalizeModelName(name, st.manifest.ModelNamingOrDefault()) == model {
						loaded = true
					}
				}
				result.Observation = &nodepolicy.ResidentLoadedObservation{ManagerIncarnation: e.switchIncarnation, Generation: e.residentGeneration.Add(1), Engine: p.Engine, Model: model, Identity: p.Identity, Loaded: loaded}
				if loaded {
					result.Outcome = nodepolicy.ResidentLoaded
					return result
				}
			}
		}
	}
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		logRefusal(contextReason(), slog.LevelInfo)
		result.Outcome = nodepolicy.ResidentTimeout
	case ctx.Err() != nil:
		logRefusal(contextReason(), slog.LevelInfo)
		result.Outcome = nodepolicy.ResidentCancelled
	case result.Observation != nil:
		result.Outcome = nodepolicy.ResidentFailed
	case loadErr != nil:
		var httpErr *ActionHTTPError
		if errors.As(loadErr, &httpErr) {
			result.Outcome = nodepolicy.ResidentFailed
		}
	}
	return result
}
