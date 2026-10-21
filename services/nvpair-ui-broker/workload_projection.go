// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"time"
	"unicode"
	"unicode/utf8"

	"nvpair-shared/nodepolicy"

	"nvpair-ui-broker/workloadstore"
)

// projectWorkloadFrame is the workload wire boundary. Diagnostics deliberately
// contain only fixed categories and counts, never decoder errors or input keys.
func projectWorkloadFrame(method string, params json.RawMessage) (json.RawMessage, string, int) {
	d := json.NewDecoder(bytes.NewReader(params))
	d.UseNumber()
	if !walkWorkloadJSON(d) {
		return nil, "malformed", 0
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, "malformed", 0
	}
	var env map[string]json.RawMessage
	if json.Unmarshal(params, &env) != nil || env == nil {
		return nil, "malformed", 0
	}
	fields := env
	removal := method == "workloads:remove"
	if !removal {
		fields = nil
		if method != "workloads:upsert" && !proxyWorkloadMethods[method] {
			return nil, "method", 0
		}
		if json.Unmarshal(env["workloadInfo"], &fields) != nil || fields == nil {
			return nil, "malformed", 0
		}
	}
	out := make(map[string]json.RawMessage)
	stripped := 0
	for key, value := range fields {
		kind := workloadFieldKinds[key]
		if removal {
			kind = ""
			if key == "workloadId" || key == "originatedFrom" {
				kind = "identity"
			}
		}
		if kind == "" {
			stripped++
			continue
		}
		if !validWorkloadValue(kind, value) {
			return nil, "malformed", 0
		}
		out[key] = value
	}
	id := "id"
	if removal {
		id = "workloadId"
	}
	var identifier string
	if json.Unmarshal(out[id], &identifier) != nil || identifier == "" {
		return nil, "malformed", 0
	}
	if !removal {
		// Match the manager's required lifecycle metadata. runId/seq and
		// nullable metadata remain additive; absence must survive projection.
		for _, key := range []string{"engine", "model", "state"} {
			if json.Unmarshal(out[key], &identifier) != nil || identifier == "" {
				return nil, "malformed", 0
			}
		}
		if _, ok := out["originatedFrom"]; !ok {
			return nil, "malformed", 0
		}
		if value, ok := out["runId"]; ok && bytes.Equal(value, []byte(`""`)) {
			return nil, "malformed", 0
		}
	}
	raw, _ := json.Marshal(out) // RawMessages came from validated JSON.
	if !removal {
		stripped += len(env) - 1
		raw, _ = json.Marshal(map[string]json.RawMessage{"workloadInfo": raw})
	}
	if stripped > 0 {
		return raw, "stripped", stripped
	}
	return raw, "accepted", 0
}

var workloadFieldKinds = map[string]string{
	"id": "identity", "model": "string", "engine": "identity",
	"runId": "identity", "state": "string", "originatedFrom": "identity",
	"scheduledOn": "node-reference", "createdAt": "integer", "seq": "integer",
	"startedAt": "nullable-integer", "completedAt": "nullable-integer",
	"error": "nullable-string", "requesterId": "nullable-string", "requestedModel": "string",
}

// Manual destination IDs have no producer-side length bound. Limit their wire
// projection to 256 UTF-8 bytes; origin remains the broker's stable host UUID.
const workloadNodeReferenceMaxBytes = 256

func validWorkloadValue(kind string, raw json.RawMessage) bool {
	if bytes.Equal(raw, []byte("null")) {
		return kind == "nullable-string" || kind == "nullable-integer"
	}
	if kind == "integer" || kind == "nullable-integer" {
		var n int64
		return json.Unmarshal(raw, &n) == nil && n >= 0
	}
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return false
	}
	if kind == "node-reference" {
		if s == "" || len(s) > workloadNodeReferenceMaxBytes || !utf8.Valid(raw) {
			return false
		}
		for _, r := range s {
			if unicode.IsControl(r) || unicode.Is(unicode.Cf, r) {
				return false
			}
		}
		return true
	}
	// Empty origin is legitimate before node stamping; placement is optional.
	return kind != "identity" || s == "" || nodepolicy.ValidateRoutingIdentity(s, s) == nil
}

// Token walking detects duplicates even inside discarded unknown objects.
func walkWorkloadJSON(d *json.Decoder) bool {
	token, err := d.Token()
	if err != nil {
		return false
	}
	delim, container := token.(json.Delim)
	if !container {
		return true
	}
	if delim != '{' && delim != '[' {
		return false
	}
	seen := make(map[string]bool)
	for d.More() {
		if delim == '{' {
			key, err := d.Token()
			if err != nil {
				return false
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return false
			}
			seen[name] = true
		}
		if !walkWorkloadJSON(d) {
			return false
		}
	}
	end, err := d.Token()
	return err == nil && ((delim == '{' && end == json.Delim('}')) || (delim == '[' && end == json.Delim(']')))
}

func projectedWorkloadInfo(info json.RawMessage) json.RawMessage {
	params, err := json.Marshal(map[string]json.RawMessage{"workloadInfo": info})
	if err != nil {
		return nil
	}
	projected, category, count := projectWorkloadFrame("workloads:upsert", params)
	slog.Debug("projected workload info", "category", category, "count", count)
	if projected == nil {
		return nil
	}
	var env struct {
		Info json.RawMessage `json:"workloadInfo"`
	}
	_ = json.Unmarshal(projected, &env)
	return env.Info
}

func projectWorkloadSnapshot(infos []json.RawMessage) []json.RawMessage {
	out := make([]json.RawMessage, 0, len(infos))
	for _, info := range infos {
		if projected := projectedWorkloadInfo(info); projected != nil {
			out = append(out, projected)
		}
	}
	return out
}

// workloadReplayFrame is a lifecycle notification replayed to a (re)started
// workload-manager to rehydrate its re-sync set.
type workloadReplayFrame struct {
	method string
	params json.RawMessage
}

// workloadTerminalReplayWindowMs bounds how recently a terminal workload must
// have finished to be replayed to a (re)started workload-manager. It must be
// >= the manager's terminalRetention (2 × resyncInterval = 60s) so a restart
// mid-window doesn't drop the promised "re-assert a terminal a couple of times"
// guarantee; the manager re-arms its own expiry on replay, so an exact match is
// enough. Duplicated (not imported) because it's a private manager constant.
const workloadTerminalReplayWindowMs int64 = 60_000

// stateReplayMethod maps a stored workload state to the lifecycle method a
// replay frame should carry, so the manager re-tracks it with the right
// terminal/active disposition. Empty for an unknown state (skip).
func stateReplayMethod(state string) string {
	switch state {
	case "queued":
		return "workload:submitted"
	case "running":
		return "workload:started"
	case "completed":
		return "workload:completed"
	// "cancelled" rides workload:errored, the same frame the proxy emits it on:
	// the manager does not validate method against state, and consumers read
	// the state out of the payload. Every state the store can hold has to
	// appear here, because an omitted one is silently left out of the replay
	// and drops off the manager's re-sync set across a supervised restart.
	case "failed", "cancelled":
		return "workload:errored"
	default:
		return ""
	}
}

// activeLocalReplayFrames builds the workload:* frames for this node's active
// AND recently-terminal local-origin workloads. Only local-origin records are
// replayed — the origin is the single writer, so this node must never re-assert
// a peer's workloads. Recent terminals are included so the manager's two-
// heartbeat terminal re-sync window survives a supervised restart.
func (b *Broker) activeLocalReplayFrames() []workloadReplayFrame {
	var frames []workloadReplayFrame
	for _, r := range b.workloads.ReplayForNode(b.nodeID, workloadTerminalReplayWindowMs) {
		method := stateReplayMethod(r.State)
		if method == "" {
			continue
		}
		params, err := json.Marshal(map[string]json.RawMessage{"workloadInfo": r.Info})
		if err != nil {
			continue
		}
		projected, category, count := projectWorkloadFrame(method, params)
		slog.Debug("projected workload replay", "category", category, "count", count)
		if projected != nil {
			frames = append(frames, workloadReplayFrame{method: method, params: projected})
		}
	}
	return frames
}

// rehydrateWorkloadManager replays this node's active local-origin workloads to
// wm's stdin so its re-sync set survives a supervised restart. Frames buffer in
// the pipe until the manager's read loop consumes them, so this works without
// waiting on the manager's readiness handshake.
func (b *Broker) rehydrateWorkloadManager(wm *workloadManagerProcess) {
	b.replayPerformanceToManager(wm)
	frames := b.activeLocalReplayFrames()
	for _, f := range frames {
		if err := wm.Forward(f.method, f.params); err != nil {
			slog.Warn("failed to replay active workload to workload-manager", "err", err)
			return
		}
	}
	if len(frames) > 0 {
		slog.Info("replayed active local workloads to workload-manager", "count", len(frames))
	}
}

// replayActiveWorkloadsToScheduler seeds a new scheduler through its existing
// workloads:upsert input. Callers hold workloadEmitMu and schedulerFeedMu so the
// replay is an ordered prefix of subsequent live fanout.
func (b *Broker) replayActiveWorkloadsToScheduler(w *rpcWorker) int {
	replayed := 0
	for _, info := range projectWorkloadSnapshot(b.workloads.ActiveSnapshot()) {
		params := map[string]json.RawMessage{"workloadInfo": info}
		if err := w.Notify("workloads:upsert", params); err != nil {
			slog.Warn("replay active workload to scheduler failed", "err", err)
			break
		}
		replayed++
	}
	return replayed
}

// workloadHistoryFlushJoinTimeout bounds how long Serve waits for the history
// flusher's shutdown flush before giving up, so a hung disk can't wedge exit.
const workloadHistoryFlushJoinTimeout = 5 * time.Second

// runWorkloadHistoryFlusher starts the workload store's coalescing flusher and
// returns a stop func that cancels it and waits (bounded) for its final shutdown
// flush to complete. The returned stop func is the SOLE terminator, so callers
// must always invoke it (Serve defers it).
//
// The flusher runs on a context DETACHED from ctx (context.WithoutCancel): a
// shutdown signal (SIGINT/SIGTERM or JSON-RPC shutdown) cancels ctx before
// Serve's deferred worker teardown runs, and the proxies can still emit a
// terminal workload:errored while tearing down. If the flusher's context were a
// child of ctx, Store.Run would final-flush and exit on that signal — before
// those teardown terminals reached the store — and the later stop() would only
// join an already-finished goroutine, leaving them unpersisted. Detaching keeps
// the flusher alive until stop(), which Serve defers AFTER the worker-teardown
// defers (LIFO), so producers are quiescent before the final flush + join.
// b.workloads must already have persistence configured (WithPersistence).
func (b *Broker) runWorkloadHistoryFlusher(ctx context.Context) func() {
	fctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	go func() {
		defer close(done)
		b.workloads.Run(fctx, 0, 0)
	}()
	return func() {
		cancel()
		select {
		case <-done:
		case <-time.After(workloadHistoryFlushJoinTimeout):
			slog.Warn("workload history flusher did not finish its shutdown flush in time")
		}
	}
}

// proxyWorkloadMethods is the set of workload notifications the broker
// recognizes coming off the proxy. The proxy only produces the lifecycle
// events today; workloads:remove is included so a future producer's
// removals route the same way rather than leaking onto the proxy:<event>
// client stream.
var proxyWorkloadMethods = map[string]bool{
	"workload:submitted": true,
	"workload:started":   true,
	"workload:completed": true,
	"workload:errored":   true,
	"workloads:remove":   true,
}

// routeProxyWorkload stamps the local node id onto a local-origin workload
// event, forwards it to the workload-manager's stdin for cluster broadcast,
// and echoes it to subscribed clients so a client's view includes local
// workloads alongside the peer-origin ones the manager relays. The proxy
// keeps emitting regardless of whether a manager is present, so the echo
// still happens with no manager supervised (the broadcast simply doesn't).
func (b *Broker) routeProxyWorkload(method string, params json.RawMessage) {
	// Unscoped relay/test entry: real child workloads are intercepted with
	// their source in forwardProxyNotificationFrom before facade dispatch.
	// Keep projection/merge semantics here, but only a process can fence a run.
	b.routeProxyWorkloadFrom(nil, method, params)
}

type proxyWorkloadEvent struct {
	method string
	params json.RawMessage
}

func (b *Broker) routeProxyWorkloadFrom(source *proxyProcess, method string, params json.RawMessage) {
	params, category, count := projectWorkloadFrame(method, params)
	slog.Debug("projected proxy workload", "category", category, "count", count)
	if params == nil {
		return
	}
	stamped := stampNodeID(params, b.nodeID)
	if b.beforeProxyWorkloadApply != nil {
		b.beforeProxyWorkloadApply()
	}
	b.workloadEmitMu.Lock()
	defer b.workloadEmitMu.Unlock()
	// Validation and store apply are indivisible with handle withdrawal. The
	// initial source check cannot protect a callback stalled behind this owner.
	// Neither workersMu nor proxySupMu is held across the output below.
	b.workersMu.Lock()
	if source != nil {
		if b.proxyProc != source {
			b.workersMu.Unlock()
			return
		}
		// Done does not withdraw this handle or join admitted callbacks. Keep
		// a delivered terminal tail until actual withdrawal/capture; the exact
		// source comparison above rejects it after a replacement is published.
	}
	var frames []proxyWorkloadEvent
	var env struct {
		Info json.RawMessage `json:"workloadInfo"`
	}
	if source != nil && json.Unmarshal(stamped, &env) == nil {
		in, ok := workloadstore.ParseIncoming(env.Info)
		if ok && in.Origin == b.nodeID && in.RunID != "" && in.RunID != b.lastLocalRunID {
			// The current handle is the sole proxy process, and deliberate
			// replacement joins its predecessor. Every other local run has
			// departed. Legacy frames without a run leave this fence unchanged.
			b.lastLocalRunID = in.RunID
			pending := b.departedRunCandidates(in.RunID, "proxy run replaced")
			frames = b.applyDepartedRunWorkloadsLocked(pending)
		}
	}
	if b.applyProxyWorkloadLocked(method, stamped) {
		frames = append(frames, proxyWorkloadEvent{method: method, params: stamped})
	}
	b.workersMu.Unlock()
	for _, frame := range frames {
		b.publishProxyWorkloadLocked(frame.method, frame.params)
	}
}

// applyProxyWorkloadLocked and publishProxyWorkloadLocked are the same
// authoritative apply/fan/forward path for proxy input and broker retirement.
// Synthetic retirement never re-enters the process/run fence. Caller holds
// workloadEmitMu; publication is only for accepted frames, after store apply.
func (b *Broker) applyProxyWorkloadLocked(method string, params json.RawMessage) bool {
	if method == "workloads:remove" {
		return b.applyWorkloadEvent(method, params, false, noSightingGuard)
	}
	return b.applyWorkloadEvent("workloads:upsert", params, false, noSightingGuard)
}

func (b *Broker) publishProxyWorkloadLocked(method string, params json.RawMessage) {
	event := "workloads:upsert"
	if method == "workloads:remove" {
		event = method
	}
	b.publishWorkloadEventLocked(event, params)
	if wm := b.getWorkloadMgr(); wm != nil {
		if err := wm.Forward(method, params); err != nil {
			slog.Warn("failed to forward workload event to workload-manager", "method", method, "err", err)
		}
	} else {
		slog.Debug("no workload-manager supervised; workload event not broadcast", "method", method)
	}
}

// stampNodeID fills in the origin node id on a workload notification when
// the producer left it empty, the way the proxy does (it has no canonical
// node id). Lifecycle events carry the id inside params.workloadInfo;
// removals carry it at the top level. The params are manipulated as generic
// JSON so the broker doesn't duplicate (and risk drifting from) the
// Workload schema, and an already-populated id is left untouched. On any
// shape it doesn't recognize it forwards the params unchanged.
func stampNodeID(params json.RawMessage, nodeID string) json.RawMessage {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(params, &envelope); err != nil {
		return params
	}
	if wiRaw, ok := envelope["workloadInfo"]; ok {
		var wi map[string]json.RawMessage
		if err := json.Unmarshal(wiRaw, &wi); err != nil {
			return params
		}
		// The Workload's origin lives under originatedFrom (spec 6); the
		// removal disambiguator (else branch) carries the same originatedFrom
		// field at the top level.
		if !setNodeIDIfEmpty(wi, "originatedFrom", nodeID) {
			return params
		}
		newWI, err := json.Marshal(wi)
		if err != nil {
			return params
		}
		envelope["workloadInfo"] = newWI
	} else if !setNodeIDIfEmpty(envelope, "originatedFrom", nodeID) {
		return params
	}
	out, err := json.Marshal(envelope)
	if err != nil {
		return params
	}
	return out
}

// setNodeIDIfEmpty sets m[key] to nodeID when it is absent or an empty
// string, returning whether it modified the map. Both the Workload origin
// (nested under workloadInfo) and the removal disambiguator (top level) use
// the "originatedFrom" key.
func setNodeIDIfEmpty(m map[string]json.RawMessage, key, nodeID string) bool {
	if cur, ok := m[key]; ok {
		var s string
		if json.Unmarshal(cur, &s) == nil && s != "" {
			return false
		}
	}
	raw, err := json.Marshal(nodeID)
	if err != nil {
		return false
	}
	m[key] = raw
	return true
}

// forwardWorkloadManagerNotification is the hook startWorkloadManager
// invokes on the manager's reader goroutine for every notification it emits
// on stdout. The manager translates peer-origin lifecycle events into
// workloads:upsert and peer-origin removals into workloads:remove (spec
// 7.0); those are relayed to subscribed clients verbatim. Its startup
// "ready" (and anything else) is internal and dropped.
func (b *Broker) forwardWorkloadManagerNotification(method string, params json.RawMessage) {
	switch method {
	case methodPerformanceReceived, methodPerformanceStatus:
		b.consumePerformanceNotification(method, params)
	case "workloads:upsert", "workloads:remove":
		b.emitWorkloadEvent(method, params)
	case methodWorkloadsCancel:
		b.handlePeerWorkloadCancel(params)
	default:
		slog.Debug("ignoring workload-manager notification", "method", method)
	}
}

// emitWorkloadEvent applies a workloads:* notification to the authoritative
// store and, only when it is a real forward change, fans it to the scheduler
// and re-emits it to a subscribed client. Both the inbound relay (peer-origin
// events off the manager's stdout) and the local echo (local-origin proxy
// workloads) funnel through here. Dropping stale/regressive upserts at the
// store is what stops an out-of-order "running" (arriving after a workload's
// terminal event) from resurrecting it on the client stream and in the
// scheduler's view. The stream is off by default, so an unsubscribed client
// still sees nothing — but the store is always kept current.
//
// The whole apply→fan→notify sequence is serialized by workloadEmitMu so the
// client-visible order matches the applied order even when the node-loss sweep
// (a separate goroutine) races a relayed event for the same workload.
func (b *Broker) emitWorkloadEvent(method string, params json.RawMessage) {
	b.emitWorkloadEventProvenance(method, params, false, noSightingGuard)
}

// emitWorkloadEventInferred is emitWorkloadEvent for a locally-inferred
// transition — the node-loss sweep guessing a departed node's jobs failed. It
// records the state as inferred so a later authoritative event from the origin
// (e.g. the node was only briefly unreachable) reconciles it away.
func (b *Broker) emitWorkloadEventInferred(method string, params json.RawMessage) {
	b.emitWorkloadEventProvenance(method, params, true, noSightingGuard)
}

// emitWorkloadEventStale is emitWorkloadEventInferred for the staleness sweep,
// which selects its candidates before it applies them. seenAt is the sighting the
// decision was based on: if the origin has spoken about this workload since, the
// guess is already known to be wrong and is dropped rather than briefly showing
// running work as failed.
func (b *Broker) emitWorkloadEventStale(method string, params json.RawMessage, seenAt time.Time) {
	b.emitWorkloadEventProvenance(method, params, true, seenAt)
}

// noSightingGuard means "apply unconditionally" — every caller except the
// staleness sweep. The zero time is safe as the sentinel because a stored record
// always carries a real sighting.
var noSightingGuard time.Time

func (b *Broker) emitWorkloadEventProvenance(method string, params json.RawMessage, inferred bool, unchangedSince time.Time) {
	params, category, count := projectWorkloadFrame(method, params)
	slog.Debug("projected workload event", "category", category, "count", count)
	if params == nil {
		return
	}
	b.workloadEmitMu.Lock()
	defer b.workloadEmitMu.Unlock()

	if !b.applyWorkloadEvent(method, params, inferred, unchangedSince) {
		return
	}
	b.publishWorkloadEventLocked(method, params)
}

// publishWorkloadEventLocked fans an already accepted store transition.
// Caller holds workloadEmitMu so observed order matches the applied order.
func (b *Broker) publishWorkloadEventLocked(method string, params json.RawMessage) {
	// The scheduler is fanned every real transition (independent of the
	// external client's opt-in) so it can keep its own catalog.
	b.fanWorkloadToScheduler(method, params)

	b.workloadsMu.Lock()
	subscribed := b.workloadsSubscribed
	b.workloadsMu.Unlock()
	if !subscribed {
		return
	}
	if err := b.codec.Notify(method, params); err != nil {
		slog.Warn("emit workloads event failed", "method", method, "err", err)
	}
}

// retireWorkloadHistory enforces the store's history caps and announces each
// retired workload as a workloads:remove, so a subscribed client drops it from
// its own mirror instead of keeping every finished job for the life of the
// session. It rides the stale-workload sweep's ticker rather than the history
// flusher, because the store is bounded whether or not persistence is on.
//
// Retirement is this node's own housekeeping: each broker applies the same caps
// to its own store, so the removal is not forwarded to the workload-manager for
// cluster broadcast.
func (b *Broker) retireWorkloadHistory() {
	b.workloadEmitMu.Lock()
	defer b.workloadEmitMu.Unlock()

	retired := b.workloads.PruneHistory()
	if len(retired) == 0 {
		return
	}
	slog.Debug("retired workload history past the retention caps", "count", len(retired))

	b.workloadsMu.Lock()
	subscribed := b.workloadsSubscribed
	b.workloadsMu.Unlock()
	for _, r := range retired {
		raw, err := json.Marshal(workloadRemoveParams{WorkloadID: r.ID, OriginatedFrom: r.Origin})
		if err != nil {
			continue
		}
		// Persisted identities may predate the wire validation. Pruning has
		// already removed them locally; validate before any outgoing removal.
		params, category, _ := projectWorkloadFrame("workloads:remove", raw)
		if params == nil {
			slog.Debug("dropped workload history removal", "category", category, "count", 1)
			continue
		}
		b.fanWorkloadToScheduler("workloads:remove", params)
		if !subscribed {
			continue
		}
		if err := b.codec.Notify("workloads:remove", params); err != nil {
			slog.Warn("emit workloads event failed", "method", "workloads:remove", "err", err)
		}
	}
}

// workloadRemoveParams is the workloads:remove payload: the (originatedFrom,
// workloadId) pair a client keys its mirror by.
type workloadRemoveParams struct {
	WorkloadID     string `json:"workloadId"`
	OriginatedFrom string `json:"originatedFrom"`
}

// fanWorkloadToScheduler forwards a workloads:upsert/remove notification to the
// scheduler child verbatim. A no-op when no scheduler is running.
func (b *Broker) fanWorkloadToScheduler(method string, params json.RawMessage) {
	b.schedulerFeedMu.Lock()
	defer b.schedulerFeedMu.Unlock()

	sc := b.getScheduler()
	if sc == nil {
		return
	}
	if err := sc.Notify(method, params); err != nil {
		slog.Warn("fan workload event to scheduler failed", "method", method, "err", err)
	}
}

// workloadsInitialResult is the workloads:get-initial response: the store's
// projected snapshot as a list of workloadInfo objects (the same shape
// carried inside each workloads:upsert), so a client can key them however it
// likes.
type workloadsInitialResult struct {
	Workloads []json.RawMessage `json:"workloads"`
}

// applyWorkloadEvent folds a workloads:* notification into the authoritative
// store and reports whether it was a real change worth propagating. A
// workloads:upsert is merged by the store's monotonic rule (stale/regressive
// or duplicate events return false, so they're neither fanned nor forwarded);
// a workloads:remove drops the entry and always propagates. Malformed or
// id-less params are ignored (false). Callers hold workloadEmitMu.
func (b *Broker) applyWorkloadEvent(method string, params json.RawMessage, inferred bool, unchangedSince time.Time) bool {
	switch method {
	case "workloads:upsert":
		var env struct {
			WorkloadInfo json.RawMessage `json:"workloadInfo"`
		}
		if err := json.Unmarshal(params, &env); err != nil || len(env.WorkloadInfo) == 0 {
			return false
		}
		in, ok := workloadstore.ParseIncoming(env.WorkloadInfo)
		if !ok {
			return false
		}
		if inferred {
			if !unchangedSince.IsZero() {
				return b.workloads.ApplyInferredUnchangedSince(in, unchangedSince)
			}
			return b.workloads.ApplyInferred(in)
		}
		return b.workloads.Apply(in)
	case "workloads:remove":
		var rm workloadRemoveParams
		if err := json.Unmarshal(params, &rm); err != nil || rm.WorkloadID == "" {
			return false
		}
		b.workloads.Remove(rm.OriginatedFrom, rm.WorkloadID)
		return true
	default:
		return false
	}
}

// failWorkloadsForNode marks every non-terminal workload pinned to a node that
// just dropped out of discovery as failed. A node leaves the directory only
// after the scanner's miss-threshold + TCP-probe liveness guard has given up on
// it — a full minute of unbroken silence, a failed probe on every scan of the
// last three quarters of it, and no inference traffic from the node in that time
// either, not a transient mDNS miss — so this is a strong
// "the node is really gone" signal. That node's own proxy can no
// longer emit a terminal workload:completed/errored (its process or host is
// gone), so the broker synthesizes a workloads:upsert transition to "failed"
// for each affected workload — clearing the stale "in progress" line while
// keeping the entry in clients' history (we upsert, never remove).
//
// A workload is affected if the departed node is either its origin
// (originatedFrom — the node that submitted it) or its executor (scheduledOn —
// the node that was running it). nodeUUID is the stable per-host UUID; the
// broker stamps that same value as originatedFrom and the proxy records it as
// scheduledOn, so the three correlate directly regardless of a
// hostname change. nodeName is the display hostname, used only in the error
// message. The caller fires this only after the node's final directory claim is
// gone (a surviving manual claim means the node is still present).
//
// The transition is applied as INFERRED (emitWorkloadEventInferred): a guess
// this node made from a discovery miss, not the origin's authoritative truth.
// That provenance is what makes this safe against a wrong guess — the sweep is
// deliberately local (does NOT re-broadcast to peers), and it is reconciled by
// the origin's authoritative re-sync (the workload-manager heartbeat re-asserts
// each node's own active + recently-terminal workloads). So if node B evicts
// node A over a transient partition and infers A's jobs failed, A's next re-sync
// carries its authoritative state and the store's provenance-aware merge
// overrides B's inferred guess — un-sticking it. If A is truly gone, nothing
// overrides the guess and it stands. The origin is the single writer and the
// convergence authority; this sweep only supplies a fast, provisional local
// answer in the gap before the next re-sync. It relies on the scanner's
// eviction being a reasonable "offline" signal — a false positive is corrected,
// not permanent, but until the correcting re-sync arrives (≤ a couple of
// heartbeats) a briefly-partitioned node's jobs show as failed here.
func (b *Broker) failWorkloadsForNode(nodeUUID, nodeName string) {
	if nodeUUID == "" {
		return
	}
	now := time.Now()
	// Match workloads by the stable HostUUID (what the proxy stamps as
	// originatedFrom / scheduledOn); the hostname is display only, so
	// use it in the error message but fall back to the UUID if it's absent.
	display := nodeName
	if display == "" {
		display = nodeUUID
	}
	errMsg := fmt.Sprintf("node %s went offline", display)

	// Snapshot the still-live workloads pinned to this node, then synthesize a
	// terminal transition for each, applied as inferred. An inferred failed
	// marks the running record terminal locally (so it leaves the next
	// ActiveForNode sweep), but yields to the origin's authoritative re-sync if
	// the node was only briefly unreachable.
	var failed []json.RawMessage
	for _, wl := range b.workloads.ActiveForNode(nodeUUID) {
		if params := markWorkloadFailed(wl.Info, now, errMsg); params != nil {
			failed = append(failed, params)
		}
	}

	if len(failed) == 0 {
		return
	}
	slog.Info("marking in-flight workloads failed for departed node", "node", display, "uuid", nodeUUID, "count", len(failed))
	for _, params := range failed {
		// Inferred: a guess this node made from a discovery miss. If the node
		// was only briefly unreachable, its authoritative re-sync reconciles
		// these back to their real state; if it's truly gone, the guess stands.
		b.emitWorkloadEventInferred("workloads:upsert", params)
	}
}

// Workload staleness sweep tuning.
//
// The origin's workload-manager re-asserts every one of its own non-terminal
// workloads to every peer on a fixed heartbeat (30s at the time of writing;
// duplicated here as a number rather than imported because it is a private
// constant of that binary). So for a workload we believe is running on another
// node, continued silence across many consecutive heartbeats means the origin no
// longer considers it active — it finished and we lost every copy of the
// terminal, or the origin stopped running.
//
// The window is deliberately generous: ten missed re-assertions rather than two,
// because the cost of waiting is a stale line for a few minutes while the cost of
// being hasty is briefly mislabelling a genuinely running remote job. A wrong
// guess is self-correcting either way — the sweep applies an INFERRED terminal,
// which the origin's next authoritative event overrides.
const (
	workloadOriginSilenceTimeout = 5 * time.Minute
	workloadStaleSweepInterval   = 60 * time.Second
)

// runStaleWorkloadSweep periodically retires remote-origin workloads whose origin
// has gone silent about them. It is the backstop that bounds how long a lost
// terminal event can display as in-flight: every other path in the workload
// pipeline is push-only, so without this a single dropped completed/errored
// leaves the line running until the process restarts.
func (b *Broker) runStaleWorkloadSweep(ctx context.Context) {
	ticker := time.NewTicker(workloadStaleSweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			b.failStaleForeignWorkloads(workloadOriginSilenceTimeout)
			b.retireWorkloadHistory()
		}
	}
}

// failStaleForeignWorkloads marks every remote-origin non-terminal workload the
// origin has not re-asserted within workloadOriginSilenceTimeout as failed.
//
// This is the same synthesized transition the node-loss sweep applies, for the
// same reason and with the same provenance, but triggered by a different signal:
// failWorkloadsForNode fires when a node leaves discovery, whereas this fires
// when a node is still present yet has stopped mentioning a workload we think it
// is running. Neither can substitute for the other — a peer whose delivery to us
// is failing stays happily in discovery.
// staleAfter is a parameter rather than the constant so a test can drive the
// boundary without waiting on the wall clock.
func (b *Broker) failStaleForeignWorkloads(staleAfter time.Duration) {
	stale := b.workloads.StaleForeign(b.nodeID, staleAfter)
	if len(stale) == 0 {
		return
	}
	now := time.Now()
	names := b.nodeDisplayNames()
	for _, wl := range stale {
		display := wl.Origin
		if name, ok := names[wl.Origin]; ok && name != "" {
			display = name
		}
		params := markWorkloadFailed(wl.Info, now, fmt.Sprintf("node %s stopped reporting this workload", display))
		if params == nil {
			continue
		}
		slog.Info("retiring workload the origin stopped reporting",
			"id", wl.ID, "origin", wl.Origin, "state", wl.State,
			"silentFor", now.Sub(wl.LastSeen))
		// Inferred, so the origin's next authoritative event wins if it turns out
		// to still be running and merely unable to reach us — and guarded on the
		// sighting this decision came from, so an origin that spoke up between the
		// selection above and this apply is not overwritten by a guess that is
		// already known to be wrong.
		b.emitWorkloadEventStale("workloads:upsert", params, wl.LastSeen)
	}
}

// nodeDisplayNames maps node UUID to advertised hostname for user-facing error
// text. Built once per sweep rather than per record: Snapshot projects and sorts
// the whole directory on every call, and a sweep can touch many records at once.
// A UUID missing from the map (a peer that has since left) falls back to itself.
func (b *Broker) nodeDisplayNames() map[string]string {
	if b.store == nil {
		return nil
	}
	nodes := b.store.Snapshot()
	names := make(map[string]string, len(nodes))
	for _, n := range nodes {
		if n.HostUUID != "" && n.Name != "" {
			names[n.HostUUID] = n.Name
		}
	}
	return names
}

// markWorkloadFailed rewrites a stored workloadInfo into a terminal "failed"
// transition and wraps it in the workloads:upsert params envelope. It edits
// the object as generic JSON — setting state/completedAt/error and leaving
// every other field (id, model, engine, originatedFrom, scheduledOn,
// createdAt, startedAt, requesterId) untouched — so the broker never
// duplicates the Workload schema. Returns nil if the stored info can't be
// parsed or re-marshaled.
//
// It takes a time.Time and converts to epoch ms itself: this is the only place
// that serializes the Workload wire format, so it is the only place that should
// know the format's representation. Callers stay in time.Time.
func markWorkloadFailed(info json.RawMessage, completedAt time.Time, errMsg string) json.RawMessage {
	return markWorkloadTerminal(info, "failed", completedAt, errMsg, -1)
}

// markWorkloadTerminal preserves identity/placement and sets an authoritative
// sequence when nonnegative. The inferred sweeps pass -1 to leave seq untouched,
// including its absence in legacy records.
func markWorkloadTerminal(info json.RawMessage, state string, completedAt time.Time, errMsg string, seq int64) json.RawMessage {
	var wi map[string]json.RawMessage
	if err := json.Unmarshal(info, &wi); err != nil {
		return nil
	}
	set := func(key string, v any) {
		if raw, err := json.Marshal(v); err == nil {
			wi[key] = raw
		}
	}
	set("state", state)
	set("completedAt", completedAt.UnixMilli())
	set("error", errMsg)
	if seq >= 0 {
		set("seq", seq)
	}
	newWI, err := json.Marshal(wi)
	if err != nil {
		return nil
	}
	env, err := json.Marshal(map[string]json.RawMessage{"workloadInfo": newWI})
	if err != nil {
		return nil
	}
	return env
}
