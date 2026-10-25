// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// pull.go is the streaming model-pull path. It runs the same manifest-declared
// pull_model action as engine:action, but instead of only returning the final
// blob it tees the engine's progress to both consumers via emitPullProgress:
// the local engine:pull-progress notification (this node's UI) and the progress
// hub (so an ec streaming handler can relay live download progress to a remote
// initiator). engine:action with action "pull_model" is routed here for exactly
// this reason — a local pull streams progress just like a remote one.
//
// Ollama's /api/pull streams newline-delimited JSON status objects
// ({"status":...,"total":N,"completed":M}); each line maps to a progress event,
// coalesced so only changes in stage/percent are emitted (a single layer streams
// many byte-progress lines at the same rendered percent). llama.cpp instead
// acknowledges POST /models immediately and reports completion on /models/sse;
// its manifest opts into that named adapter. CLI-driven pulls (LM Studio's
// `lms get`) emit one "pulling" marker and return the final result.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
	"nvpair-shared/nodepolicy"
)

const (
	// pullModelAction is the manifest action name every engine uses for model pulls.
	pullModelAction = "pull_model"
	// pullProgressProtocolLlamaCPPModelsSSE names the pinned llama.cpp router
	// protocol: subscribe first, POST /models, then await a matching terminal SSE.
	pullProgressProtocolLlamaCPPModelsSSE = "llamacpp-models-sse"
)

var errPullProgressTimeout = errors.New("model download made no progress")

// The ec pull decoder must retain the raw model boundary too: decoding a
// malformed string first would permanently replace invalid UTF-8 bytes.
func (p *pullRequest) UnmarshalJSON(data []byte) error {
	if err := validateModelRequest(data, false); err != nil {
		return err
	}
	type request pullRequest
	var next request
	if err := json.Unmarshal(data, &next); err != nil {
		return err
	}
	if err := (&Manifest{}).validateActionModelParams(pullModelAction, next.Params, false); err != nil {
		return err
	}
	*p = pullRequest(next)
	return nil
}

type pullProgressWatchdog struct {
	cancel      context.CancelCauseFunc
	timer       *time.Timer
	timeout     time.Duration
	completedBy map[string]int64
}

func newPullProgressWatchdog(parent context.Context, timeout time.Duration) (context.Context, *pullProgressWatchdog) {
	ctx, cancel := context.WithCancelCause(parent)
	timeoutErr := fmt.Errorf("%w for %s", errPullProgressTimeout, timeout)
	watchdog := &pullProgressWatchdog{
		cancel:      cancel,
		timeout:     timeout,
		completedBy: make(map[string]int64),
	}
	watchdog.timer = time.AfterFunc(timeout, func() {
		cancel(timeoutErr)
	})
	return ctx, watchdog
}

func (w *pullProgressWatchdog) stop() {
	w.timer.Stop()
	w.cancel(nil)
}

func (w *pullProgressWatchdog) recordProgress(key string, completed int64) {
	previous, seen := w.completedBy[key]
	if completed <= 0 || (seen && completed <= previous) {
		return
	}
	w.completedBy[key] = completed
	w.timer.Reset(w.timeout)
}

func (w *pullProgressWatchdog) resolveError(ctx context.Context, fallback error) error {
	cause := context.Cause(ctx)
	if errors.Is(cause, errPullProgressTimeout) {
		return cause
	}
	return fallback
}

// modelFromParams resolves conventional pull aliases, preferring Ollama's
// "name" body key then the generic {model} placeholder. Explicit descriptor
// bindings take precedence in pullProgressModel; params are never rewritten.
func modelFromParams(params json.RawMessage) string {
	if len(params) == 0 {
		return ""
	}
	var p struct {
		Name  string `json:"name"`
		Model string `json:"model"`
	}
	_ = json.Unmarshal(params, &p)
	if p.Name != "" {
		return p.Name
	}
	return p.Model
}

func (m *Manifest) pullProgressModel(params json.RawMessage) (string, error) {
	if err := m.validateActionModelParams(pullModelAction, params, true); err != nil {
		return "", err
	}
	var model string
	if binding, declared := m.Operations[engines.OperationPull]; declared {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(params, &fields); err != nil {
			return "", err
		}
		if err := json.Unmarshal(fields[binding.modelParam()], &model); err != nil {
			return "", fmt.Errorf("model must be a string")
		}
	} else {
		model = modelFromParams(params)
	}
	if err := nodepolicy.ValidateModelIdentifier(model); err != nil {
		return "", err
	}
	return model, nil
}

// PullModelStream runs the engine's pull_model action, publishing progress to
// the hub as it advances, and returns the action's final JSON result. When
// params is empty they come from the engine's pull binding (modelops.go), so a
// caller can pass just a model name.
//
// A pull downloads into the engine's model store: everything above the
// download is read-only, and the download itself runs under the engine's
// content-mutation guard (model_mutation.go), so the broker withdraws that
// engine's model identity before the first byte is requested.
func (e *Executor) PullModelStream(ctx context.Context, engine, model string, params json.RawMessage) (json.RawMessage, error) {
	token, err := e.reserveModelPull(ctx, engine, model, params)
	if err != nil {
		return nil, err
	}
	return token.run(ctx)
}

type modelPullContextKey struct{}

// Includes the stopped generation for CLI pulls that do not require an engine
// process. An empty run ID alone cannot distinguish stop/start/stop cycles.
type modelPullRun struct {
	id      string
	gen     int64
	running bool
	port    int
}

func currentModelPullRun(st *engineState) modelPullRun {
	st.mu.Lock()
	defer st.mu.Unlock()
	run := modelPullRun{id: st.runID, running: st.running, port: st.port}
	if run.id == "" {
		// Health-loop replacement can advance gen without retiring a known
		// process. Use it only as the fallback identity for a stopped CLI run.
		run.gen = st.gen
	}
	return run
}

type modelPullReservation struct {
	exec              *Executor
	id, engine, model string
	params            json.RawMessage
	st                *engineState
	act               Action
	attempt           *modelDownloadAttempt
	mu                sync.Mutex
	started           bool
	terminal          *modeldownload.Snapshot
	// Execution metadata is confined to run's goroutine.
	effects     bool
	dispatchRun *modelPullRun
}

func (e *Executor) reserveModelPull(ctx context.Context, engine, model string, params json.RawMessage) (*modelPullReservation, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model != "" || len(params) == 0 {
		if err := nodepolicy.ValidateModelIdentifier(model); err != nil {
			return nil, err
		}
	}
	st, err := e.state(engine)
	if err != nil {
		return nil, err
	}
	act, ok := st.manifest.Actions[pullModelAction]
	if !ok {
		return nil, fmt.Errorf("engine %q has no action %q", engine, pullModelAction)
	}
	if len(params) == 0 || string(params) == "null" {
		if params, err = e.pullParams(engine, model); err != nil {
			return nil, err
		}
	}
	progressModel, err := st.manifest.pullProgressModel(params)
	if err != nil {
		return nil, err
	}
	if model == "" {
		model = progressModel
	}
	if model != progressModel {
		return nil, errors.New("pull model does not match action parameters")
	}
	a, err := e.downloads.reserve(engine, model, st.manifest.ModelNamingOrDefault())
	if err != nil {
		return nil, err
	}
	token := &modelPullReservation{exec: e, id: a.entry.ID, engine: engine, model: model, params: append(json.RawMessage(nil), params...), st: st, act: act, attempt: a}
	token.emitProgress("queued", 0)
	e.pokeLoaded()
	return token, nil
}

func (t *modelPullReservation) progressEvent(stage string, percent int) ProgressEvent {
	return ProgressEvent{ID: t.id, Engine: t.engine, Model: t.model, Op: "pull", Stage: stage, Percent: percent, DownloadState: t.snapshot()}
}

func (t *modelPullReservation) snapshot() *modeldownload.Snapshot {
	t.mu.Lock()
	if t.terminal != nil {
		s := t.terminal.Clone()
		t.mu.Unlock()
		return &s
	}
	t.mu.Unlock()
	s := t.exec.downloads.snapshot()
	return &s
}

func (t *modelPullReservation) emitProgress(stage string, percent int) {
	t.exec.emitPullProgress(t.progressEvent(stage, percent))
}

func (t *modelPullReservation) cancelUnstarted() {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()
		return
	}
	t.started = true
	t.mu.Unlock()
	t.finish("failed", "interrupted", "")
	t.emitProgress("error", -1)
	t.exec.pokeLoaded()
}

func (t *modelPullReservation) finish(phase, reason, persistence string) error {
	snapshot, err := t.exec.downloads.finishSnapshot(t.attempt, phase, reason, persistence)
	if err != nil {
		return err
	}
	t.mu.Lock()
	t.terminal = &snapshot
	t.mu.Unlock()
	return nil
}

type modelPullFailure struct {
	phase, reason string
	cause         error
}

func (f *modelPullFailure) Error() string { return f.cause.Error() }
func (f *modelPullFailure) Unwrap() error { return f.cause }
func pullFailure(phase, reason string, err error) error {
	return &modelPullFailure{phase: phase, reason: reason, cause: err}
}

func (t *modelPullReservation) run(ctx context.Context) (result json.RawMessage, err error) {
	t.mu.Lock()
	if t.started {
		t.mu.Unlock()
		return nil, errDownloadBusy
	}
	t.started = true
	t.mu.Unlock()
	ctx = context.WithValue(ctx, modelPullContextKey{}, t)
	phase, reason, persistence := "complete", "", ""
	defer func() {
		if err != nil {
			phase, reason = "failed", "download_failed"
			if t.effects {
				phase, reason = "unknown", "interrupted"
			}
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				reason = "interrupted"
			}
			if errors.Is(err, errDownloadBusy) {
				reason = "busy"
			}
			var failure *modelPullFailure
			if errors.As(err, &failure) {
				phase, reason = failure.phase, failure.reason
			}
		}
		if finishErr := t.finish(phase, reason, persistence); finishErr != nil {
			err = errors.Join(err, finishErr)
		}
		stage, pct := "success", 100
		if phase != "complete" {
			stage, pct = "error", -1
		}
		t.emitProgress(stage, pct)
		t.exec.pokeLoaded()
	}()
	err = t.exec.withModelMutation(ctx, t.engine, func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := t.exec.checkPullRetry(ctx, t); err != nil {
			return err
		}
		releaseLoad := func() {}
		if binding, load := t.st.manifest.operationBinding(engines.OperationLoad); load && binding.Action == pullModelAction {
			if err := t.exec.beforeModelPullEffect(ctx); err != nil {
				return err
			}
			loadCtx := context.WithValue(ctx, modelContentWriterKey{}, t.engine)
			guarded, release, err := t.exec.guardServingLoad(loadCtx, t.st, pullModelAction, t.params)
			if err != nil {
				return err
			}
			releaseLoad = release
			ctx = guarded
		}
		var pullErr error
		result, pullErr = t.exec.runPullModel(ctx, t.st, t.engine, t.model, t.act, t.params)
		if pullErr == nil {
			// Linearize native completion against the selected execution run.
			// A later restart cannot invalidate evidence already accepted here;
			// release a bound Load guard before waiting for journal publication.
			current := currentModelPullRun(t.st)
			if t.dispatchRun == nil || current != *t.dispatchRun {
				pullErr = pullFailure("unknown", "interrupted", errors.New("model download completed after its engine run was retired"))
			}
		}
		releaseLoad()
		if pullErr != nil {
			return pullErr
		}
		uncertain, saveErr := t.exec.downloads.changeMarker(t.attempt, true)
		if saveErr != nil && !uncertain {
			return pullFailure("unknown", "persistence_error", saveErr)
		}
		if uncertain {
			persistence = "uncertain"
		}
		return nil
	})
	return result, err
}

func (e *Executor) beforeModelPullEffect(ctx context.Context) error {
	t, ok := ctx.Value(modelPullContextKey{}).(*modelPullReservation)
	if !ok {
		return errors.New("pull has no content reservation")
	}
	if t.effects {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	uncertain, err := e.downloads.changeMarker(t.attempt, false)
	if err != nil {
		phase := "failed"
		if uncertain {
			phase = "unknown"
		}
		return pullFailure(phase, "persistence_error", err)
	}
	t.effects = true
	if err := e.downloads.update(t.attempt, "in_progress", nil); err != nil {
		return err
	}
	t.emitProgress("pulling", 0)
	return nil
}

func (e *Executor) emitModelPullProgress(ctx context.Context, ev ProgressEvent, measured bool) {
	t, ok := ctx.Value(modelPullContextKey{}).(*modelPullReservation)
	if !ok {
		return
	}
	before := e.downloads.snapshot()
	var percent *int
	if measured && ev.Percent >= 0 && ev.Percent <= 100 {
		percent = &ev.Percent
	}
	if e.downloads.update(t.attempt, "in_progress", percent) != nil {
		return
	}
	if !before.Equal(e.downloads.snapshot()) {
		t.emitProgress(ev.Stage, ev.Percent)
	}
}

func (e *Executor) checkPullRetry(ctx context.Context, t *modelPullReservation) error {
	e.downloads.mu.Lock()
	_, retry := e.downloads.unsafe[t.attempt.key]
	e.downloads.mu.Unlock()
	if !retry || t.act.ProgressProtocol != pullProgressProtocolLlamaCPPModelsSSE {
		return nil
	}
	t.st.mu.Lock()
	port := t.st.port
	t.st.mu.Unlock()
	status, err := e.llamaDownloadStatus(ctx, fmt.Sprintf("http://127.0.0.1:%d", port), t.model)
	if err != nil {
		return err
	}
	if status == "downloading" {
		return errDownloadBusy
	}
	return nil
}

// runPullModel runs the pull itself: the CLI action form or the engine's
// streaming HTTP download. Its caller holds the engine's content-mutation
// guard.
func (e *Executor) runPullModel(ctx context.Context, st *engineState, engine, model string, act Action, params json.RawMessage) (json.RawMessage, error) {
	// Bind completion to the same snapshot that selects native dispatch inputs.
	// CLI pulls may execute while stopped; retain that state instead of requiring
	// a running process or manufacturing an engine incarnation for the command.
	dispatchRun := currentModelPullRun(st)
	if token, ok := ctx.Value(modelPullContextKey{}).(*modelPullReservation); ok {
		token.dispatchRun = &dispatchRun
	}
	port := dispatchRun.port
	// CLI action (e.g. lms get): no structured line progress; emit a start
	// marker and return the final result via the existing runner.
	if len(act.Cmd) > 0 {
		actionCtx, cancel := context.WithTimeout(ctx, e.actionTimeout)
		defer cancel()
		if err := e.beforeModelPullEffect(actionCtx); err != nil {
			return nil, err
		}
		result, err := e.runCmdAction(actionCtx, st, act, port, params)
		if err != nil {
			return nil, pullFailure("failed", "download_failed", err)
		}
		var outcome struct {
			Error   string `json:"error"`
			Success *bool  `json:"success"`
		}
		if json.Unmarshal(result, &outcome) == nil && (outcome.Error != "" || outcome.Success != nil && !*outcome.Success) {
			return nil, pullFailure("failed", "download_failed", errors.New("engine reported download failure"))
		}
		return result, nil
	}

	// HTTP action (e.g. Ollama /api/pull): stream NDJSON progress.
	if !dispatchRun.running {
		return nil, fmt.Errorf("engine %q is not running", engine)
	}
	pullCtx, watchdog := newPullProgressWatchdog(ctx, e.pullProgressTimeout)
	defer watchdog.stop()
	if act.ProgressProtocol == pullProgressProtocolLlamaCPPModelsSSE {
		return e.pullModelLlamaCPPSSE(pullCtx, engine, model, act, port, params, watchdog)
	}
	path, err := resolvePlaceholders(act.HTTP.Path, map[string]string{"port": strconv.Itoa(port)})
	if err != nil {
		return nil, err
	}
	url := fmt.Sprintf("http://127.0.0.1:%d%s", port, path)
	req, err := http.NewRequestWithContext(pullCtx, strings.ToUpper(act.HTTP.Method), url, bytes.NewReader(params))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if err := e.beforeModelPullEffect(pullCtx); err != nil {
		return nil, err
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("pull %q: %w", model, watchdog.resolveError(pullCtx, err))
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
		return nil, fmt.Errorf("pull %q: engine returned HTTP %d: %s", model, resp.StatusCode, strings.TrimSpace(string(data)))
	}

	// Coalesce redundant frames: a chatty engine streams many byte-progress
	// lines per layer, most of which map to the same (stage, percent) pair. We
	// keep the newest raw line for the terminal result but only notify/publish
	// when the rendered stage or percent actually changes, bounding the event
	// volume a subscriber (local UI or remote relay) has to absorb. lastPct=-1
	// is a sentinel pullProgressFromLine never yields, so the first frame emits.
	last := json.RawMessage("null")
	sawProtocol := false
	lastStage, lastPct := "", -1
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20) // status lines are small; cap generously
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 || !json.Valid(line) {
			continue
		}
		last = append(json.RawMessage(nil), line...)
		progress := decodeOllamaPullLine(line)
		if progress.Error != "" {
			return nil, pullFailure("failed", "download_failed", errors.New("engine reported download failure"))
		}
		if progress.Status == "success" {
			return last, nil
		}
		if progress.Status != "" {
			sawProtocol = true
		}
		watchdog.recordProgress(progress.progressKey(), progress.Completed)
		ev := progress.event(engine)
		if ev.Stage == lastStage && ev.Percent == lastPct {
			continue
		}
		lastStage, lastPct = ev.Stage, ev.Percent
		e.emitModelPullProgress(ctx, ev, progress.Total > 0)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("pull %q: %w", model, watchdog.resolveError(pullCtx, err))
	}
	if !sawProtocol {
		return nil, pullFailure("unknown", "unsupported", errors.New("engine has no supported download terminal"))
	}
	return nil, pullFailure("unknown", "interrupted", errors.New("download stream ended before success"))
}

// pullModelLlamaCPPSSE runs llama.cpp's asynchronous router download protocol.
// The SSE response must be open before POST /models because terminal events are
// one-shot broadcasts; subscribing afterward can miss a fast completion.
func (e *Executor) pullModelLlamaCPPSSE(ctx context.Context, engine, model string, act Action, port int, params json.RawMessage, watchdog *pullProgressWatchdog) (result json.RawMessage, pullErr error) {
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("pull model is required for %s", pullProgressProtocolLlamaCPPModelsSSE)
	}
	var requested struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(params, &requested); err != nil {
		return nil, fmt.Errorf("pull %q: decode model params: %w", model, err)
	}
	if requested.Model != model {
		return nil, fmt.Errorf("pull %q: params.model must match the requested model", model)
	}
	baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	sseReq, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models/sse", nil)
	if err != nil {
		return nil, err
	}
	sseReq.Header.Set("Accept", "text/event-stream")
	sseResp, err := e.client.Do(sseReq)
	if err != nil {
		return nil, fmt.Errorf("pull %q: subscribe to model progress: %w", model, watchdog.resolveError(ctx, err))
	}
	defer sseResp.Body.Close()
	if sseResp.StatusCode < 200 || sseResp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(sseResp.Body, 64*1024))
		return nil, fmt.Errorf("pull %q: progress stream returned HTTP %d: %s", model, sseResp.StatusCode, strings.TrimSpace(string(data)))
	}

	path, err := resolvePlaceholders(act.HTTP.Path, map[string]string{"port": strconv.Itoa(port)})
	if err != nil {
		return nil, err
	}
	if cause := context.Cause(ctx); cause != nil {
		return nil, fmt.Errorf("pull %q: %w", model, cause)
	}
	// Cancelling POST /models does not cancel the router's download child. Keep
	// the bounded handshake alive so cancellation cannot discard its acceptance.
	startCtx, cancelStart := context.WithTimeout(context.WithoutCancel(ctx), e.pullStartTimeout)
	defer cancelStart()
	startReq, err := http.NewRequestWithContext(startCtx, strings.ToUpper(act.HTTP.Method), baseURL+path, bytes.NewReader(params))
	if err != nil {
		return nil, err
	}
	startReq.Header.Set("Content-Type", "application/json")
	if err := e.beforeModelPullEffect(ctx); err != nil {
		return nil, err
	}
	startResp, err := e.client.Do(startReq)
	if err != nil {
		return nil, llamaCPPUnconfirmedStartError(ctx, model, err)
	}
	startData, readErr := io.ReadAll(io.LimitReader(startResp.Body, 64*1024))
	startResp.Body.Close()
	cancelStart()
	if readErr != nil {
		return nil, llamaCPPUnconfirmedStartError(ctx, model, readErr)
	}
	if startResp.StatusCode < 200 || startResp.StatusCode >= 300 {
		return nil, fmt.Errorf("pull %q: engine returned HTTP %d: %s", model, startResp.StatusCode, strings.TrimSpace(string(startData)))
	}
	var started struct {
		Success *bool `json:"success"`
	}
	if err := json.Unmarshal(startData, &started); err != nil {
		return nil, llamaCPPUnconfirmedStartError(ctx, model, err)
	}
	if started.Success == nil {
		return nil, llamaCPPUnconfirmedStartError(ctx, model, errors.New("start response has no success flag"))
	}
	if !*started.Success {
		return nil, fmt.Errorf("pull %q: engine did not accept the download", model)
	}
	terminal := false
	defer func() {
		if terminal {
			return
		}
		if cause := context.Cause(ctx); cause != nil {
			pullErr = fmt.Errorf("pull %q: %w", model, cause)
		}
		cleanupCtx, cancelCleanup := context.WithTimeout(context.WithoutCancel(ctx), e.pullCleanupTimeout)
		defer cancelCleanup()
		if err := e.stopLlamaCPPDownload(cleanupCtx, baseURL, model); err != nil {
			pullErr = pullFailure("unknown", "cleanup_unknown", errors.Join(pullErr, fmt.Errorf("pull %q: could not confirm download stopped: %w", model, err)))
		} else {
			reason := "interrupted"
			if errors.Is(pullErr, errPullProgressTimeout) {
				reason = "progress_timeout"
			}
			pullErr = pullFailure("failed", reason, pullErr)
		}
	}()
	if cause := context.Cause(ctx); cause != nil {
		return nil, fmt.Errorf("pull %q: %w", model, cause)
	}

	lastPct := -1
	sc := bufio.NewScanner(sseResp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		var event llamaCPPModelsEvent
		if err := json.Unmarshal(bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:"))), &event); err != nil || event.Model != model {
			continue
		}
		switch event.Event {
		case "download_progress":
			event.recordProgress(watchdog)
			pct := event.percent()
			if pct != lastPct {
				lastPct = pct
				e.emitModelPullProgress(ctx, ProgressEvent{Engine: engine, Op: "pull", Stage: "downloading", Percent: pct, Message: model}, event.hasByteProgress())
			}
		case "download_finished":
			terminal = true
			return json.RawMessage(startData), nil
		case "download_failed":
			terminal = true
			return nil, pullFailure("failed", "download_failed", fmt.Errorf("pull %q: llama.cpp reported download failure", model))
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("pull %q: progress stream: %w", model, watchdog.resolveError(ctx, err))
	}
	return nil, fmt.Errorf("pull %q: progress stream ended before completion", model)
}

// A lost acknowledgement gives us no ownership of a router download: another
// caller may already be downloading this ID. Preserve the cause without blindly
// unloading somebody else's model.
func llamaCPPUnconfirmedStartError(ctx context.Context, model string, err error) error {
	return pullFailure("unknown", "acceptance_unknown", fmt.Errorf("pull %q: download acceptance and cancellation could not be confirmed: %w", model, errors.Join(context.Cause(ctx), err)))
}

// stopLlamaCPPDownload uses the same router as the start request. Inventory is
// checked first because a missed terminal SSE may mean the model is now loaded;
// unload would then interrupt inference rather than stop an active download.
func (e *Executor) stopLlamaCPPDownload(ctx context.Context, baseURL, model string) error {
	status, err := e.llamaDownloadStatus(ctx, baseURL, model)
	if err != nil {
		return err
	}
	if status != "downloading" {
		return nil
	}
	params, err := json.Marshal(map[string]string{"model": model})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/models/unload", bytes.NewReader(params))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("stop download: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("stop download: HTTP %d", resp.StatusCode)
	}
	var stopped struct {
		Success bool `json:"success"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64*1024)).Decode(&stopped); err != nil {
		return err
	}
	if !stopped.Success {
		return errors.New("engine did not confirm download stopped")
	}
	return nil
}

func (e *Executor) llamaDownloadStatus(ctx context.Context, baseURL, model string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return "", err
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("check download inventory: %w", err)
	}
	var inventory struct {
		Data *[]struct {
			ID     string `json:"id"`
			Status struct {
				Value string `json:"value"`
			} `json:"status"`
		} `json:"data"`
	}
	decodeErr := json.NewDecoder(io.LimitReader(resp.Body, 8*1024*1024)).Decode(&inventory)
	resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("check download inventory: HTTP %d", resp.StatusCode)
	}
	if decodeErr != nil {
		return "", fmt.Errorf("decode download inventory: %w", decodeErr)
	}
	if inventory.Data == nil {
		return "", errors.New("download inventory has no data array")
	}
	status := ""
	for _, entry := range *inventory.Data {
		if entry.ID != model {
			continue
		}
		if status != "" {
			return "", errors.New("ambiguous download inventory")
		}
		switch entry.Status.Value {
		case "downloading", "downloaded", "unloaded", "loading", "loaded", "sleeping":
			status = entry.Status.Value
		default:
			return "", errors.New("download inventory has unknown model status")
		}
	}
	return status, nil
}

type llamaCPPModelsEvent struct {
	Model string `json:"model"`
	Event string `json:"event"`
	Data  struct {
		Progress map[string]struct {
			Done  int64 `json:"done"`
			Total int64 `json:"total"`
		} `json:"progress"`
	} `json:"data"`
}

func (e llamaCPPModelsEvent) recordProgress(watchdog *pullProgressWatchdog) {
	for file, progress := range e.Data.Progress {
		watchdog.recordProgress(file, progress.Done)
	}
}

func (e llamaCPPModelsEvent) percent() int {
	var done, total int64
	for _, file := range e.Data.Progress {
		if file.Total <= 0 {
			continue
		}
		total += file.Total
		if file.Done > 0 {
			done += min(file.Done, file.Total)
		}
	}
	if total == 0 {
		return 0
	}
	return int(done * 100 / total)
}

func (e llamaCPPModelsEvent) hasByteProgress() bool {
	for _, file := range e.Data.Progress {
		if file.Total > 0 {
			return true
		}
	}
	return false
}

// engineDisplayName returns the manifest display name for user-facing copy.
func (e *Executor) engineDisplayName(engine string) string {
	st, err := e.state(engine)
	if err == nil && strings.TrimSpace(st.manifest.DisplayName) != "" {
		return st.manifest.DisplayName
	}
	return engine
}

// reportPullFailed records a pull failure for the errors pipeline and returns
// the user-facing message for progress frames and JSON-RPC errors.
func (e *Executor) reportPullFailed(engine, model string, err error) string {
	var refusal *modelIdentifierRefusal
	if errors.As(err, &refusal) {
		return refusal.Error()
	}
	if validationErr := nodepolicy.ValidateModelIdentifier(model); validationErr != nil {
		return validationErr.Error()
	}
	msg := formatEnginePullError(e.engineDisplayName(engine), err)
	e.reporter.report(serviceError{
		ID: pullFailedID(engine, model), Message: msg,
		Severity: "error", Action: "retry", EngineType: engine, Operation: "pull", ModelName: model,
	})
	return msg
}

type ollamaPullLine struct {
	Error     string `json:"error"`
	Status    string `json:"status"`
	Digest    string `json:"digest"`
	Total     int64  `json:"total"`
	Completed int64  `json:"completed"`
}

func decodeOllamaPullLine(line []byte) ollamaPullLine {
	var progress ollamaPullLine
	_ = json.Unmarshal(line, &progress)
	return progress
}

func (p ollamaPullLine) progressKey() string {
	if p.Digest != "" {
		return p.Digest
	}
	return p.Status
}

func (p ollamaPullLine) event(engine string) ProgressEvent {
	pct := 0
	if p.Total > 0 {
		pct = int(p.Completed * 100 / p.Total)
	}
	return ProgressEvent{Engine: engine, Op: "pull", Stage: p.Status, Percent: pct, Message: p.Status}
}

// pullProgressFromLine maps an Ollama /api/pull status line into a ProgressEvent,
// computing a percentage when the line carries total/completed byte counts.
func pullProgressFromLine(engine string, line []byte) ProgressEvent {
	return decodeOllamaPullLine(line).event(engine)
}
