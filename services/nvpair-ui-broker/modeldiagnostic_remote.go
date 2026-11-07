// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"nvpair-shared/modelconfig"
)

type diagnosticRelayRequest struct {
	ID     string          `json:"id"`
	Method string          `json:"method"`
	Caller string          `json:"caller"`
	Params json.RawMessage `json:"params"`
}

// The source pointer, not a caller-supplied string or a current worker handle,
// fences each request and cancellation across manager replacement.
type diagnosticRelayKey struct {
	source *engineManagerSource
	id     string
}

type diagnosticPeerAuthority struct {
	source *engineManagerSource
	worker *rpcWorker
	caller string
}

type diagnosticPeerContextKey struct{}

func diagnosticPeer(ctx context.Context) *diagnosticPeerAuthority {
	peer, _ := ctx.Value(diagnosticPeerContextKey{}).(*diagnosticPeerAuthority)
	return peer
}

func diagnosticCaller(ctx context.Context) string {
	if peer := diagnosticPeer(ctx); peer != nil {
		return peer.caller
	}
	return ""
}

// Call under r.mu, then workersMu, through withDiagnosticAuthority. This checks
// the same current source and exact worker pair before acceptance and effects.
func (b *Broker) diagnosticAuthorityLocked(ctx context.Context, r *policyRuntime) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if peer := diagnosticPeer(ctx); peer != nil {
		if peer.source == nil || r.engineManagerSource != peer.source || b.engineMgrSource != peer.source || b.engineMgr != peer.worker || peer.worker == nil {
			return errModelMutationUnavailable
		}
		if err := requireRemoteCaller(peer.caller); err != nil {
			return err
		}
		// Grant-only: diagnosticGate owns the mode-aware Halt check. The
		// generic managementBlockedLocked predicate would reject configured pause.
		return b.authorizeRemoteManagementGrantLocked(r, peer.caller, "model-diagnostic", "accept")
	}
	return nil
}

func (b *Broker) withDiagnosticAuthority(ctx context.Context, mutation bool, accept func() error) error {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if mutation {
		if err := b.diagnosticAuthorityLocked(ctx, r); err != nil {
			return err
		}
	} else {
		if err := ctx.Err(); err != nil {
			return err
		}
		if peer := diagnosticPeer(ctx); peer != nil && (peer.source == nil || r.engineManagerSource != peer.source || b.engineMgrSource != peer.source || b.engineMgr != peer.worker || peer.worker == nil) {
			return errModelMutationUnavailable
		}
	}
	return accept()
}

func (b *Broker) cancelDiagnosticRelay(source *engineManagerSource, raw json.RawMessage) {
	var p struct {
		ID string `json:"id"`
	}
	if !b.engineManagerSourceCurrent(source) || modelconfig.DecodeStrict(raw, &p) != nil {
		return
	}
	b.diagnostics.mu.Lock()
	cancel := b.diagnostics.relayCancels[diagnosticRelayKey{source, p.ID}]
	b.diagnostics.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (b *Broker) handleDiagnosticRelay(source *engineManagerSource, raw json.RawMessage) {
	var request diagnosticRelayRequest
	if modelconfig.DecodeStrict(raw, &request) != nil || request.ID == "" || len(request.ID) > 128 || !b.engineManagerSourceCurrent(source) {
		return
	}
	bound := 25 * time.Second
	if request.Method == "engine:diagnostic-cancel" {
		bound = diagnosticCancelLimit
	}
	ctx, cancel := context.WithTimeout(context.Background(), bound)
	key := diagnosticRelayKey{source, request.ID}
	b.diagnostics.mu.Lock()
	if b.diagnostics.relayCancels == nil {
		b.diagnostics.relayCancels = make(map[diagnosticRelayKey]context.CancelFunc)
	}
	if b.diagnostics.relayCancels[key] != nil || len(b.diagnostics.relayCancels) >= 32 {
		b.diagnostics.mu.Unlock()
		cancel()
		return
	}
	b.diagnostics.relayCancels[key] = cancel
	b.diagnostics.mu.Unlock()
	go func() {
		defer cancel()
		defer func() { b.diagnostics.mu.Lock(); delete(b.diagnostics.relayCancels, key); b.diagnostics.mu.Unlock() }()
		worker, err := b.awaitEngineMgrPair(ctx, source)
		if err != nil || !b.engineManagerSourceCurrent(source) {
			return
		}
		ctx = context.WithValue(ctx, diagnosticPeerContextKey{}, &diagnosticPeerAuthority{source, worker, request.Caller})
		var result any
		if err = requireRemoteCaller(request.Caller); err == nil {
			result, err = b.dispatchDiagnostic(ctx, request.Method, request.Params, false)
		}
		reply := struct {
			ID        string `json:"id"`
			Result    any    `json:"result"`
			Error     string `json:"error,omitempty"`
			ErrorCode int    `json:"errorCode,omitempty"`
		}{ID: request.ID, Result: result}
		if err != nil {
			reply.Error, reply.ErrorCode = err.Error(), engineSettingsErrorCode(err)
		}
		b.replyPeerRelay(source, worker, "engine:diagnostic-reply", reply)
	}()
}

func (b *Broker) remoteDiagnostic(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, error) {
	worker := b.getEngineMgr()
	if worker == nil {
		return nil, fmt.Errorf("diagnostic target unavailable")
	}
	// A cancel may join for minutes. Do not put the rpcWorker.Call five-second
	// deadline around this target-owned operation; ctx already bounds the wait.
	raw, rpcErr, err := worker.CallNoTimeout(ctx, strings.Replace(method, "engine:", "engine:remote-", 1), params)
	if err != nil {
		return nil, err
	}
	if rpcErr != nil {
		return nil, &settingsWorkerError{code: rpcErr.Code, message: rpcErr.Message}
	}
	if !b.diagnosticWorkerCurrent(worker) {
		return nil, diagnosticError("engine manager replaced during remote diagnostic call")
	}
	// The manager projects and validates the public response before returning
	// it. Decode once more at the public broker boundary rather than forwarding
	// raw internal envelopes to clients.
	if method == "engine:diagnostic-preview" {
		var preview modelconfig.DiagnosticPreview
		if err := modelconfig.DecodeStrict(raw, &preview); err != nil {
			return nil, err
		}
		var expected modelconfig.DiagnosticPreviewRequest
		if err := modelconfig.DecodeStrict(params, &expected); err != nil {
			return nil, err
		}
		if err := preview.Validate(); err != nil {
			return nil, err
		}
		if preview.DiagnosticPreviewRequest != expected {
			return nil, diagnosticError("remote diagnostic preview identity mismatch")
		}
		return json.Marshal(preview.Clone())
	}
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		if method == "engine:diagnostic-start" {
			return nil, diagnosticError("remote diagnostic acceptance missing")
		}
		return json.RawMessage("null"), nil
	}
	var run diagnosticPublicRun
	if err := modelconfig.DecodeStrict(raw, &run); err != nil {
		return nil, err
	}
	var target struct {
		NodeID     string `json:"nodeId"`
		RunID      string `json:"runId"`
		Engine     string `json:"engine"`
		ModelID    string `json:"modelId"`
		ProfileKey string `json:"profileKey"`
	}
	// The complete request was strictly decoded by dispatchDiagnostic already.
	if err := json.Unmarshal(params, &target); err != nil {
		return nil, err
	}
	if err := run.Validate(); err != nil {
		return nil, err
	}
	if run.NodeID != target.NodeID || target.RunID != "" && run.RunID != target.RunID || method == "engine:diagnostic-start" && (run.Engine != target.Engine || run.ModelID != target.ModelID || run.ProfileKey != target.ProfileKey) {
		return nil, diagnosticError("remote diagnostic run identity mismatch")
	}
	return json.Marshal(diagnosticPublicRun{DiagnosticRun: run.DiagnosticRun.Clone(), NodeID: target.NodeID})
}

type diagnosticPublicRun struct {
	modelconfig.DiagnosticRun
	NodeID string `json:"nodeId,omitempty"`
}

// Both public local calls and the verified target relay use this decoder.
// Only a public origin may select a peer; inbound relays cannot forward again.
func (b *Broker) dispatchDiagnostic(ctx context.Context, method string, raw json.RawMessage, allowRemote bool) (any, error) {
	var nodeID string
	var params any
	switch method {
	case "engine:diagnostic-preview":
		var p modelconfig.DiagnosticPreviewRequest
		if err := modelconfig.DecodeStrict(raw, &p); err != nil {
			return nil, &invalidParamsError{err}
		}
		if err := p.Validate(); err != nil {
			return nil, &invalidParamsError{err}
		}
		nodeID, params = p.NodeID, p
	case "engine:diagnostic-start":
		var p modelconfig.DiagnosticStartRequest
		if err := modelconfig.DecodeStrict(raw, &p); err != nil {
			return nil, &invalidParamsError{err}
		}
		if err := p.Validate(); err != nil {
			return nil, &invalidParamsError{err}
		}
		nodeID, params = p.NodeID, p
	case "engine:diagnostic-status", "engine:diagnostic-cancel":
		var p modelconfig.DiagnosticRunRef
		if err := modelconfig.DecodeStrict(raw, &p); err != nil {
			return nil, &invalidParamsError{err}
		}
		validation := p
		if method == "engine:diagnostic-status" && validation.RunID == "" {
			validation.RunID = "active-lookup"
		}
		if err := validation.Validate(); err != nil {
			return nil, &invalidParamsError{err}
		}
		nodeID, params = p.NodeID, p
	default:
		return nil, &settingsWorkerError{code: -32601, message: "unknown diagnostic method"}
	}
	if nodeID != "" && nodeID != b.nodeID {
		if !allowRemote {
			return nil, diagnosticError("diagnostic relay must address this target")
		}
		return b.remoteDiagnostic(ctx, method, raw)
	}
	switch p := params.(type) {
	case modelconfig.DiagnosticPreviewRequest:
		p.NodeID = b.nodeID
		return b.previewDiagnostic(ctx, p)
	case modelconfig.DiagnosticStartRequest:
		p.NodeID = b.nodeID
		return b.startDiagnostic(ctx, p)
	case modelconfig.DiagnosticRunRef:
		if method == "engine:diagnostic-cancel" {
			return b.cancelDiagnostic(ctx, p.RunID)
		}
		var run *modelconfig.DiagnosticRun
		err := b.withDiagnosticAuthority(ctx, false, func() (err error) {
			run, err = b.diagnosticStatus(p.RunID)
			return err
		})
		return run, err
	}
	return nil, fmt.Errorf("invalid diagnostic request")
}
