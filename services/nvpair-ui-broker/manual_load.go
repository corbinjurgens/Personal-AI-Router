// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

const admittedLoadRequestMethod = "engine:admitted-load-request"
const admittedLoadReplyMethod = "engine:admitted-load-reply"

type manualLoadParams struct {
	Engine string `json:"engine"`
	Model  string `json:"model"`
}

func decodeManualLoad(raw json.RawMessage) (manualLoadParams, error) {
	var p manualLoadParams
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&p); err != nil || !engines.ValidID(p.Engine) || nodepolicy.ValidateModelIdentifier(p.Model) != nil {
		return p, fmt.Errorf("engine and model are required")
	}
	return p, nil
}

// manualLoad enters the same owner as automatic resident loads. Only profile
// qualification is omitted; native envelope, identity and capacity are required.
func (b *Broker) manualLoad(ctx context.Context, p manualLoadParams, source *engineManagerSource) *RPCError {
	e := b.captureModelConfig()
	if e.Paused || e.Live != nodepolicy.Available {
		return &RPCError{Code: codeLocalVeto, Message: (&localVetoError{}).Error()}
	}
	if reason := b.wakeRefusal(); reason != "" {
		return &RPCError{Code: -32000, Message: reason}
	}
	if e.Activation.Activation != activationActive || e.Source == nil || e.Worker == nil || e.Proxy == nil || (source != nil && e.Source != source) {
		return &RPCError{Code: -32000, Message: "Model load unavailable: current node, proxy and engine observations are required."}
	}
	ref := b.modelRef(p.Engine, p.Model)
	if !e.Running[ref.Engine] || e.Drains[ref.Engine] || e.Loaded[ref.Engine] == nil {
		return &RPCError{Code: -32000, Message: "Model load unavailable: engine is stopped, draining or not observed."}
	}
	native := e.Identity.Engines[ref.Engine]
	id := nodepolicy.StrictModelIdentity{EngineRunID: native.EngineRunID, SettingsFingerprint: native.SettingsFingerprint}
	matches := 0
	for model, digest := range native.ModelDigests {
		if b.modelRef(ref.Engine, model) == ref {
			matches++
			id.ModelDigest = digest
		}
	}
	envelope, found := native.Serving[ref.Model]
	if matches != 1 || id.Validate() != nil || !found || envelope.Validate(ref.Model, id.ModelDigest) != nil || envelope.PerSlotContextTokens <= 0 || uint64(envelope.PerSlotContextTokens) > uint64(^uint(0)>>1) {
		return &RPCError{Code: -32000, Message: "Model load unsupported: an authoritative serving envelope and model identity are required."}
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if r.models.busy {
		r.mu.Unlock()
		return &RPCError{Code: -32000, Message: "Model load busy: another model operation is active."}
	}
	if _, held := r.models.unsettled[ref]; held {
		r.mu.Unlock()
		return &RPCError{Code: -32000, Message: "Model load unavailable: the earlier load has not settled."}
	}
	r.models.busy = true
	r.mu.Unlock()
	defer func() {
		r.mu.Lock()
		r.models.busy = false
		r.models.active = nil
		r.models.last = nil
		r.mu.Unlock()
		b.signalModelConfig()
	}()
	count := 0
	for _, resident := range b.residentRefs(e.Policy) {
		if resident.Engine == ref.Engine {
			count++
		}
	}
	entry := nodepolicy.ModelConfigurationEntry{Engine: ref.Engine, Model: ref.Model}
	profile := nodepolicy.Profile{ContextTokens: int(envelope.PerSlotContextTokens)}
	entry = b.reconcileModelLoad(ctx, e, entry, profile, id, max(1, count), nodepolicy.ResidentPurposeManual)
	if entry.State != nodepolicy.ModelStateSatisfied {
		return &RPCError{Code: -32000, Message: "Model load did not complete: " + string(entry.Reason)}
	}
	return nil
}

func (b *Broker) handleManualLoad(msg *Message) {
	p, err := decodeManualLoad(msg.Params)
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	if rpcErr := b.manualLoad(context.Background(), p, nil); rpcErr != nil {
		_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
		return
	}
	_ = b.codec.Respond(msg.ID, map[string]bool{"ok": true})
}

// Peer HTTP authorization stays at engine-manager. This private transport is
// bound to that target manager; no client or peer may call it directly.
func (b *Broker) admittedLoadFrom(source *engineManagerSource, raw json.RawMessage) {
	var request struct {
		ID     string          `json:"id"`
		Params json.RawMessage `json:"params"`
	}
	if !residentEnvelope(raw) || json.Unmarshal(raw, &request) != nil || request.ID == "" || len(request.ID) > 128 || !b.engineManagerSourceCurrent(source) {
		return
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		worker, err := b.awaitEngineMgrPair(ctx, source)
		if err != nil {
			return
		}
		p, err := decodeManualLoad(request.Params)
		var rpcErr *RPCError
		if err != nil {
			rpcErr = &RPCError{Code: -32602, Message: err.Error()}
		} else {
			rpcErr = b.manualLoad(ctx, p, source)
		}
		reply := struct {
			ID     string          `json:"id"`
			Result json.RawMessage `json:"result,omitempty"`
			Error  *RPCError       `json:"error,omitempty"`
		}{ID: request.ID, Error: rpcErr}
		if rpcErr == nil {
			reply.Result = json.RawMessage(`{"ok":true}`)
		}
		b.replyPeerRelay(source, worker, admittedLoadReplyMethod, reply)
	}()
}
