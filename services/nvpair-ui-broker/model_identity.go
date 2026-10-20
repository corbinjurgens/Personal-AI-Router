// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// model_identity.go assembles the identity the proxy binds strict capacity and
// local qualification to (nodepolicy.MethodSetModelIdentity,
// docs/local-qualification.mdx). engine-manager reports each engine's run,
// launch and model content identity (engine:model-identity); this broker adds
// the fingerprint of the settings revision applied to it. An engine lacking
// either part has no identity, which withdraws its strict capacity. Before a
// settings operation touches an engine, the proxy must have acknowledged that
// the engine's identity is gone.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
)

// engineManagerIdentity is one engine's part of engine:model-identity.
type engineManagerIdentity struct {
	EngineRunID  string                             `json:"engineRunId"`
	LaunchDigest string                             `json:"launchDigest"`
	ModelDigests map[string]string                  `json:"modelDigests"`
	Serving      map[string]engines.ServingEnvelope `json:"serving,omitempty"`
}

// rpcMethodNotFound is the JSON-RPC code an older proxy answers an unknown
// method with. Such a proxy has no strict capacity to withdraw.
const rpcMethodNotFound = -32601

// A nonzero-sized allocation gives each worker a distinct, immutable source,
// including before startRPCWorker has returned its handle.
type engineManagerSource struct{ marker byte }

func (b *Broker) activateEngineManagerSource() *engineManagerSource {
	source := &engineManagerSource{}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	r.engineManagerSource = source
	r.engineIdentity = nil
	r.mu.Unlock()
	b.identityChanged()
	return source
}

// observeModelIdentity takes engine-manager's whole identity snapshot. It runs
// on engine-manager's read pump, so the proxy push goes out on its own.
func (b *Broker) observeModelIdentity(source *engineManagerSource, params json.RawMessage) {
	b.acceptModelIdentity(source, decodeModelIdentity(params))
}

func decodeModelIdentity(params json.RawMessage) map[string]engineManagerIdentity {
	var p struct {
		Engines map[string]engineManagerIdentity `json:"engines"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		// An unreadable snapshot names no identity at all.
		slog.Warn("engine-manager sent an unreadable model identity; withdrawing it", "err", err)
		p.Engines = nil
	}
	return p.Engines
}

func (b *Broker) acceptModelIdentity(source *engineManagerSource, engines map[string]engineManagerIdentity) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if source == nil || r.engineManagerSource != source {
		r.mu.Unlock()
		return
	}
	r.engineIdentity = make(map[string]engineManagerIdentity, len(engines))
	for name, id := range engines {
		models := make(map[string]string, len(id.ModelDigests))
		for model, digest := range id.ModelDigests {
			models[model] = digest
		}
		id.ModelDigests = models
		id.Serving = cloneIdentityServing(id.Serving)
		r.engineIdentity[name] = id
	}
	r.mu.Unlock()
	b.identityChanged()
}

// retireEngineManagerSource withdraws every run identity when engine-manager
// exits: the processes it reported may stop or be replaced without it.
func (b *Broker) retireEngineManagerSource(source *engineManagerSource) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if source == nil || r.engineManagerSource != source {
		r.mu.Unlock()
		return
	}
	r.engineManagerSource = nil
	r.engineIdentity = nil
	r.mu.Unlock()
	b.identityChanged()
}

// observeSettingsIdentity takes the settings snapshots publishSettingsLocked
// publishes. Only an applied revision has a fingerprint. It never waits for
// the proxy; withdrawSettingsIdentity is the confirmed path.
func (b *Broker) observeSettingsIdentity(snapshots []settings.Snapshot) {
	next := make(map[string]string, len(snapshots))
	for _, s := range snapshots {
		if fp, ok := settingsFingerprint(s); ok {
			next[s.Engine] = fp
		}
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	changed := !sameStrings(r.settingsIdentity, next)
	r.settingsIdentity = next
	r.mu.Unlock()
	if changed {
		b.identityChanged()
	}
}

// identityChanged pushes the identity to the proxy and reconsiders the
// published capability, whose models must match it (readiness.go).
func (b *Broker) identityChanged() {
	b.schedulePolicyPush(pushIdentity)
	go b.refreshStrictAdvertisement()
}

// settingsFingerprint is the SHA-256 of an applied settings revision: the
// journal epoch, the applied revision and the settings. A revision being
// applied, a failed one, or one not yet applied has none.
func settingsFingerprint(s settings.Snapshot) (string, bool) {
	if s.Epoch == "" || s.Revision == 0 || s.AppliedRevision != s.Revision || s.Phase == "applying" || s.Phase == "failed" {
		return "", false
	}
	doc, err := json.Marshal(struct {
		Version         int             `json:"version"`
		Epoch           string          `json:"epoch"`
		AppliedRevision uint64          `json:"appliedRevision"`
		Settings        settings.Config `json:"settings"`
	}{1, s.Epoch, s.AppliedRevision, s.Settings})
	if err != nil {
		return "", false
	}
	sum := sha256.Sum256(doc)
	return hex.EncodeToString(sum[:]), true
}

// modelIdentityLocked joins both parts. An engine whose joined identity does
// not validate is left out rather than failing the whole push. Caller holds
// r.mu.
func (r *policyRuntime) modelIdentityLocked() nodepolicy.SetModelIdentityParams {
	out := nodepolicy.SetModelIdentityParams{Engines: map[string]nodepolicy.EngineIdentity{}}
	for engine, em := range r.engineIdentity {
		fp, ok := r.settingsIdentity[engine]
		if !ok {
			continue
		}
		id := nodepolicy.EngineIdentity{EngineRunID: em.EngineRunID, SettingsFingerprint: fp, LaunchDigest: em.LaunchDigest, ModelDigests: map[string]string{}}
		for model, digest := range em.ModelDigests {
			id.ModelDigests[model] = digest
		}
		id.Serving = cloneIdentityServing(em.Serving)
		if id.Validate() != nil {
			continue
		}
		out.Engines[engine] = id
	}
	return out
}

func cloneIdentityServing(source map[string]engines.ServingEnvelope) map[string]engines.ServingEnvelope {
	if source == nil {
		return nil
	}
	result := make(map[string]engines.ServingEnvelope, len(source))
	for model, envelope := range source {
		result[model] = envelope.Clone()
	}
	return result
}

// identityCurrentLocked reports whether m's identity is the one this broker
// knows for its engine now: the same run and settings fingerprint, and a model
// digest for the same normalized model name. Caller holds r.mu.
func (r *policyRuntime) identityCurrentLocked(m nodepolicy.StrictAdmissionModel, naming string) bool {
	id, ok := r.modelIdentityLocked().Engines[m.Engine]
	if !ok || id.EngineRunID != m.Identity.EngineRunID || id.SettingsFingerprint != m.Identity.SettingsFingerprint {
		return false
	}
	normalize := func(model string) string {
		model = strings.TrimSpace(model)
		if nodepolicy.IsReservedAlias(model) {
			return model
		}
		return engines.NormalizeModelName(model, naming)
	}
	want := normalize(m.Model)
	found := false
	for model, digest := range id.ModelDigests {
		if normalize(model) == want {
			if digest != m.Identity.ModelDigest {
				return false
			}
			found = true
		}
	}
	return found
}

// withdrawSettingsIdentity removes engine's settings fingerprint and returns
// once the proxy has acknowledged an identity without it, so nothing can be
// admitted under the old settings while they change. With no proxy running
// there is nothing to withdraw from, and the next one is replayed the
// withdrawn identity. Any other failure is returned, and the caller must not
// change the engine.
func (b *Broker) withdrawSettingsIdentity(ctx context.Context, engine string) error {
	b.forgetSettingsIdentity(engine)
	return b.pushModelIdentityConfirmed(ctx)
}

func (b *Broker) forgetSettingsIdentity(engine string) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	delete(r.settingsIdentity, engine)
	r.mu.Unlock()
	go b.refreshStrictAdvertisement()
}

// Confirm the current identity without holding engineConfigMu across the push
// slot: proxy replay takes that slot before reading the switch journal.
func (b *Broker) pushModelIdentityConfirmed(ctx context.Context) error {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	bound := r.activationBoundLocked()
	r.mu.Unlock()
	ctx, cancel := context.WithTimeout(ctx, bound)
	defer cancel()
	if !r.acquirePush(ctx) {
		return errors.New("strict admission could not be withdrawn before the settings change: the proxy is busy")
	}
	defer r.releasePush()
	pp := b.policyProxy()
	if pp == nil {
		return nil
	}
	r.mu.Lock()
	data, err := json.Marshal(r.modelIdentityLocked())
	r.mu.Unlock()
	if err != nil {
		return err
	}
	_, rpcErr, err := pp.Call(ctx, nodepolicy.MethodSetModelIdentity, data)
	switch {
	case err != nil:
		return fmt.Errorf("strict admission could not be withdrawn before the settings change: %w", err)
	case rpcErr != nil && rpcErr.Code != rpcMethodNotFound:
		return fmt.Errorf("the proxy refused to withdraw strict admission before the settings change (code %d)", rpcErr.Code)
	}
	return nil
}

func sameStrings(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if w, ok := b[k]; !ok || w != v {
			return false
		}
	}
	return true
}
