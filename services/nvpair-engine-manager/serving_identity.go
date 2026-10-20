// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"

	"nvpair-shared/engines"
)

// The versioned manifest binds asserted full-file hashes, never local samples.
// Artifact order is the compiler-validated native consumption order.
func servingContentDigest(model engines.ServingModel) string {
	type artifact struct {
		Role   string `json:"role"`
		Size   int64  `json:"sizeBytes"`
		SHA256 string `json:"sha256"`
	}
	items := make([]artifact, 0, len(model.Artifacts))
	for _, a := range model.Artifacts {
		if a.Binding == nil || a.Binding.SizeBytes <= 0 || len(a.SHA256) != 64 {
			return ""
		}
		items = append(items, artifact{a.Role, a.Binding.SizeBytes, strings.ToLower(a.SHA256)})
	}
	if len(items) == 0 {
		return ""
	}
	data, _ := json.Marshal(struct {
		Version   string     `json:"version"`
		Artifacts []artifact `json:"artifacts"`
	}{"pair-serving-content-v1", items})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func servingLaunchDigest(command launchCommand, prepared string) string {
	base := launchDigest(command)
	if base == "" || !contentDigest.MatchString(prepared) {
		return ""
	}
	data, _ := json.Marshal(struct {
		Version  string `json:"version"`
		Launch   string `json:"launchDigest"`
		Prepared string `json:"preparedDigest"`
	}{"pair-serving-launch-v1", base, prepared})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func servingRetentionDisabled(e engines.ServingEnvelope) bool { return e.Retention == "-1" }

func servingEnvelopes(prepared *llamaServingLaunch) []engines.ServingEnvelope {
	result := make([]engines.ServingEnvelope, 0, len(prepared.Envelopes))
	for i, envelope := range prepared.Envelopes {
		if i >= len(prepared.Config.Models) {
			return nil
		}
		envelope.ContentDigest = servingContentDigest(prepared.Config.Models[i])
		result = append(result, envelope.Clone())
	}
	return result
}

// Caller holds st.mu. Only the immutable launch-validated declaration is used;
// native inventory is not an attestation authority.
func servingIdentityLocked(st *engineState) engineModelIdentity {
	id := engineModelIdentity{EngineRunID: st.runID, LaunchDigest: st.launchDigest, ModelDigests: map[string]string{}}
	rt := st.plat.Runtime
	if rt.servingLaunch == nil || rt.servingDrift != "" || !rt.Serving.Equal(rt.servingLaunch.Config) {
		return id
	}
	envelopes := servingEnvelopes(rt.servingLaunch)
	names := map[string]string{}
	for _, envelope := range envelopes {
		normalized := engines.NormalizeModelName(envelope.ModelID, st.manifest.ModelNamingOrDefault())
		if _, exists := names[normalized]; exists || envelope.Validate(envelope.ModelID, envelope.ContentDigest) != nil {
			return id
		}
		names[normalized] = envelope.ModelID
	}
	id.Serving = make(map[string]engines.ServingEnvelope, len(envelopes))
	for _, envelope := range envelopes {
		id.ModelDigests[envelope.ModelID] = envelope.ContentDigest
		id.Serving[envelope.ModelID] = envelope
	}
	return id
}

// ServingIdentityRefusal is a cold validation refusal, not a load failure.
// Reasons contain operational state only, never paths or raw launch material.
type ServingIdentityRefusal struct{ Reason string }

func (r *ServingIdentityRefusal) Error() string { return r.Reason }

type servingRunSnapshot struct {
	run    string
	proc   *managedProc
	rt     Runtime
	config *engines.ServingConfig
}

func snapshotServingRun(st *engineState) (servingRunSnapshot, error) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if !st.running || st.adopted || st.proc == nil || st.runID == "" || st.launchDigest == "" {
		return servingRunSnapshot{}, &ServingIdentityRefusal{"Serving identity requires a current owned run."}
	}
	rt := st.plat.Runtime
	if rt.servingLaunch == nil || rt.Serving == nil || rt.servingDrift != "" {
		return servingRunSnapshot{}, &ServingIdentityRefusal{"Serving launch is missing or withdrawn."}
	}
	config := rt.Serving
	rt.Serving = rt.Serving.Clone()
	return servingRunSnapshot{st.runID, st.proc, rt, config}, nil
}

// Caller holds st.mu. Replaced runs/configurations cannot inherit I/O results.
func (s servingRunSnapshot) current(st *engineState) bool {
	return st.running && !st.adopted && st.runID == s.run && st.proc == s.proc &&
		st.plat.Runtime.servingLaunch == s.rt.servingLaunch && st.plat.Runtime.Serving == s.config && st.plat.Runtime.Serving.Equal(s.rt.Serving)
}

func validateServingSnapshot(ctx context.Context, st *engineState, s servingRunSnapshot) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if launchDigest(s.rt.servingCommand) != s.rt.servingBaseDigest {
		return &ServingIdentityRefusal{"Serving executable no longer matches the owned launch."}
	}
	compiled, err := compilePreparedServing(st, s.rt.servingCommand, s.rt, true)
	if err != nil || compiled.Digest != s.rt.servingLaunch.Digest {
		return &ServingIdentityRefusal{"Serving inputs no longer match the owned launch; renewed verification is required."}
	}
	return ctx.Err()
}

// currentServingEnvelope performs only bounded local validation, outside locks.
// It never probes, loads, or changes configuration. SR owns requested-value and
// resident-set comparisons at its call sites.
func (e *Executor) currentServingEnvelope(ctx context.Context, engine, model string) (engines.ServingEnvelope, engineModelIdentity, error) {
	refuse := func(reason string) (engines.ServingEnvelope, engineModelIdentity, error) {
		return engines.ServingEnvelope{}, engineModelIdentity{}, &ServingIdentityRefusal{reason}
	}
	st, err := e.state(engine)
	if err != nil {
		return refuse("Serving engine is unavailable.")
	}
	s, err := snapshotServingRun(st)
	if err != nil {
		return engines.ServingEnvelope{}, engineModelIdentity{}, err
	}
	if err := validateServingSnapshot(ctx, st, s); err != nil {
		return refuse("Serving inputs failed current launch validation.")
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if !s.current(st) || st.plat.Runtime.servingDrift != "" {
		return refuse("Serving run or configuration changed during validation.")
	}
	id := servingIdentityLocked(st)
	envelope, ok := id.Serving[model]
	if !ok {
		return refuse("Model has no current declared serving envelope.")
	}
	return envelope.Clone(), id, nil
}

// Full inventory reconciliation and action pokes share the existing watcher.
// Validation failure latches until successful explicit apply/start, even if the
// bytes are later restored. Cancellation alone does not invalidate a launch.
func (e *Executor) refreshServingIdentity(ctx context.Context) {
	for _, engine := range e.reg.Names() {
		st, err := e.state(engine)
		if err != nil {
			continue
		}
		s, err := snapshotServingRun(st)
		if err != nil {
			continue
		}
		err = validateServingSnapshot(ctx, st, s)
		if err == nil || ctx.Err() != nil {
			continue
		}
		st.mu.Lock()
		if s.current(st) {
			st.plat.Runtime.servingDrift = "Serving inputs no longer match the owned launch."
		}
		st.mu.Unlock()
	}
}
