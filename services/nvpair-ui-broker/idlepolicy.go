// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// idlepolicy.go releases idle model memory and handles admission wake.
// Each model has its own activity and residency baseline. Engines stay running.

import (
	"context"
	"log/slog"
	"sort"
	"strings"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// idleCheckInterval is how often the idle policy runs. Coarse on purpose: the
// thresholds are minutes, and an idle node should cost next to nothing.
const idleCheckInterval = 30 * time.Second

// runIdlePolicy runs the idle policy until ctx ends.
func (b *Broker) runIdlePolicy(ctx context.Context) {
	b.signalModelConfig()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	wake := r.models.wake
	r.mu.Unlock()
	ticker := time.NewTicker(idleCheckInterval)
	defer ticker.Stop()
	for {
		r.mu.Lock()
		retryAt := r.models.retryAt
		retryDelay := retryAt.Sub(r.now())
		r.mu.Unlock()
		var retry *time.Timer
		var retryC <-chan time.Time
		if !retryAt.IsZero() {
			retry = time.NewTimer(max(0, retryDelay))
			retryC = retry.C
		}
		select {
		case <-ctx.Done():
			if retry != nil {
				retry.Stop()
			}
			return
		case <-ticker.C:
			if retry != nil {
				retry.Stop()
			}
			b.idleTick(ctx)
			b.reconcileModelConfig(ctx)
		case <-wake:
			if retry != nil {
				retry.Stop()
			}
			b.reconcileModelConfig(ctx)
		case <-retryC:
			b.reconcileModelConfig(ctx)
		}
	}
}

type idleUnload struct {
	engine string
	models []string
}

// idleTick unloads each nonresident model once per idle baseline. Activity on
// another model of the same engine does not renew this model's baseline.
func (b *Broker) idleTick(ctx context.Context) {
	pp := b.policyProxy()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	idle := r.current.Idle
	if r.live != nodepolicy.Available || idle.UnloadAfterMinutes <= 0 {
		r.mu.Unlock()
		return
	}
	now := r.now()
	unloadAfter := time.Duration(idle.UnloadAfterMinutes) * time.Minute
	engines := make([]string, 0, len(r.running))
	for engine := range r.running {
		engines = append(engines, engine)
	}
	sort.Strings(engines)
	var unloads []idleUnload
	drains := r.engineDrainsLocked(pp)
	for _, engine := range engines {
		if drains[engine] {
			continue
		}
		for _, model := range r.loaded[engine] {
			ref := b.modelRef(engine, model)
			if b.protectedResidentLocked(engine, model) || modelActive(r.admission, ref) {
				continue
			}
			base := r.idleBaselineLocked(ref)
			if base.IsZero() {
				r.residencyAt[ref] = now
				continue
			}
			if now.Sub(base) >= unloadAfter && !r.idleUnloadedAt[ref].Equal(base) {
				r.idleUnloadedAt[ref] = base
				unloads = append(unloads, idleUnload{engine: engine, models: []string{model}})
			}
		}
	}
	r.mu.Unlock()

	for _, u := range unloads {
		if ctx.Err() != nil {
			return
		}
		slog.Info("unloading idle model", "engine", u.engine)
		b.unloadModels(u.engine, u.models, "idle")
	}
}

func modelActive(state nodepolicy.AdmissionState, ref nodepolicy.ModelRef) bool {
	return state.ActiveByModel[ref.Engine][ref.Model] > 0
}

// protectedResidentLocked checks the current desired resident set at the
// destructive action boundary. An empty model checks whether the engine has
// any resident preference. Caller holds the policy runtime mutex.
func (b *Broker) protectedResidentLocked(engine, model string) bool {
	r := &b.nodePolicy
	naming, _ := b.engineModelNaming(engine)
	normalize := func(name string) string {
		return engines.NormalizeModelName(strings.TrimSpace(name), naming)
	}
	for _, ref := range r.current.ResidentModels {
		if ref.Engine == engine && (model == "" || normalize(ref.Model) == normalize(model)) {
			return true
		}
	}
	return false
}

// idleBaselineLocked is when this model last did anything. Caller holds r.mu.
func (r *policyRuntime) idleBaselineLocked(ref nodepolicy.ModelRef) time.Time {
	base := r.runningSince[ref.Engine]
	if ms := r.admission.LastActivityMs[ref.Engine][ref.Model]; ms > 0 {
		if t := time.UnixMilli(ms); t.After(base) {
			base = t
		}
	}
	if t := r.residencyAt[ref]; t.After(base) {
		base = t
	}
	return base
}

// wakeOnDemand answers the proxy's admission/wake. It is refused while the
// node is not available, and a wake already in
// flight for the engine is not repeated.
func (b *Broker) wakeOnDemand(engine string) {
	if reason := b.wakeRefusal(); reason != "" {
		slog.Info("not starting engine on demand", "engine", engine, "reason", reason)
		return
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if r.waking[engine] {
		r.mu.Unlock()
		return
	}
	r.waking[engine] = true
	r.mu.Unlock()
	defer func() { r.mu.Lock(); delete(r.waking, engine); r.mu.Unlock() }()
	slog.Info("starting engine on demand", "engine", engine)
	if _, err := b.callEngine(context.Background(), engineWakeMethod, map[string]string{"engine": engine}); err != nil {
		slog.Warn("on-demand engine start failed", "engine", engine, "err", err)
	}
}
