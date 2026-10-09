// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// idlepolicy.go is the cross-engine idle policy and start-on-demand
// (FORK_DESIGN.md §3.5). The proxy reports when each engine last started or
// finished a request; every idleCheckInterval the broker unloads an idle
// engine's models and stops an idle engine, as idle.* says. A stopped engine
// keeps its saved intent On, so the proxy's admission/wake starts it again when
// a request for it arrives. An engine saved Off is never started by any of this,
// and nothing is started while the node is paused.

import (
	"context"
	"log/slog"
	"sort"
	"time"

	"nvpair-shared/nodepolicy"
)

// idleCheckInterval is how often the idle policy runs. Coarse on purpose: the
// thresholds are minutes, and an idle node should cost next to nothing.
const idleCheckInterval = 30 * time.Second

// runIdlePolicy runs the idle policy until ctx ends.
func (b *Broker) runIdlePolicy(ctx context.Context) {
	ticker := time.NewTicker(idleCheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			b.idleTick(ctx)
		}
	}
}

type idleUnload struct {
	engine string
	models []string
}

// idleTick applies the idle policy once. An engine is idle from the latest of
// its last admitted activity, the moment it was seen to start, and the moment
// it last gained a loaded model (a model loaded outside admission is activity
// too). Each action is taken once per idle baseline, so an engine that refuses
// to stop (one run outside PAIR, say) is not retried every tick.
func (b *Broker) idleTick(ctx context.Context) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	idle := r.current.Idle
	if r.live == nodepolicy.Draining || (idle.UnloadAfterMinutes <= 0 && idle.StopEngineAfterMinutes <= 0) {
		r.mu.Unlock()
		return
	}
	now := r.now()
	unloadAfter := time.Duration(idle.UnloadAfterMinutes) * time.Minute
	stopAfter := time.Duration(idle.StopEngineAfterMinutes) * time.Minute
	engines := make([]string, 0, len(r.running))
	for engine := range r.running {
		engines = append(engines, engine)
	}
	sort.Strings(engines)
	var unloads []idleUnload
	var sleeps []string
	for _, engine := range engines {
		if r.drains[engine] || r.admission.ActiveByEngine[engine] > 0 {
			continue
		}
		base := r.idleBaselineLocked(engine)
		if base.IsZero() {
			r.runningSince[engine] = now
			continue
		}
		idleFor := now.Sub(base)
		if unloadAfter > 0 && idleFor >= unloadAfter && len(r.loaded[engine]) > 0 && !r.idleUnloadedAt[engine].Equal(base) {
			r.idleUnloadedAt[engine] = base
			unloads = append(unloads, idleUnload{engine: engine, models: append([]string{}, r.loaded[engine]...)})
		}
		if stopAfter > 0 && idleFor >= stopAfter && r.intent[engine] && !r.idleSleptAt[engine].Equal(base) {
			r.idleSleptAt[engine] = base
			sleeps = append(sleeps, engine)
		}
	}
	r.mu.Unlock()

	for _, u := range unloads {
		slog.Info("unloading idle engine's models", "engine", u.engine, "models", len(u.models))
		b.unloadModels(u.engine, u.models, "idle")
	}
	for _, engine := range sleeps {
		if ctx.Err() != nil {
			return
		}
		slog.Info("stopping idle engine", "engine", engine)
		if _, err := b.callEngine(ctx, engineSleepMethod, map[string]string{"engine": engine}); err != nil {
			slog.Warn("could not stop idle engine", "engine", engine, "err", err)
		}
	}
}

// idleBaselineLocked is when engine last did anything. Caller holds r.mu.
func (r *policyRuntime) idleBaselineLocked(engine string) time.Time {
	base := r.runningSince[engine]
	if ms := r.admission.LastActivityMs[engine]; ms > 0 {
		if t := time.UnixMilli(ms); t.After(base) {
			base = t
		}
	}
	if t := r.residencyAt[engine]; t.After(base) {
		base = t
	}
	return base
}

// wakeOnDemand answers the proxy's admission/wake. It is refused while the
// node is not available and for an engine saved Off, and a wake already in
// flight for the engine is not repeated.
func (b *Broker) wakeOnDemand(engine string) {
	if reason := b.wakeRefusal(); reason != "" {
		slog.Info("not starting engine on demand", "engine", engine, "reason", reason)
		return
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	enabled, known := r.intent[engine]
	if (known && !enabled) || r.waking[engine] {
		r.mu.Unlock()
		if known && !enabled {
			slog.Info("not starting engine on demand: it is saved Off", "engine", engine)
		}
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
