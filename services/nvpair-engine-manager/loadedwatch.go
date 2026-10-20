// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"sort"
	"time"

	"nvpair-shared/performancewire"
)

// modelsChangedMethod is the push emitted whenever an engine's set of loaded
// (in-memory) models changes.
const modelsChangedMethod = "engine:models-changed"

// defaultLoadedPollSeconds is the default cadence of the loaded-model watcher,
// shared by NewExecutor and the --loaded-poll-interval flag so the two can't
// drift. 0 (via the flag) disables the watcher.
const defaultLoadedPollSeconds = 5

// modelsChangedParams is the engine:models-changed payload: which engine's
// loaded set changed (the trigger) plus the full current model result — the same
// shape engine:models returns — so a push-driven consumer swaps its whole
// snapshot instead of applying a delta.
type modelsChangedParams struct {
	Engine string       `json:"engine"`
	Models ModelsResult `json:"models"`
}

// residencyNeutralActions are engine actions that only read state and so can
// never change what's resident in memory. pokeLoaded is skipped after them (see
// Manager.runAction) to avoid a wasted out-of-cycle sweep. Anything not listed
// here — load_model/unload_model/run_model/pull_model/delete_model, and chat,
// which can JIT-load a model — is treated as potentially residency-affecting.
var residencyNeutralActions = map[string]bool{
	"list_models":     true,
	"list_downloaded": true,
	"loaded_models":   true,
}

// watchLoaded polls the loaded-model set of every running engine and emits
// engine:models-changed whenever it changes — covering explicit load/unload, LM
// Studio JIT auto-load, and TTL/idle auto-eviction. The engines themselves push
// nothing, so this internal poll is the only signal; keeping it here means a
// push-driven consumer never has to poll. It runs until ctx is cancelled and is
// pokeable (see Executor.pokeLoaded) for sub-tick latency after an explicit
// action. The first sweep seeds loaded names and publishes any memory baseline;
// later memory changes use the same notification even if names are unchanged.
func (e *Executor) watchLoaded(ctx context.Context) {
	interval := e.loadedPollInterval
	if interval <= 0 {
		return // watcher disabled
	}
	t := time.NewTicker(interval)
	defer t.Stop()

	var prev map[string][]string
	var previousMemory *performancewire.MemoryEvidence
	var lastInventory time.Time
	var identity *modelIdentityParams
	seeded := false
	for {
		poked := false
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		case <-e.loadedPoke:
			poked = true
		}
		// A poke follows a PAIR action (pull, delete, load) that may have changed
		// the installed list, so it always refreshes the inventory too.
		fullInventory := poked || time.Since(lastInventory) >= inventoryReconcileInterval
		if fullInventory {
			lastInventory = time.Now()
		}
		changed, next, res := e.sweepLoaded(ctx, prev, fullInventory)
		prev = next
		memoryChanged := changedMemoryEngines(previousMemory, res.MemoryEvidence)
		previousMemory = res.MemoryEvidence
		// Identity is reported from the first sweep on: its consumer starts
		// with none, so it must never wait for a change.
		identity = e.publishModelIdentity(identity)
		if !seeded {
			seeded = true
			changed = nil // loaded names retain their existing first-sweep behavior
		}
		changed = mergeMemoryChanges(changed, memoryChanged)
		for _, name := range changed {
			e.notify(modelsChangedMethod, modelsChangedParams{Engine: name, Models: res})
		}
	}
}

// A successful first memory baseline, memory-only changes and withdrawal use
// the existing models notification. Elapsed ages and a new successful query's
// generation alone do not constitute a changed memory status.
func changedMemoryEngines(previous, current *performancewire.MemoryEvidence) []string {
	before := memoryStatusByEngine(previous)
	after := memoryStatusByEngine(current)
	changed := make(map[string]bool)
	for name, status := range before {
		if !bytes.Equal(status, after[name]) {
			changed[name] = true
		}
	}
	for name, status := range after {
		if !bytes.Equal(status, before[name]) {
			changed[name] = true
		}
	}
	var names []string
	for name := range changed {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func memoryStatusByEngine(evidence *performancewire.MemoryEvidence) map[string][]byte {
	byEngine := make(map[string][]performancewire.MemoryEvidenceEntry)
	if evidence != nil {
		for _, entry := range evidence.Entries {
			entry.Freshness.SampleAgeMs = nil
			entry.Freshness.QualificationAgeMs = nil
			entry.Freshness.Generation = nil
			byEngine[entry.Engine] = append(byEngine[entry.Engine], entry)
		}
	}
	status := make(map[string][]byte, len(byEngine))
	for name, entries := range byEngine {
		data, err := json.Marshal(entries)
		if err == nil {
			status[name] = data
		}
	}
	return status
}

func mergeMemoryChanges(names, memoryNames []string) []string {
	set := make(map[string]bool, len(names)+len(memoryNames))
	for _, name := range names {
		set[name] = true
	}
	for _, name := range memoryNames {
		set[name] = true
	}
	result := make([]string, 0, len(set))
	for name := range set {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}

// sweepLoaded performs one loaded-set poll: it fetches the current model result
// and computes which engines' loaded sets changed relative to prevLoaded. It
// returns those engine names, the refreshed baseline to carry into the next
// sweep, and the full result (the push payload). The baseline is prevLoaded with
// every engine queried this sweep overwritten by its fresh value; an engine
// absent from this sweep keeps its last-good entry (see changedEngines) so a
// transient loaded_models miss isn't mistaken for an unload. It never mutates
// prevLoaded. Split out of watchLoaded so the diff logic is unit-testable
// without goroutines or timers.
func (e *Executor) sweepLoaded(ctx context.Context, prevLoaded map[string][]string, fullInventory bool) (changed []string, next map[string][]string, res ModelsResult) {
	res = e.modelsSweep(ctx, fullInventory)
	changed = changedEngines(prevLoaded, res.LoadedByEngine)
	next = make(map[string][]string, len(prevLoaded)+len(res.LoadedByEngine))
	for name, ld := range prevLoaded {
		next[name] = ld
	}
	for name, ld := range res.LoadedByEngine {
		next[name] = ld
	}
	return changed, next, res
}

// pokeLoaded requests an immediate loaded-set check (non-blocking, coalesced).
// Called after an action that may change residency (load/unload/pull/…) so the
// engine:models-changed push doesn't wait up to a full poll interval.
func (e *Executor) pokeLoaded() {
	select {
	case e.loadedPoke <- struct{}{}:
	default:
	}
}

// changedEngines returns, sorted, the engine names whose loaded set was reported
// this sweep (present in cur) and differs from the baseline prev — either a new
// key or a key whose set changed order-insensitively.
//
// A key present only in prev is deliberately NOT reported. The watcher can't
// tell an engine that genuinely stopped from one whose loaded_models query
// merely blipped this sweep (both drop the key), and a real stop already reaches
// consumers via engine:state-changed. Treating every disappearance as a change
// would emit spurious engine:models-changed churn — a drop then a re-add — on a
// single transient failure, so the watcher keeps such keys as last-good (see
// sweepLoaded) and waits for a definitive reading.
func changedEngines(prev, cur map[string][]string) []string {
	var changed []string
	for name, cv := range cur {
		if pv, ok := prev[name]; !ok || !sameStringSet(pv, cv) {
			changed = append(changed, name)
		}
	}
	sort.Strings(changed)
	return changed
}

// sameStringSet reports whether a and b contain the same elements, ignoring
// order and treating nil and empty as equal.
func sameStringSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	counts := make(map[string]int, len(a))
	for _, s := range a {
		counts[s]++
	}
	for _, s := range b {
		counts[s]--
		if counts[s] < 0 {
			return false
		}
	}
	return true
}
