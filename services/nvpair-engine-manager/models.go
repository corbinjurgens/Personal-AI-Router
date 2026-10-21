// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"sync"
	"time"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/performancewire"
)

// modelsTimeout bounds the whole engine:models sweep so one hung engine can't
// stall the model list. list_models is a cheap loopback GET, so this is short
// (unlike the 30-minute engine:action ceiling that also has to cover pulls).
const modelsTimeout = 5 * time.Second

// ModelsResult is the model payload the LAN HTTP surface (/v1/models) and the
// engine:models RPC return: the flat, de-duplicated union of every running
// engine's models plus the per-engine attribution keyed by engine name (e.g.
// "ollama", "lmstudio"). Models is retained unchanged for consumers that only
// need the node-level set; ByEngine is the additive attribution a per-engine
// consumer needs (a node's engine card, and the proxies' own-engine model-owner
// ranking). ByEngine includes every running engine whose inventory was
// successfully queried, so an empty list means "running, no models available"
// while a missing key means "not running / not queryable".
//
// LoadedByEngine names the models currently resident in memory, per engine —
// normally a subset of that engine's ByEngine list. It is populated for every
// running engine that declares a loaded_models action AND was successfully
// queried, so an engine key with an empty list means "running, nothing loaded"
// while a MISSING key means "not running / not queryable / no loaded endpoint" —
// a distinction a consumer needs to reflect real state. The subset
// relationship is best-effort: list_models and loaded_models are queried
// independently, so in the rare sweep where an engine's list_models fails but
// loaded_models succeeds, a loaded name can appear here without a matching
// ByEngine/Models entry until the next successful sweep. Its own omitempty drops
// the whole map when no engine reports loaded state.
//
// Policy is this node's policy advertisement as the broker last pushed it
// (nodeprojection.go). It is omitted until the first push, and an older peer
// never sends it; either way a reader treats the node as declaring nothing. The
// engine:models-changed notification carries no Policy.
type ModelsResult struct {
	Models         []string                        `json:"models"`
	ByEngine       map[string][]string             `json:"modelsByEngine,omitempty"`
	LoadedByEngine map[string][]string             `json:"loadedByEngine,omitempty"`
	Policy         *nodepolicy.Advertisement       `json:"policy,omitempty"`
	MemoryEvidence *performancewire.MemoryEvidence `json:"memoryEvidence,omitempty"`
}

// UnmarshalJSON keeps optional memory evidence from invalidating the model
// inventory. The evidence decoder remains strict; rejected evidence is absent.
func (m *ModelsResult) UnmarshalJSON(data []byte) error {
	type inventory ModelsResult
	next := inventory(*m)
	next.MemoryEvidence = nil
	wire := struct {
		*inventory
		MemoryEvidence json.RawMessage `json:"memoryEvidence"`
	}{inventory: &next}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	if len(wire.MemoryEvidence) != 0 && !bytes.Equal(bytes.TrimSpace(wire.MemoryEvidence), []byte("null")) {
		var evidence performancewire.MemoryEvidence
		if err := json.Unmarshal(wire.MemoryEvidence, &evidence); err == nil {
			next.MemoryEvidence = &evidence
		}
	}
	*m = ModelsResult(next)
	return nil
}

// Models returns the union of model names served by every installed, running
// engine that declares a list_models action with a result-extraction spec —
// the flat node-level list. It is a thin wrapper over {@link ModelsResult} kept
// for callers that only need the union (and the existing test surface).
func (e *Executor) Models(ctx context.Context) []string {
	return e.ModelsResult(ctx).Models
}

// ModelsResult queries each installed, running engine's loopback list_models
// endpoint (reusing the manifest action machinery), normalizes the
// engine-specific shapes, and returns both the flat de-duplicated union and the
// per-engine attribution — from a single sweep, so a caller needing both
// (the /v1/models HTTP surface, the engine:models RPC) never sweeps twice.
// Stopped engines contribute nothing — their model list isn't queryable while
// the server is down. Best-effort: a per-engine failure is logged and skipped
// so the engines that did answer still surface their models.
//
// The per-engine queries run CONCURRENTLY. A host running several engines (e.g.
// Ollama + LM Studio) must not have the later engines squeezed out of a single
// shared deadline: sequential queries under one budget meant a slow first engine
// left little for the second, and the peer's discovery daemon fetches this over
// HTTP with its own (shorter) timeout, so a squeezed second engine silently
// dropped its models for peers. Concurrency bounds the sweep by the slowest
// single engine, not the sum.
func (e *Executor) ModelsResult(ctx context.Context) ModelsResult {
	res := e.modelsSweep(ctx, true)
	res.Policy = e.nodeProjection.advertisement()
	return res
}

// inventoryReconcileInterval is how often the loaded-model watcher refreshes
// each engine's installed inventory. Residency (what is loaded) changes on its
// own — JIT loads, TTL evictions — so the watcher checks it every tick, but the
// installed list only changes when someone pulls or deletes a model. PAIR's own
// pulls and deletes poke the watcher, which forces a full sweep; this interval
// only bounds how long a change made outside PAIR takes to show up.
const inventoryReconcileInterval = 60 * time.Second

// modelsSweep is ModelsResult with control over the installed inventory. With
// fullInventory false, an engine whose list_models is a separate request from
// loaded_models reuses its last inventory instead of querying it again; an
// engine with no cached inventory, or whose two actions share one request, is
// still queried. Every successful inventory query refreshes the cache.
func (e *Executor) modelsSweep(ctx context.Context, fullInventory bool) ModelsResult {
	ctx, cancel := context.WithTimeout(ctx, modelsTimeout)
	defer cancel()
	sweep := e.sweeps.Add(1)
	cached := e.cachedInventory()
	memoryCapture := e.captureMemoryEvidence()

	engineNames := e.reg.Names()
	// Results indexed by engine position so the flattened output stays
	// deterministic (registration order) despite concurrent completion.
	perEngine := make([][]string, len(engineNames))
	digests := make([]observedDigests, len(engineNames))
	listOK := make([]bool, len(engineNames))
	loaded := make([][]string, len(engineNames))
	loadedOK := make([]bool, len(engineNames))
	loadedRaw := make([]json.RawMessage, len(engineNames))
	var wg sync.WaitGroup
	for i, name := range engineNames {
		mf, ok := e.reg.Get(name)
		if !ok {
			continue
		}
		var listSpec, loadedSpec *ActionResult
		listAct, hasList := mf.Actions["list_models"]
		if hasList {
			listSpec = listAct.Result
		}
		loadedAct, hasLoaded := mf.Actions["loaded_models"]
		if hasLoaded {
			loadedSpec = loadedAct.Result
		}
		shared := listSpec != nil && loadedSpec != nil && sameReadRequest(listAct, loadedAct)
		inv, haveInv := cached[name]
		// An inventory that carries content digests is read on every sweep:
		// strict admission identity must not wait for the reconcile interval.
		reuseInv := !fullInventory && !shared && haveInv && (listSpec == nil || listSpec.Digest == "")
		wg.Add(1)
		go func(i int, name string, listSpec, loadedSpec *ActionResult, shared bool) {
			defer wg.Done()
			loadedTransportFailed := false
			servingState, stateErr := e.state(name)
			if stateErr == nil {
				if run, err := snapshotLlamaObservation(servingState); err == nil {
					defer func() {
						e.observeLlamaServing(ctx, servingState, run, sweep, loaded[i], loadedOK[i], loadedRaw[i], loadedTransportFailed)
					}()
				}
			}
			st, err := e.Status(name)
			if err != nil || !st.Running {
				return
			}
			var listRaw json.RawMessage
			var listErr error
			if listSpec != nil && reuseInv {
				perEngine[i] = inv
				listOK[i] = true
			} else if listSpec != nil {
				e.inventoryMu.Lock()
				epoch := e.inventoryDigests[name].epoch
				e.inventoryMu.Unlock()
				run := e.engineRunID(name)
				listRaw, listErr = e.Action(ctx, name, "list_models", nil)
				if listErr != nil {
					slog.Debug("engine:models list_models failed", "engine", name, "err", listErr)
				} else if models, ok := extractStringsResult(listRaw, listSpec); ok {
					perEngine[i] = models
					listOK[i] = true
					// Digests are labelled with the run read before the request,
					// so a run that changed meanwhile never inherits them.
					digests[i] = observedDigests{runID: run, models: extractDigests(listRaw, listSpec), epoch: epoch}
				} else {
					slog.Debug("engine:models list_models returned an invalid inventory", "engine", name)
				}
			}
			if loadedSpec != nil {
				raw, err := listRaw, listErr
				if !shared {
					raw, err = e.Action(ctx, name, "loaded_models", nil)
				}
				if err != nil {
					loadedTransportFailed = true
					slog.Debug("engine:models loaded_models failed", "engine", name, "err", err)
				} else if models, ok := extractStringsResult(raw, loadedSpec); ok {
					loadedRaw[i] = raw
					// A successful query — even an empty result — records the key
					// so a consumer can tell "running, nothing loaded" apart from
					// "unknown". An action or shape error leaves loadedOK[i] false.
					loaded[i] = models
					loadedOK[i] = true
				} else {
					slog.Debug("engine:models loaded_models returned an invalid inventory", "engine", name)
				}
			}
		}(i, name, listSpec, loadedSpec, shared)
	}
	wg.Wait()

	fresh := make(map[string][]string)
	freshDigests := make(map[string]observedDigests)
	for i, ok := range listOK {
		if ok {
			fresh[engineNames[i]] = perEngine[i]
		}
		if digests[i].models != nil {
			freshDigests[engineNames[i]] = digests[i]
		}
	}
	e.storeInventory(fresh, freshDigests, sweep)
	var memoryEntries []performancewire.MemoryEvidenceEntry
	for i, name := range engineNames {
		if !loadedOK[i] {
			continue
		}
		entries := e.memoryEvidenceEntries(memoryCapture, name, loadedRaw[i])
		entries = memoryUnknownLoadedNames(memoryCapture, name, loaded[i], entries)
		memoryEntries = append(memoryEntries, entries...)
	}
	memoryEntries = e.finalizeMemoryEvidence(memoryCapture, memoryEntries)

	res := ModelsResult{Models: []string{}, MemoryEvidence: boundedMemoryEvidence(memoryEntries)}
	seen := map[string]bool{}
	for i, models := range perEngine {
		if listOK[i] {
			if res.ByEngine == nil {
				res.ByEngine = make(map[string][]string)
			}
			// Normalize nil -> [] so the wire distinguishes a successful empty
			// inventory from a missing/not-queryable engine.
			if models == nil {
				models = []string{}
			}
			res.ByEngine[engineNames[i]] = models
			for _, m := range models {
				if m != "" && !seen[m] {
					seen[m] = true
					res.Models = append(res.Models, m)
				}
			}
		}
		if loadedOK[i] {
			if res.LoadedByEngine == nil {
				res.LoadedByEngine = make(map[string][]string)
			}
			// Normalize nil -> [] so the wire carries an empty array ("queried,
			// nothing resident"), never null.
			ld := loaded[i]
			if ld == nil {
				ld = []string{}
			}
			res.LoadedByEngine[engineNames[i]] = ld
		}
	}
	return res
}

// extractStrings is the value-only wrapper used by focused extractor tests.
// Callers that must distinguish a valid empty inventory from an invalid
// response use extractStringsResult.
func extractStrings(raw json.RawMessage, spec *ActionResult) []string {
	out, ok := extractStringsResult(raw, spec)
	if !ok || len(out) == 0 {
		return nil
	}
	return out
}

// extractStringsResult pulls spec.Field from each element of the required
// top-level spec.Array. A present empty array is a successful, authoritative
// empty inventory. A missing, null, or wrong-typed array is unknown and returns
// ok=false so callers do not label it authoritative empty. Malformed rows
// inside a valid array are skipped, preserving additive response compatibility. For an
// unfiltered inventory, however, a non-empty array with no usable names is
// malformed rather than authoritative empty.
func extractStringsResult(raw json.RawMessage, spec *ActionResult) ([]string, bool) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, false
	}
	arrRaw, ok := obj[spec.Array]
	if !ok || bytes.Equal(bytes.TrimSpace(arrRaw), []byte("null")) {
		return nil, false
	}
	var arr []map[string]json.RawMessage
	if err := json.Unmarshal(arrRaw, &arr); err != nil {
		return nil, false
	}
	out := make([]string, 0)
	for _, el := range arr {
		if spec.Match != nil && !matchRow(el, spec.Match) {
			continue
		}
		fv, ok := el[spec.Field]
		if !ok {
			continue
		}
		var s string
		if err := json.Unmarshal(fv, &s); err == nil && s != "" {
			out = append(out, s)
		}
	}
	if spec.Match == nil && len(arr) > 0 && len(out) == 0 {
		return nil, false
	}
	return out, true
}

// matchRow reports whether an element passes an ActionResult row filter.
// Match.Field is a validated dot-separated object path. With Match.In set, the
// resolved value must decode as a JSON string equal to one of In. With
// Match.Nonempty set, it must decode as a JSON array with length > 0 (LM Studio
// /api/v1/models loaded_instances). A missing or wrong-typed path fails the
// match, so a row we cannot classify is excluded rather than counted as loaded.
func matchRow(el map[string]json.RawMessage, m *ResultMatch) bool {
	fv, ok := resolveObjectPath(el, m.Field)
	if !ok {
		return false
	}
	if m.Nonempty {
		var arr []json.RawMessage
		if err := json.Unmarshal(fv, &arr); err != nil {
			return false
		}
		return len(arr) > 0
	}
	var s string
	if err := json.Unmarshal(fv, &s); err != nil {
		return false
	}
	for _, want := range m.In {
		if s == want {
			return true
		}
	}
	return false
}

func resolveObjectPath(obj map[string]json.RawMessage, path string) (json.RawMessage, bool) {
	parts := strings.Split(path, ".")
	value, ok := obj[parts[0]]
	for _, part := range parts[1:] {
		if !ok {
			return nil, false
		}
		var nested map[string]json.RawMessage
		if err := json.Unmarshal(value, &nested); err != nil {
			return nil, false
		}
		value, ok = nested[part]
	}
	return value, ok
}

// sameReadRequest reports whether two manifest actions issue the identical
// side-effect-free HTTP request, so one response can serve both. LM Studio and
// llama.cpp answer list_models and loaded_models from the same endpoint and only
// filter the result differently; without this the loaded-model watcher fetched
// that endpoint twice on every tick.
func sameReadRequest(a, b Action) bool {
	if a.HTTP == nil || b.HTTP == nil {
		return false
	}
	if len(a.Cmd) > 0 || len(b.Cmd) > 0 || a.RemovePath != nil || b.RemovePath != nil {
		return false
	}
	if a.RestartAfter || b.RestartAfter || a.ProgressProtocol != "" || b.ProgressProtocol != "" || a.ModelResolution != "" || b.ModelResolution != "" {
		return false
	}
	return a.HTTP.Method == b.HTTP.Method &&
		a.HTTP.Path == b.HTTP.Path &&
		a.HTTP.ParamsIn == b.HTTP.ParamsIn &&
		bytes.Equal(a.HTTP.BodySchema, b.HTTP.BodySchema)
}

// cachedInventory returns a copy of the last installed inventory per engine.
func (e *Executor) cachedInventory() map[string][]string {
	e.inventoryMu.Lock()
	defer e.inventoryMu.Unlock()
	out := make(map[string][]string, len(e.inventory))
	for name, models := range e.inventory {
		out[name] = models
	}
	return out
}

// storeInventory replaces the cache with the engines answered this sweep. An
// engine missing from the sweep (stopped, or its query failed) loses its entry,
// so a restarted engine is always queried afresh rather than served stale. The
// installed lists and content digests are both replaced only when no sweep that
// began later has stored its own: an older listing never replaces a newer one.
func (e *Executor) storeInventory(fresh map[string][]string, digests map[string]observedDigests, sweep uint64) {
	e.inventoryMu.Lock()
	defer e.inventoryMu.Unlock()
	if sweep > e.digestSweep {
		e.inventory = fresh
		e.digestSweep = sweep
		if e.inventoryDigests == nil {
			e.inventoryDigests = make(map[string]observedDigests)
		}
		for name, current := range e.inventoryDigests {
			current.models = nil
			current.runID = ""
			e.inventoryDigests[name] = current
		}
		for name, observed := range digests {
			current := e.inventoryDigests[name]
			if !current.suppressed && observed.epoch == current.epoch {
				e.inventoryDigests[name] = observed
			}
		}
	}
}
