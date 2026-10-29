// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
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
// DownloadState is the current target-owned lifecycle, captured with the ready
// inventory mask. This producer always supplies it; nil means unsupported when
// reading an older or malformed extension. It never establishes residency.
type ModelsResult struct {
	Models         []string                        `json:"models"`
	ByEngine       map[string][]string             `json:"modelsByEngine,omitempty"`
	LoadedByEngine map[string][]string             `json:"loadedByEngine,omitempty"`
	Policy         *nodepolicy.Advertisement       `json:"policy,omitempty"`
	MemoryEvidence *performancewire.MemoryEvidence `json:"memoryEvidence,omitempty"`
	DownloadState  *modeldownload.Snapshot         `json:"downloadState,omitempty"`

	// Local observation fences survive until the notification's publication.
	observations map[string]observedDigests
	sweep        uint64
}

// UnmarshalJSON rejects malformed download readiness without discarding real
// residency. Optional memory evidence remains independent of content inventory.
func (m *ModelsResult) UnmarshalJSON(data []byte) error {
	type inventory ModelsResult
	next := inventory(*m)
	next.MemoryEvidence = nil
	next.DownloadState = nil
	next.observations, next.sweep = nil, 0
	wire := struct {
		*inventory
		MemoryEvidence json.RawMessage `json:"memoryEvidence"`
		DownloadState  json.RawMessage `json:"downloadState"`
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
	if len(wire.DownloadState) != 0 && !bytes.Equal(bytes.TrimSpace(wire.DownloadState), []byte("null")) {
		snapshot, err := modeldownload.Parse(wire.DownloadState)
		if err != nil {
			// An invalid extension cannot certify content from its accompanying
			// names. Residency remains an independent observation.
			next.Models, next.ByEngine = []string{}, nil
		} else {
			next.DownloadState = &snapshot
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
	res.observations, res.sweep = nil, 0
	return res
}

// inventoryReconcileInterval is how often the loaded-model watcher refreshes
// each engine's installed inventory. Residency (what is loaded) changes on its
// own — JIT loads, TTL evictions — so the watcher checks it every tick, but the
// installed list only changes when someone pulls or deletes a model. PAIR's own
// pulls and deletes poke the watcher, which forces a full sweep; this interval
// only bounds how long a change made outside PAIR takes to show up.
const inventoryReconcileInterval = 60 * time.Second

type inventoryMembership int

const (
	inventoryFailed inventoryMembership = iota
	inventoryAbsent
	inventoryAmbiguous
	inventoryUnready
	inventoryPresent
)

// engineInventory is a single fresh native observation, not the sweep cache.
type engineInventory struct {
	engine, runID, launchDigest, modelsDir, naming string
	epoch                                          uint64
	raw                                            json.RawMessage
	err                                            error
	models, candidates, ready                      []string
	digests                                        observedDigests
	ok                                             bool
}

func (r engineInventory) membership(model string) inventoryMembership {
	if !r.ok || nodepolicy.ValidateModelIdentifier(model) != nil {
		return inventoryFailed
	}
	key := engines.NormalizeModelName(model, r.naming)
	count := 0
	for _, name := range r.candidates {
		if engines.NormalizeModelName(name, r.naming) == key {
			count++
		}
	}
	if count == 0 {
		return inventoryAbsent
	}
	if count != 1 {
		return inventoryAmbiguous
	}
	for _, name := range r.ready {
		if engines.NormalizeModelName(name, r.naming) == key {
			return inventoryPresent
		}
	}
	return inventoryUnready
}

func (e *Executor) readEngineInventory(ctx context.Context, name string) engineInventory {
	r := engineInventory{engine: name}
	mf, ok := e.reg.Get(name)
	if !ok {
		return r
	}
	act, ok := mf.Actions["list_models"]
	if !ok || act.Result == nil {
		return r
	}
	st, err := e.state(name)
	if err != nil {
		return r
	}
	st.mu.Lock()
	if !st.running {
		st.mu.Unlock()
		return r
	}
	r.runID, r.launchDigest, r.modelsDir = st.runID, st.launchDigest, st.modelsDir
	st.mu.Unlock()
	r.naming = mf.ModelNamingOrDefault()
	e.inventoryMu.Lock()
	r.epoch = e.inventoryDigests[name].epoch
	e.inventoryMu.Unlock()
	r.raw, r.err = e.Action(ctx, name, "list_models", nil)
	if r.err != nil {
		return r
	}
	r.candidates, ok = extractStringsResult(r.raw, act.Result)
	if !ok {
		return r
	}
	r.models, ok = extractReadyStringsResult(r.raw, act.Result, mf)
	if !ok {
		return r
	}
	r.digests = observedDigests{runID: r.runID, models: extractDigests(r.raw, act.Result), epoch: r.epoch}
	listed := make(map[string]bool, len(r.models))
	for _, model := range r.models {
		listed[model] = true
	}
	for model := range r.digests.models {
		if !listed[model] {
			delete(r.digests.models, model)
		}
	}
	e.inventoryMu.Lock()
	current := e.inventoryObservationCurrent(name, r.digests)
	if current && e.downloads != nil {
		e.downloads.mu.Lock()
		r.ready = e.readyModelNamesLocked(name, r.models, r.naming)
		e.downloads.mu.Unlock()
	}
	e.inventoryMu.Unlock()
	st.mu.Lock()
	current = current && st.running && st.runID == r.runID && st.launchDigest == r.launchDigest && st.modelsDir == r.modelsDir
	st.mu.Unlock()
	r.ok = current && e.downloads != nil
	return r
}

// modelsSweep is ModelsResult with control over the installed inventory. With
// fullInventory false, an engine whose list_models is a separate request from
// loaded_models reuses its last inventory instead of querying it again; an
// engine with no cached inventory, or whose two actions share one request, is
// still queried. Every successful inventory query refreshes the cache.
func (e *Executor) modelsSweep(ctx context.Context, fullInventory bool) ModelsResult {
	ctx, cancel := context.WithTimeout(ctx, modelsTimeout)
	defer cancel()
	sweep := e.sweeps.Add(1)
	cached, cachedObservations := e.cachedInventoryObservation()
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
				servingState.mu.Lock()
				capture := servingState.bufferCapture
				servingState.mu.Unlock()
				bufferBoundary := capture.captureBoundary()
				run, runErr := snapshotLlamaObservation(servingState)
				defer func() {
					sharedProps := map[string]modelBufferProps{}
					if runErr == nil {
						e.observeLlamaServing(ctx, servingState, run, sweep, loaded[i], loadedOK[i], loadedRaw[i], loadedTransportFailed, sharedProps)
					}
					e.observeModelBufferSweep(ctx, servingState, capture, sweep, loaded[i], loadedOK[i] && !loadedTransportFailed, loadedRaw[i], sharedProps, bufferBoundary)
				}()
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
				digests[i] = cachedObservations[name]
			} else if listSpec != nil {
				observation := e.readEngineInventory(ctx, name)
				listRaw, listErr = observation.raw, observation.err
				if listErr != nil {
					category, status := inventoryFailureCategory(listErr)
					slog.Debug("engine:models list_models failed", "engine", name, "action", "list_models", "category", category, "status", status)
				} else if observation.ok {
					perEngine[i] = observation.models
					listOK[i] = true
					digests[i] = observation.digests
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
					category, status := inventoryFailureCategory(err)
					slog.Debug("engine:models loaded_models failed", "engine", name, "action", "loaded_models", "category", category, "status", status)
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

	res := ModelsResult{ByEngine: fresh, MemoryEvidence: boundedMemoryEvidence(memoryEntries), observations: freshDigests, sweep: sweep}
	for i := range perEngine {
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
	return e.currentModelsResult(res)
}

// currentModelsResult fences the inventory and captures its readiness mask and
// lifecycle together. The gate is innermost: no I/O, notify, or owner acquisition
// occurs while it is held. LoadedByEngine is deliberately never masked.
func (e *Executor) currentModelsResult(res ModelsResult) ModelsResult {
	engineNames, naming := e.inventoryModelNaming()
	e.inventoryMu.Lock()
	defer e.inventoryMu.Unlock()
	for name := range res.ByEngine {
		if !e.inventoryObservationCurrent(name, res.observations[name]) {
			delete(res.ByEngine, name)
			continue
		}
		if res.sweep < e.digestSweep {
			// A later accepted sweep owns the inventory, but completing an
			// earlier same-run read does not make that usable inventory unknown.
			models, ok := e.inventory[name]
			observed := e.inventoryDigests[name]
			if !ok || !e.inventoryObservationCurrent(name, observed) {
				delete(res.ByEngine, name)
				continue
			}
			res.ByEngine[name] = append([]string{}, models...)
			res.observations[name] = observed
		}
	}
	res.Models = []string{}
	if e.downloads == nil {
		res.ByEngine = nil
		res.DownloadState = nil
		return res
	}
	e.downloads.mu.Lock()
	defer e.downloads.mu.Unlock()
	snapshot := e.downloads.snapshotLocked()
	res.DownloadState = &snapshot
	seen := make(map[string]bool)
	for _, name := range engineNames {
		models, ok := res.ByEngine[name]
		if !ok {
			continue
		}
		models = e.readyModelNamesLocked(name, models, naming[name])
		res.ByEngine[name] = models
		for _, model := range models {
			if !seen[model] {
				seen[model] = true
				res.Models = append(res.Models, model)
			}
		}
	}
	return res
}

// The named SSE adapter's router can expose a first shard or a transient
// downloaded status after either success or failure. Only stable producer
// statuses contribute ready names; repeated identities are ambiguous.
func extractReadyStringsResult(raw json.RawMessage, spec *ActionResult, mf *Manifest) ([]string, bool) {
	if mf.Actions["pull_model"].ProgressProtocol != pullProgressProtocolLlamaCPPModelsSSE {
		return extractStringsResult(raw, spec)
	}
	if _, ok := extractStringsResult(raw, spec); !ok {
		return nil, false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, false
	}
	var rows []map[string]json.RawMessage
	if err := json.Unmarshal(obj[spec.Array], &rows); err != nil {
		return nil, false
	}
	counts := make(map[string]int)
	for _, row := range rows {
		var model string
		if json.Unmarshal(row[spec.Field], &model) == nil {
			counts[model]++
		}
	}
	ready := make([]string, 0, len(rows))
	stable := &ResultMatch{Field: "status.value", In: []string{"unloaded", "loading", "loaded", "sleeping"}}
	for _, row := range rows {
		var model string
		if json.Unmarshal(row[spec.Field], &model) != nil || counts[model] != 1 ||
			nodepolicy.ValidateModelIdentifierJSON(row[spec.Field]) != nil ||
			!matchRow(row, stable) || spec.Match != nil && !matchRow(row, spec.Match) {
			continue
		}
		ready = append(ready, model)
	}
	return ready, true
}

// Inventory failures can retain response bodies, credentials or transport URLs.
// Classify without formatting arbitrary errors; the action's public error stays
// unchanged. Zero status means no HTTP response status was available.
func inventoryFailureCategory(err error) (string, int) {
	var response *ActionHTTPError
	if errors.As(err, &response) {
		return "http-status", response.StatusCode
	}
	return "action-failed", 0
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
// unfiltered inventory, however, a non-empty array with no string fields is
// malformed rather than authoritative empty. Strings rejected by identifier
// validation are filtered entries, so an all-rejected list is known-empty.
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
	haveString := false
	for _, el := range arr {
		if spec.Match != nil && !matchRow(el, spec.Match) {
			continue
		}
		fv, ok := el[spec.Field]
		if !ok {
			continue
		}
		var s string
		if err := json.Unmarshal(fv, &s); err == nil && !bytes.Equal(bytes.TrimSpace(fv), []byte("null")) {
			haveString = true
			if nodepolicy.ValidateModelIdentifierJSON(fv) != nil {
				continue
			}
			out = append(out, s)
		}
	}
	if spec.Match == nil && len(arr) > 0 && !haveString {
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
	models, _ := e.cachedInventoryObservation()
	return models
}

func (e *Executor) cachedInventoryObservation() (map[string][]string, map[string]observedDigests) {
	_, naming := e.inventoryModelNaming()
	e.inventoryMu.Lock()
	defer e.inventoryMu.Unlock()
	out := make(map[string][]string, len(e.inventory))
	observations := make(map[string]observedDigests, len(e.inventory))
	for name, models := range e.inventory {
		observed := e.inventoryDigests[name]
		if e.inventoryObservationCurrent(name, observed) {
			out[name] = validModelIdentifiers(models)
			observations[name] = observed
		}
	}
	if e.downloads == nil {
		return nil, nil
	}
	e.downloads.mu.Lock()
	defer e.downloads.mu.Unlock()
	for name, models := range out {
		out[name] = e.readyModelNamesLocked(name, models, naming[name])
	}
	return out, observations
}

// inventoryObservationCurrent is called with inventoryMu held and without the
// download gate. Lifecycle percentage revisions are intentionally irrelevant.
func (e *Executor) inventoryObservationCurrent(engine string, observed observedDigests) bool {
	current := e.inventoryDigests[engine]
	return !current.suppressed && observed.epoch == current.epoch && observed.runID == e.engineRunID(engine)
}

func (e *Executor) inventoryModelNaming() ([]string, map[string]string) {
	names := e.reg.Names()
	naming := make(map[string]string, len(names))
	for _, name := range names {
		if manifest, ok := e.reg.Get(name); ok {
			naming[name] = manifest.ModelNamingOrDefault()
		}
	}
	return names, naming
}

func (e *Executor) modelReadyLocked(engine, model, naming string) bool {
	if e.downloads == nil {
		return false
	}
	key, err := downloadKey(engine, model, naming)
	return err == nil && !e.downloads.blockedLocked(key)
}

func (e *Executor) readyModelNamesLocked(engine string, models []string, naming string) []string {
	ready := make([]string, 0, len(models))
	for _, model := range models {
		if e.modelReadyLocked(engine, model, naming) {
			ready = append(ready, model)
		}
	}
	return ready
}

func validModelIdentifiers(models []string) []string {
	out := make([]string, 0, len(models))
	for _, model := range models {
		if nodepolicy.ValidateModelIdentifier(model) == nil {
			out = append(out, model)
		}
	}
	return out
}

// storeInventory replaces the cache with the engines answered this sweep. An
// engine missing from the sweep (stopped, or its query failed) loses its entry,
// so a restarted engine is always queried afresh rather than served stale. The
// installed lists and content digests are both replaced only when no sweep that
// began later has stored its own: an older listing never replaces a newer one.
func (e *Executor) storeInventory(fresh map[string][]string, digests map[string]observedDigests, sweep uint64) {
	_, naming := e.inventoryModelNaming()
	e.inventoryMu.Lock()
	defer e.inventoryMu.Unlock()
	for name, observed := range digests {
		if !e.inventoryObservationCurrent(name, observed) {
			delete(fresh, name)
			delete(digests, name)
		}
	}
	for name := range fresh {
		if _, ok := digests[name]; !ok {
			delete(fresh, name)
		}
	}
	if sweep <= e.digestSweep {
		return
	}
	if e.downloads != nil {
		e.downloads.mu.Lock()
		defer e.downloads.mu.Unlock()
	}
	for name, models := range fresh {
		fresh[name] = e.readyModelNamesLocked(name, models, naming[name])
	}
	e.inventory = make(map[string][]string, len(fresh))
	for name, models := range fresh {
		e.inventory[name] = append([]string{}, models...)
	}
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
		ready := make(map[string]string, len(observed.models))
		for model, digest := range observed.models {
			if e.modelReadyLocked(name, model, naming[name]) {
				ready[model] = digest
			}
		}
		observed.models = ready
		e.inventoryDigests[name] = observed
	}
}
