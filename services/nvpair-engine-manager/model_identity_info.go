// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/gguffacts"
)

const modelIdentityRefreshInterval = 10 * time.Minute

// A failed resolution is not retried sooner unless the run or configuration
// changes or a refresh arrives.
const modelIdentityFailureRetry = time.Minute

type modelIdentityInfoEntry struct {
	identity      engines.ModelIdentity
	binding, run  string
	checked       time.Time
	configuration modelInspectionSnapshot
	failure       string // set when resolution failed; nothing is published
	source        *sourceSnapshot
}

// The run goroutine owns entries and pass. snapshot only copies the published
// values; refresh coalesces requests and never waits for file I/O.
type modelIdentityInfo struct {
	executor *Executor
	poke     chan struct{}
	mu       sync.Mutex
	values   map[string]map[string]engines.ModelIdentity
	entries  map[string]map[string]modelIdentityInfoEntry
	pass     context.CancelFunc
}

func newModelIdentityInfo(e *Executor) *modelIdentityInfo {
	return &modelIdentityInfo{executor: e, poke: make(chan struct{}, 1), entries: map[string]map[string]modelIdentityInfoEntry{}}
}

func (m *modelIdentityInfo) snapshot() map[string]map[string]engines.ModelIdentity {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]map[string]engines.ModelIdentity, len(m.values))
	for engine, models := range m.values {
		out[engine] = make(map[string]engines.ModelIdentity, len(models))
		for name, identity := range models {
			identity.Files = slices.Clone(identity.Files)
			out[engine][name] = identity
		}
	}
	return out
}

func (m *modelIdentityInfo) refresh() {
	select {
	case m.poke <- struct{}{}:
	default:
	}
}

func (m *modelIdentityInfo) run(ctx context.Context) {
	ticker := time.NewTicker(defaultLoadedPollSeconds * time.Second)
	defer ticker.Stop()
	force := false
	for {
		if ctx.Err() != nil {
			return
		}
		// A refresh that a requested read interrupted stays pending.
		force = !m.reconcile(ctx, force, time.Now()) && force
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		case <-m.poke:
			force = true
		}
	}
}

// reconcile reports whether the pass finished without yielding to a request.
func (m *modelIdentityInfo) reconcile(ctx context.Context, force bool, now time.Time) bool {
	e := m.executor
	// A requested inspection or preview cancels the pass through the shared
	// gate; models the pass has not finished keep their previous entries.
	pass, cancel := context.WithCancel(ctx)
	m.pass = cancel
	defer func() { cancel(); m.pass = nil }()
	e.inventoryMu.Lock()
	inventory := make(map[string][]string, len(e.inventory))
	for engine, models := range e.inventory {
		inventory[engine] = slices.Clone(models)
	}
	e.inventoryMu.Unlock()
	next := map[string]map[string]modelIdentityInfoEntry{}
	for engine, models := range inventory {
		if ctx.Err() != nil {
			return false
		}
		manifest, ok := e.reg.Get(engine)
		if !ok || manifest.FamilyOrDefault() != engines.NameLlamaCPP {
			continue
		}
		next[engine] = map[string]modelIdentityInfoEntry{}
		for _, model := range models {
			previous, exists := m.entries[engine][model]
			entry, keep := m.resolve(pass, engine, model, previous, exists, force, now)
			if keep {
				next[engine][model] = entry
			}
			if ctx.Err() != nil {
				return false
			}
		}
	}
	m.entries = next
	values := map[string]map[string]engines.ModelIdentity{}
	for engine, entries := range next {
		values[engine] = map[string]engines.ModelIdentity{}
		for model, entry := range entries {
			if entry.failure == "" {
				values[engine][model] = entry.identity
			}
		}
	}
	m.mu.Lock()
	m.values = values
	m.mu.Unlock()
	return pass.Err() == nil
}

func (m *modelIdentityInfo) resolve(parent context.Context, engine, model string, old modelIdentityInfoEntry, exists, force bool, now time.Time) (modelIdentityInfoEntry, bool) {
	ctx, cancel := context.WithTimeout(parent, modelInspectionBudget)
	defer cancel()
	var value modelIdentityInfoEntry
	var keep bool
	if err := retryModelRead(ctx, func(retry *bool) error {
		value, keep = m.resolveOnce(ctx, engine, model, old, exists, force, now, retry)
		return nil
	}); err != nil {
		return old, exists
	}
	return value, keep
}

func (m *modelIdentityInfo) resolveOnce(parent context.Context, engine, model string, old modelIdentityInfoEntry, exists, force bool, now time.Time, retry *bool) (modelIdentityInfoEntry, bool) {
	e := m.executor
	if !e.modelInspection.claimBackground(m.pass) {
		*retry = true
		return old, exists
	}
	defer e.modelInspection.release()
	start := time.Now()
	ctx, cancel := context.WithTimeout(parent, modelInspectionBudget)
	defer cancel()
	if e.downloads == nil {
		return modelIdentityInfoEntry{}, false
	}
	st, err := e.state(engine)
	if err != nil {
		return modelIdentityInfoEntry{}, false
	}
	// Serialize passive scans with mutations so their read leases cannot
	// turn an otherwise admissible content mutation into a busy refusal.
	select {
	case st.contentMu <- struct{}{}:
	default:
		*retry = true
		return old, exists
	}
	defer func() { <-st.contentMu }()
	downloads := e.downloads
	key := downloadContentKey{engine: engine}
	release, err := downloads.acquireRead(key, false)
	if err != nil {
		return old, exists
	}
	defer release()
	downloads.mu.Lock()
	source, revision := downloads.source, downloads.revision
	downloads.mu.Unlock()
	run := e.engineRunID(engine)
	snapshot, err := inspectionSnapshot(st, engine)
	if errors.Is(err, errModelInspectionBusy) {
		*retry = true
		return old, exists
	}
	if err != nil {
		return modelIdentityInfoEntry{}, false
	}
	interval := modelIdentityRefreshInterval
	if old.failure != "" {
		interval = modelIdentityFailureRetry
	}
	// A resolved entry also rechecks its file and preset evidence, which costs
	// metadata calls only, so a file replaced in place is not served stale.
	if exists && !force && run == old.run && snapshot.equal(old.configuration) && !now.Before(old.checked) && now.Sub(old.checked) < interval &&
		(old.failure != "" || validateResolutionEvidence(ctx, old.source) == nil) {
		return old, true
	}
	// Remember a failure for this run and configuration so that the next pass
	// does not repeat it while the gate is wanted elsewhere.
	fail := func(reason string) (modelIdentityInfoEntry, bool) {
		if errors.Is(parent.Err(), context.Canceled) {
			return old, exists // Yielded or stopped; not a resolution failure.
		}
		level := slog.LevelInfo
		if old.failure == reason {
			level = slog.LevelDebug
		}
		slog.Log(parent, level, "model identity resolution failed", "engine", engine, "model", model, "reason", reason, "duration", time.Since(start).Round(time.Millisecond), "retry_after", modelIdentityFailureRetry)
		return modelIdentityInfoEntry{run: run, checked: now, configuration: snapshot, failure: reason}, true
	}
	access := servingFileAccess{open: e.modelInspection.open}
	access.recognize = func(ctx context.Context, model string) (engineInventory, error) {
		observation, err := e.recognizeModelSource(ctx, engine, model)
		if err == nil {
			err = e.nativeSourceCurrent(ctx, st, snapshot.store, observation)
		}
		return observation, err
	}
	list, _, err := resolveLlamaArtifacts(ctx, snapshot.state, snapshot.command, snapshot.store, model, access)
	if err != nil {
		var field *ServingFieldError
		switch {
		case errors.As(err, &field):
			return fail(field.Reason)
		case ctx.Err() != nil:
			return fail("inspection budget exceeded")
		}
		return fail("artifacts unavailable")
	}
	binding := exportBinding(list, nil)
	identity := old.identity
	perSlot := modelIdentityConfiguredContext(snapshot, model, list)
	if !exists || old.failure != "" || binding != old.binding || run != old.run || !snapshot.equal(old.configuration) || perSlot != old.identity.ConfiguredContextTokens {
		var valid bool
		identity, valid = e.modelIdentityFromArtifactsContext(ctx, engine, model, list, perSlot, access)
		if !valid && ctx.Err() != nil {
			return fail("inspection budget exceeded")
		}
		if !valid {
			return fail("identity metadata invalid")
		}
	}
	current, err := inspectionSnapshot(st, engine)
	if err != nil || !snapshot.equal(current) || e.engineRunID(engine) != run {
		*retry = true
		return old, exists
	}
	if validateResolutionEvidence(ctx, list.source) != nil {
		if ctx.Err() != nil {
			return fail("inspection budget exceeded")
		}
		return fail("artifact evidence changed")
	}
	downloads.mu.Lock()
	changed := e.downloads != downloads || downloads.source != source || downloads.revision != revision || downloads.blockedLocked(key) || downloads.writers[engine]
	downloads.mu.Unlock()
	if changed || ctx.Err() != nil {
		return old, exists
	}
	return modelIdentityInfoEntry{identity: identity, binding: binding, run: run, checked: now, configuration: snapshot, source: list.source}, true
}

// Read exactly the selector bytes already checked by artifact resolution.
func modelIdentityConfiguredContext(snapshot modelInspectionSnapshot, model string, list modelFileList) int64 {
	rt := snapshot.state.plat.Runtime
	var configured int64
	found := false
	for _, load := range rt.ModelLoads {
		if load.ModelID == model {
			if load.Resolved == nil {
				return 0
			}
			context := int64(load.Resolved.PerSlotContext)
			if found && configured != context {
				return 0 // Artifact metadata cannot attribute one sibling's context.
			}
			configured, found = context, true
		}
	}
	if found {
		return configured
	}
	cli, env, err := llamaArtifactLaunchOptions(snapshot.command, rt.EditableLaunch)
	if err != nil {
		return 0
	}
	options := map[string]string{}
	for key, value := range env {
		options[key] = value
	}
	preset := env["models-preset"]
	if value, ok := cli["models-preset"]; ok {
		preset = value
	}
	if preset != "" && list.source != nil {
		for _, selector := range list.source.selectors {
			if !configuredAs(selector.file.configured, preset) {
				continue
			}
			sections, err := readLlamaArtifactPreset(selector.bytes)
			if err != nil || sections["*"].err != nil || sections[model].err != nil {
				return 0
			}
			for _, values := range []map[string]string{sections["*"].options, sections[model].options} {
				for key, value := range values {
					options[key] = value
				}
			}
		}
	}
	for key, value := range cli {
		options[key] = value
	}
	context, err := llamaPositive(options["ctx-size"])
	if err != nil {
		return 0
	}
	parallel := int64(1)
	if value, ok := options["parallel"]; ok {
		parallel, err = llamaPositive(value)
		if err != nil {
			return 0
		}
	}
	return context / parallel
}

var modelIdentityQuant = regexp.MustCompile(`(?i)[-_.]((?:UD-)?(?:I?Q[0-9]_[A-Z0-9_]+|Q[0-9]_[0-9]|TQ[12]_0|MXFP4(?:_MOE)?|BF16|F16|F32))(?:-[0-9]{5}-of-[0-9]{5})?\.gguf$`)
var modelIdentityHex = regexp.MustCompile(`^[0-9a-f]{64}$`)
var modelIdentityRevision = regexp.MustCompile(`^[0-9a-f]{40}$`)
var modelIdentityRepoURL = regexp.MustCompile(`^https://huggingface\.co/([^/]+)/([^/]+)/?$`)
var modelIdentityFileTypes = map[uint32]string{0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}

func (e *Executor) modelIdentityFromArtifacts(engine, modelID string, list modelFileList, perSlotContext int64) (engines.ModelIdentity, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), modelInspectionBudget)
	defer cancel()
	return e.modelIdentityFromArtifactsContext(ctx, engine, modelID, list, perSlotContext, servingFileAccess{})
}

func (e *Executor) modelIdentityFromArtifactsContext(ctx context.Context, engine, modelID string, list modelFileList, perSlotContext int64, access servingFileAccess) (engines.ModelIdentity, bool) {
	identity := engines.ModelIdentity{Strength: "declared", ConfiguredContextTokens: perSlotContext}
	var items []contentArtifact
	allDigests := true
	repo, tag, repoErr := splitLlamaModel(modelID)
	if repoErr == nil && tag != "" {
		identity.Quantization, identity.QuantizationSource = tag, "hf-tag"
	}
	for _, file := range list.Files {
		name := filepath.Base(filepath.FromSlash(file.Path))
		digest := file.SHA256
		if !modelIdentityHex.MatchString(digest) {
			digest = ""
			if filepath.Base(filepath.Dir(file.abs)) == "blobs" && modelIdentityHex.MatchString(filepath.Base(file.abs)) {
				digest = filepath.Base(file.abs)
			}
		}
		identity.Files = append(identity.Files, engines.ModelIdentityFile{Role: file.Role, Name: name, SizeBytes: file.Size, SHA256: digest})
		if file.Role != "model" {
			continue
		}
		if file.Size < 0 || identity.TotalBytes > math.MaxInt64-file.Size {
			return engines.ModelIdentity{}, false
		}
		identity.TotalBytes += file.Size
		items = append(items, contentArtifact{file.Role, file.Size, digest})
		allDigests = allDigests && digest != ""
		if identity.Quantization == "" {
			if match := modelIdentityQuant.FindStringSubmatch(name); match != nil {
				identity.Quantization, identity.QuantizationSource = strings.ToUpper(match[1]), "file-name"
			}
		}
		if identity.Repository == "" {
			parts := strings.Split(filepath.ToSlash(file.configured), "/")
			for i := 0; i+2 < len(parts); i++ {
				if !strings.HasPrefix(parts[i], "models--") || parts[i+1] != "snapshots" || !modelIdentityRevision.MatchString(strings.ToLower(parts[i+2])) {
					continue
				}
				owner, name, ok := strings.Cut(strings.TrimPrefix(parts[i], "models--"), "--")
				if ok {
					identity.Repository, identity.Revision = owner+"/"+name, strings.ToLower(parts[i+2])
					if repoErr == nil && filepath.Base(llamaRepoDir("", repo)) == parts[i] {
						identity.Repository = repo
					}
					break
				}
			}
		}
	}
	if repoErr == nil && list.Layout != llamaArtifactsLayout && list.Revision != "" {
		identity.Repository, identity.Revision = repo, strings.ToLower(list.Revision)
	}
	facts, err := decodeInspectionFacts(ctx, list, access)
	if err == nil {
		projected, _ := projectInspectionFacts(facts)
		identity.Architecture = projected.Architecture
		if projected.ContextTokens != nil {
			identity.TrainedContextTokens = int64(*projected.ContextTokens)
		}
		for _, shard := range facts.Shards {
			if shard.Role != gguffacts.RoleMain {
				continue
			}
			identity.GeneralName, identity.SizeLabel = shard.General.Name, shard.General.SizeLabel
			if identity.Repository == "" {
				if match := modelIdentityRepoURL.FindStringSubmatch(shard.General.RepoURL); match != nil {
					identity.Repository = match[1] + "/" + match[2]
				}
			}
			if identity.Quantization == "" && shard.General.FileType != nil {
				identity.Quantization = modelIdentityFileTypes[*shard.General.FileType]
				if identity.Quantization != "" {
					identity.QuantizationSource = "gguf-file-type"
				}
			}
			break
		}
	}
	if allDigests && len(items) != 0 {
		identity.ContentID, identity.Strength = contentDigestOf(items), "verified"
	}
	return identity, identity.Validate() == nil && ctx.Err() == nil
}
