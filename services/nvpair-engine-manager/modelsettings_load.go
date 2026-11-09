// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/modelconfig"
	"nvpair-shared/nodepolicy"
)

const externalConversionResolver = "external-conversion/llama"

func modelLoadNativeFit(requested modelconfig.RequestedLoad) bool {
	return requested.Fit != nil && *requested.Fit == "on"
}

func modelLoadUnpriced(requested modelconfig.RequestedLoad) bool {
	return modelLoadNativeFit(requested) || len(modelLoadGPUSelection(requested)) > 0
}

func modelLoadUnpricedReason(requested modelconfig.RequestedLoad) string {
	if modelLoadNativeFit(requested) {
		return "native-fit"
	}
	return "gpu-selection"
}

func modelLoadAllowedPresetKeys(loads []modelconfig.ModelLoad) map[string]map[string]bool {
	allowed := map[string]map[string]bool{}
	for _, load := range loads {
		allowed[load.ProfileKey] = map[string]bool{}
		if modelLoadNativeFit(load.Requested) {
			allowed[load.ProfileKey] = map[string]bool{"fit": true, "fit-target": true, "fit-ctx": true}
		}
		for key := range load.Requested.ExtraOptions {
			allowed[load.ProfileKey][key] = true
		}
	}
	return allowed
}

// This only selects conversion fields. The preset parser still checks section
// identity and syntax, then convertTypedLlamaOptions validates merged values.
func modelConversionAllowedKeys(data []byte) map[string]map[string]bool {
	allowed := map[string]map[string]bool{}
	section := ""
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = line[1 : len(line)-1]
			allowed[section] = map[string]bool{}
		} else if name, _, ok := strings.Cut(line, "="); ok && allowed[section] != nil {
			name = strings.TrimSpace(name)
			key := llamaPresetOptionKey(name)
			if key == "" {
				allowed[section][name] = true
			} else if key == "fit" || key == "fit-target" || key == "fit-ctx" || key == "models-preset" || key == "models-max" || key == "sleep-idle-seconds" || slices.ContainsFunc(llamaServingOptions, func(option llamaServingOption) bool { return option.key == key && option.unsupported }) {
				allowed[section][key] = true
			}
		}
	}
	return allowed
}

func modelLoadError(model, field, reason string) error {
	return &ServingFieldError{Field: "modelLoads." + field, ModelID: model, Reason: reason}
}

// Descriptor ownership and the integrated serving table remain the native
// vocabulary. Raw options cannot replace typed or manager-owned controls.
func validateModelExtraOptions(model string, extras map[string]string) error {
	if len(extras) == 0 {
		return nil
	}
	if err := (modelconfig.RequestedLoad{ExtraOptions: extras}).Validate(); err != nil {
		return modelLoadError(model, "extraOptions", err.Error())
	}
	matrix, err := llamaModelSettingsOptions()
	if err != nil {
		return err
	}
	for _, key := range sortedKeys(extras) {
		if key == "version" || key == "sleep-idle-seconds" {
			return modelLoadError(model, "extraOptions", "PAIR owns "+key+".")
		}
		if canonical := llamaPresetOptionKey(key); canonical != "" && canonical != key {
			return modelLoadError(model, "extraOptions", "Use the canonical key "+canonical+" for "+key+".")
		}
		if err := validateLlamaPresetValue(key, extras[key], model); err != nil {
			return modelLoadError(model, "extraOptions", key+": "+err.Error())
		}
		for _, option := range matrix {
			if option.Native.key != key {
				continue
			}
			d := option.Descriptor
			reason := ""
			switch {
			case d.Owner == "manager/router" || d.Owner == "artifacts":
				reason = "PAIR owns " + key + "."
			case d.Supported && !strings.HasPrefix(d.Field, "native."):
				reason = "Use the typed field for " + key + "."
			case option.Native.unsupported && !option.Native.unpricedExtra:
				reason = key + " changes allocation and is not accepted."
			}
			if reason != "" {
				return modelLoadError(model, "extraOptions", reason)
			}
			if err := validateLlamaValue(key, extras[key]); err != nil {
				return modelLoadError(model, "extraOptions", key+": "+err.Error())
			}
		}
		// Thread controls have typed load owners. Sampling controls outside
		// the allocation table remain native defaults, below request overrides.
		if key == "threads" || key == "threads-batch" {
			return modelLoadError(model, "extraOptions", "Use the typed field for "+key+".")
		}
		if key == "kv-unified" || key == "kv-unified-per-slot" {
			return modelLoadError(model, "extraOptions", "PAIR owns "+key+".")
		}
	}
	return nil
}

// The existing engine-wide read covers aliases sharing the same artifacts.
// Source replacement during a read is a refusal, even if sampled files agree.
func (e *Executor) modelLoadContentRead(engine string) (func(), func() error, error) {
	if e.downloads == nil || e.reg == nil {
		return nil, nil, modelLoadError("", "artifacts", "Model content admission is unavailable.")
	}
	downloads, registry := e.downloads, e.reg
	key := downloadContentKey{engine: engine}
	release, err := downloads.acquireRead(key, false)
	if err != nil {
		return nil, nil, modelLoadError("", "artifacts", "Model content is busy or unsafe.")
	}
	downloads.mu.Lock()
	source, revision := downloads.source, downloads.revision
	downloads.mu.Unlock()
	check := func() error {
		downloads.mu.Lock()
		changed := e.downloads != downloads || e.reg != registry || downloads.source != source || downloads.revision != revision || downloads.blockedLocked(key) || downloads.writers[engine]
		downloads.mu.Unlock()
		if changed {
			return modelLoadError("", "artifacts", "Model content source changed during preparation.")
		}
		return nil
	}
	return release, check, nil
}

// Typed presets share the serving generation owner, but do not declare serving
// bounds or manufacture content hashes, memory assertions or qualification.
func compileTypedLlama(st *engineState, command launchCommand, rt Runtime, approved bool, advisory ...*map[string]modelEstimateInput) (*llamaServingLaunch, error) {
	return compileTypedLlamaExpected(st, command, rt, approved, servingFileAccess{}, nil, advisory...)
}

func compileTypedLlamaExpected(st *engineState, command launchCommand, rt Runtime, approved bool, access servingFileAccess, expected *typedLlamaInputEvidence, advisory ...*map[string]modelEstimateInput) (*llamaServingLaunch, error) {
	return compileTypedLlamaExpectedContext(context.Background(), st, command, rt, approved, access, expected, advisory...)
}

func compileTypedLlamaExpectedContext(ctx context.Context, st *engineState, command launchCommand, rt Runtime, approved bool, access servingFileAccess, expected *typedLlamaInputEvidence, advisory ...*map[string]modelEstimateInput) (*llamaServingLaunch, error) {
	if st.manifest.FamilyOrDefault() != "llamacpp" {
		return nil, modelLoadError("", "engine", "Typed loads require an owned llama.cpp process.")
	}
	owner := rt.modelLoadOwner
	if owner == nil || owner.downloads == nil {
		return nil, modelLoadError("", "artifacts", "Model content admission is unavailable.")
	}
	release, recheck, err := owner.modelLoadContentRead(st.manifest.Engine)
	if err != nil {
		return nil, modelLoadError("", "artifacts", "Model content is busy or unsafe.")
	}
	defer release()
	ctx, cancel := context.WithTimeout(ctx, modelInspectionBudget)
	defer cancel()
	prepared, err := compileTypedLlamaInputsExpected(ctx, st, command, rt, approved, access, expected, advisory...)
	if err != nil {
		return nil, err
	}
	if err := recheck(); err != nil {
		return nil, err
	}
	return prepared, nil
}

// Reject overlapping sources even if their values happen to agree. Reset then
// really means native/default inheritance, rather than exposing an old flag.
func typedLlamaLaunchOptions(command launchCommand, rt Runtime) (int64, error) {
	cli, env, err := llamaLaunchOptions(command, rt.EditableLaunch)
	if err != nil {
		return 0, modelLoadError("", "launchText", err.Error())
	}
	for _, options := range []map[string]string{cli, env} {
		for _, key := range sortedKeys(options) {
			if key != "models-max" {
				return 0, modelLoadError("", key, "Conflicting native source; explicitly convert the complete external router before selecting typed mode.")
			}
		}
	}
	// Threads are in the preset grammar but are deliberately not envelope
	// options. Their typed ownership still requires conflict detection.
	for _, token := range command.Args {
		name, _, _ := strings.Cut(token, "=")
		name = llamaLongOption(name)
		if slices.Contains([]string{"--threads", "-t", "--threads-batch", "-tb"}, name) || (!strings.HasPrefix(name, "--") && (strings.HasPrefix(name, "-t") && len(name) > 2 && strings.ContainsAny(name[2:3], "0123456789+-") || strings.HasPrefix(name, "-tb") && len(name) > 3 && strings.ContainsAny(name[3:4], "0123456789+-"))) {
			return 0, modelLoadError("", "threads", "Conflicting native thread source.")
		}
	}
	for _, assignment := range append(os.Environ(), literalAssignments(command.Env)...) {
		name, _, _ := strings.Cut(assignment, "=")
		name = environmentKey(name)
		if name == "LLAMA_ARG_THREADS" || name == "LLAMA_ARG_THREADS_BATCH" {
			return 0, modelLoadError("", "threads", "Conflicting native thread environment source.")
		}
	}
	return 0, nil
}

func literalAssignments(env map[string]string) []string {
	values := make([]string, 0, len(env))
	for _, key := range sortedKeys(env) {
		values = append(values, key+"="+env[key])
	}
	return values
}

func compileTypedLlamaInputs(ctx context.Context, st *engineState, command launchCommand, rt Runtime, approved bool, advisory ...*map[string]modelEstimateInput) (*llamaServingLaunch, error) {
	return compileTypedLlamaInputsExpected(ctx, st, command, rt, approved, servingFileAccess{}, nil, advisory...)
}

// Retain request-local object identities across compilation and conversion.
// Serialized bindings remain the authority when an operation first starts.
type typedLlamaInputEvidence struct {
	build     sourceEvidence
	selectors []sourceEvidence
	models    map[string]modelFileList
	libraries map[string]sourceEvidence
}

func (e *typedLlamaInputEvidence) checkRecognition(ctx context.Context, owner *Executor) error {
	for _, list := range e.models {
		if list.source == nil || list.source.inventory == nil {
			continue
		}
		store := engineModelStore{engine: list.Engine, adapter: "llamacpp", root: list.source.root}
		current, err := typedLlamaRecognition(owner, store)(ctx, list.Model)
		old := list.source.inventory
		if err != nil || current.runID != old.runID || current.launchDigest != old.launchDigest || !slices.Equal(current.raw, old.raw) {
			return errModelInspectionChanged
		}
	}
	return nil
}

func (e *typedLlamaInputEvidence) checkedAccess(ctx context.Context, access servingFileAccess) (servingFileAccess, error) {
	if e.libraries == nil {
		return access, fmt.Errorf("missing native library evidence")
	}
	if err := validateSource(e.build); err != nil {
		return access, err
	}
	files := map[string]sourceEvidence{e.build.configured: e.build}
	for _, library := range e.libraries {
		if err := validateSource(library); err != nil {
			return access, err
		}
		files[library.configured] = library
	}
	for _, selector := range e.selectors {
		if err := validateSource(selector); err != nil {
			return access, err
		}
		files[selector.configured] = selector
	}
	for _, list := range e.models {
		if err := validateResolutionEvidence(ctx, list.source); err != nil {
			return access, err
		}
		for _, selector := range list.source.selectors {
			files[selector.file.configured] = selector.file
		}
		for _, file := range list.source.artifacts {
			files[file.configured] = sourceEvidence{configured: file.configured, target: file.abs, info: file.info, size: file.Size, mtime: file.info.ModTime().UnixNano()}
		}
	}
	original := access
	access.open = func(path string) (servingArtifactFile, error) {
		file, ok := files[path]
		if !ok {
			return nil, errNoArtifactBinding
		}
		return original.checkedOpen(ctx, file.configured, file.target, file.info)
	}
	return access, nil
}

func typedLlamaStore(st *engineState, command launchCommand) engineModelStore {
	store := engineModelStore{engine: st.manifest.Engine, adapter: "llamacpp", root: st.modelsDir}
	if cache := os.Getenv("LLAMA_CACHE"); cache != "" {
		store.root = expandPath(cache)
	}
	for name, value := range command.Env {
		if environmentKey(name) == environmentKey("LLAMA_CACHE") && value != "" {
			store.root = expandPath(value)
		}
	}
	store.root = filepath.Clean(store.root)
	return store
}

func typedLlamaRecognition(owner *Executor, store engineModelStore) func(context.Context, string) (engineInventory, error) {
	return func(ctx context.Context, model string) (engineInventory, error) {
		if owner == nil {
			return engineInventory{}, errNoArtifactBinding
		}
		st, err := owner.state(store.engine)
		if err != nil {
			return engineInventory{}, errModelInspectionChanged
		}
		observation, err := owner.recognizeModelSource(ctx, store.engine, model)
		if err != nil {
			return engineInventory{}, err
		}
		// Preview and lifecycle already own opMu. Resolve the current launch
		// from an immutable snapshot instead of reentering modelStoreFor.
		snapshot := snapshotLlamaState(st)
		st.mu.Lock()
		port := st.port
		current := st.running && st.runID == observation.runID && st.launchDigest == observation.launchDigest && st.modelsDir == observation.modelsDir
		st.mu.Unlock()
		command, err := launchForState(snapshot, port)
		if err != nil || !current || typedLlamaStore(snapshot, command) != store {
			return engineInventory{}, errModelInspectionChanged
		}
		owner.inventoryMu.Lock()
		current = owner.inventoryObservationCurrent(store.engine, observation.digests)
		owner.inventoryMu.Unlock()
		if !current || ctx.Err() != nil {
			return engineInventory{}, errModelInspectionChanged
		}
		return observation, nil
	}
}

func compileTypedLlamaInputsExpected(ctx context.Context, st *engineState, command launchCommand, rt Runtime, approved bool, access servingFileAccess, expected *typedLlamaInputEvidence, advisory ...*map[string]modelEstimateInput) (*llamaServingLaunch, error) {
	// Advisory output is deliberately distinct from the launch result and every
	// native/review digest. Only preview asks for it; starts and apply ignore it.
	for _, output := range advisory {
		if output != nil {
			*output = nil
		}
	}
	var estimates map[string]modelEstimateInput
	if len(advisory) != 0 {
		estimates = make(map[string]modelEstimateInput)
	}
	capacity, err := typedLlamaLaunchOptions(command, rt)
	if err != nil {
		return nil, err
	}
	requests := modelconfig.CloneLoads(rt.ModelLoads)
	operation, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
	for i := range requests {
		requests[i].Resolved = nil
	}
	loads, err := modelconfig.CanonicalLoads(requests)
	if err != nil || len(loads) == 0 {
		return nil, modelLoadError("", "requested", "A valid complete typed model set is required.")
	}
	// CanonicalLoads alone repairs missing keys. Keep old artifact assertions
	// separately from new preferences, whose allocation may intentionally differ.
	assertedLoads := modelconfig.CloneLoads(rt.ModelLoads)
	for i := range assertedLoads {
		assertedLoads[i].Requested = modelconfig.RequestedLoad{}
	}
	assertedLoads, err = modelconfig.CanonicalLoads(assertedLoads)
	if err != nil {
		return nil, err
	}
	if err := validateModelLoadArtifactAssertions(assertedLoads, rt.Serving); err != nil {
		return nil, err
	}
	for i := range loads {
		load := &loads[i]
		if modelLoadHasAuto(load.Requested) && (approved || operation.RecalculateModel != load.ProfileKey) {
			load.Resolved = rtLoadResolution(assertedLoads, load.ProfileKey)
		}
	}
	for _, load := range loads {
		if err := validateModelLoadGPUSelection(load.ProfileKey, load.Requested); err != nil {
			return nil, err
		}
		if len(modelLoadGPUSelection(load.Requested)) == 0 {
			continue
		}
		for _, other := range loads {
			if other.ProfileKey != load.ProfileKey && other.Requested.GPU != nil && other.Requested.GPU.Mode == "auto" && (other.Resolved == nil || other.Resolved.GPU.Mode == "layers" && other.Resolved.GPU.Layers != nil && *other.Resolved.GPU.Layers > 0) {
				return nil, modelLoadError(load.ProfileKey, "extraOptions", load.ProfileKey+" selects GPUs and cannot share an engine with reviewed Auto profile "+other.ProfileKey+"; set that profile's GPU mode to layers.")
			}
		}
	}
	// Refuse changes to frozen Auto intent before any advisory memory read.
	// GPU-selection refusals above keep their established precedence.
	for _, load := range loads {
		if modelLoadHasAuto(load.Requested) && load.Resolved != nil {
			if err := modelAutoFrozen(load.ProfileKey, load.Requested, load.Resolved); err != nil {
				return nil, err
			}
		}
	}
	if expected != nil {
		if expected.build.configured != command.Bin || validateSource(expected.build) != nil {
			return nil, modelLoadError("", "nativeBuildId", "Prepared executable changed before compilation.")
		}
		if err := expected.checkRecognition(ctx, rt.modelLoadOwner); err != nil {
			return nil, modelLoadError("", "artifacts", "Prepared native recognition changed before compilation.")
		}
		access, err = expected.checkedAccess(ctx, access)
		if err != nil || expected.build.configured != command.Bin {
			return nil, modelLoadError("", "artifacts", "Prepared inputs changed before compilation.")
		}
	}
	movedArtifacts, err := bindMovedArtifacts(ctx, loads, assertedLoads, access, func(load modelconfig.ModelLoad) bool {
		if approved {
			return false
		}
		if modelLoadHasAuto(load.Requested) {
			return operation.RecalculateModel == load.ProfileKey
		}
		return rt.acceptMoved || operation.RecalculateModel != ""
	})
	if err != nil {
		return nil, err
	}
	var buildEvidence sourceEvidence
	if expected != nil {
		buildEvidence = expected.build
	} else {
		buildEvidence, err = captureSource(command.Bin, false)
	}
	if err != nil {
		return nil, modelLoadError("", "nativeBuildId", "Native executable binding is unavailable.")
	}
	build, err := access.bindingExpected(ctx, buildEvidence.configured, buildEvidence.target, buildEvidence.info)
	if err != nil {
		return nil, modelLoadError("", "nativeBuildId", "Native executable binding is unavailable.")
	}
	var libraries map[string]sourceEvidence
	if expected != nil {
		libraries = expected.libraries
	}
	native, libraries, err := nativeBuildSnapshotExpected(ctx, buildEvidence, access, libraries)
	if err != nil {
		return nil, modelLoadError("", "nativeBuildId", "Native build content is unavailable or changed.")
	}
	buildID := native.Build
	nativeOptions := nativeHelpResults.lookup(buildID)
	if known, explicit := ctx.Value(nativeHelpDiscoveryKey{}).(bool); explicit && !known {
		nativeOptions = nil
	}
	for _, load := range loads {
		if err := validateNativeExtraOptions(load.ProfileKey, load.Requested.ExtraOptions, nativeOptions); err != nil {
			return nil, err
		}
	}
	nativeSource := ""
	reviewedSource := func(command launchCommand) (string, error) {
		if command.Bin != buildEvidence.configured || validateSource(buildEvidence) != nil {
			return "", modelLoadError("", "nativeBuildId", "Native executable changed during compilation.")
		}
		source, current, err := modelReviewedNativeSourceExpected(ctx, st, command, access, buildEvidence, libraries)
		if err == nil {
			libraries = current
		}
		return source, err
	}
	// Auto keeps the content build ID separate from its environment-bound source.
	for _, load := range loads {
		if !modelLoadHasAuto(load.Requested) {
			continue
		}
		if load.Resolved != nil {
			command, err = modelConstrainCommand(command, loads)
		} else {
			continue
		}
		if err != nil {
			return nil, err
		}
		nativeSource, err = reviewedSource(command)
		if err != nil {
			return nil, err
		}
		break
	}
	store := typedLlamaStore(st, command)
	if !filepath.IsAbs(store.root) {
		return nil, modelLoadError("", "artifacts", "An absolute model store is required.")
	}
	// Artifact resolution must not compare a new request to its previous
	// allocation. Only the old target-owned artifact assertions are input here.
	resolutionRT := rt
	resolutionRT.Serving = nil // declaration validation follows against bytes
	resolutionRT.ModelLoads = modelconfig.CloneLoads(loads)
	for i := range resolutionRT.ModelLoads {
		load := &resolutionRT.ModelLoads[i]
		load.Requested = modelconfig.RequestedLoad{}
		load.Resolved = rtLoadResolution(assertedLoads, load.ProfileKey)
	}
	plat := *st.plat
	plat.Runtime = resolutionRT
	resolver := &engineState{manifest: st.manifest, plat: &plat, modelsDir: st.modelsDir}
	access.scan = &modelArtifactScan{}
	access.recognize = typedLlamaRecognition(rt.modelLoadOwner, store)
	// Auto constrains device launch inputs later. Revalidate artifact selection
	// against the original resolver command, whose selection controls are fixed.
	resolutionCommand := command
	sections := map[string]map[string]string{}
	lists := make([]modelFileList, len(loads))
	for i := range loads {
		load := &loads[i]
		if err := nodepolicy.ValidateProfileKey(load.ProfileKey); err != nil {
			return nil, modelLoadError(load.ProfileKey, "profileKey", err.Error())
		}
		if err := validateLlamaModelID(load.ProfileKey); err != nil {
			return nil, modelLoadError(load.ProfileKey, "profileKey", "Use an exact canonical native profile key.")
		}
		if err := validateLlamaArtifactModelID(load.ModelID); err != nil {
			return nil, modelLoadError(load.ProfileKey, "modelId", "Use an exact native artifact model ID.")
		}
		list, _, err := resolveLlamaArtifacts(ctx, resolver, command, store, load.ModelID, access)
		if err != nil {
			return nil, modelLoadError(load.ProfileKey, "artifacts", "Exact artifacts are missing, ambiguous, changed or exceed the bounded resolver limits.")
		}
		facts, factsErr := decodeInspectionFacts(ctx, list, access)
		var layerCount *int32
		if factsErr == nil {
			projected, _ := projectInspectionFacts(facts)
			layerCount = projected.LayerCount
		}
		var artifacts []modelconfig.ArtifactBinding
		paths := map[string]string{}
		for _, file := range list.Files {
			if file.Role != "model" && file.Role != "draft" {
				return nil, modelLoadError(load.ProfileKey, "artifacts", "Companion artifacts have no typed native mapping.")
			}
			binding, err := access.bindingExpected(ctx, file.configured, file.abs, file.info)
			if err != nil {
				return nil, modelLoadError(load.ProfileKey, "artifacts", "Artifact changed during compilation.")
			}
			artifacts = append(artifacts, modelconfig.ArtifactBinding{Role: file.Role, Path: binding.SourcePath, ResolvedTarget: binding.ResolvedTarget, Size: strconv.FormatInt(binding.SizeBytes, 10), MtimeNS: binding.MtimeNS, SampleSHA256: binding.SampleSHA256})
			key := "model"
			if file.Role == "draft" {
				key = "model-draft"
			}
			if paths[key] == "" {
				paths[key] = file.configured
			}
		}
		var options map[string]string
		var resolved *modelconfig.ResolvedLoad
		if modelLoadHasAuto(load.Requested) {
			if load.Resolved != nil {
				options, resolved, err = modelAutoCompileFrozen(load.ProfileKey, load.Requested, load.Resolved, layerCount, artifacts, buildID)
			} else {
				if approved {
					return nil, staleApproval(modelLoadError(load.ProfileKey, "auto", "Accepted Auto resolution missing; explicitly Recalculate."))
				}
				op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
				if op.Memory == nil {
					return nil, modelSettingsParamsError(fmt.Errorf("target memory telemetry unavailable"))
				}
				memory, memoryErr := op.Memory.ModelAutoMemory(ctx)
				if memoryErr != nil {
					return nil, memoryErr
				}
				op.Memory = modelAutoSnapshotMemory{memory: memory}
				operation.Memory = op.Memory
				ctx = context.WithValue(ctx, modelAutoOperationKey{}, op)
				op.NativeSource = func(candidate *modelconfig.ResolvedLoad) (string, error) {
					finite := []modelconfig.ModelLoad{{ProfileKey: load.ProfileKey, ModelID: load.ModelID, Requested: load.Requested, Resolved: candidate}}
					constrained, err := modelConstrainCommand(command, finite)
					if err != nil {
						return "", err
					}
					return reviewedSource(constrained)
				}
				input := modelEstimateInput{Facts: facts, Bindings: modelEstimateBindings(artifacts), Allocation: &modelconfig.ResolvedLoad{Artifacts: artifacts, Options: modelEffectiveLoadOptions(load.Requested)}, BuildID: buildID, Support: modelEstimatorSupport}
				var state modelEstimateState
				estimates := &state
				if rt.modelLoadOwner != nil {
					estimates = &rt.modelLoadOwner.modelEstimates
				}
				proposal, resolveErr := resolveModelSettingsAutoInput(ctx, load.ProfileKey, load.Requested, input, op, estimates)
				if op.Proposals != nil {
					op.Proposals[load.ProfileKey] = proposal
				}
				if resolveErr != nil {
					return nil, resolveErr
				}
				options, resolved, err = modelAutoCompileCandidate(load.ProfileKey, proposal.Resolved, layerCount)
			}
		} else {
			options, resolved, err = typedLlamaModelOptions(load.ProfileKey, modelEffectiveLoadOptions(load.Requested), layerCount)
			if err == nil {
				resolved.Artifacts, resolved.NativeBuildID = artifacts, buildID
			}
		}
		if err != nil {
			return nil, err
		}
		if resolved.DeviceBinding != nil && strings.HasPrefix(resolved.ResolverRevision, "auto/") {
			options["device"] = resolved.DeviceBinding.Selector
		}
		maps.Copy(options, paths)
		if previous := rtLoadResolution(assertedLoads, load.ProfileKey); previous != nil && !modelLoadHasAuto(load.Requested) {
			resolved.ResolverRevision = previous.ResolverRevision
			resolved.ReviewedInputID = previous.ReviewedInputID
			if strings.HasPrefix(previous.ResolverRevision, "auto/") {
				resolved.ResolverRevision, resolved.ReviewedInputID = "", ""
				if source := modelAutoConversionSource(previous); source != "" {
					resolved.ResolverRevision, resolved.ReviewedInputID = externalConversionResolver, source
				}
			}
			if resolved.ResolverRevision != "" && resolved.ReviewedInputID != "" && resolved.GPU.Mode == "layers" && resolved.GPU.Layers != nil && *resolved.GPU.Layers > 0 && previous.DeviceBinding != nil {
				binding := *previous.DeviceBinding
				resolved.DeviceBinding = &binding
			}
		}
		sections[load.ProfileKey], lists[i] = options, list
		load.Resolved = resolved
		if estimates != nil && factsErr == nil {
			// Carry the same decoded facts and bindings, without a second read,
			// resolver or parser. Allocation is detached from ordinary outputs.
			allocation := resolved.Clone()
			// This ceiling is already emitted by the compiler; carrying it does
			// not materialize a new user preference or native default.
			if cacheRAM, err := strconv.ParseInt(options["cache-ram"], 10, 32); err == nil {
				mib := int32(cacheRAM)
				allocation.Options.CacheRAM = &mib
			}
			estimates[load.ProfileKey] = modelEstimateInput{Facts: facts, Bindings: modelEstimateBindings(resolved.Artifacts), Allocation: &allocation, BuildID: buildID, Support: modelEstimatorSupport}
			if operation.Memory != nil && !approved {
				input := estimates[load.ProfileKey]
				input.Memory, err = operation.Memory.ModelAutoMemory(ctx)
				if err != nil {
					return nil, err
				}
				operation.Memory = modelAutoSnapshotMemory{memory: input.Memory}
				ctx = context.WithValue(ctx, modelAutoOperationKey{}, operation)
				estimates[load.ProfileKey] = input
			}
		}
	}
	command, err = modelConstrainCommand(command, loads)
	if err != nil {
		return nil, err
	}
	for i := range loads {
		load := &loads[i]
		if !modelLoadHasAuto(load.Requested) {
			continue
		}
		current, sourceErr := reviewedSource(command)
		if sourceErr != nil {
			return nil, sourceErr
		}
		nativeSource = current
		if modelAutoNativeSource(load.Resolved) != current {
			if approved || rtLoadResolution(assertedLoads, load.ProfileKey) != nil && operation.RecalculateModel != load.ProfileKey {
				return nil, staleApproval(modelLoadError(load.ProfileKey, "nativeBuildId", "Reviewed native source changed; explicitly Recalculate."))
			}
			conversion, memory, ok := modelAutoReviewParts(load.Resolved.ReviewedInputID)
			if !ok {
				return nil, modelLoadError(load.ProfileKey, "auto", "Invalid reviewed input.")
			}
			load.Resolved.ReviewedInputID, err = modelAutoReviewHash(load.ProfileKey, load.Requested, load.Resolved, conversion, memory, current)
			if err != nil {
				return nil, err
			}
		}
	}
	data, err := typedLlamaPreset(sections)
	if err != nil {
		return nil, err
	}
	result := &llamaServingLaunch{Snapshot: data, ModelLoads: loads}
	if rt.Serving != nil {
		// Existing declarations keep their own model order, artifact assertions,
		// output bounds and memory validation. No typed entry creates one.
		declaredRT := rt
		declaredRT.Serving = rt.Serving.Clone()
		for i := range declaredRT.Serving.Models {
			model := &declaredRT.Serving.Models[i]
			model.ExpectedNativeSourceDigest = ""
			if sections[model.ModelID] == nil {
				return nil, modelLoadError(model.ModelID, "serving", "Declaration and typed router model sets must match exactly.")
			}
			for _, artifact := range model.Artifacts {
				if artifact.Binding != nil && movedArtifacts[artifact.Binding.SourcePath] != "" {
					artifact.Binding.MtimeNS = movedArtifacts[artifact.Binding.SourcePath]
				}
			}
			sections[model.ModelID]["n-predict"] = strconv.FormatInt(model.MaxOutputTokens, 10)
		}
		data, err = typedLlamaPreset(sections)
		if err != nil {
			return nil, err
		}
		sectionEvidence := &llamaServingEvidence{artifacts: map[string]sourceEvidence{}}
		for _, list := range lists {
			for _, file := range list.Files {
				sectionEvidence.artifacts[file.configured] = sourceEvidence{configured: file.configured, target: file.abs, info: file.info, size: file.Size, mtime: file.info.ModTime().UnixNano()}
			}
		}
		result, err = compileLlamaServingSectionsExpected(ctx, declaredRT, false, "", data, nil, sections, nil, nil, capacity, access, sectionEvidence)
		if err != nil {
			return nil, err
		}
		result.Snapshot = data
		result.ModelLoads = loads
		for i := range result.Config.Models {
			result.Config.Models[i].ExpectedNativeSourceDigest = ""
		}
	}
	presetDigest := modelLoadDigest(result.Snapshot)
	identityCommand, err := typedLlamaIdentityCommand(command, rt)
	if err != nil {
		return nil, err
	}
	// Digest fields are outputs, excluded from their own input identity.
	identity, err := json.Marshal(struct {
		Loads       []modelconfig.ModelLoad
		Build       *engines.ServingBinding
		Command     launchCommand
		Capacity    int64
		Snapshot    []byte
		Declaration *engines.ServingConfig
	}{loads, build, identityCommand, capacity, result.Snapshot, result.Config})
	if err != nil {
		return nil, err
	}
	result.Digest = modelLoadDigest(identity)
	if result.Config != nil {
		for i := range result.Config.Models {
			expected := rt.Serving.Models[i].ExpectedNativeSourceDigest
			if approved && expected != result.Digest {
				return nil, staleApproval(modelLoadError(result.Config.Models[i].ModelID, "serving", "Declaration requires a fresh typed preview."))
			}
			result.Config.Models[i].ExpectedNativeSourceDigest = result.Digest
		}
		for i := range result.Envelopes {
			result.Envelopes[i].NativeSourceDigest = result.Digest
		}
	}
	for i := range loads {
		loads[i].Resolved.SourceInputDigest = result.Digest
		loads[i].Resolved.PresetDigest = presetDigest
		if approved && !reflect.DeepEqual(loads[i].Resolved, rtLoadResolution(assertedLoads, loads[i].ProfileKey)) {
			return nil, staleApproval(modelLoadError(loads[i].ProfileKey, "resolved", "Typed inputs changed; obtain a fresh preview before starting."))
		}
	}
	// Re-resolve through Task 5 after facts and samples. Do not trust the first
	// resolution or a cached metadata read as source revalidation.
	if expected != nil {
		if err := expected.checkRecognition(ctx, rt.modelLoadOwner); err != nil {
			return nil, modelLoadError("", "artifacts", "Prepared native recognition changed during compilation.")
		}
	}
	for i, load := range loads {
		finalAccess := access
		if original := lists[i].source.inventory; original != nil {
			recognize := access.recognize
			finalAccess.recognize = func(ctx context.Context, model string) (engineInventory, error) {
				observation, err := recognize(ctx, model)
				if err != nil || observation.runID != original.runID || observation.launchDigest != original.launchDigest || !slices.Equal(observation.raw, original.raw) {
					return engineInventory{}, errModelInspectionChanged
				}
				return observation, nil
			}
		}
		after, _, err := resolveLlamaArtifactsExpected(ctx, resolver, resolutionCommand, store, load.ModelID, finalAccess, lists[i])
		if err != nil || after.binding != lists[i].binding || !sameTypedFiles(lists[i], after) {
			return nil, modelLoadError(load.ProfileKey, "artifacts", "Artifact source changed during compilation.")
		}
	}
	if current, err := access.bindingExpected(ctx, buildEvidence.configured, buildEvidence.target, buildEvidence.info); err != nil || *current != *build {
		return nil, modelLoadError("", "nativeBuildId", "Native executable changed during compilation.")
	}
	if current, _, err := nativeBuildSnapshotExpected(ctx, buildEvidence, access, libraries); err != nil || current.Build != buildID {
		return nil, modelLoadError("", "nativeBuildId", "Native build changed during compilation.")
	}
	if _, err := modelConstrainCommand(command, loads); err != nil {
		return nil, err
	}
	// The known b12020 common/common.h default is separate from device report
	// recognition. Recognizing another build's devices does not prove swa_full.
	if estimates != nil {
		constrained, constrainErr := modelConstrainCommand(command, loads)
		_, sourceErr := reviewedSource(constrained)
		if constrainErr == nil && sourceErr == nil && len(libraries) != 0 && modelLoadHasKnownSWADefault(st, buildEvidence) {
			for model, input := range estimates {
				full := sections[model]["swa-full"] == "true"
				input.SWAFull = &full
				estimates[model] = input
			}
		}
	}
	for _, load := range loads {
		if !modelLoadHasAuto(load.Requested) {
			continue
		}
		current, err := reviewedSource(command)
		if err != nil || current != nativeSource {
			return nil, modelLoadError("", "nativeBuildId", "Native source changed during compilation.")
		}
		break
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := modelconfig.CanonicalLoads(loads); err != nil {
		return nil, err
	}
	if result.evidence == nil {
		result.evidence = &llamaServingEvidence{}
	}
	inputs := &typedLlamaInputEvidence{build: buildEvidence, models: make(map[string]modelFileList, len(loads)), libraries: libraries}
	for i, load := range loads {
		inputs.models[load.ProfileKey] = lists[i]
	}
	result.evidence.typed = inputs
	for _, output := range advisory {
		if output != nil {
			*output = estimates
		}
	}
	return result, nil
}

// The inherited SWA default was derived from b12020 (0ec4027a) common/common.h.
// Retain its existing managed artifact-declaration assurance independently of
// the device helper's per-build report check. New builds need separate evidence
// for this semantic default; a content fingerprint or device report is not it.
func modelLoadHasKnownSWADefault(st *engineState, executable sourceEvidence) bool {
	st.mu.Lock()
	root := st.installDir
	known := false
	if st.plat != nil && st.plat.Install != nil {
		for _, artifact := range st.plat.Install.Artifacts {
			if artifact.Name == "server" && strings.Contains(artifact.URL, "/releases/download/b12020/") && len(artifact.SHA256) == 64 {
				known = true
			}
		}
	}
	st.mu.Unlock()
	if !known || !isManagedInstallPath(executable.configured, root) {
		return false
	}
	resolved, err := resolveExistingPath(root)
	return err == nil && resolved == filepath.Dir(executable.target)
}

// Auto-assigned endpoints are lifecycle outputs. Bind the managed zero-port
// intent, while keeping every unrelated launch argument/environment value.
func typedLlamaIdentityCommand(command launchCommand, rt Runtime) (launchCommand, error) {
	if rt.Port != 0 || rt.EditableLaunch == nil {
		return command, nil
	}
	identity := launchCommand{Bin: command.Bin, Env: maps.Clone(command.Env)}
	host := effectiveBind(rt.Bind, "")
	for name := range identity.Env {
		control := rt.EditableLaunch.environmentControl(name)
		if control != nil && strings.Contains(control.Value, "{server.port}") {
			value, ok := control.managedValue(host, "0")
			if !ok {
				return launchCommand{}, modelLoadError("", "serverPort", "Invalid managed endpoint binding.")
			}
			identity.Env[name] = value
		}
	}
	for i := 0; i < len(command.Args); {
		control, _, last, err := rt.EditableLaunch.readControl(command.Args, i)
		if err != nil {
			return launchCommand{}, err
		}
		if control != nil {
			if strings.Contains(control.Value, "{server.port}") {
				value, ok := control.managedValue(host, "0")
				if !ok || control.Implicit != nil {
					return launchCommand{}, modelLoadError("", "serverPort", "Invalid managed endpoint binding.")
				}
				identity.Args = append(identity.Args, control.flag(), value)
			} else {
				identity.Args = append(identity.Args, command.Args[i:last+1]...)
			}
			i = last + 1
			continue
		}
		_, values := llamaLaunchArgument(command.Args[i])
		end := min(i+1+values, len(command.Args))
		identity.Args = append(identity.Args, command.Args[i:end]...)
		i = end
	}
	return identity, nil
}

// typedApprovalStale reports saved typed settings that restore or start would
// refuse: the same two comparisons the approved compile makes. An unapproved
// typed compile replaces the saved values with the ones it computed, so unlike a
// finite compile it cannot refuse the difference. Saved loads carry their own
// digests, so a configuration with no declaration still has an approval to lose.
func typedApprovalStale(rt Runtime, prepared *llamaServingLaunch) bool {
	if len(rt.ModelLoads) == 0 {
		return false
	}
	if slices.ContainsFunc(prepared.ModelLoads, func(load modelconfig.ModelLoad) bool {
		return !reflect.DeepEqual(load.Resolved, rtLoadResolution(rt.ModelLoads, load.ProfileKey))
	}) {
		return true
	}
	return rt.Serving != nil && slices.ContainsFunc(rt.Serving.Models, func(model engines.ServingModel) bool {
		return model.ExpectedNativeSourceDigest != prepared.Digest
	})
}

// bindMovedArtifacts compares each saved binding to its file before resolution.
// Typed loads hold no full hash, so a new modified time is the only signal for a
// change between the samples: matching samples cannot show the content is the
// same, and only a reviewed acceptance may bind the new time. It updates asserted
// in place for accepted files and returns their new times by path. A refusal
// names every moved file it does not accept. A missing or unsafe file, or a
// same-time sample change, is left to the resolver's refusal.
func bindMovedArtifacts(ctx context.Context, loads, asserted []modelconfig.ModelLoad, access servingFileAccess, accept func(modelconfig.ModelLoad) bool) (map[string]string, error) {
	moved := map[string]string{}
	var refused []string
	refusedProfile, recalculate := "", false
	for _, load := range loads {
		resolved := rtLoadResolution(asserted, load.ProfileKey)
		if resolved == nil {
			continue
		}
		for i := range resolved.Artifacts {
			artifact := &resolved.Artifacts[i]
			if !artifactPathSafe(artifact.Path) {
				continue
			}
			file, err := captureSource(artifact.Path, false)
			if err != nil {
				continue
			}
			mtime := strconv.FormatInt(file.mtime, 10)
			if file.target == artifact.ResolvedTarget && strconv.FormatInt(file.size, 10) == artifact.Size && mtime == artifact.MtimeNS {
				continue
			}
			sampleMatches := false
			if file.target == artifact.ResolvedTarget && strconv.FormatInt(file.size, 10) == artifact.Size {
				current, err := access.bindingExpected(ctx, file.configured, file.target, file.info)
				if err != nil {
					continue
				}
				sampleMatches = current.SampleSHA256 == artifact.SampleSHA256
			}
			if !sampleMatches {
				return nil, modelLoadError(load.ProfileKey, "artifacts", fmt.Sprintf("Model file %s is not the file these settings saved: its location, size or samples differ. Restore the original file to use these settings.", filepath.Base(artifact.Path)))
			}
			if !accept(load) {
				// A refusal that says Recalculate names an Auto profile to Recalculate.
				auto := modelLoadHasAuto(load.Requested)
				if refusedProfile == "" || (auto && !recalculate) {
					refusedProfile = load.ProfileKey
				}
				if !slices.Contains(refused, artifact.Path) {
					refused = append(refused, artifact.Path)
				}
				recalculate = recalculate || auto
				continue
			}
			artifact.MtimeNS = mtime
			moved[artifact.Path] = mtime
		}
	}
	if len(refused) != 0 {
		return nil, staleApproval(modelLoadError(refusedProfile, "artifacts", modelconfig.MovedArtifactReason(refused, recalculate)))
	}
	return moved, nil
}

func rtLoadResolution(loads []modelconfig.ModelLoad, profileKey string) *modelconfig.ResolvedLoad {
	for _, load := range loads {
		if load.ProfileKey == profileKey {
			return load.Resolved
		}
	}
	return nil
}

// New profiles have no sampled assertions yet. Siblings of one exact artifact
// selector must agree on bindings and declared content, independent of allocation.
func validateModelLoadArtifactAssertions(loads []modelconfig.ModelLoad, serving *engines.ServingConfig) error {
	assertions := map[string][]modelconfig.ArtifactBinding{}
	declarations := map[string][]engines.ServingArtifact{}
	if serving != nil {
		for _, model := range serving.Models {
			declarations[model.ModelID] = model.Artifacts
		}
	}
	declaredAssertions := map[string][]engines.ServingArtifact{}
	for _, load := range loads {
		if load.Resolved != nil {
			if previous, exists := assertions[load.ModelID]; exists && !reflect.DeepEqual(previous, load.Resolved.Artifacts) {
				return modelLoadError(load.ProfileKey, "artifacts", "Profiles of one model disagree on artifacts.")
			}
			assertions[load.ModelID] = load.Resolved.Artifacts
		}
		if artifacts, declared := declarations[load.ProfileKey]; declared {
			if previous, exists := declaredAssertions[load.ModelID]; exists && !reflect.DeepEqual(previous, artifacts) {
				return modelLoadError(load.ProfileKey, "artifacts", "Profiles of one model disagree on artifacts.")
			}
			declaredAssertions[load.ModelID] = artifacts
		}
	}
	return nil
}

// Export consumes persisted artifact assertions without reconstructing a native
// build identity or executing the typed launch/Auto allocation compiler.
func validateTypedLlamaExport(rt Runtime, model string) error {
	var selected *modelconfig.ModelLoad
	for i := range rt.ModelLoads {
		load := &rt.ModelLoads[i]
		if load.ModelID == model {
			if load.Resolved == nil || load.Validate() != nil {
				return errNoArtifactBinding
			}
			if selected != nil && !reflect.DeepEqual(selected.Resolved.Artifacts, load.Resolved.Artifacts) {
				return errNoArtifactBinding
			}
			selected = load
		}
	}
	return nil
}

func sameTypedFiles(a, b modelFileList) bool {
	if len(a.Files) != len(b.Files) {
		return false
	}
	for i, f := range a.Files {
		if f.Role != b.Files[i].Role || f.configured != b.Files[i].configured || f.sample != b.Files[i].sample || !sameCopyFile(f.info, b.Files[i].info) {
			return false
		}
	}
	return true
}

func modelLoadDigest(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func modelLoadChangeProfileKey(change *modelconfig.ModelChange) string {
	if change.ProfileKey != "" {
		return change.ProfileKey
	}
	return change.ModelID
}

func modelLoadPreviewDigest(native, conversion string, change *modelconfig.ModelChange, baseline settings.Config) (string, error) {
	identity, err := json.Marshal(struct {
		Native, Conversion  string
		ProfileKey, ModelID string
		Remove              bool
		Patch               *modelconfig.LoadPatch
		Baseline            settings.Config
	}{native, conversion, modelLoadChangeProfileKey(change), change.ModelID, change.Remove, change.Load, baseline})
	if err != nil {
		return "", err
	}
	return modelLoadDigest(identity), nil
}

func typedLlamaPreset(sections map[string]map[string]string) ([]byte, error) {
	var out strings.Builder
	for _, model := range sortedKeys(sections) {
		out.WriteString("[" + model + "]\n")
		for _, key := range sortedKeys(sections[model]) {
			value := sections[model][key]
			if err := validateLlamaPresetValue(key, value, model); err != nil {
				return nil, err
			}
			out.WriteString(key + " = " + value + "\n")
		}
		out.WriteString("\n")
	}
	if out.Len() > servingSmallFileLimit {
		return nil, modelLoadError("", "preset", "Generated preset exceeds the bounded size limit.")
	}
	return []byte(out.String()), nil
}

// Typed mode has already rejected overlapping CLI/environment/preset owners.
// Materialize pinned finite defaults once, leaving automatic flash attention
// unresolved and thread-dependent scratch explicitly unknown in the estimator.
func modelEffectiveLoadOptions(requested modelconfig.RequestedLoad) modelconfig.RequestedLoad {
	r := requested.Clone()
	batch, ubatch, cache, slots := int32(2048), int32(512), int32(512), int32(1)
	f16, offload := "f16", true
	if r.BatchSize == nil {
		r.BatchSize = &batch
	}
	if r.UBatchSize == nil {
		r.UBatchSize = &ubatch
	}
	if r.CacheRAM == nil {
		r.CacheRAM = &cache
	}
	if r.Slots == nil {
		r.Slots = &slots
	}
	if r.CacheTypeK == nil {
		r.CacheTypeK = &f16
	}
	if r.CacheTypeV == nil {
		value := f16
		r.CacheTypeV = &value
	}
	if r.KVOffload == nil {
		r.KVOffload = &offload
	}
	if r.OpOffload == nil {
		value := offload
		r.OpOffload = &value
	}
	return r
}

func typedLlamaModelOptions(model string, requested modelconfig.RequestedLoad, layers *int32) (map[string]string, *modelconfig.ResolvedLoad, error) {
	fail := func(field, reason string) (map[string]string, *modelconfig.ResolvedLoad, error) {
		return nil, nil, modelLoadError(model, field, reason)
	}
	if err := requested.Validate(); err != nil {
		if field, _, ok := strings.Cut(err.Error(), ":"); ok && (strings.HasPrefix(field, "cpuExperts.") || field == "extraOptions") {
			return fail(field, err.Error())
		}
		return fail("requested", err.Error())
	}
	if err := validateModelLoadGPUSelection(model, requested); err != nil {
		return nil, nil, err
	}
	if err := validateModelExtraOptions(model, requested.ExtraOptions); err != nil {
		return nil, nil, err
	}
	tokens, slots := int32(8192), int32(1)
	if requested.Context != nil {
		if requested.Context.Mode == "auto" {
			return fail("context", "Auto requires a reviewed finite resolution.")
		}
		tokens = requested.Context.Tokens
	}
	if requested.Slots != nil {
		slots = *requested.Slots
	}
	per, total, err := modelconfig.TotalContext(tokens, slots)
	if err != nil {
		return fail("context", err.Error())
	}
	gpu := modelconfig.GPU{Mode: "native-default"}
	if requested.GPU != nil {
		gpu = *requested.GPU
	}
	switch gpu.Mode {
	case "auto":
		return fail("gpu", "Auto requires a reviewed finite resolution.")
	case "native-default":
		if layers == nil || *layers == math.MaxInt32 {
			return fail("gpu", "Native default placement requires a bounded model layer count; select explicit CPU or layers.")
		}
		n := *layers + 1 // pinned native all-layers default includes output layer
		gpu = modelconfig.GPU{Mode: "layers", Layers: &n}
	case "layers":
		if layers == nil {
			return fail("gpu.layers", "Layer placement requires the model layer count.")
		}
		if int64(*gpu.Layers) > int64(*layers)+1 {
			return fail("gpu.layers", "Layer placement exceeds the model plus output layer.")
		}
	}
	options := map[string]string{"ctx-size": strconv.Itoa(int(total)), "parallel": strconv.Itoa(int(slots)), "kv-unified": "false", "fit": "off", "sleep-idle-seconds": "-1", "cache-ram": "512", "load-on-startup": "false", "gpu-layers": "0"}
	if modelLoadNativeFit(requested) {
		options["fit"] = "on"
		if requested.FitTarget != nil {
			options["fit-target"] = *requested.FitTarget
		}
		if requested.FitCtx != nil {
			options["fit-ctx"] = strconv.Itoa(int(*requested.FitCtx))
		}
	}
	if gpu.Mode == "layers" {
		options["gpu-layers"] = strconv.Itoa(int(*gpu.Layers))
	}
	if modelLoadNativeFit(requested) && (requested.GPU == nil || requested.GPU.Mode == "native-default") {
		// The resolved layer count bounds the request, not native fit's actual
		// placement. Fit loads have no finite configuration key or estimate.
		options["gpu-layers"] = "auto"
	}
	// llama.cpp b11146 common/arg.cpp:2755-2771 maps only these finite
	// expert overrides. none deliberately emits neither native flag.
	// https://github.com/ggml-org/llama.cpp/blob/b11146/common/arg.cpp#L2755-L2771
	if experts := requested.CPUExperts; experts != nil {
		switch experts.Mode {
		case "all":
			options["cpu-moe"] = "true"
		case "layers":
			if layers == nil || *experts.Layers > *layers {
				return fail("cpuExperts.layers", "CPU expert placement requires a target block count and cannot exceed it.")
			}
			options["n-cpu-moe"] = strconv.Itoa(int(*experts.Layers))
		}
	}
	for _, field := range []struct {
		name, key string
		value     *string
	}{{"cacheTypeK", "cache-type-k", requested.CacheTypeK}, {"cacheTypeV", "cache-type-v", requested.CacheTypeV}} {
		if field.value != nil {
			if *field.value != "f16" && *field.value != "q8_0" {
				return fail(field.name, "This precision requires validated build/architecture support; only f16 and q8_0 are currently mapped.")
			}
			options[field.key] = *field.value
		}
	}
	if requested.FlashAttention != nil && *requested.FlashAttention != "inherit" {
		options["flash-attn"] = *requested.FlashAttention
	}
	for _, field := range []struct {
		key   string
		value *int32
	}{{"batch-size", requested.BatchSize}, {"ubatch-size", requested.UBatchSize}, {"threads", requested.Threads}, {"threads-batch", requested.ThreadsBatch}, {"cache-ram", requested.CacheRAM}} {
		if field.value != nil {
			options[field.key] = strconv.Itoa(int(*field.value))
		}
	}
	// Pinned native inherited batch values, used only for cross-field checking.
	batch, ubatch := int32(2048), int32(512)
	if requested.BatchSize != nil {
		batch = *requested.BatchSize
	}
	if requested.UBatchSize != nil {
		ubatch = *requested.UBatchSize
	}
	if ubatch > batch {
		return fail("ubatchSize", "Physical batch exceeds the effective logical batch.")
	}
	for _, field := range []struct {
		key   string
		value *bool
	}{{"kv-offload", requested.KVOffload}, {"op-offload", requested.OpOffload}} {
		if field.value != nil {
			options[field.key] = strconv.FormatBool(*field.value)
		}
	}
	maps.Copy(options, requested.ExtraOptions)
	resolvedOptions := requested.Clone()
	resolvedOptions.Context = &modelconfig.Context{Mode: "fixed", Tokens: per}
	resolvedOptions.Slots = &slots
	resolvedOptions.GPU = &gpu
	unified := false
	return options, &modelconfig.ResolvedLoad{PerSlotContext: per, TotalContext: total, Slots: slots, KVUnified: &unified, GPU: gpu, Options: resolvedOptions}, nil
}

// Read-only worker preview. Acceptance and policy implications remain with
// the settings operation owner.
func (e *Executor) previewModelLoadChange(ctx context.Context, st *engineState, p settings.Request) (settings.Preview, error) {
	if err := validateModelLoadPreview(st, p); err != nil {
		result := settings.Preview{Settings: p.Settings.Clone(), Args: []string{}, Env: []string{}, Errors: map[string]string{}}
		return modelLoadPreviewFailure(ctx, p, result, err)
	}
	// Validate before discovery, then establish one metadata deadline for all
	// attempts. Help keeps its separate budget and is never repeated here.
	_, known := e.nativeOptions(ctx, st)
	ctx = context.WithValue(ctx, nativeHelpDiscoveryKey{}, known)
	ctx, cancel := context.WithTimeout(ctx, modelInspectionBudget)
	defer cancel()
	var result settings.Preview
	var pending *modelEstimateInput
	err := retryModelRead(ctx, func(retry *bool) error {
		var err error
		pending = nil
		result, err = e.previewModelLoadChangeOnce(ctx, st, p, retry, &pending)
		return err
	})
	if err != nil || pending == nil {
		return result, err
	}
	// All authoritative work is complete and its content lease is released.
	// The advisory estimate uses the remaining metadata budget; its failure,
	// panic or expiration is an Unknown estimate, never a Preview refusal.
	attachModelLoadEstimate(&result, e.modelEstimates.estimate(ctx, *pending))
	return result, nil
}

// A reply-size failure after attachment drops only the advisory estimate.
func attachModelLoadEstimate(result *settings.Preview, estimate modelconfig.Estimate) {
	result.Estimate = &estimate
	if err := modelconfig.CheckPayloadSize(*result); err != nil {
		result.Estimate = nil
	}
}

func validateModelLoadPreview(st *engineState, p settings.Request) error {
	profileKey := ""
	if p.ModelChange != nil {
		profileKey = modelLoadChangeProfileKey(p.ModelChange)
	}
	if err := p.Validate(); err != nil {
		return err
	}
	if p.ModelChange != nil && p.ModelChange.Load != nil {
		if err := validateModelLoadGPUSelection(profileKey, p.ModelChange.Load.Set); err != nil {
			return err
		}
		st.mu.Lock()
		loads := modelconfig.CloneLoads(st.plat.Runtime.ModelLoads)
		st.mu.Unlock()
		if len(loads) != 0 {
			// A selected patch can inherit the other half of an invalid
			// placement combination. Merge through the existing owner before
			// native help, not only before the later typed compilation.
			if _, err := patchModelLoad(loads, profileKey, p.ModelChange.ModelID, *p.ModelChange.Load); err != nil {
				return err
			}
		}
	}
	if st.manifest.FamilyOrDefault() != "llamacpp" || p.ModelChange == nil || p.ModelChange.Load == nil && !p.ModelChange.Remove || p.ModelChange.Policy != nil {
		return modelLoadError("", "request", "Only a local llama.cpp load-only preview is available.")
	}
	return nil
}

func modelLoadPreviewFailure(ctx context.Context, p settings.Request, result settings.Preview, err error) (settings.Preview, error) {
	if ctx.Err() != nil {
		return settings.Preview{}, ctx.Err()
	}
	field := "modelLoads"
	var typed *ServingFieldError
	if errors.As(err, &typed) {
		field = typed.Field
	}
	result.Errors[field] = err.Error()
	if p.ModelChange != nil {
		op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
		if proposal, ok := op.Proposals[modelLoadChangeProfileKey(p.ModelChange)]; ok {
			result.Auto = &proposal
		}
	}
	return result, nil
}

func (e *Executor) previewModelLoadChangeOnce(ctx context.Context, st *engineState, p settings.Request, retry *bool, pending **modelEstimateInput) (settings.Preview, error) {
	result := settings.Preview{Settings: p.Settings.Clone(), Args: []string{}, Env: []string{}, Errors: map[string]string{}}
	fail := func(err error) (settings.Preview, error) { return modelLoadPreviewFailure(ctx, p, result, err) }
	if err := validateModelLoadPreview(st, p); err != nil {
		return fail(err)
	}
	profileKey := modelLoadChangeProfileKey(p.ModelChange)
	if !e.modelInspection.active.CompareAndSwap(false, true) {
		*retry = true
		return fail(modelLoadError("", "busy", "Another model metadata preview or inspection is active."))
	}
	defer e.modelInspection.active.Store(false)
	release, recheck, err := e.modelLoadContentRead(p.Engine)
	if err != nil {
		return fail(err)
	}
	defer release()
	snapshot, err := inspectionSnapshot(st, p.Engine)
	if err != nil {
		*retry = errors.Is(err, errModelInspectionBusy)
		return fail(err)
	}
	rt := snapshot.state.plat.Runtime
	rt.modelLoadOwner = e
	textRT := rt
	if llamaServingRuntime(snapshot.state, rt) {
		textRT = llamaServingBase(rt)
	}
	text, err := snapshot.command.argumentText(textRT)
	if err != nil {
		return fail(err)
	}
	baseline := (settings.Config{WorkingSetLimitMiB: rt.WorkingSetLimitMiB, ServerPort: rt.Port, ProxyPort: p.Settings.ProxyPort, LaunchText: text, Serving: rt.Serving, ModelLoads: rt.ModelLoads}).Clone()
	if err := p.ValidateModelChange(baseline, false); err != nil {
		return fail(err)
	}
	before := map[string]map[string]string{}
	conversionDigest := ""
	cli, env, err := llamaLaunchOptions(snapshot.command, rt.EditableLaunch)
	if err != nil {
		return fail(err)
	}
	if len(rt.ModelLoads) == 0 && (cli["models-preset"] != "" || env["models-preset"] != "") {
		conversion, err := e.previewTypedLlamaConversion(ctx, snapshot.state, snapshot.command, rt)
		if err != nil {
			return fail(err)
		}
		rt, before, conversionDigest = conversion.Runtime, conversion.Before, conversion.SourceDigest
	} else if len(rt.ModelLoads) != 0 {
		beforeRT := rt
		if p.ModelChange.Load != nil && modelLoadHasAuto(p.ModelChange.Load.Set) {
			// Render the selected model's old finite settings for the diff,
			// without requiring its old native build to remain installed. This
			// is not a launch or a new recommendation. Artifact bindings and
			// every sibling's frozen identity still pass the normal compiler.
			// A moved file is judged by the Recalculate compile below, since
			// nothing this render binds is saved.
			beforeRT.acceptMoved = true
			beforeRT.ModelLoads = modelconfig.CloneLoads(rt.ModelLoads)
			for i := range beforeRT.ModelLoads {
				load := &beforeRT.ModelLoads[i]
				if load.ProfileKey == profileKey && modelLoadHasAuto(load.Requested) {
					if err := modelAutoFrozen(load.ProfileKey, load.Requested, load.Resolved); err != nil {
						return fail(err)
					}
					load.Requested = load.Resolved.Options.Clone()
				}
			}
		}
		prepared, err := compileTypedLlamaInputs(ctx, snapshot.state, snapshot.command, beforeRT, false)
		if err != nil {
			return fail(err)
		}
		before, err = parseLlamaServingPresetAllowing(prepared.Snapshot, modelLoadAllowedPresetKeys(prepared.ModelLoads))
		if err != nil {
			return fail(err)
		}
	}
	// Every explicit Auto patch is Recalculate, even with identical preferences.
	// Preserve conversion provenance independently before clearing the old result.
	conversionSource := modelAutoConversionSource(rtLoadResolution(rt.ModelLoads, profileKey))
	if conversionDigest != "" {
		conversionSource = conversionDigest
	}
	if p.ModelChange.Remove {
		rt.ModelLoads, err = removeModelLoad(rt.ModelLoads, profileKey, p.ModelChange.ModelID)
		if err == nil {
			rt.Serving = removeModelLoadDeclaration(rt.Serving, profileKey)
		}
	} else {
		rt.ModelLoads, err = patchModelLoad(rt.ModelLoads, profileKey, p.ModelChange.ModelID, *p.ModelChange.Load)
	}
	if err != nil {
		return fail(err)
	}
	op, _ := ctx.Value(modelAutoOperationKey{}).(modelAutoOperation)
	op.ConversionSource = conversionSource
	if p.ModelChange.Load != nil && modelLoadHasAuto(p.ModelChange.Load.Set) {
		op.RecalculateModel = profileKey
	}
	op.Proposals = make(map[string]modelconfig.AutoProposal)
	ctx = context.WithValue(ctx, modelAutoOperationKey{}, op)
	vars := snapshot.state.pathVars()
	vars["host"], vars["port"] = effectiveBind(rt.Bind, ""), strconv.Itoa(rt.Port)
	command, err := resolveProcessLaunch(llamaServingBase(rt), snapshot.command.Bin, vars)
	if err != nil {
		return fail(err)
	}
	var advisory map[string]modelEstimateInput
	prepared := &llamaServingLaunch{}
	if len(rt.ModelLoads) != 0 {
		prepared, err = compileTypedLlamaInputs(ctx, snapshot.state, command, rt, false, &advisory)
		if err != nil {
			return fail(err)
		}
	}
	result.Identities = map[string]engines.ModelIdentity{}
	for _, load := range prepared.ModelLoads {
		if conversionDigest == "" && load.ProfileKey != profileKey {
			continue
		}
		list, exists := prepared.evidence.typed.models[load.ProfileKey]
		if !exists || load.Resolved == nil {
			return fail(modelLoadError(load.ProfileKey, "artifacts", "Compiled profile artifact evidence is missing."))
		}
		identity, ok := e.modelIdentityFromArtifacts(p.Engine, load.ModelID, list, int64(load.Resolved.PerSlotContext))
		if !ok {
			return fail(modelLoadError(load.ProfileKey, "artifacts", "Model identity is unavailable."))
		}
		result.Identities[load.ProfileKey] = identity
	}
	current, err := inspectionSnapshot(st, p.Engine)
	if err != nil || !snapshot.equal(current) {
		*retry = true
		return fail(modelLoadError("", "baseline", "Engine settings changed during preview."))
	}
	if err := recheck(); err != nil {
		return fail(err)
	}
	if err := ctx.Err(); err != nil {
		return settings.Preview{}, err
	}
	result.Settings = baseline.Clone()
	result.Settings.ModelLoads, result.Settings.Serving = modelconfig.CloneLoads(prepared.ModelLoads), prepared.Config.Clone()
	command, err = modelConstrainCommand(command, prepared.ModelLoads)
	if err != nil {
		return fail(err)
	}
	result.Settings.LaunchText, err = command.argumentText(llamaServingBase(rt))
	if err != nil {
		return fail(err)
	}
	st.mu.Lock()
	running, adopted := st.running, st.adopted
	st.mu.Unlock()
	if running && adopted {
		return fail(modelLoadError("", "ownership", "Stop this engine in its own application before changing settings."))
	}
	changed := baseline.LaunchText != result.Settings.LaunchText || !baseline.Serving.Equal(result.Settings.Serving) || !modelconfig.EqualLoads(baseline.ModelLoads, result.Settings.ModelLoads)
	result.Restart = running && changed
	models := map[string]bool{}
	for model := range before {
		if model != "*" {
			models[model] = true
		}
	}
	for _, load := range prepared.ModelLoads {
		models[load.ProfileKey] = true
	}
	result.Impact = e.modelLoadImpact(p.Engine, baseline, result.Settings, running, changed, models)
	if rt.LaunchArgs != nil {
		result.Args = slices.Clone(*rt.LaunchArgs)
	}
	result.Env = literalAssignments(command.Env)
	if len(prepared.ModelLoads) != 0 {
		result.Serving = preparedServingReadback(snapshot.state, prepared)
	}
	after, err := parseLlamaServingPresetAllowing(prepared.Snapshot, modelLoadAllowedPresetKeys(prepared.ModelLoads))
	if err != nil {
		return fail(err)
	}
	diffModels := map[string]bool{}
	for model := range before {
		diffModels[model] = true
	}
	for model := range after {
		diffModels[model] = true
	}
	for _, model := range sortedKeys(diffModels) {
		keys := maps.Clone(after[model])
		if keys == nil {
			keys = map[string]string{}
		}
		maps.Copy(keys, before[model])
		for _, key := range sortedKeys(keys) {
			var old, next any
			if value, exists := before[model][key]; exists {
				old = value
			}
			if value, exists := after[model][key]; exists {
				next = value
			}
			a, err := json.Marshal(old)
			if err != nil {
				return fail(err)
			}
			b, err := json.Marshal(next)
			if err != nil {
				return fail(err)
			}
			result.Diff = append(result.Diff, modelconfig.FieldDiff{Field: "models." + model + "." + key, Before: a, After: b})
		}
	}
	a, err := json.Marshal(baseline.LaunchText)
	if err != nil {
		return fail(err)
	}
	b, err := json.Marshal(result.Settings.LaunchText)
	if err != nil {
		return fail(err)
	}
	result.Diff = append(result.Diff, modelconfig.FieldDiff{Field: "launchText", Before: a, After: b})
	result.Diff = append(result.Diff, modelconfig.MovedArtifactDiff(baseline.ModelLoads, prepared.ModelLoads)...)
	result.PreviewDigest, err = modelLoadPreviewDigest(prepared.Digest, conversionDigest, p.ModelChange, baseline)
	if err != nil {
		return fail(err)
	}
	if err := modelconfig.CheckPayloadSize(result); err != nil {
		return fail(err)
	}
	// All authoritative work is already complete. A fresh estimate is left to
	// the caller, after this attempt releases its content lease.
	if p.ModelChange.Remove {
		return result, nil
	}
	if proposal, ok := op.Proposals[profileKey]; ok {
		result.Auto = &proposal
		attachModelLoadEstimate(&result, proposal.Estimate)
	} else {
		in := advisory[profileKey]
		*pending = &in
	}
	return result, nil
}

func (e *Executor) modelLoadImpact(engine string, before, after settings.Config, running, changed bool, models map[string]bool) *modelconfig.Impact {
	if models == nil {
		models = map[string]bool{}
	}
	for _, config := range []settings.Config{before, after} {
		for _, load := range config.ModelLoads {
			models[load.ProfileKey] = true
		}
		if config.Serving != nil {
			for _, model := range config.Serving.Models {
				models[model.ModelID] = true
			}
		}
	}
	timeout := int32(60)
	interrupt, staysOff := running && changed, !running
	if err := e.activationVeto(engine); err != nil {
		staysOff = true
	}
	scope := "none"
	if changed {
		scope = "router"
	}
	impact := &modelconfig.Impact{RestartScope: scope, AffectedModels: sortedKeys(models), DrainTimeoutSeconds: &timeout, MayInterruptActive: &interrupt, StaysOff: &staysOff, ActiveWorkPolicy: "proceed-after-timeout"}
	if running {
		impact.Unknowns = []string{"Loaded-model readback is not part of this native settings preview; the broker includes current runtime observations before acceptance."}
	}
	return impact
}

// Preparation seam for Task 7. It changes selected requested leaves
// only; previous target artifact assertions survive for exact re-resolution.
func patchModelLoad(loads []modelconfig.ModelLoad, profileKey, modelID string, patch modelconfig.LoadPatch) ([]modelconfig.ModelLoad, error) {
	if err := patch.Validate(); err != nil {
		return nil, err
	}
	if err := nodepolicy.ValidateProfileKey(profileKey); err != nil {
		return nil, modelLoadError(profileKey, "profileKey", err.Error())
	}
	if err := validateLlamaModelID(profileKey); err != nil {
		return nil, err
	}
	if err := validateLlamaArtifactModelID(modelID); err != nil {
		return nil, err
	}
	out := modelconfig.CloneLoads(loads)
	i := slices.IndexFunc(out, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == profileKey })
	if i < 0 {
		out = append(out, modelconfig.ModelLoad{ProfileKey: profileKey, ModelID: modelID})
		i = len(out) - 1
	} else if out[i].ModelID != modelID {
		return nil, modelLoadError(profileKey, "modelId", "A profile keeps its model; create a new profile.")
	}
	before, err := json.Marshal(out[i].Requested)
	if err != nil {
		return nil, err
	}
	selected, err := json.Marshal(patch.Set)
	if err != nil {
		return nil, err
	}
	var fields, sets map[string]json.RawMessage
	if err := json.Unmarshal(before, &fields); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(selected, &sets); err != nil {
		return nil, err
	}
	maps.Copy(fields, sets)
	for _, key := range patch.Clear {
		delete(fields, key)
	}
	data, err := json.Marshal(fields)
	if err != nil {
		return nil, err
	}
	var requested modelconfig.RequestedLoad
	if err := modelconfig.DecodeStrict(data, &requested); err != nil {
		return nil, err
	}
	if err := requested.Validate(); err != nil {
		field, reason, ok := strings.Cut(err.Error(), ":")
		if ok {
			return nil, modelLoadError(profileKey, field, strings.TrimSpace(reason))
		}
		return nil, err
	}
	if err := validateModelLoadGPUSelection(profileKey, requested); err != nil {
		return nil, err
	}
	out[i].Requested = requested
	slices.SortFunc(out, func(a, b modelconfig.ModelLoad) int { return strings.Compare(a.ProfileKey, b.ProfileKey) })
	return out, nil
}

func removeModelLoad(loads []modelconfig.ModelLoad, profileKey, modelID string) ([]modelconfig.ModelLoad, error) {
	out := modelconfig.CloneLoads(loads)
	i := slices.IndexFunc(out, func(load modelconfig.ModelLoad) bool { return load.ProfileKey == profileKey })
	if i < 0 {
		return nil, modelLoadError(profileKey, "profileKey", "Profile load does not exist.")
	}
	if out[i].ModelID != modelID {
		return nil, modelLoadError(profileKey, "modelId", "A profile keeps its model; create a new profile.")
	}
	out = slices.Delete(out, i, i+1)
	slices.SortFunc(out, func(a, b modelconfig.ModelLoad) int { return strings.Compare(a.ProfileKey, b.ProfileKey) })
	return out, nil
}

func removeModelLoadDeclaration(serving *engines.ServingConfig, profileKey string) *engines.ServingConfig {
	if serving == nil {
		return nil
	}
	serving = serving.Clone()
	serving.Models = slices.DeleteFunc(serving.Models, func(model engines.ServingModel) bool { return model.ModelID == profileKey })
	if len(serving.Models) == 0 {
		return nil
	}
	return serving
}

// The complete source digest binds the current external baseline during
// conversion. The broker owns acceptance of the selected patch and preview.
type typedLlamaConversion struct {
	Runtime      Runtime
	SourceDigest string
	Diff         []modelconfig.FieldDiff
	Before       map[string]map[string]string
	inputs       *typedLlamaInputEvidence
}

// Configure receives frozen settings and optional parent-accepted removal
// metadata, never a patch to apply again. Reconstruct the complete external
// router and bind the exact selected omission to the original native review.
func (e *Executor) validateTypedLlamaConversion(ctx context.Context, st *engineState, rt Runtime, desired settings.Config, removal *modelconfig.ModelChange) (func() error, error) {
	if removal != nil {
		if err := settings.ValidateReviewedModelChange(removal); err != nil {
			return nil, err
		}
	}
	if len(rt.ModelLoads) != 0 || st.manifest.FamilyOrDefault() != "llamacpp" {
		return nil, nil
	}
	command, err := launchForState(st, rt.Port)
	if err != nil {
		return nil, err
	}
	cli, env, err := llamaLaunchOptions(command, rt.EditableLaunch)
	if err != nil {
		return nil, err
	}
	if cli["models-preset"] == "" && env["models-preset"] == "" {
		return nil, nil
	}
	if len(desired.ModelLoads) == 0 && removal == nil {
		// An ordinary external-router edit keeps an external source. Dropping
		// the complete router into default mode requires the same reviewed
		// selection as removing a section into a nonempty typed configuration.
		preview := e.previewLaunchLocked(st, settings.Request{Engine: st.manifest.Engine, Settings: desired}, false)
		if len(preview.Errors) != 0 || preview.Conflict != nil {
			return nil, modelLoadError("", "conversion", "External router settings failed authoritative validation.")
		}
		next := rt
		next.Port, next.Serving = desired.ServerPort, desired.Serving.Clone()
		next.LaunchArgs, next.LaunchEnv = &preview.Args, &preview.Env
		vars := st.pathVars()
		vars["host"], vars["port"] = effectiveBind(next.Bind, ""), strconv.Itoa(next.Port)
		proposed, err := resolveProcessLaunch(llamaServingBase(next), command.Bin, vars)
		if err != nil {
			return nil, err
		}
		cli, env, err := llamaLaunchOptions(proposed, next.EditableLaunch)
		if err != nil {
			return nil, err
		}
		if cli["models-preset"] == "" && env["models-preset"] == "" {
			return nil, modelLoadError("", "conversion", "Removing an external router requires the reviewed profile selection.")
		}
		return nil, nil
	}
	// Use an immutable baseline: after persistence st contains the typed map.
	baseline := strataPreparationState(st)
	// The accepted target paths and serialized metadata must match before the
	// external preview can sample them. Keep those identities through its reads.
	approvedInputs := &typedLlamaInputEvidence{models: map[string]modelFileList{}}
	approvedInputs.build, err = captureSource(command.Bin, false)
	if err != nil {
		return nil, err
	}
	_, approvedInputs.libraries, err = nativeBuildSnapshotExpected(ctx, approvedInputs.build, servingFileAccess{}, nil)
	if err != nil {
		return nil, modelLoadError("", "nativeBuildId", "Native build content is unavailable or changed.")
	}
	for _, load := range desired.ModelLoads {
		if load.Resolved == nil {
			return nil, modelLoadError(load.ProfileKey, "conversion", "Accepted artifact bindings are missing.")
		}
		list := modelFileList{source: &sourceSnapshot{}}
		for _, artifact := range load.Resolved.Artifacts {
			file, err := captureSource(artifact.Path, false)
			if err != nil || file.target != artifact.ResolvedTarget || strconv.FormatInt(file.size, 10) != artifact.Size || strconv.FormatInt(file.mtime, 10) != artifact.MtimeNS {
				return nil, modelLoadError(load.ProfileKey, "conversion", "Accepted artifacts changed since preview.")
			}
			list.source.artifacts = append(list.source.artifacts, modelFile{configured: file.configured, abs: file.target, info: file.info, Size: file.size})
		}
		approvedInputs.models[load.ProfileKey] = list
	}
	source := cli["models-preset"]
	if source == "" {
		source = env["models-preset"]
	}
	selector, err := captureSource(source, false)
	if err != nil {
		return nil, err
	}
	// This first capture has no saved selector metadata. Subsequent reads keep
	// this object; the complete serialized review digest still checks its bytes.
	approvedInputs.selectors = []sourceEvidence{selector}
	if removal != nil {
		// The removed section has no desired resolution. Resolve only that
		// selection from the checked original preset, then retain its opened
		// objects through complete-router reconstruction and the final CAS.
		access := servingFileAccess{}
		original := access
		access.open = func(path string) (servingArtifactFile, error) {
			if path == selector.configured {
				return original.checkedOpen(ctx, selector.configured, selector.target, selector.info)
			}
			return original.openFile(path)
		}
		store := typedLlamaStore(baseline, command)
		access.scan = &modelArtifactScan{}
		access.recognize = typedLlamaRecognition(e, store)
		list, _, err := resolveLlamaArtifacts(ctx, baseline, command, store, removal.ModelID, access)
		if err != nil {
			return nil, modelLoadError(modelLoadChangeProfileKey(removal), "conversion", "Reviewed removed artifacts are unavailable or changed.")
		}
		approvedInputs.models[modelLoadChangeProfileKey(removal)] = list
	}
	conversion, err := e.previewTypedLlamaConversionExpected(ctx, baseline, command, rt, servingFileAccess{}, approvedInputs)
	if err != nil {
		return nil, err
	}
	removed := false
	for _, load := range conversion.Runtime.ModelLoads {
		if removal != nil && load.ProfileKey == modelLoadChangeProfileKey(removal) {
			if load.ModelID != removal.ModelID || slices.ContainsFunc(desired.ModelLoads, func(next modelconfig.ModelLoad) bool { return next.ProfileKey == load.ProfileKey }) {
				return nil, modelLoadError(load.ProfileKey, "conversion", "Accepted conversion must remove exactly the reviewed profile and model.")
			}
			removed = true
			continue
		}
		resolved := rtLoadResolution(desired.ModelLoads, load.ProfileKey)
		if resolved == nil || modelAutoConversionSource(resolved) != conversion.SourceDigest {
			return nil, modelLoadError(load.ProfileKey, "conversion", "External router source changed since preview; preview again before applying.")
		}
		if resolved == nil || !reflect.DeepEqual(resolved.Artifacts, load.Resolved.Artifacts) || resolved.NativeBuildID != load.Resolved.NativeBuildID {
			return nil, modelLoadError(load.ProfileKey, "conversion", "Accepted conversion must preserve the complete current router artifact set and native build.")
		}
	}
	if removal != nil {
		if !removed || len(desired.ModelLoads) != len(conversion.Runtime.ModelLoads)-1 || desired.ServerPort != rt.Port || !reflect.DeepEqual(desired.WorkingSetLimitMiB, rt.WorkingSetLimitMiB) {
			return nil, modelLoadError(modelLoadChangeProfileKey(removal), "conversion", "Accepted conversion must preserve every unselected router profile.")
		}
		beforeServing := removeModelLoadDeclaration(conversion.Runtime.Serving, modelLoadChangeProfileKey(removal))
		afterServing := desired.Serving.Clone()
		for _, serving := range []*engines.ServingConfig{beforeServing, afterServing} {
			if serving != nil {
				for i := range serving.Models {
					// The normal frozen compiler checks this new digest after
					// removal; every declaration input must otherwise survive.
					serving.Models[i].ExpectedNativeSourceDigest = ""
				}
			}
		}
		if !beforeServing.Equal(afterServing) {
			return nil, modelLoadError(modelLoadChangeProfileKey(removal), "conversion", "Accepted removal must preserve unselected serving declarations.")
		}
		native := ""
		for _, load := range desired.ModelLoads {
			i := slices.IndexFunc(conversion.Runtime.ModelLoads, func(original modelconfig.ModelLoad) bool { return original.ProfileKey == load.ProfileKey })
			if i < 0 || load.ModelID != conversion.Runtime.ModelLoads[i].ModelID || !reflect.DeepEqual(load.Requested, conversion.Runtime.ModelLoads[i].Requested) {
				return nil, modelLoadError(load.ProfileKey, "conversion", "Accepted removal must preserve unselected router settings.")
			}
			if native != "" && native != load.Resolved.SourceInputDigest {
				return nil, modelLoadError(load.ProfileKey, "conversion", "Accepted removal requires one frozen native review.")
			}
			native = load.Resolved.SourceInputDigest
		}
		text, err := command.argumentText(llamaServingBase(rt))
		if err != nil {
			return nil, err
		}
		before := (settings.Config{WorkingSetLimitMiB: rt.WorkingSetLimitMiB, ServerPort: rt.Port, ProxyPort: desired.ProxyPort, LaunchText: text, Serving: rt.Serving, ModelLoads: rt.ModelLoads}).Clone()
		digest, err := modelLoadPreviewDigest(native, conversion.SourceDigest, removal, before)
		if err != nil {
			return nil, err
		}
		if digest != removal.PreviewDigest {
			return nil, modelLoadError(modelLoadChangeProfileKey(removal), "conversion", "External router source or reviewed removal changed since preview; preview again before applying.")
		}
	}
	vars := baseline.pathVars()
	vars["host"], vars["port"] = effectiveBind(rt.Bind, ""), strconv.Itoa(desired.ServerPort)
	converted := conversion.Runtime
	converted.Port = desired.ServerPort
	convertedCommand, err := resolveProcessLaunch(llamaServingBase(converted), command.Bin, vars)
	if err != nil {
		return nil, err
	}
	convertedCommand, err = modelConstrainCommand(convertedCommand, desired.ModelLoads)
	if err != nil {
		return nil, err
	}
	for _, load := range desired.ModelLoads {
		if load.Resolved == nil || !strings.HasPrefix(load.Resolved.ResolverRevision, "auto/") {
			continue
		}
		if convertedCommand.Bin != conversion.inputs.build.configured || validateSource(conversion.inputs.build) != nil {
			return nil, modelLoadError(load.ProfileKey, "nativeBuildId", "Native executable changed during conversion.")
		}
		current, libraries, err := modelReviewedNativeSourceExpected(ctx, st, convertedCommand, servingFileAccess{}, conversion.inputs.build, conversion.inputs.libraries)
		if err != nil || current != modelAutoNativeSource(load.Resolved) {
			return nil, modelLoadError(load.ProfileKey, "conversion", "Reviewed native library/environment source changed.")
		}
		conversion.inputs.libraries = libraries
	}
	text, err := convertedCommand.argumentText(llamaServingBase(converted))
	if err != nil {
		return nil, err
	}
	if text != desired.LaunchText {
		return nil, modelLoadError("", "conversion", "Accepted conversion must preserve unrelated router launch settings.")
	}
	return func() error {
		current, err := e.previewTypedLlamaConversionExpected(ctx, baseline, command, rt, servingFileAccess{}, conversion.inputs)
		if err != nil {
			return err
		}
		if current.SourceDigest != conversion.SourceDigest {
			return modelLoadError("", "conversion", "External router source changed during application; obtain a fresh preview.")
		}
		return nil
	}, nil
}

// Caller holds opMu, or supplies an immutable state snapshot and revalidates
// it at the settings owner before publishing/accepting the returned proposal.
func (e *Executor) previewTypedLlamaConversion(ctx context.Context, st *engineState, command launchCommand, rt Runtime) (typedLlamaConversion, error) {
	return e.previewTypedLlamaConversionExpected(ctx, st, command, rt, servingFileAccess{}, nil)
}

func (e *Executor) previewTypedLlamaConversionExpected(ctx context.Context, st *engineState, command launchCommand, rt Runtime, access servingFileAccess, expected *typedLlamaInputEvidence) (typedLlamaConversion, error) {
	refuse := func(model, field, reason string) (typedLlamaConversion, error) {
		return typedLlamaConversion{}, modelLoadError(model, field, reason)
	}
	if len(rt.ModelLoads) != 0 || st.manifest.FamilyOrDefault() != "llamacpp" {
		return refuse("", "conversion", "Conversion requires an external llama.cpp preset as the single authority.")
	}
	if e.downloads == nil {
		return refuse("", "artifacts", "Model content admission is unavailable.")
	}
	release, recheck, err := e.modelLoadContentRead(st.manifest.Engine)
	if err != nil {
		return refuse("", "artifacts", "Model content is busy or unsafe.")
	}
	defer release()
	ctx, cancel := context.WithTimeout(ctx, modelInspectionBudget)
	defer cancel()
	cli, env, err := llamaLaunchOptions(command, rt.EditableLaunch)
	if err != nil {
		return refuse("", "launchText", err.Error())
	}
	source := cli["models-preset"]
	if source == "" {
		source = env["models-preset"]
	}
	if !filepath.IsAbs(source) {
		return refuse("", "models-preset", "Conversion requires one absolute external preset.")
	}
	if expected != nil {
		if err := expected.checkRecognition(ctx, e); err != nil {
			return refuse("", "artifacts", "Prepared native recognition changed before conversion.")
		}
		access, err = expected.checkedAccess(ctx, access)
		if err != nil || expected.build.configured != command.Bin {
			return refuse("", "conversion", "Prepared external inputs changed before conversion.")
		}
	}
	inputs := &typedLlamaInputEvidence{models: map[string]modelFileList{}}
	if expected != nil {
		inputs.build = expected.build
		inputs.libraries = expected.libraries
	} else {
		inputs.build, err = captureSource(command.Bin, false)
		if err == nil {
			_, inputs.libraries, err = nativeBuildSnapshotExpected(ctx, inputs.build, access, nil)
		}
	}
	if err != nil {
		return refuse("", "nativeBuildId", "Native executable binding is unavailable.")
	}
	var sourceFile sourceEvidence
	if expected != nil {
		if len(expected.selectors) != 1 || expected.selectors[0].configured != source {
			return refuse("", "models-preset", "Prepared external preset selection changed.")
		}
		sourceFile = expected.selectors[0]
	} else {
		sourceFile, err = captureSource(source, false)
	}
	if err != nil {
		return refuse("", "models-preset", "External preset is unavailable.")
	}
	inputs.selectors = []sourceEvidence{sourceFile}
	data, binding, err := access.sourceExpected(ctx, sourceFile, servingSmallFileLimit, false)
	if err != nil {
		return refuse("", "models-preset", "External preset is missing, changed or exceeds its bounded size.")
	}
	// Resolution also reads this preset. Keep its first opened-object evidence
	// while allowing the resolver to capture the initially selected artifacts.
	sourceAccess := access
	access.open = func(path string) (servingArtifactFile, error) {
		if path == sourceFile.configured {
			return sourceAccess.checkedOpen(ctx, sourceFile.configured, sourceFile.target, sourceFile.info)
		}
		return sourceAccess.openFile(path)
	}
	// Conversion validates these fields through RequestedLoad before compiling.
	// Operator launch continues to use the strict parser.
	sections, err := parseLlamaServingPresetAllowing(data, modelConversionAllowedKeys(data))
	if err != nil {
		return refuse("", "conversion", err.Error())
	}
	store := typedLlamaStore(st, command)
	access.scan = &modelArtifactScan{}
	access.recognize = typedLlamaRecognition(e, store)
	converted := rt
	converted.ModelLoads = nil
	converted.modelLoadOwner = e
	before := map[string]map[string]string{}
	for _, model := range sortedKeys(sections) {
		if model == "*" {
			continue
		}
		options := map[string]string{"ctx-size": "8192", "parallel": "1", "kv-unified": "false", "fit": "off", "sleep-idle-seconds": "-1", "cache-ram": "512", "load-on-startup": "false"}
		maps.Copy(options, env)
		maps.Copy(options, sections["*"])
		maps.Copy(options, sections[model])
		maps.Copy(options, cli)
		delete(options, "models-preset")
		delete(options, "models-max")
		before[model] = maps.Clone(options)
		delete(options, "sleep-idle-seconds")
		requested, err := convertTypedLlamaOptions(model, options)
		if err != nil {
			return typedLlamaConversion{}, err
		}
		list, _, err := resolveLlamaArtifacts(ctx, st, command, store, model, access)
		if err != nil {
			return refuse(model, "artifacts", "Complete external artifacts cannot be resolved safely.")
		}
		inputs.models[model] = list
		// This temporary valid record carries only bindings into the canonical
		// resolver. The compiler replaces every finite field and digest below.
		resolved := &modelconfig.ResolvedLoad{PerSlotContext: 8192, TotalContext: 8192, Slots: 1, GPU: modelconfig.GPU{Mode: "cpu"}, NativeBuildID: "binding-only", SourceInputDigest: strings.Repeat("0", 64), PresetDigest: strings.Repeat("0", 64)}
		for _, f := range list.Files {
			b, err := access.bindingExpected(ctx, f.configured, f.abs, f.info)
			if err != nil {
				return refuse(model, "artifacts", "External artifact changed during conversion.")
			}
			resolved.Artifacts = append(resolved.Artifacts, modelconfig.ArtifactBinding{Role: f.Role, Path: b.SourcePath, ResolvedTarget: b.ResolvedTarget, Size: strconv.FormatInt(b.SizeBytes, 10), MtimeNS: b.MtimeNS, SampleSHA256: b.SampleSHA256})
		}
		converted.ModelLoads = append(converted.ModelLoads, modelconfig.ModelLoad{ProfileKey: model, ModelID: model, Requested: requested, Resolved: resolved})
	}
	if len(converted.ModelLoads) == 0 {
		return refuse("", "conversion", "No complete external router model set was found.")
	}
	// Remove only recognized source controls from proposed literal arguments.
	// Unrelated router tokens retain exact order and values.
	args := []string{}
	for i := 0; i < len(command.Args); i++ {
		if rt.EditableLaunch != nil {
			control, _, last, err := rt.EditableLaunch.readControl(command.Args, i)
			if err != nil {
				return typedLlamaConversion{}, err
			}
			if control != nil {
				if _, managed := control.managedValue("127.0.0.1", strconv.Itoa(rt.Port)); managed {
					i = last
					continue
				}
			}
		}
		key, values := llamaLaunchArgument(command.Args[i])
		end := min(i+1+values, len(command.Args))
		option := slices.IndexFunc(llamaServingOptions, func(o llamaServingOption) bool { return o.key == key })
		if option < 0 || key == "models-max" || !strings.HasPrefix(command.Args[i], "-") {
			args = append(args, command.Args[i:end]...)
		}
		i = end - 1
	}
	proposedCommand := command
	proposedCommand.Args = args
	proposedCommand.Env = maps.Clone(command.Env)
	// An inherited allocation variable cannot be removed by saving launch_env.
	// Refuse it rather than pretending the conversion owns the OS environment.
	for _, assignment := range os.Environ() {
		name, _, _ := strings.Cut(assignment, "=")
		for _, option := range llamaServingOptions {
			if name != "" && (name == option.env || name == option.negEnv) && option.key != "models-max" {
				return refuse("", option.key, "Inherited native allocation sources must be removed before conversion.")
			}
		}
	}
	for name := range proposedCommand.Env {
		for _, option := range llamaServingOptions {
			if (name == option.env || name == option.negEnv) && option.key != "models-max" {
				delete(proposedCommand.Env, name)
			}
		}
	}
	converted.LaunchArgs = &args
	environment := literalAssignments(proposedCommand.Env)
	converted.LaunchEnv = &environment
	vars := st.pathVars()
	vars["host"] = effectiveBind(rt.Bind, "")
	vars["port"] = strconv.Itoa(rt.Port)
	proposedCommand, err = resolveProcessLaunch(llamaServingBase(converted), command.Bin, vars)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	prepared, err := compileTypedLlamaInputsExpected(ctx, st, proposedCommand, converted, false, access, inputs)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	inputs.libraries = prepared.evidence.typed.libraries
	converted.ModelLoads, converted.Serving = prepared.ModelLoads, prepared.Config
	after, err := access.bindingExpected(ctx, sourceFile.configured, sourceFile.target, sourceFile.info)
	if err != nil || *after != *binding {
		return refuse("", "models-preset", "External source changed during conversion.")
	}
	review, err := json.Marshal(struct {
		Source *engines.ServingBinding
		Bytes  []byte
		Before launchCommand
		After  []modelconfig.ModelLoad
	}{binding, data, command, prepared.ModelLoads})
	if err != nil {
		return typedLlamaConversion{}, err
	}
	sourceDigest := modelLoadDigest(review)
	for i := range converted.ModelLoads {
		converted.ModelLoads[i].Resolved.ResolverRevision = externalConversionResolver
		converted.ModelLoads[i].Resolved.ReviewedInputID = sourceDigest
	}
	prepared, err = compileTypedLlamaInputsExpected(ctx, st, proposedCommand, converted, false, access, inputs)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	converted.ModelLoads, converted.Serving = prepared.ModelLoads, prepared.Config
	result := typedLlamaConversion{Runtime: converted, SourceDigest: sourceDigest, Before: before, inputs: inputs}
	generated, err := parseLlamaServingPresetAllowing(prepared.Snapshot, modelLoadAllowedPresetKeys(prepared.ModelLoads))
	if err != nil {
		return typedLlamaConversion{}, err
	}
	for _, load := range converted.ModelLoads {
		beforeOptions := before[load.ProfileKey]
		keys := maps.Clone(beforeOptions)
		maps.Copy(keys, generated[load.ProfileKey])
		for _, key := range sortedKeys(keys) {
			before, err := json.Marshal(beforeOptions[key])
			if err != nil {
				return typedLlamaConversion{}, err
			}
			after, err := json.Marshal(generated[load.ProfileKey][key])
			if err != nil {
				return typedLlamaConversion{}, err
			}
			result.Diff = append(result.Diff, modelconfig.FieldDiff{Field: "models." + load.ProfileKey + "." + key, Before: before, After: after})
		}
	}
	beforeArgs, err := json.Marshal(command.Args)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	afterArgs, err := json.Marshal(args)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	result.Diff = append(result.Diff, modelconfig.FieldDiff{Field: "launch_args", Before: beforeArgs, After: afterArgs})
	beforeEnv, err := json.Marshal(command.Env)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	afterEnv, err := json.Marshal(proposedCommand.Env)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	result.Diff = append(result.Diff, modelconfig.FieldDiff{Field: "launch_env", Before: beforeEnv, After: afterEnv})
	if err := recheck(); err != nil {
		return typedLlamaConversion{}, err
	}
	return result, nil
}

func convertTypedLlamaOptions(model string, options map[string]string) (modelconfig.RequestedLoad, error) {
	r := modelconfig.RequestedLoad{}
	for _, key := range sortedKeys(options) {
		value := options[key]
		fail := func() (modelconfig.RequestedLoad, error) {
			return modelconfig.RequestedLoad{}, modelLoadError(model, key, "External field cannot be represented exactly; conversion preserves no partial model subset.")
		}
		integer := func() (*int32, error) {
			n, err := strconv.ParseInt(value, 10, 32)
			if err != nil {
				return nil, err
			}
			v := int32(n)
			return &v, nil
		}
		switch key {
		case "model", "model-draft":
		case "cpu-moe":
			b, err := llamaBool(value)
			if err != nil || b != "true" || r.CPUExperts != nil {
				return fail()
			}
			r.CPUExperts = &modelconfig.CPUExperts{Mode: "all"}
		case "n-cpu-moe":
			n, err := integer()
			if err != nil || *n < 0 || r.CPUExperts != nil {
				return fail()
			}
			r.CPUExperts = &modelconfig.CPUExperts{Mode: "layers", Layers: n}
		case "ctx-size":
			n, err := integer()
			if err != nil {
				return fail()
			}
			r.Context = &modelconfig.Context{Mode: "fixed", Tokens: *n}
		case "parallel", "gpu-layers", "batch-size", "ubatch-size", "threads", "threads-batch", "cache-ram":
			if key == "gpu-layers" && (value == "auto" || value == "all") {
				fit, err := llamaBool(options["fit"])
				if options["fit"] != "" && err != nil || value == "auto" && fit != "true" || value == "all" && fit == "true" {
					return fail()
				}
				r.GPU = &modelconfig.GPU{Mode: "native-default"}
				continue
			}
			n, err := integer()
			if err != nil {
				return fail()
			}
			switch key {
			case "parallel":
				r.Slots = n
			case "gpu-layers":
				if *n < 0 {
					return fail()
				}
				r.GPU = &modelconfig.GPU{Mode: "layers", Layers: n}
				if *n == 0 {
					r.GPU = &modelconfig.GPU{Mode: "cpu"}
				}
			case "batch-size":
				r.BatchSize = n
			case "ubatch-size":
				r.UBatchSize = n
			case "threads":
				r.Threads = n
			case "threads-batch":
				r.ThreadsBatch = n
			case "cache-ram":
				r.CacheRAM = n
			}
		case "cache-type-k":
			r.CacheTypeK = &value
		case "cache-type-v":
			r.CacheTypeV = &value
		case "flash-attn":
			if value == "auto" {
				value = "inherit"
			}
			r.FlashAttention = &value
		case "kv-offload", "op-offload":
			text, err := llamaBool(value)
			if err != nil {
				return fail()
			}
			b := text == "true"
			if key == "kv-offload" {
				r.KVOffload = &b
			} else {
				r.OpOffload = &b
			}
		case "fit":
			b, err := llamaBool(value)
			if err != nil {
				return fail()
			}
			fit := "off"
			if b == "true" {
				fit = "on"
			}
			r.Fit = &fit
		case "fit-target":
			r.FitTarget = &value
		case "fit-ctx":
			n, err := integer()
			if err != nil {
				return fail()
			}
			r.FitCtx = n
		case "load-on-startup", "kv-unified":
			b, err := llamaBool(value)
			if err != nil || b != "false" {
				return fail()
			}
		default:
			if err := validateModelExtraOptions(model, map[string]string{key: value}); err != nil {
				return modelconfig.RequestedLoad{}, err
			}
			if r.ExtraOptions == nil {
				r.ExtraOptions = map[string]string{}
			}
			r.ExtraOptions[key] = value
		}
	}
	slots := int32(1)
	if r.Slots != nil {
		slots = *r.Slots
	}
	if r.Context != nil {
		if slots <= 0 || r.Context.Tokens%slots != 0 {
			return modelconfig.RequestedLoad{}, modelLoadError(model, "context", "External total context must divide exactly into native slots.")
		}
		r.Context.Tokens /= slots
		if r.Context.Tokens%256 != 0 {
			return modelconfig.RequestedLoad{}, modelLoadError(model, "context", "Conversion cannot change native context padding.")
		}
	}
	return r, r.Validate()
}

// Native spellings, aliases, negative switches and environment names belong to
// llamaServingOptions. This projection adds presentation and model ownership,
// never another native parser/flag table. Preset describes the future B compiler
// mapping; Supported describes the B field vocabulary, not apply readiness.
type modelSettingsOption struct {
	Native     llamaServingOption
	Descriptor modelconfig.OptionDescriptor
	Preset     string
}

// Deliberately explicit: adding an actual adapter option without an ownership
// decision fails coverage instead of silently inventing a supported control.
func modelSettingsNativeLabels() map[string]string {
	return map[string]string{
		"models-preset": "Generated model preset", "models-max": "Router model capacity",
		"ctx-size": "Tokens per request", "parallel": "Native slots",
		"kv-unified-per-slot": "Unified KV per-slot cap", "kv-unified": "Unified KV allocation",
		"n-predict": "Native output token limit", "gpu-layers": "GPU placement",
		"split-mode": "GPU split mode", "tensor-split": "GPU tensor proportions",
		"main-gpu": "Main native GPU", "device": "Native devices",
		"backend-sampling": "Engine-side sampling", "sleep-idle-seconds": "Native idle sleep (PAIR-owned)",
		"cache-ram": "Host prompt cache (MiB)", "fit": "Native automatic fitting",
		"cache-prompt": "Prompt caching", "cache-idle-slots": "Idle slot caching",
		"batch-size": "Logical prompt batch", "ubatch-size": "Physical prompt batch",
		"cont-batching": "Continuous batching", "cache-type-k": "K cache precision",
		"cache-type-v": "V cache precision", "swa-full": "Full sliding-window cache",
		"flash-attn": "Flash Attention", "context-shift": "Context shifting",
		"rope-scaling": "RoPE scaling mode", "rope-scale": "RoPE context scale",
		"rope-freq-base": "RoPE frequency base", "rope-freq-scale": "RoPE frequency scale",
		"yarn-orig-ctx": "YaRN original context", "yarn-ext-factor": "YaRN extension factor",
		"yarn-attn-factor": "YaRN attention factor", "yarn-beta-slow": "YaRN slow correction",
		"yarn-beta-fast": "YaRN fast correction", "override-kv": "Metadata overrides",
		"override-tensor": "Tensor placement overrides", "cpu-moe": "CPU expert placement",
		"n-cpu-moe": "CPU expert layers", "n-cpu-ffn": "CPU feed-forward layers",
		"moe-cache-mib": "GPU expert cache (MiB)",
		"kv-offload":    "Offload KV cache", "op-offload": "Offload operations",
		"repack": "Weight repacking", "no-host": "Disable host buffers",
		"load-mode": "Model loading mode", "lazy-mode": "Lazy loading mode",
		"numa": "NUMA placement", "rpc": "Native RPC devices",
		"models-autoload": "Router automatic model loading", "warmup": "Native warmup",
		"model": "Resolved model artifacts", "model-draft": "Resolved draft artifacts",
		"alias": "Native model alias", "mmproj": "Multimodal projector path",
		"mmproj-url": "Multimodal projector URL", "mmproj-auto": "Automatic projector selection",
		"mmproj-offload": "Projector offload", "mmproj-device": "Projector device",
		"image-min-tokens": "Minimum image tokens", "image-max-tokens": "Maximum image tokens",
		"mtmd-batch-max-tokens": "Multimodal batch tokens", "lora": "LoRA adapters",
		"lora-scaled": "Scaled LoRA adapters", "control-vector": "Control vectors",
		"control-vector-scaled": "Scaled control vectors", "control-vector-layer-range": "Control vector layers",
		"models-dir": "Native model directory", "model-url": "Native model URL",
		"hf-repo": "Native repository", "hf-file": "Native repository file",
		"docker-repo": "Native container repository", "hf-repo-draft": "Native draft repository",
		"spec-draft-type-k": "Draft K cache precision", "spec-draft-type-v": "Draft V cache precision",
		"spec-draft-override-tensor": "Draft tensor overrides", "spec-draft-cpu-moe": "Draft CPU experts",
		"spec-draft-n-cpu-moe": "Draft CPU expert layers", "spec-draft-device": "Draft devices",
		"spec-draft-ngl": "Draft GPU layers", "spec-draft-backend-sampling": "Draft engine-side sampling",
		"spec-draft-n-max": "Maximum speculative draft tokens", "spec-type": "Speculation type",
		"spec-default": "Default speculation preset", "mtp": "Multi-token prediction preset",
		"dflash": "DFlash preset", "eagle3": "Eagle3 preset",
		"embd-gemma-default": "Gemma embedding preset", "fim-qwen-1.5b-default": "Qwen 1.5B fill-in-middle preset",
		"fim-qwen-3b-default": "Qwen 3B fill-in-middle preset", "fim-qwen-7b-default": "Qwen 7B fill-in-middle preset",
		"fim-qwen-7b-spec": "Qwen 7B speculative preset", "fim-qwen-14b-spec": "Qwen 14B speculative preset",
		"fim-qwen-30b-default": "Qwen 30B fill-in-middle preset", "gpt-oss-20b-default": "GPT OSS 20B preset",
		"gpt-oss-120b-default": "GPT OSS 120B preset", "vision-gemma-4b-default": "Gemma 4B vision preset",
		"vision-gemma-12b-default": "Gemma 12B vision preset", "fit-target": "Native fit memory target",
		"fit-ctx": "Native fit context target", "cache-reuse": "Prompt cache reuse",
		"ctx-checkpoints": "Context checkpoints", "checkpoint-min-step": "Minimum checkpoint spacing",
		"prompt-cache": "Prompt cache file", "prompt-cache-all": "Cache all prompts",
		"prompt-cache-ro": "Read-only prompt cache", "slot-save-path": "Slot save directory",
		"load-on-startup": "Startup model loading",
	}
}

func modelSettingsNumber(value string) *json.Number {
	n := json.Number(value)
	return &n
}

func modelSettingsInteger(field, label, minimum, defaultValue, defaultSource, effect string) modelconfig.OptionDescriptor {
	d := modelconfig.OptionDescriptor{Field: field, Label: label, Kind: "integer", Minimum: modelSettingsNumber(minimum), Maximum: modelSettingsNumber("2147483647"), Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: defaultSource, IdentityEffect: "native allocation", EstimateEffect: effect}
	if defaultValue != "" {
		d.Default = json.RawMessage(defaultValue)
	}
	return d
}

func llamaModelSettingsOptions() ([]modelSettingsOption, error) {
	labels := modelSettingsNativeLabels()
	options := make([]modelSettingsOption, 0, len(llamaServingOptions))
	for _, native := range llamaServingOptions {
		label, ok := labels[native.key]
		if !ok {
			return nil, fmt.Errorf("model options: missing metadata for %s", native.key)
		}
		d := modelconfig.OptionDescriptor{Field: "native." + native.key, Label: label, Kind: "string", Supported: false, Reason: "No typed override; requires a mapped native build and identity/estimate support.", DefaultSource: "inherited native/model (not inspected)", Owner: "model-load", Phase: "D", IdentityEffect: "native semantics", EstimateEffect: "unknown when selected"}
		if native.bool {
			d.Kind = "boolean"
		}
		if native.unsupported {
			d.Reason = "Native finite adapter rejects this control; typed override unavailable."
		}
		if native.unpricedExtra {
			d.Reason = "Accepted only as an unpriced extra option; no typed override."
		}
		preset := "none: typed override refused"
		switch native.key {
		case "ctx-size":
			d = modelSettingsInteger("context", label, "1", "8192", "finite adapter inherited baseline (not Auto)", "KV/state/workspace")
			d.Maximum = modelSettingsNumber("2147483392")
			preset = "ctx-size = checked padded tokens per request * slots; non-unified KV"
		case "parallel":
			d = modelSettingsInteger("slots", label, "1", "1", "finite adapter inherited baseline", "KV/state/workspace")
			preset = "parallel = slots"
		case "gpu-layers":
			d = modelconfig.OptionDescriptor{Field: "gpu", Label: label, Kind: "enum", Values: []string{"native-default", "cpu", "layers"}, Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native placement; unresolved", IdentityEffect: "placement", EstimateEffect: "host/device weights and KV"}
			preset = "gpu-layers = 0 for CPU, checked finite layers; inherited native-default must resolve; Auto unavailable"
		case "cpu-moe":
			d = modelconfig.OptionDescriptor{Field: "cpuExperts", Label: "CPU expert placement", Kind: "enum", Values: []string{"none", "all", "layers"}, Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native placement; no expert override", IdentityEffect: "expert placement", EstimateEffect: "host/device expert weights; changed expert compute mapping unpriced"}
			preset = "none emits neither expert flag; all emits cpu-moe = true; layers emits n-cpu-moe = N"
		case "n-cpu-moe":
			d = modelSettingsInteger("cpuExperts.layers", label, "0", "", "required only for layers; bounded by target block count", "host/device expert weights; changed expert compute mapping unpriced")
			preset = "n-cpu-moe = explicit nonnegative int32 not exceeding target block count"
		case "cache-type-k", "cache-type-v":
			field := "cacheTypeK"
			if native.key == "cache-type-v" {
				field = "cacheTypeV"
			}
			d = modelconfig.OptionDescriptor{Field: field, Label: label, Kind: "enum", Values: []string{"f16", "q8_0"}, Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native precision; unresolved", IdentityEffect: "KV precision/quality", EstimateEffect: "block-aware KV row sizes"}
			preset = native.key + " = explicit precision; bf16/q4_0 need validated build/architecture support"
		case "flash-attn":
			d = modelconfig.OptionDescriptor{Field: "flashAttention", Label: label, Kind: "enum", Values: []string{"inherit", "on", "off"}, Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native policy; unresolved", IdentityEffect: "attention implementation", EstimateEffect: "KV/workspace support"}
			preset = "flash-attn = on/off when explicit; inherit emits no override; validate target combination"
		case "batch-size", "ubatch-size":
			field := "batchSize"
			if native.key == "ubatch-size" {
				field = "ubatchSize"
			}
			d = modelSettingsInteger(field, label, "1", "", "inherited native default; unresolved", "workspace; ubatch must not exceed batch")
			preset = native.key + " = explicit positive int32"
		case "kv-offload", "op-offload":
			field := "kvOffload"
			if native.key == "op-offload" {
				field = "opOffload"
			}
			d = modelconfig.OptionDescriptor{Field: field, Label: label, Kind: "boolean", Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native policy; unresolved", IdentityEffect: "placement", EstimateEffect: "host/device allocation"}
			preset = native.key + " = explicit true/false"
		case "sleep-idle-seconds":
			d.Owner, d.Phase = "manager/router", "B"
			d.Reason = "PAIR owns model idle release; native idle sleep is disabled."
			d.IdentityEffect, d.EstimateEffect = "manager-owned fixed output", "PAIR per-model idle release"
			preset = "sleep-idle-seconds = -1"
		case "cache-ram":
			d = modelSettingsInteger("cacheRAM", label, "0", "512", "finite adapter inherited baseline (MiB)", "host prompt cache; not admission memory")
			preset = "cache-ram = explicit nonnegative int32 MiB; zero disables the cache"
		case "fit":
			d = modelconfig.OptionDescriptor{Field: "fit", Label: label, Kind: "enum", Values: []string{"off", "on"}, Default: json.RawMessage(`"off"`), DefaultSource: "typed load baseline", Supported: true, Portable: true, Owner: "model-load", Phase: "B", IdentityEffect: "native placement", EstimateEffect: "unpriced when on"}
			preset = "fit = off/on; native fit is unpriced"
		case "fit-target":
			d = modelconfig.OptionDescriptor{Field: "fitTarget", Label: label, Kind: "string", Supported: true, Portable: false, Owner: "model-load", Phase: "B", DefaultSource: "native default; unresolved", IdentityEffect: "native placement", EstimateEffect: "unpriced"}
			preset = "fit-target = comma-separated MiB margins; requires fit on"
		case "fit-ctx":
			d = modelSettingsInteger("fitCtx", label, "0", "", "native fit minimum context; unresolved", "unpriced")
			maximum := json.Number("16777216")
			d.Maximum = &maximum
			preset = "fit-ctx = minimum context; requires fit on"
		case "models-preset", "models-max", "models-autoload", "load-on-startup":
			d.Owner, d.Phase = "manager/router", "B"
			d.Reason = "Manager-owned output/router control; no model-local override."
			d.IdentityEffect, d.EstimateEffect = "router startup", "router capacity and startup effects"
			if native.key == "load-on-startup" {
				d.Kind, d.Default, d.DefaultSource = "boolean", json.RawMessage(`false`), "finite adapter enforced output"
				preset = "load-on-startup = false"
			} else if native.key == "models-preset" {
				preset = "manager-generated immutable preset path"
			} else if native.key == "models-max" {
				preset = "models-max = 0 (PAIR admission owns capacity)"
			}
		case "model", "model-draft", "alias", "models-dir", "model-url", "hf-repo", "hf-file", "docker-repo", "hf-repo-draft":
			d.Owner = "artifacts"
			d.IdentityEffect, d.EstimateEffect = "ordered artifact identity", "weights and companions"
			d.Reason = "Target resolver owns paths and exact identity; no portable path override."
			if native.key == "model" || native.key == "model-draft" {
				d.Phase, preset = "B", "target-resolved ordered model/draft shards; preserve supported draft declarations"
			}
		case "kv-unified", "kv-unified-per-slot":
			d.Reason = "Existing unified records remain readable; typed edits require D support."
			d.EstimateEffect = "KV sharing and slot arithmetic"
		case "n-predict":
			d.Owner, d.IdentityEffect, d.EstimateEffect = "request/native", "request semantics", "output work; not KV context"
		}
		// Typed fit controls are validated before generated reparsing. The
		// operator preset table still rejects them under its finite contract.
		if native.unsupported && d.Supported && native.key != "fit-target" && native.key != "fit-ctx" {
			d.Supported, d.Portable = false, false
			d.Reason = "Native finite adapter rejects this control; typed override unavailable."
		}
		if err := d.Validate(); err != nil {
			return nil, err
		}
		options = append(options, modelSettingsOption{Native: native, Descriptor: d, Preset: preset})
	}
	if err := validateModelSettingsOptionCoverage(llamaServingOptions, options); err != nil {
		return nil, err
	}
	return options, nil
}

// Compare against actual native entries, including every alias/env/negative flag
// and rejection/extra-option metadata. Tests also pass a projection with a removed actual row.
func validateModelSettingsOptionCoverage(native []llamaServingOption, options []modelSettingsOption) error {
	byKey := make(map[string]llamaServingOption, len(options))
	for _, option := range options {
		if _, duplicate := byKey[option.Native.key]; duplicate {
			return fmt.Errorf("model options: duplicate native key %s", option.Native.key)
		}
		byKey[option.Native.key] = option.Native
	}
	for _, want := range native {
		got, ok := byKey[want.key]
		if !ok || !slices.Equal(got.flags, want.flags) || !slices.Equal(got.negativeFlags, want.negativeFlags) || got.env != want.env || got.negEnv != want.negEnv || got.bool != want.bool || got.unsupported != want.unsupported || got.unpricedExtra != want.unpricedExtra {
			return fmt.Errorf("model options: missing or mismatched native option %s", want.key)
		}
	}
	if len(byKey) != len(native) {
		return fmt.Errorf("model options: unexpected native option")
	}
	return nil
}

// Controls outside the allocation parser have request/template ownership. Their
// descriptors add no native spelling table and never materialize recommendations.
func modelSettingsExtraOptions() []modelconfig.OptionDescriptor {
	options := []modelconfig.OptionDescriptor{
		{Field: "extraOptions", Label: "Extra options", Kind: "options", Supported: true, Portable: true, NonPortableKeys: slices.Clone(llamaGPUSelectionKeys), Owner: "model-load", IdentityEffect: "native options", EstimateEffect: "unpriced options"},
		modelSettingsInteger("threads", "Generation threads", "1", "", "inherited native default; unresolved", "CPU work/workspace; no core-count tuning"),
		modelSettingsInteger("threadsBatch", "Prompt-processing threads", "1", "", "inherited native threads default; unresolved", "CPU work/workspace"),
		modelSettingsInteger("gpu.layers", "Finite GPU layers (target facts required)", "0", "", "inherited native placement; unresolved", "host/device weights; target layer count bounds this value"),
	}
	for _, field := range []struct{ key, label string }{{"cacheTypeK.alternatives", "Additional K cache precisions"}, {"cacheTypeV.alternatives", "Additional V cache precisions"}} {
		options = append(options, modelconfig.OptionDescriptor{Field: field.key, Label: field.label, Kind: "enum", Values: []string{"bf16", "q4_0"}, Supported: false, Reason: "Validated native build/architecture support unavailable; quantized V also requires validated Flash Attention.", Owner: "model-load", Phase: "B conditional", DefaultSource: "inherited native precision; unresolved", IdentityEffect: "KV precision/quality", EstimateEffect: "block-aware KV sizes"})
	}
	for _, field := range []struct{ key, label string }{{"context.auto", "Auto context"}, {"gpu.auto", "Auto placement"}} {
		options = append(options, modelconfig.OptionDescriptor{Field: field.key, Label: field.label, Kind: "enum", Values: []string{"auto"}, Supported: false, Reason: "Reviewed Auto resolver and owner trade choices are pending; no defaults selected.", Owner: "model-load", Phase: "C pending O2", DefaultSource: "none selected", IdentityEffect: "frozen reviewed native resolution", EstimateEffect: "requires bounded facts and mapped target memory"})
	}
	for _, field := range []struct{ key, label, kind, min, max string }{
		{"temperature", "Temperature", "number", "0", ""},
		{"top_p", "Top P", "number", "0", "1"},
		{"top_k", "Top K", "integer", "0", "9007199254740991"},
		{"min_p", "Min P", "number", "0", "1"},
		{"repeat_penalty", "Repeat penalty (positive)", "number", "0", ""},
		{"frequency_penalty", "Frequency penalty", "number", "-2", "2"},
		{"presence_penalty", "Presence penalty", "number", "-2", "2"},
		{"seed", "Seed", "integer", "-9007199254740991", "9007199254740991"},
	} {
		d := modelconfig.OptionDescriptor{Field: "generation." + field.key, Label: field.label, Kind: field.kind, Minimum: modelSettingsNumber(field.min), Supported: true, Portable: true, Owner: "request-policy", Phase: "A", DefaultSource: "inherited native/model recommendation; absent remains absent", IdentityEffect: "request/qualification fingerprint", EstimateEffect: "no native load allocation"}
		if field.max != "" {
			d.Maximum = modelSettingsNumber(field.max)
		}
		options = append(options, d)
	}
	options = append(options, modelconfig.OptionDescriptor{Field: "requestOptionsMode", Label: "Generation preference mode", Kind: "enum", Values: []string{"defaults", "enforced"}, Supported: true, Portable: true, Owner: "request-policy", Phase: "A", DefaultSource: "existing policy", IdentityEffect: "request/qualification fingerprint", EstimateEffect: "no native load allocation"})
	for _, field := range []struct{ key, label string }{
		{"samplers", "Additional samplers"}, {"output", "Output/stop options"},
		{"jinja", "Jinja rendering"}, {"chatTemplate", "Chat template"}, {"chatTemplateFile", "Chat template file"},
		{"chatTemplateKwargs", "Template arguments / thinking"}, {"skipChatParsing", "Chat parsing"},
		{"reasoning", "Reasoning mode"}, {"reasoningFormat", "Reasoning format"}, {"reasoningEffort", "Reasoning effort"},
		{"reasoningBudget", "Reasoning token budget"}, {"reasoningBudgetMessage", "Reasoning budget message"},
		{"reasoningPreserve", "Preserve reasoning history"}, {"grammar", "Grammar / structured output"},
	} {
		options = append(options, modelconfig.OptionDescriptor{Field: field.key, Label: field.label, Kind: "string", Supported: false, Reason: "Inherited behavior only; typed override requires pinned request/native mapping and identity support.", Owner: "request/template", Phase: "D", DefaultSource: "embedded/native behavior; not inspected", IdentityEffect: "prompt/request semantics", EstimateEffect: "unknown extra token/work cost"})
	}
	return options
}
