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
)

const externalConversionResolver = "external-conversion/llama"

func modelLoadError(model, field, reason string) error {
	return &ServingFieldError{Field: "modelLoads." + field, ModelID: model, Reason: reason}
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
func compileTypedLlama(st *engineState, command launchCommand, rt Runtime, approved bool) (*llamaServingLaunch, error) {
	if st.manifest.FamilyOrDefault() != "llamacpp" || rt.modeOrDefault() != "process" {
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
	ctx, cancel := context.WithTimeout(context.Background(), modelInspectionBudget)
	defer cancel()
	prepared, err := compileTypedLlamaInputs(ctx, st, command, rt, approved)
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
	capacity := cli["models-max"]
	if capacity == "" {
		capacity = env["models-max"]
	}
	if capacity == "" {
		capacity = "1"
	}
	return llamaPositive(capacity)
}

func literalAssignments(env map[string]string) []string {
	values := make([]string, 0, len(env))
	for _, key := range sortedKeys(env) {
		values = append(values, key+"="+env[key])
	}
	return values
}

func compileTypedLlamaInputs(ctx context.Context, st *engineState, command launchCommand, rt Runtime, approved bool) (*llamaServingLaunch, error) {
	capacity, err := typedLlamaLaunchOptions(command, rt)
	if err != nil {
		return nil, err
	}
	requests := modelconfig.CloneLoads(rt.ModelLoads)
	for i := range requests {
		requests[i].Resolved = nil
	}
	loads, err := modelconfig.CanonicalLoads(requests)
	if err != nil || len(loads) == 0 {
		return nil, modelLoadError("", "requested", "A valid complete typed model set is required.")
	}
	build, err := servingFileBinding(command.Bin)
	if err != nil {
		return nil, modelLoadError("", "nativeBuildId", "Native executable binding is unavailable.")
	}
	buildBytes, err := json.Marshal(build)
	if err != nil {
		return nil, err
	}
	buildID := modelLoadDigest(buildBytes)
	store := engineModelStore{engine: st.manifest.Engine, adapter: "llamacpp", root: st.modelsDir}
	if cache := os.Getenv("LLAMA_CACHE"); cache != "" {
		store.root = expandPath(cache)
	}
	for name, value := range command.Env {
		if environmentKey(name) == environmentKey("LLAMA_CACHE") && value != "" {
			store.root = expandPath(value)
		}
	}
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
		load.Resolved = rtLoadResolution(rt.ModelLoads, load.ModelID)
	}
	plat := *st.plat
	plat.Runtime = resolutionRT
	resolver := &engineState{manifest: st.manifest, plat: &plat, modelsDir: st.modelsDir}
	access := servingFileAccess{scan: &modelArtifactScan{}}
	sections := map[string]map[string]string{}
	lists := make([]modelFileList, len(loads))
	for i := range loads {
		load := &loads[i]
		if err := validateLlamaModelID(load.ModelID); err != nil {
			return nil, modelLoadError(load.ModelID, "modelId", "Use an exact canonical native model ID.")
		}
		list, _, err := resolveLlamaArtifacts(ctx, resolver, command, store, load.ModelID, access)
		if err != nil {
			return nil, modelLoadError(load.ModelID, "artifacts", "Exact artifacts are missing, ambiguous, changed or exceed the bounded resolver limits.")
		}
		facts, factsErr := decodeInspectionFacts(ctx, list, access)
		var layerCount *int32
		if factsErr == nil {
			projected, _ := projectInspectionFacts(facts)
			layerCount = projected.LayerCount
		}
		options, resolved, err := typedLlamaModelOptions(load.ModelID, load.Requested, layerCount)
		if err != nil {
			return nil, err
		}
		for _, file := range list.Files {
			if file.Role != "model" && file.Role != "draft" {
				return nil, modelLoadError(load.ModelID, "artifacts", "Companion artifacts have no typed native mapping.")
			}
			binding, err := access.binding(ctx, file.configured)
			if err != nil {
				return nil, modelLoadError(load.ModelID, "artifacts", "Artifact changed during compilation.")
			}
			resolved.Artifacts = append(resolved.Artifacts, modelconfig.ArtifactBinding{Role: file.Role, Path: binding.SourcePath, ResolvedTarget: binding.ResolvedTarget, Size: strconv.FormatInt(binding.SizeBytes, 10), MtimeNS: binding.MtimeNS, SampleSHA256: binding.SampleSHA256})
			key := "model"
			if file.Role == "draft" {
				key = "model-draft"
			}
			if options[key] == "" {
				options[key] = file.configured
			}
		}
		resolved.NativeBuildID = buildID
		if previous := rtLoadResolution(rt.ModelLoads, load.ModelID); previous != nil {
			resolved.ResolverRevision = previous.ResolverRevision
			resolved.ReviewedInputID = previous.ReviewedInputID
		}
		sections[load.ModelID], lists[i] = options, list
		load.Resolved = resolved
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
			sections[model.ModelID]["n-predict"] = strconv.FormatInt(model.MaxOutputTokens, 10)
		}
		data, err = typedLlamaPreset(sections)
		if err != nil {
			return nil, err
		}
		result, err = compileLlamaServingSections(declaredRT, false, "", data, nil, sections, nil, nil, capacity)
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
				return nil, modelLoadError(result.Config.Models[i].ModelID, "serving", "Declaration requires a fresh typed preview.")
			}
			result.Config.Models[i].ExpectedNativeSourceDigest = result.Digest
			result.Envelopes[i].NativeSourceDigest = result.Digest
		}
	}
	for i := range loads {
		loads[i].Resolved.SourceInputDigest = result.Digest
		loads[i].Resolved.PresetDigest = presetDigest
		if approved && !reflect.DeepEqual(loads[i].Resolved, rtLoadResolution(rt.ModelLoads, loads[i].ModelID)) {
			return nil, modelLoadError(loads[i].ModelID, "resolved", "Typed inputs changed; obtain a fresh preview before starting.")
		}
	}
	// Re-resolve through Task 5 after facts and samples. Do not trust the first
	// resolution or a cached metadata read as source revalidation.
	for i, load := range loads {
		after, _, err := resolveLlamaArtifacts(ctx, resolver, command, store, load.ModelID, access)
		if err != nil || after.binding != lists[i].binding || !sameTypedFiles(lists[i], after) {
			return nil, modelLoadError(load.ModelID, "artifacts", "Artifact source changed during compilation.")
		}
	}
	if current, err := servingFileBinding(command.Bin); err != nil || *current != *build {
		return nil, modelLoadError("", "nativeBuildId", "Native executable changed during compilation.")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := modelconfig.CanonicalLoads(loads); err != nil {
		return nil, err
	}
	return result, nil
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

func rtLoadResolution(loads []modelconfig.ModelLoad, id string) *modelconfig.ResolvedLoad {
	for _, load := range loads {
		if load.ModelID == id {
			return load.Resolved
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

func typedLlamaModelOptions(model string, requested modelconfig.RequestedLoad, layers *int32) (map[string]string, *modelconfig.ResolvedLoad, error) {
	fail := func(field, reason string) (map[string]string, *modelconfig.ResolvedLoad, error) {
		return nil, nil, modelLoadError(model, field, reason)
	}
	if err := requested.Validate(); err != nil {
		return fail("requested", err.Error())
	}
	tokens, slots := int32(8192), int32(1)
	if requested.Context != nil {
		if requested.Context.Mode == "auto" {
			return fail("context", "Reviewed Auto is unavailable.")
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
		return fail("gpu", "Reviewed Auto is unavailable.")
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
	options := map[string]string{"ctx-size": strconv.Itoa(int(total)), "parallel": strconv.Itoa(int(slots)), "kv-unified": "false", "fit": "off", "sleep-idle-seconds": "300", "cache-ram": "512", "load-on-startup": "false", "gpu-layers": "0"}
	if gpu.Mode == "layers" {
		options["gpu-layers"] = strconv.Itoa(int(*gpu.Layers))
	}
	for _, field := range []struct {
		name, key string
		value     *string
	}{{"cacheTypeK", "cache-type-k", requested.CacheTypeK}, {"cacheTypeV", "cache-type-v", requested.CacheTypeV}} {
		if field.value != nil {
			if *field.value != "f16" {
				return fail(field.name, "This precision requires validated build/architecture support; only f16 is currently mapped.")
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
	}{{"batch-size", requested.BatchSize}, {"ubatch-size", requested.UBatchSize}, {"threads", requested.Threads}, {"threads-batch", requested.ThreadsBatch}, {"sleep-idle-seconds", requested.SleepIdleSeconds}, {"cache-ram", requested.CacheRAM}} {
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
	ctx, cancel := context.WithTimeout(ctx, modelInspectionBudget)
	defer cancel()
	result := settings.Preview{Settings: p.Settings.Clone(), Args: []string{}, Env: []string{}, Errors: map[string]string{}}
	fail := func(err error) (settings.Preview, error) {
		if ctx.Err() != nil {
			return settings.Preview{}, ctx.Err()
		}
		field := "modelLoads"
		var typed *ServingFieldError
		if errors.As(err, &typed) {
			field = typed.Field
		}
		result.Errors[field] = err.Error()
		return result, nil
	}
	if err := p.Validate(); err != nil {
		return fail(err)
	}
	if st.manifest.FamilyOrDefault() != "llamacpp" || p.ModelChange == nil || p.ModelChange.Load == nil || p.ModelChange.Policy != nil {
		return fail(modelLoadError("", "request", "Only a local llama.cpp load-only preview is available."))
	}
	if !e.modelInspection.active.CompareAndSwap(false, true) {
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
	baseline := settings.Config{ServerPort: rt.Port, ProxyPort: p.Settings.ProxyPort, LaunchText: text, Serving: rt.Serving.Clone(), ModelLoads: modelconfig.CloneLoads(rt.ModelLoads)}
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
		prepared, err := compileTypedLlamaInputs(ctx, snapshot.state, snapshot.command, rt, false)
		if err != nil {
			return fail(err)
		}
		before, err = parseLlamaServingPreset(prepared.Snapshot)
		if err != nil {
			return fail(err)
		}
	}
	rt.ModelLoads, err = patchModelLoad(rt.ModelLoads, p.ModelChange.ModelID, *p.ModelChange.Load)
	if err != nil {
		return fail(err)
	}
	vars := snapshot.state.pathVars()
	vars["host"], vars["port"] = effectiveBind(rt.Bind, ""), strconv.Itoa(rt.Port)
	command, err := resolveProcessLaunch(llamaServingBase(rt), snapshot.command.Bin, vars)
	if err != nil {
		return fail(err)
	}
	prepared, err := compileTypedLlamaInputs(ctx, snapshot.state, command, rt, false)
	if err != nil {
		return fail(err)
	}
	current, err := inspectionSnapshot(st, p.Engine)
	if err != nil || !snapshot.equal(current) {
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
		models[model] = true
	}
	for _, load := range prepared.ModelLoads {
		models[load.ModelID] = true
	}
	result.Impact = e.modelLoadImpact(p.Engine, baseline, result.Settings, running, changed, models)
	if rt.LaunchArgs != nil {
		result.Args = slices.Clone(*rt.LaunchArgs)
	}
	result.Env = literalAssignments(command.Env)
	result.Serving = preparedServingReadback(snapshot.state, prepared)
	after, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		return fail(err)
	}
	for _, model := range sortedKeys(after) {
		keys := maps.Clone(after[model])
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
	identity, err := json.Marshal(struct {
		Native, Conversion string
		Patch              *modelconfig.LoadPatch
	}{prepared.Digest, conversionDigest, p.ModelChange.Load})
	if err != nil {
		return fail(err)
	}
	result.PreviewDigest = modelLoadDigest(identity)
	if err := modelconfig.CheckPayloadSize(result); err != nil {
		return fail(err)
	}
	return result, nil
}

func (e *Executor) modelLoadImpact(engine string, before, after settings.Config, running, changed bool, models map[string]bool) *modelconfig.Impact {
	if models == nil {
		models = map[string]bool{}
	}
	for _, config := range []settings.Config{before, after} {
		for _, load := range config.ModelLoads {
			models[load.ModelID] = true
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
func patchModelLoad(loads []modelconfig.ModelLoad, model string, patch modelconfig.LoadPatch) ([]modelconfig.ModelLoad, error) {
	if err := patch.Validate(); err != nil {
		return nil, err
	}
	out := modelconfig.CloneLoads(loads)
	i := slices.IndexFunc(out, func(load modelconfig.ModelLoad) bool { return load.ModelID == model })
	if i < 0 {
		out = append(out, modelconfig.ModelLoad{ModelID: model})
		i = len(out) - 1
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
		return nil, err
	}
	out[i].Requested = requested
	slices.SortFunc(out, func(a, b modelconfig.ModelLoad) int { return strings.Compare(a.ModelID, b.ModelID) })
	return out, nil
}

// The complete source digest binds the current external baseline during
// conversion. The broker owns acceptance of the selected patch and preview.
type typedLlamaConversion struct {
	Runtime      Runtime
	SourceDigest string
	Diff         []modelconfig.FieldDiff
	Before       map[string]map[string]string
}

// Configure receives frozen settings, not the selected patch. Reconstruct the
// complete current external router at this boundary, preserve its artifact set
// and unrelated launch inputs, then let the normal compiler validate the exact
// reviewed requested/resolved identity. Never use generated output as input.
func (e *Executor) validateTypedLlamaConversion(ctx context.Context, st *engineState, rt Runtime, desired settings.Config) (func() error, error) {
	if len(rt.ModelLoads) != 0 || len(desired.ModelLoads) == 0 || st.manifest.FamilyOrDefault() != "llamacpp" {
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
	// Use an immutable baseline: after persistence st contains the typed map.
	baseline := strataPreparationState(st)
	conversion, err := e.previewTypedLlamaConversion(ctx, baseline, command, rt)
	if err != nil {
		return nil, err
	}
	for _, load := range conversion.Runtime.ModelLoads {
		resolved := rtLoadResolution(desired.ModelLoads, load.ModelID)
		if resolved == nil || resolved.ResolverRevision != externalConversionResolver || resolved.ReviewedInputID != conversion.SourceDigest {
			return nil, modelLoadError(load.ModelID, "conversion", "External router source changed since preview; preview again before applying.")
		}
		if resolved == nil || !reflect.DeepEqual(resolved.Artifacts, load.Resolved.Artifacts) || resolved.NativeBuildID != load.Resolved.NativeBuildID {
			return nil, modelLoadError(load.ModelID, "conversion", "Accepted conversion must preserve the complete current router artifact set and native build.")
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
	text, err := convertedCommand.argumentText(llamaServingBase(converted))
	if err != nil {
		return nil, err
	}
	if text != desired.LaunchText {
		return nil, modelLoadError("", "conversion", "Accepted conversion must preserve unrelated router launch settings.")
	}
	return func() error {
		current, err := e.previewTypedLlamaConversion(ctx, baseline, command, rt)
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
	data, binding, err := (servingFileAccess{}).source(ctx, source)
	if err != nil {
		return refuse("", "models-preset", "External preset is missing, changed or exceeds its bounded size.")
	}
	sections, err := parseLlamaServingPreset(data)
	if err != nil {
		return refuse("", "conversion", err.Error())
	}
	store := engineModelStore{engine: st.manifest.Engine, adapter: "llamacpp", root: st.modelsDir}
	if cache := os.Getenv("LLAMA_CACHE"); cache != "" {
		store.root = expandPath(cache)
	}
	for name, value := range command.Env {
		if environmentKey(name) == environmentKey("LLAMA_CACHE") && value != "" {
			store.root = expandPath(value)
		}
	}
	access := servingFileAccess{scan: &modelArtifactScan{}}
	converted := rt
	converted.ModelLoads = nil
	converted.modelLoadOwner = e
	before := map[string]map[string]string{}
	for _, model := range sortedKeys(sections) {
		if model == "*" {
			continue
		}
		options := map[string]string{"ctx-size": "8192", "parallel": "1", "kv-unified": "false", "fit": "off", "sleep-idle-seconds": "300", "cache-ram": "512", "load-on-startup": "false"}
		maps.Copy(options, env)
		maps.Copy(options, sections["*"])
		maps.Copy(options, sections[model])
		maps.Copy(options, cli)
		delete(options, "models-preset")
		delete(options, "models-max")
		before[model] = maps.Clone(options)
		requested, err := convertTypedLlamaOptions(model, options)
		if err != nil {
			return typedLlamaConversion{}, err
		}
		list, _, err := resolveLlamaArtifacts(ctx, st, command, store, model, access)
		if err != nil {
			return refuse(model, "artifacts", "Complete external artifacts cannot be resolved safely.")
		}
		// This temporary valid record carries only bindings into the canonical
		// resolver. The compiler replaces every finite field and digest below.
		resolved := &modelconfig.ResolvedLoad{PerSlotContext: 8192, TotalContext: 8192, Slots: 1, GPU: modelconfig.GPU{Mode: "cpu"}, NativeBuildID: "binding-only", SourceInputDigest: strings.Repeat("0", 64), PresetDigest: strings.Repeat("0", 64)}
		for _, f := range list.Files {
			b, err := access.binding(ctx, f.configured)
			if err != nil {
				return refuse(model, "artifacts", "External artifact changed during conversion.")
			}
			resolved.Artifacts = append(resolved.Artifacts, modelconfig.ArtifactBinding{Role: f.Role, Path: b.SourcePath, ResolvedTarget: b.ResolvedTarget, Size: strconv.FormatInt(b.SizeBytes, 10), MtimeNS: b.MtimeNS, SampleSHA256: b.SampleSHA256})
		}
		converted.ModelLoads = append(converted.ModelLoads, modelconfig.ModelLoad{ModelID: model, Requested: requested, Resolved: resolved})
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
	prepared, err := compileTypedLlamaInputs(ctx, st, proposedCommand, converted, false)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	converted.ModelLoads, converted.Serving = prepared.ModelLoads, prepared.Config
	after, err := (servingFileAccess{}).binding(ctx, source)
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
	prepared, err = compileTypedLlamaInputs(ctx, st, proposedCommand, converted, false)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	converted.ModelLoads, converted.Serving = prepared.ModelLoads, prepared.Config
	result := typedLlamaConversion{Runtime: converted, SourceDigest: sourceDigest, Before: before}
	generated, err := parseLlamaServingPreset(prepared.Snapshot)
	if err != nil {
		return typedLlamaConversion{}, err
	}
	for _, load := range converted.ModelLoads {
		beforeOptions := before[load.ModelID]
		keys := maps.Clone(beforeOptions)
		maps.Copy(keys, generated[load.ModelID])
		for _, key := range sortedKeys(keys) {
			before, err := json.Marshal(beforeOptions[key])
			if err != nil {
				return typedLlamaConversion{}, err
			}
			after, err := json.Marshal(generated[load.ModelID][key])
			if err != nil {
				return typedLlamaConversion{}, err
			}
			result.Diff = append(result.Diff, modelconfig.FieldDiff{Field: "models." + load.ModelID + "." + key, Before: before, After: after})
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
		case "ctx-size":
			n, err := integer()
			if err != nil {
				return fail()
			}
			r.Context = &modelconfig.Context{Mode: "fixed", Tokens: *n}
		case "parallel", "gpu-layers", "batch-size", "ubatch-size", "threads", "threads-batch", "sleep-idle-seconds", "cache-ram":
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
			case "sleep-idle-seconds":
				r.SleepIdleSeconds = n
			case "cache-ram":
				r.CacheRAM = n
			}
		case "cache-type-k":
			r.CacheTypeK = &value
		case "cache-type-v":
			r.CacheTypeV = &value
		case "flash-attn":
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
			if value != "off" {
				return fail()
			}
		case "load-on-startup", "kv-unified":
			b, err := llamaBool(value)
			if err != nil || b != "false" {
				return fail()
			}
		default:
			return fail()
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
		"backend-sampling": "Engine-side sampling", "sleep-idle-seconds": "Unload after idle (seconds; -1 or positive)",
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
		"kv-offload": "Offload KV cache", "op-offload": "Offload operations",
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
		case "cache-type-k", "cache-type-v":
			field := "cacheTypeK"
			if native.key == "cache-type-v" {
				field = "cacheTypeV"
			}
			d = modelconfig.OptionDescriptor{Field: field, Label: label, Kind: "enum", Values: []string{"f16"}, Supported: true, Portable: true, Owner: "model-load", Phase: "B", DefaultSource: "inherited native precision; unresolved", IdentityEffect: "KV precision/quality", EstimateEffect: "block-aware KV row sizes"}
			preset = native.key + " = explicit precision; bf16/q8_0/q4_0 need validated build/architecture support"
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
			d = modelSettingsInteger("sleepIdleSeconds", label, "-1", "300", "finite adapter inherited baseline", "retention; keep-loaded policy takes precedence")
			preset = "sleep-idle-seconds = -1 or positive int32 (zero invalid)"
		case "cache-ram":
			d = modelSettingsInteger("cacheRAM", label, "1", "512", "finite adapter inherited baseline (MiB)", "host prompt cache; not admission memory")
			preset = "cache-ram = explicit positive int32 MiB"
		case "models-preset", "models-max", "models-autoload", "fit", "fit-target", "fit-ctx", "load-on-startup":
			d.Owner, d.Phase = "manager/router", "B"
			d.Reason = "Manager-owned output/router control; no model-local override."
			d.IdentityEffect, d.EstimateEffect = "router startup", "router capacity and startup effects"
			if native.key == "fit" {
				d.Kind, d.Default, d.DefaultSource = "enum", json.RawMessage(`"off"`), "finite adapter enforced output"
				d.Values, preset = []string{"off"}, "fit = off; reviewed Auto resolves before compilation"
			} else if native.key == "load-on-startup" {
				d.Kind, d.Default, d.DefaultSource = "boolean", json.RawMessage(`false`), "finite adapter enforced output"
				preset = "load-on-startup = false"
			} else if native.key == "models-preset" {
				preset = "manager-generated immutable preset path"
			} else if native.key == "models-max" {
				preset = "preserve existing finite router capacity"
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
		if native.unsupported && d.Supported {
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
// and rejection bit. Tests also pass a projection with a removed actual row.
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
		if !ok || !slices.Equal(got.flags, want.flags) || !slices.Equal(got.negativeFlags, want.negativeFlags) || got.env != want.env || got.negEnv != want.negEnv || got.bool != want.bool || got.unsupported != want.unsupported {
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
		modelSettingsInteger("threads", "Generation threads", "1", "", "inherited native default; unresolved", "CPU work/workspace; no core-count tuning"),
		modelSettingsInteger("threadsBatch", "Prompt-processing threads", "1", "", "inherited native threads default; unresolved", "CPU work/workspace"),
		modelSettingsInteger("gpu.layers", "Finite GPU layers (target facts required)", "0", "", "inherited native placement; unresolved", "host/device weights; target layer count bounds this value"),
	}
	for _, field := range []struct{ key, label string }{{"cacheTypeK.alternatives", "Additional K cache precisions"}, {"cacheTypeV.alternatives", "Additional V cache precisions"}} {
		options = append(options, modelconfig.OptionDescriptor{Field: field.key, Label: field.label, Kind: "enum", Values: []string{"bf16", "q8_0", "q4_0"}, Supported: false, Reason: "Validated native build/architecture support unavailable; quantized V also requires validated Flash Attention.", Owner: "model-load", Phase: "B conditional", DefaultSource: "inherited native precision; unresolved", IdentityEffect: "KV precision/quality", EstimateEffect: "block-aware KV sizes"})
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
