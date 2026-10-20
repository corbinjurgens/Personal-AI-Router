// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"nvpair-shared/engines"
)

type strataRuntimeHash struct {
	info os.FileInfo
	sum  string
}

var strataRuntimeHashes = struct {
	sync.Mutex
	files map[string]strataRuntimeHash
}{files: map[string]strataRuntimeHash{}}

func strataTarget(path string) (string, error) {
	target, err := filepath.EvalSymlinks(path)
	if runtime.GOOS == "windows" {
		target = strings.ToLower(target)
	}
	return target, err
}

// Check every declaration before any full read, including aliases of pack and
// draft artifacts whose extensions do not identify them as model content.
func strataRuntimePaths(config *engines.ServingConfig) ([]string, error) {
	models := map[string]bool{}
	for _, model := range config.Models {
		for _, artifact := range model.Artifacts {
			if artifact.Binding == nil {
				return nil, servingError("Missing Strata artifact binding.")
			}
			if !strings.HasPrefix(artifact.Role, "runtime:") {
				target, err := strataTarget(artifact.Binding.SourcePath)
				if err != nil {
					return nil, err
				}
				models[target] = true
			}
		}
	}
	var paths []string
	for _, model := range config.Models {
		for _, artifact := range model.Artifacts {
			if !strings.HasPrefix(artifact.Role, "runtime:") {
				continue
			}
			path := artifact.Binding.SourcePath
			target, err := strataTarget(path)
			if err != nil {
				return nil, err
			}
			if models[target] || strings.EqualFold(filepath.Ext(path), ".gguf") || strings.EqualFold(filepath.Ext(target), ".gguf") {
				return nil, servingError("Strata runtime artifacts must not reference model targets or GGUF files.")
			}
			paths = append(paths, path)
		}
	}
	return paths, nil
}

// Only this preparation phase may stream runtime files. It runs without opMu;
// compilation under opMu only checks the resulting size/mtime/file identity.
func prepareStrataRuntimeHashes(ctx context.Context, config *engines.ServingConfig) error {
	if config == nil {
		return nil
	}
	paths, err := strataRuntimePaths(config)
	if err != nil {
		return err
	}
	for _, path := range paths {
		if err := ctx.Err(); err != nil {
			return err
		}
		before, err := os.Stat(path)
		if err != nil {
			return err
		}
		strataRuntimeHashes.Lock()
		previous, known := strataRuntimeHashes.files[path]
		strataRuntimeHashes.Unlock()
		replaced := !known || !os.SameFile(before, previous.info)
		// A waiter on an existing shared hash must also be cancellable.
		type result struct {
			sum string
			err error
		}
		done := make(chan result, 1)
		go func() {
			// The shared cache keys size/mtime, so a replaced inode must not
			// reuse either its cached digest or an older in-flight read.
			if replaced {
				sum, err := hashFile(ctx, path, nil)
				done <- result{sum, err}
				return
			}
			sum, err := fileHashes.sum(ctx, path, nil)
			done <- result{sum, err}
		}()
		var hashed result
		select {
		case <-ctx.Done():
			return ctx.Err()
		case hashed = <-done:
		}
		if hashed.err != nil {
			return hashed.err
		}
		after, err := os.Stat(path)
		if err != nil || !os.SameFile(before, after) || before.Size() != after.Size() || before.ModTime() != after.ModTime() {
			return servingError("Strata runtime changed during preparation.")
		}
		if replaced {
			fileHashes.remember(path, hashed.sum)
		}
		strataRuntimeHashes.Lock()
		strataRuntimeHashes.files[path] = strataRuntimeHash{after, hashed.sum}
		strataRuntimeHashes.Unlock()
	}
	return nil
}

func strataRuntimeDigest(path string) (string, error) {
	info, err := os.Stat(path)
	strataRuntimeHashes.Lock()
	cached, ok := strataRuntimeHashes.files[path]
	strataRuntimeHashes.Unlock()
	if err != nil || !ok || !os.SameFile(info, cached.info) || info.Size() != cached.info.Size() || info.ModTime() != cached.info.ModTime() {
		return "", servingError("Strata runtime changed during preparation; obtain a fresh preview.")
	}
	return cached.sum, nil
}

// A supplied config, including nil to disable serving, is the proposed state.
// Without one, prepare the current state for a lifecycle operation or readback.
func prepareStrataState(ctx context.Context, st *engineState, requested ...*engines.ServingConfig) error {
	st.mu.Lock()
	current := st.plat.Runtime.Serving.Clone()
	strata := st.manifest.FamilyOrDefault() == "strata"
	st.mu.Unlock()
	if !strata {
		return nil
	}
	if len(requested) > 0 {
		current = requested[0]
	}
	return prepareStrataRuntimeHashes(ctx, current)
}

// Prepared subset of corbinjurgens/Strata f6726098 serve/server.py and
// src/program/generate.cpp. Automatic fitting, mutable profiles, hooks, vision,
// native_experts indirection and implicit allocation controls are unsupported.
type strataServingConfig struct {
	Exe         string   `json:"exe"`
	Args        []string `json:"args"`
	CWD         string   `json:"cwd"`
	Tokenizer   string   `json:"tokenizer"`
	ModelName   string   `json:"model_name"`
	Parallel    int64    `json:"parallel"`
	LibDirs     []string `json:"lib_dirs"`
	IdleUnloadS int64    `json:"idle_unload_s"`
}

func strataServingRuntime(st *engineState, config *engines.ServingConfig) bool {
	return config != nil && st.manifest.FamilyOrDefault() == "strata"
}

func preparedServingRuntime(st *engineState, config *engines.ServingConfig) bool {
	return llamaServingRuntime(st, config) || strataServingRuntime(st, config)
}

func compilePreparedServing(st *engineState, command launchCommand, rt Runtime, approved bool) (*llamaServingLaunch, error) {
	if strataServingRuntime(st, rt.Serving) {
		return compileStrataServing(command, rt, approved)
	}
	return compileLlamaServing(command, rt, approved)
}

func preparedServingReadback(st *engineState, prepared *llamaServingLaunch) *engines.ServingReadback {
	result := llamaServingReadback(prepared)
	if strataServingRuntime(st, prepared.Config) {
		config, _ := parseStrataServingConfig(prepared.Snapshot)
		result.Reason = "Native launch configured; loaded limits and content qualification remain unverified. Library search paths (lib_dirs: " + strings.Join(config.LibDirs, ", ") + "; inherited PATH/LD_LIBRARY_PATH) and Python imports beyond declared files are unattested."
	}
	return result
}

func strataDirectory(path string) error {
	info, err := os.Stat(path)
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || err != nil || !info.IsDir() {
		return servingError("Prepared Strata directories must be absolute existing directories.")
	}
	return nil
}

// Only exact argparse spellings are accepted; abbreviation and Python switches
// can select another program or silently bypass the reviewed configuration.
func strataLaunchSource(command launchCommand) (string, error) {
	if len(command.Args) < 1 || !filepath.IsAbs(command.Bin) || !filepath.IsAbs(command.Args[0]) {
		return "", servingError("Prepared Strata requires absolute Python and server script paths.")
	}
	values := map[string]string{}
	for i := 1; i < len(command.Args); i++ {
		key := command.Args[i]
		if _, duplicate := values[key]; duplicate {
			return "", servingError("Duplicate Strata server arguments are unsupported.")
		}
		switch key {
		case "--lazy":
			values[key] = "true"
		case "--engine", "--config", "--host", "--port":
			if i+1 == len(command.Args) {
				return "", servingError("Missing Strata server argument value.")
			}
			i++
			values[key] = command.Args[i]
		default:
			return "", servingError("Unsupported Strata Python/server launch mode or argument.")
		}
	}
	if values["--engine"] != "strata" || values["--lazy"] != "true" || values["--host"] != "127.0.0.1" || !filepath.IsAbs(values["--config"]) {
		return "", servingError("Prepared Strata requires native mode, explicit loopback, an absolute config and lazy HTTP startup.")
	}
	if n, err := strconv.Atoi(values["--port"]); err != nil || n < 1 || n > 65535 {
		return "", servingError("Prepared Strata requires an explicit server port.")
	}
	// Native STRATA_* and Python injection controls have many build-specific
	// spellings. Fail closed on their presence instead of guessing precedence.
	env := map[string]string{}
	for _, assignment := range os.Environ() {
		key, value, _ := strings.Cut(assignment, "=")
		env[environmentKey(key)] = value
	}
	for key, value := range command.Env {
		env[environmentKey(key)] = value
	}
	for key := range env {
		key = strings.ToUpper(key)
		if strings.HasPrefix(key, "STRATA_") || strings.HasPrefix(key, "PYTHON") {
			return "", servingError("Strata/Python environment overrides are unsupported in prepared mode.")
		}
	}
	return values["--config"], nil
}

// Decode a flat, explicit config, rejecting duplicate keys before typed decoding.
func parseStrataServingConfig(data []byte) (strataServingConfig, error) {
	var config strataServingConfig
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return config, servingError("Prepared Strata config must be a JSON object.")
	}
	seen := map[string]bool{}
	for decoder.More() {
		token, err := decoder.Token()
		key, ok := token.(string)
		if err != nil || !ok || seen[key] {
			return config, servingError("Duplicate or malformed Strata config key.")
		}
		seen[key] = true
		var value json.RawMessage
		if decoder.Decode(&value) != nil || bytes.Equal(value, []byte("null")) {
			return config, servingError("Null or malformed Strata config values are unsupported.")
		}
	}
	decoder = json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&config) != nil || decoder.Decode(new(json.RawMessage)) != io.EOF {
		return config, servingError("Unsupported or malformed prepared Strata config.")
	}
	return config, nil
}

func compileStrataServing(command launchCommand, rt Runtime, approved bool) (*llamaServingLaunch, error) {
	if rt.modeOrDefault() != "process" || rt.Serving == nil || len(rt.Serving.Validate()) != 0 || len(rt.Serving.Models) != 1 {
		return nil, servingError("Prepared Strata requires one declared model and an owned process launch.")
	}
	if _, err := strataRuntimePaths(rt.Serving); err != nil {
		return nil, err
	}
	source, err := strataLaunchSource(command)
	if err != nil {
		return nil, err
	}
	data, binding, err := servingSource(source)
	if err != nil {
		return nil, servingError("Strata config is missing, unstable or oversized.")
	}
	config, err := parseStrataServingConfig(data)
	if err != nil {
		return nil, err
	}
	if err := strataDirectory(config.CWD); err != nil {
		return nil, err
	}
	for _, dir := range config.LibDirs {
		if err := strataDirectory(dir); err != nil {
			return nil, err
		}
	}
	if config.Parallel < 1 || config.Parallel > 8 || config.IdleUnloadS < 0 {
		return nil, servingError("Strata requires explicit parallel in 1..8 and nonnegative idle retention.")
	}
	result := &llamaServingLaunch{Config: rt.Serving.Clone(), Source: source}
	model := &result.Config.Models[0]
	if config.ModelName != model.ModelID {
		return nil, servingError("Strata model_name must exactly match the declared model.")
	}
	// Paths map to their required role. Directories are never artifacts: their
	// selected files are enumerated below, without recursive discovery.
	paths := map[string]string{}
	add := func(path, role string) error {
		if !filepath.IsAbs(path) || filepath.Clean(path) != path {
			return servingError("Explicit Strata file paths must be absolute and unambiguous.")
		}
		if previous, exists := paths[path]; exists && previous != role {
			return servingError("Strata file has ambiguous artifact roles.")
		}
		paths[path] = role
		return nil
	}
	for _, entry := range []struct{ path, role string }{{command.Bin, "runtime:python"}, {command.Args[0], "runtime:server"}, {source, "runtime:config"}, {config.Exe, "runtime:native-exe"}} {
		if err := add(entry.path, entry.role); err != nil {
			return nil, err
		}
	}
	options := map[string]string{}
	placement := []string{}
	for i := 0; i < len(config.Args); i++ {
		key := config.Args[i]
		canonical := key
		if key == "--slots" {
			canonical = "--batch"
		}
		if _, exists := options[canonical]; exists {
			return nil, servingError("Duplicate or conflicting native Strata arguments.")
		}
		if key == "--serve" {
			options[key] = "true"
			continue
		}
		if i+1 == len(config.Args) {
			return nil, servingError("Missing native Strata argument value.")
		}
		i++
		value := config.Args[i]
		options[canonical] = value
		switch key {
		case "--max-context", "--batch":
		case "--slots":
		case "--native", "--ple-gguf", "--native-head-gguf", "--embd-gguf":
			shards, err := llamaArtifactPaths(value)
			if err != nil {
				return nil, err
			}
			for _, path := range shards {
				if err := add(path, "model"); err != nil {
					return nil, err
				}
			}
		case "--expert-profile":
			if err := add(value, "runtime:profile"); err != nil {
				return nil, err
			}
		case "--pack", "--mtp":
			if err := strataDirectory(value); err != nil {
				return nil, err
			}
			files := []string{"index.txt", "dense.bin", "experts.bin"}
			role := "model"
			if key == "--mtp" {
				files = []string{"dense.txt", "dense.bin", "experts.bin"}
				role = "draft"
			}
			for _, name := range files {
				if err := add(filepath.Join(value, name), role); err != nil {
					return nil, err
				}
			}
			for _, name := range []string{"embd.bin", "extra.bin", "manifest.json", "draft_vocab.bin"} {
				path := filepath.Join(value, name)
				if _, err := os.Lstat(path); err == nil {
					if err := add(path, role); err != nil {
						return nil, err
					}
				} else if !os.IsNotExist(err) {
					return nil, servingError("Cannot inspect prepared pack file.")
				}
			}
			if _, err := os.Lstat(filepath.Join(value, "native_experts.txt")); !os.IsNotExist(err) {
				return nil, servingError("Pack native_experts indirection is unsupported.")
			}
		case "--prefill", "--vram-reserve-mib", "--expert-cache", "--spec":
			if _, err := llamaPositive(value); err != nil {
				return nil, err
			}
			placement = append(placement, key+"="+value)
		default:
			return nil, servingError("Unsupported native Strata argument; implicit paths, fitting and allocation modes are forbidden.")
		}
	}
	contextTokens, ctxErr := llamaPositive(options["--max-context"])
	slots := config.Parallel
	if explicit, exists := options["--batch"]; exists {
		n, err := llamaPositive(explicit)
		if err != nil || n != slots || n < 2 {
			return nil, servingError("Native batch/slots must agree with config parallel without clamping.")
		}
	}
	// Pinned generate.cpp f6726098 rejects max_context > 2147483647LL - 8.
	if ctxErr != nil || contextTokens > (math.MaxInt32-8)/slots || contextTokens <= model.MaxOutputTokens || options["--serve"] != "true" || options["--native"] == "" || options["--pack"] == "" {
		return nil, servingError("Strata requires explicit finite context, native model, pack and serve mode; output must fit the per-slot context.")
	}
	if options["--expert-cache"] == "" || options["--prefill"] == "" {
		return nil, servingError("Strata requires explicit fixed expert cache and prefill placement.")
	}
	if n, err := llamaPositive(options["--spec"]); err != nil || n < 2 || n > 8 {
		return nil, servingError("Strata requires an explicit supported speculative window in 2..8.")
	}
	if err := strataDirectory(config.Tokenizer); err != nil {
		return nil, err
	}
	for _, file := range []string{"vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"} {
		if err := add(filepath.Join(config.Tokenizer, file), "runtime:tokenizer:"+file); err != nil {
			return nil, err
		}
	}
	declared := map[string]bool{}
	for i := range model.Artifacts {
		artifact := &model.Artifacts[i]
		if artifact.Binding == nil || !filepath.IsAbs(artifact.Binding.SourcePath) || declared[artifact.Binding.SourcePath] {
			return nil, servingError("Every Strata artifact requires a unique absolute declared file binding.")
		}
		path := artifact.Binding.SourcePath
		role, selected := paths[path]
		if selected && ((role == "model" && artifact.Role != "model" && artifact.Role != "model-shard" && !strings.HasPrefix(artifact.Role, "model:")) || (role != "model" && artifact.Role != role)) {
			return nil, servingError("Selected Strata file has an incorrect artifact role.")
		}
		if !selected && !strings.HasPrefix(artifact.Role, "runtime:") {
			return nil, servingError("Unselected Strata model artifact.")
		}
		current, err := servingFileBinding(path)
		if err != nil || *current != *artifact.Binding {
			return nil, servingError("Strata artifact binding changed or is missing.")
		}
		if strings.HasPrefix(artifact.Role, "runtime:") {
			sum, err := strataRuntimeDigest(path)
			if err != nil {
				return nil, err
			}
			after, bindingErr := servingFileBinding(path)
			if err != nil || sum != artifact.SHA256 || bindingErr != nil || *after != *current {
				return nil, servingError("Strata runtime full SHA256 mismatch or unstable file.")
			}
		}
		declared[path] = true
	}
	for path := range paths {
		if !declared[path] {
			return nil, servingError("Explicit Strata dependency has no declared artifact.")
		}
	}
	result.Snapshot, err = json.Marshal(config)
	if err != nil || len(result.Snapshot) > servingSmallFileLimit {
		return nil, servingError("Cannot prepare bounded Strata config snapshot.")
	}
	model.ExpectedNativeSourceDigest = ""
	encoded, err := json.Marshal(struct {
		Version int
		Source  *engines.ServingBinding
		Bytes   []byte
		Config  *engines.ServingConfig
		Command launchCommand
	}{1, binding, data, result.Config, command})
	if err != nil {
		return nil, servingError("Cannot bind prepared Strata inputs.")
	}
	digest := sha256.Sum256(encoded)
	result.Digest = hex.EncodeToString(digest[:])
	expected := rt.Serving.Models[0].ExpectedNativeSourceDigest
	if (approved && expected == "") || (expected != "" && expected != result.Digest) {
		return nil, servingError("Strata native source conflict; obtain a fresh preview.")
	}
	model.ExpectedNativeSourceDigest = result.Digest
	result.Envelopes = []engines.ServingEnvelope{{ModelID: model.ModelID, PerSlotContextTokens: contextTokens, TotalContextTokens: contextTokens * slots, Slots: slots, MaxOutputTokens: model.MaxOutputTokens, Memory: model.Memory, Placement: placement, Retention: strconv.FormatInt(config.IdleUnloadS, 10), RouterCapacity: 1, NativeSourceDigest: result.Digest}}
	return result, nil
}

func recheckPreparedServing(st *engineState, launch launchCommand, rt Runtime, prepared *llamaServingLaunch) error {
	current, err := compilePreparedServing(st, launch, rt, true)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(current, prepared) {
		return servingError("Native serving inputs changed before spawn.")
	}
	return nil
}

func preparedSnapshotCommand(st *engineState, command launchCommand, snapshot string) launchCommand {
	if strataServingRuntime(st, st.plat.Runtime.Serving) {
		command.Args = append([]string(nil), command.Args...)
		for i := 1; i+1 < len(command.Args); i++ {
			if command.Args[i] == "--config" {
				command.Args[i+1] = snapshot
				break
			}
		}
		return command
	}
	return llamaSnapshotCommand(command, snapshot)
}
