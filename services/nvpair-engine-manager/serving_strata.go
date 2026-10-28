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
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	"nvpair-shared/engines"
	settings "nvpair-shared/enginesettings"
	"nvpair-shared/nodepolicy"
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
	var modelFiles []os.FileInfo
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
				info, err := os.Stat(target)
				if err != nil {
					return nil, err
				}
				modelFiles = append(modelFiles, info)
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
			info, err := os.Stat(target)
			if err != nil {
				return nil, err
			}
			// Hard links can name model content through an unrelated runtime path.
			for _, modelFile := range modelFiles {
				if os.SameFile(info, modelFile) {
					return nil, servingError("Strata runtime artifacts must not reference model targets or GGUF files.")
				}
			}
			paths = append(paths, path)
		}
	}
	return paths, nil
}

// Only this preparation phase may stream runtime files. It runs without opMu;
// compilation under opMu only checks the resulting size/mtime/file identity.
func prepareStrataRuntimeHashes(ctx context.Context, command launchCommand, rt Runtime) error {
	if rt.Serving == nil {
		return nil
	}
	prepared, err := deriveStrataServing(command, rt)
	if err != nil {
		return err
	}
	paths, err := strataRuntimePaths(prepared.Config)
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

// Snapshot only launch inputs under mu; never carry a lifecycle lock into I/O.
func strataPreparationState(st *engineState) *engineState {
	st.mu.Lock()
	defer st.mu.Unlock()
	plat := *st.plat
	plat.Runtime.Serving = plat.Runtime.Serving.Clone()
	return &engineState{manifest: st.manifest, plat: &plat, binPath: st.binPath, installDir: st.installDir, modelsDir: st.modelsDir}
}

func prepareStrataState(ctx context.Context, st *engineState) error {
	snapshot := strataPreparationState(st)
	if snapshot.manifest.FamilyOrDefault() != "strata" || snapshot.plat.Runtime.Serving == nil {
		return nil
	}
	command, err := launchForState(snapshot, snapshot.plat.Runtime.Port)
	if err != nil {
		return err
	}
	return prepareStrataRuntimeHashes(ctx, command, snapshot.plat.Runtime)
}

func (e *Executor) prepareStrataRequest(ctx context.Context, st *engineState, request settings.Request) error {
	snapshot := strataPreparationState(st)
	if snapshot.manifest.FamilyOrDefault() != "strata" || request.Settings.Serving == nil {
		return nil
	}
	var command launchCommand
	var rt Runtime
	preview := e.previewLaunch(snapshot, request, func(candidate launchCommand, proposed Runtime) (*llamaServingLaunch, error) {
		command, rt = candidate, proposed
		return deriveStrataServing(candidate, proposed)
	})
	if len(preview.Errors) > 0 || preview.Conflict != nil {
		// Nothing may be hashed. The ordinary locked preview repeats the same
		// validation and preserves its field errors and port-conflict response.
		return nil
	}
	return prepareStrataRuntimeHashes(ctx, command, rt)
}

// Prepared subset of corbinjurgens/Strata serve/server.py and
// src/program/generate.cpp, from f6726098 to the current main 0e70753c.
// Automatic fitting, the automatic memory policy, authentication, mutable
// profiles, hooks and vision are unsupported.
//
// The fork's generated profile also carries fields beyond the original subset.
// Log, Env, GPU and FitMaxTokens are read by the server and reach it through the
// snapshot. BuildMetadata, GPUsAsked, Port, Host and OpenBrowser are validated
// and then dropped from the snapshot: the server reads none of them or lets the
// command line decide, and PAIR owns that command line.
type strataServingConfig struct {
	Exe          string            `json:"exe"`
	Args         []string          `json:"args"`
	CWD          string            `json:"cwd"`
	Tokenizer    string            `json:"tokenizer"`
	ModelName    string            `json:"model_name"`
	Parallel     int64             `json:"parallel"`
	LibDirs      []string          `json:"lib_dirs"`
	IdleUnloadS  int64             `json:"idle_unload_s"`
	Log          *string           `json:"log,omitempty"`
	Env          map[string]string `json:"env,omitempty"`
	GPU          *int64            `json:"gpu,omitempty"`
	FitMaxTokens *bool             `json:"fit_max_tokens,omitempty"`

	// APIKeyFile names the file holding the native API key. An inline
	// api_key is rejected: the config's bytes are hashed into advertised
	// fields, so a key there could be guessed offline from them. The key file
	// is bound by identity only and is never forwarded; the server gets the key
	// as STRATA_API_KEY, and PAIR's own requests to this engine carry it.
	APIKeyFile *string `json:"api_key_file,omitempty"`

	BuildMetadata *string `json:"build_metadata,omitempty"`
	GPUsAsked     *bool   `json:"gpus_asked,omitempty"`
	Port          *int64  `json:"port,omitempty"`
	Host          *string `json:"host,omitempty"`
	OpenBrowser   *bool   `json:"open_browser,omitempty"`
}

// Only the memory controls the fork's planner generates are accepted in the
// config env block, each with the syntax it writes. Anything else could change
// native behavior outside the reviewed profile.
var strataEnvironment = map[string]func(string) bool{
	"STRATA_RESIDENT_HEADROOM_GIB": func(v string) bool { return strataDecimal(v, true) },
	"STRATA_FILE_CACHE_MIB": func(v string) bool {
		n, err := strconv.ParseUint(v, 10, 64)
		return err == nil && strconv.FormatUint(n, 10) == v && n <= math.MaxUint64>>20
	},
	"STRATA_FILE_CACHE_COMPACT": func(v string) bool { return v == "0" || v == "1" },
	"STRATA_OWNED_PRICE":        func(v string) bool { return v == "exact" },
}

var strataKVFormats = []string{"fp16", "int8", "q4_0", "q4", "k8v4"}

// strataAPIKeyPattern is printable ASCII without spaces or commas, which the
// server would read as a list of keys.
var strataAPIKeyPattern = regexp.MustCompile(`^[\x21-\x2b\x2d-\x7e]{16,512}$`)

var strataDecimalPattern = regexp.MustCompile(`^[0-9]{1,6}(\.[0-9]{1,6})?$`)

// strataDecimal accepts a plain decimal number up to 4096; zero only when allowed.
func strataDecimal(value string, allowZero bool) bool {
	if !strataDecimalPattern.MatchString(value) {
		return false
	}
	n, err := strconv.ParseFloat(value, 64)
	return err == nil && n <= 4096 && (n > 0 || allowZero)
}

func strataUnreviewed(field string) error {
	return servingError("Strata config field " + field + " is not a reviewed value.")
}

// validateStrataExtras checks the fields beyond the original subset. The log
// is the only one that names a file: it must be an absolute clean path whose
// directory exists and must not be a declared input.
func validateStrataExtras(config strataServingConfig) error {
	if config.GPU != nil && (*config.GPU < 0 || *config.GPU > 255) {
		return strataUnreviewed("gpu")
	}
	if config.Port != nil && (*config.Port < 1 || *config.Port > 65535) {
		return strataUnreviewed("port")
	}
	if config.Host != nil && *config.Host != "127.0.0.1" {
		return strataUnreviewed("host")
	}
	if config.BuildMetadata != nil && *config.BuildMetadata == "" {
		return strataUnreviewed("build_metadata")
	}
	if config.OpenBrowser != nil && *config.OpenBrowser {
		return strataUnreviewed("open_browser")
	}
	if config.Log != nil {
		path := *config.Log
		if !filepath.IsAbs(path) || filepath.Clean(path) != path || strataDirectory(filepath.Dir(path)) != nil {
			return servingError("Strata log must be an absolute clean file path in an existing directory.")
		}
	}
	for name, value := range config.Env {
		if valid, reviewed := strataEnvironment[name]; !reviewed || !valid(value) {
			return strataUnreviewed("env")
		}
	}
	return nil
}

func strataServingRuntime(st *engineState, config *engines.ServingConfig) bool {
	return config != nil && st.manifest.FamilyOrDefault() == "strata"
}

func preparedServingRuntime(st *engineState, rt Runtime) bool {
	return llamaServingRuntime(st, rt) || strataServingRuntime(st, rt.Serving)
}

func compilePreparedServing(st *engineState, command launchCommand, rt Runtime, approved bool, advisory ...*map[string]modelEstimateInput) (*llamaServingLaunch, error) {
	if len(rt.ModelLoads) != 0 {
		return compileTypedLlama(st, command, rt, approved, advisory...)
	}
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
		return servingError("Prepared Strata directories must be absolute, clean, existing directories in native platform form.")
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
	// The typed decode below folds case, so a key that differs only by case
	// would silently replace the first value: names are compared case-insensitively.
	seen := map[string]bool{}
	for decoder.More() {
		token, err := decoder.Token()
		key, ok := token.(string)
		if err != nil || !ok || seen[strings.ToLower(key)] {
			return config, servingError("Duplicate or malformed Strata config key.")
		}
		seen[strings.ToLower(key)] = true
		var value json.RawMessage
		if decoder.Decode(&value) != nil || bytes.Equal(value, []byte("null")) {
			return config, servingError("Null or malformed Strata config values are unsupported.")
		}
		if strings.EqualFold(key, "model_name") {
			if err := nodepolicy.ValidateModelIdentifierJSON(value); err != nil {
				return config, servingError(err.Error())
			}
		}
		if strings.EqualFold(key, "env") && !strataDistinctNames(value) {
			return config, servingError("Duplicate or malformed Strata config key.")
		}
	}
	decoder = json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&config) != nil || decoder.Decode(new(json.RawMessage)) != io.EOF {
		return config, servingError("Unsupported or malformed prepared Strata config.")
	}
	return config, nil
}

// strataDistinctNames reports whether a JSON object has no two keys equal
// ignoring case. A value that is not an object is left to the typed decode.
func strataDistinctNames(raw json.RawMessage) bool {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return true
	}
	seen := map[string]bool{}
	for decoder.More() {
		token, err := decoder.Token()
		key, ok := token.(string)
		if err != nil || !ok || seen[strings.ToUpper(key)] {
			return false
		}
		seen[strings.ToUpper(key)] = true
		var skipped json.RawMessage
		if decoder.Decode(&skipped) != nil {
			return false
		}
	}
	return true
}

func compileStrataServing(command launchCommand, rt Runtime, approved bool) (*llamaServingLaunch, error) {
	result, err := deriveStrataServing(command, rt)
	if err != nil {
		return nil, err
	}
	for _, artifact := range result.Config.Models[0].Artifacts {
		if !strings.HasPrefix(artifact.Role, "runtime:") {
			continue
		}
		path := artifact.Binding.SourcePath
		sum, err := strataRuntimeDigest(path)
		if err != nil {
			return nil, err
		}
		after, bindingErr := servingFileBinding(path)
		if sum != artifact.SHA256 || bindingErr != nil || *after != *artifact.Binding {
			return nil, servingError("Strata runtime full SHA256 mismatch or unstable file.")
		}
	}
	expected := rt.Serving.Models[0].ExpectedNativeSourceDigest
	if (approved && expected == "") || (expected != "" && expected != result.Digest) {
		return nil, servingError("Strata native source conflict; obtain a fresh preview.")
	}
	return result, nil
}

// Derive and validate the complete selected-file closure using only bounded
// reads. Both pre-lock hashing and locked compilation use this role check.
func deriveStrataServing(command launchCommand, rt Runtime) (*llamaServingLaunch, error) {
	if rt.modeOrDefault() != "process" || rt.Serving == nil || len(servingDeclarationErrors(rt.Serving)) != 0 || len(rt.Serving.Models) != 1 {
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
	if err := validateStrataExtras(config); err != nil {
		return nil, err
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
		target, err := strataTarget(path)
		if err != nil {
			return err
		}
		if previous, exists := paths[target]; exists && previous != role {
			return servingError("Strata file has ambiguous artifact roles.")
		}
		paths[target] = role
		return nil
	}
	for _, entry := range []struct{ path, role string }{{command.Bin, "runtime:python"}, {command.Args[0], "runtime:server"}, {source, "runtime:config"}, {config.Exe, "runtime:native-exe"}} {
		if err := add(entry.path, entry.role); err != nil {
			return nil, err
		}
	}
	options := map[string]string{}
	placement := []string{}
	expertTable := ""
	for i := 0; i < len(config.Args); i++ {
		key := config.Args[i]
		canonical := key
		if key == "--slots" {
			canonical = "--batch"
		}
		if _, exists := options[canonical]; exists {
			return nil, servingError("Duplicate or conflicting native Strata arguments.")
		}
		// The server prepends --serve itself, so a config may omit it; the other
		// value-less flag is part of the reviewed placement.
		if key == "--serve" || key == "--no-prefill-borrow" {
			options[key] = "true"
			if key != "--serve" {
				placement = append(placement, key)
			}
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
			// The fork's packer normally writes native_experts.txt and no
			// experts.bin. The table and the shards it names are bound below;
			// both sources together would make the expert source ambiguous.
			table := filepath.Join(value, "native_experts.txt")
			if _, err := os.Lstat(table); err == nil {
				if key == "--mtp" {
					return nil, servingError("Pack native_experts indirection is unsupported.")
				}
				if _, err := os.Lstat(filepath.Join(value, "experts.bin")); !os.IsNotExist(err) {
					return nil, servingError("Pack has both experts.bin and native_experts.txt; the expert source is ambiguous.")
				}
				files = []string{"index.txt", "dense.bin"}
				expertTable = table
			} else if !os.IsNotExist(err) {
				return nil, servingError("Cannot inspect prepared pack file.")
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
			if key == "--pack" && expertTable != "" {
				if err := add(expertTable, role); err != nil {
					return nil, err
				}
			}
		case "--prefill", "--vram-reserve-mib", "--expert-cache", "--spec", "--kv-resident":
			if _, err := llamaPositive(value); err != nil {
				return nil, err
			}
			placement = append(placement, key+"="+value)
		case "--kv":
			if !slices.Contains(strataKVFormats, value) {
				return nil, servingError("Unsupported Strata KV format; use " + strings.Join(strataKVFormats, ", ") + ".")
			}
			placement = append(placement, key+"="+value)
		case "--resident-budget-gib":
			if !strataDecimal(value, false) {
				return nil, servingError("Strata resident budget must be a positive decimal number of GiB.")
			}
			placement = append(placement, key+"="+value)
		case "--suffix-draft", "--lookup-chain":
			if value != "0" {
				return nil, servingError("Strata speculative alternatives are supported only when explicitly disabled with 0.")
			}
			placement = append(placement, key+"="+value)
		default:
			return nil, servingError("Unsupported native Strata argument; implicit paths, fitting and allocation modes are forbidden.")
		}
	}
	if expertTable != "" {
		if err := strataExpertTable(expertTable, options["--native"], add); err != nil {
			return nil, err
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
	if ctxErr != nil || contextTokens > (math.MaxInt32-8)/slots || contextTokens <= model.MaxOutputTokens || options["--native"] == "" || options["--pack"] == "" {
		return nil, servingError("Strata requires explicit finite context, native model and pack; output must fit the per-slot context.")
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
		if artifact.Binding == nil || !filepath.IsAbs(artifact.Binding.SourcePath) {
			return nil, servingError("Every Strata artifact requires a unique absolute declared file binding.")
		}
		path := artifact.Binding.SourcePath
		target, err := strataTarget(path)
		if err != nil || declared[target] {
			return nil, servingError("Every Strata artifact requires a unique resolved file binding.")
		}
		role, selected := paths[target]
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
		declared[target] = true
	}
	for path := range paths {
		if !declared[path] {
			return nil, servingError("Explicit Strata dependency has no declared artifact.")
		}
	}
	// The server opens the log for append. It must be a new or existing regular
	// file, never a link, and never the same file as any declared input, by
	// identity, so a hard link cannot make an input writable.
	if config.Log != nil {
		logInfo, err := os.Lstat(*config.Log)
		switch {
		case os.IsNotExist(err):
		case err != nil || !logInfo.Mode().IsRegular():
			return nil, servingError("Strata log must be a regular file, not a link or device.")
		default:
			for _, artifact := range model.Artifacts {
				if info, err := os.Stat(artifact.Binding.SourcePath); err != nil || os.SameFile(logInfo, info) {
					return nil, servingError("Strata log must not be a declared input file.")
				}
			}
		}
	}
	names := make([]string, 0, len(config.Env))
	for name := range config.Env {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		placement = append(placement, "env:"+name+"="+config.Env[name])
	}
	forwarded := config
	var keyIdentity *strataKeyIdentity
	if config.APIKeyFile != nil {
		key, identity, err := readStrataKeyFile(*config.APIKeyFile, model.Artifacts)
		if err != nil {
			return nil, err
		}
		result.Credential, keyIdentity = key, identity
	}
	forwarded.BuildMetadata, forwarded.GPUsAsked, forwarded.Port, forwarded.Host, forwarded.OpenBrowser, forwarded.APIKeyFile = nil, nil, nil, nil, nil, nil
	result.Snapshot, err = json.Marshal(forwarded)
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
		// The key file by identity, never content: changing the key changes
		// the approval without any advertised field deriving from the key.
		Key *strataKeyIdentity `json:",omitempty"`
	}{1, binding, data, result.Config, command, keyIdentity})
	if err != nil {
		return nil, servingError("Cannot bind prepared Strata inputs.")
	}
	digest := sha256.Sum256(encoded)
	result.Digest = hex.EncodeToString(digest[:])
	model.ExpectedNativeSourceDigest = result.Digest
	result.Envelopes = []engines.ServingEnvelope{{ModelID: model.ModelID, PerSlotContextTokens: contextTokens, TotalContextTokens: contextTokens * slots, Slots: slots, MaxOutputTokens: model.MaxOutputTokens, Memory: model.Memory, Placement: placement, Retention: strconv.FormatInt(config.IdleUnloadS, 10), RouterCapacity: 1, NativeSourceDigest: result.Digest}}
	return result, nil
}

// strataExpertTable binds the fork packer's native_experts.txt: a header naming
// version v3 or v4, then one line per layer in order from zero with eight
// unsigned numbers and an optional shard column (one name, or for v4 a gate,up,
// down list). Each named shard sits beside the first --native shard and must be
// a declared model file. Anything else in the table is rejected.
func strataExpertTable(table, native string, add func(path, role string) error) error {
	reject := func(reason string) error {
		return servingError("The native_experts table " + reason + ".")
	}
	if native == "" {
		return reject("needs --native to place its shards")
	}
	data, _, err := servingSource(table)
	if err != nil {
		return reject("is missing, unstable or oversized")
	}
	// The engine reads a version from every comment that starts with the tag and
	// refuses one it does not know, so every such comment is checked here, and
	// they must all agree with the header.
	const tag = "# strata native experts v"
	lines := strings.Split(string(data), "\n")
	version := ""
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, tag) {
			if i == 0 {
				return reject("has no version header")
			}
			continue
		}
		this := ""
		for _, supported := range []string{"v3", "v4"} {
			if strings.HasPrefix(line, "# strata native experts "+supported+":") {
				this = supported
			}
		}
		if this == "" || (version != "" && this != version) {
			return reject("has an unsupported or inconsistent version header")
		}
		version = this
	}
	shards := map[string]bool{}
	layers := 0
	for _, line := range lines[1:] {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 8 || len(fields) > 9 {
			return reject("has a layer line that is not eight numbers and an optional shard column")
		}
		for i, field := range fields[:8] {
			n, err := strconv.ParseUint(field, 10, 63)
			if err != nil || (i == 0 && n != uint64(layers)) {
				return reject("has a bad number or layers that do not run from zero in order")
			}
		}
		layers++
		if len(fields) == 9 {
			names := strings.Split(fields[8], ",")
			if len(names) != 1 && (version == "v3" || len(names) != 3) {
				return reject("has a shard column that is not one name or a v4 gate,up,down list")
			}
			// In a v4 list an empty component means the first --native shard,
			// which the packer writes for roles stored there.
			named := 0
			for _, name := range names {
				if name == "" && len(names) == 3 {
					continue
				}
				lower := strings.ToLower(name)
				if name == "" || name != filepath.Base(name) || strings.ContainsAny(name, `/\:`) || name == "." || name == ".." || !strings.HasSuffix(lower, ".gguf") {
					return reject("names a shard that is not a plain .gguf file name")
				}
				shards[name] = true
				named++
			}
			if named == 0 {
				return reject("has a v4 shard list that names no shard")
			}
		}
	}
	if layers == 0 {
		return reject("has no layers")
	}
	names := make([]string, 0, len(shards))
	for name := range shards {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		if err := add(filepath.Join(filepath.Dir(native), name), "model"); err != nil {
			return err
		}
	}
	return nil
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
	if len(st.plat.Runtime.ModelLoads) != 0 {
		cli, env, err := llamaLaunchOptions(command, st.plat.Runtime.EditableLaunch)
		if err == nil && cli["models-max"] == "" && env["models-max"] == "" {
			command.Args = append(slices.Clone(command.Args), "--models-max", "1")
		}
	}
	return llamaSnapshotCommand(command, snapshot, st.plat.Runtime.ModelLoads)
}
