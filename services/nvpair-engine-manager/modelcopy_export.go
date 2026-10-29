// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
)

const llamaArtifactsLayout = "llama-artifacts-v1"
const maxCopyArtifacts = 1024
const maxCopyManifest = 1 << 20
const maxCopyProgressFrame = 64 << 10

func copyPathKey(path string) string {
	return strings.Map(func(r rune) rune {
		smallest := r
		for folded := unicode.SimpleFold(r); folded != r; folded = unicode.SimpleFold(folded) {
			if folded < smallest {
				smallest = folded
			}
		}
		return smallest
	}, path)
}

func copyPercent(done, total int64) int {
	if total <= 0 {
		return 0
	}
	return int(float64(done) / float64(total) * 100)
}

// Decode the artifact array incrementally, rejecting excess before appending
// another entry. A supplied array length never controls an allocation.
func decodeCopyManifest(data []byte) (modelFileList, error) {
	if len(data) > maxCopyManifest {
		return modelFileList{}, fmt.Errorf("copy manifest exceeds bounds")
	}
	var header struct {
		Engine   string          `json:"engine"`
		Model    string          `json:"model"`
		Layout   string          `json:"layout"`
		ExportID string          `json:"exportId"`
		Revision string          `json:"revision"`
		Files    json.RawMessage `json:"files"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		return modelFileList{}, err
	}
	list := modelFileList{Engine: header.Engine, Model: header.Model, Layout: header.Layout, ExportID: header.ExportID, Revision: header.Revision}
	dec := json.NewDecoder(bytes.NewReader(header.Files))
	open, err := dec.Token()
	if err != nil || open != json.Delim('[') {
		return modelFileList{}, fmt.Errorf("invalid copy artifact array")
	}
	for dec.More() {
		if len(list.Files) >= maxCopyArtifacts {
			return modelFileList{}, fmt.Errorf("too many copy artifacts")
		}
		var f modelFile
		if err := dec.Decode(&f); err != nil {
			return modelFileList{}, err
		}
		list.Files = append(list.Files, f)
	}
	if _, err := dec.Token(); err != nil {
		return modelFileList{}, err
	}
	if dec.Decode(new(json.RawMessage)) != io.EOF {
		return modelFileList{}, fmt.Errorf("invalid copy artifact trailer")
	}
	return list, nil
}

// Each frame is bounded independently; a long hash stream may contain any
// number of coalesced progress frames. Never ReadAll the cumulative stream.
func readCopyFrame(reader *bufio.Reader) ([]byte, error) {
	var line []byte
	limit := maxCopyProgressFrame
	for {
		fragment, err := reader.ReadSlice('\n')
		if len(line) == 0 {
			// Large result frames use the existing streamFrame envelope, whose
			// first field is type. Other frames remain bounded before decoding.
			dec := json.NewDecoder(bytes.NewReader(fragment))
			open, e1 := dec.Token()
			key, e2 := dec.Token()
			kind, e3 := dec.Token()
			if e1 == nil && e2 == nil && e3 == nil && open == json.Delim('{') && key == "type" && kind == "result" {
				limit = maxCopyManifest + 4096
			}
		}
		if len(line)+len(fragment) > limit {
			return nil, fmt.Errorf("copy frame exceeds bounds")
		}
		line = append(line, fragment...)
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		if err != nil {
			return nil, err
		}
		return line, nil
	}
}

type issuedCopyKey struct {
	state *engineState
	model string
}
type issuedCopyExport struct {
	id, binding string
	infos       []os.FileInfo
}

// Only the latest issued identity per local alias is retained. A finite cache
// prevents pinned readers from growing source state without bound. Restart or
// capacity eviction requires a fresh listing; unchanged listings are stable.
var issuedCopyExports = struct {
	sync.Mutex
	entries map[issuedCopyKey]issuedCopyExport
	order   []issuedCopyKey
}{entries: map[issuedCopyKey]issuedCopyExport{}}

func rememberCopyExport(st *engineState, list modelFileList) {
	key := issuedCopyKey{st, list.Model}
	issuedCopyExports.Lock()
	defer issuedCopyExports.Unlock()
	if _, ok := issuedCopyExports.entries[key]; !ok {
		if len(issuedCopyExports.order) >= 256 {
			delete(issuedCopyExports.entries, issuedCopyExports.order[0])
			issuedCopyExports.order = issuedCopyExports.order[1:]
		}
		issuedCopyExports.order = append(issuedCopyExports.order, key)
	}
	infos := make([]os.FileInfo, len(list.Files))
	for i, f := range list.Files {
		infos[i] = f.info
	}
	issuedCopyExports.entries[key] = issuedCopyExport{list.ExportID, list.binding, infos}
}

func checkIssuedCopyExport(st *engineState, list modelFileList) bool {
	key := issuedCopyKey{st, list.Model}
	issuedCopyExports.Lock()
	defer issuedCopyExports.Unlock()
	issued, ok := issuedCopyExports.entries[key]
	if !ok || issued.id != list.ExportID || issued.binding != list.binding || len(issued.infos) != len(list.Files) {
		if ok {
			issuedCopyExports.entries[key] = issuedCopyExport{}
		}
		return false
	}
	for i, f := range list.Files {
		if !sameCopyFile(issued.infos[i], f.info) {
			issuedCopyExports.entries[key] = issuedCopyExport{}
			return false
		}
	}
	return true
}

// A cold full hash may outlive a settings update. Re-resolve after hashing so
// a manifest is never issued from a configuration superseded during that read.
func (e *Executor) recheckCopyExport(ctx context.Context, list modelFileList) error {
	current, err := e.modelSourceFiles(ctx, list.Engine, list.Model)
	if err != nil || current.Layout != list.Layout || current.binding != list.binding || len(current.Files) != len(list.Files) {
		return fmt.Errorf("custom artifact binding changed during hashing")
	}
	for i, f := range current.Files {
		if !sameCopyFile(f.info, list.Files[i].info) {
			return fmt.Errorf("custom artifact file changed during hashing")
		}
	}
	return nil
}

func exportBinding(list modelFileList, sourceBinding any) string {
	// This digest is source-local and never sent over the wire. Include the
	// complete small preset/config snapshot and every exact resolved binding.
	type binding struct {
		Configured, Resolved, Sample string
		Size, Mtime                  int64
	}
	bindings := []binding{}
	for _, f := range list.Files {
		bindings = append(bindings, binding{f.configured, f.abs, f.sample, f.info.Size(), f.info.ModTime().UnixNano()})
	}
	data, _ := json.Marshal(struct {
		Source any
		Files  []binding
	}{sourceBinding, bindings})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

type copyDurability string

const (
	copyDurabilityConfirmed copyDurability = "confirmed"
	copyDurabilityUncertain copyDurability = "uncertain"
)

var errNoArtifactBinding = errors.New("No unique local artifact binding for this model; configure its source files on the source device.")

func sameCopyFile(a, b os.FileInfo) bool {
	return a != nil && b != nil && a.Mode().IsRegular() && b.Mode().IsRegular() && os.SameFile(a, b) && a.Size() == b.Size() && a.ModTime() == b.ModTime()
}

// artifactExportID hashes UTF-8 encoding/json output, without a trailing newline,
// of this exact field order: version, model, artifacts; each artifact has role,
// path, size, sha256. Version is 1; artifact order is native model then draft
// shard order. JSON string escaping is Go encoding/json's default HTML escaping.
// Host paths, engine ID and serving settings are deliberately absent.
func artifactExportID(list modelFileList) (string, error) {
	type artifact struct {
		Role   string `json:"role"`
		Path   string `json:"path"`
		Size   int64  `json:"size"`
		SHA256 string `json:"sha256"`
	}
	manifest := struct {
		Version   int        `json:"version"`
		Model     string     `json:"model"`
		Artifacts []artifact `json:"artifacts"`
	}{Version: 1, Model: list.Model}
	if len(list.Files) == 0 || len(list.Files) > maxCopyArtifacts {
		return "", fmt.Errorf("invalid custom artifact count")
	}
	var total int64
	seen := map[string]bool{}
	draft := false
	for _, f := range list.Files {
		if f.Role == "draft" {
			draft = true
		} else if f.Role != "model" || draft {
			return "", fmt.Errorf("invalid custom artifact order or role")
		}
		if !validModelRelPath(f.Path) || !strings.HasPrefix(f.Path, f.Role+"/") || strings.Count(f.Path, "/") != 1 || !strings.HasSuffix(f.Path, ".gguf") || !validSHA256(f.SHA256) || f.Size < 0 || f.Size > int64(^uint64(0)>>1)-total {
			return "", fmt.Errorf("invalid custom artifact metadata")
		}
		fold := copyPathKey(f.Path)
		if seen[fold] {
			return "", fmt.Errorf("duplicate or case-colliding custom artifact path")
		}
		seen[fold] = true
		total += f.Size
		manifest.Artifacts = append(manifest.Artifacts, artifact{f.Role, f.Path, f.Size, f.SHA256})
	}
	if list.Files[0].Role != "model" {
		return "", fmt.Errorf("custom export requires a model artifact")
	}
	data, err := json.Marshal(manifest)
	if err != nil || len(data) > maxCopyManifest {
		return "", fmt.Errorf("custom manifest exceeds bounds")
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// llamaConsumedPaths is the shared compiler/export path-selection owner. The
// parser has already refused includes, projectors, adapters and remote sources.
func llamaConsumedPaths(source string, options map[string]string, model string, limit int) ([]string, []string, error) {
	var paths, roles []string
	for _, key := range []string{"model", "model-draft"} {
		path, ok := options[key]
		if !ok {
			if key == "model" {
				return nil, nil, servingError("Every model preset requires an existing local model path.")
			}
			continue
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(filepath.Dir(source), path)
		}
		path = filepath.Clean(path)
		if err := validateLlamaPresetValue(key, path, model); err != nil {
			return nil, nil, err
		}
		options[key] = path
		if limit > 0 {
			count := 1
			if parts := llamaShardName.FindStringSubmatch(path); parts != nil {
				var err error
				count, err = strconv.Atoi(parts[3])
				if err != nil {
					return nil, nil, servingError("Invalid custom shard count.")
				}
			}
			if count > limit-len(paths) {
				return nil, nil, errModelArtifactLimit
			}
		}
		shards, err := llamaArtifactPaths(path)
		if err != nil {
			return nil, nil, servingError("Invalid or excessive custom artifact shards.")
		}
		role := "model"
		if key == "model-draft" {
			role = "draft"
		}
		for _, shard := range shards {
			paths = append(paths, shard)
			roles = append(roles, role)
		}
	}
	return paths, roles, nil
}

// Take immutable launch/config inputs under the existing locks, and do all
// preset reading, binding sampling and compilation after releasing them.
func copyLaunchSnapshot(st *engineState) (*engineState, launchCommand, error) {
	st.opMu.Lock()
	defer st.opMu.Unlock()
	snapshot := snapshotLlamaState(st)
	command, err := launchForState(snapshot, snapshot.plat.Runtime.Port)
	return snapshot, command, err
}

func (e *Executor) llamaExportFiles(ctx context.Context, st *engineState, store engineModelStore, model string) (modelFileList, error) {
	if err := ctx.Err(); err != nil {
		return modelFileList{}, err
	}
	snapshot, command, err := copyLaunchSnapshot(st)
	if err != nil {
		return modelFileList{}, errNoArtifactBinding
	}
	list, presetSelected, err := resolveLlamaArtifacts(ctx, snapshot, command, store, model, servingFileAccess{})
	if err != nil {
		return modelFileList{}, err
	}
	// Approval and export hashing retain their existing owners. Inspection uses
	// only the resolver below and never compiles every serving declaration.
	typedSelected := slices.ContainsFunc(snapshot.plat.Runtime.ModelLoads, func(load modelconfig.ModelLoad) bool { return load.ModelID == model })
	if (presetSelected || typedSelected) && snapshot.plat.Runtime.Serving != nil {
		snapshot.plat.Runtime.modelLoadOwner = e
		compiled, err := compilePreparedServing(snapshot, command, snapshot.plat.Runtime, true)
		if err != nil {
			return modelFileList{}, fmt.Errorf("configured serving artifact binding failed: %w", err)
		}
		matched := false
		for _, declared := range compiled.Config.Models {
			if declared.ModelID != model {
				continue
			}
			matched = true
			if len(declared.Artifacts) != len(list.Files) {
				return modelFileList{}, fmt.Errorf("configured artifact set changed during resolution")
			}
			for i, artifact := range declared.Artifacts {
				list.Files[i].asserted = artifact.SHA256
			}
		}
		if !matched {
			return modelFileList{}, errNoArtifactBinding
		}
	}
	current, currentCommand, currentErr := copyLaunchSnapshot(st)
	if currentErr != nil || !reflect.DeepEqual(command, currentCommand) || !snapshot.plat.Runtime.Serving.Equal(current.plat.Runtime.Serving) || !modelconfig.EqualLoads(snapshot.plat.Runtime.ModelLoads, current.plat.Runtime.ModelLoads) {
		return modelFileList{}, fmt.Errorf("custom artifact configuration changed during export resolution")
	}
	return list, nil
}

// resolveLlamaArtifacts is the single pre-hash path-selection seam. Inputs are
// immutable snapshots, and the caller owns content admission and final recheck.
// The boolean identifies a matching preset so copy approval applies to that
// source, without making receipt-only exports depend on unrelated declarations.
func resolveLlamaArtifacts(ctx context.Context, snapshot *engineState, command launchCommand, store engineModelStore, model string, access servingFileAccess) (modelFileList, bool, error) {
	if err := ctx.Err(); err != nil {
		return modelFileList{}, false, err
	}
	if access.scan == nil {
		access.scan = &modelArtifactScan{}
	}
	if !artifactPathSafe(store.root) {
		return modelFileList{}, false, errNoArtifactBinding
	}
	if err := validateLlamaModelID(model); err != nil {
		return modelFileList{}, false, errNoArtifactBinding
	}
	rt := snapshot.plat.Runtime
	presetConfigured := rt.Serving != nil && len(rt.ModelLoads) == 0
	for _, arg := range command.Args {
		name, _, _ := strings.Cut(arg, "=")
		if llamaLongOption(name) == "--models-preset" {
			presetConfigured = true
		}
	}
	for key, value := range command.Env {
		if environmentKey(key) == "LLAMA_ARG_MODELS_PRESET" && value != "" {
			presetConfigured = true
		}
	}
	if os.Getenv("LLAMA_ARG_MODELS_PRESET") != "" {
		presetConfigured = true
	}
	var candidates []modelFileList
	presetSelected := false
	if presetConfigured {
		cli, env, parseErr := llamaLaunchOptions(command, rt.EditableLaunch)
		if parseErr != nil {
			return modelFileList{}, false, fmt.Errorf("custom artifact configuration is unsupported: %w", parseErr)
		}
		source := env["models-preset"]
		if v, ok := cli["models-preset"]; ok {
			source = v
		}
		if !artifactPathSafe(source) {
			return modelFileList{}, false, errNoArtifactBinding
		}
		data, binding, readErr := access.source(ctx, source)
		if readErr != nil {
			return modelFileList{}, false, fmt.Errorf("configured custom preset is missing or unstable")
		}
		sections, parseErr := parseLlamaServingPreset(data)
		if parseErr != nil {
			return modelFileList{}, false, fmt.Errorf("custom artifact configuration is unsupported: %w", parseErr)
		}
		preset, found := sections[model]
		if found && model != "*" {
			presetSelected = true
			options := map[string]string{}
			for _, values := range []map[string]string{env, sections["*"], preset, cli} {
				for key, value := range values {
					options[key] = value
				}
			}
			paths, roles, pathErr := llamaConsumedPaths(source, options, model, maxCopyArtifacts)
			if pathErr != nil {
				return modelFileList{}, false, pathErr
			}
			list := modelFileList{Engine: store.engine, Model: model, Layout: llamaArtifactsLayout}
			for i, path := range paths {
				if err := ctx.Err(); err != nil {
					return modelFileList{}, false, err
				}
				if !artifactPathSafe(path) || !strings.HasSuffix(path, ".gguf") {
					return modelFileList{}, false, fmt.Errorf("custom copies require local GGUF model and draft artifacts")
				}
				current, bindErr := access.binding(ctx, path)
				if bindErr != nil {
					return modelFileList{}, false, fmt.Errorf("configured model or draft shard is missing or unstable")
				}
				info, statErr := os.Stat(current.ResolvedTarget)
				if statErr != nil {
					return modelFileList{}, false, fmt.Errorf("configured artifact is unavailable")
				}
				list.Files = append(list.Files, modelFile{Role: roles[i], Path: roles[i] + "/" + filepath.Base(path), Size: info.Size(), abs: current.ResolvedTarget, configured: current.SourcePath, sample: current.SampleSHA256, info: info})
			}
			after, afterErr := access.binding(ctx, source)
			if afterErr != nil || *binding != *after {
				return modelFileList{}, false, fmt.Errorf("configured preset changed during export resolution")
			}
			candidates = append(candidates, list)
			candidates[len(candidates)-1].binding = exportBinding(list, struct {
				Preset  string
				Bytes   []byte
				Command launchCommand
				Serving any
			}{source, data, command, rt.Serving})
		} else if rt.Serving != nil {
			for _, declared := range rt.Serving.Models {
				if declared.ModelID == model {
					return modelFileList{}, false, fmt.Errorf("declared model has no exact configured preset")
				}
			}
		}
	}
	receipt, receiptErr := readCopyReceiptWithAccess(ctx, store, model, access)
	if receiptErr != nil && !errors.Is(receiptErr, os.ErrNotExist) {
		return modelFileList{}, false, receiptErr
	}
	if receiptErr == nil {
		list := receipt.List
		for i := range list.Files {
			f := &list.Files[i]
			path := filepath.Join(copyAliasDir(store.root, model), list.ExportID, filepath.FromSlash(f.Path))
			if !artifactPathSafe(path) {
				return modelFileList{}, false, errNoArtifactBinding
			}
			if err := safeCopyDestinationBounded(ctx, store.root, path, access.scan); err != nil {
				return modelFileList{}, false, fmt.Errorf("copied artifact location is unsafe: %w", err)
			}
			binding, err := access.binding(ctx, path)
			if err != nil {
				return modelFileList{}, false, fmt.Errorf("committed copy artifact is missing or unstable")
			}
			info, err := os.Stat(binding.ResolvedTarget)
			if err != nil {
				return modelFileList{}, false, fmt.Errorf("committed copy artifact unavailable")
			}
			f.abs, f.configured, f.info, f.asserted, f.SHA256 = binding.ResolvedTarget, binding.SourcePath, info, f.SHA256, ""
			f.sample = binding.SampleSHA256
		}
		candidates = append(candidates, list)
		candidates[len(candidates)-1].binding = exportBinding(list, struct {
			Receipt copyReceipt
			Binding string
		}{receipt, receipt.binding})
	}
	for _, load := range rt.ModelLoads {
		if load.ModelID != model || load.Resolved == nil {
			continue
		}
		list, err := resolvedLoadArtifacts(ctx, store, model, load, access)
		if err != nil {
			return modelFileList{}, false, err
		}
		candidates = append(candidates, list)
	}
	if len(candidates) > 0 {
		for _, candidate := range candidates[1:] {
			if len(candidate.Files) != len(candidates[0].Files) {
				return modelFileList{}, false, errNoArtifactBinding
			}
			for i, f := range candidate.Files {
				first := candidates[0].Files[i]
				if f.Role != first.Role || f.abs != first.abs {
					return modelFileList{}, false, errNoArtifactBinding
				}
			}
		}
		if err := validateResolvedArtifacts(candidates[0]); err != nil {
			return modelFileList{}, false, err
		}
		// Bind every agreeing source, so changing even the receipt beside a
		// preset invalidates derived cache entries and in-flight inspections.
		bindings := make([]string, 0, len(candidates))
		for _, candidate := range candidates {
			bindings = append(bindings, candidate.binding)
		}
		candidates[0].binding = exportBinding(candidates[0], bindings)
		return candidates[0], presetSelected, nil
	}
	if _, _, err := splitLlamaModel(model); err == nil {
		list, err := boundedLlamaCacheArtifacts(ctx, store, model, access)
		return list, false, err
	}
	return modelFileList{}, false, errNoArtifactBinding
}

func copyAliasDir(root, model string) string {
	sum := sha256.Sum256([]byte(model))
	return filepath.Join(root, ".pair-copies", hex.EncodeToString(sum[:]))
}

type copyReceipt struct {
	Version int           `json:"version"`
	List    modelFileList `json:"manifest"`
	binding string
}

func readCopyReceipt(store engineModelStore, model string) (copyReceipt, error) {
	return readCopyReceiptWithAccess(context.Background(), store, model, servingFileAccess{scan: &modelArtifactScan{}})
}

func readCopyReceiptWithAccess(ctx context.Context, store engineModelStore, model string, access servingFileAccess) (copyReceipt, error) {
	path := filepath.Join(copyAliasDir(store.root, model), "copy.json")
	if err := safeCopyDestinationBounded(ctx, store.root, path, access.scan); err != nil {
		return copyReceipt{}, fmt.Errorf("copy receipt location is unsafe: %w", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return copyReceipt{}, os.ErrNotExist
		}
		return copyReceipt{}, fmt.Errorf("copy receipt is unavailable")
	}
	if !info.Mode().IsRegular() || info.Size() > maxCopyManifest {
		return copyReceipt{}, fmt.Errorf("invalid copy receipt")
	}
	data, binding, err := access.source(ctx, path)
	if err != nil || len(data) > maxCopyManifest {
		return copyReceipt{}, fmt.Errorf("invalid copy receipt")
	}
	var envelope struct {
		Version  int             `json:"version"`
		Manifest json.RawMessage `json:"manifest"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&envelope); err != nil {
		return copyReceipt{}, fmt.Errorf("invalid copy receipt")
	}
	if dec.Decode(new(json.RawMessage)) != io.EOF {
		return copyReceipt{}, fmt.Errorf("invalid copy receipt trailer")
	}
	list, err := decodeCopyManifest(envelope.Manifest)
	if err != nil {
		return copyReceipt{}, fmt.Errorf("invalid copy receipt")
	}
	receipt := copyReceipt{Version: envelope.Version, List: list, binding: binding.SampleSHA256}
	id, err := artifactExportID(receipt.List)
	if err != nil || receipt.Version != 1 || receipt.List.Engine != store.engine || receipt.List.Model != model || receipt.List.Layout != llamaArtifactsLayout || id != receipt.List.ExportID {
		return copyReceipt{}, fmt.Errorf("invalid copy receipt identity")
	}
	return receipt, nil
}

// A retry may prove completion without a reachable source. Reading the receipt
// alone is insufficient: every committed artifact is confined and fully hashed
// under the same destination writer lock as ordinary copy. No files are created.
func (e *Executor) completedCopy(ctx context.Context, engine, model, expected string, emit func(copyProgress)) (copyModelResult, bool, error) {
	_, store, err := e.modelStoreFor(engine)
	if err != nil {
		return copyModelResult{}, false, err
	}
	if store.adapter != "llamacpp" {
		return copyModelResult{}, false, nil
	}
	unlock, err := copyDestLocks.lock(ctx, store.root+"\x00"+model)
	if err != nil {
		return copyModelResult{}, false, err
	}
	defer unlock()
	receipt, err := readCopyReceipt(store, model)
	if errors.Is(err, os.ErrNotExist) {
		return copyModelResult{}, false, nil
	}
	if err != nil {
		return copyModelResult{}, false, err
	}
	if receipt.List.ExportID != expected {
		return copyModelResult{}, false, copyConflict("committed copy differs from expected export identity")
	}
	root, err := os.OpenRoot(store.root)
	if err != nil {
		return copyModelResult{}, false, fmt.Errorf("cannot open destination model store")
	}
	defer root.Close()
	res := copyModelResult{Engine: engine, Model: model, Layout: llamaArtifactsLayout, ExportID: expected, Durability: copyDurabilityUncertain, RequiresConfiguration: true, Files: len(receipt.List.Files), FilesSkipped: len(receipt.List.Files)}
	for _, f := range receipt.List.Files {
		res.BytesTotal += f.Size
	}
	var done int64
	var lastEmit time.Time
	report := func(file string, bytes int64, force bool) {
		if emit == nil || !force && time.Since(lastEmit) < copyProgressInterval {
			return
		}
		lastEmit = time.Now()
		emit(copyProgress{remoteProgress: remoteProgress{Stage: "checking", Percent: copyPercent(done+bytes, res.BytesTotal)}, ExportID: expected, File: file, BytesDone: done + bytes, BytesTotal: res.BytesTotal})
	}
	report("", 0, true)
	for _, f := range receipt.List.Files {
		dest := filepath.Join(copyAliasDir(store.root, model), expected, filepath.FromSlash(f.Path))
		if err := safeCopyDestination(store.root, dest); err != nil {
			return res, false, err
		}
		rel, err := filepath.Rel(store.root, dest)
		if err != nil {
			return res, false, err
		}
		file, err := root.Open(rel)
		if errors.Is(err, os.ErrNotExist) {
			return copyModelResult{}, false, nil
		}
		if err != nil {
			return res, false, fmt.Errorf("cannot verify committed copy artifact")
		}
		before, statErr := file.Stat()
		if statErr != nil || !before.Mode().IsRegular() || before.Size() != f.Size {
			_ = file.Close()
			return res, false, copyConflict("committed copy artifact does not match its receipt")
		}
		hash := sha256.New()
		_, hashErr := copyWithContext(ctx, hash, file, func(n int64) { report(f.Path, n, false) })
		closeErr := file.Close()
		after, afterErr := root.Lstat(rel)
		if afterErr != nil || !sameCopyFile(before, after) {
			return res, false, copyConflict("committed copy artifact changed")
		}
		if hashErr != nil {
			return res, false, hashErr
		}
		if closeErr != nil {
			return res, false, fmt.Errorf("cannot verify committed copy artifact")
		}
		if before.Size() != f.Size || hex.EncodeToString(hash.Sum(nil)) != f.SHA256 {
			return res, false, copyConflict("committed copy artifact does not match its receipt")
		}
		done += f.Size
	}
	return res, true, ctx.Err()
}

// Existing components below the trusted store root must be ordinary directories
// and files. Reject symlinks (including .part and receipt paths) and Windows
// case aliases before any creation, and repeat under each writer lock.
func safeCopyDestination(root, dest string) error {
	return safeCopyDestinationBounded(context.Background(), root, dest, &modelArtifactScan{})
}

func safeCopyDestinationBounded(ctx context.Context, root, dest string, scan *modelArtifactScan) error {
	if scan == nil {
		scan = &modelArtifactScan{}
	}
	if !pathWithinRoot(root, dest) || filepath.Clean(root) == filepath.Clean(dest) {
		return fmt.Errorf("copy destination escapes model store")
	}
	rel, err := filepath.Rel(root, dest)
	if err != nil {
		return err
	}
	p := root
	for _, seg := range strings.Split(rel, string(filepath.Separator)) {
		err := scan.directory(ctx, p, func(entry os.DirEntry) error {
			if strings.EqualFold(entry.Name(), seg) && entry.Name() != seg {
				return fmt.Errorf("copy destination has a case-colliding path")
			}
			return nil
		})
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			if errors.Is(err, errModelArtifactLimit) || ctx.Err() != nil {
				return err
			}
			return fmt.Errorf("cannot inspect copy destination")
		}
		p = filepath.Join(p, seg)
		info, err := os.Lstat(p)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return fmt.Errorf("cannot inspect copy destination")
		}
		if info.Mode()&(os.ModeSymlink|os.ModeIrregular) != 0 {
			return fmt.Errorf("copy destination contains a link or irregular entry")
		}
		if p != dest && (!info.IsDir() || info.Mode().Type() != os.ModeDir) {
			return fmt.Errorf("copy destination parent is not a directory")
		}
		if p == dest && !info.Mode().IsRegular() {
			return fmt.Errorf("copy destination is not a regular file")
		}
	}
	realRoot, err := resolveExistingPath(nearestExistingDir(root))
	if err != nil {
		return fmt.Errorf("cannot resolve copy store")
	}
	realDest, err := resolveExistingPath(nearestExistingDir(filepath.Dir(dest)))
	if err != nil || (!pathWithinRoot(realRoot, realDest) && realRoot != realDest) {
		return fmt.Errorf("copy destination escapes resolved model store")
	}
	return nil
}

func customArtifactPlan(store engineModelStore, model string, list modelFileList) (copyPlan, error) {
	id, err := artifactExportID(list)
	if err != nil || id != list.ExportID {
		return copyPlan{}, fmt.Errorf("invalid custom export identity")
	}
	receiptData, err := json.Marshal(copyReceipt{Version: 1, List: list})
	if err != nil || len(receiptData) > maxCopyManifest {
		return copyPlan{}, fmt.Errorf("copy receipt exceeds bounds")
	}
	aliasDir := copyAliasDir(store.root, model)
	durability := copyDurabilityUncertain
	plan := copyPlan{layout: llamaArtifactsLayout, exportID: id, durability: &durability}
	for _, f := range list.Files {
		plan.files = append(plan.files, plannedFile{modelFile: f, dest: filepath.Join(aliasDir, id, filepath.FromSlash(f.Path))})
	}
	plan.preflight = func(ctx context.Context) error {
		receipt, err := readCopyReceipt(store, model)
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		if err != nil {
			return err
		}
		if receipt.List.ExportID != id {
			return copyConflict("copy conflict: this alias already has a different committed export")
		}
		return ctx.Err()
	}
	if err := plan.preflight(context.Background()); err != nil {
		return copyPlan{}, err
	}
	plan.finalize = func(ctx context.Context) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := plan.preflight(ctx); err != nil {
			return err
		}
		for _, f := range plan.files {
			if err := safeCopyDestination(store.root, f.dest); err != nil {
				return err
			}
			match, err := destMatches(ctx, f)
			if err != nil {
				return err
			}
			if !match {
				return fmt.Errorf("custom artifacts are incomplete")
			}
		}
		// An identical receipt is a verified no-op. Reading it establishes no
		// durability guarantee, so this result retains the uncertain qualifier.
		if _, err := readCopyReceipt(store, model); err == nil {
			durability = copyDurabilityUncertain
			return nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		path := filepath.Join(aliasDir, "copy.json")
		if err := safeCopyDestination(store.root, path); err != nil {
			return err
		}
		if err := os.MkdirAll(aliasDir, 0o755); err != nil {
			return err
		}
		durability, err = publishCopyReceipt(path, receiptData)
		return err
	}
	return plan, nil
}

// Unlike writeFileAtomic this copy-specific publication keeps the landed but
// uncertain outcome in the typed result; it never turns that outcome into an
// unqualified success.
func publishCopyReceipt(path string, data []byte) (copyDurability, error) {
	err := engineFileReplacer.WriteFile(path, data, 0o644)
	if errors.Is(err, durablefile.ErrReplaced) {
		return copyDurabilityUncertain, nil
	}
	if err != nil {
		return "", err
	}
	return copyDurabilityConfirmed, nil
}
