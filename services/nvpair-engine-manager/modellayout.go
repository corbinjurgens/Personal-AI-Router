// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// modellayout.go knows how each engine family stores a model on disk, which model copy
// needs on both ends: the source lists exactly the files that make up a model
// (modelfiles.go), and the destination decides where each one goes and in what
// order, so the engine only sees the model once it is complete (modelcopy.go).
//
//   - Ollama keeps content-addressed blobs (blobs/sha256-<hex>) and one manifest
//     per model (manifests/<host>/<namespace>/<name>/<tag>) under OLLAMA_MODELS,
//     ~/.ollama/models by default. A model exists for Ollama once its manifest
//     does, so the manifest is written last.
//   - llama.cpp's router keeps a Hugging Face hub cache under LLAMA_CACHE:
//     models--<owner>--<repo>/{refs/<branch>, snapshots/<commit>/<files>,
//     blobs/<oid>}. A model id "owner/repo:TAG" names the snapshot GGUF that
//     matches TAG (all of its splits), plus the multimodal projector beside it.
//     The selection mirrors llama.cpp's own (common/download.cpp, b11146).
//   - LM Studio keeps models under its models root, where `lms ls --json` maps a
//     model id to a path: a GGUF file, or a directory for MLX models.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
)

// errModelCopyUnsupported marks an engine whose model layout this service does
// not know how to copy.
var errModelCopyUnsupported = errors.New("model copy is not supported for this engine")

// engineModelStore is one engine's on-disk model store, as model copy sees it.
// adapter is the layout the engine's family selects (modelStoreAdapter), so a
// fork with its own ID and store is copied with its family's layout.
type engineModelStore struct {
	engine  string
	adapter string
	root    string // absolute
	cli     string // LM Studio's lms, resolved
}

// modelStoreFor resolves an engine's model store on this node.
func (e *Executor) modelStoreFor(engine string) (*engineState, engineModelStore, error) {
	st, err := e.state(engine)
	if err != nil {
		return nil, engineModelStore{}, err
	}
	store := engineModelStore{engine: engine, adapter: modelStoreAdapter(st.manifest.FamilyOrDefault())}
	switch store.adapter {
	case engines.NameOllama:
		store.root = engineEnvValue(st, "OLLAMA_MODELS")
		if store.root == "" && st.modelsDir != "" {
			store.root = filepath.Join(st.modelsDir, "models")
		}
	case engines.NameLlamaCPP:
		store.root = engineEnvValue(st, "LLAMA_CACHE")
		if store.root == "" {
			store.root = st.modelsDir
		}
	case engines.NameLMStudio:
		store.root = st.modelsDir
		if cli := st.plat.Runtime.CLI; cli != "" {
			store.cli = expandPath(cli)
		}
	default:
		return nil, engineModelStore{}, fmt.Errorf("%w: %q", errModelCopyUnsupported, engine)
	}
	if store.root == "" {
		return nil, engineModelStore{}, fmt.Errorf("engine %q declares no model store on this platform", engine)
	}
	store.root = filepath.Clean(store.root)
	if !filepath.IsAbs(store.root) {
		return nil, engineModelStore{}, fmt.Errorf("engine %q model store %q is not an absolute path", engine, store.root)
	}
	return st, store, nil
}

// engineEnvValue returns the value the engine is launched with for an
// environment variable that names a directory: the manifest's (or the user's
// saved launch settings') value, else the one this process inherited, which the
// engine inherits too. Empty when neither sets it.
func engineEnvValue(st *engineState, key string) string {
	st.opMu.Lock()
	st.mu.Lock()
	port := st.port
	st.mu.Unlock()
	launch, err := launchForState(st, port)
	st.opMu.Unlock()
	if err == nil {
		for k, v := range launch.Env {
			if environmentKey(k) == environmentKey(key) && strings.TrimSpace(v) != "" {
				return expandPath(v)
			}
		}
	}
	return expandPath(os.Getenv(key))
}

// modelSourceFiles lists the files that make up model in engine's store, with
// each file's resolved path. A sha256 is filled in only where the layout
// already names it (content-addressed blobs); hashModelFiles computes the rest.
func (e *Executor) modelSourceFiles(ctx context.Context, engine, model string) (modelFileList, error) {
	return e.modelSourceFilesWithAccess(ctx, engine, model, servingFileAccess{})
}

// Rechecks retain the original selection authority, including metadata for files
// which will only be reached after reading a selector or enumerating a directory.
func (e *Executor) modelSourceFilesExpected(ctx context.Context, engine, model string, access servingFileAccess, expected modelFileList) (modelFileList, error) {
	if expected.Engine != engine || expected.Model != model || expected.source == nil {
		return modelFileList{}, errModelInspectionChanged
	}
	return e.modelSourceFilesBound(ctx, engine, model, access, &expected)
}

func (e *Executor) modelSourceFilesWithAccess(ctx context.Context, engine, model string, access servingFileAccess) (modelFileList, error) {
	return e.modelSourceFilesBound(ctx, engine, model, access, nil)
}

func (e *Executor) modelSourceFilesBound(ctx context.Context, engine, model string, access servingFileAccess, expected *modelFileList) (modelFileList, error) {
	if err := ctx.Err(); err != nil {
		return modelFileList{}, err
	}
	st, store, err := e.modelStoreFor(engine)
	if err != nil {
		return modelFileList{}, err
	}
	var list modelFileList
	switch store.adapter {
	case engines.NameOllama:
		var old *sourceSnapshot
		if expected != nil {
			old = expected.source
			if old.engine != engine || old.root != store.root || old.inventory == nil ||
				e.nativeSourceCurrent(ctx, st, store, *old.inventory) != nil || validateSourceSnapshot(ctx, old) != nil {
				return modelFileList{}, errModelInspectionChanged
			}
		}
		observation, recognitionErr := e.recognizeModelSource(ctx, engine, model)
		if recognitionErr != nil {
			return modelFileList{}, recognitionErr
		}
		if err := e.nativeSourceCurrent(ctx, st, store, observation); err != nil {
			return modelFileList{}, err
		}
		if old != nil && (old.run != observation.runID || old.configuration != observation.launchDigest || !bytes.Equal(old.recognition, observation.raw)) {
			return modelFileList{}, errModelInspectionChanged
		}
		list, err = ollamaSourceFilesBound(ctx, store.root, model, access, old)
		if err == nil {
			current, recognitionErr := e.recognizeModelSource(ctx, engine, model)
			if recognitionErr != nil {
				return modelFileList{}, recognitionErr
			}
			if current.runID != observation.runID || current.launchDigest != observation.launchDigest || !bytes.Equal(current.raw, observation.raw) {
				return modelFileList{}, errModelInspectionChanged
			}
			err = e.nativeSourceCurrent(ctx, st, store, current)
			native := nativeSourceSnapshot(store, observation, list.Files)
			list.source.engine, list.source.run, list.source.configuration = native.engine, native.run, native.configuration
			list.source.recognition, list.source.inventory = native.recognition, native.inventory
			if err == nil {
				err = validateSourceSnapshot(ctx, list.source)
			}
		}
	case engines.NameLlamaCPP:
		list, err = e.llamaExportFilesBound(ctx, st, store, model, access, expected)
	case engines.NameLMStudio:
		list, err = e.lmstudioSourceFilesBound(ctx, st, store, model, access, expected)
	}
	if err != nil {
		return modelFileList{}, err
	}
	list.Engine, list.Model = engine, model
	return list, nil
}

// Recognition uses the inventory owner's fresh per-engine read, never the sweep
// cache. Store and launch comparisons remain with the selecting source owner.
func (e *Executor) recognizeModelSource(ctx context.Context, engine, model string) (engineInventory, error) {
	if err := ctx.Err(); err != nil {
		return engineInventory{}, err
	}
	observation := e.readEngineInventory(ctx, engine)
	switch observation.membership(model) {
	case inventoryAbsent:
		return engineInventory{}, fmt.Errorf("%w: native engine does not list the requested model", errModelNotFound)
	case inventoryPresent:
		return observation, nil
	default:
		return engineInventory{}, fmt.Errorf("native model recognition is unavailable or ambiguous")
	}
}

func (e *Executor) nativeSourceCurrent(ctx context.Context, st *engineState, store engineModelStore, observation engineInventory) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	current, currentStore, err := e.modelStoreFor(store.engine)
	if err != nil || current != st || currentStore != store || !observation.ok || observation.engine != store.engine {
		return fmt.Errorf("native model source configuration changed")
	}
	e.inventoryMu.Lock()
	valid := e.inventoryObservationCurrent(store.engine, observation.digests)
	e.inventoryMu.Unlock()
	st.mu.Lock()
	valid = valid && st.running && st.runID == observation.runID && st.launchDigest == observation.launchDigest && st.modelsDir == observation.modelsDir
	st.mu.Unlock()
	if !valid {
		return fmt.Errorf("native model recognition changed")
	}
	return nil
}

func nativeSourceSnapshot(store engineModelStore, observation engineInventory, files []modelFile) *sourceSnapshot {
	return &sourceSnapshot{
		engine: store.engine, run: observation.runID, configuration: observation.launchDigest, root: store.root,
		recognition: append([]byte(nil), observation.raw...), inventory: &observation,
		artifacts: append([]modelFile(nil), files...),
	}
}

// plannedFile is one file the destination must hold, and where.
type plannedFile struct {
	modelFile
	dest     string
	root     string
	exportID string
	// check, when set, runs on the verified download before it is renamed into
	// place (Ollama checks a manifest's blobs are all present first).
	check func(part string) error
}

// copyPlan is the destination's view of a copy: the files in the order they
// must be written (the one that makes the model visible last) and an
// engine-specific step once all are in place.
type copyPlan struct {
	root       string
	files      []plannedFile
	finalize   func(ctx context.Context) error
	layout     string
	exportID   string
	durability *copyDurability
	preflight  func(context.Context) error
}

// planModelCopy validates a source listing against this node's layout for the
// engine and maps each file to its destination.
func (e *Executor) planModelCopy(st *engineState, store engineModelStore, model string, list modelFileList) (copyPlan, error) {
	if list.Engine != store.engine || list.Model != model {
		return copyPlan{}, fmt.Errorf("source listing identity does not match request")
	}
	if len(list.Files) > maxCopyArtifacts {
		return copyPlan{}, fmt.Errorf("source listed too many artifacts")
	}
	seen := make(map[string]bool, len(list.Files))
	var total int64
	for _, f := range list.Files {
		if !validModelRelPath(f.Path) {
			return copyPlan{}, fmt.Errorf("source listed an unsafe path %q", f.Path)
		}
		folded := copyPathKey(f.Path)
		if seen[folded] {
			return copyPlan{}, fmt.Errorf("source listed %q twice", f.Path)
		}
		seen[folded] = true
		if !validSHA256(f.SHA256) {
			return copyPlan{}, fmt.Errorf("source listed %q without a valid sha256", f.Path)
		}
		if f.Size < 0 {
			return copyPlan{}, fmt.Errorf("source listed %q with a negative size", f.Path)
		}
		if f.Size > math.MaxInt64-total {
			return copyPlan{}, fmt.Errorf("source size total overflows")
		}
		total += f.Size
	}
	if len(list.Files) == 0 {
		return copyPlan{}, fmt.Errorf("source listed no files for %q", model)
	}
	var (
		plan copyPlan
		err  error
	)
	if list.Layout != "" {
		if list.Layout != llamaArtifactsLayout || store.adapter != engines.NameLlamaCPP {
			return copyPlan{}, fmt.Errorf("unsupported copy layout")
		}
		plan, err = customArtifactPlan(store, model, list)
	} else {
		switch store.adapter {
		case engines.NameOllama:
			plan, err = ollamaPlan(store.root, model, list)
		case engines.NameLlamaCPP:
			plan, err = e.llamacppPlan(st, store.root, model, list)
		case engines.NameLMStudio:
			plan, err = e.lmstudioPlan(st, store, list)
		default:
			return copyPlan{}, fmt.Errorf("%w: %q", errModelCopyUnsupported, store.engine)
		}
	}
	if err != nil {
		return copyPlan{}, err
	}
	for i := range plan.files {
		f := &plan.files[i]
		if !pathWithinRoot(store.root, f.dest) || filepath.Clean(f.dest) == store.root {
			return copyPlan{}, fmt.Errorf("%q would be written outside the model store", f.Path)
		}
		f.root = store.root
		f.exportID = list.ExportID
		if err := safeCopyDestination(store.root, f.dest); err != nil {
			return copyPlan{}, err
		}
	}
	plan.root = store.root
	return plan, nil
}

// ---- Ollama ----

const ollamaDefaultRegistry = "registry.ollama.ai"

var ollamaNameSegment = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_.-]*$`)

// ollamaManifestRel maps an Ollama model name ("llama3.2", "user/model:tag",
// "hf.co/owner/repo:Q4_K_M") to its manifest path relative to the models root.
func ollamaManifestRel(model string) (string, error) {
	name := strings.TrimSpace(model)
	tag := "latest"
	if i := strings.LastIndex(name, ":"); i > strings.LastIndex(name, "/") {
		name, tag = name[:i], name[i+1:]
	}
	parts := strings.Split(name, "/")
	switch len(parts) {
	case 1:
		parts = []string{ollamaDefaultRegistry, "library", parts[0]}
	case 2:
		parts = []string{ollamaDefaultRegistry, parts[0], parts[1]}
	case 3:
	default:
		return "", fmt.Errorf("invalid Ollama model name %q", model)
	}
	segs := append(parts, tag)
	for _, s := range segs {
		if !ollamaNameSegment.MatchString(s) || strings.HasSuffix(s, ".") {
			return "", fmt.Errorf("invalid Ollama model name %q", model)
		}
	}
	return "manifests/" + strings.Join(segs, "/"), nil
}

type ollamaManifest struct {
	Config ollamaLayer   `json:"config"`
	Layers []ollamaLayer `json:"layers"`
}

type ollamaLayer struct {
	Digest string `json:"digest"`
	Size   int64  `json:"size"`
}

// blobs returns the manifest's distinct blob digests (hex) with their declared
// sizes, config first.
func (m ollamaManifest) blobs() ([]ollamaLayer, error) {
	var out []ollamaLayer
	seen := map[string]bool{}
	for _, l := range append([]ollamaLayer{m.Config}, m.Layers...) {
		if l.Digest == "" {
			continue
		}
		hex, ok := strings.CutPrefix(l.Digest, "sha256:")
		if !ok || !validSHA256(hex) {
			return nil, fmt.Errorf("manifest names an unsupported digest %q", l.Digest)
		}
		if seen[hex] {
			continue
		}
		seen[hex] = true
		out = append(out, ollamaLayer{Digest: hex, Size: l.Size})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("manifest names no blobs")
	}
	return out, nil
}

func parseOllamaManifest(data []byte) (ollamaManifest, error) {
	var m ollamaManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return m, fmt.Errorf("parse manifest: %w", err)
	}
	return m, nil
}

func readSmallFile(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("%s is larger than %d bytes", path, limit)
	}
	return data, nil
}

const ollamaManifestLimit = 4 << 20

func ollamaSourceFiles(root, model string) (modelFileList, error) {
	return ollamaSourceFilesWithAccess(context.Background(), root, model, servingFileAccess{})
}

func ollamaSourceFilesWithAccess(ctx context.Context, root, model string, access servingFileAccess) (modelFileList, error) {
	return ollamaSourceFilesBound(ctx, root, model, access, nil)
}

func ollamaSourceFilesBound(ctx context.Context, root, model string, access servingFileAccess, expected *sourceSnapshot) (modelFileList, error) {
	if expected != nil && (expected.root != root || len(expected.selectors) != 1 || validateSourceSnapshot(ctx, expected) != nil) {
		return modelFileList{}, errModelInspectionChanged
	}
	if expected != nil && (expected.selectors[0].binding == nil || int64(len(expected.selectors[0].bytes)) != expected.selectors[0].file.size) {
		return modelFileList{}, errModelInspectionChanged
	}
	rel, err := ollamaManifestRel(model)
	if err != nil {
		return modelFileList{}, fmt.Errorf("%w: %v", errModelNotFound, err)
	}
	source := &sourceSnapshot{root: root}
	scan := &llamaCacheScan{snapshot: source, expected: expected}
	path := filepath.Join(root, filepath.FromSlash(rel))
	for _, dir := range []string{root, filepath.Dir(path)} {
		if _, err := scan.directory(dir); err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return modelFileList{}, fmt.Errorf("%w: Ollama has no model %q on this node", errModelNotFound, model)
			}
			return modelFileList{}, err
		}
	}
	manifest, err := nativeSourceArtifact(root, rel, "", expected)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return modelFileList{}, fmt.Errorf("%w: Ollama has no model %q on this node", errModelNotFound, model)
		}
		return modelFileList{}, err
	}
	selector := selectorEvidence{file: sourceEvidence{configured: manifest.configured, target: manifest.abs,
		info: manifest.info, size: manifest.Size, mtime: manifest.info.ModTime().UnixNano()}}
	if expected != nil {
		selector = expected.selectors[0]
		if selector.file.configured != path {
			return modelFileList{}, errModelInspectionChanged
		}
	}
	data, binding, err := access.sourceExpected(ctx, selector.file, ollamaManifestLimit, false)
	if err != nil {
		return modelFileList{}, err
	}
	if expected != nil && (!bytes.Equal(selector.bytes, data) || selector.binding == nil || *selector.binding != *binding) {
		return modelFileList{}, errModelInspectionChanged
	}
	source.selectors = []selectorEvidence{{file: selector.file, bytes: append([]byte(nil), data...), binding: binding}}
	m, err := parseOllamaManifest(data)
	if err != nil {
		return modelFileList{}, err
	}
	declared := map[string]int64{}
	for _, layer := range append([]ollamaLayer{m.Config}, m.Layers...) {
		if layer.Digest == "" {
			continue
		}
		if size, ok := declared[layer.Digest]; layer.Size < 0 || (ok && size != layer.Size) {
			return modelFileList{}, fmt.Errorf("manifest has inconsistent blob sizes")
		}
		declared[layer.Digest] = layer.Size
	}
	blobs, err := m.blobs()
	if err != nil {
		return modelFileList{}, err
	}
	if len(blobs) >= maxCopyArtifacts {
		return modelFileList{}, errModelArtifactLimit
	}
	if _, err := scan.directory(filepath.Join(root, "blobs")); err != nil {
		return modelFileList{}, err
	}
	list := modelFileList{source: source}
	for _, b := range blobs {
		if err := ctx.Err(); err != nil {
			return modelFileList{}, err
		}
		brel := "blobs/sha256-" + b.Digest
		file, err := nativeSourceArtifact(root, brel, b.Digest, expected)
		if err != nil {
			return modelFileList{}, fmt.Errorf("model %q is incomplete on this node (%s): %w", model, brel, err)
		}
		if file.Size != b.Size {
			return modelFileList{}, fmt.Errorf("model %q is incomplete on this node: %s is %d bytes, manifest says %d", model, brel, file.Size, b.Size)
		}
		// The blob's name is its sha256; the destination verifies it.
		list.Files = append(list.Files, file)
	}
	list.Files = append(list.Files, manifest)
	if err := finishNativeSource(ctx, &list, expected); err != nil {
		return modelFileList{}, err
	}
	// Re-read only the bound selector after the complete artifact preflight.
	data, finalBinding, err := access.sourceExpected(ctx, selector.file, ollamaManifestLimit, false)
	if err != nil || !bytes.Equal(data, source.selectors[0].bytes) || finalBinding == nil || *finalBinding != *binding {
		return modelFileList{}, errModelInspectionChanged
	}
	if err := validateSourceSnapshot(ctx, source); err != nil {
		return modelFileList{}, err
	}
	return list, nil
}

// Keep native file admission at the existing confinement owner. Rechecks look
// up the old evidence before touching a newly selected path and never renew it.
func nativeSourceArtifact(root, rel, digest string, expected *sourceSnapshot) (modelFile, error) {
	path := filepath.Join(root, filepath.FromSlash(rel))
	var old *modelFile
	if expected != nil {
		for i := range expected.artifacts {
			if expected.artifacts[i].Path == rel {
				old = &expected.artifacts[i]
				break
			}
		}
		if old == nil || old.configured != path || old.SHA256 != digest || old.info == nil ||
			validateSource(sourceEvidence{configured: old.configured, target: old.abs, info: old.info,
				size: old.Size, mtime: old.info.ModTime().UnixNano()}) != nil {
			return modelFile{}, errModelInspectionChanged
		}
	}
	real, info, err := confinedRegularFile(root, path)
	if err != nil {
		return modelFile{}, err
	}
	if old != nil {
		if old.abs != real || !sameSource(old.info, info, false) {
			return modelFile{}, errModelInspectionChanged
		}
		return *old, nil
	}
	evidence, err := captureSource(path, false)
	if err != nil || evidence.target != real || !sameSource(evidence.info, info, false) {
		return modelFile{}, errModelInspectionChanged
	}
	return modelFile{Path: rel, Size: evidence.size, SHA256: digest,
		configured: evidence.configured, abs: evidence.target, info: evidence.info}, nil
}

func finishNativeSource(ctx context.Context, list *modelFileList, expected *sourceSnapshot) error {
	if expected != nil {
		if len(list.Files) != len(expected.artifacts) || len(list.source.directories) != len(expected.directories) || len(list.source.selectors) != len(expected.selectors) {
			return errModelInspectionChanged
		}
		for i, dir := range list.source.directories {
			if dir.configured != expected.directories[i].configured {
				return errModelInspectionChanged
			}
		}
		for i, file := range list.Files {
			old := expected.artifacts[i]
			if file.Path != old.Path || file.Size != old.Size || file.SHA256 != old.SHA256 || file.Role != old.Role ||
				file.configured != old.configured || file.abs != old.abs || !sameSource(file.info, old.info, false) {
				return errModelInspectionChanged
			}
		}
	}
	list.source.artifacts = append([]modelFile(nil), list.Files...)
	return validateSourceSnapshot(ctx, list.source)
}

func ollamaPlan(root, model string, list modelFileList) (copyPlan, error) {
	rel, err := ollamaManifestRel(model)
	if err != nil {
		return copyPlan{}, err
	}
	var plan copyPlan
	var manifest *plannedFile
	for _, f := range list.Files {
		pf := plannedFile{modelFile: f, dest: filepath.Join(root, filepath.FromSlash(f.Path))}
		if f.Path == rel {
			manifest = &pf
			continue
		}
		hex, ok := strings.CutPrefix(f.Path, "blobs/sha256-")
		if !ok || !validSHA256(hex) {
			return copyPlan{}, fmt.Errorf("unexpected file %q in an Ollama model listing", f.Path)
		}
		if f.SHA256 != hex {
			return copyPlan{}, fmt.Errorf("blob %q is listed with a different sha256", f.Path)
		}
		plan.files = append(plan.files, pf)
	}
	if manifest == nil {
		return copyPlan{}, fmt.Errorf("the source's listing has no manifest %q for %q", rel, model)
	}
	// The manifest is what makes the model exist for Ollama: write it last, and
	// only once every blob it names is in place.
	manifest.check = func(part string) error {
		data, err := readSmallFile(part, ollamaManifestLimit)
		if err != nil {
			return err
		}
		m, err := parseOllamaManifest(data)
		if err != nil {
			return err
		}
		blobs, err := m.blobs()
		if err != nil {
			return err
		}
		for _, b := range blobs {
			info, err := os.Stat(filepath.Join(root, "blobs", "sha256-"+b.Digest))
			if err != nil {
				return fmt.Errorf("manifest names blob %s, which is not on this node", b.Digest)
			}
			if b.Size > 0 && info.Size() != b.Size {
				return fmt.Errorf("blob %s is %d bytes, manifest says %d", b.Digest, info.Size(), b.Size)
			}
		}
		return nil
	}
	plan.files = append(plan.files, *manifest)
	// Ollama reads manifests from disk for every listing, so nothing else is
	// needed for the model to appear.
	plan.finalize = func(context.Context) error { return nil }
	return plan, nil
}

// ---- llama.cpp ----

var (
	hfRepoIDRe = regexp.MustCompile(`^[A-Za-z0-9_]+(?:[.-][A-Za-z0-9_]+)*/[A-Za-z0-9_]+(?:[.-][A-Za-z0-9_]+)*$`)
	hfTagRe    = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)
	commitRe   = regexp.MustCompile(`^[0-9a-fA-F]{40}$`)
)

// splitLlamaModel splits "owner/repo[:TAG]" like llama.cpp's
// common_download_split_repo_tag.
func splitLlamaModel(model string) (repo, tag string, err error) {
	parts := strings.Split(strings.TrimSpace(model), ":")
	repo = parts[0]
	if len(parts) > 1 {
		tag = parts[len(parts)-1]
	}
	if len(repo) > 256 || !hfRepoIDRe.MatchString(repo) {
		return "", "", fmt.Errorf("invalid llama.cpp model id %q, expected <owner>/<repo>[:quant]", model)
	}
	if tag != "" && !hfTagRe.MatchString(tag) {
		return "", "", fmt.Errorf("invalid llama.cpp model id %q, expected <owner>/<repo>[:quant]", model)
	}
	return repo, tag, nil
}

func llamaRepoDir(root, repo string) string {
	return filepath.Join(root, "models--"+strings.ReplaceAll(repo, "/", "--"))
}

// llamaCachedRef returns the snapshot commit llama.cpp would use for a cached
// repo: refs/main, else the first valid ref.
func llamaCachedRef(repoDir string) string {
	commit, err := (&modelArtifactScan{}).cachedRef(context.Background(), repoDir, servingFileAccess{})
	if err != nil {
		return ""
	}
	return commit
}

type ggufSplit struct {
	prefix string
	tag    string
	index  int
	count  int
}

var (
	ggufSplitRe = regexp.MustCompile(`(?i)^(.+)-([0-9]{5})-of-([0-9]{5})$`)
	ggufTagRe   = regexp.MustCompile(`(?i)[-.]([A-Z0-9_]+)$`)
)

func ggufSplitInfo(path string) (ggufSplit, bool) {
	prefix, ok := strings.CutSuffix(path, ".gguf")
	if !ok {
		return ggufSplit{}, false
	}
	s := ggufSplit{index: 1, count: 1}
	if m := ggufSplitRe.FindStringSubmatch(prefix); m != nil {
		s.index, _ = strconv.Atoi(m[2])
		s.count, _ = strconv.Atoi(m[3])
		prefix = m[1]
	}
	if m := ggufTagRe.FindStringSubmatch(prefix); m != nil {
		s.tag = strings.ToUpper(m[1])
	}
	s.prefix = prefix
	return s, true
}

func quantBits(tag string) int {
	i := strings.IndexAny(tag, "0123456789")
	if i < 0 {
		return 0
	}
	j := i
	for j < len(tag) && tag[j] >= '0' && tag[j] <= '9' {
		j++
	}
	n, _ := strconv.Atoi(tag[i:j])
	return n
}

func ggufIsModel(path string) bool {
	if !strings.HasSuffix(path, ".gguf") {
		return false
	}
	base := path[strings.LastIndex(path, "/")+1:]
	for _, sidecar := range []string{"mmproj", "imatrix", "mtp-", "eagle3-", "dflash-", "dspark-"} {
		if strings.Contains(base, sidecar) {
			return false
		}
	}
	return true
}

// llamaModelFileSet picks the files of a cached repo snapshot that make up the
// model for tag: the best matching GGUF with all of its splits, then the
// multimodal projector llama.cpp would pair with it, if one is cached.
func llamaModelFileSet(paths []string, tag string) []string {
	sorted := append([]string(nil), paths...)
	sort.Strings(sorted)
	tags := []string{tag}
	if tag == "" {
		tags = []string{"Q4_K_M", "Q8_0"}
	}
	primary := ""
	for _, t := range tags {
		re := regexp.MustCompile(`(?i)` + regexp.QuoteMeta(t) + `[.-]`)
		for _, p := range sorted {
			if !ggufIsModel(p) || !re.MatchString(p) {
				continue
			}
			if s, _ := ggufSplitInfo(p); s.count > 1 && s.index != 1 {
				continue
			}
			primary = p
			break
		}
		if primary != "" {
			break
		}
	}
	if primary == "" && tag == "" {
		for _, p := range sorted {
			if !ggufIsModel(p) {
				continue
			}
			if s, _ := ggufSplitInfo(p); s.count > 1 && s.index != 1 {
				continue
			}
			primary = p
			break
		}
	}
	if primary == "" {
		return nil
	}
	var out []string
	ps, _ := ggufSplitInfo(primary)
	if ps.count <= 1 {
		out = append(out, primary)
	} else {
		for _, p := range sorted {
			if s, ok := ggufSplitInfo(p); ok && s.count == ps.count && s.prefix == ps.prefix {
				out = append(out, p)
			}
		}
	}
	if mmproj := llamaBestSibling(sorted, primary, "mmproj"); mmproj != "" {
		out = append(out, mmproj)
	}
	return out
}

// llamaBestSibling mirrors llama.cpp's find_best_sibling with no tag: among GGUFs
// containing keyword in the model's directory or an ancestor of it, prefer the
// deepest, then the closest quantization.
func llamaBestSibling(paths []string, model, keyword string) string {
	ms, _ := ggufSplitInfo(model)
	modelBits := quantBits(ms.tag)
	modelDir := strings.Split(model, "/")
	modelDir = modelDir[:len(modelDir)-1]
	best, bestDepth, bestDiff := "", 0, 0
	for _, p := range paths {
		if !strings.HasSuffix(p, ".gguf") || !strings.Contains(p, keyword) {
			continue
		}
		dir := strings.Split(p, "/")
		dir = dir[:len(dir)-1]
		if len(dir) > len(modelDir) {
			continue
		}
		match := true
		for i := range dir {
			if dir[i] != modelDir[i] {
				match = false
				break
			}
		}
		if !match {
			continue
		}
		s, _ := ggufSplitInfo(p)
		diff := quantBits(s.tag) - modelBits
		if diff < 0 {
			diff = -diff
		}
		if best == "" || len(dir) > bestDepth || (len(dir) == bestDepth && diff < bestDiff) {
			best, bestDepth, bestDiff = p, len(dir), diff
		}
	}
	return best
}

func llamacppSourceFiles(root, model string) (modelFileList, error) {
	return boundedLlamaCacheArtifacts(context.Background(), engineModelStore{root: root, adapter: engines.NameLlamaCPP}, model, servingFileAccess{})
}

// ReadDir(n) bounds each allocation; the shared budget covers refs and every
// visited snapshot entry, including entries which are not model artifacts.
const modelArtifactScanEntries = 100000

type modelArtifactScan struct{ entries int }

func (b *modelArtifactScan) directory(ctx context.Context, path string, visit func(os.DirEntry) error) error {
	return b.directoryWithAccess(ctx, path, servingFileAccess{}, visit)
}

func (b *modelArtifactScan) directoryWithAccess(ctx context.Context, path string, access servingFileAccess, visit func(os.DirEntry) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	f, err := access.openDirectory(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for {
		entries, err := f.ReadDir(128)
		for _, entry := range entries {
			if err := ctx.Err(); err != nil {
				return err
			}
			if b.entries >= modelArtifactScanEntries || len(entry.Name()) > 1024 {
				return errModelArtifactLimit
			}
			b.entries++
			if err := visit(entry); err != nil {
				return err
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func (b *modelArtifactScan) directoryExpected(ctx context.Context, evidence sourceEvidence, access servingFileAccess, visit func(os.DirEntry) error) error {
	entries, err := access.readDirectory(ctx, evidence, modelArtifactScanEntries-b.entries)
	if err != nil {
		return err
	}
	// Reserve the whole batch before recursion so buffered sibling entries
	// remain charged to the shared budget while a child is enumerated.
	b.entries += len(entries)
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(entry.Name()) > 1024 {
			return errModelArtifactLimit
		}
		if err := visit(entry); err != nil {
			return err
		}
	}
	return nil
}

// HF selection retains metadata before each first read. Rechecks use only the
// old records, with a complete metadata preflight before any content access.
type llamaCacheScan struct {
	budget   *modelArtifactScan
	access   servingFileAccess
	snapshot *sourceSnapshot
	expected *sourceSnapshot
	strict   bool
}

func (s *llamaCacheScan) directory(path string) (sourceEvidence, error) {
	path, err := filepath.Abs(path)
	if err != nil {
		return sourceEvidence{}, err
	}
	for _, evidence := range s.snapshot.directories {
		if evidence.configured == path {
			return evidence, nil
		}
	}
	var evidence sourceEvidence
	if s.expected != nil {
		for _, old := range s.expected.directories {
			if old.configured == path {
				evidence = old
				break
			}
		}
		err = validateSource(evidence)
	} else {
		evidence, err = captureSource(path, true)
	}
	if err != nil {
		return sourceEvidence{}, err
	}
	for _, previous := range s.snapshot.directories {
		if os.SameFile(previous.info, evidence.info) {
			return sourceEvidence{}, errNoArtifactBinding
		}
	}
	s.snapshot.directories = append(s.snapshot.directories, evidence)
	return evidence, nil
}

func (s *llamaCacheScan) enumerate(ctx context.Context, path string, visit func(os.DirEntry) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	evidence, err := s.directory(path)
	if err != nil {
		return err
	}
	return s.budget.directoryExpected(ctx, evidence, s.access, visit)
}

func (s *llamaCacheScan) selector(ctx context.Context, path, target string, info os.FileInfo) ([]byte, error) {
	var evidence sourceEvidence
	var old selectorEvidence
	var err error
	if s.expected != nil {
		i := len(s.snapshot.selectors)
		if i >= len(s.expected.selectors) || s.expected.selectors[i].file.configured != path {
			return nil, errModelInspectionChanged
		}
		old = s.expected.selectors[i]
		evidence = old.file
		err = validateSource(evidence)
	} else {
		evidence, err = captureSource(path, false)
	}
	if err != nil {
		return nil, err
	}
	if evidence.target != target || !sameSource(evidence.info, info, false) {
		return nil, errModelInspectionChanged
	}
	if evidence.size > 4096 {
		return nil, errModelArtifactLimit
	}
	data, binding, err := s.access.sourceExpected(ctx, evidence, 4096, true)
	if err != nil {
		return nil, err
	}
	if s.expected != nil && (!bytes.Equal(old.bytes, data) || old.binding == nil || *old.binding != *binding) {
		return nil, errModelInspectionChanged
	}
	s.snapshot.selectors = append(s.snapshot.selectors, selectorEvidence{file: evidence, bytes: data, binding: binding})
	return data, nil
}

func validateLlamaSource(ctx context.Context, snapshot *sourceSnapshot) error {
	if snapshot == nil || len(snapshot.selectors) == 0 {
		return errModelInspectionChanged
	}
	return validateSourceSnapshot(ctx, snapshot)
}

func validateSourceSnapshot(ctx context.Context, snapshot *sourceSnapshot) error {
	if snapshot == nil || len(snapshot.directories) == 0 || len(snapshot.artifacts) == 0 {
		return errModelInspectionChanged
	}
	check := func(evidence sourceEvidence) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		return validateSource(evidence)
	}
	for _, evidence := range snapshot.directories {
		if err := check(evidence); err != nil {
			return err
		}
	}
	for _, selector := range snapshot.selectors {
		if err := check(selector.file); err != nil {
			return err
		}
	}
	for _, artifact := range snapshot.artifacts {
		if artifact.info == nil {
			return errModelInspectionChanged
		}
		if err := check(sourceEvidence{configured: artifact.configured, target: artifact.abs, info: artifact.info,
			size: artifact.Size, mtime: artifact.info.ModTime().UnixNano()}); err != nil {
			return err
		}
	}
	return nil
}

func (b *modelArtifactScan) cachedRef(ctx context.Context, repoDir string, access servingFileAccess) (string, error) {
	scan := &llamaCacheScan{budget: b, access: access, snapshot: &sourceSnapshot{}, strict: true}
	return scan.cachedRef(ctx, repoDir)
}

func (s *llamaCacheScan) cachedRef(ctx context.Context, repoDir string) (string, error) {
	refs := filepath.Join(repoDir, "refs")
	if _, err := s.directory(refs); err != nil {
		return "", err
	}
	read := func(name string) (string, error) {
		path := filepath.Join(refs, name)
		var strictTarget string
		var strictInfo os.FileInfo
		if s.strict {
			var err error
			strictTarget, strictInfo, err = confinedRegularFile(repoDir, path)
			if err != nil {
				if errors.Is(err, os.ErrNotExist) {
					return "", os.ErrNotExist
				}
				return "", errNoArtifactBinding
			}
		}
		evidence, err := captureSource(path, false)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return "", os.ErrNotExist
			}
			return "", errNoArtifactBinding
		}
		if s.strict && (evidence.target != strictTarget || !sameSource(evidence.info, strictInfo, false)) {
			return "", errNoArtifactBinding
		}
		if evidence.size > 4096 {
			return "", errModelArtifactLimit
		}
		data, err := s.selector(ctx, path, evidence.target, evidence.info)
		if err != nil {
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			return "", errModelInspectionChanged
		}
		line, _, _ := strings.Cut(string(data), "\n")
		line = strings.TrimSpace(line)
		if !commitRe.MatchString(line) {
			return "", nil
		}
		return line, nil
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	commit, err := read("main")
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	if commit != "" {
		return commit, nil
	}
	var names []string
	if err := s.enumerate(ctx, refs, func(entry os.DirEntry) error {
		path := filepath.Join(refs, entry.Name())
		link, err := removalEntryIsLink(path, entry.Type())
		if err != nil {
			return errNoArtifactBinding
		}
		if entry.Type().IsRegular() || link {
			names = append(names, entry.Name())
		} else if !entry.IsDir() {
			return errNoArtifactBinding
		}
		return nil
	}); err != nil {
		return "", err
	}
	sort.Strings(names)
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		commit, err := read(name)
		if err != nil {
			return "", err
		}
		if commit != "" {
			return commit, nil
		}
	}
	return "", errModelNotFound
}

func boundedLlamaCacheArtifacts(ctx context.Context, store engineModelStore, model string, access servingFileAccess) (modelFileList, error) {
	return resolveBoundedLlamaCacheArtifacts(ctx, store, model, access, nil)
}

func boundedLlamaCacheArtifactsExpected(ctx context.Context, store engineModelStore, model string, access servingFileAccess, expected modelFileList) (modelFileList, error) {
	if expected.Engine != store.engine || expected.Model != model || expected.source == nil || expected.source.root != store.root || expected.source.engine != store.engine {
		return modelFileList{}, errModelInspectionChanged
	}
	if err := validateLlamaSource(ctx, expected.source); err != nil {
		return modelFileList{}, err
	}
	list, err := resolveBoundedLlamaCacheArtifacts(ctx, store, model, access, expected.source)
	if err != nil {
		return modelFileList{}, err
	}
	if list.Revision != expected.Revision || list.binding != expected.binding {
		return modelFileList{}, errModelInspectionChanged
	}
	return list, nil
}

func resolveBoundedLlamaCacheArtifacts(ctx context.Context, store engineModelStore, model string, access servingFileAccess, expected *sourceSnapshot) (modelFileList, error) {
	if err := ctx.Err(); err != nil {
		return modelFileList{}, err
	}
	root := store.root
	repo, tag, err := splitLlamaModel(model)
	if err != nil {
		return modelFileList{}, fmt.Errorf("%w: %v", errModelNotFound, err)
	}
	repoDir := llamaRepoDir(root, repo)
	scan := access.scan
	if scan == nil {
		scan = &modelArtifactScan{}
	}
	source := &sourceSnapshot{engine: store.engine, root: root}
	hf := &llamaCacheScan{budget: scan, access: access, snapshot: source, expected: expected}
	for _, dir := range []string{root, repoDir} {
		if _, err := hf.directory(dir); err != nil {
			return modelFileList{}, err
		}
	}
	commit, err := hf.cachedRef(ctx, repoDir)
	if err != nil {
		if errors.Is(err, errModelArtifactLimit) || errors.Is(err, errNoArtifactBinding) || ctx.Err() != nil {
			return modelFileList{}, err
		}
		return modelFileList{}, fmt.Errorf("%w: llama.cpp has no cached %q on this node", errModelNotFound, repo)
	}
	snap := filepath.Join(repoDir, "snapshots", commit)
	if _, err := hf.directory(filepath.Join(repoDir, "snapshots")); err != nil {
		return modelFileList{}, err
	}
	var rels []string
	seenPaths := map[string]bool{}
	var walk func(string, int) error
	walk = func(dir string, depth int) error {
		if depth > 64 {
			return errModelArtifactLimit
		}
		return hf.enumerate(ctx, dir, func(d os.DirEntry) error {
			path := filepath.Join(dir, d.Name())
			link, err := removalEntryIsLink(path, d.Type())
			if err != nil {
				return errNoArtifactBinding
			}
			isDir := d.IsDir()
			if link {
				info, err := os.Stat(path)
				if err != nil || (!info.IsDir() && !info.Mode().IsRegular()) {
					return errNoArtifactBinding
				}
				isDir = info.IsDir()
			} else if !isDir && !d.Type().IsRegular() {
				return errNoArtifactBinding
			}
			if isDir {
				return walk(path, depth+1)
			}
			if !strings.HasSuffix(d.Name(), ".gguf") {
				return nil
			}
			if len(rels) >= maxCopyArtifacts {
				return errModelArtifactLimit
			}
			rel, err := filepath.Rel(snap, path)
			if err != nil || !validModelRelPath(filepath.ToSlash(rel)) {
				return fmt.Errorf("unsafe cache artifact path")
			}
			key := copyPathKey(filepath.ToSlash(rel))
			if seenPaths[key] {
				return errNoArtifactBinding
			}
			seenPaths[key] = true
			rels = append(rels, filepath.ToSlash(rel))
			return nil
		})
	}
	err = walk(snap, 0)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return modelFileList{}, fmt.Errorf("%w: llama.cpp has no cached %q on this node", errModelNotFound, repo)
		}
		return modelFileList{}, err
	}
	selected := llamaModelFileSet(rels, tag)
	if len(selected) == 0 {
		return modelFileList{}, fmt.Errorf("%w: llama.cpp has no cached GGUF for %q on this node", errModelNotFound, model)
	}
	blobs, _ := resolveExistingPath(filepath.Join(repoDir, "blobs"))
	list := modelFileList{Engine: store.engine, Model: model, Revision: commit, source: source}
	for _, rel := range selected {
		if err := ctx.Err(); err != nil {
			return modelFileList{}, err
		}
		if !validModelRelPath(rel) {
			return modelFileList{}, fmt.Errorf("cached file %q has a name this service cannot copy safely", rel)
		}
		path := filepath.Join(snap, filepath.FromSlash(rel))
		evidence, err := captureSource(path, false)
		if err != nil || !artifactPathSafe(path) || !artifactPathSafe(evidence.target) {
			return modelFileList{}, errModelInspectionChanged
		}
		real, info := evidence.target, evidence.info
		role := "model"
		if !ggufIsModel(rel) {
			role = "companion"
		}
		f := modelFile{Path: rel, Role: role, Size: info.Size(), abs: real, configured: evidence.configured, info: evidence.info}
		// A snapshot entry that links to an LFS blob is named by its sha256.
		if blobs != "" && filepath.Dir(real) == blobs && validSHA256(filepath.Base(real)) {
			f.SHA256 = filepath.Base(real)
		}
		list.Files = append(list.Files, f)
	}
	// Compare the complete selection before sampling any selected artifact.
	if expected != nil {
		if len(list.Files) != len(expected.artifacts) || len(source.selectors) != len(expected.selectors) || len(source.directories) != len(expected.directories) {
			return modelFileList{}, errModelInspectionChanged
		}
		for i, dir := range source.directories {
			if dir.configured != expected.directories[i].configured {
				return modelFileList{}, errModelInspectionChanged
			}
		}
		for i, file := range list.Files {
			old := expected.artifacts[i]
			if file.Path != old.Path || file.Role != old.Role || file.configured != old.configured || file.abs != old.abs || !sameSource(file.info, old.info, false) {
				return modelFileList{}, errModelInspectionChanged
			}
			list.Files[i] = old
		}
	}
	for i := range list.Files {
		file := &list.Files[i]
		binding, err := access.bindingExpected(ctx, file.configured, file.abs, file.info)
		if err != nil || (expected != nil && binding.SampleSHA256 != file.sample) {
			return modelFileList{}, errModelInspectionChanged
		}
		file.sample = binding.SampleSHA256
	}
	if err := validateResolvedArtifacts(list); err != nil {
		return modelFileList{}, err
	}
	list.binding = exportBinding(list, struct{ Root, Revision string }{root, commit})
	source.artifacts = append([]modelFile(nil), list.Files...)
	if err := validateLlamaSource(ctx, source); err != nil {
		return modelFileList{}, err
	}
	for _, selector := range source.selectors {
		data, binding, err := access.sourceExpected(ctx, selector.file, 4096, true)
		if err != nil || !bytes.Equal(data, selector.bytes) || *binding != *selector.binding {
			return modelFileList{}, errModelInspectionChanged
		}
	}
	if err := validateLlamaSource(ctx, source); err != nil {
		return modelFileList{}, err
	}
	return list, nil
}

func (e *Executor) llamacppPlan(st *engineState, root, model string, list modelFileList) (copyPlan, error) {
	repo, _, err := splitLlamaModel(model)
	if err != nil {
		return copyPlan{}, err
	}
	repoDir := llamaRepoDir(root, repo)
	// A repo this node already caches keeps its own revision, so the files it
	// already holds stay listed; a new one takes the source's.
	commit := llamaCachedRef(repoDir)
	writeRef := commit == ""
	if writeRef {
		commit = list.Revision
		if !commitRe.MatchString(commit) {
			return copyPlan{}, fmt.Errorf("the source listed no valid llama.cpp cache revision for %q", model)
		}
	}
	snap := filepath.Join(repoDir, "snapshots", commit)
	var plan copyPlan
	var last []plannedFile
	for _, f := range list.Files {
		if !strings.HasSuffix(f.Path, ".gguf") {
			return copyPlan{}, fmt.Errorf("unexpected file %q in a llama.cpp model listing", f.Path)
		}
		pf := plannedFile{modelFile: f, dest: filepath.Join(snap, filepath.FromSlash(f.Path))}
		// llama.cpp lists a cached model by its first GGUF split, so that file
		// goes last and the model appears only once the rest are in place.
		if s, _ := ggufSplitInfo(f.Path); ggufIsModel(f.Path) && s.index == 1 {
			last = append(last, pf)
			continue
		}
		plan.files = append(plan.files, pf)
	}
	plan.files = append(plan.files, last...)
	plan.finalize = func(ctx context.Context) error {
		if writeRef {
			if err := writeFileAtomic(filepath.Join(repoDir, "refs", "main"), []byte(commit)); err != nil {
				return fmt.Errorf("write llama.cpp cache ref: %w", err)
			}
		}
		e.reloadLlamaModels(ctx, st)
		return nil
	}
	return plan, nil
}

// reloadLlamaModels asks a running llama.cpp router to rescan its cache
// (GET /models?reload=1), so a copied model is listed without a restart. It is
// best effort: a router that misses it rescans when next started.
func (e *Executor) reloadLlamaModels(ctx context.Context, st *engineState) {
	if _, err := e.reloadLlamaModelsResult(ctx, st); err != nil {
		slog.Warn("llama.cpp model rescan failed")
	}
}

// reloadLlamaModelsResult forces a native cache reconstruction. Unlike the
// copy-finalization wrapper, a caller using absence as proof must check errors.
func (e *Executor) reloadLlamaModelsResult(ctx context.Context, st *engineState) (json.RawMessage, error) {
	st.mu.Lock()
	running, port := st.running, st.port
	st.mu.Unlock()
	if !running || port <= 0 {
		return nil, errors.New("engine is not running")
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/models?reload=1", port), nil)
	if err != nil {
		return nil, err
	}
	// A redirect can discard reload=1 and return an ordinary cached listing.
	// Only this exact reconstruction request can establish absence.
	client := *e.client
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := client.Do(req)
	if err != nil {
		return nil, errors.New("model rescan failed")
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8*1024*1024+1))
	if err != nil || len(data) > 8*1024*1024 || !json.Valid(data) {
		return nil, errors.New("invalid model rescan response")
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("model rescan refused")
	}
	return data, nil
}

// writeFileAtomic durably replaces path. A landed replacement with uncertain
// durability is adopted and reported as a warning.
func writeFileAtomic(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if err := engineFileReplacer.WriteFile(path, data, 0o644); err != nil {
		if !errors.Is(err, durablefile.ErrReplaced) {
			return err
		}
		slog.Warn("engine file replacement landed with uncertain durability", "path", path, "err", err)
	}
	return nil
}

// ---- LM Studio ----

// lmsListModels runs `lms ls --json`. A variable so tests can stand in for the
// CLI.
var lmsListModels = func(ctx context.Context, e *Executor, cli string) ([]lmsListEntry, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	return e.lmsList(ctx, cli)
}

func (e *Executor) lmstudioSourceFiles(ctx context.Context, store engineModelStore, model string) (modelFileList, error) {
	st, current, err := e.modelStoreFor(store.engine)
	if err != nil || current != store {
		return modelFileList{}, errModelInspectionChanged
	}
	return e.lmstudioSourceFilesBound(ctx, st, store, model, servingFileAccess{}, nil)
}

// LM Studio's downloaded row group is the native selector. Freeze its values,
// together with the launch and namespace, before enumerating the selected model.
func (e *Executor) lmstudioSourceFilesBound(ctx context.Context, st *engineState, store engineModelStore, model string, access servingFileAccess, expected *modelFileList) (modelFileList, error) {
	configuration := func() (string, string, error) {
		if err := ctx.Err(); err != nil {
			return "", "", err
		}
		current, currentStore, err := e.modelStoreFor(store.engine)
		if err != nil || current != st || currentStore != store {
			return "", "", errModelInspectionChanged
		}
		st.mu.Lock()
		run := st.runID
		data, err := json.Marshal(struct {
			CLI, Adapter, ModelsDir, Launch string
			Running                         bool
		}{store.cli, store.adapter, st.modelsDir, st.launchDigest, st.running})
		st.mu.Unlock()
		return run, string(data), err
	}
	run, config, err := configuration()
	if err != nil {
		return modelFileList{}, err
	}
	var old *sourceSnapshot
	if expected != nil {
		old = expected.source
		if old == nil || old.engine != store.engine || old.root != store.root || old.run != run || old.configuration != config || validateSourceSnapshot(ctx, old) != nil {
			return modelFileList{}, errModelInspectionChanged
		}
	}
	matched, recognition, err := e.lmstudioSourceRows(ctx, store, model)
	if err != nil {
		return modelFileList{}, err
	}
	if old != nil && !bytes.Equal(old.recognition, recognition) {
		return modelFileList{}, errModelInspectionChanged
	}
	checkConfiguration := func() error {
		currentRun, currentConfig, err := configuration()
		if err != nil || currentRun != run || currentConfig != config {
			return errModelInspectionChanged
		}
		return nil
	}
	if err := checkConfiguration(); err != nil {
		return modelFileList{}, err
	}
	if old != nil {
		if err := validateSourceSnapshot(ctx, old); err != nil {
			return modelFileList{}, err
		}
	}
	source := &sourceSnapshot{engine: store.engine, root: store.root, run: run, configuration: config, recognition: recognition}
	budget := access.scan
	if budget == nil {
		budget = &modelArtifactScan{}
	}
	scan := &llamaCacheScan{budget: budget, access: access, snapshot: source, expected: old}
	root, err := scan.directory(store.root)
	if err != nil {
		return modelFileList{}, err
	}
	list := modelFileList{source: source}
	added := map[string]bool{}
	add := func(abs string) error {
		rel, err := filepath.Rel(store.root, abs)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if !validModelRelPath(rel) {
			return fmt.Errorf("model file has an unsafe relative name")
		}
		key := copyPathKey(rel)
		if added[key] {
			for _, file := range list.Files {
				if file.Path == rel {
					return nil
				}
			}
			return errNoArtifactBinding
		}
		if len(list.Files) >= maxCopyArtifacts {
			return errModelArtifactLimit
		}
		file, err := nativeSourceArtifact(store.root, rel, "", old)
		if err != nil {
			return err
		}
		added[key] = true
		list.Files = append(list.Files, file)
		return nil
	}
	var walk func(string, int) error
	walk = func(dir string, depth int) error {
		if depth > 64 {
			return errModelArtifactLimit
		}
		resolved, err := resolveExistingPath(dir)
		if err != nil || !pathWithinRoot(root.target, resolved) {
			return fmt.Errorf("unsafe native model directory")
		}
		return scan.enumerate(ctx, dir, func(entry os.DirEntry) error {
			path := filepath.Join(dir, entry.Name())
			if entry.IsDir() {
				return walk(path, depth+1)
			}
			return add(path)
		})
	}
	for _, entry := range matched {
		if err := ctx.Err(); err != nil {
			return modelFileList{}, err
		}
		rel := normalizeSlashPath(entry.Path)
		abs := filepath.Join(store.root, filepath.FromSlash(rel))
		info, err := os.Stat(abs)
		if err != nil {
			return modelFileList{}, fmt.Errorf("LM Studio model is not on disk: %w", err)
		}
		if info.IsDir() {
			// WalkDir did not follow a selected directory symlink. Keep that
			// admission boundary while replacing its unchecked enumeration.
			entryInfo, err := os.Lstat(abs)
			if err != nil || !entryInfo.IsDir() || entryInfo.Mode()&os.ModeSymlink != 0 {
				return modelFileList{}, fmt.Errorf("unsupported native model directory link")
			}
			if err := walk(abs, 0); err != nil {
				return modelFileList{}, err
			}
			continue
		}
		if err := add(abs); err != nil {
			return modelFileList{}, err
		}
		// Enumerate the configured namespace, never the resolved file's parent.
		err = scan.enumerate(ctx, filepath.Dir(abs), func(s os.DirEntry) error {
			name := strings.ToLower(s.Name())
			if !s.IsDir() && strings.HasSuffix(name, ".gguf") && strings.Contains(name, "mmproj") {
				return add(filepath.Join(filepath.Dir(abs), s.Name()))
			}
			return nil
		})
		if err != nil {
			return modelFileList{}, err
		}
	}
	if err := finishNativeSource(ctx, &list, old); err != nil {
		return modelFileList{}, err
	}
	if err := checkConfiguration(); err != nil {
		return modelFileList{}, err
	}
	_, currentRows, err := e.lmstudioSourceRows(ctx, store, model)
	if err != nil {
		return modelFileList{}, err
	}
	if !bytes.Equal(recognition, currentRows) {
		return modelFileList{}, errModelInspectionChanged
	}
	if err := checkConfiguration(); err != nil {
		return modelFileList{}, err
	}
	if err := validateSourceSnapshot(ctx, source); err != nil {
		return modelFileList{}, err
	}
	return list, nil
}

func (e *Executor) lmstudioSourceRows(ctx context.Context, store engineModelStore, model string) ([]lmsListEntry, []byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	entries, err := lmsListModels(ctx, e, store.cli)
	if err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	var matched []lmsListEntry
	var group []lmsListEntry
	seen := map[string]lmsListEntry{}
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		if !lmsEntryMatchesModel(entry, model) {
			continue
		}
		p := normalizeSlashPath(entry.Path)
		if p == "" || !validModelRelPath(p) {
			return nil, nil, fmt.Errorf("LM Studio reports an unsafe model path")
		}
		if len(group) >= maxCopyArtifacts {
			return nil, nil, errModelArtifactLimit
		}
		group = append(group, entry)
		if previous, ok := seen[p]; ok {
			if previous != entry {
				return nil, nil, fmt.Errorf("LM Studio reports ambiguous model rows")
			}
			continue
		}
		seen[p] = entry
		matched = append(matched, entry)
	}
	if len(matched) == 0 {
		return nil, nil, fmt.Errorf("%w: LM Studio has no model %q on this node", errModelNotFound, model)
	}
	for _, entry := range matched[1:] {
		if entry.ModelKey == "" || entry.ModelKey != matched[0].ModelKey {
			return nil, nil, fmt.Errorf("model id matches distinct LM Studio models; name one by its model key")
		}
	}
	data, err := json.Marshal(group)
	return matched, data, err
}

func (e *Executor) lmstudioPlan(st *engineState, store engineModelStore, list modelFileList) (copyPlan, error) {
	var plan copyPlan
	var last []plannedFile
	for _, f := range list.Files {
		pf := plannedFile{modelFile: f, dest: filepath.Join(store.root, filepath.FromSlash(f.Path))}
		if ggufIsModel(f.Path) {
			last = append(last, pf)
			continue
		}
		plan.files = append(plan.files, pf)
	}
	plan.files = append(plan.files, last...)
	// LM Studio builds its model index at startup and offers no rescan, the same
	// reason its delete_model declares restart_after: restart a running server
	// so it lists the new model. A stopped one indexes it when next started.
	plan.finalize = func(ctx context.Context) error {
		return e.restartAfterAction(ctx, st, store.engine, "copy_model")
	}
	return plan, nil
}
