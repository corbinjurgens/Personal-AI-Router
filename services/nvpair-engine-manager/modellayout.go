// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// modellayout.go knows how each engine stores a model on disk, which model copy
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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// errModelCopyUnsupported marks an engine whose model layout this service does
// not know how to copy.
var errModelCopyUnsupported = errors.New("model copy is not supported for this engine")

// engineModelStore is one engine's on-disk model store, as model copy sees it.
type engineModelStore struct {
	engine string
	root   string // absolute
	cli    string // LM Studio's lms, resolved
}

// modelStoreFor resolves an engine's model store on this node.
func (e *Executor) modelStoreFor(engine string) (*engineState, engineModelStore, error) {
	st, err := e.state(engine)
	if err != nil {
		return nil, engineModelStore{}, err
	}
	store := engineModelStore{engine: engine}
	switch engine {
	case "ollama":
		store.root = engineEnvValue(st, "OLLAMA_MODELS")
		if store.root == "" && st.modelsDir != "" {
			store.root = filepath.Join(st.modelsDir, "models")
		}
	case "llamacpp":
		store.root = engineEnvValue(st, "LLAMA_CACHE")
		if store.root == "" {
			store.root = st.modelsDir
		}
	case "lmstudio":
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
	_, store, err := e.modelStoreFor(engine)
	if err != nil {
		return modelFileList{}, err
	}
	var list modelFileList
	switch engine {
	case "ollama":
		list, err = ollamaSourceFiles(store.root, model)
	case "llamacpp":
		list, err = llamacppSourceFiles(store.root, model)
	case "lmstudio":
		list, err = e.lmstudioSourceFiles(ctx, store, model)
	}
	if err != nil {
		return modelFileList{}, err
	}
	list.Engine, list.Model = engine, model
	return list, nil
}

// plannedFile is one file the destination must hold, and where.
type plannedFile struct {
	modelFile
	dest string
	// check, when set, runs on the verified download before it is renamed into
	// place (Ollama checks a manifest's blobs are all present first).
	check func(part string) error
}

// copyPlan is the destination's view of a copy: the files in the order they
// must be written (the one that makes the model visible last) and an
// engine-specific step once all are in place.
type copyPlan struct {
	root     string
	files    []plannedFile
	finalize func(ctx context.Context) error
}

// planModelCopy validates a source listing against this node's layout for the
// engine and maps each file to its destination.
func (e *Executor) planModelCopy(st *engineState, store engineModelStore, model string, list modelFileList) (copyPlan, error) {
	seen := make(map[string]bool, len(list.Files))
	for _, f := range list.Files {
		if !validModelRelPath(f.Path) {
			return copyPlan{}, fmt.Errorf("source listed an unsafe path %q", f.Path)
		}
		if seen[f.Path] {
			return copyPlan{}, fmt.Errorf("source listed %q twice", f.Path)
		}
		seen[f.Path] = true
		if !validSHA256(f.SHA256) {
			return copyPlan{}, fmt.Errorf("source listed %q without a valid sha256", f.Path)
		}
		if f.Size < 0 {
			return copyPlan{}, fmt.Errorf("source listed %q with a negative size", f.Path)
		}
	}
	if len(list.Files) == 0 {
		return copyPlan{}, fmt.Errorf("source listed no files for %q", model)
	}
	var (
		plan copyPlan
		err  error
	)
	switch store.engine {
	case "ollama":
		plan, err = ollamaPlan(store.root, model, list)
	case "llamacpp":
		plan, err = e.llamacppPlan(st, store.root, model, list)
	case "lmstudio":
		plan, err = e.lmstudioPlan(st, store.root, list)
	default:
		return copyPlan{}, fmt.Errorf("%w: %q", errModelCopyUnsupported, store.engine)
	}
	if err != nil {
		return copyPlan{}, err
	}
	for _, f := range plan.files {
		if !pathWithinRoot(store.root, f.dest) || filepath.Clean(f.dest) == store.root {
			return copyPlan{}, fmt.Errorf("%q would be written outside the model store", f.Path)
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
	rel, err := ollamaManifestRel(model)
	if err != nil {
		return modelFileList{}, fmt.Errorf("%w: %v", errModelNotFound, err)
	}
	manifestPath, info, err := confinedRegularFile(root, filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return modelFileList{}, fmt.Errorf("%w: Ollama has no model %q on this node", errModelNotFound, model)
		}
		return modelFileList{}, err
	}
	data, err := readSmallFile(manifestPath, ollamaManifestLimit)
	if err != nil {
		return modelFileList{}, err
	}
	m, err := parseOllamaManifest(data)
	if err != nil {
		return modelFileList{}, err
	}
	blobs, err := m.blobs()
	if err != nil {
		return modelFileList{}, err
	}
	var list modelFileList
	for _, b := range blobs {
		brel := "blobs/sha256-" + b.Digest
		real, binfo, err := confinedRegularFile(root, filepath.Join(root, filepath.FromSlash(brel)))
		if err != nil {
			return modelFileList{}, fmt.Errorf("model %q is incomplete on this node (%s): %w", model, brel, err)
		}
		if b.Size > 0 && binfo.Size() != b.Size {
			return modelFileList{}, fmt.Errorf("model %q is incomplete on this node: %s is %d bytes, manifest says %d", model, brel, binfo.Size(), b.Size)
		}
		// The blob's name is its sha256; the destination verifies it.
		list.Files = append(list.Files, modelFile{Path: brel, Size: binfo.Size(), SHA256: b.Digest, abs: real})
	}
	list.Files = append(list.Files, modelFile{Path: rel, Size: info.Size(), abs: manifestPath})
	return list, nil
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
	refs := filepath.Join(repoDir, "refs")
	read := func(name string) string {
		data, err := readSmallFile(filepath.Join(refs, name), 4096)
		if err != nil {
			return ""
		}
		line, _, _ := strings.Cut(string(data), "\n")
		line = strings.TrimSpace(line)
		if !commitRe.MatchString(line) {
			return ""
		}
		return line
	}
	if c := read("main"); c != "" {
		return c
	}
	entries, err := os.ReadDir(refs)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		if entry.Type().IsRegular() {
			if c := read(entry.Name()); c != "" {
				return c
			}
		}
	}
	return ""
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
	repo, tag, err := splitLlamaModel(model)
	if err != nil {
		return modelFileList{}, fmt.Errorf("%w: %v", errModelNotFound, err)
	}
	repoDir := llamaRepoDir(root, repo)
	commit := llamaCachedRef(repoDir)
	if commit == "" {
		return modelFileList{}, fmt.Errorf("%w: llama.cpp has no cached %q on this node", errModelNotFound, repo)
	}
	snap := filepath.Join(repoDir, "snapshots", commit)
	var rels []string
	err = filepath.WalkDir(snap, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || (!d.Type().IsRegular() && d.Type()&fs.ModeSymlink == 0) {
			return nil
		}
		rel, err := filepath.Rel(snap, path)
		if err != nil {
			return err
		}
		rels = append(rels, filepath.ToSlash(rel))
		return nil
	})
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
	blobs, _ := filepath.EvalSymlinks(filepath.Join(repoDir, "blobs"))
	list := modelFileList{Revision: commit}
	for _, rel := range selected {
		if !validModelRelPath(rel) {
			return modelFileList{}, fmt.Errorf("cached file %q has a name this service cannot copy safely", rel)
		}
		real, info, err := confinedRegularFile(repoDir, filepath.Join(snap, filepath.FromSlash(rel)))
		if err != nil {
			return modelFileList{}, err
		}
		f := modelFile{Path: rel, Size: info.Size(), abs: real}
		// A snapshot entry that links to an LFS blob is named by its sha256.
		if blobs != "" && filepath.Dir(real) == blobs && validSHA256(filepath.Base(real)) {
			f.SHA256 = filepath.Base(real)
		}
		list.Files = append(list.Files, f)
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
	st.mu.Lock()
	running, port := st.running, st.port
	st.mu.Unlock()
	if !running || port <= 0 {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/models?reload=1", port), nil)
	if err != nil {
		return
	}
	resp, err := e.client.Do(req)
	if err != nil {
		slog.Warn("llama.cpp model rescan failed", "err", err)
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		slog.Warn("llama.cpp model rescan failed", "status", resp.StatusCode)
	}
}

// writeFileAtomic writes data to a temporary file beside path and renames it
// into place.
func writeFileAtomic(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
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
	entries, err := lmsListModels(ctx, e, store.cli)
	if err != nil {
		return modelFileList{}, err
	}
	var matched []lmsListEntry
	seen := map[string]bool{}
	for _, entry := range entries {
		p := normalizeSlashPath(entry.Path)
		if p == "" || seen[p] || !lmsEntryMatchesModel(entry, model) {
			continue
		}
		seen[p] = true
		matched = append(matched, entry)
	}
	if len(matched) == 0 {
		return modelFileList{}, fmt.Errorf("%w: LM Studio has no model %q on this node", errModelNotFound, model)
	}
	for _, entry := range matched[1:] {
		if entry.ModelKey == "" || entry.ModelKey != matched[0].ModelKey {
			return modelFileList{}, fmt.Errorf("model id %q matches %d LM Studio models; name one by its model key", model, len(matched))
		}
	}
	var list modelFileList
	added := map[string]bool{}
	add := func(abs string) error {
		real, info, err := confinedRegularFile(store.root, abs)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(store.root, abs)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if !validModelRelPath(rel) {
			return fmt.Errorf("model file %q has a name this service cannot copy safely", rel)
		}
		if added[rel] {
			return nil
		}
		added[rel] = true
		list.Files = append(list.Files, modelFile{Path: rel, Size: info.Size(), abs: real})
		return nil
	}
	for _, entry := range matched {
		rel := normalizeSlashPath(entry.Path)
		if !validModelRelPath(rel) {
			return modelFileList{}, fmt.Errorf("LM Studio reports an unsafe path %q for %q", entry.Path, model)
		}
		abs := filepath.Join(store.root, filepath.FromSlash(rel))
		info, err := os.Stat(abs)
		if err != nil {
			return modelFileList{}, fmt.Errorf("LM Studio lists %q but it is not on disk: %w", rel, err)
		}
		if info.IsDir() {
			// An MLX model is a directory of weights and configuration.
			err = filepath.WalkDir(abs, func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if d.IsDir() {
					return nil
				}
				return add(path)
			})
			if err != nil {
				return modelFileList{}, err
			}
			continue
		}
		if err := add(abs); err != nil {
			return modelFileList{}, err
		}
		// A vision GGUF's projector sits beside it and LM Studio pairs them.
		siblings, _ := os.ReadDir(filepath.Dir(abs))
		for _, s := range siblings {
			name := s.Name()
			if !s.IsDir() && strings.HasSuffix(strings.ToLower(name), ".gguf") && strings.Contains(strings.ToLower(name), "mmproj") {
				if err := add(filepath.Join(filepath.Dir(abs), name)); err != nil {
					return modelFileList{}, err
				}
			}
		}
	}
	return list, nil
}

func (e *Executor) lmstudioPlan(st *engineState, root string, list modelFileList) (copyPlan, error) {
	var plan copyPlan
	var last []plannedFile
	for _, f := range list.Files {
		pf := plannedFile{modelFile: f, dest: filepath.Join(root, filepath.FromSlash(f.Path))}
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
		return e.restartAfterAction(ctx, st, "lmstudio", "copy_model")
	}
	return plan, nil
}
