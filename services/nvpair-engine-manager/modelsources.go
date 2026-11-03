// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// modelsources.go finds models that other applications keep on this node, so
// an engine can load them where they are. A model source is a directory with a
// known layout. The built-in sources are Ollama's and LM Studio's model
// folders; neither application needs to be installed or running, and their
// files are only read, never moved or changed.
//
// Sources are engine-neutral: each model found carries its file format, and an
// engine family that loads that format offers it. llama.cpp offers GGUF source
// models through a generated native preset (llamaSourcePreset). Peer copy,
// model settings inspection and serving presets cover only an engine's own
// store, not source models.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
)

const (
	// sourceLayoutOllama is Ollama's store: one manifest per model at
	// manifests/<host>/<namespace>/<name>/<tag>, naming content-addressed
	// blobs/sha256-<hex>.
	sourceLayoutOllama = "ollama"
	// sourceLayoutGGUFTree is a folder of GGUF files, nested at most
	// sourceTreeDepth directories deep, such as LM Studio's
	// <publisher>/<repository>/<file>.gguf.
	sourceLayoutGGUFTree = "gguf-tree"

	sourceFormatGGUF = "gguf"

	sourceTreeDepth     = 4
	sourceScanEntries   = 100000
	sourceManifestLimit = 1 << 20

	ollamaModelLayer     = "application/vnd.ollama.image.model"
	ollamaProjectorLayer = "application/vnd.ollama.image.projector"
	ollamaAdapterLayer   = "application/vnd.ollama.image.adapter"
	ollamaDefaultHost    = "registry.ollama.ai"
	ollamaDefaultSpace   = "library"
)

// modelSource is one directory to search. name prefixes the IDs of the models
// it holds, so a source model never shares an ID with an engine's own model.
type modelSource struct {
	name   string
	layout string
	root   string
}

// sourceModel is one model a source holds.
type sourceModel struct {
	id     string
	format string
	path   string // the model file; the first shard of a split GGUF
	mmproj string // the multimodal projector the source pairs with it, if any
}

// defaultModelSources are the folders Ollama and LM Studio keep models in by
// default. OLLAMA_MODELS replaces Ollama's default, as it does for Ollama.
func defaultModelSources() []modelSource {
	home, _ := os.UserHomeDir()
	var sources []modelSource
	add := func(name, layout, root string) {
		if root == "" || !filepath.IsAbs(root) {
			return
		}
		sources = append(sources, modelSource{name: name, layout: layout, root: filepath.Clean(root)})
	}
	if dir := os.Getenv("OLLAMA_MODELS"); dir != "" {
		add("ollama", sourceLayoutOllama, expandPath(dir))
	} else {
		if home != "" {
			add("ollama", sourceLayoutOllama, filepath.Join(home, ".ollama", "models"))
		}
		if runtime.GOOS == "linux" {
			// The Linux installer runs Ollama as a service with its own home.
			add("ollama", sourceLayoutOllama, "/usr/share/ollama/.ollama/models")
		}
	}
	if home != "" {
		add("lmstudio", sourceLayoutGGUFTree, filepath.Join(home, ".lmstudio", "models"))
		add("lmstudio", sourceLayoutGGUFTree, filepath.Join(home, ".cache", "lm-studio", "models"))
	}
	return sources
}

// scanModelSources lists the models in sources, sorted by ID. A missing or
// unreadable source holds no models; the first source to hold an ID keeps it.
func scanModelSources(ctx context.Context, sources []modelSource) []sourceModel {
	var models []sourceModel
	seen := map[string]bool{}
	for _, source := range sources {
		var found []sourceModel
		switch source.layout {
		case sourceLayoutOllama:
			found = scanOllamaSource(ctx, source)
		case sourceLayoutGGUFTree:
			found = scanGGUFTreeSource(ctx, source)
		}
		for _, model := range found {
			if !seen[model.id] {
				seen[model.id] = true
				models = append(models, model)
			}
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].id < models[j].id })
	return models
}

var ollamaSourceSegment = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_.-]*$`)

// scanOllamaSource offers each manifest whose model layer is a GGUF blob as
// "ollama/<name>@<tag>", where name is Ollama's own short form: library
// models drop the default host and namespace, other namespaces keep theirs.
func scanOllamaSource(ctx context.Context, source modelSource) []sourceModel {
	var models []sourceModel
	walkSourceFiles(ctx, filepath.Join(source.root, "manifests"), 3, func(file string, rel []string) {
		if len(rel) != 4 {
			return
		}
		for _, segment := range rel {
			if !ollamaSourceSegment.MatchString(segment) {
				return
			}
		}
		data, err := readSmallFile(file, sourceManifestLimit)
		if err != nil {
			return
		}
		var manifest struct {
			Layers []struct {
				MediaType string `json:"mediaType"`
				Digest    string `json:"digest"`
			} `json:"layers"`
		}
		if json.Unmarshal(data, &manifest) != nil {
			return
		}
		var model sourceModel
		for _, layer := range manifest.Layers {
			if layer.MediaType == ollamaAdapterLayer {
				// Served without its adapter, the model would answer differently.
				return
			}
			hex, ok := strings.CutPrefix(layer.Digest, "sha256:")
			if !ok || !validSHA256(hex) {
				continue
			}
			blob := filepath.Join(source.root, "blobs", "sha256-"+hex)
			switch layer.MediaType {
			case ollamaModelLayer:
				model.path = blob
			case ollamaProjectorLayer:
				model.mmproj = blob
			}
		}
		if model.path == "" || !isGGUFFile(model.path) {
			return
		}
		if model.mmproj != "" && !isGGUFFile(model.mmproj) {
			model.mmproj = ""
		}
		name := rel[2]
		switch {
		case rel[0] != ollamaDefaultHost:
			name = rel[0] + "/" + rel[1] + "/" + rel[2]
		case rel[1] != ollamaDefaultSpace:
			name = rel[1] + "/" + rel[2]
		}
		model.id = source.name + "/" + name + "@" + rel[3]
		model.format = sourceFormatGGUF
		models = append(models, model)
	})
	return models
}

var ggufFirstShard = regexp.MustCompile(`(?i)-00001-of-[0-9]{5}$`)

// scanGGUFTreeSource offers each GGUF model file as "<source>/<path>", its path
// relative to the root without the .gguf suffix or a split's shard suffix.
// Only a split's first shard is offered; llama.cpp finds the rest beside it. A
// folder's single projector is paired with every model in that folder.
func scanGGUFTreeSource(ctx context.Context, source modelSource) []sourceModel {
	type folder struct{ models, projectors []string }
	folders := map[string]*folder{}
	walkSourceFiles(ctx, source.root, sourceTreeDepth, func(file string, rel []string) {
		base := strings.ToLower(rel[len(rel)-1])
		if !strings.HasSuffix(base, ".gguf") {
			return
		}
		dir := path.Join(rel[:len(rel)-1]...)
		f := folders[dir]
		if f == nil {
			f = &folder{}
			folders[dir] = f
		}
		if strings.Contains(base, "mmproj") {
			f.projectors = append(f.projectors, file)
			return
		}
		if split, ok := ggufSplitInfo(base); !ok || split.index != 1 || !ggufIsModel(base) {
			return
		}
		f.models = append(f.models, file)
	})
	var models []sourceModel
	for dir, f := range folders {
		mmproj := ""
		if len(f.projectors) == 1 && isGGUFFile(f.projectors[0]) {
			mmproj = f.projectors[0]
		}
		for _, file := range f.models {
			if !isGGUFFile(file) {
				continue
			}
			stem := filepath.Base(file)
			stem = stem[:len(stem)-len(".gguf")]
			stem = ggufFirstShard.ReplaceAllString(stem, "")
			models = append(models, sourceModel{
				id:     source.name + "/" + path.Join(dir, stem),
				format: sourceFormatGGUF,
				path:   file,
				mmproj: mmproj,
			})
		}
	}
	return models
}

// walkSourceFiles calls visit for every file (or link to one) at most depth
// directories below root, with its path relative to root as segments. It skips
// hidden entries, does not follow directory links, and visits at most
// sourceScanEntries directory entries.
func walkSourceFiles(ctx context.Context, root string, depth int, visit func(file string, rel []string)) {
	budget := sourceScanEntries
	var walk func(dir string, rel []string)
	walk = func(dir string, rel []string) {
		entries, err := os.ReadDir(dir)
		if err != nil {
			if !errors.Is(err, os.ErrNotExist) && len(rel) == 0 {
				slog.Warn("model source is unreadable", "root", root)
			}
			return
		}
		for _, entry := range entries {
			budget--
			if budget < 0 || ctx.Err() != nil {
				return
			}
			name := entry.Name()
			if strings.HasPrefix(name, ".") {
				continue
			}
			file := filepath.Join(dir, name)
			next := append(rel[:len(rel):len(rel)], name)
			if entry.IsDir() {
				if len(rel) < depth {
					walk(file, next)
				}
				continue
			}
			if info, err := os.Stat(file); err == nil && info.Mode().IsRegular() {
				visit(file, next)
			}
		}
	}
	walk(root, nil)
}

// isGGUFFile reports whether file starts with the GGUF magic.
func isGGUFFile(file string) bool {
	f, err := os.Open(file)
	if err != nil {
		return false
	}
	defer f.Close()
	magic := make([]byte, 4)
	_, err = io.ReadFull(f, magic)
	return err == nil && bytes.Equal(magic, []byte("GGUF"))
}

// sourcePresetUnsafe are characters a native preset line cannot carry
// literally: comment markers, substitution syntax and line breaks.
const sourcePresetUnsafe = "#;$%{}\r\n\x00"

// llamaSourcePreset writes the native preset that offers this node's GGUF source
// models to a llama.cpp router and returns its path. It returns "" when there
// is nothing to offer, or when the engine's own preset already decides its
// models: a serving preset, typed loads, or a --models-preset launch option.
func (e *Executor) llamaSourcePreset(ctx context.Context, st *engineState, engine string, rt Runtime, launch launchCommand) string {
	if len(e.modelSources) == 0 || st.manifest.FamilyOrDefault() != "llamacpp" || rt.Serving != nil || len(rt.ModelLoads) != 0 || launchNamesModelsPreset(launch) {
		return ""
	}
	var text strings.Builder
	text.WriteString("# Written by PAIR at each start: models found in this node's model sources.\n")
	offered := 0
	for _, model := range scanModelSources(ctx, e.modelSources) {
		if model.format != sourceFormatGGUF || validateLlamaModelID(model.id) != nil ||
			strings.ContainsAny(model.path+model.mmproj, sourcePresetUnsafe) {
			continue
		}
		text.WriteString("\n[" + model.id + "]\nmodel = " + model.path + "\n")
		if model.mmproj != "" {
			text.WriteString("mmproj = " + model.mmproj + "\n")
		}
		offered++
	}
	file := filepath.Join(e.baseDir, "model-sources", engine+".ini")
	if offered == 0 {
		_ = os.Remove(file)
		return ""
	}
	if err := writeFileAtomic(file, []byte(text.String())); err != nil {
		slog.Warn("model source preset not written; source models are not offered", "engine", engine, "err", err)
		return ""
	}
	return file
}

// launchNamesModelsPreset reports whether launch already selects a native
// preset; llama-server reads only one.
func launchNamesModelsPreset(launch launchCommand) bool {
	for _, arg := range launch.Args {
		name, _, _ := strings.Cut(arg, "=")
		if llamaLongOption(name) == "--models-preset" {
			return true
		}
	}
	for key := range launch.Env {
		if environmentKey(key) == environmentKey("LLAMA_ARG_MODELS_PRESET") {
			return true
		}
	}
	return false
}
