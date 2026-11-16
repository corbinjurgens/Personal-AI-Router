// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// modelsources_linked.go keeps the GGUF files an operator links by hand, the
// third model source beside Ollama's and LM Studio's folders. A link is only a
// record of the file's path: PAIR serves the file where it is and never writes,
// moves or deletes it. Unlinking forgets the record.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"nvpair-shared/durablefile"
)

const (
	linkedModelsFile    = "linked-models.json"
	linkedModelsVersion = 1
	linkedSourceName    = "local"
	maxLinkedPath       = 4096
)

// linkedModel is one linked file, kept per engine.
type linkedModel struct {
	Engine string `json:"engine"`
	ID     string `json:"id"`
	Path   string `json:"path"`
	MMProj string `json:"mmproj,omitempty"`
}

// linkedModelStore is the persisted list of linked files, loaded on first use.
type linkedModelStore struct {
	mu      sync.Mutex
	loaded  bool
	entries []linkedModel
}

func (e *Executor) linkedModelsPath() string {
	return filepath.Join(e.baseDir, linkedModelsFile)
}

// linkedEntries loads the store if needed. A malformed file is reported rather
// than replaced, so a hand edit gone wrong never loses every link. The caller
// holds e.linked.mu.
func (e *Executor) linkedEntries() ([]linkedModel, error) {
	if e.linked.loaded {
		return e.linked.entries, nil
	}
	data, err := durablefile.ReadFile(e.linkedModelsPath())
	if errors.Is(err, os.ErrNotExist) {
		e.linked.loaded = true
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read linked models: %w", err)
	}
	var file struct {
		Version int           `json:"version"`
		Entries []linkedModel `json:"entries"`
	}
	if err := json.Unmarshal(data, &file); err != nil || file.Version != linkedModelsVersion {
		return nil, fmt.Errorf("%s is not a valid linked-model list; fix or remove it", linkedModelsFile)
	}
	e.linked.entries, e.linked.loaded = file.Entries, true
	return e.linked.entries, nil
}

func (e *Executor) saveLinkedEntries(entries []linkedModel) error {
	if entries == nil {
		entries = []linkedModel{}
	}
	data, err := json.MarshalIndent(struct {
		Version int           `json:"version"`
		Entries []linkedModel `json:"entries"`
	}{linkedModelsVersion, entries}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(e.baseDir, 0o700); err != nil {
		return fmt.Errorf("save linked models: %w", err)
	}
	if err := durablefile.WriteFile(e.linkedModelsPath(), data, 0o600); err != nil {
		return fmt.Errorf("save linked models: %w", err)
	}
	e.linked.entries = entries
	return nil
}

// checkLinkableFile returns the cleaned path of a GGUF model file PAIR may serve
// in place or import: an absolute .gguf path to a regular file with the GGUF
// magic that is not a projector, helper or later shard, and whose path a
// preset line can carry.
func checkLinkableFile(file string) (string, error) {
	if len(file) > maxLinkedPath || !filepath.IsAbs(file) {
		return "", modelSettingsParamsError(errors.New("Choose a file by its full path."))
	}
	file = filepath.Clean(file)
	lower := strings.ToLower(filepath.Base(file))
	if !strings.HasSuffix(lower, ".gguf") {
		return "", modelSettingsParamsError(errors.New("Only GGUF model files (.gguf) can be added."))
	}
	if !ggufIsModel(lower) {
		return "", modelSettingsParamsError(errors.New("This file is a projector or helper file, not a model."))
	}
	if strings.ContainsAny(file, sourcePresetUnsafe) {
		return "", modelSettingsParamsError(errors.New("The file's path contains characters llama.cpp cannot read from a preset (# ; $ % { })."))
	}
	if split, ok := ggufSplitInfo(lower); ok && split.index != 1 {
		return "", modelSettingsParamsError(errors.New("This is one part of a split model; choose its first part (-00001-of-...)."))
	}
	if info, err := os.Stat(file); err != nil || !info.Mode().IsRegular() {
		return "", modelSettingsParamsError(errors.New("The file does not exist or is not a regular file."))
	}
	if !isGGUFFile(file) {
		return "", modelSettingsParamsError(errors.New("The file is not a GGUF model."))
	}
	return file, nil
}

// linkModel records file as a model engine can serve in place and returns its
// ID, "local/<file stem>". Linking a path already linked returns its ID.
func (e *Executor) linkModel(engine, file string) (string, error) {
	file, err := checkLinkableFile(file)
	if err != nil {
		return "", err
	}
	base := filepath.Base(file)
	stem := ggufFirstShard.ReplaceAllString(base[:len(base)-len(".gguf")], "")

	e.linked.mu.Lock()
	defer e.linked.mu.Unlock()
	entries, err := e.linkedEntries()
	if err != nil {
		return "", err
	}
	taken := map[string]bool{}
	for _, entry := range entries {
		if entry.Engine == engine && entry.Path == file {
			return entry.ID, nil
		}
		if entry.Engine == engine {
			taken[entry.ID] = true
		}
	}
	id := linkedSourceName + "/" + stem
	for n := 2; taken[id]; n++ {
		id = linkedSourceName + "/" + stem + "-" + strconv.Itoa(n)
	}
	if validateLlamaModelID(id) != nil {
		return "", modelSettingsParamsError(errors.New("The file name cannot be used as a model name; rename the file and try again."))
	}
	entry := linkedModel{Engine: engine, ID: id, Path: file, MMProj: folderProjector(filepath.Dir(file))}
	if err := e.saveLinkedEntries(append(append([]linkedModel(nil), entries...), entry)); err != nil {
		return "", err
	}
	return id, nil
}

// folderProjector is the folder's one projector GGUF, or "" when there is none
// or more than one to choose from.
func folderProjector(dir string) string {
	files, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	found := ""
	for _, f := range files {
		name := strings.ToLower(f.Name())
		if f.IsDir() || !strings.HasSuffix(name, ".gguf") || !strings.Contains(name, "mmproj") {
			continue
		}
		if found != "" {
			return ""
		}
		found = filepath.Join(dir, f.Name())
	}
	if found == "" || strings.ContainsAny(found, sourcePresetUnsafe) || !isGGUFFile(found) {
		return ""
	}
	return found
}

// unlinkModel forgets a linked file. The file itself is untouched.
func (e *Executor) unlinkModel(engine, id string) error {
	e.linked.mu.Lock()
	defer e.linked.mu.Unlock()
	entries, err := e.linkedEntries()
	if err != nil {
		return err
	}
	kept := make([]linkedModel, 0, len(entries))
	for _, entry := range entries {
		if entry.Engine != engine || entry.ID != id {
			kept = append(kept, entry)
		}
	}
	if len(kept) == len(entries) {
		return modelSettingsParamsError(fmt.Errorf("%s is not a linked file", id))
	}
	return e.saveLinkedEntries(kept)
}

// sourceModelsFor lists every model engine can serve from outside its store:
// the default sources' models, then linked files that still exist. The first
// holder of an ID keeps it. An unreadable link list contributes nothing here;
// the listing method reports it.
func (e *Executor) sourceModelsFor(ctx context.Context, engine string) []sourceModel {
	models := scanModelSources(ctx, e.modelSources)
	seen := make(map[string]bool, len(models))
	for _, model := range models {
		seen[model.id] = true
	}
	e.linked.mu.Lock()
	entries, _ := e.linkedEntries()
	entries = append([]linkedModel(nil), entries...)
	e.linked.mu.Unlock()
	for _, entry := range entries {
		if entry.Engine != engine || seen[entry.ID] {
			continue
		}
		if info, err := os.Stat(entry.Path); err != nil || !info.Mode().IsRegular() {
			continue
		}
		seen[entry.ID] = true
		models = append(models, sourceModel{id: entry.ID, format: sourceFormatGGUF, path: entry.Path, mmproj: entry.MMProj})
	}
	return models
}

// sourceModelShards is every file of a source model: its path, plus the other
// shards beside it when it is the first of a split GGUF.
func sourceModelShards(model sourceModel) []string {
	dir, base := filepath.Split(model.path)
	split, ok := ggufSplitInfo(base)
	if !ok || split.count <= 1 {
		return []string{model.path}
	}
	shards := make([]string, split.count)
	for i := range shards {
		shards[i] = filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", split.prefix, i+1, split.count))
	}
	shards[0] = model.path
	return shards
}

// isSourceModelID reports an ID only a model source can hold, so resolution
// scans sources for nothing else.
func isSourceModelID(id string) bool {
	for _, prefix := range []string{"ollama/", "lmstudio/", linkedSourceName + "/"} {
		if strings.HasPrefix(id, prefix) {
			return true
		}
	}
	return false
}

// sourceModelsHook gives the artifact resolver this executor's source list for
// engine, or nil without an executor.
func (e *Executor) sourceModelsHook(engine string) func(context.Context) []sourceModel {
	if e == nil {
		return nil
	}
	return func(ctx context.Context) []sourceModel { return e.sourceModelsFor(ctx, engine) }
}
