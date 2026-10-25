// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sort"
	"unicode/utf8"

	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
	"nvpair-shared/nodepolicy"
)

const modelDownloadFile = "model-download-state.json"

var errDownloadPersistence = errors.New("model download persistence unavailable")

type downloadContentKey struct {
	engine string
	model  string
}

type modelDownloadMarker struct {
	ID         string `json:"id"`
	Engine     string `json:"engine"`
	Model      string `json:"model"`
	ContentKey string `json:"contentKey"`
	Naming     string `json:"naming"`
}

func downloadKey(engine, model, naming string) (downloadContentKey, error) {
	if len(engine) > engines.MaxDynamicIDLen || !engines.ValidID(engine) {
		return downloadContentKey{}, errors.New("invalid model download engine")
	}
	if err := nodepolicy.ValidateModelIdentifier(model); err != nil {
		return downloadContentKey{}, err
	}
	if naming != engines.ModelNamingExact && naming != engines.ModelNamingImpliedLatestTag {
		return downloadContentKey{}, errors.New("unsupported model naming")
	}
	key := engines.NormalizeModelName(model, naming)
	if err := nodepolicy.ValidateModelIdentifier(key); err != nil {
		return downloadContentKey{}, err
	}
	return downloadContentKey{engine: engine, model: key}, nil
}

type modelDownloadJournal struct {
	Version int                   `json:"version"`
	Entries []modelDownloadMarker `json:"entries"`
}

// The owner serializes capture, publication and adoption. The store performs
// only bounded private-file I/O and does not hold the admission gate.
type modelDownloadStore struct {
	path    string
	replace durablefile.Replacer
}

func newModelDownloadStore(baseDir string) modelDownloadStore {
	if baseDir == "" {
		return modelDownloadStore{}
	}
	return modelDownloadStore{path: filepath.Join(baseDir, modelDownloadFile)}
}

func (s modelDownloadStore) load() (map[downloadContentKey]modelDownloadMarker, error) {
	markers := make(map[downloadContentKey]modelDownloadMarker)
	if s.path == "" {
		return nil, errDownloadPersistence
	}
	f, err := os.Open(s.path)
	if errors.Is(err, os.ErrNotExist) {
		// A missing journal is safe only in an existing usable directory.
		info, dirErr := os.Stat(filepath.Dir(s.path))
		if dirErr == nil && info.IsDir() {
			return markers, nil
		}
		return nil, errDownloadPersistence
	}
	if err != nil {
		return nil, errDownloadPersistence
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, modeldownload.MaxJournalBytes+1))
	if err != nil || len(data) > modeldownload.MaxJournalBytes || !utf8.Valid(data) {
		return nil, errDownloadPersistence
	}
	var journal modelDownloadJournal
	if json.Unmarshal(data, &journal) != nil || journal.Version != 1 || journal.Entries == nil || len(journal.Entries) > modeldownload.MaxUnsafeEntries {
		return nil, errDownloadPersistence
	}
	ids := make(map[string]bool)
	for _, marker := range journal.Entries {
		key, err := downloadKey(marker.Engine, marker.Model, marker.Naming)
		if err != nil || key.model != marker.ContentKey || ids[marker.ID] {
			return nil, errDownloadPersistence
		}
		// Reuse the wire validator, including engine and operation-ID syntax.
		probe := modeldownload.Snapshot{Version: 1, SourceID: marker.ID, Revision: "0", Entries: []modeldownload.Entry{{ID: marker.ID, Engine: marker.Engine, Model: marker.Model, Phase: "unknown", BlocksReadiness: true}}}
		if probe.Validate() != nil {
			return nil, errDownloadPersistence
		}
		if _, duplicate := markers[key]; duplicate {
			return nil, errDownloadPersistence
		}
		markers[key] = marker
		ids[marker.ID] = true
	}
	return markers, nil
}

func (s modelDownloadStore) save(markers map[downloadContentKey]modelDownloadMarker) (landed, uncertain bool, err error) {
	if s.path == "" || len(markers) > modeldownload.MaxUnsafeEntries {
		return false, false, errDownloadPersistence
	}
	journal := modelDownloadJournal{Version: 1, Entries: make([]modelDownloadMarker, 0, len(markers))}
	for _, marker := range markers {
		journal.Entries = append(journal.Entries, marker)
	}
	sort.Slice(journal.Entries, func(i, j int) bool {
		a, b := journal.Entries[i], journal.Entries[j]
		if a.Engine != b.Engine {
			return a.Engine < b.Engine
		}
		return a.ContentKey < b.ContentKey
	})
	data, err := json.Marshal(journal)
	if err != nil || len(data) > modeldownload.MaxJournalBytes {
		return false, false, errDownloadPersistence
	}
	err = s.replace.WriteFile(s.path, data, 0600)
	if err == nil {
		return true, false, nil
	}
	if errors.Is(err, durablefile.ErrReplaced) {
		return true, true, errDownloadPersistence
	}
	return false, false, errDownloadPersistence
}
