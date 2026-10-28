// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
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
	Version              int                   `json:"version"`
	Entries              []modelDownloadMarker `json:"entries"`
	Copies               []modelCopyMarker     `json:"copies"`
	CopiesComplete       bool                  `json:"copiesComplete"`
	CopyRecoveryRequired bool                  `json:"copyRecoveryRequired"`
}

// Only unresolved provenance belongs here, never transport or execution state.
type modelCopyMarker struct {
	OpID               string `json:"opId"`
	Engine             string `json:"engine"`
	Model              string `json:"model"`
	ContentKey         string `json:"contentKey"`
	Naming             string `json:"naming"`
	ExpectedExportID   string `json:"expectedExportId"`
	Role               string `json:"role"`
	OwnerNodeID        string `json:"ownerNodeId"`
	SourceNodeID       string `json:"sourceNodeId"`
	DestinationNodeID  string `json:"destinationNodeId"`
	ManagerIncarnation string `json:"managerIncarnation"`
	Phase              string `json:"phase"`
}

func (m *modelCopyMarker) UnmarshalJSON(data []byte) error {
	if err := downloadJournalObject(data, "opId", "engine", "model", "contentKey", "naming", "expectedExportId", "role", "ownerNodeId", "sourceNodeId", "destinationNodeId", "managerIncarnation", "phase"); err != nil {
		return err
	}
	type plain modelCopyMarker
	return decodeCopyObject(data, (*plain)(m), "opId", "engine", "model", "contentKey", "naming", "expectedExportId", "role", "ownerNodeId", "sourceNodeId", "destinationNodeId", "managerIncarnation", "phase")
}

func (m modelCopyMarker) validate() error {
	key, err := downloadKey(m.Engine, m.Model, m.Naming)
	if err != nil || key.model != m.ContentKey || !lowerHex(m.OpID, 32) || !lowerHex(m.ManagerIncarnation, 32) ||
		!copySelector(m.OwnerNodeID) || !copySelector(m.SourceNodeID) || !copySelector(m.DestinationNodeID) || m.SourceNodeID == m.DestinationNodeID ||
		m.ExpectedExportID != "" && !lowerHex(m.ExpectedExportID, 64) || m.Phase != "unsafe" && m.Phase != "uncertain" {
		return errDownloadPersistence
	}
	if m.Role != "controller" && m.Role != "destination" || (m.Role == "destination") != (m.OwnerNodeID == m.DestinationNodeID) {
		return errDownloadPersistence
	}
	return nil
}

// Journal recovery must not choose between duplicate or case-variant fields.
// Require the exact private shape before encoding/json's permissive decoding.
func downloadJournalObject(data []byte, required ...string) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return errDownloadPersistence
	}
	seen := make(map[string]bool, len(required))
	for decoder.More() {
		token, err := decoder.Token()
		name, ok := token.(string)
		if err != nil || !ok || seen[name] || !slices.Contains(required, name) {
			return errDownloadPersistence
		}
		var value json.RawMessage
		if decoder.Decode(&value) != nil || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return errDownloadPersistence
		}
		seen[name] = true
	}
	if _, err := decoder.Token(); err != nil || len(seen) != len(required) {
		return errDownloadPersistence
	}
	if _, err := decoder.Token(); err != io.EOF {
		return errDownloadPersistence
	}
	return nil
}

func (j *modelDownloadJournal) UnmarshalJSON(data []byte) error {
	if err := downloadJournalObject(data, "version", "entries", "copies", "copiesComplete", "copyRecoveryRequired"); err != nil {
		return err
	}
	type plain modelDownloadJournal
	return json.Unmarshal(data, (*plain)(j))
}

func (m *modelDownloadMarker) UnmarshalJSON(data []byte) error {
	if err := downloadJournalObject(data, "id", "engine", "model", "contentKey", "naming"); err != nil {
		return err
	}
	type plain modelDownloadMarker
	return json.Unmarshal(data, (*plain)(m))
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

// Download-only callers still validate the entire current content document.
func (s modelDownloadStore) load() (map[downloadContentKey]modelDownloadMarker, error) {
	markers, _, _, _, err := s.loadJournal()
	return markers, err
}

func (s modelDownloadStore) loadJournal() (map[downloadContentKey]modelDownloadMarker, map[string]modelCopyMarker, bool, bool, error) {
	markers := make(map[downloadContentKey]modelDownloadMarker)
	copies := make(map[string]modelCopyMarker)
	if s.path == "" {
		return nil, nil, false, false, errDownloadPersistence
	}
	f, err := os.Open(s.path)
	if errors.Is(err, os.ErrNotExist) {
		// Missing historical copy provenance is never a known-clear baseline.
		// Downloads retain their existing missing-file behavior, and every later
		// download write preserves this explicit lack of copy completeness.
		info, dirErr := os.Stat(filepath.Dir(s.path))
		if dirErr == nil && info.IsDir() {
			return markers, copies, false, false, nil
		}
		return nil, nil, false, false, errDownloadPersistence
	}
	if err != nil {
		return nil, nil, false, false, errDownloadPersistence
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, modeldownload.MaxJournalBytes+1))
	if err != nil || len(data) > modeldownload.MaxJournalBytes || !utf8.Valid(data) {
		return nil, nil, false, false, errDownloadPersistence
	}
	var journal modelDownloadJournal
	if json.Unmarshal(data, &journal) != nil {
		return nil, nil, false, false, errDownloadPersistence
	}
	if journal.Version != 1 || journal.Entries == nil || journal.Copies == nil || len(journal.Entries)+len(journal.Copies) > modeldownload.MaxUnsafeEntries {
		return nil, nil, false, false, errDownloadPersistence
	}
	ids := make(map[string]bool)
	for _, marker := range journal.Entries {
		key, err := downloadKey(marker.Engine, marker.Model, marker.Naming)
		if err != nil || key.model != marker.ContentKey || ids[marker.ID] {
			return nil, nil, false, false, errDownloadPersistence
		}
		// Reuse the wire validator, including engine and operation-ID syntax.
		probe := modeldownload.Snapshot{Version: 1, SourceID: marker.ID, Revision: "0", Entries: []modeldownload.Entry{{ID: marker.ID, Engine: marker.Engine, Model: marker.Model, Phase: "unknown", BlocksReadiness: true}}}
		if probe.Validate() != nil {
			return nil, nil, false, false, errDownloadPersistence
		}
		if _, duplicate := markers[key]; duplicate {
			return nil, nil, false, false, errDownloadPersistence
		}
		markers[key] = marker
		ids[marker.ID] = true
	}
	identities := make(map[string]bool)
	for _, marker := range journal.Copies {
		identity := marker.Role + "\x00" + marker.SourceNodeID + "\x00" + marker.DestinationNodeID + "\x00" + marker.Engine + "\x00" + marker.ContentKey
		if marker.validate() != nil || ids[marker.OpID] || identities[identity] {
			return nil, nil, false, false, errDownloadPersistence
		}
		ids[marker.OpID], identities[identity] = true, true
		copies[marker.OpID] = marker
	}
	return markers, copies, journal.CopiesComplete, journal.CopyRecoveryRequired, nil
}

func (s modelDownloadStore) save(markers map[downloadContentKey]modelDownloadMarker, copies map[string]modelCopyMarker, copiesComplete, copyRecoveryRequired bool) (landed, uncertain bool, err error) {
	if s.path == "" || len(markers)+len(copies) > modeldownload.MaxUnsafeEntries {
		return false, false, errDownloadPersistence
	}
	journal := modelDownloadJournal{Version: 1, Entries: make([]modelDownloadMarker, 0, len(markers)), Copies: make([]modelCopyMarker, 0, len(copies)), CopiesComplete: copiesComplete, CopyRecoveryRequired: copyRecoveryRequired}
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
	for _, marker := range copies {
		journal.Copies = append(journal.Copies, marker)
	}
	sort.Slice(journal.Copies, func(i, j int) bool { return journal.Copies[i].OpID < journal.Copies[j].OpID })
	data, err := json.Marshal(journal)
	if err != nil || len(data) > modeldownload.MaxJournalBytes {
		return false, false, errDownloadPersistence
	}
	// Validate our own output too, before the first filesystem effect.
	var checked modelDownloadJournal
	if json.Unmarshal(data, &checked) != nil {
		return false, false, errDownloadPersistence
	}
	ids, identities := make(map[string]bool), make(map[string]bool)
	for _, marker := range journal.Entries {
		key, keyErr := downloadKey(marker.Engine, marker.Model, marker.Naming)
		if keyErr != nil || key.model != marker.ContentKey || !lowerHex(marker.ID, 32) || ids[marker.ID] || markers[key] != marker {
			return false, false, errDownloadPersistence
		}
		ids[marker.ID] = true
	}
	for _, marker := range journal.Copies {
		identity := marker.Role + "\x00" + marker.SourceNodeID + "\x00" + marker.DestinationNodeID + "\x00" + marker.Engine + "\x00" + marker.ContentKey
		if marker.validate() != nil || ids[marker.OpID] || identities[identity] || copies[marker.OpID] != marker {
			return false, false, errDownloadPersistence
		}
		ids[marker.OpID], identities[identity] = true, true
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
