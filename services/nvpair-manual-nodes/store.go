// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
)

// stateFileName is the manual-node list inside the per-user data directory.
const stateFileName = "manual-nodes.json"

// stateVersion is the on-disk schema version.
const stateVersion = 1

// persistedState is the file's shape: the user's entries exactly as added, so
// a hostname stays a hostname and is resolved afresh on every probe.
type persistedState struct {
	Version int           `json:"version"`
	Nodes   []ManualEntry `json:"nodes"`
}

// loadEntries reads the persisted list. A missing file is an empty list.
func loadEntries(path string) ([]ManualEntry, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var st persistedState
	if err := json.Unmarshal(data, &st); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if st.Version != stateVersion {
		return nil, fmt.Errorf("%s has unsupported version %d", path, st.Version)
	}
	out := st.Nodes[:0]
	for _, e := range st.Nodes {
		if e.Address != "" {
			out = append(out, e)
		}
	}
	return out, nil
}

// saveEntries replaces the persisted list atomically: written to a temporary
// file beside it, synced, then renamed over it, so a crash leaves either the
// old list or the new one. The file is owner-only, since it names the user's
// machines and their addresses.
func saveEntries(path string, entries []ManualEntry) error {
	sorted := append([]ManualEntry(nil), entries...)
	sort.Slice(sorted, func(i, j int) bool { return nodeID(sorted[i]) < nodeID(sorted[j]) })
	data, err := json.MarshalIndent(persistedState{Version: stateVersion, Nodes: sorted}, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op once renamed
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// persist writes the current entries to the state file, if there is one. A
// failure is logged rather than returned: the node is tracked either way, and
// refusing node/add because the disk is full would be worse than a node that
// has to be re-added after a restart.
func (m *Manager) persist() {
	if m.statePath == "" {
		return
	}
	m.saveMu.Lock()
	defer m.saveMu.Unlock()
	m.mu.RLock()
	entries := make([]ManualEntry, 0, len(m.nodes))
	for _, tn := range m.nodes {
		entries = append(entries, tn.entry)
	}
	m.mu.RUnlock()
	if err := saveEntries(m.statePath, entries); err != nil {
		slog.Warn("could not save manual nodes", "path", m.statePath, "err", err)
	}
}

// replay re-adds every persisted entry, which probes each one and emits
// node/discovered exactly as a node/add would.
func (m *Manager) replay() {
	if m.statePath == "" {
		return
	}
	entries, err := loadEntries(m.statePath)
	if err != nil {
		// Moved aside rather than overwritten by the next add, so the
		// operator can still recover the entries from it.
		aside := m.statePath + ".invalid"
		slog.Warn("could not load saved manual nodes; moving the file aside",
			"path", m.statePath, "movedTo", aside, "err", err)
		_ = os.Rename(m.statePath, aside)
		return
	}
	for _, e := range entries {
		m.addNode(e)
	}
	if len(entries) > 0 {
		slog.Info("restored saved manual nodes", "count", len(entries))
	}
}
