// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"os"
	"path/filepath"

	"nvpair-shared/appdir"
	"nvpair-shared/durablefile"
)

type persistedPort struct {
	Port int `json:"port"`
}

// chooseStartupPort resolves the port to bind. A previously chosen port wins
// over the flag so the proxy comes back where the user last put it, except
// when the broker forces an explicit port or when the stored value is one the
// engine itself owns — restoring that would put the proxy on the engine's
// port and guarantee a bind conflict.
func chooseStartupPort(p engineProfile, flagPort int, ignorePersisted bool, persisted int, hasPersisted bool) int {
	if ignorePersisted || !hasPersisted {
		return flagPort
	}
	if p.ReservedPersistedPort != 0 && persisted == p.ReservedPersistedPort {
		return flagPort
	}
	return persisted
}

func proxyPortPath(p engineProfile) (string, error) {
	return appdir.Path(p.PortFile)
}

// loadPersistedPort returns the previously chosen proxy port, if a valid one
// was saved. Any error (no file, bad JSON, out-of-range) reports "none" so
// startup falls back to the --port flag / default.
func loadPersistedPort(p engineProfile) (int, bool) {
	path, err := proxyPortPath(p)
	if err != nil {
		return 0, false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	var pp persistedPort
	if err := json.Unmarshal(data, &pp); err != nil {
		return 0, false
	}
	if pp.Port < 1 || pp.Port > 65535 {
		return 0, false
	}
	return pp.Port, true
}

// savePersistedPort durably replaces the chosen port. ErrReplaced means the
// new port is on disk even though its durability could not be confirmed.
func savePersistedPort(p engineProfile, port int, files durablefile.Replacer) error {
	path, err := proxyPortPath(p)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(persistedPort{Port: port})
	if err != nil {
		return err
	}
	return files.WriteFile(path, data, 0o644)
}
