// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package buildinfo exposes only the executable's bounded public source identity.
package buildinfo

import (
	"encoding/json"
	"runtime/debug"
	"strings"
)

// ID is an optional Git source revision, with +dirty for modified sources.
// It is diagnostic metadata, not an artifact hash or a trust assertion.
type ID string

// Current reads this executable's embedded Go VCS metadata. Unknown metadata
// remains empty rather than implying an unmodified source revision.
func Current() ID {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	return fromSettings(info.Settings)
}

func fromSettings(settings []debug.BuildSetting) ID {
	var vcs, revision, modified string
	for _, setting := range settings {
		switch setting.Key {
		case "vcs":
			vcs = setting.Value
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value
		}
	}
	if vcs != "git" || (modified != "true" && modified != "false") || (len(revision) != 40 && len(revision) != 64) {
		return ""
	}
	id := "git:" + revision
	if modified == "true" {
		id += "+dirty"
	}
	if !valid(id) {
		return ""
	}
	return ID(id)
}

func valid(value string) bool {
	if len(value) > 74 || !strings.HasPrefix(value, "git:") {
		return false
	}
	revision := strings.TrimSuffix(value[4:], "+dirty")
	if len(revision) != 40 && len(revision) != 64 {
		return false
	}
	for _, c := range revision {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// UnmarshalJSON isolates malformed optional metadata from its enclosing node.
// The encoded bound permits JSON Unicode escapes for every valid ASCII byte.
func (id *ID) UnmarshalJSON(data []byte) error {
	*id = ""
	if len(data) > 74*6+2 {
		return nil
	}
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return nil
	}
	if valid(value) {
		*id = ID(value)
	}
	return nil
}
