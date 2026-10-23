// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"nvpair-shared/engines"
)

// strataKeyFileLimit bounds the read: the longest accepted key plus a line end.
const strataKeyFileLimit = 514

// strataKeyIdentity binds the key file without its content: whatever is hashed
// into an advertised field must not let a weak key be guessed offline. A
// rewrite that keeps the same file, size and modification time is not seen.
type strataKeyIdentity struct {
	Path    string
	Size    int64
	MtimeNS string
	FileID  string
}

// readStrataKeyFile reads the native API key with a bounded read of a regular,
// undeclared file and returns it with the file's identity. No error repeats
// any of the file's content.
func readStrataKeyFile(path string, artifacts []engines.ServingArtifact) (string, *strataKeyIdentity, error) {
	unusable := servingError("Strata api_key_file is missing, empty, unreadable or not a regular file.")
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return "", nil, servingError("Strata api_key_file must be an absolute clean path.")
	}
	before, err := os.Lstat(path)
	if err != nil || !before.Mode().IsRegular() || before.Size() == 0 || before.Size() > strataKeyFileLimit {
		return "", nil, unusable
	}
	for _, artifact := range artifacts {
		if artifact.Binding == nil {
			continue
		}
		if info, err := os.Stat(artifact.Binding.SourcePath); err != nil || os.SameFile(before, info) {
			return "", nil, servingError("Strata api_key_file must not be a declared artifact.")
		}
	}
	// The open itself neither follows a link nor blocks on a FIFO swapped in
	// after the check; the handle must then be the same regular file, and the
	// path must still name it without a link once the read is done.
	f, err := strataOpenKeyFile(path)
	if err != nil {
		return "", nil, unusable
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(before, opened) {
		return "", nil, servingError("Strata api_key_file changed while it was read.")
	}
	data, err := io.ReadAll(io.LimitReader(f, strataKeyFileLimit+1))
	if err != nil {
		return "", nil, unusable
	}
	id, idErr := strataFileID(f)
	after, statErr := f.Stat()
	named, lstatErr := os.Lstat(path)
	if idErr != nil || statErr != nil || lstatErr != nil || !named.Mode().IsRegular() || !os.SameFile(named, after) || !os.SameFile(before, after) || after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) || int64(len(data)) != after.Size() {
		return "", nil, servingError("Strata api_key_file changed while it was read.")
	}
	// One line end, as an editor or echo writes it, is not part of the key.
	key := strings.TrimSuffix(strings.TrimSuffix(string(data), "\n"), "\r")
	if !strataAPIKeyPattern.MatchString(key) {
		return "", nil, servingError("Strata api_key_file does not hold one reviewed key.")
	}
	return key, &strataKeyIdentity{Path: path, Size: after.Size(), MtimeNS: strconv.FormatInt(after.ModTime().UnixNano(), 10), FileID: id}, nil
}
