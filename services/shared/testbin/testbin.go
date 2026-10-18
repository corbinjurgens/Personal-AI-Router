// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package testbin allocates executable directories for test fixtures.
package testbin

import (
	"bufio"
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

const EnvRoot = "NVPAIR_TEST_BIN_DIR"

//go:embed programs/*.txt
var manifest embed.FS

// Programs returns the sorted module/role/program paths in the manifest.
func Programs() ([]string, error) {
	return parsePrograms(manifest)
}

func parsePrograms(files fs.FS) ([]string, error) {
	names, err := fs.Glob(files, "programs/*.txt")
	if err != nil {
		return nil, err
	}
	seen := make(map[string]bool)
	var programs []string
	for _, name := range names {
		data, err := fs.ReadFile(files, name)
		if err != nil {
			return nil, err
		}
		module := strings.TrimSuffix(filepath.Base(name), ".txt")
		scanner := bufio.NewScanner(strings.NewReader(string(data)))
		line := 0
		for scanner.Scan() {
			line++
			entry := strings.TrimSpace(scanner.Text())
			if entry == "" || strings.HasPrefix(entry, "#") {
				continue
			}
			parts := strings.Split(entry, "/")
			if len(parts) != 3 || !validName(parts[0]) || !validName(parts[1]) || !validName(parts[2]) || strings.HasSuffix(strings.ToLower(parts[2]), ".exe") {
				return nil, fmt.Errorf("%s:%d: invalid program %q; want module/role/program without .exe", name, line, entry)
			}
			if parts[0] != module {
				return nil, fmt.Errorf("%s:%d: module %q must own its manifest file", name, line, parts[0])
			}
			if seen[entry] {
				return nil, fmt.Errorf("%s:%d: duplicate program %q", name, line, entry)
			}
			seen[entry] = true
			programs = append(programs, entry)
		}
		if err := scanner.Err(); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
	}
	sort.Strings(programs)
	return programs, nil
}

func validName(name string) bool {
	if name == "" || name == "." || name == ".." {
		return false
	}
	for _, c := range name {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}

func validate(module, role string, programs []string) error {
	if len(programs) == 0 {
		return fmt.Errorf("testbin: at least one program is required for %s/%s", module, role)
	}
	listed, err := Programs()
	if err != nil {
		return err
	}
	for _, program := range programs {
		entry := module + "/" + role + "/" + program
		index := sort.SearchStrings(listed, entry)
		if index == len(listed) || listed[index] != entry {
			return fmt.Errorf("testbin: missing manifest line %q", entry)
		}
	}
	return nil
}

func fixedDir(module, role string) (string, error) {
	root := os.Getenv(EnvRoot)
	if !filepath.IsAbs(root) {
		return "", fmt.Errorf("testbin: %s must be absolute: %q", EnvRoot, root)
	}
	dir := filepath.Join(root, module, role)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", fmt.Errorf("testbin: create directory: %w", err)
	}
	return dir, nil
}

// Dir returns a temporary directory, or a persistent module/role directory
// beneath EnvRoot when set. Every program must be listed in the manifest.
func Dir(tb testing.TB, module, role string, programs ...string) string {
	tb.Helper()
	if err := validate(module, role, programs); err != nil {
		tb.Fatalf("%v", err)
	}
	if os.Getenv(EnvRoot) == "" {
		return tb.TempDir()
	}
	dir, err := fixedDir(module, role)
	if err != nil {
		tb.Fatalf("%v", err)
	}
	return dir
}

// MainDir is Dir's TestMain counterpart. Cleanup removes temporary directories
// and leaves fixed directories intact.
func MainDir(module, role, pattern string, programs ...string) (dir string, cleanup func(), err error) {
	if err := validate(module, role, programs); err != nil {
		return "", nil, err
	}
	if os.Getenv(EnvRoot) != "" {
		dir, err := fixedDir(module, role)
		if err != nil {
			return "", nil, err
		}
		return dir, func() {}, nil
	}
	dir, err = os.MkdirTemp("", pattern)
	if err != nil {
		return "", nil, err
	}
	return dir, func() { _ = os.RemoveAll(dir) }, nil
}

// Exe adds the executable suffix used on Windows.
func Exe(name string) string {
	if runtime.GOOS == "windows" {
		return name + ".exe"
	}
	return name
}
