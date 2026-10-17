// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package durablefile

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

var errInjected = errors.New("injected failure")

// faultFile wraps a real temporary file and fails the chosen step. Close
// always releases the underlying descriptor.
type faultFile struct {
	File
	writeErr, syncErr, closeErr error
}

func (f faultFile) Write(p []byte) (int, error) {
	if f.writeErr != nil {
		return 0, f.writeErr
	}
	return f.File.Write(p)
}

func (f faultFile) Sync() error {
	if f.syncErr != nil {
		return f.syncErr
	}
	return f.File.Sync()
}

func (f faultFile) Close() error {
	err := f.File.Close()
	if f.closeErr != nil {
		return f.closeErr
	}
	return err
}

func writeTarget(t *testing.T, dir, content string) string {
	t.Helper()
	path := filepath.Join(dir, "policy.json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("seed target: %v", err)
	}
	return path
}

func readString(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(b)
}

// assertOnlyTarget fails when dir holds anything besides the target file, which
// is how a leaked temporary sibling shows up.
func assertOnlyTarget(t *testing.T, dir, target string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir %s: %v", dir, err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	if want := []string{filepath.Base(target)}; !reflect.DeepEqual(names, want) {
		t.Fatalf("directory entries = %v, want %v", names, want)
	}
}

func TestWriteFilePublishesBytes(t *testing.T) {
	t.Run("replaces an existing file", func(t *testing.T) {
		dir := t.TempDir()
		path := writeTarget(t, dir, "old contents that are longer than the new ones")
		if err := WriteFile(path, []byte("new"), 0o600); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		if got := readString(t, path); got != "new" {
			t.Fatalf("target = %q, want %q", got, "new")
		}
		assertOnlyTarget(t, dir, path)
	})
	t.Run("creates a missing file", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "policy.json")
		if err := WriteFile(path, []byte("fresh"), 0o600); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		if got := readString(t, path); got != "fresh" {
			t.Fatalf("target = %q, want %q", got, "fresh")
		}
		assertOnlyTarget(t, dir, path)
	})
	t.Run("reports a missing parent directory", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "absent", "policy.json")
		err := WriteFile(path, []byte("x"), 0o600)
		if !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), "create temp") {
			t.Fatalf("error = %v, want a create temp failure wrapping ErrNotExist", err)
		}
	})
}

func TestWriteFileFailureBeforeReplaceKeepsTarget(t *testing.T) {
	const original = "original bytes"
	test := func(name, wantStep string, inject func(*Replacer)) {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := writeTarget(t, dir, original)
			var r Replacer
			inject(&r)
			err := r.WriteFile(path, []byte("replacement"), 0o600)
			if !errors.Is(err, errInjected) {
				t.Fatalf("error = %v, want it to wrap the injected failure", err)
			}
			if !strings.Contains(err.Error(), wantStep) {
				t.Fatalf("error = %q, want it to name step %q", err, wantStep)
			}
			if got := readString(t, path); got != original {
				t.Fatalf("target = %q, want unchanged %q", got, original)
			}
			assertOnlyTarget(t, dir, path)
		})
	}
	fault := func(f faultFile) func(*Replacer) {
		return func(r *Replacer) {
			r.CreateTemp = func(dir, base string, perm os.FileMode) (File, error) {
				real, err := createTemp(dir, base, perm)
				if err != nil {
					return nil, err
				}
				f.File = real
				return f, nil
			}
		}
	}
	test("temp create fails", "create temp", func(r *Replacer) {
		r.CreateTemp = func(string, string, os.FileMode) (File, error) { return nil, errInjected }
	})
	test("temp write fails", "write", fault(faultFile{writeErr: errInjected}))
	test("temp sync fails", "sync", fault(faultFile{syncErr: errInjected}))
	test("temp close fails", "close", fault(faultFile{closeErr: errInjected}))
	test("replace fails", "replace", func(r *Replacer) {
		r.Replace = func(string, string) error { return errInjected }
	})
}

func TestWriteFileDirectorySyncFailureLeavesNewBytes(t *testing.T) {
	dir := t.TempDir()
	path := writeTarget(t, dir, "original bytes")
	r := Replacer{SyncDir: func(string) error { return errInjected }}
	err := r.WriteFile(path, []byte("replacement"), 0o600)
	if !errors.Is(err, errInjected) || !strings.Contains(err.Error(), "sync directory") {
		t.Fatalf("error = %v, want a sync directory failure wrapping the injected failure", err)
	}
	if got := readString(t, path); got != "replacement" {
		t.Fatalf("target = %q, want the complete new bytes", got)
	}
	assertOnlyTarget(t, dir, path)
}

func TestWriteFileOrdersSteps(t *testing.T) {
	dir := t.TempDir()
	path := writeTarget(t, dir, "original bytes")
	var steps []string
	var syncedBeforeReplace string
	r := Replacer{
		CreateTemp: func(dir, base string, perm os.FileMode) (File, error) {
			steps = append(steps, "create")
			real, err := createTemp(dir, base, perm)
			if err != nil {
				return nil, err
			}
			return &recordingFile{File: real, steps: &steps}, nil
		},
		Replace: func(tmp, target string) error {
			steps = append(steps, "replace")
			syncedBeforeReplace = readString(t, tmp)
			if got := readString(t, target); got != "original bytes" {
				t.Errorf("target at replace = %q, want it untouched until the replace", got)
			}
			return replaceFile(tmp, target)
		},
		SyncDir: func(d string) error {
			steps = append(steps, "syncdir")
			if d != dir {
				t.Errorf("SyncDir dir = %q, want %q", d, dir)
			}
			return syncDir(d)
		},
	}
	if err := r.WriteFile(path, []byte("replacement"), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	want := []string{"create", "write", "sync", "close", "replace", "syncdir"}
	if !reflect.DeepEqual(steps, want) {
		t.Fatalf("steps = %v, want %v", steps, want)
	}
	if syncedBeforeReplace != "replacement" {
		t.Fatalf("temp contents at replace = %q, want the complete new bytes", syncedBeforeReplace)
	}
}

type recordingFile struct {
	File
	steps *[]string
}

func (f *recordingFile) Write(p []byte) (int, error) {
	*f.steps = append(*f.steps, "write")
	return f.File.Write(p)
}

func (f *recordingFile) Sync() error {
	*f.steps = append(*f.steps, "sync")
	return f.File.Sync()
}

func (f *recordingFile) Close() error {
	*f.steps = append(*f.steps, "close")
	return f.File.Close()
}

func TestWriteFileRetriesTransientReplace(t *testing.T) {
	errTransient := errors.New("transient conflict")
	isTransient := func(err error) bool { return errors.Is(err, errTransient) }
	type outcome struct {
		err      error
		attempts int
		delays   []time.Duration
		target   string
	}
	// run drives WriteFile with a Replace that fails failures times before
	// moving the file.
	run := func(t *testing.T, failures int, failWith error) outcome {
		t.Helper()
		dir := t.TempDir()
		path := writeTarget(t, dir, "original bytes")
		attempts := 0
		var delays []time.Duration
		r := Replacer{
			Replace: func(tmp, target string) error {
				attempts++
				if attempts <= failures {
					return failWith
				}
				return replaceFile(tmp, target)
			},
			IsTransient: isTransient,
			Sleep:       func(d time.Duration) { delays = append(delays, d) },
		}
		err := r.WriteFile(path, []byte("replacement"), 0o600)
		assertOnlyTarget(t, dir, path)
		return outcome{err: err, attempts: attempts, delays: delays, target: readString(t, path)}
	}
	t.Run("succeeds once the conflict clears", func(t *testing.T) {
		got := run(t, 2, errTransient)
		if got.err != nil {
			t.Fatalf("WriteFile: %v", got.err)
		}
		if got.attempts != 3 {
			t.Fatalf("attempts = %d, want 3", got.attempts)
		}
		if want := []time.Duration{50 * time.Millisecond, 100 * time.Millisecond}; !reflect.DeepEqual(got.delays, want) {
			t.Fatalf("delays = %v, want %v", got.delays, want)
		}
		if got.target != "replacement" {
			t.Fatalf("target = %q, want the new bytes", got.target)
		}
	})
	t.Run("gives up after ten attempts and keeps the target", func(t *testing.T) {
		got := run(t, 1000, errTransient)
		if !errors.Is(got.err, errTransient) || !strings.Contains(got.err.Error(), "replace") {
			t.Fatalf("error = %v, want a replace failure wrapping the transient error", got.err)
		}
		if got.attempts != 10 || len(got.delays) != 9 {
			t.Fatalf("attempts = %d, sleeps = %d, want 10 and 9", got.attempts, len(got.delays))
		}
		if got.target != "original bytes" {
			t.Fatalf("target = %q, want unchanged", got.target)
		}
	})
	t.Run("does not retry a permanent error", func(t *testing.T) {
		got := run(t, 1000, errInjected)
		if !errors.Is(got.err, errInjected) {
			t.Fatalf("error = %v, want the permanent error", got.err)
		}
		if got.attempts != 1 || len(got.delays) != 0 {
			t.Fatalf("attempts = %d, sleeps = %d, want 1 and 0", got.attempts, len(got.delays))
		}
		if got.target != "original bytes" {
			t.Fatalf("target = %q, want unchanged", got.target)
		}
	})
}

func TestCreateTempNamesAreUniqueSiblings(t *testing.T) {
	dir := t.TempDir()
	seen := map[string]bool{}
	for i := 0; i < 20; i++ {
		f, err := createTemp(dir, "policy.json", 0o600)
		if err != nil {
			t.Fatalf("createTemp: %v", err)
		}
		if err := f.Close(); err != nil {
			t.Fatalf("close temp: %v", err)
		}
		if filepath.Dir(f.Name()) != dir {
			t.Fatalf("temp %q is not a sibling in %q", f.Name(), dir)
		}
		if seen[f.Name()] {
			t.Fatalf("temp name %q was minted twice", f.Name())
		}
		seen[f.Name()] = true
	}
}
