// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"nvpair-shared/durablefile"
)

type syncFailureFile struct {
	*os.File
	err error
}

func (f syncFailureFile) Sync() error { return f.err }

func useEngineFileReplacer(t *testing.T, replacer durablefile.Replacer) {
	t.Helper()
	previous := engineFileReplacer
	engineFileReplacer = replacer
	t.Cleanup(func() { engineFileReplacer = previous })
}

func assertDurableFile(t *testing.T, path string, want []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("file bytes = %q, want %q", got, want)
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != filepath.Base(path) {
		t.Fatalf("directory entries = %v, want only %s", entries, filepath.Base(path))
	}
}

// Both wrappers must use the same replacement guarantees while retaining their
// distinct byte formats and modes. These tests never run in parallel: the
// replacer is a package-wide fault-injection seam.
func durableWriterCases(t *testing.T, test func(*testing.T, func(string) error, []byte, os.FileMode)) {
	t.Helper()
	t.Run("cache ref", func(t *testing.T) {
		data := []byte("0123456789012345678901234567890123456789")
		test(t, func(path string) error { return writeFileAtomic(path, data) }, data, 0o644)
	})
	t.Run("JSON", func(t *testing.T) {
		test(t, func(path string) error {
			return writeJSONAtomic(path, map[string]string{"engine": "test"})
		}, []byte("{\n  \"engine\": \"test\"\n}"), 0o600)
	})
}

func TestDurableWritersPreserveTargetOnSyncFailure(t *testing.T) {
	durableWriterCases(t, func(t *testing.T, write func(string) error, _ []byte, _ os.FileMode) {
		path := filepath.Join(t.TempDir(), "target")
		old := []byte("previous contents")
		if err := os.WriteFile(path, old, 0o600); err != nil {
			t.Fatal(err)
		}
		failure := errors.New("injected temp sync failure")
		useEngineFileReplacer(t, durablefile.Replacer{
			CreateTemp: func(dir, base string, _ os.FileMode) (durablefile.File, error) {
				file, err := os.CreateTemp(dir, base+".*.tmp")
				if err != nil {
					return nil, err
				}
				return syncFailureFile{File: file, err: failure}, nil
			},
		})
		if err := write(path); !errors.Is(err, failure) || errors.Is(err, durablefile.ErrReplaced) {
			t.Fatalf("write error = %v, want pre-replacement sync failure", err)
		}
		assertDurableFile(t, path, old)
	})
}

func TestDurableWritersAdoptUncertainReplacement(t *testing.T) {
	durableWriterCases(t, func(t *testing.T, write func(string) error, want []byte, _ os.FileMode) {
		path := filepath.Join(t.TempDir(), "target")
		if err := os.WriteFile(path, []byte("previous contents"), 0o600); err != nil {
			t.Fatal(err)
		}
		useEngineFileReplacer(t, durablefile.Replacer{
			SyncDir: func(string) error { return errors.New("injected directory sync failure") },
		})
		var logs bytes.Buffer
		previous := slog.Default()
		slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
		t.Cleanup(func() { slog.SetDefault(previous) })
		if err := write(path); err != nil {
			t.Fatalf("landed replacement reported failure: %v", err)
		}
		assertDurableFile(t, path, want)
		var warning struct {
			Level string `json:"level"`
			Path  string `json:"path"`
		}
		if err := json.Unmarshal(logs.Bytes(), &warning); err != nil {
			t.Fatalf("decode warning: %v", err)
		}
		if warning.Level != "WARN" || warning.Path != path {
			t.Fatalf("warning = %+v, want WARN with path %q", warning, path)
		}
	})
}

func TestDurableWritersPreserveFormatAndMode(t *testing.T) {
	durableWriterCases(t, func(t *testing.T, write func(string) error, want []byte, mode os.FileMode) {
		path := filepath.Join(t.TempDir(), "target")
		if err := write(path); err != nil {
			t.Fatal(err)
		}
		assertDurableFile(t, path, want)
		// The old writers create files with these modes before umask. A direct
		// write supplies a reference that respects the current process umask.
		control := filepath.Join(t.TempDir(), "control")
		if err := os.WriteFile(control, nil, mode); err != nil {
			t.Fatal(err)
		}
		got, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		baseline, err := os.Stat(control)
		if err != nil {
			t.Fatal(err)
		}
		if runtime.GOOS != "windows" && got.Mode().Perm() != baseline.Mode().Perm() {
			t.Fatalf("mode = %o, want %o", got.Mode().Perm(), baseline.Mode().Perm())
		}
		if err := write(path); err != nil {
			t.Fatalf("replace existing file: %v", err)
		}
		assertDurableFile(t, path, want)
	})
}

func TestDurableCacheRefCreatesParent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "refs", "main")
	if err := writeFileAtomic(path, []byte("revision")); err != nil {
		t.Fatal(err)
	}
	assertDurableFile(t, path, []byte("revision"))
	control := filepath.Join(t.TempDir(), "refs")
	if err := os.Mkdir(control, 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := os.Stat(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := os.Stat(control)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && got.Mode().Perm() != baseline.Mode().Perm() {
		t.Fatalf("parent mode = %o, want %o", got.Mode().Perm(), baseline.Mode().Perm())
	}
}

func TestDurableSetPortAdoptsUncertainReplacement(t *testing.T) {
	dir := t.TempDir()
	ex := newPortFixtureExecutor(t, dir)
	port := privateTestPorts(t, 1, hostPort(t, ex.reg, "ollama"))[0]
	useEngineFileReplacer(t, durablefile.Replacer{
		SyncDir: func(string) error { return errors.New("injected directory sync failure") },
	})
	status, err := ex.SetPort(context.Background(), "ollama", port)
	if err != nil {
		t.Fatal(err)
	}
	st, err := ex.state("ollama")
	if err != nil {
		t.Fatal(err)
	}
	if status.Port != port || st.port != port || st.plat.Runtime.Port != port {
		t.Fatalf("port status=%d cached=%d runtime=%d, want %d", status.Port, st.port, st.plat.Runtime.Port, port)
	}
	if got := hostPort(t, reloadPortFixture(t, ex.reg, dir), "ollama"); got != port {
		t.Fatalf("reloaded port = %d, want %d", got, port)
	}
}

func TestDurableDesiredStateAdoptsUncertainReplacement(t *testing.T) {
	store := newDesiredStateStore(t.TempDir())
	if err := store.set("test", false, intentLocal); err != nil {
		t.Fatal(err)
	}
	useEngineFileReplacer(t, durablefile.Replacer{
		SyncDir: func(string) error { return errors.New("injected directory sync failure") },
	})
	if err := store.set("test", true, intentLocal); err != nil {
		t.Fatal(err)
	}
	enabled, known, err := store.get("test")
	if err != nil || !known || !enabled {
		t.Fatalf("saved intent: enabled=%v known=%v err=%v", enabled, known, err)
	}
	assertDurableFile(t, store.path, []byte("{\n  \"engines\": {\n    \"test\": true\n  }\n}"))
}

func TestDurableInstallMarkerAdoptsUncertainReplacement(t *testing.T) {
	dir := t.TempDir()
	useEngineFileReplacer(t, durablefile.Replacer{
		SyncDir: func(string) error { return errors.New("injected directory sync failure") },
	})
	if err := writeInstallMarker(dir, "test"); err != nil {
		t.Fatal(err)
	}
	if !installedByPAIR(dir) {
		t.Fatal("landed install marker was not adopted")
	}
	data, err := os.ReadFile(installMarkerPath(dir))
	if err != nil {
		t.Fatal(err)
	}
	var marker installMarker
	if err := json.Unmarshal(data, &marker); err != nil {
		t.Fatal(err)
	}
	if marker.Engine != "test" {
		t.Fatalf("marker engine = %q, want test", marker.Engine)
	}
	if _, err := time.Parse(time.RFC3339, marker.InstalledAt); err != nil {
		t.Fatalf("marker timestamp: %v", err)
	}
}

func TestDurableLlamaFinalizeAdoptsUncertainReplacement(t *testing.T) {
	root := t.TempDir()
	const revision = "0123456789012345678901234567890123456789"
	ex := &Executor{}
	// A stopped router returns before any HTTP request in reloadLlamaModels.
	plan, err := ex.llamacppPlan(&engineState{}, root, "owner/repo", modelFileList{Revision: revision})
	if err != nil {
		t.Fatal(err)
	}
	useEngineFileReplacer(t, durablefile.Replacer{
		SyncDir: func(string) error { return errors.New("injected directory sync failure") },
	})
	if err := plan.finalize(context.Background()); err != nil {
		t.Fatalf("landed cache ref reported failed finalization: %v", err)
	}
	repo := llamaRepoDir(root, "owner/repo")
	if got := llamaCachedRef(repo); got != revision {
		t.Fatalf("cached revision = %q, want %q", got, revision)
	}
	assertDurableFile(t, filepath.Join(repo, "refs", "main"), []byte(revision))
}

func TestDurableWritersWindowsSharingViolation(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows reader sharing contract")
	}
	durableWriterCases(t, func(t *testing.T, write func(string) error, want []byte, _ os.FileMode) {
		path := filepath.Join(t.TempDir(), "target")
		old := []byte("previous contents")
		if err := os.WriteFile(path, old, 0o600); err != nil {
			t.Fatal(err)
		}
		// Go's Windows os.Open shares reads/writes but not deletion, preventing
		// MoveFileEx from replacing the target until this reader closes.
		reader, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if reader != nil {
				if err := reader.Close(); err != nil {
					t.Error(err)
				}
			}
		})
		waits := 0
		useEngineFileReplacer(t, durablefile.Replacer{Sleep: func(time.Duration) { waits++ }})
		if err := write(path); err == nil || errors.Is(err, durablefile.ErrReplaced) {
			t.Fatalf("held-reader replacement error = %v, want pre-replacement failure", err)
		}
		if waits != 9 {
			t.Fatalf("retry waits = %d, want 9 before exhausting 10 attempts", waits)
		}
		assertDurableFile(t, path, old)
		if err := reader.Close(); err != nil {
			t.Fatal(err)
		}
		reader = nil
		if err := write(path); err != nil {
			t.Fatalf("replace after releasing reader: %v", err)
		}
		assertDurableFile(t, path, want)
	})
}
