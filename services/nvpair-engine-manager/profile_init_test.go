// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/engines"
)

func TestFreshProfileReloadAndCopyProvenance(t *testing.T) {
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	current := filepath.Join(base, "app")
	if _, err := initializeFreshProfile(current, filepath.Join(base, "previous"), durablefile.Replacer{}); err != nil {
		t.Fatal(err)
	}
	install := filepath.Join(current, "engine-bin")
	source := newOpID()
	d := newModelDownloads(source, install, map[string]string{"llamacpp": engines.ModelNamingExact})
	reg := &Registry{engines: map[string]*Manifest{"llamacpp": {Engine: "llamacpp", ModelNaming: engines.ModelNamingExact}}}
	m := &Manager{exec: &Executor{switchIncarnation: source}, copyJournal: d, copyRegistry: reg}
	evidence := m.ownerReservations()
	if !evidence.Complete || source == "" || evidence.Token.Incarnation != source || evidence.Token.DownloadSource != source {
		t.Fatalf("baseline evidence: %+v", evidence)
	}
	requireOwnerBool(t, evidence.Busy, false)
	emptySuccessor := newOpID()
	emptyDownloads := newModelDownloads(emptySuccessor, install, map[string]string{"llamacpp": engines.ModelNamingExact})
	emptyManager := &Manager{exec: &Executor{switchIncarnation: emptySuccessor}, copyJournal: emptyDownloads, copyRegistry: reg}
	emptyEvidence := emptyManager.ownerReservations()
	if emptySuccessor == source || emptySuccessor == "" || !emptyEvidence.Complete || emptyEvidence.Token.Incarnation != emptySuccessor || emptyEvidence.Token.DownloadSource != emptySuccessor {
		t.Fatal("successor lost empty baseline")
	}
	p := copyRequest{OpID: newOpID(), Engine: "llamacpp", Model: "model", SourceNodeID: "source", DestinationNodeID: "destination"}
	ctx, release, err := m.reserveCopy(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) {
		_, copies, complete, recovery, err := d.store.loadJournal()
		if err != nil || !complete || recovery || len(copies) != 1 || copies[p.OpID].ManagerIncarnation != source {
			t.Fatalf("before effect: copies=%v complete=%t recovery=%t err=%v", copies, complete, recovery, err)
		}
		return copyModelResult{}, copyUnknown(io.EOF)
	})
	release()
	if err == nil {
		t.Fatal("unknown completion succeeded")
	}
	successor := newOpID()
	if successor == source {
		t.Fatal("successor reused incarnation")
	}
	next := newModelDownloads(successor, install, map[string]string{"llamacpp": engines.ModelNamingExact})
	_, copies, complete, _, err := next.store.loadJournal()
	if err != nil || !complete || len(copies) != 1 || copies[p.OpID].ManagerIncarnation != source {
		t.Fatalf("successor provenance: copies=%v complete=%t err=%v", copies, complete, err)
	}
	nextManager := &Manager{exec: &Executor{switchIncarnation: successor}, copyJournal: next, copyRegistry: reg}
	if nextManager.ownerReservations().Complete {
		t.Fatal("unknown predecessor marked complete")
	}
}

func TestFreshProfileCLI(t *testing.T) {
	cfg, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	home, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, managerBin, "--initialize-fresh-profile")
	cmd.Env = isolatedManagerEnv(cfg, home)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("CLI: %v: %s", err, out)
	}
	if !strings.Contains(string(out), "fresh journal publication and readback confirmed") || strings.Contains(string(out), "ready") {
		t.Fatalf("unexpected output: %s", out)
	}
	for _, text := range []string{"jsonrpc", "listening", "loaded ", "using stdio", "using IPC"} {
		if strings.Contains(string(out), text) {
			t.Fatalf("runtime output: %s", out)
		}
	}
	for _, root := range []string{cfg, home} {
		for path, content := range freshTree(t, root) {
			if content != "directory" && filepath.Base(path) != modelDownloadFile {
				t.Fatalf("unexpected runtime artifact: %s", path)
			}
		}
	}
	if runtime.GOOS == "windows" && !strings.Contains(string(out), "fresh-directory durability: UNVERIFIED (unsupported)") {
		t.Fatalf("missing Windows status: %s", out)
	}
	ctx2, cancel2 := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel2()
	second := exec.CommandContext(ctx2, managerBin, "--initialize-fresh-profile")
	second.Env = isolatedManagerEnv(cfg, home)
	if out, err := second.CombinedOutput(); err == nil {
		t.Fatalf("second invocation succeeded: %s", out)
	}
}

func TestFreshProfileCLIRejectsArguments(t *testing.T) {
	for _, args := range [][]string{
		{"--initialize-fresh-profile=false"},
		{"--initialize-fresh-profile", "--version"},
		{"--initialize-fresh-profile", "--uninstall-managed=false"},
		{"--initialize-fresh-profile", "--http-port=0"},
		{"--initialize-fresh-profile", "--control-port=0"},
		{"--initialize-fresh-profile", "--reserved-port=0"},
		{"--initialize-fresh-profile", "--ipc="},
		{"--initialize-fresh-profile", "--cluster-dir="},
		{"--initialize-fresh-profile", "--loaded-poll-interval=0"},
		{"--initialize-fresh-profile", "--control-host=127.0.0.1"},
		{"--initialize-fresh-profile", "extra"},
		{"extra", "--initialize-fresh-profile"},
		{"--version", "extra", "--initialize-fresh-profile"},
		{"--version", "--", "--initialize-fresh-profile"},
		{"--", "--initialize-fresh-profile"},
	} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			cfg, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			home, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, managerBin, args...)
			cmd.Env = isolatedManagerEnv(cfg, home)
			out, err := cmd.CombinedOutput()
			if err == nil || ctx.Err() != nil {
				t.Fatalf("expected clean refusal: %v %s", err, out)
			}
			if strings.Contains(string(out), "[nvpair-engine-manager]") || strings.Contains(string(out), `"jsonrpc"`) {
				t.Fatal("argument refusal entered runtime logging or transport")
			}
			for _, root := range []string{cfg, home} {
				entries, err := os.ReadDir(root)
				if err != nil || len(entries) != 0 {
					t.Fatalf("runtime artifacts in %s: %v %v", root, entries, err)
				}
			}
		})
	}
}

func TestFreshProfileWindowsLocalBaseGuard(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows environment guard; pending PC3")
	}
	for _, value := range []string{"", "relative"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("LOCALAPPDATA", value)
			t.Setenv("APPDATA", t.TempDir())
			if err := offlineEnvironmentGuard(); err == nil {
				t.Fatal("invalid LOCALAPPDATA accepted")
			}
		})
	}
	t.Setenv("LOCALAPPDATA", "restore")
	if err := os.Unsetenv("LOCALAPPDATA"); err != nil {
		t.Fatal(err)
	}
	if err := offlineEnvironmentGuard(); err == nil {
		t.Fatal("missing LOCALAPPDATA accepted")
	}
}

func TestFreshProfilePersistenceAndRefusal(t *testing.T) {
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	current := filepath.Join(base, "org", "app")
	previous := filepath.Join(base, "old", "app")
	unsupported, err := initializeFreshProfile(current, previous, durablefile.Replacer{})
	if err != nil {
		t.Fatal(err)
	}
	if unsupported != (runtime.GOOS == "windows") {
		t.Fatalf("unsupported = %t", unsupported)
	}
	store := newModelDownloadStore(filepath.Join(current, "engine-bin"))
	if runtime.GOOS != "windows" {
		for path, mode := range map[string]os.FileMode{current: 0700, filepath.Join(current, "engine-bin"): 0700, store.path: 0600} {
			info, err := os.Stat(path)
			if err != nil || info.Mode().Perm() != mode {
				t.Fatalf("private mode %s: info=%v err=%v", path, info, err)
			}
		}
	}
	downloads, copies, complete, recovery, err := store.loadJournal()
	if err != nil || len(downloads) != 0 || len(copies) != 0 || !complete || recovery {
		t.Fatalf("readback: downloads=%v copies=%v complete=%t recovery=%t err=%v", downloads, copies, complete, recovery, err)
	}
	before, err := os.ReadFile(store.path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := initializeFreshProfile(current, previous, durablefile.Replacer{}); err == nil {
		t.Fatal("second initialization succeeded")
	}
	after, err := os.ReadFile(store.path)
	if err != nil || string(after) != string(before) {
		t.Fatalf("journal changed: %v", err)
	}
}

func TestFreshProfileDirectoryConfirmations(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix directory confirmation; Windows unsupported capability tested separately")
	}
	for failure := -1; failure < 4; failure++ {
		t.Run(fmt.Sprintf("boundary_%d", failure), func(t *testing.T) {
			base, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			current := filepath.Join(base, "one", "org", "app")
			previous := filepath.Join(base, "old", "app")
			var calls []string
			r := durablefile.Replacer{SyncDir: func(path string) error {
				calls = append(calls, path)
				// The first call confirms journal replacement.
				if failure >= 0 && len(calls)-2 == failure {
					return errors.New("confirmation failed")
				}
				return nil
			}}
			_, err = initializeFreshProfile(current, previous, r)
			if (err != nil) != (failure >= 0) {
				t.Fatalf("error = %v", err)
			}
			want := []string{filepath.Join(current, "engine-bin"), current, filepath.Dir(current), filepath.Join(base, "one"), base}
			if failure >= 0 {
				want = want[:failure+2]
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("sync calls = %v, want %v", calls, want)
			}
			if _, err := os.ReadFile(filepath.Join(current, "engine-bin", modelDownloadFile)); err != nil {
				t.Fatal(err)
			}
			if _, err := initializeFreshProfile(current, previous, durablefile.Replacer{}); err == nil {
				t.Fatal("second initialization succeeded")
			}
		})
	}
}

func TestFreshProfileCompetingAllocations(t *testing.T) {
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	current := filepath.Join(base, "app")
	previous := filepath.Join(base, "previous")
	var wg sync.WaitGroup
	var writesMu sync.Mutex
	writes := 0
	replace := durablefile.Replacer{CreateTemp: func(dir, base string, mode os.FileMode) (durablefile.File, error) {
		writesMu.Lock()
		writes++
		writesMu.Unlock()
		return os.CreateTemp(dir, base+".*.tmp")
	}}
	results := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := initializeFreshProfile(current, previous, replace)
			results <- err
		}()
	}
	wg.Wait()
	close(results)
	success := 0
	for err := range results {
		if err == nil {
			success++
		}
	}
	if success != 1 {
		t.Fatalf("successes = %d, want 1", success)
	}
	if writes != 1 {
		t.Fatalf("journal publication attempts=%d, want exactly winner's write", writes)
	}
	_, _, complete, recovery, err := newModelDownloadStore(filepath.Join(current, "engine-bin")).loadJournal()
	if err != nil || !complete || recovery {
		t.Fatalf("winner journal: complete=%t recovery=%t err=%v", complete, recovery, err)
	}
}

func TestFreshProfileInspectionFailure(t *testing.T) {
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	current := filepath.Join(base, "org", "app")
	previous := filepath.Join(base, "old", "app")
	_, err = initializeFreshProfileWithLstat(current, previous, durablefile.Replacer{}, func(string) (os.FileInfo, error) { return nil, errors.New("inspection failed") })
	if err == nil {
		t.Fatal("inspection failure accepted")
	}
	if _, err := os.Lstat(filepath.Dir(current)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unexpected mutation: %v", err)
	}
}

func TestFreshProfileExistingPathsRefused(t *testing.T) {
	for _, name := range []string{"current directory", "current file", "current symlink", "previous directory"} {
		t.Run(name, func(t *testing.T) {
			base, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			current, previous := filepath.Join(base, "current"), filepath.Join(base, "previous")
			switch name {
			case "current directory":
				err = os.Mkdir(current, 0700)
			case "current file":
				err = os.WriteFile(current, []byte("unchanged"), 0600)
			case "current symlink":
				err = os.Symlink(filepath.Join(base, "missing"), current)
			case "previous directory":
				err = os.Mkdir(previous, 0700)
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, err := initializeFreshProfile(current, previous, durablefile.Replacer{}); err == nil {
				t.Fatal("existing profile accepted")
			}
			if name == "current file" {
				data, err := os.ReadFile(current)
				if err != nil || string(data) != "unchanged" {
					t.Fatalf("existing file changed: %q %v", data, err)
				}
			}
		})
	}
}

func TestFreshProfileSharedOrgParent(t *testing.T) {
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	org := filepath.Join(base, "Nvidia Corporation")
	if err := os.Mkdir(org, 0700); err != nil {
		t.Fatal(err)
	}
	current := filepath.Join(org, "Personal AI Router")
	previous := filepath.Join(base, "NVIDIA Corporation", "PAIR")
	aliasLstat := func(path string) (os.FileInfo, error) {
		if path == previous {
			path = filepath.Join(org, "PAIR")
		}
		return os.Lstat(path)
	}
	if _, err := initializeFreshProfileWithLstat(current, previous, durablefile.Replacer{}, aliasLstat); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(org, "PAIR"), 0700); err != nil {
		t.Fatal(err)
	}
	other := filepath.Join(org, "another-app")
	if _, err := initializeFreshProfileWithLstat(other, previous, durablefile.Replacer{}, aliasLstat); err == nil {
		t.Fatal("previous profile under shared org accepted")
	}
	if _, err := os.Lstat(other); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unexpected allocation: %v", err)
	}
}

func TestFreshProfilePublicationFailuresPreserveAllocation(t *testing.T) {
	for _, landed := range []bool{false, true} {
		t.Run(fmt.Sprintf("landed_%t", landed), func(t *testing.T) {
			base, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			current, previous := filepath.Join(base, "current"), filepath.Join(base, "previous")
			r := durablefile.Replacer{Replace: func(tmp, target string) error {
				if landed {
					if err := os.Rename(tmp, target); err != nil {
						return err
					}
					return durablefile.ErrReplaced
				}
				return errors.New("before replacement")
			}}
			if _, err := initializeFreshProfile(current, previous, r); err == nil {
				t.Fatal("failed publication accepted")
			}
			_, err = os.Stat(filepath.Join(current, "engine-bin", modelDownloadFile))
			if landed && err != nil || !landed && !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("journal state: %v", err)
			}
			if _, err := initializeFreshProfile(current, previous, durablefile.Replacer{}); err == nil {
				t.Fatal("retry accepted")
			}
		})
	}
}

func freshTestBase(t *testing.T) string {
	t.Helper()
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return base
}

func freshTree(t *testing.T, root string) map[string]string {
	t.Helper()
	result := map[string]string{}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			target, err := os.Readlink(path)
			if err != nil {
				return err
			}
			result[rel] = "link:" + target
		} else if entry.IsDir() {
			result[rel] = "directory"
		} else {
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			result[rel] = string(data)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestFreshProfileExistingHistoryPreserved(t *testing.T) {
	for _, shape := range []string{"populated", "deleted journal", "deleted child", "malformed", "obsolete", "previous file", "previous symlink"} {
		t.Run(shape, func(t *testing.T) {
			base := freshTestBase(t)
			current, previous := filepath.Join(base, "current"), filepath.Join(base, "previous")
			if strings.HasPrefix(shape, "previous") {
				var err error
				if shape == "previous file" {
					err = os.WriteFile(previous, []byte("history"), 0600)
				} else {
					err = os.Symlink(filepath.Join(base, "missing"), previous)
				}
				if err != nil {
					t.Fatal(err)
				}
			} else {
				install := filepath.Join(current, "engine-bin")
				if err := os.MkdirAll(install, 0700); err != nil {
					t.Fatal(err)
				}
				data := []byte("historical contents")
				if shape == "obsolete" {
					data = []byte(`{"version":1,"entries":[],"copies":[],"copiesComplete":true}`)
				}
				if err := os.WriteFile(filepath.Join(install, modelDownloadFile), data, 0600); err != nil {
					t.Fatal(err)
				}
				if shape == "deleted journal" {
					if err := os.Remove(filepath.Join(install, modelDownloadFile)); err != nil {
						t.Fatal(err)
					}
				}
				if shape == "deleted child" {
					if err := os.RemoveAll(install); err != nil {
						t.Fatal(err)
					}
				}
			}
			before := freshTree(t, base)
			if _, err := initializeFreshProfile(current, previous, durablefile.Replacer{}); err == nil {
				t.Fatal("accepted existing history")
			}
			if after := freshTree(t, base); !reflect.DeepEqual(before, after) {
				t.Fatalf("changed existing state: before=%v after=%v", before, after)
			}
		})
	}
}

func TestFreshProfileAncestorInspection(t *testing.T) {
	for _, side := range []string{"current", "previous"} {
		for _, shape := range []string{"file", "symlink", "nested symlink", "inspection error"} {
			t.Run(side+" "+shape, func(t *testing.T) {
				base := freshTestBase(t)
				ancestor := filepath.Join(base, "ancestor")
				current, previous := filepath.Join(base, "current"), filepath.Join(base, "previous")
				if shape == "file" {
					if err := os.WriteFile(ancestor, []byte("preserve"), 0600); err != nil {
						t.Fatal(err)
					}
				} else {
					real := filepath.Join(base, "real")
					if err := os.MkdirAll(filepath.Join(real, "existing"), 0700); err != nil {
						t.Fatal(err)
					}
					if err := os.Symlink(real, ancestor); err != nil {
						t.Fatal(err)
					}
				}
				target := filepath.Join(ancestor, "app")
				if shape == "nested symlink" {
					target = filepath.Join(ancestor, "existing", "app")
				}
				if side == "current" {
					current = target
				} else {
					previous = target
				}
				before := freshTree(t, base)
				inspect := func(path string) (os.FileInfo, error) {
					if shape == "inspection error" && path == ancestor {
						return nil, os.ErrPermission
					}
					return os.Lstat(path)
				}
				if _, err := initializeFreshProfileWithLstat(current, previous, durablefile.Replacer{}, inspect); err == nil {
					t.Fatal("accepted invalid ancestor")
				}
				if after := freshTree(t, base); !reflect.DeepEqual(before, after) {
					t.Fatalf("mutated state: %v", after)
				}
			})
		}
	}
}

func TestFreshProfileAllocationFailures(t *testing.T) {
	for _, boundary := range []string{"one", "org", "app", "engine-bin"} {
		t.Run(boundary, func(t *testing.T) {
			base := freshTestBase(t)
			current, previous := filepath.Join(base, "one", "org", "app"), filepath.Join(base, "previous")
			var created []string
			mkdir := func(path string, mode os.FileMode) error {
				if filepath.Base(path) == boundary {
					return os.ErrPermission
				}
				if err := os.Mkdir(path, mode); err != nil {
					return err
				}
				created = append(created, path)
				return nil
			}
			if _, err := initializeFreshProfileWithFS(current, previous, durablefile.Replacer{}, os.Lstat, mkdir, readFreshProfile); err == nil {
				t.Fatal("accepted allocation failure")
			}
			for _, path := range created {
				if info, err := os.Lstat(path); err != nil || !info.IsDir() {
					t.Fatalf("allocation removed: %s %v", path, err)
				}
			}
			if _, err := os.Lstat(filepath.Join(current, "engine-bin", modelDownloadFile)); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("published after allocation failure: %v", err)
			}
		})
	}
}

func TestFreshProfileReadbackFailures(t *testing.T) {
	for _, shape := range []string{"read error", "malformed", "old", "incomplete", "recovery"} {
		t.Run(shape, func(t *testing.T) {
			base := freshTestBase(t)
			current, previous := filepath.Join(base, "app"), filepath.Join(base, "previous")
			read := func(store modelDownloadStore) error {
				if shape == "read error" {
					return os.ErrPermission
				}
				data := `{"version":1,"entries":[],"copies":[],"copiesComplete":true,"copyRecoveryRequired":false}`
				switch shape {
				case "malformed":
					data = "{"
				case "old":
					data = `{"version":1,"entries":[],"copies":[],"copiesComplete":true}`
				case "incomplete":
					data = strings.Replace(data, `"copiesComplete":true`, `"copiesComplete":false`, 1)
				case "recovery":
					data = strings.Replace(data, `"copyRecoveryRequired":false`, `"copyRecoveryRequired":true`, 1)
				}
				if err := os.WriteFile(store.path, []byte(data), 0600); err != nil {
					t.Fatal(err)
				}
				return readFreshProfile(store)
			}
			if _, err := initializeFreshProfileWithFS(current, previous, durablefile.Replacer{}, os.Lstat, os.Mkdir, read); err == nil {
				t.Fatal("accepted failed readback")
			}
			before := freshTree(t, base)
			if _, err := initializeFreshProfile(current, previous, durablefile.Replacer{}); err == nil {
				t.Fatal("retry accepted")
			}
			if !reflect.DeepEqual(before, freshTree(t, base)) {
				t.Fatal("retry mutated state")
			}
		})
	}
}

func TestFreshProfileExistingAncestorConfirmationChains(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix directory confirmation; Windows unsupported capability tested separately")
	}
	for _, existing := range []string{"base", "org"} {
		t.Run(existing, func(t *testing.T) {
			boundaries := 3
			if existing == "org" {
				boundaries = 2
			}
			for failure := -1; failure < boundaries; failure++ {
				t.Run(fmt.Sprintf("confirmation %d", failure), func(t *testing.T) {
					base := freshTestBase(t)
					org := filepath.Join(base, "org")
					if existing == "org" {
						if err := os.Mkdir(org, 0700); err != nil {
							t.Fatal(err)
						}
					}
					current, previous := filepath.Join(org, "app"), filepath.Join(base, "previous")
					want := []string{filepath.Join(current, "engine-bin"), current, org}
					if existing == "base" {
						want = append(want, base)
					}
					var calls []string
					r := durablefile.Replacer{SyncDir: func(path string) error {
						calls = append(calls, path)
						if failure >= 0 && len(calls) == failure+2 {
							return os.ErrPermission
						}
						return nil
					}}
					_, err := initializeFreshProfile(current, previous, r)
					if (err != nil) != (failure >= 0) {
						t.Fatalf("error=%v", err)
					}
					if failure >= 0 {
						want = want[:failure+2]
					}
					if !reflect.DeepEqual(calls, want) {
						t.Fatalf("calls=%v want=%v", calls, want)
					}
					if err := readFreshProfile(newModelDownloadStore(filepath.Join(current, "engine-bin"))); err != nil {
						t.Fatal(err)
					}
					before := freshTree(t, base)
					if _, err := initializeFreshProfile(current, previous, durablefile.Replacer{}); err == nil {
						t.Fatal("retry accepted")
					}
					if !reflect.DeepEqual(before, freshTree(t, base)) {
						t.Fatal("retry changed state")
					}
				})
			}
		})
	}
}

func TestFreshProfileMatchingCopyCompletion(t *testing.T) {
	base := freshTestBase(t)
	current := filepath.Join(base, "app")
	if _, err := initializeFreshProfile(current, filepath.Join(base, "previous"), durablefile.Replacer{}); err != nil {
		t.Fatal(err)
	}
	d := newModelDownloads(newOpID(), filepath.Join(current, "engine-bin"), map[string]string{"llamacpp": engines.ModelNamingExact})
	reg := &Registry{engines: map[string]*Manifest{"llamacpp": {Engine: "llamacpp", ModelNaming: engines.ModelNamingExact}}}
	m := &Manager{exec: &Executor{switchIncarnation: d.source}, copyJournal: d, copyRegistry: reg}
	old := modelCopyMarker{OpID: newOpID(), Engine: "llamacpp", Model: "old", ContentKey: "old", Naming: engines.ModelNamingExact, Role: "controller", OwnerNodeID: "controller", SourceNodeID: "source", DestinationNodeID: "destination", ManagerIncarnation: newOpID(), Phase: "unsafe"}
	if landed, uncertain, err := d.store.save(nil, map[string]modelCopyMarker{old.OpID: old}, true, false); err != nil || !landed || uncertain {
		t.Fatalf("seed predecessor: %v", err)
	}
	// Reload preserves predecessor provenance while allowing ordinary new retention.
	d = newModelDownloads(d.source, filepath.Join(current, "engine-bin"), map[string]string{"llamacpp": engines.ModelNamingExact})
	m.copyJournal = d
	p := copyRequest{OpID: newOpID(), Engine: "llamacpp", Model: "model", SourceNodeID: "source", DestinationNodeID: "destination"}
	ctx, release, err := m.reserveCopy(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	called := false
	_, err = m.copyWithEvidence(ctx, p, "destination", func() (copyModelResult, error) {
		called = true
		_, copies, complete, recovery, err := d.store.loadJournal()
		if err != nil || len(copies) != 2 || !complete || recovery || copies[p.OpID].ManagerIncarnation != d.source || copies[old.OpID] != old {
			t.Fatalf("before effect: %v %v", copies, err)
		}
		return copyModelResult{Engine: p.Engine, Model: p.Model, Files: 1}, nil
	})
	if err != nil || !called {
		t.Fatalf("completion: called=%t err=%v", called, err)
	}
	_, copies, complete, recovery, err := d.store.loadJournal()
	if err != nil || len(copies) != 1 || copies[old.OpID] != old || !complete || recovery {
		t.Fatalf("retirement changed predecessor: %v %v", copies, err)
	}
	if err := d.changeCopyMarker(old, true); err == nil {
		t.Fatal("predecessor retired by successor")
	}
}

func TestFreshProfileDownloadBeforeEffect(t *testing.T) {
	base := freshTestBase(t)
	current := filepath.Join(base, "app")
	if _, err := initializeFreshProfile(current, filepath.Join(base, "previous"), durablefile.Replacer{}); err != nil {
		t.Fatal(err)
	}
	d := newModelDownloads(newOpID(), filepath.Join(current, "engine-bin"), map[string]string{"llamacpp": engines.ModelNamingExact})
	a, err := d.reserve("llamacpp", "model", engines.ModelNamingExact)
	if err != nil {
		t.Fatal(err)
	}
	e := &Executor{downloads: d, progress: newProgressHub()}
	token := &modelPullReservation{exec: e, attempt: a, engine: "llamacpp", model: "model", id: a.entry.ID}
	ctx := context.WithValue(context.Background(), modelPullContextKey{}, token)
	if err := e.beforeModelPullEffect(ctx); err != nil {
		t.Fatal(err)
	}
	// Synchronous fake effect entry reads the actual strict journal.
	markers, copies, complete, recovery, err := d.store.loadJournal()
	key := downloadContentKey{engine: "llamacpp", model: "model"}
	if err != nil || len(markers) != 1 || markers[key].ID != a.entry.ID || len(copies) != 0 || !complete || recovery || !token.effects {
		t.Fatalf("before download effect: %v %v", markers, err)
	}
	if uncertain, err := d.changeMarker(a, true); err != nil || uncertain {
		t.Fatalf("matching completion: %v", err)
	}
	markers, copies, complete, recovery, err = d.store.loadJournal()
	if err != nil || len(markers) != 0 || len(copies) != 0 || !complete || recovery {
		t.Fatalf("completion flags: %v", err)
	}
}

func TestFreshProfileInvalidPaths(t *testing.T) {
	base := freshTestBase(t)
	for _, paths := range [][2]string{{"", filepath.Join(base, "old")}, {"relative", filepath.Join(base, "old")}, {filepath.Join(base, "app"), ""}, {base, base}, {string(filepath.Separator), filepath.Join(base, "old")}} {
		if _, err := initializeFreshProfile(paths[0], paths[1], durablefile.Replacer{}); err == nil {
			t.Fatalf("accepted %q", paths)
		}
	}
	if len(freshTree(t, base)) != 1 {
		t.Fatal("invalid paths mutated base")
	}
}

func TestFreshProfileWindowsLocalBaseCLI(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows Local-base dispatch; pending PC3")
	}
	for _, shape := range []string{"missing", "empty", "relative", "current", "previous"} {
		t.Run(shape, func(t *testing.T) {
			cfg, home := freshTestBase(t), freshTestBase(t)
			if shape == "current" || shape == "previous" {
				path := filepath.Join(cfg, "Nvidia Corporation", "Personal AI Router")
				if shape == "previous" {
					path = filepath.Join(cfg, "NVIDIA Corporation", "PAIR")
				}
				if err := os.MkdirAll(path, 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(path, "history"), []byte("preserve"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			beforeCfg, beforeHome := freshTree(t, cfg), freshTree(t, home)
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, managerBin, "--initialize-fresh-profile")
			cmd.Env = isolatedManagerEnv(cfg, home)
			if shape == "missing" || shape == "empty" || shape == "relative" {
				var env []string
				for _, item := range cmd.Env {
					if !strings.HasPrefix(strings.ToUpper(item), "LOCALAPPDATA=") {
						env = append(env, item)
					}
				}
				if shape == "empty" {
					env = append(env, "LOCALAPPDATA=")
				}
				if shape == "relative" {
					env = append(env, "LOCALAPPDATA=relative")
				}
				cmd.Env = env
			}
			out, err := cmd.CombinedOutput()
			if err == nil || ctx.Err() != nil || strings.Contains(string(out), "ready") {
				t.Fatalf("expected clean refusal: %v %s", err, out)
			}
			if !reflect.DeepEqual(beforeCfg, freshTree(t, cfg)) || !reflect.DeepEqual(beforeHome, freshTree(t, home)) {
				t.Fatal("refusal mutated private roots")
			}
		})
	}
}

func TestFreshProfileDesktopOrgAliasing(t *testing.T) {
	for _, existing := range []string{"absent apps", "current app", "previous app"} {
		t.Run(existing, func(t *testing.T) {
			base := freshTestBase(t)
			org := filepath.Join(base, "Nvidia Corporation")
			oldOrg := filepath.Join(base, "NVIDIA Corporation")
			if err := os.Mkdir(org, 0700); err != nil {
				t.Fatal(err)
			}
			current, previous := filepath.Join(org, "Personal AI Router"), filepath.Join(oldOrg, "PAIR")
			if existing != "absent apps" {
				path := current
				if existing == "previous app" {
					path = filepath.Join(org, "PAIR")
				}
				if err := os.Mkdir(path, 0700); err != nil {
					t.Fatal(err)
				}
			}
			inspect := os.Lstat
			if _, err := os.Lstat(oldOrg); errors.Is(err, os.ErrNotExist) {
				t.Log("case-sensitive host: injected org aliasing policy; native aliasing UNVERIFIED")
				inspect = func(path string) (os.FileInfo, error) {
					if path == oldOrg || strings.HasPrefix(path, oldOrg+string(filepath.Separator)) {
						path = org + strings.TrimPrefix(path, oldOrg)
					}
					return os.Lstat(path)
				}
			} else if err != nil {
				t.Fatal(err)
			} else {
				t.Log("native case-insensitive org lookup exercised")
			}
			before := freshTree(t, base)
			_, err := initializeFreshProfileWithLstat(current, previous, durablefile.Replacer{}, inspect)
			if existing == "absent apps" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil {
				t.Fatal("existing desktop app accepted")
			}
			if !reflect.DeepEqual(before, freshTree(t, base)) {
				t.Fatal("existing desktop app changed")
			}
		})
	}
}

func TestFreshProfileWindowsDirectoryFailure(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows supported-operation error classification; pending PC3")
	}
	base := freshTestBase(t)
	current, previous := filepath.Join(base, "app"), filepath.Join(base, "previous")
	calls := 0
	r := durablefile.Replacer{SyncDir: func(string) error {
		calls++
		if calls == 2 {
			return os.ErrPermission
		}
		return nil
	}}
	if unsupported, err := initializeFreshProfile(current, previous, r); err == nil || unsupported {
		t.Fatalf("failed operation became unsupported success: %t %v", unsupported, err)
	}
	if err := readFreshProfile(newModelDownloadStore(filepath.Join(current, "engine-bin"))); err != nil {
		t.Fatal(err)
	}
	before := freshTree(t, base)
	if _, err := initializeFreshProfile(current, previous, durablefile.Replacer{}); err == nil {
		t.Fatal("retry accepted")
	}
	if !reflect.DeepEqual(before, freshTree(t, base)) {
		t.Fatal("retry changed state")
	}
}
