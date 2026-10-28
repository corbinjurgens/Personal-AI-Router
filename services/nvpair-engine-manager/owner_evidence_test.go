// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/buildapply"
	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
)

func TestOwnerEvidenceUniverseMembership(t *testing.T) {
	e := &Executor{reg: &Registry{engines: map[string]*Manifest{"registered": {}}}, engines: map[string]*engineState{}, stopOnly: map[string]*engineState{"cleanup": {}}}
	saved := desiredOwnerEvidence{State: desiredStateFile{Engines: map[string]bool{"retained": false}, AllocatedPorts: map[string]int{"port": 0}}}
	got, ok := e.ownerUniverseMembership(saved)
	want := []ownerUniverseEntry{{Name: "cleanup", Cleanup: true}, {Name: "port", Cleanup: true}, {Name: "registered", Registered: true}, {Name: "retained"}}
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("membership = %+v, %v; want %+v", got, ok, want)
	}
	got[0].Name = "changed"
	again, ok := e.ownerUniverseMembership(saved)
	if !ok || !reflect.DeepEqual(again, want) || len(e.engines) != 0 {
		t.Fatalf("membership changed owner state: %+v, %v", again, ok)
	}
	e.mu.Lock()
	_, ok = e.ownerUniverseMembership(saved)
	e.mu.Unlock()
	if ok {
		t.Fatal("contended membership must be unknown")
	}
}

// No constructor, logs, content gate, watcher, engine, file or native owner is
// initialized by these fixtures. The saved reader is the Task1 detached seam.
func ownerEvidenceFixture(t *testing.T) (*Executor, *engineState, desiredOwnerEvidence) {
	t.Helper()
	root := filepath.Join(os.TempDir(), "pair-owner-evidence-inert")
	platform := Platform{ModelsDir: filepath.Join(root, "models"), Runtime: Runtime{Port: 0}}
	manifest := &Manifest{Engine: "fake", Platforms: map[string]Platform{runtime.GOOS + "/" + runtime.GOARCH: platform}}
	st := &engineState{manifest: manifest, plat: &platform, installDir: filepath.Join(root, "fake"), modelsDir: platform.ModelsDir}
	e := &Executor{switchIncarnation: "manager-fixture", reg: &Registry{engines: map[string]*Manifest{"fake": manifest}}, engines: map[string]*engineState{"fake": st}, stopOnly: map[string]*engineState{},
		desired:   newDesiredStateStore(t.TempDir()),
		processes: func(context.Context) ([]string, error) { panic("evidence invoked process lookup") },
		client:    &http.Client{Transport: ownerEvidenceRejectHTTP{}}, emit: func(string, any) { panic("evidence notified") },
		inventory: map[string][]string{"fake": {"must-not-be-read"}},
	}
	return e, st, desiredOwnerEvidence{State: desiredStateFile{Engines: map[string]bool{}, OffBy: map[string]intentOrigin{}, Locked: map[string]bool{}}}
}

type ownerEvidenceRejectHTTP struct{}

func (ownerEvidenceRejectHTTP) RoundTrip(*http.Request) (*http.Response, error) {
	panic("evidence contacted a native engine")
}

func readOwnerFixture(e *Executor, saved desiredOwnerEvidence) ownerUniverseEvidence {
	return e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) { return saved, nil }, func() ([]string, bool) { return nil, true })
}

func ownerFixtureEntry(t *testing.T, got ownerUniverseEvidence, name string) ownerEngineEvidence {
	t.Helper()
	if !got.Complete || got.Incarnation != "manager-fixture" {
		t.Fatalf("incomplete fixture evidence: %+v", got)
	}
	for _, entry := range got.Entries {
		if entry.Name == name {
			return entry
		}
	}
	t.Fatalf("engine %q missing from %+v", name, got)
	return ownerEngineEvidence{}
}

func requireOwnerUnknown(t *testing.T, observation buildapply.Observation[bool]) {
	t.Helper()
	if observation.Known || observation.Value != nil {
		t.Fatalf("wanted unknown, got %+v", observation)
	}
}

func requireOwnerBool(t *testing.T, observation buildapply.Observation[bool], want bool) {
	t.Helper()
	if !observation.Known || observation.Value == nil || *observation.Value != want {
		t.Fatalf("observation = %+v, want known %v", observation, want)
	}
}

func TestOwnerEvidenceUniverse(t *testing.T) {
	t.Run("full union retains unsupported saved and reservation names", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		e.reg.engines["unsupported"] = &Manifest{}
		e.reg.engines["cold"] = e.reg.engines["fake"]
		e.stopOnly["removed-host"] = &engineState{plat: &Platform{}}
		saved.State.Engines["saved-off"] = false
		saved.State.OffBy["origin-only"] = intentPeer
		saved.State.Locked["saved-unlocked"] = false
		saved.State.ProcessStops = map[string]processStopEvidence{"unsupported": {PID: 12}, "unregistered-cleanup": {PID: 13}}
		saved.State.AllocatedPorts = map[string]int{"port-zero": 0}
		saved.State.PendingSwitch = &settings.SwitchCheckpoint{SwitchPrepareRequest: settings.SwitchPrepareRequest{SourceEngine: "checkpoint-source", TargetEngine: "checkpoint-target"}, Phase: settings.SwitchCheckpointReleased}
		e.downloads = &modelDownloads{active: map[downloadContentKey]*modelDownloadAttempt{{engine: "download"}: {}}, unsafe: map[downloadContentKey]modelDownloadMarker{{engine: "unsafe"}: {}}, writers: map[string]bool{"writer": true}, readers: map[string]int{"reader": 1}}
		got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) { return saved, nil }, func() ([]string, bool) { return []string{"copy"}, true })
		want := []string{"checkpoint-source", "checkpoint-target", "cold", "copy", "download", "fake", "origin-only", "port-zero", "reader", "removed-host", "saved-off", "saved-unlocked", "unregistered-cleanup", "unsafe", "unsupported", "writer"}
		var names []string
		for _, entry := range got.Entries {
			names = append(names, entry.Name)
			requireOwnerUnknown(t, entry.Running)
			requireOwnerUnknown(t, entry.PositiveStop)
			requireOwnerUnknown(t, entry.StopAuthority)
			if entry.RootsComplete || entry.RootsReason != "unmodeled_data_roots" {
				t.Fatalf("invented root coverage: %+v", entry)
			}
		}
		if !got.Complete || !reflect.DeepEqual(names, want) {
			t.Fatalf("universe = %+v; names want %v", got, want)
		}
		unsupported := ownerFixtureEntry(t, got, "unsupported")
		if !unsupported.Registered || unsupported.Supported || !unsupported.Cleanup {
			t.Fatalf("unsupported cleanup lost: %+v", unsupported)
		}
		if entry := ownerFixtureEntry(t, got, "unregistered-cleanup"); entry.Registered || !entry.Cleanup {
			t.Fatalf("retained cleanup lost: %+v", entry)
		}
		if len(e.engines) != 1 || st.logs != nil || st.contentMu != nil || st.relaunch != nil || st.healthStop != nil {
			t.Fatal("evidence initialized runtime owners")
		}
		if !reflect.DeepEqual(e.inventory, map[string][]string{"fake": {"must-not-be-read"}}) {
			t.Fatal("evidence touched model inventory")
		}
	})

	t.Run("installed and stopped publications are detached", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		st.installed = true
		st.observeOwnerInstalledLocked()
		st.observeOwnerPresenceLocked(true)
		got := readOwnerFixture(e, saved)
		entry := ownerFixtureEntry(t, got, "fake")
		requireOwnerBool(t, entry.Installed, true)
		requireOwnerBool(t, entry.Running, false)
		requireOwnerBool(t, entry.PositiveStop, true)
		requireOwnerBool(t, entry.StopAuthority, true)
		if entry.AuthorityReason != "no_live_process" || entry.InstallRoot != st.installDir || entry.ModelRoot != st.modelsDir {
			t.Fatalf("stopped roots/authority = %+v", entry)
		}
		*entry.Installed.Value = false
		got.Entries[0].ModelRoot = "changed"
		again := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerBool(t, again.Installed, true)
		if again.ModelRoot != st.modelsDir {
			t.Fatal("response aliases roots")
		}
	})

	t.Run("configuration change invalidates old observations", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		st.installed = true
		st.observeOwnerInstalledLocked()
		st.observeOwnerPresenceLocked(true)
		st.modelsDir = filepath.Join(st.installDir, "new-models")
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerUnknown(t, entry.Installed)
		requireOwnerUnknown(t, entry.StopAuthority)
		if entry.ModelRoot != st.modelsDir {
			t.Fatal("live configuration roots not captured")
		}
	})

	t.Run("unresolved roots never become known empty coverage", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		st.modelsDir = "${UNRESOLVED}/models"
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		if entry.ModelRoot != "" || entry.RootsComplete || entry.RootsReason == "" {
			t.Fatalf("unresolved root = %+v", entry)
		}
	})

	t.Run("owner contention returns without waiting", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		check := func(t *testing.T) {
			t.Helper()
			if got := readOwnerFixture(e, saved); got.Complete || got.Reason != "busy" {
				t.Fatalf("contended capture = %+v", got)
			}
		}
		e.mu.Lock()
		check(t)
		e.mu.Unlock()
		st.mu.Lock()
		check(t)
		st.mu.Unlock()
		e.switchAdmission.Lock()
		check(t)
		e.switchAdmission.Unlock()
		e.downloads = &modelDownloads{}
		e.downloads.mu.Lock()
		check(t)
		e.downloads.mu.Unlock()
		e.desired = &desiredStateStore{}
		e.desired.mu.Lock()
		if _, err := e.desired.tryOwnerEvidence(); err == nil {
			t.Fatal("contended desired state reported known")
		}
		e.desired.mu.Unlock()
		manager := &Manager{exec: e}
		manager.copyMu.Lock()
		if _, ok := manager.ownerCopyNames(); ok {
			t.Fatal("contended copy owner reported known")
		}
		manager.copyMu.Unlock()
	})

	t.Run("copy owner contributes names without reading progress or native state", func(t *testing.T) {
		e, _, _ := ownerEvidenceFixture(t)
		manager := &Manager{exec: e, copies: map[string]*activeCopy{"one": {copyRequest: copyRequest{Engine: "retained-copy"}}}}
		names, ok := manager.ownerCopyNames()
		if !ok || !reflect.DeepEqual(names, []string{"retained-copy"}) {
			t.Fatalf("copy names = %v, %v", names, ok)
		}
		names[0] = "changed"
		if manager.copies["one"].Engine != "retained-copy" {
			t.Fatal("copy projection aliases owner")
		}
	})

	t.Run("unavailable saved owner does not claim a complete union", func(t *testing.T) {
		e, _, _ := ownerEvidenceFixture(t)
		got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) { return desiredOwnerEvidence{}, errors.New("unavailable") }, func() ([]string, bool) { return nil, true })
		if got.Complete || got.Reason != "saved_state_unknown" || got.Incarnation != e.switchIncarnation {
			t.Fatalf("saved failure = %+v", got)
		}
	})

	t.Run("changes between captures invalidate the section", func(t *testing.T) {
		check := func(name string, change func(*Executor, *engineState, *desiredOwnerEvidence)) {
			t.Run(name, func(t *testing.T) {
				e, st, saved := ownerEvidenceFixture(t)
				reads := 0
				got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) {
					reads++
					if reads == 2 {
						change(e, st, &saved)
					}
					return saved, nil
				}, func() ([]string, bool) { return nil, true })
				if got.Complete || got.Reason != "busy" {
					t.Fatalf("changed owner accepted: %+v", got)
				}
			})
		}
		check("roots", func(_ *Executor, st *engineState, _ *desiredOwnerEvidence) { st.modelsDir += "-replacement" })
		check("state membership", func(e *Executor, _ *engineState, _ *desiredOwnerEvidence) {
			e.engines["late"] = &engineState{plat: &Platform{}}
		})
		check("run", func(_ *Executor, st *engineState, _ *desiredOwnerEvidence) { st.runID = "next-run" })
		check("cached state", func(_ *Executor, st *engineState, _ *desiredOwnerEvidence) { st.installed = true })
		check("saved intent", func(_ *Executor, _ *engineState, saved *desiredOwnerEvidence) {
			saved.State = desiredStateFile{Engines: map[string]bool{"late": true}}
		})
		check("checkpoint", func(_ *Executor, _ *engineState, saved *desiredOwnerEvidence) {
			saved.State.PendingSwitch = &settings.SwitchCheckpoint{Phase: settings.SwitchCheckpointReserved}
		})
	})
}

func TestOwnerEvidenceStopAuthority(t *testing.T) {
	owned := func(t *testing.T) (*Executor, *engineState, desiredOwnerEvidence) {
		t.Helper()
		e, st, saved := ownerEvidenceFixture(t)
		st.proc = &managedProc{cmd: &exec.Cmd{Process: &os.Process{Pid: 4321}}, done: make(chan struct{})}
		st.runID, st.running, st.installed = "run-one", true, true
		st.observeOwnerInstalledLocked()
		st.observeOwnerPresenceLocked(false)
		return e, st, saved
	}
	t.Run("owned process retains scoped authority without OS proof", func(t *testing.T) {
		e, _, saved := owned(t)
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerBool(t, entry.StopAuthority, true)
		requireOwnerBool(t, entry.Running, true)
		requireOwnerBool(t, entry.PositiveStop, false)
		if entry.AuthorityReason != "owned_process_lifetime" {
			t.Fatalf("authority = %+v", entry)
		}
	})
	unknown := func(name, reason string, change func(*engineState)) {
		t.Run(name, func(t *testing.T) {
			e, st, saved := owned(t)
			change(st)
			entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
			requireOwnerUnknown(t, entry.StopAuthority)
			if entry.AuthorityReason != reason {
				t.Fatalf("authority = %+v; want %s", entry, reason)
			}
		})
	}
	unknown("closed process lifetime", "process_lifetime_closed", func(st *engineState) { close(st.proc.done) })
	unknown("process without lifetime", "process_lifetime_closed", func(st *engineState) { st.proc.done = nil })
	unknown("same PID replacement process", "unobserved", func(st *engineState) {
		st.proc = &managedProc{cmd: &exec.Cmd{Process: &os.Process{Pid: 4321}}, done: make(chan struct{})}
	})
	unknown("same PID replacement run", "unobserved", func(st *engineState) { st.runID = "run-two" })
	unknown("empty run", "unobserved", func(st *engineState) { st.runID = "" })
	unknown("reclaim failure", "reclaim_unknown", func(st *engineState) { st.reclaimFailed = true })
	unknown("PID only locator", "reclaim_unknown", func(st *engineState) { st.reclaimPID = 4321 })
	unknown("command without scoped lifetime", "scoped_lifetime_unknown", func(st *engineState) {
		st.proc = nil
		st.plat.Runtime.Mode = "command"
		st.observeOwnerPresenceLocked(false)
	})
	unknown("command endpoint stopped without physical process proof", "scoped_lifetime_unknown", func(st *engineState) {
		st.proc, st.running, st.runID = nil, false, ""
		st.plat.Runtime.Mode = "command"
		st.observeOwnerPresenceLocked(false)
	})
	t.Run("adopted command is known false even with Stop configured", func(t *testing.T) {
		e, st, saved := ownerEvidenceFixture(t)
		st.plat.Runtime.Mode, st.plat.Runtime.Stop = "command", &StopSpec{Cmd: []string{"must-not-run"}}
		st.running, st.adopted, st.runID = true, true, "external-run"
		st.observeOwnerPresenceLocked(false)
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerBool(t, entry.StopAuthority, false)
		if entry.AuthorityReason != "external_management" {
			t.Fatalf("adopted authority = %+v", entry)
		}
	})
	t.Run("default false is never positive stopped evidence", func(t *testing.T) {
		e, _, saved := ownerEvidenceFixture(t)
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerUnknown(t, entry.Running)
		requireOwnerUnknown(t, entry.PositiveStop)
		requireOwnerUnknown(t, entry.StopAuthority)
	})
	t.Run("retirement does not republish withdrawn observations", func(t *testing.T) {
		e, st, saved := owned(t)
		done, err := e.admitMutation("fake")
		if err != nil {
			t.Fatalf("mutation admission: %v", err)
		}
		done()
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerUnknown(t, entry.Installed)
		requireOwnerUnknown(t, entry.Running)
		requireOwnerUnknown(t, entry.PositiveStop)
		requireOwnerUnknown(t, entry.StopAuthority)
		if !st.running || st.proc == nil || st.runID != "run-one" {
			t.Fatal("withdrawal changed lifecycle state")
		}
	})

	// Hold the actual admission owner while opMu and cached state stay unchanged.
	// Channels delimit the interval; evidence cannot rely on a lucky scheduling
	// delay or on seeing a native operation's intermediate state.
	admitted := func(name string, enter func(*Executor) (func(), error)) {
		t.Run(name, func(t *testing.T) {
			e, st, saved := owned(t)
			entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var entryErr error
			go func() {
				defer close(finished)
				done, err := enter(e)
				entryErr = err
				close(entered)
				if err != nil {
					return
				}
				<-release
				done()
			}()
			<-entered
			defer func() { close(release); <-finished }()
			if entryErr != nil {
				t.Fatalf("admission: %v", entryErr)
			}
			// Republish the exact old cache while still admitted. It stays busy.
			st.mu.Lock()
			st.observeOwnerInstalledLocked()
			st.observeOwnerPresenceLocked(false)
			st.mu.Unlock()
			entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
			requireOwnerUnknown(t, entry.Installed)
			requireOwnerUnknown(t, entry.Running)
			requireOwnerUnknown(t, entry.StopAuthority)
			if entry.AuthorityReason != "busy" || !st.running || st.runID != "run-one" {
				t.Fatalf("admitted interval = %+v", entry)
			}
		})
	}
	track := func(e *Executor) (func(), error) { return e.admitMutation("fake") }
	admitted("install admitted before native work", track)
	admitted("start waiting for lifecycle lock", track)
	admitted("ordinary stop admission", func(e *Executor) (func(), error) { return e.admitStop("fake", intentPeer, "") })
	admitted("global mutation", func(e *Executor) (func(), error) {
		e.switchAdmission.Lock()
		defer e.switchAdmission.Unlock()
		return e.trackMutationLocked("")
	})
	admitted("cold activation", func(e *Executor) (func(), error) {
		// An absent saved document is the normal cold admission baseline; no
		// directory/file is created by its existing read owner.
		return e.admitBoundEngineMutation("fake", "cold-operation")
	})
	admitted("checkpoint work", func(e *Executor) (func(), error) { return e.admitOwnerCheckpoint() })

	t.Run("unreleased checkpoint blocks unchanged lifecycle cache", func(t *testing.T) {
		e, _, saved := owned(t)
		saved.State.PendingSwitch = &settings.SwitchCheckpoint{SwitchPrepareRequest: settings.SwitchPrepareRequest{SourceEngine: "fake"}, Phase: settings.SwitchCheckpointReserved}
		entry := ownerFixtureEntry(t, readOwnerFixture(e, saved), "fake")
		requireOwnerUnknown(t, entry.StopAuthority)
		if entry.AuthorityReason != "busy" {
			t.Fatalf("checkpoint authority = %+v", entry)
		}
	})
	t.Run("admission enter exit ABA invalidates otherwise identical captures", func(t *testing.T) {
		e, _, saved := owned(t)
		reads := 0
		got := e.captureOwnerUniverse(func() (desiredOwnerEvidence, error) {
			reads++
			if reads == 2 {
				done, err := e.admitOwnerCheckpoint()
				if err != nil {
					t.Fatalf("checkpoint admission: %v", err)
				}
				done()
			}
			return saved, nil
		}, func() ([]string, bool) { return nil, true })
		if got.Complete || e.ownerAdmissionGeneration != 2 || e.ownerAdmissionHeld != 0 {
			t.Fatalf("ABA accepted: %+v, generation=%d held=%d", got, e.ownerAdmissionGeneration, e.ownerAdmissionHeld)
		}
	})
	t.Run("cold activation retires inside the same generation fence", func(t *testing.T) {
		e, _, _ := owned(t)
		done, err := e.admitBoundEngineMutation("fake", "cold-operation")
		if err != nil {
			t.Fatalf("cold admission: %v", err)
		}
		if e.boundActivation == nil || e.ownerAdmissionGeneration != 1 || e.ownerAdmissionHeld != 1 || e.switchMutations[""] != 1 {
			t.Fatal("cold entry lacks its reserved generation")
		}
		done()
		if e.boundActivation != nil || e.ownerAdmissionGeneration != 2 || e.ownerAdmissionHeld != 0 || e.switchMutations[""] != 0 || e.switchMutations["fake"] != 0 {
			t.Fatal("cold retirement escaped generation fence")
		}
	})
	t.Run("admission reserves retirement without wrap or duplicate retirement", func(t *testing.T) {
		e, _, _ := owned(t)
		e.ownerAdmissionGeneration = math.MaxUint64 - 3
		first, err := e.admitOwnerCheckpoint()
		if err != nil {
			t.Fatalf("first admission: %v", err)
		}
		if done, err := e.admitOwnerCheckpoint(); err == nil || done != nil {
			t.Fatal("accepted work without retirement capacity")
		}
		first()
		first()
		if e.ownerAdmissionGeneration != math.MaxUint64-1 || e.ownerAdmissionHeld != 0 {
			t.Fatalf("retirement = %d, held=%d", e.ownerAdmissionGeneration, e.ownerAdmissionHeld)
		}
		if done, err := e.admitBoundEngineMutation("fake", "cold"); err == nil || done != nil || e.boundActivation != nil {
			t.Fatal("cold activation accepted exhausted generation")
		}
		if done, err := e.admitStop("fake", intentLocal, ""); err == nil || done != nil || e.switchMutations["fake"] != 0 {
			t.Fatal("stop admitted effects without retirement capacity")
		}
	})
}

func TestOwnerEvidenceDesiredState(t *testing.T) {
	newStore := func(t *testing.T, document string) *desiredStateStore {
		t.Helper()
		store := newDesiredStateStore(t.TempDir())
		if document != "" {
			if err := os.WriteFile(store.path, []byte(document), 0o600); err != nil {
				t.Fatalf("write desired state fixture: %v", err)
			}
		}
		return store
	}
	read := func(t *testing.T, store *desiredStateStore) desiredOwnerEvidence {
		t.Helper()
		got, err := store.ownerEvidence()
		if err != nil {
			t.Fatalf("read owner evidence: %v", err)
		}
		if got.State.Engines == nil || got.State.OffBy == nil || got.State.Locked == nil {
			t.Fatalf("complete saved state lacks maps: %+v", got.State)
		}
		if len(got.IntentRevision) != len("sha256:")+64 || !strings.HasPrefix(got.IntentRevision, "sha256:") ||
			len(got.LockRevision) != len("sha256:")+64 || !strings.HasPrefix(got.LockRevision, "sha256:") {
			t.Fatalf("missing semantic revisions: %+v", got)
		}
		return got
	}
	encode := func(t *testing.T, state desiredStateFile) string {
		t.Helper()
		data, err := json.Marshal(state)
		if err != nil {
			t.Fatalf("encode desired state fixture: %v", err)
		}
		return string(data)
	}
	unknown := func(t *testing.T, store *desiredStateStore) {
		t.Helper()
		got, err := store.ownerEvidence()
		if err == nil {
			t.Fatalf("invalid or unavailable document reported known: %+v", got)
		}
		if !reflect.DeepEqual(got, desiredOwnerEvidence{}) {
			t.Fatalf("unknown document leaked a partial baseline: %+v", got)
		}
	}
	// Apply to accepted and rejected reads. Count attempted writes, including
	// writes that would otherwise leave no changed bytes.
	unchanged := func(t *testing.T, store *desiredStateStore, action func()) {
		t.Helper()
		before, statErr := os.Stat(store.path)
		if statErr != nil && !os.IsNotExist(statErr) {
			t.Fatalf("stat before read: %v", statErr)
		}
		var original []byte
		if statErr == nil {
			var err error
			original, err = os.ReadFile(store.path)
			if err != nil {
				t.Fatalf("read original bytes: %v", err)
			}
		}
		writes := 0
		useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) {
			writes++
			return nil, errors.New("evidence must not write")
		}})
		action()
		if writes != 0 {
			t.Fatalf("evidence attempted %d writes", writes)
		}
		after, err := os.Stat(store.path)
		if os.IsNotExist(statErr) {
			if !os.IsNotExist(err) {
				t.Fatalf("missing file changed: stat error %v", err)
			}
		} else {
			if err != nil {
				t.Fatalf("stat after read: %v", err)
			}
			data, err := os.ReadFile(store.path)
			if err != nil {
				t.Fatalf("read bytes after evidence: %v", err)
			}
			if !bytes.Equal(data, original) || !before.ModTime().Equal(after.ModTime()) || before.Mode() != after.Mode() {
				t.Fatalf("evidence changed saved document: before=%q after=%q", original, data)
			}
		}
		entries, err := os.ReadDir(filepath.Dir(store.path))
		if err != nil {
			t.Fatalf("read owner directory: %v", err)
		}
		for _, entry := range entries {
			if entry.Name() != desiredStateFileName {
				t.Fatalf("evidence created an extra file: %s", entry.Name())
			}
		}
	}

	t.Run("explicit intent and effective Off origins", func(t *testing.T) {
		store := newStore(t, `{
			"engines":{"local-off":false,"peer-off":false,"legacy-off":false,"on":true},
			"offBy":{"local-off":"local","peer-off":"peer"}
		}`)
		unchanged(t, store, func() {
			got := read(t, store)
			wantEngines := map[string]bool{"local-off": false, "peer-off": false, "legacy-off": false, "on": true}
			wantOrigins := map[string]intentOrigin{"local-off": intentLocal, "peer-off": intentPeer, "legacy-off": intentLocal}
			if !reflect.DeepEqual(got.State.Engines, wantEngines) || !reflect.DeepEqual(got.State.OffBy, wantOrigins) {
				t.Fatalf("saved intent = %+v, want engines=%v origins=%v", got.State, wantEngines, wantOrigins)
			}
			if _, present := got.State.Engines["unset"]; present {
				t.Fatal("unset intent became explicit Off")
			}
			if got.State.PendingSwitch != nil || got.State.PendingSwitchPorts != nil {
				t.Fatal("absent checkpoint became present")
			}
		})
	})
	t.Run("retains unregistered intent locks and cleanup keys", func(t *testing.T) {
		stop := processStopEvidence{Bin: filepath.Join(t.TempDir(), "inert-image"), Port: 23456, Address: "127.0.0.1:23456", PID: 123}
		state := desiredStateFile{
			Engines: map[string]bool{"removed-on": true, "removed-locked": false}, OffBy: map[string]intentOrigin{"removed-locked": intentLocal},
			Locked:       map[string]bool{"removed-locked": true, "removed-unlocked": false},
			ProcessStops: map[string]processStopEvidence{"removed-stop": stop}, AllocatedPorts: map[string]int{"removed-port": 23457},
		}
		store := newStore(t, encode(t, state))
		unchanged(t, store, func() {
			got := read(t, store)
			if !reflect.DeepEqual(got.State, state) {
				t.Fatalf("retained document = %+v, want %+v", got.State, state)
			}
			if _, present := got.State.Engines["removed-unlocked"]; present {
				t.Fatal("false lock key fabricated intent")
			}
		})
	})
	t.Run("missing file is complete empty and stays missing", func(t *testing.T) {
		store := newStore(t, "")
		unchanged(t, store, func() {
			got := read(t, store)
			if len(got.State.Engines) != 0 || len(got.State.OffBy) != 0 || len(got.State.Locked) != 0 ||
				len(got.State.ProcessStops) != 0 || len(got.State.AllocatedPorts) != 0 || got.State.PendingSwitch != nil || got.State.PendingSwitchPorts != nil {
				t.Fatalf("missing saved document became nonempty: %+v", got.State)
			}
		})
	})
	t.Run("unresolved base directory is unknown", func(t *testing.T) {
		unknown(t, newDesiredStateStore(""))
	})
	t.Run("unreadable document is unknown", func(t *testing.T) {
		store := newStore(t, "")
		if err := os.Mkdir(store.path, 0o700); err != nil {
			t.Fatalf("create non-file fixture: %v", err)
		}
		unknown(t, store)
		info, err := os.Stat(store.path)
		if err != nil || !info.IsDir() {
			t.Fatalf("read changed non-file fixture: info=%v err=%v", info, err)
		}
	})

	reject := func(name, document string) {
		t.Run(name, func(t *testing.T) {
			store := newStore(t, document)
			unchanged(t, store, func() { unknown(t, store) })
		})
	}
	reject("malformed JSON is unknown without repair", `{"engines":`)
	reject("null document is unknown", `null`)
	reject("non-object document is unknown", `[]`)
	reject("null engines are unknown", `{"engines":null}`)
	reject("non-object engines are unknown", `{"engines":[]}`)
	reject("null intent is unknown", `{"engines":{"fake":null}}`)
	reject("non-boolean intent is unknown", `{"engines":{"fake":"false"}}`)
	reject("unknown origin is unknown", `{"engines":{"fake":false},"offBy":{"fake":"external"}}`)
	reject("null origin is unknown", `{"engines":{"fake":false},"offBy":{"fake":null}}`)
	reject("empty origin is unknown", `{"engines":{"fake":false},"offBy":{"fake":""}}`)
	reject("null origin map is unknown", `{"engines":{"fake":false},"offBy":null}`)
	reject("origin without intent is unknown", `{"engines":{},"offBy":{"removed":"peer"}}`)
	reject("origin for On is unknown", `{"engines":{"fake":true},"offBy":{"fake":"local"}}`)
	reject("null locks are unknown", `{"engines":{},"locked":null}`)
	reject("null lock is unknown", `{"engines":{},"locked":{"fake":null}}`)
	reject("non-boolean lock is unknown", `{"engines":{},"locked":{"fake":1}}`)
	reject("locked On is unknown", `{"engines":{"fake":true},"locked":{"fake":true}}`)
	reject("lock without saved intent is unknown", `{"engines":{},"locked":{"fake":true}}`)
	reject("locked peer Off is inconsistent", `{"engines":{"fake":false},"offBy":{"fake":"peer"},"locked":{"fake":true}}`)
	reject("invalid allocated port is unknown", `{"allocatedPorts":{"removed":65536}}`)
	reject("null allocated port is unknown", `{"allocatedPorts":{"removed":null}}`)
	reject("null cleanup map is unknown", `{"processStops":null}`)
	reject("invalid cleanup record is unknown", `{"processStops":{"removed":{"bin":"relative-image","port":12345,"address":"127.0.0.1:12345","pid":0}}}`)
	reject("orphan checkpoint ports are unknown", `{"pendingSwitchPorts":{"source":0,"target":0}}`)
	reject("invalid checkpoint is unknown", `{"pendingSwitch":{"phase":"reserved"}}`)
	reject("unrecognized saved field is unknown", `{"Engines":{"fake":false}}`)

	compareRevisions := func(name, first, second string, sameIntent, sameLocks bool) {
		t.Run(name, func(t *testing.T) {
			a := read(t, newStore(t, first))
			b := read(t, newStore(t, second))
			if (a.IntentRevision == b.IntentRevision) != sameIntent || (a.LockRevision == b.LockRevision) != sameLocks {
				t.Fatalf("revision equality intent=%v locks=%v, want intent=%v locks=%v; first=%+v second=%+v",
					a.IntentRevision == b.IntentRevision, a.LockRevision == b.LockRevision, sameIntent, sameLocks, a, b)
			}
		})
	}
	compareRevisions("canonical map order and whitespace across owner instances",
		`{"engines":{"a":false,"b":true},"offBy":{"a":"peer"},"locked":{"a":false,"b":false}}`,
		`{ "locked": {"b":false,"a":false}, "offBy":{"a":"peer"}, "engines":{"b":true,"a":false} }`, true, true)
	compareRevisions("legacy and explicit local Off have equal semantics", `{"engines":{"a":false}}`,
		`{"engines":{"a":false},"offBy":{"a":"local"}}`, true, true)
	compareRevisions("missing and explicit empty document have equal semantics", "", `{"engines":{},"offBy":{},"locked":{}}`, true, true)
	compareRevisions("unset and explicit Off have distinct intent revisions", `{}`, `{"engines":{"a":false}}`, false, true)
	compareRevisions("On and Off have distinct intent revisions", `{"engines":{"a":true}}`, `{"engines":{"a":false}}`, false, true)
	compareRevisions("local and peer Off have distinct intent revisions", `{"engines":{"a":false},"offBy":{"a":"local"}}`,
		`{"engines":{"a":false},"offBy":{"a":"peer"}}`, false, true)
	compareRevisions("false lock key is retained in lock revision", `{}`, `{"locked":{"removed":false}}`, true, false)
	compareRevisions("true and false locks have distinct lock revisions", `{"engines":{"a":false},"locked":{"a":false}}`,
		`{"engines":{"a":false},"locked":{"a":true}}`, true, false)
	compareRevisions("execution evidence does not change intent or lock revisions", `{"engines":{"a":false}}`,
		`{"engines":{"a":false},"allocatedPorts":{"removed":23456}}`, true, true)

	lockTransition := func(name, document string) {
		t.Run(name, func(t *testing.T) {
			store := newStore(t, document)
			before := read(t, store)
			changed, err := store.setLocked("fake", true)
			if err != nil || !changed {
				t.Fatalf("save lock and local Off: changed=%v err=%v", changed, err)
			}
			locked := read(t, store)
			enabled, present := locked.State.Engines["fake"]
			if !present || enabled || locked.State.OffBy["fake"] != intentLocal || !locked.State.Locked["fake"] {
				t.Fatalf("lock failed to capture local Off: %+v", locked.State)
			}
			if before.IntentRevision == locked.IntentRevision || before.LockRevision == locked.LockRevision {
				t.Fatal("lock-induced intent and lock revisions did not change")
			}
			changed, err = store.setLocked("fake", false)
			if err != nil || changed {
				t.Fatalf("unlock: changed=%v err=%v", changed, err)
			}
			unlocked := read(t, store)
			if !reflect.DeepEqual(unlocked.State.Engines, locked.State.Engines) || !reflect.DeepEqual(unlocked.State.OffBy, locked.State.OffBy) ||
				unlocked.IntentRevision != locked.IntentRevision || unlocked.LockRevision != before.LockRevision {
				t.Fatalf("unlock changed saved Off or failed to remove lock: locked=%+v unlocked=%+v", locked, unlocked)
			}
			if _, present := unlocked.State.Locked["fake"]; present {
				t.Fatal("unlock retained lock key")
			}
		})
	}
	lockTransition("locking On records local Off and unlock preserves Off", `{"engines":{"fake":true}}`)
	lockTransition("locking unset records local Off and unlock preserves Off", `{}`)
	lockTransition("locking peer Off records local Off and unlock preserves Off", `{"engines":{"fake":false},"offBy":{"fake":"peer"}}`)

	checkpointState := func() desiredStateFile {
		return desiredStateFile{
			Engines: map[string]bool{"source": true, "target": false}, OffBy: map[string]intentOrigin{"target": intentPeer}, Locked: map[string]bool{"target": false},
			PendingSwitch: &settings.SwitchCheckpoint{
				SwitchPrepareRequest: settings.SwitchPrepareRequest{
					OperationID: "owner-evidence", Caller: "fixture", Origin: settings.SwitchOriginLocal,
					SourceEngine: "source", TargetEngine: "target", PayloadHash: "sha256:" + strings.Repeat("a", 64),
					ConfigurationFence: "e3v1:" + strings.Repeat("b", 64), ManagerIncarnation: "predecessor",
					Deadline: time.Unix(1700000000, 0).UTC(),
				},
				Phase:     settings.SwitchCheckpointReserved,
				PreSource: settings.SwitchEngineState{Intent: settings.SwitchIntentOn},
				PreTarget: settings.SwitchEngineState{Intent: settings.SwitchIntentOff}, TargetProvenance: string(intentPeer),
				Source: settings.SwitchEngineState{Intent: settings.SwitchIntentOn}, Target: settings.SwitchEngineState{Intent: settings.SwitchIntentOff},
			},
			PendingSwitchPorts: &switchCheckpointPorts{Source: 23456, Target: 23457, SourcePID: 123, SourceRunID: "old-run"},
		}
	}
	t.Run("checkpoint status preserves unknown runtime observations", func(t *testing.T) {
		state := checkpointState()
		store := newStore(t, encode(t, state))
		unchanged(t, store, func() {
			got := read(t, store)
			if !reflect.DeepEqual(got.State, state) {
				t.Fatalf("checkpoint changed: got=%+v want=%+v", got.State, state)
			}
			for _, observed := range []settings.SwitchEngineState{got.State.PendingSwitch.PreSource, got.State.PendingSwitch.PreTarget, got.State.PendingSwitch.Source, got.State.PendingSwitch.Target} {
				if observed.Running != nil || observed.Healthy != nil || observed.ObservedAt != nil || observed.TerminationVerified {
					t.Fatalf("unknown checkpoint observation fabricated runtime evidence: %+v", observed)
				}
			}
		})
	})
	badCheckpoint := func(name string, change func(*settings.SwitchCheckpoint)) {
		t.Run(name, func(t *testing.T) {
			state := checkpointState()
			change(state.PendingSwitch)
			store := newStore(t, encode(t, state))
			unchanged(t, store, func() { unknown(t, store) })
		})
	}
	badCheckpoint("invalid checkpoint phase is unknown", func(cp *settings.SwitchCheckpoint) { cp.Phase = "invalid" })
	badCheckpoint("invalid checkpoint origin is unknown", func(cp *settings.SwitchCheckpoint) { cp.Origin = "peer" })
	badCheckpoint("invalid checkpoint Off origin is unknown", func(cp *settings.SwitchCheckpoint) { cp.TargetProvenance = "external" })
	badCheckpoint("checkpoint Off origin for On is unknown", func(cp *settings.SwitchCheckpoint) { cp.SourceProvenance = "local" })
	badCheckpoint("invalid checkpoint observation is unknown", func(cp *settings.SwitchCheckpoint) { cp.Source.Intent = "invalid" })
	badCheckpoint("checkpoint cannot fabricate termination from unknown running", func(cp *settings.SwitchCheckpoint) { cp.Source.TerminationVerified = true })
	for _, document := range []string{
		`{"engines":{"hidden":null},"engines":{}}`,
		`{"engines":{"hidden":true,"hidden":false}}`,
	} {
		t.Run("duplicate members refuse "+document, func(t *testing.T) {
			store := newStore(t, document)
			unchanged(t, store, func() { unknown(t, store) })
		})
	}
	t.Run("case variant nested null refuses", func(t *testing.T) {
		state := checkpointState()
		state.ProcessStops = map[string]processStopEvidence{"removed": {Bin: filepath.Join(t.TempDir(), "inert"), Port: 23456, Address: "127.0.0.1:23456", PID: 123}}
		document := strings.Replace(encode(t, state), `"pid":123`, `"PID":null`, 1)
		store := newStore(t, document)
		unchanged(t, store, func() { unknown(t, store) })
	})
	for _, path := range [][]string{
		{"pendingSwitch", "targetCommitted"},
		{"pendingSwitch", "targetProvenance"},
		{"pendingSwitch", "source", "terminationVerified"},
		{"pendingSwitchPorts", "source"},
		{"processStops", "removed", "pid"},
	} {
		t.Run("null nested scalar "+strings.Join(path, "."), func(t *testing.T) {
			state := checkpointState()
			state.ProcessStops = map[string]processStopEvidence{"removed": {Bin: filepath.Join(t.TempDir(), "inert"), Port: 23456, Address: "127.0.0.1:23456", PID: 123}}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal([]byte(encode(t, state)), &fields); err != nil {
				t.Fatal(err)
			}
			var setNull func(map[string]json.RawMessage, []string)
			setNull = func(object map[string]json.RawMessage, keys []string) {
				if len(keys) == 1 {
					object[keys[0]] = json.RawMessage("null")
					return
				}
				var nested map[string]json.RawMessage
				if err := json.Unmarshal(object[keys[0]], &nested); err != nil {
					t.Fatal(err)
				}
				setNull(nested, keys[1:])
				data, err := json.Marshal(nested)
				if err != nil {
					t.Fatal(err)
				}
				object[keys[0]] = data
			}
			setNull(fields, path)
			data, err := json.Marshal(fields)
			if err != nil {
				t.Fatal(err)
			}
			store := newStore(t, string(data))
			unchanged(t, store, func() { unknown(t, store) })
		})
	}
	t.Run("checkpoint endpoint validation covers complete document", func(t *testing.T) {
		state := checkpointState()
		state.PendingSwitchPorts.Target = -1
		store := newStore(t, encode(t, state))
		unchanged(t, store, func() { unknown(t, store) })
	})
	t.Run("checkpoint does not change intent or lock semantic digests", func(t *testing.T) {
		state := checkpointState()
		before := read(t, newStore(t, encode(t, state)))
		state.PendingSwitch = nil
		state.PendingSwitchPorts = nil
		after := read(t, newStore(t, encode(t, state)))
		if before.IntentRevision != after.IntentRevision || before.LockRevision != after.LockRevision {
			t.Fatal("checkpoint changed protected intent or lock semantics")
		}
	})

	t.Run("all returned maps and nested checkpoint pointers are detached", func(t *testing.T) {
		state := checkpointState()
		stopped, healthy := false, false
		observedAt := time.Unix(1700000001, 0).UTC()
		state.PendingSwitch.Source.Running = &stopped
		state.PendingSwitch.Source.Healthy = &healthy
		state.PendingSwitch.Source.ObservedAt = &observedAt
		state.PendingSwitch.Source.TerminationVerified = true
		state.PendingSwitch.Error = &settings.SwitchError{Code: settings.SwitchOperationErrorCode, Kind: settings.SwitchErrorInterrupted, Message: "fixture"}
		stop := processStopEvidence{Bin: filepath.Join(t.TempDir(), "inert-image"), Port: 23456, Address: "127.0.0.1:23456", PID: 123}
		state.ProcessStops = map[string]processStopEvidence{"removed": stop}
		state.AllocatedPorts = map[string]int{"removed": 23456}
		store := newStore(t, encode(t, state))
		unchanged(t, store, func() {
			first, second := read(t, store), read(t, store)
			first.State.Engines["source"] = false
			first.State.OffBy["target"] = intentLocal
			first.State.Locked["target"] = true
			first.State.ProcessStops["removed"] = processStopEvidence{}
			first.State.AllocatedPorts["removed"] = 1
			first.State.PendingSwitch.Phase = settings.SwitchCheckpointReleased
			first.State.PendingSwitchPorts.Source = 1
			*first.State.PendingSwitch.Source.Running = true
			*first.State.PendingSwitch.Source.Healthy = true
			*first.State.PendingSwitch.Source.ObservedAt = time.Time{}
			first.State.PendingSwitch.Error.Message = "mutated"
			third := read(t, store)
			if !reflect.DeepEqual(second.State, state) || !reflect.DeepEqual(third, second) {
				t.Fatalf("returned evidence aliases another capture: second=%+v third=%+v", second, third)
			}
		})
	})

	t.Run("setLocked and evidence capture one joint intent lock baseline", func(t *testing.T) {
		store := newStore(t, `{"engines":{"fake":true}}`)
		before := read(t, store)
		saving := make(chan struct{})
		releaseSave := make(chan struct{})
		writerDone := make(chan struct{})
		var writerChanged bool
		var writerErr error
		useEngineFileReplacer(t, durablefile.Replacer{CreateTemp: func(dir, base string, mode os.FileMode) (durablefile.File, error) {
			close(saving)
			<-releaseSave
			return os.CreateTemp(dir, base+".*.tmp")
		}})
		go func() {
			defer close(writerDone)
			writerChanged, writerErr = store.setLocked("fake", true)
		}()
		select {
		case <-saving: // setLocked changed both fields and still owns desired.mu.
		case <-writerDone:
			t.Fatalf("setLocked did not reach the save boundary: %v", writerErr)
		}
		readerStarted := make(chan struct{})
		readerDone := make(chan struct{})
		captureBoundary := make(chan struct{})
		releaseCapture := make(chan struct{})
		var captured desiredOwnerEvidence
		var captureErr error
		go func() {
			defer close(readerDone)
			close(readerStarted)
			captured, captureErr = store.captureOwnerEvidence(func() ([]byte, error) {
				close(captureBoundary)
				<-releaseCapture
				if store.mu.TryLock() {
					store.mu.Unlock()
					return nil, errors.New("capture read does not hold desired-state owner lock")
				}
				return store.readDocument()
			})
		}()
		<-readerStarted
		var releaseOnce sync.Once
		release := func() { releaseOnce.Do(func() { close(releaseSave) }) }
		// Release and join both owners before cleanup, including on failure.
		defer func() {
			release()
			select {
			case <-releaseCapture:
			default:
				close(releaseCapture)
			}
			<-writerDone
			<-readerDone
		}()
		if store.mu.TryLock() {
			store.mu.Unlock()
			t.Fatal("setLocked released the owner boundary before saving")
		}
		select {
		case <-readerDone:
			t.Fatalf("evidence escaped the held desired-state owner: %+v, %v", captured, captureErr)
		default:
		}
		release()
		<-writerDone
		<-captureBoundary
		close(releaseCapture)
		<-readerDone
		if writerErr != nil || !writerChanged || captureErr != nil {
			t.Fatalf("joint capture: changed=%v write=%v read=%v", writerChanged, writerErr, captureErr)
		}
		enabled, present := captured.State.Engines["fake"]
		if !present || enabled || captured.State.OffBy["fake"] != intentLocal || !captured.State.Locked["fake"] ||
			captured.IntentRevision == before.IntentRevision || captured.LockRevision == before.LockRevision {
			t.Fatalf("capture combined different intent/lock baselines: before=%+v after=%+v", before, captured)
		}
		if after := read(t, store); !reflect.DeepEqual(after, captured) {
			t.Fatalf("joint capture revisions do not describe saved document: captured=%+v after=%+v", captured, after)
		}
	})
}
