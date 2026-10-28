// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	settings "nvpair-shared/enginesettings"
)

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
