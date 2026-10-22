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
	"runtime"
	"slices"
	"sync"
	"testing"
	"time"

	"nvpair-shared/durablefile"
)

func injectClusterReplacer(t *testing.T, replacer durablefile.Replacer) {
	t.Helper()
	original := clusterFileReplacer
	clusterFileReplacer = replacer
	t.Cleanup(func() { clusterFileReplacer = original })
}

func assertNoWriteTemps(t *testing.T, dir string) {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "*.tmp"))
	if err != nil {
		t.Fatalf("find temp files: %v", err)
	}
	if len(files) != 0 {
		t.Fatalf("temporary residue: %v", files)
	}
}

func createHintTemp(dir, base string, mode os.FileMode) (durablefile.File, error) {
	f, err := os.CreateTemp(dir, base+".*.tmp")
	if err != nil {
		return nil, err
	}
	if err := f.Chmod(mode); err != nil {
		closeErr := f.Close()
		removeErr := os.Remove(f.Name())
		return nil, errors.Join(err, closeErr, removeErr)
	}
	return f, nil
}

type hintWriteSchedule struct {
	mu            sync.Mutex
	created       int
	payloads      [][]ClusterNode
	events        []string
	held, release chan struct{}
	once          sync.Once
	failure       error
	workers       []<-chan struct{}
}

type hintSyncFile struct {
	durablefile.File
	schedule *hintWriteSchedule
	first    bool
	data     []byte
}

func (f *hintSyncFile) Write(data []byte) (int, error) {
	f.data = append(f.data, data...)
	return f.File.Write(data)
}

func (f *hintSyncFile) Sync() error {
	if err := f.File.Sync(); err != nil {
		return err
	}
	var payload []ClusterNode
	if err := json.Unmarshal(f.data, &payload); err != nil {
		return err
	}
	s := f.schedule
	s.mu.Lock()
	s.payloads = append(s.payloads, payload)
	s.events = append(s.events, "sync")
	s.mu.Unlock()
	if f.first {
		close(s.held)
		select {
		case <-s.release:
			return s.failure
		case <-time.After(8 * time.Second):
			return errors.New("held member sync release timed out")
		}
	}
	return nil
}

func newHintWriteSchedule(t *testing.T, m *Manager, failure error) *hintWriteSchedule {
	t.Helper()
	s := &hintWriteSchedule{held: make(chan struct{}), release: make(chan struct{}), failure: failure}
	injectClusterReplacer(t, durablefile.Replacer{CreateTemp: func(dir, base string, mode os.FileMode) (durablefile.File, error) {
		file, err := createHintTemp(dir, base, mode)
		if err != nil {
			return nil, err
		}
		if dir != m.clusterDir || base != membersFile {
			return file, nil
		}
		s.mu.Lock()
		s.created++
		first := s.created == 1
		s.mu.Unlock()
		return &hintSyncFile{File: file, schedule: s, first: first}, nil
	}})
	// Registered after injection: release and join every writer BEFORE restoring
	// the global seam, including when a deliberate control fails an assertion.
	t.Cleanup(func() {
		s.unblock()
		for _, done := range s.workers {
			select {
			case <-done:
			case <-time.After(8 * time.Second):
				t.Error("member writer did not join")
			}
		}
		s.mu.Lock()
		t.Logf("member write events=%v payloads=%+v", s.events, s.payloads)
		s.mu.Unlock()
	})
	return s
}

func (s *hintWriteSchedule) unblock() { s.once.Do(func() { close(s.release) }) }

func (s *hintWriteSchedule) join(t *testing.T) {
	t.Helper()
	for _, done := range s.workers {
		select {
		case <-done:
		case <-time.After(8 * time.Second):
			t.Fatal("member writer failed to join")
		}
	}
}

func (s *hintWriteSchedule) start(name string, fn func() error) <-chan error {
	entered, done, result := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	s.workers = append(s.workers, done)
	go func() {
		defer close(done)
		s.mu.Lock()
		s.events = append(s.events, name+":entered")
		s.mu.Unlock()
		close(entered)
		err := fn()
		s.mu.Lock()
		s.events = append(s.events, name+":completed")
		s.mu.Unlock()
		result <- err
	}()
	<-entered
	return result
}

func waitHintWriter(t *testing.T, result <-chan error) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(8 * time.Second):
		t.Fatal("member writer timed out")
		return nil
	}
}

func assertHeldMemberStore(t *testing.T, m *Manager, s *hintWriteSchedule) bool {
	t.Helper()
	select {
	case <-s.held:
	case <-time.After(8 * time.Second):
		t.Fatal("member writer did not reach held Sync")
	}
	owned := !m.membersStoreMu.TryLock()
	if !owned {
		m.membersStoreMu.Unlock()
		t.Error("held member Sync does not own membersStoreMu")
	}
	if m.memMu.TryLock() {
		m.memMu.Unlock()
	} else {
		t.Error("held member Sync retains memMu")
	}
	return owned
}

func payloadHints(t *testing.T, payload []ClusterNode, uuid string) []string {
	t.Helper()
	for _, member := range payload {
		if member.NodeUUID == uuid {
			return member.AdvertisedAddresses
		}
	}
	t.Fatal("captured member payload omitted peer")
	return nil
}

func TestMembersSnapshotReplacementSerialized(t *testing.T) {
	f := newPeerHintFixture(t)
	s := newHintWriteSchedule(t, f.m, nil)
	old := s.start("old writer", f.m.persistMembersErr)
	owned := assertHeldMemberStore(t, f.m, s)
	fresh := s.start("hint commit", f.commit)
	if !owned {
		// The compiling unsafe control can complete B while old A is held.
		if err := waitHintWriter(t, fresh); err != nil {
			t.Error(err)
		}
	}
	s.unblock()
	if err := waitHintWriter(t, old); err != nil {
		t.Fatal(err)
	}
	if owned {
		if err := waitHintWriter(t, fresh); err != nil {
			t.Fatal(err)
		}
	}
	s.join(t)
	assertPeerHints(t, f, f.descriptor.Addresses)
	if len(s.payloads) != 2 || len(payloadHints(t, s.payloads[0], f.peer.UUID)) != 0 || !slices.Equal(payloadHints(t, s.payloads[1], f.peer.UUID), f.descriptor.Addresses) {
		t.Fatalf("member write order did not preserve old then fresh snapshots: %+v", s.payloads)
	}
	clusterFileReplacer = durablefile.Replacer{}
	if err := f.commit(); err != nil {
		t.Fatal(err)
	}
	assertPeerHints(t, f, f.descriptor.Addresses)
	reopenPeerHints(t, f, f.descriptor.Addresses)
}

func TestPeerServicesHintRollbackExcludesMemberWriter(t *testing.T) {
	f := newPeerHintFixture(t)
	failure := errors.New("injected held member sync failure")
	s := newHintWriteSchedule(t, f.m, failure)
	hint := s.start("hint commit", f.commit)
	owned := assertHeldMemberStore(t, f.m, s)
	// An ordinary display/address update is deliberately outside storage.
	f.m.setMemberAddr(f.peer.UUID, "192.0.2.66", 25423)
	writer := s.start("ordinary writer", f.m.persistMembersErr)
	if !owned {
		if err := waitHintWriter(t, writer); err != nil {
			t.Error(err)
		}
	}
	s.unblock()
	if err := waitHintWriter(t, hint); !errors.Is(err, failure) {
		t.Fatalf("hint error=%v", err)
	}
	if owned {
		if err := waitHintWriter(t, writer); err != nil {
			t.Fatal(err)
		}
	}
	s.join(t)
	assertPeerHints(t, f, nil)
	stored := storedHintMember(t, f.m, f.peer.UUID)
	if stored.IPAddress != "192.0.2.66" || stored.Port != 25423 {
		t.Fatalf("rollback lost unrelated update: %+v", stored)
	}
	if len(s.payloads) != 2 || len(payloadHints(t, s.payloads[1], f.peer.UUID)) != 0 {
		t.Fatalf("ordinary writer leaked speculative hints: %+v", s.payloads)
	}
	assertNoPeerServicesClaim(t, f.output.Bytes())
	assertNoWriteTemps(t, f.m.clusterDir)
}

func TestPeerServicesHintRetryPersistsAfterRollback(t *testing.T) {
	t.Run("retry after pre-replacement failure", func(t *testing.T) {
		f := newPeerHintFixture(t)
		failure := errors.New("injected sync failure")
		s := newHintWriteSchedule(t, f.m, failure)
		result := s.start("failed hint", f.commit)
		assertHeldMemberStore(t, f.m, s)
		s.unblock()
		if err := waitHintWriter(t, result); !errors.Is(err, failure) {
			t.Fatalf("hint error=%v", err)
		}
		s.join(t)
		assertPeerHints(t, f, nil)
		clusterFileReplacer = durablefile.Replacer{CreateTemp: func(dir, base string, mode os.FileMode) (durablefile.File, error) {
			return createHintTemp(dir, base, mode)
		}}
		before, err := os.ReadFile(f.m.membersPath())
		if err != nil {
			t.Fatal(err)
		}
		if err := f.commit(); err != nil {
			t.Fatal(err)
		}
		after, err := os.ReadFile(f.m.membersPath())
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Equal(before, after) {
			t.Fatal("retry did not persist new hints")
		}
		assertPeerHints(t, f, f.descriptor.Addresses)
		clusterFileReplacer = durablefile.Replacer{}
		reopenPeerHints(t, f, f.descriptor.Addresses)
		assertNoWriteTemps(t, f.m.clusterDir)
	})
	t.Run("landed replacement is adopted", func(t *testing.T) {
		f := newPeerHintFixture(t)
		injectClusterReplacer(t, durablefile.Replacer{SyncDir: func(string) error { return errors.New("injected directory sync failure") }})
		if err := f.commit(); err != nil {
			t.Fatal(err)
		}
		assertPeerHints(t, f, f.descriptor.Addresses)
		clusterFileReplacer = durablefile.Replacer{}
		reopenPeerHints(t, f, f.descriptor.Addresses)
	})
}

func TestAdmissionDurableWriteKeepsMemoryAligned(t *testing.T) {
	test := func(name string, replacer durablefile.Replacer, wantCounter uint64, wantErr error) {
		t.Run(name, func(t *testing.T) {
			m := &Manager{clusterDir: t.TempDir(), admissionCounter: 7}
			if err := m.persistAdmissionLocked(); err != nil {
				t.Fatalf("seed admission: %v", err)
			}
			before, err := os.ReadFile(m.admissionPath())
			if err != nil {
				t.Fatal(err)
			}
			injectClusterReplacer(t, replacer)
			epoch, err := m.reserveAdmissionEpoch()
			if !errors.Is(err, wantErr) {
				t.Fatalf("reserve error = %v, want %v", err, wantErr)
			}
			if wantErr == nil && epoch != wantCounter {
				t.Fatalf("epoch = %d, want %d", epoch, wantCounter)
			}
			after, err := os.ReadFile(m.admissionPath())
			if err != nil {
				t.Fatal(err)
			}
			var state admissionState
			if err := json.Unmarshal(after, &state); err != nil {
				t.Fatal(err)
			}
			if state.Counter != wantCounter || m.admissionCounter != wantCounter {
				t.Fatalf("disk counter = %d, memory = %d, want %d", state.Counter, m.admissionCounter, wantCounter)
			}
			if wantErr != nil && !bytes.Equal(before, after) {
				t.Fatal("pre-replacement failure changed target")
			}
			assertNoWriteTemps(t, m.clusterDir)
		})
	}
	failure := errors.New("injected replacement failure")
	test("pre-replacement failure rolls back", durablefile.Replacer{Replace: func(string, string) error { return failure }}, 7, failure)
	test("landed replacement is adopted", durablefile.Replacer{SyncDir: func(string) error { return failure }}, 8, nil)
}

func TestTrustStoreAdoptsLandedDurableWrite(t *testing.T) {
	ts, count := newAnnouncingStore(t)
	pin := testPin(t, "principal-peer")
	first := Endorsement{By: "trusted-peer", Sig: "signature-1"}
	second := Endorsement{By: "trusted-peer", SigV2: "signature-2"}
	pin.Endorsements = []Endorsement{first}
	if err := ts.Pin(pin); err != nil {
		t.Fatalf("seed pin: %v", err)
	}
	injectClusterReplacer(t, durablefile.Replacer{SyncDir: func(string) error { return errors.New("injected directory sync failure") }})
	if err := ts.AddEndorsements(pin.NodeUUID, []Endorsement{second}); err != nil {
		t.Fatalf("adopt landed endorsement: %v", err)
	}
	if count() != 2 {
		t.Fatalf("announcements = %d, want 2", count())
	}
	assertStoredEndorsements(t, ts, pin.NodeUUID, []Endorsement{first, second})
	assertNoWriteTemps(t, ts.dir)
}

func TestIdentityDurableWriteFailurePreservesExistingKeypair(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cluster")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	cert, key, err := generateLeaf("existing-principal", "existing-host")
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string][]byte{"node.key": key, "node.crt": cert} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	failure := errors.New("injected key replacement failure")
	injectClusterReplacer(t, durablefile.Replacer{Replace: func(string, string) error { return failure }})
	id, err := loadOrMintIdentity(dir)
	if !errors.Is(err, failure) || id != nil {
		t.Fatalf("identity returned = %v, error = %v, want nil and injected failure", id != nil, err)
	}
	for name, want := range map[string][]byte{"node.key": key, "node.crt": cert} {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("pre-replacement failure changed %s", name)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "identity.json")); !os.IsNotExist(err) {
		t.Fatalf("identity file after failed key write: %v", err)
	}
	assertNoWriteTemps(t, dir)
}

func TestIdentityCompletesAfterMidSequenceDurabilityFailure(t *testing.T) {
	test := func(name string, failAt int) {
		t.Run(name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "cluster")
			calls := 0
			// Only the selected post-replacement sync fails. The real platform
			// replacement still runs for all three files.
			injectClusterReplacer(t, durablefile.Replacer{SyncDir: func(string) error {
				calls++
				if calls == failAt {
					return errors.New("injected directory sync failure")
				}
				return nil
			}})
			id, err := loadOrMintIdentity(dir)
			if err != nil {
				t.Fatalf("mint identity: %v", err)
			}
			if calls != 3 {
				t.Fatalf("completed writes = %d, want key, cert and identity", calls)
			}
			loaded, err := loadOrMintIdentity(dir)
			if err != nil {
				t.Fatalf("reload complete identity: %v", err)
			}
			if id.NodeUUID != loaded.NodeUUID || id.CertFingerprint != loaded.CertFingerprint ||
				!bytes.Equal(id.KeyPEM, loaded.KeyPEM) || !bytes.Equal(id.CertPEM, loaded.CertPEM) ||
				!reflect.DeepEqual(id.Signer, loaded.Signer) {
				t.Fatal("returned identity differs from persisted identity")
			}
			assertNoWriteTemps(t, dir)
		})
	}
	test("key sync fails", 1)
	test("certificate sync fails", 2)
}

func TestAtomicWritePreservesRequestedMode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows does not expose POSIX permission bits")
	}
	test := func(name string, perm os.FileMode) {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "target")
			if err := os.WriteFile(path, []byte("old"), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := atomicWrite(path, []byte("new"), perm); err != nil {
				t.Fatal(err)
			}
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			if info.Mode().Perm() != perm {
				t.Fatalf("mode = %o, want %o", info.Mode().Perm(), perm)
			}
		})
	}
	test("private key", 0o600)
	test("public certificate", 0o644)
}

func TestAtomicWriteWindowsOpenReaderRetries(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows sharing contract")
	}
	path := filepath.Join(t.TempDir(), "target")
	if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	sleeps := 0
	injectClusterReplacer(t, durablefile.Replacer{Sleep: func(time.Duration) { sleeps++ }})
	if err := atomicWrite(path, []byte("replacement"), 0o600); err == nil {
		t.Fatal("replacement succeeded while a reader held the target")
	}
	if sleeps != 9 {
		t.Fatalf("retry sleeps = %d, want 9 (10 attempts)", sleeps)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "original" {
		t.Fatal("exhausted retries changed original")
	}
	assertNoWriteTemps(t, filepath.Dir(path))
	if err := reader.Close(); err != nil {
		t.Fatal(err)
	}
	if err := atomicWrite(path, []byte("replacement"), 0o600); err != nil {
		t.Fatalf("replace after reader closes: %v", err)
	}
	got, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "replacement" {
		t.Fatal("successful replacement did not land")
	}
	assertNoWriteTemps(t, filepath.Dir(path))
}
