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
