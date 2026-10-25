// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
)

func TestModelDownloadJournalPreReplacementFailure(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
	if _, err := d.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(d.store.path)
	if err != nil {
		t.Fatal(err)
	}
	d.store.replace.Replace = func(string, string) error { return errors.New("injected") }
	if uncertain, err := d.changeMarker(a, true); err == nil || uncertain {
		t.Fatalf("removal = %v/%v", uncertain, err)
	}
	after, err := os.ReadFile(d.store.path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) || len(d.unsafe) != 1 {
		t.Fatal("failed removal changed disk or memory")
	}
	if err := d.finish(a, "unknown", "persistence_error", ""); err != nil {
		t.Fatal(err)
	}
	if !d.snapshot().Entries[0].BlocksReadiness {
		t.Fatal("failed removal promoted content")
	}
}

func TestModelDownloadJournalLandedRemovalQualification(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
	if _, err := d.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	d.store.replace.SyncDir = func(string) error { return errors.New("injected") }
	uncertain, err := d.changeMarker(a, true)
	if err == nil || !uncertain || len(d.unsafe) != 0 {
		t.Fatalf("landed removal = %v/%v/%d", uncertain, err, len(d.unsafe))
	}
	if err := d.finish(a, "complete", "", "uncertain"); err != nil {
		t.Fatal(err)
	}
	s := d.snapshot()
	if s.Entries[0].Persistence != "uncertain" || s.Entries[0].BlocksReadiness {
		t.Fatalf("qualification = %+v", s)
	}
	recovered := newModelDownloads(newOpID(), filepath.Dir(d.store.path), map[string]string{"llamacpp": engines.ModelNamingExact})
	if recovered.recoveryRequired || len(recovered.unsafe) != 0 {
		t.Fatal("memory disagreed with landed disk")
	}
}

func TestModelDownloadJournalLandedAdditionRefusesEffect(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
	d.store.replace.SyncDir = func(string) error { return errors.New("injected") }
	uncertain, err := d.changeMarker(a, false)
	if err == nil || !uncertain || len(d.unsafe) != 1 {
		t.Fatalf("addition = %v/%v/%d", uncertain, err, len(d.unsafe))
	}
}

func TestModelDownloadJournalRecovery(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "ollama", "demo", engines.ModelNamingImpliedLatestTag)
	if _, err := d.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(d.store.path)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0600 {
		t.Fatalf("mode = %o", info.Mode().Perm())
	}
	recovered := newModelDownloads(newOpID(), filepath.Dir(d.store.path), map[string]string{"ollama": engines.ModelNamingImpliedLatestTag})
	s := recovered.snapshot()
	if s.SourceID == d.source || len(s.Entries) != 1 || s.Entries[0].Phase != "unknown" || !s.Entries[0].BlocksReadiness {
		t.Fatalf("recovery = %+v", s)
	}
	changed := newModelDownloads(newOpID(), filepath.Dir(d.store.path), map[string]string{"ollama": engines.ModelNamingExact})
	if !changed.snapshot().RecoveryRequired {
		t.Fatal("changed naming reinterpreted marker")
	}
}

func TestModelDownloadJournalMalformedRecovery(t *testing.T) {
	test := func(name, input string) {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, modelDownloadFile)
			if err := os.WriteFile(path, []byte(input), 0600); err != nil {
				t.Fatal(err)
			}
			d := newModelDownloads(newOpID(), dir, nil)
			if !d.snapshot().RecoveryRequired {
				t.Fatal("malformed state accepted")
			}
			if _, err := d.reserve("llamacpp", "model", engines.ModelNamingExact); !errors.Is(err, errDownloadRecovery) {
				t.Fatalf("reservation = %v", err)
			}
			got, err := os.ReadFile(path)
			if err != nil || string(got) != input {
				t.Fatal("journal was rewritten")
			}
		})
	}
	test("invalid JSON", "{")
	test("null entries", `{"version":1,"entries":null}`)
	test("duplicate entries", `{"version":1,"entries":[],"entries":[]}`)
	test("case variant", `{"version":1,"Entries":[]}`)
	test("unknown field", `{"version":1,"entries":[],"unexpected":true}`)
	test("duplicate marker model", `{"version":1,"entries":[{"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","engine":"llamacpp","model":"one","model":"two","contentKey":"two","naming":"exact"}]}`)
	test("oversize journal", strings.Repeat(" ", modeldownload.MaxJournalBytes+1))
}

func TestModelDownloadJournalEncodedByteBound(t *testing.T) {
	d := testModelDownloads(t)
	for i := 0; i < modeldownload.MaxUnsafeEntries; i++ {
		model := strings.Repeat("<", 500) + string(rune('a'+i))
		a := reserveModelDownload(t, d, "llamacpp", model, engines.ModelNamingExact)
		_, err := d.changeMarker(a, false)
		if err != nil {
			if len(d.unsafe) == 0 || len(d.unsafe) >= modeldownload.MaxUnsafeEntries {
				t.Fatal("byte bound did not precede slot bound")
			}
			return
		}
	}
	t.Fatal("escaped journal exceeded byte limit")
}

func TestModelDownloadJournalSerializesCrossEnginePublication(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "one", engines.ModelNamingExact)
	b := reserveModelDownload(t, d, "ollama", "two", engines.ModelNamingImpliedLatestTag)
	entered, release := make(chan struct{}), make(chan struct{})
	var firstSync sync.Once
	d.store.replace.SyncDir = func(string) error { firstSync.Do(func() { close(entered); <-release }); return nil }
	first := make(chan error, 1)
	go func() { _, err := d.changeMarker(a, false); first <- err }()
	<-entered
	second := make(chan error, 1)
	go func() { _, err := d.changeMarker(b, false); second <- err }()
	// The short gate stays usable while the journal writer is held.
	if len(d.snapshot().Entries) != 2 {
		t.Fatal("snapshot blocked or lost reservation")
	}
	close(release)
	if err := <-first; err != nil {
		t.Fatal(err)
	}
	if err := <-second; err != nil {
		t.Fatal(err)
	}
	markers, err := d.store.load()
	if err != nil || len(markers) != 2 {
		t.Fatalf("cross-engine markers = %d/%v", len(markers), err)
	}
}
