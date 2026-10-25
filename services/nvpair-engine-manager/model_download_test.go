// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"nvpair-shared/engines"
	"nvpair-shared/modeldownload"
)

func testModelDownloads(t *testing.T) *modelDownloads {
	t.Helper()
	return newModelDownloads(newOpID(), t.TempDir(), map[string]string{"ollama": engines.ModelNamingImpliedLatestTag, "llamacpp": engines.ModelNamingExact})
}

func reserveModelDownload(t *testing.T, d *modelDownloads, engine, model, naming string) *modelDownloadAttempt {
	t.Helper()
	a, err := d.reserve(engine, model, naming)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestModelDownloadNamingEquivalence(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "ollama", "demo", engines.ModelNamingImpliedLatestTag)
	if a.entry.Model != "demo" || a.key.model != "demo:latest" {
		t.Fatalf("raw spelling/key = %q/%q", a.entry.Model, a.key.model)
	}
	if _, err := d.reserve("ollama", "demo:latest", engines.ModelNamingImpliedLatestTag); !errors.Is(err, errDownloadBusy) {
		t.Fatalf("duplicate = %v", err)
	}
	if !d.blocked(downloadContentKey{"ollama", "demo:latest"}) {
		t.Fatal("equivalent key was not fenced")
	}
	if _, err := d.changeMarker(a, false); err != nil {
		t.Fatal(err)
	}
	if err := d.finish(a, "failed", "download_failed", ""); err != nil {
		t.Fatal(err)
	}
	b := reserveModelDownload(t, d, "ollama", "demo:latest", engines.ModelNamingImpliedLatestTag)
	if err := d.finish(b, "failed", "interrupted", ""); err != nil {
		t.Fatal(err)
	}
	s := d.snapshot()
	if len(s.Entries) != 1 || s.Entries[0].ID != b.entry.ID || !s.Entries[0].BlocksReadiness {
		t.Fatalf("retry lost earlier unsafe marker: %+v", s)
	}
}

func TestModelDownloadFixedCapacity(t *testing.T) {
	d := testModelDownloads(t)
	for i := 0; i < modeldownload.MaxUnsafeEntries; i++ {
		reserveModelDownload(t, d, "llamacpp", fmt.Sprintf("m%d", i), engines.ModelNamingExact)
	}
	if _, err := d.reserve("llamacpp", "overflow", engines.ModelNamingExact); !errors.Is(err, errDownloadCapacity) {
		t.Fatalf("overflow = %v", err)
	}
	if got := len(d.snapshot().Entries); got != modeldownload.MaxUnsafeEntries {
		t.Fatalf("entries = %d", got)
	}
}

func TestModelDownloadTerminalRetention(t *testing.T) {
	d := testModelDownloads(t)
	for i := 0; i < 100; i++ {
		a := reserveModelDownload(t, d, "llamacpp", fmt.Sprintf("m%d", i), engines.ModelNamingExact)
		if err := d.finish(a, "failed", "interrupted", ""); err != nil {
			t.Fatal(err)
		}
	}
	s := d.snapshot()
	if len(s.Entries) != modeldownload.MaxRecentEntries || len(d.recent) != modeldownload.MaxRecentEntries {
		t.Fatalf("unbounded outcomes: %d/%d", len(s.Entries), len(d.recent))
	}
	for _, e := range s.Entries {
		if e.BlocksReadiness {
			t.Fatal("pre-effect failure invalidated content")
		}
	}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestModelDownloadGateOrderings(t *testing.T) {
	d := testModelDownloads(t)
	key := downloadContentKey{"llamacpp", "model"}
	reader, err := d.acquireRead(key, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.acquireWrite("llamacpp"); !errors.Is(err, errDownloadBusy) {
		t.Fatalf("writer behind reader = %v", err)
	}
	other, err := d.acquireWrite("ollama")
	if err != nil {
		t.Fatal(err)
	}
	other()
	reader()
	reader()
	writer, err := d.acquireWrite("llamacpp")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.acquireRead(key, false); !errors.Is(err, errDownloadBusy) {
		t.Fatalf("reader behind writer = %v", err)
	}
	writer()
	writer()
	reader, err = d.acquireRead(key, false)
	if err != nil {
		t.Fatal(err)
	}
	reader()
}

func TestModelDownloadRevisionCoalescingAndExhaustion(t *testing.T) {
	d := testModelDownloads(t)
	a := reserveModelDownload(t, d, "llamacpp", "model", engines.ModelNamingExact)
	p := 10
	if err := d.update(a, "in_progress", &p); err != nil {
		t.Fatal(err)
	}
	before := d.snapshot()
	if err := d.update(a, "in_progress", &p); err != nil {
		t.Fatal(err)
	}
	if !before.Equal(d.snapshot()) {
		t.Fatal("unchanged percent advanced revision")
	}
	d.mu.Lock()
	d.revision = math.MaxUint64
	d.mu.Unlock()
	if _, err := d.reserve("llamacpp", "other", engines.ModelNamingExact); !errors.Is(err, errDownloadCapacity) {
		t.Fatalf("exhausted revision = %v", err)
	}
	if d.snapshot().Revision != "18446744073709551615" {
		t.Fatal("revision wrapped")
	}
}
