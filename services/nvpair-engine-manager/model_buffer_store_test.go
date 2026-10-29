// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/modelconfig"
)

func modelBufferStoreFixture(t *testing.T) (*modelBufferStore, modelBufferStoredSample) {
	t.Helper()
	dir := t.TempDir()
	if err := os.Chmod(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	s := newModelBufferStore(dir)
	now := time.Date(2026, 10, 28, 0, 0, 0, 0, time.UTC)
	s.now = func() time.Time { return now }
	sample := modelBufferStoredSample{ModelIdentity: strings.Repeat("a", 64), Observation: modelconfig.BufferObservation{Status: "partial", SampleID: strings.Repeat("b", 64), RecordedAt: now.Format(time.RFC3339Nano), Reason: "missing-boundary"}}
	return s, sample
}

func TestModelBufferStoreRetention(t *testing.T) {
	t.Run("encoded and conservatively charged memory evict before count", func(t *testing.T) {
		s, sample := modelBufferStoreFixture(t)
		for i := range 128 {
			sample.Observation.Rows = append(sample.Observation.Rows, modelconfig.BufferReportRow{Component: "model", Label: "CPU", Phase: "load", Sequence: fmt.Sprint(i + 1), Stream: "stdout", Role: "unattributed", ReportedMiB: strings.Repeat("9", 100) + ".00", Unit: "MiB", Precision: "0.01", SourceKind: "log-scraped", IntervalStatus: "unknown", Coverage: "unknown"})
		}
		for i := range 100 {
			sample.Observation.SampleID = fmt.Sprintf("%064x", i)
			if got := s.publish(sample); got != "saved" {
				t.Fatalf("bounded publication %d: %s", i, got)
			}
		}
		encoded, err := json.Marshal(s.samples)
		if err != nil || len(encoded) > modelBufferHistoryBytes || len(encoded)*4 > 16<<20 || len(s.samples) >= 100 {
			t.Fatal("encoded or retained budget did not evict old completed samples")
		}
		if got := s.newest(sample.ModelIdentity); got.SampleID != sample.Observation.SampleID {
			t.Fatal("retention evicted newest completed sample")
		}
	})
	s, sample := modelBufferStoreFixture(t)
	for i := 0; i < 130; i++ {
		sample.Observation.SampleID = fmt.Sprintf("%064x", i)
		if status := s.publish(sample); status != "saved" {
			t.Fatalf("publication %d: %s", i, status)
		}
	}
	if len(s.samples) != 128 {
		t.Fatalf("retained %d samples", len(s.samples))
	}
	s.now = func() time.Time { return time.Date(2026, 12, 1, 0, 0, 0, 0, time.UTC) }
	if got := s.newest(sample.ModelIdentity); got.Status != "unknown" {
		t.Fatalf("expired sample: %+v", got)
	}
}

func TestModelBufferStoreConfigurationKeyRoundtrip(t *testing.T) {
	for _, key := range []string{"", strings.Repeat("c", 64)} {
		t.Run("key="+key, func(t *testing.T) {
			s, sample := modelBufferStoreFixture(t)
			sample.Observation.ConfigurationKey = key
			if got := s.publish(sample); got != "saved" {
				t.Fatalf("publish=%s", got)
			}
			reopened := newModelBufferStore(filepath.Dir(s.path))
			reopened.now = s.now
			got := reopened.newest(sample.ModelIdentity)
			if got.SampleID != sample.Observation.SampleID || got.ConfigurationKey != key {
				t.Fatalf("reopened sample=%+v", got)
			}
		})
	}
}

func TestModelBufferStorePrivacy(t *testing.T) {
	s, sample := modelBufferStoreFixture(t)
	sample.ModelIdentity = "/private/model.gguf"
	if got := s.publish(sample); got != "failed" {
		t.Fatalf("persisted path identity: %s", got)
	}
	if _, err := os.Stat(s.path); !os.IsNotExist(err) {
		t.Fatalf("invalid sample created history: %v", err)
	}
	_, sample = modelBufferStoreFixture(t)
	for _, mutate := range []func(*modelconfig.BufferObservation){
		func(o *modelconfig.BufferObservation) { o.Reason = "/private/raw/log" },
		func(o *modelconfig.BufferObservation) { o.EstimatorRevision = "secret native args" },
		func(o *modelconfig.BufferObservation) {
			o.Artifacts = []modelconfig.ArtifactSummary{{Role: "model", Identity: "/private/model.gguf", Size: "1", Assurance: "sampled-local-binding"}}
		},
		func(o *modelconfig.BufferObservation) {
			value := "/private/secret"
			o.CapturedOptions = &modelconfig.RequestedLoad{CacheTypeK: &value}
		},
	} {
		candidate := sample
		candidate.Observation = candidate.Observation.Clone()
		mutate(&candidate.Observation)
		if got := s.publish(candidate); got != "failed" {
			t.Fatalf("persisted non-allowlisted value: %s", got)
		}
	}
}

func TestModelBufferStoreRecovery(t *testing.T) {
	t.Run("corrupt documents preserve bytes and disable publication", func(t *testing.T) {
		for _, data := range [][]byte{
			[]byte(`[{`),
			[]byte(`[] trailing`),
			[]byte(`null`),
			[]byte(strings.Repeat("x", modelBufferHistoryBytes+1)),
			[]byte(`[{"modelIdentity":"` + strings.Repeat("a", 64) + `","observation":{"status":"partial","reason":"secret raw text","sampleId":"` + strings.Repeat("b", 64) + `","recordedAt":"2026-10-28T00:00:00Z","current":false}}]`),
		} {
			s, sample := modelBufferStoreFixture(t)
			if err := os.WriteFile(s.path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			recovered := newModelBufferStore(filepath.Dir(s.path))
			if got := recovered.newest(sample.ModelIdentity); got.Reason != "history-unavailable" || recovered.publish(sample) != "history-unavailable" {
				t.Fatal("corruption was silently repaired")
			}
			preserved, err := os.ReadFile(s.path)
			if err != nil || string(preserved) != string(data) {
				t.Fatal("corrupt history was overwritten")
			}
		}
	})
	s, sample := modelBufferStoreFixture(t)
	if got := s.publish(sample); got != "saved" {
		t.Fatal(got)
	}
	recovered := newModelBufferStore(filepath.Dir(s.path))
	recovered.now = s.now
	if got := recovered.newest(sample.ModelIdentity); got.Current || got.SampleID != sample.Observation.SampleID {
		t.Fatalf("recovered currentness: %+v", got)
	}
	bad := []byte(`[{"modelIdentity":"private","modelIdentity":"duplicate"}]`)
	if err := os.WriteFile(s.path, bad, 0o600); err != nil {
		t.Fatal(err)
	}
	corrupt := newModelBufferStore(filepath.Dir(s.path))
	if !corrupt.writeDisabled || corrupt.publish(sample) != "history-unavailable" {
		t.Fatal("corrupt history remained writable")
	}
	data, err := os.ReadFile(s.path)
	if err != nil || string(data) != string(bad) {
		t.Fatalf("corruption overwritten: %v", err)
	}
}

func TestModelBufferStorePublication(t *testing.T) {
	t.Run("pre replacement failure retains old state", func(t *testing.T) {
		s, sample := modelBufferStoreFixture(t)
		if got := s.publish(sample); got != "saved" {
			t.Fatal(got)
		}
		oldID := sample.Observation.SampleID
		sample.Observation.SampleID = strings.Repeat("c", 64)
		s.replacer = durablefile.Replacer{CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) { return nil, errors.New("injected") }}
		if got := s.publish(sample); got != "failed" || len(s.samples) != 1 {
			t.Fatalf("failed write adopted: %s, %d", got, len(s.samples))
		}
		if got := s.newest(sample.ModelIdentity); got.SampleID != oldID {
			t.Fatal("pre-replacement failure replaced the last complete read view")
		}
	})
	t.Run("landed uncertain write adopted once", func(t *testing.T) {
		s, sample := modelBufferStoreFixture(t)
		s.replacer = durablefile.Replacer{SyncDir: func(string) error { return errors.New("injected") }}
		if got := s.publish(sample); got != "uncertain" || len(s.samples) != 1 {
			t.Fatalf("landed write: %s, %d", got, len(s.samples))
		}
		if got := s.publish(sample); got != "uncertain" || len(s.samples) != 1 {
			t.Fatalf("retry duplicated: %s, %d", got, len(s.samples))
		}
		if got := s.newest(sample.ModelIdentity); got.Persistence != "uncertain" {
			t.Fatalf("hidden uncertainty: %+v", got)
		}
	})
}

func TestModelBufferStoreConcurrentReaders(t *testing.T) {
	s, sample := modelBufferStoreFixture(t)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 8; j++ {
				_ = s.newest(sample.ModelIdentity)
			}
		}()
	}
	for i := 0; i < 8; i++ {
		if got := s.publish(sample); got != "saved" {
			t.Fatal(got)
		}
	}
	wg.Wait()
	if len(s.samples) != 1 {
		t.Fatal("retry duplicated history")
	}
	t.Run("readers keep the old snapshot while disk publication blocks", func(t *testing.T) {
		s, sample := modelBufferStoreFixture(t)
		entered, release := make(chan struct{}), make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		defer unblock()
		s.replacer = durablefile.Replacer{SyncDir: func(string) error { close(entered); <-release; return nil }}
		published := make(chan string, 1)
		go func() { published <- s.publish(sample) }()
		select {
		case <-entered:
		case <-time.After(5 * time.Second):
			t.Fatal("publisher did not reach injected boundary")
		}
		read := make(chan modelconfig.BufferObservation, 1)
		go func() { read <- s.newest(sample.ModelIdentity) }()
		select {
		case got := <-read:
			if got.SampleID != "" {
				t.Fatal("adopted candidate before publication completed")
			}
		case <-time.After(time.Second):
			t.Fatal("history read waited for disk I/O")
		}
		unblock()
		select {
		case status := <-published:
			if status != "saved" {
				t.Fatal(status)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("publisher did not finish")
		}
	})
}
