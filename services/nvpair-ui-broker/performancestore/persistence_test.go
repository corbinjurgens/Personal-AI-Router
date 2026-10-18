// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package performancestore

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/performancewire"
)

var errInjected = errors.New("injected failure")

func snapshotPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "state", FileName)
}

// readFileSnapshot decodes the snapshot file as written.
func readFileSnapshot(t *testing.T, path string) snapshot {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read snapshot: %v", err)
	}
	var snap snapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		t.Fatalf("decode snapshot: %v", err)
	}
	return snap
}

func mustSave(t *testing.T, s *Store) {
	t.Helper()
	if err := s.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}
}

func mustPublished(t *testing.T, s *Store) performancewire.Summary {
	t.Helper()
	sum, ok := s.Published("inc-1")
	if !ok {
		t.Fatal("no published snapshot")
	}
	if err := sum.Validate(); err != nil {
		t.Fatalf("published summary is invalid: %v", err)
	}
	return sum
}

func TestMemoryOnlyStoreNeverPublishes(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	r.record(testKey("m"), completed())
	mustSave(t, s)
	if err := s.Load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	if _, ok := s.Published("inc-1"); ok {
		t.Fatal("memory-only store published a snapshot")
	}
}

func TestFirstSavePublishesEmptySnapshot(t *testing.T) {
	path := snapshotPath(t)
	s, _ := newTestStore(t, path)
	if err := s.Load(); err != nil {
		t.Fatalf("load missing file: %v", err)
	}
	if _, ok := s.Published("inc-1"); ok {
		t.Fatal("published before any save")
	}
	mustSave(t, s)
	sum := mustPublished(t, s)
	if sum.Entries == nil || len(sum.Entries) != 0 {
		t.Fatalf("entries = %#v, want an explicit empty list", sum.Entries)
	}
	data, err := json.Marshal(sum)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatalf("decode fields: %v", err)
	}
	if string(fields["entries"]) != "[]" {
		t.Fatalf("entries encoded as %s, want []", fields["entries"])
	}
}

func TestSaveWithoutChangesKeepsRevision(t *testing.T) {
	path := snapshotPath(t)
	s, c := newTestStore(t, path)
	r := &recorder{t: t, s: s}
	r.record(testKey("m"), completed())
	mustSave(t, s)
	first := mustPublished(t, s).Revision
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}

	c.advance(time.Minute)
	mustSave(t, s)
	again := mustPublished(t, s)
	if again.Revision != first {
		t.Fatalf("unchanged save moved revision %d to %d", first, again.Revision)
	}
	if again.Entries[0].NewestAgeMs != 60_000 {
		t.Fatalf("retransmitted age = %d, want 60000", again.Entries[0].NewestAgeMs)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if !after.ModTime().Equal(info.ModTime()) {
		t.Fatal("unchanged save rewrote the file")
	}
}

func TestPublishedExcludesUnsavedSamples(t *testing.T) {
	s, _ := newTestStore(t, snapshotPath(t))
	r := &recorder{t: t, s: s}
	r.record(testKey("saved"), completed())
	mustSave(t, s)
	r.record(testKey("unsaved"), completed())

	if n := len(mustPublished(t, s).Entries); n != 1 {
		t.Fatalf("published %d entries, want only the saved one", n)
	}
	if n := len(s.Local().Entries); n != 2 {
		t.Fatalf("local view has %d entries, want both", n)
	}
	if !s.Stats().Dirty {
		t.Fatal("store with an unsaved sample is not dirty")
	}
}

func TestLoadReplaysPublishedSnapshot(t *testing.T) {
	path := snapshotPath(t)
	s, c := newTestStore(t, path)
	r := &recorder{t: t, s: s}
	m := completed()
	m.TotalMs = ptr[int64](250)
	m.OutputTokens, m.DecodeNs = ptr[int64](0), ptr[int64](1_000_000)
	r.record(testKey("a"), m)
	r.record(testKey("b"), performancewire.Measurement{Outcome: performancewire.OutcomeFailed})
	mustSave(t, s)
	c.advance(time.Second)
	want := mustPublished(t, s)

	restarted, rc := newTestStore(t, path)
	rc.t = c.t
	if err := restarted.Load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	got := mustPublished(t, restarted)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("replayed summary differs:\n got %+v\nwant %+v", got, want)
	}
	if restarted.Stats().Dirty {
		t.Fatal("a freshly loaded store is dirty")
	}
	mustSave(t, restarted)
	if rev := mustPublished(t, restarted).Revision; rev != want.Revision {
		t.Fatalf("saving an unchanged replay moved revision %d to %d", want.Revision, rev)
	}
}

func TestRevisionsIncrease(t *testing.T) {
	t.Run("each change takes a higher revision", func(t *testing.T) {
		s, c := newTestStore(t, snapshotPath(t))
		r := &recorder{t: t, s: s}
		r.record(testKey("m"), completed())
		mustSave(t, s)
		first := mustPublished(t, s).Revision
		if first < c.t.UnixMilli() {
			t.Fatalf("first revision %d is below the clock %d", first, c.t.UnixMilli())
		}
		r.record(testKey("m"), completed()) // same millisecond
		mustSave(t, s)
		if second := mustPublished(t, s).Revision; second != first+1 {
			t.Fatalf("second revision = %d, want %d", second, first+1)
		}
	})
	t.Run("clock rollback after restart still increases", func(t *testing.T) {
		path := snapshotPath(t)
		s, c := newTestStore(t, path)
		r := &recorder{t: t, s: s}
		r.record(testKey("m"), completed())
		mustSave(t, s)
		first := mustPublished(t, s).Revision

		restarted, rc := newTestStore(t, path)
		rc.t = c.t.Add(-time.Hour)
		if err := restarted.Load(); err != nil {
			t.Fatalf("load: %v", err)
		}
		r2 := &recorder{t: t, s: restarted}
		r2.record(testKey("m"), completed())
		mustSave(t, restarted)
		if rev := mustPublished(t, restarted).Revision; rev != first+1 {
			t.Fatalf("revision after rollback = %d, want %d", rev, first+1)
		}
	})
	t.Run("lost file never reissues an old revision", func(t *testing.T) {
		path := snapshotPath(t)
		s, c := newTestStore(t, path)
		r := &recorder{t: t, s: s}
		r.record(testKey("m"), completed())
		mustSave(t, s)
		first := mustPublished(t, s).Revision
		if err := os.WriteFile(path, []byte("{"), 0o600); err != nil {
			t.Fatalf("corrupt file: %v", err)
		}

		restarted, rc := newTestStore(t, path)
		rc.t = c.t.Add(time.Second)
		if err := restarted.Load(); !errors.Is(err, ErrCorrupt) {
			t.Fatalf("load = %v, want ErrCorrupt", err)
		}
		mustSave(t, restarted)
		if rev := mustPublished(t, restarted).Revision; rev <= first {
			t.Fatalf("revision after a lost file = %d, want above %d", rev, first)
		}
	})
}

func TestLoadRejectsCorruptSnapshots(t *testing.T) {
	valid := func() snapshot {
		return snapshot{
			SchemaVersion: FileSchemaVersion,
			Host:          testHost,
			Revision:      5,
			Keys: []snapshotKey{{
				Key:     testKey("m"),
				Samples: []record{{ObservedAtMs: 1_760_000_000_000, Measurement: completed()}},
			}},
		}
	}
	reject := func(name string, mutate func(*snapshot), raw string) {
		t.Run(name, func(t *testing.T) {
			path := snapshotPath(t)
			if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
				t.Fatalf("mkdir: %v", err)
			}
			data := []byte(raw)
			if mutate != nil {
				snap := valid()
				mutate(&snap)
				var err error
				if data, err = json.Marshal(snap); err != nil {
					t.Fatalf("marshal: %v", err)
				}
			}
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatalf("write: %v", err)
			}
			s, _ := newTestStore(t, path)
			if err := s.Load(); !errors.Is(err, ErrCorrupt) {
				t.Fatalf("load = %v, want ErrCorrupt", err)
			}
			if st := s.Stats(); st.Keys != 0 || !st.Dirty || st.PublishedRevision != 0 {
				t.Fatalf("stats after corrupt load = %+v, want empty, dirty, unpublished", st)
			}
		})
	}
	reject("truncated JSON", nil, `{"schemaVersion":1,`)
	reject("unknown field", nil, `{"schemaVersion":1,"host":"host-a","revision":5,"keys":[],"extra":true}`)
	reject("trailing data", nil, `{"schemaVersion":1,"host":"host-a","revision":5,"keys":[]} {}`)
	reject("newer schema", func(s *snapshot) { s.SchemaVersion = FileSchemaVersion + 1 }, "")
	reject("another host", func(s *snapshot) { s.Host = "host-b" }, "")
	reject("zero revision", func(s *snapshot) { s.Revision = 0 }, "")
	reject("foreign key host", func(s *snapshot) { s.Keys[0].Key.Host = "host-b" }, "")
	reject("repeated key", func(s *snapshot) { s.Keys = append(s.Keys, s.Keys[0]) }, "")
	reject("empty key", func(s *snapshot) { s.Keys[0].Samples = nil }, "")
	reject("too many samples", func(s *snapshot) {
		for len(s.Keys[0].Samples) <= performancewire.MaxSamplesPerKey {
			s.Keys[0].Samples = append(s.Keys[0].Samples, s.Keys[0].Samples[0])
		}
	}, "")
	reject("invalid measurement", func(s *snapshot) { s.Keys[0].Samples[0].TotalMs = ptr[int64](-1) }, "")
	reject("missing observation time", func(s *snapshot) { s.Keys[0].Samples[0].ObservedAtMs = 0 }, "")
}

func TestLoadRejectsOversizeFile(t *testing.T) {
	path := snapshotPath(t)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(strings.Repeat(" ", MaxFileBytes+1)), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	s, _ := newTestStore(t, path)
	if err := s.Load(); !errors.Is(err, ErrOversize) {
		t.Fatalf("load = %v, want ErrOversize", err)
	}
	mustSave(t, s)
	if info, err := os.Stat(path); err != nil || info.Size() > MaxFileBytes {
		t.Fatalf("file after save: %v, %v", info, err)
	}
}

func TestLoadPrunesExpiredSamples(t *testing.T) {
	path := snapshotPath(t)
	s, c := newTestStore(t, path)
	r := &recorder{t: t, s: s}
	r.record(testKey("old"), completed())
	c.advance(24 * time.Hour)
	r.record(testKey("recent"), completed())
	mustSave(t, s)
	saved := mustPublished(t, s).Revision

	restarted, rc := newTestStore(t, path)
	rc.t = c.t.Add(time.Duration(performancewire.RetentionMs)*time.Millisecond - time.Hour)
	if err := restarted.Load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	if !restarted.Stats().Dirty {
		t.Fatal("pruning a loaded sample left the store clean")
	}
	replay := mustPublished(t, restarted)
	if replay.Revision != saved || len(replay.Entries) != 1 || replay.Entries[0].Key != testKey("recent") {
		t.Fatalf("replay = %+v, want revision %d with only the recent key", replay, saved)
	}
	mustSave(t, restarted)
	if rev := mustPublished(t, restarted).Revision; rev <= saved {
		t.Fatalf("revision after pruning = %d, want above %d", rev, saved)
	}
	if keys := readFileSnapshot(t, path).Keys; len(keys) != 1 {
		t.Fatalf("file holds %d keys after pruning, want 1", len(keys))
	}
}

func TestSaveEvictsKeysToFitFile(t *testing.T) {
	path := snapshotPath(t)
	s, c := newTestStore(t, path)
	r := &recorder{t: t, s: s}
	for i := range performancewire.MaxEntries {
		for range performancewire.MaxSamplesPerKey {
			r.record(worstCaseKey(i), maxMeasurement())
		}
		c.advance(time.Millisecond)
	}
	mustSave(t, s)

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Size() > MaxFileBytes {
		t.Fatalf("file is %d bytes, over %d", info.Size(), MaxFileBytes)
	}
	st := s.Stats()
	if st.EvictedKeys == 0 || st.Keys+int(st.EvictedKeys) != performancewire.MaxEntries || st.Dirty {
		t.Fatalf("stats = %+v, want evictions accounting for every key and a clean store", st)
	}
	if _, ok := s.keys[worstCaseKey(0)]; ok {
		t.Fatal("least recently measured key survived file eviction")
	}
	if _, ok := s.keys[worstCaseKey(performancewire.MaxEntries-1)]; !ok {
		t.Fatal("most recently measured key was evicted")
	}

	restarted, rc := newTestStore(t, path)
	rc.t = c.t
	if err := restarted.Load(); err != nil {
		t.Fatalf("load capped file: %v", err)
	}
	if got := restarted.Stats().Keys; got != st.Keys {
		t.Fatalf("reloaded %d keys, want %d", got, st.Keys)
	}
}

func TestSaveFailureBeforeReplaceKeepsPublishedSnapshot(t *testing.T) {
	path := snapshotPath(t)
	s, _ := newTestStore(t, path)
	r := &recorder{t: t, s: s}
	r.record(testKey("first"), completed())
	mustSave(t, s)
	before := mustPublished(t, s)
	beforeFile, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	r.record(testKey("second"), completed())
	s.replacer = durablefile.Replacer{
		Replace:     func(tmp, target string) error { return errInjected },
		IsTransient: func(error) bool { return false },
	}
	err = s.Save()
	if !errors.Is(err, errInjected) || errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("save = %v, want the injected pre-replacement failure", err)
	}
	afterFile, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(afterFile) != string(beforeFile) {
		t.Fatal("a failed save changed the file")
	}
	if got := mustPublished(t, s); !reflect.DeepEqual(got, before) {
		t.Fatalf("published changed after a failed save: %+v", got)
	}
	if !s.Stats().Dirty {
		t.Fatal("store is clean after a failed save")
	}

	s.replacer = durablefile.Replacer{}
	mustSave(t, s)
	after := mustPublished(t, s)
	if after.Revision <= before.Revision || len(after.Entries) != 2 {
		t.Fatalf("retry published %+v, want a higher revision with both keys", after)
	}
}

func TestSaveFailureAfterReplaceAdoptsRevision(t *testing.T) {
	path := snapshotPath(t)
	s, _ := newTestStore(t, path)
	r := &recorder{t: t, s: s}
	r.record(testKey("first"), completed())
	mustSave(t, s)
	before := mustPublished(t, s).Revision

	r.record(testKey("second"), completed())
	s.replacer = durablefile.Replacer{SyncDir: func(string) error { return errInjected }}
	err := s.Save()
	if !errors.Is(err, durablefile.ErrReplaced) || !errors.Is(err, errInjected) {
		t.Fatalf("save = %v, want ErrReplaced wrapping the injected failure", err)
	}
	after := mustPublished(t, s)
	if after.Revision <= before || len(after.Entries) != 2 {
		t.Fatalf("published %+v, want the landed revision with both keys", after)
	}
	if file := readFileSnapshot(t, path); file.Revision != after.Revision {
		t.Fatalf("file revision %d, published %d", file.Revision, after.Revision)
	}
	if s.Stats().Dirty {
		t.Fatal("store is dirty after its snapshot landed")
	}
}
