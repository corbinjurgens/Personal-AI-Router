// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package performancestore

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
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

func TestCheckpointPersistsNewRevisionWithoutSampleChanges(t *testing.T) {
	path := snapshotPath(t)
	s, c := newTestStore(t, path)
	r := &recorder{t: t, s: s}
	r.record(testKey("m"), completed())
	mustSave(t, s)
	first := mustPublished(t, s).Revision
	c.advance(time.Minute)
	if err := s.Checkpoint(); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	got := mustPublished(t, s)
	if got.Revision != first+60_000 || len(got.Entries) != 1 || got.Entries[0].Samples != 1 || got.Entries[0].NewestAgeMs != 60_000 {
		t.Fatalf("checkpoint summary = %+v, want a new revision with the unchanged minute-old sample", got)
	}
	file := readFileSnapshot(t, path)
	if file.Revision != got.Revision || file.SavedAtMs != c.t.UnixMilli() || file.AgeClockMs != c.t.UnixMilli() {
		t.Fatalf("checkpoint file = %+v, want revision %d at %d", file, got.Revision, c.t.UnixMilli())
	}
	if s.Stats().Dirty {
		t.Fatal("checkpoint left unchanged samples dirty")
	}
	mustSave(t, s)
	if after := mustPublished(t, s).Revision; after != got.Revision {
		t.Fatalf("clean Save revision = %d, want %d", after, got.Revision)
	}
}

func TestCheckpointMemoryOnlyNeverPublishes(t *testing.T) {
	s, _ := newTestStore(t, "")
	if err := s.Checkpoint(); err != nil {
		t.Fatalf("memory-only checkpoint: %v", err)
	}
	if _, ok := s.Published("inc-1"); ok {
		t.Fatal("memory-only checkpoint published a snapshot")
	}
}

func TestCheckpointFailureBeforeReplaceKeepsPublishedRevision(t *testing.T) {
	path := snapshotPath(t)
	s, _ := newTestStore(t, path)
	mustSave(t, s)
	before := mustPublished(t, s)
	fileBefore, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read before checkpoint: %v", err)
	}
	s.replacer = durablefile.Replacer{
		Replace:     func(string, string) error { return errInjected },
		IsTransient: func(error) bool { return false },
	}
	if err := s.Checkpoint(); !errors.Is(err, errInjected) || errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("checkpoint = %v, want injected pre-replacement failure", err)
	}
	if got := mustPublished(t, s); !reflect.DeepEqual(got, before) {
		t.Fatalf("failed checkpoint published %+v, want %+v", got, before)
	}
	fileAfter, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read after checkpoint: %v", err)
	}
	if string(fileAfter) != string(fileBefore) {
		t.Fatal("failed checkpoint changed the file")
	}
	s.replacer = durablefile.Replacer{}
	if err := s.Checkpoint(); err != nil {
		t.Fatalf("retry checkpoint: %v", err)
	}
	if got := mustPublished(t, s).Revision; got != before.Revision+2 {
		t.Fatalf("retry revision = %d, want %d (failed issuance skipped)", got, before.Revision+2)
	}
}

func TestCheckpointFailureAfterReplaceAdoptsRevision(t *testing.T) {
	path := snapshotPath(t)
	s, _ := newTestStore(t, path)
	mustSave(t, s)
	before := mustPublished(t, s).Revision
	s.replacer = durablefile.Replacer{SyncDir: func(string) error { return errInjected }}
	if err := s.Checkpoint(); !errors.Is(err, errInjected) || !errors.Is(err, durablefile.ErrReplaced) {
		t.Fatalf("checkpoint = %v, want replaced durability failure", err)
	}
	got := mustPublished(t, s).Revision
	if got != before+1 || readFileSnapshot(t, path).Revision != got || s.Stats().Dirty {
		t.Fatalf("landed checkpoint revision = %d, stats = %+v, want saved revision %d", got, s.Stats(), before+1)
	}
}

func TestLoadPreservesCheckpointAgeDuringWallRollback(t *testing.T) {
	path := snapshotPath(t)
	s, c := newTestStore(t, path)
	r := &recorder{t: t, s: s}
	r.record(testKey("m"), completed())
	observed := c.t.UnixMilli()
	c.advance(time.Minute)
	mustSave(t, s)
	before := mustPublished(t, s)
	restarted, rc := newTestStore(t, path)
	rc.t = c.t.Add(-time.Hour)
	if err := restarted.Load(); err != nil {
		t.Fatalf("load after rollback: %v", err)
	}
	got := mustPublished(t, restarted)
	if len(got.Entries) != 1 || got.Entries[0].NewestAgeMs != 60_000 || !got.Entries[0].ClockUncertain || got.Entries[0].NewestObservedAtMs != observed {
		t.Fatalf("restarted entry = %+v, want minute-old, uncertain sample with original observation %d", got.Entries, observed)
	}
	if err := restarted.Checkpoint(); err != nil {
		t.Fatalf("checkpoint after rollback: %v", err)
	}
	if rev := mustPublished(t, restarted).Revision; rev != before.Revision+1 {
		t.Fatalf("rollback checkpoint revision = %d, want %d", rev, before.Revision+1)
	}
	// A further restart must retain the logical age baseline, then add the
	// nonnegative wall time elapsed since the rollback checkpoint.
	again, ac := newTestStore(t, path)
	ac.t = rc.t.Add(30 * time.Second)
	if err := again.Load(); err != nil {
		t.Fatalf("load rollback checkpoint: %v", err)
	}
	if age := mustPublished(t, again).Entries[0].NewestAgeMs; age != 90_000 {
		t.Fatalf("age after second restart = %d, want 90000", age)
	}
}

func TestRecordAfterRollbackStartsWithZeroAge(t *testing.T) {
	s, c := newTestStore(t, snapshotPath(t))
	r := &recorder{t: t, s: s}
	r.record(testKey("old"), completed())
	c.advance(time.Minute)
	mustSave(t, s)
	c.advance(-time.Hour)
	r.record(testKey("new"), completed())
	for _, e := range s.Local().Entries {
		wantAge := int64(60_000)
		if e.Key.Model == "new" {
			wantAge = 0
			if e.NewestObservedAtMs != c.t.UnixMilli() {
				t.Fatalf("new observation = %d, want wall time %d", e.NewestObservedAtMs, c.t.UnixMilli())
			}
		}
		if e.NewestAgeMs != wantAge {
			t.Fatalf("%s age = %d, want %d", e.Key.Model, e.NewestAgeMs, wantAge)
		}
	}
	if got := len(s.Local().Entries); got != 2 {
		t.Fatalf("retained entries = %d, want 2", got)
	}
}

func TestLoadOldSnapshotUsesConservativeAgeFloor(t *testing.T) {
	path := snapshotPath(t)
	const observed int64 = 1_760_000_000_000
	const revision int64 = observed + 60_000
	data := []byte(`{"schemaVersion":1,"host":"host-a","revision":1760000060000,"keys":[{"key":{"host":"host-a","engine":"llamacpp-custom","model":"m","fingerprint":"fp-1"},"profile":{},"samples":[{"observedAtMs":1760000000000,"outcome":"completed"}]}]}`)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("mkdir legacy fixture: %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write legacy fixture: %v", err)
	}
	s, c := newTestStore(t, path)
	c.advance(-time.Hour)
	if err := s.Load(); err != nil {
		t.Fatalf("load old snapshot: %v", err)
	}
	got := mustPublished(t, s)
	if got.Revision != revision || len(got.Entries) != 1 || got.Entries[0].NewestAgeMs != 60_000 || got.Entries[0].NewestObservedAtMs != observed {
		t.Fatalf("old snapshot summary = %+v, want revision %d and age 60000", got, revision)
	}
}

func TestRevisionExhaustionLeavesSavedSnapshotIntact(t *testing.T) {
	path := snapshotPath(t)
	s, _ := newTestStore(t, path)
	s.issued = math.MaxInt64 - 1
	mustSave(t, s)
	restarted, _ := newTestStore(t, path)
	if err := restarted.Load(); err != nil {
		t.Fatalf("load maximum revision: %v", err)
	}
	r := &recorder{t: t, s: restarted}
	r.record(testKey("unsaved"), completed())
	if err := restarted.Save(); !errors.Is(err, ErrRevisionExhausted) {
		t.Fatalf("Save at maximum revision = %v, want ErrRevisionExhausted", err)
	}
	if err := restarted.Checkpoint(); !errors.Is(err, ErrRevisionExhausted) {
		t.Fatalf("Checkpoint at maximum revision = %v, want ErrRevisionExhausted", err)
	}
	if got := mustPublished(t, restarted); got.Revision != math.MaxInt64 || len(got.Entries) != 0 {
		t.Fatalf("exhausted publication = %+v, want saved empty maximum revision", got)
	}
	if got := readFileSnapshot(t, path).Revision; got != math.MaxInt64 {
		t.Fatalf("exhausted file revision = %d, want maximum", got)
	}
}

func TestLoadQuarantinesCorruptHistoryWithoutReplacingEarlierQuarantine(t *testing.T) {
	path := snapshotPath(t)
	s, c := newTestStore(t, path)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("mkdir history: %v", err)
	}
	const earlier = "earlier retained history"
	const corrupt = `{"privateUnrecognizedField":"do not log this value"}`
	firstQuarantine := fmt.Sprintf("%s.corrupt-%d", path, c.t.UnixMilli())
	if err := os.WriteFile(firstQuarantine, []byte(earlier), 0o600); err != nil {
		t.Fatalf("write earlier quarantine: %v", err)
	}
	if err := os.WriteFile(path, []byte(corrupt), 0o600); err != nil {
		t.Fatalf("write corrupt history: %v", err)
	}
	var log bytes.Buffer
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&log, nil)))
	t.Cleanup(func() { slog.SetDefault(previousLogger) })
	if err := s.Load(); err != nil {
		t.Fatalf("recover corrupt history: %v", err)
	}
	quarantined := fmt.Sprintf("%s.corrupt-%d", path, c.t.UnixMilli()+1)
	var warning struct {
		Time  string `json:"time"`
		Level string `json:"level"`
		Msg   string `json:"msg"`
		Path  string `json:"path"`
	}
	decoder := json.NewDecoder(&log)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&warning); err != nil {
		t.Fatalf("decode path-only quarantine warning: %v", err)
	}
	if warning.Level != "WARN" || warning.Msg != "performance history quarantined" || warning.Path != quarantined {
		t.Fatalf("quarantine warning = %+v, want WARN naming only %s", warning, quarantined)
	}
	checkBytes := func(file, want string) {
		t.Helper()
		got, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read preserved history %s: %v", file, err)
		}
		if string(got) != want {
			t.Fatalf("preserved history %s = %q, want %q", file, got, want)
		}
	}
	checkBytes(firstQuarantine, earlier)
	checkBytes(quarantined, corrupt)
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("original history after quarantine = %v, want missing", err)
	}
	if stats := s.Stats(); !stats.HistoryReset || !stats.Dirty || stats.PublishedRevision != 0 || stats.Samples != 0 {
		t.Fatalf("recovered stats = %+v, want empty, dirty, unpublished history reset", stats)
	}
	if err := s.Checkpoint(); err != nil {
		t.Fatalf("checkpoint recovered history: %v", err)
	}
	got := mustPublished(t, s)
	if got.Revision != c.t.UnixMilli() || len(got.Entries) != 0 {
		t.Fatalf("recovered checkpoint = %+v, want empty snapshot at current clock", got)
	}
	checkBytes(firstQuarantine, earlier)
	checkBytes(quarantined, corrupt)
}

func TestLoadQuarantineFailureDisablesPersistenceAndPreservesOriginal(t *testing.T) {
	path := snapshotPath(t)
	s, c := newTestStore(t, path)
	// A directory is unreadable as a snapshot and must remain untouched rather
	// than moving a tree when a regular history file was expected.
	if err := os.MkdirAll(path, 0o700); err != nil {
		t.Fatalf("mkdir unreadable snapshot path: %v", err)
	}
	marker := filepath.Join(path, "retained")
	if err := os.WriteFile(marker, []byte("original"), 0o600); err != nil {
		t.Fatalf("write original marker: %v", err)
	}
	err := s.Load()
	if err == nil || err.Error() != "performancestore: snapshot quarantine failed; persistence disabled: "+path {
		t.Fatalf("load = %v, want path-only disabled-persistence diagnostic", err)
	}
	if stats := s.Stats(); stats.HistoryReset || stats.PublishedRevision != 0 || s.path != "" {
		t.Fatalf("failed-quarantine stats = %+v, path = %q, want disabled persistence without reset", stats, s.path)
	}
	r := &recorder{t: t, s: s}
	r.record(testKey("local"), completed())
	mustSave(t, s)
	if err := s.Checkpoint(); err != nil {
		t.Fatalf("disabled-persistence checkpoint: %v", err)
	}
	if _, ok := s.Published("inc-1"); ok {
		t.Fatal("failed quarantine subsequently published a replacement")
	}
	if got := onlyEntry(t, s).Samples; got != 1 {
		t.Fatalf("live samples after failed quarantine = %d, want 1", got)
	}
	got, err := os.ReadFile(marker)
	if err != nil || string(got) != "original" {
		t.Fatalf("original after Save/Checkpoint = %q, %v, want preserved", got, err)
	}
	reservation := fmt.Sprintf("%s.corrupt-%d", path, c.t.UnixMilli())
	if _, err := os.Stat(reservation); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed quarantine reservation = %v, want removed", err)
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
	t.Run("corrupt history recovers with a higher revision when wall time advances", func(t *testing.T) {
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
		if err := restarted.Load(); err != nil {
			t.Fatalf("recover corrupt history: %v", err)
		}
		if !restarted.Stats().HistoryReset {
			t.Fatal("recovered corrupt history did not report history reset")
		}
		mustSave(t, restarted)
		if rev := mustPublished(t, restarted).Revision; rev <= first {
			t.Fatalf("revision after a lost file = %d, want above %d", rev, first)
		}
	})
}

func TestLoadQuarantinesCorruptSnapshots(t *testing.T) {
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
			if _, err := readSnapshot(path, testHost); !errors.Is(err, ErrCorrupt) {
				t.Fatalf("read invalid snapshot = %v, want ErrCorrupt", err)
			}
			s, c := newTestStore(t, path)
			if err := s.Load(); err != nil {
				t.Fatalf("recover invalid snapshot: %v", err)
			}
			if st := s.Stats(); st.Keys != 0 || !st.Dirty || st.PublishedRevision != 0 || !st.HistoryReset {
				t.Fatalf("stats after corrupt load = %+v, want empty, dirty, unpublished reset", st)
			}
			if _, ok := s.Published("inc-1"); ok {
				t.Fatal("recovered corrupt snapshot published before Save")
			}
			quarantined := fmt.Sprintf("%s.corrupt-%d", path, c.t.UnixMilli())
			preserved, err := os.ReadFile(quarantined)
			if err != nil || !bytes.Equal(preserved, data) {
				t.Fatalf("quarantined bytes match = %t, error = %v, want preserved original", bytes.Equal(preserved, data), err)
			}
			if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("original snapshot after quarantine = %v, want missing", err)
			}
		})
	}
	reject("truncated JSON", nil, `{"schemaVersion":1,`)
	reject("unknown field", nil, `{"schemaVersion":1,"host":"host-a","revision":5,"keys":[],"extra":true}`)
	reject("trailing data", nil, `{"schemaVersion":1,"host":"host-a","revision":5,"keys":[]} {}`)
	reject("newer schema", func(s *snapshot) { s.SchemaVersion = FileSchemaVersion + 1 }, "")
	reject("another host", func(s *snapshot) { s.Host = "host-b" }, "")
	reject("zero revision", func(s *snapshot) { s.Revision = 0 }, "")
	reject("negative saved time", func(s *snapshot) { s.SavedAtMs = -1 }, "")
	reject("incomplete age clock", func(s *snapshot) { s.AgeClockMs = 1_760_000_000_000 }, "")
	reject("age clock before saved time", func(s *snapshot) {
		s.SavedAtMs, s.AgeClockMs = 1_760_000_000_001, 1_760_000_000_000
	}, "")
	reject("negative sample age clock", func(s *snapshot) { s.Keys[0].Samples[0].AgeAtMs = -1 }, "")
	reject("sample age clock without anchor", func(s *snapshot) { s.Keys[0].Samples[0].AgeAtMs = 1_760_000_000_000 }, "")
	reject("sample age clock after snapshot", func(s *snapshot) {
		s.SavedAtMs, s.AgeClockMs = 1_760_000_000_000, 1_760_000_000_000
		s.Keys[0].Samples[0].AgeAtMs = 1_760_000_000_001
	}, "")
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

func TestLoadQuarantinesOversizeFile(t *testing.T) {
	path := snapshotPath(t)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	data := []byte(strings.Repeat(" ", MaxFileBytes+1))
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := readSnapshot(path, testHost); !errors.Is(err, ErrOversize) {
		t.Fatalf("read oversize snapshot = %v, want ErrOversize", err)
	}
	s, c := newTestStore(t, path)
	if err := s.Load(); err != nil {
		t.Fatalf("recover oversize snapshot: %v", err)
	}
	if stats := s.Stats(); !stats.HistoryReset || !stats.Dirty || stats.PublishedRevision != 0 || stats.Samples != 0 {
		t.Fatalf("oversize recovery stats = %+v, want empty, dirty, unpublished reset", stats)
	}
	if _, ok := s.Published("inc-1"); ok {
		t.Fatal("oversize recovery published before Save")
	}
	quarantined := fmt.Sprintf("%s.corrupt-%d", path, c.t.UnixMilli())
	preserved, err := os.ReadFile(quarantined)
	if err != nil || !bytes.Equal(preserved, data) {
		t.Fatalf("oversize preserved bytes match = %t, error = %v", bytes.Equal(preserved, data), err)
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
