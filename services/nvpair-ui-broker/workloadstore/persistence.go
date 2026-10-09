// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package workloadstore

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"time"
)

// Retention and persistence defaults. The history caps bound the in-memory
// store whether or not persistence is on; only historic (terminal) records are
// written to disk, and active records are ephemeral and rebuilt from the live
// event stream, so a previous session's "running" entries are never restored.
const (
	DefaultHistoryCap    = 10000 // keep newest N terminal records
	DefaultHistoryMaxAge = 7 * 24 * time.Hour
	DefaultRotations     = 3               // rotated backups kept alongside the primary
	DefaultFlushInterval = 5 * time.Second // coalesced dirty flush cadence
	DefaultDumpInterval  = 5 * time.Minute // rotation checkpoint cadence
)

// WithPersistence enables on-disk persistence of historic records at path, and
// returns the store for chaining. Call Load once at startup and Run to start the
// coalescing flusher. With an empty path (the default), the store is purely
// in-memory and every persistence method is a no-op.
func (s *Store) WithPersistence(path string) *Store {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.path = path
	s.rotations = DefaultRotations
	return s
}

// Load reads persisted historic records into the store. A missing file is a
// clean first run (no error); a corrupt primary falls back to the newest good
// rotation. Any parse error is returned for logging but is non-fatal — the
// store simply starts with whatever it could recover.
func (s *Store) Load() error {
	if s.path == "" {
		return nil
	}
	infos, err := readWithFallback(s.path, s.rotations)
	now := s.now()

	s.mu.Lock()
	for _, info := range infos {
		in, ok := ParseIncoming(info)
		if !ok {
			continue
		}
		rec := recordFrom(in, now)
		if !rec.Terminal {
			continue // only historic records are persisted
		}
		s.records[keyOf(in)] = rec
	}
	s.pruneLocked(now.UnixMilli())
	s.dirty = false
	s.mu.Unlock()
	return err
}

// Flush writes the historic set to the primary file (atomic temp+rename) when
// the store is dirty. A no-op when persistence is off or nothing changed since
// the last write. On write failure the dirty flag is restored so a later flush
// retries.
func (s *Store) Flush() error {
	if s.path == "" {
		return nil
	}
	s.mu.Lock()
	if !s.dirty {
		s.mu.Unlock()
		return nil
	}
	infos := s.historyInfosLocked()
	s.dirty = false
	path := s.path
	s.mu.Unlock()

	if err := writeSnapshotFile(path, infos); err != nil {
		s.mu.Lock()
		s.dirty = true
		s.mu.Unlock()
		return err
	}
	return nil
}

// Checkpoint rotates the existing primary to a numbered backup and writes a
// fresh primary, giving periodic rollovers so a single corrupt/oversized file
// can't wipe all history. Independent of the dirty flag.
func (s *Store) Checkpoint() error {
	if s.path == "" {
		return nil
	}
	s.mu.Lock()
	infos := s.historyInfosLocked()
	path, rotations := s.path, s.rotations
	s.dirty = false
	s.mu.Unlock()

	rotate(path, rotations)
	return writeSnapshotFile(path, infos)
}

// Run drives the coalescing flusher until ctx is cancelled: a dirty flush every
// flushInterval, a rotation checkpoint every dumpInterval, and a final flush on
// shutdown. A no-op when persistence is off. Intervals ≤ 0 use the defaults.
func (s *Store) Run(ctx context.Context, flushInterval, dumpInterval time.Duration) {
	if s.path == "" {
		return
	}
	if flushInterval <= 0 {
		flushInterval = DefaultFlushInterval
	}
	if dumpInterval <= 0 {
		dumpInterval = DefaultDumpInterval
	}
	flushT := time.NewTicker(flushInterval)
	dumpT := time.NewTicker(dumpInterval)
	defer flushT.Stop()
	defer dumpT.Stop()

	for {
		select {
		case <-ctx.Done():
			if err := s.Flush(); err != nil {
				slog.Warn("workload store final flush failed", "err", err)
			}
			return
		case <-flushT.C:
			if err := s.Flush(); err != nil {
				slog.Warn("workload store flush failed", "err", err)
			}
		case <-dumpT.C:
			if err := s.Checkpoint(); err != nil {
				slog.Warn("workload store checkpoint failed", "err", err)
			}
		}
	}
}

// historyInfosLocked returns the Info of every terminal record, oldest-first by
// completion time, for a stable on-disk ordering. Caller holds s.mu.
func (s *Store) historyInfosLocked() []json.RawMessage {
	recs := make([]Record, 0, len(s.records))
	for _, r := range s.records {
		// Only authoritative terminals are durable history; an inferred guess
		// (node-loss sweep) is in-memory only and never written to disk.
		if r.Terminal && !r.Inferred {
			recs = append(recs, r)
		}
	}
	sort.Slice(recs, func(i, j int) bool {
		return completionOrder(recs[i]) < completionOrder(recs[j])
	})
	out := make([]json.RawMessage, len(recs))
	for i, r := range recs {
		out[i] = r.Info
	}
	return out
}

// Retired names a workload the history caps dropped from the store, in the
// (originatedFrom, workloadId) shape a workloads:remove carries.
type Retired struct {
	Origin string
	ID     string
}

// PruneHistory enforces the age and count caps on historic (terminal) records
// and returns what clients must be told to forget. Load prunes silently because
// nothing has been announced yet; once the store is live, the broker calls this
// periodically and turns each Retired into a workloads:remove, so a subscriber's
// mirror is bounded by the same caps as the store instead of growing for the
// life of the session. Flush and Checkpoint deliberately do not prune, so a
// record is never dropped without its removal being announced.
//
// A pair is reported only when no record sharing it survives. Clients key by
// (originatedFrom, workloadId) and a proxy restart reuses ids, so retiring an
// old generation must not erase a newer one that the client already shows.
func (s *Store) PruneHistory() []Retired {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pruneLocked(s.now().UnixMilli())
}

// pruneLocked enforces the age and count caps on historic (terminal) records
// and returns the retired pairs no surviving record still carries. Active
// records are never pruned. Caller holds s.mu.
func (s *Store) pruneLocked(now int64) []Retired {
	var dropped []Record
	drop := func(k Key, r Record) {
		delete(s.records, k)
		if !r.Inferred {
			s.dirty = true
		}
		dropped = append(dropped, r)
	}
	if s.maxAgeMs > 0 {
		for k, r := range s.records {
			if r.Terminal && now-completionOrder(r) > s.maxAgeMs {
				drop(k, r)
			}
		}
	}
	if s.historyCap > 0 {
		type termEntry struct {
			key Key
			ord int64
		}
		term := make([]termEntry, 0)
		for k, r := range s.records {
			if r.Terminal {
				term = append(term, termEntry{key: k, ord: completionOrder(r)})
			}
		}
		if len(term) > s.historyCap {
			sort.Slice(term, func(i, j int) bool {
				return term[i].ord < term[j].ord // oldest first
			})
			for i := 0; i < len(term)-s.historyCap; i++ {
				drop(term[i].key, s.records[term[i].key])
			}
		}
	}
	if len(dropped) == 0 {
		return nil
	}

	surviving := make(map[Retired]bool, len(s.records))
	for _, r := range s.records {
		surviving[Retired{Origin: r.Origin, ID: r.ID}] = true
	}
	seen := make(map[Retired]bool, len(dropped))
	retired := make([]Retired, 0, len(dropped))
	for _, r := range dropped {
		pair := Retired{Origin: r.Origin, ID: r.ID}
		if surviving[pair] || seen[pair] {
			continue
		}
		seen[pair] = true
		retired = append(retired, pair)
	}
	return retired
}

// completionOrder is a record's sort/age key: its workload completedAt when
// known, else the store clock at last update (covers a record that reached
// terminal without a completedAt on the wire).
func completionOrder(r Record) int64 {
	if r.CompletedAt > 0 {
		return r.CompletedAt
	}
	return r.LastUpdated
}

func writeSnapshotFile(path string, infos []json.RawMessage) error {
	data, err := json.Marshal(infos)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func readSnapshotFile(path string) ([]json.RawMessage, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var infos []json.RawMessage
	if err := json.Unmarshal(data, &infos); err != nil {
		return nil, err
	}
	return infos, nil
}

// readWithFallback reads the primary snapshot, falling back to the newest good
// rotation if the primary is missing or corrupt. Returns (nil, nil) on a clean
// first run (no files at all), or (nil, err) when files exist but none parse.
func readWithFallback(path string, rotations int) ([]json.RawMessage, error) {
	candidates := []string{path}
	for i := 1; i <= rotations; i++ {
		candidates = append(candidates, fmt.Sprintf("%s.%d", path, i))
	}
	var lastErr error
	sawFile := false
	for _, c := range candidates {
		infos, err := readSnapshotFile(c)
		if err == nil {
			return infos, nil
		}
		if !os.IsNotExist(err) {
			sawFile = true
			lastErr = err
		}
	}
	if !sawFile {
		return nil, nil
	}
	return nil, lastErr
}

// rotate shifts path -> path.1 -> path.2 ... dropping beyond keep. Best-effort:
// missing files are fine.
func rotate(path string, keep int) {
	if keep <= 0 {
		return
	}
	_ = os.Remove(fmt.Sprintf("%s.%d", path, keep))
	for i := keep - 1; i >= 1; i-- {
		_ = os.Rename(fmt.Sprintf("%s.%d", path, i), fmt.Sprintf("%s.%d", path, i+1))
	}
	_ = os.Rename(path, path+".1")
}
