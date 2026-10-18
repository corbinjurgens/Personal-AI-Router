// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package performancestore

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"

	"nvpair-shared/durablefile"
	"nvpair-shared/performancewire"
)

// FileName is the snapshot file under the per-user application directory.
const FileName = "performance-history.json"

// FileSchemaVersion is the snapshot layout Load accepts and Save writes.
const FileSchemaVersion = 1

// MaxFileBytes bounds a snapshot file. Save evicts the least recently
// measured keys until the snapshot fits; Load refuses a larger file unread.
const MaxFileBytes = 2 << 20

var (
	// ErrCorrupt reports a snapshot file that does not parse or validate.
	ErrCorrupt = errors.New("performancestore: snapshot is corrupt")
	// ErrOversize reports a snapshot file over MaxFileBytes.
	ErrOversize = errors.New("performancestore: snapshot exceeds the file limit")
	// ErrRevisionExhausted reports that no larger positive revision exists.
	ErrRevisionExhausted = errors.New("performancestore: snapshot revision exhausted")
)

// snapshot is the persisted form: every retained sample at one revision.
type snapshot struct {
	SchemaVersion int    `json:"schemaVersion"`
	Host          string `json:"host"`
	Revision      int64  `json:"revision"`
	// SavedAtMs and AgeClockMs anchor source ages across restart. Both are
	// absent in older schema-v1 files; observedAtMs remains real wall time.
	SavedAtMs  int64         `json:"savedAtMs,omitempty"`
	AgeClockMs int64         `json:"ageClockMs,omitempty"`
	Keys       []snapshotKey `json:"keys"`
}

type snapshotKey struct {
	Key     performancewire.ProfileKey `json:"key"`
	Profile performancewire.Profile    `json:"profile"`
	Samples []record                   `json:"samples"`
}

// keyStates returns an independently owned copy of the snapshot's samples.
func (sn *snapshot) keyStates() map[performancewire.ProfileKey]*keyState {
	keys := make(map[performancewire.ProfileKey]*keyState, len(sn.Keys))
	for _, k := range sn.Keys {
		keys[k.Key] = &keyState{profile: k.Profile, samples: slices.Clone(k.Samples)}
	}
	return keys
}

// Published returns the summary of the last saved or loaded snapshot under the
// given admission incarnation, with sample ages as of now. It reports false
// when no snapshot has been published. Samples recorded since then, or a save
// that failed before replacing the file, do not appear until a later save
// succeeds. Ages advance without changing revision: transport callers must
// freeze one summary per revision and Checkpoint before publishing rebased ages.
func (s *Store) Published(incarnation string) (performancewire.Summary, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.published == nil {
		return performancewire.Summary{}, false
	}
	wallMs, ageMs := s.clockLocked()
	return buildSummary(s.host, incarnation, s.published.Revision, s.published.keyStates(), wallMs, ageMs), true
}

// Load replaces the store's samples with the snapshot file's, making it the
// published snapshot. Call it once, before recording. A missing file is a
// clean first run. An unreadable, corrupt or oversize file is renamed aside;
// Load returns nil with Stats.HistoryReset set, and the empty, dirty store can
// save again. If quarantine fails, a path-only error is returned and persistence is
// disabled so a later save cannot overwrite the original. Lost/corrupt history
// cannot preserve its revision floor under wall rollback. Loaded samples
// outside the window are pruned, which also leaves the store dirty.
func (s *Store) Load() error {
	if s.path == "" {
		return nil
	}
	snap, err := readSnapshot(s.path, s.host)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		path := s.path
		quarantined, quarantineErr := quarantineSnapshot(path, s.now().UnixMilli())
		if quarantineErr != nil {
			s.path = ""
			return &snapshotLoadError{path: path, cause: errors.Join(err, quarantineErr)}
		}
		s.mu.Lock()
		s.stats.HistoryReset = true
		s.mu.Unlock()
		slog.Warn("performance history quarantined", "path", quarantined)
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.keys = snap.keyStates()
	s.published = snap
	s.issued = max(s.issued, snap.Revision)
	s.savedChanges = s.changes
	now := s.now()
	wallMs := now.UnixMilli()
	// Older snapshots have no explicit age baseline. Their time-based revision
	// is at least the save's wall time, so it is a conservative age floor.
	ageMs := max(wallMs, snap.Revision)
	if snap.SavedAtMs > 0 {
		ageMs = addElapsed(snap.AgeClockMs, max(int64(0), wallMs-snap.SavedAtMs))
	}
	s.clockAt, s.clockMs = now, ageMs
	s.clockFloorMs = max(s.clockFloorMs, ageMs)
	s.pruneLocked(wallMs, s.clockFloorMs)
	return nil
}

// snapshotLoadError deliberately omits parse details and file contents from
// its diagnostic. errors.Is still exposes the load/recovery failure category.
type snapshotLoadError struct {
	path  string
	cause error
}

func (e *snapshotLoadError) Error() string {
	return "performancestore: snapshot quarantine failed; persistence disabled: " + e.path
}

func (e *snapshotLoadError) Unwrap() error { return e.cause }

// quarantineSnapshot reserves a previously unused sibling name before Rename.
// Rename replaces only this operation's empty reservation, never an earlier
// quarantined file. This is a move, not a byte copy, on every platform.
func quarantineSnapshot(path string, stamp int64) (string, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", errors.New("snapshot path is not a regular file")
	}
	for {
		target := fmt.Sprintf("%s.corrupt-%d", path, stamp)
		reservation, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if errors.Is(err, fs.ErrExist) && stamp < math.MaxInt64 {
			stamp++
			continue
		}
		if err != nil {
			return "", err
		}
		if err := reservation.Close(); err != nil {
			_ = os.Remove(target)
			return "", err
		}
		if err := os.Rename(path, target); err != nil {
			_ = os.Remove(target)
			return "", err
		}
		return target, nil
	}
}

// Save publishes the live samples as a new snapshot revision when they differ
// from the published one, through durablefile. The revision is the larger of
// the previous one plus one and the current Unix time in milliseconds. A valid
// loaded snapshot preserves that floor even under wall rollback. A lost or
// corrupt file loses its floor; wall time alone cannot guarantee recovery order.
//
// A failure before the replacement leaves the file and the published snapshot
// unchanged and the store dirty. A failure after the replacement wraps
// durablefile.ErrReplaced: the file holds the new revision, so it is adopted
// and published while the durability failure is returned.
func (s *Store) Save() error {
	return s.save(false)
}

// Checkpoint persists a new revision even when no samples changed. Callers
// rebasing source ages after a transport restart must checkpoint before freezing
// and publishing the new summary. Its replacement/error semantics match Save.
// Like Save, it never publishes when persistence is disabled.
func (s *Store) Checkpoint() error {
	return s.save(true)
}

func (s *Store) save(checkpoint bool) error {
	if s.path == "" {
		return nil
	}
	s.saveMu.Lock()
	defer s.saveMu.Unlock()

	s.mu.Lock()
	wallMs, ageMs := s.clockLocked()
	s.pruneLocked(wallMs, ageMs)
	if !checkpoint && s.changes == s.savedChanges {
		s.mu.Unlock()
		return nil
	}
	snap, data, err := s.encodeLocked(wallMs, ageMs)
	changes := s.changes
	s.mu.Unlock()
	if err != nil {
		return err
	}

	err = s.write(data)
	if err != nil && !errors.Is(err, durablefile.ErrReplaced) {
		return fmt.Errorf("performancestore: save revision %d: %w", snap.Revision, err)
	}
	s.mu.Lock()
	s.published = snap
	s.savedChanges = changes
	s.mu.Unlock()
	if err != nil {
		return fmt.Errorf("performancestore: save revision %d: %w", snap.Revision, err)
	}
	return nil
}

// encodeLocked assigns the next revision and encodes the live samples, first
// evicting the least recently measured keys until the file fits MaxFileBytes.
// Caller holds s.mu.
func (s *Store) encodeLocked(wallMs, ageMs int64) (*snapshot, []byte, error) {
	if s.issued == math.MaxInt64 {
		return nil, nil, ErrRevisionExhausted
	}
	rev := max(s.issued+1, wallMs)
	s.issued = rev
	snap := &snapshot{
		SchemaVersion: FileSchemaVersion,
		Host:          s.host,
		Revision:      rev,
		SavedAtMs:     wallMs,
		AgeClockMs:    ageMs,
		Keys:          make([]snapshotKey, 0, len(s.keys)),
	}
	for k, st := range s.keys {
		snap.Keys = append(snap.Keys, snapshotKey{Key: k, Profile: st.profile, Samples: slices.Clone(st.samples)})
	}
	// Most recently measured first, so eviction takes from the tail.
	sort.Slice(snap.Keys, func(i, j int) bool {
		a, b := newestObserved(snap.Keys[i].Samples), newestObserved(snap.Keys[j].Samples)
		if a != b {
			return a > b
		}
		return keyLess(snap.Keys[i].Key, snap.Keys[j].Key)
	})

	envelope := *snap
	envelope.Keys = []snapshotKey{}
	size := encodedLen(envelope) + 1 // trailing newline
	for i, k := range snap.Keys {
		n := encodedLen(k)
		if i > 0 {
			n++ // separating comma
		}
		if size+n > MaxFileBytes {
			for _, evicted := range snap.Keys[i:] {
				delete(s.keys, evicted.Key)
				s.stats.EvictedKeys++
				s.changes++
			}
			snap.Keys = snap.Keys[:i]
			break
		}
		size += n
	}

	data, err := json.Marshal(snap)
	if err != nil {
		return nil, nil, fmt.Errorf("performancestore: encode snapshot: %w", err)
	}
	data = append(data, '\n')
	if len(data) > MaxFileBytes {
		return nil, nil, fmt.Errorf("performancestore: snapshot is %d bytes, over %d", len(data), MaxFileBytes)
	}
	return snap, data, nil
}

func (s *Store) write(data []byte) error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	return s.replacer.WriteFile(s.path, data, 0o600)
}

// readSnapshot reads and validates the snapshot at path for host. It reads at
// most one byte past MaxFileBytes.
func readSnapshot(path, host string) (*snapshot, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, MaxFileBytes+1))
	if err != nil {
		return nil, fmt.Errorf("performancestore: read %s: %w", path, err)
	}
	if len(data) > MaxFileBytes {
		return nil, fmt.Errorf("%w: %s is over %d bytes", ErrOversize, path, MaxFileBytes)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var snap snapshot
	if err := dec.Decode(&snap); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorrupt, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("%w: trailing data", ErrCorrupt)
	}
	if err := snap.validate(host); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCorrupt, err)
	}
	return &snap, nil
}

func (sn *snapshot) validate(host string) error {
	if sn.SchemaVersion != FileSchemaVersion {
		return fmt.Errorf("schema version %d is not supported", sn.SchemaVersion)
	}
	if sn.Host != host {
		return errors.New("snapshot belongs to another host")
	}
	if sn.Revision <= 0 {
		return errors.New("revision must be positive")
	}
	if sn.SavedAtMs < 0 || sn.AgeClockMs < 0 || (sn.SavedAtMs == 0) != (sn.AgeClockMs == 0) || sn.AgeClockMs < sn.SavedAtMs {
		return errors.New("snapshot has an invalid age clock")
	}
	if len(sn.Keys) > performancewire.MaxEntries {
		return fmt.Errorf("%d keys exceed %d", len(sn.Keys), performancewire.MaxEntries)
	}
	seen := make(map[performancewire.ProfileKey]bool, len(sn.Keys))
	for i, k := range sn.Keys {
		if err := k.Key.Validate(); err != nil {
			return fmt.Errorf("key %d: %w", i, err)
		}
		if k.Key.Host != host || seen[k.Key] {
			return fmt.Errorf("key %d is foreign or repeated", i)
		}
		seen[k.Key] = true
		if err := k.Profile.Validate(); err != nil {
			return fmt.Errorf("key %d: %w", i, err)
		}
		if len(k.Samples) == 0 || len(k.Samples) > performancewire.MaxSamplesPerKey {
			return fmt.Errorf("key %d has %d samples, outside 1..%d", i, len(k.Samples), performancewire.MaxSamplesPerKey)
		}
		for j, r := range k.Samples {
			if r.ObservedAtMs <= 0 {
				return fmt.Errorf("key %d sample %d has no observation time", i, j)
			}
			if r.AgeAtMs < 0 || (r.AgeAtMs > 0 && (sn.AgeClockMs == 0 || r.AgeAtMs > sn.AgeClockMs)) {
				return fmt.Errorf("key %d sample %d has an invalid age clock", i, j)
			}
			if err := r.Measurement.Validate(); err != nil {
				return fmt.Errorf("key %d sample %d: %w", i, j, err)
			}
		}
	}
	return nil
}
