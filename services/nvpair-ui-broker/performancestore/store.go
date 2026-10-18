// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package performancestore is the broker's bounded store of its own host's
// inference performance samples. It retains at most performancewire.MaxEntries
// profile keys of performancewire.MaxSamplesPerKey samples each, for at most
// performancewire.RetentionMs, and aggregates them into
// performancewire.Summary entries. It persists revisioned snapshots of the
// samples; only a saved snapshot is published to peers.
//
// The store never blocks inference: a rejected sample or failed save is
// reported to the caller and leaves the live local view working.
package performancestore

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"sync"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/performancewire"
)

// maxProducers bounds the live producer incarnations whose last sequence the
// store remembers for duplicate detection.
const maxProducers = 16

// ErrDuplicate reports a sample whose sequence its producer already reported.
var ErrDuplicate = errors.New("performancestore: duplicate sample")

// record is one retained sample: the store's receipt time in Unix
// milliseconds and what the sample measured.
type record struct {
	ObservedAtMs int64 `json:"observedAtMs"`
	// AgeAtMs is receipt on the nondecreasing age clock. Zero identifies an
	// older snapshot whose receipt used ObservedAtMs for both clocks.
	AgeAtMs int64 `json:"ageAtMs,omitempty"`
	performancewire.Measurement
}

func (r record) ageAtMs() int64 {
	if r.AgeAtMs != 0 {
		return r.AgeAtMs
	}
	return r.ObservedAtMs
}

// keyState is one profile key's retained samples, oldest received first.
type keyState struct {
	profile performancewire.Profile
	samples []record
}

// Stats reports the store's size and what it dropped since it started.
type Stats struct {
	Keys    int
	Samples int
	// Dirty is true when the live samples differ from the published snapshot.
	Dirty bool
	// PublishedRevision is the revision of the last saved or loaded snapshot,
	// or 0 when none exists.
	PublishedRevision int64
	// HistoryReset is true after Load successfully quarantined unreadable
	// history. Persistence remains enabled for a fresh empty snapshot.
	HistoryReset bool
	// EvictedKeys counts keys dropped to admit a new key or to fit the file.
	EvictedKeys uint64
	// EvictedSamples counts samples dropped when a key's ring was full.
	EvictedSamples uint64
	// ExpiredSamples counts samples dropped as outside the retention window.
	ExpiredSamples uint64
	// Duplicates counts samples ignored as already reported.
	Duplicates uint64
}

// Store aggregates one host's performance samples. Its zero value is not
// usable; construct it with New.
type Store struct {
	host string
	path string
	now  func() time.Time
	// replacer publishes snapshots; tests inject failures through it.
	replacer durablefile.Replacer

	// saveMu serializes Save so revisions are adopted in order.
	saveMu sync.Mutex

	mu        sync.Mutex
	keys      map[performancewire.ProfileKey]*keyState
	producers map[string]producerState
	// clockAt anchors elapsed time (monotonic when available); clockMs is
	// its age-clock value and clockFloorMs prevents wall rollback rejuvenation.
	clockAt      time.Time
	clockMs      int64
	clockFloorMs int64
	// changes counts live mutations; savedChanges is its value at the last
	// adopted snapshot, so the store is dirty when they differ.
	changes      uint64
	savedChanges uint64
	// seen orders producers by their last accepted sample.
	seen uint64
	// issued is the largest revision assigned, adopted or not.
	issued    int64
	published *snapshot
	stats     Stats
}

type producerState struct {
	seq      uint64
	lastSeen uint64
}

// New returns an empty store for the measuring host. A nonempty path enables
// Load and Save of snapshots at that file; with an empty path the store is
// memory-only and never publishes.
func New(host, path string) *Store {
	return &Store{
		host:      host,
		path:      path,
		now:       time.Now,
		keys:      make(map[performancewire.ProfileKey]*keyState),
		producers: make(map[string]producerState),
		// A store that has published nothing is dirty, so its first save
		// publishes an explicit empty snapshot.
		changes: 1,
	}
}

// Record validates and retains one sample, stamped with the store's clock. It
// returns ErrDuplicate, ignoring the sample, when its producer already
// reported that sequence. A full ring drops its oldest sample, and a new key
// beyond the cap evicts the key whose newest sample is oldest. The store takes
// ownership of the sample's pointers; the caller must not modify them after.
func (s *Store) Record(sample performancewire.Sample) error {
	if err := sample.Validate(); err != nil {
		return fmt.Errorf("performancestore: %w", err)
	}
	if sample.Key.Host != s.host {
		return errors.New("performancestore: sample was measured on another host")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.acceptSeqLocked(sample.Producer, sample.Seq); err != nil {
		return err
	}
	st := s.keys[sample.Key]
	if st == nil {
		if len(s.keys) >= performancewire.MaxEntries {
			s.evictLeastRecentLocked()
		}
		st = &keyState{}
		s.keys[sample.Key] = st
	}
	st.profile = sample.Profile
	if len(st.samples) >= performancewire.MaxSamplesPerKey {
		st.samples = slices.Delete(st.samples, 0, len(st.samples)-performancewire.MaxSamplesPerKey+1)
		s.stats.EvictedSamples++
	}
	wallMs, ageMs := s.clockLocked()
	st.samples = append(st.samples, record{ObservedAtMs: wallMs, AgeAtMs: ageMs, Measurement: sample.Measurement})
	s.changes++
	return nil
}

// acceptSeqLocked admits a sequence above the producer's last one. It forgets
// the least recently seen producer beyond maxProducers. Caller holds s.mu.
func (s *Store) acceptSeqLocked(producer string, seq uint64) error {
	p, ok := s.producers[producer]
	if ok && seq <= p.seq {
		s.stats.Duplicates++
		return ErrDuplicate
	}
	if !ok && len(s.producers) >= maxProducers {
		oldest := ""
		for id, q := range s.producers {
			if oldest == "" || q.lastSeen < s.producers[oldest].lastSeen {
				oldest = id
			}
		}
		delete(s.producers, oldest)
	}
	s.seen++
	s.producers[producer] = producerState{seq: seq, lastSeen: s.seen}
	return nil
}

// evictLeastRecentLocked drops the key whose newest sample is oldest. Caller
// holds s.mu.
func (s *Store) evictLeastRecentLocked() {
	var victim performancewire.ProfileKey
	found := false
	var victimNewest int64
	for k, st := range s.keys {
		n := newestObserved(st.samples)
		if !found || n < victimNewest || (n == victimNewest && keyLess(k, victim)) {
			victim, victimNewest, found = k, n, true
		}
	}
	if found {
		delete(s.keys, victim)
		s.stats.EvictedKeys++
		s.changes++
	}
}

// Local returns the live aggregates, including samples not yet saved, for the
// local display. Its Revision is 0 and its AdmissionIncarnation empty: it is
// not a publishable summary.
func (s *Store) Local() performancewire.Summary {
	s.mu.Lock()
	defer s.mu.Unlock()
	wallMs, ageMs := s.clockLocked()
	return buildSummary(s.host, "", 0, s.keys, wallMs, ageMs)
}

// clockLocked returns wall time for observation metadata and a nondecreasing
// clock for ages. The fixed anchor retains sub-millisecond elapsed time between
// reads. A wall step forward may make samples older; a step back cannot make
// them younger. Caller holds s.mu.
func (s *Store) clockLocked() (int64, int64) {
	now := s.now()
	wallMs := now.UnixMilli()
	if s.clockAt.IsZero() {
		s.clockAt, s.clockMs = now, wallMs
	}
	elapsed := max(int64(0), now.Sub(s.clockAt).Milliseconds())
	ageMs := addElapsed(s.clockMs, elapsed)
	if wallMs > ageMs {
		s.clockAt, s.clockMs, ageMs = now, wallMs, wallMs
	}
	ageMs = max(s.clockFloorMs, ageMs)
	s.clockFloorMs = ageMs
	return wallMs, ageMs
}

// addElapsed saturates the age clock rather than overflowing into young ages.
func addElapsed(base, elapsed int64) int64 {
	if base > math.MaxInt64-elapsed {
		return math.MaxInt64
	}
	return base + elapsed
}

// Stats reports the store's current size and drop counters.
func (s *Store) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.stats
	st.Keys = len(s.keys)
	for _, k := range s.keys {
		st.Samples += len(k.samples)
	}
	st.Dirty = s.changes != s.savedChanges
	if s.published != nil {
		st.PublishedRevision = s.published.Revision
	}
	return st
}

// pruneLocked drops samples outside the retention window and keys left empty.
// Caller holds s.mu.
func (s *Store) pruneLocked(wallMs, ageMs int64) {
	for k, st := range s.keys {
		kept := st.samples[:0]
		for _, r := range st.samples {
			if retained(r, wallMs, ageMs) {
				kept = append(kept, r)
			}
		}
		if dropped := len(st.samples) - len(kept); dropped > 0 {
			s.stats.ExpiredSamples += uint64(dropped)
			s.changes++
		}
		st.samples = kept
		if len(kept) == 0 {
			delete(s.keys, k)
		}
	}
}

// retained reports whether a sample is inside the retention window. A sample
// stamped later than now is kept as clock-uncertain unless it lies a whole
// window in the future, where it would otherwise never age out.
func retained(r record, wallMs, ageMs int64) bool {
	age := ageMs - r.ageAtMs()
	return age <= performancewire.RetentionMs && age >= -performancewire.RetentionMs &&
		wallMs-r.ObservedAtMs >= -performancewire.RetentionMs
}

func newestObserved(samples []record) int64 {
	var n int64
	for _, r := range samples {
		n = max(n, r.ObservedAtMs)
	}
	return n
}

func keyLess(a, b performancewire.ProfileKey) bool {
	if a.Host != b.Host {
		return a.Host < b.Host
	}
	if a.Engine != b.Engine {
		return a.Engine < b.Engine
	}
	if a.Model != b.Model {
		return a.Model < b.Model
	}
	return a.Fingerprint < b.Fingerprint
}

// aggregate computes one key's entry from its samples inside the window. It
// reports false when no sample is retained.
//
// Latency distributions use completed samples whose protocol end was not
// known to be missing. Phase rates additionally require a known protocol end,
// a known token count and a positive matching phase duration, and are
// token-weighted. Unknown values are skipped, never counted as zero.
func aggregate(key performancewire.ProfileKey, st *keyState, wallMs, ageMs int64) (performancewire.Entry, bool) {
	e := performancewire.Entry{Key: key, Profile: st.profile}
	var admission, header, firstBody, firstToken, total, load []int64
	var prefill, decode rateSum
	for _, r := range st.samples {
		if !retained(r, wallMs, ageMs) {
			continue
		}
		age := ageMs - r.ageAtMs()
		if age < 0 {
			age = 0
			e.ClockUncertain = true
		}
		if wallMs < r.ObservedAtMs {
			e.ClockUncertain = true
		}
		if e.Samples == 0 {
			e.NewestAgeMs, e.OldestAgeMs = age, age
		}
		e.NewestAgeMs = min(e.NewestAgeMs, age)
		e.OldestAgeMs = max(e.OldestAgeMs, age)
		e.NewestObservedAtMs = max(e.NewestObservedAtMs, r.ObservedAtMs)
		e.Samples++

		switch r.Outcome {
		case performancewire.OutcomeFailed:
			e.Failed++
			continue
		case performancewire.OutcomeCancelled:
			e.Cancelled++
			continue
		}
		e.Completed++
		if r.ProtocolComplete != nil && !*r.ProtocolComplete {
			e.Incomplete++
			continue
		}
		admission = appendKnown(admission, r.AdmissionWaitMs)
		header = appendKnown(header, r.HeaderMs)
		firstBody = appendKnown(firstBody, r.FirstBodyMs)
		firstToken = appendKnown(firstToken, r.FirstTokenMs)
		total = appendKnown(total, r.TotalMs)
		load = appendKnown(load, r.LoadNs)
		if r.ProtocolComplete != nil {
			prefill.add(r.PromptTokens, r.PrefillNs)
			decode.add(r.OutputTokens, r.DecodeNs)
		}
	}
	if e.Samples == 0 {
		return performancewire.Entry{}, false
	}
	e.AdmissionWaitMs = distribution(admission)
	e.HeaderMs = distribution(header)
	e.FirstBodyMs = distribution(firstBody)
	e.FirstTokenMs = distribution(firstToken)
	e.TotalMs = distribution(total)
	e.LoadNs = distribution(load)
	e.Prefill = prefill.rate()
	e.Decode = decode.rate()
	return e, true
}

func appendKnown(vs []int64, v *int64) []int64 {
	if v == nil {
		return vs
	}
	return append(vs, *v)
}

// distribution returns the nearest-rank median and 95th percentile, or nil
// when nothing was measured.
func distribution(vs []int64) *performancewire.Distribution {
	if len(vs) == 0 {
		return nil
	}
	slices.Sort(vs)
	return &performancewire.Distribution{
		Samples: len(vs),
		Median:  vs[nearestRank(50, len(vs))],
		P95:     vs[nearestRank(95, len(vs))],
	}
}

// nearestRank is the zero-based index of the p-th percentile of n sorted values.
func nearestRank(p, n int) int {
	return (p*n+99)/100 - 1
}

type rateSum struct {
	samples int
	tokens  int64
	ns      int64
}

func (r *rateSum) add(tokens, ns *int64) {
	if tokens == nil || ns == nil || *ns <= 0 {
		return
	}
	r.samples++
	r.tokens += *tokens
	r.ns += *ns
}

func (r rateSum) rate() *performancewire.Rate {
	if r.samples == 0 {
		return nil
	}
	return &performancewire.Rate{
		Samples:         r.samples,
		Tokens:          r.tokens,
		DurationNs:      r.ns,
		TokensPerSecond: performancewire.TokensPerSecond(r.tokens, r.ns),
	}
}

// buildSummary aggregates keys into a summary, most recently measured key
// first. It omits the least recent entries that would take the encoded summary
// over performancewire.MaxPayloadBytes, counting them in OmittedEntries.
func buildSummary(host, incarnation string, revision int64, keys map[performancewire.ProfileKey]*keyState, wallMs, ageMs int64) performancewire.Summary {
	entries := make([]performancewire.Entry, 0, len(keys))
	for k, st := range keys {
		if e, ok := aggregate(k, st, wallMs, ageMs); ok {
			entries = append(entries, e)
		}
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].NewestObservedAtMs != entries[j].NewestObservedAtMs {
			return entries[i].NewestObservedAtMs > entries[j].NewestObservedAtMs
		}
		return keyLess(entries[i].Key, entries[j].Key)
	})
	sum := performancewire.Summary{
		Version:              performancewire.SchemaVersion,
		MeasuringHost:        host,
		AdmissionIncarnation: incarnation,
		Revision:             revision,
		Window: performancewire.Window{
			MaxAgeMs:         performancewire.RetentionMs,
			MaxSamplesPerKey: performancewire.MaxSamplesPerKey,
		},
		Entries: entries,
	}
	fitPayload(&sum)
	return sum
}

// fitPayload trims trailing entries until the encoded summary fits. The
// envelope is measured with the largest omitted count it could carry.
func fitPayload(sum *performancewire.Summary) {
	envelope := *sum
	envelope.Entries = []performancewire.Entry{}
	envelope.OmittedEntries = performancewire.MaxEntries
	size := encodedLen(envelope)
	keep := 0
	for _, e := range sum.Entries {
		n := encodedLen(e)
		if keep > 0 {
			n++ // separating comma
		}
		if size+n > performancewire.MaxPayloadBytes {
			break
		}
		size += n
		keep++
	}
	sum.OmittedEntries = len(sum.Entries) - keep
	sum.Entries = sum.Entries[:keep]
}

// encodedLen is v's JSON length, or more than any limit when it cannot be
// encoded, so the caller leaves it out.
func encodedLen(v any) int {
	data, err := json.Marshal(v)
	if err != nil {
		return math.MaxInt32
	}
	return len(data)
}
