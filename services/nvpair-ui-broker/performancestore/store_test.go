// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package performancestore

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"nvpair-shared/performancewire"
)

const testHost = "host-a"

func ptr[T any](v T) *T { return &v }

// clock is a settable store clock.
type clock struct{ t time.Time }

func (c *clock) now() time.Time { return c.t }

func (c *clock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newTestStore(t *testing.T, path string) (*Store, *clock) {
	t.Helper()
	c := &clock{t: time.UnixMilli(1_760_000_000_000)}
	s := New(testHost, path)
	s.now = c.now
	return s, c
}

func testKey(model string) performancewire.ProfileKey {
	return performancewire.ProfileKey{Host: testHost, Engine: "llamacpp-custom", Model: model, Fingerprint: "fp-1"}
}

// recorder records samples for one producer with increasing sequences.
type recorder struct {
	t   *testing.T
	s   *Store
	seq uint64
}

func (r *recorder) record(key performancewire.ProfileKey, m performancewire.Measurement) {
	r.t.Helper()
	r.seq++
	err := r.s.Record(performancewire.Sample{
		Scope:       performancewire.ScopeLocalExecution,
		Producer:    "proxy-1",
		Seq:         r.seq,
		Key:         key,
		Measurement: m,
	})
	if err != nil {
		r.t.Fatalf("record seq %d: %v", r.seq, err)
	}
}

func completed() performancewire.Measurement {
	return performancewire.Measurement{Outcome: performancewire.OutcomeCompleted, ProtocolComplete: ptr(true)}
}

// onlyEntry returns the single entry of the store's local summary.
func onlyEntry(t *testing.T, s *Store) performancewire.Entry {
	t.Helper()
	sum := s.Local()
	if len(sum.Entries) != 1 {
		t.Fatalf("local summary has %d entries, want 1", len(sum.Entries))
	}
	return sum.Entries[0]
}

func TestMissingMetricsLeaveAggregatesUnknown(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	r.record(testKey("m"), performancewire.Measurement{Outcome: performancewire.OutcomeCompleted})

	e := onlyEntry(t, s)
	if e.Samples != 1 || e.Completed != 1 {
		t.Fatalf("counts = samples %d completed %d, want 1 and 1", e.Samples, e.Completed)
	}
	if e.AdmissionWaitMs != nil || e.HeaderMs != nil || e.FirstBodyMs != nil || e.FirstTokenMs != nil ||
		e.TotalMs != nil || e.LoadNs != nil || e.Prefill != nil || e.Decode != nil || e.EngineDecode != nil {
		t.Fatalf("unmeasured metrics produced aggregates: %+v", e)
	}
}

func TestMeasuredZeroIsNotUnknown(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	zero := completed()
	zero.FirstBodyMs = ptr[int64](0)
	zero.OutputTokens = ptr[int64](0)
	zero.DecodeNs = ptr[int64](500_000_000)
	r.record(testKey("m"), zero)
	r.record(testKey("m"), completed()) // measured nothing

	e := onlyEntry(t, s)
	if e.FirstBodyMs == nil || *e.FirstBodyMs != (performancewire.Distribution{Samples: 1, Median: 0, P95: 0}) {
		t.Fatalf("firstBodyMs = %+v, want one measured zero", e.FirstBodyMs)
	}
	if e.Decode == nil || e.Decode.Samples != 1 || e.Decode.Tokens != 0 || e.Decode.TokensPerSecond != 0 {
		t.Fatalf("decode = %+v, want one measured zero rate", e.Decode)
	}
	if e.TotalMs != nil {
		t.Fatalf("totalMs = %+v, want unknown", e.TotalMs)
	}
}

func TestIncompleteSamplesAreExcludedFromThroughput(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	withDecode := func(m performancewire.Measurement) performancewire.Measurement {
		m.OutputTokens = ptr[int64](1000)
		m.DecodeNs = ptr[int64](1_000_000_000)
		m.FirstBodyMs = ptr[int64](10)
		return m
	}
	good := withDecode(completed())
	good.OutputTokens = ptr[int64](50)
	r.record(testKey("m"), good)
	truncated := withDecode(completed())
	truncated.ProtocolComplete = ptr(false)
	r.record(testKey("m"), truncated)
	unknownEnd := withDecode(completed())
	unknownEnd.ProtocolComplete = nil
	r.record(testKey("m"), unknownEnd)
	r.record(testKey("m"), withDecode(performancewire.Measurement{Outcome: performancewire.OutcomeFailed}))
	r.record(testKey("m"), withDecode(performancewire.Measurement{Outcome: performancewire.OutcomeCancelled}))
	zeroPhase := completed()
	zeroPhase.OutputTokens = ptr[int64](10)
	zeroPhase.DecodeNs = ptr[int64](0)
	r.record(testKey("m"), zeroPhase)

	e := onlyEntry(t, s)
	if e.Samples != 6 || e.Completed != 4 || e.Incomplete != 1 || e.Failed != 1 || e.Cancelled != 1 {
		t.Fatalf("counts = %+v", e)
	}
	if e.Decode == nil || e.Decode.Samples != 1 || e.Decode.Tokens != 50 || e.Decode.TokensPerSecond != 50 {
		t.Fatalf("decode = %+v, want only the complete sample", e.Decode)
	}
	// Latency still counts a completed sample whose protocol end is unknown.
	if e.FirstBodyMs == nil || e.FirstBodyMs.Samples != 2 {
		t.Fatalf("firstBodyMs = %+v, want the two not-incomplete completed samples", e.FirstBodyMs)
	}
}

func TestPhaseRatesAreTokenWeighted(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	add := func(prompt, prefillNs, output, decodeNs int64) {
		m := completed()
		m.PromptTokens, m.PrefillNs = ptr(prompt), ptr(prefillNs)
		m.OutputTokens, m.DecodeNs = ptr(output), ptr(decodeNs)
		r.record(testKey("m"), m)
	}
	// Per-sample decode rates 100/s and 500/s average to 300/s; weighting by
	// tokens gives 1100 tokens over 3 s.
	add(1000, 1_000_000_000, 100, 1_000_000_000)
	add(3000, 1_000_000_000, 1000, 2_000_000_000)

	e := onlyEntry(t, s)
	if e.Decode == nil || e.Decode.Tokens != 1100 || e.Decode.DurationNs != 3_000_000_000 ||
		math.Abs(e.Decode.TokensPerSecond-1100.0/3) > 1e-9 {
		t.Fatalf("decode = %+v, want 1100 tokens over 3 s", e.Decode)
	}
	if e.Prefill == nil || e.Prefill.Samples != 2 || e.Prefill.TokensPerSecond != 2000 {
		t.Fatalf("prefill = %+v, want 4000 tokens over 2 s", e.Prefill)
	}
}

func TestEngineDecodeWeightsReportedRatesByDuration(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	add := func(output, duration int64, rate float64) {
		m := completed()
		m.OutputTokens = ptr(output)
		m.EngineDecodeNs = ptr(duration)
		m.EngineDecodeTokensPerSecond = ptr(rate)
		r.record(testKey("m"), m)
	}
	add(20, 400_000_000, 55)
	add(200, 600_000_000, 45)

	e := onlyEntry(t, s)
	want := performancewire.ReportedRate{Samples: 2, DurationNs: 1_000_000_000, TokensPerSecond: 49}
	if e.EngineDecode == nil || *e.EngineDecode != want {
		t.Fatalf("engineDecode = %+v, want %+v independent of output tokens", e.EngineDecode, want)
	}
	if e.Decode != nil {
		t.Fatalf("legacy decode = %+v, want unknown without legacy intervals", e.Decode)
	}
}

func TestEngineDecodeExcludesUnqualifiedAndUnknownRates(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*performancewire.Measurement)
	}{
		{"failed", func(m *performancewire.Measurement) { m.Outcome = performancewire.OutcomeFailed }},
		{"cancelled", func(m *performancewire.Measurement) { m.Outcome = performancewire.OutcomeCancelled }},
		{"incomplete", func(m *performancewire.Measurement) { m.ProtocolComplete = ptr(false) }},
		{"unknown protocol end", func(m *performancewire.Measurement) { m.ProtocolComplete = nil }},
		{"unknown duration", func(m *performancewire.Measurement) { m.EngineDecodeNs = nil }},
		{"zero duration", func(m *performancewire.Measurement) { m.EngineDecodeNs = ptr[int64](0) }},
		{"unknown rate", func(m *performancewire.Measurement) { m.EngineDecodeTokensPerSecond = nil }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, _ := newTestStore(t, "")
			r := &recorder{t: t, s: s}
			m := completed()
			m.EngineDecodeNs = ptr[int64](400_000_000)
			m.EngineDecodeTokensPerSecond = ptr(55.0)
			tt.mutate(&m)
			r.record(testKey("m"), m)
			if e := onlyEntry(t, s); e.EngineDecode != nil {
				t.Fatalf("engineDecode = %+v, want unknown", e.EngineDecode)
			}
		})
	}
}

func TestEngineDecodePreservesMeasuredZero(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	m := completed()
	m.EngineDecodeNs = ptr[int64](400_000_000)
	m.EngineDecodeTokensPerSecond = ptr(0.0)
	r.record(testKey("m"), m)
	r.record(testKey("m"), completed())
	want := performancewire.ReportedRate{Samples: 1, DurationNs: 400_000_000, TokensPerSecond: 0}
	if e := onlyEntry(t, s); e.EngineDecode == nil || *e.EngineDecode != want {
		t.Fatalf("engineDecode = %+v, want measured zero %+v", e.EngineDecode, want)
	}
}

func TestRecordOwnsReportedRatePointers(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	m := completed()
	m.EngineDecodeNs = ptr[int64](400_000_000)
	m.EngineDecodeTokensPerSecond = ptr(55.0)
	r.record(testKey("m"), m)
	*m.EngineDecodeNs = 1
	*m.EngineDecodeTokensPerSecond = 0
	*m.ProtocolComplete = false
	want := performancewire.ReportedRate{Samples: 1, DurationNs: 400_000_000, TokensPerSecond: 55}
	if e := onlyEntry(t, s); e.EngineDecode == nil || *e.EngineDecode != want {
		t.Fatalf("engineDecode after caller mutation = %+v, want %+v", e.EngineDecode, want)
	}
}

func TestEngineDecodeMaximumWindowRemainsValid(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	var duration int64
	for i := range performancewire.MaxSamplesPerKey {
		m := completed()
		// Different intervals exercise rounding in the weighted sum at the
		// maximum rate, where a slight overshoot would invalidate the summary.
		m.EngineDecodeNs = ptr(performancewire.MaxPhaseNs - int64(i)*123_456_789)
		m.EngineDecodeTokensPerSecond = ptr(performancewire.MaxTokensPerSecond)
		duration += *m.EngineDecodeNs
		r.record(testKey("m"), m)
	}
	e := onlyEntry(t, s)
	if e.EngineDecode == nil || e.EngineDecode.DurationNs != duration ||
		e.EngineDecode.TokensPerSecond != performancewire.MaxTokensPerSecond {
		t.Fatalf("maximum window reported rate = %+v", e.EngineDecode)
	}
	summary := s.Local()
	summary.AdmissionIncarnation, summary.Revision = "inc-1", 1
	if err := summary.Validate(); err != nil {
		t.Fatalf("maximum window summary is invalid: %v", err)
	}
}

func TestLatencyPercentilesUseNearestRank(t *testing.T) {
	test := func(name string, values []int64, wantMedian, wantP95 int64) {
		t.Run(name, func(t *testing.T) {
			s, _ := newTestStore(t, "")
			r := &recorder{t: t, s: s}
			for _, v := range values {
				m := completed()
				m.TotalMs = ptr(v)
				r.record(testKey("m"), m)
			}
			got := onlyEntry(t, s).TotalMs
			want := performancewire.Distribution{Samples: len(values), Median: wantMedian, P95: wantP95}
			if got == nil || *got != want {
				t.Fatalf("totalMs = %+v, want %+v", got, want)
			}
		})
	}
	test("one sample", []int64{42}, 42, 42)
	test("two samples take the lower median", []int64{20, 10}, 10, 20)
	twenty := make([]int64, 0, 20)
	for i := int64(20); i >= 1; i-- {
		twenty = append(twenty, i)
	}
	test("twenty samples", twenty, 10, 19)
}

func TestRingKeepsNewestSamples(t *testing.T) {
	s, _ := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	for i := range performancewire.MaxSamplesPerKey + 6 {
		m := completed()
		m.TotalMs = ptr(int64(i))
		r.record(testKey("m"), m)
	}
	st := s.keys[testKey("m")]
	if len(st.samples) != performancewire.MaxSamplesPerKey || *st.samples[0].TotalMs != 6 {
		t.Fatalf("ring holds %d samples starting at %d, want %d starting at 6",
			len(st.samples), *st.samples[0].TotalMs, performancewire.MaxSamplesPerKey)
	}
	if got := s.Stats().EvictedSamples; got != 6 {
		t.Fatalf("evicted samples = %d, want 6", got)
	}
}

func TestKeyCapEvictsLeastRecentlyMeasuredKey(t *testing.T) {
	s, c := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	for i := range performancewire.MaxEntries {
		r.record(testKey(fmt.Sprintf("m-%d", i)), completed())
		c.advance(time.Millisecond)
	}
	r.record(testKey("m-0"), completed()) // m-1 is now the least recent
	r.record(testKey("new"), completed())

	if _, ok := s.keys[testKey("m-1")]; ok {
		t.Fatal("least recently measured key m-1 survived")
	}
	for _, model := range []string{"m-0", "m-2", "new"} {
		if _, ok := s.keys[testKey(model)]; !ok {
			t.Fatalf("key %s was evicted", model)
		}
	}
	st := s.Stats()
	if st.Keys != performancewire.MaxEntries || st.EvictedKeys != 1 {
		t.Fatalf("stats = %+v, want %d keys and one eviction", st, performancewire.MaxEntries)
	}
}

func TestDuplicateSequenceIsIgnored(t *testing.T) {
	s, _ := newTestStore(t, "")
	sample := performancewire.Sample{
		Scope:       performancewire.ScopeLocalExecution,
		Producer:    "proxy-1",
		Seq:         5,
		Key:         testKey("m"),
		Measurement: completed(),
	}
	if err := s.Record(sample); err != nil {
		t.Fatalf("first record: %v", err)
	}
	for _, seq := range []uint64{5, 4} {
		sample.Seq = seq
		if err := s.Record(sample); !errors.Is(err, ErrDuplicate) {
			t.Fatalf("record seq %d = %v, want ErrDuplicate", seq, err)
		}
	}
	sample.Producer, sample.Seq = "proxy-2", 1
	if err := s.Record(sample); err != nil {
		t.Fatalf("record from a new producer: %v", err)
	}
	if st := s.Stats(); st.Samples != 2 || st.Duplicates != 2 {
		t.Fatalf("stats = %+v, want 2 samples and 2 duplicates", st)
	}
}

func TestRecordRejectsForeignAndInvalidSamples(t *testing.T) {
	s, _ := newTestStore(t, "")
	reject := func(name string, mutate func(*performancewire.Sample), want string) {
		t.Run(name, func(t *testing.T) {
			sample := performancewire.Sample{
				Scope:       performancewire.ScopeLocalExecution,
				Producer:    "proxy-1",
				Seq:         1,
				Key:         testKey("m"),
				Measurement: completed(),
			}
			mutate(&sample)
			err := s.Record(sample)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("Record() = %v, want error containing %q", err, want)
			}
		})
	}
	reject("another host", func(sm *performancewire.Sample) { sm.Key.Host = "host-b" }, "another host")
	reject("negative latency", func(sm *performancewire.Sample) { sm.TotalMs = ptr[int64](-1) }, "totalMs")
	if st := s.Stats(); st.Samples != 0 {
		t.Fatalf("rejected samples were retained: %+v", st)
	}
}

func TestClockAnomalies(t *testing.T) {
	t.Run("future sample is clock uncertain", func(t *testing.T) {
		s, c := newTestStore(t, "")
		r := &recorder{t: t, s: s}
		r.record(testKey("m"), completed())
		c.advance(-time.Hour) // wall clock stepped back
		e := onlyEntry(t, s)
		if !e.ClockUncertain || e.NewestAgeMs != 0 || e.OldestAgeMs != 0 {
			t.Fatalf("entry = %+v, want clock uncertain at age 0", e)
		}
	})
	t.Run("ages follow the clock", func(t *testing.T) {
		s, c := newTestStore(t, "")
		r := &recorder{t: t, s: s}
		r.record(testKey("m"), completed())
		c.advance(time.Minute)
		r.record(testKey("m"), completed())
		c.advance(time.Second)
		e := onlyEntry(t, s)
		if e.ClockUncertain || e.NewestAgeMs != 1000 || e.OldestAgeMs != 61_000 {
			t.Fatalf("entry = %+v, want ages 1000 and 61000", e)
		}
	})
	t.Run("sample past retention is excluded", func(t *testing.T) {
		s, c := newTestStore(t, "")
		r := &recorder{t: t, s: s}
		r.record(testKey("m"), completed())
		c.advance(time.Duration(performancewire.RetentionMs+1) * time.Millisecond)
		if n := len(s.Local().Entries); n != 0 {
			t.Fatalf("expired sample still aggregated into %d entries", n)
		}
	})
	t.Run("sample a whole window in the future is excluded", func(t *testing.T) {
		s, c := newTestStore(t, "")
		r := &recorder{t: t, s: s}
		r.record(testKey("m"), completed())
		c.advance(-time.Duration(performancewire.RetentionMs+1) * time.Millisecond)
		if n := len(s.Local().Entries); n != 0 {
			t.Fatalf("far-future sample still aggregated into %d entries", n)
		}
	})
}

func TestWallRollbackDoesNotRejuvenateLocalOrPublishedAges(t *testing.T) {
	s, c := newTestStore(t, snapshotPath(t))
	r := &recorder{t: t, s: s}
	r.record(testKey("m"), completed())
	c.advance(time.Minute)
	mustSave(t, s)
	c.advance(-30 * time.Second)
	if age := onlyEntry(t, s).NewestAgeMs; age != 60_000 {
		t.Fatalf("local age after rollback = %d, want 60000", age)
	}
	if age := mustPublished(t, s).Entries[0].NewestAgeMs; age != 60_000 {
		t.Fatalf("published age after rollback = %d, want 60000", age)
	}
}

func TestAgeClockAddsElapsedWhenAheadOfWallTime(t *testing.T) {
	s, _ := newTestStore(t, "")
	// Add preserves Go's monotonic reading. The logical age clock is ahead
	// of wall time, as it would be after loading a rollback checkpoint.
	anchor := time.Now()
	s.now = func() time.Time { return anchor }
	s.clockAt, s.clockMs = anchor, anchor.UnixMilli()+3_600_000
	r := &recorder{t: t, s: s}
	r.record(testKey("m"), completed())
	s.now = func() time.Time { return anchor.Add(time.Minute) }
	if age := onlyEntry(t, s).NewestAgeMs; age != 60_000 {
		t.Fatalf("elapsed age = %d, want 60000 despite wall time below age clock", age)
	}
}

func TestAgeClockRetainsSubMillisecondElapsed(t *testing.T) {
	s, _ := newTestStore(t, "")
	anchor := time.Now()
	s.now = func() time.Time { return anchor }
	s.clockAt, s.clockMs = anchor, anchor.UnixMilli()+3_600_000
	r := &recorder{t: t, s: s}
	r.record(testKey("m"), completed())
	for _, elapsed := range []time.Duration{400 * time.Microsecond, 800 * time.Microsecond, 1200 * time.Microsecond} {
		s.now = func() time.Time { return anchor.Add(elapsed) }
		s.Local()
	}
	if age := onlyEntry(t, s).NewestAgeMs; age != 1 {
		t.Fatalf("age after frequent reads = %d, want 1 millisecond", age)
	}
}

// maxMeasurement sets every metric to a wide value, the worst case for
// encoded size.
func maxMeasurement() performancewire.Measurement {
	m := completed()
	m.AdmissionWaitMs = ptr(performancewire.MaxLatencyMs)
	m.HeaderMs = ptr(performancewire.MaxLatencyMs)
	m.FirstBodyMs = ptr(performancewire.MaxLatencyMs)
	m.FirstTokenMs = ptr(performancewire.MaxLatencyMs)
	m.TotalMs = ptr(performancewire.MaxLatencyMs)
	m.PromptTokens = ptr(performancewire.MaxTokens)
	m.CachedPromptTokens = ptr(performancewire.MaxTokens)
	m.OutputTokens = ptr(performancewire.MaxTokens - 1)
	m.PrefillNs = ptr(performancewire.MaxPhaseNs)
	m.DecodeNs = ptr(performancewire.MaxPhaseNs - 7)
	m.LoadNs = ptr(performancewire.MaxPhaseNs)
	return m
}

// worstCaseKey is a distinct key with every identity field at its limit.
func worstCaseKey(i int) performancewire.ProfileKey {
	suffix := fmt.Sprintf("-%03d", i)
	return performancewire.ProfileKey{
		Host:        testHost,
		Engine:      strings.Repeat("e", performancewire.MaxIDBytes),
		Model:       strings.Repeat("m", performancewire.MaxModelBytes-len(suffix)) + suffix,
		Fingerprint: strings.Repeat("f", performancewire.MaxIDBytes),
	}
}

func TestSummaryOmitsEntriesToFitPayload(t *testing.T) {
	s, c := newTestStore(t, "")
	r := &recorder{t: t, s: s}
	for i := range performancewire.MaxEntries {
		r.record(worstCaseKey(i), maxMeasurement())
		c.advance(time.Millisecond)
	}
	sum := s.Local()
	sum.Revision, sum.AdmissionIncarnation = 1, strings.Repeat("i", performancewire.MaxIDBytes)
	if sum.OmittedEntries == 0 {
		t.Fatal("worst-case summary omitted nothing; the payload limit is not exercised")
	}
	if len(sum.Entries)+sum.OmittedEntries != performancewire.MaxEntries {
		t.Fatalf("entries %d + omitted %d != %d", len(sum.Entries), sum.OmittedEntries, performancewire.MaxEntries)
	}
	if sum.Entries[0].Key != worstCaseKey(performancewire.MaxEntries-1) {
		t.Fatalf("first entry %s is not the most recently measured key", sum.Entries[0].Key.Model)
	}
	data, err := performancewire.EncodeSummary(sum)
	if err != nil {
		t.Fatalf("encode worst-case summary: %v", err)
	}
	if _, err := performancewire.DecodeSummary(data); err != nil {
		t.Fatalf("decode worst-case summary: %v", err)
	}
}
