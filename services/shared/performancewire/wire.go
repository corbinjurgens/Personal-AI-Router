// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package performancewire defines the shared contract for inference
// performance telemetry: one destination-local execution sample, the profile
// identity samples are aggregated under, and the full-summary message a
// measuring host publishes to its admitted peers.
//
// Every measurement is nullable. A nil pointer means the value is unknown; a
// pointer to zero is a measured zero. Consumers must never read an unknown
// value, an absent aggregate or a stale summary as zero latency or as a fast
// node.
//
// A Summary is a full replacement of the measuring host's own aggregates at one
// persisted revision, never a delta to add to a receiver's totals, so a
// duplicate, retried or heartbeat summary cannot multiply samples.
package performancewire

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"unicode"
	"unicode/utf8"
)

// MethodSummary is the peer message that carries a Summary.
const MethodSummary = "performance:summary"

// SchemaVersion is the Summary version this package produces and accepts.
const SchemaVersion = 2

// Validation limits. They bound what a receiver caches per peer and what a
// local store retains; they are contract values, not settings.
const (
	// MaxPayloadBytes bounds one encoded Summary.
	MaxPayloadBytes = 128 << 10
	// MaxEntries bounds the profile keys in one Summary and in a local store.
	MaxEntries = 128
	// MaxSamplesPerKey bounds the samples one key's aggregates are computed from.
	MaxSamplesPerKey = 64
	// RetentionMs is the longest a sample contributes to aggregates: seven days.
	RetentionMs int64 = 7 * 24 * 60 * 60 * 1000

	// MaxIDBytes bounds identifiers: hosts, engine IDs, fingerprints,
	// producers and admission incarnations.
	MaxIDBytes = 128
	// MaxModelBytes bounds a canonical model identity.
	MaxModelBytes = 256
	// MaxLabelBytes bounds a display-only label such as a quantization.
	MaxLabelBytes = 64

	// MaxLatencyMs bounds one observed latency or duration: one day.
	MaxLatencyMs int64 = 24 * 60 * 60 * 1000
	// MaxPhaseNs bounds one engine-reported phase or load duration: one day.
	MaxPhaseNs int64 = 24 * 60 * 60 * 1000 * 1000 * 1000
	// MaxTokens bounds one token count, and a served context length.
	MaxTokens int64 = 100_000_000
	// MaxTokensPerSecond bounds one sample's phase rate and so every
	// token-weighted aggregate rate, which cannot exceed its largest input.
	MaxTokensPerSecond = 10_000_000.0
)

// Scope names the boundary a sample was observed at.
type Scope string

// ScopeLocalExecution is one actual execution on the measuring host's local
// engine, timed from the engine-facing dispatch boundary. It is distinct from
// an origin router's end-to-end request timings.
const ScopeLocalExecution Scope = "localExecution"

// Outcome is how a local execution ended at the transport level.
type Outcome string

const (
	OutcomeCompleted Outcome = "completed"
	OutcomeFailed    Outcome = "failed"
	OutcomeCancelled Outcome = "cancelled"
)

// ProfileKey identifies what a sample measured. Two samples are aggregated
// together only when every field matches.
type ProfileKey struct {
	// Host is the measuring host's UUID.
	Host string `json:"host"`
	// Engine is the actual serving engine's stable variant ID, such as
	// "llamacpp-custom", not the ingress facade or a requested alias.
	Engine string `json:"engine"`
	// Model is the canonical concrete model identity.
	Model string `json:"model"`
	// Fingerprint identifies the effective serving profile: the nonsecret
	// runtime settings that affect speed. It is opaque to this package.
	Fingerprint string `json:"fingerprint"`
}

// Validate checks the key's fields against the contract limits.
func (k ProfileKey) Validate() error {
	if err := checkID("host", k.Host); err != nil {
		return err
	}
	if err := checkID("engine", k.Engine); err != nil {
		return err
	}
	if err := checkText("model", k.Model, MaxModelBytes, true); err != nil {
		return err
	}
	return checkID("fingerprint", k.Fingerprint)
}

// Profile is display-only description of a key. It never takes part in
// aggregation identity; an empty field or nil context is unknown.
type Profile struct {
	// EngineFamily groups engine variants for display, such as "llamacpp".
	EngineFamily  string `json:"engineFamily,omitempty"`
	Quantization  string `json:"quantization,omitempty"`
	ContextTokens *int64 `json:"contextTokens,omitempty"`
}

// Validate checks the profile's fields against the contract limits.
func (p Profile) Validate() error {
	if p.EngineFamily != "" {
		if err := checkID("engineFamily", p.EngineFamily); err != nil {
			return err
		}
	}
	if err := checkText("quantization", p.Quantization, MaxLabelBytes, false); err != nil {
		return err
	}
	if p.ContextTokens != nil && (*p.ContextTokens < 1 || *p.ContextTokens > MaxTokens) {
		return fmt.Errorf("contextTokens %d is outside 1..%d", *p.ContextTokens, MaxTokens)
	}
	return nil
}

// Measurement is what one local execution observed. Latencies are integer
// milliseconds from the engine-facing dispatch, except AdmissionWaitMs, which
// runs from ingress or admission entry to the granted ticket. Engine-reported
// phase durations are integer nanoseconds. Token counts are exact
// engine-reported counts, never estimates.
type Measurement struct {
	Outcome Outcome `json:"outcome"`
	// ProtocolComplete says whether the response reached its protocol-defined
	// end. Nil means no extractor could tell.
	ProtocolComplete *bool  `json:"protocolComplete,omitempty"`
	AdmissionWaitMs  *int64 `json:"admissionWaitMs,omitempty"`
	HeaderMs         *int64 `json:"headerMs,omitempty"`
	// FirstBodyMs is the first response body byte, which is not token evidence.
	FirstBodyMs *int64 `json:"firstBodyMs,omitempty"`
	// FirstTokenMs is the first protocol-defined output token, set only by a
	// qualified extractor.
	FirstTokenMs *int64 `json:"firstTokenMs,omitempty"`
	TotalMs      *int64 `json:"totalMs,omitempty"`
	// PromptTokens counts prompt tokens actually evaluated, excluding
	// CachedPromptTokens reused from a cache.
	PromptTokens       *int64 `json:"promptTokens,omitempty"`
	CachedPromptTokens *int64 `json:"cachedPromptTokens,omitempty"`
	OutputTokens       *int64 `json:"outputTokens,omitempty"`
	// PrefillNs is the engine-reported duration that evaluated PromptTokens.
	PrefillNs *int64 `json:"prefillNs,omitempty"`
	// DecodeNs is the engine-reported duration that generated OutputTokens.
	DecodeNs *int64 `json:"decodeNs,omitempty"`
	// EngineDecodeNs is the interval underlying the separately reported native
	// decode rate. Its work count need not equal OutputTokens.
	EngineDecodeNs *int64 `json:"engineDecodeNs,omitempty"`
	// EngineDecodeTokensPerSecond preserves the engine's reported native rate,
	// including rounding. It never implies an exact native token count.
	EngineDecodeTokensPerSecond *float64 `json:"engineDecodeTokensPerSecond,omitempty"`
	// LoadNs is an engine-reported model load duration.
	LoadNs *int64 `json:"loadNs,omitempty"`
}

// Validate checks the measurement's outcome, ranges and phase rates.
func (m Measurement) Validate() error {
	switch m.Outcome {
	case OutcomeCompleted, OutcomeFailed, OutcomeCancelled:
	default:
		return fmt.Errorf("outcome %q is not supported", m.Outcome)
	}
	for _, f := range []struct {
		name string
		v    *int64
		max  int64
	}{
		{"admissionWaitMs", m.AdmissionWaitMs, MaxLatencyMs},
		{"headerMs", m.HeaderMs, MaxLatencyMs},
		{"firstBodyMs", m.FirstBodyMs, MaxLatencyMs},
		{"firstTokenMs", m.FirstTokenMs, MaxLatencyMs},
		{"totalMs", m.TotalMs, MaxLatencyMs},
		{"promptTokens", m.PromptTokens, MaxTokens},
		{"cachedPromptTokens", m.CachedPromptTokens, MaxTokens},
		{"outputTokens", m.OutputTokens, MaxTokens},
		{"prefillNs", m.PrefillNs, MaxPhaseNs},
		{"decodeNs", m.DecodeNs, MaxPhaseNs},
		{"engineDecodeNs", m.EngineDecodeNs, MaxPhaseNs},
		{"loadNs", m.LoadNs, MaxPhaseNs},
	} {
		if f.v != nil && (*f.v < 0 || *f.v > f.max) {
			return fmt.Errorf("%s %d is outside 0..%d", f.name, *f.v, f.max)
		}
	}
	if m.EngineDecodeTokensPerSecond != nil {
		if err := checkReportedRate("engineDecodeTokensPerSecond", *m.EngineDecodeTokensPerSecond); err != nil {
			return err
		}
	}
	if err := checkPhaseRate("prefill", m.PromptTokens, m.PrefillNs); err != nil {
		return err
	}
	return checkPhaseRate("decode", m.OutputTokens, m.DecodeNs)
}

// checkPhaseRate refuses a phase whose own rate exceeds MaxTokensPerSecond, so
// no token-weighted aggregate can exceed it either.
func checkPhaseRate(name string, tokens, ns *int64) error {
	if tokens == nil || ns == nil || *ns <= 0 {
		return nil
	}
	if r := TokensPerSecond(*tokens, *ns); r > MaxTokensPerSecond {
		return fmt.Errorf("%s rate %.0f tokens/s exceeds %.0f", name, r, MaxTokensPerSecond)
	}
	return nil
}

// TokensPerSecond is tokens over a positive nanosecond duration.
func TokensPerSecond(tokens, ns int64) float64 {
	return float64(tokens) / (float64(ns) / 1e9)
}

// Sample is one local execution as its producer reports it to the broker that
// owns aggregation. The broker stamps the observation time on receipt.
type Sample struct {
	Scope Scope `json:"scope"`
	// Producer identifies the live producing process incarnation. Seq increases
	// by at least one per sample within it, so a repeated (Producer, Seq) is a
	// duplicate.
	Producer string     `json:"producer"`
	Seq      uint64     `json:"seq"`
	Key      ProfileKey `json:"key"`
	Profile  Profile    `json:"profile"`
	Measurement
}

// Validate checks the sample's scope, identity and measurement.
func (s Sample) Validate() error {
	if s.Scope != ScopeLocalExecution {
		return fmt.Errorf("scope %q is not supported", s.Scope)
	}
	if err := checkID("producer", s.Producer); err != nil {
		return err
	}
	if s.Seq == 0 {
		return errors.New("seq must be positive")
	}
	if err := s.Key.Validate(); err != nil {
		return err
	}
	if err := s.Profile.Validate(); err != nil {
		return err
	}
	return s.Measurement.Validate()
}

// Distribution summarizes one latency or duration metric over the samples that
// measured it, by nearest rank. Its unit is the one its field name states.
type Distribution struct {
	Samples int   `json:"samples"`
	Median  int64 `json:"median"`
	P95     int64 `json:"p95"`
}

// Rate is a token-weighted phase rate: the sum of qualified tokens over the
// sum of their matching phase durations.
type Rate struct {
	Samples         int     `json:"samples"`
	Tokens          int64   `json:"tokens"`
	DurationNs      int64   `json:"durationNs"`
	TokensPerSecond float64 `json:"tokensPerSecond"`
}

// ReportedRate is the duration-weighted mean of qualified engine-reported
// native rates: sum(rate_i * duration_i) / sum(duration_i). Rounded native
// rates do not establish an exact integer token total.
type ReportedRate struct {
	Samples         int     `json:"samples"`
	DurationNs      int64   `json:"durationNs"`
	TokensPerSecond float64 `json:"tokensPerSecond"`
}

// UnmarshalJSON requires all reported-rate fields. In particular, an absent or
// null rate must not become a measured zero through Go's numeric zero value.
func (r *ReportedRate) UnmarshalJSON(data []byte) error {
	var fields struct {
		Samples         *int     `json:"samples"`
		DurationNs      *int64   `json:"durationNs"`
		TokensPerSecond *float64 `json:"tokensPerSecond"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&fields); err != nil {
		return err
	}
	if fields.Samples == nil || fields.DurationNs == nil || fields.TokensPerSecond == nil {
		return errors.New("engineDecode requires samples, durationNs and tokensPerSecond")
	}
	*r = ReportedRate{Samples: *fields.Samples, DurationNs: *fields.DurationNs, TokensPerSecond: *fields.TokensPerSecond}
	return nil
}

// Entry is one profile key's aggregates. A nil aggregate means no retained
// sample qualified for it.
type Entry struct {
	Key     ProfileKey `json:"key"`
	Profile Profile    `json:"profile"`
	// Samples counts every retained sample; Completed, Failed and Cancelled
	// partition it. Incomplete counts the completed samples whose protocol
	// end was known to be missing.
	Samples    int `json:"samples"`
	Completed  int `json:"completed"`
	Incomplete int `json:"incomplete"`
	Failed     int `json:"failed"`
	Cancelled  int `json:"cancelled"`
	// NewestAgeMs and OldestAgeMs are the source ages of the newest and oldest
	// retained samples when the summary was built. A receiver advances them by
	// its own elapsed time; a heartbeat never makes an old run new.
	NewestAgeMs int64 `json:"newestAgeMs"`
	OldestAgeMs int64 `json:"oldestAgeMs"`
	// NewestObservedAtMs is the measuring host's wall clock for the newest
	// sample, in Unix milliseconds, for display only.
	NewestObservedAtMs int64 `json:"newestObservedAtMs"`
	// ClockUncertain marks a key holding a sample stamped later than the
	// measuring host's current clock, whose age is therefore unreliable.
	ClockUncertain bool `json:"clockUncertain,omitempty"`
	// OutputTokens sums known counts from successful, protocol-complete samples,
	// independently of decode timing. Nil means no count qualified; zero is measured.
	OutputTokens *int64 `json:"outputTokens,omitempty"`

	AdmissionWaitMs *Distribution `json:"admissionWaitMs,omitempty"`
	HeaderMs        *Distribution `json:"headerMs,omitempty"`
	FirstBodyMs     *Distribution `json:"firstBodyMs,omitempty"`
	FirstTokenMs    *Distribution `json:"firstTokenMs,omitempty"`
	TotalMs         *Distribution `json:"totalMs,omitempty"`
	LoadNs          *Distribution `json:"loadNs,omitempty"`
	Prefill         *Rate         `json:"prefill,omitempty"`
	Decode          *Rate         `json:"decode,omitempty"`
	EngineDecode    *ReportedRate `json:"engineDecode,omitempty"`
}

// Window describes the retention the aggregates were computed under.
type Window struct {
	MaxAgeMs         int64 `json:"maxAgeMs"`
	MaxSamplesPerKey int   `json:"maxSamplesPerKey"`
}

// Summary is a measuring host's full set of own aggregates at one revision.
type Summary struct {
	Version int `json:"version"`
	// MeasuringHost is the host UUID every entry was measured on. A receiver
	// binds it to the authenticated sender.
	MeasuringHost string `json:"measuringHost"`
	// AdmissionIncarnation is the sender's current admission incarnation.
	AdmissionIncarnation string `json:"admissionIncarnation"`
	// Revision is the persisted snapshot revision. It only increases for a
	// measuring host, and a retransmission repeats it.
	Revision int64  `json:"revision"`
	Window   Window `json:"window"`
	// Entries is empty, not absent, when the history is empty.
	Entries []Entry `json:"entries"`
	// OmittedEntries counts keys left out to keep the payload within
	// MaxPayloadBytes.
	OmittedEntries int `json:"omittedEntries,omitempty"`
}

// Validate checks the summary against the contract limits.
func (s Summary) Validate() error {
	if s.Version != SchemaVersion {
		return fmt.Errorf("summary version %d is not supported", s.Version)
	}
	if err := checkID("measuringHost", s.MeasuringHost); err != nil {
		return err
	}
	if err := checkID("admissionIncarnation", s.AdmissionIncarnation); err != nil {
		return err
	}
	if s.Revision <= 0 {
		return errors.New("revision must be positive")
	}
	if s.Window.MaxAgeMs <= 0 || s.Window.MaxAgeMs > RetentionMs {
		return fmt.Errorf("window maxAgeMs %d is outside 1..%d", s.Window.MaxAgeMs, RetentionMs)
	}
	if s.Window.MaxSamplesPerKey <= 0 || s.Window.MaxSamplesPerKey > MaxSamplesPerKey {
		return fmt.Errorf("window maxSamplesPerKey %d is outside 1..%d", s.Window.MaxSamplesPerKey, MaxSamplesPerKey)
	}
	if len(s.Entries) > MaxEntries {
		return fmt.Errorf("%d entries exceed %d", len(s.Entries), MaxEntries)
	}
	if s.OmittedEntries < 0 || len(s.Entries)+s.OmittedEntries > MaxEntries {
		return fmt.Errorf("omittedEntries %d is outside 0..%d", s.OmittedEntries, MaxEntries-len(s.Entries))
	}
	seen := make(map[ProfileKey]bool, len(s.Entries))
	for i, e := range s.Entries {
		if err := e.validate(s.MeasuringHost, s.Window); err != nil {
			return fmt.Errorf("entry %d: %w", i, err)
		}
		if seen[e.Key] {
			return fmt.Errorf("entry %d repeats a key", i)
		}
		seen[e.Key] = true
	}
	return nil
}

func checkReportedRate(name string, rate float64) error {
	if math.IsNaN(rate) || math.IsInf(rate, 0) || rate < 0 || rate > MaxTokensPerSecond {
		return fmt.Errorf("%s %v is not a finite rate within 0..%.0f", name, rate, MaxTokensPerSecond)
	}
	return nil
}

func (e Entry) validate(host string, w Window) error {
	if err := e.Key.Validate(); err != nil {
		return err
	}
	if e.Key.Host != host {
		return errors.New("key host is not the measuring host")
	}
	if err := e.Profile.Validate(); err != nil {
		return err
	}
	if e.Samples < 1 || e.Samples > w.MaxSamplesPerKey {
		return fmt.Errorf("samples %d is outside 1..%d", e.Samples, w.MaxSamplesPerKey)
	}
	if e.Completed < 0 || e.Failed < 0 || e.Cancelled < 0 || e.Completed+e.Failed+e.Cancelled != e.Samples {
		return errors.New("outcome counts do not partition samples")
	}
	if e.Incomplete < 0 || e.Incomplete > e.Completed {
		return fmt.Errorf("incomplete %d is outside 0..%d", e.Incomplete, e.Completed)
	}
	if e.OutputTokens != nil {
		eligible := e.Completed - e.Incomplete
		if eligible < 1 || eligible > e.Samples || *e.OutputTokens < 0 || *e.OutputTokens > int64(eligible)*MaxTokens {
			return errors.New("outputTokens requires eligible completed samples and a total within their token bound")
		}
	}
	if e.NewestAgeMs < 0 || e.NewestAgeMs > e.OldestAgeMs || e.OldestAgeMs > w.MaxAgeMs {
		return errors.New("sample ages are out of order or outside the window")
	}
	if e.NewestObservedAtMs <= 0 {
		return errors.New("newestObservedAtMs must be positive")
	}
	for _, d := range []struct {
		name string
		d    *Distribution
		max  int64
	}{
		{"admissionWaitMs", e.AdmissionWaitMs, MaxLatencyMs},
		{"headerMs", e.HeaderMs, MaxLatencyMs},
		{"firstBodyMs", e.FirstBodyMs, MaxLatencyMs},
		{"firstTokenMs", e.FirstTokenMs, MaxLatencyMs},
		{"totalMs", e.TotalMs, MaxLatencyMs},
		{"loadNs", e.LoadNs, MaxPhaseNs},
	} {
		if d.d == nil {
			continue
		}
		if d.d.Samples < 1 || d.d.Samples > e.Completed {
			return fmt.Errorf("%s samples %d is outside 1..%d", d.name, d.d.Samples, e.Completed)
		}
		if d.d.Median < 0 || d.d.Median > d.d.P95 || d.d.P95 > d.max {
			return fmt.Errorf("%s percentiles are out of order or outside 0..%d", d.name, d.max)
		}
	}
	for _, r := range []struct {
		name string
		r    *Rate
	}{{"prefill", e.Prefill}, {"decode", e.Decode}} {
		if r.r == nil {
			continue
		}
		if r.r.Samples < 1 || r.r.Samples > e.Completed {
			return fmt.Errorf("%s samples %d is outside 1..%d", r.name, r.r.Samples, e.Completed)
		}
		if r.r.Tokens < 0 || r.r.Tokens > int64(r.r.Samples)*MaxTokens {
			return fmt.Errorf("%s tokens %d is out of range", r.name, r.r.Tokens)
		}
		if r.r.DurationNs <= 0 || r.r.DurationNs > int64(r.r.Samples)*MaxPhaseNs {
			return fmt.Errorf("%s durationNs %d is out of range", r.name, r.r.DurationNs)
		}
		tps := r.r.TokensPerSecond
		if math.IsNaN(tps) || math.IsInf(tps, 0) || tps < 0 || tps > MaxTokensPerSecond {
			return fmt.Errorf("%s tokensPerSecond %v is not a finite rate within 0..%.0f", r.name, tps, MaxTokensPerSecond)
		}
	}
	if r := e.EngineDecode; r != nil {
		if r.Samples < 1 || r.Samples > e.Completed-e.Incomplete {
			return fmt.Errorf("engineDecode samples %d is outside 1..%d", r.Samples, e.Completed-e.Incomplete)
		}
		if r.DurationNs <= 0 || r.DurationNs > int64(r.Samples)*MaxPhaseNs {
			return fmt.Errorf("engineDecode durationNs %d is out of range", r.DurationNs)
		}
		if err := checkReportedRate("engineDecode tokensPerSecond", r.TokensPerSecond); err != nil {
			return err
		}
	}
	return nil
}

// EncodeSummary validates s and encodes it within MaxPayloadBytes.
func EncodeSummary(s Summary) ([]byte, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	data, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	if len(data) > MaxPayloadBytes {
		return nil, fmt.Errorf("summary is %d bytes, over %d", len(data), MaxPayloadBytes)
	}
	return data, nil
}

// DecodeSummary strictly decodes and validates one received Summary. It
// refuses a payload over MaxPayloadBytes before parsing it, unknown fields and
// trailing data.
func DecodeSummary(data []byte) (Summary, error) {
	if len(data) > MaxPayloadBytes {
		return Summary{}, fmt.Errorf("summary is %d bytes, over %d", len(data), MaxPayloadBytes)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var s Summary
	if err := dec.Decode(&s); err != nil {
		return Summary{}, fmt.Errorf("decode summary: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return Summary{}, errors.New("decode summary: trailing data")
	}
	if err := s.Validate(); err != nil {
		return Summary{}, err
	}
	return s, nil
}

// checkID requires a nonempty identifier of at most MaxIDBytes letters,
// digits, '.', '_', ':' or '-'.
func checkID(name, v string) error {
	if v == "" {
		return fmt.Errorf("%s is required", name)
	}
	if len(v) > MaxIDBytes {
		return fmt.Errorf("%s is longer than %d bytes", name, MaxIDBytes)
	}
	for _, r := range v {
		if r > unicode.MaxASCII || !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '.' || r == '_' || r == ':' || r == '-') {
			return fmt.Errorf("%s contains %q", name, r)
		}
	}
	return nil
}

// checkText requires valid UTF-8 without control characters, at most max
// bytes, and nonempty when required.
func checkText(name, v string, max int, required bool) error {
	if v == "" {
		if required {
			return fmt.Errorf("%s is required", name)
		}
		return nil
	}
	if len(v) > max {
		return fmt.Errorf("%s is longer than %d bytes", name, max)
	}
	if !utf8.ValidString(v) {
		return fmt.Errorf("%s is not valid UTF-8", name)
	}
	for _, r := range v {
		if unicode.IsControl(r) {
			return fmt.Errorf("%s contains a control character", name)
		}
	}
	return nil
}
