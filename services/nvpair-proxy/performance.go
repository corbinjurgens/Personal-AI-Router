// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"math/big"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/performancewire"
)

const performanceQueueCapacity = 128

// The FIFO lock never covers codec IO. Counters saturate and have a fixed set
// of reasons; neither pressure nor missing identity allocates drop records.
type performanceQueue struct {
	mu         sync.Mutex
	items      [performanceQueueCapacity]performancewire.Sample
	head, size int
	seq        uint64
	drops      [4]uint64 // full, identity/options, stale identity, invalid sample
	wake       chan struct{}
}

func (p *Proxy) dropPerformance(reason int) {
	q := &p.performance
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.drops[reason] != ^uint64(0) {
		q.drops[reason]++
	}
}

func (p *Proxy) enqueuePerformance(s performancewire.Sample) {
	q := &p.performance
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.size == len(q.items) || q.seq == ^uint64(0) {
		if q.drops[0] != ^uint64(0) {
			q.drops[0]++
		}
		return
	}
	s.Scope, s.Producer, s.Seq = performancewire.ScopeLocalExecution, p.runID, q.seq+1
	if s.Validate() != nil {
		if q.drops[3] != ^uint64(0) {
			q.drops[3]++
		}
		return
	}
	q.seq++
	q.items[(q.head+q.size)%len(q.items)] = s
	q.size++
	select {
	case q.wake <- struct{}{}:
	default:
	}
}

func (p *Proxy) sendPerformance(ctx context.Context) {
	q := &p.performance
	for {
		if ctx.Err() != nil {
			return
		}
		q.mu.Lock()
		if q.size == 0 {
			q.mu.Unlock()
			select {
			case <-ctx.Done():
				return
			case <-q.wake:
			}
			continue
		}
		s := q.items[q.head]
		q.items[q.head] = performancewire.Sample{}
		q.head = (q.head + 1) % len(q.items)
		q.size--
		q.mu.Unlock()
		// Like the proxy's other notifications, an active write belongs to the
		// codec's lifetime. Cancellation is checked between writes, never joined
		// by request handling or shutdown.
		if ctx.Err() != nil {
			return
		}
		_ = p.codec.Notify("performance:sample", s)
	}
}

// performanceFingerprint is the only serving-profile hash in the proxy. The
// joined identity is read under its owner's lock and reduced to this model's
// digest. No filename, alias, response model or placeholder supplies identity.
func (f *facade) performanceFingerprint(model, api string, body []byte) (performancewire.ProfileKey, performanceExtractQualification, bool) {
	family := f.profile.Family
	if family == "" {
		family = f.profile.Name
	}
	q := performanceExtractQualification{EngineFamily: family, APIPath: api, Streaming: true}
	switch family {
	case "llamacpp":
		q.SchemaID = performanceExtractLlamaSchema
	case "strata":
		q.SchemaID = performanceExtractStrataSchema
	case "ollama":
	default:
		return performancewire.ProfileKey{}, q, false
	}
	a := f.host.admission
	a.mu.Lock()
	id, exists := a.strict.identity[f.profile.Name]
	digest, found := a.modelDigestLocked(id, admKey{engine: f.profile.Name, model: f.profile.normalizeModel(model)})
	valid := exists && found && id.Validate() == nil
	// Copy only immutable scalars; the projection's map never escapes its lock.
	run, settings, launch := id.EngineRunID, id.SettingsFingerprint, id.LaunchDigest
	a.mu.Unlock()
	host, _ := f.host.selfIdentity(f.profile.Name)
	if !valid || host == "" {
		return performancewire.ProfileKey{}, q, false
	}
	options, ok := performanceOptions(body, family)
	if !ok {
		return performancewire.ProfileKey{}, q, false
	}
	encoded, err := json.Marshal(struct {
		Run, Settings, Launch, Model, API, Schema string
		Options                                   map[string]any
	}{run, settings, launch, digest, api, family + ":" + q.SchemaID, options})
	if err != nil {
		return performancewire.ProfileKey{}, q, false
	}
	sum := sha256.Sum256(encoded)
	key := performancewire.ProfileKey{Host: host, Engine: f.profile.Name, Model: f.profile.normalizeModel(model), Fingerprint: hex.EncodeToString(sum[:])}
	return key, q, key.Validate() == nil
}

// Canonicalize the effective post-rewrite/default options, excluding inference
// content. Unknown top-level overrides suppress observation. Numeric spellings
// normalize exactly, including integers above float64's exact range.
func performanceOptions(body []byte, family string) (map[string]any, bool) {
	if !uniqueKeys(body) {
		return nil, false
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var fields map[string]any
	if dec.Decode(&fields) != nil || fields == nil {
		return nil, false
	}
	options := make(map[string]any)
	for key, value := range fields {
		switch key {
		case "model", "messages", "prompt", "input":
			continue
		case "tools":
			options["hasTools"] = value != nil
			continue
		}
		allowed := slices.Contains(chatRequestKeys, key) || key == "n"
		if family == "ollama" {
			allowed = allowed || key == "options" || key == "keep_alive" || key == "format" || key == "raw" || key == "think"
		}
		if !allowed {
			return nil, false
		}
		options[key] = performanceCanonicalNumber(value)
	}
	return options, true
}

func performanceCanonicalNumber(v any) any {
	switch n := v.(type) {
	case json.Number:
		if len(n) > 64 {
			return n
		}
		if i := strings.IndexAny(string(n), "eE"); i >= 0 {
			e, err := strconv.Atoi(string(n[i+1:]))
			if err != nil || e < -308 || e > 308 {
				return n
			}
		}
		if r, ok := new(big.Rat).SetString(string(n)); ok {
			// JSON decimal numbers have no prime factors other than 2 and 5
			// in their denominator. This bound preserves every accepted digit.
			decimal := strings.TrimRight(strings.TrimRight(r.FloatString(372), "0"), ".")
			return json.Number(decimal)
		}
	case map[string]any:
		for k, value := range n {
			n[k] = performanceCanonicalNumber(value)
		}
	case []any:
		for i, value := range n {
			n[i] = performanceCanonicalNumber(value)
		}
	}
	return v
}

type performanceAttempt struct {
	mu                sync.Mutex // first-byte peek can outlive a timed-out attempt
	f                 *facade
	key               performancewire.ProfileKey
	qualification     performanceExtractQualification
	body              []byte // request-owned, released with the attempt; never published
	api, model        string
	parent            context.Context
	wait              time.Duration
	start             time.Time
	header, firstBody *int64
	status            int
	reader            *streamFrameReader
}

func (f *facade) newPerformanceAttempt(parent context.Context, model, api string, body []byte, wait time.Duration) *performanceAttempt {
	key, qualification, ok := f.performanceFingerprint(model, api, body)
	if !ok {
		f.host.dropPerformance(1)
		return nil
	}
	return &performanceAttempt{f: f, key: key, qualification: qualification, body: body, api: api, model: model, parent: parent, wait: wait}
}

type performanceTransport struct {
	http.RoundTripper
	attempt *performanceAttempt
}

func (t performanceTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	a := t.attempt
	if a == nil {
		return t.RoundTripper.RoundTrip(r)
	}
	a.start = time.Now()
	resp, err := t.RoundTripper.RoundTrip(r)
	if resp != nil {
		a.header = performanceMilliseconds(time.Since(a.start))
		a.status = resp.StatusCode
		resp.Body = &performanceFirstBody{ReadCloser: resp.Body, attempt: a}
	}
	return resp, err
}

type performanceFirstBody struct {
	io.ReadCloser
	attempt *performanceAttempt
}

func (b *performanceFirstBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.attempt.mu.Lock()
	defer b.attempt.mu.Unlock()
	if n > 0 && b.attempt.firstBody == nil {
		b.attempt.firstBody = performanceMilliseconds(time.Since(b.attempt.start))
	}
	return n, err
}

func performanceMilliseconds(d time.Duration) *int64 {
	v := d.Milliseconds()
	if v < 0 || v > performancewire.MaxLatencyMs {
		return nil
	}
	return &v
}

// installResponseStream is shared by origin candidates and local ingress. Each
// response gets exactly SC's one reader. A peer-forwarding origin passes no
// attempt and therefore has no numeric producer.
func installResponseStream(resp *http.Response, request originStreamRequest, parent context.Context, idle time.Duration, a *performanceAttempt) (*streamFrameReader, error) {
	managed, validMedia := request.manages(resp.Header.Get("Content-Type"))
	if !managed || resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, nil
	}
	config := streamFrameConfig{Protocol: request.protocol, Upstream: resp.Body, Parent: parent, IdleTimeout: idle, Choices: request.choices, PeerNormalized: request.peerNormalized}
	if a != nil {
		config.Extract = func(payload []byte) performanceExtractMetadata {
			return extractPerformanceObserved(payload, a.qualification)
		}
	}
	reader, err := newStreamFrameReader(config)
	if err != nil {
		return nil, err
	}
	if !validMedia || (resp.Header.Get("Content-Encoding") != "" && !strings.EqualFold(resp.Header.Get("Content-Encoding"), "identity")) {
		reader.mu.Lock()
		reader.finishInterruptedLocked(streamInterruptReasonMalformed)
		reader.mu.Unlock()
	}
	resp.Body = reader
	resp.ContentLength = -1
	clearOriginStreamIntegrity(resp.Header)
	clearOriginStreamIntegrity(resp.Trailer)
	if a != nil {
		a.reader = reader
	}
	return reader, nil
}

func (a *performanceAttempt) finish(sc *statusCapture, returned bool) {
	if a == nil || a.start.IsZero() {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	key, _, valid := a.f.performanceFingerprint(a.model, a.api, a.body)
	if !valid || key != a.key {
		a.f.host.dropPerformance(2)
		return
	}
	m := performancewire.Measurement{Outcome: performancewire.OutcomeFailed, AdmissionWaitMs: performanceMilliseconds(a.wait), HeaderMs: a.header, FirstBodyMs: a.firstBody, TotalMs: performanceMilliseconds(time.Since(a.start))}
	if a.parent.Err() != nil {
		m.Outcome = performancewire.OutcomeCancelled
	}
	if a.reader != nil {
		result := a.reader.Result()
		complete := result.Outcome == streamOutcomeCompleted && returned && sc.wroteErr == nil && a.parent.Err() == nil
		m.ProtocolComplete = &complete
		n := result.Numeric
		m.PromptTokens, m.CachedPromptTokens, m.OutputTokens = n.PromptTokens, n.CachedPromptTokens, n.OutputTokens
		m.PrefillNs, m.DecodeNs, m.LoadNs = n.PrefillNs, n.DecodeNs, n.LoadNs
		m.EngineDecodeNs, m.EngineDecodeTokensPerSecond = n.EngineDecodeNs, n.EngineDecodeTokensPerSecond
		if result.Outcome == streamOutcomeCancelled {
			m.Outcome = performancewire.OutcomeCancelled
		}
		if complete {
			m.Outcome = performancewire.OutcomeCompleted
		}
	} else if returned && sc.wroteErr == nil && a.parent.Err() == nil && a.status >= 200 && a.status < 300 {
		m.Outcome = performancewire.OutcomeCompleted // unobserved protocol stays nil
	}
	a.f.host.enqueuePerformance(performancewire.Sample{Key: a.key, Profile: performancewire.Profile{EngineFamily: a.qualification.EngineFamily}, Measurement: m})
}

// Metadata merges independent fields, never counters. Conflicts are sticky.
type performanceNumbers struct {
	value     performanceExtractMetadata
	conflicts uint16
}

func mergePerformanceNumber[T comparable](dst **T, src *T, conflicts *uint16, bit uint16) {
	if *conflicts&bit != 0 {
		*dst = nil
		return
	}
	if src == nil {
		return
	}
	if *dst != nil && **dst != *src {
		*conflicts |= bit
		*dst = nil
		return
	}
	v := *src
	*dst = &v
}
func (n *performanceNumbers) merge(v performanceExtractMetadata) {
	n.conflicts |= v.invalid
	fields := []struct {
		dst **int64
		src *int64
	}{
		{&n.value.PromptTokens, v.PromptTokens}, {&n.value.CachedPromptTokens, v.CachedPromptTokens},
		{&n.value.OutputTokens, v.OutputTokens}, {&n.value.PrefillNs, v.PrefillNs}, {&n.value.DecodeNs, v.DecodeNs},
		{&n.value.LoadNs, v.LoadNs}, {&n.value.EngineDecodeNs, v.EngineDecodeNs},
	}
	for i, field := range fields {
		mergePerformanceNumber(field.dst, field.src, &n.conflicts, 1<<i)
	}
	mergePerformanceNumber(&n.value.EngineDecodeTokensPerSecond, v.EngineDecodeTokensPerSecond, &n.conflicts, 1<<7)
	n.value.PhaseDisqualified = n.value.PhaseDisqualified || v.PhaseDisqualified
}
func (n *performanceNumbers) snapshot() performanceExtractMetadata {
	var copy performanceNumbers
	copy.merge(n.value)
	if n.value.PhaseDisqualified {
		copy.value.PromptTokens = nil
		copy.value.PrefillNs = nil
		copy.value.DecodeNs = nil
		copy.value.EngineDecodeNs = nil
		copy.value.EngineDecodeTokensPerSecond = nil
	}
	if n.conflicts&(1<<0) != 0 {
		copy.value.PrefillNs = nil
	}
	if n.conflicts&(1<<2) != 0 {
		copy.value.DecodeNs = nil
		copy.value.EngineDecodeNs = nil
		copy.value.EngineDecodeTokensPerSecond = nil
	}
	if n.conflicts&(1<<6) != 0 {
		copy.value.EngineDecodeTokensPerSecond = nil
	}
	return copy.value
}

// Mark present-but-unusable evidence as unknown even if an earlier object had
// usable counters. Absence still permits independently split usage/timings.
func extractPerformanceObserved(payload []byte, q performanceExtractQualification) performanceExtractMetadata {
	v := extractPerformancePayload(payload, q)
	f := performanceExtractObject(payload)
	var fields map[string]json.RawMessage
	if q.EngineFamily == "ollama" && (q.APIPath == "/api/chat" || q.APIPath == "/api/generate") {
		fields = f
		for _, check := range []struct {
			key   string
			value *int64
			bit   uint16
		}{
			{"eval_count", v.OutputTokens, 1 << 2}, {"eval_duration", v.DecodeNs, 1 << 4}, {"load_duration", v.LoadNs, 1 << 5}, {"prompt_eval_duration", v.PrefillNs, 1 << 3},
		} {
			if _, present := fields[check.key]; present && check.value == nil {
				v.invalid |= check.bit
			}
		}
		return v
	}
	timings := performanceExtractObject(f["timings"])
	usage := performanceExtractObject(f["usage"])
	if raw, present := f["timings"]; present && performanceExtractObject(raw) == nil {
		v.invalid |= 0xcf
	}
	if raw, present := f["usage"]; present && performanceExtractObject(raw) == nil {
		v.invalid |= 1<<2 | 1<<1
	}
	for _, check := range []struct {
		key   string
		value *int64
		bit   uint16
	}{
		{"prompt_n", v.PromptTokens, 1 << 0}, {"cache_n", v.CachedPromptTokens, 1 << 1}, {"predicted_n", v.OutputTokens, 1 << 2},
		{"prompt_ms", v.PrefillNs, 1 << 3}, {"predicted_ms", v.EngineDecodeNs, 1 << 6},
	} {
		if _, present := timings[check.key]; present && check.value == nil {
			v.invalid |= check.bit
		}
	}
	if _, present := usage["completion_tokens"]; present && v.OutputTokens == nil {
		v.invalid |= 1 << 2
	}
	if q.EngineFamily == "strata" {
		details := performanceExtractObject(usage["prompt_tokens_details"])
		if raw, present := usage["prompt_tokens_details"]; present && performanceExtractObject(raw) == nil {
			v.invalid |= 1 << 1
		}
		if _, present := details["cached_tokens"]; present && v.CachedPromptTokens == nil {
			v.invalid |= 1 << 1
		}
	}
	if _, present := timings["predicted_per_second"]; present && v.EngineDecodeTokensPerSecond == nil {
		v.invalid |= 1 << 7
	}
	return v
}
