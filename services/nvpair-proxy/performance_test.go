// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"nvpair-shared/nodepolicy"
	"nvpair-shared/performancewire"
)

const performanceStrataFinal = `{"choices":[{"index":0,"finish_reason":"stop"}],"usage":{"completion_tokens":20},"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":55}}`
const performanceLlamaFinal = `{"choices":[{"index":0,"finish_reason":"stop"}],"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`

func performanceIdentity() nodepolicy.EngineIdentity {
	return nodepolicy.EngineIdentity{EngineRunID: "runtime-1", SettingsFingerprint: strings.Repeat("a", 64), LaunchDigest: strings.Repeat("b", 64), ModelDigests: map[string]string{"m": strings.Repeat("c", 64), "fixture-model": strings.Repeat("c", 64)}}
}

func performanceIdentify(p *Proxy, host string) {
	f := p.soleFacade()
	p.noteSelf(host, f.profile.Name, []string{"m", "fixture-model"})
	p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.profile.Name: performanceIdentity()})
}

func performanceProxy(t *testing.T, engine string) *Proxy {
	t.Helper()
	profile, ok := profileFor(engine)
	if engine == "strata" {
		var err error
		profile, err = dynamicProfile(strataDescriptor(t))
		if err != nil {
			t.Fatal(err)
		}
		ok = true
	}
	if !ok {
		t.Fatalf("missing profile %s", engine)
	}
	disc := NewDiscovery()
	disc.AddManual(selfNode("destination", profile.FacadePort, "m"))
	p := testProxy(profile, disc, profile.FacadePort)
	performanceIdentify(p, "destination")
	t.Cleanup(p.closeIdleTransports)
	return p
}

func performanceSamples(p *Proxy) []performancewire.Sample {
	q := &p.performance
	q.mu.Lock()
	defer q.mu.Unlock()
	samples := make([]performancewire.Sample, q.size)
	for i := range samples {
		samples[i] = q.items[(q.head+i)%len(q.items)]
	}
	return samples
}

func performanceOne(t *testing.T, p *Proxy) performancewire.Sample {
	t.Helper()
	s := performanceSamples(p)
	if len(s) != 1 {
		t.Fatalf("samples=%d, want one; drops=%v", len(s), p.performance.drops)
	}
	if err := s[0].Validate(); err != nil {
		t.Fatal(err)
	}
	return s[0]
}

func performanceUpstream(t *testing.T, payload string) *httptest.Server {
	t.Helper()
	u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, payload); err != nil {
			t.Error(err)
		}
	}))
	t.Cleanup(u.Close)
	return u
}

func TestPerformanceDestinationSelf(t *testing.T) {
	test := func(name, engine, final, body string, rate float64) {
		t.Run(name, func(t *testing.T) {
			p := performanceProxy(t, engine)
			u := performanceUpstream(t, "data: "+final+"\n\ndata: [DONE]\n\n")
			setBackend(t, p.soleFacade(), u.URL, true)
			w := httptest.NewRecorder()
			p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(body)))
			s := performanceOne(t, p)
			if w.Code != http.StatusOK || s.Key.Host != "destination" || s.Key.Engine != p.soleFacade().profile.Name || s.Key.Model != p.soleFacade().profile.normalizeModel("m") {
				t.Fatalf("status/key=%d %+v", w.Code, s.Key)
			}
			if s.Outcome != performancewire.OutcomeCompleted || s.ProtocolComplete == nil || !*s.ProtocolComplete || s.EngineDecodeTokensPerSecond == nil || *s.EngineDecodeTokensPerSecond != rate || s.DecodeNs != nil || s.FirstTokenMs != nil {
				t.Fatalf("measurement=%+v", s.Measurement)
			}
			if s.HeaderMs == nil || s.FirstBodyMs == nil || s.TotalMs == nil || s.AdmissionWaitMs == nil {
				t.Fatalf("missing local milestone: %+v", s.Measurement)
			}
			if *s.HeaderMs > *s.FirstBodyMs || *s.FirstBodyMs > *s.TotalMs {
				t.Fatalf("unordered local milestones: %+v", s.Measurement)
			}
		})
	}
	test("llama finish without usage", "llamacpp", performanceLlamaFinal, `{"model":"m","stream":true}`, 40)
	test("llama empty choices usage", "llamacpp", `{"choices":[],"usage":{"completion_tokens":11},"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`, `{"model":"m","stream":true,"stream_options":{"include_usage":true}}`, 40)
	test("Strata native count differs", "strata", performanceStrataFinal, `{"model":"m","stream":true}`, 55)
}

func TestPerformancePeerOriginNeverCounts(t *testing.T) {
	for _, transport := range []string{"direct", "reverse"} {
		t.Run(transport, func(t *testing.T) {
			fx := newProxyReverseFixture(t, "llamacpp", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				if _, err := io.WriteString(w, "data: "+performanceLlamaFinal+"\n\ndata: [DONE]\n\n"); err != nil {
					t.Error(err)
				}
			}))
			performanceIdentify(fx.p, "origin")
			performanceIdentify(fx.peer, "host-peer")
			fx.directHealthy.Store(transport == "direct")
			w := httptest.NewRecorder()
			fx.p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"fixture-model","stream":true}`)))
			if w.Code != http.StatusOK {
				t.Fatalf("status=%d", w.Code)
			}
			if s := performanceSamples(fx.p); len(s) != 0 {
				t.Fatalf("origin double-counted: %+v", s)
			}
			s := performanceOne(t, fx.peer)
			if s.Key.Host != "host-peer" || s.Key.Model != "fixture-model" || s.EngineDecodeTokensPerSecond == nil || *s.EngineDecodeTokensPerSecond != 40 {
				t.Fatalf("destination=%+v", s)
			}
			if transport == "reverse" && fx.reverseCalls.Load() == 0 {
				t.Fatal("reverse path was not exercised")
			}
		})
	}
}

func TestPerformanceStrictDestinations(t *testing.T) {
	for _, path := range []string{"self", "ingress"} {
		t.Run(path, func(t *testing.T) {
			p := testProxy(ollamaOnlyProfile(t), NewDiscovery(), 0)
			configureStrict(t, p.admission, filepath.Join(t.TempDir(), "capacity.json"))
			p.noteSelf("destination", "ollama", []string{"m"})
			u := performanceUpstream(t, "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"tool_calls\"}],\"usage\":{\"completion_tokens\":7}}\n\ndata: [DONE]\n\n")
			setBackend(t, p.soleFacade(), u.URL, true)
			w := httptest.NewRecorder()
			serveStrictHTTP(p, path, w, strictHTTPRequestWith(t, `{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":16,"stream":true}`))
			s := performanceOne(t, p)
			if s.OutputTokens == nil || *s.OutputTokens != 7 || s.Outcome != performancewire.OutcomeCompleted {
				t.Fatalf("sample=%+v", s)
			}
			if p.admission.currentActive() != 1 {
				t.Fatal("telemetry released strict uncertainty")
			}
		})
	}
}

func TestPerformanceMissingIdentitySuppresses(t *testing.T) {
	for _, missing := range []string{"engine", "model", "run", "settings", "launch", "host"} {
		t.Run(missing, func(t *testing.T) {
			p := performanceProxy(t, "llamacpp")
			id := performanceIdentity()
			switch missing {
			case "model":
				id.ModelDigests = map[string]string{}
			case "run":
				id.EngineRunID = ""
			case "settings":
				id.SettingsFingerprint = ""
			case "launch":
				id.LaunchDigest = ""
			case "host":
				p.selfNodeID = ""
			}
			ids := map[string]nodepolicy.EngineIdentity{p.soleFacade().profile.Name: id}
			if missing == "engine" {
				ids = nil
			}
			p.admission.setModelIdentity(ids)
			u := performanceUpstream(t, "data: "+performanceLlamaFinal+"\n\ndata: [DONE]\n\n")
			setBackend(t, p.soleFacade(), u.URL, true)
			w := httptest.NewRecorder()
			p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)), "peer")
			if w.Code != http.StatusOK || len(performanceSamples(p)) != 0 || p.performance.drops[1] != 1 {
				t.Fatalf("status=%d samples=%d drops=%v", w.Code, len(performanceSamples(p)), p.performance.drops)
			}
		})
	}
}

func TestPerformanceFingerprint(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	f := p.soleFacade()
	base := []byte(`{"model":"m","messages":[],"stream":true,"temperature":1}`)
	key, _, ok := f.performanceFingerprint("m", routeChatCompletions, base)
	if !ok {
		t.Fatal("valid projection refused")
	}
	same, _, ok := f.performanceFingerprint("m", routeChatCompletions, []byte(`{"temperature":1e0,"stream":true,"messages":[{"role":"user","content":"different"}],"model":"alias"}`))
	if !ok || same != key {
		t.Fatalf("equivalent effective options changed key: %+v %+v", key, same)
	}
	for _, change := range []string{"options", "runtime", "settings", "launch", "model", "api", "schema"} {
		t.Run(change, func(t *testing.T) {
			p := performanceProxy(t, "llamacpp")
			f := p.soleFacade()
			body := base
			api := routeChatCompletions
			id := performanceIdentity()
			switch change {
			case "options":
				body = []byte(`{"model":"m","stream":true,"temperature":0}`)
			case "runtime":
				id.EngineRunID = "runtime-2"
			case "settings":
				id.SettingsFingerprint = strings.Repeat("d", 64)
			case "launch":
				id.LaunchDigest = strings.Repeat("d", 64)
			case "model":
				id.ModelDigests["m"] = strings.Repeat("d", 64)
			case "api":
				api = "/v1/completions"
			case "schema":
				f.profile.Family = "strata"
			}
			p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{f.profile.Name: id})
			changed, _, ok := f.performanceFingerprint("m", api, body)
			if !ok || changed.Fingerprint == key.Fingerprint {
				t.Fatalf("%s did not separate profile", change)
			}
		})
	}
	if _, _, ok := f.performanceFingerprint("m", routeChatCompletions, []byte(`{"model":"m","mystery_override":true}`)); ok {
		t.Fatal("unknown override qualified")
	}
}

func TestPerformanceOptionNumbers(t *testing.T) {
	for _, pair := range [][2]string{{"1", "1e0"}, {"-0.25", "-25e-2"}, {"0", "-0"}, {"9007199254740993", "9007199254740993.0"}, {"1e-20", "10e-21"}} {
		t.Run(pair[0], func(t *testing.T) {
			a := performanceCanonicalNumber(json.Number(pair[0]))
			b := performanceCanonicalNumber(json.Number(pair[1]))
			if a != b {
				t.Fatalf("equivalent options differ: %v / %v", a, b)
			}
		})
	}
	// Outside the lexical bound, retain the exact JSON number without expansion.
	if performanceCanonicalNumber(json.Number("0e999999999")) != json.Number("0e999999999") {
		t.Fatal("unbounded exponent expanded")
	}
}

func TestPerformanceFixedDecimalCanonicalBoundaries(t *testing.T) {
	for _, raw := range []string{strings.Repeat("9", 64), "0." + strings.Repeat("1", 62), "1e308", "1e-308", "-1e-308", "0e-308", "-0", "123.4500", "1.234567890123456789e-308"} {
		t.Run(raw, func(t *testing.T) {
			rat, ok := new(big.Rat).SetString(raw)
			if !ok {
				t.Fatal("invalid oracle fixture")
			}
			want := json.Number(strings.TrimRight(strings.TrimRight(rat.FloatString(372), "0"), "."))
			if got := performanceCanonicalNumber(json.Number(raw)); got != want {
				t.Fatalf("canonical decimal=%v want %v", got, want)
			}
		})
	}
	for _, raw := range []string{strings.Repeat("9", 65), "1e309", "1e-309", "0e999999999"} {
		if got := performanceCanonicalNumber(json.Number(raw)); got != json.Number(raw) {
			t.Fatal("outside-bound number changed lexical form")
		}
	}
}

func TestPerformanceFixedDecimalExtractionMatchesLegacy(t *testing.T) {
	legacy := func(raw string) (*int64, *float64) {
		if len(raw) == 0 || len(raw) > 64 || raw[0] < '0' || raw[0] > '9' || !json.Valid([]byte(raw)) {
			return nil, nil
		}
		if i := strings.IndexAny(raw, "eE"); i >= 0 {
			e, err := strconv.Atoi(raw[i+1:])
			if err != nil || e < -308 || e > 308 {
				return nil, nil
			}
		}
		r, ok := new(big.Rat).SetString(raw)
		if !ok || r.Sign() < 0 {
			return nil, nil
		}
		var ns *int64
		scaled := new(big.Rat).Mul(new(big.Rat).Set(r), big.NewRat(1_000_000, 1))
		if scaled.Cmp(big.NewRat(86_400_000_000_000, 1)) <= 0 {
			q, rem := new(big.Int), new(big.Int)
			q.QuoRem(scaled.Num(), scaled.Denom(), rem)
			if rem.Lsh(rem, 1).Cmp(scaled.Denom()) >= 0 {
				q.Add(q, big.NewInt(1))
			}
			v := q.Int64()
			ns = &v
		}
		var rate *float64
		if r.Cmp(big.NewRat(10_000_000, 1)) <= 0 {
			v, _ := r.Float64()
			if !(v == 0 && r.Sign() > 0) {
				rate = &v
			}
		}
		return ns, rate
	}
	for _, raw := range []string{"0", "0.0000005", "0.000000499999999999999999", "0.000000500000000000000001", "86400000", "86400000.000000000000000000000000001", "10000000", "10000000.0000000000000000000000000001", "1e-308", "1e308", "1e309", "1e-309", "0." + strings.Repeat("1", 62), "0." + strings.Repeat("1", 63), "0." + strings.Repeat("0", 55) + "1e-308", "-0", "null", "01", "1e", "{}"} {
		t.Run(raw, func(t *testing.T) {
			wantNs, wantRate := legacy(raw)
			gotNs := performanceExtractMilliseconds(json.RawMessage(raw))
			gotRate := performanceExtractRate(json.RawMessage(raw))
			if !reflect.DeepEqual(gotNs, wantNs) || !reflect.DeepEqual(gotRate, wantRate) {
				t.Fatalf("decimal extraction differs from exact oracle: ns=%v/%v rate=%v/%v", gotNs, wantNs, gotRate, wantRate)
			}
		})
	}
}

func TestPerformanceFixedDecimalInvalidEvidenceIsSticky(t *testing.T) {
	q := performanceExtractQualification{EngineFamily: "llamacpp", APIPath: routeChatCompletions, Streaming: true, SchemaID: performanceExtractLlamaSchema}
	var numbers performanceNumbers
	numbers.merge(extractPerformanceObserved([]byte(`{"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`), q))
	if first := numbers.snapshot(); first.EngineDecodeNs == nil || *first.EngineDecodeNs != 250_000_000 {
		t.Fatal("fixture valid duration absent")
	}
	numbers.merge(extractPerformanceObserved([]byte(`{"timings":{"predicted_ms":null}}`), q))
	numbers.merge(extractPerformanceObserved([]byte(`{"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`), q))
	got := numbers.snapshot()
	if got.EngineDecodeNs != nil || got.EngineDecodeTokensPerSecond != nil || got.OutputTokens == nil || *got.OutputTokens != 11 {
		t.Fatal("invalid decimal evidence was restored or damaged independent count")
	}
}

func TestPerformanceFixedDecimalExactBinaryMaximum(t *testing.T) {
	for _, maximum := range []float64{0, 0.1, 0.5, math.SmallestNonzeroFloat64, 1e-308, math.MaxFloat64} {
		for _, raw := range []string{"0", "0.1", "0.1000000000000000055511151231257827021181583404541015625", "0.1000000000000000055511151231257827021181583404541015626", "0.5", "1e-308", "1e308", "0.000000000000001e-308", "0.0000000000000001e-308"} {
			rat, ok := new(big.Rat).SetString(raw)
			if !ok {
				t.Fatal("invalid oracle fixture")
			}
			var want *float64
			if rat.Cmp(new(big.Rat).SetFloat64(maximum)) <= 0 {
				v, _ := rat.Float64()
				if !(v == 0 && rat.Sign() > 0) {
					want = &v
				}
			}
			got := performanceExtractDecimal(json.RawMessage(raw), maximum)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("exact binary maximum comparison differs: raw=%s maximum=%.17g got=%v want=%v", raw, maximum, got, want)
			}
		}
	}
}

func TestPerformancePreparationAllowanceTransfer(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	body := []byte(`{"model":"m","stream":true,"temperature":1}`)
	// Seven lexical tokens; only the decoded keys (22 bytes) are retained.
	// The ignored model value has no decoded arena allocation.
	const wantStorage int64 = 136*7 + 22 + 24*3 + 4*8192
	n, err := performancePreparationBytes(body)
	if err != nil || n != wantStorage {
		t.Fatalf("preparation capacity=%d err=%v, want %d", n, err, wantStorage)
	}
	root, err := newBufferBudget(96<<10+wantStorage, 0)
	if err != nil {
		t.Fatal(err)
	}
	scope, err := root.newRequest(1 << 30)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.WithValue(context.Background(), ownedInferenceContextKey{}, &ownedInferenceBuffers{scope: scope})
	ctx, release, err := preparePerformanceBuffers(ctx, body)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	a, err := p.soleFacade().newPerformanceAttemptOwned(ctx, "m", routeChatCompletions, body, 0)
	if err != nil || a == nil {
		t.Fatalf("transferred full-root allowance was recharged: attempt=%v err=%v", a != nil, err)
	}
	if root.used != 96<<10+wantStorage {
		t.Fatalf("live charge=%d", root.used)
	}
	release()
	if root.used != 0 || root.live != 0 {
		t.Fatalf("released preparation retained bytes=%d leases=%d", root.used, root.live)
	}
}

func TestPerformanceQueueOwnsScalarSnapshot(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	key, _, ok := p.soleFacade().performanceFingerprint("m", routeChatCompletions, []byte(`{"model":"m","stream":true}`))
	if !ok {
		t.Fatal("fixture identity refused")
	}
	upstreamCount := int64(7)
	frame := []byte("data: {\"choices\":[]}\n\ndata: [DONE]\n\n")
	r, err := newStreamFrameReader(streamFrameConfig{
		Protocol: streamProtocolSSE, Upstream: io.NopCloser(bytes.NewReader(frame)), Budget: newFrameBudget(64 << 20),
		Extract: func([]byte) performanceExtractMetadata {
			return performanceExtractMetadata{OutputTokens: &upstreamCount}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var prefix [1]byte
	if n, err := r.Read(prefix[:]); err != nil || n != 1 {
		t.Fatalf("first frame read=%d err=%v", n, err)
	}
	r.mu.Lock()
	reusedFrame := r.buf
	r.mu.Unlock()
	if len(reusedFrame) == 0 {
		t.Fatal("fixture did not retain frame storage")
	}
	if _, err := io.Copy(io.Discard, r); err != nil {
		t.Fatal(err)
	}
	snapshot := r.Result()
	if snapshot.Numeric.OutputTokens == nil || *snapshot.Numeric.OutputTokens != 7 {
		t.Fatal("fixture numeric snapshot absent")
	}
	suppliedCount := *snapshot.Numeric.OutputTokens
	supplied := performancewire.Sample{
		Key: key, Profile: performancewire.Profile{EngineFamily: "llamacpp"},
		Measurement: performancewire.Measurement{Outcome: performancewire.OutcomeCompleted, OutputTokens: &suppliedCount},
	}
	p.enqueuePerformance(supplied)
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	upstreamCount = 91
	*supplied.OutputTokens = 93
	*snapshot.Numeric.OutputTokens = 94
	r.mu.Lock()
	if r.numeric.value.OutputTokens == nil {
		r.mu.Unlock()
		t.Fatal("fixture reader numeric state absent")
	}
	*r.numeric.value.OutputTokens = 92
	r.mu.Unlock()
	for i := range reusedFrame {
		reusedFrame[i] = 'x'
	}
	for i := range frame {
		frame[i] = 'y'
	}
	samples := performanceSamples(p)
	if len(samples) != 1 || samples[0].OutputTokens == nil || *samples[0].OutputTokens != 7 {
		t.Fatalf("queue scalar changed after upstream/reader/frame reuse: samples=%+v", samples)
	}
}

func TestPerformanceQueueCopiesEveryNullableScalar(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	key, _, ok := p.soleFacade().performanceFingerprint("m", routeChatCompletions, []byte(`{"model":"m"}`))
	if !ok {
		t.Fatal("fixture identity refused")
	}
	complete := true
	one := int64(1)
	phase := int64(1_000_000_000)
	rate := float64(1)
	s := performancewire.Sample{Key: key, Profile: performancewire.Profile{EngineFamily: "llamacpp"}, Measurement: performancewire.Measurement{
		Outcome: performancewire.OutcomeCompleted, ProtocolComplete: &complete, AdmissionWaitMs: &one, HeaderMs: &one, FirstBodyMs: &one, FirstTokenMs: &one, TotalMs: &one,
		PromptTokens: &one, CachedPromptTokens: &one, OutputTokens: &one, PrefillNs: &phase, DecodeNs: &phase, EngineDecodeNs: &phase, LoadNs: &phase, EngineDecodeTokensPerSecond: &rate,
	}}
	if err := s.Measurement.Validate(); err != nil {
		t.Fatal(err)
	}
	p.enqueuePerformance(s)
	complete = false
	one = 99
	phase = 99
	rate = 99
	samples := performanceSamples(p)
	if len(samples) != 1 {
		t.Fatalf("queued samples=%d", len(samples))
	}
	got := samples[0].Measurement
	if got.ProtocolComplete == nil || !*got.ProtocolComplete || got.EngineDecodeTokensPerSecond == nil || *got.EngineDecodeTokensPerSecond != 1 {
		t.Fatal("queue retained boolean/rate alias")
	}
	for _, v := range []*int64{got.AdmissionWaitMs, got.HeaderMs, got.FirstBodyMs, got.FirstTokenMs, got.TotalMs, got.PromptTokens, got.CachedPromptTokens, got.OutputTokens} {
		if v == nil || *v != 1 {
			t.Fatal("queue retained integer scalar alias")
		}
	}
	for _, v := range []*int64{got.PrefillNs, got.DecodeNs, got.EngineDecodeNs, got.LoadNs} {
		if v == nil || *v != 1_000_000_000 {
			t.Fatal("queue retained phase scalar alias")
		}
	}
}

func TestPerformanceQueueOwnsProfileContextScalar(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	key, _, ok := p.soleFacade().performanceFingerprint("m", routeChatCompletions, []byte(`{"model":"m"}`))
	if !ok {
		t.Fatal("fixture identity refused")
	}
	contextTokens := int64(4096)
	supplied := performancewire.Sample{
		Key:         key,
		Profile:     performancewire.Profile{EngineFamily: "llamacpp", ContextTokens: &contextTokens},
		Measurement: performancewire.Measurement{Outcome: performancewire.OutcomeCompleted},
	}
	p.enqueuePerformance(supplied)
	*supplied.Profile.ContextTokens = 8192
	got := performanceOne(t, p)
	if got.Profile.ContextTokens == nil || *got.Profile.ContextTokens != 4096 {
		t.Fatalf("queued profile context=%v, want entry-owned 4096", got.Profile.ContextTokens)
	}
	if got.Profile.ContextTokens == supplied.Profile.ContextTokens {
		t.Fatal("queued profile retained supplied context pointer")
	}
}

func TestPerformanceQueueIdentifiersOwnDecodedStorage(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	budget := bufferTestBudget(t, 192<<20, 0)
	p.buffers = budget
	// The decoded identifier is exactly the admitted 256-byte boundary. Its
	// normalized view points inside real request-owned decoded storage.
	padded := strings.Repeat(" ", 125) + "m" + strings.Repeat(" ", 130)
	r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"`+padded+`"}`)))
	defer owner.close()
	_, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(model) != 256 || model != padded {
		t.Fatalf("decoded fixture length=%d, want preserved 256-byte identifier", len(model))
	}
	view := strings.TrimSpace(model)
	if view != "m" || unsafe.StringData(view) != unsafe.StringData(model[125:]) {
		t.Fatal("fixture normalized view does not borrow decoded model storage")
	}
	borrower := owner.borrowParserAllowance()
	if borrower == nil {
		t.Fatal("fixture parser allowance absent")
	}
	defer borrower.close()
	// Every insertion field deliberately receives an actual input-derived view.
	// Scope is a fixed protocol constant; Producer is stamped from the proxy.
	p.runID = view
	supplied := performancewire.Sample{
		Key:         performancewire.ProfileKey{Host: view, Engine: view, Model: view, Fingerprint: view},
		Profile:     performancewire.Profile{EngineFamily: view, Quantization: view},
		Measurement: performancewire.Measurement{Outcome: performancewire.OutcomeCompleted},
	}
	p.enqueuePerformance(supplied)
	cleaned := make(chan struct{})
	go func() {
		defer close(cleaned)
		owner.close()
		borrower.close()
	}()
	// Queue access may overlap borrower cleanup; all strings remain immutable.
	got := performanceOne(t, p)
	waitChan(t, cleaned, "request and parser borrower cleanup")
	bufferAssertTotals(t, budget, 0, 0, 0)
	for _, field := range []struct {
		name string
		got  string
	}{
		{"producer", got.Producer},
		{"host", got.Key.Host},
		{"engine", got.Key.Engine},
		{"model", got.Key.Model},
		{"fingerprint", got.Key.Fingerprint},
		{"engine family", got.Profile.EngineFamily},
		{"quantization", got.Profile.Quantization},
	} {
		if field.got != "m" {
			t.Fatalf("queued %s changed value", field.name)
		}
		if unsafe.StringData(field.got) == unsafe.StringData(view) {
			t.Fatalf("queued %s retains decoded request backing after root drain", field.name)
		}
	}
	runtime.KeepAlive(model)
}

func TestPerformanceQueueOwnsSuppliedOutcomeStorage(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	budget := bufferTestBudget(t, 192<<20, 0)
	p.buffers = budget
	padded := strings.Repeat(" ", 125) + "completed" + strings.Repeat(" ", 122)
	r, owner := p.ownInferenceBuffers(httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"`+padded+`"}`)))
	defer owner.close()
	_, model, err := bufferBodyAndModel(httptest.NewRecorder(), r, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	view := strings.TrimSpace(model)
	if len(model) != 256 || view != "completed" {
		t.Fatalf("decoded outcome fixture bytes=%d value=%q", len(model), view)
	}
	key, _, ok := p.soleFacade().performanceFingerprint("m", routeChatCompletions, []byte(`{"model":"m"}`))
	if !ok {
		t.Fatal("performance identity unavailable")
	}
	// Production observers supply enum constants. The insertion seam also owns
	// a validated supplied enum view, so its no-alias guarantee is complete.
	p.enqueuePerformance(performancewire.Sample{
		Key: key, Profile: performancewire.Profile{EngineFamily: "llamacpp"},
		Measurement: performancewire.Measurement{Outcome: performancewire.Outcome(view)},
	})
	owner.close()
	bufferAssertTotals(t, budget, 0, 0, 0)
	got := performanceOne(t, p)
	if got.Outcome != performancewire.OutcomeCompleted || unsafe.StringData(string(got.Outcome)) == unsafe.StringData(view) {
		t.Fatal("queued supplied outcome changed value or retained request-owned storage")
	}
	runtime.KeepAlive(model)
}

// The oracle deliberately retains the old stdlib decoder/map/encoder path in
// this fixture only. No production canonical writer or identity helper is used.
func TestPerformanceFingerprintMatchesLegacyJSONHash(t *testing.T) {
	legacy := func(t *testing.T, body []byte) string {
		t.Helper()
		dec := json.NewDecoder(bytes.NewReader(body))
		dec.UseNumber()
		var fields map[string]any
		if err := dec.Decode(&fields); err != nil {
			t.Fatal(err)
		}
		var canonical func(any) any
		canonical = func(v any) any {
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
				if rat, ok := new(big.Rat).SetString(string(n)); ok {
					return json.Number(strings.TrimRight(strings.TrimRight(rat.FloatString(372), "0"), "."))
				}
			case map[string]any:
				for k, value := range n {
					n[k] = canonical(value)
				}
			case []any:
				for i, value := range n {
					n[i] = canonical(value)
				}
			}
			return v
		}
		options := map[string]any{}
		for key, value := range fields {
			switch key {
			case "model", "messages", "prompt", "input":
				continue
			case "tools":
				options["hasTools"] = value != nil
			default:
				options[key] = canonical(value)
			}
		}
		encoded, err := json.Marshal(struct {
			Run, Settings, Launch, Model, API, Schema string
			Options                                   map[string]any
		}{"runtime-1", strings.Repeat("a", 64), strings.Repeat("b", 64), strings.Repeat("c", 64), routeChatCompletions, "llamacpp:llamacpp-b11146-7fe450e1", options})
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(encoded)
		return hex.EncodeToString(sum[:])
	}
	cases := []struct {
		name string
		body []byte
	}{
		{"empty options", []byte(`{"model":"m","messages":[]}`)},
		{"tools sort and exact numbers", []byte(`{"model":"m","temperature":1e0,"tools":[],"stream":true,"max_tokens":9007199254740993.0,"top_p":1e-20}`)},
		{"nested escapes", []byte(`{"model":"m","tools":null,"response_format":{"type":"json_schema","json_schema":{"name":"x<>&\u2028\u2029","schema":{"b":[1.0,"a\"b\\c"],"a":"\ufffd"}}}}`)},
		{"invalid UTF8 decoded by old decoder", append(append([]byte(`{"model":"m","stop":"`), 0xff), []byte(`"}`)...)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := performanceProxy(t, "llamacpp")
			key, _, ok := p.soleFacade().performanceFingerprint("m", routeChatCompletions, tc.body)
			if !ok {
				t.Fatal("valid fixture refused")
			}
			if want := legacy(t, tc.body); key.Fingerprint != want {
				t.Fatalf("legacy fingerprint changed: got %s want %s", key.Fingerprint, want)
			}
		})
	}
}

func TestPerformancePreparationManyTinyIgnoredValues(t *testing.T) {
	makeBody := func(items int) []byte {
		return []byte(`{"model":"m","messages":[` + strings.TrimSuffix(strings.Repeat(`{"role":"user","content":"x"},`, items), ",") + `],"tools":[{"type":"function","function":{"name":"f"}}],"stream":true}`)
	}
	short, long := makeBody(1), makeBody(20000)
	a, err := performancePreparationBytes(short)
	if err != nil {
		t.Fatal(err)
	}
	b, err := performancePreparationBytes(long)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatalf("ignored object count retained descriptors: short=%d long=%d", a, b)
	}
	x, ok := performanceOptionsDigest(short, "llamacpp")
	if !ok {
		t.Fatal("short ignored fixture refused")
	}
	y, ok := performanceOptionsDigest(long, "llamacpp")
	if !ok || x != y {
		t.Fatal("many ignored values changed options")
	}
	for _, raw := range []string{`{"model":"m","messages":[{"x":{"K":1,"\u212a":2}}]}`, `{"model":"m","tools":[{"function":{"x":1,"X":2}}]}`} {
		if _, ok := performanceOptionsDigest([]byte(raw), "llamacpp"); ok {
			t.Fatal("duplicate ignored subtree key accepted")
		}
	}
}

func TestPerformancePreparationSkipsIgnoredContent(t *testing.T) {
	short := []byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"stream":true}`)
	long := []byte(`{"model":"` + strings.Repeat("m", 1<<20) + `","messages":[{"role":"user","content":"` + strings.Repeat("x", 1<<20) + `"}],"stream":true}`)
	a, err := performancePreparationBytes(short)
	if err != nil {
		t.Fatal(err)
	}
	b, err := performancePreparationBytes(long)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatalf("ignored payload changed arena capacity: short=%d long=%d", a, b)
	}
	x, ok := performanceOptionsDigest(short, "llamacpp")
	if !ok {
		t.Fatal("short request refused")
	}
	y, ok := performanceOptionsDigest(long, "llamacpp")
	if !ok || x != y {
		t.Fatal("ignored content changed option digest")
	}
}

func TestPerformancePreparationPreservesNestedOptions(t *testing.T) {
	a, ok := performanceOptionsDigest([]byte(`{"model":"m","options":{"b":[1e0,"a\u003cb"],"a":9007199254740993.0}}`), "ollama")
	if !ok {
		t.Fatal("valid nested options refused")
	}
	b, ok := performanceOptionsDigest([]byte(`{"options":{"a":9007199254740993,"b":[1,"a<b"]},"model":"alias"}`), "ollama")
	if !ok || a != b {
		t.Fatal("nested canonical options changed digest")
	}
	for _, raw := range []string{
		`{"model":"m","messages":[{"role":"user","ROLE":"assistant"}]}`,
		`{"model":"m","options":{"k":1,"\u006b":1}}`,
		`{"model":"m","options":{"K":1,"\u212a":1}}`,
	} {
		if _, ok := performanceOptionsDigest([]byte(raw), "ollama"); ok {
			t.Fatal("duplicate decoded/folded key accepted")
		}
	}
}

func TestPerformancePreparationImpossibleBeforeAllocation(t *testing.T) {
	root, err := newBufferBudget(120000, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	scope, err := root.newRequest(1 << 30)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.WithValue(context.Background(), ownedInferenceContextKey{}, &ownedInferenceBuffers{scope: scope})
	_, release, err := preparePerformanceBuffers(ctx, []byte(`{"model":"m","stream":true,"temperature":1}`))
	if release != nil {
		release()
	}
	if bufferRequestStatus(err) != http.StatusRequestEntityTooLarge {
		t.Fatalf("impossible preparation error=%v", err)
	}
	if root.used != 0 || root.live != 0 || root.waiters.Len() != 0 {
		t.Fatalf("failed preparation retained storage or queued: bytes=%d leases=%d waiters=%d", root.used, root.live, root.waiters.Len())
	}
}

func TestPerformanceStaleIdentityDrops(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	a := p.soleFacade().newPerformanceAttempt(context.Background(), "m", routeChatCompletions, []byte(`{"model":"m","stream":true}`), 0)
	if a == nil {
		t.Fatal("no attempt")
	}
	a.start = time.Now()
	id := performanceIdentity()
	id.EngineRunID = "runtime-2"
	p.admission.setModelIdentity(map[string]nodepolicy.EngineIdentity{"llamacpp": id})
	a.finish(&statusCapture{}, true)
	if len(performanceSamples(p)) != 0 || p.performance.drops[2] != 1 {
		t.Fatal("stale identity relabeled")
	}
}

func TestPerformanceRetryAttemptsOnce(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	var hits atomic.Int32
	u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "data: "+performanceLlamaFinal+"\n\ndata: [DONE]\n\n"); err != nil {
			t.Error(err)
		}
	}))
	defer u.Close()
	setBackend(t, p.soleFacade(), u.URL, true)
	w := httptest.NewRecorder()
	p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("retry status=%d", w.Code)
	}
	s := performanceSamples(p)
	if hits.Load() != 2 || len(s) != 2 || s[0].Seq != 1 || s[1].Seq != 2 || s[0].Outcome != performancewire.OutcomeFailed || s[1].Outcome != performancewire.OutcomeCompleted {
		t.Fatalf("retry samples=%+v hits=%d", s, hits.Load())
	}
}

func TestPerformanceFailedDialIsAttempt(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := l.Addr().String()
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	setBackend(t, p.soleFacade(), "http://"+address, true)
	w := httptest.NewRecorder()
	p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)), "peer")
	s := performanceOne(t, p)
	if s.Outcome != performancewire.OutcomeFailed || s.HeaderMs != nil || s.FirstBodyMs != nil || s.ProtocolComplete != nil {
		t.Fatalf("failed dial=%+v", s)
	}
}

func TestPerformanceRefusalHasNoAttempt(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	pol := nodepolicy.Default()
	pol.Availability = nodepolicy.Paused
	p.admission.setPolicy(pol)
	w := httptest.NewRecorder()
	p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)), "peer")
	if w.Code != http.StatusServiceUnavailable || len(performanceSamples(p)) != 0 {
		t.Fatalf("refusal=%d samples=%d", w.Code, len(performanceSamples(p)))
	}
}

func TestPerformanceNonstreamUnobserved(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	const body = `{"choices":[{"finish_reason":"stop"}],"timings":{"predicted_n":11,"predicted_ms":250,"predicted_per_second":40}}`
	u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := io.WriteString(w, body); err != nil {
			t.Error(err)
		}
	}))
	defer u.Close()
	setBackend(t, p.soleFacade(), u.URL, true)
	w := httptest.NewRecorder()
	p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":false}`)), "peer")
	s := performanceOne(t, p)
	if w.Body.String() != body || s.Outcome != performancewire.OutcomeCompleted || s.ProtocolComplete != nil || s.EngineDecodeTokensPerSecond != nil || s.OutputTokens != nil {
		t.Fatalf("nonstream observed/rewritten: %+v", s)
	}
}

type performanceBlockedWriter struct {
	entered, release chan struct{}
	once             sync.Once
}

func (w *performanceBlockedWriter) Read([]byte) (int, error) { return 0, io.EOF }
func (w *performanceBlockedWriter) Write(b []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	<-w.release
	return len(b), nil
}

func TestPerformanceSenderNeverBlocksInference(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	// Existing admission notifications share the codec's synchronous write gate.
	// Isolate the telemetry producer from that independent notification path.
	p.admission.notify = func(string, any) {}
	writer := &performanceBlockedWriter{entered: make(chan struct{}), release: make(chan struct{})}
	p.codec = NewCodec(writer)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { defer close(done); p.sendPerformance(ctx) }()
	defer func() {
		cancel()
		close(writer.release)
		waitChan(t, done, "sender returns after write completes")
	}()
	u := performanceUpstream(t, "data: "+performanceLlamaFinal+"\n\ndata: [DONE]\n\n")
	setBackend(t, p.soleFacade(), u.URL, true)
	request := func() {
		p.soleFacade().serveIngress(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":true}`)), "peer")
	}
	firstReturned := make(chan struct{})
	go func() { defer close(firstReturned); request() }()
	waitChan(t, writer.entered, "sender blocked in codec")
	waitChan(t, firstReturned, "first inference returns")
	returned := make(chan struct{})
	go func() {
		defer close(returned)
		for i := 0; i < performanceQueueCapacity+3; i++ {
			request()
		}
	}()
	waitChan(t, returned, "inference and full FIFO return")
	p.performance.mu.Lock()
	size, drops := p.performance.size, p.performance.drops[0]
	p.performance.mu.Unlock()
	if size != performanceQueueCapacity || drops != 3 {
		t.Fatalf("size=%d drops=%d", size, drops)
	}
	cancel()
	shutdown := make(chan struct{})
	go func() { p.shutdown(context.Background()); close(shutdown) }()
	waitChan(t, shutdown, "proxy shutdown while telemetry write waits")
}

type performanceGateContext struct {
	context.Context
	waiting chan struct{}
	checks  atomic.Int32
}

func (c *performanceGateContext) Err() error {
	err := c.Context.Err()
	if c.checks.Add(1) == 2 {
		close(c.waiting)
	}
	return err
}

func TestPerformanceSenderCancelsAtWriteGate(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	writer := &performanceBlockedWriter{entered: make(chan struct{}), release: make(chan struct{})}
	p.codec = NewCodec(writer)
	holder := make(chan struct{})
	go func() {
		defer close(holder)
		if err := p.codec.Notify("other", nil); err != nil {
			t.Error(err)
		}
	}()
	waitChan(t, writer.entered, "other notification holds write gate")
	parent, cancel := context.WithCancel(context.Background())
	ctx := &performanceGateContext{Context: parent, waiting: make(chan struct{})}
	done := make(chan struct{})
	defer func() {
		cancel()
		close(writer.release)
		waitChan(t, holder, "other notification released")
		waitChan(t, done, "sender cleaned up")
	}()
	// A queued sample avoids the idle select. Wait for the second Err check
	// to capture nil, so cancellation cannot pass via an early loop return.
	p.performance.items[0] = performancewire.Sample{}
	p.performance.size = 1
	go func() { defer close(done); p.sendPerformance(ctx) }()
	select {
	case <-ctx.waiting:
	case <-time.After(time.Second):
		t.Fatal("sender did not reach notification dispatch")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("sender did not exit while another notification held the write gate")
	}
}

func TestPerformanceUnobservedCopyOutcome(t *testing.T) {
	for _, scenario := range []string{"headers then stall", "body read error", "normal body"} {
		t.Run(scenario, func(t *testing.T) {
			setForTest(t, &firstBodyTimeout, 20*time.Millisecond)
			p := performanceProxy(t, "llamacpp")
			var hits atomic.Int32
			u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				w.Header().Set("Content-Type", "application/json")
				if scenario == "body read error" {
					w.Header().Set("Content-Length", "100")
				}
				w.WriteHeader(http.StatusOK)
				if scenario == "headers then stall" {
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				}
				if _, err := io.WriteString(w, `{}`); err != nil {
					t.Error(err)
				}
			}))
			defer u.Close()
			setBackend(t, p.soleFacade(), u.URL, true)
			w := httptest.NewRecorder()
			p.soleFacade().handleHTTP(w, httptest.NewRequest(http.MethodPost, routeChatCompletions, strings.NewReader(`{"model":"m","stream":false}`)))
			samples := performanceSamples(p)
			wantAttempts, wantOutcome := 1, performancewire.OutcomeFailed
			if scenario == "headers then stall" {
				wantAttempts = maxDispatchAttempts
				if w.Code != http.StatusGatewayTimeout {
					t.Fatalf("status=%d, want 504", w.Code)
				}
			}
			if scenario == "normal body" {
				wantOutcome = performancewire.OutcomeCompleted
				if w.Code != http.StatusOK || w.Body.String() != `{}` {
					t.Fatalf("status/body=%d %q", w.Code, w.Body.String())
				}
			}
			if len(samples) != wantAttempts || int(hits.Load()) != wantAttempts {
				t.Fatalf("samples=%d dispatches=%d, want %d", len(samples), hits.Load(), wantAttempts)
			}
			for i, sample := range samples {
				if sample.Outcome != wantOutcome || sample.ProtocolComplete != nil {
					t.Errorf("attempt %d measurement=%+v, want outcome %s and unknown protocol", i, sample.Measurement, wantOutcome)
				}
			}
		})
	}
}

func TestPerformanceSenderStopsOutsideWrite(t *testing.T) {
	p := performanceProxy(t, "llamacpp")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); p.sendPerformance(ctx) }()
	cancel()
	waitChan(t, done, "idle sender stops")
}

func TestPerformanceFIFOCompletionOrderAndIncarnation(t *testing.T) {
	w := &recordingWriter{}
	p := performanceProxy(t, "llamacpp")
	p.codec = NewCodec(w)
	f := p.soleFacade()
	body := []byte(`{"model":"m","stream":true}`)
	first := f.newPerformanceAttempt(context.Background(), "m", routeChatCompletions, body, time.Millisecond)
	second := f.newPerformanceAttempt(context.Background(), "m", routeChatCompletions, body, 2*time.Millisecond)
	first.start = time.Now()
	second.start = time.Now()
	second.finish(&statusCapture{}, true)
	first.finish(&statusCapture{}, true)
	s := performanceSamples(p)
	if len(s) != 2 || s[0].Seq != 1 || s[1].Seq != 2 || *s[0].AdmissionWaitMs != 2 || *s[1].AdmissionWaitMs != 1 {
		t.Fatalf("enqueue order=%+v", s)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); p.sendPerformance(ctx) }()
	deadline := time.After(2 * time.Second)
	for len(frames(t, w)) < 2 {
		select {
		case <-deadline:
			t.Fatal("sender did not drain")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	cancel()
	waitChan(t, done, "sender drained and stopped")
	for i, frame := range frames(t, w) {
		var sample performancewire.Sample
		if err := json.Unmarshal(frame.Params, &sample); err != nil {
			t.Fatal(err)
		}
		if frame.Method != "performance:sample" || sample.Seq != uint64(i+1) || sample.Producer != p.runID {
			t.Fatalf("sent=%+v", sample)
		}
	}
	p2 := performanceProxy(t, "llamacpp")
	if p2.runID == p.runID {
		t.Fatal("process incarnation reused")
	}
}

func TestPerformanceDeliveryFailureExcludesRate(t *testing.T) {
	for _, cause := range []string{"write failure", "panic", "client cancellation"} {
		t.Run(cause, func(t *testing.T) {
			p := performanceProxy(t, "strata")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			a := p.soleFacade().newPerformanceAttempt(ctx, "m", routeChatCompletions, []byte(`{"model":"m","stream":true}`), 0)
			a.start = time.Now()
			r, err := newStreamFrameReader(streamFrameConfig{Budget: newFrameBudget(64 << 20), Protocol: streamProtocolSSE, Upstream: io.NopCloser(strings.NewReader("data: " + performanceStrataFinal + "\n\ndata: [DONE]\n\n")), Extract: func(b []byte) performanceExtractMetadata { return extractPerformanceObserved(b, a.qualification) }})
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			if _, err := io.Copy(io.Discard, r); err != nil {
				t.Fatal(err)
			}
			a.reader = r
			sc := &statusCapture{}
			returned := true
			switch cause {
			case "write failure":
				sc.wroteErr = errors.New("client write")
			case "panic":
				returned = false
			case "client cancellation":
				cancel()
			}
			a.finish(sc, returned)
			s := performanceOne(t, p)
			if s.Outcome == performancewire.OutcomeCompleted || s.ProtocolComplete == nil || *s.ProtocolComplete {
				t.Fatalf("delivery failure qualified: %+v", s)
			}
		})
	}
}

func TestPerformanceOllamaNativeDestination(t *testing.T) {
	p := performanceProxy(t, "ollama")
	u := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		if _, err := io.WriteString(w, "{\"done\":true,\"eval_count\":3,\"eval_duration\":1000000000}\n"); err != nil {
			t.Error(err)
		}
	}))
	defer u.Close()
	setBackend(t, p.soleFacade(), u.URL, true)
	w := httptest.NewRecorder()
	p.soleFacade().serveIngress(w, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"m","stream":true,"prompt":"x"}`)), "peer")
	s := performanceOne(t, p)
	if s.Key.Model != "m:latest" || s.Outcome != performancewire.OutcomeCompleted || s.DecodeNs == nil || *s.DecodeNs != 1000000000 || s.OutputTokens == nil || *s.OutputTokens != 3 || s.EngineDecodeTokensPerSecond != nil {
		t.Fatalf("native measurement=%+v", s)
	}
}

func TestPerformanceStreamNumericSeam(t *testing.T) {
	test := func(name, payload string, check func(*testing.T, streamFrameResult)) {
		t.Run(name, func(t *testing.T) {
			read := func(observe bool) ([]byte, streamFrameResult) {
				config := streamFrameConfig{Budget: newFrameBudget(64 << 20), Protocol: streamProtocolSSE, Upstream: io.NopCloser(strings.NewReader(payload))}
				if observe {
					config.Extract = func(b []byte) performanceExtractMetadata {
						return extractPerformanceObserved(b, performanceExtractQualification{EngineFamily: "strata", APIPath: routeChatCompletions, Streaming: true, SchemaID: performanceExtractStrataSchema})
					}
				}
				r, err := newStreamFrameReader(config)
				if err != nil {
					t.Fatal(err)
				}
				defer r.Close()
				b, err := io.ReadAll(r)
				if err != nil {
					t.Fatal(err)
				}
				return b, r.Result()
			}
			plain, baseline := read(false)
			observed, result := read(true)
			if !bytes.Equal(plain, observed) || baseline.Outcome != result.Outcome || baseline.Reason != result.Reason {
				t.Fatal("numeric callback changed SC bytes/result")
			}
			check(t, result)
		})
	}
	completed := func(t *testing.T, r streamFrameResult) {
		t.Helper()
		if r.Outcome != streamOutcomeCompleted || r.Numeric.EngineDecodeTokensPerSecond == nil || *r.Numeric.EngineDecodeTokensPerSecond != 55 {
			t.Fatalf("result=%+v", r)
		}
	}
	test("native rate", "data: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", completed)
	test("tool finish", "data: "+strings.Replace(performanceStrataFinal, "stop", "tool_calls", 1)+"\n\ndata: [DONE]\n\n", completed)
	test("reasoning finish", "data: "+strings.Replace(performanceStrataFinal, "stop", "length", 1)+"\n\ndata: [DONE]\n\n", completed)
	test("usage timings split", "data: {\"choices\":[],\"usage\":{\"completion_tokens\":20}}\n\ndata: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", completed)
	test("repeated never adds", "data: "+performanceStrataFinal+"\n\ndata: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", completed)
	test("conflicting counters", "data: "+performanceStrataFinal+"\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":21}}\n\ndata: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Numeric.OutputTokens != nil || r.Numeric.EngineDecodeTokensPerSecond != nil {
			t.Fatalf("conflict restored: %+v", r.Numeric)
		}
	})
	test("sticky MCP", "data: {\"choices\":[],\"strata_mcp\":{}}\n\ndata: "+performanceStrataFinal+"\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if !r.Numeric.PhaseDisqualified || r.Numeric.EngineDecodeTokensPerSecond != nil {
			t.Fatal("MCP requalified")
		}
	})
	test("invalid later timing", "data: "+performanceStrataFinal+"\n\ndata: {\"choices\":[],\"timings\":{\"predicted_ms\":-1}}\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Numeric.EngineDecodeTokensPerSecond != nil {
			t.Fatal("invalid later timing retained rate")
		}
	})
	test("invalid later cache", "data: {\"choices\":[],\"timings\":{\"cache_n\":4}}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":-1}}}\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Numeric.CachedPromptTokens != nil {
			t.Fatal("invalid later cache retained count")
		}
	})
	test("usage only", "data: {\"choices\":[],\"usage\":{\"completion_tokens\":20}}\n\ndata: [DONE]\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Numeric.OutputTokens == nil || r.Numeric.EngineDecodeTokensPerSecond != nil {
			t.Fatal("usage invented speed")
		}
	})
	test("interrupted", "data: "+performanceStrataFinal+"\n\ndata: {", func(t *testing.T, r streamFrameResult) {
		if r.Outcome != streamOutcomeInterrupted {
			t.Fatalf("outcome=%v", r.Outcome)
		}
	})
	test("error", "data: {\"error\":{\"message\":\"upstream\"}}\n\n", func(t *testing.T, r streamFrameResult) {
		if r.Outcome != streamOutcomeUpstreamError {
			t.Fatalf("outcome=%v", r.Outcome)
		}
	})
}

func TestPerformanceResultOwnsNumericCopy(t *testing.T) {
	r, err := newStreamFrameReader(streamFrameConfig{Budget: newFrameBudget(64 << 20), Protocol: streamProtocolNDJSON, Upstream: io.NopCloser(strings.NewReader("{\"done\":true,\"eval_count\":3,\"eval_duration\":1000000000}\n")), Extract: func(b []byte) performanceExtractMetadata {
		return extractPerformancePayload(b, performanceExtractQualification{EngineFamily: "ollama", APIPath: "/api/generate", Streaming: true})
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := io.Copy(io.Discard, r); err != nil {
		t.Fatal(err)
	}
	a := r.Result()
	b := r.Result()
	if !reflect.DeepEqual(a, b) || a.Numeric.OutputTokens == nil {
		t.Fatal("missing stable numeric copy")
	}
	*a.Numeric.OutputTokens = 99
	if *r.Result().Numeric.OutputTokens != 3 {
		t.Fatal("result aliases reader storage")
	}
}

func TestPerformanceTimingReadFieldsInvalidateStickyEvidence(t *testing.T) {
	families := []struct {
		name, schema, final string
	}{
		{"llamacpp", performanceExtractLlamaSchema, `{"choices":[{"index":0,"finish_reason":"stop"}],"timings":{"prompt_n":10,"prompt_ms":200,"prompt_per_second":50,"prompt_per_token_ms":20,"cache_n":4,"predicted_n":11,"predicted_ms":250,"predicted_per_second":40,"predicted_per_token_ms":25},"usage":{"completion_tokens":11}}`},
		{"strata", performanceExtractStrataSchema, `{"choices":[{"index":0,"finish_reason":"stop"}],"timings":{"prompt_n":10,"prompt_ms":200,"prompt_per_second":50,"prompt_per_token_ms":20,"cache_n":4,"predicted_n":20,"predicted_ms":400,"predicted_per_second":55,"predicted_per_token_ms":18.182},"usage":{"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":4}}}`},
	}
	fields := []struct {
		name, container, key, affected string
		strataOnly                     bool
	}{
		{"prompt count", "timings", "prompt_n", "prompt", false},
		{"prompt interval", "timings", "prompt_ms", "prefill", false},
		{"prompt rate", "timings", "prompt_per_second", "prompt", false},
		{"prompt reciprocal", "timings", "prompt_per_token_ms", "prompt", false},
		{"cache count", "timings", "cache_n", "cache", false},
		{"output count", "timings", "predicted_n", "output", false},
		{"decode interval", "timings", "predicted_ms", "decode", false},
		{"decode rate", "timings", "predicted_per_second", "rate", false},
		{"decode reciprocal", "timings", "predicted_per_token_ms", "decode", false},
		{"usage output count", "usage", "completion_tokens", "output", false},
		{"usage cached count", "prompt_tokens_details", "cached_tokens", "cache", true},
	}
	for _, family := range families {
		t.Run(family.name, func(t *testing.T) {
			q := performanceExtractQualification{EngineFamily: family.name, APIPath: routeChatCompletions, Streaming: true, SchemaID: family.schema}
			for _, field := range fields {
				if field.strataOnly && family.name != "strata" {
					continue
				}
				t.Run(field.name, func(t *testing.T) {
					for _, invalid := range []string{`"bad"`, `null`, `-1`, `1e999`} {
						t.Run(invalid, func(t *testing.T) {
							patch := `{"` + field.container + `":{"` + field.key + `":` + invalid + `}}`
							if field.container == "prompt_tokens_details" {
								patch = `{"usage":` + patch + `}`
							}
							body := "data: " + family.final + "\n\ndata: " + patch + "\n\ndata: " + family.final + "\n\ndata: [DONE]\n\n"
							r, err := newStreamFrameReader(streamFrameConfig{Budget: newFrameBudget(64 << 20), Protocol: streamProtocolSSE, Upstream: io.NopCloser(strings.NewReader(body)), Extract: func(b []byte) performanceExtractMetadata {
								return extractPerformanceObserved(b, q)
							}})
							if err != nil {
								t.Fatal(err)
							}
							defer r.Close()
							if _, err := io.Copy(io.Discard, r); err != nil {
								t.Fatal(err)
							}
							result := r.Result()
							if result.Outcome != streamOutcomeCompleted {
								t.Fatalf("outcome=%v, want completed", result.Outcome)
							}
							n := result.Numeric
							switch field.affected {
							case "prompt":
								if n.PromptTokens != nil || n.PrefillNs != nil {
									t.Fatalf("invalid prompt evidence restored: %+v", n)
								}
							case "prefill":
								if n.PrefillNs != nil {
									t.Fatalf("invalid prefill interval restored: %+v", n)
								}
							case "cache":
								if n.CachedPromptTokens != nil {
									t.Fatalf("invalid cache count restored: %+v", n)
								}
							case "output":
								if n.OutputTokens != nil || n.EngineDecodeNs != nil || n.EngineDecodeTokensPerSecond != nil {
									t.Fatalf("invalid output evidence restored: %+v", n)
								}
							case "decode":
								if n.EngineDecodeNs != nil || n.EngineDecodeTokensPerSecond != nil {
									t.Fatalf("invalid decode phase restored: %+v", n)
								}
							case "rate":
								if n.EngineDecodeTokensPerSecond != nil {
									t.Fatalf("invalid decode rate restored: %+v", n)
								}
							}
							if field.affected == "prompt" || field.affected == "prefill" || field.affected == "cache" {
								if n.EngineDecodeTokensPerSecond == nil {
									t.Fatalf("unrelated decode rate removed: %+v", n)
								}
							} else if n.PromptTokens == nil || n.PrefillNs == nil || n.CachedPromptTokens == nil {
								t.Fatalf("unrelated prompt/cache evidence removed: %+v", n)
							}
						})
					}
				})
			}
		})
	}
}
