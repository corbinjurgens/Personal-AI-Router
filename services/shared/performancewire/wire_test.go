// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package performancewire

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

func ptr[T any](v T) *T { return &v }

func validSample() Sample {
	return Sample{
		Scope:    ScopeLocalExecution,
		Producer: "proxy-1",
		Seq:      1,
		Key: ProfileKey{
			Host:        "host-a",
			Engine:      "llamacpp-custom",
			Model:       "hf.co/org/model:Q4_K_M",
			Fingerprint: "fp-1",
		},
		Profile: Profile{EngineFamily: "llamacpp", Quantization: "Q4_K_M", ContextTokens: ptr[int64](8192)},
		Measurement: Measurement{
			Outcome:          OutcomeCompleted,
			ProtocolComplete: ptr(true),
			FirstBodyMs:      ptr[int64](120),
			PromptTokens:     ptr[int64](100),
			PrefillNs:        ptr[int64](1_000_000_000),
		},
	}
}

func validEntry() Entry {
	return Entry{
		Key:                validSample().Key,
		Samples:            3,
		Completed:          2,
		Incomplete:         1,
		Failed:             1,
		NewestAgeMs:        10,
		OldestAgeMs:        100,
		NewestObservedAtMs: 1_760_000_000_000,
		FirstBodyMs:        &Distribution{Samples: 1, Median: 50, P95: 50},
		Decode:             &Rate{Samples: 1, Tokens: 100, DurationNs: 1_000_000_000, TokensPerSecond: 100},
	}
}

func validSummary() Summary {
	return Summary{
		Version:              SchemaVersion,
		MeasuringHost:        "host-a",
		AdmissionIncarnation: "inc-1",
		Revision:             7,
		Window:               Window{MaxAgeMs: RetentionMs, MaxSamplesPerKey: MaxSamplesPerKey},
		Entries:              []Entry{validEntry()},
	}
}

func TestMeasurementKeepsZeroDistinctFromUnknown(t *testing.T) {
	m := Measurement{Outcome: OutcomeCompleted, FirstBodyMs: ptr[int64](0), OutputTokens: ptr[int64](0)}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var fields map[string]any
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatalf("unmarshal fields: %v", err)
	}
	if fields["firstBodyMs"] != 0.0 || fields["outputTokens"] != 0.0 {
		t.Fatalf("measured zeros were not encoded: %v", fields)
	}
	if _, ok := fields["totalMs"]; ok {
		t.Fatalf("unknown totalMs was encoded: %v", fields)
	}
	var back Measurement
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal measurement: %v", err)
	}
	if back.FirstBodyMs == nil || *back.FirstBodyMs != 0 || back.TotalMs != nil {
		t.Fatalf("round trip lost zero versus unknown: %+v", back)
	}
}

func TestSampleValidate(t *testing.T) {
	if err := validSample().Validate(); err != nil {
		t.Fatalf("valid sample rejected: %v", err)
	}
	reject := func(name string, mutate func(*Sample), want string) {
		t.Run(name, func(t *testing.T) {
			s := validSample()
			mutate(&s)
			err := s.Validate()
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("Validate() = %v, want error containing %q", err, want)
			}
		})
	}
	reject("unknown scope", func(s *Sample) { s.Scope = "origin" }, "scope")
	reject("missing producer", func(s *Sample) { s.Producer = "" }, "producer")
	reject("zero seq", func(s *Sample) { s.Seq = 0 }, "seq")
	reject("missing host", func(s *Sample) { s.Key.Host = "" }, "host")
	reject("engine with space", func(s *Sample) { s.Key.Engine = "llama cpp" }, "engine")
	reject("missing model", func(s *Sample) { s.Key.Model = "" }, "model")
	reject("model with control character", func(s *Sample) { s.Key.Model = "m\n" }, "model")
	reject("model too long", func(s *Sample) { s.Key.Model = strings.Repeat("m", MaxModelBytes+1) }, "model")
	reject("fingerprint too long", func(s *Sample) { s.Key.Fingerprint = strings.Repeat("f", MaxIDBytes+1) }, "fingerprint")
	reject("zero context", func(s *Sample) { s.Profile.ContextTokens = ptr[int64](0) }, "contextTokens")
	reject("unknown outcome", func(s *Sample) { s.Outcome = "partial" }, "outcome")
	reject("negative latency", func(s *Sample) { s.FirstBodyMs = ptr[int64](-1) }, "firstBodyMs")
	reject("latency over a day", func(s *Sample) { s.TotalMs = ptr(MaxLatencyMs + 1) }, "totalMs")
	reject("tokens over bound", func(s *Sample) { s.OutputTokens = ptr(MaxTokens + 1) }, "outputTokens")
	reject("phase over bound", func(s *Sample) { s.DecodeNs = ptr(MaxPhaseNs + 1) }, "decodeNs")
	reject("prefill rate over bound", func(s *Sample) { s.PrefillNs = ptr[int64](1) }, "prefill rate")
}

func TestSummaryValidate(t *testing.T) {
	if err := validSummary().Validate(); err != nil {
		t.Fatalf("valid summary rejected: %v", err)
	}
	empty := validSummary()
	empty.Entries = []Entry{}
	if err := empty.Validate(); err != nil {
		t.Fatalf("empty snapshot rejected: %v", err)
	}
	reject := func(name string, mutate func(*Summary), want string) {
		t.Run(name, func(t *testing.T) {
			s := validSummary()
			mutate(&s)
			err := s.Validate()
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("Validate() = %v, want error containing %q", err, want)
			}
		})
	}
	reject("old protocol version", func(s *Summary) { s.Version = 1 }, "version")
	reject("missing incarnation", func(s *Summary) { s.AdmissionIncarnation = "" }, "admissionIncarnation")
	reject("zero revision", func(s *Summary) { s.Revision = 0 }, "revision")
	reject("window beyond retention", func(s *Summary) { s.Window.MaxAgeMs = RetentionMs + 1 }, "maxAgeMs")
	reject("window beyond sample cap", func(s *Summary) { s.Window.MaxSamplesPerKey = MaxSamplesPerKey + 1 }, "maxSamplesPerKey")
	reject("too many entries", func(s *Summary) {
		s.Entries = nil
		for i := range MaxEntries + 1 {
			e := validEntry()
			e.Key.Fingerprint = fmt.Sprintf("fp-%d", i)
			s.Entries = append(s.Entries, e)
		}
	}, "entries exceed")
	reject("omitted beyond cap", func(s *Summary) { s.OmittedEntries = MaxEntries }, "omittedEntries")
	reject("entry for another host", func(s *Summary) { s.Entries[0].Key.Host = "host-b" }, "measuring host")
	reject("repeated key", func(s *Summary) { s.Entries = append(s.Entries, validEntry()) }, "repeats")
	reject("counts do not partition", func(s *Summary) { s.Entries[0].Failed = 2 }, "partition")
	reject("incomplete beyond completed", func(s *Summary) { s.Entries[0].Incomplete = 3 }, "incomplete")
	reject("samples beyond window", func(s *Summary) { s.Entries[0].Samples = MaxSamplesPerKey + 1 }, "samples")
	reject("ages out of order", func(s *Summary) { s.Entries[0].NewestAgeMs = 200 }, "ages")
	reject("age beyond window", func(s *Summary) { s.Entries[0].OldestAgeMs = RetentionMs + 1 }, "ages")
	reject("median above p95", func(s *Summary) { s.Entries[0].FirstBodyMs.Median = 60 }, "firstBodyMs")
	reject("distribution beyond completed", func(s *Summary) { s.Entries[0].FirstBodyMs.Samples = 3 }, "firstBodyMs")
	reject("NaN rate", func(s *Summary) { s.Entries[0].Decode.TokensPerSecond = math.NaN() }, "decode")
	reject("infinite rate", func(s *Summary) { s.Entries[0].Decode.TokensPerSecond = math.Inf(1) }, "decode")
	reject("negative rate", func(s *Summary) { s.Entries[0].Decode.TokensPerSecond = -1 }, "decode")
	reject("rate over bound", func(s *Summary) { s.Entries[0].Decode.TokensPerSecond = MaxTokensPerSecond * 2 }, "decode")
	reject("zero rate duration", func(s *Summary) { s.Entries[0].Decode.DurationNs = 0 }, "durationNs")
}

func TestEncodeDecodeSummaryRoundTrip(t *testing.T) {
	want := validSummary()
	data, err := EncodeSummary(want)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := DecodeSummary(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip = %+v, want %+v", got, want)
	}
}

func TestOutputTokensSummaryValidation(t *testing.T) {
	test := func(name string, completed, incomplete, failed int, tokens *int64, wantError bool) {
		t.Run(name, func(t *testing.T) {
			s := validSummary()
			e := &s.Entries[0]
			e.Completed, e.Incomplete, e.Failed = completed, incomplete, failed
			e.Samples = completed + failed
			e.FirstBodyMs, e.Decode = nil, nil
			e.OutputTokens = tokens
			err := s.Validate()
			if wantError {
				if err == nil || !strings.Contains(err.Error(), "outputTokens") {
					t.Fatalf("Validate() = %v, want outputTokens rejection", err)
				}
			} else if err != nil {
				t.Fatalf("Validate() = %v", err)
			}
		})
	}
	test("unknown all failed", 0, 0, 1, nil, false)
	test("measured zero", 1, 0, 0, ptr[int64](0), false)
	test("maximum qualified total", 3, 1, 1, ptr(2*MaxTokens), false)
	test("negative total", 1, 0, 0, ptr[int64](-1), true)
	test("over qualified bound", 3, 1, 1, ptr(2*MaxTokens+1), true)
	test("failed zero has no eligible sample", 0, 0, 1, ptr[int64](0), true)
	test("incomplete zero has no eligible sample", 1, 1, 0, ptr[int64](0), true)
}

func TestOutputTokensSummaryRoundTripOwnsPointers(t *testing.T) {
	test := func(name string, tokens *int64) {
		t.Run(name, func(t *testing.T) {
			want := validSummary()
			want.Entries[0].OutputTokens = tokens
			data, err := EncodeSummary(want)
			if err != nil {
				t.Fatalf("encode summary: %v", err)
			}
			var fields struct {
				Entries []map[string]json.RawMessage `json:"entries"`
			}
			if err := json.Unmarshal(data, &fields); err != nil {
				t.Fatalf("decode entry fields: %v", err)
			}
			if _, present := fields.Entries[0]["outputTokens"]; present != (tokens != nil) {
				t.Fatalf("outputTokens presence = %t, want %t", present, tokens != nil)
			}
			got, err := DecodeSummary(data)
			if err != nil {
				t.Fatalf("decode summary: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("round trip = %+v, want %+v", got, want)
			}
			if tokens != nil {
				*got.Entries[0].OutputTokens = -1
				if *want.Entries[0].OutputTokens < 0 {
					t.Fatal("decoded count aliases source summary")
				}
			}
		})
	}
	test("unknown omitted", nil)
	test("measured zero present", ptr[int64](0))
	test("maximum count", ptr(MaxTokens))
}

func TestDecodeSummaryRejects(t *testing.T) {
	valid, err := json.Marshal(validSummary())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	reject := func(name string, data []byte, want string) {
		t.Run(name, func(t *testing.T) {
			_, err := DecodeSummary(data)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("DecodeSummary() = %v, want error containing %q", err, want)
			}
		})
	}
	reject("payload over limit", append(valid, make([]byte, MaxPayloadBytes)...), "over")
	reject("unknown field", []byte(strings.Replace(string(valid), `{"version"`, `{"extra":1,"version"`, 1)), "unknown field")
	reject("trailing data", append(append([]byte{}, valid...), []byte(` {}`)...), "trailing")
	reject("rate overflowing float", []byte(strings.Replace(string(valid), `"tokensPerSecond":100`, `"tokensPerSecond":1e400`, 1)), "decode summary")
	reject("negative count", []byte(strings.Replace(string(valid), `"failed":1`, `"failed":-1`, 1)), "partition")
}

func TestEncodeSummaryRefusesInvalid(t *testing.T) {
	s := validSummary()
	s.Entries[0].Decode.TokensPerSecond = math.NaN()
	if _, err := EncodeSummary(s); err == nil {
		t.Fatal("EncodeSummary accepted a NaN rate")
	}
}

func TestReportedMeasurementValidation(t *testing.T) {
	test := func(name string, ns *int64, rate *float64, want string) {
		t.Run(name, func(t *testing.T) {
			m := Measurement{Outcome: OutcomeCompleted, EngineDecodeNs: ns, EngineDecodeTokensPerSecond: rate}
			err := m.Validate()
			if want == "" {
				if err != nil {
					t.Fatalf("Validate() = %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("Validate() = %v, want %q", err, want)
			}
		})
	}
	test("unknown", nil, nil, "")
	test("duration without rate", ptr[int64](400_000_000), nil, "")
	test("rate without duration", nil, ptr(55.0), "")
	test("measured zero", ptr[int64](400_000_000), ptr(0.0), "")
	test("zero interval is retained but cannot qualify", ptr[int64](0), ptr(55.0), "")
	test("maximum", ptr(MaxPhaseNs), ptr(MaxTokensPerSecond), "")
	test("negative interval", ptr[int64](-1), nil, "engineDecodeNs")
	test("interval over bound", ptr(MaxPhaseNs+1), nil, "engineDecodeNs")
	test("negative rate", nil, ptr(-1.0), "engineDecodeTokensPerSecond")
	test("rate over bound", nil, ptr(MaxTokensPerSecond+1), "engineDecodeTokensPerSecond")
	test("NaN rate", nil, ptr(math.NaN()), "engineDecodeTokensPerSecond")
	test("infinite rate", nil, ptr(math.Inf(1)), "engineDecodeTokensPerSecond")
}

func TestReportedMeasurementRoundTripKeepsZeroAndUnknown(t *testing.T) {
	want := Measurement{Outcome: OutcomeCompleted, EngineDecodeNs: ptr[int64](400_000_000), EngineDecodeTokensPerSecond: ptr(0.0)}
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got Measurement
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !reflect.DeepEqual(got, want) || got.DecodeNs != nil || got.OutputTokens != nil {
		t.Fatalf("round trip = %+v, want %+v without invented legacy measurements", got, want)
	}
	want.EngineDecodeTokensPerSecond = nil
	data, err = json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal unknown: %v", err)
	}
	got = Measurement{}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal unknown: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unknown round trip = %+v, want %+v", got, want)
	}
}

func TestReportedRateSummaryValidation(t *testing.T) {
	test := func(name string, rate ReportedRate, want string) {
		t.Run(name, func(t *testing.T) {
			s := validSummary()
			s.Entries[0].EngineDecode = &rate
			err := s.Validate()
			if want == "" {
				if err != nil {
					t.Fatalf("Validate() = %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("Validate() = %v, want %q", err, want)
			}
		})
	}
	test("reported rate independent of legacy count", ReportedRate{Samples: 1, DurationNs: 400_000_000, TokensPerSecond: 55}, "")
	test("measured zero", ReportedRate{Samples: 1, DurationNs: 400_000_000}, "")
	test("maximum", ReportedRate{Samples: 1, DurationNs: MaxPhaseNs, TokensPerSecond: MaxTokensPerSecond}, "")
	test("missing samples", ReportedRate{DurationNs: 1}, "samples")
	test("includes incomplete sample", ReportedRate{Samples: 2, DurationNs: 1}, "samples")
	test("zero duration", ReportedRate{Samples: 1}, "durationNs")
	test("duration over bound", ReportedRate{Samples: 1, DurationNs: MaxPhaseNs + 1}, "durationNs")
	test("negative rate", ReportedRate{Samples: 1, DurationNs: 1, TokensPerSecond: -1}, "tokensPerSecond")
	test("rate over bound", ReportedRate{Samples: 1, DurationNs: 1, TokensPerSecond: MaxTokensPerSecond + 1}, "tokensPerSecond")
	test("NaN", ReportedRate{Samples: 1, DurationNs: 1, TokensPerSecond: math.NaN()}, "tokensPerSecond")
	test("infinity", ReportedRate{Samples: 1, DurationNs: 1, TokensPerSecond: math.Inf(1)}, "tokensPerSecond")
}

func TestReportedRateV2RoundTripAndV1Rejection(t *testing.T) {
	want := validSummary()
	want.Entries[0].EngineDecode = &ReportedRate{Samples: 1, DurationNs: 400_000_000, TokensPerSecond: 55}
	data, err := EncodeSummary(want)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := DecodeSummary(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip = %+v, want %+v", got, want)
	}
	want.Version = 1
	want.Entries[0].EngineDecode = nil
	old, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal v1: %v", err)
	}
	if _, err := DecodeSummary(old); err == nil || !strings.Contains(err.Error(), "version") {
		t.Fatalf("DecodeSummary(v1) = %v, want version rejection", err)
	}
}

func TestReportedRateDecodeRequiresMeasuredRate(t *testing.T) {
	test := func(name, object string) {
		t.Run(name, func(t *testing.T) {
			var rate ReportedRate
			if err := json.Unmarshal([]byte(object), &rate); err == nil {
				t.Fatalf("accepted incomplete or unknown reported-rate fields: %+v", rate)
			}
		})
	}
	test("missing rate", `{"samples":1,"durationNs":400000000}`)
	test("null rate", `{"samples":1,"durationNs":400000000,"tokensPerSecond":null}`)
	test("missing interval", `{"samples":1,"tokensPerSecond":0}`)
	test("missing coverage", `{"durationNs":400000000,"tokensPerSecond":0}`)
	test("invented token total", `{"samples":1,"durationNs":400000000,"tokensPerSecond":0,"tokens":0}`)
	var rate ReportedRate
	if err := json.Unmarshal([]byte(`{"samples":1,"durationNs":400000000,"tokensPerSecond":0}`), &rate); err != nil {
		t.Fatalf("measured zero rejected: %v", err)
	}
	if rate != (ReportedRate{Samples: 1, DurationNs: 400_000_000}) {
		t.Fatalf("measured zero = %+v", rate)
	}
}
