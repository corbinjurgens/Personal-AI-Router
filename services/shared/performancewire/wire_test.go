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
	reject("unsupported version", func(s *Summary) { s.Version = 2 }, "version")
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
