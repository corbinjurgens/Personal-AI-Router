// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/performancewire"
)

func performanceExtractTestInt(value int64) *int64 { return &value }

func performanceExtractTestFloat(value float64) *float64 { return &value }

func performanceExtractTestFixture(t *testing.T, name string) []byte {
	t.Helper()
	payload, err := os.ReadFile("testdata/performance/" + name)
	if err != nil {
		t.Fatalf("read performance fixture: %v", err)
	}
	return payload
}

func performanceExtractTestEqual(t *testing.T, got, want performanceExtractMetadata) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		// Marshal only typed numeric metadata, never the source payload.
		gotJSON, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("encode extracted metadata: %v", err)
		}
		wantJSON, err := json.Marshal(want)
		if err != nil {
			t.Fatalf("encode expected metadata: %v", err)
		}
		t.Fatalf("metadata = %s, want %s", gotJSON, wantJSON)
	}
}

func TestPerformanceExtractDailyFixtures(t *testing.T) {
	llama := performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, performanceExtractLlamaSchema}
	strata := performanceExtractQualification{"strata", "/v1/chat/completions", true, performanceExtractStrataSchema}
	llamaWant := performanceExtractMetadata{
		PromptTokens: performanceExtractTestInt(10), CachedPromptTokens: performanceExtractTestInt(5),
		OutputTokens: performanceExtractTestInt(11), PrefillNs: performanceExtractTestInt(125000),
		EngineDecodeNs:              performanceExtractTestInt(250500000),
		EngineDecodeTokensPerSecond: performanceExtractTestFloat(39.92015968063872),
	}
	strataWant := performanceExtractMetadata{
		PromptTokens: performanceExtractTestInt(10), CachedPromptTokens: performanceExtractTestInt(5),
		OutputTokens: performanceExtractTestInt(20), PrefillNs: performanceExtractTestInt(12300000),
		EngineDecodeNs: performanceExtractTestInt(400000000), EngineDecodeTokensPerSecond: performanceExtractTestFloat(50),
	}
	test := func(name, file string, q performanceExtractQualification, want performanceExtractMetadata) {
		t.Run(name, func(t *testing.T) {
			payload := performanceExtractTestFixture(t, file)
			if strings.HasSuffix(file, ".txt") {
				// Fixed placement evidence, not a framing or completion implementation.
				lines := bytes.Split(bytes.TrimSpace(payload), []byte("\n"))
				if len(lines) != 5 || string(lines[4]) != "data: [DONE]" || !bytes.HasPrefix(lines[2], []byte("data: ")) {
					t.Fatal("expected final numeric payload before SSE terminal")
				}
				payload = bytes.TrimPrefix(lines[2], []byte("data: "))
			}
			original := bytes.Clone(payload)
			performanceExtractTestEqual(t, extractPerformancePayload(payload, q), want)
			performanceExtractTestEqual(t, extractPerformancePayload(payload, q), want)
			if !bytes.Equal(payload, original) {
				t.Fatal("daily extractor mutated input")
			}
		})
	}
	test("llama empty choices usage", "llamacpp-chat-stream-usage.txt", llama, llamaWant)
	test("llama tool finish without usage", "llamacpp-chat-stream-no-usage.txt", llama, llamaWant)
	llama.Streaming = false
	test("llama nonstream", "llamacpp-chat-complete.json", llama, llamaWant)
	test("Strata tool and reasoning stream", "strata-chat-stream.txt", strata, strataWant)
	strata.Streaming = false
	test("Strata nonstream", "strata-chat-complete.json", strata, strataWant)
	test("Strata native work differs from visible output", "strata-native-count-diff.json", strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20), EngineDecodeNs: performanceExtractTestInt(400000000),
			EngineDecodeTokensPerSecond: performanceExtractTestFloat(55)})
}

func TestPerformanceExtractLlamaStreamOutputCount(t *testing.T) {
	test := func(name string, stream []byte, wantInterval *int64, wantRate *float64) {
		t.Run(name, func(t *testing.T) {
			p := performanceProxy(t, "llamacpp")
			attempt, err := p.soleFacade().newPerformanceAttemptOwned(context.Background(), "m", routeChatCompletions,
				[]byte(`{"model":"m","stream":true}`), 0)
			if err != nil {
				t.Fatalf("create qualified attempt: %v", err)
			}
			if attempt == nil {
				t.Fatal("qualified llama.cpp attempt was not created")
			}
			attempt.start = time.Now()
			reader, err := newStreamFrameReader(streamFrameConfig{
				Budget:   newFrameBudget(64 << 20),
				Protocol: streamProtocolSSE,
				Upstream: io.NopCloser(bytes.NewReader(stream)),
				Extract: func(payload []byte) performanceExtractMetadata {
					return extractPerformanceObserved(payload, attempt.qualification)
				},
			})
			if err != nil {
				t.Fatalf("create SC reader: %v", err)
			}
			t.Cleanup(func() {
				if err := reader.Close(); err != nil {
					t.Errorf("close SC reader: %v", err)
				}
			})
			forwarded, err := io.ReadAll(reader)
			if err != nil {
				t.Fatalf("read SSE stream: %v", err)
			}
			if !bytes.Equal(forwarded, stream) {
				t.Fatal("SC reader changed complete upstream stream")
			}
			if result := reader.Result(); result.Outcome != streamOutcomeCompleted {
				t.Fatalf("SC outcome = %v, want completed", result.Outcome)
			}
			attempt.reader = reader
			attempt.finish(&statusCapture{}, true)
			sample := performanceOne(t, p)
			if sample.Outcome != performancewire.OutcomeCompleted || sample.ProtocolComplete == nil || !*sample.ProtocolComplete {
				t.Fatal("successful SC completion did not qualify the sample")
			}
			knownOutputTokens, legacyDecodeNs := sample.OutputTokens, sample.DecodeNs
			if knownOutputTokens == nil || *knownOutputTokens != 11 {
				t.Fatalf("output tokens = %v, want known count 11", knownOutputTokens)
			}
			if legacyDecodeNs != nil {
				t.Fatal("llama.cpp native interval populated legacy DecodeNs")
			}
			if !reflect.DeepEqual(sample.EngineDecodeNs, wantInterval) || !reflect.DeepEqual(sample.EngineDecodeTokensPerSecond, wantRate) {
				t.Fatal("native interval/rate did not remain independent of output count")
			}
			if wantRate != nil && *wantRate == performancewire.TokensPerSecond(*knownOutputTokens, *wantInterval) {
				t.Fatal("fixture must distinguish native decode steps from visible output tokens")
			}
		})
	}
	// The final empty-choices event carries matching usage and predicted_n.
	// llama.cpp reports ten decode steps for eleven visible output tokens.
	// The extractor fixture ends with one newline after [DONE]; the wire stream
	// needs a blank line to dispatch that terminal SSE event.
	nativeStream := append(performanceExtractTestFixture(t, "llamacpp-chat-stream-usage.txt"), '\n')
	test("native timings with usage", nativeStream,
		performanceExtractTestInt(250500000), performanceExtractTestFloat(39.92015968063872))
	test("usage without timings retains count and unknown rates", []byte(
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"fixture\"},\"finish_reason\":null}]}\n\n"+
			"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":11,\"total_tokens\":26}}\n\n"+
			"data: [DONE]\n\n"), nil, nil)
}

func TestPerformanceExtractFractionalMilliseconds(t *testing.T) {
	q := performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, performanceExtractLlamaSchema}
	test := func(name, number string, want *int64) {
		t.Run(name, func(t *testing.T) {
			payload := []byte(`{"timings":{"predicted_ms":` + number + `}}`)
			performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{EngineDecodeNs: want})
		})
	}
	test("exact fraction", "0.125", performanceExtractTestInt(125000))
	test("nearest nanosecond", "12.3456789", performanceExtractTestInt(12345679))
	test("exact half rounds upward", "0.0000005", performanceExtractTestInt(1))
	test("below half rounds down", "0.000000499999999999999", performanceExtractTestInt(0))
	test("decimal exponent", "1e-3", performanceExtractTestInt(1000))
	test("uppercase positive exponent", "1.25E+2", performanceExtractTestInt(125000000))
	test("explicit zero", "0", performanceExtractTestInt(0))
	test("small bounded exponent", "1e-308", performanceExtractTestInt(0))
	test("large bounded exponent", "1e308", nil)
	test("exponent over bound", "0e309", nil)
	test("exponent under bound", "0e-309", nil)
	test("huge exponent", "1e999999999999999999999999999999", nil)
	test("maximum phase", "86400000", performanceExtractTestInt(performancewire.MaxPhaseNs))
	test("above maximum before rounding", "86400000.0000001", nil)
	test("lexeme exactly bounded", "0."+strings.Repeat("0", 61)+"1", performanceExtractTestInt(0))
	test("lexeme over bound", "0."+strings.Repeat("0", 62)+"1", nil)
	test("negative", "-0.125", nil)
	test("negative zero", "-0", nil)
	test("string", `"0.125"`, nil)
	test("null", "null", nil)
	test("boolean", "true", nil)
	test("object", "{}", nil)
	test("nonfinite syntax", "NaN", nil)
	test("infinite syntax", "Infinity", nil)
}

func TestPerformanceExtractDailyInvalidFixtures(t *testing.T) {
	var cases []struct {
		Name    string
		Payload json.RawMessage
		Want    performanceExtractMetadata
	}
	if err := json.Unmarshal(performanceExtractTestFixture(t, "daily-invalid.json"), &cases); err != nil {
		t.Fatalf("decode invalid daily cases: %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("missing invalid daily cases")
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload(tc.Payload,
				performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, performanceExtractLlamaSchema}), tc.Want)
		})
	}
}

func TestPerformanceExtractDailyQualification(t *testing.T) {
	payload := performanceExtractTestFixture(t, "llamacpp-chat-complete.json")
	test := func(name string, q performanceExtractQualification) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{})
		})
	}
	test("llama needs schema", performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, ""})
	test("Strata needs schema", performanceExtractQualification{"strata", "/v1/chat/completions", true, ""})
	test("unknown schema", performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, "unreviewed"})
	test("schema family mismatch", performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, performanceExtractStrataSchema})
	test("schema API mismatch", performanceExtractQualification{"strata", "/v1/completions", true, performanceExtractStrataSchema})
	test("unknown family", performanceExtractQualification{"other", "/v1/chat/completions", true, performanceExtractLlamaSchema})
}

func TestPerformanceExtractDailyPhaseConsistency(t *testing.T) {
	llama := performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, performanceExtractLlamaSchema}
	strata := performanceExtractQualification{"strata", "/v1/chat/completions", true, performanceExtractStrataSchema}
	test := func(name, payload string, q performanceExtractQualification, want performanceExtractMetadata) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload([]byte(payload), q), want)
		})
	}
	llamaZero := performanceExtractMetadata{OutputTokens: performanceExtractTestInt(1), EngineDecodeNs: performanceExtractTestInt(125000),
		EngineDecodeTokensPerSecond: performanceExtractTestFloat(0)}
	test("one llama output has zero decode steps", `{"timings":{"predicted_n":1,"predicted_ms":0.125,"predicted_per_second":0,"predicted_per_token_ms":0}}`, llama, llamaZero)
	test("null per token when no work", `{"timings":{"predicted_n":1,"predicted_ms":0.125,"predicted_per_second":0,"predicted_per_token_ms":null}}`, llama, llamaZero)
	test("positive rate rounds to zero duration", `{"timings":{"predicted_ms":0.0000001,"predicted_per_second":1}}`, strata, performanceExtractMetadata{})
	test("missing native rate stays unknown", `{"timings":{"predicted_n":20,"predicted_ms":400}}`, strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20), EngineDecodeNs: performanceExtractTestInt(400000000)})
	test("missing prompt auxiliary keeps independent pair", `{"timings":{"prompt_n":10,"prompt_ms":0.125}}`, llama,
		performanceExtractMetadata{PromptTokens: performanceExtractTestInt(10), PrefillNs: performanceExtractTestInt(125000)})
	test("bad prompt rate keeps output", `{"timings":{"prompt_n":10,"prompt_ms":0.125,"prompt_per_second":1,"predicted_n":11}}`, llama,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(11)})
	test("llama per token contradiction", `{"timings":{"predicted_n":11,"predicted_ms":250.5,"predicted_per_second":39.92015968063872,"predicted_per_token_ms":1}}`, llama,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(11)})
	test("Strata reciprocal rounding overlaps", `{"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":55,"predicted_per_token_ms":18.182}}`, strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20), EngineDecodeNs: performanceExtractTestInt(400000000), EngineDecodeTokensPerSecond: performanceExtractTestFloat(55)})
	test("Strata reciprocal contradiction", `{"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":55,"predicted_per_token_ms":20}}`, strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20)})
	test("Strata prompt rounding interval", `{"timings":{"prompt_n":1,"prompt_ms":0.1,"prompt_per_second":8000}}`, strata,
		performanceExtractMetadata{PromptTokens: performanceExtractTestInt(1), PrefillNs: performanceExtractTestInt(100000)})
	test("Strata prompt outside rounding interval", `{"timings":{"prompt_n":1,"prompt_ms":1,"prompt_per_second":8000}}`, strata, performanceExtractMetadata{})
	test("Strata cache mismatch", `{"timings":{"cache_n":5,"prompt_n":10,"prompt_ms":1},"usage":{"prompt_tokens_details":{"cached_tokens":6}}}`, strata,
		performanceExtractMetadata{PromptTokens: performanceExtractTestInt(10), PrefillNs: performanceExtractTestInt(1000000)})
	test("duplicate decode rate", `{"timings":{"predicted_n":11,"predicted_ms":250.5,"predicted_per_second":39.92015968063872,"predicted_per_second":39.92015968063872}}`, llama,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(11)})
	test("escaped duplicate duration", `{"timings":{"predicted_n":11,"predicted_ms":250.5,"predicted_\u006ds":250.5,"predicted_per_second":39.92015968063872}}`, llama,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(11)})
	test("rate above shared limit", `{"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":10000001}}`, strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20)})
	test("negative rate", `{"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":-1}}`, strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20)})
	test("no work explicit Strata zero", `{"timings":{"predicted_n":0,"predicted_ms":1,"predicted_per_second":0,"predicted_per_token_ms":null}}`, strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(0), EngineDecodeNs: performanceExtractTestInt(1000000), EngineDecodeTokensPerSecond: performanceExtractTestFloat(0)})
	test("llama half nanosecond uncertainty", `{"timings":{"predicted_n":2,"predicted_ms":0.0001004,"predicted_per_second":9960159.3625498}}`, llama,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(2), EngineDecodeNs: performanceExtractTestInt(100), EngineDecodeTokensPerSecond: performanceExtractTestFloat(9960159.3625498)})
	test("llama relative tolerance", `{"timings":{"predicted_n":11,"predicted_ms":250.5,"predicted_per_second":39.92018}}`, llama,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(11), EngineDecodeNs: performanceExtractTestInt(250500000), EngineDecodeTokensPerSecond: performanceExtractTestFloat(39.92018)})
	test("native rate exactly at limit", `{"timings":{"predicted_ms":1,"predicted_per_second":10000000}}`, strata,
		performanceExtractMetadata{EngineDecodeNs: performanceExtractTestInt(1000000), EngineDecodeTokensPerSecond: performanceExtractTestFloat(10000000)})
	test("null native rate is unknown", `{"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":null}}`, strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20), EngineDecodeNs: performanceExtractTestInt(400000000)})
	test("Strata no work null auxiliaries", `{"timings":{"prompt_n":0,"prompt_ms":0,"prompt_per_second":null,"prompt_per_token_ms":null,"predicted_n":0,"predicted_ms":1,"predicted_per_second":null,"predicted_per_token_ms":null}}`, strata,
		performanceExtractMetadata{PromptTokens: performanceExtractTestInt(0), PrefillNs: performanceExtractTestInt(0), OutputTokens: performanceExtractTestInt(0), EngineDecodeNs: performanceExtractTestInt(1000000)})
	test("null interval keeps output", `{"timings":{"predicted_n":20,"predicted_ms":null,"predicted_per_second":55}}`, strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20)})
	test("null per token with work disqualifies phase", `{"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":55,"predicted_per_token_ms":null}}`, strata,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20)})
	test("Strata native rate survives output disagreement", `{"usage":{"completion_tokens":21},"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":55}}`, strata,
		performanceExtractMetadata{EngineDecodeNs: performanceExtractTestInt(400000000), EngineDecodeTokensPerSecond: performanceExtractTestFloat(55)})
	test("duplicate cache details preserves prompt", `{"usage":{"prompt_tokens_details":{"cached_tokens":5},"prompt_tokens_details":{"cached_tokens":5}},"timings":{"cache_n":5,"prompt_n":10,"prompt_ms":1}}`, strata,
		performanceExtractMetadata{PromptTokens: performanceExtractTestInt(10), PrefillNs: performanceExtractTestInt(1000000)})
	test("Strata prompt per token without rate matches rounding", `{"timings":{"prompt_n":10,"prompt_ms":12.3,"prompt_per_token_ms":1.234}}`, strata,
		performanceExtractMetadata{PromptTokens: performanceExtractTestInt(10), PrefillNs: performanceExtractTestInt(12300000)})
	test("Strata prompt per token without rate contradicts duration", `{"timings":{"prompt_n":10,"prompt_ms":12.3,"prompt_per_token_ms":2}}`, strata, performanceExtractMetadata{})
	test("Strata per token rounding can reach zero", `{"timings":{"predicted_ms":1,"predicted_per_second":3000000,"predicted_per_token_ms":0}}`, strata,
		performanceExtractMetadata{EngineDecodeNs: performanceExtractTestInt(1000000), EngineDecodeTokensPerSecond: performanceExtractTestFloat(3000000)})
}

func TestPerformanceExtractDailyMalformedPayloads(t *testing.T) {
	q := performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, performanceExtractLlamaSchema}
	test := func(name, payload string) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload([]byte(payload), q), performanceExtractMetadata{})
		})
	}
	test("truncated object", `{"timings":{"predicted_n":11},`)
	test("trailing document", `{"timings":{"predicted_n":11}}{}`)
	test("null document", `null`)
	test("array document", `[]`)
	test("nonfinite syntax", `{"timings":{"predicted_n":NaN}}`)
}

func TestPerformanceExtractDailyIndependentPayloads(t *testing.T) {
	q := performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, performanceExtractLlamaSchema}
	payload := performanceExtractTestFixture(t, "llamacpp-chat-complete.json")
	original := bytes.Clone(payload)
	first := extractPerformancePayload(payload, q)
	second := extractPerformancePayload(payload, q)
	if first.EngineDecodeTokensPerSecond == nil || second.EngineDecodeTokensPerSecond == nil {
		t.Fatal("expected reported rates")
	}
	*first.EngineDecodeTokensPerSecond = 0
	clear(payload)
	performanceExtractTestEqual(t, second, extractPerformancePayload(original, q))
}

func TestPerformanceExtractDailyBoundedPayload(t *testing.T) {
	q := performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, performanceExtractLlamaSchema}
	prefix, suffix := `{"timings":{"predicted_ms":0.125},"padding":"`, `"}`
	payload := []byte(prefix + strings.Repeat(" ", int(streamFrameMaxBytes)-len(prefix)-len(suffix)) + suffix)
	performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{EngineDecodeNs: performanceExtractTestInt(125000)})
	payload = append(payload, ' ')
	performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{})
}

func TestPerformanceExtractStrataMCPDisqualification(t *testing.T) {
	q := performanceExtractQualification{"strata", "/v1/chat/completions", true, performanceExtractStrataSchema}
	for _, value := range []string{`{}`, `null`, `false`, `[]`, `0`} {
		t.Run("present "+value, func(t *testing.T) {
			payload := []byte(`{"strata_mcp":` + value + `,"usage":{"completion_tokens":20},"timings":{"predicted_n":20,"predicted_ms":400,"predicted_per_second":55}}`)
			performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{
				OutputTokens: performanceExtractTestInt(20), PhaseDisqualified: true})
		})
	}
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"strata_mcp":{},"strata_\u006dcp":null}`), q),
		performanceExtractMetadata{PhaseDisqualified: true})
	q.Streaming = false
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"choices":[{"message":{"strata_mcp":[]}}],"timings":{"predicted_ms":400,"predicted_per_second":55}}`), q),
		performanceExtractMetadata{PhaseDisqualified: true})
}
