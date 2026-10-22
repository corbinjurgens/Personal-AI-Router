// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

func TestPerformanceExtractNativeFixtures(t *testing.T) {
	want := performanceExtractMetadata{
		OutputTokens: performanceExtractTestInt(18),
		DecodeNs:     performanceExtractTestInt(52479709),
		LoadNs:       performanceExtractTestInt(101397084),
		PrefillNs:    performanceExtractTestInt(13074791),
	}
	t.Run("chat terminal", func(t *testing.T) {
		payload := performanceExtractTestFixture(t, "ollama-complete.json")
		performanceExtractTestEqual(t, extractPerformancePayload(payload,
			performanceExtractQualification{"ollama", "/api/chat", true, ""}), want)
	})
	t.Run("generate terminal", func(t *testing.T) {
		lines := bytes.Split(bytes.TrimSpace(performanceExtractTestFixture(t, "ollama-stream.txt")), []byte("\n"))
		if len(lines) != 2 || !json.Valid(lines[0]) || !json.Valid(lines[1]) {
			t.Fatal("expected partial and terminal JSON fixture payloads")
		}
		performanceExtractTestEqual(t, extractPerformancePayload(lines[1],
			performanceExtractQualification{"ollama", "/api/generate", true, ""}), want)
	})
}

func TestPerformanceExtractCompatibilityUsage(t *testing.T) {
	want := performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20)}
	test := func(name, path string, streaming bool, payload []byte) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload(payload,
				performanceExtractQualification{"ollama", path, streaming, ""}), want)
		})
	}
	complete := performanceExtractTestFixture(t, "ollama-openai-complete.json")
	lines := bytes.Split(performanceExtractTestFixture(t, "ollama-openai-stream.txt"), []byte("\n"))
	if len(lines) < 7 || !bytes.HasPrefix(lines[4], []byte("data: ")) || string(lines[6]) != "data: [DONE]" {
		t.Fatal("expected usage payload before documented SSE terminal")
	}
	usage := bytes.TrimPrefix(lines[4], []byte("data: "))
	if !json.Valid(usage) {
		t.Fatal("usage fixture must be a JSON payload")
	}
	test("chat complete usage", "/v1/chat/completions", false, complete)
	test("chat stream usage", "/v1/chat/completions", true, usage)
	test("completion usage", "/v1/completions", false, []byte(`{"usage":{"completion_tokens":20}}`))
	test("completion stream usage", "/v1/completions", true, []byte(`{"usage":{"completion_tokens":20}}`))
	test("unqualified timing extensions", "/v1/chat/completions", true,
		[]byte(`{"usage":{"completion_tokens":20,"eval_duration":1000000},"eval_duration":1000000,"timings":{"predicted_ms":1}}`))
}

func TestPerformanceExtractExplicitZero(t *testing.T) {
	q := performanceExtractQualification{"ollama", "/api/chat", true, ""}
	performanceExtractTestEqual(t, extractPerformancePayload(
		[]byte(`{"eval_count":0,"eval_duration":0,"load_duration":0,"prompt_eval_duration":0}`), q),
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(0), DecodeNs: performanceExtractTestInt(0),
			LoadNs: performanceExtractTestInt(0), PrefillNs: performanceExtractTestInt(0)})
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{}`), q), performanceExtractMetadata{})
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"usage":{"completion_tokens":0}}`),
		performanceExtractQualification{"ollama", "/v1/chat/completions", true, ""}),
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(0)})
}

func TestPerformanceExtractUnknownPromptCacheSemantics(t *testing.T) {
	payload := []byte(`{"prompt_eval_count":11,"prompt_eval_cached_count":7,"prompt_eval_duration":13074791,
		"usage":{"prompt_tokens":11,"prompt_tokens_details":{"cached_tokens":7}}}`)
	performanceExtractTestEqual(t, extractPerformancePayload(payload,
		performanceExtractQualification{"ollama", "/api/chat", true, ""}),
		performanceExtractMetadata{PrefillNs: performanceExtractTestInt(13074791)})
	performanceExtractTestEqual(t, extractPerformancePayload(payload,
		performanceExtractQualification{"ollama", "/v1/chat/completions", true, ""}), performanceExtractMetadata{})
}

func TestPerformanceExtractInvalidNumbersPreserveUnrelatedMetadata(t *testing.T) {
	test := func(name, field, value string) {
		t.Run(name, func(t *testing.T) {
			payload := []byte(fmt.Sprintf(`{"%s":%s,"load_duration":7}`, field, value))
			performanceExtractTestEqual(t, extractPerformancePayload(payload,
				performanceExtractQualification{"ollama", "/api/chat", true, ""}),
				performanceExtractMetadata{LoadNs: performanceExtractTestInt(7)})
		})
	}
	test("negative count", "eval_count", "-1")
	test("fractional count", "eval_count", "1.5")
	test("decimal count", "eval_count", "18.0")
	test("exponent count", "eval_count", "1e1")
	test("negative zero count", "eval_count", "-0")
	test("count above contract", "eval_count", fmt.Sprint(performancewire.MaxTokens+1))
	test("integer overflow count", "eval_count", "9223372036854775808")
	test("unsigned overflow count", "eval_count", "18446744073709551616")
	test("string count", "eval_count", `"18"`)
	test("null count", "eval_count", "null")
	test("object count", "eval_count", `{}`)
	test("array count", "eval_count", `[]`)
	test("boolean count", "eval_count", "true")
	test("negative decode", "eval_duration", "-1")
	test("fractional decode", "eval_duration", "1.5")
	test("decode above contract", "eval_duration", fmt.Sprint(performancewire.MaxPhaseNs+1))
	test("integer overflow decode", "eval_duration", "9223372036854775808")
	test("invalid prefill", "prompt_eval_duration", "-1")
	test("prefill above contract", "prompt_eval_duration", fmt.Sprint(performancewire.MaxPhaseNs+1))
	t.Run("invalid load preserves decode", func(t *testing.T) {
		performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"eval_count":18,"eval_duration":52479709,"load_duration":-1}`),
			performanceExtractQualification{"ollama", "/api/chat", true, ""}),
			performanceExtractMetadata{OutputTokens: performanceExtractTestInt(18), DecodeNs: performanceExtractTestInt(52479709)})
	})
}

func TestPerformanceExtractDuplicateFields(t *testing.T) {
	test := func(name, payload string, want performanceExtractMetadata) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload([]byte(payload),
				performanceExtractQualification{"ollama", "/api/chat", true, ""}), want)
		})
	}
	test("count duplicate", `{"eval_count":18,"eval_count":19,"load_duration":7}`, performanceExtractMetadata{LoadNs: performanceExtractTestInt(7)})
	test("identical duplicate remains ambiguous", `{"eval_count":18,"eval_count":18}`, performanceExtractMetadata{})
	test("escaped duplicate", `{"eval_count":18,"eval_\u0063ount":18}`, performanceExtractMetadata{})
	test("third duplicate cannot restore field", `{"eval_count":18,"eval_count":19,"eval_count":20}`, performanceExtractMetadata{})
	test("duration duplicate preserves count", `{"eval_count":18,"eval_duration":52479709,"eval_duration":1}`,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(18)})
	test("load and prefill duplicates", `{"load_duration":1,"load_duration":2,"prompt_eval_duration":1,"prompt_eval_duration":2,"eval_count":18}`,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(18)})
	test("nested counters do not shadow top level", `{"message":{"eval_count":900},"eval_count":18}`,
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(18)})
}

func TestPerformanceExtractCompatibilityInvalidUsage(t *testing.T) {
	test := func(name, payload string) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload([]byte(payload),
				performanceExtractQualification{"ollama", "/v1/chat/completions", true, ""}), performanceExtractMetadata{})
		})
	}
	test("duplicate usage", `{"usage":{"completion_tokens":20},"usage":{"completion_tokens":21}}`)
	test("escaped duplicate usage", `{"usage":{"completion_tokens":20},"us\u0061ge":{"completion_tokens":20}}`)
	test("duplicate completion count", `{"usage":{"completion_tokens":20,"completion_tokens":21}}`)
	test("negative count", `{"usage":{"completion_tokens":-1}}`)
	test("fractional count", `{"usage":{"completion_tokens":1.5}}`)
	test("overflow count", `{"usage":{"completion_tokens":9223372036854775808}}`)
	test("out of contract count", fmt.Sprintf(`{"usage":{"completion_tokens":%d}}`, performancewire.MaxTokens+1))
	test("wrong usage type", `{"usage":[{"completion_tokens":20}]}`)
	test("null count", `{"usage":{"completion_tokens":null}}`)
	test("missing count", `{"usage":{"prompt_tokens":10}}`)
	t.Run("unrelated malformed usage fields", func(t *testing.T) {
		performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"usage":{"prompt_tokens":-1,"total_tokens":null,"completion_tokens":20}}`),
			performanceExtractQualification{"ollama", "/v1/chat/completions", true, ""}),
			performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20)})
	})
}

func TestPerformanceExtractImpossibleDecodePairs(t *testing.T) {
	test := func(name string, tokens, ns int64, wantDecode *int64) {
		t.Run(name, func(t *testing.T) {
			payload := []byte(fmt.Sprintf(`{"eval_count":%d,"eval_duration":%d,"load_duration":7}`, tokens, ns))
			performanceExtractTestEqual(t, extractPerformancePayload(payload,
				performanceExtractQualification{"ollama", "/api/generate", true, ""}),
				performanceExtractMetadata{OutputTokens: performanceExtractTestInt(tokens), DecodeNs: wantDecode, LoadNs: performanceExtractTestInt(7)})
		})
	}
	test("positive output in zero time", 1, 0, nil)
	test("above shared maximum rate", 1, 99, nil)
	test("at shared maximum rate", 1, 100, performanceExtractTestInt(100))
	test("zero output in positive time", 0, 1, performanceExtractTestInt(1))
	test("contract count and duration maxima", performancewire.MaxTokens, performancewire.MaxPhaseNs, performanceExtractTestInt(performancewire.MaxPhaseNs))
}

func TestPerformanceExtractTotalDurationIgnored(t *testing.T) {
	q := performanceExtractQualification{"ollama", "/api/chat", true, ""}
	payload := performanceExtractTestFixture(t, "ollama-complete.json")
	want := extractPerformancePayload(payload, q)
	for _, value := range []string{"1", "0", "9999999999999999999999999", "null"} {
		t.Run("total "+value, func(t *testing.T) {
			changed := bytes.Replace(payload, []byte(`"total_duration": 174560334`), []byte(`"total_duration": `+value), 1)
			if bytes.Equal(changed, payload) {
				t.Fatal("fixture total_duration was not replaced")
			}
			performanceExtractTestEqual(t, extractPerformancePayload(changed, q), want)
		})
	}
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"eval_count":18,"total_duration":52479709}`), q),
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(18)})
}

func TestPerformanceExtractMalformedOrMissingPayload(t *testing.T) {
	q := performanceExtractQualification{"ollama", "/api/chat", true, ""}
	test := func(name string, payload []byte) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{})
		})
	}
	test("malformed fixture", performanceExtractTestFixture(t, "malformed.txt"))
	test("missing usage fixture", performanceExtractTestFixture(t, "missing-usage.json"))
	test("empty", nil)
	test("non object", []byte(`[18]`))
	test("null object", []byte(`null`))
	test("truncated after valid count", []byte(`{"eval_count":18,`))
	test("trailing object", []byte(`{"eval_count":18}{}`))
	test("invalid number", []byte(`{"eval_count":NaN}`))
	test("raw SSE terminal", []byte(`[DONE]`))
	t.Run("compatibility missing usage fixture", func(t *testing.T) {
		performanceExtractTestEqual(t, extractPerformancePayload(performanceExtractTestFixture(t, "missing-usage.json"),
			performanceExtractQualification{"ollama", "/v1/chat/completions", true, ""}), performanceExtractMetadata{})
	})
}

func TestPerformanceExtractQualification(t *testing.T) {
	payload := []byte(`{"eval_count":18,"eval_duration":52479709,"load_duration":7,"usage":{"completion_tokens":20}}`)
	test := func(name string, q performanceExtractQualification) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{})
		})
	}
	test("unknown engine", performanceExtractQualification{"other", "/api/chat", true, ""})
	test("other engine compatibility", performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, ""})
	test("missing engine", performanceExtractQualification{"", "/api/chat", true, ""})
	test("unknown API", performanceExtractQualification{"ollama", "/api/embed", true, ""})
	test("missing API", performanceExtractQualification{"ollama", "", true, ""})
	test("nonstreaming native chat", performanceExtractQualification{"ollama", "/api/chat", false, ""})
	test("nonstreaming native generate", performanceExtractQualification{"ollama", "/api/generate", false, ""})
}

func TestPerformanceExtractPartialMetadata(t *testing.T) {
	q := performanceExtractQualification{"ollama", "/api/chat", true, ""}
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"done":false}`), q), performanceExtractMetadata{})
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"load_duration":7}`), q),
		performanceExtractMetadata{LoadNs: performanceExtractTestInt(7)})
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"eval_duration":52479709}`), q),
		performanceExtractMetadata{DecodeNs: performanceExtractTestInt(52479709)})
	// Completion is the caller's concern: the same numeric object extracts the
	// same way regardless of done, its absence, or finish_reason.
	for _, prefix := range []string{`"done":false,`, `"done":true,`, `"finish_reason":null,`, ""} {
		performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{`+prefix+`"eval_count":18}`), q),
			performanceExtractMetadata{OutputTokens: performanceExtractTestInt(18)})
	}
}

func TestPerformanceExtractBoundedPayload(t *testing.T) {
	q := performanceExtractQualification{"ollama", "/api/chat", true, ""}
	prefix, suffix := `{"eval_count":18,"padding":"`, `"}`
	payload := []byte(prefix + strings.Repeat(" ", int(streamFrameMaxBytes)-len(prefix)-len(suffix)) + suffix)
	performanceExtractTestEqual(t, extractPerformancePayload(payload, q),
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(18)})
	payload = append(payload, ' ')
	performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{})
}

func TestPerformanceExtractRepeatedPayloadIsIndependent(t *testing.T) {
	q := performanceExtractQualification{"ollama", "/api/chat", true, ""}
	payload := performanceExtractTestFixture(t, "ollama-complete.json")
	original := bytes.Clone(payload)
	want := extractPerformancePayload(payload, q)
	first := extractPerformancePayload(payload, q)
	second := extractPerformancePayload(payload, q)
	performanceExtractTestEqual(t, first, want)
	performanceExtractTestEqual(t, second, want)
	if !bytes.Equal(payload, original) {
		t.Fatal("extractor mutated its input")
	}
	if first.OutputTokens == nil || second.OutputTokens == nil {
		t.Fatal("expected independently extracted output counts")
	}
	*first.OutputTokens = 0
	clear(payload)
	performanceExtractTestEqual(t, second, want)
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
	test("native timings with usage", performanceExtractTestFixture(t, "llamacpp-chat-stream-usage.txt"),
		performanceExtractTestInt(250500000), performanceExtractTestFloat(39.92015968063872))
	test("usage without timings retains count and unknown rates", []byte(
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"fixture\"},\"finish_reason\":null}]}\n\n"+
			"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":11,\"total_tokens\":26}}\n\n"+
			"data: [DONE]\n\n"), nil, nil)
}

func TestPerformanceExtractExistingEngineSchemas(t *testing.T) {
	payload := []byte(`{"eval_count":11,"eval_duration":250500000,"usage":{"completion_tokens":11},"timings":{"predicted_n":11,"predicted_ms":250.5,"predicted_per_second":39.92015968063872}}`)
	test := func(name string, q performanceExtractQualification, want performanceExtractMetadata) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload(payload, q), want)
		})
	}
	test("Ollama native keeps legacy interval", performanceExtractQualification{"ollama", "/api/chat", true, ""},
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(11), DecodeNs: performanceExtractTestInt(250500000)})
	test("Ollama compatibility ignores native timing extensions", performanceExtractQualification{"ollama", "/v1/chat/completions", true, ""},
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(11)})
	test("Ollama native rejects foreign schema", performanceExtractQualification{"ollama", "/api/chat", true, performanceExtractLlamaSchema}, performanceExtractMetadata{})
	test("LM Studio compatibility remains unqualified", performanceExtractQualification{"lmstudio", "/v1/chat/completions", true, ""}, performanceExtractMetadata{})
	test("LM Studio cannot opt into llama schema", performanceExtractQualification{"lmstudio", "/v1/chat/completions", true, performanceExtractLlamaSchema}, performanceExtractMetadata{})
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
	test("Ollama with foreign schema", performanceExtractQualification{"ollama", "/v1/chat/completions", true, performanceExtractLlamaSchema})
}

func TestPerformanceExtractDailyPhaseConsistency(t *testing.T) {
	llama := performanceExtractQualification{"llamacpp", "/v1/chat/completions", true, performanceExtractLlamaSchema}
	strata := performanceExtractQualification{"strata", "/v1/chat/completions", true, performanceExtractStrataSchema}
	test := func(name, payload string, q performanceExtractQualification, want performanceExtractMetadata) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload([]byte(payload), q), want)
		})
	}
	zeroLlama := performanceExtractMetadata{OutputTokens: performanceExtractTestInt(1), EngineDecodeNs: performanceExtractTestInt(125000),
		EngineDecodeTokensPerSecond: performanceExtractTestFloat(0)}
	test("one llama output has zero decode steps", `{"timings":{"predicted_n":1,"predicted_ms":0.125,"predicted_per_second":0,"predicted_per_token_ms":0}}`, llama, zeroLlama)
	test("null per token when no work", `{"timings":{"predicted_n":1,"predicted_ms":0.125,"predicted_per_second":0,"predicted_per_token_ms":null}}`, llama, zeroLlama)
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
