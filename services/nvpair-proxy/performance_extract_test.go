// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"nvpair-shared/performancewire"
)

func performanceExtractTestInt(value int64) *int64 { return &value }

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
			performanceExtractQualification{"ollama", "/api/chat", true}), want)
	})
	t.Run("generate terminal", func(t *testing.T) {
		lines := bytes.Split(bytes.TrimSpace(performanceExtractTestFixture(t, "ollama-stream.txt")), []byte("\n"))
		if len(lines) != 2 || !json.Valid(lines[0]) || !json.Valid(lines[1]) {
			t.Fatal("expected partial and terminal JSON fixture payloads")
		}
		performanceExtractTestEqual(t, extractPerformancePayload(lines[1],
			performanceExtractQualification{"ollama", "/api/generate", true}), want)
	})
}

func TestPerformanceExtractCompatibilityUsage(t *testing.T) {
	want := performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20)}
	test := func(name, path string, streaming bool, payload []byte) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload(payload,
				performanceExtractQualification{"ollama", path, streaming}), want)
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
	q := performanceExtractQualification{"ollama", "/api/chat", true}
	performanceExtractTestEqual(t, extractPerformancePayload(
		[]byte(`{"eval_count":0,"eval_duration":0,"load_duration":0,"prompt_eval_duration":0}`), q),
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(0), DecodeNs: performanceExtractTestInt(0),
			LoadNs: performanceExtractTestInt(0), PrefillNs: performanceExtractTestInt(0)})
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{}`), q), performanceExtractMetadata{})
	performanceExtractTestEqual(t, extractPerformancePayload([]byte(`{"usage":{"completion_tokens":0}}`),
		performanceExtractQualification{"ollama", "/v1/chat/completions", true}),
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(0)})
}

func TestPerformanceExtractUnknownPromptCacheSemantics(t *testing.T) {
	payload := []byte(`{"prompt_eval_count":11,"prompt_eval_cached_count":7,"prompt_eval_duration":13074791,
		"usage":{"prompt_tokens":11,"prompt_tokens_details":{"cached_tokens":7}}}`)
	performanceExtractTestEqual(t, extractPerformancePayload(payload,
		performanceExtractQualification{"ollama", "/api/chat", true}),
		performanceExtractMetadata{PrefillNs: performanceExtractTestInt(13074791)})
	performanceExtractTestEqual(t, extractPerformancePayload(payload,
		performanceExtractQualification{"ollama", "/v1/chat/completions", true}), performanceExtractMetadata{})
}

func TestPerformanceExtractInvalidNumbersPreserveUnrelatedMetadata(t *testing.T) {
	test := func(name, field, value string) {
		t.Run(name, func(t *testing.T) {
			payload := []byte(fmt.Sprintf(`{"%s":%s,"load_duration":7}`, field, value))
			performanceExtractTestEqual(t, extractPerformancePayload(payload,
				performanceExtractQualification{"ollama", "/api/chat", true}),
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
			performanceExtractQualification{"ollama", "/api/chat", true}),
			performanceExtractMetadata{OutputTokens: performanceExtractTestInt(18), DecodeNs: performanceExtractTestInt(52479709)})
	})
}

func TestPerformanceExtractDuplicateFields(t *testing.T) {
	test := func(name, payload string, want performanceExtractMetadata) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload([]byte(payload),
				performanceExtractQualification{"ollama", "/api/chat", true}), want)
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
				performanceExtractQualification{"ollama", "/v1/chat/completions", true}), performanceExtractMetadata{})
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
			performanceExtractQualification{"ollama", "/v1/chat/completions", true}),
			performanceExtractMetadata{OutputTokens: performanceExtractTestInt(20)})
	})
}

func TestPerformanceExtractImpossibleDecodePairs(t *testing.T) {
	test := func(name string, tokens, ns int64, wantDecode *int64) {
		t.Run(name, func(t *testing.T) {
			payload := []byte(fmt.Sprintf(`{"eval_count":%d,"eval_duration":%d,"load_duration":7}`, tokens, ns))
			performanceExtractTestEqual(t, extractPerformancePayload(payload,
				performanceExtractQualification{"ollama", "/api/generate", true}),
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
	q := performanceExtractQualification{"ollama", "/api/chat", true}
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
	q := performanceExtractQualification{"ollama", "/api/chat", true}
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
			performanceExtractQualification{"ollama", "/v1/chat/completions", true}), performanceExtractMetadata{})
	})
}

func TestPerformanceExtractQualification(t *testing.T) {
	payload := []byte(`{"eval_count":18,"eval_duration":52479709,"load_duration":7,"usage":{"completion_tokens":20}}`)
	test := func(name string, q performanceExtractQualification) {
		t.Run(name, func(t *testing.T) {
			performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{})
		})
	}
	test("unknown engine", performanceExtractQualification{"other", "/api/chat", true})
	test("other engine compatibility", performanceExtractQualification{"llamacpp", "/v1/chat/completions", true})
	test("missing engine", performanceExtractQualification{"", "/api/chat", true})
	test("unknown API", performanceExtractQualification{"ollama", "/api/embed", true})
	test("missing API", performanceExtractQualification{"ollama", "", true})
	test("nonstreaming native chat", performanceExtractQualification{"ollama", "/api/chat", false})
	test("nonstreaming native generate", performanceExtractQualification{"ollama", "/api/generate", false})
}

func TestPerformanceExtractPartialMetadata(t *testing.T) {
	q := performanceExtractQualification{"ollama", "/api/chat", true}
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
	q := performanceExtractQualification{"ollama", "/api/chat", true}
	prefix, suffix := `{"eval_count":18,"padding":"`, `"}`
	payload := []byte(prefix + strings.Repeat(" ", int(streamFrameMaxBytes)-len(prefix)-len(suffix)) + suffix)
	performanceExtractTestEqual(t, extractPerformancePayload(payload, q),
		performanceExtractMetadata{OutputTokens: performanceExtractTestInt(18)})
	payload = append(payload, ' ')
	performanceExtractTestEqual(t, extractPerformancePayload(payload, q), performanceExtractMetadata{})
}

func TestPerformanceExtractRepeatedPayloadIsIndependent(t *testing.T) {
	q := performanceExtractQualification{"ollama", "/api/chat", true}
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
