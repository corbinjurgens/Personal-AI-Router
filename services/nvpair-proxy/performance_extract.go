// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"math"
	"math/big"
	"strconv"
	"strings"

	"nvpair-shared/performancewire"
)

// performanceExtractQualification describes the actual engine-facing API. The
// caller owns terminal classification and must supply native terminal payloads
// from SC's reader; this extractor never interprets done or finish_reason.
type performanceExtractQualification struct {
	EngineFamily string
	APIPath      string
	Streaming    bool
	SchemaID     string
}

const (
	performanceExtractLlamaSchema  = "llamacpp-b11146-7fe450e1"
	performanceExtractStrataSchema = "strata-f6726098"
)

// performanceExtractMetadata contains optional engine-reported numbers and a
// content-free phase-disqualification flag. Only pinned daily schemas establish
// evaluated prompt/cache counts; Ollama keeps its original duration-only mapping.
type performanceExtractMetadata struct {
	PromptTokens                *int64
	CachedPromptTokens          *int64
	OutputTokens                *int64
	PrefillNs                   *int64
	DecodeNs                    *int64
	LoadNs                      *int64
	EngineDecodeNs              *int64
	EngineDecodeTokensPerSecond *float64
	PhaseDisqualified           bool
	invalid                     uint16 // sticky ambiguity/invalid evidence across upstream objects
}

// extractPerformancePayload consumes one bounded JSON object synchronously. It
// neither mutates nor retains payload, and returns no content or protocol verdict.
func extractPerformancePayload(payload []byte, q performanceExtractQualification) performanceExtractMetadata {
	result := extractPerformanceMetadata(payload, q)
	// A standalone object has no prior evidence to invalidate. The observer
	// retains these private bits when merging objects across a response.
	result.invalid = 0
	return result
}

func extractPerformanceMetadata(payload []byte, q performanceExtractQualification) performanceExtractMetadata {
	var result performanceExtractMetadata
	if int64(len(payload)) > streamFrameMaxBytes {
		return result
	}
	fields := performanceExtractObject(payload)
	if q.APIPath == "/v1/chat/completions" &&
		((q.EngineFamily == "llamacpp" && q.SchemaID == performanceExtractLlamaSchema) ||
			(q.EngineFamily == "strata" && q.SchemaID == performanceExtractStrataSchema)) {
		return performanceExtractDaily(fields, q.EngineFamily == "strata")
	}
	if q.EngineFamily != "ollama" || q.SchemaID != "" {
		return result
	}
	switch q.APIPath {
	case "/api/chat", "/api/generate":
		// There is no full-response collector or nonstreaming rate qualification.
		if !q.Streaming {
			return result
		}
		result.OutputTokens = performanceExtractNumber(fields["eval_count"], performancewire.MaxTokens)
		result.DecodeNs = performanceExtractNumber(fields["eval_duration"], performancewire.MaxPhaseNs)
		result.LoadNs = performanceExtractNumber(fields["load_duration"], performancewire.MaxPhaseNs)
		result.PrefillNs = performanceExtractNumber(fields["prompt_eval_duration"], performancewire.MaxPhaseNs)
		if result.OutputTokens != nil && result.DecodeNs != nil {
			// Preserve the independently valid count, but refuse a duration that
			// would make the pair impossible or exceed the shared rate contract.
			if (*result.DecodeNs == 0 && *result.OutputTokens > 0) ||
				(*result.DecodeNs > 0 && performancewire.TokensPerSecond(*result.OutputTokens, *result.DecodeNs) > performancewire.MaxTokensPerSecond) {
				result.DecodeNs = nil
			}
		}
	case "/v1/chat/completions", "/v1/completions":
		usage := performanceExtractObject(fields["usage"])
		result.OutputTokens = performanceExtractNumber(usage["completion_tokens"], performancewire.MaxTokens)
		// Compatibility timing extensions are deliberately unqualified.
	}
	return result
}

// performanceExtractObject preserves duplicate-key ambiguity instead of JSON's
// usual last-value-wins behavior. A repeated key stays present with a nil value,
// including escaped spellings of the same key and duplicate usage containers.
// Invalid JSON invalidates the object, never inference or an unrelated frame.
func performanceExtractObject(payload []byte) map[string]json.RawMessage {
	payload = bytes.TrimSpace(payload)
	if len(payload) == 0 || payload[0] != '{' || !json.Valid(payload) {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(payload))
	if _, err := dec.Token(); err != nil {
		return nil
	}
	fields := make(map[string]json.RawMessage)
	for dec.More() {
		token, err := dec.Token()
		if err != nil {
			return nil
		}
		key, ok := token.(string)
		if !ok {
			return nil
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return nil
		}
		if _, exists := fields[key]; exists {
			fields[key] = nil
		} else {
			fields[key] = value
		}
	}
	return fields
}

func performanceExtractNumber(raw json.RawMessage, max int64) *int64 {
	// Integer JSON tokens only: strings, null, fractions and exponent notation
	// do not establish an exact engine counter or nanosecond duration here.
	if len(raw) == 0 || raw[0] < '0' || raw[0] > '9' {
		return nil
	}
	value, err := strconv.ParseInt(string(raw), 10, 64)
	if err != nil || value > max {
		return nil
	}
	return &value
}

// Daily schemas bind evaluated prompt work and engine-native decode semantics.
// They never assign the legacy visible-output-count decode interval.
func performanceExtractDaily(fields map[string]json.RawMessage, strata bool) performanceExtractMetadata {
	var result performanceExtractMetadata
	timings := performanceExtractObject(fields["timings"])
	usage := performanceExtractObject(fields["usage"])
	if _, present := fields["timings"]; present && timings == nil {
		result.invalid |= 0xcf
	}
	result.OutputTokens = performanceExtractReconcileCount(timings, "predicted_n", usage, "completion_tokens", &result.invalid, 1<<2)
	result.PromptTokens = performanceExtractRead(timings, "prompt_n", performanceExtractTokenCount, &result.invalid, 1<<0)
	result.CachedPromptTokens = performanceExtractRead(timings, "cache_n", performanceExtractTokenCount, &result.invalid, 1<<1)
	if strata {
		details := performanceExtractObject(usage["prompt_tokens_details"])
		result.CachedPromptTokens = performanceExtractReconcileCount(timings, "cache_n", details, "cached_tokens", &result.invalid, 1<<1)
		if raw, present := usage["prompt_tokens_details"]; present && performanceExtractObject(raw) == nil {
			result.CachedPromptTokens = nil
			result.invalid |= 1 << 1
		}
	}
	// An ambiguous or malformed usage container cannot corroborate its counters.
	if raw, present := fields["usage"]; present && performanceExtractObject(raw) == nil {
		result.OutputTokens = nil
		result.invalid |= 1 << 2
		if strata {
			result.CachedPromptTokens = nil
			result.invalid |= 1 << 1
		}
	}
	if strata && performanceExtractStrataMCP(fields) {
		result.PhaseDisqualified = true
		return result
	}
	result.PrefillNs = performanceExtractRead(timings, "prompt_ms", performanceExtractMilliseconds, &result.invalid, 1<<3)
	result.EngineDecodeNs = performanceExtractRead(timings, "predicted_ms", performanceExtractMilliseconds, &result.invalid, 1<<6)
	result.EngineDecodeTokensPerSecond = performanceExtractRead(timings, "predicted_per_second", performanceExtractRate, &result.invalid, 1<<7)
	promptRate := performanceExtractRead(timings, "prompt_per_second", performanceExtractRate, &result.invalid, 1<<0|1<<3)
	if !performanceExtractPhase(timings, "prompt", result.PromptTokens, result.PrefillNs, promptRate, strata) {
		result.PromptTokens, result.PrefillNs = nil, nil
		result.invalid |= 1<<0 | 1<<3
	}
	steps := result.OutputTokens
	if !strata && steps != nil {
		n := max(*steps-1, 0)
		steps = &n
	}
	if !performanceExtractPhase(timings, "predicted", steps, result.EngineDecodeNs, result.EngineDecodeTokensPerSecond, strata) {
		result.EngineDecodeNs, result.EngineDecodeTokensPerSecond = nil, nil
		result.invalid |= 1<<6 | 1<<7
	}
	if result.EngineDecodeNs == nil || *result.EngineDecodeNs == 0 || (!strata && steps == nil) {
		result.EngineDecodeTokensPerSecond = nil
	}
	if _, present := timings["predicted_per_second"]; present && result.EngineDecodeTokensPerSecond == nil {
		result.invalid |= 1 << 7
	}
	return result
}

func performanceExtractStrataMCP(fields map[string]json.RawMessage) bool {
	if _, present := fields["strata_mcp"]; present {
		return true
	}
	// openai_collect places its internal-round marker on each collected message.
	var choices []json.RawMessage
	if err := json.Unmarshal(fields["choices"], &choices); err != nil {
		return false
	}
	for _, choice := range choices {
		object := performanceExtractObject(choice)
		message := performanceExtractObject(object["message"])
		if _, present := message["strata_mcp"]; present {
			return true
		}
	}
	return false
}

// Read and record validity together so auxiliary fields cannot be omitted from
// a separate observer-maintained list of fields that invalidate prior evidence.
func performanceExtractRead[T any](fields map[string]json.RawMessage, key string, parse func(json.RawMessage) *T, invalid *uint16, bit uint16) *T {
	raw, present := fields[key]
	value := parse(raw)
	if present && value == nil {
		*invalid |= bit
	}
	return value
}

func performanceExtractTokenCount(raw json.RawMessage) *int64 {
	return performanceExtractNumber(raw, performancewire.MaxTokens)
}

func performanceExtractRate(raw json.RawMessage) *float64 {
	return performanceExtractDecimal(raw, performancewire.MaxTokensPerSecond)
}

func performanceExtractReconcileCount(a map[string]json.RawMessage, ak string, b map[string]json.RawMessage, bk string, invalid *uint16, bit uint16) *int64 {
	x := performanceExtractRead(a, ak, performanceExtractTokenCount, invalid, bit)
	y := performanceExtractRead(b, bk, performanceExtractTokenCount, invalid, bit)
	if _, present := a[ak]; present && x == nil {
		return nil
	}
	if _, present := b[bk]; present && y == nil {
		return nil
	}
	if x != nil && y != nil && *x != *y {
		*invalid |= bit
		return nil
	}
	if x != nil {
		return x
	}
	return y
}

// Bound lexical work before allocating arbitrary-precision numbers. Counts keep
// their original integer-only parser; only the pinned decimal fields use this.
func performanceExtractRational(raw json.RawMessage) *big.Rat {
	if len(raw) == 0 || len(raw) > 64 || raw[0] < '0' || raw[0] > '9' || !json.Valid(raw) {
		return nil
	}
	if i := strings.IndexAny(string(raw), "eE"); i >= 0 {
		e, err := strconv.Atoi(string(raw[i+1:]))
		if err != nil || e < -308 || e > 308 {
			return nil
		}
	}
	value, ok := new(big.Rat).SetString(string(raw))
	if !ok || value.Sign() < 0 {
		return nil
	}
	return value
}

func performanceExtractMilliseconds(raw json.RawMessage) *int64 {
	value := performanceExtractRational(raw)
	if value == nil {
		return nil
	}
	value.Mul(value, big.NewRat(1_000_000, 1))
	if value.Cmp(big.NewRat(performancewire.MaxPhaseNs, 1)) > 0 {
		return nil
	}
	quotient, remainder := new(big.Int), new(big.Int)
	quotient.QuoRem(value.Num(), value.Denom(), remainder)
	if remainder.Lsh(remainder, 1).Cmp(value.Denom()) >= 0 {
		quotient.Add(quotient, big.NewInt(1))
	}
	if !quotient.IsInt64() || quotient.Int64() > performancewire.MaxPhaseNs {
		return nil
	}
	ns := quotient.Int64()
	return &ns
}

func performanceExtractDecimal(raw json.RawMessage, maximum float64) *float64 {
	value := performanceExtractRational(raw)
	if value == nil {
		return nil
	}
	limit := new(big.Rat).SetFloat64(maximum)
	if value.Cmp(limit) > 0 {
		return nil
	}
	n, _ := value.Float64()
	if math.IsNaN(n) || math.IsInf(n, 0) || n < 0 || (n == 0 && value.Sign() > 0) {
		return nil
	}
	return &n
}

// Missing auxiliaries leave independent measurements intact. Present invalid or
// contradictory auxiliaries disqualify only their phase. Strata's decode work
// count is unavailable: it must never be inferred from output or rounded rates.
func performanceExtractPhase(fields map[string]json.RawMessage, prefix string, count, ns *int64, rate *float64, strata bool) bool {
	if raw, present := fields[prefix+"_per_second"]; present && rate == nil && !bytes.Equal(raw, []byte("null")) {
		return false
	}
	countKnown := count != nil && !(strata && prefix == "predicted")
	noWork := countKnown && *count == 0
	if strata && prefix == "predicted" {
		noWork = (rate != nil && *rate == 0) || (count != nil && *count == 0 && rate == nil)
	}
	if ns != nil {
		if rate != nil && *rate > 0 && *ns == 0 {
			return false
		}
		if countKnown {
			if *ns == 0 && *count > 0 {
				return false
			}
			if *ns > 0 {
				expected := float64(*count) * 1e9 / float64(*ns)
				if expected > performancewire.MaxTokensPerSecond {
					return false
				}
				if rate != nil && !performanceExtractRateMatches(*rate, *count, *ns, strata) {
					return false
				}
			}
		}
	}
	if raw, present := fields[prefix+"_per_token_ms"]; present {
		perToken := performanceExtractDecimal(raw, float64(performancewire.MaxPhaseNs)/1e6)
		if noWork && (bytes.Equal(raw, []byte("null")) || (perToken != nil && *perToken == 0)) {
			return true
		}
		if perToken == nil {
			return false
		}
		if strata {
			if countKnown && ns != nil && *count > 0 {
				low := float64(*count) * math.Max(0, *perToken-0.0005) * 1e6
				high := float64(*count) * (*perToken + 0.0005) * 1e6
				if low > float64(*ns)+50_000.5 || high < float64(*ns)-50_000.5 {
					return false
				}
			}
			if rate != nil {
				// The rounded rate and reciprocal time intervals must overlap.
				low := math.Max(0, *rate-0.05) * math.Max(0, *perToken-0.0005)
				high := (*rate + 0.05) * (*perToken + 0.0005)
				if low > 1000 || high < 1000 {
					return false
				}
			}
		} else if *perToken == 0 {
			return false
		} else if countKnown && ns != nil && *ns > 0 {
			if !performanceExtractRateMatches(1000 / *perToken, *count, *ns, false) {
				return false
			}
		}
	}
	return true
}

func performanceExtractRateMatches(rate float64, count, ns int64, strata bool) bool {
	uncertainty, rateError := 0.5, 0.0 // nanosecond rounding, llama relative tolerance
	if strata {
		uncertainty, rateError = 50_000.5, 0.05
	}
	low := float64(count) * 1e9 / (float64(ns) + uncertainty)
	high := math.Inf(1)
	if float64(ns) > uncertainty {
		high = float64(count) * 1e9 / (float64(ns) - uncertainty)
	}
	if count == 0 {
		high = 0
	}
	if !strata {
		low *= 1 - 1e-6
		high *= 1 + 1e-6
	}
	return rate+rateError >= low && rate-rateError <= high
}
