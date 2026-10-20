// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"math"
	"strconv"

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
		result.OutputTokens = performanceExtractNumber(fields.value("eval_count"), performancewire.MaxTokens)
		result.DecodeNs = performanceExtractNumber(fields.value("eval_duration"), performancewire.MaxPhaseNs)
		result.LoadNs = performanceExtractNumber(fields.value("load_duration"), performancewire.MaxPhaseNs)
		result.PrefillNs = performanceExtractNumber(fields.value("prompt_eval_duration"), performancewire.MaxPhaseNs)
		if result.OutputTokens != nil && result.DecodeNs != nil {
			// Preserve the independently valid count, but refuse a duration that
			// would make the pair impossible or exceed the shared rate contract.
			if (*result.DecodeNs == 0 && *result.OutputTokens > 0) ||
				(*result.DecodeNs > 0 && performancewire.TokensPerSecond(*result.OutputTokens, *result.DecodeNs) > performancewire.MaxTokensPerSecond) {
				result.DecodeNs = nil
			}
		}
	case "/v1/chat/completions", "/v1/completions":
		usage := performanceExtractObject(fields.value("usage"))
		result.OutputTokens = performanceExtractNumber(usage.value("completion_tokens"), performancewire.MaxTokens)
		// Compatibility timing extensions are deliberately unqualified.
	}
	return result
}

// performanceExtractObject preserves duplicate-key ambiguity instead of JSON's
// usual last-value-wins behavior. A repeated key stays present with a nil value,
// including escaped spellings of the same key and duplicate usage containers.
// Invalid JSON invalidates the object, never inference or an unrelated frame.
type performanceExtractFields []byte

func performanceExtractObject(payload []byte) performanceExtractFields {
	payload = bytes.TrimSpace(payload)
	if len(payload) == 0 || payload[0] != '{' || !ownedJSONValid(payload) {
		return nil
	}
	return performanceExtractFields(payload)
}

// Only the fixed set of requested numeric fields is visited. Values borrow the
// validated object; duplicate decoded spellings stay present but unusable.
func (fields performanceExtractFields) field(name string) ([]byte, bool) {
	if len(fields) == 0 {
		return nil, false
	}
	var raw []byte
	count := 0
	visitObject(fields, func(key []byte, start, end int) {
		if jsonStringEquals(key, name) {
			count++
			raw = fields[start:end]
		}
	})
	if count != 1 {
		raw = nil
	}
	return raw, count > 0
}

func (fields performanceExtractFields) value(name string) []byte {
	raw, _ := fields.field(name)
	return raw
}

func performanceExtractNumber(raw json.RawMessage, max int64) *int64 {
	// Integer JSON tokens only: strings, null, fractions and exponent notation
	// do not establish an exact engine counter or nanosecond duration here.
	if len(raw) == 0 || len(raw) > 19 || raw[0] < '0' || raw[0] > '9' {
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
func performanceExtractDaily(fields performanceExtractFields, strata bool) performanceExtractMetadata {
	var result performanceExtractMetadata
	timings := performanceExtractObject(fields.value("timings"))
	usage := performanceExtractObject(fields.value("usage"))
	if _, present := fields.field("timings"); present && timings == nil {
		result.invalid |= 0xcf
	}
	result.OutputTokens = performanceExtractReconcileCount(timings, "predicted_n", usage, "completion_tokens", &result.invalid, 1<<2)
	result.PromptTokens = performanceExtractRead(timings, "prompt_n", performanceExtractTokenCount, &result.invalid, 1<<0)
	result.CachedPromptTokens = performanceExtractRead(timings, "cache_n", performanceExtractTokenCount, &result.invalid, 1<<1)
	if strata {
		details := performanceExtractObject(usage.value("prompt_tokens_details"))
		result.CachedPromptTokens = performanceExtractReconcileCount(timings, "cache_n", details, "cached_tokens", &result.invalid, 1<<1)
		if raw, present := usage.field("prompt_tokens_details"); present && performanceExtractObject(raw) == nil {
			result.CachedPromptTokens = nil
			result.invalid |= 1 << 1
		}
	}
	// An ambiguous or malformed usage container cannot corroborate its counters.
	if raw, present := fields.field("usage"); present && performanceExtractObject(raw) == nil {
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
	if _, present := timings.field("predicted_per_second"); present && result.EngineDecodeTokensPerSecond == nil {
		result.invalid |= 1 << 7
	}
	return result
}

func performanceExtractStrataMCP(fields performanceExtractFields) bool {
	if _, present := fields.field("strata_mcp"); present {
		return true
	}
	// openai_collect places its internal-round marker on each collected message.
	choices := bytes.TrimSpace(fields.value("choices"))
	if len(choices) == 0 || choices[0] != '[' {
		return false
	}
	for rest := bytes.TrimSpace(choices[1:]); len(rest) > 0 && rest[0] != ']'; {
		end := streamJSONValueEnd(rest)
		object := performanceExtractObject(rest[:end])
		message := performanceExtractObject(object.value("message"))
		if _, present := message.field("strata_mcp"); present {
			return true
		}
		rest = bytes.TrimSpace(rest[end:])
		if len(rest) > 0 && rest[0] == ',' {
			rest = bytes.TrimSpace(rest[1:])
		}
	}
	return false
}

// Read and record validity together so auxiliary fields cannot be omitted from
// a separate observer-maintained list of fields that invalidate prior evidence.
func performanceExtractRead[T any](fields performanceExtractFields, key string, parse func(json.RawMessage) *T, invalid *uint16, bit uint16) *T {
	raw, present := fields.field(key)
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

func performanceExtractReconcileCount(a performanceExtractFields, ak string, b performanceExtractFields, bk string, invalid *uint16, bit uint16) *int64 {
	x := performanceExtractRead(a, ak, performanceExtractTokenCount, invalid, bit)
	y := performanceExtractRead(b, bk, performanceExtractTokenCount, invalid, bit)
	if _, present := a.field(ak); present && x == nil {
		return nil
	}
	if _, present := b.field(bk); present && y == nil {
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

// Decimal input has at most 64 wire characters and |exponent| <=308. Keep
// exact base-ten digits in fixed state instead of math/big's retained nat pool.
// These bounds also mean every accepted value has <=372 fractional places;
// canonical text therefore equals the former Rat.FloatString(372), trimmed.
type performanceDecimal struct {
	digits   [64]byte
	n, point int
	negative bool
}

func parsePerformanceDecimal(raw []byte) (performanceDecimal, bool) {
	var d performanceDecimal
	if len(raw) == 0 || len(raw) > 64 || !ownedJSONValid(raw) {
		return d, false
	}
	start, end := 0, len(raw)
	if raw[0] == '-' {
		d.negative = true
		start++
	}
	if start == len(raw) || raw[start] < '0' || raw[start] > '9' {
		return d, false
	}
	exponent := 0
	if e := bytes.IndexAny(raw, "eE"); e >= 0 {
		end = e
		pos, sign := e+1, 1
		if raw[pos] == '-' || raw[pos] == '+' {
			if raw[pos] == '-' {
				sign = -1
			}
			pos++
		}
		for ; pos < len(raw); pos++ {
			exponent = exponent*10 + int(raw[pos]-'0')
			if exponent > 308 {
				return d, false
			}
		}
		exponent *= sign
	}
	dot := false
	for i := start; i < end; i++ {
		if raw[i] == '.' {
			d.point = d.n
			dot = true
		} else {
			d.digits[d.n] = raw[i]
			d.n++
		}
	}
	if !dot {
		d.point = d.n
	}
	d.point += exponent
	first := 0
	for first < d.n && d.digits[first] == '0' {
		first++
	}
	if first == d.n {
		d.digits[0] = '0'
		d.n, d.point, d.negative = 1, 1, false
		return d, true
	}
	copy(d.digits[:], d.digits[first:d.n])
	d.n -= first
	d.point -= first
	for d.n > 1 && d.digits[d.n-1] == '0' {
		d.n--
	}
	return d, true
}

func (d performanceDecimal) text() string {
	var out [768]byte
	i := 0
	if d.negative {
		out[i] = '-'
		i++
	}
	if d.point <= 0 {
		out[i], out[i+1] = '0', '.'
		i += 2
		for j := 0; j < -d.point; j++ {
			out[i] = '0'
			i++
		}
		i += copy(out[i:], d.digits[:d.n])
	} else if d.point >= d.n {
		i += copy(out[i:], d.digits[:d.n])
		for j := d.n; j < d.point; j++ {
			out[i] = '0'
			i++
		}
	} else {
		i += copy(out[i:], d.digits[:d.point])
		out[i] = '.'
		i++
		i += copy(out[i:], d.digits[d.point:d.n])
	}
	return string(out[:i])
}

// Compare against an integer maximum before rounding or float conversion, so
// a tiny positive excess cannot disappear into the target representation.
func (d performanceDecimal) floor(scale int, maximum int64) (int64, bool, bool) {
	if d.digits[0] == '0' {
		return 0, false, true
	}
	point := d.point + scale
	if point > 19 {
		return 0, false, false
	}
	value := int64(0)
	for i := 0; i < point; i++ {
		digit := int64(0)
		if i < d.n {
			digit = int64(d.digits[i] - '0')
		}
		if value > (maximum-digit)/10 {
			return 0, false, false
		}
		value = value*10 + digit
	}
	fraction := point < d.n
	if value > maximum || value == maximum && fraction {
		return 0, false, false
	}
	return value, fraction, true
}

func performanceExtractMilliseconds(raw json.RawMessage) *int64 {
	value, ok := parsePerformanceDecimal(raw)
	if !ok || raw[0] < '0' || raw[0] > '9' {
		return nil
	}
	ns, _, ok := value.floor(6, performancewire.MaxPhaseNs)
	if !ok {
		return nil
	}
	point := value.point + 6
	if point >= 0 && point < value.n && value.digits[point] >= '5' {
		ns++
	}
	return &ns
}

// Expand the exact binary maximum in fixed base-ten state. A finite float64
// has <=53 mantissa bits and denominator <=2^1074: multiplying the mantissa by
// 5^1074 needs fewer than 800 decimal digits. No rounded decimal maximum and
// no big-number pool participates in this comparison.
func (d performanceDecimal) lessEqualFloat64(maximum float64) bool {
	if math.IsNaN(maximum) || math.IsInf(maximum, 0) || maximum < 0 {
		return false
	}
	if d.digits[0] == '0' {
		return true
	}
	if maximum == 0 {
		return false
	}
	bits := math.Float64bits(maximum)
	exponent := int((bits >> 52) & 0x7ff)
	mantissa := bits & ((uint64(1) << 52) - 1)
	if exponent == 0 {
		exponent = -1074
	} else {
		mantissa |= uint64(1) << 52
		exponent -= 1023 + 52
	}
	var digits [800]byte
	n := 0
	for mantissa > 0 {
		digits[n] = byte(mantissa % 10)
		n++
		mantissa /= 10
	}
	scale, factor, steps := 0, 2, exponent
	if exponent < 0 {
		scale, factor, steps = -exponent, 5, -exponent
	}
	for step := 0; step < steps; step++ {
		carry := 0
		for i := 0; i < n; i++ {
			v := int(digits[i])*factor + carry
			digits[i] = byte(v % 10)
			carry = v / 10
		}
		if carry > 0 {
			digits[n] = byte(carry)
			n++
		}
	}
	point := n - scale
	if d.point != point {
		return d.point < point
	}
	for i := 0; i < max(d.n, n); i++ {
		x, y := byte(0), byte(0)
		if i < d.n {
			x = d.digits[i] - '0'
		}
		if i < n {
			y = digits[n-1-i]
		}
		if x != y {
			return x < y
		}
	}
	return true
}

func performanceExtractDecimal(raw json.RawMessage, maximum float64) *float64 {
	value, ok := parsePerformanceDecimal(raw)
	if !ok || raw[0] < '0' || raw[0] > '9' {
		return nil
	}
	if !value.lessEqualFloat64(maximum) {
		return nil
	}
	n, err := strconv.ParseFloat(string(raw), 64)
	if err != nil || math.IsNaN(n) || math.IsInf(n, 0) || n < 0 || (n == 0 && value.digits[0] != '0') {
		return nil
	}
	return &n
}

// Missing auxiliaries leave independent measurements intact. Present invalid or
// contradictory auxiliaries disqualify only their phase. Strata's decode work
// count is unavailable: it must never be inferred from output or rounded rates.
func performanceExtractPhase(fields performanceExtractFields, prefix string, count, ns *int64, rate *float64, strata bool) bool {
	if raw, present := fields.field(prefix + "_per_second"); present && rate == nil && !bytes.Equal(raw, []byte("null")) {
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
	if raw, present := fields.field(prefix + "_per_token_ms"); present {
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
