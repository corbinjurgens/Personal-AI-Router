// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
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
}

// performanceExtractMetadata contains only optional engine-reported numbers.
// Prompt/cache counts remain unknown until version-specific semantics are pinned.
// PrefillNs preserves the documented uncached prompt evaluation duration without
// claiming an evaluated token count or a prefill rate.
type performanceExtractMetadata struct {
	PromptTokens       *int64
	CachedPromptTokens *int64
	OutputTokens       *int64
	PrefillNs          *int64
	DecodeNs           *int64
	LoadNs             *int64
}

// extractPerformancePayload consumes one bounded JSON object synchronously. It
// neither mutates nor retains payload, and returns no content or protocol verdict.
func extractPerformancePayload(payload []byte, q performanceExtractQualification) performanceExtractMetadata {
	var result performanceExtractMetadata
	if q.EngineFamily != "ollama" || int64(len(payload)) > streamFrameMaxBytes {
		return result
	}
	fields := performanceExtractObject(payload)
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
