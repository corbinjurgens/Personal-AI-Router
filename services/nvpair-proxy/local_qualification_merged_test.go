// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestQualificationMergedRejectsCaseVariantDefaultDuplicate(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	profile.RequestOptions = json.RawMessage(`{"stream_options":{"include_usage":false}}`)
	q.RequestOptions = profile.RequestOptions
	body := []byte(`{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream_options":{"Include_usage":true}}`)

	_, _, err := q.validateRequest(profile, routeChatCompletions, body)
	var rejected *strictQualificationError
	if err == nil || !errors.As(err, &rejected) {
		t.Fatalf("error = %v, want a strict qualification rejection", err)
	}
	if rejected.reason != "the request with its defaults repeats a key or is not JSON" {
		t.Fatalf("rejection reason = %q, want effective-body duplicate validation", rejected.reason)
	}
}

func TestQualificationMergedCanonicalOverrideKeepsCallerValue(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	profile.RequestOptions = json.RawMessage(`{"stream_options":{"include_usage":false}}`)
	q.RequestOptions = profile.RequestOptions
	body := []byte(`{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream_options":{"include_usage":true}}`)

	effective, _, err := q.validateRequest(profile, routeChatCompletions, body)
	if err != nil {
		t.Fatalf("request rejected: %v", err)
	}
	assertMergedIncludeUsage(t, effective, true)
}

func TestQualificationMergedDefaultsOnlyKeepsDefaultValue(t *testing.T) {
	q, profile, _ := qualificationFixture(t)
	profile.RequestOptions = json.RawMessage(`{"stream_options":{"include_usage":false}}`)
	q.RequestOptions = profile.RequestOptions
	body := []byte(`{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"max_tokens":16}`)

	effective, _, err := q.validateRequest(profile, routeChatCompletions, body)
	if err != nil {
		t.Fatalf("request rejected: %v", err)
	}
	assertMergedIncludeUsage(t, effective, false)
}

func assertMergedIncludeUsage(t *testing.T, body []byte, want bool) {
	t.Helper()
	var effective struct {
		StreamOptions struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if err := json.Unmarshal(body, &effective); err != nil {
		t.Fatalf("decode effective body: %v", err)
	}
	if effective.StreamOptions.IncludeUsage != want {
		t.Fatalf("effective include_usage = %t, want %t: %s", effective.StreamOptions.IncludeUsage, want, body)
	}
}
