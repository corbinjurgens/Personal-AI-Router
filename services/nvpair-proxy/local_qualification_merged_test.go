// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"nvpair-shared/nodepolicy"
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

func TestRequestOptionsQualificationEffectiveAccounting(t *testing.T) {
	q := testQualification("enforced", "ollama", "m")
	q.ContextTokens, q.MaxOutputTokens = 64, 32
	q.RequestOptions = json.RawMessage(`{"messages":[{"role":"user","content":"abc"}],"temperature":0.1,"top_k":0,"min_p":0.02,"repeat_penalty":1.1}`)
	q.RequestOptionsMode = nodepolicy.RequestOptionsEnforced
	profile := nodepolicy.Profile{Engine: "ollama", Model: "m", ContextTokens: 64, MaxOutputTokens: 32, RequestOptions: q.RequestOptions, RequestOptionsMode: q.RequestOptionsMode}
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("x", 200) + `"}],"max_tokens":16,"temperature":0.8}`)
	effective, bound, err := q.validateRequest(profile, routeChatCompletions, body)
	if err != nil {
		t.Fatalf("effective request rejected: %v", err)
	}
	if bound.input != 19 || bound.output != 16 {
		t.Fatalf("bound=%+v, want effective 19 input and unchanged 16 output", bound)
	}
	var got struct {
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
		Temperature float64 `json:"temperature"`
		TopK        int     `json:"top_k"`
	}
	if err := json.Unmarshal(effective, &got); err != nil {
		t.Fatalf("effective JSON: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != "abc" || got.Temperature != 0.1 || got.TopK != 0 {
		t.Fatalf("effective values=%+v", got)
	}
	profile.RequestOptionsMode = nodepolicy.RequestOptionsDefaults
	if _, _, err := q.validateRequest(profile, routeChatCompletions, body); err == nil {
		t.Fatal("defaults unexpectedly replaced the oversized caller messages")
	}
}

func TestRequestOptionsStrictAllKeysRemainUnique(t *testing.T) {
	q := testQualification("strict", "ollama", "m")
	profile := nodepolicy.Profile{Engine: "ollama", Model: "m", ContextTokens: 4096, MaxOutputTokens: 1024, RequestOptions: json.RawMessage(`{"temperature":0.1,"messages":[{"role":"user","content":"replacement"}]}`), RequestOptionsMode: nodepolicy.RequestOptionsEnforced}
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi","content":"hidden"}],"max_tokens":16}`)
	_, _, err := q.validateRequest(profile, routeChatCompletions, body)
	var invalid *strictQualificationError
	if !errors.As(err, &invalid) || invalid.reason != "the request repeats a key or is not JSON" {
		t.Fatalf("strict unconfigured duplicates accepted: %v", err)
	}
}
