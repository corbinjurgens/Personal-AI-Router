// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"nvpair-shared/nodepolicy"
)

// Synthetic evidence only. This counter is deliberately not real tokenizer or
// provider evidence and has no production registry/default installation.
func chatFixture() (*chatQualification, chatTarget) {
	now := time.Date(2026, 10, 19, 12, 0, 0, 0, time.UTC)
	q := &chatQualification{
		Ref: nodepolicy.QualificationRef{ID: "synthetic-chat", Revision: "1"}, Endpoint: "https://synthetic.invalid/v1/chat/completions", Model: "synthetic-alias", CanonicalModel: "synthetic-v1",
		AcceptedModels: map[string]string{"synthetic-alias": "synthetic-v1", "synthetic-v1": "synthetic-v1"}, CheckedAt: now.Add(-time.Hour), ValidUntil: now.Add(time.Hour), MaxAge: 24 * time.Hour, Now: func() time.Time { return now }, Nonreasoning: true,
		AccountingProof: "synthetic byte-tokenizer/template test proof only", Tokenizer: "synthetic-bytes-v1", Template: "synthetic-template-v1", InputBound: func(b []byte) (int64, error) { return int64(len(b)), nil },
		RequestOverhead: 11, MessageOverhead: 7, ToolOverhead: 5, SchemaOverhead: 3, ContextTokens: 200000, MaxOutputTokens: 1000, DefaultOutput: 100, OutputField: "max_tokens",
		Fields: map[string]bool{}, Roles: map[string]bool{"system": true, "developer": true, "user": true, "assistant": true, "tool": true}, Formats: map[string]bool{"text": true, "json_object": true, "json_schema": true}, FinishReasons: map[string]bool{"stop": true, "length": true, "tool_calls": true},
		ValidateSchema: func(json.RawMessage) error { return nil }, Usage: true, TerminalRule: "finish-then-done", MaxBodyBytes: 65536, MaxLineBytes: 4096, MaxEventBytes: 8192, MaxToolBytes: 8192, MaxToolCalls: 8,
	}
	for _, k := range chatRequestKeys {
		q.Fields[k] = true
	}
	q.Fields["parallel_tool_calls"] = true
	target := chatTarget{Protocol: nodepolicy.ProtocolOpenAIChatCompletions, Endpoint: q.Endpoint, Model: q.Model, Qualification: q.Ref, ContextTokens: q.ContextTokens, MaxOutputTokens: q.MaxOutputTokens}
	return q, target
}

type chatTestBudget struct {
	held, peak int64
	calls      int
	refuseAt   int
	limit      int64
}

func (b *chatTestBudget) acquire(n int64) (func(), error) {
	b.calls++
	if n <= 0 || b.refuseAt == b.calls || b.limit > 0 && n > b.limit-b.held {
		return nil, errors.New("synthetic capacity refusal")
	}
	b.held += n
	b.peak = max(b.peak, b.held)
	released := false
	return func() {
		if !released {
			b.held -= n
			released = true
		}
	}, nil
}

const chatSimple = `{"model":"synthetic-alias","messages":[{"role":"user","content":"日本語 English 😀"}],"max_tokens":100}`
const chatHistory = `{"model":"synthetic-alias","messages":[{"role":"system","content":"日本語"},{"role":"developer","content":"English 😀"},{"role":"user","name":"owner","content":"two tools"},{"role":"assistant","content":null,"tool_calls":[{"id":"call-a","type":"function","function":{"name":"first","arguments":"{\"city\":\"東京\"}"}},{"id":"call-b","type":"function","function":{"name":"second","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-b","content":"結果 😀"},{"role":"tool","tool_call_id":"call-a","content":"English"},{"role":"assistant","content":"complete"}],"tools":[{"type":"function","function":{"name":"first","description":"日本語 😀","parameters":{"type":"object","properties":{"type":{"type":"string"},"$ref":{"type":"string"}},"required":["type"],"additionalProperties":false},"strict":true}},{"type":"function","function":{"name":"second","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"first"}},"parallel_tool_calls":true,"response_format":{"type":"json_schema","json_schema":{"name":"answer","description":"English 日本語 😀","schema":{"type":"object","properties":{"value":{"type":"string"}}},"strict":true}},"max_tokens":100}`

func chatPrepare(t *testing.T, q *chatQualification, target chatTarget, body string) (*chatPrepared, *chatTestBudget, error) {
	t.Helper()
	b := &chatTestBudget{}
	p, e := q.prepare(context.Background(), target, http.MethodPost, routeChatCompletions, []byte(body), b.acquire)
	t.Cleanup(func() {
		if p != nil {
			p.close()
		}
		if b.held != 0 {
			t.Errorf("leaked request bytes: %d", b.held)
		}
	})
	return p, b, e
}

func TestChatRequest(t *testing.T) {
	t.Run("qualified scalars and streaming options", func(t *testing.T) {
		q, target := chatFixture()
		body := strings.TrimSuffix(chatSimple, "}") + `,"stream":true,"stream_options":{"include_usage":true},"temperature":1,"top_p":0.9,"stop":["stop","終了"],"seed":-2,"frequency_penalty":0.5,"presence_penalty":-1}`
		p, _, e := chatPrepare(t, q, target, body)
		if e != nil || !p.Stream {
			t.Fatalf("qualified scalars failed: %v", e)
		}
	})
	t.Run("preserves final defaults and multilingual text", func(t *testing.T) {
		q, target := chatFixture()
		q.Defaults = json.RawMessage(`{"temperature":0.5,"max_tokens":77}`)
		p, _, e := chatPrepare(t, q, target, strings.Replace(chatSimple, `,"max_tokens":100`, "", 1))
		if e != nil {
			t.Fatal(e)
		}
		var f struct {
			Temperature float64 `json:"temperature"`
			MaxTokens   int64   `json:"max_tokens"`
			Messages    []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if e := json.Unmarshal(p.Body, &f); e != nil {
			t.Fatal(e)
		}
		if p.Output != 77 || f.MaxTokens != 77 || f.Temperature != 0.5 || f.Messages[0].Content != "日本語 English 😀" {
			t.Fatalf("altered effective request: %+v", f)
		}
	})
	reject := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			q, target := chatFixture()
			calls := 0
			q.InputBound = func([]byte) (int64, error) { calls++; return 1, nil }
			p, _, e := chatPrepare(t, q, target, body)
			if e == nil || p != nil || calls != 0 {
				t.Fatalf("prohibited request reached counter: err=%v calls=%d", e, calls)
			}
		})
	}
	reject("duplicate", strings.Replace(chatSimple, `"model":`, `"model":"other","model":`, 1))
	reject("case variant duplicate", strings.Replace(chatSimple, `"model":`, `"Model":"other","model":`, 1))
	reject("case variant field", strings.Replace(chatSimple, `"model"`, `"Model"`, 1))
	reject("duplicate schema property", strings.Replace(chatHistory, `"properties":{"value":{"type":"string"}}`, `"properties":{"foo":{"type":"string"},"foo":{"type":"string"}}`, 1))
	reject("depth", strings.Replace(chatSimple, `"日本語 English 😀"`, strings.Repeat("[", 65)+`"x"`+strings.Repeat("]", 65), 1))
	deepSchema := strings.Repeat(`{"type":"array","items":`, 65) + `{"type":"string"}` + strings.Repeat("}", 65)
	reject("otherwise valid deep schema", strings.Replace(chatHistory, `"parameters":{"type":"object"}`, `"parameters":`+deepSchema, 1))
	reject("trailing JSON", chatSimple+` {}`)
	reject("malformed UTF8", strings.Replace(chatSimple, "English", string([]byte{0xff}), 1))
	reject("null object", "null")
	reject("array object", "[]")
	reject("empty messages", `{"model":"synthetic-alias","messages":[]}`)
	for _, field := range []string{`"stream":null`, `"stream":"true"`, `"temperature":null`, `"top_p":"one"`, `"seed":1.5`, `"stop":[null]`, `"stream_options":{"include_usage":null}`, `"max_tokens":null`} {
		reject(field, strings.Replace(chatSimple, `,"max_tokens":100`, ","+field, 1))
	}
	t.Run("qualification optional scalar intersection", func(t *testing.T) {
		q, target := chatFixture()
		delete(q.Fields, "temperature")
		_, _, e := chatPrepare(t, q, target, strings.Replace(chatSimple, `"max_tokens":100`, `"temperature":0.5,"max_tokens":100`, 1))
		if e == nil {
			t.Fatal("unqualified optional scalar accepted")
		}
	})
	t.Run("case-distinct schema properties are data", func(t *testing.T) {
		q, target := chatFixture()
		body := strings.Replace(chatHistory, `"properties":{"value":{"type":"string"}}`, `"properties":{"Foo":{"type":"string"},"foo":{"type":"string"}}`, 1)
		p, _, e := chatPrepare(t, q, target, body)
		if e != nil {
			t.Fatalf("case-distinct schema properties rejected: %v", e)
		}
		var f struct {
			ResponseFormat struct {
				JSONSchema struct {
					Schema struct {
						Properties map[string]json.RawMessage `json:"properties"`
					} `json:"schema"`
				} `json:"json_schema"`
			} `json:"response_format"`
		}
		if e := json.Unmarshal(p.Body, &f); e != nil {
			t.Fatal(e)
		}
		for _, name := range []string{"Foo", "foo"} {
			if _, ok := f.ResponseFormat.JSONSchema.Schema.Properties[name]; !ok {
				t.Fatalf("schema property %q lost", name)
			}
		}
	})
}

func TestChatUnsupported(t *testing.T) {
	for _, field := range []string{"unknown", "n", "store", "metadata", "user", "previous_response_id", "conversation", "audio", "modalities", "logprobs", "reasoning_effort", "max_context_tokens"} {
		t.Run(field, func(t *testing.T) {
			q, target := chatFixture()
			q.Fields[field] = true // A profile cannot widen the generic stateless grammar.
			calls := 0
			q.InputBound = func([]byte) (int64, error) { calls++; return 1, nil }
			_, _, e := chatPrepare(t, q, target, strings.TrimSuffix(chatSimple, "}")+`,"`+field+`":true}`)
			if e == nil || calls != 0 {
				t.Fatalf("unsupported field accepted/reached counter: %v %d", e, calls)
			}
		})
	}
	for _, route := range []string{"/v1/responses", "/v1/messages", "/v1/completions", "/v1/embeddings", "/api/chat", routeChatCompletions + "?"} {
		t.Run(route, func(t *testing.T) {
			q, target := chatFixture()
			b := &chatTestBudget{}
			p, e := q.prepare(context.Background(), target, http.MethodPost, route, []byte(chatSimple), b.acquire)
			if p != nil {
				p.close()
			}
			if e == nil || b.calls != 0 {
				t.Fatal("unsupported route allocated or accepted")
			}
		})
	}
	t.Run("protocol and method", func(t *testing.T) {
		for _, protocol := range []nodepolicy.Protocol{"responses", "messages", "native", nodepolicy.ProtocolOpenAIChatCompletions} {
			q, target := chatFixture()
			target.Protocol = protocol
			b := &chatTestBudget{}
			p, e := q.prepare(context.Background(), target, http.MethodGet, routeChatCompletions, []byte(chatSimple), b.acquire)
			if p != nil {
				p.close()
			}
			if e == nil {
				t.Fatal("unsupported protocol or method")
			}
		}
	})
}

func TestChatHistory(t *testing.T) {
	for _, format := range []string{"text", "json_object"} {
		t.Run("qualified format "+format, func(t *testing.T) {
			q, target := chatFixture()
			body := strings.TrimSuffix(chatSimple, "}") + `,"response_format":{"type":"` + format + `"}}`
			if _, _, e := chatPrepare(t, q, target, body); e != nil {
				t.Fatal(e)
			}
		})
	}
	for _, mode := range []string{"auto", "none", "required"} {
		t.Run("qualified tool choice "+mode, func(t *testing.T) {
			q, target := chatFixture()
			body := strings.Replace(chatHistory, `{"type":"function","function":{"name":"first"}}`, `"`+mode+`"`, 1)
			if _, _, e := chatPrepare(t, q, target, body); e != nil {
				t.Fatal(e)
			}
		})
	}
	t.Run("order and schema fidelity", func(t *testing.T) {
		q, target := chatFixture()
		p, _, e := chatPrepare(t, q, target, chatHistory)
		if e != nil {
			t.Fatal(e)
		}
		var want, got map[string]json.RawMessage
		if e := json.Unmarshal([]byte(chatHistory), &want); e != nil {
			t.Fatal(e)
		}
		if e := json.Unmarshal(p.Body, &got); e != nil {
			t.Fatal(e)
		}
		for key, raw := range want {
			var a, b any
			if e := json.Unmarshal(raw, &a); e != nil {
				t.Fatal(e)
			}
			if e := json.Unmarshal(got[key], &b); e != nil {
				t.Fatal(e)
			}
			if !reflect.DeepEqual(a, b) {
				t.Fatalf("changed supported field %s", key)
			}
		}
	})
	t.Run("case-distinct function arguments are data", func(t *testing.T) {
		q, target := chatFixture()
		body := strings.Replace(chatHistory, `"function":{"name":"second","arguments":"{}"}`, `"function":{"name":"second","arguments":"{\"Foo\":1,\"foo\":2}"}`, 1)
		p, _, e := chatPrepare(t, q, target, body)
		if e != nil {
			t.Fatalf("case-distinct function arguments rejected: %v", e)
		}
		var f struct {
			Messages []struct {
				ToolCalls []struct {
					Function struct {
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"messages"`
		}
		if e := json.Unmarshal(p.Body, &f); e != nil {
			t.Fatal(e)
		}
		if len(f.Messages) != 7 || len(f.Messages[3].ToolCalls) != 2 {
			t.Fatalf("changed message shape: %d messages", len(f.Messages))
		}
		var keys map[string]json.RawMessage
		args := f.Messages[3].ToolCalls[1].Function.Arguments
		if e := json.Unmarshal([]byte(args), &keys); e != nil {
			t.Fatal(e)
		}
		for _, name := range []string{"Foo", "foo"} {
			if _, ok := keys[name]; !ok {
				t.Fatalf("argument key %q lost in %s", name, args)
			}
		}
	})
	reject := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			q, target := chatFixture()
			q.InputBound = func([]byte) (int64, error) { t.Error("invalid history reached counter"); return 1, nil }
			_, _, e := chatPrepare(t, q, target, body)
			if e == nil {
				t.Fatal("invalid history accepted")
			}
		})
	}
	reject("wrong result ID", strings.Replace(chatHistory, `"tool_call_id":"call-b"`, `"tool_call_id":"unknown"`, 1))
	reject("orphan tool result", `{"model":"synthetic-alias","messages":[{"role":"tool","tool_call_id":"orphan","content":"result"}]}`)
	reject("duplicate call answered once", `{"model":"synthetic-alias","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"first","arguments":"{}"}},{"id":"a","type":"function","function":{"name":"second","arguments":"{}"}}]},{"role":"tool","tool_call_id":"a","content":"result"}]}`)
	reject("duplicate call ID", strings.Replace(chatHistory, `"id":"call-b"`, `"id":"call-a"`, 1))
	reject("duplicate result", strings.Replace(chatHistory, `"tool_call_id":"call-b"`, `"tool_call_id":"call-a"`, 1))
	reject("unresolved before turn", strings.Replace(chatHistory, `"role":"tool","tool_call_id":"call-b"`, `"role":"user","tool_call_id":"call-b"`, 1))
	reject("unresolved at end", `{"model":"synthetic-alias","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"x","type":"function","function":{"name":"first","arguments":"{}"}}]}]}`)
	reject("missing call ID", strings.Replace(chatHistory, `"id":"call-a",`, "", 1))
	reject("role-inappropriate calls", strings.Replace(chatHistory, `"role":"assistant","content":null`, `"role":"user","content":null`, 1))
	reject("unsupported role", strings.Replace(chatSimple, `"role":"user"`, `"role":"function"`, 1))
	reject("content array", strings.Replace(chatSimple, `"日本語 English 😀"`, `[{"type":"text","text":"x"}]`, 1))
	reject("null ordinary content", strings.Replace(chatSimple, `"日本語 English 😀"`, `null`, 1))
	reject("unknown nested call key", strings.Replace(chatHistory, `"id":"call-a"`, `"unknown":true,"id":"call-a"`, 1))
	reject("undeclared tool choice", strings.Replace(chatHistory, `"tool_choice":{"type":"function","function":{"name":"first"}}`, `"tool_choice":{"type":"function","function":{"name":"third"}}`, 1))
	reject("schema reference", strings.Replace(chatHistory, `"parameters":{"type":"object"}`, `"parameters":{"type":"object","$ref":"remote"}`, 1))
	reject("schema recursion", strings.Replace(chatHistory, `"schema":{"type":"object",`, `"schema":{"$recursiveRef":"#","type":"object",`, 1))
	reject("wrong strict type", strings.Replace(chatHistory, `"strict":true`, `"strict":"true"`, 1))
	reject("duplicate argument key", strings.Replace(chatHistory, `"function":{"name":"second","arguments":"{}"}`, `"function":{"name":"second","arguments":"{\"foo\":1,\"foo\":2}"}`, 1))
	t.Run("profile schema vocabulary", func(t *testing.T) {
		q, target := chatFixture()
		q.ValidateSchema = func(json.RawMessage) error { return errChat }
		_, _, e := chatPrepare(t, q, target, chatHistory)
		if e == nil {
			t.Fatal("schema bypassed supplied validator")
		}
	})
}

func TestChatAccounting(t *testing.T) {
	t.Run("full final body and explicit overhead", func(t *testing.T) {
		q, target := chatFixture()
		p, _, e := chatPrepare(t, q, target, chatHistory)
		if e != nil {
			t.Fatal(e)
		}
		want := int64(len(p.Body)) + 11 + 7*7 + 2*5 + 3*3
		if p.Input != want || p.Output != 100 {
			t.Fatalf("input/output %d/%d want %d/100", p.Input, p.Output, want)
		}
	})
	t.Run("exact context and next unit", func(t *testing.T) {
		q, target := chatFixture()
		p, _, e := chatPrepare(t, q, target, chatHistory)
		if e != nil {
			t.Fatal(e)
		}
		target.ContextTokens = p.Input + p.Output
		if _, _, e = chatPrepare(t, q, target, chatHistory); e != nil {
			t.Fatal(e)
		}
		target.ContextTokens--
		if _, _, e = chatPrepare(t, q, target, chatHistory); e == nil {
			t.Fatal("one-unit context overflow accepted")
		}
	})
	t.Run("qualified context ceiling", func(t *testing.T) {
		q, target := chatFixture()
		p, _, e := chatPrepare(t, q, target, chatSimple)
		if e != nil {
			t.Fatal(e)
		}
		q.ContextTokens = p.Input + p.Output - 1
		q.MaxOutputTokens = 100
		if _, _, e = chatPrepare(t, q, target, chatSimple); e == nil {
			t.Fatal("qualified context overflow")
		}
	})
	t.Run("output ceiling and finite default", func(t *testing.T) {
		q, target := chatFixture()
		target.MaxOutputTokens = 100
		if _, _, e := chatPrepare(t, q, target, chatSimple); e != nil {
			t.Fatal(e)
		}
		target.MaxOutputTokens = 99
		if _, _, e := chatPrepare(t, q, target, chatSimple); e == nil {
			t.Fatal("one-unit output overflow")
		}
		q.DefaultOutput = 0
		if _, _, e := chatPrepare(t, q, target, strings.Replace(chatSimple, `,"max_tokens":100`, "", 1)); e == nil {
			t.Fatal("unknown output accepted")
		}
	})
	t.Run("qualified output ceiling", func(t *testing.T) {
		q, target := chatFixture()
		q.MaxOutputTokens = 100
		if _, _, e := chatPrepare(t, q, target, chatSimple); e != nil {
			t.Fatal(e)
		}
		q.MaxOutputTokens--
		if _, _, e := chatPrepare(t, q, target, chatSimple); e == nil {
			t.Fatal("qualified one-unit output overflow")
		}
	})
	for _, body := range []string{strings.Replace(chatSimple, `"max_tokens":100`, `"max_tokens":100,"max_completion_tokens":101`, 1), strings.Replace(chatSimple, `"max_tokens":100`, `"max_tokens":0`, 1)} {
		t.Run("contradictory or zero output", func(t *testing.T) {
			q, target := chatFixture()
			if _, _, e := chatPrepare(t, q, target, body); e == nil {
				t.Fatal("bad output bound accepted")
			}
		})
	}
	for _, tc := range []struct {
		name   string
		change func(*chatQualification, *chatTarget)
	}{
		{"qualification missing", func(q *chatQualification, _ *chatTarget) { q.Ref = nodepolicy.QualificationRef{} }},
		{"unqualified target", func(q *chatQualification, target *chatTarget) {
			q.Ref = nodepolicy.QualificationRef{}
			target.Qualification = q.Ref
		}},
		{"revision mismatch", func(_ *chatQualification, target *chatTarget) { target.Qualification.Revision = "other" }},
		{"endpoint mismatch", func(_ *chatQualification, target *chatTarget) { target.Endpoint += "other" }},
		{"missing counter", func(q *chatQualification, _ *chatTarget) { q.InputBound = nil }},
		{"unproven billing", func(q *chatQualification, _ *chatTarget) { q.Nonreasoning = false }},
		{"unknown accounting", func(q *chatQualification, _ *chatTarget) { q.AccountingProof = "" }},
		{"missing time", func(q *chatQualification, _ *chatTarget) { q.CheckedAt = time.Time{} }},
		{"future evidence", func(q *chatQualification, _ *chatTarget) { q.CheckedAt = q.Now().Add(time.Minute) }},
		{"expired evidence", func(q *chatQualification, _ *chatTarget) { q.ValidUntil = q.Now() }},
		{"excessive max age", func(q *chatQualification, _ *chatTarget) { q.MaxAge = 7*24*time.Hour + time.Nanosecond }},
		{"age boundary", func(q *chatQualification, _ *chatTarget) { q.MaxAge = time.Hour }},
		{"reversed times", func(q *chatQualification, _ *chatTarget) { q.ValidUntil = q.CheckedAt }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q, target := chatFixture()
			tc.change(q, &target)
			b := &chatTestBudget{}
			p, e := q.prepare(context.Background(), target, http.MethodPost, routeChatCompletions, []byte(chatSimple), b.acquire)
			if p != nil {
				p.close()
			}
			if e == nil || b.calls != 0 {
				t.Fatalf("unqualified admission: %v allocations=%d", e, b.calls)
			}
		})
	}
	t.Run("advance across expiry requires requalification", func(t *testing.T) {
		q, target := chatFixture()
		now := q.Now()
		q.Now = func() time.Time { return now }
		if _, _, e := chatPrepare(t, q, target, chatSimple); e != nil {
			t.Fatal(e)
		}
		now = q.ValidUntil
		if _, _, e := chatPrepare(t, q, target, chatSimple); e == nil {
			t.Fatal("stale evidence accepted")
		}
		q.CheckedAt = now
		q.ValidUntil = now.Add(time.Hour)
		if _, _, e := chatPrepare(t, q, target, chatSimple); e != nil {
			t.Fatal(e)
		}
	})
	for _, n := range []int64{-1, math.MaxInt64} {
		t.Run("checked input arithmetic", func(t *testing.T) {
			q, target := chatFixture()
			q.InputBound = func([]byte) (int64, error) { return n, nil }
			if _, _, e := chatPrepare(t, q, target, chatHistory); e == nil {
				t.Fatal("unknown/overflow counter accepted")
			}
		})
	}
	t.Run("checked context addition", func(t *testing.T) {
		q, target := chatFixture()
		q.ContextTokens = math.MaxInt64
		target.ContextTokens = math.MaxInt64
		q.InputBound = func([]byte) (int64, error) { return math.MaxInt64 - 90, nil }
		if _, _, e := chatPrepare(t, q, target, chatSimple); e == nil {
			t.Fatal("input/output overflow accepted")
		}
	})
}

func chatResponse(model, content, finish, usage string, stream bool) string {
	object, key := "chat.completion", "message"
	if stream {
		object = "chat.completion.chunk"
		key = "delta"
	}
	if usage != "" {
		usage = `,"usage":` + usage
	}
	return `{"id":"synthetic-response","object":"` + object + `","created":1,"model":"` + model + `","choices":[{"index":0,"` + key + `":` + content + `,"finish_reason":` + finish + `}]` + usage + `}`
}

const chatUsageOK = `{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}`

func chatChunk(content, finish string) string {
	return chatResponse("synthetic-v1", content, finish, "", true)
}
func chatEvent(data string) string { return "data: " + data + "\n\n" }
func chatCompleteStream() string {
	return chatEvent(chatChunk(`{"role":"assistant"}`, `null`)) + chatEvent(chatChunk(`{"content":"日本語 😀"}`, `null`)) + chatEvent(chatChunk(`{}`, `"stop"`)) + chatEvent("[DONE]")
}
func chatUsageEvent(usage string) string {
	return `{"id":"synthetic-response","object":"chat.completion.chunk","created":1,"model":"synthetic-v1","choices":[],"usage":` + usage + `}`
}

func chatParse(t *testing.T, q *chatQualification, body string, stream bool) (chatParseResult, [][]byte, error) {
	t.Helper()
	b := &chatTestBudget{}
	var events [][]byte
	r, e := q.parseResponse(context.Background(), strings.NewReader(body), stream, 100, 100, q.Now().Add(-time.Second), b.acquire, func(raw []byte) error { events = append(events, bytes.Clone(raw)); return nil })
	if b.held != 0 {
		t.Fatalf("leaked response budget %d", b.held)
	}
	return r, events, e
}

func TestChatResponse(t *testing.T) {
	t.Run("exact successful body cap", func(t *testing.T) {
		q, _ := chatFixture()
		body := chatResponse("synthetic-v1", `{"role":"assistant","content":"English 日本語 😀"}`, `"stop"`, "", false)
		q.MaxBodyBytes = int64(len(body))
		q.MaxEventBytes = q.MaxBodyBytes
		q.MaxLineBytes = q.MaxBodyBytes
		q.MaxToolBytes = 1
		if _, _, e := chatParse(t, q, body, false); e != nil {
			t.Fatal(e)
		}
		q.MaxBodyBytes--
		q.MaxEventBytes--
		q.MaxLineBytes--
		if r, events, e := chatParse(t, q, body, false); e == nil || r.Complete || len(events) != 0 {
			t.Fatal("body cap bypassed")
		}
	})
	t.Run("complete multilingual JSON fidelity", func(t *testing.T) {
		q, _ := chatFixture()
		body := chatResponse("synthetic-v1", `{"role":"assistant","content":"English 日本語 😀"}`, `"stop"`, chatUsageOK, false)
		r, events, e := chatParse(t, q, body, false)
		if e != nil {
			t.Fatal(e)
		}
		if !r.Complete || r.Usage == nil || r.Usage.InputTokens != 10 || r.Usage.OutputTokens != 2 || len(events) != 1 || string(events[0]) != body || r.Terminal.IsZero() || r.FirstBody.IsZero() || r.FirstDelta.IsZero() || r.Header.IsZero() {
			t.Fatalf("wrong complete response %+v", r)
		}
	})
	for _, tc := range []struct{ name, body string }{
		{"truncated", `{"model":"synthetic-v1"`},
		{"no finish", chatResponse("synthetic-v1", `{"role":"assistant","content":"text"}`, `null`, "", false)},
		{"no text shape", chatResponse("synthetic-v1", `{"role":"assistant","content":[]}`, `"stop"`, "", false)},
		{"unsupported finish", chatResponse("synthetic-v1", `{"role":"assistant","content":"text"}`, `"content_filter"`, "", false)},
		{"extra root field", strings.Replace(chatResponse("synthetic-v1", `{"role":"assistant","content":"text"}`, `"stop"`, "", false), `"created":1`, `"created":1,"reasoning":"opaque"`, 1)},
		{"multiple choices", strings.Replace(chatResponse("synthetic-v1", `{"role":"assistant","content":"text"}`, `"stop"`, "", false), `"choices":[`, `"choices":[{},`, 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q, _ := chatFixture()
			r, events, e := chatParse(t, q, tc.body, false)
			if e == nil || r.Complete || len(events) != 0 || r.Usage != nil {
				t.Fatalf("invalid JSON exposed: %+v err=%v events=%d", r, e, len(events))
			}
		})
	}
	t.Run("function response", func(t *testing.T) {
		q, _ := chatFixture()
		r, _, e := chatParse(t, q, chatResponse("synthetic-v1", `{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"first","arguments":"{\"x\":\"東京\"}"}}]}`, `"tool_calls"`, "", false), false)
		if e != nil || !r.Complete {
			t.Fatalf("tool completion: %+v %v", r, e)
		}
	})
	t.Run("case-distinct response arguments are data", func(t *testing.T) {
		q, _ := chatFixture()
		body := chatResponse("synthetic-v1", `{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"first","arguments":"{\"Foo\":1,\"foo\":2}"}}]}`, `"tool_calls"`, "", false)
		r, _, e := chatParse(t, q, body, false)
		if e != nil || !r.Complete {
			t.Fatalf("case-distinct response arguments rejected: %+v %v", r, e)
		}
	})
	t.Run("duplicate response argument key", func(t *testing.T) {
		q, _ := chatFixture()
		body := chatResponse("synthetic-v1", `{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"first","arguments":"{\"foo\":1,\"foo\":2}"}}]}`, `"tool_calls"`, "", false)
		r, events, e := chatParse(t, q, body, false)
		if e == nil || r.Complete || len(events) != 0 {
			t.Fatalf("exact duplicate response argument accepted: %+v %v", r, e)
		}
	})
}

func TestChatSSE(t *testing.T) {
	t.Run("exact event line and total bounds", func(t *testing.T) {
		q, _ := chatFixture()
		event := chatEvent(chatChunk(`{"content":"日本語 😀"}`, `"stop"`))
		body := event + chatEvent("[DONE]")
		q.MaxEventBytes = int64(len(event))
		q.MaxLineBytes = int64(len(event) - 2)
		q.MaxBodyBytes = int64(len(body))
		q.MaxToolBytes = 1
		if r, _, e := chatParse(t, q, body, true); e != nil || !r.Complete {
			t.Fatalf("exact bounds refused %+v %v", r, e)
		}
		q.MaxLineBytes--
		if _, _, e := chatParse(t, q, body, true); e == nil {
			t.Fatal("line one byte beyond cap accepted")
		}
		q.MaxLineBytes++
		q.MaxBodyBytes--
		if _, _, e := chatParse(t, q, body, true); e == nil {
			t.Fatal("total one byte beyond cap accepted")
		}
		q.MaxBodyBytes++
		q.MaxEventBytes--
		q.MaxLineBytes = min(q.MaxLineBytes, q.MaxEventBytes)
		if _, _, e := chatParse(t, q, body, true); e == nil {
			t.Fatal("event one byte beyond cap accepted")
		}
	})
	t.Run("metadata before text CRLF split UTF8 and events", func(t *testing.T) {
		q, _ := chatFixture()
		clock := q.Now()
		q.Now = func() time.Time { clock = clock.Add(time.Millisecond); return clock }
		body := strings.ReplaceAll(chatCompleteStream(), "\n", "\r\n")
		r, events, e := chatParse(t, q, body, true)
		if e != nil {
			t.Fatal(e)
		}
		if !r.Complete || !r.Exposed || r.Usage != nil || len(events) != 4 || string(bytes.Join(events, nil)) != body || !r.FirstDelta.After(r.FirstBody) || !r.Terminal.After(r.FirstDelta) {
			t.Fatalf("stream milestones/fidelity %+v events=%d", r, len(events))
		}
	})
	t.Run("multiline data and comments", func(t *testing.T) {
		q, _ := chatFixture()
		body := ": bounded comment\n\n" + chatEvent(strings.Replace(chatChunk(`{"content":"x"}`, `"stop"`), `,"choices"`, ",\ndata: \"choices\"", 1)) + chatEvent("[DONE]")
		r, _, e := chatParse(t, q, body, true)
		if e != nil || !r.Complete {
			t.Fatalf("multiline %v %+v", e, r)
		}
	})
	for _, tc := range []struct{ name, body string }{
		{"early EOF", chatEvent(chatChunk(`{"content":"x"}`, `null`))},
		{"finish without terminal", chatEvent(chatChunk(`{"content":"x"}`, `"stop"`))},
		{"terminal without finish", chatEvent("[DONE]")},
		{"unterminated terminal", chatEvent(chatChunk(`{"content":"x"}`, `"stop"`)) + "data: [DONE]"},
		{"extra data", chatCompleteStream() + chatEvent(chatChunk(`{}`, `null`))},
		{"text after finish", chatEvent(chatChunk(`{"content":"x"}`, `"stop"`)) + chatEvent(chatChunk(`{"content":"y"}`, `null`)) + chatEvent("[DONE]")},
		{"stateful SSE", "id: remembered\n\n" + chatCompleteStream()},
		{"oversized line", ":" + strings.Repeat("x", 4097) + "\n\n" + chatCompleteStream()},
		{"oversized event", strings.Repeat(":"+strings.Repeat("x", 4000)+"\n", 3) + "\n" + chatCompleteStream()},
		{"endless comments", strings.Repeat(":x\n\n", 18000) + chatCompleteStream()},
		{"malformed UTF8 comment", ":" + string([]byte{0xff}) + "\n\n" + chatCompleteStream()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q, _ := chatFixture()
			r, _, e := chatParse(t, q, tc.body, true)
			if e == nil || r.Complete || r.Usage != nil {
				t.Fatalf("invalid stream complete %+v %v", r, e)
			}
		})
	}
	t.Run("fragmented functions", func(t *testing.T) {
		q, _ := chatFixture()
		q.MaxToolBytes = int64(len("a") + len("first") + len(`{"x":"東京😀"}`))
		body := chatEvent(chatChunk(`{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"first","arguments":"{\"x\":"}}]}`, `null`)) + chatEvent(chatChunk(`{"tool_calls":[{"index":0,"function":{"arguments":"\"東京😀\"}"}}]}`, `null`)) + chatEvent(chatChunk(`{}`, `"tool_calls"`)) + chatEvent("[DONE]")
		r, events, e := chatParse(t, q, body, true)
		if e != nil || !r.Complete || string(bytes.Join(events, nil)) != body {
			t.Fatalf("fragmented functions %+v %v", r, e)
		}
		q.MaxToolBytes--
		r, _, e = chatParse(t, q, body, true)
		if e == nil || r.Complete {
			t.Fatal("tool accumulator exceeded cap")
		}
	})
}

func TestChatIdentity(t *testing.T) {
	for _, model := range []string{"synthetic-v1", "synthetic-alias"} {
		t.Run("explicit mapping "+model, func(t *testing.T) {
			q, _ := chatFixture()
			r, _, e := chatParse(t, q, chatResponse(model, `{"role":"assistant","content":"x"}`, `"stop"`, "", false), false)
			if e != nil || !r.Complete {
				t.Fatalf("mapped model %v", e)
			}
		})
	}
	for _, model := range []string{"", "wrong-model"} {
		t.Run("invalid model "+model, func(t *testing.T) {
			q, _ := chatFixture()
			body := chatResponse(model, `{"role":"assistant","content":"x"}`, `"stop"`, chatUsageOK, false)
			r, events, e := chatParse(t, q, body, false)
			if !errors.Is(e, errChatIdentity) || r.Complete || r.Usage != nil || len(events) != 0 {
				t.Fatalf("identity did not fail before exposure %+v %v", r, e)
			}
		})
	}
	for _, tc := range []struct{ name, bad string }{
		{"missing", strings.Replace(chatChunk(`{"content":"bad"}`, `null`), `"model":"synthetic-v1",`, "", 1)},
		{"mixed text", strings.Replace(chatChunk(`{"content":"bad"}`, `null`), "synthetic-v1", "wrong-model", 1)},
		{"mixed accepted reported IDs", strings.Replace(chatChunk(`{"content":"bad"}`, `null`), "synthetic-v1", "synthetic-alias", 1)},
		{"finish", strings.Replace(chatChunk(`{}`, `"stop"`), "synthetic-v1", "wrong-model", 1)},
		{"usage", strings.Replace(chatUsageEvent(chatUsageOK), "synthetic-v1", "wrong-model", 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q, _ := chatFixture()
			prefix := chatEvent(chatChunk(`{"role":"assistant"}`, `null`))
			if tc.name == "usage" {
				prefix = chatEvent(chatChunk(`{}`, `"stop"`))
			}
			r, events, e := chatParse(t, q, prefix+chatEvent(tc.bad)+chatEvent("[DONE]"), true)
			if !errors.Is(e, errChatIdentity) || !r.Exposed || r.Complete || r.Usage != nil || len(events) != 1 {
				t.Fatalf("offending chunk exposed %+v %v events=%d", r, e, len(events))
			}
		})
	}
}

func TestChatUsage(t *testing.T) {
	t.Run("missing remains unknown", func(t *testing.T) {
		q, _ := chatFixture()
		for _, usage := range []string{"", "null"} {
			r, _, e := chatParse(t, q, chatResponse("synthetic-v1", `{"role":"assistant","content":"x"}`, `"stop"`, usage, false), false)
			if e != nil || !r.Complete || r.Usage != nil {
				t.Fatalf("missing usage %+v %v", r, e)
			}
		}
	})
	for _, usage := range []string{`{}`, `{"prompt_tokens":-1,"completion_tokens":2,"total_tokens":1}`, `{"prompt_tokens":1.5,"completion_tokens":2,"total_tokens":3}`, `{"prompt_tokens":9223372036854775808,"completion_tokens":2,"total_tokens":3}`, `{"prompt_tokens":null,"completion_tokens":2,"total_tokens":2}`, `{"prompt_tokens":1,"completion_tokens":2,"total_tokens":4}`, `{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,"completion_tokens_details":{"reasoning_tokens":1}}`, `{"prompt_tokens":9223372036854775807,"completion_tokens":1,"total_tokens":0}`} {
		t.Run(usage, func(t *testing.T) {
			q, _ := chatFixture()
			r, events, e := chatParse(t, q, chatResponse("synthetic-v1", `{"role":"assistant","content":"x"}`, `"stop"`, usage, false), false)
			if e == nil || r.Complete || r.Usage != nil || len(events) != 0 {
				t.Fatalf("malformed usage trusted %+v %v", r, e)
			}
		})
	}
	t.Run("usage after finish and repeated identical", func(t *testing.T) {
		q, _ := chatFixture()
		prefix := chatEvent(chatChunk(`{"content":"x"}`, `"stop"`))
		body := prefix + chatEvent(chatUsageEvent(chatUsageOK)) + chatEvent(chatUsageEvent(chatUsageOK)) + chatEvent("[DONE]")
		r, _, e := chatParse(t, q, body, true)
		if e != nil || !r.Complete || r.Usage == nil || r.Usage.InputTokens != 10 {
			t.Fatalf("complete usage %+v %v", r, e)
		}
		body = prefix + chatEvent(chatUsageEvent(chatUsageOK)) + chatEvent(chatUsageEvent(`{"prompt_tokens":11,"completion_tokens":2,"total_tokens":13}`)) + chatEvent("[DONE]")
		r, _, e = chatParse(t, q, body, true)
		if e == nil || r.Usage != nil {
			t.Fatal("conflicting usage trusted")
		}
	})
	t.Run("partial usage cannot reconcile", func(t *testing.T) {
		q, _ := chatFixture()
		r, _, e := chatParse(t, q, chatEvent(chatChunk(`{}`, `"stop"`))+chatEvent(chatUsageEvent(chatUsageOK)), true)
		if e == nil || r.Usage != nil || r.Complete {
			t.Fatal("partial usage trusted")
		}
	})
	t.Run("usage before finish untrusted", func(t *testing.T) {
		q, _ := chatFixture()
		r, _, e := chatParse(t, q, chatEvent(chatResponse("synthetic-v1", `{"content":"x"}`, `null`, chatUsageOK, true))+chatEvent(chatChunk(`{}`, `"stop"`))+chatEvent("[DONE]"), true)
		if e == nil || r.Usage != nil {
			t.Fatal("early usage trusted")
		}
	})
	t.Run("trusted above bound classification", func(t *testing.T) {
		q, _ := chatFixture()
		r, events, e := chatParse(t, q, chatResponse("synthetic-v1", `{"role":"assistant","content":"x"}`, `"stop"`, `{"prompt_tokens":101,"completion_tokens":2,"total_tokens":103}`, false), false)
		if !errors.Is(e, errChatBound) || !r.Complete || !r.AboveBound || r.Usage == nil || len(events) != 0 {
			t.Fatalf("lost trusted bound violation %+v %v", r, e)
		}
	})
	t.Run("stream bound violation waits for trusted completion", func(t *testing.T) {
		q, _ := chatFixture()
		prefix := chatEvent(chatChunk(`{"content":"x"}`, `"stop"`))
		partial := prefix + chatEvent(chatUsageEvent(`{"prompt_tokens":101,"completion_tokens":2,"total_tokens":103}`))
		r, events, e := chatParse(t, q, partial+chatEvent("[DONE]"), true)
		if !errors.Is(e, errChatBound) || !r.Complete || !r.AboveBound || r.Usage == nil || len(events) != 1 {
			t.Fatalf("stream bound classification %+v %v events=%d", r, e, len(events))
		}
		r, _, e = chatParse(t, q, partial, true)
		if e == nil || r.Complete || r.Usage != nil || r.AboveBound {
			t.Fatal("incomplete bound violation trusted")
		}
	})
	t.Run("usage requires qualification", func(t *testing.T) {
		q, _ := chatFixture()
		q.Usage = false
		r, _, e := chatParse(t, q, chatResponse("synthetic-v1", `{"role":"assistant","content":"x"}`, `"stop"`, chatUsageOK, false), false)
		if e == nil || r.Usage != nil {
			t.Fatal("unqualified usage accepted")
		}
	})
}

type chatCancelReader struct {
	ctx    context.Context
	cancel context.CancelFunc
	reads  int
}

func (r *chatCancelReader) Read(p []byte) (int, error) {
	r.reads++
	if r.reads == 4 {
		r.cancel()
	}
	if r.ctx.Err() != nil {
		return 0, r.ctx.Err()
	}
	p[0] = ':'
	return 1, nil
}

func TestChatBudget(t *testing.T) {
	t.Run("request refusal precedes counter", func(t *testing.T) {
		q, target := chatFixture()
		q.InputBound = func([]byte) (int64, error) { t.Error("counter called after refusal"); return 0, nil }
		b := &chatTestBudget{refuseAt: 1}
		p, e := q.prepare(context.Background(), target, http.MethodPost, routeChatCompletions, []byte(chatSimple), b.acquire)
		if p != nil {
			p.close()
		}
		if !errors.Is(e, errChatBudget) || b.held != 0 {
			t.Fatalf("budget refusal %v held=%d", e, b.held)
		}
	})
	t.Run("missing parser hook", func(t *testing.T) {
		q, _ := chatFixture()
		r, e := q.parseResponse(context.Background(), strings.NewReader(chatCompleteStream()), true, 100, 100, time.Time{}, nil, func([]byte) error { t.Error("unbudgeted exposure"); return nil })
		if e == nil || r.Complete {
			t.Fatal("missing budget accepted")
		}
	})
	t.Run("response refusal before read", func(t *testing.T) {
		q, _ := chatFixture()
		b := &chatTestBudget{refuseAt: 1}
		reader := strings.NewReader(chatCompleteStream())
		r, e := q.parseResponse(context.Background(), reader, true, 100, 100, time.Time{}, b.acquire, func([]byte) error { return nil })
		if e == nil || r.Complete || reader.Len() != len(chatCompleteStream()) || b.held != 0 {
			t.Fatal("read occurred after budget refusal")
		}
	})
	t.Run("idempotent retained request lease", func(t *testing.T) {
		q, target := chatFixture()
		b := &chatTestBudget{}
		p, e := q.prepare(context.Background(), target, http.MethodPost, routeChatCompletions, []byte(chatSimple), b.acquire)
		if e != nil {
			t.Fatal(e)
		}
		if b.held <= 0 {
			t.Fatal("request has no retained lease")
		}
		p.close()
		p.close()
		if b.held != 0 {
			t.Fatal("lease did not release exactly once")
		}
	})
	t.Run("cancellation", func(t *testing.T) {
		q, _ := chatFixture()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		reader := &chatCancelReader{ctx: ctx, cancel: cancel}
		b := &chatTestBudget{}
		r, e := q.parseResponse(ctx, reader, true, 100, 100, time.Time{}, b.acquire, func([]byte) error { return nil })
		if e == nil || r.Complete || b.held != 0 || reader.reads != 4 {
			t.Fatalf("cancel result %+v %v held=%d reads=%d", r, e, b.held, reader.reads)
		}
	})
	t.Run("sink error retains exposure and clears usage", func(t *testing.T) {
		q, _ := chatFixture()
		b := &chatTestBudget{}
		r, e := q.parseResponse(context.Background(), strings.NewReader(chatCompleteStream()), true, 100, 100, time.Time{}, b.acquire, func([]byte) error { return io.ErrClosedPipe })
		if e == nil || !r.Exposed || r.Complete || r.Usage != nil || b.held != 0 {
			t.Fatalf("sink failure %+v %v", r, e)
		}
	})
	t.Run("parser panic releases", func(t *testing.T) {
		q, _ := chatFixture()
		b := &chatTestBudget{}
		func() {
			defer func() {
				if recover() == nil {
					t.Error("panic was swallowed")
				}
			}()
			_, _ = q.parseResponse(context.Background(), strings.NewReader(chatCompleteStream()), true, 100, 100, time.Time{}, b.acquire, func([]byte) error { panic("synthetic panic") })
		}()
		if b.held != 0 {
			t.Fatalf("panic leaked %d", b.held)
		}
	})
	t.Run("request counter panic releases", func(t *testing.T) {
		q, target := chatFixture()
		b := &chatTestBudget{}
		q.InputBound = func([]byte) (int64, error) { panic("synthetic panic") }
		func() {
			defer func() {
				if recover() == nil {
					t.Error("panic swallowed")
				}
			}()
			_, _ = q.prepare(context.Background(), target, http.MethodPost, routeChatCompletions, []byte(chatSimple), b.acquire)
		}()
		if b.held != 0 {
			t.Fatal("request panic leaked")
		}
	})
	t.Run("simultaneous request response headroom", func(t *testing.T) {
		q, target := chatFixture()
		b := &chatTestBudget{}
		p, e := q.prepare(context.Background(), target, http.MethodPost, routeChatCompletions, []byte(chatSimple), b.acquire)
		if e != nil {
			t.Fatal(e)
		}
		defer p.close()
		held := b.held
		b.limit = held + 1
		r, e := q.parseResponse(context.Background(), strings.NewReader(chatCompleteStream()), true, p.Input, p.Output, time.Time{}, b.acquire, func([]byte) error { return nil })
		if e == nil || r.Complete || b.held != held {
			t.Fatal("parser exceeded simultaneous allowance")
		}
		p.close()
		if b.held != 0 {
			t.Fatal("retained allowance leaked")
		}
	})
	t.Run("finite budget accepts tiny nonstream response", func(t *testing.T) {
		q, _ := chatFixture()
		q.MaxBodyBytes = 4 << 20
		body := chatResponse("synthetic-v1", `{"role":"assistant","content":"small"}`, `"stop"`, "", false)
		b := &chatTestBudget{limit: 64 << 20}
		r, e := q.parseResponse(context.Background(), strings.NewReader(body), false, 100, 100, time.Time{}, b.acquire, func([]byte) error { return nil })
		if e != nil || !r.Complete || b.held != 0 {
			t.Fatalf("tiny response refused by a 64 MiB budget: complete=%t err=%v held=%d", r.Complete, e, b.held)
		}
	})
	t.Run("finite budget accepts small stream", func(t *testing.T) {
		q, _ := chatFixture()
		b := &chatTestBudget{limit: 1 << 20}
		r, e := q.parseResponse(context.Background(), strings.NewReader(chatCompleteStream()), true, 100, 100, time.Time{}, b.acquire, func([]byte) error { return nil })
		if e != nil || !r.Complete || b.held != 0 {
			t.Fatalf("small stream refused by a 1 MiB budget: complete=%t err=%v held=%d", r.Complete, e, b.held)
		}
	})
	t.Run("near-limit body with retained request lease", func(t *testing.T) {
		q, target := chatFixture()
		b := &chatTestBudget{limit: 64 << 20}
		p, e := q.prepare(context.Background(), target, http.MethodPost, routeChatCompletions, []byte(chatSimple), b.acquire)
		if e != nil {
			t.Fatal(e)
		}
		defer func() {
			p.close()
			if b.held != 0 {
				t.Errorf("leaked budget %d", b.held)
			}
		}()
		retained := b.held
		content := strings.Repeat("a", 3<<20)
		body := chatResponse("synthetic-v1", `{"role":"assistant","content":"`+content+`"}`, `"stop"`, "", false)
		q.MaxBodyBytes = int64(len(body))
		var exposed []byte
		r, e := q.parseResponse(context.Background(), strings.NewReader(body), false, p.Input, p.Output, time.Time{}, b.acquire, func(raw []byte) error { exposed = raw; return nil })
		if e != nil || !r.Complete || len(exposed) != len(body) {
			t.Fatalf("near-limit body refused: complete=%t err=%v exposed=%d want=%d", r.Complete, e, len(exposed), len(body))
		}
		if b.held != retained {
			t.Fatalf("parser did not release: held=%d want=%d", b.held, retained)
		}
	})
	t.Run("over budget still refuses before read", func(t *testing.T) {
		q, _ := chatFixture()
		q.MaxBodyBytes = 4 << 20
		body := chatResponse("synthetic-v1", `{"role":"assistant","content":"small"}`, `"stop"`, "", false)
		reader := strings.NewReader(body)
		b := &chatTestBudget{limit: 64 << 10}
		r, e := q.parseResponse(context.Background(), reader, false, 100, 100, time.Time{}, b.acquire, func([]byte) error { return nil })
		if !errors.Is(e, errChatBudget) || r.Complete || reader.Len() != len(body) || b.held != 0 {
			t.Fatalf("over-budget parse: err=%v complete=%t unread=%d held=%d", e, r.Complete, reader.Len(), b.held)
		}
	})
}
