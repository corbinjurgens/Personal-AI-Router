// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// This is an unwired, single-attempt Chat library. F supplies real qualification
// evidence; I must supply the process-wide budget and dispatch owner. Nothing here
// selects a provider, resolves credentials, reserves money, or performs transport.
import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"slices"
	"sync"
	"time"
	"unicode/utf8"

	"nvpair-shared/nodepolicy"
)

// chatByteBudget must acquire without waiting and return an idempotent release.
// It deliberately does not depend on the public transport's private types.
type chatByteBudget func(int64) (func(), error)

type chatTarget struct {
	Protocol                       nodepolicy.Protocol
	Endpoint, Model                string
	Qualification                  nodepolicy.QualificationRef
	ContextTokens, MaxOutputTokens int64
}

// All callbacks are trusted, in-memory qualification inputs, never policy data.
// InputBound covers the ENTIRE final serialized body (including UTF-8, roles,
// names, call IDs, arguments/results, tools and schemas). Separate overheads cover
// tokenizer/template framing not represented by those bytes. Their proof must
// name the tokenizer and template revisions; localQualification is not evidence.
type chatQualification struct {
	Ref                                                            nodepolicy.QualificationRef
	Endpoint, Model, CanonicalModel                                string
	AcceptedModels                                                 map[string]string
	CheckedAt, ValidUntil                                          time.Time
	MaxAge                                                         time.Duration
	Now                                                            func() time.Time
	Nonreasoning                                                   bool
	AccountingProof, Tokenizer, Template                           string
	InputBound                                                     func([]byte) (int64, error)
	RequestOverhead, MessageOverhead, ToolOverhead, SchemaOverhead int64
	ContextTokens, MaxOutputTokens, DefaultOutput                  int64
	OutputField                                                    string
	Defaults                                                       json.RawMessage
	Fields, Roles, Formats, FinishReasons                          map[string]bool
	ValidateSchema                                                 func(json.RawMessage) error
	Usage                                                          bool
	TerminalRule                                                   string
	MaxBodyBytes, MaxLineBytes, MaxEventBytes, MaxToolBytes        int64
	MaxToolCalls                                                   int
}

type chatPrepared struct {
	Body          []byte
	Input, Output int64
	Stream        bool
	release       func()
}

func (p *chatPrepared) close() {
	if p != nil && p.release != nil {
		p.release()
	}
}

var errChat = errors.New("qualified Chat validation failed")
var errChatIdentity = errors.New("qualified Chat identity violation")
var errChatBound = errors.New("qualified Chat usage bound violation")
var errChatBudget = errors.New("qualified Chat byte budget unavailable")

func chatAcquire(b chatByteBudget, n int64) (func(), error) {
	if b == nil || n <= 0 {
		return nil, errChatBudget
	}
	r, err := b(n)
	if err != nil || r == nil {
		if r != nil {
			r()
		}
		return nil, errChatBudget
	}
	return sync.OnceFunc(r), nil
}

func chatAdd(a, b int64) (int64, error) {
	if a < 0 || b < 0 || a > math.MaxInt64-b {
		return 0, errChat
	}
	return a + b, nil
}

func (q *chatQualification) fresh() error {
	if q == nil || q.Now == nil || q.Ref.Validate() != nil || !q.Nonreasoning ||
		q.AccountingProof == "" || q.Tokenizer == "" || q.Template == "" || q.InputBound == nil ||
		q.Endpoint == "" || q.Model == "" || q.CanonicalModel == "" || len(q.AcceptedModels) == 0 {
		return errChat
	}
	now := q.Now()
	if q.CheckedAt.IsZero() || q.ValidUntil.IsZero() || !q.CheckedAt.Before(q.ValidUntil) ||
		q.CheckedAt.After(now) || !now.Before(q.ValidUntil) || q.MaxAge <= 0 ||
		q.MaxAge > 7*24*time.Hour || now.Sub(q.CheckedAt) >= q.MaxAge {
		return errChat
	}
	if q.ContextTokens <= 0 || q.MaxOutputTokens <= 0 || q.MaxOutputTokens >= q.ContextTokens ||
		q.RequestOverhead <= 0 || q.MessageOverhead <= 0 || q.ToolOverhead <= 0 || q.SchemaOverhead <= 0 ||
		(q.OutputField != "max_tokens" && q.OutputField != "max_completion_tokens") || !q.Fields[q.OutputField] ||
		q.TerminalRule != "finish-then-done" || q.responseLimits() != nil {
		return errChat
	}
	for id, canonical := range q.AcceptedModels {
		if id == "" || canonical != q.CanonicalModel {
			return errChat
		}
	}
	return nil
}

func chatObject(raw []byte, keys ...string) (map[string]json.RawMessage, error) {
	if !utf8.Valid(raw) || !uniqueKeys(raw) {
		return nil, errChat
	}
	var f map[string]json.RawMessage
	if json.Unmarshal(raw, &f) != nil || f == nil {
		return nil, errChat
	}
	for k := range f {
		if !slices.Contains(keys, k) {
			return nil, errChat
		}
	}
	return f, nil
}

func chatString(raw json.RawMessage) (string, error) {
	var s string
	if len(raw) == 0 || raw[0] != '"' || json.Unmarshal(raw, &s) != nil {
		return "", errChat
	}
	return s, nil
}

func chatInt(raw json.RawMessage) (int64, error) {
	var n int64
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) || json.Unmarshal(raw, &n) != nil || n < 0 {
		return 0, errChat
	}
	return n, nil
}

func chatBool(raw json.RawMessage) bool {
	return bytes.Equal(raw, []byte("true")) || bytes.Equal(raw, []byte("false"))
}

// prepare retains the effective body lease until close. Failure and panic release
// it. Call fresh again immediately before dispatch; preparation is not a lease.
func (q *chatQualification) prepare(ctx context.Context, target chatTarget, method, route string, body []byte, budget chatByteBudget) (prepared *chatPrepared, err error) {
	if ctx.Err() != nil || q.fresh() != nil || target.Protocol != nodepolicy.ProtocolOpenAIChatCompletions ||
		method != http.MethodPost || route != routeChatCompletions || target.Endpoint != q.Endpoint ||
		target.Model != q.Model || target.Qualification != q.Ref || target.ContextTokens <= 0 || target.MaxOutputTokens <= 0 {
		return nil, errChat
	}
	if len(body) > 16<<20 || len(q.Defaults) > 16<<20 {
		return nil, errChat
	}
	// 256 bytes per serialized byte conservatively covers JSON maps/tokens, raw
	// copies, defaults, accounting and the returned body simultaneously. These
	// bounds also make the multiplication safe. No unlimited fallback exists.
	release, err := chatAcquire(budget, 256*(int64(len(body))+int64(len(q.Defaults))+1024))
	if err != nil {
		return nil, err
	}
	keep := false
	defer func() {
		if !keep {
			release()
		}
	}()
	keys := append(append([]string(nil), chatRequestKeys...), "parallel_tool_calls")
	f, err := chatObject(body, keys...)
	if err != nil {
		return nil, err
	}
	if len(q.Defaults) != 0 {
		d, e := chatObject(q.Defaults, keys...)
		if e != nil {
			return nil, e
		}
		for k, v := range d {
			if _, exists := f[k]; !exists {
				f[k] = v
			}
		}
	}
	for k := range f {
		if !q.Fields[k] {
			return nil, errChat
		}
	}
	model, err := chatString(f["model"])
	if err != nil || model != q.Model {
		return nil, errChat
	}
	if err := chatScalars(f); err != nil {
		return nil, err
	}
	messages, tools, schemas, err := q.requestGrammar(f)
	if err != nil {
		return nil, err
	}
	output := int64(0)
	for _, k := range []string{"max_tokens", "max_completion_tokens"} {
		if raw, exists := f[k]; exists {
			n, e := chatInt(raw)
			if e != nil || n == 0 || (output != 0 && output != n) {
				return nil, errChat
			}
			output = n
		}
	}
	if output == 0 {
		if q.DefaultOutput <= 0 {
			return nil, errChat
		}
		output = q.DefaultOutput
		f[q.OutputField], err = json.Marshal(output)
		if err != nil {
			return nil, errChat
		}
	}
	if output > min(q.MaxOutputTokens, target.MaxOutputTokens) {
		return nil, errChat
	}
	effective, err := json.Marshal(f)
	if err != nil {
		return nil, errChat
	}
	input, err := q.InputBound(effective)
	if err != nil || input < 0 {
		return nil, errChat
	}
	input, err = chatAdd(input, q.RequestOverhead)
	if err != nil {
		return nil, err
	}
	for _, part := range []struct {
		n        int
		overhead int64
	}{{messages, q.MessageOverhead}, {tools, q.ToolOverhead}, {schemas, q.SchemaOverhead}} {
		for i := 0; i < part.n; i++ {
			input, err = chatAdd(input, part.overhead)
			if err != nil {
				return nil, err
			}
		}
	}
	total, err := chatAdd(input, output)
	if err != nil || total > min(q.ContextTokens, target.ContextTokens) || ctx.Err() != nil {
		return nil, errChat
	}
	prepared = &chatPrepared{Body: effective, Input: input, Output: output, Stream: bytes.Equal(f["stream"], []byte("true")), release: release}
	keep = true
	return prepared, nil
}

func chatScalars(f map[string]json.RawMessage) error {
	for _, key := range []string{"stream", "parallel_tool_calls"} {
		if raw, ok := f[key]; ok && !chatBool(raw) {
			return errChat
		}
	}
	for _, key := range []string{"temperature", "top_p", "frequency_penalty", "presence_penalty"} {
		if raw, ok := f[key]; ok {
			var n float64
			if bytes.Equal(raw, []byte("null")) || json.Unmarshal(raw, &n) != nil {
				return errChat
			}
			switch key {
			case "temperature":
				if n < 0 || n > 2 {
					return errChat
				}
			case "top_p":
				if n < 0 || n > 1 {
					return errChat
				}
			default:
				if n < -2 || n > 2 {
					return errChat
				}
			}
		}
	}
	if raw, ok := f["seed"]; ok {
		var n int64
		if bytes.Equal(raw, []byte("null")) || json.Unmarshal(raw, &n) != nil {
			return errChat
		}
	}
	if raw, ok := f["stop"]; ok {
		if _, err := chatString(raw); err != nil {
			var stops []json.RawMessage
			if json.Unmarshal(raw, &stops) != nil || len(stops) == 0 || len(stops) > 4 {
				return errChat
			}
			for _, s := range stops {
				if _, e := chatString(s); e != nil {
					return e
				}
			}
		}
	}
	if raw, ok := f["stream_options"]; ok {
		o, err := chatObject(raw, "include_usage")
		if err != nil || len(o) != 1 || !chatBool(o["include_usage"]) || !bytes.Equal(f["stream"], []byte("true")) {
			return errChat
		}
	}
	return nil
}

func chatName(s string) bool {
	if len(s) == 0 || len(s) > 64 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

func chatFunction(raw []byte, declaration bool, schema func(json.RawMessage) error) (string, error) {
	keys := []string{"name", "arguments"}
	if declaration {
		keys = []string{"name", "description", "parameters", "strict"}
	}
	f, err := chatObject(raw, keys...)
	if err != nil {
		return "", err
	}
	name, err := chatString(f["name"])
	if err != nil || !chatName(name) {
		return "", errChat
	}
	if declaration {
		if d, ok := f["description"]; ok {
			if _, err := chatString(d); err != nil {
				return "", err
			}
		}
		if s, ok := f["strict"]; ok && !chatBool(s) {
			return "", errChat
		}
		if schema == nil || schema(f["parameters"]) != nil {
			return "", errChat
		}
	} else {
		args, err := chatString(f["arguments"])
		if err != nil || !utf8.ValidString(args) || !uniqueKeys([]byte(args)) || !isObject([]byte(args)) {
			return "", errChat
		}
	}
	return name, nil
}

// Schema keyword names are protocol, but properties/required/enum values are
// data. Even a permissive supplied validator cannot enable references/recursion.
func (q *chatQualification) schema(raw json.RawMessage) error {
	if q.ValidateSchema == nil || chatSchema(raw) != nil || q.ValidateSchema(raw) != nil {
		return errChat
	}
	return nil
}

func chatSchema(raw json.RawMessage) error {
	f, err := chatObject(raw, "type", "properties", "required", "additionalProperties", "items", "enum", "description")
	if err != nil {
		return err
	}
	if raw, ok := f["type"]; ok {
		s, e := chatString(raw)
		if e != nil || !slices.Contains([]string{"object", "array", "string", "number", "integer", "boolean", "null"}, s) {
			return errChat
		}
	} else {
		return errChat
	}
	if raw, ok := f["description"]; ok {
		if _, e := chatString(raw); e != nil {
			return e
		}
	}
	if raw, ok := f["additionalProperties"]; ok && !chatBool(raw) {
		return errChat
	}
	if raw, ok := f["properties"]; ok {
		var p map[string]json.RawMessage
		if json.Unmarshal(raw, &p) != nil || p == nil {
			return errChat
		}
		for _, s := range p {
			if chatSchema(s) != nil {
				return errChat
			}
		}
	}
	if raw, ok := f["items"]; ok && chatSchema(raw) != nil {
		return errChat
	}
	if raw, ok := f["required"]; ok {
		var names []json.RawMessage
		if json.Unmarshal(raw, &names) != nil || names == nil {
			return errChat
		}
		seen := map[string]bool{}
		for _, n := range names {
			s, e := chatString(n)
			if e != nil || seen[s] {
				return errChat
			}
			seen[s] = true
		}
	}
	if raw, ok := f["enum"]; ok {
		var v []json.RawMessage
		if json.Unmarshal(raw, &v) != nil || len(v) == 0 {
			return errChat
		}
	}
	return nil
}

func (q *chatQualification) requestGrammar(f map[string]json.RawMessage) (int, int, int, error) {
	var tools []json.RawMessage
	declared := map[string]bool{}
	if raw, ok := f["tools"]; ok {
		if json.Unmarshal(raw, &tools) != nil || len(tools) == 0 || len(tools) > q.MaxToolCalls {
			return 0, 0, 0, errChat
		}
		for _, raw := range tools {
			t, e := chatObject(raw, "type", "function")
			if e != nil || string(t["type"]) != `"function"` {
				return 0, 0, 0, errChat
			}
			name, e := chatFunction(t["function"], true, q.schema)
			if e != nil || declared[name] {
				return 0, 0, 0, errChat
			}
			declared[name] = true
		}
	}
	if raw, ok := f["tool_choice"]; ok {
		if len(tools) == 0 {
			return 0, 0, 0, errChat
		}
		if s, e := chatString(raw); e == nil {
			if !slices.Contains([]string{"auto", "none", "required"}, s) {
				return 0, 0, 0, errChat
			}
		} else {
			c, e := chatObject(raw, "type", "function")
			if e != nil || string(c["type"]) != `"function"` {
				return 0, 0, 0, errChat
			}
			n, e := chatObject(c["function"], "name")
			if e != nil {
				return 0, 0, 0, e
			}
			name, e := chatString(n["name"])
			if e != nil || !declared[name] {
				return 0, 0, 0, errChat
			}
		}
	}
	if _, ok := f["parallel_tool_calls"]; ok && len(tools) == 0 {
		return 0, 0, 0, errChat
	}
	schemas := len(tools)
	if raw, ok := f["response_format"]; ok {
		r, e := chatObject(raw, "type", "json_schema")
		if e != nil {
			return 0, 0, 0, e
		}
		kind, e := chatString(r["type"])
		if e != nil || !q.Formats[kind] {
			return 0, 0, 0, errChat
		}
		switch kind {
		case "text", "json_object":
			if _, ok := r["json_schema"]; ok {
				return 0, 0, 0, errChat
			}
		case "json_schema":
			s, e := chatObject(r["json_schema"], "name", "description", "schema", "strict")
			if e != nil {
				return 0, 0, 0, e
			}
			n, e := chatString(s["name"])
			if e != nil || !chatName(n) || q.schema(s["schema"]) != nil {
				return 0, 0, 0, errChat
			}
			if d, ok := s["description"]; ok {
				if _, e := chatString(d); e != nil {
					return 0, 0, 0, e
				}
			}
			if b, ok := s["strict"]; ok && !chatBool(b) {
				return 0, 0, 0, errChat
			}
		default:
			return 0, 0, 0, errChat
		}
		schemas++
	}
	var messages []json.RawMessage
	if json.Unmarshal(f["messages"], &messages) != nil || len(messages) == 0 {
		return 0, 0, 0, errChat
	}
	pending, seen := map[string]bool{}, map[string]bool{}
	for _, raw := range messages {
		m, e := chatObject(raw, "role", "content", "name", "tool_calls", "tool_call_id")
		if e != nil {
			return 0, 0, 0, e
		}
		role, e := chatString(m["role"])
		if e != nil || !q.Roles[role] || !slices.Contains([]string{"system", "developer", "user", "assistant", "tool"}, role) {
			return 0, 0, 0, errChat
		}
		if role != "tool" && len(pending) != 0 {
			return 0, 0, 0, errChat
		}
		if n, ok := m["name"]; ok {
			name, e := chatString(n)
			if e != nil || !chatName(name) || role == "tool" {
				return 0, 0, 0, errChat
			}
		}
		calls, hasCalls := m["tool_calls"]
		if hasCalls && (role != "assistant" || !q.Fields["tools"]) {
			return 0, 0, 0, errChat
		}
		if _, e := chatString(m["content"]); e != nil && !(role == "assistant" && hasCalls && bytes.Equal(m["content"], []byte("null"))) {
			return 0, 0, 0, errChat
		}
		if role == "tool" {
			id, e := chatString(m["tool_call_id"])
			if e != nil || !pending[id] {
				return 0, 0, 0, errChat
			}
			delete(pending, id)
		} else if _, ok := m["tool_call_id"]; ok {
			return 0, 0, 0, errChat
		}
		if hasCalls {
			var list []json.RawMessage
			if json.Unmarshal(calls, &list) != nil || len(list) == 0 || len(list) > q.MaxToolCalls {
				return 0, 0, 0, errChat
			}
			for _, raw := range list {
				c, e := chatObject(raw, "id", "type", "function")
				if e != nil || string(c["type"]) != `"function"` {
					return 0, 0, 0, errChat
				}
				id, e := chatString(c["id"])
				if e != nil || id == "" || seen[id] {
					return 0, 0, 0, errChat
				}
				if _, e := chatFunction(c["function"], false, nil); e != nil {
					return 0, 0, 0, e
				}
				seen[id] = true
				pending[id] = true
			}
		}
	}
	if len(pending) != 0 {
		return 0, 0, 0, errChat
	}
	return len(messages), len(tools), schemas, nil
}

type chatParseResult struct {
	// Complete means protocol-complete, including errChatBound. That distinct
	// error retains trusted usage for D's freeze/quarantine reconciliation.
	Complete, Exposed, AboveBound           bool
	Usage                                   *tokenUsage // nil on every incomplete/untrusted response
	Header, FirstBody, FirstDelta, Terminal time.Time
	// FirstDelta observes meaningful text/function data, not a tokenizer timing.
}

type chatToolFragment struct{ id, name, arguments string }
type chatResponseState struct {
	finished, done bool
	usage          *tokenUsage
	tools          map[int64]chatToolFragment
	toolBytes      int64
	responseID     string
	reportedModel  string
	created        int64
	hasEnvelope    bool
}

func (q *chatQualification) responseLimits() error {
	if q == nil || q.Now == nil || q.TerminalRule != "finish-then-done" ||
		q.MaxBodyBytes <= 0 || q.MaxBodyBytes > 64<<20 || q.MaxEventBytes <= 0 || q.MaxEventBytes > 1<<20 ||
		q.MaxLineBytes <= 0 || q.MaxLineBytes > q.MaxEventBytes || q.MaxEventBytes > q.MaxBodyBytes ||
		q.MaxToolBytes <= 0 || q.MaxToolBytes > q.MaxBodyBytes || q.MaxToolCalls <= 0 || q.MaxToolCalls > 128 {
		return errChat
	}
	return nil
}

// parseResponse consumes a bounded source supplied by transport. That owner must
// interrupt a blocked Read on cancellation and close it when this method returns.
// emit borrows validated bytes only for the callback duration; retaining them
// requires the caller's own budget lease. No invalid event is exposed. A callback
// panic propagates after releasing headroom, never becomes a successful response.
// Header is transport-observed; the other milestones are independently sampled.
func (q *chatQualification) parseResponse(ctx context.Context, reader io.Reader, stream bool, input, output int64, header time.Time, budget chatByteBudget, emit func([]byte) error) (result chatParseResult, err error) {
	result.Header = header
	if q.responseLimits() != nil || reader == nil || emit == nil || input < 0 || output <= 0 {
		return result, errChat
	}
	n := q.MaxBodyBytes
	if stream {
		n = q.MaxEventBytes*3 + q.MaxLineBytes + q.MaxToolBytes
	}
	release, err := chatAcquire(budget, 256*(n+1024))
	if err != nil {
		return result, err
	}
	defer release()
	defer func() {
		if !result.Complete {
			result.Usage = nil
			result.AboveBound = false
		}
	}()
	state := chatResponseState{tools: map[int64]chatToolFragment{}}
	observe := func() {
		if result.FirstBody.IsZero() {
			result.FirstBody = q.Now()
		}
	}
	expose := func(raw []byte) error {
		if ctx.Err() != nil {
			return errChat
		}
		// Exposure is conservative even when a sink writes partly then fails.
		result.Exposed = true
		if emit(raw) != nil {
			return errChat
		}
		return nil
	}
	aboveBound := func() bool {
		return state.usage != nil && (state.usage.InputTokens > input || state.usage.OutputTokens > output)
	}
	if !stream {
		body := make([]byte, 0, int(min(q.MaxBodyBytes, 4096)))
		var one [1]byte
		for {
			if ctx.Err() != nil {
				return result, errChat
			}
			n, e := reader.Read(one[:])
			if n > 0 {
				observe()
				if int64(len(body)) >= q.MaxBodyBytes {
					return result, errChat
				}
				body = append(body, one[0])
			}
			if e != nil {
				if e != io.EOF {
					return result, errChat
				}
				break
			}
			if n == 0 {
				return result, errChat
			}
		}
		meaningful, e := q.responseEvent(body, false, &state)
		if e != nil {
			return result, e
		}
		if meaningful {
			result.FirstDelta = q.Now()
		}
		if !state.finished {
			return result, errChat
		}
		result.Terminal = q.Now()
		if !aboveBound() {
			if e := expose(body); e != nil {
				return result, e
			}
		}
	} else {
		line := make([]byte, 0, int(q.MaxLineBytes))
		event := make([]byte, 0, int(q.MaxEventBytes))
		data := make([]byte, 0, int(q.MaxEventBytes))
		hasData := false
		processLine := func() error {
			text := bytes.TrimSuffix(line, []byte{'\r'})
			if len(text) == 0 {
				if !utf8.Valid(event) {
					return errChat
				}
				if hasData {
					if bytes.Equal(data, []byte("[DONE]")) {
						if !state.finished || state.done {
							return errChat
						}
						state.done = true
						result.Terminal = q.Now()
					} else {
						if state.done {
							return errChat
						}
						meaningful, e := q.responseEvent(data, true, &state)
						if e != nil {
							return e
						}
						if meaningful && result.FirstDelta.IsZero() {
							result.FirstDelta = q.Now()
						}
					}
				}
				if len(event) > 0 && !aboveBound() {
					if e := expose(event); e != nil {
						return e
					}
				}
				event = event[:0]
				data = data[:0]
				hasData = false
			} else if text[0] == ':' {
				if state.done {
					return errChat
				}
			} else {
				if state.done || !bytes.HasPrefix(text, []byte("data:")) {
					return errChat
				}
				v := text[5:]
				if len(v) > 0 && v[0] == ' ' {
					v = v[1:]
				}
				if hasData {
					data = append(data, '\n')
				}
				data = append(data, v...)
				hasData = true
			}
			line = line[:0]
			return nil
		}
		var one [1]byte
		total := int64(0)
		for {
			if ctx.Err() != nil {
				return result, errChat
			}
			n, e := reader.Read(one[:])
			if n > 0 {
				observe()
				total++
				if total > q.MaxBodyBytes || int64(len(event)) >= q.MaxEventBytes {
					return result, errChat
				}
				event = append(event, one[0])
				if one[0] == '\n' {
					if e := processLine(); e != nil {
						return result, e
					}
				} else {
					if int64(len(line)) >= q.MaxLineBytes {
						return result, errChat
					}
					line = append(line, one[0])
				}
			}
			if e != nil {
				if e != io.EOF {
					return result, errChat
				}
				break
			}
			if n == 0 {
				return result, errChat
			}
		}
		// EOF never dispatches an unterminated event or substitutes for [DONE].
		if !state.done || len(event) != 0 || len(line) != 0 {
			return result, errChat
		}
	}
	if ctx.Err() != nil {
		return result, errChat
	}
	result.Complete = true
	result.Usage = state.usage
	result.AboveBound = aboveBound()
	if result.AboveBound {
		return result, errChatBound
	}
	return result, nil
}

func (q *chatQualification) responseEvent(raw []byte, stream bool, state *chatResponseState) (bool, error) {
	f, err := chatObject(raw, "id", "object", "created", "model", "choices", "usage")
	if err != nil {
		return false, err
	}
	model, e := chatString(f["model"])
	if e != nil || q.AcceptedModels[model] != q.CanonicalModel || q.CanonicalModel == "" {
		return false, errChatIdentity
	}
	id, e := chatString(f["id"])
	if e != nil || id == "" {
		return false, errChat
	}
	created, e := chatInt(f["created"])
	if e != nil {
		return false, errChat
	}
	object, e := chatString(f["object"])
	expected := "chat.completion"
	if stream {
		expected = "chat.completion.chunk"
	}
	if e != nil || object != expected {
		return false, errChat
	}
	if state.hasEnvelope && (state.responseID != id || state.created != created || state.reportedModel != model) {
		return false, errChatIdentity
	}
	state.responseID = id
	state.reportedModel = model
	state.created = created
	state.hasEnvelope = true
	var choices []json.RawMessage
	if json.Unmarshal(f["choices"], &choices) != nil || choices == nil || len(choices) > 1 || (!stream && len(choices) != 1) {
		return false, errChat
	}
	meaningful := false
	if len(choices) == 1 {
		if state.finished {
			return false, errChat
		}
		key := "message"
		if stream {
			key = "delta"
		}
		c, e := chatObject(choices[0], "index", key, "finish_reason")
		if e != nil {
			return false, e
		}
		index, e := chatInt(c["index"])
		if e != nil || index != 0 {
			return false, errChat
		}
		meaningful, e = q.responseMessage(c[key], stream, state)
		if e != nil {
			return false, e
		}
		finish := c["finish_reason"]
		if len(finish) == 0 {
			return false, errChat
		}
		if !bytes.Equal(finish, []byte("null")) {
			reason, e := chatString(finish)
			if e != nil || !q.FinishReasons[reason] || !slices.Contains([]string{"stop", "length", "tool_calls"}, reason) {
				return false, errChat
			}
			if (reason == "tool_calls") != (len(state.tools) > 0) {
				return false, errChat
			}
			seen := map[string]bool{}
			for i := int64(0); i < int64(len(state.tools)); i++ {
				t := state.tools[i]
				if t.id == "" || seen[t.id] || !chatName(t.name) || !uniqueKeys([]byte(t.arguments)) || !isObject([]byte(t.arguments)) {
					return false, errChat
				}
				seen[t.id] = true
			}
			state.finished = true
		} else if !stream {
			return false, errChat
		}
	} else if !state.finished {
		return false, errChat
	}
	if raw, ok := f["usage"]; ok && !bytes.Equal(raw, []byte("null")) {
		if !q.Usage || !state.finished {
			return false, errChat
		}
		u, e := chatUsage(raw)
		if e != nil {
			return false, e
		}
		if state.usage != nil && *state.usage != *u {
			return false, errChat
		}
		state.usage = u
	} else if len(choices) == 0 {
		return false, errChat
	}
	return meaningful, nil
}

func chatUsage(raw []byte) (*tokenUsage, error) {
	f, err := chatObject(raw, "prompt_tokens", "completion_tokens", "total_tokens")
	if err != nil || len(f) != 3 {
		return nil, errChat
	}
	in, e := chatInt(f["prompt_tokens"])
	if e != nil {
		return nil, e
	}
	out, e := chatInt(f["completion_tokens"])
	if e != nil {
		return nil, e
	}
	total, e := chatInt(f["total_tokens"])
	if e != nil {
		return nil, e
	}
	sum, e := chatAdd(in, out)
	if e != nil || sum != total {
		return nil, errChat
	}
	return &tokenUsage{InputTokens: in, OutputTokens: out}, nil
}

func (q *chatQualification) responseMessage(raw []byte, stream bool, state *chatResponseState) (bool, error) {
	f, err := chatObject(raw, "role", "content", "tool_calls")
	if err != nil {
		return false, err
	}
	if role, ok := f["role"]; ok {
		if string(role) != `"assistant"` {
			return false, errChat
		}
	} else if !stream {
		return false, errChat
	}
	meaningful := false
	if raw, ok := f["content"]; ok {
		if !bytes.Equal(raw, []byte("null")) {
			s, e := chatString(raw)
			if e != nil {
				return false, e
			}
			meaningful = s != ""
		}
	} else if !stream {
		return false, errChat
	}
	if raw, ok := f["tool_calls"]; ok {
		if !q.Fields["tools"] {
			return false, errChat
		}
		var calls []json.RawMessage
		if json.Unmarshal(raw, &calls) != nil || len(calls) == 0 || len(calls) > q.MaxToolCalls {
			return false, errChat
		}
		for i, raw := range calls {
			keys := []string{"id", "type", "function"}
			if stream {
				keys = append(keys, "index")
			}
			c, e := chatObject(raw, keys...)
			if e != nil {
				return false, e
			}
			index := int64(i)
			if stream {
				index, e = chatInt(c["index"])
				if e != nil {
					return false, e
				}
			}
			if index >= int64(q.MaxToolCalls) {
				return false, errChat
			}
			t, exists := state.tools[index]
			if !exists {
				id, e := chatString(c["id"])
				if e != nil || id == "" || string(c["type"]) != `"function"` {
					return false, errChat
				}
				t.id = id
			} else if _, ok := c["id"]; ok {
				return false, errChat
			} else if _, ok := c["type"]; ok {
				return false, errChat
			}
			fn, e := chatObject(c["function"], "name", "arguments")
			if e != nil {
				return false, e
			}
			name, args := "", ""
			if raw, ok := fn["name"]; ok {
				name, e = chatString(raw)
				if e != nil {
					return false, e
				}
			}
			if raw, ok := fn["arguments"]; ok {
				args, e = chatString(raw)
				if e != nil {
					return false, e
				}
			} else if !stream {
				return false, errChat
			}
			if !exists && name == "" {
				return false, errChat
			}
			added := int64(len(name)) + int64(len(args))
			if !exists {
				added += int64(len(t.id))
			}
			if added > q.MaxToolBytes-state.toolBytes {
				return false, errChat
			}
			state.toolBytes += added
			t.name += name
			t.arguments += args
			state.tools[index] = t
			meaningful = meaningful || name != "" || args != ""
		}
	}
	if !stream && bytes.Equal(f["content"], []byte("null")) && len(state.tools) == 0 {
		return false, errChat
	}
	return meaningful, nil
}
