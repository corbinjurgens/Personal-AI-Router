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
	"strconv"
	"strings"
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

// JSON containers below hold views into the input, never nested serialized
// copies. Before decoding, chatDecodeStorage counts tokens without allocating.
// Its bound is 16 bytes per wire byte plus 512 bytes per token: decoded strings,
// unescaping scratch, decoded keys and string-to-byte argument copies need at
// most eight input-sized copies; 16 also covers allocator rounding and overlap.
// A token pays for a 24-byte RawMessage descriptor, 16-byte string key and map
// control/entry storage, including old/new map and slice backing arrays during
// growth (under 256 bytes per member), doubled for simultaneous validation and
// protocol/history maps. Every member/key/container/scalar is counted, including
// opaque schemas and strings containing serialized arguments (charged again
// before those arguments are parsed). No depth multiplier is needed: container
// values are views, and total live key maps are bounded by the token count.
// Fixed storage covers decoder/encoder scalar state, depth <=64, lease records,
// tiny maps and allocation size classes. Qualification callbacks and emit are
// caller-owned; emit only borrows bytes for its duration.
const chatFixedStorage = 96 << 10

func chatDecodeStorage(raw []byte) (int64, error) {
	depth, tokens := 0, int64(0)
	for i := 0; i < len(raw); {
		switch raw[i] {
		case ' ', '\t', '\r', '\n', ':', ',':
			i++
		case '"':
			tokens++
			i++
			for i < len(raw) && raw[i] != '"' {
				if raw[i] == '\\' {
					i++
				}
				i++
			}
			if i >= len(raw) {
				return 0, errChat
			}
			i++
		case '{', '[':
			depth++
			if depth > maxStrictBodyDepth {
				return 0, errChat
			}
			tokens++
			i++
		case '}', ']':
			depth--
			if depth < 0 {
				return 0, errChat
			}
			i++
		default:
			tokens++
			start := i
			for i < len(raw) && !strings.ContainsRune(" \t\r\n,:{}[]\"", rune(raw[i])) {
				i++
			}
			if i == start {
				return 0, errChat
			}
		}
	}
	if depth != 0 {
		return 0, errChat
	}
	// Inputs have an independent <=64 MiB ceiling, so this cannot overflow.
	return 16*int64(len(raw)) + 512*tokens, nil
}

func (l *chatLease) decode(raw []byte) error {
	n, err := chatDecodeStorage(raw)
	if err != nil {
		return err
	}
	if n > 0 {
		if err := l.grow(n); err != nil {
			return err
		}
	}
	if !utf8.Valid(raw) || !json.Valid(raw) {
		return errChat
	}
	return nil
}

// chatParseChunk is one nonblocking growth step while a nonstream body arrives,
// so a large configured ceiling reserves nothing before its bytes are read.
const chatParseChunk = 32 << 10

// chatLease charges a budget in nonblocking segments. A parser pays for storage
// as it grows and can drop transient headroom while keeping the bytes it
// returns; every segment release is idempotent and runs on failure or panic.
type chatLease struct {
	budget chatByteBudget
	head   *chatReservation
}

type chatReservation struct {
	release func()
	next    *chatReservation
}

func newChatLease(budget chatByteBudget) *chatLease {
	return &chatLease{budget: budget}
}

// grow charges n bytes before the caller allocates them.
func (l *chatLease) grow(n int64) error {
	// The release closure/once wrapper is itself admitted before allocation.
	release, err := chatAcquire(l.budget, n+256)
	if err != nil {
		return err
	}
	l.head = &chatReservation{release: release, next: l.head}
	return nil
}

func (l *chatLease) release() {
	for l.head != nil {
		head := l.head
		l.head = head.next
		head.release()
	}
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

// Protocol envelopes require exact canonical field spellings. Together with
// exact duplicate rejection this also rejects every case-folded duplicate.
// Property names and function arguments are opaque data: case-distinct keys
// stay valid, while exact duplicates and excess depth are still rejected.
func chatObject(raw []byte, keys ...string) (map[string]json.RawMessage, error) {
	if !chatExactUnique(raw) {
		return nil, errChat
	}
	f, err := chatMembers(raw)
	if err != nil {
		return nil, errChat
	}
	for k := range f {
		if !slices.Contains(keys, k) {
			return nil, errChat
		}
	}
	return f, nil
}

// chatExactUnique reports whether data is one JSON value nested at most
// maxStrictBodyDepth deep with no exact duplicate object key at any depth.
// Callers admit decode storage before entering this allocating validation.
func chatExactUnique(data []byte) bool {
	return json.Valid(data) && chatExactValue(bytes.TrimSpace(data), 0)
}

func chatExactValue(raw []byte, depth int) bool {
	if raw[0] != '{' && raw[0] != '[' {
		return true
	}
	if depth >= maxStrictBodyDepth {
		return false
	}
	keys := map[string]bool{}
	rest := bytes.TrimSpace(raw[1 : len(raw)-1])
	for len(rest) > 0 {
		if raw[0] == '{' {
			keyRaw, tail := chatValueView(rest)
			key, err := chatString(keyRaw)
			if err != nil || keys[key] {
				return false
			}
			keys[key] = true
			rest = bytes.TrimSpace(bytes.TrimSpace(tail)[1:]) // colon
		}
		value, tail := chatValueView(rest)
		if !chatExactValue(value, depth+1) {
			return false
		}
		rest = bytes.TrimSpace(tail)
		if len(rest) > 0 {
			rest = bytes.TrimSpace(rest[1:]) // comma
		}
	}
	return true
}

// chatValueView splits already syntax-validated JSON without allocating or
// copying. Containers and strings are scanned with bounded integer state.
func chatValueView(raw []byte) (json.RawMessage, []byte) {
	depth := 0
	for i := 0; i < len(raw); i++ {
		switch raw[i] {
		case '"':
			i++
			for raw[i] != '"' {
				if raw[i] == '\\' {
					i++
				}
				i++
			}
			if depth == 0 {
				return raw[:i+1], raw[i+1:]
			}
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				return raw[:i+1], raw[i+1:]
			}
		case ',', ':', ' ', '\t', '\r', '\n':
			if depth == 0 {
				return raw[:i], raw[i:]
			}
		}
	}
	return raw, nil
}

func chatMembers(raw []byte) (map[string]json.RawMessage, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '{' {
		return nil, errChat
	}
	f := map[string]json.RawMessage{}
	rest := bytes.TrimSpace(raw[1 : len(raw)-1])
	for len(rest) > 0 {
		keyRaw, tail := chatValueView(rest)
		key, err := chatString(keyRaw)
		if err != nil {
			return nil, err
		}
		rest = bytes.TrimSpace(bytes.TrimSpace(tail)[1:])
		f[key], tail = chatValueView(rest)
		rest = bytes.TrimSpace(tail)
		if len(rest) > 0 {
			rest = bytes.TrimSpace(rest[1:])
		}
	}
	return f, nil
}

func chatArray(raw []byte) ([]json.RawMessage, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '[' {
		return nil, errChat
	}
	values := []json.RawMessage{}
	rest := bytes.TrimSpace(raw[1 : len(raw)-1])
	for len(rest) > 0 {
		value, tail := chatValueView(rest)
		values = append(values, value)
		rest = bytes.TrimSpace(tail)
		if len(rest) > 0 {
			rest = bytes.TrimSpace(rest[1:])
		}
	}
	return values, nil
}

// chatExactObject checks opaque arguments with exact duplicate/depth rules.
// Its key maps are covered by the separately admitted argument token count.
func chatExactObject(data []byte) bool {
	trimmed := bytes.TrimLeft(data, " \t\r\n")
	return len(trimmed) > 0 && trimmed[0] == '{' && chatExactUnique(data)
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
	// Transient parsing is charged from the actual input and token count and
	// dropped once the effective body exists; only that body stays leased. A
	// configured ceiling reserves nothing, and no unlimited fallback exists.
	transient := newChatLease(budget)
	retained := newChatLease(budget)
	keep := false
	defer func() {
		if !keep {
			retained.release()
		}
		transient.release()
	}()
	if err := transient.grow(chatFixedStorage); err != nil {
		return nil, err
	}
	if err := transient.decode(body); err != nil {
		return nil, err
	}
	if len(q.Defaults) > 0 {
		if err := transient.decode(q.Defaults); err != nil {
			return nil, err
		}
	}
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
	messages, tools, schemas, err := q.requestGrammar(f, transient)
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
		f[q.OutputField] = strconv.AppendInt(nil, output, 10)
	}
	if output > min(q.MaxOutputTokens, target.MaxOutputTokens) {
		return nil, errChat
	}
	// Every key is a canonical protocol spelling. Serialize the borrowed values
	// directly into an exactly sized buffer, admitting it BEFORE allocation.
	size := 1
	for key, value := range f {
		size += len(key) + len(value) + 4 // quotes, colon, comma (or final brace)
	}
	if err := retained.grow(2 * int64(size)); err != nil {
		return nil, err
	}
	effective := make([]byte, 0, size)
	effective = append(effective, '{')
	for _, key := range keys {
		if value, ok := f[key]; ok {
			if len(effective) > 1 {
				effective = append(effective, ',')
			}
			effective = append(effective, '"')
			effective = append(effective, key...)
			effective = append(effective, '"', ':')
			effective = append(effective, value...)
		}
	}
	effective = append(effective, '}')
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
	prepared = &chatPrepared{Body: effective, Input: input, Output: output, Stream: bytes.Equal(f["stream"], []byte("true")), release: retained.release}
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
			stops, e := chatArray(raw)
			if e != nil || len(stops) == 0 || len(stops) > 4 {
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

func chatFunction(raw []byte, declaration bool, schema func(json.RawMessage) error, lease *chatLease) (string, error) {
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
		if err != nil {
			return "", errChat
		}
		if err := lease.decode([]byte(args)); err != nil {
			return "", err
		}
		if !chatExactObject([]byte(args)) {
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
		p, e := chatMembers(raw)
		if e != nil {
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
		names, e := chatArray(raw)
		if e != nil {
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
		v, e := chatArray(raw)
		if e != nil || len(v) == 0 {
			return errChat
		}
	}
	return nil
}

func (q *chatQualification) requestGrammar(f map[string]json.RawMessage, lease *chatLease) (int, int, int, error) {
	var tools []json.RawMessage
	declared := map[string]bool{}
	if raw, ok := f["tools"]; ok {
		var e error
		tools, e = chatArray(raw)
		if e != nil || len(tools) == 0 || len(tools) > q.MaxToolCalls {
			return 0, 0, 0, errChat
		}
		for _, raw := range tools {
			t, e := chatObject(raw, "type", "function")
			if e != nil || string(t["type"]) != `"function"` {
				return 0, 0, 0, errChat
			}
			name, e := chatFunction(t["function"], true, q.schema, lease)
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
	messages, e := chatArray(f["messages"])
	if e != nil || len(messages) == 0 {
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
			list, e := chatArray(calls)
			if e != nil || len(list) == 0 || len(list) > q.MaxToolCalls {
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
				if _, e := chatFunction(c["function"], false, nil, lease); e != nil {
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
	allowance, err := q.acquireParserAllowance(budget, stream)
	if err != nil {
		return result, err
	}
	defer allowance.release()
	return q.parseResponseWithAllowance(ctx, reader, stream, input, output, header, allowance, emit)
}

// One ownership record transfers pre-admitted parser storage to exactly one
// parser. Standalone callers acquire their own through parseResponse.
type chatParserAllowance struct {
	mu     sync.Mutex
	lease  *chatLease
	owner  *chatQualification
	stream bool
}

func (q *chatQualification) acquireParserAllowance(budget chatByteBudget, stream bool) (*chatParserAllowance, error) {
	if budget == nil || q.responseLimits() != nil {
		return nil, errChatBudget
	}
	lease := newChatLease(budget)
	if err := lease.grow(chatFixedStorage); err != nil {
		return nil, err
	}
	if stream {
		fixed := 2*q.MaxLineBytes + 6*q.MaxEventBytes + 4*q.MaxToolBytes + 512*int64(q.MaxToolCalls)
		if err := lease.grow(fixed); err != nil {
			lease.release()
			return nil, err
		}
	}
	return &chatParserAllowance{lease: lease, owner: q, stream: stream}, nil
}

func (a *chatParserAllowance) release() {
	a.mu.Lock()
	lease := a.lease
	a.lease = nil
	a.mu.Unlock()
	if lease != nil {
		lease.release()
	}
}

func (a *chatParserAllowance) take(q *chatQualification, stream bool) *chatLease {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.owner != q || a.stream != stream {
		return nil
	}
	lease := a.lease
	a.lease = nil
	return lease
}

func (q *chatQualification) parseResponseWithAllowance(ctx context.Context, reader io.Reader, stream bool, input, output int64, header time.Time, allowance *chatParserAllowance, emit func([]byte) error) (result chatParseResult, err error) {
	result.Header = header
	if q.responseLimits() != nil || reader == nil || emit == nil || input < 0 || output <= 0 {
		return result, errChat
	}
	lease := allowance.take(q, stream)
	if lease == nil {
		return result, errChatBudget
	}
	defer lease.release()
	budget := lease.budget
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
		readSize := min(int64(chatParseChunk), q.MaxBodyBytes)
		// Double each raw capacity to cover allocator size-class rounding.
		if err := lease.grow(4 * readSize); err != nil {
			return result, err
		}
		scratch := make([]byte, int(readSize))
		body := make([]byte, 0, int(readSize))
		for {
			if ctx.Err() != nil {
				return result, errChat
			}
			n, e := reader.Read(scratch)
			if n > 0 {
				observe()
				need := int64(len(body)) + int64(n)
				if need > q.MaxBodyBytes {
					return result, errChat
				}
				if int64(cap(body)) < need {
					next := min(2*int64(cap(body)), q.MaxBodyBytes)
					if next < need {
						next = need
					}
					// Keep every old allocation charged through return. Charging
					// the FULL new capacity covers old/new overlap without relying
					// on GC or reusing the old buffer's allowance.
					if err := lease.grow(2 * next); err != nil {
						return result, errChatBudget
					}
					grown := make([]byte, len(body), int(next))
					copy(grown, body)
					body = grown
				}
				body = append(body, scratch[:n]...)
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
		if err := lease.decode(body); err != nil {
			return result, err
		}
		meaningful, e := q.responseEvent(body, false, &state, lease)
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
						eventLease := newChatLease(budget)
						defer eventLease.release()
						if err := eventLease.decode(data); err != nil {
							return err
						}
						meaningful, e := q.responseEvent(data, true, &state, eventLease)
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

func (q *chatQualification) responseEvent(raw []byte, stream bool, state *chatResponseState, lease *chatLease) (bool, error) {
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
	choices, e := chatArray(f["choices"])
	if e != nil || len(choices) > 1 || (!stream && len(choices) != 1) {
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
				if err := lease.decode([]byte(t.arguments)); err != nil {
					return false, err
				}
				if t.id == "" || seen[t.id] || !chatName(t.name) || !chatExactObject([]byte(t.arguments)) {
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
		calls, e := chatArray(raw)
		if e != nil || len(calls) == 0 || len(calls) > q.MaxToolCalls {
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
