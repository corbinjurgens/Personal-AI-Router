// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Local qualification (docs/local-qualification.mdx): the finite bounds under
// which this node may admit a hybrid request for a local model, and the
// validation of each strict request against them before any wake, unload, slot
// or inference.
//
// A qualification is node-local evidence, read once from the artifact file
// <appdir>/local-qualification.json. It names the engine, the model, the model
// content and launches it was measured on, the tokenizer and chat template, the
// served context and output ceiling, the accounting method that bounds input
// tokens, and the request shapes it covers. A profile opts in by naming one
// qualification id and revision; strict capacity is installed only while the
// engine's current identity (nodepolicy.EngineIdentity) matches it. Absent or
// unusable evidence leaves nothing qualified: there is no estimated fallback.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"slices"
	"strings"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// qualificationFileName is the artifact file under the per-user app data dir.
const qualificationFileName = "local-qualification.json"

// qualificationDocumentVersion is the artifact schema version.
const qualificationDocumentVersion = 1

// accountingUTF8Bytes bounds input tokens by rendered UTF-8 bytes: a
// byte-level BPE tokenizer emits at most one token per byte. Template tokens
// are covered by the declared per-request, per-message, per-tool and schema
// overheads. It is the only supported method.
const accountingUTF8Bytes = "utf8-bytes"

// routeChatCompletions is the only request shape a qualification may cover.
const routeChatCompletions = "/v1/chat/completions"

// maxQualificationBytes bounds the artifact file.
const maxQualificationBytes = 1 << 20

// maxQualifiedContextTokens bounds a qualified context, and with it every
// overhead, so the accounted sums of a body within the proxy's size limit
// cannot overflow.
const maxQualifiedContextTokens = 1 << 24

var (
	qualificationDigest = regexp.MustCompile(`^[0-9a-f]{64}$`)
	qualificationLabel  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,127}$`)
)

// localQualification is one qualified (engine, model, content, launch,
// defaults) combination and its bounds.
type localQualification struct {
	ID              string                  `json:"id"`
	Revision        string                  `json:"revision"`
	Engine          string                  `json:"engine"`
	Model           string                  `json:"model"`
	ModelDigests    []string                `json:"modelDigests"`
	LaunchDigests   []string                `json:"launchDigests"`
	RequestOptions  json.RawMessage         `json:"requestOptions,omitempty"`
	Tokenizer       string                  `json:"tokenizer"`
	ChatTemplate    string                  `json:"chatTemplate"`
	ContextTokens   int                     `json:"contextTokens"`
	MaxOutputTokens int                     `json:"maxOutputTokens"`
	Routes          []string                `json:"routes"`
	Accounting      qualificationAccounting `json:"accounting"`
	Evidence        string                  `json:"evidence"`
}

type qualificationAccounting struct {
	Method                string `json:"method"`
	RequestOverheadTokens int    `json:"requestOverheadTokens"`
	MessageOverheadTokens int    `json:"messageOverheadTokens"`
	ToolOverheadTokens    int    `json:"toolOverheadTokens"`
	SchemaOverheadTokens  int    `json:"schemaOverheadTokens"`
}

// qualificationRegistry holds the node's qualifications by id and revision.
type qualificationRegistry map[nodepolicy.QualificationRef]*localQualification

func (r qualificationRegistry) lookup(ref nodepolicy.QualificationRef) (*localQualification, bool) {
	q, ok := r[ref]
	return q, ok
}

func (q localQualification) validate() error {
	if err := (nodepolicy.QualificationRef{ID: q.ID, Revision: q.Revision}).Validate(); err != nil {
		return err
	}
	if !engines.ValidRoutingID(q.Engine) {
		return fmt.Errorf("unknown engine %q", q.Engine)
	}
	if strings.TrimSpace(q.Model) == "" || nodepolicy.IsReservedAlias(q.Model) {
		return errors.New("model must name a model")
	}
	for name, list := range map[string][]string{"modelDigests": q.ModelDigests, "launchDigests": q.LaunchDigests} {
		if len(list) == 0 {
			return fmt.Errorf("%s must not be empty", name)
		}
		for _, d := range list {
			if !qualificationDigest.MatchString(d) {
				return fmt.Errorf("%s must hold lowercase SHA-256 hex", name)
			}
		}
	}
	if len(q.RequestOptions) > 0 {
		var obj map[string]json.RawMessage
		if json.Unmarshal(q.RequestOptions, &obj) != nil || obj == nil {
			return errors.New("requestOptions must be a JSON object")
		}
		for key := range obj {
			if strings.EqualFold(key, "model") {
				return errors.New("requestOptions may not set model")
			}
		}
	}
	if !qualificationLabel.MatchString(q.Tokenizer) || !qualificationLabel.MatchString(q.ChatTemplate) {
		return errors.New("tokenizer and chatTemplate must name their identity")
	}
	if q.ContextTokens < 1 || q.ContextTokens > maxQualifiedContextTokens || q.MaxOutputTokens < 1 || q.MaxOutputTokens >= q.ContextTokens {
		return fmt.Errorf("contextTokens must be 1..%d and maxOutputTokens below it", maxQualifiedContextTokens)
	}
	if len(q.Routes) == 0 {
		return errors.New("routes must not be empty")
	}
	for _, r := range q.Routes {
		if r != routeChatCompletions {
			return fmt.Errorf("unsupported route %q", r)
		}
	}
	a := q.Accounting
	if a.Method != accountingUTF8Bytes {
		return fmt.Errorf("unsupported accounting method %q", a.Method)
	}
	for _, v := range []int{a.RequestOverheadTokens, a.MessageOverheadTokens, a.ToolOverheadTokens, a.SchemaOverheadTokens} {
		if v < 0 || v >= q.ContextTokens {
			return errors.New("accounting overheads must be nonnegative and below the context")
		}
	}
	if strings.TrimSpace(q.Evidence) == "" || len(q.Evidence) > 4096 {
		return errors.New("evidence must describe the qualification in at most 4096 bytes")
	}
	return nil
}

// parseQualificationDocument decodes and validates a whole artifact. Unknown
// fields, an unsupported version, an invalid entry or a repeated id and
// revision reject the whole document.
func parseQualificationDocument(data []byte) (qualificationRegistry, error) {
	var doc struct {
		Version        *int                 `json:"version"`
		Qualifications []localQualification `json:"qualifications"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&doc); err != nil {
		return nil, fmt.Errorf("invalid qualification document: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("invalid qualification document: trailing data")
	}
	if doc.Version == nil || *doc.Version != qualificationDocumentVersion {
		return nil, fmt.Errorf("qualification document version must be %d", qualificationDocumentVersion)
	}
	out := make(qualificationRegistry, len(doc.Qualifications))
	for i := range doc.Qualifications {
		q := doc.Qualifications[i]
		if err := q.validate(); err != nil {
			return nil, fmt.Errorf("qualifications[%d]: %w", i, err)
		}
		ref := nodepolicy.QualificationRef{ID: q.ID, Revision: q.Revision}
		if _, dup := out[ref]; dup {
			return nil, fmt.Errorf("qualifications[%d]: repeated id and revision", i)
		}
		out[ref] = &q
	}
	return out, nil
}

// loadQualificationRegistry reads the artifact at path. A missing file is an
// empty registry; an unreadable or invalid one is an empty registry and an
// error, so nothing is qualified on evidence that could not be checked.
func loadQualificationRegistry(path string, read func(string) ([]byte, error)) (qualificationRegistry, error) {
	data, err := read(path)
	if os.IsNotExist(err) {
		return qualificationRegistry{}, nil
	}
	if err != nil {
		return qualificationRegistry{}, err
	}
	if len(data) > maxQualificationBytes {
		return qualificationRegistry{}, errors.New("qualification document is too large")
	}
	reg, err := parseQualificationDocument(data)
	if err != nil {
		return qualificationRegistry{}, err
	}
	return reg, nil
}

// installedQualification is what one installed strict model was qualified
// under: the evidence and the profile that named it.
type installedQualification struct {
	q       *localQualification
	profile nodepolicy.Profile
}

// strictQualificationError rejects a request body the qualification does not
// cover. Its reason names fields and bounds only, never request content.
type strictQualificationError struct{ reason string }

func (e *strictQualificationError) Error() string { return "strict qualification: " + e.reason }

func rejectBody(format string, args ...any) error {
	return &strictQualificationError{reason: fmt.Sprintf(format, args...)}
}

// strictBound is the accounted size of one validated request.
type strictBound struct {
	input, output int
}

// chatRequestKeys are the top-level fields a qualified chat request may send.
// Anything else, including any context-length override, is unsupported.
var chatRequestKeys = []string{
	"model", "messages", "tools", "tool_choice", "response_format", "max_tokens", "max_completion_tokens",
	"stream", "stream_options", "temperature", "top_p", "stop", "seed", "frequency_penalty", "presence_penalty",
}

// validateRequest checks one strict request against the qualification and the
// profile it is installed under, and returns the effective body (the request
// with the profile's defaults merged in) with its bound. The effective body is
// what is accounted and what the engine receives.
func (q *localQualification) validateRequest(profile nodepolicy.Profile, route string, body []byte) ([]byte, strictBound, error) {
	if !slices.Contains(q.Routes, route) {
		return nil, strictBound{}, rejectBody("route %s is not qualified", route)
	}
	if !uniqueKeys(body) {
		return nil, strictBound{}, rejectBody("the request repeats a key or is not JSON")
	}
	var request map[string]json.RawMessage
	if json.Unmarshal(body, &request) != nil || request == nil {
		return nil, strictBound{}, rejectBody("the request must be a JSON object")
	}
	for key := range request {
		if !slices.Contains(chatRequestKeys, key) {
			return nil, strictBound{}, rejectBody("field %q is not supported", key)
		}
	}
	effective := mergeDefaults(body, profile.RequestOptions)
	var fields map[string]json.RawMessage
	if json.Unmarshal(effective, &fields) != nil {
		return nil, strictBound{}, rejectBody("the request with its defaults is not a JSON object")
	}
	var bound strictBound
	if err := checkChatScalars(fields); err != nil {
		return nil, strictBound{}, err
	}
	output, err := outputBound(fields)
	if err != nil {
		return nil, strictBound{}, err
	}
	if ceiling := min(profile.MaxOutputTokens, q.MaxOutputTokens); output > ceiling {
		return nil, strictBound{}, rejectBody("the output bound %d exceeds the qualified ceiling %d", output, ceiling)
	}
	bound.output = output
	a := q.Accounting
	bound.input = a.RequestOverheadTokens
	n, err := accountMessages(fields["messages"], a)
	if err != nil {
		return nil, strictBound{}, err
	}
	bound.input += n
	if n, err = accountTools(fields, a); err != nil {
		return nil, strictBound{}, err
	}
	bound.input += n
	if n, err = accountResponseFormat(fields["response_format"], a); err != nil {
		return nil, strictBound{}, err
	}
	bound.input += n
	if context := min(profile.ContextTokens, q.ContextTokens); bound.input+bound.output > context {
		return nil, strictBound{}, rejectBody("the input bound %d and output bound %d exceed the qualified context %d", bound.input, bound.output, context)
	}
	return effective, bound, nil
}

func checkChatScalars(f map[string]json.RawMessage) error {
	var model string
	if json.Unmarshal(f["model"], &model) != nil || model == "" {
		return rejectBody("model must be a nonempty string")
	}
	for _, key := range []string{"temperature", "top_p", "frequency_penalty", "presence_penalty"} {
		if raw, ok := f[key]; ok {
			var v float64
			if json.Unmarshal(raw, &v) != nil {
				return rejectBody("%s must be a number", key)
			}
		}
	}
	if raw, ok := f["seed"]; ok {
		var v int64
		if json.Unmarshal(raw, &v) != nil {
			return rejectBody("seed must be an integer")
		}
	}
	if raw, ok := f["stream"]; ok {
		var v bool
		if json.Unmarshal(raw, &v) != nil {
			return rejectBody("stream must be a boolean")
		}
	}
	if raw, ok := f["stream_options"]; ok {
		var v struct {
			IncludeUsage bool `json:"include_usage"`
		}
		if decodeExact(raw, &v) != nil {
			return rejectBody("stream_options supports include_usage only")
		}
	}
	if raw, ok := f["stop"]; ok {
		var one string
		var many []string
		if json.Unmarshal(raw, &one) != nil && (json.Unmarshal(raw, &many) != nil || len(many) > 4) {
			return rejectBody("stop must be a string or at most four strings")
		}
	}
	return nil
}

// outputBound is the single finite output request. Both spellings must agree
// when both are sent; a request with neither has no bound and is rejected.
func outputBound(f map[string]json.RawMessage) (int, error) {
	bound := 0
	for _, key := range []string{"max_tokens", "max_completion_tokens"} {
		raw, ok := f[key]
		if !ok {
			continue
		}
		var v int
		if json.Unmarshal(raw, &v) != nil || v < 1 {
			return 0, rejectBody("%s must be a positive integer", key)
		}
		if bound != 0 && v != bound {
			return 0, rejectBody("max_tokens and max_completion_tokens disagree")
		}
		bound = v
	}
	if bound == 0 {
		return 0, rejectBody("a strict request needs max_tokens or max_completion_tokens")
	}
	return bound, nil
}

type chatMessage struct {
	Role       string            `json:"role"`
	Content    json.RawMessage   `json:"content"`
	Name       *string           `json:"name"`
	ToolCalls  []json.RawMessage `json:"tool_calls"`
	ToolCallID *string           `json:"tool_call_id"`
}

type chatToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func accountMessages(raw json.RawMessage, a qualificationAccounting) (int, error) {
	var messages []json.RawMessage
	if json.Unmarshal(raw, &messages) != nil || len(messages) == 0 {
		return 0, rejectBody("messages must be a nonempty array")
	}
	total := 0
	for i, m := range messages {
		var msg chatMessage
		if decodeExact(m, &msg) != nil {
			return 0, rejectBody("messages[%d] has an unsupported shape", i)
		}
		switch msg.Role {
		case "system", "user", "assistant", "tool":
		default:
			return 0, rejectBody("messages[%d].role is not supported", i)
		}
		n := a.MessageOverheadTokens + len(msg.Role)
		text, isNull, err := contentText(msg.Content)
		if err != nil {
			return 0, rejectBody("messages[%d].content: %s", i, err)
		}
		if isNull && (msg.Role != "assistant" || len(msg.ToolCalls) == 0) {
			return 0, rejectBody("messages[%d].content is required", i)
		}
		n += text
		if msg.Name != nil {
			n += len(*msg.Name)
		}
		if msg.ToolCallID != nil {
			if msg.Role != "tool" {
				return 0, rejectBody("messages[%d].tool_call_id belongs to a tool message", i)
			}
			n += len(*msg.ToolCallID)
		}
		if len(msg.ToolCalls) > 0 && msg.Role != "assistant" {
			return 0, rejectBody("messages[%d].tool_calls belong to an assistant message", i)
		}
		for j, raw := range msg.ToolCalls {
			var call chatToolCall
			if decodeExact(raw, &call) != nil || call.Type != "function" || call.Function.Name == "" {
				return 0, rejectBody("messages[%d].tool_calls[%d] must call a named function", i, j)
			}
			n += compactLen(raw)
		}
		total += n
	}
	return total, nil
}

// contentText counts the UTF-8 bytes of a message's text. Content is a string,
// an array of text parts, or null; any other part (an image) is unsupported.
func contentText(raw json.RawMessage) (n int, isNull bool, err error) {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return 0, true, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return len(s), false, nil
	}
	var parts []json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		return 0, false, errors.New("must be a string or an array of text parts")
	}
	for _, p := range parts {
		var part struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if decodeExact(p, &part) != nil || part.Type != "text" {
			return 0, false, errors.New("only text parts are supported")
		}
		n += len(part.Text)
	}
	return n, false, nil
}

func accountTools(f map[string]json.RawMessage, a qualificationAccounting) (int, error) {
	total := 0
	if raw, ok := f["tools"]; ok {
		var tools []json.RawMessage
		if json.Unmarshal(raw, &tools) != nil {
			return 0, rejectBody("tools must be an array")
		}
		for i, t := range tools {
			var tool struct {
				Type     string `json:"type"`
				Function struct {
					Name        string          `json:"name"`
					Description string          `json:"description"`
					Parameters  json.RawMessage `json:"parameters"`
					Strict      bool            `json:"strict"`
				} `json:"function"`
			}
			if decodeExact(t, &tool) != nil || tool.Type != "function" || tool.Function.Name == "" {
				return 0, rejectBody("tools[%d] must be a named function", i)
			}
			if len(tool.Function.Parameters) > 0 && !isObject(tool.Function.Parameters) {
				return 0, rejectBody("tools[%d].function.parameters must be an object", i)
			}
			total += a.ToolOverheadTokens + compactLen(t)
		}
	}
	if raw, ok := f["tool_choice"]; ok {
		var mode string
		if json.Unmarshal(raw, &mode) == nil {
			if mode != "none" && mode != "auto" && mode != "required" {
				return 0, rejectBody("tool_choice is not supported")
			}
			total += len(mode)
		} else {
			var choice struct {
				Type     string `json:"type"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}
			if decodeExact(raw, &choice) != nil || choice.Type != "function" || choice.Function.Name == "" {
				return 0, rejectBody("tool_choice is not supported")
			}
			total += compactLen(raw)
		}
	}
	return total, nil
}

func accountResponseFormat(raw json.RawMessage, a qualificationAccounting) (int, error) {
	if len(raw) == 0 {
		return 0, nil
	}
	var format struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Schema      json.RawMessage `json:"schema"`
			Strict      bool            `json:"strict"`
		} `json:"json_schema"`
	}
	if decodeExact(raw, &format) != nil {
		return 0, rejectBody("response_format has an unsupported shape")
	}
	switch {
	case format.Type == "text" && format.JSONSchema == nil:
		return 0, nil
	case format.Type == "json_object" && format.JSONSchema == nil:
		return a.SchemaOverheadTokens, nil
	case format.Type == "json_schema" && format.JSONSchema != nil && format.JSONSchema.Name != "" && isObject(format.JSONSchema.Schema):
		return a.SchemaOverheadTokens + len(format.JSONSchema.Name) + len(format.JSONSchema.Description) + compactLen(format.JSONSchema.Schema), nil
	}
	return 0, rejectBody("response_format is not supported")
}

// decodeExact decodes one JSON value, refusing fields the target does not name.
func decodeExact(raw json.RawMessage, v any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	return dec.Decode(v)
}

func isObject(raw json.RawMessage) bool {
	var obj map[string]json.RawMessage
	return json.Unmarshal(raw, &obj) == nil && obj != nil
}

// compactLen is the byte length of raw with insignificant whitespace removed.
func compactLen(raw json.RawMessage) int {
	var buf bytes.Buffer
	if json.Compact(&buf, raw) != nil {
		return len(raw)
	}
	return buf.Len()
}

// maxStrictBodyDepth bounds the nesting a strict request may use.
const maxStrictBodyDepth = 64

// uniqueKeys reports whether data is one JSON value, nested at most
// maxStrictBodyDepth deep, in which no object repeats a key at any depth, even
// in another letter case. A repeated key could be read differently by the
// engine than by this accounting, which decodes field names case-insensitively.
func uniqueKeys(data []byte) bool {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if !uniqueValue(dec, 0) {
		return false
	}
	_, err := dec.Token()
	return err == io.EOF
}

func uniqueValue(dec *json.Decoder, depth int) bool {
	tok, err := dec.Token()
	if err != nil {
		return false
	}
	d, isDelim := tok.(json.Delim)
	if !isDelim {
		return true
	}
	if depth >= maxStrictBodyDepth {
		return false
	}
	switch d {
	case '{':
		keys := map[string]bool{}
		for dec.More() {
			kt, err := dec.Token()
			key, isKey := kt.(string)
			// Upper then lower case folds the special cases (the Kelvin sign,
			// long s) the way JSON field matching does.
			folded := strings.ToLower(strings.ToUpper(key))
			if err != nil || !isKey || keys[folded] {
				return false
			}
			keys[folded] = true
			if !uniqueValue(dec, depth+1) {
				return false
			}
		}
	case '[':
		for dec.More() {
			if !uniqueValue(dec, depth+1) {
				return false
			}
		}
	default:
		return false
	}
	_, err = dec.Token()
	return err == nil
}
