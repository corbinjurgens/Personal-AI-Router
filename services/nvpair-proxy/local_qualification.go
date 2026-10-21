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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"nvpair-shared/engines"
	"nvpair-shared/nodepolicy"
)

// qualificationFileName is the artifact file under the per-user app data dir.
const qualificationFileName = "local-qualification.json"

// qualificationDocumentVersion is the artifact schema version.
const qualificationDocumentVersion = 2

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
	Shapes          *qualificationShapes    `json:"shapes"`
	Accounting      qualificationAccounting `json:"accounting"`
	Evidence        string                  `json:"evidence"`
}

// qualificationShapes declares evidence explicitly; nil members distinguish
// missing/null declarations from false and an empty list of qualified modes.
type qualificationShapes struct {
	Tools            *bool    `json:"tools"`
	StructuredOutput []string `json:"structuredOutput"`
	ReasoningReplay  *bool    `json:"reasoningReplay"`
}

func (s *qualificationShapes) validate() error {
	if s == nil {
		return errors.New("shapes must be a non-null object")
	}
	if s.Tools == nil {
		return errors.New("shapes.tools must be a non-null boolean")
	}
	if s.StructuredOutput == nil {
		return errors.New("shapes.structuredOutput must be a non-null array")
	}
	for i, mode := range s.StructuredOutput {
		if mode != "json_object" && mode != "json_schema" {
			return fmt.Errorf("unsupported shapes.structuredOutput mode %q", mode)
		}
		if slices.Contains(s.StructuredOutput[:i], mode) {
			return fmt.Errorf("repeated shapes.structuredOutput mode %q", mode)
		}
	}
	if s.ReasoningReplay == nil {
		return errors.New("shapes.reasoningReplay must be a non-null boolean")
	}
	if *s.ReasoningReplay {
		return errors.New("shapes.reasoningReplay must be false while reasoning replay is unsupported")
	}
	return nil
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
	if err := q.Shapes.validate(); err != nil {
		return err
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
	if !uniqueKeys(data) {
		return nil, errors.New("invalid qualification document: malformed JSON or repeated keys")
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
type strictQualificationError struct{ reason, message string }

func (e *strictQualificationError) Error() string {
	if e.message != "" {
		return e.message
	}
	return "strict qualification: " + e.reason
}

// errAmbiguousModel rejects an effective body whose defaults add a
// model-equivalent member; callers answer it as ambiguous-model, as on every
// other path.
var errAmbiguousModel = errors.New(ambiguousModelMessage)

func rejectBody(format string, args ...any) error {
	// Call sites format only fixed field names, fixed content diagnostics, and
	// integer bounds/indices. Their small output lives in the request's parser
	// allowance; avoid fmt's printer pool retaining it after the request ends.
	argumentSize := func(arg any) int {
		switch value := arg.(type) {
		case string:
			return len(value)
		case error:
			return len(value.Error())
		case int:
			var digits [20]byte
			return len(strconv.AppendInt(digits[:0], int64(value), 10))
		default:
			panic("unsupported strict diagnostic argument")
		}
	}
	size := len(format)
	for _, arg := range args {
		size += argumentSize(arg) - 2
	}
	out := make([]byte, 0, size)
	arg := 0
	for i := 0; i < len(format); i++ {
		if format[i] != '%' {
			out = append(out, format[i])
			continue
		}
		i++
		switch value := args[arg].(type) {
		case string:
			out = append(out, value...)
		case error:
			out = append(out, value.Error()...)
		case int:
			out = strconv.AppendInt(out, int64(value), 10)
		}
		arg++
	}
	reason := string(out)
	return &strictQualificationError{reason: reason, message: "strict qualification: " + reason}
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
func (q *localQualification) validateRequest(profile nodepolicy.Profile, route string, body []byte, contexts ...context.Context) ([]byte, strictBound, error) {
	var ctx context.Context
	if len(contexts) > 0 {
		ctx = contexts[0]
	}
	if !slices.Contains(q.Routes, route) {
		if err := retainStrictReason(ctx, len("route ")+len(route)+len(" is not qualified")); err != nil {
			return nil, strictBound{}, err
		}
		reason := "route " + route + " is not qualified"
		return nil, strictBound{}, &strictQualificationError{reason: reason, message: "strict qualification: " + reason}
	}
	unique, err := uniqueKeysContext(body, ctx)
	if err != nil {
		return nil, strictBound{}, err
	}
	if !unique {
		return nil, strictBound{}, rejectBody("the request repeats a key or is not JSON")
	}
	if !isJSONObject(body) {
		return nil, strictBound{}, rejectBody("the request must be a JSON object")
	}
	var fieldError error
	visitObject(body, func(key []byte, start, end int) {
		if fieldError != nil {
			return
		}
		if jsonStringEquals(key, "store") {
			fieldError = checkQualificationStore(body[start:end])
			return
		}
		for _, name := range chatRequestKeys {
			if jsonStringEquals(key, name) {
				return
			}
		}
		fieldError = rejectStrictField(ctx, key)
	})
	if fieldError != nil {
		return nil, strictBound{}, fieldError
	}
	effective, release, mergeErr := transformBody(ctx, body, "", false, profile.RequestOptions)
	if mergeErr != nil {
		return nil, strictBound{}, mergeErr
	}
	if ctx != nil && inferenceBuffers(ctx) != nil {
		inferenceBuffers(ctx).retainRelease(release)
	} else {
		defer release()
	}
	if ambiguousModelMember(effective) {
		return nil, strictBound{}, errAmbiguousModel
	}
	unique, err = uniqueKeysContext(effective, ctx)
	if err != nil {
		return nil, strictBound{}, err
	}
	if !unique {
		return nil, strictBound{}, rejectBody("the request with its defaults repeats a key or is not JSON")
	}
	if !isJSONObject(effective) {
		return nil, strictBound{}, rejectBody("the request with its defaults is not a JSON object")
	}
	effective, err = stripQualificationStore(ctx, effective)
	if err != nil {
		return nil, strictBound{}, err
	}
	w := newBodyWorkspace(ctx)
	defer w.close()
	var bound strictBound
	if err := checkChatScalars(effective, w); err != nil {
		return nil, strictBound{}, err
	}
	output, err := outputBound(effective)
	if err != nil {
		return nil, strictBound{}, err
	}
	if ceiling := min(profile.MaxOutputTokens, q.MaxOutputTokens); output > ceiling {
		return nil, strictBound{}, rejectBody("the output bound %d exceeds the qualified ceiling %d", output, ceiling)
	}
	bound.output = output
	a := q.Accounting
	bound.input = a.RequestOverheadTokens
	n, err := accountMessages(strictField(effective, "messages", false), a)
	if err != nil {
		return nil, strictBound{}, err
	}
	bound.input += n
	if n, err = accountTools(effective, a); err != nil {
		return nil, strictBound{}, err
	}
	bound.input += n
	if n, err = accountResponseFormat(strictField(effective, "response_format", false), a); err != nil {
		return nil, strictBound{}, err
	}
	bound.input += n
	if context := min(profile.ContextTokens, q.ContextTokens); bound.input+bound.output > context {
		return nil, strictBound{}, rejectBody("the input bound %d and output bound %d exceed the qualified context %d", bound.input, bound.output, context)
	}
	return effective, bound, nil
}

func checkQualificationStore(raw []byte) error {
	if bytes.Equal(raw, []byte("true")) {
		return rejectBody("store:true is not supported because PAIR does not store requests")
	}
	if !bytes.Equal(raw, []byte("false")) {
		return rejectBody("store must be the JSON boolean false")
	}
	return nil
}

// Remove only the validated no-op, after defaults have had their normal merge
// precedence. The new forwarding buffer stays charged until its borrowers join.
func stripQualificationStore(ctx context.Context, body []byte) ([]byte, error) {
	previousEnd := bytes.IndexByte(body, '{') + 1
	first := true
	removeStart, removeEnd := 0, 0
	var fieldError error
	visitObject(body, func(key []byte, start, end int) {
		if jsonStringEquals(key, "store") {
			fieldError = checkQualificationStore(body[start:end])
			removeStart, removeEnd = previousEnd, end
			if first {
				for removeEnd < len(body) && (body[removeEnd] == ' ' || body[removeEnd] == '\t' || body[removeEnd] == '\r' || body[removeEnd] == '\n') {
					removeEnd++
				}
				if removeEnd < len(body) && body[removeEnd] == ',' {
					removeEnd++
				}
			}
		}
		previousEnd, first = end, false
	})
	if fieldError != nil {
		return nil, fieldError
	}
	if removeEnd == 0 {
		return body, nil
	}
	size := len(body) - (removeEnd - removeStart)
	if ctx != nil {
		if owner := inferenceBuffers(ctx); owner != nil {
			lease, err := owner.scope.acquire(ctx, int64(size)+256)
			if err != nil {
				return nil, err
			}
			owner.retain(lease)
		}
	}
	out := make([]byte, 0, size)
	out = append(out, body[:removeStart]...)
	out = append(out, body[removeEnd:]...)
	return out, nil
}

// Strict request views borrow the effective body. They reproduce struct
// decoding's case-insensitive field matching and null zero values without
// allocating strings, raw-message copies, arrays, maps, or decoder buffers.
func strictField(raw []byte, name string, fold bool) []byte {
	var result []byte
	visitObject(raw, func(key []byte, start, end int) {
		if strictKeyEqual(key, name, fold) {
			result = raw[start:end]
		}
	})
	return result
}

func strictKeyEqual(raw []byte, name string, fold bool) bool {
	if !fold {
		return jsonStringEquals(raw, name)
	}
	pos := 0
	for i := 1; i < len(raw)-1; {
		if pos == len(name) {
			return false
		}
		r, next := ownedJSONRune(raw, i)
		other, n := utf8.DecodeRuneInString(name[pos:])
		if strictFoldRune(r) != strictFoldRune(other) {
			return false
		}
		i, pos = next, pos+n
	}
	return pos == len(name)
}

func strictFoldRune(r rune) rune {
	for {
		next := unicode.SimpleFold(r)
		if next <= r {
			return next
		}
		r = next
	}
}

func strictNull(raw []byte) bool { return bytes.Equal(raw, []byte("null")) }

func strictStringLen(raw []byte) (int, bool) {
	if len(raw) == 0 || strictNull(raw) {
		return 0, true
	}
	if raw[0] != '"' {
		return 0, false
	}
	return decodedJSONStringBytes(raw), true
}

func strictBool(raw []byte) bool {
	return len(raw) == 0 || strictNull(raw) || bytes.Equal(raw, []byte("true")) || bytes.Equal(raw, []byte("false"))
}

func strictObject(raw []byte, names ...string) bool {
	if len(raw) == 0 || strictNull(raw) {
		return true
	}
	valid := true
	_, object := visitObject(raw, func(key []byte, _, _ int) {
		for _, name := range names {
			if strictKeyEqual(key, name, true) {
				return
			}
		}
		valid = false
	})
	return object && valid
}

func strictStrings(raw []byte, names ...string) bool {
	for _, name := range names {
		if _, ok := strictStringLen(strictField(raw, name, true)); !ok {
			return false
		}
	}
	return true
}

func strictArray(raw []byte, visit func(int, []byte) error) (int, error) {
	if len(raw) == 0 || strictNull(raw) {
		return 0, nil
	}
	n := 0
	var err error
	if !visitArray(raw, func(value []byte) {
		if err == nil && visit != nil {
			err = visit(n, value)
		}
		n++
	}) {
		return 0, errors.New("not an array")
	}
	return n, err
}

func checkChatScalars(f []byte, w *bodyWorkspace) error {
	if n, ok := strictStringLen(strictField(f, "model", false)); !ok || n == 0 {
		return rejectBody("model must be a nonempty string")
	}
	for _, key := range []string{"temperature", "top_p", "frequency_penalty", "presence_penalty"} {
		if raw := strictField(f, key, false); len(raw) > 0 && !strictNull(raw) {
			if raw[0] != '-' && (raw[0] < '0' || raw[0] > '9') {
				return rejectBody("%s must be a number", key)
			}
			// ParseFloat's input conversion and its possible NumError clone
			// each need len(raw) bytes. Its decimal workspace is fixed-size.
			if err := w.acquire(2*int64(len(raw)) + 256); err != nil {
				return err
			}
			if _, err := strconv.ParseFloat(string(raw), 64); err != nil {
				return rejectBody("%s must be a number", key)
			}
		}
	}
	if raw := strictField(f, "seed", false); len(raw) > 0 && !strictNull(raw) && !strictInt64(raw) {
		return rejectBody("seed must be an integer")
	}
	if !strictBool(strictField(f, "stream", false)) {
		return rejectBody("stream must be a boolean")
	}
	if raw := strictField(f, "stream_options", false); len(raw) > 0 {
		if !strictObject(raw, "include_usage") || !strictBool(strictField(raw, "include_usage", true)) {
			return rejectBody("stream_options supports include_usage only")
		}
	}
	if raw := strictField(f, "stop", false); len(raw) > 0 {
		if _, ok := strictStringLen(raw); !ok {
			n, err := strictArray(raw, func(_ int, value []byte) error {
				if _, ok := strictStringLen(value); !ok {
					return errors.New("not a string")
				}
				return nil
			})
			if err != nil || n > 4 {
				return rejectBody("stop must be a string or at most four strings")
			}
		}
	}
	return nil
}

func strictInt64(raw []byte) bool {
	negative := raw[0] == '-'
	if negative {
		raw = raw[1:]
	}
	limit := uint64(1<<63 - 1)
	if negative {
		limit++
	}
	var n uint64
	for _, c := range raw {
		if c < '0' || c > '9' || n > (limit-uint64(c-'0'))/10 {
			return false
		}
		n = n*10 + uint64(c-'0')
	}
	return len(raw) > 0
}

// outputBound is the single finite output request. Both spellings must agree
// when both are sent; a request with neither has no bound and is rejected.
func outputBound(f []byte) (int, error) {
	bound := 0
	for _, key := range []string{"max_tokens", "max_completion_tokens"} {
		raw := strictField(f, key, false)
		if len(raw) == 0 {
			continue
		}
		v, ok := positiveJSONInt(raw)
		if !ok {
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

func accountMessages(raw json.RawMessage, a qualificationAccounting) (int, error) {
	total := 0
	count, err := strictArray(raw, func(i int, m []byte) error {
		calls := strictField(m, "tool_calls", true)
		if !strictObject(m, "role", "content", "name", "tool_calls", "tool_call_id") || !strictStrings(m, "role", "name", "tool_call_id") {
			return rejectBody("messages[%d] has an unsupported shape", i)
		}
		callCount, callErr := strictArray(calls, nil)
		if callErr != nil {
			return rejectBody("messages[%d] has an unsupported shape", i)
		}
		role := strictField(m, "role", true)
		if !jsonStringEquals(role, "system") && !jsonStringEquals(role, "user") && !jsonStringEquals(role, "assistant") && !jsonStringEquals(role, "tool") {
			return rejectBody("messages[%d].role is not supported", i)
		}
		roleBytes, _ := strictStringLen(role)
		n := a.MessageOverheadTokens + roleBytes
		text, isNull, err := contentText(strictField(m, "content", true))
		if err != nil {
			return rejectBody("messages[%d].content: %s", i, err)
		}
		if isNull && (!jsonStringEquals(role, "assistant") || callCount == 0) {
			return rejectBody("messages[%d].content is required", i)
		}
		n += text
		name, _ := strictStringLen(strictField(m, "name", true))
		n += name
		if id := strictField(m, "tool_call_id", true); len(id) > 0 && !strictNull(id) {
			if !jsonStringEquals(role, "tool") {
				return rejectBody("messages[%d].tool_call_id belongs to a tool message", i)
			}
			idBytes, _ := strictStringLen(id)
			n += idBytes
		}
		if callCount > 0 && !jsonStringEquals(role, "assistant") {
			return rejectBody("messages[%d].tool_calls belong to an assistant message", i)
		}
		_, err = strictArray(calls, func(j int, call []byte) error {
			fn := strictField(call, "function", true)
			name := strictField(fn, "name", true)
			nameBytes, nameOK := strictStringLen(name)
			if !strictObject(call, "id", "type", "function") || !strictStrings(call, "id", "type") || !strictObject(fn, "name", "arguments") || !strictStrings(fn, "name", "arguments") || !jsonStringEquals(strictField(call, "type", true), "function") || !nameOK || nameBytes == 0 {
				return rejectBody("messages[%d].tool_calls[%d] must call a named function", i, j)
			}
			n += compactLen(call)
			return nil
		})
		if err != nil {
			return err
		}
		total += n
		return nil
	})
	if err != nil {
		if len(raw) == 0 || strictNull(raw) || raw[0] != '[' {
			return 0, rejectBody("messages must be a nonempty array")
		}
		return 0, err
	}
	if count == 0 {
		return 0, rejectBody("messages must be a nonempty array")
	}
	return total, nil
}

// contentText counts decoded UTF-8 bytes directly in borrowed strings.
func contentText(raw json.RawMessage) (n int, isNull bool, err error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || strictNull(raw) {
		return 0, true, nil
	}
	if raw[0] == '"' {
		return decodedJSONStringBytes(raw), false, nil
	}
	_, err = strictArray(raw, func(_ int, part []byte) error {
		if !strictObject(part, "type", "text") || !strictStrings(part, "type", "text") || !jsonStringEquals(strictField(part, "type", true), "text") {
			return errors.New("only text parts are supported")
		}
		text, _ := strictStringLen(strictField(part, "text", true))
		n += text
		return nil
	})
	if raw[0] != '[' {
		return 0, false, errors.New("must be a string or an array of text parts")
	}
	return n, false, err
}

func accountTools(f []byte, a qualificationAccounting) (int, error) {
	total := 0
	if raw := strictField(f, "tools", false); len(raw) > 0 {
		_, err := strictArray(raw, func(i int, tool []byte) error {
			fn := strictField(tool, "function", true)
			name, nameOK := strictStringLen(strictField(fn, "name", true))
			if !strictObject(tool, "type", "function") || !strictStrings(tool, "type") || !strictObject(fn, "name", "description", "parameters", "strict") || !strictStrings(fn, "name", "description") || !strictBool(strictField(fn, "strict", true)) || !jsonStringEquals(strictField(tool, "type", true), "function") || !nameOK || name == 0 {
				return rejectBody("tools[%d] must be a named function", i)
			}
			if parameters := strictField(fn, "parameters", true); len(parameters) > 0 && !isJSONObject(parameters) {
				return rejectBody("tools[%d].function.parameters must be an object", i)
			}
			total += a.ToolOverheadTokens + compactLen(tool)
			return nil
		})
		if err != nil {
			if raw[0] != '[' && !strictNull(raw) {
				return 0, rejectBody("tools must be an array")
			}
			return 0, err
		}
	}
	if raw := strictField(f, "tool_choice", false); len(raw) > 0 {
		if raw[0] == '"' || strictNull(raw) {
			if !jsonStringEquals(raw, "none") && !jsonStringEquals(raw, "auto") && !jsonStringEquals(raw, "required") {
				return 0, rejectBody("tool_choice is not supported")
			}
			n, _ := strictStringLen(raw)
			total += n
		} else {
			fn := strictField(raw, "function", true)
			name, nameOK := strictStringLen(strictField(fn, "name", true))
			if !strictObject(raw, "type", "function") || !strictStrings(raw, "type") || !strictObject(fn, "name") || !strictStrings(fn, "name") || !jsonStringEquals(strictField(raw, "type", true), "function") || !nameOK || name == 0 {
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
	schema := strictField(raw, "json_schema", true)
	if !strictObject(raw, "type", "json_schema") || !strictStrings(raw, "type") || !strictObject(schema, "name", "description", "schema", "strict") || !strictStrings(schema, "name", "description") || !strictBool(strictField(schema, "strict", true)) {
		return 0, rejectBody("response_format has an unsupported shape")
	}
	typ := strictField(raw, "type", true)
	noSchema := len(schema) == 0 || strictNull(schema)
	switch {
	case jsonStringEquals(typ, "text") && noSchema:
		return 0, nil
	case jsonStringEquals(typ, "json_object") && noSchema:
		return a.SchemaOverheadTokens, nil
	case jsonStringEquals(typ, "json_schema") && !noSchema:
		name, _ := strictStringLen(strictField(schema, "name", true))
		if name > 0 && isJSONObject(strictField(schema, "schema", true)) {
			description, _ := strictStringLen(strictField(schema, "description", true))
			return a.SchemaOverheadTokens + name + description + compactLen(strictField(schema, "schema", true)), nil
		}
	}
	return 0, rejectBody("response_format is not supported")
}

// compactLen counts bytes without a buffer or encoding/json's retained pool.
// Request callers have already validated syntax; the public helper preserves
// the former invalid-input result for its fixture callers.
func compactLen(raw json.RawMessage) int {
	if !ownedJSONValid(raw) {
		return len(raw)
	}
	n := 0
	for i := 0; i < len(raw); {
		if raw[i] == '"' {
			end, _ := ownedJSONStringEnd(raw, i)
			n += end - i
			i = end
			continue
		}
		if raw[i] != ' ' && raw[i] != '\t' && raw[i] != '\r' && raw[i] != '\n' {
			n++
		}
		i++
	}
	return n
}

const maxStrictBodyDepth = 64

func uniqueKeys(data []byte) bool {
	ok, _ := uniqueKeysContext(data, nil)
	return ok
}

// The scan has fixed parser state. Each object's exact member/key storage is
// acquired before allocation; sorting in place needs no map or folded strings.
// All workspace leases remain held until this scan has dropped its views.
func uniqueKeysContext(data []byte, ctx context.Context) (bool, error) {
	if !ownedJSONValid(data) {
		return false, nil
	}
	w := newBodyWorkspace(ctx)
	defer w.close()
	depth := 0
	for i := 0; i < len(data); i++ {
		switch data[i] {
		case '"':
			end, _ := ownedJSONStringEnd(data, i)
			i = end - 1
		case '{', '[':
			if depth == maxStrictBodyDepth {
				return false, nil
			}
			depth++
			if data[i] == '{' {
				members, _, _, err := w.members(data[i:])
				if err != nil {
					return false, err
				}
				slices.SortFunc(members, func(a, b jsonMember) int { return compareStrictKeys(a.key, b.key) })
				for j := 1; j < len(members); j++ {
					if compareStrictKeys(members[j-1].key, members[j].key) == 0 {
						return false, nil
					}
				}
			}
		case '}', ']':
			depth--
		}
	}
	return true, nil
}

// Preserve the former strings.ToLower(strings.ToUpper(key)) duplicate rule,
// including its Unicode cases, without materializing either folded string.
func compareStrictKeys(a, b string) int {
	for len(a) > 0 && len(b) > 0 {
		x, xn := utf8.DecodeRuneInString(a)
		y, yn := utf8.DecodeRuneInString(b)
		x = unicode.ToLower(unicode.ToUpper(x))
		y = unicode.ToLower(unicode.ToUpper(y))
		if x < y {
			return -1
		}
		if x > y {
			return 1
		}
		a, b = a[xn:], b[yn:]
	}
	return len(a) - len(b)
}

// Dynamic qualification reasons survive the local scan. Charge their exact
// construction capacity, retained string, and Error() prefix copy through the
// request owner. The byte construction has an explicit capacity, avoiding a
// strings.Builder's allocator-rounded spare capacity.
func retainStrictReason(ctx context.Context, size int) error {
	if ctx == nil {
		return nil
	}
	owner := inferenceBuffers(ctx)
	if owner == nil {
		return nil
	}
	lease, err := owner.scope.acquire(ctx, 3*int64(size)+int64(len("strict qualification: "))+256)
	if err != nil {
		return err
	}
	owner.retain(lease)
	return nil
}

func strictQuoteRuneSize(r rune) int {
	switch r {
	case '\a', '\b', '\f', '\n', '\r', '\t', '\v', '\\', '"':
		return 2
	}
	if r < ' ' || r == 0x7f {
		return 4
	}
	if strconv.IsPrint(r) {
		return utf8.RuneLen(r)
	}
	if r < 0x10000 {
		return 6
	}
	return 10
}

func rejectStrictField(ctx context.Context, raw []byte) error {
	size := len("field \"\" is not supported")
	for i := 1; i < len(raw)-1; {
		r, next := ownedJSONRune(raw, i)
		size += strictQuoteRuneSize(r)
		i = next
	}
	if err := retainStrictReason(ctx, size); err != nil {
		return err
	}
	b := make([]byte, 0, size)
	b = append(b, "field \""...)
	const hex = "0123456789abcdef"
	for i := 1; i < len(raw)-1; {
		r, next := ownedJSONRune(raw, i)
		i = next
		escape := byte(0)
		switch r {
		case '\a':
			escape = 'a'
		case '\b':
			escape = 'b'
		case '\f':
			escape = 'f'
		case '\n':
			escape = 'n'
		case '\r':
			escape = 'r'
		case '\t':
			escape = 't'
		case '\v':
			escape = 'v'
		case '\\', '"':
			escape = byte(r)
		}
		if escape != 0 {
			b = append(b, '\\', escape)
		} else if strconv.IsPrint(r) {
			b = utf8.AppendRune(b, r)
		} else {
			b = append(b, '\\')
			digits := 4
			if r < ' ' || r == 0x7f {
				b = append(b, 'x')
				digits = 2
			} else if r < 0x10000 {
				b = append(b, 'u')
			} else {
				b = append(b, 'U')
				digits = 8
			}
			for shift := (digits - 1) * 4; shift >= 0; shift -= 4 {
				b = append(b, hex[(r>>shift)&15])
			}
		}
	}
	b = append(b, "\" is not supported"...)
	reason := string(b)
	return &strictQualificationError{reason: reason, message: "strict qualification: " + reason}
}
