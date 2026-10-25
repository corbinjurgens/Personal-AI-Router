// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"hash"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"nvpair-shared/performancewire"
)

const performanceQueueCapacity = 128

// The FIFO lock never covers codec IO. Counters saturate and have a fixed set
// of reasons; neither pressure nor missing identity allocates drop records.
// This existing fixed-capacity queue is scalar control storage, outside the
// request-byte account. Insertion owns fresh nullable scalar copies and small
// copies of every bounded identifier, including normalized views. Entries never
// alias reader/request/frame/pooled storage. No inference content or additional
// variable-size payload enters this exemption.
type performanceQueue struct {
	mu         sync.Mutex
	items      [performanceQueueCapacity]performancewire.Sample
	head, size int
	seq        uint64
	drops      [4]uint64 // full, identity/options, stale identity, invalid sample
	wake       chan struct{}
}

func (p *Proxy) dropPerformance(reason int) {
	q := &p.performance
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.drops[reason] != ^uint64(0) {
		q.drops[reason]++
	}
}

func (p *Proxy) enqueuePerformance(s performancewire.Sample) {
	q := &p.performance
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.size == len(q.items) || q.seq == ^uint64(0) {
		if q.drops[0] != ^uint64(0) {
			q.drops[0]++
		}
		return
	}
	s.Scope, s.Producer, s.Seq = performancewire.ScopeLocalExecution, p.runID, q.seq+1
	if s.Validate() != nil {
		if q.drops[3] != ^uint64(0) {
			q.drops[3]++
		}
		return
	}
	q.seq++
	s.Producer = strings.Clone(s.Producer)
	s.Key.Host = strings.Clone(s.Key.Host)
	s.Key.Engine = strings.Clone(s.Key.Engine)
	s.Key.Model = strings.Clone(s.Key.Model)
	s.Key.Fingerprint = strings.Clone(s.Key.Fingerprint)
	s.Profile.EngineFamily = strings.Clone(s.Profile.EngineFamily)
	s.Profile.Quantization = strings.Clone(s.Profile.Quantization)
	s.Profile.ContextTokens = performanceScalarCopy(s.Profile.ContextTokens)
	s.Outcome = performancewire.Outcome(strings.Clone(string(s.Outcome)))
	s.ProtocolComplete = performanceScalarCopy(s.ProtocolComplete)
	s.AdmissionWaitMs = performanceScalarCopy(s.AdmissionWaitMs)
	s.HeaderMs = performanceScalarCopy(s.HeaderMs)
	s.FirstBodyMs = performanceScalarCopy(s.FirstBodyMs)
	s.FirstTokenMs = performanceScalarCopy(s.FirstTokenMs)
	s.TotalMs = performanceScalarCopy(s.TotalMs)
	s.PromptTokens = performanceScalarCopy(s.PromptTokens)
	s.CachedPromptTokens = performanceScalarCopy(s.CachedPromptTokens)
	s.OutputTokens = performanceScalarCopy(s.OutputTokens)
	s.PrefillNs = performanceScalarCopy(s.PrefillNs)
	s.DecodeNs = performanceScalarCopy(s.DecodeNs)
	s.EngineDecodeNs = performanceScalarCopy(s.EngineDecodeNs)
	s.EngineDecodeTokensPerSecond = performanceScalarCopy(s.EngineDecodeTokensPerSecond)
	s.LoadNs = performanceScalarCopy(s.LoadNs)
	q.items[(q.head+q.size)%len(q.items)] = s
	q.size++
	select {
	case q.wake <- struct{}{}:
	default:
	}
}

func performanceScalarCopy[T int64 | float64 | bool](value *T) *T {
	if value == nil {
		return nil
	}
	v := *value
	return &v
}

func (p *Proxy) sendPerformance(ctx context.Context) {
	q := &p.performance
	for {
		if ctx.Err() != nil {
			return
		}
		q.mu.Lock()
		if q.size == 0 {
			q.mu.Unlock()
			select {
			case <-ctx.Done():
				return
			case <-q.wake:
			}
			continue
		}
		s := q.items[q.head]
		q.items[q.head] = performancewire.Sample{}
		q.head = (q.head + 1) % len(q.items)
		q.size--
		q.mu.Unlock()
		// Cancellation covers the shared write gate. The transport owner
		// interrupts an active write and retires the parent session.
		if ctx.Err() != nil {
			return
		}
		_ = p.codec.NotifyContext(ctx, "performance:sample", s)
	}
}

// performanceFingerprint is the only serving-profile hash in the proxy. The
// joined identity is read under its owner's lock and reduced to this model's
// digest. No filename, alias, response model or placeholder supplies identity.
func (f *facade) performanceFingerprint(model, api string, body []byte) (performancewire.ProfileKey, performanceExtractQualification, bool) {
	key, qualification, _, ok := f.performanceFingerprintOwned(model, api, body)
	return key, qualification, ok
}

func (f *facade) performanceFamily() string {
	family := f.profile.Family
	if family == "" {
		family = f.profile.Name
	}
	return family
}

type performanceIdentityTuple struct{ run, settings, launch, model, api, schema, host, engine, normalizedModel string }

func (f *facade) performanceCurrentIdentity(model, api string) (performancewire.ProfileKey, performanceExtractQualification, performanceIdentityTuple, bool) {
	var tuple performanceIdentityTuple
	family := f.performanceFamily()
	q := performanceExtractQualification{EngineFamily: family, APIPath: api, Streaming: true}
	switch family {
	case "llamacpp":
		q.SchemaID = performanceExtractLlamaSchema
	case "strata":
		q.SchemaID = performanceExtractStrataSchema
	case "ollama":
	default:
		return performancewire.ProfileKey{}, q, tuple, false
	}
	a := f.host.admission
	a.mu.Lock()
	id, exists := a.strict.identity[f.profile.Name]
	digest, found := a.modelDigestLocked(id, admKey{engine: f.profile.Name, model: f.profile.normalizeModel(model)})
	valid := exists && found && id.Validate() == nil
	// Copy only immutable scalars; the projection's map never escapes its lock.
	run, settings, launch := id.EngineRunID, id.SettingsFingerprint, id.LaunchDigest
	a.mu.Unlock()
	host, _ := f.host.selfIdentity(f.profile.Name)
	if !valid || host == "" {
		return performancewire.ProfileKey{}, q, tuple, false
	}
	tuple = performanceIdentityTuple{run, settings, launch, digest, api, family + ":" + q.SchemaID, host, f.profile.Name, f.profile.normalizeModel(model)}
	key := performancewire.ProfileKey{Host: host, Engine: tuple.engine, Model: tuple.normalizedModel}
	return key, q, tuple, true
}

func (f *facade) performanceFingerprintOwned(model, api string, body []byte) (performancewire.ProfileKey, performanceExtractQualification, performanceIdentityTuple, bool) {
	key, q, tuple, ok := f.performanceCurrentIdentity(model, api)
	if !ok {
		return key, q, tuple, false
	}
	h := sha256.New()
	for i, field := range []struct{ name, value string }{{"Run", tuple.run}, {"Settings", tuple.settings}, {"Launch", tuple.launch}, {"Model", tuple.model}, {"API", tuple.api}, {"Schema", tuple.schema}} {
		if i == 0 {
			h.Write([]byte("{"))
		} else {
			h.Write([]byte(","))
		}
		performanceHashString(h, []byte(field.name))
		h.Write([]byte(":"))
		performanceHashString(h, []byte(field.value))
	}
	h.Write([]byte(`,"Options":`))
	if !performanceOptionsWrite(body, q.EngineFamily, h) {
		return key, q, tuple, false
	}
	h.Write([]byte("}"))
	key.Fingerprint = hex.EncodeToString(h.Sum(nil))
	return key, q, tuple, key.Validate() == nil
}

// Canonicalize the effective post-rewrite/default options, excluding inference
// content. Unknown top-level overrides suppress observation. Numeric spellings
// normalize exactly, including integers above float64's exact range.
type performanceJSONNode struct {
	key, raw, text       []byte
	parent, begin, count int
	kind                 byte
}

type performancePreparationKey struct{}
type performancePreparationMode int

const (
	performancePrepareLocalNumeric performancePreparationMode = iota + 1
	performancePrepareLocalSyntax
	performancePreparePeer
)

// Qualification is static: missing live identity may suppress a sample, but
// must not change the parser storage admitted for this engine-facing API.
func (f *facade) performancePreparationMode(api string) performancePreparationMode {
	switch f.performanceFamily() {
	case "ollama":
		switch api {
		case "/api/chat", "/api/generate", "/v1/chat/completions", "/v1/completions":
			return performancePrepareLocalNumeric
		}
	case "llamacpp", "strata":
		if api == "/v1/chat/completions" {
			return performancePrepareLocalNumeric
		}
	}
	return performancePrepareLocalSyntax
}

type performancePreparation struct {
	bytes  int64
	frames *frameBudget
	mode   performancePreparationMode
}

// Transfer only the admitted allowance; temporary preparation cancellation
// must not become cancellation of the accepted transport or its reader.
func rebasePerformancePreparation(live, prepared context.Context) context.Context {
	if allowance, ok := prepared.Value(performancePreparationKey{}).(performancePreparation); ok {
		return context.WithValue(live, performancePreparationKey{}, allowance)
	}
	return live
}

// One descriptor/index slot per lexical token (including keys) is a deliberate
// upper capacity, not a body multiplier. A node is <=104 bytes on 64-bit Go;
// 128 plus an 8-byte index covers either supported word size. Decoded strings
// share one exact arena. A reusable borrowed-key arena covers the largest
// current object, including ignored content. Four runtime pages cover rounding.
func performancePreparationBytes(body []byte) (int64, error) {
	if !ownedJSONValid(body) {
		return 0, nil // malformed/unsupported options suppress observation
	}
	tokens, strings := performanceJSONMeasure(body)
	if tokens == 0 {
		return 0, nil
	}
	members, ok := performanceUniqueMembers(body)
	if !ok {
		return 0, nil
	}
	return 136*int64(tokens) + int64(strings) + 24*int64(members) + 4*8192, nil
}

func preparePerformanceBuffers(ctx context.Context, body []byte, request originStreamRequest, mode performancePreparationMode) (context.Context, func(), error) {
	if int64(request.choices) > streamFrameMaxChoices {
		return ctx, nil, &bufferLimitError{}
	}
	owner := inferenceBuffers(ctx)
	if owner == nil {
		return ctx, func() {}, nil
	}
	var fixed, storage *bufferLease
	var frames *frameBudget
	var n int64
	release := sync.OnceFunc(func() {
		if frames != nil {
			frames.releasePreparation()
		}
		if storage != nil {
			storage.close()
		}
		if fixed != nil {
			fixed.close()
		}
	})
	if mode != performancePreparePeer {
		var err error
		fixed, err = owner.scope.acquire(ctx, streamFrameNumericBytes)
		if err != nil {
			return ctx, nil, err
		}
		n, err = performancePreparationBytes(body)
		if err == nil {
			storage, err = owner.scope.acquire(ctx, n)
		}
		if err != nil {
			release()
			return ctx, nil, err
		}
	}
	// OpenAI's omitted/null stream flag defaults to nonstreaming. An unusual
	// SSE response is still detected later, but must acquire fresh frame grants.
	if request.disabled || !request.declared || (request.protocol != streamProtocolSSE && request.protocol != streamProtocolNDJSON) {
		return context.WithValue(ctx, performancePreparationKey{}, performancePreparation{bytes: n, mode: mode}), release, nil
	}
	// Response working storage is distinct from the request parser and the
	// fingerprint allowance: both can remain live while the response arrives.
	frames = ownedFrameBudget(ctx)
	if frames == nil {
		release()
		return ctx, nil, errStreamFrameBudget
	}
	parserBytes := int64(streamFrameSyntaxBytes)
	if mode == performancePrepareLocalNumeric {
		parserBytes = streamFrameNumericBytes
	}
	choices := max(request.choices, 1)
	if err := frames.admit(ctx, streamFrameAllowance(request.protocol, int64(choices), parserBytes)); err != nil {
		release()
		return ctx, nil, err
	}
	for _, size := range []int64{streamFrameMetadataBytes, int64(choices), streamFrameReadChunk, parserBytes, streamFrameReadChunk} {
		if err := frames.prepare(ctx, size); err != nil {
			release()
			return ctx, nil, err
		}
	}
	if request.protocol == streamProtocolSSE {
		if err := frames.prepare(ctx, streamFrameReadChunk); err != nil {
			release()
			return ctx, nil, err
		}
	}
	return context.WithValue(ctx, performancePreparationKey{}, performancePreparation{bytes: n, frames: frames, mode: mode}), release, nil
}

func performanceJSONMeasure(body []byte) (tokens, decoded int) {
	valid := true
	if _, ok := visitObject(body, func(key []byte, _, _ int) {
		if !performanceKnownOptionKey(key) {
			valid = false
		}
	}); !ok || !valid {
		return 0, 0
	}
	var measure func([]byte, []byte, bool, int)
	measure = func(raw, key []byte, observe bool, depth int) {
		raw = bytes.TrimSpace(raw)
		if depth > maxStrictBodyDepth || depth == maxStrictBodyDepth && (raw[0] == '{' || raw[0] == '[') {
			valid = false
			return
		}
		tokens++
		if key != nil {
			tokens++
			decoded += decodedJSONStringBytes(key)
		}
		if observe && raw[0] == '"' {
			decoded += decodedJSONStringBytes(raw)
		}
		if !observe {
			return
		}
		switch raw[0] {
		case '{':
			visitObject(raw, func(k []byte, start, end int) {
				measure(raw[start:end], k, observe && (depth != 0 || !performanceIgnoredKey(k)), depth+1)
			})
		case '[':
			visitArray(raw, func(value []byte) { measure(value, nil, observe, depth+1) })
		}
	}
	measure(body, nil, true, 0)
	if !valid {
		return 0, 0
	}
	return
}

// Only one object's borrowed-key descriptors are live at a time. Ignored
// messages/tools never acquire a descriptor for every nested scalar/value.
func performanceUniqueMembers(body []byte) (int, bool) {
	maximum, depth := 0, 0
	for i := 0; i < len(body); i++ {
		switch body[i] {
		case '"':
			end, _ := ownedJSONStringEnd(body, i)
			i = end - 1
		case '{', '[':
			if depth == maxStrictBodyDepth {
				return 0, false
			}
			depth++
			if body[i] == '{' {
				count := 0
				visitObject(body[i:], func([]byte, int, int) { count++ })
				if count > maximum {
					maximum = count
				}
			}
		case '}', ']':
			depth--
		}
	}
	return maximum, true
}

func performanceUniqueKeys(body []byte, maximum int) bool {
	keys := make([][]byte, 0, maximum)
	for i := 0; i < len(body); i++ {
		switch body[i] {
		case '"':
			end, _ := ownedJSONStringEnd(body, i)
			i = end - 1
		case '{':
			keys = keys[:0]
			visitObject(body[i:], func(key []byte, _, _ int) { keys = append(keys, key) })
			slices.SortFunc(keys, performanceRawKeyCompare)
			for j := 1; j < len(keys); j++ {
				if performanceRawKeyCompare(keys[j-1], keys[j]) == 0 {
					return false
				}
			}
		}
	}
	return true
}

func performanceRawKeyCompare(a, b []byte) int {
	i, j := 1, 1
	for i < len(a)-1 && j < len(b)-1 {
		x, nextI := ownedJSONRune(a, i)
		y, nextJ := ownedJSONRune(b, j)
		x = unicode.ToLower(unicode.ToUpper(x))
		y = unicode.ToLower(unicode.ToUpper(y))
		if x < y {
			return -1
		}
		if x > y {
			return 1
		}
		i, j = nextI, nextJ
	}
	return (len(a) - 1 - i) - (len(b) - 1 - j)
}

func performanceKnownOptionKey(key []byte) bool {
	for _, name := range chatRequestKeys {
		if jsonStringEquals(key, name) {
			return true
		}
	}
	for _, name := range []string{"prompt", "input", "n", "options", "keep_alive", "format", "raw", "think"} {
		if jsonStringEquals(key, name) {
			return true
		}
	}
	return false
}

func performanceIgnoredKey(key []byte) bool {
	return jsonStringEquals(key, "model") || jsonStringEquals(key, "messages") || jsonStringEquals(key, "prompt") || jsonStringEquals(key, "input") || jsonStringEquals(key, "tools")
}

func performanceOptionsDigest(body []byte, family string) ([32]byte, bool) {
	var zero [32]byte
	h := sha256.New()
	if !performanceOptionsWrite(body, family, h) {
		return zero, false
	}
	copy(zero[:], h.Sum(nil))
	return zero, true
}

func performanceOptionsWrite(body []byte, family string, h hash.Hash) bool {
	if !ownedJSONValid(body) {
		return false
	}
	tokens, decoded := performanceJSONMeasure(body)
	if tokens == 0 {
		return false
	}
	maximum, ok := performanceUniqueMembers(body)
	if !ok || !performanceUniqueKeys(body, maximum) {
		return false
	}
	nodes := make([]performanceJSONNode, 0, tokens)
	text := make([]byte, 0, decoded)
	var build func([]byte, []byte, int, int, bool) int
	build = func(raw, key []byte, parent, depth int, observe bool) int {
		raw = bytes.TrimSpace(raw)
		if depth > maxStrictBodyDepth || depth == maxStrictBodyDepth && (raw[0] == '{' || raw[0] == '[') {
			return -1
		}
		idx := len(nodes)
		nodes = append(nodes, performanceJSONNode{raw: raw, parent: parent, kind: raw[0]})
		decode := func(quoted []byte) []byte {
			start := len(text)
			for i := 1; i < len(quoted)-1; {
				r, next := ownedJSONRune(quoted, i)
				text = utf8.AppendRune(text, r)
				i = next
			}
			return text[start:len(text)]
		}
		if key != nil {
			nodes[idx].key = decode(key)
		}
		if observe && raw[0] == '"' {
			nodes[idx].text = decode(raw)
		}
		if !observe {
			return idx
		}
		valid := true
		switch raw[0] {
		case '{':
			visitObject(raw, func(k []byte, start, end int) {
				if build(raw[start:end], k, idx, depth+1, observe && (depth != 0 || !performanceIgnoredKey(k))) < 0 {
					valid = false
				}
			})
		case '[':
			visitArray(raw, func(value []byte) {
				if build(value, nil, idx, depth+1, observe) < 0 {
					valid = false
				}
			})
		}
		if !valid {
			return -1
		}
		return idx
	}
	if build(body, nil, -1, 0, true) < 0 || len(nodes) == 0 || nodes[0].kind != '{' {
		return false
	}
	order := make([]int, len(nodes)-1)
	for i := range order {
		order[i] = i + 1
	}
	compare := func(a, b int, fold bool) int {
		if nodes[a].parent != nodes[b].parent {
			return nodes[a].parent - nodes[b].parent
		}
		if nodes[nodes[a].parent].kind != '{' {
			return a - b
		}
		if fold {
			return performanceFoldCompare(nodes[a].key, nodes[b].key)
		}
		ak, bk := nodes[a].key, nodes[b].key
		if nodes[a].parent == 0 {
			if bytes.Equal(ak, []byte("tools")) {
				ak = []byte("hasTools")
			}
			if bytes.Equal(bk, []byte("tools")) {
				bk = []byte("hasTools")
			}
		}
		return bytes.Compare(ak, bk)
	}
	slices.SortFunc(order, func(a, b int) int { return compare(a, b, true) })
	for i := 1; i < len(order); i++ {
		if compare(order[i-1], order[i], true) == 0 {
			return false
		}
	}
	slices.SortFunc(order, func(a, b int) int { return compare(a, b, false) })
	for i, idx := range order {
		p := nodes[idx].parent
		if nodes[p].count == 0 {
			nodes[p].begin = i
		}
		nodes[p].count++
	}
	var emit func(int)
	emit = func(idx int) {
		n := nodes[idx]
		switch n.kind {
		case '{', '[':
			h.Write([]byte{n.kind})
			for i := 0; i < n.count; i++ {
				if i > 0 {
					h.Write([]byte(","))
				}
				child := order[n.begin+i]
				if n.kind == '{' {
					performanceHashString(h, nodes[child].key)
					h.Write([]byte(":"))
				}
				emit(child)
			}
			end := byte('}')
			if n.kind == '[' {
				end = ']'
			}
			h.Write([]byte{end})
		case '"':
			performanceHashString(h, n.text)
		default:
			if (n.kind == '-' || n.kind >= '0' && n.kind <= '9') && len(n.raw) <= 64 {
				canonical := performanceCanonicalNumber(json.Number(string(n.raw)))
				if number, ok := canonical.(json.Number); ok {
					io.WriteString(h, string(number))
				}
			} else {
				h.Write(n.raw)
			}
		}
	}
	h.Write([]byte("{"))
	emitted := false
	// hasTools sorts among the accepted options using its canonical spelling.
	for i := 0; i < nodes[0].count; i++ {
		idx := order[nodes[0].begin+i]
		key := string(nodes[idx].key)
		if key == "model" || key == "messages" || key == "prompt" || key == "input" {
			continue
		}
		allowed := slices.Contains(chatRequestKeys, key) || key == "n"
		if family == "ollama" {
			allowed = allowed || key == "options" || key == "keep_alive" || key == "format" || key == "raw" || key == "think"
		}
		if !allowed {
			return false
		}
		if emitted {
			h.Write([]byte(","))
		}
		emitted = true
		if key == "tools" {
			performanceHashString(h, []byte("hasTools"))
			h.Write([]byte(":"))
			if bytes.Equal(nodes[idx].raw, []byte("null")) {
				h.Write([]byte("false"))
			} else {
				h.Write([]byte("true"))
			}
		} else {
			performanceHashString(h, nodes[idx].key)
			h.Write([]byte(":"))
			emit(idx)
		}
	}
	h.Write([]byte("}"))
	return true
}

func performanceFoldCompare(a, b []byte) int {
	for len(a) > 0 && len(b) > 0 {
		ar, an := utf8.DecodeRune(a)
		br, bn := utf8.DecodeRune(b)
		ar = unicode.ToLower(unicode.ToUpper(ar))
		br = unicode.ToLower(unicode.ToUpper(br))
		if ar < br {
			return -1
		}
		if ar > br {
			return 1
		}
		a = a[an:]
		b = b[bn:]
	}
	return len(a) - len(b)
}

func performanceHashString(h hash.Hash, value []byte) {
	h.Write([]byte{'"'})
	const hex = "0123456789abcdef"
	var scratch [6]byte
	for len(value) > 0 {
		r, n := utf8.DecodeRune(value)
		value = value[n:]
		if r == utf8.RuneError && n == 1 {
			h.Write([]byte(`\ufffd`))
			continue
		}
		switch r {
		case '"', '\\':
			scratch[0], scratch[1] = '\\', byte(r)
			h.Write(scratch[:2])
		case '\n', '\r', '\t', '\b', '\f':
			scratch[0] = '\\'
			switch r {
			case '\n':
				scratch[1] = 'n'
			case '\r':
				scratch[1] = 'r'
			case '\t':
				scratch[1] = 't'
			case '\b':
				scratch[1] = 'b'
			case '\f':
				scratch[1] = 'f'
			}
			h.Write(scratch[:2])
		default:
			if r < 0x20 || r == '<' || r == '>' || r == '&' || r == 0x2028 || r == 0x2029 {
				scratch = [6]byte{'\\', 'u', hex[(r>>12)&15], hex[(r>>8)&15], hex[(r>>4)&15], hex[r&15]}
				h.Write(scratch[:])
			} else {
				n := utf8.EncodeRune(scratch[:], r)
				h.Write(scratch[:n])
			}
		}
	}
	h.Write([]byte{'"'})
}

func performanceCanonicalNumber(v any) any {
	switch n := v.(type) {
	case json.Number:
		if value, ok := parsePerformanceDecimal([]byte(n)); ok {
			return json.Number(value.text())
		}
	case map[string]any:
		for k, value := range n {
			n[k] = performanceCanonicalNumber(value)
		}
	case []any:
		for i, value := range n {
			n[i] = performanceCanonicalNumber(value)
		}
	}
	return v
}

type performanceAttempt struct {
	mu                 sync.Mutex // first-byte peek can outlive a timed-out attempt
	f                  *facade
	key                performancewire.ProfileKey
	qualification      performanceExtractQualification
	identity           performanceIdentityTuple // immutable IDs, no request retention
	preparationRelease func()
	api, model         string
	parent             context.Context
	wait               time.Duration
	start              time.Time
	header, firstBody  *int64
	reader             *streamFrameReader
}

func (f *facade) newPerformanceAttempt(parent context.Context, model, api string, body []byte, wait time.Duration) *performanceAttempt {
	a, _ := f.newPerformanceAttemptOwned(parent, model, api, body, wait)
	return a
}

func (f *facade) newPerformanceAttemptOwned(parent context.Context, model, api string, body []byte, wait time.Duration) (*performanceAttempt, error) {
	var release func()
	if inferenceBuffers(parent) != nil {
		if allowance, ok := parent.Value(performancePreparationKey{}).(performancePreparation); ok {
			n, err := performancePreparationBytes(body)
			if err != nil || n > allowance.bytes {
				return nil, &bufferLimitError{}
			}
		} else {
			request, err := parseOriginStreamRequest(api, body)
			if err != nil {
				return nil, err
			}
			parent, release, err = preparePerformanceBuffers(parent, body, request, f.performancePreparationMode(api))
			if err != nil {
				return nil, err
			}
		}
	}
	key, qualification, identity, ok := f.performanceFingerprintOwned(model, api, body)
	if !ok {
		if release != nil {
			release()
		}
		f.host.dropPerformance(1)
		return nil, nil
	}
	return &performanceAttempt{f: f, key: key, qualification: qualification, identity: identity, preparationRelease: release, api: api, model: model, parent: parent, wait: wait}, nil
}

type performanceTransport struct {
	http.RoundTripper
	attempt *performanceAttempt
}

func (t performanceTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	a := t.attempt
	if a == nil {
		return t.RoundTripper.RoundTrip(r)
	}
	a.start = time.Now()
	resp, err := t.RoundTripper.RoundTrip(r)
	if resp != nil {
		a.header = performanceMilliseconds(time.Since(a.start))
		resp.Body = &performanceFirstBody{ReadCloser: resp.Body, attempt: a}
	}
	return resp, err
}

type performanceFirstBody struct {
	io.ReadCloser
	attempt *performanceAttempt
}

func (b *performanceFirstBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.attempt.mu.Lock()
	defer b.attempt.mu.Unlock()
	if n > 0 && b.attempt.firstBody == nil {
		b.attempt.firstBody = performanceMilliseconds(time.Since(b.attempt.start))
	}
	return n, err
}

func performanceMilliseconds(d time.Duration) *int64 {
	v := d.Milliseconds()
	if v < 0 || v > performancewire.MaxLatencyMs {
		return nil
	}
	return &v
}

// installResponseStream is shared by origin candidates and local ingress. Each
// response gets exactly SC's one reader. A peer-forwarding origin passes no
// attempt and therefore has no numeric producer.
func installResponseStream(resp *http.Response, request originStreamRequest, parent context.Context, idle time.Duration, a *performanceAttempt) (*streamFrameReader, error) {
	managed, validMedia := request.manages(resp.Header.Get("Content-Type"))
	if !managed || resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		if sink := responseSinkFrom(parent); sink != nil && resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
			resp.Body = &sinkObservedBody{ReadCloser: resp.Body, sink: sink, response: resp, peer: request}
		}
		return nil, nil
	}
	sink := responseSinkFrom(parent)
	config := streamFrameConfig{Protocol: request.protocol, Upstream: resp.Body, Parent: parent, IdleTimeout: idle, Choices: request.choices, PeerNormalized: request.peerNormalized, Sink: sink}
	if request.peerNormalized {
		config.PeerTrailer = func() []string { return resp.Trailer.Values(peerStreamFramingHeader) }
	}
	if a != nil && a.f.performancePreparationMode(a.api) == performancePrepareLocalNumeric {
		config.Extract = func(payload []byte) performanceExtractMetadata {
			return extractPerformanceObserved(payload, a.qualification)
		}
	}
	reader, err := newStreamFrameReader(config)
	if err != nil {
		return nil, err
	}
	if !validMedia || (resp.Header.Get("Content-Encoding") != "" && !strings.EqualFold(resp.Header.Get("Content-Encoding"), "identity")) {
		reader.mu.Lock()
		reader.finishInterruptedLocked(streamInterruptReasonMalformed)
		reader.mu.Unlock()
	}
	resp.Body = reader
	resp.ContentLength = -1
	clearOriginStreamIntegrity(resp.Header)
	clearOriginStreamIntegrity(resp.Trailer)
	if a != nil {
		a.reader = reader
	}
	return reader, nil
}

func (a *performanceAttempt) finish(sc *statusCapture, returned bool) {
	if a == nil {
		return
	}
	if a.preparationRelease != nil {
		defer a.preparationRelease()
	}
	if a.start.IsZero() {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	_, _, identity, valid := a.f.performanceCurrentIdentity(a.model, a.api)
	if !valid || identity != a.identity {
		a.f.host.dropPerformance(2)
		return
	}
	m := performancewire.Measurement{Outcome: performancewire.OutcomeFailed, AdmissionWaitMs: performanceMilliseconds(a.wait), HeaderMs: a.header, FirstBodyMs: a.firstBody, TotalMs: performanceMilliseconds(time.Since(a.start))}
	parentCause := context.Cause(a.parent)
	if parentCause != nil && !errors.Is(parentCause, context.DeadlineExceeded) {
		m.Outcome = performancewire.OutcomeCancelled
	}
	if a.reader != nil {
		result := a.reader.Result()
		complete := result.Outcome == streamOutcomeCompleted && returned && sc.wroteErr == nil && a.parent.Err() == nil
		if sink := responseSinkFrom(a.parent); sink != nil {
			delivery := sink.result()
			complete = complete && delivery.DeliveryComplete && delivery.ProtocolComplete
		}
		m.ProtocolComplete = &complete
		n := result.Numeric
		m.PromptTokens, m.CachedPromptTokens, m.OutputTokens = n.PromptTokens, n.CachedPromptTokens, n.OutputTokens
		m.PrefillNs, m.DecodeNs, m.LoadNs = n.PrefillNs, n.DecodeNs, n.LoadNs
		m.EngineDecodeNs, m.EngineDecodeTokensPerSecond = n.EngineDecodeNs, n.EngineDecodeTokensPerSecond
		if result.Outcome == streamOutcomeCancelled && !errors.Is(context.Cause(a.parent), context.DeadlineExceeded) {
			m.Outcome = performancewire.OutcomeCancelled
		}
		if complete {
			m.Outcome = performancewire.OutcomeCompleted
		}
	} else if returned && sc.upstreamBody != nil && sc.upstreamBody.complete && sc.wroteErr == nil && a.parent.Err() == nil && sc.status >= 200 && sc.status < 300 {
		complete := true
		if sink := responseSinkFrom(a.parent); sink != nil {
			delivery := sink.result()
			complete = delivery.DeliveryComplete && delivery.ProtocolComplete
		}
		if complete {
			m.Outcome = performancewire.OutcomeCompleted // unobserved protocol stays nil
		}
	}
	a.f.host.enqueuePerformance(performancewire.Sample{Key: a.key, Profile: performancewire.Profile{EngineFamily: a.qualification.EngineFamily}, Measurement: m})
}

// Metadata merges independent fields, never counters. Conflicts are sticky.
type performanceNumbers struct {
	value     performanceExtractMetadata
	conflicts uint16
}

func mergePerformanceNumber[T comparable](dst **T, src *T, conflicts *uint16, bit uint16) {
	if *conflicts&bit != 0 {
		*dst = nil
		return
	}
	if src == nil {
		return
	}
	if *dst != nil && **dst != *src {
		*conflicts |= bit
		*dst = nil
		return
	}
	v := *src
	*dst = &v
}
func (n *performanceNumbers) merge(v performanceExtractMetadata) {
	n.conflicts |= v.invalid
	fields := []struct {
		dst **int64
		src *int64
	}{
		{&n.value.PromptTokens, v.PromptTokens}, {&n.value.CachedPromptTokens, v.CachedPromptTokens},
		{&n.value.OutputTokens, v.OutputTokens}, {&n.value.PrefillNs, v.PrefillNs}, {&n.value.DecodeNs, v.DecodeNs},
		{&n.value.LoadNs, v.LoadNs}, {&n.value.EngineDecodeNs, v.EngineDecodeNs},
	}
	for i, field := range fields {
		mergePerformanceNumber(field.dst, field.src, &n.conflicts, 1<<i)
	}
	mergePerformanceNumber(&n.value.EngineDecodeTokensPerSecond, v.EngineDecodeTokensPerSecond, &n.conflicts, 1<<7)
	n.value.PhaseDisqualified = n.value.PhaseDisqualified || v.PhaseDisqualified
}
func (n *performanceNumbers) snapshot() performanceExtractMetadata {
	var copy performanceNumbers
	copy.merge(n.value)
	if n.value.PhaseDisqualified {
		copy.value.PromptTokens = nil
		copy.value.PrefillNs = nil
		copy.value.DecodeNs = nil
		copy.value.EngineDecodeNs = nil
		copy.value.EngineDecodeTokensPerSecond = nil
	}
	if n.conflicts&(1<<0) != 0 {
		copy.value.PrefillNs = nil
	}
	if n.conflicts&(1<<2) != 0 {
		copy.value.DecodeNs = nil
		copy.value.EngineDecodeNs = nil
		copy.value.EngineDecodeTokensPerSecond = nil
	}
	if n.conflicts&(1<<6) != 0 {
		copy.value.EngineDecodeTokensPerSecond = nil
	}
	return copy.value
}

// Mark present-but-unusable evidence as unknown even if an earlier object had
// usable counters. Absence still permits independently split usage/timings.
func extractPerformanceObserved(payload []byte, q performanceExtractQualification) performanceExtractMetadata {
	v := extractPerformanceMetadata(payload, q)
	f := performanceExtractObject(payload)
	var fields performanceExtractFields
	if q.EngineFamily == "ollama" && (q.APIPath == "/api/chat" || q.APIPath == "/api/generate") {
		fields = f
		for _, check := range []struct {
			key   string
			value *int64
			bit   uint16
		}{
			{"eval_count", v.OutputTokens, 1 << 2}, {"eval_duration", v.DecodeNs, 1 << 4}, {"load_duration", v.LoadNs, 1 << 5}, {"prompt_eval_duration", v.PrefillNs, 1 << 3},
		} {
			if _, present := fields.field(check.key); present && check.value == nil {
				v.invalid |= check.bit
			}
		}
		return v
	}
	if q.EngineFamily == "ollama" {
		usage := performanceExtractObject(f.value("usage"))
		if _, present := f.field("usage"); present && usage == nil {
			v.invalid |= 1<<2 | 1<<1
		}
		if _, present := usage.field("completion_tokens"); present && v.OutputTokens == nil {
			v.invalid |= 1 << 2
		}
	}
	return v
}
