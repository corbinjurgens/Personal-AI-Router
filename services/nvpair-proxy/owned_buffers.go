// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"unicode/utf16"
	"unicode/utf8"
)

type ownedInferenceContextKey struct{}

// This transparent adapter only exposes storage context to local error
// serialization. statusCapture remains the existing response/commit authority.
type inferenceErrorWriter struct {
	http.ResponseWriter
	ctx context.Context
}

func (w *inferenceErrorWriter) Unwrap() http.ResponseWriter       { return w.ResponseWriter }
func (w *inferenceErrorWriter) inferenceContext() context.Context { return w.ctx }
func errorBufferContext(w http.ResponseWriter) context.Context {
	for {
		if owner, ok := w.(interface{ inferenceContext() context.Context }); ok {
			return owner.inferenceContext()
		}
		unwrapper, ok := w.(interface{ Unwrap() http.ResponseWriter })
		if !ok {
			return nil
		}
		w = unwrapper.Unwrap()
	}
}

// Immutable protocol constants do not retain per-request or pooled storage.
var inferenceBufferRefusalJSON = []byte(`{"error":"inference buffer capacity is unavailable","code":"inference-buffer"}`)

type inferenceErrorField struct {
	key   string
	parts []string
}

// Fixed-field local errors share the same nonpool storage owner. String parts
// avoid constructing another concatenated message before capacity admission.
func writeInferenceErrorFields(w http.ResponseWriter, status int, fields ...inferenceErrorField) {
	size := 2
	for i, field := range fields {
		if i > 0 {
			size++
		}
		size += quotedJSONStringSize(field.key) + 3
		for _, part := range field.parts {
			size += quotedJSONStringSize(part) - 2
		}
	}
	if ctx := errorBufferContext(w); ctx != nil {
		if owner := inferenceBuffers(ctx); owner != nil {
			lease, err := owner.scope.tryAcquire(ctx, int64(size))
			if err != nil {
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("X-Content-Type-Options", "nosniff")
				w.WriteHeader(bufferRequestStatus(err))
				_, _ = w.Write(inferenceBufferRefusalJSON)
				return
			}
			defer lease.close()
		}
	}
	body := make([]byte, 0, size)
	body = append(body, '{')
	for i, field := range fields {
		if i > 0 {
			body = append(body, ',')
		}
		body = appendJSONString(body, field.key)
		body = append(body, ':', '"')
		for _, part := range field.parts {
			body = appendJSONStringContent(body, part)
		}
		body = append(body, '"')
	}
	body = append(body, '}')
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

// ownedInferenceBuffers is a storage account, not an inference lifecycle owner.
// The handler retains it until all synchronous transport borrowers have joined.
type ownedInferenceBuffers struct {
	scope    *bufferScope
	mu       sync.Mutex
	leases   []*bufferLease
	releases []func()
	parser   *bufferLease
}

func (p *Proxy) ownInferenceBuffers(r *http.Request) (*http.Request, *ownedInferenceBuffers) {
	// Literal Proxy fixtures also use the same configured-root path.
	p.transportMu.Lock()
	if p.buffers == nil {
		p.buffers, _ = newBufferBudget(192<<20, 0)
	}
	b := p.buffers
	p.transportMu.Unlock()
	scope, _ := b.newRequest(math.MaxInt64)
	owner := &ownedInferenceBuffers{scope: scope}
	return r.WithContext(context.WithValue(r.Context(), ownedInferenceContextKey{}, owner)), owner
}

func inferenceBuffers(ctx context.Context) *ownedInferenceBuffers {
	owner, _ := ctx.Value(ownedInferenceContextKey{}).(*ownedInferenceBuffers)
	return owner
}

func (o *ownedInferenceBuffers) retain(lease *bufferLease) {
	o.mu.Lock()
	o.leases = append(o.leases, lease)
	o.mu.Unlock()
}

func (o *ownedInferenceBuffers) retainParser(lease *bufferLease) {
	o.mu.Lock()
	o.leases = append(o.leases, lease)
	o.parser = lease
	o.mu.Unlock()
}

// A bootstrap borrower shares the actual retained parser charge. An account
// alone grants no storage, and the borrow keeps the allowance live if the
// request owner closes before the reader's final borrower joins.
func (o *ownedInferenceBuffers) borrowParserAllowance() *bufferLease {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.parser == nil {
		return nil
	}
	lease, err := o.parser.borrow()
	if err != nil {
		return nil
	}
	return lease
}

func (o *ownedInferenceBuffers) close() {
	o.mu.Lock()
	leases := o.leases
	o.leases = nil
	o.parser = nil
	releases := o.releases
	o.releases = nil
	o.mu.Unlock()
	for _, lease := range leases {
		lease.close()
	}
	for _, release := range releases {
		release()
	}
}

func (o *ownedInferenceBuffers) retainRelease(release func()) {
	o.mu.Lock()
	o.releases = append(o.releases, release)
	o.mu.Unlock()
}

func bufferRequestStatus(err error) int {
	var status interface{ HTTPStatus() int }
	if errors.As(err, &status) {
		return status.HTTPStatus()
	}
	var overflow *http.MaxBytesError
	if errors.As(err, &overflow) {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusBadRequest
}

// readBody charges complete replacement capacity while the previous array is
// still borrowed. The logical length limit does not constrain physical overlap.
func (o *ownedInferenceBuffers) readBody(r *http.Request, limit int64) ([]byte, string, error) {
	if r.Body == nil {
		scratch, err := o.scope.acquire(r.Context(), 96<<10)
		if err != nil {
			return nil, "", err
		}
		o.retainParser(scratch)
		return nil, "", nil
	}
	defer r.Body.Close()
	if limit < 0 || r.ContentLength > limit {
		return nil, "", &bufferLimitError{}
	}
	var body []byte
	var storage *bufferLease
	defer func() {
		if storage != nil {
			storage.close()
		}
	}()
	for {
		var pending []byte
		var pendingLease *bufferLease
		var pendingEOF bool
		// ContentLength is a hint, not an EOF guarantee. Probe a filled known
		// representation before replacing it solely to discover EOF. When a
		// peer sends more, the byte stays charged through replacement and copy.
		if int64(len(body)) == limit || (r.ContentLength > 0 && int64(len(body)) == r.ContentLength && len(body) == cap(body)) {
			probe, err := o.scope.acquire(r.Context(), 1)
			if err != nil {
				return nil, "", err
			}
			one := make([]byte, 1)
			n, readErr := r.Body.Read(one)
			if n > 0 && int64(len(body)) == limit {
				probe.close()
				return nil, "", &bufferLimitError{}
			}
			if n == 0 && readErr == io.EOF {
				probe.close()
				break
			}
			if readErr != nil && readErr != io.EOF {
				probe.close()
				return nil, "", readErr
			}
			if n == 0 {
				probe.close()
				continue
			}
			pending, pendingLease, pendingEOF = one[:n], probe, readErr == io.EOF
		}
		if len(body) == cap(body) {
			target := int64(cap(body)) * 2
			if target < 32<<10 {
				target = 32 << 10
			}
			if len(body) == 0 && r.ContentLength > 0 {
				target = r.ContentLength
			}
			if target > limit {
				target = limit
			}
			lease, err := o.scope.acquire(r.Context(), target)
			if err != nil {
				if pendingLease != nil {
					pendingLease.close()
				}
				return nil, "", err
			}
			next := make([]byte, len(body), int(target))
			copy(next, body)
			body = next
			if storage != nil {
				storage.close()
			}
			storage = lease
		}
		if pendingLease != nil {
			body = append(body, pending...)
			pendingLease.close()
			if pendingEOF {
				break
			}
		}
		n, err := r.Body.Read(body[len(body):cap(body)])
		body = body[:len(body)+n]
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, "", err
		}
	}
	if storage != nil {
		o.retain(storage)
		storage = nil
	}
	// Fixed scanner state and key/string decoding scratch are covered
	// independently of body size, without retaining an implicit parser pool.
	scratch, err := o.scope.acquire(r.Context(), 96<<10)
	if err != nil {
		return nil, "", err
	}
	// Keep the fixed parser allowance through the last handler borrower.
	// Ambiguity, stream and model checks reuse this bounded state sequentially.
	o.retainParser(scratch)
	if !ownedJSONValid(body) {
		return body, "", nil
	}
	if err := validateBodyModelIdentifier(body); err != nil {
		return nil, "", err
	}
	object := bytes.TrimSpace(body)
	if len(object) == 0 || object[0] != '{' {
		return body, "", nil
	}
	raw := streamJSONField(object, "model")
	if len(raw) == 0 || raw[0] != '"' {
		return body, "", nil
	}
	plain := bytes.IndexByte(raw, '\\') < 0 && utf8.Valid(raw)
	decodeBytes := 0
	if plain {
		decodeBytes = len(raw) - 2
	} else {
		for i := 1; i < len(raw)-1; {
			r, next := ownedJSONRune(raw, i)
			decodeBytes += utf8.RuneLen(r)
			i = next
		}
	}
	// Builder.Grow rounds to an allocator size class/page. One runtime page
	// covers rounding; the actual retained capacity is checked below. String
	// returns the same backing array, so no decoded-byte/string overlap exists.
	reserved := int64(decodeBytes)
	if !plain {
		reserved += 8192
	}
	decoded, err := o.scope.acquire(r.Context(), reserved)
	if err != nil {
		return nil, "", err
	}
	var model string
	if plain {
		model = string(raw[1 : len(raw)-1])
	} else {
		var builder strings.Builder
		builder.Grow(decodeBytes)
		if int64(builder.Cap()) > reserved {
			decoded.close()
			return nil, "", errBufferInvalid
		}
		for i := 1; i < len(raw)-1; {
			r, next := ownedJSONRune(raw, i)
			builder.WriteRune(r)
			i = next
		}
		model = builder.String()
	}
	o.retain(decoded)
	return body, model, nil
}

// raw has already passed syntax validation. Invalid UTF-8 and unpaired
// surrogate escapes become RuneError, matching encoding/json string decoding.
func ownedJSONRune(raw []byte, i int) (rune, int) {
	if raw[i] != '\\' {
		r, n := utf8.DecodeRune(raw[i:])
		return r, i + n
	}
	i++
	switch raw[i] {
	case '"', '\\', '/':
		return rune(raw[i]), i + 1
	case 'b':
		return '\b', i + 1
	case 'f':
		return '\f', i + 1
	case 'n':
		return '\n', i + 1
	case 'r':
		return '\r', i + 1
	case 't':
		return '\t', i + 1
	}
	hex := func(start int) rune {
		var r rune
		for _, c := range raw[start : start+4] {
			r *= 16
			switch {
			case c <= '9':
				r += rune(c - '0')
			case c <= 'F':
				r += rune(c - 'A' + 10)
			default:
				r += rune(c - 'a' + 10)
			}
		}
		return r
	}
	r := hex(i + 1)
	i += 5
	if !utf16.IsSurrogate(r) {
		return r, i
	}
	if i+6 <= len(raw)-1 && raw[i] == '\\' && raw[i+1] == 'u' {
		if pair := utf16.DecodeRune(r, hex(i+2)); pair != utf8.RuneError {
			return pair, i + 6
		}
	}
	return utf8.RuneError, i
}

// ownedJSONValid validates syntax with fixed, caller-charged state. It never
// borrows encoding/json's scanner pool, which can retain an uncharged stack.
func ownedJSONValid(raw []byte) bool {
	const (
		objectFirst byte = iota
		objectKey
		objectColon
		objectNext
		arrayFirst
		arrayValue
		arrayNext
	)
	var states [10000]byte
	i, depth := 0, 0
	value := true
	space := func() {
		for i < len(raw) && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\r' || raw[i] == '\n') {
			i++
		}
	}
	for {
		space()
		if value {
			if i == len(raw) {
				return false
			}
			switch raw[i] {
			case '{', '[':
				if depth == len(states) {
					return false
				}
				states[depth] = objectFirst
				if raw[i] == '[' {
					states[depth] = arrayFirst
				}
				depth++
				i++
				value = false
				continue
			case '"':
				end, ok := ownedJSONStringEnd(raw, i)
				if !ok {
					return false
				}
				i = end
			case 't':
				if len(raw)-i < 4 || !bytes.Equal(raw[i:i+4], []byte("true")) {
					return false
				}
				i += 4
			case 'f':
				if len(raw)-i < 5 || !bytes.Equal(raw[i:i+5], []byte("false")) {
					return false
				}
				i += 5
			case 'n':
				if len(raw)-i < 4 || !bytes.Equal(raw[i:i+4], []byte("null")) {
					return false
				}
				i += 4
			default:
				if raw[i] == '-' {
					i++
					if i == len(raw) {
						return false
					}
				}
				if raw[i] == '0' {
					i++
				} else {
					if raw[i] < '1' || raw[i] > '9' {
						return false
					}
					for i < len(raw) && raw[i] >= '0' && raw[i] <= '9' {
						i++
					}
				}
				if i < len(raw) && raw[i] == '.' {
					i++
					start := i
					for i < len(raw) && raw[i] >= '0' && raw[i] <= '9' {
						i++
					}
					if i == start {
						return false
					}
				}
				if i < len(raw) && (raw[i] == 'e' || raw[i] == 'E') {
					i++
					if i < len(raw) && (raw[i] == '+' || raw[i] == '-') {
						i++
					}
					start := i
					for i < len(raw) && raw[i] >= '0' && raw[i] <= '9' {
						i++
					}
					if i == start {
						return false
					}
				}
			}
			value = false
		}
		if depth == 0 {
			space()
			return i == len(raw)
		}
		space()
		if i == len(raw) {
			return false
		}
		switch states[depth-1] {
		case objectFirst, objectKey:
			if states[depth-1] == objectFirst && raw[i] == '}' {
				depth--
				i++
				continue
			}
			end, ok := ownedJSONStringEnd(raw, i)
			if !ok {
				return false
			}
			i = end
			states[depth-1] = objectColon
		case objectColon:
			if raw[i] != ':' {
				return false
			}
			i++
			states[depth-1] = objectNext
			value = true
		case objectNext:
			if raw[i] == '}' {
				depth--
				i++
				continue
			}
			if raw[i] != ',' {
				return false
			}
			i++
			states[depth-1] = objectKey
		case arrayFirst, arrayValue:
			if states[depth-1] == arrayFirst && raw[i] == ']' {
				depth--
				i++
				continue
			}
			states[depth-1] = arrayNext
			value = true
		case arrayNext:
			if raw[i] == ']' {
				depth--
				i++
				continue
			}
			if raw[i] != ',' {
				return false
			}
			i++
			states[depth-1] = arrayValue
		}
	}
}

func ownedJSONStringEnd(raw []byte, start int) (int, bool) {
	if start >= len(raw) || raw[start] != '"' {
		return 0, false
	}
	for i := start + 1; i < len(raw); i++ {
		if raw[i] == '"' {
			return i + 1, true
		}
		if raw[i] < 0x20 {
			return 0, false
		}
		if raw[i] != '\\' {
			continue
		}
		i++
		if i == len(raw) {
			return 0, false
		}
		switch raw[i] {
		case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		case 'u':
			if len(raw)-i < 5 {
				return 0, false
			}
			for j := i + 1; j < i+5; j++ {
				c := raw[j]
				if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
					return 0, false
				}
			}
			i += 4
		default:
			return 0, false
		}
	}
	return 0, false
}
