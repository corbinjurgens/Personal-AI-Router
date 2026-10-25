// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Body edits preserve untouched bytes. The iterative plan measures member/key
// storage before allocation and writes directly into one final send body.
import (
	"bytes"
	"context"
	"net/http"
	"strings"
	"unicode"
	"unicode/utf8"

	"nvpair-shared/nodepolicy"
)

type jsonMember struct {
	key                  string
	valueStart, valueEnd int
}

// Identifiers become bounded control metadata only after this decoded-byte
// check. Padding counts before normalization, and escaped strings are measured
// without allocating their decoded representation.
const maxRequestIdentifierBytes = 256

type requestIdentifierError struct{}

func (*requestIdentifierError) Error() string {
	return "invalid model identifier"
}
func (*requestIdentifierError) HTTPStatus() int { return http.StatusBadRequest }

func requestJSONStringTooLong(raw []byte) bool {
	n := 0
	for i := 1; i < len(raw)-1; {
		r, next := ownedJSONRune(raw, i)
		n += utf8.RuneLen(r)
		if n > maxRequestIdentifierBytes {
			return true
		}
		i = next
	}
	return false
}

func validateBodyModelIdentifier(body []byte) error {
	if !ownedJSONValid(body) {
		return nil
	}
	invalid := false
	visitObject(body, func(key []byte, start, end int) {
		if (jsonKeyEqual(key, "model", true) || jsonKeyEqual(key, "name", true)) && end > start && body[start] == '"' {
			raw := body[start:end]
			if requestJSONStringTooLong(raw) || nodepolicy.ValidateModelIdentifierJSON(raw) != nil {
				invalid = true
			}
		}
	})
	if invalid {
		return &requestIdentifierError{}
	}
	return nil
}

// visitObject takes validated JSON and visits borrowed top-level values.
func visitObject(data []byte, visit func([]byte, int, int)) (int, bool) {
	i := 0
	space := func() {
		for i < len(data) && (data[i] == ' ' || data[i] == '\t' || data[i] == '\r' || data[i] == '\n') {
			i++
		}
	}
	space()
	if i == len(data) || data[i] != '{' {
		return 0, false
	}
	i++
	for {
		space()
		if i < len(data) && data[i] == '}' {
			return i, true
		}
		keyStart := i
		keyEnd, ok := ownedJSONStringEnd(data, i)
		if !ok {
			return 0, false
		}
		i = keyEnd
		space()
		if i == len(data) || data[i] != ':' {
			return 0, false
		}
		i++
		space()
		start := i
		end := i + streamJSONValueEnd(data[i:])
		for end > start && (data[end-1] == ' ' || data[end-1] == '\t' || data[end-1] == '\r' || data[end-1] == '\n') {
			end--
		}
		if visit != nil {
			visit(data[keyStart:keyEnd], start, end)
		}
		i = end
		space()
		if i < len(data) && data[i] == ',' {
			i++
			continue
		}
		if i < len(data) && data[i] == '}' {
			return i, true
		}
		return 0, false
	}
}

func decodedJSONStringBytes(raw []byte) int {
	n := 0
	for i := 1; i < len(raw)-1; {
		r, next := ownedJSONRune(raw, i)
		n += utf8.RuneLen(r)
		i = next
	}
	return n
}
func decodedJSONStringCapacity(raw []byte) int64 {
	if bytes.IndexByte(raw, '\\') < 0 && utf8.Valid(raw) {
		return int64(len(raw) - 2)
	}
	n := int64(decodedJSONStringBytes(raw))
	if n <= 1024 {
		return 2*n + 16
	}
	if n <= 32768 {
		return n + 4096
	}
	return n + 8192
}
func decodeJSONString(raw []byte) string {
	if bytes.IndexByte(raw, '\\') < 0 && utf8.Valid(raw) {
		return string(raw[1 : len(raw)-1])
	}
	var b strings.Builder
	b.Grow(decodedJSONStringBytes(raw))
	for i := 1; i < len(raw)-1; {
		r, next := ownedJSONRune(raw, i)
		b.WriteRune(r)
		i = next
	}
	return b.String()
}
func jsonKeyEqual(raw []byte, name string, fold bool) bool {
	if len(raw) > 6*len(name)+2 {
		return false
	}
	pos := 0
	for i := 1; i < len(raw)-1; {
		if pos == len(name) {
			return false
		}
		r, next := ownedJSONRune(raw, i)
		other, n := utf8.DecodeRuneInString(name[pos:])
		if r != other {
			if !fold {
				return false
			}
			folded := unicode.SimpleFold(r)
			for folded != r && folded != other {
				folded = unicode.SimpleFold(folded)
			}
			if folded != other {
				return false
			}
		}
		i, pos = next, pos+n
	}
	return pos == len(name)
}
func jsonStringEquals(raw []byte, want string) bool {
	if len(raw) < 2 || raw[0] != '"' {
		return false
	}
	pos := 0
	for i := 1; i < len(raw)-1; {
		if pos == len(want) {
			return false
		}
		r, next := ownedJSONRune(raw, i)
		other, n := utf8.DecodeRuneInString(want[pos:])
		if r != other {
			return false
		}
		pos += n
		i = next
	}
	return pos == len(want)
}

type bodyWorkspace struct {
	ctx    context.Context
	owner  *ownedInferenceBuffers
	leases []*bufferLease
}

func newBodyWorkspace(ctx context.Context) *bodyWorkspace {
	w := &bodyWorkspace{ctx: ctx}
	if ctx != nil {
		w.owner = inferenceBuffers(ctx)
	}
	return w
}
func (w *bodyWorkspace) acquire(n int64) error {
	if w.owner == nil {
		return nil
	}
	// Includes lease records and old/new bookkeeping slice capacity.
	l, err := w.owner.scope.acquire(w.ctx, n+256)
	if err != nil {
		return err
	}
	w.leases = append(w.leases, l)
	return nil
}
func (w *bodyWorkspace) close() {
	for _, l := range w.leases {
		l.close()
	}
	w.leases = nil
}
func (w *bodyWorkspace) members(data []byte) ([]jsonMember, int, bool, error) {
	count := 0
	keyBytes := int64(0)
	close, ok := visitObject(data, func(key []byte, _, _ int) { count++; keyBytes += decodedJSONStringCapacity(key) })
	if !ok {
		return nil, 0, false, nil
	}
	if err := w.acquire(32*int64(count) + keyBytes); err != nil {
		return nil, 0, false, err
	}
	members := make([]jsonMember, 0, count)
	visitObject(data, func(key []byte, start, end int) {
		members = append(members, jsonMember{decodeJSONString(key), start, end})
	})
	return members, close, true, nil
}
func objectMembers(data []byte) ([]jsonMember, int, bool) {
	if !ownedJSONValid(data) {
		return nil, 0, false
	}
	w := newBodyWorkspace(nil)
	defer w.close()
	m, c, ok, _ := w.members(data)
	return m, c, ok
}

const ambiguousModelMessage = `the request body must use a single top-level "model" member with that exact spelling`

func ambiguousModelMember(body []byte) bool {
	if !ownedJSONValid(body) {
		return false
	}
	seen, bad := false, false
	visitObject(body, func(key []byte, _, _ int) {
		if jsonKeyEqual(key, "model", true) {
			if seen || !jsonKeyEqual(key, "model", false) {
				bad = true
			}
			seen = true
		}
	})
	return bad
}
func bodyModelMatches(body []byte, model string) bool {
	if !ownedJSONValid(body) || ambiguousModelMember(body) {
		return false
	}
	matched := false
	_, ok := visitObject(body, func(key []byte, start, end int) {
		if jsonKeyEqual(key, "model", false) {
			matched = jsonStringEquals(body[start:end], model)
		}
	})
	return ok && matched
}

// Match encoding/json string spelling without its retained encoder pool.
func appendJSONString(out []byte, s string) []byte {
	out = append(out, '"')
	out = appendJSONStringContent(out, s)
	return append(out, '"')
}
func appendJSONStringContent(out []byte, s string) []byte {
	const hex = "0123456789abcdef"
	for i := 0; i < len(s); {
		r, width := utf8.DecodeRuneInString(s[i:])
		i += width
		if r == utf8.RuneError && width == 1 {
			out = append(out, `\ufffd`...)
			continue
		}
		switch r {
		case '"', '\\':
			out = append(out, '\\', byte(r))
		case '\n':
			out = append(out, '\\', 'n')
		case '\r':
			out = append(out, '\\', 'r')
		case '\t':
			out = append(out, '\\', 't')
		case '\b':
			out = append(out, '\\', 'b')
		case '\f':
			out = append(out, '\\', 'f')
		default:
			if r < 0x20 || r == '<' || r == '>' || r == '&' || r == 0x2028 || r == 0x2029 {
				out = append(out, '\\', 'u', hex[(r>>12)&15], hex[(r>>8)&15], hex[(r>>4)&15], hex[r&15])
			} else {
				out = utf8.AppendRune(out, r)
			}
		}
	}
	return out
}
func quotedJSONStringSize(s string) int {
	n := 2
	for i := 0; i < len(s); {
		r, width := utf8.DecodeRuneInString(s[i:])
		i += width
		if r == utf8.RuneError && width == 1 {
			n += 6
			continue
		}
		switch r {
		case '"', '\\', '\n', '\r', '\t', '\b', '\f':
			n += 2
		default:
			if r < 0x20 || r == '<' || r == '>' || r == '&' || r == 0x2028 || r == 0x2029 {
				n += 6
			} else {
				n += utf8.RuneLen(r)
			}
		}
	}
	return n
}

type bodyMergePlan struct {
	body, defaults          []byte
	members, defaultMembers []jsonMember
	byKey                   map[string]int
	children                []*bodyMergePlan
	parent, next, work      *bodyMergePlan
	close, size, modelIndex int
	changed                 bool
	emitIndex, pos          int
}

// The caller closes release after its final transport borrower joins. No
// intermediate rewrite, recursive merge output, or appended-default copy exists.
func transformBody(ctx context.Context, body []byte, model string, rewrite bool, defaults []byte) ([]byte, func(), error) {
	if !isJSONObject(defaults) {
		defaults = nil
	}
	if trimmed := bytes.TrimSpace(defaults); len(trimmed) >= 2 && trimmed[len(trimmed)-1] == '}' && len(bytes.TrimSpace(trimmed[1:len(trimmed)-1])) == 0 {
		defaults = nil
	}
	if !rewrite && len(defaults) == 0 {
		return body, func() {}, nil
	}
	w := newBodyWorkspace(ctx)
	defer w.close()
	if err := w.acquire(96 << 10); err != nil {
		return nil, nil, err
	}
	if !ownedJSONValid(body) {
		if rewrite {
			return nil, nil, errAmbiguousModel
		}
		return body, func() {}, nil
	}
	if len(bytes.TrimSpace(defaults)) > 0 && !ownedJSONValid(defaults) {
		defaults = nil
	}
	if len(defaults) == 0 {
		if !rewrite {
			return body, func() {}, nil
		}
		start, end, found := 0, 0, false
		visitObject(body, func(key []byte, s, e int) {
			if !found && jsonKeyEqual(key, "model", false) {
				start, end, found = s, e, true
			}
		})
		if !found {
			return nil, nil, errAmbiguousModel
		}
		size := len(body) - (end - start) + quotedJSONStringSize(model)
		var lease *bufferLease
		var err error
		if w.owner != nil {
			lease, err = w.owner.scope.acquire(ctx, int64(size)+256)
			if err != nil {
				return nil, nil, err
			}
		}
		release := func() {}
		if lease != nil {
			release = lease.close
		}
		out := make([]byte, 0, size)
		out = append(out, body[:start]...)
		out = appendJSONString(out, model)
		out = append(out, body[end:]...)
		return out, release, nil
	}
	var all, pending *bodyMergePlan
	newNode := func(body, defs []byte, parent *bodyMergePlan) (*bodyMergePlan, error) {
		if err := w.acquire(256); err != nil {
			return nil, err
		}
		members, close, ok, err := w.members(body)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, nil
		}
		defMembers, _, _, err := w.members(defs)
		if err != nil {
			return nil, err
		}
		// 128 bytes/member covers map entries/control groups and old/new
		// map growth plus the exact-capacity child pointer array.
		if err := w.acquire(128*int64(len(members)) + 512); err != nil {
			return nil, err
		}
		n := &bodyMergePlan{body: body, defaults: defs, members: members, defaultMembers: defMembers, close: close, size: len(body), modelIndex: -1, parent: parent, next: all, work: pending, byKey: make(map[string]int, len(members)), children: make([]*bodyMergePlan, len(members))}
		for i, m := range members {
			n.byKey[m.key] = i
		}
		all, pending = n, n
		return n, nil
	}
	root, err := newNode(body, defaults, nil)
	if err != nil {
		return nil, nil, err
	}
	if root == nil {
		if rewrite {
			return nil, nil, errAmbiguousModel
		}
		return body, func() {}, nil
	}
	if rewrite {
		for i, m := range root.members {
			if m.key == "model" {
				root.modelIndex = i
				root.changed = true
				break
			}
		}
		if root.modelIndex < 0 {
			return nil, nil, errAmbiguousModel
		}
	}
	for pending != nil {
		node := pending
		pending = node.work
		for _, d := range node.defaultMembers {
			i, present := node.byKey[d.key]
			if !present {
				node.changed = true
				continue
			}
			if node == root && i == root.modelIndex {
				continue
			}
			m := node.members[i]
			req, def := node.body[m.valueStart:m.valueEnd], node.defaults[d.valueStart:d.valueEnd]
			if !isJSONObject(req) || !isJSONObject(def) {
				continue
			}
			child, e := newNode(req, def, node)
			if e != nil {
				return nil, nil, e
			}
			node.children[i] = child
		}
	}
	// Children are newer than parents: this linked traversal is postorder.
	for node := all; node != nil; node = node.next {
		for i, child := range node.children {
			if child != nil && child.changed {
				m := node.members[i]
				node.size += child.size - (m.valueEnd - m.valueStart)
				node.changed = true
			}
		}
		appended := len(node.members) > 0
		for _, d := range node.defaultMembers {
			if _, present := node.byKey[d.key]; !present {
				if appended {
					node.size++
				}
				appended = true
				node.size += quotedJSONStringSize(d.key) + 1 + d.valueEnd - d.valueStart
			}
		}
		if node == root && node.modelIndex >= 0 {
			m := node.members[node.modelIndex]
			node.size += quotedJSONStringSize(model) - (m.valueEnd - m.valueStart)
		}
	}
	if !root.changed {
		return body, func() {}, nil
	}
	var lease *bufferLease
	if w.owner != nil {
		lease, err = w.owner.scope.acquire(ctx, int64(root.size)+256)
		if err != nil {
			return nil, nil, err
		}
	}
	release := func() {}
	if lease != nil {
		release = lease.close
	}
	out := make([]byte, 0, root.size)
	for node := root; node != nil; {
		if node.emitIndex < len(node.members) {
			i := node.emitIndex
			m := node.members[i]
			node.emitIndex++
			out = append(out, node.body[node.pos:m.valueStart]...)
			node.pos = m.valueEnd
			if node == root && i == node.modelIndex {
				out = appendJSONString(out, model)
				continue
			}
			if child := node.children[i]; child != nil && child.changed {
				node = child
				continue
			}
			out = append(out, node.body[m.valueStart:m.valueEnd]...)
			continue
		}
		out = append(out, node.body[node.pos:node.close]...)
		appended := len(node.members) > 0
		for _, d := range node.defaultMembers {
			if _, present := node.byKey[d.key]; !present {
				if appended {
					out = append(out, ',')
				}
				appended = true
				out = appendJSONString(out, d.key)
				out = append(out, ':')
				out = append(out, node.defaults[d.valueStart:d.valueEnd]...)
			}
		}
		out = append(out, node.body[node.close:]...)
		node = node.parent
	}
	if len(out) != root.size || cap(out) != root.size {
		release()
		return nil, nil, errBufferInvalid
	}
	return out, release, nil
}

func replaceModel(body []byte, model string) ([]byte, bool) {
	out, release, err := transformBody(nil, body, model, true, nil)
	if release != nil {
		defer release()
	}
	return out, err == nil && out != nil
}
func mergeDefaults(body, defaults []byte) []byte {
	out, release, err := transformBody(nil, body, "", false, defaults)
	if release != nil {
		defer release()
	}
	if err != nil {
		return body
	}
	return out
}
func mergeObject(body, defaults []byte) ([]byte, bool) {
	if !isJSONObject(body) || !isJSONObject(defaults) || !ownedJSONValid(body) || !ownedJSONValid(defaults) {
		return nil, false
	}
	return mergeDefaults(body, defaults), true
}
func isJSONObject(raw []byte) bool { t := bytes.TrimSpace(raw); return len(t) > 0 && t[0] == '{' }

func visitArray(raw []byte, visit func([]byte)) bool {
	raw = bytes.TrimSpace(raw)
	if len(raw) < 2 || raw[0] != '[' {
		return false
	}
	for rest := bytes.TrimSpace(raw[1:]); len(rest) > 0 && rest[0] != ']'; {
		end := streamJSONValueEnd(rest)
		if end == 0 {
			return false
		}
		if visit != nil {
			visit(bytes.TrimSpace(rest[:end]))
		}
		rest = bytes.TrimSpace(rest[end:])
		if len(rest) > 0 && rest[0] == ',' {
			rest = bytes.TrimSpace(rest[1:])
		}
	}
	return true
}

func requestJSONField(object []byte, name string) []byte {
	var result []byte
	visitObject(object, func(key []byte, start, end int) {
		if jsonKeyEqual(key, name, true) {
			result = object[start:end]
		}
	})
	return result
}

func positiveJSONInt(raw []byte) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	n := 0
	limit := int(^uint(0) >> 1)
	for _, c := range raw {
		if c < '0' || c > '9' || n > (limit-int(c-'0'))/10 {
			return 0, false
		}
		n = n*10 + int(c-'0')
	}
	return n, n > 0
}

// Struct decoding preserves an earlier value on a later type error, and
// merges repeated object members. These views retain those existing semantics.
func requestJSONArrayField(object []byte, name string) []byte {
	var result []byte
	visitObject(object, func(key []byte, start, end int) {
		if !jsonKeyEqual(key, name, true) {
			return
		}
		raw := object[start:end]
		if bytes.Equal(raw, []byte("null")) {
			result = nil
		} else if len(raw) > 0 && raw[0] == '[' {
			result = raw
		}
	})
	return result
}
func requestStructString(object []byte, name string, previous []byte) []byte {
	visitObject(object, func(key []byte, start, end int) {
		if jsonKeyEqual(key, name, true) && object[start] == '"' {
			previous = object[start:end]
		}
	})
	return previous
}
func requestNestedString(object []byte, parent, name string) []byte {
	var result []byte
	visitObject(object, func(key []byte, start, end int) {
		if !jsonKeyEqual(key, parent, true) {
			return
		}
		raw := object[start:end]
		if bytes.Equal(raw, []byte("null")) {
			result = nil
		} else if isJSONObject(raw) {
			result = requestStructString(raw, name, result)
		}
	})
	return result
}
func requestStructPositiveInt(object []byte, name string, previous int) int {
	visitObject(object, func(key []byte, start, end int) {
		if !jsonKeyEqual(key, name, true) {
			return
		}
		raw := object[start:end]
		if bytes.Equal(raw, []byte("null")) {
			previous = 0
			return
		}
		if n, ok := jsonInteger(raw); ok {
			previous = n
		}
	})
	return previous
}
func requestNestedPositiveInt(object []byte, parent, name string) int {
	result := 0
	visitObject(object, func(key []byte, start, end int) {
		if !jsonKeyEqual(key, parent, true) {
			return
		}
		raw := object[start:end]
		if bytes.Equal(raw, []byte("null")) {
			result = 0
		} else if isJSONObject(raw) {
			result = requestStructPositiveInt(raw, name, result)
		}
	})
	return result
}
func jsonInteger(raw []byte) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	negative := raw[0] == '-'
	if negative {
		raw = raw[1:]
	}
	if len(raw) == 0 {
		return 0, false
	}
	limit := -int(^uint(0) >> 1)
	if negative {
		limit--
	}
	n := 0
	for _, c := range raw {
		if c < '0' || c > '9' || n < (limit+int(c-'0'))/10 {
			return 0, false
		}
		n = n*10 - int(c-'0')
	}
	if negative {
		return n, true
	}
	return -n, true
}
