// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package jsonrpc

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"sort"
	"strconv"
	"unicode"
	"unicode/utf8"
)

// These independent transport grants never change ordinary codec limits.
const (
	HistoryRecordBytes      = 131072                                     // R
	HistoryInitialBytes     = 33554432                                   // M
	HistoryEnvelopeBytes    = 4096                                       // E, including newline
	HistoryRecordFrameBytes = HistoryRecordBytes + HistoryEnvelopeBytes  // P
	HistoryFrameBytes       = HistoryInitialBytes + HistoryEnvelopeBytes // H
	HistoryIDBytes          = 512                                        // encoded token, including quotes and escapes
)

var ErrHistoryBusy = errors.New(HistoryBusyMessage)
var ErrHistoryFraming = errors.New("history-framing")
var ErrHistoryInvalid = errors.New("history-invalid")

const HistoryBusyCode = -32096
const HistoryBusyMessage = "routing history busy; retry"
const HistoryInvalidCode = -32062

// HistoryFailure contains no input-derived text. Owners discard their private
// stage on discard, fail only the identified pending call on fail-call, and
// fail their bounded history pending set and schedule one refetch on quarantine.
// Ordinary calls must never be failed by this disposition.
type HistoryFailure struct {
	Disposition string
	Category    string
	Count       int
	FailPending bool
	// ScheduleRefetch requests one outstanding owner timer, independently of
	// failing a newly retired call. Successful BeginMethod consumes that request.
	ScheduleRefetch bool
}

func (e *HistoryFailure) Error() string { return e.Category }

func historyFailure(disposition, category string) error {
	return &HistoryFailure{Disposition: disposition, Category: category, Count: 1}
}

// HistoryProjection describes known fields, not a generic input tree. Every
// tagged struct field is required; pointers alone permit null. Array counts,
// scalar meaning and schema versions remain the supplying package's concern.
// Check is called only on stripped bytes; it must return fixed diagnostics.
type HistoryProjection struct {
	Type                                    reflect.Type
	Bytes                                   int
	Depth                                   int
	Members                                 int
	ArrayLimit                              int
	RecordType                              reflect.Type
	RecordBytes, RecordDepth, RecordMembers int
	Integer                                 func([]byte) (int64, bool)
	CheckRecord                             func([]byte) error
	Check                                   func([]byte) error
}

// ProjectHistory checks every subtree before materialization, including unknown
// members. dst and offsets are caller-owned leases: no growing buffers, maps,
// decoded unknown strings or json.Decoder token allocations. Each key uses one
// uint32 raw offset, bounded by the raw value's byte grant.
func ProjectHistory(raw, dst []byte, offsets []uint32, schema HistoryProjection) ([]byte, int, error) {
	if len(raw) > schema.Bytes || !utf8.Valid(raw) || !historyJSONValid(raw, schema.Depth) {
		return nil, 0, ErrHistoryInvalid
	}
	p := historyParser{raw: raw, out: dst[:0], offsets: offsets[:0], schema: schema}
	if err := p.value(schema.Type, 1, true); err != nil {
		return nil, 0, err
	}
	p.space()
	if p.pos != len(raw) {
		return nil, 0, ErrHistoryInvalid
	}
	if schema.Check != nil {
		if err := schema.Check(p.out); err != nil {
			return nil, 0, ErrHistoryInvalid
		}
	}
	return p.out, p.stripped, nil
}

// Validate grammar with a fixed stack before the offset walk. In particular,
// encoding/json's pooled scanner never grows from an unknown raw subtree.
// The outer classifier preserves encoding/json's ordinary 10,000-depth limit;
// each selected history value then uses its much smaller J2 depth bound.
func historyJSONValid(raw []byte, limit int) bool {
	if limit < 1 || limit > 10000 {
		return false
	}
	var stack [10001]byte
	depth, i := 0, 0
	space := func() {
		for i < len(raw) && (raw[i] == ' ' || raw[i] == '\n' || raw[i] == '\r' || raw[i] == '\t') {
			i++
		}
	}
	stringToken := func() bool {
		if i >= len(raw) || raw[i] != '"' {
			return false
		}
		i++
		for i < len(raw) {
			c := raw[i]
			i++
			if c == '"' {
				return true
			}
			if c < 32 {
				return false
			}
			if c != '\\' {
				continue
			}
			if i == len(raw) {
				return false
			}
			c = raw[i]
			i++
			switch c {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
			case 'u':
				if len(raw)-i < 4 {
					return false
				}
				for end := i + 4; i < end; i++ {
					c = raw[i]
					if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
						return false
					}
				}
			default:
				return false
			}
		}
		return false
	}
	for {
		space()
		if i >= len(raw) {
			return depth == 0 && stack[0] == 9
		}
		switch stack[depth] {
		case 9:
			return false
		case 1, 5: // object key (first or after comma)
			if stack[depth] == 1 && raw[i] == '}' {
				depth--
				i++
				continue
			}
			if !stringToken() {
				return false
			}
			stack[depth] = 2
			continue
		case 2:
			if raw[i] != ':' {
				return false
			}
			i++
			stack[depth] = 3
			continue
		case 4, 7: // object/array delimiter
			end, next := byte('}'), byte(5)
			if stack[depth] == 7 {
				end, next = ']', 8
			}
			if raw[i] == end {
				depth--
				i++
				continue
			}
			if raw[i] != ',' {
				return false
			}
			i++
			stack[depth] = next
			continue
		case 0, 3, 6, 8:
			if stack[depth] == 6 && raw[i] == ']' {
				depth--
				i++
				continue
			}
			next := byte(7)
			if stack[depth] == 0 {
				next = 9
			}
			if stack[depth] == 3 {
				next = 4
			}
			stack[depth] = next
			c := raw[i]
			switch c {
			case '{', '[':
				if depth == limit {
					return false
				}
				depth++
				stack[depth] = 1
				if c == '[' {
					stack[depth] = 6
				}
				i++
			case '"':
				if !stringToken() {
					return false
				}
			case 't', 'f', 'n':
				word := "null"
				if c == 't' {
					word = "true"
				}
				if c == 'f' {
					word = "false"
				}
				if len(raw)-i < len(word) || string(raw[i:i+len(word)]) != word {
					return false
				}
				i += len(word)
			default:
				if c == '-' {
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
		default:
			return false
		}
	}
}

type historyParser struct {
	raw, out          []byte
	pos               int
	offsets           []uint32
	schema            HistoryProjection
	members, stripped int
}

func (p *historyParser) space() {
	for p.pos < len(p.raw) && (p.raw[p.pos] == ' ' || p.raw[p.pos] == '\n' || p.raw[p.pos] == '\r' || p.raw[p.pos] == '\t') {
		p.pos++
	}
}

func (p *historyParser) emit(b []byte, keep bool) error {
	if !keep {
		return nil
	}
	if len(b) > cap(p.out)-len(p.out) {
		return ErrHistoryInvalid
	}
	p.out = append(p.out, b...)
	return nil
}

// stringEnd and skipValue are used only after grammar validation. They never decode or
// allocate an unknown token. skipValue uses a scalar nesting counter.
func stringEnd(raw []byte, start int) int {
	i := start + 1
	for i < len(raw) {
		if raw[i] == '\\' {
			i += 2
			continue
		}
		if raw[i] == '"' {
			return i + 1
		}
		i++
	}
	return len(raw)
}

func skipValue(raw []byte, start int) int {
	i, depth := start, 0
	for i < len(raw) {
		switch raw[i] {
		case '"':
			i = stringEnd(raw, i)
			if depth == 0 {
				return i
			}
			continue
		case '{', '[':
			depth++
		case '}', ']':
			if depth == 0 {
				return i
			}
			depth--
			if depth == 0 {
				return i + 1
			}
		case ',', ' ', '\r', '\n', '\t':
			if depth == 0 {
				return i
			}
		}
		i++
	}
	return i
}

// Decode one JSON string rune at a time, including encoding/json's replacement
// behavior for unpaired surrogates, without retaining decoded key names.
func keyRune(raw []byte, pos *int) rune {
	i := *pos
	if raw[i] != '\\' {
		r, n := utf8.DecodeRune(raw[i:])
		*pos += n
		return r
	}
	i++
	c := raw[i]
	i++
	if c != 'u' {
		*pos = i
		switch c {
		case 'b':
			return '\b'
		case 'f':
			return '\f'
		case 'n':
			return '\n'
		case 'r':
			return '\r'
		case 't':
			return '\t'
		}
		return rune(c)
	}
	n, _ := strconv.ParseUint(string(raw[i:i+4]), 16, 16)
	i += 4
	r := rune(n)
	if r >= 0xD800 && r <= 0xDBFF && i+6 <= len(raw) && raw[i] == '\\' && raw[i+1] == 'u' {
		low, _ := strconv.ParseUint(string(raw[i+2:i+6]), 16, 16)
		if low >= 0xDC00 && low <= 0xDFFF {
			r = 0x10000 + (r-0xD800)*1024 + rune(low) - 0xDC00
			i += 6
		}
	}
	if r >= 0xD800 && r <= 0xDFFF {
		r = utf8.RuneError
	}
	*pos = i
	return r
}

func compareKeys(raw []byte, a, b int) int {
	i, j := a+1, b+1
	for raw[i] != '"' && raw[j] != '"' {
		x, y := keyRune(raw, &i), keyRune(raw, &j)
		if x < y {
			return -1
		}
		if x > y {
			return 1
		}
	}
	if raw[i] == '"' && raw[j] == '"' {
		return 0
	}
	if raw[i] == '"' {
		return -1
	}
	return 1
}

func keyIs(raw []byte, start int, name string) bool {
	i := start + 1
	for _, r := range name {
		if raw[i] == '"' || keyRune(raw, &i) != r {
			return false
		}
	}
	return raw[i] == '"'
}

func (p *historyParser) value(typ reflect.Type, level int, keep bool) error {
	p.space()
	start := p.pos
	if keep && typ != nil && typ == p.schema.RecordType {
		if skipValue(p.raw, start)-start > p.schema.RecordBytes {
			return ErrHistoryInvalid
		}
		previous, members := p.schema, p.members
		p.schema.Depth, p.schema.Members, p.schema.ArrayLimit = previous.RecordDepth, previous.RecordMembers, 32
		p.schema.RecordType = nil
		p.members = 0
		outputStart := len(p.out)
		err := p.value(typ, 1, keep)
		p.schema = previous
		p.members += members
		if err != nil {
			return err
		}
		if previous.CheckRecord != nil && previous.CheckRecord(p.out[outputStart:]) != nil {
			return ErrHistoryInvalid
		}
		return nil
	}
	if p.raw[start] == 'n' {
		if keep && typ != nil && typ.Kind() != reflect.Pointer {
			return ErrHistoryInvalid
		}
		p.pos += 4
		return p.emit(p.raw[start:p.pos], keep)
	}
	if typ != nil && typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	switch p.raw[start] {
	case '{':
		if level > p.schema.Depth || (keep && typ != nil && typ.Kind() != reflect.Struct) {
			return ErrHistoryInvalid
		}
		return p.object(typ, level, keep)
	case '[':
		if level > p.schema.Depth || (keep && typ != nil && typ.Kind() != reflect.Slice) {
			return ErrHistoryInvalid
		}
		p.pos++
		if err := p.emit([]byte{'['}, keep); err != nil {
			return err
		}
		p.space()
		count := 0
		for p.raw[p.pos] != ']' {
			if count > 0 {
				p.pos++
				if err := p.emit([]byte{','}, keep); err != nil {
					return err
				}
			}
			count++
			if keep && count > p.schema.ArrayLimit {
				return ErrHistoryInvalid
			}
			var elem reflect.Type
			if typ != nil {
				elem = typ.Elem()
			}
			if err := p.value(elem, level+1, keep); err != nil {
				return err
			}
			p.space()
		}
		p.pos++
		return p.emit([]byte{']'}, keep)
	default:
		p.pos = skipValue(p.raw, start)
		if keep && typ != nil {
			kind := typ.Kind()
			c := p.raw[start]
			if (kind == reflect.String && c != '"') || (kind == reflect.Bool && c != 't' && c != 'f') ||
				(kind != reflect.String && kind != reflect.Bool && kind != reflect.Int && kind != reflect.Int64) ||
				((kind == reflect.Int || kind == reflect.Int64) && (c < '0' || c > '9') && c != '-') {
				return ErrHistoryInvalid
			}
			if kind == reflect.String {
				i, n := start+1, 0
				for p.raw[i] != '"' {
					n += utf8.RuneLen(keyRune(p.raw, &i))
					if n > 512 {
						return ErrHistoryInvalid
					}
				}
			}
			if (kind == reflect.Int || kind == reflect.Int64) && p.schema.Integer != nil {
				if _, ok := p.schema.Integer(p.raw[start:p.pos]); !ok {
					return ErrHistoryInvalid
				}
			}
		}
		return p.emit(p.raw[start:p.pos], keep)
	}
}

// MaterializeHistory materializes an already projected value into a typed
// destination without invoking custom UnmarshalJSON or constructing maps. Call
// only after ProjectHistory has checked shape, counts, scalar bounds and nulls.
// This also lets a schema reuse its existing typed semantic Validate method.
func MaterializeHistory(projected []byte, dst any, integer func([]byte) (int64, bool)) error {
	value := reflect.ValueOf(dst)
	if value.Kind() != reflect.Pointer || value.IsNil() || !historyJSONValid(projected, 20) {
		return ErrHistoryInvalid
	}
	p := historyMaterializer{raw: projected, integer: integer}
	return p.value(value.Elem())
}

type historyMaterializer struct {
	raw     []byte
	pos     int
	integer func([]byte) (int64, bool)
}

func (p *historyMaterializer) space() {
	for p.pos < len(p.raw) && (p.raw[p.pos] == ' ' || p.raw[p.pos] == '\n' || p.raw[p.pos] == '\r' || p.raw[p.pos] == '\t') {
		p.pos++
	}
}

func (p *historyMaterializer) value(dst reflect.Value) error {
	p.space()
	if p.raw[p.pos] == 'n' {
		if dst.Kind() != reflect.Pointer {
			return ErrHistoryInvalid
		}
		dst.SetZero()
		p.pos += 4
		return nil
	}
	if dst.Kind() == reflect.Pointer {
		dst.Set(reflect.New(dst.Type().Elem()))
		return p.value(dst.Elem())
	}
	switch dst.Kind() {
	case reflect.Struct:
		if p.raw[p.pos] != '{' {
			return ErrHistoryInvalid
		}
		p.pos++
		p.space()
		for p.raw[p.pos] != '}' {
			key := p.pos
			p.pos = stringEnd(p.raw, key)
			p.space()
			p.pos++
			index := -1
			for i := 0; i < dst.NumField(); i++ {
				if keyIs(p.raw, key, dst.Type().Field(i).Tag.Get("json")) {
					index = i
					break
				}
			}
			if index < 0 {
				return ErrHistoryInvalid
			}
			if err := p.value(dst.Field(index)); err != nil {
				return err
			}
			p.space()
			if p.raw[p.pos] == ',' {
				p.pos++
				p.space()
			}
		}
		p.pos++
		return nil
	case reflect.Slice:
		if p.raw[p.pos] != '[' {
			return ErrHistoryInvalid
		}
		i, count := p.pos+1, 0
		for {
			for p.raw[i] == ' ' || p.raw[i] == '\n' || p.raw[i] == '\r' || p.raw[i] == '\t' {
				i++
			}
			if p.raw[i] == ']' {
				break
			}
			count++
			if count > 256 {
				return ErrHistoryInvalid
			}
			i = skipValue(p.raw, i)
			for p.raw[i] == ' ' || p.raw[i] == '\n' || p.raw[i] == '\r' || p.raw[i] == '\t' {
				i++
			}
			if p.raw[i] == ']' {
				break
			}
			i++
		}
		dst.Set(reflect.MakeSlice(dst.Type(), count, count))
		p.pos++
		for i := 0; i < count; i++ {
			if i > 0 {
				p.space()
				p.pos++
			}
			if err := p.value(dst.Index(i)); err != nil {
				return err
			}
		}
		p.space()
		p.pos++
		return nil
	case reflect.String:
		if p.raw[p.pos] != '"' {
			return ErrHistoryInvalid
		}
		end := stringEnd(p.raw, p.pos)
		if end-p.pos > 3074 {
			return ErrHistoryInvalid
		}
		var value string
		if json.Unmarshal(p.raw[p.pos:end], &value) != nil || len(value) > 512 {
			return ErrHistoryInvalid
		}
		dst.SetString(value)
		p.pos = end
		return nil
	case reflect.Bool:
		if p.raw[p.pos] == 't' {
			dst.SetBool(true)
			p.pos += 4
			return nil
		}
		if p.raw[p.pos] == 'f' {
			dst.SetBool(false)
			p.pos += 5
			return nil
		}
		return ErrHistoryInvalid
	case reflect.Int, reflect.Int64:
		end := skipValue(p.raw, p.pos)
		if p.integer == nil {
			return ErrHistoryInvalid
		}
		n, ok := p.integer(p.raw[p.pos:end])
		if !ok || dst.OverflowInt(n) {
			return ErrHistoryInvalid
		}
		dst.SetInt(n)
		p.pos = end
		return nil
	default:
		return ErrHistoryInvalid
	}
}

func (p *historyParser) object(typ reflect.Type, level int, keep bool) error {
	// Reserve this object's contiguous offset range before descending. All key
	// ranges are stack-owned and reclaimed after the duplicate sort.
	base := len(p.offsets)
	i := p.pos + 1
	for {
		for p.raw[i] == ' ' || p.raw[i] == '\t' || p.raw[i] == '\n' || p.raw[i] == '\r' {
			i++
		}
		if p.raw[i] == '}' {
			break
		}
		if len(p.offsets) == cap(p.offsets) {
			return ErrHistoryInvalid
		}
		p.offsets = append(p.offsets, uint32(i))
		i = stringEnd(p.raw, i)
		for p.raw[i] != ':' {
			i++
		}
		i++
		for p.raw[i] == ' ' || p.raw[i] == '\t' || p.raw[i] == '\n' || p.raw[i] == '\r' {
			i++
		}
		i = skipValue(p.raw, i)
		for p.raw[i] == ' ' || p.raw[i] == '\t' || p.raw[i] == '\n' || p.raw[i] == '\r' {
			i++
		}
		if p.raw[i] == '}' {
			break
		}
		i++
	}
	end := len(p.offsets)
	p.members += end - base
	if p.members > p.schema.Members {
		return ErrHistoryInvalid
	}
	keys := p.offsets[base:end]
	sort.Slice(keys, func(i, j int) bool { return compareKeys(p.raw, int(keys[i]), int(keys[j])) < 0 })
	for i := 1; i < len(keys); i++ {
		if compareKeys(p.raw, int(keys[i-1]), int(keys[i])) == 0 {
			return ErrHistoryInvalid
		}
	}
	if err := p.emit([]byte{'{'}, keep); err != nil {
		return err
	}
	p.pos++
	p.space()
	var seen uint64
	written := 0
	for p.raw[p.pos] != '}' {
		key := p.pos
		keyEnd := stringEnd(p.raw, key)
		p.pos = keyEnd
		p.space()
		p.pos++
		index := -1
		if keep && typ != nil {
			if typ.NumField() > 63 {
				return ErrHistoryInvalid
			}
			for f := 0; f < typ.NumField(); f++ {
				if keyIs(p.raw, key, typ.Field(f).Tag.Get("json")) {
					index = f
					break
				}
			}
		}
		known := keep && index >= 0
		var field reflect.Type
		if known {
			seen |= uint64(1) << index
			field = typ.Field(index).Type
			if written > 0 {
				if err := p.emit([]byte{','}, true); err != nil {
					return err
				}
			}
			written++
			if err := p.emit(p.raw[key:keyEnd], true); err != nil {
				return err
			}
			if err := p.emit([]byte{':'}, true); err != nil {
				return err
			}
		} else if keep {
			p.stripped++
		}
		if err := p.value(field, level+1, known); err != nil {
			return err
		}
		p.space()
		if p.raw[p.pos] == ',' {
			p.pos++
			p.space()
		}
	}
	p.pos++
	if keep && typ != nil && seen != uint64(1)<<typ.NumField()-1 {
		return ErrHistoryInvalid
	}
	p.offsets = p.offsets[:base]
	return p.emit([]byte{'}'}, keep)
}

// HistoryCorrelation has one live initial and one retired initial slot. Retire
// does not release a transfer: clear it only when its actual borrower releases.
// The connection owner serializes it with reads and pending-call changes.
type HistoryCorrelation struct {
	pending                      [HistoryIDBytes]byte
	retired                      [HistoryIDBytes]byte
	pendingN, retiredN           int
	pendingMethod, retiredMethod string
	quarantined                  bool
	refetchQueued                bool
}

func validHistoryID(id []byte) bool {
	if len(id) == 0 || len(id) > HistoryIDBytes || !bytes.Equal(bytes.TrimSpace(id), id) {
		return false
	}
	if id[0] != '"' && id[0] != '-' && (id[0] < '0' || id[0] > '9') {
		return false
	}
	return utf8.Valid(id) && historyJSONValid(id, 1)
}

func (c *HistoryCorrelation) Begin(id []byte) error {
	return c.BeginMethod("routing:get-initial", id)
}

// Each public history call returns an initial shape. Exactly one such call may
// own the connection's initial transfer; ordinary pending calls stay elsewhere.
func (c *HistoryCorrelation) BeginMethod(method string, id []byte) error {
	switch method {
	case "routing:get-initial":
		method = "routing:get-initial"
	case "routing:subscribe":
		method = "routing:subscribe"
	case "routing:unsubscribe":
		method = "routing:unsubscribe"
	default:
		return ErrHistoryInvalid
	}
	if !validHistoryID(id) {
		return ErrHistoryInvalid
	}
	if c.pendingN != 0 || c.retiredN != 0 {
		return ErrHistoryBusy
	}
	c.pendingN = copy(c.pending[:], id)
	c.pendingMethod = method
	c.refetchQueued = false
	return nil
}

func (c *HistoryCorrelation) Retire() {
	if c.pendingN == 0 {
		return
	}
	c.retiredN = copy(c.retired[:], c.pending[:c.pendingN])
	c.pendingN = 0
	c.retiredMethod = c.pendingMethod
	c.pendingMethod = ""
}

func (c *HistoryCorrelation) ReleaseRetired() { c.retiredN = 0; c.retiredMethod = "" }
func (c *HistoryCorrelation) Complete() {
	c.pendingN = 0
	c.pendingMethod = ""
	c.quarantined = false
	c.refetchQueued = false
}
func (c *HistoryCorrelation) Quarantined() bool { return c.quarantined }

func equalID(a, b []byte) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	if a[0] == '"' && b[0] == '"' {
		// Compare before applying the encoded cap: an over-cap escaped spelling
		// can still identify the bounded pending ID and must fail that call.
		i, j := 1, 1
		for i < len(a)-1 && j < len(b)-1 {
			if keyRune(a, &i) != keyRune(b, &j) {
				return false
			}
		}
		return i == len(a)-1 && j == len(b)-1
	}
	return bytes.Equal(a, b)
}

func historyMethod(raw []byte) string {
	for _, m := range []string{"routing:get-initial", "routing:changed", "routing/history-window", "routing/request", "routing/history-window-end", "routing/history-accepted", "routing:subscribe", "routing:unsubscribe"} {
		if len(raw) > 1 && raw[0] == '"' && keyIs(raw, 0, m) {
			return m
		}
	}
	return ""
}

// Only the outer struct envelope follows encoding/json's case folding. J2
// projections and unknown subtree keys remain case-sensitive. Compare decoded
// runes directly so escaped names and Unicode simple-fold aliases cannot evade
// classification, without allocating unknown key strings.
func historyOuterKeyIs(raw []byte, start int, name string) bool {
	fold := func(r rune) rune {
		for {
			next := unicode.SimpleFold(r)
			if next <= r {
				return next
			}
			r = next
		}
	}
	i := start + 1
	for _, want := range name {
		if raw[i] == '"' || fold(keyRune(raw, &i)) != fold(want) {
			return false
		}
	}
	return raw[i] == '"'
}

const (
	historyVersionKey = iota
	historyIDKey
	historyMethodKey
	historyParamsKey
	historyResultKey
	historyErrorKey
	historyEnvelopeFields
)

func historyOuterKey(raw []byte, start int) int {
	for i, name := range [...]string{"jsonrpc", "id", "method", "params", "result", "error"} {
		if historyOuterKeyIs(raw, start, name) {
			return i
		}
	}
	return -1
}

// HistorySchema is supplied by nodepolicy, keeping this package dependency-lower.
type HistorySchema func(method string, initial bool) (HistoryProjection, bool)

// HistoryReader leases raw/output/offsets from its owner. Message values borrow
// output until the next Read; an owner must finish all use before reusing it.
type HistoryReader struct {
	r           *bufio.Reader
	raw, output []byte
	offsets     []uint32
	ordinary    int
	correlation *HistoryCorrelation
	schema      HistorySchema
	idToken     [HistoryIDBytes]byte
}

func NewHistoryCodec(rw io.ReadWriter, ordinary int, raw, output []byte, offsets []uint32, correlation *HistoryCorrelation, schema HistorySchema, interrupt func()) (*Codec, error) {
	if output == nil {
		output = raw
	}
	if ordinary < 1 || ordinary > WorkerFrameBytes || cap(raw) < max(ordinary, HistoryRecordFrameBytes) || cap(output) < HistoryRecordFrameBytes || cap(offsets)*4 > HistoryInitialBytes || cap(offsets) < HistoryRecordBytes/4 || correlation == nil || schema == nil {
		return nil, ErrHistoryInvalid
	}
	return &Codec{writer: rw, wgate: make(chan struct{}, 1), interruptWrite: interrupt, history: &HistoryReader{r: bufio.NewReaderSize(rw, 4096), raw: raw[:0:min(cap(raw), HistoryFrameBytes)], output: output[:0:min(cap(output), HistoryFrameBytes)], offsets: offsets[:0], ordinary: ordinary, correlation: correlation, schema: schema}}, nil
}

func (r *HistoryReader) Read() (*Message, error) {
	// Private source links need record/ordinary backing only. A hop that can
	// receive initial results must acquire its H/M leases before tracking a call.
	if (r.correlation.pendingN != 0 || r.correlation.retiredN != 0) && (cap(r.raw) < HistoryFrameBytes || cap(r.output) < HistoryInitialBytes || cap(r.offsets)*4 < HistoryInitialBytes) {
		return nil, ErrHistoryBusy
	}
	r.raw = r.raw[:0]
	total, discarded := 0, false
	for {
		part, err := r.r.ReadSlice('\n')
		total += len(part)
		if total > HistoryFrameBytes {
			return nil, ErrHistoryFraming
		}
		if len(part) > cap(r.raw)-len(r.raw) {
			discarded = true
		}
		if !discarded {
			r.raw = append(r.raw, part...)
		}
		if err == nil {
			break
		}
		if err == bufio.ErrBufferFull {
			if total == HistoryFrameBytes {
				return nil, ErrHistoryFraming
			}
			continue
		}
		if err == io.EOF && total == 0 {
			return nil, io.EOF
		}
		return nil, ErrHistoryFraming
	}
	if discarded {
		return r.quarantine()
	}
	return r.guard(r.raw[:len(r.raw)-1])
}

func (r *HistoryReader) quarantine() (*Message, error) {
	failPending := r.correlation.pendingN != 0
	scheduleRefetch := !r.correlation.refetchQueued
	r.correlation.quarantined = true
	r.correlation.refetchQueued = true
	if failPending {
		r.correlation.Retire()
	}
	return nil, &HistoryFailure{Disposition: "quarantine", Category: "history-invalid", Count: 1, FailPending: failPending, ScheduleRefetch: scheduleRefetch}
}

func (r *HistoryReader) guard(raw []byte) (*Message, error) {
	if !utf8.Valid(raw) || !historyJSONValid(raw, 10000) {
		return r.quarantine()
	}
	i := 0
	for i < len(raw) && bytes.IndexByte([]byte(" \r\t\n"), raw[i]) >= 0 {
		i++
	}
	if i == len(raw) || raw[i] != '{' {
		return r.quarantine()
	}
	i++
	var id, value, errorValue, versionValue []byte
	valueName := ""
	method := ""
	selected, initial, ambiguous := false, false, false
	ids, methods, values := 0, 0, 0
	var outerSeen [historyEnvelopeFields]bool
	duplicateEnvelope := false
	for {
		for raw[i] == ' ' || raw[i] == '\r' || raw[i] == '\t' || raw[i] == '\n' {
			i++
		}
		if raw[i] == '}' {
			break
		}
		key := i
		i = stringEnd(raw, i)
		for raw[i] != ':' {
			i++
		}
		i++
		for raw[i] == ' ' || raw[i] == '\r' || raw[i] == '\t' || raw[i] == '\n' {
			i++
		}
		end := skipValue(raw, i)
		token := raw[i:end]
		alias := historyOuterKey(raw, key)
		if alias >= 0 {
			if outerSeen[alias] {
				duplicateEnvelope = true
			}
			outerSeen[alias] = true
		}
		switch alias {
		case historyMethodKey:
			methods++
			if methods > 1 {
				ambiguous = true
			}
			if m := historyMethod(token); m != "" {
				selected = true
				method = m
			}
		case historyIDKey:
			ids++
			if ids > 1 {
				ambiguous = true
			}
			id = token
			if equalID(token, r.correlation.pending[:r.correlation.pendingN]) || equalID(token, r.correlation.retired[:r.correlation.retiredN]) {
				selected = true
				initial = true
			}
		case historyParamsKey, historyResultKey:
			values++
			value = token
			valueName = "params"
			if alias == historyResultKey {
				valueName = "result"
			}
		case historyErrorKey:
			errorValue = token
		case historyVersionKey:
			versionValue = token
		}
		i = end
		for raw[i] == ' ' || raw[i] == '\r' || raw[i] == '\t' || raw[i] == '\n' {
			i++
		}
		if raw[i] == ',' {
			i++
		}
	}
	if !selected {
		if len(raw)+1 > r.ordinary {
			return nil, &DecodeError{ErrHistoryInvalid}
		}
		var msg Message
		if err := json.Unmarshal(raw, &msg); err != nil || msg.JSONRPC != "2.0" {
			return nil, &DecodeError{ErrHistoryInvalid}
		}
		return &msg, nil
	}
	if ambiguous {
		return r.quarantine()
	}
	failure := func() (*Message, error) {
		if initial && equalID(id, r.correlation.pending[:r.correlation.pendingN]) {
			r.correlation.Retire()
			return nil, &HistoryFailure{Disposition: "fail-call", Category: "history-invalid", Count: 1, FailPending: true}
		}
		return nil, historyFailure("discard", "history-invalid")
	}
	if duplicateEnvelope {
		return failure()
	}
	if ids > 0 && !validHistoryID(id) {
		return failure()
	}
	if initial && methods != 0 {
		return failure()
	}
	if !initial && (methods != 1 || valueName != "params") {
		return failure()
	}
	if !initial && method != "routing:get-initial" && method != "routing:subscribe" && method != "routing:unsubscribe" && ids != 0 {
		return failure()
	}
	if !initial && (method == "routing:get-initial" || method == "routing:subscribe" || method == "routing:unsubscribe") && ids != 1 {
		return failure()
	}
	if errorValue != nil {
		if !initial || values != 0 || len(errorValue) > 512 || len(raw)+1 > HistoryEnvelopeBytes {
			return failure()
		}
	} else if values != 1 || (initial && valueName != "result") || len(raw)-len(value)+1 > HistoryEnvelopeBytes {
		return failure()
	}
	schemaMethod := method
	if initial {
		schemaMethod = r.correlation.pendingMethod
		if equalID(id, r.correlation.retired[:r.correlation.retiredN]) {
			schemaMethod = r.correlation.retiredMethod
		}
	}
	schema, ok := r.schema(schemaMethod, initial)
	if !ok {
		return failure()
	}
	// Validate outer duplicates (including unknown subtrees) using the same raw
	// parser. Projection itself is done directly into the owner's output lease.
	outer := HistoryProjection{Bytes: HistoryFrameBytes, Depth: schema.Depth + 2, Members: HistoryInitialBytes, ArrayLimit: HistoryInitialBytes}
	p := historyParser{raw: raw, offsets: r.offsets[:0], schema: outer}
	if err := p.value(nil, 1, false); err != nil {
		return failure()
	}
	var version string
	if json.Unmarshal(versionValue, &version) != nil || version != "2.0" {
		return failure()
	}
	// Envelope data is consumed before projection, which can be in-place. IDs
	// retain their exact encoded spelling in the fixed control token slot.
	if ids != 0 {
		n := copy(r.idToken[:], id)
		id = r.idToken[:n]
	}
	var projected []byte
	var rpcError *RPCError
	if errorValue != nil {
		// Error metadata follows the same strict known-field projection as
		// history values. A generic RPCError decode would case-fold unknown
		// fields such as Message onto the validated message field.
		var e struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		}
		errorSchema := HistoryProjection{
			Type: reflect.TypeOf(e), Bytes: 512, Depth: 16, Members: 2048, ArrayLimit: 2048,
			Integer: func(raw []byte) (int64, bool) {
				value, err := strconv.ParseInt(string(raw), 10, 64)
				return value, err == nil
			},
		}
		clean, _, err := ProjectHistory(errorValue, r.output, r.offsets, errorSchema)
		if err != nil || MaterializeHistory(clean, &e, errorSchema.Integer) != nil || !((e.Code == HistoryBusyCode && e.Message == HistoryBusyMessage) || (e.Code == HistoryInvalidCode && e.Message == "history-invalid")) {
			return failure()
		}
		rpcError = &RPCError{Code: e.Code, Message: e.Message}
	} else {
		var err error
		projected, _, err = ProjectHistory(value, r.output, r.offsets, schema)
		if err != nil {
			return failure()
		}
	}
	// Envelope scalars are bounded by E. No RawMessage copy of the large value.
	msg := &Message{JSONRPC: "2.0", Method: method, Error: rpcError}
	if ids != 0 {
		token := json.RawMessage(id)
		msg.ID = &token
	}
	if initial {
		msg.Result = projected
	} else {
		msg.Params = projected
	}
	if initial && equalID(id, r.correlation.retired[:r.correlation.retiredN]) {
		return nil, historyFailure("discard", "history-stale")
	}
	return msg, nil
}

// WritePreencoded borrows a complete newline frame. acquire must be bounded by
// its owner; parent is the transport lifetime, never a history-only timeout.
// A nil acquire attempts the gate once and returns busy without queueing.
func (c *Codec) WritePreencoded(acquire, parent context.Context, frame []byte) error {
	if len(frame) < 2 || len(frame) > HistoryFrameBytes || frame[len(frame)-1] != '\n' || bytes.IndexByte(frame[:len(frame)-1], '\n') >= 0 || !utf8.Valid(frame) || !historyJSONValid(frame[:len(frame)-1], 10000) {
		return ErrHistoryInvalid
	}
	if acquire != nil {
		return c.writeEncodedContext(acquire, parent, nil, c.interruptWrite, frame)
	}
	select {
	case c.wgate <- struct{}{}:
		defer func() { <-c.wgate }()
		return c.writeOwnedContext(parent, parent, nil, c.interruptWrite, frame)
	default:
		return ErrHistoryBusy
	}
}
