// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"math/bits"
	"reflect"
	"strings"

	"nvpair-shared/jsonrpc"
)

// Private DTOs are process-bound transport primitives. Their syntax does not
// establish a trusted origin, permission, native storage grant or apply fence.
type RoutingHistoryWindowKey struct {
	Request   RoutingRequestKey `json:"request"`
	CreatedAt int64             `json:"createdAt"`
	Seq       int64             `json:"seq"`
}

type RoutingHistoryCutoff struct {
	Request   RoutingRequestKey `json:"request"`
	CreatedAt int64             `json:"createdAt"`
}

type RoutingHistoryWindow struct {
	Schema     int64                     `json:"schema"`
	Generation string                    `json:"generation"`
	Cutoff     *RoutingHistoryCutoff     `json:"cutoff"`
	Keys       []RoutingHistoryWindowKey `json:"keys"`
}

// End and accepted deliberately share the exact validator and shape.
type RoutingHistoryWindowEnd struct {
	Schema     int64  `json:"schema"`
	Generation string `json:"generation"`
}

type RoutingHistoryAccepted = RoutingHistoryWindowEnd

func ValidateRoutingHistoryGeneration(generation string) error {
	const maximum = "340282366920938463463374607431768211455"
	if len(generation) == 0 || len(generation) > len(maximum) || generation[0] == '0' {
		return historyError("invalid-value")
	}
	for _, digit := range generation {
		if digit < '0' || digit > '9' {
			return historyError("invalid-value")
		}
	}
	if len(generation) == len(maximum) && generation > maximum {
		return historyError("invalid-value")
	}
	return nil
}

// RoutingHistoryGeneration is owned by the sole copier; it is not atomic and
// clocks never mint its values. Overflow leaves the old value intact.
type RoutingHistoryGeneration struct{ High, Low uint64 }

func (g *RoutingHistoryGeneration) Next() (string, error) {
	low, carry := bits.Add64(g.Low, 1, 0)
	high, overflow := bits.Add64(g.High, 0, carry)
	if overflow != 0 {
		return "", historyError("limit-exceeded")
	}
	g.High, g.Low = high, low
	var digits [39]byte
	pos := len(digits)
	for high != 0 || low != 0 {
		qhigh, rem := bits.Div64(0, high, 10)
		qlow, digit := bits.Div64(rem, low, 10)
		pos--
		digits[pos] = byte(digit) + '0'
		high, low = qhigh, qlow
	}
	return string(digits[pos:]), nil
}

func compareHistoryKeys(a, b RoutingRequestKey) int {
	if c := strings.Compare(a.OriginNodeID, b.OriginNodeID); c != 0 {
		return c
	}
	if c := strings.Compare(a.ProxyRunID, b.ProxyRunID); c != 0 {
		return c
	}
	if len(a.RequestID) < len(b.RequestID) {
		return -1
	}
	if len(a.RequestID) > len(b.RequestID) {
		return 1
	}
	return strings.Compare(a.RequestID, b.RequestID)
}

func (w RoutingHistoryWindow) Validate() error {
	i := 0
	return validateHistoryWindow(w.Schema, w.Generation, w.Keys != nil, len(w.Keys), w.Cutoff, func() (RoutingHistoryWindowKey, error) {
		key := w.Keys[i]
		i++
		return key, nil
	})
}

func validateHistoryWindow(schema int64, generation string, present bool, count int, cutoff *RoutingHistoryCutoff, next func() (RoutingHistoryWindowKey, error)) error {
	if schema != 1 || ValidateRoutingHistoryGeneration(generation) != nil || !present || count > RoutingHistoryRecords {
		return historyError("invalid-value")
	}
	var previous, least RoutingHistoryWindowKey
	for i := 0; i < count; i++ {
		key, err := next()
		if err != nil {
			return err
		}
		if key.Request.Validate() != nil || !historyNumber(key.CreatedAt) || !historyNumber(key.Seq) || key.Seq == 0 || (i > 0 && compareHistoryKeys(previous.Request, key.Request) >= 0) {
			return historyError("invalid-value")
		}
		if i == 0 || key.CreatedAt < least.CreatedAt || (key.CreatedAt == least.CreatedAt && compareHistoryKeys(key.Request, least.Request) < 0) {
			least = key
		}
		previous = key
	}
	if count < RoutingHistoryRecords {
		if cutoff != nil {
			return historyError("invalid-value")
		}
	} else if cutoff == nil || cutoff.Request != least.Request || cutoff.CreatedAt != least.CreatedAt {
		return historyError("invalid-value")
	}
	return nil
}

func (w RoutingHistoryWindowEnd) Validate() error {
	if w.Schema != 1 {
		return historyError("invalid-value")
	}
	return ValidateRoutingHistoryGeneration(w.Generation)
}

func (w *RoutingHistoryWindow) UnmarshalJSON(raw []byte) error {
	var next RoutingHistoryWindow
	if err := decodeHistoryPrivate(raw, &next); err != nil {
		return err
	}
	if err := next.Validate(); err != nil {
		return err
	}
	*w = next
	return nil
}

func (w *RoutingHistoryWindowEnd) UnmarshalJSON(raw []byte) error {
	var next RoutingHistoryWindowEnd
	if err := decodeHistoryPrivate(raw, &next); err != nil {
		return err
	}
	if err := next.Validate(); err != nil {
		return err
	}
	*w = next
	return nil
}

// Decode directly into fields so private UnmarshalJSON cannot recurse. The
// streaming guard strips unknowns first; no unknown-heavy generic tree exists.
func decodeHistoryPrivate(raw []byte, dst any) error {
	typ := reflect.TypeOf(dst).Elem()
	schema := historyProjection(typ, false)
	schema.Check = nil
	if len(raw) > RoutingHistoryRecordBytes {
		return historyError("limit-exceeded")
	}
	output := make([]byte, 0, len(raw))
	var offsets [RoutingHistoryRecordBytes / 4]uint32
	projected, _, err := jsonrpc.ProjectHistory(raw, output, offsets[:], schema)
	if err != nil {
		return historyError("invalid-shape")
	}
	if jsonrpc.MaterializeHistory(projected, dst, historyTransportInteger) != nil {
		return historyError("invalid-shape")
	}
	return nil
}

func historyProjection(typ reflect.Type, initial bool) jsonrpc.HistoryProjection {
	schema := jsonrpc.HistoryProjection{Type: typ, Bytes: RoutingHistoryRecordBytes, Depth: routingHistoryDepth, Members: routingHistoryMembers, ArrayLimit: RoutingHistoryRecords, Integer: historyTransportInteger}
	if initial {
		schema.Bytes, schema.Depth, schema.Members = RoutingHistoryInitialBytes, routingHistoryDepth+2, RoutingHistoryInitialBytes
		schema.RecordType = reflect.TypeOf(RoutingHistoryRecord{})
		schema.RecordBytes, schema.RecordDepth, schema.RecordMembers = RoutingHistoryRecordBytes, routingHistoryDepth, routingHistoryMembers
		schema.CheckRecord = validateHistoryTransportRecord
	} else if typ == reflect.TypeOf(RoutingHistoryRecord{}) {
		schema.ArrayLimit = RoutingHistoryAttempts
	}
	return schema
}

// RoutingHistoryTransportSchema supplies only known shapes and the existing J2
// semantic validator. It is not a second public J2 decoder or RPC registration.
func RoutingHistoryTransportSchema(method string, initial bool) (jsonrpc.HistoryProjection, bool) {
	if initial {
		s := historyProjection(reflect.TypeOf(RoutingHistoryInitial{}), true)
		return s, true
	}
	switch method {
	case MethodRoutingGetInitial, MethodRoutingSubscribe, MethodRoutingUnsubscribe:
		return historyProjection(reflect.TypeOf(struct{}{}), false), true
	case NotifyRoutingRequest, NotifyRoutingChanged:
		s := historyProjection(reflect.TypeOf(RoutingHistoryRecord{}), false)
		s.Check = validateHistoryTransportRecord
		return s, true
	case NotifyRoutingHistoryWindow:
		s := historyProjection(reflect.TypeOf(RoutingHistoryWindow{}), false)
		s.Check = func(raw []byte) error {
			_, err := ReadProjectedRoutingHistoryWindow(raw)
			return err
		}
		return s, true
	case NotifyRoutingHistoryWindowEnd, NotifyRoutingHistoryAccepted:
		s := historyProjection(reflect.TypeOf(RoutingHistoryWindowEnd{}), false)
		s.Check = func(raw []byte) error {
			var w RoutingHistoryWindowEnd
			if err := jsonrpc.MaterializeHistory(raw, &w, historyTransportInteger); err != nil {
				return err
			}
			return w.Validate()
		}
		return s, true
	}
	return jsonrpc.HistoryProjection{}, false
}

func validateHistoryTransportRecord(raw []byte) error {
	_, err := ReadProjectedRoutingHistoryRecord(raw)
	return err
}

func historyTransportInteger(raw []byte) (int64, bool) { return historyScalarInteger(raw) }

type historySpan struct{ start, end int }

// RoutingHistoryRecordView borrows only already-projected bytes. It has a
// bounded header and attempt offsets, never an Attempts DTO slice. Callers must
// finish use before releasing the ProjectHistory output lease.
type RoutingHistoryRecordView struct {
	Header   RoutingHistoryRecordHeader
	raw      []byte
	attempts [RoutingHistoryAttempts]historySpan
	count    int
}

func (v *RoutingHistoryRecordView) AttemptCount() int { return v.count }

// ReadAttempt replaces the caller's slot, dropping every prior referent before
// decoding. Nested semantic lookups use a separate related slot in the core.
func (v *RoutingHistoryRecordView) ReadAttempt(i int, dst *RoutingHistoryAttempt) error {
	*dst = RoutingHistoryAttempt{}
	if i < 0 || i >= v.count {
		return historyError("invalid-shape")
	}
	s := v.attempts[i]
	return jsonrpc.MaterializeHistory(v.raw[s.start:s.end], dst, historyTransportInteger)
}

// ReadProjectedRoutingHistoryRecord is a semantic adapter, not a raw JSON
// entry point: ProjectHistory must first enforce the original-byte grammar,
// shape, duplicate, unknown, null, depth, member and array limits. Only fixed
// spans are traversed here; the shared materializer decodes bounded fields.
func ReadProjectedRoutingHistoryRecord(raw []byte) (RoutingHistoryRecordView, error) {
	v := RoutingHistoryRecordView{raw: raw}
	header := reflect.ValueOf(&v.Header).Elem()
	typ := reflect.TypeOf(RoutingHistoryRecord{})
	present := false
	err := historyProjectedFields(raw, typ, func(i int, span historySpan) error {
		token := raw[span.start:span.end]
		if i < header.NumField() {
			return jsonrpc.MaterializeHistory(token, header.Field(i).Addr().Interface(), historyTransportInteger)
		}
		present = !bytes.Equal(bytes.TrimSpace(token), []byte("null"))
		if !present {
			return nil
		}
		cursor := historyProjectedCursor{raw: token, pos: 1}
		for {
			s, more, err := cursor.nextElement()
			if err != nil || !more {
				return err
			}
			if v.count == len(v.attempts) {
				return historyError("limit-exceeded")
			}
			v.attempts[v.count] = historySpan{span.start + s.start, span.start + s.end}
			v.count++
		}
	})
	if err != nil {
		return RoutingHistoryRecordView{}, err
	}
	if err := validateHistoryRecord(v.Header, present, v.count, v.ReadAttempt); err != nil {
		return RoutingHistoryRecordView{}, err
	}
	return v, nil
}

// RoutingHistoryWindowView retains a key-array byte span, not a Keys DTO
// slice. Validation holds only current/previous/least/cutoff key facts.
type RoutingHistoryWindowView struct {
	Generation string
	Cutoff     *RoutingHistoryCutoff
	keys       []byte
	count      int
}

func (v *RoutingHistoryWindowView) KeyCount() int { return v.count }

func (v *RoutingHistoryWindowView) ForEachKey(visit func(int, RoutingHistoryWindowKey) error) error {
	cursor := historyProjectedCursor{raw: v.keys, pos: 1}
	var key RoutingHistoryWindowKey
	for i := 0; i < v.count; i++ {
		span, more, err := cursor.nextElement()
		if err != nil || !more {
			return historyError("invalid-shape")
		}
		key = RoutingHistoryWindowKey{}
		if err := jsonrpc.MaterializeHistory(v.keys[span.start:span.end], &key, historyTransportInteger); err != nil {
			return err
		}
		if err := visit(i, key); err != nil {
			return err
		}
	}
	return nil
}

// As with the record view, raw must be ProjectHistory's checked output. The
// cursor locates spans only; it does not replace the original JSON guard.
func ReadProjectedRoutingHistoryWindow(raw []byte) (RoutingHistoryWindowView, error) {
	var v RoutingHistoryWindowView
	var schema int64
	err := historyProjectedFields(raw, reflect.TypeOf(RoutingHistoryWindow{}), func(i int, span historySpan) error {
		token := raw[span.start:span.end]
		switch i {
		case 0:
			return jsonrpc.MaterializeHistory(token, &schema, historyTransportInteger)
		case 1:
			return jsonrpc.MaterializeHistory(token, &v.Generation, historyTransportInteger)
		case 2:
			return jsonrpc.MaterializeHistory(token, &v.Cutoff, historyTransportInteger)
		case 3:
			v.keys = token
			cursor := historyProjectedCursor{raw: token, pos: 1}
			for {
				_, more, err := cursor.nextElement()
				if err != nil || !more {
					return err
				}
				v.count++
				if v.count > RoutingHistoryRecords {
					return historyError("invalid-value")
				}
			}
		}
		return historyError("invalid-shape")
	})
	if err != nil {
		return RoutingHistoryWindowView{}, err
	}
	cursor := historyProjectedCursor{raw: v.keys, pos: 1}
	err = validateHistoryWindow(schema, v.Generation, len(v.keys) > 0 && v.keys[0] == '[', v.count, v.Cutoff, func() (RoutingHistoryWindowKey, error) {
		span, more, err := cursor.nextElement()
		if err != nil || !more {
			return RoutingHistoryWindowKey{}, historyError("invalid-shape")
		}
		var key RoutingHistoryWindowKey
		err = jsonrpc.MaterializeHistory(v.keys[span.start:span.end], &key, historyTransportInteger)
		return key, err
	})
	if err != nil {
		return RoutingHistoryWindowView{}, err
	}
	return v, nil
}

// The following cursor only finds spans in already-checked projected JSON.
// Strings and numeric tokens stay in the leased bytes; no token-sized copy,
// unknown-field decoder, generic value tree or second offset bank is made.
type historyProjectedCursor struct {
	raw []byte
	pos int
}

func (c *historyProjectedCursor) space() {
	for c.pos < len(c.raw) && (c.raw[c.pos] == ' ' || c.raw[c.pos] == '\r' || c.raw[c.pos] == '\n' || c.raw[c.pos] == '\t') {
		c.pos++
	}
}

func (c *historyProjectedCursor) value() (historySpan, error) {
	c.space()
	start, depth, quoted := c.pos, 0, false
	for c.pos < len(c.raw) {
		b := c.raw[c.pos]
		if quoted {
			c.pos++
			if b == '\\' {
				c.pos++
			} else if b == '"' {
				quoted = false
				if depth == 0 {
					return historySpan{start, c.pos}, nil
				}
			}
			continue
		}
		switch b {
		case '"':
			quoted = true
		case '{', '[':
			depth++
		case '}', ']':
			if depth == 0 {
				if c.pos == start {
					return historySpan{}, historyError("invalid-shape")
				}
				return historySpan{start, c.pos}, nil
			}
			depth--
			if depth == 0 {
				c.pos++
				return historySpan{start, c.pos}, nil
			}
		case ',', ':', ' ', '\r', '\n', '\t':
			if depth == 0 {
				if c.pos == start {
					return historySpan{}, historyError("invalid-shape")
				}
				return historySpan{start, c.pos}, nil
			}
		}
		c.pos++
	}
	if c.pos > start && c.pos == len(c.raw) && depth == 0 && !quoted {
		return historySpan{start, c.pos}, nil
	}
	return historySpan{}, historyError("invalid-shape")
}

func (c *historyProjectedCursor) nextElement() (historySpan, bool, error) {
	if len(c.raw) < 2 || c.raw[0] != '[' {
		return historySpan{}, false, historyError("invalid-shape")
	}
	c.space()
	if c.pos < len(c.raw) && c.raw[c.pos] == ',' {
		c.pos++
		c.space()
	}
	if c.pos < len(c.raw) && c.raw[c.pos] == ']' {
		return historySpan{}, false, nil
	}
	span, err := c.value()
	return span, err == nil, err
}

func historyProjectedFields(raw []byte, typ reflect.Type, visit func(int, historySpan) error) error {
	c := historyProjectedCursor{raw: raw}
	c.space()
	if c.pos == len(raw) || raw[c.pos] != '{' {
		return historyError("invalid-shape")
	}
	c.pos++
	for {
		c.space()
		if c.pos < len(raw) && raw[c.pos] == '}' {
			return nil
		}
		key, err := c.value()
		if err != nil {
			return err
		}
		// Known names have a bounded escaped spelling. Check that bound
		// before asking the shared scalar materializer to decode a name.
		maxName := 0
		for i := 0; i < typ.NumField(); i++ {
			maxName = max(maxName, len(typ.Field(i).Tag.Get("json")))
		}
		if key.end-key.start > 6*maxName+2 {
			return historyError("invalid-shape")
		}
		var name string
		if err := jsonrpc.MaterializeHistory(raw[key.start:key.end], &name, historyTransportInteger); err != nil {
			return err
		}
		c.space()
		if c.pos == len(raw) || raw[c.pos] != ':' {
			return historyError("invalid-shape")
		}
		c.pos++
		span, err := c.value()
		if err != nil {
			return err
		}
		field := -1
		for i := 0; i < typ.NumField(); i++ {
			if typ.Field(i).Tag.Get("json") == name {
				field = i
				break
			}
		}
		if field < 0 {
			return historyError("invalid-shape")
		}
		if err := visit(field, span); err != nil {
			return err
		}
		c.space()
		if c.pos < len(raw) && raw[c.pos] == ',' {
			c.pos++
		}
	}
}
