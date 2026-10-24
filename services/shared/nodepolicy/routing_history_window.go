// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
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
	if w.Schema != 1 || ValidateRoutingHistoryGeneration(w.Generation) != nil || w.Keys == nil || len(w.Keys) > RoutingHistoryRecords {
		return historyError("invalid-value")
	}
	least := -1
	for i, key := range w.Keys {
		if key.Request.Validate() != nil || !historyNumber(key.CreatedAt) || !historyNumber(key.Seq) || key.Seq == 0 || (i > 0 && compareHistoryKeys(w.Keys[i-1].Request, key.Request) >= 0) {
			return historyError("invalid-value")
		}
		if least < 0 || key.CreatedAt < w.Keys[least].CreatedAt || (key.CreatedAt == w.Keys[least].CreatedAt && compareHistoryKeys(key.Request, w.Keys[least].Request) < 0) {
			least = i
		}
	}
	if len(w.Keys) < RoutingHistoryRecords {
		if w.Cutoff != nil {
			return historyError("invalid-value")
		}
	} else if w.Cutoff == nil || w.Cutoff.Request != w.Keys[least].Request || w.Cutoff.CreatedAt != w.Keys[least].CreatedAt {
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
			var w RoutingHistoryWindow
			if err := jsonrpc.MaterializeHistory(raw, &w, historyTransportInteger); err != nil {
				return err
			}
			return w.Validate()
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
	var record RoutingHistoryRecord
	if err := jsonrpc.MaterializeHistory(raw, &record, historyTransportInteger); err != nil {
		return err
	}
	return record.Validate()
}

func historyTransportInteger(raw []byte) (int64, bool) { return historyInteger(string(raw)) }
