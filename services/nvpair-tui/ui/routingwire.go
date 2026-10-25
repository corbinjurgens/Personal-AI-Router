// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package ui

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"reflect"
	"strings"
	"sync"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

const (
	routingHistoryCellBytes   = 65536
	routingHistoryHeaderBytes = 300
	// Capacity contract, not RSS or accepted shared semantic-scratch proof.
	routingHistoryDataBytes = 3*256*routingHistoryCellBytes + jsonrpc.HistoryFrameBytes + jsonrpc.HistoryInitialBytes + jsonrpc.HistoryRecordFrameBytes + 131072
)

// A packed cell owns UTF-8 bytes, scalar tags and presence bits, never DTO
// pointers or raw JSON. The header duplicates only bounded rank/key metadata
// inside S. The payload follows the shared Go type's fields reflectively, so
// there is no handwritten wire DTO or parallel semantic validator.
type routingHistoryCell struct{ data [routingHistoryCellBytes]byte }
type routingHistoryBank struct {
	cells    [nodepolicy.RoutingHistoryRecords]routingHistoryCell
	count    int
	floor    [routingHistoryHeaderBytes]byte
	floorSet bool
}

type routingHistoryWire struct {
	mu                         sync.Mutex
	generation                 uint64
	current, pending, baseline *routingHistoryBank
}

func newRoutingHistoryWire() *routingHistoryWire {
	return &routingHistoryWire{current: new(routingHistoryBank), pending: new(routingHistoryBank), baseline: new(routingHistoryBank)}
}

// reset and consume are the synchronous history-client factory callbacks. They
// share one generation/commit fence. No view or production subscription is
// mounted here; Task318 owns construction and Task5 owns presentation.
func (w *routingHistoryWire) reset(generation uint64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if generation < w.generation {
		return
	}
	w.generation = generation
	for _, bank := range []*routingHistoryBank{w.current, w.pending, w.baseline} {
		clear(bank.cells[:])
		bank.count = 0
		bank.floorSet = false
		clear(bank.floor[:])
	}
}

// consume receives only shared-guard-projected, semantically checked bytes.
// It stages a complete baseline, materializing one known record at a time,
// then merges with current pushes. No complete initial DTO or generic raw map
// is ever constructed. All borrowed bytes are finished before this returns.
func (w *routingHistoryWire) consume(generation uint64, initial bool, raw []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if generation != w.generation {
		return jsonrpc.ErrHistoryInvalid
	}
	copy(w.pending.cells[:], w.current.cells[:])
	w.pending.count = w.current.count
	w.pending.floor = w.current.floor
	w.pending.floorSet = w.current.floorSet
	if initial {
		w.baseline.count = 0
		if err := routingHistoryEach(raw, func(record []byte) error {
			if w.baseline.count == len(w.baseline.cells) {
				return jsonrpc.ErrHistoryInvalid
			}
			row, err := routingHistoryMaterialize(record)
			if err != nil {
				return err
			}
			if err := w.baseline.cells[w.baseline.count].store(row); err != nil {
				return err
			}
			w.baseline.count++
			return nil
		}); err != nil {
			return err
		}
		for i := 0; i < w.baseline.count; i++ {
			row, err := w.baseline.cells[i].record()
			if err != nil {
				return err
			}
			if err := w.merge(w.pending, row); err != nil {
				return err
			}
		}
	} else {
		row, err := routingHistoryMaterialize(raw)
		if err != nil {
			return err
		}
		if err := w.merge(w.pending, row); err != nil {
			return err
		}
	}
	w.current, w.pending = w.pending, w.current
	return nil
}

func routingHistoryMaterialize(raw []byte) (nodepolicy.RoutingHistoryRecord, error) {
	var row nodepolicy.RoutingHistoryRecord
	schema, ok := nodepolicy.RoutingHistoryTransportSchema(nodepolicy.NotifyRoutingChanged, false)
	if !ok {
		return row, jsonrpc.ErrHistoryInvalid
	}
	err := jsonrpc.MaterializeHistory(raw, &row, schema.Integer)
	return row, err
}

// This is iteration over already projected initial.requests, not a raw ingress
// classifier/validator. The guard has checked every original outer candidate,
// duplicate, unknown subtree and record. Key escapes need no reinterpretation:
// requests is the initial's only retained field, so its first '[' is the array.
func routingHistoryEach(raw []byte, visit func([]byte) error) error {
	start := bytes.IndexByte(raw, '[')
	if start < 0 {
		return jsonrpc.ErrHistoryInvalid
	}
	for i := start + 1; i < len(raw); {
		for i < len(raw) && (raw[i] == ',' || raw[i] == ' ' || raw[i] == '\n' || raw[i] == '\r' || raw[i] == '\t') {
			i++
		}
		if i < len(raw) && raw[i] == ']' {
			return nil
		}
		if i == len(raw) || raw[i] != '{' {
			return jsonrpc.ErrHistoryInvalid
		}
		begin, depth, quoted := i, 0, false
		for ; i < len(raw); i++ {
			c := raw[i]
			if quoted {
				if c == '\\' {
					i++
				} else if c == '"' {
					quoted = false
				}
				continue
			}
			if c == '"' {
				quoted = true
				continue
			}
			if c == '{' || c == '[' {
				depth++
			}
			if c == '}' || c == ']' {
				depth--
				if depth == 0 {
					i++
					break
				}
			}
		}
		if depth != 0 || i-begin > nodepolicy.RoutingHistoryRecordBytes {
			return jsonrpc.ErrHistoryInvalid
		}
		if err := visit(raw[begin:i]); err != nil {
			return err
		}
	}
	return jsonrpc.ErrHistoryInvalid
}

func historyHeader(row nodepolicy.RoutingHistoryRecord, header []byte) {
	clear(header)
	binary.LittleEndian.PutUint64(header, uint64(row.CreatedAt))
	binary.LittleEndian.PutUint64(header[8:], uint64(row.Seq))
	if row.State == "completed" || row.State == "failed" || row.State == "cancelled" {
		header[16] = 1
	}
	header[17] = byte(len(row.Request.OriginNodeID))
	header[18] = byte(len(row.Request.ProxyRunID))
	header[19] = byte(len(row.Request.RequestID))
	copy(header[20:148], row.Request.OriginNodeID)
	copy(header[148:276], row.Request.ProxyRunID)
	copy(header[276:296], row.Request.RequestID)
}

func historyHeaderKey(header []byte) ([]byte, []byte, []byte) {
	return header[20 : 20+int(header[17])], header[148 : 148+int(header[18])], header[276 : 276+int(header[19])]
}

func historySameKey(a, b []byte) bool {
	ao, ar, ai := historyHeaderKey(a)
	bo, br, bi := historyHeaderKey(b)
	return bytes.Equal(ao, bo) && bytes.Equal(ar, br) && bytes.Equal(ai, bi)
}

// Negative means a is newer/earlier in presentation order. Numeric IDs use
// canonical decimal length then lexical order, never floating-point parsing.
func historyRank(a, b []byte) int {
	ac, bc := binary.LittleEndian.Uint64(a), binary.LittleEndian.Uint64(b)
	if ac > bc {
		return -1
	}
	if ac < bc {
		return 1
	}
	ao, ar, ai := historyHeaderKey(a)
	bo, br, bi := historyHeaderKey(b)
	if n := bytes.Compare(ao, bo); n != 0 {
		return n
	}
	if n := bytes.Compare(ar, br); n != 0 {
		return n
	}
	if len(ai) > len(bi) {
		return -1
	}
	if len(ai) < len(bi) {
		return 1
	}
	return -bytes.Compare(ai, bi)
}

func (w *routingHistoryWire) merge(bank *routingHistoryBank, row nodepolicy.RoutingHistoryRecord) error {
	var header [routingHistoryHeaderBytes]byte
	historyHeader(row, header[:])
	for i := 0; i < bank.count; i++ {
		old := bank.cells[i].data[:routingHistoryHeaderBytes]
		if !historySameKey(old, header[:]) {
			continue
		}
		if binary.LittleEndian.Uint64(old) != uint64(row.CreatedAt) {
			return jsonrpc.ErrHistoryInvalid
		}
		if binary.LittleEndian.Uint64(old[8:]) >= uint64(row.Seq) || old[16] != 0 && header[16] == 0 {
			return nil
		}
		return bank.cells[i].store(row)
	}
	if bank.floorSet && historyRank(header[:], bank.floor[:]) >= 0 {
		return nil
	}
	position := 0
	for position < bank.count && historyRank(bank.cells[position].data[:], header[:]) < 0 {
		position++
	}
	if position == len(bank.cells) {
		return nil
	}
	next := min(bank.count+1, len(bank.cells))
	for i := next - 1; i > position; i-- {
		copy(bank.cells[i].data[:], bank.cells[i-1].data[:])
	}
	if err := bank.cells[position].store(row); err != nil {
		return err
	}
	bank.count = next
	if next == len(bank.cells) {
		last := bank.cells[next-1].data[:routingHistoryHeaderBytes]
		if !bank.floorSet || historyRank(last, bank.floor[:]) < 0 {
			copy(bank.floor[:], last)
			bank.floorSet = true
		}
	}
	return nil
}

func (cell *routingHistoryCell) store(row nodepolicy.RoutingHistoryRecord) error {
	historyHeader(row, cell.data[:routingHistoryHeaderBytes])
	p := historyPacked{bytes: cell.data[:], pos: routingHistoryHeaderBytes}
	if err := p.write(reflect.ValueOf(row)); err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(cell.data[296:], uint32(p.pos))
	return nil
}

func (cell *routingHistoryCell) record() (nodepolicy.RoutingHistoryRecord, error) {
	var row nodepolicy.RoutingHistoryRecord
	n := int(binary.LittleEndian.Uint32(cell.data[296:]))
	if n < routingHistoryHeaderBytes || n > len(cell.data) {
		return row, jsonrpc.ErrHistoryInvalid
	}
	p := historyPacked{bytes: cell.data[:n], pos: routingHistoryHeaderBytes}
	err := p.read(reflect.ValueOf(&row).Elem())
	if p.pos != n {
		return row, jsonrpc.ErrHistoryInvalid
	}
	return row, err
}

// One output loan: callers render synchronously and retain neither the record
// nor its referents. This is deliberately not a full-window snapshot API.
func (w *routingHistoryWire) withRecord(generation uint64, index int, consume func(nodepolicy.RoutingHistoryRecord)) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if generation != w.generation || index < 0 || index >= w.current.count {
		return jsonrpc.ErrHistoryInvalid
	}
	row, err := w.current.cells[index].record()
	if err != nil {
		return err
	}
	consume(row)
	return nil
}

type historyPacked struct {
	bytes []byte
	pos   int
}

func (p *historyPacked) take(n int) ([]byte, error) {
	if n < 0 || n > len(p.bytes)-p.pos {
		return nil, jsonrpc.ErrHistoryInvalid
	}
	value := p.bytes[p.pos : p.pos+n]
	p.pos += n
	return value, nil
}

func (p *historyPacked) write(v reflect.Value) error {
	switch v.Kind() {
	case reflect.Pointer:
		tag, err := p.take(1)
		if err != nil {
			return err
		}
		tag[0] = 0
		if v.IsNil() {
			return nil
		}
		tag[0] = 1
		return p.write(v.Elem())
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if err := p.write(v.Field(i)); err != nil {
				return err
			}
		}
	case reflect.Slice:
		if v.Len() > nodepolicy.RoutingHistoryAttempts {
			return jsonrpc.ErrHistoryInvalid
		}
		tag, err := p.take(2)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint16(tag, uint16(v.Len()))
		for i := 0; i < v.Len(); i++ {
			if err := p.write(v.Index(i)); err != nil {
				return err
			}
		}
	case reflect.String:
		value := v.String()
		if len(value) > 512 {
			return jsonrpc.ErrHistoryInvalid
		}
		tag, err := p.take(2)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint16(tag, uint16(len(value)))
		out, err := p.take(len(value))
		if err != nil {
			return err
		}
		copy(out, value)
	case reflect.Int, reflect.Int64:
		out, err := p.take(8)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint64(out, uint64(v.Int()))
	case reflect.Bool:
		out, err := p.take(1)
		if err != nil {
			return err
		}
		out[0] = 0
		if v.Bool() {
			out[0] = 1
		}
	default:
		return jsonrpc.ErrHistoryInvalid
	}
	return nil
}

func (p *historyPacked) read(v reflect.Value) error {
	switch v.Kind() {
	case reflect.Pointer:
		tag, err := p.take(1)
		if err != nil {
			return err
		}
		if tag[0] == 0 {
			v.SetZero()
			return nil
		}
		if tag[0] != 1 {
			return jsonrpc.ErrHistoryInvalid
		}
		v.Set(reflect.New(v.Type().Elem()))
		return p.read(v.Elem())
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if err := p.read(v.Field(i)); err != nil {
				return err
			}
		}
	case reflect.Slice:
		tag, err := p.take(2)
		if err != nil {
			return err
		}
		n := int(binary.LittleEndian.Uint16(tag))
		if n > nodepolicy.RoutingHistoryAttempts {
			return jsonrpc.ErrHistoryInvalid
		}
		v.Set(reflect.MakeSlice(v.Type(), n, n))
		for i := 0; i < n; i++ {
			if err := p.read(v.Index(i)); err != nil {
				return err
			}
		}
	case reflect.String:
		tag, err := p.take(2)
		if err != nil {
			return err
		}
		n := int(binary.LittleEndian.Uint16(tag))
		if n > 512 {
			return jsonrpc.ErrHistoryInvalid
		}
		value, err := p.take(n)
		if err != nil {
			return err
		}
		v.SetString(string(value))
	case reflect.Int, reflect.Int64:
		value, err := p.take(8)
		if err != nil {
			return err
		}
		v.SetInt(int64(binary.LittleEndian.Uint64(value)))
	case reflect.Bool:
		value, err := p.take(1)
		if err != nil {
			return err
		}
		if value[0] > 1 {
			return jsonrpc.ErrHistoryInvalid
		}
		v.SetBool(value[0] == 1)
	default:
		return jsonrpc.ErrHistoryInvalid
	}
	return nil
}

func routingHistoryMoney(value *nodepolicy.NanoUSD) string {
	if value == nil {
		return "—"
	}
	digits := string(*value)
	if _, err := nodepolicy.ParseNanoUSD(digits); err != nil {
		return "Unknown"
	}
	if len(digits) <= 9 {
		digits = strings.Repeat("0", 10-len(digits)) + digits
	}
	whole, fraction := digits[:len(digits)-9], strings.TrimRight(digits[len(digits)-9:], "0")
	if fraction == "" {
		return "$" + whole
	}
	return "$" + whole + "." + fraction
}

func routingHistoryModel(model *string) string {
	if model == nil {
		return "model not specified"
	}
	return *model
}

func routingHistoryWait(ms *int64) string {
	if ms == nil {
		return "—"
	}
	if *ms < 1000 {
		return fmt.Sprintf("%d ms", *ms)
	}
	return fmt.Sprintf("%d.%d s", *ms/1000, (*ms%1000)/100)
}

func routingHistoryReason(reason string) string {
	switch reason {
	case "":
		return ""
	case "client-gone":
		return "Client disconnected"
	case "cancel":
		return "Cancelled by user"
	case "regenerate":
		return "Regenerated"
	case "node-paused", "admission-paused":
		return "Node paused"
	case "proxy-shutdown":
		return "Proxy shutting down"
	case "proxy-exited":
		return "Proxy exited"
	case "target-left":
		return "Node left the cluster"
	case "transport":
		return "Connection error"
	case "upstream-status":
		return "Engine returned an error status"
	case "no-content":
		return "No output before timeout"
	case "stream-interrupted":
		return "Stream interrupted"
	case "upstream-error":
		return "Engine error"
	case "admission-draining":
		return "Node draining"
	case "admission-busy":
		return "Node busy"
	case "admission-no-fit":
		return "Model does not fit"
	case "admission-engine-off":
		return "Engine off"
	case "admission-wake-timeout":
		return "Engine wake timed out"
	case "engine-not-running":
		return "Engine not running"
	case "request-invalid":
		return "Request could not be adapted"
	case "no-owner":
		return "No node serves this model"
	case "regenerate-exhausted":
		return "Too many regenerations"
	case "attempts-exhausted":
		return "All attempts used"
	case "deadline":
		return "Request deadline reached"
	case "request-ceiling":
		return "Request spend ceiling reached"
	case "account-ceiling":
		return "Account ceiling reached"
	case "account-cooldown":
		return "Account cooling down"
	case "account-frozen":
		return "Account frozen"
	case "account-credential":
		return "Account credential failed"
	case "account-exhausted":
		return "Account exhausted"
	case "account-conflict":
		return "Account conflict"
	case "ledger-unavailable":
		return "Spend ledger unavailable"
	case "deployment-quarantined":
		return "Deployment quarantined"
	case "peer-framing":
		return "Peer sent a malformed frame"
	default:
		return "Unknown"
	}
}

func routingHistoryRequestReason(row nodepolicy.RoutingHistoryRecord) string {
	if row.Recoverable {
		return "Streaming paused, waiting to resume"
	}
	return routingHistoryReason(row.Reason)
}

func routingHistoryAttemptLabel(row nodepolicy.RoutingHistoryRecord, index int) string {
	if index < 0 || index >= len(row.Attempts) {
		return "Unknown"
	}
	a := row.Attempts[index]
	if a.Outcome == "interrupted" {
		for _, successor := range row.Attempts[index+1:] {
			if successor.ContinuationOf != nil && *successor.ContinuationOf == a.Index {
				return "Interrupted, resumed"
			}
		}
		if row.Recoverable {
			return "Streaming paused, waiting to resume"
		}
		return "Interrupted"
	}
	switch a.Outcome {
	case "pending":
		return "In progress"
	case "committed":
		return "Served"
	case "rejected":
		return "Rejected"
	case "failed":
		return "Failed"
	case "redispatched":
		return "Re-dispatched"
	case "cancelled":
		return "Cancelled"
	default:
		return "Unknown"
	}
}
