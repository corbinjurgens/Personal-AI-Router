// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

const (
	historyCellBytes = 65536
	historyWriters   = 8
)

const (
	historySubscribed = 1 << iota
	historyWriting
)

// All client history state is scalar. The registry owns clients; no history
// waiter list, per-client bank, or per-client history task is created.
type historyCursor struct {
	revision   uint64
	flags      uint64
	writeMu    sync.Mutex
	awaitReady atomic.Bool
}

// The key and rank are inline, alongside a compact known-field representation.
// Every cell, including its index, is exactly S bytes on supported platforms.
type historyCell struct {
	n        uint32
	lengths  [4]byte
	created  uint64
	seq      uint64
	revision uint64
	origin   [128]byte
	run      [128]byte
	id       [19]byte
	data     [historyCellBytes - 307]byte
}

func historyKeyCompare(a, b *historyCell) int {
	for _, pair := range [][2][]byte{
		{a.origin[:a.lengths[0]], b.origin[:b.lengths[0]]},
		{a.run[:a.lengths[1]], b.run[:b.lengths[1]]},
	} {
		if n := bytes.Compare(pair[0], pair[1]); n != 0 {
			return n
		}
	}
	if a.lengths[2] < b.lengths[2] {
		return -1
	}
	if a.lengths[2] > b.lengths[2] {
		return 1
	}
	return bytes.Compare(a.id[:a.lengths[2]], b.id[:b.lengths[2]])
}

// Newer ranks sort first; origin/run ties use ascending lexical order.
func historyRankCompare(a, b *historyCell) int {
	if a.created > b.created {
		return -1
	}
	if a.created < b.created {
		return 1
	}
	if n := bytes.Compare(a.origin[:a.lengths[0]], b.origin[:b.lengths[0]]); n != 0 {
		return n
	}
	if n := bytes.Compare(a.run[:a.lengths[1]], b.run[:b.lengths[1]]); n != 0 {
		return n
	}
	if a.lengths[2] > b.lengths[2] {
		return -1
	}
	if a.lengths[2] < b.lengths[2] {
		return 1
	}
	return -bytes.Compare(a.id[:a.lengths[2]], b.id[:b.lengths[2]])
}

type historyBank struct {
	cells [nodepolicy.RoutingHistoryRecords]historyCell
	n     int
}

// Pack only the shared DTO's fields, without JSON names, escaping, maps or
// retained DTO referents. All appends check the existing grant before writing.
func historyPack(dst []byte, value reflect.Value) ([]byte, error) {
	put := func(raw []byte) error {
		if len(raw) > cap(dst)-len(dst) {
			return jsonrpc.ErrHistoryInvalid
		}
		dst = append(dst, raw...)
		return nil
	}
	switch value.Kind() {
	case reflect.Pointer:
		if value.IsNil() {
			err := put([]byte{0})
			return dst, err
		}
		if err := put([]byte{1}); err != nil {
			return nil, err
		}
		return historyPack(dst, value.Elem())
	case reflect.Struct:
		for i := 0; i < value.NumField(); i++ {
			var err error
			dst, err = historyPack(dst, value.Field(i))
			if err != nil {
				return nil, err
			}
		}
	case reflect.Slice:
		if value.Len() > 32 {
			return nil, jsonrpc.ErrHistoryInvalid
		}
		if err := put([]byte{byte(value.Len())}); err != nil {
			return nil, err
		}
		for i := 0; i < value.Len(); i++ {
			var err error
			dst, err = historyPack(dst, value.Index(i))
			if err != nil {
				return nil, err
			}
		}
	case reflect.String:
		text := value.String()
		var length [2]byte
		if len(text) > 512 {
			return nil, jsonrpc.ErrHistoryInvalid
		}
		binary.LittleEndian.PutUint16(length[:], uint16(len(text)))
		if err := put(length[:]); err != nil {
			return nil, err
		}
		if err := put([]byte(text)); err != nil {
			return nil, err
		}
	case reflect.Int, reflect.Int64:
		var number [8]byte
		binary.LittleEndian.PutUint64(number[:], uint64(value.Int()))
		if err := put(number[:]); err != nil {
			return nil, err
		}
	case reflect.Bool:
		b := byte(0)
		if value.Bool() {
			b = 1
		}
		if err := put([]byte{b}); err != nil {
			return nil, err
		}
	default:
		return nil, jsonrpc.ErrHistoryInvalid
	}
	return dst, nil
}

func (c *historyCell) set(record *nodepolicy.RoutingHistoryRecord) error {
	packed, err := historyPack(c.data[:0], reflect.ValueOf(*record))
	if err != nil {
		return err
	}
	c.n = uint32(len(packed))
	c.created, c.seq = uint64(record.CreatedAt), uint64(record.Seq)
	c.lengths[0] = byte(copy(c.origin[:], record.Request.OriginNodeID))
	c.lengths[1] = byte(copy(c.run[:], record.Request.ProxyRunID))
	c.lengths[2] = byte(copy(c.id[:], record.Request.RequestID))
	c.lengths[3] = 0
	if record.EndedAt != nil {
		c.lengths[3] = 1
	}
	return nil
}

type historyEncoder struct {
	out []byte
	err error
}

func (e *historyEncoder) put(raw []byte) {
	if e.err != nil {
		return
	}
	if len(raw) > cap(e.out)-len(e.out) {
		e.err = jsonrpc.ErrHistoryInvalid
		return
	}
	e.out = append(e.out, raw...)
}

func (e *historyEncoder) text(text string) { e.put([]byte(text)) }

func (e *historyEncoder) quoted(text string) {
	const hex = "0123456789abcdef"
	e.text(`"`)
	for i := 0; i < len(text); i++ {
		b := text[i]
		switch {
		case b == '"' || b == '\\':
			e.put([]byte{'\\', b})
		case b < 32:
			e.put([]byte{'\\', 'u', '0', '0', hex[b>>4], hex[b&15]})
		default:
			e.put([]byte{b})
		}
	}
	e.text(`"`)
}

// Encode directly from the packed cell into the admitted output lease. No
// marshal/RawMessage/outer-marshal overlap or pooled encoder is involved.
func (e *historyEncoder) value(raw []byte, typ reflect.Type) []byte {
	if e.err != nil {
		return nil
	}
	take := func(n int) []byte {
		if n > len(raw) {
			e.err = jsonrpc.ErrHistoryInvalid
			return nil
		}
		v := raw[:n]
		raw = raw[n:]
		return v
	}
	switch typ.Kind() {
	case reflect.Pointer:
		flag := take(1)
		if e.err != nil {
			return nil
		}
		if flag[0] == 0 {
			e.text("null")
			return raw
		}
		return e.value(raw, typ.Elem())
	case reflect.Struct:
		e.text("{")
		for i := 0; i < typ.NumField(); i++ {
			if i > 0 {
				e.text(",")
			}
			e.quoted(typ.Field(i).Tag.Get("json"))
			e.text(":")
			raw = e.value(raw, typ.Field(i).Type)
		}
		e.text("}")
	case reflect.Slice:
		count := take(1)
		if e.err != nil {
			return nil
		}
		e.text("[")
		for i := 0; i < int(count[0]); i++ {
			if i > 0 {
				e.text(",")
			}
			raw = e.value(raw, typ.Elem())
		}
		e.text("]")
	case reflect.String:
		length := take(2)
		if e.err != nil {
			return nil
		}
		v := take(int(binary.LittleEndian.Uint16(length)))
		if e.err != nil {
			return nil
		}
		e.quoted(string(v))
	case reflect.Int, reflect.Int64:
		number := take(8)
		if e.err != nil {
			return nil
		}
		var scratch [20]byte
		e.put(strconv.AppendInt(scratch[:0], int64(binary.LittleEndian.Uint64(number)), 10))
	case reflect.Bool:
		flag := take(1)
		if e.err != nil {
			return nil
		}
		if flag[0] == 0 {
			e.text("false")
		} else {
			e.text("true")
		}
	default:
		e.err = jsonrpc.ErrHistoryInvalid
	}
	return raw
}

func (e *historyEncoder) record(cell *historyCell) {
	start := len(e.out)
	if rest := e.value(cell.data[:cell.n], reflect.TypeOf(nodepolicy.RoutingHistoryRecord{})); len(rest) != 0 {
		e.err = jsonrpc.ErrHistoryInvalid
	}
	if len(e.out)-start > jsonrpc.HistoryRecordBytes {
		e.err = jsonrpc.ErrHistoryInvalid
	}
}

type historyWrite struct {
	client     *client
	frame      []byte
	storage    []byte
	generation uint64
	revision   uint64
	initial    bool
	active     bool
	queued     bool
}

// Service.mu serializes correlation, cache, registry cursors, guard offsets and
// lease transitions. Transport reads happen before acquiring it; active writes
// happen after releasing it. A stuck peer owns only its admitted write slot.
type serviceHistory struct {
	banks             [3]*historyBank
	current           int
	work              historyCell
	clock             uint64
	generation        uint64
	lastClient        int
	wake              chan struct{}
	jobs              chan int
	writes            [historyWriters]historyWrite
	raw               []byte
	initial           []byte
	offsets           []uint32
	correlation       jsonrpc.HistoryCorrelation
	clientCorrelation jsonrpc.HistoryCorrelation
	initialSlot       int
	pending           bool
	original          [jsonrpc.HistoryIDBytes]byte
	originalN         int
	pendingRun        *brokerRun
	control           [jsonrpc.HistoryEnvelopeBytes]byte
	controlN          int
	controlActive     bool
	controlBroken     bool
	loaded            bool
	floor             historyRank
	hasFloor          bool
	rawMu             sync.Mutex
	// Oversized client input and initial output share this H arena. Client
	// readers wait outside Service.mu; initial admission only tries the gate.
	initialMu      sync.Mutex
	refetch        bool
	exhausted      bool
	retryCancelled bool
	retryStep      uint8
	retryTimer     *time.Timer
	retryArmed     bool
	wg             sync.WaitGroup
	start          sync.Once
}

type historyRank struct {
	created uint64
	origin  [128]byte
	run     [128]byte
	id      [19]byte
	lengths [3]byte
}

func (r *historyRank) compare(cell *historyCell) int {
	if cell.created > r.created {
		return -1
	}
	if cell.created < r.created {
		return 1
	}
	if n := bytes.Compare(cell.origin[:cell.lengths[0]], r.origin[:r.lengths[0]]); n != 0 {
		return n
	}
	if n := bytes.Compare(cell.run[:cell.lengths[1]], r.run[:r.lengths[1]]); n != 0 {
		return n
	}
	if cell.lengths[2] > r.lengths[2] {
		return -1
	}
	if cell.lengths[2] < r.lengths[2] {
		return 1
	}
	return -bytes.Compare(cell.id[:cell.lengths[2]], r.id[:r.lengths[2]])
}

func (r *historyRank) set(cell *historyCell) {
	r.created, r.origin, r.run, r.id = cell.created, cell.origin, cell.run, cell.id
	copy(r.lengths[:], cell.lengths[:3])
}

// Construction is opt-in until the atomic production join. Ordinary service
// constructors and transports keep their existing behavior.
func newServiceWithHistory(cfg config) *Service {
	s := newService(cfg)
	h := &serviceHistory{
		wake: make(chan struct{}, 1), jobs: make(chan int, historyWriters),
		initialSlot: -1, generation: 1,
	}
	// Only the broker response direction mints IDs. Client ingress keeps its
	// zero-value arbitrary-ID admission/refusal contract.
	_ = h.correlation.EnableGeneratedIDs()
	s.history = h
	return s
}

func (h *serviceHistory) initialGrant() {
	if h.initial == nil {
		h.initial = make([]byte, 0, jsonrpc.HistoryFrameBytes)
	}
	if h.offsets == nil {
		h.offsets = make([]uint32, jsonrpc.HistoryInitialBytes/4)
	}
}

func (h *serviceHistory) rawGrant() {
	if h.raw == nil {
		h.raw = make([]byte, 0, jsonrpc.HistoryFrameBytes)
	}
}

func (h *serviceHistory) bank(index int) *historyBank {
	if h.banks[index] == nil {
		h.banks[index] = new(historyBank)
	}
	return h.banks[index]
}

func (h *serviceHistory) outputGrant(slot int) {
	if h.writes[slot].storage == nil {
		h.writes[slot].storage = make([]byte, 0, jsonrpc.HistoryRecordFrameBytes)
	}
}

func (h *serviceHistory) signal() {
	select {
	case h.wake <- struct{}{}:
	default:
	}
}

func (s *Service) startHistory() {
	if s.history == nil {
		return
	}
	h := s.history
	h.start.Do(func() {
		h.wg.Add(historyWriters + 1)
		for i := 0; i < historyWriters; i++ {
			go func() { defer h.wg.Done(); s.historyWriter() }()
		}
		go func() { defer h.wg.Done(); s.historySchedule() }()
	})
}

func historyPublic(method string) bool {
	return method == nodepolicy.MethodRoutingSubscribe || method == nodepolicy.MethodRoutingUnsubscribe || method == nodepolicy.MethodRoutingGetInitial
}

func historyPrivate(method string) bool {
	return method == nodepolicy.NotifyRoutingRequest || method == nodepolicy.NotifyRoutingHistoryWindow || method == nodepolicy.NotifyRoutingHistoryWindowEnd || method == nodepolicy.NotifyRoutingHistoryAccepted || method == nodepolicy.NotifyRoutingChanged
}

func (h *serviceHistory) freeSlot() int {
	for i := range h.writes {
		if !h.writes[i].active {
			return i
		}
	}
	return -1
}

func (h *serviceHistory) merge(cell *historyCell) bool {
	bank := h.bank(h.current)
	index := -1
	for i := 0; i < bank.n; i++ {
		old := &bank.cells[i]
		if historyKeyCompare(old, cell) == 0 {
			if old.created != cell.created || old.seq >= cell.seq || (old.lengths[3] != 0 && cell.lengths[3] == 0) {
				return false
			}
			index = i
			break
		}
	}
	if index < 0 {
		if h.hasFloor && h.floor.compare(cell) >= 0 {
			return false
		}
		if bank.n == len(bank.cells) {
			index = 0
			for i := 1; i < bank.n; i++ {
				if historyRankCompare(&bank.cells[i], &bank.cells[index]) > 0 {
					index = i
				}
			}
			if historyRankCompare(cell, &bank.cells[index]) >= 0 {
				return false
			}
		} else {
			index = bank.n
			bank.n++
		}
	}
	h.clock++
	cell.revision = h.clock
	bank.cells[index] = *cell
	if bank.n == len(bank.cells) {
		last := &bank.cells[0]
		for i := 1; i < bank.n; i++ {
			if historyRankCompare(&bank.cells[i], last) > 0 {
				last = &bank.cells[i]
			}
		}
		if !h.hasFloor || h.floor.compare(last) < 0 {
			h.floor.set(last)
			h.hasFloor = true
		}
	}
	return true
}

var serviceHistoryRecordSchema, _ = nodepolicy.RoutingHistoryTransportSchema(nodepolicy.NotifyRoutingChanged, false)

func historyInteger(raw []byte) (int64, bool) {
	return serviceHistoryRecordSchema.Integer(raw)
}

// Guard one complete physical frame while holding Service.mu. Reconstructing
// the transient codec over the already funded raw backing keeps correlation
// changes serialized without holding a mutex across a blocking stream read.
func (h *serviceHistory) guard(frame []byte, correlation *jsonrpc.HistoryCorrelation) (*jsonrpc.Message, error) {
	if h.offsets == nil {
		h.offsets = make([]uint32, jsonrpc.HistoryInitialBytes/4)
	}
	codec, err := jsonrpc.NewHistoryCodec(bytes.NewBuffer(frame), jsonrpc.WorkerFrameBytes, frame[:0:cap(frame)], frame[:0:cap(frame)], h.offsets, correlation, nodepolicy.RoutingHistoryTransportSchema, nil)
	if err != nil {
		return nil, err
	}
	return codec.Read()
}

// Client frames retain their existing ordinary backing. Oversized client input
// can borrow the one global initial arena to reach the canonical raw guard;
// no client owns an additional H bank. Other callers receive busy while it is
// borrowed, just as while an initial output is pinned.
func readHistoryPhysical(reader *bufio.Reader, dst []byte, retain int) ([]byte, error) {
	return readHistoryPhysicalWithSpill(reader, dst, retain, nil)
}

func readHistoryPhysicalWithSpill(reader *bufio.Reader, dst []byte, retain int, acquire func([]byte) []byte) ([]byte, error) {
	dst = dst[:0]
	total, discarded := 0, false
	for {
		part, err := reader.ReadSlice('\n')
		total += len(part)
		if total > jsonrpc.HistoryFrameBytes {
			return nil, jsonrpc.ErrHistoryFraming
		}
		if total == jsonrpc.HistoryFrameBytes && (len(part) == 0 || part[len(part)-1] != '\n') {
			return nil, jsonrpc.ErrHistoryFraming
		}
		if total > retain {
			if acquire != nil {
				dst = acquire(dst)
				retain = jsonrpc.HistoryFrameBytes
			} else {
				discarded = true
			}
		}
		if !discarded {
			if len(part) > cap(dst)-len(dst) {
				capacity := min(retain, max(2*cap(dst), len(dst)+len(part)))
				next := make([]byte, len(dst), capacity)
				copy(next, dst)
				dst = next
			}
			dst = append(dst, part...)
		}
		if err == nil {
			if discarded {
				return dst[:0], jsonrpc.ErrHistoryInvalid
			}
			return dst, nil
		}
		if err == bufio.ErrBufferFull {
			continue
		}
		if err == io.EOF && total == 0 {
			return nil, io.EOF
		}
		return nil, jsonrpc.ErrHistoryFraming
	}
}

func historyFailure(err error) bool {
	var failure *jsonrpc.HistoryFailure
	var decode *jsonrpc.DecodeError
	return errors.As(err, &failure) || errors.As(err, &decode) || errors.Is(err, jsonrpc.ErrHistoryInvalid)
}

// Caller holds Service.mu. Cancellation stops future control work, without
// releasing a pending response or an actual control/output write borrower.
func (s *Service) cancelHistoryRefetchIfUnused() {
	for c := range s.clients {
		if c.history.flags&historySubscribed != 0 {
			return
		}
	}
	s.history.retryCancelled = true
	s.history.cancelRefetch()
}

// Public history methods never enter generic remapping/subscription handling.
// The sole initial ID and its raw spelling live in this global transfer owner.
func (s *Service) handleHistoryRequest(c *client, msg *jsonrpc.Message) {
	s.mu.Lock()
	h := s.history
	if msg.Method == nodepolicy.MethodRoutingUnsubscribe {
		c.history.flags &^= historySubscribed
		s.cancelHistoryRefetchIfUnused()
	} else {
		h.retryCancelled = false
	}
	if msg.Method == nodepolicy.MethodRoutingSubscribe {
		c.history.flags |= historySubscribed
	}
	slot := h.freeSlot()
	if h.initialSlot >= 0 || h.controlActive || slot < 0 || c.history.flags&historyWriting != 0 || !h.initialMu.TryLock() {
		s.mu.Unlock()
		historyControlError(c, *msg.ID, jsonrpc.HistoryBusyCode, jsonrpc.HistoryBusyMessage)
		return
	}
	h.initialGrant()
	h.outputGrant(slot)
	w := &h.writes[slot]
	*w = historyWrite{client: c, storage: w.storage, generation: h.generation, initial: true, active: true}
	c.history.flags |= historyWriting
	h.initialSlot = slot
	h.originalN = copy(h.original[:], *msg.ID)
	if (h.loaded && !h.correlation.Quarantined()) || msg.Method == nodepolicy.MethodRoutingUnsubscribe {
		h.encodeInitial(msg.Method == nodepolicy.MethodRoutingUnsubscribe)
		s.mu.Unlock()
		return
	}
	if h.exhausted || h.controlBroken {
		h.release(slot)
		s.mu.Unlock()
		historyControlError(c, *msg.ID, jsonrpc.HistoryInvalidCode, "history-invalid")
		return
	}
	run := s.broker
	if run == nil || s.stopping {
		h.release(slot)
		s.mu.Unlock()
		historyControlError(c, *msg.ID, codeUnavailable, "broker is not running")
		return
	}
	if err := h.prepareControl(run); err != nil {
		h.release(slot)
		s.mu.Unlock()
		if errors.Is(err, jsonrpc.ErrHistoryBusy) {
			historyControlError(c, *msg.ID, jsonrpc.HistoryBusyCode, jsonrpc.HistoryBusyMessage)
		} else {
			historyControlError(c, *msg.ID, jsonrpc.HistoryInvalidCode, "history-invalid")
		}
		return
	}
	generation := h.generation
	s.mu.Unlock()
	s.writeHistoryControl(run, generation)
}

// Prepare while serialized, before a reader can retire the minted token. The
// single C-funded control frame remains pinned independently of logical pending
// state, even while waiting for the ordinary broker frame gate.
func (h *serviceHistory) prepareControl(run *brokerRun) error {
	if h.controlActive || h.controlBroken {
		return jsonrpc.ErrHistoryBusy
	}
	id, err := h.correlation.BeginGeneratedMethod(nodepolicy.MethodRoutingSubscribe)
	if err != nil {
		if errors.Is(err, jsonrpc.ErrHistoryExhausted) {
			h.exhausted = true
			h.cancelRefetch()
		}
		return err
	}
	e := historyEncoder{out: h.control[:0]}
	e.text(`{"jsonrpc":"2.0","id":`)
	e.put(id)
	e.text(`,"method":"routing:subscribe","params":{}}` + "\n")
	h.controlN = len(e.out)
	h.controlActive, h.pending, h.pendingRun = true, true, run
	h.refetch = false
	h.stopRetryTimer()
	return nil
}

func (s *Service) writeHistoryControl(run *brokerRun, generation uint64) {
	h := s.history
	written, err := run.writeFrameProgress(h.control[:h.controlN])
	s.mu.Lock()
	// This actual borrower is the only writer allowed to release control E.
	// A source fence may already have installed a new correlation generation.
	h.controlActive = false
	current := h.generation == generation && s.broker == run
	if current && err != nil && written != 0 {
		h.controlBroken = true
		h.cancelRefetch()
	}
	if current && err != nil && h.pending && h.pendingRun == run {
		h.pending = false
		h.correlation.Retire()
		if h.initialSlot >= 0 {
			w := &h.writes[h.initialSlot]
			if w.client != nil {
				historyControlError(w.client, h.original[:h.originalN], jsonrpc.HistoryInvalidCode, "history-invalid")
			} else if written == 0 {
				h.queueRefetch()
			}
			h.release(h.initialSlot)
		}
	}
	h.signal()
	s.mu.Unlock()
	if err != nil && written != 0 {
		// A partial, complete-with-error or unknown-progress failed Write may
		// have delivered the control frame. Its framing cannot be reused.
		// This is not a history deadline or capacity-pressure restart.
		_ = run.proc.kill()
	}
}

func (h *serviceHistory) encodeInitial(empty bool) {
	w := &h.writes[h.initialSlot]
	e := historyEncoder{out: h.initial[:0]}
	e.text(`{"jsonrpc":"2.0","id":`)
	e.put(h.original[:h.originalN])
	e.text(`,"result":{"requests":[`)
	valueStart := len(e.out) - len(`{"requests":[`)
	if !empty {
		bank := h.banks[h.current]
		var order [nodepolicy.RoutingHistoryRecords]uint16
		for i := 0; i < bank.n; i++ {
			order[i] = uint16(i)
			for j := i; j > 0 && historyRankCompare(&bank.cells[order[j]], &bank.cells[order[j-1]]) < 0; j-- {
				order[j], order[j-1] = order[j-1], order[j]
			}
		}
		for i := 0; i < bank.n; i++ {
			if i > 0 {
				e.text(",")
			}
			e.record(&bank.cells[order[i]])
		}
	}
	e.text("]}")
	if len(e.out)-valueStart > jsonrpc.HistoryInitialBytes {
		e.err = jsonrpc.ErrHistoryInvalid
	}
	e.text("}\n")
	if e.err != nil {
		e = historyEncoder{out: h.initial[:0]}
		e.text(`{"jsonrpc":"2.0","id":`)
		e.put(h.original[:h.originalN])
		e.text(`,"error":{"code":-32062,"message":"history-invalid"}}` + "\n")
	}
	w.frame, w.revision = e.out, h.clock
	if !w.client.history.awaitReady.Load() {
		w.queued = true
		h.jobs <- h.initialSlot
	} else {
		h.signal()
	}
}

func (h *serviceHistory) release(slot int) {
	w := &h.writes[slot]
	if w.client != nil {
		w.client.history.flags &^= historyWriting
	}
	if w.initial {
		h.initialSlot, h.originalN = -1, 0
		h.initialMu.Unlock()
	}
	*w = historyWrite{storage: w.storage}
	h.signal()
}

func historyRetryDelay(step uint8) time.Duration {
	if step == 0 {
		return 250 * time.Millisecond
	}
	if step == 1 {
		return 500 * time.Millisecond
	}
	return time.Second
}

func (h *serviceHistory) stopRetryTimer() {
	if h.retryTimer != nil && h.retryArmed {
		if !h.retryTimer.Stop() {
			select {
			case <-h.retryTimer.C:
			default:
			}
		}
	}
	h.retryArmed = false
}

func (h *serviceHistory) cancelRefetch() {
	h.refetch, h.retryStep = false, 0
	h.stopRetryTimer()
	h.signal()
}

func (h *serviceHistory) queueRefetch() {
	if !h.exhausted && !h.controlBroken && !h.retryCancelled {
		h.refetch = true
		h.signal()
	}
}

func (s *Service) historySchedule() {
	h := s.history
	defer func() {
		s.mu.Lock()
		h.stopRetryTimer()
		s.mu.Unlock()
	}()
	for {
		s.mu.Lock()
		var retry <-chan time.Time
		if h.retryArmed {
			retry = h.retryTimer.C
		}
		s.mu.Unlock()
		select {
		case <-s.stopCh:
			return
		case <-h.wake:
		case <-retry:
			s.mu.Lock()
			h.retryArmed = false
			if h.refetch && h.retryStep < 2 {
				h.retryStep++
			}
			s.mu.Unlock()
			s.historyRefetch()
		}
		s.mu.Lock()
		awaitReady := false
		if h.initialSlot >= 0 {
			w := &h.writes[h.initialSlot]
			if w.client != nil && !w.queued && w.frame != nil {
				if w.client.isClosing() {
					h.release(h.initialSlot)
				} else if !w.client.history.awaitReady.Load() {
					w.queued = true
					h.jobs <- h.initialSlot
				} else {
					awaitReady = true
				}
			}
		}
		for {
			slot := h.freeSlot()
			if slot < 0 {
				break
			}
			var next, wrap *client
			for c := range s.clients {
				if c.history.awaitReady.Load() && c.history.flags&historySubscribed != 0 {
					awaitReady = true
				}
				if c.history.flags&historySubscribed == 0 || c.history.flags&historyWriting != 0 || c.history.revision >= h.clock || c.history.awaitReady.Load() || c.isClosing() {
					continue
				}
				if wrap == nil || c.id < wrap.id {
					wrap = c
				}
				if c.id > h.lastClient && (next == nil || c.id < next.id) {
					next = c
				}
			}
			if next == nil {
				next = wrap
			}
			if next == nil {
				break
			}
			h.lastClient = next.id
			var cell *historyCell
			bank := h.banks[h.current]
			for i := 0; i < bank.n; i++ {
				candidate := &bank.cells[i]
				if candidate.revision > next.history.revision && (cell == nil || candidate.revision < cell.revision) {
					cell = candidate
				}
			}
			if cell == nil {
				next.history.revision = h.clock
				continue
			}
			w := &h.writes[slot]
			h.outputGrant(slot)
			e := historyEncoder{out: w.storage[:0]}
			e.text(`{"jsonrpc":"2.0","method":"routing:changed","params":`)
			e.record(cell)
			e.text("}\n")
			if e.err != nil {
				next.history.revision = cell.revision
				continue
			}
			*w = historyWrite{client: next, frame: e.out, storage: w.storage, generation: h.generation, revision: cell.revision, active: true, queued: true}
			next.history.flags |= historyWriting
			h.jobs <- slot
		}
		// The same single timer also revisits the existing app:ready write
		// barrier. It never creates another pending call or writer.
		if (h.refetch && !h.exhausted && !h.controlBroken && !h.retryCancelled) || awaitReady {
			if !h.retryArmed {
				delay := time.Second
				if h.refetch {
					delay = historyRetryDelay(h.retryStep)
				}
				if h.retryTimer == nil {
					h.retryTimer = time.NewTimer(delay)
				} else {
					h.retryTimer.Reset(delay)
				}
				h.retryArmed = true
			}
		} else {
			h.stopRetryTimer()
		}
		s.mu.Unlock()
	}
}

func (s *Service) historyWriter() {
	h := s.history
	for {
		var slot int
		select {
		case <-s.stopCh:
			return
		case slot = <-h.jobs:
		}
		s.mu.Lock()
		w := h.writes[slot]
		s.mu.Unlock()
		err := s.writeHistoryFrame(w)
		s.mu.Lock()
		if w.generation == h.generation && err == nil {
			w.client.history.revision = max(w.client.history.revision, w.revision)
		}
		w.client.history.flags &^= historyWriting
		h.release(slot)
		s.mu.Unlock()
	}
}

// Take the stream gate before checking the generation. An unsent old frame is
// discarded; a started frame keeps the gate until actual completion, so the
// ordinary broker-reset notification cannot overtake it on this connection.
func (s *Service) writeHistoryFrame(w historyWrite) error {
	c := w.client
	c.history.writeMu.Lock()
	s.mu.Lock()
	current := w.generation == s.history.generation
	s.mu.Unlock()
	var err error
	if current {
		if c.isClosing() {
			err = io.ErrClosedPipe
		} else {
			err = c.writeFrameLocked(w.frame)
		}
	}
	c.history.writeMu.Unlock()
	if err != nil {
		c.closeNow()
	}
	return err
}

func (s *Service) historyBrokerMessage(run *brokerRun, msg *jsonrpc.Message) bool {
	h := s.history
	if msg.IsNotification() && msg.Method == methodAppReady {
		for c := range s.clients {
			if c.history.flags&historySubscribed != 0 && !h.exhausted {
				h.retryCancelled = false
				h.queueRefetch()
				break
			}
		}
	}
	if msg.IsNotification() && historyPrivate(msg.Method) {
		if h.correlation.Quarantined() {
			return true
		}
		if msg.Method == nodepolicy.NotifyRoutingChanged {
			var record nodepolicy.RoutingHistoryRecord
			if jsonrpc.MaterializeHistory(msg.Params, &record, historyInteger) == nil && record.Validate() == nil && h.work.set(&record) == nil {
				if h.merge(&h.work) {
					h.signal()
				}
			}
		}
		return true
	}
	if !msg.IsResponse() || !h.pending || h.pendingRun != run || !h.correlation.MatchesPending(*msg.ID) {
		return false
	}
	h.pending = false
	if msg.Error != nil {
		// A valid busy/error envelope finishes this logical call, but cannot
		// restore a quarantined cache. Its issued ordinal stays stale forever.
		h.correlation.Retire()
		w := &h.writes[h.initialSlot]
		c := w.client
		if c != nil {
			historyControlError(c, h.original[:h.originalN], msg.Error.Code, msg.Error.Message)
		}
		h.release(h.initialSlot)
		h.queueRefetch()
		return true
	}
	// An initial is streamed record by record into the baseline bank, never
	// materialized as a 256-record DTO. The raw guard already checked grammar,
	// unknown stripping and each record's semantic contract.
	baselineIndex := (h.current + 1) % len(h.banks)
	baseline := h.bank(baselineIndex)
	baseline.n = 0
	decoder := historyInitialRecords(msg.Result)
	for len(decoder) != 0 {
		raw, rest := historyNextRecord(decoder)
		if len(raw) == 0 {
			break
		}
		var record nodepolicy.RoutingHistoryRecord
		if baseline.n >= len(baseline.cells) || jsonrpc.MaterializeHistory(raw, &record, historyInteger) != nil || baseline.cells[baseline.n].set(&record) != nil {
			h.correlation.Retire()
			if c := h.writes[h.initialSlot].client; c != nil {
				historyControlError(c, h.original[:h.originalN], jsonrpc.HistoryInvalidCode, "history-invalid")
			}
			h.release(h.initialSlot)
			h.queueRefetch()
			return true
		}
		baseline.n++
		decoder = rest
	}
	// Merge a delayed baseline with accepted pushes in the third bank. Each
	// bank has one owner; output writers borrow encoded frames, never cells.
	target := (h.current + 2) % len(h.banks)
	h.bank(target)
	if h.banks[h.current] != nil {
		*h.banks[target] = *h.banks[h.current]
	} else {
		h.banks[target].n = 0
	}
	h.current = target
	for i := 0; i < baseline.n; i++ {
		h.merge(&baseline.cells[i])
	}
	h.loaded = true
	// Only a successfully APPLIED current initial restores history. Output H
	// and any unfinished control E loan retain their independent real owners.
	h.correlation.Complete()
	h.cancelRefetch()
	if h.writes[h.initialSlot].client == nil {
		h.release(h.initialSlot)
	} else {
		h.encodeInitial(false)
	}
	h.signal()
	return true
}

func historyControlError(c *client, id []byte, code int, message string) {
	var scratch [jsonrpc.HistoryEnvelopeBytes]byte
	frame := historyControlFrame(scratch[:0], id, code, message)
	if frame != nil {
		c.send(append([]byte(nil), frame...))
	}
}

func historyControlFrame(storage, id []byte, code int, message string) []byte {
	e := historyEncoder{out: storage[:0]}
	e.text(`{"jsonrpc":"2.0","id":`)
	e.put(id)
	e.text(`,"error":{"code":`)
	var number [20]byte
	e.put(strconv.AppendInt(number[:0], int64(code), 10))
	e.text(`,"message":`)
	e.quoted(message)
	e.text("}}\n")
	if e.err != nil {
		return nil
	}
	return e.out
}

func (s *Service) readHistoryClient(c *client) {
	reader := bufio.NewReaderSize(c.conn, 4096)
	raw := make([]byte, 0, jsonrpc.WorkerFrameBytes)
	for {
		spilled := false
		frame, err := readHistoryPhysicalWithSpill(reader, raw, jsonrpc.WorkerFrameBytes, func(prefix []byte) []byte {
			h := s.history
			h.initialMu.Lock()
			spilled = true
			s.mu.Lock()
			h.initialGrant()
			frame := append(h.initial[:0], prefix...)
			s.mu.Unlock()
			return frame
		})
		alive := true
		if err == nil {
			alive = s.handleHistoryClientFrame(c, frame)
		}
		if spilled {
			s.history.initialMu.Unlock()
			s.history.signal()
		} else {
			raw = frame
		}
		if err != nil {
			break
		}
		if !alive {
			return
		}
	}
	s.detach(c, false)
}

func (s *Service) handleHistoryClientFrame(c *client, frame []byte) bool {
	s.mu.Lock()
	h := s.history
	h.clientCorrelation = jsonrpc.HistoryCorrelation{}
	msg, err := h.guard(frame, &h.clientCorrelation)
	if err != nil {
		var failure *jsonrpc.HistoryFailure
		if errors.As(err, &failure) && failure.Refusal != nil {
			// Consume the guard's bounded borrowed ID before another Read.
			// Ordinary reply queue ownership starts before this lock is released.
			r := failure.Refusal
			historyControlError(c, r.ReplyID, r.Code, r.Message)
		}
	}
	if err == nil && historyPublic(msg.Method) && msg.ID != nil {
		// End the codec's interior ID borrow before releasing the guard owner.
		// This client's existing ordinary input backing owns the scalar until
		// dispatch returns; the one admitted initial then copies it into C.
		id := frame[cap(frame)-jsonrpc.HistoryIDBytes : cap(frame)]
		n := copy(id, *msg.ID)
		token := json.RawMessage(id[:n])
		msg.ID = &token
	}
	s.mu.Unlock()
	if err != nil {
		return true
	}
	if historyPrivate(msg.Method) {
		if msg.ID != nil {
			historyControlError(c, *msg.ID, codeMethodNotFound, "method not found")
		}
		return true
	}
	if msg.IsRequest() {
		return s.handleClientRequest(c, msg)
	}
	if msg.IsNotification() {
		if historyPublic(msg.Method) {
			return true
		}
		if msg.Method == methodShutdown {
			s.detach(c, true)
			return false
		}
		if strings.HasSuffix(msg.Method, unsubscribeSuffix) || strings.HasPrefix(msg.Method, servicePrefix) {
			return true
		}
		// Ordinary notifications have already passed their ordinary decoder.
		s.forwardNotification(frame[:len(frame)-1])
	}
	return true
}

func (s *Service) pumpHistoryStdout(run *brokerRun) {
	reader := bufio.NewReaderSize(run.proc.stdout, 4096)
	for {
		msg, frame, reply, err := s.readHistoryMessage(run, reader)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				_ = run.proc.kill()
			}
			return
		}
		if reply {
			// This is an independently owned ordinary control frame, not a raw,
			// codec or projection borrow. A stopped old writer cannot pin H.
			_ = run.writeFrame(frame)
			continue
		}
		if msg == nil {
			continue
		}
		switch {
		case msg.IsResponse():
			s.routeResponse(msg)
		case msg.IsNotification():
			s.handleBrokerNotification(run, msg.Method, frame)
		case msg.IsRequest():
			_ = run.writeMessage(&jsonrpc.Message{JSONRPC: "2.0", ID: msg.ID, Error: &jsonrpc.RPCError{Code: codeMethodNotFound, Message: "nvpair-service handles no requests from the broker"}})
		}
	}
}

// Only this scope borrows raw H or codec projection storage. Admission is
// checked under the withdrawal mutex AFTER acquiring the arena, so a retired
// pump cannot read a buffered next frame when it resumes. An admitted Read may
// finish after withdrawal, but its actual return precedes arena reuse. Ordinary
// decoded messages own their copies; selected history is synchronously applied
// or encoded into its independent output grant before this scope returns.
func (s *Service) readHistoryMessage(run *brokerRun, reader *bufio.Reader) (*jsonrpc.Message, []byte, bool, error) {
	h := s.history
	h.rawMu.Lock()
	defer h.rawMu.Unlock()
	s.mu.Lock()
	if s.broker != run || s.stopping {
		s.mu.Unlock()
		return nil, nil, false, io.EOF
	}
	generation := h.generation
	h.rawGrant()
	s.mu.Unlock()
	raw, err := readHistoryPhysical(reader, h.raw, jsonrpc.HistoryFrameBytes)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.broker != run || h.generation != generation {
		return nil, nil, false, io.EOF
	}
	if err != nil {
		return nil, nil, false, err
	}
	msg, err := h.guard(raw, &h.correlation)
	control := func(id []byte, code int, message string) (*jsonrpc.Message, []byte, bool, error) {
		var storage [jsonrpc.HistoryEnvelopeBytes]byte
		frame := historyControlFrame(storage[:0], id, code, message)
		// The bounded ordinary reply owner takes an independent copy before
		// raw/ID release, exactly as ordinary client refusal replies do.
		return nil, append([]byte(nil), frame...), true, nil
	}
	if err != nil {
		var failure *jsonrpc.HistoryFailure
		if errors.As(err, &failure) && failure.Refusal != nil {
			r := failure.Refusal
			return control(r.ReplyID, r.Code, r.Message)
		}
		s.historyGuardFailure(err)
		if historyFailure(err) {
			return nil, nil, false, nil
		}
		return nil, nil, false, err
	}
	if s.historyBrokerMessage(run, msg) {
		return nil, nil, false, nil
	}
	if msg.IsRequest() && historyPublic(msg.Method) {
		return control(*msg.ID, codeMethodNotFound, "method not found")
	}
	if msg.IsNotification() {
		return msg, append([]byte(nil), raw...), false, nil
	}
	return msg, nil, false, nil
}

// Typed history failures are handled here, before the ordinary peer pump. They
// never enter broker supervision's terminal framing path or raw diagnostics.
func (s *Service) historyGuardFailure(err error) {
	h := s.history
	var failure *jsonrpc.HistoryFailure
	if !errors.As(err, &failure) {
		return
	}
	if failure.ScheduleRefetch {
		h.queueRefetch()
	}
	if failure.FailPending && h.pending {
		h.pending = false
		if h.initialSlot >= 0 {
			if c := h.writes[h.initialSlot].client; c != nil {
				historyControlError(c, h.original[:h.originalN], jsonrpc.HistoryInvalidCode, "history-invalid")
			}
			h.release(h.initialSlot)
		}
		// The bad raw frame has finished; an unfinished control Write still
		// owns E and admission. No absent future response owns receiver bytes.
		h.queueRefetch()
	}
}

func (s *Service) historyRefetch() {
	s.mu.Lock()
	h := s.history
	if h.exhausted || h.controlBroken || h.retryCancelled {
		h.cancelRefetch()
		s.mu.Unlock()
		return
	}
	if !h.refetch || h.initialSlot >= 0 || h.controlActive || s.broker == nil || s.stopping {
		s.mu.Unlock()
		return
	}
	slot := h.freeSlot()
	if slot < 0 || !h.initialMu.TryLock() {
		s.mu.Unlock()
		return
	}
	run := s.broker
	h.initialGrant()
	h.outputGrant(slot)
	if h.prepareControl(run) != nil {
		h.initialMu.Unlock()
		s.mu.Unlock()
		return
	}
	h.initialSlot = slot
	w := &h.writes[slot]
	*w = historyWrite{storage: w.storage, generation: h.generation, initial: true, active: true}
	generation := h.generation
	s.mu.Unlock()
	s.writeHistoryControl(run, generation)
}

// Caller holds Service.mu. Already admitted output frames remain pinned in
// their write slots; clearing the cache cannot recycle those bytes.
func (s *Service) historyBrokerExited(run *brokerRun) {
	h := s.history
	h.retryCancelled = true
	h.cancelRefetch()
	h.generation++
	h.loaded, h.hasFloor, h.exhausted, h.controlBroken = false, false, false, false
	for _, bank := range h.banks {
		if bank != nil {
			bank.n = 0
		}
	}
	if h.pendingRun == run && h.pending {
		if h.initialSlot >= 0 {
			if c := h.writes[h.initialSlot].client; c != nil {
				historyControlError(c, h.original[:h.originalN], jsonrpc.HistoryInvalidCode, "history-invalid")
			}
			h.release(h.initialSlot)
		}
		h.pending = false
	}
	h.pendingRun = nil
	h.correlation = jsonrpc.HistoryCorrelation{}
	_ = h.correlation.EnableGeneratedIDs()
	for c := range s.clients {
		c.history.revision = h.clock
	}
}

func historyInitialRecords(raw []byte) []byte {
	// The shared projector emits known keys and punctuation without whitespace.
	start := bytes.IndexByte(raw, '[')
	if start < 0 {
		return nil
	}
	return raw[start+1:]
}

func historyNextRecord(raw []byte) ([]byte, []byte) {
	if len(raw) == 0 || raw[0] != '{' {
		return nil, nil
	}
	depth, quoted, escaped := 0, false, false
	for i, b := range raw {
		if quoted {
			if escaped {
				escaped = false
			} else if b == '\\' {
				escaped = true
			} else if b == '"' {
				quoted = false
			}
			continue
		}
		if b == '"' {
			quoted = true
			continue
		}
		if b == '{' || b == '[' {
			depth++
		}
		if b == '}' || b == ']' {
			depth--
		}
		if depth == 0 {
			rest := raw[i+1:]
			if len(rest) > 0 && rest[0] == ',' {
				rest = rest[1:]
			}
			return raw[:i+1], rest
		}
	}
	return nil, nil
}
