// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
)

const (
	routingHistoryCellBytes    = 65536
	routingHistoryWindowBytes  = 256 * routingHistoryCellBytes
	routingHistoryControlBytes = 131072
	routingHistoryBytes        = 2*routingHistoryWindowBytes + routingHistoryCellBytes + 2*jsonrpc.HistoryRecordFrameBytes + jsonrpc.HistoryRecordBytes + routingHistoryControlBytes
	routingHistoryReconcile    = 5 * time.Second
)

// A cell is fixed-width packed native scalar/tag/UTF-8 data, with no strings,
// slices or pointers. It never borrows a native payload or a request owner.
// This keeps the bank independent of Go DTO graphs and JSON escaping growth.
type routingProjectionCell struct {
	facts    routingNativeCopy
	sequence uint64
	_        [routingHistoryCellBytes - unsafe.Sizeof(routingNativeCopy{}) - 8]byte
}

type routingProjectionWindow struct {
	cells *[256]routingProjectionCell
	order [256]uint16 // Full-key order; source rank order is a separate concern.
	count int
}

func (w *routingProjectionWindow) find(id uint64) int {
	for i := 0; i < w.count; i++ {
		if w.cells[i].facts.id == id {
			return i
		}
	}
	return -1
}

func (w *routingProjectionWindow) sort() {
	for i := 0; i < w.count; i++ {
		w.order[i] = uint16(i)
		for j := i; j > 0 && w.cells[w.order[j]].facts.id < w.cells[w.order[j-1]].facts.id; j-- {
			w.order[j], w.order[j-1] = w.order[j-1], w.order[j]
		}
	}
}

// Exactly one copier calls reconcile, and exactly one writer calls writeCycle.
// mu protects handoff/ack scalars, never any inference observation. The writer
// owns frozen and frame[0] from scheduling through actual completion. current
// and scratch remain copier-owned while a pipe is stopped at any boundary.
type proxyRoutingHistory struct {
	mu                   sync.Mutex
	proxy                *Proxy
	parent               context.Context // Borrowed process lifetime; no history-private context.
	current, frozen      routingProjectionWindow
	scratch              routingProjectionCell
	frames               [2][jsonrpc.HistoryRecordFrameBytes]byte
	guard                [jsonrpc.HistoryRecordBytes / 4]uint32
	correlation          jsonrpc.HistoryCorrelation // No outgoing history calls or waiter IDs.
	process              routingProcessDescriptor
	generation           nodepolicy.RoutingHistoryGeneration
	generationText       [39]byte
	generationLength     int
	dirty, ready, active bool
	acknowledged, failed bool
	started              bool
	wake, writeWake      chan struct{}
}

// C census for the application-owned representation. The owner row includes
// both window indexes, process copy, generation text, all flags/mutexes and
// channel/pointer headers. The source directory includes its ONE scan bank.
// The two frames and guard also fund acknowledgement parameter projection;
// construction does not start the parent read loop or any worker.
//
// Additional conservative runtime estimates come from the inspected Go 1.25
// src/runtime/{chan,time,runtime2}.go
// and src/context/context.go: hchan=112 <=128, timeTimer=112 <=128,
// afterFuncCtx=104 <=128, sudog=88 <=96. A standard cancelCtx's small child map
// has a 48B header and one 136B interface-key group. Deletion does not reclaim
// that map charge. These estimates do not prove private layouts in uninspected
// toolchains. Tests check actual reflectable shapes/upper bounds, application
// owner referents, and the registered Write callback's actual lifetime without
// restricting the supported Go version.
//
// These are representation capacities, not allocator/GC/stack/RSS bounds.
// Existing Proxy/Codec/transport, parent cancellation tree and logging sink
// remain borrowed baseline owners. Only one J child association is added to
// that tree; no history-private context, ordinary-call subtree, log buffer or
// replacement transport is created here. Stack-resident application scratch
// and captures are conservatively included below even when they do not escape.
const (
	routingHistoryChannelControlBytes = 128
	routingHistoryTimerObjectBytes    = 128
	routingHistoryAfterFuncBytes      = 128
	routingHistoryWaiterBytes         = 96
	routingHistoryChildMapBytes       = 48 + 136

	routingHistoryOwnerControlBytes = unsafe.Sizeof(proxyRoutingHistory{}) - routingHistoryCellBytes - 2*jsonrpc.HistoryRecordFrameBytes - jsonrpc.HistoryRecordBytes +
		unsafe.Sizeof(sync.Mutex{}) + unsafe.Sizeof((*proxyRoutingHistory)(nil)) // Proxy's constructor fields.
	routingHistoryNativeControlBytes = unsafe.Sizeof(routingSourceDirectory{}) + unsafe.Sizeof(routingProcessDescriptor{}) + unsafe.Sizeof(routingFactsClock)
	routingHistoryDescriptorOverlap  = 32*800 + 528 + 288 // One current native read, no second pointer/copy bank.
	// Empty struct channel elements have zero backing: wake, writeWake, Run's
	// copier-join, run's writer-join, shared active-Write callback-join; plus the
	// lazy parent Done and AfterFunc child's Done (normally shared closedchan).
	routingHistoryRoleChannelBytes = (2 + 1 + 1 + 1 + 1 + 1) * routingHistoryChannelControlBytes
	// One reused timeTimer, its physical capacity-one Time channel, and two
	// timerWhen entries conservatively covering old/new heap bookkeeping.
	routingHistoryTimerControlBytes = routingHistoryTimerObjectBytes + routingHistoryChannelControlBytes + unsafe.Sizeof(time.Time{}) + 2*(8+8)
	// Exactly one active Write: AfterFunc object, stop closure (code+object),
	// and its retained parent-association backing. Stop/join completes before
	// another frame may acquire the codec; these never multiply per record.
	routingHistoryCallbackBytes = routingHistoryAfterFuncBytes + 16 + routingHistoryChildMapBytes
	// Copier select has three cases, idle writer has two, callback join one.
	// Charging all six simultaneously overcounts the mutually exclusive idle
	// and active writer phases, rather than hiding a waiter list in a margin.
	routingHistoryWaitControlBytes = (3 + 2 + 1) * routingHistoryWaiterBytes
	// Worker launch; timer.Stop defer; writer close(done) defer; reconcile's
	// unlock, retry, reader-clear and scratch-clear captures (code + captures).
	routingHistoryWorkerCaptureBytes = 24 + 16 + 16 + 16 + 24 + 16 + 16
	// shared writeOwnedContext: mutex/flags, interrupt closure, callback,
	// finish closure, return-error defer and gate-release defer. Also fund
	// the named error box and context.AfterFunc's stop/once inner captures.
	routingHistoryWriteCaptureBytes = unsafe.Sizeof(sync.Mutex{}) + 2 + 40 + 24 + 56 + 48 + 16 + 16 + 24 + 16
	// Canonical cycle step: caller's cursor, receiver/cursor pointers, method
	// header, slot, returned error, dirty and loop-result flags. No stored queue
	// or additional record/frame backing is introduced by stepping the writer.
	routingHistoryCycleStepBytes = unsafe.Sizeof(int(0)) + 2*unsafe.Sizeof(uintptr(0)) + unsafe.Sizeof("") +
		unsafe.Sizeof(int(0)) + unsafe.Sizeof(error(nil)) + 2*unsafe.Sizeof(false)
	// One uint128 conversion: digits, returned string backing and header.
	// The retained 39B generationText is already in the owner row.
	routingHistoryGenerationBytes = 39 + 39 + unsafe.Sizeof("")
	// One bounded encoder, one decimal scratch/conversion and frame headers
	// across writeFrame -> writeRoutingFrame -> WritePreencoded.
	routingHistoryEncoderBytes = unsafe.Sizeof(routingHistoryJSON{}) + 20 + 20 + unsafe.Sizeof("") + 3*unsafe.Sizeof([]byte{})
	// The raw input backing replaces the baseline ordinary 1MiB allowance.
	// Count every added reader/header/reference here, including its 256-byte
	// fragment buffer. Correlation is inline in the owner row above; its IDs
	// and method referents stay empty because this direction originates no calls.
	routingHistoryIngressControlBytes = unsafe.Sizeof(jsonrpc.HistoryReader{}) + unsafe.Sizeof(bufio.Reader{}) + 256 +
		unsafe.Sizeof(io.ReadWriter(nil)) + unsafe.Sizeof((func())(nil)) + unsafe.Sizeof([]byte{}) +
		routingHistoryGrammarBytes + 25600 + 2048 + // Raw guard and accepted bounded semantic/cursor scratch, including wrong-direction records.
		unsafe.Sizeof(jsonrpc.HistoryFailure{}) + unsafe.Sizeof(jsonrpc.HistoryRequestRefusal{}) +
		2*unsafe.Sizeof(jsonrpc.Message{}) + jsonrpc.HistoryIDBytes + 128 // One synchronous refusal frame, no queue.
	// Shared WritePreencoded's nonrecursive grammar validator has a fixed
	// 10001B state stack, two (raw slice + index pointer + code) closures and
	// its depth/index words. It does not use encoding/json's growing pool.
	routingHistoryGrammarBytes = 10001 + 2*(24+8+8) + 2*8
	// The receiver's bounded ack, its generation referent and one
	// fixed error value. No raw input, decoder text or per-ack retained graph.
	routingHistoryAckControlBytes = unsafe.Sizeof(nodepolicy.RoutingHistoryAccepted{}) + 39 + unsafe.Sizeof(nodepolicy.RoutingHistoryDiagnostic{})
	// One synchronous fixed category/count diagnostic. LogAttrs uses Record's
	// inline attributes; the call and handler handoff may each hold a Record.
	// All text is fixed; the existing log handler/sink is not a J output queue.
	routingHistoryDiagnosticBytes = 2*unsafe.Sizeof(slog.Record{}) + 2*unsafe.Sizeof(slog.Attr{}) +
		uintptr(len("routing history projection limited")+len("category")+len("limit-exceeded")+len("count"))

	routingHistoryControlUsed = routingHistoryOwnerControlBytes + routingHistoryNativeControlBytes + routingHistoryDescriptorOverlap +
		routingHistoryRoleChannelBytes + routingHistoryTimerControlBytes + routingHistoryCallbackBytes + routingHistoryWaitControlBytes +
		routingHistoryWorkerCaptureBytes + routingHistoryWriteCaptureBytes + routingHistoryCycleStepBytes + routingHistoryGenerationBytes + routingHistoryEncoderBytes +
		routingHistoryGrammarBytes + routingHistoryAckControlBytes + routingHistoryDiagnosticBytes + routingHistoryIngressControlBytes
)

var (
	_ [routingHistoryControlBytes - routingHistoryControlUsed]byte
	_ [routingHistoryCellBytes - unsafe.Sizeof(routingProjectionCell{})]byte
	_ [unsafe.Sizeof(routingProjectionCell{}) - routingHistoryCellBytes]byte
)

func newProxyRoutingHistory(p *Proxy) *proxyRoutingHistory {
	return &proxyRoutingHistory{
		proxy:   p,
		parent:  p.codec.root,
		current: routingProjectionWindow{cells: new([256]routingProjectionCell)},
		frozen:  routingProjectionWindow{cells: new([256]routingProjectionCell)},
		wake:    make(chan struct{}, 1), writeWake: make(chan struct{}, 1),
	}
}

func routingHistoryWake(ch chan struct{}) {
	select {
	case ch <- struct{}{}:
	default:
	}
}

// Run calls this once on the parent lifetime. There is one timer and one writer,
// including during ack loss, source churn and arbitrarily long active writes.
func (h *proxyRoutingHistory) run() {
	h.mu.Lock()
	if h.started {
		h.mu.Unlock()
		return
	}
	h.started = true
	h.mu.Unlock()
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-h.parent.Done():
				return
			case <-h.writeWake:
				h.writeCycle()
			}
		}
	}()
	timer := time.NewTimer(routingHistoryReconcile)
	defer timer.Stop()
	h.reconcile(false)
	for {
		select {
		case <-h.parent.Done():
			<-done // An active borrower, including its interrupt callback, must join.
			return
		case <-h.wake:
			h.reconcile(false)
		case <-timer.C:
			h.reconcile(true)
			timer.Reset(routingHistoryReconcile)
		}
	}
}

// A failed scan cannot publish a partial membership or mint a generation.
// Accepted per-key revisions may already be coalesced in current; they retain
// their seq on retry. Neither the bank nor any index retains native pointers.
func (h *proxyRoutingHistory) reconcile(retry bool) bool {
	d := h.proxy.routingFactSource() // Never nest a J lock under identityMu.
	if d == nil {
		return false
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	defer func() {
		// A changing/odd source does not prevent replay of the last complete
		// window. It can only prevent publishing a replacement window.
		if retry && !h.failed && !h.active && !h.acknowledged && h.generationLength != 0 {
			h.active = true
			routingHistoryWake(h.writeWake)
		}
	}()
	if h.failed || !d.scan() {
		return false
	}
	defer d.reader.clear()
	defer func() { h.scratch = routingProjectionCell{} }()
	h.ready = false
	if h.generationLength == 0 {
		h.process = *d.process
	}
	w := &h.current
	// Remove only keys actually absent from this source epoch, never live-map
	// disappearance. Completed native facts survive registry unlink.
	for i := 0; i < w.count; {
		present := false
		for j := 0; j < d.reader.count; j++ {
			if d.reader.sources[j].id.Load() == w.cells[i].facts.id {
				present = true
				break
			}
		}
		if present {
			i++
			continue
		}
		w.count--
		w.cells[i].facts, w.cells[i].sequence = w.cells[w.count].facts, w.cells[w.count].sequence
		w.cells[w.count].facts, w.cells[w.count].sequence = routingNativeCopy{}, 0
		h.dirty = true
	}
	for i := 0; i < d.reader.count; i++ {
		f := d.reader.sources[i]
		at := w.find(f.id.Load())
		var revision, sequence uint64
		if at >= 0 {
			revision, sequence = w.cells[at].facts.revision, w.cells[at].sequence
		}
		if f.epoch.Load() == 2*revision && revision != 0 {
			continue
		}
		if !d.accept(i, revision, &h.scratch.facts) {
			return false
		}
		if sequence >= factMaxRevision {
			h.failed = true // History only; never change admission or parent lifetime.
			return false
		}
		if at < 0 {
			at = w.count
			w.count++
		}
		limited := h.scratch.facts.flags&factLimitExceeded != 0 && w.cells[at].facts.flags&factLimitExceeded == 0
		// Epoch/membership/native revision acceptance precedes seq assignment.
		w.cells[at].facts, w.cells[at].sequence = h.scratch.facts, sequence+1
		h.dirty = true
		if limited {
			// The source freezes this record at its final native revision. Its
			// accepted flag is the once marker; retries/replays cannot re-report
			// it, and an evicted source never re-enters this process's window.
			slog.LogAttrs(context.Background(), slog.LevelWarn, "routing history projection limited",
				slog.String("category", "limit-exceeded"), slog.Int("count", 1))
		}
	}
	// Freeze only a complete coherent scan. A later source mutation is a new
	// copier turn, not permission for an in-flight writer to change its window.
	for i := 0; i < d.reader.count; i++ {
		f := d.reader.sources[i]
		at := w.find(f.id.Load())
		if at < 0 || f.epoch.Load() != 2*w.cells[at].facts.revision {
			return false
		}
	}
	if d.epoch.Load() != d.reader.epoch {
		return false
	}
	w.sort()
	h.ready = true
	if h.active {
		return true
	}
	if h.dirty || h.generationLength == 0 {
		generation, err := h.generation.Next()
		if err != nil {
			h.failed = true
			return false
		}
		h.generationLength = copy(h.generationText[:], generation)
		h.frozen.count, h.frozen.order = w.count, w.order
		for i := 0; i < w.count; i++ {
			h.frozen.cells[i].facts, h.frozen.cells[i].sequence = w.cells[i].facts, w.cells[i].sequence
		}
		// Clear old record bytes too; the fixed bank has no dangling referents.
		for i := w.count; i < len(h.frozen.cells); i++ {
			h.frozen.cells[i].facts, h.frozen.cells[i].sequence = routingNativeCopy{}, 0
		}
		h.dirty, h.acknowledged = false, false
	} else if !retry || h.acknowledged {
		return true
	}
	h.active = true
	routingHistoryWake(h.writeWake)
	return true
}

// The sole parent read loop owns frames[1] and guard. Project before decoding,
// using the already admitted receiver backing rather than another DTO/guard.
func (h *proxyRoutingHistory) acceptNotification(raw []byte) {
	schema, ok := nodepolicy.RoutingHistoryTransportSchema(nodepolicy.NotifyRoutingHistoryAccepted, false)
	if !ok {
		return
	}
	clean, _, err := jsonrpc.ProjectHistory(raw, h.frames[1][:0], h.guard[:], schema)
	if err != nil {
		return
	}
	var ack nodepolicy.RoutingHistoryAccepted
	if jsonrpc.MaterializeHistory(clean, &ack, schema.Integer) == nil {
		h.accepted(ack)
	}
}

// Only the exact parent notification dispatcher supplies acknowledgements.
func (h *proxyRoutingHistory) accepted(ack nodepolicy.RoutingHistoryAccepted) bool {
	if ack.Validate() != nil {
		return false
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.generationLength == 0 || !bytes.Equal([]byte(ack.Generation), h.generationText[:h.generationLength]) {
		return false
	}
	h.acknowledged = true
	return true
}

func (h *proxyRoutingHistory) writeCycle() {
	// active was set before exposing the wake, so frozen cannot be reused even
	// if the writer has not entered Write yet. Only this writer uses frame[0].
	// Each writeFrame independently acquires/releases the codec gate for ONE
	// complete frame. No gate is held between frames, while awaiting an ack,
	// or across reconciliation/copier turns. A physically blocked pipe still
	// blocks ordinary proxy/broker traffic under its existing Notify lifetime.
	cursor := 0
	for h.writeCycleStep(&cursor) {
	}
}

// The sole writer owns this stack cursor: zero selects begin, then the actual
// frozen-order records, then end. A negative cursor has completed its handoff
// and cannot write again or release a later cycle's frozen bank. Completion is
// synchronous with the actual frame/callback borrower, including errors/busy.
func (h *proxyRoutingHistory) writeCycleStep(cursor *int) bool {
	if *cursor < 0 {
		return false
	}
	method, slot := nodepolicy.NotifyRoutingHistoryWindow, -1
	if *cursor > 0 && *cursor <= h.frozen.count {
		method, slot = nodepolicy.NotifyRoutingRequest, int(h.frozen.order[*cursor-1])
	} else if *cursor > h.frozen.count {
		method = nodepolicy.NotifyRoutingHistoryWindowEnd
	}
	err := h.writeFrame(method, slot)
	*cursor += 1
	if err == nil && *cursor <= h.frozen.count+1 {
		return true
	}
	*cursor = -1
	h.mu.Lock()
	h.active = false
	if err != nil && !errors.Is(err, jsonrpc.ErrHistoryBusy) {
		h.failed = true
	}
	dirty := h.dirty
	h.mu.Unlock()
	if dirty {
		routingHistoryWake(h.wake)
	}
	return false
}

func (h *proxyRoutingHistory) writeFrame(method string, slot int) error {
	w := routingHistoryJSON{data: h.frames[0][:0]}
	w.literal(`{"jsonrpc":"2.0","method":`)
	w.text(method)
	w.literal(`,"params":`)
	switch method {
	case nodepolicy.NotifyRoutingHistoryWindow:
		h.encodeWindow(&w)
	case nodepolicy.NotifyRoutingRequest:
		h.encodeRecord(&w, &h.frozen.cells[slot])
	case nodepolicy.NotifyRoutingHistoryWindowEnd:
		w.literal(`{"schema":1,"generation":`)
		w.quote(h.generationText[:h.generationLength])
		w.literal(`}`)
	default:
		return jsonrpc.ErrHistoryInvalid
	}
	w.literal("}\n")
	if w.err != nil {
		return w.err
	}
	return h.proxy.codec.writeRoutingFrame(h.parent, w.data)
}

// Fixed-capacity JSON output never calls encoding/json or allocates a DTO.
// Models retain their validated UTF-8 bytes; only JSON quote/backslash/control
// escaping expands them, directly into the admitted P frame.
type routingHistoryJSON struct {
	data []byte
	err  error
}

func (w *routingHistoryJSON) literal(s string) {
	if w.err == nil {
		if len(s) > cap(w.data)-len(w.data) {
			w.err = jsonrpc.ErrHistoryInvalid
			return
		}
		w.data = append(w.data, s...)
	}
}

func (w *routingHistoryJSON) quote(s []byte) {
	w.literal(`"`)
	for _, c := range s {
		switch {
		case c == '"' || c == '\\':
			w.literal(`\`)
			w.byte(c)
		case c < 32:
			const hex = "0123456789abcdef"
			w.literal(`\u00`)
			w.byte(hex[c>>4])
			w.byte(hex[c&15])
		default:
			w.byte(c)
		}
	}
	w.literal(`"`)
}

func (w *routingHistoryJSON) byte(c byte) {
	if w.err == nil {
		if len(w.data) == cap(w.data) {
			w.err = jsonrpc.ErrHistoryInvalid
			return
		}
		w.data = append(w.data, c)
	}
}

func (w *routingHistoryJSON) text(s string) { w.quote([]byte(s)) }

func (w *routingHistoryJSON) number(n uint64) {
	var digits [20]byte
	w.literal(string(strconv.AppendUint(digits[:0], n, 10)))
}

func (w *routingHistoryJSON) nullable(n uint64, known bool, quoted bool) {
	if !known {
		w.literal(`null`)
		return
	}
	if quoted {
		w.literal(`"`)
	}
	w.number(n)
	if quoted {
		w.literal(`"`)
	}
}

func (h *proxyRoutingHistory) encodeKey(w *routingHistoryJSON, id uint64) {
	w.literal(`{"originNodeId":`)
	w.quote(h.process.origin[:h.process.originLen])
	w.literal(`,"proxyRunId":`)
	w.quote(h.process.run[:h.process.runLen])
	w.literal(`,"requestId":"`)
	w.number(id)
	w.literal(`"}`)
}

func (h *proxyRoutingHistory) encodeWindow(w *routingHistoryJSON) {
	w.literal(`{"schema":1,"generation":`)
	w.quote(h.generationText[:h.generationLength])
	w.literal(`,"cutoff":`)
	if h.frozen.count < 256 {
		w.literal(`null`)
	} else {
		least := &h.frozen.cells[0].facts
		for i := 1; i < h.frozen.count; i++ {
			f := &h.frozen.cells[i].facts
			if f.created < least.created || f.created == least.created && f.id < least.id {
				least = f
			}
		}
		w.literal(`{"request":`)
		h.encodeKey(w, least.id)
		w.literal(`,"createdAt":`)
		w.number(least.created)
		w.literal(`}`)
	}
	w.literal(`,"keys":[`)
	for i := 0; i < h.frozen.count; i++ {
		if i > 0 {
			w.literal(`,`)
		}
		c := &h.frozen.cells[h.frozen.order[i]]
		w.literal(`{"request":`)
		h.encodeKey(w, c.facts.id)
		w.literal(`,"createdAt":`)
		w.number(c.facts.created)
		w.literal(`,"seq":`)
		w.number(c.sequence)
		w.literal(`}`)
	}
	w.literal(`]}`)
}

var routingHistoryStates = [...]string{"routing", "streaming", "completed", "failed", "cancelled"}
var routingHistoryOutcomes = [...]string{"pending", "committed", "rejected", "failed", "redispatched", "interrupted", "cancelled"}
var routingHistoryTiers = [...]string{"", "weak", "medium", "strong"}
var routingHistoryModes = [...]string{"other", "chat", "completion", "embedding"}
var routingHistoryAccounting = [...]string{"none", "pending", "recorded", "retained", "failed", "unknown"}
var routingHistoryAccountStates = [...]string{"unknown", "ok", "cooldown", "frozen", "credential-failure", "exhausted", "probe", "conflict"}

func routingHistoryEnum(w *routingHistoryJSON, values []string, index uint64) {
	if index >= uint64(len(values)) {
		w.err = jsonrpc.ErrHistoryInvalid
		return
	}
	w.text(values[index])
}

func routingHistoryReason(w *routingHistoryJSON, disposition uint64) {
	routingHistoryEnum(w, routingFactReasons[:], disposition>>8&255)
}

func routingHistoryMoney(w *routingHistoryJSON, ceiling, granted, held, charged, flags uint64, request bool) {
	if request {
		w.literal(`{"currency":"USD","ceilingNanoUsd":`)
		w.nullable(ceiling, flags&factCeilingKnown != 0, true)
		w.literal(`,"grantedNanoUsd":`)
		w.nullable(granted, flags&factMoneyApplicable != 0, true)
	}
	w.literal(`,"heldNanoUsd":`)
	w.nullable(held, flags&factHeldKnown != 0, true)
	w.literal(`,"chargedNanoUsd":`)
	w.nullable(charged, flags&factChargeKnown != 0, true)
	w.literal(`,"accounting":`)
	routingHistoryEnum(w, routingHistoryAccounting[:], flags>>16&255)
	w.literal(`}`)
}

func routingHistoryDestination(w *routingHistoryJSON, d *routingDestinationDescriptor) {
	w.literal(`{"kind":`)
	switch d.kind {
	case 1:
		w.text("local")
	case 2:
		w.text("api")
	default:
		w.err = jsonrpc.ErrHistoryInvalid
		return
	}
	w.literal(`,"model":`)
	if d.modelKnown {
		w.quote(d.model[:d.modelLen])
	} else {
		w.literal(`null`)
	}
	w.literal(`,"nodeId":`)
	if d.kind == 1 {
		w.quote(d.first[:d.firstLen])
		w.literal(`,"engine":`)
		w.quote(d.second[:d.secondLen])
		w.literal(`,"deploymentId":null,"accountId":null}`)
	} else {
		w.literal(`null,"engine":null,"deploymentId":`)
		w.quote(d.first[:d.firstLen])
		w.literal(`,"accountId":`)
		w.quote(d.second[:d.secondLen])
		w.literal(`}`)
	}
}

func (h *proxyRoutingHistory) encodeRecord(w *routingHistoryJSON, cell *routingProjectionCell) {
	f := &cell.facts
	w.literal(`{"schema":1,"request":`)
	h.encodeKey(w, f.id)
	w.literal(`,"workload":`)
	dropped := uint64(0)
	if f.total > 32 {
		dropped = f.total - 32
	}
	bound := f.binding &^ factAttached
	if bound == 0 {
		w.literal(`null`)
	} else {
		w.literal(`{"originNodeId":`)
		w.quote(h.process.origin[:h.process.originLen])
		w.literal(`,"proxyRunId":`)
		w.quote(h.process.run[:h.process.runLen])
		w.literal(`,"engine":`)
		w.quote(f.workload.engine[:f.workload.engineLen])
		w.literal(`,"id":`)
		w.quote(f.workload.id[:f.workload.idLen])
		w.literal(`,"attemptIndex":`)
		w.number(bound - 1)
		w.literal(`,"attached":`)
		if f.binding&factAttached != 0 && bound > dropped {
			w.literal(`true}`)
		} else {
			w.literal(`false}`)
		}
	}
	w.literal(`,"seq":`)
	w.number(cell.sequence)
	w.literal(`,"requested":{"model":`)
	w.quote(f.requested.model[:f.requested.length])
	w.literal(`,"tier":`)
	routingHistoryEnum(w, routingHistoryTiers[:], uint64(f.requested.tier))
	w.literal(`,"mode":`)
	routingHistoryEnum(w, routingHistoryModes[:], uint64(f.requested.mode))
	w.literal(`},"createdAt":`)
	w.number(f.created)
	w.literal(`,"endedAt":`)
	w.nullable(f.ended, f.flags&factEndKnown != 0, false)
	w.literal(`,"state":`)
	routingHistoryEnum(w, routingHistoryStates[:], f.disposition&255)
	w.literal(`,"reason":`)
	routingHistoryReason(w, f.disposition)
	w.literal(`,"status":`)
	w.nullable(f.disposition>>16, f.disposition>>16 != 0, false)
	w.literal(`,"served":`)
	if f.served == 0 || f.served <= dropped || f.served > f.total {
		w.literal(`null`)
	} else {
		routingHistoryDestination(w, &f.attempts[(f.served-1)%32].destination)
	}
	w.literal(`,"recoverable":false,"spend":`)
	routingHistoryMoney(w, f.ceiling, f.granted, f.held, f.charged, f.flags, true)
	w.literal(`,"droppedAttempts":`)
	w.number(dropped)
	w.literal(`,"attempts":[`)
	for ordinal := dropped + 1; ordinal <= f.total; ordinal++ {
		if ordinal > dropped+1 {
			w.literal(`,`)
		}
		a := &f.attempts[(ordinal-1)%32]
		if a.ordinal != ordinal {
			w.err = jsonrpc.ErrHistoryInvalid
			return
		}
		w.literal(`{"index":`)
		w.number(ordinal - 1)
		w.literal(`,"startedAt":`)
		w.number(a.started)
		w.literal(`,"endedAt":`)
		w.nullable(a.ended, a.flags&factEndKnown != 0, false)
		w.literal(`,"destination":`)
		routingHistoryDestination(w, &a.destination)
		w.literal(`,"outcome":`)
		routingHistoryEnum(w, routingHistoryOutcomes[:], a.disposition&255)
		w.literal(`,"reason":`)
		routingHistoryReason(w, a.disposition)
		w.literal(`,"status":`)
		w.nullable(a.disposition>>16, a.disposition>>16 != 0, false)
		w.literal(`,"waits":{"admissionMs":`)
		w.number(a.admission)
		w.literal(`,"firstByteMs":`)
		w.nullable(a.firstByte, a.flags&factFirstByteKnown != 0, false)
		w.literal(`},"continuationOf":`)
		w.nullable(a.continuation-1, a.continuation != 0, false)
		w.literal(`,"account":`)
		if a.destination.kind != 2 {
			w.literal(`null`)
		} else {
			w.literal(`{"state":`)
			routingHistoryEnum(w, routingHistoryAccountStates[:], a.flags>>24&255)
			routingHistoryMoney(w, 0, 0, a.held, a.charged, a.flags, false)
		}
		w.literal(`}`)
	}
	w.literal(`]}`)
}
