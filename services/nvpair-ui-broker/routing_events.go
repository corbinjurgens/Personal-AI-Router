// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/jsonrpc"
	"nvpair-shared/nodepolicy"
	"nvpair-shared/pipeio"
)

const (
	historyCellBytes    = 65536
	historyWindowBytes  = nodepolicy.RoutingHistoryRecords * historyCellBytes
	historyControlBytes = 131072
	historyMaxSequence  = int64(9007199254740991)
	// The slot prefix contains only broker metadata and fixed key/rank fields.
	// Packed DTO bytes follow it; wire escaping is performed only at output.
	historySlotHeader   = 8*4 + 1 + 2*3 + 128*2 + 19
	historySlotOrdinal  = 0
	historySlotVersion  = 8
	historySlotSequence = 16
	historySlotCreated  = 24
	historySlotTerminal = 32
	historySlotKey      = 33
)

type routingHistoryBank struct {
	data []byte
	// Valid packed metadata is below 48KiB (header plus 32 bounded attempts).
	// Two-byte lengths fund every bank index in C without increasing C or W.
	length [nodepolicy.RoutingHistoryRecords]uint16
	count  int
}

func newRoutingHistoryBank() *routingHistoryBank {
	return &routingHistoryBank{data: make([]byte, historyWindowBytes)}
}

func (w *routingHistoryBank) cell(i int) []byte {
	return w.data[i*historyCellBytes : (i+1)*historyCellBytes]
}

func (w *routingHistoryBank) reset() {
	clear(w.length[:])
	w.count = 0
}

func historyWord(cell []byte, at int) uint64 { return binary.LittleEndian.Uint64(cell[at:]) }
func historyPutWord(cell []byte, at int, value uint64) {
	binary.LittleEndian.PutUint64(cell[at:], value)
}

func historyKeyParts(cell []byte) (origin, run, id []byte) {
	at := historySlotKey
	part := func() []byte {
		n := int(binary.LittleEndian.Uint16(cell[at:]))
		at += 2
		value := cell[at : at+n]
		at += n
		return value
	}
	origin, run, id = part(), part(), part()
	return
}

func historySameKey(a, b []byte) bool {
	ao, ar, ai := historyKeyParts(a)
	bo, br, bi := historyKeyParts(b)
	return bytes.Equal(ao, bo) && bytes.Equal(ar, br) && bytes.Equal(ai, bi)
}

// Positive means a is newer in the public ordering (origin/run lexical,
// request counter numeric descending on timestamp ties).
func historyRank(a, b []byte) int {
	ac, bc := historyWord(a, historySlotCreated), historyWord(b, historySlotCreated)
	if ac < bc {
		return -1
	}
	if ac > bc {
		return 1
	}
	ao, ar, ai := historyKeyParts(a)
	bo, br, bi := historyKeyParts(b)
	if c := bytes.Compare(ao, bo); c != 0 {
		return -c
	}
	if c := bytes.Compare(ar, br); c != 0 {
		return -c
	}
	if len(ai) < len(bi) {
		return -1
	}
	if len(ai) > len(bi) {
		return 1
	}
	return bytes.Compare(ai, bi)
}

func historyHeader(cell []byte, key nodepolicy.RoutingRequestKey, created, seq int64, ordinal uint64) {
	clear(cell[:historySlotHeader])
	historyPutWord(cell, historySlotOrdinal, ordinal)
	historyPutWord(cell, historySlotSequence, uint64(seq))
	historyPutWord(cell, historySlotCreated, uint64(created))
	at := historySlotKey
	for _, value := range []string{key.OriginNodeID, key.ProxyRunID, key.RequestID} {
		binary.LittleEndian.PutUint16(cell[at:], uint16(len(value)))
		at += 2
		at += copy(cell[at:], value)
	}
}

func (w *routingHistoryBank) find(cell []byte) int {
	for i := 0; i < w.count; i++ {
		if historySameKey(w.cell(i), cell) {
			return i
		}
	}
	return -1
}

// insert keeps only the newest window, without an auxiliary DTO collection.
func (w *routingHistoryBank) insert(cell []byte, n uint16) {
	if i := w.find(cell); i >= 0 {
		copy(w.cell(i), cell[:n])
		w.length[i] = n
		return
	}
	at := 0
	for at < w.count && historyRank(w.cell(at), cell) > 0 {
		at++
	}
	if at == nodepolicy.RoutingHistoryRecords {
		return
	}
	if w.count < nodepolicy.RoutingHistoryRecords {
		w.count++
	}
	for i := w.count - 1; i > at; i-- {
		copy(w.cell(i), w.cell(i - 1)[:w.length[i-1]])
		w.length[i] = w.length[i-1]
	}
	copy(w.cell(at), cell[:n])
	w.length[at] = n
}

// brokerRoutingHistory is the sole bank owner. mu may precede workersMu;
// workersMu must never precede mu. No writer holds mu across I/O. The fourth
// bank belongs exclusively to the global initial transfer until actual Write
// returns, including blocked, partial and failed writes.
type brokerRoutingHistory struct {
	mu                              sync.Mutex
	b                               *Broker
	parent                          context.Context
	lifetime                        *brokerMainLifetime
	accepted, stage, merge, initial *routingHistoryBank
	scratch                         [historyCellBytes]byte
	guard                           [jsonrpc.HistoryRecordBytes / 4]uint32
	readers                         routingHistoryReaders
	projected                       [jsonrpc.HistoryRecordFrameBytes]byte
	floor                           [historySlotHeader]byte
	hasFloor                        bool
	stageSource                     *proxyProcess
	stageGeneration                 string
	stageDigest                     [32]byte
	stageCutoff                     [historySlotHeader]byte
	stageHasCutoff                  bool
	lastSource                      *proxyProcess
	lastGeneration                  string
	lastDigest                      [32]byte
	version                         uint64
	retiredThrough                  uint64
	subscribed                      bool
	subscription                    uint64
	cursor                          uint64
	ackSource                       *proxyProcess
	ackGeneration                   string
	// One source-scoped quarantine/refetch token. Reconciliation belongs to
	// the producer's existing unacknowledged-window timer, not a broker Call.
	refetchSource       *proxyProcess
	quarantined         bool
	refusal             routingHistoryRefusal
	refusalActive       bool
	initialBusy         atomic.Bool
	initialWriters      sync.WaitGroup
	started             bool
	interruptLocalWrite func()
	wake                chan struct{}
	// Barrier before the short apply-time workersMu check, outside workersMu.
	beforeApply func()
	// Instance barrier after acquisition cancellation, before its callback joins.
	beforeInitialDeadlineDone func()
	beforeLocalRetirementDone func()
}

type routingHistoryRefusal struct {
	source  *proxyProcess
	id      [jsonrpc.HistoryIDBytes]byte
	n       int
	code    int
	message string
}

// This private handle is not a context.Context. Only these constructors can
// register on root: one Serve child and one history child, never ordinary RPCs.
// It is single-use, including after cancellation; no old callback can be hidden
// by replacing a lifecycle on the same handle.
type brokerMainLifetime struct {
	mu            sync.Mutex
	root          context.Context
	rootCancel    context.CancelFunc
	serve         context.Context
	history       context.Context
	historyCancel context.CancelFunc
	stopped       bool
	localRead     bool
}

func newBrokerMainLifetime() *brokerMainLifetime {
	root, cancel := context.WithCancel(context.Background())
	return &brokerMainLifetime{root: root, rootCancel: cancel}
}

func (l *brokerMainLifetime) cancel() { l.rootCancel() }

func (l *brokerMainLifetime) startServe() (context.Context, context.CancelFunc, error) {
	l.mu.Lock()
	if l.stopped || l.root.Err() != nil || l.serve != nil {
		l.mu.Unlock()
		return nil, nil, jsonrpc.ErrHistoryBusy
	}
	ctx, cancel := context.WithCancel(l.root)
	l.serve = ctx
	l.mu.Unlock()
	return ctx, func() {
		l.mu.Lock()
		l.stopped = true
		stopHistory := l.historyCancel
		l.mu.Unlock()
		if stopHistory != nil {
			stopHistory()
		}
		cancel()
	}, nil
}

// A qualified local read registration belongs to the history root, whose only
// other children are one initial acquisition and one serial proxy Write. Generic
// readLoop callers retain their original context and are not C-qualified here.
func (l *brokerMainLifetime) localRetirementParent(ctx context.Context) (context.Context, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.serve != ctx {
		return ctx, nil
	}
	if l.localRead {
		return nil, jsonrpc.ErrHistoryBusy
	}
	l.localRead = true
	return l.history, nil
}

// Task318 constructs this owner and starts run once. The typed handle enforces
// its parent construction without enabling production history startup here.
func newBrokerRoutingHistory(b *Broker, lifetime *brokerMainLifetime) (*brokerRoutingHistory, error) {
	if lifetime == nil {
		return nil, jsonrpc.ErrHistoryInvalid
	}
	lifetime.mu.Lock()
	if lifetime.stopped || lifetime.root.Err() != nil || lifetime.history != nil {
		lifetime.mu.Unlock()
		return nil, jsonrpc.ErrHistoryBusy
	}
	parent, cancel := context.WithCancel(lifetime.root)
	lifetime.history, lifetime.historyCancel = parent, cancel
	lifetime.mu.Unlock()
	return &brokerRoutingHistory{b: b, parent: parent, lifetime: lifetime, accepted: newRoutingHistoryBank(),
		stage: newRoutingHistoryBank(), merge: newRoutingHistoryBank(), initial: newRoutingHistoryBank(), wake: make(chan struct{}, 1)}, nil
}

func routingHistoryMethod(method string) bool {
	_, bare := engines.SplitAddressedMethod(method)
	switch bare {
	case nodepolicy.NotifyRoutingHistoryWindow, nodepolicy.NotifyRoutingRequest, nodepolicy.NotifyRoutingHistoryWindowEnd,
		nodepolicy.NotifyRoutingHistoryAccepted, nodepolicy.NotifyRoutingChanged,
		nodepolicy.MethodRoutingSubscribe, nodepolicy.MethodRoutingUnsubscribe, nodepolicy.MethodRoutingGetInitial:
		return true
	}
	return false
}

func routingHistoryPrivateMethod(method string) bool {
	if !routingHistoryMethod(method) {
		return false
	}
	return method != nodepolicy.MethodRoutingSubscribe && method != nodepolicy.MethodRoutingUnsubscribe && method != nodepolicy.MethodRoutingGetInitial
}

// bindRoutingHistoryActivation reads only facts already established by the
// policy owner. A DTO, first frame, facade handle or Done cannot authorize a
// binding. The caller is the verified activation completion, after unlock.
func (b *Broker) bindRoutingHistoryActivation(source *proxyProcess) bool {
	if b.routingHistory == nil || source == nil {
		return false
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	valid := r.incarnation == source && r.runID != "" && r.activeRevision != "" && r.activeRevision == r.current.Revision && r.activationErr == ""
	origin, run := b.nodeID, r.runID
	r.mu.Unlock()
	if !valid || nodepolicy.ValidateRoutingIdentity(origin, run) != nil {
		return false
	}
	b.workersMu.Lock()
	defer b.workersMu.Unlock()
	if b.proxyProc != source || b.historyIncarnation == 0 || b.historyIncarnation <= b.historyRetiredThrough {
		return false
	}
	select {
	case <-source.Done():
		return false
	default:
	}
	if b.historyRun != "" && (b.historyRun != run || b.historyOrigin != origin) {
		return false
	}
	b.historyOrigin, b.historyRun = origin, run
	return true
}

func (b *Broker) historySourceLocked(source *proxyProcess) bool {
	if source == nil || b.proxyProc != source || b.historyIncarnation == 0 ||
		b.historyIncarnation <= b.historyRetiredThrough || b.historyOrigin != b.nodeID || b.historyRun == "" {
		return false
	}
	select {
	case <-source.Done():
		return false
	default:
		return true
	}
}

func (b *Broker) fenceRoutingHistoryLocked(source *proxyProcess) {
	if source != nil && source == b.proxyProc && b.historyIncarnation > b.historyRetiredThrough {
		b.historyRetiredThrough = b.historyIncarnation
		b.historyOrigin, b.historyRun = "", ""
	}
}

func (b *Broker) fenceRoutingHistory() {
	b.workersMu.Lock()
	source := b.proxyProc
	b.fenceRoutingHistoryLocked(source)
	b.workersMu.Unlock()
	source.retireRoutingHistoryRead()
	b.wakeRoutingHistory()
}

func (b *Broker) wakeRoutingHistory() {
	if h := b.routingHistory; h != nil {
		select {
		case h.wake <- struct{}{}:
		default:
		}
	}
}

func (h *brokerRoutingHistory) discardStageLocked() {
	h.stage.reset()
	h.stageSource = nil
	h.stageGeneration = ""
	h.stageHasCutoff = false
}

// The callback changes bounded history state only. No I/O or Call runs on the
// read pump. There are no history Peer.Call waiters on this private link.
func (b *Broker) routingHistoryFailure(source *proxyProcess, failure jsonrpc.HistoryFailure) {
	if h := b.routingHistory; h != nil {
		if failure.Refusal != nil {
			// This worker hop accepts private notifications only. An unaccepted
			// wrong-direction call has no reply guarantee and cannot disturb an
			// in-progress window or schedule a refetch.
			h.stageRefusal(source, failure.Refusal)
			return
		}
		h.mu.Lock()
		b.workersMu.Lock()
		valid := b.historySourceLocked(source)
		b.workersMu.Unlock()
		if !valid {
			h.mu.Unlock()
			return
		}
		switch failure.Disposition {
		case "discard", "refuse-request":
			if h.stageSource == source {
				h.discardStageLocked()
			}
		case "quarantine", "fail-call":
			// fail-call cannot refer to an admitted history call here. Never
			// apply FailPending to the ordinary Peer pending map.
			if h.stageSource == source {
				h.discardStageLocked()
			}
			h.quarantined = true
			h.refetchSource = source
		}
		if failure.ScheduleRefetch {
			h.refetchSource = source
		}
		if h.refetchSource == source && h.ackSource == source {
			h.ackSource, h.ackGeneration = nil, ""
		}
		h.mu.Unlock()
	}
	slog.Debug("routing history frame rejected", "method", "history-unknown", "category", "history-invalid", "count", failure.Count)
}

// Best-effort refusal applies only to unaccepted wrong-direction worker calls.
// Local public controls retain their actual reply/borrow-completion contract.
// The sole funded slot stays occupied through actual Write. An occupied slot
// discards another request immediately, without copying its borrowed ID.
func (h *brokerRoutingHistory) stageRefusal(source *proxyProcess, refusal *jsonrpc.HistoryRequestRefusal) {
	if source == nil || source.peer == nil || len(refusal.ReplyID) > jsonrpc.HistoryIDBytes {
		return
	}
	h.mu.Lock()
	h.b.workersMu.Lock()
	valid := h.b.historySourceLocked(source)
	h.b.workersMu.Unlock()
	if !valid {
		h.mu.Unlock()
		return
	}
	if h.refusal.source != nil {
		h.mu.Unlock()
		slog.Debug("routing history frame rejected", "method", "history-unknown", "category", "history-worker-refusal-busy", "count", 1)
		return
	}
	h.refusal.source = source
	h.refusal.n = copy(h.refusal.id[:], refusal.ReplyID)
	h.refusal.code, h.refusal.message = refusal.Code, refusal.Message
	h.mu.Unlock()
	slog.Debug("routing history frame rejected", "method", "history-unknown", "category", "history-invalid", "count", 1)
	h.b.wakeRoutingHistory()
}

func (b *Broker) routeRoutingHistoryFrom(source *proxyProcess, method string, raw json.RawMessage) {
	h := b.routingHistory
	if h == nil || source == nil {
		return
	}
	// Addressed aliases and public methods from a worker are always refused.
	if method != nodepolicy.NotifyRoutingHistoryWindow && method != nodepolicy.NotifyRoutingRequest && method != nodepolicy.NotifyRoutingHistoryWindowEnd {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	b.workersMu.Lock()
	valid := b.historySourceLocked(source)
	origin, run, ordinal := b.historyOrigin, b.historyRun, b.historyIncarnation
	b.workersMu.Unlock()
	if !valid {
		return
	}
	schema, ok := nodepolicy.RoutingHistoryTransportSchema(method, false)
	if !ok {
		return
	}
	clean, _, err := jsonrpc.ProjectHistory(raw, h.projected[:0], h.guard[:], schema)
	if err != nil {
		h.discardStageLocked()
		return
	}
	switch method {
	case nodepolicy.NotifyRoutingHistoryWindow:
		window, err := nodepolicy.ReadProjectedRoutingHistoryWindow(clean)
		if err != nil {
			h.discardStageLocked()
			return
		}
		// Digest and source-check one bounded key at a time. The manifest
		// view borrows projected bytes; no full Keys graph survives a pass.
		digest := sha256.New()
		err = window.ForEachKey(func(_ int, key nodepolicy.RoutingHistoryWindowKey) error {
			if key.Request.OriginNodeID != origin || key.Request.ProxyRunID != run {
				return jsonrpc.ErrHistoryInvalid
			}
			historyHeader(h.scratch[:], key.Request, key.CreatedAt, key.Seq, ordinal)
			_, _ = digest.Write(h.scratch[:historySlotHeader])
			return nil
		})
		if err != nil {
			h.discardStageLocked()
			return
		}
		var sum [32]byte
		_ = digest.Sum(sum[:0])
		if h.lastSource == source {
			cmp := historyGenerationCompare(window.Generation, h.lastGeneration)
			if cmp < 0 || (cmp == 0 && sum != h.lastDigest) {
				h.discardStageLocked()
				return
			}
		}
		if h.stageSource == source {
			cmp := historyGenerationCompare(window.Generation, h.stageGeneration)
			if cmp < 0 || (cmp == 0 && sum != h.stageDigest) {
				h.discardStageLocked()
				return
			}
		}
		h.discardStageLocked()
		h.stageSource, h.stageGeneration, h.stageDigest = source, window.Generation, sum
		// Only a validated source-bound full begin clears quarantine. The
		// producer still receives no acknowledgement before a matching end.
		h.quarantined = false
		if h.refetchSource == source {
			h.refetchSource = nil
		}
		h.stage.count = window.KeyCount()
		if err := window.ForEachKey(func(i int, key nodepolicy.RoutingHistoryWindowKey) error {
			historyHeader(h.stage.cell(i), key.Request, key.CreatedAt, key.Seq, ordinal)
			return nil
		}); err != nil {
			h.discardStageLocked()
			return
		}
		if window.Cutoff != nil {
			h.stageHasCutoff = true
			historyHeader(h.stageCutoff[:], window.Cutoff.Request, window.Cutoff.CreatedAt, 0, 0)
		}
	case nodepolicy.NotifyRoutingRequest:
		if h.stageSource != source {
			return
		}
		view, err := nodepolicy.ReadProjectedRoutingHistoryRecord(clean)
		record := &view.Header
		if err != nil || record.Request.OriginNodeID != origin || record.Request.ProxyRunID != run {
			h.discardStageLocked()
			return
		}
		historyHeader(h.scratch[:], record.Request, record.CreatedAt, record.Seq, ordinal)
		i := h.stage.find(h.scratch[:])
		if i < 0 || historyWord(h.stage.cell(i), historySlotCreated) != uint64(record.CreatedAt) || historyWord(h.stage.cell(i), historySlotSequence) != uint64(record.Seq) {
			h.discardStageLocked()
			return
		}
		n, err := historyPackProjectedRecord(h.scratch[:], &view)
		if err != nil {
			h.discardStageLocked()
			return
		}
		if h.stage.length[i] != 0 && !bytes.Equal(h.stage.cell(i)[:h.stage.length[i]], h.scratch[:n]) {
			h.discardStageLocked()
			return
		}
		copy(h.stage.cell(i), h.scratch[:n])
		h.stage.length[i] = uint16(n)
	case nodepolicy.NotifyRoutingHistoryWindowEnd:
		var end nodepolicy.RoutingHistoryWindowEnd
		if jsonrpc.MaterializeHistory(clean, &end, schema.Integer) != nil || end.Validate() != nil || h.stageSource != source || end.Generation != h.stageGeneration {
			h.discardStageLocked()
			return
		}
		for i := 0; i < h.stage.count; i++ {
			if h.stage.length[i] == 0 {
				h.discardStageLocked()
				return
			}
		}
		if !h.commitStageLocked(source) {
			h.discardStageLocked()
			return
		}
		h.lastSource, h.lastGeneration, h.lastDigest = source, end.Generation, h.stageDigest
		h.ackSource, h.ackGeneration = source, end.Generation
		h.discardStageLocked()
		b.wakeRoutingHistory()
	}
}

func historyGenerationCompare(a, b string) int {
	if len(a) < len(b) {
		return -1
	}
	if len(a) > len(b) {
		return 1
	}
	return strings.Compare(a, b)
}

func (h *brokerRoutingHistory) commitStageLocked(source *proxyProcess) bool {
	h.merge.reset()
	for i := 0; i < h.accepted.count; i++ {
		h.merge.insert(h.accepted.cell(i), h.accepted.length[i])
	}
	nextVersion := h.version
	for i := 0; i < h.stage.count; i++ {
		cell, n := h.stage.cell(i), h.stage.length[i]
		old := h.accepted.find(cell)
		if old >= 0 {
			prior := h.accepted.cell(old)
			if historyWord(prior, historySlotCreated) != historyWord(cell, historySlotCreated) || historyWord(prior, historySlotSequence) > historyWord(cell, historySlotSequence) ||
				(prior[historySlotTerminal] != 0 && cell[historySlotTerminal] == 0) {
				return false
			}
			if historyWord(prior, historySlotSequence) == historyWord(cell, historySlotSequence) {
				// Broker version differs; all source facts must be identical.
				if h.accepted.length[old] != n || !bytes.Equal(prior[historySlotSequence:n], cell[historySlotSequence:n]) {
					return false
				}
				continue
			}
		} else if h.hasFloor && historyRank(cell, h.floor[:]) <= 0 {
			continue
		}
		if nextVersion == ^uint64(0) {
			return false
		}
		nextVersion++
		historyPutWord(cell, historySlotVersion, nextVersion)
		h.merge.insert(cell, n)
	}
	if h.beforeApply != nil {
		h.beforeApply()
	}
	// Copy/merge above is outside workersMu. This is the common atomic fence
	// with actual withdrawal; no callback admitted earlier can pass it later.
	h.b.workersMu.Lock()
	if !h.b.historySourceLocked(source) {
		h.b.workersMu.Unlock()
		return false
	}
	h.accepted, h.merge = h.merge, h.accepted
	h.version = nextVersion
	h.b.workersMu.Unlock()
	if h.stageHasCutoff {
		h.advanceFloorLocked(h.stageCutoff[:])
	}
	if h.accepted.count == nodepolicy.RoutingHistoryRecords {
		h.advanceFloorLocked(h.accepted.cell(h.accepted.count - 1))
	}
	return true
}

func (h *brokerRoutingHistory) advanceFloorLocked(cell []byte) {
	if !h.hasFloor || historyRank(cell, h.floor[:]) > 0 {
		copy(h.floor[:], cell[:historySlotHeader])
		h.hasFloor = true
	}
}

// retireLocked runs at the bank owner, independent of workload retirement.
// The monotonic frontier coalesces any number of departed processes to a scalar.
func (h *brokerRoutingHistory) retireLocked() {
	h.b.workersMu.Lock()
	frontier := h.b.historyRetiredThrough
	stageCurrent := h.b.historySourceLocked(h.stageSource)
	refetchCurrent := h.b.historySourceLocked(h.refetchSource)
	refusalCurrent := h.b.historySourceLocked(h.refusal.source)
	h.b.workersMu.Unlock()
	if frontier <= h.retiredThrough {
		return
	}
	if h.stageSource != nil && !stageCurrent {
		h.discardStageLocked()
	}
	if h.refetchSource != nil && !refetchCurrent {
		h.refetchSource, h.quarantined = nil, false
	}
	if !h.refusalActive && h.refusal.source != nil && !refusalCurrent {
		h.clearRefusalLocked()
	}
	h.merge.reset()
	now := time.Now().UnixMilli()
	for i := 0; i < h.accepted.count; i++ {
		cell, n := h.accepted.cell(i), h.accepted.length[i]
		if historyWord(cell, historySlotOrdinal) <= frontier && cell[historySlotTerminal] == 0 && historyWord(cell, historySlotSequence) == uint64(historyMaxSequence) {
			slog.Debug("routing history projection stopped", "category", "limit-exceeded", "count", 1)
		}
		if historyWord(cell, historySlotOrdinal) <= frontier && cell[historySlotTerminal] == 0 && historyWord(cell, historySlotSequence) < uint64(historyMaxSequence) && h.version != ^uint64(0) {
			var record nodepolicy.RoutingHistoryRecord
			if historyUnpackRecord(cell[:n], &record) != nil {
				h.merge.insert(cell, n)
				continue
			}
			end := max(now, record.CreatedAt)
			for _, attempt := range record.Attempts {
				end = max(end, attempt.StartedAt)
				if attempt.EndedAt != nil {
					end = max(end, *attempt.EndedAt)
				}
			}
			end = min(end, historyMaxSequence)
			record.Seq++
			record.State, record.Reason, record.EndedAt, record.Recoverable = "failed", "proxy-exited", &end, false
			for j := range record.Attempts {
				if record.Attempts[j].EndedAt == nil {
					record.Attempts[j].EndedAt = &end
				}
			}
			historyHeader(h.scratch[:], record.Request, record.CreatedAt, record.Seq, historyWord(cell, historySlotOrdinal))
			packed, err := historyPackRecord(h.scratch[:], record)
			if err != nil || record.Validate() != nil {
				h.merge.insert(cell, n)
				continue
			}
			h.version++
			historyPutWord(h.scratch[:], historySlotVersion, h.version)
			h.merge.insert(h.scratch[:], uint16(packed))
		} else {
			h.merge.insert(cell, n)
		}
	}
	h.b.workersMu.Lock()
	h.accepted, h.merge = h.merge, h.accepted
	h.b.workersMu.Unlock()
	h.retiredThrough = frontier
}

func (h *brokerRoutingHistory) clearRefusalLocked() {
	h.refusal = routingHistoryRefusal{}
	h.refusalActive = false
}

// Required local replies use the ordinary synchronous write owner, separately
// from the worker's best-effort refusal slot. One local dispatcher can overlap
// the single initial fallback. Both retain their bounded frame through gate
// admission and actual Write; neither enqueues or silently drops a busy reply.
func respondLocalHistoryError(codec *Codec, id *json.RawMessage, code int) {
	if err := codec.RespondHistoryError(id, code); err != nil {
		slog.Debug("routing history control reply failed", "category", "history-write-failed", "count", 1)
	}
}

func (b *Broker) handleRoutingHistoryRequest(msg *Message) {
	h := b.routingHistory
	if h == nil {
		// Explicit ordinary constructors retain ordinary ID serialization.
		if err := b.codec.RespondError(msg.ID, -32601, "method not found"); err != nil {
			slog.Debug("routing history control reply failed", "category", "history-write-failed", "count", 1)
		}
		return
	}
	// Validate params through the same lossless projector even for direct
	// dispatcher callers. Only known empty control objects are materialized.
	schema, _ := nodepolicy.RoutingHistoryTransportSchema(msg.Method, false)
	if msg.ID == nil || len(*msg.ID) > jsonrpc.HistoryIDBytes {
		respondLocalHistoryError(b.codec, nil, jsonrpc.HistoryInvalidCode)
		return
	}
	h.mu.Lock()
	_, _, validationErr := jsonrpc.ProjectHistory(msg.Params, h.projected[:0], h.guard[:], schema)
	h.mu.Unlock()
	if validationErr != nil {
		respondLocalHistoryError(b.codec, msg.ID, jsonrpc.HistoryInvalidCode)
		return
	}
	if msg.Method != nodepolicy.MethodRoutingGetInitial {
		h.mu.Lock()
		subscribed := msg.Method == nodepolicy.MethodRoutingSubscribe
		if h.subscribed != subscribed {
			h.subscribed = subscribed
			h.subscription++
			h.cursor = h.version
		}
		h.mu.Unlock()
		b.wakeRoutingHistory()
	}
	// J311 defines the initial shape for all three public history replies.
	// Subscription registration is scalar and survives a busy baseline reply.
	if !h.initialBusy.CompareAndSwap(false, true) {
		respondLocalHistoryError(b.codec, msg.ID, jsonrpc.HistoryBusyCode)
		return
	}
	// The input ID is borrowed from HistoryReader. Copy its bounded token
	// before returning to the read pump; never retain msg or its Params.
	var id [jsonrpc.HistoryIDBytes]byte
	n := copy(id[:], *msg.ID)
	h.initialWriters.Add(1)
	go func(codec *Codec) {
		defer h.initialWriters.Done()
		h.writeInitial(codec, id, n)
	}(b.codec)
}

var errRoutingHistoryInitialDeadline = errors.New("routing history initial acquisition deadline")

// Only the initial writer's own expiry maps to DeadlineExceeded at the shared
// gate and its admission recheck. A parent's cancellation cause stays distinct.
// WithCancelCause supplies first-cancellation ordering without a hidden timer.
type routingHistoryInitialAcquire struct{ context.Context }

func (a routingHistoryInitialAcquire) Err() error {
	err := a.Context.Err()
	if err == context.Canceled && context.Cause(a.Context) == errRoutingHistoryInitialDeadline {
		return context.DeadlineExceeded
	}
	return err
}

func (h *brokerRoutingHistory) writeInitial(codec *Codec, id [jsonrpc.HistoryIDBytes]byte, idN int) {
	defer h.initialBusy.Store(false)
	h.mu.Lock()
	h.retireLocked()
	h.initial.reset()
	for i := 0; i < h.accepted.count; i++ {
		h.initial.insert(h.accepted.cell(i), h.accepted.length[i])
	}
	h.mu.Unlock()
	// Two H grants: the encoder's value and the complete output frame. Both
	// stay owned until the actual write joins; caller timeout is irrelevant.
	value := historyBoundedJSON{data: make([]byte, 0, jsonrpc.HistoryFrameBytes)}
	value.literal(`{"requests":[`)
	for i := 0; i < h.initial.count; i++ {
		if i != 0 {
			value.literal(",")
		}
		if err := historyRecordJSON(&value, h.initial.cell(i)[:h.initial.length[i]]); err != nil {
			return
		}
	}
	value.literal("]}")
	if value.err != nil || len(value.data) > jsonrpc.HistoryInitialBytes {
		return
	}
	frame := historyBoundedJSON{data: make([]byte, 0, jsonrpc.HistoryFrameBytes)}
	frame.literal(`{"jsonrpc":"2.0","id":`)
	frame.raw(id[:idN])
	frame.literal(`,"result":`)
	frame.raw(value.data)
	frame.literal("}\n")
	if frame.err != nil {
		return
	}
	deadline := time.Now().Add(250 * time.Millisecond)
	acquire, cancel := context.WithCancelCause(h.parent)
	deadlineDone := make(chan struct{})
	beforeDeadlineDone := h.beforeInitialDeadlineDone
	timer := time.AfterFunc(time.Until(deadline), func() {
		cancel(errRoutingHistoryInitialDeadline)
		if beforeDeadlineDone != nil {
			beforeDeadlineDone()
		}
		close(deadlineDone)
	})
	defer func() {
		// Stop(true) prevents this callback. Stop(false) is not completion:
		// join the actual deadline tail before the outer admission defer runs.
		if !timer.Stop() {
			<-deadlineDone
		}
		cancel(nil)
	}()
	err := codec.WritePreencoded(routingHistoryInitialAcquire{acquire}, h.parent, frame.data)
	cancel(nil)
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, jsonrpc.ErrHistoryBusy) {
		token := json.RawMessage(id[:idN])
		respondLocalHistoryError(codec, &token, jsonrpc.HistoryBusyCode)
	}
}

// One writer and one retry timer. Refusal alternates with accepted work, whose
// turns alternate ack/data. Even continuous refusal refill cannot starve either.
// Pending output is scalar state; each turn reselects the newest accepted bank. Slow
// output cannot pin old accepted/staging banks or block lifecycle withdrawal.
func (h *brokerRoutingHistory) run() {
	h.mu.Lock()
	if h.started {
		h.mu.Unlock()
		return
	}
	h.started = true
	h.mu.Unlock()
	timer := time.NewTimer(time.Hour)
	if !timer.Stop() {
		<-timer.C
	}
	defer timer.Stop()
	frame := historyBoundedJSON{data: make([]byte, 0, jsonrpc.HistoryRecordFrameBytes)}
	ackTurn := true
	refusalTurn := true
	for {
		select {
		case <-h.parent.Done():
			return
		case <-h.wake:
		case <-timer.C:
		}
		for {
			frame.data, frame.err = frame.data[:0], nil
			h.mu.Lock()
			h.retireLocked()
			source, generation := h.ackSource, h.ackGeneration
			index := -1
			if h.subscribed {
				for i := 0; i < h.accepted.count; i++ {
					v := historyWord(h.accepted.cell(i), historySlotVersion)
					if v > h.cursor && (index < 0 || v < historyWord(h.accepted.cell(index), historySlotVersion)) {
						index = i
					}
				}
			}
			refusalPending := h.refusal.source != nil
			if source == nil && index < 0 && !refusalPending {
				h.mu.Unlock()
				break
			}
			refusing := refusalPending && (refusalTurn || (source == nil && index < 0))
			refusalTurn = !refusing
			ack := !refusing && source != nil && (ackTurn || index < 0)
			if !refusing {
				ackTurn = !ackTurn
			}
			version, subscription := uint64(0), h.subscription
			codec := h.b.codec
			if refusing {
				source = h.refusal.source
				h.b.workersMu.Lock()
				current := h.b.historySourceLocked(source)
				h.b.workersMu.Unlock()
				if !current || source.peer == nil {
					h.clearRefusalLocked()
					h.mu.Unlock()
					continue
				}
				codec = source.peer.Codec()
				frame.literal(`{"jsonrpc":"2.0","id":`)
				frame.raw(h.refusal.id[:h.refusal.n])
				frame.literal(`,"error":{"code":`)
				var code [20]byte
				frame.raw(strconv.AppendInt(code[:0], int64(h.refusal.code), 10))
				frame.literal(`,"message":`)
				frame.quote(h.refusal.message)
				frame.literal("}}\n")
				h.refusalActive = true
			} else if ack {
				h.b.workersMu.Lock()
				current := h.b.historySourceLocked(source)
				h.b.workersMu.Unlock()
				if !current || source.peer == nil {
					h.ackSource = nil
					h.mu.Unlock()
					continue
				}
				codec = source.peer.Codec()
				frame.literal(`{"jsonrpc":"2.0","method":"routing/history-accepted","params":{"schema":1,"generation":`)
				frame.quote(generation)
				frame.literal("}}\n")
			} else {
				cell := h.accepted.cell(index)
				version = historyWord(cell, historySlotVersion)
				frame.literal(`{"jsonrpc":"2.0","method":"routing:changed","params":`)
				if historyRecordJSON(&frame, cell[:h.accepted.length[index]]) != nil {
					frame.err = jsonrpc.ErrHistoryInvalid
				}
				frame.literal("}\n")
			}
			h.mu.Unlock()
			if frame.err != nil {
				if refusing {
					h.mu.Lock()
					h.clearRefusalLocked()
					h.mu.Unlock()
				}
				return
			}
			err := codec.WritePreencoded(nil, h.parent, frame.data)
			if errors.Is(err, jsonrpc.ErrHistoryBusy) {
				if refusing {
					h.mu.Lock()
					h.refusalActive = false
					h.mu.Unlock()
				}
				timer.Reset(250 * time.Millisecond)
				break
			}
			if err != nil {
				if (ack || refusing) && h.parent.Err() == nil {
					// A broken departed proxy pipe belongs to that source's
					// supervisor. It cannot stop the broker's client writer.
					h.mu.Lock()
					if refusing {
						h.clearRefusalLocked()
					} else if h.ackSource == source && h.ackGeneration == generation {
						h.ackSource = nil
					}
					h.mu.Unlock()
					continue
				}
				if refusing {
					h.mu.Lock()
					h.clearRefusalLocked()
					h.mu.Unlock()
				}
				return
			}
			h.mu.Lock()
			if refusing {
				h.clearRefusalLocked()
			} else if ack {
				if h.ackSource == source && h.ackGeneration == generation {
					h.ackSource = nil
				}
			} else if h.subscription == subscription {
				h.cursor = max(h.cursor, version)
			}
			h.mu.Unlock()
		}
	}
}

// Fixed-capacity binary packing is private to the schema-1 bank. Reflection
// walks only the shared known-field DTO, never an input map or unknown JSON.
// Strings are UTF-8, so escaped wire expansion cannot overflow a 64KiB cell.
func historyPackRecord(cell []byte, record nodepolicy.RoutingHistoryRecord) (int, error) {
	if record.State == "completed" || record.State == "failed" || record.State == "cancelled" {
		cell[historySlotTerminal] = 1
	}
	out := cell[historySlotHeader:historySlotHeader:len(cell)]
	if err := historyPackValue(&out, reflect.ValueOf(record)); err != nil {
		return 0, err
	}
	if historySlotHeader+len(out) > 65535 {
		return 0, jsonrpc.ErrHistoryInvalid
	}
	return historySlotHeader + len(out), nil
}

// Ingestion borrows the proper owner's projected view. The packed layout is
// the same record prefix, count byte and attempts as historyPackRecord, with
// one replaceable attempt slot and no whole-record DTO materialization.
func historyPackProjectedRecord(cell []byte, view *nodepolicy.RoutingHistoryRecordView) (int, error) {
	if view.Header.State == "completed" || view.Header.State == "failed" || view.Header.State == "cancelled" {
		cell[historySlotTerminal] = 1
	}
	out := cell[historySlotHeader:historySlotHeader:len(cell)]
	if err := historyPackValue(&out, reflect.ValueOf(view.Header)); err != nil {
		return 0, err
	}
	if len(out) == cap(out) || view.AttemptCount() > nodepolicy.RoutingHistoryAttempts {
		return 0, jsonrpc.ErrHistoryInvalid
	}
	out = append(out, byte(view.AttemptCount()))
	var attempt nodepolicy.RoutingHistoryAttempt
	for i := 0; i < view.AttemptCount(); i++ {
		if err := view.ReadAttempt(i, &attempt); err != nil {
			return 0, err
		}
		if err := historyPackValue(&out, reflect.ValueOf(attempt)); err != nil {
			return 0, err
		}
	}
	if historySlotHeader+len(out) > 65535 {
		return 0, jsonrpc.ErrHistoryInvalid
	}
	return historySlotHeader + len(out), nil
}

func historyPackValue(out *[]byte, v reflect.Value) error {
	put := func(b []byte) error {
		if len(b) > cap(*out)-len(*out) {
			return jsonrpc.ErrHistoryInvalid
		}
		*out = append(*out, b...)
		return nil
	}
	switch v.Kind() {
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if err := historyPackValue(out, v.Field(i)); err != nil {
				return err
			}
		}
	case reflect.Pointer:
		flag := byte(0)
		if !v.IsNil() {
			flag = 1
		}
		if err := put([]byte{flag}); err != nil {
			return err
		}
		if flag != 0 {
			return historyPackValue(out, v.Elem())
		}
	case reflect.String:
		s := v.String()
		if len(s) > 65535 {
			return jsonrpc.ErrHistoryInvalid
		}
		var n [2]byte
		binary.LittleEndian.PutUint16(n[:], uint16(len(s)))
		if err := put(n[:]); err != nil {
			return err
		}
		if len(s) > cap(*out)-len(*out) {
			return jsonrpc.ErrHistoryInvalid
		}
		*out = append(*out, s...)
	case reflect.Int64:
		var n [8]byte
		binary.LittleEndian.PutUint64(n[:], uint64(v.Int()))
		return put(n[:])
	case reflect.Bool:
		n := byte(0)
		if v.Bool() {
			n = 1
		}
		return put([]byte{n})
	case reflect.Slice:
		if v.Len() > nodepolicy.RoutingHistoryAttempts {
			return jsonrpc.ErrHistoryInvalid
		}
		if err := put([]byte{byte(v.Len())}); err != nil {
			return err
		}
		for i := 0; i < v.Len(); i++ {
			if err := historyPackValue(out, v.Index(i)); err != nil {
				return err
			}
		}
	default:
		return jsonrpc.ErrHistoryInvalid
	}
	return nil
}

func historyUnpackRecord(cell []byte, record *nodepolicy.RoutingHistoryRecord) error {
	if len(cell) < historySlotHeader {
		return jsonrpc.ErrHistoryInvalid
	}
	raw := cell[historySlotHeader:]
	if err := historyUnpackValue(&raw, reflect.ValueOf(record).Elem()); err != nil {
		return err
	}
	if len(raw) != 0 {
		return jsonrpc.ErrHistoryInvalid
	}
	return nil
}

func historyUnpackValue(raw *[]byte, v reflect.Value) error {
	take := func(n int) ([]byte, error) {
		if len(*raw) < n {
			return nil, jsonrpc.ErrHistoryInvalid
		}
		p := (*raw)[:n]
		*raw = (*raw)[n:]
		return p, nil
	}
	switch v.Kind() {
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if err := historyUnpackValue(raw, v.Field(i)); err != nil {
				return err
			}
		}
	case reflect.Pointer:
		flag, err := take(1)
		if err != nil {
			return err
		}
		if flag[0] > 1 {
			return jsonrpc.ErrHistoryInvalid
		}
		if flag[0] == 1 {
			v.Set(reflect.New(v.Type().Elem()))
			return historyUnpackValue(raw, v.Elem())
		}
	case reflect.String:
		n, err := take(2)
		if err != nil {
			return err
		}
		p, err := take(int(binary.LittleEndian.Uint16(n)))
		if err != nil {
			return err
		}
		v.SetString(string(p))
	case reflect.Int64:
		p, err := take(8)
		if err != nil {
			return err
		}
		v.SetInt(int64(binary.LittleEndian.Uint64(p)))
	case reflect.Bool:
		p, err := take(1)
		if err != nil {
			return err
		}
		if p[0] > 1 {
			return jsonrpc.ErrHistoryInvalid
		}
		v.SetBool(p[0] == 1)
	case reflect.Slice:
		n, err := take(1)
		if err != nil {
			return err
		}
		if int(n[0]) > nodepolicy.RoutingHistoryAttempts {
			return jsonrpc.ErrHistoryInvalid
		}
		v.Set(reflect.MakeSlice(v.Type(), int(n[0]), int(n[0])))
		for i := 0; i < v.Len(); i++ {
			if err := historyUnpackValue(raw, v.Index(i)); err != nil {
				return err
			}
		}
	default:
		return jsonrpc.ErrHistoryInvalid
	}
	return nil
}

type historyBoundedJSON struct {
	data []byte
	err  error
}

func (w *historyBoundedJSON) raw(p []byte) {
	if w.err != nil {
		return
	}
	if len(p) > cap(w.data)-len(w.data) {
		w.err = jsonrpc.ErrHistoryInvalid
		return
	}
	w.data = append(w.data, p...)
}
func (w *historyBoundedJSON) literal(s string) {
	if w.err != nil {
		return
	}
	if len(s) > cap(w.data)-len(w.data) {
		w.err = jsonrpc.ErrHistoryInvalid
		return
	}
	w.data = append(w.data, s...)
}
func (w *historyBoundedJSON) quote(s string) {
	w.literal(`"`)
	for i := 0; i < len(s); i++ {
		w.quoteByte(s[i])
	}
	w.literal(`"`)
}

func (w *historyBoundedJSON) quotePacked(s []byte) {
	w.literal(`"`)
	for _, c := range s {
		w.quoteByte(c)
	}
	w.literal(`"`)
}

func (w *historyBoundedJSON) quoteByte(c byte) {
	const hex = "0123456789abcdef"
	switch {
	case c == '"' || c == '\\':
		w.raw([]byte{'\\', c})
	case c < 32:
		w.raw([]byte{'\\', 'u', '0', '0', hex[c>>4], hex[c&15]})
	default:
		w.raw([]byte{c})
	}
}

func historyRecordJSON(w *historyBoundedJSON, cell []byte) error {
	if len(cell) < historySlotHeader {
		return jsonrpc.ErrHistoryInvalid
	}
	// Walk the existing schema's packed representation directly. Initial and
	// push encoders may overlap, so neither may allocate a record DTO graph or
	// copied string referents against the shared C allowance. The only borrowed
	// bytes are this owner's cell and its already-admitted output grant.
	raw := cell[historySlotHeader:]
	start := len(w.data)
	historyPackedJSON(w, &raw, reflect.TypeOf(nodepolicy.RoutingHistoryRecord{}))
	if len(raw) != 0 || len(w.data)-start > jsonrpc.HistoryRecordBytes {
		return jsonrpc.ErrHistoryInvalid
	}
	return w.err
}

func historyTakePacked(raw *[]byte, n int) ([]byte, bool) {
	if len(*raw) < n {
		return nil, false
	}
	p := (*raw)[:n]
	*raw = (*raw)[n:]
	return p, true
}

// The type is always the fixed RoutingHistoryRecord schema or one of its
// fields; this is the inverse of historyPackValue, not an input JSON decoder.
func historyPackedJSON(w *historyBoundedJSON, raw *[]byte, typ reflect.Type) {
	if w.err != nil {
		return
	}
	switch typ.Kind() {
	case reflect.Struct:
		w.literal("{")
		for i := 0; i < typ.NumField(); i++ {
			if i != 0 {
				w.literal(",")
			}
			field := typ.Field(i)
			w.quote(field.Tag.Get("json"))
			w.literal(":")
			historyPackedJSON(w, raw, field.Type)
		}
		w.literal("}")
	case reflect.Pointer:
		flag, ok := historyTakePacked(raw, 1)
		if !ok || flag[0] > 1 {
			w.err = jsonrpc.ErrHistoryInvalid
			return
		}
		if flag[0] == 0 {
			w.literal("null")
		} else {
			historyPackedJSON(w, raw, typ.Elem())
		}
	case reflect.String:
		n, ok := historyTakePacked(raw, 2)
		if !ok {
			w.err = jsonrpc.ErrHistoryInvalid
			return
		}
		p, ok := historyTakePacked(raw, int(binary.LittleEndian.Uint16(n)))
		if !ok {
			w.err = jsonrpc.ErrHistoryInvalid
			return
		}
		w.quotePacked(p)
	case reflect.Int64:
		p, ok := historyTakePacked(raw, 8)
		if !ok {
			w.err = jsonrpc.ErrHistoryInvalid
			return
		}
		var buf [20]byte
		w.raw(strconv.AppendInt(buf[:0], int64(binary.LittleEndian.Uint64(p)), 10))
	case reflect.Bool:
		p, ok := historyTakePacked(raw, 1)
		if !ok || p[0] > 1 {
			w.err = jsonrpc.ErrHistoryInvalid
			return
		}
		if p[0] == 1 {
			w.literal("true")
		} else {
			w.literal("false")
		}
	case reflect.Slice:
		count, ok := historyTakePacked(raw, 1)
		if !ok || int(count[0]) > nodepolicy.RoutingHistoryAttempts {
			w.err = jsonrpc.ErrHistoryInvalid
			return
		}
		w.literal("[")
		for i := 0; i < int(count[0]); i++ {
			if i != 0 {
				w.literal(",")
			}
			historyPackedJSON(w, raw, typ.Elem())
		}
		w.literal("]")
	default:
		w.err = jsonrpc.ErrHistoryInvalid
	}
}

// attachRoutingHistoryPeer is the Task318 construction seam. It enables only
// the existing pump's private-notification dispositions, not history Call.
func (p *proxyProcess) attachRoutingHistoryPeer(b *Broker, rw io.ReadWriter, interruptRead func()) error {
	if b.routingHistory == nil {
		return jsonrpc.ErrHistoryInvalid
	}
	codec, err := b.routingHistory.readers.acquire(historyProxyPrivate, rw, interruptRead, p.interruptWrite)
	if err != nil {
		return err
	}
	p.peer = jsonrpc.NewPeerWithWriteInterrupt(codec, p.interruptWrite)
	if err := codec.SetReadRetirementFence(func() {
		b.workersMu.Lock()
		b.fenceRoutingHistoryLocked(p)
		b.workersMu.Unlock()
		b.wakeRoutingHistory()
	}); err != nil {
		codec.RetireRead()
		return err
	}
	if err := p.peer.SetHistoryReadHandlers(func(msg *Message) {
		// Peer admits this handler only for raw-validated borrowed history
		// requests, never by method-name or addressed-alias inference.
		b.routingHistory.stageRefusal(p, &jsonrpc.HistoryRequestRefusal{ReplyID: *msg.ID, Code: -32601, Message: "method not found"})
	}, func(f jsonrpc.HistoryFailure) { b.routingHistoryFailure(p, f) }); err != nil {
		codec.RetireRead()
		return err
	}
	return nil
}

// Actual subprocess construction seam for318: the retained stdout descriptor,
// rather than process termination or stdin, owns read interruption.
func (p *proxyProcess) attachRoutingHistoryProxyPipe(b *Broker, stdout io.ReadCloser) error {
	if stdout == nil || p.stdin == nil {
		return jsonrpc.ErrHistoryInvalid
	}
	if file, ok := stdout.(*os.File); ok {
		owned, err := pipeio.NewReader(file)
		if err != nil {
			return err
		}
		stdout = owned
	}
	return p.attachRoutingHistoryPeer(b, readWriter{stdout, p.stdin}, func() { _ = stdout.Close() })
}

func (p *proxyProcess) retireRoutingHistoryRead() <-chan struct{} {
	if p == nil || p.peer == nil {
		return nil
	}
	if !p.peer.Codec().ReadRetirementEnabled() {
		return nil
	}
	p.peer.Close()
	return p.peer.Codec().RetireRead()
}

func (p *proxyProcess) servePeer() {
	defer p.retireRoutingHistoryRead()
	p.peer.Serve(nil, p.handleNotify)
}
