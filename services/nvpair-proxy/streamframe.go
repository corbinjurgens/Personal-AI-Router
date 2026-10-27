// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Frame-aware reading of an inference response stream.
//
// The origin proxy forwards complete upstream frames and withholds a trailing
// fragment until the rest of it arrives, so a cut stream can never expose half
// an event. A stream that ends without its protocol's terminal marker is
// interrupted: the fragment is dropped, the reader terminates with a bounded
// operational result, and the client receives the protocol's own error frame
// followed by normal EOF instead of a torn event and a dead connection.
//
// Frames are held in memory only for the life of the request. Nothing here
// logs, persists or reconstructs request or response text.

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strconv"
	"sync"
	"time"
	"unicode/utf8"
)

var errStreamFrameBudget = errors.New("stream frame reader: bootstrap buffer unavailable")
var errStreamFrameProtocol = errors.New("stream frame reader: unsupported protocol")
var errStreamFrameAccount = errors.New("stream frame reader: an inference buffer account is required")

const (
	// streamFrameMaxBytes bounds one held raw frame. A frame larger than this
	// is not an inference event this proxy is willing to buffer.
	streamFrameMaxBytes int64 = 1 << 20

	// streamFrameRequestLimitBytes bounds each request's frame scope. Aggregate
	// frame storage competes with other inference storage for the same root.
	streamFrameRequestLimitBytes int64 = 64 << 20

	// streamFrameReadChunk is the fixed read size for one upstream pull. It is
	// a read scratch, not a frame buffer, and does not grow with a frame.
	streamFrameReadChunk = 4 << 10
	// Fixed state includes the reader, channels/timer, inline lease records and
	// copied nullable numbers. Validation uses a 10000-byte scanner stack; the
	// numeric path also covers a 64-digit mantissa, 768-byte canonical output,
	// 800-digit exact binary-float comparison, copied scalars and call frames.
	streamFrameMetadataBytes = 8 << 10
	streamFrameSyntaxBytes   = 16 << 10
	streamFrameNumericBytes  = 96 << 10
	// Even the smallest reader (syntax-only NDJSON) needs this storage in
	// addition to its choice bitmap. Larger counts cannot fit any reader.
	streamFrameMaxChoices = streamFrameRequestLimitBytes - streamFrameMetadataBytes - streamFrameSyntaxBytes - 2*streamFrameReadChunk

	// defaultStreamFrameIdle bounds the wait for the next complete upstream
	// event after the first body byte committed the attempt.
	defaultStreamFrameIdle = 60 * time.Second
)

// frameBudget reserves frame-buffer bytes before they are allocated so that a
// burst of streams terminates explicitly instead of waiting for memory.
type frameBudget struct {
	mu    sync.Mutex
	limit int64
	used  int64
	scope *bufferScope
	// An admitted reader borrows this bounded envelope instead of seeking new
	// root capacity. It stays charged until the last frame allocation releases.
	allowance *bufferLease
	leases    [32]struct {
		bytes   int64
		lease   *bufferLease
		pending bool
	}
}

// Prepay replacement overlap as well as useful storage. Growth is synchronous
// under reader.mu: either two raw arrays coexist with one retained SSE payload,
// or two payload arrays coexist with one raw array. Each is at most 1 MiB; the
// two replacement pairs never coexist. No parser or extractor retains slices.
func streamFrameAllowance(protocol streamProtocol, choices, parserBytes int64) int64 {
	n := streamFrameMetadataBytes + choices + streamFrameReadChunk + parserBytes + 2*streamFrameMaxBytes
	if protocol == streamProtocolSSE {
		n += streamFrameMaxBytes
	}
	return n
}

func (b *frameBudget) admit(ctx context.Context, n int64) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.allowance != nil || b.used != 0 {
		return errBufferInvalid
	}
	lease, err := b.scope.acquire(ctx, n)
	if err != nil {
		return err
	}
	b.allowance = lease
	b.limit = n
	return nil
}

func (b *frameBudget) acquireLocked(ctx context.Context, n int64, mayWait bool) (*bufferLease, error) {
	if b.allowance != nil {
		if err := ctx.Err(); err != nil {
			return nil, &bufferBusyError{cause: err}
		}
		return b.allowance.borrow()
	}
	if mayWait {
		return b.scope.acquire(ctx, n)
	}
	return b.scope.tryAcquire(ctx, n)
}

func (b *frameBudget) releaseAllowanceLocked() {
	if b.used == 0 && b.allowance != nil {
		b.allowance.close()
		b.allowance = nil
	}
}

func newFrameBudget(limit int64) *frameBudget {
	root, _ := newBufferBudget(limit, 0)
	scope, _ := root.newRequest(limit)
	return &frameBudget{limit: limit, scope: scope}
}

func ownedFrameBudget(ctx context.Context) *frameBudget {
	if ctx == nil {
		return nil
	}
	owner := inferenceBuffers(ctx)
	if owner == nil {
		return nil
	}
	if preparation, ok := ctx.Value(performancePreparationKey{}).(performancePreparation); ok && preparation.frames != nil {
		return preparation.frames
	}
	bootstrap := owner.borrowParserAllowance()
	if bootstrap == nil {
		return nil
	}
	defer bootstrap.close() // Covers fixed wrapper/scope bookkeeping below.
	scope, _ := owner.scope.category(streamFrameRequestLimitBytes)
	return &frameBudget{limit: streamFrameRequestLimitBytes, scope: scope}
}

// reserve records n bytes against the budget. It reports false when the
// reservation would exceed the limit, and then records nothing.
func (b *frameBudget) reserve(n int64) bool {
	if n <= 0 {
		return true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	for i := range b.leases {
		if b.leases[i].pending && b.leases[i].bytes == n {
			b.leases[i].pending = false
			return true
		}
	}
	if n > b.limit-b.used {
		return false
	}
	slot := -1
	for i := range b.leases {
		if b.leases[i].lease == nil {
			slot = i
			break
		}
	}
	if slot < 0 {
		return false
	}
	lease, err := b.acquireLocked(context.Background(), n, false)
	if err != nil {
		return false
	}
	b.leases[slot].bytes, b.leases[slot].lease = n, lease
	b.leases[slot].pending = false
	b.used += n
	return true
}

func (b *frameBudget) release(n int64) {
	if n <= 0 {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	for i, record := range b.leases {
		if record.lease != nil && !record.pending && record.bytes == n {
			b.leases[i].lease = nil
			b.leases[i].bytes = 0
			b.used -= n
			record.lease.close()
			b.releaseAllowanceLocked()
			return
		}
	}
	panic("stream frame release without matching storage")
}

// Preparation transfers individual grants to the response reader. Unclaimed
// grants are released by the dispatch owner; claimed grants by the reader.
// They use the same frame category and lease table as later frame growth.
func (b *frameBudget) prepare(ctx context.Context, n int64) error {
	if n <= 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if n > b.limit-b.used {
		return &bufferLimitError{}
	}
	for i := range b.leases {
		if b.leases[i].lease != nil {
			continue
		}
		lease, err := b.acquireLocked(ctx, n, true)
		if err != nil {
			return err
		}
		b.leases[i].bytes, b.leases[i].lease, b.leases[i].pending = n, lease, true
		b.used += n
		return nil
	}
	return &bufferBusyError{}
}

func (b *frameBudget) releasePreparation() {
	b.mu.Lock()
	defer b.mu.Unlock()
	for i, record := range b.leases {
		if record.pending {
			b.used -= record.bytes
			record.lease.close()
			b.leases[i].lease = nil
			b.leases[i].bytes = 0
			b.leases[i].pending = false
		}
	}
	b.releaseAllowanceLocked()
}

// inUse is the live reservation, which tests use to prove a closed reader gave
// its buffer back.
func (b *frameBudget) inUse() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.used
}

type streamProtocol int

const (
	streamProtocolUnknown streamProtocol = iota
	streamProtocolSSE
	streamProtocolNDJSON
)

// streamOutcome is the explicit result of reading one stream. Only
// streamOutcomeStreaming is not terminal.
type streamOutcome int

const (
	streamOutcomeStreaming streamOutcome = iota
	streamOutcomeCompleted
	streamOutcomeInterrupted
	streamOutcomeUpstreamError
	streamOutcomeCancelled
)

func (o streamOutcome) String() string {
	switch o {
	case streamOutcomeCompleted:
		return "completed"
	case streamOutcomeInterrupted:
		return "interrupted"
	case streamOutcomeUpstreamError:
		return "upstream_error"
	case streamOutcomeCancelled:
		return "cancelled"
	default:
		return "streaming"
	}
}

// Bounded operational reasons. They are the only text that reaches client
// error frames and workload metadata; upstream error bodies never do.
const (
	streamInterruptReasonPrematureEOF  = "upstream closed before the stream completed"
	streamInterruptReasonReadFailed    = "upstream read failed"
	streamInterruptReasonIdle          = "upstream stopped sending events"
	streamInterruptReasonFrameTooLarge = "upstream frame exceeded the frame limit"
	streamInterruptReasonBudget        = "stream frame buffer exhausted"
	streamInterruptReasonMalformed     = "malformed upstream stream"
	streamInterruptReasonUpstreamError = "upstream reported an error"
	streamInterruptReasonPeerLink      = "peer response interrupted"
	streamInterruptReasonNodePaused    = "node paused"
)

var streamInterruptionReasons = []string{
	streamInterruptReasonPrematureEOF,
	streamInterruptReasonReadFailed,
	streamInterruptReasonIdle,
	streamInterruptReasonFrameTooLarge,
	streamInterruptReasonBudget,
	streamInterruptReasonMalformed,
	streamInterruptReasonUpstreamError,
	streamInterruptReasonPeerLink,
	streamInterruptReasonNodePaused,
}

// streamInterruptionFrames holds the protocol-correct error terminator for each
// bounded reason. The slices are shared and read-only, so terminating a stream
// never has to reserve memory after the budget has already been exhausted.
type streamInterruptionFrames struct {
	sse               []byte
	ndjson            []byte
	ndjsonAfterObject []byte
}

var streamInterruptionFramesByReason = buildStreamInterruptionFrames()

func buildStreamInterruptionFrames() map[string]streamInterruptionFrames {
	frames := make(map[string]streamInterruptionFrames, len(streamInterruptionReasons))
	for _, reason := range streamInterruptionReasons {
		message := "stream interrupted: " + reason
		// Initialization-owned immutable templates use no encoding/json pool.
		payload := appendJSONString([]byte(`{"error":{"message":`), message)
		payload = append(payload, []byte(`,"type":"server_error","code":"stream_interrupted"}}`)...)
		ndjsonPayload := appendJSONString([]byte(`{"error":`), message)
		ndjsonPayload = append(ndjsonPayload, '}')

		frames[reason] = streamInterruptionFrames{
			sse:               append(append([]byte("data: "), payload...), []byte("\n\ndata: [DONE]\n\n")...),
			ndjson:            append(ndjsonPayload, '\n'),
			ndjsonAfterObject: append(append([]byte{'\n'}, ndjsonPayload...), '\n'),
		}
	}
	return frames
}

// streamFrameResult is the per-stream outcome. Reason is a bounded operational
// message and is empty for a completed or cancelled stream.
type streamFrameResult struct {
	Outcome streamOutcome
	Reason  string
	Numeric performanceExtractMetadata
}

type streamFrameConfig struct {
	Protocol    streamProtocol
	Upstream    io.ReadCloser
	Parent      context.Context
	IdleTimeout time.Duration
	Budget      *frameBudget
	// Choices is the number of choices the request asked for. A finish_reason
	// only completes the stream when every one of them has finished.
	Choices int
	// Extract consumes only validated upstream objects, synchronously, without
	// retaining bytes or doing IO. Completion remains this reader's decision.
	Extract func([]byte) performanceExtractMetadata
	// PeerNormalized is set only after validating pinned ingress metadata.
	// Preserve that reader's bounded interruption reason and full terminator.
	PeerNormalized bool
	// PeerTrailer is read only after transport EOF. A header advertisement
	// cannot establish that the destination's terminal bytes survived the hop.
	PeerTrailer func() []string
	Sink        *responseSink
}

type streamFrameReader struct {
	protocol    streamProtocol
	upstream    io.ReadCloser
	idleTimeout time.Duration
	budget      *frameBudget
	scratch     []byte

	parent        context.Context
	deadline      time.Time
	stop          chan struct{}
	stopped       chan struct{}
	closeOnce     sync.Once
	closeErr      error
	readMu        sync.Mutex // exactly one upstream read, also joined by Close
	idleFired     bool
	choiceBytes   int64
	scratchBytes  int64
	metadataBytes int64
	parserBytes   int64
	payload       []byte
	bootstrap     *bufferLease

	mu              sync.Mutex
	buf             []byte
	reserved        int64
	readyEnd        int
	scanOff         int
	lineStart       int
	cleanEOF        bool
	deliverOff      int
	terminator      []byte
	termOff         int
	finishedChoices []bool
	allChoicesDone  bool
	outcome         streamOutcome
	reason          string
	finished        bool
	closed          bool
	extract         func([]byte) performanceExtractMetadata
	numeric         performanceNumbers
	peerNormalized  bool
	peerTrailer     func() []string
	sink            *responseSink
	pendingTerminal bool
	pendingBytes    int
	pendingOutcome  streamOutcome
	pendingReason   string
	pendingExpected []byte
	peerEnding      bool
}

func newStreamFrameReader(config streamFrameConfig) (*streamFrameReader, error) {
	if config.Upstream == nil {
		return nil, errors.New("stream frame reader: an upstream body is required")
	}
	if config.Protocol != streamProtocolSSE && config.Protocol != streamProtocolNDJSON {
		return nil, errStreamFrameProtocol
	}
	idleTimeout := config.IdleTimeout
	if idleTimeout <= 0 {
		idleTimeout = defaultStreamFrameIdle
	}
	budget := config.Budget
	var setup *bufferLease
	defer func() {
		if setup != nil {
			setup.close()
		}
	}()
	if budget == nil {
		if config.Parent == nil || inferenceBuffers(config.Parent) == nil {
			return nil, errStreamFrameAccount
		}
		setup = inferenceBuffers(config.Parent).borrowParserAllowance()
		if setup == nil {
			return nil, errStreamFrameBudget
		}
		budget = ownedFrameBudget(config.Parent)
		if budget == nil {
			return nil, errStreamFrameBudget
		}
	}
	choices := config.Choices
	if choices <= 0 {
		choices = 1
	}
	parent := config.Parent
	if parent == nil {
		parent = context.Background()
	}
	numeric := config.Extract != nil
	if preparation, ok := parent.Value(performancePreparationKey{}).(performancePreparation); ok {
		numeric = preparation.mode == performancePrepareLocalNumeric
		if !numeric {
			config.Extract = nil
		}
	}

	metadata := budget.reserve(streamFrameMetadataBytes)
	var bootstrap *bufferLease
	if !metadata {
		bootstrap, setup = setup, nil
		if owner := inferenceBuffers(parent); bootstrap == nil && owner != nil {
			bootstrap = owner.borrowParserAllowance()
		}
		if bootstrap == nil {
			// No preadmitted control grant: refuse before allocating a reader,
			// channels, timer or watcher. Account presence alone grants nothing.
			return nil, errStreamFrameBudget
		}
	}
	// A production request retains its 96 KiB parser allowance through the
	// joined transport. If the frame scope or inference root is saturated, that
	// preadmitted envelope owns this fixed <=8 KiB interruption-only reader.
	reader := &streamFrameReader{
		protocol:       config.Protocol,
		upstream:       config.Upstream,
		idleTimeout:    idleTimeout,
		budget:         budget,
		parent:         parent,
		deadline:       time.Now().Add(idleTimeout),
		stop:           make(chan struct{}),
		stopped:        make(chan struct{}),
		extract:        config.Extract,
		peerNormalized: config.PeerNormalized,
		peerTrailer:    config.PeerTrailer,
		sink:           config.Sink,
		bootstrap:      bootstrap,
	}
	if metadata {
		reader.metadataBytes = streamFrameMetadataBytes
	}
	if metadata && budget.reserve(int64(choices)) {
		reader.choiceBytes = int64(choices)
		if budget.reserve(streamFrameReadChunk) {
			reader.scratchBytes = streamFrameReadChunk
			reader.scratch = make([]byte, streamFrameReadChunk)
			reader.finishedChoices = make([]bool, choices)
		}
	}
	// Admit reusable working storage before reading upstream. Reusing these
	// allocations and growing within the admitted envelope need no new root
	// grant after a live capacity decrease. Both arrays are charged while copying.
	parserBytes := int64(streamFrameSyntaxBytes)
	if numeric {
		parserBytes = streamFrameNumericBytes
	}
	ready := reader.scratchBytes != 0 && budget.reserve(parserBytes)
	if ready {
		reader.parserBytes = parserBytes
		ready = reader.growLocked(streamFrameReadChunk)
	}
	if ready && config.Protocol == streamProtocolSSE {
		ready = reader.growPayloadLocked(streamFrameReadChunk)
	}
	if !ready {
		reader.finishInterruptedLocked(streamInterruptReasonBudget)
	}
	if reader.finished {
		close(reader.stopped)
	} else {
		go reader.watch()
	}
	return reader, nil
}

// Result reports the stream's outcome. While the stream is still streaming the
// outcome is streamOutcomeStreaming.
func (r *streamFrameReader) Result() streamFrameResult {
	r.mu.Lock()
	defer r.mu.Unlock()
	return streamFrameResult{Outcome: r.outcome, Reason: r.reason, Numeric: r.numeric.snapshot()}
}

func (r *streamFrameReader) Read(p []byte) (int, error) {
	r.readMu.Lock()
	defer r.readMu.Unlock()
	if len(p) == 0 {
		return 0, nil
	}
	for {
		r.mu.Lock()
		if r.parent.Err() != nil && (!r.finished || r.deliverOff < r.readyEnd || r.termOff < len(r.terminator)) {
			r.releaseBufferLocked()
			r.terminator = nil
			r.outcome, r.reason = streamOutcomeCancelled, ""
			r.finished = false
			r.finishLocked(streamOutcomeCancelled, "", nil)
		}
		if !r.finished && r.sink.executionWithdrawn() {
			r.finishInterruptedLocked(streamInterruptReasonNodePaused)
		}
		if r.closed {
			r.mu.Unlock()
			return 0, io.EOF
		}
		if r.deliverOff < r.readyEnd {
			n := copy(p, r.buf[r.deliverOff:r.readyEnd])
			r.deliverOff += n
			r.sink.output(n, !r.peerEnding)
			if r.deliverOff == r.readyEnd {
				r.recycleLocked()
			}
			r.mu.Unlock()
			return n, nil
		}
		if r.termOff < len(r.terminator) {
			n := copy(p, r.terminator[r.termOff:])
			r.termOff += n
			r.sink.output(n, false)
			r.mu.Unlock()
			return n, nil
		}
		if r.finished {
			r.sink.streamResult(streamFrameResult{Outcome: r.outcome, Reason: r.reason})
			r.mu.Unlock()
			<-r.stopped
			return 0, io.EOF
		}
		r.mu.Unlock()

		r.readOnce()
	}
}

// Close stops the timers and the body reader, and gives the frame buffer back.
// A reader that has no terminal outcome yet ends cancelled: its client or
// transport is gone.
func (r *streamFrameReader) Close() error {
	r.mu.Lock()
	if !r.closed {
		r.closed = true
		if !r.finished {
			r.outcome = streamOutcomeCancelled
			r.reason = ""
			r.finished = true
		}
		r.terminator = nil
		r.termOff = 0
		r.stopWatchersLocked()
		r.releaseBufferLocked()
	}
	r.mu.Unlock()
	r.closeUpstream()
	<-r.stopped
	r.readMu.Lock()
	r.mu.Lock()
	r.releaseReadStorageLocked()
	r.budget.release(r.metadataBytes)
	r.metadataBytes = 0
	if r.bootstrap != nil {
		r.bootstrap.close()
		r.bootstrap = nil
	}
	r.mu.Unlock()
	r.readMu.Unlock()
	return r.closeErr
}

func (r *streamFrameReader) closeUpstream() {
	r.closeOnce.Do(func() { r.closeErr = r.upstream.Close() })
}

// One watcher closes the body to unblock the synchronous upstream read. It
// never cancels the parent; terminal EOF and Close join it before returning.
func (r *streamFrameReader) watch() {
	defer close(r.stopped)
	timer := time.NewTimer(r.idleTimeout)
	defer timer.Stop()
	for {
		select {
		case <-r.stop:
			return
		case <-r.parent.Done():
			r.closeUpstream()
			return
		case <-timer.C:
			r.mu.Lock()
			remaining := time.Until(r.deadline)
			if r.finished || r.closed {
				r.mu.Unlock()
				return
			}
			if remaining > 0 {
				timer.Reset(remaining)
				r.mu.Unlock()
				continue
			}
			r.idleFired = true
			r.mu.Unlock()
			r.closeUpstream()
			return
		}
	}
}

func (r *streamFrameReader) readOnce() {
	r.mu.Lock()
	if r.closed || r.finished {
		r.mu.Unlock()
		return
	}
	available := int(streamFrameMaxBytes) - len(r.buf)
	readSize := len(r.scratch)
	if available < readSize {
		readSize = available
	}
	if readSize == 0 {
		readSize = 1 // distinguish an exact-limit EOF object from overflow
	}
	scratch := r.scratch[:readSize]
	r.mu.Unlock()
	n, err := r.upstream.Read(scratch)

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || r.finished {
		return
	}
	if r.parent.Err() != nil {
		r.finishLocked(streamOutcomeCancelled, "", nil)
		return
	}
	if r.sink.executionWithdrawn() {
		r.finishInterruptedLocked(streamInterruptReasonNodePaused)
		return
	}
	if r.idleFired {
		r.finishInterruptedLocked(streamInterruptReasonIdle)
		return
	}
	r.cleanEOF = errors.Is(err, io.EOF)
	if n > 0 {
		if len(r.buf)+n > int(streamFrameMaxBytes) {
			r.finishInterruptedLocked(streamInterruptReasonFrameTooLarge)
			return
		}
		if !r.appendLocked(scratch[:n]) {
			r.finishInterruptedLocked(streamInterruptReasonBudget)
			return
		}
		r.parseLocked()
		if r.finished {
			return
		}
	}
	if err != nil {
		if r.cleanEOF {
			r.parseLocked()
			if r.finished {
				return
			}
		}
		r.handleReadErrorLocked(err)
	}
}

func (r *streamFrameReader) handleReadErrorLocked(err error) {
	// A close this reader caused says more than the error the body returns.
	if r.parent.Err() != nil {
		r.finishLocked(streamOutcomeCancelled, "", nil)
		return
	}
	if r.idleFired {
		r.finishInterruptedLocked(streamInterruptReasonIdle)
		return
	}
	if errors.Is(err, io.EOF) {
		r.handleCleanEOFLocked()
		return
	}
	if r.peerTrailer != nil {
		r.finishInterruptedLocked(streamInterruptReasonPeerLink)
		return
	}
	r.finishInterruptedLocked(streamInterruptReasonReadFailed)
}

// handleCleanEOFLocked decides what an EOF without a protocol end marker means.
// A final NDJSON object without a newline is a complete frame; a final SSE event
// still needs its blank line. Anything else is a torn tail, and a finish_reason
// only completes the stream when every requested choice finished and the body
// ended cleanly.
func (r *streamFrameReader) handleCleanEOFLocked() {
	if r.pendingTerminal {
		if !r.validatePendingPeerLocked() {
			return
		}
		metadata := originStreamRequest{peerNormalized: true}
		reason, err := metadata.withPeerTrailer(r.peerTrailer())
		if err != nil {
			var invalid *peerFramingMetadataError
			if errors.As(err, &invalid) && invalid.invalid {
				r.finishInterruptedLocked(streamInterruptReasonMalformed)
			} else {
				r.finishInterruptedLocked(streamInterruptReasonPeerLink)
			}
			return
		}
		if reason != r.pendingReason {
			r.finishInterruptedLocked(streamInterruptReasonMalformed)
			return
		}
		r.readyEnd += r.pendingBytes
		r.peerEnding = r.pendingOutcome == streamOutcomeInterrupted
		if r.peerEnding {
			r.sink.planInterruption(reason)
		}
		r.finishLocked(r.pendingOutcome, reason, nil)
		return
	}
	pending := r.buf[r.readyEnd:]
	if len(pending) > 0 && r.protocol == streamProtocolNDJSON {
		verdict := r.assessNDJSONFrameLocked(pending)
		if verdict == frameBudgetExhausted {
			r.finishInterruptedLocked(streamInterruptReasonBudget)
			return
		}
		if verdict == frameMalformed {
			r.finishInterruptedLocked(streamInterruptReasonPrematureEOF)
			return
		}
		if r.peerTrailer != nil && (verdict == frameCompleted || verdict == frameUpstreamError) {
			outcome, reason := streamOutcomeCompleted, ""
			if verdict == frameUpstreamError {
				outcome, reason = streamOutcomeUpstreamError, streamInterruptReasonUpstreamError
			}
			r.holdPeerTerminalLocked(len(pending), outcome, reason, nil)
			r.handleCleanEOFLocked()
			return
		}
		r.readyEnd += len(pending)
		switch verdict {
		case frameCompleted:
			r.finishLocked(streamOutcomeCompleted, "", nil)
			return
		case frameUpstreamError:
			r.finishLocked(streamOutcomeUpstreamError, streamInterruptReasonUpstreamError, nil)
			return
		}
		// Preserve the object's bytes and add its missing record separator
		// before appending a synthetic error object.
		r.finishInterruptedLocked(streamInterruptReasonPrematureEOF)
		r.terminator = streamInterruptionFramesByReason[streamInterruptReasonPrematureEOF].ndjsonAfterObject
		return
	}
	if len(r.buf) > r.readyEnd {
		reason := streamInterruptReasonPrematureEOF
		if r.peerTrailer != nil {
			reason = streamInterruptReasonPeerLink
		}
		r.finishInterruptedLocked(reason)
		return
	}
	if r.allChoicesDone {
		if r.peerTrailer != nil {
			r.holdPeerTerminalLocked(0, streamOutcomeCompleted, "", nil)
			r.handleCleanEOFLocked()
			return
		}
		r.finishLocked(streamOutcomeCompleted, "", nil)
		return
	}
	reason := streamInterruptReasonPrematureEOF
	if r.peerTrailer != nil {
		reason = streamInterruptReasonPeerLink
	}
	r.finishInterruptedLocked(reason)
}

type frameVerdict int

const (
	frameContinue frameVerdict = iota
	frameCompleted
	frameUpstreamError
	frameMalformed
	frameBudgetExhausted
)

func (r *streamFrameReader) parseLocked() {
	for !r.finished {
		if r.pendingTerminal {
			r.validatePendingPeerLocked()
			return
		}
		pending := r.buf[r.readyEnd:]
		if len(pending) == 0 {
			return
		}
		n, complete := r.nextFrameLocked(pending)
		if !complete {
			if int64(len(pending)) > streamFrameMaxBytes {
				r.finishInterruptedLocked(streamInterruptReasonFrameTooLarge)
			}
			return
		}
		if int64(n) > streamFrameMaxBytes {
			r.finishInterruptedLocked(streamInterruptReasonFrameTooLarge)
			return
		}
		frame := pending[:n]
		if r.peerNormalized {
			for reason, terminator := range streamInterruptionFramesByReason {
				canonical := terminator.ndjson
				if r.protocol == streamProtocolSSE {
					canonical = bytes.TrimSuffix(terminator.sse, []byte("data: [DONE]\n\n"))
				}
				if bytes.Equal(frame, canonical) {
					if r.peerTrailer != nil {
						expected := terminator.ndjson
						if r.protocol == streamProtocolSSE {
							expected = terminator.sse
						}
						r.holdPeerTerminalLocked(len(expected), streamOutcomeInterrupted, reason, expected)
						return
					}
					// Replace the peer's exact canonical record with the same
					// full terminator, without re-observing synthetic metadata.
					r.finishInterruptedLocked(reason)
					return
				}
			}
		}
		var verdict frameVerdict
		switch r.protocol {
		case streamProtocolSSE:
			verdict = r.assessSSEFrameLocked(frame)
		case streamProtocolNDJSON:
			verdict = r.assessNDJSONFrameLocked(frame)
		default:
			verdict = frameMalformed
		}
		switch verdict {
		case frameBudgetExhausted:
			r.finishInterruptedLocked(streamInterruptReasonBudget)
			return
		case frameMalformed:
			r.finishInterruptedLocked(streamInterruptReasonMalformed)
			return
		case frameUpstreamError:
			if r.peerTrailer != nil {
				r.holdPeerTerminalLocked(n, streamOutcomeUpstreamError, streamInterruptReasonUpstreamError, nil)
				return
			}
			r.readyEnd += n
			r.finishLocked(streamOutcomeUpstreamError, streamInterruptReasonUpstreamError, nil)
			return
		case frameCompleted:
			if r.peerTrailer != nil {
				r.holdPeerTerminalLocked(n, streamOutcomeCompleted, "", nil)
				return
			}
			r.readyEnd += n
			r.finishLocked(streamOutcomeCompleted, "", nil)
			return
		default:
			r.readyEnd += n
			r.refreshIdleLocked()
		}
	}
}

func (r *streamFrameReader) holdPeerTerminalLocked(n int, outcome streamOutcome, reason string, expected []byte) {
	r.pendingTerminal, r.pendingBytes = true, n
	r.pendingOutcome, r.pendingReason, r.pendingExpected = outcome, reason, expected
	r.validatePendingPeerLocked()
}

func (r *streamFrameReader) validatePendingPeerLocked() bool {
	pending := r.buf[r.readyEnd:]
	if len(pending) > r.pendingBytes {
		r.finishInterruptedLocked(streamInterruptReasonMalformed)
		return false
	}
	if r.pendingExpected != nil && !bytes.Equal(pending, r.pendingExpected[:len(pending)]) {
		r.finishInterruptedLocked(streamInterruptReasonMalformed)
		return false
	}
	if r.cleanEOF && len(pending) < r.pendingBytes {
		r.finishInterruptedLocked(streamInterruptReasonPeerLink)
		return false
	}
	return true
}

func (r *streamFrameReader) nextFrameLocked(pending []byte) (int, bool) {
	if r.protocol == streamProtocolNDJSON {
		if i := bytes.IndexByte(pending[r.scanOff:], '\n'); i >= 0 {
			end := r.scanOff + i + 1
			r.scanOff = 0
			return end, true
		}
		r.scanOff = len(pending)
		return 0, false
	}
	// Scan each byte once, retaining only offsets across arbitrary read splits.
	for r.scanOff < len(pending) {
		i := r.scanOff
		if pending[i] != '\n' && pending[i] != '\r' {
			r.scanOff++
			continue
		}
		if pending[i] == '\r' && i+1 == len(pending) && !r.cleanEOF {
			return 0, false
		}
		width := 1
		if pending[i] == '\r' && i+1 < len(pending) && pending[i+1] == '\n' {
			width = 2
		}
		end := i + width
		if i == r.lineStart {
			r.scanOff, r.lineStart = 0, 0
			return end, true
		}
		r.scanOff, r.lineStart = end, end
	}
	return 0, false
}

func (r *streamFrameReader) assessSSEFrameLocked(frame []byte) frameVerdict {
	if !utf8.Valid(frame) {
		return frameMalformed
	}
	var payload []byte
	dataBytes, dataLines := 0, 0
	forEachSSELine(frame, func(line []byte) {
		if len(line) == 0 || line[0] == ':' {
			return
		}
		field, value := splitSSEField(line)
		if bytes.Equal(field, []byte("data")) {
			payload = value
			dataBytes += len(value)
			dataLines++
		}
	})
	if dataLines == 0 {
		return frameContinue
	}
	if dataLines > 1 {
		// The joined payload has its own reusable allocation: raw bytes and
		// parser scratch remain charged while it is constructed and validated.
		size := dataBytes + dataLines - 1
		if !r.growPayloadLocked(size) {
			return frameBudgetExhausted
		}
		payload = r.payload[:0]
		dataLines = 0
		forEachSSELine(frame, func(line []byte) {
			field, value := splitSSEField(line)
			if bytes.Equal(field, []byte("data")) {
				if dataLines > 0 {
					payload = append(payload, '\n')
				}
				payload = append(payload, value...)
				dataLines++
			}
		})
	}
	if bytes.Equal(bytes.TrimSpace(payload), []byte("[DONE]")) {
		return frameCompleted
	}
	if !validStreamObject(payload) {
		return frameMalformed
	}
	if r.extract != nil {
		r.numeric.merge(r.extract(payload))
	}
	if value := streamJSONField(payload, "error"); len(value) > 0 && !bytes.Equal(value, []byte("null")) {
		return frameUpstreamError
	}
	choices := streamJSONField(payload, "choices")
	if len(choices) == 0 || bytes.Equal(choices, []byte("null")) {
		return frameContinue // e.g. a usage-only chunk
	}
	if choices[0] != '[' {
		return frameMalformed
	}
	for rest := bytes.TrimSpace(choices[1:]); len(rest) > 0 && rest[0] != ']'; {
		end := streamJSONValueEnd(rest)
		choice := rest[:end]
		if choice[0] != '{' {
			return frameMalformed
		}
		finish := streamJSONField(choice, "finish_reason")
		if len(finish) > 0 && !bytes.Equal(finish, []byte("null")) {
			var index int
			rawIndex := streamJSONField(choice, "index")
			if len(rawIndex) > 20 {
				return frameMalformed
			}
			var err error
			index, err = strconv.Atoi(string(rawIndex))
			if len(finish) <= 2 || finish[0] != '"' || bytes.Equal(rawIndex, []byte("null")) || err != nil || index < 0 || index >= len(r.finishedChoices) {
				return frameMalformed
			}
			r.finishedChoices[index] = true
		}
		rest = bytes.TrimSpace(rest[end:])
		if len(rest) > 0 && rest[0] == ',' {
			rest = bytes.TrimSpace(rest[1:])
		}
	}
	r.allChoicesDone = true
	for _, finished := range r.finishedChoices {
		if !finished {
			r.allChoicesDone = false
			break
		}
	}
	return frameContinue
}

// These slices refer to already validated JSON. Scan over uninterpreted values
// without allocating copies of content, errors, or an unbounded choices list.
func validStreamObject(payload []byte) bool {
	payload = bytes.TrimSpace(payload)
	return len(payload) > 0 && payload[0] == '{' && ownedJSONValid(payload)
}

func streamJSONValueEnd(b []byte) int {
	depth := 0
	quoted, escaped := false, false
	for i, c := range b {
		if quoted {
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == '"' {
				quoted = false
			}
			continue
		}
		switch c {
		case '"':
			quoted = true
		case '{', '[':
			depth++
		case '}', ']':
			if depth == 0 {
				return i
			}
			depth--
		case ',':
			if depth == 0 {
				return i
			}
		}
	}
	return len(b)
}

func streamJSONField(object []byte, name string) []byte {
	object = bytes.TrimSpace(object)
	var result []byte
	for rest := bytes.TrimSpace(object[1:]); len(rest) > 0 && rest[0] != '}'; {
		// Find the end of the key string, respecting escapes.
		end := 1
		for end < len(rest) {
			if rest[end] == '\\' {
				end += 2
				continue
			}
			if rest[end] == '"' {
				break
			}
			end++
		}
		key := rest[:end+1]
		rest = bytes.TrimSpace(rest[end+1:])
		rest = bytes.TrimSpace(rest[1:]) // colon
		end = streamJSONValueEnd(rest)
		matches := jsonStringEquals(key, name)
		if matches {
			result = bytes.TrimSpace(rest[:end])
		}
		rest = bytes.TrimSpace(rest[end:])
		if len(rest) > 0 && rest[0] == ',' {
			rest = bytes.TrimSpace(rest[1:])
		}
	}
	return result
}

func (r *streamFrameReader) assessNDJSONFrameLocked(frame []byte) frameVerdict {
	if !utf8.Valid(frame) {
		return frameMalformed
	}
	line := bytes.TrimSuffix(frame, []byte("\n"))
	line = bytes.TrimSuffix(line, []byte("\r"))
	if len(bytes.TrimSpace(line)) == 0 {
		return frameMalformed
	}
	if !validStreamObject(line) {
		return frameMalformed
	}
	if r.extract != nil {
		r.numeric.merge(r.extract(line))
	}
	if value := streamJSONField(line, "error"); len(value) > 0 && !bytes.Equal(value, []byte("null")) {
		return frameUpstreamError
	}
	done := streamJSONField(line, "done")
	if bytes.Equal(done, []byte("true")) {
		return frameCompleted
	}
	if len(done) > 0 && !bytes.Equal(done, []byte("false")) && !bytes.Equal(done, []byte("null")) {
		return frameMalformed
	}
	return frameContinue
}

// forEachSSELine visits complete event lines without copying their bytes.
func forEachSSELine(frame []byte, visit func([]byte)) {
	for i := 0; i < len(frame); {
		j := i
		for j < len(frame) && frame[j] != '\n' && frame[j] != '\r' {
			j++
		}
		visit(frame[i:j])
		if j == len(frame) {
			break
		}
		if frame[j] == '\r' && j+1 < len(frame) && frame[j+1] == '\n' {
			j++
		}
		i = j + 1
	}
}

// splitSSEField parses one SSE field line, dropping the single optional space
// after the colon.
func splitSSEField(line []byte) (field, value []byte) {
	idx := bytes.IndexByte(line, ':')
	if idx < 0 {
		return line, nil
	}
	field = line[:idx]
	value = line[idx+1:]
	if len(value) > 0 && value[0] == ' ' {
		value = value[1:]
	}
	return field, value
}

func (r *streamFrameReader) finishInterruptedLocked(reason string) {
	frames, ok := streamInterruptionFramesByReason[reason]
	if !ok {
		frames = streamInterruptionFramesByReason[streamInterruptReasonReadFailed]
		reason = streamInterruptReasonReadFailed
	}
	r.sink.planInterruption(reason)
	terminator := frames.sse
	if r.protocol == streamProtocolNDJSON {
		terminator = frames.ndjson
	}
	r.finishLocked(streamOutcomeInterrupted, reason, terminator)
}

func (r *streamFrameReader) finishLocked(outcome streamOutcome, reason string, terminator []byte) {
	if r.finished {
		return
	}
	r.outcome = outcome
	r.reason = reason
	r.finished = true
	r.terminator = terminator
	r.termOff = 0
	// Whatever follows a terminal marker or a torn tail is not part of the
	// stream this client receives.
	r.dropTailLocked()
	if r.readyEnd == 0 {
		r.releaseBufferLocked()
	}
	r.stopWatchersLocked()
	r.closeUpstream()
	r.releaseReadStorageLocked()
}

func (r *streamFrameReader) releaseReadStorageLocked() {
	r.budget.release(r.choiceBytes)
	r.budget.release(r.scratchBytes)
	r.budget.release(r.parserBytes)
	r.budget.release(int64(cap(r.payload)))
	r.choiceBytes, r.scratchBytes = 0, 0
	r.parserBytes = 0
	r.payload = nil
	r.finishedChoices = nil
	r.scratch = nil
}

func (r *streamFrameReader) dropTailLocked() {
	r.buf = r.buf[:r.readyEnd]
}

func (r *streamFrameReader) appendLocked(data []byte) bool {
	if len(data) == 0 {
		return true
	}
	if !r.growLocked(len(r.buf) + len(data)) {
		return false
	}
	r.buf = append(r.buf, data...)
	return true
}

func (r *streamFrameReader) growLocked(need int) bool {
	if need <= cap(r.buf) {
		return true
	}
	target := cap(r.buf) * 2
	if target < need {
		target = need
	}
	if target < streamFrameReadChunk {
		target = streamFrameReadChunk
	}
	if target > int(streamFrameMaxBytes) {
		target = int(streamFrameMaxBytes)
	}
	// Both backing arrays are live during copy. Reserve the entire new
	// allocation before making it, then relinquish the replaced allocation.
	if !r.budget.reserve(int64(target)) {
		return false
	}
	next := make([]byte, len(r.buf), target)
	copy(next, r.buf)
	r.buf = next
	r.budget.release(r.reserved)
	r.reserved = int64(target)
	return true
}

func (r *streamFrameReader) growPayloadLocked(need int) bool {
	if need <= cap(r.payload) {
		return true
	}
	if !r.budget.reserve(int64(need)) {
		return false
	}
	next := make([]byte, 0, need)
	r.budget.release(int64(cap(r.payload)))
	r.payload = next
	return true
}

// recycleLocked compacts the incomplete tail after the complete frames have
// been delivered. Empty live streams retain their admitted capacity for the
// next frame; no response prefix is retained in the logical buffer.
func (r *streamFrameReader) recycleLocked() {
	if r.deliverOff < r.readyEnd {
		return
	}
	if r.readyEnd < len(r.buf) {
		if r.deliverOff == 0 {
			return
		}
		tail := copy(r.buf, r.buf[r.readyEnd:])
		r.buf = r.buf[:tail]
		r.readyEnd, r.deliverOff = 0, 0
		return
	}
	if r.finished {
		r.releaseBufferLocked()
	} else {
		r.buf = r.buf[:0]
		r.readyEnd, r.deliverOff = 0, 0
	}
}

func (r *streamFrameReader) releaseBufferLocked() {
	if r.reserved > 0 {
		r.budget.release(r.reserved)
		r.reserved = 0
	}
	r.buf = nil
	r.readyEnd, r.deliverOff = 0, 0
}

func (r *streamFrameReader) refreshIdleLocked() {
	r.deadline = time.Now().Add(r.idleTimeout)
}

func (r *streamFrameReader) stopWatchersLocked() {
	select {
	case <-r.stop:
	default:
		close(r.stop)
	}
}
