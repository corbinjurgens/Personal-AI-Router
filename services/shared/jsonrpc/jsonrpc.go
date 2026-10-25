// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package jsonrpc is the shared newline-delimited JSON-RPC 2.0 codec used by
// every NVPAIR subprocess over stdio (or an --ipc socket/pipe).
//
// It is a drop-in extraction of the per-service codec that was copy-pasted
// across ~10 binaries: the wire types (Message, RPCError), the request/
// notification/response classifiers, and the Codec itself (newline-framed
// read + concurrency-safe, short-write-safe write). A worker that only sends
// notifications and responds to requests can adopt this with no behavior
// change; the bidirectional peer/correlation layer (a child originating
// id-bearing requests upward) is built separately on top of these types.
package jsonrpc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

// Message is a single JSON-RPC 2.0 frame. A frame is a request (id+method), a
// notification (method, no id), or a response (id, no method) — see the
// IsRequest/IsNotification/IsResponse classifiers.
type Message struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Method  string           `json:"method,omitempty"`
	Params  json.RawMessage  `json:"params,omitempty"`
	Result  json.RawMessage  `json:"result,omitempty"`
	Error   *RPCError        `json:"error,omitempty"`
	// borrowedRead marks projected history storage owned by the current Read.
	borrowedRead bool
}

// RPCError is the JSON-RPC 2.0 error object.
type RPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

// DecodeError marks a recoverable per-frame decode failure: the scanner
// advanced past the bad line, so the next Read can continue. Terminal
// transport/scanner errors are returned as plain errors instead, so a read
// loop stops rather than spinning on a permanently-failing read. Callers that
// want to continue-on-bad-frame can type-assert / errors.As for *DecodeError.
type DecodeError struct{ Err error }

func (e *DecodeError) Error() string { return e.Err.Error() }
func (e *DecodeError) Unwrap() error { return e.Err }

// IsRequest reports whether the frame is an id-bearing request.
func (m *Message) IsRequest() bool {
	return m.ID != nil && m.Method != ""
}

// IsNotification reports whether the frame is a fire-and-forget notification.
func (m *Message) IsNotification() bool {
	return m.ID == nil && m.Method != ""
}

// IsResponse reports whether the frame is a response to a request (id, no
// method). Used by the parent-side demux and any bidirectional peer.
func (m *Message) IsResponse() bool {
	return m.ID != nil && m.Method == ""
}

// WorkerFrameBytes is the inbound frame cap for the hops that carry large
// worker replies: the orchestrator's generic rpcWorker links (engine-manager,
// manual-nodes, node-settings, job-scheduler), engine-manager's own inbound
// codec, and a terminal client's link to the orchestrator.
//
// It is declared here rather than per-package because the cap only works if
// every hop on a path agrees. A reply has to survive the worker's writer, the
// orchestrator's reader, and the client's reader; one hop left at a smaller
// value fails the whole path, and it fails badly — bufio.Scanner cannot resync
// past an over-long line, so the frame surfaces as a terminal read error that
// takes the peer down (see NewCodecMaxFrame).
//
// It is deliberately not repo-wide. The purpose-built links — discovery,
// errors, cluster, the proxies, workload-manager, node-info — keep their own
// smaller caps, sized to what those workers actually send. Before raising one,
// check every hop on that worker's path, not just the one that overflowed.
//
// 8 MiB is far above any real frame. The largest today is engine:catalog's
// Ollama list at roughly 1.9 MiB; the previous 1 MiB default could not carry it.
const WorkerFrameBytes = 8 << 20

// Codec handles newline-delimited JSON-RPC 2.0 over an io.ReadWriter.
// Writes are safe for concurrent use.
type Codec struct {
	scanner        *bufio.Scanner
	writer         io.Writer
	wgate          chan struct{}
	retired        bool // guarded by wgate
	interruptWrite func()
	history        atomic.Pointer[HistoryReader]
}

// ErrReadLeaseRequired prevents an opted-in reader from bypassing explicit
// borrowed-frame completion with Read. Use ReadWithRelease instead.
var ErrReadLeaseRequired = errors.New("jsonrpc read requires explicit release")

// ReadControl belongs to a fixed reader role, never to a Codec generation.
// Bind a replacement only after actual retirement completion. The old Codec
// retains only sharedTerminalReader in its existing history pointer footprint.
// All fields except the immutable reader-to-control link are guarded by mu.
type ReadControl struct {
	mu               sync.Mutex
	codec            *Codec
	reader           *HistoryReader
	epoch            uint64
	read             uint64
	retired          bool
	active           bool
	borrowed         bool
	interrupting     bool
	interrupt        func()
	fence            func()
	onRequest        func(*Message)
	onHistoryFailure func(HistoryFailure)
	done             chan struct{}
}

var sharedTerminalReader = &HistoryReader{}
var sharedReadComplete = func() <-chan struct{} {
	done := make(chan struct{})
	close(done)
	return done
}()

// Available reports actual completion, not transport/process liveness. Owners
// serialize construction separately; this does not reserve the role.
func (s *ReadControl) Available() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.codec == nil
}

// ReadRetirementEnabled is stable after publication, including after detachment.
func (c *Codec) ReadRetirementEnabled() bool {
	r := c.history.Load()
	return r == sharedTerminalReader || (r != nil && r.readControl != nil)
}

// EnableReadRetirement opts a history codec into exclusive read admission and
// explicit borrowed completion. Call once before publishing or reading it.
// interrupt must promptly unblock the ACTUAL reader (e.g. close the retained
// subprocess stdout pipe); a write interrupt or a no-op transport Close is not
// sufficient. It must target this generation, not a mutable replacement handle,
// and must not wait for Read, retirement completion, or a reader/owner lock.
// No I/O or callback runs under the read-state mutex.
//
// The transport owner admits the backing before construction and reuses it only
// after RetireRead completes. This API supplies no banks, role pool, or grants.
func (c *Codec) EnableReadRetirement(control *ReadControl, interrupt func()) error {
	r := c.history.Load()
	if r == nil || r == sharedTerminalReader || r.readControl != nil || control == nil || interrupt == nil {
		return ErrHistoryInvalid
	}
	control.mu.Lock()
	defer control.mu.Unlock()
	if control.codec != nil {
		return ErrHistoryBusy
	}
	if control.epoch == ^uint64(0) {
		return ErrHistoryExhausted
	}
	control.epoch++
	control.read = 0
	control.codec, control.reader = c, r
	control.retired, control.active, control.borrowed, control.interrupting = false, false, false, false
	control.interrupt, control.done = interrupt, make(chan struct{})
	r.readControl = control // before publication; never changed on this reader
	return nil
}

// SetReadRetirementFence installs the source withdrawal hook before publication.
// It runs once, outside the read mutex, before detachment and bank rebind. It
// must not wait for read retirement; an actual read interrupt is separate.
func (c *Codec) SetReadRetirementFence(fence func()) error {
	r := c.history.Load()
	if r == nil || r == sharedTerminalReader || r.readControl == nil {
		return ErrHistoryInvalid
	}
	s := r.readControl
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.codec != c || s.reader != r || s.retired || s.active || s.borrowed || s.fence != nil {
		return ErrHistoryInvalid
	}
	s.fence = fence
	return nil
}

// ReadWithRelease is Read with an explicit synchronous-consumption boundary.
// On an opted-in codec, only one Read or borrowed result may be outstanding;
// overlap returns ErrHistoryBusy without touching input. A non-nil release must
// be called after the last use, including HistoryFailure.Refusal.ReplyID when
// msg is nil. Release is idempotent and cannot release a later read's borrow.
// Owned ordinary messages relinquish reader storage before this call returns.
// Context cancellation or handing a result to another goroutine is not release.
//
// Without EnableReadRetirement this preserves Read's existing lifetime (history
// storage is valid until the next Read) and returns a nil release.
func (c *Codec) ReadWithRelease() (*Message, func(), error) {
	r := c.history.Load()
	if r == sharedTerminalReader {
		return nil, nil, ErrPeerClosed
	}
	if r == nil || r.readControl == nil {
		msg, err := c.readMessage()
		return msg, nil, err
	}
	return c.readHistoryWithRelease(r)
}

func (c *Codec) readHistoryWithRelease(r *HistoryReader) (*Message, func(), error) {
	// A caller can have loaded r before a retirement/rebind. It may not touch
	// bufio, correlation or banks until identity and epoch admission is atomic
	// under the fixed control. Old r's control link is immutable, not authority.
	s := r.readControl
	s.mu.Lock()
	if s.codec != c || s.reader != r || s.retired || c.history.Load() != r {
		s.mu.Unlock()
		return nil, nil, ErrPeerClosed
	}
	if s.active || s.borrowed {
		s.mu.Unlock()
		return nil, nil, ErrHistoryBusy
	}
	if s.read == ^uint64(0) {
		s.mu.Unlock()
		return nil, nil, ErrHistoryExhausted
	}
	s.read++
	epoch, read := s.epoch, s.read
	s.active = true
	s.mu.Unlock()

	msg, err := r.Read()
	var failure *HistoryFailure
	borrowed := (msg != nil && msg.borrowedRead) || errors.As(err, &failure)
	s.mu.Lock()
	// active prevents rebinding, so these must still identify this admission.
	if s.codec != c || s.reader != r || s.epoch != epoch || s.read != read {
		s.mu.Unlock()
		return nil, nil, ErrPeerClosed
	}
	s.active = false
	if s.retired {
		s.finishRetirementLocked()
		s.mu.Unlock()
		return nil, nil, ErrPeerClosed
	}
	s.borrowed = borrowed
	s.mu.Unlock()
	if !borrowed {
		return msg, nil, err
	}
	release := func() {
		// Capture only the fixed role and scalar admission tokens. No reader,
		// Codec, bank or per-generation sync.Once survives a released borrow.
		s.mu.Lock()
		if s.epoch == epoch && s.read == read && s.borrowed {
			s.borrowed = false
			s.finishRetirementLocked()
		}
		s.mu.Unlock()
	}
	return msg, release, err
}

// RetireRead permanently fences future reads, including already buffered frames.
// The returned channel closes only after active Read, its owner interruption,
// and actual borrowed consumption finish, and raw/output/offset/bufio references
// have been detached. It does not join ordinary callbacks, writes, or the pump.
// Duplicate active calls return its completion token; completed old codecs
// return the shared closed sentinel. An unopted codec returns nil: Close
// or an unconfigured interrupt is never evidence of reader-bank release.
// The first retiring caller runs the prompt owner interrupt synchronously;
// returning from that hook alone does not establish active-read completion.
//
// Fence source authority under the owner's lock first, then call this outside
// that lock (including workersMu). The interruption runs outside read-state locks.
func (c *Codec) RetireRead() <-chan struct{} {
	r := c.history.Load()
	if r == sharedTerminalReader {
		return sharedReadComplete
	}
	if r == nil || r.readControl == nil {
		return nil
	}
	s := r.readControl
	s.mu.Lock()
	if s.codec != c || s.reader != r || c.history.Load() != r {
		s.mu.Unlock()
		return sharedReadComplete
	}
	done := s.done
	if s.retired {
		s.mu.Unlock()
		return done
	}
	s.retired = true
	interrupt := s.interrupt
	active, fence := s.active, s.fence
	// This also pins the generation across the unlocked source fence.
	s.interrupting = true
	s.mu.Unlock()
	if fence != nil {
		fence()
	}
	if active {
		interrupt()
	}
	// Drop copied generation hooks before detachment can authorize rebinding.
	fence, interrupt = nil, nil
	s.mu.Lock()
	s.interrupting = false
	s.finishRetirementLocked()
	s.mu.Unlock()
	return done
}

func (s *ReadControl) finishRetirementLocked() {
	if !s.retired || s.active || s.borrowed || s.interrupting || s.codec == nil {
		return
	}
	// Clear the reader itself as well as the codec pointer: no retained shell may
	// keep bufio's buffered transport bytes or any owner-admitted bank alive.
	if r := s.reader; r != nil {
		r.r = nil
		r.raw = nil
		r.output = nil
		r.offsets = nil
		if r.correlation != nil {
			*r.correlation = HistoryCorrelation{}
		}
		r.correlation = nil
		r.schema = nil
		r.ordinary = 0
		clear(r.idToken[:])
	}
	s.codec.history.Store(sharedTerminalReader)
	s.codec.scanner = nil
	s.codec, s.reader = nil, nil
	s.interrupt, s.fence, s.onRequest, s.onHistoryFailure = nil, nil, nil, nil
	close(s.done)
	s.done = nil
}

// NewCodec wraps rw with a Codec. Reads accept frames up to 1 MiB.
func NewCodec(rw io.ReadWriter) *Codec {
	scanner := bufio.NewScanner(rw)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)
	return &Codec{
		scanner: scanner,
		writer:  rw,
		wgate:   make(chan struct{}, 1),
	}
}

// NewCodecMaxFrame is like NewCodec but caps a single inbound frame at
// maxFrame bytes (initial buffer 64 KiB, grown on demand). Used by peers that
// exchange larger frames than the 1 MiB default (e.g. nvpair-engine-manager). An
// inbound frame larger than maxFrame surfaces as a terminal scanner read error
// (bufio.Scanner cannot resync past an over-long line), so the read loop exits
// cleanly rather than spinning.
func NewCodecMaxFrame(rw io.ReadWriter, maxFrame int) *Codec {
	return NewCodecMaxFrameWithWriteInterrupt(rw, maxFrame, nil)
}

// NewCodecMaxFrameWithWriteInterrupt gives the transport owner control over
// active write cancellation and failure. interrupt must promptly and permanently
// unblock Write, without waiting for the writer or acquiring its write gate.
// NotifyContext joins both Write and interrupt before returning. An arbitrary
// io.Writer is not intrinsically cancellable; without a hook, active writes
// remain synchronous and may outlive their context. Queued cancellation and
// marshal errors never invoke interrupt.
func NewCodecMaxFrameWithWriteInterrupt(rw io.ReadWriter, maxFrame int, interrupt func()) *Codec {
	scanner := bufio.NewScanner(rw)
	scanner.Buffer(make([]byte, 0, 64*1024), maxFrame)
	return &Codec{
		scanner:        scanner,
		writer:         rw,
		wgate:          make(chan struct{}, 1),
		interruptWrite: interrupt,
	}
}

// Read returns the next frame, or io.EOF when the stream closes. A malformed
// frame (bad JSON or wrong version) is returned as a recoverable *DecodeError;
// a terminal scanner/transport error is returned as a plain error.
func (c *Codec) Read() (*Message, error) {
	r := c.history.Load()
	if r == sharedTerminalReader {
		return nil, ErrPeerClosed
	}
	if r != nil && r.readControl != nil {
		s := r.readControl
		s.mu.Lock()
		retired := s.retired || s.codec != c || s.reader != r || c.history.Load() != r
		s.mu.Unlock()
		if retired {
			return nil, ErrPeerClosed
		}
		return nil, ErrReadLeaseRequired
	}
	return c.readMessage()
}

func (c *Codec) readMessage() (*Message, error) {
	if r := c.history.Load(); r != nil {
		if r == sharedTerminalReader {
			return nil, ErrPeerClosed
		}
		return r.Read()
	}
	if !c.scanner.Scan() {
		if err := c.scanner.Err(); err != nil {
			return nil, fmt.Errorf("read error: %w", err)
		}
		return nil, io.EOF
	}
	var msg Message
	if err := json.Unmarshal(c.scanner.Bytes(), &msg); err != nil {
		return nil, &DecodeError{fmt.Errorf("invalid JSON-RPC message: %w", err)}
	}
	if msg.JSONRPC != "2.0" {
		return nil, &DecodeError{fmt.Errorf("unsupported JSON-RPC version: %q", msg.JSONRPC)}
	}
	return &msg, nil
}

func (c *Codec) write(msg *Message) error {
	return c.writeContext(context.Background(), nil, c.interruptWrite, msg)
}

// writeContext shares ownership with all codec writes. Only an active frame
// may interrupt the transport; a cancelled waiter never owns that decision.
func (c *Codec) writeContext(ctx context.Context, closed <-chan struct{}, interrupt func(), msg *Message) (err error) {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return c.writeEncodedContext(ctx, ctx, closed, interrupt, data)
}

// writeEncodedContext keeps the borrowed frame until both Write and its
// interruption callback finish. Acquisition and active lifetimes are distinct.
func (c *Codec) writeEncodedContext(acquire, ctx context.Context, closed <-chan struct{}, interrupt func(), data []byte) (err error) {
	select {
	case <-acquire.Done():
		return acquire.Err()
	case <-ctx.Done():
		return ctx.Err()
	case <-closed:
		return ErrPeerClosed
	case c.wgate <- struct{}{}:
	}
	defer func() { <-c.wgate }()
	return c.writeOwnedContext(acquire, ctx, closed, interrupt, data)
}

func (c *Codec) writeOwnedContext(acquire, ctx context.Context, closed <-chan struct{}, interrupt func(), data []byte) (err error) {
	// The gate and cancellation can become ready together. Recheck before any
	// transport access so an expired waiter cannot write later.
	if err := acquire.Err(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case <-closed:
		return ErrPeerClosed
	default:
	}
	if c.retired {
		return ErrPeerClosed
	}

	var finish func()
	var interruptWrite func()
	if interrupt != nil {
		var mu sync.Mutex
		finished, interrupted := false, false
		interruptWrite = func() {
			mu.Lock()
			if finished || interrupted {
				mu.Unlock()
				return
			}
			interrupted = true
			mu.Unlock()
			interrupt()
		}
		done := make(chan struct{})
		stop := context.AfterFunc(ctx, func() {
			interruptWrite()
			close(done)
		})
		finish = func() {
			mu.Lock()
			finished = true
			c.retired = interrupted
			mu.Unlock()
			// Join an already-started callback before releasing write ownership.
			// The Write itself always runs synchronously on this goroutine.
			if !stop() {
				<-done
			}
		}
		defer func() {
			finish()
			if c.retired && ctx.Err() != nil {
				err = ctx.Err()
			}
		}()
	}
	// Loop on Write so a short-write doesn't truncate the JSON-RPC frame
	// the peer's bufio.Scanner is about to parse — a half-frame would either
	// fail to decode or silently merge with the next frame's bytes.
	for off := 0; off < len(data); {
		if interruptWrite != nil && ctx.Err() != nil {
			interruptWrite()
			return ctx.Err()
		}
		n, werr := c.writer.Write(data[off:])
		if werr != nil {
			if interruptWrite != nil {
				interruptWrite()
				if ctx.Err() != nil {
					return ctx.Err()
				}
			}
			return werr
		}
		if n == 0 {
			if interruptWrite != nil {
				interruptWrite()
			}
			return io.ErrShortWrite
		}
		off += n
	}
	return nil
}

// Notify sends a fire-and-forget notification (method, params, no id).
func (c *Codec) Notify(method string, params any) error {
	return c.NotifyContext(context.Background(), method, params)
}

// NotifyContext sends a notification with ctx covering write-gate acquisition
// and the complete frame. Active cancellation requires an owner interrupt hook.
func (c *Codec) NotifyContext(ctx context.Context, method string, params any) error {
	return notify(func(msg *Message) error {
		return c.writeContext(ctx, nil, c.interruptWrite, msg)
	}, method, params)
}

func notify(write func(*Message) error, method string, params any) error {
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return write(&Message{
		JSONRPC: "2.0",
		Method:  method,
		Params:  raw,
	})
}

// Respond sends a successful response carrying result for the given request id.
func (c *Codec) Respond(id *json.RawMessage, result any) error {
	return respond(c.write, id, result)
}

func respond(write func(*Message) error, id *json.RawMessage, result any) error {
	raw, err := json.Marshal(result)
	if err != nil {
		return err
	}
	return write(&Message{
		JSONRPC: "2.0",
		ID:      id,
		Result:  raw,
	})
}

// RespondError sends an error response for the given request id.
func (c *Codec) RespondError(id *json.RawMessage, code int, message string) error {
	return respondErrorData(c.write, id, code, message, nil)
}

// RespondHistoryError is the required local history control reply, not the
// best-effort worker refusal path. Its caller owns the validated borrowed ID
// until return, and funds one fixed control frame. The token must already have
// passed the transport's request/ID validation (or be the guard's literal null);
// this writer never reparses rejected input or reconstructs request identity.
// The broker admits only its synchronous local dispatcher and single initial
// fallback as simultaneous callers. No goroutine, queue or frame grant is added.
// Like RespondError, admission waits for the ordinary write gate and then joins
// the actual Write/interruption. A busy gate never discards a required reply.
// Ordinary responses keep their existing json.Marshal serialization.
func (c *Codec) RespondHistoryError(id *json.RawMessage, code int) error {
	token := []byte("null")
	if id != nil {
		if len(*id) != 0 && len(*id) <= HistoryIDBytes {
			token = *id
		} else {
			code = HistoryInvalidCode
		}
	}
	var suffix string
	switch code {
	case HistoryBusyCode:
		suffix = `,"error":{"code":-32096,"message":"routing history busy; retry"}}` + "\n"
	case HistoryInvalidCode:
		suffix = `,"error":{"code":-32062,"message":"history-invalid"}}` + "\n"
	case -32601:
		suffix = `,"error":{"code":-32601,"message":"method not found"}}` + "\n"
	default:
		return ErrHistoryInvalid
	}
	const prefix = `{"jsonrpc":"2.0","id":`
	// The complete frame is below E. The synchronous local dispatcher funds
	// this 640B backing in fixed C; the initial fallback's same-sized backing
	// fits its existing writer scratch after its record encoder has returned.
	var frame [HistoryIDBytes + 128]byte
	if len(prefix)+len(token)+len(suffix) > len(frame) {
		return ErrHistoryInvalid
	}
	data := append(frame[:0], prefix...)
	data = append(data, token...)
	data = append(data, suffix...)
	return c.writeEncodedContext(context.Background(), context.Background(), nil, c.interruptWrite, data)
}

// RespondErrorData is like RespondError but carries an optional structured
// data payload (JSON-RPC 2.0 error object's "data" field). data may be nil.
func (c *Codec) RespondErrorData(id *json.RawMessage, code int, message string, data any) error {
	return respondErrorData(c.write, id, code, message, data)
}

func respondErrorData(write func(*Message) error, id *json.RawMessage, code int, message string, data any) error {
	var raw json.RawMessage
	if data != nil {
		b, err := json.Marshal(data)
		if err != nil {
			return err
		}
		raw = b
	}
	return write(&Message{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &RPCError{Code: code, Message: message, Data: raw},
	})
}
