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
	"fmt"
	"io"
	"sync"
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
	history        *HistoryReader
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
	if c.history != nil {
		return c.history.Read()
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
