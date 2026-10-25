// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package jsonrpc

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
	"unsafe"
)

// rw adapts a separate reader and writer into an io.ReadWriter for NewCodec.
type rw struct {
	io.Reader
	io.Writer
}

func writeCodec(w io.Writer) *Codec { return NewCodec(rw{Reader: bytes.NewReader(nil), Writer: w}) }
func readCodec(r io.Reader) *Codec  { return NewCodec(rw{Reader: r, Writer: io.Discard}) }

func TestLocalHistoryErrorPreservesRawIDAndOrdinarySerialization(t *testing.T) {
	test := func(name, token string, code int, wantToken string) {
		t.Run(name, func(t *testing.T) {
			var out oneByteWriter
			codec := writeCodec(&out)
			id := json.RawMessage(token)
			if err := codec.RespondHistoryError(&id, code); err != nil {
				t.Fatalf("required history error: %v", err)
			}
			if out.buf.Len() > HistoryIDBytes+128 || out.buf.Len() > HistoryEnvelopeBytes {
				t.Fatal("local control error exceeded E")
			}
			msg, err := readCodec(bytes.NewReader(out.buf.Bytes())).Read()
			if err != nil {
				t.Fatal(err)
			}
			gotToken := "null"
			if msg.ID != nil {
				gotToken = string(*msg.ID)
			}
			if gotToken != wantToken || msg.Error == nil || msg.Error.Code != code || msg.Error.Data != nil {
				t.Fatalf("raw local error: token %s, error %+v", gotToken, msg.Error)
			}
		})
	}
	for _, code := range []int{HistoryBusyCode, HistoryInvalidCode, -32601} {
		test(fmt.Sprintf("literal less-than code %d", code), `"`+strings.Repeat("<", 86)+`"`, code, `"`+strings.Repeat("<", 86)+`"`)
		test(fmt.Sprintf("encoded boundary code %d", code), `"`+strings.Repeat("<", 510)+`"`, code, `"`+strings.Repeat("<", 510)+`"`)
	}
	test("over-cap ID becomes null", `"`+strings.Repeat("<", 511)+`"`, HistoryInvalidCode, "null")
	var out bytes.Buffer
	id := json.RawMessage(`"<"`)
	if err := writeCodec(&out).RespondError(&id, -32601, "ordinary"); err != nil {
		t.Fatal(err)
	}
	msg, err := readCodec(&out).Read()
	if err != nil || msg.ID == nil || string(*msg.ID) != `"\u003c"` {
		t.Fatalf("ordinary response serialization changed: %+v / %v", msg, err)
	}
}

func TestLocalHistoryErrorWaitsForGateAndActualPartialWrite(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		w := newBlockedPeerWriter(t, 7, false)
		codec := writeCodec(w)
		codec.wgate <- struct{}{} // an existing ordinary writer owns admission
		id := json.RawMessage(`"` + strings.Repeat("<", 86) + `"`)
		done := make(chan error, 1)
		go func() { done <- codec.RespondHistoryError(&id, HistoryBusyCode) }()
		synctest.Wait()
		select {
		case err := <-done:
			t.Fatalf("mandatory reply dropped at occupied gate: %v", err)
		case <-w.entered:
			t.Fatal("required reply bypassed the existing write gate")
		default:
		}
		<-codec.wgate
		awaitPeerSignal(t, w.entered, "partial required error write")
		select {
		case err := <-done:
			t.Fatalf("required reply completed during partial Write: %v", err)
		default:
		}
		w.unblock()
		if err := awaitPeerError(t, done); err != nil {
			t.Fatal(err)
		}
		raw, _ := w.snapshot()
		msg, err := readCodec(bytes.NewReader(raw)).Read()
		if err != nil || msg.ID == nil || string(*msg.ID) != string(id) || msg.Error == nil || msg.Error.Code != HistoryBusyCode {
			t.Fatalf("partial response changed exact ID: %+v / %v", msg, err)
		}
	})
}

func TestLocalHistoryErrorWriteFailureUsesActualOwner(t *testing.T) {
	w := &failingFrameWriter{n: 7, err: io.ErrClosedPipe}
	interrupts := 0
	codec := NewCodecMaxFrameWithWriteInterrupt(rw{strings.NewReader(""), w}, WorkerFrameBytes, func() { interrupts++ })
	id := json.RawMessage(`"<"`)
	if err := codec.RespondHistoryError(&id, HistoryInvalidCode); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("actual required reply failure = %v", err)
	}
	if interrupts != 1 || w.calls != 1 {
		t.Fatal("partial required reply lost its actual write owner")
	}
	if err := codec.Notify("ordinary-after-broken-write", nil); !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("broken writer reused after required response: %v", err)
	}
}

func TestNotificationLifetimeCodecDirectFailureAndPeerOwner(t *testing.T) {
	t.Run("direct write retires codec", func(t *testing.T) {
		var interrupts atomic.Int32
		c := NewCodecMaxFrameWithWriteInterrupt(rw{bytes.NewReader(nil), &failingFrameWriter{err: io.ErrClosedPipe}}, WorkerFrameBytes, func() { interrupts.Add(1) })
		id := json.RawMessage("1")
		if err := c.Respond(&id, nil); err == nil {
			t.Fatal("failed writer succeeded")
		}
		if err := c.Notify("later", nil); !errors.Is(err, ErrPeerClosed) {
			t.Fatalf("later write: %v", err)
		}
		if interrupts.Load() != 1 {
			t.Fatalf("interrupts=%d", interrupts.Load())
		}
	})
	t.Run("peer has sole interrupt ownership", func(t *testing.T) {
		var codecInterrupts, peerInterrupts atomic.Int32
		c := NewCodecMaxFrameWithWriteInterrupt(rw{bytes.NewReader(nil), &failingFrameWriter{err: io.ErrClosedPipe}}, WorkerFrameBytes, func() { codecInterrupts.Add(1) })
		p := NewPeerWithWriteInterrupt(c, func() { peerInterrupts.Add(1) })
		if err := p.NotifyContext(context.Background(), "failed", nil); err == nil {
			t.Fatal("failed writer succeeded")
		}
		if codecInterrupts.Load() != 0 || peerInterrupts.Load() != 1 {
			t.Fatalf("codec=%d peer=%d", codecInterrupts.Load(), peerInterrupts.Load())
		}
	})
}

func TestClassifiers(t *testing.T) {
	id := json.RawMessage("1")
	req := &Message{JSONRPC: "2.0", ID: &id, Method: "m"}
	note := &Message{JSONRPC: "2.0", Method: "m"}
	resp := &Message{JSONRPC: "2.0", ID: &id}

	if !req.IsRequest() || req.IsNotification() || req.IsResponse() {
		t.Errorf("request misclassified")
	}
	if !note.IsNotification() || note.IsRequest() || note.IsResponse() {
		t.Errorf("notification misclassified")
	}
	if !resp.IsResponse() || resp.IsRequest() || resp.IsNotification() {
		t.Errorf("response misclassified")
	}
}

func TestHistoryFrameOrdinaryCodecUnchanged(t *testing.T) {
	frame := `{"jsonrpc":"2.0","method":"ordinary","params":{"value":"` + strings.Repeat("x", 2<<20) + `"}}` + "\n"
	if _, err := readCodec(strings.NewReader(frame)).Read(); err == nil {
		t.Fatal("default codec cap widened")
	}
	msg, err := NewCodecMaxFrame(rw{strings.NewReader(frame), io.Discard}, WorkerFrameBytes).Read()
	if err != nil || msg == nil || msg.Method != "ordinary" {
		t.Fatalf("ordinary worker frame rejected: %v", err)
	}
	if _, err := historyTestReader(t, frame, &HistoryCorrelation{}).Read(); err == nil {
		t.Fatal("opt-in reader widened ordinary logical cap")
	}
	ordinary := `{"jsonrpc":"2.0","method":"first","method":"last","id":"` + strings.Repeat("x", 600) + `"}` + "\n"
	msg, err = historyTestReader(t, ordinary, &HistoryCorrelation{}).Read()
	if err != nil || msg == nil || msg.Method != "last" || msg.ID == nil || len(*msg.ID) != 602 {
		t.Fatalf("ordinary decoder semantics changed: %v", err)
	}
}

func TestNotifyRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := writeCodec(&buf).Notify("node/discovered", map[string]string{"id": "n1"}); err != nil {
		t.Fatalf("Notify: %v", err)
	}
	if !bytes.HasSuffix(buf.Bytes(), []byte("\n")) {
		t.Fatalf("frame not newline-terminated: %q", buf.String())
	}
	msg, err := readCodec(&buf).Read()
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !msg.IsNotification() || msg.Method != "node/discovered" {
		t.Fatalf("unexpected frame: %+v", msg)
	}
	var params map[string]string
	if err := json.Unmarshal(msg.Params, &params); err != nil || params["id"] != "n1" {
		t.Fatalf("params round-trip failed: %v %v", params, err)
	}
}

func TestRespondAndError(t *testing.T) {
	id := json.RawMessage("7")

	var okBuf bytes.Buffer
	if err := writeCodec(&okBuf).Respond(&id, map[string]int{"level": 1}); err != nil {
		t.Fatalf("Respond: %v", err)
	}
	msg, err := readCodec(&okBuf).Read()
	if err != nil || !msg.IsResponse() || msg.Error != nil {
		t.Fatalf("bad ok response: %+v err=%v", msg, err)
	}

	var errBuf bytes.Buffer
	if err := writeCodec(&errBuf).RespondError(&id, -32601, "method not found"); err != nil {
		t.Fatalf("RespondError: %v", err)
	}
	emsg, err := readCodec(&errBuf).Read()
	if err != nil {
		t.Fatalf("Read err response: %v", err)
	}
	if emsg.Error == nil || emsg.Error.Code != -32601 || emsg.Error.Message != "method not found" {
		t.Fatalf("bad error response: %+v", emsg.Error)
	}
}

// oneByteWriter accepts a single byte per Write, forcing the codec's
// short-write loop to run to completion.
type oneByteWriter struct{ buf bytes.Buffer }

func (w *oneByteWriter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	_ = w.buf.WriteByte(p[0])
	return 1, nil
}

func TestShortWriteWritesFullFrame(t *testing.T) {
	var w oneByteWriter
	if err := writeCodec(&w).Notify("m", map[string]string{"k": "value-that-spans-many-writes"}); err != nil {
		t.Fatalf("Notify: %v", err)
	}
	msg, err := readCodec(bytes.NewReader(w.buf.Bytes())).Read()
	if err != nil {
		t.Fatalf("Read after short writes: %v", err)
	}
	var params map[string]string
	if err := json.Unmarshal(msg.Params, &params); err != nil || params["k"] != "value-that-spans-many-writes" {
		t.Fatalf("short-write corrupted frame: %v %v", params, err)
	}
}

func TestReadRejectsBadVersionAndEOF(t *testing.T) {
	if _, err := readCodec(strings.NewReader(`{"jsonrpc":"1.0","method":"m"}` + "\n")).Read(); err == nil {
		t.Fatalf("expected error for unsupported jsonrpc version")
	}
	if _, err := readCodec(strings.NewReader("")).Read(); err != io.EOF {
		t.Fatalf("expected io.EOF on empty stream, got %v", err)
	}
}

func TestReadMalformedFrameIsRecoverableDecodeError(t *testing.T) {
	// Both bad JSON and a bad version are recoverable *DecodeError so a read
	// loop can continue rather than treating them as terminal.
	for _, frame := range []string{"{not json}\n", `{"jsonrpc":"1.0"}` + "\n"} {
		_, err := readCodec(strings.NewReader(frame)).Read()
		var de *DecodeError
		if !errors.As(err, &de) {
			t.Fatalf("frame %q: want *DecodeError, got %T (%v)", frame, err, err)
		}
	}
}

func TestRespondErrorDataRoundTrip(t *testing.T) {
	id := json.RawMessage("3")
	var buf bytes.Buffer
	if err := writeCodec(&buf).RespondErrorData(&id, -32602, "bad", map[string]string{"field": "port"}); err != nil {
		t.Fatalf("RespondErrorData: %v", err)
	}
	msg, err := readCodec(&buf).Read()
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if msg.Error == nil || msg.Error.Code != -32602 {
		t.Fatalf("bad error: %+v", msg.Error)
	}
	var data map[string]string
	if err := json.Unmarshal(msg.Error.Data, &data); err != nil || data["field"] != "port" {
		t.Fatalf("error data round-trip failed: %v %v", data, err)
	}
	// nil data omits the field entirely.
	var buf2 bytes.Buffer
	_ = writeCodec(&buf2).RespondErrorData(&id, -32603, "boom", nil)
	m2, _ := readCodec(&buf2).Read()
	if len(m2.Error.Data) != 0 {
		t.Fatalf("expected empty data for nil, got %q", string(m2.Error.Data))
	}
}

func TestNewCodecMaxFrameAcceptsLargeFrame(t *testing.T) {
	big := strings.Repeat("x", 2<<20) // 2 MiB, over the 1 MiB default
	var buf bytes.Buffer
	if err := writeCodec(&buf).Notify("m", map[string]string{"blob": big}); err != nil {
		t.Fatalf("Notify: %v", err)
	}
	c := NewCodecMaxFrame(rw{Reader: bytes.NewReader(buf.Bytes()), Writer: io.Discard}, 8<<20)
	msg, err := c.Read()
	if err != nil {
		t.Fatalf("Read large frame: %v", err)
	}
	var params map[string]string
	if err := json.Unmarshal(msg.Params, &params); err != nil || len(params["blob"]) != len(big) {
		t.Fatalf("large frame round-trip failed: %v", err)
	}
}

func TestCodecInterruptibleShortWriteCompletesFrame(t *testing.T) {
	var w oneByteWriter
	c := writeCodec(&w)
	interrupts := 0
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	id := json.RawMessage("42")
	if err := c.writeContext(ctx, nil, func() { interrupts++ }, &Message{
		JSONRPC: "2.0", ID: &id, Method: "request", Params: json.RawMessage(`{"key":"value"}`),
	}); err != nil {
		t.Fatalf("interruptible short write: %v", err)
	}
	cancel()
	if err := c.Notify("after", nil); err != nil {
		t.Fatalf("Notify after completed write: %v", err)
	}
	reader := readCodec(bytes.NewReader(w.buf.Bytes()))
	msg, err := reader.Read()
	if err != nil {
		t.Fatalf("Read short-written request: %v", err)
	}
	if !msg.IsRequest() || msg.Method != "request" || string(*msg.ID) != "42" {
		t.Fatalf("short-written request = %+v", msg)
	}
	var params map[string]string
	if err := json.Unmarshal(msg.Params, &params); err != nil || params["key"] != "value" {
		t.Fatalf("short-written params = %v, err=%v", params, err)
	}
	next, err := reader.Read()
	if err != nil || !next.IsNotification() || next.Method != "after" {
		t.Fatalf("next frame = %+v, err=%v", next, err)
	}
	if _, err := reader.Read(); !errors.Is(err, io.EOF) {
		t.Fatalf("unexpected extra frame: %v", err)
	}
	if interrupts != 0 {
		t.Fatalf("completed frame interruption calls = %d, want 0", interrupts)
	}
}

type failingFrameWriter struct {
	calls int
	n     int
	err   error
}

func (w *failingFrameWriter) Write([]byte) (int, error) {
	w.calls++
	return w.n, w.err
}

func TestCodecInterruptibleWriteFailureRetiresFrame(t *testing.T) {
	test := func(name string, n int, writeErr, wantErr error) {
		t.Run(name, func(t *testing.T) {
			w := &failingFrameWriter{n: n, err: writeErr}
			c := writeCodec(w)
			interrupts := 0
			if err := c.writeContext(context.Background(), nil, func() { interrupts++ }, &Message{JSONRPC: "2.0", Method: "first"}); !errors.Is(err, wantErr) {
				t.Fatalf("write error = %v, want %v", err, wantErr)
			}
			if err := c.Notify("later", nil); !errors.Is(err, ErrPeerClosed) {
				t.Fatalf("write after failure = %v, want peer closed", err)
			}
			if w.calls != 1 || interrupts != 1 {
				t.Fatalf("writer calls=%d, interruption calls=%d; want one each", w.calls, interrupts)
			}
		})
	}
	test("zero-byte short write", 0, nil, io.ErrShortWrite)
	test("partial write error", 3, io.ErrClosedPipe, io.ErrClosedPipe)
}

func TestCodecInterruptibleInvalidFramePreservesWriter(t *testing.T) {
	var buf bytes.Buffer
	c := writeCodec(&buf)
	interrupts := 0
	err := c.writeContext(context.Background(), nil, func() { interrupts++ }, &Message{JSONRPC: "2.0", Method: "invalid", Params: json.RawMessage("{")})
	if err == nil {
		t.Fatal("invalid raw JSON frame was accepted")
	}
	if buf.Len() != 0 || interrupts != 0 {
		t.Fatalf("invalid frame wrote %d bytes and interrupted %d times", buf.Len(), interrupts)
	}
	if err := c.Notify("healthy", nil); err != nil {
		t.Fatalf("Notify after encoding failure: %v", err)
	}
	msg, err := readCodec(&buf).Read()
	if err != nil || !msg.IsNotification() || msg.Method != "healthy" {
		t.Fatalf("frame after encoding failure = %+v, err=%v", msg, err)
	}
}

func TestCodecDirectWriteFailurePreservesExistingBehavior(t *testing.T) {
	var w partialErrorWriter
	c := writeCodec(&w)
	if err := c.Notify("first", nil); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("direct Codec write error = %v, want unexpected EOF", err)
	}
	if err := c.Notify("second", nil); err != nil {
		t.Fatalf("direct Codec second write = %v, want existing caller-owned lifecycle", err)
	}
	if w.calls != 2 {
		t.Fatalf("direct Codec writer calls = %d, want 2", w.calls)
	}
	want := `{"jsonr{"jsonrpc":"2.0","method":"second","params":null}` + "\n"
	if w.buf.String() != want {
		t.Fatalf("direct Codec bytes = %q, want %q", w.buf.String(), want)
	}
}

// These are test-owned role backings, reused verbatim after each retirement.
// Production role admission belongs to the broker, not to the shared codec.
type readRetirementBacking struct {
	raw, output []byte
	offsets     []uint32
	control     *ReadControl
}

func newReadRetirementBacking() readRetirementBacking {
	return readRetirementBacking{
		raw: make([]byte, 0, 1<<20), output: make([]byte, 0, HistoryRecordFrameBytes),
		offsets: make([]uint32, HistoryRecordBytes/4),
		control: &ReadControl{},
	}
}

func TestReadRetirementFixedControlEpochAndLatePreloadedReader(t *testing.T) {
	backing := newReadRetirementBacking()
	frame := `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1}}` + "\n"
	old := newReadRetirementCodec(t, rw{strings.NewReader(frame), io.Discard}, backing, &HistoryCorrelation{}, func() {})
	preloaded := old.history.Load()
	epoch := backing.control.epoch
	_, staleRelease, err := old.ReadWithRelease()
	if err != nil || staleRelease == nil {
		t.Fatalf("first borrow: %v", err)
	}
	done := old.RetireRead()
	assertReadRetirementPending(t, done)
	staleRelease()
	awaitPeerSignal(t, done, "first generation completion")

	replacement := newReadRetirementCodec(t, rw{strings.NewReader(frame), io.Discard}, backing, &HistoryCorrelation{}, func() {})
	if replacement.history.Load().readControl != backing.control || backing.control.epoch <= epoch {
		t.Fatal("replacement did not advance the same fixed role control")
	}
	msg, release, err := replacement.ReadWithRelease()
	if err != nil || release == nil {
		t.Fatalf("replacement borrow: %v", err)
	}
	defer release()
	if msg, release, err := old.readHistoryWithRelease(preloaded); msg != nil || release != nil || !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("preloaded old reader entered rebound control: %v", err)
	}
	old.RetireRead()
	staleRelease()
	done = replacement.RetireRead()
	assertReadRetirementPending(t, done)
	var value historyTestValue
	if err := json.Unmarshal(msg.Params, &value); err != nil || value.Schema != 1 {
		t.Fatalf("stale completion changed replacement borrow: %+v, %v", value, err)
	}
	release()
	awaitPeerSignal(t, done, "replacement completion")
	control := backing.control
	if control.codec != nil || control.reader != nil || control.interrupt != nil || control.fence != nil || control.onRequest != nil || control.onHistoryFailure != nil || control.done != nil {
		t.Fatal("completed fixed control retains generation references")
	}
}

func TestReadRetirementCodecKeepsExistingPointerFootprint(t *testing.T) {
	// Mirror the pre-retirement codec: atomic publication replaces its one
	// history pointer. No per-old-codec control, epoch or completion is funded.
	type baseline struct {
		scanner        *bufio.Scanner
		writer         io.Writer
		wgate          chan struct{}
		retired        bool
		interruptWrite func()
		history        *HistoryReader
	}
	if unsafe.Sizeof(Codec{}) != unsafe.Sizeof(baseline{}) || reflect.TypeOf(Codec{}).NumField() != reflect.TypeOf(baseline{}).NumField() {
		t.Fatal("old codec shell gained a retirement tail")
	}
}

func readRetirementSchema(method string, initial bool) (HistoryProjection, bool) {
	schema, ok := historyTestSchema(method, initial)
	if !initial && (method == "routing:get-initial" || method == "routing:subscribe" || method == "routing:unsubscribe") {
		schema.Type = reflect.TypeOf(struct{}{})
	}
	return schema, ok
}

func newReadRetirementCodec(t *testing.T, transport io.ReadWriter, backing readRetirementBacking, correlation *HistoryCorrelation, interrupt func()) *Codec {
	t.Helper()
	c, err := NewHistoryCodec(transport, 1<<20, backing.raw, backing.output, backing.offsets, correlation, readRetirementSchema, nil)
	if err != nil {
		t.Fatalf("construct history codec: %v", err)
	}
	if err := c.EnableReadRetirement(backing.control, interrupt); err != nil {
		t.Fatalf("enable read retirement: %v", err)
	}
	return c
}

func assertReadRetirementPending(t *testing.T, done <-chan struct{}) {
	t.Helper()
	if done == nil {
		t.Fatal("retirement was not configured")
	}
	select {
	case <-done:
		t.Fatal("retirement completed before actual reader/borrower release")
	default:
	}
}

func assertReadRetirementDetached(t *testing.T, c *Codec, reader *HistoryReader) {
	t.Helper()
	if c.history.Load() != sharedTerminalReader || c.scanner != nil || reader.r != nil || reader.raw != nil || reader.output != nil || reader.offsets != nil || reader.correlation != nil || reader.schema != nil || reader.ordinary != 0 {
		t.Fatal("retired shell still references reader storage")
	}
	if reader.idToken != [HistoryIDBytes]byte{} {
		t.Fatal("retired shell retained owner hook or borrowed ID")
	}
	if msg, release, err := c.ReadWithRelease(); msg != nil || release != nil || !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("late leased read = %v, release present %v, error %v", msg, release != nil, err)
	}
	if msg, err := c.Read(); msg != nil || !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("late plain read = %v, error %v", msg, err)
	}
}

func TestReadRetirementRequiresExplicitOptIn(t *testing.T) {
	frame := `{"jsonrpc":"2.0","method":"ordinary","params":{"value":1}}` + "\n"
	c := readCodec(strings.NewReader(frame + frame))
	if done := c.RetireRead(); done != nil {
		t.Fatal("ordinary codec claimed retirement proof")
	}
	if err := c.EnableReadRetirement(&ReadControl{}, func() {}); !errors.Is(err, ErrHistoryInvalid) {
		t.Fatalf("ordinary codec enabled history retirement: %v", err)
	}
	if _, err := c.Read(); err != nil {
		t.Fatalf("ordinary Read changed: %v", err)
	}
	msg, release, err := c.ReadWithRelease()
	if err != nil || msg == nil || msg.borrowedRead || release != nil {
		t.Fatalf("ordinary explicit read = %v, release present %v, error %v", msg, release != nil, err)
	}
	history := historyTestReader(t, "", &HistoryCorrelation{})
	if err := history.EnableReadRetirement(&ReadControl{}, nil); !errors.Is(err, ErrHistoryInvalid) {
		t.Fatalf("nil owner interrupt admitted: %v", err)
	}
}

func TestReadRetirementFencesBufferedNextFrame(t *testing.T) {
	frame := `{"jsonrpc":"2.0","method":"ordinary","params":{"value":1}}` + "\n"
	var interrupts atomic.Int32
	c := newReadRetirementCodec(t, rw{strings.NewReader(frame + frame), io.Discard}, newReadRetirementBacking(), &HistoryCorrelation{}, func() { interrupts.Add(1) })
	reader := c.history.Load()
	if err := c.EnableReadRetirement(&ReadControl{}, func() {}); !errors.Is(err, ErrHistoryInvalid) {
		t.Fatalf("duplicate configuration admitted: %v", err)
	}
	if _, err := c.Read(); !errors.Is(err, ErrReadLeaseRequired) {
		t.Fatalf("plain read bypassed completion boundary: %v", err)
	}
	msg, release, err := c.ReadWithRelease()
	if err != nil || release != nil || msg == nil || msg.Method != "ordinary" {
		t.Fatalf("ordinary read = %v, release present %v, error %v", msg, release != nil, err)
	}
	if reader.r.Buffered() == 0 {
		t.Fatal("fixture did not buffer the next frame")
	}
	done := c.RetireRead()
	awaitPeerSignal(t, done, "buffered reader retirement")
	awaitPeerSignal(t, c.RetireRead(), "duplicate buffered reader retirement")
	if interrupts.Load() != 0 {
		t.Fatal("duplicate retirement interrupted an idle reader")
	}
	assertReadRetirementDetached(t, c, reader)
	var value struct {
		Value int `json:"value"`
	}
	if err := json.Unmarshal(msg.Params, &value); err != nil || value.Value != 1 {
		t.Fatalf("owned ordinary payload changed after detachment: %+v, %v", value, err)
	}
}

func TestReadRetirementHoldsValidatedBorrows(t *testing.T) {
	test := func(name, frame string, initial, refusal bool) {
		t.Run(name, func(t *testing.T) {
			backing := newReadRetirementBacking()
			var correlation HistoryCorrelation
			if initial {
				backing = readRetirementBacking{make([]byte, 0, HistoryFrameBytes), make([]byte, 0, HistoryFrameBytes), make([]uint32, HistoryInitialBytes/4), &ReadControl{}}
				if err := correlation.Begin([]byte("1")); err != nil {
					t.Fatal(err)
				}
			}
			c := newReadRetirementCodec(t, rw{strings.NewReader(frame + "\n"), io.Discard}, backing, &correlation, func() { t.Error("borrowed consumption is not an active transport read") })
			reader := c.history.Load()
			msg, release, err := c.ReadWithRelease()
			if release == nil {
				t.Fatalf("validated borrow has no release: %v", err)
			}
			defer release()
			var failure *HistoryFailure
			if refusal {
				if msg != nil || !errors.As(err, &failure) || failure.Refusal == nil || string(failure.Refusal.ReplyID) != `"\u0061"` {
					t.Fatalf("borrowed refusal = %v, %v", msg, err)
				}
			} else if err != nil || msg == nil || !msg.borrowedRead {
				t.Fatalf("projected message = %v, %v", msg, err)
			}
			if _, extra, err := c.ReadWithRelease(); extra != nil || !errors.Is(err, ErrHistoryBusy) {
				t.Fatalf("read admitted during borrow: %v", err)
			}
			done := c.RetireRead()
			assertReadRetirementPending(t, done)
			if reader.raw == nil || reader.r == nil || (refusal && string(failure.Refusal.ReplyID) != `"\u0061"`) {
				t.Fatal("retirement detached storage still borrowed by its consumer")
			}
			release()
			release()
			awaitPeerSignal(t, done, "borrow completion")
			assertReadRetirementDetached(t, c, reader)
		})
	}
	test("incoming request ID and params", `{"jsonrpc":"2.0","id":"\u0061","method":"routing:subscribe","params":{}}`, false, false)
	test("notification params", `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1}}`, false, false)
	test("result and ID", `{"jsonrpc":"2.0","id":1,"result":{"requests":[]}}`, true, false)
	test("error ID", `{"jsonrpc":"2.0","id":1,"error":{"code":-32096,"message":"routing history busy; retry"}}`, true, false)
	test("nil message refusal ID", `{"jsonrpc":"2.0","id":"\u0061","method":"routing:subscribe","params":null}`, false, true)
}

func TestReadRetirementDuplicateReleaseCannotReleaseNextBorrow(t *testing.T) {
	frame := `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1}}` + "\n"
	c := newReadRetirementCodec(t, rw{strings.NewReader(frame + frame), io.Discard}, newReadRetirementBacking(), &HistoryCorrelation{}, func() {})
	_, first, err := c.ReadWithRelease()
	if err != nil || first == nil {
		t.Fatalf("first borrow: %v", err)
	}
	first()
	_, second, err := c.ReadWithRelease()
	if err != nil || second == nil {
		t.Fatalf("second borrow: %v", err)
	}
	defer second()
	first()
	done := c.RetireRead()
	assertReadRetirementPending(t, done)
	second()
	awaitPeerSignal(t, done, "second borrow release")
}

func TestReadRetirementInPlaceBackingReuse(t *testing.T) {
	backing := newReadRetirementBacking()
	backing.output = nil // NewHistoryCodec's supported raw/output alias.
	frame := `{"jsonrpc":"2.0","method":"routing/request","params":{"schema":1}}` + "\n"
	old := newReadRetirementCodec(t, rw{strings.NewReader(frame), io.Discard}, backing, &HistoryCorrelation{}, func() {})
	reader := old.history.Load()
	_, releaseOld, err := old.ReadWithRelease()
	if err != nil || releaseOld == nil {
		t.Fatalf("in-place borrow: %v", err)
	}
	defer releaseOld()
	done := old.RetireRead()
	assertReadRetirementPending(t, done)
	releaseOld()
	awaitPeerSignal(t, done, "in-place reader retirement")
	assertReadRetirementDetached(t, old, reader)

	replacement := newReadRetirementCodec(t, rw{strings.NewReader(frame), io.Discard}, backing, &HistoryCorrelation{}, func() {})
	if &replacement.history.Load().raw[:cap(replacement.history.Load().raw)][0] != &backing.raw[:cap(backing.raw)][0] ||
		&replacement.history.Load().output[:cap(replacement.history.Load().output)][0] != &backing.raw[:cap(backing.raw)][0] {
		t.Fatal("replacement did not reuse identical in-place backing")
	}
	msg, releaseNew, err := replacement.ReadWithRelease()
	if err != nil || releaseNew == nil {
		t.Fatalf("replacement borrow: %v", err)
	}
	defer releaseNew()
	releaseOld()
	assertReadRetirementDetached(t, old, reader)
	var value historyTestValue
	if err := json.Unmarshal(msg.Params, &value); err != nil || value.Schema != 1 {
		t.Fatalf("late old reference changed replacement data: %+v, %v", value, err)
	}
	newDone := replacement.RetireRead()
	assertReadRetirementPending(t, newDone)
	releaseNew()
	awaitPeerSignal(t, newDone, "replacement borrow release")
}

// These handoff schedules document the local dispatcher's required ownership
// transfer. They do not substitute for testing the broker readLoop integration.
func TestReadRetirementLocalHandoffCancellation(t *testing.T) {
	frame := `{"jsonrpc":"2.0","id":"\u0061","method":"routing:subscribe","params":null}` + "\n"
	t.Run("cancel before delivery releases sender borrow", func(t *testing.T) {
		c := newReadRetirementCodec(t, rw{strings.NewReader(frame), io.Discard}, newReadRetirementBacking(), &HistoryCorrelation{}, func() {})
		reader := c.history.Load()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		ready, senderDone := make(chan struct{}), make(chan struct{})
		deliver := make(chan HistoryFailure)
		readErrors := make(chan error, 1)
		go func() {
			_, release, err := c.ReadWithRelease()
			if release == nil {
				readErrors <- errors.New("refusal did not acquire a read lease")
				close(ready)
				close(senderDone)
				return
			}
			defer close(senderDone)
			defer release()
			var failure *HistoryFailure
			if !errors.As(err, &failure) || failure.Refusal == nil {
				readErrors <- errors.New("expected a borrowed refusal")
				close(ready)
				return
			}
			close(ready)
			select {
			case deliver <- *failure:
				readErrors <- errors.New("unattended handoff unexpectedly delivered")
			case <-ctx.Done():
				readErrors <- nil
			}
		}()
		awaitPeerSignal(t, ready, "undelivered local read")
		done := c.RetireRead()
		assertReadRetirementPending(t, done)
		cancel()
		awaitPeerSignal(t, senderDone, "cancelled sender release")
		if err := awaitPeerError(t, readErrors); err != nil {
			t.Fatal(err)
		}
		awaitPeerSignal(t, done, "undelivered borrow retirement")
		assertReadRetirementDetached(t, c, reader)
	})
	t.Run("cancel after delivery waits for dispatcher consumption", func(t *testing.T) {
		c := newReadRetirementCodec(t, rw{strings.NewReader(frame), io.Discard}, newReadRetirementBacking(), &HistoryCorrelation{}, func() {})
		reader := c.history.Load()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		type handoff struct {
			failure *HistoryFailure
			release func()
			err     error
		}
		deliver, senderDone := make(chan handoff), make(chan struct{})
		go func() {
			_, release, err := c.ReadWithRelease()
			var failure *HistoryFailure
			errors.As(err, &failure)
			select {
			case deliver <- handoff{failure, release, err}:
				// Delivery transfers release to the consumer. Cancellation must
				// not clear backing while synchronous dispatch still uses it.
				<-ctx.Done()
			case <-ctx.Done():
				if release != nil {
					release()
				}
			}
			close(senderDone)
		}()
		var received handoff
		select {
		case received = <-deliver:
		case <-time.After(time.Second):
			t.Fatal("local read was not delivered")
		}
		if received.release == nil || received.failure == nil || received.failure.Refusal == nil {
			t.Fatalf("local refusal handoff failed: %v", received.err)
		}
		defer received.release()
		cancel()
		awaitPeerSignal(t, senderDone, "cancelled sender exit")
		done := c.RetireRead()
		assertReadRetirementPending(t, done)
		if string(received.failure.Refusal.ReplyID) != `"\u0061"` {
			t.Fatal("cancellation cleared dispatcher-owned refusal ID")
		}
		received.release()
		awaitPeerSignal(t, done, "dispatcher consumption after cancellation")
		assertReadRetirementDetached(t, c, reader)
	})
}

type readRetirementBlockedReader struct {
	entered chan struct{}
	finish  chan struct{}
}

func (r *readRetirementBlockedReader) Read([]byte) (int, error) {
	close(r.entered)
	<-r.finish
	return 0, io.EOF
}

func TestReadRetirementJoinsActiveReadAndOwnerInterrupt(t *testing.T) {
	test := func(name string, readFirst bool) {
		t.Run(name, func(t *testing.T) {
			r := &readRetirementBlockedReader{make(chan struct{}), make(chan struct{})}
			interruptEntered, finishInterrupt := make(chan struct{}), make(chan struct{})
			var readOnce, interruptOnce sync.Once
			finishRead := func() { readOnce.Do(func() { close(r.finish) }) }
			finishHook := func() { interruptOnce.Do(func() { close(finishInterrupt) }) }
			t.Cleanup(finishRead)
			t.Cleanup(finishHook)
			var c *Codec
			c = newReadRetirementCodec(t, rw{r, io.Discard}, newReadRetirementBacking(), &HistoryCorrelation{}, func() {
				// These reentries require the read mutex to have been released.
				c.RetireRead()
				if _, _, err := c.ReadWithRelease(); !errors.Is(err, ErrPeerClosed) {
					t.Error("retirement did not fence reentrant read")
				}
				close(interruptEntered)
				<-finishInterrupt
			})
			reader := c.history.Load()
			readDone := make(chan error, 1)
			go func() {
				msg, release, err := c.ReadWithRelease()
				if msg != nil || release != nil {
					readDone <- errors.New("retired active read exposed a frame")
					return
				}
				readDone <- err
			}()
			awaitPeerSignal(t, r.entered, "active read admission")
			if _, release, err := c.ReadWithRelease(); release != nil || !errors.Is(err, ErrHistoryBusy) {
				t.Fatalf("overlapping active read admitted: %v", err)
			}
			retireReturned := make(chan struct{})
			go func() { c.RetireRead(); close(retireReturned) }()
			awaitPeerSignal(t, interruptEntered, "owner read interrupt")
			done := c.RetireRead()
			assertReadRetirementPending(t, done)
			if readFirst {
				finishRead()
				if err := awaitPeerError(t, readDone); !errors.Is(err, ErrPeerClosed) {
					t.Fatalf("retired active read: %v", err)
				}
				assertReadRetirementPending(t, done)
				finishHook()
			} else {
				finishHook()
				awaitPeerSignal(t, retireReturned, "interrupt return")
				assertReadRetirementPending(t, done)
				finishRead()
				if err := awaitPeerError(t, readDone); !errors.Is(err, ErrPeerClosed) {
					t.Fatalf("retired active read: %v", err)
				}
			}
			awaitPeerSignal(t, retireReturned, "retirement call return")
			awaitPeerSignal(t, done, "active read and interrupt join")
			assertReadRetirementDetached(t, c, reader)
		})
	}
	test("read returns before interrupt", true)
	test("interrupt returns before read", false)
}

type readRetirementObservedReader struct {
	io.Reader
	entered chan struct{}
	once    sync.Once
}

func (r *readRetirementObservedReader) Read(b []byte) (int, error) {
	r.once.Do(func() { close(r.entered) })
	return r.Reader.Read(b)
}

// Future native gate: execute on Windows as well as other supported platforms.
// This is an actual os.Pipe read, with no socket/process/firewall fixture. Source
// presence does not establish platform interruption or broker-owner integration.
func TestReadRetirementOSPipeInterrupt(t *testing.T) {
	input, output, err := os.Pipe()
	if err != nil {
		t.Fatalf("create OS pipe: %v", err)
	}
	var closeOnce sync.Once
	closeErrors := make(chan error, 1)
	closeInput := func() { closeOnce.Do(func() { closeErrors <- input.Close() }) }
	t.Cleanup(func() {
		// Closing the writer first also unblocks a read if interruption failed.
		// This is failure cleanup, not evidence that retirement interrupted it.
		if err := output.Close(); err != nil {
			t.Errorf("close OS pipe writer: %v", err)
		}
		closeInput()
	})
	r := &readRetirementObservedReader{Reader: input, entered: make(chan struct{})}
	c := newReadRetirementCodec(t, rw{r, io.Discard}, newReadRetirementBacking(), &HistoryCorrelation{}, closeInput)
	reader := c.history.Load()
	readDone := make(chan error, 1)
	go func() {
		_, release, err := c.ReadWithRelease()
		if release != nil {
			release()
		}
		readDone <- err
	}()
	awaitPeerSignal(t, r.entered, "OS pipe Read entry")
	retireReturned := make(chan struct{})
	go func() { c.RetireRead(); close(retireReturned) }()
	if err := awaitPeerError(t, closeErrors); err != nil {
		t.Fatalf("close actual OS pipe reader: %v", err)
	}
	done := c.RetireRead()
	if err := awaitPeerError(t, readDone); !errors.Is(err, ErrPeerClosed) {
		t.Fatalf("OS pipe read after retirement: %v", err)
	}
	awaitPeerSignal(t, done, "OS pipe read completion")
	awaitPeerSignal(t, retireReturned, "OS pipe retirement return")
	assertReadRetirementDetached(t, c, reader)
}
