// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package jsonrpc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync/atomic"
	"testing"
)

// rw adapts a separate reader and writer into an io.ReadWriter for NewCodec.
type rw struct {
	io.Reader
	io.Writer
}

func writeCodec(w io.Writer) *Codec { return NewCodec(rw{Reader: bytes.NewReader(nil), Writer: w}) }
func readCodec(r io.Reader) *Codec  { return NewCodec(rw{Reader: r, Writer: io.Discard}) }

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
