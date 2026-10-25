// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

// The frame reader is asserted against complete fixtures and against bodies
// that cut a frame in half, drip bytes without finishing an event, and stall
// until the idle bound. Every case checks the exact bytes a client would have
// received, because that byte stream is the contract: complete frames only, and
// the protocol's own error frame when the stream cannot be completed.

const (
	tornOpenAIMarker = "TORN-OPENAI-FRAGMENT"
	tornOllamaMarker = "TORN-OLLAMA-FRAGMENT"
)

func TestStreamFrameNumericInputBoundary(t *testing.T) {
	for _, ending := range []string{"data: [DONE]\n\n", "data: {"} {
		t.Run(ending, func(t *testing.T) {
			var calls int
			r, err := newStreamFrameReader(streamFrameConfig{
				Budget:   newFrameBudget(64 << 20),
				Protocol: streamProtocolSSE,
				Upstream: io.NopCloser(strings.NewReader("data: {\"choices\":[]}\n\n" + ending)),
				Extract: func(payload []byte) performanceExtractMetadata {
					calls++
					if string(payload) != `{"choices":[]}` {
						t.Errorf("unexpected numeric input: %q", payload)
					}
					return performanceExtractMetadata{}
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			if _, err := io.Copy(io.Discard, r); err != nil {
				t.Fatal(err)
			}
			if calls != 1 {
				t.Fatalf("callback calls=%d, want one upstream object only", calls)
			}
		})
	}
}

func TestStreamFrameNumericScratchRefusal(t *testing.T) {
	for _, protocol := range []streamProtocol{streamProtocolSSE, streamProtocolNDJSON} {
		t.Run(strconv.Itoa(int(protocol)), func(t *testing.T) {
			input := "data: {\"choices\":[]}\n\ndata: [DONE]\n\n"
			want := wantSSEInterruption(streamInterruptReasonBudget)
			if protocol == streamProtocolNDJSON {
				input = `{"done":true}`
				want = wantNDJSONInterruption(streamInterruptReasonBudget)
			}
			budget := newFrameBudget(64 << 10)
			calls := 0
			r, err := newStreamFrameReader(streamFrameConfig{Protocol: protocol, Upstream: io.NopCloser(strings.NewReader(input)), Budget: budget, Extract: func([]byte) performanceExtractMetadata { calls++; return performanceExtractMetadata{} }})
			if err != nil {
				t.Fatal(err)
			}
			out, err := io.ReadAll(r)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(out, want) || calls != 0 {
				t.Fatalf("scratch refusal delivered upstream bytes or invoked callback: bytes=%q calls=%d", out, calls)
			}
			assertOutcome(t, r.Result(), streamOutcomeInterrupted, streamInterruptReasonBudget)
			if err := r.Close(); err != nil {
				t.Fatal(err)
			}
			if budget.inUse() != 0 {
				t.Fatalf("closed reader retained %d bytes", budget.inUse())
			}
		})
	}
}

func TestStreamFrameFixedLeaseSlots(t *testing.T) {
	budget := newFrameBudget(1 << 20)
	for i := 0; i < 32; i++ {
		if !budget.reserve(1) {
			t.Fatalf("slot %d refused", i)
		}
	}
	if budget.reserve(1) {
		t.Fatal("fixed lease storage grew beyond its capacity")
	}
	for i := 0; i < 32; i++ {
		budget.release(1)
	}
	if budget.inUse() != 0 {
		t.Fatalf("released slots retained %d bytes", budget.inUse())
	}
	if !budget.reserve(2) {
		t.Fatal("released slot was not reusable")
	}
	budget.release(2)
}

func TestStreamFrameAdmittedAllowanceBound(t *testing.T) {
	budget := newFrameBudget(200)
	root := budget.scope.budget
	if err := budget.admit(context.Background(), 100); err != nil {
		t.Fatal(err)
	}
	defer budget.releasePreparation()
	if err := budget.prepare(context.Background(), 10); err != nil {
		t.Fatal(err)
	}
	if !budget.reserve(10) || !budget.reserve(90) {
		t.Fatal("admitted allocations were refused")
	}
	if budget.reserve(1) {
		t.Fatal("frame allocations exceeded their own lease despite spare root capacity")
	}
	bufferAssertTotals(t, root, 100, 1, 0)
	if err := root.reconfigure(1, 0); err != nil {
		t.Fatal(err)
	}
	budget.releasePreparation()
	budget.release(90)
	if !budget.reserve(90) || budget.reserve(1) {
		t.Fatal("decrease changed the admitted envelope")
	}
	bufferAssertTotals(t, root, 100, 1, 0)
	budget.release(90)
	bufferAssertTotals(t, root, 100, 1, 0)
	budget.release(10)
	bufferAssertTotals(t, root, 0, 0, 0)
}

func TestStreamFramePeerInterruptionPreserved(t *testing.T) {
	for reason, frames := range streamInterruptionFramesByReason {
		for _, protocol := range []streamProtocol{streamProtocolSSE, streamProtocolNDJSON} {
			name := "SSE"
			if protocol == streamProtocolNDJSON {
				name = "NDJSON"
			}
			t.Run(reason+"/"+name, func(t *testing.T) {
				payload := frames.sse
				if protocol == streamProtocolNDJSON {
					payload = frames.ndjson
				}
				var calls int
				r, err := newStreamFrameReader(streamFrameConfig{
					Budget:   newFrameBudget(64 << 20),
					Protocol: protocol, PeerNormalized: true,
					Upstream: newChunkedStreamBody(payload, 1),
					Extract:  func([]byte) performanceExtractMetadata { calls++; return performanceExtractMetadata{} },
				})
				if err != nil {
					t.Fatal(err)
				}
				defer r.Close()
				got, err := io.ReadAll(r)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, payload) || calls != 0 {
					t.Fatalf("peer terminator changed: got=%q calls=%d", got, calls)
				}
				assertOutcome(t, r.Result(), streamOutcomeInterrupted, reason)
			})
		}
	}
}

func mustReadFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/streamframe/" + name)
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	if len(data) == 0 {
		t.Fatalf("fixture %s is empty", name)
	}
	return data
}

func wantSSEInterruption(reason string) []byte {
	return []byte("data: {\"error\":{\"message\":\"stream interrupted: " + reason +
		"\",\"type\":\"server_error\",\"code\":\"stream_interrupted\"}}\n\ndata: [DONE]\n\n")
}

func wantNDJSONInterruption(reason string) []byte {
	return []byte("{\"error\":\"stream interrupted: " + reason + "\"}\n")
}

// chunkedStreamBody hands out a fixed script of byte chunks and then reports
// clean EOF. size <= 0 delivers the whole input as one chunk.
type chunkedStreamBody struct {
	chunks [][]byte
	closed atomic.Bool
}

func newChunkedStreamBody(data []byte, size int) *chunkedStreamBody {
	var chunks [][]byte
	if size <= 0 {
		chunks = [][]byte{data}
	} else {
		for len(data) > 0 {
			n := size
			if n > len(data) {
				n = len(data)
			}
			chunks = append(chunks, data[:n])
			data = data[n:]
		}
	}
	return &chunkedStreamBody{chunks: chunks}
}

func (b *chunkedStreamBody) Read(p []byte) (int, error) {
	if b.closed.Load() {
		return 0, io.ErrClosedPipe
	}
	if len(b.chunks) == 0 {
		return 0, io.EOF
	}
	chunk := b.chunks[0]
	n := copy(p, chunk)
	if n < len(chunk) {
		b.chunks[0] = chunk[n:]
	} else {
		b.chunks = b.chunks[1:]
	}
	return n, nil
}

func (b *chunkedStreamBody) Close() error {
	b.closed.Store(true)
	return nil
}

// blockingStreamBody produces nothing and releases its reader only when closed,
// which is the shape of an engine that stopped emitting between events.
type blockingStreamBody struct {
	closed    chan struct{}
	started   chan struct{}
	startOnce sync.Once
	closeOnce sync.Once
	closes    atomic.Int32
}

func newBlockingStreamBody() *blockingStreamBody {
	return &blockingStreamBody{closed: make(chan struct{}), started: make(chan struct{})}
}

func (b *blockingStreamBody) Read(_ []byte) (int, error) {
	b.startOnce.Do(func() { close(b.started) })
	<-b.closed
	return 0, io.ErrClosedPipe
}

func (b *blockingStreamBody) Close() error {
	b.closeOnce.Do(func() { close(b.closed) })
	b.closes.Add(1)
	return nil
}

func (b *blockingStreamBody) closeCount() int {
	return int(b.closes.Load())
}

// dripStreamBody emits timed steps and then blocks. It is how a partial byte
// drip is distinguished from a complete event: the drips keep arriving, but no
// event finishes, so the idle bound must still fire.
type dripStreamBody struct {
	mu        sync.Mutex
	steps     []dripStep
	closed    chan struct{}
	closeOnce sync.Once
}

type dripStep struct {
	delay time.Duration
	data  []byte
}

func newDripStreamBody(steps ...dripStep) *dripStreamBody {
	return &dripStreamBody{steps: steps, closed: make(chan struct{})}
}

func (b *dripStreamBody) Read(p []byte) (int, error) {
	b.mu.Lock()
	if len(b.steps) == 0 {
		b.mu.Unlock()
		<-b.closed
		return 0, io.ErrClosedPipe
	}
	step := b.steps[0]
	b.steps = b.steps[1:]
	b.mu.Unlock()

	if step.delay > 0 {
		timer := time.NewTimer(step.delay)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-b.closed:
			return 0, io.ErrClosedPipe
		}
	}
	n := copy(p, step.data)
	if n < len(step.data) {
		b.mu.Lock()
		b.steps = append([]dripStep{{data: step.data[n:]}}, b.steps...)
		b.mu.Unlock()
	}
	return n, nil
}

func (b *dripStreamBody) Close() error {
	b.closeOnce.Do(func() { close(b.closed) })
	return nil
}

type streamFrameCase struct {
	protocol streamProtocol
	upstream io.ReadCloser
	idle     time.Duration
	budget   *frameBudget
	choices  int
	parent   context.Context
}

func openStreamFrameReader(t *testing.T, c streamFrameCase) *streamFrameReader {
	t.Helper()
	if c.budget == nil {
		c.budget = newFrameBudget(64 << 20)
	}
	idle := c.idle
	if idle == 0 {
		idle = time.Second
	}
	reader, err := newStreamFrameReader(streamFrameConfig{
		Protocol:    c.protocol,
		Upstream:    c.upstream,
		Parent:      c.parent,
		IdleTimeout: idle,
		Budget:      c.budget,
		Choices:     c.choices,
	})
	if err != nil {
		t.Fatalf("new stream frame reader: %v", err)
	}
	return reader
}

func runStreamFrame(t *testing.T, c streamFrameCase) ([]byte, streamFrameResult) {
	t.Helper()
	reader := openStreamFrameReader(t, c)
	out, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read stream: %v", err)
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("close stream reader: %v", err)
	}
	select {
	case <-reader.stopped:
	default:
		t.Fatal("Close returned before watcher joined")
	}
	return out, reader.Result()
}

type asyncStreamRead struct {
	out    []byte
	result streamFrameResult
	err    error
}

func readStreamAsync(reader *streamFrameReader) <-chan asyncStreamRead {
	done := make(chan asyncStreamRead, 1)
	go func() {
		out, err := io.ReadAll(reader)
		done <- asyncStreamRead{out: out, result: reader.Result(), err: err}
	}()
	return done
}

func awaitStreamRead(t *testing.T, reader *streamFrameReader, bound time.Duration) asyncStreamRead {
	t.Helper()
	select {
	case got := <-readStreamAsync(reader):
		if got.err != nil {
			t.Fatalf("read stream: %v", got.err)
		}
		return got
	case <-time.After(bound):
		t.Fatalf("stream read did not terminate within %s", bound)
		return asyncStreamRead{}
	}
}

func assertOutcome(t *testing.T, got streamFrameResult, want streamOutcome, wantReason string) {
	t.Helper()
	if got.Outcome != want {
		t.Fatalf("outcome = %s, want %s", got.Outcome, want)
	}
	if got.Reason != wantReason {
		t.Fatalf("reason = %q, want %q", got.Reason, wantReason)
	}
}

type decodedSSEError struct {
	Message string
	Type    string
	Code    string
}

// decodeFinalSSEError parses the last data event that is not the [DONE]
// terminator, so the assertion is on fields rather than on raw text.
func decodeFinalSSEError(t *testing.T, out []byte) decodedSSEError {
	t.Helper()
	var payload []byte
	for _, frame := range bytes.Split(out, []byte("\n\n")) {
		frame = bytes.TrimSpace(frame)
		if !bytes.HasPrefix(frame, []byte("data: ")) {
			continue
		}
		body := frame[len("data: "):]
		if bytes.Equal(body, []byte("[DONE]")) {
			continue
		}
		payload = body
	}
	if payload == nil {
		t.Fatalf("no non-terminal data event in %q", out)
	}
	var event struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(payload, &event); err != nil {
		t.Fatalf("decode error event %q: %v", payload, err)
	}
	return decodedSSEError{Message: event.Error.Message, Type: event.Error.Type, Code: event.Error.Code}
}

func assertSSEInterruption(t *testing.T, out []byte, reason string) {
	t.Helper()
	if !bytes.HasSuffix(out, wantSSEInterruption(reason)) {
		t.Fatalf("stream does not end with the SSE interruption frame for %q:\n%q", reason, out)
	}
	if got := bytes.Count(out, []byte("[DONE]")); got != 1 {
		t.Fatalf("stream carries %d [DONE] terminators, want exactly 1", got)
	}
	if got := bytes.Count(out, []byte("stream_interrupted")); got != 1 {
		t.Fatalf("stream carries %d interruption markers, want exactly 1", got)
	}
	event := decodeFinalSSEError(t, out)
	if event.Message != "stream interrupted: "+reason {
		t.Fatalf("error message = %q, want %q", event.Message, "stream interrupted: "+reason)
	}
	if event.Type != "server_error" || event.Code != "stream_interrupted" {
		t.Fatalf("error type/code = %q/%q, want server_error/stream_interrupted", event.Type, event.Code)
	}
}

func decodeFinalNDJSONError(t *testing.T, out []byte) string {
	t.Helper()
	lines := bytes.Split(bytes.TrimRight(out, "\n"), []byte("\n"))
	if len(lines) == 0 {
		t.Fatalf("no NDJSON line in %q", out)
	}
	var object struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(lines[len(lines)-1], &object); err != nil {
		t.Fatalf("decode final NDJSON object %q: %v", lines[len(lines)-1], err)
	}
	return object.Error
}

func assertNDJSONInterruption(t *testing.T, out []byte, reason string) {
	t.Helper()
	if !bytes.HasSuffix(out, wantNDJSONInterruption(reason)) {
		t.Fatalf("stream does not end with the NDJSON interruption object for %q:\n%q", reason, out)
	}
	if got := bytes.Count(out, []byte("stream interrupted")); got != 1 {
		t.Fatalf("stream carries %d interruption objects, want exactly 1", got)
	}
	if message := decodeFinalNDJSONError(t, out); message != "stream interrupted: "+reason {
		t.Fatalf("error message = %q, want %q", message, "stream interrupted: "+reason)
	}
}

func TestStreamFrameOpenAICompleteStream(t *testing.T) {
	content := mustReadFixture(t, "openai-complete.txt")

	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody(content, 0)})

	if !bytes.Equal(out, content) {
		t.Fatalf("forwarded bytes differ from the upstream stream:\ngot  %q\nwant %q", out, content)
	}
	assertOutcome(t, result, streamOutcomeCompleted, "")
}

func TestStreamFrameOpenAITornStreamDropsFragment(t *testing.T) {
	content := mustReadFixture(t, "openai-torn.txt")
	tornAt := bytes.Index(content, []byte(tornOpenAIMarker))
	if tornAt < 0 {
		t.Fatalf("fixture %q does not carry the torn marker", content)
	}
	frameStart := bytes.LastIndex(content[:tornAt], []byte("data: "))
	if frameStart < 0 {
		t.Fatalf("torn frame has no data line in %q", content)
	}
	want := append(append([]byte{}, content[:frameStart]...), wantSSEInterruption(streamInterruptReasonPrematureEOF)...)

	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody(content, 0)})

	if !bytes.Equal(out, want) {
		t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", out, want)
	}
	if bytes.Contains(out, []byte(tornOpenAIMarker)) {
		t.Fatalf("torn fragment reached the client: %q", out)
	}
	assertSSEInterruption(t, out, streamInterruptReasonPrematureEOF)
	assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonPrematureEOF)
}

func TestStreamFrameOllamaCompleteStream(t *testing.T) {
	content := mustReadFixture(t, "ollama-complete.txt")

	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolNDJSON, upstream: newChunkedStreamBody(content, 0)})

	if !bytes.Equal(out, content) {
		t.Fatalf("forwarded bytes differ from the upstream stream:\ngot  %q\nwant %q", out, content)
	}
	assertOutcome(t, result, streamOutcomeCompleted, "")
}

func TestStreamFrameOllamaTornStreamDropsFragment(t *testing.T) {
	content := mustReadFixture(t, "ollama-torn.txt")
	tornAt := bytes.Index(content, []byte(tornOllamaMarker))
	if tornAt < 0 {
		t.Fatalf("fixture %q does not carry the torn marker", content)
	}
	lineStart := bytes.LastIndexByte(content[:tornAt], '\n') + 1
	want := append(append([]byte{}, content[:lineStart]...), wantNDJSONInterruption(streamInterruptReasonPrematureEOF)...)

	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolNDJSON, upstream: newChunkedStreamBody(content, 0)})

	if !bytes.Equal(out, want) {
		t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", out, want)
	}
	if bytes.Contains(out, []byte(tornOllamaMarker)) {
		t.Fatalf("torn fragment reached the client: %q", out)
	}
	assertNDJSONInterruption(t, out, streamInterruptReasonPrematureEOF)
	assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonPrematureEOF)
}

func TestStreamFrameArbitraryReadSplits(t *testing.T) {
	content := mustReadFixture(t, "openai-complete.txt")

	for _, size := range []int{1, 2, 3, 7, 64} {
		t.Run("chunk/"+strconv.Itoa(size), func(t *testing.T) {
			out, result := runStreamFrame(t, streamFrameCase{
				protocol: streamProtocolSSE,
				upstream: newChunkedStreamBody(content, size),
			})
			if !bytes.Equal(out, content) {
				t.Fatalf("split size %d changed the forwarded bytes:\ngot  %q\nwant %q", size, out, content)
			}
			assertOutcome(t, result, streamOutcomeCompleted, "")
		})
	}
}

func TestStreamFrameUTF8SplitAcrossReads(t *testing.T) {
	content := []byte("data: {\"id\":\"u\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"héllo → 世界\"},\"finish_reason\":null}]}\n\n" +
		"data: [DONE]\n\n")

	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody(content, 1)})

	if !bytes.Equal(out, content) {
		t.Fatalf("byte-at-a-time reads changed the forwarded bytes:\ngot  %q\nwant %q", out, content)
	}
	assertOutcome(t, result, streamOutcomeCompleted, "")
}

func TestStreamFrameCRLFAndComments(t *testing.T) {
	content := []byte(": heartbeat\r\n\r\n" +
		"data: {\"id\":\"crlf\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\r\n\r\n" +
		"data: [DONE]\r\n\r\n")

	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody(content, 5)})

	if !bytes.Equal(out, content) {
		t.Fatalf("CRLF framing changed the forwarded bytes:\ngot  %q\nwant %q", out, content)
	}
	assertOutcome(t, result, streamOutcomeCompleted, "")
}

func TestStreamFrameMultilineDataEvent(t *testing.T) {
	content := []byte(": ping\n\n" +
		"data: {\"id\":\"multi\",\n" +
		"data: \"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n" +
		"data: [DONE]\n\n")

	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody(content, 3)})

	if !bytes.Equal(out, content) {
		t.Fatalf("multiline data changed the forwarded bytes:\ngot  %q\nwant %q", out, content)
	}
	assertOutcome(t, result, streamOutcomeCompleted, "")
}

func TestStreamFrameFinishReasonCompletion(t *testing.T) {
	singleChoice := []byte("data: {\"id\":\"fr\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n")

	t.Run("all requested choices finished at clean eof", func(t *testing.T) {
		out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody(singleChoice, 0)})

		if !bytes.Equal(out, singleChoice) {
			t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", out, singleChoice)
		}
		assertOutcome(t, result, streamOutcomeCompleted, "")
	})

	t.Run("first of two choices does not complete the stream", func(t *testing.T) {
		partial := []byte("data: {\"id\":\"fr\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\n")
		out, result := runStreamFrame(t, streamFrameCase{
			protocol: streamProtocolSSE,
			upstream: newChunkedStreamBody(partial, 0),
			choices:  2,
		})

		want := append(append([]byte{}, partial...), wantSSEInterruption(streamInterruptReasonPrematureEOF)...)
		if !bytes.Equal(out, want) {
			t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", out, want)
		}
		assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonPrematureEOF)
	})

	t.Run("both of two choices finished completes the stream", func(t *testing.T) {
		both := []byte("data: {\"id\":\"fr\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"},{\"index\":1,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		out, result := runStreamFrame(t, streamFrameCase{
			protocol: streamProtocolSSE,
			upstream: newChunkedStreamBody(both, 0),
			choices:  2,
		})

		if !bytes.Equal(out, both) {
			t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", out, both)
		}
		assertOutcome(t, result, streamOutcomeCompleted, "")
	})
}

func TestStreamFrameFinalNDJSONObjectWithoutNewline(t *testing.T) {
	first := "{\"model\":\"qwen2.5\",\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n"
	final := "{\"model\":\"qwen2.5\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}"

	t.Run("final object carries the done marker", func(t *testing.T) {
		content := []byte(first + final)
		out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolNDJSON, upstream: newChunkedStreamBody(content, 0)})

		if !bytes.Equal(out, content) {
			t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", out, content)
		}
		assertOutcome(t, result, streamOutcomeCompleted, "")
	})

	t.Run("final object without the done marker is interrupted", func(t *testing.T) {
		unfinished := "{\"model\":\"qwen2.5\",\"message\":{\"role\":\"assistant\",\"content\":\"more\"},\"done\":false}"
		content := []byte(first + unfinished)
		want := append(append([]byte{}, content...), '\n')
		want = append(want, wantNDJSONInterruption(streamInterruptReasonPrematureEOF)...)

		out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolNDJSON, upstream: newChunkedStreamBody(content, 0)})

		if !bytes.Equal(out, want) {
			t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", out, want)
		}
		assertNDJSONInterruption(t, out, streamInterruptReasonPrematureEOF)
		assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonPrematureEOF)
	})
}

func TestStreamFrameUpstreamProtocolErrorIsFailure(t *testing.T) {
	t.Run("sse error event", func(t *testing.T) {
		good := "data: {\"id\":\"err\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n"
		bad := "data: {\"id\":\"err\",\"error\":{\"message\":\"engine exploded\",\"type\":\"server_error\"}}\n\n"
		out, result := runStreamFrame(t, streamFrameCase{
			protocol: streamProtocolSSE,
			upstream: newChunkedStreamBody([]byte(good+bad), 0),
		})

		if !bytes.Equal(out, []byte(good+bad)) {
			t.Fatalf("upstream error frame was not preserved:\ngot  %q\nwant %q", out, good+bad)
		}
		if got := bytes.Count(out, []byte("stream_interrupted")); got != 0 {
			t.Fatalf("synthetic interruption frame was appended to an upstream error: %q", out)
		}
		if got := bytes.Count(out, []byte("[DONE]")); got != 0 {
			t.Fatalf("synthetic [DONE] was appended to an upstream error: %q", out)
		}
		assertOutcome(t, result, streamOutcomeUpstreamError, streamInterruptReasonUpstreamError)
		if bytes.Contains([]byte(result.Reason), []byte("engine exploded")) {
			t.Fatalf("result reason carries upstream error text: %q", result.Reason)
		}
	})

	t.Run("ndjson error object", func(t *testing.T) {
		good := "{\"model\":\"qwen2.5\",\"message\":{\"content\":\"a\"},\"done\":false}\n"
		bad := "{\"error\":\"engine exploded\"}\n"
		out, result := runStreamFrame(t, streamFrameCase{
			protocol: streamProtocolNDJSON,
			upstream: newChunkedStreamBody([]byte(good+bad), 0),
		})

		if !bytes.Equal(out, []byte(good+bad)) {
			t.Fatalf("upstream error object was not preserved:\ngot  %q\nwant %q", out, good+bad)
		}
		if got := bytes.Count(out, []byte("stream interrupted")); got != 0 {
			t.Fatalf("synthetic interruption object was appended to an upstream error: %q", out)
		}
		assertOutcome(t, result, streamOutcomeUpstreamError, streamInterruptReasonUpstreamError)
	})
}

func TestStreamFrameMalformedFrameIsInterrupted(t *testing.T) {
	t.Run("sse payload is not json", func(t *testing.T) {
		good := "data: {\"id\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n"
		bad := "data: MALFORMED-SSE-PAYLOAD\n\n"
		want := append([]byte(good), wantSSEInterruption(streamInterruptReasonMalformed)...)

		out, result := runStreamFrame(t, streamFrameCase{
			protocol: streamProtocolSSE,
			upstream: newChunkedStreamBody([]byte(good+bad), 0),
		})

		if !bytes.Equal(out, want) {
			t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", out, want)
		}
		if bytes.Contains(out, []byte("MALFORMED-SSE-PAYLOAD")) {
			t.Fatalf("malformed frame reached the client: %q", out)
		}
		assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonMalformed)
	})

	t.Run("ndjson line is not json", func(t *testing.T) {
		good := "{\"model\":\"qwen2.5\",\"message\":{\"content\":\"ok\"},\"done\":false}\n"
		bad := "MALFORMED-NDJSON-PAYLOAD\n"
		want := append([]byte(good), wantNDJSONInterruption(streamInterruptReasonMalformed)...)

		out, result := runStreamFrame(t, streamFrameCase{
			protocol: streamProtocolNDJSON,
			upstream: newChunkedStreamBody([]byte(good+bad), 0),
		})

		if !bytes.Equal(out, want) {
			t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", out, want)
		}
		if bytes.Contains(out, []byte("MALFORMED-NDJSON-PAYLOAD")) {
			t.Fatalf("malformed frame reached the client: %q", out)
		}
		assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonMalformed)
	})
}

func TestStreamFrameIdleTimeoutClosesStream(t *testing.T) {
	synctest.Test(t, testStreamFrameIdleTimeoutClosesStream)
}

func testStreamFrameIdleTimeoutClosesStream(t *testing.T) {
	body := newBlockingStreamBody()
	t.Cleanup(func() { _ = body.Close() })
	parent, cancelParent := context.WithCancel(context.Background())
	defer cancelParent()

	reader := openStreamFrameReader(t, streamFrameCase{
		protocol: streamProtocolSSE,
		upstream: body,
		idle:     80 * time.Millisecond,
		parent:   parent,
	})
	t.Cleanup(func() { _ = reader.Close() })

	started := time.Now()
	got := awaitStreamRead(t, reader, 5*time.Second)
	elapsed := time.Since(started)

	if !bytes.Equal(got.out, wantSSEInterruption(streamInterruptReasonIdle)) {
		t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", got.out, wantSSEInterruption(streamInterruptReasonIdle))
	}
	assertOutcome(t, got.result, streamOutcomeInterrupted, streamInterruptReasonIdle)
	if elapsed > 2*time.Second {
		t.Fatalf("idle bound took %s, want well under the test bound", elapsed)
	}
	if parent.Err() != nil {
		t.Fatalf("idle timeout cancelled the parent request context: %v", parent.Err())
	}
	if body.closeCount() == 0 {
		t.Fatal("idle timeout did not close the upstream body")
	}
}

func TestStreamFrameIdleRefreshesOnlyOnCompleteEvents(t *testing.T) {
	synctest.Test(t, testStreamFrameIdleRefreshesOnlyOnCompleteEvents)
}

func testStreamFrameIdleRefreshesOnlyOnCompleteEvents(t *testing.T) {
	complete := []byte("data: {\"id\":\"drip\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n")

	steps := []dripStep{{data: complete}}
	drip := []byte("partial-frame-bytes-without-a-delimiter")
	for i := 0; i < 40; i++ {
		steps = append(steps, dripStep{delay: 10 * time.Millisecond, data: drip[:2]})
	}
	body := newDripStreamBody(steps...)
	t.Cleanup(func() { _ = body.Close() })

	const idle = 100 * time.Millisecond
	reader := openStreamFrameReader(t, streamFrameCase{protocol: streamProtocolSSE, upstream: body, idle: idle})
	t.Cleanup(func() { _ = reader.Close() })

	started := time.Now()
	got := awaitStreamRead(t, reader, 5*time.Second)
	elapsed := time.Since(started)

	want := append(append([]byte{}, complete...), wantSSEInterruption(streamInterruptReasonIdle)...)
	if !bytes.Equal(got.out, want) {
		t.Fatalf("forwarded bytes differ:\ngot  %q\nwant %q", got.out, want)
	}
	assertOutcome(t, got.result, streamOutcomeInterrupted, streamInterruptReasonIdle)
	if elapsed > idle {
		t.Fatalf("partial byte drips extended the idle bound: %s elapsed with a %s bound", elapsed, idle)
	}
}

func TestStreamFrameCancellationStopsReadAndTimer(t *testing.T) {
	synctest.Test(t, testStreamFrameCancellationStopsReadAndTimer)
}

func testStreamFrameCancellationStopsReadAndTimer(t *testing.T) {
	body := newBlockingStreamBody()
	t.Cleanup(func() { _ = body.Close() })
	parent, cancelParent := context.WithCancel(context.Background())

	reader := openStreamFrameReader(t, streamFrameCase{
		protocol: streamProtocolSSE,
		upstream: body,
		idle:     5 * time.Second,
		parent:   parent,
	})
	t.Cleanup(func() { _ = reader.Close() })

	done := readStreamAsync(reader)
	time.AfterFunc(20*time.Millisecond, cancelParent)

	select {
	case got := <-done:
		if got.err != nil {
			t.Fatalf("read cancelled stream: %v", got.err)
		}
		if len(got.out) != 0 {
			t.Fatalf("cancelled stream forwarded %q, want nothing", got.out)
		}
		assertOutcome(t, got.result, streamOutcomeCancelled, "")
	case <-time.After(5 * time.Second):
		t.Fatal("cancellation did not unblock the reader")
	}
	if body.closeCount() == 0 {
		t.Fatal("cancellation did not close the upstream body")
	}
}

func TestStreamFrameFrameOverflowInterrupts(t *testing.T) {
	oversized := bytes.Repeat([]byte("x"), int(streamFrameMaxBytes)+1)
	body := newChunkedStreamBody(oversized, 32<<10)
	t.Cleanup(func() { _ = body.Close() })

	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: body})

	if !bytes.Equal(out, wantSSEInterruption(streamInterruptReasonFrameTooLarge)) {
		t.Fatalf("forwarded bytes differ:\ngot  %q\nwant the frame-limit interruption", out)
	}
	assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonFrameTooLarge)
}

func TestStreamFrameBudgetExhaustionTerminates(t *testing.T) {
	budget := newFrameBudget(8 << 10)
	body := newChunkedStreamBody(bytes.Repeat([]byte("y"), 12<<10), 4<<10)
	t.Cleanup(func() { _ = body.Close() })

	reader := openStreamFrameReader(t, streamFrameCase{protocol: streamProtocolSSE, upstream: body, budget: budget})
	out, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read stream: %v", err)
	}
	result := reader.Result()
	if err := reader.Close(); err != nil {
		t.Fatalf("close stream reader: %v", err)
	}

	if !bytes.Equal(out, wantSSEInterruption(streamInterruptReasonBudget)) {
		t.Fatalf("forwarded bytes differ:\ngot  %q\nwant the budget interruption", out)
	}
	assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonBudget)
	if used := budget.inUse(); used != 0 {
		t.Fatalf("frame budget still holds %d bytes after the reader closed", used)
	}
}

func TestStreamFrameCloseReleasesFrameBuffer(t *testing.T) {
	content := mustReadFixture(t, "openai-complete.txt")
	budget := newFrameBudget(1 << 20)
	body := newChunkedStreamBody(content, 0)

	reader := openStreamFrameReader(t, streamFrameCase{protocol: streamProtocolSSE, upstream: body, budget: budget})

	one := make([]byte, 1)
	if _, err := reader.Read(one); err != nil {
		t.Fatalf("read one byte: %v", err)
	}
	if budget.inUse() == 0 {
		t.Fatal("frame buffer was not reserved while frames were held")
	}

	rest, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read stream: %v", err)
	}
	if !bytes.Equal(append(one, rest...), content) {
		t.Fatalf("byte-at-a-time delivery changed the forwarded bytes")
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("close stream reader: %v", err)
	}
	if used := budget.inUse(); used != 0 {
		t.Fatalf("frame budget still holds %d bytes after the reader closed", used)
	}
}

func TestStreamFrameRejectsNonObjectPayloads(t *testing.T) {
	for _, payload := range []string{"", "null", "[]", "true", "42", `"text"`} {
		t.Run("sse "+payload, func(t *testing.T) {
			out, result := runStreamFrame(t, streamFrameCase{
				protocol: streamProtocolSSE,
				upstream: newChunkedStreamBody([]byte("data: "+payload+"\n\n"), 1),
			})
			if !bytes.Equal(out, wantSSEInterruption(streamInterruptReasonMalformed)) {
				t.Fatalf("non-object SSE reached client: %q", out)
			}
			assertSSEInterruption(t, out, streamInterruptReasonMalformed)
			assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonMalformed)
		})
		t.Run("ndjson "+payload, func(t *testing.T) {
			out, result := runStreamFrame(t, streamFrameCase{
				protocol: streamProtocolNDJSON,
				upstream: newChunkedStreamBody([]byte(payload+"\n"), 1),
			})
			if !bytes.Equal(out, wantNDJSONInterruption(streamInterruptReasonMalformed)) {
				t.Fatalf("non-object NDJSON reached client: %q", out)
			}
			assertNDJSONInterruption(t, out, streamInterruptReasonMalformed)
			assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonMalformed)
		})
	}
}

func TestStreamFrameInvalidUTF8IsDropped(t *testing.T) {
	test := func(name string, protocol streamProtocol, frame []byte, want []byte) {
		t.Run(name, func(t *testing.T) {
			out, result := runStreamFrame(t, streamFrameCase{protocol: protocol, upstream: newChunkedStreamBody(frame, 1)})
			if !bytes.Equal(out, want) {
				t.Fatalf("invalid UTF-8 frame reached client: %q", out)
			}
			assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonMalformed)
		})
	}
	test("sse", streamProtocolSSE, []byte("data: {\"text\":\"\xff\"}\n\n"), wantSSEInterruption(streamInterruptReasonMalformed))
	test("ndjson", streamProtocolNDJSON, []byte("{\"text\":\"\xff\"}\n"), wantNDJSONInterruption(streamInterruptReasonMalformed))
}

func TestStreamFrameCompleteFrameSizeBoundary(t *testing.T) {
	test := func(name string, protocol streamProtocol, prefix, suffix, terminal string) {
		t.Run(name, func(t *testing.T) {
			frame := append([]byte(prefix), bytes.Repeat([]byte("x"), int(streamFrameMaxBytes)-len(prefix)-len(suffix))...)
			frame = append(frame, suffix...)
			t.Run("exact limit accepted", func(t *testing.T) {
				content := append(append([]byte{}, frame...), terminal...)
				out, result := runStreamFrame(t, streamFrameCase{protocol: protocol, upstream: newChunkedStreamBody(content, 997)})
				if !bytes.Equal(out, content) {
					t.Fatalf("exact-limit frame not preserved: got %d bytes, want %d", len(out), len(content))
				}
				assertOutcome(t, result, streamOutcomeCompleted, "")
			})
			t.Run("one byte over limit dropped", func(t *testing.T) {
				oversized := append([]byte(prefix), bytes.Repeat([]byte("x"), int(streamFrameMaxBytes)+1-len(prefix)-len(suffix))...)
				oversized = append(oversized, suffix...)
				out, result := runStreamFrame(t, streamFrameCase{protocol: protocol, upstream: newChunkedStreamBody(oversized, 997)})
				want := wantSSEInterruption(streamInterruptReasonFrameTooLarge)
				if protocol == streamProtocolNDJSON {
					want = wantNDJSONInterruption(streamInterruptReasonFrameTooLarge)
				}
				if !bytes.Equal(out, want) {
					t.Fatalf("oversized complete frame reached client: %d bytes", len(out))
				}
				assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonFrameTooLarge)
			})
		})
	}
	test("sse", streamProtocolSSE, `data: {"text":"`, "\"}\n\n", "data: [DONE]\n\n")
	test("ndjson", streamProtocolNDJSON, `{"response":"`, "\",\"done\":false}\n", "{\"done\":true}\n")
}

func TestStreamFrameHeartbeatRefreshesIdle(t *testing.T) {
	synctest.Test(t, testStreamFrameHeartbeatRefreshesIdle)
}

func testStreamFrameHeartbeatRefreshesIdle(t *testing.T) {
	const idle = 300 * time.Millisecond
	heartbeat := []byte(": heartbeat\r\n\r\n")
	terminal := []byte("data: [DONE]\n\n")
	steps := []dripStep{
		{delay: 100 * time.Millisecond, data: heartbeat},
		{delay: 100 * time.Millisecond, data: heartbeat},
		{delay: 100 * time.Millisecond, data: heartbeat},
		{delay: 100 * time.Millisecond, data: terminal},
	}
	body := newDripStreamBody(steps...)
	reader := openStreamFrameReader(t, streamFrameCase{protocol: streamProtocolSSE, upstream: body, idle: idle})
	t.Cleanup(func() { _ = reader.Close() })
	got := awaitStreamRead(t, reader, 5*time.Second)
	want := append(bytes.Repeat(heartbeat, 3), terminal...)
	if !bytes.Equal(got.out, want) {
		t.Fatalf("complete heartbeat did not refresh idle: got %q, want %q", got.out, want)
	}
	assertOutcome(t, got.result, streamOutcomeCompleted, "")
}

func TestStreamFrameSharedBudgetSaturation(t *testing.T) {
	budget := newFrameBudget(64 << 10)
	content := []byte("data: {\"choices\":[]}\n\ndata: [DONE]\n\n")
	first := openStreamFrameReader(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody(content, 0), budget: budget})
	t.Cleanup(func() { _ = first.Close() })
	one := make([]byte, 1)
	if n, err := first.Read(one); err != nil || n != 1 {
		t.Fatalf("hold first reader frame: n=%d, err=%v", n, err)
	}
	held := budget.inUse()
	if held <= 0 || held > budget.limit {
		t.Fatalf("invalid live reservation: %d", held)
	}
	padding := budget.limit - held
	if !budget.reserve(padding) {
		t.Fatal("could not saturate remaining shared budget")
	}
	second, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Upstream: newChunkedStreamBody(content, 0), Budget: budget})
	if second != nil || err != errStreamFrameBudget {
		t.Fatalf("detached reader allocated beyond saturated bootstrap: reader=%v err=%v", second != nil, err)
	}
	if got := budget.inUse(); got != budget.limit {
		t.Fatalf("failed reader changed existing reservations: got %d, want %d", got, budget.limit)
	}
	budget.release(padding)
	if err := first.Close(); err != nil {
		t.Fatalf("close first reader: %v", err)
	}
	if used := budget.inUse(); used != 0 {
		t.Fatalf("closed readers retained %d reserved bytes", used)
	}
}

func TestStreamFrameDefaults(t *testing.T) {
	body := newChunkedStreamBody([]byte("data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n"), 0)
	budget := newFrameBudget(64 << 20)
	reader, err := newStreamFrameReader(streamFrameConfig{Protocol: streamProtocolSSE, Upstream: body, Budget: budget})
	if err != nil {
		t.Fatalf("create default reader: %v", err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	if reader.idleTimeout != 60*time.Second || defaultStreamFrameIdle != 60*time.Second {
		t.Fatalf("idle default = %s, want 60s", reader.idleTimeout)
	}
	if reader.budget != budget || budget.limit != 64<<20 {
		t.Fatal("reader does not use the supplied 64 MiB frame budget")
	}
	if streamFrameMaxBytes != 1<<20 {
		t.Fatalf("frame limit = %d, want 1 MiB", streamFrameMaxBytes)
	}
	if _, err := io.ReadAll(reader); err != nil {
		t.Fatalf("read default reader: %v", err)
	}
	assertOutcome(t, reader.Result(), streamOutcomeCompleted, "")
}

func TestStreamFramePreCancelledParentSuppressesTerminal(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	cancel()
	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody([]byte("data: [DONE]\n\n"), 0), parent: parent})
	if len(out) != 0 {
		t.Fatalf("pre-cancelled reader emitted %q", out)
	}
	assertOutcome(t, result, streamOutcomeCancelled, "")
}

func TestStreamFrameCloseUnblocksRead(t *testing.T) {
	body := newBlockingStreamBody()
	reader := openStreamFrameReader(t, streamFrameCase{protocol: streamProtocolSSE, upstream: body, idle: time.Hour})
	t.Cleanup(func() { _ = reader.Close() })
	done := readStreamAsync(reader)
	select {
	case <-body.started:
	case <-time.After(5 * time.Second):
		t.Fatal("upstream read did not start")
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("close blocked reader: %v", err)
	}
	select {
	case got := <-done:
		if got.err != nil || len(got.out) != 0 {
			t.Fatalf("closed read: bytes=%q, err=%v", got.out, got.err)
		}
		assertOutcome(t, got.result, streamOutcomeCancelled, "")
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not join blocked read")
	}
	if body.closeCount() == 0 {
		t.Fatal("Close did not close upstream body")
	}
}

type finalErrorStreamBody struct {
	data []byte
	err  error
}

func (b *finalErrorStreamBody) Read(p []byte) (int, error) {
	if len(b.data) == 0 {
		return 0, b.err
	}
	n := copy(p, b.data)
	b.data = b.data[n:]
	if len(b.data) == 0 {
		return n, b.err
	}
	return n, nil
}

func (b *finalErrorStreamBody) Close() error { return nil }

func TestStreamFrameTerminalBytesWithReadError(t *testing.T) {
	test := func(name string, protocol streamProtocol, content string, readError error) {
		t.Run(name, func(t *testing.T) {
			out, result := runStreamFrame(t, streamFrameCase{protocol: protocol, upstream: &finalErrorStreamBody{data: []byte(content), err: readError}})
			if !bytes.Equal(out, []byte(content)) {
				t.Fatalf("definitive terminal bytes changed: %q", out)
			}
			assertOutcome(t, result, streamOutcomeCompleted, "")
		})
	}
	test("sse done with eof", streamProtocolSSE, "data: [DONE]\n\n", io.EOF)
	test("sse done with transport error", streamProtocolSSE, "data: [DONE]\n\n", io.ErrUnexpectedEOF)
	test("ndjson done with eof", streamProtocolNDJSON, "{\"done\":true}\n", io.EOF)
	test("ndjson done with transport error", streamProtocolNDJSON, "{\"done\":true}\n", io.ErrUnexpectedEOF)
}

func TestStreamFrameFinishReasonRequiresCleanEOF(t *testing.T) {
	content := []byte("data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n")
	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: &finalErrorStreamBody{data: content, err: io.ErrUnexpectedEOF}})
	want := append(append([]byte{}, content...), wantSSEInterruption(streamInterruptReasonReadFailed)...)
	if !bytes.Equal(out, want) {
		t.Fatalf("transport failure became successful finish: %q", out)
	}
	assertSSEInterruption(t, out, streamInterruptReasonReadFailed)
	assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonReadFailed)
}

func TestStreamFrameGrowthReservesBothAllocations(t *testing.T) {
	budget := newFrameBudget(12<<10 - 1)
	reader := &streamFrameReader{budget: budget}
	if !reader.growLocked(1) {
		t.Fatal("initial frame allocation refused")
	}
	defer reader.releaseBufferLocked()
	if reader.growLocked(4<<10 + 1) {
		t.Fatal("growth bypassed the combined old and new allocation limit")
	}
	if used := budget.inUse(); used != 4<<10 {
		t.Fatalf("failed growth changed live reservation: %d", used)
	}
}

func TestStreamFrameCRDelimiterAtCleanEOF(t *testing.T) {
	content := []byte("data: [DONE]\r\r")
	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody(content, 1)})
	if !bytes.Equal(out, content) {
		t.Fatalf("CR-delimited final event changed: %q", out)
	}
	assertOutcome(t, result, streamOutcomeCompleted, "")
}

func TestStreamFrameMalformedFinishMarkers(t *testing.T) {
	for _, choice := range []string{
		`{"finish_reason":"stop"}`,
		`{"index":null,"finish_reason":"stop"}`,
		`{"index":0,"finish_reason":""}`,
		`{"index":0,"finish_reason":true}`,
		`{"index":2,"finish_reason":"stop"}`,
	} {
		t.Run(choice, func(t *testing.T) {
			out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody([]byte("data: {\"choices\":["+choice+"]}\n\n"), 1)})
			assertSSEInterruption(t, out, streamInterruptReasonMalformed)
			assertOutcome(t, result, streamOutcomeInterrupted, streamInterruptReasonMalformed)
		})
	}
}

func TestStreamFrameJSONFieldScanning(t *testing.T) {
	content := []byte("data: {\"ignored\":{\"escaped\":\"x\\\"},]y\",\"nested\":[{},null,true]},\"choi\\u0063es\":[{\"index\":0,\"finish_reason\":\"stop\"}]}\n\n")
	out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolSSE, upstream: newChunkedStreamBody(content, 1)})
	if !bytes.Equal(out, content) {
		t.Fatalf("nested JSON frame changed: %q", out)
	}
	assertOutcome(t, result, streamOutcomeCompleted, "")
}

func TestStreamFrameOllamaArbitraryReadSplits(t *testing.T) {
	content := []byte("{\"response\":\"héllo → 世界\",\"done\":false}\r\n{\"done\":true}\r\n")
	for _, size := range []int{1, 2, 3, 7, 64} {
		t.Run("chunk/"+strconv.Itoa(size), func(t *testing.T) {
			out, result := runStreamFrame(t, streamFrameCase{protocol: streamProtocolNDJSON, upstream: newChunkedStreamBody(content, size)})
			if !bytes.Equal(out, content) {
				t.Fatalf("split NDJSON changed raw bytes: %q", out)
			}
			assertOutcome(t, result, streamOutcomeCompleted, "")
		})
	}
}
