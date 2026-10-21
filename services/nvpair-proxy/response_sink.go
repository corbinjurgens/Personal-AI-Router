// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
)

// responseSink owns delivery only. Acceptance, commitment and cancellation
// decisions belong to the request owner; cancel merely reports a failed client.
// Its mutex protects metadata and is never held during client I/O or callbacks.
type responseSink struct {
	w                                                               http.ResponseWriter
	cancel                                                          context.CancelFunc
	mu                                                              sync.Mutex
	status                                                          int
	written, flushed, produced, genuine, synthetic                  int64
	writeErr                                                        error
	streamFinished, protocolComplete, endingPlanned, flushSucceeded bool
	reason                                                          string
	continuationUnavailable                                         bool
	body                                                            responseInterruptionBody
	bootstrap                                                       *bufferLease
}

type responseSinkResult struct {
	Status                               int
	Written, Flushed, Genuine, Synthetic int64
	WriteError                           error
	ProtocolComplete, DeliveryComplete   bool
	Reason                               string
	ContinuationUnavailable              bool
}

type responseSinkContextKey struct{}

func newResponseSink(w http.ResponseWriter, cancel context.CancelFunc) *responseSink {
	return &responseSink{w: w, cancel: cancel}
}

func withResponseSink(ctx context.Context, sink *responseSink) context.Context {
	// The retained 96 KiB parsing/control envelope includes this fixed sink
	// and its context adapter. Join transport borrowers before owner.close.
	if owner := inferenceBuffers(ctx); owner != nil && sink.bootstrap == nil {
		sink.bootstrap = owner.borrowParserAllowance()
		if sink.bootstrap != nil {
			owner.retainRelease(sink.close)
		}
	}
	return context.WithValue(ctx, responseSinkContextKey{}, sink)
}

func (s *responseSink) close() {
	if s.bootstrap != nil {
		s.bootstrap.close()
		s.bootstrap = nil
	}
}

func responseSinkFrom(ctx context.Context) *responseSink {
	if ctx == nil {
		return nil
	}
	sink, _ := ctx.Value(responseSinkContextKey{}).(*responseSink)
	return sink
}

func (s *responseSink) Header() http.Header         { return s.w.Header() }
func (s *responseSink) Unwrap() http.ResponseWriter { return s.w }

func (s *responseSink) WriteHeader(code int) {
	s.mu.Lock()
	if s.status != 0 {
		s.mu.Unlock()
		return
	}
	s.status = code
	s.mu.Unlock()
	s.w.WriteHeader(code)
	// Header and trailer share this internal channel. Once headers are sent,
	// remove the header value so net/http cannot repeat it as a trailer value.
	s.w.Header().Del(peerStreamFramingHeader)
}

func (s *responseSink) Write(p []byte) (int, error) {
	s.WriteHeader(http.StatusOK)
	s.mu.Lock()
	// Parser output is announced before Write. Unannounced writes are
	// request-generated control output, including waiting heartbeats/errors.
	if missing := s.written + int64(len(p)) - s.produced; missing > 0 {
		s.produced += missing
		s.synthetic += missing
	}
	s.mu.Unlock()
	n, err := s.w.Write(p)
	if err == nil && n != len(p) {
		err = io.ErrShortWrite
	}
	s.mu.Lock()
	s.written += int64(n)
	if err != nil && s.writeErr == nil {
		s.writeErr = err
	}
	s.mu.Unlock()
	if err != nil && s.cancel != nil {
		s.cancel()
	}
	return n, err
}

func (s *responseSink) FlushError() error {
	err := http.NewResponseController(s.w).Flush()
	s.mu.Lock()
	if err == nil {
		s.flushed = s.written
		s.flushSucceeded = true
	}
	if err != nil && !errors.Is(err, http.ErrNotSupported) && s.writeErr == nil {
		s.writeErr = err
	}
	s.mu.Unlock()
	if err != nil && !errors.Is(err, http.ErrNotSupported) && s.cancel != nil {
		s.cancel()
	}
	return err
}

func (s *responseSink) output(n int, genuine bool) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.produced += int64(n)
	if genuine {
		s.genuine += int64(n)
	} else {
		s.synthetic += int64(n)
	}
	s.mu.Unlock()
}

func (s *responseSink) streamResult(result streamFrameResult) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.streamFinished = result.Outcome != streamOutcomeCancelled && result.Outcome != streamOutcomeStreaming
	if result.Outcome == streamOutcomeCompleted || result.Outcome == streamOutcomeUpstreamError {
		s.endingPlanned = true
	}
	s.protocolComplete = result.Outcome == streamOutcomeCompleted
	if result.Reason != "" {
		s.reason = result.Reason
	}
	s.mu.Unlock()
}

func (s *responseSink) result() responseSinkResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	return responseSinkResult{Status: s.status, Written: s.written, Flushed: s.flushed,
		Genuine: s.genuine, Synthetic: s.synthetic, WriteError: s.writeErr,
		ProtocolComplete: s.protocolComplete,
		DeliveryComplete: s.streamFinished && s.status != 0 && s.flushSucceeded && s.writeErr == nil && s.written == s.produced && s.flushed >= s.produced,
		Reason:           s.reason, ContinuationUnavailable: s.continuationUnavailable}
}

// planInterruption records SC's cause before selecting the only available
// disposition. There is no qualified continuation executor in this cutover.
func (s *responseSink) planInterruption(reason string) bool {
	if s == nil {
		return true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.planInterruptionLocked(reason, nil)
}

// Admission locks sink metadata before the request owner. The request hook
// acquires only its own mutex and performs no callback or sink access.
func (s *responseSink) planInterruptionForAttempt(reason string, attempt *publicAttemptLifecycle) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if attempt == nil {
		return false
	}
	return s.planInterruptionLocked(reason, attempt)
}

// planInterruptionLocked requires s.mu; a supplied attempt admits the ending
// under the existing request mutex before any new sink plan is recorded.
func (s *responseSink) planInterruptionLocked(reason string, attempt *publicAttemptLifecycle) bool {
	if _, ok := streamInterruptionFramesByReason[reason]; !ok {
		reason = streamInterruptReasonReadFailed
	}
	if attempt != nil && !attempt.admitInterruption(reason) {
		return false
	}
	if !s.endingPlanned {
		s.endingPlanned = true
		s.reason = reason
		s.continuationUnavailable = true
		return true
	}
	return false
}

func interruptionBytes(protocol streamProtocol, reason string) []byte {
	frames, ok := streamInterruptionFramesByReason[reason]
	if !ok {
		frames = streamInterruptionFramesByReason[streamInterruptReasonReadFailed]
	}
	if protocol == streamProtocolNDJSON {
		return frames.ndjson
	}
	return frames.sse
}

// interruptedBody uses fixed request-owned state and initialization-owned
// immutable templates even if the parser's bootstrap allowance was refused.
func (s *responseSink) interruptedBody(protocol streamProtocol, reason string) io.ReadCloser {
	s.planInterruption(reason)
	s.body = responseInterruptionBody{sink: s, data: interruptionBytes(protocol, reason)}
	return &s.body
}

func (s *responseSink) interruptedBodyForAttempt(protocol streamProtocol, reason string, attempt *publicAttemptLifecycle) io.ReadCloser {
	if !s.planInterruptionForAttempt(reason, attempt) {
		return nil
	}
	s.body = responseInterruptionBody{sink: s, data: interruptionBytes(protocol, reason)}
	return &s.body
}

type responseInterruptionBody struct {
	sink *responseSink
	data []byte
	off  int
}

func (b *responseInterruptionBody) Read(p []byte) (int, error) {
	if b.off == len(b.data) {
		b.sink.streamResult(streamFrameResult{Outcome: streamOutcomeInterrupted, Reason: b.sink.result().Reason})
		return 0, io.EOF
	}
	n := copy(p, b.data[b.off:])
	b.off += n
	b.sink.output(n, false)
	return n, nil
}
func (b *responseInterruptionBody) Close() error { return nil }

type sinkObservedBody struct {
	io.ReadCloser
	sink     *responseSink
	response *http.Response
	peer     originStreamRequest
}

func (b *sinkObservedBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if n > 0 {
		b.sink.output(n, true)
	}
	if err == io.EOF {
		if b.peer.peerNormalized {
			reason, metadataErr := b.peer.withPeerTrailer(b.response.Trailer.Values(peerStreamFramingHeader))
			if metadataErr != nil || reason != "" {
				b.sink.streamResult(streamFrameResult{Outcome: streamOutcomeInterrupted, Reason: streamInterruptReasonPeerLink})
				if metadataErr != nil {
					return n, metadataErr
				}
				return n, errors.New("peer nonstream response interrupted")
			}
		}
		b.sink.streamResult(streamFrameResult{Outcome: streamOutcomeCompleted})
	}
	return n, err
}

func (s *responseSink) interrupt(protocol streamProtocol, reason string) error {
	if err := s.result().WriteError; err != nil {
		return err
	}
	if !s.planInterruption(reason) {
		return s.result().WriteError
	}
	data := interruptionBytes(protocol, reason)
	s.output(len(data), false)
	if _, err := s.Write(data); err != nil {
		return err
	}
	if err := s.FlushError(); err != nil {
		return err
	}
	s.streamResult(streamFrameResult{Outcome: streamOutcomeInterrupted, Reason: reason})
	return nil
}

func (s *responseSink) writePublicFrame(frame []byte) error {
	s.output(len(frame), !bytes.Equal(bytes.TrimSpace(frame), []byte("data: [DONE]")))
	if _, err := s.Write(frame); err != nil {
		return err
	}
	return s.FlushError()
}

// The destination trailer is authenticated by the existing pinned peer hop.
// The framing header is still parsed separately by withPeerMetadata. Engines
// cannot supply a trusted disposition: ingress deletes their values first.
func (s originStreamRequest) withPeerTrailer(values []string) (string, error) {
	invalid := &peerFramingMetadataError{invalid: true}
	if len(values) == 0 {
		return "", &peerFramingMetadataError{}
	}
	if !s.peerNormalized || len(values) != 1 || len(values[0]) > 128 {
		return "", invalid
	}
	if values[0] == "v1;completed" {
		return "", nil
	}
	const prefix = "v1;interrupted;"
	if !strings.HasPrefix(values[0], prefix) {
		return "", invalid
	}
	reason := strings.TrimPrefix(values[0], prefix)
	if _, ok := streamInterruptionFramesByReason[reason]; !ok {
		return "", invalid
	}
	return reason, nil
}

func peerDisposition(result streamFrameResult) string {
	if result.Outcome == streamOutcomeCompleted {
		return "v1;completed"
	}
	if _, ok := streamInterruptionFramesByReason[result.Reason]; ok {
		return "v1;interrupted;" + result.Reason
	}
	return ""
}

type destinationFramingBody struct {
	io.ReadCloser
	response *http.Response
	reader   *streamFrameReader
	sink     *responseSink
	eof      bool
}

func destinationResponseFraming(resp *http.Response, body io.ReadCloser, request originStreamRequest, reader *streamFrameReader, sink *responseSink) io.ReadCloser {
	if request.protocol == streamProtocolUnknown {
		return body
	}
	return &destinationFramingBody{ReadCloser: body, response: resp, reader: reader, sink: sink}
}

func (b *destinationFramingBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err == io.EOF {
		b.eof = true
	}
	return n, err
}

func (b *destinationFramingBody) Close() error {
	err := b.ReadCloser.Close()
	// The underlying originResponseBody deliberately strips engine trailers.
	// Only after it closes do we publish our own bounded disposition.
	b.response.Trailer.Del(peerStreamFramingHeader)
	if b.eof {
		if b.response.Trailer == nil {
			b.response.Trailer = make(http.Header)
		}
		result := streamFrameResult{Outcome: streamOutcomeCompleted}
		if b.reader != nil {
			result = b.reader.Result()
		} else if b.sink != nil {
			if r := b.sink.result(); r.Reason != "" {
				result.Outcome, result.Reason = streamOutcomeInterrupted, r.Reason
			}
		}
		if value := peerDisposition(result); value != "" {
			b.response.Trailer.Set(peerStreamFramingHeader, value)
		}
	}
	return err
}
