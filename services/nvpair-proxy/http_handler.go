// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	stderrors "errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"net/http/httputil"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"nvpair-shared/cors"
)

type statusCapture struct {
	http.ResponseWriter
	status int

	// idle bounds how long a single write of streamed bytes to the client may
	// block before it's abandoned. Zero disables the deadline. See
	// idleClientWriteTimeout for the rationale (killed-client / half-open
	// socket zombie jobs).
	idle time.Duration
	rc   *http.ResponseController
	// wroteErr retains the first error returned when writing the response body
	// to the client (e.g. a dead client's send buffer filling and the write
	// deadline tripping), so handleHTTP can mark the workload failed rather
	// than misreporting the truncated stream as completed.
	wroteErr error
	// upstreamBody is installed only for the accepted response. Its read
	// result distinguishes a full copy from a discarded or interrupted body.
	upstreamBody *originResponseBody

	// upstreamAlive is called after each successful body write, but only once the
	// upstream has committed — handleHTTP sets it in ModifyResponse, so it stays
	// nil while the only thing this writer could carry is the proxy's own error
	// body. Past that point every byte written came from the node serving the
	// request, which is proof that node is working: the liveness evidence
	// discovery cannot obtain for itself while the node is too busy to answer a
	// probe. Called on the reverse proxy's copy goroutine, so it must be cheap.
	upstreamAlive func()
}

// Unwrap exposes the underlying ResponseWriter so http.ResponseController can
// reach the connection for SetWriteDeadline (and Flush) through this wrapper.
func (sc *statusCapture) Unwrap() http.ResponseWriter { return sc.ResponseWriter }

func (sc *statusCapture) WriteHeader(code int) {
	sc.status = code
	sc.ResponseWriter.WriteHeader(code)
}

// Write bounds each streamed write to the client with a deadline so a write
// blocked on a dead/half-open client fails promptly instead of hanging the
// reverse-proxy copy indefinitely. The deadline is cleared after every
// successful write, so a legitimately slow generation with long gaps between
// tokens is never penalized — only a write actively stuck on a gone client
// trips it. The first write error is retained (wroteErr) for the caller.
func (sc *statusCapture) Write(b []byte) (int, error) {
	if sc.idle > 0 {
		if sc.rc == nil {
			sc.rc = http.NewResponseController(sc.ResponseWriter)
		}
		_ = sc.rc.SetWriteDeadline(time.Now().Add(sc.idle))
	}
	n, err := sc.ResponseWriter.Write(b)
	if err != nil {
		if sc.wroteErr == nil {
			sc.wroteErr = err
		}
	} else if sc.rc != nil {
		_ = sc.rc.SetWriteDeadline(time.Time{})
	}
	// Reported on every chunk rather than once per response so a long generation
	// keeps vouching for its node for as long as it streams. The reporter
	// coalesces, so the cost of calling this per chunk is a mutex and a clock
	// read.
	if err == nil && sc.upstreamAlive != nil {
		sc.upstreamAlive()
	}
	return n, err
}

// FlushError makes the streamed flush deadline-aware. For a streaming
// (chunked) upstream, ReverseProxy flushes after every write via
// http.NewResponseController(w).Flush — and because a small chunk buffers on
// Write without touching the socket, the actual network write for it happens
// here in Flush, not in Write. Without this method that flush reaches the
// underlying connection through Unwrap with no deadline and blocks unbounded on
// a stalled client (the same zombie the Write deadline guards against). So arm
// the same idle deadline around the flush, clear it on success, and retain a
// real flush error so the workload is classified failed. Implementing
// FlushError (which also satisfies the Flusher path via the ResponseController)
// means the flush routes through here instead of unwrapping past us.
func (sc *statusCapture) FlushError() error {
	if sc.rc == nil {
		sc.rc = http.NewResponseController(sc.ResponseWriter)
	}
	if sc.idle > 0 {
		_ = sc.rc.SetWriteDeadline(time.Now().Add(sc.idle))
	}
	err := sc.rc.Flush()
	if err != nil {
		// A ResponseWriter that genuinely can't flush is not a client failure;
		// only retain real I/O errors (e.g. the deadline tripping on a dead
		// client) so we don't misreport an unsupported-flush as a failed write.
		if !stderrors.Is(err, http.ErrNotSupported) && sc.wroteErr == nil {
			sc.wroteErr = err
		}
		return err
	}
	if sc.idle > 0 {
		_ = sc.rc.SetWriteDeadline(time.Time{})
	}
	return nil
}

// Retry bounds for a single inference request.
//
// The two are not redundant, because they bound different things. An attempt is
// consumed by a dispatch, never by a resolution: finding no eligible owner
// sends nothing, so maxDispatchAttempts governs dispatch failures while
// jobDeadline governs the wait for an owner to exist at all — a wait that costs
// no attempts and would otherwise be unbounded.
//
// Read only against dispatch failures the deadline looks unreachable, and the
// arithmetic says so: five attempts at a 120s first-content cap plus 15s of
// cumulative backoff span 615s, so every reschedule check before the last sees
// a clock under 600s. It is not dead — it is the only bound on the no-owner
// wait. Do not remove it as unreachable.
const maxDispatchAttempts = 5

// jobDeadline is a var (not a const) only so a test can shorten it; production
// never reassigns it.
var jobDeadline = 10 * time.Minute

// targetWatchInterval is how often an in-flight attempt rechecks that its
// target is still in discovery. It bounds detection latency only: the effect is
// to replace a full first-content budget spent waiting on a node that is
// already gone with a sub-second abort.
//
// A var (not a const) only so a test can shorten it; production never
// reassigns it.
var targetWatchInterval = 500 * time.Millisecond

// defaultRetryBackoff is the delay before a retry, indexed by the number of
// dispatches already made and clamped to the last entry. It also paces the poll
// while no owner is available.
var defaultRetryBackoff = []time.Duration{time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second}

// retryBackoff is the schedule in force. A var (not a const) only so a test can
// shorten it; production never reassigns it. Naming the default separately lets
// a test that asserts the schedule itself put the real one back.
var retryBackoff = defaultRetryBackoff

// backoffFor returns the delay before the next dispatch, jittered and never
// longer than the time left before the deadline.
func backoffFor(dispatches int, remaining time.Duration) time.Duration {
	if len(retryBackoff) == 0 || remaining <= 0 {
		return 0
	}
	i := dispatches - 1
	if i < 0 {
		i = 0
	}
	if i >= len(retryBackoff) {
		i = len(retryBackoff) - 1
	}
	d := retryBackoff[i]
	// Jitter by up to ±25% so a burst of requests that all failed against the
	// same node does not retry in lockstep and re-collide. The clock's
	// nanosecond low bits are the entropy source: they differ between
	// concurrent handlers, and `rand` here is crypto/rand, whose interface is
	// far heavier than spreading a backoff warrants.
	if spread := int64(d) / 2; spread > 0 {
		d = time.Duration(int64(d) - spread/2 + time.Now().UnixNano()%(spread+1))
	}
	if d > remaining {
		d = remaining
	}
	return d
}

// waitBeforeRetry sleeps d, reporting false if the request was abandoned while
// waiting. A client that has gone away should not keep a retry budget alive:
// nobody is left to receive the answer, so the remaining attempts are better
// spent on work someone is waiting for.
func waitBeforeRetry(ctx context.Context, d time.Duration) bool {
	if ctx.Err() != nil {
		return false
	}
	if d <= 0 {
		return true
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// errFirstBodyTimeout is returned by awaitFirstBody when an upstream sent its
// headers but produced no content within the budget.
var errFirstBodyTimeout = stderrors.New("upstream sent no response body within the first-byte budget")

// errTargetGone is the reason an attempt was abandoned because its target left
// discovery. It replaces the bare "context canceled" the transport reports,
// which in a log is indistinguishable from the client hanging up.
var errTargetGone = stderrors.New("target node left discovery mid-request")

// bodyWithPeek re-attaches an already-read first byte ahead of the rest of an
// upstream body, while keeping Close bound to the original so the connection
// is still released.
type bodyWithPeek struct {
	io.ReadCloser
	first []byte
	err   error
}

func (b *bodyWithPeek) Read(p []byte) (int, error) {
	if len(b.first) != 0 {
		n := copy(p, b.first)
		b.first = b.first[n:]
		if len(b.first) == 0 {
			return n, b.err
		}
		return n, nil
	}
	if b.err != nil {
		return 0, b.err
	}
	return b.ReadCloser.Read(p)
}

// awaitFirstBody blocks until the upstream produces the first byte of its
// response body, and returns that byte spliced back onto the front of the
// stream.
//
// It exists because response headers are not a reliable signal that an engine
// has started work, and the difference is engine-specific. LM Studio answers a
// streaming request with 200 and headers ~12ms after accepting it and then
// holds the connection open, silent, for the entire time the request waits
// behind other work — measured at 128s behind a single-slot generation. Ollama
// withholds headers until generation begins. Committing on headers therefore
// meant that on LM Studio the proxy bound itself to a node that had merely
// accepted the request: the header timeout could never fire, failover was
// impossible, and a node dying during the wait took the job with it.
//
// The first content byte is the signal both engines agree on, and it is a real
// one — no SSE keepalive or empty role delta arrives during the wait, so the
// byte means generation actually started.
//
// A read that ends in EOF with no bytes is an empty body, which is a complete
// (if unusual) response and commits. A genuine read error before any content
// is retryable, like any other pre-commit upstream failure.
func awaitFirstBody(body io.ReadCloser, budget time.Duration) (io.ReadCloser, error) {
	type peeked struct {
		b   []byte
		err error
	}
	raw, _ := body.(*routingResponseBody)
	var borrower *bufferLease
	if raw != nil && raw.attempt.request.sink != nil && raw.attempt.request.sink.bootstrap != nil {
		var err error
		borrower, err = raw.attempt.request.sink.bootstrap.borrow()
		if err != nil {
			return nil, errStreamFrameBudget
		}
	}
	ch := make(chan peeked)
	discard := make(chan struct{})
	done := make(chan struct{})
	if raw != nil {
		raw.attempt.request.mu.Lock()
		raw.attempt.request.firstBodyDone = done
		raw.attempt.request.mu.Unlock()
	}
	go func() {
		defer func() {
			if borrower != nil {
				borrower.close()
			}
			close(done)
		}()
		var one [1]byte
		n, err := body.Read(one[:])
		// Transfer the byte to the live handler or discard it before releasing
		// its charge. No buffered channel can retain an abandoned peek.
		select {
		case ch <- peeked{b: one[:n], err: err}:
		case <-discard:
		}
	}()

	timer := time.NewTimer(budget)
	defer timer.Stop()
	select {
	case p := <-ch:
		<-done // No second reader can overlap this peek's body Read.
		if len(p.b) == 0 && p.err != nil && !stderrors.Is(p.err, io.EOF) {
			return nil, p.err
		}
		return &bodyWithPeek{ReadCloser: body, first: p.b, err: p.err}, nil
	case <-timer.C:
		close(discard)
		committed := raw != nil && raw.attempt.expireFirstBody()
		_ = body.Close()
		if !committed {
			// The timeout won at the request owner. Close and join the
			// real raw read before another attempt can borrow its resources.
			<-done
		}
		// Commitment proves the underlying Read already returned. Only its
		// timer-stop callback can remain; its byte charge and registry entry
		// stay live until done, without another reader or upstream body.
		return nil, errFirstBodyTimeout
	}
}

// idleClientWriteTimeout bounds how long a single write of streamed response
// bytes to the client may block. A killed client can leave a half-open socket
// whose kernel send buffer fills and never drains; without this deadline the
// reverse-proxy copy blocks indefinitely (TCP retransmit backoff runs into
// minutes, and r.Context() never fires when no FIN/RST arrives), so the request
// handler never returns and its terminal workload event is never emitted — the
// "zombie job" left showing as running until PAIR restarts. statusCapture.Write
// resets the deadline after every successful write, so this only trips a write
// that is actively stuck on a gone client, never a slow-but-live generation.
//
// It is a var (not a const) only so a test can shorten it to exercise the
// deadline against a real socket; production never reassigns it.
var idleClientWriteTimeout = 30 * time.Second

// firstBodyTimeout bounds how long awaitFirstBody waits for an engine to emit
// its first content byte. It matches proxyResponseTimeout, the budget the
// transport applies to headers, because from the caller's point of view the two
// bound the same thing: how long we wait for a node to start the work. Keeping
// it generous is deliberate — it also covers a cold model load, which is real
// work rather than a stall, and which cannot be told apart from a queue wait
// from outside the engine.
//
// It is a var (not a const) only so a test can shorten it; production never
// reassigns it.
var firstBodyTimeout = proxyResponseTimeout

// retrySignal is returned from ModifyResponse to abort a retryable upstream
// response before its body streams to the client, so handleHTTP can fail over
// to the next candidate. It's a distinct type rather than errors.New(...)
// because this package aliases nvpair-shared/errors as `errors` (which has no New).
type retrySignal struct{}

func (retrySignal) Error() string { return "nvpair-proxy: retry next candidate" }

// originStreamRequest selects framing only for the supported inference routes.
// It reads the stream flag without re-encoding any part of the request.
type originStreamRequest struct {
	protocol       streamProtocol
	declared       bool
	disabled       bool
	choices        int
	peerNormalized bool
}

// Peer ingress reports the effective body it dispatched, never origin policy
// or engine-supplied metadata. This header is consumed only on pinned peer hops.
const peerStreamFramingHeader = "X-PAIR-Stream-Framing"

type peerFramingMetadataError struct{ invalid bool }

func (e *peerFramingMetadataError) Error() string {
	if e.invalid {
		return "peer framing metadata invalid (version-incompatible peer)"
	}
	return "peer framing metadata missing (version-incompatible peer)"
}

func (s originStreamRequest) peerMetadata() string {
	mode := "auto"
	if s.declared {
		mode = "declared"
	} else if s.disabled {
		mode = "disabled"
	}
	return "v1;" + mode + ";" + strconv.Itoa(s.choices)
}

func (s originStreamRequest) withPeerMetadata(values []string) (originStreamRequest, error) {
	if len(values) == 0 {
		return s, &peerFramingMetadataError{}
	}
	invalid := &peerFramingMetadataError{invalid: true}
	if len(values) != 1 {
		return s, invalid
	}
	parts := strings.Split(values[0], ";")
	if len(parts) != 3 || parts[0] != "v1" {
		return s, invalid
	}
	n, err := strconv.Atoi(parts[2])
	if err != nil || n <= 0 || strconv.Itoa(n) != parts[2] {
		return s, invalid
	}
	s.declared, s.disabled, s.choices = false, false, n
	switch parts[1] {
	case "declared":
		s.declared = true
	case "disabled":
		s.disabled = true
	case "auto":
		if s.protocol != streamProtocolSSE {
			return s, invalid
		}
	default:
		return s, invalid
	}
	s.peerNormalized = true
	return s, nil
}

func parseOriginStreamRequest(path string, body []byte) (originStreamRequest, error) {
	s := originStreamRequest{choices: 1}
	switch path {
	case "/v1/chat/completions", "/v1/completions":
		s.protocol = streamProtocolSSE
	case "/api/chat", "/api/generate":
		s.protocol, s.declared = streamProtocolNDJSON, true
	default:
		return s, nil
	}
	// Invalid JSON remains the engine's responsibility. A stream member in a
	// valid object, however, must be an unambiguous JSON boolean or null.
	if !ownedJSONValid(body) {
		return s, nil
	}
	seen := false
	var invalid error
	visitObject(body, func(key []byte, start, end int) {
		value := body[start:end]
		if jsonKeyEqual(key, "stream", false) {
			if seen || (string(value) != "true" && string(value) != "false" && string(value) != "null") {
				invalid = fmt.Errorf("stream must be a single boolean or null member")
				return
			}
			seen = true
			if string(value) != "null" {
				s.declared = string(value) == "true"
				s.disabled = !s.declared
			}
		}
		if jsonKeyEqual(key, "n", false) {
			if n, ok := positiveJSONInt(value); ok {
				s.choices = n
			}
		}
	})
	return s, invalid
}

func (s originStreamRequest) manages(contentType string) (bool, bool) {
	media, _, _ := mime.ParseMediaType(contentType)
	validMedia := media == "text/event-stream"
	if s.protocol == streamProtocolNDJSON {
		validMedia = media == "application/x-ndjson" || media == "application/json"
	}
	return s.protocol != streamProtocolUnknown && !s.disabled &&
		(s.declared || (s.protocol == streamProtocolSSE && validMedia)), validMedia
}

func clearOriginStreamIntegrity(header http.Header) {
	for _, name := range []string{"Content-Length", "Content-MD5", "Digest", "Content-Digest", "Repr-Digest", "ETag", "Content-Encoding"} {
		header.Del(name)
	}
}

// Keep the proxy's trailer map separate from the transport's response. Closing
// an interrupted body can start a background drain that still writes trailers.
// Only a read through EOF makes the upstream trailers safe to copy.
type proxyResponseTransport struct{ http.RoundTripper }

func (t proxyResponseTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	upstream, err := t.RoundTripper.RoundTrip(r)
	if err != nil {
		return upstream, err
	}
	if upstream.StatusCode == http.StatusSwitchingProtocols {
		return upstream, nil // Preserve the bidirectional upgrade body.
	}
	response := *upstream
	response.Trailer = upstream.Trailer.Clone()
	response.Body = &proxyResponseTrailerBody{ReadCloser: upstream.Body, upstream: upstream, response: &response}
	return &response, nil
}

type proxyResponseTrailerBody struct {
	io.ReadCloser
	upstream *http.Response
	response *http.Response
}

func (b *proxyResponseTrailerBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err == io.EOF {
		b.response.Trailer = b.upstream.Trailer.Clone()
	}
	return n, err
}

type originResponseBody struct {
	io.ReadCloser
	response *http.Response
	managed  bool
	complete bool
}

func (b *originResponseBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.complete = err == io.EOF
	return n, err
}

func (b *originResponseBody) Close() error {
	err := b.ReadCloser.Close()
	// Transport populates trailers at EOF, after ModifyResponse. ReverseProxy
	// closes the body before copying those trailers to the client.
	b.response.Trailer.Del(peerStreamFramingHeader)
	if b.managed {
		clearOriginStreamIntegrity(b.response.Trailer)
	}
	return err
}

func (f *facade) handleHTTP(w http.ResponseWriter, r *http.Request) {
	f.handleHTTPWithStreamIdle(w, r, defaultStreamFrameIdle)
}

// The idle interval belongs to this request's reader, never a mutable global.
func (f *facade) handleHTTPWithStreamIdle(w http.ResponseWriter, r *http.Request, streamIdle time.Duration) {
	p := f.host
	start := time.Now()
	isInf := isInferenceRequest(f.profile, r.Method, r.URL.Path)

	// Allocate the request ID up front so both code paths (rejection
	// and forward) can stamp the same value into their notification.
	// The rejection path never emits a Started event, so its ID won't
	// appear in any orchestrator in-flight map — that's fine; the
	// completion event still bumps the failed counter regardless of
	// whether a matching Started was seen.
	reqID := strconv.FormatUint(f.nextRequestID.Add(1), 10)
	if !isInf {
		f.serveProxyPassthrough(w, r, start, reqID)
		return
	}

	// The request's own context. The client leaving still cancels it, and so
	// can workload/cancel and cancelActive, each naming why in cancelReason so
	// the terminal workload event does not blame a client that never left.
	reqCtx, cancelReqCtx := context.WithCancel(r.Context())
	defer cancelReqCtx()
	r = r.WithContext(reqCtx)
	var cancelReason atomic.Pointer[string]
	cancelRequest := func(reason string) {
		cancelReason.CompareAndSwap(nil, &reason)
		cancelReqCtx()
	}
	cancelReasonOr := func(def string) string {
		if reason := cancelReason.Load(); reason != nil {
			return *reason
		}
		return def
	}
	ctl := newRoutingRequest(cancelRequest)
	// One immutable pre-dispatch deadline also bounds initial owned storage.
	// The live request context remains independent so accepted streams drain.
	ctl.deadline = start.Add(jobDeadline)
	r = r.WithContext(context.WithValue(r.Context(), routingDeadlineContextKey{}, ctl.deadline))
	defer ctl.finish()
	stopCancellation := context.AfterFunc(reqCtx, func() {
		ctl.cancelWithReason(cancelReasonOr("client disconnected before completion"))
	})
	defer stopCancellation()
	var requestOwner *routingRequest
	if isInf {
		var buffers *ownedInferenceBuffers
		r, buffers = p.ownInferenceBuffers(r)
		w = &inferenceErrorWriter{ResponseWriter: w, ctx: r.Context()}
		// Keep cancellation discoverable until every final buffer borrower has
		// joined, including registration refusal, body errors and panic unwind.
		defer func() {
			ctl.mu.Lock()
			pending := ctl.firstBodyDone
			ctl.mu.Unlock()
			if pending != nil {
				select {
				case <-pending:
				default:
					// The committed peek has finished raw IO but its pure stop
					// callback still owns charged scratch. End this same owner
					// and detach its workload while retaining registry discovery.
					ctl.finish()
					p.unregisterWorkload(requestOwner)
					buffers.close()
					go func() {
						<-pending
						p.unregisterRoutingRequest(requestOwner)
					}()
					return
				}
			}
			buffers.close()
			p.unregisterRoutingRequest(requestOwner)
		}()
		var err error
		requestOwner, err = p.registerRoutingRequest(ctl)
		if err != nil {
			writeIngressError(w, http.StatusServiceUnavailable, "identity-unavailable", "routing request identity or capacity is unavailable")
			return
		}
	}
	// Parse the request's model before choosing a node. Model eligibility only
	// applies to inference routes; control endpoints retain their existing
	// routing behavior even when their JSON happens to contain a model field.
	bodyCtx, stopBodyPreparation := context.WithDeadline(r.Context(), ctl.deadline)
	bodyBytes, model, err := bufferInferenceBody(w, r.WithContext(bodyCtx), p.requestBodyLimit())
	stopBodyPreparation()
	if err != nil {
		status := bufferRequestStatus(err)
		reason := "request body read failed"
		if status == http.StatusRequestEntityTooLarge {
			reason = "request body too large"
		} else if status == http.StatusServiceUnavailable {
			reason = "inference buffer capacity is busy"
		}
		slog.Warn("proxy request rejected",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"remote", r.RemoteAddr, "reason", reason, "limit", p.requestBodyLimit())
		writeIngressError(w, status, "request-body", "request body could not be buffered")
		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Status:   status,
			Duration: time.Since(start).Milliseconds(),
			Error:    reason,
		})
		return
	}
	if err := validateRequestIdentifiers(r, bodyBytes); err != nil {
		writeIngressError(w, bufferRequestStatus(err), "request-identifier", err.Error())
		return
	}
	// An inference body naming "model" twice is ambiguous: admission and
	// workload tracking read the last member while the tier rewrite replaces
	// the first, so the model admitted here and the model the engine decodes
	// could disagree. rejectAmbiguous refuses such a body wherever this
	// handler would otherwise read it — the strict branch, and, after the
	// alias refusal keeps its precedence, the routing path — always before
	// candidates, admission or any upstream sees it. Control endpoints keep
	// their existing behavior.
	rejectAmbiguous := func() bool {
		if !isInf || !ambiguousModelMember(bodyBytes) {
			return false
		}
		slog.Warn("proxy request rejected",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"remote", r.RemoteAddr, "reason", "ambiguous model member")
		writeIngressError(w, http.StatusBadRequest, "ambiguous-model", ambiguousModelMessage)
		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Status:   http.StatusBadRequest,
			Duration: time.Since(start).Milliseconds(),
			Error:    "ambiguous model member",
		})
		return true
	}
	strict, strictErr := parseStrictAdmission(r.Header)
	if strictErr != nil {
		writeIngressError(w, http.StatusBadRequest, "strict-admission", strictErr.Error())
		return
	}
	if strict != nil {
		if !isInf {
			writeIngressError(w, http.StatusBadRequest, "strict-admission", "strict admission requires inference")
			return
		}
		if rejectAmbiguous() {
			return
		}
		// The internal strict envelope names destination-local execution. It
		// cannot grant onward routing or inherit concrete retry loops.
		f.serveStrictLocal(w, r, bodyBytes, model, *strict, false)
		return
	}
	// A reserved hybrid alias the legacy tier path does not serve is refused
	// here, before candidates, admission, reservation, workload or upstream.
	if alias, reason, rejected := requestAliasRejection(r, bodyBytes, model, isInf); rejected {
		slog.Warn("proxy request rejected",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"remote", r.RemoteAddr, "reason", reason, "alias", alias)
		writeAliasRejection(w, alias, reason)
		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Status:   http.StatusBadRequest,
			Duration: time.Since(start).Milliseconds(),
			Error:    reason,
		})
		return
	}
	// An alias-less ambiguous body is refused only now, so a reserved alias
	// keeps the precedence its own test pins.
	if rejectAmbiguous() {
		return
	}
	if isInf {
		_, err = parseOriginStreamRequest(r.URL.Path, bodyBytes)
		if err != nil {
			writeIngressError(w, http.StatusBadRequest, "invalid-stream", "stream must be a single boolean or null member")
			_ = f.notify("proxy/request", RequestEvent{
				ID: reqID, Method: r.Method, Path: r.URL.Path,
				Status: http.StatusBadRequest, Duration: time.Since(start).Milliseconds(),
				Error: "invalid stream flag",
			})
			return
		}
	}
	if isInf {
		ctl.sink = newResponseSink(w, func() { ctl.cancelWithReason("client connection lost") })
		w = ctl.sink
		r = r.WithContext(withResponseSink(r.Context(), ctl.sink))
	}
	ctl.dispatch(f, w, r, start, reqID, bodyBytes, model, streamIdle, cancelReasonOr, requestOwner)
}

// Storage cancellation alone cannot interrupt a socket-backed Body.Read (or
// its deferred Close). Bound that same body phase at the HTTP connection and
// join the interrupt before clearing its temporary read deadline. No body
// reader is detached, and the live request/response context is not replaced.
func bufferInferenceBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, string, error) {
	controller := http.NewResponseController(w)
	deadline, bounded := r.Context().Deadline()
	if err := controller.SetReadDeadline(deadline); err != nil && !stderrors.Is(err, http.ErrNotSupported) {
		return nil, "", &bufferBusyError{cause: err}
	}
	interrupted := make(chan struct{})
	stop := context.AfterFunc(r.Context(), func() {
		defer close(interrupted)
		_ = controller.SetReadDeadline(time.Now())
	})
	body, model, err := bufferBodyAndModel(w, r, limit)
	if !stop() {
		<-interrupted
	}
	clearErr := controller.SetReadDeadline(time.Time{})
	// Impossible own footprints retain their typed413 precedence.
	if bufferRequestStatus(err) == http.StatusRequestEntityTooLarge {
		return body, model, err
	}
	cause := r.Context().Err()
	if cause == nil && bounded && !time.Now().Before(deadline) {
		cause = context.DeadlineExceeded
	}
	if cause != nil {
		return nil, "", &bufferBusyError{cause: cause}
	}
	if err == nil && clearErr != nil && !stderrors.Is(clearErr, http.ErrNotSupported) {
		return nil, "", &bufferBusyError{cause: clearErr}
	}
	return body, model, err
}

func retryableProxyStatus(code int) bool {
	switch code {
	case http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	}
	return code >= http.StatusInternalServerError
}

// Control requests retain their single pass over concrete candidates. They do
// not register inference identities, acquire inference buffers or enter its
// request owner, admission, workload, performance or retry-deadline machinery.
func (f *facade) serveProxyPassthrough(w http.ResponseWriter, r *http.Request, start time.Time, reqID string) {
	p := f.host
	body, model, err := bufferBodyAndModel(w, r, p.requestBodyLimit())
	if err != nil {
		reason := "request body read failed"
		if bufferRequestStatus(err) == http.StatusRequestEntityTooLarge {
			reason = "request body too large"
		}
		writeIngressError(w, bufferRequestStatus(err), "request-body", "request body could not be buffered")
		_ = f.notify("proxy/request", RequestEvent{ID: reqID, Method: r.Method, Path: r.URL.Path,
			Status: bufferRequestStatus(err), Duration: time.Since(start).Milliseconds(), Error: reason})
		return
	}
	if err := validateRequestIdentifiers(r, body); err != nil {
		writeIngressError(w, bufferRequestStatus(err), "request-identifier", err.Error())
		return
	}
	strict, err := parseStrictAdmission(r.Header)
	if err != nil || strict != nil {
		message := "strict admission requires inference"
		if err != nil {
			message = err.Error()
		}
		writeIngressError(w, http.StatusBadRequest, "strict-admission", message)
		return
	}
	if alias, reason, rejected := requestAliasRejection(r, body, model, false); rejected {
		writeAliasRejection(w, alias, reason)
		_ = f.notify("proxy/request", RequestEvent{ID: reqID, Method: r.Method, Path: r.URL.Path,
			Status: http.StatusBadRequest, Duration: time.Since(start).Milliseconds(), Error: reason})
		return
	}
	candidates := f.resolveCandidates("")
	if cors.IsPreflight(r) {
		targets := make([]cors.Target, 0, len(candidates))
		for _, cand := range candidates {
			targets = append(targets, cors.Target{URL: cand.url, Transport: p.candidateTransport(cand)})
		}
		sc := &statusCapture{ResponseWriter: w, status: http.StatusOK}
		cors.ServePreflight(sc, r, targets)
		_ = f.notify("proxy/request", RequestEvent{ID: reqID, Method: r.Method, Path: r.URL.Path,
			Target: "cluster", Status: sc.status, Duration: time.Since(start).Milliseconds()})
		return
	}
	if route, ok := f.profile.routeFor(r.Method, r.URL.Path); ok && route.Role.isModelList() {
		if len(candidates) != 0 {
			_ = f.notify("proxy/request-started", RequestStartedEvent{ID: reqID, Method: r.Method, Path: r.URL.Path, Target: "cluster"})
		}
		status, err := f.serveModelList(w, r, route, candidates)
		errText := ""
		if err != nil {
			errText = err.Error()
		}
		_ = f.notify("proxy/request", RequestEvent{ID: reqID, Method: r.Method, Path: r.URL.Path,
			Target: "cluster", Status: status, Duration: time.Since(start).Milliseconds(), Error: errText})
		return
	}
	if len(candidates) == 0 {
		http.Error(w, `{"error":"no active node selected or available"}`, http.StatusBadGateway)
		_ = f.notify("proxy/request", RequestEvent{ID: reqID, Method: r.Method, Path: r.URL.Path,
			Status: http.StatusBadGateway, Duration: time.Since(start).Milliseconds(), Error: "no active node"})
		return
	}
	var final *statusCapture
	var nodeID, target, proxyErr string
	var ttfb int64
	defer func() {
		aborted := recover()
		status := 0
		if final != nil {
			status = final.status
		}
		if aborted != nil && proxyErr == "" {
			proxyErr = "upstream response copy aborted"
		}
		_ = f.notify("proxy/request", RequestEvent{ID: reqID, NodeID: nodeID, Method: r.Method, Path: r.URL.Path,
			Target: target, Status: status, Duration: time.Since(start).Milliseconds(), TTFB: ttfb, Error: proxyErr})
		if aborted != nil {
			panic(aborted)
		}
	}()
	for i, cand := range candidates {
		if r.Context().Err() != nil {
			return
		}
		out := r.Clone(r.Context())
		if body != nil {
			setReplayBody(out, body)
		}
		last, retry := i == len(candidates)-1, false
		sc := &statusCapture{ResponseWriter: w, status: http.StatusOK, idle: idleClientWriteTimeout}
		proxy := &httputil.ReverseProxy{
			Director: func(req *http.Request) {
				if cand.self {
					stripInternalRoutingHeaders(req.Header)
					req.Header.Del(peerStreamFramingHeader)
				}
				req.URL.Scheme, req.URL.Host, req.Host = cand.url.Scheme, cand.url.Host, cand.url.Host
			},
			Transport: passthroughTransport{RoundTripper: proxyResponseTransport{p.candidateTransport(cand)}, candidate: cand, facade: f},
			ModifyResponse: func(resp *http.Response) error {
				resp.Header.Del(peerStreamFramingHeader)
				resp.Trailer.Del(peerStreamFramingHeader)
				if !last && retryableProxyStatus(resp.StatusCode) {
					retry = true
					return retrySignal{}
				}
				nodeID, target, proxyErr = cand.id, cand.url.Host, ""
				ttfb = time.Since(start).Milliseconds()
				final = sc
				sc.upstreamAlive = func() { p.reportActivity(cand.id) }
				_ = f.notify("proxy/request-started", RequestStartedEvent{ID: reqID, NodeID: cand.id, Method: r.Method, Path: r.URL.Path, Target: target})
				return nil
			},
			ErrorHandler: func(ew http.ResponseWriter, _ *http.Request, err error) {
				if _, ok := err.(retrySignal); ok {
					return
				}
				owner := cand.fac
				if owner == nil {
					owner = f
				}
				owner.targets.Forget(cand.id)
				if !last {
					retry = true
					return
				}
				nodeID, target, proxyErr, final = cand.id, cand.url.Host, err.Error(), sc
				writeProxyError(ew, http.StatusBadGateway, []string{"upstream error: ", proxyErr}, nil, false)
			},
		}
		proxy.ServeHTTP(sc, out)
		if !retry {
			return
		}
	}
}

// Header arrival ends control transport discovery monitoring. Unlike inference,
// this is a transport wait only: no response-body commitment or retry owner.
type passthroughTransport struct {
	http.RoundTripper
	candidate candidate
	facade    *facade
}

func (t passthroughTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if t.candidate.dormant {
		return t.RoundTripper.RoundTrip(r)
	}
	ctx, cancel := context.WithCancel(r.Context())
	type result struct {
		response *http.Response
		err      error
	}
	ready := make(chan result, 1)
	go func() { resp, err := t.RoundTripper.RoundTrip(r.WithContext(ctx)); ready <- result{resp, err} }()
	owner := t.candidate.fac
	if owner == nil {
		owner = t.facade
	}
	ticker := time.NewTicker(targetWatchInterval)
	defer ticker.Stop()
	for {
		select {
		case result := <-ready:
			if result.err != nil {
				cancel()
				return result.response, result.err
			}
			if upgrade, ok := result.response.Body.(io.ReadWriteCloser); ok && result.response.StatusCode == http.StatusSwitchingProtocols {
				result.response.Body = &passthroughUpgradeBody{ReadWriteCloser: upgrade, cancel: cancel}
				return result.response, nil
			}
			result.response.Body = &passthroughResponseBody{ReadCloser: result.response.Body, cancel: cancel}
			return result.response, nil
		case <-ctx.Done():
			result := <-ready
			if result.response != nil {
				_ = result.response.Body.Close()
			}
			cancel()
			return nil, ctx.Err()
		case <-ticker.C:
			if !owner.discovery.Has(t.candidate.id) {
				cancel()
				result := <-ready
				if result.response != nil {
					_ = result.response.Body.Close()
				}
				return nil, errTargetGone
			}
		}
	}
}

type passthroughResponseBody struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (b *passthroughResponseBody) Close() error { defer b.cancel(); return b.ReadCloser.Close() }

type passthroughUpgradeBody struct {
	io.ReadWriteCloser
	cancel context.CancelFunc
}

func (b *passthroughUpgradeBody) Close() error {
	defer b.cancel()
	return b.ReadWriteCloser.Close()
}

func writeInferencePreparationError(w http.ResponseWriter, err error) {
	if stderrors.Is(err, errAmbiguousModel) {
		writeIngressError(w, http.StatusBadRequest, "ambiguous-model", ambiguousModelMessage)
		return
	}
	writeIngressError(w, bufferRequestStatus(err), "inference-buffer", "inference request could not be prepared")
}

// writeProxyError retains its exact response capacity through the final Write.
// Parts avoid a second concatenated error string and no encoder pool keeps the
// representation alive after the request releases its lease.
func writeProxyError(w http.ResponseWriter, status int, parts, needs []string, includeNeeds bool) {
	size := len(`{"error":""}`)
	for _, part := range parts {
		size += quotedJSONStringSize(part) - 2
	}
	if includeNeeds {
		size += len(`,"needs":[]`)
		if needs == nil {
			size += 2 // null instead of []
		} else {
			for i, need := range needs {
				size += quotedJSONStringSize(need)
				if i > 0 {
					size++
				}
			}
		}
	}
	if ctx := errorBufferContext(w); ctx != nil {
		if owner := inferenceBuffers(ctx); owner != nil {
			lease, err := owner.scope.tryAcquire(ctx, int64(size))
			if err != nil {
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("X-Content-Type-Options", "nosniff")
				w.WriteHeader(bufferRequestStatus(err))
				_, _ = w.Write(inferenceBufferRefusalJSON)
				return
			}
			defer lease.close()
		}
	}
	body := make([]byte, 0, size)
	body = append(body, `{"error":"`...)
	for _, part := range parts {
		body = appendJSONStringContent(body, part)
	}
	body = append(body, '"')
	if includeNeeds {
		body = append(body, `,"needs":`...)
		if needs == nil {
			body = append(body, "null"...)
		} else {
			body = append(body, '[')
			for i, need := range needs {
				if i > 0 {
					body = append(body, ',')
				}
				body = appendJSONString(body, need)
			}
			body = append(body, ']')
		}
	}
	body = append(body, '}')
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
