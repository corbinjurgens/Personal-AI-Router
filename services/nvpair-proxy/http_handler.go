// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"net/http/httputil"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/cors"
	"nvpair-shared/nodepolicy"
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
	io.Reader
	io.Closer
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
	// Buffered so the reader goroutine can always finish. On the timeout path
	// ReverseProxy closes resp.Body, which unblocks the pending Read.
	ch := make(chan peeked, 1)
	go func() {
		var one [1]byte
		n, err := body.Read(one[:])
		ch <- peeked{b: one[:n], err: err}
	}()

	timer := time.NewTimer(budget)
	defer timer.Stop()
	select {
	case p := <-ch:
		if len(p.b) == 0 && p.err != nil && !stderrors.Is(p.err, io.EOF) {
			return nil, p.err
		}
		return bodyWithPeek{Reader: io.MultiReader(bytes.NewReader(p.b), body), Closer: body}, nil
	case <-timer.C:
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
	if !json.Valid(body) {
		return s, nil
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	opening, err := dec.Token()
	if err != nil || opening != json.Delim('{') {
		return s, nil
	}
	seen := false
	for dec.More() {
		key, err := dec.Token()
		if err != nil {
			return s, err
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return s, err
		}
		if key == "stream" {
			if seen || (string(value) != "true" && string(value) != "false" && string(value) != "null") {
				return s, fmt.Errorf("stream must be a single boolean or null member")
			}
			seen = true
			if string(value) != "null" {
				s.declared = string(value) == "true"
				s.disabled = !s.declared
			}
		}
		if key == "n" {
			var n int
			if json.Unmarshal(value, &n) == nil && n > 0 {
				s.choices = n
			}
		}
	}
	return s, nil
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

	// Allocate the request ID up front so both code paths (rejection
	// and forward) can stamp the same value into their notification.
	// The rejection path never emits a Started event, so its ID won't
	// appear in any orchestrator in-flight map — that's fine; the
	// completion event still bumps the failed counter regardless of
	// whether a matching Started was seen.
	reqID := strconv.FormatUint(f.nextRequestID.Add(1), 10)

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
	// curTicket is the admission held for a self dispatch. Released after each
	// attempt and, by this defer, on every other way out, including the
	// ErrAbortHandler unwind of a mid-stream copy error.
	var curTicket *admissionTicket
	defer func() { curTicket.release() }()

	// Parse the request's model before choosing a node. Model eligibility only
	// applies to inference routes; control endpoints retain their existing
	// routing behavior even when their JSON happens to contain a model field.
	bodyBytes, model, err := bufferBodyAndModel(w, r, p.requestBodyLimit())
	if err != nil {
		slog.Warn("proxy request rejected",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"remote", r.RemoteAddr, "reason", "request body too large", "limit", p.requestBodyLimit())
		http.Error(w, `{"error":"request body exceeds the proxy's size limit"}`, http.StatusRequestEntityTooLarge)
		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Status:   http.StatusRequestEntityTooLarge,
			Duration: time.Since(start).Milliseconds(),
			Error:    "request body too large",
		})
		return
	}
	isInf := isInferenceRequest(f.profile, r.Method, r.URL.Path)
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
	routingModel := ""
	if isInf {
		routingModel = model
	}
	// A tier request resolves to concrete (node, engine, model) candidates
	// from the policy instead of to owners of the literal model name.
	tier := ""
	var needs requestNeeds
	crossEngine := false
	if isInf && nodepolicy.IsTier(model) {
		tier = model
		needs = parseRequestNeeds(bodyBytes, r.URL.Path)
		crossEngine = openAICompatibleInference[r.URL.Path]
	}
	ctl := newWorkloadCtl(cancelRequest)
	resolve := func() []candidate {
		var cands []candidate
		if tier != "" {
			cands = f.resolveTierCandidates(tier, needs, crossEngine)
		} else {
			cands = f.resolveCandidates(routingModel)
		}
		if !ctl.hasExclusions() {
			return cands
		}
		kept := make([]candidate, 0, len(cands))
		for _, c := range cands {
			if !ctl.isExcluded(c.id) {
				kept = append(kept, c)
			}
		}
		return kept
	}
	candidates := resolve()
	if cors.IsPreflight(r) {
		targets := make([]cors.Target, 0, len(candidates))
		for _, cand := range candidates {
			targets = append(targets, cors.Target{URL: cand.url, Transport: p.candidateTransport(cand)})
		}
		sc := &statusCapture{ResponseWriter: w, status: http.StatusOK}
		cors.ServePreflight(sc, r, targets)
		f.notify("proxy/request", RequestEvent{ID: reqID, Method: r.Method, Path: r.URL.Path, Target: "cluster", Status: sc.status, Duration: time.Since(start).Milliseconds()})
		return
	}
	// held is this request's claim on a node's capacity while it is in flight.
	// resMu guards it because failover re-points it from the ReverseProxy's
	// callbacks while the disconnect watcher may be releasing it.
	var (
		resMu sync.Mutex
		held  reservation
	)
	if isInf && model != "" {
		candidates, held = f.reserveRound(candidates, tier != "")
	}
	// Released exactly once, whichever way the request ends: normal unwind,
	// client disconnect, or an early return below. Without this a node stays
	// counted as loaded until the next scheduler snapshot, so a steady trickle
	// of short requests makes it look busier than it is and drives dispatch
	// away from it.
	//
	// Both accessors release resMu by defer. The release itself runs from a
	// defer, so a resMu stranded by a panic in the move would deadlock this
	// request against its own cleanup.
	moveHeld := func(nodeID string) {
		resMu.Lock()
		defer resMu.Unlock()
		held = p.moveReservation(held, nodeID)
	}
	defer func() {
		resMu.Lock()
		defer resMu.Unlock()
		p.releaseReservation(held)
		held = reservation{}
	}()
	if matchedRoute, ok := f.profile.routeFor(r.Method, r.URL.Path); ok && matchedRoute.Role.isModelList() {
		if len(candidates) > 0 {
			_ = f.notify("proxy/request-started", RequestStartedEvent{
				ID: reqID, Method: r.Method, Path: r.URL.Path, Target: "cluster",
			})
		}
		status, err := f.serveModelList(w, r, matchedRoute, candidates)
		errText := ""
		if err != nil {
			errText = err.Error()
		}
		_ = f.notify("proxy/request", RequestEvent{
			ID: reqID, Method: r.Method, Path: r.URL.Path, Target: "cluster",
			Status: status, Duration: time.Since(start).Milliseconds(), Error: errText,
		})
		return
	}
	if len(candidates) == 0 && tier != "" {
		// Nothing anywhere can answer this tier for this request, which waiting
		// will not change.
		body, mErr := json.Marshal(struct {
			Error string   `json:"error"`
			Needs []string `json:"needs"`
		}{Error: "no eligible model for tier " + tier, Needs: needs.names()})
		if mErr != nil {
			body = []byte(`{"error":"no eligible model for tier"}`)
		}
		slog.Warn("proxy request rejected",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"remote", r.RemoteAddr, "reason", "no eligible model for tier", "tier", tier)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write(body)
		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Status:   http.StatusNotFound,
			Duration: time.Since(start).Milliseconds(),
			Error:    "no eligible model for tier",
		})
		return
	}
	if len(candidates) == 0 {
		rejectionBody := `{"error":"no active node selected or available"}`
		rejectionError := "no active node"
		if isInf && model != "" {
			rejectionBody = `{"error":"no available node advertises the requested model"}`
			rejectionError = "no node advertises requested model"
		}
		slog.Warn("proxy request rejected",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"remote", r.RemoteAddr, "reason", rejectionError)
		http.Error(w, rejectionBody, http.StatusBadGateway)
		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Status:   http.StatusBadGateway,
			Duration: time.Since(start).Milliseconds(),
			Error:    rejectionError,
		})
		return
	}
	// shouldRetry reports whether an upstream status warrants failing over to
	// the next candidate: busy/unavailable/gateway statuses, plus a 404 on an
	// inference call (an advertised owner's inventory may have become stale).
	// Genuine client errors (400/401/422…) are not retried — they'd fail
	// identically on every node.
	shouldRetry := func(code int) bool {
		switch code {
		case http.StatusRequestTimeout,
			http.StatusTooManyRequests,
			http.StatusBadGateway,
			http.StatusServiceUnavailable,
			http.StatusGatewayTimeout:
			return true
		case http.StatusNotFound:
			return isInf
		}
		return code >= 500
	}

	var (
		servedNodeID string
		servedTarget string
		ttfbMs       int64
		proxyErr     string
		finalStatus  int
		started      bool
		wl           *Workload
	)

	// wlSeq numbers this workload's events. Every mutation of wl below happens
	// either before the watcher goroutine exists or under wlMu, so a plain
	// counter is enough; see Workload.Seq for why the sequence is needed.
	var wlSeq int64
	nextWlSeq := func() int64 { wlSeq++; return wlSeq }

	// Emit workload:submitted the moment the request is admitted, before any
	// dispatch. A burst of concurrent inference requests must surface as job
	// cards immediately — the upstream engine serializes work on a single GPU
	// slot, so a card that waited for the upstream response would leave every
	// queued job invisible until the node dequeued it, one at a time (a prior
	// regression). "queued" is what makes that visible without claiming the
	// engine is generating, which only becomes true at the commit point below.
	//
	// scheduledOn is left empty here: nothing has been dispatched yet, and the
	// scheduler counts pending work by scheduledOn, so naming a node we have
	// not tried would inflate its load. Each dispatch and each gap between
	// attempts re-points it.
	if isInf && model != "" {
		createdMs := start.UnixMilli()
		// A tier request's model is the concrete model it is headed for: the
		// leading candidate's until a dispatch says otherwise.
		wlModel := model
		if tier != "" && candidates[0].model != "" {
			wlModel = candidates[0].model
		}
		wl = &Workload{
			ID:             reqID,
			Model:          wlModel,
			RequestedModel: tier,
			Engine:         f.profile.Name,
			RunID:          p.runID,
			State:          "queued",
			CreatedAt:      createdMs,
			Seq:            nextWlSeq(),
		}
		p.emitWorkload(workloadSubmittedMethod, *wl)
		key := workloadKey{engine: f.profile.Name, id: reqID}
		p.registerWorkload(key, ctl)
		defer p.unregisterWorkload(key)
		defer ctl.finish()
	}

	// The terminal workload transition (completed/errored) can be reached from
	// two places: the normal path after the stream copy unwinds below, and the
	// disconnect watcher that fires while the copy is still blocked. terminalOnce
	// guarantees exactly one is emitted; wlMu guards the shared wl fields the
	// watcher (a separate goroutine) and ModifyResponse's failover re-point both
	// touch; terminated suppresses a late started re-point once we've finalized.
	var (
		terminalOnce sync.Once
		wlMu         sync.Mutex
		terminated   bool
	)
	emitTerminal := func(state, errMsg string) {
		if wl == nil {
			return
		}
		terminalOnce.Do(func() {
			now := time.Now().UnixMilli()
			wlMu.Lock()
			terminated = true
			wl.CompletedAt = &now
			wl.State = state
			if errMsg != "" {
				wl.Error = &errMsg
			}
			wl.Seq = nextWlSeq()
			snapshot := *wl
			wlMu.Unlock()
			// Only "completed" gets its own method; every other terminal state,
			// including "cancelled", rides workload:errored. The workload
			// manager does not validate method against state and consumers read
			// the state out of the payload, so a new terminal state needs no new
			// method on the wire.
			method := workloadCompletedMethod
			if state != "completed" {
				method = workloadErroredMethod
			}
			p.emitWorkload(method, snapshot)
		})
	}

	// Watch for the client going away while the request is in flight. The
	// terminal event is otherwise emitted only after the stream copy returns;
	// a client that disconnects mid-stream can leave the copy blocked, so we
	// emit the terminal here the moment r.Context() is cancelled instead of
	// waiting for the unwind. Cancelling r.Context() (client close, or our own
	// shutdown) also propagates to the ReverseProxy's upstream request, so the
	// engine stops generating. terminalOnce keeps this from double-emitting
	// with the normal path. The half-open case (no FIN, r.Context() never
	// fires) is caught instead by statusCapture's write deadline below.
	if wl != nil {
		reqCtx := r.Context()
		finished := make(chan struct{})
		defer close(finished)
		go func() {
			select {
			case <-reqCtx.Done():
				emitTerminal("cancelled", cancelReasonOr("client disconnected before completion"))
			case <-finished:
			}
		}()
	}

	// committedSC is the statusCapture of the candidate we committed to
	// streaming; its wroteErr tells us after the fact whether the client write
	// failed (dead/half-open client) so we can mark the workload failed. It is
	// assigned at the commit point as well as after the loop, because the
	// post-loop assignment does not always run — see finalize below.
	var committedSC *statusCapture
	var committedStream *streamFrameReader

	// finalize reports this request's outcome, and it is deferred because a
	// mid-copy error is never returned to us: ReverseProxy converts one into
	// panic(http.ErrAbortHandler) whenever it detects a real server, and that
	// panic unwinds straight past the end of this function. Reported inline,
	// both the terminal workload event and the proxy/request completion were
	// lost whenever an upstream died mid-stream while the client was healthy,
	// leaving the job "running" for the life of the broker. That is worse than
	// a stuck card: the scheduler counts queued and running by scheduledOn, so
	// the ghost permanently inflated that node's pending count and biased
	// routing away from a node whose only fault was a crashed engine.
	//
	// Client-side failures never had this problem. net/http cancels the
	// request context synchronously before a write error propagates
	// (checkConnErrorWriter), so the watcher above has always emitted by the
	// time the panic arrives. The upstream-died case is the one that had no
	// reporter at all.
	//
	// Registered after the reservation release above, so on unwind this runs
	// first and the node is still counted as loaded while we report.
	defer func() {
		aborted := recover()
		// A managed copy ends with normal HTTP EOF even on protocol failure;
		// its separate result, rather than the committed status, is decisive.
		if committedStream != nil {
			result := committedStream.Result()
			if result.Outcome == streamOutcomeInterrupted || result.Outcome == streamOutcomeUpstreamError {
				proxyErr = fmt.Sprintf("stream interrupted after output began on %s: %s", servedNodeID, result.Reason)
			}
		}
		if aborted != nil && proxyErr == "" {
			proxyErr = fmt.Sprintf("stream interrupted after output began on %s: upstream response copy aborted", servedNodeID)
		}
		reportErr := proxyErr
		if committedSC != nil && finalStatus == 0 {
			// The panic skipped the post-loop assignment, but the committed
			// capture still carries the status the upstream sent.
			finalStatus = committedSC.status
		}

		slog.Debug("proxy request complete",
			"id", reqID,
			"node_id", servedNodeID,
			"method", r.Method,
			"path", r.URL.Path,
			"target", servedTarget,
			"status", finalStatus,
			"duration_ms", time.Since(start).Milliseconds(),
			"ttfb_ms", ttfbMs,
			"err", reportErr,
		)

		_ = f.notify("proxy/request", RequestEvent{
			ID:       reqID,
			NodeID:   servedNodeID,
			Method:   r.Method,
			Path:     r.URL.Path,
			Target:   servedTarget,
			Status:   finalStatus,
			Duration: time.Since(start).Milliseconds(),
			TTFB:     ttfbMs,
			Error:    reportErr,
		})

		// Terminal workload transition pairs with the workload:started emitted
		// at forward time above. Cancellation, an upstream/transport error, a
		// failed client write (dead/half-open client), or any non-2xx status is
		// a failure; a clean 2xx is a completion. The Workload carries the same
		// id so the broker (and peers) can collapse the start/finish pair.
		// Routed through emitTerminal so the disconnect watcher and this path
		// emit exactly once.
		if wl != nil {
			switch {
			case r.Context().Err() != nil:
				// The request was cancelled before it finished — either the
				// client disconnected or, on shutdown, we cancelled it to stop
				// the in-flight inference. A mid-stream cancel never reaches
				// ErrorHandler (the 200 headers are already sent), so without
				// this branch it would be misreported as completed. (The
				// watcher above usually beats us to it; emitTerminal makes that
				// a no-op.) Cancelled rather than failed: nothing went wrong
				// here, the requester stopped waiting.
				emitTerminal("cancelled", cancelReasonOr("request cancelled before completion"))
			case committedSC != nil && committedSC.wroteErr != nil:
				// The response committed but a write to (or flush toward) the
				// client failed — typically the idle deadline tripping on a
				// dead/half-open client. The same event as the branch above,
				// differing only in how we noticed: a client that vanished
				// without a FIN never cancels the context, so the write
				// deadline is what surfaces it. Classified identically.
				emitTerminal("cancelled", "client connection lost: "+committedSC.wroteErr.Error())
			case proxyErr != "" || finalStatus < http.StatusOK || finalStatus >= http.StatusMultipleChoices:
				msg := proxyErr
				if msg == "" {
					msg = fmt.Sprintf("upstream returned HTTP %d", finalStatus)
				}
				emitTerminal("failed", msg)
			default:
				emitTerminal("completed", "")
			}
		}

		if aborted != nil {
			// Preserve ReverseProxy's contract with http.Server, which recovers
			// ErrAbortHandler silently and closes the connection. Swallowing it
			// here would leave the client waiting on a response that will never
			// be finished or closed.
			panic(aborted)
		}
	}()

	// repointWorkload records where the job is currently placed: a node id while
	// an attempt is in flight, empty between attempts. The scheduler counts
	// pending work by scheduledOn, so clearing it is what stops a node we have
	// given up on from still looking busy. The state stays "queued" throughout,
	// because "running" means the engine is generating and the broker's store
	// would reject a return to "queued" as a backwards transition.
	repointWorkload := func(nodeID, concrete string) {
		if wl == nil {
			return
		}
		wlMu.Lock()
		if terminated || (wl.ScheduledOn == nodeID && (concrete == "" || wl.Model == concrete)) {
			wlMu.Unlock()
			return
		}
		wl.ScheduledOn = nodeID
		if concrete != "" {
			wl.Model = concrete
		}
		wl.Seq = nextWlSeq()
		snapshot := *wl
		wlMu.Unlock()
		p.emitWorkload(workloadSubmittedMethod, snapshot)
	}

	// releaseHeld drops the capacity claim. Called before every backoff: the
	// reservation map is process-wide and shared with the other engine's
	// facade, so holding a claim on a node we have stopped using would push
	// that engine's traffic away from a node that is in fact free. It is the
	// same reasoning as clearing scheduledOn, applied to the proxy's own
	// estimate rather than the scheduler's.
	releaseHeld := func() {
		resMu.Lock()
		defer resMu.Unlock()
		p.releaseReservation(held)
		held = reservation{}
	}
	// takeHeld re-reserves for a fresh round, replacing any claim still held.
	// reserveCandidate both orders the round and takes the claim, so a retry
	// that re-resolves has to go back through it.
	takeHeld := func(cands []candidate) []candidate {
		resMu.Lock()
		defer resMu.Unlock()
		p.releaseReservation(held)
		var out []candidate
		out, held = f.reserveRound(cands, tier != "")
		return out
	}

	// Retry loop. Each iteration dispatches to one candidate; when a round's
	// candidates are exhausted it backs off and re-resolves, so a node that
	// recovered or a model that finished pulling becomes eligible mid-retry.
	//
	// Bounded by maxDispatchAttempts for dispatch failures and by jobDeadline
	// for the whole pre-commit life. Both bounds are checked before every
	// dispatch — the deadline never truncates an attempt already in flight,
	// because killing one seconds from its first token and then failing the job
	// for being out of time would spend the entire wait and discard the result.
	//
	// Retrying is only possible before the first byte reaches the client; past
	// the commit point we are bound to that node, and a stream truncated from
	// there fails explicitly for supported streams. proxy/request-started fires
	// at the commit so it names the node that actually served. The self-forward guard lives in
	// resolveCandidates.
	// The budget is for inference only. Everything not in the engine's route
	// table is forwarded verbatim, and some of those are state-changing: a
	// POST /api/pull starts a model download. shouldRetry treats any 5xx as
	// retryable regardless of route, so without this split a failed pull would
	// be replayed up to five times — including against the node that just took
	// the work, since re-resolution can pick it again.
	//
	// Non-inference therefore gets one pass over the resolved candidates: no
	// re-resolution, no backoff, no deadline.
	maxDispatches := maxDispatchAttempts
	retryRounds := true
	if !isInf {
		maxDispatches = len(candidates)
		retryRounds = false
	}

	deadline := start.Add(jobDeadline)
	dispatches := 0
	committed := false
	respondedWithError := false
	regenExhausted := false
	// selfRejection is why this node refused its own candidate when that
	// refusal excluded it, and is the answer if nothing else remains.
	var selfRejection nodepolicy.RejectReason
	round := candidates
	next := 0

	for !committed {
		if dispatches >= maxDispatches {
			break
		}
		if retryRounds && !time.Now().Before(deadline) {
			break
		}
		// Abandon-now: the requester has gone, so there is nobody left to
		// receive an answer and the remaining attempts are better spent on work
		// someone is waiting for. The watcher has already emitted the terminal
		// as cancelled; the guard on the exhaustion response below keeps this
		// from also inventing a reason for a client that stopped listening.
		if r.Context().Err() != nil {
			break
		}
		if next >= len(round) {
			if !retryRounds {
				// Single pass: the candidate list is the whole budget.
				break
			}
			// Round exhausted, or no owner was available. Either way the job is
			// on no node right now, so drop both the placement and the capacity
			// claim before waiting.
			repointWorkload("", "")
			releaseHeld()
			// This node refused its own candidate for a reason no backoff will
			// change, and nothing else can take the request: answer now.
			if selfRejection != "" && len(resolve()) == 0 {
				respondedWithError = true
				finalStatus = http.StatusServiceUnavailable
				proxyErr = "admission rejected: " + string(selfRejection)
				writeAdmissionRejection(w, selfRejection)
				break
			}
			if !waitBeforeRetry(r.Context(), backoffFor(dispatches, time.Until(deadline))) {
				break
			}
			round = resolve()
			if isInf && model != "" {
				round = takeHeld(round)
			}
			next = 0
			if len(round) == 0 && ctl.hasExclusions() {
				// A cancel, or this node's own pause, excluded the only node
				// left. Waiting for an owner would only run out the deadline on
				// a node this request may not use.
				if selfRejection != "" {
					respondedWithError = true
					finalStatus = http.StatusServiceUnavailable
					proxyErr = "admission rejected: " + string(selfRejection)
					writeAdmissionRejection(w, selfRejection)
					break
				}
				regenExhausted = true
				break
			}
			if len(round) == 0 {
				// Every owner went away mid-flight. Not a rejection: keep
				// waiting for one to come back. This consumes no attempt, which
				// is why jobDeadline is the only bound on it.
				continue
			}
		}
		cand := round[next]
		next++
		// A node a cancel excluded stays out of this request, including later
		// entries of a round resolved before the exclusion.
		if ctl.isExcluded(cand.id) {
			continue
		}
		dispatches++
		// lastPermitted means no further dispatch can happen, so this attempt's
		// failure is the client's answer. It is not "last candidate in the
		// round" any more: another round may follow.
		lastPermitted := dispatches >= maxDispatches ||
			(retryRounds && !time.Now().Before(deadline))
		last := lastPermitted
		// What this attempt asks for: a tier candidate names its concrete model
		// and may be another engine of this process.
		candModel := model
		if cand.model != "" {
			candModel = cand.model
		}
		candEngine := cand.engineOr(f.profile.Name)
		candFacade := cand.fac
		if candFacade == nil {
			candFacade = f
		}
		repointWorkload(cand.id, candModel)
		// The claim follows the node we are about to try, not the one
		// reserveCandidate happened to pick for the round — an attempt can hold
		// a node for the whole first-content budget, and for that time it is
		// the node under load. A same-node move is a no-op.
		moveHeld(cand.id)

		// Each attempt runs on its own context, a child of the request's, so we
		// can abandon this dispatch without touching the request itself. The
		// distinction is load-bearing: the disconnect watcher and the terminal
		// classification both read r.Context(), the parent, so cancelling a
		// child cannot be mistaken for the client going away.
		attemptCtx, cancelAttempt := context.WithCancel(r.Context())
		// claim decides, once, whether this attempt commits or is abandoned;
		// see attemptClaim for why a pair of channels could not.
		claim := &attemptClaim{}
		att := &attemptCtl{claim: claim, cancel: cancelAttempt, nodeID: cand.id}
		ctl.setAttempt(att)

		// The body this attempt sends: the request's own, with a tier's concrete
		// model in place of the tier name and every other byte untouched.
		sendBody := bodyBytes
		if cand.model != "" {
			rewritten, ok := replaceModel(bodyBytes, cand.model)
			if !ok {
				cancelAttempt()
				ctl.setAttempt(nil)
				respondedWithError = true
				finalStatus = http.StatusBadRequest
				proxyErr = "model rewrite failed"
				writeIngressError(w, http.StatusBadRequest, "ambiguous-model", ambiguousModelMessage)
				break
			}
			sendBody = rewritten
		}
		// How long the destination may queue this attempt: not at all while
		// this round still has somewhere else to go, so a busy node hands the
		// request straight back; the policy's queue timeout for the last one.
		admissionWait := 0
		if next >= len(round) {
			admissionWait = p.admission.currentPolicy().Admission.QueueTimeoutSeconds
		}

		// This node's own engine is admitted here, before dispatch, the way a
		// peer's ingress admits what we send it. A rejection is a failed
		// candidate like a peer's 503, and writes nothing to the client unless
		// it is the answer.
		target := cand.url
		admissionStarted := time.Now()
		var performanceWait time.Duration
		if cand.self && isInf {
			decision := p.admission.admit(attemptCtx, admissionRequest{
				engine:  candEngine,
				model:   candModel,
				waitCap: time.Duration(admissionWait) * time.Second,
				cancel:  func() { ctl.abortLocalIf(att) },
			})
			rejectSelf := func(errText string, answer func()) bool {
				cancelAttempt()
				ctl.setAttempt(nil)
				proxyErr = errText
				if !last {
					slog.Debug("proxy self candidate not admitted, failing over",
						"id", reqID, "node_id", cand.id, "engine", candEngine, "err", errText)
					return false
				}
				respondedWithError = true
				servedNodeID = cand.id
				servedTarget = target.Host
				finalStatus = http.StatusServiceUnavailable
				answer()
				return true
			}
			if decision.err != nil {
				cancelAttempt()
				ctl.setAttempt(nil)
				if att.redispatched() {
					dispatches--
				}
				// Otherwise the request itself ended while it waited; the loop's
				// own check ends it.
				continue
			}
			if decision.reject != "" {
				if rejectSelf("admission rejected: "+string(decision.reject), func() {
					writeAdmissionRejection(w, decision.reject)
				}) {
					break
				}
				// A node that is pausing or paused will refuse every later
				// round too, so it leaves this request rather than spending
				// its retry budget on the same refusal.
				if p.admission.currentAvailability() != nodepolicy.Available {
					ctl.exclude(cand.id)
					selfRejection = decision.reject
				}
				continue
			}
			curTicket = decision.ticket
			performanceWait = time.Since(admissionStarted)
			if decision.hasProfile {
				sendBody = mergeDefaults(sendBody, decision.profile.RequestOptions)
				if !bodyModelMatches(sendBody, candModel) {
					curTicket.release()
					cancelAttempt()
					ctl.setAttempt(nil)
					respondedWithError = true
					finalStatus = http.StatusBadRequest
					proxyErr = "model defaults changed admitted model"
					writeIngressError(w, http.StatusBadRequest, "ambiguous-model", ambiguousModelMessage)
					break
				}
			}
			if cand.dormant {
				// Admission waited for the wake; the engine's address is the one
				// it came up on.
				u, ok := candFacade.localBackendTarget()
				if !ok {
					curTicket.release()
					if rejectSelf("local engine is not running", func() {
						writeIngressError(w, http.StatusServiceUnavailable, "no-local-backend",
							"no local inference backend is available on this node")
					}) {
						break
					}
					continue
				}
				target = u
			}
		}

		// Framing follows this attempt's effective body, including local profile
		// defaults. Ingress validation above still rejects ambiguous client flags
		// before admission or any body rewrite.
		var streamRequest originStreamRequest
		if isInf {
			var streamErr error
			streamRequest, streamErr = parseOriginStreamRequest(r.URL.Path, sendBody)
			if streamErr != nil {
				curTicket.release()
				cancelAttempt()
				ctl.setAttempt(nil)
				respondedWithError = true
				finalStatus = http.StatusBadRequest
				proxyErr = "invalid effective stream flag"
				writeIngressError(w, http.StatusBadRequest, "invalid-stream", "invalid effective stream flag")
				break
			}
		}

		// Watch the target while the attempt is pre-commit. A node that has
		// dropped out of discovery is not coming back to answer a request
		// already sent to it, and without this the attempt sat out the whole
		// first-content budget before failing over — while the broker had
		// already stamped the record failed from its own node-loss sweep, so
		// the UI said failed while the client still waited.
		//
		// Polling rather than subscribing keeps this out of Discovery's
		// contract, and the interval only bounds detection latency. Watching
		// stops at the commit point, because cancelling after that would kill a
		// stream that is working.
		//
		// A dormant self candidate is not in discovery until its engine has
		// been advertised again, so it is not watched.
		//
		// Read the interval once, on this goroutine: the watcher must not
		// depend on a value that could change under it mid-attempt.
		watchEvery := targetWatchInterval
		if !cand.dormant {
			go func() {
				t := time.NewTicker(watchEvery)
				defer t.Stop()
				for {
					select {
					case <-attemptCtx.Done():
						return
					case <-t.C:
						// The commit path may have claimed the outcome since the
						// last tick, in which case this attempt is no longer ours
						// to cancel.
						if claim.settled() {
							return
						}
						if candFacade.discovery.Has(cand.id) {
							continue
						}
						if claim.abandon() {
							cancelAttempt()
						}
						return
					}
				}
			}()
		}

		// ReverseProxy derives the outbound request from the one it is given,
		// so the attempt context has to travel on a copy. The replayed body
		// and the per-attempt admission header go on the copy for the same
		// reason, with headers cloned so one attempt's never leak into the
		// next.
		attemptReq := r.WithContext(attemptCtx)
		attemptReq.Header = r.Header.Clone()
		if isInf && streamRequest.protocol != streamProtocolUnknown {
			// Explicit identity also prevents Go's transport from negotiating gzip.
			attemptReq.Header.Set("Accept-Encoding", "identity")
		}
		if bodyBytes != nil {
			setReplayBody(attemptReq, sendBody)
		}
		if isInf {
			if cand.self {
				attemptReq.Header.Del(nodepolicy.AdmissionWaitHeader)
			} else {
				attemptReq.Header.Set(nodepolicy.AdmissionWaitHeader, strconv.Itoa(admissionWait))
			}
		}
		retry := false
		sc := &statusCapture{ResponseWriter: w, status: http.StatusOK, idle: idleClientWriteTimeout}
		var performance *performanceAttempt
		if cand.self && isInf {
			performance = candFacade.newPerformanceAttempt(r.Context(), candModel, r.URL.Path, sendBody, performanceWait)
		}

		proxy := &httputil.ReverseProxy{
			Director: func(req *http.Request) {
				if cand.self {
					stripInternalRoutingHeaders(req.Header)
					req.Header.Del(peerStreamFramingHeader)
				}
				req.URL.Scheme = target.Scheme
				req.URL.Host = target.Host
				req.Host = target.Host
			},
			// A remote cluster peer is dialed over mTLS (per-peer pinned config);
			// self/manual candidates use the plain transport. See candidateTransport.
			Transport: performanceTransport{RoundTripper: proxyResponseTransport{p.candidateTransport(cand)}, attempt: performance},
			// ModifyResponse fires when the upstream's status line + headers
			// have arrived but before the body streams. That's both the retry
			// decision point and, on commit, the time-to-first-byte boundary.
			ModifyResponse: func(resp *http.Response) error {
				metadata := resp.Header.Values(peerStreamFramingHeader)
				resp.Header.Del(peerStreamFramingHeader)
				resp.Trailer.Del(peerStreamFramingHeader)
				if !last && shouldRetry(resp.StatusCode) {
					// Abort before streaming: ReverseProxy closes resp.Body and
					// calls ErrorHandler with our sentinel, then we try next.
					retry = true
					return retrySignal{}
				}
				if isInf && cand.peerUUID != "" && streamRequest.protocol != streamProtocolUnknown && resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
					var err error
					streamRequest, err = streamRequest.withPeerMetadata(metadata)
					if err != nil {
						return err // existing attempt failure path; never commit body bytes
					}
				}
				// Hold the commit until the engine actually produces content.
				// Only for inference: a control endpoint's headers mean what
				// they say, and the model-list fan-out has its own client.
				// Waiting here costs a non-streaming response nothing, since
				// its headers and body arrive together. Framing is selected
				// separately, after the winning claim.
				var peeked io.ReadCloser
				if isInf {
					body, err := awaitFirstBody(resp.Body, firstBodyTimeout)
					if err != nil {
						// This response must not commit, whether or not another
						// attempt is permitted. awaitFirstBody's reader is still
						// blocked on resp.Body, so committing would put two
						// readers on one stream — the peek could swallow a byte
						// the client never sees — and against a silent upstream
						// the copy would then block with nothing left to
						// interrupt it, hanging the request past every bound we
						// set.
						//
						// Returning an error is what releases that reader:
						// ReverseProxy closes resp.Body and calls ErrorHandler.
						proxyErr = err.Error()
						if !last {
							retry = true
							slog.Warn("proxy upstream produced no content, failing over",
								"id", reqID, "node_id", cand.id, "target", target.Host,
								"path", r.URL.Path, "err", err)
							return retrySignal{}
						}
						slog.Warn("proxy upstream produced no content, retries exhausted",
							"id", reqID, "node_id", cand.id, "target", target.Host,
							"method", r.Method, "path", r.URL.Path,
							"duration_ms", time.Since(start).Milliseconds(), "err", err)
						return err
					}
					peeked = body
				}
				// Claim the attempt before any commit side effect. Losing the
				// claim means the watcher has already cancelled this attempt,
				// so committing would stream through a context that is about to
				// die and truncate the response we just started.
				if !claim.commit() {
					if att.redispatch.Load() {
						// A cancel moved this request on before it committed.
						retry = true
						return retrySignal{}
					}
					proxyErr = errTargetGone.Error()
					if !last {
						retry = true
						slog.Warn("proxy target left discovery as the response committed, failing over",
							"id", reqID, "node_id", cand.id, "target", target.Host,
							"path", r.URL.Path)
						return retrySignal{}
					}
					return errTargetGone
				}
				// The peeked byte goes back only once the commit is ours, so a
				// lost claim leaves the body untouched for ReverseProxy to close.
				if peeked != nil {
					resp.Body = peeked
				}
				if isInf {
					reader, err := installResponseStream(resp, streamRequest, r.Context(), streamIdle, performance)
					if err != nil {
						return err
					}
					committedStream = reader
				}
				if resp.StatusCode != http.StatusSwitchingProtocols {
					resp.Body = &originResponseBody{ReadCloser: resp.Body, response: resp, managed: committedStream != nil}
				}
				// Name what actually answered (FORK_DESIGN.md §3.8). Set on the
				// committed response only, so a failed-over attempt's headers
				// can never reach the client.
				if isInf {
					resp.Header.Set(nodepolicy.ModelHeader, candModel)
					resp.Header.Set(nodepolicy.EngineHeader, candEngine)
					resp.Header.Set(nodepolicy.NodeHeader, cand.id)
					if tier != "" {
						resp.Header.Set(nodepolicy.TierHeader, tier)
					} else {
						resp.Header.Del(nodepolicy.TierHeader)
					}
				}
				// Committing to this candidate — body stream is about to begin.
				ttfbMs = time.Since(start).Milliseconds()
				servedNodeID = cand.id
				servedTarget = target.Host
				proxyErr = "" // clear any error recorded from a failed-over candidate
				// Arm the liveness report only now. statusCapture also carries
				// the proxy's OWN error bodies — ReverseProxy's ErrorHandler
				// writes a failed dial's message through it — and those bytes
				// prove nothing about the node. Reaching here means the upstream
				// returned a status line, so everything written from this point
				// came from the node. Same goroutine as the body copy, so no
				// synchronization is needed.
				sc.upstreamAlive = func() { p.reportActivity(cand.id) }
				// Preserve the upstream response, including absent CORS permissions.
				if !started {
					started = true
					_ = f.notify("proxy/request-started", RequestStartedEvent{
						ID:     reqID,
						NodeID: cand.id,
						Method: r.Method,
						Path:   r.URL.Path,
						Target: target.Host,
					})
					// The claim already follows each dispatch, so this is a
					// no-op unless something reordered underneath us; it stays
					// as the belt-and-braces guarantee that the node that
					// actually served is the one counted as loaded.
					moveHeld(cand.id)
					// Record the commit here, not just after the loop: a copy
					// error aborts this handler by panic, and finalize needs to
					// know the response had committed (and on which capture) to
					// classify it.
					committedSC = sc
					// queued -> running: the engine is producing content, so
					// this is the first moment "running" is true. It also fixes
					// the placement on the node that actually served, which may
					// differ from the last node the queued updates named.
					// Guarded by wlMu against the disconnect watcher, and
					// skipped once terminated so a late transition cannot
					// resurrect a workload we have already finalized.
					if wl != nil {
						wlMu.Lock()
						if !terminated {
							startedMs := time.Now().UnixMilli()
							wl.State = "running"
							wl.StartedAt = &startedMs
							wl.ScheduledOn = cand.id
							wl.Model = candModel
							wl.Seq = nextWlSeq()
							snapshot := *wl
							wlMu.Unlock()
							p.emitWorkload(workloadStartedMethod, snapshot)
						} else {
							wlMu.Unlock()
						}
					}
				}
				return nil
			},
			ErrorHandler: func(ew http.ResponseWriter, _ *http.Request, err error) {
				if _, ok := err.(retrySignal); ok {
					return // retryable status — the loop advances to the next candidate
				}
				// A deliberate abort says nothing about the node or its
				// address, and is never the client's answer.
				if att.redispatch.Load() && claim.abandoned() {
					retry = true
					return
				}
				// Transport/dial error (not a status-based retry): forget this
				// node's confirmed address so the next request re-confirms and
				// can fail over to another of its published addresses
				// (multi-homed peer). The in-request failover below moves on to
				// the next node.
				candFacade.targets.Forget(cand.id)
				if !last {
					// Transport/dial error with candidates left: fail over.
					retry = true
					proxyErr = err.Error()
					slog.Warn("proxy upstream error, failing over",
						"id", reqID, "node_id", cand.id, "target", target.Host,
						"path", r.URL.Path, "err", err)
					return
				}
				// No further dispatch is permitted: terminal, surface it.
				respondedWithError = true
				servedNodeID = cand.id
				servedTarget = target.Host
				proxyErr = err.Error()
				slog.Warn("proxy upstream error, retries exhausted",
					"id", reqID, "node_id", cand.id, "target", target.Host,
					"method", r.Method, "path", r.URL.Path,
					"duration_ms", time.Since(start).Milliseconds(), "err", err)
				body, mErr := json.Marshal(map[string]string{
					"error": "upstream error: " + err.Error(),
				})
				if mErr != nil {
					body = []byte(`{"error":"upstream error"}`)
				}
				// A node that answered but never produced content timed out
				// rather than failed to be reached, and 504 says so.
				status := http.StatusBadGateway
				if stderrors.Is(err, errFirstBodyTimeout) {
					status = http.StatusGatewayTimeout
				}
				ew.Header().Set("Content-Type", "application/json")
				ew.Header().Set("X-Content-Type-Options", "nosniff")
				ew.WriteHeader(status)
				ew.Write(body)
			},
		}

		func() {
			returned := false
			defer func() { performance.finish(sc, returned) }()
			proxy.ServeHTTP(sc, attemptReq)
			returned = true
		}()
		cancelAttempt()
		// The admitted execution ends with the attempt: a retried attempt no
		// longer occupies this node, and a committed one has finished copying.
		curTicket.release()
		ctl.setAttempt(nil)
		switch {
		case att.redispatched():
			// Moved on by a cancel, not failed: it costs no dispatch.
			retry = true
			dispatches--
			proxyErr = "attempt aborted to re-dispatch the request"
		case claim.abandoned():
			// Name the real reason rather than the bare "context canceled" the
			// transport reports, which reads identically to a client hangup.
			proxyErr = errTargetGone.Error()
		}
		if !retry {
			finalStatus = sc.status
			committedSC = sc
			committed = true
		}
	}

	// Nothing committed and no attempt wrote the client's answer, so the loop
	// ended between attempts: either the budget ran out or, more usually, every
	// owner went away and none came back before the deadline. The dispatch
	// paths answer for themselves — a final transport failure writes its own
	// 502 and a non-retryable status is passed through — so this is the only
	// outcome left without a response.
	//
	// An abandoned request is excluded: writing to a client that has gone
	// achieves nothing, and attributing its end to a missing node would be a
	// fabricated reason for something that was the requester's own choice.
	if !committed && !respondedWithError && r.Context().Err() == nil {
		// Name what the loop was actually doing when it ran out, which is what
		// the caller needs to know. The three cases are distinguishable: the
		// budget is gone; or the deadline passed while an owner was still there
		// to try, because slow attempts can reach 10 minutes before they reach
		// five dispatches; or the deadline passed with the candidate set empty,
		// which is the wait for an owner to come back.
		reason := "no node advertising the requested model became available before the retry deadline"
		switch {
		case regenExhausted:
			reason = "no other eligible node remained after the previous one was excluded"
		case dispatches >= maxDispatches:
			reason = "every dispatch attempt failed"
		case len(round) > 0:
			reason = "the retry deadline passed while dispatch attempts were still failing"
		}
		proxyErr = reason
		finalStatus = http.StatusServiceUnavailable
		slog.Warn("proxy request exhausted its retry budget",
			"id", reqID, "method", r.Method, "path", r.URL.Path,
			"dispatches", dispatches, "duration_ms", time.Since(start).Milliseconds(),
			"reason", reason)
		body, mErr := json.Marshal(map[string]string{"error": reason})
		if mErr != nil {
			body = []byte(`{"error":"retry budget exhausted"}`)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		// Advisory only: the caller knows its own patience better than we do.
		w.Header().Set("Retry-After", "1")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write(body)
	}

	// Reporting happens in the deferred finalize above, so it survives the
	// ErrAbortHandler panic that a mid-copy error raises.
}
