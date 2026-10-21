// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"time"

	"nvpair-shared/nodepolicy"
)

const engineIdentityProbeHeader = "X-NVPAIR-Engine-Identity-Probe"

// localBackend is the explicit loopback engine the cluster mTLS ingress
// forwards to. It is supplied by the broker over node/set-local-backend and is
// deliberately NOT sourced from the discovery overlay: a request that arrived
// over the LAN mTLS ingress can only ever be dumped on this node's own local
// engine, never re-routed to a peer, so the ingress path is strictly terminal
// and cannot recurse or amplify.
type localBackend struct {
	Engine  string `json:"engine"`
	Host    string `json:"host"`
	Port    int    `json:"port"`
	Healthy bool   `json:"healthy"`
}

// currentLocalBackend snapshots the configured local engine.
func (f *facade) currentLocalBackend() localBackend {
	f.backendMu.RLock()
	defer f.backendMu.RUnlock()
	return f.backend
}

// setLocalBackend records (or, with a zero port / unhealthy flag, effectively
// clears) the local engine this facade's ingress serves.
//
// A non-loopback host is rejected rather than stored. The ingress forwards a
// pin-authenticated peer's request straight here without consulting discovery,
// so an off-box host would turn this node into a relay to an address chosen by
// whoever can reach the control channel. The broker only ever sends 127.0.0.1;
// this is the same defence-in-depth re-validation setLoopbackAlias performs on
// the alias the broker sends it.
func (f *facade) setLocalBackend(b localBackend) error {
	if b.Host != "" && !isLoopbackHost(b.Host) {
		return fmt.Errorf("local backend host %q is not loopback", b.Host)
	}
	f.storeLocalBackend(b)
	// A request waiting for this engine to wake re-checks its health now,
	// rather than at its deadline.
	if f.host != nil && f.host.admission != nil {
		f.host.admission.backendChanged()
	}
	return nil
}

func (f *facade) storeLocalBackend(b localBackend) {
	f.backendMu.Lock()
	defer f.backendMu.Unlock()
	f.backend = b
}

// localBackendTarget returns the loopback URL of the current local engine, and
// false when none is set/healthy (the ingress then answers 503 rather than
// forwarding). The host defaults to 127.0.0.1, and setLocalBackend refuses to
// store anything that is not loopback, so this is always a loopback target.
func (f *facade) localBackendTarget() (*url.URL, bool) {
	b := f.currentLocalBackend()
	if b.Port <= 0 || !b.Healthy {
		return nil, false
	}
	host := b.Host
	if host == "" {
		host = "127.0.0.1"
	}
	return &url.URL{Scheme: "http", Host: net.JoinHostPort(host, strconv.Itoa(b.Port))}, true
}

// handlePlain is the plaintext personality: it accepts requests only from
// loopback and hands them to the full local router (handleHTTP). A non-loopback
// caller — any LAN peer — is refused; peers must use the mTLS ingress. This is
// what closes the former open-relay exposure (the listener still binds all
// interfaces for the TLS personality, but plaintext is loopback-only).
func (f *facade) handlePlain(w http.ResponseWriter, r *http.Request) {
	if !isLoopbackRemote(r.RemoteAddr) {
		slog.Warn("rejected non-loopback plaintext request; cluster peers must use mTLS",
			"remote", r.RemoteAddr, "method", r.Method, "path", r.URL.Path)
		writeIngressError(w, http.StatusForbidden, "loopback-only",
			"plaintext requests are accepted only from loopback; cluster peers must use the mTLS ingress")
		return
	}
	// Engine-manager marks its private identity/action requests so this
	// compatibility facade can never be mistaken for the local Ollama backend.
	if r.Header.Get(engineIdentityProbeHeader) == "1" {
		writeIngressError(w, http.StatusConflict, "proxy-facade",
			"the compatibility facade is not the "+f.profile.DisplayName+" engine")
		return
	}
	f.handleHTTP(w, r)
}

// handleClusterIngress is the LAN mTLS personality: it authenticates the caller
// against this node's cluster pins and, once the peer is a trusted cluster
// member, forwards the request straight to the local loopback engine — exactly
// like the local plaintext path, with no route filtering. The mTLS pin is the
// sole authorization boundary (a trusted peer is treated like a local client),
// so the two personalities stay behaviorally identical toward the engine. It
// never calls resolveCandidates, so a peer request cannot be re-routed onward.
func (f *facade) handleClusterIngress(w http.ResponseWriter, r *http.Request) {
	// Re-derive membership and pins per request so a cluster left, or a peer
	// paired or removed, after startup is reflected immediately without a proxy
	// restart — a removed peer must stop being accepted right away, which is the
	// whole point of the gate.
	f.host.mesh.Refresh()
	peer, ok := f.host.mesh.VerifyClientPin(r)
	if !ok {
		writeIngressError(w, http.StatusForbidden, "cluster-auth",
			"client certificate is not a pinned member of this node's cluster")
		return
	}
	f.serveIngress(w, r, peer)
}

// serveIngress is the authenticated half of the cluster ingress. Inference is
// admitted first (FORK_DESIGN.md §3.3): the body's model is read within the
// same cap the router applies, admission decides, the model's profile
// requestOptions are merged in, and only then is the request forwarded. Every
// other route is forwarded as before.
func (f *facade) serveIngress(w http.ResponseWriter, r *http.Request, peer string) {
	start := time.Now()
	if isInferenceRequest(f.profile, r.Method, r.URL.Path) && !f.host.routingIdentityReady() {
		writeIngressError(w, http.StatusServiceUnavailable, "identity-unavailable", "routing identity is unavailable")
		return
	}
	strict, strictErr := parseStrictAdmission(r.Header)
	if strictErr != nil {
		writeIngressError(w, http.StatusBadRequest, "strict-admission", strictErr.Error())
		return
	}
	if strict != nil && strict.Request.OriginNodeID != peer {
		writeIngressError(w, http.StatusForbidden, "strict-admission", "strict request origin does not match the authenticated peer")
		return
	}
	if strict != nil && !isInferenceRequest(f.profile, r.Method, r.URL.Path) {
		writeIngressError(w, http.StatusBadRequest, "strict-admission", "strict admission requires inference")
		return
	}
	if !isInferenceRequest(f.profile, r.Method, r.URL.Path) {
		target, ok := f.localBackendTarget()
		if !ok {
			writeIngressError(w, http.StatusServiceUnavailable, "no-local-backend",
				"no local inference backend is available on this node")
			return
		}
		slog.Debug("cluster ingress forwarding to local backend",
			"peer", peer, "method", r.Method, "path", r.URL.Path, "target", target.Host)
		f.reverseProxyToLocal(w, r, target)
		return
	}

	r = r.WithContext(context.WithValue(r.Context(), routingDeadlineContextKey{}, start.Add(jobDeadline)))
	r, buffers := f.host.ownInferenceBuffers(r)
	w = &inferenceErrorWriter{ResponseWriter: w, ctx: r.Context()}
	defer buffers.close()
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	r = r.WithContext(ctx)
	bodyCtx, stopBodyPreparation := routingPreparationContext(r.Context())
	body, model, err := bufferInferenceBody(w, r.WithContext(bodyCtx), f.host.requestBodyLimit())
	stopBodyPreparation()
	if err != nil {
		writeIngressError(w, bufferRequestStatus(err), "request-body", "request body could not be buffered")
		return
	}
	sink := newResponseSink(w, cancel)
	w = sink
	r = r.WithContext(withResponseSink(ctx, sink))
	ctx = r.Context()
	if err := validateRequestIdentifiers(r, body); err != nil {
		writeIngressError(w, bufferRequestStatus(err), "request-identifier", err.Error())
		return
	}
	// Reject duplicate and unsupported model spellings before admission:
	// struct decoders and exact-key rewrites must select the same member.
	if ambiguousModelMember(body) {
		slog.Debug("cluster ingress rejected ambiguous model member",
			"peer", peer, "path", r.URL.Path)
		writeIngressError(w, http.StatusBadRequest, "ambiguous-model", ambiguousModelMessage)
		return
	}
	if _, err := parseOriginStreamRequest(r.URL.Path, body); err != nil {
		writeIngressError(w, http.StatusBadRequest, "invalid-stream", "stream must be a single boolean or null member")
		return
	}
	if strict != nil {
		f.serveStrictLocal(w, r, body, model, *strict, true)
		return
	}
	// Terminal ingress has one local transport and sink, not an origin key or
	// registry entry. Reuse the same request arbitration without another driver.
	owner := newRoutingRequest(func(string) { cancel() })
	owner.deadline, _ = ctx.Value(routingDeadlineContextKey{}).(time.Time)
	owner.sink = sink
	defer owner.finish()
	stopCancellation := context.AfterFunc(ctx, func() { owner.cancelWithReason("destination request cancelled") })
	defer stopCancellation()
	owner.watchPreOutput(ctx)
	// The execution's own context, so cancelActive can stop it without the
	// peer having gone anywhere.
	rawBody := body
	preparedCtx := ctx
	releaseBody, releasePreparation := func() {}, func() {}
	defer func() { releasePreparation(); releaseBody() }()
	admissionStarted := time.Now()
	prepareCtx, stopPreparation := routingPreparationContext(ctx)
	decision := f.host.admission.admit(prepareCtx, admissionRequest{
		engine:  f.profile.Name,
		model:   model,
		waitCap: parseAdmissionWait(r.Header),
		cancel:  cancel,
		prepare: func(ctx context.Context, profile nodepolicy.Profile, hasProfile bool, _ uint64) error {
			releasePreparation()
			releasePreparation = func() {}
			releaseBody()
			releaseBody = func() {}
			var defaults []byte
			if hasProfile {
				defaults = profile.RequestOptions
			}
			var err error
			body, releaseBody, err = transformBody(ctx, rawBody, "", false, defaults)
			if err != nil {
				releaseBody = func() {}
				return err
			}
			if hasProfile && !bodyModelMatches(body, model) {
				return errAmbiguousModel
			}
			preparedCtx, releasePreparation, err = preparePerformanceBuffers(ctx, body)
			if err != nil {
				releasePreparation = func() {}
			}
			return err
		},
	})
	stopPreparation()
	preparedCtx = rebasePerformancePreparation(ctx, preparedCtx)
	if decision.err != nil {
		decision.err = inferencePreparationRefusal(decision.err)
		if ctx.Err() == nil {
			if errors.Is(decision.err, errAmbiguousModel) {
				writeIngressError(w, http.StatusBadRequest, "ambiguous-model", ambiguousModelMessage)
			} else {
				writeIngressError(w, bufferRequestStatus(decision.err), "inference-buffer", "inference request could not be prepared")
			}
		}
		// The peer gave up while the request waited; nobody is left to answer.
		return
	}
	if decision.reject != "" {
		slog.Debug("cluster ingress admission rejected",
			"peer", peer, "path", r.URL.Path, "model", model, "reason", decision.reject)
		writeAdmissionRejection(w, decision.reject)
		return
	}
	// Deferred so it also runs when a mid-stream copy error unwinds this
	// handler by panic (see spec.md §5.4).
	defer decision.ticket.release()
	performanceWait := time.Since(admissionStarted)

	target, ok := f.localBackendTarget()
	if !ok {
		writeIngressError(w, http.StatusServiceUnavailable, "no-local-backend",
			"no local inference backend is available on this node")
		return
	}
	streamRequest, err := parseOriginStreamRequest(r.URL.Path, body)
	if err != nil {
		writeIngressError(w, http.StatusBadRequest, "invalid-stream", "invalid effective stream flag")
		return
	}
	attemptCtx, cancelAttempt := context.WithCancel(ctx)
	defer cancelAttempt()
	attempt, err := owner.newDispatchAttempt(attemptCtx, cancelAttempt, "")
	if err != nil {
		writeIngressError(w, http.StatusServiceUnavailable, "request-deadline", "inference request deadline expired")
		return
	}
	defer func() {
		result := sink.result()
		attempt.finished(publicAttemptTerminal{ProtocolComplete: result.ProtocolComplete, Delivered: result.DeliveryComplete})
	}()
	preparedCtx = rebasePerformancePreparation(attemptCtx, preparedCtx)
	out := r.WithContext(preparedCtx)
	out.Header = r.Header.Clone()
	out.Header.Del(nodepolicy.AdmissionWaitHeader)
	setReplayBody(out, body)
	if streamRequest.protocol != streamProtocolUnknown {
		out.Header.Set("Accept-Encoding", "identity")
	}
	slog.Debug("cluster ingress forwarding to local backend",
		"peer", peer, "method", r.Method, "path", r.URL.Path, "target", target.Host, "model", model)
	p := f.newLocalReverseProxy(target)
	performance, err := f.newPerformanceAttemptOwned(preparedCtx, model, r.URL.Path, body, performanceWait)
	if err != nil {
		writeIngressError(w, bufferRequestStatus(err), "inference-buffer", "inference request could not be prepared")
		return
	}
	p.Transport = performanceTransport{RoundTripper: p.Transport, attempt: performance}
	upstreamError := p.ErrorHandler
	p.ErrorHandler = func(ew http.ResponseWriter, req *http.Request, err error) {
		if owner.deadlineExpired() {
			writeIngressError(ew, http.StatusServiceUnavailable, "request-deadline", "inference request deadline expired")
			return
		}
		upstreamError(ew, req, err)
	}
	sc := &statusCapture{ResponseWriter: w, status: http.StatusOK, idle: idleClientWriteTimeout}
	p.ModifyResponse = func(resp *http.Response) error {
		if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
			if !attempt.acceptResponse() {
				return context.Canceled
			}
			resp.Body = &routingResponseBody{ReadCloser: resp.Body, attempt: attempt}
		}
		reader, err := installResponseStream(resp, streamRequest, r.Context(), 0, performance)
		if err != nil {
			if !errors.Is(err, errStreamFrameBudget) {
				return err
			}
			if closeErr := resp.Body.Close(); closeErr != nil {
				slog.Debug("stream bootstrap body close failed", "err", closeErr)
			}
			resp.Body = sink.interruptedBody(streamRequest.protocol, streamInterruptReasonBudget)
			resp.ContentLength = -1
			clearOriginStreamIntegrity(resp.Header)
			clearOriginStreamIntegrity(resp.Trailer)
		}
		setPeerStreamFraming(resp, streamRequest)
		sc.upstreamBody = &originResponseBody{ReadCloser: resp.Body, response: resp, managed: reader != nil}
		resp.Body = destinationResponseFraming(resp, sc.upstreamBody, streamRequest, reader, sink)
		return nil
	}
	returned := false
	defer func() {
		if !returned && sc.upstreamBody != nil {
			_ = sc.upstreamBody.Close()
		}
		performance.finish(sc, returned)
	}()
	p.ServeHTTP(sc, out)
	_ = sc.FlushError()
	returned = true
}

func setPeerStreamFraming(resp *http.Response, s originStreamRequest) {
	resp.Header.Del(peerStreamFramingHeader)
	resp.Trailer.Del(peerStreamFramingHeader)
	if s.protocol == streamProtocolUnknown && resp.StatusCode != http.StatusSwitchingProtocols {
		resp.Body = &originResponseBody{ReadCloser: resp.Body, response: resp}
	}
	if s.protocol != streamProtocolUnknown {
		resp.Header.Set(peerStreamFramingHeader, s.peerMetadata())
		// HTTP/1.1 needs chunked framing to carry the disposition trailer,
		// including for fixed-length nonstreaming engine responses.
		resp.ContentLength = -1
		resp.Header.Del("Content-Length")
		if resp.Trailer == nil {
			resp.Trailer = make(http.Header)
		}
		resp.Trailer[http.CanonicalHeaderKey(peerStreamFramingHeader)] = nil
	}
}

// setReplayBody installs a buffered body on an outbound request, keeping its
// length consistent with what will actually be sent.
func setReplayBody(r *http.Request, body []byte) {
	if body == nil {
		r.Body = http.NoBody
		r.ContentLength = 0
		r.Header.Del("Content-Length")
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Set("Content-Length", strconv.Itoa(len(body)))
}

// reverseProxyToLocal streams the request to the local engine, preserving
// cancellation (the request context is the proxy's root context, so a client
// disconnect or shutdown tears down the upstream call and stops generation).
func (f *facade) reverseProxyToLocal(w http.ResponseWriter, r *http.Request, target *url.URL) {
	f.newLocalReverseProxy(target).ServeHTTP(w, r)
}

func (f *facade) newLocalReverseProxy(target *url.URL) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			stripInternalRoutingHeaders(req.Header)
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.Host = target.Host
		},
		Transport: proxyResponseTransport{f.host.plainHTTPTransport()},
		ModifyResponse: func(resp *http.Response) error {
			setPeerStreamFraming(resp, originStreamRequest{})
			return nil
		},
		ErrorHandler: func(ew http.ResponseWriter, _ *http.Request, err error) {
			slog.Warn("cluster ingress upstream error", "target", target.Host, "err", err)
			writeIngressError(ew, http.StatusBadGateway, "backend-error", "local inference backend error")
		},
	}
}

// serveStrictLocal is shared by authenticated ingress and local self dispatch.
// The body is validated against the model's local qualification first, so an
// unqualified, oversized or unsupported request is refused before any wake,
// unload, slot or inference; the engine then receives the validated body with
// the profile's defaults merged in. No protocol-qualified completion signal
// exists yet; even an HTTP EOF is conservatively uncertain until the
// reconciliation owner proves termination. It must never be interpreted as
// engine-idle evidence.
func (f *facade) serveStrictLocal(w http.ResponseWriter, r *http.Request, body []byte, model string, req nodepolicy.StrictAdmissionRequest, peer bool) {
	admissionStarted := time.Now()
	if _, err := parseOriginStreamRequest(r.URL.Path, body); err != nil {
		writeIngressError(w, http.StatusBadRequest, "invalid-stream", "stream must be a single boolean or null member")
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	prepareCtx, stopPreparation := routingPreparationContext(ctx)
	defer stopPreparation()
	validated, err := f.host.admission.qualifyStrict(f.profile.Name, model, r.URL.Path, body, req.Identity, prepareCtx)
	err = inferencePreparationRefusal(err)
	if errors.Is(err, errAmbiguousModel) {
		writeIngressError(w, http.StatusBadRequest, "ambiguous-model", ambiguousModelMessage)
		return
	}
	var invalid *strictQualificationError
	if errors.As(err, &invalid) {
		writeIngressError(w, http.StatusBadRequest, "strict-qualification", err.Error())
		return
	}
	if err != nil {
		var capacity interface{ HTTPStatus() int }
		if errors.As(err, &capacity) {
			writeIngressError(w, capacity.HTTPStatus(), "inference-buffer", "inference request could not be prepared")
			return
		}
		writeStrictAdmissionError(w, err)
		return
	}
	body = validated.body
	streamRequest, err := parseOriginStreamRequest(r.URL.Path, body)
	if err != nil {
		writeIngressError(w, http.StatusBadRequest, "invalid-stream", "invalid effective stream flag")
		return
	}
	if !bodyModelMatches(body, model) {
		writeIngressError(w, http.StatusBadRequest, "ambiguous-model", ambiguousModelMessage)
		return
	}
	preparedCtx, releasePreparation, err := preparePerformanceBuffers(prepareCtx, body)
	if err != nil {
		writeIngressError(w, bufferRequestStatus(err), "inference-buffer", "inference request could not be prepared")
		return
	}
	defer releasePreparation()
	ticket, err := f.host.admission.AdmitStrict(prepareCtx, f.profile.Name, model, req, validated.generation)
	if err != nil {
		writeStrictAdmissionError(w, err)
		return
	}
	performanceWait := time.Since(admissionStarted)
	evidence := strictNotDispatched
	defer func() { ticket.Finish(evidence) }()
	t := ticket.(*strictTicket)
	f.host.admission.mu.Lock()
	t.base.cancel = cancel
	f.host.admission.mu.Unlock()
	// The request's remaining preparation bound also covers strict pre-output
	// waits. Stop it at the first body, alongside the existing strict timer.
	stopRequestDeadline := context.AfterFunc(prepareCtx, cancel)
	defer stopRequestDeadline()
	stop := t.watchPreOutput(cancel)
	defer stop()
	target, ok := f.localBackendTarget()
	if !ok {
		writeStrictAdmissionError(w, errStrictUnavailable)
		return
	}
	if err = ticket.BeforeStrictDispatch(); err != nil {
		writeStrictAdmissionError(w, err)
		return
	}
	evidence = strictUncertain
	preparedCtx = rebasePerformancePreparation(ctx, preparedCtx)
	out := r.WithContext(preparedCtx)
	out.Header = r.Header.Clone()
	stripInternalRoutingHeaders(out.Header)
	setReplayBody(out, body)
	if streamRequest.protocol != streamProtocolUnknown {
		out.Header.Set("Accept-Encoding", "identity")
	}
	p := f.newLocalReverseProxy(target)
	performance, err := f.newPerformanceAttemptOwned(preparedCtx, model, r.URL.Path, body, performanceWait)
	if err != nil {
		writeIngressError(w, bufferRequestStatus(err), "inference-buffer", "inference request could not be prepared")
		return
	}
	p.Transport = performanceTransport{RoundTripper: p.Transport, attempt: performance}
	p.ErrorHandler = func(ew http.ResponseWriter, _ *http.Request, _ error) {
		f.host.admission.mu.Lock()
		exhausted := !f.host.admission.strict.now().Before(t.deadline)
		f.host.admission.mu.Unlock()
		if deadline, ok := prepareCtx.Deadline(); ok {
			exhausted = exhausted || !time.Now().Before(deadline)
		}
		if exhausted {
			writeStrictAdmissionError(ew, context.DeadlineExceeded)
			return
		}
		writeIngressError(ew, http.StatusBadGateway, "backend-error", "local inference backend error")
	}
	sc := &statusCapture{ResponseWriter: w, status: http.StatusOK, idle: idleClientWriteTimeout}
	p.ModifyResponse = func(resp *http.Response) error {
		if peer {
			setPeerStreamFraming(resp, streamRequest)
		} else {
			setPeerStreamFraming(resp, originStreamRequest{})
		}
		f.host.admission.mu.Lock()
		remaining := t.deadline.Sub(f.host.admission.strict.now())
		f.host.admission.mu.Unlock()
		if deadline, ok := prepareCtx.Deadline(); ok {
			remaining = min(remaining, time.Until(deadline))
		}
		if remaining <= 0 {
			return context.DeadlineExceeded
		}
		b, err := awaitFirstBody(resp.Body, remaining)
		if err != nil {
			return err
		}
		resp.Body = b
		stopRequestDeadline()
		stop()
		reader, err := installResponseStream(resp, streamRequest, r.Context(), 0, performance)
		if err != nil {
			sink := responseSinkFrom(r.Context())
			if !errors.Is(err, errStreamFrameBudget) || sink == nil {
				return err
			}
			if closeErr := resp.Body.Close(); closeErr != nil {
				slog.Debug("stream bootstrap body close failed", "err", closeErr)
			}
			resp.Body = sink.interruptedBody(streamRequest.protocol, streamInterruptReasonBudget)
			resp.ContentLength = -1
			clearOriginStreamIntegrity(resp.Header)
			clearOriginStreamIntegrity(resp.Trailer)
		}
		sc.upstreamBody = &originResponseBody{ReadCloser: resp.Body, response: resp, managed: reader != nil}
		resp.Body = sc.upstreamBody
		if peer {
			resp.Body = destinationResponseFraming(resp, sc.upstreamBody, streamRequest, reader, responseSinkFrom(r.Context()))
		}
		return nil
	}
	returned := false
	defer func() {
		if !returned && sc.upstreamBody != nil {
			_ = sc.upstreamBody.Close()
		}
		performance.finish(sc, returned)
	}()
	p.ServeHTTP(sc, out)
	_ = sc.FlushError()
	returned = true
}

// isLoopbackRemote reports whether an http.Request RemoteAddr (host:port) is a
// loopback address (127.0.0.0/8 or ::1). An unparseable/empty RemoteAddr is not
// loopback, so it fails closed.
func isLoopbackRemote(remoteAddr string) bool {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// writeIngressError returns the actual failure without granting browser permissions.
func writeIngressError(w http.ResponseWriter, status int, code, msg string) {
	size := len(`{"code":,"error":}`) + quotedJSONStringSize(msg) + quotedJSONStringSize(code)
	var lease *bufferLease
	if ctx := errorBufferContext(w); ctx != nil {
		if owner := inferenceBuffers(ctx); owner != nil {
			var err error
			lease, err = owner.scope.tryAcquire(ctx, int64(size))
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
	body = append(body, `{"code":`...)
	body = appendJSONString(body, code)
	body = append(body, `,"error":`...)
	body = appendJSONString(body, msg)
	body = append(body, '}')
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
