// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// The engine-independent routing listener: one loopback port, owned by the
// process rather than by an engine facade, that serves the configured hybrid
// aliases. It follows the activated node policy (routing.listener), so a proxy
// restarted by the broker re-establishes it when the broker replays the policy.
//
// Until the hybrid cutover it lists aliases and nothing else. Every inference
// route answers feature-unavailable before reading the body, so no admission,
// wake, workload, engine or remote activity can follow from it. It has no
// engine profile, no facade and no discovery state.

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/engines"
	"nvpair-shared/errors"
)

// routingListenerBlockedID is the service-error id for a listener that could
// not bind. The broker relays process-scoped errors:report to the pipeline, and
// the proxy clears it as soon as the listener binds or is disabled.
const routingListenerBlockedID = engines.ProxyComponent + ":routing-listener-blocked"

// Machine-readable codes in a routing listener refusal.
const (
	routingCodeHostNotAllowed   = "host_not_allowed"
	routingCodeOriginNotAllowed = "origin_not_allowed"
	routingCodeNotFound         = "not_found"
	routingCodeMethodNotAllowed = "method_not_allowed"
	routingCodeQueryNotAllowed  = "query_not_allowed"
	routingCodePreflightDenied  = "preflight_not_allowed"
	// routingCodeFeatureUnavailable matches the facades' alias refusal code.
	routingCodeFeatureUnavailable = "feature_unavailable"
	// routingReasonNotInferenceReady: the listener has no dispatch path until
	// the hybrid cutover.
	routingReasonNotInferenceReady = "routing_listener_not_inference_ready"
)

// routingModelsPath is the one non-inference route the listener serves.
const routingModelsPath = "/v1/models"

// routingAllowedHeaders are the request headers a preflight may ask for.
var routingAllowedHeaders = []string{"content-type", "authorization"}

// routingDrainGrace bounds how long a rebind or disable waits for connections to
// finish. Every response here is a small body written by an instant handler, so
// anything still open after this is a client that has stalled, and the control
// loop must not wait on it.
const routingDrainGrace = 250 * time.Millisecond

// routingPreflightMaxAge is how long, in seconds, a browser may cache an
// allowed preflight.
const routingPreflightMaxAge = "600"

// routingListener owns at most one raw listener and its HTTP server, bound to
// the loopback address on the port the activated policy names. It retains the
// raw listener so a stop closes the socket even before Serve registers it.
type routingListener struct {
	host *Proxy

	// mu serializes reconcile and close, and guards everything below.
	mu     sync.Mutex
	closed bool
	srv    *http.Server
	ln     net.Listener
	port   int
	// beforeServe is an optional per-instance hook, captured before launch.
	beforeServe func()
	// cleared records that this process has told the pipeline the listener is
	// not blocked. It starts false so the first outcome after a (re)start also
	// clears an error a previous proxy process left behind.
	cleared bool
}

// reconcile makes the listener match the active policy: bound to the configured
// port when enabled, absent otherwise. A changed port closes and drains the old
// listener before the new one binds. A bind failure leaves no listener and a
// visible error, never another port. It is idempotent, so a replayed policy
// retries a collision without disturbing a healthy listener.
func (l *routingListener) reconcile() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}
	want := l.host.admission.currentPolicy().Routing.Listener
	if !want.Enabled {
		l.stopLocked()
		l.clearBlockedLocked()
		return
	}
	if l.srv != nil && l.port == want.Port {
		l.clearBlockedLocked()
		return
	}
	l.stopLocked()

	address := net.JoinHostPort("127.0.0.1", strconv.Itoa(want.Port))
	ln, err := listenLoopbackExclusive(want.Port)
	if err != nil {
		l.reportBlockedLocked(address, err)
		return
	}
	ctx := l.host.serveCtx
	if ctx == nil {
		ctx = context.Background()
	}
	srv := &http.Server{
		Handler:           &routingHandler{host: l.host, port: want.Port},
		BaseContext:       func(net.Listener) context.Context { return ctx },
		ReadHeaderTimeout: proxyReadHeaderTimeout,
		WriteTimeout:      proxyReadHeaderTimeout,
		IdleTimeout:       proxyServerIdleTimeout,
		MaxHeaderBytes:    64 << 10,
	}
	beforeServe := l.beforeServe
	go func() {
		if beforeServe != nil {
			beforeServe()
		}
		if err := srv.Serve(ln); err != nil && !stderrors.Is(err, http.ErrServerClosed) {
			slog.Error("routing listener exited", "address", address, "err", err)
		}
	}()
	l.srv, l.ln, l.port = srv, ln, want.Port
	l.clearBlockedLocked()
	slog.Info("routing listener listening", "address", address)
}

// listenLoopbackExclusive binds 127.0.0.1:port only when nothing else holds the
// port on any address a loopback client can reach. Go sets SO_REUSEADDR on its
// listeners, and on macOS that lets a loopback bind succeed beside a socket
// already bound to the wildcard address or to ::1: both would then be listening,
// and a client of the other holder (or one dialling localhost) could be answered
// by this one. So the port is first bound, then released, on the IPv4 wildcard
// (which fails while any IPv4 or dual-stack wildcard holder exists, foreign or
// one of this process's facades) and on ::1. A holder on 127.0.0.1 itself fails
// the loopback bind. The probes are released before the loopback bind because a
// platform that refuses overlapping binds would otherwise make the listener
// collide with its own probe. That leaves a window of two system calls in which
// a foreign wildcard bind can still succeed on macOS; spec section 12.6 records it.
func listenLoopbackExclusive(port int) (net.Listener, error) {
	p := strconv.Itoa(port)
	probe, err := net.Listen("tcp", net.JoinHostPort("0.0.0.0", p))
	if err != nil {
		return nil, fmt.Errorf("port %d is not free on all addresses: %w", port, err)
	}
	_ = probe.Close()
	if err := probeIPv6Loopback(p); err != nil {
		return nil, fmt.Errorf("port %d is not free on the IPv6 loopback address: %w", port, err)
	}
	return net.Listen("tcp", net.JoinHostPort("127.0.0.1", p))
}

// probeIPv6Loopback fails when another socket holds ::1:port. A host with no
// IPv6 loopback address cannot have such a holder, so it is not an error: the
// bind failing is only believed when an ephemeral ::1 bind succeeds.
func probeIPv6Loopback(port string) error {
	probe, err := net.Listen("tcp", net.JoinHostPort("::1", port))
	if err == nil {
		return probe.Close()
	}
	capable, capErr := net.Listen("tcp", net.JoinHostPort("::1", "0"))
	if capErr != nil {
		return nil
	}
	_ = capable.Close()
	return err
}

// listenWildcard binds an engine facade's all-address listener, refusing the
// port this listener holds. The reverse of listenLoopbackExclusive: on macOS a
// wildcard bind succeeds beside a loopback holder, so only this process's own
// record of the port can refuse it. The lock makes the check and the bind one
// step against a concurrent reconcile, which probes the wildcard under it.
func (l *routingListener) listenWildcard(port int) (net.Listener, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.srv != nil && l.port == port {
		return nil, fmt.Errorf("port %d is held by the routing listener", port)
	}
	return net.Listen("tcp", fmt.Sprintf(":%d", port))
}

// close stops the listener for good; a later reconcile does nothing.
func (l *routingListener) close(ctx context.Context) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.closed = true
	l.stopLockedWith(ctx)
}

func (l *routingListener) stopLocked() {
	ctx, cancel := context.WithTimeout(context.Background(), routingDrainGrace)
	defer cancel()
	l.stopLockedWith(ctx)
}

// stopLockedWith drains the HTTP server, then forces it closed if the drain
// outlasts ctx. It always closes the owned raw listener, including before Serve
// registers it. Nothing else is touched: facades keep their own listeners.
func (l *routingListener) stopLockedWith(ctx context.Context) {
	if l.srv == nil {
		return
	}
	if err := l.srv.Shutdown(ctx); err != nil {
		_ = l.srv.Close()
	}
	if err := l.ln.Close(); err != nil && !stderrors.Is(err, net.ErrClosed) {
		slog.Warn("routing listener close failed", "port", l.port, "err", err)
	}
	slog.Info("routing listener stopped", "port", l.port)
	l.srv, l.ln, l.port = nil, nil, 0
}

func (l *routingListener) reportBlockedLocked(address string, bindErr error) {
	l.cleared = false
	message := fmt.Sprintf(
		"NVPAIR could not open the routing listener on %s: %v. Stop the application using that port or choose another routing listener port, then save the policy again. No process was stopped.",
		address, bindErr)
	if err := l.host.codec.Notify("errors:report", errors.ServiceError{
		ID:       routingListenerBlockedID,
		Message:  message,
		Severity: "error",
		Action:   "none",
	}); err != nil {
		slog.Debug("failed to report routing listener bind failure", "err", err)
	}
	slog.Warn("routing listener unavailable", "address", address, "err", bindErr)
}

func (l *routingListener) clearBlockedLocked() {
	if l.cleared {
		return
	}
	l.cleared = true
	if err := l.host.codec.Notify("errors:clear", errors.ClearParams{ID: routingListenerBlockedID}); err != nil {
		slog.Debug("failed to clear routing listener error", "err", err)
	}
}

// routingHandler serves one bound listener. It reads the policy on every
// request, so allowed origins and the alias list follow activation without a
// rebind.
type routingHandler struct {
	host *Proxy
	port int
}

func (h *routingHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Vary", "Origin")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	// No handler reads a request body. Closing after the answer keeps net/http
	// from reading the unread body first, which would hold the answer back for
	// as long as a client stalls mid-body.
	if r.ContentLength != 0 {
		w.Header().Set("Connection", "close")
	}

	if !h.hostAllowed(r.Host) {
		writeRoutingError(w, http.StatusForbidden, routingCodeHostNotAllowed, "", "request host is not a loopback name of this listener")
		return
	}
	if origins := r.Header.Values("Origin"); len(origins) > 0 {
		allowed := h.host.admission.currentPolicy().Routing.Listener.AllowedOrigins
		if len(origins) != 1 || !slices.Contains(allowed, origins[0]) {
			writeRoutingError(w, http.StatusForbidden, routingCodeOriginNotAllowed, "", "browser origin is not allowed")
			return
		}
		w.Header().Set("Access-Control-Allow-Origin", origins[0])
	}

	method, ok := routingMethodFor(r.URL.Path)
	switch {
	case !ok:
		writeRoutingError(w, http.StatusNotFound, routingCodeNotFound, "", "no such route")
	case r.Method == http.MethodOptions:
		h.preflight(w, r, method)
	case r.Method != method:
		w.Header().Set("Allow", method+", OPTIONS")
		writeRoutingError(w, http.StatusMethodNotAllowed, routingCodeMethodNotAllowed, "", "method not allowed on this route")
	case r.URL.Path == routingModelsPath:
		h.listModels(w)
	case r.URL.RawQuery != "" || r.URL.ForceQuery:
		writeRoutingError(w, http.StatusBadRequest, routingCodeQueryNotAllowed, "", "inference routes accept no query string")
	default:
		writeRoutingError(w, http.StatusNotImplemented, routingCodeFeatureUnavailable, routingReasonNotInferenceReady,
			"hybrid routing is not available for inference yet")
	}
}

// hostAllowed accepts only a loopback name this listener is bound under, on its
// own port, so a page that rebinds a hostname to 127.0.0.1 is refused.
func (h *routingHandler) hostAllowed(host string) bool {
	name, port, err := net.SplitHostPort(host)
	if err != nil || port != strconv.Itoa(h.port) {
		return false
	}
	name = strings.ToLower(name)
	return name == "127.0.0.1" || name == "localhost"
}

// preflight answers a CORS preflight. An Origin that reached here is already
// allowed; a request without one is not a preflight and gets no CORS answer.
func (h *routingHandler) preflight(w http.ResponseWriter, r *http.Request, routeMethod string) {
	if r.Header.Get("Origin") == "" || r.Header.Get("Access-Control-Request-Method") == "" {
		w.Header().Set("Allow", routeMethod+", OPTIONS")
		writeRoutingError(w, http.StatusMethodNotAllowed, routingCodeMethodNotAllowed, "", "method not allowed on this route")
		return
	}
	w.Header().Add("Vary", "Access-Control-Request-Method")
	w.Header().Add("Vary", "Access-Control-Request-Headers")
	if r.Header.Get("Access-Control-Request-Method") != routeMethod {
		writeRoutingError(w, http.StatusForbidden, routingCodePreflightDenied, "", "requested method is not allowed on this route")
		return
	}
	for _, value := range r.Header.Values("Access-Control-Request-Headers") {
		for _, name := range strings.Split(value, ",") {
			name = strings.ToLower(strings.TrimSpace(name))
			if name != "" && !slices.Contains(routingAllowedHeaders, name) {
				writeRoutingError(w, http.StatusForbidden, routingCodePreflightDenied, "", "requested header is not allowed")
				return
			}
		}
	}
	w.Header().Set("Access-Control-Allow-Methods", routeMethod+", OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.Header().Set("Access-Control-Max-Age", routingPreflightMaxAge)
	w.WriteHeader(http.StatusNoContent)
}

// listModels answers with the synthetic alias list. It reads only the active
// policy: no facade, upstream request or engine CORS is involved.
func (h *routingHandler) listModels(w http.ResponseWriter) {
	data := hybridAliasRecords(h.host.admission.currentPolicy(), nil, nil)
	if data == nil {
		data = []json.RawMessage{}
	}
	body, err := json.Marshal(struct {
		Object string            `json:"object"`
		Data   []json.RawMessage `json:"data"`
	}{Object: "list", Data: data})
	if err != nil {
		writeRoutingError(w, http.StatusInternalServerError, "internal_error", "", "failed to encode model list")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(body)
}

// routingMethodFor classifies a path: the alias list is a GET and every
// inference route any engine profile declares is a POST. The inference set is
// read from the profiles so the listener rejects whatever a facade would route.
func routingMethodFor(path string) (method string, ok bool) {
	if path == routingModelsPath {
		return http.MethodGet, true
	}
	for _, p := range profiles {
		if isInferenceRequest(p, http.MethodPost, path) {
			return http.MethodPost, true
		}
	}
	return "", false
}

// writeRoutingError answers with a structured error. reason is omitted when
// empty.
func writeRoutingError(w http.ResponseWriter, status int, code, reason, message string) {
	body, err := json.Marshal(struct {
		Error  string `json:"error"`
		Code   string `json:"code"`
		Reason string `json:"reason,omitempty"`
	}{Error: message, Code: code, Reason: reason})
	if err != nil {
		body = []byte(`{"error":"internal error","code":"internal_error"}`)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
