// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"

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

	body, model, err := bufferBodyAndModel(w, r, f.host.requestBodyLimit())
	if err != nil {
		writeIngressError(w, http.StatusRequestEntityTooLarge, "body-too-large",
			"request body exceeds the proxy's size limit")
		return
	}
	// The execution's own context, so cancelActive can stop it without the
	// peer having gone anywhere.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	decision := f.host.admission.admit(ctx, admissionRequest{
		engine:  f.profile.Name,
		model:   model,
		waitCap: parseAdmissionWait(r.Header),
		cancel:  cancel,
	})
	if decision.err != nil {
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

	target, ok := f.localBackendTarget()
	if !ok {
		writeIngressError(w, http.StatusServiceUnavailable, "no-local-backend",
			"no local inference backend is available on this node")
		return
	}
	if decision.hasProfile {
		body = mergeDefaults(body, decision.profile.RequestOptions)
	}
	out := r.WithContext(ctx)
	out.Header = r.Header.Clone()
	out.Header.Del(nodepolicy.AdmissionWaitHeader)
	setReplayBody(out, body)
	slog.Debug("cluster ingress forwarding to local backend",
		"peer", peer, "method", r.Method, "path", r.URL.Path, "target", target.Host, "model", model)
	f.reverseProxyToLocal(w, out, target)
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
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.Host = target.Host
		},
		Transport: f.host.plainHTTPTransport(),
		ErrorHandler: func(ew http.ResponseWriter, _ *http.Request, err error) {
			slog.Warn("cluster ingress upstream error", "target", target.Host, "err", err)
			writeIngressError(ew, http.StatusBadGateway, "backend-error", "local inference backend error")
		},
	}
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
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	body, err := json.Marshal(map[string]string{"error": msg, "code": code})
	if err != nil {
		body = []byte(`{"error":"ingress error"}`)
	}
	_, _ = w.Write(body)
}
