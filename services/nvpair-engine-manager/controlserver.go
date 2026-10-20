// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// controlserver.go is engine-manager's cluster-scoped remote-control surface —
// the "ec" service. Unlike the plain /v1/models endpoint (em, model names are
// not secret), this surface performs privileged operations on behalf of a
// remote node (remote install/pull/start/stop + engine status), so it is
// pin-based mTLS and admits a caller ONLY while this node is a cluster member.
// Any caller that isn't a byte-for-byte pinned cluster peer is turned away with
// a real HTTP 403 rather than an opaque handshake failure — the same decision
// nvpair-errors and nvpair-workload-manager make via nvpair-shared/clustertrust.
//
// The listener is bound whenever the parent opts in with --control-port, and its
// TLS identity is resolved per handshake from the live mesh: while this node is
// not a member there is no leaf to present, so the handshake is refused and
// nothing privileged is reachable. Binding on membership instead would make this
// surface unavailable for the life of the process to a node that joined a cluster
// after engine-manager started.

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"time"

	"nvpair-shared/clustertrust"
	settings "nvpair-shared/enginesettings"
)

// controlEnginesPath returns this node's engine install/run status.
const controlEnginesPath = "/v1/engines"

const switchRecoveryReadyMethod = "engine:switch-recovery-ready"

// completeSwitchRecovery is parent-only: the pinned stdio worker handle and
// incarnation prevent an acknowledgement for an old process opening a new one.
func (e *Executor) completeSwitchRecovery(incarnation string) error {
	e.switchAdmission.Lock()
	defer e.switchAdmission.Unlock()
	if incarnation == "" || incarnation != e.switchIncarnation {
		return errors.New("switch recovery acknowledgement has a different manager incarnation")
	}
	e.desired.mu.Lock()
	defer e.desired.mu.Unlock()
	state, err := e.desired.load()
	if err != nil {
		return err
	}
	if cp := state.PendingSwitch; cp != nil && cp.Phase != settings.SwitchCheckpointReleased {
		return errors.New("switch recovery still owns an unreleased checkpoint")
	}
	e.switchRecoveryPending.Store(false)
	return nil
}

// controlServer holds the dependencies the ec handlers share.
type controlServer struct {
	exec *Executor
	mesh *clustertrust.Mesh
}

// controlClass is a registered ec route's authorization class. Any pinned peer
// may read; a mutation additionally needs the caller's remote-management grant.
type controlClass int

const (
	// controlUnclassified is the zero value: a route that never declared its
	// class fails closed instead of reaching a handler.
	controlUnclassified controlClass = iota
	controlRead
	controlMutation
)

func (c controlClass) String() string {
	switch c {
	case controlRead:
		return "read"
	case controlMutation:
		return "mutation"
	default:
		return "unclassified"
	}
}

// controlRoute is one entry in the ec route registry: the single place that
// classifies a (path, method) pair as read or mutation. Adding a route is one
// entry in controlServer.routes, so the registry and the served mux cannot
// disagree about what a route is.
type controlRoute struct {
	path   string
	method string
	class  controlClass
	// family is the bounded operation name the decision audit logs.
	family string
	handle pinnedPeerHandler
}

// ignoreCaller adapts a handler that needs no caller identity to the
// caller-preserving handler shape the registry and pin gate use.
func ignoreCaller(h http.HandlerFunc) pinnedPeerHandler {
	return func(w http.ResponseWriter, r *http.Request, _ string) { h(w, r) }
}

// pinnedCallerGate re-supplies an already verified caller to a handler that
// still expects the pin gate's shape (switchHandler). Identity always comes
// from the certificate, never from the request.
func pinnedCallerGate(caller string) func(pinnedPeerHandler) http.HandlerFunc {
	return func(h pinnedPeerHandler) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) { h(w, r, caller) }
	}
}

// requirePin gates a handler on cluster-peer mTLS, sharing the one gate the
// model surface uses (requirePinnedPeer in httpserver.go) so both of this
// service's LAN surfaces make the same decision the same way. It delegates to
// requirePinCaller rather than repeating the decision.
func (s *controlServer) requirePin(h http.HandlerFunc) http.HandlerFunc {
	return s.requirePinCaller(ignoreCaller(h))
}

// requirePinCaller is requirePin for a handler that also needs to know which
// peer is calling.
func (s *controlServer) requirePinCaller(h pinnedPeerHandler) http.HandlerFunc {
	return requirePinnedCaller(s.mesh, h)
}

// authorize is the one ec authorization implementation after the mTLS pin
// check has produced the verified caller UUID. It enforces the registered
// method, keeps recovery's read exceptions, and requires a fresh
// Mesh.RemoteManagement(caller) grant before a mutation handler runs. A
// request it cannot classify never reaches a handler.
func (s *controlServer) authorize(route controlRoute, h pinnedPeerHandler) pinnedPeerHandler {
	return func(w http.ResponseWriter, r *http.Request, caller string) {
		if h == nil || route.class == controlUnclassified || route.method == "" || route.path != r.URL.Path {
			slog.Error("refusing unclassified engine control route",
				"path", r.URL.Path, "method", r.Method, "route", route.path)
			http.Error(w, "engine control route is not classified", http.StatusInternalServerError)
			return
		}
		request, method := r, r.Method
		if method == http.MethodHead && route.class == controlRead && route.method == http.MethodGet {
			// HEAD is a read of a registered read route; handlers only know GET.
			request = r.Clone(r.Context())
			request.Method = http.MethodGet
			method = http.MethodGet
		}
		if method != route.method {
			w.Header().Set("Allow", allowedMethods(route))
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if route.class == controlRead {
			logControlDecision(s.mesh, caller, route, "allowed", "", "read route")
			h(w, request, caller)
			return
		}
		// Fresh read: never cache an allow, so a revocation between requests
		// already denies the next one. An unavailable, missing, corrupt or
		// migration-pending store denies as well.
		permission := s.mesh.RemoteManagement(caller)
		decision := "denied"
		if permission.Allowed {
			decision = "allowed"
		}
		logControlDecision(s.mesh, caller, route, decision, permission.Revision, permission.Reason)
		if !permission.Allowed {
			// A pinned peer's refused management operation. Typed 409 and a
			// bounded explanation; never raw filesystem state.
			writeCodedError(w, codeRemoteManagementDenied, remoteManagementDeniedMessage)
			return
		}
		if !s.allowControlDuringRecovery(w, r, route.class) {
			return
		}
		h(w, request, caller)
	}
}

// allowedMethods is the Allow header value for a registered route. A registered
// GET read route also answers HEAD.
func allowedMethods(route controlRoute) string {
	if route.class == controlRead && route.method == http.MethodGet {
		return http.MethodGet + ", " + http.MethodHead
	}
	return route.method
}

// Keep read-only POST queries available along with inventory and event streams;
// authoritative projections and conservative recovery use stdio, not ec. The
// registry's class replaces the old hand-kept path list.
func (s *controlServer) allowControlDuringRecovery(w http.ResponseWriter, r *http.Request, class controlClass) bool {
	if s.exec == nil || !s.exec.switchRecoveryPending.Load() || class == controlRead ||
		r.Method == http.MethodGet || r.Method == http.MethodHead {
		return true
	}
	http.Error(w, "engine manager switch recovery is pending; try again shortly", http.StatusServiceUnavailable)
	return false
}

// mux builds the ec routes from the registry. Split from the listener so tests
// can exercise the handlers over httptest without binding a real port.
func (s *controlServer) mux() *http.ServeMux {
	mux := http.NewServeMux()
	for _, route := range s.routes() {
		mux.HandleFunc(route.path, s.requirePinCaller(s.authorize(route, route.handle)))
	}
	return mux
}

// routes is the ec registry: every route this service serves, classified once.
func (s *controlServer) routes() []controlRoute {
	routes := []controlRoute{
		{controlEnginesPath, http.MethodGet, controlRead, "engine-inventory", ignoreCaller(s.handleEngines)},
		{controlInstallPath, http.MethodPost, controlMutation, "engine-install", ignoreCaller(s.handleInstall)},
		{controlPullPath, http.MethodPost, controlMutation, "engine-pull", ignoreCaller(s.handlePull)},
		{controlLoadPath, http.MethodPost, controlMutation, "engine-load", ignoreCaller(s.handleLoad)},
		{controlUnloadPath, http.MethodPost, controlMutation, "engine-unload", ignoreCaller(s.handleUnload)},
		{controlDeletePath, http.MethodPost, controlMutation, "engine-delete", ignoreCaller(s.handleDelete)},
		{controlModelFilesPath, http.MethodGet, controlRead, "model-files", ignoreCaller(s.handleModelFiles)},
		{controlModelFilePath, http.MethodGet, controlRead, "model-file", ignoreCaller(s.handleModelFile)},
		{controlStartPath, http.MethodPost, controlMutation, "engine-start", ignoreCaller(s.handleStart)},
		{controlStopPath, http.MethodPost, controlMutation, "engine-stop", ignoreCaller(s.handleStop)},
		{controlRemoteManagementPath, http.MethodGet, controlRead, remoteManagementReadFamily, s.handleRemoteManagement},
	}
	routes = append(routes, s.settingsRoutes()...)
	routes = append(routes, s.switchRoutes()...)
	routes = append(routes, s.policyRoutes()...)
	return routes
}

// handleEngines serves this node's engine status list to a pinned peer — the
// same data engine:get-installed returns over stdio.
func (s *controlServer) handleEngines(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"engines": s.exec.GetInstalled()})
}

// serveControl runs the ec mTLS control surface on 0.0.0.0:port until ctx is
// cancelled. A bind failure is non-fatal: stdio engine management (and local
// peers' remote calls being unavailable) must not take the process down.
func serveControl(ctx context.Context, port int, exec *Executor, mesh *clustertrust.Mesh) {
	addr := net.JoinHostPort("0.0.0.0", strconv.Itoa(port))
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		slog.Warn("engine-manager control endpoint disabled (bind failed)", "addr", addr, "err", err)
		return
	}
	ln = tls.NewListener(ln, mesh.ServerTLSConfig())
	srv := &http.Server{
		Handler:           (&controlServer{exec: exec, mesh: mesh}).mux(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       clustertrust.PeerListenerIdleTimeout,
	}
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutCtx)
	}()
	slog.Info("serving engine control surface (mTLS)",
		"addr", addr, "path", controlEnginesPath, "clustered", mesh.Clustered())
	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Warn("engine-manager control endpoint exited", "err", err)
	}
}
