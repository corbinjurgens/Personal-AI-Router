// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// controllifecycle.go holds the non-streaming ec endpoints: remote start and
// stop. They are plain request/response operations returning the resulting
// EngineStatus; start may run for the engine's bounded readiness allowance.
// They map onto the same lifecycle the local engine:start / engine:stop use
// (including adopted-orphan reclaim and foreign-listener decline, exactly as
// local engine:stop). A remote start is refused while the node is paused
// locally or the engine is saved Off locally (remoteVeto), and a remote stop
// saves Off with peer provenance, which a later remote start may undo.

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
)

const (
	controlStartPath = "/v1/engines/start"
	controlStopPath  = "/v1/engines/stop"
)

// startRequest is the POST /v1/engines/start body. A remote (ec) start pins the
// engine to loopback just like a local start: inference engines are never
// exposed on the LAN directly — a cluster peer reaches this node's engine only
// through the promoted proxy's pin-gated mTLS ingress, which forwards to the
// loopback backend. Binding the engine to 0.0.0.0 would re-open the very
// unauthenticated LAN surface this design removes.
type startRequest struct {
	Engine string `json:"engine"`
	Port   int    `json:"port,omitempty"`
}

// stopRequest is the POST /v1/engines/stop body.
type stopRequest struct {
	Engine string `json:"engine"`
}

func (s *controlServer) handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req startRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, maxControlBody)).Decode(&req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	if req.Engine == "" {
		http.Error(w, `"engine" is required`, http.StatusBadRequest)
		return
	}
	if req.Port < 0 || req.Port > 65535 {
		http.Error(w, "port must be between 0 and 65535", http.StatusBadRequest)
		return
	}
	ctx, done, err := s.exec.beginRemote(r.Context(), req.Engine)
	if err != nil {
		writeRemoteError(w, err)
		return
	}
	defer done()
	if err := s.exec.StartWith(ctx, req.Engine, startOpts{Port: req.Port, Bind: "127.0.0.1", Origin: intentPeer}); err != nil {
		writeRemoteError(w, vetoCause(ctx, err))
		return
	}
	s.writeStatus(w, req.Engine)
}

// writeRemoteError answers a paired node's start or load that was refused or
// failed: a local veto is a typed 409 (codeLocalVeto), an unknown node policy
// a 503, anything else a 500.
func writeRemoteError(w http.ResponseWriter, err error) {
	var veto *localVetoError
	switch {
	case errors.As(err, &veto):
		writeCodedError(w, codeLocalVeto, veto.Error())
	case errors.Is(err, errNoProjection):
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
	default:
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *controlServer) handleStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req stopRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, maxControlBody)).Decode(&req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	if req.Engine == "" {
		http.Error(w, `"engine" is required`, http.StatusBadRequest)
		return
	}
	if err := s.exec.StopFrom(req.Engine, intentPeer); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.writeStatus(w, req.Engine)
}

// writeStatus responds with the engine's current EngineStatus.
func (s *controlServer) writeStatus(w http.ResponseWriter, engine string) {
	st, err := s.exec.Status(engine)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(st)
}
