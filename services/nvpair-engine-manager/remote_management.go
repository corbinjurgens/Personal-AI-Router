// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// remote_management.go is the remote-management permission surface: the
// caller-only ec read (GET /v1/remote-management), its outbound
// engine:remote-management-get RPC, and the decision audit line the ec
// authorization gate emits. The permission itself is owned by
// nvpair-cluster-manager and read through clustertrust.Mesh.RemoteManagement;
// this service only asks it, freshly per request, and never caches an allow.
//
// A new local client must treat an old target that answers 404 as
// "unsupported", not as protected: enforcement cannot be retrofitted into an
// executable that does not have the gate.

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"

	"nvpair-shared/clustertrust"
)

const (
	// controlRemoteManagementPath is the caller-only permission read.
	controlRemoteManagementPath = "/v1/remote-management"
	// remoteManagementReadFamily is the registry's operation family for it.
	remoteManagementReadFamily = "remote-management-read"
	// remoteManagementGetMethod is the outbound RPC a local client uses.
	remoteManagementGetMethod = "engine:remote-management-get"

	remoteManagementVersion = 1
	// remoteManagementWireCode is the ec 409 body/stream wire name for
	// codeRemoteManagementDenied.
	remoteManagementWireCode = "remote_management_denied"
	// remoteManagementDeniedMessage is the bounded explanation a refused peer
	// receives. It never carries filesystem or storage detail.
	remoteManagementDeniedMessage = "Remote management is blocked on this device. Change permission locally on that device."

	remoteManagementStateAllowed     = "allowed"
	remoteManagementStateDenied      = "denied"
	remoteManagementStateUnreviewed  = "unreviewed"
	remoteManagementStateUnavailable = "unavailable"
	// remoteManagementStateUnsupported is this node's report for a target that
	// does not serve the permission read at all.
	remoteManagementStateUnsupported = "unsupported"
)

// remoteManagementView is the ec read's body: only the authenticated caller's
// effective permission. It has no peer selector and never carries the target's
// grant list.
type remoteManagementView struct {
	Version  int    `json:"version"`
	Allowed  bool   `json:"allowed"`
	Revision string `json:"revision"`
	State    string `json:"state"`
}

func (v remoteManagementView) valid() bool {
	if v.Version != remoteManagementVersion {
		return false
	}
	switch v.State {
	case remoteManagementStateAllowed, remoteManagementStateDenied,
		remoteManagementStateUnreviewed, remoteManagementStateUnavailable:
		return true
	}
	return false
}

// remoteManagementCapability is engine:remote-management-get's result. State
// carries the target's effective state, or "unsupported" when the target
// predates the read.
type remoteManagementCapability struct {
	Version  int    `json:"version"`
	Allowed  bool   `json:"allowed"`
	Revision string `json:"revision"`
	State    string `json:"state"`
}

// handleRemoteManagement answers the authenticated caller's effective
// permission. Identity comes only from the verified certificate; a peer
// selector in the query is ignored rather than honoured.
func (s *controlServer) handleRemoteManagement(w http.ResponseWriter, r *http.Request, caller string) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet+", "+http.MethodHead)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	permission := s.mesh.RemoteManagement(caller)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(remoteManagementView{
		Version:  remoteManagementVersion,
		Allowed:  permission.Allowed,
		Revision: permission.Revision,
		State:    permission.State,
	})
}

// remoteManagementGetParams is engine:remote-management-get's input.
type remoteManagementGetParams struct {
	NodeID string `json:"nodeId"`
}

// remoteManagementGet reads a peer's effective permission for this node through
// the existing pinned ec client (direct or reverse). It is a read: it needs no
// grant of its own.
func (m *Manager) remoteManagementGet(ctx context.Context, msg *Message) {
	var p remoteManagementGetParams
	if !m.parse(msg, &p) {
		return
	}
	if p.NodeID == "" {
		m.codec.RespondError(msg.ID, -32602, "nodeId is required")
		return
	}
	peer, ok := m.peers.lookup(p.NodeID)
	if !ok || m.mesh == nil {
		m.respondOrErr(msg, nil, &codedError{code: -32000,
			message: "remote management unavailable: device is offline or does not support it"})
		return
	}
	client, err := m.remoteClient(ctx, peer)
	if err != nil {
		m.respondOrErr(msg, nil, err)
		return
	}
	capability, err := client.getRemoteManagement(ctx)
	m.respondOrErr(msg, capability, err)
}

// logControlDecision emits the one audit line per gated ec request: verified
// caller, bounded operation family, allow/deny, permission revision and reason.
// Request paths, queries, bodies and secrets are never logged.
func logControlDecision(mesh *clustertrust.Mesh, caller string, route controlRoute, decision, revision, reason string) {
	target := ""
	if mesh != nil {
		target = mesh.NodeUUID()
	}
	args := []any{
		"caller", caller,
		"target", target,
		"operation", route.family,
		"class", route.class.String(),
		"decision", decision,
		"permissionRevision", revision,
		"reason", reason,
	}
	if decision == "allowed" {
		slog.Info("ec authorization decision", args...)
		return
	}
	slog.Warn("ec authorization decision", args...)
}
