// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// remote_management.go holds the broker's single remote-management decision
// helper. A paired node's management write is accepted only while this device's
// shared permission predicate still allows that peer; the predicate is
// clustertrust.Mesh.RemoteManagement, read fresh at each acceptance point. This
// is a recheck of the same permission, not a second policy and not a cache.

import (
	"errors"
	"log/slog"
	"strings"

	"nvpair-shared/clustertrust"
)

const (
	// codeRemoteManagementDenied refuses a paired node's management write on
	// this device. It matches engine-manager's -32012 so a refusal keeps its
	// meaning across the relay and back to the origin.
	codeRemoteManagementDenied = -32012
	// remoteManagementDeniedWire is the name engine-manager puts on the wire
	// for the same refusal.
	remoteManagementDeniedWire = "remote_management_denied"
	// remoteManagementDeniedMessage is the bounded, user-facing refusal. It is
	// deliberately free of device, policy and certificate details.
	remoteManagementDeniedMessage = "Remote management is blocked on this device. Change permission locally on that device."
)

// errRemoteCallerRequired refuses a peer relay notification that carries no
// caller. Such a request must never take a local caller's empty-caller path.
var errRemoteCallerRequired = errors.New("the requesting device is required")

func requireRemoteCaller(caller string) error {
	if strings.TrimSpace(caller) == "" {
		return &invalidParamsError{errRemoteCallerRequired}
	}
	return nil
}

// Replies belong to the same current worker/source pair that sent the request.
// In particular, never deliver an old request's reply through its replacement.
func (b *Broker) replyPeerRelay(source *engineManagerSource, worker *rpcWorker, method string, reply any) {
	if b.engineManagerSourceCurrent(source) && b.engineMgrPairCurrent(worker, source) {
		_ = worker.Notify(method, reply)
	}
}

// remoteManagementDeniedError is the typed refusal policyErrorCode and
// switchErrorDetail map to -32012.
type remoteManagementDeniedError struct{}

func (*remoteManagementDeniedError) Error() string { return remoteManagementDeniedMessage }

// authorizeRemoteManagement is the acceptance-time recheck. A local caller
// (empty) is always allowed. For a paired node it reads the permission fresh
// and emits one secret-free line for that acceptance decision.
func (b *Broker) authorizeRemoteManagement(caller, family, operation string, boundRequest ...bool) error {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	defer r.mu.Unlock()
	return b.authorizeRemoteManagementLocked(r, caller, family, operation, boundRequest...)
}

// authorizeRemoteManagementLocked is shared by the r.mu-held policy and
// availability entry points and the settings/switch wrapper. Permission wins
// over pause, preserving revocation's code. Never acquire engineConfigMu here:
// settings and switch admission already own it before taking r.mu.
func (b *Broker) authorizeRemoteManagementLocked(r *policyRuntime, caller, family, operation string, boundRequest ...bool) error {
	if caller == "" {
		return nil
	}
	mesh := clustertrust.Open(b.clusterDir)
	permission := mesh.RemoteManagement(caller)
	b.logRemoteManagementDecision(caller, mesh.NodeUUID(), family, operation, permission)
	if !permission.Allowed {
		// The optional hint classifies automatic switch diagnostics only. It
		// never changes authorization or the existing switch audit family.
		if family == "bound-engine" || len(boundRequest) != 0 && boundRequest[0] {
			logBoundManagementRefusal("broker.remote_management", caller, permission.State, r.runID, permission.Diagnostic)
		}
		return &remoteManagementDeniedError{}
	}
	if r.managementBlockedLocked() {
		return &localVetoError{}
	}
	return nil
}

// Keep the deciding read's metadata: another lookup could already observe a
// different authority. Incarnations are public relationship comparison IDs,
// not credentials. No certificate, key, request body or freeform error is logged.
func logBoundManagementRefusal(site, peer, permission, run string, diagnostic clustertrust.PeerServiceDiagnostic) {
	slog.Warn("bound engine management refused",
		"site", site,
		"reason", diagnostic.Predicate,
		"node_id", diagnostic.NodeID,
		"peer_id", peer,
		"cluster_id", diagnostic.ClusterID,
		"epoch", diagnostic.Epoch,
		"counter", diagnostic.Counter,
		"io_error_code", diagnostic.ErrorCode,
		"expected_incarnation", diagnostic.ExpectedIncarnation,
		"observed_incarnation", diagnostic.ObservedIncarnation,
		"changed_component_authority_compared", diagnostic.ChangedComponentCompared,
		"changed_component_authority_same", diagnostic.ChangedComponentSame,
		"proxy_run_id", run,
		"permission", permission,
	)
}

// logRemoteManagementDecision records one permission decision. It carries no
// prompts, policy bodies, certificates or key material.
func (b *Broker) logRemoteManagementDecision(caller, target, family, operation string, permission clustertrust.RemoteManagementPermission) {
	outcome := "denied"
	if permission.Allowed {
		outcome = "allowed"
	}
	slog.Info("remote management decision",
		"caller", caller,
		"target", target,
		"family", family,
		"outcome", outcome,
		"state", permission.State,
		"revision", permission.Revision,
		"reason", permission.Reason,
		"operation", operation,
	)
}
