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
func (b *Broker) authorizeRemoteManagement(caller, family, operation string) error {
	if caller == "" {
		return nil
	}
	permission := clustertrust.Open(b.clusterDir).RemoteManagement(caller)
	b.logRemoteManagementDecision(caller, family, operation, permission)
	if permission.Allowed {
		return nil
	}
	return &remoteManagementDeniedError{}
}

// logRemoteManagementDecision records one permission decision. It carries no
// prompts, policy bodies, certificates or key material.
func (b *Broker) logRemoteManagementDecision(caller, family, operation string, permission clustertrust.RemoteManagementPermission) {
	outcome := "denied"
	if permission.Allowed {
		outcome = "allowed"
	}
	slog.Info("remote management decision",
		"caller", caller,
		"family", family,
		"outcome", outcome,
		"state", permission.State,
		"revision", permission.Revision,
		"reason", permission.Reason,
		"operation", operation,
	)
}
