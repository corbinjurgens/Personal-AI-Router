// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// policy_notifications.go is the one path that announces this node's own policy
// and availability to clients. Every announcement takes the publication lock,
// reads the state then current under r.mu, releases r.mu and sends while still
// holding the publication lock. Announcements therefore reach clients in the
// order of the state they carry: a save, activation or replay that finished
// earlier can never announce older state after a later one, and the last
// announcement carries the state as it last changed. A caller has nothing to
// pass in, so it cannot announce a document it captured before a newer save.
//
// Only the client write is serialized: r.mu is never held across it, so saves,
// proxy pushes and reads go on while a client is slow to read.

import "nvpair-shared/nodepolicy"

// notifyPolicyChanged announces this node's persisted policy, its revision and
// whether the proxy has activated it, to every client. Callers announce after
// the push that activates the revision, so the announcement carries the result;
// an activation change of an unchanged revision is announced the same way.
func (b *Broker) notifyPolicyChanged() {
	r := b.nodePolicyRuntime()
	r.publishMu.Lock()
	defer r.publishMu.Unlock()
	local := b.localPolicy()
	b.notifyClients(notifyPolicyChanged, struct {
		NodeID string            `json:"nodeId"`
		Policy nodepolicy.Policy `json:"policy"`
		policyActivation
	}{b.nodeID, local.Policy, local.policyActivation})
}

// notifyAvailability tells clients the live availability and active count.
func (b *Broker) notifyAvailability() {
	r := b.nodePolicyRuntime()
	r.publishMu.Lock()
	defer r.publishMu.Unlock()
	r.mu.Lock()
	live, active := r.live, r.admission.Active
	r.mu.Unlock()
	b.notifyClients(notifyAvailabilityChanged, map[string]any{"nodeId": b.nodeID, "availability": live, "active": active})
}
