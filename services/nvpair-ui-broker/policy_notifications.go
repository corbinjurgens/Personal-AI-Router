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
// proxy pushes and reads go on while a client is slow to read. The projection
// this node publishes to engine-manager for its paired peers follows the same
// rule under its own lock.

import (
	"log/slog"
	"reflect"

	"nvpair-shared/nodepolicy"
)

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

// nodeProjection is the engine:node-projection payload: what engine-manager
// publishes about this node's policy in its pinned /v1/models document, and
// whether the node is paused locally, which engine-manager enforces against
// paired nodes' starts and loads.
type nodeProjection struct {
	Policy nodepolicy.Advertisement `json:"policy"`
	// PausedLocally is pausedLocally of the policy. It counts a pause with no
	// provenance as local, which the advertised pausedBy alone cannot say.
	PausedLocally bool `json:"pausedLocally"`
}

// publishNodeProjection sends engine-manager this node's policy advertisement,
// read from the policy as it is now. Callers publish after every persisted
// policy or availability change and whenever engine-manager reports ready, so a
// restarted engine-manager learns it again. It follows the discipline above
// under its own lock, so a projection that read older state can never reach
// engine-manager after one that read newer state.
func (b *Broker) publishNodeProjection() {
	b.sendNodeProjection(false)
}

// refreshStrictAdvertisement publishes the projection only when the strict
// admission capability it would carry differs from the one last published.
// Callers refresh after every change to what the capability rests on that
// does not publish already: a readiness report, an acknowledgement, a proxy
// start or exit, and saved engine intent.
func (b *Broker) refreshStrictAdvertisement() {
	b.sendNodeProjection(true)
}

// sendNodeProjection builds the projection from the stored policy and the
// strict readiness authoritative now (readiness.go). The proxy handle is read
// under the projection lock too, so a projection that saw a running proxy can
// never follow one that saw it gone.
func (b *Broker) sendNodeProjection(onlyOnStrictChange bool) {
	r := b.nodePolicyRuntime()
	r.projectionMu.Lock()
	defer r.projectionMu.Unlock()
	// Recovery needs authoritative policy, including a pending local pause.
	b.workersMu.Lock()
	worker := b.engineMgr
	b.workersMu.Unlock()
	if worker == nil {
		return
	}
	running := b.policyProxy()
	r.mu.Lock()
	advertisement := r.current.Advertisement()
	advertisement.StrictAdmission = r.strictReadinessLocked(running, b.engineModelNaming).Capability
	if onlyOnStrictChange && reflect.DeepEqual(advertisement.StrictAdmission, r.strictPublished) {
		r.mu.Unlock()
		return
	}
	r.strictPublished = advertisement.StrictAdmission
	projection := nodeProjection{Policy: advertisement, PausedLocally: pausedLocally(r.current) || r.localPauseRequested}
	r.mu.Unlock()
	if err := worker.Notify(engineNodeProjectionMethod, projection); err != nil {
		slog.Warn("could not publish the node projection to engine-manager", "err", err)
	}
}
