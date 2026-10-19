// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// readiness.go is the broker side of strict local admission readiness. The
// proxy's admission controller reports what it can admit under the strict
// contract (nodepolicy.NotifyStrictReadiness); the broker, which owns the policy and knows
// which proxy run acknowledged which revision, decides whether that report is
// authoritative right now and publishes its capability in the node projection.
// engine-manager only publishes what it is sent and never owns admission.
//
// A report is authoritative only while every fence holds: the proxy that sent it
// is the running incarnation and the run that acknowledged the policy, the
// revision it names is both the active and the persisted one, the availability
// push was acknowledged, and the node is available. Anything else withdraws the
// capability, and a report never outlives the proxy incarnation it came from.

import (
	"encoding/json"
	"log/slog"

	"nvpair-shared/nodepolicy"
)

// observeStrictReadiness takes one nodepolicy.NotifyStrictReadiness report. It runs on the
// proxy's read pump, so the projection goes out on its own goroutine.
func (b *Broker) observeStrictReadiness(params json.RawMessage) {
	var report nodepolicy.StrictAdmissionReadiness
	err := json.Unmarshal(params, &report)
	if err == nil {
		err = report.Validate()
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	switch {
	case err != nil:
		// The proxy's readiness is unknown now, so nothing is advertised.
		slog.Warn("proxy sent an invalid strict admission readiness; withdrawing it", "err", err)
		r.strictReadiness = nodepolicy.StrictAdmissionReadiness{}
	case r.runID != "" && report.ProxyRunID != r.runID:
		r.mu.Unlock()
		slog.Debug("discarded strict admission readiness from another proxy run", "revision", report.PolicyRevision)
		return
	default:
		r.strictReadiness = report
	}
	r.mu.Unlock()
	go b.refreshStrictAdvertisement()
}

// StrictReadiness is the readiness the node advertises now: the held report
// with its capability only while the report is authoritative.
func (b *Broker) StrictReadiness() nodepolicy.StrictAdmissionReadiness {
	r := b.nodePolicyRuntime()
	running := b.policyProxy()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.strictReadinessLocked(running)
}

// strictReadinessLocked applies the fences to the held report. running is the
// proxy process now published. Models of an engine saved Off are left out, as
// the local veto keeps them from being started for anyone; a capability with no
// model left is withdrawn. Caller holds r.mu.
func (r *policyRuntime) strictReadinessLocked(running *proxyProcess) nodepolicy.StrictAdmissionReadiness {
	held := r.strictReadiness
	out := nodepolicy.StrictAdmissionReadiness{PolicyRevision: held.PolicyRevision, ProxyRunID: held.ProxyRunID}
	authoritative := held.Capability != nil &&
		// The proxy has not exited, been stopped or been replaced.
		running != nil && running == r.incarnation &&
		// The report comes from the run that acknowledged the policy.
		r.runID != "" && held.ProxyRunID == r.runID &&
		// It names the active revision, and no newer one is pending activation.
		held.PolicyRevision == r.activeRevision && r.activeRevision == r.current.Revision &&
		// The availability that revision rests on was acknowledged.
		r.availabilityErr == "" &&
		// The node is neither paused nor draining.
		r.live == nodepolicy.Available && r.current.Availability == nodepolicy.Available
	if !authoritative {
		return out
	}
	models := make([]nodepolicy.StrictAdmissionModel, 0, len(held.Capability.Models))
	for _, m := range held.Capability.Models {
		if enabled, known := r.intent[m.Engine]; known && !enabled {
			continue
		}
		models = append(models, m)
	}
	if len(models) > 0 {
		out.Capability = &nodepolicy.StrictAdmissionCapability{Version: held.Capability.Version, Models: models}
	}
	return out
}

// withdrawStrictReadinessOnExit republishes the projection once pp exits, so a
// capability its reports backed is withdrawn without waiting for a restart. A
// handle without a process has no exit to wait for.
func (b *Broker) withdrawStrictReadinessOnExit(pp *proxyProcess) {
	if pp == nil || pp.Done() == nil {
		return
	}
	done := pp.Done()
	go func() {
		<-done
		b.refreshStrictAdvertisement()
	}()
}
