// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"

	"nvpair-shared/nodepolicy"
)

type routingCancelStatus = nodepolicy.RoutingCancelStatus

const routingCancelReason = "cancelled by request"

func (r *routingSpend) cancel(regenerate bool) bool {
	return r.cancelQualified(regenerate, nil)
}

func (r *routingSpend) cancelQualified(regenerate bool, workload *workloadKey) bool {
	r.mu.Lock()
	if r.done || r.cancelled || (workload != nil && (r.workload == nil || *r.workload != *workload)) {
		r.mu.Unlock()
		return false
	}
	_, cancel, callback, changed := r.cancelTransitionLocked(regenerate)
	r.mu.Unlock()
	runRoutingCancellation(cancel, callback)
	return changed
}

func (r *routingSpend) cancelRouting(regenerate bool) routingCancelStatus {
	r.mu.Lock()
	if r.done {
		r.mu.Unlock()
		return nodepolicy.RoutingCancelNotFound
	}
	if r.committed && regenerate {
		r.mu.Unlock()
		return nodepolicy.RoutingCancelConflictCommitted
	}
	if r.cancelled {
		r.mu.Unlock()
		return nodepolicy.RoutingCancelCancelled
	}
	status, cancel, callback, _ := r.cancelTransitionLocked(regenerate)
	r.mu.Unlock()
	runRoutingCancellation(cancel, callback)
	return status
}

// Both public surfaces enter this transition with mu held; effects run after unlock.
func (r *routingSpend) cancelTransitionLocked(regenerate bool) (routingCancelStatus, func(), func(string), bool) {
	if regenerate && !r.committed && !r.uncertain {
		if a := r.active; a != nil && !a.aborted && a.terminal == nil && a.nodeID != "" {
			a.aborted, a.redispatch = true, true
			a.state, a.reason = routingEnding, "regenerating on another concrete destination"
			r.transitionLocked(a.state, a.reason)
			if r.excluded == nil {
				r.excluded = map[string]bool{}
			}
			r.excluded[a.nodeID] = true
			return nodepolicy.RoutingCancelRegenerating, a.cancel, nil, true
		} else if a == nil {
			if r.lastNode != "" {
				if r.excluded == nil {
					r.excluded = map[string]bool{}
				}
				r.excluded[r.lastNode] = true
			}
			r.transitionLocked(routingPrepared, "regenerating between concrete attempts")
			return nodepolicy.RoutingCancelRegenerating, nil, nil, true
		}
	}
	cancel, callback, changed := r.cancelLocked(routingCancelReason)
	return nodepolicy.RoutingCancelCancelled, cancel, callback, changed
}

func runRoutingCancellation(cancel func(), callback func(string)) {
	if cancel != nil {
		cancel()
	}
	if callback != nil {
		callback(routingCancelReason)
	}
}

func (p *Proxy) handleRoutingControl(msg *Message) {
	if msg.Method == nodepolicy.MethodRoutingCancel {
		var in nodepolicy.RoutingCancelParams
		if err := json.Unmarshal(msg.Params, &in); err != nil {
			_ = p.codec.RespondError(msg.ID, -32602, "invalid routing cancellation params")
			return
		}
		status := nodepolicy.RoutingCancelNotFound
		if owner := p.routingRequestControl(in.Request); owner != nil {
			status = owner.cancelRouting(in.Regenerate)
		}
		_ = p.codec.Respond(msg.ID, nodepolicy.RoutingCancelResult{Status: status})
		return
	}
	var in nodepolicy.RoutingActiveParams
	if err := json.Unmarshal(msg.Params, &in); err != nil {
		_ = p.codec.RespondError(msg.ID, -32602, "invalid routing active params")
		return
	}
	if !p.routingSnapshot.CompareAndSwap(false, true) {
		_ = p.codec.RespondError(msg.ID, -32000, "routing snapshot busy; retry")
		return
	}
	go func() {
		defer p.routingSnapshot.Store(false)
		_ = p.codec.Respond(msg.ID, nodepolicy.RoutingActiveResult{Requests: p.activeRoutingRequests()})
	}()
}
