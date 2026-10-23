// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

func (r *routingSpend) cancel(regenerate bool) bool {
	return r.cancelQualified(regenerate, nil)
}

func (r *routingSpend) cancelQualified(regenerate bool, workload *workloadKey) bool {
	r.mu.Lock()
	if r.done || r.cancelled || (workload != nil && (r.workload == nil || *r.workload != *workload)) {
		r.mu.Unlock()
		return false
	}
	if regenerate && !r.committed && !r.uncertain {
		if a := r.active; a != nil && !a.aborted && a.terminal == nil && a.nodeID != "" {
			a.aborted, a.redispatch = true, true
			a.state, a.reason = routingEnding, "regenerating on another concrete destination"
			r.transitionLocked(a.state, a.reason)
			if r.excluded == nil {
				r.excluded = map[string]bool{}
			}
			r.excluded[a.nodeID] = true
			cancel := a.cancel
			r.mu.Unlock()
			if cancel != nil {
				cancel()
			}
			return true
		} else if a == nil {
			if r.lastNode != "" {
				if r.excluded == nil {
					r.excluded = map[string]bool{}
				}
				r.excluded[r.lastNode] = true
			}
			r.transitionLocked(routingPrepared, "regenerating between concrete attempts")
			r.mu.Unlock()
			return true
		}
	}
	const reason = "cancelled by request"
	cancel, callback, changed := r.cancelLocked(reason)
	r.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if callback != nil {
		callback(reason)
	}
	return changed
}
