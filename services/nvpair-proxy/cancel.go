// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// Cancel and regenerate (FORK_DESIGN.md §4).
//
// workload/cancel reaches an in-flight request this process originated.
// Before the response commits, regenerate aborts only the current attempt,
// excludes its node for the rest of this request, and lets the retry loop
// carry on with the remaining candidates. Without regenerate, or once the
// response has committed, the request itself is cancelled and its workload
// ends as cancelled. A committed stream is never re-dispatched, so two models'
// output is never spliced together.
//
// The commit-versus-abort race is decided by the attempt's attemptClaim, the
// same single compare-and-swap the discovery watcher uses (spec.md §5.3).

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"nvpair-shared/nodepolicy"
)

// workloadKey identifies a workload among this process's facades: ids are
// counted per facade, so the engine is part of the key.
type workloadKey struct {
	engine string
	id     string
}

// attemptCtl is one dispatch attempt, as the cancel paths see it.
type attemptCtl struct {
	claim  *attemptClaim
	cancel context.CancelFunc
	nodeID string
	// redispatch marks an abort made on purpose so the request moves on: the
	// loop treats it as a retry that costs no dispatch, never as the answer.
	redispatch atomic.Bool
}

// abortForRedispatch abandons the attempt for re-dispatch, reporting false
// when the response has already committed.
func (a *attemptCtl) abortForRedispatch() bool {
	a.redispatch.Store(true)
	if !a.claim.abandon() {
		a.redispatch.Store(false)
		return false
	}
	a.cancel()
	return true
}

// redispatched reports whether this attempt ended in a deliberate abort.
func (a *attemptCtl) redispatched() bool {
	return a.redispatch.Load() && a.claim.abandoned()
}

// workloadCtl is a request's handle for cancellation.
type workloadCtl struct {
	mu       sync.Mutex
	attempt  *attemptCtl
	lastNode string
	excluded map[string]bool
	done     bool

	// cancelRequest ends the whole request with a reason the terminal
	// workload event carries.
	cancelRequest func(reason string)
}

func newWorkloadCtl(cancelRequest func(string)) *workloadCtl {
	return &workloadCtl{excluded: map[string]bool{}, cancelRequest: cancelRequest}
}

// setAttempt records the attempt now in flight, or nil between attempts.
func (c *workloadCtl) setAttempt(a *attemptCtl) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if a == nil && c.attempt != nil {
		c.lastNode = c.attempt.nodeID
	}
	c.attempt = a
}

// exclude keeps a node out of the rest of this request.
func (c *workloadCtl) exclude(nodeID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.excluded[nodeID] = true
}

func (c *workloadCtl) isExcluded(nodeID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.excluded[nodeID]
}

func (c *workloadCtl) hasExclusions() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.excluded) > 0
}

func (c *workloadCtl) finish() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.done = true
}

// cancel applies workload/cancel, reporting whether the request was still
// live to act on.
func (c *workloadCtl) cancel(regenerate bool) bool {
	c.mu.Lock()
	if c.done {
		c.mu.Unlock()
		return false
	}
	if regenerate {
		if a := c.attempt; a != nil {
			if a.abortForRedispatch() {
				c.excluded[a.nodeID] = true
				c.mu.Unlock()
				return true
			}
			// Already committed: regenerating now would splice two answers.
		} else {
			// Between attempts nothing is running; keep the next attempt off
			// the node that was just tried.
			if c.lastNode != "" {
				c.excluded[c.lastNode] = true
			}
			c.mu.Unlock()
			return true
		}
	}
	c.mu.Unlock()
	c.cancelRequest("cancelled by request")
	return true
}

// abortLocal handles node/set-availability cancelActive for a request this
// node is executing itself: before the commit the attempt moves on to another
// candidate, after it the request ends as cancelled.
func (c *workloadCtl) abortLocal() {
	c.mu.Lock()
	a := c.attempt
	if a != nil && a.abortForRedispatch() {
		c.excluded[a.nodeID] = true
		c.mu.Unlock()
		return
	}
	c.mu.Unlock()
	c.cancelRequest("cancelled: this node stopped accepting work")
}

func (p *Proxy) registerWorkload(key workloadKey, c *workloadCtl) {
	p.workloadsMu.Lock()
	defer p.workloadsMu.Unlock()
	if p.workloads == nil {
		p.workloads = map[workloadKey]*workloadCtl{}
	}
	p.workloads[key] = c
}

func (p *Proxy) unregisterWorkload(key workloadKey) {
	p.workloadsMu.Lock()
	defer p.workloadsMu.Unlock()
	delete(p.workloads, key)
}

// handleWorkloadCancel answers workload/cancel.
func (p *Proxy) handleWorkloadCancel(msg *Message) {
	var in nodepolicy.WorkloadCancelParams
	if err := decodeStrict(msg.Params, &in); err != nil {
		_ = p.codec.RespondError(msg.ID, -32602, err.Error())
		return
	}
	if in.WorkloadID == "" {
		_ = p.codec.RespondError(msg.ID, -32602, "workloadId is required")
		return
	}
	if in.Engine != "" {
		if err := knownEngine(in.Engine); err != nil {
			_ = p.codec.RespondError(msg.ID, -32602, err.Error())
			return
		}
	}
	if in.RunID != "" && in.RunID != p.runID {
		_ = p.codec.Respond(msg.ID, nodepolicy.WorkloadCancelResult{Found: false})
		return
	}
	p.workloadsMu.Lock()
	var matches []*workloadCtl
	for key, c := range p.workloads {
		if key.id == in.WorkloadID && (in.Engine == "" || key.engine == in.Engine) {
			matches = append(matches, c)
		}
	}
	p.workloadsMu.Unlock()
	if len(matches) > 1 {
		_ = p.codec.RespondError(msg.ID, -32602,
			fmt.Sprintf("workload %q is in flight on more than one engine; pass engine", in.WorkloadID))
		return
	}
	found := len(matches) == 1 && matches[0].cancel(in.Regenerate)
	_ = p.codec.Respond(msg.ID, nodepolicy.WorkloadCancelResult{Found: found})
}
