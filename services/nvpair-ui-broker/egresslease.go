// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"math"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/nodepolicy"
)

// brokerEgressLease is guarded by policyRuntime.mu. It neither stores policy
// nor activates it. The push slot orders lease messages with policy pushes.
type brokerEgressLease struct {
	process             *proxyProcess
	sequence            int64
	withdrawn           bool
	policyPending       bool
	availabilityPending bool
	now                 func() time.Time
	anchorAt            time.Time
	anchorNanos         int64
	anchored            bool
}

func (r *policyRuntime) egressEligibleLocked(pp *proxyProcess) bool {
	if pp == nil || pp != r.incarnation || pp != r.egress.process || !r.ready || r.recoveryRequired ||
		r.runID == "" || r.activeRevision != r.current.Revision || r.activationErr != "" || r.availabilityErr != "" ||
		r.egress.withdrawn || r.egress.policyPending || r.egress.availabilityPending {
		return false
	}
	select {
	case <-pp.Done():
		return false
	default:
		return true
	}
}

func (l *brokerEgressLease) nextSequence() (string, bool) {
	if l.sequence == math.MaxInt64 {
		return "", false
	}
	l.sequence++
	return strconv.FormatInt(l.sequence, 10), true
}

// startEgressRenewal owns one job for this Serve lifetime. No immediate grant:
// each tick re-evaluates activation, and a new run must first sample its clock.
func (b *Broker) startEgressRenewal(parent context.Context) func() {
	ctx, cancel := context.WithCancel(parent)
	ticker := time.NewTicker(nodepolicy.EgressLeaseRenewalInterval)
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer ticker.Stop()
		b.runEgressRenewal(ctx, ticker.C)
	}()
	var once sync.Once
	return func() { once.Do(func() { cancel(); <-done }) }
}

func (b *Broker) runEgressRenewal(ctx context.Context, ticks <-chan time.Time) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticks:
			b.renewEgressLease(ctx)
		}
	}
}

// renewEgressLease never holds r.mu across codec IO. A tick waiting for the
// push slot must still name the same eligible incarnation/revision at send.
func (b *Broker) renewEgressLease(parent context.Context) {
	r := b.nodePolicyRuntime()
	pp := b.policyProxy()
	r.mu.Lock()
	eligible := r.egressEligibleLocked(pp)
	runID, revision := r.runID, r.current.Revision
	r.mu.Unlock()
	if !eligible {
		return
	}
	ctx, cancel := context.WithTimeout(parent, time.Second)
	defer cancel()
	if !r.acquirePush(ctx) {
		return
	}
	defer r.releasePush()
	current := b.policyProxy()
	r.mu.Lock()
	if ctx.Err() != nil || pp != current || !r.egressEligibleLocked(pp) || r.runID != runID || r.current.Revision != revision {
		r.mu.Unlock()
		return
	}
	l := &r.egress
	notAfter := int64(0)
	anchor := !l.anchored
	if !anchor {
		elapsed := l.now().Sub(l.anchorAt)
		limit := int64(nodepolicy.MaxEgressLeaseDuration)
		if elapsed < 0 || l.anchorNanos > math.MaxInt64-limit || int64(elapsed) > math.MaxInt64-limit-l.anchorNanos {
			r.mu.Unlock()
			return
		}
		notAfter = l.anchorNanos + int64(elapsed) + limit
	}
	sequence, ok := l.nextSequence()
	r.mu.Unlock()
	if !ok {
		return
	}
	in := nodepolicy.RenewEgressLeaseParams{RunID: runID, Revision: revision, Sequence: sequence, NotAfterNanos: strconv.FormatInt(notAfter, 10)}
	data, err := json.Marshal(in)
	if err != nil {
		return
	}
	result, rpcErr, callErr := pp.Call(ctx, nodepolicy.MethodRenewEgressLease, data)
	// Transport failures can be a slow synchronous proxy handler. Do not latch
	// activation closed or throw away a valid run-lifetime anchor on timeout.
	if callErr != nil || rpcErr != nil {
		return
	}
	var ack nodepolicy.RenewEgressLeaseResult
	valid := json.Unmarshal(result, &ack) == nil && ack.RunID == runID && ack.Revision == revision && ack.Sequence == sequence
	current = b.policyProxy()
	r.mu.Lock()
	if pp != current || pp != r.incarnation || r.runID != runID || !r.egressEligibleLocked(pp) || r.current.Revision != revision || ctx.Err() != nil {
		r.mu.Unlock()
		return
	}
	if !valid {
		l.anchored = false
		// A reply with an untrusted binding requires fresh exact policy AND
		// availability acknowledgement. A lease reply cannot activate either.
		r.activationErr = "the proxy did not acknowledge the egress lease binding"
		r.availabilityErr = "the proxy egress lease binding requires availability acknowledgement"
		r.mu.Unlock()
		b.schedulePolicyAnnouncement()
		return
	}
	if anchor {
		// The receiver sampled before we received this reply. Anchoring at
		// receipt makes every later deadline conservative, even after delays.
		l.anchorNanos, _ = strconv.ParseInt(ack.NowNanos, 10, 64)
		l.anchorAt, l.anchored = l.now(), true
	}
	r.mu.Unlock()
}

// withdrawEgressLease first disables renewal, then orders a bounded revoke on
// the exact old pipe. Its caller stops/joins that process even if this fails.
func (b *Broker) withdrawEgressLease() {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	pp, runID := r.incarnation, r.runID
	r.egress.withdrawn, r.egress.anchored = true, false
	r.mu.Unlock()
	if pp == nil || runID == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if !r.acquirePush(ctx) {
		return
	}
	defer r.releasePush()
	r.mu.Lock()
	if pp != r.incarnation || runID != r.runID {
		r.mu.Unlock()
		return
	}
	sequence, ok := r.egress.nextSequence()
	r.mu.Unlock()
	if !ok {
		return
	}
	data, err := json.Marshal(nodepolicy.RevokeEgressLeaseParams{RunID: runID, Sequence: sequence})
	if err != nil {
		return
	}
	result, rpcErr, callErr := pp.Call(ctx, nodepolicy.MethodRevokeEgressLease, data)
	var ack nodepolicy.RevokeEgressLeaseResult
	if callErr != nil || rpcErr != nil || json.Unmarshal(result, &ack) != nil || ack.RunID != runID || ack.Sequence != sequence {
		return // withdrawal stays latched; teardown is mandatory either way
	}
}
