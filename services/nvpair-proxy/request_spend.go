// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"math"
	"sync"
	"time"

	"nvpair-shared/nodepolicy"
)

var errRoutingSpend = errors.New("routing request allowance refused")

// routingSpend belongs to one logical request, across all destinations and
// regenerations. Only D mints holds. Settlement never subtracts granted bounds.
// Lock order is G lease -> admission (if needed) -> request. Never enter G or
// perform IO while holding mu.
type routingSpend struct {
	mu             sync.Mutex
	ceiling        int64
	granted        int64
	active         *publicAttemptLifecycle
	key            nodepolicy.RoutingRequestKey
	deadline       time.Time
	body           []byte // Borrows the request's owned body lease until dispatch returns.
	sink           *responseSink
	firstBodyDone  <-chan struct{} // Actual peek and timeout Close join, including post-read callbacks.
	ordinal        uint64
	reason         string
	lastNode       string
	excluded       map[string]bool
	onCancel       func(string)
	workload       *workloadKey
	detachWorkload func()              // Immutable registry cleanup hook; never called under mu.
	stopPreOutput  func() bool         // Set once under mu; timer collaborator, not transition state.
	facts          *routingNativeFacts // Detached at actual unregister, after all body borrowers join.
	// Immutable entry contexts duplicate existing onCancel/registry-owner
	// reachability, not body or fact ownership. Cleared at actual unregister.
	observationContext                             context.Context
	processContext                                 context.Context
	state                                          routingState
	limited, uncertain, cancelled, committed, done bool
	cancelCause                                    routingFactReason
}

type accountReserveDurability string

const (
	accountReserveNone      accountReserveDurability = "no-hold"
	accountReserveDurable   accountReserveDurability = "durable"
	accountReserveUncertain accountReserveDurability = "uncertain"
)

// An uncertain hold identifies a landed liability, not permission to resolve it.
type accountReservation struct {
	Hold       accountHold
	Durability accountReserveDurability
}

type publicAccountingDisposition string

const (
	publicAccountingRetained publicAccountingDisposition = "retained"
	publicAccountingRecorded publicAccountingDisposition = "recorded"
	publicAccountingFailed   publicAccountingDisposition = "failed"
)

type publicAttemptTerminal struct {
	ProtocolComplete, Exposed, Delivered bool
	Accounting                           publicAccountingDisposition
	Err                                  error
}

// These synchronous typed hooks are the request's lifecycle observation seam.
// Every hook is bound to this immutable attempt, including a reserve completion
// arriving after cancellation. Facts are retained after the attempt finishes.
// Callers must not hold the request mutex for any hook; each acquires it itself.
// tryAccount can run under G/admission; acceptedByGuard requires those locks in
// production. bindCancel, associated, firstSuccessfulByte and finished require
// no caller-held lock and never enter G, admission, D or transport themselves.
type publicAttemptLifecycle struct {
	routingFactAttempt
	cancel             context.CancelFunc
	ctx                context.Context
	nodeID             string
	ticket             *admissionTicket // Released after transport/body borrowers unwind.
	reservation        accountReservation
	terminal           *publicAttemptTerminal
	accountObservation accountAttemptObservation
	// The D baseline reuses the original state/reason headers and boolean
	// padding: this object remains 176B, including arbitrary old callback tails.
	// Reason text has one owner, request.reason, at the same transition sites.
	state                                    routingState
	aborted, redispatch, accepted, committed bool
}

func (r *routingSpend) newAttempt(descriptor ...*routingDestinationDescriptor) (*publicAttemptLifecycle, error) {
	if r == nil {
		return nil, errRoutingSpend
	}
	r.mu.Lock()
	if r.done || r.cancelled || r.uncertain || r.committed || r.active != nil {
		r.mu.Unlock()
		return nil, errRoutingSpend
	}
	if !r.deadline.IsZero() && !time.Now().Before(r.deadline) {
		r.mu.Unlock()
		return nil, errRoutingSpend
	}
	detach := r.detachWorkload
	r.mu.Unlock()
	// A public attempt cannot inherit an ended concrete workload target. The
	// registry callback takes identityMu -> request, never the reverse order.
	if detach != nil {
		detach()
	}
	started := time.Now()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.newAttemptLocked(started, descriptor...)
}

// newAttemptLocked requires the request mutex and publishes one attempt slot.
func (r *routingSpend) newAttemptLocked(started time.Time, descriptor ...*routingDestinationDescriptor) (*publicAttemptLifecycle, error) {
	if r.done || r.cancelled || r.uncertain || r.committed || r.active != nil {
		return nil, errRoutingSpend
	}
	if !r.deadline.IsZero() && !time.Now().Before(r.deadline) {
		return nil, errRoutingSpend
	}
	r.ordinal++
	a := &publicAttemptLifecycle{routingFactAttempt: routingFactAttempt{request: r, ordinal: r.ordinal}, state: routingPrepared}
	r.active = a
	r.transitionLocked(routingPrepared, "attempt prepared")
	if len(descriptor) != 0 {
		r.nativeAttemptLocked(&a.routingFactAttempt, descriptor[0], started)
	}
	return a, nil
}

// tryAccount tightens even a candidate which D subsequently refuses. Free and
// trial candidates do not establish a zero ceiling. Local/peer transitions need
// no operation on this object and therefore cannot reset it.
func (a *publicAttemptLifecycle) tryAccount(account nodepolicy.Account) (int64, error) {
	if a == nil || a.request == nil {
		return 0, errRoutingSpend
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != a || a.terminal != nil || a.aborted || r.done {
		return 0, errRoutingSpend
	}
	if account.Billing == nodepolicy.BillingPaid {
		ceiling, err := account.PerRequestCeiling.Int64()
		if err != nil {
			return 0, err
		}
		if !r.limited || ceiling < r.ceiling {
			r.ceiling, r.limited = ceiling, true
			if f := r.facts; f.begin() {
				f.ceiling.Store(uint64(ceiling))
				f.granted.Store(uint64(r.granted))
				f.flags.Store(f.flags.Load() | factMoneyApplicable | factCeilingKnown)
				f.end()
			}
		}
	}
	if f := r.facts; f != nil && f.flags.Load()&factMoneyApplicable == 0 && f.begin() {
		f.granted.Store(uint64(r.granted))
		f.flags.Store(f.flags.Load() | factMoneyApplicable)
		f.end()
	}
	if r.cancelled || r.uncertain {
		return 0, errRoutingSpend
	}
	if !r.limited {
		return math.MaxInt64 - r.granted, nil
	}
	if r.granted > r.ceiling {
		if account.Billing != nodepolicy.BillingPaid {
			return 0, nil // A zero-cost free/trial hold can still fit.
		}
		a.observeAttemptLocked(routingAttemptResult{factRejected, factRequestCeiling, 0, true})
		return 0, errRoutingSpend
	}
	return r.ceiling - r.granted, nil
}

func (a *publicAttemptLifecycle) bindCancel(cancel context.CancelFunc) error {
	if a == nil || a.request == nil {
		return errRoutingSpend
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != a || r.done || r.cancelled || a.aborted || a.cancel != nil || a.terminal != nil {
		return errRoutingSpend
	}
	a.cancel = cancel
	return nil
}

// associated must run even if cancellation won during D's durable write.
func (a *publicAttemptLifecycle) associated(reservation accountReservation) error {
	if a == nil || a.request == nil {
		return errRoutingSpend
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != a || a.terminal != nil || a.reservation.Durability != "" {
		return errRoutingSpend
	}
	a.reservation = reservation
	if !r.cancelled && !r.done && !a.aborted {
		a.state = routingPreSend
		r.transitionLocked(a.state, "account reservation completed")
	}
	if reservation.Durability == accountReserveUncertain {
		r.uncertain = true
	}
	if reservation.Durability != accountReserveNone {
		n, err := reservation.Hold.Reserved.Int64()
		if err != nil {
			r.uncertain = true
			return errRoutingSpend
		}
		total, err := addNanoUSD(r.granted, n)
		if err != nil {
			r.uncertain = true
			return err
		}
		r.granted = total
		if f := r.facts; f.begin() {
			f.granted.Store(uint64(total))
			f.flags.Store(f.flags.Load() | factMoneyApplicable)
			f.end()
		}
	}
	return nil
}

// acceptedByGuard is called under G's permission lock immediately before Do.
// The lock arbitrates request cancellation; no callbacks, allocation or IO.
func (a *publicAttemptLifecycle) acceptedByGuard(ctx context.Context) error {
	if a == nil || a.request == nil || ctx == nil {
		return errRoutingSpend
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != a || r.done || r.cancelled || r.uncertain || a.aborted || ctx.Err() != nil || a.terminal != nil || a.accepted ||
		a.reservation.Durability != accountReserveDurable {
		return errRoutingSpend
	}
	if !r.deadline.IsZero() && !time.Now().Before(r.deadline) {
		// Timer delivery may be late. Record the refusal while G/admission
		// still hold their locks; the caller's unwind cancels transport only
		// after those locks and this mutex have been released.
		r.cancelLockedCause(routingDeadlineReason, factDeadline)
		return errRoutingSpend
	}
	a.accepted = true
	a.state = routingAwaitBody
	r.transitionLocked(routingAwaitBody, "public final acceptance")
	return nil
}

func (a *publicAttemptLifecycle) firstSuccessfulByte() {
	a.commit()
}

func (a *publicAttemptLifecycle) finished(terminal publicAttemptTerminal) {
	if a == nil || a.request == nil {
		return
	}
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if a.terminal == nil {
		r.foldCancellationLocked()
		result := routingAttemptResult{outcome: factAttemptFailed, ended: true}
		if a.committed {
			result.outcome = factInterrupted
		}
		if terminal.Err != nil {
			result.reason = failureFactReason(terminal.Err)
		}
		a.observeAttemptLocked(result)
		a.terminal = &terminal
		a.state = routingTerminal
		if r.active == a {
			r.lastNode = a.nodeID
			r.active = nil
			if !r.cancelled && !r.committed && !r.done {
				r.transitionLocked(routingPrepared, "attempt finished before commitment")
			}
		}
	}
}

func (r *routingSpend) cancelRequest() {
	r.cancelWithCause("cancelled by request", factCancel)
}

// The continuation runs under the permission owner's lock. A nil continuation
// is the advisory post-reserve check; final acceptance supplies the request hook.
type publicAttemptGuard func(context.Context, func() error) error
