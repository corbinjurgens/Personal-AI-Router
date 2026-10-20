// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"math"
	"sync"

	"nvpair-shared/nodepolicy"
)

var errRoutingSpend = errors.New("routing request allowance refused")

// routingSpend belongs to one logical request, across all destinations and
// regenerations. Only D mints holds. Settlement never subtracts granted bounds.
// Lock order is G lease -> admission (if needed) -> request. Never enter G or
// perform IO while holding mu.
type routingSpend struct {
	mu        sync.Mutex
	ceiling   int64
	limited   bool
	granted   int64
	uncertain bool
	cancelled bool
	committed bool
	active    *publicAttemptLifecycle
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
type publicAttemptLifecycle struct {
	request     *routingSpend
	cancel      context.CancelFunc
	reservation accountReservation
	accepted    bool
	committed   bool
	terminal    *publicAttemptTerminal
}

func (r *routingSpend) newAttempt() (*publicAttemptLifecycle, error) {
	if r == nil {
		return nil, errRoutingSpend
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.cancelled || r.uncertain || r.committed || r.active != nil {
		return nil, errRoutingSpend
	}
	a := &publicAttemptLifecycle{request: r}
	r.active = a
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
	if r.active != a || a.terminal != nil {
		return 0, errRoutingSpend
	}
	if account.Billing == nodepolicy.BillingPaid {
		ceiling, err := account.PerRequestCeiling.Int64()
		if err != nil {
			return 0, err
		}
		if !r.limited || ceiling < r.ceiling {
			r.ceiling, r.limited = ceiling, true
		}
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
		return 0, errRoutingSpend
	}
	return r.ceiling - r.granted, nil
}

func (a *publicAttemptLifecycle) bindCancel(cancel context.CancelFunc) error {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != a || r.cancelled || a.cancel != nil || a.terminal != nil {
		return errRoutingSpend
	}
	a.cancel = cancel
	return nil
}

// associated must run even if cancellation won during D's durable write.
func (a *publicAttemptLifecycle) associated(reservation accountReservation) error {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if a.reservation.Durability != "" {
		return errRoutingSpend
	}
	a.reservation = reservation
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
	}
	return nil
}

// acceptedByGuard is called under G's permission lock immediately before Do.
// The lock arbitrates request cancellation; no callbacks, allocation or IO.
func (a *publicAttemptLifecycle) acceptedByGuard(ctx context.Context) error {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active != a || r.cancelled || r.uncertain || ctx.Err() != nil || a.terminal != nil || a.accepted ||
		a.reservation.Durability != accountReserveDurable {
		return errRoutingSpend
	}
	a.accepted = true
	return nil
}

func (a *publicAttemptLifecycle) firstSuccessfulByte() {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if a.accepted {
		a.committed = true
		r.committed = true
	}
}

func (a *publicAttemptLifecycle) finished(terminal publicAttemptTerminal) {
	r := a.request
	r.mu.Lock()
	defer r.mu.Unlock()
	if a.terminal == nil {
		a.terminal = &terminal
		if r.active == a {
			r.active = nil
		}
	}
}

func (r *routingSpend) cancelRequest() {
	r.mu.Lock()
	r.cancelled = true
	var cancel context.CancelFunc
	if r.active != nil {
		cancel = r.active.cancel
	}
	r.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// The continuation runs under the permission owner's lock. A nil continuation
// is the advisory post-reserve check; final acceptance supplies the request hook.
type publicAttemptGuard func(context.Context, func() error) error
