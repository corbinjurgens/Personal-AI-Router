// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"container/list"
	"context"
	"errors"
	"math"
	"net/http"
	"sync"
	"time"
)

const bufferBudgetMaxWaiters = 1024

var (
	errBufferLimit   = errors.New("owned buffer individual or category limit exceeded")
	errBufferBusy    = errors.New("owned buffer aggregate busy")
	errBufferInvalid = errors.New("invalid owned buffer size or budget")
	errBufferClosed  = errors.New("owned buffer lease closed")
)

type bufferLimitError struct{}

func (*bufferLimitError) Error() string   { return errBufferLimit.Error() }
func (*bufferLimitError) Unwrap() error   { return errBufferLimit }
func (*bufferLimitError) HTTPStatus() int { return http.StatusRequestEntityTooLarge }

type bufferBusyError struct{ cause error }

func (*bufferBusyError) Error() string       { return errBufferBusy.Error() }
func (*bufferBusyError) Is(other error) bool { return other == errBufferBusy }
func (e *bufferBusyError) Unwrap() error     { return e.cause }
func (*bufferBusyError) HTTPStatus() int     { return http.StatusServiceUnavailable }

// bufferBudget accounts owned capacity, not useful length or process RSS. A
// caller acquires before allocating and keeps the lease while storage is pooled
// or borrowed. This primitive has no production activation or policy defaults.
type bufferBudget struct {
	mu       sync.Mutex
	capacity int64
	used     int64
	live     int
	wait     time.Duration
	waiters  list.List
	// Clock functions are fixed before use; tests can supply controlled timers.
	now      func() time.Time
	newTimer func(time.Duration) *time.Timer
}

// bufferScope is a request account or a subordinate category. Pending proposals
// reserve scope room too, so concurrent waiters cannot evade a request limit.
// Every descendant checks all ancestors and charges the same process budget.
type bufferScope struct {
	budget *bufferBudget
	parent *bufferScope
	limit  int64
	used   int64
}

type bufferCharge struct {
	budget *bufferBudget
	scope  *bufferScope
	bytes  int64
	refs   int64
}

// Each holder has its own idempotent close. Do not copy a lease value; borrow or
// transfer it instead. Cancellation never closes a holder on the caller's behalf.
type bufferLease struct {
	charge *bufferCharge
	closed bool
}

type bufferWaiter struct {
	element  *list.Element
	ready    chan struct{}
	ctx      context.Context
	deadline time.Time
	scope    *bufferScope
	bytes    int64
	lease    *bufferLease
	err      error
}

func newBufferBudget(capacity int64, wait time.Duration) (*bufferBudget, error) {
	if capacity < 0 || wait < 0 {
		return nil, errBufferInvalid
	}
	return &bufferBudget{capacity: capacity, wait: wait, now: time.Now, newTimer: time.NewTimer}, nil
}

func (b *bufferBudget) newRequest(limit int64) (*bufferScope, error) {
	if limit < 0 {
		return nil, errBufferInvalid
	}
	return &bufferScope{budget: b, limit: limit}, nil
}

func (s *bufferScope) category(limit int64) (*bufferScope, error) {
	if limit < 0 {
		return nil, errBufferInvalid
	}
	return &bufferScope{budget: s.budget, parent: s, limit: limit}, nil
}

func (b *bufferBudget) acquire(ctx context.Context, capacity int64) (*bufferLease, error) {
	b.mu.Lock()
	return b.acquireLocked(ctx, capacity, nil, true)
}

func (s *bufferScope) acquire(ctx context.Context, capacity int64) (*bufferLease, error) {
	s.budget.mu.Lock()
	return s.budget.acquireLocked(ctx, capacity, s, true)
}

func (s *bufferScope) tryAcquire(ctx context.Context, capacity int64) (*bufferLease, error) {
	s.budget.mu.Lock()
	return s.budget.acquireLocked(ctx, capacity, s, false)
}

// grow acquires the entire replacement capacity, not a delta. The old lease
// remains charged until the caller has copied and closed it; borrowers can keep
// the old storage alive beyond that close. Failure leaves the old lease intact.
func (l *bufferLease) grow(ctx context.Context, capacity int64) (*bufferLease, error) {
	b := l.charge.budget
	b.mu.Lock()
	if l.closed {
		b.mu.Unlock()
		return nil, errBufferClosed
	}
	return b.acquireLocked(ctx, capacity, l.charge.scope, true)
}

// acquireLocked consumes the lock and snapshots the wait for this call. Scope
// limits and overflow are checked before any aggregate refusal or queue entry.
func (b *bufferBudget) acquireLocked(ctx context.Context, n int64, scope *bufferScope, mayWait bool) (*bufferLease, error) {
	started, wait := b.now(), b.wait
	if n < 0 {
		b.mu.Unlock()
		return nil, errBufferInvalid
	}
	for s := scope; s != nil; s = s.parent {
		if n > math.MaxInt64-s.used || n > s.limit-s.used {
			b.mu.Unlock()
			return nil, &bufferLimitError{}
		}
	}
	if err := ctx.Err(); err != nil {
		b.mu.Unlock()
		return nil, &bufferBusyError{cause: err}
	}
	// An account cannot release its own live storage while waiting for this
	// replacement. Classify that impossible footprint before joining the FIFO.
	root := scope
	for root != nil && root.parent != nil {
		root = root.parent
	}
	own := int64(0)
	if root != nil {
		own = root.used
	}
	if n > b.capacity || own > b.capacity-n {
		b.mu.Unlock()
		return nil, &bufferLimitError{}
	}
	if b.waiters.Len() == 0 && b.fitsLocked(n) {
		reserveBufferScope(scope, n)
		lease := b.grantLocked(scope, n)
		b.mu.Unlock()
		return lease, nil
	}
	if !mayWait || wait == 0 || b.waiters.Len() >= bufferBudgetMaxWaiters {
		b.mu.Unlock()
		return nil, &bufferBusyError{}
	}
	deadline := started.Add(wait)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	w := &bufferWaiter{ctx: ctx, deadline: deadline, ready: make(chan struct{}), scope: scope, bytes: n}
	reserveBufferScope(scope, n)
	w.element = b.waiters.PushBack(w)
	timer := b.newTimer(deadline.Sub(b.now()))
	b.mu.Unlock()
	defer timer.Stop()

	select {
	case <-w.ready:
	case <-ctx.Done():
	case <-timer.C:
	}

	b.mu.Lock()
	cause := ctx.Err()
	if cause == nil && !b.now().Before(deadline) {
		cause = context.DeadlineExceeded
	}
	if w.element != nil {
		b.removeLocked(w, &bufferBusyError{cause: cause})
		b.reconsiderLocked()
	}
	if w.lease != nil && cause != nil {
		// A grant can win the lock while cancellation wins the select. Return
		// the charge before reporting refusal, including scope reservations.
		w.lease.closeLocked()
		b.reconsiderLocked()
		w.lease = nil
		w.err = &bufferBusyError{cause: cause}
	}
	lease, err := w.lease, w.err
	b.mu.Unlock()
	return lease, err
}

func reserveBufferScope(scope *bufferScope, n int64) {
	for s := scope; s != nil; s = s.parent {
		s.used += n
	}
}

func (b *bufferBudget) fitsLocked(n int64) bool {
	return b.used <= b.capacity && n <= b.capacity-b.used
}

func (b *bufferBudget) grantLocked(scope *bufferScope, n int64) *bufferLease {
	b.used += n
	b.live++
	return &bufferLease{charge: &bufferCharge{budget: b, scope: scope, bytes: n, refs: 1}}
}

func (b *bufferBudget) removeLocked(w *bufferWaiter, err error) {
	b.waiters.Remove(w.element)
	w.element = nil
	w.err = err
	reserveBufferScope(w.scope, -w.bytes)
	close(w.ready)
}

// Only a live head can receive room. Expired/cancelled heads are removed before
// granting successive heads, stopping at the first proposal that does not fit.
func (b *bufferBudget) reconsiderLocked() {
	for e := b.waiters.Front(); e != nil; e = b.waiters.Front() {
		w := e.Value.(*bufferWaiter)
		cause := w.ctx.Err()
		if cause == nil && !b.now().Before(w.deadline) {
			cause = context.DeadlineExceeded
		}
		if cause != nil {
			b.removeLocked(w, &bufferBusyError{cause: cause})
			continue
		}
		if !b.fitsLocked(w.bytes) {
			break
		}
		b.waiters.Remove(e)
		w.element = nil
		w.lease = b.grantLocked(w.scope, w.bytes)
		close(w.ready)
	}
}

// reconfigure never revokes existing storage. A decrease may leave usage above
// capacity while owners drain, but no grant can increase that excess.
func (b *bufferBudget) reconfigure(capacity int64, wait time.Duration) error {
	if capacity < 0 || wait < 0 {
		return errBufferInvalid
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.capacity, b.wait = capacity, wait
	for e := b.waiters.Front(); e != nil; {
		next := e.Next()
		w := e.Value.(*bufferWaiter)
		root := w.scope
		for root != nil && root.parent != nil {
			root = root.parent
		}
		if w.bytes > capacity || (root != nil && root.used > capacity) {
			b.removeLocked(w, &bufferLimitError{})
		}
		e = next
	}
	b.reconsiderLocked()
	return nil
}

func (l *bufferLease) borrow() (*bufferLease, error) {
	b := l.charge.budget
	b.mu.Lock()
	defer b.mu.Unlock()
	if l.closed {
		return nil, errBufferClosed
	}
	if l.charge.refs == math.MaxInt64 {
		return nil, errBufferInvalid
	}
	l.charge.refs++
	return &bufferLease{charge: l.charge}, nil
}

// transfer replaces a holder without releasing or duplicating its charge.
func (l *bufferLease) transfer() (*bufferLease, error) {
	b := l.charge.budget
	b.mu.Lock()
	defer b.mu.Unlock()
	if l.closed {
		return nil, errBufferClosed
	}
	l.closed = true
	return &bufferLease{charge: l.charge}, nil
}

func (l *bufferLease) close() {
	b := l.charge.budget
	b.mu.Lock()
	l.closeLocked()
	b.reconsiderLocked()
	b.mu.Unlock()
}

func (l *bufferLease) closeLocked() {
	if l.closed {
		return
	}
	l.closed = true
	c := l.charge
	c.refs--
	if c.refs == 0 {
		c.budget.used -= c.bytes
		c.budget.live--
		reserveBufferScope(c.scope, -c.bytes)
	}
}
