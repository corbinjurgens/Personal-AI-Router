// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"math"
	"net/http"
	"runtime"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

func bufferTestBudget(t *testing.T, capacity int64, wait time.Duration) *bufferBudget {
	t.Helper()
	b, err := newBufferBudget(capacity, wait)
	if err != nil {
		t.Fatalf("new budget: %v", err)
	}
	t.Cleanup(func() { bufferAssertTotals(t, b, 0, 0, 0) })
	return b
}

func bufferAssertTotals(t *testing.T, b *bufferBudget, bytes int64, live, queued int) {
	t.Helper()
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.used != bytes || b.live != live || b.waiters.Len() != queued {
		t.Fatalf("budget totals = (%d bytes, %d leases, %d queued), want (%d, %d, %d)", b.used, b.live, b.waiters.Len(), bytes, live, queued)
	}
}

func bufferMustAcquire(t *testing.T, b *bufferBudget, n int64) *bufferLease {
	t.Helper()
	l, err := b.acquire(context.Background(), n)
	if err != nil {
		t.Fatalf("acquire %d: %v", n, err)
	}
	t.Cleanup(l.close)
	return l
}

func bufferMustScope(t *testing.T, b *bufferBudget, limit int64) *bufferScope {
	t.Helper()
	s, err := b.newRequest(limit)
	if err != nil {
		t.Fatalf("request scope: %v", err)
	}
	return s
}

func bufferAssertBusy(t *testing.T, err error) {
	t.Helper()
	var busy *bufferBusyError
	if !errors.Is(err, errBufferBusy) || !errors.As(err, &busy) || busy.HTTPStatus() != http.StatusServiceUnavailable {
		t.Fatalf("error = %v, want typed aggregate HTTP 503", err)
	}
}

func bufferAssertLimit(t *testing.T, err error) {
	t.Helper()
	var limit *bufferLimitError
	if !errors.Is(err, errBufferLimit) || !errors.As(err, &limit) || limit.HTTPStatus() != http.StatusRequestEntityTooLarge {
		t.Fatalf("error = %v, want typed individual/category HTTP 413", err)
	}
}

type bufferTestCall struct {
	done  chan struct{}
	lease *bufferLease
	err   error
}

// Cleanup always cancels and joins the caller, including on a failed assertion.
func bufferStartCall(t *testing.T, parent context.Context, acquire func(context.Context) (*bufferLease, error)) *bufferTestCall {
	t.Helper()
	ctx, cancel := context.WithCancel(parent)
	c := &bufferTestCall{done: make(chan struct{})}
	go func() {
		c.lease, c.err = acquire(ctx)
		close(c.done)
	}()
	t.Cleanup(func() {
		cancel()
		<-c.done
		if c.lease != nil {
			c.lease.close()
		}
	})
	return c
}

func bufferStartAcquire(t *testing.T, b *bufferBudget, ctx context.Context, n int64) *bufferTestCall {
	t.Helper()
	return bufferStartCall(t, ctx, func(ctx context.Context) (*bufferLease, error) { return b.acquire(ctx, n) })
}

func (c *bufferTestCall) result(t *testing.T) (*bufferLease, error) {
	t.Helper()
	select {
	case <-c.done:
		return c.lease, c.err
	default:
		t.Fatal("acquisition has not returned")
		return nil, nil
	}
}

func (c *bufferTestCall) pending(t *testing.T) {
	t.Helper()
	select {
	case <-c.done:
		t.Fatalf("acquisition returned early: lease=%v, err=%v", c.lease != nil, c.err)
	default:
	}
}

func (c *bufferTestCall) success(t *testing.T) *bufferLease {
	t.Helper()
	l, err := c.result(t)
	if err != nil || l == nil {
		t.Fatalf("acquisition = (%v, %v), want lease", l, err)
	}
	return l
}

func bufferAdvance(d time.Duration) {
	// Advance only the synctest clock, then join all newly runnable work.
	timer := time.NewTimer(d)
	<-timer.C
	synctest.Wait()
}

func TestBufferBudget(t *testing.T) {
	t.Run("exact fit", func(t *testing.T) {
		b := bufferTestBudget(t, 8, 0)
		l := bufferMustAcquire(t, b, 8)
		bufferAssertTotals(t, b, 8, 1, 0)
		if _, err := b.acquire(context.Background(), 1); err == nil {
			t.Fatal("over-capacity acquire succeeded")
		} else {
			bufferAssertBusy(t, err)
		}
		l.close()
		bufferMustAcquire(t, b, 8)
	})
	t.Run("zero", func(t *testing.T) {
		b := bufferTestBudget(t, 0, 0)
		l := bufferMustAcquire(t, b, 0)
		bufferAssertTotals(t, b, 0, 1, 0)
		l.close()
		bufferAssertTotals(t, b, 0, 0, 0)
		_, err := b.acquire(context.Background(), 1)
		bufferAssertLimit(t, err)
	})
	t.Run("invalid and overflow", func(t *testing.T) {
		if _, err := newBufferBudget(-1, 0); !errors.Is(err, errBufferInvalid) {
			t.Fatalf("negative capacity: %v", err)
		}
		if _, err := newBufferBudget(1, -time.Second); !errors.Is(err, errBufferInvalid) {
			t.Fatalf("negative wait: %v", err)
		}
		b := bufferTestBudget(t, math.MaxInt64, 0)
		if _, err := b.acquire(context.Background(), -1); !errors.Is(err, errBufferInvalid) {
			t.Fatalf("negative allocation: %v", err)
		}
		if _, err := b.newRequest(-1); !errors.Is(err, errBufferInvalid) {
			t.Fatalf("negative request limit: %v", err)
		}
		s := bufferMustScope(t, b, math.MaxInt64)
		if _, err := s.category(-1); !errors.Is(err, errBufferInvalid) {
			t.Fatalf("negative category limit: %v", err)
		}
		l, err := s.acquire(context.Background(), math.MaxInt64)
		if err != nil {
			t.Fatalf("maximum signed capacity: %v", err)
		}
		t.Cleanup(l.close)
		_, err = s.acquire(context.Background(), 1)
		bufferAssertLimit(t, err)
		_, err = b.acquire(context.Background(), 1)
		bufferAssertBusy(t, err)
		if err := b.reconfigure(-1, 0); !errors.Is(err, errBufferInvalid) {
			t.Fatalf("negative reconfigured capacity: %v", err)
		}
		if err := b.reconfigure(1, -time.Second); !errors.Is(err, errBufferInvalid) {
			t.Fatalf("negative reconfigured wait: %v", err)
		}
		bufferAssertTotals(t, b, math.MaxInt64, 1, 0)
	})
	t.Run("categories cannot evade request root", func(t *testing.T) {
		b := bufferTestBudget(t, 20, 0)
		root := bufferMustScope(t, b, 8)
		a, err := root.category(5)
		if err != nil {
			t.Fatalf("first category: %v", err)
		}
		other, err := root.category(20)
		if err != nil {
			t.Fatalf("second category: %v", err)
		}
		l, err := a.acquire(context.Background(), 5)
		if err != nil {
			t.Fatalf("category exact fit: %v", err)
		}
		t.Cleanup(l.close)
		_, err = a.acquire(context.Background(), 1)
		bufferAssertLimit(t, err)
		_, err = other.acquire(context.Background(), 4)
		bufferAssertLimit(t, err)
		child, err := other.category(2)
		if err != nil {
			t.Fatalf("nested category: %v", err)
		}
		c, err := child.acquire(context.Background(), 2)
		if err != nil {
			t.Fatalf("nested charge: %v", err)
		}
		t.Cleanup(c.close)
		bufferAssertTotals(t, b, 7, 2, 0)
		l.close()
		c.close()
		if root.used != 0 || a.used != 0 || child.used != 0 || other.used != 0 {
			t.Fatal("closed descendants retained scope charges")
		}
	})
	t.Run("overlapping growth", func(t *testing.T) {
		b := bufferTestBudget(t, 10, 0)
		old := bufferMustAcquire(t, b, 4)
		grown, err := old.grow(context.Background(), 6)
		if err != nil {
			t.Fatalf("grow: %v", err)
		}
		t.Cleanup(grown.close)
		bufferAssertTotals(t, b, 10, 2, 0)
		_, err = old.grow(context.Background(), 1)
		bufferAssertBusy(t, err)
		old.close()
		bufferAssertTotals(t, b, 6, 1, 0)
		if _, err := old.grow(context.Background(), 1); !errors.Is(err, errBufferClosed) {
			t.Fatalf("grow closed holder: %v", err)
		}
	})
	t.Run("simultaneous borrowers", func(t *testing.T) {
		b := bufferTestBudget(t, 8, 0)
		owner := bufferMustAcquire(t, b, 8)
		parser, err := owner.borrow()
		if err != nil {
			t.Fatalf("parser borrow: %v", err)
		}
		t.Cleanup(parser.close)
		stream, err := owner.borrow()
		if err != nil {
			t.Fatalf("stream borrow: %v", err)
		}
		t.Cleanup(stream.close)
		owner.close()
		parser.close()
		bufferAssertTotals(t, b, 8, 1, 0)
		stream.close()
		bufferAssertTotals(t, b, 0, 0, 0)
	})
	t.Run("transfer retains pooled capacity", func(t *testing.T) {
		b := bufferTestBudget(t, 8, 0)
		owner := bufferMustAcquire(t, b, 8)
		pool, err := owner.transfer()
		if err != nil {
			t.Fatalf("transfer: %v", err)
		}
		t.Cleanup(pool.close)
		owner.close()
		bufferAssertTotals(t, b, 8, 1, 0)
		if _, err := owner.transfer(); !errors.Is(err, errBufferClosed) {
			t.Fatalf("transfer closed holder: %v", err)
		}
		if _, err := owner.borrow(); !errors.Is(err, errBufferClosed) {
			t.Fatalf("borrow closed holder: %v", err)
		}
		parser, err := pool.transfer()
		if err != nil {
			t.Fatalf("pool to parser: %v", err)
		}
		t.Cleanup(parser.close)
		pool.close()
		bufferAssertTotals(t, b, 8, 1, 0)
		parser.close()
	})
	t.Run("double close", func(t *testing.T) {
		b := bufferTestBudget(t, 1, 0)
		l := bufferMustAcquire(t, b, 1)
		l.close()
		l.close()
		bufferAssertTotals(t, b, 0, 0, 0)
	})
	t.Run("cancelled borrower still reading", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, time.Hour)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			owner, err := b.acquire(ctx, 1)
			if err != nil {
				t.Fatalf("request acquire: %v", err)
			}
			t.Cleanup(owner.close)
			borrower, err := owner.borrow()
			if err != nil {
				t.Fatalf("borrow: %v", err)
			}
			t.Cleanup(borrower.close)
			finishRead := make(chan struct{})
			finish := sync.OnceFunc(func() { close(finishRead) })
			readDone := make(chan struct{})
			go func() {
				<-ctx.Done()
				<-finishRead
				borrower.close()
				close(readDone)
			}()
			t.Cleanup(func() { finish(); <-readDone })
			waiting := bufferStartAcquire(t, b, context.Background(), 1)
			synctest.Wait()
			cancel()
			owner.close()
			synctest.Wait()
			bufferAssertTotals(t, b, 1, 1, 1)
			waiting.pending(t)
			finish()
			<-readDone
			synctest.Wait()
			waiting.success(t).close()
		})
	})
	t.Run("wait success after release", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 2, time.Second)
			hold := bufferMustAcquire(t, b, 2)
			baseline := runtime.NumGoroutine()
			call := bufferStartAcquire(t, b, context.Background(), 2)
			synctest.Wait()
			bufferAssertTotals(t, b, 2, 1, 1)
			if got := runtime.NumGoroutine(); got != baseline+1 {
				t.Fatalf("goroutines = %d, want only caller added to %d", got, baseline)
			}
			hold.close()
			synctest.Wait()
			call.success(t).close()
		})
	})
	t.Run("timeout is typed busy", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, 5*time.Second)
			bufferMustAcquire(t, b, 1)
			call := bufferStartAcquire(t, b, context.Background(), 1)
			synctest.Wait()
			bufferAdvance(4 * time.Second)
			call.pending(t)
			bufferAdvance(time.Second)
			_, err := call.result(t)
			bufferAssertBusy(t, err)
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("timeout cause: %v", err)
			}
			bufferAssertTotals(t, b, 1, 1, 0)
		})
	})
	t.Run("cancel during wait", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, time.Hour)
			bufferMustAcquire(t, b, 1)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			call := bufferStartAcquire(t, b, ctx, 1)
			synctest.Wait()
			cancel()
			synctest.Wait()
			_, err := call.result(t)
			bufferAssertBusy(t, err)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel cause: %v", err)
			}
			bufferAssertTotals(t, b, 1, 1, 0)
		})
	})
	t.Run("earlier context deadline", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, time.Hour)
			bufferMustAcquire(t, b, 1)
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			call := bufferStartAcquire(t, b, ctx, 1)
			synctest.Wait()
			bufferAdvance(time.Second)
			_, err := call.result(t)
			bufferAssertBusy(t, err)
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("deadline cause: %v", err)
			}
		})
	})
	t.Run("wait zero refuses immediately", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, 0)
			bufferMustAcquire(t, b, 1)
			call := bufferStartAcquire(t, b, context.Background(), 1)
			synctest.Wait()
			_, err := call.result(t)
			bufferAssertBusy(t, err)
			bufferAssertTotals(t, b, 1, 1, 0)
		})
	})
	t.Run("oversized aggregate refuses immediately", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 8, time.Hour)
			call := bufferStartAcquire(t, b, context.Background(), 9)
			synctest.Wait()
			_, err := call.result(t)
			bufferAssertLimit(t, err)
			bufferAssertTotals(t, b, 0, 0, 0)
		})
	})
	t.Run("individual overflow immediate 413", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 8, time.Hour)
			bufferMustAcquire(t, b, 8)
			scope := bufferMustScope(t, b, 4)
			call := bufferStartCall(t, context.Background(), func(ctx context.Context) (*bufferLease, error) { return scope.acquire(ctx, 5) })
			synctest.Wait()
			_, err := call.result(t)
			bufferAssertLimit(t, err)
			bufferAssertTotals(t, b, 8, 1, 0)
		})
	})
	t.Run("FIFO large before later small", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Hour)
			hold := bufferMustAcquire(t, b, 6)
			large := bufferStartAcquire(t, b, context.Background(), 8)
			synctest.Wait()
			small := bufferStartAcquire(t, b, context.Background(), 2)
			synctest.Wait()
			small.pending(t)
			bufferAssertTotals(t, b, 6, 1, 2)
			hold.close()
			synctest.Wait()
			// Both successive heads fit, and were granted without another release.
			large.success(t)
			small.success(t)
			bufferAssertTotals(t, b, 10, 2, 0)
		})
	})
	t.Run("FIFO non-fitting head holds fitting tail", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Hour)
			first := bufferMustAcquire(t, b, 4)
			second := bufferMustAcquire(t, b, 4)
			large := bufferStartAcquire(t, b, context.Background(), 8)
			synctest.Wait()
			small := bufferStartAcquire(t, b, context.Background(), 2)
			synctest.Wait()
			first.close()
			synctest.Wait()
			// Six bytes are free: the tail fits but the live head does not.
			large.pending(t)
			small.pending(t)
			bufferAssertTotals(t, b, 4, 1, 2)
			second.close()
			synctest.Wait()
			large.success(t)
			small.success(t)
			bufferAssertTotals(t, b, 10, 2, 0)
		})
	})
	t.Run("cancelled head wakes successor", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Hour)
			bufferMustAcquire(t, b, 6)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			head := bufferStartAcquire(t, b, ctx, 8)
			synctest.Wait()
			next := bufferStartAcquire(t, b, context.Background(), 4)
			synctest.Wait()
			cancel()
			synctest.Wait()
			_, err := head.result(t)
			bufferAssertBusy(t, err)
			next.success(t)
			bufferAssertTotals(t, b, 10, 2, 0)
		})
	})
	t.Run("timed out head wakes successor", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Second)
			bufferMustAcquire(t, b, 6)
			head := bufferStartAcquire(t, b, context.Background(), 8)
			synctest.Wait()
			bufferAdvance(time.Second / 2)
			next := bufferStartAcquire(t, b, context.Background(), 4)
			synctest.Wait()
			bufferAdvance(time.Second / 2)
			_, err := head.result(t)
			bufferAssertBusy(t, err)
			next.success(t)
			bufferAssertTotals(t, b, 10, 2, 0)
		})
	})
	t.Run("queue overflow", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, time.Hour)
			bufferMustAcquire(t, b, 1)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := make([]*bufferTestCall, 1024)
			for i := range calls {
				calls[i] = bufferStartAcquire(t, b, ctx, 1)
			}
			synctest.Wait()
			bufferAssertTotals(t, b, 1, 1, 1024)
			overflow := bufferStartAcquire(t, b, context.Background(), 1)
			synctest.Wait()
			_, err := overflow.result(t)
			bufferAssertBusy(t, err)
			bufferAssertTotals(t, b, 1, 1, 1024)
			cancel()
			synctest.Wait()
			for _, call := range calls {
				_, err := call.result(t)
				bufferAssertBusy(t, err)
			}
			bufferAssertTotals(t, b, 1, 1, 0)
		})
	})
	t.Run("grant cancel race", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, time.Hour)
			scope := bufferMustScope(t, b, 1)
			for range 50 {
				hold := bufferMustAcquire(t, b, 1)
				ctx, cancel := context.WithCancel(context.Background())
				call := bufferStartCall(t, ctx, func(ctx context.Context) (*bufferLease, error) { return scope.acquire(ctx, 1) })
				synctest.Wait()
				// Hold the arbitration lock until both events are observable. The
				// waiter has a real grant but must see cancellation before returning.
				b.mu.Lock()
				hold.closeLocked()
				b.reconsiderLocked()
				cancel()
				b.mu.Unlock()
				synctest.Wait()
				lease, err := call.result(t)
				if lease != nil {
					t.Fatal("cancelled grant returned a lease")
				}
				bufferAssertBusy(t, err)
				bufferAssertTotals(t, b, 0, 0, 0)
				if scope.used != 0 {
					t.Fatalf("cancelled grant retained scope bytes: %d", scope.used)
				}
			}
		})
	})
	t.Run("grant timeout race", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, time.Hour)
			scope := bufferMustScope(t, b, 1)
			hold := bufferMustAcquire(t, b, 1)
			call := bufferStartCall(t, context.Background(), func(ctx context.Context) (*bufferLease, error) { return scope.acquire(ctx, 1) })
			synctest.Wait()
			// Grant, then move the allocator clock past the waiter's deadline
			// before it can arbitrate. Either outcome is valid; leaking is not.
			b.mu.Lock()
			hold.closeLocked()
			b.reconsiderLocked()
			expired := time.Now().Add(2 * time.Hour)
			b.now = func() time.Time { return expired }
			b.mu.Unlock()
			synctest.Wait()
			lease, err := call.result(t)
			if lease != nil {
				bufferAssertTotals(t, b, 1, 1, 0)
				lease.close()
			} else {
				bufferAssertBusy(t, err)
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("timeout cause: %v", err)
				}
			}
			bufferAssertTotals(t, b, 0, 0, 0)
			if scope.used != 0 {
				t.Fatalf("expired grant retained scope bytes: %d", scope.used)
			}
		})
	})
	t.Run("growth waits without releasing old capacity", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Second)
			old := bufferMustAcquire(t, b, 4)
			other := bufferMustAcquire(t, b, 4)
			call := bufferStartCall(t, context.Background(), func(ctx context.Context) (*bufferLease, error) { return old.grow(ctx, 6) })
			synctest.Wait()
			bufferAssertTotals(t, b, 8, 2, 1)
			other.close()
			synctest.Wait()
			grown := call.success(t)
			bufferAssertTotals(t, b, 10, 2, 0)
			old.close()
			bufferAssertTotals(t, b, 6, 1, 0)
			grown.close()
		})
	})
	t.Run("growth timeout retains old capacity", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Second)
			old := bufferMustAcquire(t, b, 6)
			call := bufferStartCall(t, context.Background(), func(ctx context.Context) (*bufferLease, error) { return old.grow(ctx, 6) })
			synctest.Wait()
			bufferAssertTotals(t, b, 6, 1, 1)
			bufferAdvance(time.Second)
			_, err := call.result(t)
			bufferAssertBusy(t, err)
			bufferAssertTotals(t, b, 6, 1, 0)
		})
	})
	t.Run("pending categories reserve root room", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Hour)
			bufferMustAcquire(t, b, 10)
			scope := bufferMustScope(t, b, 4)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			call := bufferStartCall(t, ctx, func(ctx context.Context) (*bufferLease, error) { return scope.acquire(ctx, 4) })
			synctest.Wait()
			_, err := scope.acquire(context.Background(), 1)
			bufferAssertLimit(t, err)
			cancel()
			synctest.Wait()
			_, err = call.result(t)
			bufferAssertBusy(t, err)
			if scope.used != 0 {
				t.Fatalf("cancelled proposal retained request room: %d", scope.used)
			}
		})
	})
	t.Run("capacity increase grants FIFO heads", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Hour)
			bufferMustAcquire(t, b, 6)
			head := bufferStartAcquire(t, b, context.Background(), 8)
			synctest.Wait()
			next := bufferStartAcquire(t, b, context.Background(), 2)
			synctest.Wait()
			if err := b.reconfigure(16, time.Hour); err != nil {
				t.Fatalf("increase capacity: %v", err)
			}
			synctest.Wait()
			head.success(t)
			next.success(t)
			bufferAssertTotals(t, b, 16, 3, 0)
		})
	})
	t.Run("capacity decrease drains existing leases", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Hour)
			a := bufferMustAcquire(t, b, 6)
			c := bufferMustAcquire(t, b, 4)
			if err := b.reconfigure(5, time.Hour); err != nil {
				t.Fatalf("decrease capacity: %v", err)
			}
			call := bufferStartAcquire(t, b, context.Background(), 1)
			synctest.Wait()
			call.pending(t)
			bufferAssertTotals(t, b, 10, 2, 1)
			c.close()
			synctest.Wait()
			call.pending(t)
			bufferAssertTotals(t, b, 6, 1, 1)
			a.close()
			synctest.Wait()
			call.success(t)
			bufferAssertTotals(t, b, 1, 1, 0)
		})
	})
	t.Run("capacity decrease refuses oversized queued entries", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 10, time.Hour)
			hold := bufferMustAcquire(t, b, 10)
			head := bufferStartAcquire(t, b, context.Background(), 8)
			synctest.Wait()
			next := bufferStartAcquire(t, b, context.Background(), 2)
			synctest.Wait()
			tail := bufferStartAcquire(t, b, context.Background(), 6)
			synctest.Wait()
			if err := b.reconfigure(5, time.Hour); err != nil {
				t.Fatalf("decrease capacity: %v", err)
			}
			synctest.Wait()
			_, err := head.result(t)
			bufferAssertLimit(t, err)
			_, err = tail.result(t)
			bufferAssertLimit(t, err)
			bufferAssertTotals(t, b, 10, 1, 1)
			next.pending(t)
			hold.close()
			synctest.Wait()
			next.success(t)
		})
	})
	t.Run("wait duration snapshots", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, 5*time.Second)
			bufferMustAcquire(t, b, 1)
			original := bufferStartAcquire(t, b, context.Background(), 1)
			synctest.Wait()
			if err := b.reconfigure(1, 0); err != nil {
				t.Fatalf("disable subsequent waits: %v", err)
			}
			immediate := bufferStartAcquire(t, b, context.Background(), 1)
			synctest.Wait()
			_, err := immediate.result(t)
			bufferAssertBusy(t, err)
			if err := b.reconfigure(1, 10*time.Second); err != nil {
				t.Fatalf("lengthen subsequent waits: %v", err)
			}
			later := bufferStartAcquire(t, b, context.Background(), 1)
			synctest.Wait()
			bufferAdvance(5 * time.Second)
			_, err = original.result(t)
			bufferAssertBusy(t, err)
			later.pending(t)
			bufferAssertTotals(t, b, 1, 1, 1)
			bufferAdvance(5 * time.Second)
			_, err = later.result(t)
			bufferAssertBusy(t, err)
		})
	})
	t.Run("repeated wait cancel timeout cycles", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 1, time.Second)
			baseline := runtime.NumGoroutine()
			for range 30 {
				hold := bufferMustAcquire(t, b, 1)
				ctx, cancel := context.WithCancel(context.Background())
				cancelled := bufferStartAcquire(t, b, ctx, 1)
				synctest.Wait()
				cancel()
				synctest.Wait()
				_, err := cancelled.result(t)
				bufferAssertBusy(t, err)
				timed := bufferStartAcquire(t, b, context.Background(), 1)
				synctest.Wait()
				bufferAdvance(time.Second)
				_, err = timed.result(t)
				bufferAssertBusy(t, err)
				granted := bufferStartAcquire(t, b, context.Background(), 1)
				synctest.Wait()
				hold.close()
				synctest.Wait()
				granted.success(t).close()
				bufferAssertTotals(t, b, 0, 0, 0)
			}
			synctest.Wait()
			if got := runtime.NumGoroutine(); got != baseline {
				t.Fatalf("goroutines after joined cycles = %d, want %d", got, baseline)
			}
		})
	})
	t.Run("concurrent acquire release", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			b := bufferTestBudget(t, 8, time.Hour)
			start := make(chan struct{})
			results := make(chan error, 32)
			for range 32 {
				go func() {
					<-start
					for range 40 {
						lease, err := b.acquire(context.Background(), 1)
						if err != nil {
							results <- err
							return
						}
						borrower, err := lease.borrow()
						if err != nil {
							lease.close()
							results <- err
							return
						}
						lease.close()
						borrower.close()
						borrower.close()
					}
					results <- nil
				}()
			}
			close(start)
			for range 32 {
				if err := <-results; err != nil {
					t.Errorf("concurrent caller: %v", err)
				}
			}
			synctest.Wait()
			bufferAssertTotals(t, b, 0, 0, 0)
		})
	})
	t.Run("concurrent borrower close and reconfigure", func(t *testing.T) {
		b := bufferTestBudget(t, 8, 0)
		owner := bufferMustAcquire(t, b, 8)
		var wg sync.WaitGroup
		start := make(chan struct{})
		errs := make(chan error, 16)
		for range 16 {
			borrower, err := owner.borrow()
			if err != nil {
				t.Fatalf("borrow: %v", err)
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				borrower.close()
				borrower.close()
				errs <- b.reconfigure(4, 0)
			}()
		}
		close(start)
		owner.close()
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Errorf("concurrent reconfigure: %v", err)
			}
		}
		bufferAssertTotals(t, b, 0, 0, 0)
	})
}
