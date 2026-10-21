// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"net/netip"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

func spendAttempt(t *testing.T, r *routingSpend) *publicAttemptLifecycle {
	t.Helper()
	a, err := r.newAttempt()
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func spendGrant(t *testing.T, r *routingSpend, ceiling nodepolicy.NanoUSD, amount nodepolicy.NanoUSD) *publicAttemptLifecycle {
	t.Helper()
	a := spendAttempt(t, r)
	remaining, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: ceiling})
	n, parseErr := amount.Int64()
	if err != nil || parseErr != nil || n > remaining {
		t.Fatalf("grant setup: remaining=%d amount=%s errors=%v/%v", remaining, amount, err, parseErr)
	}
	if err := a.associated(accountReservation{Hold: accountHold{Attempt: "synthetic-d-hold", Reserved: amount}, Durability: accountReserveDurable}); err != nil {
		t.Fatal(err)
	}
	a.finished(publicAttemptTerminal{Accounting: publicAccountingRecorded})
	return a
}

func TestRoutingSpend(t *testing.T) {
	t.Run("permission refused paid candidate still tightens", func(t *testing.T) {
		p, f, _, snapshot := egressFixture(t)
		r := &routingSpend{}
		spendGrant(t, r, "100", "60")
		policy := p.admission.currentPolicy()
		policy.Routing.Accounts[0].PerRequestCeiling = "50"
		if err := p.admission.activatePolicy(policy); err != nil {
			t.Fatal(err)
		}
		if err := f.ledger.applyRouting(policy.Routing, policy.Revision); err != nil {
			t.Fatal(err)
		}
		var err error
		snapshot, err = f.ledger.prepareChatAccount(snapshot.deployment.ID)
		if err != nil {
			t.Fatal(err)
		}
		a := spendAttempt(t, r)
		if _, err := p.runPublicChatAttempt(f.ctx, "weak", snapshot, []byte(f.body), f.profile, f.transport, f.resolve, f.budget.acquire, a, f.emit); err == nil || r.ceiling != 50 || r.granted != 60 || len(f.ledger.state.Holds) != 0 || f.calls.Load() != 0 {
			t.Fatalf("permission refusal lost ceiling: err=%v cap=%d grants=%d", err, r.ceiling, r.granted)
		}
	})
	t.Run("D enforces exact remaining without settlement refill", func(t *testing.T) {
		f := newChatAccountFixture(t)
		r := &routingSpend{}
		reserve := func(input, output int64) (accountReservation, error) {
			t.Helper()
			a := spendAttempt(t, r)
			s, err := f.ledger.prepareChatAccount("chat-a")
			if err != nil {
				t.Fatal(err)
			}
			remaining, err := a.tryAccount(s.account)
			if err != nil {
				a.finished(publicAttemptTerminal{})
				return accountReservation{}, err
			}
			reservation, reserveErr := f.ledger.reserveChatAccountAllowance(s, input, output, remaining)
			if err := a.associated(reservation); err != nil {
				t.Fatal(err)
			}
			a.finished(publicAttemptTerminal{})
			return reservation, reserveErr
		}
		setCeiling := func(ceiling nodepolicy.NanoUSD, revision string) {
			t.Helper()
			f.routing.Accounts[0].PerRequestCeiling = ceiling
			if err := f.ledger.applyRouting(f.routing, revision); err != nil {
				t.Fatal(err)
			}
		}
		setCeiling("1000", "8")
		first, err := reserve(100, 200) // D prices this immutable hold at 500.
		if err != nil || first.Hold.Reserved != "500" {
			t.Fatalf("first reserve: %+v %v", first, err)
		}
		if err := f.ledger.reconcile(first.Hold.Attempt, &tokenUsage{InputTokens: 1, OutputTokens: 1}); err != nil {
			t.Fatal(err)
		}
		if actual := f.ledger.state.hold(first.Hold.Attempt).Actual; actual != "3" {
			t.Fatalf("actual=%s want 3", actual)
		}
		setCeiling("700", "9")
		second, err := reserve(0, 100) // Exact remaining 200.
		if err != nil || second.Hold.Reserved != "200" || r.granted != 700 || r.ceiling != 700 {
			t.Fatalf("exact reserve: %+v %v request=%+v", second, err, r)
		}
		setCeiling("2000", "10")
		before := readLedgerBytes(t, filepath.Dir(f.ledger.storePath))
		if _, err := reserve(0, 1); !errors.Is(err, errRoutingSpend) {
			t.Fatalf("higher policy refilled: %v", err)
		}
		if len(f.ledger.state.Holds) != 2 || string(readLedgerBytes(t, filepath.Dir(f.ledger.storePath))) != string(before) || r.ceiling != 700 {
			t.Fatal("insufficient remainder reached a durable write")
		}
	})
	t.Run("D refused paid candidate tightens current policy", func(t *testing.T) {
		f := newChatAccountFixture(t)
		r := &routingSpend{}
		spendGrant(t, r, "1000", "100")
		f.routing.Accounts[0].PerRequestCeiling = "200"
		f.routing.Accounts[0].DailyCeiling = "0"
		if err := f.ledger.applyRouting(f.routing, "8"); err != nil {
			t.Fatal(err)
		}
		a := spendAttempt(t, r)
		s, err := f.ledger.prepareChatAccount("chat-a")
		if err != nil {
			t.Fatal(err)
		}
		remaining, err := a.tryAccount(s.account)
		if err != nil || remaining != 100 {
			t.Fatalf("remaining=%d %v", remaining, err)
		}
		reservation, err := f.ledger.reserveChatAccountAllowance(s, 0, 1, remaining)
		if !errors.Is(err, errAccountCeiling) || reservation.Durability != accountReserveNone {
			t.Fatalf("D refusal=%+v %v", reservation, err)
		}
		a.finished(publicAttemptTerminal{})
		if remaining, err := spendAttempt(t, r).tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "2000"}); err != nil || remaining != 100 || r.ceiling != 200 {
			t.Fatalf("refused candidate did not tighten: remaining=%d cap=%d %v", remaining, r.ceiling, err)
		}
	})
	t.Run("exact remaining and no settlement refill", func(t *testing.T) {
		r := &routingSpend{}
		spendGrant(t, r, "100", "60")
		a := spendAttempt(t, r)
		remaining, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "100"})
		if err != nil || remaining != 40 || r.granted != 60 {
			t.Fatalf("remaining=%d granted=%d err=%v, want 40/60", remaining, r.granted, err)
		}
		if err := a.associated(accountReservation{Hold: accountHold{Attempt: "second", Reserved: "40"}, Durability: accountReserveDurable}); err != nil {
			t.Fatal(err)
		}
		a.finished(publicAttemptTerminal{Accounting: publicAccountingRecorded})
		if remaining, err := spendAttempt(t, r).tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "100"}); err != nil || remaining != 0 || r.granted != 100 {
			t.Fatalf("settlement refilled allowance: remaining=%d granted=%d err=%v", remaining, r.granted, err)
		}
	})
	t.Run("high low high including refused candidate", func(t *testing.T) {
		r := &routingSpend{}
		spendGrant(t, r, "100", "20")
		a := spendAttempt(t, r)
		if remaining, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "30"}); err != nil || remaining != 10 {
			t.Fatalf("lower candidate: %d %v", remaining, err)
		}
		a.finished(publicAttemptTerminal{Accounting: publicAccountingRetained, Err: errAccountCooldown})
		if remaining, err := spendAttempt(t, r).tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "200"}); err != nil || remaining != 10 || r.ceiling != 30 {
			t.Fatalf("higher ceiling restored room: remaining=%d cap=%d err=%v", remaining, r.ceiling, err)
		}
	})
	t.Run("tightening below granted never undoes grants", func(t *testing.T) {
		r := &routingSpend{}
		spendGrant(t, r, "100", "60")
		a := spendAttempt(t, r)
		if _, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "50"}); !errors.Is(err, errRoutingSpend) {
			t.Fatalf("tightening below grants accepted: %v", err)
		}
		a.finished(publicAttemptTerminal{})
		next := spendAttempt(t, r)
		if _, err := next.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: "200"}); !errors.Is(err, errRoutingSpend) || r.granted != 60 || r.ceiling != 50 {
			t.Fatalf("tightened allowance reset: cap=%d granted=%d err=%v", r.ceiling, r.granted, err)
		}
		next.finished(publicAttemptTerminal{})
		if remaining, err := spendAttempt(t, r).tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingFree, PerRequestCeiling: "0"}); err != nil || remaining != 0 || r.granted != 60 || r.ceiling != 50 {
			t.Fatalf("zero-cost transition refused or reset tightened state: remaining=%d err=%v", remaining, err)
		}
	})
	t.Run("free trial and destination transitions preserve state", func(t *testing.T) {
		r := &routingSpend{}
		for _, billing := range []nodepolicy.BillingClass{nodepolicy.BillingFree, nodepolicy.BillingTrial} {
			a := spendAttempt(t, r)
			if remaining, err := a.tryAccount(nodepolicy.Account{Billing: billing, PerRequestCeiling: "0"}); err != nil || remaining != math.MaxInt64 || r.limited {
				t.Fatalf("nonpaid pinned zero: %d %v", remaining, err)
			}
			a.finished(publicAttemptTerminal{})
		}
		spendGrant(t, r, "100", "60")
		a := spendAttempt(t, r)
		if remaining, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingFree, PerRequestCeiling: "0"}); err != nil || remaining != 40 {
			t.Fatalf("free transition reset allowance: %d %v", remaining, err)
		}
		a.finished(publicAttemptTerminal{})
		// Local/peer/regenerate retain this same request object without a hold.
		if r.granted != 60 || r.ceiling != 100 {
			t.Fatal("destination transition lost request allowance")
		}
	})
	t.Run("zero and invalid ceilings", func(t *testing.T) {
		for _, ceiling := range []nodepolicy.NanoUSD{"0", "-1", "9223372036854775808"} {
			a := chatTestLifecycle(t)
			remaining, err := a.tryAccount(nodepolicy.Account{Billing: nodepolicy.BillingPaid, PerRequestCeiling: ceiling})
			if ceiling == "0" && (remaining != 0 || err != nil) || ceiling != "0" && err == nil {
				t.Fatalf("ceiling %s remaining=%d err=%v", ceiling, remaining, err)
			}
		}
	})
	t.Run("grant addition overflow blocks request", func(t *testing.T) {
		r := &routingSpend{granted: math.MaxInt64}
		a := spendAttempt(t, r)
		if err := a.associated(accountReservation{Hold: accountHold{Attempt: "overflow", Reserved: "1"}, Durability: accountReserveDurable}); !errors.Is(err, errMoneyOverflow) || !r.uncertain {
			t.Fatalf("overflow granted: %v uncertain=%v", err, r.uncertain)
		}
	})
}

func TestRoutingSpendCurrentAccountPolicy(t *testing.T) {
	tests := []struct {
		name                            string
		snapshotBilling, currentBilling nodepolicy.BillingClass
		snapshotCeiling, currentCeiling nodepolicy.NanoUSD
		priorGrant, removed             bool
		wantLimited                     bool
		wantCeiling, wantRemaining      int64
	}{
		{"stale paid snapshot observes current lower ceiling", nodepolicy.BillingPaid, nodepolicy.BillingPaid, "1000000", "100", false, false, true, 100, 100},
		{"stale lower snapshot does not invent a tried ceiling", nodepolicy.BillingPaid, nodepolicy.BillingPaid, "100", "500", false, false, true, 500, 500},
		{"current increase cannot loosen prior request grants", nodepolicy.BillingPaid, nodepolicy.BillingPaid, "1000000", "500", true, false, true, 100, 40},
		{"paid snapshot now free does not pin zero", nodepolicy.BillingPaid, nodepolicy.BillingFree, "1000000", "0", false, false, false, 0, 500},
		{"free snapshot now paid observes current ceiling", nodepolicy.BillingFree, nodepolicy.BillingPaid, "0", "100", false, false, true, 100, 100},
		{"free current account preserves prior request grants", nodepolicy.BillingPaid, nodepolicy.BillingFree, "1000000", "0", true, false, true, 100, 40},
		{"missing current account refuses without observing stale account", nodepolicy.BillingPaid, nodepolicy.BillingPaid, "1000000", "100", false, true, false, 0, 500},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, f, _, original := egressFixture(t)
			activate := func(billing nodepolicy.BillingClass, ceiling nodepolicy.NanoUSD, revision string, removed bool) {
				t.Helper()
				policy := p.admission.currentPolicy()
				policy.Revision = revision
				accounts := make([]nodepolicy.Account, 0, len(policy.Routing.Accounts))
				for _, account := range policy.Routing.Accounts {
					if account.ID == original.account.ID {
						if removed {
							continue
						}
						account.Billing, account.PerRequestCeiling = billing, ceiling
					}
					accounts = append(accounts, account)
				}
				policy.Routing.Accounts = accounts
				token, err := p.beginPublicActivation(policy)
				if err != nil {
					t.Fatal(err)
				}
				if err := p.accounts.applyRouting(policy.Routing, policy.Revision); err != nil {
					t.Fatal(err)
				}
				p.completePublicActivation(token, nil)
			}
			activate(tt.snapshotBilling, tt.snapshotCeiling, "8", false)
			snapshot, err := p.accounts.prepareChatAccount(original.deployment.ID)
			if err != nil {
				t.Fatal(err)
			}
			activate(tt.currentBilling, tt.currentCeiling, "9", tt.removed)
			r := &routingSpend{}
			wantGranted := int64(0)
			if tt.priorGrant {
				spendGrant(t, r, "100", "60")
				wantGranted = 60
			}
			a := spendAttempt(t, r)
			lookups := 0
			resolve := func(nodepolicy.CredentialRef) (string, bool) { lookups++; return "synthetic", true }
			_, err = p.runPublicChatAttempt(f.ctx, "weak", snapshot, []byte(f.body), f.profile, f.transport, resolve, f.budget.acquire, a, f.emit)
			var refusal *publicAttemptRefusal
			if !errors.As(err, &refusal) || refusal.Stage != publicBeforeReserve {
				t.Fatalf("stale candidate refusal=%v, want before-reserve", err)
			}
			if r.limited != tt.wantLimited || r.ceiling != tt.wantCeiling || r.granted != wantGranted {
				t.Fatalf("current account observation: limited=%v ceiling=%d granted=%d, want %v/%d/%d", r.limited, r.ceiling, r.granted, tt.wantLimited, tt.wantCeiling, wantGranted)
			}
			if snapshot.account.Billing != tt.snapshotBilling || snapshot.account.PerRequestCeiling != tt.snapshotCeiling || snapshot.revision != "8" {
				t.Fatal("current-policy observation mutated D's immutable snapshot")
			}
			b := spendAttempt(t, r)
			remaining, err := b.tryAccount(nodepolicy.Account{ID: "other-paid", Billing: nodepolicy.BillingPaid, PerRequestCeiling: "500"})
			if err != nil || remaining != tt.wantRemaining {
				t.Fatalf("fallback remaining=%d err=%v, want current-policy remaining %d", remaining, err, tt.wantRemaining)
			}
			if lookups != 0 || len(p.accounts.state.Holds) != 0 || f.calls.Load() != 0 || f.budget.used.Load() != 0 {
				t.Fatalf("refusal crossed lookup/reserve/Do/budget boundary: %d/%d/%d/%d", lookups, len(p.accounts.state.Holds), f.calls.Load(), f.budget.used.Load())
			}
		})
	}
}

func lifecycleRun(t *testing.T, f *chatAccountFixture, a *publicAttemptLifecycle, guard publicAttemptGuard) (chatParseResult, error) {
	t.Helper()
	s, err := f.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	return runChatAccountAttempt(f.ctx, s, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, guard, a, f.emit)
}

func TestPublicAttemptLifecycle(t *testing.T) {
	t.Run("DNS wait cancellation precedes acceptance", func(t *testing.T) {
		f := newChatAccountFixture(t)
		r := &routingSpend{}
		a := spendAttempt(t, r)
		entered := make(chan struct{})
		f.transport.resolve = func(ctx context.Context, _ string) ([]netip.Addr, error) {
			close(entered)
			<-ctx.Done()
			return nil, ctx.Err()
		}
		done := make(chan error, 1)
		go func() { _, err := lifecycleRun(t, f, a, chatTestGuard(f.guard)); done <- err }()
		<-entered
		r.mu.Lock()
		accepted := a.accepted
		r.mu.Unlock()
		if accepted {
			t.Error("DNS blocked after acceptance")
		}
		r.cancelRequest()
		if err := <-done; err == nil || a.accepted || f.calls.Load() != 0 {
			t.Fatalf("DNS cancellation crossed acceptance: err=%v accepted=%v Do=%d", err, a.accepted, f.calls.Load())
		}
	})
	t.Run("nil lifecycle denies before lookup", func(t *testing.T) {
		f := newChatAccountFixture(t)
		lookups := 0
		f.resolve = func(nodepolicy.CredentialRef) (string, bool) { lookups++; return "synthetic", true }
		if _, err := lifecycleRun(t, f, nil, chatTestGuard(f.guard)); err == nil || lookups != 0 || len(f.ledger.state.Holds) != 0 || f.calls.Load() != 0 {
			t.Fatalf("missing authority effects: err=%v lookups=%d holds=%d Do=%d", err, lookups, len(f.ledger.state.Holds), f.calls.Load())
		}
	})
	t.Run("reserve cancel attaches exact retained hold", func(t *testing.T) {
		f := newChatAccountFixture(t)
		r := &routingSpend{}
		a := spendAttempt(t, r)
		entered, resume := make(chan struct{}), make(chan struct{})
		write := f.ledger.write
		f.ledger.write = func(path string, data []byte, perm os.FileMode) error {
			close(entered)
			<-resume
			return write(path, data, perm)
		}
		done := make(chan error, 1)
		go func() { _, err := lifecycleRun(t, f, a, chatTestGuard(f.guard)); done <- err }()
		<-entered
		r.cancelRequest()
		if _, err := r.newAttempt(); err == nil {
			t.Fatal("replacement admitted during cancelled reserve")
		}
		close(resume)
		if err := <-done; err == nil || a.reservation.Durability != accountReserveDurable || a.reservation.Hold.Attempt != f.ledger.state.Holds[0].Attempt || r.granted == 0 || f.calls.Load() != 0 || a.accepted {
			t.Fatalf("cancelled reserve association: err=%v reservation=%+v granted=%d Do=%d", err, a.reservation, r.granted, f.calls.Load())
		}
		egressUnsentHold(t, f)
	})
	t.Run("uncertain reserve blocks later public attempts", func(t *testing.T) {
		f := newChatAccountFixture(t)
		r := &routingSpend{}
		a := spendAttempt(t, r)
		write := f.ledger.write
		f.ledger.write = func(path string, data []byte, perm os.FileMode) error {
			if err := write(path, data, perm); err != nil {
				return err
			}
			return durablefile.ErrReplaced
		}
		if _, err := lifecycleRun(t, f, a, chatTestGuard(f.guard)); !errors.Is(err, errChatAccounting) || a.reservation.Durability != accountReserveUncertain || a.reservation.Hold.Attempt == "" || !r.uncertain || r.granted == 0 || f.calls.Load() != 0 {
			t.Fatalf("uncertain reserve lost effects: err=%v reservation=%+v request=%+v", err, a.reservation, r)
		}
		f.ledger.write = write
		if _, err := r.newAttempt(); !errors.Is(err, errRoutingSpend) || len(f.ledger.state.Holds) != 1 || f.ledger.state.Holds[0].State != holdUncertain {
			t.Fatalf("uncertain request retried: err=%v holds=%+v", err, f.ledger.state.Holds)
		}
	})
	t.Run("definite reserve refusal carries no hold", func(t *testing.T) {
		f := newChatAccountFixture(t)
		r := &routingSpend{}
		a := spendAttempt(t, r)
		f.ledger.write = func(string, []byte, os.FileMode) error { return errors.New("injected before replacement") }
		if _, err := lifecycleRun(t, f, a, chatTestGuard(f.guard)); !errors.Is(err, errChatAccounting) || a.reservation.Durability != accountReserveNone || r.uncertain || r.granted != 0 {
			t.Fatalf("definite refusal model: %v %+v", err, a.reservation)
		}
		if _, err := r.newAttempt(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("final cancel wins before Do", func(t *testing.T) {
		f := newChatAccountFixture(t)
		r := &routingSpend{}
		a := spendAttempt(t, r)
		guard := func(ctx context.Context, accept func() error) error {
			if accept != nil {
				r.cancelRequest()
				return accept()
			}
			return nil
		}
		if _, err := lifecycleRun(t, f, a, guard); !errors.Is(err, errPublicFinalGuard) || a.accepted || f.calls.Load() != 0 {
			t.Fatalf("cancel arbitration: err=%v accepted=%v Do=%d", err, a.accepted, f.calls.Load())
		}
		egressUnsentHold(t, f)
	})
	t.Run("accepted cancellation closes accepted context", func(t *testing.T) {
		f := newChatAccountFixture(t)
		r := &routingSpend{}
		a := spendAttempt(t, r)
		entered, done := make(chan struct{}), make(chan error, 1)
		f.transport.client.Transport = publicTestRoundTrip(func(req *http.Request) (*http.Response, error) {
			f.calls.Add(1)
			close(entered)
			<-req.Context().Done()
			return nil, req.Context().Err()
		})
		go func() { _, err := lifecycleRun(t, f, a, chatTestGuard(f.guard)); done <- err }()
		<-entered
		r.cancelRequest()
		if err := <-done; err == nil || !a.accepted || a.committed || f.calls.Load() != 1 {
			t.Fatalf("accepted cancellation: %v accepted=%v committed=%v Do=%d", err, a.accepted, a.committed, f.calls.Load())
		}
	})
	for _, stream := range []bool{false, true} {
		name := "invalid JSON commits raw byte"
		if stream {
			name = "invalid event commits raw byte"
		}
		t.Run(name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			if stream {
				f.body = strings.TrimSuffix(chatSimple, "}") + `,"stream":true}`
			}
			f.response = "!invalid"
			r := &routingSpend{}
			a := spendAttempt(t, r)
			result, err := lifecycleRun(t, f, a, chatTestGuard(f.guard))
			if err == nil || !a.committed || !r.committed || result.Complete || result.Exposed || a.terminal.Delivered {
				t.Fatalf("invalid raw-byte boundary: err=%v committed=%v result=%+v", err, a.committed, result)
			}
			if _, err := r.newAttempt(); err == nil {
				t.Fatal("committed invalid protocol permitted fallback")
			}
		})
	}
	t.Run("empty success and provider error do not commit", func(t *testing.T) {
		for _, status := range []int{http.StatusOK, http.StatusBadRequest} {
			f := newChatAccountFixture(t)
			f.status = status
			f.response = ""
			if status != http.StatusOK {
				f.response = "non-success error body"
			}
			a := chatTestLifecycle(t)
			if _, err := lifecycleRun(t, f, a, chatTestGuard(f.guard)); err == nil || a.committed {
				t.Fatalf("status %d committed: err=%v commit=%v", status, err, a.committed)
			}
		}
	})
	t.Run("delivery complete survives accounting failure", func(t *testing.T) {
		f := newChatAccountFixture(t)
		a := chatTestLifecycle(t)
		f.emit = func([]byte) error {
			f.ledger.write = func(string, []byte, os.FileMode) error { return errors.New("injected accounting failure") }
			return nil
		}
		result, err := lifecycleRun(t, f, a, chatTestGuard(f.guard))
		if !errors.Is(err, errChatAccounting) || !result.Complete || !a.committed || !a.terminal.Delivered || a.terminal.Accounting != publicAccountingFailed {
			t.Fatalf("accounting rewrote delivery: result=%+v err=%v terminal=%+v", result, err, a.terminal)
		}
		if _, err := a.request.newAttempt(); err == nil {
			t.Fatal("delivered accounting failure permitted resend")
		}
	})
	t.Run("one parser allowance and one transfer", func(t *testing.T) {
		f := newChatAccountFixture(t)
		acquire := f.budget.acquire
		fixed := 0
		f.budget.ceiling = 64 << 20
		budget := func(n int64) (func(), error) {
			if n == chatFixedStorage+256 {
				fixed++
			}
			return acquire(n)
		}
		s, err := f.ledger.prepareChatAccount("chat-a")
		if err != nil {
			t.Fatal(err)
		}
		a := chatTestLifecycle(t)
		if _, err := runChatAccountAttempt(f.ctx, s, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, budget, chatTestGuard(f.guard), a, f.emit); err != nil {
			t.Fatal(err)
		}
		// Request preparation owns one fixed lease; response parsing owns one.
		if fixed != 2 {
			t.Fatalf("fixed reservations=%d, want 2 including preparation", fixed)
		}
		allowance, err := f.profile.acquireParserAllowance(budget, false)
		if err != nil {
			t.Fatal(err)
		}
		lease := allowance.take(f.profile, false)
		if lease == nil {
			t.Fatal("first transfer refused")
		}
		if allowance.take(f.profile, false) != nil {
			t.Fatal("allowance transferred twice")
		}
		allowance.release()
		if f.budget.used.Load() == 0 {
			t.Fatal("transfer released live parser storage")
		}
		lease.release()
		if f.budget.used.Load() != 0 {
			t.Fatal("parser storage leaked")
		}
	})
}

// These cases execute E/D's existing one-attempt seam with the request's real
// sink. Synthetic transport never dials a provider; all storage uses I4a's root.
func TestPublicAttemptLifecycleOwnedSink(t *testing.T) {
	attach := func(t *testing.T, f *chatAccountFixture, w http.ResponseWriter) (*routingSpend, *publicAttemptLifecycle, *responseSink) {
		t.Helper()
		ctx, root, buffers := chatTestInferenceContext(t, 64<<20, 0)
		parser, err := buffers.scope.tryAcquire(ctx, 96<<10)
		if err != nil {
			t.Fatal(err)
		}
		buffers.retainParser(parser)
		request := newRoutingRequest(nil)
		sink := newResponseSink(w, func() { request.cancelWithReason("client connection lost") })
		request.sink = sink
		f.ctx = withResponseSink(ctx, sink)
		f.body = strings.TrimSuffix(chatSimple, "}") + `,"stream":true}`
		f.response = chatCompleteStream()
		attempt := spendAttempt(t, request)
		t.Cleanup(func() {
			if f.budget.used.Load() != 0 || root.used != 96<<10 || request.active != nil {
				t.Errorf("live attempt resources: explicit=%d root=%d active=%v", f.budget.used.Load(), root.used, request.active != nil)
			}
			request.finish()
			buffers.close()
			if root.used != 0 || root.live != 0 || root.waiters.Len() != 0 {
				t.Errorf("owned sink leaked: used=%d live=%d queue=%d", root.used, root.live, root.waiters.Len())
			}
		})
		return request, attempt, sink
	}
	writer := func() *responseSinkWriter { return &responseSinkWriter{header: make(http.Header)} }

	t.Run("raw commitment preserves live public parser after request deadline", func(t *testing.T) {
		f := newChatAccountFixture(t)
		w := writer()
		request, attempt, sink := attach(t, f, w)
		request.deadline = time.Now().Add(200 * time.Millisecond)
		reader, pipe := io.Pipe()
		defer reader.Close()
		defer pipe.Close()
		release := make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(release) }) }
		defer unblock()
		transport := f.transport.client.Transport
		f.transport.client.Transport = publicTestRoundTrip(func(req *http.Request) (*http.Response, error) {
			response, err := transport.RoundTrip(req)
			if err != nil {
				return response, err
			}
			if err := response.Body.Close(); err != nil {
				return nil, err
			}
			response.Body = reader
			return response, nil
		})
		written := make(chan error, 1)
		go func() {
			end := strings.Index(f.response, "\n\n") + 2
			_, err := io.WriteString(pipe, f.response[:end])
			if err == nil {
				<-release
				_, err = io.WriteString(pipe, f.response[end:])
			}
			closeErr := pipe.Close()
			written <- errors.Join(err, closeErr)
		}()
		type completion struct {
			result chatParseResult
			err    error
		}
		done := make(chan completion, 1)
		go func() {
			result, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard))
			done <- completion{result, err}
		}()
		waitForCond(t, time.Second, "public raw commitment", func() bool { request.mu.Lock(); defer request.mu.Unlock(); return request.committed })
		<-time.After(time.Until(request.deadline) + 50*time.Millisecond)
		select {
		case result := <-done:
			unblock()
			t.Fatalf("public stream ended at preparation bound: %+v", result)
		default:
		}
		unblock()
		result := <-done
		if writeErr := <-written; writeErr != nil {
			t.Fatal(writeErr)
		}
		if result.err != nil || !result.result.Complete || request.cancelled || !attempt.terminal.Delivered || attempt.terminal.Accounting != publicAccountingRecorded || !sink.result().DeliveryComplete || w.body.String() != f.response {
			t.Fatalf("public accepted stream lost live context: err=%v complete=%v cancelled=%v terminal=%+v sink=%+v body=%q", result.err, result.result.Complete, request.cancelled, attempt.terminal, sink.result(), w.body.String())
		}
	})

	t.Run("same owned sink replaces external emitter exactly once", func(t *testing.T) {
		f := newChatAccountFixture(t)
		w := writer()
		request, attempt, sink := attach(t, f, w)
		emissions := 0
		f.emit = func([]byte) error { emissions++; return errors.New("external emitter must be unused") }
		result, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard))
		delivery := sink.result()
		if err != nil || !result.Complete || !request.committed || !attempt.committed || emissions != 0 {
			t.Fatalf("owned attempt: result=%+v err=%v committed=%v/%v external=%d", result, err, request.committed, attempt.committed, emissions)
		}
		if w.body.String() != f.response || bytes.Count(w.body.Bytes(), []byte("[DONE]")) != 1 || len(w.statuses) != 1 || w.statuses[0] != http.StatusOK || w.Header().Get("Content-Type") != "text/event-stream" {
			t.Fatalf("owned output: statuses=%v body=%q type=%q", w.statuses, w.body.String(), w.Header().Get("Content-Type"))
		}
		if !delivery.ProtocolComplete || !delivery.DeliveryComplete || delivery.Genuine == 0 || delivery.Synthetic != int64(len("data: [DONE]\n\n")) || !attempt.terminal.Delivered || attempt.terminal.Accounting != publicAccountingRecorded {
			t.Fatalf("owned facts: sink=%+v terminal=%+v", delivery, attempt.terminal)
		}
		if _, err := request.newAttempt(); !errors.Is(err, errRoutingSpend) {
			t.Fatalf("completed request admitted fallback: %v", err)
		}
	})

	t.Run("owned sink permits no external emitter", func(t *testing.T) {
		f := newChatAccountFixture(t)
		w := writer()
		_, attempt, sink := attach(t, f, w)
		f.emit = nil
		if result, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard)); err != nil || !result.Complete || !sink.result().DeliveryComplete || w.body.String() != f.response {
			t.Fatalf("nil external emitter blocked owned sink: result=%+v err=%v sink=%+v body=%q", result, err, sink.result(), w.body.String())
		}
	})

	t.Run("nonstream success requires accepted clean EOF and delivery", func(t *testing.T) {
		f := newChatAccountFixture(t)
		w := writer()
		request, attempt, sink := attach(t, f, w)
		f.body = chatSimple
		f.response = chatResponse("synthetic-v1", `{"role":"assistant","content":"ok"}`, `"stop"`, chatUsageOK, false)
		f.emit = nil
		result, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard))
		if err != nil || !result.Complete || !attempt.accepted || !request.committed || !sink.result().ProtocolComplete || !sink.result().DeliveryComplete || !attempt.terminal.Delivered || w.body.String() != f.response || w.Header().Get("Content-Type") != "application/json" {
			t.Fatalf("nonstream completion: result=%+v err=%v accepted=%v sink=%+v terminal=%+v body=%q", result, err, attempt.accepted, sink.result(), attempt.terminal, w.body.String())
		}
	})

	t.Run("nonstream read failure withholds body and emits one JSON error", func(t *testing.T) {
		f := newChatAccountFixture(t)
		w := writer()
		request, attempt, sink := attach(t, f, w)
		f.body = chatSimple
		f.response = chatResponse("synthetic-v1", `{"role":"assistant","content":"ok"}`, `"stop"`, chatUsageOK, false)
		roundTrip := f.transport.client.Transport
		f.transport.client.Transport = publicTestRoundTrip(func(req *http.Request) (*http.Response, error) {
			response, err := roundTrip.RoundTrip(req)
			if err != nil {
				return response, err
			}
			response.Body = &publicSinkReadHook{ReadCloser: response.Body, eofError: io.ErrUnexpectedEOF}
			return response, nil
		})
		f.emit = nil
		result, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard))
		want := "{\"error\":\"stream interrupted: malformed upstream stream\"}\n"
		if err == nil || result.Complete || !request.committed || !attempt.committed || w.body.String() != want || !sink.result().DeliveryComplete || sink.result().ProtocolComplete || !attempt.terminal.Delivered || w.Header().Get("Content-Type") != "application/json" {
			t.Fatalf("nonstream EOF failure: result=%+v err=%v sink=%+v terminal=%+v body=%q", result, err, sink.result(), attempt.terminal, w.body.String())
		}
	})

	t.Run("trailing data replaces withheld DONE with one clean error", func(t *testing.T) {
		f := newChatAccountFixture(t)
		w := writer()
		request, attempt, sink := attach(t, f, w)
		prefix := strings.TrimSuffix(f.response, "data: [DONE]\n\n")
		f.response += "data: {}\n\n"
		sink.WriteHeader(http.StatusAccepted)
		external := 0
		f.emit = func([]byte) error { external++; return nil }
		result, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard))
		want := prefix + string(wantSSEInterruption(streamInterruptReasonMalformed))
		if err == nil || result.Complete || !attempt.committed || !request.committed || external != 0 || w.body.String() != want || bytes.Count(w.body.Bytes(), []byte("[DONE]")) != 1 {
			t.Fatalf("trailing public response: result=%+v err=%v committed=%v/%v external=%d body=%q", result, err, attempt.committed, request.committed, external, w.body.String())
		}
		if got := sink.result(); got.ProtocolComplete || !got.DeliveryComplete || got.Reason != streamInterruptReasonMalformed || !got.ContinuationUnavailable || !attempt.terminal.Delivered || attempt.terminal.ProtocolComplete {
			t.Fatalf("interruption facts: sink=%+v terminal=%+v", got, attempt.terminal)
		}
		if len(w.statuses) != 1 || w.statuses[0] != http.StatusAccepted {
			t.Fatalf("committed status rewritten: %v", w.statuses)
		}
		if err := sink.interrupt(streamProtocolSSE, streamInterruptReasonBudget); err != nil {
			t.Fatal(err)
		}
		if w.body.String() != want {
			t.Fatal("second synthetic ending emitted")
		}
		if _, err := request.newAttempt(); !errors.Is(err, errRoutingSpend) {
			t.Fatalf("committed malformed response admitted fallback: %v", err)
		}
	})

	t.Run("postcommit parser allocation refusal uses the sink once", func(t *testing.T) {
		f := newChatAccountFixture(t)
		w := writer()
		request, attempt, sink := attach(t, f, w)
		roundTrip := f.transport.client.Transport
		f.transport.client.Transport = publicTestRoundTrip(func(req *http.Request) (*http.Response, error) {
			response, err := roundTrip.RoundTrip(req)
			if err != nil {
				return response, err
			}
			response.Body = &publicSinkReadHook{ReadCloser: response.Body, hook: func() { f.budget.ceiling = f.budget.used.Load() }}
			return response, nil
		})
		f.emit = nil
		result, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard))
		if !errors.Is(err, errChatBudget) || result.Complete || !attempt.committed || !request.committed || f.calls.Load() != 1 || !bytes.Equal(w.body.Bytes(), wantSSEInterruption(streamInterruptReasonBudget)) {
			t.Fatalf("allocation refusal: result=%+v err=%v committed=%v/%v Do=%d body=%q", result, err, attempt.committed, request.committed, f.calls.Load(), w.body.String())
		}
		if got := sink.result(); !got.DeliveryComplete || got.ProtocolComplete || got.Genuine != 0 || got.Reason != streamInterruptReasonBudget || !attempt.terminal.Delivered {
			t.Fatalf("budget ending facts: sink=%+v terminal=%+v", got, attempt.terminal)
		}
		if _, err := request.newAttempt(); !errors.Is(err, errRoutingSpend) {
			t.Fatalf("allocation refusal admitted fallback: %v", err)
		}
	})

	t.Run("delivered success survives independent accounting write failure", func(t *testing.T) {
		f := newChatAccountFixture(t)
		w := &publicSinkFlushHook{responseSinkWriter: writer()}
		request, attempt, sink := attach(t, f, w)
		accountingWrites := 0
		w.hook = func() {
			f.ledger.write = func(string, []byte, os.FileMode) error {
				accountingWrites++
				return errors.New("injected accounting failure")
			}
		}
		f.emit = func([]byte) error { t.Error("owned output called external emitter"); return nil }
		result, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard))
		if !errors.Is(err, errChatAccounting) || !result.Complete || !request.committed || !attempt.terminal.Delivered || !attempt.terminal.ProtocolComplete || attempt.terminal.Accounting != publicAccountingFailed || accountingWrites != 3 {
			t.Fatalf("accounting rewrote delivery: result=%+v err=%v terminal=%+v accounting writes=%d", result, err, attempt.terminal, accountingWrites)
		}
		if got := sink.result(); !got.ProtocolComplete || !got.DeliveryComplete || got.WriteError != nil || w.body.String() != f.response || bytes.Count(w.body.Bytes(), []byte("[DONE]")) != 1 {
			t.Fatalf("delivered success lost: sink=%+v body=%q", got, w.body.String())
		}
		if _, err := request.newAttempt(); !errors.Is(err, errRoutingSpend) {
			t.Fatalf("accounting failure admitted resend: %v", err)
		}
	})

	for _, scenario := range []string{"partial write", "flush failure", "request cancellation"} {
		t.Run(scenario+" cancels without synthetic follow-up", func(t *testing.T) {
			f := newChatAccountFixture(t)
			w := &publicSinkFlushHook{responseSinkWriter: writer()}
			request, attempt, sink := attach(t, f, w)
			first := chatEvent(chatChunk(`{"role":"assistant"}`, `null`))
			want := first
			switch scenario {
			case "partial write":
				w.short = true
				want = first[:len(first)-1]
			case "flush failure":
				w.flushErr = io.ErrClosedPipe
			case "request cancellation":
				w.hook = func() { request.cancelRequest() }
			}
			f.emit = nil
			result, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard))
			if err == nil || result.Complete || !request.cancelled || !attempt.committed || w.body.String() != want || attempt.terminal.Delivered || sink.result().DeliveryComplete {
				t.Fatalf("client cancellation: result=%+v err=%v cancelled=%v terminal=%+v sink=%+v body=%q", result, err, request.cancelled, attempt.terminal, sink.result(), w.body.String())
			}
			if bytes.Contains(w.body.Bytes(), []byte("[DONE]")) || bytes.Contains(w.body.Bytes(), []byte("stream_interrupted")) || len(w.statuses) != 1 {
				t.Fatalf("synthetic follow-up after cancellation: status=%v body=%q", w.statuses, w.body.String())
			}
			if _, err := request.newAttempt(); !errors.Is(err, errRoutingSpend) {
				t.Fatalf("cancelled request admitted fallback: %v", err)
			}
		})
	}

	for _, scenario := range []string{"cancel at final acceptance", "G refusal at final acceptance"} {
		t.Run(scenario+" retains unsent liability without sink output", func(t *testing.T) {
			p, f, _, snapshot := egressFixture(t)
			egressGrant(t, p, 1, 6*time.Second)
			w := writer()
			request, attempt, sink := attach(t, f, w)
			guard := func(ctx context.Context, accept func() error) error {
				if accept != nil {
					if scenario == "cancel at final acceptance" {
						request.cancelRequest()
					} else {
						egressRevoke(t, p, 2)
					}
				}
				return p.publicPermissionAccept("weak", snapshot, accept)
			}
			f.emit = nil
			result, err := runChatAccountAttempt(f.ctx, snapshot, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, guard, attempt, f.emit)
			if !errors.Is(err, errPublicFinalGuard) || result.Complete || attempt.accepted || attempt.committed || request.committed || f.calls.Load() != 0 || w.body.Len() != 0 || len(w.statuses) != 0 {
				t.Fatalf("unsent attempt affected sink: result=%+v err=%v accepted=%v committed=%v/%v Do=%d statuses=%v body=%q", result, err, attempt.accepted, attempt.committed, request.committed, f.calls.Load(), w.statuses, w.body.String())
			}
			if scenario == "cancel at final acceptance" {
				if !errors.Is(err, errPublicCancelledBeforeSend) || errors.Is(err, errPublicPermission) {
					t.Fatalf("cancellation lost its distinct final refusal: %v", err)
				}
			} else if !errors.Is(err, errPublicPermission) || errors.Is(err, errPublicCancelledBeforeSend) {
				t.Fatalf("G refusal collapsed into cancellation: %v", err)
			}
			if attempt.reservation.Durability != accountReserveDurable || attempt.reservation.Hold.Attempt == "" || request.granted == 0 || attempt.terminal.Accounting != publicAccountingRetained || attempt.terminal.Delivered || sink.result().DeliveryComplete || sink.result().Reason != "" {
				t.Fatalf("unsent liability/sink: reservation=%+v terminal=%+v sink=%+v granted=%d", attempt.reservation, attempt.terminal, sink.result(), request.granted)
			}
			egressUnsentHold(t, f)
		})
	}
}

type publicSinkReadHook struct {
	io.ReadCloser
	hook     func()
	eofError error
}

func (b *publicSinkReadHook) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if n > 0 && b.hook != nil {
		hook := b.hook
		b.hook = nil
		hook()
	}
	if err == io.EOF && b.eofError != nil {
		err = b.eofError
	}
	return n, err
}

type publicSinkFlushHook struct {
	*responseSinkWriter
	hook func()
}

func (w *publicSinkFlushHook) FlushError() error {
	if err := w.responseSinkWriter.FlushError(); err != nil {
		return err
	}
	if w.hook != nil {
		w.hook()
	}
	return nil
}

func TestPublicAttemptPreparationDeadline(t *testing.T) {
	for _, stage := range []string{"initial preparation", "parser allowance", "transport headroom"} {
		t.Run(stage, func(t *testing.T) {
			f := newChatAccountFixture(t)
			ctx, root, buffers := chatTestInferenceContext(t, 64<<20, time.Hour)
			ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
			defer cancel()
			defer buffers.close()
			f.ctx = ctx
			if stage == "parser allowance" {
				f.body = strings.TrimSuffix(chatSimple, "}") + `,"stream":true}`
			}
			request := newRoutingRequest(nil)
			request.deadline = time.Now().Add(250 * time.Millisecond)
			defer request.finish()
			attempt := spendAttempt(t, request)
			var lookups, guards int
			f.resolve = func(nodepolicy.CredentialRef) (string, bool) { lookups++; return "synthetic", true }
			guard := func(context.Context, func() error) error { guards++; return nil }
			held := make(chan *bufferLease, 1)
			if stage == "initial preparation" {
				held <- bufferMustAcquire(t, root, 64<<20)
			}
			fixed := 0
			budget := func(n int64) (func(), error) {
				if n == chatFixedStorage+256 {
					fixed++
					if fixed == 2 && stage != "initial preparation" {
						root.mu.Lock()
						remaining := root.capacity - root.used
						root.mu.Unlock()
						held <- bufferMustAcquire(t, root, remaining)
					}
				}
				return f.budget.acquire(n)
			}
			snapshot, err := f.ledger.prepareChatAccount("chat-a")
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() {
				_, err := runChatAccountAttempt(f.ctx, snapshot, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, budget, guard, attempt, f.emit)
				done <- err
			}()
			var hold *bufferLease
			select {
			case hold = <-held:
			case <-ctx.Done():
				t.Fatal("public fixture never reached the requested pressure stage")
			}
			defer hold.close()
			waitForCond(t, time.Second, "public preparation FIFO wait", func() bool { root.mu.Lock(); defer root.mu.Unlock(); return root.waiters.Len() == 1 })
			root.mu.Lock()
			waitDeadline := root.waiters.Front().Value.(*bufferWaiter).deadline
			root.mu.Unlock()
			if !waitDeadline.Equal(request.deadline) {
				t.Errorf("public wait deadline=%v, want same immutable %v", waitDeadline, request.deadline)
			}
			select {
			case err = <-done:
			case <-time.After(time.Second):
				cancel()
				err = <-done
				t.Error("public preparation exceeded request deadline")
			}
			if !errors.Is(err, errChatBudget) || bufferRequestStatus(err) != http.StatusServiceUnavailable || lookups != 0 || guards != 0 || f.calls.Load() != 0 || len(f.ledger.state.Holds) != 0 || request.active != nil || attempt.accepted || attempt.committed || f.budget.used.Load() != 0 {
				t.Fatalf("pre-send deadline effects: err=%v lookups=%d guards=%d Do=%d holds=%d active=%v accepted=%v committed=%v explicit=%d", err, lookups, guards, f.calls.Load(), len(f.ledger.state.Holds), request.active != nil, attempt.accepted, attempt.committed, f.budget.used.Load())
			}
			if _, err := request.newAttempt(); !errors.Is(err, errRoutingSpend) {
				t.Fatalf("expired public owner admitted another attempt: %v", err)
			}
			hold.close()
			buffers.close()
			bufferAssertTotals(t, root, 0, 0, 0)
		})
	}
}

func TestPublicAttemptDeadlineSpansAttempts(t *testing.T) {
	f := newChatAccountFixture(t)
	ctx, root, buffers := chatTestInferenceContext(t, 64<<20, time.Hour)
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	defer buffers.close()
	f.ctx = ctx
	request := newRoutingRequest(nil)
	request.deadline = time.Now().Add(400 * time.Millisecond)
	defer request.finish()
	resolve := f.resolve
	f.resolve = func(nodepolicy.CredentialRef) (string, bool) { return "", false }
	if _, err := lifecycleRun(t, f, spendAttempt(t, request), chatTestGuard(f.guard)); !errors.Is(err, errChatCredential) {
		t.Fatalf("first unsent attempt: %v", err)
	}
	if request.cancelled || request.active != nil || len(f.ledger.state.Holds) != 0 {
		t.Fatal("unsent refusal changed request authority or reserved money")
	}
	f.resolve = resolve
	entered := make(chan struct{})
	f.transport.client.Transport = publicTestRoundTrip(func(req *http.Request) (*http.Response, error) {
		f.calls.Add(1)
		close(entered)
		<-req.Context().Done()
		return nil, req.Context().Err()
	})
	attempt := spendAttempt(t, request)
	done := make(chan error, 1)
	go func() { _, err := lifecycleRun(t, f, attempt, chatTestGuard(f.guard)); done <- err }()
	waitChan(t, entered, "second public attempt headers wait")
	var err error
	select {
	case err = <-done:
	case <-time.After(time.Second):
		cancel()
		err = <-done
		t.Error("first attempt cleanup disarmed the shared request deadline")
	}
	if err == nil || !request.cancelled || attempt.committed || !attempt.accepted || f.calls.Load() != 1 || request.active != nil {
		t.Fatalf("public silent transport deadline: err=%v cancelled=%v committed=%v accepted=%v Do=%d active=%v", err, request.cancelled, attempt.committed, attempt.accepted, f.calls.Load(), request.active != nil)
	}
	if _, err := request.newAttempt(); !errors.Is(err, errRoutingSpend) {
		t.Fatalf("deadline admitted a later public attempt: %v", err)
	}
	buffers.close()
	bufferAssertTotals(t, root, 0, 0, 0)
}

func TestPublicAttemptDeadlineAtFinalAcceptance(t *testing.T) {
	p, f, _, snapshot := egressFixture(t)
	egressGrant(t, p, 1, 6*time.Second)
	owner := newRoutingRequest(nil)
	owner.deadline = time.Now().Add(150 * time.Millisecond)
	defer owner.finish()
	attempt := spendAttempt(t, owner)
	guard := func(ctx context.Context, accept func() error) error {
		if accept == nil {
			if err := p.publicPermissionAccept("weak", snapshot, nil); err != nil {
				return err
			}
			owner.mu.Lock()
			stop := owner.stopPreOutput
			owner.mu.Unlock()
			if stop == nil || !stop() {
				t.Fatal("deadline callback could not be delayed")
			}
			<-time.After(time.Until(owner.deadline) + 25*time.Millisecond)
			return nil
		}
		return p.publicPermissionAccept("weak", snapshot, accept)
	}
	result, err := runChatAccountAttempt(f.ctx, snapshot, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, guard, attempt, f.emit)
	if !errors.Is(err, errPublicFinalGuard) || !errors.Is(err, errPublicCancelledBeforeSend) || errors.Is(err, errPublicPermission) || f.calls.Load() != 0 || attempt.accepted || attempt.committed || result.Exposed || result.Complete || !owner.cancelled || owner.reason != routingDeadlineReason {
		t.Fatalf("late final acceptance crossed send boundary: err=%v Do=%d accepted=%v committed=%v result=%+v cancelled=%v reason=%q", err, f.calls.Load(), attempt.accepted, attempt.committed, result, owner.cancelled, owner.reason)
	}
	if attempt.reservation.Durability != accountReserveDurable || attempt.terminal.Accounting != publicAccountingRetained || attempt.terminal.Delivered || owner.active != nil || f.budget.used.Load() != 0 {
		t.Fatalf("deadline erased liability/retained attempt: reservation=%+v terminal=%+v active=%v bytes=%d", attempt.reservation, attempt.terminal, owner.active != nil, f.budget.used.Load())
	}
	egressUnsentHold(t, f)
}
