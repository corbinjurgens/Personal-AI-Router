// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"math"
	"net/http"
	"net/netip"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
