// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"net/netip"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

// Synthetic-only integration fixture: a RoundTripper cannot resolve DNS or
// reach a provider. Each call verifies a confirmed on-disk hold before replying.
type chatAccountFixture struct {
	ledger         *accountLedger
	clock          *ledgerClock
	routing        nodepolicy.Routing
	profile        *chatQualification
	transport      *publicClient
	budget         *publicTestBudget
	ctx            context.Context
	guard          func(context.Context) error
	resolve        func(nodepolicy.CredentialRef) (string, bool)
	emit           func([]byte) error
	body, response string
	status         int
	retry          string
	calls          atomic.Int64
}

func newChatAccountFixture(t *testing.T) *chatAccountFixture {
	t.Helper()
	q, _ := chatFixture()
	c := newLedgerClock()
	q.Now = c.now
	r := testRouting("1000000", "10000000", "10000000")
	for i := range r.Deployments {
		d := &r.Deployments[i]
		d.Endpoint, d.Model = q.Endpoint, q.Model
		d.Qualification = &q.Ref
		d.ContextTokens, d.MaxOutputTokens = int(q.ContextTokens), int(q.MaxOutputTokens)
		d.Transport.MaxResponseBytes, d.Transport.MaxEventBytes = q.MaxBodyBytes, q.MaxEventBytes
	}
	l := openTestLedger(t, t.TempDir(), c, r)
	client, err := newPublicClient(q.Endpoint, *r.Deployments[0].Transport)
	if err != nil {
		t.Fatal(err)
	}
	f := &chatAccountFixture{ledger: l, clock: c, routing: r, profile: q, transport: client,
		budget: &publicTestBudget{ceiling: 64 << 20}, ctx: context.Background(), body: chatSimple,
		guard: func(context.Context) error { return nil }, emit: func([]byte) error { return nil },
		response: chatResponse("synthetic-v1", `{"role":"assistant","content":"ok"}`, `"stop"`, chatUsageOK, false), status: http.StatusOK}
	client.resolve = func(context.Context, string) ([]netip.Addr, error) {
		return []netip.Addr{netip.MustParseAddr("8.8.8.8")}, nil
	}
	f.resolve = func(ref nodepolicy.CredentialRef) (string, bool) {
		if ref != testCredential {
			t.Errorf("wrong synthetic reference selected")
		}
		return "synthetic-selected-key", true
	}
	client.client.Transport = publicTestRoundTrip(func(req *http.Request) (*http.Response, error) {
		f.calls.Add(1)
		state, err := readAccountStore(l.storePath)
		if err != nil {
			t.Fatal(err)
		}
		if len(state.Holds) == 0 || state.Holds[len(state.Holds)-1].State != holdReserved {
			t.Fatal("provider called without a durable reservation")
		}
		if req.URL.String() != q.Endpoint || req.Header.Get("Authorization") != "Bearer synthetic-selected-key" {
			t.Error("endpoint or selected synthetic key changed")
		}
		return &http.Response{StatusCode: f.status, Header: http.Header{"Retry-After": {f.retry}},
			Body: io.NopCloser(strings.NewReader(f.response)), ContentLength: -1}, nil
	})
	t.Cleanup(func() {
		if n := f.budget.used.Load(); n != 0 {
			t.Errorf("leaked bytes: %d", n)
		}
	})
	return f
}

func (f *chatAccountFixture) run(t *testing.T) (chatParseResult, error) {
	t.Helper()
	s, err := f.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	return runChatAccountAttempt(f.ctx, s, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, chatTestGuard(f.guard), chatTestLifecycle(t), f.emit)
}

func chatTestLifecycle(t *testing.T) *publicAttemptLifecycle {
	t.Helper()
	a, err := (&routingSpend{}).newAttempt()
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func chatTestGuard(check func(context.Context) error) publicAttemptGuard {
	if check == nil {
		return nil
	}
	return func(ctx context.Context, accept func() error) error {
		if err := check(ctx); err != nil {
			return err
		}
		if accept != nil {
			return accept()
		}
		return nil
	}
}

func TestChatAccountAttemptPreSend(t *testing.T) {
	tests := []struct {
		name   string
		change func(*chatAccountFixture)
	}{
		{"missing guard", func(f *chatAccountFixture) { f.guard = nil }},
		{"missing profile", func(f *chatAccountFixture) { f.profile = nil }},
		{"expired evidence", func(f *chatAccountFixture) { f.clock.advance(2 * time.Hour) }},
		{"invalid endpoint", func(f *chatAccountFixture) { f.transport.endpoint += "?secret" }},
		{"invalid request", func(f *chatAccountFixture) { f.body = `{"store":true}` }},
		{"budget refusal", func(f *chatAccountFixture) { f.budget.ceiling = 1 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			tt.change(f)
			lookups := 0
			f.resolve = func(nodepolicy.CredentialRef) (string, bool) { lookups++; return "synthetic", true }
			if _, err := f.run(t); err == nil {
				t.Fatal("pre-send refusal accepted")
			}
			if lookups != 0 || f.calls.Load() != 0 || len(f.ledger.state.Holds) != 0 {
				t.Fatal("pre-send refusal crossed lookup/reserve/send boundary")
			}
		})
	}
	t.Run("missing budget", func(t *testing.T) {
		f := newChatAccountFixture(t)
		s, err := f.ledger.prepareChatAccount("chat-a")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := runChatAccountAttempt(f.ctx, s, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, nil, chatTestGuard(f.guard), chatTestLifecycle(t), f.emit); err == nil {
			t.Fatal("missing budget accepted")
		}
		if f.calls.Load() != 0 || len(f.ledger.state.Holds) != 0 {
			t.Fatal("missing budget crossed reservation boundary")
		}
	})
}

func TestChatAccountAttemptGuardAndCancellation(t *testing.T) {
	for _, name := range []string{"guard refusal", "cancel before lookup", "cancel after reserve", "evidence expires in guard"} {
		t.Run(name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			ctx, cancel := context.WithCancel(f.ctx)
			defer cancel()
			f.ctx = ctx
			f.guard = func(context.Context) error {
				switch name {
				case "guard refusal":
					return errors.New("synthetic-key-body-sentinel")
				case "cancel after reserve":
					cancel()
				case "evidence expires in guard":
					f.clock.advance(2 * time.Hour)
				}
				return nil
			}
			if name == "cancel before lookup" {
				cancel()
			}
			_, err := f.run(t)
			if err == nil || f.calls.Load() != 0 {
				t.Fatal("guard/cancellation allowed a send")
			}
			if strings.Contains(err.Error(), "sentinel") {
				t.Fatal("guard diagnostic leaked")
			}
			if name != "cancel before lookup" && (len(f.ledger.state.Holds) != 1 || !f.ledger.state.Holds[0].outstanding()) {
				t.Fatal("guard failure released liability")
			}
		})
	}
}

func TestChatAccountAttemptPinnedSnapshot(t *testing.T) {
	tests := []struct {
		name string
		edit func(*nodepolicy.Deployment, *nodepolicy.Account)
	}{
		{"endpoint", func(d *nodepolicy.Deployment, _ *nodepolicy.Account) {
			d.Endpoint = "https://other.invalid/v1/chat/completions"
		}},
		{"model", func(d *nodepolicy.Deployment, _ *nodepolicy.Account) { d.Model = "changed" }},
		{"reference", func(_ *nodepolicy.Deployment, a *nodepolicy.Account) { a.Credential = otherCredential }},
		{"price", func(d *nodepolicy.Deployment, _ *nodepolicy.Account) { d.Prices.InputPerMillion = "3000000" }},
		{"qualification", func(d *nodepolicy.Deployment, _ *nodepolicy.Account) {
			ref := *d.Qualification
			ref.Revision = "2"
			d.Qualification = &ref
		}},
		{"transport", func(d *nodepolicy.Deployment, _ *nodepolicy.Account) { d.Transport.IdleTimeoutSeconds++ }},
		{"bounds", func(d *nodepolicy.Deployment, _ *nodepolicy.Account) { d.ContextTokens++ }},
		{"ceiling", func(_ *nodepolicy.Deployment, a *nodepolicy.Account) { a.DailyCeiling = "20000000" }},
		{"revision", func(*nodepolicy.Deployment, *nodepolicy.Account) {}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			s, err := f.ledger.prepareChatAccount("chat-a")
			if err != nil {
				t.Fatal(err)
			}
			original := s.deployment.Endpoint
			tt.edit(&f.routing.Deployments[0], &f.routing.Accounts[0])
			revision := "7"
			if tt.name == "revision" {
				revision = "8"
			}
			if err := f.ledger.applyRouting(f.routing, revision); err != nil {
				t.Fatal(err)
			}
			if s.deployment.Endpoint != original {
				t.Fatal("snapshot aliases mutable routing")
			}
			_, err = runChatAccountAttempt(f.ctx, s, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, chatTestGuard(f.guard), chatTestLifecycle(t), f.emit)
			if err == nil || f.calls.Load() != 0 || len(f.ledger.state.Holds) != 0 {
				t.Fatal("stale snapshot allowed provider request")
			}
		})
	}
}

func TestChatAccountAttemptStoreRefusals(t *testing.T) {
	for _, name := range []string{"write failure", "replaced", "unavailable", "locked", "corrupt"} {
		t.Run(name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			switch name {
			case "write failure":
				f.ledger.write = func(string, []byte, os.FileMode) error { return errors.New("synthetic-secret-body-sentinel") }
			case "replaced":
				f.ledger.write = func(path string, data []byte, mode os.FileMode) error {
					if err := durablefile.WriteFile(path, data, mode); err != nil {
						return err
					}
					return durablefile.ErrReplaced
				}
			case "unavailable":
				f.ledger.close()
				f.ledger.storePath = ""
			case "locked":
				other := newAccountLedger()
				other.setDir(filepath.Dir(f.ledger.storePath), nil)
				other.routing, other.policyRevision = f.routing, "7"
				t.Cleanup(other.close)
				f.ledger = other
			case "corrupt":
				f.ledger.close()
				if err := os.WriteFile(f.ledger.storePath, []byte("corrupt"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			_, err := f.run(t)
			if err == nil || f.calls.Load() != 0 {
				t.Fatal("unconfirmed store allowed provider request")
			}
			if strings.Contains(err.Error(), "sentinel") {
				t.Fatal("ledger failure leaked diagnostic")
			}
			if name == "replaced" && (len(f.ledger.state.Holds) != 1 || f.ledger.state.Holds[0].State != holdUncertain) {
				t.Fatal("uncertain landed hold lost")
			}
		})
	}
}

func TestChatAccountAttemptClassification(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		retry    string
		outcome  accountOutcome
		cooldown time.Duration
	}{
		{"unauthorized", http.StatusUnauthorized, "", outcomeCredentialFailure, 0},
		{"forbidden", http.StatusForbidden, "", outcomeCredentialFailure, 0},
		{"bad request", http.StatusBadRequest, "", outcomeTransient, 0},
		{"not found", http.StatusNotFound, "", outcomeTransient, 0},
		{"unprocessable", http.StatusUnprocessableEntity, "", outcomeTransient, 0},
		{"rate limit seconds", http.StatusTooManyRequests, "5", outcomeRateLimited, 5 * time.Second},
		{"rate limit date", http.StatusTooManyRequests, "Mon, 19 Oct 2026 12:00:10 GMT", outcomeRateLimited, 10 * time.Second},
		{"rate limit overflow", http.StatusTooManyRequests, "9223372036854775807", outcomeRateLimited, defaultCooldown},
		{"rate limit signed", http.StatusTooManyRequests, "+5", outcomeRateLimited, defaultCooldown},
		{"rate limit date overflow", http.StatusTooManyRequests, "Fri, 31 Dec 9999 23:59:59 GMT", outcomeRateLimited, defaultCooldown},
		{"rate limit invalid", http.StatusTooManyRequests, "synthetic-body-key-sentinel", outcomeRateLimited, defaultCooldown},
		{"rate limit past", http.StatusTooManyRequests, "Mon, 19 Oct 2026 11:59:59 GMT", outcomeRateLimited, defaultCooldown},
		{"rate limit clamp", http.StatusTooManyRequests, "999999", outcomeRateLimited, maxCooldown},
		{"request timeout", http.StatusRequestTimeout, "", outcomeTransient, 0},
		{"server failure", http.StatusInternalServerError, "", outcomeTransient, 0},
		{"unsupported status", http.StatusCreated, "", outcomeTransient, 0},
		{"redirect", http.StatusTemporaryRedirect, "", outcomeTransient, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			f.status, f.retry, f.response = tt.status, tt.retry, "synthetic-body-key-sentinel"
			_, err := f.run(t)
			if err == nil || strings.Contains(err.Error(), "sentinel") {
				t.Fatal("status classification exposed provider material or succeeded")
			}
			h, a := f.ledger.state.Holds[0], f.ledger.state.Accounts[0]
			if h.Outcome != tt.outcome || !h.QualificationChecked || !h.outstanding() {
				t.Fatalf("wrong closed classification: %+v", h)
			}
			if tt.cooldown > 0 && (a.Cooldown == nil || a.Cooldown.Reason != cooldownRateLimited || a.Cooldown.UntilMs != f.clock.now().Add(tt.cooldown).UnixMilli()) {
				t.Fatal("generic rate limit lost bounded cooldown or became exhaustion")
			}
			if tt.cooldown == 0 && a.Cooldown != nil {
				t.Fatal("invented cooldown")
			}
			if (a.CredentialFailure != nil) != (tt.outcome == outcomeCredentialFailure) {
				t.Fatal("guessed credential failure")
			}
			if tt.outcome == outcomeCredentialFailure || tt.cooldown > 0 {
				if _, err := f.run(t); err == nil || f.calls.Load() != 1 {
					t.Fatal("restriction did not suppress next send")
				}
			}
		})
	}
}

func TestChatAccountAttemptRestrictiveStatusSurvivesBodyFailure(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusTooManyRequests} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			for _, mode := range []string{"normal", "declared oversized", "chunked oversized", "interrupted", "encoding refused"} {
				t.Run(mode, func(t *testing.T) {
					f := newChatAccountFixture(t)
					body := &publicErrorTestBody{Reader: strings.NewReader("synthetic-body-key-sentinel")}
					length := int64(-1)
					header := http.Header{"Retry-After": {"5"}}
					switch mode {
					case "declared oversized", "chunked oversized":
						body.Reader = strings.NewReader(strings.Repeat("x", int(f.transport.bounds.MaxErrorBodyBytes)+1))
						if mode == "declared oversized" {
							length = f.transport.bounds.MaxErrorBodyBytes + 1
						}
					case "interrupted":
						body.interrupted = true
					case "encoding refused":
						header.Set("Content-Encoding", "gzip")
					}
					f.transport.client.Transport = publicTestRoundTrip(func(*http.Request) (*http.Response, error) {
						f.calls.Add(1)
						return &http.Response{StatusCode: status, Header: header, Body: body, ContentLength: length}, nil
					})
					f.emit = func([]byte) error {
						t.Error("restrictive response emitted provider data")
						return nil
					}
					wantErr, wantOutcome := errChatCredential, outcomeCredentialFailure
					if status == http.StatusTooManyRequests {
						wantErr, wantOutcome = errChatRateLimit, outcomeRateLimited
					}
					result, err := f.run(t)
					if !errors.Is(err, wantErr) || result.Complete || result.Usage != nil {
						t.Errorf("first attempt: complete=%v usage=%v err=%v; want %v", result.Complete, result.Usage, err, wantErr)
					}
					if _, err := f.run(t); !errors.Is(err, errChatAccounting) {
						t.Errorf("next attempt error=%v; want accounting refusal", err)
					}
					if calls := f.calls.Load(); calls != 1 {
						t.Errorf("restrictive HTTP status lost: sends=%d; want 1", calls)
					}
					state, err := readAccountStore(f.ledger.storePath)
					if err != nil {
						t.Fatal(err)
					}
					if len(state.Holds) != 1 {
						t.Fatalf("durable holds=%d; want 1", len(state.Holds))
					}
					h, a := state.Holds[0], state.Accounts[0]
					if h.Outcome != wantOutcome || !h.QualificationChecked || !h.outstanding() || h.Actual != "" {
						t.Errorf("wrong durable classification: %+v", h)
					}
					if status == http.StatusTooManyRequests {
						if a.CredentialFailure != nil || a.Cooldown == nil || a.Cooldown.Reason != cooldownRateLimited || a.Cooldown.UntilMs != f.clock.now().Add(5*time.Second).UnixMilli() {
							t.Error("rate-limit restriction lost trusted Retry-After")
						}
					} else if a.Cooldown != nil || a.CredentialFailure == nil || a.CredentialFailure.Credential != testCredential {
						t.Error("credential suppression lost")
					}
					for _, spent := range a.Spent {
						if spent != "0" {
							t.Errorf("error response reconciled spend: %s", spent)
						}
					}
					if closes := body.closes.Load(); closes != 1 {
						t.Errorf("body closes=%d; want 1", closes)
					}
				})
			}
		})
	}
}

func TestChatAccountAttemptCredentialBoundary(t *testing.T) {
	for _, key := range []string{"", "synthetic\rkey", "synthetic\nkey"} {
		t.Run(strconv.Quote(key), func(t *testing.T) {
			f := newChatAccountFixture(t)
			f.resolve = func(ref nodepolicy.CredentialRef) (string, bool) {
				if ref != testCredential {
					t.Fatal("wrong reference")
				}
				return key, true
			}
			if _, err := f.run(t); !errors.Is(err, errChatCredential) {
				t.Fatal("unsafe synthetic key accepted")
			}
			if f.calls.Load() != 0 || len(f.ledger.state.Holds) != 0 {
				t.Fatal("unsafe credential reached reserve/send")
			}
		})
	}
	t.Run("lookup edits selected identity", func(t *testing.T) {
		f := newChatAccountFixture(t)
		f.resolve = func(ref nodepolicy.CredentialRef) (string, bool) {
			if ref != testCredential {
				t.Fatal("wrong selected reference")
			}
			f.routing.Accounts[0].Credential = otherCredential
			f.routing.Deployments[0].Endpoint = "https://changed.invalid/v1/chat/completions"
			if err := f.ledger.applyRouting(f.routing, "8"); err != nil {
				t.Fatal(err)
			}
			return "synthetic-selected-key", true
		}
		if _, err := f.run(t); err == nil || f.calls.Load() != 0 {
			t.Fatal("lookup/config race sent stale key")
		}
	})
}

func TestChatAccountAttemptParallelAliases(t *testing.T) {
	f := newChatAccountFixture(t)
	f.routing.Accounts[0].DailyCeiling = "326"
	if err := f.ledger.applyRouting(f.routing, "8"); err != nil {
		t.Fatal(err)
	}
	f.response = "truncated"
	var wg sync.WaitGroup
	for _, id := range []string{"chat-a", "chat-b"} {
		s, err := f.ledger.prepareChatAccount(id)
		if err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := runChatAccountAttempt(f.ctx, s, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, chatTestGuard(f.guard), chatTestLifecycle(t), f.emit)
			if err == nil {
				t.Error("truncated fixture unexpectedly succeeded")
			}
		}()
	}
	wg.Wait()
	if f.calls.Load() != 1 || len(f.ledger.state.Holds) != 1 {
		t.Fatal("aliases exceeded shared account ceiling")
	}
}

func TestChatAccountAttemptRestrictivePersistence(t *testing.T) {
	for _, name := range []string{"quarantine", "credential outcome", "rate limit outcome", "above bound"} {
		t.Run(name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			switch name {
			case "quarantine":
				f.response = chatResponse("wrong", `{"role":"assistant","content":"ok"}`, `"stop"`, "", false)
			case "credential outcome":
				f.status = http.StatusUnauthorized
				f.response = "failure"
			case "rate limit outcome":
				f.status = http.StatusTooManyRequests
				f.response = "failure"
			case "above bound":
				f.response = chatResponse("synthetic-v1", `{"role":"assistant","content":"ok"}`, `"stop"`, `{"prompt_tokens":10,"completion_tokens":101,"total_tokens":111}`, false)
			}
			f.ledger.write = func(path string, data []byte, mode os.FileMode) error {
				s, err := decodeLedger(data)
				if err != nil {
					return err
				}
				if len(s.Quarantine) > 0 || s.Accounts[0].CredentialFailure != nil || s.Accounts[0].Cooldown != nil {
					return errors.New("synthetic-key-body-sentinel")
				}
				return durablefile.WriteFile(path, data, mode)
			}
			if _, err := f.run(t); !errors.Is(err, errChatAccounting) {
				t.Fatalf("restrictive persistence failure not propagated: %v", err)
			}
			f.ledger.write = durablefile.WriteFile
			if _, err := f.run(t); err == nil || f.calls.Load() != 1 {
				t.Fatal("failed restrictive write permitted another send")
			}
		})
	}
}

func TestChatAccountAttemptQuarantineIdentityLifecycle(t *testing.T) {
	f := newChatAccountFixture(t)
	s, err := f.ledger.prepareChatAccount("chat-a")
	if err != nil {
		t.Fatal(err)
	}
	f.response = chatResponse("wrong", `{"role":"assistant","content":"ok"}`, `"stop"`, "", false)
	f.emit = func([]byte) error { t.Fatal("identity-invalid response exposed"); return nil }
	f.guard = func(context.Context) error {
		// Both aliases are retired while this reserved response is outstanding.
		if err := f.ledger.applyRouting(nodepolicy.Routing{}, "8"); err != nil {
			t.Fatal(err)
		}
		return nil
	}
	if _, err := runChatAccountAttempt(f.ctx, s, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, chatTestGuard(f.guard), chatTestLifecycle(t), f.emit); !errors.Is(err, errChatIdentity) {
		t.Fatal(err)
	}
	f.routing.Deployments[0].ID = "renamed"
	dir := filepath.Dir(f.ledger.storePath)
	f.ledger = reopen(t, f.ledger, dir, f.clock, f.routing)
	renamed, err := f.ledger.prepareChatAccount("renamed")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runChatAccountAttempt(f.ctx, renamed, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, f.budget.acquire, chatTestGuard(publicTestGuard), chatTestLifecycle(t), f.emit); err == nil || f.calls.Load() != 1 {
		t.Fatal("rename/retirement/reopen cleared quarantine")
	}
}

func TestChatAccountAttemptRetentionAndPriorProcess(t *testing.T) {
	f := newChatAccountFixture(t)
	if _, err := f.run(t); err != nil {
		t.Fatal(err)
	}
	completed := f.ledger.state.Holds[0].Attempt
	if !f.ledger.state.Holds[0].evictable() {
		t.Fatal("normal completion remains unfinished")
	}
	// Simulate a classification interrupted after money and outcome completed.
	pending := mustReserve(t, f.ledger, "chat-a", 0, 1)
	if err := f.ledger.reconcile(pending.Attempt, nil); err != nil {
		t.Fatal(err)
	}
	if err := f.ledger.recordOutcome(pending.Attempt, outcomeOK, 0, time.Time{}); err != nil {
		t.Fatal(err)
	}
	completeOtherLedgerAttempts(t, f.ledger)
	if f.ledger.state.hold(completed) != nil {
		t.Fatal("completed attempt did not become evictable")
	}
	if h := f.ledger.state.hold(pending.Attempt); h == nil || h.QualificationChecked {
		t.Fatal("delayed classification lost immutable hold")
	}
	f.ledger = reopen(t, f.ledger, filepath.Dir(f.ledger.storePath), f.clock, f.routing)
	wantErr(t, "prior-process classification", f.ledger.recordQualification(pending.Attempt), errAttemptUnknown)
	if h := f.ledger.state.hold(pending.Attempt); h == nil || h.QualificationChecked || h.evictable() {
		t.Fatal("silently reclaimed prior-process classification")
	}
}

func TestChatAccountAttemptIndependentCompletionFailures(t *testing.T) {
	for _, name := range []string{"outcome", "reconcile", "qualification"} {
		t.Run(name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			failed := false
			f.ledger.write = func(path string, data []byte, mode os.FileMode) error {
				s, err := decodeLedger(data)
				if err != nil {
					return err
				}
				if len(s.Holds) > 0 {
					h := s.Holds[0]
					fail := name == "outcome" && h.Outcome != "" || name == "reconcile" && h.State == holdReconciled || name == "qualification" && h.QualificationChecked
					if fail && !failed {
						failed = true
						return errors.New("synthetic-write-sentinel")
					}
				}
				return durablefile.WriteFile(path, data, mode)
			}
			if _, err := f.run(t); !errors.Is(err, errChatAccounting) || !failed {
				t.Fatal("completion write failure hidden")
			}
			h := f.ledger.state.Holds[0]
			switch name {
			case "outcome":
				if h.Outcome != "" || h.State != holdReconciled || !h.QualificationChecked {
					t.Fatal("outcome failure skipped independent completions")
				}
			case "reconcile":
				if !h.outstanding() || h.Outcome != outcomeOK || !h.QualificationChecked {
					t.Fatal("money failure released hold or skipped classification")
				}
			case "qualification":
				if h.QualificationChecked || h.State != holdReconciled || h.Outcome != outcomeOK || h.evictable() {
					t.Fatal("failed classification became complete")
				}
			}
		})
	}
}

func TestChatAccountAttemptDelayedClassification(t *testing.T) {
	f := newChatAccountFixture(t)
	f.response = chatResponse("wrong", `{"role":"assistant","content":"ok"}`, `"stop"`, "", false)
	f.guard = func(context.Context) error { completeOtherLedgerAttempts(t, f.ledger); return nil }
	// Use a custom synthetic transport because the fixture's newest hold is now
	// a completed unrelated attempt, rather than this delayed attempt.
	f.transport.client.Transport = publicTestRoundTrip(func(*http.Request) (*http.Response, error) {
		f.calls.Add(1)
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(f.response)), ContentLength: -1}, nil
	})
	if _, err := f.run(t); !errors.Is(err, errChatIdentity) {
		t.Fatal(err)
	}
	if len(f.ledger.state.Quarantine) != 1 || !f.ledger.state.Holds[0].QualificationChecked {
		t.Fatal("delayed classification lost reserved identity")
	}
}

func TestChatAccountAttemptHeadroomBeforeLookup(t *testing.T) {
	for _, name := range []string{"parser", "transport", "profile event cap"} {
		t.Run(name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			s, err := f.ledger.prepareChatAccount("chat-a")
			if err != nil {
				t.Fatal(err)
			}
			lookups := 0
			f.resolve = func(nodepolicy.CredentialRef) (string, bool) { lookups++; return "synthetic-selected-key", true }
			if name == "profile event cap" {
				f.profile.MaxEventBytes++
				f.profile.MaxLineBytes = 4096
			}
			budget := func(n int64) (func(), error) {
				if name == "parser" && n == chatFixedStorage+256 || name == "transport" && n == 3*(f.transport.bounds.MaxResponseBytes+1) {
					return nil, errChatBudget
				}
				return f.budget.acquire(n)
			}
			if _, err := runChatAccountAttempt(f.ctx, s, []byte(f.body), f.profile, f.ledger, f.transport, f.resolve, budget, chatTestGuard(f.guard), chatTestLifecycle(t), f.emit); err == nil {
				t.Fatal("headroom refusal accepted")
			}
			if lookups != 0 || f.calls.Load() != 0 || len(f.ledger.state.Holds) != 0 {
				t.Fatal("resource refusal occurred after lookup/reservation/send")
			}
		})
	}
}

func TestChatAccountAttemptTransportFailure(t *testing.T) {
	f := newChatAccountFixture(t)
	f.transport.client.Transport = publicTestRoundTrip(func(*http.Request) (*http.Response, error) {
		f.calls.Add(1)
		return nil, errors.New("synthetic-key-body-sentinel")
	})
	if _, err := f.run(t); !errors.Is(err, errChatTransient) {
		t.Fatal("transport error not sanitized")
	}
	h := f.ledger.state.Holds[0]
	if f.calls.Load() != 1 || h.Outcome != outcomeTransient || !h.QualificationChecked || !h.outstanding() {
		t.Fatal("connection failure lost classification/liability or retried")
	}
}

func TestChatAccountAttemptCompletion(t *testing.T) {
	for _, name := range []string{"trusted", "missing usage", "malformed usage", "identity", "above bound", "truncated", "post-exposure disconnect"} {
		t.Run(name, func(t *testing.T) {
			f := newChatAccountFixture(t)
			switch name {
			case "missing usage":
				f.response = chatResponse("synthetic-v1", `{"role":"assistant","content":"ok"}`, `"stop"`, "", false)
			case "malformed usage":
				f.response = chatResponse("synthetic-v1", `{"role":"assistant","content":"ok"}`, `"stop"`, `{"prompt_tokens":0,"completion_tokens":-1,"total_tokens":-1}`, false)
			case "identity":
				f.response = chatResponse("wrong", `{"role":"assistant","content":"ok"}`, `"stop"`, `{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`, false)
			case "above bound":
				f.response = chatResponse("synthetic-v1", `{"role":"assistant","content":"ok"}`, `"stop"`, `{"prompt_tokens":10,"completion_tokens":101,"total_tokens":111}`, false)
			case "truncated":
				f.response = `{"model":"synthetic-v1"`
			case "post-exposure disconnect":
				f.body = strings.TrimSuffix(chatSimple, "}") + `,"stream":true}`
				f.response = chatEvent(chatChunk(`{"content":"partial"}`, `null`)) + chatEvent(chatChunk(`{}`, `"stop"`)) + chatEvent(chatUsageEvent(`{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`))
			}
			result, err := f.run(t)
			if f.calls.Load() != 1 {
				t.Fatal("attempt did not dispatch exactly once")
			}
			if name == "above bound" {
				if _, nextErr := f.run(t); nextErr == nil || f.calls.Load() != 1 {
					t.Fatal("above-bound violation permitted another send")
				}
			}
			h := f.ledger.state.Holds[0]
			if !h.QualificationChecked || h.Outcome == "" {
				t.Fatal("closed classification left unfinished")
			}
			switch name {
			case "trusted":
				if err != nil || !h.evictable() || h.Actual != "14" || h.Outcome != outcomeOK {
					t.Fatalf("trusted completion: %v, %+v", err, h)
				}
			case "missing usage":
				if err != nil || h.Actual != h.Reserved || !h.evictable() {
					t.Fatal("missing usage did not charge full bound")
				}
			case "above bound":
				if !errors.Is(err, errChatBound) || f.ledger.state.Accounts[0].Frozen == nil || !f.ledger.state.quarantined(h.key()) {
					t.Fatal("above-bound usage did not freeze and quarantine")
				}
			default:
				if err == nil || !h.outstanding() || h.Usage != nil || h.Actual != "" {
					t.Fatal("untrusted/incomplete usage released liability")
				}
				if name == "identity" && !f.ledger.state.quarantined(h.key()) {
					t.Fatal("identity failure did not quarantine")
				}
				if name == "post-exposure disconnect" && !result.Exposed {
					t.Fatal("fixture did not expose metadata/data")
				}
			}
			state, readErr := readAccountStore(f.ledger.storePath)
			if readErr != nil {
				t.Fatal(readErr)
			}
			if !reflect.DeepEqual(state.Holds[0], h) {
				t.Fatal("completion not durable")
			}
		})
	}
}

// ledgerClock is a controllable wall clock; tests never sleep for periods or
// cooldowns.
type ledgerClock struct {
	mu sync.Mutex
	t  time.Time
}

func newLedgerClock() *ledgerClock {
	return &ledgerClock{t: time.Date(2026, 10, 19, 12, 0, 0, 0, time.UTC)}
}

func (c *ledgerClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *ledgerClock) set(t time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = t
}

func (c *ledgerClock) advance(d time.Duration) { c.set(c.now().Add(d)) }

const (
	testCredential  nodepolicy.CredentialRef = "NVPAIR_PROVIDER_SECRET_PAID"
	otherCredential nodepolicy.CredentialRef = "NVPAIR_PROVIDER_SECRET_OTHER"
)

// testDeployment is a complete deployment on account at 1 nano-USD per input
// token and 2 per output token.
func testDeployment(id, account string) nodepolicy.Deployment {
	return nodepolicy.Deployment{
		ID: id, Account: account, Model: "small-1", Family: "alpha",
		ContextTokens: 32768, MaxOutputTokens: 4096,
		Prices: &nodepolicy.Prices{Currency: nodepolicy.CurrencyUSD, Unit: nodepolicy.PriceUnit,
			InputPerMillion: "1000000", OutputPerMillion: "2000000"},
		Endpoint:      "https://api.example.com/v1/chat/completions",
		Protocol:      nodepolicy.ProtocolOpenAIChatCompletions,
		Transport:     &nodepolicy.Transport{ConnectTimeoutSeconds: 10, HeaderTimeoutSeconds: 60, IdleTimeoutSeconds: 60, MaxResponseBytes: 1 << 20, MaxErrorBodyBytes: 1 << 10, MaxEventBytes: 1 << 10},
		Qualification: &nodepolicy.QualificationRef{ID: "provider-q", Revision: "1"},
	}
}

// testRouting has one paid account with two deployments of it, the way two
// tiers or two facades share one account.
func testRouting(perRequest, daily, monthly string) nodepolicy.Routing {
	return nodepolicy.Routing{
		Accounts: []nodepolicy.Account{{
			ID: "paid", Credential: testCredential, Billing: nodepolicy.BillingPaid,
			PerRequestCeiling: nodepolicy.NanoUSD(perRequest), DailyCeiling: nodepolicy.NanoUSD(daily), MonthlyCeiling: nodepolicy.NanoUSD(monthly),
		}},
		Deployments: []nodepolicy.Deployment{testDeployment("chat-a", "paid"), testDeployment("chat-b", "paid")},
	}
}

func openTestLedger(t *testing.T, dir string, clock *ledgerClock, routing nodepolicy.Routing) *accountLedger {
	t.Helper()
	l := newAccountLedger()
	l.setDir(dir, nil)
	l.now = clock.now
	t.Cleanup(l.close)
	if err := l.applyRouting(routing, "7"); err != nil {
		t.Fatalf("applyRouting: %v", err)
	}
	return l
}

func mustReserve(t *testing.T, l *accountLedger, deployment string, in, out int64) accountHold {
	t.Helper()
	h, err := l.reserve(deployment, in, out)
	if err != nil {
		t.Fatalf("reserve(%s, %d, %d): %v", deployment, in, out, err)
	}
	return h
}

func wantErr(t *testing.T, what string, err, want error) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("%s: err = %v, want %v", what, err, want)
	}
}

// reopen closes l, as a process exit would, and opens the same store again.
func reopen(t *testing.T, l *accountLedger, dir string, clock *ledgerClock, routing nodepolicy.Routing) *accountLedger {
	t.Helper()
	l.close()
	return openTestLedger(t, dir, clock, routing)
}

func TestChargeNanoUSDIsExactAndRoundsUp(t *testing.T) {
	for _, c := range []struct {
		tokens int64
		rate   nodepolicy.NanoUSD
		want   int64
	}{
		{0, "9223372036854775807", 0},
		{1, "1", 1},
		{1_000_000, "1", 1},
		{1_000_001, "1", 2},
		{3, "333333", 1},
		{3, "333334", 2},
		{1, "9223372036854775807", 9223372036855},
		{1_000_000, "9223372036854775807", math.MaxInt64},
		{1_000_000, "9007199254740993", 9007199254740993},
		{1 << 24, "549755813887", 9223372036838},
	} {
		got, err := chargeNanoUSD(c.tokens, c.rate)
		if err != nil || got != c.want {
			t.Errorf("chargeNanoUSD(%d, %s) = %d, %v; want %d", c.tokens, c.rate, got, err, c.want)
		}
	}
	for _, c := range []struct {
		tokens int64
		rate   nodepolicy.NanoUSD
	}{
		{1_000_001, "9223372036854775807"},
		{1 << 24, "9223372036854775807"},
		{1 << 40, "10000000000000"},
		// The quotient fits uint64 exactly, but rounding it used to wrap to 0.
		{2_000_001, "9223367425171063222"},
	} {
		if _, err := chargeNanoUSD(c.tokens, c.rate); !errors.Is(err, errMoneyOverflow) {
			t.Errorf("chargeNanoUSD(%d, %s) err = %v, want overflow", c.tokens, c.rate, err)
		}
	}
	if _, err := reservationAmount(1_000_000, 1, "9223372036854775807", "1"); !errors.Is(err, errMoneyOverflow) {
		t.Errorf("reservationAmount overflowing the sum: err = %v, want overflow", err)
	}
	for _, bad := range []nodepolicy.NanoUSD{"", "-1", "1.5", "1e3", "01", "9223372036854775808"} {
		if _, err := chargeNanoUSD(1, bad); err == nil {
			t.Errorf("chargeNanoUSD(1, %q) accepted a malformed rate", bad)
		}
	}
}

// A deployment priced in anything but USD nano-USD per million tokens is never
// converted: it is refused.
func TestReserveRefusesMismatchedCurrencyAndUnit(t *testing.T) {
	for name, mutate := range map[string]func(*nodepolicy.Prices){
		"currency": func(p *nodepolicy.Prices) { p.Currency = "EUR" },
		"unit":     func(p *nodepolicy.Prices) { p.Unit = "nanoUSDPerToken" },
	} {
		t.Run(name, func(t *testing.T) {
			routing := testRouting("1000000", "1000000", "1000000")
			mutate(routing.Deployments[0].Prices)
			l := openTestLedger(t, t.TempDir(), newLedgerClock(), routing)
			_, err := l.reserve("chat-a", 1, 1)
			wantErr(t, "reserve", err, errDeploymentUnqualified)
		})
	}
	routing := testRouting("1000000", "1000000", "1000000")
	routing.Deployments[0].Prices = nil
	l := openTestLedger(t, t.TempDir(), newLedgerClock(), routing)
	_, err := l.reserve("chat-a", 1, 1)
	wantErr(t, "reserve without prices", err, errDeploymentUnqualified)
}

// Many attempts racing on two deployments of one account never hold more than
// its daily ceiling together.
func TestConcurrentAliasesShareTheCeiling(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	// Each reservation is 100 + 2*200 = 500.
	routing := testRouting("500", "5000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	var wg sync.WaitGroup
	var mu sync.Mutex
	granted := 0
	for i := range 40 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			deployment := []string{"chat-a", "chat-b"}[i%2]
			if _, err := l.reserve(deployment, 100, 200); err == nil {
				mu.Lock()
				granted++
				mu.Unlock()
			} else if !errors.Is(err, errAccountCeiling) {
				t.Errorf("reserve: %v", err)
			}
		}()
	}
	wg.Wait()
	if granted != 10 {
		t.Fatalf("granted %d reservations, want exactly 10 of 500 under a 5000 ceiling", granted)
	}
	l = reopen(t, l, dir, clock, routing)
	got, err := l.state.outstanding("paid")
	if err != nil || got != 5000 {
		t.Fatalf("durable outstanding = %d, %v; want 5000", got, err)
	}
	_, err = l.reserve("chat-b", 0, 1)
	wantErr(t, "reserve beyond the ceiling after reopening", err, errAccountCeiling)
}

func TestReserveEnforcesPerRequestMonthlyAndTokenBounds(t *testing.T) {
	l := openTestLedger(t, t.TempDir(), newLedgerClock(), testRouting("499", "1000000", "600"))
	_, err := l.reserve("chat-a", 100, 200)
	wantErr(t, "above the per-request ceiling", err, errAccountCeiling)
	mustReserve(t, l, "chat-a", 99, 200)
	_, err = l.reserve("chat-a", 2, 50)
	wantErr(t, "above the monthly ceiling", err, errAccountCeiling)
	for _, c := range [][2]int64{{-1, 1}, {0, 0}, {0, 4097}, {32768 - 4095, 4096}} {
		_, err = l.reserve("chat-a", c[0], c[1])
		wantErr(t, "token bounds "+strconv.FormatInt(c[0], 10)+"/"+strconv.FormatInt(c[1], 10), err, errTokenBounds)
	}
	_, err = l.reserve("ghost", 1, 1)
	wantErr(t, "unknown deployment", err, errUnknownDeployment)
}

func TestReconcileIsIdempotentAndUsesTrustedUsage(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	h := mustReserve(t, l, "chat-a", 100, 200)
	if h.Reserved != "500" {
		t.Fatalf("reserved = %s, want 500", h.Reserved)
	}
	usage := &tokenUsage{InputTokens: 50, OutputTokens: 100}
	if err := l.reconcile(h.Attempt, usage); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if err := l.reconcile(h.Attempt, &tokenUsage{InputTokens: 50, OutputTokens: 100}); err != nil {
		t.Fatalf("duplicate reconcile: %v", err)
	}
	wantErr(t, "conflicting reconcile", l.reconcile(h.Attempt, &tokenUsage{InputTokens: 51, OutputTokens: 100}), errAttemptConflict)
	wantErr(t, "reconcile without usage after usage", l.reconcile(h.Attempt, nil), errAttemptConflict)

	missing := mustReserve(t, l, "chat-b", 10, 20)
	if err := l.reconcile(missing.Attempt, nil); err != nil {
		t.Fatalf("reconcile without usage: %v", err)
	}
	negative := mustReserve(t, l, "chat-a", 1, 1)
	before := readLedgerBytes(t, dir)
	memory := l.state.clone()
	wantErr(t, "negative usage", l.reconcile(negative.Attempt, &tokenUsage{InputTokens: -1}), errAttemptConflict)
	if !reflect.DeepEqual(l.state, memory) || string(readLedgerBytes(t, dir)) != string(before) {
		t.Fatal("negative usage changed memory or the durable hold")
	}
	wantErr(t, "unknown attempt", l.reconcile("nobody-1", nil), errAttemptUnknown)

	l = reopen(t, l, dir, clock, routing)
	spent := l.state.account("paid").Spent[dayKey(clock.now())]
	// 50 + 2*100 = 250 from usage, plus the 50 bound charged without usage.
	if spent != "300" {
		t.Fatalf("durable spend today = %s, want 300", spent)
	}
}

// A price or ceiling edit applies to new reservations; a hold keeps the price
// snapshot it was reserved under.
func TestPriceChangeKeepsTheHoldsSnapshot(t *testing.T) {
	clock := newLedgerClock()
	l := openTestLedger(t, t.TempDir(), clock, testRouting("1000000", "1000000", "1000000"))
	h := mustReserve(t, l, "chat-a", 100, 200)

	edited := testRouting("1000000", "6000", "1000000")
	edited.Deployments[0].Prices.InputPerMillion = "10000000"
	edited.Deployments[0].Prices.OutputPerMillion = "20000000"
	if err := l.applyRouting(edited, "8"); err != nil {
		t.Fatal(err)
	}
	if err := l.reconcile(h.Attempt, &tokenUsage{InputTokens: 100, OutputTokens: 200}); err != nil {
		t.Fatal(err)
	}
	if got := l.state.account("paid").Spent[dayKey(clock.now())]; got != "500" {
		t.Fatalf("spend = %s, want 500 at the reserved price", got)
	}
	n := mustReserve(t, l, "chat-a", 10, 20)
	if n.Reserved != "500" {
		t.Fatalf("new reservation = %s, want 500 at the new price", n.Reserved)
	}
	hold := l.state.hold(n.Attempt)
	if hold.PolicyRevision != "8" || hold.InputPerMillion != "10000000" || hold.Currency != "USD" || hold.Unit != nodepolicy.PriceUnit {
		t.Fatalf("hold provenance = %+v", hold)
	}
	// 500 spent + 500 outstanding + 5910 > 6000.
	_, err := l.reserve("chat-a", 101, 245)
	wantErr(t, "lowered ceiling", err, errAccountCeiling)
}

// Unresolved holds survive restart, period rollover, account deletion and
// re-addition, and keep counting in every later period; reconciled spend
// counts only in its own period.
func TestHoldsSurviveRestartRolloverAndRetirement(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	clock.set(time.Date(2026, 10, 31, 23, 0, 0, 0, time.UTC))
	routing := testRouting("1000000", "1000", "1500")
	l := openTestLedger(t, dir, clock, routing)
	open := mustReserve(t, l, "chat-a", 100, 200) // 500, left unresolved
	done := mustReserve(t, l, "chat-a", 100, 200)
	if err := l.reconcile(done.Attempt, &tokenUsage{InputTokens: 100, OutputTokens: 200}); err != nil {
		t.Fatal(err)
	}

	// Restart after the day and the month rolled over, with the account
	// deleted from the configuration and only another one configured.
	clock.set(time.Date(2026, 11, 1, 1, 0, 0, 0, time.UTC))
	retired := nodepolicy.Routing{
		Accounts: []nodepolicy.Account{{ID: "spare", Credential: otherCredential, Billing: nodepolicy.BillingPaid,
			PerRequestCeiling: "1", DailyCeiling: "1", MonthlyCeiling: "1"}},
	}
	l = reopen(t, l, dir, clock, retired)
	h := l.state.hold(open.Attempt)
	if h == nil || h.State != holdUncertain {
		t.Fatalf("recovered hold = %+v, want it uncertain", h)
	}
	wantErr(t, "reconcile a recovered hold", l.reconcile(open.Attempt, nil), errAttemptUnknown)

	// Re-added: yesterday's reconciled 500 no longer counts, the open 500
	// does, so 500 more fits the 1000 daily ceiling and 1 more does not.
	if err := l.applyRouting(routing, "9"); err != nil {
		t.Fatal(err)
	}
	mustReserve(t, l, "chat-b", 100, 200)
	_, err := l.reserve("chat-b", 1, 1)
	wantErr(t, "reserve beyond outstanding holds", err, errAccountCeiling)
	if _, ok := l.state.account("paid").Spent["d:2026-10-31"]; ok {
		t.Fatal("spend of a past day was kept")
	}
}

func TestDayRolloverKeepsMonthlySpend(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000", "1000", "1200")
	l := openTestLedger(t, dir, clock, routing)
	h := mustReserve(t, l, "chat-a", 100, 200)
	if err := l.reconcile(h.Attempt, nil); err != nil {
		t.Fatal(err)
	}
	clock.advance(24 * time.Hour)
	l = reopen(t, l, dir, clock, routing)
	mustReserve(t, l, "chat-b", 100, 200)
	_, err := l.reserve("chat-a", 1, 100) // 500 spent + 500 held + 201 > 1200.
	wantErr(t, "monthly spend after a new day", err, errAccountCeiling)
	if a := l.state.account("paid"); a.Spent[monthKey(clock.now())] != "500" || len(a.Spent) != 1 {
		t.Fatalf("spend = %v, want only the month's 500", a.Spent)
	}
}

func TestOutstandingHoldCapRefusesWithoutDroppingLiability(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	h := mustReserve(t, l, "chat-a", 0, 1)
	next := l.state.clone()
	seed := *next.hold(h.Attempt)
	seed.State = holdUncertain
	for i := 1; i < maxLedgerHolds-maxResolvedHolds-1; i++ {
		seed.Attempt = "seed-" + strconv.Itoa(i)
		next.Holds = append(next.Holds, seed)
	}
	if err := l.commitLocked(next); err != nil {
		t.Fatal(err)
	}
	mustReserve(t, l, "chat-a", 0, 1)
	before := readLedgerBytes(t, dir)
	_, err := l.reserve("chat-b", 0, 1)
	wantErr(t, "outstanding hold cap", err, errLedgerFull)
	if string(readLedgerBytes(t, dir)) != string(before) {
		t.Fatal("a full ledger changed its stored liability")
	}
	l = reopen(t, l, dir, clock, routing)
	want := int64(2 * (maxLedgerHolds - maxResolvedHolds))
	if got, err := l.state.outstanding("paid"); err != nil || got != want {
		t.Fatalf("outstanding = %d, %v, want %d", got, err, want)
	}
}

// A store write that fails grants nothing and changes nothing. One that lands
// without confirmed durability leaves an uncertain hold and refuses every
// reservation until a complete durable rewrite succeeds.
func TestPersistenceFaultsGrantNoPermission(t *testing.T) {
	failing := errors.New("injected")
	tempFault := func(step string) durablefile.Replacer {
		return durablefile.Replacer{CreateTemp: func(dir, base string, perm os.FileMode) (durablefile.File, error) {
			f, err := os.CreateTemp(dir, base+".*.tmp")
			if err != nil {
				return nil, err
			}
			if err := f.Chmod(perm); err != nil {
				f.Close()
				return nil, err
			}
			return ledgerFaultFile{File: f, step: step, err: failing}, nil
		}}
	}
	for name, replacer := range map[string]durablefile.Replacer{
		"create": {CreateTemp: func(string, string, os.FileMode) (durablefile.File, error) { return nil, failing }},
		"write":  tempFault("write"),
		"sync":   tempFault("sync"),
		"close":  tempFault("close"),
		"rename": {Replace: func(string, string) error { return failing }},
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			clock := newLedgerClock()
			l := openTestLedger(t, dir, clock, testRouting("1000000", "1000000", "1000000"))
			mustReserve(t, l, "chat-a", 1, 1)
			before := readLedgerBytes(t, dir)
			memory := l.state.clone()
			l.write = replacer.WriteFile
			if _, err := l.reserve("chat-a", 100, 200); !errors.Is(err, failing) {
				t.Fatalf("reserve err = %v, want the injected failure", err)
			}
			after := readLedgerBytes(t, dir)
			if string(after) != string(before) || !reflect.DeepEqual(l.state, memory) {
				t.Fatalf("a failed reservation changed the store or memory (holds %d)", len(l.state.Holds))
			}
			l.write = durablefile.WriteFile
			mustReserve(t, l, "chat-a", 100, 200)
		})
	}

	t.Run("directory sync", func(t *testing.T) {
		dir := t.TempDir()
		clock := newLedgerClock()
		routing := testRouting("1000000", "1000000", "1000000")
		l := openTestLedger(t, dir, clock, routing)
		l.write = durablefile.Replacer{SyncDir: func(string) error { return errors.New("injected") }}.WriteFile
		_, err := l.reserve("chat-a", 100, 200)
		wantErr(t, "reserve landing without durability", err, durablefile.ErrReplaced)
		if len(l.state.Holds) != 1 || l.state.Holds[0].State != holdUncertain {
			t.Fatalf("landed hold = %+v, want one uncertain hold", l.state.Holds)
		}
		_, err = l.reserve("chat-a", 1, 1)
		wantErr(t, "reserve while durability is unconfirmed", err, errLedgerUnavailable)
		if len(l.state.Holds) != 1 {
			t.Fatalf("holds = %d, want only the landed one", len(l.state.Holds))
		}
		l.write = durablefile.WriteFile
		mustReserve(t, l, "chat-a", 1, 1)
		l = reopen(t, l, dir, clock, routing)
		if got, _ := l.state.outstanding("paid"); got != 503 {
			t.Fatalf("outstanding after reopen = %d, want 503 (the uncertain hold counts)", got)
		}
	})
}

type ledgerFaultFile struct {
	durablefile.File
	step string
	err  error
}

func (f ledgerFaultFile) Write(data []byte) (int, error) {
	if f.step == "write" {
		return 0, f.err
	}
	return f.File.Write(data)
}

func (f ledgerFaultFile) Sync() error {
	if f.step == "sync" {
		return f.err
	}
	return f.File.Sync()
}

func (f ledgerFaultFile) Close() error {
	err := f.File.Close()
	if f.step == "close" {
		return f.err
	}
	return err
}

func readLedgerBytes(t *testing.T, dir string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, accountStoreFileName))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestHalfOpenAdmitsExactlyOneProbe(t *testing.T) {
	clock := newLedgerClock()
	l := openTestLedger(t, t.TempDir(), clock, testRouting("1000000", "1000000", "1000000"))
	limited := mustReserve(t, l, "chat-a", 1, 1)
	if err := l.recordOutcome(limited.Attempt, outcomeRateLimited, 0, time.Time{}); err != nil {
		t.Fatal(err)
	}
	_, err := l.reserve("chat-b", 1, 1)
	wantErr(t, "reserve while cooling down", err, errAccountCooldown)
	clock.advance(defaultCooldown)

	var wg sync.WaitGroup
	probes := make(chan accountHold, 20)
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h, err := l.reserve("chat-a", 1, 1)
			if err == nil {
				probes <- h
			} else if !errors.Is(err, errAccountCooldown) {
				t.Errorf("reserve: %v", err)
			}
		}()
	}
	wg.Wait()
	close(probes)
	var probe accountHold
	n := 0
	for h := range probes {
		probe = h
		n++
	}
	if n != 1 || !probe.Probe {
		t.Fatalf("%d probes admitted (probe flag %v), want exactly one", n, probe.Probe)
	}
	if err := l.recordOutcome(probe.Attempt, outcomeOK, 0, time.Time{}); err != nil {
		t.Fatal(err)
	}
	if h := mustReserve(t, l, "chat-b", 1, 1); h.Probe {
		t.Fatal("a reservation after the probe succeeded is still a probe")
	}
}

// A success that raced a newer restriction never clears it, and a short rate
// limit never shortens a longer exhaustion.
func TestOutcomesMergeConservatively(t *testing.T) {
	clock := newLedgerClock()
	l := openTestLedger(t, t.TempDir(), clock, testRouting("1000000", "1000000", "1000000"))
	inflight := mustReserve(t, l, "chat-a", 1, 1)
	first := mustReserve(t, l, "chat-a", 1, 1)
	if err := l.recordOutcome(first.Attempt, outcomeRateLimited, 30*time.Second, time.Time{}); err != nil {
		t.Fatal(err)
	}
	clock.advance(31 * time.Second)
	probe := mustReserve(t, l, "chat-b", 1, 1)
	if !probe.Probe {
		t.Fatal("expected a probe")
	}
	// The older in-flight attempt is rate limited after the probe was taken.
	if err := l.recordOutcome(inflight.Attempt, outcomeRateLimited, 2*time.Minute, time.Time{}); err != nil {
		t.Fatal(err)
	}
	if err := l.recordOutcome(probe.Attempt, outcomeOK, 0, time.Time{}); err != nil {
		t.Fatal(err)
	}
	_, err := l.reserve("chat-a", 1, 1)
	wantErr(t, "reserve after a stale probe success", err, errAccountCooldown)

	clock.advance(2 * time.Minute)
	exhausting := mustReserve(t, l, "chat-a", 1, 1)
	if err := l.recordOutcome(exhausting.Attempt, outcomeExhausted, 0, clock.now().Add(2*time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := l.recordOutcome(first.Attempt, outcomeRateLimited, 0, time.Time{}); err != nil {
		t.Fatalf("repeating the same outcome: %v", err)
	}
	wantErr(t, "a different second outcome", l.recordOutcome(first.Attempt, outcomeOK, 0, time.Time{}), errAttemptConflict)
	c := l.state.account("paid").Cooldown
	if c.Reason != cooldownExhausted || c.UntilMs != clock.now().Add(2*time.Hour).UnixMilli() {
		t.Fatalf("cooldown = %+v, want the two-hour exhaustion", c)
	}
	late := mustReserveAfter(t, l, clock, 2*time.Hour)
	if err := l.recordOutcome(late.Attempt, outcomeRateLimited, time.Second, time.Time{}); err != nil {
		t.Fatal(err)
	}
	h := mustReserveAfter(t, l, clock, time.Minute)
	for _, bad := range []time.Time{clock.now(), clock.now().Add(33 * 24 * time.Hour)} {
		wantErr(t, "exhaustion outside its window", l.recordOutcome(h.Attempt, outcomeExhausted, 0, bad), errInvalidOutcome)
	}
}

func TestRateLimitNeverShortensExhaustion(t *testing.T) {
	clock := newLedgerClock()
	l := openTestLedger(t, t.TempDir(), clock, testRouting("1000000", "1000000", "1000000"))
	exhausted := mustReserve(t, l, "chat-a", 1, 1)
	limited := mustReserve(t, l, "chat-b", 1, 1)
	reset := clock.now().Add(2 * time.Hour)
	if err := l.recordOutcome(exhausted.Attempt, outcomeExhausted, 0, reset); err != nil {
		t.Fatal(err)
	}
	if err := l.recordOutcome(limited.Attempt, outcomeRateLimited, time.Second, time.Time{}); err != nil {
		t.Fatal(err)
	}
	c := l.state.account("paid").Cooldown
	if c.Reason != cooldownExhausted || c.UntilMs != reset.UnixMilli() {
		t.Fatalf("cooldown = %+v, want the original exhaustion", c)
	}
	clock.advance(time.Hour)
	_, err := l.reserve("chat-a", 1, 1)
	wantErr(t, "reserve before exhaustion ends", err, errAccountCooldown)
}

func TestRateLimitClampsRetryAfter(t *testing.T) {
	for _, tc := range []struct {
		name        string
		retry, want time.Duration
	}{
		{"absent", 0, defaultCooldown},
		{"below minimum", time.Millisecond, minCooldown},
		{"above maximum", 2 * time.Hour, maxCooldown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clock := newLedgerClock()
			l := openTestLedger(t, t.TempDir(), clock, testRouting("1000000", "1000000", "1000000"))
			h := mustReserve(t, l, "chat-a", 1, 1)
			if err := l.recordOutcome(h.Attempt, outcomeRateLimited, tc.retry, time.Time{}); err != nil {
				t.Fatal(err)
			}
			if got := l.state.account("paid").Cooldown.UntilMs; got != clock.now().Add(tc.want).UnixMilli() {
				t.Fatalf("deadline = %d, want %d", got, clock.now().Add(tc.want).UnixMilli())
			}
		})
	}
}

// mustReserveAfter advances the clock by d and reserves the probe.
func mustReserveAfter(t *testing.T, l *accountLedger, clock *ledgerClock, d time.Duration) accountHold {
	t.Helper()
	clock.advance(d)
	return mustReserve(t, l, "chat-a", 1, 1)
}

// A probe left unresolved by a crash becomes a cooldown, never shorter than
// a deadline recorded meanwhile.
func TestRecoveredProbeKeepsTheLongerDeadline(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	other := mustReserve(t, l, "chat-a", 1, 1)
	limited := mustReserve(t, l, "chat-a", 1, 1)
	if err := l.recordOutcome(limited.Attempt, outcomeRateLimited, time.Second, time.Time{}); err != nil {
		t.Fatal(err)
	}
	clock.advance(time.Second)
	if p := mustReserve(t, l, "chat-a", 1, 1); !p.Probe {
		t.Fatal("expected a probe")
	}
	reset := clock.now().Add(3 * time.Hour)
	if err := l.recordOutcome(other.Attempt, outcomeExhausted, 0, reset); err != nil {
		t.Fatal(err)
	}
	l = reopen(t, l, dir, clock, routing)
	a := l.state.account("paid")
	if a.Probe != "" || a.Cooldown.UntilMs != reset.UnixMilli() || a.Cooldown.Reason != cooldownExhausted {
		t.Fatalf("recovered account = %+v, want the probe cleared and the exhaustion kept", a)
	}

	clock.advance(3 * time.Hour)
	if p := mustReserve(t, l, "chat-a", 1, 1); !p.Probe {
		t.Fatal("expected a probe after the exhaustion")
	}
	l = reopen(t, l, dir, clock, routing)
	a = l.state.account("paid")
	if a.Probe != "" || a.Cooldown.Reason != cooldownProbeLost || a.Cooldown.UntilMs != clock.now().Add(probeLostCooldown).UnixMilli() {
		t.Fatalf("recovered account = %+v, want a probe-lost cooldown", a)
	}
}

func TestCredentialFailureSuppressesUntilTheReferenceChanges(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	h := mustReserve(t, l, "chat-a", 1, 1)
	if err := l.recordOutcome(h.Attempt, outcomeCredentialFailure, 0, time.Time{}); err != nil {
		t.Fatal(err)
	}
	clock.advance(24 * time.Hour)
	l = reopen(t, l, dir, clock, routing)
	_, err := l.reserve("chat-a", 1, 1)
	wantErr(t, "reserve after a credential failure", err, errAccountCredential)

	changed := testRouting("1000000", "1000000", "1000000")
	changed.Accounts[0].Credential = otherCredential
	if err := l.applyRouting(changed, "8"); err != nil {
		t.Fatal(err)
	}
	wantBindings := []ledgerBinding{{Credential: testCredential, Account: "paid"}, {Credential: otherCredential, Account: "paid"}}
	if !reflect.DeepEqual(l.state.Bindings, wantBindings) {
		t.Fatal("a reference change did not keep both reference bindings on the same account")
	}
	probe := mustReserve(t, l, "chat-a", 1, 1)
	if !probe.Probe {
		t.Fatal("a changed reference must re-enable exactly one probe")
	}
	_, err = l.reserve("chat-b", 1, 1)
	wantErr(t, "second reserve while probing", err, errAccountCooldown)
	if err := l.recordOutcome(probe.Attempt, outcomeOK, 0, time.Time{}); err != nil {
		t.Fatal(err)
	}
	mustReserve(t, l, "chat-b", 1, 1)
	if got, _ := l.state.outstanding("paid"); got != 9 {
		t.Fatalf("outstanding = %d, want every hold kept across the reference change", got)
	}
}

func TestClockRollbackRefusesReservations(t *testing.T) {
	clock := newLedgerClock()
	l := openTestLedger(t, t.TempDir(), clock, testRouting("1000000", "1000000", "1000000"))
	h := mustReserve(t, l, "chat-a", 1, 1)
	start := clock.now()
	clock.set(start.Add(-time.Hour))
	_, err := l.reserve("chat-a", 1, 1)
	wantErr(t, "reserve behind the last observed time", err, errClockRollback)
	if err := l.reconcile(h.Attempt, nil); err != nil {
		t.Fatalf("reconcile while the clock is behind: %v", err)
	}
	clock.set(start.Add(time.Second))
	mustReserve(t, l, "chat-a", 1, 1)
}

func TestCommitClockRollbackKeepsCurrentPeriodSpend(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("100", "100", "100")
	l := openTestLedger(t, dir, clock, routing)
	h := mustReserve(t, l, "chat-a", 0, 50)
	if err := l.reconcile(h.Attempt, nil); err != nil {
		t.Fatal(err)
	}
	_, err := l.reserve("chat-a", 0, 1)
	wantErr(t, "spent ceiling before rollback", err, errAccountCeiling)

	present := clock.now()
	past := present.AddDate(0, -1, 0)
	calls := 0
	l.now = func() time.Time {
		calls++
		// The outcome observes the present, then commit observes it.
		// A second sample inside commit would cross both period boundaries.
		if calls >= 3 {
			return past
		}
		return present
	}
	if err := l.recordOutcome(h.Attempt, outcomeTransient, 0, time.Time{}); err != nil {
		t.Fatal(err)
	}
	l.now = clock.now
	spent := l.state.account("paid").Spent
	_, err = l.reserve("chat-a", 0, 50)
	if spent[dayKey(present)] != "100" || spent[monthKey(present)] != "100" || !errors.Is(err, errAccountCeiling) {
		t.Fatalf("spend = %v, full-ceiling reserve = %v; want both periods at 100 and ceiling refusal", spent, err)
	}
	l = reopen(t, l, dir, clock, routing)
	_, err = l.reserve("chat-a", 0, 50)
	wantErr(t, "spent ceiling after reopen", err, errAccountCeiling)
}

func TestCredentialFailuresAcrossReferenceEditKeepCurrentSuppression(t *testing.T) {
	test := func(name string, currentFirst bool) {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			clock := newLedgerClock()
			routing := testRouting("1000000", "1000000", "1000000")
			l := openTestLedger(t, dir, clock, routing)
			old := mustReserve(t, l, "chat-a", 1, 1)
			routing.Accounts[0].Credential = otherCredential
			if err := l.applyRouting(routing, "8"); err != nil {
				t.Fatal(err)
			}
			current := mustReserve(t, l, "chat-a", 1, 1)
			attempts := []string{old.Attempt, current.Attempt}
			if currentFirst {
				attempts = []string{current.Attempt, old.Attempt}
			}
			for _, attempt := range attempts {
				if err := l.recordOutcome(attempt, outcomeCredentialFailure, 0, time.Time{}); err != nil {
					t.Fatal(err)
				}
				if attempt == current.Attempt {
					_, err := l.reserve("chat-a", 1, 1)
					wantErr(t, "current reference failure", err, errAccountCredential)
				}
			}
			assertSuppressed := func() {
				t.Helper()
				failure := l.state.account("paid").CredentialFailure
				if failure == nil || failure.Credential != otherCredential {
					t.Fatalf("failure = %+v, want the configured reference", failure)
				}
				_, err := l.reserve("chat-a", 1, 1)
				wantErr(t, "configured reference remains suppressed", err, errAccountCredential)
			}
			assertSuppressed()
			l = reopen(t, l, dir, clock, routing)
			assertSuppressed()
		})
	}
	test("current failure then obsolete failure", true)
	test("obsolete failure then current failure", false)
}

func TestCredentialFailuresDuringRetirementKeepConfiguredSuppression(t *testing.T) {
	test := func(name string, currentFirst bool, readdedCredential nodepolicy.CredentialRef) {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			clock := newLedgerClock()
			routing := testRouting("1000000", "1000000", "1000000")
			l := openTestLedger(t, dir, clock, routing)
			old := mustReserve(t, l, "chat-a", 1, 1)
			routing.Accounts[0].Credential = otherCredential
			if err := l.applyRouting(routing, "8"); err != nil {
				t.Fatal(err)
			}
			current := mustReserve(t, l, "chat-a", 1, 1)
			if err := l.applyRouting(nodepolicy.Routing{}, "9"); err != nil {
				t.Fatal(err)
			}
			retired, err := readAccountStore(filepath.Join(dir, accountStoreFileName))
			if err != nil {
				t.Fatal(err)
			}
			if retired.account("paid").LastConfiguredCredential != otherCredential {
				t.Fatal("retirement did not durably retain the configured reference")
			}
			attempts := []string{old.Attempt, current.Attempt}
			if currentFirst {
				attempts = []string{current.Attempt, old.Attempt}
			}
			for _, attempt := range attempts {
				if err := l.recordOutcome(attempt, outcomeCredentialFailure, 0, time.Time{}); err != nil {
					t.Fatal(err)
				}
			}
			// Reopen while retired as well: the identity must survive without
			// active routing to reconstruct it from.
			l = reopen(t, l, dir, clock, nodepolicy.Routing{})
			if err := l.ensureOpenLocked(); err != nil {
				t.Fatal(err)
			}
			if l.state.account("paid").LastConfiguredCredential != otherCredential {
				t.Fatal("retired reference lost on reopen")
			}
			routing.Accounts[0].Credential = readdedCredential
			if err := l.applyRouting(routing, "10"); err != nil {
				t.Fatal(err)
			}
			l = reopen(t, l, dir, clock, routing)
			failure := l.state.account("paid").CredentialFailure
			if failure == nil || failure.Credential != otherCredential {
				t.Fatal("configured reference failure lost during retirement")
			}
			grant, err := l.reserve("chat-a", 1, 1)
			if readdedCredential == otherCredential {
				wantErr(t, "re-added configured reference remains suppressed", err, errAccountCredential)
				if grant.Probe || l.state.account("paid").Probe != "" {
					t.Fatal("suppressed reference granted a probe")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !grant.Probe {
				t.Fatal("re-added different reference did not get a half-open probe")
			}
			if err := l.recordOutcome(grant.Attempt, outcomeOK, 0, time.Time{}); err != nil {
				t.Fatal(err)
			}
			if l.state.account("paid").CredentialFailure != nil {
				t.Fatal("successful probe did not clear credential suppression")
			}
		})
	}
	test("current then obsolete while retired, same reference returns", true, otherCredential)
	test("obsolete then current while retired, same reference returns", false, otherCredential)
	test("current then obsolete while retired, different reference returns", true, testCredential)
	test("obsolete then current while retired, different reference returns", false, testCredential)
}

func TestLastConfiguredCredentialStoreCompatibility(t *testing.T) {
	test := func(name string, credential nodepolicy.CredentialRef, wantInvalid bool) {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			clock := newLedgerClock()
			routing := testRouting("1000000", "1000000", "1000000")
			l := openTestLedger(t, dir, clock, routing)
			next := l.state.clone()
			next.account("paid").LastConfiguredCredential = credential
			data, err := encodeLedger(next)
			if err != nil {
				t.Fatal(err)
			}
			_, err = decodeLedger(data)
			if wantInvalid {
				wantErr(t, "invalid configured reference", err, errAccountStoreInvalid)
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := l.commitLocked(next); err != nil {
				t.Fatal(err)
			}
			l = reopen(t, l, dir, clock, routing)
			if l.state.account("paid").LastConfiguredCredential != testCredential {
				t.Fatal("routing did not populate the absent configured reference")
			}
		})
	}
	test("legacy store without configured reference loads", "", false)
	test("invalid configured reference is refused", "invalid reference", true)
}

// Usage above the reserved bound freezes the account and quarantines the
// deployment identity for good: restart, rename, retirement and re-addition
// keep it, and only a new qualification revision is admitted.
func TestUsageAboveBoundFreezesAndQuarantines(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	routing.Accounts = append(routing.Accounts, nodepolicy.Account{
		ID: "spare", Credential: otherCredential, Billing: nodepolicy.BillingPaid,
		PerRequestCeiling: "1000000", DailyCeiling: "1000000", MonthlyCeiling: "1000000",
	})
	routing.Deployments = append(routing.Deployments, testDeployment("spare-chat", "spare"))
	// chat-b is another identity on the same account, so the freeze shows.
	routing.Deployments[1].Qualification = &nodepolicy.QualificationRef{ID: "provider-q", Revision: "5"}
	l := openTestLedger(t, dir, clock, routing)
	h := mustReserve(t, l, "chat-a", 100, 200)
	wantErr(t, "reconcile above the bound", l.reconcile(h.Attempt, &tokenUsage{InputTokens: 100, OutputTokens: 201}), errUsageAboveBound)
	if got := l.state.hold(h.Attempt); got.State != holdReconciled || got.Actual != "502" || !got.QualificationChecked {
		t.Fatalf("hold = %+v, want reconciled at the actual 502 with qualification classified", got)
	}

	l = reopen(t, l, dir, clock, routing)
	_, err := l.reserve("chat-b", 1, 1)
	wantErr(t, "reserve on the frozen account", err, errAccountFrozen)
	_, err = l.reserve("spare-chat", 1, 1)
	wantErr(t, "reserve the same identity on another account", err, errDeploymentQuarantined)

	renamed := routing
	renamed.Deployments = []nodepolicy.Deployment{testDeployment("renamed", "spare")}
	if err := l.applyRouting(nodepolicy.Routing{}, "8"); err != nil {
		t.Fatal(err)
	}
	if err := l.applyRouting(renamed, "9"); err != nil {
		t.Fatal(err)
	}
	_, err = l.reserve("renamed", 1, 1)
	wantErr(t, "reserve the renamed, re-added identity", err, errDeploymentQuarantined)

	requalified := renamed
	d := testDeployment("renamed", "spare")
	d.Qualification = &nodepolicy.QualificationRef{ID: "provider-q", Revision: "2"}
	requalified.Deployments = []nodepolicy.Deployment{d}
	if err := l.applyRouting(requalified, "10"); err != nil {
		t.Fatal(err)
	}
	mustReserve(t, l, "renamed", 1, 1)
}

// When above-bound usage costs more than the signed 64-bit range holds, or
// more than spend can add, the hold stays outstanding at its bound and the
// account is still frozen and the identity quarantined.
func TestUsageAboveBoundThatOverflowsKeepsTheHoldOutstanding(t *testing.T) {
	huge := func() nodepolicy.Routing {
		r := testRouting("9223372036854775807", "9223372036854775807", "9223372036854775807")
		for i := range r.Deployments {
			r.Deployments[i].ContextTokens, r.Deployments[i].MaxOutputTokens = 2_000_000, 1_000_000
			r.Deployments[i].Prices.OutputPerMillion = "9000000000000000000"
		}
		return r
	}
	t.Run("charge", func(t *testing.T) {
		l := openTestLedger(t, t.TempDir(), newLedgerClock(), huge())
		h := mustReserve(t, l, "chat-a", 0, 1)
		wantErr(t, "reconcile", l.reconcile(h.Attempt, &tokenUsage{OutputTokens: 1_100_000}), errUsageAboveBound)
		if got := l.state.hold(h.Attempt); got.State != holdReserved || got.Actual != "" {
			t.Fatalf("hold = %+v, want it outstanding with no actual", got)
		}
		if l.state.account("paid").Frozen == nil || len(l.state.Quarantine) != 1 {
			t.Fatal("the account is not frozen or the identity not quarantined")
		}
	})
	t.Run("spend", func(t *testing.T) {
		l := openTestLedger(t, t.TempDir(), newLedgerClock(), huge())
		first := mustReserve(t, l, "chat-a", 0, 600_000)
		if err := l.reconcile(first.Attempt, &tokenUsage{OutputTokens: 600_000}); err != nil {
			t.Fatal(err)
		}
		h := mustReserve(t, l, "chat-a", 0, 1)
		wantErr(t, "reconcile", l.reconcile(h.Attempt, &tokenUsage{OutputTokens: 500_000}), errUsageAboveBound)
		if got := l.state.hold(h.Attempt); got.State != holdReserved {
			t.Fatalf("hold = %+v, want it outstanding", got)
		}
		if l.state.account("paid").Frozen == nil || len(l.state.Quarantine) != 1 {
			t.Fatal("the account is not frozen or the identity not quarantined")
		}
	})
}

func TestInBoundReconcileOverflowChangesNothing(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("9223372036854775807", "9223372036854775807", "9223372036854775807")
	routing.Deployments[0].ContextTokens = 2_000_000
	routing.Deployments[0].MaxOutputTokens = 1_000_000
	routing.Deployments[0].Prices.OutputPerMillion = "9000000000000000000"
	l := openTestLedger(t, dir, clock, routing)
	first := mustReserve(t, l, "chat-a", 0, 600_000)
	second := mustReserve(t, l, "chat-a", 0, 300_000)
	wantErr(t, "above-bound first reconcile", l.reconcile(first.Attempt, &tokenUsage{OutputTokens: 1_000_000}), errUsageAboveBound)
	before, memory := readLedgerBytes(t, dir), l.state.clone()
	wantErr(t, "in-bound spend overflow", l.reconcile(second.Attempt, &tokenUsage{OutputTokens: 300_000}), errMoneyOverflow)
	if !reflect.DeepEqual(l.state, memory) || string(readLedgerBytes(t, dir)) != string(before) {
		t.Fatal("an overflowing in-bound reconcile changed memory or disk")
	}
}

func TestRetainedReconcilesAreBoundedAndEvictedAttemptsAreUnknown(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	first := mustReserve(t, l, "chat-a", 0, 1)
	if err := l.reconcile(first.Attempt, nil); err != nil {
		t.Fatal(err)
	}
	if err := l.recordOutcome(first.Attempt, outcomeOK, 0, time.Time{}); err != nil {
		t.Fatal(err)
	}
	if err := l.recordQualification(first.Attempt); err != nil {
		t.Fatal(err)
	}
	for range maxResolvedHolds {
		h := mustReserve(t, l, "chat-b", 0, 1)
		if h.Attempt == first.Attempt {
			t.Fatal("reserve reused an attempt id")
		}
		if err := l.reconcile(h.Attempt, nil); err != nil {
			t.Fatal(err)
		}
		if err := l.recordOutcome(h.Attempt, outcomeOK, 0, time.Time{}); err != nil {
			t.Fatal(err)
		}
		if err := l.recordQualification(h.Attempt); err != nil {
			t.Fatal(err)
		}
	}
	if len(l.state.Holds) != maxResolvedHolds || l.state.hold(first.Attempt) != nil {
		t.Fatal("resolved holds were not evicted oldest first")
	}
	before := readLedgerBytes(t, dir)
	wantErr(t, "evicted reconcile", l.reconcile(first.Attempt, nil), errAttemptUnknown)
	wantErr(t, "evicted outcome", l.recordOutcome(first.Attempt, outcomeOK, 0, time.Time{}), errAttemptUnknown)
	if string(readLedgerBytes(t, dir)) != string(before) {
		t.Fatal("an evicted attempt changed the store")
	}
	l = reopen(t, l, dir, clock, routing)
	want := nanoUSDString(2 * (maxResolvedHolds + 1))
	if got := l.state.account("paid").Spent[dayKey(clock.now())]; got != want {
		t.Fatalf("spend = %s, want %s including evicted holds", got, want)
	}
}

func completeOtherLedgerAttempts(t *testing.T, l *accountLedger) {
	t.Helper()
	for range maxResolvedHolds + 1 {
		h := mustReserve(t, l, "chat-b", 0, 1)
		if err := l.reconcile(h.Attempt, nil); err != nil {
			t.Fatal(err)
		}
		if err := l.recordOutcome(h.Attempt, outcomeOK, 0, time.Time{}); err != nil {
			t.Fatal(err)
		}
		if err := l.recordQualification(h.Attempt); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPendingOutcomeSurvivesCompletedHoldRetention(t *testing.T) {
	test := func(name string, outcome accountOutcome, want error) {
		t.Run(name, func(t *testing.T) {
			clock := newLedgerClock()
			l := openTestLedger(t, t.TempDir(), clock, testRouting("1000000", "1000000", "1000000"))
			first := mustReserve(t, l, "chat-a", 0, 1)
			if err := l.reconcile(first.Attempt, nil); err != nil {
				t.Fatal(err)
			}
			if err := l.recordQualification(first.Attempt); err != nil {
				t.Fatal(err)
			}
			completeOtherLedgerAttempts(t, l)
			if err := l.recordOutcome(first.Attempt, outcome, time.Minute, clock.now().Add(time.Hour)); err != nil {
				t.Fatalf("first restrictive outcome after retention window: %v", err)
			}
			_, err := l.reserve("chat-a", 0, 1)
			wantErr(t, "late restriction must apply", err, want)
		})
	}
	test("credential failure", outcomeCredentialFailure, errAccountCredential)
	test("rate limit", outcomeRateLimited, errAccountCooldown)
	test("exhaustion", outcomeExhausted, errAccountCooldown)
}

func TestPendingQualificationSurvivesCompletedHoldRetention(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	first := mustReserve(t, l, "chat-a", 0, 1)
	key := l.state.hold(first.Attempt).key()
	if err := l.reconcile(first.Attempt, nil); err != nil {
		t.Fatal(err)
	}
	if err := l.recordOutcome(first.Attempt, outcomeOK, 0, time.Time{}); err != nil {
		t.Fatal(err)
	}
	completeOtherLedgerAttempts(t, l)
	routing.Deployments[0].Endpoint = "https://other.example.com/v1/chat/completions"
	routing.Deployments[1].ID = "renamed"
	if err := l.applyRouting(routing, "8"); err != nil {
		t.Fatal(err)
	}
	if err := l.quarantine(first.Attempt); err != nil {
		t.Fatalf("first qualification failure after retention window: %v", err)
	}
	if !l.state.quarantined(key) {
		t.Fatal("quarantine lost the immutable reservation identity")
	}
	_, err := l.reserve("renamed", 0, 1)
	wantErr(t, "reserved identity after delayed quarantine", err, errDeploymentQuarantined)
	mustReserve(t, l, "chat-a", 0, 1)
	l = reopen(t, l, dir, clock, routing)
	_, err = l.reserve("renamed", 0, 1)
	wantErr(t, "delayed quarantine after reopen", err, errDeploymentQuarantined)
}

func TestUnfinishedReconciledHoldsCountAgainstAdmissionCapacity(t *testing.T) {
	test := func(name string, outcome accountOutcome, qualificationChecked bool) {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			clock := newLedgerClock()
			routing := testRouting("1000000", "1000000", "1000000")
			l := openTestLedger(t, dir, clock, routing)
			first := mustReserve(t, l, "chat-a", 0, 1)
			next := l.state.clone()
			seed := *next.hold(first.Attempt)
			seed.State, seed.Actual = holdReconciled, seed.Reserved
			seed.Outcome, seed.QualificationChecked = outcome, qualificationChecked
			next.Holds[0] = seed
			for i := 1; i < maxLedgerHolds-maxResolvedHolds; i++ {
				seed.Attempt = "seed-" + strconv.Itoa(i)
				next.Holds = append(next.Holds, seed)
			}
			if err := addSpend(next.account("paid"), clock.now(), int64(2*len(next.Holds))); err != nil {
				t.Fatal(err)
			}
			if err := l.commitLocked(next); err != nil {
				t.Fatal(err)
			}
			before := readLedgerBytes(t, dir)
			_, err := l.reserve("chat-a", 0, 1)
			wantErr(t, "unfinished terminal processing fills admission capacity", err, errLedgerFull)
			if len(l.state.Holds) != maxLedgerHolds-maxResolvedHolds || string(readLedgerBytes(t, dir)) != string(before) {
				t.Fatal("full admission evicted unfinished holds or changed the store")
			}
			// Completing the missing classification frees one admission slot.
			if outcome == "" {
				if err := l.recordOutcome(first.Attempt, outcomeOK, 0, time.Time{}); err != nil {
					t.Fatal(err)
				}
			} else if err := l.recordQualification(first.Attempt); err != nil {
				t.Fatal(err)
			}
			mustReserve(t, l, "chat-a", 0, 1)
			l = reopen(t, l, dir, clock, routing)
			_, err = l.reserve("chat-a", 0, 1)
			wantErr(t, "unfinished capacity after reopen", err, errLedgerFull)
		})
	}
	test("pending account outcome", "", true)
	test("pending qualification classification", outcomeOK, false)
}

func TestQuarantineWriteFailureStopsAllReservations(t *testing.T) {
	dir := t.TempDir()
	l := openTestLedger(t, dir, newLedgerClock(), testRouting("1000000", "1000000", "1000000"))
	h := mustReserve(t, l, "chat-a", 1, 1)
	before, memory := readLedgerBytes(t, dir), l.state.clone()
	l.write = durablefile.Replacer{Replace: func(string, string) error { return errors.New("injected") }}.WriteFile
	if err := l.quarantine(h.Attempt); err == nil {
		t.Fatal("quarantine succeeded with failing storage")
	}
	if !reflect.DeepEqual(l.state, memory) || string(readLedgerBytes(t, dir)) != string(before) {
		t.Fatal("failed quarantine changed memory or disk")
	}
	l.write = durablefile.WriteFile
	_, err := l.reserve("chat-b", 1, 1)
	wantErr(t, "reserve after quarantine storage recovers", err, errLedgerUnavailable)
}

func TestQualificationCompletionPersistsForEachAttempt(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	passed := mustReserve(t, l, "chat-a", 0, 1)
	failed := mustReserve(t, l, "chat-a", 0, 1)
	sameKey := mustReserve(t, l, "chat-b", 0, 1)
	if err := l.recordQualification(passed.Attempt); err != nil {
		t.Fatal(err)
	}
	if err := l.quarantine(failed.Attempt); err != nil {
		t.Fatal(err)
	}
	// The key is already quarantined, but this attempt still needs its own
	// durable completion marker.
	if err := l.quarantine(sameKey.Attempt); err != nil {
		t.Fatal(err)
	}
	before := readLedgerBytes(t, dir)
	if err := l.recordQualification(passed.Attempt); err != nil {
		t.Fatal(err)
	}
	if err := l.quarantine(sameKey.Attempt); err != nil {
		t.Fatal(err)
	}
	if string(readLedgerBytes(t, dir)) != string(before) {
		t.Fatal("duplicate classification changed the store")
	}
	l = reopen(t, l, dir, clock, routing)
	for _, attempt := range []string{passed.Attempt, failed.Attempt, sameKey.Attempt} {
		h := l.state.hold(attempt)
		if h == nil || !h.QualificationChecked {
			t.Fatalf("attempt %s lost its classification on reopen", attempt)
		}
	}
}

// Quarantine names the identity the attempt was reserved under, whatever the
// deployment has since become.
func TestQuarantineFollowsTheReservedIdentity(t *testing.T) {
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, t.TempDir(), clock, routing)
	h := mustReserve(t, l, "chat-a", 1, 1)

	edited := testRouting("1000000", "1000000", "1000000")
	edited.Deployments[0].Endpoint = "https://other.example.com/v1/chat/completions"
	edited.Deployments[1].ID = "renamed"
	if err := l.applyRouting(edited, "8"); err != nil {
		t.Fatal(err)
	}
	if err := l.quarantine(h.Attempt); err != nil {
		t.Fatal(err)
	}
	_, err := l.reserve("renamed", 1, 1)
	wantErr(t, "reserve the reserved identity under a new name", err, errDeploymentQuarantined)
	mustReserve(t, l, "chat-a", 1, 1)
	wantErr(t, "quarantine an unknown attempt", l.quarantine("nobody-1"), errAttemptUnknown)
}

// A restrictive transition that cannot be recorded stops all admission for
// the process lifetime, even after storage recovers.
func TestUnrecordedRestrictionLatchesTheLedger(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	h := mustReserve(t, l, "chat-a", 1, 1)
	l.write = durablefile.Replacer{Replace: func(string, string) error { return errors.New("injected") }}.WriteFile
	if err := l.recordOutcome(h.Attempt, outcomeCredentialFailure, 0, time.Time{}); err == nil {
		t.Fatal("recordOutcome succeeded with failing storage")
	}
	l.write = durablefile.WriteFile
	_, err := l.reserve("chat-a", 1, 1)
	wantErr(t, "reserve after an unrecorded restriction", err, errLedgerUnavailable)
	l = reopen(t, l, dir, clock, routing)
	mustReserve(t, l, "chat-a", 1, 1)
}

// A credential reference stays bound to the first account that used it:
// delete and re-add under the same ID continues it, a new ID or a rename with
// that reference is refused, also after a restart, and a refused account never
// blocks policy activation.
func TestCredentialBindingsOutliveTheConfiguration(t *testing.T) {
	dir := t.TempDir()
	clock := newLedgerClock()
	routing := testRouting("1000000", "1000", "1000000")
	l := openTestLedger(t, dir, clock, routing)
	mustReserve(t, l, "chat-a", 100, 200)

	if err := l.applyRouting(nodepolicy.Routing{}, "8"); err != nil {
		t.Fatal(err)
	}
	renamed := testRouting("1000000", "1000000", "1000000")
	renamed.Accounts[0].ID = "fresh"
	renamed.Deployments = []nodepolicy.Deployment{testDeployment("chat-a", "fresh")}
	if err := l.applyRouting(renamed, "9"); err != nil {
		t.Fatalf("applyRouting with a conflicted account: %v", err)
	}
	_, err := l.reserve("chat-a", 1, 1)
	wantErr(t, "reserve on a new ID with a bound reference", err, errAccountConflict)

	l = reopen(t, l, dir, clock, renamed)
	_, err = l.reserve("chat-a", 1, 1)
	wantErr(t, "reserve on a new ID after a restart", err, errAccountConflict)

	if err := l.applyRouting(routing, "10"); err != nil {
		t.Fatal(err)
	}
	mustReserve(t, l, "chat-b", 100, 200)
	_, err = l.reserve("chat-b", 0, 1)
	wantErr(t, "re-added account keeps its outstanding holds", err, errAccountCeiling)
}

// A node/set-policy naming accounts is acknowledged whatever the ledger's
// state, and a proxy built by NewProxy has no store to touch.
func TestPolicyActivationNeverWaitsOnTheLedger(t *testing.T) {
	pol := nodepolicy.Default()
	pol.Revision = "3"
	pol.Routing.Accounts = []nodepolicy.Account{{ID: "paid", Credential: testCredential, Billing: nodepolicy.BillingPaid,
		PerRequestCeiling: "1", DailyCeiling: "1", MonthlyCeiling: "1"}}

	p, w := policyProxy(t)
	if ack := ackOf(t, rpc(t, p, w, nodepolicy.MethodSetPolicy, pol)); ack.Revision != "3" {
		t.Fatalf("ack = %+v, want revision 3", ack)
	}
	if p.accounts.storePath != "" {
		t.Fatal("NewProxy configured an account store")
	}
	_, err := p.accounts.reserve("chat-a", 1, 1)
	wantErr(t, "reserve without a store", err, errLedgerUnconfigured)

	dir := t.TempDir()
	holder := newAccountLedger()
	holder.setDir(dir, nil)
	t.Cleanup(holder.close)
	if err := holder.applyRouting(pol.Routing, "1"); err != nil {
		t.Fatal(err)
	}
	p2, w2 := policyProxy(t)
	p2.accounts.setDir(dir, nil)
	t.Cleanup(p2.accounts.close)
	if ack := ackOf(t, rpc(t, p2, w2, nodepolicy.MethodSetPolicy, pol)); ack.Revision != "3" {
		t.Fatalf("ack with a locked store = %+v, want revision 3", ack)
	}
	if got := p2.admission.currentPolicy().Revision; got != "3" {
		t.Fatalf("active revision = %s, want 3", got)
	}
	holder.close()
	pol.Revision = "4"
	if ack := ackOf(t, rpc(t, p2, w2, nodepolicy.MethodSetPolicy, pol)); ack.Revision != "4" {
		t.Fatalf("ack = %+v, want revision 4", ack)
	}
	if p2.accounts.lock == nil || len(p2.accounts.state.Bindings) != 1 {
		t.Fatal("the proxy did not take the released store and record its binding")
	}
}
