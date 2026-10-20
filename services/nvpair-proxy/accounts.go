// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// The account ledger is the proxy's single durable writer of provider account
// state: money holds and reconciliation in exact nano-USD, per-account spend in
// UTC day and month periods, credential-reference bindings, cooldowns with one
// half-open probe, credential-failure suppression, freezes and deployment
// qualification quarantine. One mutex serializes every operation from lazy
// open to adoption; an exclusive OS lock on a separate file keeps a second
// process off the same store for this process's lifetime.
//
// Every refusal fails closed: an unavailable, locked, corrupt or unwritable
// store, an arithmetic overflow, a clock behind the last observed time and
// any unrecorded restrictive transition refuse new reservations. Nothing here
// resolves an environment credential. The private Chat attempt seam below is
// unwired; its future caller supplies the selected-reference resolver and guard.

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/bits"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"nvpair-shared/durablefile"
	"nvpair-shared/nodepolicy"
)

const (
	nanoUSDTokenScale   = 1_000_000
	defaultCooldown     = 60 * time.Second
	minCooldown         = time.Second
	maxCooldown         = time.Hour
	maxExhaustionWindow = 32 * 24 * time.Hour
	probeLostCooldown   = 60 * time.Second
)

var (
	errLedgerUnavailable     = errors.New("the account ledger is unavailable")
	errLedgerUnconfigured    = errors.New("no account store is configured")
	errLedgerFull            = errors.New("the account ledger holds too many records")
	errClockRollback         = errors.New("the clock is behind the account ledger's last observed time")
	errUnknownDeployment     = errors.New("the deployment is not configured")
	errDeploymentUnqualified = errors.New("the deployment has no endpoint, protocol, qualification or USD prices")
	errDeploymentQuarantined = errors.New("the deployment's qualification is quarantined")
	errAccountConflict       = errors.New("the account's credential reference is bound to another account")
	errAccountFrozen         = errors.New("the account is frozen")
	errAccountCredential     = errors.New("the account's credential failed")
	errAccountCooldown       = errors.New("the account is cooling down")
	errAccountCeiling        = errors.New("the reservation exceeds an account ceiling")
	errTokenBounds           = errors.New("the token bounds do not fit the deployment")
	errMoneyOverflow         = errors.New("the amount exceeds the signed 64-bit nano-USD range")
	errAttemptUnknown        = errors.New("the attempt is not one this process can resolve")
	errAttemptConflict       = errors.New("the attempt was already resolved differently")
	errUsageAboveBound       = errors.New("usage exceeded the reserved bound; the account is frozen and the deployment quarantined")
	errInvalidOutcome        = errors.New("invalid account outcome")
)

// accountLedger owns the account store. The zero paths make every operation
// report the ledger unavailable, which is what in-process tests and a proxy
// without appdir get.
type accountLedger struct {
	mu sync.Mutex

	storePath, lockPath string
	pathErr             error
	now                 func() time.Time
	write               func(path string, data []byte, perm os.FileMode) error

	lock  *os.File // held from a successful open until the process ends
	state ledgerState
	// nonce and nextAttempt allocate attempt IDs: nonce is fresh per open.
	nonce       string
	nextAttempt uint64

	// broken latches for the process lifetime once a restrictive transition
	// could not be recorded: the durable state no longer says what memory
	// knows, so nothing may be admitted until a restart reads it again.
	broken error
	// unconfirmed is set when bytes landed without confirmed durability; a
	// complete durable rewrite must succeed before the next reservation.
	unconfirmed bool

	routing        nodepolicy.Routing
	policyRevision string
	// synced says the bindings of the active routing are durably recorded;
	// conflicted names its accounts whose reference is bound elsewhere.
	synced     bool
	conflicted map[string]bool
}

func newAccountLedger() *accountLedger {
	return &accountLedger{now: time.Now, write: durablefile.WriteFile}
}

// setDir places the store and its lock in dir. Called once before use.
func (l *accountLedger) setDir(dir string, err error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err != nil {
		l.pathErr = err
		return
	}
	l.storePath = filepath.Join(dir, accountStoreFileName)
	l.lockPath = filepath.Join(dir, accountLockFileName)
}

// close releases the lock. Production never calls it: the lock lives until
// the process ends.
func (l *accountLedger) close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.lock != nil {
		l.lock.Close()
		l.lock = nil
	}
}

// ensureOpenLocked opens the store on first use and retries on later calls
// after a failure. A lock taken before a failed read is released again, so a
// retry never stacks handles.
func (l *accountLedger) ensureOpenLocked() error {
	if l.broken != nil {
		return l.broken
	}
	if l.lock != nil {
		return nil
	}
	if l.pathErr != nil {
		return fmt.Errorf("%w: %v", errLedgerUnavailable, l.pathErr)
	}
	if l.storePath == "" {
		return errLedgerUnconfigured
	}
	if err := os.MkdirAll(filepath.Dir(l.storePath), 0o700); err != nil {
		return fmt.Errorf("%w: %v", errLedgerUnavailable, err)
	}
	lock, err := lockAccountStore(l.lockPath)
	if err != nil {
		return fmt.Errorf("%w: %w", errLedgerUnavailable, err)
	}
	state, err := readAccountStore(l.storePath)
	if err != nil {
		lock.Close()
		return fmt.Errorf("%w: %w", errLedgerUnavailable, err)
	}
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		lock.Close()
		return fmt.Errorf("%w: %v", errLedgerUnavailable, err)
	}
	l.recoverLocked(&state)
	l.lock, l.state, l.nonce = lock, state, hex.EncodeToString(b[:])
	l.synced = false
	return nil
}

// recoverLocked applies what an earlier process left unresolved: its holds
// become uncertain (never assumed unsent), and a probe it held becomes a
// cooldown no shorter than the one already recorded. The change is persisted
// by the next write.
func (l *accountLedger) recoverLocked(s *ledgerState) {
	nowMs := l.now().UnixMilli()
	for i := range s.Holds {
		if s.Holds[i].State == holdReserved {
			s.Holds[i].State = holdUncertain
		}
	}
	for i := range s.Accounts {
		a := &s.Accounts[i]
		if a.Probe == "" {
			continue
		}
		a.Probe = ""
		a.Cooldown = laterCooldown(a.Cooldown, &accountCooldown{UntilMs: nowMs + probeLostCooldown.Milliseconds(), Reason: cooldownProbeLost})
	}
}

// commitLocked durably replaces the store with next and adopts it. A write
// that fails before replacement leaves memory and disk unchanged. One that
// lands without confirmed durability is adopted, sets the unconfirmed latch,
// and still fails, so no permission rests on it.
func (l *accountLedger) commitLocked(next ledgerState) error {
	now := l.now()
	nowMs := now.UnixMilli()
	if nowMs >= next.LastObservedMs {
		next.LastObservedMs = nowMs
		prunePeriods(&next, now)
	}
	pruneResolved(&next)
	data, err := encodeLedger(next)
	if err != nil {
		return err
	}
	err = l.write(l.storePath, data, 0o600)
	if err != nil && !errors.Is(err, durablefile.ErrReplaced) {
		return err
	}
	l.state = next
	l.unconfirmed = err != nil
	return err
}

// commitRestrictiveLocked commits a transition that restricts admission. If it
// cannot be confirmed durable, the ledger stops admitting for the rest of the
// process lifetime.
func (l *accountLedger) commitRestrictiveLocked(next ledgerState) error {
	err := l.commitLocked(next)
	if err != nil {
		l.broken = fmt.Errorf("%w: a restrictive account transition was not recorded: %v", errLedgerUnavailable, err)
		slog.Error("account ledger stopped admitting: a restrictive transition was not durably recorded", "reason", "restrictive-transition-not-durable")
	}
	return err
}

// applyRouting adopts the activated routing. With at least one account it
// opens the ledger and records the bindings. Its error never blocks policy
// activation: spending fails closed instead.
func (l *accountLedger) applyRouting(routing nodepolicy.Routing, revision string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.routing, l.policyRevision = routing, revision
	l.synced = false
	if len(routing.Accounts) == 0 && l.lock == nil {
		return nil
	}
	if err := l.ensureOpenLocked(); err != nil {
		return err
	}
	return l.syncBindingsLocked()
}

// syncBindingsLocked binds every configured account's reference to its ID if
// unbound, creates missing account records, and marks an account whose
// reference is bound to another ID as conflicted. New bindings are written
// before any reservation can use them.
func (l *accountLedger) syncBindingsLocked() error {
	if l.synced {
		return nil
	}
	next := l.state.clone()
	conflicted := map[string]bool{}
	changed := false
	for _, a := range l.routing.Accounts {
		bound := ""
		for _, b := range next.Bindings {
			if b.Credential == a.Credential {
				bound = b.Account
			}
		}
		if bound != "" && bound != a.ID {
			conflicted[a.ID] = true
			continue
		}
		if next.account(a.ID) == nil {
			if len(next.Accounts) >= maxLedgerAccounts {
				conflicted[a.ID] = true
				continue
			}
			next.Accounts = append(next.Accounts, ledgerAccount{ID: a.ID, Spent: map[string]nodepolicy.NanoUSD{}})
			changed = true
		}
		if bound == "" {
			if len(next.Bindings) >= maxLedgerBindings {
				conflicted[a.ID] = true
				continue
			}
			next.Bindings = append(next.Bindings, ledgerBinding{Credential: a.Credential, Account: a.ID})
			changed = true
		}
		rec := next.account(a.ID)
		if rec.LastConfiguredCredential != a.Credential {
			rec.LastConfiguredCredential = a.Credential
			changed = true
		}
	}
	if changed {
		if err := l.commitLocked(next); err != nil {
			return err
		}
	}
	for id := range conflicted {
		slog.Warn("provider account refused: its credential reference is bound to another account", "account", id)
	}
	l.conflicted, l.synced = conflicted, true
	return nil
}

// admissibleLocked runs the checks every reservation shares before its own.
func (l *accountLedger) admissibleLocked() error {
	if err := l.ensureOpenLocked(); err != nil {
		return err
	}
	if l.unconfirmed {
		// Rewrite the adopted state until a write is confirmed durable.
		if err := l.commitLocked(l.state.clone()); err != nil {
			return fmt.Errorf("%w: %v", errLedgerUnavailable, err)
		}
	}
	return l.syncBindingsLocked()
}

func (s *ledgerState) account(id string) *ledgerAccount {
	for i := range s.Accounts {
		if s.Accounts[i].ID == id {
			return &s.Accounts[i]
		}
	}
	return nil
}

func (s *ledgerState) hold(attempt string) *ledgerHold {
	for i := range s.Holds {
		if s.Holds[i].Attempt == attempt {
			return &s.Holds[i]
		}
	}
	return nil
}

func (s *ledgerState) quarantined(k quarantineKey) bool {
	for _, q := range s.Quarantine {
		if q.quarantineKey == k {
			return true
		}
	}
	return false
}

// quarantine adds k unless present. It reports false when the record cap is
// reached, which the caller treats as an unrecorded restrictive transition.
func (s *ledgerState) quarantine(k quarantineKey, reason string, nowMs int64) bool {
	if s.quarantined(k) {
		return true
	}
	if len(s.Quarantine) >= maxLedgerQuarantine {
		return false
	}
	s.Quarantine = append(s.Quarantine, quarantineEntry{quarantineKey: k, Reason: reason, SinceMs: nowMs})
	return true
}

// outstanding sums an account's unresolved holds, whatever period they were
// reserved in.
func (s *ledgerState) outstanding(account string) (int64, error) {
	var sum int64
	for _, h := range s.Holds {
		if h.Account != account || !h.outstanding() {
			continue
		}
		v, err := h.Reserved.Int64()
		if err != nil {
			return 0, err
		}
		if sum, err = addNanoUSD(sum, v); err != nil {
			return 0, err
		}
	}
	return sum, nil
}

// accountHold is what a caller gets once a hold is durable.
type accountHold struct {
	Attempt  string
	Reserved nodepolicy.NanoUSD
	Probe    bool
}

// reserve durably holds the upper-bound cost of one attempt on deployment
// before it is sent. It returns only after the hold is confirmed durable.
func (l *accountLedger) reserve(deploymentID string, inputTokens, outputTokens int64) (accountHold, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.reserveLocked(deploymentID, inputTokens, outputTokens)
}

func (l *accountLedger) reserveLocked(deploymentID string, inputTokens, outputTokens int64) (accountHold, error) {
	hold, err := l.reserveLimitedLocked(deploymentID, inputTokens, outputTokens, math.MaxInt64)
	if err != nil {
		return accountHold{}, err
	}
	return hold, nil
}

func (l *accountLedger) reserveLimitedLocked(deploymentID string, inputTokens, outputTokens, remaining int64) (accountHold, error) {
	if err := l.admissibleLocked(); err != nil {
		return accountHold{}, err
	}
	now := l.now()
	nowMs := now.UnixMilli()
	if nowMs < l.state.LastObservedMs {
		return accountHold{}, errClockRollback
	}
	i := slices.IndexFunc(l.routing.Deployments, func(d nodepolicy.Deployment) bool { return d.ID == deploymentID })
	if i < 0 {
		return accountHold{}, errUnknownDeployment
	}
	d := l.routing.Deployments[i]
	j := slices.IndexFunc(l.routing.Accounts, func(a nodepolicy.Account) bool { return a.ID == d.Account })
	if j < 0 {
		return accountHold{}, errUnknownDeployment
	}
	acct := l.routing.Accounts[j]
	if l.conflicted[acct.ID] {
		return accountHold{}, errAccountConflict
	}
	if !validLedgerText(d.Endpoint) || !validLedgerText(d.Model) || d.Protocol == "" || d.Qualification == nil || d.Prices == nil ||
		d.Prices.Currency != nodepolicy.CurrencyUSD || d.Prices.Unit != nodepolicy.PriceUnit {
		return accountHold{}, errDeploymentUnqualified
	}
	key := quarantineKey{Endpoint: d.Endpoint, Model: d.Model, Qualification: *d.Qualification}
	if l.state.quarantined(key) {
		return accountHold{}, errDeploymentQuarantined
	}
	if inputTokens < 0 || outputTokens < 1 || outputTokens > int64(d.MaxOutputTokens) || inputTokens > int64(d.ContextTokens)-outputTokens {
		return accountHold{}, errTokenBounds
	}
	amount, err := reservationAmount(inputTokens, outputTokens, d.Prices.InputPerMillion, d.Prices.OutputPerMillion)
	if err != nil {
		return accountHold{}, err
	}
	if remaining < 0 || amount > remaining {
		return accountHold{}, errRoutingSpend
	}

	next := l.state.clone()
	rec := next.account(acct.ID)
	if rec == nil {
		return accountHold{}, errLedgerUnavailable
	}
	if rec.Frozen != nil {
		return accountHold{}, errAccountFrozen
	}
	halfOpen := false
	if f := rec.CredentialFailure; f != nil {
		if f.Credential == acct.Credential {
			return accountHold{}, errAccountCredential
		}
		halfOpen = true
	}
	if c := rec.Cooldown; c != nil {
		if nowMs < c.UntilMs {
			return accountHold{}, errAccountCooldown
		}
		halfOpen = true
	}
	if halfOpen && rec.Probe != "" {
		return accountHold{}, errAccountCooldown
	}
	if err := checkCeilings(next, rec, acct, amount, now); err != nil {
		return accountHold{}, err
	}
	// Pending money, outcome or qualification processing must retain its
	// immutable hold. Leave room for the bounded completed-history window.
	if next.unfinishedCount() >= maxLedgerHolds-maxResolvedHolds {
		return accountHold{}, errLedgerFull
	}
	attempt := l.newAttemptLocked()
	if halfOpen {
		rec.Probe = attempt
	}
	next.Holds = append(next.Holds, ledgerHold{
		Attempt: attempt, Account: acct.ID, Deployment: d.ID, Endpoint: d.Endpoint, Model: d.Model,
		Qualification: *d.Qualification, Credential: acct.Credential, PolicyRevision: l.policyRevision,
		Currency: d.Prices.Currency, Unit: d.Prices.Unit,
		InputPerMillion: d.Prices.InputPerMillion, OutputPerMillion: d.Prices.OutputPerMillion,
		InputTokens: inputTokens, OutputTokens: outputTokens, Reserved: nanoUSDString(amount),
		ReservedAtMs: nowMs, Generation: rec.Generation, Probe: halfOpen, State: holdReserved,
	})
	if err := l.commitLocked(next); err != nil {
		if errors.Is(err, durablefile.ErrReplaced) {
			// The hold landed but may not survive: keep it outstanding as
			// uncertain, never as one this process may still resolve.
			l.state.hold(attempt).State = holdUncertain
			return accountHold{Attempt: attempt, Reserved: nanoUSDString(amount), Probe: halfOpen},
				fmt.Errorf("%w: the reservation was not confirmed durable: %w", errLedgerUnavailable, err)
		}
		return accountHold{}, fmt.Errorf("%w: the reservation was not confirmed durable: %w", errLedgerUnavailable, err)
	}
	return accountHold{Attempt: attempt, Reserved: nanoUSDString(amount), Probe: halfOpen}, nil
}

// chatAccountSnapshot owns copies, including every deployment pointer/slice.
// It contains references only, never credential values or mutable routing state.
type chatAccountSnapshot struct {
	revision   string
	deployment nodepolicy.Deployment
	account    nodepolicy.Account
}

func (l *accountLedger) prepareChatAccount(id string) (chatAccountSnapshot, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.chatAccountLocked(id)
}

func (l *accountLedger) chatAccountLocked(id string) (chatAccountSnapshot, error) {
	for _, d := range l.routing.Deployments {
		if d.ID != id {
			continue
		}
		for _, a := range l.routing.Accounts {
			if a.ID != d.Account {
				continue
			}
			d.Capabilities = slices.Clone(d.Capabilities)
			if d.Transport != nil {
				v := *d.Transport
				d.Transport = &v
			}
			if d.Prices != nil {
				v := *d.Prices
				d.Prices = &v
			}
			if d.Qualification != nil {
				v := *d.Qualification
				d.Qualification = &v
			}
			return chatAccountSnapshot{l.policyRevision, d, a}, nil
		}
	}
	return chatAccountSnapshot{}, errUnknownDeployment
}

var errChatAttempt = errors.New("qualified Chat attempt refused")
var errChatAccounting = errors.New("qualified Chat accounting failed")
var errChatCredential = errors.New("qualified Chat credential refused")
var errChatConfiguration = errors.New("qualified Chat configuration refused")
var errChatRateLimit = errors.New("qualified Chat account cooling down")
var errChatTransient = errors.New("qualified Chat attempt failed")

func (l *accountLedger) reserveChatAccountAllowance(expected chatAccountSnapshot, input, output, remaining int64) (accountReservation, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	current, err := l.chatAccountLocked(expected.deployment.ID)
	if err != nil || !reflect.DeepEqual(current, expected) {
		return accountReservation{Durability: accountReserveNone}, errChatAttempt
	}
	hold, err := l.reserveLimitedLocked(expected.deployment.ID, input, output, remaining)
	durability := accountReserveNone
	if hold.Attempt != "" {
		durability = accountReserveDurable
		if err != nil {
			durability = accountReserveUncertain
		}
	}
	return accountReservation{Hold: hold, Durability: durability}, err
}

// runChatAccountAttempt is I's future single-attempt seam, not a route or a
// selection/retry state machine. No production caller exists. The mandatory
// guard belongs to I/G (permission, lease and revision); success is not a lease.
// All callbacks are trusted in-memory inputs. The resolver must inspect only
// its selected reference. There is no environment lookup or key cache here.
func runChatAccountAttempt(ctx context.Context, snapshot chatAccountSnapshot, body []byte,
	q *chatQualification, ledger *accountLedger, transport *publicClient,
	resolve func(nodepolicy.CredentialRef) (string, bool), budget chatByteBudget,
	guard publicAttemptGuard, lifecycle *publicAttemptLifecycle, emit func([]byte) error) (result chatParseResult, finalErr error) {
	d := snapshot.deployment
	if lifecycle == nil || lifecycle.request == nil {
		return result, errChatAttempt
	}
	terminal := publicAttemptTerminal{Accounting: publicAccountingRetained}
	defer func() {
		terminal.ProtocolComplete, terminal.Exposed = result.Complete, result.Exposed
		terminal.Err = finalErr
		lifecycle.finished(terminal)
	}()
	remaining, err := lifecycle.tryAccount(snapshot.account)
	if err != nil {
		return result, errRoutingSpend
	}
	if ledger == nil || transport == nil || resolve == nil || budget == nil || guard == nil || emit == nil ||
		ctx.Err() != nil || q.fresh() != nil || d.Transport == nil || d.Qualification == nil || d.Prices == nil ||
		nodepolicy.CheckEndpoint(d.Endpoint) != nil || transport.endpoint != d.Endpoint || transport.bounds != *d.Transport {
		return result, errChatAttempt
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	if lifecycle.bindCancel(cancel) != nil {
		return result, errChatAttempt
	}
	budget, stopWaiting := chatInferenceBudget(ctx, budget)
	if q.MaxBodyBytes > d.Transport.MaxResponseBytes || q.MaxEventBytes > d.Transport.MaxEventBytes {
		return result, errChatAttempt
	}
	ref, err := nodepolicy.ParseCredentialRef(string(snapshot.account.Credential))
	if err != nil {
		return result, errChatCredential
	}
	target := chatTarget{Protocol: d.Protocol, Endpoint: d.Endpoint, Model: d.Model, Qualification: *d.Qualification,
		ContextTokens: int64(d.ContextTokens), MaxOutputTokens: int64(d.MaxOutputTokens)}
	prepared, err := q.prepare(ctx, target, http.MethodPost, routeChatCompletions, body, budget)
	if err != nil {
		return result, err
	}
	defer prepared.close()
	// Admit finite parser and transport headroom before lookup or reservation.
	// The parser also admits each actual decoded representation as it grows.
	parserAllowance, err := q.acquireParserAllowance(budget, prepared.Stream)
	if err != nil {
		return result, err
	}
	defer parserAllowance.release()
	transportBytes := 3 * (max(d.Transport.MaxResponseBytes, d.Transport.MaxErrorBodyBytes) + 1)
	transportRelease, err := chatAcquire(budget, transportBytes)
	if err != nil {
		return result, err
	}
	defer transportRelease()
	// All later growth belongs to this admitted attempt and must interrupt
	// cleanly on refusal rather than waiting while a response is committed.
	stopWaiting()
	key, ok := resolve(ref)
	if !ok || key == "" || strings.ContainsFunc(key, unicode.IsControl) {
		return result, errChatCredential
	}
	if ctx.Err() != nil {
		return result, errChatAttempt
	}
	reservation, err := ledger.reserveChatAccountAllowance(snapshot, prepared.Input, prepared.Output, remaining)
	associationErr := lifecycle.associated(reservation)
	if err != nil || associationErr != nil {
		return result, errChatAccounting
	}
	hold := reservation.Hold
	if guard(ctx, nil) != nil || q.fresh() != nil || ctx.Err() != nil {
		// No refund: even an unsent hold has no new recovery authority here.
		return result, errChatAttempt
	}
	response, attemptErr := transport.postObserved(ctx, bytes.NewReader(prepared.Body), key, prepared.Stream,
		func(n int64) (func(), error) {
			if n != transportBytes {
				return nil, errChatBudget
			}
			return transportRelease, nil
		}, func(ctx context.Context) error {
			if q.fresh() != nil || ctx.Err() != nil {
				return errChatAttempt
			}
			return guard(ctx, func() error { return lifecycle.acceptedByGuard(ctx) })
		}, lifecycle.firstSuccessfulByte)
	if errors.Is(attemptErr, errPublicFinalGuard) {
		// Proven unsent: leave the hold unclassified and retained. There is no
		// provider outcome, cooldown, qualification or money reconciliation.
		return result, errPublicFinalGuard
	}
	outcome, retryAfter := outcomeTransient, time.Duration(0)
	if attemptErr != nil {
		attemptErr = errChatTransient
	}
	// Observed restrictive statuses remain authoritative if body handling fails.
	if response != nil {
		switch response.Status {
		case http.StatusOK:
			if attemptErr != nil {
				break
			}
			defer response.Body.Close()
			result, attemptErr = q.parseResponseWithAllowance(ctx, response.Body, prepared.Stream, prepared.Input, prepared.Output, response.HeaderAt, parserAllowance, emit)
			terminal.Delivered = attemptErr == nil && result.Complete
			if attemptErr == nil {
				outcome = outcomeOK
			}
		case http.StatusUnauthorized, http.StatusForbidden:
			outcome, attemptErr = outcomeCredentialFailure, errChatCredential
		case http.StatusBadRequest, http.StatusNotFound, http.StatusUnprocessableEntity:
			attemptErr = errChatConfiguration
		case http.StatusTooManyRequests:
			outcome, attemptErr = outcomeRateLimited, errChatRateLimit
			retryAfter = chatRetryAfter(response.RetryAfter, q.Now())
		default:
			if response.Body != nil {
				response.Body.Close()
			}
			attemptErr = errChatTransient
		}
	}
	// Independent completions must still be attempted after a sibling fails.
	outcomeErr := ledger.recordOutcome(hold.Attempt, outcome, retryAfter, time.Time{})
	var moneyErr, qualificationErr error
	if result.Complete && (attemptErr == nil || errors.Is(attemptErr, errChatBound)) {
		moneyErr = ledger.reconcile(hold.Attempt, result.Usage)
	}
	switch {
	case errors.Is(attemptErr, errChatIdentity):
		qualificationErr = ledger.quarantine(hold.Attempt)
	case result.AboveBound:
		// reconcile atomically freezes and classifies the immutable identity.
	case errors.Is(attemptErr, errChatBudget):
		// Resource refusal prevented classification; leave it unfinished.
	default:
		qualificationErr = ledger.recordQualification(hold.Attempt)
	}
	if outcomeErr != nil || (moneyErr != nil && !errors.Is(moneyErr, errUsageAboveBound)) || qualificationErr != nil {
		terminal.Accounting = publicAccountingFailed
		return result, errChatAccounting
	}
	terminal.Accounting = publicAccountingRecorded
	return result, attemptErr
}

// Invalid, past and overflowing values select D's default; D clamps valid
// durations. Provider strings never become diagnostics or downstream headers.
func chatRetryAfter(raw string, now time.Time) time.Duration {
	if seconds, err := strconv.ParseInt(raw, 10, 64); err == nil && raw != "" && raw[0] >= '0' && raw[0] <= '9' {
		if seconds <= 0 || seconds > math.MaxInt64/int64(time.Second) {
			return 0
		}
		return time.Duration(seconds) * time.Second
	}
	if at, err := http.ParseTime(raw); err == nil && at.After(now) {
		delay := at.Sub(now)
		if now.Add(delay).Equal(at) {
			return delay
		}
	}
	return 0
}

// checkCeilings admits amount against the per-request ceiling and against the
// day and month ceilings counting reconciled spend of the current period plus
// every outstanding hold.
func checkCeilings(s ledgerState, rec *ledgerAccount, acct nodepolicy.Account, amount int64, now time.Time) error {
	perRequest, err := acct.PerRequestCeiling.Int64()
	if err != nil {
		return err
	}
	if amount > perRequest {
		return errAccountCeiling
	}
	outstanding, err := s.outstanding(acct.ID)
	if err != nil {
		return err
	}
	for _, c := range []struct {
		key     string
		ceiling nodepolicy.NanoUSD
	}{
		{dayKey(now), acct.DailyCeiling},
		{monthKey(now), acct.MonthlyCeiling},
	} {
		limit, err := c.ceiling.Int64()
		if err != nil {
			return err
		}
		spent := int64(0)
		if v, ok := rec.Spent[c.key]; ok {
			if spent, err = v.Int64(); err != nil {
				return err
			}
		}
		total, err := addNanoUSD(spent, outstanding)
		if err == nil {
			total, err = addNanoUSD(total, amount)
		}
		if err != nil {
			return err
		}
		if total > limit {
			return errAccountCeiling
		}
	}
	return nil
}

// ownsLocked reports whether this process reserved attempt: its IDs carry the
// nonce of this process's open.
func (l *accountLedger) ownsLocked(attempt string) bool {
	return l.nonce != "" && strings.HasPrefix(attempt, l.nonce+"-")
}

func (s *ledgerState) unfinishedCount() int {
	n := 0
	for _, h := range s.Holds {
		if !h.evictable() {
			n++
		}
	}
	return n
}

func (l *accountLedger) newAttemptLocked() string {
	for {
		l.nextAttempt++
		id := l.nonce + "-" + strconv.FormatUint(l.nextAttempt, 10)
		if l.state.hold(id) == nil {
			return id
		}
	}
}

// reconcile resolves a hold this process reserved with the adapter's usage,
// priced by the hold's own snapshot. Nil usage means the provider answered
// without reporting usage, which charges the reserved bound. Usage above the
// hold's token bounds freezes the account and quarantines the deployment
// identity in the same write.
func (l *accountLedger) reconcile(attempt string, usage *tokenUsage) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.ensureOpenLocked(); err != nil {
		return err
	}
	h := l.state.hold(attempt)
	if h == nil || !l.ownsLocked(attempt) || h.State == holdUncertain {
		return errAttemptUnknown
	}
	if h.State == holdReconciled {
		if sameUsage(h.Usage, usage) {
			return nil
		}
		return errAttemptConflict
	}
	if usage != nil && (usage.InputTokens < 0 || usage.OutputTokens < 0) {
		return fmt.Errorf("%w: negative usage", errAttemptConflict)
	}
	next := l.state.clone()
	nh := next.hold(attempt)
	nowMs := l.now().UnixMilli()
	above := usage != nil && (usage.InputTokens > nh.InputTokens || usage.OutputTokens > nh.OutputTokens)

	actual, err := nh.Reserved.Int64()
	if usage != nil && err == nil {
		actual, err = reservationAmount(usage.InputTokens, usage.OutputTokens, nh.InputPerMillion, nh.OutputPerMillion)
	}
	var spendErr error
	if err == nil {
		spendErr = addSpend(next.account(nh.Account), time.UnixMilli(nh.ReservedAtMs), actual)
	}
	if err == nil && spendErr == nil {
		nh.State = holdReconciled
		if usage != nil {
			u := *usage
			nh.Usage = &u
		}
		nh.Actual = nanoUSDString(actual)
	}
	if !above {
		if err != nil {
			return err
		}
		if spendErr != nil {
			return spendErr
		}
		return l.commitLocked(next)
	}
	// Above the bound: whether or not the actual cost is representable, the
	// account is frozen and the identity quarantined. An unrepresentable cost
	// leaves the hold outstanding at its bound.
	rec := next.account(nh.Account)
	rec.Frozen = &accountFreeze{Attempt: attempt, Reason: freezeUsageAboveBound}
	rec.Generation++
	if !next.quarantine(nh.key(), quarantineUsageAboveBound, nowMs) {
		l.broken = fmt.Errorf("%w: quarantine limit reached", errLedgerUnavailable)
		return l.broken
	}
	nh.QualificationChecked = true
	if err := l.commitRestrictiveLocked(next); err != nil {
		return err
	}
	slog.Error("provider usage exceeded the reserved bound; account frozen and deployment quarantined",
		"account", nh.Account, "deployment", nh.Deployment)
	return errUsageAboveBound
}

func sameUsage(a, b *tokenUsage) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// addSpend adds amount to the account's spend in the day and month at.
func addSpend(rec *ledgerAccount, at time.Time, amount int64) error {
	updates := map[string]nodepolicy.NanoUSD{}
	for _, key := range []string{dayKey(at), monthKey(at)} {
		prev := int64(0)
		if v, ok := rec.Spent[key]; ok {
			var err error
			if prev, err = v.Int64(); err != nil {
				return err
			}
		}
		sum, err := addNanoUSD(prev, amount)
		if err != nil {
			return err
		}
		updates[key] = nanoUSDString(sum)
	}
	for k, v := range updates {
		rec.Spent[k] = v
	}
	return nil
}

// recordOutcome records the provider's account-level answer to an attempt
// this process reserved, once. Only the probe's success clears suppression,
// and only if no restrictive transition happened since it was reserved;
// deadlines merge to the later one. Money is untouched.
func (l *accountLedger) recordOutcome(attempt string, outcome accountOutcome, retryAfter time.Duration, resetAt time.Time) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.ensureOpenLocked(); err != nil {
		return err
	}
	h := l.state.hold(attempt)
	if h == nil || !l.ownsLocked(attempt) || h.State == holdUncertain {
		return errAttemptUnknown
	}
	if h.Outcome != "" {
		if h.Outcome == outcome {
			return nil
		}
		return errAttemptConflict
	}
	now := l.now()
	next := l.state.clone()
	nh := next.hold(attempt)
	rec := next.account(nh.Account)
	nh.Outcome = outcome
	restrictive := true
	switch outcome {
	case outcomeOK:
		restrictive = false
		if rec.Probe == attempt && rec.Generation == nh.Generation {
			rec.Cooldown, rec.CredentialFailure = nil, nil
		}
	case outcomeRateLimited:
		d := retryAfter
		if d <= 0 {
			d = defaultCooldown
		}
		d = min(max(d, minCooldown), maxCooldown)
		rec.Cooldown = laterCooldown(rec.Cooldown, &accountCooldown{UntilMs: now.Add(d).UnixMilli(), Reason: cooldownRateLimited})
	case outcomeExhausted:
		if !resetAt.After(now) || resetAt.Sub(now) > maxExhaustionWindow {
			return fmt.Errorf("%w: exhaustion reset must be in the future and within %s", errInvalidOutcome, maxExhaustionWindow)
		}
		rec.Cooldown = laterCooldown(rec.Cooldown, &accountCooldown{UntilMs: resetAt.UnixMilli(), Reason: cooldownExhausted})
	case outcomeCredentialFailure:
		// A late failure under an obsolete reference must not lift an
		// existing suppression of the last configured reference, even while
		// the account is retired.
		currentFailed := rec.CredentialFailure != nil && rec.CredentialFailure.Credential == rec.LastConfiguredCredential
		if !currentFailed {
			rec.CredentialFailure = &credentialFailure{Credential: nh.Credential}
		}
	case outcomeTransient:
		restrictive = false
	default:
		return errInvalidOutcome
	}
	if rec.Probe == attempt {
		rec.Probe = ""
	}
	if !restrictive {
		return l.commitLocked(next)
	}
	rec.Generation++
	return l.commitRestrictiveLocked(next)
}

func laterCooldown(a, b *accountCooldown) *accountCooldown {
	if a == nil || b.UntilMs > a.UntilMs {
		return b
	}
	return a
}

// recordQualification marks an attempt's qualification classification complete
// with no violation. Call only after all identity and bound checks finish;
// violations instead call quarantine. Until then its immutable hold is retained,
// even if reconciliation and the account outcome have already completed.
func (l *accountLedger) recordQualification(attempt string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.ensureOpenLocked(); err != nil {
		return err
	}
	h := l.state.hold(attempt)
	if h == nil || !l.ownsLocked(attempt) || h.State == holdUncertain {
		return errAttemptUnknown
	}
	if h.QualificationChecked {
		return nil
	}
	next := l.state.clone()
	next.hold(attempt).QualificationChecked = true
	return l.commitLocked(next)
}

// quarantine records a qualification failure of the deployment identity the
// attempt was reserved under, whatever the deployment is called or configured
// as now. Rename, retirement, re-addition and restart keep it; only a new
// qualification revision is a different identity.
func (l *accountLedger) quarantine(attempt string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.ensureOpenLocked(); err != nil {
		return err
	}
	h := l.state.hold(attempt)
	if h == nil || !l.ownsLocked(attempt) {
		return errAttemptUnknown
	}
	if l.state.quarantined(h.key()) && h.QualificationChecked {
		return nil
	}
	next := l.state.clone()
	next.hold(attempt).QualificationChecked = true
	if !next.quarantine(h.key(), quarantineQualification, l.now().UnixMilli()) {
		l.broken = fmt.Errorf("%w: quarantine limit reached", errLedgerUnavailable)
		return l.broken
	}
	return l.commitRestrictiveLocked(next)
}

// --- exact nano-USD arithmetic --------------------------------------------

// chargeNanoUSD is ceil(tokens * perMillion / 1,000,000) with a 128-bit
// product, so no intermediate overflows; the result must fit signed 64 bits.
func chargeNanoUSD(tokens int64, perMillion nodepolicy.NanoUSD) (int64, error) {
	rate, err := perMillion.Int64()
	if err != nil {
		return 0, err
	}
	if tokens < 0 {
		return 0, errTokenBounds
	}
	hi, lo := bits.Mul64(uint64(tokens), uint64(rate))
	if hi >= nanoUSDTokenScale {
		return 0, errMoneyOverflow
	}
	q, r := bits.Div64(hi, lo, nanoUSDTokenScale)
	if q > math.MaxInt64 {
		return 0, errMoneyOverflow
	}
	if r != 0 {
		q++
	}
	if q > math.MaxInt64 {
		return 0, errMoneyOverflow
	}
	return int64(q), nil
}

// reservationAmount is the upper-bound cost of input and output tokens.
func reservationAmount(inputTokens, outputTokens int64, inputPerMillion, outputPerMillion nodepolicy.NanoUSD) (int64, error) {
	in, err := chargeNanoUSD(inputTokens, inputPerMillion)
	if err != nil {
		return 0, err
	}
	out, err := chargeNanoUSD(outputTokens, outputPerMillion)
	if err != nil {
		return 0, err
	}
	return addNanoUSD(in, out)
}

func addNanoUSD(a, b int64) (int64, error) {
	if b > math.MaxInt64-a {
		return 0, errMoneyOverflow
	}
	return a + b, nil
}

func nanoUSDString(v int64) nodepolicy.NanoUSD {
	return nodepolicy.NanoUSD(strconv.FormatInt(v, 10))
}

// --- periods and pruning ---------------------------------------------------

func dayKey(t time.Time) string   { return "d:" + t.UTC().Format("2006-01-02") }
func monthKey(t time.Time) string { return "m:" + t.UTC().Format("2006-01") }

// prunePeriods drops spend of periods other than the current day and month.
// Callers skip it while the clock is behind the last observed time.
func prunePeriods(s *ledgerState, now time.Time) {
	day, month := dayKey(now), monthKey(now)
	for i := range s.Accounts {
		for k := range s.Accounts[i].Spent {
			if k != day && k != month {
				delete(s.Accounts[i].Spent, k)
			}
		}
	}
}

// pruneResolved keeps at most maxResolvedHolds resolved holds, dropping the
// oldest first. Their amounts are already in Spent and both account outcome
// and qualification classification are recorded; only duplicate memory goes.
// Unfinished holds are never dropped.
func pruneResolved(s *ledgerState) {
	resolved := 0
	for _, h := range s.Holds {
		if h.evictable() {
			resolved++
		}
	}
	drop := resolved - maxResolvedHolds
	if drop <= 0 {
		return
	}
	kept := s.Holds[:0:0]
	for _, h := range s.Holds {
		if drop > 0 && h.evictable() {
			drop--
			continue
		}
		kept = append(kept, h)
	}
	s.Holds = kept
}
