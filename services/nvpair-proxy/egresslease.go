// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"time"

	"nvpair-shared/nodepolicy"
)

var errPublicPermission = errors.New("public attempt permission refused")
var errEgressLease = errors.New("egress lease binding, sequence or deadline refused")

// egressLease is process-local authority, not a policy snapshot. Admission owns
// the current policy. Lock order is activationMu -> mu -> admission; neither mu
// nor admission may be held while entering the account ledger or doing IO.
type egressLease struct {
	activationMu sync.Mutex
	mu           sync.Mutex
	now          func() time.Duration
	sequence     int64
	notAfter     int64
	revision     string
	ready        bool
	transition   *egressActivation
}

type egressActivation struct{ revision string }

func newEgressLease(now func() time.Duration) *egressLease {
	return &egressLease{now: now}
}

func (p *Proxy) renewEgressLease(params json.RawMessage) (nodepolicy.RenewEgressLeaseResult, error) {
	var in nodepolicy.RenewEgressLeaseParams
	if err := decodeStrict(params, &in); err != nil {
		return nodepolicy.RenewEgressLeaseResult{}, err
	}
	l := p.egress
	l.mu.Lock()
	defer l.mu.Unlock()
	sequence, _ := strconv.ParseInt(in.Sequence, 10, 64) // wire validation is strict
	deadline, _ := strconv.ParseInt(in.NotAfterNanos, 10, 64)
	now := int64(l.now())
	policy := p.admission.currentPolicy()
	if in.RunID != p.runID || in.Revision != policy.Revision || !l.ready || l.transition != nil ||
		sequence <= l.sequence || now < 0 || (deadline != 0 && (deadline <= now || deadline-now > int64(nodepolicy.MaxEgressLeaseDuration))) {
		return nodepolicy.RenewEgressLeaseResult{}, errEgressLease
	}
	l.sequence = sequence
	if deadline != 0 {
		l.revision, l.notAfter = in.Revision, deadline
	}
	return nodepolicy.RenewEgressLeaseResult{RunID: p.runID, Revision: policy.Revision,
		Sequence: in.Sequence, NowNanos: strconv.FormatInt(now, 10)}, nil
}

func (p *Proxy) revokeEgressLease(params json.RawMessage) (nodepolicy.RevokeEgressLeaseResult, error) {
	var in nodepolicy.RevokeEgressLeaseParams
	if err := decodeStrict(params, &in); err != nil {
		return nodepolicy.RevokeEgressLeaseResult{}, err
	}
	l := p.egress
	l.mu.Lock()
	defer l.mu.Unlock()
	sequence, _ := strconv.ParseInt(in.Sequence, 10, 64)
	if in.RunID != p.runID || sequence <= l.sequence {
		return nodepolicy.RevokeEgressLeaseResult{}, errEgressLease
	}
	l.sequence, l.notAfter, l.revision = sequence, 0, ""
	return nodepolicy.RevokeEgressLeaseResult{RunID: p.runID, Sequence: in.Sequence}, nil
}

// beginPublicActivation validates order before withdrawing permission. Its
// token prevents an obsolete ledger completion from reopening a later policy.
func (p *Proxy) beginPublicActivation(policy nodepolicy.Policy) (*egressActivation, error) {
	capacity, wait, err := p.bufferSettings(policy)
	if err != nil {
		return nil, err
	}
	l := p.egress
	l.mu.Lock()
	defer l.mu.Unlock()
	current := p.admission.currentPolicy()
	cmp, err := nodepolicy.CompareRevisions(policy.Revision, current.Revision)
	if err != nil || cmp < 0 {
		return nil, errPublicPermission
	}
	token := &egressActivation{revision: policy.Revision}
	if err := p.admission.activatePolicy(policy); err != nil {
		return nil, err
	}
	// Install beside admission under the winning transition lock. An obsolete
	// applySetPolicy waiter can finish later but cannot reconfigure this root.
	if err := p.buffers.reconfigure(capacity, wait); err != nil {
		panic(err) // bufferSettings checked the allocator's input domain.
	}
	l.transition, l.ready, l.notAfter, l.revision = token, false, 0, ""
	return token, nil
}

func (p *Proxy) completePublicActivation(token *egressActivation, accountErr error) {
	l := p.egress
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.transition == token && p.admission.currentPolicy().Revision == token.revision {
		l.transition = nil
		l.ready = accountErr == nil
	}
}

// publicAttemptPermitted is advisory only: callers must enter the wrapper for
// each attempt. The shipped build refuses before body preparation or lookup.
func (p *Proxy) publicAttemptPermitted(tier string, snapshot chatAccountSnapshot) error {
	if !nodepolicy.PublicDispatchAvailable() {
		return errPublicPermission
	}
	return p.publicPermission(tier, snapshot)
}

// publicPermission serializes permission acceptance with policy transitions and
// lease withdrawal. Synthetic E tests exercise this mechanism without enabling
// the production wrapper. No permission/grant escapes this call.
func (p *Proxy) publicPermission(tier string, snapshot chatAccountSnapshot) error {
	return p.publicPermissionAccept(tier, snapshot, nil)
}

func (p *Proxy) publicPermissionAccept(tier string, snapshot chatAccountSnapshot, accept func() error) error {
	l := p.egress
	l.mu.Lock()
	defer l.mu.Unlock()
	policy := p.admission.currentPolicy()
	now := int64(l.now())
	if p.runID == "" || snapshot.revision == "" || snapshot.revision != policy.Revision ||
		!l.ready || l.transition != nil || l.revision != policy.Revision || now < 0 || now >= l.notAfter {
		return errPublicPermission
	}
	d, a := snapshot.deployment, snapshot.account
	if !d.Enabled || d.ID == "" || a.ID == "" || d.Account != a.ID || a.Credential == "" ||
		(a.Billing != nodepolicy.BillingFree && a.Billing != nodepolicy.BillingTrial && a.Billing != nodepolicy.BillingPaid) {
		return errPublicPermission
	}
	if !slices.ContainsFunc(policy.Routing.Deployments, func(v nodepolicy.Deployment) bool { return reflect.DeepEqual(v, d) }) ||
		!slices.ContainsFunc(policy.Routing.Accounts, func(v nodepolicy.Account) bool { return reflect.DeepEqual(v, a) }) {
		return errPublicPermission
	}
	rules, ok := policy.Routing.Tiers[tier]
	if !ok || !rules.RemoteAllowed || (a.Billing == nodepolicy.BillingPaid && !rules.PaidAllowed) ||
		(len(rules.AllowedFamilies) != 0 && !slices.Contains(rules.AllowedFamilies, d.Family)) ||
		!slices.ContainsFunc(policy.Tiers[tier], func(v nodepolicy.TierEntry) bool { return v.Kind == nodepolicy.EntryAPI && v.Deployment == d.ID }) {
		return errPublicPermission
	}
	if accept != nil {
		return accept()
	}
	return nil
}

type publicRefusalStage string

const (
	publicBeforeReserve publicRefusalStage = "before-reserve"
	publicAfterReserve  publicRefusalStage = "after-reserve"
	publicFinalGuard    publicRefusalStage = "final-guard"
)

type publicAttemptRefusal struct{ Stage publicRefusalStage }

func (e *publicAttemptRefusal) Error() string { return "public attempt refused: " + string(e.Stage) }
func (e *publicAttemptRefusal) Unwrap() error { return errPublicPermission }

// A refused snapshot can predate an account edit. G observes the account's
// current policy under its lock; D's snapshot and any existing hold stay intact.
// Lock order remains G lease -> admission -> request, with no IO here.
func (p *Proxy) observePublicAccount(snapshot chatAccountSnapshot, lifecycle *publicAttemptLifecycle) error {
	l := p.egress
	l.mu.Lock()
	defer l.mu.Unlock()
	policy := p.admission.currentPolicy()
	for _, account := range policy.Routing.Accounts {
		if account.ID == snapshot.account.ID {
			_, err := lifecycle.tryAccount(account)
			return err
		}
	}
	return errPublicPermission
}

// runPublicChatAttempt is the sole future production public-attempt entry point.
// It is deliberately unwired from HTTP ingress. Every invocation is one new
// attempt; D owns durable holds and E owns preparation, transport and accounting.
func (p *Proxy) runPublicChatAttempt(ctx context.Context, tier string,
	snapshot chatAccountSnapshot, body []byte, q *chatQualification,
	transport *publicClient, resolve func(nodepolicy.CredentialRef) (string, bool),
	budget chatByteBudget, lifecycle *publicAttemptLifecycle, emit func([]byte) error) (chatParseResult, error) {
	if p.publicAttemptPermitted(tier, snapshot) != nil {
		err := &publicAttemptRefusal{Stage: publicBeforeReserve}
		if lifecycle != nil && lifecycle.request != nil {
			// Trying a refused paid candidate still tightens the same request.
			// This is an in-memory observation, never a grant or ledger write.
			allowanceErr := p.observePublicAccount(snapshot, lifecycle)
			lifecycle.finished(publicAttemptTerminal{Accounting: publicAccountingRetained, Err: errors.Join(err, allowanceErr)})
		}
		return chatParseResult{}, err
	}
	refused := false
	guard := func(ctx context.Context, accept func() error) error {
		if !nodepolicy.PublicDispatchAvailable() || ctx.Err() != nil || p.publicPermissionAccept(tier, snapshot, accept) != nil {
			refused = true
			return errPublicPermission
		}
		return nil
	}
	result, err := runChatAccountAttempt(ctx, snapshot, body, q, p.accounts, transport, resolve, budget, guard, lifecycle, emit)
	if errors.Is(err, errPublicFinalGuard) {
		return result, &publicAttemptRefusal{Stage: publicFinalGuard}
	}
	if refused {
		return result, &publicAttemptRefusal{Stage: publicAfterReserve}
	}
	return result, err
}
