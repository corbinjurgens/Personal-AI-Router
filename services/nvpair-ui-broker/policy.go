// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

// policy.go is the broker side of the fork's node policy (FORK_DESIGN.md §3.1,
// §3.2). The broker owns <appdir>/node-policy.json and is its only writer; the
// proxy enforces it. This file holds the store, the client methods
// policy:get / policy:set, the state mirrored into the proxy (policy,
// availability, residency, saved engine intent, settings drains), and the
// remote relay that lets a paired node read or change this node's policy.
//
// Availability orchestration lives in availability.go, the idle policy and
// start-on-demand in idlepolicy.go, settings drains in settingsdrain.go.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"nvpair-shared/appdir"
	"nvpair-shared/clustertrust"
	"nvpair-shared/durablefile"
	"nvpair-shared/engineintent"
	"nvpair-shared/nodepolicy"
)

// Client-facing methods and notifications.
const (
	methodPolicyGet           = "policy:get"
	methodPolicySet           = "policy:set"
	methodSetAvailability     = "node:set-availability"
	notifyPolicyChanged       = "policy:changed"
	notifyAvailabilityChanged = "node:availability-changed"
)

// Engine-manager methods and notifications the policy uses.
const (
	engineWakeMethod           = "engine:wake"
	engineSleepMethod          = "engine:sleep"
	engineIntentMethod         = "engine:intent"
	engineIntentChanged        = "engine:intent-changed"
	engineUnloadModelMethod    = "engine:unload-model"
	engineStateChangedMethod   = "engine:state-changed"
	engineModelsChangedMethod  = "engine:models-changed"
	engineNodeProjectionMethod = "engine:node-projection"
	engineModelIdentityMethod  = "engine:model-identity"
	policyRelayRequestMethod   = "policy:request"
	policyRelayReplyMethod     = "policy:reply"
	policyRelayCancelMethod    = "policy:cancel"
)

// remotePolicyMethods maps a client method aimed at another node to the
// engine-manager method that carries it over the pinned ec surface.
var remotePolicyMethods = map[string]string{
	methodPolicyGet:       "engine:remote-policy-get",
	methodPolicySet:       "engine:remote-policy-set",
	methodSetAvailability: "engine:remote-availability-set",
}

// policyActivationTimeout bounds the whole wait for the proxy to activate a
// revision: getting the push slot and the proxy's acknowledgement together. A
// revision not acknowledged by then is pending or in error, and the proxy keeps
// enforcing what it had.
const policyActivationTimeout = proxyCallTimeout

// activationFloorDivisor sets the least budget a bounded activation starts a
// push with: one part in this many of policyActivationTimeout. A request begun
// with less cannot be told apart, when it fails, from one lost after it was
// written, so it is not begun.
const activationFloorDivisor = 10

// Activation states a client sees beside the persisted revision.
const (
	activationActive  = "active"
	activationPending = "pending"
	activationError   = "error"
)

// engineCallTimeout bounds one engine-manager call made on the policy's behalf
// (an unload, a sleep, a wake). A wake waits for engine readiness, which can
// take minutes for a large model server.
const engineCallTimeout = 5 * time.Minute

// policyRuntime is everything the node policy tracks. One mutex guards the
// state; the push slot serializes what is sent to the proxy, and every push
// reads the state at send time, so a later change can never be overwritten by
// an earlier one that happened to be sent second.
type policyRuntime struct {
	mu     sync.Mutex
	models modelConfigRuntime
	// path is the policy file. Empty keeps the policy in memory only, which is
	// what a Broker built by a test gets unless it asks for a file.
	path    string
	current nodepolicy.Policy
	ready   bool
	// recoveryRequired is set when the policy file could not be used at startup.
	// The node then runs on a paused recovery policy and nothing is persisted
	// until an explicit policy:set succeeds.
	recoveryRequired bool
	// issued is the highest revision this process has assigned, published or
	// not, so a revision is never assigned twice even when its publication
	// failed after the new bytes reached the disk.
	issued int64
	// replacer publishes the file. Its zero value uses the platform steps; a
	// test sets one step to inject a failure at that boundary.
	replacer durablefile.Replacer
	// activationTimeout replaces policyActivationTimeout when positive; a test
	// sets it to make a lost acknowledgement cheap.
	activationTimeout time.Duration
	// Activation: what the current proxy incarnation has acknowledged of the
	// node/set-policy pushes, the only requests that activate a revision.
	// incarnation is the proxy process replayPolicyToProxy last started
	// from, runID the run id of its first acknowledgement, activeRevision the
	// newest revision it acknowledged and activationErr why the newest push
	// failed. An acknowledgement from any other process or run is discarded.
	incarnation    *proxyProcess
	runID          string
	activeRevision string
	activationErr  string
	// availabilityErr is why the newest node/set-availability push to this
	// incarnation failed, empty once one is acknowledged. The proxy ignores the
	// availability inside node/set-policy, so while this is set no revision is
	// enforced as the policy says and none is reported active.
	availabilityErr string
	// egress carries only process-scoped lease sequencing and clock anchoring.
	// Persisted policy and the activation fields above remain authoritative.
	egress brokerEgressLease
	// strictReadiness is the newest strict admission readiness the proxy
	// reported (readiness.go), held until its fences match; replay clears it.
	// strictPublished is the capability the last node projection carried.
	strictReadiness nodepolicy.StrictAdmissionReadiness
	strictPublished *nodepolicy.StrictAdmissionCapability
	// live is the availability the node is actually in. It differs from
	// current.Availability only while draining.
	live         nodepolicy.Availability
	cancelActive bool
	// localPauseRequested is set while the newest accepted availability
	// request is a pause made on this node, from its acceptance until its
	// transition ends, so a paired node's resume is refused even before that
	// pause has been saved (setAvailability).
	localPauseRequested bool
	// pendingPauseRequested fences management from acceptance of either origin's
	// pause until it is stored, or a local Resume takes effect. Guarded by mu.
	pendingPauseRequested bool

	// Admission state from the proxy, and a channel closed on every update so
	// waiters can block on a change without polling.
	admission     nodepolicy.AdmissionState
	haveAdmission bool
	admissionCh   chan struct{}

	// What engine-manager reports: which engines run, what each has loaded,
	// and each engine's saved intent.
	running map[string]bool
	// An installed-state snapshot cannot overwrite a newer observation made
	// while its RPC was in flight.
	runningGeneration uint64
	runningObserved   map[string]uint64
	loaded            map[string][]string
	intent            map[string]bool
	drains            map[string]bool

	// Strict admission identity (model_identity.go): engine-manager's run,
	// launch and model content identity per engine, and this broker's
	// applied-settings fingerprint per engine. The proxy gets their join.
	engineIdentity      map[string]engineManagerIdentity
	engineManagerSource *engineManagerSource
	settingsIdentity    map[string]string

	// Idle bookkeeping: when an engine was seen to start, when it last gained a
	// loaded model, and the baseline the idle policy last acted on.
	runningSince   map[string]time.Time
	residencyAt    map[string]time.Time
	idleUnloadedAt map[string]time.Time
	idleSleptAt    map[string]time.Time
	now            func() time.Time

	// Availability transitions: transitionMu runs one at a time; the newest
	// request bumps transitionGen and cancels the one in progress.
	transitionMu     sync.Mutex
	transitionGen    uint64
	transitionCancel context.CancelFunc
	pauseSlept       map[string]bool
	restoreSkipped   bool

	waking map[string]bool

	// pushSlot is the push lock: a one-slot channel, so a caller can wait for
	// it under a deadline. Whoever holds it sends to the proxy.
	pushSlot    chan struct{}
	pendingPush map[string]bool

	// publishMu serializes this node's policy and availability announcements
	// to clients (policy_notifications.go). It is taken before r.mu and held
	// across the client write; r.mu is not.
	publishMu sync.Mutex
	// announcePending is guarded by mu until a scheduled sender takes publishMu.
	announcePending bool
	// projectionMu serializes this node's projections to engine-manager the
	// same way (policy_notifications.go). It is separate from publishMu so an
	// engine-manager slow to read never holds up client announcements.
	projectionMu sync.Mutex

	relayCancels map[string]context.CancelFunc
}

// nodePolicyRuntime returns the runtime, initializing it with the default
// policy the first time. Serve calls loadNodePolicy before anything reads it.
func (b *Broker) nodePolicyRuntime() *policyRuntime {
	r := &b.nodePolicy
	r.mu.Lock()
	r.initLocked()
	r.mu.Unlock()
	return r
}

func (r *policyRuntime) initLocked() {
	if r.ready {
		return
	}
	r.ready = true
	r.current = nodepolicy.Default()
	r.live = r.current.Availability
	r.admissionCh = make(chan struct{})
	r.running = map[string]bool{}
	r.runningObserved = map[string]uint64{}
	r.loaded = map[string][]string{}
	r.intent = map[string]bool{}
	r.drains = map[string]bool{}
	r.runningSince = map[string]time.Time{}
	r.residencyAt = map[string]time.Time{}
	r.idleUnloadedAt = map[string]time.Time{}
	r.idleSleptAt = map[string]time.Time{}
	r.pauseSlept = map[string]bool{}
	r.waking = map[string]bool{}
	r.pendingPush = map[string]bool{}
	r.pushSlot = make(chan struct{}, 1)
	r.relayCancels = map[string]context.CancelFunc{}
	r.egress.now = time.Now
	if r.now == nil {
		r.now = time.Now
	}
}

// managementBlockedLocked is the authoritative pause boundary. Provenance is
// independent: a peer pause has the same boundary as a local pause. Caller
// holds r.mu; localPauseRequested also retains existing scoped obligations.
func (r *policyRuntime) managementBlockedLocked() bool {
	return r.pendingPauseRequested || r.localPauseRequested || r.current.Availability == nodepolicy.Paused ||
		r.live == nodepolicy.Draining || r.live == nodepolicy.Paused
}

// loadNodePolicy reads the policy file at startup. A missing file means the
// default policy. A version 1 file is migrated in memory and rewritten as
// version 2 by the next successful save. A file that cannot be read, parsed or
// validated is kept as <file>.bad when it was read, and the node starts on the
// recovery policy with recoveryRequired set, so a bad edit can never stop the
// node from starting and the node never comes back more permissive than the
// file could have been.
func (b *Broker) loadNodePolicy(path string) {
	r := b.nodePolicyRuntime()
	policy := nodepolicy.Default()
	recovery := false
	data, err := os.ReadFile(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		recovery = true
		slog.Warn("node policy unreadable; policy recovery is required", "path", path, "err", err)
	default:
		parsed, perr := nodepolicy.Parse(data)
		if perr != nil {
			recovery = true
			bad := path + ".bad"
			if werr := os.WriteFile(bad, data, 0o600); werr != nil {
				slog.Warn("could not keep a copy of the invalid node policy", "path", bad, "err", werr)
			}
			slog.Warn("node policy invalid; kept a copy and policy recovery is required", "path", path, "copy", bad, "err", perr)
		} else {
			policy = parsed
		}
	}
	if recovery {
		policy = recoveryPolicy()
	}
	r.mu.Lock()
	r.path = path
	r.current = policy
	r.live = policy.Availability
	r.recoveryRequired = recovery
	r.mu.Unlock()
	slog.Info("node policy loaded", "path", path, "availability", policy.Availability, "revision", policy.Revision, "recoveryRequired", recovery)
}

// recoveryPolicy is what a node runs on while its policy file is unusable: the
// defaults, which keep remote and paid execution off and define no tiers or
// profiles, with the node paused. The failed file may have said paused, so the
// node never starts more available than it could have been.
func recoveryPolicy() nodepolicy.Policy {
	policy := nodepolicy.Default()
	policy.Availability = nodepolicy.Paused
	return policy
}

// defaultNodePolicyPath is <appdir>/node-policy.json.
func defaultNodePolicyPath() (string, error) {
	return appdir.Path(nodepolicy.FileName)
}

// saveLocked publishes the policy file through durablefile. Caller holds r.mu.
func (r *policyRuntime) saveLocked(policy nodepolicy.Policy) error {
	if r.path == "" {
		return nil
	}
	data, err := json.MarshalIndent(policy, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(r.path), 0o700); err != nil {
		return err
	}
	return r.replacer.WriteFile(r.path, append(data, '\n'), 0o600)
}

// nextRevisionLocked assigns the next revision: the larger of the previous one
// plus one and the current Unix time in nanoseconds. The clock term keeps a
// revision issued before a corrupt file was recovered from ever matching one
// issued after. Caller holds r.mu.
func (r *policyRuntime) nextRevisionLocked() (string, error) {
	base, err := nodepolicy.NanoUSD(r.current.Revision).Int64()
	if err != nil {
		return "", fmt.Errorf("the stored revision is invalid: %w", err)
	}
	base = max(base, r.issued)
	if base == math.MaxInt64 {
		return "", errors.New("the policy revision cannot advance past the signed 64-bit maximum")
	}
	next := max(base+1, r.now().UnixNano())
	r.issued = next
	return strconv.FormatInt(next, 10), nil
}

// publishLocked stamps policy with the next revision and publishes it
// durably, returning the stamped document on success or an ErrReplaced failure.
// r.current is the caller's to update when the replacement landed, even if its
// durability is uncertain. Caller holds r.mu.
func (r *policyRuntime) publishLocked(policy nodepolicy.Policy) (nodepolicy.Policy, error) {
	revision, err := r.nextRevisionLocked()
	if err != nil {
		return nodepolicy.Policy{}, err
	}
	policy.Revision = revision
	if err := r.saveLocked(policy); err != nil {
		if errors.Is(err, durablefile.ErrReplaced) {
			return policy, err
		}
		return nodepolicy.Policy{}, err
	}
	return policy, nil
}

// policyActivation says whether the revision of the policy it accompanies is
// enforced by the running proxy. It is flattened into the policy:get and policy:set results
// and the policy:changed params. The persisted revision is policy.revision.
type policyActivation struct {
	// ActiveRevision is the revision the current proxy incarnation
	// acknowledged, or empty when it has acknowledged none or none runs.
	ActiveRevision string `json:"activeRevision"`
	// Activation is activationActive when ActiveRevision has reached the
	// revision of the policy described, activationError when the newest push
	// failed and activationPending otherwise.
	Activation string `json:"activation"`
	// ActivationError is operational text, never policy content or credential
	// references.
	ActivationError string `json:"activationError,omitempty"`
}

// activationBoundLocked is the bound on one activation wait. Caller holds r.mu.
func (r *policyRuntime) activationBoundLocked() time.Duration {
	if r.activationTimeout > 0 {
		return r.activationTimeout
	}
	return policyActivationTimeout
}

// activationLocked reports whether the running proxy has activated revision,
// the revision of the policy a result or announcement describes. running is
// the proxy process now published. It is active when the proxy acknowledged
// that revision or a later one and the newest availability push, which that
// revision's availability rests on, was acknowledged too; in error when either
// newest push failed and pending otherwise. While the availability push is
// unacknowledged no revision is verifiably enforced, so no active revision is
// reported. Caller holds r.mu.
func (r *policyRuntime) activationLocked(running *proxyProcess, revision string) policyActivation {
	if running == nil || running != r.incarnation {
		return policyActivation{Activation: activationPending}
	}
	if r.availabilityErr != "" {
		return policyActivation{Activation: activationError, ActivationError: r.availabilityErr}
	}
	a := policyActivation{ActiveRevision: r.activeRevision}
	switch cmp, err := nodepolicy.CompareRevisions(r.activeRevision, revision); {
	case r.activeRevision != "" && err == nil && cmp >= 0:
		a.Activation = activationActive
	case r.activationErr != "":
		a.Activation, a.ActivationError = activationError, r.activationErr
	default:
		a.Activation = activationPending
	}
	return a
}

// policyGetResult is the policy:get result.
type policyGetResult struct {
	ModelConfiguration nodepolicy.ModelConfiguration `json:"modelConfiguration"`
	Policy             nodepolicy.Policy             `json:"policy"`
	Availability       nodepolicy.Availability       `json:"availability"`
	// RecoveryRequired is true while the node runs on the recovery policy
	// because its file was unusable; an explicit policy:set clears it.
	RecoveryRequired bool `json:"recoveryRequired"`
	policyActivation
}

// policySetResult is the policy:set result: the persisted policy and whether
// the proxy has activated it.
type policySetResult struct {
	ModelConfiguration nodepolicy.ModelConfiguration `json:"modelConfiguration"`
	Policy             nodepolicy.Policy             `json:"policy"`
	policyActivation
}

// availabilitySetResult is the node:set-availability result: the state reached
// and whether the proxy enforces the persisted revision that carries it.
type availabilitySetResult struct {
	Availability nodepolicy.Availability `json:"availability"`
	// Revision is the persisted revision the activation fields describe.
	Revision string `json:"revision"`
	policyActivation
}

// policyParams is the params shape of every policy method.
type policyParams struct {
	NodeID string          `json:"nodeId,omitempty"`
	Policy json.RawMessage `json:"policy,omitempty"`
	State  string          `json:"state,omitempty"`
	// ExpectedPersistedRevision is the policy.revision of the policy:get the
	// caller's draft is based on. Only policy:set reads it; it stays raw so a
	// stray value on another method is ignored rather than rejected.
	ExpectedPersistedRevision json.RawMessage `json:"expectedPersistedRevision,omitempty"`
}

// Typed JSON-RPC error codes of the policy methods, the same on the local
// methods and in the relay's policy:reply errorCode.
const (
	// codePolicyConflict is a policy:set whose expectedPersistedRevision is not
	// the persisted revision.
	codePolicyConflict = -32010
	// codeLocalVeto is a paired node's resume of a node paused on the node
	// itself.
	codeLocalVeto = -32011
)

// policyConflictError reports a policy:set whose draft is based on a revision
// other than the persisted one. The caller must reload and review the policy;
// resubmitting the old draft would restore what a newer write removed.
type policyConflictError struct{ expected, current string }

func (e *policyConflictError) Error() string {
	return fmt.Sprintf("policy conflict: the persisted revision is %s, not the expected %s; reload the policy and review it before saving", e.current, e.expected)
}

// parseExpectedRevision reads the precondition of a policy:set. A missing,
// null, empty or non-canonical revision is refused: there is no unconditional
// write.
func parseExpectedRevision(raw json.RawMessage) (string, error) {
	var revision string
	if len(raw) == 0 || json.Unmarshal(raw, &revision) != nil || revision == "" {
		return "", errors.New("expectedPersistedRevision is required: send the revision from the preceding policy:get")
	}
	if _, err := nodepolicy.ParseNanoUSD(revision); err != nil {
		return "", errors.New("expectedPersistedRevision must be a canonical decimal string: 0 or a nonzero leading digit, within the signed 64-bit range")
	}
	return revision, nil
}

// checkExpectedRevision is the compare-and-set of every policy:set. It runs
// first in the build step of persistPolicy, under r.mu, so two writers holding
// the same revision cannot both pass.
func checkExpectedRevision(stored nodepolicy.Policy, expected string) error {
	if stored.Revision != expected {
		return &policyConflictError{expected: expected, current: stored.Revision}
	}
	return nil
}

func (b *Broker) localPolicy() policyGetResult {
	e := b.captureModelConfig()
	configuration := b.modelConfiguration(e)
	return policyGetResult{ModelConfiguration: configuration, Policy: e.Policy, Availability: e.Live, RecoveryRequired: e.Recovery, policyActivation: e.Activation}
}

// activationOf is the activation of one revision right now. A result or
// announcement describes the policy it carries, not whichever revision is
// persisted by the time it is built.
func (b *Broker) activationOf(revision string) policyActivation {
	r := b.nodePolicyRuntime()
	running := b.policyProxy()
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.activationLocked(running, revision)
}

// peerPolicyGetResult is the policy:get result a paired node receives: the
// peer projection, without the routing section or API tier entries.
type peerPolicyGetResult struct {
	ModelConfiguration nodepolicy.ModelConfiguration `json:"modelConfiguration"`
	Policy             nodepolicy.PeerPolicy         `json:"policy"`
	Availability       nodepolicy.Availability       `json:"availability"`
}

func (b *Broker) peerPolicy() peerPolicyGetResult {
	local := b.localPolicy()
	return peerPolicyGetResult{ModelConfiguration: local.ModelConfiguration, Policy: local.Policy.PeerProjection(), Availability: local.Availability}
}

// setLocalPolicy validates, persists and applies a new policy. Availability and
// its pausedBy provenance are not part of what policy:set changes —
// node:set-availability owns them — so the incoming values are ignored and the
// persisted ones kept. The same holds for the revision, which only the broker
// assigns. Omitted fields take their defaults,
// exactly as when the file is loaded. The write applies only when the persisted
// revision is the one the caller's draft is based on (expected).
func (b *Broker) setLocalPolicy(raw, expected json.RawMessage) (nodepolicy.Policy, error) {
	want, err := parseExpectedRevision(expected)
	if err != nil {
		return nodepolicy.Policy{}, err
	}
	return b.persistPolicy(func(stored nodepolicy.Policy) (nodepolicy.Policy, error) {
		if err := checkExpectedRevision(stored, want); err != nil {
			return nodepolicy.Policy{}, err
		}
		if len(raw) == 0 {
			return nodepolicy.Policy{}, fmt.Errorf("policy is required")
		}
		if err := nodepolicy.CheckModelListKeys(raw); err != nil {
			return nodepolicy.Policy{}, fmt.Errorf("parse policy: %w", err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
			return nodepolicy.Policy{}, fmt.Errorf("policy must be a JSON object")
		}
		dropFields(fields, "availability", "pausedBy", "revision")
		stripped, err := json.Marshal(fields)
		if err != nil {
			return nodepolicy.Policy{}, err
		}
		policy, err := nodepolicy.Parse(stripped)
		if err != nil {
			return nodepolicy.Policy{}, err
		}
		policy.Availability = stored.Availability
		policy.PausedBy = stored.PausedBy
		if err := policy.Validate(); err != nil {
			return nodepolicy.Policy{}, err
		}
		if l := policy.Routing.Listener; l.Enabled {
			if owner := b.facadePortOwner(l.Port); owner != "" {
				return nodepolicy.Policy{}, fmt.Errorf("routing.listener.port %d is the port of %s; choose another port, because the listener and a facade cannot share one", l.Port, owner)
			}
		}
		return policy, nil
	})
}

// setPeerPolicy applies a paired node's policy:set. The peer may change device
// fields only: its tiers and tierPolicy must equal the peer projection of the
// stored policy and it must send no routing section; the protected sections are
// restored from the stored policy. Like a local set, it applies only against
// the persisted revision the peer's draft is based on (expected).
func (b *Broker) setPeerPolicy(raw, expected json.RawMessage) (nodepolicy.Policy, error) {
	return b.setPeerPolicyAs(raw, expected, "")
}

// setPeerPolicyAs is setPeerPolicy for a verified caller. An empty caller is
// this node's own path; a paired node's permission is rechecked inside the
// policy lock, immediately before the merged document could be saved.
func (b *Broker) setPeerPolicyAs(raw, expected json.RawMessage, caller string) (nodepolicy.Policy, error) {
	want, err := parseExpectedRevision(expected)
	if err != nil {
		return nodepolicy.Policy{}, &invalidParamsError{err}
	}
	r := b.nodePolicyRuntime()
	return b.persistPolicy(func(stored nodepolicy.Policy) (nodepolicy.Policy, error) {
		// build runs under r.mu. Recovery is the owner's to complete.
		if err := b.authorizeRemoteManagementLocked(r, caller, "policy", ""); err != nil {
			return nodepolicy.Policy{}, err
		}
		if err := checkExpectedRevision(stored, want); err != nil {
			return nodepolicy.Policy{}, err
		}
		if r.recoveryRequired {
			return nodepolicy.Policy{}, errors.New("this node's policy needs recovery; set it from this node")
		}
		if err := nodepolicy.CheckModelListKeys(raw); err != nil {
			return nodepolicy.Policy{}, &invalidParamsError{err}
		}
		policy, err := nodepolicy.MergePeerSet(stored, withoutRevision(raw))
		if err != nil {
			return nodepolicy.Policy{}, &invalidParamsError{err}
		}
		return policy, nil
	})
}

// dropFields removes the named fields a caller sent, whatever their spelling:
// the policy parser matches keys case-insensitively, and only the broker sets
// these fields.
func dropFields(fields map[string]json.RawMessage, names ...string) {
	for key := range fields {
		for _, name := range names {
			if strings.EqualFold(key, name) {
				delete(fields, key)
			}
		}
	}
}

// withoutRevision returns raw without its revision key. A document that is not
// a JSON object is returned as it is, for MergePeerSet to refuse.
func withoutRevision(raw json.RawMessage) json.RawMessage {
	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil || fields == nil {
		return raw
	}
	dropFields(fields, "revision")
	stripped, err := json.Marshal(fields)
	if err != nil {
		return raw
	}
	return stripped
}

// persistPolicy persists and applies the policy that build derives from the
// stored one. build runs in the same critical section as the save, so a policy
// derived from the stored document can never be saved over a newer one. The
// saved document carries the next revision. A build or pre-replacement error
// pushes and announces nothing and leaves the previous document in memory.
// Every landed replacement ends policy recovery and is followed by one push whose
// acknowledgement, within policyActivationTimeout in all, activates the
// revision; a push that fails or does not finish in time leaves the revision
// persisted and not active. A landed replacement whose durability failed is
// adopted and applied the same way, then reported to the caller as a server error.
func (b *Broker) persistPolicy(build func(stored nodepolicy.Policy) (nodepolicy.Policy, error)) (nodepolicy.Policy, error) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	policy, err := build(r.current)
	if err != nil {
		r.mu.Unlock()
		return nodepolicy.Policy{}, err
	}
	policy, err = r.publishLocked(policy)
	if err != nil && !errors.Is(err, durablefile.ErrReplaced) {
		r.mu.Unlock()
		return nodepolicy.Policy{}, &policySaveError{err: err}
	}
	listenerChanged := policy.Routing.Listener.Enabled != r.current.Routing.Listener.Enabled ||
		policy.Routing.Listener.Port != r.current.Routing.Listener.Port
	refsChanged := credentialRefsDiffer(policy, r.current)
	r.current = policy
	r.recoveryRequired = false
	r.activationErr = ""
	r.mu.Unlock()
	b.publishNodeProjection()
	// A listener the policy just enabled has no proxy to activate it in until
	// the background start has replayed the policy, so its revision is pending
	// until then. A changed credential reference set needs a proxy launched
	// with it.
	if listenerChanged || refsChanged {
		b.reconcileProxySupervision()
	}
	b.activatePolicyNow()
	b.notifyPolicyChanged()
	if err != nil {
		return policy, &policySaveError{err: err, revision: policy.Revision}
	}
	return policy, nil
}

// notifyClients sends a notification to every client. These are low-volume
// state changes, so they need no subscription.
func (b *Broker) notifyClients(method string, params any) {
	if b.codec == nil {
		return
	}
	if err := b.codec.Notify(method, params); err != nil {
		slog.Warn("notify clients failed", "method", method, "err", err)
	}
}

// handlePolicyRequest serves the broker-owned node policy methods, and holds
// back engine:wake while the node is not available. It reports whether it
// answered the message; an engine:wake that may proceed falls through to the
// ordinary engine relay.
func (b *Broker) handlePolicyRequest(msg *Message) bool {
	switch msg.Method {
	case methodPolicyGet, methodPolicySet, methodSetAvailability:
		go b.servePolicyRequest(msg)
		return true
	case methodWorkloadsCancel:
		go b.handleWorkloadsCancel(msg)
		return true
	case engineWakeMethod:
		if reason := b.wakeRefusal(); reason != "" {
			_ = b.codec.RespondError(msg.ID, -32000, reason)
			return true
		}
	}
	return false
}

// wakeRefusal is why nothing may be started right now, or "" when it may.
func (b *Broker) wakeRefusal() string {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	live := r.live
	r.mu.Unlock()
	if live != nodepolicy.Available {
		return fmt.Sprintf("this node is %s; engines are not started on demand", live)
	}
	return ""
}

func (b *Broker) servePolicyRequest(msg *Message) {
	var p policyParams
	if len(msg.Params) > 0 && json.Unmarshal(msg.Params, &p) != nil {
		_ = b.codec.RespondError(msg.ID, -32602, "invalid params")
		return
	}
	if p.NodeID != "" && p.NodeID != b.nodeID {
		b.relayPolicyToPeer(msg, p)
		return
	}
	result, err := b.dispatchPolicy(context.Background(), msg.Method, p)
	if err != nil {
		_ = b.codec.RespondError(msg.ID, policyErrorCode(err), err.Error())
		return
	}
	_ = b.codec.Respond(msg.ID, result)
}

// invalidParamsError marks a request the caller must change before retrying.
type invalidParamsError struct{ err error }

func (e *invalidParamsError) Error() string { return e.err.Error() }

func policyErrorCode(err error) int {
	var invalid *invalidParamsError
	var conflict *policyConflictError
	var veto *localVetoError
	var denied *remoteManagementDeniedError
	switch {
	case errors.As(err, &invalid):
		return -32602
	case errors.As(err, &conflict):
		return codePolicyConflict
	case errors.As(err, &veto):
		return codeLocalVeto
	case errors.As(err, &denied):
		return codeRemoteManagementDenied
	}
	return -32000
}

// localVetoError refuses peer management while this node is pausing or paused.
// Only a resume made on this node lifts the boundary, for either pause origin.
type localVetoError struct{}

func (*localVetoError) Error() string {
	return "This node is paused. Resume it on this node before managing it remotely."
}

// pausedLocally reports whether policy is paused by this node itself. A pause
// with no provenance, such as one saved before pausedBy existed or the
// recovery policy's, counts as local.
func pausedLocally(policy nodepolicy.Policy) bool {
	return policy.Availability == nodepolicy.Paused && policy.PausedBy != nodepolicy.PausedByPeer
}

// policySaveError marks a publication or durability failure. A nonempty revision
// identifies the document already saved and being applied. The request was fine,
// so either failure is an ordinary server error, not invalid params.
type policySaveError struct {
	err      error
	revision string
}

func (e *policySaveError) Error() string {
	if e.revision != "" {
		return fmt.Sprintf("the policy document with revision %s was saved and is being applied but may not survive a power loss: %v", e.revision, e.err)
	}
	return "the policy could not be saved: " + e.err.Error()
}

func (e *policySaveError) Unwrap() error { return e.err }

// dispatchPolicy runs one policy method against this node for a local client,
// with the full policy. The relay from paired nodes shares it for availability
// and answers policy:get and policy:set from the peer projection instead;
// neither path can reach another node through it.
func (b *Broker) dispatchPolicy(ctx context.Context, method string, p policyParams) (any, error) {
	switch method {
	case methodPolicyGet:
		return b.localPolicy(), nil
	case methodPolicySet:
		policy, err := b.setLocalPolicy(p.Policy, p.ExpectedPersistedRevision)
		if err != nil {
			var conflict *policyConflictError
			var save *policySaveError
			if errors.As(err, &conflict) || errors.As(err, &save) {
				return nil, err
			}
			return nil, &invalidParamsError{err}
		}
		return policySetResult{ModelConfiguration: b.modelConfigurationFor(policy), Policy: policy, policyActivation: b.activationOf(policy.Revision)}, nil
	case methodSetAvailability:
		availability, err := b.setAvailabilityParams(ctx, p, nodepolicy.PausedLocally, "")
		if err != nil {
			return nil, err
		}
		local := b.localPolicy()
		return availabilitySetResult{Availability: availability, Revision: local.Policy.Revision, policyActivation: local.policyActivation}, nil
	default:
		return nil, fmt.Errorf("unsupported policy method %q", method)
	}
}

// setAvailabilityParams is the node:set-availability request shared by the local
// method and the paired node's relayed one; origin says which of the two made it
// and caller is the verified paired node (empty for a local caller).
func (b *Broker) setAvailabilityParams(ctx context.Context, p policyParams, origin nodepolicy.PauseOrigin, caller string) (nodepolicy.Availability, error) {
	state := nodepolicy.Availability(p.State)
	if state != nodepolicy.Available && state != nodepolicy.Paused {
		return "", &invalidParamsError{fmt.Errorf("state must be %q or %q", nodepolicy.Available, nodepolicy.Paused)}
	}
	return b.setAvailabilityAs(ctx, state, origin, caller)
}

// relayPolicyToPeer forwards a policy call aimed at another node through
// engine-manager's pinned ec surface. On success the change is announced to
// this node's clients under the target's nodeId, so a UI watching the cluster
// sees it without polling.
func (b *Broker) relayPolicyToPeer(msg *Message, p policyParams) {
	worker := b.getEngineMgr()
	if worker == nil {
		_ = b.codec.RespondError(msg.ID, -32000, "engine manager unavailable")
		return
	}
	remote := remotePolicyMethods[msg.Method]
	err := worker.RelayRequest(remote, msg.Params, func(result json.RawMessage, rpcErr *RPCError, err error) {
		if err == nil && rpcErr == nil && msg.Method == methodPolicySet {
			err = residentReadbackSupported(p.Policy, result)
		}
		switch {
		case err != nil:
			_ = b.codec.RespondError(msg.ID, -32000, err.Error())
		case rpcErr != nil:
			_ = b.codec.RespondError(msg.ID, rpcErr.Code, rpcErr.Message)
		default:
			_ = b.codec.Respond(msg.ID, result)
			b.announceRemotePolicyChange(msg.Method, p.NodeID, result)
		}
	})
	if err != nil {
		_ = b.codec.RespondError(msg.ID, -32000, err.Error())
	}
}

func residentReadbackSupported(request, result json.RawMessage) error {
	var want map[string]json.RawMessage
	var got struct {
		Policy map[string]json.RawMessage `json:"policy"`
	}
	if json.Unmarshal(request, &want) != nil || json.Unmarshal(result, &got) != nil {
		return fmt.Errorf("unsupported policy readback")
	}
	for _, key := range []string{"servedModels", "residentModels"} {
		raw, present := want[key]
		if !present {
			continue
		}
		var refs, received []nodepolicy.Role
		if json.Unmarshal(raw, &refs) != nil {
			return fmt.Errorf("unsupported policy readback")
		}
		actual, exists := got.Policy[key]
		if key == "residentModels" && len(refs) == 0 && !exists {
			continue
		}
		if !exists || json.Unmarshal(actual, &received) != nil || !reflect.DeepEqual(refs, received) {
			return fmt.Errorf("target does not preserve %s; served/resident edits are unsupported", key)
		}
	}
	return nil
}

func (b *Broker) announceRemotePolicyChange(method, nodeID string, result json.RawMessage) {
	var r struct {
		ModelConfiguration *nodepolicy.ModelConfiguration `json:"modelConfiguration,omitempty"`
		Policy             json.RawMessage                `json:"policy"`
		Availability       string                         `json:"availability"`
	}
	if json.Unmarshal(result, &r) != nil {
		return
	}
	switch method {
	case methodPolicySet:
		if len(r.Policy) > 0 {
			payload := map[string]any{"nodeId": nodeID, "policy": r.Policy}
			if r.ModelConfiguration != nil {
				payload["modelConfiguration"] = r.ModelConfiguration
			}
			b.notifyClients(notifyPolicyChanged, payload)
		}
	case methodSetAvailability:
		if r.Availability != "" {
			b.notifyClients(notifyAvailabilityChanged, map[string]any{"nodeId": nodeID, "availability": r.Availability})
		}
	}
}

// policyRelayRequest is a paired node's policy call, relayed by engine-manager.
type policyRelayRequest struct {
	ID     string          `json:"id"`
	Method string          `json:"method"`
	Caller string          `json:"caller,omitempty"`
	Params json.RawMessage `json:"params"`
}

type policyRelayReply struct {
	ID        string `json:"id"`
	Result    any    `json:"result"`
	Error     string `json:"error,omitempty"`
	ErrorCode int    `json:"errorCode,omitempty"`
}

// relayedPolicyMethods maps the relay's method names to the client methods.
var relayedPolicyMethods = map[string]string{
	"get":              methodPolicyGet,
	"set":              methodPolicySet,
	"set-availability": methodSetAvailability,
}

// handlePolicyRelay serves a paired node's policy call. The caller must still
// be pinned, and the call always targets this node: a relayed request never
// reaches relayPolicyToPeer, so it cannot be forwarded again. A peer sees and
// changes the peer projection only (servePolicyRelay).
func (b *Broker) handlePolicyRelay(source *engineManagerSource, raw json.RawMessage) {
	var req policyRelayRequest
	if json.Unmarshal(raw, &req) != nil || req.ID == "" {
		return
	}
	if !b.engineManagerSourceCurrent(source) {
		return
	}
	r := b.nodePolicyRuntime()
	ctx, cancel := context.WithCancel(context.Background())
	r.mu.Lock()
	if len(r.relayCancels) >= 64 {
		r.mu.Unlock()
		cancel()
		b.replyPeerRelay(source, b.getEngineMgr(), policyRelayReplyMethod, policyRelayReply{ID: req.ID, Error: "too many policy requests", ErrorCode: -32000})
		return
	}
	r.relayCancels[req.ID] = cancel
	r.mu.Unlock()
	go func() {
		defer cancel()
		defer func() { r.mu.Lock(); delete(r.relayCancels, req.ID); r.mu.Unlock() }()
		worker, err := b.awaitEngineMgrPair(ctx, source)
		if err != nil || !b.engineManagerSourceCurrent(source) {
			return
		}
		reply := policyRelayReply{ID: req.ID}
		result, err := b.servePolicyRelay(ctx, req)
		if err != nil {
			reply.Error = err.Error()
			reply.ErrorCode = policyErrorCode(err)
		} else {
			reply.Result = result
		}
		b.replyPeerRelay(source, worker, policyRelayReplyMethod, reply)
	}()
}

func (b *Broker) servePolicyRelay(ctx context.Context, req policyRelayRequest) (any, error) {
	method, ok := relayedPolicyMethods[req.Method]
	if !ok {
		return nil, fmt.Errorf("unsupported policy method")
	}
	// A relay notification always carries the peer engine-manager verified.
	// Without it there is no caller to recheck, so it must not be treated as a
	// local request.
	if err := requireRemoteCaller(req.Caller); err != nil {
		return nil, err
	}
	if !clustertrust.Open(b.clusterDir).HasPin(req.Caller) {
		return nil, fmt.Errorf("the requesting device is no longer paired")
	}
	var p policyParams
	if len(req.Params) > 0 && json.Unmarshal(req.Params, &p) != nil {
		return nil, &invalidParamsError{fmt.Errorf("invalid policy request")}
	}
	p.NodeID = ""
	slog.Info("serving a paired node's policy request", "method", method, "caller", req.Caller)
	switch method {
	case methodPolicyGet:
		return b.peerPolicy(), nil
	case methodPolicySet:
		policy, err := b.setPeerPolicyAs(p.Policy, p.ExpectedPersistedRevision, req.Caller)
		if err != nil {
			return nil, err
		}
		return map[string]any{"policy": policy.PeerProjection(), "modelConfiguration": b.modelConfigurationFor(policy)}, nil
	}
	// The peer sees what its policy:get shows of this node, which has no
	// activation fields. A resume of a node paused on the node itself is
	// refused (setAvailability).
	availability, err := b.setAvailabilityParams(ctx, p, nodepolicy.PausedByPeer, req.Caller)
	if err != nil {
		return nil, err
	}
	return map[string]any{"availability": availability}, nil
}

func (b *Broker) cancelPolicyRelay(raw json.RawMessage) {
	var p struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(raw, &p) != nil {
		return
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	cancel := r.relayCancels[p.ID]
	r.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// handlePolicyEngineNotification observes engine-manager's stream for the
// policy. It consumes the policy relay's own notifications and reports true for
// them; everything else is only observed and continues to subscribed clients.
func (b *Broker) handlePolicyEngineNotification(source *engineManagerSource, method string, params json.RawMessage) bool {
	if (method == engineModelsChangedMethod || method == engineIntentChanged) && !b.engineManagerSourceCurrent(source) {
		return false
	}
	switch method {
	case policyRelayRequestMethod:
		b.handlePolicyRelay(source, params)
		return true
	case policyRelayCancelMethod:
		b.cancelPolicyRelay(params)
		return true
	case engineModelsChangedMethod:
		var p struct {
			Models struct {
				Models         []string            `json:"models"`
				LoadedByEngine map[string][]string `json:"loadedByEngine"`
			} `json:"models"`
		}
		if json.Unmarshal(params, &p) == nil && (p.Models.Models != nil || p.Models.LoadedByEngine != nil) {
			b.observeLoadedFrom(source, p.Models.LoadedByEngine)
		}
	case engineIntentChanged:
		var p engineintent.Snapshot
		if json.Unmarshal(params, &p) == nil && p.Validate() == nil {
			b.observeIntentFrom(source, p.EnabledByEngine)
		}
	}
	return false
}

// observeEngineRunningFrom records a running/stopped transition only from the
// current source. A stopped engine has no loaded models or run identity.
func (b *Broker) observeEngineRunningFrom(source *engineManagerSource, engine string, running bool) bool {
	return b.observeEngineRunningFromSnapshot(source, engine, running, nil)
}

func (b *Broker) observeEngineRunningFromSnapshot(source *engineManagerSource, engine string, running bool, snapshotGeneration *uint64) bool {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if source == nil || r.engineManagerSource != source {
		r.mu.Unlock()
		return false
	}
	if snapshotGeneration != nil && r.runningObserved[engine] > *snapshotGeneration {
		r.mu.Unlock()
		return false
	}
	r.runningGeneration++
	r.runningObserved[engine] = r.runningGeneration
	was := r.running[engine]
	if running {
		r.running[engine] = true
		if !was {
			r.runningSince[engine] = r.now()
		}
	} else {
		delete(r.running, engine)
		delete(r.loaded, engine)
	}
	changed := was != running
	// A stopped engine's run identity is void now, before engine-manager's
	// next snapshot says so.
	_, identified := r.engineIdentity[engine]
	if !running {
		delete(r.engineIdentity, engine)
	}
	r.mu.Unlock()
	if changed {
		b.schedulePolicyPush(pushResidency)
	}
	if !running && identified {
		b.identityChanged()
	}
	return true
}

// observeLoaded folds engine-manager's loadedByEngine into the residency. Only
// engines that answered are present. An absent engine has unknown inventory;
// an explicit empty slice establishes that no models are loaded.
func (b *Broker) observeLoaded(loadedByEngine map[string][]string) {
	b.observeLoadedFrom(nil, loadedByEngine)
}

func (b *Broker) observeLoadedFrom(source *engineManagerSource, loadedByEngine map[string][]string) {
	b.observeLoadedFor(source, b.policyProxy(), loadedByEngine)
}

func (b *Broker) observeLoadedFor(source *engineManagerSource, pp *proxyProcess, loadedByEngine map[string][]string) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if source != nil && source != r.engineManagerSource {
		r.mu.Unlock()
		return
	}
	if r.models.loadedSource != r.engineManagerSource || r.models.loadedProxy != pp {
		r.models.loadedRequests = nil
	}
	r.models.loadedSource = r.engineManagerSource
	r.models.loadedProxy = pp
	if r.models.loadedRequests == nil {
		r.models.loadedRequests = make(map[nodepolicy.Role]nodepolicy.ResidentLoadOptions)
	}
	for ref := range r.models.loadedRequests {
		if models, known := loadedByEngine[ref.Engine]; known && models != nil {
			found := false
			for _, model := range models {
				if b.modelRef(ref.Engine, model) == ref {
					found = true
					break
				}
			}
			if !found {
				delete(r.models.loadedRequests, ref)
			}
		}
	}
	for _, ref := range b.residentRefs(r.current) {
		if _, known := r.models.loadedRequests[ref]; known {
			continue
		}
		for _, model := range loadedByEngine[ref.Engine] {
			if b.modelRef(ref.Engine, model) == ref {
				if pr, qualified := b.residentProfile(r.current, ref); qualified {
					r.models.loadedRequests[ref] = residentLoadRequest(pr)
				}
				break
			}
		}
	}
	r.models.loadedIdentity = r.modelIdentityLocked()
	for ref := range r.models.confirmed {
		models, known := loadedByEngine[ref.Engine]
		if !known || models == nil {
			continue // Unknown inventory neither confirms nor contradicts the proof.
		}
		found := false
		for _, model := range models {
			if b.modelRef(ref.Engine, model) == ref {
				found = true
			}
		}
		if !found {
			delete(r.models.confirmed, ref)
		}
	}
	for ref, pending := range r.models.pending {
		if models, known := loadedByEngine[ref.Engine]; known && models != nil && !pending.published {
			// A later known publication is newer than the load, whatever it lists.
			pending.published = true
			r.models.pending[ref] = pending
			r.models.last = nil
		}
	}
	now := r.now()
	for engine, models := range loadedByEngine {
		if !r.running[engine] {
			r.running[engine] = true
			r.runningGeneration++
			r.runningObserved[engine] = r.runningGeneration
			r.runningSince[engine] = now
		}
		prev := r.loaded[engine]
		for _, m := range models {
			if !containsString(prev, m) {
				r.residencyAt[engine] = now
				break
			}
		}
	}
	r.loaded = make(map[string][]string, len(loadedByEngine))
	for engine, models := range loadedByEngine {
		if models != nil {
			r.loaded[engine] = append([]string{}, models...)
		}
	}
	r.mu.Unlock()
	b.schedulePolicyPush(pushResidency)
}

func (b *Broker) observeIntent(enabledByEngine map[string]bool) {
	b.observeIntentFrom(nil, enabledByEngine)
}

func (b *Broker) observeIntentFrom(source *engineManagerSource, enabledByEngine map[string]bool) {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if source != nil && source != r.engineManagerSource {
		r.mu.Unlock()
		return
	}
	r.intent = make(map[string]bool, len(enabledByEngine))
	for k, v := range enabledByEngine {
		r.intent[k] = v
	}
	r.mu.Unlock()
	b.schedulePolicyPush(pushIntent)
	// An engine saved Off leaves the strict capability. Off the read pump.
	go b.refreshStrictAdvertisement()
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// refreshEnginePolicySnapshotFrom seeds running engines, residency and saved
// intent from a (re)started engine-manager, whose watcher pushes only changes,
// and sends it the node projection. engine:ready sends the projection too, but
// it can be read before the new worker's handle is installed; this runs after.
// The worker's bound source fences running-state observations during replay.
func (b *Broker) refreshEnginePolicySnapshotFrom(w *rpcWorker, source *engineManagerSource) {
	if w == nil {
		return
	}
	b.publishNodeProjection()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if result, rpcErr, err := w.CallNoTimeout(ctx, engineIntentMethod, nil); err == nil && rpcErr == nil {
		var p engineintent.Snapshot
		if json.Unmarshal(result, &p) == nil && p.Validate() == nil {
			b.observeIntentFrom(source, p.EnabledByEngine)
		}
	}
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	runningGeneration := r.runningGeneration
	r.mu.Unlock()
	if result, rpcErr, err := w.CallNoTimeout(ctx, "engine:get-installed", nil); err == nil && rpcErr == nil {
		var p struct {
			Engines []struct {
				Engine  string `json:"engine"`
				Running bool   `json:"running"`
			} `json:"engines"`
		}
		if json.Unmarshal(result, &p) == nil {
			for _, e := range p.Engines {
				b.observeEngineRunningFromSnapshot(source, e.Engine, e.Running, &runningGeneration)
			}
		}
	}
	inventoryProxy := b.policyProxy()
	if result, rpcErr, err := w.CallNoTimeout(ctx, "engine:models", nil); err == nil && rpcErr == nil && b.policyProxy() == inventoryProxy {
		var p struct {
			Models         []string            `json:"models"`
			LoadedByEngine map[string][]string `json:"loadedByEngine"`
		}
		if json.Unmarshal(result, &p) == nil && (p.Models != nil || p.LoadedByEngine != nil) {
			b.observeLoadedFor(source, inventoryProxy, p.LoadedByEngine)
		}
	}
}

// Proxy pushes. Each kind is coalesced: scheduling one that is already pending
// is a no-op, and the push reads the state when it is sent.
const (
	pushPolicy       = "policy"
	pushAvailability = "availability"
	pushResidency    = "residency"
	pushIntent       = "intent"
	pushIdentity     = "identity"
	pushDrainPrefix  = "drain:"
)

// policyProxy is the proxy process, which every engine's facade and the routing
// listener live in. Only the process handle says the process is alive, and a
// proxy that serves no facade has no other.
func (b *Broker) policyProxy() *proxyProcess {
	return b.proxyProcessHandle()
}

// acquirePush takes the push slot, giving up when ctx ends first.
func (r *policyRuntime) acquirePush(ctx context.Context) bool {
	select {
	case r.pushSlot <- struct{}{}:
		return true
	default:
	}
	select {
	case r.pushSlot <- struct{}{}:
		return true
	case <-ctx.Done():
		return false
	}
}

func (r *policyRuntime) releasePush() { <-r.pushSlot }

// schedulePolicyPush sends one kind of state to the proxy in the background.
func (b *Broker) schedulePolicyPush(kind string) {
	b.signalModelConfig()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	if r.pendingPush[kind] {
		r.mu.Unlock()
		return
	}
	r.pendingPush[kind] = true
	r.mu.Unlock()
	go func() {
		r.acquirePush(context.Background())
		r.mu.Lock()
		delete(r.pendingPush, kind)
		r.mu.Unlock()
		changed := b.sendPolicyStateLocked(context.Background(), b.policyProxy(), kind)
		r.releasePush()
		// Nothing waits on this push, so nothing else would tell clients that
		// it activated, or failed to activate, the revision.
		if changed {
			b.notifyPolicyChanged()
		}
	}()
}

// pushPolicyNow sends one kind of state and returns once the proxy answered,
// for the steps that must not continue before the proxy knows. It waits for
// the push slot as long as it takes, so it is for state the node must not run
// without; activatePolicyNow is the bounded wait for a revision.
// It reports whether the answer changed what the broker reports of activation.
func (b *Broker) pushPolicyNow(ctx context.Context, kind string) (activationChanged bool) {
	r := b.nodePolicyRuntime()
	r.acquirePush(context.Background())
	defer r.releasePush()
	return b.sendPolicyStateLocked(ctx, b.policyProxy(), kind)
}

// announceActivationChange tells clients about the persisted policy when a
// push changed whether the proxy enforces it.
func (b *Broker) announceActivationChange(changed bool) {
	if changed {
		b.notifyPolicyChanged()
	}
}

// activatePolicyNow pushes the persisted policy and waits for the proxy's
// acknowledgement of it, getting the push slot and the answer both within
// policyActivationTimeout. When the slot is not free in time, as while a
// replay is still talking to a slow proxy, or is got with too little of the
// budget left to send the request and hear back, nothing was sent: the revision
// stays pending and the push is left scheduled, so it still goes out. A request
// that was sent and not answered in time is a failed activation instead.
func (b *Broker) activatePolicyNow() {
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	bound := r.activationBoundLocked()
	r.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), bound)
	defer cancel()
	if !r.acquirePush(ctx) {
		b.schedulePolicyPush(pushPolicy)
		return
	}
	defer r.releasePush()
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < bound/activationFloorDivisor {
		b.schedulePolicyPush(pushPolicy)
		return
	}
	b.sendPolicyStateLocked(ctx, b.policyProxy(), pushPolicy)
}

// replayPolicyToProxy re-sends everything to a freshly started proxy. It runs
// before the proxy's facades are enabled, so a paused node is paused before it
// can admit anything. Availability goes first for the same reason.
func (b *Broker) replayPolicyToProxy(ctx context.Context, pp *proxyProcess) bool {
	r := b.nodePolicyRuntime()
	if !r.acquirePush(ctx) {
		return false
	}
	defer r.releasePush()
	r.mu.Lock()
	// Work in flight on the old process died with it, and so did what it
	// acknowledged: only this process's acknowledgements count from here.
	r.admission = nodepolicy.AdmissionState{}
	r.haveAdmission = false
	r.signalAdmissionLocked()
	r.incarnation, r.runID, r.activeRevision, r.activationErr, r.availabilityErr = pp, "", "", "", ""
	if r.egress.process != pp {
		r.egress = brokerEgressLease{process: pp, now: r.egress.now}
	}
	// So did its strict readiness. The new process reports its own once it
	// has acknowledged the policy replayed below.
	r.strictReadiness = nodepolicy.StrictAdmissionReadiness{}
	kinds := []string{pushAvailability, pushPolicy, pushIntent, pushResidency, pushIdentity}
	for engine := range r.drains {
		kinds = append(kinds, pushDrainPrefix+engine)
	}
	r.mu.Unlock()
	b.refreshStrictAdvertisement()
	b.withdrawStrictReadinessOnExit(pp)
	for _, kind := range kinds {
		b.sendPolicyStateLocked(ctx, pp, kind)
	}
	if err := b.replaySwitchInvalidations(ctx, pp); err != nil {
		slog.Error("switch invalidation replay blocked proxy facades", "err", err)
		return false
	}
	b.notifyPolicyChanged()
	return true
}

// pushSwitchDrain returns the actual acknowledgement and validates the live
// incarnation. Its caller's deadline covers push-slot contention as well.
func (b *Broker) pushSwitchDrain(ctx context.Context, engine string, drain bool) (string, error) {
	r := b.nodePolicyRuntime()
	if !r.acquirePush(ctx) {
		return "", ctx.Err()
	}
	defer r.releasePush()
	pp := b.policyProxy()
	r.mu.Lock()
	runID, incarnation := r.runID, r.incarnation
	r.mu.Unlock()
	if pp == nil || pp != incarnation || runID == "" {
		return "", fmt.Errorf("proxy incarnation is unverified")
	}
	if err := switchDrainAck(ctx, pp, engine, drain); err != nil {
		return "", err
	}
	r.mu.Lock()
	current := r.incarnation == pp && r.runID == runID
	r.mu.Unlock()
	if !current || b.policyProxy() != pp {
		return "", fmt.Errorf("proxy incarnation changed during drain push")
	}
	return runID, nil
}

// sendPolicyStateLocked sends the current state of one kind. Caller holds the
// push slot. It reports whether the answer changed what the broker reports of
// activation, for a caller that has no result of its own to announce it in. An older proxy without the method only logs.
//
// A revision is active only when the proxy acknowledged the availability it
// carries as well as the policy, so a policy push first sends the availability
// again when the newest availability push was not acknowledged.
func (b *Broker) sendPolicyStateLocked(ctx context.Context, pp *proxyProcess, kind string) (activationChanged bool) {
	if pp == nil {
		return false
	}
	r := &b.nodePolicy
	if kind == pushPolicy {
		r.mu.Lock()
		retry, bound := r.availabilityErr != "", r.activationBoundLocked()
		r.mu.Unlock()
		if retry {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, bound)
			defer cancel()
			activationChanged = b.sendPolicyStateLocked(ctx, pp, pushAvailability)
		}
	}
	r.mu.Lock()
	var method string
	var params any
	// activates is the revision this push asks the proxy to activate, and
	// availability the state a node/set-availability push carries.
	var activates string
	var availability nodepolicy.Availability
	switch {
	case kind == pushPolicy:
		r.egress.policyPending = true
		method, params, activates = nodepolicy.MethodSetPolicy, nodepolicy.SetPolicyParams{OriginNodeID: b.nodeID, Policy: r.current}, r.current.Revision
	case kind == pushAvailability:
		r.egress.availabilityPending = true
		availability = r.live
		method, params = nodepolicy.MethodSetAvailability, nodepolicy.SetAvailabilityParams{State: availability, CancelActive: r.cancelActive}
	case kind == pushResidency:
		loaded := make(map[string][]string, len(r.running))
		for engine := range r.running {
			models := append([]string{}, r.loaded[engine]...)
			sort.Strings(models)
			loaded[engine] = models
		}
		method, params = nodepolicy.MethodSetResidency, nodepolicy.SetResidencyParams{LoadedByEngine: loaded}
	case kind == pushIntent:
		intent := make(map[string]bool, len(r.intent))
		for k, v := range r.intent {
			intent[k] = v
		}
		method, params = nodepolicy.MethodSetEngineIntent, nodepolicy.SetEngineIntentParams{EnabledByEngine: intent}
	case kind == pushIdentity:
		method, params = nodepolicy.MethodSetModelIdentity, r.modelIdentityLocked()
	case len(kind) > len(pushDrainPrefix) && kind[:len(pushDrainPrefix)] == pushDrainPrefix:
		engine := kind[len(pushDrainPrefix):]
		method, params = nodepolicy.MethodSetEngineDrain, nodepolicy.SetEngineDrainParams{Engine: engine, Drain: r.drains[engine]}
	default:
		r.mu.Unlock()
		return activationChanged
	}
	data, err := json.Marshal(params)
	bound := r.activationBoundLocked()
	r.mu.Unlock()
	if err != nil {
		return activationChanged
	}
	if activates != "" {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, bound)
		defer cancel()
	}
	result, rpcErr, err := pp.Call(ctx, method, data)
	switch {
	case err != nil:
		slog.Warn("could not push node policy state to the proxy", "method", method, "err", err)
	case rpcErr != nil:
		slog.Warn("proxy refused node policy state", "method", method, "code", rpcErr.Code, "err", rpcErr.Message)
	}
	// The answer can make a held strict readiness report current or stale, so
	// the capability is reconsidered once it is recorded.
	switch {
	case activates != "":
		changed := b.recordActivation(pp, activates, result, rpcErr, err)
		b.refreshStrictAdvertisement()
		return changed || activationChanged
	case kind == pushAvailability:
		changed := b.recordAvailability(pp, availability, rpcErr, err)
		b.refreshStrictAdvertisement()
		return changed || activationChanged
	}
	return activationChanged
}

// recordAvailability takes the proxy's answer to a node/set-availability push
// of state. Caller holds the push slot. An answer from a process that is not the
// current incarnation is discarded. It reports whether the recorded failure
// changed, which changes what activation reports.
func (b *Broker) recordAvailability(pp *proxyProcess, state nodepolicy.Availability, rpcErr *RPCError, callErr error) bool {
	r := &b.nodePolicy
	r.mu.Lock()
	defer r.mu.Unlock()
	if pp != r.incarnation {
		slog.Debug("discarded an availability answer from a proxy that is not the current incarnation", "state", state)
		return false
	}
	was := r.availabilityErr
	switch {
	case callErr != nil:
		r.availabilityErr = fmt.Sprintf("the proxy did not acknowledge availability %s: %v", state, callErr)
	case rpcErr != nil:
		r.availabilityErr = fmt.Sprintf("the proxy refused availability %s (code %d)", state, rpcErr.Code)
	default:
		r.availabilityErr = ""
		r.egress.availabilityPending = false
	}
	return was != r.availabilityErr
}

// recordActivation takes the proxy's answer to a node/set-policy push of
// revision. Caller holds the push slot, which serializes activations. The
// revision becomes active only on an acknowledgement from the current
// incarnation, with the run id that incarnation first acknowledged under,
// naming exactly the revision pushed. Anything else leaves activeRevision as it
// was; a failed push also records why. It reports whether the active revision
// or the recorded failure changed.
func (b *Broker) recordActivation(pp *proxyProcess, revision string, result json.RawMessage, rpcErr *RPCError, callErr error) (changed bool) {
	r := &b.nodePolicy
	r.mu.Lock()
	defer r.mu.Unlock()
	if pp != r.incarnation {
		slog.Debug("discarded an answer from a proxy that is not the current incarnation", "revision", revision)
		return false
	}
	was := policyActivation{ActiveRevision: r.activeRevision, ActivationError: r.activationErr}
	defer func() {
		changed = was != policyActivation{ActiveRevision: r.activeRevision, ActivationError: r.activationErr}
	}()
	var ack nodepolicy.PolicyAck
	failure := ""
	switch {
	case callErr != nil:
		failure = fmt.Sprintf("the proxy did not acknowledge revision %s: %v", revision, callErr)
	case rpcErr != nil:
		failure = fmt.Sprintf("the proxy refused revision %s (code %d)", revision, rpcErr.Code)
	case json.Unmarshal(result, &ack) != nil || ack.RunID == "":
		failure = fmt.Sprintf("the proxy's acknowledgement of revision %s was unreadable", revision)
	case ack.OriginNodeID != b.nodeID:
		failure = fmt.Sprintf("the proxy acknowledged a different origin for revision %s", revision)
	case r.runID != "" && ack.RunID != r.runID:
		failure = fmt.Sprintf("an acknowledgement of revision %s came from a different proxy run", revision)
	case ack.Revision != revision:
		failure = fmt.Sprintf("the proxy acknowledged revision %q, not the revision %s that was sent", ack.Revision, revision)
	default:
		r.runID = ack.RunID
		if revision == r.current.Revision {
			r.egress.policyPending = false
		}
		if cmp, err := nodepolicy.CompareRevisions(revision, r.activeRevision); err == nil && cmp > 0 {
			r.activeRevision = revision
		}
		if cmp, err := nodepolicy.CompareRevisions(r.activeRevision, r.current.Revision); err == nil && cmp >= 0 {
			r.activationErr = ""
		}
		return
	}
	r.activationErr = failure
	return false
}

// handleAdmissionNotification takes the proxy's admission/* notifications and
// reports whether method was one of them.
func (b *Broker) handleAdmissionNotification(method string, params json.RawMessage) bool {
	switch method {
	case nodepolicy.NotifyAdmissionState:
		var st nodepolicy.AdmissionState
		if json.Unmarshal(params, &st) != nil {
			slog.Warn("proxy sent an invalid admission state")
			return true
		}
		b.observeAdmission(st)
		return true
	case nodepolicy.NotifyAdmissionUnload:
		var req nodepolicy.UnloadRequest
		if json.Unmarshal(params, &req) != nil || req.Engine == "" {
			slog.Warn("proxy sent an invalid unload request")
			return true
		}
		go b.unloadModels(req.Engine, req.Models, "admission")
		return true
	case nodepolicy.NotifyAdmissionWake:
		var req nodepolicy.WakeRequest
		if json.Unmarshal(params, &req) != nil || req.Engine == "" {
			slog.Warn("proxy sent an invalid wake request")
			return true
		}
		go b.wakeOnDemand(req.Engine)
		return true
	case nodepolicy.NotifyStrictReadiness:
		b.observeStrictReadiness(params)
		b.signalModelConfig()
		return true
	}
	return false
}

func (b *Broker) observeAdmission(st nodepolicy.AdmissionState) {
	defer b.signalModelConfig()
	r := b.nodePolicyRuntime()
	r.mu.Lock()
	activeChanged := r.admission.Active != st.Active
	r.admission = st
	r.haveAdmission = true
	r.signalAdmissionLocked()
	draining := r.live == nodepolicy.Draining
	r.mu.Unlock()
	// While draining, clients follow the count down to zero.
	if draining && activeChanged {
		b.notifyAvailability()
	}
}

// signalAdmissionLocked wakes everything waiting on an admission change.
func (r *policyRuntime) signalAdmissionLocked() {
	close(r.admissionCh)
	r.admissionCh = make(chan struct{})
}

// waitAdmission blocks until pred holds for the admission state, timeout
// passes, or ctx ends, and reports whether pred held. Without a proxy, or
// before it has reported anything, nothing is running here.
func (b *Broker) waitAdmission(ctx context.Context, timeout time.Duration, pred func(nodepolicy.AdmissionState) bool) bool {
	r := b.nodePolicyRuntime()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		noProxy := b.policyProxy() == nil
		r.mu.Lock()
		ok := noProxy || !r.haveAdmission || pred(r.admission)
		ch := r.admissionCh
		r.mu.Unlock()
		if ok {
			return true
		}
		select {
		case <-ch:
		case <-timer.C:
			return false
		case <-ctx.Done():
			return false
		}
	}
}

// callEngine makes one engine-manager call on the policy's behalf.
func (b *Broker) callEngine(ctx context.Context, method string, params any) (json.RawMessage, error) {
	return callEngineOn(ctx, b.getEngineMgr(), method, params)
}

// Recovery uses its pinned manager before ordinary callers may see it.
func callEngineOn(ctx context.Context, w *rpcWorker, method string, params any) (json.RawMessage, error) {
	if w == nil {
		return nil, fmt.Errorf("engine manager unavailable")
	}
	data, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, engineCallTimeout)
	defer cancel()
	result, rpcErr, err := w.CallNoTimeout(ctx, method, data)
	if err != nil {
		return nil, err
	}
	if rpcErr != nil {
		return nil, fmt.Errorf("%s", rpcErr.Message)
	}
	return result, nil
}

// unloadModels unloads models from one engine, one at a time.
func (b *Broker) unloadModels(engine string, models []string, reason string) {
	for _, model := range models {
		if model == "" {
			continue
		}
		if reason != "pause" {
			r := b.nodePolicyRuntime()
			r.mu.Lock()
			protected := b.protectedResidentLocked(engine, model)
			r.mu.Unlock()
			if protected {
				continue
			}
		}
		if _, err := b.callEngine(context.Background(), engineUnloadModelMethod, map[string]string{"engine": engine, "model": model}); err != nil {
			slog.Warn("model unload failed", "engine", engine, "model", model, "reason", reason, "err", err)
			continue
		}
		slog.Info("model unloaded", "engine", engine, "model", model, "reason", reason)
	}
}
