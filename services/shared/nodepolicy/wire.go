// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"errors"
	"fmt"
	"strings"
)

// StrictAdmissionReadiness is the params of NotifyStrictReadiness: what the
// proxy's admission controller can admit under the strict contract for the
// revision it names. Only the broker can match its fences to the acknowledged
// proxy incarnation and desired/active policy revision before publishing
// Capability in Advertisement. Nil withdraws previous readiness.
type StrictAdmissionReadiness struct {
	PolicyRevision string                     `json:"policyRevision"`
	ProxyRunID     string                     `json:"proxyRunId"`
	Capability     *StrictAdmissionCapability `json:"capability,omitempty"`
}

func (r StrictAdmissionReadiness) Validate() error {
	if err := validateRevision(r.PolicyRevision); err != nil {
		return fmt.Errorf("policyRevision: %w", err)
	}
	if !strictID.MatchString(r.ProxyRunID) {
		return errors.New("proxyRunId must be a nonempty identity of at most 128 ASCII characters")
	}
	if r.Capability != nil {
		return r.Capability.Validate()
	}
	return nil
}

// Broker → proxy requests. Each is answered with {"ok":true} or a JSON-RPC
// error, except MethodSetPolicy, which answers with a PolicyAck. The broker
// re-sends all of them whenever the proxy (re)starts.
const (
	// MethodSetPolicy carries the whole Policy and is answered with a
	// PolicyAck of its revision. Availability inside it is ignored by the
	// proxy; MethodSetAvailability is authoritative.
	MethodSetPolicy = "node/set-policy"
	// MethodSetAvailability carries SetAvailabilityParams. The broker follows
	// a persisted availability change with MethodSetPolicy, whose revision is
	// the availability's, so availability never activates a revision itself.
	MethodSetAvailability = "node/set-availability"
	// MethodSetEngineDrain carries SetEngineDrainParams.
	MethodSetEngineDrain = "node/set-engine-drain"
	// MethodSetResidency carries SetResidencyParams.
	MethodSetResidency = "node/set-residency"
	// MethodSetEngineIntent carries SetEngineIntentParams.
	MethodSetEngineIntent = "node/set-engine-intent"
	// MethodSetModelIdentity carries SetModelIdentityParams: the identity
	// the proxy binds strict capacity and qualification to. The broker sends
	// it whenever an engine's identity changes, and synchronously when one is
	// withdrawn, so withdrawal is in force before the change it guards.
	MethodSetModelIdentity = "node/set-model-identity"
)

// Proxy → broker notifications.
const (
	// NotifyAdmissionState carries AdmissionState. Sent at most once a second
	// while it changes, and immediately when the active count reaches zero.
	NotifyAdmissionState = "admission/state"
	// NotifyAdmissionUnload carries UnloadRequest: admission wants these idle
	// models unloaded to make room. The broker unloads them; the proxy learns
	// the result from the next MethodSetResidency.
	NotifyAdmissionUnload = "admission/unload"
	// NotifyAdmissionWake carries WakeRequest: a request arrived for a stopped
	// engine. The broker starts it only if its saved intent is On and the node
	// is not paused; the proxy learns the result from node/set-local-backend.
	NotifyAdmissionWake = "admission/wake"
	// NotifyStrictReadiness carries StrictAdmissionReadiness. The proxy sends
	// it after acknowledging each MethodSetPolicy, naming the revision it
	// acknowledged and the run id of that PolicyAck, and again whenever its
	// readiness for that revision changes. An omitted capability means not
	// ready. Order against the acknowledgement does not matter: the broker
	// holds a report until its fences match what it has recorded, discards one
	// from another proxy run than the acknowledged one, and withdraws on one
	// that does not validate. It clears readiness itself on proxy restart,
	// exit, pending activation and pause.
	NotifyStrictReadiness = "admission/strict-readiness"
)

// HTTP headers.
const (
	// AdmissionHeader is set on a 503 from admission with a RejectReason.
	AdmissionHeader = "X-PAIR-Admission"
	// AdmissionWaitHeader is sent by the routing node to a destination: the
	// longest, in whole seconds, the destination may queue this request. The
	// router sends 0 when it still has other candidates to try.
	AdmissionWaitHeader = "X-PAIR-Admission-Wait"
	// ModelHeader, EngineHeader and NodeHeader are set on every routed
	// inference response and name what actually answered.
	ModelHeader  = "X-PAIR-Model"
	EngineHeader = "X-PAIR-Engine"
	NodeHeader   = "X-PAIR-Node"
	// TierHeader echoes the tier a request asked for, when it asked for one.
	TierHeader = "X-PAIR-Tier"
)

// RejectReason says why admission refused a request.
type RejectReason string

const (
	RejectPaused   RejectReason = "paused"
	RejectDraining RejectReason = "draining"
	RejectBusy     RejectReason = "busy"
	RejectNoFit    RejectReason = "no-fit"
	RejectOff      RejectReason = "engine-off"
	RejectWake     RejectReason = "wake-timeout"
)

type SetAvailabilityParams struct {
	State Availability `json:"state"`
	// CancelActive cancels work already running on this node.
	CancelActive bool `json:"cancelActive,omitempty"`
}

// PolicyAck is the proxy's answer to MethodSetPolicy: Revision is the policy
// revision it activated, the one it was sent. RunID names the proxy process, so
// the broker can discard an acknowledgement from an earlier one. It is the only
// acknowledgement that activates a revision.
type PolicyAck struct {
	Revision string `json:"revision"`
	RunID    string `json:"runId"`
}

// CompareRevisions orders two revisions, each canonical decimal or empty;
// empty is the proxy's "no revision yet" and is older than any revision.
func CompareRevisions(a, b string) (int, error) {
	x, err := revisionValue(a)
	if err != nil {
		return 0, err
	}
	y, err := revisionValue(b)
	if err != nil {
		return 0, err
	}
	switch {
	case x < y:
		return -1, nil
	case x > y:
		return 1, nil
	}
	return 0, nil
}

// revisionValue maps the empty revision to -1, below every canonical one.
func revisionValue(rev string) (int64, error) {
	if rev == "" {
		return -1, nil
	}
	n, err := NanoUSD(rev).Int64()
	if err != nil {
		return 0, errors.New("revision must be a canonical decimal string")
	}
	return n, nil
}

type SetEngineDrainParams struct {
	Engine string `json:"engine"`
	Drain  bool   `json:"drain"`
}

// SetResidencyParams is the loaded model set per engine on this node. An
// engine missing from the map is not running.
type SetResidencyParams struct {
	LoadedByEngine map[string][]string `json:"loadedByEngine"`
}

// SetEngineIntentParams is each managed engine's saved On/Off intent, so the
// proxy can reject a request for an engine saved Off without asking to wake it.
type SetEngineIntentParams struct {
	EnabledByEngine map[string]bool `json:"enabledByEngine"`
}

// SetModelIdentityParams is the whole identity of every engine the proxy may
// admit strict work for. An engine missing from the map has no identity, which
// withdraws its strict capacity.
type SetModelIdentityParams struct {
	Engines map[string]EngineIdentity `json:"engines"`
}

// EngineIdentity is one running engine's identity. EngineRunID changes with
// every engine start or adoption (engine-manager). SettingsFingerprint covers
// the applied settings revision and its epoch, and LaunchDigest the effective
// launch arguments (broker). ModelDigests maps each installed model, by the
// engine's own name, to the lowercase SHA-256 hex identifying its content
// (engine-manager). All are public configuration or content identity.
type EngineIdentity struct {
	EngineRunID         string            `json:"engineRunId"`
	SettingsFingerprint string            `json:"settingsFingerprint"`
	LaunchDigest        string            `json:"launchDigest"`
	ModelDigests        map[string]string `json:"modelDigests"`
}

func (p SetModelIdentityParams) Validate() error {
	if p.Engines == nil {
		return errors.New("engines is required")
	}
	for engine, id := range p.Engines {
		if err := id.Validate(); err != nil {
			return fmt.Errorf("engines.%s: %w", engine, err)
		}
	}
	return nil
}

func (i EngineIdentity) Validate() error {
	if !strictID.MatchString(i.EngineRunID) {
		return errors.New("engineRunId must be a nonempty identity of at most 128 ASCII characters")
	}
	if !strictDigest.MatchString(i.SettingsFingerprint) {
		return errors.New("settingsFingerprint must be lowercase SHA-256 hex")
	}
	if !strictDigest.MatchString(i.LaunchDigest) {
		return errors.New("launchDigest must be lowercase SHA-256 hex")
	}
	if i.ModelDigests == nil {
		return errors.New("modelDigests is required")
	}
	for model, digest := range i.ModelDigests {
		if strings.TrimSpace(model) == "" {
			return errors.New("modelDigests contains an empty model")
		}
		if !strictDigest.MatchString(digest) {
			return fmt.Errorf("modelDigests[%q] must be lowercase SHA-256 hex", model)
		}
	}
	return nil
}

type AdmissionState struct {
	Active         int            `json:"active"`
	ActiveByEngine map[string]int `json:"activeByEngine"`
	Queued         int            `json:"queued"`
	// LastActivityMs is the unix-ms time each engine last started or finished
	// a request on this node.
	LastActivityMs map[string]int64 `json:"lastActivityMs"`
}

type UnloadRequest struct {
	Engine string   `json:"engine"`
	Models []string `json:"models"`
}

type WakeRequest struct {
	Engine string `json:"engine"`
}

// MethodWorkloadCancel is the broker → proxy request behind workloads:cancel
// for a workload this node originated (FORK_DESIGN.md §4). Like the node/*
// methods above it is process-scoped. It carries WorkloadCancelParams and is
// answered with WorkloadCancelResult or a JSON-RPC error.
const MethodWorkloadCancel = "workload/cancel"

// WorkloadCancelParams names one in-flight workload this node's proxy
// originated. Workload ids are counted per engine facade, so two facades can
// hold the same id at once: Engine (the workload's engine field) picks one, and
// the proxy refuses an ambiguous id without it. RunID, when set, must match
// the proxy's run nonce, so a cancel aimed at an earlier proxy process finds
// nothing.
type WorkloadCancelParams struct {
	WorkloadID string `json:"workloadId"`
	Engine     string `json:"engine,omitempty"`
	RunID      string `json:"runId,omitempty"`
	// Regenerate, before the response commits, aborts the current attempt,
	// excludes its node for this request and continues dispatch elsewhere.
	// Without it, or once the response has committed, the workload ends as
	// cancelled. Two models' output is never spliced together.
	Regenerate bool `json:"regenerate,omitempty"`
}

// WorkloadCancelResult reports whether an in-flight workload matched.
type WorkloadCancelResult struct {
	Found bool `json:"found"`
}
