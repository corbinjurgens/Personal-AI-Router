// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"errors"
	"fmt"

	"nvpair-shared/engines"
)

// StrictAdmissionReadiness is the future proxy -> broker A1/A2 handoff, not a
// registered method or notification. Only the broker can match its fences to
// the acknowledged proxy incarnation and desired/active policy revision before
// publishing Capability in Advertisement. Nil withdraws previous readiness.
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
// error, except the typed acknowledgements and snapshots below. The broker
// re-sends policy state whenever the proxy (re)starts and replays pending switch
// invalidations from its journal before enabling facades.
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
	// MethodInvalidateEngine withdraws strict eligibility for an engine switch.
	MethodInvalidateEngine = "node/invalidate-engine"
	// MethodAdmissionSnapshot reads current admission counts, without notification caching.
	MethodAdmissionSnapshot = "node/admission-snapshot"
)

const InvalidationReasonEngineSwitch = "engine-switch"

// InvalidateEngineParams identifies an engine within a switch operation. The
// same operation may invalidate both source and target engines independently.
type InvalidateEngineParams struct {
	Engine      string `json:"engine"`
	OperationID string `json:"operationId"`
	Reason      string `json:"reason"`
}

func (p InvalidateEngineParams) Validate() error {
	if !engines.ValidID(p.Engine) {
		return errors.New("engine must be a valid engine ID")
	}
	// strictID is the shared switch request-id syntax: 1-128 ASCII bytes,
	// alphanumeric first, then alphanumeric or . _ : -.
	if !strictID.MatchString(p.OperationID) {
		return errors.New("operationId must be a nonempty ASCII identity of at most 128 bytes")
	}
	if p.Reason != InvalidationReasonEngineSwitch {
		return errors.New("reason must be engine-switch")
	}
	return nil
}

func (p *InvalidateEngineParams) UnmarshalJSON(data []byte) error {
	type plain InvalidateEngineParams
	var next plain
	if err := decodeStrictContract(data, &next, "engine", "operationId", "reason"); err != nil {
		return err
	}
	if err := InvalidateEngineParams(next).Validate(); err != nil {
		return err
	}
	*p = InvalidateEngineParams(next)
	return nil
}

type InvalidateEngineResult struct {
	OK    bool   `json:"ok"`
	RunID string `json:"runId"`
}

// AdmissionSnapshotParams is an explicit empty object. Request it after the
// drain acknowledgement, and validate RunID before using zero as drain proof.
type AdmissionSnapshotParams struct{}

func (p *AdmissionSnapshotParams) UnmarshalJSON(data []byte) error {
	type plain AdmissionSnapshotParams
	var next plain
	return decodeStrictContract(data, &next)
}

// AdmissionSnapshotResult includes held tickets and quarantined capacity in
// its active counts, just like admission/state. Maps are copied at request time.
type AdmissionSnapshotResult struct {
	RunID string `json:"runId"`
	AdmissionState
}

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
