// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

// Broker → proxy requests. Each is answered with {"ok":true} or a JSON-RPC
// error. The broker re-sends all of them whenever the proxy (re)starts.
const (
	// MethodSetPolicy carries the whole Policy. Availability inside it is
	// ignored by the proxy; MethodSetAvailability is authoritative.
	MethodSetPolicy = "node/set-policy"
	// MethodSetAvailability carries SetAvailabilityParams.
	MethodSetAvailability = "node/set-availability"
	// MethodSetEngineDrain carries SetEngineDrainParams.
	MethodSetEngineDrain = "node/set-engine-drain"
	// MethodSetResidency carries SetResidencyParams.
	MethodSetResidency = "node/set-residency"
	// MethodSetEngineIntent carries SetEngineIntentParams.
	MethodSetEngineIntent = "node/set-engine-intent"
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
