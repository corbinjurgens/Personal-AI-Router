// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"nvpair-shared/engines"
)

// Resident RPCs are internal broker calls, never client or peer commands.
const (
	MethodPrepareResident        = "node/prepare-resident"
	MethodReconcileResident      = "engine:reconcile-resident"
	MethodFinishResident         = "node/finish-resident"
	MethodDispatchResident       = "node/dispatch-resident"
	MethodObserveResident        = "node/observe-resident"
	MaxResidentTimeoutMs         = 60_000
	MaxModelConfigurationEntries = 256
	MaxResidentModelBytes        = 1024
)

// MaxResidentKeepAliveSeconds caps each load's idle retention at one day so a
// request cannot pin engine memory indefinitely. Policy has no equivalent bound.
const MaxResidentKeepAliveSeconds = 86_400

// ResidentLoadOptions comes from the resident profile at PolicyRevision.
// ContextLength uses the same positive int range as Profile.ContextTokens in
// Policy.Validate; that policy defines no smaller maximum. Integer decoding
// rejects fractional, non-finite and overflowing values for both options.
type ResidentLoadOptions struct {
	ContextLength    int `json:"contextLength"`
	KeepAliveSeconds int `json:"keepAliveSeconds"`
}

func (o ResidentLoadOptions) Validate() error {
	if o.ContextLength <= 0 {
		return errors.New("contextLength must be positive")
	}
	if o.KeepAliveSeconds <= 0 || o.KeepAliveSeconds > MaxResidentKeepAliveSeconds {
		return errors.New("keepAliveSeconds must be in 1..86400")
	}
	return nil
}

func (o *ResidentLoadOptions) UnmarshalJSON(data []byte) error {
	type plain ResidentLoadOptions
	var next plain
	if err := decodeResident(data, &next, []string{"contextLength", "keepAliveSeconds"}); err != nil {
		return err
	}
	if err := ResidentLoadOptions(next).Validate(); err != nil {
		return err
	}
	*o = ResidentLoadOptions(next)
	return nil
}

// DispatchResidentParams requests the final grant immediately before loading.
// The proxy verifies every binding and the deadline under its admission mutex
// and marks the reservation dispatched. Invalidation then cannot free it until
// finish or observation. ManagerIncarnation is required to bind that grant.
type DispatchResidentParams struct {
	ReservationID      string              `json:"reservationId"`
	ProxyRunID         string              `json:"proxyRunId"`
	ManagerIncarnation string              `json:"managerIncarnation"`
	OperationID        string              `json:"operationId"`
	PolicyRevision     string              `json:"policyRevision"`
	Engine             string              `json:"engine"`
	Model              string              `json:"model"`
	Identity           StrictModelIdentity `json:"identity"`
}

func (p DispatchResidentParams) Validate() error {
	if !strictID.MatchString(p.ReservationID) || !strictID.MatchString(p.ProxyRunID) ||
		!strictID.MatchString(p.ManagerIncarnation) || !strictID.MatchString(p.OperationID) {
		return errors.New("reservationId, proxyRunId, managerIncarnation and operationId must be ASCII identities of 1-128 bytes")
	}
	if err := validateRevision(p.PolicyRevision); err != nil {
		return err
	}
	if err := validateResidentRef(p.Engine, p.Model); err != nil {
		return err
	}
	return p.Identity.Validate()
}

// DispatchResidentReason reuses resident refusal codes, with final-grant codes
// for a reservation that was invalidated or is not known to this proxy run.
type DispatchResidentReason string

const (
	DispatchReasonReservationRevoked DispatchResidentReason = "reservation-revoked"
	DispatchReasonReservationUnknown DispatchResidentReason = "reservation-unknown"
)

func (r DispatchResidentReason) Validate() error {
	if r == DispatchReasonReservationRevoked || r == DispatchReasonReservationUnknown {
		return nil
	}
	return ModelConfigurationReason(r).Validate()
}

// DispatchResidentResult is an explicit final grant or a typed refusal.
// A refusal means nothing may be dispatched to the engine.
type DispatchResidentResult struct {
	Dispatched bool                   `json:"dispatched"`
	Reason     DispatchResidentReason `json:"reason,omitempty"`
}

func (r DispatchResidentResult) Validate() error {
	if r.Reason != "" {
		if err := r.Reason.Validate(); err != nil {
			return err
		}
	}
	if r.Dispatched {
		if r.Reason != "" {
			return errors.New("dispatch success must not carry a reason")
		}
		return nil
	}
	if r.Reason == "" {
		return errors.New("dispatch refusal requires a reason")
	}
	return nil
}

func (p *DispatchResidentParams) UnmarshalJSON(data []byte) error {
	type plain DispatchResidentParams
	var next plain
	if err := decodeResident(data, &next, []string{"reservationId", "proxyRunId", "managerIncarnation", "operationId", "policyRevision", "engine", "model", "identity"}); err != nil {
		return err
	}
	if err := DispatchResidentParams(next).Validate(); err != nil {
		return err
	}
	*p = DispatchResidentParams(next)
	return nil
}

func (r *DispatchResidentResult) UnmarshalJSON(data []byte) error {
	type plain DispatchResidentResult
	var next plain
	if err := decodeResident(data, &next, []string{"dispatched"}, "reason"); err != nil {
		return err
	}
	if err := DispatchResidentResult(next).Validate(); err != nil {
		return err
	}
	*r = DispatchResidentResult(next)
	return nil
}

// PrepareResidentParams asks the current proxy to evaluate and reserve under
// its admission mutex. TimeoutMs is a remaining budget, avoiding clock skew;
// the receiver fixes a deadline once, and retries must never extend it.
// Model is already normalized by the authoritative engine descriptor. Syntax
// validation here cannot establish normalization, qualification or permission.
// OperationID is resident:<positive uint64>, allocated sequentially by the
// broker for the current proxy run. Retry uncertain preparations with the exact
// same ID/payload. Manager replacement does not reset this allocator. The proxy
// exits on broker EOF; attaching a future broker to a surviving proxy would
// require explicit cursor readback rather than guessing a new sequence.
type PrepareResidentParams struct {
	PolicyRevision     string              `json:"policyRevision"`
	ProxyRunID         string              `json:"proxyRunId"`
	ManagerIncarnation string              `json:"managerIncarnation,omitempty"`
	Engine             string              `json:"engine"`
	Model              string              `json:"model"`
	Identity           StrictModelIdentity `json:"identity"`
	OperationID        string              `json:"operationId"`
	TimeoutMs          int                 `json:"timeoutMs"`
}

func (p PrepareResidentParams) Validate() error {
	if err := p.validateFields(); err != nil {
		return err
	}
	_, err := ParseResidentOperationID(p.OperationID)
	return err
}

// ReconcileResidentParams shares these field checks, but keeps operation IDs
// opaque: only the proxy grants execution and enforces its ordered ID stream.
func (p PrepareResidentParams) validateFields() error {
	if err := validateRevision(p.PolicyRevision); err != nil {
		return err
	}
	if !strictID.MatchString(p.ProxyRunID) || !strictID.MatchString(p.OperationID) {
		return errors.New("proxyRunId and operationId must be ASCII identities of 1-128 bytes")
	}
	if p.ManagerIncarnation != "" && !strictID.MatchString(p.ManagerIncarnation) {
		return errors.New("invalid managerIncarnation")
	}
	if p.TimeoutMs <= 0 || p.TimeoutMs > MaxResidentTimeoutMs {
		return errors.New("timeoutMs must be in 1..60000")
	}
	if err := validateResidentRef(p.Engine, p.Model); err != nil {
		return err
	}
	return p.Identity.Validate()
}

// ResidentOperationID formats the one canonical preparation sequence syntax.
func ResidentOperationID(sequence uint64) (string, error) {
	if sequence == 0 {
		return "", errors.New("resident operation sequence must be positive")
	}
	return "resident:" + strconv.FormatUint(sequence, 10), nil
}

func ParseResidentOperationID(id string) (uint64, error) {
	decimal, ok := strings.CutPrefix(id, "resident:")
	if !ok {
		return 0, errors.New("operationId must be resident:<positive uint64>")
	}
	sequence, err := strconv.ParseUint(decimal, 10, 64)
	if err != nil || sequence == 0 || strconv.FormatUint(sequence, 10) != decimal {
		return 0, errors.New("operationId must contain a canonical positive uint64")
	}
	return sequence, nil
}

// ModelConfigurationReason contains codes only; raw engine errors and identity
// digests must not be copied into derived status or refusal explanations.
type ModelConfigurationReason string

const (
	ModelReasonEngineOff          ModelConfigurationReason = ModelConfigurationReason(RejectOff)
	ModelReasonNoFit              ModelConfigurationReason = ModelConfigurationReason(RejectNoFit)
	ModelReasonPausedLocal        ModelConfigurationReason = "paused-local"
	ModelReasonEngineOffLocal     ModelConfigurationReason = "engine-off-local"
	ModelReasonNotServed          ModelConfigurationReason = "not-served"
	ModelReasonNotInstalled       ModelConfigurationReason = "not-installed"
	ModelReasonUnqualified        ModelConfigurationReason = "unqualified"
	ModelReasonUnsupportedLoad    ModelConfigurationReason = "unsupported-load"
	ModelReasonSwitchBusy         ModelConfigurationReason = "switch-busy"
	ModelReasonOperationTimeout   ModelConfigurationReason = "operation-timeout"
	ModelReasonObservationUnknown ModelConfigurationReason = "observation-unknown"
	ModelReasonProjectionUnknown  ModelConfigurationReason = "projection-unknown"
)

func (r ModelConfigurationReason) Validate() error {
	switch r {
	case ModelReasonEngineOff, ModelReasonNoFit, ModelReasonPausedLocal, ModelReasonEngineOffLocal,
		ModelReasonNotServed, ModelReasonNotInstalled, ModelReasonUnqualified, ModelReasonUnsupportedLoad,
		ModelReasonSwitchBusy, ModelReasonOperationTimeout, ModelReasonObservationUnknown, ModelReasonProjectionUnknown:
		return nil
	}
	return errors.New("unknown model configuration reason")
}

type ResidentRefusal struct {
	Engine string                   `json:"engine"`
	Model  string                   `json:"model"`
	Reason ModelConfigurationReason `json:"reason"`
}

func (r ResidentRefusal) Validate() error {
	if err := validateResidentRef(r.Engine, r.Model); err != nil {
		return err
	}
	return r.Reason.Validate()
}

// PrepareResidentResult is a disjoint grant/refusal: refusal never reserves.
// EvidenceGeneration identifies the evaluation, including a refused one.
type PrepareResidentResult struct {
	ReservationID      string            `json:"reservationId,omitempty"`
	EvidenceGeneration uint64            `json:"evidenceGeneration"`
	Refusals           []ResidentRefusal `json:"refusals,omitempty"`
}

func (r PrepareResidentResult) Validate() error {
	if len(r.Refusals) > MaxModelConfigurationEntries {
		return errors.New("too many refusals")
	}
	if len(r.Refusals) == 0 {
		if !strictID.MatchString(r.ReservationID) {
			return errors.New("grant requires a reservationId of 1-128 ASCII bytes")
		}
	} else if r.ReservationID != "" {
		return errors.New("refusal must not reserve")
	}
	seen := map[Role]bool{}
	for _, refusal := range r.Refusals {
		if err := refusal.Validate(); err != nil {
			return err
		}
		ref := Role{Engine: refusal.Engine, Model: refusal.Model}
		if seen[ref] {
			return errors.New("duplicate refusal reference")
		}
		seen[ref] = true
	}
	return nil
}

// ResidentReservationEvidence is process-bound permission from the broker's
// proxy session, not a credential. ManagerIncarnation is included when known.
// The manager must revalidate every binding under E3 exclusion before dispatch;
// an opaque ID alone is never authorization after revocation or replacement.
type ResidentReservationEvidence struct {
	ReservationID      string `json:"reservationId"`
	ProxyRunID         string `json:"proxyRunId"`
	ManagerIncarnation string `json:"managerIncarnation,omitempty"`
}

func (e ResidentReservationEvidence) Validate() error {
	if !strictID.MatchString(e.ReservationID) || !strictID.MatchString(e.ProxyRunID) {
		return errors.New("reservationId and proxyRunId must be ASCII identities of 1-128 bytes")
	}
	if e.ManagerIncarnation != "" && !strictID.MatchString(e.ManagerIncarnation) {
		return errors.New("invalid managerIncarnation")
	}
	return nil
}

type ReconcileResidentParams struct {
	OperationID    string                      `json:"operationId"`
	PolicyRevision string                      `json:"policyRevision"`
	Engine         string                      `json:"engine"`
	Model          string                      `json:"model"`
	Identity       StrictModelIdentity         `json:"identity"`
	TimeoutMs      int                         `json:"timeoutMs"`
	Reservation    ResidentReservationEvidence `json:"reservation"`
	Load           ResidentLoadOptions         `json:"load"`
}

func (p ReconcileResidentParams) Validate() error {
	if err := p.Load.Validate(); err != nil {
		return err
	}
	if err := p.Reservation.Validate(); err != nil {
		return err
	}
	return (PrepareResidentParams{PolicyRevision: p.PolicyRevision, ProxyRunID: p.Reservation.ProxyRunID,
		ManagerIncarnation: p.Reservation.ManagerIncarnation, Engine: p.Engine, Model: p.Model,
		Identity: p.Identity, OperationID: p.OperationID, TimeoutMs: p.TimeoutMs}).validateFields()
}

type ResidentOutcome string

const (
	ResidentLoaded        ResidentOutcome = "loaded"
	ResidentNotDispatched ResidentOutcome = "not-dispatched"
	ResidentFailed        ResidentOutcome = "failed"
	ResidentCancelled     ResidentOutcome = "cancelled"
	ResidentTimeout       ResidentOutcome = "timeout"
	ResidentUnknown       ResidentOutcome = "unknown"
)

func (o ResidentOutcome) Validate() error {
	switch o {
	case ResidentLoaded, ResidentNotDispatched, ResidentFailed, ResidentCancelled, ResidentTimeout, ResidentUnknown:
		return nil
	}
	return errors.New("unknown resident outcome")
}

// ResidentLoadedObservation binds inventory to the manager source, engine
// identity and normalized reference. Generation is scoped to that source.
// Loaded=false is authoritative absence, not missing/unknown observation.
type ResidentLoadedObservation struct {
	ManagerIncarnation string              `json:"managerIncarnation"`
	Generation         uint64              `json:"generation"`
	Engine             string              `json:"engine"`
	Model              string              `json:"model"`
	Identity           StrictModelIdentity `json:"identity"`
	Loaded             bool                `json:"loaded"`
}

func (o ResidentLoadedObservation) Validate() error {
	if !strictID.MatchString(o.ManagerIncarnation) {
		return errors.New("invalid managerIncarnation")
	}
	if err := validateResidentRef(o.Engine, o.Model); err != nil {
		return err
	}
	return o.Identity.Validate()
}

type ReconcileResidentResult struct {
	Outcome     ResidentOutcome            `json:"outcome"`
	Observation *ResidentLoadedObservation `json:"observation,omitempty"`
}

func (r ReconcileResidentResult) Validate() error {
	if err := r.Outcome.Validate(); err != nil {
		return err
	}
	if r.Observation != nil {
		if err := r.Observation.Validate(); err != nil {
			return err
		}
	}
	if r.Outcome == ResidentLoaded && (r.Observation == nil || !r.Observation.Loaded) {
		return errors.New("loaded outcome requires a loaded observation")
	}
	if r.Outcome == ResidentNotDispatched && r.Observation != nil {
		return errors.New("not-dispatched must not claim a load observation")
	}
	return nil
}

// FinishResidentParams consumes an exact reservation once. The owner retains
// the terminal result for duplicate finishes while retained; changed payloads
// conflict. Retired operations are refused, never reopened. IDs
// must never be reused in a proxy run. A stale ID cannot release another hold.
// Timeout/cancellation/failure alone is not evidence of engine termination.
// Observation may be absent when no authoritative inventory is available.
type FinishResidentParams struct {
	ReservationID string                     `json:"reservationId"`
	ProxyRunID    string                     `json:"proxyRunId"`
	OperationID   string                     `json:"operationId"`
	Outcome       ResidentOutcome            `json:"outcome"`
	Observation   *ResidentLoadedObservation `json:"observation,omitempty"`
}

func (p FinishResidentParams) Validate() error {
	if !strictID.MatchString(p.ReservationID) || !strictID.MatchString(p.ProxyRunID) || !strictID.MatchString(p.OperationID) {
		return errors.New("reservationId, proxyRunId and operationId must be ASCII identities of 1-128 bytes")
	}
	return (ReconcileResidentResult{Outcome: p.Outcome, Observation: p.Observation}).Validate()
}

// FinishResidentResult is immutable while retained, including after recovery.
// RecoveryID exists only for still-counted dispatched uncertainty. It binds a
// later fresh observation; it is not a credential or proof of engine idleness.
type FinishResidentResult struct {
	ReservationID string          `json:"reservationId"`
	Outcome       ResidentOutcome `json:"outcome"`
	RecoveryID    string          `json:"recoveryId,omitempty"`
}

func (r FinishResidentResult) Validate() error {
	if !strictID.MatchString(r.ReservationID) {
		return errors.New("invalid reservationId")
	}
	if r.RecoveryID != "" && (!strictID.MatchString(r.RecoveryID) || r.Outcome == ResidentLoaded) {
		return errors.New("invalid recoveryId for finish result")
	}
	return r.Outcome.Validate()
}

// ObserveResidentParams is an internal broker-owned recovery stream. Sequence
// is scoped to RecoveryID; it is neither a manager residentGeneration nor a
// proxy EvidenceGeneration. OperationComplete is a trusted assertion, not
// cryptographic proof. SR5 may assert it only after the same-source executor
// completion barrier and a subsequent fresh engine:models read, obtained after
// receiving RecoveryID. A timeout or legacy cached loaded map is insufficient.
// Retry an uncertain response with identical params before issuing another
// read: the first accepted observation's settlement receipt is immutable.
type ObserveResidentParams struct {
	ReservationID     string                      `json:"reservationId"`
	ProxyRunID        string                      `json:"proxyRunId"`
	OperationID       string                      `json:"operationId"`
	RecoveryID        string                      `json:"recoveryId"`
	Sequence          uint64                      `json:"sequence"`
	OperationComplete bool                        `json:"operationComplete"`
	Observation       ResidentCapacityObservation `json:"observation"`
}

type ResidentCapacityObservation struct {
	ManagerIncarnation string              `json:"managerIncarnation"`
	Engine             string              `json:"engine"`
	Model              string              `json:"model"`
	Identity           StrictModelIdentity `json:"identity"`
	Loaded             bool                `json:"loaded"`
}

func (o ResidentCapacityObservation) Validate() error {
	if !strictID.MatchString(o.ManagerIncarnation) {
		return errors.New("invalid managerIncarnation")
	}
	if err := validateResidentRef(o.Engine, o.Model); err != nil {
		return err
	}
	return o.Identity.Validate()
}

func (p ObserveResidentParams) Validate() error {
	if !strictID.MatchString(p.ReservationID) || !strictID.MatchString(p.ProxyRunID) || !strictID.MatchString(p.OperationID) || !strictID.MatchString(p.RecoveryID) {
		return errors.New("reservationId, proxyRunId, operationId and recoveryId must be ASCII identities of 1-128 bytes")
	}
	if p.Sequence == 0 || !p.OperationComplete {
		return errors.New("positive sequence and operationComplete true are required")
	}
	return p.Observation.Validate()
}

type ObserveResidentResult struct {
	ReservationID string `json:"reservationId"`
	Settled       bool   `json:"settled"`
}

func (r ObserveResidentResult) Validate() error {
	if !strictID.MatchString(r.ReservationID) || !r.Settled {
		return errors.New("valid reservationId and settled true are required")
	}
	return nil
}

type ModelConfigurationState string

const (
	ModelStateSatisfied ModelConfigurationState = "satisfied"
	ModelStatePending   ModelConfigurationState = "pending"
	ModelStateBlocked   ModelConfigurationState = "blocked"
	ModelStateUnknown   ModelConfigurationState = "unknown"
)

func (s ModelConfigurationState) Validate() error {
	switch s {
	case ModelStateSatisfied, ModelStatePending, ModelStateBlocked, ModelStateUnknown:
		return nil
	}
	return errors.New("unknown model configuration state")
}

type ModelConfigurationEntry struct {
	Engine          string                   `json:"engine"`
	Model           string                   `json:"model"`
	DesiredResident bool                     `json:"desiredResident"`
	ObservedLoaded  *bool                    `json:"observedLoaded,omitempty"`
	State           ModelConfigurationState  `json:"state"`
	Reason          ModelConfigurationReason `json:"reason,omitempty"`
}

func (e ModelConfigurationEntry) Validate() error {
	if err := validateResidentRef(e.Engine, e.Model); err != nil {
		return err
	}
	if err := e.State.Validate(); err != nil {
		return err
	}
	if e.Reason != "" {
		if err := e.Reason.Validate(); err != nil {
			return err
		}
	}
	if e.State == ModelStateSatisfied && (e.ObservedLoaded == nil || !*e.ObservedLoaded || !e.DesiredResident || e.Reason != "") {
		return errors.New("satisfied requires desired resident, loaded evidence and no reason")
	}
	return nil
}

// ModelConfiguration is derived from current desired policy, activation,
// source-bound inventory/identity and the current/last attempt; never persisted
// as a separate state machine. Missing ObservedLoaded means unknown.
type ModelConfiguration struct {
	PolicyRevision        string                    `json:"policyRevision"`
	ObservationGeneration uint64                    `json:"observationGeneration"`
	Models                []ModelConfigurationEntry `json:"models"`
}

func (c ModelConfiguration) Validate() error {
	if err := validateRevision(c.PolicyRevision); err != nil {
		return err
	}
	if c.Models == nil || len(c.Models) > MaxModelConfigurationEntries {
		return errors.New("models must be a list of at most 256 entries")
	}
	seen := map[Role]bool{}
	for _, e := range c.Models {
		if err := e.Validate(); err != nil {
			return err
		}
		ref := Role{Engine: e.Engine, Model: e.Model}
		if seen[ref] {
			return errors.New("duplicate model configuration reference")
		}
		seen[ref] = true
	}
	return nil
}

func (p *PrepareResidentParams) UnmarshalJSON(data []byte) error {
	type plain PrepareResidentParams
	var next plain
	if err := decodeResident(data, &next, []string{"policyRevision", "proxyRunId", "engine", "model", "identity", "operationId", "timeoutMs"}, "managerIncarnation"); err != nil {
		return err
	}
	if err := PrepareResidentParams(next).Validate(); err != nil {
		return err
	}
	*p = PrepareResidentParams(next)
	return nil
}

func (p *ResidentRefusal) UnmarshalJSON(data []byte) error {
	type plain ResidentRefusal
	var next plain
	if err := decodeResident(data, &next, []string{"engine", "model", "reason"}); err != nil {
		return err
	}
	if err := ResidentRefusal(next).Validate(); err != nil {
		return err
	}
	*p = ResidentRefusal(next)
	return nil
}

func (p *PrepareResidentResult) UnmarshalJSON(data []byte) error {
	type plain PrepareResidentResult
	var next plain
	if err := decodeResident(data, &next, []string{"evidenceGeneration"}, "reservationId", "refusals"); err != nil {
		return err
	}
	if err := PrepareResidentResult(next).Validate(); err != nil {
		return err
	}
	*p = PrepareResidentResult(next)
	return nil
}

func (p *ResidentReservationEvidence) UnmarshalJSON(data []byte) error {
	type plain ResidentReservationEvidence
	var next plain
	if err := decodeResident(data, &next, []string{"reservationId", "proxyRunId"}, "managerIncarnation"); err != nil {
		return err
	}
	if err := ResidentReservationEvidence(next).Validate(); err != nil {
		return err
	}
	*p = ResidentReservationEvidence(next)
	return nil
}

func (p *ReconcileResidentParams) UnmarshalJSON(data []byte) error {
	type plain ReconcileResidentParams
	var next plain
	if err := decodeResident(data, &next, []string{"operationId", "policyRevision", "engine", "model", "identity", "timeoutMs", "reservation", "load"}); err != nil {
		return err
	}
	if err := ReconcileResidentParams(next).Validate(); err != nil {
		return err
	}
	*p = ReconcileResidentParams(next)
	return nil
}

func (p *ResidentLoadedObservation) UnmarshalJSON(data []byte) error {
	type plain ResidentLoadedObservation
	var next plain
	if err := decodeResident(data, &next, []string{"managerIncarnation", "generation", "engine", "model", "identity", "loaded"}); err != nil {
		return err
	}
	if err := ResidentLoadedObservation(next).Validate(); err != nil {
		return err
	}
	*p = ResidentLoadedObservation(next)
	return nil
}

func (p *ReconcileResidentResult) UnmarshalJSON(data []byte) error {
	type plain ReconcileResidentResult
	var next plain
	if err := decodeResident(data, &next, []string{"outcome"}, "observation"); err != nil {
		return err
	}
	if err := ReconcileResidentResult(next).Validate(); err != nil {
		return err
	}
	*p = ReconcileResidentResult(next)
	return nil
}

func (p *FinishResidentParams) UnmarshalJSON(data []byte) error {
	type plain FinishResidentParams
	var next plain
	if err := decodeResident(data, &next, []string{"reservationId", "proxyRunId", "operationId", "outcome"}, "observation"); err != nil {
		return err
	}
	if err := FinishResidentParams(next).Validate(); err != nil {
		return err
	}
	*p = FinishResidentParams(next)
	return nil
}

func (p *FinishResidentResult) UnmarshalJSON(data []byte) error {
	type plain FinishResidentResult
	var next plain
	if err := decodeResident(data, &next, []string{"reservationId", "outcome"}, "recoveryId"); err != nil {
		return err
	}
	if next.RecoveryID == "" {
		fields, err := peerFields(data)
		if err != nil {
			return err
		}
		if fields["recoveryId"] != nil {
			return errors.New("recoveryId must be nonempty when present")
		}
	}
	if err := FinishResidentResult(next).Validate(); err != nil {
		return err
	}
	*p = FinishResidentResult(next)
	return nil
}

func (p *ResidentCapacityObservation) UnmarshalJSON(data []byte) error {
	type plain ResidentCapacityObservation
	var next plain
	if err := decodeResident(data, &next, []string{"managerIncarnation", "engine", "model", "identity", "loaded"}); err != nil {
		return err
	}
	if err := ResidentCapacityObservation(next).Validate(); err != nil {
		return err
	}
	*p = ResidentCapacityObservation(next)
	return nil
}

func (p *ObserveResidentParams) UnmarshalJSON(data []byte) error {
	type plain ObserveResidentParams
	var next plain
	if err := decodeResident(data, &next, []string{"reservationId", "proxyRunId", "operationId", "recoveryId", "sequence", "operationComplete", "observation"}); err != nil {
		return err
	}
	if err := ObserveResidentParams(next).Validate(); err != nil {
		return err
	}
	*p = ObserveResidentParams(next)
	return nil
}

func (p *ObserveResidentResult) UnmarshalJSON(data []byte) error {
	type plain ObserveResidentResult
	var next plain
	if err := decodeResident(data, &next, []string{"reservationId", "settled"}); err != nil {
		return err
	}
	if err := ObserveResidentResult(next).Validate(); err != nil {
		return err
	}
	*p = ObserveResidentResult(next)
	return nil
}

func (p *ModelConfigurationEntry) UnmarshalJSON(data []byte) error {
	type plain ModelConfigurationEntry
	var next plain
	if err := decodeResident(data, &next, []string{"engine", "model", "desiredResident", "state"}, "observedLoaded", "reason"); err != nil {
		return err
	}
	if err := ModelConfigurationEntry(next).Validate(); err != nil {
		return err
	}
	*p = ModelConfigurationEntry(next)
	return nil
}

func (p *ModelConfiguration) UnmarshalJSON(data []byte) error {
	type plain ModelConfiguration
	var next plain
	if err := decodeResident(data, &next, []string{"policyRevision", "observationGeneration", "models"}); err != nil {
		return err
	}
	if err := ModelConfiguration(next).Validate(); err != nil {
		return err
	}
	*p = ModelConfiguration(next)
	return nil
}

func validateResidentRef(engine, model string) error {
	if err := checkModelRef(engine, model, 0, nil); err != nil {
		return err
	}
	if len(model) > MaxResidentModelBytes || strings.TrimSpace(model) != model {
		return errors.New("model must be trimmed and at most 1024 bytes")
	}
	return nil
}

// decodeResident rejects duplicate/case-variant/unknown keys, null fields and
// missing required fields, preserving explicit false versus absent booleans.
// Alias destinations prevent recursion through their own UnmarshalJSON.
func decodeResident[T any](data []byte, dst *T, required []string, optional ...string) error {
	fields, err := peerFields(data)
	if err != nil {
		return err
	}
	for _, key := range required {
		if fields[key] == nil {
			return fmt.Errorf("%s is required", key)
		}
	}
	for key, value := range fields {
		if !slices.Contains(required, key) && !slices.Contains(optional, key) {
			return fmt.Errorf("unknown field %q", key)
		}
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("%s must not be null", key)
		}
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	return d.Decode(dst)
}

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
	// MethodSetModelIdentity carries SetModelIdentityParams: the identity
	// the proxy binds strict capacity and qualification to. The broker sends
	// it whenever an engine's identity changes, and synchronously when one is
	// withdrawn, so withdrawal is in force before the change it guards.
	MethodSetModelIdentity = "node/set-model-identity"
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
