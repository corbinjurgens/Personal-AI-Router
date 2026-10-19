// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package enginesettings

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"regexp"
	"slices"
	"time"

	"nvpair-shared/engines"
)

const (
	SwitchMethod              = "engine:switch"
	SwitchStatusMethod        = "engine:switch-status"
	SwitchChangedNotification = "engine:switch-changed"
	SwitchPrepareMethod       = "engine:switch-prepare"
	SwitchStepMethod          = "engine:switch-step"
	SwitchReconcileMethod     = "engine:switch-reconcile"
	RemoteSwitchMethod        = "engine:remote-switch"
	RemoteSwitchStatusMethod  = "engine:remote-switch-status"
	SwitchApplyPath           = "/v1/engine-switch/apply"
	SwitchStatusPath          = "/v1/engine-switch/status"

	SwitchVersion              = 1
	MaxSwitchReceipts          = 256
	MaxSwitchJournalBytes      = 2 * 1024 * 1024
	MaxSwitchRequestIDBytes    = 128
	MaxSwitchErrorMessageBytes = 1024
	MaxSwitchRequestBytes      = 8 * 1024
	SwitchOperationBudget      = 15 * time.Minute
	SwitchDrainBudget          = 60 * time.Second
	SwitchStatusWaitBudget     = 25 * time.Second
)

const (
	SwitchAccepted          = "accepted"
	SwitchValidating        = "validating"
	SwitchDraining          = "draining"
	SwitchSourceStopping    = "source-stopping"
	SwitchSourceOff         = "source-off"
	SwitchTargetStarting    = "target-starting"
	SwitchTargetReady       = "target-ready"
	SwitchCompleted         = "completed"
	SwitchFailedValidation  = "failed-validation"
	SwitchFailedDrain       = "failed-drain"
	SwitchFailedSourceStop  = "failed-source-stop"
	SwitchFailedTargetStart = "failed-target-start"
	SwitchFailedPublication = "failed-publication"
	SwitchFailedInterrupted = "failed-interrupted"
	SwitchRejected          = "rejected"

	SwitchIntentOn                  = "on"
	SwitchIntentOff                 = "off"
	SwitchIntentUnknown             = "unknown"
	SwitchOriginLocal               = "local"
	SwitchOriginRemote              = "remote"
	SwitchStepSourceStop            = "source-stop"
	SwitchStepTargetStart           = "target-start"
	SwitchStepRelease               = "release"
	SwitchCheckpointReserved        = "reserved"
	SwitchCheckpointStopPending     = "stop-pending"
	SwitchCheckpointSourceOff       = "source-off"
	SwitchCheckpointStartPending    = "start-pending"
	SwitchCheckpointTargetCommitted = "target-committed"
	SwitchCheckpointReleased        = "released"
)

const (
	SwitchInvalidParamsCode          = -32602
	SwitchConfigurationConflictCode  = -32010
	SwitchRemoteVetoCode             = -32011
	SwitchOperationErrorCode         = -32000
	SwitchErrorInvalidRequest        = "invalid_request"
	SwitchErrorReusedID              = "reused_id"
	SwitchErrorConfigurationConflict = "configuration_conflict"
	SwitchErrorRemoteVeto            = "remote_veto"
	SwitchErrorUnknownReceipt        = "unknown_receipt"
	SwitchErrorBusy                  = "busy"
	SwitchErrorFull                  = "full"
	SwitchErrorStorage               = "storage"
	SwitchErrorUnprepared            = "unprepared"
	SwitchErrorUnroutable            = "unroutable"
	SwitchErrorTargetRunning         = "target_running"
	SwitchErrorValidation            = "validation"
	SwitchErrorDrain                 = "drain"
	SwitchErrorSourceStop            = "source_stop"
	SwitchErrorTargetStart           = "target_start"
	SwitchErrorPublication           = "publication"
	SwitchErrorInterrupted           = "interrupted"
)

type SwitchRequest struct {
	NodeID           string `json:"nodeId,omitempty"`
	RequestID        string `json:"requestId"`
	SourceEngine     string `json:"sourceEngine"`
	TargetEngine     string `json:"targetEngine"`
	ExpectedRevision string `json:"expectedRevision"`
}

type SwitchStatusRequest struct {
	NodeID        string  `json:"nodeId,omitempty"`
	RequestID     string  `json:"requestId,omitempty"`
	AfterSequence *uint64 `json:"afterSequence,omitempty"`
}

type SwitchObserved struct {
	Source SwitchEngineState `json:"source"`
	Target SwitchEngineState `json:"target"`
}

type SwitchStatus struct {
	ConfigurationRevision string          `json:"configurationRevision"`
	ActiveRequestID       string          `json:"activeRequestId,omitempty"`
	Receipt               *SwitchReceipt  `json:"receipt,omitempty"`
	Observed              *SwitchObserved `json:"observed,omitempty"`
}

type SwitchResult struct {
	Receipt SwitchReceipt `json:"receipt"`
}

type SwitchEngineState struct {
	Intent              string     `json:"intent"`
	Running             *bool      `json:"running"`
	Healthy             *bool      `json:"healthy"`
	ObservedAt          *time.Time `json:"observedAt"`
	TerminationVerified bool       `json:"terminationVerified"`
}

type SwitchError struct {
	Code    int    `json:"code"`
	Kind    string `json:"kind"`
	Message string `json:"message"`
}

type SwitchReceipt struct {
	Version          int               `json:"version"`
	NodeID           string            `json:"nodeId"`
	RequestID        string            `json:"requestId"`
	Caller           string            `json:"caller"`
	PayloadHash      string            `json:"payloadHash"`
	Request          SwitchRequest     `json:"request"`
	ExpectedRevision string            `json:"expectedRevision"`
	AcceptedRevision string            `json:"acceptedRevision"`
	Sequence         uint64            `json:"sequence"`
	Phase            string            `json:"phase"`
	Terminal         bool              `json:"terminal"`
	AcceptedAt       time.Time         `json:"acceptedAt"`
	UpdatedAt        time.Time         `json:"updatedAt"`
	Deadline         time.Time         `json:"deadline"`
	FinishedAt       *time.Time        `json:"finishedAt"`
	Source           SwitchEngineState `json:"source"`
	Target           SwitchEngineState `json:"target"`
	Error            *SwitchError      `json:"error"`
	RecoveryRequired bool              `json:"recoveryRequired"`
}

// SwitchJournal never evicts receipts. A full journal refuses fresh requests;
// callers must also reserve space for terminal/recovery writes at acceptance.
type SwitchJournal struct {
	Version         int                      `json:"version"`
	Generation      uint64                   `json:"generation"`
	ActiveRequestID string                   `json:"activeRequestId"`
	Receipts        map[string]SwitchReceipt `json:"receipts"`
}

type SwitchChanged struct {
	NodeID    string `json:"nodeId"`
	RequestID string `json:"requestId"`
	Sequence  uint64 `json:"sequence"`
}

// These envelopes are broker-local executor calls, never ec lifecycle routes.
// Caller and origin are supplied by an authenticated server boundary.
type SwitchPrepareRequest struct {
	OperationID        string    `json:"operationId"`
	Caller             string    `json:"caller"`
	Origin             string    `json:"origin"`
	SourceEngine       string    `json:"sourceEngine"`
	TargetEngine       string    `json:"targetEngine"`
	PayloadHash        string    `json:"payloadHash"`
	ConfigurationFence string    `json:"configurationFence"`
	Deadline           time.Time `json:"deadline"`
}

type SwitchStepRequest struct {
	SwitchPrepareRequest
	Step string `json:"step"`
}

type SwitchReconcileRequest struct {
	SwitchPrepareRequest
}

type SwitchCheckpoint struct {
	SwitchPrepareRequest
	Phase            string            `json:"phase"`
	PreSource        SwitchEngineState `json:"preSource"`
	PreTarget        SwitchEngineState `json:"preTarget"`
	SourceProvenance string            `json:"sourceProvenance"`
	TargetProvenance string            `json:"targetProvenance"`
	Source           SwitchEngineState `json:"source"`
	Target           SwitchEngineState `json:"target"`
	TargetCommitted  bool              `json:"targetCommitted"`
	Error            *SwitchError      `json:"error"`
}

type SwitchPrepareResult struct {
	Checkpoint *SwitchCheckpoint `json:"checkpoint"`
	Observed   SwitchObserved    `json:"observed"`
	Error      *SwitchError      `json:"error"`
}

type SwitchStepResult = SwitchPrepareResult
type SwitchReconcileResult = SwitchPrepareResult

var switchRequestID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
var switchRevision = regexp.MustCompile(`^e3v1:[0-9a-f]{64}$`)
var switchPayloadHash = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func (r SwitchRequest) Validate() error {
	// Matches nodepolicy's ASCII identity syntax (its routing request counter
	// is a different identity domain and is deliberately not reused here).
	if !switchRequestID.MatchString(r.RequestID) {
		return errors.New("requestId must be a nonempty ASCII identity of at most 128 bytes")
	}
	if !engines.ValidID(r.SourceEngine) || !engines.ValidID(r.TargetEngine) {
		return errors.New("sourceEngine and targetEngine must be valid engine IDs")
	}
	if r.SourceEngine == r.TargetEngine {
		return errors.New("sourceEngine and targetEngine must differ")
	}
	if !switchRevision.MatchString(r.ExpectedRevision) {
		return errors.New("expectedRevision must be an e3v1 lowercase SHA-256 token")
	}
	return switchSize(r, MaxSwitchRequestBytes)
}

func (r SwitchStatusRequest) Validate() error {
	if r.RequestID != "" && !switchRequestID.MatchString(r.RequestID) {
		return errors.New("requestId must be a nonempty ASCII identity of at most 128 bytes")
	}
	if r.AfterSequence != nil && r.RequestID == "" {
		return errors.New("afterSequence requires requestId")
	}
	return switchSize(r, MaxSwitchRequestBytes)
}

func IsTerminal(phase string) bool {
	switch phase {
	case SwitchCompleted, SwitchFailedValidation, SwitchFailedDrain, SwitchFailedSourceStop,
		SwitchFailedTargetStart, SwitchFailedPublication, SwitchFailedInterrupted, SwitchRejected:
		return true
	}
	return false
}

func validSwitchPhase(phase string) bool {
	return IsTerminal(phase) || slices.Contains([]string{SwitchAccepted, SwitchValidating,
		SwitchDraining, SwitchSourceStopping, SwitchSourceOff, SwitchTargetStarting, SwitchTargetReady}, phase)
}

// ValidTransition describes phase changes, not idempotent checkpoint retries.
// Every nonterminal phase can be interrupted; terminal receipts are immutable.
func ValidTransition(from, to string) bool {
	if !validSwitchPhase(from) || IsTerminal(from) {
		return false
	}
	if to == SwitchFailedInterrupted {
		return true
	}
	switch from {
	case SwitchAccepted:
		return to == SwitchValidating || to == SwitchRejected
	case SwitchValidating:
		return to == SwitchDraining || to == SwitchFailedValidation || to == SwitchRejected
	case SwitchDraining:
		return to == SwitchSourceStopping || to == SwitchFailedDrain
	case SwitchSourceStopping:
		return to == SwitchSourceOff || to == SwitchFailedSourceStop
	case SwitchSourceOff:
		return to == SwitchTargetStarting
	case SwitchTargetStarting:
		return to == SwitchTargetReady || to == SwitchFailedTargetStart
	case SwitchTargetReady:
		return to == SwitchCompleted || to == SwitchFailedPublication
	}
	return false
}

func (s SwitchEngineState) Validate() error {
	if s.Intent != SwitchIntentOn && s.Intent != SwitchIntentOff && s.Intent != SwitchIntentUnknown {
		return errors.New("intent must be on, off, or unknown")
	}
	if s.ObservedAt != nil && s.ObservedAt.IsZero() {
		return errors.New("observedAt must not be zero")
	}
	if s.TerminationVerified && (s.Running == nil || *s.Running || s.ObservedAt == nil) {
		return errors.New("terminationVerified requires timestamped stopped evidence")
	}
	return nil
}

func (e SwitchError) Validate() error {
	code := SwitchOperationErrorCode
	switch e.Kind {
	case SwitchErrorInvalidRequest, SwitchErrorReusedID:
		code = SwitchInvalidParamsCode
	case SwitchErrorConfigurationConflict:
		code = SwitchConfigurationConflictCode
	case SwitchErrorRemoteVeto:
		code = SwitchRemoteVetoCode
	case SwitchErrorUnknownReceipt, SwitchErrorBusy, SwitchErrorFull, SwitchErrorStorage,
		SwitchErrorUnprepared, SwitchErrorUnroutable, SwitchErrorTargetRunning, SwitchErrorValidation,
		SwitchErrorDrain, SwitchErrorSourceStop, SwitchErrorTargetStart, SwitchErrorPublication, SwitchErrorInterrupted:
	default:
		return errors.New("unknown switch error kind")
	}
	if e.Code != code {
		return errors.New("switch error code does not match kind")
	}
	if e.Message == "" || len(e.Message) > MaxSwitchErrorMessageBytes {
		return errors.New("switch error message must contain 1 to 1024 bytes")
	}
	return nil
}

func (r SwitchReceipt) Validate() error {
	if r.Version != SwitchVersion {
		return errors.New("unsupported switch receipt version")
	}
	if err := r.Request.Validate(); err != nil {
		return err
	}
	if r.NodeID == "" || r.Caller == "" || r.Request.NodeID != r.NodeID || r.RequestID != r.Request.RequestID {
		return errors.New("switch receipt identity does not match normalized request")
	}
	if !switchPayloadHash.MatchString(r.PayloadHash) || r.PayloadHash != SwitchPayloadHash(r.Caller, r.NodeID, r.Request) {
		return errors.New("switch receipt payloadHash does not match request")
	}
	if r.ExpectedRevision != r.Request.ExpectedRevision || !switchRevision.MatchString(r.AcceptedRevision) {
		return errors.New("invalid switch receipt revision")
	}
	if r.Sequence == 0 || !validSwitchPhase(r.Phase) || r.Terminal != IsTerminal(r.Phase) {
		return errors.New("invalid switch receipt sequence, phase, or terminal flag")
	}
	if r.AcceptedAt.IsZero() || r.UpdatedAt.Before(r.AcceptedAt) || !r.Deadline.After(r.AcceptedAt) || r.Deadline.Sub(r.AcceptedAt) > SwitchOperationBudget {
		return errors.New("invalid switch receipt timestamps or operation deadline")
	}
	if r.Terminal != (r.FinishedAt != nil) || (r.FinishedAt != nil && (r.FinishedAt.Before(r.AcceptedAt) || r.FinishedAt.After(r.UpdatedAt))) {
		return errors.New("finishedAt must match terminal receipt timestamps")
	}
	if r.Terminal && r.Phase != SwitchCompleted && r.Error == nil {
		return errors.New("failed or rejected receipt requires error")
	}
	if r.Phase == SwitchCompleted && (r.Error != nil || r.RecoveryRequired) {
		return errors.New("completed receipt cannot require recovery or contain error")
	}
	if r.Error != nil {
		if err := r.Error.Validate(); err != nil {
			return err
		}
	}
	if err := r.Source.Validate(); err != nil {
		return fmt.Errorf("source: %w", err)
	}
	if err := r.Target.Validate(); err != nil {
		return fmt.Errorf("target: %w", err)
	}
	return switchSize(r, MaxSwitchJournalBytes)
}

func (j SwitchJournal) Validate() error {
	if j.Version != SwitchVersion {
		return errors.New("unsupported switch journal version")
	}
	if j.Receipts == nil || len(j.Receipts) > MaxSwitchReceipts {
		return errors.New("switch journal requires a bounded receipts map")
	}
	active := 0
	for id, receipt := range j.Receipts {
		if id != receipt.RequestID {
			return errors.New("switch journal key does not match requestId")
		}
		if err := receipt.Validate(); err != nil {
			return fmt.Errorf("receipt %q: %w", id, err)
		}
		if !receipt.Terminal {
			active++
			if id != j.ActiveRequestID {
				return errors.New("nonterminal receipt must be active")
			}
		}
	}
	if active > 1 {
		return errors.New("switch journal has multiple active requests")
	}
	if j.ActiveRequestID != "" {
		r, ok := j.Receipts[j.ActiveRequestID]
		// A terminal receipt can retain the active fence until cleanup completes.
		if !ok || (r.Terminal && !r.RecoveryRequired) {
			return errors.New("invalid activeRequestId")
		}
	}
	if len(j.Receipts) > 0 && j.Generation == 0 {
		return errors.New("nonempty switch journal requires generation")
	}
	return switchSize(j, MaxSwitchJournalBytes)
}

func switchSize(value any, limit int) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(data) > limit {
		return fmt.Errorf("switch JSON exceeds %d bytes", limit)
	}
	return nil
}

func decodeSwitchJSON[T any](data []byte, limit int) (T, error) {
	var result T
	if len(data) > limit {
		return result, fmt.Errorf("switch JSON exceeds %d bytes", limit)
	}
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return result, errors.New("switch JSON must be an object")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&result); err != nil {
		return result, err
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); err != io.EOF {
		return result, errors.New("switch JSON contains trailing data")
	}
	return result, nil
}

func DecodeSwitchRequest(data []byte) (SwitchRequest, error) {
	r, err := decodeSwitchJSON[SwitchRequest](data, MaxSwitchRequestBytes)
	if err == nil {
		err = r.Validate()
	}
	return r, err
}

func DecodeSwitchStatusRequest(data []byte) (SwitchStatusRequest, error) {
	r, err := decodeSwitchJSON[SwitchStatusRequest](data, MaxSwitchRequestBytes)
	if err == nil {
		err = r.Validate()
	}
	return r, err
}

func DecodeSwitchJournal(data []byte) (SwitchJournal, error) {
	j, err := decodeSwitchJSON[SwitchJournal](data, MaxSwitchJournalBytes)
	if err == nil {
		err = j.Validate()
	}
	return j, err
}

// SwitchPayloadHash hashes a JSON array in this fixed order: version, caller,
// canonical target node identity, request ID, source, target, expected revision.
// request.NodeID is a transport alias; only canonicalNodeID binds the target.
// There are no timestamps or runtime observations in this encoding.
func SwitchPayloadHash(caller, canonicalNodeID string, request SwitchRequest) string {
	return "sha256:" + switchDigest([]any{SwitchVersion, caller, canonicalNodeID,
		request.RequestID, request.SourceEngine, request.TargetEngine, request.ExpectedRevision})
}

type SwitchConfigurationEngine struct {
	Engine           string `json:"engine"`
	SettingsRevision uint64 `json:"settingsRevision"`
	ConfigDigest     string `json:"configDigest"`
	Intent           string `json:"intent"`
	Provenance       string `json:"provenance"`
}

// SwitchConfigurationSnapshot contains configuration only. Its collector owns
// authentication and consistency; running/health, Wake/Sleep, notification
// sequences and timestamps deliberately have no representation here.
type SwitchConfigurationSnapshot struct {
	NodeID                     string                      `json:"nodeId"`
	JournalGeneration          uint64                      `json:"journalGeneration"`
	ManagerIncarnation         string                      `json:"managerIncarnation"`
	Engines                    []SwitchConfigurationEngine `json:"engines"`
	RegistryRuntimeFingerprint string                      `json:"registryRuntimeFingerprint"`
}

// SwitchConfigurationToken hashes [version,node,generation,incarnation,engines,
// fingerprint]. Engine entries use their declared JSON field order, sorted by
// all fields to make even duplicate-ID input deterministic without mutating it.
func SwitchConfigurationToken(snapshot SwitchConfigurationSnapshot) string {
	entries := append([]SwitchConfigurationEngine{}, snapshot.Engines...)
	slices.SortFunc(entries, func(a, b SwitchConfigurationEngine) int {
		return cmp.Or(cmp.Compare(a.Engine, b.Engine), cmp.Compare(a.SettingsRevision, b.SettingsRevision),
			cmp.Compare(a.ConfigDigest, b.ConfigDigest), cmp.Compare(a.Intent, b.Intent), cmp.Compare(a.Provenance, b.Provenance))
	})
	return "e3v1:" + switchDigest([]any{SwitchVersion, snapshot.NodeID, snapshot.JournalGeneration,
		snapshot.ManagerIncarnation, entries, snapshot.RegistryRuntimeFingerprint})
}

func switchDigest(value any) string {
	// Only fixed structs, strings, uint64s and arrays enter this private helper;
	// their JSON marshaling cannot fail.
	data, _ := json.Marshal(value)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
