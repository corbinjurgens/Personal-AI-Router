// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package buildapply

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"nvpair-shared/nodepolicy"
)

const (
	MaxRequestBytes              = 16 << 10
	MaxStatusBytes               = 128 << 10
	MaxPathBytes                 = 4096
	MaxDetailBytes               = 1024
	MaxPackageBytes        int64 = 8 << 30
	MaxFileBytes           int64 = 2 << 30
	MaxEntries                   = 100000
	PreviewLifetimeSeconds       = 60
	CodeMaintenanceBusy          = -32015
	RefusalMaintenanceBusy       = "maintenance_busy"
	RefusalUnavailable           = "unavailable"
)

// Observation distinguishes unknown from a known false, zero or empty value.
type Observation[T any] struct {
	Known bool `json:"known"`
	Value *T   `json:"value"`
}

func Known[T any](value T) Observation[T] { return Observation[T]{Known: true, Value: &value} }
func (o Observation[T]) MarshalJSON() ([]byte, error) {
	type wire Observation[T]
	if o.Known != (o.Value != nil) {
		return nil, errors.New("inconsistent observation")
	}
	if o.Known {
		value, err := json.Marshal(o.Value)
		if err != nil {
			return nil, err
		}
		if bytes.Equal(value, []byte("null")) {
			return nil, errors.New("known observation cannot be null")
		}
	}
	return json.Marshal(wire(o))
}
func (o *Observation[T]) UnmarshalJSON(data []byte) error {
	var value struct {
		Known *bool           `json:"known"`
		Value json.RawMessage `json:"value"`
	}
	if err := decodeResponse(data, &value, "known", "value"); err != nil {
		return err
	}
	if value.Known == nil || len(value.Value) == 0 {
		return errors.New("incomplete observation")
	}
	var observed *T
	if err := json.Unmarshal(value.Value, &observed); err != nil {
		return err
	}
	if *value.Known != (observed != nil) {
		return errors.New("inconsistent observation")
	}
	*o = Observation[T]{Known: *value.Known, Value: observed}
	return nil
}

// Source metadata and measured bytes are independent; neither authenticates a publisher.
type Identity struct {
	BuildID        Observation[string] `json:"buildId"`
	SourceRevision Observation[string] `json:"sourceRevision"`
	Dirty          Observation[bool]   `json:"dirty"`
	SHA256         Observation[string] `json:"sha256"`
}

// GUIFacts come from native main, never from a renderer or manifest.
type GUIFacts struct {
	PID        int    `json:"pid"`
	Executable string `json:"executable"`
	BuildID    string `json:"buildId"`
}
type PreviewRequest struct {
	CandidatePath string   `json:"candidatePath"`
	GUI           GUIFacts `json:"gui"`
}
type Consent struct {
	AllowInterruption bool `json:"allowInterruption"`
	AllowEngineStop   bool `json:"allowEngineStop"`
}
type StartRequest struct {
	PreviewID string   `json:"previewId"`
	Consent   Consent  `json:"consent"`
	GUI       GUIFacts `json:"gui"`
}
type StatusRequest struct {
	OperationID string `json:"operationId"`
}
type GUIReadyRequest struct {
	OperationID string   `json:"operationId"`
	GUI         GUIFacts `json:"gui"`
}

// Fences identify exact read lifetimes, not capabilities or schema generations.
type Fences struct {
	RuntimeGeneration    Observation[string] `json:"runtimeGeneration"`
	ServiceInstance      Observation[string] `json:"serviceInstance"`
	BrokerGeneration     Observation[string] `json:"brokerGeneration"`
	ManagerIncarnation   Observation[string] `json:"managerIncarnation"`
	PolicyRevision       Observation[string] `json:"policyRevision"`
	SettingsRevision     Observation[string] `json:"settingsRevision"`
	IntentRevision       Observation[string] `json:"intentRevision"`
	LocksRevision        Observation[string] `json:"locksRevision"`
	AvailabilityRevision Observation[string] `json:"availabilityRevision"`
}
type EngineFact struct {
	Engine              string                `json:"engine"`
	Active              Observation[bool]     `json:"active"`
	Running             Observation[bool]     `json:"running"`
	LoadedModels        Observation[[]string] `json:"loadedModels"`
	DesiredState        Observation[string]   `json:"desiredState"`
	Locked              Observation[bool]     `json:"locked"`
	ScopedStopAuthority Observation[bool]     `json:"scopedStopAuthority"`
}
type ModelOperation struct {
	Engine      string `json:"engine"`
	Kind        string `json:"kind"`
	BlocksApply bool   `json:"blocksApply"`
}

// Only public normalization differences and digests leave the settings owner.
type SettingsDifference struct {
	Field     string            `json:"field"`
	Stored    Observation[bool] `json:"stored"`
	Effective Observation[bool] `json:"effective"`
	Persisted Observation[bool] `json:"persisted"`
}
type ExpectedSettings struct {
	PolicyDigest         Observation[string] `json:"policyDigest"`
	EngineSettingsDigest Observation[string] `json:"engineSettingsDigest"`
	IntentDigest         Observation[string] `json:"intentDigest"`
	LocksDigest          Observation[string] `json:"locksDigest"`
	StopEngines          Observation[bool]   `json:"stopEngines"`
	UnloadModels         Observation[bool]   `json:"unloadModels"`
}
type InterruptionPolicy struct {
	OnActive                Observation[nodepolicy.OnActive] `json:"onActive"`
	DrainTimeoutSeconds     Observation[int]                 `json:"drainTimeoutSeconds"`
	StopsEngines            bool                             `json:"stopsEngines"`
	UnloadsModels           bool                             `json:"unloadsModels"`
	CommittedStreamsMayFail bool                             `json:"committedStreamsMayFail"`
	IncludesWorkBeforeFence bool                             `json:"includesWorkBeforeFence"`
}
type RestorePolicy struct {
	RemainPaused           bool `json:"remainPaused"`
	RequireLocalResume     bool `json:"requireLocalResume"`
	PreserveIntentAndLocks bool `json:"preserveIntentAndLocks"`
	PreserveNewerPause     bool `json:"preserveNewerPause"`
	AutomaticRollback      bool `json:"automaticRollback"`
}
type Refusal struct {
	Code   string `json:"code"`
	Detail string `json:"detail"`
}

// Contract only. Task 3 owns authoritative settings hooks, lock lifetime and
// reciprocal refusal tests, including internal settings callers.
type MaintenanceState struct {
	BuildApplyActive    Observation[bool] `json:"buildApplyActive"`
	ModelSettingsActive Observation[bool] `json:"modelSettingsActive"`
}
type Impact struct {
	Complete            bool                                 `json:"complete"`
	Fences              Fences                               `json:"fences"`
	Maintenance         MaintenanceState                     `json:"maintenance"`
	Availability        Observation[nodepolicy.Availability] `json:"availability"`
	PauseOrigin         Observation[string]                  `json:"pauseOrigin"`
	Engines             Observation[[]EngineFact]            `json:"engines"`
	ActiveWork          Observation[int]                     `json:"activeWork"`
	ModelOperations     Observation[[]ModelOperation]        `json:"modelOperations"`
	SettingsDifferences Observation[[]SettingsDifference]    `json:"settingsDifferences"`
	ExpectedSettings    ExpectedSettings                     `json:"expectedSettings"`
	InterruptionPolicy  InterruptionPolicy                   `json:"interruptionPolicy"`
	RestorePolicy       RestorePolicy                        `json:"restorePolicy"`
	Refusals            []Refusal                            `json:"refusals"`
}
type BuildApplyPreview struct {
	PreviewID            string              `json:"previewId"`
	ExpiresAt            string              `json:"expiresAt"`
	HostID               Observation[string] `json:"hostId"`
	Platform             string              `json:"platform"`
	Architecture         string              `json:"architecture"`
	CurrentGUI           Identity            `json:"currentGui"`
	BundledService       Identity            `json:"bundledService"`
	RequestedGUI         Identity            `json:"requestedGui"`
	RequestedService     Identity            `json:"requestedService"`
	RunningService       Identity            `json:"runningService"`
	PackageSHA256        Observation[string] `json:"packageSha256"`
	TargetLayout         Observation[Layout] `json:"targetLayout"`
	ObservationsComplete bool                `json:"observationsComplete"`
	Impact               Impact              `json:"impact"`
}
type Phase string

const (
	PhasePreflight      Phase = "preflight"
	PhasePrepare        Phase = "prepare"
	PhaseQuiesce        Phase = "quiesce"
	PhaseShutdown       Phase = "shutdown"
	PhaseSwap           Phase = "swap"
	PhaseRestart        Phase = "restart"
	PhaseVerifyBuild    Phase = "verify-build"
	PhaseVerifySettings Phase = "verify-settings"
	PhaseFinish         Phase = "finish"
)

type Outcome string

const (
	OutcomeSucceeded   Outcome = "succeeded"
	OutcomeRefused     Outcome = "refused"
	OutcomeFailed      Outcome = "failed"
	OutcomeInterrupted Outcome = "interrupted"
	OutcomeUncertain   Outcome = "uncertain"
)

type Confidence string

const (
	ConfidenceUnknown    Confidence = "unknown"
	ConfidenceIncomplete Confidence = "incomplete"
	ConfidenceVerified   Confidence = "verified"
)

type BuildApplyFailure struct {
	Phase              Phase             `json:"phase"`
	Reason             string            `json:"reason"`
	Detail             string            `json:"detail"`
	BytesChanged       Observation[bool] `json:"bytesChanged"`
	SettingsChanged    Observation[bool] `json:"settingsChanged"`
	RunningIdentity    Identity          `json:"runningIdentity"`
	ReadbackConfidence Confidence        `json:"readbackConfidence"`
	SafeNextAction     string            `json:"safeNextAction"`
}
type BuildApplyOperation struct {
	OperationID           string             `json:"operationId"`
	PreviewID             string             `json:"previewId"`
	OwnershipAcknowledged bool               `json:"ownershipAcknowledged"`
	Phase                 Phase              `json:"phase"`
	Outcome               *Outcome           `json:"outcome"`
	Failure               *BuildApplyFailure `json:"failure"`
	Saved                 Observation[bool]  `json:"saved"`
	Enforced              Observation[bool]  `json:"enforced"`
	PhysicalStop          Observation[bool]  `json:"physicalStop"`
	RemainsPaused         Observation[bool]  `json:"remainsPaused"`
	RunningService        Identity           `json:"runningService"`
	RunningGUI            Identity           `json:"runningGui"`
	ReadbackConfidence    Confidence         `json:"readbackConfidence"`
	Fences                Fences             `json:"fences"`
}

type StatusState string

const (
	StatusIdle     StatusState = "idle"
	StatusPending  StatusState = "pending"
	StatusTerminal StatusState = "terminal"
	StatusUnknown  StatusState = "unknown"
)

// Only idle establishes that no operation is recorded. Unknown retains a
// readable operation's correlation without asserting its current lifecycle.
// An unreadable receipt has no trustworthy correlation: return an unavailable
// Result[BuildApplyStatus], never an idle value or a fabricated preview ID.
// These observations alone do not authorize startup or ownership handoff.
type BuildApplyStatus struct {
	State     StatusState          `json:"state"`
	Operation *BuildApplyOperation `json:"operation"`
}

func (s BuildApplyStatus) validate() error {
	if s.State == StatusIdle {
		if s.Operation != nil {
			return errors.New("idle build apply status carries an operation")
		}
		return nil
	}
	if s.Operation == nil {
		return errors.New("build apply status lacks a correlated operation")
	}
	if err := s.Operation.validate(); err != nil {
		return err
	}
	switch s.State {
	case StatusPending:
		if s.Operation.Outcome != nil {
			return errors.New("pending build apply status carries an outcome")
		}
	case StatusTerminal:
		if s.Operation.Outcome == nil {
			return errors.New("terminal build apply status lacks an outcome")
		}
	case StatusUnknown:
	default:
		return errors.New("invalid build apply status state")
	}
	return nil
}

func (s BuildApplyStatus) MarshalJSON() ([]byte, error) {
	type wire BuildApplyStatus
	if err := s.validate(); err != nil {
		return nil, err
	}
	return json.Marshal(wire(s))
}

func (s *BuildApplyStatus) UnmarshalJSON(data []byte) error {
	type wire BuildApplyStatus
	var value wire
	if len(data) > MaxStatusBytes {
		return errors.New("build apply status exceeds limit")
	}
	if err := decodeResponse(data, &value, "state", "operation"); err != nil {
		return err
	}
	status := BuildApplyStatus(value)
	if err := status.validate(); err != nil {
		return err
	}
	*s = status
	return nil
}

func (o BuildApplyOperation) validate() error {
	if !token(o.OperationID, false) || !token(o.PreviewID, false) || !validPhase(o.Phase) ||
		!validConfidence(o.ReadbackConfidence) || !validIdentity(o.RunningService) || !validIdentity(o.RunningGUI) {
		return errors.New("invalid build apply operation")
	}
	if o.Outcome == nil {
		if o.Failure != nil {
			return errors.New("pending build apply operation carries a failure")
		}
	} else if !validOutcome(*o.Outcome) || (*o.Outcome == OutcomeSucceeded && o.Failure != nil) {
		return errors.New("invalid build apply outcome")
	}
	if o.Failure != nil && (!validFailure(*o.Failure) || o.Failure.Phase != o.Phase) {
		return errors.New("invalid build apply failure")
	}
	for _, value := range []Observation[string]{
		o.Fences.RuntimeGeneration, o.Fences.ServiceInstance, o.Fences.BrokerGeneration,
		o.Fences.ManagerIncarnation, o.Fences.PolicyRevision, o.Fences.SettingsRevision,
		o.Fences.IntentRevision, o.Fences.LocksRevision, o.Fences.AvailabilityRevision,
	} {
		if !validObservation(value, func(value string) bool { return responseText(value, MaxPathBytes) }) {
			return errors.New("invalid build apply fence")
		}
	}
	for _, value := range []Observation[bool]{o.Saved, o.Enforced, o.PhysicalStop, o.RemainsPaused} {
		if !validObservation(value, func(bool) bool { return true }) {
			return errors.New("invalid build apply observation")
		}
	}
	return nil
}

func (o BuildApplyOperation) MarshalJSON() ([]byte, error) {
	type wire BuildApplyOperation
	if err := o.validate(); err != nil {
		return nil, err
	}
	return json.Marshal(wire(o))
}

func (o *BuildApplyOperation) UnmarshalJSON(data []byte) error {
	type wire BuildApplyOperation
	var value wire
	if err := decodeResponse(data, &value,
		"operationId", "previewId", "ownershipAcknowledged", "phase", "outcome", "failure",
		"saved", "enforced", "physicalStop", "remainsPaused", "runningService", "runningGui",
		"readbackConfidence", "fences"); err != nil {
		return err
	}
	operation := BuildApplyOperation(value)
	if err := operation.validate(); err != nil {
		return err
	}
	*o = operation
	return nil
}

func (i *Identity) UnmarshalJSON(data []byte) error {
	type wire Identity
	var value wire
	if err := decodeResponse(data, &value, "buildId", "sourceRevision", "dirty", "sha256"); err != nil {
		return err
	}
	if !validIdentity(Identity(value)) {
		return errors.New("invalid build identity")
	}
	*i = Identity(value)
	return nil
}

func (f *Fences) UnmarshalJSON(data []byte) error {
	type wire Fences
	var value wire
	if err := decodeResponse(data, &value,
		"runtimeGeneration", "serviceInstance", "brokerGeneration", "managerIncarnation",
		"policyRevision", "settingsRevision", "intentRevision", "locksRevision", "availabilityRevision"); err != nil {
		return err
	}
	*f = Fences(value)
	return nil
}

func (f *BuildApplyFailure) UnmarshalJSON(data []byte) error {
	type wire BuildApplyFailure
	var value wire
	if err := decodeResponse(data, &value,
		"phase", "reason", "detail", "bytesChanged", "settingsChanged", "runningIdentity",
		"readbackConfidence", "safeNextAction"); err != nil {
		return err
	}
	if !validFailure(BuildApplyFailure(value)) {
		return errors.New("invalid build apply failure")
	}
	*f = BuildApplyFailure(value)
	return nil
}

func validObservation[T any](value Observation[T], valid func(T) bool) bool {
	return value.Known == (value.Value != nil) && (!value.Known || valid(*value.Value))
}

func validIdentity(value Identity) bool {
	return validObservation(value.BuildID, func(value string) bool {
		return strings.HasPrefix(value, "git:") && wireRevision(strings.TrimSuffix(strings.TrimPrefix(value, "git:"), "+dirty"))
	}) && validObservation(value.SourceRevision, wireRevision) &&
		validObservation(value.Dirty, func(bool) bool { return true }) &&
		validObservation(value.SHA256, func(value string) bool { return len(value) == 64 && wireRevision(value) })
}

func wireRevision(value string) bool {
	if len(value) != 40 && len(value) != 64 {
		return false
	}
	for _, c := range value {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func validFailure(value BuildApplyFailure) bool {
	return validPhase(value.Phase) && responseText(value.Reason, 128) && responseText(value.Detail, MaxDetailBytes) &&
		validObservation(value.BytesChanged, func(bool) bool { return true }) &&
		validObservation(value.SettingsChanged, func(bool) bool { return true }) &&
		validIdentity(value.RunningIdentity) && validConfidence(value.ReadbackConfidence) && responseText(value.SafeNextAction, MaxDetailBytes)
}

func validPhase(value Phase) bool {
	switch value {
	case PhasePreflight, PhasePrepare, PhaseQuiesce, PhaseShutdown, PhaseSwap, PhaseRestart, PhaseVerifyBuild, PhaseVerifySettings, PhaseFinish:
		return true
	}
	return false
}

func validOutcome(value Outcome) bool {
	switch value {
	case OutcomeSucceeded, OutcomeRefused, OutcomeFailed, OutcomeInterrupted, OutcomeUncertain:
		return true
	}
	return false
}

func validConfidence(value Confidence) bool {
	return value == ConfidenceUnknown || value == ConfidenceIncomplete || value == ConfidenceVerified
}

func responseText(value string, limit int) bool {
	if !utf8.ValidString(value) || len(value) > limit {
		return false
	}
	for _, c := range value {
		if c < 32 {
			return false
		}
	}
	return true
}

type Result[T any] struct {
	Available bool     `json:"available"`
	Value     *T       `json:"value"`
	Refusal   *Refusal `json:"refusal"`
}

func (r Result[T]) validate() error {
	if r.Available {
		if r.Value == nil || r.Refusal != nil {
			return errors.New("invalid available build apply result")
		}
	} else if r.Value != nil || r.Refusal == nil || !responseText(r.Refusal.Code, 128) || !responseText(r.Refusal.Detail, MaxDetailBytes) {
		return errors.New("invalid unavailable build apply result")
	}
	return nil
}

func (r Result[T]) MarshalJSON() ([]byte, error) {
	type wire Result[T]
	if err := r.validate(); err != nil {
		return nil, err
	}
	return json.Marshal(wire(r))
}

func (r *Result[T]) UnmarshalJSON(data []byte) error {
	type wire Result[T]
	var value wire
	if err := decodeResponse(data, &value, "available", "value", "refusal"); err != nil {
		return err
	}
	result := Result[T](value)
	if err := result.validate(); err != nil {
		return err
	}
	*r = result
	return nil
}

func (r *Refusal) UnmarshalJSON(data []byte) error {
	type wire Refusal
	var value wire
	if err := decodeResponse(data, &value, "code", "detail"); err != nil {
		return err
	}
	if !responseText(value.Code, 128) || !responseText(value.Detail, MaxDetailBytes) {
		return errors.New("invalid build apply refusal")
	}
	*r = Refusal(value)
	return nil
}

// DecodeStatusResult accepts only bounded complete status output. A nonempty
// expectedPreviewID requires an exact match, including on unknown snapshots;
// idle cannot satisfy that correlation. An unavailable result grants nothing.
func DecodeStatusResult(data []byte, expectedPreviewID string) (Result[BuildApplyStatus], error) {
	var result Result[BuildApplyStatus]
	if len(data) == 0 || len(data) > MaxStatusBytes {
		return result, errors.New("invalid build apply status size")
	}
	if !token(expectedPreviewID, true) {
		return result, errors.New("invalid expected build apply preview")
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return Result[BuildApplyStatus]{}, err
	}
	if expectedPreviewID != "" && result.Available &&
		(result.Value.Operation == nil || result.Value.Operation.PreviewID != expectedPreviewID) {
		return Result[BuildApplyStatus]{}, errors.New("build apply preview mismatch")
	}
	return result, nil
}

// EncodeStatusResult bounds the entire command/RPC envelope, including refusal
// output. The command owner must propagate an error without emitting idle.
func EncodeStatusResult(result Result[BuildApplyStatus]) ([]byte, error) {
	data, err := json.Marshal(result)
	if err == nil && len(data) > MaxStatusBytes {
		return nil, errors.New("build apply status exceeds limit")
	}
	return data, err
}

// Response fields are required even when their value is explicitly null. This
// prevents encoding/json's zero values from turning absent facts into idle,
// known false, an unacknowledged operation or an unavailable result.
func decodeResponse(data []byte, value any, fields ...string) error {
	if err := uniqueJSON(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return errors.New("ambiguous build apply response")
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if len(raw) != len(fields) {
		return errors.New("incomplete build apply response")
	}
	for _, field := range fields {
		entry, ok := raw[field]
		if !ok {
			return errors.New("missing build apply response field")
		}
		switch field {
		case "outcome", "failure", "operation", "value", "refusal":
		default:
			if bytes.Equal(bytes.TrimSpace(entry), []byte("null")) {
				return errors.New("null build apply response field")
			}
		}
	}
	return decodeStrict(data, value)
}

func Unavailable[T any]() Result[T] {
	return Result[T]{Refusal: &Refusal{Code: RefusalUnavailable, Detail: "Build apply is unavailable."}}
}
func MaintenanceBusy[T any]() Result[T] {
	return Result[T]{Refusal: &Refusal{Code: RefusalMaintenanceBusy, Detail: "Build apply and model settings maintenance are mutually exclusive."}}
}

// These broker requests are service-owned, with no remote subject or destination.
type ImpactRequest struct{}
type PrepareRequest struct {
	OperationID string  `json:"operationId"`
	PreviewID   string  `json:"previewId"`
	Fences      Fences  `json:"fences"`
	Consent     Consent `json:"consent"`
}
type FinishRequest struct {
	OperationID string `json:"operationId"`
	Fences      Fences `json:"fences"`
}

// DecodeRequest is strict at every nesting level. Native owners must still
// revalidate actual paths/processes; well-formed facts are not authority.
func DecodeRequest(data []byte, request any) error {
	if len(data) == 0 || len(data) > MaxRequestBytes {
		return errors.New("invalid build apply request size")
	}
	if err := uniqueJSON(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return errors.New("ambiguous build apply request")
	}
	if err := decodeStrict(data, request); err != nil {
		return errors.New("invalid build apply request")
	}
	valid := true
	switch r := request.(type) {
	case *PreviewRequest:
		valid = localPath(r.CandidatePath) && validGUI(r.GUI)
	case *StartRequest:
		valid = token(r.PreviewID, false) && validGUI(r.GUI)
	case *StatusRequest:
		valid = token(r.OperationID, true)
	case *GUIReadyRequest:
		valid = token(r.OperationID, false) && validGUI(r.GUI)
	case *ImpactRequest:
	case *PrepareRequest:
		valid = token(r.OperationID, false) && token(r.PreviewID, false) && knownFences(r.Fences)
	case *FinishRequest:
		valid = token(r.OperationID, false) && knownFences(r.Fences)
	default:
		valid = false
	}
	if !valid {
		return errors.New("invalid build apply request facts")
	}
	return nil
}

func uniqueJSON(decoder *json.Decoder, depth int) error {
	if depth > 16 {
		return errors.New("request nesting limit")
	}
	value, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := value.(json.Delim)
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		keys := map[string]bool{}
		for decoder.More() {
			key, err := decoder.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || keys[name] {
				return errors.New("duplicate request field")
			}
			keys[name] = true
			if err := uniqueJSON(decoder, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for decoder.More() {
			if err := uniqueJSON(decoder, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("invalid delimiter")
	}
	_, err = decoder.Token()
	return err
}
func knownFences(f Fences) bool {
	for _, value := range []Observation[string]{
		f.RuntimeGeneration,
		f.ServiceInstance, f.BrokerGeneration, f.ManagerIncarnation, f.PolicyRevision,
		f.SettingsRevision, f.IntentRevision, f.LocksRevision, f.AvailabilityRevision,
	} {
		if !value.Known || value.Value == nil || len(*value.Value) == 0 || len(*value.Value) > 128 ||
			strings.ContainsAny(*value.Value, "\x00\r\n") {
			return false
		}
	}
	return true
}
func decodeStrict(data []byte, value any) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return errors.New("object required")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return errors.New("trailing JSON")
	}
	return nil
}
func token(value string, empty bool) bool {
	if value == "" {
		return empty
	}
	if len(value) > 128 {
		return false
	}
	for _, c := range value {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_') {
			return false
		}
	}
	return true
}
func localPath(value string) bool {
	return utf8.ValidString(value) && len(value) <= MaxPathBytes && filepath.IsAbs(value) && !strings.ContainsAny(value, "\x00\r\n")
}
func validGUI(gui GUIFacts) bool {
	return gui.PID > 0 && gui.PID <= 2147483647 && localPath(gui.Executable) && len(gui.BuildID) <= 128 && !strings.ContainsAny(gui.BuildID, "\x00\r\n")
}
