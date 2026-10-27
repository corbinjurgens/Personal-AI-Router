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
	if err := decodeStrict(data, &value); err != nil {
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
type Result[T any] struct {
	Available bool     `json:"available"`
	Value     *T       `json:"value"`
	Refusal   *Refusal `json:"refusal"`
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
		valid = token(r.OperationID, false) && knownFences(r.Fences)
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
