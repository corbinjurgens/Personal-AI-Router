// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// StrictAdmissionVersion is independent of the persisted policy schema version.
const StrictAdmissionVersion = 1

// StrictAdmissionHeader carries one JSON StrictAdmissionRequest on trusted
// inference ingress only. A1 must strip it before forwarding to an engine.
const StrictAdmissionHeader = "X-PAIR-Strict-Admission"

// ErrStrictAdmissionVersion distinguishes unsupported from malformed contracts.
var ErrStrictAdmissionVersion = errors.New("unsupported strict-admission version")

// RoutingRequestKey is origin-private, process-wide identity, never affinity or
// a per-facade workload ID. RequestID is a positive signed-64-bit decimal string.
type RoutingRequestKey struct {
	OriginNodeID string `json:"originNodeId"`
	ProxyRunID   string `json:"proxyRunId"`
	RequestID    string `json:"requestId"`
}

var strictID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
var strictDigest = regexp.MustCompile(`^[0-9a-f]{64}$`)

// ValidateRoutingIdentity checks the origin and process incarnation syntax.
func ValidateRoutingIdentity(originNodeID, proxyRunID string) error {
	if !strictID.MatchString(originNodeID) || !strictID.MatchString(proxyRunID) {
		return errors.New("originNodeId and proxyRunId must be ASCII identities of 1-128 bytes")
	}
	return nil
}

func (k RoutingRequestKey) Validate() error {
	if err := ValidateRoutingIdentity(k.OriginNodeID, k.ProxyRunID); err != nil {
		return err
	}
	n, err := NanoUSD(k.RequestID).Int64()
	if err != nil || n < 1 {
		return errors.New("requestId must be a positive canonical signed-64-bit decimal string")
	}
	return nil
}

func (k *RoutingRequestKey) UnmarshalJSON(data []byte) error {
	type wire RoutingRequestKey
	var next wire
	if err := decodeStrictContract(data, &next, "originNodeId", "proxyRunId", "requestId"); err != nil {
		return err
	}
	if err := RoutingRequestKey(next).Validate(); err != nil {
		return err
	}
	*k = RoutingRequestKey(next)
	return nil
}

// RoutingCancelStatus is the canonical outcome of request cancellation.
type RoutingCancelStatus string

const (
	RoutingCancelNotFound          RoutingCancelStatus = "not_found"
	RoutingCancelConflictCommitted RoutingCancelStatus = "conflict_committed"
	RoutingCancelRegenerating      RoutingCancelStatus = "regenerating"
	RoutingCancelCancelled         RoutingCancelStatus = "cancelled"
)

type RoutingCancelParams struct {
	Request    RoutingRequestKey `json:"request"`
	Regenerate bool              `json:"regenerate"`
}

func (p *RoutingCancelParams) UnmarshalJSON(data []byte) error {
	type wire RoutingCancelParams
	var next wire
	if err := decodeRoutingControl(data, &next, "request", "regenerate"); err != nil {
		return err
	}
	if err := next.Request.Validate(); err != nil {
		return err
	}
	*p = RoutingCancelParams(next)
	return nil
}

type RoutingCancelResult struct {
	Status RoutingCancelStatus `json:"status"`
}

func (r *RoutingCancelResult) UnmarshalJSON(data []byte) error {
	type wire RoutingCancelResult
	var next wire
	if err := decodeRoutingControl(data, &next, "status"); err != nil {
		return err
	}
	switch next.Status {
	case RoutingCancelNotFound, RoutingCancelConflictCommitted, RoutingCancelRegenerating, RoutingCancelCancelled:
	default:
		return errors.New("invalid routing cancellation status")
	}
	*r = RoutingCancelResult(next)
	return nil
}

type RoutingActiveParams struct{}

func (p *RoutingActiveParams) UnmarshalJSON(data []byte) error {
	type wire RoutingActiveParams
	var next wire
	return decodeRoutingControl(data, &next)
}

type RoutingActiveResult struct {
	Requests []RoutingRequestKey `json:"requests"`
}

func (r *RoutingActiveResult) UnmarshalJSON(data []byte) error {
	type wire RoutingActiveResult
	var next wire
	if err := decodeRoutingControl(data, &next, "requests"); err != nil {
		return err
	}
	if next.Requests == nil {
		return errors.New("requests must be an array")
	}
	*r = RoutingActiveResult(next)
	return nil
}

// Decode the original bytes so nested keys keep their duplicate-field checks
// and request IDs never pass through a floating-point representation.
func decodeRoutingControl(data []byte, dst any, required ...string) error {
	fields, err := peerFields(data)
	if err != nil {
		return err
	}
	for key := range fields {
		if !slices.Contains(required, key) {
			return fmt.Errorf("unknown field %q", key)
		}
	}
	for _, key := range required {
		v, ok := fields[key]
		if !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return fmt.Errorf("%s is required and must not be null", key)
		}
	}
	return json.Unmarshal(data, dst)
}

// StrictModelIdentity fences engine restart, effective launch/settings changes,
// and same-name model replacement. Digests are lowercase SHA-256 hex of public
// configuration/content identity, never credentials. Qualification is separate.
type StrictModelIdentity struct {
	EngineRunID         string `json:"engineRunId"`
	SettingsFingerprint string `json:"settingsFingerprint"`
	ModelDigest         string `json:"modelDigest"`
}

func (i StrictModelIdentity) Validate() error {
	if !strictID.MatchString(i.EngineRunID) {
		return errors.New("engineRunId must be a nonempty identity of at most 128 ASCII characters")
	}
	if !strictDigest.MatchString(i.SettingsFingerprint) {
		return errors.New("settingsFingerprint must be lowercase SHA-256 hex")
	}
	if !strictDigest.MatchString(i.ModelDigest) {
		return errors.New("modelDigest must be lowercase SHA-256 hex")
	}
	return nil
}

func (i *StrictModelIdentity) UnmarshalJSON(data []byte) error {
	type wire StrictModelIdentity
	var next wire
	if err := decodeStrictContract(data, &next, "engineRunId", "settingsFingerprint", "modelDigest"); err != nil {
		return err
	}
	if err := StrictModelIdentity(next).Validate(); err != nil {
		return err
	}
	*i = StrictModelIdentity(next)
	return nil
}

// StrictAdmissionRequest uses remaining durations, never cross-machine wall
// clocks. Queue and cold waits both consume PreOutputBudgetMs. Zero cold budget
// forbids wake/unload; no timeout may turn a rejection into admission.
type StrictAdmissionRequest struct {
	Version           int                 `json:"version"`
	QueueBudgetMs     int                 `json:"queueBudgetMs"`
	ColdStartBudgetMs int                 `json:"coldStartBudgetMs"`
	PreOutputBudgetMs int                 `json:"preOutputBudgetMs"`
	Request           RoutingRequestKey   `json:"request"`
	Identity          StrictModelIdentity `json:"identity"`
}

func (r StrictAdmissionRequest) Validate() error {
	if r.Version != StrictAdmissionVersion {
		return ErrStrictAdmissionVersion
	}
	if r.PreOutputBudgetMs < 1 || r.PreOutputBudgetMs > maxLimitSeconds*1000 {
		return errors.New("preOutputBudgetMs must be between 1 and 600000")
	}
	if r.QueueBudgetMs < 0 || r.QueueBudgetMs > r.PreOutputBudgetMs {
		return errors.New("queueBudgetMs must be between 0 and preOutputBudgetMs")
	}
	if r.ColdStartBudgetMs < 0 || r.ColdStartBudgetMs > r.PreOutputBudgetMs {
		return errors.New("coldStartBudgetMs must be between 0 and preOutputBudgetMs")
	}
	if err := r.Request.Validate(); err != nil {
		return fmt.Errorf("request: %w", err)
	}
	if err := r.Identity.Validate(); err != nil {
		return fmt.Errorf("identity: %w", err)
	}
	return nil
}

func (r *StrictAdmissionRequest) UnmarshalJSON(data []byte) error {
	type wire StrictAdmissionRequest
	var next wire
	if err := decodeStrictContract(data, &next, "version", "queueBudgetMs", "coldStartBudgetMs", "preOutputBudgetMs", "request", "identity"); err != nil {
		return err
	}
	if err := StrictAdmissionRequest(next).Validate(); err != nil {
		return err
	}
	*r = StrictAdmissionRequest(next)
	return nil
}

// decodeStrictContract rejects duplicate keys, unknown fields and null/missing
// required fields. peerFields scans keys before decoding so a repeated key
// cannot silently collapse to the last occurrence. Canonicalizing JSON numbers
// first accepts integral spellings such as 1.0 and 1e0 exactly as the desktop's
// JSON parser does. All numeric bounds fit int32.
func decodeStrictContract(data []byte, dst any, required ...string) error {
	fields, err := peerFields(data)
	if err != nil {
		return err
	}
	for key := range fields {
		if !slices.Contains(required, key) {
			return fmt.Errorf("unknown field %q", key)
		}
	}
	for _, key := range required {
		v, ok := fields[key]
		if !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return fmt.Errorf("%s is required and must not be null", key)
		}
	}
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	normalized, err := json.Marshal(value)
	if err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(normalized))
	d.DisallowUnknownFields()
	return d.Decode(dst)
}

// History bounds are independent of inference body and live-registry budgets.
const (
	RoutingHistoryRecordBytes        = 128 * 1024
	RoutingHistoryInitialBytes       = 32 * 1024 * 1024
	RoutingHistoryRecords            = 256
	RoutingHistoryAttempts           = 32
	routingHistoryDepth              = 16
	routingHistoryMembers            = 2048
	routingHistorySafeInteger  int64 = 9007199254740991
)

// RoutingHistoryDiagnostic contains only a fixed category and a count. Raw
// decoder errors, unknown names and values must never reach diagnostics.
type RoutingHistoryDiagnostic struct {
	Category string
	Count    int
}

func (d RoutingHistoryDiagnostic) Error() string { return d.Category }

func historyError(category string) error {
	return RoutingHistoryDiagnostic{Category: category, Count: 1}
}

type RoutingHistoryDestination struct {
	Kind         string  `json:"kind"`
	Model        string  `json:"model"`
	NodeID       *string `json:"nodeId"`
	Engine       *string `json:"engine"`
	DeploymentID *string `json:"deploymentId"`
	AccountID    *string `json:"accountId"`
}

type RoutingHistoryWorkload struct {
	OriginNodeID string `json:"originNodeId"`
	ProxyRunID   string `json:"proxyRunId"`
	Engine       string `json:"engine"`
	ID           string `json:"id"`
	AttemptIndex int64  `json:"attemptIndex"`
	Attached     bool   `json:"attached"`
}

type RoutingHistoryRequested struct {
	Model string `json:"model"`
	Tier  string `json:"tier"`
	Mode  string `json:"mode"`
}

type RoutingHistoryWaits struct {
	AdmissionMs int64  `json:"admissionMs"`
	FirstByteMs *int64 `json:"firstByteMs"`
}

type RoutingHistoryAccount struct {
	State          string   `json:"state"`
	HeldNanoUSD    *NanoUSD `json:"heldNanoUsd"`
	ChargedNanoUSD *NanoUSD `json:"chargedNanoUsd"`
	Accounting     string   `json:"accounting"`
}

type RoutingHistoryAttempt struct {
	Index          int64                     `json:"index"`
	StartedAt      int64                     `json:"startedAt"`
	EndedAt        *int64                    `json:"endedAt"`
	Destination    RoutingHistoryDestination `json:"destination"`
	Outcome        string                    `json:"outcome"`
	Reason         string                    `json:"reason"`
	Status         *int64                    `json:"status"`
	Waits          RoutingHistoryWaits       `json:"waits"`
	ContinuationOf *int64                    `json:"continuationOf"`
	Account        *RoutingHistoryAccount    `json:"account"`
}

type RoutingHistorySpend struct {
	Currency       string   `json:"currency"`
	CeilingNanoUSD *NanoUSD `json:"ceilingNanoUsd"`
	GrantedNanoUSD *NanoUSD `json:"grantedNanoUsd"`
	HeldNanoUSD    *NanoUSD `json:"heldNanoUsd"`
	ChargedNanoUSD *NanoUSD `json:"chargedNanoUsd"`
	Accounting     string   `json:"accounting"`
}

// Nullable fields deliberately have no omitempty: null is part of schema 1.
type RoutingHistoryRecord struct {
	Schema          int64                      `json:"schema"`
	Request         RoutingRequestKey          `json:"request"`
	Workload        *RoutingHistoryWorkload    `json:"workload"`
	Seq             int64                      `json:"seq"`
	Requested       RoutingHistoryRequested    `json:"requested"`
	CreatedAt       int64                      `json:"createdAt"`
	EndedAt         *int64                     `json:"endedAt"`
	State           string                     `json:"state"`
	Reason          string                     `json:"reason"`
	Status          *int64                     `json:"status"`
	Served          *RoutingHistoryDestination `json:"served"`
	Recoverable     bool                       `json:"recoverable"`
	Spend           RoutingHistorySpend        `json:"spend"`
	DroppedAttempts int64                      `json:"droppedAttempts"`
	Attempts        []RoutingHistoryAttempt    `json:"attempts"`
}

type RoutingHistoryInitial struct {
	Requests []RoutingHistoryRecord `json:"requests"`
}

func (r *RoutingHistoryRecord) UnmarshalJSON(data []byte) error {
	next, _, err := DecodeRoutingHistoryRecord(data)
	if err == nil {
		*r = next
	}
	return err
}

func (r *RoutingHistoryInitial) UnmarshalJSON(data []byte) error {
	next, _, err := DecodeRoutingHistoryInitial(data)
	if err == nil {
		*r = next
	}
	return err
}

// DecodeRoutingHistoryRecord validates original bytes before projection can
// collapse duplicate names. Use this entry point at raw transport boundaries.
func DecodeRoutingHistoryRecord(data []byte) (RoutingHistoryRecord, RoutingHistoryDiagnostic, error) {
	var next RoutingHistoryRecord
	if err := historyScan(data, RoutingHistoryRecordBytes, routingHistoryDepth, routingHistoryMembers); err != nil {
		return next, RoutingHistoryDiagnostic{}, err
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	var raw any
	if err := d.Decode(&raw); err != nil {
		return next, RoutingHistoryDiagnostic{}, historyError("invalid-shape")
	}
	stripped := 0
	if err := historyProject(raw, reflect.ValueOf(&next).Elem(), &stripped); err != nil {
		return RoutingHistoryRecord{}, RoutingHistoryDiagnostic{}, err
	}
	if err := next.Validate(); err != nil {
		return RoutingHistoryRecord{}, RoutingHistoryDiagnostic{}, err
	}
	encoded, err := json.Marshal(next)
	if err != nil {
		return RoutingHistoryRecord{}, RoutingHistoryDiagnostic{}, historyError("invalid-value")
	}
	if len(encoded) > RoutingHistoryRecordBytes {
		return RoutingHistoryRecord{}, RoutingHistoryDiagnostic{}, historyError("limit-exceeded")
	}
	return next, historyStripped(stripped), nil
}

func historyStripped(count int) RoutingHistoryDiagnostic {
	if count == 0 {
		return RoutingHistoryDiagnostic{}
	}
	return RoutingHistoryDiagnostic{Category: "unknown-fields-stripped", Count: count}
}

func DecodeRoutingHistoryInitial(data []byte) (RoutingHistoryInitial, RoutingHistoryDiagnostic, error) {
	var next RoutingHistoryInitial
	// The envelope adds an object and array around each record. Per-record
	// limits are checked against each original RawMessage below. Envelope
	// unknowns remain bounded by bytes, without borrowing a record's members.
	if err := historyScan(data, RoutingHistoryInitialBytes, routingHistoryDepth+2, RoutingHistoryInitialBytes); err != nil {
		return next, RoutingHistoryDiagnostic{}, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil || fields == nil {
		return next, RoutingHistoryDiagnostic{}, historyError("invalid-shape")
	}
	raw, ok := fields["requests"]
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return next, RoutingHistoryDiagnostic{}, historyError("invalid-shape")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	start, err := d.Token()
	if err != nil || start != json.Delim('[') {
		return next, RoutingHistoryDiagnostic{}, historyError("invalid-shape")
	}
	next.Requests = make([]RoutingHistoryRecord, 0)
	stripped := len(fields) - 1
	for d.More() {
		if len(next.Requests) == RoutingHistoryRecords {
			return RoutingHistoryInitial{}, RoutingHistoryDiagnostic{}, historyError("limit-exceeded")
		}
		var record json.RawMessage
		if err := d.Decode(&record); err != nil {
			return RoutingHistoryInitial{}, RoutingHistoryDiagnostic{}, historyError("invalid-shape")
		}
		value, diag, err := DecodeRoutingHistoryRecord(record)
		if err != nil {
			return RoutingHistoryInitial{}, RoutingHistoryDiagnostic{}, err
		}
		next.Requests = append(next.Requests, value)
		stripped += diag.Count
	}
	encoded, err := json.Marshal(next)
	if err != nil {
		return RoutingHistoryInitial{}, RoutingHistoryDiagnostic{}, historyError("invalid-value")
	}
	if len(encoded) > RoutingHistoryInitialBytes {
		return RoutingHistoryInitial{}, RoutingHistoryDiagnostic{}, historyError("limit-exceeded")
	}
	return next, historyStripped(stripped), nil
}

// Walk all raw subtrees, including discarded fields, before map decoding.
func historyScan(data []byte, size, depth, members int) error {
	if len(data) > size {
		return historyError("limit-exceeded")
	}
	if !utf8.Valid(data) {
		return historyError("invalid-value")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	count := 0
	if err := historyWalk(d, 1, depth, members, &count); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return historyError("invalid-shape")
	}
	return nil
}

func historyWalk(d *json.Decoder, level, depth, members int, count *int) error {
	token, err := d.Token()
	if err != nil {
		return historyError("invalid-shape")
	}
	delim, container := token.(json.Delim)
	if !container {
		return nil
	}
	if level > depth {
		return historyError("limit-exceeded")
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			keyToken, err := d.Token()
			if err != nil {
				return historyError("invalid-shape")
			}
			key, ok := keyToken.(string)
			if !ok {
				return historyError("invalid-shape")
			}
			if seen[key] {
				return historyError("duplicate-field")
			}
			seen[key] = true
			(*count)++
			if *count > members {
				return historyError("limit-exceeded")
			}
			if err := historyWalk(d, level+1, depth, members, count); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := historyWalk(d, level+1, depth, members, count); err != nil {
				return err
			}
		}
	default:
		return historyError("invalid-shape")
	}
	end, err := d.Token()
	if err != nil || (delim == '{' && end != json.Delim('}')) || (delim == '[' && end != json.Delim(']')) {
		return historyError("invalid-shape")
	}
	return nil
}

// The Go field tags are the projection allowlist and required-key list. Pointer
// fields are present-but-nullable; no permissive json.Unmarshal null coercions
// or case-insensitive field matches are used here.
func historyProject(raw any, dst reflect.Value, stripped *int) error {
	if dst.Kind() == reflect.Pointer {
		if raw == nil {
			return nil
		}
		dst.Set(reflect.New(dst.Type().Elem()))
		return historyProject(raw, dst.Elem(), stripped)
	}
	if raw == nil {
		return historyError("invalid-shape")
	}
	switch dst.Kind() {
	case reflect.Struct:
		fields, ok := raw.(map[string]any)
		if !ok {
			return historyError("invalid-shape")
		}
		known := 0
		for i := 0; i < dst.NumField(); i++ {
			name := dst.Type().Field(i).Tag.Get("json")
			value, present := fields[name]
			if !present {
				return historyError("invalid-shape")
			}
			if err := historyProject(value, dst.Field(i), stripped); err != nil {
				return err
			}
			known++
		}
		*stripped += len(fields) - known
	case reflect.Slice:
		values, ok := raw.([]any)
		if !ok {
			return historyError("invalid-shape")
		}
		if len(values) > RoutingHistoryAttempts {
			return historyError("limit-exceeded")
		}
		dst.Set(reflect.MakeSlice(dst.Type(), len(values), len(values)))
		for i, value := range values {
			if err := historyProject(value, dst.Index(i), stripped); err != nil {
				return err
			}
		}
	case reflect.String:
		value, ok := raw.(string)
		if !ok {
			return historyError("invalid-shape")
		}
		// json.Decoder already owns these bytes. Only the bounded, validated
		// typed projection escapes; neither raw input nor unknown maps survive.
		dst.SetString(value)
	case reflect.Bool:
		value, ok := raw.(bool)
		if !ok {
			return historyError("invalid-shape")
		}
		dst.SetBool(value)
	case reflect.Int64:
		value, ok := raw.(json.Number)
		if !ok {
			return historyError("invalid-shape")
		}
		n, ok := historyInteger(string(value))
		if !ok {
			return historyError("invalid-value")
		}
		dst.SetInt(n)
	default:
		return historyError("invalid-shape")
	}
	return nil
}

// Exact integral JSON numbers without float rounding or exponent-sized memory.
func historyInteger(value string) (int64, bool) {
	negative := strings.HasPrefix(value, "-")
	value = strings.TrimPrefix(value, "-")
	mantissa, exponent, hasExponent := strings.Cut(strings.ToLower(value), "e")
	whole, fraction, _ := strings.Cut(mantissa, ".")
	digits := strings.TrimLeft(whole+fraction, "0")
	if digits == "" {
		return 0, true
	}
	if negative {
		return 0, false
	}
	var shift int64
	if hasExponent {
		var err error
		shift, err = strconv.ParseInt(exponent, 10, 64)
		if err != nil || shift < -RoutingHistoryRecordBytes || shift > RoutingHistoryRecordBytes {
			return 0, false
		}
	}
	shift -= int64(len(fraction))
	for shift < 0 && strings.HasSuffix(digits, "0") {
		digits = strings.TrimSuffix(digits, "0")
		shift++
	}
	if shift < 0 || shift > 16 || int64(len(digits))+shift > 16 {
		return 0, false
	}
	digits += strings.Repeat("0", int(shift))
	n, err := strconv.ParseInt(digits, 10, 64)
	return n, err == nil && n <= routingHistorySafeInteger
}

func historyEnum(value, choices string) bool {
	return slices.Contains(strings.Split(choices, "|"), value)
}

const historyReasons = "|unknown|client-gone|cancel|regenerate|node-paused|proxy-shutdown|proxy-exited|target-left|transport|upstream-status|no-content|stream-interrupted|upstream-error|admission-paused|admission-draining|admission-busy|admission-no-fit|admission-engine-off|admission-wake-timeout|engine-not-running|request-invalid|no-owner|regenerate-exhausted|attempts-exhausted|deadline|request-ceiling|account-ceiling|account-cooldown|account-frozen|account-credential|account-exhausted|account-conflict|ledger-unavailable|deployment-quarantined|peer-framing"
const historyAccounting = "none|pending|recorded|retained|failed|unknown"

func historyModel(model string, empty bool) bool {
	if (!empty && model == "") || len(model) > 512 || !utf8.ValidString(model) {
		return false
	}
	return !strings.ContainsFunc(model, unicode.IsControl)
}

func historyNumber(n int64) bool          { return n >= 0 && n <= routingHistorySafeInteger }
func historyNullableNumber(n *int64) bool { return n == nil || historyNumber(*n) }
func historyStatus(n *int64) bool         { return n == nil || (*n >= 100 && *n <= 599) }
func historyID(id *string) bool           { return id != nil && strictID.MatchString(*id) }
func historyMoney(values ...*NanoUSD) bool {
	for _, value := range values {
		if value != nil {
			if _, err := ParseNanoUSD(string(*value)); err != nil {
				return false
			}
		}
	}
	return true
}

func (d RoutingHistoryDestination) valid() bool {
	if !historyModel(d.Model, false) {
		return false
	}
	switch d.Kind {
	case "local":
		return historyID(d.NodeID) && historyID(d.Engine) && d.DeploymentID == nil && d.AccountID == nil
	case "api":
		return historyID(d.DeploymentID) && historyID(d.AccountID) && d.NodeID == nil && d.Engine == nil
	default:
		return false
	}
}

// Validate checks only schema facts. Links into the dropped prefix must still
// be verified by the future producer against I's complete attempt facts.
func (r RoutingHistoryRecord) Validate() error {
	invalid := func() error { return historyError("invalid-value") }
	// Retained lifecycle facts establish that I already validated a model,
	// even when all attempts have since left the presentation window.
	modelValidated := len(r.Attempts) > 0 || r.DroppedAttempts > 0 || r.Workload != nil || r.Served != nil ||
		r.State == "completed" || r.State == "streaming"
	if r.Schema != 1 || r.Request.Validate() != nil || !historyNumber(r.Seq) || r.Seq == 0 ||
		!historyNumber(r.CreatedAt) || !historyNullableNumber(r.EndedAt) || (r.EndedAt != nil && *r.EndedAt < r.CreatedAt) ||
		!historyStatus(r.Status) || !historyNumber(r.DroppedAttempts) || !historyModel(r.Requested.Model, !modelValidated) ||
		!historyEnum(r.Requested.Tier, "|weak|medium|strong") || !historyEnum(r.Requested.Mode, "chat|completion|embedding|other") ||
		!historyEnum(r.State, "routing|streaming|completed|failed|cancelled") || !historyEnum(r.Reason, historyReasons) ||
		r.Spend.Currency != "USD" || !historyEnum(r.Spend.Accounting, historyAccounting) ||
		!historyMoney(r.Spend.CeilingNanoUSD, r.Spend.GrantedNanoUSD, r.Spend.HeldNanoUSD, r.Spend.ChargedNanoUSD) || r.Attempts == nil {
		return invalid()
	}
	if len(r.Attempts) > RoutingHistoryAttempts {
		return historyError("limit-exceeded")
	}
	terminal := historyEnum(r.State, "completed|failed|cancelled")
	if terminal && (r.EndedAt == nil || r.Recoverable) {
		return invalid()
	}
	if !terminal && r.EndedAt != nil {
		return invalid()
	}
	if (r.State == "completed") != (r.Served != nil) {
		return invalid()
	}
	if r.Served != nil && !r.Served.valid() {
		return invalid()
	}
	served, awaiting := false, false
	for i, a := range r.Attempts {
		// Subtraction avoids overflowing the safe-integer domain at its edge.
		if !historyNumber(a.Index) || a.Index < r.DroppedAttempts || a.Index-r.DroppedAttempts != int64(i) ||
			!historyNumber(a.StartedAt) || a.StartedAt < r.CreatedAt || !historyNullableNumber(a.EndedAt) ||
			(a.EndedAt != nil && *a.EndedAt < a.StartedAt) || (r.EndedAt != nil && (a.StartedAt > *r.EndedAt || (a.EndedAt != nil && *a.EndedAt > *r.EndedAt))) ||
			!a.Destination.valid() || !historyEnum(a.Outcome, "pending|committed|rejected|failed|redispatched|interrupted|cancelled") ||
			!historyEnum(a.Reason, historyReasons) || !historyStatus(a.Status) || !historyNumber(a.Waits.AdmissionMs) ||
			!historyNullableNumber(a.Waits.FirstByteMs) || !historyNullableNumber(a.ContinuationOf) {
			return invalid()
		}
		if terminal && a.EndedAt == nil {
			return invalid()
		}
		if historyEnum(a.Outcome, "rejected|failed|redispatched|interrupted|cancelled") && a.EndedAt == nil {
			return invalid()
		}
		if a.Destination.Kind == "local" {
			if a.Account != nil {
				return invalid()
			}
		} else {
			if a.Account == nil || !historyEnum(a.Account.State, "ok|cooldown|frozen|credential-failure|exhausted|probe|conflict|unknown") ||
				!historyEnum(a.Account.Accounting, historyAccounting) || !historyMoney(a.Account.HeldNanoUSD, a.Account.ChargedNanoUSD) {
				return invalid()
			}
		}
		if a.ContinuationOf != nil {
			predecessor := *a.ContinuationOf
			if predecessor >= a.Index {
				return invalid()
			}
			if predecessor >= r.DroppedAttempts {
				offset := predecessor - r.DroppedAttempts
				if offset >= int64(i) || r.Attempts[int(offset)].Outcome != "interrupted" {
					return invalid()
				}
			}
		}
		if a.Outcome == "interrupted" {
			linked := false
			for _, successor := range r.Attempts[i+1:] {
				if successor.ContinuationOf != nil && *successor.ContinuationOf == a.Index {
					linked = true
					break
				}
			}
			if !linked {
				awaiting = true
			}
		}
		if r.Served != nil && a.Outcome == "committed" && a.EndedAt != nil && reflect.DeepEqual(a.Destination, *r.Served) {
			served = true
		}
	}
	if r.Served != nil && !served {
		return invalid()
	}
	if r.Recoverable && (r.State != "streaming" || !awaiting) {
		return invalid()
	}
	if w := r.Workload; w != nil {
		if w.OriginNodeID != r.Request.OriginNodeID || w.ProxyRunID != r.Request.ProxyRunID || !strictID.MatchString(w.Engine) ||
			!strictID.MatchString(w.ID) || !historyNumber(w.AttemptIndex) {
			return invalid()
		}
		if w.AttemptIndex < r.DroppedAttempts {
			if w.Attached {
				return invalid()
			}
		} else {
			offset := w.AttemptIndex - r.DroppedAttempts
			if offset >= int64(len(r.Attempts)) {
				return invalid()
			}
			a := r.Attempts[int(offset)]
			if a.Destination.Kind != "local" || a.Destination.Engine == nil || *a.Destination.Engine != w.Engine {
				return invalid()
			}
		}
	}
	return nil
}
