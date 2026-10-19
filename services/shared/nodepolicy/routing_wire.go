// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
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

func (k RoutingRequestKey) Validate() error {
	if !strictID.MatchString(k.OriginNodeID) {
		return errors.New("originNodeId must be a nonempty identity of at most 128 ASCII characters")
	}
	if !strictID.MatchString(k.ProxyRunID) {
		return errors.New("proxyRunId must be a nonempty identity of at most 128 ASCII characters")
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
