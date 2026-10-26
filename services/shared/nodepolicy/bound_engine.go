// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

import (
	"errors"
	"strconv"
	"time"
	"unicode/utf8"

	"nvpair-shared/engines"
)

// These contracts belong exclusively to the proxy, broker and manager parent
// channels. A peer identity is evidence from pinned ingress, never HTTP input.
const (
	MethodPrepareBoundEngine   = "admission/prepare-bound-engine"
	MethodCancelBoundEngine    = "admission/cancel-bound-engine"
	NotifyBoundEnginePrepared  = "node/bound-engine-prepared"
	MethodActivateBoundEngine  = "engine:activate-bound-engine"
	MaxBoundEngineBudgetMs     = 90_000
	MaxBoundEngineMessageBytes = 256
	EnginePrepareBudgetHeader  = "X-PAIR-Engine-Prepare-Budget-Ms"
	SwitchRequestIDHeader      = "X-PAIR-Switch-Request-Id"
)

// EnginePrepareBudget is an allowance, never activation authority. Absence
// permits the normal bound; malformed, repeated and nonpositive values fail
// closed. Parse before multiplying so an overflowing value cannot wrap time.
func EnginePrepareBudget(values []string) time.Duration {
	if len(values) == 0 {
		return MaxBoundEngineBudgetMs * time.Millisecond
	}
	if len(values) != 1 {
		return 0
	}
	n, err := strconv.ParseUint(values[0], 10, 64)
	if err != nil || n == 0 || strconv.FormatUint(n, 10) != values[0] {
		return 0
	}
	return time.Duration(min(n, MaxBoundEngineBudgetMs)) * time.Millisecond
}

type PrepareBoundEngineRequest struct {
	RequestID       string `json:"requestId"`
	ProxyRunID      string `json:"proxyRunId"`
	PolicyRevision  string `json:"policyRevision"`
	Engine          string `json:"engine"`
	Model           string `json:"model"`
	BudgetMs        int    `json:"budgetMs"`
	PeerID          string `json:"peerId,omitempty"`
	PeerIncarnation string `json:"peerIncarnation,omitempty"`
}

func boundEnginePeer(peer, incarnation string) error {
	if peer == "" && incarnation == "" {
		return nil
	}
	if !strictID.MatchString(peer) || !strictID.MatchString(incarnation) {
		return errors.New("peerId and peerIncarnation must be a complete bounded identity pair")
	}
	return nil
}

func boundEnginePeerFields(data []byte, peer, incarnation string) error {
	fields, err := peerFields(data)
	if err != nil {
		return err
	}
	if (fields["peerId"] != nil || fields["peerIncarnation"] != nil) && (peer == "" || incarnation == "") {
		return errors.New("local requests must omit both peer fields")
	}
	return nil
}

func (p PrepareBoundEngineRequest) Validate() error {
	if err := (CancelBoundEngineRequest{p.RequestID, p.ProxyRunID}).Validate(); err != nil {
		return err
	}
	if err := validateRevision(p.PolicyRevision); err != nil {
		return err
	}
	if err := checkModelRef(p.Engine, p.Model, 0, nil); err != nil {
		return err
	}
	if p.BudgetMs < 1 || p.BudgetMs > MaxBoundEngineBudgetMs {
		return errors.New("budgetMs must be between 1 and 90000")
	}
	return boundEnginePeer(p.PeerID, p.PeerIncarnation)
}

func (p *PrepareBoundEngineRequest) UnmarshalJSON(data []byte) error {
	type plain PrepareBoundEngineRequest
	var next plain
	if err := decodeResident(data, &next, []string{"requestId", "proxyRunId", "policyRevision", "engine", "model", "budgetMs"}, "peerId", "peerIncarnation"); err != nil {
		return err
	}
	if err := boundEnginePeerFields(data, next.PeerID, next.PeerIncarnation); err != nil {
		return err
	}
	if err := PrepareBoundEngineRequest(next).Validate(); err != nil {
		return err
	}
	*p = PrepareBoundEngineRequest(next)
	return nil
}

type CancelBoundEngineRequest struct {
	RequestID  string `json:"requestId"`
	ProxyRunID string `json:"proxyRunId"`
}

func (p CancelBoundEngineRequest) Validate() error {
	if !strictID.MatchString(p.RequestID) || !strictID.MatchString(p.ProxyRunID) {
		return errors.New("requestId and proxyRunId must be bounded ASCII identities")
	}
	return nil
}

func (p *CancelBoundEngineRequest) UnmarshalJSON(data []byte) error {
	type plain CancelBoundEngineRequest
	var next plain
	if err := decodeResident(data, &next, []string{"requestId", "proxyRunId"}); err != nil {
		return err
	}
	if err := CancelBoundEngineRequest(next).Validate(); err != nil {
		return err
	}
	*p = CancelBoundEngineRequest(next)
	return nil
}

type BoundEnginePrepared struct {
	RequestID       string `json:"requestId"`
	ProxyRunID      string `json:"proxyRunId"`
	Success         bool   `json:"success"`
	Reason          string `json:"reason,omitempty"`
	Message         string `json:"message,omitempty"`
	Code            int    `json:"code,omitempty"`
	SwitchRequestID string `json:"switchRequestId,omitempty"`
}

func (p BoundEnginePrepared) Validate() error {
	if err := (CancelBoundEngineRequest{p.RequestID, p.ProxyRunID}).Validate(); err != nil {
		return err
	}
	if p.SwitchRequestID != "" && !strictID.MatchString(p.SwitchRequestID) {
		return errors.New("invalid switchRequestId")
	}
	if p.Success {
		if p.Reason != "" || p.Message != "" || p.Code != 0 {
			return errors.New("success must not carry a refusal")
		}
		return nil
	}
	switch p.Reason {
	case "engine-switch-timeout", "engine-switch-busy", "engine-switch-cooldown", "engine-locked", "management-denied", "model-unavailable", "paused", "draining", "no-fit", "stale-revision", "unavailable":
	default:
		return errors.New("invalid preparation reason")
	}
	if p.Code != -32000 && p.Code != -32010 && p.Code != -32011 && p.Code != -32012 && p.Code != -32013 && p.Code != -32602 {
		return errors.New("invalid preparation RPC code")
	}
	if p.Message == "" || len(p.Message) > MaxBoundEngineMessageBytes || !utf8.ValidString(p.Message) {
		return errors.New("invalid preparation message")
	}
	for _, r := range p.Message {
		if r < 32 || r >= 127 && r <= 159 {
			return errors.New("invalid preparation message control character")
		}
	}
	return nil
}

func (p *BoundEnginePrepared) UnmarshalJSON(data []byte) error {
	type plain BoundEnginePrepared
	var next plain
	if err := decodeResident(data, &next, []string{"requestId", "proxyRunId", "success"}, "reason", "message", "code", "switchRequestId"); err != nil {
		return err
	}
	if err := BoundEnginePrepared(next).Validate(); err != nil {
		return err
	}
	*p = BoundEnginePrepared(next)
	return nil
}

// ActivateBoundEngineRequest authorizes only a bound cold start. It cannot
// stop an engine, change settings, load a model, or forward to another node.
type ActivateBoundEngineRequest struct {
	OperationID        string    `json:"operationId"`
	Engine             string    `json:"engine"`
	Model              string    `json:"model"`
	PolicyRevision     string    `json:"policyRevision"`
	ManagerIncarnation string    `json:"managerIncarnation"`
	Deadline           time.Time `json:"deadline"`
	PeerID             string    `json:"peerId,omitempty"`
	PeerIncarnation    string    `json:"peerIncarnation,omitempty"`
}

func (p ActivateBoundEngineRequest) Validate() error {
	if !strictID.MatchString(p.OperationID) || !strictID.MatchString(p.ManagerIncarnation) {
		return errors.New("operation and manager identities are required")
	}
	if !engines.ValidID(p.Engine) {
		return errors.New("invalid engine")
	}
	if err := checkModelRef(p.Engine, p.Model, 0, nil); err != nil {
		return err
	}
	if err := validateRevision(p.PolicyRevision); err != nil {
		return err
	}
	if p.Deadline.IsZero() {
		return errors.New("deadline is required")
	}
	return boundEnginePeer(p.PeerID, p.PeerIncarnation)
}

func (p *ActivateBoundEngineRequest) UnmarshalJSON(data []byte) error {
	type plain ActivateBoundEngineRequest
	var next plain
	if err := decodeResident(data, &next, []string{"operationId", "engine", "model", "policyRevision", "managerIncarnation", "deadline"}, "peerId", "peerIncarnation"); err != nil {
		return err
	}
	if err := boundEnginePeerFields(data, next.PeerID, next.PeerIncarnation); err != nil {
		return err
	}
	if err := ActivateBoundEngineRequest(next).Validate(); err != nil {
		return err
	}
	*p = ActivateBoundEngineRequest(next)
	return nil
}
