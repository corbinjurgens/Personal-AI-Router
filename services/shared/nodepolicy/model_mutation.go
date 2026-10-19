// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nodepolicy

// model_mutation.go is the internal handshake engine-manager uses before it
// changes an engine's installed models. The broker answers only after the
// running proxy acknowledged an identity without that engine (model_identity.go),
// so strict admission cannot keep admitting under a tuple the mutation is about
// to invalidate. Both messages are notifications between the two processes, not
// client requests, and carry operational identity only: no prompt, request
// body, credential or model content.

import (
	"errors"
	"strings"

	"nvpair-shared/engines"
)

// Model mutation notification methods. engine-manager sends the request; the
// broker consumes it on engine-manager's read pump and answers through the
// worker that sent it, never through a later replacement.
const (
	MethodModelMutationRequest = "engine:model-mutation-request"
	MethodModelMutationReply   = "engine:model-mutation-reply"
)

// Failure codes a ModelMutationReply may carry. The error text is the
// operator-facing detail; the code lets the requester tell a stale pairing from
// a proxy that did not confirm, without parsing text.
const (
	// ModelMutationUnavailable: the request did not come from the broker's
	// current bound engine-manager worker, or that pairing ended before the
	// withdrawal could be confirmed.
	ModelMutationUnavailable = iota + 1
	// ModelMutationWithdrawalFailed: the bound worker exists but the proxy did
	// not confirm the withdrawal (busy push slot, refusal, transport failure).
	ModelMutationWithdrawalFailed
)

// ModelMutationRequest asks the broker to withdraw one engine's model identity
// before engine-manager changes that engine's content. RequestID correlates the
// reply. Engine names the whole engine whose identity is withdrawn, not one
// model: every model of it is affected by an install, delete or pull.
type ModelMutationRequest struct {
	RequestID string `json:"requestId"`
	Engine    string `json:"engine"`
}

func (p ModelMutationRequest) Validate() error {
	if !strictID.MatchString(p.RequestID) {
		return errors.New("requestId must be a nonempty identity of at most 128 ASCII characters")
	}
	if !engines.ValidID(p.Engine) {
		return errors.New("engine must be a valid engine ID")
	}
	return nil
}

func (p *ModelMutationRequest) UnmarshalJSON(data []byte) error {
	type plain ModelMutationRequest
	var next plain
	if err := decodeStrictContract(data, &next, "requestId", "engine"); err != nil {
		return err
	}
	if err := ModelMutationRequest(next).Validate(); err != nil {
		return err
	}
	*p = ModelMutationRequest(next)
	return nil
}

// ModelMutationReply answers a ModelMutationRequest. OK true means the proxy
// acknowledged an identity without Engine, so the mutation may proceed. OK
// false carries Error and, when the broker could classify it, Code; a success
// carries both explicitly empty/zero. A reply without OK is malformed rather
// than a refusal.
type ModelMutationReply struct {
	RequestID string `json:"requestId"`
	Engine    string `json:"engine"`
	OK        bool   `json:"ok"`
	Error     string `json:"error"`
	Code      int    `json:"code"`
}

func (p ModelMutationReply) Validate() error {
	if !strictID.MatchString(p.RequestID) {
		return errors.New("requestId must be a nonempty identity of at most 128 ASCII characters")
	}
	if !engines.ValidID(p.Engine) {
		return errors.New("engine must be a valid engine ID")
	}
	if p.OK {
		if p.Error != "" || p.Code != 0 {
			return errors.New("a successful reply carries no error or code")
		}
		return nil
	}
	if strings.TrimSpace(p.Error) == "" {
		return errors.New("a failed reply must carry an error")
	}
	return nil
}

func (p *ModelMutationReply) UnmarshalJSON(data []byte) error {
	type plain ModelMutationReply
	var next plain
	if err := decodeStrictContract(data, &next, "requestId", "engine", "ok", "error", "code"); err != nil {
		return err
	}
	if err := ModelMutationReply(next).Validate(); err != nil {
		return err
	}
	*p = ModelMutationReply(next)
	return nil
}
