// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"

	"nvpair-shared/nodepolicy"
)

// strictDestinationAdmission is A1's process-wide admission seam, shared by all
// facades and paired ingress. ingress.go:serveIngress and http_handler.go:handleHTTP's
// self dispatch will call it after trusted-contract parsing and model
// normalization, before wake/unload or inference. A0 provides no implementation.
// Identity, availability, finite capacity and current qualification must all be
// revalidated atomically. Deadlines are monotonic, derived from remaining
// durations, and may only shrink. Concrete requests keep their existing path.
type strictDestinationAdmission interface {
	AdmitStrict(context.Context, string, string, nodepolicy.StrictAdmissionRequest) (strictAdmissionTicket, error)
	StrictReadiness() nodepolicy.StrictAdmissionReadiness
}

// strictAdmissionTicket distinguishes cancellation from proved termination.
// BeforeStrictDispatch must durably mark the engine dirty before any inference
// bytes are sent; failure prevents dispatch. Finish is idempotent. Uncertain
// work retains capacity across cancellation/restart until authoritative engine
// reconciliation; elapsed time alone never releases it.
type strictAdmissionTicket interface {
	BeforeStrictDispatch() error
	Finish(strictTerminalEvidence)
}

type strictTerminalEvidence string

const (
	strictNotDispatched       strictTerminalEvidence = "not-dispatched"
	strictQualifiedComplete   strictTerminalEvidence = "qualified-complete"
	strictAuthoritativeIdle   strictTerminalEvidence = "authoritative-idle"
	strictVerifiedReplacement strictTerminalEvidence = "verified-replacement"
	strictUncertain           strictTerminalEvidence = "uncertain"
)
