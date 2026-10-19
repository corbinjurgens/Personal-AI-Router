// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "nvpair-shared/nodepolicy"

// notifyStrictReadiness is the proxy -> broker notification that carries the
// proxy controller's nodepolicy.StrictAdmissionReadiness (A1a -> A1b). The
// proxy sends it after acknowledging each node/set-policy, naming the revision
// it acknowledged and the run id of that PolicyAck, and again whenever its
// readiness for that revision changes. An omitted capability means not ready.
// Order against the acknowledgement does not matter: the broker holds a report
// until its fences match what it has recorded. A report from another proxy run
// than the acknowledged one is discarded, and one that does not validate
// withdraws whatever was held. The proxy needs no other message: the broker
// clears readiness itself on restart, exit, pending activation and pause.
const notifyStrictReadiness = "admission/strict-readiness"

// strictReadinessSource is the A1 -> A2 seam. A1's process-wide proxy controller
// supplies reconciled readiness, fenced by the active policy revision and proxy
// incarnation. A2 may publish only when both fences match the broker's current
// acknowledgement. Startup, disconnect, pending activation or unknown identity
// must clear readiness. The Broker implements it (readiness.go) from the
// notifyStrictReadiness reports, and the node projection carries its capability.
type strictReadinessSource interface {
	StrictReadiness() nodepolicy.StrictAdmissionReadiness
}

var _ strictReadinessSource = (*Broker)(nil)

// strictAdvertisementSink is A2's inventory transport seam: policy.go's
// acknowledged policy push will supply the whole advertisement to engine-manager
// for its ModelsResult.Policy. A nil capability means withdraw, never retain the
// previous value. Transport must not substitute for destination revalidation.
type strictAdvertisementSink interface {
	PublishPolicyAdvertisement(nodepolicy.Advertisement) error
}
