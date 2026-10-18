// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "nvpair-shared/nodepolicy"

// strictReadinessSource is the A1 -> A2 seam. A1's process-wide proxy controller
// supplies reconciled readiness, fenced by the active policy revision and proxy
// incarnation. A2 may publish only when both fences match the broker's current
// acknowledgement. Startup, disconnect, pending activation or unknown identity
// must clear readiness. There is deliberately no implementation or RPC in A0.
type strictReadinessSource interface {
	StrictReadiness() nodepolicy.StrictAdmissionReadiness
}

// strictAdvertisementSink is A2's inventory transport seam: policy.go's
// acknowledged policy push will supply the whole advertisement to engine-manager
// for its ModelsResult.Policy. A nil capability means withdraw, never retain the
// previous value. Transport must not substitute for destination revalidation.
type strictAdvertisementSink interface {
	PublishPolicyAdvertisement(nodepolicy.Advertisement) error
}
