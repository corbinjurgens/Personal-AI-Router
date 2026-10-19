// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import "nvpair-shared/nodepolicy"

// strictReadinessSource is the A1 -> A2 seam. A1's process-wide proxy controller
// supplies reconciled readiness with nodepolicy.NotifyStrictReadiness, fenced by
// the active policy revision and proxy incarnation. The broker publishes it only
// when both fences match its current acknowledgement. Startup, disconnect,
// pending activation or unknown identity must clear readiness. The Broker
// implements it (readiness.go) from those reports, and the node projection
// (engine:node-projection, policy_notifications.go) carries its capability to
// engine-manager's inventory, where a nil capability withdraws the previous one.
// Transport never substitutes for destination revalidation.
type strictReadinessSource interface {
	StrictReadiness() nodepolicy.StrictAdmissionReadiness
}

var _ strictReadinessSource = (*Broker)(nil)
