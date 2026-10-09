// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineType } from '@/shared/types/engines'

/** A node's live availability. `draining` is a pause in progress. */
export type NodeAvailability = 'available' | 'draining' | 'paused'

/** The state a caller can request; `draining` is only ever reached, never asked for. */
export type RequestedAvailability = 'available' | 'paused'

/**
 * A node's saved policy and live availability. The policy travels as JSON text
 * because the editor shows and edits it as text and the service validates it.
 */
export interface NodePolicyDocument {
    policy: string
    availability: NodeAvailability
}

/** Names a node other than this one; omitted for this PC. */
export interface NodePolicyTarget {
    nodeId?: string
}

export interface NodePolicyWrite extends NodePolicyTarget {
    policy: string
}

/** A node's availability changed; the broker relays every node's changes under its id. */
export interface NodeAvailabilityChange {
    nodeId: string
    availability: NodeAvailability
}

/** A node's saved policy changed, as JSON text like {@link NodePolicyDocument.policy}. */
export interface NodePolicyChange {
    nodeId: string
    policy: string
}

export interface NodeAvailabilityRequest extends NodePolicyTarget {
    state: RequestedAvailability
}

/** Cancels one job; only the node the job originated on acts on it. */
export interface WorkloadCancelRequest {
    originatedFrom: string
    workloadId: string
    engine: EngineType
    runId?: string
    /** Re-dispatch to another device instead of ending the request. */
    regenerate: boolean
}
