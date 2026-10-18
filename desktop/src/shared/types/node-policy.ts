// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineType } from '@/shared/types/engines'

/** A node's live availability. `draining` is a pause in progress. */
export type NodeAvailability = 'available' | 'draining' | 'paused'

/** The state a caller can request; `draining` is only ever reached, never asked for. */
export type RequestedAvailability = 'available' | 'paused'

/** Whether a node's proxy enforces its saved policy: `active` once the proxy acknowledged it. */
export type PolicyActivation = 'active' | 'pending' | 'error'

/** A node's report on enforcing its saved policy. A paired node reports none. */
export interface PolicyActivationStatus {
    state: PolicyActivation
    /** The revision the proxy acknowledged, or empty when it has acknowledged none. */
    activeRevision: string
    /** Operational text explaining an `error`. */
    error?: string
}

/**
 * A node's saved policy. The policy travels as JSON text because the editor
 * shows and edits it as text and the service validates it. Revisions are
 * decimal strings: they exceed what a JavaScript number holds exactly.
 */
export interface NodePolicySnapshot {
    policy: string
    /** The persisted revision (`policy.revision`); the base the next save is checked against. */
    revision: string
    /** Whether the proxy enforces this revision; null when the node reports none. */
    activation: PolicyActivationStatus | null
}

/** A node's saved policy and live availability. */
export interface NodePolicyDocument extends NodePolicySnapshot {
    availability: NodeAvailability
    /** The saved file was unreadable and defaults are in use. */
    recoveryRequired: boolean
}

/** Names a node other than this one; omitted for this PC. */
export interface NodePolicyTarget {
    nodeId?: string
}

export interface NodePolicyWrite extends NodePolicyTarget {
    policy: string
    /** The persisted revision the draft was edited from; the service refuses a save over a newer one. */
    expectedPersistedRevision: string
}

/** A node's availability changed; the broker relays every node's changes under its id. */
export interface NodeAvailabilityChange {
    nodeId: string
    availability: NodeAvailability
}

/** A node's saved policy changed, as JSON text like {@link NodePolicySnapshot.policy}. */
export interface NodePolicyChange extends NodePolicySnapshot {
    nodeId: string
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
