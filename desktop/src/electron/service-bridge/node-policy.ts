// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    NodeAvailability,
    NodeAvailabilityChange,
    NodePolicyChange,
    NodePolicyDocument,
    NodePolicySnapshot,
    NodePolicyTarget,
    NodePolicyWrite,
    PolicyActivationStatus,
    WorkloadCancelRequest
} from '@/shared/types/node-policy'
import { engineManagerName } from '@/shared/utils/engines'
import getErrorString from '@/shared/utils/get-error-string'
import { availabilityValue } from './node-availability'
import type { JsonObject, JsonValue } from './json-rpc-client'

function objectValue(value: JsonValue | undefined): JsonObject | null {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

/** Broker params that name a node; an omitted id means this PC. */
export function nodeTargetParams(target: NodePolicyTarget): JsonObject {
    return target.nodeId ? { nodeId: target.nodeId } : {}
}

/** `workloads:cancel` params. The broker spells the engine the way engine-manager does. */
export function workloadCancelParams(request: WorkloadCancelRequest): JsonObject {
    const params: JsonObject = {
        originatedFrom: request.originatedFrom,
        workloadId: request.workloadId,
        engine: engineManagerName(request.engine),
        regenerate: request.regenerate
    }
    if (request.runId) params.runId = request.runId
    return params
}

/** Parse the `{ok}` result of `workloads:cancel`. */
export function parseCancelResult(result: JsonValue | undefined): { ok: boolean } {
    return { ok: objectValue(result)?.ok === true }
}

// A revision is "0" or a decimal without leading zeros. It stays a string: it
// exceeds what a JavaScript number holds exactly.
const REVISION_PATTERN = /^(0|[1-9][0-9]*)$/

function revisionValue(value: JsonValue | undefined): string | null {
    return typeof value === 'string' && REVISION_PATTERN.test(value) ? value : null
}

/**
 * The enforcement report that accompanies a policy. A paired node's answers carry
 * none, which is null; a report that is present must be well formed.
 */
function activationStatus(body: JsonObject): PolicyActivationStatus | null {
    const { activation, activeRevision, activationError } = body
    if (activation === undefined) return null
    if (activation !== 'active' && activation !== 'pending' && activation !== 'error') {
        throw new Error('The service returned an unknown policy activation state')
    }
    const active = activeRevision === '' ? '' : revisionValue(activeRevision)
    if (active === null) throw new Error('The service returned an invalid active policy revision')
    if (activationError !== undefined && typeof activationError !== 'string') {
        throw new Error('The service returned an invalid policy activation error')
    }
    return {
        state: activation,
        activeRevision: active,
        ...(activationError ? { error: activationError } : {})
    }
}

function policySnapshot(body: JsonObject | null): NodePolicySnapshot {
    const policy = objectValue(body?.policy)
    if (!body || !policy) throw new Error('The service returned no node policy')
    const revision = revisionValue(policy.revision)
    if (revision === null) throw new Error('The service returned a node policy without a revision')
    return {
        policy: JSON.stringify(policy, null, 2),
        revision,
        activation: activationStatus(body)
    }
}

/** Parse a `policy:get` result into the policy as editable text plus the live availability. */
export function parsePolicyDocument(result: JsonValue | undefined): NodePolicyDocument {
    const body = objectValue(result)
    const availability = availabilityValue(body?.availability)
    if (!availability) throw new Error('The service returned no node availability')
    // A paired node's projection omits the flag.
    if (body?.recoveryRequired !== undefined && typeof body.recoveryRequired !== 'boolean') {
        throw new Error('The service returned an invalid policy recovery flag')
    }
    return {
        ...policySnapshot(body),
        availability,
        recoveryRequired: body?.recoveryRequired === true
    }
}

/** Parse a `policy:set` result: the policy as persisted, after defaults were applied. */
export function parsePolicySetResult(result: JsonValue | undefined): NodePolicySnapshot {
    return policySnapshot(objectValue(result))
}

/** Parse a `node:set-availability` result. */
export function parseAvailabilityResult(result: JsonValue | undefined): {
    availability: NodeAvailability
} {
    const availability = availabilityValue(objectValue(result)?.availability)
    if (!availability) throw new Error('The service returned no node availability')
    return { availability }
}

/** Parse a `node:availability-changed {nodeId, availability, active}` push; null if malformed. */
export function parseAvailabilityChange(
    params: JsonValue | undefined
): NodeAvailabilityChange | null {
    const body = objectValue(params)
    const availability = availabilityValue(body?.availability)
    if (!body || typeof body.nodeId !== 'string' || body.nodeId === '' || !availability) return null
    return { nodeId: body.nodeId, availability }
}

/** Parse a `policy:changed {nodeId, policy, activation...}` push; null if malformed. */
export function parsePolicyChange(params: JsonValue | undefined): NodePolicyChange | null {
    const body = objectValue(params)
    if (!body || typeof body.nodeId !== 'string' || body.nodeId === '') return null
    try {
        return { nodeId: body.nodeId, ...policySnapshot(body) }
    } catch {
        return null
    }
}

/**
 * Read the policy the editor sends back. The broker validates the content; this
 * only rejects text that is not a JSON object, so the user sees a parse error
 * rather than a vague rejection.
 */
export function parsePolicyText(text: string): JsonObject {
    let parsed: JsonValue
    try {
        parsed = JSON.parse(text)
    } catch (err) {
        throw new Error(`The policy is not valid JSON: ${getErrorString(err)}`)
    }
    const policy = objectValue(parsed)
    if (!policy) throw new Error('The policy must be a JSON object')
    return policy
}

/** `policy:set` params: the edited policy, checked against the revision it was edited from. */
export function policySetParams(request: NodePolicyWrite): JsonObject {
    if (revisionValue(request.expectedPersistedRevision) === null) {
        throw new Error('The policy revision it was edited from must be a decimal string')
    }
    return {
        ...nodeTargetParams(request),
        policy: parsePolicyText(request.policy),
        expectedPersistedRevision: request.expectedPersistedRevision
    }
}
