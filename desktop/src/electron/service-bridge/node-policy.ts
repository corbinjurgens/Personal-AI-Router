// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    NodeAvailability,
    NodePolicyDocument,
    NodePolicyTarget,
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

function policyText(value: JsonValue | undefined): string {
    if (objectValue(value) === null) throw new Error('The service returned no node policy')
    return JSON.stringify(value, null, 2)
}

/** Parse a `policy:get` result into the policy as editable text plus the live availability. */
export function parsePolicyDocument(result: JsonValue | undefined): NodePolicyDocument {
    const body = objectValue(result)
    const availability = availabilityValue(body?.availability)
    if (!availability) throw new Error('The service returned no node availability')
    return { policy: policyText(body?.policy), availability }
}

/** Parse a `policy:set` result: the policy as persisted, after defaults were applied. */
export function parsePolicySetResult(result: JsonValue | undefined): { policy: string } {
    return { policy: policyText(objectValue(result)?.policy) }
}

/** Parse a `node:set-availability` result. */
export function parseAvailabilityResult(result: JsonValue | undefined): {
    availability: NodeAvailability
} {
    const availability = availabilityValue(objectValue(result)?.availability)
    if (!availability) throw new Error('The service returned no node availability')
    return { availability }
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
