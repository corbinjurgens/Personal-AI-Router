// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RoutingRequestKey } from '@/shared/types/node-policy'
import type {
    RoutingActiveParams,
    RoutingActiveResult,
    RoutingCancelParams,
    RoutingCancelResult
} from '@/shared/types/routing'
import type { JsonObject, JsonValue } from './json-rpc-client'

function strictObject(value: JsonValue | undefined, fields: string[]): JsonObject {
    if (
        !value ||
        typeof value !== 'object' ||
        Array.isArray(value) ||
        Object.keys(value).some(key => !fields.includes(key)) ||
        fields.some(key => value[key] === undefined || value[key] === null)
    ) {
        throw new Error('Invalid routing control fields')
    }
    return value
}

function identity(value: JsonValue | undefined): string {
    if (
        typeof value !== 'string' ||
        /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.exec(value)?.[0] !== value
    ) {
        throw new Error('Invalid routing identity')
    }
    return value
}

function requestKey(value: JsonValue | undefined): RoutingRequestKey {
    const body = strictObject(value, ['originNodeId', 'proxyRunId', 'requestId'])
    const requestId = body.requestId
    if (
        typeof requestId !== 'string' ||
        /^[1-9][0-9]*$/.exec(requestId)?.[0] !== requestId ||
        requestId.length > 19 ||
        BigInt(requestId) > 9223372036854775807n
    ) {
        throw new Error('Invalid routing requestId')
    }
    return {
        originNodeId: identity(body.originNodeId),
        proxyRunId: identity(body.proxyRunId),
        requestId
    }
}

/** Validate decoded params before the broker call, retaining unexpected fields for rejection. */
export function routingCancelParams(value: RoutingCancelParams | undefined): JsonObject {
    const body = strictObject(
        value ? { ...value, request: value.request ? { ...value.request } : null } : undefined,
        ['request', 'regenerate']
    )
    const request = requestKey(body.request)
    if (typeof body.regenerate !== 'boolean') throw new Error('Invalid routing regenerate flag')
    return { request: { ...request }, regenerate: body.regenerate }
}

export function routingActiveParams(value: RoutingActiveParams | undefined): JsonObject {
    strictObject(value, [])
    return {}
}

export function parseRoutingCancelResult(value: JsonValue | undefined): RoutingCancelResult {
    const { status } = strictObject(value, ['status'])
    if (
        status !== 'not_found' &&
        status !== 'conflict_committed' &&
        status !== 'regenerating' &&
        status !== 'cancelled'
    ) {
        throw new Error('Invalid routing cancellation status')
    }
    return { status }
}

export function parseRoutingActiveResult(value: JsonValue | undefined): RoutingActiveResult {
    const { requests } = strictObject(value, ['requests'])
    if (!Array.isArray(requests)) throw new Error('Invalid routing active requests')
    return { requests: requests.map(requestKey) }
}
