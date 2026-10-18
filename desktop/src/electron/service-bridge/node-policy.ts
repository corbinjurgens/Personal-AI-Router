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
    WorkloadCancelRequest,
    RoutingRequestKey,
    StrictModelIdentity,
    StrictAdmissionRequest,
    StrictAdmissionModel,
    StrictAdmissionCapability
} from '@/shared/types/node-policy'
import { engineManagerName } from '@/shared/utils/engines'
import getErrorString from '@/shared/utils/get-error-string'
import { availabilityValue } from './node-availability'
import type { JsonObject, JsonValue } from './json-rpc-client'
import { EngineManagerNames } from '@/shared/constants/engines'

function objectValue(value: JsonValue | undefined): JsonObject | null {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return null
    return value
}

const STRICT_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/
const STRICT_DIGEST = /^[0-9a-f]{64}$/

function strictObject(value: JsonValue | undefined, fields: string[]): JsonObject {
    const body = objectValue(value)
    if (
        !body ||
        Object.keys(body).some(key => !fields.includes(key)) ||
        fields.some(key => body[key] === undefined || body[key] === null)
    ) {
        throw new Error('Invalid strict-admission object fields')
    }
    return body
}

function strictString(value: JsonValue | undefined, pattern: RegExp): string {
    const normalized = typeof value === 'string' ? value.toWellFormed() : null
    if (normalized === null || normalized.match(pattern)?.[0] !== normalized) {
        throw new Error('Invalid strict-admission identity')
    }
    return normalized
}

function strictInteger(value: JsonValue | undefined, min: number, max: number): number {
    if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
        throw new Error('Invalid strict-admission integer')
    }
    return value
}

function strictModelIdentity(value: JsonValue | undefined): StrictModelIdentity {
    const body = strictObject(value, ['engineRunId', 'settingsFingerprint', 'modelDigest'])
    return {
        engineRunId: strictString(body.engineRunId, STRICT_ID),
        settingsFingerprint: strictString(body.settingsFingerprint, STRICT_DIGEST),
        modelDigest: strictString(body.modelDigest, STRICT_DIGEST)
    }
}

function routingRequestKey(value: JsonValue | undefined): RoutingRequestKey {
    const body = strictObject(value, ['originNodeId', 'proxyRunId', 'requestId'])
    const requestId = strictString(body.requestId, /^[1-9][0-9]*$/)
    if (BigInt(requestId) > 9223372036854775807n) throw new Error('Invalid routing requestId')
    return {
        originNodeId: strictString(body.originNodeId, STRICT_ID),
        proxyRunId: strictString(body.proxyRunId, STRICT_ID),
        requestId
    }
}

/** A0 contract parser only; it grants no dispatch permission. */
export function parseStrictAdmissionRequest(value: JsonValue | undefined): StrictAdmissionRequest {
    const body = strictObject(value, [
        'version',
        'queueBudgetMs',
        'coldStartBudgetMs',
        'preOutputBudgetMs',
        'request',
        'identity'
    ])
    if (body.version !== 1) throw new Error('Unsupported strict-admission version')
    const preOutputBudgetMs = strictInteger(body.preOutputBudgetMs, 1, 600000)
    return {
        version: 1,
        queueBudgetMs: strictInteger(body.queueBudgetMs, 0, preOutputBudgetMs),
        coldStartBudgetMs: strictInteger(body.coldStartBudgetMs, 0, preOutputBudgetMs),
        preOutputBudgetMs,
        request: routingRequestKey(body.request),
        identity: strictModelIdentity(body.identity)
    }
}

/** Decode only the inventory policy's capability. Future versions are excluded. */
export function parseStrictAdmissionCapability(
    value: JsonValue | undefined
): StrictAdmissionCapability {
    if (value === undefined) return { state: 'absent' }
    const envelope = objectValue(value)
    if (!envelope) throw new Error('Invalid strict-admission capability')
    const version = strictInteger(envelope.version, 1, 2147483647)
    if (version !== 1) return { state: 'unsupported', version }
    const body = strictObject(value, ['version', 'models'])
    if (!Array.isArray(body.models) || body.models.length < 1 || body.models.length > 1024) {
        throw new Error('Invalid strict-admission models')
    }
    const seen = new Set<string>()
    const models: StrictAdmissionModel[] = body.models.map(value => {
        const model = strictObject(value, ['engine', 'model', 'maxConcurrent', 'identity'])
        // Match encoding/json's replacement of unpaired UTF-16 surrogates before
        // validation, duplicate checks and downstream identity use.
        const engine = typeof model.engine === 'string' ? model.engine.toWellFormed() : null
        const name = typeof model.model === 'string' ? model.model.toWellFormed() : null
        if (
            engine === null ||
            !Object.values(EngineManagerNames).some(value => value === engine) ||
            name === null ||
            name === '' ||
            /^(weak|medium|strong)(-api|-local)?$/.exec(name)?.[0] === name
        ) {
            throw new Error('Invalid strict-admission engine/model')
        }
        const key = JSON.stringify([engine, name])
        if (seen.has(key)) throw new Error('Duplicate strict-admission engine/model')
        seen.add(key)
        return {
            engine,
            model: name,
            maxConcurrent: strictInteger(model.maxConcurrent, 1, 1048576),
            identity: strictModelIdentity(model.identity)
        }
    })
    return { state: 'valid', version: 1, models }
}

/** Extract the A0 contract at the Advertisement boundary; legacy fields stay opaque. */
export function parseAdvertisementStrictAdmission(
    value: JsonValue | undefined
): StrictAdmissionCapability {
    const body = objectValue(value)
    if (!body) throw new Error('Invalid policy advertisement')
    for (const key of Object.keys(body)) {
        // Unicode case folding also covers the long s accepted by Go EqualFold.
        if (key !== 'strictAdmission' && /^strictAdmission$/iu.exec(key)?.[0] === key) {
            throw new Error(`${key} must use the exact field name strictAdmission`)
        }
    }
    if (body.strictAdmission === null) throw new Error('strictAdmission must not be null')
    return parseStrictAdmissionCapability(body.strictAdmission)
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
