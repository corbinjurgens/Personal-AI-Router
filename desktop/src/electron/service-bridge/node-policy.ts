// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    NodeModelReference,
    EngineBindingOffers,
    NodeAvailabilityResult,
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
    StrictAdmissionCapability,
    NanoUsd,
    DeploymentCapability,
    DeploymentTransport,
    DeploymentPrices,
    QualificationRef,
    RoutingDeployment,
    ModelConfiguration,
    ModelConfigurationEntry
} from '@/shared/types/node-policy'
import { engineManagerName, isValidRoutingEngineId } from '@/shared/utils/engines'
import getErrorString from '@/shared/utils/get-error-string'
import { prettyJsonSource } from '@/shared/utils/json-source'
import { availabilityValue, pauseOriginValue } from './node-availability'
import type { JsonObject, JsonValue } from './json-rpc-client'

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

function validModelIdentifier(value: string): boolean {
    return (
        value.length > 0 &&
        value.length <= 512 &&
        new TextEncoder().encode(value).byteLength <= 512 &&
        Array.from(value).every(character => {
            const code = character.charCodeAt(0)
            return code >= 0x20 && (code < 0x7f || code > 0x9f)
        })
    )
}

/** Optional offer errors disable preparation only; inventory stays independent. */
export function parseAdvertisementEngineBindingOffers(
    value: JsonValue | undefined
): EngineBindingOffers {
    const advertisement = objectValue(value)
    if (!advertisement) return { state: 'invalid' }
    const keys = Object.keys(advertisement).filter(key => /^engineBindingOffers$/iu.test(key))
    if (keys.length === 0) return { state: 'absent' }
    if (keys.length !== 1 || keys[0] !== 'engineBindingOffers') return { state: 'invalid' }
    try {
        const envelope = objectValue(advertisement.engineBindingOffers)
        if (!envelope) return { state: 'invalid' }
        const version = strictInteger(envelope.version, 1, 2147483647)
        if (version !== 1) return { state: 'unsupported', version }
        const body = strictObject(envelope, ['version', 'models'])
        if (!Array.isArray(body.models) || body.models.length === 0 || body.models.length > 256) {
            return { state: 'invalid' }
        }
        const seen = new Set<string>()
        const models = body.models.map(value => {
            const ref = strictObject(value, ['engine', 'model'])
            const engine = ref.engine
            const model = typeof ref.model === 'string' ? ref.model.toWellFormed() : null
            if (
                typeof engine !== 'string' ||
                !isValidRoutingEngineId(engine) ||
                typeof model !== 'string' ||
                !validModelIdentifier(model) ||
                /^(weak|medium|strong)(-api|-local)?$/.test(model) ||
                new TextEncoder().encode(
                    JSON.stringify({ engine, model })
                        .replace(
                            /[<>&]/g,
                            character =>
                                `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`
                        )
                        .replace(/\u2028/g, '\\u2028')
                        .replace(/\u2029/g, '\\u2029')
                ).byteLength > 256 ||
                seen.has(model)
            ) {
                throw new Error('Invalid engine binding offer')
            }
            seen.add(model)
            return { engine, model }
        })
        return { state: 'valid', version: 1, models }
    } catch {
        return { state: 'invalid' }
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
            !isValidRoutingEngineId(engine) ||
            name === null ||
            !validModelIdentifier(name) ||
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

const ROUTING_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
// Shared character for character with nodepolicy's endpointPattern.
const ENDPOINT =
    /^https:\/\/((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?)(:[1-9][0-9]{0,4})?((?:\/[A-Za-z0-9._~-]+)+)$/
const LOCAL_NAME_SUFFIXES = ['localhost', 'local', 'localdomain', 'internal', 'arpa']
const NANO_USD = /^(0|[1-9][0-9]*)$/
const INT64_MAX = 9223372036854775807n
const MAX_TOKEN_BOUND = 1 << 24
const CAPABILITIES: DeploymentCapability[] = ['tools', 'vision', 'structured', 'embeddings']

function routingObject(
    value: JsonValue | undefined,
    required: string[],
    optional: string[] = []
): JsonObject {
    const body = objectValue(value)
    if (
        !body ||
        Object.keys(body).some(key => !required.includes(key) && !optional.includes(key)) ||
        required.some(key => body[key] === undefined)
    ) {
        throw new Error('Invalid routing object fields')
    }
    return body
}

function routingInteger(value: JsonValue | undefined, min: number, max: number): number {
    if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
        throw new Error('Invalid routing integer')
    }
    return value
}

function routingIdentifier(value: JsonValue | undefined): string {
    if (typeof value !== 'string' || !ROUTING_ID.test(value)) throw new Error('Invalid routing id')
    return value
}

/** Canonical nano-USD digits within the signed 64-bit range, kept as a string. */
function nanoUsd(value: JsonValue | undefined): NanoUsd {
    if (typeof value !== 'string' || !NANO_USD.test(value) || BigInt(value) > INT64_MAX) {
        throw new Error('Invalid nano-USD amount')
    }
    return value
}

function endpointValue(value: JsonValue | undefined): string {
    const match = typeof value === 'string' && value.length <= 2048 ? ENDPOINT.exec(value) : null
    if (!match) throw new Error('Invalid deployment endpoint')
    const [, host, port, path] = match
    const labels = host.split('.')
    if (
        host.length > 253 ||
        LOCAL_NAME_SUFFIXES.includes(labels[labels.length - 1]) ||
        port === ':443' ||
        (port !== undefined && Number(port.slice(1)) > 65535) ||
        path
            .slice(1)
            .split('/')
            .some(segment => segment === '.' || segment === '..')
    ) {
        throw new Error('Invalid deployment endpoint')
    }
    return match[0]
}

function deploymentTransport(value: JsonValue | undefined): DeploymentTransport {
    const body = routingObject(value, [
        'connectTimeoutSeconds',
        'headerTimeoutSeconds',
        'idleTimeoutSeconds',
        'maxResponseBytes',
        'maxErrorBodyBytes',
        'maxEventBytes'
    ])
    const maxResponseBytes = routingInteger(body.maxResponseBytes, 1, 64 * 1024 * 1024)
    return {
        connectTimeoutSeconds: routingInteger(body.connectTimeoutSeconds, 1, 60),
        headerTimeoutSeconds: routingInteger(body.headerTimeoutSeconds, 1, 600),
        idleTimeoutSeconds: routingInteger(body.idleTimeoutSeconds, 1, 600),
        maxResponseBytes,
        maxErrorBodyBytes: routingInteger(body.maxErrorBodyBytes, 1, 1024 * 1024),
        maxEventBytes: routingInteger(body.maxEventBytes, 1, maxResponseBytes)
    }
}

function deploymentPrices(value: JsonValue | undefined): DeploymentPrices {
    const body = routingObject(value, ['currency', 'unit', 'inputPerMillion', 'outputPerMillion'])
    if (body.currency !== 'USD' || body.unit !== 'nanoUSDPerMillionTokens') {
        throw new Error('Deployment prices must be USD nano-USD per million tokens')
    }
    return {
        currency: 'USD',
        unit: 'nanoUSDPerMillionTokens',
        inputPerMillion: nanoUsd(body.inputPerMillion),
        outputPerMillion: nanoUsd(body.outputPerMillion)
    }
}

function qualificationRef(value: JsonValue | undefined): QualificationRef {
    const body = routingObject(value, ['id', 'revision'])
    const revision = typeof body.revision === 'string' ? body.revision : ''
    if (!NANO_USD.test(revision) || revision === '0' || BigInt(revision) > INT64_MAX) {
        throw new Error('Invalid qualification revision')
    }
    return { id: strictString(body.id, STRICT_ID), revision }
}

function routingDeployment(value: JsonValue | undefined, accounts: Set<string>): RoutingDeployment {
    const body = routingObject(
        value,
        ['id', 'account', 'model', 'family', 'contextTokens', 'maxOutputTokens', 'enabled'],
        ['capabilities', 'prices', 'endpoint', 'protocol', 'transport', 'qualification']
    )
    const account = routingIdentifier(body.account)
    if (!accounts.has(account)) throw new Error('Deployment names an undefined account')
    // Match encoding/json's replacement of unpaired UTF-16 surrogates.
    const model = typeof body.model === 'string' ? body.model.toWellFormed() : ''
    if (model === '' || /^(weak|medium|strong)(-api|-local)?$/.exec(model)?.[0] === model) {
        throw new Error('Invalid deployment model')
    }
    const capabilities = body.capabilities === undefined ? [] : body.capabilities
    if (!Array.isArray(capabilities)) throw new Error('Invalid deployment capabilities')
    const contextTokens = routingInteger(body.contextTokens, 1, MAX_TOKEN_BOUND)
    if (body.protocol !== undefined && body.protocol !== 'openai-chat-completions') {
        throw new Error('Unsupported deployment protocol')
    }
    if (typeof body.enabled !== 'boolean') throw new Error('Invalid deployment enabled flag')
    const deployment: RoutingDeployment = {
        id: routingIdentifier(body.id),
        account,
        model,
        family: routingIdentifier(body.family),
        capabilities: capabilities.map(capability => {
            const known = CAPABILITIES.find(c => c === capability)
            if (!known) throw new Error('Unknown deployment capability')
            return known
        }),
        contextTokens,
        maxOutputTokens: routingInteger(body.maxOutputTokens, 1, contextTokens),
        prices: body.prices === undefined ? null : deploymentPrices(body.prices),
        endpoint: body.endpoint === undefined ? null : endpointValue(body.endpoint),
        protocol: body.protocol === undefined ? null : 'openai-chat-completions',
        transport: body.transport === undefined ? null : deploymentTransport(body.transport),
        qualification:
            body.qualification === undefined ? null : qualificationRef(body.qualification),
        enabled: body.enabled
    }
    if (
        deployment.enabled &&
        (!deployment.endpoint ||
            !deployment.protocol ||
            !deployment.transport ||
            !deployment.qualification ||
            !deployment.prices)
    ) {
        throw new Error(
            'An enabled deployment needs an endpoint, protocol, transport, qualification and prices'
        )
    }
    return deployment
}

/**
 * Parse the deployments of a policy's origin-only `routing` section as the
 * broker returns it (canonical keys, no unknown fields). Money stays exact
 * strings; this grants no permission and does not decide whether remote
 * execution is available.
 */
export function parseRoutingDeployments(routing: JsonValue | undefined): RoutingDeployment[] {
    const body = objectValue(routing)
    if (!body || !Array.isArray(body.accounts) || !Array.isArray(body.deployments)) {
        throw new Error('Invalid routing section')
    }
    const accounts = new Set(
        body.accounts.map(account => routingIdentifier(objectValue(account)?.id))
    )
    const seen = new Set<string>()
    return body.deployments.map(value => {
        const deployment = routingDeployment(value, accounts)
        if (seen.has(deployment.id)) throw new Error('Duplicate deployment id')
        seen.add(deployment.id)
        return deployment
    })
}

/** Broker params that name a node; an omitted id means this PC. */
export function nodeTargetParams(target: NodePolicyTarget): JsonObject {
    return target.nodeId ? { nodeId: target.nodeId } : {}
}

/** `workloads:cancel` params. The broker spells the engine the way engine-manager does. */
export function workloadCancelParams(request: WorkloadCancelRequest): JsonObject {
    if (!request.originatedFrom || !request.engine || !request.runId || !request.workloadId) {
        throw new Error('Complete workload cancellation target is required')
    }
    const params: JsonObject = {
        originatedFrom: request.originatedFrom,
        workloadId: request.workloadId,
        engine: engineManagerName(request.engine),
        runId: request.runId,
        regenerate: request.regenerate
    }
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

function modelConfigurationEntry(value: JsonValue): ModelConfigurationEntry {
    const body = objectValue(value)
    const fields = ['engine', 'model', 'desiredResident', 'observedLoaded', 'state', 'reason']
    if (!body || Object.keys(body).some(key => !fields.includes(key))) {
        throw new Error('Invalid model configuration entry')
    }
    const { desiredResident, observedLoaded, state, reason } = body
    // Match encoding/json before validation and duplicate-reference checks.
    const engine = typeof body.engine === 'string' ? body.engine.toWellFormed() : null
    const model = typeof body.model === 'string' ? body.model.toWellFormed() : null
    if (
        typeof engine !== 'string' ||
        !isValidRoutingEngineId(engine) ||
        typeof model !== 'string' ||
        model === '' ||
        /^\p{White_Space}|\p{White_Space}$/u.test(model) ||
        new TextEncoder().encode(model).length > 1024 ||
        /^(weak|medium|strong)(-api|-local)?$/.exec(model)?.[0] === model ||
        typeof desiredResident !== 'boolean' ||
        (observedLoaded !== undefined && typeof observedLoaded !== 'boolean') ||
        (state !== 'satisfied' &&
            state !== 'pending' &&
            state !== 'blocked' &&
            state !== 'unknown') ||
        (reason !== undefined &&
            reason !== '' &&
            reason !== 'engine-off' &&
            reason !== 'no-fit' &&
            reason !== 'paused' &&
            reason !== 'engine-locked' &&
            reason !== 'paused-local' &&
            reason !== 'engine-off-local' &&
            reason !== 'not-served' &&
            reason !== 'not-installed' &&
            reason !== 'unqualified' &&
            reason !== 'unsupported-load' &&
            reason !== 'switch-busy' &&
            reason !== 'load-failed' &&
            reason !== 'operation-timeout' &&
            reason !== 'observation-unknown' &&
            reason !== 'projection-unknown') ||
        (state === 'satisfied' &&
            (!desiredResident ||
                observedLoaded !== true ||
                (reason !== undefined && reason !== '')))
    ) {
        throw new Error('Invalid model configuration entry')
    }
    return {
        engine,
        model,
        desiredResident,
        state,
        ...(observedLoaded !== undefined ? { observedLoaded } : {}),
        ...(reason !== undefined ? { reason } : {})
    }
}

function modelConfiguration(value: JsonValue | undefined): ModelConfiguration | undefined {
    if (value === undefined) return undefined
    const body = strictObject(value, ['policyRevision', 'observationGeneration', 'models'])
    const policyRevision = revisionValue(body.policyRevision)
    const { observationGeneration } = body
    if (
        policyRevision === null ||
        BigInt(policyRevision) > 9223372036854775807n ||
        // Go carries a uint64, but the broker starts it at zero in memory and adds one
        // per completed observation pass, so it never reaches 2^53. JSON.parse has
        // already rounded anything larger, so only safe integers are exact; reject the rest.
        typeof observationGeneration !== 'number' ||
        !Number.isSafeInteger(observationGeneration) ||
        observationGeneration < 0 ||
        !Array.isArray(body.models) ||
        body.models.length > 256
    ) {
        throw new Error('Invalid model configuration')
    }
    const models = body.models.map(modelConfigurationEntry)
    const references = models.map(entry => JSON.stringify([entry.engine, entry.model]))
    if (new Set(references).size !== references.length) {
        throw new Error('Duplicate model configuration reference')
    }
    return { policyRevision, observationGeneration, models }
}

function policySnapshot(body: JsonObject | null): NodePolicySnapshot {
    if (!body || typeof body.policy !== 'string') {
        throw new Error('The service returned no node policy')
    }
    const policy = parsePolicyText(body.policy)
    const revision = revisionValue(policy.revision)
    if (revision === null) throw new Error('The service returned a node policy without a revision')
    const configuration = modelConfiguration(body.modelConfiguration)
    return {
        policy: prettyJsonSource(body.policy),
        revision,
        activation: activationStatus(body),
        ...(configuration !== undefined ? { modelConfiguration: configuration } : {})
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
        ...parseAvailabilityResult(result),
        recoveryRequired: body?.recoveryRequired === true
    }
}

/** Parse a `policy:set` result: the policy as persisted, after defaults were applied. */
export function parsePolicySetResult(result: JsonValue | undefined): NodePolicySnapshot {
    return policySnapshot(objectValue(result))
}

/** Parse a `node:set-availability` result. */
export function parseAvailabilityResult(result: JsonValue | undefined): NodeAvailabilityResult {
    const body = objectValue(result)
    const availability = availabilityValue(body?.availability)
    if (!availability) throw new Error('The service returned no node availability')
    const pausedBy = body?.pausedBy
    if (pausedBy !== undefined && pausedBy !== '' && pausedBy !== 'local' && pausedBy !== 'peer') {
        throw new Error('The service returned invalid pause provenance')
    }
    const origin = availability !== 'available' ? pauseOriginValue(pausedBy) : undefined
    return {
        availability,
        ...(origin !== undefined ? { pausedBy: origin } : {})
    }
}

/** Parse a `node:availability-changed {nodeId, availability, active}` push; null if malformed. */
export function parseAvailabilityChange(
    params: JsonValue | undefined
): NodeAvailabilityChange | null {
    const body = objectValue(params)
    const availability = availabilityValue(body?.availability)
    if (!body || typeof body.nodeId !== 'string' || body.nodeId === '' || !availability) return null
    try {
        return { nodeId: body.nodeId, ...parseAvailabilityResult(params) }
    } catch {
        return null
    }
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
 * rejects malformed presence-sensitive model preferences at this boundary too.
 * Go remains authoritative for the complete policy and raw duplicate keys.
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
    validateModelPreferences(policy)
    return policy
}

function modelReferences(
    value: JsonValue | undefined,
    binding = false
): NodeModelReference[] | undefined {
    if (value === undefined) return undefined
    if (!Array.isArray(value) || value.length > 256) throw new Error('Invalid model list')
    const seen = new Set<string>()
    return value.map(entry => {
        const ref = strictObject(entry, ['engine', 'model'])
        const engine = typeof ref.engine === 'string' ? ref.engine.toWellFormed() : ''
        const model = typeof ref.model === 'string' ? ref.model.toWellFormed() : ''
        if (
            !isValidRoutingEngineId(engine) ||
            model === '' ||
            (binding &&
                (Array.from(model).some(character => {
                    const code = character.charCodeAt(0)
                    return code < 0x20 || (code >= 0x7f && code <= 0x9f)
                }) ||
                    new TextEncoder().encode(model).length > 512 ||
                    new TextEncoder().encode(
                        JSON.stringify({ engine, model })
                            .replace(/[<>&]/g, character => {
                                return `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`
                            })
                            .replace(/\u2028/g, '\\u2028')
                            .replace(/\u2029/g, '\\u2029')
                    ).length > 256)) ||
            /^(weak|medium|strong)(-api|-local)?$/.exec(model)?.[0] === model
        ) {
            throw new Error('Invalid model reference')
        }
        const key = JSON.stringify(binding ? model : [engine, model])
        if (seen.has(key)) throw new Error('Duplicate model reference')
        seen.add(key)
        return { engine, model }
    })
}

// JSON.parse collapses exact duplicate keys; Go validates raw duplicate keys.
// Validate decoded SR fields without reconstructing the policy or losing [].
function validateModelPreferences(policy: JsonObject): void {
    for (const key of Object.keys(policy)) {
        for (const name of ['servedModels', 'residentModels', 'engineBindings']) {
            if (key !== name && new RegExp(`^${name}$`, 'iu').test(key)) {
                throw new Error(`${key} must use the exact field name ${name}`)
            }
        }
    }
    const served = modelReferences(policy.servedModels)
    const residents = modelReferences(policy.residentModels)
    // Descriptor-dependent overlap and served permission belong to the target.
    modelReferences(policy.engineBindings, true)
    for (const ref of residents ?? []) {
        if (
            served !== undefined &&
            !served.some(s => s.engine === ref.engine && s.model === ref.model)
        ) {
            throw new Error('residentModels must be a subset of servedModels')
        }
        const profile = Array.isArray(policy.profiles)
            ? policy.profiles
                  .map(objectValue)
                  .find(p => p?.engine === ref.engine && p.model === ref.model)
            : undefined
        if (!profile) throw new Error('Resident model requires a qualified profile')
        for (const field of ['maxConcurrent', 'memoryBytes']) {
            routingInteger(profile[field], 1, Number.MAX_SAFE_INTEGER)
        }
        // Mirror Go's validateOutputAndQualification: an output ceiling must be
        // positive and below a nonzero contextTokens.
        const contextTokens = routingInteger(profile.contextTokens, 1, Number.MAX_SAFE_INTEGER)
        const maxOutputTokens = routingInteger(profile.maxOutputTokens, 1, Number.MAX_SAFE_INTEGER)
        if (maxOutputTokens >= contextTokens) {
            throw new Error('maxOutputTokens must be below a nonzero contextTokens')
        }
        qualificationRef(profile.qualification)
    }
}

/** `policy:set` params: the edited policy, checked against the revision it was edited from. */
export function policySetParams(request: NodePolicyWrite): JsonObject {
    if (revisionValue(request.expectedPersistedRevision) === null) {
        throw new Error('The policy revision it was edited from must be a decimal string')
    }
    parsePolicyText(request.policy)
    return {
        ...nodeTargetParams(request),
        policy: request.policy,
        expectedPersistedRevision: request.expectedPersistedRevision
    }
}
