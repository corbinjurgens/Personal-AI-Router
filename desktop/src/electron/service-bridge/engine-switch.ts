// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    SwitchRequest,
    SwitchReceipt,
    SwitchStatus,
    SwitchChanged,
    SwitchEngineState,
    SwitchPhase,
    SwitchError,
    SwitchErrorKind
} from '@/shared/types/engine-switch'
import type { JsonObject, JsonValue } from './json-rpc-client'

function invalid(): never {
    throw new Error('Invalid engine switch response.')
}
function object(value: JsonValue | undefined, keys: string[]): JsonObject {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return invalid()
    if (Object.keys(value).some(key => !keys.includes(key))) return invalid()
    return value
}
function text(value: JsonValue | undefined): string {
    return typeof value === 'string' && value.length > 0 ? value : invalid()
}
function flag(value: JsonValue | undefined): boolean {
    return typeof value === 'boolean' ? value : invalid()
}
function sequence(value: JsonValue | undefined): number {
    return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : invalid()
}
function matching(value: JsonValue | undefined, pattern: RegExp): string {
    const result = text(value)
    return pattern.test(result) ? result : invalid()
}
const identity = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/
const revision = /^e3v1:[0-9a-f]{64}$/
function timestamp(value: JsonValue | undefined): string {
    const result = text(value)
    return /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?(?:Z|[+-]\d\d:\d\d)$/.test(result) &&
        Number.isFinite(Date.parse(result)) &&
        Date.parse(result) !== -62135596800000
        ? result
        : invalid()
}
function state(value: JsonValue | undefined): SwitchEngineState {
    const row = object(value, ['intent', 'running', 'healthy', 'observedAt', 'terminationVerified'])
    const intent = row.intent
    if (intent !== 'on' && intent !== 'off' && intent !== 'unknown') return invalid()
    const result: SwitchEngineState = {
        intent,
        running: row.running === null ? null : flag(row.running),
        healthy: row.healthy === null ? null : flag(row.healthy),
        observedAt: row.observedAt === null ? null : timestamp(row.observedAt),
        terminationVerified: flag(row.terminationVerified)
    }
    if (result.terminationVerified && (result.running !== false || result.observedAt === null))
        return invalid()
    return result
}
function phase(value: JsonValue | undefined): SwitchPhase {
    switch (value) {
        case 'accepted':
        case 'validating':
        case 'draining':
        case 'source-stopping':
        case 'source-off':
        case 'target-starting':
        case 'target-ready':
        case 'completed':
        case 'failed-validation':
        case 'failed-drain':
        case 'failed-source-stop':
        case 'failed-target-start':
        case 'failed-publication':
        case 'failed-interrupted':
        case 'rejected':
            return value
        default:
            return invalid()
    }
}
function errorKind(value: JsonValue | undefined): SwitchErrorKind {
    switch (value) {
        case 'invalid_request':
        case 'reused_id':
        case 'configuration_conflict':
        case 'remote_veto':
        case 'engine_locked':
        case 'unknown_receipt':
        case 'busy':
        case 'full':
        case 'storage':
        case 'unprepared':
        case 'unroutable':
        case 'target_running':
        case 'validation':
        case 'drain':
        case 'source_stop':
        case 'target_start':
        case 'publication':
        case 'interrupted':
            return value
        default:
            return invalid()
    }
}
function switchError(value: JsonValue | undefined): SwitchError | null {
    if (value === null) return null
    const row = object(value, ['code', 'kind', 'message'])
    const kind = errorKind(row.kind)
    const defaultCode =
        kind === 'invalid_request' || kind === 'reused_id'
            ? -32602
            : kind === 'configuration_conflict'
              ? -32010
              : kind === 'remote_veto'
                ? -32011
                : kind === 'engine_locked'
                  ? -32013
                  : -32000
    // Broker switchErrorDetail preserves PG denial as validation/-32012.
    const code = kind === 'validation' && row.code === -32012 ? -32012 : defaultCode
    const message = text(row.message)
    if (row.code !== code || Buffer.byteLength(message, 'utf8') > 1024) return invalid()
    return { code, kind, message }
}
function request(value: JsonValue | undefined): SwitchRequest {
    const row = object(value, [
        'nodeId',
        'requestId',
        'sourceEngine',
        'targetEngine',
        'expectedRevision'
    ])
    const sourceEngine = matching(row.sourceEngine, /^[a-zA-Z0-9._-]+$/)
    const targetEngine = matching(row.targetEngine, /^[a-zA-Z0-9._-]+$/)
    if (
        sourceEngine === targetEngine ||
        [sourceEngine, targetEngine].some(id => id === '.' || id === '..')
    )
        return invalid()
    return {
        ...(row.nodeId === undefined ? {} : { nodeId: text(row.nodeId) }),
        requestId: matching(row.requestId, identity),
        sourceEngine,
        targetEngine,
        expectedRevision: matching(row.expectedRevision, revision)
    }
}
function receipt(value: JsonValue | undefined): SwitchReceipt {
    const row = object(value, [
        'version',
        'nodeId',
        'requestId',
        'caller',
        'payloadHash',
        'request',
        'expectedRevision',
        'acceptedRevision',
        'sequence',
        'phase',
        'terminal',
        'acceptedAt',
        'updatedAt',
        'deadline',
        'finishedAt',
        'source',
        'target',
        'error',
        'recoveryRequired'
    ])
    const result: SwitchReceipt = {
        version: sequence(row.version),
        nodeId: text(row.nodeId),
        requestId: matching(row.requestId, identity),
        caller: text(row.caller),
        payloadHash: matching(row.payloadHash, /^sha256:[0-9a-f]{64}$/),
        request: request(row.request),
        expectedRevision: matching(row.expectedRevision, revision),
        acceptedRevision: matching(row.acceptedRevision, revision),
        sequence: sequence(row.sequence),
        phase: phase(row.phase),
        terminal: flag(row.terminal),
        acceptedAt: timestamp(row.acceptedAt),
        updatedAt: timestamp(row.updatedAt),
        deadline: timestamp(row.deadline),
        finishedAt: row.finishedAt === null ? null : timestamp(row.finishedAt),
        source: state(row.source),
        target: state(row.target),
        error: switchError(row.error),
        recoveryRequired: flag(row.recoveryRequired)
    }
    const terminal =
        result.phase === 'completed' ||
        result.phase === 'rejected' ||
        result.phase.startsWith('failed-')
    const accepted = Date.parse(result.acceptedAt)
    const updated = Date.parse(result.updatedAt)
    const deadline = Date.parse(result.deadline)
    if (
        result.version !== 1 ||
        result.nodeId !== result.request.nodeId ||
        result.requestId !== result.request.requestId ||
        result.expectedRevision !== result.request.expectedRevision ||
        terminal !== result.terminal ||
        terminal !== (result.finishedAt !== null) ||
        updated < accepted ||
        deadline <= accepted ||
        deadline - accepted > 900000 ||
        (result.finishedAt !== null &&
            (Date.parse(result.finishedAt) < accepted ||
                Date.parse(result.finishedAt) > updated)) ||
        (terminal && result.phase !== 'completed' && result.error === null) ||
        (result.phase === 'completed' && (result.error !== null || result.recoveryRequired))
    )
        return invalid()
    return result
}
export function parseEngineSwitchResult(value: JsonValue | undefined): { receipt: SwitchReceipt } {
    const row = object(value, ['receipt'])
    return { receipt: receipt(row.receipt) }
}
export function parseEngineSwitchStatus(value: JsonValue | undefined): SwitchStatus {
    const row = object(value, ['configurationRevision', 'activeRequestId', 'receipt', 'observed'])
    const observed =
        row.observed === undefined ? undefined : object(row.observed, ['source', 'target'])
    return {
        configurationRevision: matching(row.configurationRevision, revision),
        ...(row.activeRequestId === undefined
            ? {}
            : { activeRequestId: matching(row.activeRequestId, identity) }),
        ...(row.receipt === undefined ? {} : { receipt: receipt(row.receipt) }),
        ...(observed === undefined
            ? {}
            : { observed: { source: state(observed.source), target: state(observed.target) } })
    }
}
export function parseEngineSwitchChanged(value: JsonValue | undefined): SwitchChanged {
    const row = object(value, ['nodeId', 'requestId', 'sequence'])
    return {
        nodeId: text(row.nodeId),
        requestId: matching(row.requestId, identity),
        sequence: sequence(row.sequence)
    }
}
