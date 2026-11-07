// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    DiagnosticPreview,
    DiagnosticPreviewRequest,
    DiagnosticRun
} from '@/shared/types/model-diagnostic'
import { isValidRoutingEngineId } from '@/shared/utils/engines'
import { isValidProfileKey } from '@/shared/utils/node-policy-model-settings'
import type { JsonObject, JsonValue } from './json-rpc-client'
import { diagnosticTimestamp, measuredProfile, parseDiagnosticIdentity } from './engine-settings'

// The broker owns a three-minute cleanup join, including the manager's stop budget.
export const DIAGNOSTIC_CANCEL_TIMEOUT_MS = 210_000

function invalid(): never {
    throw new Error('Invalid model diagnostic payload.')
}

function checkJSON(value: JsonValue, depth = 0): void {
    if (depth > 32) invalid()
    if (typeof value === 'number' && !Number.isFinite(value)) invalid()
    if (Array.isArray(value)) value.forEach(item => checkJSON(item, depth + 1))
    else if (value !== null && typeof value === 'object')
        Object.values(value).forEach(item => checkJSON(item, depth + 1))
}

function payload(value: JsonValue | undefined): void {
    if (value === undefined) invalid()
    checkJSON(value)
    if (Buffer.byteLength(JSON.stringify(value), 'utf8') > 64 * 1024) invalid()
}

function shape(value: JsonValue | undefined, fields: readonly string[]): JsonObject {
    if (!value || typeof value !== 'object' || Array.isArray(value)) invalid()
    if (Object.keys(value).some(key => !fields.includes(key))) invalid()
    return value
}

function text(value: JsonValue | undefined, required = true): string {
    if (
        typeof value !== 'string' ||
        (required && value.length === 0) ||
        Buffer.byteLength(value, 'utf8') > 256 ||
        /[\p{Cc}\p{Cs}]/u.test(value)
    )
        invalid()
    return value
}

function optionalText(value: JsonValue | undefined): string | undefined {
    return value === undefined ? undefined : text(value, false)
}

function integer(value: JsonValue | undefined, maximum = 2147483647): number {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value > maximum)
        invalid()
    return value
}

function flag(value: JsonValue | undefined): boolean {
    return typeof value === 'boolean' ? value : invalid()
}

function digest(value: JsonValue | undefined): string | undefined {
    if (value === undefined) return undefined
    const result = text(value)
    return /^[0-9a-f]{64}$/.test(result) ? result : invalid()
}

function list(value: JsonValue | undefined): JsonValue[] {
    if (value === null || value === undefined) return []
    return Array.isArray(value) && value.length <= 64 ? value : invalid()
}

function target(row: JsonObject): DiagnosticPreviewRequest {
    const engine = text(row.engine)
    const profileKey = text(row.profileKey)
    if (!isValidRoutingEngineId(engine) || !isValidProfileKey(profileKey)) invalid()
    return {
        engine,
        modelId: text(row.modelId),
        profileKey,
        ...(row.nodeId === undefined ? {} : { nodeId: text(row.nodeId, false) })
    }
}

export function diagnosticPreviewParams(value: JsonValue | undefined): JsonObject {
    payload(value)
    return { ...target(shape(value, ['engine', 'modelId', 'profileKey', 'nodeId'])) }
}

export function diagnosticStartParams(value: JsonValue | undefined): JsonObject {
    payload(value)
    const row = shape(value, ['engine', 'modelId', 'profileKey', 'nodeId', 'previewId', 'consent'])
    if (row.consent !== true) invalid()
    return { ...target(row), previewId: text(row.previewId), consent: true }
}

export function diagnosticRunParams(value: JsonValue | undefined, cancel = false): JsonObject {
    payload(value)
    const row = shape(value, ['nodeId', 'runId'])
    if (cancel && (row.runId === undefined || row.runId === '')) invalid()
    return {
        ...(row.nodeId === undefined ? {} : { nodeId: text(row.nodeId, false) }),
        ...(row.runId === undefined ? {} : { runId: text(row.runId, cancel) })
    }
}

export function parseDiagnosticPreview(value: JsonValue | undefined): DiagnosticPreview {
    payload(value)
    const row = shape(value, [
        'engine',
        'nodeId',
        'modelId',
        'profileKey',
        'configurationKey',
        'executionKey',
        'configurationKeyReason',
        'nativeBuildId',
        'buildInfo',
        'cpuModel',
        'totalRamBytes',
        'gpus',
        'artifacts',
        'requested',
        'perSlotContext',
        'slots',
        'placement',
        'previewId',
        'plannedPerSlotContext',
        'plannedSlots',
        'lowPromptTokens',
        'highPromptTokens',
        'generateTokens',
        'timeLimitSeconds',
        'unloadModels',
        'reloadedAfter',
        'drainEngines',
        'startsEngine',
        'warnings',
        'expiresAt'
    ])
    const model = (value: JsonValue): Omit<DiagnosticPreviewRequest, 'nodeId'> =>
        target(shape(value, ['engine', 'modelId', 'profileKey']))
    const result: DiagnosticPreview = {
        ...target(row),
        ...parseDiagnosticIdentity(row),
        previewId: text(row.previewId),
        plannedPerSlotContext: integer(row.plannedPerSlotContext),
        plannedSlots: integer(row.plannedSlots, 8),
        lowPromptTokens: integer(row.lowPromptTokens),
        highPromptTokens: integer(row.highPromptTokens),
        generateTokens: integer(row.generateTokens),
        timeLimitSeconds: integer(row.timeLimitSeconds, 1200),
        unloadModels: list(row.unloadModels).map(model),
        reloadedAfter: list(row.reloadedAfter).map(model),
        drainEngines: list(row.drainEngines).map(value => text(value, false)),
        startsEngine: flag(row.startsEngine),
        warnings: list(row.warnings).map(value => text(value, false)),
        expiresAt: diagnosticTimestamp(row.expiresAt)
    }
    const knownTargets =
        result.plannedPerSlotContext >= 208 &&
        result.plannedSlots >= 1 &&
        result.highPromptTokens === result.plannedPerSlotContext - 144 &&
        result.lowPromptTokens === Math.min(512, Math.floor(result.highPromptTokens / 2))
    const unknownTargets =
        result.plannedPerSlotContext === 0 &&
        result.lowPromptTokens === 0 &&
        result.highPromptTokens === 0
    if (
        result.generateTokens !== 128 ||
        result.timeLimitSeconds === 0 ||
        (!knownTargets && !unknownTargets)
    )
        invalid()
    return result
}

function phase(value: JsonValue | undefined): DiagnosticRun['phase'] {
    switch (value) {
        case 'reserving':
        case 'draining':
        case 'unloading':
        case 'baseline':
        case 'starting':
        case 'loading':
        case 'low':
        case 'high':
        case 'restoring':
        case 'done':
            return value
        default:
            return invalid()
    }
}

function outcome(value: JsonValue | undefined): DiagnosticRun['outcome'] {
    switch (value) {
        case 'running':
        case 'completed':
        case 'failed':
        case 'cancelled':
        case 'cleanup-failed':
            return value
        default:
            return invalid()
    }
}

export function parseDiagnosticRun(value: JsonValue | undefined): DiagnosticRun {
    payload(value)
    const row = shape(value, [
        'runId',
        'nodeId',
        'engine',
        'modelId',
        'profileKey',
        'configurationKey',
        'executionKey',
        'phase',
        'outcome',
        'failedPhase',
        'reason',
        'startedAt',
        'elapsedMs',
        'result'
    ])
    const result: DiagnosticRun = {
        ...target(row),
        runId: text(row.runId),
        configurationKey: digest(row.configurationKey),
        executionKey: digest(row.executionKey),
        phase: phase(row.phase),
        outcome: outcome(row.outcome),
        failedPhase: row.failedPhase === undefined ? undefined : phase(row.failedPhase),
        reason: optionalText(row.reason),
        startedAt: diagnosticTimestamp(row.startedAt),
        elapsedMs: integer(row.elapsedMs, Number.MAX_SAFE_INTEGER),
        result: row.result === undefined ? undefined : measuredProfile(row.result)
    }
    if (Boolean(result.configurationKey) === Boolean(result.executionKey)) invalid()
    if (
        result.result &&
        (result.result.runId !== result.runId ||
            result.result.configurationKey !== result.configurationKey ||
            result.result.executionKey !== result.executionKey ||
            result.result.outcome !== result.outcome)
    )
        invalid()
    return result
}

export function parseDiagnosticStatus(value: JsonValue | undefined): DiagnosticRun | null {
    return value === null ? null : parseDiagnosticRun(value)
}
