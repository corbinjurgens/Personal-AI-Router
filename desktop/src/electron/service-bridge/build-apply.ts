// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BrowserWindow } from 'electron'
import { parseServiceBuildId } from '@/shared/utils/service-build'
import { availabilityValue } from './node-availability'
import { BUILD_APPLY_MAX_STATUS_BYTES } from '@/shared/types/build-apply'
import type {
    Observation,
    BuildIdentity,
    BuildApplyFences,
    BuildApplyEngineFact,
    BuildApplyImpact,
    BuildApplyPreview,
    BuildApplyOperation,
    BuildApplyStatus,
    BuildApplyStatusState,
    BuildApplyFailure,
    BuildApplyPhase,
    BuildApplyOutcome,
    BuildApplyConfidence,
    BuildApplyResult,
    BuildApplyLayout,
    BuildApplyRefusal
} from '@/shared/types/build-apply'
import type { JsonObject, JsonValue } from './json-rpc-client'

function invalid(): never {
    throw new Error('Invalid build apply response.')
}
function object(value: JsonValue | undefined): JsonObject {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return invalid()
    return value
}
function fields(value: JsonValue | undefined, names: string[]): JsonObject {
    const row = object(value)
    if (Object.keys(row).length !== names.length || names.some(name => !Object.hasOwn(row, name)))
        return invalid()
    return row
}
function text(value: JsonValue | undefined, limit = 4096): string {
    if (
        typeof value !== 'string' ||
        Buffer.byteLength(value, 'utf8') > limit ||
        Array.from(value).some(character => character.charCodeAt(0) < 32)
    )
        return invalid()
    return value
}
function token(value: JsonValue | undefined): string {
    const id = text(value, 128)
    return /^[a-zA-Z0-9_-]+$/.test(id) ? id : invalid()
}
function flag(value: JsonValue | undefined): boolean {
    return typeof value === 'boolean' ? value : invalid()
}
function count(value: JsonValue | undefined): number {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
        ? value
        : invalid()
}
function list<T>(value: JsonValue | undefined, read: (entry: JsonValue) => T): T[] {
    if (!Array.isArray(value) || value.length > 100000) return invalid()
    return value.map(entry => read(entry))
}
function observed<T>(value: JsonValue | undefined, read: (entry: JsonValue) => T): Observation<T> {
    const row = fields(value, ['known', 'value'])
    const known = flag(row.known)
    if (!known) {
        if (row.value !== null) return invalid()
        return { known: false, value: null }
    }
    if (row.value === undefined || row.value === null) return invalid()
    return { known: true, value: read(row.value) }
}
function identity(value: JsonValue | undefined): BuildIdentity {
    const row = fields(value, ['buildId', 'sourceRevision', 'dirty', 'sha256'])
    return {
        buildId: observed(row.buildId, value => parseServiceBuildId(text(value, 128)) ?? invalid()),
        sourceRevision: observed(row.sourceRevision, revision),
        dirty: observed(row.dirty, flag),
        sha256: observed(row.sha256, digest)
    }
}
function fences(value: JsonValue | undefined): BuildApplyFences {
    const row = fields(value, [
        'runtimeGeneration',
        'serviceInstance',
        'brokerGeneration',
        'managerIncarnation',
        'policyRevision',
        'settingsRevision',
        'intentRevision',
        'locksRevision',
        'availabilityRevision'
    ])
    return {
        runtimeGeneration: observed(row.runtimeGeneration, text),
        serviceInstance: observed(row.serviceInstance, text),
        brokerGeneration: observed(row.brokerGeneration, text),
        managerIncarnation: observed(row.managerIncarnation, text),
        policyRevision: observed(row.policyRevision, text),
        settingsRevision: observed(row.settingsRevision, text),
        intentRevision: observed(row.intentRevision, text),
        locksRevision: observed(row.locksRevision, text),
        availabilityRevision: observed(row.availabilityRevision, text)
    }
}
function revision(value: JsonValue): string {
    const result = text(value, 64)
    return /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(result) ? result : invalid()
}
function digest(value: JsonValue): string {
    const result = text(value, 64)
    return /^[a-f0-9]{64}$/.test(result) ? result : invalid()
}
function engine(value: JsonValue): BuildApplyEngineFact {
    const row = object(value)
    return {
        engine: text(row.engine),
        active: observed(row.active, flag),
        running: observed(row.running, flag),
        loadedModels: observed(row.loadedModels, value => list(value, text)),
        desiredState: observed(row.desiredState, text),
        locked: observed(row.locked, flag),
        scopedStopAuthority: observed(row.scopedStopAuthority, flag)
    }
}
function refusal(value: JsonValue | undefined): BuildApplyRefusal {
    const row = fields(value, ['code', 'detail'])
    return { code: text(row.code, 128), detail: text(row.detail, 1024) }
}
function layout(value: JsonValue): BuildApplyLayout {
    const row = object(value)
    return {
        rootName: text(row.rootName),
        executable: text(row.executable),
        serviceDir: text(row.serviceDir),
        serviceExecutable: text(row.serviceExecutable)
    }
}
function impact(value: JsonValue | undefined): BuildApplyImpact {
    const row = object(value)
    const maintenance = object(row.maintenance)
    const settings = object(row.expectedSettings)
    const interruption = object(row.interruptionPolicy)
    const restore = object(row.restorePolicy)
    return {
        complete: flag(row.complete),
        fences: fences(row.fences),
        maintenance: {
            buildApplyActive: observed(maintenance.buildApplyActive, flag),
            modelSettingsActive: observed(maintenance.modelSettingsActive, flag)
        },
        availability: observed(row.availability, value => availabilityValue(value) ?? invalid()),
        pauseOrigin: observed(row.pauseOrigin, text),
        engines: observed(row.engines, value => list(value, engine)),
        activeWork: observed(row.activeWork, count),
        modelOperations: observed(row.modelOperations, value =>
            list(value, entry => {
                const operation = object(entry)
                return {
                    engine: text(operation.engine),
                    kind: text(operation.kind),
                    blocksApply: flag(operation.blocksApply)
                }
            })
        ),
        settingsDifferences: observed(row.settingsDifferences, value =>
            list(value, entry => {
                const difference = object(entry)
                return {
                    field: text(difference.field),
                    stored: observed(difference.stored, flag),
                    effective: observed(difference.effective, flag),
                    persisted: observed(difference.persisted, flag)
                }
            })
        ),
        expectedSettings: {
            policyDigest: observed(settings.policyDigest, text),
            engineSettingsDigest: observed(settings.engineSettingsDigest, text),
            intentDigest: observed(settings.intentDigest, text),
            locksDigest: observed(settings.locksDigest, text),
            stopEngines: observed(settings.stopEngines, flag),
            unloadModels: observed(settings.unloadModels, flag)
        },
        interruptionPolicy: {
            onActive: observed(interruption.onActive, value => {
                if (value !== 'finish' && value !== 'cancel') return invalid()
                return value
            }),
            drainTimeoutSeconds: observed(interruption.drainTimeoutSeconds, count),
            stopsEngines: flag(interruption.stopsEngines),
            unloadsModels: flag(interruption.unloadsModels),
            committedStreamsMayFail: flag(interruption.committedStreamsMayFail),
            includesWorkBeforeFence: flag(interruption.includesWorkBeforeFence)
        },
        restorePolicy: {
            remainPaused: flag(restore.remainPaused),
            requireLocalResume: flag(restore.requireLocalResume),
            preserveIntentAndLocks: flag(restore.preserveIntentAndLocks),
            preserveNewerPause: flag(restore.preserveNewerPause),
            automaticRollback: flag(restore.automaticRollback)
        },
        refusals: list(row.refusals, refusal)
    }
}
function preview(value: JsonValue): BuildApplyPreview {
    const row = object(value)
    return {
        previewId: token(row.previewId),
        expiresAt: text(row.expiresAt, 64),
        hostId: observed(row.hostId, text),
        platform: text(row.platform, 32),
        architecture: text(row.architecture, 32),
        currentGui: identity(row.currentGui),
        bundledService: identity(row.bundledService),
        requestedGui: identity(row.requestedGui),
        requestedService: identity(row.requestedService),
        runningService: identity(row.runningService),
        packageSha256: observed(row.packageSha256, digest),
        targetLayout: observed(row.targetLayout, layout),
        observationsComplete: flag(row.observationsComplete),
        impact: impact(row.impact)
    }
}
function phase(value: JsonValue | undefined): BuildApplyPhase {
    switch (value) {
        case 'preflight':
        case 'prepare':
        case 'quiesce':
        case 'shutdown':
        case 'swap':
        case 'restart':
        case 'verify-build':
        case 'verify-settings':
        case 'finish':
            return value
        default:
            return invalid()
    }
}
function outcome(value: JsonValue | undefined): BuildApplyOutcome | null {
    switch (value) {
        case null:
        case 'succeeded':
        case 'refused':
        case 'failed':
        case 'interrupted':
        case 'uncertain':
            return value
        default:
            return invalid()
    }
}
function confidence(value: JsonValue | undefined): BuildApplyConfidence {
    switch (value) {
        case 'unknown':
        case 'incomplete':
        case 'verified':
            return value
        default:
            return invalid()
    }
}
function failure(value: JsonValue | undefined): BuildApplyFailure | null {
    if (value === null) return null
    const row = fields(value, [
        'phase',
        'reason',
        'detail',
        'bytesChanged',
        'settingsChanged',
        'runningIdentity',
        'readbackConfidence',
        'safeNextAction'
    ])
    return {
        phase: phase(row.phase),
        reason: text(row.reason, 128),
        detail: text(row.detail, 1024),
        bytesChanged: observed(row.bytesChanged, flag),
        settingsChanged: observed(row.settingsChanged, flag),
        runningIdentity: identity(row.runningIdentity),
        readbackConfidence: confidence(row.readbackConfidence),
        safeNextAction: text(row.safeNextAction, 1024)
    }
}
export function parseBuildApplyOperation(
    value: JsonValue | undefined,
    expectedPreviewId = ''
): BuildApplyOperation {
    if (expectedPreviewId !== '') token(expectedPreviewId)
    const row = fields(value, [
        'operationId',
        'previewId',
        'ownershipAcknowledged',
        'phase',
        'outcome',
        'failure',
        'saved',
        'enforced',
        'physicalStop',
        'remainsPaused',
        'runningService',
        'runningGui',
        'readbackConfidence',
        'fences'
    ])
    const operation: BuildApplyOperation = {
        operationId: token(row.operationId),
        previewId: token(row.previewId),
        ownershipAcknowledged: flag(row.ownershipAcknowledged),
        phase: phase(row.phase),
        outcome: outcome(row.outcome),
        failure: failure(row.failure),
        saved: observed(row.saved, flag),
        enforced: observed(row.enforced, flag),
        physicalStop: observed(row.physicalStop, flag),
        remainsPaused: observed(row.remainsPaused, flag),
        runningService: identity(row.runningService),
        runningGui: identity(row.runningGui),
        readbackConfidence: confidence(row.readbackConfidence),
        fences: fences(row.fences)
    }
    if (
        (expectedPreviewId !== '' && operation.previewId !== expectedPreviewId) ||
        (operation.outcome === null && operation.failure !== null) ||
        (operation.outcome === 'succeeded' && operation.failure !== null) ||
        (operation.failure !== null && operation.failure.phase !== operation.phase)
    )
        return invalid()
    return operation
}
function result<T>(
    value: JsonValue | undefined,
    read: (entry: JsonValue) => T
): BuildApplyResult<T> {
    const row = fields(value, ['available', 'value', 'refusal'])
    if (!flag(row.available)) {
        if (row.value !== null) return invalid()
        return { available: false, value: null, refusal: refusal(row.refusal) }
    }
    if (row.refusal !== null || row.value === null || row.value === undefined) return invalid()
    return { available: true, value: read(row.value), refusal: null }
}
export function parseBuildApplyPreviewResult(
    value: JsonValue | undefined
): BuildApplyResult<BuildApplyPreview> {
    return result(value, preview)
}
export function parseBuildApplyOperationResult(
    value: JsonValue | undefined,
    expectedPreviewId = ''
): BuildApplyResult<BuildApplyOperation> {
    if (expectedPreviewId !== '') token(expectedPreviewId)
    return result(value, entry => parseBuildApplyOperation(entry, expectedPreviewId))
}

export function parseBuildApplyStatusResult(
    value: JsonValue | undefined,
    expectedPreviewId = ''
): BuildApplyResult<BuildApplyStatus> {
    if (expectedPreviewId !== '') token(expectedPreviewId)
    if (
        value === undefined ||
        Buffer.byteLength(JSON.stringify(value), 'utf8') > BUILD_APPLY_MAX_STATUS_BYTES
    )
        return invalid()
    return result<BuildApplyStatus>(value, entry => {
        const row = fields(entry, ['state', 'operation'])
        const state = statusState(row.state)
        if (state === 'idle') {
            if (row.operation !== null || expectedPreviewId !== '') return invalid()
            return { state: 'idle', operation: null }
        }
        const operation = parseBuildApplyOperation(row.operation, expectedPreviewId)
        switch (state) {
            case 'pending':
                if (operation.outcome !== null) return invalid()
                return { state: 'pending', operation }
            case 'terminal':
                if (operation.outcome === null) return invalid()
                return { state: 'terminal', operation }
            case 'unknown':
                return { state: 'unknown', operation }
        }
    })
}

function statusState(value: JsonValue | undefined): BuildApplyStatusState {
    switch (value) {
        case 'idle':
        case 'pending':
        case 'terminal':
        case 'unknown':
            return value
        default:
            return invalid()
    }
}

let operationObserver: ((operation: BuildApplyOperation) => void) | null = null

export function setBuildApplyOperationObserver(
    observer: (operation: BuildApplyOperation) => void
): void {
    operationObserver = observer
}

// Called only by the existing private ModularSupervisor notification dispatcher.
// Native IPC uses the same window fanout as other Electron-native operations.
export function dispatchBuildApplyChanged(value: JsonValue | undefined): void {
    const operation = parseBuildApplyOperation(value)
    operationObserver?.(operation)
    for (const window of BrowserWindow.getAllWindows()) {
        if (!window.isDestroyed() && !window.webContents.isDestroyed()) {
            window.webContents.send('build-apply:changed', operation)
        }
    }
}
