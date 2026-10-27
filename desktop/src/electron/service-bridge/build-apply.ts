// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BrowserWindow } from 'electron'
import { parseServiceBuildId } from '@/shared/utils/service-build'
import { availabilityValue } from './node-availability'
import type {
    Observation,
    BuildIdentity,
    BuildApplyFences,
    BuildApplyEngineFact,
    BuildApplyImpact,
    BuildApplyPreview,
    BuildApplyOperation,
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
function text(value: JsonValue | undefined, limit = 4096): string {
    if (
        typeof value !== 'string' ||
        Buffer.byteLength(value, 'utf8') > limit ||
        Array.from(value).some(character => character.charCodeAt(0) < 32)
    )
        return invalid()
    return value
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
    const row = object(value)
    const known = flag(row.known)
    if (!known) {
        if (row.value !== null) return invalid()
        return { known: false, value: null }
    }
    if (row.value === undefined || row.value === null) return invalid()
    return { known: true, value: read(row.value) }
}
function identity(value: JsonValue | undefined): BuildIdentity {
    const row = object(value)
    return {
        buildId: observed(row.buildId, value => parseServiceBuildId(text(value, 128)) ?? invalid()),
        sourceRevision: observed(row.sourceRevision, revision),
        dirty: observed(row.dirty, flag),
        sha256: observed(row.sha256, digest)
    }
}
function fences(value: JsonValue | undefined): BuildApplyFences {
    const row = object(value)
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
    const row = object(value)
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
        previewId: text(row.previewId, 128),
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
    const row = object(value)
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
export function parseBuildApplyOperation(value: JsonValue | undefined): BuildApplyOperation {
    const row = object(value)
    return {
        operationId: text(row.operationId, 128),
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
}
function result<T>(
    value: JsonValue | undefined,
    read: (entry: JsonValue) => T
): BuildApplyResult<T> {
    const row = object(value)
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
    value: JsonValue | undefined
): BuildApplyResult<BuildApplyOperation> {
    return result(value, parseBuildApplyOperation)
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
