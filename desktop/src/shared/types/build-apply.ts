// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NodeAvailability } from './node-policy'

export type Observation<T> = { known: false; value: null } | { known: true; value: T }

export interface BuildIdentity {
    buildId: Observation<string>
    sourceRevision: Observation<string>
    dirty: Observation<boolean>
    sha256: Observation<string>
}
export interface BuildApplyGuiFacts {
    pid: number
    executable: string
    buildId: string
}
export interface BuildApplyPreviewRequest {
    candidatePath: string
    gui: BuildApplyGuiFacts
}
export interface BuildApplyConsent {
    allowInterruption: boolean
    allowEngineStop: boolean
}
export interface BuildApplyStartRequest {
    previewId: string
    consent: BuildApplyConsent
    gui: BuildApplyGuiFacts
}
export interface BuildApplyStatusRequest {
    operationId: string
}
export interface BuildApplyGuiReadyRequest {
    operationId: string
    gui: BuildApplyGuiFacts
}
export interface BuildApplyFences {
    runtimeGeneration: Observation<string>
    serviceInstance: Observation<string>
    brokerGeneration: Observation<string>
    managerIncarnation: Observation<string>
    policyRevision: Observation<string>
    settingsRevision: Observation<string>
    intentRevision: Observation<string>
    locksRevision: Observation<string>
    availabilityRevision: Observation<string>
}
export interface BuildApplyEngineFact {
    engine: string
    active: Observation<boolean>
    running: Observation<boolean>
    loadedModels: Observation<string[]>
    desiredState: Observation<string>
    locked: Observation<boolean>
    scopedStopAuthority: Observation<boolean>
}
export interface BuildApplyModelOperation {
    engine: string
    kind: string
    blocksApply: boolean
}
export interface BuildApplySettingsDifference {
    field: string
    stored: Observation<boolean>
    effective: Observation<boolean>
    persisted: Observation<boolean>
}
export interface BuildApplyExpectedSettings {
    policyDigest: Observation<string>
    engineSettingsDigest: Observation<string>
    intentDigest: Observation<string>
    locksDigest: Observation<string>
    stopEngines: Observation<boolean>
    unloadModels: Observation<boolean>
}
export interface BuildApplyInterruptionPolicy {
    onActive: Observation<'finish' | 'cancel'>
    drainTimeoutSeconds: Observation<number>
    stopsEngines: boolean
    unloadsModels: boolean
    committedStreamsMayFail: boolean
    includesWorkBeforeFence: boolean
}
export interface BuildApplyRestorePolicy {
    remainPaused: boolean
    requireLocalResume: boolean
    preserveIntentAndLocks: boolean
    preserveNewerPause: boolean
    automaticRollback: boolean
}
export interface BuildApplyRefusal {
    code: string
    detail: string
}
export interface BuildApplyMaintenanceState {
    buildApplyActive: Observation<boolean>
    modelSettingsActive: Observation<boolean>
}
export interface BuildApplyImpact {
    complete: boolean
    fences: BuildApplyFences
    maintenance: BuildApplyMaintenanceState
    availability: Observation<NodeAvailability>
    pauseOrigin: Observation<string>
    engines: Observation<BuildApplyEngineFact[]>
    activeWork: Observation<number>
    modelOperations: Observation<BuildApplyModelOperation[]>
    settingsDifferences: Observation<BuildApplySettingsDifference[]>
    expectedSettings: BuildApplyExpectedSettings
    interruptionPolicy: BuildApplyInterruptionPolicy
    restorePolicy: BuildApplyRestorePolicy
    refusals: BuildApplyRefusal[]
}
export interface BuildApplyLayout {
    rootName: string
    executable: string
    serviceDir: string
    serviceExecutable: string
}
export interface BuildApplyPreview {
    previewId: string
    expiresAt: string
    hostId: Observation<string>
    platform: string
    architecture: string
    currentGui: BuildIdentity
    bundledService: BuildIdentity
    requestedGui: BuildIdentity
    requestedService: BuildIdentity
    runningService: BuildIdentity
    packageSha256: Observation<string>
    targetLayout: Observation<BuildApplyLayout>
    observationsComplete: boolean
    impact: BuildApplyImpact
}
export type BuildApplyPhase =
    | 'preflight'
    | 'prepare'
    | 'quiesce'
    | 'shutdown'
    | 'swap'
    | 'restart'
    | 'verify-build'
    | 'verify-settings'
    | 'finish'
export type BuildApplyOutcome = 'succeeded' | 'refused' | 'failed' | 'interrupted' | 'uncertain'
export type BuildApplyConfidence = 'unknown' | 'incomplete' | 'verified'
export interface BuildApplyFailure {
    phase: BuildApplyPhase
    reason: string
    detail: string
    bytesChanged: Observation<boolean>
    settingsChanged: Observation<boolean>
    runningIdentity: BuildIdentity
    readbackConfidence: BuildApplyConfidence
    safeNextAction: string
}
export interface BuildApplyOperation {
    operationId: string
    ownershipAcknowledged: boolean
    phase: BuildApplyPhase
    outcome: BuildApplyOutcome | null
    failure: BuildApplyFailure | null
    saved: Observation<boolean>
    enforced: Observation<boolean>
    physicalStop: Observation<boolean>
    remainsPaused: Observation<boolean>
    runningService: BuildIdentity
    runningGui: BuildIdentity
    readbackConfidence: BuildApplyConfidence
    fences: BuildApplyFences
}
export type BuildApplyResult<T> =
    | { available: true; value: T; refusal: null }
    | { available: false; value: null; refusal: BuildApplyRefusal }
export interface BuildApplyPrepareRequest {
    operationId: string
    fences: BuildApplyFences
    consent: BuildApplyConsent
}
export interface BuildApplyFinishRequest {
    operationId: string
    fences: BuildApplyFences
}
export const BUILD_APPLY_MAINTENANCE_BUSY = 'maintenance_busy'
