// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    BuildApplyPreview,
    BuildApplyPreviewRequest,
    BuildApplyOperation,
    BuildApplyResult,
    BuildApplyStartRequest,
    BuildApplyStatusRequest
} from '@/shared/types/build-apply'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import {
    parseBuildApplyPreviewResult,
    parseBuildApplyOperationResult,
    setBuildApplyOperationObserver
} from '@/electron/service-bridge/build-apply'
import { chooseCandidatePath, guiFacts, quitForBuildApply } from './native'

let selected: { preview: BuildApplyPreview; generation: number } | null = null
let selectionGeneration = 0
let statusRead: { key: string; promise: Promise<BuildApplyResult<BuildApplyOperation>> } | null =
    null
let handoff: { operationId: string; promise: Promise<void> | null } | null = null

function completeHandoff(operation: BuildApplyOperation, generation: number): Promise<void> {
    const current = handoff
    const supervisor = getModularSupervisor()
    if (
        !current ||
        operation.operationId !== current.operationId ||
        generation !== supervisor.getBuildApplyGeneration()
    )
        return Promise.resolve()
    if (operation.outcome !== null) {
        handoff = null
        return Promise.resolve()
    }
    if (!operation.ownershipAcknowledged) return Promise.resolve()
    if (!current.promise) current.promise = quitForBuildApply(() => supervisor.stop())
    return current.promise
}

setBuildApplyOperationObserver(operation => {
    // Retain the promise so explicit status reads also observe handoff errors.
    void completeHandoff(operation, getModularSupervisor().getBuildApplyGeneration()).catch(
        () => {}
    )
})

function completePreview(preview: BuildApplyPreview): boolean {
    const impact = preview.impact
    const identities = [
        preview.currentGui,
        preview.bundledService,
        preview.runningService,
        preview.requestedGui,
        preview.requestedService
    ]
    return (
        preview.observationsComplete &&
        impact.complete &&
        preview.hostId.known &&
        preview.packageSha256.known &&
        preview.targetLayout.known &&
        identities.every(
            value =>
                value.buildId.known &&
                value.sourceRevision.known &&
                value.dirty.known &&
                value.sha256.known
        ) &&
        Object.values(impact.fences).every(value => value.known) &&
        Object.values(impact.expectedSettings).every(value => value.known) &&
        impact.maintenance.buildApplyActive.known &&
        !impact.maintenance.buildApplyActive.value &&
        impact.maintenance.modelSettingsActive.known &&
        !impact.maintenance.modelSettingsActive.value &&
        impact.engines.known &&
        impact.engines.value.every(engine =>
            Object.values(engine).every(value => typeof value === 'string' || value.known)
        ) &&
        impact.activeWork.known &&
        impact.modelOperations.known &&
        !impact.modelOperations.value.some(operation => operation.blocksApply) &&
        impact.settingsDifferences.known &&
        impact.availability.known &&
        impact.pauseOrigin.known &&
        impact.interruptionPolicy.onActive.known &&
        impact.interruptionPolicy.drainTimeoutSeconds.known &&
        impact.restorePolicy.remainPaused &&
        impact.restorePolicy.requireLocalResume &&
        impact.restorePolicy.preserveIntentAndLocks &&
        impact.restorePolicy.preserveNewerPause &&
        !impact.restorePolicy.automaticRollback &&
        impact.refusals.length === 0
    )
}
function validId(value: string, empty = false): boolean {
    return (
        typeof value === 'string' &&
        ((empty && value === '') || /^[a-zA-Z0-9_-]{1,128}$/.test(value))
    )
}
export async function chooseBuildApplyPreview(): Promise<BuildApplyResult<BuildApplyPreview>> {
    selected = null
    const selection = ++selectionGeneration
    const supervisor = getModularSupervisor()
    const generation = supervisor.getBuildApplyGeneration()
    const candidatePath = await chooseCandidatePath()
    if (candidatePath === null) {
        return {
            available: false,
            value: null,
            refusal: { code: 'cancelled', detail: 'No build selected.' }
        }
    }
    const gui = guiFacts()
    const request: BuildApplyPreviewRequest = { candidatePath, gui }
    const result = parseBuildApplyPreviewResult(
        await getModularSupervisor().callBuildApply('service/build-apply-preview', {
            ...request,
            gui: { ...request.gui }
        })
    )
    if (selection !== selectionGeneration || generation !== supervisor.getBuildApplyGeneration()) {
        throw new Error('Service connection changed. Choose a staged build for a new preview.')
    }
    if (result.available) selected = { preview: result.value, generation }
    return result
}
export async function startBuildApply(
    request: Omit<BuildApplyStartRequest, 'gui'>
): Promise<BuildApplyResult<BuildApplyOperation>> {
    if (
        !request ||
        Object.keys(request).some(key => key !== 'previewId' && key !== 'consent') ||
        !validId(request.previewId) ||
        !request.consent ||
        Object.keys(request.consent).some(
            key => key !== 'allowInterruption' && key !== 'allowEngineStop'
        ) ||
        typeof request.consent.allowInterruption !== 'boolean' ||
        typeof request.consent.allowEngineStop !== 'boolean'
    ) {
        throw new Error('Invalid build apply consent.')
    }
    if (!request.consent.allowEngineStop || !request.consent.allowInterruption) {
        selected = null
        return {
            available: false,
            value: null,
            refusal: {
                code: 'consent_required',
                detail: 'Full service restart requires interruption and engine-stop consent. Choose a staged build for a new preview.'
            }
        }
    }
    const supervisor = getModularSupervisor()
    const choice = selected
    selected = null
    if (
        !choice ||
        choice.preview.previewId !== request.previewId ||
        choice.generation !== supervisor.getBuildApplyGeneration() ||
        !completePreview(choice.preview) ||
        !Number.isFinite(Date.parse(choice.preview.expiresAt)) ||
        Date.parse(choice.preview.expiresAt) <= Date.now()
    ) {
        return {
            available: false,
            value: null,
            refusal: {
                code: 'preview_required',
                detail: 'The preview is stale or incomplete. Choose a staged build for a new preview and consent again.'
            }
        }
    }
    const gui = guiFacts()
    if (!gui.buildId) {
        return {
            available: false,
            value: null,
            refusal: {
                code: 'identity_unknown',
                detail: 'The running GUI build identity is unknown. Choose a build with complete identity metadata.'
            }
        }
    }
    let result: BuildApplyResult<BuildApplyOperation>
    try {
        result = parseBuildApplyOperationResult(
            await supervisor.callBuildApply('service/build-apply-start', {
                previewId: request.previewId,
                consent: { ...request.consent },
                gui: { ...gui }
            })
        )
    } catch {
        // The first reply is the only current proof of our operation ID. Latest
        // status cannot correlate this initiation until the service returns its
        // admitting preview ID. Never retry or hand off an unknown outcome.
        return {
            available: false,
            value: null,
            refusal: {
                code: 'outcome_unknown',
                detail: 'Build application outcome is unknown. Check status before any further action; do not retry application.'
            }
        }
    }
    if (result.available && validId(result.value.operationId)) {
        handoff = {
            operationId: result.value.operationId,
            promise: null
        }
        await completeHandoff(result.value, choice.generation)
    }
    return result
}
export async function getBuildApplyStatus(
    request: BuildApplyStatusRequest
): Promise<BuildApplyResult<BuildApplyOperation>> {
    if (
        !request ||
        Object.keys(request).some(key => key !== 'operationId') ||
        !validId(request.operationId, true)
    ) {
        throw new Error('Invalid build apply operation.')
    }
    const supervisor = getModularSupervisor()
    const generation = supervisor.getBuildApplyGeneration()
    const key = `${generation}:${request.operationId}`
    if (statusRead?.key === key) return statusRead.promise
    const read = async (): Promise<BuildApplyResult<BuildApplyOperation>> => {
        let result: BuildApplyResult<BuildApplyOperation>
        try {
            result = parseBuildApplyOperationResult(
                await supervisor.callBuildApply('service/build-apply-status', {
                    operationId: request.operationId
                })
            )
        } catch {
            result = {
                available: false,
                value: null,
                refusal: {
                    code: 'unavailable',
                    detail: 'Build operation status is not available in this build or connection. Reconnect and read status before any further action.'
                }
            }
        }
        if (
            generation !== supervisor.getBuildApplyGeneration() ||
            (result.available &&
                request.operationId !== '' &&
                result.value.operationId !== request.operationId)
        ) {
            throw new Error(
                'Build apply connection changed or operation changed. Read status again.'
            )
        }
        if (result.available) await completeHandoff(result.value, generation)
        return result
    }
    const promise = read()
    statusRead = { key, promise }
    try {
        return await promise
    } finally {
        if (statusRead?.promise === promise) statusRead = null
    }
}

// This hook runs before connector auto-start and normal single-instance focus.
// Do not dial through connectOrStart here: the transaction owns pending startup.
export async function handleBuildApplyEntry(argv: readonly string[]): Promise<number | null> {
    const arguments_ = argv.filter(value => value.startsWith('--pair-build-apply'))
    if (arguments_.length === 0) return null
    if (arguments_.length !== 1 || !arguments_[0].startsWith('--pair-build-apply=')) return 64
    const operationId = arguments_[0].slice('--pair-build-apply='.length)
    if (!validId(operationId)) return 64
    try {
        const result = parseBuildApplyOperationResult(
            await getModularSupervisor().acknowledgeBuildApplyGui({
                operationId,
                gui: { ...guiFacts() }
            })
        )
        return result.available &&
            result.value.operationId === operationId &&
            result.value.ownershipAcknowledged
            ? null
            : 69
    } catch {
        return 69
    }
}
