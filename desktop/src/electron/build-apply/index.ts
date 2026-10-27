// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { dialog } from 'electron'
import path from 'node:path'
import type {
    BuildApplyGuiFacts,
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
    parseBuildApplyOperationResult
} from '@/electron/service-bridge/build-apply'

function guiFacts(): BuildApplyGuiFacts {
    return {
        pid: process.pid,
        executable: process.execPath,
        buildId: typeof __PAIR_GUI_BUILD_ID__ === 'string' ? __PAIR_GUI_BUILD_ID__ : ''
    }
}
function validId(value: string, empty = false): boolean {
    return (
        typeof value === 'string' &&
        ((empty && value === '') || /^[a-zA-Z0-9_-]{1,128}$/.test(value))
    )
}
export async function chooseBuildApplyPreview(): Promise<BuildApplyResult<BuildApplyPreview>> {
    const selection = await dialog.showOpenDialog({
        title: 'Choose staged build',
        properties: ['openDirectory', 'treatPackageAsDirectory']
    })
    if (selection.canceled || selection.filePaths.length !== 1) {
        return {
            available: false,
            value: null,
            refusal: { code: 'cancelled', detail: 'No build selected.' }
        }
    }
    const candidatePath = selection.filePaths[0]
    if (!path.isAbsolute(candidatePath) || Buffer.byteLength(candidatePath, 'utf8') > 4096) {
        throw new Error('Invalid staged build path.')
    }
    const gui = guiFacts()
    const request: BuildApplyPreviewRequest = { candidatePath, gui }
    return parseBuildApplyPreviewResult(
        await getModularSupervisor().callBuildApply('service/build-apply-preview', {
            ...request,
            gui: { ...request.gui }
        })
    )
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
    const gui = guiFacts()
    return parseBuildApplyOperationResult(
        await getModularSupervisor().callBuildApply('service/build-apply-start', {
            previewId: request.previewId,
            consent: { ...request.consent },
            gui: { ...gui }
        })
    )
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
    return parseBuildApplyOperationResult(
        await getModularSupervisor().callBuildApply('service/build-apply-status', {
            operationId: request.operationId
        })
    )
}

// This hook runs before connector auto-start and normal single-instance focus.
// Task 4 replaces the unavailable acknowledgement with attach-only gui-ready.
// Do not dial through connectOrStart here: the transaction owns pending startup.
export async function handleBuildApplyEntry(argv: readonly string[]): Promise<number | null> {
    const arguments_ = argv.filter(value => value.startsWith('--pair-build-apply'))
    if (arguments_.length === 0) return null
    if (arguments_.length !== 1 || !arguments_[0].startsWith('--pair-build-apply=')) return 64
    const operationId = arguments_[0].slice('--pair-build-apply='.length)
    if (!validId(operationId)) return 64
    return 69
}
