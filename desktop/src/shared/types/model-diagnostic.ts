// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { MeasuredProfile } from '@/shared/types/model-config'

export type DiagnosticPreviewRequest = {
    engine: string
    nodeId?: string
    modelId: string
    profileKey: string
}

type DiagnosticIdentity = Pick<
    MeasuredProfile,
    | 'configurationKey'
    | 'executionKey'
    | 'configurationKeyReason'
    | 'nativeBuildId'
    | 'buildInfo'
    | 'cpuModel'
    | 'totalRamBytes'
    | 'gpus'
    | 'artifacts'
    | 'requested'
    | 'perSlotContext'
    | 'slots'
    | 'placement'
>

export type DiagnosticPreview = DiagnosticPreviewRequest &
    DiagnosticIdentity & {
        previewId: string
        plannedPerSlotContext: number
        plannedSlots: number
        lowPromptTokens: number
        highPromptTokens: number
        generateTokens: number
        timeLimitSeconds: number
        unloadModels: Array<Omit<DiagnosticPreviewRequest, 'nodeId'>>
        reloadedAfter: Array<Omit<DiagnosticPreviewRequest, 'nodeId'>>
        drainEngines: string[]
        startsEngine: boolean
        warnings: string[]
        expiresAt: string
    }

export type DiagnosticStartRequest = DiagnosticPreviewRequest & {
    previewId: string
    consent: boolean
}

export type DiagnosticRunRef = { nodeId?: string; runId: string }

/** Only broker public state, never manager reservation authority or markers. */
export type DiagnosticRun = {
    runId: string
    nodeId?: string
    engine: string
    modelId: string
    profileKey: string
    configurationKey?: string
    executionKey?: string
    phase: NonNullable<MeasuredProfile['failedPhase']>
    outcome: 'running' | MeasuredProfile['outcome']
    failedPhase?: MeasuredProfile['failedPhase']
    reason?: string
    startedAt: string
    elapsedMs: number
    result?: MeasuredProfile
}
