// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JsonValue } from '@/shared/types/json'

export type RequestedLoad = {
    context?: { mode: 'fixed' | 'auto'; tokens?: number; floor?: number; cap?: number }
    slots?: number
    gpu?: { mode: 'native-default' | 'cpu' | 'layers' | 'auto'; layers?: number }
    cacheTypeK?: 'f16' | 'bf16' | 'q8_0' | 'q4_0'
    cacheTypeV?: 'f16' | 'bf16' | 'q8_0' | 'q4_0'
    flashAttention?: 'inherit' | 'on' | 'off'
    batchSize?: number
    ubatchSize?: number
    threads?: number
    threadsBatch?: number
    kvOffload?: boolean
    opOffload?: boolean
    sleepIdleSeconds?: number
    cacheRAM?: number
}
export type ResolvedLoad = {
    perSlotContext: number
    totalContext: number
    slots: number
    kvUnified?: boolean
    gpu: { mode: 'cpu' | 'layers'; layers?: number }
    options: RequestedLoad
    artifacts: Array<{
        role: string
        path: string
        resolvedTarget: string
        /** Positive int64 decimal text, preserving file sizes exactly. */
        size: string
        mtimeNs: string
        sampleSha256: string
    }>
    nativeBuildId: string
    sourceInputDigest: string
    presetDigest: string
    resolverRevision?: string
    reviewedInputId?: string
    deviceBinding?: {
        hostId: string
        domainId: string
        deviceId: string
        selector: string
        digest: string
    }
}
export type ModelLoad = {
    modelId: string
    requested: RequestedLoad
    resolved?: ResolvedLoad
}
export type ModelChange = {
    modelId: string
    expectedPolicyRevision: string
    load?: { set: RequestedLoad; clear?: Array<keyof RequestedLoad> }
    policy?: {
        generation?: Partial<
            Record<
                | 'temperature'
                | 'top_p'
                | 'top_k'
                | 'min_p'
                | 'repeat_penalty'
                | 'frequency_penalty'
                | 'presence_penalty'
                | 'seed',
                number | null
            >
        >
        requestOptionsMode?: 'defaults' | 'enforced'
        roles?: Partial<Record<'weak' | 'medium' | 'strong', boolean>>
        served?: boolean
        resident?: boolean
        binding?: boolean
    }
    previewDigest?: string
    supersedesRequestId?: string
}
export type ArtifactSummary = {
    role: 'model' | 'draft' | 'companion'
    identity: string
    size: string
    assurance: string
}
export type Estimate = {
    status: 'estimated' | 'partial' | 'unknown' | 'unavailable'
    source?: string
    buildId?: string
    timestamp?: string
    memorySource?: string
    memoryTimestamp?: string
    capacityFit?: 'unknown' | 'exceeds' | 'tight' | 'fits'
    freeFit?: 'unknown' | 'exceeds' | 'tight' | 'fits'
    /** Advisory uint64 decimal text, never native allocation authority. */
    hostBytes?: string
    deviceBytes?: string
    domains?: Array<{
        id: string
        kind: 'host' | 'device' | 'shared'
        deviceId?: string
        capacityBytes?: string
        freeBytes?: string
        requiredBytes?: string
        reserveBytes?: string
        capacityFit?: 'unknown' | 'exceeds' | 'tight' | 'fits'
        freeFit?: 'unknown' | 'exceeds' | 'tight' | 'fits'
        source?: string
        sampleId?: string
        capacitySource?: string
        capacityTimestamp?: string
        freeSource?: string
        freeTimestamp?: string
        remainingValidityMs?: number
        unknowns?: string[]
    }>
    components?: Array<{
        name: string
        domainId?: string
        status: 'estimated' | 'partial' | 'unknown' | 'excluded'
        bytes?: string
        reason?: string
    }>
    assumptions?: string[]
    unknowns?: string[]
}
/** Engine uses the service routing ID, including descriptor-defined engines. */
export type InspectModelRequest = { engine: string; nodeId?: string; modelId: string }
export type InspectModelResult = {
    engine: string
    nodeId?: string
    modelId: string
    status: 'resolved' | 'missing' | 'ambiguous' | 'busy' | 'unknown' | 'unavailable'
    reason?: string
    source?: string
    assurance?: string
    artifacts?: ArtifactSummary[]
    facts?: {
        status: string
        source?: string
        architecture?: string
        contextTokens?: number
        layerCount?: number
        templatePresent?: boolean
        templateDigest?: string
        dimensions?: Record<string, string>
        unknowns?: string[]
    }
    options?: Array<{
        field: string
        label: string
        kind: 'integer' | 'number' | 'boolean' | 'enum' | 'string'
        minimum?: number
        maximum?: number
        values?: string[]
        default?: JsonValue
        defaultSource?: string
        supported: boolean
        reason?: string
        portable: boolean
        owner?: string
        phase?: string
        identityEffect?: string
        estimateEffect?: string
    }>
    estimate: Estimate
    loadObservation?: LoadBufferObservation
}

type LoadBufferComponent =
    | 'weights'
    | 'attention-cache'
    | 'recurrent-state'
    | 'compute-workspace'
    | 'prompt-cache'
    | 'companions'
type LoadBufferReason =
    | 'observation-omitted'
    | 'response-budget'
    | 'history-unavailable'
    | 'no-matching-sample'
    | 'missing-boundary'
    | 'unsupported-build'
    | 'clock-reversal'
    | 'unmapped-device'
    | 'mixed-role'
    | 'incomplete-reserve'
    | 'oversized-frame'
    | 'truncated-frame'
    | 'invalid-utf8'
    | 'read-failure'
    | 'capture-budget'
    | 'unknown-baseline'
    | 'source-changed'
    | 'source-unavailable'
    | 'unsupported-buffer'
/** Detached, service-owned evidence for one load, outside settings authority. */
type LoadBufferObservation = {
    status: 'unknown' | 'partial' | 'reported'
    current: boolean
    reason?: LoadBufferReason
    sampleId?: string
    recordedAt?: string
    persistence?: 'pending' | 'saved' | 'uncertain' | 'failed' | 'history-unavailable'
    upstreamPin?: '7fe450e19305b828c199d602c23a8337aaa1f03b'
    buildAssurance?: 'recognized-self-reported-pin'
    estimatorRevision?: string
    inputIdentity?: string
    sourceIdentity?: string
    optionsIdentity?: string
    topologyIdentity?: string
    loadPhase?: 'initial' | 'wakeup'
    perSlotContext?: number
    slots?: number
    capturedOptions?: RequestedLoad
    requestedOptions?: RequestedLoad
    optionUnknowns?: Array<
        | 'mmap'
        | 'load-mode'
        | 'flash-attention'
        | 'batch'
        | 'ubatch'
        | 'cache-k'
        | 'cache-v'
        | 'kv-offload'
        | 'op-offload'
        | 'placement'
        | 'speculation'
    >
    artifacts?: ArtifactSummary[]
    baseline?: Array<{
        component: LoadBufferComponent
        status: 'estimated' | 'partial' | 'unknown' | 'excluded'
        bytes?: string
    }>
    comparisons?: Array<{
        component: LoadBufferComponent
        status: 'compatible' | 'unknown'
        estimatedBytes?: string
        lowerBytes?: string
        upperBytes?: string
        deltaLowerBytes?: string
        deltaUpperBytes?: string
    }>
    rows?: Array<{
        component: 'model' | 'kv' | 'compute' | 'output' | 'recurrent'
        label: string
        phase: 'load' | 'reserve'
        sequence: string
        stream?: 'stdout' | 'stderr'
        role: 'model' | 'mixed' | 'unattributed'
        reportedMiB: string
        unit: 'MiB'
        precision: '0.01' | 'unknown'
        sourceKind: 'log-scraped'
        intervalStatus: 'bounded' | 'unknown'
        lowerBytes?: string
        upperBytes?: string
        estimatedBytes?: string
        deviceBinding?: string
        poolIdentity?: string
        coverage: 'partial' | 'unknown' | 'compatible'
    }>
    reasons?: LoadBufferReason[]
}
export type ModelImpact = {
    restartScope: string
    affectedModels?: string[]
    drainTimeoutSeconds?: number
    mayInterruptActive?: boolean
    staysOff?: boolean
    activeWorkPolicy?: string
    unknowns?: string[]
}
export type AutoProposal = {
    status: string
    requested: RequestedLoad
    resolved?: ResolvedLoad
    estimate: Estimate
    memorySource?: string
    memoryTimestamp?: string
    resolverRevision?: string
    reviewedInputId?: string
    warnings?: string[]
}
