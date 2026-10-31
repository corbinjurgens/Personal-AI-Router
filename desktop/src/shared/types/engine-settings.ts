// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    AutoProposal,
    Estimate,
    ArtifactSummary,
    ModelChange,
    ModelImpact,
    ModelLoad
} from '@/shared/types/model-config'
import type { JsonValue } from '@/shared/types/json'

export interface EngineSettingsConfig {
    serverPort: number
    proxyPort: number
    /** Arguments and leading environment assignments, without an executable/subcommand. */
    launchText: string
    workingSetLimitMiB?: number
    serving?: EngineServingConfig
    modelLoads?: ModelLoad[]
}

type ServingMemory = {
    hostRamBytes: number
    vramBytes: number
}

type ServingArtifact = {
    role: string
    sha256: string
    provenance: { kind: string; reference?: string }
    binding?: {
        sourcePath: string
        resolvedTarget: string
        sizeBytes: number
        /** Decimal int64 text, preserving nanoseconds without rounding. */
        mtimeNs: string
        sampleSha256: string
    }
}

export type EngineServingConfig = {
    version: 1
    models: Array<{
        modelId: string
        maxOutputTokens: number
        memory: ServingMemory
        artifacts: ServingArtifact[]
        expectedNativeSourceDigest?: string
    }>
}

type ServingEnvelope = {
    modelId: string
    perSlotContextTokens: number
    totalContextTokens: number
    slots: number
    maxOutputTokens: number
    memory: ServingMemory
    nativeSourceDigest: string
    placement?: string[]
    retention?: string
    routerCapacity?: number
}

export type EngineServingReadback = {
    status:
        | 'unconfigured'
        | 'unverified'
        | 'configured'
        | 'launchValidated'
        | 'observed'
        | 'mismatch'
    reason?: string
    configured?: ServingEnvelope[]
    launchValidated?: ServingEnvelope[]
    observed?: Array<{ modelId: string; perSlotContextTokens: number; slots: number }>
}

import type { EngineManagerName } from '@/shared/types/engines'

export interface EngineSettingsTarget {
    nodeId: string
    /** Engine-manager spelling — the settings contract is relayed verbatim. */
    engine: EngineManagerName
}

export interface EngineSettingsRequest extends EngineSettingsTarget {
    /** Existing engine counter, restricted to nonnegative safe JSON integers. */
    expectedRevision: number
    requestId?: string
    settings: EngineSettingsConfig
    resolution?: 'server' | 'launch'
    modelChange?: ModelChange
}

export interface EngineSettingsSnapshot extends EngineSettingsTarget {
    workingSetLimitStatus?: 'applied' | 'unverified' | 'not-running' | 'unenforced-adopted'
    serving?: EngineServingReadback
    revision: number
    appliedRevision: number
    sequence: number
    epoch: string
    settings: EngineSettingsConfig
    effectiveServerPort: number
    effectiveProxyPort: number
    running: boolean
    adopted: boolean
    editable: boolean
    reason: string
    format: string
    phase: 'idle' | 'applying' | 'succeeded' | 'failed'
    error: string
    requestId: string
    coupledStage?: string
    policyRevision?: string
    unresolvedError?: string
    effectiveSettings?: EngineSettingsConfig
}

export interface EngineSettingsPreview {
    serving?: EngineServingReadback
    settings: EngineSettingsConfig
    revision: number
    conflict?: { serverPort: number; launchPort: number }
    errors: Record<string, string>
    restart: boolean
    rebind: boolean
    policyRevision?: string
    previewDigest?: string
    diff?: Array<{ field: string; before?: JsonValue; after?: JsonValue }>
    requiredArtifacts?: ArtifactSummary[]
    estimate?: Estimate
    impact?: ModelImpact
    auto?: AutoProposal
}

export interface EngineSettingsReceipt {
    revision: number
    phase: 'applying' | 'succeeded' | 'failed'
}
