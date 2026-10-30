// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Control metadata only. Engine is the manager ID, never a display label. */
export interface ModelCopyRequest {
    opId: string
    sourceNodeId: string
    destinationNodeId: string
    engine: string
    model: string
    expectedExportId?: string
}

export interface ModelCopyProgress {
    opId: string
    /** Legacy wire attribution remains the source. */
    node: string
    sourceNodeId: string
    destinationNodeId: string
    engine: string
    model: string
    op: 'copy'
    /** Terminal stages: done, error, cancelled, unknown (transport/acknowledgement loss). */
    stage: string
    percent?: number
    message?: string
    exportId?: string
    file?: string
    bytesDone?: number
    bytesTotal?: number
}

export interface ActiveModelCopy extends ModelCopyRequest {
    progress?: ModelCopyProgress
}

/** Byte agreement with an independently acquired public provider statement.
 * This says nothing about model safety, author trust or a signature. */
export interface ModelPublisherFile {
    path: string
    status: 'verified' | 'mismatch' | 'unavailable'
    provider?: 'huggingface'
    repository?: string
    revision?: string
    file?: string
    algorithm: 'sha256'
    expectedDigest?: string
    actualDigest: string
    acquisitionBasis: 'provider' | 'cache' | 'unavailable'
    acquiredAt?: string
}

export interface ModelPublisherComparison {
    /** Mismatch wins; verified requires every file; otherwise unavailable. */
    status: 'verified' | 'mismatch' | 'unavailable'
    files: ModelPublisherFile[]
    /** Local receipt replay retains the original acquisition basis and time. */
    replayed?: boolean
}

/** Verified artifact copy, independently of native inventory or serving readiness. */
export interface ModelCopyResult {
    engine: string
    model: string
    files: number
    filesSkipped: number
    /** Total verified artifact bytes; also the completed progress value on success. */
    bytesTotal: number
    /** Network bytes including retries; may exceed bytesTotal. */
    bytesCopied: number
    layout?: string
    exportId?: string
    durability?: 'confirmed' | 'uncertain'
    requiresConfiguration?: boolean
    publisherComparison?: ModelPublisherComparison
}

export interface ModelCopyResponse {
    opId: string
    result: ModelCopyResult
}

export interface ModelCopyCapabilities {
    version: 1
    engine: string
    available: boolean
    installed: boolean
    modelStore: string
    exportLayouts: string[]
    importLayouts: string[]
    writable: boolean
    reason: string
    identityBoundRetry: boolean
    receiptDurability: boolean
}
