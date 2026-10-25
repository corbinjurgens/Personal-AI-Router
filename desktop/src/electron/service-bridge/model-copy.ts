// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineProgress } from '@/shared/types/engines'
import type {
    ActiveModelCopy,
    ModelCopyCapabilities,
    ModelCopyProgress,
    ModelCopyRequest,
    ModelCopyResponse
} from '@/shared/types/model-copy'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import type { JsonObject, JsonValue } from './json-rpc-client'

function invalid(): never {
    throw new Error('Invalid model copy response')
}
function object(value: JsonValue | undefined): JsonObject {
    return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : invalid()
}
function text(value: JsonValue | undefined): string {
    return typeof value === 'string' ? value : invalid()
}
function nonempty(value: JsonValue | undefined): string {
    const result = text(value)
    return result.length > 0 ? result : invalid()
}
function flag(value: JsonValue | undefined): boolean {
    return typeof value === 'boolean' ? value : invalid()
}
function count(value: JsonValue | undefined): number {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
        ? value
        : invalid()
}
function strings(value: JsonValue | undefined): string[] {
    if (!Array.isArray(value)) return invalid()
    return value.map(nonempty)
}
function optionalText(value: JsonValue | undefined): string | undefined {
    return value === undefined ? undefined : nonempty(value)
}
function exportIdentity(value: JsonValue | undefined): string | undefined {
    const id = optionalText(value)
    return id === undefined || /^[a-f0-9]{64}$/.test(id) ? id : invalid()
}
function operation(value: JsonValue | undefined): string {
    const id = nonempty(value)
    return /^[a-f0-9]{32}$/.test(id) ? id : invalid()
}
function requestFrom(value: JsonValue | undefined): ModelCopyRequest {
    const row = object(value)
    const engine = nonempty(row.engine)
    if (!engineTypeFromManagerName(engine)) return invalid()
    const result: ModelCopyRequest = {
        opId: operation(row.opId),
        sourceNodeId: nonempty(row.sourceNodeId),
        destinationNodeId: nonempty(row.destinationNodeId),
        engine,
        model: nonempty(row.model)
    }
    if (result.sourceNodeId === result.destinationNodeId) return invalid()
    const expected = exportIdentity(row.expectedExportId)
    if (expected !== undefined) result.expectedExportId = expected
    return result
}

export function sameModelCopy(a: ModelCopyRequest, b: ModelCopyRequest): boolean {
    return (
        a.opId === b.opId &&
        a.sourceNodeId === b.sourceNodeId &&
        a.destinationNodeId === b.destinationNodeId &&
        a.engine === b.engine &&
        a.model === b.model
    )
}

/** Never backfill identity from a source/engine pair: destinations may be concurrent. */
export function parseModelCopyFrame(value: JsonValue | undefined): ModelCopyProgress | null {
    try {
        const row = object(value)
        const request = requestFrom(value)
        if (row.op !== 'copy' || row.node !== request.sourceNodeId) return null
        const percent =
            typeof row.percent === 'number' && Number.isFinite(row.percent) && row.percent >= 0
                ? row.percent
                : undefined
        return {
            opId: request.opId,
            sourceNodeId: request.sourceNodeId,
            destinationNodeId: request.destinationNodeId,
            engine: request.engine,
            model: request.model,
            node: request.sourceNodeId,
            op: 'copy',
            stage: nonempty(row.stage),
            percent: percent === undefined ? undefined : Math.min(percent, 100),
            message: optionalText(row.message),
            exportId: exportIdentity(row.exportId),
            file: optionalText(row.file),
            bytesDone: row.bytesDone === undefined ? undefined : count(row.bytesDone),
            bytesTotal: row.bytesTotal === undefined ? undefined : count(row.bytesTotal)
        }
    } catch {
        return null
    }
}

export function isModelCopyTerminal(stage: string): boolean {
    return stage === 'done' || stage === 'error' || stage === 'cancelled'
}

export function modelCopyProgress(frame: ModelCopyProgress, nodeName: string): EngineProgress {
    const engineType = engineTypeFromManagerName(frame.engine)
    if (!engineType) return invalid()
    return {
        engineType,
        nodeId: frame.destinationNodeId,
        nodeName,
        operation: 'copy',
        opId: frame.opId,
        sourceNodeId: frame.sourceNodeId,
        destinationNodeId: frame.destinationNodeId,
        model: frame.model,
        status: frame.stage,
        percent: frame.percent,
        completed: frame.bytesDone,
        total: frame.bytesTotal,
        exportId: frame.exportId,
        file: frame.file,
        message: frame.message
    }
}

export function parseModelCopyResult(
    value: JsonValue | undefined,
    request: ModelCopyRequest
): ModelCopyResponse {
    const row = object(value)
    const result = object(row.result)
    if (
        row.opId !== request.opId ||
        result.engine !== request.engine ||
        result.model !== request.model
    )
        return invalid()
    const durability = result.durability
    if (durability !== undefined && durability !== 'confirmed' && durability !== 'uncertain')
        return invalid()
    const exportId = exportIdentity(result.exportId)
    if (request.expectedExportId && exportId !== request.expectedExportId) return invalid()
    const layout = optionalText(result.layout)
    const requiresConfiguration =
        result.requiresConfiguration === undefined ? undefined : flag(result.requiresConfiguration)
    if (
        layout === 'llama-artifacts-v1' &&
        (!exportId || !durability || requiresConfiguration !== true)
    )
        return invalid()
    const files = count(result.files)
    const filesSkipped = count(result.filesSkipped)
    const bytesTotal = count(result.bytesTotal)
    const bytesCopied = count(result.bytesCopied)
    if (filesSkipped > files || bytesCopied > bytesTotal) return invalid()
    return {
        opId: request.opId,
        result: {
            engine: request.engine,
            model: request.model,
            files,
            filesSkipped,
            bytesTotal,
            bytesCopied,
            layout,
            exportId,
            durability,
            requiresConfiguration
        }
    }
}

export function parseModelCopyCapabilities(
    value: JsonValue | undefined,
    engine: string
): ModelCopyCapabilities {
    const row = object(value)
    if (row.version !== 1 || row.engine !== engine) return invalid()
    return {
        version: 1,
        engine,
        available: flag(row.available),
        installed: flag(row.installed),
        modelStore: text(row.modelStore),
        exportLayouts: strings(row.exportLayouts),
        importLayouts: strings(row.importLayouts),
        writable: flag(row.writable),
        reason: text(row.reason),
        identityBoundRetry: flag(row.identityBoundRetry),
        receiptDurability: flag(row.receiptDurability)
    }
}

export function parseActiveModelCopies(value: JsonValue | undefined): ActiveModelCopy[] {
    const rows = object(value).operations
    if (!Array.isArray(rows) || rows.length > 8) return invalid()
    const ids = new Set<string>()
    return rows.map(value => {
        const row = object(value)
        const request = requestFrom(value)
        if (ids.has(request.opId)) return invalid()
        ids.add(request.opId)
        if (row.progress === undefined) return request
        const progress = parseModelCopyFrame(row.progress)
        if (!progress || !sameModelCopy(progress, request)) return invalid()
        return { ...request, progress }
    })
}

export function parseModelCopyCancel(value: JsonValue | undefined): { cancelled: boolean } {
    return { cancelled: flag(object(value).cancelled) }
}
