// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineProgress, EngineType } from '@/shared/types/engines'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import type { JsonValue } from './json-rpc-client'

/** One `engine:remote-progress` frame with `op: "copy"`, as it applies to this PC. */
interface ModelCopyFrame {
    /** The source peer the model is copied from. */
    nodeId: string
    engineType: EngineType
    stage: string
    percent?: number
    file?: string
    bytesDone?: number
    bytesTotal?: number
    message?: string
}

function optionalString(value: JsonValue | undefined): string | undefined {
    return typeof value === 'string' && value !== '' ? value : undefined
}

function optionalNumber(value: JsonValue | undefined): number | undefined {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

/** Parse a copy frame; null when it does not name a source node and a known engine. */
export function parseModelCopyFrame(params: JsonValue | undefined): ModelCopyFrame | null {
    if (params === null || typeof params !== 'object' || Array.isArray(params)) return null
    const nodeId = optionalString(params.node)
    const engineName = optionalString(params.engine)
    const engineType = engineName ? engineTypeFromManagerName(engineName) : null
    if (!nodeId || !engineType) return null
    const percent = optionalNumber(params.percent)
    return {
        nodeId,
        engineType,
        stage: optionalString(params.stage) ?? 'working',
        percent: percent === undefined ? undefined : Math.min(percent, 100),
        file: optionalString(params.file),
        bytesDone: optionalNumber(params.bytesDone),
        bytesTotal: optionalNumber(params.bytesTotal),
        message: optionalString(params.message)
    }
}

/** Whether a stage ends the copy, successfully (`done`) or not (`error`). */
export function isModelCopyTerminal(stage: string): boolean {
    return stage === 'done' || stage === 'error'
}

/** The progress entry a non-terminal copy frame shows, keyed on the source node's model row. */
export function modelCopyProgress(
    frame: ModelCopyFrame,
    model: string,
    nodeName: string
): EngineProgress {
    return {
        engineType: frame.engineType,
        nodeId: frame.nodeId,
        nodeName,
        operation: 'copy',
        model,
        status: frame.stage,
        percent: frame.percent,
        completed: frame.bytesDone,
        total: frame.bytesTotal
    }
}
