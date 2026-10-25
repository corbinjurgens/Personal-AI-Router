// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Engine progress domain -- delivered from the backend as progress push events.
 * Unified progress channel for all operations: install, pull, load, etc.
 * Keyed by ${nodeId}:${engineType}:${operation}:${model?} for concurrent ops.
 */

import { EngineOperationType, EngineProgress, EngineType } from '@/shared/types/engines'

/** Build the map key for an EngineProgress entry. */
export function engineProgressKey(p: {
    nodeId: string
    engineType: EngineType
    operation: string
    model?: string
    opId?: string
    sourceNodeId?: string
    destinationNodeId?: string
}): string {
    if (p.operation === 'copy' && p.opId) {
        return `${p.nodeId}:copy:${JSON.stringify([p.opId, p.sourceNodeId, p.destinationNodeId, p.engineType, p.model])}`
    }
    return p.model
        ? `${p.nodeId}:${p.engineType}:${p.operation}:${p.model}`
        : `${p.nodeId}:${p.engineType}:${p.operation}`
}

const emptyCache = new Map<string, EngineProgress>()

/**
 * Stable placeholder when no progress entry exists for this key.
 * Returns the same reference for the same key so downstream `memo`
 * components don't re-render when nothing has changed.
 */
export function emptyEngineProgress(at: {
    nodeId: string
    engineType: EngineType
    operation: EngineOperationType
    model?: string
}): EngineProgress {
    const key = engineProgressKey(at)
    const cached = emptyCache.get(key)
    if (cached) return cached
    const entry: EngineProgress = {
        engineType: at.engineType,
        nodeId: at.nodeId,
        nodeName: '',
        operation: at.operation,
        model: at.model,
        status: 'idle'
    }
    emptyCache.set(key, entry)
    return entry
}

/** Whether this exact model copy operation has not ended. */
export function isEngineCopyInProgress(
    p: Pick<EngineProgress, 'operation' | 'status'> | undefined
): boolean {
    return (
        p?.operation === 'copy' &&
        !['idle', 'done', 'error', 'cancelled', 'unknown'].includes(p.status)
    )
}

/** Whether UI should treat this progress as an active pull (show cancel, spinner, etc.). */
export function isEnginePullInProgress(
    p: Pick<EngineProgress, 'operation' | 'status'> | undefined
): boolean {
    if (!p || p.operation !== 'pull') return false
    if (p.status === 'idle') return false
    const s = p.status.toLowerCase()
    return s !== 'complete' && s !== 'error' && s !== 'failed' && s !== 'unknown'
}

/** Bridge observations are source-fenced; preserve their revision order in renderer hydration too. */
export function shouldApplyEngineProgress(
    next: EngineProgress,
    previous?: EngineProgress
): boolean {
    if (previous?.operation === 'copy' && next.operation === 'copy') {
        if (
            previous.opId !== next.opId ||
            previous.sourceNodeId !== next.sourceNodeId ||
            previous.destinationNodeId !== next.destinationNodeId ||
            previous.model !== next.model ||
            previous.engineType !== next.engineType
        )
            return false
        // A late snapshot/progress frame must not revive settled work or erase
        // its verified setup/durability qualification. Unknown may hydrate active.
        if (['done', 'error', 'cancelled'].includes(previous.status)) return false
    }
    if (!previous?.downloadState) return true
    if (!next.downloadState) return false
    if (next.downloadSourceId !== previous.downloadSourceId) return true
    if (!next.downloadRevision || !previous.downloadRevision) return false
    const revision = BigInt(next.downloadRevision)
    const baseline = BigInt(previous.downloadRevision)
    return (
        revision > baseline ||
        (revision === baseline && next.downloadState.id === previous.downloadState.id)
    )
}
