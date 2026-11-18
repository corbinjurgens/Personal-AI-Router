// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EngineProgress, EngineType } from '@/shared/types/engines'
import type { ModelDownloadEntry } from '@/shared/types/model-download'
import { engineProgressKey } from '@/shared/utils/engine-progress'

export interface ModelDownload {
    /** The progress-store key of the model's pull. */
    key: string
    nodeId: string
    engineType: EngineType
    model: string
    /** The node's current download entry; absent while `starting`. */
    entry?: ModelDownloadEntry
    /** A download was requested and the node has not reported it yet. */
    starting: boolean
}

export function isDownloadActive(download: ModelDownload | undefined): boolean {
    return (
        download?.starting === true ||
        download?.entry?.phase === 'queued' ||
        download?.entry?.phase === 'in_progress'
    )
}

/**
 * Every model download the renderer knows of, one per node, engine and model:
 * the progress store's pull entries (projected from each node's current
 * download state, outcomes included) and, from the click until the node's
 * first report, the pending pull that started it.
 */
export function listModelDownloads(
    progress: ReadonlyMap<string, EngineProgress>,
    pending: Iterable<{ action: string; nodeId: string; engineType: EngineType; model?: string }>
): ModelDownload[] {
    const downloads = new Map<string, ModelDownload>()
    for (const [key, item] of progress) {
        if (item.operation !== 'pull' || item.downloadStateCurrent !== true) continue
        if (!item.downloadState || !item.model) continue
        downloads.set(key, {
            key,
            nodeId: item.nodeId,
            engineType: item.engineType,
            model: item.model,
            entry: item.downloadState,
            starting: false
        })
    }
    for (const action of pending) {
        if (action.action !== 'pullModel' || !action.model) continue
        const { nodeId, engineType, model } = action
        const key = engineProgressKey({ nodeId, engineType, operation: 'pull', model })
        if (isDownloadActive(downloads.get(key))) continue
        downloads.set(key, { key, nodeId, engineType, model, starting: true })
    }
    return [...downloads.values()]
}
