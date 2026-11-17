// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Source models store -- models each node serves in place from Ollama, LM
 * Studio or linked files, and the copies in its store that native inventory
 * does not list.
 *
 * The engine manager owns the list; this store only mirrors its last
 * `engine:source-models` answer per node. Sources are not pushed, so the
 * Models tab refreshes on mount and after a link, unlink or copy. A delete
 * runs through the fire-and-forget engine command, so its row is dropped
 * here when it is sent; a refused delete reports an error and the row returns
 * on the next refresh.
 */
import { create } from 'zustand'
import type { SourceModel } from '@/shared/types/engine-api'
import type { EngineType } from '@/shared/types/engines'
import getErrorString from '@/shared/utils/get-error-string'

/** The only engine that serves source models today. */
export const SOURCE_MODELS_ENGINE_TYPE: EngineType = 'llama-cpp'

interface SourceModelsEntry {
    models: SourceModel[]
    loading: boolean
    error: string
}

interface SourceModelsState {
    entries: Map<string, SourceModelsEntry>
    refresh(nodeId: string): Promise<void>
    refreshAll(nodeIds: readonly string[]): Promise<void>
    /** Drop a row whose delete was sent. */
    forget(nodeId: string, id: string): void
}

export const useSourceModelsStore = create<SourceModelsState>((set, get) => {
    const update = (nodeId: string, patch: Partial<SourceModelsEntry>) =>
        set(state => {
            const entries = new Map(state.entries)
            const previous = entries.get(nodeId) ?? { models: [], loading: false, error: '' }
            entries.set(nodeId, { ...previous, ...patch })
            return { entries }
        })
    return {
        entries: new Map(),
        refresh: async nodeId => {
            update(nodeId, { loading: true })
            try {
                const res = await window.pairApi.engines.sourceModels(
                    nodeId,
                    SOURCE_MODELS_ENGINE_TYPE
                )
                update(nodeId, { models: res.models, loading: false, error: res.error ?? '' })
            } catch (err) {
                update(nodeId, { loading: false, error: getErrorString(err) })
            }
        },
        refreshAll: async nodeIds => {
            await Promise.all(nodeIds.map(nodeId => get().refresh(nodeId)))
        },
        forget: (nodeId, id) => {
            const entry = get().entries.get(nodeId)
            if (entry) update(nodeId, { models: entry.models.filter(model => model.id !== id) })
        }
    }
})
