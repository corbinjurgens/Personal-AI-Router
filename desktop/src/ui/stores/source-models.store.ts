// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Source models store -- models each node serves in place from Ollama, LM
 * Studio or linked files.
 *
 * The engine manager owns the list; this store only mirrors its last
 * `engine:source-models` answer per node. Sources are not pushed, so the
 * Models tab refreshes on mount and after a link or unlink.
 */
import { create } from 'zustand'
import type { SourceModel } from '@/shared/types/engine-api'
import getErrorString from '@/shared/utils/get-error-string'

/** The only engine that serves source models today. */
export const SOURCE_MODELS_ENGINE = 'llamacpp'

interface SourceModelsEntry {
    models: SourceModel[]
    loading: boolean
    error: string
}

interface SourceModelsState {
    entries: Map<string, SourceModelsEntry>
    refresh(nodeId: string): Promise<void>
    refreshAll(nodeIds: readonly string[]): Promise<void>
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
                const res = await window.pairApi.engines.sourceModels(nodeId, SOURCE_MODELS_ENGINE)
                update(nodeId, { models: res.models, loading: false, error: res.error ?? '' })
            } catch (err) {
                update(nodeId, { loading: false, error: getErrorString(err) })
            }
        },
        refreshAll: async nodeIds => {
            await Promise.all(nodeIds.map(nodeId => get().refresh(nodeId)))
        }
    }
})
