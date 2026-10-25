// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Engine models store -- populated entirely by engine state patches.
 * No component ever calls a setter directly. Only push handlers update this store.
 */
import { create } from 'zustand'
import type { EngineInitialState } from '@/shared/types/engine-api'
import type { EngineModels, EngineType } from '@/shared/types/engines'
import { emptyEngineModels } from '@/shared/utils/engines'

function modelsKey(nodeId: string, engineType: EngineType): string {
    return `${nodeId}:${engineType}`
}

interface EngineModelsState {
    models: Map<string, EngineModels>
    initialize(initial?: EngineInitialState | Promise<EngineInitialState>): Promise<void>
    cleanup(): void
    /** Always defined; uses {@link emptyEngineModels} when no list has been pushed yet. */
    getModels(nodeId: string, engineType: EngineType): EngineModels
}

let unsubs: Array<() => void> = []
let generation = 0
const retiredSources = new Map<string, Set<string>>()

export const useEngineModelsStore = create<EngineModelsState>((set, get) => ({
    models: new Map(),

    initialize: async initialSnapshot => {
        // Idempotent: drop any prior subscriptions before re-subscribing so a
        // state:request-refresh (which re-runs initialize) never stacks handlers.
        unsubs.forEach(u => u())
        unsubs = []
        const currentGeneration = ++generation
        const pushed = new Set<string>()
        const removedNodes = new Set<string>()
        if (!window.pairApi) return

        const apply = (models: EngineModels, hydrating: boolean): void => {
            const key = modelsKey(models.nodeId, models.engineType)
            const previous = get().models.get(key)
            const before = previous?.downloadState
            const next = models.downloadState
            if (next && retiredSources.get(key)?.has(next.sourceId)) return
            if (before) {
                if (!next) {
                    // Loss of support/currentness cannot leave last-good data ready.
                    models = { ...previous, downloadStateCurrent: false }
                } else if (before.sourceId === next.sourceId) {
                    const older = BigInt(next.revision) < BigInt(before.revision)
                    if (older || (hydrating && next.revision === before.revision)) {
                        if (models.downloadStateCurrent !== true) {
                            models = { ...previous, downloadStateCurrent: false }
                        } else return
                    }
                } else {
                    // A prefetched snapshot cannot order two incarnations. Only a
                    // current observation can replace and retire an accepted source.
                    if (models.downloadStateCurrent !== true) {
                        models = { ...previous, downloadStateCurrent: false }
                    } else {
                        if (hydrating) return
                        const retired = retiredSources.get(key) ?? new Set<string>()
                        retired.add(before.sourceId)
                        retiredSources.set(key, retired)
                    }
                }
            }
            const map = new Map(get().models)
            map.set(key, models)
            set({ models: map })
        }
        unsubs.push(
            window.pairApi.engines.onStateChanged(patch => {
                const models = patch.models
                if (!models) return
                const key = modelsKey(models.nodeId, models.engineType)
                pushed.add(key)
                apply(models, false)
            }),
            window.pairApi.engines.onProgress(progress => {
                if (
                    !progress.downloadState ||
                    !progress.downloadSourceId ||
                    !progress.downloadRevision
                )
                    return
                const key = modelsKey(progress.nodeId, progress.engineType)
                const previous = get().models.get(key)
                const before = previous?.downloadState
                if (!before || retiredSources.get(key)?.has(progress.downloadSourceId)) return
                if (before.sourceId !== progress.downloadSourceId) {
                    if (progress.downloadStateCurrent !== true) return
                    const retired = retiredSources.get(key) ?? new Set<string>()
                    retired.add(before.sourceId)
                    retiredSources.set(key, retired)
                } else if (
                    BigInt(progress.downloadRevision) <= BigInt(before.revision) &&
                    progress.downloadStateCurrent === true
                )
                    return
                pushed.add(key)
                const map = new Map(get().models)
                map.set(key, { ...previous, downloadStateCurrent: false })
                set({ models: map })
            }),
            // Drop a departed node's model lists so a rejoin never shows stale
            // models from before it left.
            window.pairApi.nodes.onRemove(nodeId => {
                removedNodes.add(nodeId)
                const prefix = `${nodeId}:`
                const map = new Map(get().models)
                let changed = false
                for (const key of Array.from(map.keys())) {
                    if (key.startsWith(prefix)) {
                        map.delete(key)
                        changed = true
                    }
                }
                if (changed) set({ models: map })
            })
        )
        try {
            const initial = await (initialSnapshot ?? window.pairApi.engines.getInitialState())
            if (currentGeneration !== generation) return
            for (const models of initial.models) {
                if (removedNodes.has(models.nodeId)) continue
                const key = modelsKey(models.nodeId, models.engineType)
                // An internal fetch may restore freshness at the same revision
                // when no push superseded it. Supplied snapshots have no such order.
                apply(
                    models,
                    (initialSnapshot !== undefined && !(initialSnapshot instanceof Promise)) ||
                        pushed.has(key)
                )
            }
        } catch {
            console.error('Failed to initialize engine models store')
        }
    },

    cleanup: () => {
        generation++
        unsubs.forEach(u => u())
        unsubs = []
        set({
            models: new Map(
                Array.from(get().models, ([key, models]) => [
                    key,
                    { ...models, downloadStateCurrent: false }
                ])
            )
        })
    },

    getModels: (nodeId, engineType) => {
        return (
            get().models.get(modelsKey(nodeId, engineType)) ?? emptyEngineModels(nodeId, engineType)
        )
    }
}))
