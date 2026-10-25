// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Engine progress store -- ephemeral, auto-clearing.
 * Populated entirely by engine progress push events. Entries are removed when
 * the service sends a progress-cleared event (operation complete/error).
 *
 * On reconnect, the initial state snapshot includes active progress entries
 * so in-flight operations resume displaying immediately.
 */
import { create } from 'zustand'
import type { EngineInitialState } from '@/shared/types/engine-api'
import type {
    EngineModels,
    EngineOperationType,
    EngineProgress,
    EngineType
} from '@/shared/types/engines'
import {
    emptyEngineProgress,
    engineProgressKey,
    isEngineCopyInProgress,
    shouldApplyEngineProgress
} from '@/shared/utils/engine-progress'

interface EngineProgressState {
    progress: Map<string, EngineProgress>
    initialize(initial?: EngineInitialState | Promise<EngineInitialState>): Promise<void>
    cleanup(): void
    /**
     * Always defined; uses {@link emptyEngineProgress} with `status: 'idle'` when no entry exists.
     */
    getProgress(
        nodeId: string,
        engineType: EngineType,
        operation: EngineOperationType,
        model?: string
    ): EngineProgress
    getProgressForNode(nodeId: string): EngineProgress[]
}

let unsubs: Array<() => void> = []
let generation = 0
const retiredSources = new Map<string, Set<string>>()
const sources = new Map<string, string>()

export const useEngineProgressStore = create<EngineProgressState>((set, get) => ({
    progress: new Map(),

    initialize: async initialSnapshot => {
        // Idempotent: drop any prior subscriptions before re-subscribing so a
        // state:request-refresh (which re-runs initialize) never stacks handlers.
        unsubs.forEach(u => u())
        unsubs = []
        const currentGeneration = ++generation
        const pushed = new Set<string>()
        const removedNodes = new Set<string>()
        const pushedModels = new Set<string>()
        let hydrating = true

        if (!window.pairApi) return

        const acceptSource = (
            nodeId: string,
            engineType: EngineType,
            source: string,
            canReplace: boolean
        ): boolean => {
            const key = `${nodeId}:${engineType}`
            if (retiredSources.get(key)?.has(source)) return false
            const previous = sources.get(key)
            if (previous && previous !== source) {
                if (!canReplace) return false
                const retired = retiredSources.get(key) ?? new Set<string>()
                retired.add(previous)
                retiredSources.set(key, retired)
            }
            sources.set(key, source)
            return true
        }
        const apply = (progress: EngineProgress, fromSnapshot: boolean): void => {
            const key = engineProgressKey(progress)
            const previous = get().progress.get(key)
            if (
                progress.downloadSourceId &&
                !acceptSource(
                    progress.nodeId,
                    progress.engineType,
                    progress.downloadSourceId,
                    !fromSnapshot && progress.downloadStateCurrent === true
                )
            )
                return
            if (previous?.downloadState && progress.downloadState) {
                if (previous.downloadSourceId !== progress.downloadSourceId) {
                    if (fromSnapshot || progress.downloadStateCurrent !== true) return
                } else if (
                    fromSnapshot &&
                    previous.downloadRevision === progress.downloadRevision &&
                    progress.downloadStateCurrent !== false
                )
                    return
            }
            if (!shouldApplyEngineProgress(progress, previous)) {
                if (progress.downloadStateCurrent !== false || !previous) return
                progress = { ...previous, downloadStateCurrent: false }
            }
            const map = new Map(get().progress)
            map.set(key, progress)
            set({ progress: map })
        }
        const observeModels = (models: EngineModels, fromSnapshot: boolean): void => {
            const snapshot = models.downloadState
            if (
                snapshot &&
                retiredSources.get(`${models.nodeId}:${models.engineType}`)?.has(snapshot.sourceId)
            )
                return
            if (
                models.downloadStateCurrent === true &&
                (!snapshot ||
                    !acceptSource(
                        models.nodeId,
                        models.engineType,
                        snapshot.sourceId,
                        !fromSnapshot && models.downloadStateCurrent === true
                    ))
            )
                return
            const map = new Map(get().progress)
            for (const [key, p] of map) {
                if (
                    p.nodeId !== models.nodeId ||
                    p.engineType !== models.engineType ||
                    !p.downloadState
                )
                    continue
                if (models.downloadStateCurrent !== true) {
                    map.set(key, { ...p, downloadStateCurrent: false })
                } else if (
                    snapshot &&
                    (p.downloadSourceId !== snapshot.sourceId ||
                        (p.downloadRevision &&
                            BigInt(p.downloadRevision) <= BigInt(snapshot.revision)))
                ) {
                    // A current canonical snapshot supersedes an older lifecycle
                    // fence; omission from activeProgress alone never does.
                    map.delete(key)
                }
            }
            set({ progress: map })
        }
        unsubs.push(
            window.pairApi.engines.onStateChanged(patch => {
                if (patch.models) {
                    if (hydrating) pushedModels.add(`${patch.nodeId}:${patch.engineType}`)
                    observeModels(patch.models, false)
                }
            }),
            window.pairApi.engines.onProgress(progress => {
                const key = engineProgressKey(progress)
                if (hydrating) pushed.add(key)
                apply(progress, false)
            }),
            window.pairApi.engines.onProgressRemove(key => {
                if (hydrating) pushed.add(key)
                if (get().progress.get(key)?.downloadState) return
                const map = new Map(get().progress)
                if (map.delete(key)) {
                    set({ progress: map })
                }
            }),
            // Drop a departed node's in-flight progress entries so a stuck spinner
            // (e.g. a remote pull whose terminal was missed) can't linger after the
            // node is gone.
            window.pairApi.nodes.onRemove(nodeId => {
                removedNodes.add(nodeId)
                const prefix = `${nodeId}:`
                const map = new Map(get().progress)
                let changed = false
                for (const [key, progress] of map) {
                    if (
                        progress.operation === 'copy' &&
                        (progress.sourceNodeId === nodeId || progress.destinationNodeId === nodeId)
                    ) {
                        if (isEngineCopyInProgress(progress)) {
                            map.set(key, {
                                ...progress,
                                status: 'unknown',
                                message:
                                    'Copy outcome unknown after device disconnection. Reconnect and verify with an identity-bound retry when available.'
                            })
                            changed = true
                        }
                        continue
                    }
                    if (key.startsWith(prefix)) {
                        map.delete(key)
                        changed = true
                    }
                }
                if (changed) set({ progress: map })
            })
        )
        try {
            const initial = await (initialSnapshot ?? window.pairApi.engines.getInitialState())
            if (currentGeneration !== generation) return
            // activeProgress may omit terminal lifecycle entries. Keep their
            // revision fence until canonical model observations supersede it.
            const map = new Map(
                Array.from(get().progress).filter(
                    ([key, p]) => p.downloadState || p.operation === 'copy' || pushed.has(key)
                )
            )
            set({ progress: map })
            for (const models of initial.models) {
                if (!removedNodes.has(models.nodeId))
                    observeModels(
                        models,
                        (initialSnapshot !== undefined && !(initialSnapshot instanceof Promise)) ||
                            pushedModels.has(`${models.nodeId}:${models.engineType}`)
                    )
            }
            for (const p of initial.activeProgress) {
                if (removedNodes.has(p.nodeId)) continue
                const key = engineProgressKey(p)
                if (!pushed.has(key))
                    apply(p, initialSnapshot !== undefined && !(initialSnapshot instanceof Promise))
            }
            const observedCopies = new Set(initial.activeProgress.map(engineProgressKey))
            const reconciled = new Map(get().progress)
            for (const [key, progress] of reconciled) {
                if (
                    isEngineCopyInProgress(progress) &&
                    !pushed.has(key) &&
                    !observedCopies.has(key)
                ) {
                    reconciled.set(key, {
                        ...progress,
                        status: 'unknown',
                        message:
                            'Copy outcome unknown after reconnect. Retry with the export identity when available.'
                    })
                }
            }
            set({ progress: reconciled })
        } catch {
            console.error('Failed to initialize engine progress store')
        } finally {
            hydrating = false
            pushed.clear()
        }
    },

    cleanup: () => {
        generation++
        unsubs.forEach(u => u())
        unsubs = []
        set({
            progress: new Map(
                Array.from(get().progress, ([key, progress]) => [
                    key,
                    progress.downloadState ? { ...progress, downloadStateCurrent: false } : progress
                ])
            )
        })
    },

    getProgress: (nodeId, engineType, operation, model) => {
        return (
            get().progress.get(engineProgressKey({ nodeId, engineType, operation, model })) ??
            emptyEngineProgress({ nodeId, engineType, operation, model })
        )
    },

    getProgressForNode: nodeId => {
        const results: EngineProgress[] = []
        for (const [key, p] of get().progress) {
            if (key.startsWith(`${nodeId}:`)) results.push(p)
        }
        return results
    }
}))
