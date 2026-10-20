// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Engine status store -- populated entirely by engine state patches.
 * No component ever calls a setter directly. Only push handlers update this store.
 *
 * Structure: outer key = node id, inner key = engine type. Sparse: only types the
 * service has reported appear; use {@link getEnginesForNode} for a full UI row list.
 *
 * It also holds this node's engine descriptors, which say what every engine is
 * (label, family, supported model operations). A peer's engine is looked up in
 * the same set; one absent from it renders with safe defaults.
 */
import { create } from 'zustand'
import type { EngineInitialState, EngineIntentSnapshot } from '@/shared/types/engine-api'
import { emptyEngineStatus } from '@/shared/utils/engines'
import type {
    EngineDescriptor,
    EngineStatusByNode,
    EngineStatusData,
    EngineType
} from '@/shared/types/engines'

function descriptorMap(descriptors: EngineDescriptor[]): Map<EngineType, EngineDescriptor> {
    return new Map(descriptors.map(d => [d.engineType, d]))
}

interface EngineStatusState {
    intent: EngineIntentSnapshot | null
    statusByNode: EngineStatusByNode
    descriptors: Map<EngineType, EngineDescriptor>
    initialize(initial?: EngineInitialState): Promise<void>
    cleanup(): void
    /** Always defined; uses {@link emptyEngineStatus} when the service has not reported this pair. */
    getStatus(nodeId: string, engineType: EngineType): EngineStatusData
}

let unsubs: Array<() => void> = []
let initializationGeneration = 0
let intentGeneration = 0

export const useEngineStatusStore = create<EngineStatusState>((set, get) => ({
    intent: null,
    statusByNode: new Map(),
    descriptors: new Map(),

    initialize: async initialSnapshot => {
        // Idempotent: initialize() runs on connect and again on a
        // state:request-refresh without an intervening cleanup(), so drop any
        // prior subscriptions before re-subscribing or push handlers stack.
        unsubs.forEach(u => u())
        unsubs = []
        const generation = ++initializationGeneration
        const intentRead = ++intentGeneration
        set({ intent: null })
        if (!window.pairApi) return
        unsubs.push(
            window.pairApi.engines.onIntentChanged(intent => {
                if (generation !== initializationGeneration) return
                intentGeneration++
                set({ intent })
            })
        )
        const hydrateIntent = window.pairApi.engines.getIntent().then(
            intent => {
                if (generation !== initializationGeneration || intentRead !== intentGeneration)
                    return
                set({ intent })
            },
            () => {
                if (generation !== initializationGeneration || intentRead !== intentGeneration)
                    return
                set({ intent: null })
            }
        )
        try {
            const initial = initialSnapshot ?? (await window.pairApi.engines.getInitialState())
            if (generation !== initializationGeneration) return
            const byNode: EngineStatusByNode = new Map()
            for (const s of initial.statuses) {
                let inner = byNode.get(s.nodeId)
                if (!inner) {
                    inner = new Map()
                    byNode.set(s.nodeId, inner)
                }
                inner.set(s.engineType, s)
            }
            set({ statusByNode: byNode, descriptors: descriptorMap(initial.descriptors) })
        } catch {
            console.error('Failed to initialize engine status store')
        }

        if (generation !== initializationGeneration) return

        unsubs.push(
            window.pairApi.engines.onDescriptorsChanged(descriptors => {
                set({ descriptors: descriptorMap(descriptors) })
            }),
            window.pairApi.engines.onStateChanged(patch => {
                const status = patch.status
                if (!status) return
                const { nodeId, engineType } = status
                const outer = new Map(get().statusByNode)
                const inner = new Map(outer.get(nodeId) ?? new Map<EngineType, EngineStatusData>())
                inner.set(engineType, status)
                outer.set(nodeId, inner)
                set({ statusByNode: outer })
            }),
            // Drop a departed node's status so a later rejoin never renders stale
            // installed/running state from before it left.
            window.pairApi.nodes.onRemove(nodeId => {
                if (!get().statusByNode.has(nodeId)) return
                const outer = new Map(get().statusByNode)
                outer.delete(nodeId)
                set({ statusByNode: outer })
            })
        )
        await hydrateIntent
    },

    cleanup: () => {
        initializationGeneration++
        intentGeneration++
        unsubs.forEach(u => u())
        unsubs = []
        set({ intent: null })
    },

    getStatus: (nodeId, engineType) => {
        return (
            get().statusByNode.get(nodeId)?.get(engineType) ?? emptyEngineStatus(nodeId, engineType)
        )
    }
}))

/** This node's descriptor for an engine, or undefined when it holds none. */
export function useEngineDescriptor(engineType: EngineType): EngineDescriptor | undefined {
    return useEngineStatusStore(state => state.descriptors.get(engineType))
}
