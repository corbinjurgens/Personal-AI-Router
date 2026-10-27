// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NodeItem } from '@/shared/types/nodes'
import deepEqual from '@/ui/utils/deep-equal'
import { create } from 'zustand'

interface NodesStore {
    nodes: Map<string, NodeItem>
    fetchedNodes: boolean
    initialize: () => Promise<void>
    /** Re-fetch node list from service (e.g. after state reset / leave cluster). Does not add new listeners. */
    refresh: () => Promise<void>
    cleanup: () => void
    clearNodes: (selfId: string) => void
}

let unsubs: Array<() => void> = []
let generation = 0
let readGeneration = 0
const reads = new Set<{ changes: Map<string, NodeItem | null>; sawUpsert: boolean }>()

export const useNodesStore = create<NodesStore>((set, get) => ({
    nodes: new Map(),
    fetchedNodes: false,

    initialize: async () => {
        get().cleanup()
        const current = generation
        if (window.pairApi) {
            unsubs.push(
                window.pairApi.nodes.onUpsert((node: NodeItem) => {
                    if (current !== generation) return
                    for (const read of reads) {
                        read.changes.set(node.id, node)
                        read.sawUpsert = true
                    }
                    const prev = get().nodes
                    const existing = prev.get(node.id)
                    if (existing && deepEqual(existing, node)) return
                    const updated = new Map(prev)
                    updated.set(node.id, node)
                    set({ nodes: updated, fetchedNodes: true })
                }),
                window.pairApi.nodes.onRemove((nodeId: string) => {
                    if (current !== generation) return
                    for (const read of reads) read.changes.set(nodeId, null)
                    const prev = get().nodes
                    if (!prev.has(nodeId)) return
                    const updated = new Map(prev)
                    updated.delete(nodeId)
                    set({ nodes: updated })
                })
            )
        }
        await get().refresh()
    },

    refresh: async () => {
        const current = generation
        const read = ++readGeneration
        const observation = { changes: new Map<string, NodeItem | null>(), sawUpsert: false }
        reads.add(observation)
        try {
            const { nodes: nodeMap, fetchedNodes } = await window.pairApi.nodes.getInitial()
            if (current !== generation || read !== readGeneration) return
            const map = new Map(Object.entries(nodeMap))
            for (const [id, node] of observation.changes) {
                if (node) map.set(id, node)
                else map.delete(id)
            }
            set({ nodes: map, fetchedNodes: fetchedNodes || observation.sawUpsert })
        } catch (error) {
            console.error('Failed to refresh nodes store:', error)
        } finally {
            reads.delete(observation)
        }
    },

    clearNodes: (selfId: string) => {
        readGeneration++
        const prev = get().nodes
        const thisNode = prev.get(selfId)
        if (!thisNode) {
            set({ nodes: new Map() })
            return
        }

        set({ nodes: new Map([[selfId, thisNode]]) })
    },

    cleanup: () => {
        generation++
        readGeneration++
        reads.clear()
        unsubs.forEach(u => u())
        unsubs = []
    }
}))
