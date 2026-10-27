// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import type { EngineInitialState } from '@/shared/types/engine-api'
import type { EngineDescriptor, EngineStatusData } from '@/shared/types/engines'
import { emptyEngineStatus } from '@/shared/utils/engines'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'

function deferred<T>() {
    let resolve: (value: T) => void = () => {
        throw new Error('not initialized')
    }
    const promise = new Promise<T>(done => {
        resolve = done
    })
    return { promise, resolve }
}

const node: NodeItem = {
    id: 'peer',
    name: 'old',
    status: 'active',
    ipAddress: '127.0.0.1',
    port: 1,
    allIpAddresses: [],
    os: 'Linux',
    topology: { cpu: { model: '', cores: 1, threads: 1 }, gpus: [], ram: 0, storage: [] }
}
const initial: EngineInitialState = {
    descriptors: [],
    statuses: [emptyEngineStatus('peer', 'ollama')],
    models: [],
    activeProgress: [],
    updateAvailable: []
}
const newer: EngineStatusData = { ...initial.statuses[0], processStatus: 'running' }
const descriptor: EngineDescriptor = {
    engineType: 'ollama',
    family: 'ollama',
    label: 'New descriptor',
    available: true,
    protocols: ['ollama-api'],
    modelNaming: 'exact',
    modelStore: '',
    catalog: '',
    operations: {}
}

afterEach(() => {
    useNodesStore.getState().cleanup()
    useEngineStatusStore.getState().cleanup()
    useNodesStore.setState({ nodes: new Map(), fetchedNodes: false })
    useEngineStatusStore.setState({ statusByNode: new Map(), descriptors: new Map() })
    vi.unstubAllGlobals()
})

describe('snapshot ordering', () => {
    it('nodes receives an upsert while initial snapshot is pending', async () => {
        const pending = deferred<{ nodes: Record<string, NodeItem>; fetchedNodes: boolean }>()
        let push: (value: NodeItem) => void = () => {}
        vi.stubGlobal('window', {
            pairApi: {
                nodes: {
                    getInitial: () => pending.promise,
                    onUpsert: (callback: (value: NodeItem) => void) => {
                        push = callback
                        return () => {}
                    },
                    onRemove: () => () => {}
                }
            }
        })
        const loading = useNodesStore.getState().initialize()
        push({ ...node, name: 'new' })
        pending.resolve({ nodes: { peer: node }, fetchedNodes: true })
        await loading
        expect(useNodesStore.getState().nodes.get('peer')?.name).toBe('new')
    })

    it('nodes refresh retains an upsert newer than the pending snapshot', async () => {
        const pending = deferred<{ nodes: Record<string, NodeItem>; fetchedNodes: boolean }>()
        let push: (value: NodeItem) => void = () => {}
        let read: () => Promise<{ nodes: Record<string, NodeItem>; fetchedNodes: boolean }> = () =>
            Promise.resolve({ nodes: { peer: node }, fetchedNodes: true })
        vi.stubGlobal('window', {
            pairApi: {
                nodes: {
                    getInitial: () => read(),
                    onUpsert: (callback: (value: NodeItem) => void) => {
                        push = callback
                        return () => {}
                    },
                    onRemove: () => () => {}
                }
            }
        })
        await useNodesStore.getState().initialize()
        read = () => pending.promise
        const loading = useNodesStore.getState().refresh()
        push({ ...node, name: 'new' })
        pending.resolve({ nodes: { peer: node }, fetchedNodes: true })
        await loading
        expect(useNodesStore.getState().nodes.get('peer')?.name).toBe('new')
    })

    it('node removal during hydration does not resurrect a departed node or discard other nodes', async () => {
        const pending = deferred<{ nodes: Record<string, NodeItem>; fetchedNodes: boolean }>()
        let remove: (id: string) => void = () => {}
        vi.stubGlobal('window', {
            pairApi: {
                nodes: {
                    getInitial: () => pending.promise,
                    onUpsert: () => () => {},
                    onRemove: (callback: typeof remove) => {
                        remove = callback
                        return () => {}
                    }
                }
            }
        })
        const loading = useNodesStore.getState().initialize()
        remove('peer')
        pending.resolve({
            nodes: { peer: node, other: { ...node, id: 'other' } },
            fetchedNodes: true
        })
        await loading
        expect([...useNodesStore.getState().nodes.keys()]).toEqual(['other'])
    })

    it('ignores an older refresh that finishes after a newer refresh', async () => {
        const first = deferred<{ nodes: Record<string, NodeItem>; fetchedNodes: boolean }>()
        const second = deferred<{ nodes: Record<string, NodeItem>; fetchedNodes: boolean }>()
        const read = vi.fn<
            () => Promise<{ nodes: Record<string, NodeItem>; fetchedNodes: boolean }>
        >(() => Promise.resolve({ nodes: { peer: node }, fetchedNodes: true }))
        vi.stubGlobal('window', {
            pairApi: {
                nodes: {
                    getInitial: read,
                    onUpsert: () => () => {},
                    onRemove: () => () => {}
                }
            }
        })
        await useNodesStore.getState().initialize()
        read.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
        const older = useNodesStore.getState().refresh()
        const latest = useNodesStore.getState().refresh()
        second.resolve({ nodes: { peer: { ...node, name: 'latest' } }, fetchedNodes: true })
        await latest
        first.resolve({ nodes: { peer: node }, fetchedNodes: true })
        await older
        expect(useNodesStore.getState().nodes.get('peer')?.name).toBe('latest')
    })

    it('ignores an old node read after cleanup', async () => {
        const pending = deferred<{ nodes: Record<string, NodeItem>; fetchedNodes: boolean }>()
        vi.stubGlobal('window', {
            pairApi: {
                nodes: {
                    getInitial: () => pending.promise,
                    onUpsert: () => () => {},
                    onRemove: () => () => {}
                }
            }
        })
        const loading = useNodesStore.getState().initialize()
        useNodesStore.getState().cleanup()
        pending.resolve({ nodes: { peer: node }, fetchedNodes: true })
        await loading
        expect(useNodesStore.getState().nodes.size).toBe(0)
    })

    it('engine status receives a push while its initial snapshot is pending', async () => {
        const pending = deferred<EngineInitialState>()
        let push: (value: { status: typeof newer }) => void = () => {}
        vi.stubGlobal('window', {
            pairApi: {
                nodes: { onRemove: () => () => {} },
                engines: {
                    getInitialState: () => pending.promise,
                    getIntent: async () => null,
                    onIntentChanged: () => () => {},
                    onDescriptorsChanged: () => () => {},
                    onStateChanged: (callback: typeof push) => {
                        push = callback
                        return () => {}
                    }
                }
            }
        })
        const loading = useEngineStatusStore.getState().initialize()
        push({ status: newer })
        pending.resolve(initial)
        await loading
        expect(useEngineStatusStore.getState().getStatus('peer', 'ollama').processStatus).toBe(
            'running'
        )
    })

    it('replays node removal and rejoin in order while retaining other snapshot statuses', async () => {
        const pending = deferred<EngineInitialState>()
        let push: (value: { status: typeof newer }) => void = () => {}
        let remove: (id: string) => void = () => {}
        vi.stubGlobal('window', {
            pairApi: {
                nodes: {
                    onRemove: (callback: typeof remove) => {
                        remove = callback
                        return () => {}
                    }
                },
                engines: {
                    getInitialState: () => pending.promise,
                    getIntent: async () => null,
                    onIntentChanged: () => () => {},
                    onDescriptorsChanged: () => () => {},
                    onStateChanged: (callback: typeof push) => {
                        push = callback
                        return () => {}
                    }
                }
            }
        })
        const loading = useEngineStatusStore.getState().initialize()
        remove('peer')
        push({ status: newer })
        pending.resolve({
            ...initial,
            statuses: [
                ...initial.statuses,
                emptyEngineStatus('peer', 'lm-studio'),
                emptyEngineStatus('other', 'ollama')
            ]
        })
        await loading
        expect(useEngineStatusStore.getState().getStatus('peer', 'ollama').processStatus).toBe(
            'running'
        )
        expect(useEngineStatusStore.getState().statusByNode.has('other')).toBe(true)
        expect(useEngineStatusStore.getState().statusByNode.get('peer')?.has('lm-studio')).toBe(
            false
        )
    })

    it('keeps a descriptor push over an older pending snapshot', async () => {
        const pending = deferred<EngineInitialState>()
        let push: (value: EngineDescriptor[]) => void = () => {}
        vi.stubGlobal('window', {
            pairApi: {
                nodes: { onRemove: () => () => {} },
                engines: {
                    getInitialState: () => pending.promise,
                    getIntent: async () => null,
                    onIntentChanged: () => () => {},
                    onDescriptorsChanged: (callback: typeof push) => {
                        push = callback
                        return () => {}
                    },
                    onStateChanged: () => () => {}
                }
            }
        })
        const loading = useEngineStatusStore.getState().initialize()
        push([descriptor])
        pending.resolve({ ...initial, descriptors: [{ ...descriptor, label: 'Old descriptor' }] })
        await loading
        expect(useEngineStatusStore.getState().descriptors.get('ollama')).toEqual(descriptor)
    })

    it('ignores an obsolete engine snapshot after cleanup and reinitialization', async () => {
        const pending = deferred<EngineInitialState>()
        const descriptorsOff = vi.fn<() => void>()
        const statesOff = vi.fn<() => void>()
        const intentOff = vi.fn<() => void>()
        const nodesOff = vi.fn<() => void>()
        vi.stubGlobal('window', {
            pairApi: {
                nodes: { onRemove: () => nodesOff },
                engines: {
                    getInitialState: () => pending.promise,
                    getIntent: async () => null,
                    onIntentChanged: () => intentOff,
                    onDescriptorsChanged: () => descriptorsOff,
                    onStateChanged: () => statesOff
                }
            }
        })
        const obsolete = useEngineStatusStore.getState().initialize()
        useEngineStatusStore.getState().cleanup()
        for (const unsubscribe of [descriptorsOff, statesOff, intentOff, nodesOff]) {
            expect(unsubscribe).toHaveBeenCalledOnce()
        }
        await useEngineStatusStore.getState().initialize({
            ...initial,
            statuses: [newer],
            descriptors: [descriptor]
        })
        pending.resolve({ ...initial, descriptors: [{ ...descriptor, label: 'Old descriptor' }] })
        await obsolete
        expect(useEngineStatusStore.getState().getStatus('peer', 'ollama')).toEqual(newer)
        expect(useEngineStatusStore.getState().descriptors.get('ollama')).toEqual(descriptor)
    })
})
