// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, expect, it, vi } from 'vitest'
import type { EngineInitialState } from '@/shared/types/engine-api'
import type { EngineStatusData } from '@/shared/types/engines'
import { emptyEngineStatus } from '@/shared/utils/engines'
import { connectAndInitialize } from '@/ui/stores/init'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'

vi.mock('@/ui/utils/window-kind', () => ({ isTrayWindow: true }))
vi.mock('@/ui/stores/service-status.store', () => ({
    useServiceStatusStore: {
        getState: () => ({ initialize: async () => {} }),
        subscribe: () => () => {}
    }
}))
vi.mock('@/ui/stores/connection.store', () => ({
    useConnectionStore: { getState: () => ({ initialize: async () => {}, cleanup: () => {} }) }
}))
vi.mock('@/ui/stores/nodes.store', () => ({
    useNodesStore: {
        getState: () => ({ initialize: async () => {}, refresh: async () => {}, cleanup: () => {} })
    }
}))
vi.mock('@/ui/stores/cluster-invitations.store', () => ({
    useClusterInvitationsStore: {
        getState: () => ({ initialize: async () => {}, refresh: async () => {}, cleanup: () => {} })
    }
}))
vi.mock('@/ui/stores/metrics.store', () => ({
    useMetricsStore: {
        getState: () => ({ initialize: () => {}, cleanup: () => {}, clearAll: () => {} })
    }
}))
vi.mock('@/ui/stores/workloads.store', () => ({
    useWorkloadsStore: {
        getState: () => ({ initialize: async () => {}, refresh: async () => {}, cleanup: () => {} })
    }
}))

function deferred<T>() {
    let resolve: (value: T) => void = () => {
        throw new Error('not initialized')
    }
    const promise = new Promise<T>(done => {
        resolve = done
    })
    return { promise, resolve }
}

afterEach(() => {
    useEngineStatusStore.getState().cleanup()
    useEngineStatusStore.setState({ statusByNode: new Map(), descriptors: new Map() })
    vi.unstubAllGlobals()
})

it('subscribes to engine status during shared startup and resync snapshots', async () => {
    const snapshot = deferred<EngineInitialState>()
    const started = deferred<void>()
    const refreshSnapshot = deferred<EngineInitialState>()
    const refreshStarted = deferred<void>()
    let refresh = () => {}
    let reads = 0
    let push: (patch: { status: EngineStatusData }) => void = () => {}
    const old = emptyEngineStatus('peer', 'ollama')
    vi.stubGlobal('window', {
        pairApi: {
            app: { getInitial: async () => undefined },
            cluster: { getInitial: async () => undefined },
            nodes: { onRemove: () => () => {} },
            connection: {
                onStateRequestRefresh: (callback: () => void) => {
                    refresh = callback
                    return () => {}
                }
            },
            engines: {
                getInitialState: () => {
                    reads++
                    if (reads === 1) {
                        started.resolve()
                        return snapshot.promise
                    }
                    refreshStarted.resolve()
                    return refreshSnapshot.promise
                },
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
    const loading = connectAndInitialize()
    await started.promise
    push({ status: { ...old, processStatus: 'running' } })
    snapshot.resolve({
        descriptors: [],
        statuses: [old],
        models: [],
        activeProgress: [],
        updateAvailable: []
    })
    await loading
    expect(useEngineStatusStore.getState().getStatus('peer', 'ollama').processStatus).toBe(
        'running'
    )
    refresh()
    await refreshStarted.promise
    push({ status: { ...old, processStatus: 'stopped' } })
    refreshSnapshot.resolve({
        descriptors: [],
        statuses: [old, emptyEngineStatus('other', 'ollama')],
        models: [],
        activeProgress: [],
        updateAvailable: []
    })
    await vi.waitFor(() =>
        expect(useEngineStatusStore.getState().statusByNode.has('other')).toBe(true)
    )
    expect(useEngineStatusStore.getState().getStatus('peer', 'ollama').processStatus).toBe(
        'stopped'
    )
    expect(reads).toBe(2)
})
