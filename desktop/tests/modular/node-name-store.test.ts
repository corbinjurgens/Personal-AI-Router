// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { NodeNameResult } from '@/shared/types/cluster'
import type { ClusterInitialSnapshot } from '@/shared/types/bootstrap'
import type { IClusterApi } from '@/ui/api/pair-api'
import { getPairTransport } from '@/ui/api/bootstrap'
import { ServiceCallError } from '@/ui/api/service-transport'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'

vi.mock('@/ui/api/bootstrap', () => ({ getPairTransport: vi.fn() }))

const getInitial = vi.fn<IClusterApi['getInitial']>()
const setNodeName = vi.fn<IClusterApi['setNodeName']>()
let namePush: ((result: NodeNameResult) => void) | undefined

function snapshot(name: string, customName: string | null): ClusterInitialSnapshot {
    return {
        info: { clusterId: null, isClustered: false, clusterFriendlyName: '' },
        identity: { nodeUuid: 'self', name, customName, certFingerprint: '', clusterId: '' },
        members: [],
        pendingInvites: []
    }
}

function deferred<T>() {
    let resolve: (value: T) => void = () => {
        throw new Error('deferred not initialized')
    }
    const promise = new Promise<T>(finish => {
        resolve = finish
    })
    return { promise, resolve }
}

const store = () => useClusterInvitationsStore.getState()

beforeEach(() => {
    getInitial.mockReset().mockResolvedValue(snapshot('PAIR 1a2b3c4d', null))
    setNodeName.mockReset()
    vi.mocked(getPairTransport).mockReturnValue(null)
    vi.stubGlobal('window', {
        pairApi: {
            cluster: {
                getInitial,
                setNodeName,
                getJoinRequests: vi.fn<IClusterApi['getJoinRequests']>().mockResolvedValue({
                    supported: false,
                    canAskToJoin: false,
                    requiresSoloReplacement: false,
                    blockedReason: 'unsupported',
                    requests: []
                }),
                getRemoteManagement: vi
                    .fn<IClusterApi['getRemoteManagement']>()
                    .mockResolvedValue({ version: 1, revision: '', grants: [] }),
                onJoinRequestsChanged: () => () => {},
                onRemoteManagementChanged: () => () => {},
                onNodeNameChanged: (callback: (result: NodeNameResult) => void) => {
                    namePush = callback
                    return () => {
                        namePush = undefined
                    }
                },
                onInviteReceived: () => () => {},
                onPendingInvitesChanged: () => () => {}
            },
            nodes: { onMembersChanged: () => () => {} }
        }
    })
})

afterEach(() => {
    store().cleanup()
    useClusterInvitationsStore.setState({ selfName: '', selfCustomName: null, nameChange: null })
    vi.unstubAllGlobals()
})

describe('this node name state', () => {
    it('takes the name and custom name from the identity snapshot', async () => {
        await store().initialize(snapshot('Studio', 'Studio'))
        expect(store().selfName).toBe('Studio')
        expect(store().selfCustomName).toBe('Studio')
    })

    it('applies a name push', async () => {
        await store().initialize(snapshot('PAIR 1a2b3c4d', null))
        namePush?.({ nodeUuid: 'self', name: 'Studio', customName: 'Studio' })
        expect(store().selfName).toBe('Studio')
        expect(store().selfCustomName).toBe('Studio')
    })

    it('stores the saved name and reports saved', async () => {
        await store().initialize(snapshot('PAIR 1a2b3c4d', null))
        setNodeName.mockResolvedValue({ nodeUuid: 'self', name: 'Studio', customName: 'Studio' })
        await store().setSelfName('Studio')
        expect(setNodeName).toHaveBeenCalledExactlyOnceWith('Studio')
        expect(store().selfName).toBe('Studio')
        expect(store().selfCustomName).toBe('Studio')
        expect(store().nameChange).toEqual({ phase: 'saved' })
    })

    it('reports saving while the request is in flight', async () => {
        await store().initialize(snapshot('PAIR 1a2b3c4d', null))
        const pending = deferred<NodeNameResult>()
        setNodeName.mockReturnValue(pending.promise)
        const save = store().setSelfName(null)
        expect(store().nameChange).toEqual({ phase: 'saving' })
        pending.resolve({ nodeUuid: 'self', name: 'PAIR 1a2b3c4d', customName: null })
        await save
        expect(store().nameChange).toEqual({ phase: 'saved' })
    })

    it('keeps the old name and exposes the service message on failure', async () => {
        await store().initialize(snapshot('Studio', 'Studio'))
        setNodeName.mockRejectedValue(
            new ServiceCallError('name must be at most 63 characters', -32602)
        )
        await store().setSelfName('x'.repeat(80))
        expect(store().selfName).toBe('Studio')
        expect(store().selfCustomName).toBe('Studio')
        expect(store().nameChange).toEqual({
            phase: 'error',
            message: 'name must be at most 63 characters'
        })
    })

    it('clears the reported outcome', async () => {
        await store().initialize(snapshot('Studio', 'Studio'))
        setNodeName.mockResolvedValue({ nodeUuid: 'self', name: 'Desk', customName: 'Desk' })
        await store().setSelfName('Desk')
        store().clearNameChange()
        expect(store().nameChange).toBeNull()
    })

    it('keeps a push received during the initial fetch over the older snapshot', async () => {
        const fetched = deferred<ClusterInitialSnapshot>()
        getInitial.mockReturnValue(fetched.promise)
        const initializing = store().initialize()
        namePush?.({ nodeUuid: 'self', name: 'Studio', customName: 'Studio' })
        fetched.resolve(snapshot('PAIR 1a2b3c4d', null))
        await initializing
        expect(getInitial).toHaveBeenCalledOnce()
        expect(store().selfName).toBe('Studio')
        expect(store().selfCustomName).toBe('Studio')
    })
})
