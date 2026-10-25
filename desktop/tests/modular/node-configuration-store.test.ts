// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { RemoteManagementCapability } from '@/shared/types/cluster'
import type { ClusterInitialSnapshot } from '@/shared/types/bootstrap'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type { IClusterApi } from '@/ui/api/pair-api'
import { getPairTransport } from '@/ui/api/bootstrap'
import { createEngineApi } from '@/ui/api/engine-api'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import {
    useNodeConfigurationStore,
    watchNodeManagement
} from '@/ui/stores/node-configuration.store'

vi.mock('@/ui/api/bootstrap', () => ({ getPairTransport: vi.fn() }))

const read = vi.fn<IClusterApi['getRemoteManagementForNode']>()
const connects = new Set<() => void>()
const disconnects = new Set<() => void>()
let connected = true
const transport: ServiceTransport = {
    get connected() {
        return connected
    },
    invoke: vi.fn<ServiceTransport['invoke']>(),
    subscribePush: vi.fn<ServiceTransport['subscribePush']>(),
    onConnect: callback => {
        connects.add(callback)
        return () => {
            connects.delete(callback)
        }
    },
    onDisconnect: callback => {
        disconnects.add(callback)
        return () => {
            disconnects.delete(callback)
        }
    },
    onAuthFailure: () => () => {},
    destroy: vi.fn<ServiceTransport['destroy']>()
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

function capability(
    state: RemoteManagementCapability['state'],
    revision = '5'
): RemoteManagementCapability {
    return { version: 1, state, allowed: state === 'allowed', revision }
}

const cleanups: Array<() => void> = []
const store = () => useNodeConfigurationStore.getState()

function clusterSnapshot(certFingerprint = 'A'): ClusterInitialSnapshot {
    return {
        identity: {
            nodeUuid: 'self',
            nodeId: 'self.local',
            name: 'Self',
            certFingerprint,
            clusterId: 'home'
        },
        info: { clusterId: 'home', isClustered: true, clusterFriendlyName: 'Home' },
        members: [
            {
                nodeUuid: 'peer',
                id: 'peer.local',
                name: 'Peer',
                ipAddress: '192.0.2.1',
                port: 1,
                clusterId: 'home',
                state: 'member',
                joinedAt: 1,
                lastSeen: 1
            }
        ],
        pendingInvites: []
    }
}

beforeEach(() => {
    connected = true
    vi.mocked(getPairTransport).mockReturnValue(transport)
    read.mockReset().mockResolvedValue(capability('allowed'))
    vi.mocked(transport.invoke).mockReset().mockResolvedValue(null)
    vi.stubGlobal('window', {
        pairApi: {
            cluster: {
                getRemoteManagementForNode: read,
                getRemoteManagement: vi.fn().mockResolvedValue({
                    version: 1,
                    revision: '1',
                    grants: [{ peerUuid: 'peer', allowed: false, state: 'denied' }]
                })
            }
        }
    })
    useClusterInvitationsStore.getState().cleanup()
    useNodeConfigurationStore.setState({ management: new Map() })
})

afterEach(() => {
    cleanups.splice(0).forEach(stop => stop())
    useClusterInvitationsStore.getState().cleanup()
    vi.unstubAllGlobals()
})

describe('shared target management capability', () => {
    const staleStates: RemoteManagementCapability['state'][] = ['unreviewed', 'denied']
    it.each(staleStates)(
        'rereads an open %s permission after a successful remote management command',
        async state => {
            read.mockResolvedValueOnce(capability(state))
            cleanups.push(watchNodeManagement('peer', 'peer'))
            await store().readManagement('peer', 'peer')
            expect(store().management.get('peer')?.state).toBe(state)

            createEngineApi(transport).toggle('ollama', 'peer')

            await vi.waitFor(() => expect(store().management.get('peer')?.state).toBe('allowed'))
            expect(read).toHaveBeenCalledTimes(2)
            expect(connects.size).toBe(1)
        }
    )

    it('does not refresh permission after a failed management command or create an unwatched target', async () => {
        read.mockResolvedValue(capability('denied'))
        cleanups.push(watchNodeManagement('peer', 'peer'))
        await store().readManagement('peer', 'peer')
        vi.mocked(transport.invoke).mockRejectedValueOnce(new Error('Remote management denied'))
        const api = createEngineApi(transport)
        api.toggle('ollama', 'peer')
        await Promise.resolve()
        await Promise.resolve()
        api.toggle('ollama', 'unwatched')
        await Promise.resolve()
        await Promise.resolve()
        expect(read).toHaveBeenCalledTimes(1)
        expect(store().management.get('peer')?.state).toBe('denied')
        expect(store().management.has('unwatched')).toBe(false)
    })

    it('coalesces mutations during an older read into one fresh capability read', async () => {
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        const initial = store().readManagement('peer', 'peer')
        const api = createEngineApi(transport)
        api.toggle('ollama', 'peer')
        api.toggle('llama-cpp', 'peer')
        await Promise.resolve()
        await Promise.resolve()
        expect(read).toHaveBeenCalledTimes(1)

        old.resolve(capability('unreviewed'))
        await initial
        await store().readManagement('peer', 'peer')

        expect(read).toHaveBeenCalledTimes(2)
        expect(store().management.get('peer')?.state).toBe('allowed')
    })

    it('fences an old capability after same-UUID removal and readmission', async () => {
        const cluster = useClusterInvitationsStore.getState()
        const snapshot = clusterSnapshot()
        cluster.hydrate(snapshot)
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        const pending = store().readManagement('peer', 'peer')
        await Promise.resolve()
        read.mockResolvedValue(capability('denied', '1'))
        cluster.hydrate({ ...snapshot, members: [] })
        expect(store().management.get('peer')?.state).toBe('unreachable')
        cluster.hydrate({
            ...snapshot,
            members: snapshot.members.map(member => ({ ...member, joinedAt: 2 }))
        })
        await vi.waitFor(() =>
            expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
        )
        old.resolve(capability('allowed', '99'))
        await pending
        expect(read).toHaveBeenCalledTimes(2)
        expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
    })

    it('fences an old caller capability when the local certificate changes', async () => {
        const cluster = useClusterInvitationsStore.getState()
        const snapshot = clusterSnapshot()
        cluster.hydrate(snapshot)
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        const pending = store().readManagement('peer', 'peer')
        await Promise.resolve()
        read.mockResolvedValue(capability('denied', '1'))
        // Reuse the exact member array: the identity itself must fence authority.
        cluster.hydrate({
            ...snapshot,
            identity: { ...snapshot.identity, certFingerprint: 'B' }
        })
        await vi.waitFor(() =>
            expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
        )
        old.resolve(capability('allowed', '99'))
        await pending
        expect(read).toHaveBeenCalledTimes(2)
        expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
    })

    it('does not issue a queued capability read after cluster cleanup', async () => {
        const cluster = useClusterInvitationsStore.getState()
        cluster.hydrate(clusterSnapshot())
        cleanups.push(watchNodeManagement('peer', 'peer'))
        cluster.cleanup()
        await Promise.resolve()
        await Promise.resolve()
        expect(read).not.toHaveBeenCalled()
        expect(store().management.get('peer')?.state).toBe('unreachable')
    })

    it('does not reread capability for name and last-seen changes alone', async () => {
        const cluster = useClusterInvitationsStore.getState()
        const snapshot = clusterSnapshot()
        cluster.hydrate(snapshot)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        await store().readManagement('peer', 'peer')
        cluster.hydrate({
            ...snapshot,
            members: snapshot.members.map(member => ({ ...member, name: 'New name', lastSeen: 2 }))
        })
        await Promise.resolve()
        expect(read).toHaveBeenCalledTimes(1)
        expect(store().management.get('peer')?.state).toBe('allowed')
    })

    it('shares one target read and transport lifecycle between consumers', async () => {
        const first = watchNodeManagement('peer', 'peer')
        const second = watchNodeManagement('peer', 'peer')
        cleanups.push(first, second)
        await store().readManagement('peer', 'peer')
        expect(read).toHaveBeenCalledExactlyOnceWith('peer')
        expect(store().management.get('peer')).toMatchObject({ state: 'allowed', revision: '5' })
        expect(connects.size).toBe(1)
        expect(disconnects.size).toBe(1)
        first()
        first()
        expect(connects.size).toBe(1)
        expect(store().management.get('peer')?.state).toBe('allowed')
        second()
        expect(connects.size).toBe(0)
        expect(disconnects.size).toBe(0)
        expect(store().management.get('peer')?.state).toBe('unreachable')
    })

    it.each(['denied', 'unreviewed', 'unavailable', 'unsupported'])(
        'retains %s as a refusal with an explanation',
        async state => {
            if (
                state !== 'denied' &&
                state !== 'unreviewed' &&
                state !== 'unavailable' &&
                state !== 'unsupported'
            )
                throw new Error('invalid test state')
            read.mockResolvedValue(capability(state, state === 'unsupported' ? '' : '5'))
            await store().readManagement('peer', 'peer')
            expect(store().management.get('peer')?.state).toBe(state)
            expect(store().management.get('peer')?.message).toBeTruthy()
        }
    )

    it('never promotes contradictory or revisionless remote allow data', async () => {
        read.mockResolvedValueOnce({ ...capability('denied'), allowed: true })
        await store().readManagement('peer', 'peer')
        expect(store().management.get('peer')?.state).toBe('unavailable')
        read.mockResolvedValueOnce(capability('allowed', ''))
        await store().readManagement('peer', 'peer')
        expect(store().management.get('peer')?.state).toBe('unavailable')
    })

    it('reports transport errors as unreachable rather than unsupported', async () => {
        read.mockRejectedValue(new Error('device offline'))
        await store().readManagement('peer', 'peer')
        expect(store().management.get('peer')).toMatchObject({
            state: 'unreachable',
            message: 'device offline'
        })
    })

    it('uses an explicit local target without requesting any remote grant', async () => {
        await store().readManagement('self', undefined)
        expect(store().management.get('self')).toMatchObject({ state: 'local', revision: '' })
        expect(read).not.toHaveBeenCalled()
    })

    it('fences an old read after disconnect and reads again on reconnect without mutations', async () => {
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        cleanups.push(watchNodeManagement('peer', 'peer'))
        const pending = store().readManagement('peer', 'peer')
        await Promise.resolve()
        expect(store().management.get('peer')?.message).toContain('Checking')
        connected = false
        disconnects.forEach(callback => callback())
        expect(store().management.get('peer')?.state).toBe('unreachable')
        connected = true
        read.mockResolvedValue(capability('denied', '1'))
        connects.forEach(callback => callback())
        await store().readManagement('peer', 'peer')
        old.resolve(capability('allowed', '99'))
        await pending
        expect(store().management.get('peer')).toMatchObject({ state: 'denied', revision: '1' })
        expect(read).toHaveBeenCalledTimes(2)
    })

    it('fences a read after the last watcher leaves and an old cleanup cannot detach its replacement', async () => {
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValueOnce(old.promise)
        const stop = watchNodeManagement('slot', 'old-peer')
        cleanups.push(stop)
        const pending = store().readManagement('slot', 'old-peer')
        await Promise.resolve()
        const replacement = watchNodeManagement('slot', 'new-peer')
        cleanups.push(replacement)
        read.mockResolvedValue(capability('denied', '1'))
        stop()
        await store().readManagement('slot', 'new-peer')
        old.resolve(capability('allowed', '50'))
        await pending
        expect(store().management.get('slot')).toMatchObject({ state: 'denied', revision: '1' })
        expect(connects.size).toBe(1)
        replacement()
        expect(connects.size).toBe(0)
    })

    it('discards an allow arriving after the last watcher closes', async () => {
        const old = deferred<RemoteManagementCapability>()
        read.mockReturnValue(old.promise)
        const stop = watchNodeManagement('closed', 'closed')
        cleanups.push(stop)
        const pending = store().readManagement('closed', 'closed')
        stop()
        old.resolve(capability('allowed'))
        await pending
        expect(store().management.get('closed')?.state).toBe('unreachable')
    })

    it('a read after definite refusal refreshes capability without keeping an old allow', async () => {
        await store().readManagement('peer', 'peer')
        read.mockResolvedValue(capability('denied', '6'))
        const refresh = store().readManagement('peer', 'peer')
        expect(store().management.get('peer')?.state).toBe('loading')
        await refresh
        expect(store().management.get('peer')?.state).toBe('denied')
    })
})
