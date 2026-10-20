// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClusterNode, Invite, RemoteManagementSnapshot } from '@/shared/types/cluster'
import type { ClusterInitialSnapshot } from '@/shared/types/bootstrap'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type { IClusterApi } from '@/ui/api/pair-api'
import { getPairTransport } from '@/ui/api/bootstrap'
import { ServiceCallError } from '@/ui/api/service-transport'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { MODULAR_INVITE_STATUS_POLL_INTERVAL_MS } from '@/shared/constants/modular-runtime'

vi.mock('@/ui/api/bootstrap', () => ({ getPairTransport: vi.fn() }))

const invite: Invite = {
    inviteId: 'one',
    fromNodeId: 'sender',
    fromNodeUuid: 'uuid',
    fromNodeName: 'Sender',
    toNodeId: null,
    clusterId: 'cluster',
    clusterFriendlyName: 'Home',
    pin: null,
    state: 'pending',
    reason: '',
    createdAt: 1,
    respondedAt: null
}
const respond = vi.fn<IClusterApi['respondToInvite']>()
const status = vi.fn<IClusterApi['inviteStatus']>()
const getPermissions = vi.fn<IClusterApi['getRemoteManagement']>()
const setPermission = vi.fn<IClusterApi['setRemoteManagement']>()
const getInitial = vi.fn<IClusterApi['getInitial']>()
let push: ((invites: Invite[]) => void) | undefined
let permissionPush: ((snapshot: RemoteManagementSnapshot) => void) | undefined
let memberPush: ((members: ClusterNode[]) => void) | undefined
const connectCallbacks = new Set<() => void>()
const disconnectCallbacks = new Set<() => void>()
let transportConnected = true
const transport: ServiceTransport = {
    get connected() {
        return transportConnected
    },
    invoke: vi.fn<ServiceTransport['invoke']>(),
    subscribePush: vi.fn<ServiceTransport['subscribePush']>(),
    onConnect: callback => {
        connectCallbacks.add(callback)
        return () => {
            connectCallbacks.delete(callback)
        }
    },
    onDisconnect: callback => {
        disconnectCallbacks.add(callback)
        return () => {
            disconnectCallbacks.delete(callback)
        }
    },
    onAuthFailure: () => () => {},
    destroy: vi.fn<ServiceTransport['destroy']>()
}

const member: ClusterNode = {
    id: 'peer',
    nodeUuid: 'peer-uuid',
    name: 'Peer',
    ipAddress: '192.0.2.1',
    port: 1,
    clusterId: 'cluster',
    state: 'member',
    joinedAt: 1,
    lastSeen: 1
}

function initial(members: ClusterNode[] = [member]): ClusterInitialSnapshot {
    return {
        info: { clusterId: 'cluster', isClustered: true, clusterFriendlyName: 'Home' },
        identity: {
            nodeUuid: 'self',
            nodeId: 'self',
            name: 'Self',
            certFingerprint: 'fingerprint',
            clusterId: 'cluster'
        },
        members,
        pendingInvites: [invite]
    }
}

function permissions(revision: string, allowed: boolean): RemoteManagementSnapshot {
    return {
        version: 1,
        revision,
        grants: [{ peerUuid: member.nodeUuid, allowed, state: allowed ? 'allowed' : 'denied' }]
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

beforeEach(() => {
    vi.useFakeTimers()
    respond.mockReset()
    status.mockReset()
    getPermissions.mockReset().mockResolvedValue(permissions('1', false))
    setPermission.mockReset()
    getInitial.mockReset().mockResolvedValue(initial())
    vi.mocked(getPairTransport).mockReturnValue(transport)
    transportConnected = true
    vi.stubGlobal('window', {
        pairApi: {
            cluster: {
                respondToInvite: respond,
                inviteStatus: status,
                getInitial,
                getRemoteManagement: getPermissions,
                setRemoteManagement: setPermission,
                onRemoteManagementChanged: (
                    callback: (snapshot: RemoteManagementSnapshot) => void
                ) => {
                    permissionPush = callback
                    return () => {
                        permissionPush = undefined
                    }
                },
                onInviteReceived: () => () => {},
                onPendingInvitesChanged: (callback: (invites: Invite[]) => void) => {
                    push = callback
                    return () => {}
                }
            },
            nodes: {
                onMembersChanged: (callback: (members: ClusterNode[]) => void) => {
                    memberPush = callback
                    return () => {
                        memberPush = undefined
                    }
                }
            }
        }
    })
    useClusterInvitationsStore.setState({ pendingInvites: [invite], activeInviteId: 'one' })
    status.mockResolvedValue(invite)
})
afterEach(() => {
    useClusterInvitationsStore.getState().cleanup()
    vi.useRealTimers()
    vi.unstubAllGlobals()
})

describe('accepted invitation outcome tracking', () => {
    it.each([true, false])('sends the explicit acceptance permission choice %s', async allowed => {
        respond.mockResolvedValue({ ...invite, state: 'paired' })
        await useClusterInvitationsStore.getState().respondToInvite('one', true, '123456', allowed)
        expect(respond).toHaveBeenCalledExactlyOnceWith('one', true, '123456', allowed)
    })

    it('declines without a permission choice', async () => {
        respond.mockResolvedValue({ ...invite, state: 'declined' })
        await useClusterInvitationsStore.getState().respondToInvite('one', false)
        expect(respond).toHaveBeenCalledExactlyOnceWith('one', false)
    })
    it('retries a dismissed submitted card after its pruning status read loses transport', async () => {
        const store = useClusterInvitationsStore.getState()
        await store.initialize({
            info: { clusterId: null, isClustered: false, clusterFriendlyName: '' },
            identity: {
                nodeUuid: 'self',
                nodeId: 'self',
                name: 'Self',
                certFingerprint: '',
                clusterId: ''
            },
            members: [],
            pendingInvites: [invite]
        })
        respond.mockResolvedValue(invite)
        await store.respondToInvite('one', true, '123456', false)
        store.clearActiveInvite()
        status
            .mockRejectedValueOnce(new Error('connection reset'))
            .mockResolvedValue({ ...invite, state: 'paired' })
        expect(push).toBeTypeOf('function')
        push?.([])
        await store.pollInvite('one')
        expect(status).toHaveBeenCalledTimes(1)
        expect(useClusterInvitationsStore.getState().inviteResults.one.state).toBe('pending')
        expect(useClusterInvitationsStore.getState().pendingInvites).toEqual([invite])
        await vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS)
        expect(status).toHaveBeenCalledTimes(2)
        expect(status).toHaveBeenNthCalledWith(2, 'one')
        expect(useClusterInvitationsStore.getState().pendingInvites).toEqual([])
        expect(useClusterInvitationsStore.getState().inviteResults.one.state).toBe('paired')
        expect(useClusterInvitationsStore.getState().activeInviteId).toBeNull()
        await vi.advanceTimersByTimeAsync(30_000)
        expect(status).toHaveBeenCalledTimes(2)
        expect(vi.getTimerCount()).toBe(0)
    })

    it('permits one outstanding status request per invite and discards it after cleanup', async () => {
        const store = useClusterInvitationsStore.getState()
        respond.mockResolvedValue(invite)
        await store.respondToInvite('one', true, '123456', false)
        let finish: ((value: Invite) => void) | undefined
        status.mockImplementation(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        const request = store.pollInvite('one')
        const duplicate = store.pollInvite('one')
        await vi.advanceTimersByTimeAsync(MODULAR_INVITE_STATUS_POLL_INTERVAL_MS * 5)
        expect(status).toHaveBeenCalledTimes(1)
        expect(useClusterInvitationsStore.getState().inviteResults.one.state).toBe('pending')
        store.cleanup()
        expect(vi.getTimerCount()).toBe(0)
        expect(finish).toBeTypeOf('function')
        finish?.({ ...invite, state: 'paired' })
        await Promise.all([request, duplicate])
        expect(useClusterInvitationsStore.getState().inviteResults).toEqual({})
        await vi.advanceTimersByTimeAsync(30_000)
        expect(status).toHaveBeenCalledTimes(1)
    })
    it('updates a dismissed pending card from the exact terminal outcome after a pruning notification', async () => {
        const store = useClusterInvitationsStore.getState()
        await store.initialize({
            info: { clusterId: null, isClustered: false, clusterFriendlyName: '' },
            identity: {
                nodeUuid: 'self',
                nodeId: 'self',
                name: 'Self',
                certFingerprint: '',
                clusterId: ''
            },
            members: [],
            pendingInvites: [invite]
        })
        respond.mockResolvedValue(invite)
        await useClusterInvitationsStore.getState().respondToInvite('one', true, '123456', false)
        useClusterInvitationsStore.getState().clearActiveInvite()
        status.mockResolvedValue({ ...invite, state: 'paired' })
        expect(push).toBeTypeOf('function')
        push?.([])
        await store.pollInvite('one')
        expect(status).toHaveBeenCalledWith('one')
        expect(useClusterInvitationsStore.getState().pendingInvites).toEqual([])
        expect(useClusterInvitationsStore.getState().activeInviteId).toBeNull()
        expect(useClusterInvitationsStore.getState().inviteResults.one.state).toBe('paired')
    })
    it('blocks a second accepted PIN and retains pending state across close/reopen', async () => {
        respond.mockResolvedValue({ ...invite, reason: 'rollback-pending' })
        await useClusterInvitationsStore.getState().respondToInvite('one', true, '123456', false)
        useClusterInvitationsStore.getState().clearActiveInvite()
        useClusterInvitationsStore.getState().setActiveInvite('one')
        await useClusterInvitationsStore.getState().respondToInvite('one', true, '654321', false)
        expect(respond).toHaveBeenCalledTimes(1)
        expect(useClusterInvitationsStore.getState().pendingInvites).toEqual([
            { ...invite, reason: 'rollback-pending' }
        ])
    })

    it('matches the exact invite and prevents a delayed command from overwriting a terminal poll', async () => {
        let finish: ((value: Invite) => void) | undefined
        respond.mockImplementation(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        const command = useClusterInvitationsStore
            .getState()
            .respondToInvite('one', true, '123456', false)
        status.mockResolvedValue({ ...invite, inviteId: 'stale', state: 'paired' })
        await useClusterInvitationsStore.getState().pollInvite('one')
        expect(useClusterInvitationsStore.getState().inviteResults.one.state).toBe('pending')
        status.mockResolvedValue({ ...invite, state: 'failed', reason: 'write-failed' })
        await useClusterInvitationsStore.getState().pollInvite('one')
        expect(finish).toBeTypeOf('function')
        finish?.(invite)
        expect(await command).toEqual({ ...invite, state: 'failed', reason: 'write-failed' })
        expect(status).toHaveBeenNthCalledWith(1, 'one')
        expect(useClusterInvitationsStore.getState().pendingInvites).toEqual([])
    })

    it('preserves the submitted invite when the pending notification races its response', async () => {
        await useClusterInvitationsStore.getState().initialize({
            info: { clusterId: null, isClustered: false, clusterFriendlyName: '' },
            identity: {
                nodeUuid: 'self',
                nodeId: 'self',
                name: 'Self',
                certFingerprint: '',
                clusterId: ''
            },
            members: [],
            pendingInvites: [invite]
        })
        let finish: ((value: Invite) => void) | undefined
        respond.mockImplementation(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        const command = useClusterInvitationsStore
            .getState()
            .respondToInvite('one', true, '123456', false)
        expect(push).toBeTypeOf('function')
        push?.([])
        expect(useClusterInvitationsStore.getState().activeInviteId).toBe('one')
        expect(useClusterInvitationsStore.getState().pendingInvites).toEqual([invite])
        status.mockResolvedValue({ ...invite, state: 'paired' })
        await useClusterInvitationsStore.getState().pollInvite('one')
        expect(finish).toBeTypeOf('function')
        finish?.(invite)
        await command
        expect(useClusterInvitationsStore.getState().inviteResults.one.state).toBe('paired')
        expect(useClusterInvitationsStore.getState().pendingInvites).toEqual([])
    })

    it('keeps transport silence pending and reports an unknown outcome without pairing', async () => {
        respond.mockResolvedValue({ ...invite, reason: 'awaiting-confirmation' })
        await useClusterInvitationsStore.getState().respondToInvite('one', true, '123456', false)
        status.mockRejectedValueOnce(new Error('connection reset'))
        await useClusterInvitationsStore.getState().pollInvite('one')
        expect(useClusterInvitationsStore.getState().inviteResults.one.state).toBe('pending')
        status.mockRejectedValueOnce(new Error('-32001: unknown inviteId'))
        await useClusterInvitationsStore.getState().pollInvite('one')
        expect(useClusterInvitationsStore.getState().inviteResults.one).toMatchObject({
            state: 'expired',
            reason: 'outcome-unknown'
        })
    })

    it('does not restore stale results after store cleanup', async () => {
        let finish: ((value: Invite) => void) | undefined
        respond.mockImplementation(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        const command = useClusterInvitationsStore
            .getState()
            .respondToInvite('one', true, '123456', false)
        useClusterInvitationsStore.getState().cleanup()
        expect(finish).toBeTypeOf('function')
        finish?.({ ...invite, state: 'paired' })
        await command
        expect(useClusterInvitationsStore.getState().inviteResults).toEqual({})
    })
})

describe('current device permission state', () => {
    const store = () => useClusterInvitationsStore.getState()

    it('subscribes before the baseline and preserves a newer push against that baseline and older pushes', async () => {
        const baseline = deferred<RemoteManagementSnapshot>()
        getPermissions.mockImplementation(() => {
            expect(permissionPush).toBeTypeOf('function')
            return baseline.promise
        })
        const attach = store().initialize(initial())
        await Promise.resolve()
        permissionPush?.(permissions('9007199254740994', true))
        baseline.resolve(permissions('9007199254740993', false))
        await attach
        permissionPush?.(permissions('9', false))
        expect(store().remoteManagement).toEqual(permissions('9007199254740994', true))
        expect(getPermissions).toHaveBeenCalledTimes(1)
    })

    it('keeps a newer denial when an older setter reply arrives', async () => {
        await store().initialize(initial())
        const reply = deferred<RemoteManagementSnapshot>()
        setPermission.mockReturnValue(reply.promise)
        const save = store().setRemoteManagement(member.nodeUuid, true)
        expect(setPermission).toHaveBeenCalledExactlyOnceWith({
            peerUuid: 'peer-uuid',
            allowed: true,
            expectedRevision: '1'
        })
        expect(store().remoteManagement).toEqual(permissions('1', false))
        permissionPush?.(permissions('3', false))
        reply.resolve(permissions('2', true))
        await save
        expect(store().remoteManagement).toEqual(permissions('3', false))
        expect(store().remoteManagementChanges['peer-uuid']).toMatchObject({
            allowed: true,
            phase: 'conflict'
        })
    })

    it('fences a pending baseline and setter behind an empty revision invalidation', async () => {
        await store().initialize(initial())
        const baseline = deferred<RemoteManagementSnapshot>()
        const reply = deferred<RemoteManagementSnapshot>()
        getPermissions.mockReturnValue(baseline.promise)
        setPermission.mockReturnValue(reply.promise)
        const read = store().refreshRemoteManagement()
        const save = store().setRemoteManagement(member.nodeUuid, true)
        permissionPush?.({ version: 1, revision: '', grants: [] })
        baseline.resolve(permissions('8', true))
        reply.resolve(permissions('9', true))
        await Promise.all([read, save])
        expect(store().remoteManagement).toBeNull()
        expect(store().remoteManagementChanges['peer-uuid'].phase).toBe('unknown')
        await store().setRemoteManagement(member.nodeUuid, true)
        expect(setPermission).toHaveBeenCalledTimes(1)
    })

    it('does not claim an unavailable setter response was saved and reads back without replay', async () => {
        await store().initialize(initial())
        setPermission.mockResolvedValue({ version: 1, revision: '', grants: [] })
        getPermissions.mockResolvedValue(permissions('2', false))
        await store().setRemoteManagement(member.nodeUuid, true)
        expect(store().remoteManagementChanges['peer-uuid'].phase).toBe('unknown')
        expect(store().remoteManagement).toEqual(permissions('2', false))
        expect(setPermission).toHaveBeenCalledTimes(1)
    })

    it('reads back a CAS conflict and preserves the unsent choice without replay', async () => {
        await store().initialize(initial())
        setPermission.mockRejectedValue(new ServiceCallError('stale permission revision', -32010))
        getPermissions.mockResolvedValue(permissions('4', false))
        await store().setRemoteManagement(member.nodeUuid, true)
        expect(store().remoteManagement).toEqual(permissions('4', false))
        expect(store().remoteManagementChanges['peer-uuid']).toEqual({
            allowed: true,
            phase: 'conflict',
            message: 'stale permission revision'
        })
        expect(setPermission).toHaveBeenCalledTimes(1)
    })

    it('reads back an uncertain save and sends again only on a new explicit action', async () => {
        await store().initialize(initial())
        setPermission.mockRejectedValueOnce(new Error('reply lost'))
        getPermissions.mockResolvedValue(permissions('2', true))
        await store().setRemoteManagement(member.nodeUuid, true)
        expect(store().remoteManagement).toEqual(permissions('2', true))
        expect(store().remoteManagementChanges['peer-uuid'].phase).toBe('unknown')
        expect(setPermission).toHaveBeenCalledTimes(1)
        setPermission.mockResolvedValue(permissions('3', false))
        await store().setRemoteManagement(member.nodeUuid, false)
        expect(setPermission).toHaveBeenLastCalledWith({
            peerUuid: 'peer-uuid',
            allowed: false,
            expectedRevision: '2'
        })
        expect(store().remoteManagementChanges['peer-uuid'].phase).toBe('saved')
    })

    it('invalidates removed and readmitted members and ignores their obsolete responses and pushes', async () => {
        getPermissions.mockResolvedValue(permissions('1', true))
        await store().initialize(initial())
        const removed = deferred<RemoteManagementSnapshot>()
        const readmitted = deferred<RemoteManagementSnapshot>()
        getPermissions.mockReturnValueOnce(removed.promise).mockReturnValueOnce(readmitted.promise)
        memberPush?.([])
        await Promise.resolve()
        expect(store().remoteManagement).toBeNull()
        await store().setRemoteManagement(member.nodeUuid, true)
        expect(setPermission).not.toHaveBeenCalled()
        memberPush?.([{ ...member, joinedAt: 2 }])
        await Promise.resolve()
        permissionPush?.(permissions('1', true))
        expect(store().remoteManagement).toBeNull()
        removed.resolve(permissions('2', true))
        readmitted.resolve(permissions('3', false))
        await store().refreshRemoteManagement()
        permissionPush?.(permissions('1', true))
        expect(store().remoteManagement).toEqual(permissions('3', false))
    })

    it('fences old connection and target reads while reconnect only reads state', async () => {
        await store().initialize(initial())
        const oldRead = deferred<RemoteManagementSnapshot>()
        getPermissions.mockReturnValueOnce(oldRead.promise)
        const request = store().refreshRemoteManagement()
        await Promise.resolve()
        transportConnected = false
        disconnectCallbacks.forEach(callback => callback())
        expect(store().remoteManagement).toBeNull()
        transportConnected = true
        getPermissions.mockResolvedValue(permissions('1', false))
        connectCallbacks.forEach(callback => callback())
        await getInitial.mock.results.at(-1)?.value
        await Promise.resolve()
        await store().refreshRemoteManagement()
        oldRead.resolve(permissions('50', true))
        await request
        expect(store().remoteManagement).toEqual(permissions('1', false))
        const targetRead = deferred<RemoteManagementSnapshot>()
        getPermissions.mockReturnValueOnce(targetRead.promise)
        const stale = store().refreshRemoteManagement()
        await Promise.resolve()
        getPermissions.mockResolvedValue(permissions('1', false))
        const next = initial()
        next.identity.nodeUuid = 'replacement-self'
        store().hydrate(next)
        targetRead.resolve(permissions('70', true))
        await stale
        expect(store().remoteManagement?.revision).not.toBe('70')
        expect(setPermission).not.toHaveBeenCalled()
    })

    it('replaces subscriptions on duplicate initialization and cleanup fences pending reads', async () => {
        await store().initialize(initial())
        await store().initialize(initial())
        expect(connectCallbacks.size).toBe(1)
        expect(disconnectCallbacks.size).toBe(1)
        const pending = deferred<RemoteManagementSnapshot>()
        getPermissions.mockReturnValue(pending.promise)
        const read = store().refreshRemoteManagement()
        store().cleanup()
        pending.resolve(permissions('30', true))
        await read
        expect(permissionPush).toBeUndefined()
        expect(connectCallbacks.size).toBe(0)
        expect(disconnectCallbacks.size).toBe(0)
        expect(store().remoteManagement).toBeNull()
    })

    it('does not adopt an old identity push while the replacement identity baseline is pending', async () => {
        getPermissions.mockResolvedValue(permissions('100', true))
        await store().initialize(initial())
        const replacement = deferred<RemoteManagementSnapshot>()
        getPermissions.mockReturnValue(replacement.promise)
        const next = initial()
        next.identity.nodeUuid = 'replacement-self'
        store().hydrate(next)
        const read = store().refreshRemoteManagement()
        await Promise.resolve()
        permissionPush?.(permissions('100', true))
        try {
            expect(store().remoteManagement).toBeNull()
        } finally {
            replacement.resolve(permissions('1', false))
            await read
            await store().refreshRemoteManagement()
        }
        expect(store().remoteManagement).toEqual(permissions('1', false))
    })

    it('does not adopt an old identity push after the replacement identity baseline has landed', async () => {
        getPermissions.mockResolvedValue(permissions('100', true))
        await store().initialize(initial())
        getPermissions.mockResolvedValue(permissions('1', false))
        const next = initial()
        next.identity.nodeUuid = 'replacement-self'
        store().hydrate(next)
        await store().refreshRemoteManagement()
        expect(store().remoteManagement).toEqual(permissions('1', false))
        permissionPush?.(permissions('100', true))
        expect(store().remoteManagement?.grants.some(grant => grant.allowed)).not.toBe(true)
        await store().refreshRemoteManagement()
        expect(store().remoteManagement).toEqual(permissions('1', false))
    })

    it('coalesces an old identity push burst behind the current read and one fresh readback', async () => {
        getPermissions.mockResolvedValue(permissions('100', true))
        await store().initialize(initial())
        getPermissions.mockResolvedValue(permissions('1', false))
        const next = initial()
        next.identity.nodeUuid = 'burst-self'
        store().hydrate(next)
        await store().refreshRemoteManagement()
        const current = deferred<RemoteManagementSnapshot>()
        getPermissions
            .mockReturnValueOnce(current.promise)
            .mockResolvedValue(permissions('2', false))
        const read = store().refreshRemoteManagement()
        await Promise.resolve()
        const readsBeforePushes = getPermissions.mock.calls.length
        permissionPush?.(permissions('100', true))
        permissionPush?.(permissions('101', true))
        permissionPush?.({ version: 1, revision: '', grants: [] })
        permissionPush?.(permissions('102', true))
        expect(store().remoteManagement).toBeNull()
        expect(getPermissions).toHaveBeenCalledTimes(readsBeforePushes)
        current.resolve(permissions('2', true))
        await read
        await store().refreshRemoteManagement()
        expect(getPermissions).toHaveBeenCalledTimes(readsBeforePushes + 1)
        expect(store().remoteManagement).toEqual(permissions('2', false))
    })

    it('cancels a queued identity readback on cleanup without issuing a post-cleanup RPC', async () => {
        await store().initialize(initial())
        const next = initial()
        next.identity.nodeUuid = 'cleanup-self'
        store().hydrate(next)
        await store().refreshRemoteManagement()
        const callback = permissionPush
        const reads = getPermissions.mock.calls.length
        callback?.(permissions('100', true))
        store().cleanup()
        callback?.(permissions('101', true))
        await Promise.resolve()
        await Promise.resolve()
        expect(getPermissions).toHaveBeenCalledTimes(reads)
        expect(store().remoteManagement).toBeNull()
    })

    it('cancels a follow-up waiting behind an identity read when cleanup occurs', async () => {
        await store().initialize(initial())
        const next = initial()
        next.identity.nodeUuid = 'waiting-self'
        store().hydrate(next)
        await store().refreshRemoteManagement()
        const pending = deferred<RemoteManagementSnapshot>()
        getPermissions.mockReturnValue(pending.promise)
        const read = store().refreshRemoteManagement()
        await Promise.resolve()
        permissionPush?.(permissions('100', true))
        const reads = getPermissions.mock.calls.length
        store().cleanup()
        pending.resolve(permissions('3', true))
        await read
        await Promise.resolve()
        expect(getPermissions).toHaveBeenCalledTimes(reads)
        expect(store().remoteManagement).toBeNull()
    })

    it('does not clear the identity push fence during later initialization', async () => {
        await store().initialize(initial())
        const next = initial()
        next.identity.nodeUuid = 'sticky-self'
        store().hydrate(next)
        await store().refreshRemoteManagement()
        await store().initialize(initial())
        permissionPush?.(permissions('100', true))
        expect(store().remoteManagement).toBeNull()
        await store().refreshRemoteManagement()
        expect(store().remoteManagement).toEqual(permissions('1', false))
    })
})
