// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Invite } from '@/shared/types/cluster'
import type { IClusterApi } from '@/ui/api/pair-api'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { MODULAR_INVITE_STATUS_POLL_INTERVAL_MS } from '@/shared/constants/modular-runtime'

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
let push: ((invites: Invite[]) => void) | undefined

beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('window', {
        pairApi: {
            cluster: {
                respondToInvite: respond,
                inviteStatus: status,
                onInviteReceived: () => () => {},
                onPendingInvitesChanged: (callback: (invites: Invite[]) => void) => {
                    push = callback
                    return () => {}
                }
            },
            nodes: { onMembersChanged: () => () => {} }
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
        await store.respondToInvite('one', true, '123456')
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
        await store.respondToInvite('one', true, '123456')
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
        await useClusterInvitationsStore.getState().respondToInvite('one', true, '123456')
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
        await useClusterInvitationsStore.getState().respondToInvite('one', true, '123456')
        useClusterInvitationsStore.getState().clearActiveInvite()
        useClusterInvitationsStore.getState().setActiveInvite('one')
        await useClusterInvitationsStore.getState().respondToInvite('one', true, '654321')
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
        const command = useClusterInvitationsStore.getState().respondToInvite('one', true, '123456')
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
        const command = useClusterInvitationsStore.getState().respondToInvite('one', true, '123456')
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
        await useClusterInvitationsStore.getState().respondToInvite('one', true, '123456')
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
        const command = useClusterInvitationsStore.getState().respondToInvite('one', true, '123456')
        useClusterInvitationsStore.getState().cleanup()
        expect(finish).toBeTypeOf('function')
        finish?.({ ...invite, state: 'paired' })
        await command
        expect(useClusterInvitationsStore.getState().inviteResults).toEqual({})
    })
})
