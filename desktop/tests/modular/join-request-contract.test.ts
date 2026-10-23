// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { JoinRequest, JoinRequestsSnapshot } from '@/shared/types/cluster'
import type { PreloadServiceTransport, ServiceTransport } from '@/shared/types/service-bridge'
import type { StructuredLogger } from '@/shared/types/log'
import type { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import type { getEngineHubModels } from '@/electron/service-bridge/model-catalog'
import type {
    startNodeInfoPoller,
    stopNodeInfoPoller
} from '@/electron/service-bridge/node-info-poller'
import {
    JsonRpcResponseError,
    JsonRpcSocketClient,
    type JsonObject,
    type JsonValue
} from '@/electron/service-bridge/json-rpc-client'
import { parseJoinRequest, parseJoinRequestsSnapshot } from '@/electron/service-bridge/cluster-json'
import { emitBridgePush } from '@/electron/service-bridge/broadcaster'
import { createOverviewWindow } from '@/electron/window'
import { createPairApi } from '@/ui/api/pair-api'
import { serviceTransport, ServiceCallError } from '@/ui/api/service-transport'

type BridgeState = ReturnType<typeof getModularBridgeState>
const mocks = vi.hoisted(() => ({
    state: {
        handleNotification: vi.fn<BridgeState['handleNotification']>(),
        clearWorkloads: vi.fn<BridgeState['clearWorkloads']>(),
        applyEngineIntent: vi.fn<BridgeState['applyEngineIntent']>(),
        invalidateEngineLocks: vi.fn<BridgeState['invalidateEngineLocks']>(),
        setLocalDiscoveryModelRefresher: vi.fn<BridgeState['setLocalDiscoveryModelRefresher']>()
    }
}))

vi.mock('electron', () => ({ app: { isPackaged: false, getAppPath: () => process.cwd() } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<typeof createOverviewWindow>() }))
vi.mock('@/electron/config/ui-config', () => ({ isFirstRun: () => false }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({
    emitBridgePush: vi.fn<typeof emitBridgePush>()
}))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn<StructuredLogger['info']>(),
        warn: vi.fn<StructuredLogger['warn']>(),
        error: vi.fn<StructuredLogger['error']>(),
        verbose: vi.fn<StructuredLogger['verbose']>()
    })
}))
vi.mock('@/electron/service-bridge/node-info-poller', () => ({
    startNodeInfoPoller: vi.fn<typeof startNodeInfoPoller>(),
    stopNodeInfoPoller: vi.fn<typeof stopNodeInfoPoller>()
}))
vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => mocks.state,
    isUpstreamUnreachableError: () => false,
    parseServiceErrors: () => []
}))
vi.mock('@/electron/service-bridge/model-catalog', () => ({
    getEngineHubModels: vi.fn<typeof getEngineHubModels>()
}))

import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'

const request = {
    requestId: '0123456789abcdef0123456789abcdef',
    direction: 'outgoing',
    peerNodeUuid: 'member-uuid',
    peerName: 'Member',
    peerAddress: '192.168.1.20',
    peerPort: 14321,
    state: 'awaiting-approval',
    inviteId: null,
    reason: '',
    createdAt: 1000,
    expiresAt: 301000
} satisfies JoinRequest
const snapshot = {
    supported: true,
    canAskToJoin: true,
    requiresSoloReplacement: false,
    blockedReason: '',
    requests: [request]
} satisfies JoinRequestsSnapshot
const unavailable = {
    supported: true,
    canAskToJoin: false,
    requiresSoloReplacement: false,
    blockedReason: 'state-unavailable',
    requests: []
} satisfies JoinRequestsSnapshot

function mockTransport() {
    return {
        connected: true,
        invoke: vi.fn<ServiceTransport['invoke']>(),
        subscribePush: vi.fn<ServiceTransport['subscribePush']>(() => () => {}),
        onConnect: vi.fn<ServiceTransport['onConnect']>(() => () => {}),
        onDisconnect: vi.fn<ServiceTransport['onDisconnect']>(() => () => {}),
        onAuthFailure: vi.fn<ServiceTransport['onAuthFailure']>(() => () => {}),
        destroy: vi.fn<ServiceTransport['destroy']>()
    } satisfies ServiceTransport
}

describe('strict join request parsing', () => {
    it.each<JoinRequest['state']>([
        'awaiting-approval',
        'awaiting-requester',
        'pairing',
        'paired',
        'declined',
        'canceled',
        'expired',
        'failed',
        'rejected'
    ])('preserves every record field in state %s', state => {
        const value = { ...request, state, inviteId: 'child-invite', reason: 'connection-refused' }
        expect(parseJoinRequest(value)).toEqual(value)
    })

    it('preserves incoming claims and nullable child invite without adding a PIN', () => {
        const value = { ...request, direction: 'incoming', peerNodeUuid: '', peerName: '' }
        expect(parseJoinRequest(value)).toEqual(value)
        expect(parseJoinRequest(value)).not.toHaveProperty('pin')
    })

    it.each(Object.keys(request))('rejects a record missing %s', field => {
        const value: JsonObject = { ...request }
        delete value[field]
        expect(() => parseJoinRequest(value)).toThrow('Invalid join request response')
    })

    it.each<{ name: string; patch: JsonObject }>([
        { name: 'empty ID', patch: { requestId: '' } },
        { name: 'numeric ID', patch: { requestId: 1 } },
        { name: 'unknown direction', patch: { direction: 'sideways' } },
        { name: 'invalid peer UUID', patch: { peerNodeUuid: null } },
        { name: 'invalid peer name', patch: { peerName: [] } },
        { name: 'invalid peer address', patch: { peerAddress: false } },
        { name: 'string port', patch: { peerPort: '14321' } },
        { name: 'zero port', patch: { peerPort: 0 } },
        { name: 'oversize port', patch: { peerPort: 65536 } },
        { name: 'fractional port', patch: { peerPort: 14321.5 } },
        { name: 'unknown state', patch: { state: 'approved' } },
        { name: 'ordinary invite state', patch: { state: 'pending' } },
        { name: 'invalid child invite', patch: { inviteId: 1 } },
        { name: 'empty child invite', patch: { inviteId: '' } },
        { name: 'invalid reason', patch: { reason: null } },
        { name: 'string creation time', patch: { createdAt: '1000' } },
        { name: 'negative creation time', patch: { createdAt: -1 } },
        { name: 'fractional expiration', patch: { expiresAt: 1000.5 } },
        { name: 'nonfinite expiration', patch: { expiresAt: Infinity } },
        { name: 'expiration before creation', patch: { expiresAt: 999 } },
        { name: 'unexpected PIN field', patch: { pin: null } },
        { name: 'unexpected grant field', patch: { allowRemoteManagement: true } }
    ])('rejects $name', ({ patch }) => {
        expect(() => parseJoinRequest({ ...request, ...patch })).toThrow(
            'Invalid join request response'
        )
    })

    it.each<JsonValue | undefined>([undefined, null, [], 'request'])(
        'rejects non-record %s',
        value => {
            expect(() => parseJoinRequest(value)).toThrow('Invalid join request response')
        }
    )
})

describe('join request snapshot parsing', () => {
    it('preserves eligibility, solo consent requirement and the full request set', () => {
        const value = { ...snapshot, requiresSoloReplacement: true }
        expect(parseJoinRequestsSnapshot(value)).toEqual(value)
    })

    it.each(['already-clustered', 'pairing-busy', 'state-unavailable'])(
        'preserves authoritative blocked reason %s',
        blockedReason => {
            const value = { ...snapshot, canAskToJoin: false, blockedReason }
            expect(parseJoinRequestsSnapshot(value)).toEqual(value)
        }
    )

    it.each(Object.keys(snapshot))('invalidates a snapshot missing %s', field => {
        const value: JsonObject = { ...snapshot }
        delete value[field]
        expect(parseJoinRequestsSnapshot(value)).toEqual(unavailable)
    })

    it.each<{ name: string; value: JsonValue | undefined }>([
        { name: 'missing snapshot', value: undefined },
        { name: 'null snapshot', value: null },
        { name: 'array snapshot', value: [] },
        { name: 'false support from the wire', value: { ...snapshot, supported: false } },
        { name: 'string eligibility', value: { ...snapshot, canAskToJoin: 'true' } },
        { name: 'invalid solo flag', value: { ...snapshot, requiresSoloReplacement: null } },
        { name: 'unknown blocked reason', value: { ...snapshot, blockedReason: 'offline' } },
        {
            name: 'contradictory eligibility',
            value: { ...snapshot, blockedReason: 'pairing-busy' }
        },
        { name: 'missing blocked reason', value: { ...snapshot, canAskToJoin: false } },
        { name: 'invalid request array', value: { ...snapshot, requests: {} } },
        {
            name: 'malformed request among valid records',
            value: { ...snapshot, requests: [request, { ...request, state: 'approved' }] }
        },
        { name: 'duplicate request IDs', value: { ...snapshot, requests: [request, request] } },
        { name: 'unexpected PIN field', value: { ...snapshot, pin: null } }
    ])('invalidates eligibility for $name', ({ value }) => {
        expect(parseJoinRequestsSnapshot(value)).toEqual(unavailable)
    })
})

describe('join request API and bridge propagation', () => {
    beforeEach(() => {
        vi.spyOn(getModularSupervisor(), 'callProcess').mockResolvedValue(request)
    })

    it('reads the independent getter without fetching cluster bootstrap', async () => {
        const transport = mockTransport()
        transport.invoke.mockResolvedValue(snapshot)
        await expect(createPairApi(transport).cluster.getJoinRequests()).resolves.toEqual(snapshot)
        expect(transport.invoke.mock.calls).toEqual([['cluster:get-join-requests']])
        vi.mocked(getModularSupervisor().callProcess).mockResolvedValue(snapshot)
        await expect(
            handleServiceBridgeInvoke('cluster:get-join-requests', undefined)
        ).resolves.toEqual(snapshot)
        expect(getModularSupervisor().callProcess).toHaveBeenCalledExactlyOnceWith(
            'broker',
            'cluster:get-join-requests',
            {},
            undefined
        )
    })

    it.each([false, true])(
        'relays explicit solo replacement consent %s and peer selector',
        async replaceSoloCluster => {
            const payload = {
                address: 'member.local',
                port: 23456,
                nodeId: 'member-uuid',
                replaceSoloCluster
            }
            const transport = mockTransport()
            transport.invoke.mockResolvedValue(request)
            await expect(createPairApi(transport).cluster.askToJoin(payload)).resolves.toEqual(
                request
            )
            expect(transport.invoke).toHaveBeenCalledWith('cluster:ask-to-join', payload)
            await expect(
                handleServiceBridgeInvoke('cluster:ask-to-join', payload)
            ).resolves.toEqual(request)
            expect(getModularSupervisor().callProcess).toHaveBeenCalledExactlyOnceWith(
                'broker',
                'cluster:ask-to-join',
                payload,
                undefined
            )
        }
    )

    it('leaves optional target resolution fields to the manager and strips unrelated IPC fields', async () => {
        const payload = {
            address: '192.168.1.20',
            replaceSoloCluster: false,
            pin: 'unused',
            allowRemoteManagement: true
        }
        await handleServiceBridgeInvoke('cluster:ask-to-join', payload)
        expect(getModularSupervisor().callProcess).toHaveBeenCalledExactlyOnceWith(
            'broker',
            'cluster:ask-to-join',
            { address: payload.address, replaceSoloCluster: false },
            undefined
        )
    })

    it.each([false, true])(
        'relays the admitting device grant choice %s without a PIN',
        async allowRemoteManagement => {
            const payload = {
                requestId: request.requestId,
                accept: true,
                allowRemoteManagement
            } satisfies Parameters<
                typeof handleServiceBridgeInvoke<'cluster:respond-to-join-request'>
            >[1]
            const transport = mockTransport()
            await createPairApi(transport).cluster.respondToJoinRequest(
                request.requestId,
                true,
                allowRemoteManagement
            )
            expect(transport.invoke).toHaveBeenCalledWith(
                'cluster:respond-to-join-request',
                payload
            )
            const extraFields = {
                ...payload,
                pin: 'unused'
            }
            await handleServiceBridgeInvoke('cluster:respond-to-join-request', extraFields)
            expect(getModularSupervisor().callProcess).toHaveBeenCalledExactlyOnceWith(
                'broker',
                'cluster:respond-to-join-request',
                payload,
                undefined
            )
        }
    )

    it('relays decline with only the request ID and decision', async () => {
        const payload = { requestId: request.requestId, accept: false } satisfies Parameters<
            typeof handleServiceBridgeInvoke<'cluster:respond-to-join-request'>
        >[1]
        const transport = mockTransport()
        await createPairApi(transport).cluster.respondToJoinRequest(request.requestId, false)
        expect(transport.invoke).toHaveBeenCalledWith('cluster:respond-to-join-request', payload)
        await handleServiceBridgeInvoke('cluster:respond-to-join-request', payload)
        expect(getModularSupervisor().callProcess).toHaveBeenCalledExactlyOnceWith(
            'broker',
            'cluster:respond-to-join-request',
            payload,
            undefined
        )
    })

    it('relays cancellation by exact request ID', async () => {
        const transport = mockTransport()
        await createPairApi(transport).cluster.cancelJoinRequest(request.requestId)
        expect(transport.invoke).toHaveBeenCalledWith('cluster:cancel-join-request', {
            requestId: request.requestId
        })
        await handleServiceBridgeInvoke('cluster:cancel-join-request', {
            requestId: request.requestId
        })
        expect(getModularSupervisor().callProcess).toHaveBeenCalledExactlyOnceWith(
            'broker',
            'cluster:cancel-join-request',
            { requestId: request.requestId },
            undefined
        )
    })

    it.each([
        {
            name: 'missing solo consent',
            channel: 'cluster:ask-to-join',
            payload: { address: 'member.local' }
        },
        {
            name: 'string solo consent',
            channel: 'cluster:ask-to-join',
            payload: { address: 'member.local', replaceSoloCluster: 'true' }
        },
        {
            name: 'missing address',
            channel: 'cluster:ask-to-join',
            payload: { replaceSoloCluster: false }
        },
        {
            name: 'invalid port',
            channel: 'cluster:ask-to-join',
            payload: { address: 'member.local', replaceSoloCluster: false, port: 0 }
        },
        {
            name: 'missing approval grant',
            channel: 'cluster:respond-to-join-request',
            payload: { requestId: request.requestId, accept: true }
        },
        {
            name: 'string approval grant',
            channel: 'cluster:respond-to-join-request',
            payload: { requestId: request.requestId, accept: true, allowRemoteManagement: 'true' }
        },
        {
            name: 'missing response ID',
            channel: 'cluster:respond-to-join-request',
            payload: { accept: false }
        },
        { name: 'missing cancel ID', channel: 'cluster:cancel-join-request', payload: {} }
    ])('rejects malformed IPC input: $name', async ({ channel, payload }) => {
        await expect(
            Reflect.apply(handleServiceBridgeInvoke, undefined, [channel, payload])
        ).rejects.toThrow()
        expect(getModularSupervisor().callProcess).not.toHaveBeenCalled()
    })

    it('marks only method-not-found from an old local service unsupported', async () => {
        vi.mocked(getModularSupervisor().callProcess).mockRejectedValue(
            new JsonRpcResponseError(-32601, 'method not found')
        )
        await expect(
            handleServiceBridgeInvoke('cluster:get-join-requests', undefined)
        ).resolves.toEqual({
            supported: false,
            canAskToJoin: false,
            requiresSoloReplacement: false,
            blockedReason: 'unsupported',
            requests: []
        })
    })

    it.each([
        new Error('connection refused'),
        new Error('method not found'),
        new JsonRpcResponseError(-32004, 'state-unavailable')
    ])('does not infer support or eligibility from failure: %s', async error => {
        vi.mocked(getModularSupervisor().callProcess).mockRejectedValue(error)
        await expect(
            handleServiceBridgeInvoke('cluster:get-join-requests', undefined)
        ).rejects.toBe(error)
    })

    it.each([-32001, -32002, -32602, -32004, -32601])(
        'preserves mutation error code %s and message',
        async code => {
            const refusal = new JsonRpcResponseError(code, 'pairing-busy')
            vi.mocked(getModularSupervisor().callProcess).mockRejectedValue(refusal)
            await expect(
                handleServiceBridgeInvoke('cluster:ask-to-join', {
                    address: 'member.local',
                    replaceSoloCluster: false
                })
            ).rejects.toBe(refusal)
            const invoke = vi
                .fn<PreloadServiceTransport['invoke']>()
                .mockResolvedValue({ success: false, error: refusal.message, code })
            const preload = { ...mockTransport(), invoke } satisfies PreloadServiceTransport
            const result = createPairApi(serviceTransport(preload)).cluster.askToJoin({
                address: 'member.local',
                replaceSoloCluster: false
            })
            await expect(result).rejects.toBeInstanceOf(ServiceCallError)
            await expect(result).rejects.toMatchObject({ code, message: refusal.message })
        }
    )

    it('retrieves the child PIN only through an explicit local invite-status call', async () => {
        const approved = { ...request, state: 'pairing', inviteId: 'child-invite' }
        vi.mocked(getModularSupervisor().callProcess).mockResolvedValueOnce(approved)
        const result = await handleServiceBridgeInvoke('cluster:respond-to-join-request', {
            requestId: request.requestId,
            accept: true,
            allowRemoteManagement: false
        })
        expect(result).toEqual(approved)
        expect(result).not.toHaveProperty('pin')
        expect(getModularSupervisor().callProcess).toHaveBeenCalledTimes(1)
        vi.mocked(getModularSupervisor().callProcess).mockResolvedValueOnce({
            inviteId: 'child-invite',
            pin: '123456',
            state: 'pending'
        })
        const invite = await handleServiceBridgeInvoke('cluster:invite-status', {
            inviteId: 'child-invite'
        })
        expect(invite.inviteId).toBe('child-invite')
        expect(typeof invite.pin).toBe('string')
        expect(getModularSupervisor().callProcess).toHaveBeenLastCalledWith(
            'broker',
            'cluster:invite-status',
            { inviteId: 'child-invite' },
            undefined
        )
    })

    it('subscribes to the full snapshot push and returns its unsubscribe function', () => {
        const transport = mockTransport()
        const unsubscribe = vi.fn<() => void>()
        transport.subscribePush.mockReturnValue(unsubscribe)
        const callback = vi.fn<(value: JoinRequestsSnapshot) => void>()
        const result = createPairApi(transport).cluster.onJoinRequestsChanged(callback)
        expect(transport.subscribePush).toHaveBeenCalledWith(
            'cluster:join-requests-changed',
            callback
        )
        expect(result).toBe(unsubscribe)
    })
})

describe('supervisor join request push dispatch', () => {
    it('forwards full snapshots through the broker demux and invalidates malformed updates without opening a window', async () => {
        const clients: JsonRpcSocketClient[] = []
        vi.spyOn(fs, 'existsSync').mockImplementation(file => String(file).includes('nvpair-'))
        vi.spyOn(JsonRpcSocketClient.prototype, 'start').mockImplementation(function (
            this: JsonRpcSocketClient
        ) {
            clients.push(this)
        })
        vi.spyOn(JsonRpcSocketClient.prototype, 'detach').mockResolvedValue(undefined)
        const supervisor = getModularSupervisor()
        supervisor.start()
        try {
            const client = clients[0]
            if (!client) throw new Error('Supervisor did not attach the broker notification source')
            client.emit('notification', {
                source: 'broker',
                method: 'cluster:join-requests-changed',
                params: snapshot
            })
            expect(emitBridgePush).toHaveBeenLastCalledWith(
                'cluster:join-requests-changed',
                snapshot
            )
            client.emit('notification', {
                source: 'broker',
                method: 'cluster:join-requests-changed',
                params: { ...snapshot, requests: [{ ...request, state: 'approved' }] }
            })
            expect(emitBridgePush).toHaveBeenLastCalledWith(
                'cluster:join-requests-changed',
                unavailable
            )
            expect(createOverviewWindow).not.toHaveBeenCalled()
        } finally {
            await supervisor.stop()
        }
    })
})
