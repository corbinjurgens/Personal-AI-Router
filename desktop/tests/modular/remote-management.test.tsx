// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import fs from 'fs'
import type {
    ServiceBridgePushMessage,
    ServiceTransport,
    PreloadServiceTransport
} from '@/shared/types/service-bridge'
import type {
    AvailableNode,
    ClusterNode,
    Invite,
    RemoteManagementCapability,
    RemoteManagementSnapshot
} from '@/shared/types/cluster'
import type { EngineSettingsSnapshot } from '@/shared/types/engine-settings'
import type { NodePolicyDocument, NodePolicySnapshot } from '@/shared/types/node-policy'
import type { NodeItem } from '@/shared/types/nodes'
import type { StructuredLogger } from '@/shared/types/log'
import type {
    startNodeInfoPoller,
    stopNodeInfoPoller
} from '@/electron/service-bridge/node-info-poller'
import type { getEngineHubModels } from '@/electron/service-bridge/model-catalog'
import type { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import {
    JsonRpcResponseError,
    JsonRpcSocketClient,
    type JsonValue
} from '@/electron/service-bridge/json-rpc-client'
import {
    parseRemoteManagementCapability,
    parseRemoteManagementSnapshot
} from '@/electron/service-bridge/cluster-json'
import {
    createPairApi,
    type IClusterApi,
    type INodePolicyApi,
    type IPairApi
} from '@/ui/api/pair-api'
import { serviceTransport, ServiceCallError } from '@/ui/api/service-transport'
import { REMOTE_MANAGEMENT_DENIED_CODE } from '@/shared/constants/service-errors'
import { AddNodeModal } from '@/ui/components/AddNodeModal'
import AvailableNodeCard from '@/ui/components/ClusterSettings/AvailableNode'
import ActiveNode from '@/ui/components/ClusterSettings/ActiveNode'
import RoleEditor from '@/ui/components/RoutingSettings/RoleEditor'
import RoutingSettings from '@/ui/components/RoutingSettings/RoutingSettings'
import { EngineSettingsSection } from '@/ui/components/BackendRow/EngineSettingsSection'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useDiscoveredNodesStore } from '@/ui/stores/discovered-nodes.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { emptyEngineStatus } from '@/shared/utils/engines'

vi.mock('@/ui/components/RoutingSettings/BulkNodeConfiguration', () => ({ default: () => null }))

type BridgeState = ReturnType<typeof getModularBridgeState>

const mocks = vi.hoisted(() => ({
    send: vi.fn<(channel: string, message: ServiceBridgePushMessage) => void>(),
    state: {
        handleNotification: vi.fn<BridgeState['handleNotification']>(),
        clearWorkloads: vi.fn<BridgeState['clearWorkloads']>(),
        getSelfId: vi.fn<BridgeState['getSelfId']>(() => 'self'),
        setLocalDiscoveryModelRefresher: vi.fn<BridgeState['setLocalDiscoveryModelRefresher']>(),
        applyEngineIntent: vi.fn<BridgeState['applyEngineIntent']>(),
        invalidateEngineLocks: vi.fn<BridgeState['invalidateEngineLocks']>(),
        beginEngineIntentRead: vi.fn<BridgeState['beginEngineIntentRead']>(() => 0),
        getEngineIntent: vi.fn<BridgeState['getEngineIntent']>(() => null),
        prunePendingInvite: vi.fn<BridgeState['prunePendingInvite']>()
    }
}))

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: {
        getAllWindows: () => [
            {
                isDestroyed: () => false,
                webContents: { isDestroyed: () => false, send: mocks.send }
            }
        ]
    }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<() => void>() }))
vi.mock('@/electron/config/ui-config', () => ({ isFirstRun: () => false }))
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

const allowedSnapshot = {
    version: 1,
    revision: '9007199254740993',
    grants: [{ peerUuid: 'peer-1', allowed: true, state: 'allowed' }]
} satisfies RemoteManagementSnapshot
const unavailableSnapshot = { version: 1, revision: '', grants: [] }
const unavailableCapability = { version: 1, revision: '', allowed: false, state: 'unavailable' }
const inviteResult = {
    inviteId: 'invite-1',
    fromNodeId: 'sender.local',
    fromNodeUuid: 'sender-uuid',
    fromNodeName: 'Sender',
    toNodeId: null,
    clusterId: 'cluster-1',
    clusterFriendlyName: 'Home',
    pin: null,
    state: 'pending',
    reason: '',
    createdAt: 1,
    respondedAt: null
} satisfies Invite

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

describe('remote management parsing', () => {
    it('preserves canonical uint64 revisions without Number conversion', () => {
        expect(parseRemoteManagementSnapshot(allowedSnapshot)).toEqual(allowedSnapshot)
        expect(
            parseRemoteManagementSnapshot({
                version: 1,
                revision: '18446744073709551615',
                grants: []
            })
        ).toEqual({ version: 1, revision: '18446744073709551615', grants: [] })
    })

    it('keeps unavailable authority distinct from an authoritative empty list', () => {
        expect(parseRemoteManagementSnapshot({ version: 1, revision: '', grants: [] })).toEqual(
            unavailableSnapshot
        )
        expect(parseRemoteManagementSnapshot({ version: 1, revision: '1', grants: [] })).toEqual({
            version: 1,
            revision: '1',
            grants: []
        })
        expect(
            parseRemoteManagementSnapshot({
                version: 1,
                revision: '',
                grants: [{ peerUuid: 'peer-1', allowed: false, state: 'unavailable' }]
            })
        ).toEqual({
            version: 1,
            revision: '',
            grants: [{ peerUuid: 'peer-1', allowed: false, state: 'unavailable' }]
        })
    })

    it.each([
        { name: 'missing payload', value: undefined },
        { name: 'unsupported version', value: { ...allowedSnapshot, version: 2 } },
        {
            name: 'missing grant boolean',
            value: { version: 1, revision: '1', grants: [{ peerUuid: 'peer-1', state: 'allowed' }] }
        },
        {
            name: 'string boolean',
            value: {
                version: 1,
                revision: '1',
                grants: [{ peerUuid: 'peer-1', allowed: 'true', state: 'allowed' }]
            }
        },
        {
            name: 'unknown state',
            value: {
                version: 1,
                revision: '1',
                grants: [{ peerUuid: 'peer-1', allowed: true, state: 'future' }]
            }
        },
        {
            name: 'contradictory allow',
            value: {
                version: 1,
                revision: '1',
                grants: [{ peerUuid: 'peer-1', allowed: true, state: 'denied' }]
            }
        },
        {
            name: 'contradictory denial',
            value: {
                version: 1,
                revision: '1',
                grants: [{ peerUuid: 'peer-1', allowed: false, state: 'allowed' }]
            }
        },
        { name: 'allow without authority revision', value: { ...allowedSnapshot, revision: '' } },
        {
            name: 'duplicate peer',
            value: {
                ...allowedSnapshot,
                grants: [...allowedSnapshot.grants, ...allowedSnapshot.grants]
            }
        },
        { name: 'extra remote selector', value: { ...allowedSnapshot, nodeId: 'target' } },
        {
            name: 'invalid principal',
            value: {
                version: 1,
                revision: '1',
                grants: [{ peerUuid: '../peer', allowed: true, state: 'allowed' }]
            }
        },
        {
            name: 'unicode whitespace principal',
            value: {
                version: 1,
                revision: '1',
                grants: [{ peerUuid: 'peer\u0085', allowed: true, state: 'allowed' }]
            }
        },
        {
            name: 'UTF8 principal byte limit',
            value: {
                version: 1,
                revision: '1',
                grants: [{ peerUuid: 'é'.repeat(65), allowed: true, state: 'allowed' }]
            }
        },
        {
            name: 'too many grants',
            value: {
                version: 1,
                revision: '1',
                grants: Array.from({ length: 1025 }, (_, i) => ({
                    peerUuid: `peer-${i}`,
                    allowed: true,
                    state: 'allowed'
                }))
            }
        },
        { name: 'oversize payload', value: { ...allowedSnapshot, extra: 'x'.repeat(1 << 20) } }
    ] satisfies { name: string; value: JsonValue | undefined }[])(
        'invalidates a local snapshot: $name',
        ({ value }) => {
            expect(parseRemoteManagementSnapshot(value)).toEqual(unavailableSnapshot)
        }
    )

    it.each(['0', '01', '-1', '1.0', '18446744073709551616', '100000000000000000000'])(
        'rejects noncanonical or overflowing revision %s',
        revision => {
            expect(parseRemoteManagementSnapshot({ ...allowedSnapshot, revision })).toEqual(
                unavailableSnapshot
            )
            expect(
                parseRemoteManagementCapability({
                    version: 1,
                    revision,
                    allowed: true,
                    state: 'allowed'
                })
            ).toEqual(unavailableCapability)
        }
    )

    it('accepts the Go principal and peer-count boundaries', () => {
        const grants = Array.from({ length: 1024 }, (_, i) => ({
            peerUuid: `peer-${i}`,
            allowed: false,
            state: 'unreviewed'
        }))
        expect(
            parseRemoteManagementSnapshot({ version: 1, revision: '1', grants }).grants
        ).toHaveLength(1024)
        expect(
            parseRemoteManagementSnapshot({
                version: 1,
                revision: '1',
                grants: [{ peerUuid: 'é'.repeat(64), allowed: false, state: 'denied' }]
            }).grants
        ).toEqual([{ peerUuid: 'é'.repeat(64), allowed: false, state: 'denied' }])
    })

    it.each([
        { allowed: true, state: 'allowed', revision: '3' },
        { allowed: false, state: 'denied', revision: '3' },
        { allowed: false, state: 'unreviewed', revision: '3' },
        { allowed: false, state: 'unavailable', revision: '' },
        { allowed: false, state: 'unsupported', revision: '' }
    ])('preserves caller-only capability $state', capability => {
        expect(parseRemoteManagementCapability({ version: 1, ...capability })).toEqual({
            version: 1,
            ...capability
        })
    })

    it.each<{ name: string; value: JsonValue }>([
        { name: 'full local list', value: allowedSnapshot },
        { name: 'missing allowed', value: { version: 1, revision: '1', state: 'allowed' } },
        {
            name: 'unsupported version',
            value: { version: 2, revision: '1', allowed: true, state: 'allowed' }
        },
        {
            name: 'unknown state',
            value: { version: 1, revision: '1', allowed: true, state: 'future' }
        },
        {
            name: 'contradictory allow',
            value: { version: 1, revision: '1', allowed: true, state: 'denied' }
        },
        {
            name: 'contradictory denial',
            value: { version: 1, revision: '1', allowed: false, state: 'allowed' }
        },
        {
            name: 'empty allow revision',
            value: { version: 1, revision: '', allowed: true, state: 'allowed' }
        },
        {
            name: 'unsupported usable revision',
            value: { version: 1, revision: '1', allowed: false, state: 'unsupported' }
        },
        {
            name: 'remote peer selector',
            value: { version: 1, revision: '1', allowed: true, state: 'allowed', peerUuid: 'peer' }
        },
        {
            name: 'oversize response',
            value: {
                version: 1,
                revision: '1',
                allowed: true,
                state: 'allowed',
                extra: 'x'.repeat(64 << 10)
            }
        }
    ])('fails closed for remote $name', ({ value }) => {
        expect(parseRemoteManagementCapability(value)).toEqual(unavailableCapability)
    })
})

describe('permission and pairing API contracts', () => {
    beforeEach(() => {
        vi.spyOn(getModularSupervisor(), 'callProcess').mockResolvedValue(inviteResult)
    })

    it.each([true, false])(
        'sends the explicit outbound choice %s through both API and bridge',
        async allowRemoteManagement => {
            const transport = mockTransport()
            transport.invoke.mockResolvedValue(inviteResult)
            await createPairApi(transport).cluster.inviteNode('192.168.1.2', allowRemoteManagement)
            expect(transport.invoke).toHaveBeenCalledWith('cluster:invite-node', {
                ipAddress: '192.168.1.2',
                allowRemoteManagement
            })
            await handleServiceBridgeInvoke('cluster:invite-node', {
                ipAddress: '192.168.1.2',
                allowRemoteManagement
            })
            expect(getModularSupervisor().callProcess).toHaveBeenCalledWith(
                'broker',
                'cluster:invite-node',
                {
                    address: '192.168.1.2',
                    port: 14321,
                    allowRemoteManagement
                },
                35000
            )
        }
    )

    it.each([true, false])(
        'sends accepted choice %s and PIN through both API and bridge',
        async allowRemoteManagement => {
            const transport = mockTransport()
            transport.invoke.mockResolvedValue(inviteResult)
            await createPairApi(transport).cluster.respondToInvite(
                'invite-1',
                true,
                '123456',
                allowRemoteManagement
            )
            const request = {
                inviteId: 'invite-1',
                accept: true,
                pin: '123456',
                allowRemoteManagement
            } satisfies Parameters<typeof handleServiceBridgeInvoke<'cluster:respond-to-invite'>>[1]
            expect(transport.invoke).toHaveBeenCalledWith('cluster:respond-to-invite', request)
            await handleServiceBridgeInvoke('cluster:respond-to-invite', request)
            expect(getModularSupervisor().callProcess).toHaveBeenCalledWith(
                'broker',
                'cluster:respond-to-invite',
                request,
                35000
            )
        }
    )

    it('declines without a PIN or management choice', async () => {
        const transport = mockTransport()
        transport.invoke.mockResolvedValue(inviteResult)
        await createPairApi(transport).cluster.respondToInvite('invite-1', false)
        expect(transport.invoke).toHaveBeenCalledWith('cluster:respond-to-invite', {
            inviteId: 'invite-1',
            accept: false
        })
        await handleServiceBridgeInvoke('cluster:respond-to-invite', {
            inviteId: 'invite-1',
            accept: false
        })
        expect(getModularSupervisor().callProcess).toHaveBeenCalledWith(
            'broker',
            'cluster:respond-to-invite',
            { inviteId: 'invite-1', accept: false },
            35000
        )
    })

    it.each([
        {
            name: 'missing outbound choice',
            channel: 'cluster:invite-node',
            payload: { ipAddress: '192.168.1.2' }
        },
        {
            name: 'invalid outbound choice',
            channel: 'cluster:invite-node',
            payload: { ipAddress: '192.168.1.2', allowRemoteManagement: 'true' }
        },
        {
            name: 'missing accepted choice',
            channel: 'cluster:respond-to-invite',
            payload: { inviteId: 'invite-1', accept: true, pin: '123456' }
        },
        {
            name: 'invalid accepted choice',
            channel: 'cluster:respond-to-invite',
            payload: { inviteId: 'invite-1', accept: true, pin: '123456', allowRemoteManagement: 1 }
        },
        {
            name: 'missing accepted PIN',
            channel: 'cluster:respond-to-invite',
            payload: { inviteId: 'invite-1', accept: true, allowRemoteManagement: false }
        }
    ])('rejects malformed IPC input: $name', async ({ channel, payload }) => {
        await expect(
            Reflect.apply(handleServiceBridgeInvoke, undefined, [channel, payload])
        ).rejects.toThrow('remote management choice')
        expect(getModularSupervisor().callProcess).not.toHaveBeenCalled()
    })

    it('reconstructs only local setter fields and only the remote node selector', async () => {
        vi.mocked(getModularSupervisor().callProcess).mockResolvedValue(allowedSnapshot)
        const local = {
            peerUuid: 'peer-1',
            allowed: false,
            expectedRevision: '8',
            nodeId: 'forbidden-remote'
        }
        await handleServiceBridgeInvoke('cluster:set-remote-management', local)
        expect(getModularSupervisor().callProcess).toHaveBeenCalledWith(
            'broker',
            'cluster:set-remote-management',
            {
                peerUuid: 'peer-1',
                allowed: false,
                expectedRevision: '8'
            },
            undefined
        )
        const remote = { nodeId: 'target', peerUuid: 'forbidden-caller' }
        await handleServiceBridgeInvoke('engine:remote-management-get', remote)
        expect(getModularSupervisor().callProcess).toHaveBeenCalledWith(
            'broker',
            'engine:remote-management-get',
            { nodeId: 'target' }
        )
    })

    it('binds local read, CAS write, caller-only read and push to canonical channels', async () => {
        const transport = mockTransport()
        transport.invoke.mockResolvedValue(allowedSnapshot)
        const api = createPairApi(transport).cluster
        const callback = vi.fn<(snapshot: RemoteManagementSnapshot) => void>()
        await api.getRemoteManagement()
        await api.setRemoteManagement({ peerUuid: 'peer-1', allowed: true, expectedRevision: '1' })
        await api.getRemoteManagementForNode('target')
        api.onRemoteManagementChanged(callback)
        expect(transport.invoke.mock.calls).toEqual([
            ['cluster:get-remote-management'],
            [
                'cluster:set-remote-management',
                { peerUuid: 'peer-1', allowed: true, expectedRevision: '1' }
            ],
            ['engine:remote-management-get', { nodeId: 'target' }]
        ])
        expect(transport.subscribePush).toHaveBeenCalledWith(
            'cluster:remote-management-changed',
            callback
        )
    })

    it('does not turn baseline errors into an authoritative empty list', async () => {
        vi.mocked(getModularSupervisor().callProcess).mockRejectedValue(
            new JsonRpcResponseError(-32000, 'unavailable')
        )
        await expect(
            handleServiceBridgeInvoke('cluster:get-remote-management', undefined)
        ).rejects.toMatchObject({ code: -32000 })
    })

    it('parses local baseline and setter replies rather than relaying malformed authority', async () => {
        vi.mocked(getModularSupervisor().callProcess).mockResolvedValueOnce(allowedSnapshot)
        await expect(
            handleServiceBridgeInvoke('cluster:get-remote-management', undefined)
        ).resolves.toEqual(allowedSnapshot)
        expect(getModularSupervisor().callProcess).toHaveBeenCalledWith(
            'broker',
            'cluster:get-remote-management',
            {},
            undefined
        )
        vi.mocked(getModularSupervisor().callProcess).mockResolvedValue({
            ...allowedSnapshot,
            version: 2
        })
        await expect(
            handleServiceBridgeInvoke('cluster:get-remote-management', undefined)
        ).resolves.toEqual(unavailableSnapshot)
        await expect(
            handleServiceBridgeInvoke('cluster:set-remote-management', {
                peerUuid: 'peer-1',
                allowed: false,
                expectedRevision: '1'
            })
        ).resolves.toEqual(unavailableSnapshot)
    })

    it('parses the caller-only read separately from the local full-list reply', async () => {
        vi.mocked(getModularSupervisor().callProcess).mockResolvedValueOnce({
            version: 1,
            revision: '5',
            allowed: true,
            state: 'allowed'
        })
        await expect(
            handleServiceBridgeInvoke('engine:remote-management-get', { nodeId: 'target' })
        ).resolves.toEqual({ version: 1, revision: '5', allowed: true, state: 'allowed' })
        vi.mocked(getModularSupervisor().callProcess).mockResolvedValueOnce(allowedSnapshot)
        await expect(
            handleServiceBridgeInvoke('engine:remote-management-get', { nodeId: 'target' })
        ).resolves.toEqual(unavailableCapability)
    })

    it.each([-32010, -32011, -32012, -32199])(
        'preserves service code %s through bridge and renderer transport',
        async code => {
            const refusal = new JsonRpcResponseError(code, 'service refused')
            vi.mocked(getModularSupervisor().callProcess).mockRejectedValue(refusal)
            await expect(
                handleServiceBridgeInvoke('cluster:set-remote-management', {
                    peerUuid: 'peer-1',
                    allowed: true,
                    expectedRevision: '1'
                })
            ).rejects.toBe(refusal)
            const invoke = vi.fn<PreloadServiceTransport['invoke']>()
            invoke.mockResolvedValue({ success: false, error: 'service refused', code })
            const preload = { ...mockTransport(), invoke } satisfies PreloadServiceTransport
            const result = createPairApi(
                serviceTransport(preload)
            ).cluster.getRemoteManagementForNode('target')
            await expect(result).rejects.toBeInstanceOf(ServiceCallError)
            await expect(result).rejects.toMatchObject({ message: 'service refused', code })
            expect(REMOTE_MANAGEMENT_DENIED_CODE).toBe(-32012)
        }
    )
})

describe('supervisor permission push dispatch', () => {
    it.each([
        {
            name: 'malformed update',
            update: {
                version: 1,
                revision: '2',
                grants: [{ peerUuid: 'peer-1', allowed: true, state: 'future' }]
            }
        },
        { name: 'unsupported update', update: { ...allowedSnapshot, version: 2 } },
        { name: 'unavailable update', update: { version: 1, revision: '', grants: [] } }
    ])('forwards an invalidation after a prior allow: $name', async ({ update }) => {
        const clients: JsonRpcSocketClient[] = []
        vi.spyOn(fs, 'existsSync').mockImplementation(file => String(file).includes('nvpair-'))
        vi.spyOn(JsonRpcSocketClient.prototype, 'start').mockImplementation(function (
            this: JsonRpcSocketClient
        ) {
            clients.push(this)
        })
        const supervisor = getModularSupervisor()
        supervisor.start()
        try {
            const client = clients[0]
            if (!client) throw new Error('Supervisor did not attach the broker notification source')
            client.emit('notification', {
                source: 'broker',
                method: 'cluster:remote-management-changed',
                params: allowedSnapshot
            })
            expect(mocks.send).toHaveBeenLastCalledWith('service-bridge:push', {
                channel: 'cluster:remote-management-changed',
                payload: allowedSnapshot
            })
            client.emit('notification', {
                source: 'broker',
                method: 'cluster:remote-management-changed',
                params: update
            })
            expect(mocks.send).toHaveBeenLastCalledWith('service-bridge:push', {
                channel: 'cluster:remote-management-changed',
                payload: unavailableSnapshot
            })
            expect(mocks.send).toHaveBeenCalledTimes(2)
        } finally {
            await supervisor.stop()
        }
    })
})

describe('permission controls', () => {
    const peer: AvailableNode = {
        id: 'peer-1',
        name: 'Peer',
        ipAddress: '192.0.2.7',
        port: 14321,
        lastSeen: 1,
        trusted: false,
        clustered: false
    }
    const member: ClusterNode = {
        id: 'display-hostname',
        nodeUuid: 'peer-1',
        name: 'Peer',
        ipAddress: peer.ipAddress,
        port: peer.port,
        clusterId: 'home',
        state: 'member',
        joinedAt: 1,
        lastSeen: 1
    }
    const peerNode: NodeItem = {
        id: 'peer-1',
        name: 'Peer',
        status: 'active',
        ipAddress: peer.ipAddress,
        port: 1,
        allIpAddresses: [],
        os: 'Linux',
        topology: { cpu: { model: 'CPU', cores: 4, threads: 4 }, gpus: [], ram: 4096, storage: [] }
    }
    const settings: EngineSettingsSnapshot = {
        nodeId: 'peer-1',
        engine: 'ollama',
        revision: 1,
        appliedRevision: 1,
        sequence: 1,
        epoch: 'one',
        settings: { serverPort: 1235, proxyPort: 11434, launchText: 'serve' },
        effectiveServerPort: 1235,
        effectiveProxyPort: 11434,
        running: true,
        adopted: false,
        editable: true,
        reason: '',
        format: 'pair-arguments-v1',
        phase: 'idle',
        error: '',
        requestId: ''
    }
    const policy: NodePolicyDocument = {
        policy: '{"version":2,"revision":"5","availability":"available"}',
        revision: '5',
        activation: null,
        availability: 'available',
        recoveryRequired: false
    }
    const inviteNode = vi.fn<IClusterApi['inviteNode']>()
    const capability = vi.fn<IClusterApi['getRemoteManagementForNode']>()
    const getPermissions = vi.fn<IClusterApi['getRemoteManagement']>()
    const setPermission = vi.fn<IClusterApi['setRemoteManagement']>()
    const getPolicy = vi.fn<INodePolicyApi['get']>()
    const setPolicy = vi.fn<INodePolicyApi['set']>()
    const availability = vi.fn<INodePolicyApi['setAvailability']>()
    const getSettings = vi.fn<IPairApi['engines']['getSettings']>()
    const previewSettings = vi.fn<IPairApi['engines']['previewSettings']>()
    const applySettings = vi.fn<IPairApi['engines']['applySettings']>()
    let container: HTMLDivElement
    let root: Root

    function button(label: string): HTMLButtonElement {
        const found = Array.from(document.querySelectorAll('button')).find(
            item => item.textContent?.trim() === label || item.getAttribute('aria-label') === label
        )
        if (!found) throw new Error(`Missing ${label} button`)
        return found
    }
    async function changeText(input: HTMLInputElement | HTMLTextAreaElement | null, value: string) {
        if (!input) throw new Error('Missing text field')
        const prototype =
            input instanceof HTMLTextAreaElement
                ? HTMLTextAreaElement.prototype
                : HTMLInputElement.prototype
        const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set
        if (!setter) throw new Error('Missing value setter')
        await act(async () => {
            setter.call(input, value)
            input.dispatchEvent(new Event('input', { bubbles: true }))
        })
    }
    async function allow() {
        const input = document.querySelector<HTMLInputElement>('input[type="radio"][value="allow"]')
        if (!input) throw new Error('Missing Allow choice')
        await act(async () => input.click())
    }
    async function choosePeer() {
        await act(async () => button('Select node').click())
        const item = Array.from(
            document.querySelectorAll<HTMLElement>('[role="menuitem"],button')
        ).find(value => value.textContent?.trim() === 'Peer')
        if (!item) throw new Error('Missing Peer node menu item')
        await act(async () => item.click())
    }
    async function renderSettings() {
        await act(async () =>
            root.render(
                <EngineSettingsSection nodeId="peer-1" engineType="ollama" disabled={false} />
            )
        )
    }
    async function initializeMembership() {
        await useClusterInvitationsStore.getState().initialize({
            identity: {
                nodeUuid: 'self',
                nodeId: 'self.local',
                name: 'Self',
                certFingerprint: 'cert',
                clusterId: 'home'
            },
            info: { clusterId: 'home', isClustered: true, clusterFriendlyName: 'Home' },
            members: [member],
            pendingInvites: []
        })
    }
    beforeEach(() => {
        vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
        vi.stubGlobal('CSS', { supports: () => true })
        Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
            configurable: true,
            value: function (this: HTMLDialogElement) {
                this.open = true
            }
        })
        Object.defineProperty(HTMLDialogElement.prototype, 'close', {
            configurable: true,
            value: function (this: HTMLDialogElement) {
                this.open = false
            }
        })
        inviteNode.mockReset().mockResolvedValue({ ...inviteResult, state: 'paired' })
        capability
            .mockReset()
            .mockResolvedValue({ version: 1, revision: '1', allowed: true, state: 'allowed' })
        getPermissions.mockReset().mockResolvedValue({
            version: 1,
            revision: '8',
            grants: [{ peerUuid: 'peer-1', allowed: false, state: 'denied' }]
        })
        setPermission.mockReset()
        getPolicy.mockReset().mockResolvedValue(policy)
        setPolicy.mockReset().mockResolvedValue({ ...policy, revision: '6' })
        availability.mockReset().mockResolvedValue({ availability: 'paused' })
        getSettings.mockReset().mockResolvedValue(settings)
        previewSettings.mockReset().mockImplementation(async request => ({
            settings: request.settings,
            revision: 1,
            errors: {},
            restart: false,
            rebind: false
        }))
        applySettings.mockReset().mockResolvedValue({ revision: 2, phase: 'succeeded' })
        const api = createPairApi(mockTransport())
        api.cluster.inviteNode = inviteNode
        api.cluster.getRemoteManagementForNode = capability
        api.cluster.getRemoteManagement = getPermissions
        api.cluster.setRemoteManagement = setPermission
        api.nodePolicy.get = getPolicy
        api.nodePolicy.set = setPolicy
        api.nodePolicy.setAvailability = availability
        api.engines.getSettings = getSettings
        api.engines.previewSettings = previewSettings
        api.engines.applySettings = applySettings
        Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
        useConnectionStore.setState({ connected: true, selfId: 'self' })
        useClusterInvitationsStore.getState().cleanup()
        useDiscoveredNodesStore.setState({ nodes: [peer] })
        useNodesStore.setState({ nodes: new Map([['peer-1', peerNode]]) })
        useNodeConfigurationStore.setState({ editors: new Map(), management: new Map() })
        useEngineSettingsStore.setState({ entries: {} })
        useEngineStatusStore.setState({ statusByNode: new Map() })
        container = document.createElement('div')
        document.body.append(container)
        root = createRoot(container)
    })
    afterEach(async () => {
        await act(async () => root.unmount())
        useClusterInvitationsStore.getState().cleanup()
        useDiscoveredNodesStore.setState({ nodes: [] })
        useNodesStore.setState({ nodes: new Map() })
        useNodeConfigurationStore.setState({ editors: new Map(), management: new Map() })
        container.remove()
        vi.unstubAllGlobals()
    })

    it.each([
        { path: 'manual', allowed: false },
        { path: 'manual', allowed: true },
        { path: 'discovered', allowed: false },
        { path: 'discovered', allowed: true },
        { path: 'available card', allowed: false },
        { path: 'available card', allowed: true }
    ])(
        'sends explicit $allowed from the actual $path invitation control',
        async ({ path, allowed }) => {
            if (path === 'available card')
                await act(async () => root.render(<AvailableNodeCard node={peer} />))
            else
                await act(async () =>
                    root.render(
                        <AddNodeModal open onOpenChange={vi.fn<(open: boolean) => void>()} />
                    )
                )
            expect(document.querySelector<HTMLInputElement>('input[value="block"]')?.checked).toBe(
                true
            )
            expect(document.body.textContent).toContain(
                'This choice applies to the device you are using. Reading status and using models this device serves remain available. Engine locks and node pause still apply.'
            )
            if (allowed) await allow()
            if (path === 'manual')
                await changeText(
                    document.querySelector<HTMLInputElement>('input[placeholder="192.168.1.100"]'),
                    peer.ipAddress
                )
            const invites = Array.from(document.querySelectorAll('button')).filter(
                value => value.textContent?.trim() === 'Invite'
            )
            const action = path === 'discovered' ? invites[1] : invites[0]
            if (!action) throw new Error('Missing invitation action')
            await act(async () => action.click())
            expect(inviteNode).toHaveBeenCalledExactlyOnceWith(peer.ipAddress, allowed)
            expect(document.body.textContent).toContain(
                'Reading the saved remote-management permission'
            )
            expect(document.body.textContent).not.toContain('Pairing saved:')
            await act(async () => initializeMembership())
            expect(document.body.textContent).toContain('is blocked from changing this device')
            expect(document.body.textContent).not.toContain('may manage this device')
        }
    )

    it('saves only the UUID grant on this device with its authoritative CAS revision', async () => {
        await initializeMembership()
        let finish: ((snapshot: RemoteManagementSnapshot) => void) | undefined
        setPermission.mockImplementation(
            () =>
                new Promise(resolve => {
                    finish = resolve
                })
        )
        await act(async () => root.render(<ActiveNode node={member} />))
        await allow()
        expect(document.body.textContent).toContain('Saved: blocked')
        await act(async () => button('Save permission').click())
        expect(setPermission).toHaveBeenCalledExactlyOnceWith({
            peerUuid: 'peer-1',
            allowed: true,
            expectedRevision: '8'
        })
        expect(document.body.textContent).toContain('Saved: blocked')
        expect(button('Save permission').disabled).toBe(true)
        if (!finish) throw new Error('Missing permission request')
        await act(async () =>
            finish?.({
                version: 1,
                revision: '9',
                grants: [{ peerUuid: 'peer-1', allowed: true, state: 'allowed' }]
            })
        )
        expect(document.body.textContent).toContain('Saved: allowed')
        expect(capability).not.toHaveBeenCalled()
    })

    it('keeps a newer unsent member choice when an obsolete save completes after identity replacement', async () => {
        await initializeMembership()
        let finishOldSave: ((snapshot: RemoteManagementSnapshot) => void) | undefined
        setPermission
            .mockImplementationOnce(
                () =>
                    new Promise(resolve => {
                        finishOldSave = resolve
                    })
            )
            .mockResolvedValueOnce({
                version: 1,
                revision: '10',
                grants: [{ peerUuid: 'peer-1', allowed: true, state: 'allowed' }]
            })
        await act(async () => root.render(<ActiveNode node={member} />))
        await allow()
        await act(async () => button('Save permission').click())
        getPermissions.mockResolvedValue({
            version: 1,
            revision: '9',
            grants: [{ peerUuid: 'peer-1', allowed: false, state: 'denied' }]
        })
        await act(async () => {
            useClusterInvitationsStore.getState().hydrate({
                identity: {
                    nodeUuid: 'self',
                    nodeId: 'self.local',
                    name: 'Self',
                    certFingerprint: 'replacement-cert',
                    clusterId: 'home'
                },
                info: { clusterId: 'home', isClustered: true, clusterFriendlyName: 'Home' },
                members: [member],
                pendingInvites: []
            })
            await useClusterInvitationsStore.getState().refreshRemoteManagement()
        })
        expect(button('Save permission').disabled).toBe(false)
        await act(async () => button('Save permission').click())
        expect(document.body.textContent).toContain('Saved: allowed')
        const block = document.querySelector<HTMLInputElement>('input[value="block"]')
        if (!block) throw new Error('Missing Block choice')
        await act(async () => block.click())
        expect(block.checked).toBe(true)
        if (!finishOldSave) throw new Error('Missing obsolete save request')
        await act(async () =>
            finishOldSave?.({
                version: 1,
                revision: '9',
                grants: [{ peerUuid: 'peer-1', allowed: true, state: 'allowed' }]
            })
        )
        expect(setPermission.mock.calls).toEqual([
            [{ peerUuid: 'peer-1', allowed: true, expectedRevision: '8' }],
            [{ peerUuid: 'peer-1', allowed: true, expectedRevision: '9' }]
        ])
        expect(useClusterInvitationsStore.getState().remoteManagement).toEqual({
            version: 1,
            revision: '10',
            grants: [{ peerUuid: 'peer-1', allowed: true, state: 'allowed' }]
        })
        expect(document.body.textContent).toContain('Saved: allowed')
        expect(document.querySelector<HTMLInputElement>('input[value="block"]')?.checked).toBe(true)
    })

    it.each([
        { code: -32010, explanation: 'The saved permission changed' },
        { code: -32000, explanation: 'The save could not be confirmed' }
    ])(
        'keeps the local choice and reads authoritative state after save error $code',
        async ({ code, explanation }) => {
            await initializeMembership()
            setPermission.mockRejectedValueOnce(new ServiceCallError('save refused', code))
            getPermissions.mockResolvedValue({
                version: 1,
                revision: '9',
                grants: [{ peerUuid: 'peer-1', allowed: false, state: 'denied' }]
            })
            await act(async () => root.render(<ActiveNode node={member} />))
            await allow()
            await act(async () => button('Save permission').click())
            expect(document.body.textContent).toContain(explanation)
            expect(document.body.textContent).toContain('Saved: blocked')
            expect(document.querySelector<HTMLInputElement>('input[value="allow"]')?.checked).toBe(
                true
            )
            expect(setPermission).toHaveBeenCalledExactlyOnceWith({
                peerUuid: 'peer-1',
                allowed: true,
                expectedRevision: '8'
            })
            expect(useClusterInvitationsStore.getState().remoteManagement?.revision).toBe('9')
        }
    )

    it.each([
        { state: 'denied', revision: '1', message: 'blocked remote management' },
        { state: 'unreviewed', revision: '1', message: 'has not granted remote management' },
        { state: 'unavailable', revision: '', message: 'permission is unavailable' },
        {
            state: 'unsupported',
            revision: '',
            message: 'does not support remote management permissions'
        }
    ] satisfies {
        state: RemoteManagementCapability['state']
        revision: string
        message: string
    }[])(
        'keeps role and engine settings readable with $state edits disabled',
        async ({ state, revision, message }) => {
            capability.mockResolvedValue({ version: 1, allowed: false, state, revision })
            useNodeConfigurationStore.getState().adopt('peer-1', policy)
            useNodeConfigurationStore
                .getState()
                .setRole('peer-1', 'weak', { engine: 'ollama', model: 'draft-model' })
            await act(async () =>
                root.render(
                    <>
                        <RoleEditor
                            targetId="peer-1"
                            nodeId="peer-1"
                            onSaved={vi.fn<(saved: NodePolicySnapshot) => void>()}
                        />
                        <EngineSettingsSection
                            nodeId="peer-1"
                            engineType="ollama"
                            disabled={false}
                        />
                    </>
                )
            )
            expect(document.body.textContent).toContain(message)
            expect(button('Save roles').disabled).toBe(true)
            expect(
                useNodeConfigurationStore.getState().editors.get('peer-1')?.draft.weak?.model
            ).toBe('draft-model')
            expect(button('Apply').disabled).toBe(true)
            expect(
                document.querySelector<HTMLInputElement>('input[aria-label="Server port"]')
                    ?.disabled
            ).toBe(true)
            expect(
                document.querySelector<HTMLInputElement>('input[aria-label="Server port"]')?.value
            ).toBe('1235')
            expect(applySettings).not.toHaveBeenCalled()
            expect(setPolicy).not.toHaveBeenCalled()
            expect(capability).toHaveBeenCalledTimes(1)
        }
    )

    it('does not treat an explicit empty remote selector as this device', async () => {
        capability.mockResolvedValue({
            version: 1,
            allowed: false,
            revision: '',
            state: 'unavailable'
        })
        useNodeConfigurationStore.getState().adopt('invalid-target', policy)
        useNodeConfigurationStore
            .getState()
            .setRole('invalid-target', 'weak', { engine: 'ollama', model: 'draft-model' })
        await act(async () =>
            root.render(
                <RoleEditor
                    targetId="invalid-target"
                    nodeId=""
                    onSaved={vi.fn<(saved: NodePolicySnapshot) => void>()}
                />
            )
        )
        expect(button('Save roles').disabled).toBe(true)
        expect(document.body.textContent).toContain('permission is unavailable')
        expect(capability).toHaveBeenCalledExactlyOnceWith('')
        expect(setPolicy).not.toHaveBeenCalled()
    })

    it('keeps a settings draft after a definite permission refusal and distinguishes local veto', async () => {
        applySettings.mockRejectedValueOnce(new ServiceCallError('no authority', -32012))
        await renderSettings()
        await changeText(
            document.querySelector<HTMLInputElement>('input[aria-label="Server port"]'),
            '2345'
        )
        await act(async () => button('Apply').click())
        expect(document.body.textContent).toContain('refused remote management permission')
        expect(document.body.textContent).not.toContain('the target refused the operation')
        expect(
            document.querySelector<HTMLInputElement>('input[aria-label="Server port"]')?.value
        ).toBe('2345')
        expect(applySettings).toHaveBeenCalledTimes(1)
        expect(usePendingActionsStore.getState().isSettingsPending('peer-1', 'ollama')).toBe(false)
    })

    it.each([
        {
            name: 'engine lock',
            code: -32013,
            detail: 'engine locked on target',
            label: 'Engine locked'
        },
        {
            name: 'fresh pause',
            code: -32011,
            detail: 'refused',
            label: 'Node paused',
            state: 'paused'
        },
        {
            name: 'fresh draining',
            code: -32011,
            detail: 'refused',
            label: 'Node paused',
            state: 'draining'
        },
        {
            name: 'bounded pause message',
            code: -32011,
            detail: 'This node is paused. Resume it on this node before managing it remotely.',
            label: 'Node paused'
        },
        {
            name: 'older saved Off',
            code: -32011,
            detail: 'engine was turned off locally',
            label: 'Target refused'
        },
        {
            name: 'scoped stop',
            code: -32011,
            detail: 'local stop cancelled this operation',
            label: 'Target refused'
        },
        {
            name: 'configuration conflict',
            code: -32010,
            detail: 'revision changed',
            label: 'Configuration conflict'
        }
    ] satisfies {
        name: string
        code: number
        detail: string
        label: string
        state?: NodePolicyDocument['availability']
    }[])(
        'keeps the settings draft and target detail for $name',
        async ({ code, detail, label, state }) => {
            applySettings.mockRejectedValueOnce(new ServiceCallError(detail, code))
            await renderSettings()
            if (state) getPolicy.mockResolvedValue({ ...policy, availability: state })
            await changeText(
                document.querySelector<HTMLInputElement>('input[aria-label="Server port"]'),
                '2345'
            )
            await act(async () => button('Apply').click())
            expect(document.body.textContent).toContain(`${label}: ${detail}`)
            expect(
                document.querySelector<HTMLInputElement>('input[aria-label="Server port"]')?.value
            ).toBe('2345')
            expect(applySettings).toHaveBeenCalledTimes(1)
            expect(document.body.textContent).not.toContain('Off always wins')
        }
    )

    it('allows metadata-only settings apply to a locked engine', async () => {
        useEngineStatusStore.setState({
            statusByNode: new Map([
                [
                    'peer-1',
                    new Map([
                        ['ollama', { ...emptyEngineStatus('peer-1', 'ollama'), locked: true }]
                    ])
                ]
            ])
        })
        await renderSettings()
        await changeText(
            document.querySelector<HTMLInputElement>('input[aria-label="Server port"]'),
            '2345'
        )
        expect(button('Apply').disabled).toBe(false)
        await act(async () => button('Apply').click())
        expect(applySettings).toHaveBeenCalledTimes(1)
        expect(applySettings).toHaveBeenCalledWith(
            expect.objectContaining({
                nodeId: 'peer-1',
                engine: 'ollama',
                settings: expect.objectContaining({ serverPort: 2345 })
            })
        )
    })

    it('keeps settings read-only when the peer is unreachable rather than calling it unsupported', async () => {
        capability.mockRejectedValue(new Error('device is offline'))
        await renderSettings()
        expect(document.body.textContent).toContain('device is offline')
        expect(document.body.textContent).not.toContain(
            'does not support remote management permissions'
        )
        expect(button('Apply').disabled).toBe(true)
        expect(
            document.querySelector<HTMLInputElement>('input[aria-label="Server port"]')?.disabled
        ).toBe(true)
    })

    it('locks a now-denied settings editor without replacing its unsent draft', async () => {
        await renderSettings()
        await changeText(
            document.querySelector<HTMLInputElement>('input[aria-label="Server port"]'),
            '2345'
        )
        capability.mockResolvedValue({ version: 1, allowed: false, revision: '2', state: 'denied' })
        await act(async () =>
            useNodeConfigurationStore.getState().readManagement('peer-1', 'peer-1')
        )
        const input = document.querySelector<HTMLInputElement>('input[aria-label="Server port"]')
        expect(input?.value).toBe('2345')
        expect(input?.disabled).toBe(true)
        expect(button('Apply').disabled).toBe(true)
        expect(applySettings).not.toHaveBeenCalled()
    })

    it('does not enable a new target when the previous target returns a late allow', async () => {
        let finish: ((result: RemoteManagementCapability) => void) | undefined
        capability.mockImplementation(nodeId =>
            nodeId === 'peer-1'
                ? new Promise(resolve => {
                      finish = resolve
                  })
                : Promise.resolve({ version: 1, allowed: false, revision: '1', state: 'denied' })
        )
        await renderSettings()
        await act(async () =>
            root.render(
                <EngineSettingsSection nodeId="peer-2" engineType="ollama" disabled={false} />
            )
        )
        if (!finish) throw new Error('Missing first target capability request')
        await act(async () =>
            finish?.({ version: 1, allowed: true, revision: '1', state: 'allowed' })
        )
        expect(button('Apply').disabled).toBe(true)
        expect(document.body.textContent).toContain('blocked remote management')
        expect(
            document.querySelector<HTMLInputElement>('input[aria-label="Server port"]')?.disabled
        ).toBe(true)
        expect(useNodeConfigurationStore.getState().management.get('peer-2')?.state).toBe('denied')
        expect(applySettings).not.toHaveBeenCalled()
    })

    it('disables remote raw policy and availability while keeping their saved status visible', async () => {
        capability.mockResolvedValue({ version: 1, allowed: false, revision: '1', state: 'denied' })
        await act(async () => root.render(<RoutingSettings />))
        await choosePeer()
        expect(document.body.textContent).toContain('blocked remote management')
        expect(document.body.textContent).toContain('available')
        expect(button('Pause').disabled).toBe(true)
        expect(button('Save').disabled).toBe(true)
        expect(
            document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Node policy JSON"]')
                ?.value
        ).toBe(policy.policy)
        expect(setPolicy).not.toHaveBeenCalled()
        expect(availability).not.toHaveBeenCalled()
    })

    it('preserves an unsent remote policy through reconnect and never replays a write', async () => {
        await act(async () => root.render(<RoutingSettings />))
        await choosePeer()
        const draft = '{"version":2,"revision":"5","admission":{"maxResidentModels":2}}'
        await changeText(
            document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Node policy JSON"]'),
            draft
        )
        await act(async () => useConnectionStore.setState({ connected: false }))
        getPolicy.mockResolvedValue({
            ...policy,
            revision: '6',
            policy: '{"version":2,"revision":"6"}'
        })
        await act(async () => useConnectionStore.setState({ connected: true }))
        expect(
            document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Node policy JSON"]')
                ?.value
        ).toBe(draft)
        expect(document.body.textContent).toContain('saved policy changed')
        expect(button('Save').disabled).toBe(true)
        expect(setPolicy).not.toHaveBeenCalled()
        expect(availability).not.toHaveBeenCalled()
    })

    it.each([
        { code: -32012, message: 'refused remote management permission' },
        { code: -32011, message: 'the target refused the operation' }
    ])('preserves a raw policy draft after definite refusal $code', async ({ code, message }) => {
        setPolicy.mockRejectedValueOnce(new ServiceCallError('refused', code))
        await act(async () => root.render(<RoutingSettings />))
        await choosePeer()
        const draft = '{"version":2,"revision":"5","admission":{"maxResidentModels":2}}'
        await changeText(
            document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Node policy JSON"]'),
            draft
        )
        await act(async () => button('Save').click())
        expect(document.body.textContent).toContain(message)
        expect(document.body.textContent).not.toContain(
            'Could not confirm whether the save took effect'
        )
        expect(
            document.querySelector<HTMLTextAreaElement>('textarea[aria-label="Node policy JSON"]')
                ?.value
        ).toBe(draft)
        expect(setPolicy).toHaveBeenCalledExactlyOnceWith({
            nodeId: 'peer-1',
            policy: draft,
            expectedPersistedRevision: '5'
        })
    })

    it.each([
        { code: -32012, message: 'refused remote management permission' },
        { code: -32011, message: 'the target refused the operation' }
    ])(
        'shows availability refusal $code without changing saved availability',
        async ({ code, message }) => {
            availability.mockRejectedValueOnce(new ServiceCallError('refused', code))
            await act(async () => root.render(<RoutingSettings />))
            await choosePeer()
            await act(async () => button('Pause').click())
            expect(document.body.textContent).toContain(message)
            expect(button('Pause')).toBeDefined()
            expect(availability).toHaveBeenCalledExactlyOnceWith('paused', 'peer-1')
        }
    )
})
