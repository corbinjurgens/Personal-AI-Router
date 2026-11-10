// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { NodeNameResult } from '@/shared/types/cluster'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type { StructuredLogger } from '@/shared/types/log'
import type { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import type { getEngineHubModels } from '@/electron/service-bridge/model-catalog'
import type {
    startNodeInfoPoller,
    stopNodeInfoPoller
} from '@/electron/service-bridge/node-info-poller'
import { JsonRpcSocketClient, type JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { parseNodeNameResult } from '@/electron/service-bridge/cluster-json'
import { emitBridgePush } from '@/electron/service-bridge/broadcaster'
import type { createOverviewWindow } from '@/electron/window'
import { createPairApi } from '@/ui/api/pair-api'

type BridgeState = ReturnType<typeof getModularBridgeState>
const mocks = vi.hoisted(() => ({
    state: {
        handleNotification: vi.fn<BridgeState['handleNotification']>(),
        clearWorkloads: vi.fn<BridgeState['clearWorkloads']>(),
        reconcileModelCopies: vi.fn<BridgeState['reconcileModelCopies']>(),
        invalidateModelObservations: vi.fn<BridgeState['invalidateModelObservations']>(),
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

const renamed = {
    nodeUuid: 'self-uuid',
    name: 'Studio',
    customName: 'Studio'
} satisfies NodeNameResult

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

describe('node name result parsing', () => {
    it('keeps a custom name', () => {
        expect(parseNodeNameResult(renamed)).toEqual(renamed)
    })

    it('keeps a null custom name for the generated default', () => {
        const value = { nodeUuid: 'self-uuid', name: 'PAIR 1a2b3c4d', customName: null }
        expect(parseNodeNameResult(value)).toEqual(value)
    })

    it.each<{ name: string; value: JsonValue | undefined }>([
        { name: 'missing payload', value: undefined },
        { name: 'missing node UUID', value: { name: 'Studio', customName: null } },
        { name: 'missing name', value: { nodeUuid: 'self-uuid', customName: null } },
        { name: 'missing custom name', value: { nodeUuid: 'self-uuid', name: 'Studio' } },
        { name: 'numeric custom name', value: { ...renamed, customName: 1 } }
    ])('rejects $name', ({ value }) => {
        expect(() => parseNodeNameResult(value)).toThrow('Invalid node name response')
    })
})

describe('set node name invoke', () => {
    beforeEach(() => {
        vi.spyOn(getModularSupervisor(), 'callProcess').mockResolvedValue(renamed)
    })

    it('relays the name from the renderer API through the broker', async () => {
        const transport = mockTransport()
        transport.invoke.mockResolvedValue(renamed)
        await expect(createPairApi(transport).cluster.setNodeName('Studio')).resolves.toEqual(
            renamed
        )
        expect(transport.invoke).toHaveBeenCalledExactlyOnceWith('cluster:set-node-name', {
            name: 'Studio'
        })
        await expect(
            handleServiceBridgeInvoke('cluster:set-node-name', { name: 'Studio' })
        ).resolves.toEqual(renamed)
        expect(getModularSupervisor().callProcess).toHaveBeenCalledExactlyOnceWith(
            'broker',
            'cluster:set-node-name',
            { name: 'Studio' },
            undefined
        )
    })

    it('sends null to return to the generated default', async () => {
        const cleared = { nodeUuid: 'self-uuid', name: 'PAIR 1a2b3c4d', customName: null }
        vi.mocked(getModularSupervisor().callProcess).mockResolvedValue(cleared)
        await expect(
            handleServiceBridgeInvoke('cluster:set-node-name', { name: null })
        ).resolves.toEqual(cleared)
        expect(getModularSupervisor().callProcess).toHaveBeenCalledExactlyOnceWith(
            'broker',
            'cluster:set-node-name',
            { name: null },
            undefined
        )
    })

    it('subscribes the renderer to the name push', () => {
        const transport = mockTransport()
        const callback = vi.fn<(result: NodeNameResult) => void>()
        createPairApi(transport).cluster.onNodeNameChanged(callback)
        expect(transport.subscribePush).toHaveBeenCalledExactlyOnceWith(
            'cluster:node-name-changed',
            callback
        )
    })
})

describe('supervisor node name push dispatch', () => {
    it('emits parsed name updates and drops malformed ones', async () => {
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
                method: 'cluster:node-name-changed',
                params: renamed
            })
            expect(emitBridgePush).toHaveBeenCalledExactlyOnceWith(
                'cluster:node-name-changed',
                renamed
            )
            client.emit('notification', {
                source: 'broker',
                method: 'cluster:node-name-changed',
                params: { nodeUuid: 'self-uuid' }
            })
            expect(emitBridgePush).toHaveBeenCalledOnce()
        } finally {
            await supervisor.stop()
        }
    })
})
