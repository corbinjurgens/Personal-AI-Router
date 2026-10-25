// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import type { getEngineHubModels } from '@/electron/service-bridge/model-catalog'
import type { emitBridgePush } from '@/electron/service-bridge/broadcaster'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'

type Supervisor = ReturnType<typeof getModularSupervisor>
const mocks = vi.hoisted(() => ({
    call: vi.fn<Supervisor['callProcess']>(),
    clearSolo: vi.fn<Supervisor['clearAutoCreatedSoloForInvite']>()
}))

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() }
}))
vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => ({
        callProcess: mocks.call,
        clearAutoCreatedSoloForInvite: mocks.clearSolo
    })
}))
vi.mock('@/electron/service-bridge/model-catalog', () => ({
    getEngineHubModels: vi.fn<typeof getEngineHubModels>()
}))
vi.mock('@/electron/service-bridge/broadcaster', () => ({
    emitBridgePush: vi.fn<typeof emitBridgePush>()
}))

import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'

const peerId = 'peer-uuid'
const address = '192.0.2.61'
const state = getModularBridgeState()

function projectNode(id = peerId, name = 'scanner-host'): void {
    state.handleNotification({
        source: 'broker',
        method: 'discovery:nodes-changed',
        params: {
            nodes: [{ hostUuid: id, id: name, name, ipAddress: address, port: 14318 }]
        }
    })
}

function workerEntries(nodes: JsonObject[], removed = true): void {
    mocks.call.mockImplementation(async (_process, method): Promise<JsonValue> => {
        if (method === 'nodes/list') return { nodes }
        if (method === 'node/remove') return { removed }
        if (method === 'nodes:remove') return { nodeId: peerId, removed: true }
        if (method === 'cluster:leave') return { left: true }
        throw new Error(`Unexpected method: ${method}`)
    })
}

describe('manual discovery cleanup during cluster removal', () => {
    beforeEach(() => {
        vi.resetAllMocks()
        state.setSelfId('self-uuid')
        projectNode()
    })

    it('uses the named entry ID when the scanner projection wins', async () => {
        workerEntries([
            {
                id: 'manual:name:saved-alias',
                name: 'saved-alias',
                address: 'peer.example',
                hostUuid: peerId
            }
        ])
        expect(state.getNodeHostname(peerId)).toBe('scanner-host')
        expect(state.getNodeAddresses(peerId)).toEqual([address])

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call.mock.calls).toEqual([
            ['broker', 'nodes/list'],
            ['broker', 'node/remove', { id: 'manual:name:saved-alias' }],
            ['broker', 'nodes:remove', { nodeId: peerId }, undefined]
        ])
        expect(result).toEqual({ nodeId: peerId, removed: true })
    })

    it('resolves an address-only entry that has not reported a host UUID', async () => {
        workerEntries([{ id: `manual:${address}`, address }])

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call).toHaveBeenNthCalledWith(2, 'broker', 'node/remove', {
            id: `manual:${address}`
        })
        expect(mocks.call).toHaveBeenCalledTimes(3)
        expect(result).toEqual({ nodeId: peerId, removed: true })
    })

    it('refuses cleanup for a manual projection missing from the current list', async () => {
        projectNode(peerId, 'manual:name:saved-alias')
        workerEntries([])

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call.mock.calls).toEqual([
            ['broker', 'nodes/list'],
            ['broker', 'nodes:remove', { nodeId: peerId }, undefined]
        ])
        expect(result).toEqual({
            nodeId: peerId,
            removed: true,
            manualCleanupError: 'No saved manual entry matches this node'
        })
    })

    it('silently skips cleanup for an ordinary discovered peer with no manual entry', async () => {
        workerEntries([])

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call.mock.calls).toEqual([
            ['broker', 'nodes/list'],
            ['broker', 'nodes:remove', { nodeId: peerId }, undefined]
        ])
        expect(result).toEqual({ nodeId: peerId, removed: true })
    })

    it('does not remove a different host with the same address and display name', async () => {
        workerEntries([
            {
                id: 'manual:name:scanner-host',
                name: 'scanner-host',
                address,
                hostUuid: 'other-uuid'
            }
        ])

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call).toHaveBeenCalledTimes(2)
        expect(mocks.call).not.toHaveBeenCalledWith('broker', 'node/remove', expect.anything())
        expect(result).toEqual({ nodeId: peerId, removed: true })
    })

    it('does not match an unrelated entry by display name when its host UUID is absent', async () => {
        workerEntries([
            { id: 'manual:name:scanner-host', name: 'scanner-host', address: '192.0.2.99' }
        ])

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call.mock.calls).toEqual([
            ['broker', 'nodes/list'],
            ['broker', 'nodes:remove', { nodeId: peerId }, undefined]
        ])
        expect(result).toEqual({ nodeId: peerId, removed: true })
    })

    it('uses a canonical manual projection ID even when the entry address has changed', async () => {
        projectNode(peerId, 'manual:name:saved-alias')
        workerEntries([{ id: 'manual:name:saved-alias', address: '192.0.2.99' }])

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call).toHaveBeenNthCalledWith(2, 'broker', 'node/remove', {
            id: 'manual:name:saved-alias'
        })
        expect(mocks.call).toHaveBeenCalledTimes(3)
        expect(result).toEqual({ nodeId: peerId, removed: true })
    })

    it('returns a cleanup error when the worker refuses the resolved ID', async () => {
        workerEntries([{ id: 'manual:name:saved-alias', address, hostUuid: peerId }], false)

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call).toHaveBeenNthCalledWith(2, 'broker', 'node/remove', {
            id: 'manual:name:saved-alias'
        })
        expect(result).toEqual({
            nodeId: peerId,
            removed: true,
            manualCleanupError: 'The saved manual entry was not removed'
        })
    })

    it('refuses a matching entry without an ID while still removing the member', async () => {
        workerEntries([{ name: 'saved-alias', address, hostUuid: peerId }])

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call).toHaveBeenCalledTimes(2)
        expect(result).toEqual({
            nodeId: peerId,
            removed: true,
            manualCleanupError: 'The matching manual entry has no canonical ID'
        })
    })

    it('refuses ambiguous manual aliases instead of guessing an ID', async () => {
        workerEntries([
            { id: 'manual:name:first', address, hostUuid: peerId },
            { id: 'manual:name:second', address, hostUuid: peerId }
        ])

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call).toHaveBeenCalledTimes(2)
        expect(result).toEqual({
            nodeId: peerId,
            removed: true,
            manualCleanupError: 'More than one saved manual entry matches this node'
        })
    })

    it('preserves self-departure when reading manual entries fails', async () => {
        projectNode('self-uuid', 'manual:name:self')
        mocks.call.mockRejectedValueOnce(new Error('manual-nodes unavailable'))
        mocks.call.mockResolvedValueOnce({ left: true })

        const result = await handleServiceBridgeInvoke('nodes:remove-member', {
            nodeId: 'self-uuid'
        })

        expect(mocks.call.mock.calls).toEqual([
            ['broker', 'nodes/list'],
            ['broker', 'cluster:leave', {}, undefined]
        ])
        expect(mocks.clearSolo).toHaveBeenCalledOnce()
        expect(result).toEqual({
            nodeId: 'self-uuid',
            removed: true,
            manualCleanupError: 'manual-nodes unavailable'
        })
    })

    it('does not report manual cleanup failure for an ordinary peer when the worker is unavailable', async () => {
        mocks.call.mockRejectedValueOnce(new Error('manual-nodes unavailable'))
        mocks.call.mockResolvedValueOnce({ nodeId: peerId, removed: true })

        const result = await handleServiceBridgeInvoke('nodes:remove-member', { nodeId: peerId })

        expect(mocks.call.mock.calls).toEqual([
            ['broker', 'nodes/list'],
            ['broker', 'nodes:remove', { nodeId: peerId }, undefined]
        ])
        expect(result).toEqual({ nodeId: peerId, removed: true })
    })
})
