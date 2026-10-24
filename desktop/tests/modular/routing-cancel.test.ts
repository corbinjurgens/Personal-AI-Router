// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import type { JsonValue } from '@/electron/service-bridge/json-rpc-client'
import type { RoutingCancelParams, RoutingCancelResult } from '@/shared/types/routing'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type { WsInvokeRequest } from '@/shared/types/ws-channels'
import { createPairApi } from '@/ui/api/pair-api'

const mocks = vi.hoisted(() => ({
    callProcess: vi.fn<ReturnType<typeof getModularSupervisor>['callProcess']>()
}))

vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => ({ callProcess: mocks.callProcess })
}))
vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => ({}),
    isUpstreamUnreachableError: () => false,
    parseServiceErrors: () => []
}))
vi.mock('@/electron/service-bridge/model-catalog', () => ({
    getEngineHubModels: () => ({ models: [] })
}))

import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import { JsonRpcResponseError } from '@/electron/service-bridge/json-rpc-client'
import { routingActiveParams, routingCancelParams } from '@/electron/service-bridge/routing'

const key = {
    originNodeId: 'local-node',
    proxyRunId: 'proxy:run-1',
    requestId: '9223372036854775807'
}
const cancel = { request: key, regenerate: false } satisfies RoutingCancelParams

const malformedKeys: { name: string; value: JsonValue }[] = [
    { name: 'null key', value: null },
    { name: 'array key', value: [] },
    { name: 'missing origin', value: { proxyRunId: 'run', requestId: '1' } },
    { name: 'missing run', value: { originNodeId: 'node', requestId: '1' } },
    { name: 'missing counter', value: { originNodeId: 'node', proxyRunId: 'run' } },
    { name: 'foreign field', value: { ...key, model: 'private-model' } },
    { name: 'case alias', value: { ...key, RequestId: '1' } },
    { name: 'empty origin', value: { ...key, originNodeId: '' } },
    { name: 'non-ASCII origin', value: { ...key, originNodeId: 'ノード' } },
    { name: 'numeric origin', value: { ...key, originNodeId: 1 } },
    { name: 'invalid first character', value: { ...key, proxyRunId: '-run' } },
    { name: 'overlong run', value: { ...key, proxyRunId: 'a'.repeat(129) } },
    { name: 'trailing identity newline', value: { ...key, proxyRunId: 'run\n' } },
    { name: 'null run', value: { ...key, proxyRunId: null } },
    { name: 'numeric counter', value: { ...key, requestId: 9007199254740992 } },
    { name: 'zero counter', value: { ...key, requestId: '0' } },
    { name: 'leading zero', value: { ...key, requestId: '01' } },
    { name: 'negative counter', value: { ...key, requestId: '-1' } },
    { name: 'signed counter', value: { ...key, requestId: '+1' } },
    { name: 'decimal counter', value: { ...key, requestId: '1.0' } },
    { name: 'exponent counter', value: { ...key, requestId: '1e2' } },
    { name: 'counter overflow', value: { ...key, requestId: '9223372036854775808' } },
    { name: 'overlong counter', value: { ...key, requestId: '1'.repeat(100) } },
    { name: 'trailing counter newline', value: { ...key, requestId: '1\n' } }
]

beforeEach(() => {
    mocks.callProcess.mockReset()
})

describe('routing cancellation bridge', () => {
    it.each<RoutingCancelResult['status']>([
        'not_found',
        'conflict_committed',
        'regenerating',
        'cancelled'
    ])('returns the exact %s outcome through one broker call', async status => {
        mocks.callProcess.mockResolvedValue({ status })
        await expect(handleServiceBridgeInvoke('routing:cancel', cancel)).resolves.toEqual({
            status
        })
        expect(mocks.callProcess.mock.calls).toEqual([
            [
                'broker',
                'routing:cancel',
                {
                    request: {
                        originNodeId: 'local-node',
                        proxyRunId: 'proxy:run-1',
                        requestId: '9223372036854775807'
                    },
                    regenerate: false
                }
            ]
        ])
    })

    it('forwards an explicit regenerate request without converting its exact counter', async () => {
        mocks.callProcess.mockResolvedValue({ status: 'regenerating' })
        await handleServiceBridgeInvoke('routing:cancel', {
            request: { ...key, requestId: '9007199254740993' },
            regenerate: true
        })
        expect(mocks.callProcess.mock.calls).toEqual([
            [
                'broker',
                'routing:cancel',
                {
                    request: {
                        originNodeId: 'local-node',
                        proxyRunId: 'proxy:run-1',
                        requestId: '9007199254740993'
                    },
                    regenerate: true
                }
            ]
        ])
    })

    it.each(malformedKeys)('rejects $name before forwarding', async ({ value }) => {
        const payload: WsInvokeRequest<'routing:cancel'> = JSON.parse(
            JSON.stringify({ request: value, regenerate: false })
        )
        await expect(handleServiceBridgeInvoke('routing:cancel', payload)).rejects.toThrow()
        expect(mocks.callProcess).not.toHaveBeenCalled()
    })

    it.each<{ name: string; value: JsonValue }>([
        { name: 'null params', value: null },
        { name: 'array params', value: [] },
        { name: 'missing request', value: { regenerate: false } },
        { name: 'missing regenerate', value: { request: key } },
        { name: 'null regenerate', value: { request: key, regenerate: null } },
        { name: 'string regenerate', value: { request: key, regenerate: 'false' } },
        { name: 'numeric regenerate', value: { request: key, regenerate: 0 } },
        { name: 'content field', value: { ...cancel, messages: ['private-content'] } },
        { name: 'peer target', value: { ...cancel, nodeId: 'peer' } }
    ])('rejects $name before forwarding', async ({ value }) => {
        const payload: WsInvokeRequest<'routing:cancel'> = JSON.parse(JSON.stringify(value))
        await expect(handleServiceBridgeInvoke('routing:cancel', payload)).rejects.toThrow()
        expect(mocks.callProcess).not.toHaveBeenCalled()
    })

    it.each<{ name: string; value: JsonValue | undefined }>([
        { name: 'absent response', value: undefined },
        { name: 'null response', value: null },
        { name: 'array response', value: [] },
        { name: 'missing status', value: {} },
        { name: 'legacy success', value: { ok: true } },
        { name: 'null status', value: { status: null } },
        { name: 'boolean status', value: { status: true } },
        { name: 'unknown status', value: { status: 'success' } },
        { name: 'case variant', value: { status: 'Cancelled' } },
        { name: 'content leak', value: { status: 'cancelled', content: 'private-content' } }
    ])('rejects $name rather than fabricating success', async ({ value }) => {
        mocks.callProcess.mockResolvedValue(value)
        await expect(handleServiceBridgeInvoke('routing:cancel', cancel)).rejects.toThrow()
        expect(mocks.callProcess).toHaveBeenCalledTimes(1)
    })
})

describe('routing active bridge', () => {
    it('returns only exact canonical keys in broker order', async () => {
        mocks.callProcess.mockResolvedValue({
            requests: [{ ...key, requestId: '9' }, { ...key, requestId: '10' }, key]
        })
        await expect(handleServiceBridgeInvoke('routing:active', {})).resolves.toEqual({
            requests: [
                { originNodeId: 'local-node', proxyRunId: 'proxy:run-1', requestId: '9' },
                { originNodeId: 'local-node', proxyRunId: 'proxy:run-1', requestId: '10' },
                {
                    originNodeId: 'local-node',
                    proxyRunId: 'proxy:run-1',
                    requestId: '9223372036854775807'
                }
            ]
        })
        expect(mocks.callProcess.mock.calls).toEqual([['broker', 'routing:active', {}]])
    })

    it('accepts an explicitly empty active snapshot', async () => {
        mocks.callProcess.mockResolvedValue({ requests: [] })
        await expect(handleServiceBridgeInvoke('routing:active', {})).resolves.toEqual({
            requests: []
        })
    })

    it.each<{ name: string; value: JsonValue }>([
        { name: 'null params', value: null },
        { name: 'array params', value: [] },
        { name: 'string params', value: '' },
        { name: 'peer target', value: { nodeId: 'peer' } },
        { name: 'request filter', value: { request: key } }
    ])('rejects $name without a broker call', async ({ value }) => {
        const payload: WsInvokeRequest<'routing:active'> = JSON.parse(JSON.stringify(value))
        await expect(handleServiceBridgeInvoke('routing:active', payload)).rejects.toThrow()
        expect(mocks.callProcess).not.toHaveBeenCalled()
    })

    it.each<{ name: string; value: JsonValue | undefined }>([
        { name: 'absent response', value: undefined },
        { name: 'null response', value: null },
        { name: 'array response', value: [] },
        { name: 'missing requests', value: {} },
        { name: 'null requests', value: { requests: null } },
        { name: 'object requests', value: { requests: {} } },
        { name: 'content leak', value: { requests: [], messages: ['private-content'] } }
    ])('rejects $name', async ({ value }) => {
        mocks.callProcess.mockResolvedValue(value)
        await expect(handleServiceBridgeInvoke('routing:active', {})).rejects.toThrow()
        expect(mocks.callProcess).toHaveBeenCalledTimes(1)
    })

    it.each(malformedKeys)('rejects an active entry with $name', async ({ value }) => {
        mocks.callProcess.mockResolvedValue({ requests: [key, value] })
        await expect(handleServiceBridgeInvoke('routing:active', {})).rejects.toThrow()
    })
})

describe('routing control errors', () => {
    it('requires params even for an empty active request', () => {
        expect(() => routingActiveParams(undefined)).toThrow()
        expect(() => routingCancelParams(undefined)).toThrow()
        expect(mocks.callProcess).not.toHaveBeenCalled()
    })

    it.each(['cancel', 'active'])('preserves the upstream %s RPC rejection', async method => {
        const error = new JsonRpcResponseError(-32000, 'routing snapshot busy; retry')
        mocks.callProcess.mockRejectedValue(error)
        const result =
            method === 'cancel'
                ? handleServiceBridgeInvoke('routing:cancel', cancel)
                : handleServiceBridgeInvoke('routing:active', {})
        await expect(result).rejects.toBe(error)
        expect(mocks.callProcess).toHaveBeenCalledTimes(1)
    })

    it('accepts the wire identity length and punctuation boundary', async () => {
        const request = {
            originNodeId: 'a'.repeat(128),
            proxyRunId: 'A._:-09',
            requestId: '1'
        }
        mocks.callProcess.mockResolvedValue({ requests: [request] })
        await expect(handleServiceBridgeInvoke('routing:active', {})).resolves.toEqual({
            requests: [request]
        })
    })
})

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

describe('Pair routing API', () => {
    it('binds cancellation to the public typed channel', async () => {
        const transport = mockTransport()
        transport.invoke.mockResolvedValue({ status: 'conflict_committed' })
        const api = createPairApi(transport)
        await expect(api.routing.cancel(cancel)).resolves.toEqual({
            status: 'conflict_committed'
        })
        expect(transport.invoke.mock.calls).toEqual([['routing:cancel', cancel]])
    })

    it('sends the exact empty object for active keys', async () => {
        const transport = mockTransport()
        transport.invoke.mockResolvedValue({ requests: [key] })
        await expect(createPairApi(transport).routing.active()).resolves.toEqual({
            requests: [key]
        })
        expect(transport.invoke.mock.calls).toEqual([['routing:active', {}]])
    })

    it.each(['cancel', 'active'])('preserves a transport %s rejection', async method => {
        const transport = mockTransport()
        const error = new JsonRpcResponseError(-32601, 'method not found')
        transport.invoke.mockRejectedValue(error)
        const api = createPairApi(transport)
        await expect(
            method === 'cancel' ? api.routing.cancel(cancel) : api.routing.active()
        ).rejects.toBe(error)
        expect(transport.invoke).toHaveBeenCalledTimes(1)
    })
})
