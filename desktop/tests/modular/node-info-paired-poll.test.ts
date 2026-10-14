// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { MODULAR_NODE_INFO_POLL_INTERVAL_MS } from '@/shared/constants/modular-runtime'
import type { emitBridgePush } from '@/electron/service-bridge/broadcaster'

const mocks = vi.hoisted(() => ({
    push: vi.fn<typeof emitBridgePush>(),
    warn: vi.fn<(payload: { message: string; data?: { reason?: string; url?: string } }) => void>(),
    info: vi.fn<(payload: { message: string }) => void>()
}))

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({ emitBridgePush: mocks.push }))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: mocks.info,
        warn: mocks.warn,
        error: vi.fn(),
        verbose: vi.fn()
    })
}))

import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import {
    setNodeInfoPollerVisible,
    startNodeInfoPoller,
    stopNodeInfoPoller
} from '@/electron/service-bridge/node-info-poller'

const getter = vi.fn<Parameters<typeof startNodeInfoPoller>[0]>()
const fetchMock = vi.fn<typeof fetch>()
const state = getModularBridgeState()
const peer: JsonObject = {
    hostUuid: 'peer',
    name: 'paired-host',
    trusted: true,
    clustered: true,
    ipAddress: '192.0.2.10',
    port: 14318
}

function publish(nodes: JsonObject[]): void {
    state.handleNotification({
        source: 'broker',
        method: 'discovery:nodes-changed',
        params: { nodes }
    })
}

function sample(hostUuid = 'peer', utilization = 42): JsonObject {
    return {
        hostUuid,
        clusterUuid: 'principal',
        telemetryValid: true,
        msSince: 100,
        GPUs: [{ name: 'GPU', vram_bytes: 8192, vram_used_bytes: 2048, utilization_percent: 73 }],
        cpu: { name: 'CPU', cores: 8, utilization_percent: utilization },
        memory: { total_bytes: 16384, used_bytes: 4096 }
    }
}

function pendingRead(): {
    promise: Promise<JsonValue | undefined>
    resolve: (value: JsonValue) => void
    reject: (error: Error) => void
} {
    let resolve: (value: JsonValue) => void = () => {
        throw new Error('pending read not initialized')
    }
    let reject: (error: Error) => void = () => {
        throw new Error('pending read not initialized')
    }
    const promise = new Promise<JsonValue | undefined>((accept, refuse) => {
        resolve = accept
        reject = refuse
    })
    return { promise, resolve, reject }
}

async function begin(): Promise<void> {
    setNodeInfoPollerVisible(true)
    startNodeInfoPoller(getter)
    await vi.advanceTimersByTimeAsync(0)
}

describe('node-info startup before identity hydration', () => {
    afterEach(() => {
        stopNodeInfoPoller()
        setNodeInfoPollerVisible(false)
        vi.unstubAllGlobals()
        vi.useRealTimers()
    })

    it('waits for self identity before polling trusted entries, then uses loopback and the paired getter', async () => {
        vi.useFakeTimers()
        vi.stubGlobal('fetch', fetchMock)
        expect(state.getSelfId()).toBeNull()
        publish([{ ...peer, hostUuid: 'self', ipAddress: '192.0.2.20' }, peer])
        getter.mockResolvedValue(sample('peer'))
        fetchMock.mockResolvedValue(new Response(JSON.stringify(sample('self'))))
        mocks.warn.mockClear()

        await begin()
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 3)

        expect(state.getNodeInfoPollTargets()).toEqual([])
        expect(getter).not.toHaveBeenCalled()
        expect(fetchMock).not.toHaveBeenCalled()
        expect(mocks.warn).not.toHaveBeenCalled()

        state.setSelfId('self')
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)

        expect(getter).toHaveBeenCalledExactlyOnceWith('peer')
        expect(fetchMock).toHaveBeenCalledExactlyOnceWith(
            'http://127.0.0.1:14318/v1/node-info',
            expect.objectContaining({ signal: expect.any(AbortSignal) })
        )
        expect(state.getNodesInitial().nodes.self.topology.cpu.model).toBe('CPU')
        expect(state.getNodesInitial().nodes.peer.topology.cpu.model).toBe('CPU')
        expect(mocks.warn).not.toHaveBeenCalled()
    })
})

describe('paired node-info polling', () => {
    beforeEach(() => {
        vi.useFakeTimers()
        vi.stubGlobal('fetch', fetchMock)
        getter.mockReset()
        getter.mockImplementation(hostUuid => Promise.resolve(sample(hostUuid)))
        fetchMock.mockReset()
        fetchMock.mockRejectedValue(new Error('unexpected direct fetch'))
        state.setSelfId('self')
        publish([])
        publish([peer])
        mocks.push.mockClear()
        mocks.warn.mockClear()
        mocks.info.mockClear()
    })

    afterEach(() => {
        stopNodeInfoPoller()
        setNodeInfoPollerVisible(false)
        vi.unstubAllGlobals()
        vi.useRealTimers()
    })

    it('reads paired telemetry by host identity without choosing a direct endpoint', async () => {
        await begin()

        expect(getter).toHaveBeenCalledExactlyOnceWith('peer')
        expect(fetchMock).not.toHaveBeenCalled()
        expect(state.getNodesInitial().nodes.peer.topology).toMatchObject({
            cpu: { model: 'CPU', cores: 8 },
            gpus: [{ name: 'GPU', vramTotal: 8192 }],
            ram: 16384
        })
        expect(mocks.push).toHaveBeenCalledWith(
            'metrics:update',
            expect.objectContaining({
                id: 'peer',
                current: expect.objectContaining({ cpuUtilization: 42, memoryUsage: 25 })
            })
        )
    })

    it('includes a paired reverse target with no direct address or node-info port', async () => {
        publish([{ hostUuid: 'peer', trusted: true, clustered: true }])
        expect(state.getNodeInfoPollTargets()).toEqual([
            { id: 'peer', hosts: [], port: 0, paired: true, generation: expect.any(Number) }
        ])

        await begin()

        expect(getter).toHaveBeenCalledExactlyOnceWith('peer')
        expect(fetchMock).not.toHaveBeenCalled()
        expect(state.getNodesInitial().nodes.peer.topology.cpu.model).toBe('CPU')
    })

    it('keeps self, unpaired and foreign-cluster targets on direct HTTP', async () => {
        publish([
            {
                hostUuid: 'self',
                trusted: true,
                clustered: true,
                ipAddress: '192.0.2.20',
                port: 14318
            },
            { hostUuid: 'unpaired', ipAddress: '192.0.2.21', port: 14318 },
            {
                hostUuid: 'foreign',
                trusted: false,
                clustered: true,
                ipAddress: '192.0.2.22',
                port: 14318
            }
        ])
        fetchMock.mockImplementation(input => {
            const url = String(input)
            const hostUuid = url.includes('192.0.2.21')
                ? 'unpaired'
                : url.includes('192.0.2.22')
                  ? 'foreign'
                  : 'self'
            return Promise.resolve(new Response(JSON.stringify(sample(hostUuid))))
        })

        await begin()

        expect(getter).not.toHaveBeenCalled()
        expect(fetchMock.mock.calls.map(([input]) => String(input))).toEqual([
            'http://127.0.0.1:14318/v1/node-info',
            'http://192.0.2.21:14318/v1/node-info',
            'http://192.0.2.22:14318/v1/node-info'
        ])
        expect(state.getNodesInitial().nodes.self.topology.cpu.model).toBe('CPU')
        expect(state.getNodesInitial().nodes.unpaired.topology.cpu.model).toBe('CPU')
        expect(state.getNodesInitial().nodes.foreign.topology.cpu.model).toBe('CPU')
    })

    it('uses trust rather than foreign-cluster membership to select the paired getter', async () => {
        publish([{ ...peer, clustered: false }])

        await begin()

        expect(getter).toHaveBeenCalledExactlyOnceWith('peer')
        expect(fetchMock).not.toHaveBeenCalled()
    })

    it('still excludes self and unpaired targets without a usable direct endpoint', async () => {
        publish([
            { hostUuid: 'self', trusted: true, clustered: true },
            { hostUuid: 'unpaired' },
            { hostUuid: 'foreign', trusted: false, clustered: true }
        ])
        expect(state.getNodeInfoPollTargets()).toEqual([])

        await begin()

        expect(getter).not.toHaveBeenCalled()
        expect(fetchMock).not.toHaveBeenCalled()
    })

    it('backs off an outage without refreshing old metrics or changing membership, then recovers', async () => {
        state.mergeNodeInfoResponse('peer', sample('peer', 12))
        const membership = state.getAvailableNodes()
        mocks.push.mockClear()
        getter.mockRejectedValueOnce(new Error('node information unavailable'))

        await begin()

        expect(getter).toHaveBeenCalledTimes(1)
        expect(mocks.push).not.toHaveBeenCalled()
        expect(state.getAvailableNodes()).toEqual(membership)
        expect(mocks.warn).toHaveBeenCalledExactlyOnceWith(
            expect.objectContaining({
                data: expect.objectContaining({
                    reason: 'node information unavailable',
                    url: 'discovery:get-node-info(peer)'
                })
            })
        )
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)
        expect(getter).toHaveBeenCalledTimes(1)
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)

        expect(getter).toHaveBeenCalledTimes(2)
        expect(fetchMock).not.toHaveBeenCalled()
        expect(state.getAvailableNodes()).toEqual(membership)
        expect(mocks.push.mock.calls.map(([channel]) => channel)).toEqual(['metrics:update'])
        expect(mocks.push).toHaveBeenCalledWith(
            'metrics:update',
            expect.objectContaining({ current: expect.objectContaining({ cpuUtilization: 42 }) })
        )
        expect(mocks.info).toHaveBeenCalledWith(
            expect.objectContaining({ message: 'Node info is answering again' })
        )
    })

    it.each([
        { name: 'different host', response: sample('other') },
        { name: 'missing identity', response: { cpu: { name: 'wrong CPU' } } },
        { name: 'empty response', response: undefined }
    ])('rejects paired getter data with $name', async ({ response }) => {
        getter.mockResolvedValue(response)
        const membership = state.getAvailableNodes()

        await begin()

        expect(getter).toHaveBeenCalledExactlyOnceWith('peer')
        expect(mocks.push).not.toHaveBeenCalled()
        expect(fetchMock).not.toHaveBeenCalled()
        expect(state.getAvailableNodes()).toEqual(membership)
        expect(mocks.warn).toHaveBeenCalledOnce()
        expect(state.getNodesInitial().nodes.peer.topology.cpu.model).toBe('')
    })

    it.each(['success', 'failure'])('discards a stopped in-flight RPC %s', async outcome => {
        const pending = pendingRead()
        getter.mockReturnValue(pending.promise)
        await begin()
        expect(getter).toHaveBeenCalledTimes(1)
        stopNodeInfoPoller()
        if (outcome === 'success') pending.resolve(sample())
        else pending.reject(new Error('late outage'))
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 2)

        expect(mocks.push).not.toHaveBeenCalled()
        expect(mocks.warn).not.toHaveBeenCalled()
        expect(fetchMock).not.toHaveBeenCalled()
        expect(getter).toHaveBeenCalledTimes(1)
    })

    it('disowns an old target result before the next timer tick', async () => {
        const old = pendingRead()
        getter.mockReturnValueOnce(old.promise)
        await begin()
        publish([{ ...peer, port: 14319 }])
        mocks.push.mockClear()
        old.resolve(sample('peer', 99))
        await vi.advanceTimersByTimeAsync(0)

        expect(mocks.push).not.toHaveBeenCalled()
        expect(mocks.warn).not.toHaveBeenCalled()
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)
        expect(getter).toHaveBeenCalledTimes(2)
        expect(mocks.push).toHaveBeenCalledWith(
            'metrics:update',
            expect.objectContaining({ current: expect.objectContaining({ cpuUtilization: 42 }) })
        )
    })

    it.each([
        {
            name: 'trust off then on',
            replace: () => {
                publish([{ ...peer, trusted: false }])
                publish([peer])
            }
        },
        {
            name: 'removal then recreation',
            replace: () => {
                publish([])
                publish([peer])
            }
        },
        {
            name: 'broker removal with a retained proxy source',
            replace: () => {
                state.handleNotification({
                    source: 'ollama-proxy',
                    method: 'node/discovered',
                    params: {
                        id: 'peer',
                        host: 'paired-host',
                        port: 11434,
                        addresses: ['192.0.2.10'],
                        ip: '192.0.2.10'
                    }
                })
                publish([])
                publish([peer])
                state.handleNotification({
                    source: 'ollama-proxy',
                    method: 'node/removed',
                    params: { id: 'peer' }
                })
            }
        },
        {
            name: 'endpoint change then restoration',
            replace: () => {
                publish([{ ...peer, port: 14319 }])
                publish([peer])
            }
        },
        {
            name: 'self identity change then restoration',
            replace: () => {
                state.setSelfId('peer')
                state.setSelfId('self')
            }
        }
    ])('disowns an old paired sample after $name', async ({ replace }) => {
        const old = pendingRead()
        getter.mockReturnValueOnce(old.promise)
        await begin()
        replace()
        mocks.push.mockClear()
        old.resolve(sample('peer', 99))
        await vi.advanceTimersByTimeAsync(0)

        expect(mocks.push).not.toHaveBeenCalled()
        expect(mocks.warn).not.toHaveBeenCalled()
        expect(getter).toHaveBeenCalledTimes(1)
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)
        expect(getter).toHaveBeenCalledTimes(2)
        expect(fetchMock).not.toHaveBeenCalled()
        expect(mocks.push).toHaveBeenCalledWith(
            'metrics:update',
            expect.objectContaining({ current: expect.objectContaining({ cpuUtilization: 42 }) })
        )
    })

    it('keeps a pending sample current across unchanged discovery refreshes', async () => {
        const pending = pendingRead()
        getter.mockReturnValueOnce(pending.promise)
        await begin()
        const target = state.getNodeInfoPollTargets()
        await vi.advanceTimersByTimeAsync(100)
        publish([peer])
        await vi.advanceTimersByTimeAsync(100)
        publish([peer])
        expect(state.getNodeInfoPollTargets()).toEqual(target)
        mocks.push.mockClear()
        pending.resolve(sample('peer', 99))
        await vi.advanceTimersByTimeAsync(0)

        expect(getter).toHaveBeenCalledTimes(1)
        expect(fetchMock).not.toHaveBeenCalled()
        expect(mocks.warn).not.toHaveBeenCalled()
        expect(mocks.push).toHaveBeenCalledWith(
            'metrics:update',
            expect.objectContaining({ current: expect.objectContaining({ cpuUtilization: 99 }) })
        )
    })

    it('disowns a direct sample after endpoint eligibility is withdrawn and restored', async () => {
        const direct = { ...peer, trusted: false, clustered: false }
        publish([direct])
        let resolve: (response: Response) => void = () => {
            throw new Error('direct read not initialized')
        }
        fetchMock.mockReturnValueOnce(
            new Promise<Response>(accept => {
                resolve = accept
            })
        )
        await begin()
        publish([{ ...direct, ipAddress: '', ipAddresses: [] }])
        expect(state.getNodeInfoPollTargets()).toEqual([])
        publish([direct])
        mocks.push.mockClear()
        resolve(new Response(JSON.stringify(sample('peer', 99))))
        await vi.advanceTimersByTimeAsync(0)

        expect(mocks.push).not.toHaveBeenCalled()
        expect(mocks.warn).not.toHaveBeenCalled()
        expect(getter).not.toHaveBeenCalled()
        fetchMock.mockResolvedValue(new Response(JSON.stringify(sample('peer', 24))))
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)
        expect(fetchMock).toHaveBeenCalledTimes(2)
        expect(mocks.push).toHaveBeenCalledWith(
            'metrics:update',
            expect.objectContaining({ current: expect.objectContaining({ cpuUtilization: 24 }) })
        )
    })

    it('starts a new target while an old RPC is pending and preserves its in-flight gate', async () => {
        const old = pendingRead()
        const current = pendingRead()
        getter.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise)
        await begin()
        publish([{ ...peer, port: 14319 }])
        mocks.push.mockClear()
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)
        expect(getter).toHaveBeenCalledTimes(2)
        old.resolve(sample('peer', 99))
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)

        expect(getter).toHaveBeenCalledTimes(2)
        expect(mocks.push).not.toHaveBeenCalled()
        current.resolve(sample())
        await vi.advanceTimersByTimeAsync(0)
        expect(mocks.push).toHaveBeenCalledWith(
            'metrics:update',
            expect.objectContaining({ current: expect.objectContaining({ cpuUtilization: 42 }) })
        )
    })

    it('preserves the restarted run gate when an abandoned RPC finishes', async () => {
        const old = pendingRead()
        const current = pendingRead()
        getter.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise)
        await begin()
        stopNodeInfoPoller()
        startNodeInfoPoller(getter)
        old.resolve(sample('peer', 99))
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 2)

        expect(getter).toHaveBeenCalledTimes(2)
        expect(mocks.push).not.toHaveBeenCalled()
        current.resolve(sample())
        await vi.advanceTimersByTimeAsync(0)
        expect(mocks.push).toHaveBeenCalledWith(
            'metrics:update',
            expect.objectContaining({ current: expect.objectContaining({ cpuUtilization: 42 }) })
        )
    })

    it('retries a changed paired target immediately during an outage backoff', async () => {
        getter.mockRejectedValueOnce(new Error('node information unavailable'))
        await begin()
        publish([{ ...peer, port: 14319 }])
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)

        expect(getter).toHaveBeenCalledTimes(2)
        expect(fetchMock).not.toHaveBeenCalled()
        expect(state.getNodesInitial().nodes.peer.topology.cpu.model).toBe('CPU')
        expect(mocks.info).toHaveBeenCalledWith(
            expect.objectContaining({ message: 'Node info is answering again' })
        )
    })

    it('discards removed-peer results without resurrecting membership', async () => {
        const pending = pendingRead()
        getter.mockReturnValue(pending.promise)
        await begin()
        publish([])
        mocks.push.mockClear()
        pending.resolve(sample())
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)

        expect(getter).toHaveBeenCalledTimes(1)
        expect(state.getAvailableNodes()).toEqual([])
        expect(mocks.push).not.toHaveBeenCalled()
        expect(mocks.warn).not.toHaveBeenCalled()
    })

    it('switches to direct polling after trust changes while discarding the old paired read', async () => {
        const pending = pendingRead()
        getter.mockReturnValue(pending.promise)
        await begin()
        publish([{ ...peer, trusted: false, clustered: false }])
        mocks.push.mockClear()
        pending.resolve(sample('peer', 99))
        await vi.advanceTimersByTimeAsync(0)
        expect(mocks.push).not.toHaveBeenCalled()
        fetchMock.mockResolvedValue(new Response(JSON.stringify(sample('peer', 24))))
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS)

        expect(getter).toHaveBeenCalledTimes(1)
        expect(fetchMock).toHaveBeenCalledExactlyOnceWith(
            'http://192.0.2.10:14318/v1/node-info',
            expect.objectContaining({ signal: expect.any(AbortSignal) })
        )
        expect(mocks.push).toHaveBeenCalledWith(
            'metrics:update',
            expect.objectContaining({ current: expect.objectContaining({ cpuUtilization: 24 }) })
        )
    })

    it('polls healthy peers on cadence while another paired read is pending', async () => {
        const pending = pendingRead()
        publish([peer, { ...peer, hostUuid: 'healthy' }])
        getter.mockImplementation(hostUuid =>
            hostUuid === 'peer' ? pending.promise : Promise.resolve(sample(hostUuid))
        )
        await begin()
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 2)

        expect(getter.mock.calls).toEqual([['peer'], ['healthy'], ['healthy'], ['healthy']])
        expect(fetchMock).not.toHaveBeenCalled()
        expect(state.getNodesInitial().nodes.healthy.topology.cpu.model).toBe('CPU')
        pending.resolve(sample())
        await vi.advanceTimersByTimeAsync(0)
    })

    it('keeps paired polling paused while every window is hidden', async () => {
        startNodeInfoPoller(getter)
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 3)
        expect(getter).not.toHaveBeenCalled()
        setNodeInfoPollerVisible(true)
        await vi.advanceTimersByTimeAsync(0)
        expect(getter).toHaveBeenCalledTimes(1)
        setNodeInfoPollerVisible(false)
        await vi.advanceTimersByTimeAsync(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 3)
        expect(getter).toHaveBeenCalledTimes(1)
        expect(fetchMock).not.toHaveBeenCalled()
    })
})
