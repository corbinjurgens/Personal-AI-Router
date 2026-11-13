// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import os from 'os'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { initPlatform } from '@/electron/globals'
import type { JsonRpcNotification } from '@/electron/service-bridge/json-rpc-client'

const PEER = 'uuid-remote-peer'
const SELF = 'uuid-self'

const mocks = vi.hoisted(() => {
    const calls: string[] = []
    return { applyRemoteEngineFacts: vi.fn(), calls }
})

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        verbose: vi.fn()
    })
}))
vi.mock('@/electron/config/ui-config', () => ({ isFirstRun: () => false }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({ emitBridgePush: vi.fn() }))
vi.mock('@/electron/service-bridge/node-info-poller', () => ({
    startNodeInfoPoller: vi.fn(),
    stopNodeInfoPoller: vi.fn()
}))
vi.mock('@/electron/service-bridge/modular-state', () => {
    // Every bridge-state method the broker-ready pass touches is a no-op,
    // except the two this test observes.
    const known: Record<string, unknown> = {
        getSelfId: () => SELF,
        getAvailableNodes: () => [{ id: PEER }],
        applyRemoteEngineFacts: mocks.applyRemoteEngineFacts,
        getPendingInvites: () => [],
        getProxyPort: () => null
    }
    const state = new Proxy(known, {
        get: (target, key: string) => (key in target ? target[key] : () => undefined)
    })
    return {
        getModularBridgeState: () => state,
        isUpstreamUnreachableError: () => false,
        parseServiceErrors: () => [],
        PROXY_ENGINES: [],
        PROXY_NODE_SOURCES: []
    }
})
vi.mock('@/electron/service-bridge/workload-projection', () => ({
    parseWorkloadsInitial: () => []
}))

import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'

// The supervisor's notification handler, broker-ready flag and broker-ready
// pass are private; element access reaches them with their declared types.
const supervisor = getModularSupervisor()

function notify(params: JsonRpcNotification['params']): void {
    supervisor['handleNotification']({
        source: 'broker',
        method: 'engine:remote-installed-changed',
        params
    })
}

describe('remote engine status push', () => {
    beforeEach(() => {
        vi.useFakeTimers()
        mocks.applyRemoteEngineFacts.mockReset()
        mocks.calls.length = 0
        supervisor['brokerReady'] = true
        vi.spyOn(supervisor, 'callProcess').mockImplementation(async (_name, method) => {
            mocks.calls.push(method)
            if (method === 'engine:remote-get-installed') return { engines: [] }
            throw new Error('not served in this test')
        })
    })

    afterEach(() => {
        vi.restoreAllMocks()
        vi.useRealTimers()
    })

    it('applies a pushed engine list to that node', () => {
        const engines = [{ engine: 'llamacpp', installed: true, running: true, healthy: true }]
        notify({ nodeId: PEER, engines })

        expect(mocks.applyRemoteEngineFacts).toHaveBeenCalledTimes(1)
        expect(mocks.applyRemoteEngineFacts).toHaveBeenCalledWith(PEER, { engines })
    })

    it('ignores malformed, self and pre-ready notifications', () => {
        notify({ nodeId: PEER })
        notify({ nodeId: 7, engines: [] })
        notify({ nodeId: PEER, engines: { engine: 'llamacpp' } })
        notify([PEER, []])
        notify({ nodeId: SELF, engines: [] })
        supervisor['brokerReady'] = false
        notify({ nodeId: PEER, engines: [] })

        expect(mocks.applyRemoteEngineFacts).not.toHaveBeenCalled()
    })

    it('drops a read that started before a newer push', async () => {
        let answer: (value: Awaited<ReturnType<typeof supervisor.callProcess>>) => void = () => {}
        vi.spyOn(supervisor, 'callProcess').mockImplementation((_name, method) => {
            mocks.calls.push(method)
            return new Promise(resolve => {
                answer = resolve
            })
        })
        const read = supervisor.refreshRemoteEngineStatus(PEER)
        const pushed = [{ engine: 'llamacpp', installed: true, running: true, healthy: true }]
        notify({ nodeId: PEER, engines: pushed })
        answer({ engines: [{ engine: 'llamacpp', installed: true, running: false }] })
        await read

        expect(mocks.applyRemoteEngineFacts).toHaveBeenCalledTimes(1)
        expect(mocks.applyRemoteEngineFacts).toHaveBeenCalledWith(PEER, { engines: pushed })
    })

    it('reads remote engine status on broker ready but never on a timer', async () => {
        const userData = process.env.PAIR_USER_DATA
        if (!userData) throw new Error('Missing isolated user data')
        initPlatform({
            getUserData: () => userData,
            getTemp: () => os.tmpdir(),
            getResourcesPath: () => userData
        })
        await supervisor['onBrokerReady']()
        const readyReads = mocks.calls.filter(m => m === 'engine:remote-get-installed').length
        expect(readyReads).toBeGreaterThan(0)

        await vi.advanceTimersByTimeAsync(60_000)

        expect(mocks.calls.filter(m => m === 'engine:remote-get-installed').length).toBe(readyReads)
    })
})
