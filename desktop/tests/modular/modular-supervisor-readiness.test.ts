// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
    JsonRpcSubprocess,
    type JsonRpcNotification
} from '@/electron/service-bridge/json-rpc-subprocess'

const mocks = vi.hoisted(() => ({
    pendingInviteIds: new Set<string>(),
    bridgeState: {
        handleNotification: vi.fn(),
        getSelfId: vi.fn(() => null),
        getProxyPort: vi.fn(() => null),
        getPendingInvites: () => Array.from(mocks.pendingInviteIds),
        addPendingInvite: (invite: { inviteId: string; state: string }) => {
            if (invite.state === 'pending') mocks.pendingInviteIds.add(invite.inviteId)
        }
    },
    emitBridgePush: vi.fn()
}))

vi.mock('electron', () => ({
    app: {
        isPackaged: false,
        getAppPath: () => process.cwd()
    }
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

vi.mock('@/electron/config/ui-config', () => ({
    isFirstRun: () => false
}))

vi.mock('@/electron/service-bridge/broadcaster', () => ({
    emitBridgePush: mocks.emitBridgePush
}))

vi.mock('@/electron/service-bridge/manual-nodes-store', () => ({
    listManualNodeEntries: () => []
}))

vi.mock('@/electron/service-bridge/node-info-poller', () => ({
    startNodeInfoPoller: vi.fn(),
    stopNodeInfoPoller: vi.fn()
}))

vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => mocks.bridgeState,
    isUpstreamUnreachableError: () => false,
    parseServiceErrors: () => [],
    parseWorkloadsInitial: () => [],
    PROXY_ENGINES: ['ollama', 'lm-studio'],
    PROXY_NODE_SOURCES: ['ollama-proxy', 'lmstudio-proxy']
}))

import {
    getModularSupervisor,
    ModularStartupTimeoutError
} from '@/electron/service-bridge/modular-supervisor'
import { createOverviewWindow } from '@/electron/window'

interface ReadinessHarness {
    readonly ready: boolean
    processes: Map<string, JsonRpcSubprocess>
    isReady: boolean
    brokerReady: boolean
    brokerHydrationDone: boolean
    readinessReported: boolean
    nextReadinessWaiterId: number
    readinessWaiters: Map<number, { timeout: ReturnType<typeof setTimeout> }>
    onReady?: () => void
    onBrokerReady: () => Promise<void>
    setOnReady: (callback: () => void) => void
    waitUntilReady: (timeoutMs: number) => Promise<void>
    handleNotification: (notification: JsonRpcNotification) => void
    attachChildHandlers: (child: JsonRpcSubprocess) => void
}

const supervisor = getModularSupervisor() as unknown as ReadinessHarness

function notify(method: string, params?: JsonRpcNotification['params']): void {
    supervisor.handleNotification({ source: 'broker', method, params })
}

describe('modular supervisor readiness', () => {
    beforeEach(() => {
        vi.useFakeTimers()
        for (const waiter of supervisor.readinessWaiters.values()) {
            clearTimeout(waiter.timeout)
        }
        supervisor.readinessWaiters.clear()
        supervisor.nextReadinessWaiterId = 0
        supervisor.readinessReported = false
        supervisor.isReady = true
        supervisor.brokerReady = false
        supervisor.brokerHydrationDone = false
        supervisor.processes.clear()
        supervisor.onReady = undefined
        supervisor.onBrokerReady = vi.fn().mockResolvedValue(undefined)
    })

    afterEach(() => {
        for (const waiter of supervisor.readinessWaiters.values()) {
            clearTimeout(waiter.timeout)
        }
        supervisor.readinessWaiters.clear()
        vi.useRealTimers()
    })

    it('requires app:ready but not optional proxy readiness', async () => {
        const onReady = vi.fn()
        supervisor.setOnReady(onReady)
        const waiting = supervisor.waitUntilReady(1_000)
        let settled = false
        void waiting.then(() => {
            settled = true
        })

        notify('ollama-proxy:ready', { port: 11434 })
        await Promise.resolve()

        expect(supervisor.ready).toBe(false)
        expect(settled).toBe(false)
        expect(onReady).not.toHaveBeenCalled()

        notify('app:ready')

        await expect(waiting).resolves.toBeUndefined()
        expect(supervisor.ready).toBe(true)
        expect(onReady).toHaveBeenCalledOnce()
    })

    it('keeps the service ready and refreshes capability state when the proxy arrives late', async () => {
        const onReady = vi.fn()
        supervisor.setOnReady(onReady)

        notify('app:ready')
        await Promise.resolve()
        mocks.emitBridgePush.mockClear()
        supervisor.brokerHydrationDone = true

        notify('ollama-proxy:error', { message: 'address already in use' })
        notify('ollama-proxy:ready', { port: 11435 })

        expect(supervisor.ready).toBe(true)
        expect(onReady).toHaveBeenCalledOnce()
        expect(mocks.bridgeState.handleNotification).toHaveBeenLastCalledWith({
            source: 'ollama-proxy',
            method: 'ready',
            params: { port: 11435 }
        })
        expect(mocks.emitBridgePush).toHaveBeenCalledWith('state:request-refresh', undefined)
    })

    it('diagnoses a missing app:ready and still recovers from a late notification', async () => {
        const onReady = vi.fn()
        supervisor.setOnReady(onReady)
        const waiting = supervisor.waitUntilReady(15_000)
        const rejection = expect(waiting).rejects.toEqual(new ModularStartupTimeoutError(15_000))

        await vi.advanceTimersByTimeAsync(15_000)
        await rejection

        notify('app:ready')

        expect(supervisor.ready).toBe(true)
        expect(onReady).toHaveBeenCalledOnce()
    })

    it('ignores readiness from a replaced broker generation', () => {
        const oldBroker = new JsonRpcSubprocess('broker', 'old-broker')
        const newBroker = new JsonRpcSubprocess('broker', 'new-broker')
        supervisor.processes.set('broker', oldBroker)
        supervisor.attachChildHandlers(oldBroker)

        oldBroker.emit('exit', { source: 'broker', code: 1 })
        supervisor.processes.set('broker', newBroker)
        supervisor.attachChildHandlers(newBroker)
        supervisor.isReady = true

        oldBroker.emit('notification', { source: 'broker', method: 'app:ready' })

        expect(supervisor.ready).toBe(false)

        newBroker.emit('notification', { source: 'broker', method: 'app:ready' })

        expect(supervisor.ready).toBe(true)

        oldBroker.emit('exit', { source: 'broker', code: 1 })

        expect(supervisor.ready).toBe(true)
    })
})

// The PIN prompt lives in Overview. The tray popup used to raise Overview for a
// new inbound invite while it sat resident; it now exists only while open, so
// main raises Overview itself.
describe('inbound pairing invites', () => {
    beforeEach(() => {
        mocks.pendingInviteIds.clear()
        vi.mocked(createOverviewWindow).mockClear()
    })

    it('raises Overview for a new pending invite, once', () => {
        notify('cluster:invite-received', { inviteId: 'inv-1', state: 'pending' })
        expect(createOverviewWindow).toHaveBeenCalledOnce()

        notify('cluster:invite-received', { inviteId: 'inv-1', state: 'pending' })
        expect(createOverviewWindow).toHaveBeenCalledOnce()
    })

    it('does not raise Overview for an invite that is no longer pending', () => {
        notify('cluster:invite-received', { inviteId: 'inv-2', state: 'expired' })
        expect(createOverviewWindow).not.toHaveBeenCalled()
    })
})
