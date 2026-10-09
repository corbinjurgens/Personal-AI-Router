// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { EventEmitter } from 'events'
import { MODULAR_NODE_INFO_POLL_INTERVAL_MS } from '@/shared/constants/modular-runtime'
import type { getModularBridgeState } from '@/electron/service-bridge/modular-state'

type BridgeState = ReturnType<typeof getModularBridgeState>

const mocks = vi.hoisted(() => ({
    state: {
        getNodeInfoPollTargets: vi.fn<BridgeState['getNodeInfoPollTargets']>(),
        mergeNodeInfoResponse: vi.fn<BridgeState['mergeNodeInfoResponse']>()
    }
}))

vi.mock('@/electron/service-bridge/modular-state', () => ({
    getModularBridgeState: () => mocks.state
}))

vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        verbose: vi.fn()
    })
}))

import {
    setNodeInfoPollerVisible,
    startNodeInfoPoller,
    stopNodeInfoPoller
} from '@/electron/service-bridge/node-info-poller'
import { trackWindowVisibility } from '@/electron/window-visibility'

const fetchMock = vi.fn<typeof fetch>()

function answer(): Promise<Response> {
    return Promise.resolve(new Response(JSON.stringify({ hostUuid: 'uuid-a' }), { status: 200 }))
}

/** Runs one poll interval and reports how many polls it made. */
async function pollsDuring(ms: number): Promise<number> {
    fetchMock.mockClear()
    await vi.advanceTimersByTimeAsync(ms)
    return fetchMock.mock.calls.length
}

// The poller feeds only what the windows show, so it runs only while the
// service is up (start/stop) and some app window is on screen.
describe('node info poller visibility gating', () => {
    beforeEach(() => {
        vi.useFakeTimers()
        vi.stubGlobal('fetch', fetchMock)
        fetchMock.mockReset()
        fetchMock.mockImplementation(answer)
        mocks.state.getNodeInfoPollTargets.mockReturnValue([
            { id: 'uuid-a', hosts: ['192.0.2.1'], port: 14318 }
        ])
    })

    afterEach(() => {
        stopNodeInfoPoller()
        setNodeInfoPollerVisible(false)
        vi.unstubAllGlobals()
        vi.useRealTimers()
    })

    it('does not poll while started with no window visible', async () => {
        startNodeInfoPoller()
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 3)).toBe(0)
    })

    it('polls at once when a window becomes visible, then on the interval', async () => {
        startNodeInfoPoller()
        setNodeInfoPollerVisible(true)
        expect(fetchMock).toHaveBeenCalledTimes(1)

        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS)).toBe(1)
    })

    it('pauses when every window is hidden and resumes with an immediate poll', async () => {
        startNodeInfoPoller()
        setNodeInfoPollerVisible(true)
        await vi.advanceTimersByTimeAsync(0)

        setNodeInfoPollerVisible(false)
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 5)).toBe(0)

        fetchMock.mockClear()
        setNodeInfoPollerVisible(true)
        expect(fetchMock).toHaveBeenCalledTimes(1)
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS)).toBe(1)
    })

    it('does not poll for a visible window while the service is stopped', async () => {
        setNodeInfoPollerVisible(true)
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 3)).toBe(0)

        startNodeInfoPoller()
        expect(fetchMock).toHaveBeenCalledTimes(1)

        stopNodeInfoPoller()
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 3)).toBe(0)
    })

    it('does not double the interval when visibility is reported twice', async () => {
        startNodeInfoPoller()
        setNodeInfoPollerVisible(true)
        setNodeInfoPollerVisible(true)
        await vi.advanceTimersByTimeAsync(0)

        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS)).toBe(1)
    })
})

/** Stands in for a BrowserWindow: the visibility queries and lifecycle events. */
class FakeWindow extends EventEmitter {
    visible = false
    minimized = false
    destroyed = false

    isDestroyed(): boolean {
        return this.destroyed
    }

    isVisible(): boolean {
        return this.visible
    }

    isMinimized(): boolean {
        return this.minimized
    }

    show(): void {
        this.visible = true
        this.emit('show')
    }

    hide(): void {
        this.visible = false
        this.emit('hide')
    }

    minimize(): void {
        this.minimized = true
        this.emit('minimize')
    }

    restore(): void {
        this.minimized = false
        this.emit('restore')
    }

    close(): void {
        this.destroyed = true
        this.emit('closed')
    }
}

describe('window visibility tracking drives the poller', () => {
    beforeEach(() => {
        vi.useFakeTimers()
        vi.stubGlobal('fetch', fetchMock)
        fetchMock.mockReset()
        fetchMock.mockImplementation(answer)
        mocks.state.getNodeInfoPollTargets.mockReturnValue([
            { id: 'uuid-a', hosts: ['192.0.2.1'], port: 14318 }
        ])
        startNodeInfoPoller()
    })

    afterEach(() => {
        stopNodeInfoPoller()
        vi.unstubAllGlobals()
        vi.useRealTimers()
    })

    it('runs while either window is on screen and stops when none is', async () => {
        const overview = new FakeWindow()
        const tray = new FakeWindow()
        trackWindowVisibility(overview)
        trackWindowVisibility(tray)
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 2)).toBe(0)

        overview.show()
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS)).toBe(1)

        // Overview minimized while the tray popup is open: still polling.
        tray.show()
        overview.minimize()
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS)).toBe(1)

        tray.hide()
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 3)).toBe(0)

        fetchMock.mockClear()
        overview.restore()
        expect(fetchMock).toHaveBeenCalledTimes(1)

        overview.close()
        tray.close()
        expect(await pollsDuring(MODULAR_NODE_INFO_POLL_INTERVAL_MS * 3)).toBe(0)
    })
})
