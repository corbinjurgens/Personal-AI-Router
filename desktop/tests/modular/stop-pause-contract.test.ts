// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
    parseAvailabilityChange,
    parseAvailabilityResult,
    parsePolicyDocument
} from '@/electron/service-bridge/node-policy'
import {
    availabilityMenuItem,
    NodeAvailabilityState
} from '@/electron/service-bridge/node-availability'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import type { NodeAvailabilityRequest } from '@/shared/types/node-policy'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'
import { destroyTray, initTray } from '@/electron/tray'

interface TrayMenuItem {
    label?: string
    enabled?: boolean
    click?: () => void
}

const trayMocks = vi.hoisted(() => {
    let menu: TrayMenuItem[] = []
    class FakeTray {
        setToolTip = vi.fn<() => void>()
        setContextMenu = vi.fn<() => void>()
        on(): this {
            return this
        }
        destroy = vi.fn<() => void>()
    }
    return {
        FakeTray,
        createTrayWindow: vi.fn<() => void>(),
        get menu() {
            return menu
        },
        buildFromTemplate: (items: TrayMenuItem[]) => {
            menu = items
            return { popup: vi.fn<() => void>() }
        }
    }
})

vi.mock('electron', () => {
    const image = { resize: () => image }
    return {
        app: {
            isPackaged: false,
            getAppPath: () => process.cwd(),
            whenReady: () => Promise.resolve()
        },
        Menu: { buildFromTemplate: trayMocks.buildFromTemplate },
        nativeImage: { createFromPath: () => image },
        screen: {
            on: vi.fn<(event: string, listener: () => void) => void>(),
            removeListener: vi.fn<(event: string, listener: () => void) => void>()
        },
        Tray: trayMocks.FakeTray
    }
})
vi.mock('@/electron/window', () => ({
    createOverviewWindow: vi.fn<() => void>(),
    createTrayWindow: trayMocks.createTrayWindow,
    getTrayWindow: () => null
}))
vi.mock('@/electron/config/ui-config', () => ({
    isFirstRun: () => false,
    getTrayMode: () => 'menu',
    setTrayMode: vi.fn<(mode: string) => void>()
}))
vi.mock('@/electron/connector', () => ({ destroyConnector: vi.fn<() => Promise<void>>() }))
vi.mock('@/shared/utils/platform', () => ({ currentPlatform: () => 'win32' }))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn<() => void>(),
        warn: vi.fn<() => void>(),
        error: vi.fn<() => void>(),
        verbose: vi.fn<() => void>()
    })
}))

describe('stop/pause availability contract', () => {
    it.each(['local', 'peer'])(
        'preserves actual %s provenance in results, pushes and readback',
        pausedBy => {
            const result = { availability: 'paused', pausedBy }
            expect(parseAvailabilityResult(result)).toEqual(result)
            expect(parseAvailabilityChange({ nodeId: 'target', ...result, active: 0 })).toEqual({
                nodeId: 'target',
                ...result
            })
            expect(
                parsePolicyDocument({
                    policy: '{"revision":"2"}',
                    ...result,
                    activeRevision: '2',
                    activation: 'active'
                })
            ).toMatchObject({ ...result, activation: { state: 'active', activeRevision: '2' } })
        }
    )

    it('keeps missing provenance unknown and clears it while available', () => {
        expect(parseAvailabilityResult({ availability: 'paused' }).pausedBy).toBeUndefined()
        expect(
            parseAvailabilityResult({ availability: 'draining', pausedBy: '' }).pausedBy
        ).toBeUndefined()
        expect(parseAvailabilityResult({ availability: 'available', pausedBy: 'peer' })).toEqual({
            availability: 'available'
        })
        expect(
            parseAvailabilityChange({
                nodeId: 'target',
                availability: 'available',
                pausedBy: 'local'
            })
        ).toEqual({ nodeId: 'target', availability: 'available' })
    })

    it('rejects fabricated provenance', () => {
        expect(() =>
            parseAvailabilityResult({ availability: 'paused', pausedBy: 'some-peer-id' })
        ).toThrow('invalid pause provenance')
        expect(
            parseAvailabilityChange({ nodeId: 'target', availability: 'paused', pausedBy: null })
        ).toBeNull()
    })

    it('retains self provenance and ignores peer pushes in tray projection', () => {
        const state = new NodeAvailabilityState(() => 'self')
        const listener = vi.fn<() => void>()
        state.onChange(listener)
        state.applyResult({ availability: 'draining', pausedBy: 'local' })
        expect(state.pausedBy).toBe('local')
        state.applyNotification({ nodeId: 'peer', availability: 'paused', pausedBy: 'peer' })
        expect(state.availability).toBe('draining')
        expect(state.pausedBy).toBe('local')
        state.applyNotification({ nodeId: 'self', availability: 'draining', pausedBy: 'peer' })
        expect(state.pausedBy).toBe('peer')
        expect(listener).toHaveBeenCalledTimes(2)
        state.applyNotification({ nodeId: 'self', availability: 'available', pausedBy: 'peer' })
        expect(state.pausedBy).toBeUndefined()
        state.reset()
        expect(state.pausedBy).toBeUndefined()
    })
})

describe('single typed availability request', () => {
    it('passes the request object unchanged through pairApi', async () => {
        const invoke = vi.fn<ServiceTransport['invoke']>()
        const transport: ServiceTransport = {
            connected: true,
            invoke,
            subscribePush: vi.fn<ServiceTransport['subscribePush']>(() => () => {}),
            onConnect: vi.fn<ServiceTransport['onConnect']>(() => () => {}),
            onDisconnect: vi.fn<ServiceTransport['onDisconnect']>(() => () => {}),
            onAuthFailure: vi.fn<ServiceTransport['onAuthFailure']>(() => () => {}),
            destroy: vi.fn<ServiceTransport['destroy']>()
        }
        const api = createPairApi(transport)
        const requests: NodeAvailabilityRequest[] = [
            { state: 'paused' },
            { state: 'paused', nodeId: 'peer', pauseMode: 'configured' },
            { state: 'paused', pauseMode: 'immediate' },
            { state: 'available' }
        ]
        for (const request of requests) {
            // The unavailable transport reply cannot trigger management refresh.
            invoke.mockRejectedValueOnce(new Error('refused'))
            await expect(api.nodePolicy.setAvailability(request)).rejects.toThrow('refused')
            expect(invoke).toHaveBeenLastCalledWith('node:set-availability', request)
            expect(invoke.mock.lastCall?.[1]).toBe(request)
        }
    })

    it('carries mode through supervisor and leaves tray state alone for a remote result', async () => {
        const supervisor = getModularSupervisor()
        const call = vi.spyOn(supervisor, 'callProcess')
        call.mockResolvedValue({ availability: 'paused', pausedBy: 'peer' })
        const remote: NodeAvailabilityRequest = {
            nodeId: 'peer',
            state: 'paused',
            pauseMode: 'configured'
        }
        expect(await supervisor.setAvailability(remote)).toEqual({
            availability: 'paused',
            pausedBy: 'peer'
        })
        expect(call).toHaveBeenLastCalledWith(
            'broker',
            'node:set-availability',
            remote,
            expect.any(Number)
        )
        expect(supervisor.getAvailabilityMenuItem().enabled).toBe(false)

        call.mockResolvedValueOnce({ availability: 'paused', pausedBy: 'local' })
        await expect(
            supervisor.setAvailability({ state: 'paused', pauseMode: 'immediate' })
        ).resolves.toEqual({ availability: 'paused', pausedBy: 'local' })
        expect(call).toHaveBeenNthCalledWith(
            2,
            'broker',
            'node:set-availability',
            { state: 'paused', pauseMode: 'immediate' },
            expect.any(Number)
        )
        expect(call).toHaveBeenLastCalledWith('broker', 'policy:get', {})
        expect(call).toHaveBeenCalledTimes(3)
        expect(supervisor.getAvailabilityMenuItem()).toMatchObject({
            request: 'available',
            enabled: true
        })
        call.mockRestore()
    })
})

describe('self-only tray immediate stop', () => {
    const supervisor = getModularSupervisor()
    let item = availabilityMenuItem('available')
    let refreshMenu: (() => void) | undefined

    function menuItem(label: string): TrayMenuItem {
        const found = trayMocks.menu.find(entry => entry.label === label)
        if (!found) throw new Error(`Missing tray item: ${label}`)
        return found
    }

    function report(availability: Parameters<typeof availabilityMenuItem>[0]): void {
        item = availabilityMenuItem(availability)
        if (!refreshMenu) throw new Error('Missing tray availability subscription')
        refreshMenu()
    }

    beforeEach(async () => {
        vi.useFakeTimers()
        item = availabilityMenuItem('available')
        trayMocks.createTrayWindow.mockClear()
        vi.spyOn(supervisor, 'getAvailabilityMenuItem').mockImplementation(() => item)
        vi.spyOn(supervisor, 'onAvailabilityChanged').mockImplementation(listener => {
            refreshMenu = listener
            return () => {
                refreshMenu = undefined
            }
        })
        vi.spyOn(supervisor, 'setAvailability').mockResolvedValue(undefined)
        vi.spyOn(supervisor, 'refreshAvailability').mockResolvedValue()
        await initTray()
    })

    afterEach(() => {
        destroyTray()
        vi.restoreAllMocks()
        vi.useRealTimers()
    })

    it('sends the exact local immediate payload without opening a window', async () => {
        expect(menuItem('Stop local models now').enabled).toBe(true)
        menuItem('Stop local models now').click?.()
        await vi.advanceTimersByTimeAsync(0)
        expect(supervisor.setAvailability).toHaveBeenCalledExactlyOnceWith({
            state: 'paused',
            pauseMode: 'immediate'
        })
        expect(trayMocks.createTrayWindow).not.toHaveBeenCalled()
    })

    it('keeps configured Pause and local Resume payloads unchanged', async () => {
        menuItem('Pause inference on this PC').click?.()
        await vi.advanceTimersByTimeAsync(0)
        report('paused')
        menuItem('Resume inference on this PC').click?.()
        await vi.advanceTimersByTimeAsync(0)
        expect(vi.mocked(supervisor.setAvailability).mock.calls.map(call => call[0])).toEqual([
            { state: 'paused', pauseMode: 'configured' },
            { state: 'available' }
        ])
    })

    it('escalates a configured drain once and ignores its late failure', async () => {
        let failConfigured: ((error: Error) => void) | undefined
        let finishImmediate: (() => void) | undefined
        vi.mocked(supervisor.setAvailability)
            .mockImplementationOnce(
                () =>
                    new Promise((_resolve, reject) => {
                        failConfigured = reject
                    })
            )
            .mockImplementationOnce(
                () =>
                    new Promise(resolve => {
                        finishImmediate = () => resolve(undefined)
                    })
            )
        menuItem('Pause inference on this PC').click?.()
        expect(menuItem('Stop local models now').enabled).toBe(false)
        menuItem('Stop local models now').click?.()
        expect(supervisor.setAvailability).toHaveBeenCalledTimes(1)
        report('draining')
        expect(menuItem('Stop local models now').enabled).toBe(true)
        menuItem('Stop local models now').click?.()
        expect(menuItem('Stop local models now').enabled).toBe(false)
        expect(menuItem('Resume inference on this PC (pausing)').enabled).toBe(true)
        menuItem('Stop local models now').click?.()
        expect(vi.mocked(supervisor.setAvailability).mock.calls.map(call => call[0])).toEqual([
            { state: 'paused', pauseMode: 'configured' },
            { state: 'paused', pauseMode: 'immediate' }
        ])
        if (!failConfigured || !finishImmediate) throw new Error('Missing tray pause callbacks')
        failConfigured(new Error('Superseded configured pause'))
        await vi.advanceTimersByTimeAsync(0)
        expect(
            trayMocks.menu.some(entry => entry.label?.includes('Superseded configured pause'))
        ).toBe(false)
        expect(menuItem('Stop local models now').enabled).toBe(false)
        finishImmediate()
        await vi.advanceTimersByTimeAsync(0)
        expect(menuItem('Stop local models now').enabled).toBe(true)
    })

    it('retains incomplete-stop errors and offers local retry and Resume', async () => {
        vi.mocked(supervisor.setAvailability).mockRejectedValueOnce(
            new Error('Owned engine still running')
        )
        menuItem('Stop local models now').click?.()
        report('draining')
        await vi.advanceTimersByTimeAsync(0)
        expect(
            menuItem('Availability request failed: Error: Owned engine still running').enabled
        ).toBe(false)
        expect(menuItem('Stop local models now').enabled).toBe(true)
        expect(menuItem('Retry pause on this PC').enabled).toBe(true)
        expect(menuItem('Resume inference on this PC (pausing)').enabled).toBe(true)
        menuItem('Stop local models now').click?.()
        await vi.advanceTimersByTimeAsync(0)
        expect(supervisor.setAvailability).toHaveBeenNthCalledWith(2, {
            state: 'paused',
            pauseMode: 'immediate'
        })
        report('paused')
        menuItem('Resume inference on this PC').click?.()
        await vi.advanceTimersByTimeAsync(0)
        expect(supervisor.setAvailability).toHaveBeenNthCalledWith(3, { state: 'available' })
    })

    it('keeps local Resume usable during immediate stop and ignores its superseded result', async () => {
        report('draining')
        let failImmediate: ((error: Error) => void) | undefined
        vi.mocked(supervisor.setAvailability).mockImplementationOnce(
            () =>
                new Promise((_resolve, reject) => {
                    failImmediate = reject
                })
        )
        menuItem('Stop local models now').click?.()
        menuItem('Resume inference on this PC (pausing)').click?.()
        await vi.advanceTimersByTimeAsync(0)
        report('available')
        if (!failImmediate) throw new Error('Missing immediate stop callback')
        failImmediate(new Error('Superseded immediate stop'))
        await vi.advanceTimersByTimeAsync(0)
        expect(vi.mocked(supervisor.setAvailability).mock.calls.map(call => call[0])).toEqual([
            { state: 'paused', pauseMode: 'immediate' },
            { state: 'available' }
        ])
        expect(
            trayMocks.menu.some(entry => entry.label?.includes('Superseded immediate stop'))
        ).toBe(false)
        expect(menuItem('Pause inference on this PC').enabled).toBe(true)
    })

    it('does not send immediate stop while local Resume is pending', async () => {
        report('draining')
        let finishResume: (() => void) | undefined
        vi.mocked(supervisor.setAvailability).mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    finishResume = () => resolve(undefined)
                })
        )
        menuItem('Resume inference on this PC (pausing)').click?.()
        expect(menuItem('Stop local models now').enabled).toBe(false)
        menuItem('Stop local models now').click?.()
        expect(supervisor.setAvailability).toHaveBeenCalledExactlyOnceWith({ state: 'available' })
        if (!finishResume) throw new Error('Missing Resume callback')
        finishResume()
        await vi.advanceTimersByTimeAsync(0)
    })

    it('requires self readback when availability is unknown and retains the stop error', async () => {
        vi.mocked(supervisor.setAvailability).mockRejectedValueOnce(new Error('Stop reply lost'))
        menuItem('Stop local models now').click?.()
        report(null)
        await vi.advanceTimersByTimeAsync(0)
        expect(menuItem('Stop local models now').enabled).toBe(false)
        menuItem('Stop local models now').click?.()
        expect(supervisor.setAvailability).toHaveBeenCalledTimes(1)
        expect(menuItem('Availability request failed: Error: Stop reply lost').enabled).toBe(false)
        vi.mocked(supervisor.refreshAvailability).mockImplementationOnce(async () =>
            report('draining')
        )
        menuItem('Refresh availability on this PC').click?.()
        await vi.advanceTimersByTimeAsync(0)
        expect(supervisor.refreshAvailability).toHaveBeenCalledExactlyOnceWith()
        expect(menuItem('Availability request failed: Error: Stop reply lost').enabled).toBe(false)
        expect(menuItem('Stop local models now').enabled).toBe(true)
        expect(trayMocks.createTrayWindow).not.toHaveBeenCalled()
    })
})
