// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import type { JsonValue } from '@/electron/service-bridge/json-rpc-client'
import {
    availabilityMenuItem,
    NodeAvailabilityState
} from '@/electron/service-bridge/node-availability'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'

vi.mock('electron', () => ({ app: { isPackaged: false, getAppPath: () => process.cwd() } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<() => void>() }))
vi.mock('@/electron/config/ui-config', () => ({ isFirstRun: () => false }))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn<() => void>(),
        warn: vi.fn<() => void>(),
        error: vi.fn<() => void>(),
        verbose: vi.fn<() => void>()
    })
}))

const SELF = 'node-self'

function state(selfId: string | null = SELF): NodeAvailabilityState {
    return new NodeAvailabilityState(() => selfId)
}

describe('availabilityMenuItem', () => {
    it('offers Pause while available', () => {
        expect(availabilityMenuItem('available')).toEqual({
            label: 'Pause inference on this PC',
            enabled: true,
            request: 'paused'
        })
    })

    it('offers Resume while paused', () => {
        expect(availabilityMenuItem('paused')).toEqual({
            label: 'Resume inference on this PC',
            enabled: true,
            request: 'available'
        })
    })

    it('keeps local Resume and retry available while draining', () => {
        expect(availabilityMenuItem('draining')).toEqual({
            label: 'Resume inference on this PC (pausing)',
            enabled: true,
            request: 'available',
            retryPause: true
        })
    })

    it('shows a disabled Pause until the state is known', () => {
        expect(availabilityMenuItem(null)).toEqual({
            label: 'Pause inference on this PC',
            enabled: false,
            request: null,
            refresh: true
        })
    })
})

describe('NodeAvailabilityState', () => {
    it('invalidates a read on equal self pushes and reset, while peer pushes leave its generation alone', () => {
        const s = state()
        s.applyResult({ availability: 'draining' })
        const before = s.observation
        s.applyNotification({ nodeId: 'peer', availability: 'paused' })
        expect(s.observation).toBe(before)
        s.applyNotification({ nodeId: SELF, availability: 'draining' })
        expect(s.observation).toBeGreaterThan(before)
        const pushed = s.observation
        s.reset()
        expect(s.observation).toBeGreaterThan(pushed)
    })
    it('starts unknown', () => {
        expect(state().availability).toBeNull()
        expect(state().menuItem().enabled).toBe(false)
    })

    it('reads the policy:get result at hydrate time', () => {
        const s = state()
        s.applyResult({ policy: { pause: { onActive: 'drain' } }, availability: 'paused' })

        expect(s.availability).toBe('paused')
        expect(s.menuItem().label).toBe('Resume inference on this PC')
    })

    it('ignores a result without a recognized availability', () => {
        const s = state()
        s.applyResult({ availability: 'paused' })
        const unusable: (JsonValue | undefined)[] = [
            undefined,
            null,
            'paused',
            [],
            {},
            { availability: 'asleep' }
        ]
        for (const result of unusable) {
            s.applyResult(result)
        }

        expect(s.availability).toBe('paused')
    })

    it('follows availability-changed pushes for this node', () => {
        const s = state()
        s.applyNotification({ nodeId: SELF, availability: 'draining', active: 3 })
        expect(s.menuItem().label).toBe('Resume inference on this PC (pausing)')

        s.applyNotification({ nodeId: SELF, availability: 'paused', active: 0 })
        expect(s.menuItem().label).toBe('Resume inference on this PC')

        s.applyNotification({ nodeId: SELF, availability: 'available', active: 0 })
        expect(s.menuItem().label).toBe('Pause inference on this PC')
        expect(s.menuItem().enabled).toBe(true)
    })

    it('ignores pushes about other nodes', () => {
        const s = state()
        s.applyResult({ availability: 'available' })

        s.applyNotification({ nodeId: 'node-other', availability: 'paused', active: 0 })

        expect(s.availability).toBe('available')
    })

    it('ignores pushes while this node is not known yet', () => {
        const s = state(null)

        s.applyNotification({ nodeId: 'node-other', availability: 'paused', active: 0 })
        s.applyNotification({ availability: 'paused', active: 0 })

        expect(s.availability).toBeNull()
    })

    it('ignores malformed pushes', () => {
        const s = state()
        s.applyResult({ availability: 'available' })

        const malformed: (JsonValue | undefined)[] = [
            undefined,
            null,
            'paused',
            [],
            { nodeId: SELF }
        ]
        for (const params of malformed) {
            s.applyNotification(params)
        }
        s.applyNotification({ nodeId: SELF, availability: 'asleep' })

        expect(s.availability).toBe('available')
    })

    it('tells listeners only about changes, and stops after unsubscribe', () => {
        const s = state()
        const listener = vi.fn()
        const unsubscribe = s.onChange(listener)

        s.applyResult({ availability: 'available' })
        s.applyNotification({ nodeId: SELF, availability: 'available', active: 0 })
        expect(listener).toHaveBeenCalledTimes(1)

        s.applyNotification({ nodeId: SELF, availability: 'draining', active: 1 })
        expect(listener).toHaveBeenCalledTimes(2)

        unsubscribe()
        s.applyNotification({ nodeId: SELF, availability: 'paused', active: 0 })
        expect(listener).toHaveBeenCalledTimes(2)
    })

    it('forgets the state on reset', () => {
        const s = state()
        const listener = vi.fn()
        s.applyResult({ availability: 'paused' })
        s.onChange(listener)

        s.reset()

        expect(s.availability).toBeNull()
        expect(s.menuItem().enabled).toBe(false)
        expect(listener).toHaveBeenCalledOnce()
    })
})

describe('tray authoritative self readback without a renderer', () => {
    it('reports uncertainty instead of accepting a malformed readback as success', async () => {
        const supervisor = getModularSupervisor()
        vi.spyOn(supervisor, 'callProcess')
            .mockResolvedValueOnce({ availability: 'paused' })
            .mockResolvedValueOnce({ availability: 'not-a-state' })
        await expect(supervisor.setAvailability({ state: 'paused' })).rejects.toThrow('unconfirmed')
        expect(supervisor.getAvailabilityMenuItem().enabled).toBe(false)
    })
    it('uses policy readback rather than a successful mutation reply', async () => {
        const supervisor = getModularSupervisor()
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockResolvedValueOnce({ availability: 'paused' })
            .mockResolvedValueOnce({ availability: 'draining', pausedBy: 'local' })
        await supervisor.setAvailability({ state: 'paused', pauseMode: 'configured' })
        expect(call).toHaveBeenNthCalledWith(2, 'broker', 'policy:get', {})
        expect(supervisor.getAvailabilityMenuItem()).toMatchObject({
            request: 'available',
            retryPause: true
        })
    })

    it('preserves a lost reply rejection while reading its installed fence', async () => {
        const supervisor = getModularSupervisor()
        const error = new Error('Lost availability reply')
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockRejectedValueOnce(error)
            .mockResolvedValueOnce({ availability: 'paused', pausedBy: 'peer' })
        await expect(
            supervisor.setAvailability({ state: 'paused', pauseMode: 'configured' })
        ).rejects.toBe(error)
        expect(call).toHaveBeenCalledTimes(2)
        expect(call).toHaveBeenLastCalledWith('broker', 'policy:get', {})
        expect(supervisor.getAvailabilityMenuItem()).toMatchObject({
            request: 'available',
            enabled: true
        })
    })

    it('disables unknown state after failed readback without masking the mutation error', async () => {
        const supervisor = getModularSupervisor()
        const error = new Error('Owned engine did not stop')
        vi.spyOn(supervisor, 'callProcess')
            .mockRejectedValueOnce(error)
            .mockRejectedValueOnce(new Error('Readback disconnected'))
        await expect(supervisor.setAvailability({ state: 'paused' })).rejects.toBe(error)
        expect(supervisor.getAvailabilityMenuItem().enabled).toBe(false)
    })

    it('manually refreshes unknown self availability without resending a mutation', async () => {
        const supervisor = getModularSupervisor()
        const error = new Error('Owned engine did not stop')
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockRejectedValueOnce(error)
            .mockRejectedValueOnce(new Error('Readback disconnected'))
            .mockRejectedValueOnce(new Error('Still disconnected'))
            .mockResolvedValueOnce({ availability: 'draining', pausedBy: 'local' })
        await expect(
            supervisor.setAvailability({ state: 'paused', pauseMode: 'configured' })
        ).rejects.toBe(error)
        expect(supervisor.getAvailabilityMenuItem()).toMatchObject({
            enabled: false,
            refresh: true
        })
        await expect(supervisor.refreshAvailability()).rejects.toThrow('Still disconnected')
        expect(supervisor.getAvailabilityMenuItem()).toMatchObject({
            enabled: false,
            refresh: true
        })
        await supervisor.refreshAvailability()
        expect(call.mock.calls.map(args => args.slice(0, 3))).toEqual([
            ['broker', 'node:set-availability', { state: 'paused', pauseMode: 'configured' }],
            ['broker', 'policy:get', {}],
            ['broker', 'policy:get', {}],
            ['broker', 'policy:get', {}]
        ])
        expect(supervisor.getAvailabilityMenuItem()).toMatchObject({
            request: 'available',
            enabled: true,
            retryPause: true
        })
    })

    it('ignores an old configured-pause completion after a newer local Resume', async () => {
        const supervisor = getModularSupervisor()
        let finish: ((value: JsonValue) => void) | undefined
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockImplementationOnce(
                () =>
                    new Promise(resolve => {
                        finish = resolve
                    })
            )
            .mockResolvedValueOnce({ availability: 'available' })
            .mockResolvedValueOnce({ availability: 'available' })
        const old = supervisor.setAvailability({ state: 'paused', pauseMode: 'configured' })
        await supervisor.setAvailability({ state: 'available' })
        if (!finish) throw new Error('Missing held pause reply')
        finish({ availability: 'paused' })
        await old
        expect(call).toHaveBeenCalledTimes(3)
        expect(supervisor.getAvailabilityMenuItem()).toMatchObject({
            request: 'paused',
            enabled: true
        })
    })
})
