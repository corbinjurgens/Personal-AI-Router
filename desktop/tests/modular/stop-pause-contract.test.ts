// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import {
    parseAvailabilityChange,
    parseAvailabilityResult,
    parsePolicyDocument
} from '@/electron/service-bridge/node-policy'
import { NodeAvailabilityState } from '@/electron/service-bridge/node-availability'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import type { NodeAvailabilityRequest } from '@/shared/types/node-policy'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'

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

        call.mockRejectedValueOnce(
            new Error('Immediate pause is unavailable until safe execution stopping is enabled.')
        )
        await expect(
            supervisor.setAvailability({ state: 'paused', pauseMode: 'immediate' })
        ).rejects.toThrow('unavailable')
        expect(call).toHaveBeenLastCalledWith(
            'broker',
            'node:set-availability',
            { state: 'paused', pauseMode: 'immediate' },
            expect.any(Number)
        )
        expect(supervisor.getAvailabilityMenuItem().enabled).toBe(false)
        call.mockRestore()
    })
})
