// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import type { JsonValue } from '@/electron/service-bridge/json-rpc-client'
import {
    availabilityMenuItem,
    NodeAvailabilityState
} from '@/electron/service-bridge/node-availability'

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

    it('shows a disabled Pausing... while draining', () => {
        expect(availabilityMenuItem('draining')).toEqual({
            label: 'Pausing…',
            enabled: false,
            request: null
        })
    })

    it('shows a disabled Pause until the state is known', () => {
        expect(availabilityMenuItem(null)).toEqual({
            label: 'Pause inference on this PC',
            enabled: false,
            request: null
        })
    })
})

describe('NodeAvailabilityState', () => {
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
        expect(s.menuItem().label).toBe('Pausing…')

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
