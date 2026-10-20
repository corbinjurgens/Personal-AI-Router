// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ServiceStatus } from '@/shared/types/ipc-channels'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import type { IEngineApi } from '@/ui/api/engine-api'

/**
 * Renderer store guard: with the modular service stopped no engine
 * command can reach the broker, so the store must not enter (and must drop) the
 * optimistic "working" state that would otherwise spin until the safety-net
 * timeout. Runs in the `node` unit project with a stubbed `window`.
 */

let statusCb: ((status: ServiceStatus) => void) | null = null
let stateCb: Parameters<IEngineApi['onStateChanged']>[0]
let progressCb: Parameters<IEngineApi['onProgress']>[0]

function makeFakeWindow() {
    const noopUnsub = (): void => {}
    return {
        pairApi: {
            engines: {
                onStateChanged: (cb: typeof stateCb) => {
                    stateCb = cb
                    return noopUnsub
                },
                onProgress: (cb: typeof progressCb) => {
                    progressCb = cb
                    return noopUnsub
                },
                onSettingsChanged: () => noopUnsub
            },
            errors: {
                onUpdate: () => noopUnsub
            }
        },
        windowApi: {
            service: {
                getStatus: vi
                    .fn()
                    .mockResolvedValue({ connectorStatus: 'connected', weSpawned: true }),
                onStatusChanged: (cb: (status: ServiceStatus) => void) => {
                    statusCb = cb
                    return noopUnsub
                }
            }
        }
    }
}

function pushStatus(connectorStatus: ServiceStatus['connectorStatus']): void {
    statusCb?.({ connectorStatus, weSpawned: true })
}

describe('pending-actions store honours the modular connector status', () => {
    beforeEach(() => {
        statusCb = null
        vi.stubGlobal('window', makeFakeWindow())
        usePendingActionsStore.getState().initialize()
    })

    afterEach(() => {
        usePendingActionsStore.getState().cleanup()
        vi.unstubAllGlobals()
    })

    it('records no optimistic entry while the service is stopped', () => {
        pushStatus('disconnected')

        usePendingActionsStore
            .getState()
            .begin({ command: 'toggle', engineType: 'ollama', nodeId: 'node-1' })

        expect(
            usePendingActionsStore.getState().getLifecyclePending('node-1', 'ollama')
        ).toBeUndefined()
    })

    it('drops in-flight entries the instant the connector reports disconnected', () => {
        pushStatus('connected')

        usePendingActionsStore
            .getState()
            .begin({ command: 'toggle', engineType: 'ollama', nodeId: 'node-1' })
        expect(usePendingActionsStore.getState().getLifecyclePending('node-1', 'ollama')).toBe(
            'toggle'
        )

        pushStatus('disconnected')
        expect(
            usePendingActionsStore.getState().getLifecyclePending('node-1', 'ollama')
        ).toBeUndefined()
    })

    it('switch send gap reserves both engines and rejects concurrent submission', () => {
        pushStatus('connected')
        const store = usePendingActionsStore.getState()
        expect(store.beginSwitch('node-1', 'switch-a', 'ollama', 'llamacpp')).toBe(true)
        expect(store.beginSwitch('node-1', 'switch-b', 'llamacpp', 'ollama')).toBe(false)
        expect(store.isSwitchPending('node-1', 'ollama')).toBe(true)
        expect(store.isSwitchPending('node-1', 'llamacpp')).toBe(true)
        expect(store.isSwitchPending('node-1', 'other')).toBe(false)
        store.endSwitch('node-1', 'wrong-request')
        expect(store.isSwitchPending('node-1', 'ollama')).toBe(true)
        store.endSwitch('node-1', 'switch-a')
        expect(store.isSwitchPending('node-1', 'ollama')).toBe(false)
        expect(store.isSwitchPending('node-1', 'llamacpp')).toBe(false)
    })

    it('switch correlation does not expire with ordinary action safety nets', () => {
        vi.useFakeTimers()
        try {
            pushStatus('connected')
            const store = usePendingActionsStore.getState()
            expect(store.beginSwitch('node-1', 'switch-a', 'ollama', 'llamacpp')).toBe(true)
            vi.advanceTimersByTime(120_000)
            expect(store.isSwitchPending('node-1', 'ollama')).toBe(true)
        } finally {
            vi.useRealTimers()
        }
    })
    it('ordinary status and progress pushes preserve switch correlation', () => {
        pushStatus('connected')
        const store = usePendingActionsStore.getState()
        expect(store.beginSwitch('node-1', 'switch-a', 'ollama', 'llamacpp')).toBe(true)
        stateCb({
            nodeId: 'node-1',
            engineType: 'ollama',
            status: {
                nodeId: 'node-1',
                engineType: 'ollama',
                processStatus: 'stopped',
                enginePort: null,
                proxyPort: null
            }
        })
        progressCb({
            nodeId: 'node-1',
            nodeName: 'Node 1',
            engineType: 'ollama',
            operation: 'pull',
            model: 'fixture',
            status: 'completed',
            percent: 100
        })
        expect(usePendingActionsStore.getState().switches.get('node-1')).toEqual({
            requestId: 'switch-a',
            source: 'ollama',
            target: 'llamacpp'
        })
    })
})
