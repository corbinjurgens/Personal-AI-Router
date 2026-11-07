// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ServiceStatus } from '@/shared/types/ipc-channels'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import type { IEngineApi } from '@/ui/api/engine-api'
import type { EngineType } from '@/shared/types/engines'

/**
 * Renderer store guard: with the modular service stopped no engine
 * command can reach the broker, so the store must not enter (and must drop) the
 * optimistic "working" state that would otherwise spin until the safety-net
 * timeout. Runs in the `node` unit project with a stubbed `window`.
 */

let statusCb: ((status: ServiceStatus) => void) | null = null

function makeFakeWindow() {
    const noopUnsub = (): void => {}
    return {
        pairApi: {
            engines: {
                onStateChanged: vi.fn<IEngineApi['onStateChanged']>(() => noopUnsub),
                onProgress: vi.fn<IEngineApi['onProgress']>(() => noopUnsub),
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
            .begin({ command: 'install', engineType: 'llama-cpp', nodeId: 'node-1' })

        expect(
            usePendingActionsStore.getState().getLifecyclePending('node-1', 'llama-cpp')
        ).toBeUndefined()
    })

    it('drops in-flight entries the instant the connector reports disconnected', () => {
        pushStatus('connected')

        usePendingActionsStore
            .getState()
            .begin({ command: 'install', engineType: 'llama-cpp', nodeId: 'node-1' })
        expect(usePendingActionsStore.getState().getLifecyclePending('node-1', 'llama-cpp')).toBe(
            'install'
        )

        pushStatus('disconnected')
        expect(
            usePendingActionsStore.getState().getLifecyclePending('node-1', 'llama-cpp')
        ).toBeUndefined()
    })
})

describe('profile deployment attempt tracking', () => {
    const target = {
        nodeId: 'node-1',
        engineType: 'llama-cpp',
        profileKey: 'coder-local'
    } satisfies {
        nodeId: string
        engineType: EngineType
        profileKey: string
    }

    beforeEach(async () => {
        vi.useFakeTimers()
        statusCb = null
        vi.stubGlobal('window', makeFakeWindow())
        usePendingActionsStore.getState().initialize()
        await Promise.resolve()
        pushStatus('connected')
    })

    afterEach(() => {
        const store = usePendingActionsStore.getState()
        store.cleanup()
        for (const id of usePendingActionsStore.getState().deployments.keys()) store.clearDeploy(id)
        vi.useRealTimers()
        vi.unstubAllGlobals()
    })

    it('keeps stages, copy correlation and outcomes independent across targets and sibling attempts', () => {
        const store = usePendingActionsStore.getState()
        store.beginDeploy('first', target)
        store.beginDeploy('sibling', { ...target, profileKey: 'coder-api' })
        store.beginDeploy('remote', { ...target, nodeId: 'node-2' })
        expect(usePendingActionsStore.getState().deployments.get('first')).toEqual({
            ...target,
            stage: 'review'
        })
        store.updateDeploy('first', { stage: 'copy', copyOpId: 'copy-1', modelKey: 'artifact-1' })
        store.updateDeploy('remote', {
            stage: 'apply',
            disposition: 'refused',
            message: 'Policy changed'
        })
        expect(usePendingActionsStore.getState().deployments.get('first')).toEqual({
            ...target,
            stage: 'copy',
            copyOpId: 'copy-1',
            modelKey: 'artifact-1'
        })
        expect(usePendingActionsStore.getState().deployments.get('sibling')).toEqual({
            ...target,
            profileKey: 'coder-api',
            stage: 'review'
        })
        expect(usePendingActionsStore.getState().deployments.get('remote')).toEqual({
            ...target,
            nodeId: 'node-2',
            stage: 'apply',
            disposition: 'refused',
            message: 'Policy changed'
        })
    })

    it('ignores stale updates after clearing an attempt without touching its replacement', () => {
        const store = usePendingActionsStore.getState()
        store.beginDeploy('old', target)
        store.clearDeploy('old')
        store.beginDeploy('new', target)
        const before = usePendingActionsStore.getState().deployments
        store.updateDeploy('old', { stage: 'apply', disposition: 'applied' })
        store.updateDeploy('absent', { disposition: 'cancelled' })
        store.clearDeploy('old')
        expect(usePendingActionsStore.getState().deployments).toBe(before)
        expect(Array.from(before.entries())).toEqual([['new', { ...target, stage: 'review' }]])
    })

    it.each(['copy', 'review', 'apply'] satisfies Array<'copy' | 'review' | 'apply'>)(
        'preserves an unknown %s attempt across disconnect, timeout and reconnect until readback',
        stage => {
            const store = usePendingActionsStore.getState()
            store.beginDeploy('attempt', target)
            store.updateDeploy('attempt', { stage, copyOpId: 'copy-1', modelKey: 'artifact-1' })
            pushStatus('disconnected')
            vi.advanceTimersByTime(180_000)
            pushStatus('connected')
            expect(usePendingActionsStore.getState().deployments.get('attempt')).toEqual({
                ...target,
                stage,
                copyOpId: 'copy-1',
                modelKey: 'artifact-1',
                disposition: 'unknown'
            })
            expect(window.windowApi.service.getStatus).toHaveBeenCalledTimes(1)
            store.updateDeploy('attempt', { stage: 'apply', disposition: 'applied' })
            expect(usePendingActionsStore.getState().deployments.get('attempt')?.disposition).toBe(
                'applied'
            )
        }
    )

    it.each(['copy', 'review', 'apply'] satisfies Array<'copy' | 'review' | 'apply'>)(
        'keeps an unresolved %s row observable through cleanup and reinitialization',
        async stage => {
            const store = usePendingActionsStore.getState()
            store.beginDeploy('attempt', target)
            store.updateDeploy('attempt', { stage, copyOpId: 'copy-1' })
            store.cleanup()
            expect(usePendingActionsStore.getState().deployments.get('attempt')).toEqual({
                ...target,
                stage,
                copyOpId: 'copy-1',
                disposition: 'unknown'
            })
            store.initialize()
            await Promise.resolve()
            expect(usePendingActionsStore.getState().deployments.get('attempt')?.disposition).toBe(
                'unknown'
            )
        }
    )

    it('marks active rows unknown when the initial connector snapshot is disconnected', async () => {
        const store = usePendingActionsStore.getState()
        store.cleanup()
        const fakeWindow = makeFakeWindow()
        fakeWindow.windowApi.service.getStatus.mockResolvedValue({
            connectorStatus: 'disconnected',
            weSpawned: true
        })
        vi.stubGlobal('window', fakeWindow)
        store.beginDeploy('attempt', target)
        store.updateDeploy('attempt', { stage: 'copy', copyOpId: 'copy-1' })
        store.initialize()
        await Promise.resolve()
        expect(usePendingActionsStore.getState().deployments.get('attempt')).toEqual({
            ...target,
            stage: 'copy',
            copyOpId: 'copy-1',
            disposition: 'unknown'
        })
    })

    it('retains applied, refused, cancelled and unknown truth through disconnect and cleanup', () => {
        const store = usePendingActionsStore.getState()
        store.beginDeploy('success', target)
        store.updateDeploy('success', { stage: 'apply', disposition: 'applied' })
        store.beginDeploy('refusal', target)
        store.updateDeploy('refusal', {
            stage: 'review',
            disposition: 'refused',
            message: 'Locked'
        })
        store.beginDeploy('cancel', target)
        store.updateDeploy('cancel', {
            stage: 'copy',
            disposition: 'cancelled',
            copyOpId: 'copy-2'
        })
        store.beginDeploy('unresolved', target)
        store.updateDeploy('unresolved', {
            stage: 'apply',
            disposition: 'unknown',
            message: 'Reply lost'
        })
        pushStatus('disconnected')
        store.cleanup()
        expect(Array.from(usePendingActionsStore.getState().deployments.entries())).toEqual([
            ['success', { ...target, stage: 'apply', disposition: 'applied' }],
            ['refusal', { ...target, stage: 'review', disposition: 'refused', message: 'Locked' }],
            ['cancel', { ...target, stage: 'copy', disposition: 'cancelled', copyOpId: 'copy-2' }],
            [
                'unresolved',
                { ...target, stage: 'apply', disposition: 'unknown', message: 'Reply lost' }
            ]
        ])
    })

    it('does not expire active deploy rows or let settings cover settle them', () => {
        const store = usePendingActionsStore.getState()
        store.beginDeploy('attempt', target)
        store.updateDeploy('attempt', { stage: 'apply' })
        store.beginSettings(target.nodeId, target.engineType, 'settings-1')
        expect(store.isSettingsPending(target.nodeId, target.engineType)).toBe(true)
        store.endSettings(target.nodeId, target.engineType, 'stale-settings')
        expect(store.isSettingsPending(target.nodeId, target.engineType)).toBe(true)
        store.endSettings(target.nodeId, target.engineType, 'settings-1')
        expect(store.isSettingsPending(target.nodeId, target.engineType)).toBe(false)
        vi.advanceTimersByTime(180_000)
        expect(usePendingActionsStore.getState().deployments.get('attempt')).toEqual({
            ...target,
            stage: 'apply'
        })
    })
})
