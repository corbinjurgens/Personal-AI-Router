// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

import type { EngineProcessStatus, EngineProgress } from '@/shared/types/engines'
import { subscribePush } from '@/electron/service-bridge/push-bus'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'

let unsubscribe = (): void => {}

afterEach(() => {
    unsubscribe()
    unsubscribe = () => {}
    getModularBridgeState().clearPendingEngineOp('llama-cpp')
    vi.useRealTimers()
})

describe('llama.cpp install state', () => {
    it.each(['llamacpp', 'llama-cpp'])('clears failed operations for %s', engineId => {
        const state = getModularBridgeState()
        const statuses: EngineProcessStatus[] = []
        unsubscribe = subscribePush(event => {
            if (
                event.channel === 'engines:state-changed' &&
                event.payload.engineType === 'llama-cpp' &&
                event.payload.status
            ) {
                statuses.push(event.payload.status.processStatus)
            }
        })
        state.setSelfId('local-node')
        state.applyEngineManagerStatus({
            engine: 'llamacpp',
            installed: false,
            running: false,
            port: 8081
        })
        state.beginLocalEngineOp('llama-cpp', 'installing')
        expect(statuses.at(-1)).toBe('installing')
        state.failLocalEngineOp(engineId, 'install')
        expect(statuses.at(-1)).toBe('not-installed')
    })

    it('does not revert a quiet install to Download after 90 seconds', () => {
        vi.useFakeTimers()
        const state = getModularBridgeState()
        const statuses: EngineProcessStatus[] = []
        unsubscribe = subscribePush(event => {
            if (
                event.channel === 'engines:state-changed' &&
                event.payload.engineType === 'llama-cpp' &&
                event.payload.status
            ) {
                statuses.push(event.payload.status.processStatus)
            }
        })

        state.setSelfId('local-node')
        state.applyEngineManagerStatus({
            engine: 'llamacpp',
            installed: false,
            running: false,
            port: 8081
        })
        state.beginLocalEngineOp('llama-cpp', 'installing')

        vi.advanceTimersByTime(120_000)
        expect(statuses.at(-1)).toBe('installing')

        state.applyEngineManagerProgress({ engine: 'llamacpp', stage: 'installing' })
        vi.advanceTimersByTime(30 * 60_000 - 1)
        expect(statuses.at(-1)).toBe('installing')
        vi.advanceTimersByTime(1)
        expect(statuses.at(-1)).toBe('not-installed')

        state.beginLocalEngineOp('llama-cpp', 'starting')
        vi.advanceTimersByTime(90_000)
        expect(statuses.at(-1)).toBe('not-installed')
    })
})

describe('install progress percent', () => {
    function captureInstallProgress(): EngineProgress[] {
        const frames: EngineProgress[] = []
        unsubscribe = subscribePush(event => {
            if (
                event.channel === 'engines:progress-changed' &&
                event.payload.operation === 'install'
            ) {
                frames.push(event.payload)
            }
        })
        return frames
    }

    it('drops a local install to indeterminate after a measured download', () => {
        const state = getModularBridgeState()
        const frames = captureInstallProgress()
        state.setSelfId('local-node')

        state.applyEngineManagerProgress({ engine: 'llamacpp', stage: 'downloading', percent: 42 })
        expect(frames.at(-1)?.percent).toBe(42)

        state.applyEngineManagerProgress({ engine: 'llamacpp', stage: 'downloading', percent: 100 })
        state.applyEngineManagerProgress({ engine: 'llamacpp', stage: 'installing' })
        expect(frames.at(-1)?.status).toBe('installing')
        expect(frames.at(-1)?.percent).toBeUndefined()
    })

    it('drops a remote install to indeterminate after a measured download', () => {
        const state = getModularBridgeState()
        const frames = captureInstallProgress()
        const frame = { node: 'remote-node', engine: 'llamacpp', op: 'install' }

        state.applyRemoteEngineProgress({ ...frame, stage: 'downloading', percent: 100 })
        expect(frames.at(-1)?.percent).toBe(100)

        state.applyRemoteEngineProgress({ ...frame, stage: 'installing' })
        expect(frames.at(-1)?.status).toBe('installing')
        expect(frames.at(-1)?.percent).toBeUndefined()

        state.applyRemoteEngineProgress({ ...frame, stage: 'failed', percent: -1 })
    })
})
