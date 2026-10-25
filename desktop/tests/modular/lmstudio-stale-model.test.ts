// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { createOverviewWindow } from '@/electron/window'
import type { JsonObject } from '@/electron/service-bridge/json-rpc-client'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { parseModelsResult } from '@/electron/service-bridge/model-download'
import { getModularSupervisor, pullModelParams } from '@/electron/service-bridge/modular-supervisor'

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<typeof createOverviewWindow>() }))

let fixtureId = 0

function inventory(names: string[], revision = '1'): JsonObject {
    return {
        models: names,
        modelsByEngine: { lmstudio: names },
        loadedByEngine: { lmstudio: [] },
        downloadState: {
            version: 1,
            sourceId: fixtureId.toString(16).padStart(32, '0'),
            revision,
            recoveryRequired: false,
            entries: []
        }
    }
}

function setup(names: string[]) {
    const state = getModularBridgeState()
    const supervisor = getModularSupervisor()
    const nodeId = `lmstudio-current-inventory-${fixtureId}`
    state.setSelfId(nodeId)
    state.applyEngineManagerStatus({
        engine: 'lmstudio',
        installed: true,
        running: true,
        healthy: true,
        port: 1235
    })
    const discoveryNode = (models: string[]) => ({
        hostUuid: nodeId,
        name: nodeId,
        ipAddress: '127.0.0.1',
        port: 14318,
        models,
        modelsByEngine: { lmstudio: models }
    })
    state.handleNotification({
        source: 'broker',
        method: 'discovery:nodes-changed',
        params: { nodes: [discoveryNode(names)] }
    })
    state.applyLocalModelsResult(parseModelsResult(inventory(names)))
    const hasProcess = vi.spyOn(supervisor, 'hasProcess').mockReturnValue(true)
    const call = vi.spyOn(supervisor, 'callProcess').mockResolvedValue(inventory(names))
    const rows = () =>
        state
            .getEngineInitialState()
            .models.find(models => models.nodeId === nodeId && models.engineType === 'lm-studio')
            ?.models ?? []
    return { state, supervisor, discoveryNode, hasProcess, call, rows }
}

async function flush(): Promise<void> {
    await Promise.resolve()
    await Promise.resolve()
}

beforeEach(() => {
    fixtureId++
    vi.useFakeTimers()
    getModularBridgeState().setLocalDiscoveryModelRefresher(() => {})
})

afterEach(() => {
    const supervisor = getModularSupervisor()
    vi.spyOn(supervisor, 'hasProcess').mockReturnValue(false)
    supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: false })
    getModularBridgeState().setLocalDiscoveryModelRefresher(() => {})
    vi.restoreAllMocks()
    vi.useRealTimers()
})

describe('canonical engine model wire formats', () => {
    it('selects engine-attributed canonical names and refuses raw router inventories', () => {
        expect(
            parseModelsResult({ models: ['ready'], modelsByEngine: { llamacpp: ['ready'] } })
                .modelsByEngine
        ).toEqual({ llamacpp: ['ready'] })
        expect(() =>
            parseModelsResult({ data: [{ id: 'partial', status: { value: 'downloading' } }] })
        ).toThrow('malformed inventory')
        expect(() => parseModelsResult({ models: [{ key: 'native-name' }] })).toThrow(
            'malformed inventory'
        )
    })

    it('distinguishes a successfully queried empty engine from an unqueryable one', () => {
        expect(parseModelsResult(inventory([])).modelsByEngine).toEqual({ lmstudio: [] })
        expect(parseModelsResult({ models: [] }).modelsByEngine).toBeUndefined()
        expect(() =>
            parseModelsResult({ models: [], modelsByEngine: { lmstudio: [null] } })
        ).toThrow('malformed inventory')
    })

    it('uses the manifest model field for pulls', () => {
        expect(pullModelParams('ollama', 'demo')).toEqual({ name: 'demo' })
        expect(pullModelParams('lmstudio', 'demo')).toEqual({ model: 'demo' })
        expect(pullModelParams('llamacpp', 'demo')).toEqual({ model: 'demo' })
    })
})

describe('LM Studio canonical inventory reconciliation', () => {
    it('retains successful empty inventory after a failure instead of resurrecting discovery names', async () => {
        const { supervisor, call, rows } = setup(['stale-discovery-model'])
        call.mockResolvedValueOnce(inventory([])).mockRejectedValueOnce(new Error('offline'))
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: true })
        await flush()
        expect(rows()).toEqual([])
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: true })
        await flush()
        expect(rows()).toEqual([])
        expect(call).toHaveBeenNthCalledWith(1, 'broker', 'engine:models')
        expect(call).toHaveBeenCalledTimes(2)
    })

    it('retains last-good names with unavailable readiness when a current query fails', async () => {
        const { supervisor, call, rows } = setup(['last-good'])
        expect(rows()[0]?.downloaded).toBe(true)
        call.mockRejectedValueOnce(new Error('offline'))
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: true })
        await flush()
        expect(rows()).toMatchObject([
            { name: 'last-good', downloaded: false, downloadStateCurrent: false }
        ])
    })

    it('preserves last-good inventory across stop and a malformed restart query', async () => {
        const { supervisor, call, rows } = setup(['last-good'])
        const stopped = { ...inventory([]), modelsByEngine: {} }
        call.mockResolvedValueOnce(stopped).mockResolvedValueOnce({ models: null })
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: false })
        expect(rows()).toMatchObject([{ name: 'last-good', downloaded: false }])
        await flush()
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: true })
        await flush()
        expect(rows()).toMatchObject([
            { name: 'last-good', downloaded: false, downloadStateCurrent: false }
        ])
        expect(call).toHaveBeenCalledTimes(2)
    })

    it('does not let an older running refresh restore readiness after a later stop', async () => {
        const { supervisor, hasProcess, call, rows } = setup(['last-good'])
        const delayed = Promise.withResolvers<JsonObject>()
        call.mockReturnValueOnce(delayed.promise)
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: true })
        hasProcess.mockReturnValue(false)
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: false })
        delayed.resolve(inventory(['late-model']))
        await flush()
        expect(rows()).toMatchObject([{ name: 'last-good', downloaded: false }])
        expect(call).toHaveBeenCalledOnce()
    })

    it('refreshes a deleted final model through the broker while its proxy is unavailable', async () => {
        const { state, supervisor, discoveryNode, call, rows } = setup(['deleted-model'])
        state.setLocalDiscoveryModelRefresher(engine =>
            supervisor.refreshDiscoveryEngineModels(engine)
        )
        call.mockResolvedValueOnce(inventory(['deleted-model']))
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: true })
        await flush()
        call.mockResolvedValueOnce(inventory([]))
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: { nodes: [discoveryNode([])] }
        })
        await flush()
        expect(call).toHaveBeenCalledTimes(2)
        expect(rows()).toEqual([])
    })

    it('refreshes only the engine whose attribution changed when the union did not', () => {
        const { state, discoveryNode } = setup(['shared-model'])
        state.applyEngineManagerStatus({
            engine: 'ollama',
            installed: true,
            running: true,
            healthy: true
        })
        const node = {
            ...discoveryNode(['shared-model']),
            modelsByEngine: {
                ollama: ['shared-model'],
                lmstudio: ['shared-model']
            }
        }
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: [node]
        })
        const refresh = vi.fn<Parameters<typeof state.setLocalDiscoveryModelRefresher>[0]>()
        state.setLocalDiscoveryModelRefresher(refresh)
        state.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: [{ ...node, modelsByEngine: { ollama: ['shared-model'], lmstudio: [] } }]
        })
        expect(refresh).toHaveBeenCalledExactlyOnceWith('lm-studio')
    })

    it('retries a failed discovery refresh without another discovery event', async () => {
        const { supervisor, call, rows } = setup(['deleted-model'])
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: true })
        await flush()
        call.mockRejectedValueOnce(new Error('temporary failure')).mockResolvedValueOnce(
            inventory([])
        )
        supervisor.refreshDiscoveryEngineModels('lm-studio')
        await flush()
        expect(call).toHaveBeenCalledTimes(2)
        await vi.advanceTimersByTimeAsync(1_000)
        expect(call).toHaveBeenCalledTimes(3)
        expect(rows()).toEqual([])
    })

    it('cancels a pending discovery retry when the engine stops', async () => {
        const { supervisor, call } = setup(['ready'])
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: true })
        await flush()
        call.mockRejectedValueOnce(new Error('schedule retry'))
        supervisor.refreshDiscoveryEngineModels('lm-studio')
        await flush()
        supervisor.refreshManagedEngineModels({ engine: 'lmstudio', running: false })
        await flush()
        expect(call).toHaveBeenCalledTimes(3)
        await vi.advanceTimersByTimeAsync(2_000)
        expect(call).toHaveBeenCalledTimes(3)
    })
})
