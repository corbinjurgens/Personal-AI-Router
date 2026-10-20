// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import os from 'os'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { JsonRpcSocketClient, type JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { emitBridgePush } from '@/electron/service-bridge/broadcaster'
import { initPlatform } from '@/electron/globals'

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))
vi.mock('@/electron/config/ui-config', () => ({ isFirstRun: () => false }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({
    emitBridgePush: vi.fn<typeof emitBridgePush>()
}))
vi.mock('@/electron/service-bridge/node-info-poller', () => ({
    startNodeInfoPoller: vi.fn(),
    stopNodeInfoPoller: vi.fn()
}))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        verbose: vi.fn()
    })
}))

function deferred() {
    let resolve: (value: JsonValue) => void = () => {
        throw new Error('Deferred response was not initialized')
    }
    const promise = new Promise<JsonValue>(done => {
        resolve = done
    })
    return { promise, resolve }
}

function baseline(model: string, includeRemoved = false): JsonValue {
    const workloads = [{ id: 'kept', engine: 'ollama', model, state: 'completed', createdAt: 1 }]
    if (includeRemoved) {
        workloads.push({
            id: 'removed',
            engine: 'ollama',
            model: 'removed-model',
            state: 'completed',
            createdAt: 2
        })
    }
    return { workloads }
}

const supervisor = getModularSupervisor()
const state = getModularBridgeState()
let client: JsonRpcSocketClient
let response: (method: string) => Promise<JsonValue>
const methods: string[] = []

function ready(): void {
    client.emit('notification', { source: 'broker', method: 'app:ready' })
}

function refreshCount(): number {
    return vi
        .mocked(emitBridgePush)
        .mock.calls.filter(([channel]) => channel === 'state:request-refresh').length
}

async function flush(): Promise<void> {
    // Drain the finite hydration chain without advancing periodic poll timers.
    for (let turn = 0; turn < 100; turn++) await Promise.resolve()
}

describe('broker session hydration', () => {
    beforeEach(() => {
        const userData = process.env.PAIR_USER_DATA
        if (!userData) throw new Error('Isolated user data was not configured')
        initPlatform({
            getUserData: () => userData,
            getTemp: () => os.tmpdir(),
            getResourcesPath: () => userData
        })
        vi.useFakeTimers()
        methods.length = 0
        response = async method => {
            if (method === 'workloads:get-initial') return baseline('fresh')
            if (method === 'cluster:get-node-id') return { nodeUuid: 'self' }
            if (method === 'nodes:get-initial') return { nodes: [] }
            if (method === 'engine:get-installed' || method === 'engine:descriptors')
                return { engines: [] }
            return {}
        }
        vi.spyOn(fs, 'existsSync').mockImplementation(file => String(file).includes('nvpair-'))
        const startedClients: JsonRpcSocketClient[] = []
        vi.spyOn(JsonRpcSocketClient.prototype, 'start').mockImplementation(function (
            this: JsonRpcSocketClient
        ) {
            startedClients.push(this)
        })
        vi.spyOn(JsonRpcSocketClient.prototype, 'detach').mockResolvedValue(undefined)
        vi.spyOn(JsonRpcSocketClient.prototype, 'call').mockImplementation(async method => {
            methods.push(method)
            return response(method)
        })
        vi.mocked(emitBridgePush).mockClear()
        supervisor.start()
        const startedClient = startedClients[0]
        if (!startedClient) throw new Error('Supervisor did not start a transport client')
        client = startedClient
    })

    afterEach(async () => {
        await supervisor.stop()
        vi.restoreAllMocks()
        vi.useRealTimers()
    })

    it('drops an old baseline released after replacement hydration and removes absent rows', async () => {
        const old = deferred()
        let baselineRead = 0
        const defaultResponse = response
        response = method => {
            if (method !== 'workloads:get-initial') return defaultResponse(method)
            baselineRead++
            if (baselineRead === 1) return Promise.resolve(baseline('before', true))
            if (baselineRead === 2) return old.promise
            return Promise.resolve(baseline('after'))
        }
        ready()
        await flush()
        expect(
            Object.values(state.getWorkloads())
                .map(job => job.id)
                .sort()
        ).toEqual(['kept', 'removed'])

        client.emit('broker-restarted')
        ready()
        await flush()
        expect(baselineRead).toBe(2)
        expect(state.getWorkloads()).toEqual({})
        const completedBeforeReplacement = refreshCount()

        client.emit('disconnected')
        ready()
        await flush()
        expect(baselineRead).toBe(3)
        expect(
            Object.values(state.getWorkloads()).map(job => ({ id: job.id, model: job.model }))
        ).toEqual([{ id: 'kept', model: 'after' }])
        expect(refreshCount()).toBe(completedBeforeReplacement + 1)
        const callsAfterFreshBaseline = methods.length

        old.resolve(baseline('obsolete', true))
        await flush()
        expect(
            Object.values(state.getWorkloads()).map(job => ({ id: job.id, model: job.model }))
        ).toEqual([{ id: 'kept', model: 'after' }])
        expect(methods).toHaveLength(callsAfterFreshBaseline)
        expect(refreshCount()).toBe(completedBeforeReplacement + 1)
    })

    it('keeps early readiness separate from fresh hydration completion', async () => {
        const pending = deferred()
        const defaultResponse = response
        response = method =>
            method === 'workloads:get-initial' ? pending.promise : defaultResponse(method)
        ready()
        await supervisor.waitUntilReady(100)
        await flush()
        expect(methods).toContain('workloads:get-initial')
        expect(refreshCount()).toBe(0)
        expect(state.getWorkloads()).toEqual({})

        pending.resolve(baseline('accepted'))
        await flush()
        expect(Object.values(state.getWorkloads()).map(job => job.model)).toEqual(['accepted'])
        expect(refreshCount()).toBe(1)
    })

    it('does not apply an obsolete policy read or resume its later engine hydration', async () => {
        const oldPolicy = deferred()
        let policyReads = 0
        const defaultResponse = response
        response = method => {
            if (method !== 'policy:get') return defaultResponse(method)
            policyReads++
            return policyReads === 1
                ? oldPolicy.promise
                : Promise.resolve({ availability: 'available' })
        }
        ready()
        await flush()
        expect(policyReads).toBe(1)
        expect(methods).not.toContain('engine:descriptors')

        client.emit('disconnected')
        ready()
        await flush()
        expect(policyReads).toBe(2)
        expect(supervisor.getAvailabilityMenuItem()).toEqual({
            label: 'Pause inference on this PC',
            enabled: true,
            request: 'paused'
        })
        expect(methods.filter(method => method === 'engine:descriptors')).toHaveLength(1)
        const completed = refreshCount()
        oldPolicy.resolve({ availability: 'paused' })
        await flush()
        expect(supervisor.getAvailabilityMenuItem()).toEqual({
            label: 'Pause inference on this PC',
            enabled: true,
            request: 'paused'
        })
        expect(methods.filter(method => method === 'engine:descriptors')).toHaveLength(1)
        expect(refreshCount()).toBe(completed)
    })

    it('does not continue hydration or emit completion after detach mid-call', async () => {
        const pending = deferred()
        response = async method => (method === 'discovery:subscribe' ? pending.promise : {})
        ready()
        await flush()
        expect(methods).toEqual(['performance:subscribe', 'discovery:subscribe', 'performance:get'])
        await supervisor.stop()
        pending.resolve({})
        await flush()
        expect(methods).toEqual([
            'performance:subscribe',
            'discovery:subscribe',
            'performance:get',
            'performance:unsubscribe'
        ])
        expect(refreshCount()).toBe(0)
    })
})
