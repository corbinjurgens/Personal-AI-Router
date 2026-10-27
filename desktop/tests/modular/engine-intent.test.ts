// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import os from 'os'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { parseEngineIntent } from '@/electron/service-bridge/engine-intent'
import { JsonRpcSocketClient, type JsonValue } from '@/electron/service-bridge/json-rpc-client'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import { emitBridgePush } from '@/electron/service-bridge/broadcaster'
import { initPlatform } from '@/electron/globals'
import type { EngineInitialState, EngineIntentSnapshot } from '@/shared/types/engine-api'
import type { PreloadServiceTransport } from '@/shared/types/service-bridge'
import { createEngineApi } from '@/ui/api/engine-api'
import { serviceTransport } from '@/ui/api/service-transport'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'

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

const local: EngineIntentSnapshot = {
    enabledByEngine: { ollama: false, lmstudio: true, custom: false },
    offByEngine: { ollama: 'local' }
}
const peer: EngineIntentSnapshot = {
    enabledByEngine: { ollama: false, lmstudio: false },
    offByEngine: { ollama: 'peer', lmstudio: 'local' }
}
const initial: EngineInitialState = {
    descriptors: [],
    statuses: [],
    models: [],
    activeProgress: [],
    updateAvailable: []
}

function wire(snapshot: EngineIntentSnapshot): JsonValue {
    return {
        enabledByEngine: { ...snapshot.enabledByEngine },
        offByEngine: { ...snapshot.offByEngine }
    }
}

function deferred<T>() {
    let resolve: (value: T) => void = () => {
        throw new Error('Uninitialized deferred')
    }
    let reject: (error: Error) => void = () => {
        throw new Error('Uninitialized deferred')
    }
    const promise = new Promise<T>((done, fail) => {
        resolve = done
        reject = fail
    })
    return { promise, resolve, reject }
}

async function flush(): Promise<void> {
    for (let turn = 0; turn < 100; turn++) await Promise.resolve()
}

describe('strict engine intent parser', () => {
    it('preserves local and peer origins, On and default false without origin', () => {
        expect(parseEngineIntent(wire(local))).toEqual(local)
        expect(parseEngineIntent(wire(peer))).toEqual(peer)
    })
    it('accepts present empty maps as an authoritative empty snapshot', () => {
        expect(parseEngineIntent({ enabledByEngine: {}, offByEngine: {} })).toEqual({
            enabledByEngine: {},
            offByEngine: {}
        })
    })
    it('preserves arbitrary registered names including object prototype keys', () => {
        const value: JsonValue = JSON.parse(
            '{"enabledByEngine":{"__proto__":false},"offByEngine":{"__proto__":"peer"}}'
        )
        expect(parseEngineIntent(value)).toEqual(value)
    })
    const invalid: [string, JsonValue | undefined][] = [
        ['missing snapshot', undefined],
        ['null snapshot', null],
        ['array snapshot', []],
        ['missing enabled map', { offByEngine: {} }],
        ['missing origin map', { enabledByEngine: {} }],
        ['null enabled map', { enabledByEngine: null, offByEngine: {} }],
        ['array enabled map', { enabledByEngine: [], offByEngine: {} }],
        ['scalar enabled map', { enabledByEngine: true, offByEngine: {} }],
        ['null origin map', { enabledByEngine: {}, offByEngine: null }],
        ['array origin map', { enabledByEngine: {}, offByEngine: [] }],
        ['scalar origin map', { enabledByEngine: {}, offByEngine: 'local' }],
        ['nonboolean enabled', { enabledByEngine: { ollama: 0 }, offByEngine: {} }],
        [
            'invalid origin',
            { enabledByEngine: { ollama: false }, offByEngine: { ollama: 'remote' } }
        ],
        [
            'nonstring origin',
            { enabledByEngine: { ollama: false }, offByEngine: { ollama: false } }
        ],
        ['origin for On', { enabledByEngine: { ollama: true }, offByEngine: { ollama: 'local' } }],
        ['origin for unreported engine', { enabledByEngine: {}, offByEngine: { ollama: 'peer' } }],
        ['origin for inherited key', { enabledByEngine: {}, offByEngine: { constructor: 'local' } }]
    ]
    it.each(invalid)('rejects %s', (_name, value) => {
        expect(() => parseEngineIntent(value)).toThrow('Invalid engine intent snapshot')
    })
})

describe('local intent bridge', () => {
    const supervisor = getModularSupervisor()
    const state = getModularBridgeState()
    let client: JsonRpcSocketClient
    let read: () => Promise<JsonValue>
    const methods: string[] = []

    function notify(method: string, params?: JsonValue): void {
        client.emit('notification', { source: 'broker', method, params })
    }

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
        read = async () => wire(local)
        vi.spyOn(fs, 'existsSync').mockImplementation(file => String(file).includes('nvpair-'))
        const clients: JsonRpcSocketClient[] = []
        vi.spyOn(JsonRpcSocketClient.prototype, 'start').mockImplementation(function (
            this: JsonRpcSocketClient
        ) {
            clients.push(this)
        })
        vi.spyOn(JsonRpcSocketClient.prototype, 'detach').mockResolvedValue(undefined)
        vi.spyOn(JsonRpcSocketClient.prototype, 'call').mockImplementation(async method => {
            methods.push(method)
            if (method === 'engine:intent') return read()
            if (method === 'cluster:get-node-id') return { nodeUuid: 'self' }
            if (method === 'workloads:get-initial') return { workloads: [] }
            if (method === 'engine:descriptors' || method === 'engine:get-installed')
                return { engines: [] }
            if (method === 'nodes:get-initial') return { nodes: [] }
            return {}
        })
        state.setSelfId('self')
        supervisor.start()
        const started = clients[0]
        if (!started) throw new Error('Supervisor did not start a mocked client')
        client = started
    })
    afterEach(async () => {
        await supervisor.stop()
        vi.restoreAllMocks()
        vi.useRealTimers()
    })

    it('hydrates after subscription and exposes the complete typed push', async () => {
        expect(state.getEngineIntent()).toBeNull()
        notify('app:ready')
        await flush()
        expect(methods.indexOf('engine:subscribe')).toBeLessThan(methods.indexOf('engine:intent'))
        expect(state.getEngineIntent()).toEqual(local)
        await expect(handleServiceBridgeInvoke('engines:get-intent', undefined)).resolves.toEqual(
            local
        )
        expect(emitBridgePush).toHaveBeenCalledWith('engines:intent-changed', local)
    })
    it('applies push atomically without altering lifecycle status', async () => {
        notify('app:ready')
        await flush()
        const statuses = state.getEngineInitialState().statuses
        notify('engine:intent-changed', wire(peer))
        expect(state.getEngineIntent()).toEqual(peer)
        expect(state.getEngineInitialState().statuses).toEqual(statuses)
        expect(emitBridgePush).toHaveBeenCalledWith('engines:intent-changed', peer)
    })
    it('rejects malformed push and clears both maps to null', async () => {
        notify('app:ready')
        await flush()
        notify('engine:intent-changed', { enabledByEngine: { ollama: false } })
        expect(state.getEngineIntent()).toBeNull()
        expect(emitBridgePush).toHaveBeenCalledWith('engines:intent-changed', null)
    })
    it('ignores late hydration after a newer intent push', async () => {
        const pending = deferred<JsonValue>()
        read = () => pending.promise
        notify('app:ready')
        await flush()
        expect(methods).toContain('engine:intent')
        notify('engine:intent-changed', wire(peer))
        pending.resolve(wire(local))
        await flush()
        expect(state.getEngineIntent()).toEqual(peer)
    })
    it('ignores a failed read after a newer push', async () => {
        const pending = deferred<JsonValue>()
        read = () => pending.promise
        notify('app:ready')
        await flush()
        notify('engine:intent-changed', wire(peer))
        pending.reject(new Error('read unavailable'))
        await flush()
        expect(state.getEngineIntent()).toEqual(peer)
    })
    it.each(['disconnected', 'broker-restarted', 'exit'])(
        'invalidates on %s and fences the previous session',
        async event => {
            const pending = deferred<JsonValue>()
            read = () => pending.promise
            notify('app:ready')
            await flush()
            notify('engine:intent-changed', wire(peer))
            if (event === 'exit') client.emit('exit', { source: 'broker', code: 1 })
            else if (event === 'disconnected') client.emit('disconnected')
            else client.emit('broker-restarted')
            expect(state.getEngineIntent()).toBeNull()
            pending.resolve(wire(local))
            await flush()
            expect(state.getEngineIntent()).toBeNull()
        }
    )
    it.each(['disconnected', 'broker-restarted', 'exit', 'engine:ready'])(
        'forgets a reported engine lock on %s',
        async event => {
            notify('app:ready')
            await flush()
            state.setSelfId('lock-invalidation-self')
            state.applyEngineManagerStatus({
                engine: 'llamacpp',
                installed: true,
                running: false,
                port: 0,
                locked: false
            })
            const lockOf = () =>
                state
                    .getEngineInitialState()
                    .statuses.find(
                        row =>
                            row.nodeId === 'lock-invalidation-self' &&
                            row.engineType === 'llama-cpp'
                    )
            expect(lockOf()).toHaveProperty('locked', false)
            if (event === 'exit') client.emit('exit', { source: 'broker', code: 1 })
            else if (event === 'engine:ready') notify('engine:ready')
            else if (event === 'disconnected') client.emit('disconnected')
            else client.emit('broker-restarted')
            await flush()
            expect(lockOf()).toMatchObject({ processStatus: 'stopped' })
            expect(lockOf()).not.toHaveProperty('locked')
        }
    )
    it('keeps a new broker snapshot when the old session read arrives last', async () => {
        const pending = deferred<JsonValue>()
        read = () => pending.promise
        notify('app:ready')
        await flush()
        client.emit('broker-restarted')
        read = async () => wire(peer)
        notify('app:ready')
        await flush()
        expect(state.getEngineIntent()).toEqual(peer)
        pending.resolve(wire(local))
        await flush()
        expect(state.getEngineIntent()).toEqual(peer)
    })
    it('does not start intent hydration when a subscription finishes in an old broker session', async () => {
        const pending = deferred<JsonValue>()
        const call = vi.mocked(JsonRpcSocketClient.prototype.call)
        const respond = call.getMockImplementation()
        if (!respond) throw new Error('Missing mocked socket response')
        call.mockImplementation(function (this: JsonRpcSocketClient, method, params, timeout) {
            if (method === 'engine:subscribe') return pending.promise
            return respond.call(this, method, params, timeout)
        })
        notify('app:ready')
        await flush()
        client.emit('broker-restarted')
        pending.resolve({})
        await flush()
        expect(methods).not.toContain('engine:intent')
        expect(state.getEngineIntent()).toBeNull()
    })
    it('invalidates on manager ready and rehydrates the new manager', async () => {
        notify('app:ready')
        await flush()
        read = async () => wire(peer)
        notify('engine:ready')
        expect(state.getEngineIntent()).toBeNull()
        await flush()
        expect(state.getEngineIntent()).toEqual(peer)
    })
    it.each(['rejected', 'malformed'])('clears intent after a %s manager read', async failure => {
        notify('app:ready')
        await flush()
        read =
            failure === 'rejected'
                ? async () => {
                      throw new Error('unavailable')
                  }
                : async () => ({})
        notify('engine:ready')
        await flush()
        expect(state.getEngineIntent()).toBeNull()
    })
    it('fences old manager hydration after restart invalidation', async () => {
        const old = deferred<JsonValue>()
        const fresh = deferred<JsonValue>()
        read = () => old.promise
        notify('app:ready')
        await flush()
        read = () => fresh.promise
        notify('engine:ready')
        old.resolve(wire(local))
        await flush()
        expect(state.getEngineIntent()).toBeNull()
        fresh.resolve(wire(peer))
        await flush()
        expect(state.getEngineIntent()).toEqual(peer)
    })
})

function apiFixture() {
    const invoke = vi
        .fn<PreloadServiceTransport['invoke']>()
        .mockResolvedValue({ success: true, data: local })
    const unsubscribe = vi.fn<() => void>()
    const subscribePush = vi.fn<PreloadServiceTransport['subscribePush']>(() => unsubscribe)
    const preload: PreloadServiceTransport = {
        connected: true,
        invoke,
        invokeHistory: vi.fn<PreloadServiceTransport['invokeHistory']>(() => {
            throw new Error('Unexpected history invoke in ordinary transport fixture')
        }),
        subscribePush,
        onConnect: () => () => {},
        onDisconnect: () => () => {},
        onAuthFailure: () => () => {},
        destroy: () => {}
    }
    return { api: createEngineApi(serviceTransport(preload)), invoke, subscribePush, unsubscribe }
}

describe('intent typed API and renderer store', () => {
    afterEach(() => {
        useEngineStatusStore.getState().cleanup()
        vi.restoreAllMocks()
        vi.unstubAllGlobals()
    })
    it('uses the typed get and push channels with teardown', async () => {
        const { api, invoke, subscribePush, unsubscribe } = apiFixture()
        await expect(api.getIntent()).resolves.toEqual(local)
        expect(invoke).toHaveBeenCalledWith('engines:get-intent')
        const callback = vi.fn<Parameters<typeof api.onIntentChanged>[0]>()
        api.onIntentChanged(callback)()
        expect(subscribePush).toHaveBeenCalledWith('engines:intent-changed', callback)
        expect(unsubscribe).toHaveBeenCalledOnce()
    })

    function storeFixture() {
        const { api } = apiFixture()
        const pending = deferred<EngineIntentSnapshot | null>()
        let push: Parameters<typeof api.onIntentChanged>[0] = () => {
            throw new Error('Not subscribed')
        }
        const unsubscribe = vi.fn<() => void>()
        const subscribe = vi.spyOn(api, 'onIntentChanged').mockImplementation(callback => {
            push = callback
            return unsubscribe
        })
        const read = vi.spyOn(api, 'getIntent').mockImplementation(() => {
            expect(subscribe).toHaveBeenCalledOnce()
            return pending.promise
        })
        const removed = vi.fn<() => void>()
        vi.stubGlobal('window', { pairApi: { engines: api, nodes: { onRemove: () => removed } } })
        return {
            pending,
            read,
            push: (value: EngineIntentSnapshot | null) => push(value),
            unsubscribe,
            removed
        }
    }
    it('subscribes before fetching and retains a newer push over hydration', async () => {
        const fixture = storeFixture()
        const initializing = useEngineStatusStore.getState().initialize(initial)
        fixture.push(peer)
        fixture.pending.resolve(local)
        await initializing
        expect(useEngineStatusStore.getState().intent).toEqual(peer)
        expect(useEngineStatusStore.getState().statusByNode.size).toBe(0)
    })
    it('hydrates a complete snapshot without deriving lifecycle facts', async () => {
        const fixture = storeFixture()
        const initializing = useEngineStatusStore.getState().initialize(initial)
        fixture.pending.resolve(local)
        await initializing
        expect(useEngineStatusStore.getState().intent).toEqual(local)
        expect(useEngineStatusStore.getState().getStatus('self', 'ollama').processStatus).toBe(
            'initializing'
        )
    })
    it('retains null after disconnect push despite an older read', async () => {
        const fixture = storeFixture()
        const initializing = useEngineStatusStore.getState().initialize(initial)
        fixture.push(peer)
        fixture.push(null)
        fixture.pending.resolve(local)
        await initializing
        expect(useEngineStatusStore.getState().intent).toBeNull()
    })
    it('keeps null on failed read', async () => {
        const fixture = storeFixture()
        const initializing = useEngineStatusStore.getState().initialize(initial)
        fixture.pending.reject(new Error('unavailable'))
        await initializing
        expect(useEngineStatusStore.getState().intent).toBeNull()
    })
    it('reinitializes without stacking subscriptions or accepting a prior read', async () => {
        const fixture = storeFixture()
        const first = useEngineStatusStore.getState().initialize(initial)
        fixture.read.mockResolvedValueOnce(peer)
        await useEngineStatusStore.getState().initialize(initial)
        fixture.pending.resolve(local)
        await first
        expect(useEngineStatusStore.getState().intent).toEqual(peer)
        expect(fixture.unsubscribe).toHaveBeenCalledOnce()
        expect(fixture.removed).toHaveBeenCalledOnce()
    })
    it('removes lifecycle subscriptions after teardown during the initial fetch', async () => {
        const { api } = apiFixture()
        const pending = deferred<EngineInitialState>()
        vi.spyOn(api, 'getInitialState').mockReturnValue(pending.promise)
        const descriptors = vi.spyOn(api, 'onDescriptorsChanged')
        const states = vi.spyOn(api, 'onStateChanged')
        const unsubscribe = vi.fn<() => void>()
        vi.stubGlobal('window', {
            pairApi: { engines: api, nodes: { onRemove: () => unsubscribe } }
        })
        const initializing = useEngineStatusStore.getState().initialize()
        useEngineStatusStore.getState().cleanup()
        pending.resolve(initial)
        await initializing
        expect(useEngineStatusStore.getState().intent).toBeNull()
        expect(descriptors).toHaveBeenCalledOnce()
        expect(states).toHaveBeenCalledOnce()
        expect(unsubscribe).toHaveBeenCalledOnce()
    })
    it('cleans subscriptions and ignores read and callback completion after teardown', async () => {
        const fixture = storeFixture()
        const initializing = useEngineStatusStore.getState().initialize(initial)
        useEngineStatusStore.getState().cleanup()
        fixture.push(peer)
        fixture.pending.resolve(local)
        await initializing
        expect(useEngineStatusStore.getState().intent).toBeNull()
        expect(fixture.unsubscribe).toHaveBeenCalledOnce()
        expect(fixture.removed).toHaveBeenCalledOnce()
    })
})
