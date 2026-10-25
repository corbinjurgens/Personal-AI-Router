// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EventEmitter } from 'events'
import fs from 'fs'
import path from 'path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ipcMain, ipcRenderer } from 'electron'
import type {
    PreloadServiceTransport,
    ServiceBridgePushMessage,
    ServiceInvokeArgs
} from '@/shared/types/service-bridge'
import type { IpcResult, IpcChannelKey, IpcChannelMap } from '@/shared/types/ipc-channels'
import type { WsInvokeChannel, WsInvokeResponse } from '@/shared/types/ws-channels'
import type { RoutingHistoryReady, RoutingHistoryRecord } from '@/shared/types/routing'

const fixtures = vi.hoisted(() => {
    const windows: TestWindow[] = []
    const handlers = new Map<string, Parameters<typeof ipcMain.handle>[1]>()
    return { windows, handlers, wanted: vi.fn<(wanted: boolean) => void>() }
})
vi.mock('electron', () => ({
    ipcRenderer: {
        invoke: vi.fn(async () => ({ success: true, data: null })),
        on: vi.fn(),
        removeListener: vi.fn()
    },
    nativeImage: { createFromPath: () => ({}) },
    BrowserWindow: class {
        constructor() {
            return new TestWindow()
        }
        static getAllWindows() {
            return fixtures.windows
        }
        static fromId(id: number) {
            return fixtures.windows.find(window => window.id === id && !window.isDestroyed())
        }
        static fromWebContents(sender: object) {
            return fixtures.windows.find(
                window => window.contents === sender && !window.isDestroyed()
            )
        }
    },
    ipcMain: {
        handle: (channel: string, handler: Parameters<typeof ipcMain.handle>[1]) =>
            fixtures.handlers.set(channel, handler)
    }
}))
vi.mock('@electron-toolkit/utils', () => ({ is: { dev: false } }))
vi.mock('@/electron/open-external', () => ({ openExternalSafe: vi.fn() }))
vi.mock('@/electron/window-visibility', () => ({ trackWindowVisibility: vi.fn() }))
vi.mock('@/electron/service-bridge/modular-supervisor', () => ({
    getModularSupervisor: () => ({ setRoutingHistorySubscribed: fixtures.wanted })
}))
vi.mock('@/electron/service-bridge/empty-handlers', () => ({
    handleServiceBridgeInvoke: vi.fn(async () => ({ requests: [] }))
}))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        verbose: vi.fn()
    })
}))

import { registerServiceBridgeIpc } from '@/electron/ipc/service-bridge.ipc'
import { getRoutingDelivery } from '@/electron/service-bridge/routing-delivery'
import { applyRoutingHistoryChanged } from '@/electron/service-bridge/routing'
import { emitBridgePush } from '@/electron/service-bridge/broadcaster'
import { RoutingHistorySubscription } from '@/electron/service-bridge/routing'
import {
    JsonRpcResponseError,
    type JsonRpcSocketClient
} from '@/electron/service-bridge/json-rpc-client'
import type { RoutingHistoryMethod } from '@/electron/service-bridge/routing-json'
import { createPairTransport } from '@/preload/api/pair-transport.api'
import { serviceTransport } from '@/ui/api/service-transport'
import { createRoutingHistory } from '@/ui/api/routing-history'
import { createOverviewWindow, createTrayWindow } from '@/electron/window'

class TestWindow extends EventEmitter {
    readonly id = ++nextWindowId
    readonly contents = new Renderer()
    private gone = false
    constructor() {
        super()
        fixtures.windows.push(this)
    }
    get webContents(): Renderer {
        if (this.gone) throw new Error('webContents getter after actual close')
        return this.contents
    }
    isDestroyed(): boolean {
        return this.gone
    }
    isMaximized(): boolean {
        return false
    }
    show(): void {
        /* No native window in this fixture. */
    }
    focus(): void {
        /* No native window in this fixture. */
    }
    loadURL(): Promise<void> {
        return Promise.resolve()
    }
    loadFile(): Promise<void> {
        return Promise.resolve()
    }
    destroy(): void {
        if (this.gone) return
        this.gone = true
        this.contents.destroy()
        this.emit('closed')
    }
}
let nextWindowId = 0

class Renderer extends EventEmitter {
    readonly id = ++nextRendererId
    readonly mainFrame = {}
    readonly ready: RoutingHistoryReady[] = []
    readonly ordinary: ServiceBridgePushMessage[] = []
    readonly ipc = new RendererIpc(this)
    private gone = false
    getURL(): string {
        return 'file:///synthetic/index.html'
    }
    isDestroyed(): boolean {
        return this.gone
    }
    send(channel: string, message: RoutingHistoryReady | ServiceBridgePushMessage): void {
        if (channel === 'routing-history:ready' && 'generation' in message) this.ready.push(message)
        else if ('channel' in message) {
            this.ordinary.push(message)
            this.ipc.events.emit('service-bridge:push', {}, message)
        }
    }
    setWindowOpenHandler(): void {
        /* Navigation is emitted explicitly. */
    }
    destroy(): void {
        this.gone = true
        this.emit('destroyed')
    }
}
let nextRendererId = 1000
type PushListener = (event: Electron.IpcRendererEvent, message: ServiceBridgePushMessage) => void
type ReadyListener = (event: Electron.IpcRendererEvent, message: RoutingHistoryReady) => void
class RendererIpc {
    readonly events = new EventEmitter()
    readonly invoke = vi.fn<typeof ipcRenderer.invoke>()
    pauseData = false
    inFlight = 0
    maximumInFlight = 0
    maximumPacketBytes = 0
    private held: {
        result: IpcResult<WsInvokeResponse<WsInvokeChannel>>
        resolve: (result: IpcResult<WsInvokeResponse<WsInvokeChannel>>) => void
    } | null = null
    constructor(sender: Renderer) {
        this.invoke.mockImplementation(async (channel, payload) => {
            const handler = fixtures.handlers.get(channel)
            if (!handler) throw new Error('missing production IPC handler')
            const result: IpcResult<WsInvokeResponse<WsInvokeChannel>> = await Reflect.apply(
                handler,
                undefined,
                [{ sender, senderFrame: sender.mainFrame }, payload]
            )
            const copy = structuredClone(result)
            if (
                result.success &&
                result.data &&
                typeof result.data === 'object' &&
                'state' in result.data
            ) {
                if (result.data.state === 'data' || result.data.state === 'end') {
                    if (result.data.state === 'data') {
                        const bytes = 'bytes' in result.data ? result.data.bytes : null
                        if (!(bytes instanceof Uint8Array))
                            throw new Error('missing packed record bytes')
                        expect(bytes.byteLength).toBe(bytes.buffer.byteLength)
                        this.maximumPacketBytes = Math.max(
                            this.maximumPacketBytes,
                            bytes.byteLength
                        )
                    }
                    this.maximumInFlight = Math.max(this.maximumInFlight, ++this.inFlight)
                    if (this.pauseData)
                        return new Promise<IpcResult<WsInvokeResponse<WsInvokeChannel>>>(
                            resolve => {
                                if (this.held) throw new Error('second queued transfer')
                                this.held = { result: copy, resolve }
                            }
                        )
                } else if (
                    channel === 'routing-history:consumed' &&
                    (result.data.state === 'accepted' || result.data.state === 'retired')
                )
                    this.inFlight--
            }
            return copy
        })
    }
    on(channel: 'service-bridge:push', listener: PushListener): void
    on(channel: 'routing-history:ready', listener: ReadyListener): void
    on(channel: string, listener: PushListener | ReadyListener): void {
        this.events.on(channel, listener)
    }
    removeListener(channel: 'service-bridge:push', listener: PushListener): void
    removeListener(channel: 'routing-history:ready', listener: ReadyListener): void
    removeListener(channel: string, listener: PushListener | ReadyListener): void {
        this.events.removeListener(channel, listener)
    }
    resume(): void {
        const held = this.held
        this.held = null
        this.pauseData = false
        held?.resolve(held.result)
    }
}
function isolatedTransport(sender: Renderer) {
    const preload = createPairTransport(sender.ipc)
    const bridge: PreloadServiceTransport = {
        get connected() {
            return preload.connected
        },
        invoke<C extends WsInvokeChannel>(
            ...args: ServiceInvokeArgs<C>
        ): Promise<IpcResult<WsInvokeResponse<C>>> {
            return preload.invoke<C>(...args).then(result => structuredClone(result))
        },
        invokeHistory<C extends WsInvokeChannel>(
            complete: (result: IpcResult<WsInvokeResponse<C>>) => undefined,
            ...args: ServiceInvokeArgs<C>
        ): undefined {
            const returned = preload.invokeHistory<C>(
                result => {
                    const completion = complete(structuredClone(result))
                    expect(completion).toBeUndefined()
                    return undefined
                },
                ...args
            )
            expect(returned).toBeUndefined()
            return undefined
        },
        subscribePush: (channel, fn) =>
            preload.subscribePush(channel, payload => fn(structuredClone(payload))),
        onConnect: fn => preload.onConnect(fn),
        onDisconnect: fn => preload.onDisconnect(fn),
        onAuthFailure: fn => preload.onAuthFailure(fn),
        destroy: () => preload.destroy()
    }
    return serviceTransport(bridge)
}
const rendererFrames = new Map<number, FrameRequestCallback>()
let nextFrame = 0
async function receiveReady(sender: Renderer): Promise<void> {
    const token = sender.ready.shift()
    if (token) sender.ipc.events.emit('routing-history:ready', {}, token)
    await settle()
}
function runFrame(): void {
    const entry = rendererFrames.entries().next().value
    if (!entry) throw new Error('missing renderer frame')
    rendererFrames.delete(entry[0])
    entry[1](0)
}
async function commitInitial(renderer: Renderer, initialized: Promise<void>): Promise<void> {
    for (let turn = 0; turn < 258; turn++) {
        await receiveReady(renderer)
        if (rendererFrames.size) {
            runFrame()
            await initialized
            return
        }
    }
    throw new Error('initial did not reach its scalar end')
}
function sender(): Renderer {
    if (fixtures.windows.length === 0) createTrayWindow()
    else createOverviewWindow()
    const window = fixtures.windows.at(-1)
    if (!window) throw new Error('missing production window')
    window.contents.emit('did-finish-load')
    return window.contents
}
function owner(renderer: Renderer): TestWindow {
    const window = fixtures.windows.find(window => window.contents === renderer)
    if (!window) throw new Error('missing actual sender registration')
    return window
}
async function invoke<C extends IpcChannelKey>(
    renderer: Renderer,
    channel: C,
    payload: IpcChannelMap[C]['request'],
    frame = renderer.mainFrame
): Promise<IpcResult<IpcChannelMap[C]['response']>> {
    const handler = fixtures.handlers.get(channel)
    if (!handler) throw new Error('missing production handler')
    return Reflect.apply(handler, undefined, [{ sender: renderer, senderFrame: frame }, payload])
}
function record(id: number): RoutingHistoryRecord {
    return {
        schema: 1,
        request: { originNodeId: 'node', proxyRunId: 'run', requestId: String(id) },
        workload: null,
        seq: 1,
        requested: { model: '', tier: '', mode: 'other' },
        createdAt: id,
        endedAt: id,
        state: 'failed',
        reason: 'request-invalid',
        status: null,
        served: null,
        recoverable: false,
        spend: {
            currency: 'USD',
            ceilingNanoUsd: null,
            grantedNanoUsd: null,
            heldNanoUsd: null,
            chargedNanoUsd: null,
            accounting: 'none'
        },
        droppedAttempts: 0,
        attempts: []
    }
}

beforeEach(() => {
    rendererFrames.clear()
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
        rendererFrames.set(++nextFrame, callback)
        return nextFrame
    })
    vi.stubGlobal('cancelAnimationFrame', (id: number) => {
        rendererFrames.delete(id)
    })
    fixtures.windows.length = 0
    fixtures.handlers.clear()
    fixtures.wanted.mockReset()
    registerServiceBridgeIpc()
    getRoutingDelivery().reset()
    getRoutingDelivery().initialResult(new TextEncoder().encode('{"requests":[]}'))
})
afterEach(() => {
    for (const window of fixtures.windows) {
        window.destroy()
        // Extra synthetic registrations in the no-subscriber-cap case have no
        // application window lifecycle; retire their fixture owner explicitly.
        getRoutingDelivery().contextDestroyed(window)
    }
    vi.unstubAllGlobals()
})

class HistoryPeer implements Pick<
    JsonRpcSocketClient,
    'callHistory' | 'cancelHistory' | 'setHistoryHandlers' | 'historyWritable'
> {
    readonly calls: RoutingHistoryMethod[] = []
    busy = true
    unsubscribeBusy = false
    blockedWrite = false
    handlers: Parameters<JsonRpcSocketClient['setHistoryHandlers']>[0] = null
    get historyWritable(): boolean {
        return !this.blockedWrite
    }
    cancelHistory(): void {
        /* No pending response in this peer. */
    }
    setHistoryHandlers(handlers: Parameters<JsonRpcSocketClient['setHistoryHandlers']>[0]): void {
        this.handlers = handlers
    }
    async callHistory(
        method: RoutingHistoryMethod,
        consume: (raw: Uint8Array) => boolean
    ): Promise<void> {
        this.calls.push(method)
        if (
            this.blockedWrite ||
            (method === 'routing:unsubscribe' ? this.unsubscribeBusy : this.busy)
        )
            throw new JsonRpcResponseError(-32096, 'routing history busy; retry')
        if (!consume(new TextEncoder().encode('{"requests":[]}')))
            throw new Error('history-invalid')
    }
}
class HeldHistoryPeer extends HistoryPeer {
    hold = false
    completion: (() => boolean) | null = null
    private rejectHeld: ((error: Error) => void) | null = null
    override cancelHistory(): void {
        this.rejectHeld?.(new Error('history-invalid'))
        this.rejectHeld = null
    }
    override callHistory(
        method: RoutingHistoryMethod,
        consume: (raw: Uint8Array) => boolean
    ): Promise<void> {
        if (!this.hold) return super.callHistory(method, consume)
        this.calls.push(method)
        return new Promise((resolve, reject) => {
            this.rejectHeld = reject
            this.completion = () => {
                const applied = consume(new TextEncoder().encode('{"requests":[]}'))
                if (applied) resolve()
                else reject(new Error('history-invalid'))
                return applied
            }
        })
    }
}
async function settle(): Promise<void> {
    for (let i = 0; i < 24; i++) await Promise.resolve()
}

describe('production history subscription retry owner', () => {
    it('cancels an obsolete cleanup retry on new demand and completes cleanup before resubscribing', async () => {
        vi.useFakeTimers()
        const owner = new RoutingHistorySubscription(),
            peer = new HistoryPeer()
        try {
            peer.busy = false
            owner.connect(peer)
            owner.setWanted(true)
            await settle()
            peer.unsubscribeBusy = true
            owner.setWanted(false)
            await settle()
            expect(vi.getTimerCount()).toBe(1)
            peer.unsubscribeBusy = false
            owner.setWanted(true)
            await settle()
            expect(peer.calls).toEqual([
                'routing:subscribe',
                'routing:unsubscribe',
                'routing:unsubscribe',
                'routing:subscribe'
            ])
            expect(vi.getTimerCount()).toBe(0)
            await vi.advanceTimersByTimeAsync(5000)
            expect(peer.calls).toHaveLength(4)
        } finally {
            owner.disconnected()
            vi.useRealTimers()
        }
    })

    it('cancels a coalesced refetch timer on exhaustion and resumes only after a new connection fence', async () => {
        vi.useFakeTimers()
        class ExhaustedPeer extends HistoryPeer {
            override callHistory(method: RoutingHistoryMethod): Promise<void> {
                this.calls.push(method)
                this.handlers?.invalid()
                return Promise.reject(new Error('history-exhausted'))
            }
        }
        const owner = new RoutingHistorySubscription(),
            peer = new ExhaustedPeer(),
            replacement = new HistoryPeer()
        try {
            owner.connect(peer)
            owner.setWanted(true)
            await settle()
            expect(vi.getTimerCount()).toBe(0)
            peer.handlers?.writable()
            await vi.advanceTimersByTimeAsync(5000)
            expect(peer.calls).toEqual(['routing:subscribe'])
            owner.disconnected()
            replacement.busy = false
            owner.connect(replacement)
            await settle()
            expect(replacement.calls).toEqual(['routing:subscribe'])
        } finally {
            owner.disconnected()
            vi.useRealTimers()
        }
    })

    it('retargets its one timer to busy unsubscribe completion even with no future writable callback', async () => {
        vi.useFakeTimers()
        const owner = new RoutingHistorySubscription(),
            peer = new HistoryPeer()
        try {
            peer.busy = false
            owner.connect(peer)
            owner.setWanted(true)
            await settle()
            peer.unsubscribeBusy = true
            owner.setWanted(false)
            await settle()
            expect(peer.calls).toEqual(['routing:subscribe', 'routing:unsubscribe'])
            expect(vi.getTimerCount()).toBe(1)
            await vi.advanceTimersByTimeAsync(250)
            expect(peer.calls).toHaveLength(3)
            await vi.advanceTimersByTimeAsync(500)
            expect(peer.calls).toHaveLength(4)
            expect(vi.getTimerCount()).toBe(1)
            peer.unsubscribeBusy = false
            await vi.advanceTimersByTimeAsync(1000)
            expect(peer.calls).toEqual([
                'routing:subscribe',
                'routing:unsubscribe',
                'routing:unsubscribe',
                'routing:unsubscribe',
                'routing:unsubscribe'
            ])
            expect(vi.getTimerCount()).toBe(0)
            await vi.advanceTimersByTimeAsync(5000)
            expect(peer.calls).toHaveLength(5)
        } finally {
            owner.disconnected()
            vi.useRealTimers()
        }
    })

    it('retains native unsubscribe completion across a broker-only source reset', async () => {
        vi.useFakeTimers()
        const owner = new RoutingHistorySubscription(),
            peer = new HistoryPeer()
        try {
            peer.busy = false
            owner.connect(peer)
            owner.setWanted(true)
            await settle()
            peer.unsubscribeBusy = true
            owner.setWanted(false)
            await settle()
            owner.sourceReset()
            await settle()
            expect(peer.calls.at(-1)).toBe('routing:unsubscribe')
            expect(vi.getTimerCount()).toBe(1)
            peer.unsubscribeBusy = false
            owner.connect(peer)
            await vi.advanceTimersByTimeAsync(250)
            expect(peer.calls.at(-1)).toBe('routing:unsubscribe')
            expect(vi.getTimerCount()).toBe(0)
        } finally {
            owner.disconnected()
            vi.useRealTimers()
        }
    })

    it('waits for the actual active control writer before admitting unsubscribe', async () => {
        vi.useFakeTimers()
        const owner = new RoutingHistorySubscription(),
            peer = new HistoryPeer()
        try {
            peer.busy = false
            owner.connect(peer)
            owner.setWanted(true)
            await settle()
            peer.blockedWrite = true
            owner.setWanted(false)
            await settle()
            expect(peer.calls).toEqual(['routing:subscribe'])
            expect(vi.getTimerCount()).toBe(0)
            await vi.advanceTimersByTimeAsync(5000)
            expect(peer.calls).toHaveLength(1)
            peer.blockedWrite = false
            peer.handlers?.writable()
            await settle()
            expect(peer.calls).toEqual(['routing:subscribe', 'routing:unsubscribe'])
            expect(vi.getTimerCount()).toBe(0)
        } finally {
            owner.disconnected()
            vi.useRealTimers()
        }
    })

    it('fences an obsolete unsubscribe completion after new demand without clearing the new subscription', async () => {
        vi.useFakeTimers()
        const applied = vi.fn<(raw: Uint8Array, initial: boolean) => undefined>(() => undefined)
        const owner = new RoutingHistorySubscription(applied),
            peer = new HeldHistoryPeer()
        try {
            peer.busy = false
            owner.connect(peer)
            owner.setWanted(true)
            await settle()
            peer.hold = true
            owner.setWanted(false)
            await settle()
            const old = peer.completion
            if (!old) throw new Error('missing old completion')
            peer.hold = false
            owner.setWanted(true)
            await settle()
            expect(old()).toBe(false)
            await settle()
            expect(peer.calls).toEqual([
                'routing:subscribe',
                'routing:unsubscribe',
                'routing:unsubscribe',
                'routing:subscribe'
            ])
            expect(applied).toHaveBeenCalledTimes(2)
            expect(vi.getTimerCount()).toBe(0)
        } finally {
            owner.disconnected()
            vi.useRealTimers()
        }
    })

    it('fences old completions and retry handlers across actual connection destruction', async () => {
        vi.useFakeTimers()
        const applied = vi.fn<(raw: Uint8Array, initial: boolean) => undefined>(() => undefined)
        const owner = new RoutingHistorySubscription(applied),
            first = new HeldHistoryPeer(),
            second = new HistoryPeer()
        try {
            first.hold = true
            second.busy = false
            owner.connect(first)
            owner.setWanted(true)
            await settle()
            const old = first.completion,
                handlers = first.handlers
            if (!old) throw new Error('missing old completion')
            owner.disconnected()
            owner.connect(second)
            await settle()
            expect(old()).toBe(false)
            handlers?.invalid()
            handlers?.writable()
            await vi.advanceTimersByTimeAsync(5000)
            expect(second.calls).toEqual(['routing:subscribe'])
            expect(applied).toHaveBeenCalledOnce()
            expect(vi.getTimerCount()).toBe(0)
        } finally {
            owner.disconnected()
            vi.useRealTimers()
        }
    })

    it('has one 250/500/1000 then 1000 ms retry and cancels it on the last unsubscribe', async () => {
        vi.useFakeTimers()
        const owner = new RoutingHistorySubscription(),
            peer = new HistoryPeer()
        try {
            owner.connect(peer)
            owner.setWanted(true)
            await settle()
            expect(peer.calls).toEqual(['routing:subscribe'])
            expect(vi.getTimerCount()).toBe(1)
            await vi.advanceTimersByTimeAsync(250)
            expect(peer.calls).toHaveLength(2)
            await vi.advanceTimersByTimeAsync(500)
            expect(peer.calls).toHaveLength(3)
            await vi.advanceTimersByTimeAsync(1000)
            expect(peer.calls).toHaveLength(4)
            expect(vi.getTimerCount()).toBe(1)
            owner.setWanted(false)
            await settle()
            expect(peer.calls.at(-1)).toBe('routing:unsubscribe')
            expect(vi.getTimerCount()).toBe(0)
            await vi.advanceTimersByTimeAsync(5000)
            expect(peer.calls).toHaveLength(5)
        } finally {
            owner.disconnected()
            vi.useRealTimers()
        }
    })

    it('fences a disconnected callback and waits for actual control-write completion without a retry timer', async () => {
        vi.useFakeTimers()
        const owner = new RoutingHistorySubscription(),
            peer = new HistoryPeer()
        try {
            peer.blockedWrite = true
            owner.connect(peer)
            owner.setWanted(true)
            await settle()
            expect(vi.getTimerCount()).toBe(0)
            const old = peer.handlers
            owner.disconnected()
            old?.invalid()
            old?.writable()
            await vi.advanceTimersByTimeAsync(5000)
            expect(peer.calls).toHaveLength(0)
            expect(peer.handlers).toBeNull()
        } finally {
            owner.disconnected()
            vi.useRealTimers()
        }
    })
})

describe('production history sender IPC and broadcaster', () => {
    it('commits a corpus-derived model-less initial and update through production IPC', async () => {
        const corpus: { cases: { record: RoutingHistoryRecord }[] } = JSON.parse(
            fs.readFileSync(
                path.resolve('../services/shared/nodepolicy/testdata/routing-request.json'),
                'utf8'
            )
        )
        const row = corpus.cases[0].record
        row.requested.model = ''
        for (const attempt of row.attempts) attempt.destination.model = null
        if (row.served) row.served.model = null
        getRoutingDelivery().initialResult(
            new TextEncoder().encode(JSON.stringify({ requests: [row] }))
        )
        const a = sender(),
            transport = isolatedTransport(a),
            history = createRoutingHistory(transport)
        const initial = history.getInitial()
        await settle()
        await commitInitial(a, initial)
        await settle()
        expect(
            history.consumeCurrent(0, value => {
                expect(value).toEqual(row)
                return undefined
            })
        ).toBe(true)
        row.seq++
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(row)))
        await receiveReady(a)
        runFrame()
        await settle()
        expect(
            history.consumeCurrent(0, value => {
                expect(value).toEqual(row)
                return undefined
            })
        ).toBe(true)
        expect(a.ipc.maximumInFlight).toBe(1)
        expect(getRoutingDelivery().capacity.initialLeases).toBe(0)
        history.destroy()
        await settle()
        await receiveReady(a)
        transport.destroy()
    })

    it('loads a full retained window through one-record IPC loans and one scalar end', async () => {
        const a = sender(),
            transport = isolatedTransport(a),
            history = createRoutingHistory(transport)
        for (let id = 1; id <= 256; id++)
            applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(id))))
        const initial = history.getInitial()
        await settle()
        await commitInitial(a, initial)
        await settle()
        expect(history.size).toBe(256)
        expect(a.ipc.maximumInFlight).toBe(1)
        expect(a.ipc.maximumPacketBytes).toBeLessThanOrEqual(32752)
        expect(getRoutingDelivery().capacity.windowBytes).toBe(3 * 256 * 65536)
        expect(getRoutingDelivery().capacity.initialLeases).toBe(0)
        history.consumeCurrent(0, row => {
            expect(row.request.requestId).toBe('256')
            return undefined
        })
        history.destroy()
        await settle()
        await receiveReady(a)
        transport.destroy()
    })

    it('stages one frozen initial record per loan and publishes a coherent baseline only after scalar end', async () => {
        const a = sender(),
            transport = isolatedTransport(a),
            history = createRoutingHistory(transport)
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(1))))
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(2))))
        const initial = history.getInitial(),
            finished = vi.fn<() => void>()
        void initial.then(finished)
        await settle()
        await receiveReady(a)
        expect(rendererFrames.size).toBe(0)
        expect(history.size).toBe(0)
        expect(history.loading).toBe(true)
        expect(getRoutingDelivery().capacity.initialLeases).toBe(1)
        const newer = record(1)
        newer.seq = 2
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(newer)))
        await receiveReady(a)
        expect(rendererFrames.size).toBe(0)
        expect(history.size).toBe(0)
        expect(finished).not.toHaveBeenCalled()
        expect(getRoutingDelivery().capacity.initialLeases).toBe(1)
        await receiveReady(a)
        expect(rendererFrames.size).toBe(1)
        expect(history.size).toBe(0)
        runFrame()
        await initial
        await settle()
        expect(history.size).toBe(2)
        expect(getRoutingDelivery().capacity.initialLeases).toBe(0)
        history.consumeCurrent(1, row => {
            expect(row.seq).toBe(1)
            return undefined
        })
        await receiveReady(a)
        runFrame()
        await settle()
        history.consumeCurrent(1, row => {
            expect(row.seq).toBe(2)
            return undefined
        })
        expect(a.ipc.maximumInFlight).toBe(1)
        history.destroy()
        await settle()
        await receiveReady(a)
        transport.destroy()
    })

    it('keeps the global initial reservation between record acknowledgements and through a retired scalar end', async () => {
        const a = sender(),
            b = sender(),
            transportA = isolatedTransport(a),
            transportB = isolatedTransport(b)
        const historyA = createRoutingHistory(transportA),
            historyB = createRoutingHistory(transportB)
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(1))))
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(2))))
        const baselineB = historyB.getInitial()
        await settle()
        await commitInitial(b, baselineB)
        const baselineA = historyA.getInitial()
        const retiredBaseline = expect(baselineA).rejects.toThrow('history-retired')
        await settle()
        await receiveReady(a)
        expect(a.ipc.inFlight).toBe(0)
        expect(historyA.size).toBe(0)
        expect(getRoutingDelivery().capacity.initialLeases).toBe(1)
        a.ipc.pauseData = true
        const retired = historyA.unsubscribe()
        await settle()
        await retiredBaseline
        await receiveReady(a)
        expect(getRoutingDelivery().capacity.initialLeases).toBe(1)
        expect(a.ipc.inFlight).toBe(1)
        await expect(transportB.invoke('routing:active', {})).resolves.toEqual({ requests: [] })
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(3))))
        await receiveReady(b)
        runFrame()
        await settle()
        expect(historyB.size).toBe(3)
        a.ipc.resume()
        await retired
        await settle()
        expect(historyA.size).toBe(0)
        expect(getRoutingDelivery().capacity.initialLeases).toBe(0)
        historyA.destroy()
        historyB.destroy()
        await settle()
        await receiveReady(b)
        transportA.destroy()
        transportB.destroy()
    })

    it('keeps a stopped IPC initial pinned while native unsubscribe retries and ordinary traffic progresses', async () => {
        vi.useFakeTimers()
        const subscription = new RoutingHistorySubscription(),
            peer = new HistoryPeer()
        peer.busy = false
        subscription.connect(peer)
        fixtures.wanted.mockImplementation(active => subscription.setWanted(active))
        const a = sender(),
            b = sender(),
            transportA = isolatedTransport(a),
            transportB = isolatedTransport(b)
        owner(b).emit('ready-to-show')
        const history = createRoutingHistory(transportA)
        try {
            applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(1))))
            const initial = history.getInitial()
            const retiredInitial = expect(initial).rejects.toThrow('history-retired')
            await settle()
            await receiveReady(a)
            expect(getRoutingDelivery().capacity.initialLeases).toBe(1)
            expect(a.ipc.inFlight).toBe(0)
            a.ipc.pauseData = true
            peer.unsubscribeBusy = true
            const retired = history.unsubscribe()
            await settle()
            await retiredInitial
            await receiveReady(a)
            expect(a.ipc.inFlight).toBe(1)
            expect(vi.getTimerCount()).toBe(1)
            await expect(transportB.invoke('routing:active', {})).resolves.toEqual({ requests: [] })
            emitBridgePush('state:request-refresh', undefined)
            expect(b.ordinary.at(-1)?.channel).toBe('state:request-refresh')
            peer.unsubscribeBusy = false
            await vi.advanceTimersByTimeAsync(250)
            expect(peer.calls.at(-1)).toBe('routing:unsubscribe')
            expect(vi.getTimerCount()).toBe(0)
            expect(getRoutingDelivery().capacity.initialLeases).toBe(1)
            a.ipc.resume()
            await settle()
            await retired
            expect(getRoutingDelivery().capacity.initialLeases).toBe(0)
        } finally {
            history.destroy()
            await settle()
            transportA.destroy()
            transportB.destroy()
            subscription.disconnected()
            vi.useRealTimers()
        }
    })

    it('drains disposal retirement before detaching and permits a replacement owner in the same context', async () => {
        const a = sender(),
            transport = isolatedTransport(a),
            history = createRoutingHistory(transport)
        const first = history.getInitial()
        await settle()
        await commitInitial(a, first)
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(1))))
        history.destroy()
        await settle()
        await receiveReady(a)
        const replacement = createRoutingHistory(transport)
        const initialized = replacement.getInitial()
        await settle()
        await commitInitial(a, initialized)
        expect(replacement.size).toBe(1)
        expect(a.ipc.maximumInFlight).toBe(1)
        replacement.destroy()
        await settle()
        await receiveReady(a)
        transport.destroy()
    })

    it('drains a delivered ready token during unsubscribe and can subscribe again in the same context', async () => {
        const a = sender(),
            transport = isolatedTransport(a),
            history = createRoutingHistory(transport)
        const first = history.getInitial()
        await settle()
        await commitInitial(a, first)
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(1))))
        expect(a.ready).toHaveLength(1)
        const retired = history.unsubscribe()
        await settle()
        // This token was posted before unsubscribe, but the renderer was paused
        // before its native callback. Retirement must drain it without data.
        await receiveReady(a)
        await retired
        const replacement = history.getInitial()
        await settle()
        await commitInitial(a, replacement)
        expect(history.size).toBe(1)
        expect(a.ipc.maximumInFlight).toBe(1)
        history.destroy()
        await settle()
        transport.destroy()
    })

    it('merges delayed initial state with accepted updates before the production renderer consumes it', async () => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
        const row = record(1)
        row.seq = 2
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(row)))
        row.seq = 1
        getRoutingDelivery().initialResult(
            new TextEncoder().encode(JSON.stringify({ requests: [row] }))
        )
        row.seq = 3
        row.state = 'routing'
        row.endedAt = null
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(row)))
        const a = sender(),
            transport = isolatedTransport(a),
            history = createRoutingHistory(transport)
        const initial = history.getInitial()
        await settle()
        await commitInitial(a, initial)
        history.consumeCurrent(0, current => {
            expect(current.seq).toBe(2)
            expect(current.state).toBe('failed')
            return undefined
        })
        getRoutingDelivery().setLoading(true)
        await receiveReady(a)
        expect(history.loading).toBe(true)
        expect(history.size).toBe(1)
        getRoutingDelivery().initialResult(new TextEncoder().encode('{"requests":[]}'))
        await vi.advanceTimersByTimeAsync(250)
        expect(history.loading).toBe(false)
        expect(history.size).toBe(1)
        history.destroy()
        await settle()
        transport.destroy()
        vi.useRealTimers()
    })

    it('rejects an old acknowledgement after the production crash callback and a replacement generation', async () => {
        const a = sender(),
            delivery = getRoutingDelivery()
        const oldGeneration = delivery.subscribe(owner(a))
        a.ready.shift()
        const old = delivery.pull(owner(a), { generation: oldGeneration, initial: true })
        if (old.state !== 'end') throw new Error('expected old initial end')
        a.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 1 })
        const generation = delivery.subscribe(owner(a))
        a.ready.shift()
        const replacement = delivery.pull(owner(a), { generation, initial: true })
        if (replacement.state !== 'end') throw new Error('expected replacement initial end')
        expect(
            await invoke(a, 'routing-history:consumed', {
                generation: oldGeneration,
                lease: old.lease
            })
        ).toMatchObject({ success: true, data: { state: 'stale' } })
        expect(delivery.capacity.initialLeases).toBe(1)
        expect(delivery.consumed(owner(a), { generation, lease: replacement.lease }).state).toBe(
            'accepted'
        )
    })

    it('keeps B and ordinary invokes progressing while A is stopped before preload receipt, then converges A', async () => {
        const a = sender(),
            b = sender(),
            transportA = isolatedTransport(a),
            transportB = isolatedTransport(b)
        const historyA = createRoutingHistory(transportA),
            historyB = createRoutingHistory(transportB)
        const initialB = historyB.getInitial()
        await settle()
        await commitInitial(b, initialB)
        a.ipc.pauseData = true
        const initialA = historyA.getInitial()
        await settle()
        await receiveReady(a)
        expect(a.ipc.inFlight).toBe(1)
        expect(rendererFrames.size).toBe(0)
        for (let id = 1; id <= 300; id++)
            applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(id))))
        expect(a.ready.length).toBeLessThanOrEqual(1)
        expect(b.ready).toHaveLength(1)
        await expect(transportB.invoke('routing:active', {})).resolves.toEqual({ requests: [] })
        emitBridgePush('state:request-refresh', undefined)
        expect(b.ordinary.at(-1)?.channel).toBe('state:request-refresh')
        for (let turn = 0; turn < 300 && b.ready.length; turn++) {
            await receiveReady(b)
            if (rendererFrames.size) runFrame()
            await settle()
        }
        expect(historyB.size).toBe(256)
        expect(a.ipc.maximumInFlight).toBe(1)
        expect(b.ipc.maximumInFlight).toBe(1)
        a.ipc.resume()
        await settle()
        runFrame()
        await initialA
        for (let turn = 0; turn < 300 && a.ready.length; turn++) {
            await receiveReady(a)
            if (rendererFrames.size) runFrame()
            await settle()
        }
        expect(historyA.size).toBe(256)
        historyA.consumeCurrent(0, row => {
            expect(row.request.requestId).toBe('300')
            return undefined
        })
        historyA.destroy()
        historyB.destroy()
        await settle()
        transportA.destroy()
        transportB.destroy()
    })
    it('admits concurrent row loans beside the global initial while another renderer and ordinary RPC progress', async () => {
        const a = sender(),
            b = sender(),
            c = new TestWindow().contents,
            d = new TestWindow().contents
        const transportA = isolatedTransport(a),
            transportB = isolatedTransport(b)
        const transportC = isolatedTransport(c),
            transportD = isolatedTransport(d)
        const historyA = createRoutingHistory(transportA),
            historyB = createRoutingHistory(transportB)
        const historyC = createRoutingHistory(transportC),
            historyD = createRoutingHistory(transportD)
        const initialA = historyA.getInitial()
        await settle()
        await commitInitial(a, initialA)
        const initialB = historyB.getInitial()
        await settle()
        await commitInitial(b, initialB)
        const initialC = historyC.getInitial()
        await settle()
        await commitInitial(c, initialC)
        a.ipc.pauseData = true
        c.ipc.pauseData = true
        d.ipc.pauseData = true
        applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(1))))
        await receiveReady(a)
        await receiveReady(c)
        const initialD = historyD.getInitial()
        await settle()
        await receiveReady(d)
        expect([a.ipc.inFlight, c.ipc.inFlight, d.ipc.inFlight]).toEqual([1, 1, 1])
        expect(getRoutingDelivery().capacity.initialLeases).toBe(1)
        await expect(transportB.invoke('routing:active', {})).resolves.toEqual({ requests: [] })
        await receiveReady(b)
        runFrame()
        await settle()
        expect(historyB.size).toBe(1)
        expect([a.ipc.inFlight, c.ipc.inFlight, d.ipc.inFlight]).toEqual([1, 1, 1])
        a.ipc.resume()
        await settle()
        runFrame()
        await settle()
        c.ipc.resume()
        await settle()
        runFrame()
        await settle()
        d.ipc.resume()
        await settle()
        await commitInitial(d, initialD)
        expect([historyA.size, historyC.size, historyD.size]).toEqual([1, 1, 1])
        expect([
            a.ipc.maximumInFlight,
            b.ipc.maximumInFlight,
            c.ipc.maximumInFlight,
            d.ipc.maximumInFlight
        ]).toEqual([1, 1, 1, 1])
        expect(getRoutingDelivery().capacity.initialLeases).toBe(0)
        historyA.destroy()
        historyB.destroy()
        historyC.destroy()
        historyD.destroy()
        await settle()
        await receiveReady(a)
        await receiveReady(b)
        await receiveReady(c)
        await receiveReady(d)
        transportA.destroy()
        transportB.destroy()
        transportC.destroy()
        transportD.destroy()
    })

    it('uses the actual event sender and keeps the initial lease until its exact acknowledgement', async () => {
        const a = sender(),
            b = sender()
        const generationA = getRoutingDelivery().subscribe(owner(a)),
            generationB = getRoutingDelivery().subscribe(owner(b))
        a.ready.shift()
        b.ready.shift()
        const first = getRoutingDelivery().pull(owner(a), {
            generation: generationA,
            initial: true
        })
        expect(first.state).toBe('end')
        if (first.state !== 'end') throw new Error('expected initial end')
        expect(
            getRoutingDelivery().pull(owner(b), { generation: generationB, initial: true }).state
        ).toBe('busy')
        expect(
            await invoke(b, 'routing-history:consumed', {
                generation: generationA,
                lease: first.lease
            })
        ).toMatchObject({ success: true, data: { state: 'stale' } })
        expect(
            await invoke(
                a,
                'routing-history:consumed',
                { generation: generationA, lease: first.lease },
                {}
            )
        ).toMatchObject({ success: false, error: 'history-invalid' })
        expect(
            await invoke(a, 'routing-history:consumed', {
                generation: generationA + 1,
                lease: first.lease
            })
        ).toMatchObject({ success: true, data: { state: 'stale' } })
        expect(
            await invoke(a, 'routing-history:consumed', {
                generation: generationA,
                lease: first.lease + 1
            })
        ).toMatchObject({ success: true, data: { state: 'stale' } })
        expect(getRoutingDelivery().capacity.initialLeases).toBe(1)
        expect(
            await invoke(a, 'routing-history:pull', { generation: generationA, initial: true })
        ).toMatchObject({ success: true, data: { state: 'busy' } })
        expect(
            await invoke(a, 'service-bridge:invoke', { channel: 'routing:active', payload: {} })
        ).toMatchObject({ success: true, data: { requests: [] } })
        expect(
            await invoke(a, 'routing-history:consumed', {
                generation: generationA,
                lease: first.lease
            })
        ).toMatchObject({ success: true, data: { state: 'accepted' } })
        expect(getRoutingDelivery().capacity.initialLeases).toBe(0)
        expect(
            await invoke(a, 'routing-history:consumed', {
                generation: generationA,
                lease: first.lease
            })
        ).toMatchObject({ success: true, data: { state: 'stale' } })
    })

    it('coalesces stopped A across more than a window while B and ordinary pushes remain usable', () => {
        const a = sender(),
            b = sender(),
            delivery = getRoutingDelivery()
        const ga = delivery.subscribe(owner(a)),
            gb = delivery.subscribe(owner(b))
        a.ready.shift()
        b.ready.shift()
        const baselineB = delivery.pull(owner(b), { generation: gb, initial: true })
        if (baselineB.state !== 'end') throw new Error('expected B baseline end')
        delivery.consumed(owner(b), { generation: gb, lease: baselineB.lease })
        const pinnedA = delivery.pull(owner(a), { generation: ga, initial: true })
        if (pinnedA.state !== 'end') throw new Error('expected A baseline end')
        for (let id = 1; id <= 300; id++)
            applyRoutingHistoryChanged(new TextEncoder().encode(JSON.stringify(record(id))))
        expect(a.ready.length).toBeLessThanOrEqual(1)
        expect(b.ready).toHaveLength(1)
        b.ready.shift()
        expect(delivery.pull(owner(b), { generation: gb, initial: false }).state).toBe('data')
        expect(delivery.pull(owner(a), { generation: ga, initial: true }).state).toBe('busy')
        emitBridgePush('state:request-refresh', undefined)
        emitBridgePush('routing:changed', record(301))
        expect(a.ordinary.map(message => message.channel)).toEqual(['state:request-refresh'])
        expect(b.ordinary.map(message => message.channel)).toEqual(['state:request-refresh'])
        expect(delivery.capacity.windowBytes).toBe(3 * 256 * 65536)
        delivery.consumed(owner(a), { generation: ga, lease: pinnedA.lease })
    })

    it('retires unsubscribe and disconnect without reclaiming an outstanding transfer', async () => {
        const a = sender(),
            b = sender(),
            delivery = getRoutingDelivery()
        const ga = delivery.subscribe(owner(a)),
            gb = delivery.subscribe(owner(b))
        a.ready.shift()
        b.ready.shift()
        const first = delivery.pull(owner(a), { generation: ga, initial: true })
        if (first.state !== 'end') throw new Error('expected initial end')
        await invoke(a, 'service-bridge:invoke', { channel: 'routing:unsubscribe' })
        delivery.reset()
        delivery.initialResult(new TextEncoder().encode('{"requests":[]}'))
        expect(delivery.pull(owner(b), { generation: gb, initial: true }).state).toBe('retired')
        expect(delivery.consumed(owner(a), { generation: ga, lease: first.lease }).state).toBe(
            'retired'
        )
        owner(a).destroy()
        expect(delivery.consumed(owner(a), { generation: ga, lease: first.lease }).state).toBe(
            'stale'
        )
        owner(b).destroy()
        expect(fixtures.wanted).toHaveBeenLastCalledWith(false)
    })

    it('adds no lifecycle listener or captured registration for history generations', () => {
        const a = sender(),
            delivery = getRoutingDelivery()
        for (let turn = 0; turn < 20; turn++) {
            delivery.subscribe(owner(a))
            a.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 1 })
        }
        delivery.subscribe(owner(a))
        expect(
            Object.keys(owner(a))
                .filter(key => key.startsWith('routing'))
                .sort()
        ).toEqual(['routingCursor', 'routingFlags', 'routingGeneration', 'routingLease'])
        expect(Object.keys(a).some(key => key.startsWith('routing'))).toBe(false)
        expect(a.listenerCount('did-navigate')).toBe(1)
        expect(a.listenerCount('render-process-gone')).toBe(1)
        expect(a.listenerCount('destroyed')).toBe(1)
    })

    it.each(['destroyed', 'render-process-gone', 'did-navigate'])(
        'releases a pinned transfer on the production %s callback',
        async event => {
            const a = sender(),
                b = sender(),
                delivery = getRoutingDelivery()
            const generation = delivery.subscribe(owner(a))
            a.ready.shift()
            const pinned = delivery.pull(owner(a), { generation, initial: true })
            if (pinned.state !== 'end') throw new Error('expected pinned initial')
            a.emit('did-navigate-in-page', {}, 'file:///synthetic/index.html#section', true)
            a.emit('did-frame-navigate', {}, 'file:///child.html', 200, 'OK', false)
            expect(delivery.capacity.initialLeases).toBe(1)
            if (event === 'destroyed') a.destroy()
            else if (event === 'render-process-gone')
                a.emit(event, {}, { reason: 'crashed', exitCode: 1 })
            else a.emit(event, {}, 'file:///replacement.html', 200, 'OK')
            expect(delivery.capacity.initialLeases).toBe(0)
            expect(delivery.consumed(owner(a), { generation, lease: pinned.lease }).state).toBe(
                'stale'
            )
            const next = delivery.subscribe(owner(b))
            b.ready.shift()
            expect(delivery.pull(owner(b), { generation: next, initial: true }).state).toBe('end')
            if (event !== 'destroyed') {
                const replacement = delivery.subscribe(owner(a))
                expect(replacement).toBeGreaterThan(generation)
                expect(
                    await invoke(a, 'routing-history:consumed', { generation, lease: pinned.lease })
                ).toMatchObject({ success: true, data: { state: 'stale' } })
            }
        }
    )

    it('does not release on canceled navigation or close intent, but releases at the production closed callback', () => {
        const a = sender(),
            b = sender(),
            delivery = getRoutingDelivery()
        const generation = delivery.subscribe(owner(a))
        a.ready.shift()
        const pinned = delivery.pull(owner(a), { generation, initial: true })
        if (pinned.state !== 'end') throw new Error('expected pinned initial end')
        const preventDefault = vi.fn<() => void>()
        a.emit('will-navigate', { preventDefault }, 'https://example.invalid/')
        owner(a).emit('close', { preventDefault })
        expect(preventDefault).toHaveBeenCalledOnce()
        expect(delivery.capacity.initialLeases).toBe(1)
        expect(() => owner(a).destroy()).not.toThrow()
        expect(delivery.capacity.initialLeases).toBe(0)
        expect(delivery.consumed(owner(a), { generation, lease: pinned.lease }).state).toBe('stale')
        expect(Object.keys(owner(a)).some(key => key.startsWith('routing'))).toBe(false)
        const next = delivery.subscribe(owner(b))
        b.ready.shift()
        expect(delivery.pull(owner(b), { generation: next, initial: true }).state).toBe('end')
    })

    it('does not treat a late finish-load event as destruction of a newly admitted context loan', () => {
        sender()
        const overview = sender(),
            delivery = getRoutingDelivery()
        const generation = delivery.subscribe(owner(overview))
        overview.ready.shift()
        const admitted = delivery.pull(owner(overview), { generation, initial: true })
        if (admitted.state !== 'end')
            throw new Error('expected admitted replacement-context end loan')
        // Renderer scripts can invoke before did-finish-load. That event alone
        // cannot distinguish this new loan from a prior document's borrower.
        overview.emit('did-finish-load')
        expect(overview.listenerCount('did-finish-load')).toBe(1)
        expect(delivery.capacity.initialLeases).toBe(1)
        expect(
            delivery.consumed(owner(overview), { generation, lease: admitted.lease }).state
        ).toBe('accepted')
    })
})
