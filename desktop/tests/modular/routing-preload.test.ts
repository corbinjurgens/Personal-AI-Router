// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ipcRenderer } from 'electron'
import type { IpcResult } from '@/shared/types/ipc-channels'
import { RoutingHistoryBank, type RoutingHistoryReady, type RoutingHistoryRecord, type RoutingHistoryCommit } from '@/shared/types/routing'

const boundary = vi.hoisted(() => {
    const listeners = new Map<string, (...args: Parameters<Parameters<typeof ipcRenderer.on>[1]>) => void>()
    return { invoke: vi.fn<typeof ipcRenderer.invoke>(), listeners }
})
vi.mock('electron', () => ({ ipcRenderer: {
    invoke: boundary.invoke,
    on: (channel: string, listener: Parameters<typeof ipcRenderer.on>[1]) => { boundary.listeners.set(channel, listener) },
    removeListener: (channel: string) => { boundary.listeners.delete(channel) }
} }))

import { createPairTransport } from '@/preload/api/pair-transport.api'
import { serviceTransport } from '@/ui/api/service-transport'
import { createRoutingHistory } from '@/ui/api/routing-history'
import { createPairApi } from '@/ui/api/pair-api'

const frames = new Map<number, FrameRequestCallback>()
let pairTransport = createPairTransport(ipcRenderer)
let nextFrame = 0
async function settle(): Promise<void> { for (let i = 0; i < 12; i++) await Promise.resolve() }
function ready(message: RoutingHistoryReady): void {
    const listener = boundary.listeners.get('routing-history:ready')
    if (!listener) throw new Error('missing production preload listener')
    Reflect.apply(listener, undefined, [{}, message])
}
function commitFrame(): void {
    const entry = frames.entries().next().value
    if (!entry) throw new Error('missing renderer frame')
    frames.delete(entry[0]); entry[1](0)
}
function fixture(): RoutingHistoryRecord {
    return {
        schema: 1, request: { originNodeId: 'node', proxyRunId: 'run', requestId: '1' },
        workload: null, seq: 1, requested: { model: '', tier: '', mode: 'other' },
        createdAt: 1, endedAt: 1, state: 'failed', reason: 'request-invalid', status: null,
        served: null, recoverable: false, spend: {
            currency: 'USD', ceilingNanoUsd: null, grantedNanoUsd: null,
            heldNanoUsd: null, chargedNanoUsd: null, accounting: 'none'
        }, droppedAttempts: 0, attempts: []
    }
}
function replies(records: RoutingHistoryRecord[] = []): void {
    const bank = new RoutingHistoryBank()
    for (const row of records) bank.apply(new TextEncoder().encode(JSON.stringify(row)))
    boundary.invoke.mockResolvedValueOnce({ success: true, data: { generation: 1 } })
    for (let index = 0; index < bank.size; index++) boundary.invoke
        .mockResolvedValueOnce({ success: true, data: { state: 'data', generation: 1, lease: index + 1, initial: true, index, bytes: bank.at(index).slice() } })
        .mockResolvedValueOnce({ success: true, data: { state: 'accepted', generation: 1 } })
    boundary.invoke.mockResolvedValueOnce({ success: true, data: { state: 'end', generation: 1, lease: bank.size + 1, count: bank.size, retired: false } })
        .mockResolvedValueOnce({ success: true, data: { state: 'accepted', generation: 1 } })
}
beforeEach(() => {
    boundary.invoke.mockReset(); boundary.listeners.clear(); frames.clear()
    pairTransport = createPairTransport(ipcRenderer)
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { frames.set(++nextFrame, callback); return nextFrame })
    vi.stubGlobal('cancelAnimationFrame', (id: number) => { frames.delete(id) })
})
afterEach(async () => { pairTransport.destroy(); await settle(); vi.unstubAllGlobals() })

describe('production preload and bounded renderer API', () => {
    it('returns undefined for history callbacks and rejects the Promise entry before native transmission', async () => {
        const complete = vi.fn<(result: IpcResult<RoutingHistoryReady>) => undefined>(() => undefined)
        boundary.invoke.mockResolvedValueOnce({ success: true, data: { generation: 7 } })
        expect(pairTransport.invokeHistory(complete, 'routing:subscribe')).toBeUndefined()
        expect(complete).not.toHaveBeenCalled()
        expect(() => pairTransport.invoke('routing:subscribe')).toThrow('history-invalid')
        await settle()
        expect(complete.mock.calls).toEqual([[{ success: true, data: { generation: 7 } }]])
        expect(boundary.invoke).toHaveBeenCalledTimes(1)
        const rejected = vi.fn<(result: IpcResult<{ requests: RoutingHistoryRecord[] }>) => undefined>(() => undefined)
        expect(pairTransport.invokeHistory(rejected, 'routing:get-initial')).toBeUndefined()
        expect(rejected.mock.calls).toEqual([[{ success: false, error: 'history-invalid' }]])
        expect(boundary.invoke).toHaveBeenCalledTimes(1)
    })

    it('keeps one native history completion while ordinary invokes proceed and clears failure before reuse', async () => {
        let fail: ((error: Error) => void) | null = null
        boundary.invoke.mockReturnValueOnce(new Promise((_resolve, reject) => { fail = reject }))
            .mockResolvedValueOnce({ success: true, data: { requests: [] } })
        const complete = vi.fn<(result: IpcResult<RoutingHistoryReady>) => undefined>(() => undefined)
        const busy = vi.fn<(result: IpcResult<null>) => undefined>(() => undefined)
        expect(pairTransport.invokeHistory(complete, 'routing:subscribe')).toBeUndefined()
        expect(pairTransport.invokeHistory(busy, 'routing:unsubscribe')).toBeUndefined()
        expect(busy.mock.calls).toEqual([[{ success: false, error: 'history-busy' }]])
        await expect(serviceTransport(pairTransport).invoke('routing:active', {})).resolves.toEqual({ requests: [] })
        if (!fail) throw new Error('missing native completion')
        Reflect.apply(fail, undefined, [new Error('private-native-error-sentinel')])
        fail = null
        await settle()
        expect(complete.mock.calls).toEqual([[{ success: false, error: 'history-invalid' }]])
        boundary.invoke.mockResolvedValueOnce({ success: true, data: null })
        expect(pairTransport.invokeHistory(busy, 'routing:unsubscribe')).toBeUndefined()
        await settle()
        expect(busy.mock.calls.at(-1)).toEqual([{ success: true, data: null }])
    })

    it('delivers retired and stale control results without Promise returns across repeated callback turns', async () => {
        const transport = serviceTransport(pairTransport)
        for (let turn = 1; turn <= 40; turn++) {
            boundary.invoke.mockResolvedValueOnce({ success: true, data: { state: 'retired', generation: turn } })
                .mockResolvedValueOnce({ success: true, data: { state: 'stale', generation: turn } })
            await expect(transport.invoke('routing-history:pull', { generation: turn, initial: false }))
                .resolves.toEqual({ state: 'retired', generation: turn })
            await expect(transport.invoke('routing-history:consumed', { generation: turn, lease: turn }))
                .resolves.toEqual({ state: 'stale', generation: turn })
        }
        expect(boundary.invoke).toHaveBeenCalledTimes(80)
    })

    it('completes callback failures and preserves ordinary errors without a Promise fallback', async () => {
        const transport = serviceTransport(pairTransport)
        boundary.invoke.mockResolvedValueOnce({ success: false, error: 'history-invalid', code: -32096 })
            .mockImplementationOnce(() => { throw new Error('private-synchronous-native-sentinel') })
            .mockResolvedValueOnce({ success: false, error: 'ordinary-refusal', code: -32012 })
        await expect(transport.invoke('routing:subscribe')).rejects.toMatchObject({ message: 'history-invalid', code: -32096 })
        await expect(transport.invoke('routing:unsubscribe')).rejects.toMatchObject({ message: 'history-invalid' })
        await expect(transport.invoke('routing:active', {})).rejects.toMatchObject({ message: 'ordinary-refusal', code: -32012 })
        expect(boundary.invoke).toHaveBeenCalledTimes(3)
    })

    it('rejects reentrant history admission until the synchronous callback returns', async () => {
        const nested = vi.fn<(result: IpcResult<null>) => undefined>(() => undefined)
        boundary.invoke.mockResolvedValueOnce({ success: true, data: { generation: 1 } })
        pairTransport.invokeHistory(() => {
            expect(pairTransport.invokeHistory(nested, 'routing:unsubscribe')).toBeUndefined()
            return undefined
        }, 'routing:subscribe')
        await settle()
        expect(nested.mock.calls).toEqual([[{ success: false, error: 'history-busy' }]])
        expect(boundary.invoke).toHaveBeenCalledTimes(1)
    })

    it('detaches completion and reports failure even when a scalar control listener throws', async () => {
        const remove = pairTransport.subscribePush('routing-history:committed', () => { throw new Error('listener-failed') })
        const transport = serviceTransport(pairTransport)
        boundary.invoke.mockResolvedValueOnce({ success: true, data: null })
            .mockResolvedValueOnce({ success: true, data: { generation: 3 } })
        await expect(transport.invoke('routing:unsubscribe')).rejects.toThrow('history-invalid')
        remove()
        await expect(transport.invoke('routing:subscribe')).resolves.toEqual({ generation: 3 })
        expect(boundary.invoke).toHaveBeenCalledTimes(2)
    })

    it('defers destroy cleanup until actual native completion and returns only undefined', async () => {
        let finish: ((result: IpcResult<RoutingHistoryReady>) => void) | null = null
        boundary.invoke.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
            .mockResolvedValueOnce({ success: true, data: null })
        let completions = 0
        pairTransport.invokeHistory(() => { completions++; return undefined }, 'routing:subscribe')
        expect(pairTransport.destroy()).toBeUndefined()
        expect(pairTransport.destroy()).toBeUndefined()
        expect(boundary.invoke).toHaveBeenCalledTimes(1)
        if (!finish) throw new Error('missing native completion')
        Reflect.apply(finish, undefined, [{ success: true, data: { generation: 9 } }])
        finish = null
        await settle()
        expect(completions).toBe(1)
        expect(boundary.invoke.mock.calls).toEqual([
            ['service-bridge:invoke', { channel: 'routing:subscribe', payload: undefined }],
            ['service-bridge:invoke', { channel: 'routing:unsubscribe', payload: undefined }]
        ])
    })

    it('keeps destroy cleanup behind an admitted packet until its validated consumed result', async () => {
        const bank = new RoutingHistoryBank(), transport = serviceTransport(pairTransport)
        bank.apply(new TextEncoder().encode(JSON.stringify(fixture())))
        boundary.invoke.mockResolvedValueOnce({ success: true, data: { generation: 1 } })
            .mockResolvedValueOnce({ success: true, data: { state: 'data', generation: 1, lease: 2, initial: true, index: 0, bytes: bank.at(0).slice() } })
            .mockResolvedValueOnce({ success: true, data: { state: 'stale', generation: 1 } })
            .mockResolvedValueOnce({ success: true, data: { state: 'accepted', generation: 1 } })
            .mockResolvedValueOnce({ success: true, data: null })
        await transport.invoke('routing:subscribe')
        await expect(transport.invoke('routing-history:pull', { generation: 1, initial: true })).resolves.toMatchObject({ state: 'data', lease: 2 })
        expect(pairTransport.destroy()).toBeUndefined()
        expect(boundary.invoke).toHaveBeenCalledTimes(2)
        await transport.invoke('routing-history:consumed', { generation: 1, lease: 1 })
        expect(boundary.invoke).toHaveBeenCalledTimes(3)
        await transport.invoke('routing-history:consumed', { generation: 1, lease: 2 })
        await settle()
        expect(boundary.invoke.mock.calls.map(([channel]) => channel)).toEqual([
            'service-bridge:invoke', 'routing-history:pull', 'routing-history:consumed', 'routing-history:consumed', 'service-bridge:invoke'
        ])
    })

    it('rejects a mismatching scalar end without acknowledging or exposing a partial baseline', async () => {
        const bank = new RoutingHistoryBank()
        bank.apply(new TextEncoder().encode(JSON.stringify(fixture())))
        boundary.invoke.mockResolvedValueOnce({ success: true, data: { generation: 1 } })
            .mockResolvedValueOnce({ success: true, data: { state: 'data', generation: 1, lease: 1, initial: true, index: 0, bytes: bank.at(0).slice() } })
            .mockResolvedValueOnce({ success: true, data: { state: 'accepted', generation: 1 } })
            .mockResolvedValueOnce({ success: true, data: { state: 'end', generation: 1, lease: 2, count: 2, retired: false } })
        const history = createRoutingHistory(serviceTransport(pairTransport))
        const initial = history.getInitial(), rejected = expect(initial).rejects.toThrow('history-invalid')
        await settle(); ready({ generation: 1 }); await settle()
        ready({ generation: 1 }); await rejected
        expect(history.size).toBe(0)
        expect(frames.size).toBe(0)
        expect(boundary.invoke.mock.calls.filter(([channel]) => channel === 'routing-history:consumed')).toEqual([
            ['routing-history:consumed', { generation: 1, lease: 1 }]
        ])
        history.destroy()
    })

    it('coalesces concurrent and repeated initial loads and resolves only void after rAF commit and ack', async () => {
        replies()
        const history = createPairApi(serviceTransport(pairTransport)).routing
        const first = history.getInitial()
        expect(history.getInitial()).toBe(first)
        await settle()
        ready({ generation: 1 }); await settle()
        expect(boundary.invoke.mock.calls.map(([channel]) => channel)).toEqual(['service-bridge:invoke', 'routing-history:pull'])
        expect(history.loading).toBe(true)
        commitFrame(); await settle()
        await expect(first).resolves.toBeUndefined()
        expect(history.getInitial()).toBe(first)
        expect(history.size).toBe(0)
        expect(history.loading).toBe(false)
        expect(boundary.invoke.mock.calls.map(([channel]) => channel)).toEqual([
            'service-bridge:invoke', 'routing-history:pull', 'routing-history:consumed'
        ])
        boundary.invoke.mockResolvedValueOnce({ success: true, data: null })
            .mockResolvedValueOnce({ success: true, data: { state: 'retired', generation: 0 } })
        const retired = history.unsubscribe()
        await settle(); ready({ generation: 1 }); await retired
    })

    it('consumes staged records but does not acknowledge scalar end before coherent rAF commit', async () => {
        replies([fixture()])
        const history = createRoutingHistory(serviceTransport(pairTransport))
        const changed = vi.fn<(token: RoutingHistoryCommit) => undefined>(() => undefined)
        history.onChanged(changed)
        const first = history.getInitial()
        await settle(); ready({ generation: 1 }); await settle()
        expect(changed).not.toHaveBeenCalled()
        expect(history.size).toBe(0)
        expect(frames.size).toBe(0)
        expect(boundary.invoke).toHaveBeenCalledTimes(3)
        ready({ generation: 1 }); await settle()
        expect(boundary.invoke).toHaveBeenCalledTimes(4)
        commitFrame(); await first
        expect(changed).toHaveBeenCalledWith({ generation: 1, revision: 2 })
        expect(boundary.invoke.mock.calls[4]).toEqual(['routing-history:consumed', { generation: 1, lease: 2 }])
        history.destroy()
    })

    it('rejects nested, async and mutating output loans without allocating a window result', async () => {
        replies([fixture()])
        const history = createRoutingHistory(serviceTransport(pairTransport))
        const first = history.getInitial()
        await settle(); ready({ generation: 1 }); await settle(); ready({ generation: 1 }); await settle(); commitFrame(); await first
        expect(history.consumeCurrent(0, row => {
            expect(row.request.requestId).toBe('1')
            expect(() => history.consumeCurrent(0, () => undefined)).toThrow('history-busy')
            expect(() => history.getInitial()).toThrow('history-busy')
            expect(() => history.unsubscribe()).toThrow('history-busy')
            return undefined
        })).toBe(true)
        // @ts-expect-error A loan callback explicitly returns undefined, not Promise<void>.
        expect(() => history.consumeCurrent(0, async () => {})).toThrow('history-invalid')
        expect(history.consumeCurrent(1, () => undefined)).toBe(false)
        history.destroy()
    })

    it('keeps a busy initial loading with no renderer retry timer or early empty result', async () => {
        boundary.invoke.mockResolvedValueOnce({ success: true, data: { generation: 1 } })
            .mockResolvedValueOnce({ success: true, data: { state: 'busy', generation: 1 } })
        const history = createRoutingHistory(serviceTransport(pairTransport))
        const first = history.getInitial(), finished = vi.fn<() => void>()
        void first.then(finished)
        await settle(); ready({ generation: 1 }); await settle()
        expect(finished).not.toHaveBeenCalled()
        expect(history.loading).toBe(true)
        expect(frames.size).toBe(0)
        boundary.invoke.mockResolvedValueOnce({ success: true, data: {
            state: 'end', generation: 1, lease: 1, count: 0, retired: false
        } }).mockResolvedValueOnce({ success: true, data: { state: 'accepted', generation: 1 } })
        ready({ generation: 1 }); await settle(); commitFrame(); await first
        expect(finished).toHaveBeenCalledOnce()
        history.destroy()
    })

    it('finishes an admitted retired commit before acknowledgement and cancels its visible rAF on unsubscribe', async () => {
        replies([fixture()])
        const history = createRoutingHistory(serviceTransport(pairTransport))
        const first = history.getInitial()
        const cancelled = expect(first).rejects.toThrow('history-retired')
        await settle(); ready({ generation: 1 }); await settle()
        ready({ generation: 1 }); await settle()
        expect(frames.size).toBe(1)
        boundary.invoke.mockReset()
        boundary.invoke.mockResolvedValueOnce({ success: true, data: null })
            .mockResolvedValueOnce({ success: true, data: { state: 'retired', generation: 0 } })
        await history.unsubscribe(); await cancelled; await settle()
        expect(frames.size).toBe(0)
        expect(history.size).toBe(0)
        expect(boundary.invoke.mock.calls.some(([channel]) => channel === 'routing-history:consumed')).toBe(true)
        history.destroy()
    })
})
