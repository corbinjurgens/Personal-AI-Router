// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import net from 'net'
import os from 'os'
import path from 'path'
import readline from 'readline'
import { Duplex } from 'stream'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
    JsonRpcResponseError,
    JsonRpcSocketClient,
    type JsonRpcNotification
} from '@/electron/service-bridge/json-rpc-client'

vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        verbose: vi.fn()
    })
}))

interface Frame {
    id?: number | string
    method?: string
    params?: object
    error?: { code: number; message: string }
}

/** A stand-in for nvpair-service: records what clients send and answers requests. */
class FakeService {
    readonly received: Frame[] = []
    readonly sockets: net.Socket[] = []
    private readonly server: net.Server
    private readonly endpoint: string

    constructor(dir: string) {
        this.endpoint = path.join(dir, 'service.sock')
        this.server = net.createServer(socket => {
            this.sockets.push(socket)
            readline.createInterface({ input: socket }).on('line', line => {
                const frame: Frame = JSON.parse(line)
                this.received.push(frame)
                if (frame.method === 'refuse') {
                    const error = { code: -32011, message: 'paused on the node itself' }
                    socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, error }) + '\n')
                } else if (frame.id !== undefined && frame.method !== 'hang') {
                    const result = frame.method === 'echo' ? frame.params : null
                    socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }) + '\n')
                }
            })
            socket.on('error', () => {})
        })
    }

    listen(): Promise<void> {
        return new Promise(resolve => this.server.listen(this.endpoint, resolve))
    }

    connect(): Promise<net.Socket> {
        return new Promise((resolve, reject) => {
            const socket = net.connect(this.endpoint)
            socket.once('connect', () => resolve(socket))
            socket.once('error', reject)
        })
    }

    push(message: object): void {
        for (const socket of this.sockets) socket.write(JSON.stringify(message) + '\n')
    }

    dropClients(): void {
        for (const socket of this.sockets.splice(0)) socket.destroy()
    }

    close(): Promise<void> {
        this.dropClients()
        return new Promise(resolve => this.server.close(() => resolve()))
    }
}

const describeUnix = process.platform === 'win32' ? describe.skip : describe

/** An in-process service transport; no sockets or service processes are opened. */
class MemoryService extends Duplex {
    readonly received: Frame[] = []
    blockedWrite = false
    private finishWrite: (() => void) | null = null

    override _read(): void {
        // The test supplies service frames through reply().
    }

    override _write(chunk: Buffer, _encoding: BufferEncoding, done: () => void): void {
        const frame: Frame = JSON.parse(chunk.toString())
        this.received.push(frame)
        if (this.blockedWrite) this.finishWrite = done
        else done()
    }

    releaseWrite(): void {
        const finish = this.finishWrite
        this.finishWrite = null
        this.blockedWrite = false
        finish?.()
    }

    reply(id: number | string, result: object): void {
        this.push(JSON.stringify({ jsonrpc: '2.0', id, result }) + '\n')
    }

    async disconnect(): Promise<void> {
        await new Promise<void>(resolve => {
            this.once('close', resolve)
            this.destroy()
        })
    }
}

describe('JsonRpcSocketClient history ingress', () => {
    it('keeps one refusal reserved through its write and lets the next ordinary call follow it', async () => {
        const service = new MemoryService()
        const client = new JsonRpcSocketClient(
            'broker',
            async () => service,
            [],
            async () => service
        )
        const turn = () => new Promise<void>(resolve => setImmediate(resolve))
        try {
            client.start()
            await Promise.resolve()
            service.blockedWrite = true
            service.push(
                '{"jsonrpc":"2.0","id":"private-1","method":"routing/request","params":{}}\n'
            )
            await turn()
            expect(service.received).toEqual([
                {
                    jsonrpc: '2.0',
                    id: 'private-1',
                    error: { code: -32601, message: 'method not found' }
                }
            ])
            const ordinary = client.call('echo', { marker: 'after' })
            await turn()
            expect(service.received).toHaveLength(1)
            expect(service.writableLength).toBeLessThanOrEqual(4096 + 128)
            service.releaseWrite()
            await turn()
            expect(service.received.map(frame => frame.method ?? frame.error?.code)).toEqual([
                -32601,
                'echo'
            ])
            service.reply(1, { ok: true })
            await expect(ordinary).resolves.toEqual({ ok: true })
            service.push(
                '{"jsonrpc":"2.0","id":"private-2","method":"routing/request","params":{}}\n'
            )
            await turn()
            expect(service.received.at(-1)?.id).toBe('private-2')
            expect(client.running).toBe(true)
        } finally {
            service.releaseWrite()
            await client.detach()
        }
    })

    it('resets a pinned connection on a refusal flood and reconnects without an ordinary output backlog', async () => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
        const first = new MemoryService(),
            replacement = new MemoryService()
        const connect = vi
            .fn<() => Promise<Duplex>>()
            .mockResolvedValueOnce(first)
            .mockResolvedValueOnce(replacement)
        const client = new JsonRpcSocketClient('broker', connect, [1], async () => replacement)
        const disconnected = vi.fn<() => void>()
        client.on('disconnected', disconnected)
        const turn = () => new Promise<void>(resolve => setImmediate(resolve))
        try {
            client.start()
            await Promise.resolve()
            first.blockedWrite = true
            const before = expect(client.call('echo', { marker: 'before' })).rejects.toThrow(
                'connection closed before replying'
            )
            await turn()
            const originalBytes = first.writableLength
            first.push(
                Array.from(
                    { length: 1000 },
                    (_, index) =>
                        JSON.stringify({
                            jsonrpc: '2.0',
                            id: `private-${index}`,
                            method: 'routing/request',
                            params: {}
                        }) + '\n'
                ).join('')
            )
            await turn()
            await before
            expect(first.destroyed).toBe(true)
            expect(disconnected).toHaveBeenCalledOnce()
            expect(first.received).toHaveLength(1)
            expect(first.writableLength).toBeLessThanOrEqual(originalBytes + 4096)
            await expect(client.call('echo', { marker: 'after-flood' })).rejects.toThrow(
                'not running'
            )
            await vi.advanceTimersByTimeAsync(1)
            expect(connect).toHaveBeenCalledTimes(2)
            // The old native callback is still pinned; it cannot block this transport.
            const ordinary = client.call('echo', { marker: 'replacement' })
            await turn()
            expect(replacement.received).toMatchObject([
                { method: 'echo', params: { marker: 'replacement' } }
            ])
            const id = replacement.received[0]?.id
            if (id === undefined) throw new Error('missing replacement request ID')
            replacement.reply(id, { ok: true })
            await expect(ordinary).resolves.toEqual({ ok: true })
            first.releaseWrite()
            await turn()
            expect(first.received.filter(frame => frame.error)).toHaveLength(0)
            expect(first.received).toHaveLength(1)
        } finally {
            first.releaseWrite()
            await client.detach()
            vi.useRealTimers()
        }
    })

    it('pins its actual control write after logical cancellation while ordinary input still resolves', async () => {
        const service = new MemoryService()
        const client = new JsonRpcSocketClient(
            'broker',
            async () => service,
            [],
            async () => service
        )
        try {
            client.start()
            await Promise.resolve()
            const ordinary = client.call('echo', {})
            await Promise.resolve()
            service.blockedWrite = true
            const initial = client.callHistory('routing:get-initial', () => true)
            const retired = expect(initial).rejects.toThrow('history-invalid')
            client.cancelHistory()
            await retired
            expect(client.historyWritable).toBe(false)
            await expect(client.callHistory('routing:unsubscribe', () => true)).rejects.toThrow(
                'routing history busy; retry'
            )
            service.reply(1, { ok: true })
            await expect(ordinary).resolves.toEqual({ ok: true })
            service.releaseWrite()
            await Promise.resolve()
            expect(client.historyWritable).toBe(true)
            const cleanup = client.callHistory('routing:unsubscribe', () => true)
            expect(service.received.map(frame => frame.id)).toEqual([1, 'jh:1', 'jh:2'])
            service.reply('jh:2', { requests: [] })
            await cleanup
        } finally {
            service.releaseWrite()
            await client.detach()
        }
    })

    it('rejects ordinary call, send and notify history methods before transmission', async () => {
        const service = new MemoryService()
        const client = new JsonRpcSocketClient(
            'broker',
            async () => service,
            [],
            async () => service
        )
        try {
            client.start()
            await Promise.resolve()
            await expect(client.call('routing:get-initial', {})).rejects.toThrow('history-invalid')
            await expect(client.send('routing:subscribe', {})).rejects.toThrow('history-invalid')
            await expect(client.notify('routing:unsubscribe', {})).rejects.toThrow(
                'history-invalid'
            )
            expect(service.received).toEqual([])
        } finally {
            await client.detach()
        }
    })

    it('mints disjoint history IDs and consumes only guarded known fields', async () => {
        const service = new MemoryService()
        const client = new JsonRpcSocketClient(
            'broker',
            async () => service,
            [],
            async () => service
        )
        try {
            client.start()
            await Promise.resolve()
            const seen: string[] = []
            const response = client.callHistory('routing:get-initial', raw => {
                seen.push(new TextDecoder().decode(raw))
                return true
            })
            expect(service.received[0]?.id).toBe('jh:1')
            service.reply('jh:1', { requests: [], discarded: 'synthetic-secret' })
            await response
            expect(seen).toEqual(['{"requests":[]}'])
        } finally {
            await client.detach()
        }
    })

    it('quarantines mixed ID candidates while ordinary RPC and same-connection refetch progress', async () => {
        const service = new MemoryService()
        const client = new JsonRpcSocketClient(
            'broker',
            async () => service,
            [],
            async () => service
        )
        try {
            client.start()
            await Promise.resolve()
            const first = client.callHistory('routing:get-initial', () => true)
            const rejected = expect(first).rejects.toThrow('history-invalid')
            const ordinary = client.call('echo', {})
            await Promise.resolve()
            service.push('{"jsonrpc":"2.0","id":"jh:1","\\u0069d":1,"result":{}}\n')
            await rejected
            service.reply(1, { ok: true })
            await expect(ordinary).resolves.toEqual({ ok: true })
            const next = client.callHistory('routing:get-initial', () => true)
            service.reply('jh:1', { requests: [] })
            service.reply('jh:999', { discarded: 'synthetic-secret' })
            service.reply('jh:2', { requests: [] })
            await next
        } finally {
            await client.detach()
        }
    })

    it('reports only fixed diagnostics for malformed history before raw trace while ordinary RPC progresses', async () => {
        const service = new MemoryService()
        const client = new JsonRpcSocketClient(
            'broker',
            async () => service,
            [],
            async () => service
        )
        const logs: string[] = []
        client.on('log', entry => logs.push(entry.text))
        client.setProtocolTrace(true)
        try {
            client.start()
            await Promise.resolve()
            for (const params of [
                '{"sentinel":"synthetic-secret","\\u0073entinel":"other-secret"}',
                '{"\\u0073entinel":"other-secret","sentinel":"synthetic-secret"}'
            ])
                service.push(`{"jsonrpc":"2.0","method":"routing:changed","params":${params}}\n`)
            service.push('{"jsonrpc":"2.0","method":"synthetic-secret",\n')
            const ordinary = client.call('echo', {})
            await Promise.resolve()
            service.reply(1, { ok: true })
            await expect(ordinary).resolves.toEqual({ ok: true })
            expect(logs.slice(0, 3)).toEqual(Array(3).fill('history-unknown history-invalid 1'))
            expect(logs.join(' ')).not.toMatch(/synthetic-secret|other-secret|sentinel/)
        } finally {
            await client.detach()
        }
    })

    it('closes physical overflow through the existing single reconnect owner without raw trace', async () => {
        vi.useFakeTimers()
        const first = new MemoryService(),
            second = new MemoryService()
        const connect = vi
            .fn<() => Promise<Duplex>>()
            .mockResolvedValueOnce(first)
            .mockResolvedValueOnce(second)
        const client = new JsonRpcSocketClient('broker', connect, [1], async () => second)
        const lost = vi.fn(),
            trace = vi.fn()
        client.on('disconnected', lost)
        client.on('log', trace)
        client.setProtocolTrace(true)
        try {
            client.start()
            await vi.advanceTimersByTimeAsync(0)
            const initial = client.callHistory('routing:get-initial', () => true)
            const initialFailed = expect(initial).rejects.toThrow('history-invalid')
            expect(first.received[0]?.id).toBe('jh:1')
            trace.mockClear()
            first.push(Buffer.alloc(33558528, 120))
            await vi.advanceTimersByTimeAsync(1)
            await initialFailed
            expect(lost).toHaveBeenCalledOnce()
            expect(connect).toHaveBeenCalledTimes(2)
            expect(client.running).toBe(true)
            expect(trace).not.toHaveBeenCalled()
            const applied = vi.fn<(raw: Uint8Array) => boolean>(() => true)
            const replacement = client.callHistory('routing:get-initial', applied)
            expect(second.received[0]?.id).toBe('jh:1')
            // An already retired socket cannot settle a replacement call even
            // though the generated ordinal restarts in its new generation.
            first.emit(
                'data',
                Buffer.from('{"jsonrpc":"2.0","id":"jh:1","result":{"requests":[]}}\n')
            )
            expect(applied).not.toHaveBeenCalled()
            second.reply('jh:1', { requests: [] })
            await replacement
            expect(applied).toHaveBeenCalledOnce()
            const ordinary = client.call('echo', {})
            await Promise.resolve()
            second.reply(1, { ok: true })
            await expect(ordinary).resolves.toEqual({ ok: true })
            expect(connect).toHaveBeenCalledTimes(2)
        } finally {
            await client.detach()
            vi.useRealTimers()
        }
    })

    it('stops history at uint64 exhaustion while ordinary numeric calls remain usable', async () => {
        const service = new MemoryService()
        const client = new JsonRpcSocketClient(
            'broker',
            async () => service,
            [],
            async () => service
        )
        try {
            client.start()
            await Promise.resolve()
            Object.defineProperty(client, 'historyIssued', {
                value: 18446744073709551615n,
                writable: true
            })
            await expect(client.callHistory('routing:get-initial', () => true)).rejects.toThrow(
                'history-exhausted'
            )
            const ordinary = client.call('echo', {})
            await Promise.resolve()
            service.reply(1, { ok: true })
            await expect(ordinary).resolves.toEqual({ ok: true })
            expect(service.received.map(frame => frame.id)).toEqual([1])
        } finally {
            await client.detach()
        }
    })
})

describe('JsonRpcSocketClient in-process recovery', () => {
    let client: JsonRpcSocketClient | undefined

    beforeEach(() => vi.useFakeTimers())

    afterEach(async () => {
        await client?.detach()
        client = undefined
        vi.useRealTimers()
    })

    async function attach(
        connect: () => Promise<Duplex>,
        dial: () => Promise<Duplex>
    ): Promise<JsonRpcSocketClient> {
        client = new JsonRpcSocketClient('broker', connect, [5, 10], dial)
        client.start()
        await vi.advanceTimersByTimeAsync(0)
        expect(client.running).toBe(true)
        return client
    }

    it('recovers by dial only after exhausting the finite start retries', async () => {
        const original = new MemoryService()
        const replacement = new MemoryService()
        const connect = vi
            .fn<() => Promise<Duplex>>()
            .mockResolvedValueOnce(original)
            .mockRejectedValue(new Error('start unavailable'))
        const dial = vi
            .fn<() => Promise<Duplex>>()
            .mockRejectedValueOnce(new Error('still unavailable'))
            .mockResolvedValue(replacement)
        const rpc = await attach(connect, dial)
        const exited = vi.fn()
        rpc.on('exit', exited)

        await original.disconnect()
        await vi.advanceTimersByTimeAsync(15)
        expect(connect).toHaveBeenCalledTimes(3)
        expect(dial).not.toHaveBeenCalled()
        await vi.advanceTimersByTimeAsync(10)
        expect(dial).toHaveBeenCalledTimes(1)
        expect(rpc.running).toBe(false)
        await vi.advanceTimersByTimeAsync(10)

        expect(dial).toHaveBeenCalledTimes(2)
        expect(connect).toHaveBeenCalledTimes(3)
        expect(exited).not.toHaveBeenCalled()
        expect(rpc.running).toBe(true)
        const fresh = rpc.call('fresh')
        await vi.advanceTimersByTimeAsync(0)
        expect(replacement.received).toEqual([{ jsonrpc: '2.0', id: 1, method: 'fresh' }])
        replacement.reply(1, { generation: 'replacement' })
        await expect(fresh).resolves.toEqual({ generation: 'replacement' })
    })

    it('destroys a dial completed after detach and never retries it', async () => {
        const original = new MemoryService()
        const late = new MemoryService()
        const pendingDial = Promise.withResolvers<Duplex>()
        const connect = vi
            .fn<() => Promise<Duplex>>()
            .mockResolvedValueOnce(original)
            .mockRejectedValue(new Error('unavailable'))
        const dial = vi.fn<() => Promise<Duplex>>().mockReturnValue(pendingDial.promise)
        const rpc = await attach(connect, dial)
        await original.disconnect()
        await vi.advanceTimersByTimeAsync(25)
        expect(dial).toHaveBeenCalledOnce()

        await rpc.detach()
        pendingDial.resolve(late)
        await vi.advanceTimersByTimeAsync(100)

        expect(late.destroyed).toBe(true)
        expect(rpc.running).toBe(false)
        expect(dial).toHaveBeenCalledOnce()
        expect(connect).toHaveBeenCalledTimes(3)
        expect(vi.getTimerCount()).toBe(0)
    })

    it('cancels dial-only backoff on detach', async () => {
        const original = new MemoryService()
        const connect = vi
            .fn<() => Promise<Duplex>>()
            .mockResolvedValueOnce(original)
            .mockRejectedValue(new Error('unavailable'))
        const dial = vi.fn<() => Promise<Duplex>>().mockRejectedValue(new Error('unavailable'))
        const rpc = await attach(connect, dial)
        await original.disconnect()
        await vi.advanceTimersByTimeAsync(25)
        expect(dial).toHaveBeenCalledOnce()
        expect(vi.getTimerCount()).toBe(1)

        await rpc.detach()
        expect(vi.getTimerCount()).toBe(0)
        await vi.advanceTimersByTimeAsync(1_000)
        expect(dial).toHaveBeenCalledOnce()
        expect(connect).toHaveBeenCalledTimes(3)
        expect(rpc.running).toBe(false)
    })

    it('rejects a pending mutation on loss without replaying it after recovery', async () => {
        const original = new MemoryService()
        const replacement = new MemoryService()
        const connect = vi
            .fn<() => Promise<Duplex>>()
            .mockResolvedValueOnce(original)
            .mockRejectedValue(new Error('unavailable'))
        const dial = vi.fn<() => Promise<Duplex>>().mockResolvedValue(replacement)
        const rpc = await attach(connect, dial)
        const pending = rpc.call('mutate', { value: 'once' }, null)
        const rejected = expect(pending).rejects.toThrow('broker connection closed before replying')
        await vi.advanceTimersByTimeAsync(0)
        expect(original.received).toEqual([
            { jsonrpc: '2.0', id: 1, method: 'mutate', params: { value: 'once' } }
        ])

        await original.disconnect()
        await rejected
        await expect(rpc.call('during-outage')).rejects.toThrow('broker is not running')
        await vi.advanceTimersByTimeAsync(25)
        expect(rpc.running).toBe(true)
        expect(replacement.received).toEqual([])
        const fresh = rpc.call('fresh')
        await vi.advanceTimersByTimeAsync(0)
        expect(replacement.received).toEqual([{ jsonrpc: '2.0', id: 2, method: 'fresh' }])
        replacement.reply(2, { ok: true })
        await expect(fresh).resolves.toEqual({ ok: true })
    })

    it('keeps initial startup failure terminal without dialing or scheduling retries', async () => {
        const connect = vi.fn<() => Promise<Duplex>>().mockRejectedValue(new Error('cannot start'))
        const dial = vi.fn<() => Promise<Duplex>>()
        client = new JsonRpcSocketClient('broker', connect, [5, 10], dial)
        const exited = vi.fn()
        client.on('exit', exited)
        client.start()
        await vi.advanceTimersByTimeAsync(1_000)

        expect(connect).toHaveBeenCalledOnce()
        expect(dial).not.toHaveBeenCalled()
        expect(exited).toHaveBeenCalledExactlyOnceWith({ source: 'broker', code: null })
        expect(client.running).toBe(false)
        expect(vi.getTimerCount()).toBe(0)
    })
})

describeUnix('JsonRpcSocketClient', () => {
    let dir: string
    let service: FakeService
    let client: JsonRpcSocketClient | undefined

    beforeEach(async () => {
        dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pair-rpc-'))
        service = new FakeService(dir)
        await service.listen()
    })

    afterEach(async () => {
        await client?.detach()
        client = undefined
        await service.close()
        fs.rmSync(dir, { recursive: true, force: true })
    })

    function attach(
        connect: () => Promise<net.Socket> = () => service.connect(),
        delays: readonly number[] = [10, 10]
    ): JsonRpcSocketClient {
        const attached = new JsonRpcSocketClient('broker', connect, delays, () => service.connect())
        client = attached
        return attached
    }

    async function until(condition: () => boolean): Promise<void> {
        await vi.waitFor(() => expect(condition()).toBe(true))
    }

    it('answers requests and delivers notifications under the client name', async () => {
        const rpc = attach()
        const notifications: JsonRpcNotification[] = []
        rpc.on('notification', n => notifications.push(n))
        rpc.start()
        await until(() => rpc.running)

        await expect(rpc.call('echo', { hello: 'world' })).resolves.toEqual({ hello: 'world' })

        service.push({ jsonrpc: '2.0', method: 'app:ready', params: { ok: true } })
        await until(() => notifications.length === 1)
        expect(notifications[0]).toEqual({
            source: 'broker',
            method: 'app:ready',
            params: { ok: true }
        })
    })

    it('rejects an error response with its code beside the message', async () => {
        const rpc = attach()
        rpc.start()
        await until(() => rpc.running)

        const failure = await rpc.call('refuse').catch((err: unknown) => err)
        expect(failure).toBeInstanceOf(JsonRpcResponseError)
        expect(failure).toHaveProperty('code', -32011)
        expect(failure).toHaveProperty('message', '-32011: paused on the node itself')
    })

    it('rejects calls while not connected', async () => {
        const rpc = attach()

        await expect(rpc.call('echo')).rejects.toThrow('broker is not running')
    })

    it('turns service/log into a redacted stderr log entry', async () => {
        const rpc = attach()
        const logs: { source: string; stream: string; text: string }[] = []
        rpc.on('log', entry => logs.push(entry))
        rpc.start()
        await until(() => rpc.running)

        service.push({
            jsonrpc: '2.0',
            method: 'service/log',
            params: {
                source: 'nvpair-ui-broker',
                stream: 'stderr',
                text: '10:00:00.000 [broker] INFO invite {"inviteId":"a","pin":"481920"}'
            }
        })
        await until(() => logs.length === 1)

        expect(logs[0].source).toBe('broker')
        expect(logs[0].stream).toBe('stderr')
        expect(logs[0].text).not.toContain('481920')
        expect(logs[0].text).toContain('[redacted]')
    })

    it('does not surface service/log or broker-restarted as bridge notifications', async () => {
        const rpc = attach()
        const methods: string[] = []
        const restarted = vi.fn()
        rpc.on('notification', n => methods.push(n.method))
        rpc.on('broker-restarted', restarted)
        rpc.start()
        await until(() => rpc.running)

        service.push({ jsonrpc: '2.0', method: 'service/log', params: { text: 'x' } })
        service.push({ jsonrpc: '2.0', method: 'service/broker-restarted', params: {} })
        service.push({ jsonrpc: '2.0', method: 'after' })
        await until(() => methods.length === 1)

        expect(methods).toEqual(['after'])
        expect(restarted).toHaveBeenCalledOnce()
    })

    it('detaches without sending shutdown', async () => {
        const rpc = attach()
        rpc.start()
        await until(() => rpc.running)
        await rpc.call('echo')

        await rpc.detach()
        await until(() => service.sockets.every(socket => socket.destroyed))

        expect(service.received.map(frame => frame.method)).toEqual(['echo'])
        expect(rpc.running).toBe(false)
    })

    it('asks the service to stop, then detaches without reconnecting', async () => {
        const connect = vi.fn(() => service.connect())
        const rpc = attach(connect)
        const exited = vi.fn()
        rpc.on('exit', exited)
        rpc.start()
        await until(() => rpc.running)

        await rpc.stopService(2_000)
        await new Promise(resolve => setTimeout(resolve, 60))

        expect(service.received.map(frame => frame.method)).toEqual(['service/stop'])
        expect(connect).toHaveBeenCalledOnce()
        expect(exited).not.toHaveBeenCalled()
    })

    it('reconnects after an unexpected drop and keeps working', async () => {
        const connect = vi.fn(() => service.connect())
        const rpc = attach(connect)
        const disconnected = vi.fn()
        rpc.on('disconnected', disconnected)
        rpc.start()
        await until(() => rpc.running)

        service.dropClients()
        await until(() => disconnected.mock.calls.length === 1)
        await until(() => rpc.running)

        expect(connect).toHaveBeenCalledTimes(2)
        await expect(rpc.call('echo', { again: true })).resolves.toEqual({ again: true })
    })

    it('fails calls that were waiting when the connection dropped', async () => {
        const rpc = attach()
        rpc.start()
        await until(() => rpc.running)

        const waiting = rpc.call('hang', null, null)
        await until(() => service.received.length === 1)
        service.dropClients()

        await expect(waiting).rejects.toThrow('broker connection closed before replying')
    })

    it('dials the service after every connect-or-start retry has failed', async () => {
        let attempts = 0
        const rpc = attach(() => {
            attempts++
            return attempts === 1 ? service.connect() : Promise.reject(new Error('refused'))
        }, [5, 5, 5])
        const exited = vi.fn()
        rpc.on('exit', exited)
        rpc.start()
        await until(() => rpc.running)

        service.dropClients()
        await until(() => attempts === 4 && rpc.running)

        expect(attempts).toBe(4)
        expect(exited).not.toHaveBeenCalled()
        await expect(rpc.call('echo', { recovered: true })).resolves.toEqual({ recovered: true })
    })

    it('reports exit when the first connection fails', async () => {
        const rpc = attach(() => Promise.reject(new Error('could not start')), [])
        const exited = vi.fn()
        rpc.on('exit', exited)

        rpc.start()
        await until(() => exited.mock.calls.length === 1)

        expect(rpc.running).toBe(false)
    })
})
