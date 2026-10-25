// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EventEmitter } from 'events'
import type { Duplex } from 'stream'
import getErrorString from '@/shared/utils/get-error-string'
import { createStructuredLogger } from '@/shared/utils/log'
import { redactSensitiveLogText } from '@/electron/redact-log'
import { ROUTING_FRAME_BYTES, getRoutingJsonProjector, type RoutingHistoryMethod } from './routing-json'

export type JsonValue = string | number | boolean | null | JsonObject | JsonValue[]

export interface JsonObject {
    [key: string]: JsonValue
}

interface JsonRpcError {
    code: number
    message: string
    data?: JsonValue
}

/**
 * A JSON-RPC error response. The message reads `<code>: <message>`; `code`
 * carries the code itself, so callers classify by code, not by text.
 */
export class JsonRpcResponseError extends Error {
    readonly code: number

    constructor(code: number, message: string) {
        super(`${code}: ${message}`)
        this.code = code
    }
}

type JsonRpcId = number | string

interface JsonRpcMessage {
    jsonrpc: '2.0'
    id?: JsonRpcId
    method?: string
    params?: JsonValue
    result?: JsonValue
    error?: JsonRpcError
}

export interface JsonRpcNotification {
    source: string
    method: string
    params?: JsonValue
}

/**
 * A request the service sent to us (it carries both `id` and `method`). The
 * owner answers via `respond` / `respondError` exactly once. Nothing behind the
 * service currently sends inbound requests; `handleChildRequest` replies
 * method-not-found.
 */
export interface JsonRpcInboundRequest {
    source: string
    method: string
    params?: JsonValue
    id: JsonRpcId
}

interface PendingCall {
    resolve: (value: JsonValue | undefined) => void
    reject: (error: Error) => void
    timeout?: ReturnType<typeof setTimeout>
}

interface PendingHistory {
    ordinal: bigint
    method: RoutingHistoryMethod
    consume: (raw: Uint8Array) => boolean
    resolve: () => void
    reject: (error: Error) => void
    timer: ReturnType<typeof setTimeout>
}
interface HistoryHandlers {
    notification: (raw: Uint8Array) => undefined
    invalid: () => void
    writable: () => void
}

interface JsonRpcClientEvents {
    notification: [JsonRpcNotification]
    request: [JsonRpcInboundRequest]
    log: [{ source: string; stream: 'stdout' | 'stderr'; text: string }]
    /** The connection dropped unexpectedly; a reconnect is under way. */
    disconnected: []
    /** The service replaced a broker that died. Its `app:ready` follows. */
    'broker-restarted': []
    /** The client gave up: the first connection failed, or every reconnect did. */
    exit: [{ source: string; code: number | null }]
}

/** Pauses before each reconnect attempt after an unexpected drop. */
const DEFAULT_RECONNECT_DELAYS_MS = [500, 1_000, 2_000, 5_000, 10_000]

const SERVICE_LOG_METHOD = 'service/log'
const SERVICE_BROKER_RESTARTED_METHOD = 'service/broker-restarted'

const log = createStructuredLogger('service-bridge')
// One fixed desktop history control writer across client/socket replacements.
// Its active borrower survives logical cancellation and releases only on the
// actual write callback. The wake always targets the current connection owner.
const historyControl = new Uint8Array(4096)
let historyControlBusy = false
let historyWake: (() => void) | null = null
let historyWakeOwner: JsonRpcSocketClient | null = null

function isHistoryCall(method: string): method is RoutingHistoryMethod {
    return method === 'routing:get-initial' || method === 'routing:subscribe' || method === 'routing:unsubscribe'
}

function isJsonObject(value: JsonValue | undefined): value is JsonObject {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function parseError(value: JsonValue | undefined): JsonRpcError | undefined {
    if (!isJsonObject(value)) return undefined
    const { code, message } = value
    if (typeof code !== 'number' || typeof message !== 'string') return undefined
    return { code, message }
}

/** Read one frame; null when the line is not a JSON-RPC 2.0 object. */
function parseMessage(line: string): JsonRpcMessage | null {
    let value: JsonValue
    try {
        value = JSON.parse(line)
    } catch {
        return null
    }
    if (!isJsonObject(value) || value.jsonrpc !== '2.0') return null

    const message: JsonRpcMessage = { jsonrpc: '2.0' }
    if (typeof value.id === 'number' || typeof value.id === 'string') message.id = value.id
    if (typeof value.method === 'string') message.method = value.method
    if (value.params !== undefined) message.params = value.params
    if (value.result !== undefined) message.result = value.result
    const error = parseError(value.error)
    if (error) message.error = error
    return message
}

/**
 * A JSON-RPC 2.0 client for `nvpair-service`, over a local socket or named pipe.
 *
 * It presents the same events and request/notify API the supervisor used when
 * it owned the broker's stdio, so everything above it keeps working. The service
 * multiplexes the broker for every attached client; this client only attaches.
 * {@link detach} closes the connection and never stops the service.
 *
 * `connect` yields a connected stream (starting the service if need be). The
 * first connection is attempted once. After an unexpected drop the client
 * reconnects with the finite `connect` retry list, then keeps trying `dial`
 * without starting the service until deliberately detached.
 */
export class JsonRpcSocketClient extends EventEmitter<JsonRpcClientEvents> {
    readonly name: string

    private readonly connect: () => Promise<Duplex>
    private readonly reconnectDelaysMs: readonly number[]
    private readonly dial: () => Promise<Duplex>
    private cancelReconnectWait?: () => void
    private socket: Duplex | null = null
    private started = false
    private closed = false
    // Bumped on every start and detach, so a connect or reconnect loop that
    // outlives its session notices and stops instead of attaching a stale socket.
    private epoch = 0
    private pending = new Map<number, PendingCall>()
    private nextId = 0
    private writeTail: Promise<void> = Promise.resolve()
    private traceProtocol = false
    private rawFrame: Uint8Array | null = null
    private readonly historyProjector = getRoutingJsonProjector()
    private historyHandlers: HistoryHandlers | null = null
    private rawLength = 0
    private historyIssued = 0n
    private historyPending: PendingHistory | null = null
    private historyQuarantined = false

    constructor(
        name: string,
        connect: () => Promise<Duplex>,
        reconnectDelaysMs: readonly number[] = DEFAULT_RECONNECT_DELAYS_MS,
        dial: () => Promise<Duplex>
    ) {
        super()
        this.name = name
        this.connect = connect
        this.reconnectDelaysMs = reconnectDelaysMs
        this.dial = dial
    }

    /** Whether a connection to the service is currently open. */
    get running(): boolean {
        return this.socket !== null
    }

    /**
     * Whether JSON-RPC frames received from the service are emitted as `log`
     * entries. Only the `debug` service log level wants them; the broker's own
     * log lines (`service/log`) are always emitted.
     */
    setProtocolTrace(enabled: boolean): void {
        this.traceProtocol = enabled
    }

    /** Begin connecting in the background. Failure is reported as `exit`. */
    start(): void {
        if (this.started) return
        this.started = true
        this.closed = false
        const epoch = ++this.epoch
        void this.connectOnce(epoch).then(connected => {
            if (!connected && this.epoch === epoch) this.giveUp()
        })
    }

    async call(
        method: string,
        params?: JsonValue,
        timeoutMs: number | null = 10_000
    ): Promise<JsonValue | undefined> {
        if (isHistoryCall(method)) {
            throw new Error('history-invalid')
        }
        if (!this.socket) throw new Error(`${this.name} is not running`)

        const id = ++this.nextId
        const message: JsonRpcMessage = { jsonrpc: '2.0', id, method, params }

        return new Promise((resolve, reject) => {
            const timeout =
                timeoutMs === null
                    ? undefined
                    : setTimeout(() => {
                          this.pending.delete(id)
                          reject(new Error(`${this.name} ${method} timed out`))
                      }, timeoutMs)
            this.pending.set(id, { resolve, reject, timeout })
            this.write(message).catch(err => {
                if (timeout) clearTimeout(timeout)
                this.pending.delete(id)
                reject(err)
            })
        })
    }

    /** One generated call; its raw result is consumed before the receiver reuses backing. */
    callHistory(method: RoutingHistoryMethod, consume: (raw: Uint8Array) => boolean): Promise<void> {
        const socket = this.socket
        if (!isHistoryCall(method)) return Promise.reject(new Error('history-invalid'))
        if (!socket) return Promise.reject(new Error('history-invalid'))
        if (this.historyPending || historyControlBusy) return Promise.reject(new JsonRpcResponseError(-32096, 'routing history busy; retry'))
        if (this.historyIssued === 18446744073709551615n) return Promise.reject(new Error('history-exhausted'))
        const ordinal = ++this.historyIssued
        const frame = `{"jsonrpc":"2.0","id":"jh:${ordinal}","method":"${method}","params":{}}\n`
        const length = new TextEncoder().encodeInto(frame, historyControl).written
        historyControlBusy = true
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                if (this.historyPending?.ordinal !== ordinal || this.socket !== socket) return
                this.historyPending = null
                reject(new Error('history-invalid'))
            }, 10000)
            this.historyPending = { ordinal, method, consume, resolve, reject, timer }
            socket.write(historyControl.subarray(0, length), error => {
                // This fixed write slot belongs to actual completion, across
                // logical timeout/quarantine and even socket replacement.
                historyControlBusy = false
                if (error && this.socket === socket) socket.destroy(error)
                historyWake?.()
            })
        })
    }
    get historyWritable(): boolean { return !historyControlBusy }
    cancelHistory(): void { this.rejectHistory() }
    /** Borrowed bytes must be consumed synchronously; they never enter an event queue. */
    setHistoryHandlers(handlers: HistoryHandlers | null): void {
        this.historyHandlers = handlers
        if (handlers) {
            historyWakeOwner = this
            historyWake = () => this.historyHandlers?.writable()
        } else if (historyWakeOwner === this) { historyWakeOwner = null; historyWake = null }
    }

    /**
     * Fire a request without waiting for the response. Use
     * {@link sendWithResponse} only when a caller must observe admission. The
     * backend runs it in its own goroutine and reports through notifications
     * (`engine:install-progress`, `engine:state-changed`, `errors:report`) per
     * the reactive command model — so there is no result to await and, crucially,
     * no short request timeout to fabricate a failure on long-running operations.
     */
    send(method: string, params?: JsonValue): Promise<void> {
        if (isHistoryCall(method)) return Promise.reject(new Error('history-invalid'))
        const id = ++this.nextId
        return this.write({ jsonrpc: '2.0', id, method, params })
    }

    /** Observe a long-running request's eventual response without imposing a timeout. */
    sendWithResponse(method: string, params?: JsonValue): Promise<void> {
        return this.call(method, params, null).then(() => undefined)
    }

    notify(method: string, params?: JsonValue): Promise<void> {
        if (isHistoryCall(method)) return Promise.reject(new Error('history-invalid'))
        return this.write({ jsonrpc: '2.0', method, params })
    }

    /** Reply to a service-initiated request (see {@link JsonRpcInboundRequest}). */
    respond(id: JsonRpcId, result: JsonValue): Promise<void> {
        return this.write({ jsonrpc: '2.0', id, result })
    }

    respondError(id: JsonRpcId, code: number, message: string): Promise<void> {
        return this.write({ jsonrpc: '2.0', id, error: { code, message } })
    }

    /**
     * Close the connection and leave the service running. No `shutdown` is sent:
     * the service answers one locally and the broker keeps serving the other
     * clients and the inference already in flight.
     */
    async detach(): Promise<void> {
        this.closed = true
        this.started = false
        this.epoch++
        this.cancelReconnectWait?.()
        this.failPending(new Error(`${this.name} detached before replying`))
        this.setHistoryHandlers(null)
        const socket = this.socket
        this.socket = null
        this.rawFrame = null
        this.rawLength = 0
        this.historyIssued = 0n
        if (!socket) return
        await new Promise<void>(resolve => {
            socket.once('close', () => resolve())
            socket.destroy()
        })
    }

    /**
     * Ask the service to stop (it stops the broker cleanly, then exits), then
     * detach. The service answers once the broker has exited, which can take up
     * to ~20 s, and then drops every client, so a connection closed before the
     * answer arrives also means it stopped.
     */
    async stopService(timeoutMs = 30_000): Promise<void> {
        // From here a closed connection is the expected result, not a drop.
        this.closed = true
        try {
            await this.call('service/stop', null, timeoutMs)
        } catch (err) {
            log.warn({
                sublevel: this.name,
                message: `service/stop did not complete cleanly: ${getErrorString(err)}`
            })
        }
        await this.detach()
    }

    /** One connection attempt; true when connected. */
    private async connectOnce(
        epoch: number,
        connect: () => Promise<Duplex> = this.connect
    ): Promise<boolean> {
        let socket: Duplex
        try {
            socket = await connect()
        } catch (err) {
            log.error({
                sublevel: this.name,
                message: `Could not attach to the service: ${getErrorString(err)}`
            })
            return false
        }
        if (this.epoch !== epoch || this.closed) {
            socket.destroy()
            return true
        }
        this.attach(socket, epoch)
        return true
    }

    private attach(socket: Duplex, epoch: number): void {
        this.socket = socket
        this.rawFrame = new Uint8Array(ROUTING_FRAME_BYTES)
        this.rawLength = 0
        this.historyIssued = 0n
        this.historyQuarantined = false
        socket.on('data', (chunk: Buffer) => {
            if (this.socket !== socket || !this.rawFrame || socket.destroyed) return
            const rawFrame = this.rawFrame
            for (const byte of chunk) {
                if (this.rawLength === ROUTING_FRAME_BYTES || (byte !== 10 && this.rawLength === ROUTING_FRAME_BYTES - 1)) {
                    // The existing close owner fails pending work and schedules
                    // its one reconnect. No input bytes enter diagnostics.
                    socket.destroy()
                    return
                }
                rawFrame[this.rawLength++] = byte
                if (byte === 10) {
                    this.handleRawFrame(rawFrame.subarray(0, this.rawLength - 1))
                    this.rawLength = 0
                    if (this.socket !== socket || socket.destroyed) return
                }
            }
        })
        // An error always precedes a close that matters; handling close alone
        // keeps a single path, and the listener stops Node treating it as fatal.
        socket.on('error', err => {
            log.warn({
                sublevel: this.name,
                message: `Service connection error: ${getErrorString(err)}`
            })
        })
        socket.on('close', () => {
            if (this.socket !== socket) return
            this.socket = null
            this.rawFrame = null
            this.rawLength = 0
            this.historyIssued = 0n
            this.setHistoryHandlers(null)
            this.failPending(new Error(`${this.name} connection closed before replying`))
            if (this.closed) return
            this.emit('disconnected')
            void this.reconnect(epoch)
        })
    }

    private rejectHistory(): void {
        const pending = this.historyPending
        this.historyPending = null
        if (pending) { clearTimeout(pending.timer); pending.reject(new Error('history-invalid')) }
    }

    private handleRawFrame(raw: Uint8Array): void {
        if (raw.length === 0) return
        const selected = this.historyProjector.frame(raw, this.historyPending?.ordinal ?? null, this.historyIssued)
        if (selected.kind === 'ordinary') {
            if (raw.length + 1 > 8 * 1024 * 1024) {
                this.emit('log', { source: this.name, stream: 'stdout', text: 'history-unknown limit-exceeded 1' })
                return
            }
            this.handleLine(new TextDecoder().decode(raw))
            return
        }
        if (selected.kind === 'discard') return
        if (selected.kind === 'refuse') {
            const socket = this.socket
            if (!socket) return
            const id = new TextDecoder().decode(selected.id)
            const frame = `{"jsonrpc":"2.0","id":${id},"error":{"code":${selected.code},"message":"${selected.message}"}}\n`
            // A fixed E-sized refusal is an ordinary RPC reply, copied before
            // this raw lease ends. It follows the pre-existing ordinary writer;
            // no history output bank, refusal queue or per-ID state is added.
            void this.writeFrame(socket, frame).catch(() => {})
            return
        }
        if (selected.kind === 'quarantine') {
            this.historyQuarantined = true
            this.rejectHistory()
            this.historyHandlers?.invalid()
            return
        }
        if (selected.kind === 'error') {
            const pending = this.historyPending
            this.historyPending = null
            if (pending) {
                clearTimeout(pending.timer)
                pending.reject(new JsonRpcResponseError(selected.code, selected.message))
            }
            return
        }
        try {
            if (selected.kind === 'notification') {
                if (!this.historyQuarantined && this.historyHandlers?.notification(selected.value) !== undefined) throw new Error('history-invalid')
            } else {
                const pending = this.historyPending
                if (!pending) return
                if (pending.consume(selected.value) !== true) { this.rejectHistory(); return }
                this.historyPending = null
                if (pending.method !== 'routing:unsubscribe') this.historyQuarantined = false
                clearTimeout(pending.timer)
                pending.resolve()
            }
        } catch {
            this.historyQuarantined = true
            this.rejectHistory()
            this.historyHandlers?.invalid()
        }
    }

    private async reconnect(epoch: number): Promise<void> {
        for (const delay of this.reconnectDelaysMs) {
            await this.waitToReconnect(delay)
            if (this.epoch !== epoch || this.closed) return
            if (await this.connectOnce(epoch)) return
        }
        const delay = this.reconnectDelaysMs.at(-1) ?? 10_000
        while (this.epoch === epoch && !this.closed) {
            await this.waitToReconnect(delay)
            if (this.epoch !== epoch || this.closed) return
            if (await this.connectOnce(epoch, this.dial)) return
        }
    }

    private waitToReconnect(delay: number): Promise<void> {
        return new Promise(resolve => {
            const finish = (): void => {
                clearTimeout(timer)
                this.cancelReconnectWait = undefined
                resolve()
            }
            const timer = setTimeout(finish, delay)
            this.cancelReconnectWait = finish
        })
    }

    private giveUp(): void {
        this.started = false
        this.emit('exit', { source: this.name, code: null })
    }

    private async write(message: JsonRpcMessage): Promise<void> {
        const socket = this.socket
        if (!socket) throw new Error(`${this.name} is not running`)

        return this.writeFrame(socket, message)
    }
    private writeFrame(socket: Duplex, frame: string | JsonRpcMessage): Promise<void> {
        this.writeTail = this.writeTail.then(
            () =>
                new Promise<void>((resolve, reject) => {
                    socket.write(typeof frame === 'string' ? frame : `${JSON.stringify(frame)}\n`, err => {
                        if (err) {
                            reject(err)
                            return
                        }
                        resolve()
                    })
                })
        )
        return this.writeTail
    }

    private handleLine(line: string): void {
        if (!line) return

        const message = parseMessage(line)

        // Only guarded ordinary frames can reach protocol trace. A failed parse
        // never exposes its input. Redact the trace copy while preserving the
        // ordinary message consumed by the pairing flow.
        if (message && this.traceProtocol) {
            this.emit('log', {
                source: this.name,
                stream: 'stdout',
                text: redactSensitiveLogText(line)
            })
        }
        if (!message) return

        if (message.id !== undefined && message.method) {
            this.emit('request', {
                source: this.name,
                method: message.method,
                params: message.params,
                id: message.id
            })
            return
        }

        if (message.id !== undefined) {
            this.deliverResponse(message)
            return
        }

        if (message.method) this.handleNotification(message.method, message.params)
    }

    private handleNotification(method: string, params: JsonValue | undefined): void {
        if (method === SERVICE_LOG_METHOD) {
            this.emitServiceLog(params)
            return
        }
        if (method === SERVICE_BROKER_RESTARTED_METHOD) {
            this.emit('broker-restarted')
            return
        }
        this.emit('notification', { source: this.name, method, params })
    }

    /**
     * Broker stderr, relayed by the service. It is redacted here exactly as a
     * child's stderr line was when this process owned the broker, and attributed
     * to this client's name: the stream carries the log of every worker, and the
     * debug panel has always filed it under the broker.
     */
    private emitServiceLog(params: JsonValue | undefined): void {
        if (!isJsonObject(params) || typeof params.text !== 'string' || !params.text) return
        this.emit('log', {
            source: this.name,
            stream: params.stream === 'stdout' ? 'stdout' : 'stderr',
            text: redactSensitiveLogText(params.text)
        })
    }

    private deliverResponse(message: JsonRpcMessage): void {
        if (typeof message.id !== 'number') return
        const pending = this.pending.get(message.id)
        if (!pending) return

        this.pending.delete(message.id)
        if (pending.timeout) clearTimeout(pending.timeout)

        if (message.error) {
            pending.reject(new JsonRpcResponseError(message.error.code, message.error.message))
            return
        }

        pending.resolve(message.result)
    }

    private failPending(error: Error): void {
        this.rejectHistory()
        for (const pending of Array.from(this.pending.values())) {
            if (pending.timeout) clearTimeout(pending.timeout)
            pending.reject(error)
        }
        this.pending.clear()
    }
}
