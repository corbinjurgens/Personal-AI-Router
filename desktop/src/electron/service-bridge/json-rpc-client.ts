// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import readline from 'readline'
import { EventEmitter } from 'events'
import type { Duplex } from 'stream'
import getErrorString from '@/shared/utils/get-error-string'
import { createStructuredLogger } from '@/shared/utils/log'
import { redactSensitiveLogText } from '@/electron/redact-log'

export type JsonValue = string | number | boolean | null | JsonObject | JsonValue[]

export interface JsonObject {
    [key: string]: JsonValue
}

interface JsonRpcError {
    code: number
    message: string
    data?: JsonValue
}

export class JsonRpcResponseError extends Error {}

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
 * reconnects with backoff, calling `connect` again each time, and reports
 * `exit` only when every attempt has failed.
 */
export class JsonRpcSocketClient extends EventEmitter<JsonRpcClientEvents> {
    readonly name: string

    private readonly connect: () => Promise<Duplex>
    private readonly reconnectDelaysMs: readonly number[]
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

    constructor(
        name: string,
        connect: () => Promise<Duplex>,
        reconnectDelaysMs: readonly number[] = DEFAULT_RECONNECT_DELAYS_MS
    ) {
        super()
        this.name = name
        this.connect = connect
        this.reconnectDelaysMs = reconnectDelaysMs
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

    /**
     * Fire a request without waiting for the response. Use
     * {@link sendWithResponse} only when a caller must observe admission. The
     * backend runs it in its own goroutine and reports through notifications
     * (`engine:install-progress`, `engine:state-changed`, `errors:report`) per
     * the reactive command model — so there is no result to await and, crucially,
     * no short request timeout to fabricate a failure on long-running operations.
     */
    send(method: string, params?: JsonValue): Promise<void> {
        const id = ++this.nextId
        return this.write({ jsonrpc: '2.0', id, method, params })
    }

    /** Observe a long-running request's eventual response without imposing a timeout. */
    sendWithResponse(method: string, params?: JsonValue): Promise<void> {
        return this.call(method, params, null).then(() => undefined)
    }

    notify(method: string, params?: JsonValue): Promise<void> {
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
        this.failPending(new Error(`${this.name} detached before replying`))
        const socket = this.socket
        this.socket = null
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
    private async connectOnce(epoch: number): Promise<boolean> {
        let socket: Duplex
        try {
            socket = await this.connect()
        } catch (err) {
            log.error({
                sublevel: this.name,
                message: `Could not attach to the service: ${getErrorString(err)}`
            })
            return false
        }
        if (this.epoch !== epoch) {
            socket.destroy()
            return true
        }
        this.attach(socket, epoch)
        return true
    }

    private attach(socket: Duplex, epoch: number): void {
        this.socket = socket
        readline.createInterface({ input: socket }).on('line', line => {
            if (this.socket === socket) this.handleLine(line)
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
            this.failPending(new Error(`${this.name} connection closed before replying`))
            if (this.closed) return
            this.emit('disconnected')
            void this.reconnect(epoch)
        })
    }

    private async reconnect(epoch: number): Promise<void> {
        for (const delay of this.reconnectDelaysMs) {
            await new Promise<void>(resolve => setTimeout(resolve, delay))
            if (this.epoch !== epoch) return
            if (await this.connectOnce(epoch)) return
        }
        if (this.epoch === epoch) this.giveUp()
    }

    private giveUp(): void {
        this.started = false
        this.emit('exit', { source: this.name, code: null })
    }

    private async write(message: JsonRpcMessage): Promise<void> {
        const socket = this.socket
        if (!socket) throw new Error(`${this.name} is not running`)

        this.writeTail = this.writeTail.then(
            () =>
                new Promise<void>((resolve, reject) => {
                    socket.write(`${JSON.stringify(message)}\n`, err => {
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

        // A protocol frame is logged only while tracing, and then it is redacted
        // first: it costs a redaction pass, a debug panel entry, and a file line
        // per frame. A line that is not a frame is not protocol traffic, so it is
        // always handed over. Redact only what is logged; `message` stays intact
        // for the handling below so the pairing flow still receives the real PIN
        // it has to display.
        if (!message || this.traceProtocol) {
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
            pending.reject(
                new JsonRpcResponseError(`${message.error.code}: ${message.error.message}`)
            )
            return
        }

        pending.resolve(message.result)
    }

    private failPending(error: Error): void {
        for (const pending of Array.from(this.pending.values())) {
            if (pending.timeout) clearTimeout(pending.timeout)
            pending.reject(error)
        }
        this.pending.clear()
    }
}
