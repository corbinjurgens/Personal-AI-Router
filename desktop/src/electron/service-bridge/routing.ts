// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RoutingRequestKey } from '@/shared/types/node-policy'
import type {
    RoutingActiveParams,
    RoutingActiveResult,
    RoutingCancelParams,
    RoutingCancelResult
} from '@/shared/types/routing'
import type { JsonObject, JsonValue, JsonRpcSocketClient } from './json-rpc-client'
import { getRoutingDelivery } from './routing-delivery'

function strictObject(value: JsonValue | undefined, fields: string[]): JsonObject {
    if (
        !value ||
        typeof value !== 'object' ||
        Array.isArray(value) ||
        Object.keys(value).some(key => !fields.includes(key)) ||
        fields.some(key => value[key] === undefined || value[key] === null)
    ) {
        throw new Error('Invalid routing control fields')
    }
    return value
}

function identity(value: JsonValue | undefined): string {
    if (
        typeof value !== 'string' ||
        /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.exec(value)?.[0] !== value
    ) {
        throw new Error('Invalid routing identity')
    }
    return value
}

function requestKey(value: JsonValue | undefined): RoutingRequestKey {
    const body = strictObject(value, ['originNodeId', 'proxyRunId', 'requestId'])
    const requestId = body.requestId
    if (
        typeof requestId !== 'string' ||
        /^[1-9][0-9]*$/.exec(requestId)?.[0] !== requestId ||
        requestId.length > 19 ||
        BigInt(requestId) > 9223372036854775807n
    ) {
        throw new Error('Invalid routing requestId')
    }
    return {
        originNodeId: identity(body.originNodeId),
        proxyRunId: identity(body.proxyRunId),
        requestId
    }
}

/** Validate decoded params before the broker call, retaining unexpected fields for rejection. */
export function routingCancelParams(value: RoutingCancelParams | undefined): JsonObject {
    const body = strictObject(
        value ? { ...value, request: value.request ? { ...value.request } : null } : undefined,
        ['request', 'regenerate']
    )
    const request = requestKey(body.request)
    if (typeof body.regenerate !== 'boolean') throw new Error('Invalid routing regenerate flag')
    return { request: { ...request }, regenerate: body.regenerate }
}

export function routingActiveParams(value: RoutingActiveParams | undefined): JsonObject {
    strictObject(value, [])
    return {}
}

export function parseRoutingCancelResult(value: JsonValue | undefined): RoutingCancelResult {
    const { status } = strictObject(value, ['status'])
    if (
        status !== 'not_found' &&
        status !== 'conflict_committed' &&
        status !== 'regenerating' &&
        status !== 'cancelled'
    ) {
        throw new Error('Invalid routing cancellation status')
    }
    return { status }
}

export function parseRoutingActiveResult(value: JsonValue | undefined): RoutingActiveResult {
    const { requests } = strictObject(value, ['requests'])
    if (!Array.isArray(requests)) throw new Error('Invalid routing active requests')
    return { requests: requests.map(requestKey) }
}

export function applyRoutingHistoryInitial(raw: Uint8Array): void {
    getRoutingDelivery().initialResult(raw)
}

export function applyRoutingHistoryChanged(raw: Uint8Array): void {
    getRoutingDelivery().record(raw)
}

/** Sender-owned IPC handles these channels before generic service dispatch. */
export function historySenderRequired(): never { throw new Error('history-invalid') }

type HistoryPeer = Pick<JsonRpcSocketClient, 'callHistory' | 'cancelHistory' | 'setHistoryHandlers' | 'historyWritable'>

/** One desktop subscription and one timer, retargeted to native cleanup when needed. */
export class RoutingHistorySubscription {
    constructor(private readonly apply: (raw: Uint8Array, initial: boolean) => undefined = (raw, initial) => {
        if (initial) applyRoutingHistoryInitial(raw)
        else applyRoutingHistoryChanged(raw)
        return undefined
    }) {}
    private peer: HistoryPeer | null = null
    private generation = 0
    private intent = 0
    private paused = true
    private wanted = false
    private needsInitial = true
    private needsUnsubscribe = false
    private flight = false
    private waitingWrite = false
    private exhausted = false
    private timer: ReturnType<typeof setTimeout> | null = null
    private retry = 0

    setWanted(wanted: boolean): void {
        if (this.wanted === wanted) return
        this.wanted = wanted
        this.needsInitial = true
        if (!wanted && this.peer) this.needsUnsubscribe = true
        this.retireIntent()
        this.waitingWrite = this.peer !== null && !this.peer.historyWritable
        this.drive()
    }
    connect(peer: HistoryPeer): void {
        if (this.peer === peer) { this.paused = false; this.drive(); return }
        // A replacement cannot erase cleanup belonging to an unfenced native
        // connection. The actual close owner must call disconnected first.
        if (this.peer || this.exhausted) return
        this.peer = peer
        this.paused = false
        const generation = this.generation
        peer.setHistoryHandlers({
            notification: raw => {
                if (this.peer === peer && generation === this.generation && this.wanted && !this.paused && !this.exhausted && this.apply(raw, false) !== undefined) throw new Error('history-invalid')
                return undefined
            },
            invalid: () => {
                if (this.peer !== peer || generation !== this.generation || !this.wanted || this.paused || this.exhausted) return
                this.needsInitial = true; getRoutingDelivery().setLoading(true)
                this.scheduleRetry()
            },
            writable: () => {
                if (this.peer !== peer || generation !== this.generation) return
                this.waitingWrite = false
                this.drive()
            }
        })
        this.drive()
    }
    /** Broker replacement fences source callbacks; the service connection survives. */
    sourceReset(): void {
        this.paused = true; this.needsInitial = true
        this.retireIntent()
        getRoutingDelivery().reset()
        this.drive()
    }
    /** Only an actual connection close, or a connection that never opened. */
    disconnected(): void {
        const canAdvance = this.generation < Number.MAX_SAFE_INTEGER
        if (canAdvance) this.generation++
        this.cancelRetry()
        this.peer?.setHistoryHandlers(null)
        this.peer?.cancelHistory()
        this.peer = null
        this.intent = 0; this.paused = true
        this.flight = false; this.waitingWrite = false; this.exhausted = !canAdvance
        this.needsInitial = true; this.needsUnsubscribe = false
        getRoutingDelivery().reset()
    }
    private retireIntent(): void {
        this.cancelRetry()
        if (this.intent === Number.MAX_SAFE_INTEGER) this.exhausted = true
        else this.intent++
        // Logical cancellation never releases the client's active E writer.
        // Keep flight until its one existing promise chain settles; changes
        // only update scalar desired state, without another output waiter.
        this.peer?.cancelHistory()
    }
    private cancelRetry(): void {
        if (this.timer) clearTimeout(this.timer)
        this.timer = null; this.retry = 0
    }
    private scheduleRetry(): void {
        if (this.timer || !this.peer || this.exhausted || (!this.needsUnsubscribe && (!this.wanted || this.paused))) return
        const generation = this.generation, intent = this.intent, delay = this.retry === 0 ? 250 : this.retry === 1 ? 500 : 1000
        this.retry = Math.min(this.retry + 1, 2)
        this.timer = setTimeout(() => {
            this.timer = null
            if (generation === this.generation && intent === this.intent) this.drive()
        }, delay)
    }
    private drive(): void {
        const peer = this.peer
        if (!peer || this.flight || this.timer || this.waitingWrite || this.exhausted) return
        if (!this.needsUnsubscribe && (!this.wanted || !this.needsInitial || this.paused)) return
        if (!peer.historyWritable) { this.waitingWrite = true; return }
        const generation = this.generation, intent = this.intent, unsubscribe = this.needsUnsubscribe
        this.flight = true
        if (!unsubscribe) getRoutingDelivery().setLoading(true)
        let retry = false
        void peer.callHistory(unsubscribe ? 'routing:unsubscribe' : 'routing:subscribe', raw => {
            if (this.peer !== peer || generation !== this.generation || intent !== this.intent || this.exhausted) return false
            if (unsubscribe) return true
            if (!this.wanted || this.paused) return false
            if (this.apply(raw, true) !== undefined) throw new Error('history-invalid')
            return true
        }).then(() => {
            if (this.peer !== peer || generation !== this.generation || intent !== this.intent || this.exhausted) return
            if (unsubscribe) this.needsUnsubscribe = false
            else this.needsInitial = false
            this.cancelRetry()
        }).catch(error => {
            if (this.peer !== peer || generation !== this.generation || intent !== this.intent || this.exhausted) return
            if (error instanceof Error && error.message === 'history-exhausted') { this.exhausted = true; this.cancelRetry(); return }
            if (!peer.historyWritable) this.waitingWrite = true
            else retry = true
            // Busy is not native completion. The same timer now retries cleanup
            // until a matching success or actual connection destruction.
        }).finally(() => {
            if (this.peer !== peer || generation !== this.generation) return
            this.flight = false
            if (intent !== this.intent) this.drive()
            else if (retry) this.scheduleRetry()
            else this.drive()
        })
    }
}
