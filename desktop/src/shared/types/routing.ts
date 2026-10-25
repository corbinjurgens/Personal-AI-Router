// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RoutingRequestKey } from '@/shared/types/node-policy'
import type { JsonValue } from './json'

export interface RoutingCancelParams {
    request: RoutingRequestKey
    regenerate: boolean
}

export interface RoutingCancelResult {
    status: 'not_found' | 'conflict_committed' | 'regenerating' | 'cancelled'
}

export type RoutingActiveParams = Record<string, never>

export interface RoutingActiveResult {
    requests: RoutingRequestKey[]
}

export type RoutingHistoryState = 'routing' | 'streaming' | 'completed' | 'failed' | 'cancelled'
export type RoutingHistoryAccounting = 'none' | 'pending' | 'recorded' | 'retained' | 'failed' | 'unknown'
export type RoutingHistoryReason = '' | 'unknown' | 'client-gone' | 'cancel' | 'regenerate' |
    'node-paused' | 'proxy-shutdown' | 'proxy-exited' | 'target-left' | 'transport' |
    'upstream-status' | 'no-content' | 'stream-interrupted' | 'upstream-error' |
    'admission-paused' | 'admission-draining' | 'admission-busy' | 'admission-no-fit' |
    'admission-engine-off' | 'admission-wake-timeout' | 'engine-not-running' | 'request-invalid' |
    'no-owner' | 'regenerate-exhausted' | 'attempts-exhausted' | 'deadline' | 'request-ceiling' |
    'account-ceiling' | 'account-cooldown' | 'account-frozen' | 'account-credential' |
    'account-exhausted' | 'account-conflict' | 'ledger-unavailable' | 'deployment-quarantined' | 'peer-framing'

export interface RoutingHistoryDestination {
    kind: 'local' | 'api'
    model: string
    nodeId: string | null
    engine: string | null
    deploymentId: string | null
    accountId: string | null
}

export interface RoutingHistoryAccount {
    state: 'ok' | 'cooldown' | 'frozen' | 'credential-failure' | 'exhausted' | 'probe' | 'conflict' | 'unknown'
    heldNanoUsd: string | null
    chargedNanoUsd: string | null
    accounting: RoutingHistoryAccounting
}

export interface RoutingHistoryAttempt {
    index: number
    startedAt: number
    endedAt: number | null
    destination: RoutingHistoryDestination
    outcome: 'pending' | 'committed' | 'rejected' | 'failed' | 'redispatched' | 'interrupted' | 'cancelled'
    reason: RoutingHistoryReason
    status: number | null
    waits: { admissionMs: number; firstByteMs: number | null }
    continuationOf: number | null
    account: RoutingHistoryAccount | null
}

export interface RoutingHistoryRecord {
    schema: 1
    request: RoutingRequestKey
    workload: {
        originNodeId: string; proxyRunId: string; engine: string; id: string
        attemptIndex: number; attached: boolean
    } | null
    seq: number
    requested: { model: string; tier: '' | 'weak' | 'medium' | 'strong'; mode: 'chat' | 'completion' | 'embedding' | 'other' }
    createdAt: number
    endedAt: number | null
    state: RoutingHistoryState
    reason: RoutingHistoryReason
    status: number | null
    served: RoutingHistoryDestination | null
    recoverable: boolean
    spend: {
        currency: 'USD'; ceilingNanoUsd: string | null; grantedNanoUsd: string | null
        heldNanoUsd: string | null; chargedNanoUsd: string | null; accounting: RoutingHistoryAccounting
    }
    droppedAttempts: number
    attempts: RoutingHistoryAttempt[]
}

export interface RoutingHistoryInitial { requests: RoutingHistoryRecord[] }

export interface RoutingHistoryReady { generation: number }
export type RoutingHistoryCommit = { generation: number; revision: number } | { generation: number; loading: boolean }
export interface RoutingHistoryPull { generation: number; initial: boolean }
export interface RoutingHistoryConsumed { generation: number; lease: number }
export interface RoutingHistoryConsumption { state: 'accepted' | 'retired' | 'stale'; generation: number }
export type RoutingHistoryTransfer =
    | { state: 'busy'; generation: number; loading?: boolean }
    | { state: 'retired'; generation: number }
    | { state: 'data'; generation: number; lease: number; initial: true; index: number; bytes: Uint8Array }
    | { state: 'data'; generation: number; lease: number; initial: false; bytes: Uint8Array }
    | { state: 'end'; generation: number; lease: number; count: number; retired: boolean }

type ReadonlyHistory<T> = T extends readonly (infer Item)[] ? readonly ReadonlyHistory<Item>[] : T extends object ? { readonly [K in keyof T]: ReadonlyHistory<T[K]> } : T
export type RoutingHistoryOutput = ReadonlyHistory<RoutingHistoryRecord>
/** Return undefined explicitly. Promise-returning consumers are not output loans. */
export type RoutingHistoryConsumer = (record: RoutingHistoryOutput) => undefined

export interface RoutingHistoryShape {
    kind: 'object' | 'array' | 'string' | 'number' | 'boolean'
    nullable?: boolean
    fields?: readonly [string, RoutingHistoryShape][]
    item?: RoutingHistoryShape
    limit?: number
    enum?: 'destination' | 'state' | 'outcome' | 'reason' | 'accounting' | 'accountState' | 'tier' | 'mode' | 'currency'
}
const text: RoutingHistoryShape = { kind: 'string', limit: 512 }
const identityText: RoutingHistoryShape = { kind: 'string', limit: 128 }
const shortText: RoutingHistoryShape = { kind: 'string', limit: 32 }
const moneyText: RoutingHistoryShape = { kind: 'string', limit: 19, nullable: true }
const integer: RoutingHistoryShape = { kind: 'number' }
const nullableText: RoutingHistoryShape = { ...identityText, nullable: true }
const nullableInteger: RoutingHistoryShape = { ...integer, nullable: true }
const flag: RoutingHistoryShape = { kind: 'boolean' }
const keyShape: RoutingHistoryShape = { kind: 'object', fields: [['originNodeId', identityText], ['proxyRunId', identityText], ['requestId', { kind: 'string', limit: 19 }]] }
const destinationShape: RoutingHistoryShape = { kind: 'object', fields: [
    ['kind', { kind: 'string', limit: 5, enum: 'destination' }], ['model', text], ['nodeId', nullableText], ['engine', nullableText],
    ['deploymentId', nullableText], ['accountId', nullableText]
] }
const accountShape: RoutingHistoryShape = { kind: 'object', nullable: true, fields: [
    ['state', { kind: 'string', limit: 18, enum: 'accountState' }], ['heldNanoUsd', moneyText], ['chargedNanoUsd', moneyText], ['accounting', { kind: 'string', limit: 8, enum: 'accounting' }]
] }
const attemptShape: RoutingHistoryShape = { kind: 'object', fields: [
    ['index', integer], ['startedAt', integer], ['endedAt', nullableInteger], ['destination', destinationShape],
    ['outcome', { kind: 'string', limit: 13, enum: 'outcome' }], ['reason', { ...shortText, enum: 'reason' }], ['status', nullableInteger],
    ['waits', { kind: 'object', fields: [['admissionMs', integer], ['firstByteMs', nullableInteger]] }],
    ['continuationOf', nullableInteger], ['account', accountShape]
] }
export const routingHistoryRecordShape: RoutingHistoryShape = { kind: 'object', fields: [
    ['schema', integer], ['request', keyShape],
    ['workload', { kind: 'object', nullable: true, fields: [
        ['originNodeId', identityText], ['proxyRunId', identityText], ['engine', identityText], ['id', identityText], ['attemptIndex', integer], ['attached', flag]
    ] }], ['seq', integer],
    ['requested', { kind: 'object', fields: [['model', text], ['tier', { kind: 'string', limit: 6, enum: 'tier' }], ['mode', { kind: 'string', limit: 10, enum: 'mode' }]] }],
    ['createdAt', integer], ['endedAt', nullableInteger], ['state', { kind: 'string', limit: 9, enum: 'state' }], ['reason', { ...shortText, enum: 'reason' }],
    ['status', nullableInteger], ['served', { ...destinationShape, nullable: true }], ['recoverable', flag],
    ['spend', { kind: 'object', fields: [
        ['currency', { kind: 'string', limit: 3, enum: 'currency' }], ['ceilingNanoUsd', moneyText], ['grantedNanoUsd', moneyText],
        ['heldNanoUsd', moneyText], ['chargedNanoUsd', moneyText], ['accounting', { kind: 'string', limit: 8, enum: 'accounting' }]
    ] }], ['droppedAttempts', integer], ['attempts', { kind: 'array', item: attemptShape, limit: 32 }]
] }
export const routingHistoryInitialShape: RoutingHistoryShape = {
    kind: 'object', fields: [['requests', { kind: 'array', item: routingHistoryRecordShape, limit: 256 }]]
}

const reasons: readonly RoutingHistoryReason[] = [
    '', 'unknown', 'client-gone', 'cancel', 'regenerate', 'node-paused', 'proxy-shutdown',
    'proxy-exited', 'target-left', 'transport', 'upstream-status', 'no-content',
    'stream-interrupted', 'upstream-error', 'admission-paused', 'admission-draining',
    'admission-busy', 'admission-no-fit', 'admission-engine-off', 'admission-wake-timeout',
    'engine-not-running', 'request-invalid', 'no-owner', 'regenerate-exhausted',
    'attempts-exhausted', 'deadline', 'request-ceiling', 'account-ceiling', 'account-cooldown',
    'account-frozen', 'account-credential', 'account-exhausted', 'account-conflict',
    'ledger-unavailable', 'deployment-quarantined', 'peer-framing'
]
const accounting: readonly RoutingHistoryAccounting[] = ['none', 'pending', 'recorded', 'retained', 'failed', 'unknown']
const historyEnums: Record<NonNullable<RoutingHistoryShape['enum']>, readonly string[]> = {
    destination: ['local', 'api'], state: ['routing', 'streaming', 'completed', 'failed', 'cancelled'],
    outcome: ['pending', 'committed', 'rejected', 'failed', 'redispatched', 'interrupted', 'cancelled'],
    reason: reasons, accounting,
    accountState: ['ok', 'cooldown', 'frozen', 'credential-failure', 'exhausted', 'probe', 'conflict', 'unknown'],
    tier: ['', 'weak', 'medium', 'strong'], mode: ['chat', 'completion', 'embedding', 'other'], currency: ['USD']
}

function invalid(): never { throw new Error('invalid-value') }
function isObject(value: JsonValue | RoutingHistoryOutput | undefined): value is { [key: string]: JsonValue } {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
}
function object(value: JsonValue | RoutingHistoryOutput | undefined): { [key: string]: JsonValue } {
    if (!isObject(value)) throw new Error('invalid-shape')
    return value
}
function choice<T extends string>(value: JsonValue | undefined, choices: readonly T[]): T {
    for (const option of choices) if (option === value) return option
    return invalid()
}
function number(value: JsonValue | undefined): number {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) return invalid()
    return value
}
function nullableNumber(value: JsonValue | undefined): number | null { return value === null ? null : number(value) }
function boolean(value: JsonValue | undefined): boolean {
    if (typeof value !== 'boolean') return invalid()
    return value
}
function id(value: JsonValue | undefined): string {
    if (typeof value !== 'string' || /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.exec(value)?.[0] !== value) return invalid()
    return value
}
function decimal(value: JsonValue | undefined, positive = false): string {
    if (typeof value !== 'string' || /^(0|[1-9][0-9]*)$/.exec(value)?.[0] !== value ||
        value.length > 19 || (value.length === 19 && value > '9223372036854775807') || (positive && value === '0')) return invalid()
    return value
}
function money(value: JsonValue | undefined): string | null { return value === null ? null : decimal(value) }
function model(value: JsonValue | undefined, empty: boolean): string {
    if (typeof value !== 'string' || (!empty && value === '') || !value.isWellFormed()) return invalid()
    let bytes = 0
    for (let i = 0; i < value.length; i++) {
        const c = value.charCodeAt(i)
        if (c < 32 || (c >= 127 && c <= 159)) return invalid()
        if (c < 128) bytes++
        else if (c < 2048) bytes += 2
        else if (c >= 0xd800 && c <= 0xdbff) { bytes += 4; i++ }
        else bytes += 3
        if (bytes > 512) return invalid()
    }
    return value
}
function status(value: JsonValue | undefined): number | null {
    const next = nullableNumber(value)
    if (next !== null && (next < 100 || next > 599)) return invalid()
    return next
}
function destination(value: JsonValue | undefined): { [key: string]: JsonValue } {
    const v = object(value)
    const kind = choice(v.kind, ['local', 'api'])
    model(v.model, false)
    if (kind === 'local') {
        id(v.nodeId); id(v.engine)
        if (v.deploymentId !== null || v.accountId !== null) return invalid()
    } else {
        id(v.deploymentId); id(v.accountId)
        if (v.nodeId !== null || v.engine !== null) return invalid()
    }
    return v
}
function sameDestination(a: { [key: string]: JsonValue }, b: { [key: string]: JsonValue }): boolean {
    return a.kind === b.kind && a.model === b.model && a.nodeId === b.nodeId && a.engine === b.engine &&
        a.deploymentId === b.deploymentId && a.accountId === b.accountId
}
function attempt(value: JsonValue): { [key: string]: JsonValue } {
    const v = object(value)
    const w = object(v.waits)
    if (v.account !== null) {
        const a = object(v.account)
        choice(a.state, ['ok', 'cooldown', 'frozen', 'credential-failure', 'exhausted', 'probe', 'conflict', 'unknown'])
        money(a.heldNanoUsd); money(a.chargedNanoUsd); choice(a.accounting, accounting)
    }
    number(v.index); nullableNumber(v.continuationOf)
    const start = number(v.startedAt), end = nullableNumber(v.endedAt)
    const d = destination(v.destination)
    const outcome = choice(v.outcome, ['pending', 'committed', 'rejected', 'failed', 'redispatched', 'interrupted', 'cancelled'])
    choice(v.reason, reasons); status(v.status); number(w.admissionMs); nullableNumber(w.firstByteMs)
    if ((d.kind === 'local') !== (v.account === null) || (end !== null && end < start) ||
        (outcome !== 'pending' && outcome !== 'committed' && end === null)) return invalid()
    return v
}

/** Check the one materialized known-field graph in place; never construct another DTO. */
function validRecord(value: JsonValue | RoutingHistoryRecord): value is RoutingHistoryRecord {
    const v = object(value)
    const k = object(v.request), q = object(v.requested), s = object(v.spend)
    if (v.schema !== 1 || !Array.isArray(v.attempts)) return invalid()
    if (v.attempts.length > 32) throw new Error('limit-exceeded')
    const attempts = v.attempts
    const workload = v.workload === null ? null : object(v.workload)
    if (v.workload !== null) {
        const w = object(v.workload)
        id(w.originNodeId); id(w.proxyRunId); id(w.engine); id(w.id); number(w.attemptIndex); boolean(w.attached)
    }
    const state = choice<RoutingHistoryState>(v.state, ['routing', 'streaming', 'completed', 'failed', 'cancelled'])
    const droppedAttempts = number(v.droppedAttempts)
    id(k.originNodeId); id(k.proxyRunId); decimal(k.requestId, true)
    model(q.model, attempts.length === 0 && droppedAttempts === 0 && workload === null && v.served === null && state !== 'completed' && state !== 'streaming')
    choice(q.tier, ['', 'weak', 'medium', 'strong']); choice(q.mode, ['chat', 'completion', 'embedding', 'other'])
    choice(v.reason, reasons); status(v.status)
    choice(s.currency, ['USD']); choice(s.accounting, accounting)
    money(s.ceilingNanoUsd); money(s.grantedNanoUsd); money(s.heldNanoUsd); money(s.chargedNanoUsd)
    const createdAt = number(v.createdAt), endedAt = nullableNumber(v.endedAt), recoverable = boolean(v.recoverable)
    const servedDestination = v.served === null ? null : destination(v.served)
    const terminal = state === 'completed' || state === 'failed' || state === 'cancelled'
    if (number(v.seq) === 0 || terminal !== (endedAt !== null) || (terminal && recoverable) ||
        (endedAt !== null && endedAt < createdAt) || (state === 'completed') !== (servedDestination !== null)) return invalid()
    let served = false, awaiting = false
    for (let i = 0; i < attempts.length; i++) {
        const a = attempt(attempts[i])
        const index = number(a.index), start = number(a.startedAt), end = nullableNumber(a.endedAt), predecessor = nullableNumber(a.continuationOf)
        if (index < droppedAttempts || index - droppedAttempts !== i || start < createdAt ||
            (terminal && end === null) || (endedAt !== null && (start > endedAt || (end !== null && end > endedAt)))) return invalid()
        if (predecessor !== null && (predecessor >= index ||
            (predecessor >= droppedAttempts && object(attempts[predecessor - droppedAttempts]).outcome !== 'interrupted'))) return invalid()
        if (a.outcome === 'interrupted') {
            let linked = false
            for (let j = i + 1; j < attempts.length; j++) if (object(attempts[j]).continuationOf === index) linked = true
            if (!linked) awaiting = true
        }
        if (servedDestination && a.outcome === 'committed' && end !== null && sameDestination(object(a.destination), servedDestination)) served = true
    }
    if ((servedDestination !== null && !served) || (recoverable && (state !== 'streaming' || !awaiting))) return invalid()
    if (workload) {
        if (workload.originNodeId !== k.originNodeId || workload.proxyRunId !== k.proxyRunId) return invalid()
        const index = number(workload.attemptIndex)
        if (index < droppedAttempts) { if (workload.attached) return invalid() }
        else {
            const a = object(attempts[index - droppedAttempts]), d = object(a.destination)
            if (d.kind !== 'local' || d.engine !== workload.engine) return invalid()
        }
    }
    return true
}

const cellBytes = 65536, headerBytes = 320
// Valid schema-1 records require at most 30,768 packed bytes, including the
// format tag. Private IPC loans contain one such record, never a packed window.
// Native serializer capacities/lifetimes remain a separate proof from this cap.
// The private cap never lowers R/M.
const packedRecordBytes = 32752
const emptyBytes = new Uint8Array(0)
let readingRecord = false
let outputLoan = false

/** Packed input only: unknowns were stripped at ingress, not parsed into this graph. */
class RecordReader {
    raw: Uint8Array = emptyBytes
    pos = 0
    private space(): void {
        while (this.raw[this.pos] === 32 || this.raw[this.pos] === 9 || this.raw[this.pos] === 10 || this.raw[this.pos] === 13) this.pos++
    }
    private string(limit: number): string {
        const start = this.pos
        if (this.raw[this.pos++] !== 34) throw new Error('invalid-shape')
        let escaped = false
        while (this.pos < this.raw.length) {
            const c = this.raw[this.pos++]
            if (!escaped && c === 34) {
                if (this.pos - start > limit * 6 + 2) return invalid()
                let value: JsonValue
                try { value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(this.raw.subarray(start, this.pos))) }
                catch { throw new Error('invalid-shape') }
                if (typeof value !== 'string' || !value.isWellFormed()) return invalid()
                let bytes = 0
                for (let i = 0; i < value.length; i++) {
                    const unit = value.charCodeAt(i)
                    bytes += unit < 128 ? 1 : unit < 2048 ? 2 : unit >= 0xd800 && unit <= 0xdbff ? 4 : 3
                    if (unit >= 0xd800 && unit <= 0xdbff) i++
                }
                if (bytes > limit) return invalid()
                return value
            }
            if (escaped) escaped = false
            else escaped = c === 92
            if (this.pos - start > limit * 6 + 2) return invalid()
        }
        throw new Error('invalid-shape')
    }
    private word(word: string): void {
        for (let i = 0; i < word.length; i++) if (this.raw[this.pos++] !== word.charCodeAt(i)) throw new Error('invalid-shape')
    }
    private byte(): number {
        if (this.pos === this.raw.length) throw new Error('invalid-shape')
        return this.raw[this.pos++]
    }
    packed(shape: RoutingHistoryShape): JsonValue {
        if (shape.nullable) {
            const tag = this.byte()
            if (tag === 0) return null
            if (tag !== 1) throw new Error('invalid-shape')
        }
        if (shape.kind === 'object' && shape.fields) {
            const value: { [key: string]: JsonValue } = {}
            for (const [name, child] of shape.fields) value[name] = this.packed(child)
            return value
        }
        if (shape.kind === 'array' && shape.item && shape.limit !== undefined) {
            const count = this.byte()
            if (count > shape.limit) throw new Error('limit-exceeded')
            const value = new Array<JsonValue>(count)
            for (let i = 0; i < count; i++) value[i] = this.packed(shape.item)
            return value
        }
        if (shape.kind === 'number') {
            if (this.pos + 8 > this.raw.length) throw new Error('invalid-shape')
            const value = new DataView(this.raw.buffer, this.raw.byteOffset + this.pos, 8).getFloat64(0)
            this.pos += 8
            return number(value)
        }
        if (shape.kind === 'boolean') {
            const tag = this.byte()
            if (tag > 1) throw new Error('invalid-shape')
            return tag === 1
        }
        if (shape.kind !== 'string') throw new Error('invalid-shape')
        if (shape.enum) {
            const value = historyEnums[shape.enum][this.byte()]
            if (value === undefined) return invalid()
            return value
        }
        const size = this.byte() * 256 + this.byte()
        if (size > (shape.limit ?? 512) || this.pos + size > this.raw.length) return invalid()
        let value: string
        try { value = new TextDecoder('utf-8', { fatal: true }).decode(this.raw.subarray(this.pos, this.pos + size)) }
        catch { return invalid() }
        this.pos += size
        return value
    }
    value(shape: RoutingHistoryShape, depth = 1): JsonValue {
        if (depth > 16) throw new Error('limit-exceeded')
        this.space()
        if (shape.nullable && this.raw[this.pos] === 110) { this.word('null'); return null }
        if (shape.kind === 'object' && shape.fields) {
            if (this.raw[this.pos++] !== 123) throw new Error('invalid-shape')
            const result: { [key: string]: JsonValue } = {}
            let seen = 0
            this.space()
            while (this.raw[this.pos] !== 125) {
                const name = this.string(32)
                let index = -1
                for (let i = 0; i < shape.fields.length; i++) if (shape.fields[i][0] === name) { index = i; break }
                if (index < 0) throw new Error('invalid-shape')
                if (seen & (1 << index)) throw new Error('duplicate-field')
                seen |= 1 << index
                this.space()
                if (this.raw[this.pos++] !== 58) throw new Error('invalid-shape')
                // Retain only the shared schema key. The decoded comparison
                // token is bounded scratch, never a per-record key referent.
                result[shape.fields[index][0]] = this.value(shape.fields[index][1], depth + 1)
                this.space()
                if (this.raw[this.pos] !== 44) break
                this.pos++; this.space()
                if (this.raw[this.pos] === 125) throw new Error('invalid-shape')
            }
            if (this.raw[this.pos++] !== 125 || seen !== (1 << shape.fields.length) - 1) throw new Error('invalid-shape')
            return result
        }
        if (shape.kind === 'array' && shape.item && shape.limit !== undefined) {
            if (this.raw[this.pos++] !== 91) throw new Error('invalid-shape')
            const result = new Array<JsonValue>(shape.limit)
            let count = 0
            this.space()
            while (this.raw[this.pos] !== 93) {
                if (count === shape.limit) throw new Error('limit-exceeded')
                result[count++] = this.value(shape.item, depth + 1)
                this.space()
                if (this.raw[this.pos] !== 44) break
                this.pos++; this.space()
                if (this.raw[this.pos] === 93) throw new Error('invalid-shape')
            }
            if (this.raw[this.pos++] !== 93) throw new Error('invalid-shape')
            result.length = count
            return result
        }
        if (shape.kind === 'string') {
            const value = this.string(shape.limit ?? 512)
            if (!shape.enum) return value
            // Closed values share the fixed schema table, including JSON input.
            // Only the bounded comparison token is transient per field.
            for (const option of historyEnums[shape.enum]) if (option === value) return option
            return invalid()
        }
        if (shape.kind === 'boolean') {
            if (this.raw[this.pos] === 116) { this.word('true'); return true }
            this.word('false'); return false
        }
        if (shape.kind !== 'number') throw new Error('invalid-shape')
        const start = this.pos
        while (this.raw[this.pos] >= 48 && this.raw[this.pos] <= 57) this.pos++
        const size = this.pos - start
        if (size === 0 || size > 16 || (size > 1 && this.raw[start] === 48)) return invalid()
        let value = 0
        for (let i = start; i < this.pos; i++) value = value * 10 + this.raw[i] - 48
        return number(value)
    }
    end(): void { this.space(); if (this.pos !== this.raw.length) throw new Error('invalid-shape') }
}
const recordReader = new RecordReader()

function readRecord(raw: Uint8Array, consume: RoutingHistoryConsumer, loan: boolean, packed = false): void {
    if (readingRecord) throw new Error('history-busy')
    if (raw.byteLength > (packed ? packedRecordBytes : 131072)) throw new Error('limit-exceeded')
    readingRecord = true; outputLoan = loan
    recordReader.raw = raw; recordReader.pos = 0
    try {
        if (packed && recordReader.raw[recordReader.pos++] !== 1) throw new Error('invalid-shape')
        const value = packed ? recordReader.packed(routingHistoryRecordShape) : recordReader.value(routingHistoryRecordShape)
        if (packed) { if (recordReader.pos !== raw.length) throw new Error('invalid-shape') }
        else recordReader.end()
        if (!validRecord(value)) return invalid()
        if (consume(value) !== undefined) throw new Error('history-invalid')
    } finally {
        recordReader.raw = emptyBytes; recordReader.pos = 0
        readingRecord = false; outputLoan = false
    }
}

/**
 * One synchronous decoded-record loan, including its nested referents. Do not
 * retain, clone, mutate, await, reenter, or return the record or any of its data.
 * The callback must explicitly return undefined. No bytes or iterator escape.
 */
export function consumeRoutingHistoryRecord(raw: Uint8Array, consume: RoutingHistoryConsumer): void {
    readRecord(raw, consume, true)
}

let packedScratch: Uint8Array | null = null
let packedPosition = 0
function packValue(shape: RoutingHistoryShape, value: JsonValue | RoutingHistoryOutput): void {
    const target = packedScratch
    if (!target) throw new Error('history-invalid')
    const byte = (n: number): void => {
        if (packedPosition === target.length) throw new Error('limit-exceeded')
        target[packedPosition++] = n
    }
    if (shape.nullable) { byte(value === null ? 0 : 1); if (value === null) return }
    if (shape.kind === 'object' && shape.fields) {
        const fields = object(value)
        for (const [name, child] of shape.fields) packValue(child, fields[name])
    } else if (shape.kind === 'array' && shape.item) {
        if (!Array.isArray(value)) throw new Error('invalid-shape')
        byte(value.length)
        for (const child of value) packValue(shape.item, child)
    } else if (shape.kind === 'number') {
        if (packedPosition + 8 > target.length) throw new Error('limit-exceeded')
        if (typeof value !== 'number') return invalid()
        new DataView(target.buffer, packedPosition, 8).setFloat64(0, value)
        packedPosition += 8
    } else if (shape.kind === 'boolean') byte(value === true ? 1 : 0)
    else {
        if (typeof value !== 'string') return invalid()
        if (shape.enum) {
            const index = historyEnums[shape.enum].indexOf(value)
            if (index < 0) return invalid()
            byte(index)
        } else {
            const start = packedPosition
            byte(0); byte(0)
            const result = new TextEncoder().encodeInto(value, target.subarray(packedPosition))
            if (result.read !== value.length || result.written > (shape.limit ?? 512)) throw new Error('limit-exceeded')
            target[start] = result.written >> 8
            target[start + 1] = result.written & 255
            packedPosition += result.written
        }
    }
}

/** An idle renderer owner releases P after every actual borrower has completed. */
export function releaseRoutingHistoryScratch(): void {
    if (readingRecord) throw new Error('history-busy')
    packedScratch = null; packedPosition = 0
}

/** Iterate an already projected initial without materializing its array. */
export function routingHistoryRecords(raw: Uint8Array, consume: (record: Uint8Array) => undefined): void {
    const prefix = '{"requests":['
    for (let i = 0; i < prefix.length; i++) if (raw[i] !== prefix.charCodeAt(i)) throw new Error('invalid-shape')
    let start = prefix.length, depth = 0, quoted = false, escaped = false, count = 0
    for (let i = start; i < raw.length; i++) {
        const c = raw[i]
        if (quoted) {
            if (escaped) escaped = false
            else if (c === 92) escaped = true
            else if (c === 34) quoted = false
        } else if (c === 34) quoted = true
        else if (c === 123 || c === 91) depth++
        else if (c === 125 || c === 93) {
            if (depth === 0) {
                if (c !== 93 || i + 2 !== raw.length || raw[i + 1] !== 125 || start !== i) throw new Error('invalid-shape')
                return
            }
            depth--
            if (depth === 0) {
                if (++count > 256) throw new Error('limit-exceeded')
                if (consume(raw.subarray(start, i + 1)) !== undefined) throw new Error('history-invalid')
                start = i + 1
                if (raw[start] === 44) { start++; i++ }
            }
        }
    }
    throw new Error('invalid-shape')
}

/** Packed UTF-8 current/pending/baseline bank. No retained DTO graph or key map. */
export class RoutingHistoryBank {
    private readonly data = new Uint8Array(256 * cellBytes)
    private readonly lengths = new Uint32Array(256)
    private readonly order = new Uint16Array(256)
    private readonly incoming = new Uint8Array(headerBytes)
    private readonly floor = new Uint8Array(headerBytes)
    private count = 0
    private hasFloor = false

    clear(): void {
        if (outputLoan) throw new Error('history-busy')
        this.data.fill(0); this.lengths.fill(0); this.count = 0; this.hasFloor = false; this.floor.fill(0)
    }
    get size(): number { return this.count }
    get capacity() {
        return { windowBytes: this.data.byteLength, controlBytes: this.lengths.byteLength + this.order.byteLength + this.incoming.byteLength + this.floor.byteLength }
    }
    copyFrom(source: RoutingHistoryBank): void {
        if (outputLoan) throw new Error('history-busy')
        this.data.set(source.data); this.lengths.set(source.lengths); this.order.set(source.order)
        this.floor.set(source.floor); this.hasFloor = source.hasFloor; this.count = source.count
    }
    private header(slot: number): Uint8Array { return this.data.subarray(slot * cellBytes, slot * cellBytes + headerBytes) }
    private rank(a: Uint8Array, b: Uint8Array): number {
        const av = new DataView(a.buffer, a.byteOffset, headerBytes), bv = new DataView(b.buffer, b.byteOffset, headerBytes)
        const time = av.getFloat64(0) - bv.getFloat64(0)
        if (time !== 0) return time < 0 ? -1 : 1
        // Rank ascends by time, then reverse origin/run lexical order, then
        // numeric request ID. The visible window reads this order backwards.
        for (const offset of [24, 153]) {
            for (let i = 1; i <= Math.max(a[offset], b[offset]); i++) {
                const x = i <= a[offset] ? a[offset + i] : 0, y = i <= b[offset] ? b[offset + i] : 0
                if (x !== y) return x < y ? 1 : -1
            }
        }
        if (a[282] !== b[282]) return a[282] < b[282] ? -1 : 1
        for (let i = 1; i <= a[282]; i++) if (a[282 + i] !== b[282 + i]) return a[282 + i] < b[282 + i] ? -1 : 1
        return 0
    }
    private sameKey(a: Uint8Array, b: Uint8Array): boolean {
        for (let i = 24; i < 302; i++) if (a[i] !== b[i]) return false
        return true
    }
    apply(raw: Uint8Array, revision = 0): boolean {
        let accepted = false
        readRecord(raw, value => {
            accepted = this.applyRecord(value, revision)
            return undefined
        }, false)
        return accepted
    }
    private applyRecord(value: RoutingHistoryOutput, revision: number): boolean {
        const scratch = packedScratch ??= new Uint8Array(packedRecordBytes)
        packedPosition = 1; scratch[0] = 1
        packValue(routingHistoryRecordShape, value)
        this.incoming.fill(0)
        const view = new DataView(this.incoming.buffer)
        view.setFloat64(0, value.createdAt); view.setFloat64(8, value.seq)
        view.setFloat64(304, revision)
        this.incoming[16] = value.state === 'completed' || value.state === 'failed' || value.state === 'cancelled' ? 1 : 0
        const write = (offset: number, text: string): void => {
            this.incoming[offset] = text.length
            for (let i = 0; i < text.length; i++) this.incoming[offset + 1 + i] = text.charCodeAt(i)
        }
        write(24, value.request.originNodeId); write(153, value.request.proxyRunId); write(282, value.request.requestId)
        let slot = -1
        for (let i = 0; i < this.count; i++) {
            const candidate = this.order[i], header = this.header(candidate)
            if (!this.sameKey(header, this.incoming)) continue
            const old = new DataView(header.buffer, header.byteOffset, headerBytes)
            if (old.getFloat64(0) !== value.createdAt || old.getFloat64(8) >= value.seq || (header[16] === 1 && this.incoming[16] === 0)) return false
            slot = candidate
            break
        }
        if (slot < 0) {
            if (this.hasFloor && this.rank(this.incoming, this.floor) <= 0) return false
            if (this.count === 256) {
                slot = this.order[0]
                if (this.rank(this.incoming, this.header(slot)) <= 0) return false
                this.floor.set(this.header(slot)); this.hasFloor = true
            } else { slot = this.count; this.order[this.count++] = slot }
        }
        this.header(slot).set(this.incoming)
        this.data.set(scratch.subarray(0, packedPosition), slot * cellBytes + headerBytes)
        this.lengths[slot] = packedPosition
        // Bounded insertion sort of scalar slot indices.
        for (let i = 1; i < this.count; i++) {
            const entry = this.order[i]
            let j = i
            while (j > 0 && this.rank(this.header(entry), this.header(this.order[j - 1])) < 0) { this.order[j] = this.order[j - 1]; j-- }
            this.order[j] = entry
        }
        return true
    }
    consume(index: number, consume: RoutingHistoryConsumer): boolean {
        if (readingRecord) throw new Error('history-busy')
        if (!Number.isInteger(index) || index < 0 || index >= this.count) return false
        readRecord(this.at(index), consume, true, true)
        return true
    }
    applyPacked(raw: Uint8Array, revision = 0): boolean {
        let accepted = false
        readRecord(raw, value => { accepted = this.applyRecord(value, revision); return undefined }, false, true)
        return accepted
    }
    after(cursor: number): { raw: Uint8Array; cursor: number } | null {
        let best = Number.MAX_SAFE_INTEGER, selected = -1
        for (let i = 0; i < this.count; i++) {
            const slot = this.order[i], header = this.header(slot)
            const revision = new DataView(header.buffer, header.byteOffset, headerBytes).getFloat64(304)
            if (revision > cursor && revision <= best) { best = revision; selected = this.count - 1 - i }
        }
        return selected < 0 ? null : { raw: this.at(selected), cursor: best }
    }
    /** Borrowed record; consumers complete synchronously before a bank mutation. */
    at(index: number): Uint8Array {
        if (!Number.isInteger(index) || index < 0 || index >= this.count) throw new Error('invalid-value')
        const slot = this.order[this.count - 1 - index], start = slot * cellBytes + headerBytes
        return this.data.subarray(start, start + this.lengths[slot])
    }
}
