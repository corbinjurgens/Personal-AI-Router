// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    consumeRoutingHistoryRecord, routingHistoryRecordShape, routingHistoryInitialShape,
    type RoutingHistoryShape
} from '@/shared/types/routing'

const ROUTING_RECORD_BYTES = 131072
const ROUTING_INITIAL_BYTES = 33554432
export const ROUTING_FRAME_BYTES = 33558528

export type RoutingHistoryMethod = 'routing:subscribe' | 'routing:unsubscribe' | 'routing:get-initial'
type HistoryMethod = RoutingHistoryMethod | 'routing:changed' | 'routing/request' |
    'routing/history-window' | 'routing/history-window-end' | 'routing/history-accepted'
type RoutingFrameSelection =
    | { kind: 'ordinary' }
    | { kind: 'quarantine' }
    | { kind: 'discard'; category?: 'history-stale' | 'history-invalid' }
    | { kind: 'refuse'; id: Uint8Array; code: number; message: string }
    | { kind: 'notification'; value: Uint8Array }
    | { kind: 'response'; value: Uint8Array }
    | { kind: 'error'; code: number; message: string }

const methods: readonly HistoryMethod[] = ['routing:subscribe', 'routing:unsubscribe', 'routing:get-initial',
    'routing:changed', 'routing/request', 'routing/history-window', 'routing/history-window-end', 'routing/history-accepted']

const record = routingHistoryRecordShape, initial = routingHistoryInitialShape

function fail(category = 'invalid-shape'): never { throw new Error(category) }
function whitespace(c: number): boolean { return c === 32 || c === 9 || c === 10 || c === 13 }
function hex(c: number): number {
    if (c >= 48 && c <= 57) return c - 48
    if (c >= 65 && c <= 70) return c - 55
    if (c >= 97 && c <= 102) return c - 87
    return fail()
}

/**
 * One reusable raw owner: H output, M key offsets, and 5,001 bytes of fixed
 * grammar/duplicate control. Returned views borrow output until the next call.
 * No decoded unknown key, value, token string or generic input tree is retained.
 */
export class RoutingJsonProjector {
    private outputBytes: Uint8Array | null = null
    private keyOffsets: Uint32Array | null = null
    private get output(): Uint8Array { return this.outputBytes ??= new Uint8Array(ROUTING_FRAME_BYTES) }
    private get offsets(): Uint32Array { return this.keyOffsets ??= new Uint32Array(ROUTING_INITIAL_BYTES / 4) }
    private readonly stack = new Uint8Array(5001)
    private raw: Uint8Array = new Uint8Array(0)
    private pos = 0
    private written = 0
    private runePosition = 0
    private members = 0

    get capacity() { return { outputBytes: this.outputBytes?.byteLength ?? 0, offsetBytes: this.keyOffsets?.byteLength ?? 0, grammarBytes: this.stack.byteLength } }

    /** All candidates are inspected before any generic decode or raw trace. */
    frame(raw: Uint8Array, pending: bigint | null, issued: bigint): RoutingFrameSelection {
        this.raw = raw
        try {
            // Ordinary syntax remains last-wins. Duplicate proof is selected
            // only after inspecting every outer candidate, in either order.
            this.guard(10000, ROUTING_INITIAL_BYTES, false)
            this.pos = 0; this.space()
            if (raw[this.pos++] !== 123) return { kind: 'quarantine' }
            let ids = 0, methodCount = 0, values = 0, errors = 0
            let method: HistoryMethod | null = null, methodIsString = false
            let reserved = false, matched = false, ordinal = 0n
            let idStart = 0, idEnd = 0, valueStart = 0, valueEnd = 0, errorStart = 0, errorEnd = 0
            let version = false, result = false, params = false, outerSeen = 0, duplicateOuter = false
            this.space()
            while (raw[this.pos] !== 125) {
                const keyAt = this.pos
                this.pos = this.stringEnd(this.pos); this.space(); this.pos++; this.space()
                const start = this.pos
                this.skip()
                const end = this.pos
                const outerNames = ['jsonrpc', 'method', 'id', 'result', 'params', 'error']
                for (let index = 0; index < outerNames.length; index++) {
                    if (!this.outerKey(keyAt, outerNames[index])) continue
                    if (outerSeen & (1 << index)) duplicateOuter = true
                    outerSeen |= 1 << index
                }
                if (this.outerKey(keyAt, 'method')) {
                    methodCount++
                    methodIsString = raw[start] === 34 && end - start > 2
                    if (raw[start] === 34) for (const candidate of methods) if (this.keyEquals(start, candidate)) method = candidate
                } else if (this.outerKey(keyAt, 'id')) {
                    ids++; idStart = start; idEnd = end
                    this.runePosition = start + 1
                    if (raw[start] === 34 && this.prefix(start, 'jh:')) {
                        reserved = true
                        let n = 0n, digits = 0, valid = true
                        while (this.runePosition < end - 1) {
                            const c = this.nextRune()
                            if (c < 48 || c > 57 || (digits === 0 && c === 48) || digits >= 20) valid = false
                            if (valid) n = n * 10n + BigInt(c - 48)
                            digits++
                        }
                        ordinal = valid && digits > 0 && n <= 18446744073709551615n ? n : 0n
                        if (ordinal !== 0n && ordinal === pending) matched = true
                    }
                } else if (this.outerKey(keyAt, 'result') || this.outerKey(keyAt, 'params')) {
                    values++; valueStart = start; valueEnd = end
                    result = this.outerKey(keyAt, 'result'); params = !result
                } else if (this.outerKey(keyAt, 'error')) {
                    errors++; errorStart = start; errorEnd = end
                } else if (this.outerKey(keyAt, 'jsonrpc')) version = raw[start] === 34 && this.keyEquals(start, '2.0')
                this.space()
                if (raw[this.pos] === 44) { this.pos++; this.space() } else break
            }
            const ordinaryRequest = methodCount === 1 && ids === 1 && methodIsString && method === null && !(outerSeen & (1 << 3)) && errors === 0
            if (method === null && (!reserved || ordinaryRequest)) return version ? { kind: 'ordinary' } : { kind: 'quarantine' }
            if (ids > 1 || methodCount > 1) return { kind: 'quarantine' }
            if (reserved && methodCount === 0 && !matched) return { kind: 'discard', category: ordinal > 0n && ordinal <= issued ? 'history-stale' : 'history-invalid' }
            try { this.guard(10000, ROUTING_INITIAL_BYTES) } catch { return matched ? { kind: 'error', code: -32062, message: 'history-invalid' } : { kind: 'discard' } }
            if (duplicateOuter) return matched ? { kind: 'error', code: -32062, message: 'history-invalid' } : { kind: 'discard' }
            if (!version) return matched ? { kind: 'error', code: -32062, message: 'history-invalid' } : { kind: 'discard' }
            if (methodCount === 1 && ids === 1 && method !== null && !result && errors === 0) {
                if (raw[idStart] !== 34 && raw[idStart] !== 45 && !(raw[idStart] >= 48 && raw[idStart] <= 57)) return { kind: 'discard' }
                if (idEnd - idStart > 512) return { kind: 'refuse', id: new TextEncoder().encode('null'), code: -32062, message: 'history-invalid' }
                // Desktop accepts no incoming history requests. Preserve exact
                // raw IDs; outbound ID collisions never settle a pending call.
                const publicMethod = method === 'routing:subscribe' || method === 'routing:unsubscribe' || method === 'routing:get-initial'
                let invalid = false
                if (publicMethod) {
                    const envelope = this.raw
                    try { this.project(raw.subarray(valueStart, valueEnd), { kind: 'object', fields: [] }, ROUTING_RECORD_BYTES, 16, 2048) }
                    catch { invalid = true }
                    finally { this.raw = envelope }
                }
                return { kind: 'refuse', id: raw.subarray(idStart, idEnd), code: invalid ? -32062 : -32601, message: invalid ? 'history-invalid' : 'method not found' }
            }
            if (methodCount === 0 && matched) {
                if (idEnd - idStart > 512 || raw.length - (valueEnd - valueStart) + 1 > 4096) return { kind: 'error', code: -32062, message: 'history-invalid' }
                if (errors === 1 && values === 0 && errorEnd - errorStart <= 512 && raw.length + 1 <= 4096) {
                    // Even bounded error numbers are checked before a floating
                    // parser could round a fractional code into a known one.
                    this.pos = errorStart
                    if (raw[this.pos++] !== 123) return { kind: 'error', code: -32062, message: 'history-invalid' }
                    let code = 0, busyMessage = false, codePresent = false
                    this.space()
                    while (raw[this.pos] !== 125) {
                        const name = this.pos
                        this.pos = this.stringEnd(this.pos); this.space(); this.pos++; this.space()
                        const start = this.pos
                        this.skip()
                        if (this.keyEquals(name, 'code')) {
                            codePresent = true
                            if (this.numberEnd(start) !== this.pos) return { kind: 'error', code: -32062, message: 'history-invalid' }
                            code = this.pos - start === 6 && raw[start] === 45 && raw[start + 1] === 51 && raw[start + 2] === 50 && raw[start + 3] === 48 && raw[start + 4] === 57 && raw[start + 5] === 54 ? -32096 : -32062
                        } else if (this.keyEquals(name, 'message')) busyMessage = raw[start] === 34 && this.keyEquals(start, 'routing history busy; retry')
                        this.space()
                        if (raw[this.pos] === 44) { this.pos++; this.space() } else break
                    }
                    if (codePresent && code === -32096 && busyMessage) return { kind: 'error', code: -32096, message: 'routing history busy; retry' }
                    return { kind: 'error', code: -32062, message: 'history-invalid' }
                }
                if (values !== 1 || !result || errors !== 0) return { kind: 'error', code: -32062, message: 'history-invalid' }
                try { return { kind: 'response', value: this.initial(raw.subarray(valueStart, valueEnd)) } }
                catch { return { kind: 'error', code: -32062, message: 'history-invalid' } }
            }
            if (method === 'routing:changed' && ids === 0 && values === 1 && params && errors === 0 && raw.length - (valueEnd - valueStart) + 1 <= 4096) {
                try { return { kind: 'notification', value: this.record(raw.subarray(valueStart, valueEnd)) } }
                catch { return { kind: 'discard' } }
            }
            return { kind: 'discard' }
        } catch { return { kind: 'quarantine' } }
        finally { this.raw = new Uint8Array(0) }
    }

    private prefix(start: number, value: string): boolean {
        this.runePosition = start + 1
        for (let i = 0; i < value.length; i++) if (this.nextRune() !== value.charCodeAt(i)) return false
        return true
    }
    private outerKey(start: number, value: string): boolean {
        this.runePosition = start + 1
        for (let i = 0; i < value.length; i++) {
            let c = this.nextRune()
            if (c >= 65 && c <= 90) c += 32
            if (c === 383) c = 115
            if (c === 8490) c = 107
            if (c !== value.charCodeAt(i)) return false
        }
        return this.raw[this.runePosition] === 34
    }

    record(raw: Uint8Array): Uint8Array {
        const value = this.project(raw, record, ROUTING_RECORD_BYTES, 16, 2048)
        consumeRoutingHistoryRecord(value, () => undefined)
        return value
    }

    initial(raw: Uint8Array): Uint8Array {
        return this.project(raw, initial, ROUTING_INITIAL_BYTES, 18, ROUTING_INITIAL_BYTES)
    }

    private project(raw: Uint8Array, shape: RoutingHistoryShape, bytes: number, depth: number, members: number): Uint8Array {
        if (raw.byteLength > bytes) return fail('limit-exceeded')
        this.raw = raw
        this.pos = 0
        this.written = 0
        try {
            this.guard(depth, members)
            this.pos = 0
            this.value(shape, 1)
            return this.output.subarray(0, this.written)
        } finally { this.raw = new Uint8Array(0) }
    }

    private space(): void { while (whitespace(this.raw[this.pos])) this.pos++ }
    private ascii(value: string): void {
        if (this.written + value.length > this.output.length) return fail('limit-exceeded')
        for (let i = 0; i < value.length; i++) this.output[this.written++] = value.charCodeAt(i)
    }
    private byte(value: number): void {
        if (this.written === this.output.length) return fail('limit-exceeded')
        this.output[this.written++] = value
    }
    private rune(value: number): void {
        if (value === 34 || value === 92) { this.byte(92); this.byte(value) }
        else if (value < 32) this.ascii(`\\u${value.toString(16).padStart(4, '0')}`)
        else if (value < 128) this.byte(value)
        else if (value < 2048) { this.byte(192 | (value >> 6)); this.byte(128 | (value & 63)) }
        else if (value < 65536) {
            this.byte(224 | (value >> 12)); this.byte(128 | ((value >> 6) & 63)); this.byte(128 | (value & 63))
        } else {
            this.byte(240 | (value >> 18)); this.byte(128 | ((value >> 12) & 63))
            this.byte(128 | ((value >> 6) & 63)); this.byte(128 | (value & 63))
        }
    }
    private unit(at: number): number {
        return hex(this.raw[at]) * 4096 + hex(this.raw[at + 1]) * 256 + hex(this.raw[at + 2]) * 16 + hex(this.raw[at + 3])
    }
    private nextRune(): number {
        let i = this.runePosition
        const c = this.raw[i++]
        if (c === 92) {
            const escape = this.raw[i++]
            this.runePosition = i
            if (escape === 34 || escape === 92 || escape === 47) return escape
            if (escape === 98) return 8
            if (escape === 102) return 12
            if (escape === 110) return 10
            if (escape === 114) return 13
            if (escape === 116) return 9
            if (escape !== 117) return fail()
            let value = this.unit(i)
            i += 4
            if (value >= 0xd800 && value <= 0xdbff && this.raw[i] === 92 && this.raw[i + 1] === 117) {
                const low = this.unit(i + 2)
                if (low >= 0xdc00 && low <= 0xdfff) { value = 65536 + (value - 0xd800) * 1024 + low - 0xdc00; i += 6 }
            }
            this.runePosition = i
            return value >= 0xd800 && value <= 0xdfff ? 0xfffd : value
        }
        if (c < 128) {
            if (c < 32 || c === undefined) return fail()
            this.runePosition = i
            return c
        }
        const n = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : fail('invalid-value')
        let value = c & (n === 1 ? 31 : n === 2 ? 15 : 7)
        for (let j = 0; j < n; j++) {
            const part = this.raw[i++]
            if (part < 128 || part > 191 || part === undefined) return fail('invalid-value')
            value = value * 64 + (part & 63)
        }
        if (value < (n === 1 ? 128 : n === 2 ? 2048 : 65536) || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return fail('invalid-value')
        this.runePosition = i
        return value
    }
    private stringEnd(start: number, copy = false): number {
        if (this.raw[start] !== 34) return fail()
        if (copy) this.byte(34)
        this.runePosition = start + 1
        while (this.runePosition < this.raw.length && this.raw[this.runePosition] !== 34) {
            const value = this.nextRune()
            if (copy) this.rune(value)
        }
        if (this.raw[this.runePosition] !== 34) return fail()
        if (copy) this.byte(34)
        return this.runePosition + 1
    }
    private keyEquals(start: number, value: string): boolean {
        this.runePosition = start + 1
        for (let i = 0; i < value.length; i++) if (this.nextRune() !== value.charCodeAt(i)) return false
        return this.raw[this.runePosition] === 34
    }
    private compare(a: number, b: number): number {
        let left = a + 1, right = b + 1
        while (this.raw[left] !== 34 && this.raw[right] !== 34) {
            this.runePosition = left
            const x = this.nextRune()
            left = this.runePosition
            this.runePosition = right
            const y = this.nextRune()
            right = this.runePosition
            if (x !== y) return x < y ? -1 : 1
        }
        return this.raw[left] === this.raw[right] ? 0 : this.raw[left] === 34 ? -1 : 1
    }
    private duplicates(start: number, count: number): void {
        const sift = (root: number, end: number): void => {
            while (root * 2 + 1 < end) {
                let child = root * 2 + 1
                if (child + 1 < end && this.compare(this.offsets[start + child], this.offsets[start + child + 1]) < 0) child++
                if (this.compare(this.offsets[start + root], this.offsets[start + child]) >= 0) return
                const tmp = this.offsets[start + root]
                this.offsets[start + root] = this.offsets[start + child]
                this.offsets[start + child] = tmp
                root = child
            }
        }
        for (let i = Math.floor(count / 2) - 1; i >= 0; i--) sift(i, count)
        for (let i = count - 1; i > 0; i--) {
            const tmp = this.offsets[start]
            this.offsets[start] = this.offsets[start + i]
            this.offsets[start + i] = tmp
            sift(0, i)
        }
        for (let i = 1; i < count; i++) if (this.compare(this.offsets[start + i - 1], this.offsets[start + i]) === 0) fail('duplicate-field')
    }
    private numberEnd(start: number): number {
        let i = start
        if (this.raw[i] === 45) i++
        if (this.raw[i] === 48) i++
        else {
            if (this.raw[i] < 49 || this.raw[i] > 57 || this.raw[i] === undefined) return fail()
            while (this.raw[i] >= 48 && this.raw[i] <= 57) i++
        }
        if (this.raw[i] === 46) {
            i++
            const begin = i
            while (this.raw[i] >= 48 && this.raw[i] <= 57) i++
            if (i === begin) return fail()
        }
        if (this.raw[i] === 69 || this.raw[i] === 101) {
            i++
            if (this.raw[i] === 43 || this.raw[i] === 45) i++
            const begin = i
            while (this.raw[i] >= 48 && this.raw[i] <= 57) i++
            if (i === begin) return fail()
        }
        return i
    }
    private integer(start: number, end: number): number {
        let i = start, negative = false
        if (this.raw[i] === 45) { negative = true; i++ }
        let whole = 0, digit = 0, first = -1, last = -1, mantissaEnd = end
        let fraction = false
        for (; i < end; i++) {
            const c = this.raw[i]
            if (c === 69 || c === 101) { mantissaEnd = i; break }
            if (c === 46) { fraction = true; continue }
            if (!fraction) whole++
            if (c !== 48) { if (first < 0) first = digit; last = digit }
            digit++
        }
        // Go's canonical scalar accepts all exact zero spellings, including -0.
        if (first < 0) return 0
        if (negative) return fail('invalid-value')
        let exponent = 0, sign = 1
        if (i < end) {
            i++
            if (this.raw[i] === 43) i++
            else if (this.raw[i] === 45) { sign = -1; i++ }
            for (; i < end; i++) {
                exponent = exponent * 10 + this.raw[i] - 48
                if (exponent > ROUTING_RECORD_BYTES) return fail('invalid-value')
            }
        }
        const length = whole + exponent * sign - first
        if (length < 1 || length > 16 || last - first >= length) return fail('invalid-value')
        let value = 0, position = 0, taken = 0
        for (i = start; i < mantissaEnd && taken < length; i++) {
            const c = this.raw[i]
            if (c === 45 || c === 46) continue
            if (position++ < first) continue
            value = value * 10 + c - 48
            taken++
        }
        while (taken++ < length) value *= 10
        if (!Number.isSafeInteger(value)) return fail('invalid-value')
        return value
    }
    private literal(start: number, word: string): number {
        for (let i = 0; i < word.length; i++) if (this.raw[start + i] !== word.charCodeAt(i)) return fail()
        return start + word.length
    }
    private scalar(start: number): number {
        const c = this.raw[start]
        if (c === 34) return this.stringEnd(start)
        if (c === 116) return this.literal(start, 'true')
        if (c === 102) return this.literal(start, 'false')
        if (c === 110) return this.literal(start, 'null')
        return this.numberEnd(start)
    }
    private stateAt(depth: number): number { return (this.stack[depth >> 1] >> ((depth & 1) * 4)) & 15 }
    private setState(depth: number, state: number): void {
        const shift = (depth & 1) * 4
        this.stack[depth >> 1] = (this.stack[depth >> 1] & ~(15 << shift)) | (state << shift)
    }
    private guard(limit: number, memberLimit: number, duplicates = true): void {
        this.stack.fill(0)
        let depth = 0, i = 0
        this.members = 0
        for (;;) {
            while (whitespace(this.raw[i])) i++
            const state = this.stateAt(depth), c = this.raw[i]
            if (state === 9) {
                if (i !== this.raw.length) return fail()
                break
            }
            if (i >= this.raw.length) return fail()
            if (state === 1 || state === 5) {
                if (state === 1 && c === 125) {
                    depth--; i++; continue
                }
                if (c !== 34) return fail()
                if (++this.members > memberLimit) return fail('limit-exceeded')
                i = this.stringEnd(i)
                this.setState(depth, 2)
            } else if (state === 2) {
                if (c !== 58) return fail()
                i++; this.setState(depth, 3)
            } else if (state === 4 || state === 7) {
                if (c === (state === 4 ? 125 : 93)) {
                    depth--; i++
                } else {
                    if (c !== 44) return fail()
                    i++; this.setState(depth, state === 4 ? 5 : 8)
                }
            } else {
                if (state === 6 && c === 93) { depth--; i++; continue }
                this.setState(depth, state === 0 ? 9 : state === 3 ? 4 : 7)
                if (c === 123 || c === 91) {
                    if (depth === limit) return fail('limit-exceeded')
                    depth++
                    this.setState(depth, c === 123 ? 1 : 6)
                    i++
                } else i = this.scalar(i)
            }
        }
        if (!duplicates) return
        // Each object's immediate key range is proved and released before
        // scanning its children. No per-depth offset table or unknown strings.
        for (i = 0; i < this.raw.length;) {
            if (this.raw[i] === 34) { i = this.stringEnd(i); continue }
            if (this.raw[i] !== 123) { i++; continue }
            this.pos = i + 1
            this.space()
            let count = 0
            while (this.raw[this.pos] !== 125) {
                if (count === this.offsets.length) return fail('limit-exceeded')
                this.offsets[count++] = this.pos
                this.pos = this.stringEnd(this.pos); this.space(); this.pos++; this.space()
                this.skip(); this.space()
                if (this.raw[this.pos] !== 44) break
                this.pos++; this.space()
            }
            this.duplicates(0, count)
            i++
        }
    }
    private skip(): void {
        let depth = 0
        do {
            this.space()
            const c = this.raw[this.pos]
            if (c === 123 || c === 91) { depth++; this.pos++ }
            else if (c === 125 || c === 93) { depth--; this.pos++ }
            else if (c === 44 || c === 58) this.pos++
            else this.pos = this.scalar(this.pos)
        } while (depth > 0)
    }
    private value(shape: RoutingHistoryShape, depth: number): void {
        this.space()
        if (shape.nullable && this.raw[this.pos] === 110) { this.pos = this.literal(this.pos, 'null'); this.ascii('null'); return }
        if (shape.kind === 'object') {
            if (this.raw[this.pos++] !== 123 || !shape.fields) return fail()
            this.byte(123)
            let seen = 0, kept = 0
            this.space()
            while (this.raw[this.pos] !== 125) {
                const keyAt = this.pos
                this.pos = this.stringEnd(this.pos)
                this.space(); this.pos++; this.space()
                let index = -1
                for (let i = 0; i < shape.fields.length; i++) if (this.keyEquals(keyAt, shape.fields[i][0])) { index = i; break }
                if (index < 0) this.skip()
                else {
                    seen |= 1 << index
                    if (kept++) this.byte(44)
                    this.ascii(`"${shape.fields[index][0]}":`)
                    this.value(shape.fields[index][1], depth + 1)
                }
                this.space()
                if (this.raw[this.pos] === 44) { this.pos++; this.space() } else break
            }
            if (this.raw[this.pos++] !== 125 || seen !== (1 << shape.fields.length) - 1) return fail()
            this.byte(125)
        } else if (shape.kind === 'array') {
            if (this.raw[this.pos++] !== 91 || !shape.item || shape.limit === undefined) return fail()
            this.byte(91)
            let count = 0
            this.space()
            while (this.raw[this.pos] !== 93) {
                if (count === shape.limit) return fail('limit-exceeded')
                if (count++) this.byte(44)
                const start = this.pos, outputStart = this.written
                if (shape.item === record) {
                    this.skip()
                    const end = this.pos, enclosing = this.raw
                    if (end - start > ROUTING_RECORD_BYTES) return fail('limit-exceeded')
                    this.raw = enclosing.subarray(start, end)
                    try { this.guard(16, 2048) } finally { this.raw = enclosing }
                    this.pos = start
                }
                this.value(shape.item, depth + 1)
                if (shape.item === record && (this.pos - start > ROUTING_RECORD_BYTES || this.written - outputStart > ROUTING_RECORD_BYTES)) return fail('limit-exceeded')
                if (shape.item === record) consumeRoutingHistoryRecord(this.output.subarray(outputStart, this.written), () => undefined)
                this.space()
                if (this.raw[this.pos] === 44) { this.pos++; this.space() } else break
            }
            if (this.raw[this.pos++] !== 93) return fail()
            this.byte(93)
        } else if (shape.kind === 'string') {
            const outputStart = this.written
            this.pos = this.stringEnd(this.pos, true)
            // All known history strings are bounded metadata. A loose encoded
            // bound permits escaped controls for semantic rejection later.
            if (this.written - outputStart > (shape.limit ?? 512) * 6 + 2) return fail('invalid-value')
        }
        else if (shape.kind === 'number') {
            const end = this.numberEnd(this.pos)
            this.ascii(String(this.integer(this.pos, end))); this.pos = end
        } else {
            const word = this.raw[this.pos] === 116 ? 'true' : 'false'
            this.pos = this.literal(this.pos, word); this.ascii(word)
        }
    }
}

let projector: RoutingJsonProjector | null = null
export function getRoutingJsonProjector(): RoutingJsonProjector { return projector ??= new RoutingJsonProjector() }
