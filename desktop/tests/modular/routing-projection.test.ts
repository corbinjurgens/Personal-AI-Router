// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'
import type { JsonValue } from '@/shared/types/json'
import {
    consumeRoutingHistoryRecord,
    RoutingHistoryBank,
    type RoutingHistoryAttempt,
    type RoutingHistoryRecord
} from '@/shared/types/routing'
import { RoutingJsonProjector } from '@/electron/service-bridge/routing-json'

interface Corpus {
    cases: { name: string; record: JsonValue }[]
}

const corpus: Corpus = JSON.parse(
    fs.readFileSync(
        path.resolve('../services/shared/nodepolicy/testdata/routing-request.json'),
        'utf8'
    )
)

function modelLessLocalRecord(): RoutingHistoryRecord {
    const record: RoutingHistoryRecord = JSON.parse(JSON.stringify(corpus.cases[0].record))
    for (const attempt of record.attempts) attempt.destination.model = null
    if (record.served) record.served.model = null
    record.requested.model = ''
    return record
}

function altered(
    caseIndex: number,
    pathParts: readonly string[],
    replacement: JsonValue | undefined
): Uint8Array {
    const value: JsonValue = JSON.parse(JSON.stringify(corpus.cases[caseIndex].record))
    let parent = value
    for (const part of pathParts.slice(0, -1)) {
        if (!parent || typeof parent !== 'object') throw new Error('invalid fixture path')
        parent = Array.isArray(parent) ? parent[Number(part)] : parent[part]
    }
    if (!parent || typeof parent !== 'object' || Array.isArray(parent))
        throw new Error('invalid fixture parent')
    const leaf = pathParts[pathParts.length - 1]
    if (replacement === undefined) delete parent[leaf]
    else parent[leaf] = replacement
    return new TextEncoder().encode(JSON.stringify(value))
}
const requiredPaths = [
    'schema',
    'request',
    'request.originNodeId',
    'request.proxyRunId',
    'request.requestId',
    'workload',
    'workload.originNodeId',
    'workload.proxyRunId',
    'workload.engine',
    'workload.id',
    'workload.attemptIndex',
    'workload.attached',
    'seq',
    'requested',
    'requested.model',
    'requested.tier',
    'requested.mode',
    'createdAt',
    'endedAt',
    'state',
    'reason',
    'status',
    'served',
    'served.kind',
    'served.model',
    'served.nodeId',
    'served.engine',
    'served.deploymentId',
    'served.accountId',
    'recoverable',
    'spend',
    'spend.currency',
    'spend.ceilingNanoUsd',
    'spend.grantedNanoUsd',
    'spend.heldNanoUsd',
    'spend.chargedNanoUsd',
    'spend.accounting',
    'droppedAttempts',
    'attempts',
    'attempts.0.index',
    'attempts.0.startedAt',
    'attempts.0.endedAt',
    'attempts.0.destination',
    'attempts.0.destination.kind',
    'attempts.0.destination.model',
    'attempts.0.destination.nodeId',
    'attempts.0.destination.engine',
    'attempts.0.destination.deploymentId',
    'attempts.0.destination.accountId',
    'attempts.0.outcome',
    'attempts.0.reason',
    'attempts.0.status',
    'attempts.0.waits',
    'attempts.0.waits.admissionMs',
    'attempts.0.waits.firstByteMs',
    'attempts.0.continuationOf',
    'attempts.0.account'
]

describe('routing history schema parity', () => {
    it.each(['', 'weak', 'coder', 'r1.fast_v2', `r${'a'.repeat(62)}9`])(
        'preserves requested role %j through projection, storage and packed IPC',
        tier => {
            const record: RoutingHistoryRecord = JSON.parse(JSON.stringify(corpus.cases[0].record))
            record.requested.tier = tier
            const raw = new TextEncoder().encode(JSON.stringify(record))
            const projected = new RoutingJsonProjector().record(raw)
            consumeRoutingHistoryRecord(projected, value => {
                expect(value).toEqual(record)
                return undefined
            })
            const main = new RoutingHistoryBank(),
                renderer = new RoutingHistoryBank()
            expect(main.apply(raw)).toBe(true)
            expect(
                main.consume(0, value => {
                    expect(value).toEqual(record)
                    return undefined
                })
            ).toBe(true)
            expect(renderer.applyPacked(main.at(0).slice())).toBe(true)
            expect(
                renderer.consume(0, value => {
                    expect(value).toEqual(record)
                    return undefined
                })
            ).toBe(true)
        }
    )

    it.each<JsonValue>([
        'a'.repeat(65),
        'Coder',
        '-coder',
        '.coder',
        '_coder',
        'co:der',
        'codér',
        '\uFEFFcoder',
        'coder\r',
        'coder\n',
        null,
        1,
        true,
        {},
        []
    ])('rejects invalid requested role %j at projection and storage entry points', tier => {
        const raw = altered(0, ['requested', 'tier'], tier)
        expect(() => new RoutingJsonProjector().record(raw)).toThrow()
        expect(() => consumeRoutingHistoryRecord(raw, () => undefined)).toThrow()
        const bank = new RoutingHistoryBank()
        expect(() => bank.apply(raw)).toThrow()
        expect(bank.size).toBe(0)
    })

    it.each([
        'a'.repeat(65),
        'Coder',
        '-oder',
        '.oder',
        '_oder',
        'co:er',
        'codér',
        '\uFEFFcoder',
        'code\r',
        'code\n'
    ])('rejects invalid requested role %j in packed IPC', tier => {
        const record: RoutingHistoryRecord = JSON.parse(JSON.stringify(corpus.cases[0].record))
        record.requested.tier = 'coder'
        const raw = new TextEncoder().encode(JSON.stringify(record))
        const main = new RoutingHistoryBank(),
            renderer = new RoutingHistoryBank()
        expect(main.apply(raw)).toBe(true)
        const packed = main.at(0).slice()
        expect(renderer.applyPacked(packed)).toBe(true)
        const offset = Buffer.from(packed).indexOf('coder')
        expect(offset).toBeGreaterThanOrEqual(2)
        const replacement = new TextEncoder().encode(tier)
        // Keep the packet framing coherent, including the non-ASCII case.
        const invalid = new Uint8Array(packed.length - 5 + replacement.length)
        invalid.set(packed.subarray(0, offset))
        invalid[offset - 1] = replacement.length
        invalid.set(replacement, offset)
        invalid.set(packed.subarray(offset + 5), offset + replacement.length)
        expect(() => renderer.applyPacked(invalid)).toThrow()
        expect(renderer.size).toBe(1)
        expect(
            renderer.consume(0, value => {
                expect(value).toEqual(record)
                return undefined
            })
        ).toBe(true)
        record.seq += 1
        record.requested.tier = 'recovered'
        expect(main.apply(new TextEncoder().encode(JSON.stringify(record)))).toBe(true)
        expect(renderer.applyPacked(main.at(0).slice())).toBe(true)
        expect(
            renderer.consume(0, value => {
                expect(value).toEqual(record)
                return undefined
            })
        ).toBe(true)
    })

    it('projects a model-less local record without losing its real node and engine', () => {
        const record = modelLessLocalRecord()
        const raw = new TextEncoder().encode(JSON.stringify(record))
        const projected = new RoutingJsonProjector().record(raw)
        consumeRoutingHistoryRecord(projected, value => {
            expect(value).toEqual(record)
            expect(value.attempts[0].destination).toMatchObject({
                model: null,
                nodeId: 'synthetic-busy',
                engine: 'llamacpp'
            })
            return undefined
        })
    })

    it('accepts a model-less record alongside other rows in initial projection and correlation', () => {
        const initial = { requests: [modelLessLocalRecord(), corpus.cases[1].record] }
        const raw = new TextEncoder().encode(JSON.stringify(initial))
        const projector = new RoutingJsonProjector()
        expect(JSON.parse(new TextDecoder().decode(projector.initial(raw)))).toEqual(initial)
        const response = projector.frame(
            new TextEncoder().encode(
                JSON.stringify({ jsonrpc: '2.0', id: 'jh:1', result: initial })
            ),
            1n,
            1n
        )
        if (response.kind !== 'response') throw new Error('expected a guarded initial response')
        expect(JSON.parse(new TextDecoder().decode(response.value))).toEqual(initial)
    })

    it('preserves null models through the packed bank and IPC packet representation', () => {
        const record = modelLessLocalRecord(),
            main = new RoutingHistoryBank(),
            renderer = new RoutingHistoryBank()
        record.spend.ceilingNanoUsd = '9223372036854775807'
        record.spend.grantedNanoUsd = '9007199254740993'
        record.spend.heldNanoUsd = '0'
        record.spend.chargedNanoUsd = null
        expect(main.apply(new TextEncoder().encode(JSON.stringify(record)))).toBe(true)
        expect(renderer.applyPacked(main.at(0).slice())).toBe(true)
        expect(
            renderer.consume(0, value => {
                expect(value).toEqual(record)
                expect(value.spend).toMatchObject({
                    ceilingNanoUsd: '9223372036854775807',
                    grantedNanoUsd: '9007199254740993',
                    heldNanoUsd: '0',
                    chargedNanoUsd: null
                })
                return undefined
            })
        ).toBe(true)
    })

    it.each([0, 1])('rejects an empty non-null destination model in corpus case %i', index => {
        expect(() =>
            new RoutingJsonProjector().record(
                altered(index, ['attempts', '0', 'destination', 'model'], '')
            )
        ).toThrow('invalid-value')
    })

    it('rejects an empty served model', () => {
        expect(() =>
            new RoutingJsonProjector().record(altered(0, ['served', 'model'], ''))
        ).toThrow('invalid-value')
    })

    it('rejects a null API destination model', () => {
        expect(() =>
            new RoutingJsonProjector().record(
                altered(1, ['attempts', '0', 'destination', 'model'], null)
            )
        ).toThrow('invalid-value')
    })

    it('retains the requested-model restriction outside the model-unknown local case', () => {
        expect(() =>
            new RoutingJsonProjector().record(altered(0, ['requested', 'model'], ''))
        ).toThrow('invalid-value')
    })

    it.each<'nodeId' | 'engine'>(['nodeId', 'engine'])(
        'requires the real %s for a model-unknown local attempt',
        field => {
            const record = modelLessLocalRecord()
            record.attempts[0].destination[field] = null
            expect(() =>
                new RoutingJsonProjector().record(new TextEncoder().encode(JSON.stringify(record)))
            ).toThrow('invalid-value')
        }
    )

    it.each(requiredPaths)('requires the present key %s, including nullable fields', pathName => {
        expect(() =>
            new RoutingJsonProjector().record(altered(0, pathName.split('.'), undefined))
        ).toThrow('invalid-shape')
    })
    it.each(['state', 'heldNanoUsd', 'chargedNanoUsd', 'accounting'])(
        'requires API account.%s',
        leaf => {
            expect(() =>
                new RoutingJsonProjector().record(
                    altered(1, ['attempts', '0', 'account', leaf], undefined)
                )
            ).toThrow('invalid-shape')
        }
    )
    it.each<JsonValue>([
        '01',
        '-1',
        '+1',
        '1.0',
        '1e0',
        '9223372036854775808',
        0,
        0.1,
        true,
        {},
        []
    ])('rejects a noncanonical money value', value => {
        expect(() =>
            new RoutingJsonProjector().record(altered(1, ['spend', 'chargedNanoUsd'], value))
        ).toThrow()
    })
    it.each(['state', 'reason'])('rejects unknown request %s without echoing the value', field => {
        expect(() =>
            new RoutingJsonProjector().record(altered(0, [field], 'synthetic-private-enum'))
        ).toThrow('invalid-value')
    })
    it.each(['outcome', 'reason'])('rejects unknown attempt %s', field => {
        expect(() =>
            new RoutingJsonProjector().record(
                altered(0, ['attempts', '0', field], 'synthetic-private-enum')
            )
        ).toThrow('invalid-value')
    })
    it.each(['\r', '\n', '\u2028', '\u2029'])(
        'rejects a trailing line terminator in identities and canonical decimals',
        suffix => {
            const raw = JSON.stringify(corpus.cases[1].record)
            const invalidOrigin = raw.replace(
                '"synthetic-origin"',
                JSON.stringify(`synthetic-origin${suffix}`)
            )
            const invalidRequest = raw.replace(
                '"requestId":"2"',
                `"requestId":${JSON.stringify(`2${suffix}`)}`
            )
            const invalidMoney = raw.replace('"250000000"', JSON.stringify(`250000000${suffix}`))
            for (const input of [invalidOrigin, invalidRequest, invalidMoney]) {
                expect(() =>
                    consumeRoutingHistoryRecord(new TextEncoder().encode(input), () => undefined)
                ).toThrow('invalid-value')
            }
        }
    )
    it('bounds a maximal valid packed record loan without changing the JSON wire schema', () => {
        const identifier = 'a'.repeat(128),
            money = '9223372036854775807',
            maximum = Number.MAX_SAFE_INTEGER
        const attempts: RoutingHistoryAttempt[] = []
        for (let i = 0; i < 32; i++)
            attempts.push({
                index: maximum - 32 + i,
                startedAt: maximum - 1,
                endedAt: maximum,
                destination: {
                    kind: 'api',
                    model: '"'.repeat(512),
                    nodeId: null,
                    engine: null,
                    deploymentId: identifier,
                    accountId: identifier
                },
                outcome: 'committed',
                reason: 'admission-wake-timeout',
                status: 599,
                waits: { admissionMs: maximum, firstByteMs: maximum },
                continuationOf: maximum - 33,
                account: {
                    state: 'credential-failure',
                    heldNanoUsd: money,
                    chargedNanoUsd: money,
                    accounting: 'recorded'
                }
            })
        const value: RoutingHistoryRecord = {
            schema: 1,
            request: { originNodeId: identifier, proxyRunId: identifier, requestId: money },
            workload: {
                originNodeId: identifier,
                proxyRunId: identifier,
                engine: identifier,
                id: identifier,
                attemptIndex: 0,
                attached: false
            },
            seq: maximum,
            requested: { model: '"'.repeat(512), tier: 'r'.repeat(64), mode: 'completion' },
            createdAt: maximum - 1,
            endedAt: maximum,
            state: 'completed',
            reason: 'admission-wake-timeout',
            status: 599,
            served: attempts[31].destination,
            recoverable: false,
            spend: {
                currency: 'USD',
                ceilingNanoUsd: money,
                grantedNanoUsd: money,
                heldNanoUsd: money,
                chargedNanoUsd: money,
                accounting: 'recorded'
            },
            droppedAttempts: maximum - 32,
            attempts
        }
        const wire = JSON.stringify(value),
            projector = new RoutingJsonProjector()
        const projected = projector.record(new TextEncoder().encode(wire))
        consumeRoutingHistoryRecord(projected, row => {
            expect(row).toEqual(value)
            return undefined
        })
        expect(() =>
            projector.initial(new TextEncoder().encode(`{"requests":[${wire}]}`))
        ).not.toThrow()
        // R admits wire whitespace too; S constrains packed cells, not JSON.
        const recordWithinR = new TextEncoder().encode(wire.padEnd(131072, ' '))
        expect(recordWithinR.byteLength).toBe(131072)
        consumeRoutingHistoryRecord(recordWithinR, () => undefined)
        const bank = new RoutingHistoryBank()
        for (let i = 0; i < 256; i++) {
            value.request.requestId = (9223372036854775552n + BigInt(i)).toString()
            expect(bank.apply(new TextEncoder().encode(JSON.stringify(value)))).toBe(true)
        }
        expect(bank.size).toBe(256)
        // Source-derived fixture size; the schema ceiling also covers other valid shapes.
        expect(bank.at(0).byteLength).toBe(30766)
        expect(bank.at(0).byteLength).toBeLessThanOrEqual(30833)
        expect(bank.at(0).byteLength).toBeLessThanOrEqual(32752)
        const packed = bank.at(0).slice()
        expect(packed.byteLength).toBe(packed.buffer.byteLength)
        expect(packed.byteLength + 4096).toBeLessThanOrEqual(135168)
        expect(bank.capacity).toEqual({ windowBytes: 16777216, controlBytes: 2176 })
        expect(
            bank.consume(0, row => {
                expect(row).toEqual(value)
                return undefined
            })
        ).toBe(true)
        const renderer = new RoutingHistoryBank()
        expect(renderer.applyPacked(packed)).toBe(true)
        expect(
            renderer.consume(0, row => {
                expect(row).toEqual(value)
                return undefined
            })
        ).toBe(true)
    })
    it.each(corpus.cases)('preserves the canonical $name fixture', ({ record }) => {
        consumeRoutingHistoryRecord(new TextEncoder().encode(JSON.stringify(record)), output => {
            expect(output).toEqual(record)
            return undefined
        })
    })

    it('preserves exact money and rejects an unsafe money spelling', () => {
        const raw = JSON.stringify(corpus.cases[1].record)
        consumeRoutingHistoryRecord(new TextEncoder().encode(raw), record => {
            expect(record.spend.chargedNanoUsd).toBe('123456789')
            return undefined
        })
        const invalid = raw.replace('"123456789"', '"9223372036854775808"')
        expect(() =>
            consumeRoutingHistoryRecord(new TextEncoder().encode(invalid), () => undefined)
        ).toThrow('invalid-value')
    })

    it('rejects nested output loans and releases the reader after a throwing callback', () => {
        const raw = new TextEncoder().encode(JSON.stringify(corpus.cases[0].record))
        expect(() =>
            consumeRoutingHistoryRecord(raw, () => {
                consumeRoutingHistoryRecord(raw, () => undefined)
                return undefined
            })
        ).toThrow('history-busy')
        expect(() =>
            consumeRoutingHistoryRecord(raw, () => {
                throw new Error('consumer failed')
            })
        ).toThrow('consumer failed')
        consumeRoutingHistoryRecord(raw, record => {
            expect(record.seq).toBe(3)
            return undefined
        })
    })

    it('rejects async output consumers at the typed and runtime boundaries', () => {
        const raw = new TextEncoder().encode(JSON.stringify(corpus.cases[0].record))
        // @ts-expect-error An output loan must return undefined, never Promise<void>.
        expect(() => consumeRoutingHistoryRecord(raw, async () => {})).toThrow('history-invalid')
    })

    it('preserves a committed terminal against delayed baseline and lower revisions', () => {
        const bank = new RoutingHistoryBank()
        const raw = JSON.stringify(corpus.cases[0].record)
        expect(bank.apply(new TextEncoder().encode(raw))).toBe(true)
        expect(bank.apply(new TextEncoder().encode(raw.replace('"seq":3', '"seq":2')))).toBe(false)
        expect(bank.apply(new TextEncoder().encode(raw))).toBe(false)
        bank.consume(0, record => {
            expect(record.state).toBe('completed')
            return undefined
        })
    })
})

describe('routing history raw projection', () => {
    it.each(['"id":"jh:1","\\u0069d":7', '"ID":7,"id":"\\u006ah:1"'])(
        'quarantines mixed candidates before ordinary materialization: %s',
        candidates => {
            const projector = new RoutingJsonProjector()
            const raw = new TextEncoder().encode(
                `{"jsonrpc":"2.0",${candidates},"result":{"discard":"synthetic-secret"}}`
            )
            expect(projector.frame(raw, 1n, 1n).kind).toBe('quarantine')
        }
    )

    it.each(['jh:1', 'jh:999', 'jh:0', 'jh:01', 'jh:18446744073709551616'])(
        'discards nonpending reserved response %s',
        id => {
            const projector = new RoutingJsonProjector()
            const raw = new TextEncoder().encode(
                JSON.stringify({ jsonrpc: '2.0', id, result: { discarded: 'synthetic-secret' } })
            )
            expect(projector.frame(raw, 2n, 2n).kind).toBe('discard')
        }
    )

    it('keeps an ordinary incoming request with a colliding outbound history ID ordinary', () => {
        const projector = new RoutingJsonProjector()
        const raw = new TextEncoder().encode(
            '{"jsonrpc":"2.0","id":"jh:1","method":"echo","params":{}}'
        )
        expect(projector.frame(raw, 1n, 1n).kind).toBe('ordinary')
    })

    it('preserves the exact admitted raw private-request refusal token', () => {
        const projector = new RoutingJsonProjector()
        const raw = new TextEncoder().encode(
            '{"jsonrpc":"2.0","id":"\\u006ah:1","method":"routing/request","params":{}}'
        )
        const selected = projector.frame(raw, 1n, 1n)
        expect(selected.kind).toBe('refuse')
        if (selected.kind !== 'refuse') throw new Error('expected refusal')
        expect(new TextDecoder().decode(selected.id)).toBe('"\\u006ah:1"')
        expect(selected.code).toBe(-32601)
    })

    it('refuses an over-512-byte incoming history ID with null', () => {
        const projector = new RoutingJsonProjector()
        const raw = new TextEncoder().encode(
            JSON.stringify({
                jsonrpc: '2.0',
                id: 'x'.repeat(511),
                method: 'routing:subscribe',
                params: {}
            })
        )
        const selected = projector.frame(raw, null, 0n)
        expect(selected.kind).toBe('refuse')
        if (selected.kind !== 'refuse') throw new Error('expected refusal')
        expect(new TextDecoder().decode(selected.id)).toBe('null')
        expect(selected.code).toBe(-32062)
    })

    it('strips unknown-heavy values before materialization', () => {
        const projector = new RoutingJsonProjector()
        const raw = new TextEncoder().encode(
            `{"requests":[],"discard":"${'x'.repeat(9 * 1024 * 1024)}"}`
        )
        const projected = projector.initial(raw)
        expect(new TextDecoder().decode(projected)).toBe('{"requests":[]}')
    })

    it.each([
        '{"requests":[],"discard":{"a":0,"\\u0061":1}}',
        '{"discard":{"\\u0061":1,"a":0},"requests":[]}'
    ])('rejects recursively escaped duplicate unknown members', raw => {
        const projector = new RoutingJsonProjector()
        expect(() => projector.initial(new TextEncoder().encode(raw))).toThrow('duplicate-field')
    })

    it('rejects fractional safe-integer rounding before JSON.parse', () => {
        const projector = new RoutingJsonProjector()
        const raw = JSON.stringify(corpus.cases[0].record).replace(
            '"seq":3',
            '"seq":3.0000000000000001'
        )
        expect(() => projector.record(new TextEncoder().encode(raw))).toThrow('invalid-value')
    })

    it('accepts long exact integral spellings without token-sized numeric scratch', () => {
        const projector = new RoutingJsonProjector()
        const raw = JSON.stringify(corpus.cases[0].record).replace(
            '"seq":3',
            `"seq":3.${'0'.repeat(66000)}`
        )
        const projected = projector.record(new TextEncoder().encode(raw))
        consumeRoutingHistoryRecord(projected, value => {
            expect(value.seq).toBe(3)
            return undefined
        })
    })
})
