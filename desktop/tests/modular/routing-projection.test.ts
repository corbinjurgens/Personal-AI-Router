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
type SelectionJson = NonNullable<RoutingHistoryRecord['selection']>
type CandidateJson = SelectionJson['winner']

function numericCandidate(overrides: Partial<CandidateJson> = {}): CandidateJson {
    return {
        nodeId: 'synthetic-node',
        engine: 'llamacpp',
        model: 'synthetic-model',
        totalMs: 120,
        terms: { service: 100, queue: 20, pressure: 0, cold: 0 },
        evidenceAgeMs: 4500,
        samples: 3,
        imputed: false,
        ...overrides
    }
}
function plainCandidate(): CandidateJson {
    return numericCandidate({ totalMs: null, terms: null, evidenceAgeMs: null, samples: 0 })
}
function selectionRecord(selection: SelectionJson | null): RoutingHistoryRecord {
    const record: RoutingHistoryRecord = JSON.parse(JSON.stringify(corpus.cases[0].record))
    record.selection = selection
    return record
}
function encoded(record: RoutingHistoryRecord): Uint8Array {
    return new TextEncoder().encode(JSON.stringify(record))
}
// One-character strings and sixteen-digit numbers: padding then fills the rest of the cap.
function wideCandidate(): CandidateJson {
    const wide = Number.MAX_SAFE_INTEGER
    return numericCandidate({
        model: 'm',
        nodeId: 'n',
        engine: 'e',
        totalMs: wide,
        terms: { service: wide, queue: wide, pressure: wide, cold: wide },
        evidenceAgeMs: wide,
        samples: wide
    })
}
function selectionPaddedTo(bytes: number): SelectionJson {
    const selection: SelectionJson = {
        decision: 'measured',
        tierRank: 0,
        warm: true,
        winner: wideCandidate(),
        runnerUp: wideCandidate()
    }
    if (!selection.runnerUp) throw new Error('fixture needs a runner-up')
    const fields: [CandidateJson, 'model' | 'nodeId' | 'engine', number][] = [
        [selection.winner, 'model', 512],
        [selection.runnerUp, 'model', 512],
        [selection.winner, 'nodeId', 128],
        [selection.runnerUp, 'nodeId', 128],
        [selection.winner, 'engine', 128],
        [selection.runnerUp, 'engine', 128]
    ]
    for (const [candidate, field, limit] of fields) {
        const grow = Math.min(limit - 1, bytes - JSON.stringify(selection).length)
        if (grow > 0) candidate[field] = candidate[field].padEnd(1 + grow, 'x')
    }
    return selection
}

// Longest identifiers and models, with zero-valued numbers: the schema maximum
// that still fits the 2048-byte encoding cap.
function maximalSelection(): SelectionJson {
    const candidate = (): CandidateJson =>
        numericCandidate({
            nodeId: 'a'.repeat(128),
            engine: 'e'.repeat(128),
            model: 'm'.repeat(512),
            totalMs: 0,
            terms: { service: 0, queue: 0, pressure: 0, cold: 0 },
            evidenceAgeMs: 0,
            samples: 0
        })
    return {
        decision: 'measured',
        tierRank: 0,
        warm: true,
        winner: candidate(),
        runnerUp: candidate()
    }
}
// Per candidate: 130 + 130 + 514 + 9 + (1 + 4 * 8) + 9 + 8 + 1; plus the selection tag,
// decision, rank, warm and the runner-up tag.
const maximalSelectionPackedBytes = 1 + 1 + 8 + 1 + 834 + 1 + 834
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
    'selection',
    'selection.decision',
    'selection.tierRank',
    'selection.warm',
    'selection.winner',
    'selection.winner.nodeId',
    'selection.winner.engine',
    'selection.winner.model',
    'selection.winner.totalMs',
    'selection.winner.terms',
    'selection.winner.terms.service',
    'selection.winner.terms.queue',
    'selection.winner.terms.pressure',
    'selection.winner.terms.cold',
    'selection.winner.evidenceAgeMs',
    'selection.winner.samples',
    'selection.winner.imputed',
    'selection.runnerUp',
    'selection.runnerUp.nodeId',
    'selection.runnerUp.terms',
    'selection.runnerUp.terms.cold',
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
            selection: maximalSelection(),
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
        // 30,766 bytes without a selection, plus the selection's packed size.
        expect(bank.at(0).byteLength).toBe(30766 + maximalSelectionPackedBytes)
        expect(bank.at(0).byteLength).toBeLessThanOrEqual(32752)
        expect(JSON.stringify(value.selection).length).toBeLessThanOrEqual(2048)
        const packed = bank.at(0).slice()
        expect(packed.byteLength).toBe(packed.buffer.byteLength)
        expect(packed.byteLength + 4096).toBeLessThanOrEqual(135168)
        expect(bank.capacity).toEqual({ windowBytes: 16777216, controlBytes: 3200 })
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

describe('routing selection record member', () => {
    const accepted: [string, SelectionJson | null][] = [
        ['null', null],
        [
            'numeric with a runner-up',
            {
                decision: 'measured',
                tierRank: 2,
                warm: false,
                winner: numericCandidate(),
                runnerUp: numericCandidate({ nodeId: 'synthetic-peer', imputed: true, samples: 0 })
            }
        ],
        [
            'numeric without a runner-up',
            {
                decision: 'probe',
                tierRank: 0,
                warm: true,
                winner: numericCandidate({ evidenceAgeMs: null }),
                runnerUp: null
            }
        ],
        [
            'imputed',
            {
                decision: 'imputed',
                tierRank: 1,
                warm: true,
                winner: numericCandidate({ imputed: true }),
                runnerUp: numericCandidate()
            }
        ],
        [
            'pinned',
            {
                decision: 'pinned',
                tierRank: 0,
                warm: true,
                winner: plainCandidate(),
                runnerUp: null
            }
        ],
        [
            'baseline',
            {
                decision: 'baseline',
                tierRank: 3,
                warm: false,
                winner: plainCandidate(),
                runnerUp: null
            }
        ],
        [
            'safe-integer bounds',
            {
                decision: 'measured',
                tierRank: Number.MAX_SAFE_INTEGER,
                warm: true,
                winner: numericCandidate({
                    totalMs: Number.MAX_SAFE_INTEGER,
                    terms: {
                        service: 0,
                        queue: 1,
                        pressure: 2,
                        cold: Number.MAX_SAFE_INTEGER
                    },
                    evidenceAgeMs: Number.MAX_SAFE_INTEGER,
                    samples: Number.MAX_SAFE_INTEGER
                }),
                runnerUp: null
            }
        ]
    ]
    it.each(accepted)(
        'round-trips a %s selection through projection, bank and packed IPC',
        (_, selection) => {
            const record = selectionRecord(selection)
            const raw = encoded(record)
            consumeRoutingHistoryRecord(new RoutingJsonProjector().record(raw), value => {
                expect(value).toEqual(record)
                return undefined
            })
            const main = new RoutingHistoryBank(),
                renderer = new RoutingHistoryBank()
            expect(main.apply(raw)).toBe(true)
            expect(renderer.applyPacked(main.at(0).slice())).toBe(true)
            expect(
                renderer.consume(0, value => {
                    expect(value).toEqual(record)
                    expect(value.selection).toEqual(selection)
                    return undefined
                })
            ).toBe(true)
        }
    )

    it('carries the selection member in the corpus fixtures', () => {
        const records: RoutingHistoryRecord[] = corpus.cases.map(entry =>
            JSON.parse(JSON.stringify(entry.record))
        )
        expect(records[0].selection?.runnerUp).not.toBeNull()
        expect(records[1].selection?.decision).toBe('baseline')
        expect(records[2].selection).toBeNull()
    })

    const rejected: [string, (selection: SelectionJson) => void][] = [
        ['a bad decision', s => void Object.assign(s, { decision: 'cheapest' })],
        ['an empty decision', s => void Object.assign(s, { decision: '' })],
        ['a negative tier rank', s => void (s.tierRank = -1)],
        ['a fractional total', s => void (s.winner.totalMs = 1.5)],
        [
            'an unsafe term value',
            s => void Object.assign(s.winner.terms ?? {}, { cold: Number.MAX_SAFE_INTEGER + 1 })
        ],
        ['a negative term value', s => void Object.assign(s.winner.terms ?? {}, { queue: -1 })],
        [
            'a fractional term value',
            s => void Object.assign(s.winner.terms ?? {}, { service: 0.5 })
        ],
        ['a string term value', s => void Object.assign(s.winner.terms ?? {}, { pressure: '0' })],
        [
            'a missing term member',
            s => void (s.winner.terms && Reflect.deleteProperty(s.winner.terms, 'cold'))
        ],
        ['an array of terms', s => void Object.assign(s.winner, { terms: [] })],
        ['an unsafe sample count', s => void (s.winner.samples = Number.MAX_SAFE_INTEGER + 1)],
        ['a negative evidence age', s => void (s.winner.evidenceAgeMs = -1)],
        ['a string integer', s => void Object.assign(s, { tierRank: '1' })],
        ['a non-boolean warm', s => void Object.assign(s, { warm: 1 })],
        ['a null total on a numeric winner', s => void (s.winner.totalMs = null)],
        [
            'a null total on a numeric runner-up',
            s => void (s.runnerUp && (s.runnerUp.totalMs = null))
        ],
        [
            'a runner-up on a pinned decision',
            s => {
                s.decision = 'pinned'
                s.winner = plainCandidate()
            }
        ],
        [
            'a numeric winner on a baseline decision',
            s => {
                s.decision = 'baseline'
                s.runnerUp = null
            }
        ],
        [
            'terms on a baseline winner',
            s => {
                s.decision = 'baseline'
                s.winner = numericCandidate({ totalMs: null })
                s.runnerUp = null
            }
        ],
        [
            'a total on a baseline winner',
            s => {
                s.decision = 'baseline'
                s.winner = numericCandidate({ terms: null })
                s.runnerUp = null
            }
        ],
        ['null terms on a numeric winner', s => void (s.winner.terms = null)],
        ['null terms on a numeric runner-up', s => void (s.runnerUp && (s.runnerUp.terms = null))],
        ['an empty model', s => void (s.winner.model = '')],
        ['an over-long model', s => void (s.winner.model = 'm'.repeat(513))],
        ['a control character in the model', s => void (s.winner.model = 'bad\u0007model')],
        ['an over-long node identifier', s => void (s.winner.nodeId = 'n'.repeat(129))],
        ['a malformed engine identifier', s => void (s.winner.engine = '-llamacpp')],
        [
            'a runner-up node identifier with a slash',
            s => void (s.runnerUp && (s.runnerUp.nodeId = 'a/b'))
        ],
        [
            'an encoded selection above 2048 bytes',
            s => void Object.assign(s, selectionPaddedTo(2049))
        ]
    ]
    it.each(rejected)('rejects %s in every entry point', (_, mutate) => {
        const selection: SelectionJson = {
            decision: 'measured',
            tierRank: 0,
            warm: true,
            winner: numericCandidate(),
            runnerUp: numericCandidate()
        }
        mutate(selection)
        const raw = encoded(selectionRecord(selection))
        expect(() => consumeRoutingHistoryRecord(raw, () => undefined)).toThrow()
        expect(() => new RoutingJsonProjector().record(raw)).toThrow()
        const bank = new RoutingHistoryBank()
        expect(() => bank.apply(raw)).toThrow()
        expect(bank.size).toBe(0)
    })

    it('accepts an encoded selection of exactly 2048 bytes', () => {
        const selection = selectionPaddedTo(2048)
        expect(JSON.stringify(selection).length).toBe(2048)
        const record = selectionRecord(selection)
        consumeRoutingHistoryRecord(encoded(record), value => {
            expect(value.selection).toEqual(selection)
            return undefined
        })
        expect(new RoutingHistoryBank().apply(encoded(record))).toBe(true)
    })

    it('rejects an unknown selection key at storage and strips it at projection', () => {
        for (const owner of [[], ['winner'], ['winner', 'terms'], ['runnerUp']]) {
            const raw = altered(0, ['selection', ...owner, 'prompt'], 'synthetic-private-text')
            expect(() => consumeRoutingHistoryRecord(raw, () => undefined)).toThrow('invalid-shape')
            expect(() => new RoutingHistoryBank().apply(raw)).toThrow('invalid-shape')
            const projected = new RoutingJsonProjector().record(raw)
            expect(new TextDecoder().decode(projected)).not.toContain('synthetic-private-text')
            consumeRoutingHistoryRecord(projected, value => {
                expect(value).toEqual(corpus.cases[0].record)
                return undefined
            })
        }
    })

    it('rejects an unknown key in a packed selection and a truncated packed record', () => {
        const main = new RoutingHistoryBank()
        expect(main.apply(encoded(selectionRecord(accepted[1][1])))).toBe(true)
        const packed = main.at(0).slice()
        const renderer = new RoutingHistoryBank()
        expect(() => renderer.applyPacked(packed.subarray(0, packed.length - 1))).toThrow()
        expect(() => renderer.applyPacked(packed.subarray(0, packed.length - 20))).toThrow()
        expect(renderer.size).toBe(0)
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

describe('routing history bank storage', () => {
    // clear and copyFrom touch only the bytes each cell has held. Read the
    // private window directly: no public read shows bytes past a record's length.
    const windowOf = (bank: RoutingHistoryBank): Uint8Array => bank['data']
    const firstDifference = (a: Uint8Array, b: Uint8Array): number => {
        if (a.byteLength !== b.byteLength) return 0
        for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return i
        return -1
    }
    const version = (seq: number, model: string): Uint8Array => {
        const record: RoutingHistoryRecord = JSON.parse(JSON.stringify(corpus.cases[0].record))
        record.seq = seq
        record.requested.model = model
        return encoded(record)
    }

    it('copies and clears the bytes a shorter replacement left behind', () => {
        const source = new RoutingHistoryBank(),
            destination = new RoutingHistoryBank()
        expect(source.apply(version(3, 'm'.repeat(48)))).toBe(true)
        expect(source.apply(version(4, 'm'))).toBe(true)
        expect(destination.apply(version(3, 'd'.repeat(64)))).toBe(true)

        destination.copyFrom(source)
        expect(firstDifference(windowOf(destination), windowOf(source))).toBe(-1)
        expect(
            destination.consume(0, row => {
                expect(row.seq).toBe(4)
                expect(row.requested.model).toBe('m')
                return undefined
            })
        ).toBe(true)

        const empty = new Uint8Array(windowOf(source).byteLength)
        source.clear()
        destination.clear()
        expect(firstDifference(windowOf(source), empty)).toBe(-1)
        expect(firstDifference(windowOf(destination), empty)).toBe(-1)
        expect([source.size, destination.size]).toEqual([0, 0])
    })
})
