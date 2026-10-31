// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { parseInspectModel } from '@/electron/service-bridge/engine-settings'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'

function observation(): JsonObject {
    return {
        status: 'partial',
        current: true,
        reason: 'unmapped-device',
        sampleId: 'a'.repeat(64),
        recordedAt: '2026-10-28T00:00:00.123456789Z',
        persistence: 'uncertain',
        upstreamPin: '0ec4027a2a0183de87c007069adddc06e665f7de',
        buildAssurance: 'recognized-self-reported-pin',
        estimatorRevision:
            'pair-gguf/2;gguf-parser-go/v0.27.0@9257743a3c1775b882ba1b58ec7f5a856c2cc385',
        inputIdentity: 'b'.repeat(64),
        sourceIdentity: 'c'.repeat(64),
        optionsIdentity: 'd'.repeat(64),
        loadPhase: 'wakeup',
        perSlotContext: 8192,
        slots: 2,
        capturedOptions: {
            context: { mode: 'fixed', tokens: 8192 },
            slots: 2,
            gpu: { mode: 'cpu' }
        },
        requestedOptions: {
            context: { mode: 'auto', floor: 4096, cap: 8192 },
            gpu: { mode: 'cpu' }
        },
        optionUnknowns: ['mmap', 'load-mode'],
        artifacts: [
            {
                role: 'model',
                identity: 'e'.repeat(64),
                size: '9007199254740993',
                assurance: 'sampled-local-binding'
            }
        ],
        baseline: [{ component: 'weights', status: 'estimated', bytes: '1048576' }],
        comparisons: [
            {
                component: 'weights',
                status: 'compatible',
                estimatedBytes: '1048576',
                lowerBytes: '1043333',
                upperBytes: '1053819',
                deltaLowerBytes: '-5243',
                deltaUpperBytes: '5243'
            }
        ],
        rows: [
            {
                component: 'model',
                label: 'CPU',
                phase: 'load',
                sequence: '1',
                stream: 'stdout',
                role: 'model',
                reportedMiB: '1.00',
                unit: 'MiB',
                precision: '0.01',
                sourceKind: 'log-scraped',
                intervalStatus: 'bounded',
                lowerBytes: '1043333',
                upperBytes: '1053819',
                coverage: 'unknown'
            }
        ]
    }
}
function inspection(loadObservation?: JsonValue): JsonObject {
    return {
        engine: 'llamacpp',
        nodeId: 'self',
        modelId: 'exact-model',
        status: 'unknown',
        estimate: { status: 'unknown' },
        ...(loadObservation === undefined ? {} : { loadObservation })
    }
}
function object(value: JsonValue | undefined): JsonObject {
    if (!value || typeof value !== 'object' || Array.isArray(value))
        throw new Error('Invalid test object')
    return value
}
function first(value: JsonValue | undefined): JsonObject {
    if (!Array.isArray(value)) throw new Error('Invalid test array')
    return object(value[0])
}

describe('load buffer Inspect contract', () => {
    it('uses the neutral component vocabulary for estimates, baselines and comparisons', () => {
        const source = observation()
        source.baseline = [{ component: 'runtime-overhead', status: 'unknown' }]
        source.comparisons = [{ component: 'runtime-overhead', status: 'unknown' }]
        const input = inspection(source)
        input.estimate = {
            status: 'partial',
            components: [{ name: 'runtime-overhead', status: 'unknown', reason: 'unpriced' }]
        }
        const result = parseInspectModel(input)
        expect(result.estimate.components?.[0].name).toBe('runtime-overhead')
        expect(result.loadObservation?.baseline?.[0].component).toBe('runtime-overhead')
        expect(result.loadObservation?.comparisons?.[0].component).toBe('runtime-overhead')
    })
    it('preserves rounded service intervals, frozen baseline and self-reported assurance', () => {
        const source = observation()
        const parsed = parseInspectModel(inspection(source)).loadObservation
        expect(parsed).toMatchObject(source)
        expect(parsed?.artifacts?.[0].size).toBe('9007199254740993')
        first(source.rows).reportedMiB = '99.00'
        first(source.baseline).bytes = '0'
        object(source.capturedOptions).slots = 9
        expect(parsed?.rows?.[0].reportedMiB).toBe('1.00')
        expect(parsed?.baseline?.[0].bytes).toBe('1048576')
        expect(parsed?.capturedOptions?.slots).toBe(2)
    })
    it('maps an absent optional observation to Unknown omitted without claiming no load', () => {
        expect(parseInspectModel(inspection()).loadObservation).toEqual({
            status: 'unknown',
            current: false,
            reason: 'observation-omitted'
        })
    })
    it('accepts compact omission and unsupported-build Unknown summaries', () => {
        for (const reason of ['response-budget', 'unsupported-build', 'history-unavailable']) {
            expect(
                parseInspectModel(inspection({ status: 'unknown', current: false, reason }))
                    .loadObservation
            ).toMatchObject({ status: 'unknown', current: false, reason })
        }
    })
    it('retains a numeric report above the interval budget without exact byte claims', () => {
        const value = observation(),
            row = first(value.rows)
        row.reportedMiB = '1073741824.00'
        row.intervalStatus = 'unknown'
        delete row.lowerBytes
        delete row.upperBytes
        const parsed = parseInspectModel(inspection(value)).loadObservation?.rows?.[0]
        expect(parsed).toMatchObject({ reportedMiB: '1073741824.00', intervalStatus: 'unknown' })
        expect(parsed?.lowerBytes).toBeUndefined()
    })
    const malformed: Array<{ name: string; mutate: (value: JsonObject) => void }> = [
        {
            name: 'raw artifact path',
            mutate: value => {
                first(value.artifacts).identity = '/private/model.gguf'
            }
        },
        {
            name: 'unknown current sample',
            mutate: value => {
                value.status = 'unknown'
            }
        },
        {
            name: 'null options',
            mutate: value => {
                value.capturedOptions = null
            }
        },
        {
            name: 'unknown options',
            mutate: value => {
                object(value.capturedOptions).rawArgs = '--secret'
            }
        },
        {
            name: 'raw path identity',
            mutate: value => {
                value.sourceIdentity = '/private/model.gguf'
            }
        },
        {
            name: 'raw log field',
            mutate: value => {
                value.rawLog = 'secret output'
            }
        },
        {
            name: 'raw diagnostic reason',
            mutate: value => {
                value.reason = 'private log output'
            }
        },
        {
            name: 'unverified clean-source assertion',
            mutate: value => {
                value.buildAssurance = 'verified-clean-source'
            }
        },
        {
            name: 'abbreviated pin',
            mutate: value => {
                value.upstreamPin = '7fe450e'
            }
        },
        {
            name: 'arbitrary estimator text',
            mutate: value => {
                value.estimatorRevision = 'private/path'
            }
        },
        {
            name: 'invalid calendar date',
            mutate: value => {
                value.recordedAt = '2026-02-30T00:00:00Z'
            }
        },
        {
            name: 'unknown component',
            mutate: value => {
                first(value.baseline).component = 'whole-process'
            }
        },
        {
            name: 'duplicate baseline',
            mutate: value => {
                value.baseline = [first(value.baseline), first(value.baseline)]
            }
        },
        {
            name: 'duplicate comparison',
            mutate: value => {
                value.comparisons = [first(value.comparisons), first(value.comparisons)]
            }
        },
        {
            name: 'duplicate source frame',
            mutate: value => {
                value.rows = [first(value.rows), first(value.rows)]
            }
        },
        {
            name: 'different frozen estimate',
            mutate: value => {
                first(value.comparisons).estimatedBytes = '1'
            }
        },
        {
            name: 'fabricated delta',
            mutate: value => {
                first(value.comparisons).deltaUpperBytes = '5244'
            }
        },
        {
            name: 'partial baseline with compatible comparison',
            mutate: value => {
                first(value.baseline).status = 'partial'
            }
        },
        {
            name: 'negative zero delta',
            mutate: value => {
                first(value.comparisons).deltaLowerBytes = '-0'
            }
        },
        {
            name: 'noncanonical byte quantity',
            mutate: value => {
                first(value.baseline).bytes = '01'
            }
        },
        {
            name: 'overflowing quantity',
            mutate: value => {
                first(value.baseline).bytes = '18446744073709551616'
            }
        },
        {
            name: 'legacy Metal label',
            mutate: value => {
                first(value.rows).label = 'Metal0'
            }
        },
        {
            name: 'out of bound CUDA label',
            mutate: value => {
                first(value.rows).label = 'CUDA64'
            }
        },
        {
            name: 'leading zero device label',
            mutate: value => {
                first(value.rows).label = 'CUDA00'
            }
        },
        {
            name: 'raw device description',
            mutate: value => {
                first(value.rows).poolIdentity = 'GPU name'
            }
        },
        {
            name: 'NaN report',
            mutate: value => {
                first(value.rows).reportedMiB = 'NaN'
            }
        },
        {
            name: 'negative report',
            mutate: value => {
                first(value.rows).reportedMiB = '-1.00'
            }
        },
        {
            name: 'incorrect rounded endpoints',
            mutate: value => {
                first(value.rows).lowerBytes = '1048576'
            }
        },
        {
            name: 'unknown interval with endpoints',
            mutate: value => {
                first(value.rows).intervalStatus = 'unknown'
            }
        },
        {
            name: 'fit JSONL source',
            mutate: value => {
                first(value.rows).sourceKind = 'fit_memory_breakdown'
            }
        },
        {
            name: 'unsupported unit',
            mutate: value => {
                first(value.rows).unit = 'MB'
            }
        },
        {
            name: 'zero frame sequence',
            mutate: value => {
                first(value.rows).sequence = '0'
            }
        },
        {
            name: 'null rows',
            mutate: value => {
                value.rows = null
            }
        },
        {
            name: 'too many reason codes',
            mutate: value => {
                value.reasons = Array.from({ length: 17 }, () => 'capture-budget')
            }
        },
        {
            name: 'too many option entries',
            mutate: value => {
                value.optionUnknowns = Array.from({ length: 65 }, () => 'mmap')
            }
        },
        {
            name: 'summary encoded budget',
            mutate: value => {
                value.rows = Array.from({ length: 30 }, (_, index) => ({
                    ...first(value.rows),
                    sequence: String(index + 1)
                }))
            }
        },
        {
            name: 'row count budget',
            mutate: value => {
                value.rows = Array.from({ length: 257 }, (_, index) => ({
                    ...first(value.rows),
                    sequence: String(index + 1)
                }))
            }
        }
    ]
    it.each(malformed)('rejects $name', ({ mutate }) => {
        const value = observation()
        mutate(value)
        expect(() => parseInspectModel(inspection(value))).toThrow()
    })
    it('rejects null observations and whole-reply payload pressure', () => {
        expect(() => parseInspectModel(inspection(null))).toThrow()
        expect(() => parseInspectModel({ ...inspection(), reason: 'x'.repeat(65536) })).toThrow()
    })
})
