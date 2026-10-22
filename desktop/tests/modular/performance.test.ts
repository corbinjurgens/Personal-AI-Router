// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'
import {
    parsePerformanceProjection,
    PerformanceSubscription
} from '@/electron/service-bridge/performance'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { emitBridgePush } from '@/electron/service-bridge/broadcaster'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({ emitBridgePush: vi.fn() }))

function entry(extra: JsonObject = {}): JsonObject {
    return {
        key: {
            host: 'peer',
            engine: 'llamacpp-custom',
            model: 'model.gguf',
            fingerprint: 'profile-1'
        },
        profile: { engineFamily: 'llamacpp', quantization: 'Q4_K_M', contextTokens: 8192 },
        samples: 1,
        completed: 1,
        failed: 0,
        cancelled: 0,
        incomplete: 0,
        newestAgeMs: 20,
        oldestAgeMs: 30,
        newestObservedAtMs: 1000,
        headerMs: { samples: 1, median: 0, p95: 0 },
        ...extra
    }
}

function summary(extra: JsonObject = {}): JsonObject {
    return {
        version: 2,
        measuringHost: 'peer',
        admissionIncarnation: '42',
        revision: 2,
        window: { maxAgeMs: 604800000, maxSamplesPerKey: 64 },
        entries: [entry()],
        ...extra
    }
}

function peer(extra: JsonObject = {}): JsonObject {
    return {
        measuringHost: 'peer',
        status: 'fresh',
        lastSeenAgeMs: 5,
        expiredEntries: 0,
        summary: summary(),
        ...extra
    }
}

function projection(peers: JsonValue[] = [peer()], extra: JsonObject = {}): JsonObject {
    return {
        local: {
            measuringHost: 'local',
            status: 'missing',
            lastSeenAgeMs: 0,
            expiredEntries: 0,
            summary: summary({
                measuringHost: 'local',
                admissionIncarnation: '',
                revision: 0,
                entries: []
            })
        },
        peers,
        expiredPeers: 0,
        persistence: {
            degraded: false,
            publicationBlocked: false,
            error: '/private/path',
            stats: {
                Keys: 0,
                Samples: 0,
                Dirty: false,
                PublishedRevision: 0,
                HistoryReset: false,
                EvictedKeys: 2,
                EvictedSamples: 3,
                ExpiredSamples: 4,
                Duplicates: 0
            }
        },
        ...extra
    }
}

function distinctEntries(length: number): JsonObject[] {
    return Array.from({ length }, (_, index) =>
        entry({
            key: { host: 'peer', engine: 'engine', model: 'model', fingerprint: `fp-${index}` }
        })
    )
}

function distinctPeers(length: number): JsonObject[] {
    return Array.from({ length }, (_, index) =>
        peer({ measuringHost: `peer-${index}`, status: 'missing', summary: null })
    )
}

const state = getModularBridgeState()
function change(value: JsonValue | undefined): void {
    state.handleNotification({ source: 'broker', method: 'performance:changed', params: value })
}
function discover(host = 'peer', port = 14318): void {
    state.handleNotification({
        source: 'broker',
        method: 'discovery:nodes-changed',
        params: {
            nodes: [
                { hostUuid: host, name: 'Test host', ipAddress: '192.0.2.20', port, lastSeen: 1000 }
            ]
        }
    })
}

describe('bounded broker performance projection', () => {
    it('preserves actual profile, measured zero and nullable unknowns', () => {
        const result = parsePerformanceProjection(projection())
        expect(result.get('peer')?.summary?.entries[0]).toEqual({
            key: {
                host: 'peer',
                engine: 'llamacpp-custom',
                model: 'model.gguf',
                fingerprint: 'profile-1'
            },
            profile: { engineFamily: 'llamacpp', quantization: 'Q4_K_M', contextTokens: 8192 },
            samples: 1,
            completed: 1,
            failed: 0,
            cancelled: 0,
            incomplete: 0,
            outputTokens: null,
            newestAgeMs: 20,
            oldestAgeMs: 30,
            newestObservedAtMs: 1000,
            clockUncertain: false,
            headerMs: { samples: 1, median: 0, p95: 0 },
            admissionWaitMs: null,
            firstBodyMs: null,
            firstTokenMs: null,
            totalMs: null,
            loadNs: null,
            prefill: null,
            decode: null,
            engineDecode: null
        })
        expect(result.get('local')).toMatchObject({
            status: 'missing',
            summary: { revision: 0, admissionIncarnation: '', entries: [] },
            evictions: { keys: 2, samples: 3, expiredSamples: 4 }
        })
        expect(JSON.stringify([...result.values()])).not.toContain('/private/path')
        expect(JSON.stringify([...result.values()])).not.toContain('PublishedRevision')
    })

    const acceptedOutputTotals: [string, JsonObject, number | null][] = [
        ['absent', {}, null],
        ['explicit null', { outputTokens: null }, null],
        ['measured zero', { outputTokens: 0 }, 0],
        ['known total without a decode rate', { outputTokens: 42 }, 42],
        [
            'qualified maximum',
            { samples: 3, completed: 3, incomplete: 1, outputTokens: 200_000_000 },
            200_000_000
        ],
        [
            'retention maximum',
            { samples: 64, completed: 64, outputTokens: 6_400_000_000 },
            6_400_000_000
        ]
    ]
    it.each(acceptedOutputTotals)(
        'preserves %s output tokens independently of decode rates',
        (_name, extra, expected) => {
            const measured = parsePerformanceProjection(
                projection([peer({ summary: summary({ entries: [entry(extra)] }) })])
            ).get('peer')?.summary?.entries[0]
            expect(measured?.outputTokens).toBe(expected)
            expect(measured?.decode).toBeNull()
            expect(measured?.engineDecode).toBeNull()
        }
    )

    const invalidOutputTotals: [string, JsonObject][] = [
        ['negative', { outputTokens: -1 }],
        ['fractional', { outputTokens: 0.5 }],
        ['string', { outputTokens: '1' }],
        ['boolean', { outputTokens: false }],
        ['object', { outputTokens: {} }],
        ['array', { outputTokens: [] }],
        ['nonfinite', { outputTokens: Infinity }],
        ['unsafe integer', { outputTokens: Number.MAX_SAFE_INTEGER + 1 }],
        ['over bound', { outputTokens: 100_000_001 }],
        [
            'incomplete coverage',
            { samples: 3, completed: 3, incomplete: 1, outputTokens: 200_000_001 }
        ],
        ['no qualified completion', { incomplete: 1, outputTokens: 1 }],
        ['zero without qualified completions', { incomplete: 1, outputTokens: 0 }],
        ['zero with only failed samples', { completed: 0, failed: 1, outputTokens: 0 }],
        ['over retention maximum', { samples: 64, completed: 64, outputTokens: 6_400_000_001 }]
    ]
    it.each(invalidOutputTotals)('rejects %s output tokens', (_name, extra) => {
        expect(
            parsePerformanceProjection(
                projection([peer({ summary: summary({ entries: [entry(extra)] }) })])
            ).size
        ).toBe(0)
    })

    it('accepts broker-aged oldest samples beyond wire retention', () => {
        const value = projection([
            peer({
                summary: summary({
                    entries: [entry({ oldestAgeMs: 604800100, clockUncertain: true })]
                })
            })
        ])
        expect(
            parsePerformanceProjection(value).get('peer')?.summary?.entries[0]?.oldestAgeMs
        ).toBe(604800100)
    })

    it('normalizes explicit null metrics and retains zero weighted rates', () => {
        const value = projection([
            peer({
                summary: summary({
                    entries: [
                        entry({
                            totalMs: null,
                            decode: { samples: 1, tokens: 0, durationNs: 100, tokensPerSecond: 0 }
                        })
                    ]
                })
            })
        ])
        const measured = parsePerformanceProjection(value).get('peer')?.summary?.entries[0]
        expect(measured?.totalMs).toBeNull()
        expect(measured?.decode).toEqual({
            samples: 1,
            tokens: 0,
            durationNs: 100,
            tokensPerSecond: 0
        })
    })

    it.each([49, 0])('preserves reported decode rate %s separately from legacy decode', speed => {
        const engineDecode = { samples: 1, durationNs: 1_000_000_000, tokensPerSecond: speed }
        const decode = { samples: 1, tokens: 100, durationNs: 1_000_000_000, tokensPerSecond: 100 }
        const measured = parsePerformanceProjection(
            projection([peer({ summary: summary({ entries: [entry({ engineDecode, decode })] }) })])
        ).get('peer')?.summary?.entries[0]
        expect(measured?.engineDecode).toEqual(engineDecode)
        expect(measured?.decode).toEqual(decode)
        expect(measured?.key).toEqual({
            host: 'peer',
            engine: 'llamacpp-custom',
            model: 'model.gguf',
            fingerprint: 'profile-1'
        })
    })

    it('normalizes explicit null reported decode to unknown', () => {
        const measured = parsePerformanceProjection(
            projection([peer({ summary: summary({ entries: [entry({ engineDecode: null })] }) })])
        ).get('peer')?.summary?.entries[0]
        expect(measured?.engineDecode).toBeNull()
    })

    it('retains destination Strata identity and fractional native interval without deriving output throughput', () => {
        const key = {
            host: 'peer',
            engine: 'strata-custom',
            model: 'resolved-child/model',
            fingerprint: 'strata-runtime-config-profile'
        }
        const profile = { engineFamily: 'strata', quantization: 'Q6_K', contextTokens: 16384 }
        const engineDecode = { samples: 1, durationNs: 400_500_000, tokensPerSecond: 55.1 }
        const measured = parsePerformanceProjection(
            projection([
                peer({
                    summary: summary({
                        entries: [entry({ key, profile, engineDecode, decode: null })]
                    })
                })
            ])
        ).get('peer')?.summary?.entries[0]

        expect(measured?.key).toEqual(key)
        expect(measured?.profile).toEqual(profile)
        expect(measured?.engineDecode).toEqual(engineDecode)
        expect(measured?.decode).toBeNull()
        expect(measured?.firstTokenMs).toBeNull()
    })

    const invalidReportedRates: [string, JsonObject][] = [
        ['missing samples', { durationNs: 100, tokensPerSecond: 1 }],
        ['null samples', { samples: null, durationNs: 100, tokensPerSecond: 1 }],
        ['missing duration', { samples: 1, tokensPerSecond: 1 }],
        ['null duration', { samples: 1, durationNs: null, tokensPerSecond: 1 }],
        ['zero duration', { samples: 1, durationNs: 0, tokensPerSecond: 1 }],
        ['excessive duration', { samples: 1, durationNs: 86_400_000_000_001, tokensPerSecond: 1 }],
        ['missing rate', { samples: 1, durationNs: 100 }],
        ['null rate', { samples: 1, durationNs: 100, tokensPerSecond: null }],
        ['negative rate', { samples: 1, durationNs: 100, tokensPerSecond: -1 }],
        ['nonfinite rate', { samples: 1, durationNs: 100, tokensPerSecond: Infinity }],
        ['excessive rate', { samples: 1, durationNs: 100, tokensPerSecond: 10_000_001 }],
        ['unqualified samples', { samples: 2, durationNs: 100, tokensPerSecond: 1 }],
        ['invented tokens', { samples: 1, durationNs: 100, tokensPerSecond: 1, tokens: 1 }]
    ]
    it.each(invalidReportedRates)('rejects reported decode with %s', (_name, engineDecode) => {
        expect(
            parsePerformanceProjection(
                projection([peer({ summary: summary({ entries: [entry({ engineDecode })] }) })])
            ).size
        ).toBe(0)
    })

    it('rejects reported decode coverage that includes protocol-incomplete samples', () => {
        expect(
            parsePerformanceProjection(
                projection([
                    peer({
                        summary: summary({
                            entries: [
                                entry({
                                    incomplete: 1,
                                    engineDecode: {
                                        samples: 1,
                                        durationNs: 100,
                                        tokensPerSecond: 1
                                    }
                                })
                            ]
                        })
                    })
                ])
            ).size
        ).toBe(0)
    })

    const invalidEntries: [string, JsonObject][] = [
        ['nonfinite metric', { headerMs: { samples: 1, median: Infinity, p95: Infinity } }],
        ['negative count', { failed: -1 }],
        ['outcome partition', { completed: 0 }],
        [
            'host mismatch',
            { key: { host: 'other', engine: 'engine', model: 'model', fingerprint: 'fp' } }
        ],
        ['private entry field', { prompt: 'secret' }],
        ['fractional age', { newestAgeMs: 1.2 }],
        ['invalid profile', { profile: { contextTokens: 0 } }],
        ['reversed percentile', { totalMs: { samples: 1, median: 10, p95: 0 } }],
        ['too many samples', { samples: 65, completed: 65 }],
        [
            'excessive rate',
            { prefill: { samples: 1, tokens: 1, durationNs: 1, tokensPerSecond: 10000001 } }
        ]
    ]
    it.each(invalidEntries)('rejects %s atomically', (_name, extra) => {
        expect(
            parsePerformanceProjection(
                projection([peer({ summary: summary({ entries: [entry(extra)] }) })])
            ).size
        ).toBe(0)
    })

    it.each([
        [
            'duplicate keys',
            projection([peer({ summary: summary({ entries: [entry(), entry()] }) })])
        ],
        ['peer revision zero', projection([peer({ summary: summary({ revision: 0 }) })])],
        ['v1 summary', projection([peer({ summary: summary({ version: 1 }) })])],
        [
            'expired newest sample',
            projection([
                peer({
                    summary: summary({
                        entries: [entry({ newestAgeMs: 604800001, oldestAgeMs: 604800002 })]
                    })
                })
            ])
        ],
        ['unknown with summary', projection([peer({ status: 'unknown' })])],
        ['missing with samples', projection([peer({ status: 'missing' })])],
        ['fresh without summary', projection([peer({ summary: null })])],
        ['stale without summary', projection([peer({ status: 'stale', summary: null })])],
        [
            'peer missing epoch',
            projection([peer({ summary: summary({ admissionIncarnation: '' }) })])
        ],
        ['duplicate hosts', projection([peer(), peer()])],
        ['node limit', projection(distinctPeers(129))],
        [
            'entry limit',
            projection([peer({ summary: summary({ entries: distinctEntries(129) }) })])
        ],
        ['projection bytes', projection([], { padding: 'x'.repeat(17 * 1024 * 1024) })],
        ['private node field', projection([peer({ admissionIdentity: 'secret' })])],
        ['unknown persistence field', projection([], { persistence: { raw: 'secret' } })]
    ])('rejects %s', (_name, value) => {
        expect(parsePerformanceProjection(value).size).toBe(0)
    })

    it('accepts the maximum distinct peer and entry cardinalities', () => {
        expect(parsePerformanceProjection(projection(distinctPeers(128))).size).toBe(129)
        expect(
            parsePerformanceProjection(
                projection([peer({ summary: summary({ entries: distinctEntries(128) }) })])
            ).get('peer')?.summary?.entries
        ).toHaveLength(128)
    })

    it('rejects an oversized summary with otherwise valid fields and distinct keys', () => {
        const entries = Array.from({ length: 128 }, (_, index) =>
            entry({
                key: {
                    host: 'peer',
                    engine: 'e'.repeat(128),
                    model: 'm'.repeat(256),
                    fingerprint: `fp-${index}`.padEnd(128, 'f')
                },
                profile: {
                    engineFamily: 'e'.repeat(128),
                    quantization: 'q'.repeat(64),
                    contextTokens: 100000000
                },
                admissionWaitMs: { samples: 1, median: 86400000, p95: 86400000 },
                headerMs: { samples: 1, median: 86400000, p95: 86400000 },
                firstBodyMs: { samples: 1, median: 86400000, p95: 86400000 },
                firstTokenMs: { samples: 1, median: 86400000, p95: 86400000 },
                totalMs: { samples: 1, median: 86400000, p95: 86400000 },
                loadNs: { samples: 1, median: 86400000000000, p95: 86400000000000 },
                prefill: {
                    samples: 1,
                    tokens: 100000000,
                    durationNs: 86400000000000,
                    tokensPerSecond: 1
                },
                decode: {
                    samples: 1,
                    tokens: 100000000,
                    durationNs: 86400000000000,
                    tokensPerSecond: 1
                }
            })
        )
        const large = summary({ entries })
        expect(Buffer.byteLength(JSON.stringify(large))).toBeGreaterThan(128 * 1024)
        expect(parsePerformanceProjection(projection([peer({ summary: large })])).size).toBe(0)
        expect(
            parsePerformanceProjection(
                projection([peer({ summary: summary({ entries: entries.slice(0, 64) }) })])
            ).get('peer')?.summary?.entries
        ).toHaveLength(64)
    })

    it.each(['fresh', 'stale', 'unknown', 'missing'])(
        'retains coherent broker %s status without invented measurements',
        status => {
            const parsed = parsePerformanceProjection(
                projection([
                    peer({
                        status,
                        summary: status === 'fresh' || status === 'stale' ? summary() : null
                    })
                ])
            ).get('peer')
            expect(parsed).toMatchObject({
                measuringHost: 'peer',
                status,
                lastSeenAgeMs: 5,
                expiredEntries: 0
            })
            expect(parsed?.summary?.entries[0]?.firstTokenMs).toBe(
                status === 'fresh' || status === 'stale' ? null : undefined
            )
        }
    )
})

describe('node snapshot performance attachment', () => {
    beforeEach(() => {
        change(undefined)
        discover()
        vi.mocked(emitBridgePush).mockClear()
    })

    it('replaces output totals in node snapshots and pushes without retaining unknown counts', () => {
        for (const outputTokens of [42, 0, null]) {
            change(
                projection([peer({ summary: summary({ entries: [entry({ outputTokens })] }) })])
            )
            expect(
                state.getNodesInitial().nodes.peer?.performance?.summary?.entries[0]?.outputTokens
            ).toBe(outputTokens)
            expect(emitBridgePush).toHaveBeenLastCalledWith(
                'nodes:upsert',
                expect.objectContaining({
                    performance: expect.objectContaining({
                        summary: expect.objectContaining({
                            entries: [expect.objectContaining({ outputTokens })]
                        })
                    })
                })
            )
        }
    })

    it('emits a performance-only node update without creating workloads or engine events', () => {
        change(projection())
        expect(state.getNodesInitial().nodes.peer?.performance?.status).toBe('fresh')
        expect(vi.mocked(emitBridgePush).mock.calls.map(call => call[0])).toEqual(['nodes:upsert'])
        vi.mocked(emitBridgePush).mockClear()
        change(projection())
        expect(emitBridgePush).not.toHaveBeenCalled()
    })

    it('removes prior fresh data on a malformed replacement while preserving identity', () => {
        change(projection())
        change(
            projection([
                peer({ summary: summary({ entries: [entry({ messages: ['private'] })] }) })
            ])
        )
        expect(state.getNodesInitial().nodes.peer).toMatchObject({ id: 'peer', name: 'Test host' })
        expect(state.getNodesInitial().nodes.peer?.performance).toBeUndefined()
    })

    it('replaces a revision without adding samples on repeated or aged summaries', () => {
        change(projection())
        change(
            projection([
                peer({
                    summary: summary({
                        revision: 3,
                        entries: [entry({ newestAgeMs: 100, oldestAgeMs: 200 })]
                    })
                })
            ])
        )
        change(
            projection([
                peer({
                    summary: summary({
                        revision: 3,
                        entries: [entry({ newestAgeMs: 110, oldestAgeMs: 210 })]
                    })
                })
            ])
        )
        expect(state.getNodesInitial().nodes.peer?.performance?.summary).toMatchObject({
            revision: 3,
            entries: [{ samples: 1, newestAgeMs: 110, oldestAgeMs: 210 }]
        })
    })

    it('replaces native rate summaries without summing counts or retaining a prior rate', () => {
        const reported = (revision: number, engineDecode: JsonValue): JsonObject =>
            projection([
                peer({
                    summary: summary({
                        revision,
                        entries: [
                            entry({
                                samples: 3,
                                completed: 2,
                                cancelled: 1,
                                engineDecode,
                                decode: null
                            })
                        ]
                    })
                })
            ])
        const first = reported(3, {
            samples: 2,
            durationNs: 1_000_000_000,
            tokensPerSecond: 49
        })
        change(first)
        change(first)
        expect(state.getNodesInitial().nodes.peer?.performance?.summary?.entries[0]).toMatchObject({
            samples: 3,
            completed: 2,
            cancelled: 1,
            engineDecode: { samples: 2, durationNs: 1_000_000_000, tokensPerSecond: 49 }
        })

        change(reported(4, { samples: 1, durationNs: 400_500_000, tokensPerSecond: 0 }))
        expect(state.getNodesInitial().nodes.peer?.performance?.summary?.entries[0]).toMatchObject({
            samples: 3,
            engineDecode: { samples: 1, durationNs: 400_500_000, tokensPerSecond: 0 }
        })
        change(reported(5, null))
        expect(state.getNodesInitial().nodes.peer?.performance?.summary?.entries[0]).toMatchObject({
            samples: 3,
            engineDecode: null,
            decode: null
        })
    })

    it('keeps route inputs and engine selections unchanged by native performance replacements', () => {
        const available = state.getAvailableNodes()
        const pollTargets = state.getNodeInfoPollTargets()
        const engines = state.getEngineInitialState()
        const workloads = state.getWorkloads()
        const snapshot = projection([
            peer({
                summary: summary({
                    entries: [
                        entry({
                            engineDecode: {
                                samples: 1,
                                durationNs: 400_500_000,
                                tokensPerSecond: 55.1
                            }
                        })
                    ]
                })
            })
        ])

        change(snapshot)
        change(snapshot)

        expect(state.getAvailableNodes()).toEqual(available)
        expect(state.getNodeInfoPollTargets()).toEqual(pollTargets)
        expect(state.getEngineInitialState()).toEqual(engines)
        expect(state.getWorkloads()).toEqual(workloads)
        expect(vi.mocked(emitBridgePush).mock.calls.map(call => call[0])).toEqual(['nodes:upsert'])
    })

    it('attaches cached data to late discovery and never to a different replacement host', () => {
        discover('unrelated')
        change(projection())
        discover()
        expect(state.getNodesInitial().nodes.peer?.performance?.measuringHost).toBe('peer')
        discover('replacement')
        expect(state.getNodesInitial().nodes.peer).toBeUndefined()
        expect(state.getNodesInitial().nodes.replacement?.performance).toBeUndefined()
    })

    it('clears performance on disconnect while retaining the offline node identity', () => {
        discover('peer', 0)
        change(projection())
        expect(state.getNodesInitial().nodes.peer?.status).toBe('offline')
        change(undefined)
        expect(state.getNodesInitial().nodes.peer?.id).toBe('peer')
        expect(state.getNodesInitial().nodes.peer?.performance).toBeUndefined()
    })
})

describe('performance subscription lifecycle', () => {
    it('does not bootstrap when stop races an outstanding subscription reply', async () => {
        let finishSubscribe: (value: JsonValue) => void = () => {
            throw new Error('subscribe not requested')
        }
        const call = vi.fn<(method: string) => Promise<JsonValue | undefined>>(method =>
            method === 'performance:subscribe'
                ? new Promise(resolve => {
                      finishSubscribe = resolve
                  })
                : Promise.resolve(null)
        )
        const apply = vi.fn<(value: JsonValue | undefined) => void>()
        const subscription = new PerformanceSubscription(call, apply)
        subscription.ready()
        await subscription.stop()
        finishSubscribe({ subscribed: true })
        await Promise.resolve()
        await Promise.resolve()
        expect(call.mock.calls).toEqual([['performance:subscribe'], ['performance:unsubscribe']])
        expect(apply).toHaveBeenCalledExactlyOnceWith(undefined)
    })

    it('subscribes once, bootstraps and unsubscribes before caller closes the connection', async () => {
        const events: string[] = []
        const call = vi.fn<(method: string) => Promise<JsonValue | undefined>>(async method => {
            events.push(method)
            return projection()
        })
        const apply = vi.fn<(value: JsonValue | undefined) => void>()
        const subscription = new PerformanceSubscription(call, apply)
        subscription.ready()
        subscription.ready()
        await vi.waitFor(() => expect(apply).toHaveBeenCalledWith(projection()))
        await subscription.stop()
        events.push('close')
        expect(events).toEqual([
            'performance:subscribe',
            'performance:get',
            'performance:unsubscribe',
            'close'
        ])
        expect(apply).toHaveBeenLastCalledWith(undefined)
        await subscription.stop()
        expect(call).toHaveBeenCalledTimes(3)
    })

    it('fences a late bootstrap after a newer push', async () => {
        let resolveBootstrap: (value: JsonValue) => void = () => {
            throw new Error('bootstrap not requested')
        }
        const call = vi.fn<(method: string) => Promise<JsonValue | undefined>>(method =>
            method === 'performance:get'
                ? new Promise(resolve => {
                      resolveBootstrap = resolve
                  })
                : Promise.resolve(null)
        )
        const apply = vi.fn<(value: JsonValue | undefined) => void>()
        const subscription = new PerformanceSubscription(call, apply)
        subscription.ready()
        await vi.waitFor(() => expect(call).toHaveBeenCalledWith('performance:get'))
        subscription.changed(projection([peer({ status: 'stale' })]))
        resolveBootstrap(projection())
        await Promise.resolve()
        await Promise.resolve()
        expect(apply).toHaveBeenCalledExactlyOnceWith(projection([peer({ status: 'stale' })]))
    })

    it('fences old broker generation and ignores pushes while disconnected', async () => {
        let resolveOld: (value: JsonValue) => void = () => {
            throw new Error('bootstrap not requested')
        }
        const call = vi.fn<(method: string) => Promise<JsonValue | undefined>>(method =>
            method === 'performance:get'
                ? new Promise(resolve => {
                      resolveOld = resolve
                  })
                : Promise.resolve(null)
        )
        const apply = vi.fn<(value: JsonValue | undefined) => void>()
        const subscription = new PerformanceSubscription(call, apply)
        subscription.ready()
        await vi.waitFor(() => expect(call).toHaveBeenCalledWith('performance:get'))
        subscription.reset()
        subscription.changed(projection())
        resolveOld(projection())
        await Promise.resolve()
        await Promise.resolve()
        expect(apply).toHaveBeenCalledExactlyOnceWith(undefined)
        subscription.ready()
        expect(call.mock.calls.filter(args => args[0] === 'performance:subscribe')).toHaveLength(2)
        await subscription.stop()
    })
})
