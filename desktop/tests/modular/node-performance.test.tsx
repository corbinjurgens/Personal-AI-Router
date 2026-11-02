// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import type {
    NodePerformance as NodePerformanceData,
    PerformanceEntry
} from '@/shared/types/performance'
import NodePerformance from '@/ui/components/NodeList/NodePerformance'
import { useMetricsStore } from '@/ui/stores/metrics.store'

vi.mock('@/ui/stores/metrics.store', async importOriginal => {
    const actual = await importOriginal<typeof import('@/ui/stores/metrics.store')>()
    return {
        ...actual,
        useMetricsStore: Object.assign(
            <T,>(selector: (state: ReturnType<typeof useMetricsStore.getState>) => T): T =>
                selector(actual.useMetricsStore.getState()),
            actual.useMetricsStore
        )
    }
})

vi.mock('@/ui/components/SvgLineChart', () => ({
    default: function Chart() {
        return <div aria-label="Hardware utilization chart" />
    }
}))

const node: NodeItem = {
    id: 'node-one',
    name: 'Node one',
    status: 'active',
    ipAddress: '127.0.0.1',
    port: 0,
    allIpAddresses: [],
    os: 'Windows',
    topology: {
        cpu: { model: 'Test CPU', cores: 4, threads: 8 },
        gpus: [{ id: 'gpu-one', name: 'Test GPU', vramTotal: 1024 }],
        inferenceHardwareIds: [],
        ram: 4096,
        storage: []
    }
}

function entry(): PerformanceEntry {
    return {
        key: {
            host: 'source-one',
            engine: 'llamacpp-custom',
            model: 'model-one',
            fingerprint: 'fp-one'
        },
        profile: { engineFamily: 'llamacpp', quantization: 'Q4_K_M', contextTokens: 8192 },
        samples: 8,
        completed: 6,
        failed: 1,
        cancelled: 1,
        incomplete: 3,
        newestAgeMs: 1234,
        oldestAgeMs: 4567,
        newestObservedAtMs: 1000,
        clockUncertain: false,
        admissionWaitMs: { samples: 6, median: 1, p95: 2 },
        headerMs: { samples: 2, median: 12, p95: 20 },
        firstBodyMs: { samples: 3, median: 21, p95: 40 },
        firstTokenMs: { samples: 4, median: 31, p95: 50 },
        totalMs: { samples: 5, median: 100, p95: 150 },
        loadNs: { samples: 1, median: 1_500_000, p95: 2_500_000 },
        prefill: { samples: 3, tokens: 100, durationNs: 2_000_000_000, tokensPerSecond: 50 },
        decode: { samples: 2, tokens: 20, durationNs: 1_000_000_000, tokensPerSecond: 20 },
        engineDecode: null
    }
}

function performance(entries: PerformanceEntry[] = [entry()]): NodePerformanceData {
    const measuringHost = entries[0]?.key.host ?? 'source-one'
    return {
        measuringHost,
        status: 'fresh',
        lastSeenAgeMs: 100,
        expiredEntries: 2,
        summary: {
            version: 2,
            measuringHost,
            admissionIncarnation: 'incarnation-one',
            revision: 1,
            window: { maxAgeMs: 604_800_000, maxSamplesPerKey: 64 },
            entries,
            omittedEntries: 3
        }
    }
}

let markup = ''

beforeEach(() => {
    vi.spyOn(Date, 'now').mockReturnValue(1000)
    useMetricsStore.getState().clearAll()
})

afterEach(() => {
    useMetricsStore.getState().clearAll()
})

function render(performanceData?: NodePerformanceData): void {
    markup = renderToStaticMarkup(
        <NodePerformance node={{ ...node, performance: performanceData }} />
    )
}

function text(): string {
    return markup.replace(/<[^>]*>/g, '')
}

function tableCount(): number {
    return Array.from(markup.matchAll(/<table\b/g)).length
}

function cells(label: string): string[] {
    const rows = Array.from(markup.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g), row =>
        Array.from(row[1].matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/g), cell =>
            cell[1].replace(/<[^>]*>/g, '')
        )
    )
    const found = rows.find(row => row[0] === label)
    if (!found) throw new Error(`Missing metric row: ${label}`)
    return found
}

describe('measured run performance', () => {
    it('shows an explicit unknown empty state without hardware history', () => {
        render()
        expect(markup).toContain('aria-label="Measured run performance"')
        expect(text()).toContain('Measured runs · Unknown')
        expect(text()).toContain('No measured runs reported for this node.')
        expect(text()).toContain('Last seen age: -- ms')
        expect(text()).toContain('No metrics yet')
        expect(text()).toContain('Test GPU')
        expect(text()).toContain('-- / 1 KB')
        expect(tableCount()).toBe(0)
    })

    it('keeps TTFT separate from body/header latency and shows per-metric coverage', () => {
        render(performance())
        expect(cells('TTFT (first output token)')).toEqual([
            'TTFT (first output token)',
            '31',
            '50',
            '4 / 8'
        ])
        expect(cells('Response headers')).toEqual(['Response headers', '12', '20', '2 / 8'])
        expect(cells('First body byte')).toEqual(['First body byte', '21', '40', '3 / 8'])
        expect(cells('Total execution')).toEqual(['Total execution', '100', '150', '5 / 8'])
        expect(cells('Admission wait')).toEqual(['Admission wait', '1', '2', '6 / 8'])
        expect(cells('Engine load')).toEqual(['Engine load', '1.50', '2.50', '1 / 8'])
        expect(cells('Prefill')).toEqual(['Prefill', '50', '3 / 8'])
        expect(cells('Decode')).toEqual(['Decode', '20', '2 / 8'])
    })

    it('preserves null latency and rates as unknown while rendering measured zero', () => {
        const measured = entry()
        measured.firstTokenMs = null
        measured.headerMs = { samples: 2, median: 0, p95: 0 }
        measured.prefill = null
        measured.decode = { samples: 2, tokens: 0, durationNs: 1_000_000, tokensPerSecond: 0 }
        measured.profile = { engineFamily: '', quantization: '', contextTokens: null }
        render(performance([measured]))
        expect(cells('TTFT (first output token)')).toEqual([
            'TTFT (first output token)',
            '--',
            '--',
            '0 / 8'
        ])
        expect(cells('Response headers')).toEqual(['Response headers', '0', '0', '2 / 8'])
        expect(cells('Prefill')).toEqual(['Prefill', '--', '0 / 8'])
        expect(cells('Decode')).toEqual(['Decode', '0', '2 / 8'])
        expect(text()).toContain('Family: -- · Quantization: -- · Context tokens: --')
    })

    it('shows the reported decode aggregate and its coverage without mixing legacy rates', () => {
        const measured = entry()
        measured.engineDecode = {
            samples: 3,
            durationNs: 1_000_000_000,
            tokensPerSecond: 49.126
        }
        render(performance([measured]))
        expect(cells('Engine-reported decode')).toEqual([
            'Engine-reported decode',
            '49.13',
            '3 / 8'
        ])
        expect(text()).toContain('duration-weighted mean of reported rates')
        expect(text()).toContain('may be rounded by the engine')
        expect(text()).toContain('does not reconstruct a token total')
        expect(text()).toContain('Source host: source-one · Profile: fp-one')
        expect(() => cells('Decode')).toThrow('Missing metric row: Decode')
    })

    it('shows a measured zero engine-reported rate with its own coverage', () => {
        const measured = entry()
        measured.engineDecode = { samples: 1, durationNs: 1_000_000, tokensPerSecond: 0 }
        render(performance([measured]))
        expect(cells('Engine-reported decode')).toEqual(['Engine-reported decode', '0', '1 / 8'])
    })

    it('shows decode as unknown when neither aggregate is available', () => {
        const measured = entry()
        measured.decode = null
        render(performance([measured]))
        expect(cells('Decode')).toEqual(['Decode', '--', '0 / 8'])
        expect(text()).not.toContain('Engine-reported decode')
    })

    it('replaces a destination native rate with measured zero and then unknown without accumulating coverage', () => {
        const measured = entry()
        measured.key = {
            host: 'destination-b',
            engine: 'strata-custom',
            model: 'resolved-child/model',
            fingerprint: 'strata-runtime-config-profile'
        }
        measured.profile = { engineFamily: 'strata', quantization: 'Q6_K', contextTokens: 16384 }
        measured.engineDecode = {
            samples: 2,
            durationNs: 1_000_500_000,
            tokensPerSecond: 49.126
        }
        measured.decode = null

        render(performance([measured]))
        render(performance([measured]))
        expect(cells('Engine-reported decode')).toEqual([
            'Engine-reported decode',
            '49.13',
            '2 / 8'
        ])
        expect(text().match(/Samples: 8/g)).toHaveLength(1)
        expect(text()).toContain('Engine: strata-custom · Model: resolved-child/model')
        expect(text()).toContain(
            'Source host: destination-b · Profile: strata-runtime-config-profile'
        )
        expect(text()).toContain('Family: strata · Quantization: Q6_K · Context tokens: 16384')

        measured.engineDecode = { samples: 1, durationNs: 400_500_000, tokensPerSecond: 0 }
        render(performance([measured]))
        expect(cells('Engine-reported decode')).toEqual(['Engine-reported decode', '0', '1 / 8'])
        expect(text()).not.toContain('49.13')

        measured.engineDecode = null
        render(performance([measured]))
        expect(cells('Decode')).toEqual(['Decode', '--', '0 / 8'])
        expect(text()).not.toContain('Engine-reported decode')
        expect(text()).toContain(
            'Source host: destination-b · Profile: strata-runtime-config-profile'
        )
        expect(text().match(/Samples: 8/g)).toHaveLength(1)
    })

    const statuses: { status: NodePerformanceData['status']; label: string }[] = [
        { status: 'fresh', label: 'Fresh' },
        { status: 'stale', label: 'Stale' },
        { status: 'unknown', label: 'Unknown' },
        { status: 'missing', label: 'Unknown' }
    ]

    it.each(statuses)(
        'labels $status summary freshness independently of measurements',
        ({ status, label }) => {
            render({
                ...performance(),
                status,
                summary:
                    status === 'fresh' || status === 'stale' ? performance().summary : undefined
            })
            expect(text()).toContain(`Measured runs · ${label}`)
            if (status === 'fresh' || status === 'stale') {
                expect(cells('Total execution')).toEqual(['Total execution', '100', '150', '5 / 8'])
            } else {
                expect(tableCount()).toBe(0)
                expect(text()).toContain('No measured runs reported for this node.')
            }
        }
    )

    it('shows source identity, profile metadata, sample ages and separate outcome counts', () => {
        const measured = entry()
        measured.clockUncertain = true
        render(performance([measured]))
        expect(text()).toContain('Engine: llamacpp-custom · Model: model-one')
        expect(text()).toContain('Source host: source-one · Profile: fp-one')
        expect(text()).toContain('Family: llamacpp · Quantization: Q4_K_M · Context tokens: 8192')
        expect(text()).toContain('Last seen age: 100 ms')
        expect(text()).toContain(
            'Newest sample age: 1234 ms · Oldest sample age: 4567 ms · Clock uncertain'
        )
        expect(text()).toContain(
            'Samples: 8 · Completed: 6 · Failed: 1 · Cancelled: 1 · Protocol incomplete: 3'
        )
    })

    it('renders distinct profiles and wraps long model and fingerprint identities', () => {
        const long = entry()
        long.key = {
            ...long.key,
            model: 'model/'.repeat(40),
            fingerprint: 'fingerprint-'.repeat(10)
        }
        render(performance([entry(), long]))
        expect(tableCount()).toBe(4)
        const leaves = Array.from(markup.matchAll(/<([a-z][a-z0-9]*)\b([^>]*)>([^<]*)<\/\1>/g))
        const model = leaves.find(element => element[3].includes(long.key.model))
        const profile = leaves.find(element => element[3].includes(long.key.fingerprint))
        expect(model?.[3]).toContain(long.key.model)
        expect(profile?.[3]).toContain(long.key.fingerprint)
        expect(model?.[2]).toContain('[overflow-wrap:anywhere]')
        expect(profile?.[2]).toContain('[overflow-wrap:anywhere]')
    })

    it('shows omissions, expiration, local evictions and persistence/publication warnings', () => {
        render({
            ...performance([]),
            evictions: { keys: 4, samples: 5, expiredSamples: 6 },
            persistenceDegraded: true,
            publicationBlocked: true
        })
        expect(text()).toContain('No measured runs reported for this node.')
        expect(text()).toContain('Omitted profiles: 3 · Expired profiles: 2')
        expect(text()).toContain('Local evictions — profiles: 4 · Samples: 5 · Expired samples: 6')
        expect(text()).toContain(
            'Performance persistence degraded; cached measurements may not survive restart.'
        )
        expect(text()).toContain('Performance publication blocked.')
    })

    it('renders replacement snapshots using only the supplied host and totals', () => {
        render(performance())
        render(performance())
        expect(tableCount()).toBe(2)
        expect(text().match(/Samples: 8/g)).toHaveLength(1)
        const replacement = entry()
        replacement.key = {
            host: 'source-two',
            engine: 'ollama',
            model: 'model-two',
            fingerprint: 'fp-two'
        }
        replacement.samples = 16
        replacement.completed = 14
        replacement.profile.engineFamily = 'ollama'
        render(performance([replacement]))
        expect(text()).toContain('Source host: source-two · Profile: fp-two')
        expect(text()).toContain('Samples: 16')
        expect(text()).not.toContain('source-one')
        expect(text()).not.toContain('model-one')
        expect(text()).not.toContain('Samples: 24')
        render()
        expect(tableCount()).toBe(0)
        expect(text()).not.toContain('source-two')
    })

    it('keeps detailed hardware null, stale and measured-zero readings independent of run data', () => {
        useMetricsStore.setState({
            nodeMetrics: new Map([
                [
                    node.id,
                    {
                        cpuUtilization: [{ timestamp: 1000, value: 0 }],
                        memoryUsage: [],
                        gpuUtilization: [],
                        gpuVramUsage: [{ id: 'gpu-one', data: [{ timestamp: 0, value: 25 }] }]
                    }
                ]
            ])
        })
        vi.spyOn(Date, 'now').mockReturnValue(11001)
        render(performance())
        expect(text()).toContain('Test GPU')
        expect(text()).toContain('Usage--')
        expect(text()).toContain('VRAMStale / 1 KB')
        expect(text()).toContain('Test CPUStale (4 cores)')
        expect(text()).toContain('Memory-- / 4 KB')
        vi.spyOn(Date, 'now').mockReturnValue(1000)
        render(performance())
        expect(text()).toContain('Test CPU0% (4 cores)')
        expect(cells('Total execution')).toEqual(['Total execution', '100', '150', '5 / 8'])
    })
})
