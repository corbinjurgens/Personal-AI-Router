// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
    NodePerformance,
    PerformanceDistribution,
    PerformanceEntry,
    PerformanceRate,
    PerformanceReportedRate,
    PerformanceSummary
} from '@/shared/types/performance'
import type { JsonObject, JsonValue } from './json-rpc-client'

const retentionMs = 604_800_000
const latencyMs = 86_400_000
const phaseNs = 86_400_000_000_000

function object(value: JsonValue | undefined, fields: string[]): JsonObject {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('object')
    if (Object.keys(value).some(key => !fields.includes(key))) throw new Error('field')
    return value
}

function count(value: JsonValue | undefined, max = Number.MAX_SAFE_INTEGER, min = 0): number {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min || value > max) {
        throw new Error('count')
    }
    return value
}

function bool(value: JsonValue | undefined): boolean {
    if (typeof value !== 'boolean') throw new Error('boolean')
    return value
}

function text(value: JsonValue | undefined, max: number, required = true): string {
    if (
        typeof value !== 'string' ||
        (required && !value) ||
        Buffer.byteLength(value) > max ||
        /[\p{Cc}\p{Cs}]/u.test(value)
    ) {
        throw new Error('text')
    }
    return value
}

function id(value: JsonValue | undefined): string {
    const result = text(value, 128)
    if (!/^[A-Za-z0-9._:-]+$/.test(result)) throw new Error('id')
    return result
}

function distribution(
    value: JsonValue | undefined,
    completed: number,
    max: number
): PerformanceDistribution | null {
    if (value == null) return null
    const obj = object(value, ['samples', 'median', 'p95'])
    const median = count(obj.median, max)
    return { samples: count(obj.samples, completed, 1), median, p95: count(obj.p95, max, median) }
}

function rate(value: JsonValue | undefined, completed: number): PerformanceRate | null {
    if (value == null) return null
    const obj = object(value, ['samples', 'tokens', 'durationNs', 'tokensPerSecond'])
    const samples = count(obj.samples, completed, 1)
    const speed = obj.tokensPerSecond
    if (typeof speed !== 'number' || !Number.isFinite(speed) || speed < 0 || speed > 10_000_000)
        throw new Error('rate')
    return {
        samples,
        tokens: count(obj.tokens, samples * 100_000_000),
        durationNs: count(obj.durationNs, samples * phaseNs, 1),
        tokensPerSecond: speed
    }
}

function reportedRate(
    value: JsonValue | undefined,
    completed: number
): PerformanceReportedRate | null {
    if (value == null) return null
    const obj = object(value, ['samples', 'durationNs', 'tokensPerSecond'])
    const samples = count(obj.samples, completed, 1)
    const speed = obj.tokensPerSecond
    if (typeof speed !== 'number' || !Number.isFinite(speed) || speed < 0 || speed > 10_000_000)
        throw new Error('reported rate')
    return {
        samples,
        durationNs: count(obj.durationNs, samples * phaseNs, 1),
        tokensPerSecond: speed
    }
}

function entry(value: JsonValue, host: string, maxSamples: number): PerformanceEntry {
    const obj = object(value, [
        'key',
        'profile',
        'samples',
        'completed',
        'incomplete',
        'outputTokens',
        'failed',
        'cancelled',
        'newestAgeMs',
        'oldestAgeMs',
        'newestObservedAtMs',
        'clockUncertain',
        'admissionWaitMs',
        'headerMs',
        'firstBodyMs',
        'firstTokenMs',
        'totalMs',
        'loadNs',
        'prefill',
        'decode',
        'engineDecode'
    ])
    const key = object(obj.key, ['host', 'engine', 'model', 'fingerprint'])
    if (id(key.host) !== host) throw new Error('host')
    const profile = object(obj.profile, ['engineFamily', 'quantization', 'contextTokens'])
    const samples = count(obj.samples, maxSamples, 1)
    const completed = count(obj.completed, samples)
    const incomplete = count(obj.incomplete, completed)
    const failed = count(obj.failed, samples)
    const cancelled = count(obj.cancelled, samples)
    if (completed + failed + cancelled !== samples) throw new Error('outcomes')
    if (obj.outputTokens != null && completed - incomplete < 1) throw new Error('output tokens')
    const newestAgeMs = count(obj.newestAgeMs)
    return {
        key: {
            host,
            engine: id(key.engine),
            model: text(key.model, 256),
            fingerprint: id(key.fingerprint)
        },
        profile: {
            engineFamily:
                profile.engineFamily === undefined || profile.engineFamily === ''
                    ? ''
                    : id(profile.engineFamily),
            quantization: text(profile.quantization ?? '', 64, false),
            contextTokens:
                profile.contextTokens == null ? null : count(profile.contextTokens, 100_000_000, 1)
        },
        samples,
        completed,
        failed,
        cancelled,
        incomplete,
        outputTokens:
            obj.outputTokens == null
                ? null
                : count(obj.outputTokens, (completed - incomplete) * 100_000_000),
        newestAgeMs,
        // Broker-aged display summaries can have an oldest sample beyond retention.
        oldestAgeMs: count(obj.oldestAgeMs, Number.MAX_SAFE_INTEGER, newestAgeMs),
        newestObservedAtMs: count(obj.newestObservedAtMs, Number.MAX_SAFE_INTEGER, 1),
        clockUncertain: obj.clockUncertain === undefined ? false : bool(obj.clockUncertain),
        admissionWaitMs: distribution(obj.admissionWaitMs, completed, latencyMs),
        headerMs: distribution(obj.headerMs, completed, latencyMs),
        firstBodyMs: distribution(obj.firstBodyMs, completed, latencyMs),
        firstTokenMs: distribution(obj.firstTokenMs, completed, latencyMs),
        totalMs: distribution(obj.totalMs, completed, latencyMs),
        loadNs: distribution(obj.loadNs, completed, phaseNs),
        prefill: rate(obj.prefill, completed),
        decode: rate(obj.decode, completed),
        engineDecode: reportedRate(obj.engineDecode, completed - incomplete)
    }
}

function summary(value: JsonValue, host: string, local: boolean): PerformanceSummary {
    if (Buffer.byteLength(JSON.stringify(value)) > 128 * 1024) throw new Error('summary size')
    const obj = object(value, [
        'version',
        'measuringHost',
        'admissionIncarnation',
        'revision',
        'window',
        'entries',
        'omittedEntries'
    ])
    if (obj.version !== 2 || id(obj.measuringHost) !== host) throw new Error('summary host/version')
    const epoch = local && obj.admissionIncarnation === '' ? '' : id(obj.admissionIncarnation)
    const window = object(obj.window, ['maxAgeMs', 'maxSamplesPerKey'])
    const maxAgeMs = count(window.maxAgeMs, retentionMs, 1)
    const maxSamplesPerKey = count(window.maxSamplesPerKey, 64, 1)
    if (!Array.isArray(obj.entries) || obj.entries.length > 128) throw new Error('entries')
    const entries = obj.entries.map(value => entry(value, host, maxSamplesPerKey))
    if (entries.some(value => value.newestAgeMs > maxAgeMs)) throw new Error('expired samples')
    const keys = new Set(entries.map(value => JSON.stringify(value.key)))
    if (keys.size !== entries.length) throw new Error('duplicate key')
    return {
        version: 2,
        measuringHost: host,
        admissionIncarnation: epoch,
        revision: count(obj.revision, Number.MAX_SAFE_INTEGER, local ? 0 : 1),
        window: { maxAgeMs, maxSamplesPerKey },
        entries,
        omittedEntries: count(obj.omittedEntries ?? 0, 128 - entries.length)
    }
}

function node(value: JsonValue, local: boolean): NodePerformance {
    const obj = object(value, [
        'measuringHost',
        'status',
        'lastSeenAgeMs',
        'summary',
        'expiredEntries'
    ])
    const measuringHost = id(obj.measuringHost)
    const status = obj.status
    if (status !== 'fresh' && status !== 'stale' && status !== 'unknown' && status !== 'missing')
        throw new Error('status')
    const parsedSummary =
        obj.summary == null ? undefined : summary(obj.summary, measuringHost, local)
    if (!parsedSummary && (status === 'fresh' || status === 'stale'))
        throw new Error('missing summary')
    if (
        parsedSummary &&
        (status === 'unknown' || (status === 'missing' && parsedSummary.entries.length))
    )
        throw new Error('summary status')
    return {
        measuringHost,
        status,
        lastSeenAgeMs: count(obj.lastSeenAgeMs),
        expiredEntries: count(obj.expiredEntries, 128),
        ...(parsedSummary ? { summary: parsedSummary } : {})
    }
}

/** All-or-nothing replacement; invalid performance never invalidates node identity. */
export function parsePerformanceProjection(
    value: JsonValue | undefined
): Map<string, NodePerformance> {
    try {
        if (value === undefined || Buffer.byteLength(JSON.stringify(value)) > 17 * 1024 * 1024)
            throw new Error('projection size')
        const obj = object(value, ['local', 'peers', 'persistence', 'expiredPeers'])
        count(obj.expiredPeers)
        if (!Array.isArray(obj.peers) || obj.peers.length > 128 || !obj.local)
            throw new Error('nodes')
        const local = node(obj.local, true)
        const persistence = object(obj.persistence, [
            'degraded',
            'publicationBlocked',
            'error',
            'stats'
        ])
        const stats = object(persistence.stats, [
            'Keys',
            'Samples',
            'Dirty',
            'PublishedRevision',
            'HistoryReset',
            'EvictedKeys',
            'EvictedSamples',
            'ExpiredSamples',
            'Duplicates'
        ])
        count(stats.Keys, 128)
        count(stats.Samples, 128 * 64)
        bool(stats.Dirty)
        count(stats.PublishedRevision)
        bool(stats.HistoryReset)
        count(stats.Duplicates)
        if (persistence.error !== undefined) text(persistence.error, 4096, false)
        local.evictions = {
            keys: count(stats.EvictedKeys),
            samples: count(stats.EvictedSamples),
            expiredSamples: count(stats.ExpiredSamples)
        }
        local.persistenceDegraded = bool(persistence.degraded)
        local.publicationBlocked = bool(persistence.publicationBlocked)
        const result = new Map([[local.measuringHost, local]])
        for (const value of obj.peers) {
            const peer = node(value, false)
            if (result.has(peer.measuringHost)) throw new Error('duplicate host')
            result.set(peer.measuringHost, peer)
        }
        return result
    } catch {
        return new Map()
    }
}

/** One broker attachment owns one subscription and its bootstrap race fence. */
export class PerformanceSubscription {
    private generation = 0
    private sequence = 0
    private attached = false

    constructor(
        private readonly call: (method: string) => Promise<JsonValue | undefined>,
        private readonly apply: (value: JsonValue | undefined) => void
    ) {}

    ready(): void {
        if (this.attached) return
        this.attached = true
        const generation = this.generation
        const sequence = this.sequence
        void this.call('performance:subscribe')
            .then(() => {
                if (this.attached && generation === this.generation)
                    return this.call('performance:get')
                return undefined
            })
            .then(value => {
                if (this.attached && generation === this.generation && sequence === this.sequence)
                    this.apply(value)
            })
            .catch(() => {})
    }

    changed(value: JsonValue | undefined): void {
        if (!this.attached) return
        this.sequence += 1
        this.apply(value)
    }

    reset(): void {
        this.attached = false
        this.generation += 1
        this.apply(undefined)
    }

    async stop(): Promise<void> {
        const attached = this.attached
        this.reset()
        if (attached) await this.call('performance:unsubscribe').catch(() => {})
    }
}
