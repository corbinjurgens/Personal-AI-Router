// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface PerformanceDistribution {
    samples: number
    median: number
    p95: number
}

export interface PerformanceRate {
    samples: number
    tokens: number
    durationNs: number
    tokensPerSecond: number
}

export interface PerformanceReportedRate {
    samples: number
    durationNs: number
    tokensPerSecond: number
}

export interface PerformanceEntry {
    key: { host: string; engine: string; model: string; fingerprint: string }
    profile: { engineFamily: string; quantization: string; contextTokens: number | null }
    samples: number
    completed: number
    incomplete: number
    outputTokens: number | null
    failed: number
    cancelled: number
    newestAgeMs: number
    oldestAgeMs: number
    newestObservedAtMs: number
    clockUncertain: boolean
    admissionWaitMs: PerformanceDistribution | null
    headerMs: PerformanceDistribution | null
    firstBodyMs: PerformanceDistribution | null
    firstTokenMs: PerformanceDistribution | null
    totalMs: PerformanceDistribution | null
    loadNs: PerformanceDistribution | null
    prefill: PerformanceRate | null
    decode: PerformanceRate | null
    engineDecode: PerformanceReportedRate | null
}

export interface PerformanceSummary {
    version: 2
    measuringHost: string
    admissionIncarnation: string
    revision: number
    window: { maxAgeMs: number; maxSamplesPerKey: number }
    entries: PerformanceEntry[]
    omittedEntries: number
}

export interface NodePerformance {
    measuringHost: string
    status: 'fresh' | 'stale' | 'unknown' | 'missing'
    lastSeenAgeMs: number
    summary?: PerformanceSummary
    expiredEntries: number
    evictions?: { keys: number; samples: number; expiredSamples: number }
    persistenceDegraded?: boolean
    publicationBlocked?: boolean
}
