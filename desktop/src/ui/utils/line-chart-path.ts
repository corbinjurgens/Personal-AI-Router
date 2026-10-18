// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PerformanceMetric } from '@/ui/types/types'

export interface MetricDataset {
    data: PerformanceMetric[]
    label: string
    color: string
    key: string
}

const PADDING = 4

// Permit jitter at the slower two-second poll cadence, but never connect
// observations separated by more than 2.5 poll intervals, even with sparse history.
const MAX_SAMPLE_GAP_MS = 5_000

function buildSmoothPath(points: { x: number; y: number }[], tension = 0.3): string {
    if (points.length === 0) return ''
    if (points.length === 1) return `M ${points[0].x} ${points[0].y}`

    const d: string[] = [`M ${points[0].x} ${points[0].y}`]

    for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[Math.max(0, i - 1)]
        const p1 = points[i]
        const p2 = points[i + 1]
        const p3 = points[Math.min(points.length - 1, i + 2)]

        const cp1x = p1.x + ((p2.x - p0.x) * tension) / 3
        const cp1y = p1.y + ((p2.y - p0.y) * tension) / 3
        const cp2x = p2.x - ((p3.x - p1.x) * tension) / 3
        const cp2y = p2.y - ((p3.y - p1.y) * tension) / 3

        d.push(`C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`)
    }

    return d.join(' ')
}

export function buildLineChartPaths(
    datasets: MetricDataset[],
    maxDataPoints: number,
    yAxisMax: number,
    chartW: number,
    chartH: number
) {
    if (chartW <= 0 || chartH <= 0 || yAxisMax <= 0) return []
    const recentDatasets = datasets.map(dataset => ({
        ...dataset,
        data: dataset.data
            .slice(-maxDataPoints)
            .filter(sample => Number.isFinite(sample.timestamp) && Number.isFinite(sample.value))
    }))
    // A shared time domain keeps independently sampled CPU/GPU series aligned.
    const timestamps = recentDatasets.flatMap(dataset =>
        dataset.data.map(sample => sample.timestamp)
    )
    const start = Math.min(...timestamps)
    const end = Math.max(...timestamps)

    return recentDatasets.map(dataset => {
        const segments: { x: number; y: number }[][] = []
        dataset.data.forEach((sample, index) => {
            const previous = dataset.data[index - 1]
            if (!previous || sample.timestamp - previous.timestamp > MAX_SAMPLE_GAP_MS) {
                segments.push([])
            }
            segments[segments.length - 1].push({
                x:
                    PADDING +
                    (end > start ? ((sample.timestamp - start) / (end - start)) * chartW : 0),
                y: PADDING + chartH - (Math.min(sample.value, yAxisMax) / yAxisMax) * chartH
            })
        })
        const lines = segments.map(segment => buildSmoothPath(segment))
        // Close each area separately so its fill cannot bridge an observation gap.
        const fills = segments.map((segment, index) => {
            const first = segment[0]
            const last = segment[segment.length - 1]
            return `${lines[index]} L ${last.x} ${PADDING + chartH} L ${first.x} ${PADDING + chartH} Z`
        })
        return {
            key: dataset.key,
            color: dataset.color,
            line: lines.join(' '),
            fill: fills.join(' ')
        }
    })
}
