// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import type { PerformanceMetric } from '@/ui/types/types'

export interface MetricDataset {
    data: PerformanceMetric[]
    label: string
    color: string
    key: string
}

interface SvgLineChartProps {
    datasets: MetricDataset[]
    maxDataPoints?: number
    yAxisMax?: number
    selectedKey?: string
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

function SvgLineChart({
    datasets,
    maxDataPoints = 30,
    yAxisMax = 100,
    selectedKey
}: SvgLineChartProps) {
    const uid = useId().replace(/:/g, '')
    const containerRef = useRef<HTMLDivElement>(null)
    const resizeRafRef = useRef(0)
    const [size, setSize] = useState({ w: 300, h: 200 })

    const measure = useCallback(() => {
        if (!containerRef.current) return
        const { width, height } = containerRef.current.getBoundingClientRect()
        if (width > 0 && height > 0) {
            setSize(prev =>
                prev.w === width && prev.h === height ? prev : { w: width, h: height }
            )
        }
    }, [])

    const scheduleResize = useCallback(() => {
        if (resizeRafRef.current) return
        resizeRafRef.current = requestAnimationFrame(() => {
            resizeRafRef.current = 0
            measure()
        })
    }, [measure])

    useEffect(() => {
        measure()
        const el = containerRef.current
        if (!el) return
        const ro = new ResizeObserver(scheduleResize)
        ro.observe(el)
        return () => {
            ro.disconnect()
            cancelAnimationFrame(resizeRafRef.current)
            resizeRafRef.current = 0
        }
    }, [measure, scheduleResize])

    const chartW = size.w - PADDING * 2
    const chartH = size.h - PADDING * 2

    const paths = useMemo(
        () => buildLineChartPaths(datasets, maxDataPoints, yAxisMax, chartW, chartH),
        [datasets, maxDataPoints, yAxisMax, chartW, chartH]
    )

    const clipId = `${uid}-clip`

    return (
        <div ref={containerRef} className="w-full h-full line-chart-container">
            <div className="line-chart-container-bg absolute inset-0 opacity-10" />
            <div className="app-bg absolute inset-0 opacity-15" />
            <div className="line-chart-container-shadow absolute inset-0 opacity-10" />
            <svg
                width={size.w}
                height={size.h}
                viewBox={`0 0 ${size.w} ${size.h}`}
                className="block relative"
            >
                <defs>
                    <clipPath id={clipId}>
                        <rect x={PADDING} y={0} width={chartW} height={size.h} />
                    </clipPath>
                    {paths.map(p => (
                        <linearGradient
                            key={`grad-${p.key}`}
                            id={`${uid}-grad-${p.key}`}
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                        >
                            <stop offset="0%" stopColor={p.color} stopOpacity={0.25} />
                            <stop offset="100%" stopColor={p.color} stopOpacity={0} />
                        </linearGradient>
                    ))}
                </defs>

                <g clipPath={`url(#${clipId})`}>
                    {paths.map(p => {
                        if (!p.line) return null
                        const visible = !selectedKey || selectedKey === p.key
                        return (
                            <g
                                key={p.key}
                                style={{
                                    opacity: visible ? 1 : 0.08,
                                    transition: 'opacity 0.2s ease'
                                }}
                            >
                                <path d={p.fill} fill={`url(#${uid}-grad-${p.key})`} />
                                <path
                                    d={p.line}
                                    fill="none"
                                    stroke={p.color}
                                    strokeWidth={2}
                                    strokeLinejoin="round"
                                    strokeLinecap="round"
                                />
                            </g>
                        )
                    })}
                </g>
            </svg>
            <div className="line-chart-container-shadow absolute inset-0 opacity-15" />
        </div>
    )
}

export default memo(SvgLineChart)
