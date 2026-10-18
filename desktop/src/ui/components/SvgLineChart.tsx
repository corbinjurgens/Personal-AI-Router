// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { buildLineChartPaths, type MetricDataset } from '@/ui/utils/line-chart-path'

interface SvgLineChartProps {
    datasets: MetricDataset[]
    maxDataPoints?: number
    yAxisMax?: number
    selectedKey?: string
}

const PADDING = 4

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
