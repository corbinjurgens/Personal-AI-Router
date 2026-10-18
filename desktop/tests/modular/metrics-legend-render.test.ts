// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import NodeChartLegend from '@/ui/components/NodeList/NodeChartLegend'
import { getMetricReading } from '@/ui/utils/build-radial-chart-metrics'

let container: HTMLDivElement
let root: Root

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.unstubAllGlobals()
})

it.each([
    { name: 'unknown', samples: [], now: 1000, label: '--', muted: true },
    {
        name: 'stale',
        samples: [{ timestamp: 1000, value: 45 }],
        now: 11001,
        label: 'Stale',
        muted: true
    },
    {
        name: 'real zero',
        samples: [{ timestamp: 1000, value: 0 }],
        now: 1000,
        label: '0%',
        muted: false
    }
])(
    'renders $name GPU utilization with the correct label and emphasis',
    async ({ samples, now, label, muted }) => {
        const usage = getMetricReading(samples, now)
        await act(async () =>
            root.render(
                createElement(NodeChartLegend, {
                    gpuInfo: [
                        {
                            id: 'gpu',
                            name: 'GPU',
                            vramTotal: 1024,
                            vramTotalFormatted: '1 KB',
                            vramUsageFormatted: usage.value === undefined ? usage.label : '0 B',
                            usage,
                            vramUsage: usage,
                            usageColor: '#fff',
                            vramColor: '#fff'
                        }
                    ]
                })
            )
        )
        const rendered = Array.from(container.querySelectorAll('*')).find(
            element => element.textContent === label && element.children.length === 0
        )
        expect(rendered?.textContent).toBe(label)
        expect(rendered?.classList.contains('text-subtle-color')).toBe(muted)
        expect(container.textContent).toContain(
            `${usage.value === undefined ? usage.label : '0 B'} / 1 KB`
        )
        expect(container.textContent).not.toContain('NaN')
    }
)

it('renders unknown CPU and stale RAM labels while keeping static capacity', async () => {
    await act(async () =>
        root.render(
            createElement(NodeChartLegend, {
                gpuInfo: [],
                cpuFallbackInfo: {
                    model: 'CPU',
                    usage: getMetricReading(undefined, 11001),
                    usageColor: '#fff',
                    memoryUsage: getMetricReading([{ timestamp: 1000, value: 50 }], 11001),
                    memoryUsageFormatted: 'Stale',
                    memoryTotalFormatted: '4 GB',
                    memoryColor: '#fff'
                }
            })
        )
    )
    expect(container.textContent).toContain('CPU--')
    expect(container.textContent).toContain('RAMStale / 4 GB')
    expect(container.textContent).not.toContain('--%')
    expect(container.textContent).not.toContain('Stale%')
})
