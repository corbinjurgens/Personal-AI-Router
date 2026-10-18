// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import type { NodeMetricsHistory } from '@/ui/stores/metrics.store'
import { buildRadialChartMetrics } from '@/ui/utils/build-radial-chart-metrics'
import { gpuArrayValue, parseProxyNode, toMetrics } from '@/electron/service-bridge/node-projection'

const topology: NodeItem['topology'] = {
    cpu: { model: 'CPU', cores: 4, threads: 4 },
    gpus: [{ id: 'gpu', name: 'GPU', vramTotal: 1024 }],
    ram: 4096,
    storage: []
}

function reading(): NodeMetricsHistory {
    return {
        cpuUtilization: [],
        memoryUsage: [],
        gpuUtilization: [{ id: 'gpu', data: [{ timestamp: 1000, value: 40 }] }],
        gpuVramUsage: [{ id: 'gpu', data: [{ timestamp: 1000, value: 50 }] }]
    }
}

beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(1000)
})

afterEach(() => vi.useRealTimers())

it('hides last-good rings after the observation ages without inserting samples', () => {
    const metrics = reading()
    vi.setSystemTime(11000)
    expect(buildRadialChartMetrics(topology, metrics).map(metric => metric.shown)).toEqual([
        true,
        true
    ])
    vi.setSystemTime(11001)
    expect(buildRadialChartMetrics(topology, metrics).map(metric => metric.shown)).toEqual([
        false,
        false
    ])
    expect(metrics.gpuUtilization[0].data).toEqual([{ timestamp: 1000, value: 40 }])
    expect(vi.getTimerCount()).toBe(0)
})

it('shows no numeric rings when GPU observations are unknown', () => {
    expect(buildRadialChartMetrics(topology, undefined).every(metric => !metric.shown)).toBe(true)
    const metrics = reading()
    metrics.gpuUtilization = []
    metrics.gpuVramUsage = []
    expect(buildRadialChartMetrics(topology, metrics).every(metric => !metric.shown)).toBe(true)
})

it('joins GPU observations by identity after another GPU disappears', () => {
    const metrics = reading()
    metrics.gpuUtilization.unshift({ id: 'removed', data: [{ timestamp: 1000, value: 99 }] })
    expect(buildRadialChartMetrics(topology, metrics)[0]).toMatchObject({ value: 40, shown: true })
})

it('keeps a real zero visible', () => {
    const metrics = reading()
    metrics.gpuUtilization[0].data[0].value = 0
    expect(buildRadialChartMetrics(topology, metrics)[0]).toMatchObject({ value: 0, shown: true })
})

it('preserves missing node-info dynamic fields as unknown through projection', () => {
    const node = parseProxyNode({ id: 'node', name: 'Node' }, 'ollama')
    if (!node) throw new Error('fixture node was not parsed')
    node.gpus = gpuArrayValue([{ name: 'GPU', vram_bytes: 1024 }])
    const metrics = toMetrics(node).current
    expect(metrics.cpuUtilization).toBeUndefined()
    expect(metrics.memoryUsage).toBeUndefined()
    expect(metrics.gpuUtilization[0].value).toBeUndefined()
    expect(metrics.gpuVramUsage[0].value).toBeUndefined()
})
