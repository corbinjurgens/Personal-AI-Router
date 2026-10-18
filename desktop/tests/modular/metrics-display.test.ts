// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import type { NodeMetricsHistory } from '@/ui/stores/metrics.store'
import { getMetricReading } from '@/ui/utils/build-radial-chart-metrics'
import { buildPerformanceHardwareInfo } from '@/ui/utils/performance-hardware-info'

describe('metric display freshness', () => {
    it('renders missing history as unknown rather than zero', () => {
        expect(getMetricReading(undefined, 1000)).toEqual({ status: 'unknown', label: '--' })
        expect(getMetricReading([], 1000)).toEqual({ status: 'unknown', label: '--' })
    })

    it('labels a reading stale only after the shared ten-second boundary', () => {
        const samples = [{ timestamp: 1000, value: 45 }]
        expect(getMetricReading(samples, 11000)).toEqual({
            status: 'current',
            value: 45,
            label: '45%'
        })
        expect(getMetricReading(samples, 11001)).toEqual({ status: 'stale', label: 'Stale' })
        expect(samples).toEqual([{ timestamp: 1000, value: 45 }])
    })

    it('preserves a real zero as a current zero percent reading', () => {
        expect(getMetricReading([{ timestamp: 1000, value: 0 }], 1000)).toEqual({
            status: 'current',
            value: 0,
            label: '0%'
        })
    })

    it('treats invalid and future observations as unknown', () => {
        expect(getMetricReading([{ timestamp: 1000, value: NaN }], 1000).label).toBe('--')
        expect(getMetricReading([{ timestamp: NaN, value: 40 }], 1000).label).toBe('--')
        expect(getMetricReading([{ timestamp: 2000, value: 40 }], 1000).label).toBe('--')
    })
})

const topology: NodeItem['topology'] = {
    cpu: { model: 'CPU', cores: 4, threads: 4 },
    gpus: [
        { id: 'first', name: 'First GPU', vramTotal: 1024 },
        { id: 'second', name: 'Second GPU', vramTotal: 2048 }
    ],
    ram: 4096,
    storage: []
}

it('joins utilization and VRAM independently by GPU id despite reordered or absent series', () => {
    const metrics: NodeMetricsHistory = {
        cpuUtilization: [],
        memoryUsage: [],
        gpuUtilization: [
            { id: 'second', data: [{ timestamp: 1000, value: 80 }] },
            { id: 'first', data: [{ timestamp: 1000, value: 0 }] }
        ],
        gpuVramUsage: [{ id: 'second', data: [{ timestamp: 1000, value: 50 }] }]
    }
    const info = buildPerformanceHardwareInfo(topology, metrics, 1000)
    expect(
        info.gpus.map(gpu => ({
            id: gpu.id,
            usage: gpu.utilization.label,
            vram: gpu.vramUsage.label
        }))
    ).toEqual([
        { id: 'first', usage: '0%', vram: '--' },
        { id: 'second', usage: '80%', vram: '50%' }
    ])
    expect(info.cpuUtilization.label).toBe('--')
    expect(info.memoryUsage.label).toBe('--')
})

it('ages all detailed labels using the supplied clock while preserving static hardware', () => {
    const metrics: NodeMetricsHistory = {
        cpuUtilization: [{ timestamp: 1000, value: 0 }],
        memoryUsage: [{ timestamp: 1000, value: 30 }],
        gpuUtilization: [{ id: 'first', data: [{ timestamp: 1000, value: 45 }] }],
        gpuVramUsage: [{ id: 'first', data: [{ timestamp: 1000, value: 50 }] }]
    }
    const info = buildPerformanceHardwareInfo(topology, metrics, 11001)
    expect([
        info.cpuUtilization.label,
        info.memoryUsage.label,
        info.gpus[0].utilization.label,
        info.gpus[0].vramUsage.label
    ]).toEqual(['Stale', 'Stale', 'Stale', 'Stale'])
    expect(info.gpus[0].vramTotal).toBe(1024)
    expect(info.cpuModel).toBe('CPU')
})
