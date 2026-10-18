// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { NodeItemMetrics, NodeItemMetricsEntry } from '@/shared/types/metrics'
import { useMetricsStore } from '@/ui/stores/metrics.store'

let update: (metrics: NodeItemMetrics) => void
let remove: (id: string) => void

function publish(timestamp: number, changes: Partial<NodeItemMetricsEntry> = {}): void {
    update({
        id: 'node',
        current: {
            timestamp,
            cpuUtilization: 20,
            memoryUsage: 30,
            gpuUtilization: [{ id: 'gpu', value: 40 }],
            gpuVramUsage: [{ id: 'gpu', value: 50 }],
            ...changes
        },
        historical: []
    })
}

function history() {
    const result = useMetricsStore.getState().nodeMetrics.get('node')
    if (!result) throw new Error('missing node history')
    return result
}

beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('window', {
        pairApi: {
            nodes: {
                onRemove: (callback: typeof remove) => {
                    remove = callback
                    return () => {}
                }
            },
            metrics: {
                onUpdate: (callback: typeof update) => {
                    update = callback
                    return () => {}
                }
            }
        }
    })
    useMetricsStore.getState().clearAll()
    useMetricsStore.getState().initialize()
})

afterEach(() => {
    useMetricsStore.getState().cleanup()
    vi.unstubAllGlobals()
    vi.useRealTimers()
})

describe('observed metrics history', () => {
    it('starts with only the received sample', () => {
        publish(1000)
        expect(history().cpuUtilization).toEqual([{ timestamp: 1000, value: 20 }])
        expect(history().gpuUtilization[0].data).toEqual([{ timestamp: 1000, value: 40 }])
    })

    it('keeps missing readings unknown and accepts a real zero', () => {
        publish(1000, {
            cpuUtilization: undefined,
            memoryUsage: undefined,
            gpuUtilization: [{ id: 'gpu' }],
            gpuVramUsage: []
        })
        expect(history().cpuUtilization).toEqual([])
        expect(history().memoryUsage).toEqual([])
        expect(history().gpuUtilization[0].data).toEqual([])
        expect(history().gpuVramUsage).toEqual([])
        publish(2000, { cpuUtilization: 0 })
        expect(history().cpuUtilization).toEqual([{ timestamp: 2000, value: 0 }])
    })

    it('prunes disappeared GPUs even with an unchanged sample timestamp', () => {
        publish(1000)
        publish(1000, { gpuUtilization: [], gpuVramUsage: [] })
        expect(history().gpuUtilization).toEqual([])
        expect(history().gpuVramUsage).toEqual([])
    })

    it('does not fill a hidden-window gap on resume', () => {
        vi.setSystemTime(1000)
        publish(Date.now())
        vi.setSystemTime(121000)
        publish(Date.now())
        expect(history().cpuUtilization).toEqual([
            { timestamp: 1000, value: 20 },
            { timestamp: 121000, value: 20 }
        ])
        expect(history().gpuUtilization[0].data.map(sample => sample.timestamp)).toEqual([
            1000, 121000
        ])
        expect(vi.getTimerCount()).toBe(0)
    })

    it('bounds every series at the latest 60 actual observations', () => {
        for (let timestamp = 1; timestamp <= 75; timestamp++) publish(timestamp)
        for (const data of [
            history().cpuUtilization,
            history().memoryUsage,
            history().gpuUtilization[0].data,
            history().gpuVramUsage[0].data
        ]) {
            expect(data).toHaveLength(60)
            expect(data[0].timestamp).toBe(16)
            expect(data[59].timestamp).toBe(75)
        }
    })

    it('deduplicates each series independently of missing CPU observations', () => {
        publish(1000, { cpuUtilization: undefined })
        publish(1000)
        publish(999)
        expect(history().cpuUtilization).toEqual([{ timestamp: 1000, value: 20 }])
        expect(history().gpuUtilization[0].data).toEqual([{ timestamp: 1000, value: 40 }])
    })

    it('removes history with its node', () => {
        publish(1000)
        remove('node')
        expect(useMetricsStore.getState().nodeMetrics.has('node')).toBe(false)
    })
})
