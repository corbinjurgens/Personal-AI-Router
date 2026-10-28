// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { JsonObject } from '@/electron/service-bridge/json-rpc-client'
import type { NodeItemMetrics } from '@/shared/types/metrics'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { subscribePush } from '@/electron/service-bridge/push-bus'
import { useMetricsStore } from '@/ui/stores/metrics.store'
import { getMetricReading } from '@/ui/utils/build-radial-chart-metrics'

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))

const state = getModularBridgeState()
const id = 'observation-node'
let events: string[]
let metrics: NodeItemMetrics | undefined
let unsubscribe = () => {}

function response(changes: JsonObject = {}): JsonObject {
    return {
        GPUs: [{ name: 'GPU', vram_bytes: 1024, vram_used_bytes: 512, utilization_percent: 40 }],
        cpu: { name: 'CPU', cores: 4, utilization_percent: 20 },
        memory: { total_bytes: 4096, used_bytes: 1024 },
        memoryObservation: {
            runId: 'fixture',
            domains: [{ id: 'system', kind: 'shared', deviceIds: ['ioreg:a'],
                total: { bytes: '4096', source: 'fixture', sampledAt: '2026-10-28T00:00:00Z', ageMs: 0, valid: true },
                free: { bytes: '0', source: 'fixture', sampledAt: '2026-10-28T00:00:00Z', ageMs: 0, valid: true }
            }]
        },
        telemetryValid: true,
        msSince: 0,
        ...changes
    }
}

function current() {
    if (!metrics) throw new Error('no metrics push received')
    return metrics.current
}

function history() {
    const value = useMetricsStore.getState().nodeMetrics.get(id)
    if (!value) throw new Error('no metrics history received')
    return value
}

function discover(name = 'Host'): void {
    state.handleNotification({
        source: 'broker',
        method: 'discovery:nodes-changed',
        params: { nodes: [{ hostUuid: id, name, ipAddress: '192.0.2.1', port: 14318 }] }
    })
}

beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(20000)
    vi.stubGlobal('window', {
        pairApi: {
            nodes: { onRemove: () => () => {} },
            metrics: {
                onUpdate: (callback: (value: NodeItemMetrics) => void) =>
                    subscribePush(event => {
                        if (event.channel === 'metrics:update') callback(event.payload)
                    })
            }
        }
    })
    useMetricsStore.getState().clearAll()
    useMetricsStore.getState().initialize()
    state.handleNotification({
        source: 'broker',
        method: 'discovery:nodes-changed',
        params: { nodes: [] }
    })
    discover()
    events = []
    metrics = undefined
    unsubscribe = subscribePush(event => {
        events.push(event.channel)
        if (event.channel === 'metrics:update') metrics = event.payload
    })
})

afterEach(() => {
    unsubscribe()
    useMetricsStore.getState().cleanup()
    vi.unstubAllGlobals()
    vi.useRealTimers()
})

it('carries a fresh GPU observation through the hardware upsert', () => {
    state.mergeNodeInfoResponse(id, response())
    expect(events).toContain('nodes:upsert')
    expect(current().gpuUtilization).toEqual([{ id: `${id}:gpu:0`, timestamp: 20000, value: 40 }])
    expect(getMetricReading(history().gpuUtilization[0].data)).toMatchObject({
        status: 'current',
        value: 40
    })
})

it('deduplicates a retained sample on the metrics-only path while CPU and RAM stay receipt-timed', () => {
    state.mergeNodeInfoResponse(id, response())
    events = []
    vi.setSystemTime(30001)
    state.mergeNodeInfoResponse(id, response({ msSince: 10001 }))
    expect(events).toEqual(['metrics:update'])
    expect(current().gpuUtilization[0].timestamp).toBe(20000)
    expect(history().gpuUtilization[0].data).toEqual([{ timestamp: 20000, value: 40 }])
    expect(history().gpuVramUsage[0].data).toEqual([{ timestamp: 20000, value: 50 }])
    expect(getMetricReading(history().gpuUtilization[0].data)).toEqual({
        status: 'stale',
        label: 'Stale'
    })
    expect(history().cpuUtilization.map(sample => sample.timestamp)).toEqual([20000, 30001])
    expect(history().memoryUsage.map(sample => sample.timestamp)).toEqual([20000, 30001])
})

it('clears GPU readings immediately when telemetry becomes invalid', () => {
    state.mergeNodeInfoResponse(id, response())
    state.mergeNodeInfoResponse(id, response({ telemetryValid: false }))
    expect(current().gpuUtilization).toEqual([])
    expect(current().gpuVramUsage).toEqual([])
    expect(history().gpuUtilization).toEqual([])
    expect(getMetricReading(history().gpuUtilization[0]?.data)).toEqual({
        status: 'unknown',
        label: '--'
    })
    expect(current().cpuUtilization).toBe(20)
})

it.each([
    { name: 'missing', age: undefined },
    { name: 'negative', age: -1 },
    { name: 'infinite', age: Infinity },
    { name: 'not a number', age: NaN }
])('leaves $name GPU age unknown', ({ age }) => {
    const body = response()
    if (age === undefined) delete body.msSince
    else body.msSince = age
    state.mergeNodeInfoResponse(id, body)
    expect(current().gpuUtilization).toEqual([])
    expect(history().gpuUtilization).toEqual([])
})

it('clamps an age older than the epoch to zero and accepts that first observation', () => {
    state.mergeNodeInfoResponse(id, response({ msSince: Number.MAX_VALUE }))
    expect(current().gpuUtilization[0].timestamp).toBe(0)
    expect(history().gpuUtilization[0].data).toEqual([{ timestamp: 0, value: 40 }])
})

it('accepts the first real GPU observation after an invalid response', () => {
    state.mergeNodeInfoResponse(id, response({ telemetryValid: false }))
    state.mergeNodeInfoResponse(id, response({ msSince: 5000 }))
    expect(history().gpuUtilization[0].data).toEqual([{ timestamp: 15000, value: 40 }])
})

it('preserves observation age through broker and proxy discovery merges', () => {
    state.mergeNodeInfoResponse(id, response({ msSince: 5000 }))
    vi.setSystemTime(22000)
    state.handleNotification({
        source: 'ollama-proxy',
        method: 'node/discovered',
        params: { id, host: 'Host', port: 11434, ip: '192.0.2.1' }
    })
    expect(current().gpuUtilization[0].timestamp).toBe(15000)
    discover('Updated host')
    expect(current().gpuUtilization[0].timestamp).toBe(15000)
    expect(history().gpuUtilization[0].data).toEqual([{ timestamp: 15000, value: 40 }])
})
