// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NodeItem } from '@/shared/types/nodes'
import type { RadialMetric } from '@/ui/components/GpuRadialChartSvg'
import type { NodeMetricsHistory } from '@/ui/stores/metrics.store'
import { CHART_COLORS } from '@/ui/constants/colors'
import { getGpuColor, getVramColor } from '@/ui/utils/colors'
import { hasInferenceReadyGpu, isGpuInferenceReady } from '@/ui/utils/gpu-inference'

// Match the TUI's five normal poll intervals. No timer is created here: the
// existing render/poll cadence checks the actual observation timestamp.
const MAX_SAMPLE_AGE_MS = 10_000

export function getMetricReading(
    data: { timestamp: number; value: number }[] | undefined,
    now = Date.now()
): { status: 'current' | 'stale' | 'unknown'; value?: number; label: string } {
    const sample = data?.[data.length - 1]
    if (!sample || !Number.isFinite(sample.value)) return { status: 'unknown', label: '--' }
    const age = now - sample.timestamp
    if (!Number.isFinite(age) || age < 0) return { status: 'unknown', label: '--' }
    return age <= MAX_SAMPLE_AGE_MS
        ? { status: 'current', value: sample.value, label: `${Math.floor(sample.value)}%` }
        : { status: 'stale', label: 'Stale' }
}

function getLatest(data: { timestamp: number; value: number }[] | undefined): number | undefined {
    return getMetricReading(data).value
}

export function buildRadialChartMetrics(
    topology: NodeItem['topology'],
    nodeMetrics: NodeMetricsHistory | undefined,
    // Show only the GPUs the backend marks inference-ready (or all GPUs when it
    // reports no readiness list); see gpu-inference.ts. A node with no
    // inference-ready GPU falls through to the CPU + RAM rings below.
    hasGpu = hasInferenceReadyGpu(topology)
): RadialMetric[] {
    if (!hasGpu) {
        const cpu = getLatest(nodeMetrics?.cpuUtilization)
        const memory = getLatest(nodeMetrics?.memoryUsage)
        return [
            {
                value: cpu ?? 0,
                color: CHART_COLORS.CPU,
                shown: cpu !== undefined
            },
            {
                value: memory ?? 0,
                color: CHART_COLORS.MEMORY,
                shown: memory !== undefined
            }
        ]
    }

    // Keep the original GPU index for colors, joining observations by identity
    // after inventory changes. Omit the rings for any GPU the backend
    // did not mark inference-ready (when it sends a readiness list).
    return topology.gpus.flatMap((gpu, gpuIdx) => {
        if (!isGpuInferenceReady(gpu, topology.inferenceHardwareIds)) return []
        const gpuUtil = getLatest(
            nodeMetrics?.gpuUtilization.find(entry => entry.id === gpu.id)?.data
        )
        const gpuVram = getLatest(
            nodeMetrics?.gpuVramUsage.find(entry => entry.id === gpu.id)?.data
        )
        return [
            {
                value: gpuUtil ?? 0,
                color: getGpuColor(gpuIdx),
                shown: gpuUtil !== undefined
            },
            {
                value: gpuVram ?? 0,
                color: getVramColor(gpuIdx),
                shown: gpuVram !== undefined
            }
        ]
    })
}
