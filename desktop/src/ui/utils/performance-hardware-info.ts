// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NodeItem } from '@/shared/types/nodes'
import type { NodeMetricsHistory } from '@/ui/stores/metrics.store'
import { getGpuColor, getVramColor } from '@/ui/utils/colors'
import { formatBytes } from '@/ui/utils/formatters'
import { getMetricReading } from '@/ui/utils/build-radial-chart-metrics'

export function buildPerformanceHardwareInfo(
    topology: NodeItem['topology'],
    nodeMetrics: NodeMetricsHistory | undefined,
    now: number
) {
    return {
        // Detailed view includes all GPUs; the radial filters inference-ready GPUs.
        gpus: topology.gpus.map((gpu, index) => ({
            id: gpu.id,
            name: gpu.name,
            vramTotal: gpu.vramTotal,
            vramFormatted: formatBytes(gpu.vramTotal, 1),
            utilization: getMetricReading(
                nodeMetrics?.gpuUtilization.find(entry => entry.id === gpu.id)?.data,
                now
            ),
            vramUsage: getMetricReading(
                nodeMetrics?.gpuVramUsage.find(entry => entry.id === gpu.id)?.data,
                now
            ),
            color: getGpuColor(index),
            vramColor: getVramColor(index)
        })),
        cpuCores: topology.cpu.cores,
        cpuModel: topology.cpu.model,
        cpuUtilization: getMetricReading(nodeMetrics?.cpuUtilization, now),
        memory: formatBytes(topology.ram, 1),
        memoryTotal: topology.ram,
        memoryUsage: getMetricReading(nodeMetrics?.memoryUsage, now)
    }
}
