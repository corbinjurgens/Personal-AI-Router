// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { getMetricReading } from '@/ui/utils/build-radial-chart-metrics'

export type GpuInfo = {
    id: string
    name: string
    // hliId: string;
    vramTotal: number
    vramTotalFormatted: string
    vramUsageFormatted: string
    vramUsage: ReturnType<typeof getMetricReading>
    usage: ReturnType<typeof getMetricReading>
    usageColor: string
    vramColor: string
}

export type CpuFallbackInfo = {
    model: string
    usage: ReturnType<typeof getMetricReading>
    usageColor: string
    memoryUsage: ReturnType<typeof getMetricReading>
    memoryUsageFormatted: string
    memoryTotalFormatted: string
    memoryColor: string
}
