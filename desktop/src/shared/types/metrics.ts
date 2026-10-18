// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface GpuMetricValue {
    id: string // GPU ID from topology
    timestamp?: number // GPU observation time; absent for receipt-timed sources
    value?: number // percentage; absent when no observation is available
}

export interface NodeItemMetricsEntry {
    timestamp: number // milliseconds since epoch
    cpuUtilization?: number // percentage
    memoryUsage?: number // percentage
    gpuUtilization: GpuMetricValue[] // percentage per GPU
    gpuVramUsage: GpuMetricValue[] // percentage per GPU
}

export interface NodeItemMetrics {
    id: string
    current: NodeItemMetricsEntry
    historical: NodeItemMetricsEntry[]
}
