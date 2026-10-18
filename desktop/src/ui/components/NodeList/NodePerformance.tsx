// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useState } from 'react'
import { Flex, Stack, Text } from '@nvidia/foundations-react-core'
import SvgLineChart from '@/ui/components/SvgLineChart'
import type { MetricDataset } from '@/ui/utils/line-chart-path'
import type { NodeItem } from '@/shared/types/nodes'
import { useMetricsStore } from '@/ui/stores/metrics.store'
import { getGpuColor, getVramColor } from '@/ui/utils/colors'
import { formatBytes } from '@/ui/utils/formatters'
import { CHART_COLORS } from '@/ui/constants/colors'
import { buildPerformanceHardwareInfo } from '@/ui/utils/performance-hardware-info'

export default function NodePerformance({
    node,
    className
}: {
    node: NodeItem
    className?: string
}) {
    const nodeMetrics = useMetricsStore(state => state.nodeMetrics.get(node.id))
    const now = Date.now()
    const [selectedMetric, setSelectedMetric] = useState<string | undefined>(undefined)

    const handleLegendClick = useCallback((metricKey: string) => {
        // If clicking the same metric that's already selected, show all
        // Otherwise, show only the clicked metric (solo it)
        setSelectedMetric(prev => (prev === metricKey ? undefined : metricKey))
    }, [])

    // The detailed performance view intentionally charts EVERY detected GPU,
    // including any the radial omits (the radial filters to inference-ready GPUs
    // via inferenceHardwareIds, src/ui/utils/gpu-inference.ts). Here the user is
    // inspecting raw hardware activity, so an iGPU's utilization is useful signal
    // rather than noise.
    const datasets = useMemo(() => {
        const gpuCount = node.topology.gpus.length
        const result: MetricDataset[] = []

        // Add GPU utilization datasets (cycling through color palette)
        node.topology.gpus.forEach((gpu, index) => {
            const color = getGpuColor(index)
            result.push({
                data: nodeMetrics?.gpuUtilization.find(entry => entry.id === gpu.id)?.data ?? [],
                label: gpuCount > 1 ? `GPU ${index}` : 'GPU',
                color,
                key: `gpu-${gpu.id}`
            })
        })

        // Add CPU
        result.push({
            data: nodeMetrics?.cpuUtilization ?? [],
            label: 'CPU',
            color: CHART_COLORS.CPU,
            key: 'cpu'
        })

        // Add RAM
        result.push({
            data: nodeMetrics?.memoryUsage ?? [],
            label: 'RAM',
            color: CHART_COLORS.MEMORY,
            key: 'memory'
        })

        // Add VRAM datasets (cycling through color palette)
        node.topology.gpus.forEach((gpu, index) => {
            const color = getVramColor(index)
            result.push({
                data: nodeMetrics?.gpuVramUsage.find(entry => entry.id === gpu.id)?.data ?? [],
                label: gpuCount > 1 ? `VRAM ${index}` : 'VRAM',
                color,
                key: `vram-${gpu.id}`
            })
        })

        return result
    }, [node.topology.gpus, nodeMetrics])

    const hardwareInfo = useMemo(
        () => buildPerformanceHardwareInfo(node.topology, nodeMetrics, now),
        [node.topology, nodeMetrics, now]
    )

    return (
        <Flex
            className={`w-full min-w-0 flex-nowrap max-[1000px]:flex-wrap ${className ? className : ''}`}
            align="stretch"
            gap="2"
        >
            {/* Chart Section (placeholder when no metrics) */}
            <div className="grow min-w-0 overflow-hidden min-h-[200px] relative">
                <SvgLineChart
                    datasets={datasets}
                    maxDataPoints={30}
                    yAxisMax={100}
                    selectedKey={selectedMetric}
                />
                {!nodeMetrics && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            No metrics yet
                        </Text>
                    </div>
                )}
            </div>

            {/* Custom Legend / Hardware Info Section */}
            <Stack gap="4" className="shrink-0 w-50 ml-2 self-center">
                {/* GPU Metrics */}
                {hardwareInfo.gpus.map(gpu => {
                    // Check if this GPU's metrics are selected
                    const isGpuSelected =
                        selectedMetric === undefined ||
                        selectedMetric === `gpu-${gpu.id}` ||
                        selectedMetric === `vram-${gpu.id}`

                    return (
                        <Stack gap="0" key={gpu.id}>
                            <Text
                                kind="body/semibold/sm"
                                className="transition-opacity text-nowrap text-ellipsis overflow-hidden"
                                style={{ opacity: isGpuSelected ? 1 : 0.5 }}
                                title={gpu.name}
                            >
                                {gpu.name}
                            </Text>

                            <Stack gap="1">
                                {/* GPU Utilization */}
                                <Flex
                                    align="center"
                                    gap="2"
                                    className="cursor-pointer"
                                    onClick={() => handleLegendClick(`gpu-${gpu.id}`)}
                                >
                                    <div
                                        className="w-3 h-3 min-w-3 min-h-3 max-w-3 max-h-3 rounded-full transition-opacity"
                                        style={{
                                            backgroundColor: `${gpu.color}BF`,
                                            border: `2px solid ${gpu.color}`,
                                            marginTop: '1px',
                                            opacity:
                                                selectedMetric === undefined ||
                                                selectedMetric === `gpu-${gpu.id}`
                                                    ? 1
                                                    : 0.3
                                        }}
                                    />

                                    <Flex
                                        align="center"
                                        gap="2"
                                        className="transition-opacity"
                                        style={{
                                            opacity:
                                                selectedMetric === undefined ||
                                                selectedMetric === `gpu-${gpu.id}`
                                                    ? 1
                                                    : 0.5
                                        }}
                                    >
                                        <Text kind="body/regular/sm">Usage</Text>
                                        <Text
                                            kind="body/regular/sm"
                                            className={
                                                gpu.utilization.status !== 'current'
                                                    ? 'text-subtle-color'
                                                    : undefined
                                            }
                                        >
                                            {gpu.utilization.label}
                                        </Text>
                                    </Flex>
                                </Flex>

                                {/* GPU VRAM */}
                                <Flex
                                    align="center"
                                    gap="2"
                                    className="cursor-pointer"
                                    onClick={() => handleLegendClick(`vram-${gpu.id}`)}
                                >
                                    <div
                                        className="w-3 h-3 min-w-3 min-h-3 max-w-3 max-h-3 rounded-full transition-opacity"
                                        style={{
                                            backgroundColor: `${gpu.vramColor}BF`,
                                            border: `2px solid ${gpu.vramColor}`,
                                            marginTop: '1px',
                                            opacity:
                                                selectedMetric === undefined ||
                                                selectedMetric === `vram-${gpu.id}`
                                                    ? 1
                                                    : 0.3
                                        }}
                                    />
                                    <Flex
                                        align="center"
                                        gap="2"
                                        className="transition-opacity"
                                        style={{
                                            opacity:
                                                selectedMetric === undefined ||
                                                selectedMetric === `vram-${gpu.id}`
                                                    ? 1
                                                    : 0.5
                                        }}
                                    >
                                        <Text kind="body/regular/sm">VRAM</Text>
                                        <Text
                                            kind="body/regular/sm"
                                            className={
                                                gpu.vramUsage.status !== 'current'
                                                    ? 'text-subtle-color'
                                                    : undefined
                                            }
                                        >
                                            {gpu.vramUsage.value === undefined
                                                ? gpu.vramUsage.label
                                                : formatBytes(
                                                      Math.floor(
                                                          (gpu.vramTotal * gpu.vramUsage.value) /
                                                              100
                                                      ),
                                                      1
                                                  )}{' '}
                                            / {gpu.vramFormatted}
                                        </Text>
                                    </Flex>
                                </Flex>
                            </Stack>
                        </Stack>
                    )
                })}

                {/* CPU */}
                <Flex
                    align="start"
                    gap="2"
                    className="cursor-pointer"
                    onClick={() => handleLegendClick('cpu')}
                >
                    <div
                        className="w-3 h-3 min-w-3 min-h-3 max-w-3 max-h-3 rounded-full transition-opacity"
                        style={{
                            backgroundColor: `${CHART_COLORS.CPU}BF`,
                            border: `2px solid ${CHART_COLORS.CPU}`,
                            marginTop: '4px',
                            opacity:
                                selectedMetric === undefined || selectedMetric === 'cpu' ? 1 : 0.3
                        }}
                    />
                    <Stack
                        className="transition-opacity overflow-hidden"
                        style={{
                            opacity:
                                selectedMetric === undefined || selectedMetric === 'cpu' ? 1 : 0.5
                        }}
                    >
                        <Text
                            kind="body/semibold/sm"
                            className="text-nowrap text-ellipsis overflow-hidden"
                            title={hardwareInfo.cpuModel}
                        >
                            {hardwareInfo.cpuModel}
                        </Text>
                        <Text
                            kind="body/regular/sm"
                            className={
                                hardwareInfo.cpuUtilization.status !== 'current'
                                    ? 'text-subtle-color'
                                    : undefined
                            }
                        >
                            {hardwareInfo.cpuUtilization.label}{' '}
                            {hardwareInfo.cpuCores ? `(${hardwareInfo.cpuCores} cores)` : ''}
                        </Text>
                    </Stack>
                </Flex>

                {/* Memory */}
                <Flex
                    align="start"
                    gap="2"
                    className="cursor-pointer"
                    onClick={() => handleLegendClick('memory')}
                >
                    <div
                        className="w-3 h-3 min-w-3 min-h-3 max-w-3 max-h-3 rounded-full transition-opacity"
                        style={{
                            backgroundColor: `${CHART_COLORS.MEMORY}BF`,
                            border: `2px solid ${CHART_COLORS.MEMORY}`,
                            marginTop: '4px',
                            opacity:
                                selectedMetric === undefined || selectedMetric === 'memory'
                                    ? 1
                                    : 0.3
                        }}
                    />
                    <Stack
                        className="transition-opacity"
                        style={{
                            opacity:
                                selectedMetric === undefined || selectedMetric === 'memory'
                                    ? 1
                                    : 0.5
                        }}
                    >
                        <Text kind="body/semibold/sm">Memory</Text>
                        <Text
                            kind="body/regular/sm"
                            className={
                                hardwareInfo.memoryUsage.status !== 'current'
                                    ? 'text-subtle-color'
                                    : undefined
                            }
                        >
                            {hardwareInfo.memoryUsage.value === undefined
                                ? hardwareInfo.memoryUsage.label
                                : formatBytes(
                                      Math.floor(
                                          (hardwareInfo.memoryTotal *
                                              hardwareInfo.memoryUsage.value) /
                                              100
                                      ),
                                      1
                                  )}{' '}
                            / {hardwareInfo.memory}
                        </Text>
                    </Stack>
                </Flex>
            </Stack>
        </Flex>
    )
}
