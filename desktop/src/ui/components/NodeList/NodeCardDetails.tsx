// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import { BarChartOutlined, EditOutlined, SettingsOutlined } from '@/ui/components/icons'
import { DismissibleTooltip } from '@/ui/components/DismissibleTooltip/DismissibleTooltip'
import type { NodeItem } from '@/shared/types/nodes'
import { useMetricsStore } from '@/ui/stores/metrics.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useNodeActiveJobCount } from '@/ui/stores/workloads.store'
import { getGpuColor, getVramColor } from '@/ui/utils/colors'
import GpuRadialChartSvg from '@/ui/components/GpuRadialChartSvg'
import NodeLabel from './NodeLabel'
import NodeActive from '@/ui/components/NodeActive/NodeActive'
import { hasFirstMetrics } from '@/ui/utils/has-first-metrics'
import type { CpuFallbackInfo, GpuInfo } from '@/ui/types/node-hardware'
import { formatBytes, formatServiceBuildShort } from '@/ui/utils/formatters'
import { CHART_COLORS } from '@/ui/constants/colors'
import NodePerformance from './NodePerformance'
import NodeChartLegend from './NodeChartLegend'
import { buildRadialChartMetrics, getMetricReading } from '@/ui/utils/build-radial-chart-metrics'
import { hasInferenceReadyGpu, isGpuInferenceReady } from '@/ui/utils/gpu-inference'
import NodeEnginesInline from './NodeEnginesInline'
import NodeDeclaredRoles from './NodeDeclaredRoles'
import NodeAvailabilityControls from './NodeAvailabilityControls'
import { NodeNameModal, type NodeNameTarget } from '@/ui/components/NodeName/NodeNameModal'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'

const MAX_CHART_SIZE = 60

interface NodeCardDetailsProps {
    node: NodeItem
    /** Hardware label telling this node apart from others with the same name. */
    distinguishingLabel?: string
}

function NodeCardDetails({ node, distinguishingLabel }: NodeCardDetailsProps) {
    const nodeMetrics = useMetricsStore(state => state.nodeMetrics.get(node.id))
    const isLocal = useConnectionStore(state => state.selfId === node.id)
    const [isExpanded, setIsExpanded] = useState(false)
    const focusNodeSettings = useOverviewUiStore(state => state.focusNodeSettings)
    const selfName = useClusterInvitationsStore(state => state.selfName)
    const selfCustomName = useClusterInvitationsStore(state => state.selfCustomName)
    const member = useClusterInvitationsStore(state =>
        state.members.find(value => value.nodeUuid === node.id && value.state === 'member')
    )
    const [renaming, setRenaming] = useState(false)
    const closeRename = useCallback(() => setRenaming(false), [])
    // "Has a GPU" means "has an inference-ready GPU" per the backend readiness
    // list (or any GPU when the backend sends no list); see gpu-inference.ts.
    // A node with none renders the CPU + RAM fallback instead of a GPU ring.
    const hasGpu = hasInferenceReadyGpu(node.topology)
    const baseColor = hasGpu ? getGpuColor(0) : CHART_COLORS.CPU
    const containerRef = useRef<HTMLDivElement>(null)
    const [chartSize, setChartSize] = useState(MAX_CHART_SIZE)
    // Below this card width the expand buttons drop from a right-hand column to
    // a third row under the engine toggles so the node info isn't squeezed.
    const [isNarrow, setIsNarrow] = useState(false)
    const resizeRafRef = useRef(0)

    // Recheck age on existing renders, including expand/resume, without a timer.
    const now = Date.now()
    const chartMetrics = buildRadialChartMetrics(node.topology, nodeMetrics, hasGpu)

    const metricsReady = hasFirstMetrics(node, nodeMetrics)

    const gpuInfo: GpuInfo[] = useMemo(() => {
        if (!hasGpu) return []

        // Keep the original index for colors, join metrics by GPU identity,
        // and drop any GPU the backend did not mark
        // inference-ready so the legend mirrors the radial rings exactly.
        return node.topology.gpus.flatMap((gpu, index) => {
            if (!isGpuInferenceReady(gpu, node.topology.inferenceHardwareIds)) return []
            const usage = getMetricReading(
                nodeMetrics?.gpuUtilization.find(entry => entry.id === gpu.id)?.data,
                now
            )
            const vramUsage = getMetricReading(
                nodeMetrics?.gpuVramUsage.find(entry => entry.id === gpu.id)?.data,
                now
            )

            return [
                {
                    id: gpu.id,
                    name: gpu.name,
                    vramTotal: gpu.vramTotal,
                    vramTotalFormatted: formatBytes(gpu.vramTotal, 1),
                    vramUsageFormatted:
                        vramUsage.value === undefined
                            ? vramUsage.label
                            : formatBytes((gpu.vramTotal * vramUsage.value) / 100, 1),
                    vramUsage,
                    usage,
                    usageColor: getGpuColor(index),
                    vramColor: getVramColor(index)
                }
            ]
        })
    }, [hasGpu, node, nodeMetrics, now])

    // Memoized so the label's segment list keeps a stable identity across
    // renders; NodeLabel observes its segments and would otherwise rebuild the
    // ResizeObserver every render.
    const gpuLabels = useMemo(
        () =>
            hasGpu
                ? node.topology.gpus
                      .filter(gpu => isGpuInferenceReady(gpu, node.topology.inferenceHardwareIds))
                      .map(gpu => gpu.name)
                : [],
        [hasGpu, node.topology]
    )

    const cpuFallbackInfo: CpuFallbackInfo | null = useMemo(() => {
        if (hasGpu) return null
        const ramTotal = node.topology.ram
        const cpuUsage = getMetricReading(nodeMetrics?.cpuUtilization, now)
        const memUsage = getMetricReading(nodeMetrics?.memoryUsage, now)
        return {
            model: node.topology.cpu.model,
            usage: cpuUsage,
            usageColor: CHART_COLORS.CPU,
            memoryUsage: memUsage,
            memoryUsageFormatted:
                memUsage.value === undefined
                    ? memUsage.label
                    : formatBytes((ramTotal * memUsage.value) / 100, 1),
            memoryTotalFormatted: formatBytes(ramTotal, 1),
            memoryColor: CHART_COLORS.MEMORY
        }
    }, [hasGpu, node, nodeMetrics, now])

    const jobCount = useNodeActiveJobCount(node.id)

    const togglePerformance = useCallback(() => {
        setIsExpanded(prev => !prev)
    }, [])

    const measure = useCallback(() => {
        if (!containerRef.current) return
        const { width } = containerRef.current.getBoundingClientRect()
        if (width <= 0) return

        setIsNarrow(width < 550)

        if (width < 300) {
            setChartSize(30)
        } else {
            setChartSize(MAX_CHART_SIZE)
        }
    }, [])

    const scheduleResize = useCallback(() => {
        if (resizeRafRef.current) return
        resizeRafRef.current = requestAnimationFrame(() => {
            resizeRafRef.current = 0
            measure()
        })
    }, [measure])

    useEffect(() => {
        measure()
        const el = containerRef.current
        if (!el) return
        const ro = new ResizeObserver(scheduleResize)
        ro.observe(el)
        return () => {
            ro.disconnect()
        }
    }, [measure, scheduleResize])

    // This node renames itself; another node can be renamed or nicknamed only as a cluster member.
    const renameTarget: NodeNameTarget | null = isLocal
        ? {
              kind: 'self',
              nodeUuid: node.id,
              shownName: selfName || node.name,
              customName: selfCustomName
          }
        : member
          ? { kind: 'peer', node: member }
          : null

    const expandButtons = (placement: 'left' | 'top') => (
        <>
            <DismissibleTooltip slotContent="Show performance" placement={placement}>
                <Button
                    kind={isExpanded ? 'primary' : 'tertiary'}
                    color="neutral"
                    size="tiny"
                    className="px-2"
                    onClick={togglePerformance}
                    aria-label={
                        isExpanded ? 'Collapse performance metrics' : 'Expand performance metrics'
                    }
                    aria-expanded={isExpanded}
                >
                    <BarChartOutlined style={{ fontSize: 14 }} />
                </Button>
            </DismissibleTooltip>
            <DismissibleTooltip slotContent="Node settings" placement={placement}>
                <Button
                    kind="tertiary"
                    color="neutral"
                    size="tiny"
                    className="px-2"
                    onClick={() => focusNodeSettings(node.id)}
                    aria-label="Node settings"
                >
                    <SettingsOutlined style={{ fontSize: 14 }} />
                </Button>
            </DismissibleTooltip>
            {renameTarget && (
                <DismissibleTooltip slotContent="Rename" placement={placement}>
                    <Button
                        kind="tertiary"
                        color="neutral"
                        size="tiny"
                        className="px-2"
                        onClick={() => setRenaming(true)}
                        aria-label="Rename"
                    >
                        <EditOutlined style={{ fontSize: 14 }} />
                    </Button>
                </DismissibleTooltip>
            )}
        </>
    )

    return (
        <div className="node-card pair-paper" data-node-id={node.id} ref={containerRef}>
            <div className="min-w-0 w-full">
                <Flex align="center" gap="3" className="w-full min-w-0 relative">
                    {chartSize <= 30 ? null : (
                        <DismissibleTooltip
                            slotContent={
                                <Stack gap="3">
                                    <NodeChartLegend
                                        gpuInfo={gpuInfo}
                                        cpuFallbackInfo={cpuFallbackInfo}
                                    />
                                </Stack>
                            }
                        >
                            <div className="flex items-center gap-1 shrink-0">
                                <div
                                    className="relative rounded-full"
                                    data-node-gpu-chart-anchor=""
                                >
                                    {metricsReady ? (
                                        <>
                                            <NodeActive show={jobCount > 0} color={baseColor} />
                                            <GpuRadialChartSvg
                                                metrics={chartMetrics}
                                                size={chartSize}
                                                isActive={`0 0 2px 1px ${baseColor}88`}
                                                cutout={0.2}
                                                ringSpacing={1}
                                                valueCornerRadius={0}
                                                bgColor="#00000055"
                                            />
                                        </>
                                    ) : (
                                        <div
                                            aria-hidden
                                            style={{ width: chartSize, height: chartSize }}
                                        />
                                    )}
                                </div>
                            </div>
                        </DismissibleTooltip>
                    )}

                    <Stack className="min-w-0 grow" gap="3">
                        <div className="ml-1">
                            <NodeLabel
                                name={node.name}
                                ipAddress={node.ipAddress}
                                gpuLabels={gpuLabels}
                                isLocal={isLocal}
                                distinguishingLabel={distinguishingLabel}
                            />
                        </div>
                        <Flex
                            align="center"
                            wrap="wrap"
                            gap="4"
                            className="shrink min-w-0 ml-1 -mt-1"
                        >
                            <NodeEnginesInline nodeId={node.id} />
                        </Flex>
                        <div className="ml-1 -mt-1 empty:hidden">
                            <NodeDeclaredRoles nodeId={node.id} />
                        </div>
                        <NodeAvailabilityControls
                            targetId={node.id}
                            nodeId={isLocal ? undefined : node.id}
                            name={node.name}
                            online={node.status !== 'offline'}
                        />
                        <Text
                            kind="body/regular/sm"
                            className="ml-1 text-subtle-color break-all"
                            title={node.serviceBuildId}
                        >
                            {`Service build: ${node.serviceBuildId ? formatServiceBuildShort(node.serviceBuildId) : '--'}`}
                            {node.status === 'offline' &&
                                (node.serviceBuildId
                                    ? ' (cached; node offline)'
                                    : ' (node offline)')}
                        </Text>
                        {isNarrow && (
                            <Flex align="center" gap="2" className="ml-1 -mt-0.5">
                                {expandButtons('top')}
                            </Flex>
                        )}
                    </Stack>

                    {!isNarrow && (
                        <Stack className="shrink-0 mt-1" gap="2">
                            {expandButtons('left')}
                        </Stack>
                    )}
                </Flex>
                <div
                    style={{
                        display: 'grid',
                        gridTemplateRows: isExpanded ? '1fr' : '0fr',
                        transition: 'grid-template-rows 0.25s ease'
                    }}
                >
                    <div className="overflow-hidden min-h-0">
                        <NodePerformance node={node} className="mt-5" />
                    </div>
                </div>
            </div>
            {renameTarget && renaming && (
                <NodeNameModal target={renameTarget} open={renaming} onClose={closeRename} />
            )}
        </div>
    )
}

export default memo(NodeCardDetails)
