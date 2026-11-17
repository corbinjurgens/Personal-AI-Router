// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { Button, Flex, ProgressBar, Stack, Text } from '@nvidia/foundations-react-core'
import type { EngineProgress } from '@/shared/types/engines'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { formatCopyProgressLabel, formatDownloadStatus } from '@/ui/utils/formatters'
import { Download, Layers } from '@/ui/components/icons'

function TransferRow({
    icon,
    model,
    route,
    percent,
    label,
    action
}: {
    icon: ReactNode
    model: string
    route: string
    percent?: number
    label: string
    action?: ReactNode
}) {
    return (
        <Flex align="center" gap="3">
            <span className="text-subtle-color shrink-0">{icon}</span>
            <Stack gap="1" className="min-w-0 grow">
                <Flex align="center" justify="between" gap="3">
                    <Text kind="body/semibold/sm" className="truncate" title={model}>
                        {model}
                    </Text>
                    <Text kind="body/regular/xs" className="text-subtle-color shrink-0">
                        {route}
                    </Text>
                </Flex>
                {percent !== undefined && (
                    <ProgressBar
                        size="small"
                        value={Math.min(100, Math.max(0, percent))}
                        aria-label={`Progress for ${model}`}
                    />
                )}
                <Text kind="body/regular/xs" className="text-subtle-color">
                    {percent !== undefined ? `${label} · ${Math.round(percent)}%` : label}
                </Text>
            </Stack>
            {action}
        </Flex>
    )
}

/** Model downloads and copies currently running anywhere in the cluster. */
export function TransfersPanel({
    nodeNames,
    onViewCopy
}: {
    nodeNames: ReadonlyMap<string, string>
    /** Reopens a copy's dialog so its progress and result can be inspected. */
    onViewCopy: (item: EngineProgress) => void
}) {
    const progress = useEngineProgressStore(state => state.progress)
    const inventory = useEngineModelsStore(state => state.models)
    const copies = Array.from(progress.values()).filter(item => item.operation === 'copy')
    // Downloads come from each node's current download state, which owns the
    // download phase, so a download started elsewhere shows here too.
    const downloads = Array.from(inventory.values()).flatMap(list =>
        list.downloadStateCurrent === true && list.downloadState
            ? list.downloadState.entries
                  .filter(entry => entry.phase === 'queued' || entry.phase === 'in_progress')
                  .map(entry => ({ nodeId: list.nodeId, entry }))
            : []
    )
    const count = copies.length + downloads.length
    if (count === 0) return null
    const name = (id: string | undefined) => (id ? (nodeNames.get(id) ?? id) : '')
    return (
        <section className="pair-paper p-4" aria-label="Transfers in progress">
            <Stack gap="3">
                <Flex align="center" justify="between">
                    <span className="pair-eyebrow">In progress</span>
                    <Text kind="body/regular/xs" className="text-subtle-color">
                        {count} {count === 1 ? 'transfer' : 'transfers'}
                    </Text>
                </Flex>
                {downloads.map(({ nodeId, entry }) => (
                    <TransferRow
                        key={`${nodeId}:${entry.id}`}
                        icon={<Download />}
                        model={entry.model}
                        route={`To ${name(nodeId)}`}
                        percent={entry.phase === 'in_progress' ? entry.percent : undefined}
                        label={formatDownloadStatus(entry.phase, true)}
                    />
                ))}
                {copies.map(item => (
                    <TransferRow
                        key={item.opId ?? `${item.nodeId}:${item.engineType}:${item.model ?? ''}`}
                        icon={<Layers />}
                        model={item.model ?? ''}
                        route={`${name(item.sourceNodeId)} to ${name(item.destinationNodeId ?? item.nodeId)}`}
                        percent={item.percent}
                        label={formatCopyProgressLabel(item)}
                        action={
                            <Button
                                size="small"
                                kind="tertiary"
                                className="shrink-0"
                                onClick={() => onViewCopy(item)}
                            >
                                View copy
                            </Button>
                        }
                    />
                ))}
            </Stack>
        </section>
    )
}
