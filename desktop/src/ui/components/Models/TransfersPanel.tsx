// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { Button, Flex, ProgressBar, Stack, Text } from '@nvidia/foundations-react-core'
import type { EngineProgress } from '@/shared/types/engines'
import type { NodeItem } from '@/shared/types/nodes'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useLocalStorageState } from '@/ui/hooks/useLocalStorageState'
import { engineProgressKey } from '@/shared/utils/engine-progress'
import { formatCopyProgressLabel } from '@/ui/utils/formatters'
import {
    isDownloadActive,
    listModelDownloads,
    type ModelDownload
} from '@/ui/utils/model-downloads'
import { Download, Layers } from '@/ui/components/icons'
import { DownloadProgress } from './DownloadProgress'

const DISMISSED_KEY = 'pair.transfers.dismissedDownloads'

function parseDismissed(raw: unknown): string[] {
    return Array.isArray(raw) ? raw.filter(item => typeof item === 'string') : []
}

/** A finished download is dismissed by its node and target-minted id. */
function outcomeId(download: ModelDownload): string {
    return `${download.nodeId}:${download.entry?.id ?? ''}`
}

function TransferRow({
    icon,
    model,
    route,
    status,
    action
}: {
    icon: ReactNode
    model: string
    route: string
    status: ReactNode
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
                {status}
            </Stack>
            {action}
        </Flex>
    )
}

/**
 * Model downloads and copies anywhere in the cluster: running ones with their
 * progress from the moment Download is clicked, and finished downloads with
 * their outcome until the viewer dismisses them.
 */
export function TransfersPanel({
    nodesById,
    onViewCopy
}: {
    nodesById: ReadonlyMap<string, NodeItem>
    /** Reopens a copy's dialog so its progress and result can be inspected. */
    onViewCopy: (item: EngineProgress) => void
}) {
    const progress = useEngineProgressStore(state => state.progress)
    const pending = usePendingActionsStore(state => state.pending)
    const [dismissed, setDismissed] = useLocalStorageState<string[]>(DISMISSED_KEY, [], {
        parse: parseDismissed
    })
    const copies = Array.from(progress.values()).filter(item => item.operation === 'copy')
    // Departed nodes are left out.
    const downloads = listModelDownloads(progress, pending.values()).filter(download =>
        nodesById.has(download.nodeId)
    )
    const running = downloads.filter(isDownloadActive)
    const finished = downloads.filter(
        download => !isDownloadActive(download) && !dismissed.includes(outcomeId(download))
    )
    // Keep only ids the nodes still report, so the stored list stays bounded.
    const dismiss = (ids: string[]) => {
        const reported = new Set(downloads.map(outcomeId))
        setDismissed(current => [...new Set([...current, ...ids])].filter(id => reported.has(id)))
    }
    const count = copies.length + running.length
    if (count + finished.length === 0) return null
    const name = (id: string | undefined) => (id ? (nodesById.get(id)?.name ?? id) : '')
    const downloadRow = (download: ModelDownload, finishedRow: boolean) => (
        <TransferRow
            key={download.key}
            icon={<Download />}
            model={download.model}
            route={`To ${name(download.nodeId)}`}
            status={
                <DownloadProgress
                    label={download.model}
                    entry={download.entry}
                    starting={download.starting}
                />
            }
            action={
                finishedRow && (
                    <Button
                        size="small"
                        kind="tertiary"
                        className="shrink-0"
                        aria-label={`Dismiss ${download.model}`}
                        onClick={() => dismiss([outcomeId(download)])}
                    >
                        Dismiss
                    </Button>
                )
            }
        />
    )
    return (
        <section
            className="pair-paper p-4 shrink-0 max-h-56 overflow-y-auto"
            aria-label="Transfers"
        >
            <Stack gap="3">
                <Flex align="center" justify="between" gap="2">
                    <span className="pair-eyebrow">Transfers</span>
                    <Flex align="center" gap="2">
                        <Text kind="body/regular/xs" className="text-subtle-color">
                            {count} running
                        </Text>
                        {finished.length > 0 && (
                            <Button
                                size="small"
                                kind="tertiary"
                                onClick={() => dismiss(finished.map(outcomeId))}
                            >
                                Clear finished
                            </Button>
                        )}
                    </Flex>
                </Flex>
                {running.map(download => downloadRow(download, false))}
                {copies.map(item => (
                    <TransferRow
                        key={engineProgressKey(item)}
                        icon={<Layers />}
                        model={item.model ?? ''}
                        route={`${name(item.sourceNodeId)} to ${name(item.destinationNodeId ?? item.nodeId)}`}
                        status={
                            <Stack gap="1">
                                {item.percent !== undefined && (
                                    <ProgressBar
                                        size="small"
                                        value={Math.min(100, Math.max(0, item.percent))}
                                        aria-label={`Copy progress for ${item.model ?? ''}`}
                                    />
                                )}
                                <Text kind="body/regular/xs" className="text-subtle-color">
                                    {item.percent !== undefined
                                        ? `${formatCopyProgressLabel(item)} · ${Math.round(item.percent)}%`
                                        : formatCopyProgressLabel(item)}
                                </Text>
                            </Stack>
                        }
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
                {finished.map(download => downloadRow(download, true))}
            </Stack>
        </section>
    )
}
