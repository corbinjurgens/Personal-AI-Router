// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import { Badge, Button, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core'
import type { EngineCatalogFile } from '@/shared/types/engine-api'
import type { ModelItem } from '@/shared/types/engines'
import type { NodeItem } from '@/shared/types/nodes'
import { engineProgressKey } from '@/shared/utils/engine-progress'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { formatBytes } from '@/ui/utils/formatters'
import { isDownloadActive, listModelDownloads } from '@/ui/utils/model-downloads'
import { DownloadProgress } from '@/ui/components/Models/DownloadProgress'
import { isHubEntryDownloaded } from '@/ui/utils/match-downloaded-model'
import { NodeChip } from '@/ui/components/Page/PagePrimitives'
import { Download } from '@/ui/components/icons'

const COLLAPSED_COUNT = 6

/**
 * The quantizations of one repository, smallest first, each with the nodes
 * that already hold it and a download to the target node.
 */
export function QuantizationList({
    files,
    targetNodeId,
    targetName,
    downloadedModels,
    holders,
    recommended,
    onDownload
}: {
    files: readonly EngineCatalogFile[]
    targetNodeId: string
    targetName: string
    /** The target node's llama.cpp models, to mark files it already holds. */
    downloadedModels: readonly ModelItem[]
    /** The nodes that hold a quantization, for the "On" chips. */
    holders: (quantization: string) => NodeItem[]
    /** The quantization the catalog suggests, marked as a starting point. */
    recommended?: string
    onDownload: (name: string) => void
}) {
    const [showAll, setShowAll] = useState(false)
    const progress = useEngineProgressStore(state => state.progress)
    const pending = usePendingActionsStore(state => state.pending)
    const downloads = new Map(
        listModelDownloads(progress, pending.values()).map(download => [download.key, download])
    )
    // The catalog names a plain quantization ("Q4_K_M"); some publishers label
    // that file with a prefix ("UD-Q4_K_M").
    const wanted = recommended?.toUpperCase()
    const suggested = wanted
        ? (files.find(file => file.quantization.toUpperCase() === wanted) ??
          files.find(file => file.quantization.toUpperCase().endsWith(`-${wanted}`)))
        : undefined
    // The suggested file leads; the rest follow from smallest to largest.
    const sorted = [...files].sort(
        (a, b) => Number(b === suggested) - Number(a === suggested) || a.sizeBytes - b.sizeBytes
    )
    const visible = showAll ? sorted : sorted.slice(0, COLLAPSED_COUNT)
    return (
        <Stack gap="2">
            <ul className="flex flex-col gap-1.5" aria-label="Quantizations">
                {visible.map(file => {
                    const onTarget = isHubEntryDownloaded(
                        'llama-cpp',
                        { id: file.name },
                        downloadedModels
                    )
                    const download = downloads.get(
                        engineProgressKey({
                            nodeId: targetNodeId,
                            engineType: 'llama-cpp',
                            operation: 'pull',
                            model: file.name
                        })
                    )
                    const pulling = isDownloadActive(download)
                    // Once the file is on the target its "On" note says so.
                    const showDownload =
                        download !== undefined &&
                        !(onTarget && download.entry?.phase === 'complete')
                    const nodes = holders(file.quantization)
                    return (
                        <li
                            key={file.name}
                            className="pair-tile px-3 py-2.5 flex items-center gap-3 min-w-0"
                            style={
                                file === suggested
                                    ? {
                                          boxShadow:
                                              'inset 0 0 0 1px color-mix(in srgb, var(--color-brand, #76b900) 45%, transparent)'
                                      }
                                    : undefined
                            }
                        >
                            <Stack gap="1" className="min-w-0 grow">
                                <Flex align="center" gap="2" wrap="wrap">
                                    <Tag color="green" density="compact" readOnly>
                                        {file.quantization}
                                    </Tag>
                                    <Text kind="body/semibold/sm">
                                        {formatBytes(file.sizeBytes, 1)}
                                    </Text>
                                    {file.shards > 1 && (
                                        <Text kind="body/regular/xs" className="text-subtle-color">
                                            {file.shards} parts
                                        </Text>
                                    )}
                                    {file === suggested && (
                                        <Badge color="green" kind="outline">
                                            Suggested
                                        </Badge>
                                    )}
                                </Flex>
                                {nodes.length > 0 && (
                                    <Flex align="center" gap="1.5" wrap="wrap">
                                        <Text kind="body/regular/xs" className="text-subtle-color">
                                            On
                                        </Text>
                                        {nodes.map(node => (
                                            <NodeChip key={node.id} node={node} />
                                        ))}
                                    </Flex>
                                )}
                                {showDownload && (
                                    <DownloadProgress
                                        label={file.quantization}
                                        entry={download.entry}
                                        starting={download.starting}
                                    />
                                )}
                            </Stack>
                            {onTarget ? (
                                <Text kind="body/regular/sm" className="text-subtle-color shrink-0">
                                    On {targetName}
                                </Text>
                            ) : (
                                <Button
                                    size="small"
                                    kind={file === suggested ? 'primary' : 'secondary'}
                                    color={file === suggested ? 'brand' : undefined}
                                    className="shrink-0"
                                    disabled={pulling}
                                    aria-label={`Download ${file.quantization}`}
                                    onClick={() => onDownload(file.name)}
                                >
                                    <Download />
                                    {pulling
                                        ? 'Downloading'
                                        : download?.entry?.phase === 'failed'
                                          ? 'Retry'
                                          : 'Download'}
                                </Button>
                            )}
                        </li>
                    )
                })}
            </ul>
            {sorted.length > COLLAPSED_COUNT && (
                <Button
                    kind="tertiary"
                    size="small"
                    className="self-start"
                    onClick={() => setShowAll(value => !value)}
                >
                    {showAll ? 'Show fewer' : `Show all ${sorted.length} quantizations`}
                </Button>
            )}
        </Stack>
    )
}
