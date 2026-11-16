// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from 'react'
import { Button, Flex, ProgressBar, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { useOverviewNodes } from '@/ui/hooks/useOverviewNodes'
import { useModelDelete } from '@/ui/hooks/useModelDelete'
import { groupClusterModels, type ClusterModelLocation } from '@/ui/utils/group-cluster-models'
import { formatCopyProgressLabel } from '@/ui/utils/formatters'
import { ModelCopyModal } from '@/ui/components/ModelManager/ModelCopyModal'
import { Search } from '@/ui/components/icons'
import { ModelGroup } from './ModelGroup'
import { ModelDeleteConfirmation } from './ModelDeleteConfirmation'
import { ModelDownloadDialog } from './ModelDownloadDialog'
import { SourceModelsSection } from './SourceModelsSection'
import { useSourceModelsStore } from '@/ui/stores/source-models.store'
import type { EngineType } from '@/shared/types/engines'

function plural(count: number, one: string, many = `${one}s`): string {
    return `${count} ${count === 1 ? one : many}`
}

export function ModelsView() {
    const inventory = useEngineModelsStore(state => state.models)
    const progress = useEngineProgressStore(state => state.progress)
    const nodes = useOverviewNodes()
    const nodeNames = useMemo(() => new Map(nodes.map(node => [node.id, node.name])), [nodes])
    const groups = useMemo(
        () =>
            groupClusterModels(
                Array.from(inventory.values()).filter(list => nodeNames.has(list.nodeId))
            ),
        [inventory, nodeNames]
    )
    const [query, setQuery] = useState('')
    const refreshSources = useSourceModelsStore(state => state.refreshAll)
    // Sources are not pushed: read them when the tab opens or the node set changes.
    const nodeKey = nodes.map(node => node.id).join(' ')
    useEffect(() => {
        void refreshSources(nodeKey ? nodeKey.split(' ') : [])
    }, [nodeKey, refreshSources])
    const needle = query.trim().toLowerCase()
    const visible = needle
        ? groups.filter(group =>
              [
                  group.name,
                  ...group.variants.flatMap(variant => [
                      variant.quantization,
                      ...variant.locations.flatMap(location => [
                          location.model.name,
                          nodeNames.get(location.nodeId) ?? location.nodeId
                      ])
                  ])
              ].some(text => text.toLowerCase().includes(needle))
          )
        : groups
    const copies = groups.reduce(
        (sum, group) =>
            sum + group.variants.reduce((count, variant) => count + variant.locations.length, 0),
        0
    )
    const holders = new Set(
        groups.flatMap(group =>
            group.variants.flatMap(variant => variant.locations.map(location => location.nodeId))
        )
    ).size
    const transfers = Array.from(progress.values()).filter(item => item.operation === 'copy')
    const deletion = useModelDelete()
    const [copy, setCopy] = useState<{
        engineType: EngineType
        source: string
        model: string
        opId?: string
        destination?: string
    } | null>(null)
    const [downloading, setDownloading] = useState(false)
    const openCopy = (location: ClusterModelLocation) =>
        setCopy({
            engineType: location.engineType,
            source: location.nodeId,
            model: location.model.name
        })

    return (
        <Stack gap="5" className="px-6 py-5 w-full max-w-300">
            <Flex align="end" justify="between" gap="4" wrap="wrap">
                <Stack gap="1">
                    <Text kind="title/sm">Models</Text>
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        {groups.length === 0
                            ? 'Downloaded models on every node in the cluster'
                            : `${plural(groups.length, 'model')} · ${plural(copies, 'copy', 'copies')} on ${plural(holders, 'node')}`}
                    </Text>
                </Stack>
                <Flex align="center" gap="3" wrap="wrap" className="max-w-full">
                    {groups.length > 0 && (
                        <Flex align="center" gap="2" className="w-72 max-w-full">
                            <Search className="shrink-0 text-subtle-color" />
                            <TextInput
                                placeholder="Filter by model, quantization or node"
                                aria-label="Filter models"
                                value={query}
                                onValueChange={setQuery}
                                className="flex-1 min-w-0"
                            />
                        </Flex>
                    )}
                    <Button kind="primary" color="brand" onClick={() => setDownloading(true)}>
                        Download model
                    </Button>
                </Flex>
            </Flex>

            {transfers.length > 0 && (
                <div className="pair-paper p-4">
                    <Stack gap="3">
                        <Text kind="body/semibold/md">Copies in progress</Text>
                        {transfers.map(item => (
                            <Flex key={item.opId} align="center" gap="3" justify="between">
                                <Stack gap="1" className="min-w-0 grow">
                                    <Text kind="body/regular/sm" className="truncate">
                                        {item.model} · {formatCopyProgressLabel(item)}
                                    </Text>
                                    {item.percent !== undefined && (
                                        <ProgressBar
                                            size="small"
                                            value={Math.min(100, Math.max(0, item.percent))}
                                            aria-label={`Copy progress for ${item.model ?? ''}`}
                                        />
                                    )}
                                </Stack>
                                <Button
                                    size="small"
                                    kind="tertiary"
                                    onClick={() =>
                                        setCopy({
                                            engineType: item.engineType,
                                            source: item.sourceNodeId ?? '',
                                            model: item.model ?? '',
                                            opId: item.opId
                                        })
                                    }
                                >
                                    View copy
                                </Button>
                            </Flex>
                        ))}
                    </Stack>
                </div>
            )}

            {groups.length === 0 ? (
                <div className="pair-paper p-6">
                    <Stack gap="1">
                        <Text kind="body/semibold/md">No model inventory available</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            Models appear here once a node reports its downloaded models.
                        </Text>
                    </Stack>
                </div>
            ) : visible.length === 0 ? (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    No models match “{query.trim()}”.
                </Text>
            ) : (
                visible.map(group => (
                    <ModelGroup
                        key={group.key}
                        group={group}
                        nodeNames={nodeNames}
                        onCopy={openCopy}
                        onDelete={location => {
                            void deletion.request(
                                location.engineType,
                                location.nodeId,
                                location.model.name
                            )
                        }}
                    />
                ))
            )}
            <SourceModelsSection nodes={nodes} needle={needle} />
            {copy && (
                <ModelCopyModal
                    key={`${copy.engineType}:${copy.source}:${copy.model}:${copy.opId ?? ''}:${copy.destination ?? ''}`}
                    open
                    onOpenChange={open => {
                        if (!open) setCopy(null)
                    }}
                    engineType={copy.engineType}
                    initialSource={copy.source}
                    initialModel={copy.model}
                    initialOpId={copy.opId}
                    fixedDestination={copy.destination}
                />
            )}
            {downloading && (
                <ModelDownloadDialog
                    open
                    onOpenChange={setDownloading}
                    nodes={nodes}
                    onCopy={(source, destination) => {
                        setDownloading(false)
                        setCopy({
                            engineType: source.engineType,
                            source: source.nodeId,
                            model: source.model.name,
                            destination
                        })
                    }}
                />
            )}
            <ModelDeleteConfirmation deletion={deletion} />
        </Stack>
    )
}
