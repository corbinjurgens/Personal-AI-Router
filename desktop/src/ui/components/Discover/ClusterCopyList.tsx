// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react'
import { Button, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core'
import type { NodeItem } from '@/shared/types/nodes'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import {
    clusterVariantSize,
    groupClusterModels,
    type ClusterModelLocation
} from '@/ui/utils/group-cluster-models'
import { formatBytes } from '@/ui/utils/formatters'
import { hubModelFacts } from '@/ui/utils/hub-model-facts'
import { Monogram, NodeChip } from '@/ui/components/Page/PagePrimitives'
import { Layers } from '@/ui/components/icons'

/**
 * Models other cluster nodes hold, each copyable to the target node. Copying
 * over the local network is usually much faster than downloading again.
 */
export function ClusterCopyList({
    targetNodeId,
    nodesById,
    onCopy
}: {
    targetNodeId: string
    nodesById: ReadonlyMap<string, NodeItem>
    onCopy: (source: ClusterModelLocation) => void
}) {
    const inventory = useEngineModelsStore(state => state.models)
    const rows = useMemo(
        () =>
            groupClusterModels(
                Array.from(inventory.values()).filter(list => nodesById.has(list.nodeId))
            ).flatMap(group =>
                group.variants
                    .filter(variant =>
                        variant.locations.some(location => location.nodeId !== targetNodeId)
                    )
                    .map(variant => ({ group, variant }))
            ),
        [inventory, nodesById, targetNodeId]
    )
    const name = (nodeId: string) => nodesById.get(nodeId)?.name ?? nodeId

    return (
        <section className="pair-paper p-5 min-h-0 overflow-y-auto" aria-label="Your cluster">
            {rows.length === 0 ? (
                <Stack gap="1" className="py-6 text-center">
                    <Text kind="body/semibold/sm">Nothing to copy yet</Text>
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        No other node in the cluster reports a downloaded model.
                    </Text>
                </Stack>
            ) : (
                <ul className="flex flex-col gap-2" aria-label="Models on other nodes">
                    {rows.map(({ group, variant }) => {
                        const here = variant.locations.find(
                            location => location.nodeId === targetNodeId
                        )
                        const source = variant.locations.find(
                            location => location.nodeId !== targetNodeId
                        )
                        const holders = [
                            ...new Set(
                                variant.locations
                                    .filter(location => location.nodeId !== targetNodeId)
                                    .map(location => location.nodeId)
                            )
                        ].flatMap(id => nodesById.get(id) ?? [])
                        const size = clusterVariantSize(variant)
                        const facts = hubModelFacts(group.name)
                        return (
                            <li
                                key={variant.key}
                                className="pair-tile px-3 py-2.5 flex items-center gap-3"
                            >
                                <Monogram name={facts.title} size={32} />
                                <Stack gap="1" className="min-w-0 grow">
                                    <Text
                                        kind="body/semibold/sm"
                                        className="truncate"
                                        title={group.name}
                                    >
                                        {group.name}
                                    </Text>
                                    <Flex align="center" gap="2" wrap="wrap">
                                        <Tag color="green" density="compact" readOnly>
                                            {variant.quantization}
                                        </Tag>
                                        {size !== undefined && (
                                            <Text
                                                kind="body/regular/xs"
                                                className="text-subtle-color"
                                            >
                                                {formatBytes(size, 1)}
                                            </Text>
                                        )}
                                        {holders.map(holder => (
                                            <NodeChip key={holder.id} node={holder} />
                                        ))}
                                    </Flex>
                                </Stack>
                                {here ? (
                                    <Text
                                        kind="body/regular/sm"
                                        className="text-subtle-color shrink-0"
                                    >
                                        Already on {name(targetNodeId)}
                                    </Text>
                                ) : (
                                    source && (
                                        <Button
                                            size="small"
                                            kind="secondary"
                                            className="shrink-0"
                                            aria-label={`Copy ${group.name} ${variant.quantization} here`}
                                            onClick={() => onCopy(source)}
                                        >
                                            <Layers />
                                            Copy here
                                        </Button>
                                    )
                                )}
                            </li>
                        )
                    })}
                </ul>
            )}
        </section>
    )
}
