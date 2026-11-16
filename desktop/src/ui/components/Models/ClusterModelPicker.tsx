// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react'
import { Button, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import {
    clusterVariantSize,
    groupClusterModels,
    type ClusterModelLocation
} from '@/ui/utils/group-cluster-models'
import { formatBytes } from '@/ui/utils/formatters'

/** Models held by other cluster nodes, each copyable to the target node. */
export function ClusterModelPicker({
    targetNodeId,
    nodeNames,
    onCopy
}: {
    targetNodeId: string
    nodeNames: ReadonlyMap<string, string>
    onCopy: (source: ClusterModelLocation) => void
}) {
    const inventory = useEngineModelsStore(state => state.models)
    const rows = useMemo(
        () =>
            groupClusterModels(
                Array.from(inventory.values()).filter(list => nodeNames.has(list.nodeId))
            ).flatMap(group =>
                group.variants
                    .filter(variant =>
                        variant.locations.some(location => location.nodeId !== targetNodeId)
                    )
                    .map(variant => ({ group, variant }))
            ),
        [inventory, nodeNames, targetNodeId]
    )
    const name = (nodeId: string) => nodeNames.get(nodeId) ?? nodeId

    if (rows.length === 0) {
        return (
            <Text kind="body/regular/sm" className="text-subtle-color">
                No other node in the cluster reports a downloaded model.
            </Text>
        )
    }
    return (
        <Stack gap="2" className="min-h-0 overflow-y-auto">
            {rows.map(({ group, variant }) => {
                const here = variant.locations.find(location => location.nodeId === targetNodeId)
                const source = variant.locations.find(location => location.nodeId !== targetNodeId)
                const holders = [
                    ...new Set(
                        variant.locations
                            .filter(location => location.nodeId !== targetNodeId)
                            .map(location => name(location.nodeId))
                    )
                ]
                const size = clusterVariantSize(variant)
                return (
                    <Flex
                        key={variant.key}
                        align="center"
                        justify="between"
                        gap="3"
                        className="rounded-md bg-[#ffffff08] px-3 py-2"
                    >
                        <Stack gap="1" className="min-w-0">
                            <Text kind="body/semibold/sm" className="truncate" title={group.name}>
                                {group.name}
                            </Text>
                            <Flex align="center" gap="2" wrap="wrap">
                                <Tag color="green" density="compact" readOnly>
                                    {variant.quantization}
                                </Tag>
                                <Text kind="body/regular/sm" className="text-subtle-color">
                                    {size !== undefined ? `${formatBytes(size, 1)} · ` : ''}
                                    On {holders.join(', ')}
                                </Text>
                            </Flex>
                        </Stack>
                        {here ? (
                            <Text kind="body/regular/sm" className="text-subtle-color shrink-0">
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
                                    Copy here
                                </Button>
                            )
                        )}
                    </Flex>
                )
            })}
        </Stack>
    )
}
