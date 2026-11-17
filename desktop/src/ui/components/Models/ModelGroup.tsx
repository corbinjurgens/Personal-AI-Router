// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Button, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core'
import {
    clusterVariantSize,
    type ClusterModelGroup,
    type ClusterModelLocation
} from '@/ui/utils/group-cluster-models'
import { Monogram, NodeChip } from '@/ui/components/Page/PagePrimitives'
import { Layers } from '@/ui/components/icons'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineDescriptor } from '@/ui/stores/engine-status.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { engineCapabilities } from '@/ui/constants/engine-capabilities'
import { engineLabel, engineManagerName } from '@/shared/utils/engines'
import { isEnginePullInProgress } from '@/shared/utils/engine-progress'
import { formatBytes } from '@/ui/utils/formatters'
import { hubModelFacts } from '@/ui/utils/hub-model-facts'

function ModelLocation({
    location,
    nodeName,
    onCopy,
    onDelete
}: {
    location: ClusterModelLocation
    nodeName: string
    onCopy: (location: ClusterModelLocation) => void
    onDelete: (location: ClusterModelLocation) => void
}) {
    const { nodeId, engineType, model } = location
    const local = useConnectionStore(state => state.selfId === nodeId)
    const inventory = useEngineModelsStore(state => state.models.get(`${nodeId}:${engineType}`))
    const progress = useEngineProgressStore(state => state.progress)
    const pending = usePendingActionsStore(state => state.pending)
    void progress
    void pending
    const descriptor = useEngineDescriptor(engineType)
    const caps = engineCapabilities(engineType, descriptor)
    const allowed = (copy: boolean) => {
        const current = useEngineModelsStore.getState().getModels(nodeId, engineType)
        const pull = useEngineProgressStore
            .getState()
            .getProgress(nodeId, engineType, 'pull', model.name)
        const entry = current.downloadState?.entries.find(
            row => row.engine === engineManagerName(engineType) && row.model === model.name
        )
        if (
            usePendingActionsStore.getState().getModelPending(nodeId, engineType, model.name) ||
            isEnginePullInProgress(pull) ||
            entry?.phase === 'queued' ||
            entry?.phase === 'in_progress'
        )
            return false
        if (!copy) return caps.hasDeleteModel
        return (
            caps.hasModelCopy &&
            current.downloadStateCurrent === true &&
            !!current.downloadState &&
            !current.downloadState.recoveryRequired &&
            current.models.some(row => row.name === model.name && row.downloaded) &&
            !entry?.blocksReadiness &&
            !pull.downloadState?.blocksReadiness &&
            (!pull.downloadState ||
                (pull.downloadStateCurrent === true &&
                    pull.downloadSourceId === current.downloadState.sourceId &&
                    pull.downloadRevision === current.downloadState.revision))
        )
    }
    void inventory
    return (
        <li className="flex items-center gap-3 py-2 pair-divider-top first:border-t-0">
            <Flex gap="2" align="center" className="min-w-0 grow">
                <NodeChip name={nodeName} local={local} title={`Served as ${model.name}`} />
                <Text
                    kind="mono/sm"
                    className="text-subtle-color truncate min-w-0"
                    title={model.name}
                >
                    {model.name}
                </Text>
                <Text kind="body/regular/xs" className="text-subtle2 shrink-0 hidden md:inline">
                    {engineLabel(engineType, descriptor)}
                </Text>
            </Flex>
            <Flex gap="1" className="shrink-0">
                <Button
                    size="small"
                    kind="secondary"
                    disabled={!allowed(true)}
                    onClick={() => {
                        if (allowed(true)) onCopy(location)
                    }}
                >
                    <Layers />
                    Copy to node
                </Button>
                <Button
                    size="small"
                    kind="tertiary"
                    disabled={!allowed(false)}
                    onClick={() => {
                        if (allowed(false)) onDelete(location)
                    }}
                >
                    Delete on node
                </Button>
            </Flex>
        </li>
    )
}

/** The model name leads and the publisher sits beneath it. */
function splitRepository(name: string): { title: string; owner?: string } {
    const slash = name.indexOf('/')
    return {
        title: hubModelFacts(name).title,
        owner: slash > 0 && slash < name.length - 1 ? name.slice(0, slash) : undefined
    }
}

export function ModelGroup({
    group,
    nodeNames,
    onCopy,
    onDelete
}: {
    group: ClusterModelGroup
    nodeNames: ReadonlyMap<string, string>
    onCopy: (location: ClusterModelLocation) => void
    onDelete: (location: ClusterModelLocation) => void
}) {
    const { title, owner } = splitRepository(group.name)
    const holderIds = [
        ...new Set(
            group.variants.flatMap(variant => variant.locations.map(location => location.nodeId))
        )
    ]
    return (
        <section className="pair-paper p-4" aria-label={group.name}>
            <Flex align="center" gap="3">
                <Monogram name={title} size={40} />
                <Stack gap="0" className="min-w-0 grow">
                    <Text kind="body/semibold/lg" className="truncate" title={group.name}>
                        {title}
                    </Text>
                    <Text kind="body/regular/sm" className="text-subtle-color truncate">
                        {[
                            owner,
                            `${group.variants.length} ${group.variants.length === 1 ? 'quantization' : 'quantizations'}`
                        ]
                            .filter(Boolean)
                            .join(' · ')}
                    </Text>
                </Stack>
                <Flex gap="1" wrap="wrap" justify="end" className="shrink-0 max-w-[45%]">
                    {holderIds.map(id => (
                        <NodeChip key={id} name={nodeNames.get(id) ?? id} />
                    ))}
                </Flex>
            </Flex>
            <Stack gap="2" className="mt-4">
                {group.variants.map(variant => {
                    const size = clusterVariantSize(variant)
                    return (
                        <div key={variant.key} className="pair-tile px-3 pt-2.5 pb-1">
                            <Flex align="center" gap="2" wrap="wrap" className="pb-1.5">
                                <Tag color="green" density="compact" readOnly>
                                    {variant.quantization}
                                </Tag>
                                {size !== undefined && (
                                    <Text kind="body/semibold/sm">{formatBytes(size, 1)}</Text>
                                )}
                                {variant.unverified && (
                                    <Badge
                                        color="yellow"
                                        title="These copies share a name and quantization, but PAIR has not confirmed they are the same file."
                                    >
                                        Same file not verified
                                    </Badge>
                                )}
                            </Flex>
                            <ul className="m-0 p-0 list-none">
                                {variant.locations.map(location => (
                                    <ModelLocation
                                        key={`${location.nodeId}:${location.engineType}:${location.model.name}`}
                                        location={location}
                                        nodeName={nodeNames.get(location.nodeId) ?? location.nodeId}
                                        onCopy={onCopy}
                                        onDelete={onDelete}
                                    />
                                ))}
                            </ul>
                        </div>
                    )
                })}
            </Stack>
        </section>
    )
}
