// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Button, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core'
import type {
    ClusterModelGroup,
    ClusterModelLocation,
    ClusterModelVariant
} from '@/ui/utils/group-cluster-models'
import { LocalBadge } from '@/ui/components/LocalBadge'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineDescriptor } from '@/ui/stores/engine-status.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { engineCapabilities } from '@/ui/constants/engine-capabilities'
import { engineLabel, engineManagerName } from '@/shared/utils/engines'
import { isEnginePullInProgress } from '@/shared/utils/engine-progress'
import { formatBytes } from '@/ui/utils/formatters'

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
        <Flex
            gap="3"
            align="center"
            justify="between"
            className="py-2 border-t border-[#ffffff0f] first:border-t-0"
        >
            <Flex gap="2" align="center" className="min-w-0">
                <Text kind="body/semibold/sm" className="shrink-0">
                    {nodeName}
                </Text>
                {local && <LocalBadge />}
                <Text
                    kind="body/regular/sm"
                    className="text-subtle-color truncate"
                    title={model.name}
                >
                    · {model.name}
                </Text>
                <Text kind="body/regular/xs" className="text-subtle2 shrink-0">
                    {engineLabel(engineType, descriptor)}
                </Text>
            </Flex>
            <Flex gap="2" className="shrink-0">
                <Button
                    size="small"
                    kind="secondary"
                    disabled={!allowed(true)}
                    onClick={() => {
                        if (allowed(true)) onCopy(location)
                    }}
                >
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
        </Flex>
    )
}

function variantSize(variant: ClusterModelVariant): number | undefined {
    const model = variant.locations[0].model
    const bytes = model.identity?.totalBytes ?? model.size
    return bytes > 0 ? bytes : undefined
}

/** Splits `owner/name` so the model name leads and the publisher sits beneath it. */
function splitRepository(name: string): { title: string; owner?: string } {
    const slash = name.indexOf('/')
    return slash > 0 && slash < name.length - 1
        ? { title: name.slice(slash + 1), owner: name.slice(0, slash) }
        : { title: name }
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
    const holders = new Set(
        group.variants.flatMap(variant => variant.locations.map(location => location.nodeId))
    ).size
    return (
        <section className="pair-paper p-4" aria-label={group.name}>
            <Flex align="start" justify="between" gap="3">
                <Stack gap="1" className="min-w-0">
                    <Text kind="body/semibold/lg" className="break-words" title={group.name}>
                        {title}
                    </Text>
                    {owner && (
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            {owner}
                        </Text>
                    )}
                </Stack>
                <Text kind="body/regular/sm" className="text-subtle-color shrink-0">
                    On {holders} {holders === 1 ? 'node' : 'nodes'}
                </Text>
            </Flex>
            <Stack gap="2" className="mt-3">
                {group.variants.map(variant => {
                    const size = variantSize(variant)
                    return (
                        <div key={variant.key} className="rounded-md bg-[#ffffff08] px-3 py-2">
                            <Flex align="center" gap="2" wrap="wrap" className="pb-1">
                                <Tag color="green" density="compact" readOnly>
                                    {variant.quantization}
                                </Tag>
                                {size !== undefined && (
                                    <Text kind="body/regular/sm" className="text-subtle-color">
                                        {formatBytes(size, 1)}
                                    </Text>
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
                            {variant.locations.map(location => (
                                <ModelLocation
                                    key={`${location.nodeId}:${location.engineType}:${location.model.name}`}
                                    location={location}
                                    nodeName={nodeNames.get(location.nodeId) ?? location.nodeId}
                                    onCopy={onCopy}
                                    onDelete={onDelete}
                                />
                            ))}
                        </div>
                    )
                })}
            </Stack>
        </section>
    )
}
