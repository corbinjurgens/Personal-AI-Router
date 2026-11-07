// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { ClusterModelGroup, ClusterModelLocation } from '@/ui/utils/group-cluster-models'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineDescriptor } from '@/ui/stores/engine-status.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { engineCapabilities } from '@/ui/constants/engine-capabilities'
import { engineManagerName } from '@/shared/utils/engines'
import { isEnginePullInProgress } from '@/shared/utils/engine-progress'

function ModelLocation({
    location,
    onCopy,
    onDelete
}: {
    location: ClusterModelLocation
    onCopy: (location: ClusterModelLocation) => void
    onDelete: (location: ClusterModelLocation) => void
}) {
    const { nodeId, engineType, model } = location
    const nodes = useNodesStore(state => state.nodes)
    const inventory = useEngineModelsStore(state => state.models.get(`${nodeId}:${engineType}`))
    const progress = useEngineProgressStore(state => state.progress)
    const pending = usePendingActionsStore(state => state.pending)
    void progress
    void pending
    const caps = engineCapabilities(engineType, useEngineDescriptor(engineType))
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
        <Flex gap="3" align="center" justify="between">
            <Text kind="body/regular/sm">
                {nodes.get(nodeId)?.name ?? nodeId} · {model.name} · {engineType}
            </Text>
            <Flex gap="2">
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

export function ModelGroup({
    group,
    onCopy,
    onDelete
}: {
    group: ClusterModelGroup
    onCopy: (location: ClusterModelLocation) => void
    onDelete: (location: ClusterModelLocation) => void
}) {
    return (
        <Stack gap="3">
            <Text kind="body/bold/md">{group.name}</Text>
            {group.variants.map(variant => (
                <Stack key={variant.key} gap="2" className="pl-4">
                    <Text kind="body/regular/sm">{variant.quantization}</Text>
                    {variant.unverified && (
                        <Text kind="body/regular/sm">Same file not verified</Text>
                    )}
                    {variant.locations.map(location => (
                        <ModelLocation
                            key={`${location.nodeId}:${location.engineType}:${location.model.name}`}
                            location={location}
                            onCopy={onCopy}
                            onDelete={onDelete}
                        />
                    ))}
                </Stack>
            ))}
        </Stack>
    )
}
