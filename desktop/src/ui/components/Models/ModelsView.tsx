// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react'
import { Button, Stack, Text } from '@nvidia/foundations-react-core'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { useOverviewNodes } from '@/ui/hooks/useOverviewNodes'
import { useModelDelete } from '@/ui/hooks/useModelDelete'
import { groupClusterModels, type ClusterModelLocation } from '@/ui/utils/group-cluster-models'
import { formatCopyProgressLabel } from '@/ui/utils/formatters'
import { ModelCopyModal } from '@/ui/components/ModelManager/ModelCopyModal'
import { ModelGroup } from './ModelGroup'
import { ModelDeleteConfirmation } from './ModelDeleteConfirmation'
import type { EngineType } from '@/shared/types/engines'

export function ModelsView() {
    const inventory = useEngineModelsStore(state => state.models)
    const progress = useEngineProgressStore(state => state.progress)
    const nodes = useOverviewNodes()
    const groups = useMemo(() => {
        const ids = new Set(nodes.map(node => node.id))
        return groupClusterModels(
            Array.from(inventory.values()).filter(list => ids.has(list.nodeId))
        )
    }, [inventory, nodes])
    const deletion = useModelDelete()
    const [copy, setCopy] = useState<{
        engineType: EngineType
        source: string
        model: string
        opId?: string
    } | null>(null)
    const openCopy = (location: ClusterModelLocation) =>
        setCopy({
            engineType: location.engineType,
            source: location.nodeId,
            model: location.model.name
        })

    return (
        <Stack gap="6" className="p-4">
            <Text kind="body/bold/md">Models</Text>
            {groups.length === 0 && <Text>No model inventory available</Text>}
            {groups.map(group => (
                <ModelGroup
                    key={group.key}
                    group={group}
                    onCopy={openCopy}
                    onDelete={location => {
                        void deletion.request(
                            location.engineType,
                            location.nodeId,
                            location.model.name
                        )
                    }}
                />
            ))}
            {Array.from(progress.values())
                .filter(item => item.operation === 'copy')
                .map(item => (
                    <Stack gap="2" key={item.opId}>
                        <Text>
                            {item.model} · {formatCopyProgressLabel(item)}
                        </Text>
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
                    </Stack>
                ))}
            {copy && (
                <ModelCopyModal
                    key={`${copy.engineType}:${copy.source}:${copy.model}:${copy.opId ?? ''}`}
                    open
                    onOpenChange={open => {
                        if (!open) setCopy(null)
                    }}
                    engineType={copy.engineType}
                    initialSource={copy.source}
                    initialModel={copy.model}
                    initialOpId={copy.opId}
                />
            )}
            <ModelDeleteConfirmation deletion={deletion} />
        </Stack>
    )
}
