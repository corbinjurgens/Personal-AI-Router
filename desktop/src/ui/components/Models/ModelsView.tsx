// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from 'react'
import { Button, Flex, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useOverviewNodes } from '@/ui/hooks/useOverviewNodes'
import { useModelDelete } from '@/ui/hooks/useModelDelete'
import {
    clusterVariantSize,
    groupClusterModels,
    type ClusterModelLocation
} from '@/ui/utils/group-cluster-models'
import { formatBytes } from '@/ui/utils/formatters'
import { ModelCopyModal } from '@/ui/components/ModelManager/ModelCopyModal'
import { Download, Search } from '@/ui/components/icons'
import { PageHeading, StatTile } from '@/ui/components/Page/PagePrimitives'
import { TransfersPanel } from './TransfersPanel'
import { ModelGroup } from './ModelGroup'
import { ModelDeleteConfirmation } from './ModelDeleteConfirmation'
import { SourceModelsSection } from './SourceModelsSection'
import { AddModelFileDialog } from './AddModelFileDialog'
import ModelSettingsPanel from '@/ui/components/ModelSettings/ModelSettingsPanel'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { SOURCE_MODELS_ENGINE_TYPE, useSourceModelsStore } from '@/ui/stores/source-models.store'
import { engineManagerName } from '@/shared/utils/engines'
import type { EngineType } from '@/shared/types/engines'

export function ModelsView() {
    const inventory = useEngineModelsStore(state => state.models)
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
    const deletion = useModelDelete()
    const [copy, setCopy] = useState<{
        engineType: EngineType
        source: string
        model: string
        opId?: string
        destination?: string
    } | null>(null)
    const setActiveTab = useOverviewUiStore(state => state.setActiveTab)
    const [addingFile, setAddingFile] = useState(false)
    const [profileFor, setProfileFor] = useState<string | null>(null)
    const selfId = useConnectionStore(state => state.selfId)
    const openCopy = (location: ClusterModelLocation) =>
        setCopy({
            engineType: location.engineType,
            source: location.nodeId,
            model: location.model.name
        })

    const totalBytes = groups.reduce(
        (sum, group) =>
            sum +
            group.variants.reduce(
                (bytes, variant) =>
                    bytes + (clusterVariantSize(variant) ?? 0) * variant.locations.length,
                0
            ),
        0
    )

    return (
        <Stack gap="5" className="px-6 py-5 w-full max-w-300">
            <PageHeading
                title="Models"
                description="Every model downloaded in your cluster, grouped by model and quantization."
                actions={
                    <>
                        <Button kind="secondary" onClick={() => setAddingFile(true)}>
                            Add from file
                        </Button>
                        <Button
                            kind="primary"
                            color="brand"
                            onClick={() => setActiveTab('discover')}
                        >
                            <Download />
                            Download model
                        </Button>
                    </>
                }
            />

            {groups.length > 0 && (
                <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                    <StatTile label="Models" value={groups.length} />
                    <StatTile label="Copies" value={copies} />
                    <StatTile label="Nodes with models" value={holders} />
                    <StatTile
                        label="Disk used"
                        value={totalBytes > 0 ? formatBytes(totalBytes, 1) : 'Not reported'}
                    />
                </div>
            )}

            <TransfersPanel
                nodeNames={nodeNames}
                onViewCopy={item =>
                    setCopy({
                        engineType: item.engineType,
                        source: item.sourceNodeId ?? '',
                        model: item.model ?? '',
                        opId: item.opId
                    })
                }
            />

            {groups.length > 0 && (
                <TextInput
                    placeholder="Filter by model, quantization or node"
                    aria-label="Filter models"
                    value={query}
                    onValueChange={setQuery}
                    slotStart={<Search className="text-subtle-color" />}
                />
            )}

            {groups.length === 0 ? (
                <div className="pair-paper px-6 py-10">
                    <Stack gap="3" align="center" className="text-center">
                        <Text kind="title/sm">No models yet</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color max-w-120">
                            Models appear here once a node reports its downloaded models. Find one
                            to download, or add a GGUF file you already have.
                        </Text>
                        <Flex gap="2">
                            <Button kind="secondary" onClick={() => setAddingFile(true)}>
                                Add from file
                            </Button>
                            <Button
                                kind="primary"
                                color="brand"
                                onClick={() => setActiveTab('discover')}
                            >
                                Discover models
                            </Button>
                        </Flex>
                    </Stack>
                </div>
            ) : visible.length === 0 ? (
                <Text kind="body/regular/sm" className="text-subtle-color">
                    No models match “{query.trim()}”.
                </Text>
            ) : (
                <Stack gap="3">
                    {visible.map(group => (
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
                    ))}
                </Stack>
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
            {addingFile && (
                <AddModelFileDialog onOpenChange={setAddingFile} onCreateProfile={setProfileFor} />
            )}
            {profileFor && selfId && (
                <ModelSettingsPanel
                    key={profileFor}
                    targetId={selfId}
                    nodeName={nodeNames.get(selfId) ?? 'This PC'}
                    identity={{
                        engine: engineManagerName(SOURCE_MODELS_ENGINE_TYPE),
                        model: profileFor
                    }}
                    open
                    onOpenChange={open => {
                        if (!open) setProfileFor(null)
                    }}
                />
            )}
            <ModelDeleteConfirmation deletion={deletion} />
        </Stack>
    )
}
