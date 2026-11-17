// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react'
import { Flex, SegmentedControl, Select, Stack, Text } from '@nvidia/foundations-react-core'
import type { EngineType } from '@/shared/types/engines'
import { useOverviewNodes } from '@/ui/hooks/useOverviewNodes'
import { useClusterModelHolders } from '@/ui/hooks/useClusterModelHolders'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { ModelCopyModal } from '@/ui/components/ModelManager/ModelCopyModal'
import { PageHeading } from '@/ui/components/Page/PagePrimitives'
import { TransfersPanel } from '@/ui/components/Models/TransfersPanel'
import { DiscoverBrowse } from './DiscoverBrowse'
import { RepoLookup } from './RepoLookup'
import { ClusterCopyList } from './ClusterCopyList'

type DiscoverSource = 'browse' | 'link' | 'cluster'

const SOURCE_ITEMS: { value: DiscoverSource; children: string }[] = [
    { value: 'browse', children: 'Hugging Face' },
    { value: 'link', children: 'Paste a link' },
    { value: 'cluster', children: 'Your cluster' }
]

function isDiscoverSource(value: string): value is DiscoverSource {
    return SOURCE_ITEMS.some(item => item.value === value)
}

const NO_MODELS: never[] = []

/** Find models to download, or copy from another node, onto a chosen node. */
export default function DiscoverView() {
    const nodes = useOverviewNodes()
    const selfId = useConnectionStore(state => state.selfId)
    const nodeNames = useMemo(() => new Map(nodes.map(node => [node.id, node.name])), [nodes])
    const holders = useClusterModelHolders(nodeNames)
    const [chosen, setChosen] = useState('')
    const [source, setSource] = useState<DiscoverSource>('browse')
    const [copy, setCopy] = useState<{
        engineType: EngineType
        source: string
        model: string
        opId?: string
        destination?: string
    } | null>(null)
    const target =
        nodes.find(node => node.id === chosen)?.id ??
        nodes.find(node => node.id === selfId)?.id ??
        nodes[0]?.id ??
        ''
    const targetName = nodeNames.get(target) ?? target
    const downloaded =
        useEngineModelsStore(state => state.models.get(`${target}:llama-cpp`)?.models) ?? NO_MODELS
    const download = (name: string) => {
        if (target) window.pairApi.engines.pullModel('llama-cpp', target, name)
    }

    return (
        <Stack gap="4" className="px-6 py-5 w-full max-w-360 min-h-0 grow">
            <PageHeading
                title="Discover"
                description="Find a model and download it to any node in your cluster."
                actions={
                    <Flex align="center" gap="2">
                        <Text kind="body/regular/sm" className="text-subtle-color shrink-0">
                            Download to
                        </Text>
                        <Select
                            aria-label="Download to"
                            className="min-w-44"
                            value={target}
                            items={nodes.map(node => ({
                                value: node.id,
                                children: node.id === selfId ? `${node.name} (this PC)` : node.name
                            }))}
                            onValueChange={(value: string) => setChosen(value)}
                        />
                    </Flex>
                }
            />
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
            <SegmentedControl
                aria-label="Where to find models"
                className="self-start"
                value={source}
                items={SOURCE_ITEMS}
                onValueChange={(value: string) => {
                    if (isDiscoverSource(value)) setSource(value)
                }}
            />
            {source === 'browse' && (
                <DiscoverBrowse
                    targetNodeId={target}
                    targetName={targetName}
                    downloadedModels={downloaded}
                    holders={holders}
                    onDownload={download}
                />
            )}
            {source === 'link' && (
                <RepoLookup
                    targetNodeId={target}
                    targetName={targetName}
                    downloadedModels={downloaded}
                    holders={holders}
                    onDownload={download}
                />
            )}
            {source === 'cluster' && (
                <ClusterCopyList
                    targetNodeId={target}
                    nodeNames={nodeNames}
                    onCopy={location =>
                        setCopy({
                            engineType: location.engineType,
                            source: location.nodeId,
                            model: location.model.name,
                            destination: target
                        })
                    }
                />
            )}
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
        </Stack>
    )
}
