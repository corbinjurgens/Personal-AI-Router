// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import {
    FormField,
    ModalContent,
    ModalDialog,
    ModalRoot,
    Select,
    Stack,
    Tabs
} from '@nvidia/foundations-react-core'
import type { NodeItem } from '@/shared/types/nodes'
import { DialogHeader } from '@/ui/components/DialogHeader'
import { ModelHubContent } from '@/ui/components/ModelHub/ModelHubContent'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import type { ClusterModelLocation } from '@/ui/utils/group-cluster-models'
import { RepoQuantPicker } from './RepoQuantPicker'
import { ClusterModelPicker } from './ClusterModelPicker'

const TAB_ITEMS = [
    { value: 'browse', children: 'Browse' },
    { value: 'repo', children: 'From a repo' },
    { value: 'cluster', children: 'In your cluster' }
]

/** Download a model to a chosen node from the catalog, a pasted repo, or another node. */
export function ModelDownloadDialog({
    open,
    onOpenChange,
    nodes,
    onCopy
}: {
    open: boolean
    onOpenChange: (open: boolean) => void
    nodes: readonly NodeItem[]
    /** Copy a model another node holds to the target node. */
    onCopy: (source: ClusterModelLocation, targetNodeId: string) => void
}) {
    const selfId = useConnectionStore(state => state.selfId)
    const [chosen, setChosen] = useState('')
    const [tab, setTab] = useState('browse')
    const target =
        nodes.find(node => node.id === chosen)?.id ??
        nodes.find(node => node.id === selfId)?.id ??
        nodes[0]?.id ??
        ''
    const downloaded = useEngineModelsStore(
        state => state.models.get(`${target}:llama-cpp`)?.models
    )
    const nodeNames = new Map(nodes.map(node => [node.id, node.name]))
    const download = (name: string) => {
        if (!target) return
        window.pairApi.engines.pullModel('llama-cpp', target, name)
        onOpenChange(false)
    }

    return (
        <ModalRoot open={open} onOpenChange={onOpenChange} hideCloseButton>
            <ModalDialog>
                <ModalContent
                    className="no-drag-elements model-hub-modal"
                    style={{
                        width: 'min(760px, calc(100vw - 96px))',
                        maxWidth: 'calc(100vw - 32px)',
                        maxHeight: 'calc(100vh - 32px)'
                    }}
                >
                    <DialogHeader
                        className="no-drag-elements"
                        onClose={() => onOpenChange(false)}
                        divider={false}
                    >
                        Download model
                    </DialogHeader>
                    <Stack
                        gap="4"
                        className="min-h-0 pt-2"
                        style={{ height: 'min(560px, calc(100vh - 180px))' }}
                    >
                        <FormField slotLabel="Download to">
                            <Select
                                aria-label="Download to"
                                value={target}
                                items={nodes.map(node => ({
                                    value: node.id,
                                    children:
                                        node.id === selfId ? `${node.name} (this PC)` : node.name
                                }))}
                                onValueChange={(value: string) => setChosen(value)}
                            />
                        </FormField>
                        <Tabs
                            value={tab}
                            onValueChange={value => setTab(String(value))}
                            items={TAB_ITEMS}
                            className="shrink-0"
                        />
                        {tab === 'browse' && (
                            <ModelHubContent
                                key={target}
                                engine="llama-cpp"
                                multiple={false}
                                onSubmit={entries => {
                                    const name = entries[0]?.name
                                    if (name) download(name)
                                }}
                                downloadedModels={downloaded}
                            />
                        )}
                        {tab === 'repo' && (
                            <RepoQuantPicker
                                downloadedModels={downloaded ?? []}
                                onDownload={download}
                            />
                        )}
                        {tab === 'cluster' && (
                            <ClusterModelPicker
                                targetNodeId={target}
                                nodeNames={nodeNames}
                                onCopy={source => onCopy(source, target)}
                            />
                        )}
                    </Stack>
                </ModalContent>
            </ModalDialog>
        </ModalRoot>
    )
}
