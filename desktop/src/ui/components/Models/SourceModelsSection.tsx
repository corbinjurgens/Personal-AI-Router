// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import { Badge, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { SourceModel } from '@/shared/types/engine-api'
import type { NodeItem } from '@/shared/types/nodes'
import { ConfirmModal } from '@/ui/components/ConfirmModal'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { LocalBadge } from '@/ui/components/LocalBadge'
import ModelSettingsPanel from '@/ui/components/ModelSettings/ModelSettingsPanel'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { SOURCE_MODELS_ENGINE, useSourceModelsStore } from '@/ui/stores/source-models.store'
import { formatBytes } from '@/ui/utils/formatters'
import getErrorString from '@/shared/utils/get-error-string'

const SOURCE_LABELS: Record<SourceModel['source'], string> = {
    ollama: 'Ollama',
    lmstudio: 'LM Studio',
    local: 'Linked file'
}

/** Whether a source model matches the Models tab filter text. */
function sourceModelMatches(model: SourceModel, nodeName: string, needle: string): boolean {
    return [model.id, model.name, nodeName, SOURCE_LABELS[model.source]].some(text =>
        text.toLowerCase().includes(needle)
    )
}

/** Models each node serves in place from Ollama, LM Studio or linked files. */
export function SourceModelsSection({
    nodes,
    needle
}: {
    nodes: readonly NodeItem[]
    /** Lower-cased filter text from the Models tab header; empty shows all. */
    needle: string
}) {
    const entries = useSourceModelsStore(state => state.entries)
    const refresh = useSourceModelsStore(state => state.refresh)
    const selfId = useConnectionStore(state => state.selfId)
    const [configure, setConfigure] = useState<{ node: NodeItem; id: string } | null>(null)
    const [unlinking, setUnlinking] = useState<SourceModel | null>(null)
    const [unlinkError, setUnlinkError] = useState('')

    const cards = nodes.flatMap(node => {
        const entry = entries.get(node.id)
        if (!entry) return []
        const models = needle
            ? entry.models.filter(model => sourceModelMatches(model, node.name, needle))
            : entry.models
        if (models.length === 0 && !entry.error) return []
        return [{ node, models, error: entry.error }]
    })
    if (cards.length === 0 && !configure) return null

    const unlink = async (model: SourceModel) => {
        setUnlinkError('')
        try {
            await window.pairApi.engines.unlinkModel(SOURCE_MODELS_ENGINE, model.id)
        } catch (err) {
            setUnlinkError(getErrorString(err))
        }
        if (selfId) await refresh(selfId)
    }

    return (
        <Stack gap="3">
            <Stack gap="1">
                <Text kind="body/semibold/md">Outside the model store</Text>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Models in Ollama or LM Studio folders, and files you linked. PAIR serves them
                    where they are.
                </Text>
            </Stack>
            {unlinkError && (
                <InlineErrorBanner message={unlinkError} onClose={() => setUnlinkError('')} />
            )}
            {cards.map(({ node, models, error }) => (
                <section
                    key={node.id}
                    className="pair-paper p-4"
                    aria-label={`${node.name} outside the model store`}
                >
                    <Stack gap="2">
                        <Flex align="center" gap="2">
                            <Text kind="body/semibold/sm">{node.name}</Text>
                            {node.id === selfId && <LocalBadge />}
                        </Flex>
                        {error && (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                {error}
                            </Text>
                        )}
                        {models.map(model => (
                            <Flex
                                key={model.id}
                                gap="3"
                                align="center"
                                justify="between"
                                className="py-2 border-t border-[#ffffff0f] first:border-t-0"
                            >
                                <Flex gap="2" align="center" className="min-w-0">
                                    <Badge color={model.source === 'local' ? 'blue' : 'gray'}>
                                        {SOURCE_LABELS[model.source]}
                                    </Badge>
                                    <Text
                                        kind="body/regular/sm"
                                        className="truncate"
                                        title={model.id}
                                    >
                                        {model.name}
                                    </Text>
                                    <Text kind="body/regular/xs" className="text-subtle2 shrink-0">
                                        {formatBytes(model.sizeBytes, 1)}
                                        {model.shards > 1 ? ` · ${model.shards} parts` : ''}
                                    </Text>
                                </Flex>
                                <Flex gap="2" className="shrink-0">
                                    <Button
                                        size="small"
                                        kind="secondary"
                                        aria-label={`Create profile for ${model.name}`}
                                        onClick={() => setConfigure({ node, id: model.id })}
                                    >
                                        Create profile
                                    </Button>
                                    {model.source === 'local' && node.id === selfId && (
                                        <Button
                                            size="small"
                                            kind="tertiary"
                                            aria-label={`Unlink ${model.name}`}
                                            onClick={() => setUnlinking(model)}
                                        >
                                            Unlink
                                        </Button>
                                    )}
                                </Flex>
                            </Flex>
                        ))}
                    </Stack>
                </section>
            ))}
            {configure && (
                <ModelSettingsPanel
                    key={`${configure.node.id}:${configure.id}`}
                    targetId={configure.node.id}
                    nodeId={configure.node.id === selfId ? undefined : configure.node.id}
                    nodeName={configure.node.name}
                    identity={{ engine: SOURCE_MODELS_ENGINE, model: configure.id }}
                    open
                    onOpenChange={open => {
                        if (!open) setConfigure(null)
                    }}
                />
            )}
            {unlinking && (
                <ConfirmModal
                    open
                    onOpenChange={open => {
                        if (!open) setUnlinking(null)
                    }}
                    title="Unlink file"
                    message={`PAIR stops listing ${unlinking.name}. The file itself is not deleted.`}
                    confirmLabel="Unlink"
                    onConfirm={() => void unlink(unlinking)}
                />
            )}
        </Stack>
    )
}
