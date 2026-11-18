// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import { Badge, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core'
import type { SourceModel } from '@/shared/types/engine-api'
import type { NodeItem } from '@/shared/types/nodes'
import { ConfirmModal } from '@/ui/components/ConfirmModal'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { NodeChip } from '@/ui/components/Page/PagePrimitives'
import ModelSettingsPanel from '@/ui/components/ModelSettings/ModelSettingsPanel'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { SOURCE_MODELS_ENGINE_TYPE, useSourceModelsStore } from '@/ui/stores/source-models.store'
import { formatBytes } from '@/ui/utils/formatters'
import { engineManagerName } from '@/shared/utils/engines'
import getErrorString from '@/shared/utils/get-error-string'
import { readableServiceError } from '@/ui/utils/readable-service-error'

const SOURCE_LABELS: Record<SourceModel['source'], string> = {
    ollama: 'Ollama',
    lmstudio: 'LM Studio',
    local: 'Linked file',
    copy: 'Copied file'
}

const BADGE_COLORS: Record<SourceModel['source'], 'blue' | 'green' | 'gray'> = {
    ollama: 'gray',
    lmstudio: 'gray',
    local: 'blue',
    copy: 'green'
}

/** The store name for a copied source model: its ID without the source, "/" and "@" as "-". */
function storeName(id: string): string {
    return id.replace(/^(ollama|lmstudio)\//, '').replace(/[/@]/g, '-')
}

/** Whether a source model matches the Models tab filter text. */
function sourceModelMatches(model: SourceModel, nodeName: string, needle: string): boolean {
    return [model.id, model.name, nodeName, SOURCE_LABELS[model.source]].some(text =>
        text.toLowerCase().includes(needle)
    )
}

/**
 * Models each node serves in place from Ollama, LM Studio or linked files, and
 * the copied files in its model folder that native inventory does not list.
 */
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
    const forget = useSourceModelsStore(state => state.forget)
    const selfId = useConnectionStore(state => state.selfId)
    const [configure, setConfigure] = useState<{ node: NodeItem; id: string } | null>(null)
    const [unlinking, setUnlinking] = useState<SourceModel | null>(null)
    const [actionError, setActionError] = useState('')
    const [copying, setCopying] = useState<SourceModel | null>(null)
    const [busyCopy, setBusyCopy] = useState('')
    const [copied, setCopied] = useState('')
    const [deleting, setDeleting] = useState<{ node: NodeItem; model: SourceModel } | null>(null)

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
        setActionError('')
        try {
            await window.pairApi.engines.unlinkModel(SOURCE_MODELS_ENGINE_TYPE, model.id)
        } catch (err) {
            setActionError(getErrorString(err))
        }
        if (selfId) await refresh(selfId)
    }

    const copyIntoStore = async (model: SourceModel) => {
        setActionError('')
        setCopied('')
        setBusyCopy(model.id)
        try {
            const res = await window.pairApi.engines.importModel(
                SOURCE_MODELS_ENGINE_TYPE,
                { sourceId: model.id },
                storeName(model.id)
            )
            setCopied(res.id)
            if (selfId) void refresh(selfId)
        } catch (err) {
            setActionError(getErrorString(err))
        } finally {
            setBusyCopy('')
        }
    }

    // The engine command reports a refused or failed delete as an app error.
    const deleteCopy = (node: NodeItem, model: SourceModel) => {
        window.pairApi.engines.deleteModel(SOURCE_MODELS_ENGINE_TYPE, node.id, model.id)
        forget(node.id, model.id)
    }
    const self = nodes.find(node => node.id === selfId)

    return (
        <Stack gap="3">
            <Stack gap="1" className="pt-2">
                <Text kind="title/sm" role="heading" aria-level={2}>
                    Added files and other folders
                </Text>
                <Text kind="body/regular/sm" className="text-subtle-color">
                    Files you copied into PAIR&apos;s model folder, models in Ollama or LM Studio
                    folders, and files you linked. PAIR serves the last two where they are.
                </Text>
            </Stack>
            {copied && self && (
                <InlineErrorBanner
                    severity="info"
                    message={`Copied into PAIR's model folder as ${copied}.`}
                    onClose={() => setCopied('')}
                >
                    <Button
                        size="small"
                        kind="secondary"
                        onClick={() => setConfigure({ node: self, id: copied })}
                    >
                        Create profile
                    </Button>
                </InlineErrorBanner>
            )}
            {actionError && (
                <InlineErrorBanner message={actionError} onClose={() => setActionError('')} />
            )}
            {cards.map(({ node, models, error }) => (
                <section
                    key={node.id}
                    className="pair-paper p-4"
                    aria-label={`${node.name} added files and other folders`}
                >
                    <Stack gap="2">
                        <Flex align="center" justify="between" gap="2">
                            <NodeChip node={node} local={node.id === selfId} />
                            {models.length > 0 && (
                                <Text kind="body/regular/xs" className="text-subtle-color">
                                    {models.length} {models.length === 1 ? 'model' : 'models'}
                                </Text>
                            )}
                        </Flex>
                        {error && (
                            <Text
                                kind="body/regular/sm"
                                className="text-subtle-color"
                                title={error}
                            >
                                {node.status === 'offline'
                                    ? `${node.name} is offline, so its folders can't be read.`
                                    : `Couldn't read ${node.name}'s folders: ${readableServiceError(error)}`}
                            </Text>
                        )}
                        {models.length > 0 && (
                            <ul className="m-0 p-0 list-none">
                                {models.map(model => (
                                    <li
                                        key={model.id}
                                        className="flex items-center justify-between gap-3 py-2 pair-divider-top first:border-t-0"
                                    >
                                        <Flex gap="2" align="center" className="min-w-0">
                                            <Badge color={BADGE_COLORS[model.source]}>
                                                {SOURCE_LABELS[model.source]}
                                            </Badge>
                                            <Text
                                                kind="body/semibold/sm"
                                                className="truncate"
                                                title={model.id}
                                            >
                                                {model.name}
                                            </Text>
                                            <Text
                                                kind="body/regular/xs"
                                                className="text-subtle-color shrink-0"
                                            >
                                                {formatBytes(model.sizeBytes, 1)}
                                                {model.shards > 1 ? ` · ${model.shards} parts` : ''}
                                            </Text>
                                        </Flex>
                                        <Flex gap="1" className="shrink-0">
                                            <Button
                                                size="small"
                                                kind="secondary"
                                                aria-label={`Create profile for ${model.name}`}
                                                onClick={() => setConfigure({ node, id: model.id })}
                                            >
                                                Create profile
                                            </Button>
                                            {(model.source === 'ollama' ||
                                                model.source === 'lmstudio') &&
                                                node.id === selfId && (
                                                    <Button
                                                        size="small"
                                                        kind="tertiary"
                                                        disabled={busyCopy !== ''}
                                                        aria-label={`Copy ${model.name} into store`}
                                                        onClick={() => setCopying(model)}
                                                    >
                                                        {busyCopy === model.id
                                                            ? 'Copying…'
                                                            : 'Copy into store'}
                                                    </Button>
                                                )}
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
                                            {model.source === 'copy' && (
                                                <Button
                                                    size="small"
                                                    kind="tertiary"
                                                    color="danger"
                                                    aria-label={`Delete ${model.name}`}
                                                    onClick={() => setDeleting({ node, model })}
                                                >
                                                    Delete
                                                </Button>
                                            )}
                                        </Flex>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </Stack>
                </section>
            ))}
            {configure && (
                <ModelSettingsPanel
                    key={`${configure.node.id}:${configure.id}`}
                    targetId={configure.node.id}
                    nodeId={configure.node.id === selfId ? undefined : configure.node.id}
                    nodeName={configure.node.name}
                    identity={{
                        engine: engineManagerName(SOURCE_MODELS_ENGINE_TYPE),
                        model: configure.id
                    }}
                    open
                    onOpenChange={open => {
                        if (!open) setConfigure(null)
                    }}
                />
            )}
            {copying && (
                <ConfirmModal
                    open
                    onOpenChange={open => {
                        if (!open) setCopying(null)
                    }}
                    title="Copy into PAIR's model folder"
                    message={`PAIR copies ${copying.name} (${formatBytes(copying.sizeBytes, 1)}) into its own model folder. The original stays where it is.`}
                    confirmLabel="Copy"
                    onConfirm={() => void copyIntoStore(copying)}
                />
            )}
            {deleting && (
                <ConfirmModal
                    open
                    onOpenChange={open => {
                        if (!open) setDeleting(null)
                    }}
                    title="Delete copied file"
                    message={`Delete ${deleting.model.name} (${formatBytes(deleting.model.sizeBytes, 1)}) from PAIR's model folder on ${deleting.node.name}? A model a profile uses cannot be deleted; remove those profiles first.`}
                    confirmLabel="Delete"
                    confirmColor="danger"
                    onConfirm={() => deleteCopy(deleting.node, deleting.model)}
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
