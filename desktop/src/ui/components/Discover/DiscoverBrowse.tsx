// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { memo, useMemo, useState } from 'react'
import { Badge, Flex, Select, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { ModelItem } from '@/shared/types/engines'
import type { ModelEntry } from '@/ui/types/model-hub'
import type { ClusterModelHolders } from '@/ui/hooks/useClusterModelHolders'
import { useModelHubSearch } from '@/ui/hooks/useModelHubSearch'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { Monogram } from '@/ui/components/Page/PagePrimitives'
import { Download, Favorite, Search } from '@/ui/components/icons'
import { hubModelFacts } from '@/ui/utils/hub-model-facts'
import { isHubEntryDownloaded } from '@/ui/utils/match-downloaded-model'
import { formatCompactCount, formatRelativeTime } from '@/ui/utils/formatters'
import { ModelDetail } from './ModelDetail'

type BrowseSort = 'popular' | 'liked' | 'recent' | 'name'

const SORT_ITEMS: { value: BrowseSort; children: string }[] = [
    { value: 'popular', children: 'Most downloaded' },
    { value: 'liked', children: 'Most liked' },
    { value: 'recent', children: 'Recently updated' },
    { value: 'name', children: 'Name' }
]

function isBrowseSort(value: string): value is BrowseSort {
    return SORT_ITEMS.some(item => item.value === value)
}

function sortModels(models: readonly ModelEntry[], sort: BrowseSort): ModelEntry[] {
    const copy = [...models]
    switch (sort) {
        case 'popular':
            return copy.sort((a, b) => b.downloads - a.downloads)
        case 'liked':
            return copy.sort((a, b) => b.likes - a.likes)
        case 'recent':
            return copy.sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime())
        case 'name':
            return copy.sort((a, b) =>
                hubModelFacts(a.name).title.localeCompare(hubModelFacts(b.name).title)
            )
    }
}

const CatalogRow = memo(function CatalogRow({
    model,
    selected,
    onTarget,
    inCluster,
    onSelect
}: {
    model: ModelEntry
    selected: boolean
    onTarget: boolean
    inCluster: boolean
    onSelect: (id: string) => void
}) {
    const facts = hubModelFacts(model.name, model.tags)
    return (
        <li>
            <button
                type="button"
                className="pair-list-item flex items-start gap-3 px-3 py-2.5 cursor-pointer"
                aria-current={selected}
                onClick={() => onSelect(model.id)}
            >
                <Monogram name={facts.title} size={34} />
                <Stack gap="1" className="min-w-0 grow">
                    <Flex align="center" gap="2" className="min-w-0">
                        <Text kind="body/semibold/sm" className="truncate" title={facts.repo}>
                            {facts.title}
                        </Text>
                        {facts.parameters && (
                            <Text kind="body/regular/xs" className="text-subtle-color shrink-0">
                                {facts.parameters}
                            </Text>
                        )}
                    </Flex>
                    <Flex align="center" gap="3" className="text-subtle-color min-w-0">
                        <Text kind="body/regular/xs" className="truncate">
                            {model.author}
                        </Text>
                        <Flex align="center" gap="1" className="shrink-0">
                            <Download style={{ fontSize: 12 }} />
                            <Text kind="body/regular/xs">
                                {formatCompactCount(model.downloads)}
                            </Text>
                        </Flex>
                        <Flex align="center" gap="1" className="shrink-0">
                            <Favorite style={{ fontSize: 12 }} />
                            <Text kind="body/regular/xs">{formatCompactCount(model.likes)}</Text>
                        </Flex>
                        <Text kind="body/regular/xs" className="shrink-0 hidden xl:inline">
                            {formatRelativeTime(model.updatedAt.getTime())}
                        </Text>
                    </Flex>
                    {(onTarget || inCluster) && (
                        <Flex gap="1">
                            {onTarget ? (
                                <Badge color="green">Downloaded</Badge>
                            ) : (
                                <Badge color="blue">In your cluster</Badge>
                            )}
                        </Flex>
                    )}
                </Stack>
            </button>
        </li>
    )
})

/** Search and browse the Hugging Face catalog, with a detail panel for the chosen model. */
export function DiscoverBrowse({
    targetNodeId,
    targetName,
    downloadedModels,
    holders,
    onDownload
}: {
    targetNodeId: string
    targetName: string
    downloadedModels: readonly ModelItem[]
    holders: ClusterModelHolders
    onDownload: (name: string) => void
}) {
    const hub = useModelHubSearch('llama-cpp')
    const [sort, setSort] = useState<BrowseSort>('popular')
    const [selectedId, setSelectedId] = useState<string | null>(null)
    const sorted = useMemo(() => sortModels(hub.models, sort), [hub.models, sort])
    const selected = sorted.find(model => model.id === selectedId) ?? sorted[0]

    return (
        <div className="grid min-h-0 grow gap-4 grid-cols-1 md:grid-cols-[minmax(280px,2fr)_3fr]">
            <section
                className="pair-paper flex flex-col min-h-0 min-w-0 overflow-hidden"
                aria-label="Catalog"
            >
                <Stack gap="2" className="p-3 pb-2">
                    <form
                        onSubmit={event => {
                            event.preventDefault()
                            void hub.search(hub.query)
                        }}
                    >
                        <TextInput
                            aria-label="Search models"
                            placeholder="Search Hugging Face, press Enter"
                            value={hub.query}
                            onValueChange={hub.changeQuery}
                            slotStart={<Search className="text-subtle-color" />}
                            slotEnd={
                                hub.searching ? (
                                    <span className="spinner-element-small" role="status" />
                                ) : undefined
                            }
                        />
                    </form>
                    <Flex align="center" justify="between" gap="2">
                        <Text
                            kind="body/regular/xs"
                            className="text-subtle-color whitespace-nowrap"
                        >
                            {hub.loading
                                ? 'Loading catalog'
                                : `${sorted.length} ${sorted.length === 1 ? 'model' : 'models'}`}
                        </Text>
                        <Select
                            aria-label="Sort models"
                            size="small"
                            value={sort}
                            items={SORT_ITEMS}
                            onValueChange={(value: string) => {
                                if (isBrowseSort(value)) setSort(value)
                            }}
                        />
                    </Flex>
                    {hub.errors.map(error => (
                        <InlineErrorBanner
                            key={error.id}
                            message={error.message}
                            onClose={() => hub.dismissError(error.id)}
                        />
                    ))}
                </Stack>
                <div className="model-hub-scroll overflow-y-auto min-h-0 grow px-2 pb-2">
                    {hub.loading ? (
                        <Flex align="center" justify="center" gap="2" className="p-6">
                            <span className="spinner-element" role="status" aria-label="" />
                            <Text kind="body/regular/sm">Loading models</Text>
                        </Flex>
                    ) : sorted.length === 0 ? (
                        <Stack gap="1" className="p-6 text-center">
                            <Text kind="body/semibold/sm">No models found</Text>
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                Press Enter to search all of Hugging Face, or paste a link instead.
                            </Text>
                        </Stack>
                    ) : (
                        <ul className="flex flex-col gap-0.5" aria-label="Models">
                            {sorted.map(model => (
                                <CatalogRow
                                    key={model.id}
                                    model={model}
                                    selected={model.id === selected?.id}
                                    onTarget={isHubEntryDownloaded(
                                        'llama-cpp',
                                        model,
                                        downloadedModels
                                    )}
                                    inCluster={
                                        holders.forRepo(hubModelFacts(model.name).repo).length > 0
                                    }
                                    onSelect={setSelectedId}
                                />
                            ))}
                        </ul>
                    )}
                </div>
            </section>
            <section
                className="pair-paper p-5 min-h-0 min-w-0 overflow-y-auto model-hub-scroll"
                aria-label="Model details"
            >
                {selected ? (
                    <ModelDetail
                        key={selected.id}
                        model={selected}
                        targetNodeId={targetNodeId}
                        targetName={targetName}
                        downloadedModels={downloadedModels}
                        holders={holders}
                        onDownload={onDownload}
                    />
                ) : (
                    <Stack gap="2" align="center" justify="center" className="h-full p-8">
                        <Text kind="body/semibold/md">Pick a model</Text>
                        <Text kind="body/regular/sm" className="text-subtle-color text-center">
                            Its details and every quantization you can download appear here.
                        </Text>
                    </Stack>
                )}
            </section>
        </div>
    )
}
