// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react'
import { Button, Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core'
import type { EngineCatalogFilesResponse } from '@/shared/types/engine-api'
import type { ModelItem } from '@/shared/types/engines'
import type { ModelEntry } from '@/ui/types/model-hub'
import type { ClusterModelHolders } from '@/ui/hooks/useClusterModelHolders'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { Monogram, StatTile } from '@/ui/components/Page/PagePrimitives'
import { OpenInNew } from '@/ui/components/icons'
import { hubModelFacts } from '@/ui/utils/hub-model-facts'
import { formatCompactCount, formatRelativeTime } from '@/ui/utils/formatters'
import getErrorString from '@/shared/utils/get-error-string'
import { readableServiceError } from '@/ui/utils/readable-service-error'
import { QuantizationList } from './QuantizationList'

function Fact({ label, value }: { label: string; value?: string }) {
    if (!value) return null
    return (
        <div className="min-w-0">
            <dt className="pair-eyebrow">{label}</dt>
            <dd className="m-0 mt-0.5">
                <Text kind="body/regular/sm" className="break-words">
                    {value}
                </Text>
            </dd>
        </div>
    )
}

/** Everything known about one catalog model, and its quantizations to download. */
export function ModelDetail({
    model,
    targetNodeId,
    targetName,
    downloadedModels,
    holders,
    onDownload
}: {
    model: ModelEntry
    targetNodeId: string
    targetName: string
    downloadedModels: readonly ModelItem[]
    holders: ClusterModelHolders
    onDownload: (name: string) => void
}) {
    const facts = hubModelFacts(model.name, model.tags, model.parameterSize)
    const [listing, setListing] = useState<EngineCatalogFilesResponse | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState('')

    useEffect(() => {
        let active = true
        setListing(null)
        setError('')
        setLoading(true)
        window.pairApi.engines
            .catalogFiles('llama-cpp', facts.repo)
            .then(result => {
                if (active) setListing(result)
            })
            .catch(err => {
                if (active) setError(readableServiceError(getErrorString(err)))
            })
            .finally(() => {
                if (active) setLoading(false)
            })
        return () => {
            active = false
        }
    }, [facts.repo])

    const parameters =
        facts.parameters && facts.activeParameters
            ? `${facts.parameters} (${facts.activeParameters} active)`
            : facts.parameters
    const languages =
        facts.languages.length > 6
            ? `${facts.languages.slice(0, 6).join(', ')} and ${facts.languages.length - 6} more`
            : facts.languages.join(', ')

    return (
        <Stack gap="5" className="min-w-0" aria-label={`${facts.repo} details`}>
            <Flex align="start" gap="3">
                <Monogram name={facts.title} size={48} />
                <Stack gap="1" className="min-w-0 grow">
                    <Text kind="title/sm" className="break-words">
                        {facts.title}
                    </Text>
                    <Text kind="body/regular/sm" className="text-subtle-color break-all">
                        {facts.repo}
                    </Text>
                </Stack>
                {model.url && (
                    <Button
                        kind="tertiary"
                        size="small"
                        className="shrink-0"
                        onClick={() => window.open(model.url, '_blank')}
                    >
                        Hugging Face
                        <OpenInNew />
                    </Button>
                )}
            </Flex>

            <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
                <StatTile label="Downloads" value={formatCompactCount(model.downloads)} />
                <StatTile label="Likes" value={formatCompactCount(model.likes)} />
                <StatTile label="Parameters" value={parameters ?? 'Not stated'} />
                <StatTile label="Updated" value={formatRelativeTime(model.updatedAt.getTime())} />
            </div>

            {facts.capabilities.length > 0 && (
                <Flex gap="2" wrap="wrap" aria-label="Capabilities">
                    {facts.capabilities.map(label => (
                        <Tag key={label} color="gray" density="compact" readOnly>
                            {label}
                        </Tag>
                    ))}
                </Flex>
            )}

            {(facts.baseModel || facts.license || languages) && (
                <dl className="grid grid-cols-1 gap-x-6 gap-y-3 m-0 sm:grid-cols-2">
                    <Fact label="Based on" value={facts.baseModel} />
                    <Fact label="Licence" value={facts.license} />
                    <Fact label="Publisher" value={model.author} />
                    <Fact label="Languages" value={languages} />
                </dl>
            )}

            <Stack gap="2">
                <Flex align="baseline" justify="between" gap="2" wrap="wrap">
                    <Text kind="body/semibold/md" className="whitespace-nowrap">
                        Choose a quantization
                    </Text>
                    <Text kind="body/regular/xs" className="text-subtle-color">
                        Smaller files use less memory; larger ones keep more quality.
                    </Text>
                </Flex>
                {loading ? (
                    <Flex align="center" gap="2" className="py-4" role="status">
                        <span className="spinner-element-small" aria-hidden="true" />
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            Reading the repository&apos;s files
                        </Text>
                    </Flex>
                ) : error ? (
                    <InlineErrorBanner message={error} />
                ) : listing && listing.files.length > 0 ? (
                    <QuantizationList
                        files={listing.files}
                        targetNodeId={targetNodeId}
                        targetName={targetName}
                        downloadedModels={downloadedModels}
                        holders={quantization => holders.forQuantization(facts.repo, quantization)}
                        recommended={facts.quantization}
                        onDownload={onDownload}
                    />
                ) : (
                    <Text kind="body/regular/sm" className="text-subtle-color">
                        This repository has no GGUF model files PAIR can download.
                    </Text>
                )}
            </Stack>
        </Stack>
    )
}
