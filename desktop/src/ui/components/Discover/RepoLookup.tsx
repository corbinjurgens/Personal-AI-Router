// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import { Button, Flex, FormField, Stack, Text, TextInput } from '@nvidia/foundations-react-core'
import type { EngineCatalogFilesResponse } from '@/shared/types/engine-api'
import type { ModelItem } from '@/shared/types/engines'
import type { ClusterModelHolders } from '@/ui/hooks/useClusterModelHolders'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { Monogram } from '@/ui/components/Page/PagePrimitives'
import { hubModelFacts } from '@/ui/utils/hub-model-facts'
import getErrorString from '@/shared/utils/get-error-string'
import { readableServiceError } from '@/ui/utils/readable-service-error'
import { QuantizationList } from './QuantizationList'

/** Paste a Hugging Face repository or link, then download one of its quantizations. */
export function RepoLookup({
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
    const [text, setText] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [listing, setListing] = useState<EngineCatalogFilesResponse | null>(null)

    const find = async () => {
        const repo = text.trim()
        if (!repo || loading) return
        setLoading(true)
        setError('')
        try {
            setListing(await window.pairApi.engines.catalogFiles('llama-cpp', repo))
        } catch (err) {
            setListing(null)
            setError(readableServiceError(getErrorString(err)))
        } finally {
            setLoading(false)
        }
    }
    const facts = listing ? hubModelFacts(listing.repo) : undefined

    return (
        <section className="pair-paper p-5 min-h-0 overflow-y-auto" aria-label="Paste a link">
            <Stack gap="5" className="max-w-200">
                <form
                    onSubmit={event => {
                        event.preventDefault()
                        void find()
                    }}
                >
                    <Flex align="end" gap="2">
                        <FormField
                            slotLabel="Hugging Face repo or link"
                            slotHelp="Any open GGUF repository works, not only the ones in the catalog."
                            className="flex-1 min-w-0"
                        >
                            <TextInput
                                aria-label="Hugging Face repo or link"
                                placeholder="owner/model-GGUF or https://huggingface.co/..."
                                value={text}
                                onValueChange={setText}
                            />
                        </FormField>
                        <Button
                            type="submit"
                            kind="primary"
                            color="brand"
                            disabled={loading || text.trim() === ''}
                        >
                            {loading && (
                                <span
                                    className="spinner-element-small"
                                    role="status"
                                    aria-label=""
                                />
                            )}
                            Find files
                        </Button>
                    </Flex>
                </form>
                {error && <InlineErrorBanner message={error} />}
                {listing && facts && (
                    <Stack gap="3">
                        <Flex align="center" gap="3">
                            <Monogram name={facts.title} />
                            <Stack gap="0" className="min-w-0">
                                <Text kind="body/semibold/md" className="break-all">
                                    {facts.title}
                                </Text>
                                <Text
                                    kind="body/regular/sm"
                                    className="text-subtle-color break-all"
                                >
                                    {listing.repo}
                                </Text>
                            </Stack>
                        </Flex>
                        {listing.files.length === 0 ? (
                            <Text kind="body/regular/sm" className="text-subtle-color">
                                This repository has no GGUF model files PAIR can download.
                            </Text>
                        ) : (
                            <QuantizationList
                                files={listing.files}
                                targetNodeId={targetNodeId}
                                targetName={targetName}
                                downloadedModels={downloadedModels}
                                holders={quantization =>
                                    holders.forQuantization(listing.repo, quantization)
                                }
                                onDownload={onDownload}
                            />
                        )}
                    </Stack>
                )}
            </Stack>
        </section>
    )
}
