// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react'
import {
    Button,
    Flex,
    FormField,
    Stack,
    Tag,
    Text,
    TextInput
} from '@nvidia/foundations-react-core'
import type { EngineCatalogFilesResponse } from '@/shared/types/engine-api'
import type { ModelItem } from '@/shared/types/engines'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { formatBytes } from '@/ui/utils/formatters'
import { isHubEntryDownloaded } from '@/ui/utils/match-downloaded-model'
import getErrorString from '@/shared/utils/get-error-string'

/** Paste a Hugging Face repository or link, then download one of its quantizations. */
export function RepoQuantPicker({
    downloadedModels,
    onDownload
}: {
    /** The target node's llama.cpp models, to mark files it already holds. */
    downloadedModels: readonly ModelItem[]
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
            setError(getErrorString(err))
        } finally {
            setLoading(false)
        }
    }

    const downloaded = (name: string) =>
        isHubEntryDownloaded(
            'llama-cpp',
            { id: name, name, author: '', url: '', updatedAt: new Date(0) },
            downloadedModels
        )

    return (
        <Stack gap="4" className="min-h-0">
            <form
                onSubmit={event => {
                    event.preventDefault()
                    void find()
                }}
            >
                <Flex align="end" gap="2">
                    <FormField slotLabel="Hugging Face repo or link" className="flex-1 min-w-0">
                        <TextInput
                            aria-label="Hugging Face repo or link"
                            placeholder="owner/model-GGUF or https://huggingface.co/..."
                            value={text}
                            onValueChange={setText}
                        />
                    </FormField>
                    <Button type="submit" kind="secondary" disabled={loading || text.trim() === ''}>
                        {loading && (
                            <span className="spinner-element-small" role="status" aria-label="" />
                        )}
                        Find files
                    </Button>
                </Flex>
            </form>
            {error && <InlineErrorBanner message={error} />}
            {listing && (
                <Stack gap="2" className="min-h-0 overflow-y-auto">
                    <Text kind="body/semibold/md" className="break-all">
                        {listing.repo}
                    </Text>
                    {listing.files.length === 0 ? (
                        <Text kind="body/regular/sm" className="text-subtle-color">
                            This repository has no GGUF model files PAIR can download.
                        </Text>
                    ) : (
                        listing.files.map(file => (
                            <Flex
                                key={file.name}
                                align="center"
                                justify="between"
                                gap="3"
                                className="rounded-md bg-[#ffffff08] px-3 py-2"
                            >
                                <Flex align="center" gap="2" className="min-w-0">
                                    <Tag color="green" density="compact" readOnly>
                                        {file.quantization}
                                    </Tag>
                                    <Text kind="body/regular/sm" className="text-subtle-color">
                                        {formatBytes(file.sizeBytes, 1)}
                                        {file.shards > 1 ? ` · ${file.shards} parts` : ''}
                                    </Text>
                                </Flex>
                                {downloaded(file.name) ? (
                                    <Button size="small" kind="tertiary" disabled>
                                        Downloaded
                                    </Button>
                                ) : (
                                    <Button
                                        size="small"
                                        kind="primary"
                                        color="brand"
                                        aria-label={`Download ${file.quantization}`}
                                        onClick={() => onDownload(file.name)}
                                    >
                                        Download
                                    </Button>
                                )}
                            </Flex>
                        ))
                    )}
                </Stack>
            )}
        </Stack>
    )
}
