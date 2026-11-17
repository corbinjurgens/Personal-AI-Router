// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Stack } from '@nvidia/foundations-react-core'
import { ModelHubSearchBar } from './ModelHubSearchBar'
import { ModelHubActions } from './ModelHubActions'
import { InlineErrorBanner } from '@/ui/components/InlineErrorBanner'
import { ModelEntry, SortState } from '@/ui/types/model-hub'
import { ModelHubList } from './ModelHubList'
import { resolveStoredSort, writeStoredSort } from '@/ui/utils/model-hub-content-storage'
import { isHubEntryDownloaded } from '@/ui/utils/match-downloaded-model'
import { EngineType } from '@/shared/types/engines'
import type { ModelItem } from '@/ui/types/engine-info'
import { useModelHubSearch } from '@/ui/hooks/useModelHubSearch'

type ModelHubContentProps = {
    engine: EngineType | null
    multiple: boolean
    onSubmit: (models: ModelEntry[]) => void
    /** Current ready models are hidden; failed or unavailable content remains available for explicit retry. */
    downloadedModels?: readonly ModelItem[]
}

const SORT_DEFAULT_STATE: SortState = {
    sort: 'lastModified',
    sortDescending: false,
    allDirections: {
        lastModified: false,
        name: false,
        size: false
    }
}

export const ModelHubContent = ({
    engine,
    multiple,
    onSubmit,
    downloadedModels
}: ModelHubContentProps) => {
    const [disabledListClick, setDisabledListClick] = useState(false)
    const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set())
    const [sort, setSort] = useState<SortState>(() =>
        engine ? resolveStoredSort(engine, SORT_DEFAULT_STATE) : { ...SORT_DEFAULT_STATE }
    )
    const {
        adapterType,
        loading,
        models: queryFiltered,
        query,
        searching,
        errors,
        changeQuery,
        search,
        dismissError
    } = useModelHubSearch(engine)

    useEffect(() => {
        setSelectedModels(new Set())
        setSort(engine ? resolveStoredSort(engine, SORT_DEFAULT_STATE) : { ...SORT_DEFAULT_STATE })
    }, [engine])

    const visibleModels = useMemo(() => {
        if (!engine || !downloadedModels || downloadedModels.length === 0) return queryFiltered
        return queryFiltered.filter(m => !isHubEntryDownloaded(adapterType, m, downloadedModels))
    }, [engine, adapterType, queryFiltered, downloadedModels])

    const handleSortPersist = useCallback(
        (next: SortState) => {
            setSort(next)
            if (engine) {
                writeStoredSort(engine, next)
            }
        },
        [engine]
    )

    const modelsRef = useRef(queryFiltered)
    modelsRef.current = queryFiltered

    const handleSubmit = useCallback(
        (ids: string[]) => {
            const modelSelected: ModelEntry[] = []
            for (const id of ids) {
                const found = modelsRef.current.find(model => model.id === id)
                if (found !== undefined) {
                    modelSelected.push(found)
                }
            }

            if (modelSelected.length > 0) {
                onSubmit(modelSelected)
            }
        },
        [onSubmit]
    )

    const onToggleSelectModel = useCallback((modelId: string) => {
        startTransition(() => {
            setSelectedModels(prev => {
                const newSet = new Set(prev)
                newSet.has(modelId) ? newSet.delete(modelId) : newSet.add(modelId)
                return newSet
            })
        })
    }, [])

    const handleListToggle = useCallback(
        (modelId: string) => {
            if (multiple) {
                onToggleSelectModel(modelId)
            } else {
                handleSubmit([modelId])
            }
        },
        [multiple, onToggleSelectModel, handleSubmit]
    )

    return (
        <Stack gap="0" className="overflow-hidden min-h-0 min-w-0 grow w-full">
            <Stack gap="4" className="overflow-hidden min-h-0 min-w-0 grow w-full">
                <Stack gap="1">
                    {errors.length > 0 && (
                        <Stack gap="1" className="mb-4">
                            {errors.map(error => (
                                <InlineErrorBanner
                                    key={error.id}
                                    message={error.message}
                                    onClose={() => dismissError(error.id)}
                                />
                            ))}
                        </Stack>
                    )}
                    <ModelHubSearchBar
                        onMenuOpenChange={setDisabledListClick}
                        query={query}
                        searching={searching}
                        onQueryChange={changeQuery}
                        onSearch={search}
                        sort={sort}
                        onSort={handleSortPersist}
                    />
                </Stack>
                <ModelHubList
                    className="grow min-h-0"
                    disabled={disabledListClick}
                    multiple={multiple}
                    loading={loading}
                    models={visibleModels}
                    selectedModels={selectedModels}
                    onToggleSelectModel={handleListToggle}
                    sort={sort}
                />

                {!!multiple && (
                    <ModelHubActions
                        onSubmit={() => handleSubmit(Array.from(selectedModels))}
                        disabled={selectedModels.size === 0 || loading}
                        count={selectedModels.size}
                    />
                )}
            </Stack>
        </Stack>
    )
}
