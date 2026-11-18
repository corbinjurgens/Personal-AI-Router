// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ModelEntry } from '@/ui/types/model-hub'
import type { EngineType } from '@/shared/types/engines'
import { engineAdapterType } from '@/shared/utils/engines'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { mergeModelHubResults, searchEngineHub } from '@/ui/utils/model-hub-search'
import getErrorString from '@/shared/utils/get-error-string'

interface ModelHubSearchError {
    id: string
    message: string
}

/**
 * An engine's model catalog: the populated list, filtered locally as the
 * query changes, plus an explicit search of the whole upstream source.
 */
export function useModelHubSearch(engine: EngineType | null) {
    const [loading, setLoading] = useState(true)
    const [allModels, setAllModels] = useState<ModelEntry[]>([])
    const [remoteModels, setRemoteModels] = useState<ModelEntry[]>([])
    const [query, setQuery] = useState('')
    const [searching, setSearching] = useState(false)
    const [errors, setErrors] = useState<ModelHubSearchError[]>([])
    // The built-in engine whose catalogue search and model naming this engine
    // shares: itself, or a variant's family.
    const descriptor = useEngineStatusStore(s => (engine ? s.descriptors.get(engine) : undefined))
    const adapterType = engine ? engineAdapterType(engine, descriptor) : null

    // Per-engine raw-response cache so reopening an engine is instant and
    // keystrokes filter locally instead of re-invoking the search IPC.
    const cacheRef = useRef<Map<EngineType, ModelEntry[]>>(new Map())
    // Monotonic generation guard so a stale in-flight fetch never overwrites
    // results for the engine the user has since switched to.
    const fetchGenRef = useRef(0)
    const searchGenRef = useRef(0)
    const submittedQueryRef = useRef('')

    const addError = useCallback((message: string) => {
        setErrors(prev => [...prev, { id: performance.now().toString(), message }])
    }, [])

    const loadModels = useCallback(
        async (backend: EngineType) => {
            const cached = cacheRef.current.get(backend)
            if (cached) {
                setAllModels(cached)
                setLoading(false)
                return
            }
            const gen = ++fetchGenRef.current
            setLoading(true)
            try {
                const result = await searchEngineHub(backend)
                if (fetchGenRef.current !== gen) return
                cacheRef.current.set(backend, result)
                setAllModels(result)
                setLoading(false)
            } catch (error) {
                if (fetchGenRef.current !== gen) return
                console.warn('Model hub load error', error)
                addError(getErrorString(error))
                setAllModels([])
                setLoading(false)
            }
        },
        [addError]
    )

    useEffect(() => {
        setRemoteModels([])
        setSearching(false)
        setQuery('')
        submittedQueryRef.current = ''
        if (!engine) {
            setAllModels([])
            return
        }
        void loadModels(engine)
        return () => {
            // Invalidate in-flight population or search for the previous engine.
            fetchGenRef.current += 1
            searchGenRef.current += 1
        }
    }, [engine, loadModels])

    const models = useMemo(() => {
        const q = query.toLowerCase().trim()
        if (!q) return allModels
        const populatedMatches = allModels.filter(m => m.name.toLowerCase().includes(q))
        return mergeModelHubResults(populatedMatches, remoteModels)
    }, [allModels, query, remoteModels])

    const changeQuery = useCallback((nextQuery: string) => {
        setQuery(nextQuery)
        if (nextQuery.trim() === submittedQueryRef.current) return
        searchGenRef.current += 1
        submittedQueryRef.current = ''
        setRemoteModels([])
        setSearching(false)
    }, [])

    const search = useCallback(
        async (nextQuery: string) => {
            const normalized = nextQuery.trim()
            setQuery(nextQuery)
            if (!engine || adapterType !== 'llama-cpp' || normalized.length === 0) return

            const generation = ++searchGenRef.current
            submittedQueryRef.current = normalized
            setRemoteModels([])
            setSearching(true)
            try {
                const result = await searchEngineHub(engine, normalized)
                if (searchGenRef.current !== generation) return
                setRemoteModels(result)
            } catch (error) {
                if (searchGenRef.current !== generation) return
                addError(getErrorString(error))
            } finally {
                if (searchGenRef.current === generation) setSearching(false)
            }
        },
        [engine, adapterType, addError]
    )

    const dismissError = useCallback(
        (id: string) => setErrors(prev => prev.filter(e => e.id !== id)),
        []
    )

    return {
        adapterType,
        loading,
        models,
        query,
        searching,
        errors,
        changeQuery,
        search,
        dismissError
    }
}
