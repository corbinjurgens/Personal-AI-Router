// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type { RoutingHistoryConsumer } from '@/shared/types/routing'
import { useServiceStatusStore } from '@/ui/stores/service-status.store'

interface RoutingStore {
    revision: number
    connected: boolean
    loading: boolean
    error: boolean
    initialize: () => void
    refresh: () => Promise<void>
    cleanup: () => void
}

let generation = 0
let loadGeneration = 0
let unsubs: Array<() => void> = []
let refreshCall: Promise<void> | null = null

// The API owns the bounded current/rAF/baseline banks and all ordering. Keep
// only scalar presentation state here; never duplicate its decoded records.
export const useRoutingStore = create<RoutingStore>((set, get) => {
    const update = (): void => {
        set(state => ({
            revision: state.revision + 1,
            loading: state.connected && !state.error && window.pairApi.routing.loading
        }))
    }
    const load = async (epoch: number): Promise<void> => {
        set({ loading: true, error: false })
        try {
            await window.pairApi.routing.subscribe()
            if (epoch === loadGeneration) {
                set({ error: false })
                update()
            }
        } catch {
            if (epoch === loadGeneration) {
                set({ error: true, loading: false })
                update()
            }
        }
    }
    return {
        revision: 0,
        connected: false,
        loading: false,
        error: false,
        initialize: () => {
            get().cleanup()
            const epoch = generation
            const connected =
                useServiceStatusStore.getState().status.connectorStatus === 'connected'
            set({ connected, loading: connected, error: false })
            unsubs.push(
                window.pairApi.routing.onChanged(() => {
                    if (epoch === generation) update()
                    return undefined
                })
            )
            // The preload transport stays connected to Electron even when the
            // broker is gone. Only service:status reports that native lifecycle.
            unsubs.push(
                useServiceStatusStore.subscribe((state, previous) => {
                    if (
                        epoch !== generation ||
                        state.status.connectorStatus === previous.status.connectorStatus
                    )
                        return
                    const connected = state.status.connectorStatus === 'connected'
                    set({ connected, loading: connected, error: false })
                    if (connected) void get().refresh()
                    else {
                        loadGeneration++
                        refreshCall = null
                        void window.pairApi.routing.unsubscribe().catch(() => {})
                        update()
                    }
                })
            )
            if (connected) void load(++loadGeneration)
        },
        refresh: () => {
            if (!get().connected) return Promise.resolve()
            if (refreshCall) return refreshCall
            // Retire the old load before unsubscribe rejects its initial promise.
            const epoch = ++loadGeneration
            set({ loading: true, error: false })
            const pending = (async () => {
                try {
                    await window.pairApi.routing.unsubscribe()
                    if (epoch !== loadGeneration) return
                    await load(epoch)
                } catch {
                    if (epoch === loadGeneration) set({ error: true, loading: false })
                }
            })().finally(() => {
                if (refreshCall === pending) refreshCall = null
            })
            refreshCall = pending
            return pending
        },
        cleanup: () => {
            generation++
            loadGeneration++
            refreshCall = null
            unsubs.forEach(remove => remove())
            unsubs = []
            void window.pairApi?.routing.unsubscribe().catch(() => {})
            set(state => ({
                connected: false,
                loading: false,
                error: false,
                revision: state.revision + 1
            }))
        }
    }
})

/** Synchronous rendering loan; no history graph escapes into React props/state. */
export function consumeRoutingRow(index: number, consume: RoutingHistoryConsumer): boolean {
    return window.pairApi?.routing.consumeCurrent(index, consume) ?? false
}
