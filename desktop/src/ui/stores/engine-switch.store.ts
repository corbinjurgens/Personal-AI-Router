// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand'
import type { SwitchReceipt, SwitchRequest, SwitchStatus } from '@/shared/types/engine-switch'
import getErrorString from '@/shared/utils/get-error-string'
import { serviceErrorCode } from '@/ui/api/service-transport'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'

interface SwitchEntry {
    revision?: string
    activeRequestId?: string
    request?: SwitchRequest
    receipt?: SwitchReceipt
    observed?: SwitchStatus['observed']
    own?: boolean
    unknown?: boolean
    unavailable?: string
}
interface SwitchStore {
    entries: Record<string, SwitchEntry>
    refresh(nodeId: string): Promise<boolean>
    submit(nodeId: string, source: string, target: string): Promise<void>
    reserved(nodeId: string, engine: string): boolean
}
interface Watch {
    consumers: number
    generation: number
    read: number
    timer?: ReturnType<typeof setTimeout>
}
const watches = new Map<string, Watch>()
// Retain an outstanding RPC's slot even after retiring its callbacks. The
// protocol has bounded waits but no cancellation channel.
const waits = new Set<string>()
let unsubs: Array<() => void> = []

function connected(nodeId: string): boolean {
    return (
        useConnectionStore.getState().connected &&
        useNodesStore.getState().nodes.get(nodeId)?.status !== 'offline'
    )
}
function current(nodeId: string, watch: Watch, generation: number): boolean {
    return watches.get(nodeId) === watch && watch.generation === generation && connected(nodeId)
}
function put(nodeId: string, change: Partial<SwitchEntry>): void {
    useEngineSwitchStore.setState(state => ({
        entries: { ...state.entries, [nodeId]: { ...state.entries[nodeId], ...change } }
    }))
}
function unresolved(entry: SwitchEntry | undefined): boolean {
    return !!(
        entry?.unknown ||
        entry?.activeRequestId ||
        entry?.receipt?.recoveryRequired ||
        (entry?.receipt && !entry.receipt.terminal)
    )
}
function merge(nodeId: string, requestId: string, receipt: SwitchReceipt): boolean {
    const entry = useEngineSwitchStore.getState().entries[nodeId]
    if (
        receipt.nodeId !== nodeId ||
        receipt.requestId !== requestId ||
        receipt.request.requestId !== requestId ||
        entry?.request?.requestId !== requestId ||
        receipt.request.sourceEngine !== entry.request.sourceEngine ||
        receipt.request.targetEngine !== entry.request.targetEngine ||
        receipt.expectedRevision !== entry.request.expectedRevision ||
        (entry.receipt && receipt.sequence <= entry.receipt.sequence)
    )
        return false
    put(nodeId, { receipt, unknown: false, unavailable: undefined })
    usePendingActionsStore.getState().endSwitch(nodeId, requestId)
    if (receipt.error?.code === -32012)
        void useNodeConfigurationStore
            .getState()
            .readManagement(
                nodeId,
                useConnectionStore.getState().selfId === nodeId ? undefined : nodeId
            )
    return true
}
function refreshPermission(nodeId: string, code: number | undefined): void {
    if (code === -32012)
        void useNodeConfigurationStore
            .getState()
            .readManagement(
                nodeId,
                useConnectionStore.getState().selfId === nodeId ? undefined : nodeId
            )
}
function scheduleWait(nodeId: string): void {
    const watch = watches.get(nodeId)
    const entry = useEngineSwitchStore.getState().entries[nodeId]
    if (!watch?.consumers || !connected(nodeId) || !entry?.request || entry.unavailable) return
    if (entry.receipt?.terminal) return
    const requestId = entry.request.requestId
    const key = JSON.stringify([nodeId, requestId])
    if (waits.has(key) || watch.timer !== undefined) return
    const generation = watch.generation
    // A wait belongs to its starting evidence. A new request or accepted
    // receipt supersedes both its result and its error.
    const currentWait = (): boolean => {
        const latest = useEngineSwitchStore.getState().entries[nodeId]
        return (
            current(nodeId, watch, generation) &&
            latest?.request === entry.request &&
            latest?.receipt === entry.receipt
        )
    }
    waits.add(key)
    void window.pairApi.engines
        .switchStatus({ nodeId, requestId, afterSequence: entry.receipt?.sequence })
        .then(status => {
            if (!currentWait()) return
            const merged = status.receipt && merge(nodeId, requestId, status.receipt)
            if (merged && status.receipt?.terminal)
                void useEngineSwitchStore.getState().refresh(nodeId)
            if (!status.receipt)
                put(nodeId, { unavailable: 'Switch receipt unavailable. Refresh readback.' })
            if (merged && status.observed) put(nodeId, { observed: status.observed })
        })
        .catch(error => {
            if (currentWait()) {
                put(nodeId, { unavailable: `Switch receipt unavailable: ${getErrorString(error)}` })
                refreshPermission(nodeId, serviceErrorCode(error))
            }
        })
        .finally(() => {
            waits.delete(key)
            const latest = watches.get(nodeId)
            if (!latest?.consumers || !connected(nodeId)) return
            // An unchanged bounded wait is only an invitation to wait again.
            latest.timer = setTimeout(() => {
                latest.timer = undefined
                scheduleWait(nodeId)
            }, 0)
        })
}
function retire(nodeId: string): void {
    const watch = watches.get(nodeId)
    if (watch) {
        watch.generation++
        watch.read++
        clearTimeout(watch.timer)
        watch.timer = undefined
    }
    put(nodeId, {
        revision: undefined,
        unavailable: 'Device disconnected. Refresh after reconnecting.'
    })
}

export const useEngineSwitchStore = create<SwitchStore>((_set, get) => ({
    entries: {},
    reserved: (nodeId, engine) => {
        const entry = get().entries[nodeId]
        const request = entry?.request
        return (
            usePendingActionsStore.getState().isSwitchPending(nodeId, engine) ||
            (!!request &&
                unresolved(entry) &&
                (request.sourceEngine === engine || request.targetEngine === engine)) ||
            (!!entry?.activeRequestId && !entry.request)
        )
    },
    refresh: async nodeId => {
        const watch = watches.get(nodeId)
        if (!watch?.consumers || !connected(nodeId)) return false
        const generation = watch.generation
        const read = ++watch.read
        put(nodeId, { revision: undefined })
        try {
            const status = await window.pairApi.engines.switchStatus({ nodeId })
            if (!current(nodeId, watch, generation) || watch.read !== read) return false
            put(nodeId, {
                revision: status.configurationRevision || undefined,
                activeRequestId: status.activeRequestId,
                unavailable: status.configurationRevision
                    ? undefined
                    : 'Switch readback unavailable.'
            })
            const entry = get().entries[nodeId]
            const ownUnresolved =
                entry?.own &&
                (entry.unknown ||
                    entry.receipt?.recoveryRequired ||
                    (entry.receipt && !entry.receipt.terminal))
            const requestId = ownUnresolved
                ? entry.request?.requestId
                : (status.activeRequestId ?? entry?.request?.requestId)
            if (requestId) {
                if (
                    !ownUnresolved &&
                    status.activeRequestId &&
                    status.activeRequestId !== entry?.request?.requestId
                )
                    put(nodeId, {
                        request: undefined,
                        receipt: undefined,
                        observed: undefined,
                        own: false,
                        unknown: false
                    })
                const result = await window.pairApi.engines.switchStatus({ nodeId, requestId })
                if (!current(nodeId, watch, generation) || watch.read !== read) return false
                const receipt = result.receipt
                if (!receipt || receipt.nodeId !== nodeId || receipt.requestId !== requestId) {
                    put(nodeId, {
                        unavailable: 'Switch busy; receipt unavailable. Refresh readback.'
                    })
                    return false
                }
                if (entry?.request?.requestId !== requestId)
                    put(nodeId, {
                        request: receipt.request,
                        receipt: undefined,
                        own: false,
                        unknown: false
                    })
                merge(nodeId, requestId, receipt)
                if (
                    result.observed &&
                    (get().entries[nodeId].receipt?.sequence ?? 0) <= receipt.sequence
                )
                    put(nodeId, { observed: result.observed })
                scheduleWait(nodeId)
            } else if (entry?.request && (entry.unknown || entry.receipt?.recoveryRequired)) {
                scheduleWait(nodeId)
            }
            return !!status.configurationRevision
        } catch (error) {
            if (current(nodeId, watch, generation) && watch.read === read) {
                put(nodeId, {
                    revision: undefined,
                    unavailable: `Switch readback unavailable: ${getErrorString(error)}`
                })
                refreshPermission(nodeId, serviceErrorCode(error))
            }
            return false
        }
    },
    submit: async (nodeId, source, target) => {
        if (
            !source ||
            !target ||
            source === target ||
            !connected(nodeId) ||
            unresolved(get().entries[nodeId])
        )
            return
        const watch = watches.get(nodeId)
        if (!watch?.consumers) return
        const requestId = crypto.randomUUID()
        const pending = usePendingActionsStore.getState()
        if (!pending.beginSwitch(nodeId, requestId, source, target)) return
        const generation = watch.generation
        try {
            const local = useConnectionStore.getState().selfId === nodeId
            if (!local) {
                await useNodeConfigurationStore.getState().readManagement(nodeId, nodeId)
                if (
                    useNodeConfigurationStore.getState().management.get(nodeId)?.state !== 'allowed'
                )
                    return
            }
            if (!(await get().refresh(nodeId)) || !current(nodeId, watch, generation)) return
            const entry = get().entries[nodeId]
            if (!entry?.revision || unresolved(entry)) return
            if (
                !local &&
                useNodeConfigurationStore.getState().management.get(nodeId)?.state !== 'allowed'
            )
                return
            const request: SwitchRequest = {
                nodeId,
                requestId,
                sourceEngine: source,
                targetEngine: target,
                expectedRevision: entry.revision
            }
            put(nodeId, {
                request,
                receipt: undefined,
                observed: undefined,
                own: true,
                unknown: true,
                unavailable: undefined
            })
            try {
                const result = await window.pairApi.engines.switchEngine(request)
                if (!current(nodeId, watch, generation)) return
                merge(nodeId, requestId, result.receipt)
            } catch (error) {
                const latest = get().entries[nodeId]
                if (
                    !current(nodeId, watch, generation) ||
                    latest?.request?.requestId !== requestId ||
                    latest.receipt
                )
                    return
                const code = serviceErrorCode(error)
                put(nodeId, {
                    unavailable: `${code === -32012 ? 'Remote management permission denied' : code === -32011 ? 'Local Off veto' : code === -32010 ? 'Configuration conflict' : 'Switch outcome unknown'}: ${getErrorString(error)}. Refresh readback; the request will not be resent.`
                })
                if (code === -32012)
                    void useNodeConfigurationStore
                        .getState()
                        .readManagement(nodeId, local ? undefined : nodeId)
            }
            scheduleWait(nodeId)
        } finally {
            pending.endSwitch(nodeId, requestId)
        }
    }
}))

/** Install listeners before the first read; editors on the same node share them. */
export function watchEngineSwitch(nodeId: string): () => void {
    let watch = watches.get(nodeId)
    if (!watch) {
        watch = { consumers: 0, generation: 0, read: 0 }
        watches.set(nodeId, watch)
    }
    watch.consumers++
    if (unsubs.length === 0) {
        unsubs = [
            window.pairApi.engines.onSwitchChanged(change => {
                const watched = watches.get(change.nodeId)
                if (!watched?.consumers) return
                // Notifications carry no receipt, and cannot acknowledge a phase.
                void useEngineSwitchStore.getState().refresh(change.nodeId)
            }),
            window.pairApi.engines.onSettingsDisconnected(({ nodeId: id }) => retire(id)),
            useConnectionStore.subscribe((state, previous) => {
                if (state.connected === previous.connected && state.selfId === previous.selfId)
                    return
                for (const id of watches.keys()) {
                    retire(id)
                    if (state.connected) void useEngineSwitchStore.getState().refresh(id)
                }
            }),
            useNodesStore.subscribe((state, previous) => {
                for (const id of watches.keys()) {
                    const offline = state.nodes.get(id)?.status === 'offline'
                    if (offline === (previous.nodes.get(id)?.status === 'offline')) continue
                    retire(id)
                    if (!offline) void useEngineSwitchStore.getState().refresh(id)
                }
            })
        ]
    }
    if (watch.consumers === 1) void useEngineSwitchStore.getState().refresh(nodeId)
    let stopped = false
    return () => {
        if (stopped) return
        stopped = true
        if (--watch.consumers === 0) {
            retire(nodeId)
            watches.delete(nodeId)
        }
        if (watches.size === 0) {
            unsubs.forEach(stop => stop())
            unsubs = []
        }
    }
}
