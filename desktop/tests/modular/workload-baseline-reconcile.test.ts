// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { WorkloadState } from '@/shared/types/workloads'

const pushes = vi.hoisted(() => ({
    log: [] as { channel: string; key: string }[]
}))

vi.mock('electron', () => ({ BrowserWindow: { getAllWindows: () => [] } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn() }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({
    emitBridgePush: (channel: string, payload: { workloadId?: string; id?: string }): void => {
        if (channel !== 'workloads:upsert' && channel !== 'workloads:remove') return
        pushes.log.push({ channel, key: payload.workloadId ?? payload.id ?? '' })
    }
}))

import {
    getModularBridgeState,
    parseWorkloadsInitial
} from '@/electron/service-bridge/modular-state'

const ORIGIN = 'uuid-reconcile'

function info(id: string, state: WorkloadState, model = 'm') {
    return { id, engine: 'ollama', state, model, originatedFrom: ORIGIN, createdAt: 1 }
}

function upsert(id: string, state: WorkloadState, model = 'm'): void {
    getModularBridgeState().upsertWorkloadFromInfo({ workloadInfo: info(id, state, model) })
}

function seed(since: number, entries: ReturnType<typeof info>[]) {
    return getModularBridgeState().seedWorkloads(
        parseWorkloadsInitial({ workloads: entries }),
        since
    )
}

function key(id: string): string {
    return `${ORIGIN}\u0000${id}`
}

// `seedWorkloads` used to only add, so a missed `workloads:remove` left a
// finished job in the catalog for the rest of the session. These pin that a
// baseline now repairs the catalog without undoing realtime events that raced
// the request.
describe('workload baseline reconciliation', () => {
    beforeEach(() => {
        getModularBridgeState().clearWorkloads()
        pushes.log.length = 0
    })

    it('drops a terminal entry the baseline no longer lists, and says so', () => {
        upsert('done', 'completed')
        upsert('kept', 'completed')
        pushes.log.length = 0

        const since = getModularBridgeState().beginWorkloadBaseline()
        const seeded = seed(since, [info('kept', 'completed')])

        expect(seeded[key('done')]).toBeUndefined()
        expect(seeded[key('kept')]).toBeDefined()
        expect(pushes.log).toEqual([{ channel: 'workloads:remove', key: 'done' }])
    })

    it('drops failed and cancelled entries the same way', () => {
        upsert('failed', 'failed')
        upsert('cancelled', 'cancelled')

        const since = getModularBridgeState().beginWorkloadBaseline()
        const seeded = seed(since, [])

        expect(seeded[key('failed')]).toBeUndefined()
        expect(seeded[key('cancelled')]).toBeUndefined()
    })

    it('never drops an active entry the baseline is missing', () => {
        upsert('queued', 'queued')
        upsert('running', 'running')
        pushes.log.length = 0

        const since = getModularBridgeState().beginWorkloadBaseline()
        const seeded = seed(since, [])

        expect(seeded[key('queued')]).toMatchObject({ state: 'queued' })
        expect(seeded[key('running')]).toMatchObject({ state: 'running' })
        expect(pushes.log).toEqual([])
    })

    it('keeps a terminal entry upserted after the baseline request was sent', () => {
        const since = getModularBridgeState().beginWorkloadBaseline()
        // Raced the request: the broker may have taken its snapshot first.
        upsert('raced', 'completed')
        pushes.log.length = 0

        const seeded = seed(since, [])

        expect(seeded[key('raced')]).toMatchObject({ state: 'completed' })
        expect(pushes.log).toEqual([])
    })

    it('does not let an older baseline row overwrite an upsert that raced it', () => {
        const since = getModularBridgeState().beginWorkloadBaseline()
        upsert('job', 'completed', 'fresh')

        const seeded = seed(since, [info('job', 'running', 'stale')])

        expect(seeded[key('job')]).toMatchObject({ state: 'completed', model: 'fresh' })
    })

    it('does not resurrect a job removed while the baseline request was in flight', () => {
        upsert('gone', 'completed')
        const since = getModularBridgeState().beginWorkloadBaseline()
        getModularBridgeState().removeWorkloadFromParams({
            workloadId: 'gone',
            originatedFrom: ORIGIN
        })

        const seeded = seed(since, [info('gone', 'completed')])

        expect(seeded[key('gone')]).toBeUndefined()
    })

    it('repairs a missed transition on an entry the stream has not touched since', () => {
        upsert('stuck', 'running')
        pushes.log.length = 0

        const since = getModularBridgeState().beginWorkloadBaseline()
        const seeded = seed(since, [info('stuck', 'completed')])

        expect(seeded[key('stuck')]).toMatchObject({ state: 'completed' })
        expect(pushes.log).toEqual([{ channel: 'workloads:upsert', key: 'stuck' }])
    })

    it('fills in unseen baseline jobs without a push and leaves equal entries alone', () => {
        upsert('same', 'completed')
        pushes.log.length = 0

        const since = getModularBridgeState().beginWorkloadBaseline()
        const seeded = seed(since, [info('same', 'completed'), info('new', 'queued')])

        expect(seeded[key('new')]).toMatchObject({ state: 'queued' })
        expect(seeded[key('same')]).toMatchObject({ state: 'completed' })
        expect(pushes.log).toEqual([])
    })
})
