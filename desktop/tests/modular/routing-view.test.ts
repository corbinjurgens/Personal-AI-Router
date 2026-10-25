// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import fs from 'fs'
import path from 'path'
import { EventEmitter } from 'events'
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
    RoutingHistoryBank,
    type RoutingHistoryRecord,
    type RoutingHistoryTransfer
} from '@/shared/types/routing'
import type { Workload } from '@/shared/types/workloads'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'
import RoutingHistory from '@/ui/components/RoutingSettings/RoutingHistory'
import WorkloadItemCard from '@/ui/components/Workloads/WorkloadItemCard'
import { consumeRoutingRow, useRoutingStore } from '@/ui/stores/routing.store'
import { useServiceStatusStore } from '@/ui/stores/service-status.store'
import {
    routingAttemptOutcome,
    routingDisplayReason,
    routingMoney,
    routingReason,
    routingRequestKey,
    routingState,
    routingSummary,
    routingWait,
    workloadRoutingSummary
} from '@/ui/utils/routing-labels'

const corpus: { cases: { name: string; record: RoutingHistoryRecord }[] } = JSON.parse(
    fs.readFileSync(
        path.resolve('../services/shared/nodepolicy/testdata/routing-request.json'),
        'utf8'
    )
)
function fixture(index = 0): RoutingHistoryRecord {
    return structuredClone(corpus.cases[index].record)
}
function workloadFor(record: RoutingHistoryRecord): Workload {
    const association = record.workload
    if (!association) throw new Error('fixture needs association')
    return {
        id: association.id,
        originatedFrom: association.originNodeId,
        runId: association.proxyRunId,
        engine: 'llama-cpp',
        model: record.requested.model,
        state: 'failed',
        createdAt: record.createdAt,
        startedAt: record.createdAt,
        completedAt: record.endedAt,
        error: 'untrusted-workload-error',
        requesterId: null
    }
}

describe('routing presentation labels', () => {
    it.each([
        ['0', '$0'],
        ['1', '$0.000000001'],
        ['123456789', '$0.123456789'],
        ['1000000000', '$1'],
        ['9223372036854775807', '$9223372036.854775807'],
        [null, '—']
    ])('formats nano-USD %s exactly', (value, expected) =>
        expect(routingMoney(value)).toBe(expected)
    )

    it.each([
        [null, '—'],
        [350, '350 ms'],
        [1234, '1.2 s']
    ])('formats wait %s', (value, expected) => expect(routingWait(value)).toBe(expected))

    it('uses closed reason labels and never reflects unrecognized text or object members', () => {
        expect(routingReason('upstream-status')).toBe('Engine returned an error status')
        expect(routingReason('private-body-sentinel')).toBe('Unknown')
        expect(routingReason('__proto__')).toBe('Unknown')
        expect(routingReason('')).toBe('')
    })

    it('uses full keys, winning dispositions, and real continuation links', () => {
        const completed = fixture()
        expect(routingSummary(completed)).toBe(
            `Routing: 2 attempts, served on synthetic-node / llamacpp`
        )
        completed.spend.accounting = 'failed'
        expect(routingState(completed)).toBe('Completed')
        const interrupted = fixture(1)
        expect(routingSummary(interrupted)).toBe('Routing failed: Stream interrupted')
        const paused = fixture(3)
        expect(routingSummary(paused)).toBe('Streaming paused, waiting to resume')
        expect(routingAttemptOutcome(paused, paused.attempts[0])).toBe(
            'Streaming paused, waiting to resume'
        )
        expect(routingDisplayReason(paused, 'stream-interrupted')).toBe('')
        paused.recoverable = false
        expect(routingAttemptOutcome(paused, paused.attempts[0])).toBe('Interrupted')
        const resumed = fixture(4)
        expect(routingAttemptOutcome(resumed, resumed.attempts[0])).toBe('Interrupted, resumed')
        expect(routingSummary(fixture(2))).toBe('Routing cancelled: Cancelled by user')
        expect(routingRequestKey(completed.request)).toBe(
            `${completed.request.originNodeId}/${completed.request.proxyRunId}/${completed.request.requestId}`
        )
    })

    it('matches every owner identity field and omits a trimmed associated attempt', () => {
        const record = fixture()
        const workload = workloadFor(record)
        expect(workloadRoutingSummary(record, workload)?.text).toContain('Routing:')
        const differentWorkloads: Workload[] = [
            { ...workload, id: 'reused' },
            { ...workload, originatedFrom: 'other' },
            { ...workload, runId: 'other' },
            { ...workload, engine: 'ollama' },
            { ...workload, runId: undefined }
        ]
        for (const different of differentWorkloads) {
            expect(workloadRoutingSummary(record, different)).toBeNull()
        }
        if (!record.workload) throw new Error('missing association')
        record.workload.attemptIndex = 999
        expect(workloadRoutingSummary(record, workload)).toBeNull()
    })

    it('keeps detached local summaries independent of later API liveness and destination', () => {
        const record = fixture()
        const workload = workloadFor(record)
        if (!record.workload) throw new Error('missing association')
        record.workload.attached = false
        record.workload.attemptIndex = 0
        const before = workloadRoutingSummary(record, workload)
        record.state = 'streaming'
        record.recoverable = true
        record.attempts.push(fixture(1).attempts[0])
        record.served = fixture(1).attempts[0].destination
        expect(workloadRoutingSummary(record, workload)).toEqual(before)
        expect(before?.text).not.toContain('account')
        expect(before?.error).toBe(true)
    })
})

const invoke = vi.fn<ServiceTransport['invoke']>()
const pushes = new EventEmitter()
const lifecycle = new EventEmitter()
const frames = new Map<number, FrameRequestCallback>()
let frameId = 0
let lease = 0
let transport: ServiceTransport
let container: HTMLDivElement
let root: Root
let bank: RoutingHistoryBank

async function settle(): Promise<void> {
    for (let i = 0; i < 20; i++) await Promise.resolve()
}
async function packet(transfer: RoutingHistoryTransfer): Promise<void> {
    invoke
        .mockResolvedValueOnce(transfer)
        .mockResolvedValueOnce({ state: 'accepted', generation: transfer.generation })
    pushes.emit('routing-history:ready', { generation: transfer.generation })
    await settle()
    for (const [id, callback] of frames) {
        frames.delete(id)
        callback(0)
    }
    if ('lease' in transfer)
        pushes.emit('routing-history:committed', {
            generation: transfer.generation,
            revision: transfer.lease
        })
    await settle()
}
async function initial(records: RoutingHistoryRecord[]): Promise<void> {
    invoke.mockResolvedValueOnce({ generation: 1 })
    useRoutingStore.getState().initialize()
    await settle()
    for (let index = 0; index < records.length; index++) {
        bank.clear()
        bank.apply(new TextEncoder().encode(JSON.stringify(records[index])))
        await packet({
            state: 'data',
            generation: 1,
            lease: ++lease,
            initial: true,
            index,
            bytes: bank.at(0).slice()
        })
    }
    await packet({
        state: 'end',
        generation: 1,
        lease: ++lease,
        count: records.length,
        retired: false
    })
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => true })
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
        frames.set(++frameId, callback)
        return frameId
    })
    vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))
    invoke.mockReset().mockResolvedValue(null)
    pushes.removeAllListeners()
    lifecycle.removeAllListeners()
    frames.clear()
    lease = 0
    bank = new RoutingHistoryBank()
    transport = {
        connected: true,
        invoke,
        subscribePush: (channel, callback) => {
            pushes.on(channel, callback)
            return () => {
                pushes.off(channel, callback)
            }
        },
        onConnect: callback => {
            lifecycle.on('connect', callback)
            return () => {
                lifecycle.off('connect', callback)
            }
        },
        onDisconnect: callback => {
            lifecycle.on('disconnect', callback)
            return () => {
                lifecycle.off('disconnect', callback)
            }
        },
        onAuthFailure: () => () => {},
        destroy: () => {}
    }
    useServiceStatusStore.setState({ status: { connectorStatus: 'connected', weSpawned: false } })
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: createPairApi(transport)
    })
    // initialize's initial cleanup retires an existing desired subscription only;
    // an idle API needs no native unsubscribe call.
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})
afterEach(async () => {
    await act(async () => {
        root.unmount()
        useRoutingStore.getState().cleanup()
        await settle()
    })
    container.remove()
    vi.unstubAllGlobals()
})

describe('bounded history presentation and lifecycle', () => {
    it('renders empty and disconnected states and subscribes before its initial pull', async () => {
        await initial([])
        await act(async () => root.render(createElement(RoutingHistory)))
        expect(container.textContent).toContain('No routed requests yet.')
        expect(invoke.mock.calls[0][0]).toBe('routing:subscribe')
        expect(invoke.mock.calls[1][0]).toBe('routing-history:pull')
        await act(async () =>
            useServiceStatusStore.setState({
                status: { connectorStatus: 'disconnected', weSpawned: false }
            })
        )
        expect(container.textContent).toContain('Disconnected')
        expect(window.pairApi.routing.size).toBe(0)
    })

    it('renders corpus details with focusable expansion, amount meanings and completed accounting failure', async () => {
        const records = corpus.cases.map((_, index) => fixture(index))
        records[0].spend.accounting = 'failed'
        records[0].requested.model = 'm'.repeat(512)
        records[0].attempts[0].destination.model = null
        await initial(records)
        await act(async () => root.render(createElement(RoutingHistory)))
        expect(container.textContent).toContain('Completed')
        expect(container.textContent).toContain('Cumulative granted:')
        expect(container.textContent).toContain('Current held:')
        expect(container.textContent).toContain('Accounting: failed')
        expect(container.textContent).toContain('(+2 older)')
        const button = Array.from(container.querySelectorAll('button')).find(value =>
            value.getAttribute('aria-label')?.includes(routingRequestKey(records[0].request))
        )
        if (!button) throw new Error('missing expansion button')
        button.focus()
        expect(document.activeElement).toBe(button)
        await act(async () => button.click())
        expect(button.getAttribute('aria-expanded')).toBe('true')
        expect(document.getElementById(button.getAttribute('aria-controls') ?? '')).not.toBeNull()
        expect(container.textContent).toContain('model not specified')
        expect(container.textContent).toContain('Admission: 350 ms')
        expect(container.textContent).toContain('First byte: 1.2 s')
        expect(container.textContent).toContain('HTTP status:')
        expect(container.textContent).toContain('Account state:')
        expect(container.querySelector('[title]')?.getAttribute('aria-label')).toBeTruthy()
        expect(
            container
                .querySelector('[aria-label="Routing history table"]')
                ?.classList.contains('overflow-x-auto')
        ).toBe(true)
        await act(async () => button.click())
        expect(button.getAttribute('aria-expanded')).toBe('false')
    })

    it('shows only exact local attempt failures on workload cards without reflecting the workload error', async () => {
        const record = fixture()
        if (!record.workload) throw new Error('missing association')
        record.workload.attached = false
        record.workload.attemptIndex = 0
        const workload = workloadFor(record)
        await initial([record])
        await act(async () => root.render(createElement(WorkloadItemCard, { workload })))
        expect(container.textContent).toContain('Routing: local attempt 1, Rejected: Node busy')
        expect(container.textContent).not.toContain('untrusted-workload-error')
    })

    it('keeps detached interruption text and styling unchanged when a later account failure wins', () => {
        const record = fixture(3)
        record.workload = {
            originNodeId: record.request.originNodeId,
            proxyRunId: record.request.proxyRunId,
            engine: 'llamacpp',
            id: 'detached',
            attemptIndex: 0,
            attached: false
        }
        const workload = workloadFor(record)
        bank.apply(new TextEncoder().encode(JSON.stringify(record)))
        const paused = workloadRoutingSummary(record, workload)
        record.state = 'failed'
        record.recoverable = false
        record.reason = 'account-ceiling'
        record.endedAt = record.createdAt + 2000
        record.seq++
        expect(bank.apply(new TextEncoder().encode(JSON.stringify(record)))).toBe(true)
        expect(workloadRoutingSummary(record, workload)).toEqual(paused)
        expect(paused).toEqual({ text: 'Routing: local attempt 1, Interrupted', error: false })
    })

    it('suppresses failed-card styling for an authoritative recoverable interruption', async () => {
        const record = fixture(3)
        record.workload = {
            originNodeId: record.request.originNodeId,
            proxyRunId: record.request.proxyRunId,
            engine: 'llamacpp',
            id: 'paused-job',
            attemptIndex: 0,
            attached: true
        }
        const workload = workloadFor(record)
        await initial([record])
        await act(async () => root.render(createElement(WorkloadItemCard, { workload })))
        expect(container.textContent).toContain('Streaming paused, waiting to resume')
        expect(container.querySelector('.workload-card-failed')).toBeNull()
        expect(container.textContent).not.toContain('untrusted-workload-error')
        expect(workload.state).toBe('failed')
    })

    it('retires a pending initial before refresh without presenting a false error', async () => {
        invoke.mockResolvedValueOnce({ generation: 1 })
        useRoutingStore.getState().initialize()
        await settle()
        invoke.mockResolvedValueOnce(null)
        const refreshed = useRoutingStore.getState().refresh()
        await settle()
        invoke
            .mockResolvedValueOnce({ state: 'retired', generation: 1 })
            .mockResolvedValueOnce({ generation: 2 })
        pushes.emit('routing-history:ready', { generation: 1 })
        await settle()
        expect(window.pairApi.routing.loading).toBe(true)
        expect(useRoutingStore.getState()).toMatchObject({
            connected: true,
            loading: true,
            error: false
        })
        await packet({ state: 'end', generation: 2, lease: 1, count: 0, retired: false })
        await refreshed
        expect(useRoutingStore.getState()).toMatchObject({
            connected: true,
            loading: false,
            error: false
        })
    })

    it('does not subscribe while the broker is stopped even though preload stays connected', async () => {
        useServiceStatusStore.setState({
            status: { connectorStatus: 'disconnected', weSpawned: false }
        })
        useRoutingStore.getState().initialize()
        await act(async () => root.render(createElement(RoutingHistory)))
        expect(transport.connected).toBe(true)
        expect(container.textContent).toContain('Disconnected')
        expect(useRoutingStore.getState()).toMatchObject({
            connected: false,
            loading: false,
            error: false
        })
        expect(invoke).not.toHaveBeenCalled()
    })

    it('rejects stale/equal and terminal-to-active revisions across delayed baseline and rAF updates', async () => {
        const terminal = fixture()
        terminal.seq = 5
        await initial([terminal])
        const old = fixture()
        old.seq = 4
        bank.apply(new TextEncoder().encode(JSON.stringify(old)))
        await packet({
            state: 'data',
            generation: 1,
            lease: ++lease,
            initial: false,
            bytes: bank.at(0).slice()
        })
        const lateActive = fixture()
        lateActive.seq = 6
        lateActive.state = 'routing'
        lateActive.endedAt = null
        lateActive.served = null
        bank.clear()
        bank.apply(new TextEncoder().encode(JSON.stringify(lateActive)))
        const before = useRoutingStore.getState().revision
        await packet({
            state: 'data',
            generation: 1,
            lease: ++lease,
            initial: false,
            bytes: bank.at(0).slice()
        })
        expect(useRoutingStore.getState().revision).toBeGreaterThan(before)
        consumeRoutingRow(0, value => {
            expect(value.seq).toBe(5)
            expect(value.state).toBe('completed')
            return undefined
        })
    })

    it('publishes a changed row only after rAF and ignores retired connection callbacks', async () => {
        const record = fixture()
        await initial([record])
        const updated = fixture()
        updated.seq++
        updated.spend.accounting = 'failed'
        bank.clear()
        bank.apply(new TextEncoder().encode(JSON.stringify(updated)))
        const nextLease = ++lease
        invoke
            .mockResolvedValueOnce({
                state: 'data',
                generation: 1,
                lease: nextLease,
                initial: false,
                bytes: bank.at(0).slice()
            })
            .mockResolvedValueOnce({ state: 'accepted', generation: 1 })
        pushes.emit('routing-history:ready', { generation: 1 })
        await settle()
        consumeRoutingRow(0, value => {
            expect(value.seq).toBe(record.seq)
            return undefined
        })
        const before = useRoutingStore.getState().revision
        lifecycle.emit('disconnect')
        useServiceStatusStore.setState({
            status: { connectorStatus: 'disconnected', weSpawned: false }
        })
        expect(window.pairApi.routing.size).toBe(0)
        // A cancelled frame callback and its delayed commit belong to the old run.
        pushes.emit('routing-history:committed', { generation: 1, revision: nextLease })
        await settle()
        expect(useRoutingStore.getState().revision).toBeGreaterThan(before)
        expect(window.pairApi.routing.size).toBe(0)
        invoke
            .mockResolvedValueOnce({ state: 'retired', generation: 1 })
            .mockResolvedValueOnce({ generation: 2 })
        pushes.emit('routing-history:ready', { generation: 1 })
        await settle()
        lifecycle.emit('connect')
        useServiceStatusStore.setState({
            status: { connectorStatus: 'connected', weSpawned: false }
        })
        await settle()
        expect(useRoutingStore.getState().connected).toBe(true)
        expect(
            invoke.mock.calls.filter(([channel]) => channel === 'routing:subscribe')
        ).toHaveLength(2)
        pushes.emit('routing-history:committed', { generation: 1, revision: nextLease })
        expect(window.pairApi.routing.size).toBe(0)
    })

    it('fences a delayed initial completion after store cleanup', async () => {
        invoke.mockResolvedValueOnce({ generation: 1 })
        useRoutingStore.getState().initialize()
        await settle()
        const staleListeners = pushes.listeners('routing-history:committed')
        useRoutingStore.getState().cleanup()
        const revision = useRoutingStore.getState().revision
        for (const callback of staleListeners)
            Reflect.apply(callback, undefined, [{ generation: 1, revision: 1 }])
        await settle()
        expect(useRoutingStore.getState()).toMatchObject({
            revision,
            connected: false,
            loading: false,
            error: false
        })
    })

    it('bounds the newest window and prevents evicted delayed rows from resurrecting', () => {
        for (let id = 1; id <= 257; id++) {
            const row = fixture()
            row.request.requestId = String(id)
            row.createdAt = 1700000000000 - 300 + id
            bank.apply(new TextEncoder().encode(JSON.stringify(row)))
        }
        expect(bank.size).toBe(256)
        const stale = fixture()
        stale.request.requestId = '1'
        stale.createdAt = 1700000000000 - 299
        stale.seq = 99
        expect(bank.apply(new TextEncoder().encode(JSON.stringify(stale)))).toBe(false)
        bank.consume(0, value => {
            expect(value.request.requestId).toBe('257')
            return undefined
        })
    })
})
