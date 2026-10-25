// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { createOverviewWindow } from '@/electron/window'
import type { JsonObject, JsonValue } from '@/electron/service-bridge/json-rpc-client'
import {
    ModelDownloadObservation,
    parseModelDownloadSnapshot,
    parseModelsResult
} from '@/electron/service-bridge/model-download'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import { isEnginePullInProgress } from '@/shared/utils/engine-progress'

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<typeof createOverviewWindow>() }))

function entry(overrides: JsonObject = {}): JsonObject {
    return {
        id: '1'.repeat(32),
        engine: 'llamacpp',
        model: 'catalog/model',
        phase: 'in_progress',
        blocksReadiness: true,
        ...overrides
    }
}

function snapshot(
    revision = '1',
    entries: JsonValue[] = [entry()],
    source = 'a'.repeat(32)
): JsonObject {
    return { version: 1, sourceId: source, revision, recoveryRequired: false, entries }
}

function observation(wire: JsonValue) {
    const parsed = parseModelDownloadSnapshot(wire)
    if (!parsed) throw new Error('Invalid snapshot fixture')
    return parsed
}

describe('download snapshot validation', () => {
    it('preserves measured zero and persistence qualification', () => {
        const wire = snapshot('18446744073709551615', [
            entry({
                phase: 'complete',
                blocksReadiness: false,
                percent: 0,
                persistence: 'uncertain'
            })
        ])
        expect(parseModelDownloadSnapshot(wire)).toEqual(wire)
    })

    const invalid: Array<{ name: string; value: JsonValue }> = [
        { name: 'missing support', value: null },
        { name: 'unsupported version', value: { ...snapshot(), version: 2 } },
        { name: 'noncanonical revision', value: snapshot('01') },
        { name: 'revision overflow', value: snapshot('18446744073709551616') },
        { name: 'invalid source', value: snapshot('1', [], 'source') },
        {
            name: 'missing entries',
            value: { version: 1, sourceId: 'a'.repeat(32), revision: '1', recoveryRequired: false }
        },
        { name: 'unknown phase', value: snapshot('1', [entry({ phase: 'ready' })]) },
        {
            name: 'missing readiness boolean',
            value: snapshot('1', [
                { id: '1'.repeat(32), engine: 'llamacpp', model: 'catalog/model', phase: 'failed' }
            ])
        },
        {
            name: 'active nonblocking entry',
            value: snapshot('1', [entry({ blocksReadiness: false })])
        },
        { name: 'blocking complete entry', value: snapshot('1', [entry({ phase: 'complete' })]) },
        { name: 'invalid percent', value: snapshot('1', [entry({ percent: 101 })]) },
        { name: 'null percent', value: snapshot('1', [entry({ percent: null })]) },
        {
            name: 'unsafe reason text',
            value: snapshot('1', [entry({ phase: 'failed', reason: '/private/path' })])
        },
        { name: 'progress reason', value: snapshot('1', [entry({ reason: 'busy' })]) },
        {
            name: 'progress persistence',
            value: snapshot('1', [entry({ persistence: 'uncertain' })])
        },
        { name: 'duplicate id', value: snapshot('1', [entry(), entry({ model: 'other' })]) },
        { name: 'duplicate key', value: snapshot('1', [entry(), entry({ id: '2'.repeat(32) })]) },
        {
            name: 'entry overflow',
            value: snapshot(
                '1',
                Array.from({ length: 65 }, (_, index) =>
                    entry({ id: index.toString(16).padStart(32, '0'), model: `model/${index}` })
                )
            )
        },
        {
            name: 'unsafe capacity overflow',
            value: snapshot(
                '1',
                Array.from({ length: 33 }, (_, index) =>
                    entry({ id: index.toString(16).padStart(32, '0'), model: `model/${index}` })
                )
            )
        },
        { name: 'oversize snapshot', value: { ...snapshot(), extra: 'x'.repeat(256 * 1024) } }
    ]
    it.each(invalid)('rejects $name', ({ value }) => {
        expect(parseModelDownloadSnapshot(value)).toBeUndefined()
    })

    it('keeps canonical names when lifecycle parsing fails, without granting currentness', () => {
        const result = parseModelsResult({
            models: ['ready'],
            modelsByEngine: { llamacpp: ['ready'] },
            downloadState: { version: 99 }
        })
        expect(result.modelsByEngine).toEqual({ llamacpp: ['ready'] })
        expect(result.downloadState).toBeUndefined()
        expect(result.downloadStateCurrent).toBe(false)
    })

    it('distinguishes unsupported data from a current known-empty snapshot', () => {
        expect(
            parseModelsResult({ models: [], modelsByEngine: {} }).downloadStateCurrent
        ).toBeUndefined()
        const result = parseModelsResult({
            models: [],
            modelsByEngine: {},
            downloadState: snapshot('1', [])
        })
        expect(result.downloadStateCurrent).toBe(true)
        expect(result.downloadState?.entries).toEqual([])
    })

    it('refuses a raw vendor inventory as a canonical snapshot', () => {
        expect(() => parseModelsResult({ data: [{ id: 'partial' }] })).toThrow(
            'malformed inventory'
        )
        expect(() => parseModelsResult({ models: [{ name: 'partial' }] })).toThrow(
            'malformed inventory'
        )
    })
})

describe('download source and revision fencing', () => {
    it('ignores an older revision and restores freshness on a same-revision read', () => {
        const state = new ModelDownloadObservation()
        state.accept(observation(snapshot('4')), true)
        expect(state.accept(observation(snapshot('3')), true)).toBe(false)
        state.accept(undefined, false)
        expect(state.current).toBe(false)
        expect(state.snapshot?.revision).toBe('4')
        expect(state.accept(observation(snapshot('4')), true)).toBe(true)
        expect(state.current).toBe(true)
    })

    it('replaces a source only from a current observation and rejects its retired source', () => {
        const state = new ModelDownloadObservation()
        state.accept(observation(snapshot()), true)
        expect(state.accept(observation(snapshot('0', [], 'b'.repeat(32))), false)).toBe(false)
        expect(state.accept(observation(snapshot('0', [], 'b'.repeat(32))), true)).toBe(true)
        expect(state.accept(observation(snapshot('100')), true)).toBe(false)
        expect(state.snapshot?.sourceId).toBe('b'.repeat(32))
    })

    it('rejects a baseline begun before a source replacement', () => {
        const state = new ModelDownloadObservation()
        const sequence = state.sequence
        state.accept(observation(snapshot('0', [], 'b'.repeat(32))), true)
        expect(state.accept(observation(snapshot()), true, sequence)).toBe(false)
        expect(state.snapshot?.sourceId).toBe('b'.repeat(32))
    })

    it('accepts a newer same-source baseline after intervening percentage updates', () => {
        const state = new ModelDownloadObservation()
        state.accept(observation(snapshot('1')), true)
        const sequence = state.sequence
        state.accept(observation(snapshot('2', [entry({ percent: 21 })])), true)
        expect(
            state.accept(observation(snapshot('3', [entry({ percent: 22 })])), true, sequence)
        ).toBe(true)
        expect(state.snapshot?.revision).toBe('3')
    })

    it('treats conflicting facts at the same revision as unavailable', () => {
        const state = new ModelDownloadObservation()
        state.accept(observation(snapshot('1')), true)
        state.accept(observation(snapshot('1', [entry({ percent: 42 })])), true)
        expect(state.current).toBe(false)
        expect(state.snapshot?.entries[0]?.percent).toBeUndefined()
    })
})

const state = getModularBridgeState()
let caseId = 0
let source: string
let nodeId: string

function localSnapshot(revision = '1', entries: JsonValue[] = [entry()]): JsonObject {
    return snapshot(revision, entries, source)
}

function localModels(
    revision = '1',
    entries: JsonValue[] = [entry()],
    names = ['catalog/model'],
    loaded: string[] = []
) {
    return parseModelsResult({
        models: names,
        modelsByEngine: { llamacpp: names },
        loadedByEngine: { llamacpp: loaded },
        downloadState: localSnapshot(revision, entries)
    })
}

function rows() {
    return (
        state
            .getEngineInitialState()
            .models.find(models => models.nodeId === nodeId && models.engineType === 'llama-cpp')
            ?.models ?? []
    )
}

describe('target-owned desktop download projection', () => {
    beforeEach(() => {
        caseId++
        nodeId = `download-local-${caseId}`
        source = caseId.toString(16).padStart(32, '0')
        state.setSelfId(nodeId)
        state.applyEngineManagerStatus({
            engine: 'llamacpp',
            installed: true,
            running: true,
            healthy: true
        })
        state.applyLocalModelsResult(localModels('0', [], []))
    })

    afterEach(() => vi.restoreAllMocks())

    it('projects an unseeded target pull and loaded residency independently of readiness', () => {
        state.applyLocalModelsResult(localModels('1', [], ['catalog/model'], ['catalog/model']))
        state.applyLocalEngineProgress({
            engine: 'llamacpp',
            op: 'pull',
            stage: 'pulling',
            id: '1'.repeat(32),
            model: 'catalog/model',
            downloadState: localSnapshot('2', [entry({ percent: 21 })])
        })
        expect(rows()).toMatchObject([
            {
                name: 'catalog/model',
                downloaded: false,
                status: 'loaded',
                downloadState: { id: '1'.repeat(32), phase: 'in_progress', percent: 21 }
            }
        ])
        expect(state.getEngineInitialState().activeProgress).toContainEqual(
            expect.objectContaining({
                nodeId,
                model: 'catalog/model',
                percent: 21,
                downloadSourceId: source
            })
        )
    })

    it('keeps two exact models on one engine without overwriting their identities', () => {
        const queued = entry({ id: '2'.repeat(32), model: 'catalog/second', phase: 'queued' })
        state.applyLocalModelsResult(localModels('1', [entry({ percent: 21 }), queued], []))
        expect(
            rows().map(row => [row.name, row.downloadState?.id, row.downloadState?.percent])
        ).toEqual([
            ['catalog/model', '1'.repeat(32), 21],
            ['catalog/second', '2'.repeat(32), undefined]
        ])
        state.finishModelPull('llama-cpp', 'catalog/model')
        expect(rows()).toHaveLength(2)
        expect(state.isModelPullActive('llama-cpp', 'catalog/second')).toBe(true)
    })

    it('does not let RPC settlement or an old terminal erase a newer attempt', () => {
        state.applyLocalModelsResult(
            localModels('4', [entry({ id: '2'.repeat(32), percent: 12 })], [])
        )
        state.finishModelPull('llama-cpp', 'catalog/model')
        state.applyLocalEngineProgress({
            engine: 'llamacpp',
            op: 'pull',
            id: '1'.repeat(32),
            model: 'catalog/model',
            stage: 'done',
            downloadState: localSnapshot('3', [
                entry({ phase: 'complete', blocksReadiness: false })
            ])
        })
        expect(rows()[0]?.downloadState).toMatchObject({
            id: '2'.repeat(32),
            phase: 'in_progress',
            percent: 12
        })
        expect(state.isModelPullActive('llama-cpp', 'catalog/model')).toBe(true)
    })

    it('keeps completion separate from current inventory and preserves last-good names after query failure', () => {
        const complete = entry({ phase: 'complete', blocksReadiness: false })
        state.applyLocalModelsResult(localModels('2', [complete], []))
        expect(rows()[0]).toMatchObject({
            downloaded: false,
            status: 'idle',
            downloadState: { phase: 'complete' }
        })
        state.applyLocalModelsResult(localModels('2', [complete]))
        expect(rows()[0]?.downloaded).toBe(true)
        state.applyLocalModelsResult(
            parseModelsResult({ models: [], downloadState: localSnapshot('2', [complete]) })
        )
        expect(rows()[0]).toMatchObject({ name: 'catalog/model', downloaded: false })
    })

    it('requires fresh canonical inventory after terminal progress even when the name existed before the pull', () => {
        state.applyLocalModelsResult(localModels('1', []))
        state.applyLocalEngineProgress({
            engine: 'llamacpp',
            op: 'pull',
            id: '1'.repeat(32),
            model: 'catalog/model',
            downloadState: localSnapshot('2')
        })
        const complete = entry({ phase: 'complete', blocksReadiness: false })
        state.applyLocalEngineProgress({
            engine: 'llamacpp',
            op: 'pull',
            id: '1'.repeat(32),
            model: 'catalog/model',
            downloadState: localSnapshot('3', [complete])
        })
        expect(rows()[0]).toMatchObject({ downloaded: false, downloadState: { phase: 'complete' } })
        state.applyLocalModelsResult(localModels('3', [complete]))
        expect(rows()[0]?.downloaded).toBe(true)
    })

    it('keeps an unrelated engine ready across percentage-only pull updates', () => {
        state.applyLocalModelsResult(
            parseModelsResult({
                models: ['safe'],
                modelsByEngine: {
                    ollama: ['safe'],
                    llamacpp: []
                },
                downloadState: localSnapshot('1')
            })
        )
        state.applyLocalEngineProgress({
            engine: 'llamacpp',
            op: 'pull',
            id: '1'.repeat(32),
            model: 'catalog/model',
            downloadState: localSnapshot('2', [entry({ percent: 42 })])
        })
        expect(
            state
                .getEngineInitialState()
                .models.find(models => models.nodeId === nodeId && models.engineType === 'ollama')
                ?.models[0]?.downloaded
        ).toBe(true)
    })

    it('preserves failed and unknown operation rows independently of inventory membership', () => {
        state.applyLocalModelsResult(
            localModels('2', [entry({ phase: 'failed', reason: 'download_failed' })])
        )
        expect(rows()[0]?.downloaded).toBe(false)
        state.applyLocalModelsResult(
            localModels('3', [entry({ phase: 'failed', reason: 'busy', blocksReadiness: false })])
        )
        expect(rows()[0]?.downloaded).toBe(true)
        expect(
            isEnginePullInProgress(
                state
                    .getEngineInitialState()
                    .activeProgress.find(progress => progress.nodeId === nodeId)
            )
        ).toBe(false)
    })

    it('revokes readiness on disconnect and restores it from a same-revision reconnect snapshot', () => {
        const result = localModels('1', [])
        state.applyLocalModelsResult(result)
        expect(rows()[0]?.downloaded).toBe(true)
        state.invalidateModelObservations()
        expect(rows()[0]).toMatchObject({
            name: 'catalog/model',
            downloaded: false,
            downloadStateCurrent: false
        })
        state.applyLocalModelsResult(result)
        expect(rows()[0]?.downloaded).toBe(true)
    })

    it('rejects stale RPC inventory after a newer notification and hydrates a refresh from target facts', () => {
        const sequence = state.beginLocalModelsRead()
        state.applyLocalEngineProgress({
            engine: 'llamacpp',
            op: 'pull',
            id: '1'.repeat(32),
            model: 'catalog/model',
            downloadState: localSnapshot('3')
        })
        state.applyLocalModelsResult(localModels('2', [], ['stale']), sequence)
        expect(rows().map(row => row.name)).toEqual(['catalog/model'])
        expect(
            state
                .getEngineInitialState()
                .activeProgress.find(progress => progress.nodeId === nodeId)?.downloadState?.phase
        ).toBe('in_progress')
    })

    it('does not let a late equal-revision inventory or failed read replace a newer full notification', () => {
        const sequence = state.beginLocalModelsRead()
        state.applyLocalModelsResult(localModels('0', [], ['fresh']))
        state.applyLocalModelsResult(localModels('0', [], ['stale']), sequence)
        state.invalidateLocalModels('llama-cpp', sequence)
        expect(rows()).toMatchObject([
            { name: 'fresh', downloaded: true, downloadStateCurrent: true }
        ])
    })

    it('requires a current read before an unseen notification source can replace the hydrated baseline', () => {
        state.applyLocalModelsResult(localModels('1', [], ['safe']))
        const unseen = snapshot(
            '999',
            [entry({ phase: 'complete', blocksReadiness: false })],
            'b'.repeat(32)
        )
        expect(
            state.applyLocalEngineProgress({
                engine: 'llamacpp',
                op: 'pull',
                id: '1'.repeat(32),
                model: 'catalog/model',
                downloadState: unseen
            })
        ).toBe(true)
        expect(rows()).toMatchObject([{ name: 'safe', downloaded: false }])
        expect(
            state
                .getEngineInitialState()
                .models.find(
                    models => models.nodeId === nodeId && models.engineType === 'llama-cpp'
                )?.downloadState?.sourceId
        ).toBe(source)
        const notification = parseModelsResult({
            models: ['foreign'],
            modelsByEngine: { llamacpp: ['foreign'] },
            downloadState: unseen
        })
        expect(state.applyLocalModelsNotification(notification)).toBe(true)
        expect(rows().map(row => row.name)).toEqual(['safe'])
        state.applyLocalModelsResult(localModels('1', [], ['safe']))
        expect(rows()[0]?.downloaded).toBe(true)
        state.applyLocalModelsResult(notification, state.beginLocalModelsRead())
        expect(rows().map(row => row.name)).toEqual(['foreign', 'catalog/model'])
    })

    it('does not turn malformed, unsupported or recovery-required state into ready inventory', () => {
        state.applyLocalModelsResult(
            parseModelsResult({
                models: ['catalog/model'],
                modelsByEngine: { llamacpp: ['catalog/model'] }
            })
        )
        expect(rows()[0]?.downloaded).toBe(false)
        state.applyLocalModelsResult(
            parseModelsResult({
                models: ['catalog/model'],
                modelsByEngine: { llamacpp: ['catalog/model'] },
                downloadState: { ...localSnapshot('2', []), recoveryRequired: true }
            })
        )
        expect(rows()[0]?.downloaded).toBe(false)
    })

    it('reads canonical inventory through engine:models rather than a raw list action', async () => {
        const supervisor = getModularSupervisor()
        vi.spyOn(supervisor, 'hasProcess').mockReturnValue(true)
        const call = vi.spyOn(supervisor, 'callProcess').mockResolvedValue({
            models: ['canonical'],
            modelsByEngine: { llamacpp: ['canonical'] },
            downloadState: localSnapshot('1', [])
        })
        supervisor.refreshDiscoveryEngineModels('llama-cpp')
        await vi.waitFor(() => expect(rows()[0]?.name).toBe('canonical'))
        expect(call).toHaveBeenCalledExactlyOnceWith('broker', 'engine:models')
    })
})

describe('observer discovery download projection', () => {
    it('hydrates exact lifecycle rows without origin state and retains stale facts without readiness', () => {
        const remoteId = 'download-observer-target'
        const wire = snapshot('5', [entry({ percent: 21 })], 'f'.repeat(32))
        function update(current: boolean, downloadState: JsonObject = wire): void {
            state.handleNotification({
                source: 'broker',
                method: 'discovery:nodes-changed',
                params: {
                    nodes: [
                        {
                            hostUuid: remoteId,
                            name: 'target',
                            ipAddress: '127.0.0.2',
                            port: 1234,
                            models: ['catalog/model'],
                            modelsByEngine: { llamacpp: ['catalog/model'] },
                            loadedByEngine: { llamacpp: ['catalog/model'] },
                            downloadState,
                            downloadStateCurrent: current
                        }
                    ]
                }
            })
        }
        function remoteRows() {
            return (
                state
                    .getEngineInitialState()
                    .models.find(
                        models => models.nodeId === remoteId && models.engineType === 'llama-cpp'
                    )?.models ?? []
            )
        }
        update(true)
        expect(remoteRows()[0]).toMatchObject({
            name: 'catalog/model',
            downloaded: false,
            status: 'loaded',
            downloadStateCurrent: true,
            downloadState: { percent: 21 }
        })
        update(false)
        expect(remoteRows()[0]?.downloadStateCurrent).toBe(false)
        const complete = snapshot(
            '6',
            [entry({ phase: 'complete', blocksReadiness: false })],
            'f'.repeat(32)
        )
        update(true, complete)
        expect(remoteRows()[0]?.downloaded).toBe(true)
        update(true, wire)
        expect(remoteRows()[0]?.downloadState?.phase).toBe('complete')
    })
})
