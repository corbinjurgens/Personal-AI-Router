// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import fs from 'fs'
import os from 'os'
import {
    modelCopyProgress,
    parseModelCopyFrame,
    parseModelCopyResult,
    parseModelCopyCapabilities,
    parseActiveModelCopies,
    isModelCopyTerminal
} from '@/electron/service-bridge/model-copy'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import { emitBridgePush } from '@/electron/service-bridge/broadcaster'
import { publishPush, subscribePush } from '@/electron/service-bridge/push-bus'
import { handleServiceBridgeInvoke } from '@/electron/service-bridge/empty-handlers'
import {
    JsonRpcResponseError,
    JsonRpcSocketClient,
    type JsonValue
} from '@/electron/service-bridge/json-rpc-client'
import { initPlatform } from '@/electron/globals'
import { engineProgressKey, isEngineCopyInProgress } from '@/shared/utils/engine-progress'
import type { ModelCopyRequest } from '@/shared/types/model-copy'
import type { EngineCommandPayload } from '@/shared/types/engine-api'
import type { createOverviewWindow } from '@/electron/window'
import { createEngineApi } from '@/ui/api/engine-api'
import { createPairApi } from '@/ui/api/pair-api'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { serviceTransport } from '@/ui/api/service-transport'
import type { PreloadServiceTransport, ServiceTransport } from '@/shared/types/service-bridge'
import { formatCopyProgressLabel } from '@/ui/utils/formatters'

vi.mock('electron', () => ({ app: { isPackaged: false, getAppPath: () => process.cwd() } }))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<typeof createOverviewWindow>() }))
vi.mock('@/electron/config/ui-config', () => ({ isFirstRun: () => false }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({
    emitBridgePush: vi.fn<typeof emitBridgePush>()
}))
vi.mock('@/electron/service-bridge/node-info-poller', () => ({
    startNodeInfoPoller: vi.fn<() => void>(),
    stopNodeInfoPoller: vi.fn<() => void>()
}))

const request: ModelCopyRequest = {
    opId: 'a'.repeat(32),
    sourceNodeId: 'MSI',
    destinationNodeId: 'Joes',
    engine: 'llamacpp',
    model: 'tiel-coder-35b-a3b-msi-ctx32768'
}
const exportId = 'b'.repeat(64)
const result = {
    engine: request.engine,
    model: request.model,
    files: 2,
    filesSkipped: 0,
    bytesTotal: 19,
    bytesCopied: 19,
    layout: 'llama-artifacts-v1',
    exportId,
    durability: 'uncertain',
    requiresConfiguration: true
}
function frame(overrides: Record<string, JsonValue> = {}): JsonValue {
    return {
        ...request,
        node: request.sourceNodeId,
        op: 'copy',
        stage: 'downloading',
        percent: 42,
        bytesDone: 8,
        bytesTotal: 19,
        exportId,
        file: 'model/part.gguf',
        message: 'Copying',
        ...overrides
    }
}
function deferred<T>() {
    let resolve: (value: T) => void = () => {
        throw new Error('Uninitialized deferred')
    }
    let reject: (error: Error) => void = () => {
        throw new Error('Uninitialized deferred')
    }
    const promise = new Promise<T>((done, fail) => {
        resolve = done
        reject = fail
    })
    return { promise, resolve, reject }
}
async function flush(): Promise<void> {
    for (let i = 0; i < 100; i++) await Promise.resolve()
}
const state = getModularBridgeState()
const supervisor = getModularSupervisor()
let attempt = 100
beforeEach(() => {
    request.opId = (++attempt).toString(16).padStart(32, 'a')
})
afterEach(() => {
    state.reconcileModelCopies([])
    vi.restoreAllMocks()
})

describe('copy wire boundary', () => {
    it('formats copy phase and measured percent independently of the destination', () => {
        expect(formatCopyProgressLabel({ status: 'starting' })).toBe('Copy · starting')
        expect(formatCopyProgressLabel({ status: 'downloading', percent: 41.6 })).toBe(
            'Copy · downloading · 42%'
        )
    })
    it('accepts fractional progress and caps measured percentages without inventing unknown percent', () => {
        expect(parseModelCopyFrame(frame({ percent: 42.5 }))?.percent).toBe(42.5)
        expect(parseModelCopyFrame(frame({ percent: 140 }))?.percent).toBe(100)
        expect(parseModelCopyFrame(frame({ percent: -1 }))?.percent).toBeUndefined()
    })
    it('preserves the complete identity and attributes progress to the destination', () => {
        const parsed = parseModelCopyFrame(frame())
        if (!parsed) throw new Error('Valid copy frame rejected')
        expect(parsed).toMatchObject({ ...request, node: 'MSI', exportId, bytesDone: 8 })
        expect(modelCopyProgress(parsed, 'Joes')).toMatchObject({
            nodeId: 'Joes',
            sourceNodeId: 'MSI',
            destinationNodeId: 'Joes',
            opId: request.opId,
            engineType: 'llama-cpp',
            model: request.model,
            completed: 8,
            total: 19,
            exportId,
            file: 'model/part.gguf',
            message: 'Copying'
        })
    })
    const invalidFrames: [string, Record<string, JsonValue>][] = [
        ['wrong operation', { op: 'pull' }],
        ['missing identity', { opId: '' }],
        ['wrong source', { node: 'other' }],
        ['same endpoints', { destinationNodeId: 'MSI' }],
        ['missing model', { model: '' }],
        ['invalid engine', { engine: '../bad' }]
    ]
    it.each(invalidFrames)('rejects %s without source-only backfill', (_label, override) => {
        expect(parseModelCopyFrame(frame(override))).toBeNull()
    })
    it('retains typed setup and durability qualification with terminal byte counts', () => {
        expect(parseModelCopyResult({ opId: request.opId, result }, request)).toEqual({
            opId: request.opId,
            result
        })
    })
    it('accepts verified content when retry traffic exceeds the artifact byte total', () => {
        const retried = { ...result, bytesTotal: 100, bytesCopied: 150 }
        expect(
            parseModelCopyResult({ opId: request.opId, result: retried }, request).result
        ).toEqual(retried)
    })
    const invalidByteCounts: [string, Record<string, number>][] = [
        ['zero artifact count', { files: 0 }],
        ['negative traffic', { bytesCopied: -1 }],
        ['fractional traffic', { bytesCopied: 1.5 }],
        ['negative artifact total', { bytesTotal: -1 }],
        ['more skipped files than files', { filesSkipped: result.files + 1 }]
    ]
    it.each(invalidByteCounts)('still rejects %s', (_label, values) => {
        expect(() =>
            parseModelCopyResult({ opId: request.opId, result: { ...result, ...values } }, request)
        ).toThrow()
    })
    it('recognizes explicit uncertainty as a terminal wire stage', () => {
        expect(isModelCopyTerminal('unknown')).toBe(true)
        expect(parseModelCopyFrame(frame({ stage: 'unknown' }))?.stage).toBe('unknown')
    })
    const invalidResults: [string, JsonValue][] = [
        ['wrong operation', { opId: 'c'.repeat(32), result }],
        ['wrong engine', { opId: request.opId, result: { ...result, engine: 'ollama' } }],
        ['wrong model', { opId: request.opId, result: { ...result, model: 'other' } }],
        ['invalid durability', { opId: request.opId, result: { ...result, durability: 'yes' } }],
        ['no result', { opId: request.opId }]
    ]
    it.each(invalidResults)('refuses %s as a successful result', (_label, value) => {
        expect(() => parseModelCopyResult(value, request)).toThrow()
    })
    it('keeps unavailable and uninstalled capability facts without granting readiness', () => {
        const capability = {
            version: 1,
            engine: 'llamacpp',
            available: false,
            installed: false,
            modelStore: '',
            exportLayouts: [],
            importLayouts: [],
            writable: false,
            reason: 'Unsupported on this device',
            identityBoundRetry: false,
            receiptDurability: false
        }
        expect(parseModelCopyCapabilities(capability, 'llamacpp')).toEqual(capability)
        expect(() => parseModelCopyCapabilities({}, 'llamacpp')).toThrow()
        expect(() =>
            parseModelCopyCapabilities({ ...capability, version: 2 }, 'llamacpp')
        ).toThrow()
        expect(() =>
            parseModelCopyCapabilities({ ...capability, engine: 'ollama' }, 'llamacpp')
        ).toThrow()
    })
    it('validates snapshot tuples and refuses mismatched nested progress', () => {
        expect(
            parseActiveModelCopies({ operations: [{ ...request, progress: frame() }] })
        ).toHaveLength(1)
        expect(() =>
            parseActiveModelCopies({
                operations: [{ ...request, progress: frame({ destinationNodeId: 'other' }) }]
            })
        ).toThrow()
        expect(() => parseActiveModelCopies({})).toThrow()
    })
    it('retains identity-bound retries and refuses a rebound export result', () => {
        const retry = { ...request, expectedExportId: exportId }
        expect(parseActiveModelCopies({ operations: [retry] })).toEqual([retry])
        expect(() =>
            parseModelCopyResult(
                { opId: request.opId, result: { ...result, exportId: 'c'.repeat(64) } },
                retry
            )
        ).toThrow()
    })
    it('rejects duplicate or excessive active operations before state mutation', () => {
        expect(() =>
            parseActiveModelCopies({ operations: [{ ...request }, { ...request }] })
        ).toThrow()
        expect(() =>
            parseActiveModelCopies({
                operations: Array.from({ length: 9 }, (_, i) => ({
                    ...request,
                    opId: i.toString(16).padStart(32, '0')
                }))
            })
        ).toThrow()
    })
    it('keeps ordinary layout results without inventing custom readiness qualifications', () => {
        const ordinary = {
            engine: request.engine,
            model: request.model,
            files: 1,
            filesSkipped: 1,
            bytesTotal: 10,
            bytesCopied: 0
        }
        expect(
            parseModelCopyResult({ opId: request.opId, result: ordinary }, request).result
        ).toMatchObject(ordinary)
        expect(
            parseModelCopyResult({ opId: request.opId, result: ordinary }, request).result
                .requiresConfiguration
        ).toBeUndefined()
    })
})

function publisherRecord(overrides: Record<string, JsonValue> = {}) {
    return {
        path: 'model/tiny.gguf',
        status: 'verified',
        provider: 'huggingface',
        repository: 'owner/tiny-GGUF',
        revision: 'a'.repeat(40),
        file: 'tiny-Q4_K_M.gguf',
        algorithm: 'sha256',
        expectedDigest: 'b'.repeat(64),
        actualDigest: 'b'.repeat(64),
        acquisitionBasis: 'cache',
        acquiredAt: '2026-01-01T00:00:00Z',
        ...overrides
    }
}

describe('publisher comparison boundary', () => {
    it('preserves exact provenance through result decoding and bridge snapshots', () => {
        const publisherComparison = {
            status: 'verified',
            files: [publisherRecord(), publisherRecord({ path: 'draft/tiny.gguf' })],
            replayed: true
        }
        const response = parseModelCopyResult(
            {
                opId: request.opId,
                result: { ...result, publisherComparison }
            },
            request
        )
        expect(response.result.publisherComparison).toEqual(publisherComparison)
        state.beginModelCopy(request)
        state.finishModelCopy(request, 'done', response.result)
        expect(
            state.getEngineInitialState().activeProgress.find(row => row.opId === request.opId)
        ).toMatchObject({ status: 'done', copyResult: { publisherComparison } })
    })

    it('does not infer a comparison from a successful transfer with missing optional fields', () => {
        expect(
            parseModelCopyResult({ opId: request.opId, result }, request).result.publisherComparison
        ).toBeUndefined()
    })

    const invalidPublisherRecords: [string, Record<string, JsonValue>][] = [
        ['missing actual full digest', { actualDigest: '' }],
        ['truncated digest', { actualDigest: 'abc' }],
        ['uppercase digest', { actualDigest: 'B'.repeat(64) }],
        ['verified disagreement', { expectedDigest: 'c'.repeat(64) }],
        ['mismatch agreement', { status: 'mismatch' }],
        ['unrecognized status', { status: 'trusted' }],
        ['unknown authority field', { trusted: true }],
        ['provider URL', { provider: 'https://example.invalid' }],
        ['request URL field', { url: 'https://example.invalid' }],
        ['mutable revision', { revision: 'main' }],
        ['absolute file', { file: '/tmp/file.gguf' }],
        ['path traversal', { file: '../file.gguf' }],
        ['oversize path', { path: 'a'.repeat(1025) }],
        ['control path', { path: 'bad\u0000file' }],
        ['lone surrogate', { path: 'bad\ud800file' }],
        ['missing acquisition basis', { acquisitionBasis: '' }],
        ['unparseable time', { acquiredAt: 'yesterday' }],
        ['normalized invalid day', { acquiredAt: '2026-02-30T00:00:00Z' }],
        ['future time', { acquiredAt: '9999-01-01T00:00:00Z' }],
        ['null time', { acquiredAt: null }]
    ]
    it.each(invalidPublisherRecords)('rejects %s', (_label, changes) => {
        expect(() =>
            parseModelCopyResult(
                {
                    opId: request.opId,
                    result: {
                        ...result,
                        files: 1,
                        publisherComparison: {
                            status: 'verified',
                            files: [publisherRecord(changes)]
                        }
                    }
                },
                request
            )
        ).toThrow('Invalid model copy response')
    })

    it('aggregates a missing companion as unavailable and retains visible mismatches', () => {
        const unavailable = {
            path: 'draft/tiny.gguf',
            status: 'unavailable',
            algorithm: 'sha256',
            actualDigest: 'c'.repeat(64),
            acquisitionBasis: 'unavailable'
        }
        const parse = (status: string, files: JsonValue[]) =>
            parseModelCopyResult(
                {
                    opId: request.opId,
                    result: { ...result, publisherComparison: { status, files } }
                },
                request
            ).result.publisherComparison
        expect(parse('unavailable', [publisherRecord(), unavailable])?.status).toBe('unavailable')
        expect(
            parse('mismatch', [
                publisherRecord({ status: 'mismatch', expectedDigest: 'd'.repeat(64) }),
                unavailable
            ])?.status
        ).toBe('mismatch')
        expect(() => parse('verified', [publisherRecord(), unavailable])).toThrow()
        expect(() => parse('verified', [publisherRecord()])).toThrow()
        expect(() => parse('verified', [publisherRecord(), publisherRecord()])).toThrow()
    })

    it('rejects oversized observations and null optional records', () => {
        expect(() =>
            parseModelCopyResult(
                {
                    opId: request.opId,
                    result: {
                        ...result,
                        publisherComparison: {
                            status: 'verified',
                            files: Array.from({ length: 1025 }, () => publisherRecord())
                        }
                    }
                },
                request
            )
        ).toThrow()
        expect(() =>
            parseModelCopyResult(
                {
                    opId: request.opId,
                    result: {
                        ...result,
                        publisherComparison: null
                    }
                },
                request
            )
        ).toThrow()
    })

    it('relays mismatch as a successful result through the supervisor without new control operations', async () => {
        state.setSelfId('Mac')
        const publisherComparison = {
            status: 'mismatch',
            files: [publisherRecord({ status: 'mismatch', expectedDigest: 'd'.repeat(64) })]
        }
        const observed = { ...result, files: 1, publisherComparison }
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockImplementation(async (_name, method): Promise<JsonValue> => {
                if (method === 'engine:copy-model') return { opId: request.opId, result: observed }
                if (method === 'engine:remote-get-installed') return { engines: [] }
                if (method === 'discovery:get-nodes') return { nodes: [] }
                throw new Error(`Unexpected operation ${method}`)
            })
        await expect(supervisor.copyModel(request)).resolves.toMatchObject({ result: observed })
        const outcome = state
            .getEngineInitialState()
            .activeProgress.find(row => row.opId === request.opId)
        expect(outcome).toMatchObject({ status: 'done', copyResult: { publisherComparison } })
        expect(outcome?.message).toContain('Publisher comparison: mismatch.')
        expect(call.mock.calls.map(([, method]) => method)).toEqual([
            'engine:copy-model',
            'engine:remote-get-installed',
            'discovery:get-nodes'
        ])
    })
})

describe('copy state', () => {
    it('isolates simultaneous destinations and rejects unknown terminal or mismatched tuple progress', () => {
        const other = { ...request, opId: 'c'.repeat(32), destinationNodeId: 'PC3' }
        state.beginModelCopy(request)
        state.beginModelCopy(other)
        vi.mocked(emitBridgePush).mockClear()
        state.applyRemoteEngineProgress(frame({ opId: 'd'.repeat(32), stage: 'done' }))
        state.applyRemoteEngineProgress(frame({ destinationNodeId: 'PC3' }))
        expect(emitBridgePush).not.toHaveBeenCalled()
        state.applyRemoteEngineProgress(frame())
        state.applyRemoteEngineProgress(frame({ ...other, bytesDone: 3 }))
        const copies = state
            .getEngineInitialState()
            .activeProgress.filter(p => p.operation === 'copy' && isEngineCopyInProgress(p))
        expect(copies).toHaveLength(2)
        expect(new Set(copies.map(engineProgressKey)).size).toBe(2)
        expect(copies.map(p => [p.nodeId, p.sourceNodeId, p.completed])).toEqual([
            ['Joes', 'MSI', 8],
            ['PC3', 'MSI', 3]
        ])
    })
    it('hydrates actual active operations and reports missing outcomes as unknown', () => {
        state.beginModelCopy(request)
        state.reconcileModelCopies([
            { ...request, progress: parseModelCopyFrame(frame()) ?? undefined }
        ])
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'downloading', exportId })
        state.reconcileModelCopies([])
        const outcome = state
            .getEngineInitialState()
            .activeProgress.find(p => p.opId === request.opId)
        expect(outcome).toMatchObject({ status: 'unknown', exportId })
        expect(outcome?.copyResult).toBeUndefined()
        expect(isEngineCopyInProgress(outcome)).toBe(false)
    })
    it('reactivates an unknown operation from a real snapshot without requiring a progress frame', () => {
        const retry = { ...request, expectedExportId: exportId }
        state.beginModelCopy(retry)
        state.reconcileModelCopies([])
        state.reconcileModelCopies([retry])
        const progress = state
            .getEngineInitialState()
            .activeProgress.find(p => p.opId === request.opId)
        expect(progress).toMatchObject({ status: 'starting', expectedExportId: exportId })
        expect(isEngineCopyInProgress(progress)).toBe(true)
    })
    it('bounds active entries and coalesces repeated progress', () => {
        for (let i = 0; i < 8; i++)
            state.beginModelCopy({ ...request, opId: i.toString(16).padStart(32, '0') })
        expect(() => state.beginModelCopy({ ...request, opId: 'e'.repeat(32) })).toThrow()
        const opId = '0'.repeat(32)
        for (let i = 0; i < 20; i++) state.applyRemoteEngineProgress(frame({ opId, bytesDone: i }))
        expect(
            state.getEngineInitialState().activeProgress.filter(p => p.opId === opId)
        ).toHaveLength(1)
    })
    it('keeps the active bound when an older snapshot races a newer local admission', () => {
        const older = Array.from({ length: 8 }, (_, i) => ({
            ...request,
            opId: i.toString(16).padStart(32, 'd')
        }))
        state.reconcileModelCopies(older)
        state.reconcileModelCopies([])
        const baseline = state.beginModelCopyRead()
        state.beginModelCopy(request)
        state.reconcileModelCopies(older, baseline)
        expect(
            state.getEngineInitialState().activeProgress.filter(isEngineCopyInProgress)
        ).toHaveLength(8)
    })
    it('does not fabricate success from a terminal notification without the awaited result', () => {
        state.beginModelCopy(request)
        state.applyRemoteEngineProgress(frame({ stage: 'done', bytesDone: 19 }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'awaiting-result' })
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
                ?.copyResult
        ).toBeUndefined()
    })
    it('settles detached terminal notifications as unknown and keeps the learned export identity', () => {
        state.reconcileModelCopies(
            parseActiveModelCopies({ operations: [{ ...request, progress: frame() }] })
        )
        state.applyRemoteEngineProgress(frame({ stage: 'done', bytesDone: 19 }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'unknown', exportId, completed: 19 })
    })
    it('preserves newer progress and completion across an older active snapshot', () => {
        state.beginModelCopy(request)
        const baseline = state.beginModelCopyRead()
        state.applyRemoteEngineProgress(frame())
        state.reconcileModelCopies([], baseline)
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)?.status
        ).toBe('downloading')
        state.finishModelCopy(
            request,
            'done',
            parseModelCopyResult({ opId: request.opId, result }, request).result
        )
        state.reconcileModelCopies([{ ...request }], baseline)
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)?.status
        ).toBe('done')
    })
    it('rejects late progress carrying a different export or model without altering the copy', () => {
        state.beginModelCopy(request)
        state.applyRemoteEngineProgress(frame())
        state.applyRemoteEngineProgress(frame({ exportId: 'f'.repeat(64), bytesDone: 18 }))
        state.applyRemoteEngineProgress(frame({ model: 'other', bytesDone: 18 }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ exportId, completed: 8 })
    })
    it('retains bounded last outcomes rather than an unrestricted terminal history', () => {
        for (let i = 0; i < 24; i++) {
            const operation = { ...request, opId: i.toString(16).padStart(32, 'f') }
            state.beginModelCopy(operation)
            state.finishModelCopy(operation, 'error', undefined, 'Synthetic conflict')
        }
        expect(
            state.getEngineInitialState().activeProgress.filter(p => p.operation === 'copy')
        ).toHaveLength(8)
    })
    it('keeps ordinary pull keys and progress independent from copy outcomes', () => {
        const key = engineProgressKey({
            nodeId: 'Joes',
            engineType: 'ollama',
            operation: 'pull',
            model: 'qwen:4b'
        })
        state.beginRemoteModelPull('Joes', 'ollama', 'qwen:4b')
        state.beginModelCopy(request)
        state.finishModelCopy(request, 'unknown')
        expect(key).toBe('Joes:ollama:pull:qwen:4b')
        expect(
            state.getEngineInitialState().activeProgress.some(p => engineProgressKey(p) === key)
        ).toBe(true)
        state.finishRemoteModelPull('Joes', 'ollama', 'qwen:4b')
    })
})

describe('bridge copy outcomes in the renderer', () => {
    beforeEach(async () => {
        const api = createPairApi({
            connected: true,
            invoke: vi.fn<ServiceTransport['invoke']>(),
            subscribePush: () => () => {},
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        })
        api.engines.onProgress = listener =>
            subscribePush(event => {
                if (event.channel === 'engines:progress-changed') listener(event.payload)
            })
        vi.stubGlobal('window', { pairApi: api })
        vi.mocked(emitBridgePush).mockImplementation(publishPush)
        useEngineProgressStore.setState({ progress: new Map() })
        await useEngineProgressStore.getState().initialize({
            descriptors: [],
            statuses: [],
            models: [],
            activeProgress: [],
            updateAvailable: []
        })
    })
    afterEach(() => {
        useEngineProgressStore.getState().cleanup()
        vi.unstubAllGlobals()
    })

    it.each(['hydrated', 'live observed'])(
        '%s completion publishes only an unknown outcome without its verified reply',
        origin => {
            if (origin === 'hydrated') state.reconcileModelCopies([{ ...request }])
            state.applyRemoteEngineProgress(frame())
            vi.mocked(emitBridgePush).mockClear()
            state.applyRemoteEngineProgress(frame({ stage: 'done', bytesDone: 19 }))
            const bridge = state
                .getEngineInitialState()
                .activeProgress.find(p => p.opId === request.opId)
            if (!bridge) throw new Error('Missing bridge copy outcome')
            expect(bridge).toMatchObject({ status: 'unknown', exportId, completed: 19, total: 19 })
            expect(bridge.copyResult).toBeUndefined()
            expect(bridge.message).toContain('did not receive its verified result')
            expect(
                useEngineProgressStore.getState().progress.get(engineProgressKey(bridge))
            ).toEqual(bridge)
            expect(
                vi
                    .mocked(emitBridgePush)
                    .mock.calls.filter(([channel]) => channel === 'engines:progress-changed')
            ).toEqual([['engines:progress-changed', bridge]])
        }
    )

    it('keeps a verified awaited completion immutable across late progress and renderer hydration', async () => {
        const pending = deferred<JsonValue>()
        vi.spyOn(supervisor, 'callProcess').mockImplementation(async (_name, method) => {
            if (method === 'engine:copy-model') return pending.promise
            if (method === 'engine:remote-get-installed') return { engines: [] }
            if (method === 'discovery:get-nodes') return { nodes: [] }
            throw new Error(`Unexpected call ${method}`)
        })
        const copying = supervisor.copyModel(request)
        state.applyRemoteEngineProgress(frame({ stage: 'done', bytesDone: 19 }))
        expect(
            Array.from(useEngineProgressStore.getState().progress.values()).find(
                p => p.opId === request.opId
            )?.status
        ).toBe('awaiting-result')
        pending.resolve({ opId: request.opId, result })
        await copying
        const bridge = state
            .getEngineInitialState()
            .activeProgress.find(p => p.opId === request.opId)
        if (!bridge) throw new Error('Missing verified copy outcome')
        expect(bridge).toMatchObject({ status: 'done', copyResult: result })
        expect(bridge.message).toContain('serving configuration')
        expect(bridge.message).toContain('durability could not be confirmed')
        state.applyRemoteEngineProgress(frame({ stage: 'unknown' }))
        state.applyRemoteEngineProgress(frame({ bytesDone: 1 }))
        publishPush('engines:progress-changed', {
            ...bridge,
            status: 'unknown',
            copyResult: undefined
        })
        await useEngineProgressStore.getState().initialize({
            descriptors: [],
            statuses: [],
            models: [],
            activeProgress: [{ ...bridge, status: 'downloading', copyResult: undefined }],
            updateAvailable: []
        })
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toEqual(bridge)
        expect(useEngineProgressStore.getState().progress.get(engineProgressKey(bridge))).toEqual(
            bridge
        )
    })
})

describe('copy supervisor and typed API', () => {
    it('refreshes ordinary-layout inventory on the actual destination without adding setup claims', async () => {
        state.setSelfId('Mac')
        const ordinary = {
            engine: request.engine,
            model: request.model,
            files: 1,
            filesSkipped: 0,
            bytesTotal: 19,
            bytesCopied: 19
        }
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockImplementation(async (_name, method): Promise<JsonValue> => {
                if (method === 'engine:copy-model') return { opId: request.opId, result: ordinary }
                if (method === 'engine:remote-get-installed') return { engines: [] }
                if (method === 'discovery:get-nodes')
                    return {
                        nodes: [
                            {
                                hostUuid: 'Joes',
                                name: 'Joes',
                                models: ['ordinary-native-model'],
                                modelsByEngine: { llamacpp: ['ordinary-native-model'] }
                            }
                        ]
                    }
                throw new Error(`Unexpected controller operation ${method}`)
            })
        await supervisor.copyModel(request)
        expect(call).toHaveBeenCalledWith('broker', 'engine:remote-get-installed', { node: 'Joes' })
        expect(
            state
                .getEngineInitialState()
                .models.find(row => row.nodeId === 'Joes' && row.engineType === 'llama-cpp')
                ?.models.map(row => row.name)
        ).toEqual(['ordinary-native-model'])
        const outcome = state
            .getEngineInitialState()
            .activeProgress.find(row => row.opId === request.opId)
        expect(outcome).toMatchObject({ status: 'done', copyResult: ordinary })
        expect(outcome?.message).not.toContain('serving configuration')
        expect(
            call.mock.calls.some(
                ([, method, params]) =>
                    method === 'engine:status' ||
                    (params &&
                        typeof params === 'object' &&
                        !Array.isArray(params) &&
                        params.node === 'Mac')
            )
        ).toBe(false)
    })
    it('refreshes the destination after a verified retried copy and completes artifact progress at its byte total', async () => {
        state.setSelfId('Mac')
        const retried = { ...result, bytesTotal: 100, bytesCopied: 150 }
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockImplementation(async (_name, method): Promise<JsonValue> => {
                if (method === 'engine:copy-model') return { opId: request.opId, result: retried }
                if (method === 'engine:remote-get-installed') return { engines: [] }
                if (method === 'discovery:get-nodes') return { nodes: [] }
                throw new Error(`Unexpected call ${method}`)
            })
        await expect(supervisor.copyModel(request)).resolves.toMatchObject({ result: retried })
        expect(call).toHaveBeenCalledWith('broker', 'engine:remote-get-installed', { node: 'Joes' })
        expect(call).toHaveBeenCalledWith('broker', 'discovery:get-nodes')
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({
            status: 'done',
            nodeId: 'Joes',
            completed: 100,
            total: 100,
            copyResult: { bytesCopied: 150, bytesTotal: 100 }
        })
    })
    it('waits for the real response without a pull timeout and refreshes only the destination', async () => {
        state.setSelfId('Mac')
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockImplementation(async (_name, method): Promise<JsonValue> => {
                if (method === 'engine:copy-model') return { opId: request.opId, result }
                if (method === 'engine:remote-get-installed') return { engines: [] }
                if (method === 'discovery:get-nodes') return { nodes: [] }
                throw new Error(`Unexpected call ${method}`)
            })
        expect(await handleServiceBridgeInvoke('engine:copy-model', request)).toEqual({
            opId: request.opId,
            result
        })
        expect(call).toHaveBeenCalledWith('broker', 'engine:copy-model', request, null)
        expect(call).toHaveBeenCalledWith('broker', 'engine:remote-get-installed', { node: 'Joes' })
        expect(call).not.toHaveBeenCalledWith('broker', 'engine:action', expect.anything())
        const completed = state
            .getEngineInitialState()
            .activeProgress.find(p => p.opId === request.opId)
        expect(completed).toMatchObject({ status: 'done', nodeId: 'Joes', copyResult: result })
        expect(completed?.message).toContain('serving configuration')
        expect(completed?.message).toContain('durability could not be confirmed')
        expect(
            state
                .getEngineInitialState()
                .models.flatMap(m => m.models)
                .some(m => m.name === request.model)
        ).toBe(false)
    })
    it('retains an unknown outcome on transport loss without refreshing inventories', async () => {
        vi.spyOn(supervisor, 'callProcess').mockRejectedValue(
            new Error('Connection closed before replying')
        )
        await expect(supervisor.copyModel(request)).rejects.toThrow('Connection closed')
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'unknown' })
        expect(supervisor.callProcess).toHaveBeenCalledTimes(1)
    })
    it('preserves an RPC stream-loss code while leaving its copy outcome unknown', async () => {
        vi.spyOn(supervisor, 'callProcess').mockRejectedValue(
            new JsonRpcResponseError(-32000, 'copy stream ended before terminal result')
        )
        await expect(supervisor.copyModel(request)).rejects.toMatchObject({ code: -32000 })
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'unknown' })
    })
    it('uses a matching terminal failure to distinguish a definite failure from stream uncertainty', async () => {
        vi.spyOn(supervisor, 'callProcess').mockImplementation(async () => {
            state.applyRemoteEngineProgress(
                frame({ stage: 'error', message: 'Source binding changed' })
            )
            throw new JsonRpcResponseError(-32000, 'Source binding changed')
        })
        await expect(supervisor.copyModel(request)).rejects.toMatchObject({ code: -32000 })
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'error', message: 'Source binding changed' })
    })
    it('relays only canonical control fields and performs no controller file I/O', async () => {
        state.setSelfId('Mac')
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockImplementation(async (_name, method): Promise<JsonValue> => {
                if (method === 'engine:copy-model') return { opId: request.opId, result }
                return { engines: [], nodes: [] }
            })
        const read = vi.spyOn(fs, 'readFileSync').mockImplementation(() => {
            throw new Error('Controller artifact read')
        })
        const write = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {
            throw new Error('Controller artifact write')
        })
        const open = vi.spyOn(fs, 'openSync').mockImplementation(() => {
            throw new Error('Controller artifact open')
        })
        const untrusted = {
            ...request,
            path: '/synthetic/source.gguf',
            url: 'https://invalid.example/model'
        }
        await expect(
            handleServiceBridgeInvoke('engine:copy-model', untrusted)
        ).resolves.toMatchObject({ result })
        expect(call).toHaveBeenCalledWith('broker', 'engine:copy-model', request, null)
        expect(read).not.toHaveBeenCalled()
        expect(write).not.toHaveBeenCalled()
        expect(open).not.toHaveBeenCalled()
    })
    it('stays pending beyond the old pull duration until the real terminal response', async () => {
        vi.useFakeTimers()
        const pending = deferred<JsonValue>()
        vi.spyOn(supervisor, 'callProcess').mockImplementation((_name, method) =>
            method === 'engine:copy-model'
                ? pending.promise
                : Promise.resolve({ engines: [], nodes: [] })
        )
        let settled = false
        try {
            const copying = supervisor.copyModel(request).then(value => {
                settled = true
                return value
            })
            await vi.advanceTimersByTimeAsync(7 * 60 * 60 * 1000)
            expect(settled).toBe(false)
            expect(
                state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
                    ?.copyResult
            ).toBeUndefined()
            pending.resolve({ opId: request.opId, result })
            expect(await copying).toMatchObject({ result })
        } finally {
            vi.useRealTimers()
        }
    })
    it('updates only the destination from the scanner snapshot and never synthesizes a controller model', async () => {
        state.setSelfId('Mac')
        const snapshot = {
            nodes: [
                {
                    hostUuid: 'Mac',
                    name: 'Mac',
                    models: ['must-not-adopt'],
                    modelsByEngine: { llamacpp: ['must-not-adopt'] }
                },
                {
                    hostUuid: 'Joes',
                    name: 'Joes',
                    models: ['native-model'],
                    modelsByEngine: { llamacpp: ['native-model'] }
                }
            ]
        }
        const inventory = vi.spyOn(state, 'applyModelCopyDestinationInventory')
        vi.spyOn(supervisor, 'callProcess').mockImplementation(
            async (_name, method): Promise<JsonValue> => {
                if (method === 'engine:copy-model') return { opId: request.opId, result }
                if (method === 'discovery:get-nodes') return snapshot
                return { engines: [] }
            }
        )
        vi.mocked(emitBridgePush).mockClear()
        await supervisor.copyModel(request)
        expect(inventory).toHaveBeenCalledWith('Joes', snapshot)
        expect(
            state
                .getEngineInitialState()
                .models.find(m => m.nodeId === 'Joes' && m.engineType === 'llama-cpp')
                ?.models.map(m => m.name)
        ).toEqual(['native-model'])
        expect(
            state
                .getEngineInitialState()
                .models.filter(m => m.nodeId === 'Mac')
                .flatMap(m => m.models)
                .some(m => m.name === 'must-not-adopt')
        ).toBe(false)
    })
    it('rejects a wrong operation result as unknown rather than refreshing or completing', async () => {
        vi.spyOn(supervisor, 'callProcess').mockResolvedValue({ opId: 'b'.repeat(32), result })
        await expect(supervisor.copyModel(request)).rejects.toThrow('Invalid model copy response')
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)?.status
        ).toBe('unknown')
        expect(supervisor.callProcess).toHaveBeenCalledTimes(1)
    })
    it('marks a cancelled copy from its exact terminal frame and eventual RPC rejection', async () => {
        vi.spyOn(supervisor, 'callProcess').mockImplementation(async () => {
            state.applyRemoteEngineProgress(
                frame({ stage: 'cancelled', message: 'Copy cancelled' })
            )
            throw new JsonRpcResponseError(-32000, 'Copy cancelled')
        })
        await expect(supervisor.copyModel(request)).rejects.toThrow('Copy cancelled')
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)?.status
        ).toBe('cancelled')
    })
    it('preserves management denial codes and never treats refusal as success', async () => {
        vi.spyOn(supervisor, 'callProcess').mockRejectedValue(
            new JsonRpcResponseError(-32012, 'remote_management_denied')
        )
        await expect(supervisor.copyModel(request)).rejects.toMatchObject({ code: -32012 })
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'error' })
    })
    it('relays cancellation by exact opId and reads the active Go snapshot', async () => {
        const call = vi
            .spyOn(supervisor, 'callProcess')
            .mockImplementation(async (_name, method): Promise<JsonValue> => {
                if (method === 'engine:cancel-model-copy') return { cancelled: false }
                if (method === 'engine:active-model-copies')
                    return { operations: [{ ...request, progress: frame() }] }
                throw new Error(`Unexpected call ${method}`)
            })
        await expect(
            handleServiceBridgeInvoke('engine:cancel-model-copy', { opId: 'f'.repeat(32) })
        ).resolves.toEqual({ cancelled: false })
        await expect(
            handleServiceBridgeInvoke('engine:active-model-copies', undefined)
        ).resolves.toMatchObject({ operations: [request] })
        expect(call).toHaveBeenCalledWith('broker', 'engine:cancel-model-copy', {
            opId: 'f'.repeat(32)
        })
    })
    it('relays capability facts for the exact endpoint and preserves unsupported method errors', async () => {
        const capability = {
            version: 1,
            engine: 'llamacpp',
            available: true,
            installed: false,
            modelStore: 'llamacpp',
            exportLayouts: ['llama-artifacts-v1'],
            importLayouts: ['llama-artifacts-v1'],
            writable: false,
            reason: 'Engine is not installed',
            identityBoundRetry: true,
            receiptDurability: true
        }
        const call = vi.spyOn(supervisor, 'callProcess').mockResolvedValue(capability)
        await expect(
            handleServiceBridgeInvoke('engine:model-copy-capabilities', {
                nodeId: 'Joes',
                engine: 'llamacpp'
            })
        ).resolves.toEqual(capability)
        expect(call).toHaveBeenCalledWith('broker', 'engine:model-copy-capabilities', {
            nodeId: 'Joes',
            engine: 'llamacpp'
        })
        call.mockRejectedValue(new JsonRpcResponseError(-32601, 'method not found'))
        await expect(
            handleServiceBridgeInvoke('engine:model-copy-capabilities', {
                nodeId: 'Joes',
                engine: 'llamacpp'
            })
        ).rejects.toMatchObject({ code: -32601 })
    })
    it('uses the same canonical coordinator for the public Copy-to-this-PC wrapper', async () => {
        state.setSelfId('Mac')
        const copy = vi.spyOn(supervisor, 'copyModel').mockResolvedValue({
            opId: request.opId,
            result: { ...result, durability: 'uncertain' }
        })
        await supervisor.copyModelToThisPc('MSI', 'llamacpp', 'llama-cpp', request.model)
        expect(copy).toHaveBeenCalledWith({
            ...request,
            opId: expect.stringMatching(/^[0-9a-f]{32}$/),
            destinationNodeId: 'Mac'
        })
    })
    it('keeps the legacy renderer command compatible and routes it through the canonical coordinator', async () => {
        state.setSelfId('Mac')
        vi.spyOn(supervisor, 'hasProcess').mockReturnValue(true)
        const copy = vi.spyOn(supervisor, 'copyModel').mockResolvedValue({
            opId: request.opId,
            result: { ...result, durability: 'uncertain' }
        })
        const invoke = vi
            .fn<PreloadServiceTransport['invoke']>()
            .mockResolvedValue({ success: true, data: null })
        const preload: PreloadServiceTransport = {
            connected: true,
            invoke,
            invokeHistory: () => {
                throw new Error('Unexpected history invocation')
            },
            subscribePush: () => () => {},
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        }
        createEngineApi(serviceTransport(preload)).copyModelToThisPc(
            'llama-cpp',
            'MSI',
            request.model
        )
        const command = {
            command: 'copyModelToThisPc',
            engineType: 'llama-cpp',
            nodeId: 'MSI',
            model: request.model
        } satisfies EngineCommandPayload
        expect(invoke).toHaveBeenCalledWith('engine:command', command)
        await handleServiceBridgeInvoke('engine:command', command)
        expect(copy).toHaveBeenCalledWith({
            ...request,
            opId: expect.stringMatching(/^[0-9a-f]{32}$/),
            destinationNodeId: 'Mac'
        })
    })
    it('exposes typed renderer copy/capability/cancel/snapshot channels', async () => {
        const invoke = vi
            .fn<PreloadServiceTransport['invoke']>()
            .mockResolvedValue({ success: false, error: 'unsupported', code: -32601 })
        const preload: PreloadServiceTransport = {
            connected: true,
            invoke,
            invokeHistory: () => {
                throw new Error('Unexpected history invocation')
            },
            subscribePush: () => () => {},
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        }
        const api = createEngineApi(serviceTransport(preload))
        await expect(api.copyModel(request)).rejects.toMatchObject({ code: -32601 })
        await expect(
            api.modelCopyCapabilities({ nodeId: 'Joes', engine: 'llamacpp' })
        ).rejects.toMatchObject({ code: -32601 })
        await expect(api.cancelModelCopy(request.opId)).rejects.toMatchObject({ code: -32601 })
        await expect(api.activeModelCopies()).rejects.toMatchObject({ code: -32601 })
        expect(invoke.mock.calls.map(args => args[0])).toEqual([
            'engine:copy-model',
            'engine:model-copy-capabilities',
            'engine:cancel-model-copy',
            'engine:active-model-copies'
        ])
    })
})

describe('production supervisor copy reconnect', () => {
    it('subscribes then hydrates the Go active set and leaves a vanished operation unknown', async () => {
        const userData = process.env.PAIR_USER_DATA
        if (!userData) throw new Error('Missing isolated user data')
        initPlatform({
            getUserData: () => userData,
            getTemp: () => os.tmpdir(),
            getResourcesPath: () => userData
        })
        vi.useFakeTimers()
        const clients: JsonRpcSocketClient[] = []
        let operations: JsonValue[] = [{ ...request, progress: frame() }]
        const calls: string[] = []
        vi.spyOn(fs, 'existsSync').mockImplementation(file => String(file).includes('nvpair-'))
        vi.spyOn(JsonRpcSocketClient.prototype, 'start').mockImplementation(function (
            this: JsonRpcSocketClient
        ) {
            clients.push(this)
        })
        vi.spyOn(JsonRpcSocketClient.prototype, 'detach').mockResolvedValue(undefined)
        vi.spyOn(JsonRpcSocketClient.prototype, 'call').mockImplementation(
            async (method): Promise<JsonValue> => {
                calls.push(method)
                if (method === 'engine:active-model-copies') return { operations }
                if (method === 'cluster:get-node-id') return { nodeUuid: 'Mac' }
                if (method === 'workloads:get-initial') return { workloads: [] }
                if (method === 'engine:descriptors' || method === 'engine:get-installed')
                    return { engines: [] }
                if (method === 'nodes:get-initial') return { nodes: [] }
                return {}
            }
        )
        try {
            supervisor.start()
            const client = clients[0]
            if (!client) throw new Error('Supervisor did not create its client')
            client.emit('notification', { source: 'broker', method: 'app:ready' })
            await flush()
            expect(calls.indexOf('engine:subscribe')).toBeLessThan(
                calls.indexOf('engine:active-model-copies')
            )
            expect(
                state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
            ).toMatchObject({ status: 'downloading', exportId })
            client.emit('disconnected')
            expect(
                state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
                    ?.status
            ).toBe('unknown')
            operations = []
            client.emit('notification', { source: 'broker', method: 'app:ready' })
            await flush()
            expect(calls.filter(method => method === 'engine:active-model-copies')).toHaveLength(2)
            expect(
                state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
            ).toMatchObject({ status: 'unknown', exportId })
            expect(
                state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
                    ?.copyResult
            ).toBeUndefined()
        } finally {
            await supervisor.stop()
            vi.useRealTimers()
        }
    })
})

describe('already-open copy observers', () => {
    const calls: string[] = []
    const clients: JsonRpcSocketClient[] = []

    beforeEach(() => {
        const userData = process.env.PAIR_USER_DATA
        if (!userData) throw new Error('Missing isolated user data')
        initPlatform({
            getUserData: () => userData,
            getTemp: () => os.tmpdir(),
            getResourcesPath: () => userData
        })
        vi.useFakeTimers()
        calls.length = 0
        clients.length = 0
        vi.spyOn(fs, 'existsSync').mockImplementation(file => String(file).includes('nvpair-'))
        vi.spyOn(JsonRpcSocketClient.prototype, 'start').mockImplementation(function (
            this: JsonRpcSocketClient
        ) {
            clients.push(this)
        })
        vi.spyOn(JsonRpcSocketClient.prototype, 'detach').mockResolvedValue(undefined)
        vi.spyOn(JsonRpcSocketClient.prototype, 'call').mockImplementation(
            async (method): Promise<JsonValue> => {
                calls.push(method)
                if (method === 'engine:active-model-copies') return { operations: [] }
                if (method === 'cluster:get-node-id')
                    return { nodeUuid: state.getSelfId() ?? 'Joes' }
                if (method === 'workloads:get-initial') return { workloads: [] }
                if (method === 'engine:descriptors' || method === 'engine:get-installed')
                    return { engines: [] }
                if (method === 'nodes:get-initial') return { nodes: [] }
                return {}
            }
        )
    })
    afterEach(async () => {
        await supervisor.stop()
        vi.useRealTimers()
    })

    async function connectObserver(selfId = 'Joes'): Promise<JsonRpcSocketClient> {
        state.setSelfId(selfId)
        supervisor.start()
        const client = clients[0]
        if (!client) throw new Error('Supervisor did not create its client')
        client.emit('notification', { source: 'broker', method: 'app:ready' })
        await flush()
        expect(calls).toContain('engine:subscribe')
        expect(calls.filter(method => method === 'engine:active-model-copies')).toHaveLength(1)
        return client
    }
    function notify(client: JsonRpcSocketClient, progress: JsonValue): void {
        client.emit('notification', {
            source: 'broker',
            method: 'engine:remote-progress',
            params: progress
        })
    }

    it.each(['Joes', 'Other observer'])(
        'adopts an admitted inbound copy after %s has already connected',
        async selfId => {
            const client = await connectObserver(selfId)
            expect(
                state.getEngineInitialState().activeProgress.some(p => p.opId === request.opId)
            ).toBe(false)
            vi.mocked(emitBridgePush).mockClear()
            notify(client, frame())
            expect(emitBridgePush).toHaveBeenCalledWith(
                'engines:progress-changed',
                expect.objectContaining({
                    opId: request.opId,
                    sourceNodeId: 'MSI',
                    destinationNodeId: 'Joes',
                    nodeId: 'Joes',
                    engineType: 'llama-cpp',
                    model: request.model,
                    status: 'downloading',
                    exportId,
                    completed: 8
                })
            )
            notify(client, frame({ bytesDone: 12 }))
            expect(
                state.getEngineInitialState().activeProgress.filter(p => p.opId === request.opId)
            ).toHaveLength(1)
            expect(
                state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
                    ?.completed
            ).toBe(12)
            notify(client, frame({ stage: 'done', bytesDone: 19 }))
            expect(
                state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
            ).toMatchObject({ status: 'unknown', exportId, completed: 19 })
            expect(
                state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
                    ?.copyResult
            ).toBeUndefined()
            expect(calls.filter(method => method === 'engine:active-model-copies')).toHaveLength(1)
        }
    )

    it('ignores terminal-only unknown operations, invalid identity and mismatched tuples', async () => {
        const client = await connectObserver()
        const strayId = '1'.repeat(32)
        vi.mocked(emitBridgePush).mockClear()
        notify(client, frame({ opId: strayId, stage: 'done' }))
        notify(client, frame({ opId: strayId, stage: 'error' }))
        notify(client, frame({ opId: strayId, stage: 'cancelled' }))
        notify(client, frame({ opId: strayId, stage: 'unknown' }))
        notify(client, frame({ opId: '', stage: 'downloading' }))
        expect(emitBridgePush).not.toHaveBeenCalled()
        notify(client, frame())
        vi.mocked(emitBridgePush).mockClear()
        notify(client, frame({ destinationNodeId: 'different', stage: 'done' }))
        notify(client, frame({ model: 'different', bytesDone: 18 }))
        expect(emitBridgePush).not.toHaveBeenCalled()
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'downloading', completed: 8 })
        expect(state.getEngineInitialState().activeProgress.some(p => p.opId === strayId)).toBe(
            false
        )
    })

    it('keeps explicit unknown terminal progress uncertain through the awaited RPC settlement', async () => {
        const client = await connectObserver('Mac')
        const pending = deferred<JsonValue>()
        const call = vi.spyOn(supervisor, 'callProcess').mockReturnValue(pending.promise)
        const copying = supervisor.copyModel(request)
        const failed = expect(copying).rejects.toMatchObject({ code: -32010 })
        notify(client, frame())
        notify(client, frame({ stage: 'unknown', message: 'Destination acknowledgement lost' }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'unknown', exportId })
        pending.reject(new JsonRpcResponseError(-32010, 'Synthetic settlement error'))
        await failed
        const outcome = state
            .getEngineInitialState()
            .activeProgress.find(p => p.opId === request.opId)
        expect(outcome).toMatchObject({
            status: 'unknown',
            message: 'Destination acknowledgement lost'
        })
        expect(outcome?.error).toBeUndefined()
        expect(outcome?.copyResult).toBeUndefined()
        expect(isEngineCopyInProgress(outcome)).toBe(false)
        expect(call).toHaveBeenCalledTimes(1)
    })

    it('settles explicit unknown progress for an observer without asserting failure or success', async () => {
        const client = await connectObserver()
        notify(client, frame())
        notify(client, frame({ stage: 'unknown', message: 'Destination acknowledgement lost' }))
        const outcome = state
            .getEngineInitialState()
            .activeProgress.find(p => p.opId === request.opId)
        expect(outcome).toMatchObject({
            status: 'unknown',
            message: 'Destination acknowledgement lost',
            exportId
        })
        expect(outcome?.error).toBeUndefined()
        expect(outcome?.copyResult).toBeUndefined()
        expect(isEngineCopyInProgress(outcome)).toBe(false)
        notify(client, frame({ stage: 'error', message: 'Late contradictory terminal frame' }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toEqual(outcome)
    })

    it('preserves awaited result ownership and imported operations when progress arrives', async () => {
        const client = await connectObserver()
        const imported = {
            ...request,
            opId: request.opId.replace(/^./, 'b'),
            expectedExportId: exportId
        }
        state.beginModelCopy(request)
        state.reconcileModelCopies([{ ...request }, imported])
        notify(client, frame())
        notify(client, frame({ ...imported }))
        notify(client, frame({ stage: 'done' }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)?.status
        ).toBe('awaiting-result')
        notify(client, frame({ ...imported, stage: 'done' }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === imported.opId)
        ).toMatchObject({ status: 'unknown', expectedExportId: exportId })
        state.finishModelCopy(
            request,
            'done',
            parseModelCopyResult({ opId: request.opId, result }, request).result
        )
        notify(client, frame({ bytesDone: 1 }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === request.opId)
        ).toMatchObject({ status: 'done', copyResult: result })
    })

    it('keeps observed active copies and last outcomes bounded without displacing admitted work', async () => {
        const client = await connectObserver()
        const activeIds = Array.from(
            { length: 8 },
            (_, index) => `${request.opId.slice(0, 30)}${index.toString(16).padStart(2, '0')}`
        )
        const first = activeIds[0]
        if (!first) throw new Error('Missing first active operation')
        const awaited = { ...request, opId: first }
        state.beginModelCopy(awaited)
        for (const opId of activeIds) notify(client, frame({ opId }))
        const excessId = `${request.opId.slice(0, 30)}ff`
        expect(() => notify(client, frame({ opId: excessId }))).not.toThrow()
        expect(
            state.getEngineInitialState().activeProgress.filter(isEngineCopyInProgress)
        ).toHaveLength(8)
        expect(state.getEngineInitialState().activeProgress.some(p => p.opId === excessId)).toBe(
            false
        )
        notify(client, frame({ opId: first, bytesDone: 15 }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === first)?.completed
        ).toBe(15)
        for (const opId of activeIds) notify(client, frame({ opId, stage: 'done' }))
        expect(
            state.getEngineInitialState().activeProgress.find(p => p.opId === first)?.status
        ).toBe('awaiting-result')
        state.finishModelCopy(
            awaited,
            'done',
            parseModelCopyResult({ opId: first, result }, awaited).result
        )
        for (let index = 8; index < 24; index++) {
            const opId = `${request.opId.slice(0, 30)}${index.toString(16).padStart(2, '0')}`
            notify(client, frame({ opId }))
            notify(client, frame({ opId, stage: 'done' }))
        }
        expect(
            state.getEngineInitialState().activeProgress.filter(p => p.operation === 'copy')
        ).toHaveLength(8)
        expect(
            state.getEngineInitialState().activeProgress.filter(isEngineCopyInProgress)
        ).toHaveLength(0)
    })
})
