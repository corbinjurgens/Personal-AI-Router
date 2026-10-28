// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import fs from 'node:fs'
import { PassThrough } from 'node:stream'
import type { Duplex } from 'node:stream'
import { dialog } from 'electron'
import type {
    BuildApplyOperation,
    BuildApplyPreview,
    Observation
} from '@/shared/types/build-apply'
import type { JsonValue } from '@/shared/types/json'
import {
    chooseBuildApplyPreview,
    startBuildApply,
    getBuildApplyStatus,
    handleBuildApplyEntry
} from '@/electron/build-apply'
import { JsonRpcSocketClient } from '@/electron/service-bridge/json-rpc-client'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import * as connection from '@/electron/service-bridge/service-connection'
import * as statusReader from '@/electron/build-apply/status'

const native = vi.hoisted(() => {
    const order: string[] = []
    return { order, quit: vi.fn<() => void>() }
})
vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => '/inert-pair-fixture', quit: native.quit },
    dialog: { showOpenDialog: vi.fn<typeof dialog.showOpenDialog>() },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('@/electron/build-apply/status', () => ({
    readBuildApplyStatus: vi.fn<typeof statusReader.readBuildApplyStatus>(),
    requireBuildApplyStartup: vi.fn<typeof statusReader.requireBuildApplyStartup>()
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<() => void>() }))
vi.mock('@/electron/service-bridge/broadcaster', () => ({ emitBridgePush: vi.fn<() => void>() }))
vi.mock('@/electron/config/ui-config', () => ({ isFirstRun: () => false }))
vi.mock('@/electron/service-bridge/node-info-poller', () => ({
    startNodeInfoPoller: vi.fn<() => void>(),
    stopNodeInfoPoller: vi.fn<() => void>()
}))
vi.mock('@/shared/utils/log', () => ({
    createStructuredLogger: () => ({
        info: vi.fn<() => void>(),
        warn: vi.fn<() => void>(),
        error: vi.fn<() => void>(),
        verbose: vi.fn<() => void>()
    })
}))

function known<T>(value: T): Observation<T> {
    return { known: true, value }
}
function wire<T>(value: T): JsonValue {
    const json: JsonValue = JSON.parse(JSON.stringify(value))
    return json
}
function preview(): BuildApplyPreview {
    const identity = {
        buildId: known('git:' + 'a'.repeat(40)),
        sourceRevision: known('a'.repeat(40)),
        dirty: known(false),
        sha256: known('b'.repeat(64))
    }
    const fences = {
        runtimeGeneration: known('runtime'),
        serviceInstance: known('service'),
        brokerGeneration: known('broker'),
        managerIncarnation: known('manager'),
        policyRevision: known('policy'),
        settingsRevision: known('settings'),
        intentRevision: known('intent'),
        locksRevision: known('locks'),
        availabilityRevision: known('availability')
    }
    return {
        previewId: 'preview_1',
        expiresAt: new Date(Date.now() + 60000).toISOString(),
        hostId: known('local'),
        platform: 'darwin',
        architecture: 'arm64',
        currentGui: identity,
        bundledService: identity,
        requestedGui: identity,
        requestedService: identity,
        runningService: identity,
        packageSha256: known('c'.repeat(64)),
        targetLayout: known({
            rootName: 'NVIDIA PAIR.app',
            executable: 'Contents/MacOS/NVIDIA PAIR',
            serviceDir: 'Contents/Resources/cli-bin',
            serviceExecutable: 'nvpair-service'
        }),
        observationsComplete: true,
        impact: {
            complete: true,
            fences,
            maintenance: { buildApplyActive: known(false), modelSettingsActive: known(false) },
            availability: known('available'),
            pauseOrigin: known('none'),
            engines: known([]),
            activeWork: known(0),
            modelOperations: known([]),
            settingsDifferences: known([]),
            expectedSettings: {
                policyDigest: known('d'.repeat(64)),
                engineSettingsDigest: known('d'.repeat(64)),
                intentDigest: known('d'.repeat(64)),
                locksDigest: known('d'.repeat(64)),
                stopEngines: known(true),
                unloadModels: known(true)
            },
            interruptionPolicy: {
                onActive: known('finish'),
                drainTimeoutSeconds: known(30),
                stopsEngines: true,
                unloadsModels: true,
                committedStreamsMayFail: true,
                includesWorkBeforeFence: true
            },
            restorePolicy: {
                remainPaused: true,
                requireLocalResume: true,
                preserveIntentAndLocks: true,
                preserveNewerPause: true,
                automaticRollback: false
            },
            refusals: []
        }
    }
}
function operation(): BuildApplyOperation {
    const value = preview()
    return {
        operationId: 'operation_1',
        previewId: 'preview_1',
        ownershipAcknowledged: true,
        phase: 'quiesce',
        outcome: null,
        failure: null,
        saved: known(true),
        enforced: known(false),
        physicalStop: known(false),
        remainsPaused: known(true),
        runningService: value.runningService,
        runningGui: value.currentGui,
        readbackConfidence: 'incomplete',
        fences: value.impact.fences
    }
}
const consent = { allowInterruption: true, allowEngineStop: true }
const supervisor = getModularSupervisor()
const clients: JsonRpcSocketClient[] = []
function currentClient(): JsonRpcSocketClient {
    const client = clients.at(-1)
    if (!client) throw new Error('Missing client')
    return client
}
beforeEach(() => {
    vi.mocked(statusReader.requireBuildApplyStartup).mockResolvedValue(undefined)
    vi.mocked(statusReader.readBuildApplyStatus).mockResolvedValue({
        available: true,
        value: { state: 'idle', operation: null, startupAllowed: true },
        refusal: null
    })
    vi.stubGlobal('__PAIR_GUI_BUILD_ID__', 'git:' + 'a'.repeat(40))
    native.order.length = 0
    clients.length = 0
    vi.spyOn(fs, 'existsSync').mockReturnValue(true)
    vi.spyOn(JsonRpcSocketClient.prototype, 'start').mockImplementation(function (
        this: JsonRpcSocketClient
    ) {
        clients.push(this)
    })
    vi.spyOn(JsonRpcSocketClient.prototype, 'detach').mockResolvedValue(undefined)
    vi.spyOn(JsonRpcSocketClient.prototype, 'call').mockImplementation(async method => {
        if (method === 'service/build-apply-preview')
            return wire({ available: true, value: preview(), refusal: null })
        if (method === 'service/build-apply-status')
            return wire({
                available: true,
                value: { state: 'pending', operation: operation(), startupAllowed: false },
                refusal: null
            })
        native.order.push('acknowledged')
        return wire({ available: true, value: operation(), refusal: null })
    })
    supervisor.start()
    vi.spyOn(supervisor, 'ready', 'get').mockReturnValue(true)
    vi.mocked(dialog.showOpenDialog).mockResolvedValue({
        canceled: false,
        filePaths: ['/inert/staged.app']
    })
    native.quit.mockImplementation(() => {
        native.order.push('quit')
    })
})
afterEach(async () => {
    // Retire a known pending operation as the fixture service completes it.
    // This models the process ending without leaking handoff state to a new GUI.
    clients.at(-1)?.emit('notification', {
        source: 'broker',
        method: 'service/build-apply-changed',
        params: wire({ ...operation(), outcome: 'refused' })
    })
    vi.restoreAllMocks()
    await supervisor.stop()
    vi.unstubAllGlobals()
})

describe('native handoff and fencing', () => {
    it('requires startup authority before stopping the service', async () => {
        const stop = vi
            .spyOn(JsonRpcSocketClient.prototype, 'stopService')
            .mockResolvedValue(undefined)
        const detach = vi.mocked(JsonRpcSocketClient.prototype.detach)
        await supervisor.stopService()
        expect(statusReader.requireBuildApplyStartup).toHaveBeenCalledOnce()
        expect(stop).toHaveBeenCalledOnce()
        expect(detach).not.toHaveBeenCalled()
    })
    it('does not detach or stop when startup authority is denied', async () => {
        vi.mocked(statusReader.requireBuildApplyStartup).mockRejectedValueOnce(new Error('pending'))
        const stop = vi.spyOn(JsonRpcSocketClient.prototype, 'stopService')
        const detach = vi.mocked(JsonRpcSocketClient.prototype.detach)
        await expect(supervisor.stopService()).rejects.toThrow('pending')
        expect(stop).not.toHaveBeenCalled()
        expect(detach).not.toHaveBeenCalled()
    })
    it('refuses mutation when the native embedded GUI identity is unknown', async () => {
        await chooseBuildApplyPreview()
        vi.stubGlobal('__PAIR_GUI_BUILD_ID__', '')
        vi.mocked(JsonRpcSocketClient.prototype.call).mockClear()
        const result = await startBuildApply({ previewId: 'preview_1', consent })
        expect(result).toMatchObject({ available: false, refusal: { code: 'identity_unknown' } })
        expect(JsonRpcSocketClient.prototype.call).not.toHaveBeenCalled()
        expect(native.quit).not.toHaveBeenCalled()
    })
    it('sends trusted native GUI facts and detaches only after durable ownership acknowledgement', async () => {
        await chooseBuildApplyPreview()
        vi.spyOn(supervisor, 'stop').mockImplementation(async () => {
            native.order.push('detach')
        })
        const stopService = vi.spyOn(supervisor, 'stopService')
        const result = await startBuildApply({ previewId: 'preview_1', consent })
        expect(result.available).toBe(true)
        expect(native.order).toEqual(['acknowledged', 'detach', 'quit'])
        expect(JsonRpcSocketClient.prototype.call).toHaveBeenLastCalledWith(
            'service/build-apply-start',
            {
                previewId: 'preview_1',
                consent,
                gui: {
                    pid: process.pid,
                    executable: process.execPath,
                    buildId: 'git:' + 'a'.repeat(40)
                }
            }
        )
        expect(stopService).not.toHaveBeenCalled()
    })
    it('does not quit on an accepted response without ownership acknowledgement', async () => {
        await chooseBuildApplyPreview()
        vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
            wire({
                available: true,
                value: { ...operation(), ownershipAcknowledged: false },
                refusal: null
            })
        )
        await startBuildApply({ previewId: 'preview_1', consent })
        expect(native.quit).not.toHaveBeenCalled()
    })
    it('completes handoff once when exact-operation status recovers a delayed acknowledgement', async () => {
        await chooseBuildApplyPreview()
        const detach = vi.spyOn(supervisor, 'stop').mockResolvedValue(undefined)
        vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
            wire({
                available: true,
                value: { ...operation(), ownershipAcknowledged: false },
                refusal: null
            })
        )
        await startBuildApply({ previewId: 'preview_1', consent })
        expect(native.quit).not.toHaveBeenCalled()
        await getBuildApplyStatus({ operationId: 'operation_1' })
        await getBuildApplyStatus({ operationId: 'operation_1' })
        expect(detach).toHaveBeenCalledOnce()
        expect(native.quit).toHaveBeenCalledOnce()
    })
    it.each(['status', 'notification'])(
        'completes the known initiating operation after reconnect using a fresh %s acknowledgement',
        async source => {
            await chooseBuildApplyPreview()
            const detach = vi.spyOn(supervisor, 'stop').mockResolvedValue(undefined)
            vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
                wire({
                    available: true,
                    value: { ...operation(), ownershipAcknowledged: false },
                    refusal: null
                })
            )
            await startBuildApply({ previewId: 'preview_1', consent })
            currentClient().emit('disconnected')
            expect(native.quit).not.toHaveBeenCalled()
            if (source === 'notification')
                currentClient().emit('notification', {
                    source: 'broker',
                    method: 'service/build-apply-changed',
                    params: wire(operation())
                })
            await getBuildApplyStatus({ operationId: 'operation_1' })
            expect(detach).toHaveBeenCalledOnce()
            expect(native.quit).toHaveBeenCalledOnce()
        }
    )
    it('hands off only the initiating operation when a delayed acknowledgement is pushed', async () => {
        await chooseBuildApplyPreview()
        const detach = vi.spyOn(supervisor, 'stop').mockResolvedValue(undefined)
        vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
            wire({
                available: true,
                value: { ...operation(), ownershipAcknowledged: false },
                refusal: null
            })
        )
        await startBuildApply({ previewId: 'preview_1', consent })
        currentClient().emit('notification', {
            source: 'broker',
            method: 'service/build-apply-changed',
            params: wire({ ...operation(), operationId: 'operation_other' })
        })
        expect(detach).not.toHaveBeenCalled()
        for (let repeat = 0; repeat < 2; repeat++)
            currentClient().emit('notification', {
                source: 'broker',
                method: 'service/build-apply-changed',
                params: wire(operation())
            })
        await getBuildApplyStatus({ operationId: 'operation_1' })
        expect(detach).toHaveBeenCalledOnce()
        expect(native.quit).toHaveBeenCalledOnce()
    })
    it('refuses no-stop before any mutation RPC and requires fresh consent', async () => {
        await chooseBuildApplyPreview()
        vi.mocked(JsonRpcSocketClient.prototype.call).mockClear()
        expect(
            (
                await startBuildApply({
                    previewId: 'preview_1',
                    consent: { ...consent, allowEngineStop: false }
                })
            ).available
        ).toBe(false)
        expect((await startBuildApply({ previewId: 'preview_1', consent })).available).toBe(false)
        expect(JsonRpcSocketClient.prototype.call).not.toHaveBeenCalled()
        expect(native.quit).not.toHaveBeenCalled()
    })
    it.each(['expired', 'unknown', 'connection changed'])(
        'refuses %s preview before any mutation RPC',
        async scenario => {
            const value = preview()
            if (scenario === 'expired') value.expiresAt = new Date(Date.now() - 1).toISOString()
            if (scenario === 'unknown') value.requestedService.dirty = { known: false, value: null }
            vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
                wire({ available: true, value, refusal: null })
            )
            await chooseBuildApplyPreview()
            if (scenario === 'connection changed') currentClient().emit('disconnected')
            vi.mocked(JsonRpcSocketClient.prototype.call).mockClear()
            expect((await startBuildApply({ previewId: 'preview_1', consent })).available).toBe(
                false
            )
            expect(JsonRpcSocketClient.prototype.call).not.toHaveBeenCalled()
        }
    )
    it('rejects the previous receipt while start is pending and recovers its lost reply', async () => {
        await chooseBuildApplyPreview()
        const detach = vi.spyOn(supervisor, 'stop').mockResolvedValue(undefined)
        let rejectStart: (reason: Error) => void = () => {
            throw new Error('Missing start rejection')
        }
        vi.mocked(JsonRpcSocketClient.prototype.call).mockReturnValueOnce(
            new Promise((_resolve, reject) => {
                rejectStart = reject
            })
        )
        const start = startBuildApply({ previewId: 'preview_1', consent })
        vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
            wire({
                available: true,
                value: {
                    state: 'terminal',
                    startupAllowed: true,
                    operation: {
                        ...operation(),
                        operationId: 'operation_other',
                        previewId: 'preview_other',
                        outcome: 'refused'
                    }
                },
                refusal: null
            })
        )
        const previous = await getBuildApplyStatus({ operationId: '' })
        expect(previous).toMatchObject({ available: false, refusal: { code: 'outcome_unknown' } })
        rejectStart(new Error('Lost start reply'))
        expect(await start).toMatchObject({
            available: false,
            refusal: { code: 'outcome_unknown' }
        })
        await getBuildApplyStatus({
            operationId: previous.available ? previous.value.operationId : ''
        })
        expect(detach).toHaveBeenCalledOnce()
        expect(native.quit).toHaveBeenCalledOnce()
    })
    it('hands off once when matching status acknowledges before the start reply arrives', async () => {
        await chooseBuildApplyPreview()
        const detach = vi.spyOn(supervisor, 'stop').mockResolvedValue(undefined)
        let resolveStart: (value: JsonValue) => void = () => {
            throw new Error('Missing start resolution')
        }
        vi.mocked(JsonRpcSocketClient.prototype.call).mockReturnValueOnce(
            new Promise(resolve => {
                resolveStart = resolve
            })
        )
        const start = startBuildApply({ previewId: 'preview_1', consent })
        await getBuildApplyStatus({ operationId: '' })
        resolveStart(wire({ available: true, value: operation(), refusal: null }))
        await start
        expect(detach).toHaveBeenCalledOnce()
        expect(native.quit).toHaveBeenCalledOnce()
    })
    it('does not hand off a lost start when status belongs to another preview', async () => {
        await chooseBuildApplyPreview()
        vi.mocked(JsonRpcSocketClient.prototype.call)
            .mockClear()
            .mockRejectedValueOnce(new Error('Connection lost'))
        const result = await startBuildApply({ previewId: 'preview_1', consent })
        expect(result).toMatchObject({
            available: false,
            refusal: { code: 'outcome_unknown', detail: expect.stringContaining('Check status') }
        })
        expect(
            vi.mocked(JsonRpcSocketClient.prototype.call).mock.calls.map(([method]) => method)
        ).toEqual(['service/build-apply-start'])
        vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
            wire({
                available: true,
                value: {
                    state: 'pending',
                    operation: { ...operation(), previewId: 'preview_other' },
                    startupAllowed: false
                },
                refusal: null
            })
        )
        const unrelated = await getBuildApplyStatus({ operationId: '' })
        expect(unrelated).toMatchObject({ available: false, refusal: { code: 'outcome_unknown' } })
        expect(native.quit).not.toHaveBeenCalled()
        expect((await chooseBuildApplyPreview()).available).toBe(false)
        // Mirror the renderer: only available results replace its requested ID.
        const nextId = unrelated.available ? unrelated.value.operationId : ''
        await getBuildApplyStatus({ operationId: nextId })
        expect(native.quit).toHaveBeenCalledOnce()
    })
    it('hands off once when a wholly lost reply is correlated by durable preview ID', async () => {
        await chooseBuildApplyPreview()
        const detach = vi.spyOn(supervisor, 'stop').mockResolvedValue(undefined)
        vi.mocked(JsonRpcSocketClient.prototype.call).mockRejectedValueOnce(new Error('lost'))
        await startBuildApply({ previewId: 'preview_1', consent })
        await getBuildApplyStatus({ operationId: '' })
        await getBuildApplyStatus({ operationId: '' })
        expect(detach).toHaveBeenCalledOnce()
        expect(native.quit).toHaveBeenCalledOnce()
    })
    it('does not hand off a lost reply from unknown status', async () => {
        await chooseBuildApplyPreview()
        vi.mocked(JsonRpcSocketClient.prototype.call).mockRejectedValueOnce(new Error('lost'))
        await startBuildApply({ previewId: 'preview_1', consent })
        vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
            wire({
                available: true,
                value: { state: 'unknown', operation: operation(), startupAllowed: false },
                refusal: null
            })
        )
        await getBuildApplyStatus({ operationId: '' })
        expect(native.quit).not.toHaveBeenCalled()
        await getBuildApplyStatus({ operationId: '' })
    })
    it('does not report success while terminal publication still blocks startup', async () => {
        vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
            wire({
                available: true,
                value: {
                    state: 'terminal',
                    operation: { ...operation(), outcome: 'succeeded' },
                    startupAllowed: false
                },
                refusal: null
            })
        )
        expect(await getBuildApplyStatus({ operationId: '' })).toMatchObject({
            available: false,
            refusal: { code: 'status_unknown' },
            uncertainOperation: { operationId: 'operation_1', outcome: 'succeeded' }
        })
        expect(native.quit).not.toHaveBeenCalled()
    })
    it('rehydrates a lost reply through the no-start status reader when the service is off', async () => {
        await chooseBuildApplyPreview()
        const detach = vi.spyOn(supervisor, 'stop').mockResolvedValue(undefined)
        vi.mocked(JsonRpcSocketClient.prototype.call).mockRejectedValueOnce(new Error('lost'))
        await startBuildApply({ previewId: 'preview_1', consent })
        vi.mocked(JsonRpcSocketClient.prototype.call).mockRejectedValueOnce(new Error('offline'))
        vi.mocked(statusReader.readBuildApplyStatus).mockResolvedValueOnce({
            available: true,
            value: { state: 'pending', operation: operation(), startupAllowed: false },
            refusal: null
        })
        await getBuildApplyStatus({ operationId: '' })
        expect(statusReader.readBuildApplyStatus).toHaveBeenCalledOnce()
        expect(detach).toHaveBeenCalledOnce()
        expect(native.quit).toHaveBeenCalledOnce()
    })
    it('resolves a lost reply with a closed matching terminal result without quitting', async () => {
        await chooseBuildApplyPreview()
        vi.mocked(JsonRpcSocketClient.prototype.call).mockRejectedValueOnce(new Error('lost'))
        await startBuildApply({ previewId: 'preview_1', consent })
        vi.mocked(JsonRpcSocketClient.prototype.call).mockResolvedValueOnce(
            wire({
                available: true,
                value: {
                    state: 'terminal',
                    operation: { ...operation(), outcome: 'refused' },
                    startupAllowed: true
                },
                refusal: null
            })
        )
        expect((await getBuildApplyStatus({ operationId: '' })).available).toBe(true)
        expect(native.quit).not.toHaveBeenCalled()
        expect((await chooseBuildApplyPreview()).available).toBe(true)
    })
    it('deduplicates concurrent status reads and rejects a late previous-generation result', async () => {
        let complete: (value: JsonValue) => void = () => {
            throw new Error('Missing deferred result')
        }
        vi.mocked(JsonRpcSocketClient.prototype.call).mockReturnValueOnce(
            new Promise(resolve => {
                complete = resolve
            })
        )
        const first = getBuildApplyStatus({ operationId: 'operation_1' })
        const second = getBuildApplyStatus({ operationId: 'operation_1' })
        const settled = Promise.allSettled([first, second])
        expect(JsonRpcSocketClient.prototype.call).toHaveBeenCalledOnce()
        currentClient().emit('disconnected')
        complete(
            wire({
                available: true,
                value: { state: 'pending', operation: operation(), startupAllowed: false },
                refusal: null
            })
        )
        expect((await settled).map(result => result.status)).toEqual(['rejected', 'rejected'])
    })
    it('rejects a late preview reply after connection replacement', async () => {
        let complete: (value: JsonValue) => void = () => {
            throw new Error('Missing deferred result')
        }
        vi.mocked(JsonRpcSocketClient.prototype.call).mockReturnValueOnce(
            new Promise(resolve => {
                complete = resolve
            })
        )
        const pending = chooseBuildApplyPreview()
        const settled = Promise.allSettled([pending])
        await Promise.resolve()
        await Promise.resolve()
        currentClient().emit('disconnected')
        complete(wire({ available: true, value: preview(), refusal: null }))
        expect((await settled)[0].status).toBe('rejected')
    })
    it('acknowledges the exact early startup operation through an attach-only client', async () => {
        vi.spyOn(connection, 'connectOrStart')
        vi.mocked(JsonRpcSocketClient.prototype.start).mockImplementation(function (
            this: JsonRpcSocketClient
        ) {
            this.emit('notification', { source: 'broker', method: 'app:ready', params: {} })
        })
        expect(await handleBuildApplyEntry(['--pair-build-apply=operation_1'])).toBeNull()
        expect(JsonRpcSocketClient.prototype.call).toHaveBeenCalledWith(
            'service/build-apply-gui-ready',
            {
                operationId: 'operation_1',
                gui: {
                    pid: process.pid,
                    executable: process.execPath,
                    buildId: 'git:' + 'a'.repeat(40)
                }
            }
        )
        expect(connection.connectOrStart).not.toHaveBeenCalled()
    })
    it('rejects stale status for a lost reply before accepting fresh correlation', async () => {
        await chooseBuildApplyPreview()
        const detach = vi.spyOn(supervisor, 'stop').mockResolvedValue(undefined)
        vi.mocked(JsonRpcSocketClient.prototype.call).mockRejectedValueOnce(new Error('lost'))
        await startBuildApply({ previewId: 'preview_1', consent })
        let complete: (value: JsonValue) => void = () => {
            throw new Error('missing deferred result')
        }
        vi.mocked(JsonRpcSocketClient.prototype.call).mockReturnValueOnce(
            new Promise(resolve => {
                complete = resolve
            })
        )
        const read = getBuildApplyStatus({ operationId: '' })
        const settled = Promise.allSettled([read])
        currentClient().emit('disconnected')
        complete(
            wire({
                available: true,
                value: { state: 'pending', operation: operation(), startupAllowed: false },
                refusal: null
            })
        )
        expect((await settled)[0].status).toBe('rejected')
        expect(detach).not.toHaveBeenCalled()
        expect(native.quit).not.toHaveBeenCalled()
        await getBuildApplyStatus({ operationId: '' })
        expect(detach).toHaveBeenCalledOnce()
        expect(native.quit).toHaveBeenCalledOnce()
    })
    it('hands off a lost reply from a fresh correlated status after reconnect', async () => {
        await chooseBuildApplyPreview()
        const detach = vi.spyOn(supervisor, 'stop').mockResolvedValue(undefined)
        vi.mocked(JsonRpcSocketClient.prototype.call).mockRejectedValueOnce(new Error('lost'))
        await startBuildApply({ previewId: 'preview_1', consent })
        currentClient().emit('disconnected')
        await getBuildApplyStatus({ operationId: '' })
        expect(detach).toHaveBeenCalledOnce()
        expect(native.quit).toHaveBeenCalledOnce()
    })
})

it('keeps ordinary cold start available without calling the deferred status command', async () => {
    const start = vi.fn<() => void>()
    const connect = vi
        .fn<(endpoint: string) => Promise<Duplex>>()
        .mockRejectedValueOnce(new Error('Not running'))
    const socket = new PassThrough()
    connect.mockResolvedValue(socket)
    expect(
        await connection.connectOrStart(
            { endpoint: '/private/inert.sock' },
            {
                connect,
                startService: start,
                now: Date.now,
                sleep: async () => {}
            }
        )
    ).toBe(socket)
    expect(start).toHaveBeenCalledOnce()
    expect(connect).toHaveBeenCalledTimes(2)
    socket.destroy()
})

it('allows attachment to a running service without evaluating a startup mutation gate', async () => {
    const socket = new PassThrough()
    const start = vi.fn<() => void>()
    expect(
        await connection.connectOrStart(
            { endpoint: '/private/inert.sock' },
            {
                connect: async () => socket,
                startService: start,
                now: Date.now,
                sleep: async () => {}
            }
        )
    ).toBe(socket)
    expect(start).not.toHaveBeenCalled()
    socket.destroy()
})
