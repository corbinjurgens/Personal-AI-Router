// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import * as childProcess from 'node:child_process'
import { buildIdentity } from '../../scripts/build-identity'
import { dialog, ipcRenderer } from 'electron'
import type {
    Observation,
    BuildIdentity,
    BuildApplyFences,
    BuildApplyPreview,
    BuildApplyOperation,
    BuildApplyPrepareRequest,
    BuildApplyFinishRequest,
    BuildApplyGuiReadyRequest
} from '@/shared/types/build-apply'
import { BUILD_APPLY_MAINTENANCE_BUSY } from '@/shared/types/build-apply'
import { BUILD_APPLY_MAINTENANCE_BUSY_CODE } from '@/shared/constants/service-errors'
import type { JsonValue } from '@/shared/types/json'
import { JsonRpcSocketClient } from '@/electron/service-bridge/json-rpc-client'
import { getModularSupervisor } from '@/electron/service-bridge/modular-supervisor'
import {
    parseBuildApplyPreviewResult,
    parseBuildApplyOperationResult,
    parseBuildApplyOperation
} from '@/electron/service-bridge/build-apply'
import {
    chooseBuildApplyPreview,
    startBuildApply,
    getBuildApplyStatus,
    handleBuildApplyEntry
} from '@/electron/build-apply'
import { buildApplyApi } from '@/preload/api/build-apply.api'

const native = vi.hoisted(() => ({
    send: vi.fn<(channel: string, operation: BuildApplyOperation) => void>(),
    on: vi.fn<
        (
            channel: string,
            handler: (event: Electron.IpcRendererEvent, operation: BuildApplyOperation) => void
        ) => void
    >(),
    remove: vi.fn<
        (
            channel: string,
            handler: (event: Electron.IpcRendererEvent, operation: BuildApplyOperation) => void
        ) => void
    >()
}))
vi.mock('node:child_process', async importOriginal => {
    const original = await importOriginal<typeof childProcess>()
    return { ...original, execFileSync: vi.fn<typeof childProcess.execFileSync>() }
})
vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => '/inert-pair-fixture' },
    dialog: { showOpenDialog: vi.fn<typeof dialog.showOpenDialog>() },
    BrowserWindow: {
        getAllWindows: () => [
            {
                isDestroyed: () => false,
                webContents: { isDestroyed: () => false, send: native.send }
            }
        ]
    },
    ipcRenderer: {
        on: native.on,
        removeListener: native.remove,
        invoke: vi.fn<typeof ipcRenderer.invoke>()
    }
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
function absent<T>(): Observation<T> {
    return { known: false, value: null }
}
function wire<T>(value: T): JsonValue {
    const json: JsonValue = JSON.parse(JSON.stringify(value))
    return json
}
function identity(stamp: string): BuildIdentity {
    const revision = Buffer.from(stamp).toString('hex').padEnd(40, '0').slice(0, 40)
    return {
        buildId: known('git:' + revision),
        sourceRevision: known(revision),
        dirty: known(false),
        sha256: known('a'.repeat(64))
    }
}
function fences(): BuildApplyFences {
    return {
        runtimeGeneration: known('runtime-1'),
        serviceInstance: known('service-1'),
        brokerGeneration: known('broker-2'),
        managerIncarnation: known('manager-3'),
        policyRevision: known('4'),
        settingsRevision: known('5'),
        intentRevision: known('6'),
        locksRevision: known('7'),
        availabilityRevision: known('8')
    }
}
function preview(): BuildApplyPreview {
    return {
        previewId: 'preview_1',
        expiresAt: '2026-10-28T00:01:00Z',
        hostId: known('local-host'),
        platform: 'windows',
        architecture: 'amd64',
        currentGui: identity('current-gui'),
        bundledService: identity('bundled-service'),
        requestedGui: identity('requested-gui'),
        requestedService: {
            ...identity('requested-service'),
            buildId: absent(),
            sourceRevision: absent(),
            dirty: absent()
        },
        runningService: identity('running-service'),
        packageSha256: known('b'.repeat(64)),
        targetLayout: known({
            rootName: '',
            executable: 'PAIR.exe',
            serviceDir: 'resources/cli-bin',
            serviceExecutable: 'nvpair-service.exe'
        }),
        observationsComplete: false,
        impact: {
            complete: false,
            fences: fences(),
            maintenance: { buildApplyActive: known(false), modelSettingsActive: absent() },
            availability: known('paused'),
            pauseOrigin: known('peer'),
            engines: known([
                {
                    engine: 'ollama',
                    active: known(false),
                    running: absent(),
                    loadedModels: known([]),
                    desiredState: known('off'),
                    locked: known(true),
                    scopedStopAuthority: absent()
                }
            ]),
            activeWork: absent(),
            modelOperations: known([{ engine: 'ollama', kind: 'copy', blocksApply: true }]),
            settingsDifferences: known([
                {
                    field: 'pause.stopEngines',
                    stored: known(false),
                    effective: known(true),
                    persisted: known(false)
                }
            ]),
            expectedSettings: {
                policyDigest: known('policy-digest'),
                engineSettingsDigest: known('settings-digest'),
                intentDigest: known('intent-digest'),
                locksDigest: known('locks-digest'),
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
            refusals: [{ code: 'identity_incomplete', detail: 'Go source metadata is unknown.' }]
        }
    }
}
function operation(): BuildApplyOperation {
    return {
        operationId: 'operation_1',
        ownershipAcknowledged: true,
        phase: 'verify-settings',
        outcome: 'uncertain',
        saved: known(true),
        enforced: absent(),
        physicalStop: known(true),
        remainsPaused: known(true),
        runningService: identity('new-running-service'),
        runningGui: identity('new-running-gui'),
        readbackConfidence: 'incomplete',
        fences: fences(),
        failure: {
            phase: 'verify-settings',
            reason: 'settings_changed',
            detail: 'Settings readback differs.',
            bytesChanged: known(true),
            settingsChanged: absent(),
            runningIdentity: identity('new-running-service'),
            readbackConfidence: 'incomplete',
            safeNextAction: 'Keep paused and inspect the retained operation.'
        }
    }
}

describe('build apply payloads', () => {
    it('detects a genuinely modified tracked file in an isolated git checkout', () => {
        const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'pair-gui-identity-'))
        const run = (command: string, args: string[], cwd = directory): string => {
            const result = childProcess.spawnSync(command, args, {
                cwd,
                encoding: 'utf8',
                timeout: 10000,
                maxBuffer: 64 * 1024,
                windowsHide: true
            })
            if (result.error || result.status !== 0)
                throw new Error(`Identity fixture command failed: ${result.error ?? result.stderr}`)
            return result.stdout.trim()
        }
        try {
            run('git', ['init', '--quiet'])
            fs.writeFileSync(path.join(directory, 'source.txt'), 'reviewed source\n')
            run('git', ['add', '--', 'source.txt'])
            run('git', [
                '-c',
                'user.name=PAIR fixture',
                '-c',
                'user.email=fixture@example.invalid',
                '-c',
                'commit.gpgsign=false',
                'commit',
                '--quiet',
                '--no-signoff',
                '-m',
                'fixture'
            ])
            const revision = run('git', ['rev-parse', 'HEAD'])
            const modulePath = path.resolve('scripts/build-identity.ts')
            const script = `const { buildIdentity } = require(${JSON.stringify(modulePath)}); console.log(buildIdentity(process.argv[1]));`
            const stamp = (): string =>
                run(
                    process.execPath,
                    ['--require', 'tsx/cjs', '--eval', script, directory],
                    process.cwd()
                )
            expect(stamp()).toBe(`git:${revision}`)
            fs.writeFileSync(path.join(directory, 'source.txt'), 'changed tracked source\n')
            expect(run('git', ['status', '--porcelain'])).toBe('M source.txt')
            expect(stamp()).toBe(`git:${revision}+dirty`)
        } finally {
            fs.rmSync(directory, { recursive: true, force: true })
        }
    })
    it('embeds only build-time GUI source facts and keeps a failed stamp unknown', () => {
        const git = vi.mocked(childProcess.execFileSync)
        git.mockReset()
        try {
            git.mockReturnValueOnce('a'.repeat(40)).mockReturnValueOnce('')
            expect(buildIdentity('/inert-build-source')).toBe('git:' + 'a'.repeat(40))
            git.mockReturnValueOnce('a'.repeat(40)).mockReturnValueOnce(' M desktop/source.ts')
            expect(buildIdentity('/inert-build-source')).toBe('git:' + 'a'.repeat(40) + '+dirty')
            expect(git).toHaveBeenCalledWith(
                'git',
                ['rev-parse', 'HEAD'],
                expect.objectContaining({
                    cwd: '/inert-build-source',
                    maxBuffer: 64 * 1024,
                    timeout: 10000,
                    windowsHide: true
                })
            )
            git.mockImplementationOnce(() => {
                throw new Error('VCS unavailable')
            })
            expect(buildIdentity('/inert-build-source')).toBe('')
        } finally {
            git.mockReset()
        }
    })
    it('preserves all preview fields and independent unknown VCS observations', () => {
        const value = preview()
        expect(
            parseBuildApplyPreviewResult(wire({ available: true, value, refusal: null }))
        ).toEqual({
            available: true,
            value,
            refusal: null
        })
        expect(value.requestedService.dirty).toEqual({ known: false, value: null })
        expect(value.packageSha256).not.toEqual(value.requestedService.buildId)
    })
    it('preserves failure phase, confidence, pause and separate saved/enforced/physical-stop facts', () => {
        const value = operation()
        expect(parseBuildApplyOperation(wire(value))).toEqual(value)
        expect(
            parseBuildApplyOperationResult(wire({ available: true, value, refusal: null })).value
        ).toEqual(value)
    })
    it('retains the reciprocal maintenance refusal without inventing ownership', () => {
        const result = {
            available: false,
            value: null,
            refusal: { code: BUILD_APPLY_MAINTENANCE_BUSY, detail: 'Maintenance busy.' }
        }
        expect(parseBuildApplyOperationResult(wire(result))).toEqual(result)
        expect(BUILD_APPLY_MAINTENANCE_BUSY_CODE).toBe(-32015)
        const prepare: BuildApplyPrepareRequest = {
            operationId: 'operation_1',
            fences: fences(),
            consent: { allowInterruption: true, allowEngineStop: true }
        }
        const finish: BuildApplyFinishRequest = {
            operationId: prepare.operationId,
            fences: prepare.fences
        }
        const ready: BuildApplyGuiReadyRequest = {
            operationId: finish.operationId,
            gui: { pid: 42, executable: '/native/PAIR.exe', buildId: 'embedded-gui' }
        }
        expect(Object.keys(prepare).sort()).toEqual(['consent', 'fences', 'operationId'])
        expect(Object.keys(ready).sort()).toEqual(['gui', 'operationId'])
    })
    it('rejects missing meaningful fields, inconsistent known values and malformed results', () => {
        const value = wire(operation())
        if (!value || typeof value !== 'object' || Array.isArray(value))
            throw new Error('Invalid fixture.')
        expect(() =>
            parseBuildApplyOperation({ ...value, physicalStop: { known: true, value: null } })
        ).toThrow()
        expect(() =>
            parseBuildApplyOperation({ ...value, saved: { known: false, value: false } })
        ).toThrow()
        expect(() => parseBuildApplyOperation({ ...value, fences: {} })).toThrow()
        expect(() => parseBuildApplyOperation({ ...value, phase: 'done' })).toThrow()
        expect(() =>
            parseBuildApplyOperation({ ...value, failure: { detail: 'x'.repeat(1025) } })
        ).toThrow()
        expect(() =>
            parseBuildApplyOperationResult({ available: false, value, refusal: null })
        ).toThrow()
    })
})

describe('native local build apply registration', () => {
    let client: JsonRpcSocketClient | undefined
    const supervisor = getModularSupervisor()
    beforeEach(() => {
        client = undefined
        const startedClients: JsonRpcSocketClient[] = []
        vi.spyOn(fs, 'existsSync').mockReturnValue(true)
        vi.spyOn(JsonRpcSocketClient.prototype, 'start').mockImplementation(function (
            this: JsonRpcSocketClient
        ) {
            startedClients.push(this)
        })
        vi.spyOn(JsonRpcSocketClient.prototype, 'detach').mockResolvedValue(undefined)
        vi.spyOn(JsonRpcSocketClient.prototype, 'call').mockResolvedValue({
            available: false,
            value: null,
            refusal: { code: 'unavailable', detail: 'Build apply is unavailable.' }
        })
        supervisor.start()
        client = startedClients.at(-1)
    })
    afterEach(async () => {
        await supervisor.stop()
        vi.restoreAllMocks()
        vi.clearAllMocks()
    })
    it('delivers the existing private supervisor notification through native preload', () => {
        const callback = vi.fn<(value: BuildApplyOperation) => void>()
        const unsubscribe = buildApplyApi.onChanged(callback)
        const subscription = native.on.mock.calls.at(-1)
        if (!subscription || !client) throw new Error('Native attachment was not registered.')
        client.emit('notification', {
            source: 'broker',
            method: 'service/build-apply-changed',
            params: wire(operation())
        })
        expect(native.send).toHaveBeenCalledWith('build-apply:changed', operation())
        // Deliver the fanout through the actual preload subscription.
        const event: Electron.IpcRendererEvent = {
            preventDefault: vi.fn<() => void>(),
            defaultPrevented: false,
            ports: [],
            sender: ipcRenderer
        }
        subscription[1](event, operation())
        expect(callback).toHaveBeenCalledWith(operation())
        unsubscribe()
        expect(native.remove).toHaveBeenCalledWith('build-apply:changed', subscription[1])
    })
    it('uses only picker-owned paths and native GUI facts, while unavailable replies launch nothing', async () => {
        vi.spyOn(supervisor, 'ready', 'get').mockReturnValue(true)
        vi.mocked(dialog.showOpenDialog).mockResolvedValue({
            canceled: false,
            filePaths: [process.execPath]
        })
        const result = await chooseBuildApplyPreview()
        expect(result.available).toBe(false)
        const call = vi.mocked(JsonRpcSocketClient.prototype.call)
        expect(call).toHaveBeenCalledWith('service/build-apply-preview', {
            candidatePath: process.execPath,
            gui: { pid: process.pid, executable: process.execPath, buildId: '' }
        })
        const start = await startBuildApply({
            previewId: 'preview_1',
            consent: { allowInterruption: true, allowEngineStop: false }
        })
        expect(start).toMatchObject({
            available: false,
            value: null,
            refusal: { code: 'consent_required' }
        })
        expect(call.mock.calls.map(([method]) => method)).toEqual(['service/build-apply-preview'])
        expect(vi.mocked(JsonRpcSocketClient.prototype.start)).toHaveBeenCalledTimes(1)
    })
    it('does not launch or request anything without an attached ready service', async () => {
        const result = await getBuildApplyStatus({ operationId: '' })
        expect(result).toMatchObject({ available: false, value: null })
        expect(JsonRpcSocketClient.prototype.call).not.toHaveBeenCalled()
    })
    it('rejects renderer attempts to supply paths, GUI identity or a peer selector', async () => {
        const consent = { allowInterruption: true, allowEngineStop: true }
        await expect(
            startBuildApply({ previewId: 'preview_1', consent, ...{ candidatePath: '/untrusted' } })
        ).rejects.toThrow('Invalid')
        await expect(
            startBuildApply({ previewId: 'preview_1', consent, ...{ gui: { pid: 1 } } })
        ).rejects.toThrow('Invalid')
        await expect(
            getBuildApplyStatus({ operationId: '', ...{ nodeId: 'peer' } })
        ).rejects.toThrow('Invalid')
        expect(JsonRpcSocketClient.prototype.call).not.toHaveBeenCalled()
    })
    it('invalidates an in-flight response when the existing connection changes', async () => {
        vi.spyOn(supervisor, 'ready', 'get').mockReturnValue(true)
        let resolve: (value: JsonValue) => void = () => {
            throw new Error('Missing deferred response.')
        }
        const response = new Promise<JsonValue>(done => {
            resolve = done
        })
        vi.mocked(JsonRpcSocketClient.prototype.call).mockReturnValueOnce(response)
        const pending = getBuildApplyStatus({ operationId: 'operation_1' })
        if (!client) throw new Error('Missing attachment.')
        client.emit('disconnected')
        resolve({
            available: false,
            value: null,
            refusal: { code: 'unavailable', detail: 'Unavailable.' }
        })
        await expect(pending).rejects.toThrow('connection changed')
    })
    it('acknowledges build startup through an attach-only client', async () => {
        const ready = vi.spyOn(supervisor, 'acknowledgeBuildApplyGui').mockResolvedValue({
            available: false,
            value: null,
            refusal: { code: 'unavailable', detail: 'GUI acknowledgement unavailable.' }
        })
        expect(await handleBuildApplyEntry(['--pair-build-apply=operation_1'])).toBe(69)
        expect(ready).toHaveBeenCalledExactlyOnceWith({
            operationId: 'operation_1',
            gui: { pid: process.pid, executable: process.execPath, buildId: '' }
        })
        expect(await handleBuildApplyEntry(['--pair-build-apply=../bad'])).toBe(64)
        expect(await handleBuildApplyEntry([])).toBeNull()
        expect(JsonRpcSocketClient.prototype.call).not.toHaveBeenCalled()
    })
})
