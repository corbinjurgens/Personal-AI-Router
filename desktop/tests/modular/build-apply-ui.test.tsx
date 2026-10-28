// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
    BuildApplyOperation,
    BuildApplyPreview,
    Observation
} from '@/shared/types/build-apply'
import type { IBuildApplyApi } from '@/preload/api/build-apply.api'
import type { IUpdateApi } from '@/preload/api/update.api'
import type { IServiceApi } from '@/preload/api/service.api'
import type { IWindowActionApi } from '@/preload/api/window.api'
import type { ServiceStatus } from '@/shared/types/ipc-channels'
import ApplicationUpdatesCard from '@/ui/components/ServiceSettings/UpdatesSettings'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'

vi.mock('@/ui/api/bootstrap', () => ({ isElectron: true, getPairTransport: () => null }))
function known<T>(value: T): Observation<T> {
    return { known: true, value }
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
            availability: known('paused'),
            pauseOrigin: known('peer'),
            engines: known([
                {
                    engine: 'ollama',
                    active: known(false),
                    running: known(false),
                    loadedModels: known(['model']),
                    desiredState: known('off'),
                    locked: known(true),
                    scopedStopAuthority: known(true)
                }
            ]),
            activeWork: known(3),
            modelOperations: known([]),
            settingsDifferences: known([
                {
                    field: 'pause.stopEngines',
                    stored: known(false),
                    effective: known(true),
                    persisted: known(false)
                }
            ]),
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
        phase: 'finish',
        outcome: 'succeeded',
        failure: null,
        saved: known(true),
        enforced: known(true),
        physicalStop: known(true),
        remainsPaused: known(true),
        runningService: value.runningService,
        runningGui: value.currentGui,
        readbackConfidence: 'verified',
        fences: value.impact.fences
    }
}
const api = {
    choosePreview: vi.fn<IBuildApplyApi['choosePreview']>(),
    start: vi.fn<IBuildApplyApi['start']>(),
    status: vi.fn<IBuildApplyApi['status']>(),
    onChanged: vi.fn<IBuildApplyApi['onChanged']>()
} satisfies IBuildApplyApi
const updates = {
    getStatus: vi.fn<IUpdateApi['getStatus']>(),
    check: vi.fn<IUpdateApi['check']>(),
    download: vi.fn<IUpdateApi['download']>(),
    install: vi.fn<IUpdateApi['install']>(),
    onStatusChanged: vi.fn<IUpdateApi['onStatusChanged']>()
} satisfies IUpdateApi
const onService = vi.fn<IServiceApi['onStatusChanged']>()
const copy = vi.fn<IWindowActionApi['copyToClipboard']>()
const listeners = new Set<(value: BuildApplyOperation) => void>()
const serviceListeners = new Set<(value: ServiceStatus) => void>()
let container: HTMLDivElement
let root: Root
function button(label: string, scope: ParentNode = document): HTMLButtonElement {
    const found = Array.from(scope.querySelectorAll('button')).find(
        value => value.textContent?.trim() === label
    )
    if (!found) throw new Error(`Missing button: ${label}`)
    return found
}
async function click(label: string, scope: ParentNode = document): Promise<void> {
    await act(async () => button(label, scope).click())
}
async function choose(): Promise<void> {
    await click('Choose staged build')
}
async function publish(value: BuildApplyOperation): Promise<void> {
    await act(async () => {
        listeners.forEach(listener => listener(value))
    })
}
async function service(connectorStatus: ServiceStatus['connectorStatus']): Promise<void> {
    await act(async () => {
        serviceListeners.forEach(listener => listener({ connectorStatus, weSpawned: true }))
    })
}

beforeEach(async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false, escape: (value: string) => value })
    for (const method of ['show', 'showModal'])
        Object.defineProperty(HTMLDialogElement.prototype, method, {
            configurable: true,
            value: function (this: HTMLDialogElement) {
                this.open = true
            }
        })
    Object.defineProperty(HTMLDialogElement.prototype, 'close', {
        configurable: true,
        value: function (this: HTMLDialogElement) {
            this.open = false
        }
    })
    listeners.clear()
    serviceListeners.clear()
    api.choosePreview.mockResolvedValue({ available: true, value: preview(), refusal: null })
    api.start.mockResolvedValue({ available: true, value: operation(), refusal: null })
    api.status.mockResolvedValue({
        available: false,
        value: null,
        refusal: {
            code: 'unavailable',
            detail: 'Build apply is unavailable until the service transaction lands.'
        }
    })
    api.onChanged.mockImplementation(listener => {
        listeners.add(listener)
        return () => {
            listeners.delete(listener)
        }
    })
    onService.mockImplementation(listener => {
        serviceListeners.add(listener)
        return () => {
            serviceListeners.delete(listener)
        }
    })
    updates.getStatus.mockResolvedValue({
        phase: 'idle',
        currentVersion: 'test',
        latestVersion: null,
        downloadPercent: null,
        error: null
    })
    updates.onStatusChanged.mockReturnValue(() => {})
    updates.check.mockResolvedValue(undefined)
    copy.mockResolvedValue(undefined)
    const native = {
        buildApply: api,
        update: updates,
        service: { onStatusChanged: onService },
        window: { copyToClipboard: copy }
    } satisfies {
        buildApply: IBuildApplyApi
        update: IUpdateApi
        service: Pick<IServiceApi, 'onStatusChanged'>
        window: Pick<IWindowActionApi, 'copyToClipboard'>
    }
    Object.defineProperty(window, 'windowApi', { configurable: true, value: native })
    useConnectionStore.setState({ selfId: 'self', connected: true })
    useNodesStore.setState({ nodes: new Map() })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
    await act(async () => root.render(<ApplicationUpdatesCard />))
})
afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
})

describe('staged application UI', () => {
    it('retains uncertain offline recovery facts without reporting a successful apply', async () => {
        const receipt = operation()
        api.status.mockResolvedValue({
            available: false,
            value: null,
            refusal: {
                code: 'status_unknown',
                detail: 'Build status is uncertain. Use explicit recovery.'
            },
            uncertainOperation: {
                ...receipt,
                failure: {
                    phase: 'finish',
                    reason: 'terminal_publication_uncertain',
                    detail: 'Publication is uncertain.',
                    bytesChanged: known(true),
                    settingsChanged: known(false),
                    runningIdentity: receipt.runningService,
                    readbackConfidence: 'unknown',
                    safeNextAction: 'Retain the bundle and recover this operation.'
                }
            }
        })
        await service('disconnected')
        expect(document.body.textContent).toContain('Operation operation_1')
        expect(document.body.textContent).toContain('terminal_publication_uncertain')
        expect(document.body.textContent).toContain(
            'nvpair-service apply-build recover operation_1'
        )
        expect(document.body.textContent).not.toContain('Build applied; node remains paused')
        await publish(receipt)
        expect(document.body.textContent).not.toContain('Build applied; node remains paused')
    })
    it('replaces a previously displayed success when status becomes uncertain', async () => {
        api.status.mockResolvedValue({ available: true, value: operation(), refusal: null })
        await publish(operation())
        expect(document.body.textContent).toContain('Build applied; node remains paused')
        api.status.mockResolvedValue({
            available: false,
            value: null,
            refusal: {
                code: 'status_unknown',
                detail: 'Build status is uncertain. Use explicit recovery.'
            }
        })
        await click('Read build status', container)
        expect(document.body.textContent).toContain('Build status is uncertain.')
        expect(document.body.textContent).not.toContain('Build applied; node remains paused')
    })
    it('shows an unavailable preview without leaving the picker busy or offering application', async () => {
        api.choosePreview.mockResolvedValue({
            available: false,
            value: null,
            refusal: {
                code: 'unavailable',
                detail: 'Build preview is not available in this build.'
            }
        })
        await choose()
        expect(document.body.textContent).toContain('Build preview is not available in this build.')
        expect(button('Choose staged build').disabled).toBe(false)
        expect(document.querySelector('dialog[open]')).toBeNull()
        expect(api.start).not.toHaveBeenCalled()
    })
    it('shows an unavailable application and status without claiming success or retrying start', async () => {
        api.start.mockResolvedValue({
            available: false,
            value: null,
            refusal: {
                code: 'unavailable',
                detail: 'Build application is not available in this build.'
            }
        })
        await choose()
        await click('Apply build and restart')
        const modal = Array.from(document.querySelectorAll('dialog[open]')).at(-1)
        const checkbox = modal?.querySelector('input[type="checkbox"]')
        if (!modal || !(checkbox instanceof HTMLInputElement))
            throw new Error('Missing consent modal')
        await act(async () => checkbox.click())
        await click('Apply build and restart', modal)
        expect(document.body.textContent).toContain(
            'Build application is not available in this build.'
        )
        expect(button('Choose staged build').disabled).toBe(false)
        expect(document.body.textContent).not.toContain('Build applied; node remains paused')
        await click('Read build status')
        expect(document.body.textContent).toContain(
            'Build apply is unavailable until the service transaction lands.'
        )
        expect(api.start).toHaveBeenCalledOnce()
    })
    it('expires consent without refreshing or applying a preview behind the user', async () => {
        vi.useFakeTimers()
        const value = preview()
        value.expiresAt = new Date(Date.now() + 1000).toISOString()
        api.choosePreview.mockResolvedValue({ available: true, value, refusal: null })
        await choose()
        await click('Apply build and restart')
        const modal = Array.from(document.querySelectorAll('dialog[open]')).at(-1)
        const checkbox = modal?.querySelector('input[type="checkbox"]')
        if (!modal || !(checkbox instanceof HTMLInputElement))
            throw new Error('Missing consent modal')
        await act(async () => checkbox.click())
        await act(async () => vi.advanceTimersByTimeAsync(2000))
        expect(button('Apply build and restart', modal).disabled).toBe(true)
        expect(api.choosePreview).toHaveBeenCalledOnce()
        expect(api.start).not.toHaveBeenCalled()
    })
    it('shows the detailed impact, canonical settings and distinct identities in a wrapping side panel', async () => {
        await choose()
        const panel = document.querySelector('dialog[open]')
        expect(panel?.getAttribute('aria-labelledby')).toBe('build-preview-heading')
        expect(document.getElementById('build-preview-heading')?.textContent).toBe(
            'Preview build on this device'
        )
        expect(panel?.textContent).toContain('Preview build on this device')
        for (const label of [
            'Current GUI',
            'Bundled service',
            'Attached running service',
            'Requested GUI',
            'Requested service',
            'stored false, effective true',
            'Active work: 3',
            'saved intent off, locked true',
            'Committed streams may fail',
            'ordinary local Resume',
            'not publisher authenticity'
        ]) {
            expect(panel?.textContent?.toLowerCase()).toContain(label.toLowerCase())
        }
        expect(panel?.querySelector('.wrap-anywhere')).not.toBeNull()
        await click('Close preview')
        expect(document.activeElement).toBe(button('Choose staged build'))
        await click('Preview build')
        expect(document.querySelector('dialog[open]')).not.toBeNull()
    })
    it.each(['unknown', 'expired'])('disables application for an %s preview', async scenario => {
        const value = preview()
        if (scenario === 'unknown')
            value.runningService.sourceRevision = { known: false, value: null }
        else value.expiresAt = new Date(Date.now() - 1000).toISOString()
        api.choosePreview.mockResolvedValue({ available: true, value, refusal: null })
        await choose()
        expect(button('Apply build and restart').disabled).toBe(true)
        await click('Apply build and restart')
        expect(api.start).not.toHaveBeenCalled()
        expect(document.body.textContent).toContain(
            scenario === 'unknown' ? 'Unknown' : 'Preview expired'
        )
    })
    it('requires fresh explicit consent in a modal and sends only the preview ID and consent', async () => {
        await choose()
        await click('Apply build and restart')
        const modal = Array.from(document.querySelectorAll('dialog[open]')).at(-1)
        if (!modal) throw new Error('Missing consent modal')
        expect(modal.getAttribute('aria-labelledby')).toBe('build-apply-confirm-heading')
        expect(document.getElementById('build-apply-confirm-heading')?.textContent).toBe(
            'Apply build and restart on this device?'
        )
        expect(button('Apply build and restart', modal).disabled).toBe(true)
        expect(modal.textContent).toContain('even if engines currently appear stopped')
        const checkbox = modal.querySelector('input[type="checkbox"]')
        if (!(checkbox instanceof HTMLInputElement)) throw new Error('Missing engine-stop consent')
        await act(async () => checkbox.click())
        await click('Apply build and restart', modal)
        expect(api.start).toHaveBeenCalledExactlyOnceWith({
            previewId: 'preview_1',
            consent: { allowInterruption: true, allowEngineStop: true }
        })
        expect(document.body.textContent).toContain('Build applied; node remains paused')
        expect(document.body.textContent).toContain('Ordinary explicit Resume')
        expect(updates.install).not.toHaveBeenCalled()
    })
    it('retains failed phase, swap uncertainty, safe recovery and redacted diagnostics after remount', async () => {
        const value = operation()
        value.phase = 'verify-settings'
        value.outcome = 'uncertain'
        value.readbackConfidence = 'incomplete'
        value.failure = {
            phase: 'verify-settings',
            reason: 'settings_changed',
            detail: 'Private detail is visible only in the retained result',
            bytesChanged: known(true),
            settingsChanged: { known: false, value: null },
            runningIdentity: value.runningService,
            readbackConfidence: 'incomplete',
            safeNextAction: 'Keep paused and explicitly recover.'
        }
        api.status.mockResolvedValue({ available: true, value, refusal: null })
        await act(async () => root.unmount())
        root = createRoot(container)
        await act(async () => root.render(<ApplicationUpdatesCard />))
        expect(document.body.textContent).toContain('Phase: verify-settings')
        expect(document.body.textContent).toContain(
            'Bytes changed: true. Settings changed: Unknown'
        )
        expect(document.body.textContent).toContain(
            'nvpair-service apply-build recover operation_1'
        )
        expect(document.body.textContent).not.toContain('Build applied; node remains paused')
        await click('Copy operational diagnostics')
        const copied = copy.mock.calls[0]?.[0]
        if (!copied) throw new Error('Missing diagnostics')
        const diagnostics: Record<string, string> = JSON.parse(copied)
        expect(diagnostics.phase).toBe('verify-settings')
        expect(copied).not.toContain('Private detail')
    })
    it('does not describe incomplete success as verified application', async () => {
        const incomplete = {
            ...operation(),
            remainsPaused: { known: false, value: null }
        } satisfies BuildApplyOperation
        api.status.mockResolvedValue({ available: true, value: incomplete, refusal: null })
        await publish(incomplete)
        expect(document.body.textContent).toContain('Build readback incomplete')
        expect(document.body.textContent).not.toContain('Build applied; node remains paused')
    })
    it('offers no direct install action for a downloaded distribution update', async () => {
        updates.getStatus.mockResolvedValue({
            phase: 'downloaded',
            currentVersion: 'test',
            latestVersion: 'next',
            downloadPercent: 100,
            error: null
        })
        await act(async () => root.unmount())
        root = createRoot(container)
        await act(async () => root.render(<ApplicationUpdatesCard />))
        expect(document.body.textContent).toContain('verified-package adapter')
        expect(document.body.textContent).not.toContain('Restart & install')
        expect(updates.install).not.toHaveBeenCalled()
        await choose()
        expect(button('Apply build and restart').disabled).toBe(false)
    })
    it('keeps local staged apply usable when distribution updates are unconfigured', async () => {
        updates.getStatus.mockResolvedValue({
            phase: 'error',
            currentVersion: 'test',
            latestVersion: null,
            downloadPercent: null,
            error: 'Distribution updates are not configured for this build'
        })
        await act(async () => root.unmount())
        root = createRoot(container)
        await act(async () => root.render(<ApplicationUpdatesCard />))
        await choose()
        expect(button('Apply build and restart').disabled).toBe(false)
    })
})

describe('reconnect and snapshot ordering', () => {
    it('keeps uncertain reconnect authority when a success notification arrives first', async () => {
        let complete: (value: Awaited<ReturnType<IBuildApplyApi['status']>>) => void = () => {
            throw new Error('Missing deferred status')
        }
        api.status.mockReturnValueOnce(
            new Promise(resolve => {
                complete = resolve
            })
        )
        await service('reconnecting')
        await publish(operation())
        expect(document.body.textContent).not.toContain('Build applied; node remains paused')
        await act(async () =>
            complete({
                available: false,
                value: null,
                uncertainOperation: operation(),
                refusal: { code: 'status_unknown', detail: 'Terminal publication is uncertain.' }
            })
        )
        expect(document.body.textContent).not.toContain('Build applied; node remains paused')
        expect(document.body.textContent).toContain('Terminal publication is uncertain.')
        expect(document.body.textContent).toContain(
            'nvpair-service apply-build recover operation_1'
        )
    })
    it('does not adopt an unrelated notification during or after a lost start reply', async () => {
        let complete: (value: Awaited<ReturnType<IBuildApplyApi['start']>>) => void = () => {
            throw new Error('Missing deferred start')
        }
        api.start.mockReturnValueOnce(
            new Promise(resolve => {
                complete = resolve
            })
        )
        api.status.mockResolvedValue({
            available: false,
            value: null,
            refusal: { code: 'outcome_unknown', detail: 'Outcome unknown. Check status.' }
        })
        await choose()
        await click('Apply build and restart')
        const modal = Array.from(document.querySelectorAll('dialog[open]')).at(-1)
        const checkbox = modal?.querySelector('input[type="checkbox"]')
        if (!modal || !(checkbox instanceof HTMLInputElement))
            throw new Error('Missing consent modal')
        await act(async () => checkbox.click())
        await click('Apply build and restart', modal)
        const unrelated = {
            ...operation(),
            operationId: 'operation_other',
            previewId: 'preview_other'
        }
        await publish(unrelated)
        await act(async () =>
            complete({
                available: false,
                value: null,
                refusal: { code: 'outcome_unknown', detail: 'Outcome unknown. Check status.' }
            })
        )
        await publish(unrelated)
        expect(document.body.textContent).not.toContain('Operation operation_other')
        expect(api.status.mock.calls.every(([request]) => request.operationId === '')).toBe(true)
        api.status.mockResolvedValue({ available: true, value: operation(), refusal: null })
        await click('Read build status')
        expect(document.body.textContent).toContain('Operation operation_1')
        expect(api.start).toHaveBeenCalledOnce()
    })
    it('discards late preview replies after disconnect and deduplicates repeated reconnect status', async () => {
        let complete: (
            value: Awaited<ReturnType<IBuildApplyApi['choosePreview']>>
        ) => void = () => {
            throw new Error('Missing deferred preview')
        }
        api.choosePreview.mockReturnValueOnce(
            new Promise(resolve => {
                complete = resolve
            })
        )
        await click('Choose staged build')
        await service('reconnecting')
        await act(async () => complete({ available: true, value: preview(), refusal: null }))
        expect(document.querySelector('dialog[open]')).toBeNull()
        api.status.mockClear()
        await service('connected')
        await service('connected')
        expect(api.status).toHaveBeenCalledOnce()
        expect(api.start).not.toHaveBeenCalled()
    })
    it('retires an old generation status without overwriting the reconnect result', async () => {
        let complete: (value: Awaited<ReturnType<IBuildApplyApi['status']>>) => void = () => {
            throw new Error('Missing deferred status')
        }
        api.status.mockReturnValueOnce(
            new Promise(resolve => {
                complete = resolve
            })
        )
        await click('Read build status')
        api.status.mockResolvedValue({ available: true, value: operation(), refusal: null })
        await service('connected')
        await act(async () =>
            complete({
                available: true,
                value: { ...operation(), outcome: 'failed', phase: 'swap' },
                refusal: null
            })
        )
        expect(document.body.textContent).toContain('Build applied; node remains paused')
        expect(document.body.textContent).not.toContain('Phase: swap')
    })
})
