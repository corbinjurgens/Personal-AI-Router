// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { createOverviewWindow } from '@/electron/window'
import { getModularBridgeState } from '@/electron/service-bridge/modular-state'
import { parseModelsResult } from '@/electron/service-bridge/model-download'
import type { JsonObject } from '@/electron/service-bridge/json-rpc-client'
import type {
    EngineInitialState,
    EngineStatePatch,
    EngineCommandPayload
} from '@/shared/types/engine-api'
import type { EngineProgress } from '@/shared/types/engines'
import type { ModelDownloadEntry } from '@/shared/types/model-download'
import type { PreloadServiceTransport } from '@/shared/types/service-bridge'
import { createEngineApi, type IEngineApi } from '@/ui/api/engine-api'
import { serviceTransport } from '@/ui/api/service-transport'
import { ModelManager } from '@/ui/components/ModelManager/ModelManager'
import { ModelHubContent } from '@/ui/components/ModelHub/ModelHubContent'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useConnectionStore } from '@/ui/stores/connection.store'
import type { BackendInfo } from '@/ui/types/engine-info'

vi.mock('electron', () => ({
    app: { isPackaged: false, getAppPath: () => process.cwd() },
    BrowserWindow: { getAllWindows: () => [] }
}))
vi.mock('@/electron/window', () => ({ createOverviewWindow: vi.fn<typeof createOverviewWindow>() }))

const MODEL = 'owner/model:Q4_K_M'
let sourceId: string
const bridge = getModularBridgeState()
const invoke = vi.fn<PreloadServiceTransport['invoke']>()
const stateListeners = new Set<Parameters<IEngineApi['onStateChanged']>[0]>()
const progressListeners = new Set<Parameters<IEngineApi['onProgress']>[0]>()
let root: Root
let container: HTMLDivElement
let nodeId: string
let caseId = 0
let activeMenuId = ''

function entry(changes: Partial<ModelDownloadEntry> = {}): ModelDownloadEntry {
    return {
        id: '1'.repeat(32),
        engine: 'llamacpp',
        model: MODEL,
        phase: 'in_progress',
        blocksReadiness: true,
        ...changes
    }
}

function wireEntry(row: ModelDownloadEntry): JsonObject {
    return { ...row }
}

function report(
    entries: ModelDownloadEntry[],
    names: string[] = [],
    loaded: string[] = [],
    revision = '1'
): void {
    bridge.applyLocalModelsResult(
        parseModelsResult({
            models: names,
            modelsByEngine: { llamacpp: names },
            loadedByEngine: { llamacpp: loaded },
            downloadState: {
                version: 1,
                sourceId,
                revision,
                recoveryRequired: false,
                entries: entries.map(wireEntry)
            }
        })
    )
}

function publish(initial = bridge.getEngineInitialState()): void {
    for (const models of initial.models.filter(row => row.nodeId === nodeId)) {
        const patch: EngineStatePatch = { nodeId, engineType: models.engineType, models }
        for (const listener of stateListeners) listener(patch)
    }
    for (const progress of initial.activeProgress.filter(row => row.nodeId === nodeId)) {
        for (const listener of progressListeners) listener(progress)
    }
}

function backend(): BackendInfo {
    return {
        type: 'llama-cpp',
        displayName: 'llama.cpp',
        source: 'installed',
        processStatus: 'running',
        port: 8081,
        proxyPort: 54303,
        models: useEngineModelsStore.getState().getModels(nodeId, 'llama-cpp').models
    }
}

async function mount(): Promise<void> {
    const initial = bridge.getEngineInitialState()
    await useEngineModelsStore.getState().initialize(initial)
    await useEngineProgressStore.getState().initialize(initial)
    usePendingActionsStore.getState().initialize()
    await act(async () => root.render(<ModelManager backend={backend()} nodeId={nodeId} />))
}

async function openActions(model = MODEL): Promise<void> {
    const name = [...container.querySelectorAll('[title]')].find(
        row => row.getAttribute('title') === model
    )
    const trigger = name?.parentElement?.parentElement?.querySelector(
        'button[aria-haspopup="menu"]'
    )
    if (!(trigger instanceof HTMLButtonElement)) throw new Error('Missing model actions')
    activeMenuId = trigger.getAttribute('aria-controls') ?? ''
    const menu = document.getElementById(activeMenuId)
    if (!menu) throw new Error('Missing model menu')
    await act(async () => {
        trigger.click()
        // jsdom does not implement native popovers; deliver the browser's
        // toggle notification to the real KUI dropdown and its focus handler.
        const toggle = new Event('toggle')
        Object.defineProperty(toggle, 'newState', { value: 'open' })
        menu.dispatchEvent(toggle)
    })
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
}

function item(label: string): HTMLElement {
    const menu = document.getElementById(activeMenuId)
    if (!menu) throw new Error('No open model menu')
    const found = [...menu.querySelectorAll('[role="menuitem"]')].find(
        row => row.textContent === label
    )
    if (!(found instanceof HTMLElement)) throw new Error(`Missing menu item ${label}`)
    return found
}

function expectDisabled(label: string): void {
    const target = item(label)
    expect(target.getAttribute('aria-disabled') === 'true' || target.hasAttribute('disabled')).toBe(
        true
    )
    act(() => target.dispatchEvent(new MouseEvent('click', { bubbles: true })))
}

type InvokePayload = (typeof invoke.mock.calls)[number][1]
function isCommand(payload: InvokePayload): payload is EngineCommandPayload {
    return typeof payload === 'object' && payload !== null && 'command' in payload
}
function commands(): EngineCommandPayload[] {
    return invoke.mock.calls.flatMap(([channel, payload]) =>
        channel === 'engine:command' && isCommand(payload) ? [payload] : []
    )
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false, escape: (value: string) => value })
    stateListeners.clear()
    progressListeners.clear()
    activeMenuId = ''
    invoke.mockReset().mockResolvedValue({ success: true, data: null })
    const api = createEngineApi(
        serviceTransport({
            connected: true,
            invoke,
            subscribePush: () => () => {},
            onConnect: () => () => {},
            onDisconnect: () => () => {},
            onAuthFailure: () => () => {},
            destroy: () => {}
        })
    )
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: {
            engines: {
                ...api,
                onStateChanged: (callback: Parameters<IEngineApi['onStateChanged']>[0]) => {
                    stateListeners.add(callback)
                    return () => stateListeners.delete(callback)
                },
                onProgress: (callback: Parameters<IEngineApi['onProgress']>[0]) => {
                    progressListeners.add(callback)
                    return () => progressListeners.delete(callback)
                },
                onProgressRemove: () => () => {},
                onSettingsChanged: () => () => {}
            },
            nodes: { onRemove: () => () => {} },
            errors: { onUpdate: () => () => {} }
        }
    })
    caseId++
    sourceId = caseId.toString(16).padStart(32, '0')
    nodeId = `readiness-${caseId}`
    bridge.setSelfId(nodeId)
    bridge.applyEngineManagerStatus({
        engine: 'llamacpp',
        installed: true,
        running: true,
        healthy: true
    })
    report([], [], [], '0')
    useEngineProgressStore.setState({ progress: new Map() })
    useEngineModelsStore.setState({ models: new Map() })
    useConnectionStore.setState({ selfId: 'controller' })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(() => {
    act(() => root.unmount())
    container.remove()
    useEngineModelsStore.getState().cleanup()
    useEngineProgressStore.getState().cleanup()
    usePendingActionsStore.getState().cleanup()
    vi.unstubAllGlobals()
})

describe('rendered target download readiness', () => {
    it('shows an unseeded target download once even when raw inventory names it, and suppresses menu clicks', async () => {
        report([entry({ percent: 21 })], [MODEL])
        await mount()
        expect(container.querySelectorAll(`[title="${MODEL}"]`)).toHaveLength(1)
        expect(container.textContent).toContain('Downloading')
        const progress = container.querySelector('[role="progressbar"]')
        expect(progress?.getAttribute('aria-label')).toBe(`Downloading ${MODEL}`)
        expect(progress?.getAttribute('aria-valuenow')).toBe('21')
        await openActions()
        for (const label of ['Load', 'Copy to this PC', 'Delete', 'Eject']) expectDisabled(label)
        expect(commands()).toEqual([])
    })

    it('keeps unknown percent indeterminate and preserves keyboard focus on progress', async () => {
        report([entry()])
        await mount()
        const button = container.querySelector('button')
        button?.focus()
        expect(container.querySelector('[role="progressbar"]')?.hasAttribute('aria-valuenow')).toBe(
            false
        )
        act(() => {
            report([entry({ percent: 0 })], [], [], '2')
            publish()
        })
        expect(container.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe(
            '0'
        )
        expect(document.activeElement).toBe(button)
    })

    it('shows two exact queued models without an invented progress percent', async () => {
        report([
            entry({ phase: 'queued' }),
            entry({ id: '2'.repeat(32), model: 'owner/second:Q8', phase: 'queued' })
        ])
        await mount()
        expect(container.textContent?.match(/Download queued/g)).toHaveLength(2)
        expect(container.querySelectorAll('[role="progressbar"]')).toHaveLength(0)
        expect(container.querySelectorAll('button[aria-label^="Open "]')).toHaveLength(2)
        await openActions('owner/second:Q8')
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
    })

    it('keeps a failed first shard unavailable even if inventory still lists it', async () => {
        report([entry({ phase: 'failed', reason: 'download_failed' })], [MODEL])
        await mount()
        expect(container.textContent).toContain('Download failed')
        expect(container.querySelector('[role="progressbar"]')).toBeNull()
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
    })

    it('waits for canonical inventory after completion, then allows an explicit Load without claiming loaded', async () => {
        const complete = entry({ phase: 'complete', blocksReadiness: false })
        report([complete])
        await mount()
        expect(container.textContent).toContain('Download complete')
        expect(container.textContent).toContain('Content readiness unknown')
        expect(container.textContent).not.toContain('Loaded')
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        act(() => {
            report([complete], [MODEL])
            publish()
        })
        await act(async () => item('Load').click())
        expect(commands()).toEqual([
            { command: 'loadModel', engineType: 'llama-cpp', nodeId, model: MODEL }
        ])
    })

    it('preserves observed loaded residency while an unsafe download blocks conflicting actions', async () => {
        report([entry({ percent: 21 })], [MODEL], [MODEL])
        await mount()
        expect(container.textContent).toContain('Loaded')
        expect(container.textContent).toContain('Downloading')
        expect(container.textContent).not.toContain('Download complete')
        await openActions()
        for (const label of ['Load', 'Copy to this PC', 'Eject', 'Delete']) expectDisabled(label)
        expect(commands()).toEqual([])
    })

    it('allows ready content after a pre-effect failure', async () => {
        report([entry({ phase: 'failed', blocksReadiness: false, reason: 'busy' })], [MODEL])
        await mount()
        expect(container.textContent).toContain('Download failed')
        await openActions()
        await act(async () => item('Load').click())
        expect(commands()[0]?.command).toBe('loadModel')
    })

    it('preserves existing Eject and Delete conditions after an unsafe terminal failure', async () => {
        report([entry({ phase: 'failed' })], [MODEL], [MODEL])
        await mount()
        await openActions()
        await act(async () => item('Eject').click())
        expect(commands()[0]?.command).toBe('unloadModel')
    })

    it('allows explicit Delete outside an active pull', async () => {
        report([entry({ phase: 'failed' })], [MODEL])
        await mount()
        await openActions()
        await act(async () => item('Delete').click())
        expect(commands()).toEqual([
            { command: 'deleteModel', engineType: 'llama-cpp', nodeId, model: MODEL }
        ])
    })

    it('does not certify supported loaded-only residency as downloaded or complete', async () => {
        report([], [], [MODEL])
        await mount()
        expect(container.textContent).toContain('Loaded')
        expect(container.textContent).toContain('Content readiness unknown')
        expect(container.textContent).not.toContain('Download complete')
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
    })

    it('renders an unresolved current outcome as unknown without a success bar', async () => {
        report([entry({ phase: 'unknown', reason: 'interrupted', percent: 100 })], [MODEL])
        await mount()
        expect(container.textContent).toContain('Download status unknown')
        expect(container.querySelector('[role="progressbar"]')).toBeNull()
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
    })

    it('keeps a newer pushed download while a delayed initial snapshot hydrates and does not stack subscriptions', async () => {
        report([entry({ phase: 'failed' })])
        const initial = bridge.getEngineInitialState()
        await useEngineModelsStore.getState().initialize(initial)
        let resolveInitial: (snapshot: EngineInitialState) => void = () => {
            throw new Error('Initial snapshot not requested')
        }
        vi.spyOn(window.pairApi.engines, 'getInitialState').mockReturnValue(
            new Promise(resolve => {
                resolveInitial = resolve
            })
        )
        const hydration = useEngineProgressStore.getState().initialize()
        const newer: EngineProgress = {
            nodeId,
            nodeName: '',
            engineType: 'llama-cpp',
            operation: 'pull',
            model: MODEL,
            status: 'in_progress',
            downloadState: entry({ id: '2'.repeat(32), percent: 42 }),
            downloadStateCurrent: true,
            downloadSourceId: sourceId,
            downloadRevision: '2'
        }
        for (const listener of progressListeners) listener(newer)
        resolveInitial(initial)
        await hydration
        await act(async () => root.render(<ModelManager backend={backend()} nodeId={nodeId} />))
        expect(container.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe(
            '42'
        )
        await act(async () =>
            useEngineProgressStore.getState().initialize({ ...initial, activeProgress: [newer] })
        )
        expect(progressListeners.size).toBe(2)
    })

    it('keeps a push received before supplied hydration from enabling Load or Copy', async () => {
        report([], [MODEL])
        await mount()
        const delayed = bridge.getEngineInitialState()
        act(() => {
            report([entry({ percent: 42 })], [MODEL], [], '2')
            publish()
        })
        await act(async () => {
            await useEngineModelsStore.getState().initialize(delayed)
            await useEngineProgressStore.getState().initialize(delayed)
        })
        await openActions()
        await act(async () => item('Load').click())
        expect(commands()).toEqual([])
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(container.textContent).toContain('Downloading')
        expect(commands()).toEqual([])
    })

    it('rejects lower revision model and progress pushes after accepting an active pull', async () => {
        report([entry({ phase: 'failed', blocksReadiness: false })], [MODEL])
        await mount()
        const older = bridge.getEngineInitialState()
        act(() => {
            report([entry({ id: '2'.repeat(32), percent: 42 })], [MODEL], [], '2')
            publish()
            publish(older)
        })
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(container.textContent).toContain('Downloading')
        expect(commands()).toEqual([])
    })

    it('invalidates current ready content on a disconnect push without accepting delayed ready hydration', async () => {
        report([], [MODEL])
        await mount()
        const delayed = bridge.getEngineInitialState()
        act(() => {
            bridge.invalidateModelObservations()
            publish()
        })
        await act(async () => {
            await useEngineModelsStore.getState().initialize(delayed)
            await useEngineProgressStore.getState().initialize(delayed)
        })
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(container.textContent).toContain('Disconnected or download status unavailable')
        expect(commands()).toEqual([])
    })

    it('treats an unavailable replacement source as unknown while preserving the last accepted identity', async () => {
        report([], [MODEL])
        await mount()
        const acceptedSource = sourceId
        sourceId = 'e'.repeat(32)
        report([], [MODEL])
        bridge.invalidateModelObservations()
        await act(async () => {
            const unavailable = bridge.getEngineInitialState()
            await useEngineModelsStore.getState().initialize(unavailable)
            await useEngineProgressStore.getState().initialize(unavailable)
        })
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(
            useEngineModelsStore.getState().getModels(nodeId, 'llama-cpp').downloadState?.sourceId
        ).toBe(acceptedSource)
        expect(commands()).toEqual([])
    })

    it('retains a completed progress fence omitted from active hydration until current inventory catches up', async () => {
        report([], [MODEL])
        await mount()
        const delayed = bridge.getEngineInitialState()
        const completed = entry({ phase: 'complete', blocksReadiness: false })
        const complete: EngineProgress = {
            nodeId,
            nodeName: '',
            engineType: 'llama-cpp',
            operation: 'pull',
            model: MODEL,
            status: 'complete',
            downloadState: completed,
            downloadStateCurrent: true,
            downloadSourceId: sourceId,
            downloadRevision: '2'
        }
        act(() => {
            for (const listener of progressListeners) listener(complete)
        })
        await act(async () => useEngineProgressStore.getState().initialize(delayed))
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
        act(() => {
            report([completed], [MODEL], [], '2')
            publish()
        })
        await act(async () => item('Load').click())
        expect(commands()[0]?.command).toBe('loadModel')
    })

    it('subscribes to model pushes before an internal initial fetch resolves', async () => {
        report([], [MODEL])
        const delayed = bridge.getEngineInitialState()
        let resolveInitial: (snapshot: EngineInitialState) => void = () => {
            throw new Error('Initial snapshot not requested')
        }
        vi.spyOn(window.pairApi.engines, 'getInitialState').mockReturnValue(
            new Promise(resolve => {
                resolveInitial = resolve
            })
        )
        const hydration = useEngineModelsStore.getState().initialize()
        report([entry({ percent: 42 })], [MODEL], [], '2')
        publish()
        resolveInitial(delayed)
        await hydration
        await useEngineProgressStore.getState().initialize({ ...delayed, activeProgress: [] })
        await act(async () => root.render(<ModelManager backend={backend()} nodeId={nodeId} />))
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(container.textContent).toContain('Downloading')
        expect(commands()).toEqual([])
    })

    it('rejects old source hydration and pushes after accepting a replacement source', async () => {
        report([], [MODEL])
        await mount()
        const retired = bridge.getEngineInitialState()
        sourceId = 'f'.repeat(32)
        act(() => {
            report([entry({ percent: 42 })], [MODEL])
            publish()
        })
        await act(async () => {
            await useEngineModelsStore.getState().initialize(retired)
            await useEngineProgressStore.getState().initialize(retired)
            publish(retired)
        })
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(container.textContent).toContain('Downloading')
        expect(
            useEngineModelsStore.getState().getModels(nodeId, 'llama-cpp').downloadState?.sourceId
        ).toBe(sourceId)
        expect(commands()).toEqual([])
    })

    it('ignores a model fetch from an older initialization generation', async () => {
        report([], [MODEL])
        const delayed = bridge.getEngineInitialState()
        let resolveInitial: (snapshot: EngineInitialState) => void = () => {
            throw new Error('Initial snapshot not requested')
        }
        vi.spyOn(window.pairApi.engines, 'getInitialState').mockReturnValue(
            new Promise(resolve => {
                resolveInitial = resolve
            })
        )
        const hydration = useEngineModelsStore.getState().initialize()
        report([entry({ percent: 42 })], [MODEL], [], '2')
        await mount()
        await act(async () => {
            resolveInitial(delayed)
            await hydration
        })
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(stateListeners.size).toBe(3)
        expect(commands()).toEqual([])
    })

    it('invalidates ready observations on cleanup and restores freshness only with a fresh read', async () => {
        report([], [MODEL])
        await mount()
        const delayed = bridge.getEngineInitialState()
        act(() => {
            useEngineModelsStore.getState().cleanup()
            useEngineProgressStore.getState().cleanup()
        })
        await act(async () => {
            await useEngineModelsStore.getState().initialize(delayed)
            await useEngineProgressStore.getState().initialize(delayed)
        })
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
        vi.spyOn(window.pairApi.engines, 'getInitialState').mockResolvedValue(delayed)
        await act(async () => {
            await useEngineModelsStore.getState().initialize()
            await useEngineProgressStore.getState().initialize()
        })
        await act(async () => item('Load').click())
        expect(commands()[0]?.command).toBe('loadModel')
    })

    it('clears Starting download when a new failed operation arrives after missed active frames', async () => {
        report([entry({ phase: 'failed', blocksReadiness: false })], [MODEL])
        await mount()
        act(() =>
            usePendingActionsStore.getState().begin({
                command: 'pullModel',
                engineType: 'llama-cpp',
                nodeId,
                model: MODEL
            })
        )
        expect(container.textContent).toContain('Starting download...')
        act(() => {
            report(
                [
                    entry({
                        id: '2'.repeat(32),
                        phase: 'failed',
                        blocksReadiness: false,
                        reason: 'busy'
                    })
                ],
                [MODEL],
                [],
                '2'
            )
            publish()
        })
        expect(container.textContent).not.toContain('Starting download...')
        expect(container.textContent).toContain('Download failed')
        expect(
            usePendingActionsStore.getState().getModelPending(nodeId, 'llama-cpp', MODEL)
        ).toBeUndefined()
        await openActions()
        await act(async () => item('Load').click())
        expect(commands()[0]?.command).toBe('loadModel')
    })

    it('shows stale disconnected facts without a completed bar or ready actions', async () => {
        report([entry({ percent: 100 })], [MODEL])
        bridge.invalidateModelObservations()
        await mount()
        expect(container.textContent).toContain('Download status unknown')
        expect(container.textContent).toContain('Disconnected or download status unavailable')
        expect(container.querySelector('[role="progressbar"]')).toBeNull()
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
    })

    it('treats missing lifecycle support as unknown despite a listed loaded model', async () => {
        bridge.applyLocalModelsResult(
            parseModelsResult({
                models: [MODEL],
                modelsByEngine: { llamacpp: [MODEL] },
                loadedByEngine: { llamacpp: [MODEL] }
            })
        )
        await mount()
        expect(container.textContent).toContain('Loaded')
        expect(container.textContent).toContain('Download status unknown')
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
    })

    it('shows unavailable status even when an unsupported target has no model rows', async () => {
        bridge.applyLocalModelsResult(
            parseModelsResult({ models: [], modelsByEngine: { llamacpp: [] } })
        )
        await mount()
        expect(container.textContent).toContain('Download status unknown')
        expect(container.textContent).toContain('Disconnected or download status unavailable')
        expect(container.textContent).not.toContain('No model/weight files downloaded')
        expect(commands()).toEqual([])
    })

    it('rechecks current store facts when an already-open menu is clicked before React rerenders', async () => {
        report([], [MODEL])
        await mount()
        await openActions()
        const load = item('Load')
        act(() => {
            report([entry({ percent: 21 })], [MODEL], [], '2')
            publish()
            load.click()
        })
        expect(commands()).toEqual([])
    })

    it('hands origin pending state to exact authoritative progress and rejects a late old terminal', async () => {
        report([entry({ id: '1'.repeat(32), phase: 'failed' })], [MODEL])
        await mount()
        act(() =>
            usePendingActionsStore
                .getState()
                .begin({ command: 'pullModel', engineType: 'llama-cpp', nodeId, model: MODEL })
        )
        act(() => {
            report([entry({ id: '2'.repeat(32), percent: 12 })], [MODEL], [], '3')
            publish()
        })
        expect(
            usePendingActionsStore.getState().getModelPending(nodeId, 'llama-cpp', MODEL)
        ).toBeUndefined()
        const old: EngineProgress = {
            nodeId,
            nodeName: '',
            engineType: 'llama-cpp',
            operation: 'pull',
            model: MODEL,
            status: 'complete',
            downloadState: entry({ phase: 'complete', blocksReadiness: false }),
            downloadStateCurrent: true,
            downloadSourceId: sourceId,
            downloadRevision: '2'
        }
        act(() => {
            for (const listener of progressListeners) listener(old)
        })
        expect(container.textContent).toContain('Downloading')
        expect(container.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe(
            '12'
        )
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
    })

    it('does not certify an earlier ready inventory from a newer complete progress event', async () => {
        report([], [MODEL])
        await mount()
        await openActions()
        const load = item('Load')
        const complete: EngineProgress = {
            nodeId,
            nodeName: '',
            engineType: 'llama-cpp',
            operation: 'pull',
            model: MODEL,
            status: 'complete',
            downloadState: entry({ phase: 'complete', blocksReadiness: false }),
            downloadStateCurrent: true,
            downloadSourceId: sourceId,
            downloadRevision: '2'
        }
        act(() => {
            for (const listener of progressListeners) listener(complete)
            load.click()
        })
        expect(container.textContent).toContain('Download complete')
        expect(container.textContent).toContain('Content readiness unknown')
        expect(commands()).toEqual([])
    })

    it('hydrates another observer from discovery without an initiating request', async () => {
        const remote = `${nodeId}-remote`
        bridge.handleNotification({
            source: 'broker',
            method: 'discovery:nodes-changed',
            params: {
                nodes: [
                    {
                        hostUuid: remote,
                        name: 'Target',
                        ipAddress: '127.0.0.2',
                        port: 1234,
                        models: [MODEL],
                        modelsByEngine: { llamacpp: [MODEL] },
                        downloadState: {
                            version: 1,
                            sourceId,
                            revision: '1',
                            recoveryRequired: false,
                            entries: [wireEntry(entry({ percent: 21 }))]
                        },
                        downloadStateCurrent: true
                    }
                ]
            }
        })
        bridge.applyRemoteEngineFacts(remote, {
            engines: [{ engine: 'llamacpp', installed: true, running: true, port: 8081 }]
        })
        nodeId = remote
        await mount()
        expect(container.textContent).toContain('Downloading')
        await openActions()
        expectDisabled('Load')
        expectDisabled('Copy to this PC')
        expect(commands()).toEqual([])
    })

    it('keeps failed content in ModelHub and retries only on explicit catalog selection', async () => {
        report([entry({ phase: 'failed' })], [MODEL])
        await mount()
        const search = vi.spyOn(window.pairApi.engines, 'searchHub').mockResolvedValue({
            models: [
                {
                    id: MODEL,
                    name: MODEL,
                    author: 'owner',
                    url: '',
                    updatedAt: '2026-10-25T00:00:00Z',
                    downloads: 0,
                    likes: 0,
                    tags: []
                }
            ]
        })
        await act(async () =>
            root.render(
                <ModelHubContent
                    engine="llama-cpp"
                    multiple={false}
                    downloadedModels={
                        useEngineModelsStore.getState().getModels(nodeId, 'llama-cpp').models
                    }
                    onSubmit={models => {
                        for (const model of models)
                            window.pairApi.engines.pullModel('llama-cpp', nodeId, model.name)
                    }}
                />
            )
        )
        expect(search).toHaveBeenCalledOnce()
        expect(commands()).toEqual([])
        const row = container.querySelector('[data-vlist-index="0"] > div')
        if (!(row instanceof HTMLElement)) throw new Error('Failed model missing from catalog')
        await act(async () => row.click())
        expect(commands()).toEqual([
            { command: 'pullModel', engineType: 'llama-cpp', nodeId, model: MODEL }
        ])
    })
})
