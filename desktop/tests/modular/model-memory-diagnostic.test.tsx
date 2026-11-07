// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { InspectModelResult, MeasuredProfile } from '@/shared/types/model-config'
import type { DiagnosticPreview, DiagnosticRun } from '@/shared/types/model-diagnostic'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type { EngineDescriptor } from '@/shared/types/engines'
import type { EngineSettingsSnapshot } from '@/shared/types/engine-settings'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import type { IEngineApi } from '@/ui/api/engine-api'
import { createPairApi } from '@/ui/api/pair-api'
import ModelMemoryDiagnostic from '@/ui/components/ModelSettings/ModelMemoryDiagnostic'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'

const target = { nodeId: 'self', engine: 'llamacpp', modelId: 'model', profileKey: 'profile' }
const inspection: InspectModelResult = {
    ...target,
    status: 'resolved',
    configurationKey: 'configuration',
    nativeBuildId: 'current-build',
    estimate: { status: 'unknown' }
}
const preview: DiagnosticPreview = {
    ...target,
    configurationKey: 'configuration',
    nativeBuildId: 'current-build',
    buildInfo: 'llama build',
    cpuModel: 'CPU',
    totalRamBytes: '34359738368',
    gpus: [],
    artifacts: [{ role: 'model', identity: 'artifact', size: '1024', assurance: 'local' }],
    requested: {},
    perSlotContext: 2048,
    slots: 1,
    placement: [{ domainId: 'host', kind: 'host' }],
    previewId: 'reviewed-effects',
    plannedPerSlotContext: 2048,
    plannedSlots: 1,
    lowPromptTokens: 512,
    highPromptTokens: 1904,
    generateTokens: 128,
    timeLimitSeconds: 900,
    unloadModels: [{ engine: 'llamacpp', modelId: 'other', profileKey: 'other-profile' }],
    reloadedAfter: [{ engine: 'llamacpp', modelId: 'other', profileKey: 'other-profile' }],
    drainEngines: ['llamacpp', 'ollama'],
    startsEngine: true,
    warnings: ['Configured pause is preserved.', 'RAM use exceeds half of total RAM.'],
    expiresAt: '2026-11-07T01:05:00Z'
}
const running: DiagnosticRun = {
    ...target,
    runId: 'run',
    configurationKey: 'configuration',
    phase: 'loading',
    outcome: 'running',
    startedAt: '2026-11-07T01:00:00Z',
    elapsedMs: 12000
}
const result: MeasuredProfile = {
    configurationKey: 'configuration',
    nativeBuildId: 'current-build',
    buildInfo: 'llama build',
    cpuModel: 'CPU',
    totalRamBytes: '34359738368',
    gpus: [],
    artifacts: preview.artifacts,
    requested: {},
    perSlotContext: 2048,
    slots: 1,
    placement: preview.placement,
    runId: 'run',
    recordedAt: '2026-11-07T01:01:00Z',
    outcome: 'completed',
    buildPin: 'pin-123',
    pools: [
        {
            domainId: 'host',
            kind: 'host',
            baselineAvailableBytes: '8589934592',
            peakUsedBytes: '2147483648',
            source: 'scanner',
            samples: 20,
            intervalMs: 2000
        }
    ],
    low: {
        coverage: 'complete',
        requests: [
            {
                targetPromptTokens: 512,
                promptTokens: 511,
                promptPerSecond: 100,
                decodePerSecond: 20,
                generatedTokens: 128
            }
        ],
        combinedGeneratedPerSecond: 20,
        wallMs: 7000
    },
    high: {
        coverage: 'partial',
        requests: [
            {
                targetPromptTokens: 1904,
                promptTokens: 1800,
                promptPerSecond: 90,
                decodePerSecond: 18,
                generatedTokens: 128
            }
        ],
        combinedGeneratedPerSecond: 18,
        wallMs: 8000,
        overlapWindowSampled: false
    },
    notes: ['Full context was not filled.']
}
const getPreview = vi.fn<IEngineApi['diagnosticPreview']>()
const start = vi.fn<IEngineApi['diagnosticStart']>()
const status = vi.fn<IEngineApi['diagnosticStatus']>()
const cancel = vi.fn<IEngineApi['diagnosticCancel']>()
const subscribe = vi.fn<IEngineApi['onDiagnosticChanged']>()
const inspect = vi.fn<IEngineApi['inspectModel']>()
const subscribeSettings = vi.fn<IEngineApi['onSettingsChanged']>()
const subscribeSettingsDisconnected = vi.fn<IEngineApi['onSettingsDisconnected']>()
let changed: ((run: DiagnosticRun) => void) | undefined
let settingsChanged: ((snapshot: EngineSettingsSnapshot) => void) | undefined
let settingsDisconnected: ((target: { nodeId: string }) => void) | undefined
let root: Root
let container: HTMLDivElement

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false })
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-11-07T01:00:00Z'))
    Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
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
    const transport: ServiceTransport = {
        connected: true,
        invoke: vi.fn<ServiceTransport['invoke']>(),
        subscribePush: () => () => {},
        onConnect: () => () => {},
        onDisconnect: () => () => {},
        onAuthFailure: () => () => {},
        destroy: () => {}
    }
    const api = createPairApi(transport)
    getPreview.mockReset().mockResolvedValue(preview)
    start.mockReset().mockImplementation(async () => {
        status.mockResolvedValue(running)
        return running
    })
    status.mockReset().mockResolvedValue(null)
    cancel.mockReset().mockResolvedValue(null)
    inspect.mockReset().mockResolvedValue(inspection)
    subscribeSettings.mockReset().mockImplementation(callback => {
        settingsChanged = callback
        return () => {
            settingsChanged = undefined
        }
    })
    subscribeSettingsDisconnected.mockReset().mockImplementation(callback => {
        settingsDisconnected = callback
        return () => {
            settingsDisconnected = undefined
        }
    })
    subscribe.mockReset().mockImplementation(callback => {
        changed = callback
        return () => {
            changed = undefined
        }
    })
    api.engines.diagnosticPreview = getPreview
    api.engines.diagnosticStart = start
    api.engines.diagnosticStatus = status
    api.engines.diagnosticCancel = cancel
    api.engines.onDiagnosticChanged = subscribe
    api.engines.inspectModel = inspect
    api.engines.onSettingsChanged = subscribeSettings
    api.engines.onSettingsDisconnected = subscribeSettingsDisconnected
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ connected: true, selfId: 'self' })
    useEngineSettingsStore.setState({ diagnostics: {}, modelDrafts: {}, entries: {} })
    useEngineStatusStore.setState({ descriptors: new Map() })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.useRealTimers()
    vi.unstubAllGlobals()
})

async function render(value = inspection, applied = true, disabled = false) {
    await act(async () =>
        root.render(
            <ModelMemoryDiagnostic
                {...target}
                inspection={value}
                applied={applied}
                disabled={disabled}
            />
        )
    )
}
function button(label: string): HTMLButtonElement {
    const found = Array.from(document.querySelectorAll('button')).find(
        item => item.textContent === label
    )
    if (!found) throw new Error(`Missing button: ${label}`)
    return found
}
async function click(label: string) {
    await act(async () => button(label).click())
}
async function push(run: DiagnosticRun) {
    const callback = changed
    if (!callback) throw new Error('Diagnostic push listener was not installed')
    await act(async () => callback(run))
}

describe('applied profile memory measurement', () => {
    it('requires an applicable configuration or execution key on an applied profile', async () => {
        await render({ ...inspection, configurationKey: undefined })
        expect(container.textContent).not.toContain('Measure this configuration')
        expect(getPreview).not.toHaveBeenCalled()
        await render(inspection, false)
        expect(container.textContent).not.toContain('Measure this configuration')
        expect(getPreview).not.toHaveBeenCalled()
    })

    it('does not preview or measure unsaved drafts', async () => {
        await render(inspection, true, true)
        expect(button('Measure this configuration').disabled).toBe(true)
        expect(getPreview).not.toHaveBeenCalled()
        expect(container.textContent).toContain('Save or discard draft changes')
    })

    it('accepts an applied descriptor-backed llama family profile', async () => {
        const engineType = engineTypeFromManagerName('llama-variant')
        if (!engineType) throw new Error('Variant engine identity was refused')
        const descriptor: EngineDescriptor = {
            engineType,
            family: 'llamacpp',
            label: 'Llama variant',
            available: true,
            protocols: [],
            modelNaming: '',
            modelStore: '',
            catalog: '',
            operations: {}
        }
        useEngineStatusStore.setState({ descriptors: new Map([[engineType, descriptor]]) })
        getPreview.mockResolvedValue({ ...preview, engine: 'llama-variant' })
        await act(async () =>
            root.render(
                <ModelMemoryDiagnostic
                    {...target}
                    engine="llama-variant"
                    applied
                    inspection={{ ...inspection, engine: 'llama-variant' }}
                />
            )
        )
        expect(button('Measure this configuration').disabled).toBe(false)
        expect(getPreview).toHaveBeenCalledExactlyOnceWith({ ...target, engine: 'llama-variant' })
    })

    it('shows all consent effects and sends the reviewed preview ID with explicit consent', async () => {
        await render()
        expect(getPreview).toHaveBeenCalledExactlyOnceWith(target)
        expect(subscribe.mock.invocationCallOrder[0]).toBeLessThan(
            status.mock.invocationCallOrder[0]
        )
        await click('Measure this configuration')
        const dialog = document.querySelector('dialog')
        if (!dialog) throw new Error('Missing consent dialog')
        expect(dialog.textContent).toContain('Models unloaded: llamacpp: other (other-profile)')
        expect(dialog.textContent).toContain(
            'Models reloaded afterwards: llamacpp: other (other-profile)'
        )
        expect(dialog.textContent).toContain('Engines held: llamacpp, ollama')
        expect(dialog.textContent).toContain('Starts llamacpp')
        expect(dialog.textContent).toContain('Duration bound: 900 seconds')
        expect(dialog.textContent).toContain('Other GPU activity')
        expect(dialog.textContent).toContain('Configured pause is preserved.')
        expect(dialog.textContent).toContain('RAM use exceeds half of total RAM.')
        expect(dialog.textContent).toContain('Profile profile, model model (llamacpp)')
        expect(dialog.textContent).toContain('CPU: CPU. RAM: 32 GB.')
        expect(dialog.textContent).toContain(
            'This uses as much memory as the configuration needs at full context. If it does not fit, the engine can stop; PAIR will record that.'
        )
        expect(start).not.toHaveBeenCalled()
        await click('Measure')
        expect(start).toHaveBeenCalledExactlyOnceWith({
            ...target,
            consent: true,
            previewId: 'reviewed-effects'
        })
    })

    it('Cancel in the consent modal has no execution effects', async () => {
        await render()
        await click('Measure this configuration')
        await click('Cancel')
        expect(start).not.toHaveBeenCalled()
        expect(cancel).not.toHaveBeenCalled()
    })

    it('shows backend-owned Halt refusal and disables Measure', async () => {
        getPreview.mockRejectedValue(new Error('Halt refuses diagnostic activation.'))
        await render()
        expect(button('Measure this configuration').disabled).toBe(true)
        expect(container.textContent).toContain('Halt refuses diagnostic activation.')
    })

    it('expires consent together with the reviewed preview', async () => {
        await render()
        await click('Measure this configuration')
        await act(async () => vi.advanceTimersByTimeAsync(300001))
        expect(button('Measure').disabled).toBe(true)
        expect(container.textContent).toContain('Measurement preview expired.')
        expect(start).not.toHaveBeenCalled()
    })

    it('shows push phase and elapsed time and cancels by run ID', async () => {
        await render()
        await push({ ...running, phase: 'high', elapsedMs: 42100 })
        expect(container.textContent).toContain('Measuring memory: high, 42 s elapsed')
        await click('Cancel measurement')
        expect(cancel).toHaveBeenCalledExactlyOnceWith({ nodeId: 'self', runId: 'run' })
    })

    it('retries cleanup through the cancel method while engines stay held', async () => {
        await render()
        await push({
            ...running,
            outcome: 'cleanup-failed',
            phase: 'restoring',
            reason: 'Stop timed out'
        })
        expect(container.textContent).toContain('Engines remain held until cleanup succeeds.')
        await click('Retry cleanup')
        expect(cancel).toHaveBeenCalledExactlyOnceWith({ nodeId: 'self', runId: 'run' })
    })

    it('reads the cancelled run by ID instead of losing cleanup failure to active discovery', async () => {
        await render()
        await push(running)
        const held: DiagnosticRun = {
            ...running,
            outcome: 'cleanup-failed',
            phase: 'restoring',
            reason: 'Child still running'
        }
        cancel.mockResolvedValueOnce(held)
        status.mockImplementation(async request => (request.runId === 'run' ? held : null))
        await click('Cancel measurement')
        expect(status).toHaveBeenLastCalledWith({ nodeId: 'self', runId: 'run' })
        expect(button('Retry cleanup').disabled).toBe(false)
        expect(container.textContent).toContain('Engines remain held until cleanup succeeds.')
    })

    it('discovers a peer-started run after reconnect and retires old status reads', async () => {
        await render()
        await act(async () => useConnectionStore.setState({ connected: false }))
        status.mockResolvedValue(running)
        await act(async () => useConnectionStore.setState({ connected: true }))
        expect(status).toHaveBeenLastCalledWith({ nodeId: 'self' })
        expect(container.textContent).toContain('Measuring memory: loading, 12 s elapsed')
        expect(button('Cancel measurement').disabled).toBe(false)
    })

    it('keeps a push that arrives before the active-status snapshot resolves', async () => {
        let resolveStatus: (value: DiagnosticRun | null) => void = () => {
            throw new Error('Missing resolver')
        }
        status.mockImplementation(
            () =>
                new Promise(resolve => {
                    resolveStatus = resolve
                })
        )
        await render()
        await push({ ...running, phase: 'high', elapsedMs: 42000 })
        await act(async () => resolveStatus(running))
        expect(container.textContent).toContain('Measuring memory: high, 42 s elapsed')
    })

    it('invalidates an in-flight preview when an active run is discovered', async () => {
        let resolvePreview: ((value: DiagnosticPreview) => void) | undefined
        getPreview.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    resolvePreview = resolve
                })
        )
        await render()
        await push(running)
        if (!resolvePreview) throw new Error('Preview request was not sent')
        await act(async () => resolvePreview?.(preview))
        expect(useEngineSettingsStore.getState().diagnostics.self.preview).toBeUndefined()
        expect(button('Measure this configuration').disabled).toBe(true)
    })

    it('reconciles an accepted run by ID even while discovery is pending', async () => {
        await render()
        let resolveDiscovery: ((value: DiagnosticRun | null) => void) | undefined
        status.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    resolveDiscovery = resolve
                })
        )
        let discovery: Promise<void> | undefined
        await act(async () => {
            discovery = useEngineSettingsStore.getState().readDiagnostic('self')
        })
        const completed: DiagnosticRun = {
            ...running,
            phase: 'done',
            outcome: 'completed',
            result
        }
        start.mockResolvedValueOnce(running)
        status.mockResolvedValueOnce(completed)
        await click('Measure this configuration')
        await click('Measure')
        expect(status).toHaveBeenLastCalledWith({ nodeId: 'self', runId: 'run' })
        expect(useEngineSettingsStore.getState().diagnostics.self.run).toEqual(completed)
        if (!resolveDiscovery || !discovery) throw new Error('Discovery request was not sent')
        await act(async () => {
            resolveDiscovery?.(null)
            await discovery
        })
        expect(useEngineSettingsStore.getState().diagnostics.self.run).toEqual(completed)
    })

    it('rejects an older same-run status read after a newer phase push', async () => {
        await render()
        await push({ ...running, phase: 'high', elapsedMs: 42000 })
        status.mockResolvedValueOnce(running)
        await act(async () => useEngineSettingsStore.getState().readDiagnostic('self'))
        expect(container.textContent).toContain('Measuring memory: high, 42 s elapsed')
    })

    it('reconciles uncertain start acceptance even while an older discovery read is pending', async () => {
        await render()
        let resolveDiscovery: ((value: DiagnosticRun | null) => void) | undefined
        status.mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    resolveDiscovery = resolve
                })
        )
        let discovery: Promise<void> | undefined
        await act(async () => {
            discovery = useEngineSettingsStore.getState().readDiagnostic('self')
        })
        start.mockRejectedValueOnce(new Error('Start reply lost'))
        status.mockResolvedValueOnce(running)
        await click('Measure this configuration')
        await click('Measure')
        expect(useEngineSettingsStore.getState().diagnostics.self.run).toEqual(running)
        expect(status).toHaveBeenLastCalledWith({ nodeId: 'self' })
        if (!resolveDiscovery || !discovery) throw new Error('Discovery request was not sent')
        await act(async () => {
            resolveDiscovery?.(null)
            await discovery
        })
        expect(useEngineSettingsStore.getState().diagnostics.self.run).toEqual(running)
    })

    it('discovers retained cleanup after manager settings reconnect without service reconnect', async () => {
        await render()
        if (!settingsDisconnected || !settingsChanged) throw new Error('Settings watchers missing')
        await act(async () => settingsDisconnected?.({ nodeId: 'self' }))
        expect(useEngineSettingsStore.getState().diagnostics.self.current).toBe(false)
        status.mockResolvedValueOnce({ ...running, outcome: 'cleanup-failed', phase: 'restoring' })
        const snapshot: EngineSettingsSnapshot = {
            nodeId: 'self',
            engine: 'llamacpp',
            revision: 1,
            appliedRevision: 1,
            sequence: 1,
            epoch: 'replacement',
            settings: { serverPort: 1, proxyPort: 2, launchText: '' },
            effectiveServerPort: 1,
            effectiveProxyPort: 2,
            running: false,
            adopted: false,
            editable: true,
            reason: '',
            format: '',
            phase: 'idle',
            error: '',
            requestId: ''
        }
        await act(async () => settingsChanged?.(snapshot))
        expect(button('Retry cleanup').disabled).toBe(false)
        expect(status).toHaveBeenLastCalledWith({ nodeId: 'self' })
    })

    it('ignores a late start receipt and keeps cancellation pending until cleanup settles', async () => {
        let resolveStart: (value: DiagnosticRun) => void = () => {
            throw new Error('Missing start resolver')
        }
        let resolveCancel: (value: DiagnosticRun | null) => void = () => {
            throw new Error('Missing cancel resolver')
        }
        start.mockImplementation(
            () =>
                new Promise(resolve => {
                    resolveStart = resolve
                })
        )
        cancel.mockImplementation(
            () =>
                new Promise(resolve => {
                    resolveCancel = resolve
                })
        )
        await render()
        await click('Measure this configuration')
        await click('Measure')
        const high: DiagnosticRun = { ...running, phase: 'high', elapsedMs: 42000 }
        await push(high)
        expect(button('Cancel measurement').disabled).toBe(false)
        expect(container.textContent).toContain('Measuring memory: high, 42 s elapsed')
        await click('Cancel measurement')
        await act(async () => resolveStart(running))
        expect(button('Cancel measurement').disabled).toBe(true)
        const failed: DiagnosticRun = { ...high, outcome: 'cleanup-failed', phase: 'restoring' }
        await push(failed)
        expect(button('Retry cleanup').disabled).toBe(true)
        status.mockResolvedValue(failed)
        await act(async () => resolveCancel(failed))
        expect(button('Retry cleanup').disabled).toBe(false)
    })

    it('renders sampled pools, actual request tokens and rates, throughput and partial coverage', async () => {
        await render()
        await push({ ...running, phase: 'done', outcome: 'completed', result })
        expect(container.textContent).toContain('Measured peak 2 GB')
        expect(container.textContent).toContain('Baseline available: 8 GB')
        expect(container.textContent).toContain('interval 2000 ms')
        expect(container.textContent).toContain('Sampling can miss brief peaks.')
        expect(container.textContent).toContain('Short prompt (512 tokens)')
        expect(container.textContent).toContain('actual prompt 511 tokens')
        expect(container.textContent).toContain('prompt rate 100.0 tokens/s')
        expect(container.textContent).toContain('decode rate 20.0 tokens/s')
        expect(container.textContent).toContain('Full context (1904 tokens)')
        expect(container.textContent).toContain('Combined throughput: 18.0 tokens/s')
        expect(container.textContent).toContain('Partial coverage.')
        expect(container.textContent).toContain(
            'Measured on this machine, 2026-11-07T01:01:00Z, build pin-123'
        )
    })

    it('renders failed phase and reason without manufacturing measured peaks', async () => {
        await render()
        await push({
            ...running,
            phase: 'done',
            outcome: 'failed',
            failedPhase: 'loading',
            reason: 'Engine exited'
        })
        expect(container.textContent).toContain('Did not complete: loading - Engine exited')
        expect(container.textContent).not.toContain('Measured peak')
    })

    it('refreshes the measured profile after completion without inferring currentness from its key', async () => {
        await render()
        await push({ ...running, phase: 'done', outcome: 'completed', result })
        expect(inspect).toHaveBeenCalledExactlyOnceWith(target)
        expect(container.textContent).toContain('Measurement currentness awaits inspection')
        expect(container.textContent).not.toContain('Current measurement for this configuration.')
        await render({
            ...inspection,
            memoryProfiles: [
                {
                    configurationKey: 'configuration',
                    kind: 'measured',
                    recordedAt: result.recordedAt,
                    current: false,
                    measured: result
                }
            ]
        })
        expect(container.textContent).toContain(
            'Historical measurement; configuration or machine identity'
        )
        expect(container.textContent).not.toContain('Current measurement for this configuration.')
    })

    it('refreshes inspection again when the broker publishes the persisted result after terminal status', async () => {
        await render()
        await push({ ...running, phase: 'done', outcome: 'completed' })
        expect(inspect).toHaveBeenCalledTimes(1)
        await push({ ...running, phase: 'done', outcome: 'completed', result })
        expect(inspect).toHaveBeenCalledTimes(2)
    })

    it('uses target-inspected currentness and keeps missing phase observations absent', async () => {
        const incomplete = { ...result, low: undefined, high: undefined }
        await render({
            ...inspection,
            memoryProfiles: [
                {
                    configurationKey: 'configuration',
                    kind: 'measured',
                    recordedAt: result.recordedAt,
                    current: true,
                    measured: incomplete
                }
            ]
        })
        expect(container.textContent).toContain('Current measurement for this configuration.')
        expect(container.textContent).toContain('Short prompt: Not sampled.')
        expect(container.textContent).toContain('Full context: Not sampled.')
        expect(container.textContent).not.toContain('Complete coverage.')
    })

    it('labels earlier-build records as historical evidence', async () => {
        await render({
            ...inspection,
            memoryProfiles: [
                {
                    configurationKey: 'old-key',
                    kind: 'measured',
                    recordedAt: result.recordedAt,
                    current: false,
                    measured: { ...result, nativeBuildId: 'old-build' }
                }
            ]
        })
        expect(container.textContent).toContain('Earlier build: measured on pin-123')
        expect(container.textContent).toContain('measure the current build')
    })

    it('measures unpriced native-fit execution keys and discloses post-load targets', async () => {
        const unpriced = {
            ...preview,
            configurationKey: undefined,
            executionKey: 'execution',
            configurationKeyReason: 'native-fit',
            plannedPerSlotContext: 0,
            perSlotContext: 0,
            lowPromptTokens: 0,
            highPromptTokens: 0
        } satisfies DiagnosticPreview
        getPreview.mockResolvedValue(unpriced)
        await render({
            ...inspection,
            configurationKey: undefined,
            executionKey: 'execution',
            configurationKeyReason: 'native-fit'
        })
        expect(button('Measure this configuration').disabled).toBe(false)
        await click('Measure this configuration')
        expect(document.querySelector('dialog')?.textContent).toContain(
            'Native-fit context and prompt targets are determined after load.'
        )
        await push({
            ...running,
            configurationKey: undefined,
            executionKey: 'execution',
            phase: 'done',
            outcome: 'completed',
            result: {
                ...result,
                configurationKey: undefined,
                executionKey: 'execution',
                configurationKeyReason: 'native-fit'
            }
        })
        expect(container.textContent).toContain('Diagnostic only (native-fit)')
        expect(container.textContent).toContain(
            'No memory estimate or estimate comparison is available'
        )
    })

    it('offers diagnostic-only measurement for GPU-selection execution keys', async () => {
        getPreview.mockResolvedValue({
            ...preview,
            configurationKey: undefined,
            executionKey: 'gpu-execution',
            configurationKeyReason: 'gpu-selection'
        })
        await render({
            ...inspection,
            configurationKey: undefined,
            executionKey: 'gpu-execution',
            configurationKeyReason: 'gpu-selection'
        })
        expect(button('Measure this configuration').disabled).toBe(false)
        await push({
            ...running,
            configurationKey: undefined,
            executionKey: 'gpu-execution',
            phase: 'done',
            outcome: 'completed',
            result: {
                ...result,
                configurationKey: undefined,
                executionKey: 'gpu-execution',
                configurationKeyReason: 'gpu-selection'
            }
        })
        expect(container.textContent).toContain('Diagnostic only (gpu-selection)')
        expect(container.textContent).not.toContain('Estimate source')
    })
})
