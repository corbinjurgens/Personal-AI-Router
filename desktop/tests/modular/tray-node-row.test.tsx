// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { DiagnosticRun } from '@/shared/types/model-diagnostic'
import type { NodeItem } from '@/shared/types/nodes'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type { IEngineApi } from '@/ui/api/engine-api'
import { createPairApi } from '@/ui/api/pair-api'
import ModelMemoryDiagnostic from '@/ui/components/ModelSettings/ModelMemoryDiagnostic'
import TrayNodeRow from '@/ui/components/TrayApp/TrayNodeRow'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useEngineSettingsStore } from '@/ui/stores/engine-settings.store'
import { useEngineStatusStore } from '@/ui/stores/engine-status.store'
import { useMetricsStore } from '@/ui/stores/metrics.store'

const node: NodeItem = {
    id: 'self',
    name: 'Target',
    status: 'active',
    ipAddress: '127.0.0.1',
    port: 1,
    allIpAddresses: ['127.0.0.1'],
    os: 'MacOS',
    topology: {
        cpu: { model: 'Fixture CPU', cores: 1, threads: 1 },
        ram: 8192,
        gpus: [],
        storage: []
    }
}
const run: DiagnosticRun = {
    runId: 'peer-started-run',
    engine: 'llamacpp',
    modelId: 'fixture.gguf',
    profileKey: 'profile',
    configurationKey: 'configuration',
    phase: 'high',
    outcome: 'running',
    startedAt: '2026-11-07T01:00:00Z',
    elapsedMs: 2000
}
const status = vi.fn<IEngineApi['diagnosticStatus']>()
const cancel = vi.fn<IEngineApi['diagnosticCancel']>()
const focusNode = vi.fn<Window['windowApi']['window']['focusNode']>()
let changed: ((value: DiagnosticRun) => void) | undefined
let root: Root
let container: HTMLDivElement
const originalPairApi = Object.getOwnPropertyDescriptor(window, 'pairApi')
const originalWindowApi = Object.getOwnPropertyDescriptor(window, 'windowApi')

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false })
    vi.useFakeTimers()
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
    status.mockReset().mockResolvedValue(run)
    cancel.mockReset().mockImplementation(async () => {
        const cancelled: DiagnosticRun = { ...run, phase: 'done', outcome: 'cancelled' }
        status.mockResolvedValue(cancelled)
        return cancelled
    })
    focusNode.mockReset().mockResolvedValue(undefined)
    api.engines.diagnosticStatus = status
    api.engines.diagnosticCancel = cancel
    api.engines.onDiagnosticChanged = callback => {
        changed = callback
        return () => {
            changed = undefined
        }
    }
    api.engines.onSettingsChanged = () => () => {}
    api.engines.onSettingsDisconnected = () => () => {}
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    Object.defineProperty(window, 'windowApi', {
        configurable: true,
        value: { window: { focusNode } }
    })
    useConnectionStore.setState({ connected: true, selfId: node.id })
    useEngineSettingsStore.setState({ diagnostics: {}, modelDrafts: {}, entries: {} })
    useEngineStatusStore.setState({ statusByNode: new Map(), descriptors: new Map() })
    useMetricsStore.setState({ nodeMetrics: new Map() })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.useRealTimers()
    vi.unstubAllGlobals()
    if (originalPairApi) Object.defineProperty(window, 'pairApi', originalPairApi)
    else Reflect.deleteProperty(window, 'pairApi')
    if (originalWindowApi) Object.defineProperty(window, 'windowApi', originalWindowApi)
    else Reflect.deleteProperty(window, 'windowApi')
})

function cancelButton(label = 'Cancel'): HTMLButtonElement {
    const button = Array.from(container.querySelectorAll('button')).find(
        item => item.textContent === label
    )
    if (!button) throw new Error(`Missing diagnostic ${label} button`)
    return button
}

describe('target tray diagnostics', () => {
    it('discovers a peer-started run and cancels its exact run without opening settings', async () => {
        await act(async () => root.render(<TrayNodeRow node={node} activeWorkloads={[]} />))
        expect(status).toHaveBeenCalledExactlyOnceWith({ nodeId: 'self' })
        expect(container.textContent).toContain('Measuring memory')
        await act(async () => cancelButton().click())
        expect(cancel).toHaveBeenCalledExactlyOnceWith({
            nodeId: 'self',
            runId: 'peer-started-run'
        })
        expect(status).toHaveBeenLastCalledWith({ nodeId: 'self', runId: 'peer-started-run' })
        expect(focusNode).not.toHaveBeenCalled()
        expect(container.textContent).not.toContain('Measuring memory')
    })

    it('discovers on reconnect and follows target pushes without polling', async () => {
        useConnectionStore.setState({ connected: false })
        await act(async () => root.render(<TrayNodeRow node={node} activeWorkloads={[]} />))
        expect(status).not.toHaveBeenCalled()
        await act(async () => useConnectionStore.setState({ connected: true }))
        expect(status).toHaveBeenCalledExactlyOnceWith({ nodeId: 'self' })
        await act(async () => vi.advanceTimersByTime(6000))
        expect(status).toHaveBeenCalledTimes(1)
        await act(async () => changed?.({ ...run, outcome: 'cleanup-failed', phase: 'restoring' }))
        expect(container.textContent).toContain('Measuring memory: cleanup required')
        expect(container.textContent).toContain('Retry cleanup')
    })

    it('does not create remote watchers for peer rows', async () => {
        await act(async () =>
            root.render(<TrayNodeRow node={{ ...node, id: 'peer' }} activeWorkloads={[]} />)
        )
        await act(async () => vi.advanceTimersByTime(6000))
        expect(status).not.toHaveBeenCalled()
        expect(container.textContent).not.toContain('Measuring memory')
    })

    it('disables cancellation when the target observation is disconnected', async () => {
        await act(async () => root.render(<TrayNodeRow node={node} activeWorkloads={[]} />))
        await act(async () => useConnectionStore.setState({ connected: false }))
        expect(cancelButton().disabled).toBe(true)
        expect(container.textContent).toContain('reconnecting')
    })

    it('shares local cleanup-status retries with an open diagnostic panel', async () => {
        const held: DiagnosticRun = { ...run, outcome: 'cleanup-failed', phase: 'restoring' }
        useEngineSettingsStore.getState().receiveDiagnostic('self', held)
        status.mockRejectedValueOnce(new Error('Transient status failure')).mockResolvedValue(held)
        await act(async () =>
            root.render(
                <>
                    <TrayNodeRow node={node} activeWorkloads={[]} />
                    <ModelMemoryDiagnostic
                        nodeId="self"
                        engine="llamacpp"
                        modelId={run.modelId}
                        applied={false}
                    />
                </>
            )
        )
        expect(status).toHaveBeenCalledTimes(1)
        expect(cancelButton('Retry cleanup').disabled).toBe(true)
        await act(async () => vi.advanceTimersByTimeAsync(2000))
        expect(status).toHaveBeenCalledTimes(2)
        expect(status).toHaveBeenLastCalledWith({ nodeId: 'self', runId: run.runId })
        expect(cancelButton('Retry cleanup').disabled).toBe(false)
        expect(container.textContent).toContain('Measuring memory: cleanup required')
        expect(useEngineSettingsStore.getState().diagnostics.self.run).toEqual(held)
        await act(async () => cancelButton('Retry cleanup').click())
        expect(cancel).toHaveBeenCalledExactlyOnceWith({ nodeId: 'self', runId: run.runId })
    })

    it('keeps failed local retries disabled and retires an in-flight retry on unmount', async () => {
        const held: DiagnosticRun = { ...run, outcome: 'cleanup-failed', phase: 'restoring' }
        useEngineSettingsStore.getState().receiveDiagnostic('self', held)
        let resolveStatus: ((value: DiagnosticRun | null) => void) | undefined
        status
            .mockRejectedValueOnce(new Error('First status failure'))
            .mockRejectedValueOnce(new Error('Second status failure'))
            .mockImplementationOnce(
                () =>
                    new Promise(resolve => {
                        resolveStatus = resolve
                    })
            )
        await act(async () => root.render(<TrayNodeRow node={node} activeWorkloads={[]} />))
        await act(async () => vi.advanceTimersByTimeAsync(2000))
        expect(status).toHaveBeenCalledTimes(2)
        expect(cancelButton('Retry cleanup').disabled).toBe(true)
        expect(useEngineSettingsStore.getState().diagnostics.self.run).toEqual(held)
        await act(async () => vi.advanceTimersByTimeAsync(2000))
        expect(status).toHaveBeenCalledTimes(3)
        await act(async () => vi.advanceTimersByTimeAsync(10000))
        expect(status).toHaveBeenCalledTimes(3)
        await act(async () => root.render(null))
        if (!resolveStatus) throw new Error('Status retry was not sent')
        await act(async () => resolveStatus?.({ ...run, phase: 'done', outcome: 'cancelled' }))
        await act(async () => vi.advanceTimersByTimeAsync(10000))
        expect(status).toHaveBeenCalledTimes(3)
        expect(changed).toBeUndefined()
        expect(useEngineSettingsStore.getState().diagnostics.self).toMatchObject({
            current: false,
            reading: false,
            run: held
        })
    })

    it('pauses local retries while disconnected and retires the old response before recovery', async () => {
        const held: DiagnosticRun = { ...run, outcome: 'cleanup-failed', phase: 'restoring' }
        useEngineSettingsStore.getState().receiveDiagnostic('self', held)
        let resolveStatus: ((value: DiagnosticRun | null) => void) | undefined
        status.mockRejectedValueOnce(new Error('Transient status failure')).mockImplementationOnce(
            () =>
                new Promise(resolve => {
                    resolveStatus = resolve
                })
        )
        await act(async () => root.render(<TrayNodeRow node={node} activeWorkloads={[]} />))
        await act(async () => vi.advanceTimersByTimeAsync(2000))
        expect(status).toHaveBeenCalledTimes(2)
        await act(async () => useConnectionStore.setState({ connected: false }))
        if (!resolveStatus) throw new Error('Status retry was not sent')
        await act(async () => resolveStatus?.({ ...run, phase: 'done', outcome: 'cancelled' }))
        await act(async () => vi.advanceTimersByTimeAsync(10000))
        expect(status).toHaveBeenCalledTimes(2)
        expect(cancelButton('Retry cleanup').disabled).toBe(true)
        expect(useEngineSettingsStore.getState().diagnostics.self).toMatchObject({
            current: false,
            reading: false,
            run: held
        })
        status.mockResolvedValueOnce(held)
        await act(async () => useConnectionStore.setState({ connected: true }))
        expect(status).toHaveBeenLastCalledWith({ nodeId: 'self', runId: run.runId })
        expect(cancelButton('Retry cleanup').disabled).toBe(false)
        await act(async () => vi.advanceTimersByTimeAsync(6000))
        expect(status).toHaveBeenCalledTimes(3)
    })
})

describe('origin diagnostic dialog polling', () => {
    it('polls the peer every two seconds only while open', async () => {
        status.mockResolvedValue(null)
        const render = (open: boolean) =>
            act(async () =>
                root.render(
                    <ModelMemoryDiagnostic
                        nodeId="peer"
                        engine="llamacpp"
                        modelId="fixture.gguf"
                        applied={false}
                        open={open}
                    />
                )
            )
        await render(false)
        await act(async () => vi.advanceTimersByTime(4000))
        expect(status).not.toHaveBeenCalled()
        await render(true)
        expect(status).toHaveBeenCalledExactlyOnceWith({ nodeId: 'peer' })
        await act(async () => vi.advanceTimersByTime(1999))
        expect(status).toHaveBeenCalledTimes(1)
        await act(async () => vi.advanceTimersByTime(1))
        expect(status).toHaveBeenCalledTimes(2)
        await render(false)
        await act(async () => vi.advanceTimersByTime(6000))
        expect(status).toHaveBeenCalledTimes(2)
    })
})
