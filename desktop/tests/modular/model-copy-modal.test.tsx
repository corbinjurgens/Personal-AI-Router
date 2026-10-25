// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { EngineInitialState } from '@/shared/types/engine-api'
import type { EngineModels, EngineProgress } from '@/shared/types/engines'
import type { ModelCopyCapabilities, ModelCopyResponse } from '@/shared/types/model-copy'
import type { NodeItem } from '@/shared/types/nodes'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { engineProgressKey } from '@/shared/utils/engine-progress'
import { createPairApi } from '@/ui/api/pair-api'
import type { IEngineApi } from '@/ui/api/engine-api'
import { ServiceCallError } from '@/ui/api/service-transport'
import { ModelCopyModal } from '@/ui/components/ModelManager/ModelCopyModal'
import { ModelManager } from '@/ui/components/ModelManager/ModelManager'
import ModelRow from '@/ui/components/ModelManager/ModelRow'
import { EngineCapabilities } from '@/ui/constants/engine-capabilities'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'

const alias = 'tiel-coder-35b-a3b-msi-ctx32768'
const exportId = 'e'.repeat(64)
const snapshot = vi.fn<IEngineApi['getInitialState']>()
const capabilities = vi.fn<IEngineApi['modelCopyCapabilities']>()
const start = vi.fn<IEngineApi['copyModel']>()
const cancel = vi.fn<IEngineApi['cancelModelCopy']>()
const close = vi.fn<(open: boolean) => void>()
let root: Root
let container: HTMLDivElement
let onProgress: Parameters<IEngineApi['onProgress']>[0] | undefined
let onRemove: Parameters<IEngineApi['onProgressRemove']>[0] | undefined
let onStateChanged: Parameters<IEngineApi['onStateChanged']>[0] | undefined

function deferred<T>() {
    let resolve: (value: T) => void = () => {
        throw new Error('Deferred not initialized')
    }
    const promise = new Promise<T>(done => {
        resolve = done
    })
    return { promise, resolve }
}
function node(id: string, name: string): NodeItem {
    return {
        id,
        name,
        status: 'active',
        os: 'Windows',
        ipAddress: '',
        port: 0,
        allIpAddresses: [],
        topology: { cpu: { model: '', cores: 0, threads: 0 }, gpus: [], ram: 0, storage: [] }
    }
}
function inventory(nodeId: string, names = [alias]): EngineModels {
    return {
        nodeId,
        engineType: 'llama-cpp',
        downloadStateCurrent: true,
        downloadState: {
            version: 1,
            sourceId: 'a'.repeat(32),
            revision: '1',
            recoveryRequired: false,
            entries: []
        },
        models: names.map(name => ({
            name,
            size: 19,
            downloaded: true,
            downloadStateCurrent: true,
            status: 'idle',
            parameterSize: '',
            quantization: '',
            family: '',
            digest: '',
            sizeVram: null,
            expiresAt: null,
            expiry: '10m',
            capabilities: [],
            pullKey: 'artifact/model.gguf'
        }))
    }
}
function initial(activeProgress: EngineProgress[] = []): EngineInitialState {
    return {
        descriptors: [],
        statuses: [],
        models: [inventory('msi-id'), inventory('joes-id', ['another-model'])],
        activeProgress,
        updateAvailable: []
    }
}
function capability(changes: Partial<ModelCopyCapabilities> = {}): ModelCopyCapabilities {
    return {
        version: 1,
        engine: 'llamacpp',
        available: true,
        installed: true,
        modelStore: 'llamacpp',
        exportLayouts: ['llama-artifacts-v1'],
        importLayouts: ['llama-artifacts-v1'],
        writable: true,
        reason: '',
        identityBoundRetry: true,
        receiptDurability: true,
        ...changes
    }
}
function progress(changes: Partial<EngineProgress> = {}): EngineProgress {
    return {
        opId: 'a'.repeat(32),
        sourceNodeId: 'msi-id',
        destinationNodeId: 'joes-id',
        nodeId: 'joes-id',
        nodeName: 'Joes',
        engineType: 'llama-cpp',
        model: alias,
        operation: 'copy',
        status: 'downloading',
        exportId,
        completed: 8,
        total: 19,
        ...changes
    }
}
function publish(value: EngineProgress): void {
    useEngineProgressStore.setState(state => ({
        progress: new Map(state.progress).set(engineProgressKey(value), value)
    }))
}
function button(label: string): HTMLButtonElement {
    const found = Array.from(document.querySelectorAll('button')).find(
        row => row.textContent?.trim() === label
    )
    if (!found) throw new Error(`Missing ${label} button`)
    return found
}
function field(index: number): HTMLElement {
    const found = container.querySelectorAll('[role="listbox"]')[index]
    if (!(found instanceof HTMLElement)) throw new Error(`Missing field ${index}`)
    return found
}
async function select(index: number, value: string) {
    const found = Array.from(field(index).querySelectorAll('button')).find(
        row => row.value === value
    )
    if (!found) throw new Error(`Missing option ${index}: ${value}`)
    await act(async () => found.click())
}
async function mount(opId?: string, open = true) {
    await act(async () =>
        root.render(
            <ModelCopyModal
                open={open}
                onOpenChange={close}
                engineType="llama-cpp"
                initialSource="msi-id"
                initialModel={alias}
                initialOpId={opId}
            />
        )
    )
}
async function chooseDestination() {
    await select(2, 'joes-id')
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false, escape: (value: string) => value })
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
    snapshot.mockReset().mockResolvedValue(initial())
    capabilities.mockReset().mockResolvedValue(capability())
    start.mockReset().mockReturnValue(new Promise<ModelCopyResponse>(() => {}))
    cancel.mockReset().mockResolvedValue({ cancelled: true })
    close.mockReset()
    api.engines.getInitialState = snapshot
    api.engines.modelCopyCapabilities = capabilities
    api.engines.copyModel = start
    api.engines.cancelModelCopy = cancel
    api.engines.onProgress = listener => {
        onProgress = listener
        return () => {
            onProgress = undefined
        }
    }
    api.engines.onProgressRemove = listener => {
        onRemove = listener
        return () => {
            onRemove = undefined
        }
    }
    api.engines.onStateChanged = listener => {
        onStateChanged = listener
        return () => {
            onStateChanged = undefined
        }
    }
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ selfId: 'mac-id', connected: true })
    useNodesStore.setState({
        nodes: new Map(
            [node('mac-id', 'Mac'), node('msi-id', 'MSI'), node('joes-id', 'Joes')].map(row => [
                row.id,
                row
            ])
        )
    })
    useEngineModelsStore.setState({ models: new Map() })
    useEngineProgressStore.setState({ progress: new Map() })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})
afterEach(async () => {
    await act(async () => root.unmount())
    useEngineProgressStore.getState().cleanup()
    useEngineModelsStore.getState().cleanup()
    useEngineProgressStore.setState({ progress: new Map() })
    container.remove()
    vi.unstubAllGlobals()
})

describe('copy selection', () => {
    it('requires an independent destination, leaving the controller unselected', async () => {
        await mount()
        expect(button('Copy model').disabled).toBe(true)
        expect(start).not.toHaveBeenCalled()
        expect(field(2).querySelector('button[value="msi-id"]')?.hasAttribute('disabled')).toBe(
            true
        )
        expect(container.querySelector('[aria-label="Destination device"]')?.textContent).toContain(
            'Choose a destination'
        )
        await chooseDestination()
        expect(button('Copy model').disabled).toBe(false)
        expect(container.textContent).toContain(
            `Copy ${alias} from MSI to Joes. The original stays on MSI.`
        )
        expect(container.textContent).toContain('Transfers directly between these devices.')
    })
    it('reloads the editable source inventory and clears the previous model', async () => {
        await mount()
        await select(0, 'joes-id')
        expect(snapshot).toHaveBeenCalledTimes(2)
        expect(field(1).textContent).toContain('another-model')
        expect(field(1).textContent).not.toContain(alias)
        expect(container.querySelector('[aria-label="Model"]')?.textContent).toContain(
            'Choose an installed model'
        )
        expect(button('Copy model').disabled).toBe(true)
        await select(1, 'another-model')
        await select(2, 'msi-id')
        await act(async () => button('Copy model').click())
        expect(start).toHaveBeenCalledWith(
            expect.objectContaining({
                sourceNodeId: 'joes-id',
                destinationNodeId: 'msi-id',
                model: 'another-model'
            })
        )
    })
    it('ignores inventory that resolves after the source selection has changed', async () => {
        const old = deferred<EngineInitialState>()
        snapshot.mockReturnValueOnce(old.promise)
        await mount()
        await select(0, 'joes-id')
        await act(async () =>
            old.resolve({ ...initial(), models: [inventory('msi-id', ['obsolete-model'])] })
        )
        expect(field(1).textContent).toContain('another-model')
        expect(field(1).textContent).not.toContain('obsolete-model')
    })
    it('sends exact node and model IDs, never artifact IDs or display labels, once', async () => {
        await mount()
        await chooseDestination()
        await act(async () => {
            button('Copy model').click()
            button('Copy model').click()
        })
        expect(start).toHaveBeenCalledTimes(1)
        expect(start.mock.calls[0][0]).toEqual({
            opId: expect.stringMatching(/^[a-f0-9]{32}$/),
            sourceNodeId: 'msi-id',
            destinationNodeId: 'joes-id',
            engine: 'llamacpp',
            model: alias
        })
        expect(capabilities).toHaveBeenCalledWith({ nodeId: 'msi-id', engine: 'llamacpp' })
        expect(capabilities).toHaveBeenCalledWith({ nodeId: 'joes-id', engine: 'llamacpp' })
        expect(capabilities.mock.calls.some(([request]) => request.nodeId === 'mac-id')).toBe(false)
    })
    it('offers Copy to device on a local ready model even without upstream pull support', async () => {
        const row = inventory('mac-id').models[0]
        await act(async () =>
            root.render(
                <ModelRow
                    model={row}
                    isRunning={false}
                    capabilities={{ ...EngineCapabilities['llama-cpp'], hasPull: false }}
                    canCopy
                    displayName={name => name}
                    onAction={() => {}}
                    onExpiryChange={() => {}}
                />
            )
        )
        const option = Array.from(container.querySelectorAll('[role="menuitem"]')).find(
            item => item.textContent === 'Copy to device…'
        )
        expect(option).toBeDefined()
        expect(option?.getAttribute('aria-disabled')).not.toBe('true')
    })
})

describe('capability and authority checks', () => {
    const refusals: [string, Partial<ModelCopyCapabilities>, string][] = [
        [
            'unknown authority',
            { writable: false, reason: 'Permission unknown' },
            'Permission unknown'
        ],
        [
            'permission revoked',
            { writable: false, reason: 'Remote management denied' },
            'Remote management denied'
        ],
        ['engine not installed', { installed: false }, 'Install this engine'],
        [
            'unsupported platform',
            { available: false, reason: 'Engine unsupported on this device' },
            'Engine unsupported'
        ],
        ['no layout intersection', { importLayouts: ['other-layout'] }, 'no compatible copy layout']
    ]
    it.each(refusals)('blocks %s with its actual explanation', async (_label, changes, message) => {
        capabilities.mockImplementation(async target =>
            capability(target.nodeId === 'joes-id' ? changes : {})
        )
        await mount()
        await chooseDestination()
        expect(container.textContent).toContain(message)
        expect(button('Copy model').disabled).toBe(true)
        expect(start).not.toHaveBeenCalled()
    })
    it('explains an old peer without assuming capability or installing anything', async () => {
        capabilities.mockRejectedValue(new ServiceCallError('method not found', -32601))
        await mount()
        await chooseDestination()
        expect(container.textContent).toContain('Unsupported: this device needs a PAIR version')
        expect(button('Copy model').disabled).toBe(true)
    })
    it('checks again at submission and refuses a revoked grant', async () => {
        await mount()
        await chooseDestination()
        capabilities.mockImplementation(async target =>
            capability(
                target.nodeId === 'joes-id' ? { writable: false, reason: 'Permission revoked' } : {}
            )
        )
        await act(async () => button('Copy model').click())
        expect(container.textContent).toContain('Permission revoked')
        expect(start).not.toHaveBeenCalled()
    })
    it('fences a late allowed capability for a different destination', async () => {
        const old = deferred<ModelCopyCapabilities>()
        capabilities.mockImplementation(target =>
            target.nodeId === 'joes-id' ? old.promise : Promise.resolve(capability())
        )
        await mount()
        await chooseDestination()
        capabilities.mockImplementation(async target =>
            capability(
                target.nodeId === 'mac-id'
                    ? { writable: false, reason: 'Current destination denied' }
                    : {}
            )
        )
        await select(2, 'mac-id')
        await act(async () => old.resolve(capability()))
        expect(container.textContent).toContain('Current destination denied')
        expect(button('Copy model').disabled).toBe(true)
    })
    it('blocks an offline destination', async () => {
        useNodesStore.setState(state => ({
            nodes: new Map(state.nodes).set('joes-id', {
                ...node('joes-id', 'Joes'),
                status: 'offline'
            })
        }))
        await mount()
        await chooseDestination()
        expect(container.textContent).toContain('Destination device is offline')
        expect(button('Copy model').disabled).toBe(true)
    })
    it('rechecks source readiness before dispatch', async () => {
        await mount()
        await chooseDestination()
        snapshot.mockResolvedValue({ ...initial(), models: [] })
        await act(async () => button('Copy model').click())
        expect(container.textContent).toContain('no longer ready in the source inventory')
        expect(start).not.toHaveBeenCalled()
    })
    it('refuses an older ready submission snapshot after a canonical source invalidation', async () => {
        await useEngineModelsStore.getState().initialize(initial())
        await mount()
        await chooseDestination()
        const old = deferred<EngineInitialState>()
        snapshot.mockReturnValueOnce(old.promise)
        await act(async () => button('Copy model').click())
        expect(snapshot).toHaveBeenCalledTimes(2)
        await act(async () =>
            onStateChanged?.({
                nodeId: 'msi-id',
                engineType: 'llama-cpp',
                models: { ...inventory('msi-id'), downloadStateCurrent: false }
            })
        )
        await act(async () => old.resolve(initial()))
        expect(
            useEngineModelsStore.getState().getModels('msi-id', 'llama-cpp').downloadStateCurrent
        ).toBe(false)
        expect(start).not.toHaveBeenCalled()
        expect(container.textContent).toContain('no longer ready in the source inventory')
    })
    it.each([
        ['newer revision', 'a'.repeat(32), '2'],
        ['new manager incarnation', 'b'.repeat(32), '1']
    ])('refuses a submission snapshot superseded by a %s', async (_label, sourceId, revision) => {
        await useEngineModelsStore.getState().initialize(initial())
        await mount()
        await chooseDestination()
        const old = deferred<EngineInitialState>()
        snapshot.mockReturnValueOnce(old.promise)
        await act(async () => button('Copy model').click())
        await act(async () =>
            onStateChanged?.({
                nodeId: 'msi-id',
                engineType: 'llama-cpp',
                models: {
                    ...inventory('msi-id'),
                    downloadState: {
                        version: 1,
                        sourceId,
                        revision,
                        recoveryRequired: false,
                        entries: []
                    }
                }
            })
        )
        await act(async () => old.resolve(initial()))
        expect(start).not.toHaveBeenCalled()
    })
    it.each([
        ['newer revision', 'c'.repeat(32), 'c'.repeat(32), '2'],
        ['new manager incarnation', 'd'.repeat(32), 'e'.repeat(32), '1'],
        ['same revision blocking readiness', 'f'.repeat(32), 'f'.repeat(32), '1']
    ])(
        'refuses live source lifecycle progress with %s while its inventory read is pending',
        async (_label, baselineSourceId, sourceId, revision) => {
            const current = inventory('msi-id')
            if (!current.downloadState) throw new Error('Missing source snapshot')
            current.downloadState.sourceId = baselineSourceId
            const ready = { ...initial(), models: [current] }
            snapshot.mockResolvedValue(ready)
            useEngineModelsStore.setState({ models: new Map([['msi-id:llama-cpp', current]]) })
            await useEngineProgressStore.getState().initialize(ready)
            await mount()
            await chooseDestination()
            const old = deferred<EngineInitialState>()
            snapshot.mockReturnValueOnce(old.promise)
            await act(async () => button('Copy model').click())
            const lifecycle: EngineProgress = {
                nodeId: 'msi-id',
                nodeName: 'MSI',
                engineType: 'llama-cpp',
                operation: 'pull',
                model: alias,
                status: 'downloading',
                downloadSourceId: sourceId,
                downloadRevision: revision,
                downloadStateCurrent: true,
                downloadState: {
                    id: 'download-op',
                    engine: 'llamacpp',
                    model: alias,
                    phase: 'in_progress',
                    blocksReadiness: true
                }
            }
            await act(async () => onProgress?.(lifecycle))
            expect(
                useEngineProgressStore.getState().progress.get(engineProgressKey(lifecycle))
            ).toEqual(lifecycle)
            await act(async () => old.resolve(ready))
            expect(useEngineModelsStore.getState().models.get('msi-id:llama-cpp')).toBe(current)
            expect(start).not.toHaveBeenCalled()
        }
    )
    const unrelatedLifecycle: [string, Partial<EngineProgress>][] = [
        ['another model', { model: 'another-model' }],
        ['another engine', { engineType: 'ollama' }],
        ['another device', { nodeId: 'joes-id' }],
        ['an older lifecycle revision', { downloadRevision: '1' }]
    ]
    it.each(unrelatedLifecycle)(
        'dispatches the exact selection when only %s changes during the inventory read',
        async (_label, changes) => {
            const current = inventory('msi-id')
            if (!current.downloadState) throw new Error('Missing source snapshot')
            current.downloadState.sourceId = '7'.repeat(32)
            current.downloadState.revision = '2'
            const ready = { ...initial(), models: [current] }
            snapshot.mockResolvedValue(ready)
            useEngineModelsStore.setState({ models: new Map([['msi-id:llama-cpp', current]]) })
            await useEngineProgressStore.getState().initialize(ready)
            await mount()
            await chooseDestination()
            const old = deferred<EngineInitialState>()
            snapshot.mockReturnValueOnce(old.promise)
            await act(async () => button('Copy model').click())
            await act(async () =>
                onProgress?.({
                    nodeId: 'msi-id',
                    nodeName: 'MSI',
                    engineType: 'llama-cpp',
                    operation: 'pull',
                    model: alias,
                    status: 'downloading',
                    downloadSourceId: current.downloadState?.sourceId,
                    downloadRevision: '3',
                    downloadStateCurrent: true,
                    downloadState: {
                        id: 'download-op',
                        engine: 'llamacpp',
                        model: alias,
                        phase: 'in_progress',
                        blocksReadiness: true
                    },
                    ...changes
                })
            )
            await act(async () => old.resolve(ready))
            expect(start).toHaveBeenCalledTimes(1)
            expect(start.mock.calls[0][0]).toEqual({
                opId: expect.stringMatching(/^[a-f0-9]{32}$/),
                sourceNodeId: 'msi-id',
                destinationNodeId: 'joes-id',
                engine: 'llamacpp',
                model: alias
            })
        }
    )
})

describe('copy operation lifecycle', () => {
    it('shows exact tuple progress and byte counts; closing does not cancel and reopening hydrates it', async () => {
        publish(progress())
        await mount('a'.repeat(32))
        expect(container.textContent).toContain(`${alias} · MSI → Joes · Copy · downloading`)
        expect(container.textContent).toContain('8 B / 19 B')
        expect(
            container.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')
        ).toBeNull()
        await act(async () => button('Close').click())
        expect(close).toHaveBeenCalledWith(false)
        expect(cancel).not.toHaveBeenCalled()
        await mount('a'.repeat(32), false)
        await mount('a'.repeat(32))
        expect(button('Cancel copy')).toBeDefined()
        close.mockClear()
        await act(async () => button('Cancel copy').click())
        expect(cancel).toHaveBeenCalledExactlyOnceWith('a'.repeat(32))
        expect(close).not.toHaveBeenCalled()
    })
    it('does not attribute another destination or wrong operation terminal to the selected copy', async () => {
        publish(progress({ percent: 42 }))
        await mount('a'.repeat(32))
        await act(async () => {
            publish(
                progress({
                    opId: 'b'.repeat(32),
                    destinationNodeId: 'mac-id',
                    nodeId: 'mac-id',
                    status: 'done',
                    message: 'Wrong success'
                })
            )
            publish(progress({ opId: 'c'.repeat(32), status: 'error', message: 'Wrong failure' }))
        })
        expect(container.textContent).not.toContain('Wrong success')
        expect(container.textContent).not.toContain('Wrong failure')
        expect(container.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe(
            '42'
        )
        expect(button('Cancel copy')).toBeDefined()
    })
    it('keeps copied custom artifacts setup-required and publication durability qualified after reopen', async () => {
        const result = {
            engine: 'llamacpp',
            model: alias,
            files: 1,
            filesSkipped: 0,
            bytesTotal: 19,
            bytesCopied: 19,
            layout: 'llama-artifacts-v1',
            exportId,
            durability: 'uncertain',
            requiresConfiguration: true
        } satisfies ModelCopyResponse['result']
        const message =
            "Copied to Joes. Set up this model's serving configuration on Joes before loading it. Files copied; publication durability could not be confirmed."
        publish(progress({ status: 'done', copyResult: result, message }))
        await mount('a'.repeat(32))
        expect(container.textContent).toContain(message)
        expect(container.querySelector('[role="progressbar"]')).toBeNull()
        expect(useEngineModelsStore.getState().models.size).toBe(0)
        await mount('a'.repeat(32), false)
        await mount('a'.repeat(32))
        expect(container.textContent).toContain('publication durability could not be confirmed')
        expect(button('Copy model').disabled).toBe(true)
    })
    it('retains identity-bound retry and source-offline receipt verification without relaxing destination authority', async () => {
        publish(progress({ status: 'unknown', message: 'Copy outcome unknown' }))
        useNodesStore.setState(state => ({
            nodes: new Map(state.nodes).set('msi-id', {
                ...node('msi-id', 'MSI'),
                status: 'offline'
            })
        }))
        capabilities.mockImplementation(async target => {
            if (target.nodeId === 'msi-id') throw new Error('Source offline')
            return capability()
        })
        await mount('a'.repeat(32))
        expect(container.textContent).toContain('same export identity')
        await act(async () => button('Retry copy').click())
        expect(start).toHaveBeenCalledWith(
            expect.objectContaining({
                expectedExportId: exportId,
                sourceNodeId: 'msi-id',
                destinationNodeId: 'joes-id',
                model: alias
            })
        )
        expect(
            capabilities.mock.calls.filter(([target]) => target.nodeId === 'joes-id')
        ).toHaveLength(2)
    })
    it('preserves constrained retry when the destination reports an alias conflict', async () => {
        publish(progress({ status: 'error' }))
        start.mockRejectedValue(
            new ServiceCallError('Alias now denotes a different export', -32010)
        )
        await mount('a'.repeat(32))
        await act(async () => button('Retry copy').click())
        expect(start.mock.calls[0][0].expectedExportId).toBe(exportId)
        expect(container.textContent).toContain('Existing content will not be overwritten')
        expect(container.textContent).toContain('Alias now denotes a different export')
        await act(async () => button('Retry copy').click())
        expect(start.mock.calls[1][0].expectedExportId).toBe(exportId)
    })
    it('labels an alias-only retry a fresh resolution and does not attach a guessed export ID', async () => {
        publish(progress({ status: 'cancelled', exportId: undefined }))
        await mount('a'.repeat(32))
        expect(container.textContent).toContain('Retry resolves this name again')
        await act(async () => button('Retry copy').click())
        expect(start.mock.calls[0][0].expectedExportId).toBeUndefined()
    })
    it('keeps independent operations visible on source and destination lists', async () => {
        publish(progress())
        publish(
            progress({
                opId: 'b'.repeat(32),
                nodeId: 'mac-id',
                destinationNodeId: 'mac-id',
                nodeName: 'Mac'
            })
        )
        await act(async () =>
            root.render(
                <ModelManager
                    nodeId="msi-id"
                    backend={{
                        type: 'llama-cpp',
                        displayName: 'llama.cpp',
                        source: 'installed',
                        processStatus: 'stopped',
                        port: 8081,
                        proxyPort: 54303,
                        models: []
                    }}
                />
            )
        )
        expect(container.textContent).toContain('MSI → Joes')
        expect(container.textContent).toContain('MSI → Mac')
        expect(
            Array.from(container.querySelectorAll('button')).filter(
                row => row.textContent === 'View copy'
            )
        ).toHaveLength(2)
    })
})

describe('renderer copy hydration', () => {
    it('marks a disappeared operation unknown and retains its export ID for retry', async () => {
        publish(progress())
        await useEngineProgressStore.getState().initialize(initial())
        expect(
            useEngineProgressStore.getState().progress.get(engineProgressKey(progress()))
        ).toMatchObject({ status: 'unknown', exportId })
        await mount('a'.repeat(32))
        expect(container.textContent).toContain('outcome unknown after reconnect')
        expect(button('Retry copy').disabled).toBe(false)
    })
    it('preserves a settled durability qualification across stale hydration and late progress', async () => {
        const done = progress({
            status: 'done',
            message: 'Files copied; publication durability could not be confirmed.'
        })
        publish(done)
        await useEngineProgressStore.getState().initialize(initial([progress()]))
        await act(async () => onProgress?.(progress()))
        expect(useEngineProgressStore.getState().progress.get(engineProgressKey(done))).toEqual(
            done
        )
        await useEngineProgressStore.getState().initialize(initial())
        expect(useEngineProgressStore.getState().progress.get(engineProgressKey(done))).toEqual(
            done
        )
    })
    it('hydrates active progress and removes it when the bridge evicts its bounded outcome', async () => {
        await useEngineProgressStore.getState().initialize(initial([progress({ percent: 17 })]))
        await mount('a'.repeat(32))
        expect(container.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe(
            '17'
        )
        await act(async () => onRemove?.(engineProgressKey(progress())))
        expect(useEngineProgressStore.getState().progress.size).toBe(0)
    })
})
