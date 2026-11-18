// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { EngineModels, EngineProgress, ModelIdentity } from '@/shared/types/engines'
import type { ModelDownloadEntry } from '@/shared/types/model-download'
import type { EngineSettingsSnapshot } from '@/shared/types/engine-settings'
import type { ModelCopyResponse } from '@/shared/types/model-copy'
import type { NodeItem } from '@/shared/types/nodes'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'
import type { IEngineApi } from '@/ui/api/engine-api'
import { engineProgressKey } from '@/shared/utils/engine-progress'
import { engineTypeFromManagerName } from '@/shared/utils/engines'
import { ModelsView } from '@/ui/components/Models/ModelsView'
import { ModelManager } from '@/ui/components/ModelManager/ModelManager'
import { useOverviewNodes } from '@/ui/hooks/useOverviewNodes'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'
import { usePendingActionsStore } from '@/ui/stores/pending-actions.store'
import { useNodeConfigurationStore } from '@/ui/stores/node-configuration.store'
import { useEngineSettingsStore } from '@/ui/stores/engine-settings.store'

vi.mock('@/ui/hooks/useOverviewNodes', () => ({
    useOverviewNodes: vi.fn<typeof useOverviewNodes>()
}))

const identity: ModelIdentity = {
    strength: 'declared',
    repository: 'owner/Tiel',
    quantization: 'Q4_K_M',
    totalBytes: 100
}
const deleteModel = vi.fn<IEngineApi['deleteModel']>()
const copyModel = vi.fn<IEngineApi['copyModel']>()
const getInitialState = vi.fn<IEngineApi['getInitialState']>()
const capabilities = vi.fn<IEngineApi['modelCopyCapabilities']>()
let root: Root
let container: HTMLDivElement

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

function inventory(nodeId: string, name: string): EngineModels {
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
        models: [
            {
                name,
                identity,
                size: 100,
                downloaded: true,
                downloadStateCurrent: true,
                status: 'idle',
                parameterSize: '',
                family: '',
                sizeVram: null,
                expiresAt: null,
                expiry: '0',
                capabilities: []
            }
        ]
    }
}

function settings(): EngineSettingsSnapshot {
    return {
        nodeId: 'msi',
        engine: 'llamacpp',
        revision: 1,
        appliedRevision: 1,
        epoch: 'epoch',
        sequence: 1,
        settings: {
            serverPort: 1,
            proxyPort: 2,
            launchText: '',
            modelLoads: [{ profileKey: 'lfm-64k', modelId: 'tiel-msi', requested: {} }]
        },
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
}

function button(label: string): HTMLButtonElement {
    const found = Array.from(document.querySelectorAll('button')).find(
        row => row.textContent?.trim() === label
    )
    if (!found) throw new Error(`Missing button: ${label}`)
    return found
}

async function select(index: number, value: string): Promise<void> {
    const field = document.querySelectorAll('[role="listbox"]')[index]
    if (!field) throw new Error(`Missing copy field: ${index}`)
    const option = Array.from(field.querySelectorAll('button')).find(row => row.value === value)
    if (!option) throw new Error(`Missing copy option: ${value}`)
    expect(option.disabled).toBe(false)
    await act(async () => option.click())
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
        invoke: vi.fn<ServiceTransport['invoke']>(() => {
            throw new Error('Unexpected service invocation')
        }),
        subscribePush: () => () => {},
        onConnect: () => () => {},
        onDisconnect: () => () => {},
        onAuthFailure: () => () => {},
        destroy: () => {}
    }
    const api = createPairApi(transport)
    deleteModel.mockReset()
    copyModel.mockReset().mockReturnValue(new Promise<ModelCopyResponse>(() => {}))
    getInitialState.mockReset().mockResolvedValue({
        descriptors: [],
        statuses: [],
        models: [inventory('msi', 'tiel-msi'), inventory('joes', 'tiel-joes')],
        activeProgress: [],
        updateAvailable: []
    })
    capabilities.mockReset().mockResolvedValue({
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
        receiptDurability: true
    })
    api.engines.deleteModel = deleteModel
    api.engines.copyModel = copyModel
    api.engines.getInitialState = getInitialState
    api.engines.modelCopyCapabilities = capabilities
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    const nodes = [node('msi', 'MSI'), node('joes', 'Joes')]
    vi.mocked(useOverviewNodes).mockReturnValue(nodes)
    useNodesStore.setState({ nodes: new Map(nodes.map(row => [row.id, row])) })
    useConnectionStore.setState({ selfId: 'msi', connected: true })
    useEngineModelsStore.setState({
        models: new Map([
            ['msi:llama-cpp', inventory('msi', 'tiel-msi')],
            ['joes:llama-cpp', inventory('joes', 'tiel-joes')]
        ])
    })
    useEngineProgressStore.setState({ progress: new Map() })
    usePendingActionsStore.setState({ pending: new Map() })
    useNodeConfigurationStore.setState({
        editors: new Map([
            [
                'msi',
                {
                    phase: 'draft',
                    draft: [],
                    baseline: {
                        policy: JSON.stringify({
                            profiles: [
                                {
                                    key: 'lfm-64k',
                                    displayName: 'LFM 64K',
                                    engine: 'llamacpp',
                                    model: 'tiel-msi'
                                }
                            ],
                            roles: [{ name: 'coder', profiles: ['lfm-64k'] }]
                        }),
                        revision: '1',
                        roles: [{ name: 'coder', profiles: ['lfm-64k'] }]
                    }
                }
            ]
        ])
    })
    useEngineSettingsStore.setState({
        entries: {
            'msi:llamacpp': { snapshot: settings(), arrival: 1 }
        }
    })
    const saved = useNodeConfigurationStore.getState().editors.get('msi')?.baseline
    if (!saved) throw new Error('Missing saved policy fixture')
    api.nodePolicy.get = vi.fn<typeof api.nodePolicy.get>().mockResolvedValue({
        policy: saved.policy,
        revision: saved.revision,
        availability: 'available',
        activation: null,
        recoveryRequired: false
    })
    vi.spyOn(useEngineSettingsStore.getState(), 'reload').mockResolvedValue(undefined)
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    useEngineModelsStore.setState({ models: new Map() })
    useEngineProgressStore.setState({ progress: new Map() })
    usePendingActionsStore.setState({ pending: new Map() })
    useNodeConfigurationStore.setState({ editors: new Map() })
    useEngineSettingsStore.setState({ entries: {} })
})

describe('Models tab', () => {
    it('gives a served profile key precedence over another load artifact id in delete usage', async () => {
        const snapshot = settings()
        snapshot.settings.modelLoads?.push({
            profileKey: 'tiel-msi',
            modelId: 'other-artifact',
            requested: {}
        })
        useEngineSettingsStore.setState({ entries: { 'msi:llamacpp': { snapshot, arrival: 1 } } })
        window.pairApi.nodePolicy.get = vi
            .fn<typeof window.pairApi.nodePolicy.get>()
            .mockResolvedValue({
                revision: '1',
                activation: null,
                availability: 'available',
                recoveryRequired: false,
                policy: JSON.stringify({
                    profiles: [
                        {
                            key: 'lfm-64k',
                            displayName: 'First',
                            engine: 'llamacpp',
                            model: 'lfm-64k'
                        },
                        {
                            key: 'tiel-msi',
                            displayName: 'Second',
                            engine: 'llamacpp',
                            model: 'tiel-msi'
                        }
                    ],
                    roles: [
                        { name: 'coder', profiles: ['lfm-64k'] },
                        { name: 'second-role', profiles: ['tiel-msi'] }
                    ]
                })
            })
        await act(async () => root.render(<ModelsView />))
        await act(async () => button('Delete on node').click())
        const message = document.querySelector('dialog')?.textContent ?? ''
        expect(message).toContain(
            'Profiles that use it stop serving: tiel-msi. Roles affected: second-role.'
        )
        expect(message).not.toContain('lfm-64k')
        expect(message).not.toContain('coder')
        expect(deleteModel).not.toHaveBeenCalled()
    })
    it('shows declared grouped models with both node locations and an unverified note', async () => {
        await act(async () => root.render(<ModelsView />))
        const group = container.querySelector('section[aria-label="owner/Tiel"]')
        expect(group?.textContent).toContain('Tiel')
        const rows = Array.from(group?.querySelectorAll('li') ?? []).map(row => row.textContent)
        expect(rows).toHaveLength(2)
        expect(rows[0]).toContain('MSI')
        expect(rows[0]).toContain('tiel-msi')
        expect(rows[1]).toContain('Joes')
        expect(rows[1]).toContain('tiel-joes')
        expect(group?.textContent).toContain('Same file not verified')
    })

    it('filters model groups by node name and explains an empty match', async () => {
        await act(async () => root.render(<ModelsView />))
        const filter = container.querySelector<HTMLInputElement>(
            'input[aria-label="Filter models"]'
        )
        if (!filter) throw new Error('Missing model filter')
        const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
        await act(async () => {
            setValue?.call(filter, 'joes')
            filter.dispatchEvent(new Event('input', { bubbles: true }))
        })
        expect(container.querySelector('section[aria-label="owner/Tiel"]')).not.toBeNull()
        await act(async () => {
            setValue?.call(filter, 'no-such-model')
            filter.dispatchEvent(new Event('input', { bubbles: true }))
        })
        expect(container.querySelector('section[aria-label="owner/Tiel"]')).toBeNull()
        expect(container.textContent).toContain('No models match “no-such-model”.')
    })

    it('opens the existing copy modal at the source and copies to an unverified group member', async () => {
        await act(async () => root.render(<ModelsView />))
        expect(button('Copy to node').disabled).toBe(false)
        await act(async () => button('Copy to node').click())
        expect(document.body.textContent).toContain('Copy model')
        await select(2, 'joes')
        expect(document.body.textContent).toContain('Copy tiel-msi from MSI to Joes')
        expect(button('Copy model').disabled).toBe(false)
        await act(async () => button('Copy model').click())
        expect(copyModel).toHaveBeenCalledExactlyOnceWith(
            expect.objectContaining({
                sourceNodeId: 'msi',
                destinationNodeId: 'joes',
                engine: 'llamacpp',
                model: 'tiel-msi'
            })
        )
    })

    it('lists the affected profile and role and deletes only after confirmation', async () => {
        useNodeConfigurationStore.setState({ editors: new Map() })
        await act(async () => root.render(<ModelsView />))
        await act(async () => button('Delete on node').click())
        expect(document.body.textContent).toContain('Delete tiel-msi on MSI?')
        expect(document.body.textContent).toContain('Profiles that use it stop serving: lfm-64k')
        expect(document.body.textContent).toContain('Roles affected: coder')
        expect(deleteModel).not.toHaveBeenCalled()
        await act(async () => button('Delete model').click())
        expect(deleteModel).toHaveBeenCalledExactlyOnceWith('llama-cpp', 'msi', 'tiel-msi')
    })

    it('lists profile and role usage when the selected inventory row is the served profile key', async () => {
        useEngineModelsStore.setState({
            models: new Map([['msi:llama-cpp', inventory('msi', 'lfm-64k')]])
        })
        await act(async () => root.render(<ModelsView />))
        await act(async () => button('Delete on node').click())
        expect(document.body.textContent).toContain('Profiles that use it stop serving: lfm-64k')
        expect(document.body.textContent).toContain('Roles affected: coder')
        expect(deleteModel).not.toHaveBeenCalled()
        await act(async () => button('Delete model').click())
        expect(deleteModel).toHaveBeenCalledExactlyOnceWith('llama-cpp', 'msi', 'lfm-64k')
    })

    it('reads affected profiles afresh instead of using a cached policy without the new profile', async () => {
        useNodeConfigurationStore.setState({
            editors: new Map([
                [
                    'msi',
                    {
                        phase: 'draft',
                        draft: [],
                        baseline: { policy: '{}', revision: '0', roles: [] }
                    }
                ]
            ])
        })
        await act(async () => root.render(<ModelsView />))
        await act(async () => button('Delete on node').click())
        expect(window.pairApi.nodePolicy.get).toHaveBeenCalledOnce()
        expect(document.body.textContent).toContain('Profiles that use it stop serving: lfm-64k')
        expect(document.body.textContent).toContain('Roles affected: coder')
        expect(deleteModel).not.toHaveBeenCalled()
    })

    it('cancels deletion without calling the service', async () => {
        await act(async () => root.render(<ModelsView />))
        await act(async () => button('Delete on node').click())
        await act(async () => button('Cancel').click())
        expect(deleteModel).not.toHaveBeenCalled()
    })

    it('lists a running download once although every engine list carries the node snapshot', async () => {
        const fork = engineTypeFromManagerName('llamacpp-fork')
        if (!fork) throw new Error('Missing descriptor engine type')
        const entry: ModelDownloadEntry = {
            id: 'download-1',
            engine: 'llamacpp',
            model: 'owner/New-GGUF:Q4_K_M',
            phase: 'in_progress',
            blocksReadiness: true,
            percent: 40
        }
        const withSnapshot = (list: EngineModels): EngineModels => ({
            ...list,
            downloadState: list.downloadState && { ...list.downloadState, entries: [entry] }
        })
        const pull = (nodeId: string): EngineProgress => ({
            nodeId,
            nodeName: nodeId,
            engineType: 'llama-cpp',
            operation: 'pull',
            model: entry.model,
            status: entry.phase,
            percent: entry.percent,
            downloadState: entry,
            downloadStateCurrent: true
        })
        useEngineModelsStore.setState({
            models: new Map([
                ['msi:llama-cpp', withSnapshot(inventory('msi', 'tiel-msi'))],
                [
                    `msi:${fork}`,
                    withSnapshot({ ...inventory('msi', 'fork-model'), engineType: fork })
                ]
            ])
        })
        useEngineProgressStore.setState({
            progress: new Map(
                [pull('msi'), pull('departed')].map(item => [engineProgressKey(item), item])
            )
        })
        await act(async () => root.render(<ModelsView />))
        const panel = container.querySelector('section[aria-label="Transfers"]')
        const text = panel?.textContent ?? ''
        expect(text).toContain('1 running')
        expect(text.match(/To MSI/g)).toHaveLength(1)
        expect(text).not.toContain('departed')
    })

    it('shows a queued download with a moving bar on the Models page', async () => {
        const entry: ModelDownloadEntry = {
            id: 'download-2',
            engine: 'llamacpp',
            model: 'owner/Queued-GGUF:Q4_K_M',
            phase: 'queued',
            blocksReadiness: true
        }
        const item: EngineProgress = {
            nodeId: 'joes',
            nodeName: 'Joes',
            engineType: 'llama-cpp',
            operation: 'pull',
            model: entry.model,
            status: entry.phase,
            downloadState: entry,
            downloadStateCurrent: true
        }
        useEngineProgressStore.setState({ progress: new Map([[engineProgressKey(item), item]]) })
        await act(async () => root.render(<ModelsView />))
        const panel = container.querySelector('section[aria-label="Transfers"]')
        const bar = panel?.querySelector(
            `[role="progressbar"][aria-label="Download progress for ${entry.model}"]`
        )
        expect(bar).not.toBeNull()
        expect(bar?.getAttribute('aria-valuenow')).toBeNull()
        expect(panel?.textContent).toContain('Download queued')
        expect(panel?.textContent).toContain('To Joes')
    })

    it('dims the chips of an offline holder in the group header and its row', async () => {
        vi.mocked(useOverviewNodes).mockReturnValue([
            node('msi', 'MSI'),
            { ...node('joes', 'Joes'), status: 'offline' }
        ])
        await act(async () => root.render(<ModelsView />))
        const group = container.querySelector('section[aria-label="owner/Tiel"]')
        const chips = Array.from(group?.querySelectorAll('.pair-node-chip') ?? []).map(chip => [
            chip.textContent,
            chip.getAttribute('data-offline')
        ])
        expect(chips).toEqual([
            ['MSI', 'false'],
            ['Joes', 'true'],
            [expect.stringContaining('MSI'), 'false'],
            ['Joes', 'true']
        ])
    })

    it('shows a chip for each of two nodes that share a name', async () => {
        const errors = vi.spyOn(console, 'error').mockImplementation(() => {})
        vi.mocked(useOverviewNodes).mockReturnValue([node('msi', 'PC'), node('joes', 'PC')])
        await act(async () => root.render(<ModelsView />))
        const group = container.querySelector('section[aria-label="owner/Tiel"]')
        expect(group?.querySelectorAll('.pair-node-chip')).toHaveLength(4)
        const keyWarnings = errors.mock.calls.filter(call =>
            call.some(arg => String(arg).includes('same key'))
        )
        expect(keyWarnings).toEqual([])
    })

    it.each(['tiel-msi', 'lfm-64k'])(
        'confirms per-engine deletion for inventory name %s',
        async modelName => {
            const observed = inventory('msi', modelName)
            useEngineModelsStore.setState({ models: new Map([['msi:llama-cpp', observed]]) })
            await act(async () =>
                root.render(
                    <ModelManager
                        nodeId="msi"
                        backend={{
                            type: 'llama-cpp',
                            displayName: 'llama.cpp',
                            source: 'installed',
                            processStatus: 'running',
                            port: 1,
                            proxyPort: 2,
                            models: observed.models
                        }}
                    />
                )
            )
            const trigger = container.querySelector('button[aria-haspopup="menu"]')
            if (!(trigger instanceof HTMLButtonElement)) throw new Error('Missing model actions')
            const menu = document.getElementById(trigger.getAttribute('aria-controls') ?? '')
            if (!menu) throw new Error('Missing model menu')
            await act(async () => {
                trigger.click()
                const toggle = new Event('toggle')
                Object.defineProperty(toggle, 'newState', { value: 'open' })
                menu.dispatchEvent(toggle)
            })
            const remove = Array.from(menu.querySelectorAll('[role="menuitem"]')).find(
                row => row.textContent === 'Delete'
            )
            if (!(remove instanceof HTMLElement)) throw new Error('Missing Delete action')
            await act(async () => remove.click())
            expect(document.body.textContent).toContain(
                'Profiles that use it stop serving: lfm-64k'
            )
            expect(document.body.textContent).toContain('Roles affected: coder')
            expect(deleteModel).not.toHaveBeenCalled()
            await act(async () => button('Delete model').click())
            expect(deleteModel).toHaveBeenCalledExactlyOnceWith('llama-cpp', 'msi', modelName)
        }
    )
})
