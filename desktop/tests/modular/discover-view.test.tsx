// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { EngineModels, ModelIdentity } from '@/shared/types/engines'
import type { NodeItem } from '@/shared/types/nodes'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'
import type { IEngineApi } from '@/ui/api/engine-api'
import { ModelsView } from '@/ui/components/Models/ModelsView'
import DiscoverView from '@/ui/components/Discover/DiscoverView'
import { useOverviewUiStore } from '@/ui/stores/overview-ui.store'
import { useOverviewNodes } from '@/ui/hooks/useOverviewNodes'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useNodesStore } from '@/ui/stores/nodes.store'
import { useEngineModelsStore } from '@/ui/stores/engine-models.store'
import { useEngineProgressStore } from '@/ui/stores/engine-progress.store'

vi.mock('@/ui/hooks/useOverviewNodes', () => ({
    useOverviewNodes: vi.fn<typeof useOverviewNodes>()
}))

const catalogFiles = vi.fn<IEngineApi['catalogFiles']>()
const pullModel = vi.fn<IEngineApi['pullModel']>()
const searchHub = vi.fn<IEngineApi['searchHub']>()
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

function inventory(
    nodeId: string,
    models: { name: string; identity: ModelIdentity }[]
): EngineModels {
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
        models: models.map(({ name, identity }) => ({
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
        }))
    }
}

const shared: ModelIdentity = {
    strength: 'declared',
    repository: 'owner/Tiel',
    quantization: 'Q4_K_M',
    totalBytes: 100
}
const joesOnly: ModelIdentity = {
    strength: 'declared',
    repository: 'owner/Other',
    quantization: 'Q8_0',
    totalBytes: 200
}

function button(label: string): HTMLButtonElement {
    const found = Array.from(document.querySelectorAll('button')).find(
        row => row.textContent?.trim() === label || row.getAttribute('aria-label') === label
    )
    if (!found) throw new Error(`Missing button: ${label}`)
    return found
}

async function openPage(): Promise<void> {
    await act(async () => root.render(<DiscoverView />))
}

function catalogEntry(name: string, downloads: number) {
    return {
        id: name,
        name,
        author: name.slice(0, name.indexOf('/')),
        url: `https://huggingface.co/${name.slice(0, name.lastIndexOf(':'))}`,
        downloads,
        likes: 0,
        updatedAt: '2026-11-13T00:00:00.000Z',
        tags: []
    }
}

async function pickModel(title: string): Promise<void> {
    const row = Array.from(
        document.querySelectorAll<HTMLButtonElement>('[aria-label="Models"] button')
    ).find(item => item.textContent?.includes(title))
    if (!row) throw new Error(`Missing catalog row: ${title}`)
    await act(async () => row.click())
}

async function typeSearch(text: string): Promise<void> {
    const input = document.querySelector<HTMLInputElement>('input[aria-label="Search models"]')
    if (!input) throw new Error('Missing search field')
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    await act(async () => {
        setValue?.call(input, text)
        input.dispatchEvent(new Event('input', { bubbles: true }))
    })
}

function sources(): HTMLInputElement[] {
    return Array.from(document.querySelectorAll<HTMLInputElement>('input[type="radio"]'))
}

async function openSource(label: string): Promise<void> {
    const source = sources().find(input => input.closest('label')?.textContent?.includes(label))
    if (!source) throw new Error(`Missing source: ${label}`)
    await act(async () => source.click())
}

async function findFiles(text: string): Promise<void> {
    const input = document.querySelector<HTMLInputElement>(
        'input[aria-label="Hugging Face repo or link"]'
    )
    if (!input) throw new Error('Missing repo field')
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    await act(async () => {
        setValue?.call(input, text)
        input.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => button('Find files').click())
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => false, escape: (value: string) => value })
    vi.stubGlobal(
        'ResizeObserver',
        class {
            observe = vi.fn<ResizeObserver['observe']>()
            unobserve = vi.fn<ResizeObserver['unobserve']>()
            disconnect = vi.fn<ResizeObserver['disconnect']>()
        }
    )
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
    catalogFiles.mockReset()
    pullModel.mockReset()
    searchHub.mockReset().mockResolvedValue({ models: [] })
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
    api.engines.catalogFiles = catalogFiles
    api.engines.pullModel = pullModel
    api.engines.searchHub = searchHub
    api.engines.modelCopyCapabilities = capabilities
    api.engines.getInitialState = vi.fn<IEngineApi['getInitialState']>().mockResolvedValue({
        descriptors: [],
        statuses: [],
        models: [],
        activeProgress: [],
        updateAvailable: []
    })
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    const nodes = [node('msi', 'MSI'), node('joes', 'Joes')]
    vi.mocked(useOverviewNodes).mockReturnValue(nodes)
    useNodesStore.setState({ nodes: new Map(nodes.map(row => [row.id, row])) })
    useConnectionStore.setState({ selfId: 'msi', connected: true })
    useEngineModelsStore.setState({
        models: new Map([
            ['msi:llama-cpp', inventory('msi', [{ name: 'tiel-msi', identity: shared }])],
            [
                'joes:llama-cpp',
                inventory('joes', [
                    { name: 'tiel-joes', identity: shared },
                    { name: 'other-joes', identity: joesOnly }
                ])
            ]
        ])
    })
    useEngineProgressStore.setState({ progress: new Map() })
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
})

describe('Discover page', () => {
    it('opens from the Models tab Download model button', async () => {
        useOverviewUiStore.setState({ activeTab: 'models' })
        await act(async () => root.render(<ModelsView />))
        await act(async () => button('Download model').click())
        expect(useOverviewUiStore.getState().activeTab).toBe('discover')
    })

    it('offers three sources', async () => {
        await openPage()
        const labels = ['Hugging Face', 'Paste a link', 'Your cluster']
        expect(sources()).toHaveLength(labels.length)
        labels.forEach((label, index) =>
            expect(sources()[index].closest('label')?.textContent).toContain(label)
        )
    })

    it('shows catalog facts and downloads the suggested quantization of the chosen model', async () => {
        searchHub.mockResolvedValue({
            models: [
                {
                    id: 'owner/Model-27B-GGUF:Q4_K_M',
                    name: 'owner/Model-27B-GGUF:Q4_K_M',
                    author: 'owner',
                    url: 'https://huggingface.co/owner/Model-27B-GGUF',
                    downloads: 6_800_000,
                    likes: 6100,
                    updatedAt: '2026-11-13T00:00:00.000Z',
                    tags: [
                        'gguf',
                        'license:apache-2.0',
                        'base_model:quantized:base/Model-27B',
                        'en'
                    ]
                }
            ]
        })
        catalogFiles.mockResolvedValue({
            repo: 'owner/Model-27B-GGUF',
            revision: 'a'.repeat(40),
            files: [
                {
                    quantization: 'UD-IQ1_S',
                    name: 'owner/Model-27B-GGUF:UD-IQ1_S',
                    sizeBytes: 100,
                    shards: 1
                },
                {
                    quantization: 'UD-Q4_K_M',
                    name: 'owner/Model-27B-GGUF:UD-Q4_K_M',
                    sizeBytes: 400,
                    shards: 1
                }
            ]
        })
        await openPage()
        await pickModel('Model 27B')
        expect(catalogFiles).toHaveBeenCalledWith('llama-cpp', 'owner/Model-27B-GGUF')
        const details = document.querySelector('[aria-label="owner/Model-27B-GGUF details"]')
        expect(details?.textContent).toContain('6.8M')
        expect(details?.textContent).toContain('Apache 2.0')
        expect(details?.textContent).toContain('base/Model-27B')
        expect(details?.textContent).toContain('27B')
        const first = details?.querySelector('[aria-label="Quantizations"] li')
        expect(first?.textContent).toContain('UD-Q4_K_M')
        expect(first?.textContent).toContain('Suggested')
        await act(async () => button('Download UD-Q4_K_M').click())
        expect(pullModel).toHaveBeenCalledWith('llama-cpp', 'msi', 'owner/Model-27B-GGUF:UD-Q4_K_M')
    })

    it('lists no repository files until a model is picked', async () => {
        searchHub.mockResolvedValue({
            models: [
                catalogEntry('owner/Alpha-GGUF:Q4_K_M', 20),
                catalogEntry('owner/Beta-GGUF:Q4_K_M', 10)
            ]
        })
        catalogFiles.mockResolvedValue({ repo: 'owner/Beta-GGUF', revision: 'b', files: [] })
        await openPage()
        expect(catalogFiles).not.toHaveBeenCalled()
        expect(document.body.textContent).toContain('Pick a model')
        await pickModel('Beta')
        expect(catalogFiles).toHaveBeenCalledExactlyOnceWith('llama-cpp', 'owner/Beta-GGUF')
    })

    it('keeps the picked model and lists its files once while the filter changes', async () => {
        searchHub.mockResolvedValue({
            models: [
                catalogEntry('owner/Alpha-GGUF:Q4_K_M', 20),
                catalogEntry('owner/Beta-GGUF:Q4_K_M', 10)
            ]
        })
        catalogFiles.mockResolvedValue({ repo: 'owner/Beta-GGUF', revision: 'b', files: [] })
        await openPage()
        await pickModel('Beta')
        await typeSearch('alp')
        await typeSearch('alpha')
        expect(catalogFiles).toHaveBeenCalledExactlyOnceWith('llama-cpp', 'owner/Beta-GGUF')
        expect(document.querySelector('[aria-label="owner/Beta-GGUF details"]')).not.toBeNull()
    })

    it('ignores a download by another engine of the same name on the target', async () => {
        catalogFiles.mockResolvedValue({
            repo: 'owner/Model-GGUF',
            revision: 'a'.repeat(40),
            files: [
                {
                    quantization: 'Q4_K_M',
                    name: 'owner/Model-GGUF:Q4_K_M',
                    sizeBytes: 400,
                    shards: 1
                }
            ]
        })
        const msi = inventory('msi', [])
        useEngineModelsStore.setState({
            models: new Map([
                [
                    'msi:llama-cpp',
                    {
                        ...msi,
                        downloadState: msi.downloadState && {
                            ...msi.downloadState,
                            entries: [
                                {
                                    id: 'other-engine',
                                    engine: 'llamacpp-fork',
                                    model: 'owner/Model-GGUF:Q4_K_M',
                                    phase: 'in_progress',
                                    blocksReadiness: true,
                                    percent: 50
                                }
                            ]
                        }
                    }
                ]
            ])
        })
        await openPage()
        await openSource('Paste a link')
        await findFiles('owner/Model-GGUF')
        const download = button('Download Q4_K_M')
        expect(download.disabled).toBe(false)
        expect(download.textContent).toContain('Download')
        expect(download.textContent).not.toContain('Downloading')
    })

    it('dims an offline holder in the cluster list', async () => {
        vi.mocked(useOverviewNodes).mockReturnValue([
            node('msi', 'MSI'),
            { ...node('joes', 'Joes'), status: 'offline' }
        ])
        await openPage()
        await openSource('Your cluster')
        const chips = Array.from(document.querySelectorAll('.pair-node-chip')).map(chip => [
            chip.textContent,
            chip.getAttribute('data-offline')
        ])
        expect(chips.length).toBeGreaterThan(0)
        expect(chips).toEqual(chips.map(() => ['Joes', 'true']))
    })

    it('downloads a chosen quantization of a pasted repository to the target node', async () => {
        catalogFiles.mockResolvedValue({
            repo: 'owner/Model-GGUF',
            revision: 'a'.repeat(40),
            files: [
                {
                    quantization: 'Q4_K_M',
                    name: 'owner/Model-GGUF:Q4_K_M',
                    sizeBytes: 400,
                    shards: 1
                },
                { quantization: 'Q8_0', name: 'owner/Model-GGUF:Q8_0', sizeBytes: 800, shards: 2 }
            ]
        })
        await openPage()
        await openSource('Paste a link')
        await findFiles('https://huggingface.co/owner/Model-GGUF')
        expect(catalogFiles).toHaveBeenCalledWith(
            'llama-cpp',
            'https://huggingface.co/owner/Model-GGUF'
        )
        expect(document.body.textContent).toContain('owner/Model-GGUF')
        expect(document.body.textContent).toContain('2 parts')
        await act(async () => button('Download Q4_K_M').click())
        expect(pullModel).toHaveBeenCalledWith('llama-cpp', 'msi', 'owner/Model-GGUF:Q4_K_M')
    })

    it('shows why a repository cannot be listed', async () => {
        catalogFiles.mockRejectedValue(
            new Error('This repository is private or gated; PAIR downloads only open repositories.')
        )
        await openPage()
        await openSource('Paste a link')
        await findFiles('owner/Gated')
        expect(document.body.textContent).toContain('private or gated')
        expect(pullModel).not.toHaveBeenCalled()
    })

    it('offers a model held only by another node and opens its copy to the target', async () => {
        await openPage()
        await openSource('Your cluster')
        expect(document.body.textContent).toContain('Already on MSI')
        await act(async () => button('Copy owner/Other Q8_0 here').click())
        expect(document.body.textContent).toContain('Copy model')
        expect(capabilities).toHaveBeenCalledWith(
            expect.objectContaining({ nodeId: 'joes', engine: 'llamacpp' })
        )
    })
})
