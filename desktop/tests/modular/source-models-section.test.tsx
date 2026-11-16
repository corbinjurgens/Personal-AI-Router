// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { NodeItem } from '@/shared/types/nodes'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import { createPairApi } from '@/ui/api/pair-api'
import type { IEngineApi } from '@/ui/api/engine-api'
import { SourceModelsSection } from '@/ui/components/Models/SourceModelsSection'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useSourceModelsStore } from '@/ui/stores/source-models.store'

vi.mock('@/ui/components/ModelSettings/ModelSettingsPanel', () => ({
    default: ({ identity, targetId }: { identity: { model: string }; targetId: string }) => (
        <div data-panel-model={identity.model} data-panel-target={targetId} />
    )
}))

const unlinkModel = vi.fn<IEngineApi['unlinkModel']>()
const sourceModels = vi.fn<IEngineApi['sourceModels']>()
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

const nodes = [node('msi', 'MSI'), node('joes', 'Joes')]

function button(label: string): HTMLButtonElement {
    const found = Array.from(document.querySelectorAll('button')).find(
        row => row.textContent?.trim() === label || row.getAttribute('aria-label') === label
    )
    if (!found) throw new Error(`Missing button: ${label}`)
    return found
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
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
    unlinkModel.mockReset().mockResolvedValue(null)
    sourceModels.mockReset().mockResolvedValue({ models: [] })
    api.engines.unlinkModel = unlinkModel
    api.engines.sourceModels = sourceModels
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    useConnectionStore.setState({ selfId: 'msi', connected: true })
    useSourceModelsStore.setState({
        entries: new Map([
            [
                'msi',
                {
                    loading: false,
                    error: '',
                    models: [
                        {
                            id: 'local/Mine',
                            source: 'local',
                            name: 'Mine.gguf',
                            sizeBytes: 2048,
                            shards: 1,
                            hasProjector: false
                        }
                    ]
                }
            ],
            [
                'joes',
                {
                    loading: false,
                    error: '',
                    models: [
                        {
                            id: 'ollama/tiny@latest',
                            source: 'ollama',
                            name: 'tiny@latest',
                            sizeBytes: 4096,
                            shards: 1,
                            hasProjector: false
                        },
                        {
                            id: 'local/Theirs',
                            source: 'local',
                            name: 'Theirs.gguf',
                            sizeBytes: 1,
                            shards: 2,
                            hasProjector: false
                        }
                    ]
                }
            ]
        ])
    })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    useSourceModelsStore.setState({ entries: new Map() })
})

describe('Outside the model store', () => {
    it('lists each node with its source rows and unlinks only local linked files', async () => {
        await act(async () => root.render(<SourceModelsSection nodes={nodes} needle="" />))
        expect(container.textContent).toContain('Outside the model store')
        expect(
            container.querySelector('section[aria-label="MSI outside the model store"]')
        ).not.toBeNull()
        expect(container.textContent).toContain('Ollama')
        expect(container.textContent).toContain('Linked file')
        expect(container.textContent).toContain('2 parts')
        expect(() => button('Unlink Mine.gguf')).not.toThrow()
        expect(() => button('Unlink Theirs.gguf')).toThrow()
    })

    it('opens profile settings for the source model on its node', async () => {
        await act(async () => root.render(<SourceModelsSection nodes={nodes} needle="" />))
        await act(async () => button('Create profile for tiny@latest').click())
        const panel = document.querySelector('[data-panel-model]')
        expect(panel?.getAttribute('data-panel-model')).toBe('ollama/tiny@latest')
        expect(panel?.getAttribute('data-panel-target')).toBe('joes')
    })

    it('unlinks after confirmation and shows a refusal', async () => {
        unlinkModel.mockRejectedValueOnce(
            new Error(
                'local/Mine is used by profile(s) chat; remove or change those profiles first.'
            )
        )
        await act(async () => root.render(<SourceModelsSection nodes={nodes} needle="" />))
        await act(async () => button('Unlink Mine.gguf').click())
        expect(document.body.textContent).toContain('The file itself is not deleted.')
        const confirm = Array.from(document.querySelectorAll('dialog button')).find(
            row => row.textContent?.trim() === 'Unlink'
        )
        if (!(confirm instanceof HTMLButtonElement)) throw new Error('Missing confirm button')
        await act(async () => confirm.click())
        expect(unlinkModel).toHaveBeenCalledWith('llamacpp', 'local/Mine')
        expect(document.body.textContent).toContain('used by profile(s) chat')
        expect(sourceModels).toHaveBeenCalledWith('msi', 'llamacpp')
    })

    it('filters rows by the Models tab filter', async () => {
        await act(async () => root.render(<SourceModelsSection nodes={nodes} needle="tiny" />))
        expect(container.textContent).toContain('tiny@latest')
        expect(container.textContent).not.toContain('Mine.gguf')
        expect(
            container.querySelector('section[aria-label="MSI outside the model store"]')
        ).toBeNull()
    })
})
