// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ServiceTransport } from '@/shared/types/service-bridge'
import type { IWindowApi } from '@/preload/api'
import { createPairApi } from '@/ui/api/pair-api'
import type { IEngineApi } from '@/ui/api/engine-api'
import { AddModelFileDialog } from '@/ui/components/Models/AddModelFileDialog'
import { useConnectionStore } from '@/ui/stores/connection.store'
import { useSourceModelsStore } from '@/ui/stores/source-models.store'

const chooseModelFile = vi.fn<IWindowApi['chooseModelFile']>()
const importModel = vi.fn<IEngineApi['importModel']>()
const linkModel = vi.fn<IEngineApi['linkModel']>()
const sourceModels = vi.fn<IEngineApi['sourceModels']>()
const onOpenChange = vi.fn<(open: boolean) => void>()
const onCreateProfile = vi.fn<(id: string) => void>()
const picked = 'C:\\models\\Mine-00001-of-00002.gguf'
let root: Root
let container: HTMLDivElement

function button(label: string): HTMLButtonElement {
    const found = Array.from(document.querySelectorAll('button')).find(
        row => row.textContent?.trim() === label
    )
    if (!found) throw new Error(`Missing button: ${label}`)
    return found
}

async function chooseMode(value: 'copy' | 'link'): Promise<void> {
    const radio = document.querySelector<HTMLElement>(
        `input[type="radio"][value="${value}"], [role="radio"][value="${value}"]`
    )
    if (!radio) throw new Error(`Missing radio: ${value}`)
    await act(async () => radio.click())
}

async function render(): Promise<void> {
    await act(async () =>
        root.render(
            <AddModelFileDialog onOpenChange={onOpenChange} onCreateProfile={onCreateProfile} />
        )
    )
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
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
    for (const mock of [chooseModelFile, importModel, linkModel, onOpenChange, onCreateProfile])
        mock.mockReset()
    sourceModels.mockReset().mockResolvedValue({ models: [] })
    chooseModelFile.mockResolvedValue(picked)
    api.engines.importModel = importModel
    api.engines.linkModel = linkModel
    api.engines.sourceModels = sourceModels
    Object.defineProperty(window, 'pairApi', { configurable: true, value: api })
    Object.defineProperty(window, 'windowApi', {
        configurable: true,
        value: { chooseModelFile }
    })
    useConnectionStore.setState({ selfId: 'msi', connected: true })
    useSourceModelsStore.setState({ entries: new Map() })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
})

describe('Add model from file', () => {
    it('stays at the file step when the picker is cancelled', async () => {
        chooseModelFile.mockResolvedValue(null)
        await render()
        await act(async () => button('Choose file…').click())
        expect(document.body.textContent).toContain('No file chosen')
        expect(document.body.textContent).not.toContain('How should PAIR use it?')
        expect(button('Add').disabled).toBe(true)
    })

    it('copies the file under its stem and offers a profile', async () => {
        importModel.mockResolvedValue({
            id: 'Mine',
            layout: 'llama-artifacts-v1',
            exportId: 'e',
            durability: 'confirmed'
        })
        await render()
        await act(async () => button('Choose file…').click())
        expect(document.body.textContent).toContain('Mine-00001-of-00002.gguf')
        expect(button('Add').disabled).toBe(true)
        await chooseMode('copy')
        const name = document.querySelector<HTMLInputElement>(
            'input[aria-label="Model name in PAIR"]'
        )
        expect(name?.value).toBe('Mine')
        await act(async () => button('Add').click())
        expect(importModel).toHaveBeenCalledWith('llama-cpp', { path: picked }, 'Mine')
        expect(document.body.textContent).toContain('Added as Mine')
        await act(async () => button('Create profile').click())
        expect(onCreateProfile).toHaveBeenCalledWith('Mine')
    })

    it('links the file in place and refreshes the outside-store list', async () => {
        linkModel.mockResolvedValue({ id: 'local/Mine' })
        await render()
        await act(async () => button('Choose file…').click())
        await chooseMode('link')
        expect(document.querySelector('input[aria-label="Model name in PAIR"]')).toBeNull()
        await act(async () => button('Add').click())
        expect(linkModel).toHaveBeenCalledWith('llama-cpp', picked)
        expect(document.body.textContent).toContain('Added as local/Mine')
        expect(sourceModels).toHaveBeenCalledWith('msi', 'llama-cpp')
    })

    it('keeps the dialog open with the reason when an import fails', async () => {
        importModel.mockRejectedValue(
            new Error('copy conflict: this alias already has a different committed export')
        )
        await render()
        await act(async () => button('Choose file…').click())
        await chooseMode('copy')
        await act(async () => button('Add').click())
        expect(document.body.textContent).toContain('copy conflict')
        expect(onOpenChange).not.toHaveBeenCalled()
    })
})
