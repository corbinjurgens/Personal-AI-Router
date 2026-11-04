// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import HaltControl from '@/ui/components/MainApp/HaltControl'
import * as configuration from '@/ui/stores/node-configuration.store'
import { useConnectionStore } from '@/ui/stores/connection.store'

const request =
    vi.fn<
        ReturnType<typeof configuration.useNodeConfigurationStore.getState>['requestAvailability']
    >()
let root: Root
let container: HTMLDivElement
let originalRequest: ReturnType<
    typeof configuration.useNodeConfigurationStore.getState
>['requestAvailability']

function button(label: string) {
    const result = Array.from(document.querySelectorAll('button')).find(
        item => item.textContent?.trim() === label
    )
    if (!result) throw new Error(`Missing button ${label}`)
    return result
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    vi.stubGlobal('CSS', { supports: () => true })
    vi.stubGlobal(
        'ResizeObserver',
        class {
            observe() {
                return undefined
            }
            disconnect() {
                return undefined
            }
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
    vi.spyOn(configuration, 'watchNodeAvailability').mockReturnValue(() => {})
    request.mockReset().mockResolvedValue(undefined)
    const store = configuration.useNodeConfigurationStore
    originalRequest = store.getState().requestAvailability
    store.setState({
        requestAvailability: request,
        availability: new Map([
            ['self', { value: 'available', current: true, stale: false, reading: false }]
        ])
    })
    useConnectionStore.setState({ selfId: 'self', connected: true })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    configuration.useNodeConfigurationStore.setState({
        requestAvailability: originalRequest,
        availability: new Map()
    })
    container.remove()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
})

describe('toolbar Halt control', () => {
    it('cancels confirmation without sending a request', async () => {
        await act(async () => root.render(<HaltControl targetId="self" />))
        await act(async () => button('Halt').click())
        await act(async () => button('Cancel').click())
        expect(request).not.toHaveBeenCalled()
    })

    it('shows peer pause provenance and service errors inline', async () => {
        configuration.useNodeConfigurationStore.setState({
            availability: new Map([
                [
                    'self',
                    {
                        value: 'paused',
                        pausedBy: 'peer',
                        current: true,
                        stale: false,
                        reading: false,
                        error: 'Pause could not stop every engine'
                    }
                ]
            ])
        })
        await act(async () => root.render(<HaltControl targetId="self" />))
        expect(container.textContent).toContain('Paused by another node')
        expect(container.querySelector('[role="alert"]')?.textContent).toBe(
            'Pause could not stop every engine'
        )
        expect(button('Resume').disabled).toBe(false)
    })
    it('requires confirmation before sending the local immediate pause', async () => {
        await act(async () => root.render(<HaltControl targetId="self" />))
        await act(async () => button('Halt').click())
        expect(request).not.toHaveBeenCalled()
        expect(document.body.textContent).toContain('Halt this PC?')
        expect(document.body.textContent).toContain(
            'Running requests stop now, models unload, and other nodes cannot resume it.'
        )
        const dialog = document.querySelector('dialog')
        const confirm = Array.from(dialog?.querySelectorAll('button') ?? []).find(
            item => item.textContent === 'Halt'
        )
        if (!confirm) throw new Error('Missing Halt confirmation')
        await act(async () => confirm.click())
        expect(request).toHaveBeenCalledExactlyOnceWith('self', 'paused', 'immediate')
    })

    it('resumes a locally paused node', async () => {
        configuration.useNodeConfigurationStore.setState({
            availability: new Map([
                [
                    'self',
                    {
                        value: 'paused',
                        pausedBy: 'local',
                        current: true,
                        stale: false,
                        reading: false
                    }
                ]
            ])
        })
        await act(async () => root.render(<HaltControl targetId="self" />))
        await act(async () => button('Resume').click())
        expect(request).toHaveBeenCalledExactlyOnceWith('self', 'available', undefined)
    })

    it('disables the control and announces a pending immediate pause', async () => {
        configuration.useNodeConfigurationStore.setState({
            availability: new Map([
                [
                    'self',
                    {
                        value: 'available',
                        current: true,
                        stale: false,
                        reading: false,
                        pending: 'paused',
                        pendingPauseMode: 'immediate'
                    }
                ]
            ])
        })
        await act(async () => root.render(<HaltControl targetId="self" />))
        expect(button('Halt').disabled).toBe(true)
        expect(button('Halt').getAttribute('aria-busy')).toBe('true')
        await act(async () => button('Halt').click())
        expect(request).not.toHaveBeenCalled()
    })

    it('renders no control or watcher for a remote node', async () => {
        await act(async () => root.render(<HaltControl targetId="peer" />))
        expect(container.textContent).toBe('')
        expect(configuration.watchNodeAvailability).not.toHaveBeenCalled()
        expect(request).not.toHaveBeenCalled()
    })
})
