// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClusterNode } from '@/shared/types/cluster'
import type { IClusterApi } from '@/ui/api/pair-api'
import { NodeNameModal } from '@/ui/components/NodeName/NodeNameModal'
import { ServiceCallError } from '@/ui/api/service-transport'
import { useClusterInvitationsStore } from '@/ui/stores/cluster-invitations.store'
import { useConnectionStore } from '@/ui/stores/connection.store'

const setNodeName = vi.fn<IClusterApi['setNodeName']>()
const onClose = vi.fn<() => void>()

const peer: ClusterNode = {
    nodeUuid: 'peer-uuid',
    name: 'Studio',
    displayName: 'Studio',
    ipAddress: '192.0.2.2',
    port: 1,
    clusterId: 'home',
    state: 'member',
    joinedAt: 1,
    lastSeen: 1
}
const self: ClusterNode = {
    ...peer,
    nodeUuid: 'self-uuid',
    name: 'PAIR 1a2b3c4d',
    displayName: 'PAIR 1a2b3c4d'
}

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

let container: HTMLDivElement
let root: Root

function button(label: string): HTMLButtonElement | undefined {
    return Array.from(document.querySelectorAll('button')).find(
        value => value.textContent === label
    )
}

function nameInput(): HTMLInputElement {
    const input = document.querySelector<HTMLInputElement>('input[aria-label="Node name"]')
    if (!input) throw new Error('Missing node name input')
    return input
}

async function typeName(value: string): Promise<void> {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    if (!setter) throw new Error('Missing input value setter')
    const input = nameInput()
    await act(async () => {
        setter.call(input, value)
        input.dispatchEvent(new Event('input', { bubbles: true }))
    })
}

async function click(label: string): Promise<void> {
    const found = button(label)
    if (!found) throw new Error(`Missing ${label} button`)
    await act(async () => found.click())
}

async function renderModal(customName: string | null, shownName = 'PAIR 1a2b3c4d'): Promise<void> {
    await act(async () =>
        root.render(
            <NodeNameModal
                nodeUuid="self-uuid"
                shownName={shownName}
                customName={customName}
                open
                onClose={onClose}
            />
        )
    )
}

beforeEach(() => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
    setNodeName.mockReset()
    onClose.mockReset()
    Object.defineProperty(window, 'pairApi', {
        configurable: true,
        value: { cluster: { setNodeName } }
    })
    Object.defineProperty(window, 'windowApi', {
        configurable: true,
        value: { platform: 'Windows' }
    })
    useConnectionStore.setState({ selfId: 'self-uuid' })
    useClusterInvitationsStore.setState({ members: [self, peer], nameChange: null })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
})

afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    useClusterInvitationsStore.setState({ members: [], nameChange: null })
    vi.unstubAllGlobals()
})

describe('node name modal', () => {
    it('prefills the shown name when no custom name is set', async () => {
        await renderModal(null)
        expect(nameInput().value).toBe('PAIR 1a2b3c4d')
    })

    it('prefills the custom name when one is set', async () => {
        await renderModal('Desk', 'Desk')
        expect(nameInput().value).toBe('Desk')
    })

    it('saves the trimmed name and closes', async () => {
        setNodeName.mockResolvedValue({ nodeUuid: 'self-uuid', name: 'Desk', customName: 'Desk' })
        await renderModal(null)
        await typeName('  Desk  ')
        await click('Save')
        expect(setNodeName).toHaveBeenCalledExactlyOnceWith('Desk')
        expect(onClose).toHaveBeenCalledOnce()
    })

    it('disables Save for an empty name', async () => {
        await renderModal(null)
        await typeName('   ')
        expect(button('Save')?.disabled).toBe(true)
    })

    it('sends null from Use default name', async () => {
        setNodeName.mockResolvedValue({
            nodeUuid: 'self-uuid',
            name: 'PAIR 1a2b3c4d',
            customName: null
        })
        await renderModal('Desk', 'Desk')
        await click('Use default name')
        expect(setNodeName).toHaveBeenCalledExactlyOnceWith(null)
        expect(onClose).toHaveBeenCalledOnce()
    })

    it('hides Use default name without a custom name', async () => {
        await renderModal(null)
        expect(button('Use default name')).toBeUndefined()
    })

    it('warns about a duplicate shown name and still allows Save', async () => {
        await renderModal(null)
        await typeName('studio')
        expect(document.querySelector('[role="status"]')?.textContent).toBe(
            'Another node is already shown as "Studio". You can still save; PAIR adds a hardware label to tell them apart.'
        )
        expect(button('Save')?.disabled).toBe(false)
    })

    it('shows the service error and stays open', async () => {
        setNodeName.mockRejectedValue(
            new ServiceCallError('name must be at most 63 characters', -32602)
        )
        await renderModal(null)
        await typeName('x'.repeat(80))
        await click('Save')
        expect(document.querySelector('[role="alert"]')?.textContent).toBe(
            'name must be at most 63 characters'
        )
        expect(onClose).not.toHaveBeenCalled()
    })
})
